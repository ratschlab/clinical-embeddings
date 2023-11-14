import copy
import enum
import logging
import math
import os
import pickle
from functools import partial
from typing import Optional

import gin
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.nn.utils import weight_norm
from torch.utils.tensorboard import SummaryWriter


def remove_some_duplicates(card_list):
    prev = -1
    res = []
    for el in card_list:
        if el == 1 or el != prev:
            res.append(el)
        prev = el
    return res


@gin.configurable("masking")
def parrallel_recomb(q_t, kv_t, att_type="all", local_context=3, bin_size=None):
    """Return mask of attention matrix (ts_q, ts_kv)"""
    with torch.no_grad():
        q_t[q_t == -1.0] = float(
            "inf"
        )  # We want padded to attend to everyone to avoid any nan.
        kv_t[kv_t == -1.0] = float("inf")  # We want no one to attend the padded values

        if bin_size is not None:  # General case where we use unaligned timesteps.
            q_t = q_t / bin_size
            starts_q = q_t[:, 0:1].clone()  # Needed because of Memory allocation issue
            q_t -= starts_q
            kv_t = kv_t / bin_size
            starts_kv = kv_t[
                :, 0:1
            ].clone()  # Needed because of Memory allocation issue
            kv_t -= starts_kv

        bs, ts_q = q_t.size()
        _, ts_kv = kv_t.size()
        q_t_rep = q_t.view(bs, ts_q, 1).repeat(1, 1, ts_kv)
        kv_t_rep = kv_t.view(bs, 1, ts_kv).repeat(1, ts_q, 1)
        diff_mask = (q_t_rep - kv_t_rep).to(q_t_rep.device)
        if att_type == "all":
            return (diff_mask >= 0).float()
        if att_type == "local":
            return (
                (diff_mask >= 0) * (diff_mask <= local_context)
                + (diff_mask == float("inf"))
            ).float()
        if att_type == "strided":
            return (
                (diff_mask >= 0) * (torch.floor(diff_mask) % local_context == 0)
                + (diff_mask == float("inf"))
            ).float()


class _TokenInitialization(enum.Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"

    @classmethod
    def from_str(cls, initialization):
        try:
            return cls(initialization)
        except ValueError:
            valid_values = [x.value for x in _TokenInitialization]
            raise ValueError(f"initialization must be one of {valid_values}")

    def apply(self, x, d):
        d_sqrt_inv = 1 / math.sqrt(d)
        if self == _TokenInitialization.UNIFORM:
            # used in the paper "Revisiting Deep Learning Models for Tabular Data";
            # is equivalent to `nn.init.kaiming_uniform_(x, a=math.sqrt(5))` (which is
            # used by torch to initialize nn.Linear.weight, for example)
            nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif self == _TokenInitialization.NORMAL:
            nn.init.normal_(x, std=d_sqrt_inv)


@gin.configurable("CLSToken")
class CLSToken(nn.Module):
    """[CLS]-token for BERT-like inference.
    When used as a module, the [CLS]-token is appended **to the end** of each item in
    the batch.

    References:
        * [devlin2018bert] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" 2018
        * code from [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    def __init__(self, d_token, initialization):
        """
        Args:
            d_token: the size of token
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`. In
                the paper [gorishniy2021revisiting], the 'uniform' initialization was
                used.
        """
        super().__init__()
        initialization_ = _TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(torch.Tensor(d_token))
        initialization_.apply(self.weight, d_token)

    def expand(self, *leading_dimensions):
        """Expand (repeat) the underlying [CLS]-token to a tensor with the given leading dimensions.
        A possible use case is building a batch of [CLS]-tokens. See `CLSToken` for
        examples of usage.
        Note:
            Under the hood, the `torch.Tensor.expand` method is applied to the
            underlying :code:`weight` parameter, so gradients will be propagated as
            expected.
        Args:
            leading_dimensions: the additional new dimensions
        Returns:
            tensor of the shape :code:`(*leading_dimensions, len(self.weight))`
        """
        if not leading_dimensions:
            return self.weight
        new_dims = (1,) * (len(leading_dimensions) - 1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, x):
        """Append self **to the end** of each item in the batch (see `CLSToken`)."""
        return torch.cat([x, self.expand(len(x), 1)], dim=1)


@gin.configurable("Linear")
class Linear(nn.Module):
    """Linear Embedding Module."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.feature_tokens = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.feature_tokens(x)  # (bs, sq, hidden_dim)


@gin.configurable("MLP")
class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim=64,
        depth=1,
        activation="relu",
        do=0.0,
        ln=True,
    ):
        super().__init__()
        embedding_layers = []
        if activation == "relu":
            activation_fn = nn.ReLU
        else:
            raise Exception("Activation has to be relu")
        for k in range(depth):
            if k == 0:
                if depth == 1:
                    embedding_layers.append(nn.Linear(input_dim, hidden_dim))
                if depth > 1:
                    embedding_layers.append(nn.Linear(input_dim, latent_dim))
                    if ln:
                        embedding_layers.append(nn.LayerNorm(latent_dim))
                    embedding_layers.append(activation_fn())
                    embedding_layers.append(nn.Dropout(do))
            elif k == depth - 1:
                embedding_layers.append(nn.Linear(latent_dim, hidden_dim))

            else:
                embedding_layers.append(nn.Linear(latent_dim, latent_dim))
                if ln:
                    embedding_layers.append(nn.LayerNorm(latent_dim))
                embedding_layers.append(activation_fn())
                embedding_layers.append(nn.Dropout(do))
        self.embedding_layer = nn.Sequential(*embedding_layers)

    def forward(self, x):
        return self.embedding_layer(x)


class ResNetBlock(nn.Module):
    def __init__(self, emb_dim, latent_dim, do):
        super().__init__()

        self.batch_norm = nn.BatchNorm1d(emb_dim)
        self.linear_first = nn.Linear(emb_dim, latent_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(do)
        self.linear_second = nn.Linear(latent_dim, emb_dim)

    def forward(self, x):
        x_init = x
        x = self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.linear_first(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.linear_second(x)
        x = x + x_init
        return x


@gin.configurable("ResNet")
class ResNet(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, latent_dim=64, do=0.0, depth=1
    ):  # TODO: implement ln like MLP
        super().__init__()

        self.input_embedding = nn.Linear(input_dim, hidden_dim)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                ResNetBlock(emb_dim=hidden_dim, latent_dim=latent_dim, do=do)
            )

        self.tblocks = nn.Sequential(*tblocks)
        self.batch_norm_out = nn.BatchNorm1d(hidden_dim)
        # self.relu_out = nn.ReLU()

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.tblocks(x)
        x = self.batch_norm_out(x.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # (N, L, C) to (N, C, L) and back
        # x = self.relu_out(x)
        return x


class Time2Vec(nn.Module):
    def __init__(self, emb):
        super().__init__()

        self.zero = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, emb - 1)

    def forward(self, x):
        bs, n, emb = x.size()

        position = torch.arange(0, n, dtype=torch.float).unsqueeze(1)
        zero_pos = self.zero(position)
        period_pos = torch.sin(self.periodic(position))

        time_emb = torch.cat([zero_pos, period_pos], -1)

        return x + time_emb


class OrthogonalEmbeddings(nn.Module):
    """
    Creates an orthogonal basis for categorical embeddings
    A parameter free approach to encode categories; best
    paired with a bias term to represent the categorical itself.
    """

    def __init__(self, num_categories: int, token_dim: int) -> None:
        super().__init__()

        assert_msg = f"[{self.__class__.__name__}] require token dim {token_dim} >= num. cat. {num_categories}"
        assert token_dim >= num_categories, assert_msg

        random_mat = torch.randn(token_dim, token_dim)
        u_mat, _, vh_mat = torch.linalg.svd(random_mat)
        ortho_mat = u_mat @ vh_mat

        self.ortho_basis = ortho_mat[:num_categories, :]

    def forward(self, x: torch.Tensor):
        """
        Forward Method

        Parameter
        ---------
        x: torch.Tensor
            batch of one-hot encoded categorical values
            shape: [#batch, #classes]
        """
        return x @ self.ortho_basis.to(x.device)


@gin.configurable("FeatureTokenizer")
class FeatureTokenizer(nn.Module):

    """
    from Revisiting Deep Learning Models for Tabular Data

    x = (x_cat, x_num)
    x_num - (bs, sq, d_numerical)
    x_cat - (bs, sq, dim_cat), where dim_cat = sum(categories), already one-hot vectors!

    TEST:
    x_num = torch.rand(1,1,5)
    x_cat = torch.tensor([[[1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]]])
    categories = [4, 6, 13, 5]
    d_token = 3

    ft = FeatureTokenizer(5,categories,3)
    res = ft((x_num, x_cat))

    assert res.size()== (1, 1, 10, 3) - result has shape (bs, sq, number of x_num features+number of x_cat features+CLS, d_token)
                                        number of x_num features = 5, number of x_cat features: 4 features with range from categories

    """

    def __init__(
        self,
        d_inp,
        d_token,
        categories,
        bias=True,
        categorical_embedding: str = "linear",
    ):
        """
        Constructor for `FeatureTokenizer`

        Parameter
        ---------
        d_inp: int
            number of features to embed
        d_token: int
            embedding dimension
        categories: list[int]
            list of categorical variables and their cardinalities
        bias: bool
            whether to add and learn a bias term
        categorical_embedding: str
            form of embedding categoricals
            options: {linear, orthonormal}
        """
        super().__init__()

        self.categories = categories

        # self.activation = torch.nn.Sigmoid()

        d_numerical = d_inp - sum(self.categories)

        if not self.categories:
            d_bias = d_numerical
        else:
            d_bias = d_numerical + len(categories)

        self.weight_num = nn.Parameter(
            torch.Tensor(d_numerical + 1, d_token)
        )  # +1 for CLS token
        nn.init.kaiming_uniform_(self.weight_num, a=math.sqrt(5))

        if categorical_embedding == "linear":
            categorical_module = partial(nn.Linear, bias=False)
        elif categorical_embedding == "orthonormal":
            logging.info(
                f"[{self.__class__.__name__}] using 'orthonormal' categorical embeddings"
            )
            categorical_module = OrthogonalEmbeddings
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] categorical embedding: {categorical_embedding} not impl."
            )

        self.weight_cat = [
            categorical_module(cat_i, d_token) for cat_i in self.categories
        ]
        self.weight_cat = nn.ModuleList(self.weight_cat)

        self.bias = nn.Parameter(torch.Tensor(d_bias, d_token)) if bias else None
        if self.bias is not None:
            nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, x):

        x_cat = x[0]
        x_num = x[1]

        num_size = x_num.size(-1)
        cat_size = x_cat.size(-1)

        assert (num_size or cat_size) != 0

        if cat_size == 0:
            x_some = x_num
        else:
            x_some = x_cat

        bsXsq, _ = x_some.size()

        x_num = torch.cat(
            ([] if num_size == 0 else [x_num])
            + [torch.ones(bsXsq, 1, device=x_some.device)],
            dim=-1,
        )  # CLS token in the END!

        x = self.weight_num[None] * x_num[..., None]

        if self.categories:
            x_cat = torch.split(x_cat, self.categories, dim=-1)
            x_cat = [self.weight_cat[i](x_cat[i]) for i in range(len(x_cat))]
            x_cat = torch.stack(x_cat, -2)
            x = torch.cat([x_cat, x], -2)  # CLS token in the END!

        if self.bias is not None:
            bias = torch.cat(
                [
                    self.bias,
                    torch.zeros(1, self.bias.shape[1], device=x_some.device),
                ]
            )

            x = x + bias[None]
        return x


@gin.configurable("Bert_Head")
class Bert_Head(nn.Module):
    """BERT-like inference with CLS token."""

    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.normalization = nn.LayerNorm(input_dim)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        x = x[
            :, -1
        ]  # CLS token is the last one,second dim corresponds to the modalities
        x = self.normalization(x)
        x = self.relu(x)
        x = self.linear(x)
        return x


@gin.configurable("FeatureTokenizer_Transformer")
class FeatureTokenizer_Transformer(nn.Module):
    """FT-Transformer Embedding Module from Revisiting Deep Learning Models for Tabular Data."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        categories=None,
        token_dim=gin.REQUIRED,
        transformer=gin.REQUIRED,
    ):
        super().__init__()

        self.feature_tokens = FeatureTokenizer(
            input_dim, token_dim, categories=categories
        )
        self.transformer = transformer(token_dim)
        self.head = Bert_Head(token_dim, hidden_dim)

    def forward(self, x):
        bs, sq, dim_num = x[0].size()
        bs, sq, dim_cat = x[1].size()

        x = (
            x[0].view(bs * sq, dim_num),
            x[1].view(bs * sq, dim_cat),
        )  # dim = features+1 for CLS token
        x = self.feature_tokens(x)
        x = self.transformer(x)  # (bs*sq, feature+cls token, feature_dim)
        x = self.head(x)  # (bs*sq, out_dim)
        x = x.view(bs, sq, x.size(-1))
        return x


class WeightedAverage(torch.nn.Module):
    def __init__(self, mod_count):
        super(WeightedAverage, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(mod_count))

    def forward(self, x):
        x = torch.stack(x)
        weights = F.softmax(self.weights, dim=0)
        weight_layer = torch.sum((x * weights.view(-1, 1, 1, 1)), dim=0)
        return weight_layer


@gin.configurable("Splitted_Embedding")
class Splitted_Embedding(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        embedding_model=gin.REQUIRED,
        merge="concat",
        agg_feature_mode=None,
        path_to_cat_dict=None,
        reindex_modalities=None,
        initialize_emb_blocks: bool = True,
    ):
        super().__init__()

        self.emb_blocks = []
        self.reindex_modalities = reindex_modalities
        if self.reindex_modalities is not None:
            self.split_size = [len(el) for el in self.reindex_modalities]
            # assert sum(self.split_size) == input_dim
            self.reindex_modalities = [
                item for el in self.reindex_modalities for item in el
            ]

        else:
            self.split_size = input_dim

        # to know what features are categorical and numerical in each mod split
        self.path_to_cat_dict = (
            path_to_cat_dict  # dict, keys: cat col number, value:cardinality
        )
        if self.path_to_cat_dict is not None:

            with open(path_to_cat_dict, "rb") as file:
                cat_dict = pickle.load(file)

            (
                self.reindex_for_split,
                self.features_split_size,
                self.cardinality_list,
            ) = self.compute_cardinality_list(
                self.reindex_modalities, self.split_size, cat_dict
            )

        self.hidden_dims = hidden_dims
        if isinstance(hidden_dims, int):
            # Compute hidden_dim of embedding for each modality
            if merge == "concat":
                hidden_dim = hidden_dims // len(self.split_size)
                self.hidden_dims = [hidden_dim for _ in range(len(self.split_size))]

                # Add dimension to total required size
                for i in range(hidden_dims - sum(self.hidden_dims)):
                    self.hidden_dims[i] += 1

            elif merge == "mean" or merge == "attention_cls" or "weighted_avg":
                self.hidden_dims = [hidden_dims for _ in range(len(self.split_size))]
            else:
                raise Exception("not implemented yet")

        assert len(self.hidden_dims) == len(self.split_size)

        if initialize_emb_blocks:
            for i in range(len(self.split_size)):
                input_dim = self.split_size[i]
                hidden_dim = self.hidden_dims[i]
                if (
                    self.path_to_cat_dict
                ):  # only need it when we use FeatureTokenizerTransformer as embedding model
                    self.emb_blocks.append(
                        embedding_model(
                            input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            categories=self.cardinality_list[i],
                        )
                    )
                else:
                    self.emb_blocks.append(
                        embedding_model(input_dim=input_dim, hidden_dim=hidden_dim)
                    )
            self.emb_blocks = nn.ModuleList(self.emb_blocks)
        else:
            logging.warning(
                f"[{self.__class__.__name__}] not initializing `self.emb_blocks`"
            )
            self.emb_blocks = None

        self.merge = merge
        logging.info(f"[{self.__class__.__name__}] merge: {self.merge}")
        if self.merge == "weighted_avg":
            self.wa_aggregation = WeightedAverage(len(self.cardinality_list))

        if self.merge == "attention_cls":
            hidden_dim = self.hidden_dims[0]
            self.cls_token = CLSToken(hidden_dim, "uniform")
            self.head = Bert_Head(hidden_dim, hidden_dim)
            if agg_feature_mode is None:
                raise Exception(
                    "Specify agg_feature_mode = @aggregation/TransformerBlock in config file."
                )
            self.attention_aggregation = agg_feature_mode(hidden_dim)

    @staticmethod
    def compute_cardinality_list(reindex_modalities, split_size, cat_dict):
        """
        Computes the cardinality list with the size of each categorical variable
        for each group/split of modalities

        Parameter
        ---------
        reindex_modalities: list[list[int]]
            reindex list to make the groups of modalities
        split_size: list[int]
            the size of each group/split
        cat_dict: dict[]
            dictionary which stores the type for each variable
        """

        reindex_type_cols = [
            ("cat", cat_dict[col_ids]) if col_ids in cat_dict else ("num", None)
            for col_ids in reindex_modalities
        ]  # tuples (feature_type (cat or num), cardinality if cat, None if num)
        it = iter(reindex_type_cols)
        splitted_type_cols = [
            [next(it) for _ in range(size)] for size in split_size
        ]  # split by split sizes
        splitted_type_cols = [
            [(i, el[0], el[1]) for i, el in enumerate(elem)]
            for elem in splitted_type_cols
        ]  # list of tuples (ids in split_size, feature_type, cardinality or None)
        sorted_type_cols = [
            sorted(data, key=lambda tup: tup[1]) for data in splitted_type_cols
        ]  # sorted by the type of feature

        reindex_for_split = [
            [tup[0] for tup in elem] for elem in sorted_type_cols
        ]  # reindex for each mdality
        features_split_size = [
            [
                sum(1 for i in split if i[1] == "cat"),
                sum(1 for i in split if i[1] == "num"),
            ]
            for split in sorted_type_cols
        ]  # split sizes for each modality

        cardinality_list = [
            [tup[2] for tup in elem if tup[1] == "cat"] for elem in sorted_type_cols
        ]  # list of cardinalities for each modality

        assert [len(elem) for elem in cardinality_list] == [
            elem[0] for elem in features_split_size
        ]
        cardinality_list = [
            remove_some_duplicates(elem) for elem in cardinality_list
        ]  # ordered!!

        return reindex_for_split, features_split_size, cardinality_list

    def aggregate(self, x):
        # Merge embedding for each modality
        if self.merge == "concat":
            # Concatenate embedding from each modality
            # Output has dimension (bs x sq x hidden_dim)
            # where hidden_dim = sum(hidden_dim_of_modalities)
            return torch.cat(x, dim=-1)  # TODO: fix this bug

        elif self.merge == "mean":
            # Average embeddings of modalities
            # Output has dimension (bs x sq x hidden_dim)
            # where hidden_dim = hidden_dim_of_modalities

            return sum(x) / len(x)
            # return torch.mean(torch.stack(x, dim=0), dim=0)

        elif self.merge == "sum":
            return sum(x)

        elif self.merge == "attention_cls":
            bs, sq, _ = x[0].size()
            features = [
                f.reshape(-1, f.size(-1)) for f in x
            ]  # (bs, sq, dim) --> (bs*sq, dim)
            cat_features = torch.stack(features, 1)  # (bs*sq, splits, dim)
            cls_features = self.cls_token(
                cat_features
            )  # add cls token at the end --> (bs*sq, splits+1, dim)
            cls_features_att = self.attention_aggregation(cls_features)
            cls_token = self.head(cls_features_att)  # (bs*sq, final_dim)
            return cls_token.view(bs, sq, cls_token.size(-1))

        elif self.merge == "weighted_avg":
            return self.wa_aggregation(x)

        elif self.merge == "none":
            return x[0]  # get single split

        else:
            raise Exception("not implemented yet")

    def apply_embedding_per_modality(self, x):
        if self.reindex_modalities is not None:
            # Re-index data in order of modalities
            x = x[..., self.reindex_modalities]
        # Split dataset into distinct modalities
        x_split = torch.split(x, self.split_size, dim=-1)

        if self.path_to_cat_dict is not None:
            reindexed_tensors = [
                mod_i[..., self.reindex_for_split[i]] for i, mod_i in enumerate(x_split)
            ]
            # list of tuples, first element in the tuple - categorical feature tensor, second - numerical
            x_split = [
                torch.split(mod_i, self.features_split_size[i], dim=-1)
                for i, mod_i in enumerate(reindexed_tensors)
            ]

        outputs = []
        for i, x_k in enumerate(x_split):
            # Apply embedding model to each modality
            outputs.append(self.emb_blocks[i](x_k))

        return outputs

    def forward(self, x):
        outputs = self.apply_embedding_per_modality(x)
        return self.aggregate(outputs)


class PositionalEncoding(nn.Module):
    "Positional Encoding, mostly from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html"

    def __init__(self, emb, max_len=3000):
        super().__init__()
        emb_tensor = emb if emb % 2 == 0 else emb + 1
        pe = torch.zeros(max_len, emb_tensor)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_tensor, 2).float() * (-math.log(10000.0) / emb_tensor)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        bs, n, emb = x.size()
        return x + self.pe[:, :n, :emb]


class SelfAttentionSimple(nn.Module):
    def __init__(self, emb, mask=True):
        super().__init__()
        self.emb = emb
        self.mask = mask

    def forward(self, x):
        emb = self.emb
        queries = x[1].permute(1, 0, 2)  # (bs, 1, emb)
        keys = x[0].transpose(1, 2)
        values = x[0]

        queries = queries / (emb ** (1 / 2))
        keys = keys / (emb ** (1 / 2))
        dot = torch.bmm(queries, keys)
        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values)
        return out.squeeze()


class SelfAttention(nn.Module):
    """Multi Head Attention block from Attention is All You Need.
    Input has shape (batch_size, n_timestemps, emb).

    ----------
    emb:
        Dimension of the input vector.
    hidden:
        Dimension of query, key, value matrixes.
    heads:
        Number of heads.

    mask:
        Mask the future timestemps
    """

    def __init__(
        self,
        emb,
        hidden,
        heads=8,
        mask=True,
        att_type="all",
        local_context=None,
        mask_aggregation="union",
        dropout_att=0.0,
    ):
        """Initialize the Multi Head Block."""
        super().__init__()

        self.emb = emb
        self.heads = heads
        self.hidden = hidden
        self.mask = mask
        self.drop_att = nn.Dropout(dropout_att)

        # Sparse transformer specific params
        self.att_type = att_type
        self.local_context = local_context
        self.mask_aggregation = mask_aggregation

        # Query, keys and value matrices
        self.w_keys = nn.Linear(emb, hidden * heads, bias=False)
        self.w_queries = nn.Linear(emb, hidden * heads, bias=False)
        self.w_values = nn.Linear(emb, hidden * heads, bias=False)

        # Output linear function
        self.unifyheads = nn.Linear(heads * hidden, emb)

    def forward(self, x):
        """
        x:
            Input data tensor with shape (batch_size, n_timestemps, emb)
        hidden:
            Hidden dim (dimension of query, key, value matrixes)

        Returns
            Self attention tensor with shape (batch_size, n_timestemps, emb)
        """
        # bs - batch_size, n - vectors number, emb - embedding dimensionality
        bs, n, emb = x.size()
        h = self.heads
        hidden = self.hidden

        keys = self.w_keys(x).view(bs, n, h, hidden)
        queries = self.w_queries(x).view(bs, n, h, hidden)
        values = self.w_values(x).view(bs, n, h, hidden)

        # fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(bs * h, n, hidden)
        queries = queries.transpose(1, 2).contiguous().view(bs * h, n, hidden)
        values = values.transpose(1, 2).contiguous().view(bs * h, n, hidden)

        # dive on the square oot of dimensionality
        queries = queries / (hidden ** (1 / 2))
        keys = keys / (hidden ** (1 / 2))

        # dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        if self.mask:  # We deal with different masking and recombination types here
            if isinstance(self.att_type, list):  # Local and sparse attention
                if self.mask_aggregation == "union":
                    mask_tensor = 0
                    for att_type in self.att_type:
                        mask_tensor += parrallel_recomb(
                            torch.arange(
                                1, n + 1, dtype=torch.float, device=dot.device
                            ).reshape(1, -1),
                            torch.arange(
                                1, n + 1, dtype=torch.float, device=dot.device
                            ).reshape(1, -1),
                            att_type,
                            self.local_context,
                        )[0]
                    mask_tensor = torch.clamp(mask_tensor, 0, 1)
                    dot = torch.where(
                        mask_tensor.bool(),
                        dot,
                        torch.tensor(float("-inf")).to(dot.device),
                    ).view(bs * h, n, n)

                elif self.mask_aggregation == "split":

                    dot_list = list(
                        torch.split(dot, dot.shape[0] // len(self.att_type), dim=0)
                    )
                    for i, att_type in enumerate(self.att_type):
                        mask_tensor = parrallel_recomb(
                            torch.arange(
                                1, n + 1, dtype=torch.float, device=dot.device
                            ).reshape(1, -1),
                            torch.arange(
                                1, n + 1, dtype=torch.float, device=dot.device
                            ).reshape(1, -1),
                            att_type,
                            self.local_context,
                        )[0]

                        dot_list[i] = torch.where(
                            mask_tensor.bool(),
                            dot_list[i],
                            torch.tensor(float("-inf")).to(dot.device),
                        ).view(*dot_list[i].shape)
                    dot = torch.cat(dot_list, dim=0)
            else:  # Full causal masking
                mask_tensor = parrallel_recomb(
                    torch.arange(
                        1, n + 1, dtype=torch.float, device=dot.device
                    ).reshape(1, -1),
                    torch.arange(
                        1, n + 1, dtype=torch.float, device=dot.device
                    ).reshape(1, -1),
                    self.att_type,
                    self.local_context,
                )[0]
                dot = torch.where(
                    mask_tensor.bool(), dot, torch.tensor(float("-inf")).to(dot.device)
                ).view(bs * h, n, n)

        # dot now has row-wise self-attention probabilities
        dot = F.softmax(dot, dim=2)

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(bs, h, n, hidden)

        # apply the dropout
        out = self.drop_att(out)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(bs, n, h * hidden)
        return self.unifyheads(out)


class SparseBlock(nn.Module):
    def __init__(
        self,
        emb,
        hidden,
        heads,
        ff_hidden_mult,
        dropout=0.0,
        mask=True,
        mask_aggregation="union",
        local_context=3,
        dropout_att=0.0,
    ):
        super().__init__()

        self.attention = SelfAttention(
            emb,
            hidden,
            heads=heads,
            mask=mask,
            mask_aggregation=mask_aggregation,
            local_context=local_context,
            att_type=["strided", "local"],
            dropout_att=dropout_att,
        )
        self.mask = mask
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb),
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention.forward(x)
        x = self.norm1(attended + x)
        x = self.drop(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)

        x = self.drop(x)

        return x


class LocalBlock(nn.Module):
    def __init__(
        self,
        emb,
        hidden,
        heads,
        ff_hidden_mult,
        dropout=0.0,
        mask=True,
        local_context=3,
        dropout_att=0.0,
    ):
        super().__init__()

        self.attention = SelfAttention(
            emb,
            hidden,
            heads=heads,
            mask=mask,
            mask_aggregation=None,
            local_context=local_context,
            att_type="local",
            dropout_att=dropout_att,
        )
        self.mask = mask
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb),
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention.forward(x)
        x = self.norm1(attended + x)
        x = self.drop(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)

        x = self.drop(x)

        return x


@gin.configurable("TransformerBlock")
class TransformerBlock(nn.Module):
    def __init__(
        self,
        emb,
        hidden,
        heads,
        ff_hidden_mult,
        dropout=0.0,
        mask=True,
        dropout_att=0.0,
    ):
        super().__init__()

        self.attention = SelfAttention(
            emb, hidden, heads=heads, mask=mask, dropout_att=dropout_att
        )
        self.mask = mask
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb),
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention.forward(x)
        x = self.norm1(attended + x)
        x = self.drop(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)

        x = self.drop(x)

        return x


@gin.configurable("StackedTransformerBlocks")
class StackedTransformerBlocks(nn.Module):
    def __init__(
        self,
        emb,
        hidden,
        heads,
        ff_hidden_mult,
        depth: int = 1,
        dropout: float = 0.0,
        mask: bool = True,
        dropout_att=0.0,
    ):
        super().__init__()

        layers = [
            TransformerBlock(
                emb=emb,
                hidden=hidden,
                heads=heads,
                ff_hidden_mult=ff_hidden_mult,
                dropout=dropout,
                mask=mask,
                dropout_att=dropout_att,
            )
            for _ in range(depth)
        ]
        self.transformer_blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.transformer_blocks(x)


# From TCN original paper https://github.com/locuslab/TCN
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            dim=None,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            dim=None,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
