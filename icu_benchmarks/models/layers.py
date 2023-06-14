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


@gin.configurable("MovingAverageClustering")
class MovingAverageClustering(nn.Module):
    """
    Initialize the model
    model = MovingAverageClustering(num_clusters, num_features)

    Use to upd assignments
    assignments = model(data, model.prev_assignments)
    model.prev_assignments = assignments
    """

    def __init__(
        self,
        num_clusters: int,
        num_features: int,
        decay: float = 0.99,
        initialization_method: str = "random",
        prior_assignments: list[list[int]] = None,
        prior_gmm_scale_factor: float = 2.0,
        plot_dim_red: str = "pca",
    ):
        super(MovingAverageClustering, self).__init__()

        self.num_clusters = num_clusters
        self.num_features = num_features
        self.decay = decay
        self.initialization_method = initialization_method
        self.plot_dim_red = plot_dim_red

        self.centroids = nn.Parameter(
            torch.zeros(num_clusters, num_features), requires_grad=False
        )
        self.prev_assignments = None

        if prior_assignments is not None:
            assert "prior" in initialization_method
            assert len(prior_assignments) == num_clusters
            self.prior_assignments = prior_assignments

        if initialization_method == "prior-gmm":
            self.prior_gmm_scale_factor = prior_gmm_scale_factor
            logging.info(
                f"[{self.__class__.__name__}] prior_gmm_scale_factor: {self.prior_gmm_scale_factor}"
            )

    def init(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Initialize the centroids based on a passed data sample

        Parameter
        ---------
        x: torch.Tensor
            feature matrix to be clustered
        """
        logging.info(
            f"[{self.__class__.__name__}] Initializing centroids with {self.initialization_method} method"
        )
        if self.initialization_method == "random":
            # assert False, "Not yet implemented: random clustering init"
            self.centroids.data = x[torch.randint(x.shape[0], (self.num_clusters,)), :]
            return None

        elif self.initialization_method == "kmeans++":
            self.kmeans_plus_plus_init(x)
            return None

        elif self.initialization_method == "prior-random":
            self.centroids.data = torch.stack(
                [
                    torch.mean(x[torch.tensor(self.prior_assignments[i]), :], dim=0)
                    for i in range(self.num_clusters)
                ]
            )
            return None

        elif self.initialization_method == "prior-gmm":

            scale_factor = self.prior_gmm_scale_factor
            mv_gaussian = torch.distributions.multivariate_normal.MultivariateNormal(
                torch.zeros(self.num_features),
                scale_factor * torch.eye(self.num_features),
            )
            mixture_centroids = mv_gaussian.sample((self.num_clusters,))
            self.centroids.data = mixture_centroids

            # recreate x matrix based on priors
            new_x = torch.empty(x.shape)
            for i, cluster in enumerate(self.prior_assignments):
                cluster_mv_gaussian = (
                    torch.distributions.multivariate_normal.MultivariateNormal(
                        mixture_centroids[i, :], torch.eye(self.num_features)
                    )
                )
                for j in cluster:
                    new_x[j, :] = cluster_mv_gaussian.sample()

            return new_x

        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Unsupported init. method: {self.initialization_method}"
            )

    def kmeans_plus_plus_init(self, x):
        self.centroids.data[0] = x[torch.randint(x.shape[0], (1,))].squeeze()

        for i in range(1, self.num_clusters):
            distances = torch.cdist(x, self.centroids[:i])
            min_distances = torch.min(distances, dim=1).values
            min_distances_squared = min_distances**2
            probabilities = min_distances_squared / torch.sum(min_distances_squared)
            new_centroid = x[torch.multinomial(probabilities, 1)].squeeze()
            self.centroids.data[i] = new_centroid

    def get_assignments(self, x: torch.Tensor):

        distances = torch.cdist(x, self.centroids)
        return torch.argmin(distances, dim=1)

    def forward(self, x: torch.Tensor, prev_assignments: torch.Tensor = None):

        prev_assignments = prev_assignments.to(self.centroids.device)

        distances = torch.cdist(x, self.centroids)
        new_assignments = torch.argmin(distances, dim=1)

        updated_centroids = False
        if prev_assignments is not None and torch.all(
            new_assignments == prev_assignments
        ):
            self.update_centroids(x, new_assignments)
            distances = torch.cdist(x, self.centroids)
            new_assignments = torch.argmin(distances, dim=1)
            updated_centroids = True

        return new_assignments, updated_centroids

    def update_centroids(self, x, assignments):
        one_hot_assignments = torch.zeros(
            x.shape[0], self.num_clusters, device=x.device
        ).scatter_(1, assignments.unsqueeze(1), 1)
        cluster_data_sum = one_hot_assignments.t().mm(x)
        cluster_counts = one_hot_assignments.sum(dim=0).view(-1, 1)
        new_centroids = cluster_data_sum / cluster_counts
        self.centroids.data = (
            1 - self.decay
        ) * self.centroids.data + self.decay * new_centroids

    def compute_inertia(self, x):
        distances = torch.cdist(x, self.centroids)
        cluster_assignments = torch.argmin(distances, dim=1)
        inertia = torch.sum(distances[torch.arange(x.size(0)), cluster_assignments])

        return inertia

    def functional_ratio(self, x):
        distances = torch.cdist(x, self.centroids)
        cluster_assignments = torch.argmin(distances, dim=1)

        # Intracluster distances
        one_hot_assignments = torch.zeros(
            x.shape[0], self.num_clusters, device=x.device
        ).scatter_(1, cluster_assignments.unsqueeze(1), 1)
        squared_distances = distances**2
        weighted_squared_distances = one_hot_assignments * squared_distances
        intracluster_distances = weighted_squared_distances.sum(
            dim=0
        ) / one_hot_assignments.sum(dim=0)
        total_intracluster_distance = intracluster_distances.sum()

        # Intercluster distances
        intercluster_distances = torch.cdist(self.centroids, self.centroids)
        num_unique_pairs = self.num_clusters * (self.num_clusters - 1) / 2
        upper_triangular_indices = torch.triu_indices(
            row=self.num_clusters, col=self.num_clusters, offset=1, device=x.device
        )
        sum_upper_triangular_intercluster = intercluster_distances[
            upper_triangular_indices[0], upper_triangular_indices[1]
        ].sum()
        avg_intercluster_distance = sum_upper_triangular_intercluster / num_unique_pairs

        return total_intracluster_distance / avg_intercluster_distance

    def regularization(self, x, cl_assignments):

        num_clusters = cl_assignments.size(1)
        num_data = x.size(0)

        centroids = cl_assignments.t() @ x / cl_assignments.sum(dim=0).view(-1, 1)
        distances = torch.cdist(x, centroids)

        # Intracluster distances
        intracluster_distances = (distances * cl_assignments).sum()

        # Intercluster distances
        num_clusters = centroids.shape[0]
        inter_cluster_distances_matrix = torch.cdist(centroids, centroids)
        num_unique_pairs = num_clusters * (num_clusters - 1) / 2
        upper_triangular_indices = torch.triu_indices(
            row=num_clusters, col=num_clusters, offset=1
        )
        sum_upper_triangular_intercluster = inter_cluster_distances_matrix[
            upper_triangular_indices[0], upper_triangular_indices[1]
        ].sum()
        avg_intercluster_distance = sum_upper_triangular_intercluster / num_unique_pairs

        return intracluster_distances / avg_intercluster_distance

    def soft_regularization(self, x):

        num_clusters = self.centroids.size(1)
        num_data = x.size(0)

        cl_assignments = x @ self.centroids.t()
        cl_assignments = torch.softmax(cl_assignments, dim=1)

        centroids = cl_assignments.t() @ x / cl_assignments.sum(dim=0).view(-1, 1)
        distances = torch.cdist(x, centroids)

        # Intracluster distances
        intracluster_distances = (distances * cl_assignments).sum()

        # Intercluster distances
        num_clusters = centroids.shape[0]
        inter_cluster_distances_matrix = torch.cdist(centroids, centroids)
        num_unique_pairs = num_clusters * (num_clusters - 1) / 2
        upper_triangular_indices = torch.triu_indices(
            row=num_clusters, col=num_clusters, offset=1
        )
        sum_upper_triangular_intercluster = inter_cluster_distances_matrix[
            upper_triangular_indices[0], upper_triangular_indices[1]
        ].sum()
        avg_intercluster_distance = sum_upper_triangular_intercluster / num_unique_pairs

        return intracluster_distances / avg_intercluster_distance

    def entropy_regularization(self, cl_assignments):
        """
        Compute the entropy regularization term for the soft clustering assignments (cl_assignments).
        """
        distribution = dist.Categorical(probs=cl_assignments)
        entropy = distribution.entropy()
        regularization = entropy.mean()

        return regularization

    def plot_clusters(self, x: torch.Tensor):

        centroids = self.centroids.data.detach().cpu().numpy()
        num_centroids = len(centroids)
        features = x.detach().cpu().numpy()
        data = np.concatenate((centroids, features), axis=0)

        if self.plot_dim_red == "pca":
            pca = PCA(n_components=2)
        elif self.plot_dim_red == "tsne":
            pca = TSNE(n_components=2)
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Unsupported dim. reduction method: {self.plot_dim_red}"
            )

        pca_data = pca.fit_transform(data)
        centroids_pca = pca_data[:num_centroids]
        features_pca = pca_data[num_centroids:]

        centroids_df = pd.DataFrame(data=centroids_pca, columns=["pc1", "pc2"])
        centroids_df["type"] = "centroid"

        features_df = pd.DataFrame(data=features_pca, columns=["pc1", "pc2"])
        features_df["type"] = "feature"

        data_df = pd.concat([centroids_df, features_df], axis=0)

        plot = sns.scatterplot(
            data=data_df,
            x="pc1",
            y="pc2",
            hue="type",
            sizes={"centroid": 20, "feature": 10},
            alpha=0.75,
            palette="muted",
        )
        return plot.get_figure()


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


@gin.configurable("ClusteredSplittedEmbedding")
class ClusteredSplittedEmbedding(Splitted_Embedding):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        token_dim,
        merge="concat",
        agg_feature_mode=None,
        path_to_cat_dict=None,
        reindex_modalities=None,
        transformer=gin.REQUIRED,
        transformer_shared: bool = True,
        cluster_splitting: bool = False,
        num_clusters_k: int = 1,
        clusters_init: str = "random",
        clustering_approach: str = "bias",
    ):

        self.source_reindex_modalities = copy.deepcopy(reindex_modalities)

        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dim,
            embedding_model=None,
            merge=merge,
            agg_feature_mode=agg_feature_mode,
            path_to_cat_dict=path_to_cat_dict,
            reindex_modalities=reindex_modalities,
            initialize_emb_blocks=False,
        )

        # Check feasability of merge operation
        assert merge in {"attention_cls", "mean", "sum"}

        # Compute Cardinality List for "no_split" (torch.arange(231))
        assert path_to_cat_dict is not None
        with open(path_to_cat_dict, "rb") as file:
            cat_dict = pickle.load(file)

        no_split_indexing = [i for i in range(input_dim)]
        no_split_size = [len(no_split_indexing)]

        (
            self.reindex_for_ftt,
            self.features_sizes_ftt,
            cardinality_list,
        ) = self.compute_cardinality_list(no_split_indexing, no_split_size, cat_dict)
        no_split_cardinality_list = cardinality_list[0]

        # initialize shared feature tokenizer
        assert (
            path_to_cat_dict is not None
        ), f"Please pass `path_to_cat_dict` for {self.__class__.__name__}"
        self.feature_tokenizer = FeatureTokenizer(
            input_dim, token_dim, categories=no_split_cardinality_list
        )

        # initialize shared embedding block: transformer and head
        # see: `FeatureTokenizer_Transformer`
        self.feature_cls_token = CLSToken(token_dim, "uniform")
        self.feature_head = Bert_Head(token_dim, hidden_dim)

        # initialize clustering module
        assert clusters_init in {"random", "kmeans++", "prior-random", "prior-gmm"}
        if "prior" in clusters_init:
            self.clustering_split = self.source_split_to_clustering_split(
                self.source_reindex_modalities,
                self.reindex_for_ftt[0],
                cat_dict,
                num_expanded_features=input_dim,
            )
            num_clusters_k = len(self.clustering_split)
            logging.warning(
                f"[{self.__class__.__name__}] prior split, reset number of clusters to {num_clusters_k}"
            )
        else:
            prior_split = self.source_split_to_clustering_split(
                self.source_reindex_modalities,
                self.reindex_for_ftt[0],
                cat_dict,
                num_expanded_features=input_dim,
            )
            self.clustering_split, _ = self.get_random_init_cluster(
                num_clusters_k, sum(map(len, prior_split))
            )

        # initialize shared embedding block: transformer and head
        # see: `FeatureTokenizer_Transformer`
        self.transformer_shared = transformer_shared
        if self.transformer_shared:
            logging.info(f"[{self.__class__.__name__}] initialized shared Transfomer")
            self.feature_transformer = transformer(token_dim)
            self.apply_feature_transformer = lambda _, x: self.feature_transformer(x)
        else:
            logging.info(
                f"[{self.__class__.__name__}] initialized per-cluster Transfomer"
            )
            self.feature_transformer = nn.ModuleList(
                [transformer(token_dim) for _ in range(num_clusters_k)]
            )
            self.apply_feature_transformer = lambda i, x: self.feature_transformer[i](x)

        self.clustering_split_reindex = [
            item for el in self.clustering_split for item in el
        ]
        self.clustering_split_sizes = [len(el) for el in self.clustering_split]
        logging.info(
            f"[{self.__class__.__name__}] initial clustering split sizes: {self.clustering_split_sizes} -> {sum(self.clustering_split_sizes)}"
        )

        self.cluster_splitting = cluster_splitting
        if self.cluster_splitting:
            self.clustering_assignments = torch.zeros(
                (sum(self.clustering_split_sizes),), dtype=torch.int32
            )
            self.num_clusters_k = num_clusters_k

            cluster_dimension = token_dim
            if clustering_approach != "bias":
                cluster_dimension = 2 * token_dim
            self.clustering = MovingAverageClustering(
                num_clusters_k,
                cluster_dimension,
                initialization_method=clusters_init,
                prior_assignments=self.clustering_split
                if "prior" in clusters_init
                else None,
            )

            self.clustering_approach = clustering_approach
            feature_matrix = self.extract_embeddings_for_clustering(
                self.feature_tokenizer
            )
            logging.info(
                f"[{self.__class__.__name__}] feature matrix dim: {feature_matrix.shape} with {self.clustering_approach}"
            )
            new_feature_matrix = self.clustering.init(feature_matrix)

            if clusters_init == "prior-gmm":
                assert new_feature_matrix is not None
                assert self.clustering_approach == "bias"
                self.feature_tokenizer.bias.data = new_feature_matrix
                self.clustering_assignments = self.clustering.get_assignments(
                    new_feature_matrix
                )
                logging.info(
                    f"[{self.__class__.__name__}] initialized bias weights with GMM for clustering"
                )
            else:
                assert new_feature_matrix is None
                self.clustering_assignments = self.clustering.get_assignments(
                    feature_matrix
                )

    def get_random_init_cluster(self, num_clusters_k: int, num_features: int):
        """
        Compute a random cluster assignment

        Parameter
        ---------
        num_clusters_k: int
            Number of clusters to assign
        num_features: int
            Number of features to assign
        """
        mean_cluster_size = num_features // num_clusters_k
        group_means = [mean_cluster_size for _ in range(num_clusters_k)]
        group_sample = np.random.dirichlet(group_means)
        group_sample = (group_sample * num_features).astype(np.int32)

        diff = num_features - sum(group_sample)
        group_sample[-1] += diff

        ids = np.arange(num_features)
        group_split = np.array_split(ids, np.cumsum(group_sample))[:-1]

        assert len(group_split) == num_clusters_k
        assert sum(group_sample) == num_features

        return group_split, group_sample

    def extract_embeddings_for_clustering(
        self, feature_tokenizer: FeatureTokenizer
    ) -> torch.Tensor:

        if self.clustering_approach == "bias":
            return feature_tokenizer.bias

        elif self.clustering_approach == "bias_ext_catzero":

            bias = feature_tokenizer.bias
            numerical_base = feature_tokenizer.weight_num[:-1]

            num_cats = len(feature_tokenizer.weight_cat)
            cats_zeros = torch.zeros(
                (num_cats, numerical_base.shape[-1]), device=numerical_base.device
            )

            joined_ext = torch.cat((numerical_base, cats_zeros), dim=0)
            return torch.cat((bias, joined_ext), dim=-1)

        elif self.clustering_approach == "bias_avg_linear":

            bias = feature_tokenizer.bias
            numerical_base = feature_tokenizer.weight_num[:-1]  # ignore CLS

            cat_base = [
                torch.mean(linear.weight, dim=1)
                for linear in feature_tokenizer.weight_cat
            ]
            cat_base = torch.stack(cat_base, dim=0)

            joined_base = torch.cat((numerical_base, cat_base), dim=0)

            return torch.cat((bias, joined_base), dim=-1)

        elif self.clustering_approach == "bias_sum_linear":

            bias = feature_tokenizer.bias
            numerical_base = feature_tokenizer.weight_num[:-1]  # ignore CLS

            cat_base = [
                torch.sum(linear.weight, dim=1)
                for linear in feature_tokenizer.weight_cat
            ]
            cat_base = torch.stack(cat_base, dim=0)

            joined_base = torch.cat((numerical_base, cat_base), dim=0)

            return torch.cat((bias, joined_base), dim=-1)

        else:
            msg = f"[{self.__class__.__name__}] clustering approach {self.clustering_approach} not supported"
            raise ValueError(msg)

    def update_clustering_split(self, new_cluster_assignments: torch.Tensor):
        """
        Computes new cluster assignments index lists based on the
        assignment tensor computed by the clustering

        Paramater
        ---------
        new_cluster_assignments: torch.Tensor
            shape: (#features)
        """

        updated_reindex_modalities = [[] for _ in range(self.num_clusters_k)]
        for idx, cluster in enumerate(new_cluster_assignments):
            updated_reindex_modalities[int(cluster)].append(idx)

        self.clustering_split = updated_reindex_modalities
        self.clustering_split_reindex = [
            item for el in self.clustering_split for item in el
        ]
        self.clustering_split_sizes = [len(el) for el in self.clustering_split]

        # logging.info(f"[{self.__class__.__name__}] updated split based on clustering")
        # logging.info(f"[{self.__class__.__name__}] cluster sizes: {self.clustering_split_sizes}")

    def source_split_to_clustering_split(
        self,
        source_reindex_modalities,
        reindex_for_ftt,
        cat_dict,
        num_expanded_features: int = 231,
    ):
        """
        The source prior split is expressed w.r.t. raw data
        the clustering is performed on the reindexed data after the
        feature tokenizer. This function maps a prior split to the indeces
        after the feature tokenizer.

        Parameter
        ---------
        source_reindex_modalities: list[list[int]]
            the prior split of features into groups
        reindex_for_ftt: list[list[int]]
            the reindexing of given a `no_split` into
            categoricals first and then numericals
        cat_dict:
            stores which raw feature index is categorical and its cardinality
        num_expanded_features: int
            number of expanded features (one-hot encoded)
            corresponds to the number of features before the feature tokenizer
        """
        variable_type = [
            (col_ids, "cat", cat_dict[col_ids])
            if col_ids in cat_dict
            else (col_ids, "num", None)
            for col_ids in range(num_expanded_features)
        ]

        pre_ftt_to_post_ftt = {}
        for i, pre_ftt in enumerate(reindex_for_ftt):
            pre_ftt_to_post_ftt[pre_ftt] = i

        # rewrite prior grouping w.r.t. rearranged (cat first, then numericals)
        clustering_prior_split = []
        for group in source_reindex_modalities:
            new_group_idx = [(idx, pre_ftt_to_post_ftt[idx]) for idx in group]
            clustering_prior_split.append(new_group_idx)

        # create map for collapsed categoricals
        i_index = 0
        i_post_ft = 0
        post_ft_collapse_map = {}
        while i_post_ft < num_expanded_features:

            # get the pre ft index and the associated metadata
            i_pre_ft = reindex_for_ftt[i_post_ft]
            variable_tuple = variable_type[i_pre_ft]

            # numericals and binary categoricals are not collapsed
            if variable_tuple[1] == "num" or (
                variable_tuple[1] == "cat" and variable_tuple[2] == 1
            ):
                post_ft_collapse_map[i_post_ft] = i_index
                i_post_ft += 1

            # multi-class categoricals are collapsed, we store
            # the collapsed index `i_index` for each index prior to the collapse
            else:
                assert variable_tuple[1] == "cat" and variable_tuple[2] > 1
                for j_offset in range(variable_tuple[2]):
                    post_ft_collapse_map[i_post_ft + j_offset] = i_index
                i_post_ft += variable_tuple[2]
            i_index += 1

        # apply the collapse
        for i in range(len(clustering_prior_split)):
            clustering_prior_split[i] = list(
                set(
                    post_ft_collapse_map[splti_idx[1]]
                    for splti_idx in clustering_prior_split[i]
                )
            )

        num_collapsed_features = sum(len(group) for group in clustering_prior_split)
        logging.info(
            f"[{self.__class__.__name__}] prior split sizes: {num_expanded_features} -> {num_collapsed_features}"
        )

        return clustering_prior_split

    def apply_embedding_per_modality(self, x):
        """
        - Map variables to embedding space
        - Group based on current split estimate
        - Aggregate for each group
        - Return list of embeddings per group/split
        """

        # run feature tokenizer (map raw features to embeddings)
        # Reindex for FT to correctly order nums. and cats.
        x_reindexed = x[..., self.reindex_for_ftt[0]]
        # list of tuples, first element in the tuple - categorical feature tensor, second - numerical
        x_split = torch.split(x_reindexed, self.features_sizes_ftt[0], dim=-1)

        bs, sq, dim_num = x_split[0].size()
        dim_cat = x_split[1].shape[2]
        x_split = (
            x_split[0].view(bs * sq, dim_num),
            x_split[1].view(bs * sq, dim_cat),
        )  # dim = features+1 for CLS token

        x_embeds = self.feature_tokenizer(x_split)

        # drop CLS
        x_embeds = x_embeds[:, :-1, :]

        # reindex based on the current split estimate
        x_grouped = x_embeds[:, self.clustering_split_reindex, :]
        x_grouped = torch.split(x_grouped, self.clustering_split_sizes, dim=1)

        # add CLS to each group and map each group
        x_group_mapped = []
        for i, x_group in enumerate(x_grouped):
            x_group = self.feature_cls_token(x_group)

            # x_trans = self.feature_transformer(x_group) # (bs*sq, feature+cls token, feature_dim)
            x_trans = self.apply_feature_transformer(i, x_group)

            x_cls = self.feature_head(x_trans)  # (bs*sq, out_dim)
            x_cls = x_cls.view(bs, sq, x_cls.size(-1))
            x_group_mapped.append(x_cls)

        return x_group_mapped


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
