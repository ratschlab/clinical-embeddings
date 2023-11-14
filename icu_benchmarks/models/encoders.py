import gin
import numpy as np
import torch
import torch.nn as nn

from icu_benchmarks.models.layers import (
    LocalBlock,
    PositionalEncoding,
    SelfAttentionSimple,
    SparseBlock,
    TemporalBlock,
    TransformerBlock,
    parrallel_recomb,
)


@gin.configurable("GRUAtt")
class GRUNetAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, num_classes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.attention = SelfAttentionSimple(hidden_dim)

        self.logit = nn.Linear(hidden_dim, num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)
        return h0

    def forward(self, x):
        h0 = self.init_hidden(x)
        out, hn = self.rnn(x, h0)
        pred = self.attention([out, hn])
        pred = self.logit(pred)
        return pred


@gin.configurable("LSTM")
class LSTMNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer_dim,
        num_classes,
        embedding_layer=gin.REQUIRED,
        nb_auxiliary_regression=0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.embedding_layer = embedding_layer(input_dim, hidden_dim)

        self.rnn = nn.LSTM(hidden_dim, hidden_dim, layer_dim, batch_first=True)
        self.logit = nn.Linear(hidden_dim, num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if nb_auxiliary_regression > 0:
            self.aux_pred_layers = []
            for _ in range(nb_auxiliary_regression):
                self.aux_pred_layers.append(nn.Linear(hidden_dim, 1))
            self.aux_pred_layers = nn.ModuleList(self.aux_pred_layers)
        else:
            self.aux_pred_layers = None

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.to(self.device) for t in (h0, c0)]

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        emb = self.embedding_layer(x)
        out, hn = self.rnn(emb, (h0, c0))
        pred = self.logit(out)
        if self.aux_pred_layers is None:
            return pred
        else:
            aux_pred = torch.cat([layer(out) for layer in self.aux_pred_layers], dim=-1)
            return pred, aux_pred


@gin.configurable("GRU")
class GRUNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer_dim,
        num_classes,
        embedding_layer=gin.REQUIRED,
        nb_auxiliary_regression=0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.embedding_layer = embedding_layer(input_dim, hidden_dim)

        self.rnn = nn.GRU(hidden_dim, hidden_dim, layer_dim, batch_first=True)
        self.logit = nn.Linear(hidden_dim, num_classes)
        if nb_auxiliary_regression > 0:
            self.aux_pred_layers = []
            for _ in range(nb_auxiliary_regression):
                self.aux_pred_layers.append(nn.Linear(hidden_dim, 1))
            self.aux_pred_layers = nn.ModuleList(self.aux_pred_layers)
        else:
            self.aux_pred_layers = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)
        return h0

    def forward(self, x):
        h0 = self.init_hidden(x)
        emb = self.embedding_layer(x)
        out, hn = self.rnn(emb, h0)
        pred = self.logit(out)
        if self.aux_pred_layers is None:
            return pred
        else:
            aux_pred = torch.cat([layer(out) for layer in self.aux_pred_layers], dim=-1)
            return pred, aux_pred


@gin.configurable("Transformer")
class Transformer(nn.Module):
    def __init__(
        self,
        emb,
        hidden,
        heads,
        ff_hidden_mult,
        depth,
        num_classes,
        dropout=0.0,
        pos_encoding=True,
        dropout_att=0.0,
        embedding_layer=gin.REQUIRED,
        nb_auxiliary_regression=0,
    ):
        super().__init__()
        self.embedding_layer = embedding_layer(emb, hidden)

        if pos_encoding:
            self.pos_encoder = PositionalEncoding(hidden)
        else:
            self.pos_encoder = None

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(
                    emb=hidden,
                    hidden=hidden,
                    heads=heads,
                    mask=True,
                    ff_hidden_mult=ff_hidden_mult,
                    dropout=dropout,
                    dropout_att=dropout_att,
                )
            )

        self.tblocks = nn.Sequential(*tblocks)
        self.logit = nn.Linear(hidden, num_classes)
        if nb_auxiliary_regression > 0:
            self.aux_pred_layers = []
            for _ in range(nb_auxiliary_regression):
                self.aux_pred_layers.append(nn.Linear(hidden, 1))
            self.aux_pred_layers = nn.ModuleList(self.aux_pred_layers)
        else:
            self.aux_pred_layers = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = self.embedding_layer(x)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        x = self.tblocks(x)
        pred = self.logit(x)
        if self.aux_pred_layers is None:
            return pred
        else:
            aux_pred = torch.cat([layer(x) for layer in self.aux_pred_layers], dim=-1)
            return pred, aux_pred


@gin.configurable("LocalTransformer")
class LocalTransformer(nn.Module):
    def __init__(
        self,
        emb,
        hidden,
        heads,
        ff_hidden_mult,
        depth,
        num_classes,
        dropout=0.0,
        pos_encoding=True,
        local_context=1,
        dropout_att=0.0,
        embedding_layer=gin.REQUIRED,
        nb_auxiliary_regression=0,
    ):
        super().__init__()
        self.embedding_layer = embedding_layer(emb, hidden)

        if pos_encoding:
            self.pos_encoder = PositionalEncoding(hidden)
        else:
            self.pos_encoder = None

        tblocks = []
        for i in range(depth):
            tblocks.append(
                LocalBlock(
                    emb=hidden,
                    hidden=hidden,
                    heads=heads,
                    mask=True,
                    ff_hidden_mult=ff_hidden_mult,
                    local_context=local_context,
                    dropout=dropout,
                    dropout_att=dropout_att,
                )
            )

        self.tblocks = nn.Sequential(*tblocks)
        self.logit = nn.Linear(hidden, num_classes)
        if nb_auxiliary_regression > 0:
            self.aux_pred_layers = []
            for _ in range(nb_auxiliary_regression):
                self.aux_pred_layers.append(nn.Linear(hidden, 1))
            self.aux_pred_layers = nn.ModuleList(self.aux_pred_layers)
        else:
            self.aux_pred_layers = None

    def forward(self, x):
        x = self.embedding_layer(x)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        x = self.tblocks(x)
        pred = self.logit(x)
        if self.aux_pred_layers is None:
            return pred
        else:
            aux_pred = torch.cat([layer(x) for layer in self.aux_pred_layers], dim=-1)
            return pred, aux_pred


# From TCN original paper https://github.com/locuslab/TCN
@gin.configurable("TCN")
class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_channels,
        num_classes,
        max_seq_length=0,
        kernel_size=2,
        dropout=0.0,
        embedding_layer=gin.REQUIRED,
        nb_auxiliary_regression=0,
    ):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.embedding_layer = embedding_layer(num_inputs, num_channels)

        # We compute automatically the depth based on the desired seq_length.
        if isinstance(num_channels, int) and max_seq_length:
            num_channels = [num_channels] * int(
                np.ceil(np.log(max_seq_length / 2) / np.log(kernel_size)) + 1
            )
        elif isinstance(num_channels, int) and not max_seq_length:
            raise Exception(
                "a maximum sequence length needs to be provided if num_channels is int"
            )

        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_channels[i - 1] if i != 0 else num_channels[0]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)
        self.logit = nn.Linear(num_channels[-1], num_classes)
        if nb_auxiliary_regression > 0:
            self.aux_pred_layers = []
            for _ in range(nb_auxiliary_regression):
                self.aux_pred_layers.append(nn.Linear(num_channels[-1], 1))
            self.aux_pred_layers = nn.ModuleList(self.aux_pred_layers)
        else:
            self.aux_pred_layers = None

    def forward(self, x):
        x = self.embedding_layer(x)
        x = x.permute(0, 2, 1)  # Permute to channel first
        o = self.network(x)
        o = o.permute(0, 2, 1)  # Permute to channel last
        pred = self.logit(o)
        if self.aux_pred_layers is None:
            return pred
        else:
            aux_pred = torch.cat([layer(o) for layer in self.aux_pred_layers], dim=-1)
            return pred, aux_pred
