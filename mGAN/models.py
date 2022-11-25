import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mGAN.layers import GraphConvolution, GraphAggregation, MultiGraphConvolutionLayers, MultiDenseLayer
from mGAN.hyperparams import Hyperparameters


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Discriminator(nn.Module):
    """
    Discriminator adapted from 
    https://github.com/ZhenyueQin/Implementation-MolGAN-PyTorch/blob/master/models_gan.py
    
    Now implemted to work with the Hyperparameter class from moflow
    """
    def __init__(self, hyper_params: Hyperparameters):
        super(Discriminator, self).__init__()
        self.hyper_params = hyper_params  # hold all the parameters, easy for save and load for further usage

        # More parameters derived from hyper_params for easy use
        self.b_n_type = hyper_params.b_n_type  # 4
        self.a_n_node = hyper_params.a_n_node  # 9
        self.a_n_type = hyper_params.a_n_type  # 5

        self.conv_dim = hyper_params.conv_dim  # [[128, 64], 128, [128, 64]]

        # unpack
        self.graph_conv_dim = self.conv_dim[0]  # [128, 64]
        self.aux_dim = self.conv_dim[1]  # 128
        self.linear_dim = self.conv_dim[2]  # [128, 64]

        self.with_features= hyper_params.with_features  # False
        self.f_dim= hyper_params.f_dim  # 0

        self.lam= hyper_params.lam  # 10

        self.activation_f = hyper_params.activation

        # discriminator
        self.gcn_layer = GraphConvolution(self.a_n_type, self.graph_conv_dim, self.b_n_type, self.with_features, self.f_dim, self.dropout_rate)
        self.agg_layer = GraphAggregation(self.graph_conv_dim[-1] + self.a_n_type, self.aux_dim, self.activation_f, self.with_features, self.f_dim, self.dropout_rate)
        self.multi_dense_layer = MultiDenseLayer(self.aux_dim, self.linear_dim, self.activation_f, dropout_rate=self.dropout_rate)

        self.output_layer = nn.Linear(self.linear_dim[-1], 1)

    def forward(self, adj, hidden, node, activation=None):
        adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        h_1 = self.gcn_layer(node, adj, hidden)
        h = self.agg_layer(h_1, node, hidden)
        h = self.multi_dense_layer(h)

        output = self.output_layer(h)
        output = activation(output) if activation is not None else output

        return output, h


# class Discriminator2(nn.Module):
#     """Discriminator network with PatchGAN."""
#     def __init__(self, conv_dim, self.a_n_type, self.b_n_type, self.dropout, batch=self.True):
#         super(Discriminator, self).__init__()
#         self.batch= batch
#         self.graph_conv_dim, self.aux_dim, self.linear_dim = conv_dim

#         # discriminator
#         self.gcn_layer = GraphConvolution(self.a_n_type, self.graph_conv_dim, self.b_n_type, self.dropout)
#     self.    self.agg_layer = GraphAggregation(self.graph_conv_dim[-1], self.aux_dim, self.b_n_type, self.dropout)

#    self.     # multi dense layer
#         layers = []
#         for c0, c1 in zip([self.aux_dim]+self.linear_dim[:-1], self.linear_dim):
#             layers.append(nn.Linear(c0,c1))
#             layers.append(nn.Dropout(dropout))
#         self.multi_linear_layer = nn.Sequential(*layers)

#         # linear layer with tanh activation (tf.layers.dense)
#         self.linear_tanh= nn.Sequential(
#             nn.Linear(self.linear_dim[-1], self.aux_dim // 8),
#             nn.Tanh(),
#         )

#         self.output_layer = nn.Linear(self.linear_dim[-1], 1)

#     def forward(self, adj, hidden, node, activation=None):
#         adj = adj[:,:,:,1:].permute(0,3,1,2)
#         annotations = torch.cat((hidden, node), -1) if hidden is not None else node
#         h = self.gcn_layer(annotations, adj)
#         annotations = torch.cat((h, hidden, node) if hidden is not None else (h, node), -1)
#         h = self.agg_layer(annotations, torch.tanh)
#         h = self.multi_linear_layer(h)
        
#         # batch discriminator implementation
#         if self.batch:
#             output_batch= self.linear_tanh(h)
#             output_batch= torch.mean(output_batch, 0, keepdim=True)
#             output_batch= self.linear_tanh(output_batch)


#         output = self.output_layer(h)
#         output = activation(output) if activation is not None else output

#         return output, h