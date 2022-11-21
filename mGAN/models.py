import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mGAN.layers import GraphConvolution, GraphAggregation, MultiGraphConvolutionLayers, MultiDenseLayer


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
    """Discriminator from 
    https://github.com/ZhenyueQin/Implementation-MolGAN-PyTorch/blob/master/models_gan.py"""
    def __init__(self, conv_dim, m_dim, b_dim, with_features=False, f_dim=0, dropout_rate=0.):
        super(Discriminator, self).__init__()
        self.activation_f = nn.Tanh()
        graph_conv_dim, aux_dim, linear_dim = conv_dim

        # discriminator
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim, with_features, f_dim, dropout_rate)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1] + m_dim, aux_dim, self.activation_f, with_features, f_dim, dropout_rate)
        self.multi_dense_layer = MultiDenseLayer(aux_dim, linear_dim, self.activation_f, dropout_rate=dropout_rate)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

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
#     def __init__(self, conv_dim, m_dim, b_dim, dropout, batch=True):
#         super(Discriminator, self).__init__()
#         self.batch= batch
#         self.graph_conv_dim, self.aux_dim, self.linear_dim = conv_dim

#         # discriminator
#         self.gcn_layer = GraphConvolution(m_dim, self.graph_conv_dim, b_dim, dropout)
#         self.agg_layer = GraphAggregation(self.graph_conv_dim[-1], self.aux_dim, b_dim, dropout)

#         # multi dense layer
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