import json
import os
import numpy as np
from tabulate import tabulate
import torch
import torch.nn as nn


class Hyperparameters:
    def __init__(self,
                 # for bonds
                 b_n_type=4,
                 # for atoms
                 a_n_node=9, a_n_type=5,
                 # conv dim
                 conv_dim=[[128, 64], 128, [128, 64]],
                 # feature aggregation
                 with_features=False, f_dim=0,
                 # General
                 dropout_rate=0., activation=nn.Tanh(), path=None, seed=420):
        """
        :param b_n_type: Number of bond types/channels (b_dim)
        :param a_n_node: Maximum number of atoms in a molecule 
        :param a_n_type: Number of atom types (m_dim)
        :param conv_dim: convolution dimensions (graph_conv_dim, aux_dim, linear_dim)
        :param with_features: whether or not to aggregate features
        :param f_dim: feature dimensions
        :param dropout_rate: rate to perform dropout
        :param activation: activation function
        :param path:
        :param noise_scale:
        """
        self.b_n_type = b_n_type  # 4

        self.a_n_node = a_n_node  # 9
        self.a_n_type = a_n_type  # 5

        self.conv_dim = conv_dim  # [[128, 64], 128, [128, 64]]
        
        # unpack
        self.graph_conv_dim = self.conv_dim[0]  # [128, 64]
        self.aux_dim = self.conv_dim[1]  # 128
        self.linear_dim = self.conv_dim[2]  # [128, 64]

        self.with_features= with_features  # False
        self.f_dim= f_dim  # 0

        self.dropout_rate= dropout_rate  # 0
        self.activation= activation  # tanh

        self.path = path  # None
        self.seed = seed  # 420

        # load function in the initialization by path argument
        if path is not None:
            if os.path.exists(path) and os.path.isfile(path):
                with open(path, "r") as f:
                    obj = json.load(f)
                    for (key, value) in obj.items():
                        setattr(self, key, value)
            else:
                raise Exception("{} does not exist".format(path))

    def save(self, path):
        self.path = path
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True, cls=NumpyEncoder)

    def print(self):
        rows = []
        for key, value in self.__dict__.items():
            rows.append([key, value])
        print(tabulate(rows))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.numpy().tolist()  # what if I use obj.detach().tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':
    hyper = Hyperparameters()
    hyper.save('test_saving_hyper.json')
