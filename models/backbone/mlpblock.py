import torch
import torch.nn as nn
from pathlib import Path

class MLPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        input_dim = config.get("input_dim", 256)
        hidden_dims = config.get("hidden_dims", [256, 128])
        out_dim = config.get("out_dim", 256)
        activation_name = config.get("activation", "relu").lower()
        normalization = config.get("normalization", "batchnorm").lower()
        dropout_p = config.get("dropout", 0.0)

        act_map = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "leakyrelu": nn.LeakyReLU(0.2, inplace=True),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        activation = act_map.get(activation_name, nn.ReLU(inplace=True))

        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if normalization == "batchnorm":
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            elif normalization == "layernorm":
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(activation)
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
        layers.append(nn.Linear(hidden_dims[-1], out_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)  # (B, out_dim)
