import torch.nn as nn

class FusionMLP(nn.Module):
    """MLP-based fusion — project concatenated features through MLP"""
    def __init__(self, in_dim, hidden_dims=[256, 128], dropout=0.2):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.net = nn.Sequential(*layers)

    def forward(self, z_video, z_tab):
        x = torch.cat([z_video, z_tab], dim=1)
        return self.net(x)
