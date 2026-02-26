import torch.nn as nn
import torch

class FusionConcat(nn.Module):
    """Simple concatenation of video + tabular vectors"""

    def __init__(self, in_dim=None):
        super().__init__()
        self.out_dim = in_dim  # expected fused dimension

    def forward(self, z_video, z_tab):
        return torch.cat([z_video, z_tab], dim=1)