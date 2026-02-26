import torch
import torch.nn as nn
from pathlib import Path
from models.backbone.mlpblock import MLPEncoder


class TabularEncoder(nn.Module):
    def __init__(self, tab_cfg):
        super().__init__()

        # build the encoder (config-based)
        self.encoder = MLPEncoder(tab_cfg)

        # load pretrained weights if specified
        pretrained_path = tab_cfg.get("pretrained_path", None)
        freeze = tab_cfg.get("freeze", False)

        if pretrained_path and Path(pretrained_path).exists():
            self._load_pretrained(pretrained_path, freeze)
        else:
            if pretrained_path:
                print(f"⚠️ Pretrained path not found: {pretrained_path}")

        self.out_dim = tab_cfg.get("out_dim", 256)

    def _load_pretrained(self, path, freeze=False):
        try:
            ckpt = torch.load(path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            model_dict = self.encoder.state_dict()
            compat = {k: v for k, v in state_dict.items()
                      if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(compat)
            self.encoder.load_state_dict(model_dict)
            print(f"✅ Loaded {len(compat)}/{len(model_dict)} params from {path}")
        except Exception as e:
            print(f"⚠️ Failed to load pretrained FCN: {e}")

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("🧊 Tabular encoder frozen.")

    def forward(self, x):
        t = self.encoder(x)
        if t.ndim > 2:
            t = t.squeeze(-1)
        return t
