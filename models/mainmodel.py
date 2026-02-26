import torch
import torch.nn as nn
from models.backbone.custom3dnet import Custom3DNet 
from models.backbone.r2plus1d import R2Plus1DModel
from models.backbone.resnet import ResNetModel
from models.backbone.fcnet import TabularEncoder
from models.backbone.mlpblock import MLPEncoder
from models.fusion import FUSION_REGISTRY
from models.fusion.crossatt import MultiModalCrossAttention
from models.selfattention.videoattblock import VideoAttentionBlock


class MultimodalModel(nn.Module):
    """
    Builds video encoder + tabular encoder + fusion + heads from YAML.
    """
    def __init__(self, cfg, task_info: dict):
        super().__init__()
        mcfg = cfg["model"]
        self.task_info = task_info
        self.use_tabular = mcfg.get("use_tabular", True)
        self.use_video = mcfg.get("use_video", True)

        # ---------------------- VIDEO BACKBONE ----------------------
        self.use_video = mcfg.get("use_video", True)
        
        if self.use_video:
            backbone_name = mcfg.get("video_backbone", "custom3dnet").lower()
            if backbone_name == "custom3dnet":
                self.video_encoder = Custom3DNet(mcfg)
            elif backbone_name == "r2plus1d":
                self.video_encoder = R2Plus1DModel(mcfg)
            elif backbone_name == "resnet":
                self.video_encoder = ResNetModel(mcfg)
            else:
                raise ValueError(f"❌ Unknown video backbone: {backbone_name}")

            # ---------------------- VIDEO ATTENTION BLOCK ----------------------
            self.video_att = VideoAttentionBlock(
                in_channels=self.video_encoder.out_dim,
                cfg=mcfg
            )
            # ⚠️ For flatten mode, video_att.out_dim is not reliable,
            # so we will NOT use Dv here for concat fusion.
            # But we keep it for other fusion types if needed.
            Dv_backbone = self.video_encoder.out_dim
        else:
            self.video_encoder = None
            self.video_att = None
            Dv_backbone = 0  # important!
        
        # ---------------------- TABULAR ENCODER ----------------------
        if self.use_tabular:
            self.tabular_encoder = TabularEncoder(mcfg["tabular"])   # FIXED (added)
            Dt = self.tabular_encoder.out_dim
        else:
            self.tabular_encoder = None
            Dt = 0

        # ---------------------- FUSION LOGIC ----------------------
        fcfg = mcfg["fusion"]
        ftype = fcfg.get("type", "mlp").lower()
        pred_cfg = mcfg.get("predhead", {})

        # ---- CASE 1: BOTH VIDEO + TABULAR ----
        if self.use_video and self.use_tabular:

            if ftype == "mlp":
                Dv = Dv_backbone
                self.fusion = FUSION_REGISTRY["mlp"](
                    in_dim=Dv + Dt,
                    hidden_dims=fcfg.get("hidden_dims", [256, 128]),
                    dropout=fcfg.get("dropout", 0.2),
                )
                fusion_out_dim = self.fusion.out_dim

            elif ftype == "concatenate":
                self.fusion = FUSION_REGISTRY["concatenate"]()
                fusion_out_dim = pred_cfg.get("input_dim", Dv_backbone + Dt)

            elif ftype == "cross_attention":
                self.fusion = MultiModalCrossAttention(fcfg)
                fusion_out_dim = fcfg.get("embed_dim", 256)

            else:
                raise ValueError(f"Unknown fusion type: {ftype}")

        # ---- CASE 2: VIDEO ONLY ----
        elif self.use_video and not self.use_tabular:
            self.fusion = nn.Identity()

            # NEW: fusion_out_dim MUST equal video attention output size
            if hasattr(self.video_att, "out_dim"):
                fusion_out_dim = self.video_att.out_dim
            else:
                raise ValueError("VideoAttentionBlock must implement out_dim attribute!")

        # ---- CASE 3: TABULAR ONLY ----
        elif self.use_tabular and not self.use_video:
            self.fusion = nn.Identity()
            fusion_out_dim = Dt

        # ---- CASE 4: NONE ----
        else:
            raise ValueError("❌ Both use_video=False and use_tabular=False. Nothing to train.")

        # ---------------------- HEADS ----------------------
        pred_cfg = mcfg.get("predhead", {})
        self.multitask_head = bool(pred_cfg.get("multitask_head", True))
        self.heads = nn.ModuleDict({
            task: MLPEncoder({       # 🔹 reuse your existing MLPEncoder
                "input_dim": pred_cfg.get("input_dim", 64),
                "hidden_dims": pred_cfg.get("hidden_dims", [128]),
                "out_dim": pred_cfg.get("out_dim", 1),
                "activation": pred_cfg.get("activation", "relu"),
                "normalization": pred_cfg.get("normalization", "batchnorm"),
                "dropout": pred_cfg.get("dropout", 0.3),
            })
            for task in task_info
        })

    # ---------------------- FORWARD ----------------------
    def forward(self, video=None, tabular=None):

        # 1. Video
        if self.use_video:
            feat = self.video_encoder(video)
            v = self.video_att(feat)
        else:
            v = None

        # 2. Tabular
        if self.use_tabular:
            t = self.tabular_encoder(tabular)
        else:
            t = None
        
        if self.use_video:
            assert v.dim() == 2, f"Expected pooled video embedding (B,D), got {v.shape}"
        if self.use_tabular:
            assert t.dim() == 2, f"Expected tabular embedding (B,D), got {t.shape}"

        # 3. Fusion depending on active modalities
        if self.use_video and self.use_tabular:
            z = self.fusion(v, t)

        elif self.use_video:
            z = v # Identity

        elif self.use_tabular:
            z = t # Identity

        # 4. Heads
        if self.multitask_head:
            return {task: head(z) for task, head in self.heads.items()}

        (only_task, only_head), = self.heads.items()
        return only_head(z)