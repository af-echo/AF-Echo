import torch
import torch.nn as nn


class MultiModalCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed_dim = config.get("embed_dim", 256)
        self.num_heads = config.get("num_heads", 4)
        self.dropout = config.get("dropout", 0.1)
        self.fusion_mode = config.get("fusion_mode", "bidirectional").lower()
        self.combine_mode = config.get("combine_mode", "avg").lower()

        # 🔹 Separate token counts per modality
        self.tokens_per_video = config.get("tokens_per_video", 1)
        self.tokens_per_tabular = config.get("tokens_per_tabular", 1)

        # 🔹 Learnable token projection layers (persistent parameters)
        if self.tokens_per_video > 1:
            self.video_token_proj = nn.Linear(
                self.embed_dim, self.embed_dim * self.tokens_per_video
            )
        else:
            self.video_token_proj = None

        if self.tokens_per_tabular > 1:
            self.tab_token_proj = nn.Linear(
                self.embed_dim, self.embed_dim * self.tokens_per_tabular
            )
        else:
            self.tab_token_proj = None

        # 🔹 Multihead cross-attention layers
        self.attn_v2t = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.attn_t2v = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )

        # 🔹 Normalization
        self.norm_v = nn.LayerNorm(self.embed_dim)
        self.norm_t = nn.LayerNorm(self.embed_dim)

        # 🔹 Optional projection if concat fusion
        if self.fusion_mode == "bidirectional" and self.combine_mode == "concat":
            self.proj = nn.Linear(2 * self.embed_dim, self.embed_dim)
        else:
            self.proj = nn.Identity()

    # ----------------------------------------------------------------------
    def _expand_tokens(self, z, proj_layer, n_tokens):
        """Expands single embedding (B, D) → (B, n_tokens, D)."""
        if n_tokens == 1:
            return z.unsqueeze(1)  # (B, 1, D)
        else:
            z_exp = proj_layer(z).reshape(z.size(0), n_tokens, self.embed_dim)
            return z_exp

    # ----------------------------------------------------------------------
    def forward(self, z_video, z_tab):
        """
        Args:
            z_video: (B, D)
            z_tab:   (B, D)
        Returns:
            fused: (B, D)
        """
        B, D = z_video.shape

        # Expand tokens according to configuration
        z_video = self._expand_tokens(z_video, self.video_token_proj, self.tokens_per_video)
        z_tab = self._expand_tokens(z_tab, self.tab_token_proj, self.tokens_per_tabular)

        # # -------------------------------
        # # Perform cross-attention
        # # -------------------------------
        if self.fusion_mode == "v2t":
            attn_out, attn_w = self.attn_v2t(query=z_video, key=z_tab, value=z_tab,
                                            need_weights=True, average_attn_weights=False)
            fused = self.norm_v(z_video + attn_out)
            # 🔹 Store attention for inspection
            self.last_attn_v2t = attn_w.detach().cpu()  # (B, heads, tokens_v, tokens_t)

        elif self.fusion_mode == "t2v":
            attn_out, attn_w = self.attn_t2v(query=z_tab, key=z_video, value=z_video,
                                            need_weights=True, average_attn_weights=False)
            fused = self.norm_t(z_tab + attn_out)
            self.last_attn_t2v = attn_w.detach().cpu()

        elif self.fusion_mode == "bidirectional":
            v2t_out, attn_v2t = self.attn_v2t(query=z_video, key=z_tab, value=z_tab,
                                            need_weights=True, average_attn_weights=False)
            t2v_out, attn_t2v = self.attn_t2v(query=z_tab, key=z_video, value=z_video,
                                            need_weights=True, average_attn_weights=False)

            # 🔹 Save both directions
            self.last_attn_v2t = attn_v2t.detach().cpu()
            self.last_attn_t2v = attn_t2v.detach().cpu()

            v2t_out = self.norm_v(z_video + v2t_out)
            t2v_out = self.norm_t(z_tab + t2v_out)

            if self.combine_mode == "avg":
                fused = 0.5 * (v2t_out.mean(dim=1) + t2v_out.mean(dim=1)).unsqueeze(1)
            elif self.combine_mode == "concat":
                fused = torch.cat(
                    [v2t_out.mean(dim=1), t2v_out.mean(dim=1)], dim=-1
                )
                fused = self.proj(fused)  # (B, D)
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        # -------------------------------
        # Pool over tokens
        # -------------------------------
        fused = fused.mean(dim=1)  # (B, D)
        return fused




        # if self.fusion_mode == "v2t":
        #     attn_out, _ = self.attn_v2t(query=z_video, key=z_tab, value=z_tab)
        #     fused = self.norm_v(z_video + attn_out)

        # elif self.fusion_mode == "t2v":
        #     attn_out, _ = self.attn_t2v(query=z_tab, key=z_video, value=z_video)
        #     fused = self.norm_t(z_tab + attn_out)

        # elif self.fusion_mode == "bidirectional":
        #     v2t_out, _ = self.attn_v2t(query=z_video, key=z_tab, value=z_tab)
        #     v2t_out = self.norm_v(z_video + v2t_out)

        #     t2v_out, _ = self.attn_t2v(query=z_tab, key=z_video, value=z_video)
        #     t2v_out = self.norm_t(z_tab + t2v_out)

        #     if self.combine_mode == "avg":
        #         fused = 0.5 * (v2t_out.mean(dim=1) + t2v_out.mean(dim=1)).unsqueeze(1)
        #     elif self.combine_mode == "concat":
        #         fused = torch.cat(
        #             [v2t_out.mean(dim=1), t2v_out.mean(dim=1)], dim=-1
        #         ).unsqueeze(1)
        #         fused = self.proj(fused)
        # else:
        #     raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")