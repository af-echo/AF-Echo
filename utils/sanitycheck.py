import torch
from models.selfattention.videoattblock import VideoAttentionBlock

def sanity_check(model, loaders, device):
    print("\n=============== SANITY CHECK START ===============")

    model.eval()
    with torch.no_grad():
        print("\n----------- Model Architectuer ------------")
        print(f"use_video   = {model.use_video}")
        print(f"use_tabular = {model.use_tabular}")

        # --------------------------------------
        # Show all local attentions
        # --------------------------------------
        print("\n[Check] In-layer attention detected:")
        for name, module in model.video_encoder.named_modules():
            if isinstance(module, VideoAttentionBlock):
                module.debug = True        # allow debug printing
                module._printed = False    # allow print only once
                print(f"  - {name} ")
        
        # --------------------------------------
        # Enable global attention debugging
        # --------------------------------------
        print("\n[Check] Global attention detected!")
        if hasattr(model, "video_att"):
            model.video_att.debug = True
            model.video_att._printed = False
            # print("  - global attention (debug enabled)")
        # print("------------------------------------------")

        # --------------------------------------
        # Load one mini-batch
        # --------------------------------------
        sample_batch = next(iter(loaders["train"]))

        video_batch  = sample_batch["frames"].to(device) if model.use_video else None
        tab_batch    = sample_batch["tabular"].to(device) if model.use_tabular else None

        # --------------------------------------
        # Forward pass
        # --------------------------------------
        outputs = model(video_batch, tab_batch)

        print("\n------------- Output Statistics -------------")
        for task_name, y_out in outputs.items():
            print(f"[Check] Prediction Task:  {task_name}: mean={y_out.mean().item():.4f}")

        # --------------------------------------
        # Video branch debug
        # --------------------------------------
        if model.use_video:
            feat = model.video_encoder(video_batch)
            print(f"[Check] Backbone features output: {feat.shape}")

            v = model.video_att(feat)
            print(f"[Check] Global attention output: {v.shape}")
        else:
            print("[ATTENTION!] Video branch DISABLED")

        # --------------------------------------
        # Tabular branch debug
        # --------------------------------------
        if model.use_tabular:
            t = model.tabular_encoder(tab_batch)
            print(f"[Check] Tabular features output: {t.shape}")
        else:
            print("[ATTENTION!] Tabular branch DISABLED")
        print("------------------------------------------")
        # --------------------------------------
        # Fusion
        # --------------------------------------
        print("\n------------- Fusion Output -------------")
        if model.use_video and model.use_tabular:
            fused = model.fusion(v, t)
            print(f"[Check] Fusion BOTH | fused.shape = {fused.shape}")
        elif model.use_video:
            fused = model.fusion(v)
            print(f"[Check] Fusion VIDEO ONLY | fused.shape = {fused.shape}")
        else:
            fused = model.fusion(t)
            print(f"[Check] Fusion TABULAR ONLY | fused.shape = {fused.shape}")
        print("------------------------------------------")
    print("=================== SANITY CHECK END =====================\n")
# # -----------------------------------------------------------
#     # 🔍 Sanity check: model output range (to verify logits)
#     model.eval()
#     with torch.no_grad():

#         print("\n================ Sanity Check: Model Branches ================")
#         print(f"use_video   = {model.use_video}")
#         print(f"use_tabular = {model.use_tabular}")

#         # Take one small batch from the training loader
#         sample_batch = next(iter(loaders["train"]))

#         # ------------------------ VIDEO INPUT ------------------------
#         if model.use_video:
#             video_batch = sample_batch["frames"].to(device)
#         else:
#             video_batch = None

#         # ------------------------ TABULAR INPUT ----------------------
#         if model.use_tabular:
#             tab_batch = sample_batch.get("tabular", None)
#             if tab_batch is not None:
#                 tab_batch = tab_batch.to(device)
#         else:
#             tab_batch = None

#         # ------------------------ MODEL OUTPUT -----------------------
#         outputs = model(video_batch, tab_batch)

#         print("\n================ Sanity Check: Model Outputs ================")
#         for task_name, y_out in outputs.items():
#             print(f"[Sanity check] Task: {task_name}")
#             print(f"  Output mean: {y_out.mean().item():.4f}")

#         # ------------------------ VIDEO BRANCH -----------------------
#         if model.use_video:
#             feat = model.video_encoder(video_batch)
#             print(f"[Check] Backbone output (before attention): {feat.shape}")

#             v = model.video_att(feat)
#             print(f"[Check] Video attention output: {v.shape}")
#         else:
#             print("[Check] Video branch DISABLED")

#         # ---------------------- TABULAR BRANCH -----------------------
#         if model.use_tabular:
#             t = model.tabular_encoder(tab_batch)
#             print(f"[Check] Tabular encoder output: {t.shape}")
#         else:
#             print("[Check] Tabular branch DISABLED")

#         # -------------------------- FUSION ---------------------------
#         if model.use_video and model.use_tabular:
#             fused = model.fusion(v, t)
#             print(f"[Check] Fusion BOTH | fused.shape = {fused.shape}")
#         elif model.use_video:
#             fused = model.fusion(v)
#             print(f"[Check] Fusion VIDEO ONLY | fused.shape = {fused.shape}")
#         elif model.use_tabular:
#             fused = model.fusion(t)
#             print(f"[Check] Fusion TABULAR ONLY | fused.shape = {fused.shape}")

#         print("\n=============================================================")
#     # -----------------------------------------------------------