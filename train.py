import os
import json
from pyexpat import model
import yaml
import random
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, KFold
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb as wandb
from tqdm import tqdm
from joblib import dump
import time 
import cv2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

# local imports
from utils.task_type import infer_task_types
from data.loader import EchoDataset
from utils.losses import compute_multitask_loss
from models.backbone.r2plus1d import R2Plus1DModel
from models.backbone.resnet import ResNetModel
from models.backbone.custom3dnet import Custom3DNet
from models.mainmodel import MultimodalModel
from utils.transform import compute_video_mean_std
from utils.custcollate import multitask_collate
from utils.sanitycheck import sanity_check
from seliency.saliency import input_gradient_saliency, saliency_to_heatmap, overlay_heatmap, integrated_gradients_video
from data.preprocess import load_or_create_balanced_subset , balance_videos_globally


# ===============================================================
# ============================ Utils ============================
# ===============================================================

def print_video_level_stats(df_split, split_name, label_col):
    labels = df_split.groupby("video_name")[label_col].first()
    counts = labels.value_counts().sort_index()
    ratios = counts / counts.sum()

    print(f"\n{split_name.upper()} VIDEO-LEVEL CLASS DISTRIBUTION:")
    for cls in counts.index:
        print(f"  Class {cls}: {counts[cls]} videos ({ratios[cls]*100:.1f}%)")

def print_clip_level_stats(df_split, split_name, label_col):
    counts = df_split[label_col].value_counts().sort_index()
    ratios = counts / counts.sum()

    print(f"\n{split_name.upper()} CLIP-LEVEL DISTRIBUTION:")
    for cls in counts.index:
        print(f"  Class {cls}: {counts[cls]} clips ({ratios[cls]*100:.1f}%)")


# ---------------------------- SEED -----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------- To Device Batch ----------------------
def to_device_batch(batch: dict, device: torch.device, use_video: bool = True, use_tabular: bool = True):
    inputs = {}
    # VIDEO 
    if use_video:
        # Always present in batch["frames"], but only moved if needed
        inputs["video"] = batch["frames"].to(device, non_blocking=True)
    else:
        inputs["video"] = None  # Explicitly mark inactive branch
    # TABULAR 
    if use_tabular:
        tab = batch.get("tabular", None)
        if tab is not None:
            tab = tab.to(device, non_blocking=True)
        inputs["tabular"] = tab
    else:
        inputs["tabular"] = None
    # TARGETS 
    targets = {k: v.to(device) for k, v in batch["targets"].items()}
    # META 
    meta = {
        "video_id": batch.get("video_id", None),
        "video_name": batch.get("video_name", None),
        "fps": batch.get("fps", None),
        "num_frames": batch.get("num_frames", None),
    }
    return inputs, targets, meta

# ================================================================
# ================== Build Model and Loader ======================
# ================================================================

# ---------------- Model factory ----------------
def create_model_from_cfg(model_cfg: dict, task_info: dict) -> nn.Module:
    name = str(model_cfg.get("name", "")).lower()
    if name in ("r2plus1d", "r(2+1)d", "r2p1d"):
        return R2Plus1DModel(**model_cfg)
    elif name in ("resnet3d", "resnet"):
        return ResNetModel(**model_cfg)
    elif name in ("custom3d", "custom3dnet"):
        return Custom3DNet(**model_cfg)
    elif name in ("multimodal", "multimodel"):
        return MultimodalModel(cfg={"model": model_cfg}, task_info=task_info)
    else:
        raise ValueError(f"Unknown model type '{name}' in model_cfg")
    
# ---------------- Data Loader ----------------
def build_loaders_from_cfg(cfg):
    print("==> Loading Tabular Dataset:", cfg["data"]["csv_path"])
    df = pd.read_csv(cfg["data"]["csv_path"])

    print("==> Loading Video Dataset:", cfg["data"]["video_root"])
    video_root = Path(cfg["data"]["video_root"])

    print("\n===== Pair Checking Video and Tabular =========")

    # Normalize video IDs from disk
    existing_video_ids = {
        p.stem.lower().strip()
        for p in video_root.glob("*.avi")
    }

    # Normalize video_id column in CSV
    df["video_id_norm"] = (
        df["video_id"]
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(".avi", "", regex=False)
    )

    # Keep only paired rows
    df_paired = df[df["video_id_norm"].isin(existing_video_ids)].copy()

    print(f"Original rows: {len(df)}")
    print(f"Paired rows:   {len(df_paired)}")
    print(f"Dropped rows:  {len(df) - len(df_paired)}")

    # Optional but VERY useful diagnostics
    print(f"Original unique videos: {df['video_id_norm'].nunique()}")
    print(f"Paired unique videos:   {df_paired['video_id_norm'].nunique()}")

    print("=============== Building Dataset ================")
    df = pd.read_csv(cfg["data"]["csv_path"])
    task_cols = cfg["data"]["task_cols"]
    tabular_cols = cfg["data"].get("tabular_features", None)

    # ------------ TASK TYPE INFERENCE (on full df) -------------
    task_type = infer_task_types(df, task_cols, exclude_cols=["video_id"])
    regression_cols = [t for t in task_type if task_type[t] == "regression"]
    binary_cols     = [t for t in task_type if task_type[t] == "binary"]

    # -------- GLOBAL VIDEO-LEVEL BALANCING BEFORE SPLIT---------
    if cfg.get("balance", {}).get("mode") == "video_global":
        target_col = cfg["balance"]["target_col"]
        df = balance_videos_globally(df, target_col, seed=cfg.get("seed", 42))

    # ----------- SPLIT BY UNIQUE video_name, NOT ROWS------------
    unique_videos = df["video_name"].unique()

    val_ratio  = float(cfg["data"].get("val_ratio", 0.15))
    test_ratio = float(cfg["data"].get("test_ratio", 0.15))
    seed = cfg.get("seed", 42)

    # stratify by label if requested
    if "stratify_by" in cfg["data"]:
        strat_col = df.groupby("video_name")[cfg["data"]["stratify_by"]].first()
        train_videos, rest_videos = train_test_split(
            unique_videos, 
            test_size=val_ratio + test_ratio,
            random_state=seed,
            stratify=strat_col.loc[unique_videos]
        )
        rest_labels = strat_col.loc[rest_videos]
        val_size = val_ratio / (val_ratio + test_ratio)
        val_videos, test_videos = train_test_split(
            rest_videos,
            test_size=1 - val_size,
            random_state=seed,
            stratify=rest_labels
        )
    else:
        train_videos, rest_videos = train_test_split(
            unique_videos,
            test_size=val_ratio + test_ratio,
            random_state=seed
        )
        val_size = val_ratio / (val_ratio + test_ratio)
        val_videos, test_videos = train_test_split(
            rest_videos,
            test_size=1 - val_size,
            random_state=seed
        )

    # attach clips back
    train_df = df[df["video_name"].isin(train_videos)].reset_index(drop=True)
    val_df   = df[df["video_name"].isin(val_videos)].reset_index(drop=True)
    test_df  = df[df["video_name"].isin(test_videos)].reset_index(drop=True)

    print("==> Split Sizes (video-level):")
    print(f" Train videos: {len(train_videos)} | Clips: {len(train_df)}")
    print(f" Val videos:   {len(val_videos)}   | Clips: {len(val_df)}")
    print(f" Test videos:  {len(test_videos)}  | Clips: {len(test_df)}")

    # -------------LEAKAGE CHECK-----------------------------
    train_vids = set(train_df["video_name"])
    val_vids   = set(val_df["video_name"])
    test_vids  = set(test_df["video_name"])

    assert train_vids.isdisjoint(val_vids), \
        f"❌ DATA LEAKAGE DETECTED: {train_vids.intersection(val_vids)} appear in BOTH train and val sets!"

    assert train_vids.isdisjoint(test_vids), \
        f"❌ DATA LEAKAGE DETECTED: {train_vids.intersection(test_vids)} appear in BOTH train and test sets!"

    assert val_vids.isdisjoint(test_vids), \
        f"❌ DATA LEAKAGE DETECTED: {val_vids.intersection(test_vids)} appear in BOTH val and test sets!"

    print('✓ No data leakage detected between splits (video-level separation OK).')

    # ------------TABULAR NORMALIZER ONLY ON TRAIN----------------
    if len(regression_cols) > 0:
        reg_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", MinMaxScaler())
        ])
        reg_pipeline.fit(train_df[regression_cols])
        dump(reg_pipeline, "regression_tabular_normalizer.joblib")
    else:
        reg_pipeline = None

    # -------------VIDEO NORMALIZATION-----------------------------
    normalization = cfg["data"].get("normalization", "imagenet")
    if normalization == "dataset":
        video_paths = [
            os.path.join(cfg["data"]["video_root"], f"{vid}.avi")
            for vid in train_df["video_id"].values
        ]
        mean, std = compute_video_mean_std(video_paths)
        dataset_stats = (mean, std)
    else:
        dataset_stats = None

    # -----------DATASETS AND LOADERS-------------------------------
    use_video = cfg["model"].get("use_video", True)
    use_tabular = cfg["model"].get("use_tabular", True)

    def make_ds(df_split, is_train: bool):
        return EchoDataset(
            video_dir=cfg["data"]["video_root"],
            tabular_df=df_split,
            task_type=task_type,
            tabular_normalizer=reg_pipeline,
            regression_cols=regression_cols,
            binary_cols=binary_cols,
            tabular_cols=tabular_cols,
            size=cfg["data"].get("size", (112,112)),
            clip_len=cfg["data"].get("clip_len", 32),
            sampling=cfg["data"].get("sampling", "center"),
            normalization=normalization,
            dataset_stats=dataset_stats,
            frame_stride=cfg["data"].get("frame_stride", 1),
            grayscale=cfg["data"].get("grayscale", False),
            use_video=use_video,
            is_train=is_train
        )

    loaders = {
        "train": DataLoader(make_ds(train_df, is_train=True), batch_size=cfg["train"]["batch_size"],
                            shuffle=True,  num_workers=4, pin_memory=True,
                            collate_fn=multitask_collate),
        "val":   DataLoader(make_ds(val_df, is_train=False),   batch_size=cfg["train"]["batch_size"],
                            shuffle=False, num_workers=4, pin_memory=True,
                            collate_fn=multitask_collate),
        "test":  DataLoader(make_ds(test_df, is_train=False),  batch_size=cfg["train"]["batch_size"],
                            shuffle=False, num_workers=4, pin_memory=True,
                            collate_fn=multitask_collate)
    }

    label_col = cfg["data"]["task_cols"][1]

    print_video_level_stats(train_df, "train", label_col)
    print_video_level_stats(val_df,   "val",   label_col)
    print_video_level_stats(test_df,  "test",  label_col)

    print_clip_level_stats(train_df, "train", label_col)
    print_clip_level_stats(val_df,   "val",   label_col)
    print_clip_level_stats(test_df,  "test",  label_col)

    return loaders, task_type, regression_cols, reg_pipeline

# ===============================================================
# =================== Train main ================================
# ===============================================================

# ---------------- Training Epoche ----------------
def train_one_epoch(model, loader, optimizer, device, loss_fn, task_info, loss_weights, use_tqdm=False, 
                    epoch=None, agg="mean", threshold=0.5):
    

    model.train()
    running_loss = 0.0
    all_targets, all_preds, all_probs = {}, {}, {}
    iterator = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False) if use_tqdm else loader

    for batch in iterator:
        inputs, targets, _ = to_device_batch(batch, device, model.use_video, model.use_tabular)
        optimizer.zero_grad()
        if model.use_video and model.use_tabular:
            outputs = model(inputs["video"], inputs["tabular"])
        elif model.use_video:
            outputs = model(inputs["video"], None)
        elif model.use_tabular:
            outputs = model(None, inputs["tabular"])
        else:
            raise RuntimeError("Both video and tabular disabled — cannot train.")
        loss, _ = loss_fn(outputs, targets, task_info, loss_weights, agg=agg)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        for task, y_true in targets.items():
            y_true_np = y_true.detach().cpu().numpy().ravel().tolist()
            all_targets.setdefault(task, []).extend(y_true_np)
            y_out = outputs[task]
            if task_info[task] == 'binary':
                if y_out.shape[-1] == 1:
                    prob = torch.sigmoid(y_out).detach().cpu().numpy().ravel()

                elif y_out.shape[-1] == 2:
                    logit = (y_out[..., 1] - y_out[..., 0])
                    prob = torch.sigmoid(logit).detach().cpu().numpy().ravel()
                else:
                    prob = torch.sigmoid(y_out).detach().cpu().numpy().ravel()

                pred = (prob >= threshold).astype(float)
                all_probs.setdefault(task, []).extend(prob.tolist())
                all_preds.setdefault(task, []).extend(pred.tolist())
            else:
                pred = y_out.detach().cpu().numpy().ravel()
                all_preds.setdefault(task, []).extend(pred.tolist())

    # ---- average loss ----
    avg_loss = running_loss / max(1, len(loader))

    # ---- Compute metrics (per-task + overall) ----
    acc_vals, mae_vals = [], []
    train_stats = {}

    for task, y_true in all_targets.items():
        y_true = np.array(y_true)
        pred = np.array(all_preds[task])

        if task_info[task] == "binary":
            prob = np.array(all_probs[task])
            acc = accuracy_score(y_true, pred)
            acc_vals.append(acc)
            train_stats[f"{task}/accuracy"] = acc
            if len(np.unique(y_true)) > 1:
                auroc = roc_auc_score(y_true, prob)
                train_stats[f"{task}/auroc"] = auroc

        else:  # regression
            mae = mean_absolute_error(y_true, pred)
            mae_vals.append(mae)
            train_stats[f"{task}/mae"] = mae

    # ---- Add overall (aggregated) metrics ----
    if acc_vals:
        train_stats["train/overall_accuracy"] = float(np.mean(acc_vals))
    if mae_vals:
        train_stats["train/overall_mae"] = float(np.mean(mae_vals))

    return avg_loss, train_stats

# ---------------- Training loop ----------------
def run_train(cfg, model, loaders, task_info, loss_weights, device, use_tqdm):
    torch.autograd.set_detect_anomaly(True)
    # --- Start timer ---
    start_time = time.time()

    # --- Model parameter summary ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    model_size_mb = total_params * 4 / (1024 ** 2)

    print("\n================ Model Summary ================")
    print(f"Total parameters   : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters   : {frozen_params:,}")
    print(f"Approx model size   : {model_size_mb:.2f} MB (float32)")
    print("==============================================\n")

    # -- Training loop ---
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"].get("weight_decay", 1e-4))
    epochs = cfg["train"]["epochs"]
    out_dir = Path(cfg["train"].get("out_dir", "runs/exp"))
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val = float("inf")
    start_epoch = 0

    if cfg["train"].get("resume") and last_path.exists():
        state = torch.load(last_path)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = state["epoch"] + 1
        print(f"==> Resumed from epoch {start_epoch}")

    patience = cfg["train"].get("early_stopping_patience", 10)  
    early_stop_counter = 0
    best_auroc = -float("inf")

    for epoch in range(start_epoch, epochs + 1):
        # ---- Training ----
        train_loss, train_stats = train_one_epoch(
            model, loaders["train"], optimizer, device,
            compute_multitask_loss, task_info, loss_weights,
            use_tqdm, epoch=epoch, agg=cfg["train"].get("loss_agg", "mean"),
            threshold=cfg.get("eval", {}).get("threshold", 0.5)
        )
        
        # ---- Validation ----
        if loaders.get("val") is not None:
            overall, _ = evaluate_model(
                model, loaders["val"], device, compute_multitask_loss,
                task_info=task_info, loss_weights=loss_weights,
                threshold=cfg.get("eval", {}).get("threshold", 0.5),
                inverse_transform=False,
                agg=cfg["train"].get("loss_agg", "mean"),
                metric_level=cfg.get("eval", {}).get("metric_level", "clip"), 
                video_key="video_name",
                video_reduce_cfg=cfg["eval"].get("video_reduce", None)           
            )
            val_loss = overall["loss"]
            val_auroc = overall.get("auroc", None)
        else:
            val_loss, overall = float("nan"), {}

        scheduler.step()
        
        # ---- Save checkpoints ----
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss
        }, last_path)

        if val_loss < best_val:
            best_val = val_loss
            early_stop_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss
            }, best_path)
        else:
            early_stop_counter += 1
            
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_auroc={val_auroc:.4f}")    
        if early_stop_counter >= patience:
            print(f"⏹️ Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
            break

        # ---- Log to wandb ----
        if cfg.get("wandb", {}).get("enable", False):
            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": scheduler.get_last_lr()[0],
            }

            for k, v in overall.items():
                if not "/" in k:  
                    log_dict[f"val/{k}"] = v

            # add overall training metrics
            for k, v in train_stats.items():
                if "overall" in k or "auroc" in k:
                    log_dict[k] = v

            wandb.log(log_dict)

    end_time = time.time()
    elapsed = end_time - start_time
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    print("\n================ Training Finished ================")
    print(f"Total training time: {h:02d}h:{m:02d}m:{s:02d}s")
    print(f"Best validation loss: {best_val:.4f}")
    print("===================================================\n")
    return best_path

# ===============================================================
# =================== Evalueate main ============================
# ===============================================================

# ------------------ save eval metrices -------------------------
def save_metrics(out_dir: Path, split_name: str,
                 overall: Dict[str, float],
                 per_task: Dict[str, Dict[str, float]]):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{split_name}_overall.json").open("w") as f:
        json.dump(overall, f, indent=2)
    with (out_dir / f"{split_name}_per_task.json").open("w") as f:
        json.dump(per_task, f, indent=2)
# ---------------------- Sigmoid --------------------------------
def _sigmoid(x):
    return 1 / (1 + np.exp(-x))
# ------------------- Reducer Functions -------------------------
def reduce_video_scores(vals, reduce_cfg):
    """
    vals: list[float] (probs or logits)
    reduce_cfg: dict with keys {type, ...}
    """
    rtype = reduce_cfg["type"]

    if rtype == "logit_mean":
        return float(_sigmoid(np.mean(vals)))

    if rtype == "prob_mean":
        return float(np.mean(vals))

    if rtype == "prob_max":
        return float(np.max(vals))

    if rtype == "median":
        return float(np.median(vals))

    if rtype == "topk_mean":
        k = reduce_cfg.get("topk", 0.25)
        k = max(1, int(len(vals) * k))
        topk_vals = np.sort(vals)[-k:]
        return float(np.mean(topk_vals))

    raise ValueError(f"Unknown video_reduce type: {rtype}")

def smooth_grad(model, video, n_samples=20, noise_std=0.02):
    grads = []
    for _ in range(n_samples):
        noise = torch.randn_like(video) * noise_std
        sal = input_gradient_saliency(model, video + noise)
        grads.append(sal)
    return torch.mean(torch.stack(grads), dim=0)
# ------------------ Evaluating one Epoch -----------------------
@torch.no_grad()
def evaluate_model(
    model, dataloader, device, loss_fn=compute_multitask_loss,
    task_info=None, loss_weights=None, threshold=0.5,
    inverse_transform=False, reg_pipeline=None, regression_cols=None,
    agg="mean",
    metric_level="clip",
    video_key="video_name",
    video_reduce_cfg=None,
    compute_saliency=False,
    saliency_out_dir=None,
    max_saliency_batches=3
 ):
    model.eval()
    losses = []

    # ---- Prepare saliency output directories if needed ---
    if compute_saliency and saliency_out_dir is not None:
        saliency_out_dir = Path(saliency_out_dir)
        saliency_out_dir.mkdir(parents=True, exist_ok=True)

    if compute_saliency and saliency_out_dir is not None:
        saliency_only_dir = saliency_out_dir / "heatmaps"
        overlay_dir = saliency_out_dir / "overlays"

        saliency_only_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

    if video_reduce_cfg is None:
        video_reduce_cfg = {"type": "prob_mean"}

    all_targets, all_probs, all_preds = defaultdict(list), defaultdict(list), defaultdict(list)
    vid_prob_buckets = defaultdict(lambda: defaultdict(list))
    vid_true_bucket = defaultdict(dict)

    for batch in dataloader:
        inputs, targets, meta = to_device_batch(batch, device, model.use_video, model.use_tabular)
        outputs = model(inputs["video"], inputs["tabular"] if model.use_tabular else None)

        # ---- Compute saliency maps for first few batches if enabled ----
        if compute_saliency and len(losses) < max_saliency_batches:
            with torch.enable_grad():
                sal = integrated_gradients_video(
                    model, inputs["video"],
                    task_name="Rythm_M24",
                    steps=16,
                    baseline="blur",
                    chunk_size=2,
                    normalize="per_frame",
                )

            video_np = inputs["video"].detach().cpu().numpy()

            for i in range(sal.shape[0]):
                vid = meta["video_name"][i]

                frames = video_np[i]  # (C,T,H,W)
                if frames.shape[0] == 1:
                    frames = np.repeat(frames, 3, axis=0)
                frames = np.transpose(frames, (1, 2, 3, 0))  # (T,H,W,3)

                T = sal.shape[1]
                frame_stride = 1
                frame_ids = list(range(0, T, frame_stride))

                for t in frame_ids:
                    sal_t = sal[i, t].cpu().numpy()          # (H,W) in [0,1]

                    heatmap = saliency_to_heatmap(sal_t)

                    # --- save pure heatmap ---
                    heatmap_path = saliency_only_dir / f"{vid}_t{t}.png"
                    cv2.imwrite(
                        str(heatmap_path),
                        cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
                    )

                    # --- optional overlay ---
                    overlay = overlay_heatmap(frames[t], heatmap, alpha=0.50)
                    overlay_path = overlay_dir / f"{vid}_t{t}.png"
                    cv2.imwrite(
                        str(overlay_path),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                    )

        loss, _ = loss_fn(outputs, targets, task_info, loss_weights, agg=agg)
        losses.append(loss.item())

        if metric_level == "video":
            if meta is None or video_key not in meta:
                raise RuntimeError(
                    f"metric_level='video' requires meta['{video_key}'] in each batch."
                )
            video_names = meta[video_key]  
            if video_names is None:
                raise RuntimeError(
                    f"meta['{video_key}'] is None. Check Dataset/Collate keys (expected 'video_name')."
                )

        for task, y_true in targets.items():
            y_true_np = y_true.cpu().numpy().ravel()
            y_out = outputs[task]

            # ---- BINARY ----
            if task_info[task] == "binary":
                if y_out.shape[-1] == 2:
                    logits = (y_out[..., 1] - y_out[..., 0]).cpu().numpy().ravel()
                else:
                    logits = y_out.cpu().numpy().ravel()

                prob = 1 / (1 + np.exp(-logits))

                if metric_level == "clip":
                    pred = (prob >= threshold).astype(float)
                    all_targets[task].extend(y_true_np.tolist())
                    all_probs[task].extend(prob.tolist())
                    all_preds[task].extend(pred.tolist())
                else:
                    for i, vid in enumerate(video_names):
                        t = float(y_true_np[i])
                        if task in vid_true_bucket[vid] and vid_true_bucket[vid][task] != t:
                            raise RuntimeError(f"Inconsistent label for video {vid}")
                        vid_true_bucket[vid][task] = t
                        vid_prob_buckets[vid][task].append(float(prob[i]))

            # ---- REGRESSION ----
            else:
                pred = y_out.cpu().numpy().ravel()
                if metric_level == "clip":
                    all_targets[task].extend(y_true_np.tolist())
                    all_preds[task].extend(pred.tolist())
                else:
                    for i, vid in enumerate(video_names):
                        vid_true_bucket[vid][task] = float(y_true_np[i])
                        vid_prob_buckets[vid][task].append(float(pred[i]))

    overall_stats = {"loss": float(np.mean(losses))}
    per_task_stats = {}

    # -------- VIDEO LEVEL METRICS --------
    if metric_level == "video":
        for task, ttype in task_info.items():
            y_true, y_score = [], []

            for vid, tasks in vid_prob_buckets.items():
                if task not in tasks:
                    continue
                y_true.append(vid_true_bucket[vid][task])
                y_score.append(reduce_video_scores(tasks[task], video_reduce_cfg))

            y_true = np.array(y_true)
            y_score = np.array(y_score)

            stats = {}
            if ttype == "binary":
                pred = (y_score >= threshold).astype(float)
                stats["auroc"] = roc_auc_score(y_true, y_score)
                stats["auprc"] = average_precision_score(y_true, y_score)
                stats["accuracy"] = accuracy_score(y_true, pred)
                stats["f1"] = f1_score(y_true, pred, zero_division=0)
              
            else:
                stats["mae"] = mean_absolute_error(y_true, y_score)

            per_task_stats[task] = stats

      
        for vid in list(vid_prob_buckets.keys())[:5]:
            vals = vid_prob_buckets[vid][task]
            logit_v = np.mean(vals)
            prob_v = 1 / (1 + np.exp(-logit_v))
            y_v = vid_true_bucket[vid][task]
           
    # -------- CLIP LEVEL (unchanged) --------
    else:
        print("\n[DEBUG] Computing clip-level metrics")
        for task in all_targets:
            y_true = np.array(all_targets[task])
            prob = np.array(all_probs[task])
            pred = np.array(all_preds[task])
            per_task_stats[task] = {
                "auroc": roc_auc_score(y_true, prob),
                "auprc": average_precision_score(y_true, prob),
                "accuracy": accuracy_score(y_true, pred),
                "f1": f1_score(y_true, pred, zero_division=0),
            }

    # -------- OVERALL --------
    # overall_stats["auroc"] = np.mean([v["auroc"] for v in per_task_stats.values() if "auroc" in v])
    # overall_stats["accuracy"] = np.mean([v["accuracy"] for v in per_task_stats.values()])
    # overall_stats["f1"] = np.mean([v["f1"] for v in per_task_stats.values()])
  
    overall_stats["classification"] = {
        "auroc": np.mean([v["auroc"] for v in per_task_stats.values() if "auroc" in v]),
        "accuracy": np.mean([v["accuracy"] for v in per_task_stats.values() if "accuracy" in v]),
        "f1": np.mean([v["f1"] for v in per_task_stats.values() if "f1" in v]),
    }

    overall_stats["regression"] = {
        "mae": np.mean([v["mae"] for v in per_task_stats.values() if "mae" in v])
    }

    return overall_stats, per_task_stats

# ------------------ main Function ------------------------------
def run_eval(cfg, model, loader, task_info, loss_weights, device, split: str, inverse_transform=False ,reg_pipeline=None, regression_cols=None):
    out_dir = Path(cfg["train"].get("out_dir", "runs/exp"))
    ckpt = cfg.get("eval", {}).get("checkpoint", str(out_dir / "best.pt"))
    ckpt_path = Path(ckpt)

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"⚠️ Checkpoint not found at {ckpt_path}. Evaluating current weights.")
    overall, per_task = evaluate_model(
        model, loader, device, compute_multitask_loss, task_info=task_info, loss_weights=loss_weights, threshold=cfg.get("eval", {}).get("threshold", 0.5), 
        inverse_transform=inverse_transform, reg_pipeline=reg_pipeline, regression_cols=regression_cols, metric_level=cfg.get("eval", {}).get("metric_level", "clip"), video_key="video_name",
        video_reduce_cfg=cfg["eval"].get("video_reduce", None), compute_saliency=cfg["eval"].get("saliency", False), saliency_out_dir = Path(cfg["train"]["out_dir"]) / "saliency" ,max_saliency_batches=cfg["eval"].get("saliency_batches", 1))
    # print(f"Metrices are reported in video level")
    print(f"\n{split.upper()} OVERALL:", overall)
    print(f"\n{split.upper()} PER-TASK:", json.dumps(per_task, indent=2))

    metrics_dir = out_dir / "metrics"
    save_metrics(metrics_dir, split, overall, per_task)

# ---------------- Main Function ----------------
def main_from_config(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.get("cpu", False) else "cpu")

    if cfg.get("wandb", {}).get("enable", False):
        wandb.init(
            project=cfg["wandb"].get("project", "echo-project"),
            name=cfg["wandb"].get("name", None),
            config=cfg,
        )

    # build Dataset
    loaders, task_info , regression_cols, reg_pipeline = build_loaders_from_cfg(cfg)

    # Build Model
    model = create_model_from_cfg(cfg["model"], task_info).to(device)

    # Sanity Check
    if cfg["sanity_check"]:
        sanity_check(model, loaders, device)
    
    # Get Training Options
    loss_weights = cfg["train"].get("loss_weights", None)
    phase = str(cfg.get("phase", "train")).lower()
    use_tqdm = cfg["train"].get("show_tqdm", False)

    if phase == "train":
        print("=============== Starting Training ===============")
        if loaders["train"] is None:
            raise RuntimeError("No TRAIN split available.")
        best_ckpt = run_train(cfg, model, loaders, task_info, loss_weights, device , use_tqdm)
        # optional test right after training
        print("=============== Starting Testing ===============")
        if cfg["train"].get("eval_after_train", True) and loaders["test"] is not None:
            cfg.setdefault("eval", {})["checkpoint"] = str(best_ckpt)
            run_eval(cfg, model, loaders["test"], task_info, loss_weights, device, split="test")

    elif phase == "val":
        print("=============== Starting Validation ===============")
        if loaders["val"] is None:
            raise RuntimeError("No VAL split available.")
        run_eval(cfg, model, loaders["val"], task_info, loss_weights, device,
         split="val", inverse_transform=False,
         reg_pipeline=reg_pipeline, regression_cols=regression_cols)

    elif phase == "test":
        print("=============== Starting Testing ===============")
        if loaders["test"] is None:
            raise RuntimeError("No TEST split available.")
        run_eval(cfg, model, loaders["test"], task_info, loss_weights, device,
         split="test", inverse_transform=True,
         reg_pipeline=reg_pipeline, regression_cols=regression_cols)

    else:
        raise ValueError(f"Unsupported phase: {phase}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("Config-driven Echo Trainer")
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()
    main_from_config(args.config)