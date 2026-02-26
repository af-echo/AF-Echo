import torch
import numpy as np
import cv2
import torch
import torch.nn.functional as F

def input_gradient_saliency(model, video, class_idx=None):
    """
    video: (B, C, T, H, W)
    returns: saliency (B, T, H, W)
    """
    model.eval()

    video = video.detach().clone()
    video.requires_grad_(True)

    model.zero_grad(set_to_none=True)

    outputs = model(video, None)   # video-only
    y_out = outputs["Rythm_M24"]    # <-- your binary task name

    # binary logit
    if y_out.shape[-1] == 2:
        score = y_out[:, 1] - y_out[:, 0]
    else:
        score = y_out[:, 0]

    score.sum().backward()

    grad = video.grad               # (B,C,T,H,W)
    sal = grad.abs().mean(dim=1)    # (B,T,H,W)

    # normalize per video
    # sal -= sal.amin(dim=(1,2,3), keepdim=True)
    # sal /= (sal.amax(dim=(1,2,3), keepdim=True) + 1e-6)
    # robust normalization (per video)
    flat = sal.view(sal.size(0), -1)
    p_low  = torch.quantile(flat, 0.05, dim=1).view(-1,1,1,1)
    p_high = torch.quantile(flat, 0.95, dim=1).view(-1,1,1,1)

    sal = torch.clamp(sal, p_low, p_high)
    sal = (sal - p_low) / (p_high - p_low + 1e-6)

    return sal.detach()



def saliency_to_heatmap(sal):
    """
    sal: (H, W) in [0,1]
    returns: (H, W, 3) RGB heatmap
    """
    sal_uint8 = np.uint8(255 * sal)
    heatmap = cv2.applyColorMap(sal_uint8, cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def overlay_heatmap(frame, heatmap, alpha=0.3):
    """
    frame: (H,W,3) RGB
        - uint8 [0,255] OR
        - float [0,1]
    heatmap: (H,W,3) uint8 RGB
    """

    # ---- Fix frame intensity ----
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 1)
        frame = (frame * 255.0).astype(np.uint8)

    # ---- Safety: heatmap to uint8 ----
    if heatmap.dtype != np.uint8:
        heatmap = heatmap.astype(np.uint8)

    # ---- Blend ----
    overlay = cv2.addWeighted(
        frame, 1.0 - alpha,
        heatmap, alpha,
        0
    )

    return overlay



import torch

def normalize_saliency_per_frame(sal, q_low=0.01, q_high=0.99, eps=1e-6):
    # sal: (B,T,H,W) torch
    B,T,H,W = sal.shape
    sal2 = sal.view(B*T, -1)
    lo = torch.quantile(sal2, q_low, dim=1).view(B,T,1,1)
    hi = torch.quantile(sal2, q_high, dim=1).view(B,T,1,1)
    sal = torch.clamp(sal, lo, hi)
    sal = (sal - lo) / (hi - lo + eps)
    return sal



def integrated_gradients_video(
    model,
    video,                      # (B,C,T,H,W) float tensor on device
    task_name="Rythm_M24",
    steps=16,                   # 16–32 typically looks better than 8
    baseline="zeros",           # "zeros" | "mean" | "blur"
    blur_kernel=(1, 9, 9),      # (t,h,w) for blur baseline; t=1 recommended
    blur_sigma=3.0,
    chunk_size=2,               # number of alpha steps processed at once
    normalize="per_frame",      # "per_frame" | "per_video" | "none"
    q_low=0.01,                 # robust contrast (1%–99%)
    q_high=0.99,
    eps=1e-6,
):
    """
    Returns:
        sal: (B,T,H,W) float in [0,1] (unless normalize="none")
    Notes:
        - Make sure `video` is the *same tensor* your model normally receives
          (i.e., already preprocessed/normalized the same way).
        - This function computes IG for the logit of `task_name`:
            - if output has 2 logits: uses (logit1 - logit0)
            - else: uses out[:,0]
    """
    model.eval()

    x = video.detach()
    B, C, T, H, W = x.shape

    # ----------- Baseline -----------
    if baseline == "zeros":
        x0 = torch.zeros_like(x)

    elif baseline == "mean":
        # per-sample temporal mean baseline (keeps spatial structure)
        x0 = x.mean(dim=2, keepdim=True).expand_as(x)

    elif baseline == "blur":
        # blur each frame spatially; keep temporal axis intact
        # We'll apply a depthwise-ish blur via conv3d with groups=C
        kt, kh, kw = blur_kernel
        if kt != 1:
            raise ValueError("For ultrasound, recommend kt=1 (no temporal blur).")

        # Build Gaussian kernel (kh x kw), then expand to (C,1,1,kh,kw) for groups=C conv3d
        # Do it in torch for device safety
        yy, xx = torch.meshgrid(
            torch.arange(kh, device=x.device) - (kh - 1) / 2,
            torch.arange(kw, device=x.device) - (kw - 1) / 2,
            indexing="ij",
        )
        kernel2d = torch.exp(-(xx**2 + yy**2) / (2 * blur_sigma**2))
        kernel2d = kernel2d / kernel2d.sum()
        kernel3d = kernel2d.view(1, 1, 1, kh, kw)  # (1,1,1,kh,kw)
        weight = kernel3d.repeat(C, 1, 1, 1, 1)    # (C,1,1,kh,kw)

        pad_h = kh // 2
        pad_w = kw // 2

        x_pad = F.pad(x, (pad_w, pad_w, pad_h, pad_h, 0, 0), mode="reflect")  # pad W,H only
        x0 = F.conv3d(x_pad, weight, bias=None, stride=1, padding=0, groups=C)

    else:
        raise ValueError("baseline must be one of: 'zeros', 'mean', 'blur'")

    # ----------- Alphas -----------
    # Use steps points in (0,1], exclude 0 to avoid baseline gradient issues
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=x.device)[1:]

    total_grad = torch.zeros_like(x)

    # ----------- IG integration loop (chunked over alpha) -----------
    for i in range(0, len(alphas), chunk_size):
        a = alphas[i:i + chunk_size]  # (m,)
        m = a.numel()

        # shape to broadcast: (m,1,1,1,1,1)
        a = a.view(m, 1, 1, 1, 1, 1)

        # Interpolate between baseline and input
        # Detach to make leaf, then require grads
        x_step = (x0.unsqueeze(0) + a * (x.unsqueeze(0) - x0.unsqueeze(0))).detach()
        x_step.requires_grad_(True)

        # Flatten alpha dimension into batch: (m*B, C, T, H, W)
        x_step_flat = x_step.view(m * B, C, T, H, W)

        # Forward
        out = model(x_step_flat, None)[task_name]  # (mB,2) or (mB,1) or (mB,)
        if out.ndim == 1:
            score = out
        else:
            if out.shape[-1] == 2:
                score = out[:, 1] - out[:, 0]
            else:
                score = out[:, 0]

        # Backward: d(score.sum)/d(x_step_flat)
        grad_flat = torch.autograd.grad(
            outputs=score.sum(),
            inputs=x_step_flat,
            create_graph=False,
            retain_graph=False,
            allow_unused=False
        )[0]

        if grad_flat is None:
            raise RuntimeError(
                "Integrated Gradients: grad is None. "
                "Gradient path is broken (e.g., detach inside model)."
            )

        # Unflatten: (m,B,C,T,H,W) then sum over m
        grad = grad_flat.view(m, B, C, T, H, W)
        total_grad += grad.sum(dim=0)

        # Free
        del x_step, x_step_flat, out, score, grad_flat, grad

    avg_grad = total_grad / len(alphas)      # (B,C,T,H,W)
    ig = (x - x0) * avg_grad                 # (B,C,T,H,W)

    # Saliency as channel-aggregated magnitude
    sal = ig.abs().mean(dim=1)               # (B,T,H,W)

    # ----------- Robust normalization for contrast -----------
    if normalize == "none":
        return sal.detach()

    elif normalize == "per_video":
        # normalize across T*H*W per sample
        flat = sal.view(B, -1)
        lo = torch.quantile(flat, q_low, dim=1).view(B, 1, 1, 1)
        hi = torch.quantile(flat, q_high, dim=1).view(B, 1, 1, 1)
        sal = torch.clamp(sal, lo, hi)
        sal = (sal - lo) / (hi - lo + eps)
        return sal.detach()

    elif normalize == "per_frame":
        # normalize each frame separately (best for avoiding low contrast + temporal flicker)
        flat = sal.view(B * T, -1)
        lo = torch.quantile(flat, q_low, dim=1).view(B, T, 1, 1)
        hi = torch.quantile(flat, q_high, dim=1).view(B, T, 1, 1)
        sal = torch.clamp(sal, lo, hi)
        sal = (sal - lo) / (hi - lo + eps)
        return sal.detach()

    else:
        raise ValueError("normalize must be one of: 'per_frame', 'per_video', 'none'")