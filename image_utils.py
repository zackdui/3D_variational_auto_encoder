# This file contains utils for saving images to disk and wandb
import torch.nn.functional as F
from typing import Dict, Tuple, Literal, Optional
import torch.distributed as dist
from PIL import Image
import logging
import torch
import tempfile
import imageio.v2 as imageio
import numpy as np
import os

EvalMode = Literal["full", "patch"]

def safe_delete(path):
    try:
        os.remove(path)
    except OSError:
        pass

def prepare_for_wandb(slice_2d):
    """
    slice_2d: torch.Tensor or np.ndarray, shape (H, W) or (1, H, W)
              values expected in [-1, 1]
    returns: np.ndarray uint8, shape (H, W)
    """
    if hasattr(slice_2d, "detach"):  # torch.Tensor
        slice_2d = slice_2d.detach().cpu().numpy()

    slice_2d = np.squeeze(slice_2d)          # drop channel if (1, H, W)
    slice_2d = np.clip(slice_2d, -1.0, 1.0)  # enforce range
    slice_2d = (slice_2d + 1.0) / 2.0        # [-1,1] -> [0,1]
    slice_2d = (slice_2d * 255.0).round().astype("uint8")
    return slice_2d

def volume_to_gif_frames(volume_3d, every_n: int = 1):
    """
    volume_3d: torch.Tensor or np.ndarray, shape (D, H, W)
               values in [-1, 1]
    every_n: use every n-th slice to keep GIF small
    returns: list of uint8 (H, W) frames
    """
    if hasattr(volume_3d, "detach"):
        volume_3d = volume_3d.detach().cpu().numpy()
    D = volume_3d.shape[0]
    frames = []
    for d in range(0, D, every_n):
        frame = prepare_for_wandb(volume_3d[d])
        frames.append(frame)
    return frames

def save_gif(frames, fps: int = 10) -> str:
    """
    frames: list of (H, W) or (H, W, 3) uint8 arrays
    returns: path to a temporary .gif file
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    tmp.close()
    imageio.mimsave(tmp.name, frames, fps=fps)
    return tmp.name

def save_mp4(frames, fps=10):
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    imageio.mimsave(tmp.name, frames, fps=fps)  # imageio automatically picks mp4 writer
    return tmp.name

def save_side_by_side_slices(
    x_vol: torch.Tensor,          # (C, D, H, W) in [-1, 1]
    recon_vol: torch.Tensor,      # (C, D, H, W), may exceed [-1, 1]
    save_dir: str,
    num_slices: int = 10,       # how many slices to save
    start_idx: int = 80,           # first slice index to save
    merge_to_grayscale: bool = False,
    logger: logging.Logger | None =None,
) -> None:
    """
    Save 2D slices side-by-side (original | reconstructed) as PNG images.

    Assumptions:
        - x_vol is (C, D, H, W) with pixel intensities in [-1, 1]
        - recon_vol is same shape but may exceed [-1, 1] → we clip
        - C can be 1 (CT) or 3 (RGB)
    
    Options:
        - merge_to_grayscale=True collapses RGB → 1 channel.
        - num_slices: number of slices to save
        - start_idx: beginning slice index

    Output:
        save_dir/
            slice_000.png
            slice_001.png
            ...
    """
    logger = logger or logging.getLogger(__name__)

    os.makedirs(save_dir, exist_ok=True)

    if x_vol.shape != recon_vol.shape:
        raise ValueError(f"Shape mismatch: {x_vol.shape} vs {recon_vol.shape}")

    C, D, H, W = x_vol.shape

    # Reduce to grayscale if desired
    if merge_to_grayscale and C > 1:
        x_vol     = x_vol.mean(dim=0, keepdim=True)
        recon_vol = recon_vol.mean(dim=0, keepdim=True)
        C = 1

    # Clip reconstruction to [-1, 1]
    recon_vol = recon_vol.clamp(-1, 1)

    # How many slices to save
    end_idx = D if num_slices is None else min(D, start_idx + num_slices)

    def to_uint8(img):
        """(-1,1) → [0,255] uint8 tensor."""
        img = (img + 1) * 0.5       # map → [0,1]
        img = img.clamp(0, 1)
        return (img * 255).byte()

    # Loop over slices
    for i in range(start_idx, end_idx):
        x_slice = x_vol[:, i]      # (C, H, W)
        r_slice = recon_vol[:, i]

        x_u8 = to_uint8(x_slice)
        r_u8 = to_uint8(r_slice)

        # Concatenate horizontally: (C, H, 2W)
        paired = torch.cat([x_u8, r_u8], dim=-1)

        # Convert to numpy HWC
        if C == 1:
            img = paired.squeeze(0).numpy()       # (H, 2W)
            pil = Image.fromarray(img, mode="L")
        else:
            img = paired.permute(1, 2, 0).numpy() # (H, 2W, 3)
            pil = Image.fromarray(img, mode="RGB")

        pil.save(os.path.join(save_dir, f"slice_{i:04d}.png"))

    logger.info(f"[eval] Saved slices {start_idx} to {end_idx-1} → {save_dir}")

def _make_hann_window_3d(patch_size: Tuple[int, int, int], device):
    """3D Hann window with values in [0,1], shaped (1,1,D,H,W)."""
    pd, ph, pw = patch_size
    wz = torch.hann_window(pd, periodic=False, device=device)
    wy = torch.hann_window(ph, periodic=False, device=device)
    wx = torch.hann_window(pw, periodic=False, device=device)

    # outer product → (D,H,W)
    w = wz.view(pd, 1, 1) * wy.view(1, ph, 1) * wx.view(1, 1, pw)
    w = w / w.max()  # normalize max to 1
    return w.view(1, 1, pd, ph, pw)  # (1,1,D,H,W)

@torch.no_grad()
def sliding_window_reconstruct(
    model: torch.nn.Module,
    x: torch.Tensor,
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    device: Optional[torch.device] = None,
    use_blending: bool = True,
) -> torch.Tensor:
    """
    Reconstruct a full 3D volume using a patch-based model via sliding windows.

    Args:
        model: patch-based model taking inputs of shape (B, C, d, h, w).
        x: input volume, shape (B, C, D, H, W).
        patch_size: (d, h, w) of patches used during training.
        stride: sliding window stride in each dim (dz, dy, dx).
        device: device to run model on; if None, uses x.device.

    Returns:
        recon: reconstructed volume, shape (B, C, D, H, W).
    """
    if device is None:
        device = x.device

    model_was_training = model.training
    model.eval()

    B, C, D, H, W = x.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    recon = torch.zeros_like(x, device=device)
    weight = torch.zeros_like(x, device=device)

    if use_blending:
        w_patch = _make_hann_window_3d(patch_size, device=device)  # (1,1,pd,ph,pw)
    else:
        w_patch = None

    # Helper to generate start indices that always hit the end
    def make_starts(L, p, s):
        if L <= p:
            return [0]
        starts = list(range(0, L - p + 1, s))
        if starts[-1] != L - p:
            starts.append(L - p)
        return starts

    z_starts = make_starts(D, pd, sd)
    y_starts = make_starts(H, ph, sh)
    x_starts = make_starts(W, pw, sw)

    for z0 in z_starts:
        z1 = z0 + pd
        for y0 in y_starts:
            y1 = y0 + ph
            for x0 in x_starts:
                x1 = x0 + pw

                # (B, C, d, h, w)
                patch = x[:, :, z0:z1, y0:y1, x0:x1].to(device)

                out_patch = model(patch)  # assume same shape as patch
                if out_patch.shape != patch.shape:
                    raise RuntimeError(
                        f"Patch model output shape {out_patch.shape} != input patch shape {patch.shape}"
                    )

                if use_blending:
                    # broadcast w_patch to (B,C,...) automatically
                    weighted = out_patch * w_patch
                    recon[:, :, z0:z1, y0:y1, x0:x1] += weighted
                    weight[:, :, z0:z1, y0:y1, x0:x1] += w_patch
                else:
                    recon[:, :, z0:z1, y0:y1, x0:x1] += out_patch
                    weight[:, :, z0:z1, y0:y1, x0:x1] += 1.0
                # recon[:, :, z0:z1, y0:y1, x0:x1] += out_patch
                # weight[:, :, z0:z1, y0:y1, x0:x1] += 1.0

    # Avoid division by zero (shouldn't happen if windows cover full vol)
    recon = recon / weight.clamp_min(1e-8)

    if model_was_training:
        model.train()

    return recon


@torch.no_grad()
def reconstruct_volume(
    model: torch.nn.Module,
    x: torch.Tensor,
    mode: EvalMode,
    patch_size: Optional[Tuple[int, int, int]] = None,
    stride: Optional[Tuple[int, int, int]] = None,
    device: Optional[torch.device] = None,
    use_blending: bool = True,
) -> torch.Tensor:
    """
    Unified entry point to reconstruct full volumes with either:
      - a full-volume model ("full"), or
      - a patch-based model ("patch") using sliding windows.

    Args:
        model: nn.Module.
        x: input volume, shape (B, C, D, H, W).
        mode: "full" or "patch".
        patch_size: required if mode == "patch".
        stride: required if mode == "patch".
        device: device to run model on.

    Returns:
        recon: same shape as x.
    """
    if device is None:
        device = x.device

    if mode == "full":
        model_was_training = model.training
        model.eval()
        out = model(x.to(device))
        if out.shape != x.shape:
            raise RuntimeError(
                f"Full model output shape {out.shape} != input shape {x.shape}"
            )
        if model_was_training:
            model.train()
        return out

    elif mode == "patch":
        assert patch_size is not None and stride is not None, \
            "patch_size and stride must be provided for patch mode."
        return sliding_window_reconstruct(model, x, patch_size, stride, device=device, use_blending=use_blending)

    else:
        raise ValueError(f"Unknown mode: {mode}")
