# This file will be used to store the evalutation metrics
import os
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Literal, Optional
import torch.distributed as dist
# from PIL import Image
import logging
from tqdm import tqdm
import wandb
import numpy as np

from image_utils import (reconstruct_volume, 
                         safe_delete, 
                         save_gif, 
                         save_side_by_side_slices, 
                         save_mp4, 
                         prepare_for_wandb,
                         volume_to_gif_frames)

EvalMode = Literal["full", "patch"]


# def save_side_by_side_slices(
#     x_vol: torch.Tensor,          # (C, D, H, W) in [-1, 1]
#     recon_vol: torch.Tensor,      # (C, D, H, W), may exceed [-1, 1]
#     save_dir: str,
#     num_slices: int = 10,       # how many slices to save
#     start_idx: int = 80,           # first slice index to save
#     merge_to_grayscale: bool = False,
#     logger: logging.Logger | None =None,
# ) -> None:
#     """
#     Save 2D slices side-by-side (original | reconstructed) as PNG images.

#     Assumptions:
#         - x_vol is (C, D, H, W) with pixel intensities in [-1, 1]
#         - recon_vol is same shape but may exceed [-1, 1] → we clip
#         - C can be 1 (CT) or 3 (RGB)
    
#     Options:
#         - merge_to_grayscale=True collapses RGB → 1 channel.
#         - num_slices: number of slices to save
#         - start_idx: beginning slice index

#     Output:
#         save_dir/
#             slice_000.png
#             slice_001.png
#             ...
#     """
#     logger = logger or logging.getLogger(__name__)

#     os.makedirs(save_dir, exist_ok=True)

#     if x_vol.shape != recon_vol.shape:
#         raise ValueError(f"Shape mismatch: {x_vol.shape} vs {recon_vol.shape}")

#     C, D, H, W = x_vol.shape

#     # Reduce to grayscale if desired
#     if merge_to_grayscale and C > 1:
#         x_vol     = x_vol.mean(dim=0, keepdim=True)
#         recon_vol = recon_vol.mean(dim=0, keepdim=True)
#         C = 1

#     # Clip reconstruction to [-1, 1]
#     recon_vol = recon_vol.clamp(-1, 1)

#     # How many slices to save
#     end_idx = D if num_slices is None else min(D, start_idx + num_slices)

#     def to_uint8(img):
#         """(-1,1) → [0,255] uint8 tensor."""
#         img = (img + 1) * 0.5       # map → [0,1]
#         img = img.clamp(0, 1)
#         return (img * 255).byte()

#     # Loop over slices
#     for i in range(start_idx, end_idx):
#         x_slice = x_vol[:, i]      # (C, H, W)
#         r_slice = recon_vol[:, i]

#         x_u8 = to_uint8(x_slice)
#         r_u8 = to_uint8(r_slice)

#         # Concatenate horizontally: (C, H, 2W)
#         paired = torch.cat([x_u8, r_u8], dim=-1)

#         # Convert to numpy HWC
#         if C == 1:
#             img = paired.squeeze(0).numpy()       # (H, 2W)
#             pil = Image.fromarray(img, mode="L")
#         else:
#             img = paired.permute(1, 2, 0).numpy() # (H, 2W, 3)
#             pil = Image.fromarray(img, mode="RGB")

#         pil.save(os.path.join(save_dir, f"slice_{i:04d}.png"))

#     logger.info(f"[eval] Saved slices {start_idx} to {end_idx-1} → {save_dir}")

# def _make_hann_window_3d(patch_size: Tuple[int, int, int], device):
#     """3D Hann window with values in [0,1], shaped (1,1,D,H,W)."""
#     pd, ph, pw = patch_size
#     wz = torch.hann_window(pd, periodic=False, device=device)
#     wy = torch.hann_window(ph, periodic=False, device=device)
#     wx = torch.hann_window(pw, periodic=False, device=device)

#     # outer product → (D,H,W)
#     w = wz.view(pd, 1, 1) * wy.view(1, ph, 1) * wx.view(1, 1, pw)
#     w = w / w.max()  # normalize max to 1
#     return w.view(1, 1, pd, ph, pw)  # (1,1,D,H,W)

# @torch.no_grad()
# def sliding_window_reconstruct(
#     model: torch.nn.Module,
#     x: torch.Tensor,
#     patch_size: Tuple[int, int, int],
#     stride: Tuple[int, int, int],
#     device: Optional[torch.device] = None,
#     use_blending: bool = True,
# ) -> torch.Tensor:
#     """
#     Reconstruct a full 3D volume using a patch-based model via sliding windows.

#     Args:
#         model: patch-based model taking inputs of shape (B, C, d, h, w).
#         x: input volume, shape (B, C, D, H, W).
#         patch_size: (d, h, w) of patches used during training.
#         stride: sliding window stride in each dim (dz, dy, dx).
#         device: device to run model on; if None, uses x.device.

#     Returns:
#         recon: reconstructed volume, shape (B, C, D, H, W).
#     """
#     if device is None:
#         device = x.device

#     model_was_training = model.training
#     model.eval()

#     B, C, D, H, W = x.shape
#     pd, ph, pw = patch_size
#     sd, sh, sw = stride

#     recon = torch.zeros_like(x, device=device)
#     weight = torch.zeros_like(x, device=device)

#     if use_blending:
#         w_patch = _make_hann_window_3d(patch_size, device=device)  # (1,1,pd,ph,pw)
#     else:
#         w_patch = None

#     # Helper to generate start indices that always hit the end
#     def make_starts(L, p, s):
#         if L <= p:
#             return [0]
#         starts = list(range(0, L - p + 1, s))
#         if starts[-1] != L - p:
#             starts.append(L - p)
#         return starts

#     z_starts = make_starts(D, pd, sd)
#     y_starts = make_starts(H, ph, sh)
#     x_starts = make_starts(W, pw, sw)

#     for z0 in z_starts:
#         z1 = z0 + pd
#         for y0 in y_starts:
#             y1 = y0 + ph
#             for x0 in x_starts:
#                 x1 = x0 + pw

#                 # (B, C, d, h, w)
#                 patch = x[:, :, z0:z1, y0:y1, x0:x1].to(device)

#                 out_patch = model(patch)  # assume same shape as patch
#                 if out_patch.shape != patch.shape:
#                     raise RuntimeError(
#                         f"Patch model output shape {out_patch.shape} != input patch shape {patch.shape}"
#                     )

#                 if use_blending:
#                     # broadcast w_patch to (B,C,...) automatically
#                     weighted = out_patch * w_patch
#                     recon[:, :, z0:z1, y0:y1, x0:x1] += weighted
#                     weight[:, :, z0:z1, y0:y1, x0:x1] += w_patch
#                 else:
#                     recon[:, :, z0:z1, y0:y1, x0:x1] += out_patch
#                     weight[:, :, z0:z1, y0:y1, x0:x1] += 1.0
#                 # recon[:, :, z0:z1, y0:y1, x0:x1] += out_patch
#                 # weight[:, :, z0:z1, y0:y1, x0:x1] += 1.0

#     # Avoid division by zero (shouldn't happen if windows cover full vol)
#     recon = recon / weight.clamp_min(1e-8)

#     if model_was_training:
#         model.train()

#     return recon


# @torch.no_grad()
# def reconstruct_volume(
#     model: torch.nn.Module,
#     x: torch.Tensor,
#     mode: EvalMode,
#     patch_size: Optional[Tuple[int, int, int]] = None,
#     stride: Optional[Tuple[int, int, int]] = None,
#     device: Optional[torch.device] = None,
#     use_blending: bool = True,
# ) -> torch.Tensor:
#     """
#     Unified entry point to reconstruct full volumes with either:
#       - a full-volume model ("full"), or
#       - a patch-based model ("patch") using sliding windows.

#     Args:
#         model: nn.Module.
#         x: input volume, shape (B, C, D, H, W).
#         mode: "full" or "patch".
#         patch_size: required if mode == "patch".
#         stride: required if mode == "patch".
#         device: device to run model on.

#     Returns:
#         recon: same shape as x.
#     """
#     if device is None:
#         device = x.device

#     if mode == "full":
#         model_was_training = model.training
#         model.eval()
#         out = model(x.to(device))
#         if out.shape != x.shape:
#             raise RuntimeError(
#                 f"Full model output shape {out.shape} != input shape {x.shape}"
#             )
#         if model_was_training:
#             model.train()
#         return out

#     elif mode == "patch":
#         assert patch_size is not None and stride is not None, \
#             "patch_size and stride must be provided for patch mode."
#         return sliding_window_reconstruct(model, x, patch_size, stride, device=device, use_blending=use_blending)

#     else:
#         raise ValueError(f"Unknown mode: {mode}")

@torch.no_grad()
def evaluate_model_on_full_volumes(
    model: torch.nn.Module,
    mode: EvalMode,
    dataloader,
    patch_size: Optional[Tuple[int, int, int]] = None,
    stride: Optional[Tuple[int, int, int]] = None,
    use_blending: bool = True,
    device: Optional[torch.device] = None,
    save_example_dir: Optional[str] = None,
    example_volume_index: int = 0,
    logger: logging.Logger | None =None,
    use_wandb: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Evaluate a single model on a test set of full 3D volumes.

    - Supports "full" and "patch" modes.
    - Optionally saves ONE example volume where each 2D slice is
      [original | reconstructed] as a .pt tensor.
    - use_blending if in patch mode this will use blending when reconstructing the full image

    Returns:
        {
            "per_volume_mse": tensor (N_volumes_local,)  # per-process if DDP
            "mean_mse": scalar tensor                   # global mean if DDP
        }
    """
    logger = logger or logging.getLogger(__name__)
    # DDP detection
    is_distributed = dist.is_available() and dist.is_initialized()
    if device is None:
        if is_distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_distributed:
        rank = dist.get_rank()
    else:
        rank = 0

    is_main_process = (rank == 0)

    per_batch_mses = []  # list of (B,) tensors on CPU

    global_vol_idx = 0
    example_saved = False

    desc = "Eval"
    
    iterator = tqdm(
        dataloader,
        desc=desc,
        disable=not is_main_process,
    )
    
    for step, batch in enumerate(iterator):
        if isinstance(batch, dict):
            x = batch["image"]
        else:
            x = batch
        x = x.to(device)  # (B, C, D, H, W)
        B = x.size(0)

        recon = reconstruct_volume(
            model,
            x,
            mode=mode,
            patch_size=patch_size,
            stride=stride,
            device=device,
            use_blending=use_blending,
        )

        mse_per_vol = F.mse_loss(recon, x, reduction="none")
        mse_per_vol = mse_per_vol.view(B, -1).mean(dim=1)  # (B,)
        per_batch_mses.append(mse_per_vol.cpu())

        # Save example only once, and only on rank 0 if DDP.
        if (
            save_example_dir is not None
            and not example_saved
            and rank == 0
        ):
            start_idx = global_vol_idx
            end_idx = global_vol_idx + B
            # Verify example volume index is in range
            if start_idx <= example_volume_index < end_idx:
                local_idx = example_volume_index - start_idx
                x_vol = x[local_idx].detach().cpu()         # (C, D, H, W)
                recon_vol = recon[local_idx].detach().cpu() # (C, D, H, W)

                save_side_by_side_slices(x_vol, 
                                         recon_vol,
                                         save_dir=save_example_dir,
                                         logger=logger)
                
                if use_wandb:
                    vol_in = x[0, 0]   # (D, H, W)
                    vol_out = recon[0, 0]

                    # ---- 2D middle slice images (what you already had) ----
                    mid = vol_in.shape[0] // 2
                    x_slice = prepare_for_wandb(vol_in[mid])
                    out_slice = prepare_for_wandb(vol_out[mid])

                    wandb.log({
                        "recon/inputs_mid_slice": wandb.Image(x_slice),
                        "recon/outputs_mid_slice": wandb.Image(out_slice),
                    })

                    # ---- 3D scrollable GIF: input volume ----
                    input_frames = volume_to_gif_frames(vol_in, every_n=2)  # every 2 slices, tweak as needed
                    input_gif_path = save_gif(input_frames, fps=10)
                    input_mp4_path = save_mp4(input_frames, fps=10)

                    # ---- 3D scrollable GIF: side-by-side (input | recon) ----
                    recon_frames = volume_to_gif_frames(vol_out, every_n=2)
                    # ensure same length
                    n_frames = min(len(input_frames), len(recon_frames))
                    side_by_side_frames = [
                        np.concatenate([input_frames[i], recon_frames[i]], axis=1)
                        for i in range(n_frames)
                    ]
                    side_gif_path = save_gif(side_by_side_frames, fps=10)
                    side_mp4_path = save_mp4(side_by_side_frames, fps=10)

                    wandb.log({
                        "recon/volume_scroll_input": wandb.Video(input_gif_path, fps=10, format="gif"),
                        "recon/volume_scroll_side_by_side": wandb.Video(side_gif_path, fps=10, format="gif"),
                    })

                    wandb.log({
                        "recon/volume_scroll_input_mp4": wandb.Video(input_mp4_path, fps=10, format="mp4"),
                        "recon/volume_scroll_side_by_side_mp4": wandb.Video(side_mp4_path, fps=10, format="mp4")
                    })

                    safe_delete(input_gif_path)
                    safe_delete(side_gif_path)
                    safe_delete(input_mp4_path)
                    safe_delete(side_mp4_path)
                    
                example_saved = True


        global_vol_idx += B

    # Local concatenation (per-process if DDP)
    per_volume_mse_local = torch.cat(per_batch_mses, dim=0)  # (N_local,)

    # Compute mean MSE, with proper aggregation across ranks if DDP
    local_sum = per_volume_mse_local.sum().to(device)
    local_count = torch.tensor(
        per_volume_mse_local.numel(), dtype=torch.float32, device=device
    )

    if is_distributed:
        tensor = torch.stack([local_sum, local_count])  # (2,)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        global_sum, global_count = tensor[0], tensor[1]
        mean_mse = (global_sum / global_count).cpu()
    else:
        mean_mse = (local_sum / local_count).cpu()

    return {
        "per_volume_mse": per_volume_mse_local,  # local-only if DDP
        "mean_mse": mean_mse,                    # global if DDP
    }

# Example Usage
if __name__ == "__main__":
    # results = evaluate_model_on_full_volumes(
    #     model=my_vae,
    #     mode="patch",  # or "full"
    #     dataloader=test_loader,
    #     patch_size=(64, 128, 128),
    #     stride=(32, 64, 64),
    #     device=None,
    #     save_example_dir="./eval_examples",
    #     example_volume_index=0,
    # )

    # print("Mean MSE:", results["mean_mse"].item())
    pass

