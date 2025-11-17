# This file will be used to store the evalutation metrics
import os
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Literal, Optional
import torch.distributed as dist
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

