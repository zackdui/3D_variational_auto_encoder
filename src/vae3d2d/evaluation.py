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

from .image_utils import (reconstruct_volume, 
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
    num_examples_to_save: int = 1,
    is_hu: bool = True,
    logger: logging.Logger | None =None,
    use_wandb: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Evaluate a 3D reconstruction model over an entire dataset of full volumes.

    This function iterates through a dataloader of 3D volumes, runs inference 
    using either full-volume mode or patch-based reconstruction, computes 
    per-volume and aggregated MSE metrics, and optionally saves a single
    example reconstruction for qualitative inspection.

    It is designed to run in both single-process and DistributedDataParallel (DDP)
    settings. In DDP, metrics are aggregated across processes, while example
    saving and optional W&B logging are performed only on rank 0.

    Parameters
    ----------
    model : torch.nn.Module
        The trained 3D model to evaluate. Must accept tensors of shape
        (B, C, D, H, W) and return tensors of the same shape.

    mode : {"full", "patch"}
        Evaluation mode:
        - "full": run inference on the entire volume in a single forward pass.
        - "patch": extract sliding-window patches, run inference on each, and
          stitch the outputs back together using `patch_size` and `stride`.

    dataloader : torch.utils.data.DataLoader
        Yields full 3D volumes. Each batch should have shape (1, C, D, H, W).

    patch_size : tuple[int, int, int], optional
        Size of 3D patches `(D, H, W)` for patch-based inference.

    stride : tuple[int, int, int], optional
        Sliding-window stride for patch extraction in patch mode.

    use_blending : bool, default=True
        If True, apply a smooth blending window (e.g., Hann) to overlapping
        patches during reconstruction. If False, overlaps are averaged.

    device : torch.device, optional
        Device on which evaluation is performed. If None, uses the model’s
        current device.

    save_example_dir : str or None, default=None
        If provided, saves **one** example volume (determined by
        `example_volume_index`) containing a tensor of shape
        `(C, D, H, W, 2)` where each slice is:
        `[original | reconstructed]`.

    num_examples_to_save : int, default=1
        Number of example volumes to save.

    is_hu : bool, default=True
        If True, assumes input volumes are in HU (Hounsfield Units) and applies
        appropriate windowing for visualization.

    logger : logging.Logger or None, default=None
        Optional logger for progress updates and debugging messages.

    use_wandb : bool, default=False
        If True, logs the example volume and scalar metrics to Weights & Biases.
        Only executed by rank 0 when using DDP.

    Returns
    -------
    dict
        A dictionary containing:
        
        - **"per_volume_mse"** : torch.Tensor (N_local,)
              Per-volume MSE computed on the subset of data handled by the local
              process (under DDP); contains all volumes in single-process mode.

        - **"mean_mse"** : torch.Tensor (scalar)
              Global mean MSE aggregated across all processes (if DDP).

    Notes
    -----
    - The caller must set `model.eval()` before calling this function.
    - In patch mode, `patch_size` and `stride` must be supplied.
    - Only rank 0 performs saving and W&B logging.
    - This function operates under `torch.no_grad()` to reduce memory usage.
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
    local_l1_sum = 0.0
    local_l1_count = 0.0
    examples_saved = 0

    global_vol_idx = 0

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
        l1_per_vol = F.l1_loss(recon, x, reduction="mean")
        mse_per_vol = mse_per_vol.view(B, -1).mean(dim=1)  # (B,)
        per_batch_mses.append(mse_per_vol.cpu())
        local_l1_sum += l1_per_vol.detach()
        local_l1_count += 1

        # Save example only once, and only on rank 0 if DDP.
        if (
            save_example_dir is not None
            and examples_saved < num_examples_to_save
            and rank == 0
        ):

            for local_idx in range(B):
            # Verify example volume index is in range
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
                    input_frames = volume_to_gif_frames(vol_in, every_n=2, is_hu=is_hu)  # every 2 slices, tweak as needed
                    input_gif_path = save_gif(input_frames, fps=10)
                    input_mp4_path = save_mp4(input_frames, fps=10)

                    # ---- 3D scrollable GIF: side-by-side (input | recon) ----
                    recon_frames = volume_to_gif_frames(vol_out, every_n=2, is_hu=is_hu)
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
                    
                examples_saved += 1
                if examples_saved >= num_examples_to_save:
                    break

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

        l1_tensor = torch.stack([local_l1_sum, local_l1_count])
        dist.all_reduce(l1_tensor, op=dist.ReduceOp.SUM)

        global_l1_sum, global_l1_count = l1_tensor.tolist()
        global_l1_avg = global_l1_sum / max(global_l1_count, 1)
    else:
        mean_mse = (local_sum / local_count).cpu()
        global_l1_avg = local_l1_sum / max(local_l1_count, 1)

    if rank == 0 and use_wandb:
        table = wandb.Table(columns=["metric", "value"])
        table.add_data("mean_mse", float(mean_mse))
        table.add_data("mean_l1",  float(global_l1_avg.item()))
        wandb.log({"final_metrics": table}, commit=True)
        wandb.run.summary["final/mean_mse"] = float(mean_mse)
        wandb.run.summary["final/mean_l1"]  = float(global_l1_avg.item())

    return {
        "per_volume_mse": per_volume_mse_local,  # local-only if DDP
        "mean_mse": mean_mse,                    # global if DDP
        "global_l1_avg": global_l1_avg
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

