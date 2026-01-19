
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import logging
import wandb
import time

from .evaluation import evaluate_model_on_full_volumes

def eval_model_3D(model, 
               dataset, 
               batch_size=1, 
               mode="full", 
               patch_size=None, 
               stride=None, 
               save_dir="./test_sample", 
               use_blending=True, 
               logger: logging.Logger | None =None,
               use_wandb=False,
               wandb_run_name="run01",
               wandb_project_name="3d_vae_evals",
               num_examples_to_save=1,
               is_hu=True):
    """
    Evaluate a 3D reconstruction model on a dataset using either full-volume
    inference or patch-based inference with optional blending.

    This function works for both DistributedDataParallel (DDP) and
    single-process (non-DDP) setups. It loads volumes from `dataset`,
    performs forward passes through `model`, optionally reconstructs the
    volume from patches, and saves outputs for inspection. Optionally logs
    metrics and visualizations to Weights & Biases.

    Parameters
    ----------
    model : torch.nn.Module
        The trained 3D model to evaluate. Must accept inputs of shape
        (B, C, D, H, W) and return a reconstructed volume of the same size.

    dataset : torch.utils.data.Dataset
        A dataset yielding full 3D volumes. Each item must be a tensor of
        shape (C, D, H, W) or a dict containing such a tensor.

    batch_size : int, default=1
        Batch size for evaluation. Full-volume evaluation typically
        requires batch_size=1 due to memory constraints.

    mode : {"full", "patch"}, default="full"
        Evaluation mode:
        - "full": Run the model directly on full 3D volumes.
        - "patch": Extract overlapping patches of size `patch_size` using
          the given `stride`, run inference on each patch, and stitch the
          outputs back together. Useful when the full volume does not fit
          in GPU memory.

    patch_size : tuple[int, int, int], optional
        Size of 3D patches `(D, H, W)` when `mode="patch"`. Required if
        patch mode is used.

    stride : tuple[int, int, int], optional
        Sliding-window stride for patch extraction. Smaller strides give
        more overlap at the cost of more compute.

    save_dir : str, default="./test_sample"
        Directory where reconstructed volumes and optional visualizations
        (e.g., mid-slice images, GIFs, MP4s) will be written.

    use_blending : bool, default=True
        If True, overlapping patches are merged using a blending window
        (e.g., Hann window). If False, patch outputs are simply averaged
        in overlapping regions.

    logger : logging.Logger or None, default=None
        Logger for printing progress and debug information. If None, a
        default module-level logger is created.

    use_wandb : bool, default=False
        Whether to log evaluation metrics and sample visualizations to
        Weights & Biases. Only the rank-0 process logs under DDP.

    wandb_run_name : str, default="run01"
        Name of the W&B run to use when `use_wandb=True`.

    wandb_project_name : str, default="3d_vae_evals"
        Name of the W&B project to use when `use_wandb=True`.

    num_examples_to_save : int, default=1
        Number of example volumes to save.

    is_hu : bool, default=True
        If True, assumes input volumes are in HU (Hounsfield Units) and applies
        appropriate windowing for visualization.

    Returns
    -------
    dict
        A dictionary coming from `evaluate_model_on_full_volumes()` with the keys:

        - `"per_volume_mse"` : dict[int, float]
            Per-volume MSE for the subset of data processed by the local process
            (if DDP). In single-process mode, contains metrics for all volumes.

        - `"mean_mse"` : float
            Global mean MSE over *all* volumes (aggregated across processes when
            using DDP).
    Notes
    -----
    - This function does NOT call `model.eval()` internally; callers
      should ensure the model is in eval mode.
    - When running under DDP, only rank 0 performs file I/O and W&B logging.
    - If `mode="patch"`, both `patch_size` and `stride` must be supplied.
    """
    logger = logger or logging.getLogger(__name__)

    hparams = {
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "model_hparams": model.get_hparams(),
        "base_dataset_class": dataset.__class__.__name__,
        "patch_size": patch_size,
        "stride": stride,
        "save_dir": save_dir,
        "use_blending": use_blending,
        "logger_name": logger.name,
        "num_gpus_visible": torch.cuda.device_count(),
    }

    # Initialize DDP if launched with torchrun.
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        # Fallback (single process, single GPU/CPU)
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_distributed = dist.is_available() and dist.is_initialized()

    if rank == 0 and use_wandb:  # only main process
        wandb.init(
            project=wandb_project_name,     # choose a project name (creates if missing)
            name=f"eval_{wandb_run_name}_{time.time()}",        # run name
            config=hparams,       # logs hyperparameters automatically
        )
        logger.info("Initialized Weights & Biases run: %s (id=%s)", wandb.run.name, wandb.run.id)


    model.to(device)

    if is_distributed:
        # Should be able to remove local rank here
        local_rank = int(os.environ["LOCAL_RANK"])
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False if sampler is not None else False,
        num_workers=4,
        pin_memory=True,
    )

    save_dir_here = save_dir if rank == 0 else None

    results = evaluate_model_on_full_volumes(
        model=model,
        mode=mode,
        dataloader=dataloader,
        patch_size=patch_size,
        stride=stride,
        device=device,
        save_example_dir=save_dir_here,
        example_volume_index=0,
        use_blending=use_blending,
        num_examples_to_save=num_examples_to_save,
        is_hu=is_hu,
        logger=logger,
        use_wandb=use_wandb
    )

    # Only rank 0 prints the global mean
    if rank == 0:
        logger.info(f"[Eval] mean MSE: {results['mean_mse'].item():.6f}")
        wandb.finish()

    if is_distributed:
        dist.destroy_process_group()
    return results