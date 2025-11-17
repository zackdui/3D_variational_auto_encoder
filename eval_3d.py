
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import logging
import wandb
import time


from evaluation import evaluate_model_on_full_volumes

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
               wandb_run_name="run01",):
    """
    This function works for ddp initialization and non-ddp initialization
    
    mode must be "full" or "patch"
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
            project="3d-vae",     # choose a project name (creates if missing)
            name=f"eval_{wandb_run_name}_{time.time()}",        # optional run name
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
        logger=logger,
        use_wandb=use_wandb
    )

    # Only rank 0 prints the global mean
    if rank == 0:
        logger.info(f"[Eval] mean MSE: {results['mean_mse'].item():.6f}")
        wandb.finish()

    if is_distributed:
        dist.destroy_process_group()