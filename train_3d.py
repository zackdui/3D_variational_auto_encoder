
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, random_split
import logging
import wandb
import time

from model import CustomVAE
from trainer import Trainer
from dataloaders import RandomPatch3DDataset, AllPatch3DDataset, RandomVolumeDataset


def training_3D(model: CustomVAE,
                base_dataset, 
                optimizer_cls=torch.optim.AdamW,
                optimizer_kwargs=None,
                accum_steps=1,
                epochs=10,
                patching: str | None = None, 
                train_split=.9, 
                patch_size=(32, 128, 128), 
                patches_per_volume=4,
                train_batch=1,
                val_batch=1,
                num_workers=0,
                model_file_name="model.pt",
                use_amp=False,
                amp_dtype=torch.bfloat16,
                use_grad_scaler=False,
                logger: logging.Logger | None =None,
                use_wandb=False,
                wandb_run_name="run01",):
    """
    This function works for ddp initialization and non-ddp initialization
    base_dataset: Should be a dataset that just returns volume
    patching: Can be "full" or "random_parts"
    """
    logger = logger or logging.getLogger(__name__)

    hparams = {
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "model_hparams": model.get_hparams(),
        "base_dataset_class": base_dataset.__class__.__name__,
        "optimizer_cls": f"{optimizer_cls.__module__}.{optimizer_cls.__name__}",
        "optimizer_kwargs": optimizer_kwargs or {},
        "accum_steps": accum_steps,
        "epochs": epochs,
        "patching": patching,
        "train_split": train_split,
        "patch_size": patch_size,
        "patches_per_volume": patches_per_volume,
        "train_batch": train_batch,
        "val_batch": val_batch,
        "num_workers": num_workers,
        "model_file_name": model_file_name,
        "use_amp": use_amp,
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "use_grad_scaler": use_grad_scaler,
        "logger_name": logger.name,
        "num_gpus_visible": torch.cuda.device_count(),
    }


    if optimizer_kwargs is None:
        optimizer_kwargs = dict(lr=1e-4, weight_decay=1e-5)

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
            name=f"{wandb_run_name}_{time.time()}",        # optional run name
            config=hparams,       # logs hyperparameters automatically
        )
        logger.info("Initialized Weights & Biases run: %s", wandb.run.name)

    # DDP Setup
    # dist.init_process_group(backend="nccl")

    # rank = dist.get_rank()
    # world_size = dist.get_world_size()
    # local_rank = int(os.environ["LOCAL_RANK"])

    # torch.cuda.set_device(local_rank)
    # device = torch.device(f"cuda:{local_rank}")
    
    # Datasets
    train_ds, val_ds = random_split(base_dataset, [train_split, 1 - train_split])
    if patching is None:
        train_dataset = train_ds
        val_dataset = val_ds
    elif patching == "full":
        # This stride leads to 50% overlap
        train_dataset = AllPatch3DDataset(train_ds, patch_size, stride=(patch_size[0]//2, patch_size[1]//2, patch_size[2]//2))
        val_dataset   = AllPatch3DDataset(val_ds, patch_size, stride=(patch_size[0]//2, patch_size[1]//2, patch_size[2]//2))
    elif patching == "random_parts":
        train_dataset = RandomPatch3DDataset(train_ds, patch_size, patches_per_volume=patches_per_volume)
        val_dataset   = RandomPatch3DDataset(val_ds, patch_size, patches_per_volume=patches_per_volume)
    else:
        raise ValueError("Patching needs to be either None, full or random_parts")
    
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=(patching != "full"),
            drop_last=False,
        )

        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        train_sampler=None
        val_sampler=None

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True,
    )

    # Define your model
    model.to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

    # Train Model
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        accum_steps=accum_steps,
        is_main_process=(rank == 0),
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        use_grad_scaler=use_grad_scaler,
        logger=logger,
        use_wandb=use_wandb,
    )

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        train_sampler=train_sampler
    )

    if rank == 0:
        logger.info("Training finished.")
        if is_distributed:
            model.module.save_checkpoint(filename=model_file_name)
        else:
            model.save_checkpoint(filename=model_file_name)

        logger.info(f"Model saved to {model_file_name}")
        wandb.finish()

    if is_distributed:
        dist.destroy_process_group()
    