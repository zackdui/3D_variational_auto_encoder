
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
                wandb_run_name="run01",
                checkpoint_dir="./checkpoints",
                save_every_steps=500,
                best_check_every_steps=100):
    """
    Train a 3D VAE model on full volumes or randomly sampled 3D patches.

    This function handles dataset splitting, dataloader construction, optimizer
    setup, automatic mixed precision (AMP), gradient accumulation, checkpointing,
    and optional W&B logging. It supports both single-process and DDP training.
    Training can be done either on full volumes or on randomly sampled patches
    extracted from each volume.

    Parameters
    ----------
    model : CustomVAE
        The model to train. Must return `(reconstruction, loss)` during the
        forward pass.

    base_dataset : torch.utils.data.Dataset
        A dataset returning full 3D volumes of shape (C, D, H, W) or a dict
        containing such a tensor.

    optimizer_cls : torch.optim.Optimizer, default=torch.optim.AdamW
        Optimizer class used for training.

    optimizer_kwargs : dict or None, default=None
        Keyword arguments passed to the optimizer class.

    accum_steps : int, default=1
        Number of gradient accumulation steps. Effective batch size is
        `train_batch * accum_steps`.

    epochs : int, default=10
        Number of full training epochs.

    patching : {"full", "random_parts", None}, default=None
        How the input volumes are provided to the model:
        - "full": full-volume training.
        - "random_parts": sample patches of size `patch_size` from each volume.
        - None: same as "full".

    train_split : float, default=0.9
        Fraction of dataset used for training; remainder is validation.

    patch_size : tuple[int, int, int], default=(32, 128, 128)
        Patch size `(D, H, W)` used when `patching="random_parts"`.

    patches_per_volume : int, default=4
        Number of random patches drawn per volume per epoch in patching mode. Only used if `patching="random_parts"`.

    train_batch : int, default=1
        Batch size for the training dataloader.

    val_batch : int, default=1
        Batch size for the validation dataloader.

    num_workers : int, default=0
        Number of dataloader worker processes.

    model_file_name : str, default="model.pt"
        Filename for the final saved model.

    use_amp : bool, default=False
        Enable automatic mixed precision training.

    amp_dtype : torch.dtype, default=torch.bfloat16
        AMP dtype to use (recommended: bfloat16 on newer GPUs).

    use_grad_scaler : bool, default=False
        Whether to apply `torch.cuda.GradScaler` for stable float16 training.

    logger : logging.Logger or None, default=None
        Logger for status messages. If None, a default logger is used.

    use_wandb : bool, default=False
        Enable Weights & Biases logging (only on rank 0 during DDP).

    wandb_run_name : str, default="run01"
        Name of the W&B run if logging is enabled.

    checkpoint_dir : str, default="./checkpoints"
        Directory for saving periodic and best model checkpoints.

    save_every_steps : int, default=500
        Frequency (in steps) for saving normal checkpoints.

    best_check_every_steps : int, default=100
        Frequency for evaluating validation loss and saving the "best" model.

    Returns
    -------
    history : dict
        Dictionary containing per-epoch loss curves:

        {
            "train_loss": [float_per_epoch, ...],
            "val_loss":   [float_per_epoch, ...],
        }

        These lists grow to length `epochs`.

    Notes
    -----
    - DDP initialization happens inside this function if environment variables
      indicate a distributed launch.
    - Only rank 0 writes checkpoints and logs to W&B.
    - Patch sampling occurs every epoch, giving the model new subvolumes each
      time.
    - AMP + gradient scaling (if enabled) can significantly reduce memory usage.
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
        "checkpoint_dir": checkpoint_dir,
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
        checkpoint_dir=checkpoint_dir,
        save_every_steps=save_every_steps,
        best_check_every_steps=best_check_every_steps,
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
    