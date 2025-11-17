# This function is the entry point to run training and testing

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset, random_split
import logging
from pathlib import Path
import wandb
import time

from model import CustomVAE, AttnParams
from trainer import Trainer
from dataloaders import RandomPatch3DDataset, AllPatch3DDataset, RandomVolumeDataset
from feasibility import find_max_batch_size_power2
from evaluation import evaluate_model_on_full_volumes
from eval_3d import eval_model_3D
from train_3d import training_3D

def setup_logger(save_dir: str = None, name: str = "vae_trainer", train=True) -> logging.Logger:
    if save_dir is None:
        cwd = os.getcwd()              # get current working directory
        save_path = Path(os.path.join(cwd, "logging", name))
    else:
        save_path = Path(os.path.join(save_dir, name))
        # save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    if train:
        log_file = save_path / "train.log"
    else:
        log_file = save_path / "eval.log"

    logging.basicConfig(
        level=logging.INFO,  # log INFO and above
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),          # → console
            logging.FileHandler(log_file),    # → train.log
        ],
    )
    logger = logging.getLogger(name)
    logger.info("Logger initialized. Writing logs to %s", log_file)
    return logger

# def training_3D(model: CustomVAE,
#                 base_dataset, 
#                 optimizer_cls=torch.optim.AdamW,
#                 optimizer_kwargs=None,
#                 accum_steps=1,
#                 epochs=10,
#                 patching: str | None = None, 
#                 train_split=.9, 
#                 patch_size=(32, 128, 128), 
#                 patches_per_volume=4,
#                 train_batch=1,
#                 val_batch=1,
#                 num_workers=0,
#                 model_file_name="model.pt",
#                 use_amp=False,
#                 amp_dtype=torch.bfloat16,
#                 use_grad_scaler=False,
#                 logger: logging.Logger | None =None,
#                 use_wandb=False,
#                 wandb_run_name="run01",):
#     """
#     This function works for ddp initialization and non-ddp initialization
#     base_dataset: Should be a dataset that just returns volume
#     patching: Can be "full" or "random_parts"
#     """
#     logger = logger or logging.getLogger(__name__)

#     hparams = {
#         "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
#         "model_hparams": model.get_hparams(),
#         "base_dataset_class": base_dataset.__class__.__name__,
#         "optimizer_cls": f"{optimizer_cls.__module__}.{optimizer_cls.__name__}",
#         "optimizer_kwargs": optimizer_kwargs or {},
#         "accum_steps": accum_steps,
#         "epochs": epochs,
#         "patching": patching,
#         "train_split": train_split,
#         "patch_size": patch_size,
#         "patches_per_volume": patches_per_volume,
#         "train_batch": train_batch,
#         "val_batch": val_batch,
#         "num_workers": num_workers,
#         "model_file_name": model_file_name,
#         "use_amp": use_amp,
#         "amp_dtype": str(amp_dtype).replace("torch.", ""),
#         "use_grad_scaler": use_grad_scaler,
#         "logger_name": logger.name,
#         "num_gpus_visible": torch.cuda.device_count(),
#     }


#     if optimizer_kwargs is None:
#         optimizer_kwargs = dict(lr=1e-4, weight_decay=1e-5)

#     # Initialize DDP if launched with torchrun.
#     if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
#         rank = int(os.environ["RANK"])
#         world_size = int(os.environ["WORLD_SIZE"])
#         dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

#         local_rank = int(os.environ["LOCAL_RANK"])
#         torch.cuda.set_device(local_rank)
#         device = torch.device(f"cuda:{local_rank}")
#     else:
#         # Fallback (single process, single GPU/CPU)
#         rank = 0
#         world_size = 1
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     is_distributed = dist.is_available() and dist.is_initialized()

#     if rank == 0 and use_wandb:  # only main process
#         wandb.init(
#             project="3d-vae",     # choose a project name (creates if missing)
#             name=f"{wandb_run_name}_{time.time()}",        # optional run name
#             config=hparams,       # logs hyperparameters automatically
#         )
#         logger.info("Initialized Weights & Biases run: %s", wandb.run.name)

#     # DDP Setup
#     # dist.init_process_group(backend="nccl")

#     # rank = dist.get_rank()
#     # world_size = dist.get_world_size()
#     # local_rank = int(os.environ["LOCAL_RANK"])

#     # torch.cuda.set_device(local_rank)
#     # device = torch.device(f"cuda:{local_rank}")
    
#     # Datasets
#     train_ds, val_ds = random_split(base_dataset, [train_split, 1 - train_split])
#     if patching is None:
#         train_dataset = train_ds
#         val_dataset = val_ds
#     elif patching == "full":
#         # This stride leads to 50% overlap
#         train_dataset = AllPatch3DDataset(train_ds, patch_size, stride=(patch_size[0]//2, patch_size[1]//2, patch_size[2]//2))
#         val_dataset   = AllPatch3DDataset(val_ds, patch_size, stride=(patch_size[0]//2, patch_size[1]//2, patch_size[2]//2))
#     elif patching == "random_parts":
#         train_dataset = RandomPatch3DDataset(train_ds, patch_size, patches_per_volume=patches_per_volume)
#         val_dataset   = RandomPatch3DDataset(val_ds, patch_size, patches_per_volume=patches_per_volume)
#     else:
#         raise ValueError("Patching needs to be either None, full or random_parts")
    
#     if is_distributed:
#         train_sampler = DistributedSampler(
#             train_dataset,
#             num_replicas=world_size,
#             rank=rank,
#             shuffle=(patching != "full"),
#             drop_last=False,
#         )

#         val_sampler = DistributedSampler(
#             val_dataset,
#             num_replicas=world_size,
#             rank=rank,
#             shuffle=False,
#             drop_last=False,
#         )
#     else:
#         train_sampler=None
#         val_sampler=None

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=train_batch,
#         sampler=train_sampler,
#         num_workers=num_workers,
#         pin_memory=True,
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=val_batch,
#         sampler=val_sampler,
#         num_workers=0,
#         pin_memory=True,
#     )

#     # Define your model
#     model.to(device)
#     if is_distributed:
#         model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
#     optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

#     # Train Model
#     trainer = Trainer(
#         model=model,
#         optimizer=optimizer,
#         device=device,
#         accum_steps=accum_steps,
#         is_main_process=(rank == 0),
#         use_amp=use_amp,
#         amp_dtype=amp_dtype,
#         use_grad_scaler=use_grad_scaler,
#         logger=logger,
#         use_wandb=use_wandb,
#     )

#     history = trainer.train(
#         train_loader=train_loader,
#         val_loader=val_loader,
#         num_epochs=epochs,
#         train_sampler=train_sampler
#     )

#     if rank == 0:
#         logger.info("Training finished.")
#         if is_distributed:
#             model.module.save_checkpoint(filename=model_file_name)
#         else:
#             model.save_checkpoint(filename=model_file_name)

#         logger.info(f"Model saved to {model_file_name}")
#         wandb.finish()

#     if is_distributed:
#         dist.destroy_process_group()
    

# def eval_model_3D(model, 
#                dataset, 
#                batch_size=1, 
#                mode="full", 
#                patch_size=None, 
#                stride=None, 
#                save_dir="./test_sample", 
#                use_blending=True, 
#                logger: logging.Logger | None =None,
#                use_wandb=False,
#                wandb_run_name="run01",):
#     """
#     This function works for ddp initialization and non-ddp initialization
    
#     mode must be "full" or "patch"
#     """
#     logger = logger or logging.getLogger(__name__)

#     hparams = {
#         "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
#         "model_hparams": model.get_hparams(),
#         "base_dataset_class": dataset.__class__.__name__,
#         "patch_size": patch_size,
#         "stride": stride,
#         "save_dir": save_dir,
#         "use_blending": use_blending,
#         "logger_name": logger.name,
#         "num_gpus_visible": torch.cuda.device_count(),
#     }


#     # Initialize DDP if launched with torchrun.
#     if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
#         rank = int(os.environ["RANK"])
#         world_size = int(os.environ["WORLD_SIZE"])
#         dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

#         local_rank = int(os.environ["LOCAL_RANK"])
#         torch.cuda.set_device(local_rank)
#         device = torch.device(f"cuda:{local_rank}")
#     else:
#         # Fallback (single process, single GPU/CPU)
#         rank = 0
#         world_size = 1
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     is_distributed = dist.is_available() and dist.is_initialized()

#     if rank == 0 and use_wandb:  # only main process
#         wandb.init(
#             project="3d-vae",     # choose a project name (creates if missing)
#             name=f"eval_{wandb_run_name}_{time.time()}",        # optional run name
#             config=hparams,       # logs hyperparameters automatically
#         )
#         logger.info("Initialized Weights & Biases run: %s (id=%s)", wandb.run.name, wandb.run.id)


#     model.to(device)

#     if is_distributed:
#         # Should be able to remove local rank here
#         local_rank = int(os.environ["LOCAL_RANK"])
#         model = DDP(model, device_ids=[local_rank], output_device=local_rank)

#     if is_distributed:
#         sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
#     else:
#         sampler = None

#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         sampler=sampler,
#         shuffle=False if sampler is not None else False,
#         num_workers=4,
#         pin_memory=True,
#     )

#     save_dir_here = save_dir if rank == 0 else None

#     results = evaluate_model_on_full_volumes(
#         model=model,
#         mode=mode,
#         dataloader=dataloader,
#         patch_size=patch_size,
#         stride=stride,
#         device=device,
#         save_example_dir=save_dir_here,
#         example_volume_index=0,
#         use_blending=use_blending,
#         logger=logger,
#         use_wandb=use_wandb
#     )

#     # Only rank 0 prints the global mean
#     if rank == 0:
#         logger.info(f"[Eval] mean MSE: {results['mean_mse'].item():.6f}")
#         wandb.finish()

#     if is_distributed:
#         dist.destroy_process_group()



if __name__ == "__main__":
    # For DDP on K gpu's training run 
    # CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=K main.py
    # CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 main.py
    check_feasibility = False
    train_model = False
    # Don't run eval and train in the same run for DDP
    eval_model_bool = True
    eval_model_checkpoint = "./saved_models/patch_test.pt"
    train_wandb_name = "test_run"
    eval_wandb_name = "test_run"
    ################ Check Batch Size Feasibility ################
    if check_feasibility:
        attn_params = AttnParams(num_heads=4, 
                                 dim_head=32, 
                                 dropout=.1, 
                                 window_size=(4, 8, 8), 
                                 use_rel_pos_bias=True)
        model = CustomVAE(blocks_down=(1,2,2,4), blocks_up=(1,1,1), use_attn=True, attn_params=attn_params)

        spatial_size = (1, 64, 128, 128)
        # size (1, 32, 128, 128) and model CustomVAE(blocks_down=(1,2,4), blocks_up=(1,1), use_attn=True, attn_params=attn_params) has 128 batch size
        # size (1, 64, 128, 128) and model CustomVAE(blocks_down=(1,2,4), blocks_up=(1,1), use_attn=True, attn_params=attn_params) has 64 batch size

        max_bs = find_max_batch_size_power2(
            model,
            spatial_size=spatial_size,
            device="cuda",
            max_power=7,    # test 1 → 128
            use_amp=False,
            amp_dtype=torch.bfloat16,
        )

        print("Largest power-of-two batch:", max_bs)

    ################# Training Code ##############################
    if train_model:
        model_name = "no_ddp_test.pt"
        logger_train = setup_logger(name=f"{model_name[:-3]}_logs")
        # Dataset
        base_dataset = RandomVolumeDataset(20, shape=(1, 64, 128, 128))

        # Model
        attn_params1 = AttnParams(num_heads=4, dim_head=32, dropout=.1, window_size=(4, 8, 8), use_rel_pos_bias=True)
        attn_params2 = AttnParams(num_heads=4, dim_head=32, dropout=.1)
        blocks_down=(1,2,2,4)
        blocks_up=(1,1,1)
        use_attn=True
        attn_params=attn_params1
        
        model = CustomVAE(blocks_down=blocks_down, blocks_up=blocks_up, use_attn=use_attn, attn_params=attn_params1)
        optimizer_cls = torch.optim.AdamW
        

        # Training
        training_3D(model, 
                    base_dataset, 
                    optimizer_cls=optimizer_cls, 
                    accum_steps=4, 
                    epochs=3, 
                    optimizer_kwargs=dict(lr=1e-4, 
                                          weight_decay=1e-5), 
                    model_file_name=model_name,
                    logger=logger_train,
                    use_wandb=True,
                    wandb_run_name=train_wandb_name)

    ########################## Evaluation Code #####################
    if eval_model_bool:
        # Load Model for Testing
        base_dataset = RandomVolumeDataset(10, shape=(1, 208, 512, 512))
        filename = os.path.basename(eval_model_checkpoint)
        name_only = os.path.splitext(filename)[0]
        model = CustomVAE.load_from_checkpoint(eval_model_checkpoint, map_location="cuda")
        logger_val = setup_logger(name=f"{name_only}_logs", train=False)
        batch_size = 1
        mode = "patch"
        patch_size = (64, 128, 128)
        stride = (32, 64, 64)
        use_blending=True
        save_dir = "./test_output_patch_blend"

        # Evaluate Model
        eval_model_3D(model, 
                   base_dataset, 
                   batch_size, 
                   mode, 
                   patch_size, 
                   stride, 
                   save_dir, 
                   use_blending,
                   logger=logger_val,
                   use_wandb=True,
                   wandb_run_name=train_wandb_name)
        
        
        
        # Add debug to model
        
      
    



# Go back to model delete some more unnessesary things
# Fix the loss functions for the VAE in the model