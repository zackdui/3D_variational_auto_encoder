# This file is the entry point to run training and testing

import os
import torch
import logging
from pathlib import Path

from model import CustomVAE, AttnParams
from dataloaders import RandomVolumeDataset
from feasibility import find_max_batch_size_power2
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


if __name__ == "__main__":
    # For DDP on K gpu's training run 
    # CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=K main.py
    # CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 main.py

    check_feasibility = False
    train_model = True
    # Don't run eval and train in the same run for DDP
    eval_model_bool = False
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
        model_name = "random_test.pt"
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

        model = CustomVAE(vae_use_log_var=True, blocks_down=blocks_down, blocks_up=blocks_up, use_attn=use_attn, attn_params=attn_params1)
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
                    wandb_run_name=train_wandb_name,
                    checkpoint_dir=f"{model_name[:-3]}_checkpoints",
                    save_every_steps=2,
                    best_check_every_steps=1)

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
        
        
      
