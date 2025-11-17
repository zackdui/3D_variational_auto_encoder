## This file will contain the Trainer to train our VAE model
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import Optimizer
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import logging
import wandb
import tempfile
import imageio.v2 as imageio
import numpy as np
import os

from image_utils import safe_delete, prepare_for_wandb, volume_to_gif_frames, save_gif, save_mp4

# def safe_delete(path):
#     try:
#         os.remove(path)
#     except OSError:
#         pass

# def prepare_for_wandb(slice_2d):
#     """
#     slice_2d: torch.Tensor or np.ndarray, shape (H, W) or (1, H, W)
#               values expected in [-1, 1]
#     returns: np.ndarray uint8, shape (H, W)
#     """
#     if hasattr(slice_2d, "detach"):  # torch.Tensor
#         slice_2d = slice_2d.detach().cpu().numpy()

#     slice_2d = np.squeeze(slice_2d)          # drop channel if (1, H, W)
#     slice_2d = np.clip(slice_2d, -1.0, 1.0)  # enforce range
#     slice_2d = (slice_2d + 1.0) / 2.0        # [-1,1] -> [0,1]
#     slice_2d = (slice_2d * 255.0).round().astype("uint8")
#     return slice_2d

# def volume_to_gif_frames(volume_3d, every_n: int = 1):
#     """
#     volume_3d: torch.Tensor or np.ndarray, shape (D, H, W)
#                values in [-1, 1]
#     every_n: use every n-th slice to keep GIF small
#     returns: list of uint8 (H, W) frames
#     """
#     if hasattr(volume_3d, "detach"):
#         volume_3d = volume_3d.detach().cpu().numpy()
#     D = volume_3d.shape[0]
#     frames = []
#     for d in range(0, D, every_n):
#         frame = prepare_for_wandb(volume_3d[d])
#         frames.append(frame)
#     return frames

# def save_gif(frames, fps: int = 10) -> str:
#     """
#     frames: list of (H, W) or (H, W, 3) uint8 arrays
#     returns: path to a temporary .gif file
#     """
#     tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
#     tmp.close()
#     imageio.mimsave(tmp.name, frames, fps=fps)
#     return tmp.name

# def save_mp4(frames, fps=10):
#     tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
#     tmp.close()
#     imageio.mimsave(tmp.name, frames, fps=fps)  # imageio automatically picks mp4 writer
#     return tmp.name

class Trainer:
    def __init__(
        self,
        model,
        optimizer: Optimizer,
        device=None,
        accum_steps: int = 1,
        is_main_process: bool = True,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        use_grad_scaler: bool | None = None,
        logger: logging.Logger | None = None,
        use_wandb: bool = False,
    ):
        """
        model: your VAE; in train mode forward returns (out, loss),
               in eval mode returns out only.
        optimizer: an *instance* of torch.optim.Optimizer (e.g. AdamW(...))
        device: torch.device or string, e.g. 'cuda:0'
        accum_steps: gradient accumulation steps. If >1, will accumulate gradients
        is_main_process: set False for non-zero ranks in DDP so only rank 0 prints/logs
        use_amp: whether to enable mixed-precision training
        amp_dtype: torch.float16 (works on V100 + A6000 + A100) or torch.bfloat16 (A6000 and A100 only)
        use_grad_scaler: if None, auto-choose (True for fp16, False otherwise)
        """
        self.model = model
        self.optimizer = optimizer              # must be an instance, not a class
        self.accum_steps = max(1, accum_steps)
        self.device = device if device is not None else next(model.parameters()).device
        self.is_main_process = is_main_process
        self.is_distributed = dist.is_available() and dist.is_initialized()

        if self.is_distributed:
            self.dims = self.model.module.get_hparams()["spatial_dims"]
        else:
            self.dims = self.model.get_hparams()["spatial_dims"]

        # AMP configuration
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype

        if use_grad_scaler is None:
            # fp16 → use GradScaler; bf16 or no AMP → no scaler
            self.use_grad_scaler = self.use_amp and (self.amp_dtype == torch.float16)
        else:
            self.use_grad_scaler = use_grad_scaler

        self.scaler = GradScaler() if self.use_grad_scaler else None

        self.logger = logger or logging.getLogger(__name__)
        # Only main process prints/logs big summary lines
        if self.is_main_process:
            self.logger.info("Trainer initialized on device=%s", self.device)
            self.logger.info(
                "Settings: accum_steps=%d, use_amp=%s, amp_dtype=%s, use_grad_scaler=%s",
                self.accum_steps, self.use_amp, self.amp_dtype, self.use_grad_scaler,
            )

        self.use_wandb = use_wandb and self.is_main_process

    def ddp_average(self, value: torch.Tensor) -> float:
        """
        All-reduce & average a scalar tensor across all processes.
        Assumes torch.distributed.is_initialized() is True.
        """
        # value is a scalar tensor on self.device
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= dist.get_world_size()
        return value.item()
    
    def train_epoch(self, dataloader, epoch: int | None = None, num_epochs: int | None = None):
        self.model.train()
        self.optimizer.zero_grad()

        if self.is_main_process:
            self.logger.info("Starting train_epoch (epoch=%s / %s)", epoch, num_epochs)

        running_loss = 0.0
        num_batches = 0

        # tqdm over the dataloader, only on main process
        desc = "Train"
        if epoch is not None and num_epochs is not None:
            desc = f"Epoch {epoch+1}/{num_epochs} [train]"

        iterator = tqdm(
            dataloader,
            desc=desc,
            disable=not self.is_main_process,
        )

        for step, batch in enumerate(iterator):
            # assuming dataloader yields just x; if (x, y) later, unpack here
            x = batch.to(self.device, non_blocking=True)

            # ----- forward with optional AMP -----
            if self.use_amp:
                with autocast(device_type="cuda", dtype=self.amp_dtype):
                    out, loss = self.model(x)  # model is in train mode → (out, loss)
            else:
                out, loss = self.model(x)

            # gradient accumulation
            loss = loss / self.accum_steps

            # ----- backward with optional GradScaler -----
            if self.use_grad_scaler and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # out, loss = self.model(x)  # model is in train mode → (out, loss)
            # loss = loss / self.accum_steps
            # loss.backward()

            running_loss += loss.item()
            num_batches += 1

            # optimizer step when we've accumulated enough grads
            if ((step + 1) % self.accum_steps == 0) or (step + 1 == len(dataloader)):
                if self.use_grad_scaler and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # update tqdm postfix with current (unscaled) loss
            if self.is_main_process:
                iterator.set_postfix(loss=float(loss.item()))

            # if ((step + 1) % self.accum_steps == 0) or (step + 1 == len(dataloader)):
            #     self.optimizer.step()
            #     self.optimizer.zero_grad()

        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

        # If using DDP, average loss across all ranks:
        if dist.is_available() and dist.is_initialized():
            avg_loss_tensor = torch.tensor(avg_loss, device=self.device, dtype=torch.float32)
            avg_loss = self.ddp_average(avg_loss_tensor)

        return avg_loss

    @torch.no_grad()
    def eval_epoch(self, dataloader, epoch: int | None = None, num_epochs: int | None = None):
        """
        In eval mode, your model forward returns only the reconstruction / output.
        Eval loss here is simple MSE reconstruction loss.
        """
        self.model.eval()

        if self.is_main_process:
            self.logger.info("Starting eval_epoch (epoch=%s / %s)", epoch, num_epochs)

        desc = "Eval"
        if epoch is not None and num_epochs is not None:
            desc = f"Epoch {epoch+1}/{num_epochs} [val]"
        
        iterator = tqdm(
            dataloader,
            desc=desc,
            disable=not self.is_main_process,
        )

        b_num = 0
        losses = []
        for batch in iterator:
            b_num += 1
            x = batch.to(self.device, non_blocking=True)

            # AMP is still useful in eval for memory/speed
            if self.use_amp:
                with autocast(device_type="cuda", dtype=self.amp_dtype):
                    out = self.model(x)  # eval mode → only out
                    loss = F.mse_loss(out, x)
            else:
                out = self.model(x)
                loss = F.mse_loss(out, x)

            # out = self.model(x)  # eval mode → only out
            # loss = F.mse_loss(out, x)
            losses.append(loss.item())
            if self.is_main_process:
                iterator.set_postfix(val_loss=float(loss.item()))

            # These functions are dependent on it being a 3D generation
            if self.use_wandb and self.is_main_process and b_num == 1 and self.dims == 3:
                # x, out: (N, C, D, H, W)
                vol_in = x[0, 0]   # (D, H, W)
                vol_out = out[0, 0]

                # ---- 2D middle slice images (what you already had) ----
                mid = vol_in.shape[0] // 2
                x_slice = prepare_for_wandb(vol_in[mid])
                out_slice = prepare_for_wandb(vol_out[mid])

                wandb.log({
                    "recon/inputs_mid_slice": wandb.Image(x_slice),
                    "recon/outputs_mid_slice": wandb.Image(out_slice),
                })

                if epoch % 5 == 0:
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

        
        loss_average = sum(losses) / max(len(losses), 1)

        if dist.is_available() and dist.is_initialized():
            loss_tensor = torch.tensor(loss_average, device=self.device, dtype=torch.float32)
            loss_average = self.ddp_average(loss_tensor)
     
        return loss_average
    
    def train(self, train_loader, val_loader=None, num_epochs: int = 10,
              train_sampler=None):
        """
        High-level training loop.
        train_sampler: DistributedSampler in DDP (so we can call set_epoch).
        """
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(num_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # train_loss = self.train_epoch(train_loader)
            train_loss = self.train_epoch(train_loader, epoch=epoch, num_epochs=num_epochs)
            history["train_loss"].append(train_loss)

            if self.is_main_process:
                # print(f"[Epoch {epoch+1}/{num_epochs}] global_avg_train_loss = {train_loss:.4f}")
                self.logger.info(
                                    "[Epoch %d/%d] global_avg_train_loss = %.4f",
                                    epoch + 1, num_epochs, train_loss,
                                )

            eval_loss = None
            if val_loader is not None:
                eval_loss = self.eval_epoch(val_loader, epoch=epoch, num_epochs=num_epochs)
                # eval_loss = self.eval_epoch(val_loader)
                history["val_loss"].append(eval_loss)

                if self.is_main_process:
                    # print(f"[Epoch {epoch+1}/{num_epochs}] global_avg_val_recon_loss = {eval_loss:.4f}")

                    self.logger.info(
                                        "[Epoch %d/%d] global_avg_val_recon_loss = %.4f",
                                        epoch + 1, num_epochs, eval_loss,
                                    )
                    
            if self.use_wandb and self.is_main_process:
                log_dict = {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                }
                if eval_loss is not None:
                    log_dict["val/loss"] = eval_loss

                lr = self.optimizer.param_groups[0]["lr"]
                log_dict["lr"] = lr

                wandb.log(log_dict)
        return history

    
# Example usage:
if __name__ == "__main__":
    # trainer = Trainer(
    #     model=model_ddp,
    #     optimizer=optimizer,
    #     device=device,
    #     accum_steps=4,
    #     is_main_process=(rank == 0),
    #     use_amp=False,                 # ← off
    # )
    # trainer = Trainer(
    #     model=model_ddp,
    #     optimizer=optimizer,
    #     device=device,
    #     accum_steps=accum_steps,
    #     is_main_process=(rank == 0),
    #     use_amp=True,
    #     amp_dtype=torch.float16,       # good on both GPUs
    #     # use_grad_scaler=None → auto: True for fp16
    # )
    # trainer = Trainer(
    #     model=model_ddp,
    #     optimizer=optimizer,
    #     device=device,
    #     accum_steps=accum_steps,
    #     is_main_process=(rank == 0),
    #     use_amp=True,
    #     amp_dtype=torch.bfloat16,      # A6000 supports this
    #     use_grad_scaler=False,         # bf16 usually doesn’t need scaling
    # )
    pass

