## This file will contain the Trainer to train our VAE model
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import Optimizer
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import logging
import wandb
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt


from .image_utils import (safe_delete, 
                          prepare_for_wandb,
                          prepare_for_wandb_hu, 
                          volume_to_gif_frames, 
                          save_gif, 
                          save_mp4,
                          log_slice_montage)

def isfinite_all(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()

def check_finite(name: str, t: torch.Tensor):
    if not isfinite_all(t):
        print(f"\n[NON-FINITE] {name}: shape={tuple(t.shape)} "
              f"min={t.nan_to_num().min().item():.3e} "
              f"max={t.nan_to_num().max().item():.3e}")
        raise RuntimeError(f"Non-finite detected: {name}")
    
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
        checkpoint_dir: str | Path = "./checkpoints",
        save_every_steps: int | None = None,
        best_check_every_steps: int | None = None,
    ):
        """
        Initialize a Trainer for 3D VAE training with support for gradient
        accumulation, mixed precision, checkpointing, and optional distributed
        operation.

        This also includes 30% of the steps as beta warmups.

        Parameters
        ----------
        model : torch.nn.Module
            The model being trained.  
            - In **train mode**, `model(x)` must return `(output, loss)`.  
            - In **eval mode**, `model(x)` must return `output` only.

        optimizer : torch.optim.Optimizer
            An instantiated optimizer (e.g., `torch.optim.AdamW(model.parameters(), ...)`).

        device : torch.device or str, optional
            Device used for training (e.g., `"cuda"`, `"cuda:0"`, `"cpu"`). If None,
            uses the model's current device.

        accum_steps : int, default=1
            Number of gradient accumulation steps.  
            Backprop is called every batch, but the optimizer updates parameters
            only after `accum_steps` backward passes.

        is_main_process : bool, default=True
            Whether this process is the “main” (rank 0) process.  
            Only rank 0 performs logging, printing, checkpointing, and W&B logging.

        use_amp : bool, default=False
            Enable automatic mixed precision (AMP) for the forward/backward pass.

        amp_dtype : torch.dtype, default=torch.float16
            Precision used inside the AMP autocast context.  
            - `torch.float16`: supported on most GPUs (V100, A6000, A100, etc.)  
            - `torch.bfloat16`: recommended on A6000/A100-class hardware

        use_grad_scaler : bool or None, default=None
            Whether to use `torch.cuda.GradScaler`.  
            - If `None`: automatically chosen (`True` when using float16 AMP).  
            - For bfloat16 AMP, this is typically set to `False`.

        logger : logging.Logger or None, default=None
            Optional Python logger for progress and debug messages.

        use_wandb : bool, default=False
            Whether to log training metrics, losses, and optional reconstructions
            to Weights & Biases. Only the main process logs when running under DDP.

        checkpoint_dir : str or Path, default="./checkpoints"
            Directory where periodic and best-model checkpoints will be stored.

        save_every_steps : int or None, default=None
            Save a checkpoint every N optimizer **steps**.  
            If None, periodic checkpoints are disabled.

        best_check_every_steps : int or None, default=None
            Validate and update the “best” model every N steps.  
            If None, best-checkpoint evaluation is disabled.

        Notes
        -----
        **Definition of a “step”**  
        A *step* means one optimizer update:

        - **Single GPU (non-DDP):**  
        One step occurs every `accum_steps` batches  
        → effectively `accum_steps × batch_size` samples per optimizer update.

        - **DDP (multi-GPU):**  
        Every GPU processes `accum_steps` batches before a synchronized update:  
        → `accum_steps × batch_size_per_gpu × world_size` total samples contribute  
            to one optimizer update.

        This ensures consistent semantics across distributed training.

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

        # --- Checkpointing setup ---
        
        self.checkpoint_dir = checkpoint_dir
        if self.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        # How often to save within an epoch (in optimizer steps); None disables
        self.save_every_steps = save_every_steps
        self.best_check_every_steps = best_check_every_steps

        # Track global optimizer steps (not batches)
        self.global_step = 0
        # Warmup steps is based on 30% of total data and therefore beta steps will increment every batch
        self.warmup_steps = 0
        self.beta_steps = 0
        if self.is_distributed:
            self.max_beta = self.model.module.get_beta()
        else:
            self.max_beta = self.model.get_beta()

        # Best validation loss so far (for "best model" saving)
        self.best_val_loss = float("inf")
        self.best_train_loss = float("inf")

    def ddp_average(self, value: torch.Tensor) -> float:
        """
        All-reduce & average a scalar tensor across all processes.
        Assumes torch.distributed.is_initialized() is True.
        """
        # value is a scalar tensor on self.device
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= dist.get_world_size()
        return value.item()
    
    def save_model(self, filename: str):
        """
        Save the model using its built-in `save_checkpoint` method.
        Handles DDP vs non-DDP and only runs on main process.
        """
        if not self.is_main_process:
            return
        full_path = os.path.join(self.checkpoint_dir, filename)

        if self.is_distributed:
            self.model.module.save_checkpoint(save_dir=self.checkpoint_dir, filename=filename)
        else:
            self.model.save_checkpoint(save_dir=self.checkpoint_dir, filename=filename)

        self.logger.info("Saved model checkpoint to %s", full_path)

    def train_epoch(self, dataloader, epoch: int | None = None, num_epochs: int | None = None):
        self.model.train()
        self.optimizer.zero_grad()

        if self.is_main_process:
            self.logger.info("Starting train_epoch (epoch=%s / %s)", epoch, num_epochs)

        running_loss = 0.0
        num_batches = 0
        window_loss_sum = 0.0
        window_loss_count = 0
        window_recon_sum = 0.0

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
            self.beta_steps += 1
            # assuming dataloader yields just x; if (x, y) later, unpack here
            x = batch.to(self.device, non_blocking=True)

            current_beta = self.beta_warmup(self.beta_steps)
            if self.is_distributed:
                self.model.module.set_beta(current_beta)
            else:
                self.model.set_beta(current_beta)

            if self.use_wandb:
                wandb.log({"train/beta": current_beta})

            # ----- forward with optional AMP -----
            if self.use_amp:
                with autocast(device_type="cuda", dtype=self.amp_dtype):
                    out, loss = self.model(x)  # model is in train mode → (out, loss)
                    check_finite("loss (forward)", loss.detach())
                    check_finite("out (forward)", out.detach())
            else:
                out, loss = self.model(x)
                check_finite("loss (forward)", loss.detach())
                check_finite("out (forward)", out.detach())

            # Don't divide loss by accum_steps yet, need raw loss for logging
            raw_loss = loss.detach()
            # gradient accumulation
            loss = loss / self.accum_steps

            # ----- backward with optional GradScaler -----
            if self.use_grad_scaler and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    print("\n[NON-FINITE GRAD]", n,
                        "grad min/max:",
                        p.grad.nan_to_num().min().item(),
                        p.grad.nan_to_num().max().item())
                    raise RuntimeError(f"Non-finite grad at {n}")

            total_grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            if self.is_main_process and self.use_wandb:  # rank == 0
                wandb.log({"train/grad_norm": total_grad_norm})

            running_loss += raw_loss.item()
            num_batches += 1
            window_loss_sum += raw_loss.item()
            window_loss_count += 1
            window_recon_sum += F.l1_loss(out, x, reduction='mean').item()
            
            # optimizer step when we've accumulated enough grads
            if ((step + 1) % self.accum_steps == 0) or (step + 1 == len(dataloader)):
                if self.use_grad_scaler and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    true_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    true_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()

                # Chack and make sure everything is finite
                for n, p in self.model.named_parameters():
                    if not torch.isfinite(p).all():
                        print("\n[NON-FINITE PARAM]", n,
                            "param min/max:",
                            p.data.nan_to_num().min().item(),
                            p.data.nan_to_num().max().item())
                        raise RuntimeError(f"Non-finite param at {n}")

                if self.is_main_process and self.use_wandb:
                    wandb.log({"grad_norm_true": true_grad_norm})

                # --- increment global step after each optimizer step ---
                self.global_step += 1

                # --- save model every N optimizer steps ---
                if (
                    self.save_every_steps is not None
                    and self.global_step % self.save_every_steps == 0
                    and self.is_main_process
                ):
                    # name it however you like; .pt or .ckpt etc.
                    self.save_model(filename=f"step_{self.global_step}.pt")

                # --- check window-averaged loss every best_check_every_steps ---
                if (
                    self.best_check_every_steps is not None
                    and self.global_step % self.best_check_every_steps == 0
                    and window_loss_count > 0
                ):
                    # compute global window average across all ranks (if DDP)
                    if self.is_distributed:
                        # pack [sum, count] into a tensor on the correct device
                        stats = torch.tensor(
                            [window_loss_sum, float(window_loss_count), window_recon_sum],
                            device=self.device,
                            dtype=torch.float32,
                        )
                        # all_reduce with SUM so we get global totals
                        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                        total_loss_sum, total_count, total_recon_sum = stats.tolist()
                        window_avg = total_loss_sum / max(total_count, 1.0)

                        window_recon_avg = total_recon_sum / max(total_count, 1.0)

                    else:
                        # single-process case
                        window_avg = window_loss_sum / window_loss_count

                        window_recon_avg = window_recon_sum / window_loss_count

                    # only main process updates best + saves
                    if self.is_main_process:
                        # ---- log to standard logger ----
                        self.logger.info(
                            "Global step %d: window-avg train loss = %.6f",
                            self.global_step,
                            window_avg,
                        )

                        # ---- log to Weights & Biases ----
                        if self.use_wandb:
                            # recon_l1 = (out - x).abs().mean().item()
                            max_err = (out - x).abs().max().item()
                            out_range = (out.min(), out.max())
                            wandb.log(
                                {
                                    "train/window_avg_loss": window_avg,
                                    "train/global_step": self.global_step,
                                    "train/recon_l1": window_recon_avg,
                                    "train/max_err_single_batch": max_err,
                                    "train/out_range": out_range,
                                },
                            )

                        if window_avg < self.best_train_loss:
                            self.best_train_loss = window_avg
                            self.save_model(filename="best_train_window.pt")

                    # reset window stats after each check
                    window_loss_sum = 0.0
                    window_loss_count = 0
                    window_recon_sum = 0.0

            # update tqdm postfix with current (unscaled) loss
            if self.is_main_process:
                iterator.set_postfix(loss=float(raw_loss.item()))

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
        recons = []
        for batch in iterator:
            b_num += 1
            x = batch.to(self.device, non_blocking=True)

            # AMP is still useful in eval for memory/speed
            if self.use_amp:
                with autocast(device_type="cuda", dtype=self.amp_dtype):
                    out = self.model(x)  # eval mode → only out
                    loss = F.mse_loss(out, x)
                    recon_l1 = F.l1_loss(out, x, reduction='mean').item()
            else:
                out = self.model(x)
                loss = F.mse_loss(out, x)
                recon_l1 = F.l1_loss(out, x, reduction='mean').item()


            losses.append(loss.item())
            recons.append(recon_l1)
            if self.is_main_process:
                iterator.set_postfix(val_loss=float(loss.item()))

            # These functions are dependent on it being a 3D generation
            if self.use_wandb and self.is_main_process and b_num == 1 and self.dims == 3:
                # x, out: (N, C, D, H, W)
                vol_in = x[0, 0]   # (D, H, W)
                vol_out = out[0, 0]

                # ---- 2D middle slice images (what you already had) ----
                mid = vol_in.shape[0] // 2
                x_slice = prepare_for_wandb_hu(vol_in[mid])
                out_slice = prepare_for_wandb_hu(vol_out[mid])

                wandb.log({
                    "recon/inputs_mid_slice": wandb.Image(x_slice),
                    "recon/outputs_mid_slice": wandb.Image(out_slice),
                })

                if True: # epoch % 5 == 0:
                    # ---- 3D scrollable GIF: input volume ----
                    input_frames = volume_to_gif_frames(vol_in, every_n=2)  # every 2 slices, tweak as needed
                    input_gif_path = save_gif(input_frames, fps=10)
                    input_mp4_path = save_mp4(input_frames, fps=10)

                    # ---- 3D scrollable GIF: side-by-side (input | recon) ----
                    recon_frames = volume_to_gif_frames(vol_out, every_n=2)
                    output_gif_path = save_gif(recon_frames, fps=10)
                    output_mp4_path = save_mp4(recon_frames, fps=10)
                    # ensure same length
                    n_frames = min(len(input_frames), len(recon_frames))
                    side_by_side_frames = [
                        np.concatenate([input_frames[i], recon_frames[i]], axis=1)
                        for i in range(n_frames)
                    ]
                    side_gif_path = save_gif(side_by_side_frames, fps=3)
                    side_mp4_path = save_mp4(side_by_side_frames, fps=10)

                    wandb.log({
                        "recon/volume_scroll_input": wandb.Video(input_gif_path, format="gif"),
                        "recon/volume_scroll_side_by_side": wandb.Video(side_gif_path, format="gif"),
                        "recon/volume_scroll_output": wandb.Video(output_gif_path,format="gif"),
                    })

                    wandb.log({
                        "recon/volume_scroll_input_mp4": wandb.Video(input_mp4_path, format="mp4"),
                        "recon/volume_scroll_side_by_side_mp4": wandb.Video(side_mp4_path, format="mp4"),
                        "recon/volume_scroll_output_mp4": wandb.Video(output_mp4_path, format="mp4"),
                    })

                    safe_delete(input_gif_path)
                    safe_delete(side_gif_path)
                    safe_delete(input_mp4_path)
                    safe_delete(side_mp4_path)
                    safe_delete(output_gif_path)
                    safe_delete(output_mp4_path)

                    ## Now code to check the smoothness of the latent space
                    if self.is_distributed:
                        encoder = self.model.module.encode
                        decoder = self.model.module.decode
                    else:
                        encoder = self.model.encode
                        decoder = self.model.decode
                    x_a = x[0:1,:,:,:,:]
                    x_b = None
                    if x.shape[0] >= 2:
                        x_b = x[1:2,:,:,:,:]
                    with torch.no_grad():
                        if self.use_amp:
                            with torch.amp.autocast():
                                z_a, stnd_a, eps_a, _ = encoder(x_a)
                                if x_b is not None:
                                    z_b, stnd_b, eps_b, _ = encoder(x_b)
                        else:
                            z_a, stnd_a, eps_a, _ = encoder(x_a)
                            if x_b is not None:
                                z_b, stnd_b, eps_b, _ = encoder(x_b)
                        decoded_slices = []
                        labels = []
                        interpolated_slices = []
                        interpolated_labels = []

                        # This test is the latent smooth locally around an image
                        for sig in [0.0, 0.01, .25, 0.5, 1.0, 2.0]:
                            z2 = z_a + stnd_a * sig * torch.randn_like(z_a)
                            if self.use_amp:
                                with torch.amp.autocast():
                                    x_dec = decoder(z2)
                            else:
                                x_dec = decoder(z2)

                            vol = x_dec[0, 0]              # (D, H, W)
                            mid = vol.shape[0] // 2
                            
                            img = prepare_for_wandb_hu(vol[mid])   # returns 2D array for wandb
                            decoded_slices.append(img)
                            labels.append(f"σ={sig:g}")

                        plt_fig = log_slice_montage(decoded_slices, labels)
                        wandb.log({"latent/local_noise_mid_slice": wandb.Image(plt_fig)})
                        plt.close(plt_fig)

                        # Interpolate between z_a and z_b
                        # This test is to check are there smooth latents between two samples in the dataset
                        # Very important for the downstream task of predicting one image based off an earlier one
                        if z_b is not None:
                            for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
                                z_interp = (1 - alpha) * z_a + alpha * z_b
                                if self.use_amp:
                                    with torch.amp.autocast():
                                        x_interp = decoder(z_interp)
                                else:
                                    x_interp = decoder(z_interp)
                                vol_interp = x_interp[0, 0]
                                mid_interp = vol_interp.shape[0] // 2
                                img_interp = prepare_for_wandb_hu(vol_interp[mid_interp])
                                interpolated_slices.append(img_interp)
                                interpolated_labels.append(f"α={alpha:g}")

                            plt_fig2 = log_slice_montage(interpolated_slices, interpolated_labels)
                            wandb.log({"latent/interpolation": wandb.Image(plt_fig2)})
                            plt.close(plt_fig2)

                        # Decode pure noise to see if it is a standard gaussion latent
                        gauss_noise = torch.randn_like(z_a)
                        dec_gauss = decoder(gauss_noise)
                        vol_gauss = dec_gauss[0, 0]
                        mid_gauss = vol_gauss.shape[0] // 2
                        img_gauss = prepare_for_wandb_hu(vol_gauss[mid_gauss])
                        wandb.log({"latent/gaussian_noise_decoded": wandb.Image(img_gauss)})


        loss_average = sum(losses) / max(len(losses), 1)
        recon_l1_average = sum(recons) / max(len(recons), 1)

        if dist.is_available() and dist.is_initialized():
            loss_tensor = torch.tensor(loss_average, device=self.device, dtype=torch.float32)
            loss_average = self.ddp_average(loss_tensor)
     
        return loss_average, recon_l1_average
    
    def beta_warmup(self, step):
        if step >= self.warmup_steps:
            return self.max_beta
        return self.max_beta * (step / self.warmup_steps)

    def train(self, train_loader, val_loader=None, num_epochs: int = 10,
              train_sampler=None, val_sampler=None):
        """
        High-level training loop.
        train_sampler: DistributedSampler in DDP (so we can call set_epoch).

        Note for full latent space theck the validation batch size must be 2 or greater  
        """
        history = {"train_loss": [], "val_loss": []}

        # warmup should work even if it is ddp because each loader will have fewer elements
        self.warmup_steps = .3 * num_epochs * len(train_loader)

        for epoch in range(num_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

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
                eval_loss, recon_loss = self.eval_epoch(val_loader, epoch=epoch, num_epochs=num_epochs)

                history["val_loss"].append(eval_loss)

                if self.is_main_process:
                    # print(f"[Epoch {epoch+1}/{num_epochs}] global_avg_val_recon_loss = {eval_loss:.4f}")

                    self.logger.info(
                                        "[Epoch %d/%d] global_avg_val_recon_loss = %.4f",
                                        epoch + 1, num_epochs, eval_loss,
                                    )
                    # --- save best model so far (based on validation loss) ---
                    if eval_loss is not None and eval_loss < self.best_val_loss:
                        self.best_val_loss = eval_loss
                        # you can include epoch in the name if you like
                        self.save_model(filename="best_eval_model.pt")
                    
            if self.use_wandb and self.is_main_process:
                log_dict = {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                }
                if eval_loss is not None:
                    log_dict["val/loss"] = eval_loss
                    log_dict["val/recon_l1"] = recon_loss

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

