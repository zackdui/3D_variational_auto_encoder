# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -------------------------------------------------------------------------
# Modifications made by Zack Duitz, 2025.
# Significant changes include: Removing skip connections, Adding Attention, Adjusting for pure VAE.
# This file is therefore a modified version of the original MONAI file.
# -------------------------------------------------------------------------

# This file defines the custom VAE Model

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
import os
import dataclasses

from model_utils.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from model_utils.layers.factories import Dropout
from model_utils.layers.utils import get_act_layer, get_norm_layer
from model_utils.utils import UpsampleMode
from model_utils.blocks import SelfAttentionND


__all__ = ["CustomVAE", "AttnParams"]

@dataclass
class AttnParams:
    """
    dim_head = None use C_enc // num_heads
    window_size = None is for global attention
    """
    num_heads: int = 3
    dim_head: Optional[int] = 32 
    dropout: float = 0.0
    window_size: Optional[Tuple[int, ...]] = None 
    use_rel_pos_bias: Optional[bool] = None


class CustomVAE(nn.Module):
    """
    CustomVAE is loosly based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    The module contains the variational autoencoder (VAE).
    The model supports 2D or 3D inputs.

    Every Down Layer past the first will divide each spatial dimension by 2.
    With Default parameters VAE_down (right before mean and std layers) will have (B, smallest_filters, D/16, H/16, W/16)
    Ouput of the encoder is (B, latent_filters, D/16, H/16, W/16) with defaults.

    Args:
        vae_latent_channels: Number of channels to have in the latent representation
        use_log_var: If this is true, it will use logvar for std instead of std directly
        beta: In vae loss what beta to apply
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 8.
        in_channels: number of input channels for the network. Defaults to 1.
        dropout_prob: probability of an element to be zero-ed. Defaults to ``None``.
        act: activation type and arguments. Defaults to ``RELU``.
        norm: feature normalization type and arguments. Defaults to ``GROUP``.
        blocks_down: number of down sample blocks in each layer. Defaults to ``[1,2,2,4]``.
        blocks_up: number of up sample blocks in each layer. Defaults to ``[1,1,1]``. Must have one number less than blocks down
        upsample_mode: [``"deconv"``, ``"nontrainable"``, ``"pixelshuffle"``]
            The mode of upsampling manipulations.
            Using the ``nontrainable`` modes cannot guarantee the model's reproducibility. Defaults to``nontrainable``.

            - ``deconv``, uses transposed convolution layers.
            - ``nontrainable``, uses non-trainable `linear` interpolation.
            - ``pixelshuffle``, uses :py:class:`monai.networks.blocks.SubpixelUpsample`.
        use_attn: Whether to add an attention layer before producing mean and variance. Defaults to False
        attn_params: takes in AttnParams; the parameters for the attention layer if it is used
            window_size = None is for global attention
        debug_mode: Whether to print shape information or not
    """

    def __init__(
        self,
        vae_latent_channels: int = 4,
        vae_use_log_var: bool = False,
        beta: float = 1.0,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        dropout_prob: float | None = None,
        act: str | tuple = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
        use_attn: bool = False,
        attn_params: AttnParams = AttnParams(),
        debug_mode: bool = False,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.debug_mode = debug_mode
        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.blocks_down = blocks_down
        self.latent_filters = vae_latent_channels
        self.smallest_filters = 16
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_attn = use_attn
        if isinstance(attn_params, dict):
            attn_params = AttnParams(**attn_params)
        self.attn_params = attn_params
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters) # (B, C_out, D, H, W)
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.conv_final = self._make_final_conv(self.out_channels)

        if self.spatial_dims == 2:
            self.attn_layer = SelfAttentionND(
                in_channels=self.smallest_filters,
                num_heads=self.attn_params.num_heads,
                dim_head=self.attn_params.dim_head,       
                dropout=self.attn_params.dropout,
                spatial_dims=2,
                window_size=self.attn_params.window_size,      # or None for global
                use_rel_pos_bias=self.attn_params.use_rel_pos_bias,
            )
        else:
            self.attn_layer = SelfAttentionND(
                in_channels=self.smallest_filters,
                num_heads=self.attn_params.num_heads,
                dim_head=self.attn_params.dim_head,      
                dropout=self.attn_params.dropout,
                spatial_dims=3,
                window_size=self.attn_params.window_size,    # global attention
                use_rel_pos_bias=self.attn_params.use_rel_pos_bias,
            )


        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

        
        self.use_log_var = vae_use_log_var
        self.beta = beta
        
        self._prepare_vae_modules()
        self.vae_conv_final = self._make_final_conv(in_channels)

        self.hparams = dict(
            vae_latent_channels=vae_latent_channels,
            vae_use_log_var=vae_use_log_var,
            beta=beta,
            spatial_dims=spatial_dims,
            init_filters=init_filters,
            in_channels=in_channels,
            dropout_prob=dropout_prob,
            act=act,
            norm=self.norm,  # already post-processed
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            upsample_mode=str(self.upsample_mode.value),  # constructor accepts string too
            use_attn=use_attn,
            attn_params=dataclasses.asdict(self.attn_params),
            debug_mode=debug_mode
        )
    def get_hparams(self):
        return self.hparams
    
    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2**i
            pre_conv = (
                get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                if i > 0
                else nn.Identity()
            )
            # Res Blocks will not change dimensions just add blocks
            down_layer = nn.Sequential(
                pre_conv, *[ResBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(item)]
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            up_layers.append(
                nn.Sequential(
                    *[
                        ResBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )
        return up_layers, up_samples
    
    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )
    
    def _prepare_vae_modules(self):
        zoom = 2 ** (len(self.blocks_down) - 1)
        v_filters = self.init_filters * zoom
       
        self.vae_down = nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=v_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, v_filters, self.smallest_filters, stride=2, bias=True),
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.smallest_filters),
            self.act_mod,
        )

        self.vae_fc_up_sample = nn.Sequential(
            get_conv_layer(self.spatial_dims, self.smallest_filters, v_filters, kernel_size=1),
            get_upsample_layer(self.spatial_dims, v_filters, upsample_mode=self.upsample_mode),
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=v_filters),
            self.act_mod,
        )

        ## Added a conv layer for mean and conv layer for std
        self.vae_mean_layer = get_conv_layer(self.spatial_dims, self.smallest_filters, self.latent_filters, kernel_size=1)
        self.vae_std_layer = get_conv_layer(self.spatial_dims, self.smallest_filters, self.latent_filters, kernel_size=1)

        self.latent_proj =nn.Sequential(
                            get_conv_layer(self.spatial_dims, self.latent_filters, self.smallest_filters, kernel_size=1),
                            self.act_mod,
                        )

    def forward(self, x):
        """
        In train mode it will return the decoded image and the loss
        In eval mode it will just return the decoded image
        """
        net_input = x
        z_mean, z_sigma, eps, logvar = self.encode(x)
        z_sample = self.sample(z_mean, z_sigma, eps)
        decoded = self.decode(z_sample)
        if self.debug_mode:
            print("Z_mean shape:", z_mean.shape)
            print("Z_sigma shape:", z_sigma.shape)
            print("Decoded shape:", decoded.shape)
            print("eps shape:", eps.shape)
        # Only in train mode return the loss
        if self.training:
            if self.use_log_var:
                # kl_per_elem = -0.5 * torch.sum(1 + logvar - z_mean.pow(2) - logvar.exp())
                kl_elem = -0.5 * (1 + logvar - z_mean.pow(2) - logvar.exp())
                # flatten all non-batch dims:
                kl_per_sample = kl_elem.view(kl_elem.size(0), -1).sum(dim=1)  # (B,)
                kl = kl_per_sample.mean()
            else: 
                var = z_sigma.pow(2)
                kl_per_elem = 0.5 * (z_mean.pow(2) + var - torch.log(1e-8 + var) - 1)

                kl_per_sample = kl_per_elem.view(kl_per_elem.size(0), -1).sum(dim=1)
                kl = kl_per_sample.mean()
             

            recon = F.mse_loss(net_input, decoded, reduction="none")
            recon = recon.view(recon.size(0), -1).mean(dim=1)         # per-sample mean
            recon_loss = recon.mean() 
            
            vae_loss =  recon_loss + self.beta * kl
            
            return decoded, vae_loss
        return decoded
    
    def encode(self, x):
        # Input to the first layer should be (B, C, D, H, W)
        if self.debug_mode:
            print("Before full encode:", x.shape)
        x = self.convInit(x)
        if self.debug_mode:
            print("After init encode:", x.shape)
        if self.dropout_prob is not None:
            x = self.dropout(x)
        if self.debug_mode:
            print("After dropout encode:", x.shape)

        for down in self.down_layers:
            x = down(x)
            if self.debug_mode:
                print("After down layer encode:", x.shape)
        if self.debug_mode:
            print("after main encode:", x.shape)

        x_vae = self.vae_down(x)
        if self.debug_mode:
            print("Shape After vae down:", x_vae.shape)
        if self.use_attn:
            x_vae = self.attn_layer(x_vae)

        z_mean = self.vae_mean_layer(x_vae)
        z_sigma = self.vae_std_layer(x_vae)

        z_mean_rand = torch.randn_like(z_mean)
        z_mean_rand.requires_grad_(False)

        if self.use_log_var:
            log_var = z_sigma
            std     = torch.exp(0.5 * z_sigma)   # σ = exp(.5 * logvar)
        else:
            std = F.softplus(z_sigma)
        return z_mean, std, z_mean_rand, log_var if self.use_log_var else None
       

    
    def decode(self, x):
        x_vae = self.latent_proj(x)
        x_vae = self.vae_fc_up_sample(x_vae)

        for up, upl in zip(self.up_samples, self.up_layers):
            x_vae = up(x_vae)
            x_vae = upl(x_vae)

        x_vae = self.vae_conv_final(x_vae)
        
        return x_vae

    def sample(self, z_mean, z_sigma, eps=None):
        """
        This function samples a z given a mean and sigma. It can be used with 
        presampled noise or if None, it will sample the noise.
        """
        if eps is None:
            eps = torch.randn_like(z_sigma)
        x_vae = z_mean + z_sigma * eps
        return x_vae
    

    def save_checkpoint(
        self,
        filename: str = "model.pt",
        extra: Optional[dict] = None,
        save_dir: Optional[str] = None,
    ) -> str:
        """
        Save model weights + hyperparameters (and optional extra info).

        Args:
            filename: name of the file to save, e.g. "vae.pt".
            extra: optional dict with optimizer state, epoch, metrics, etc.
            save_dir: optional override for save location. If None, uses
                      ./saved_models/ in the current working directory.

        Returns:
            Full path to the saved checkpoint.
        """

    
        # Default save directory = "<cwd>/saved_models"
        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), "saved_models")

        # Create directory if needed
        os.makedirs(save_dir, exist_ok=True)

        # Full path
        path = os.path.join(save_dir, filename)

        # Prepare checkpoint
        ckpt = {
            "hparams": self.hparams,
            "state_dict": self.state_dict(),
        }
        if extra is not None:
            ckpt["extra"] = extra

        # Save
        torch.save(ckpt, path)
        print(f"[CustomVAE] Saved checkpoint to: {path}")

        return path

    @classmethod
    def load_from_checkpoint(
        cls,
        path: str,
        map_location: Optional[torch.device | str] = None,
        **override_hparams,
    ) -> CustomVAE:
        """
        Recreate a CustomVAE from a checkpoint saved with `save_checkpoint`.

        You can override some hyperparameters at load time, e.g.:

            model = CustomVAE.load_from_checkpoint(
                "ckpt.pt",
                spatial_dims=2,
            )

        (Only do this if you know the architecture remains compatible.)
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=True)

        hparams = ckpt["hparams"].copy()
        hparams.update(override_hparams)

        # upsample_mode is stored as string -> CustomVAE accepts string or enum
        model = cls(**hparams)
        model.load_state_dict(ckpt["state_dict"])
        return model




if __name__ == "__main__":
    # Example Usage and Simple Test
    attn_params = AttnParams(window_size=(3, 8, 8), use_rel_pos_bias=True, dim_head=32)
    model = CustomVAE(spatial_dims=3, use_attn=True, attn_params=attn_params)
    moel_params = sum(p.numel() for p in model.parameters())
    # Shape is (B, C, D, H, W)
    x_input = torch.rand(size=(2, 1, 208, 512, 512))
    # x_input = torch.rand(size=(2, 1, 512, 512))
    img, loss = model(x_input)

    # model.save_checkpoint("vae.pt")
    # model.save_checkpoint(
    #         "vae_epoch12.pt",
    #         extra={"epoch": 12, "val_loss": 0.013}
    #     )
    # model.save_checkpoint(
    #         "custom_vae_2d.pt",
    #         extra={"optimizer": optimizer.state_dict(), "epoch": epoch},
    #     )
    # optimizer.load_state_dict(ckpt["extra"]["optimizer"])

    # model = CustomVAE.load_from_checkpoint("custom_vae_2d.pt", map_location="cuda")