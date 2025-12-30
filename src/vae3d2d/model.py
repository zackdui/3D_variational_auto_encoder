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
# Significant changes include: Removing skip connections, Adding Attention, Adjusting for pure VAE, Many additional inputs.
# Original file is barely recognizable.
# This file is therefore a modified version of the original MONAI file.
# -------------------------------------------------------------------------

# This file defines the custom VAE Model

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from dataclasses import dataclass
from typing import Optional, Tuple
import os
import dataclasses
from monai.losses import SSIMLoss

from .model_utils.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from .model_utils.layers.factories import Dropout
from .model_utils.layers.utils import get_act_layer, get_norm_layer
from .model_utils.utils import UpsampleMode
from .model_utils.blocks import SelfAttentionND
from .model_utils.blocks.upsample_new import DecoderUpsampleBlock3D_MONAI
from .loss_functions import Eagle_Loss_3D, gradient_loss_3d, focal_frequency_loss_3d

# New refine block to try
# refine = nn.Sequential(
#     nn.Conv3d(c_out, c_out, kernel_size=3, padding=1, bias=False),
#     nn.GroupNorm(num_groups=min(16, c_out), num_channels=c_out),
#     nn.GELU(),
#     nn.Conv3d(c_out, c_out, kernel_size=3, padding=1, bias=False),
#     nn.GroupNorm(num_groups=min(16, c_out), num_channels=c_out),
#     nn.GELU(),
# )

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

loss_strings_to_classes = {
    "ssim_3d": SSIMLoss(spatial_dims=3, data_range=2.0),
    "eagle_loss_3d": Eagle_Loss_3D(patch_size=3, cpu=False, cutoff=0.5),
    "gradient_loss_3d": gradient_loss_3d,
    "focal_frequency_loss_3d": focal_frequency_loss_3d,
    'mse': nn.MSELoss(reduction='none'),
    'l1': nn.L1Loss(reduction='none'),
}

ACTIVATION_REGISTRY = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "softplus": nn.Softplus,
}

def maybe_checkpoint(module, x, use_checkpoint: bool, is_training: bool):
    # Only checkpoint if:
    #  - user enabled it
    #  - we're in training mode
    #  - this tensor participates in grad computation
    do_ckpt = use_checkpoint and is_training and x.requires_grad
    if not do_ckpt:
        return module(x)

    def forward_fn(t):
        return module(t)
    return cp.checkpoint(forward_fn, x, use_reentrant=False)

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
        model_name: name of the model to be returned by get_name()
        vae_latent_channels: Number of channels to have in the latent representation
        use_log_var: If this is true, it will use logvar for std instead of std directly
        beta: In vae loss what beta to apply
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 8.
        smallest_filters: number of channels at the bottleneck of the VAE before latent channels. Defaults to 64.
        in_channels: number of input channels for the network. Defaults to 1.
        dropout_prob: probability of an element to be zero-ed. Defaults to ``None``.
        act: activation type and arguments. Defaults to ``RELU``.
            ex. ("GELU", {"approximate": "tanh"}) or ("GELU", {"approximate": "none"})
        norm: feature normalization type and arguments. Defaults to ``GROUP``.
        custom_losses: A list of loss functions to add to the total loss
            - Each loss function should take in (input, target) and return a tensor loss
            - Each loss is a string key from loss_strings_to_classes which maps to a loss function
            - valid strings are: 'ssim_3d', 'eagle_loss_3d', 'gradient_loss_3d', 'focal_frequency_loss_3d', 'mse', 'l1'
            - defaults to ['l1']
        custom_loss_weights: A list of weights for each custom loss function
        final_activation: Final activation to apply to the output. Defaults to "tanh". Should be a string key from ACTIVATION_REGISTRY or None for no activation.
            - Valid strings are: 'relu', 'leaky_relu', 'gelu',  'sigmoid', 'tanh', 'elu', 'softplus'
        blocks_down: number of down sample blocks in each layer. Defaults to ``[1,2,2,4]``.
        blocks_up: number of up sample blocks in each layer. Defaults to ``[1,1,1]``. Must have one number less than blocks down
            For none of either blocks_down=(1,) and blocks_up=()
        downsample_strides: sequence of downsample strides for each down block past the first. If None, will use (2,2,2) for each down block except first.
            - Strides of different dimensions for depth vs height and width can not be used with any pixelshuffle upsample_mode
        vae_down_stride: downsample stride for the vae down layer. Defaults to (2, 2, 2).
        res_block_weight: the weight to put on the actual network of all res_blocks
        upsample_mode: [``"deconv"``, ``"nontrainable"``, ``"pixelshuffle"``, ``"pixelshuffle_v2"``]
            The mode of upsampling manipulations.
            Using the ``nontrainable`` modes cannot guarantee the model's reproducibility. Defaults to``nontrainable``.

            - ``deconv``, uses transposed convolution layers.
            - ``nontrainable``, uses non-trainable `linear` interpolation.
            - ``pixelshuffle``, uses :py:class:`monai.networks.blocks.SubpixelUpsample`.
            - ``pixelshuffle_v2``, uses :py:class:`DecoderUpsampleBlock3D_MONAI`.
        use_attn: Whether to add an attention layer before producing mean and variance. Defaults to False
        attn_params: takes in AttnParams; the parameters for the attention layer if it is used
            window_size = None is for global attention
        use_checkpoint: Whether to use gradient checkpointing to save memory. Defaults to False.
        debug_mode: Whether to print shape information or not
    """

    def __init__(
        self,
        model_name: str = "default_vae",
        vae_latent_channels: int = 4,
        vae_use_log_var: bool = False,
        beta: float = 1.0,
        spatial_dims: int = 3,
        init_filters: int = 8,
        smallest_filters: int = 64,
        in_channels: int = 1,
        dropout_prob: float | None = None,
        act: str | tuple = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        num_groups: int = 8,
        custom_losses = None,
        custom_loss_weights = None,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        downsample_strides: Optional[Sequence[Tuple[int, ...]]] = None,
        vae_down_stride: Optional[Tuple[int, ...]] = None,
        res_block_weight: float = .01,
        final_activation: Optional[str] = "tanh",
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
        use_attn: bool = False,
        attn_params: AttnParams = AttnParams(),
        use_checkpoint: bool = False,
        debug_mode: bool = False,
    ):
        super().__init__()
        # torch.autograd.set_detect_anomaly(True)

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.model_name = model_name
        self.debug_mode = debug_mode
        self.use_checkpoint = use_checkpoint
        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.blocks_down = blocks_down

        self.downsample_strides = downsample_strides
        self.vae_down_stride = vae_down_stride
        # asserts to make sure inputs are valid
        assert len(blocks_up) == len(blocks_down) - 1, "blocks_up must have one less element than blocks_down"
        assert len(self.downsample_strides) == len(self.blocks_down) - 1, "downsample_strides must have one less element than blocks_down"
        
        if self.downsample_strides is None:
            self.downsample_strides = [(2, 2, 2)] * (len(self.blocks_down) - 1)
        if self.vae_down_stride is None:
            self.vae_down_stride = (2, 2, 2)

        self.res_block_weight = res_block_weight

        self.latent_filters = vae_latent_channels
        self.smallest_filters = smallest_filters
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_attn = use_attn
        if isinstance(attn_params, dict):
            attn_params = AttnParams(**attn_params)
        self.attn_params = attn_params
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters) # (B, C_out, D, H, W)
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()

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

        # Create custom loss functions
        if custom_losses is None:
            custom_losses = ["l1"]
        if custom_loss_weights is None:
            custom_loss_weights = [1.0]
        self.custom_losses = custom_losses
        self.loss_functions = []
        for loss_func in self.custom_losses:
            if isinstance(loss_func, str):
                if loss_func.lower() in loss_strings_to_classes:
                    self.loss_functions.append(loss_strings_to_classes[loss_func.lower()])
                else:
                    raise ValueError(f"Unknown loss function string: {loss_func}")
        self.custom_loss_weights = custom_loss_weights
        self.use_log_var = vae_use_log_var
        self.beta = beta
        
        self._prepare_vae_modules()
        self.vae_conv_final = self._make_final_conv(in_channels)
        self.final_activation = final_activation
        if self.final_activation is None or str(self.final_activation).lower() == "none":
            self.final_act_mod = nn.Identity()
        elif self.final_activation.lower() in ACTIVATION_REGISTRY:
            self.final_act_mod = ACTIVATION_REGISTRY[self.final_activation.lower()]()
        else:
            raise ValueError(f"Unsupported final activation: {self.final_activation}")


        self.hparams = dict(
            model_name=model_name,
            vae_latent_channels=vae_latent_channels,
            vae_use_log_var=vae_use_log_var,
            beta=beta,
            spatial_dims=spatial_dims,
            init_filters=init_filters,
            smallest_filters=self.smallest_filters,
            in_channels=in_channels,
            dropout_prob=dropout_prob,
            act=act,
            norm=self.norm,  # already post-processed
            num_groups=num_groups,
            custom_losses=self.custom_losses,
            custom_loss_weights=self.custom_loss_weights,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            downsample_strides=self.downsample_strides,
            vae_down_stride=self.vae_down_stride,
            res_block_weight=self.res_block_weight,
            final_activation=final_activation,
            upsample_mode=str(self.upsample_mode.value),  # constructor accepts string too
            use_attn=use_attn,
            attn_params=dataclasses.asdict(self.attn_params),
            use_checkpoint=self.use_checkpoint,
            debug_mode=debug_mode
        )

    def get_hparams(self):
        return self.hparams

    def get_name(self):
        return self.model_name

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2**i
            if i > 0:
                stride_i = self.downsample_strides[i - 1]
            pre_conv = (
                get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=stride_i)
                if i > 0
                else nn.Identity()
            )
            # Res Blocks will not change dimensions just add blocks
            down_layer = nn.Sequential(
                pre_conv, *[ResBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act, net_weight=self.res_block_weight) for _ in range(item)]
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
                        ResBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act, net_weight=self.res_block_weight)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            if self.upsample_mode != UpsampleMode.PIXELSHUFFLE_V2:
                stride_ind = n_up - i - 1
                stride_up = self.downsample_strides[stride_ind]
                up_samples.append(
                    nn.Sequential(
                        *[
                            get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                            get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode, scale_factor=stride_up),
                        ]
                    )
                )
            else:
                up_samples.append(
                    DecoderUpsampleBlock3D_MONAI(
                        in_channels=sample_in_channels,
                        out_channels=sample_in_channels // 2,
                        scale_factor=2,
                        act=self.act_mod,   # use your activation
                    )
                )
        return up_layers, up_samples
    
    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            # get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            # self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )
    
    def _prepare_vae_modules(self):
        zoom = 2 ** (len(self.blocks_down) - 1)
        v_filters = self.init_filters * zoom
       
        self.vae_down = nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=v_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, v_filters, self.smallest_filters, stride=self.vae_down_stride, bias=True),
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.smallest_filters),
            self.act_mod,
        )

        if self.upsample_mode != UpsampleMode.PIXELSHUFFLE_V2:
            self.vae_fc_up_sample = nn.Sequential(
                get_conv_layer(self.spatial_dims, self.smallest_filters, v_filters, kernel_size=1),
                get_upsample_layer(self.spatial_dims, v_filters, upsample_mode=self.upsample_mode, scale_factor=self.vae_down_stride),
                get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=v_filters),
                self.act_mod,
            )
        else:
            self.vae_fc_up_sample = DecoderUpsampleBlock3D_MONAI(
                in_channels=self.smallest_filters,
                out_channels=v_filters,
                scale_factor=2,
                act=nn.GELU,  # or whatever you're using for self.act_mod
            )


        ## Added a conv layer for mean and conv layer for std
        self.vae_mean_layer = get_conv_layer(self.spatial_dims, self.smallest_filters, self.latent_filters, kernel_size=1)
        self.vae_std_layer = get_conv_layer(self.spatial_dims, self.smallest_filters, self.latent_filters, kernel_size=1)

        self.latent_proj =nn.Sequential(
                            get_conv_layer(self.spatial_dims, self.latent_filters, self.smallest_filters, kernel_size=1),
                            self.act_mod,
                        )
        
    def get_loss(self, outputs, targets):
        total_custom_loss = 0.0
        for loss_func, weights in zip(self.loss_functions, self.custom_loss_weights):
            loss_value = loss_func(outputs, targets)

            # If the loss is not a scalar take the mean
            if hasattr(loss_value, "ndim") and loss_value.ndim > 0:
                loss_value = loss_value.mean()

            total_custom_loss += weights * loss_value
        return total_custom_loss

    def forward(self, x):
        """
        In train mode it will return the decoded image and the loss
        In eval mode it will just return the decoded image
        """
        net_input = x
        z_mean, z_sigma, eps, logvar = self.encode(x)
        z_sample = self.sample(z_mean, z_sigma, eps)
        # In debug mode remove sampling for easier debugging
        if self.debug_mode:
            decoded = self.decode(z_mean)
        else:
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
                kl = kl_elem.view(kl_elem.size(0), -1).mean(dim=1).mean()  # MEAN over latent elems
            else: 
                var = z_sigma.pow(2)
                logvar = 2.0 * torch.log(z_sigma)
                kl_per_elem = 0.5 * (z_mean.pow(2) + var - logvar - 1)

                kl = kl_per_elem.view(kl_per_elem.size(0), -1).mean()

        if self.training:     
            recon_loss = self.get_loss(decoded, net_input)

            vae_loss = recon_loss + self.beta * kl

            return decoded, vae_loss
        return decoded
    
    def encode(self, x):
        # Input to the first layer should be (B, C, D, H, W)
        if self.debug_mode:
            print("Before full encode:", x.shape)
        x = maybe_checkpoint(self.convInit, x, self.use_checkpoint, self.training)
        if self.debug_mode:
            print("After init encode:", x.shape)
        if self.dropout_prob is not None:
            x = self.dropout(x)
        if self.debug_mode:
            print("After dropout encode:", x.shape)

        for down in self.down_layers:
            x = maybe_checkpoint(down, x, self.use_checkpoint, self.training)
            if self.debug_mode:
                print("After down layer encode:", x.shape)
        if self.debug_mode:
            print("after main encode:", x.shape)
        # x_vae = x
        x_vae = maybe_checkpoint(self.vae_down, x, self.use_checkpoint, self.training)
        if self.debug_mode:
            print("Shape After vae down:", x_vae.shape)
        if self.use_attn:
            x_vae = maybe_checkpoint(self.attn_layer, x_vae, self.use_checkpoint, self.training)
            if self.debug_mode:
                print("Shape After attn layer:", x_vae.shape)
        z_mean = maybe_checkpoint(self.vae_mean_layer, x_vae, self.use_checkpoint, self.training)
        if self.debug_mode:
            print("Z_mean shape after vae_mean_layer:", z_mean.shape)
        z_sigma = maybe_checkpoint(self.vae_std_layer, x_vae, self.use_checkpoint, self.training)

        z_mean_rand = torch.randn_like(z_mean)
        z_mean_rand.requires_grad_(False)

        # Consider adding some epsilon to maintain numerical stability
        if self.use_log_var:
            log_var = z_sigma
            log_var = torch.clamp(log_var, min=-30.0, max=20.0)  # typical safe range
            std     = torch.exp(0.5 * z_sigma)   # σ = exp(.5 * logvar)
        else:
            std = F.softplus(z_sigma) + 1e-6
        return z_mean, std, z_mean_rand, log_var if self.use_log_var else None
       

    
    def decode(self, x):
        x_vae = maybe_checkpoint(self.latent_proj, x, self.use_checkpoint, self.training)
        x_vae = self.vae_fc_up_sample(x_vae)

        for up, upl in zip(self.up_samples, self.up_layers):
            x_vae = maybe_checkpoint(up, x_vae, self.use_checkpoint, self.training)
            x_vae = maybe_checkpoint(upl, x_vae, self.use_checkpoint, self.training)

        x_vae = maybe_checkpoint(self.vae_conv_final, x_vae, self.use_checkpoint, self.training)

        x_vae = self.final_act_mod(x_vae)

        return x_vae

    def sample(self, z_mean, z_std, eps=None):
        """
        This function samples a z given a mean and sigma. It can be used with 
        presampled noise or if None, it will sample the noise.
        """
        if eps is None:
            eps = torch.randn_like(z_std)
        x_vae = z_mean + z_std * eps
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