
from __future__ import annotations

import torch.nn as nn
from .upsample import SubpixelUpsample



class DecoderUpsampleBlock3D_MONAI(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor=2,
        groups=16,
        act=nn.GELU
    ):
        super().__init__()

        # 1. Learned upsampling via PixelShuffle3D wrapped by MONAI
        self.up = SubpixelUpsample(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            scale_factor=scale_factor,
        )

        # 2. Post-shuffle refinement
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(groups, out_channels),
                         num_channels=out_channels),
            act(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(groups, out_channels),
                         num_channels=out_channels),
            act(),
        )

    def forward(self, x):
        x = self.up(x)     # Subpixel upsample (learned)
        x = self.conv(x)   # brightness + detail restoration
        return x