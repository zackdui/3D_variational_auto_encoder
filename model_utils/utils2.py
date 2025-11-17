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
# Significant changes include: Removing most of the file and changing placement
# This file is therefore a modified version of the original MONAI file.
# -------------------------------------------------------------------------

"""
Utilities and types for defining networks, these depend on PyTorch.
"""

from __future__ import annotations


import torch
import torch.nn as nn

from model_utils.utils.module import optional_import

onnx, _ = optional_import("onnx")
onnxreference, _ = optional_import("onnx.reference")
onnxruntime, _ = optional_import("onnxruntime")
polygraphy, polygraphy_imported = optional_import("polygraphy")
torch_tensorrt, _ = optional_import("torch_tensorrt", "1.4.0")

__all__ = [
    "icnr_init",
    "pixelshuffle",
    "has_nvfuser_instance_norm",
]

_has_nvfuser = None

def has_nvfuser_instance_norm():
    """whether the current environment has InstanceNorm3dNVFuser
    https://github.com/NVIDIA/apex/blob/23.05-devel/apex/normalization/instance_norm.py#L15-L16
    """
    global _has_nvfuser
    if _has_nvfuser is not None:
        return _has_nvfuser

    _, _has_nvfuser = optional_import("apex.normalization", name="InstanceNorm3dNVFuser")
    if not _has_nvfuser:
        return False
    try:
        import importlib

        importlib.import_module("instance_norm_nvfuser_cuda")
    except ImportError:
        _has_nvfuser = False
    return _has_nvfuser




def icnr_init(conv, upsample_factor, init=nn.init.kaiming_normal_):
    """
    ICNR initialization for 2D/3D kernels adapted from Aitken et al.,2017 , "Checkerboard artifact free
    sub-pixel convolution".
    """
    out_channels, in_channels, *dims = conv.weight.shape
    scale_factor = upsample_factor ** len(dims)

    oc2 = int(out_channels / scale_factor)

    kernel = torch.zeros([oc2, in_channels] + dims)
    kernel = init(kernel)
    kernel = kernel.transpose(0, 1)
    kernel = kernel.reshape(oc2, in_channels, -1)
    kernel = kernel.repeat(1, 1, scale_factor)
    kernel = kernel.reshape([in_channels, out_channels] + dims)
    kernel = kernel.transpose(0, 1)
    conv.weight.data.copy_(kernel)


def pixelshuffle(x: torch.Tensor, spatial_dims: int, scale_factor: int) -> torch.Tensor:
    """
    Apply pixel shuffle to the tensor `x` with spatial dimensions `spatial_dims` and scaling factor `scale_factor`.

    See: Shi et al., 2016, "Real-Time Single Image and Video Super-Resolution
    Using a nEfficient Sub-Pixel Convolutional Neural Network."

    See: Aitken et al., 2017, "Checkerboard artifact free sub-pixel convolution".

    Args:
        x: Input tensor with shape BCHW[D]
        spatial_dims: number of spatial dimensions, typically 2 or 3 for 2D or 3D
        scale_factor: factor to rescale the spatial dimensions by, must be >=1

    Returns:
        Reshuffled version of `x`.

    Raises:
        ValueError: When input channels of `x` are not divisible by (scale_factor ** spatial_dims)
    """
    dim, factor = spatial_dims, scale_factor
    input_size = list(x.size())
    batch_size, channels = input_size[:2]
    scale_divisor = factor**dim

    if channels % scale_divisor != 0:
        raise ValueError(
            f"Number of input channels ({channels}) must be evenly "
            f"divisible by scale_factor ** dimensions ({factor}**{dim}={scale_divisor})."
        )

    org_channels = int(channels // scale_divisor)
    output_size = [batch_size, org_channels] + [d * factor for d in input_size[2:]]

    indices = list(range(2, 2 + 2 * dim))
    indices = indices[dim:] + indices[:dim]
    permute_indices = [0, 1]
    for idx in range(dim):
        permute_indices.extend(indices[idx::dim])

    x = x.reshape([batch_size, org_channels] + [factor] * dim + input_size[2:])
    x = x.permute(permute_indices).reshape(output_size)
    return x

