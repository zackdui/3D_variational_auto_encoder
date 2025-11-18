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
# Significant changes include: Removing most of the file 
# This file is therefore a modified version of the original MONAI file.
# -------------------------------------------------------------------------

from __future__ import annotations
from enum import Enum

__all__ = [
    "InterpolateMode",
    "UpsampleMode",
]


class StrEnum(str, Enum):
    """
    Enum subclass that converts its value to a string.

    .. code-block:: python

        from monai.utils import StrEnum

        class Example(StrEnum):
            MODE_A = "A"
            MODE_B = "B"

        assert (list(Example) == ["A", "B"])
        assert Example.MODE_A == "A"
        assert str(Example.MODE_A) == "A"
        assert monai.utils.look_up_option("A", Example) == "A"
    """

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

class UpsampleMode(StrEnum):
    """
    See also: :py:class:`monai.networks.blocks.UpSample`
    """

    DECONV = "deconv"
    DECONVGROUP = "deconvgroup"
    NONTRAINABLE = "nontrainable"  # e.g. using torch.nn.Upsample
    PIXELSHUFFLE = "pixelshuffle"


class InterpolateMode(StrEnum):
    """
    See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
    """

    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    LINEAR = "linear"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    TRILINEAR = "trilinear"
    AREA = "area"

