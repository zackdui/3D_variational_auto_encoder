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
# Significant changes include: Removing most of the file and changed utils name
# This file is therefore a modified version of the original MONAI file.
# -------------------------------------------------------------------------

from __future__ import annotations
from .utils2 import has_nvfuser_instance_norm, icnr_init, pixelshuffle

