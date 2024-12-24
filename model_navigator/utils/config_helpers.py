# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Find Max Batch size pipelines builders."""

from typing import Dict, List

from model_navigator.configuration import DeviceKind, Format
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config import (
    ModelConfig,
)
from model_navigator.core.logger import LOGGER


def do_find_device_max_batch_size(config: CommonConfig, models_config: Dict[Format, List[ModelConfig]]) -> bool:
    """Verify find max batch size is required.
    功能：依据以下条件判断是否需要进行 device max batch size 的搜索
    1. 目标设备是否是 cuda，仅在目标设备是 cuda 时进行搜索；
    2. 仅 model configs 中存在 adaptiv format 时进行搜索。不太清楚什么叫做 adptive，估计是支持动态类型的？
    3. model 支持 batch_dim 时

    Args:
        config: A configuration for pipelines
        models_config: List of model configs per format

    Returns:
        True if search is required, false, otherwise
    """
    model_formats = models_config.keys()
    # 什么叫做 adapative formats？
    adaptive_formats = {Format.TORCH_TRT, Format.TENSORRT, Format.TF_TRT, Format.TORCH_EXPORTEDPROGRAM}

    matching_formats = adaptive_formats.intersection(set(model_formats))
    # 如果 device 不是 cuda，或者不存在需要进行 batch_size 搜索的 format，就不搜索
    if len(matching_formats) == 0 or config.target_device != DeviceKind.CUDA:
        LOGGER.debug("No matching formats found")
        return False

    run_search = config.batch_dim is not None

    if not run_search:
        LOGGER.debug("Run search disabled.")
        return False

    return True
