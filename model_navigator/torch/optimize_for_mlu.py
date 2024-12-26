# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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
"""Torch optimize API."""

import pathlib
from typing import Optional, Sequence, Tuple, Type, Union

import torch  # pytype: disable=import-error

from model_navigator.configuration import (
    DEFAULT_TORCH_TARGET_FORMATS,
    DEFAULT_TORCH_TARGET_FORMATS_FOR_MLU,
    CustomConfig,
    DeviceKind,
    Format,
    OptimizationProfile,
    SizedDataLoader,
    VerifyFunction,
    map_custom_configs,
)
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.constants import DEFAULT_SAMPLE_COUNT
from model_navigator.configuration.model.model_config_builder import ModelConfigBuilder
from model_navigator.core.logger import LOGGER
from model_navigator.frameworks import Framework
from model_navigator.package import Package
from model_navigator.pipelines.builders import (
    correctness_builder,
    performance_builder,
    preprocessing_builder,
    tensorrt_conversion_builder,
    magicmind_conversion_builder,
    torch_conversion_builder,
    torch_export_builder,
    torch_tensorrt_conversion_builder,
    verify_builder,
)
from model_navigator.pipelines.builders.find_device_max_batch_size import find_device_max_batch_size_builder
from model_navigator.pipelines.builders.torch import torch_dynamo_onnx_builder, torch_exportedprogram_builder
from model_navigator.pipelines.wrappers.optimize import optimize_pipeline
from model_navigator.runners.base import NavigatorRunner
from model_navigator.runners.utils import default_runners, filter_runners
from model_navigator.utils import enums


def optimize(
    model: torch.nn.Module,
    dataloader: SizedDataLoader,
    sample_count: Optional[int] = DEFAULT_SAMPLE_COUNT,
    batching: Optional[bool] = True,
    input_names: Optional[Tuple[str, ...]] = None,
    output_names: Optional[Tuple[str, ...]] = None,
    target_formats: Optional[Tuple[Union[str, Format], ...]] = None,
    target_device: Optional[DeviceKind] = DeviceKind.MLU,
    runners: Optional[Tuple[Union[str, Type[NavigatorRunner]], ...]] = None,
    optimization_profile: Optional[OptimizationProfile] = None,
    workspace: Optional[pathlib.Path] = None,
    verbose: Optional[bool] = False,
    debug: Optional[bool] = False,
    verify_func: Optional[VerifyFunction] = None,
    custom_configs: Optional[Sequence[CustomConfig]] = None,
) -> Package:
    """

    该 API 能够将格式为 torch.nn.module 的 model 自动化转换为能够在 mlu370 上运行的 format

    具体包括自动执行 导出、转换、数值正确性校验、性能采样、以及模型校验

    Args:
        模型相关：
        model: PyTorch model object

        输入样本相关：
        dataloader: Sized iterable with data that will be feed to the model
        sample_count: Limits how many samples will be used from dataloader
        batching: Enable or disable batching on first (index 0) dimension of the model
        input_names: Model input names
        output_names: Model output names

        # 目标设备与格式
        target_formats: Target model formats for optimize process
        target_device: Target device for optimize process, default is CUDA

        # 其他可用参数配置
        runners: Use only runners provided as parameter
        optimization_profile: Optimization profile for conversion and profiling
        workspace: Workspace where packages will be extracted
        verbose: Enable verbose logging
        debug: Enable debug logging from commands
        verify_func: Function for additional model verification
        custom_configs: Sequence of CustomConfigs used to control produced artifacts

    Returns:
        Package descriptor representing created package.
    """

    ################################################
    ## 参数检查, 并提供默认选择 ######################
    ################################################

    # 确定 target formats
    if target_formats is None:
        target_formats = DEFAULT_TORCH_TARGET_FORMATS_FOR_MLU
        LOGGER.info(f"Using default target formats: {[tf.name for tf in target_formats]}")

    # TODO 这里的 default_runner 是干嘛用的？ #########################################################
    if runners is None:
        runners = default_runners(device_kind=target_device)
    else:
        runners = filter_runners(runners, device_kind=target_device)

    # 将 target_format 和 runner_names 转换为正确的 对象元组
    target_formats = enums.parse(target_formats, Format)
    runner_names = enums.parse(runners, lambda runner: runner if isinstance(runner, str) else runner.name())

    # 创建用于优化流程的 profile 配置
    if optimization_profile is None:
        optimization_profile = OptimizationProfile()

    # source formats
    if Format.TORCH not in target_formats:
        target_formats = (Format.TORCH,) + target_formats

    config = CommonConfig(
        framework=Framework.TORCH,
        model=model,
        dataloader=dataloader,
        target_formats=target_formats,
        sample_count=sample_count,
        _input_names=input_names,
        _output_names=output_names,
        target_device=target_device,
        batch_dim=0 if batching else None,
        runner_names=runner_names,
        optimization_profile=optimization_profile,
        verbose=verbose,
        debug=debug,
        verify_func=verify_func,
        custom_configs=map_custom_configs(custom_configs=custom_configs),
    )

    #######################################################################
    ###### 构建转换路径上所有 model #### ####################################
    #######################################################################
    # 根据 target_device 和 target_format 决定转换路径上所有可能的 model
    # 用 model_configs 来表示这些 model，model_config 还决定了 export 或者 convert 时使用的参数
    # 如果用户提供了 custom_config，则使用 custom_config
    models_config = ModelConfigBuilder.generate_model_config(
        framework=Framework.TORCH,
        target_formats=target_formats,
        custom_configs=custom_configs,
    )

    builders = [
        # 推理出 input_metadata 和 output_metadata，
        # 保存 input_sample 和 output_sample
        preprocessing_builder, #
        # 使用 model（model_path）、input/output_metadata、batch_dim、model_config 导出模型
        torch_export_builder, #
        # 使用 runner 求出可能的 device max batch size
        # find_device_max_batch_size_builder, #
        # torch_exportedprogram_builder, #
        # torch_dynamo_onnx_builder, #
        # torch_conversion_builder, #
        # torch_tensorrt_conversion_builder, #
        # tensorrt_conversion_builder, #
        magicmind_conversion_builder, # 将 onnx 转换为 magicmind

        # # 使用 runner 进行 Model 的正确性校验与性能采样
        # correctness_builder, #
        # performance_builder, #
    ]
    if verify_func:
        builders.append(verify_builder)

    package = optimize_pipeline(
        model=model,
        workspace=workspace,
        builders=builders,
        config=config,
        models_config=models_config,
    )

    return package
