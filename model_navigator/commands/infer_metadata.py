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
"""Inputs and outputs metadata commands."""

import pathlib
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

from model_navigator.commands.base import Command, CommandOutput, CommandStatus
from model_navigator.commands.execution_context import ExecutionContext
from model_navigator.configuration import OptimizationProfile, SizedDataLoader, SizedIterable, TensorRTProfile
from model_navigator.configuration.runner.runner_config import RunnerConfig
from model_navigator.core.dataloader import extract_sample, load_samples, to_numpy, validate_sample_input
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import (
    FRAMEWORK_TO_TENSOR_TYPE,
    PyTreeMetadata,
    TensorMetadata,
    TensorSpec,
)
from model_navigator.core.workspace import Workspace
from model_navigator.exceptions import ModelNavigatorUserInputError
from model_navigator.frameworks import Framework, is_torch_available
from model_navigator.frameworks.onnx.utils import get_onnx_io_names
from model_navigator.frameworks.tensorrt.utils import get_tensorrt_io_names
from model_navigator.runners.base import NavigatorRunner
from model_navigator.utils import module
from model_navigator.utils.common import optimal_batch_size

torch = module.lazy_import("torch")


def _extract_axes_shapes(
    dataloader: Union[SizedDataLoader, Iterator],
    pytree_metadata: PyTreeMetadata,
    input_names: Sequence[str],
    input_ndims: Sequence[int],
    num_samples: int,
    framework: Framework,
    check_len: bool = True,
) -> Dict[str, Dict[int, List[int]]]:
    assert not (check_len) or isinstance(
        dataloader, (SizedIterable, Sequence)
    ), "dataloader is not an instance of SizedDataLoader, unable to check length."

    axes_shapes = {name: {ax: [] for ax in range(ndim)} for name, ndim in zip(input_names, input_ndims)}
    for i, sample in enumerate(dataloader):
        if i >= num_samples:
            LOGGER.warning(f"len(dataloader)={len(dataloader)}, but more samples found.")
            break
        validate_sample_input(sample, FRAMEWORK_TO_TENSOR_TYPE[framework])
        sample = {n: to_numpy(t, framework) for n, t in pytree_metadata.flatten_sample(sample).items()}
        for name, tensor in sample.items():
            for k, dim in enumerate(tensor.shape):
                axes_shapes[name][k].append(dim)

    if check_len:
        assert i + 1 >= len(dataloader), f"len(dataloader)={len(dataloader)}, but only {i + 1} samples found."

    return axes_shapes


def _get_metadata_from_axes_shapes(pytree_metadata, axes_shapes, batch_dim, dtypes):
    metadata = TensorMetadata(pytree_metadata=pytree_metadata)
    for name, axes in axes_shapes.items():
        tensor_shape = []
        for ax, shapes in axes.items():
            if ax == batch_dim or min(shapes) != max(shapes):
                tensor_shape.append(-1)
            else:
                tensor_shape.append(shapes[0])
        metadata.add(name, tuple(tensor_shape), dtypes[name])
    return metadata


def _extract_max_batch_size(axes_shapes: Dict[str, Dict[int, List[int]]], batch_dim: Optional[int]) -> int:
    if batch_dim is not None:
        return max(list(axes_shapes.values())[0][batch_dim])
    return 0


def _get_trt_profile_from_axes_shapes(axes_shapes, batch_dim, config_max_batch_size=None):
    trt_profile = TensorRTProfile()
    for name, axes in axes_shapes.items():
        min_opt_max = []
        for ax, shapes in axes.items():
            if ax == batch_dim:  # min bs = 1
                if config_max_batch_size and (config_max_batch_size < max(shapes)):
                    raise ModelNavigatorUserInputError(
                        f"Given configuration maximum batch size ({config_max_batch_size}) "
                        f"is smaller than the encountered batch size ({max(shapes)})."
                    )
                max_batch_size = config_max_batch_size or max(shapes)
                opt_shape = optimal_batch_size(max_batch_size)
                min_opt_max.append((1, opt_shape, max_batch_size))
            else:
                min_opt_max.append((min(shapes), int(np.median(shapes)), max(shapes)))
        if min_opt_max:
            trt_profile.add(name, *list(zip(*min_opt_max)))
    return trt_profile


def _assert_all_inputs_have_same_pytree_metadata(
    dataloader: Union[SizedDataLoader, Iterator],
    pytree_metadata: PyTreeMetadata,
) -> bool:
    for sample in dataloader:
        if not pytree_metadata.is_compatible_with(sample):
            raise ModelNavigatorUserInputError(
                f"All inputs must have the same structure.\n"
                f"Input structure: {pytree_metadata}\n"
                f"Sample: {sample}."
            )


class InferInputMetadata(Command, is_required=True):
    """Command to collect model inputs metadata.

    """

    def _run(
        self,
        model: Union[object, pathlib.Path],
        framework: Framework,
        dataloader: SizedDataLoader,
        optimization_profile: OptimizationProfile,
        _input_names: Optional[Tuple[str, ...]] = None,
        batch_dim: Optional[int] = None,
    ) -> CommandOutput:
        """Execute the InferInputMetadata command.
        功能：
        1. 校验 dataloader 提供的所有 input sample 是否合法，在以下几个方面
            1. 其数据结构是否是合法的 pytree 结构
            2. 其中 tensor 的数据类型 framework 是否支持
            3. 是否 dataloader 中所有的 input samples 具备同样的 pytree 结构
        2. 构建 input sample 的 metadata，具体包括两部分
            1. pytree metadata，描述了input sample 的数据结构
            2. tensor metadata，描述了所有张量的 spec(name, shape, dtype)
            值得注意的是，pytree metadata 作为 tensor metadata 的一部分被保存
        3. 柑橘 dataloader 中提供的所有 input samples
           计算每个 tensor 的动态尺度范围，提供给 trt_profile
        4. 校验 optimization_profile 中的 dataloader 是否与 参数中的 dataloader 兼容

        Args:
            framework: Framework of model to run inference
            model: A model object or path to file
            dataloader: Dataloader for providing samples
            optimization_profile: Optimization profile
            _input_names: Name of model inputs
            batch_dim: Location of batch dimension in data samples

        Returns:
            CommandOutput object
        """
        # 从 dataloader 中获取一个 input sample
        sample = next(iter(dataloader))
        # 校验该 input sample 对于当前框架是否合法，包括检查 tensor 类型
        validate_sample_input(sample, FRAMEWORK_TO_TENSOR_TYPE[framework])
        if framework == Framework.ONNX:
            if _input_names is not None:
                LOGGER.warning("ONNX input names are not supported yet. `input_names` will be ignored.")
            _input_names, _ = get_onnx_io_names(model)
        elif framework == Framework.TENSORRT:
            _input_names, _ = get_tensorrt_io_names(model)

        # 构建 dataloader 中 input sample 的 pytreemeta，
        # 其记录了 input 中不同数据之间的结构，以及 tensor 的类型
        pytree_metadata = PyTreeMetadata.from_sample(
            sample, tensor_type=FRAMEWORK_TO_TENSOR_TYPE[framework], names=_input_names, prefix="input"
        )
        # 校验是否 dataloader 中全部的 input 都具备相同的 pytreeMetadata
        _assert_all_inputs_have_same_pytree_metadata(dataloader, pytree_metadata)
        input_sample = {}
        input_dtypes = {}
        # 获得 input sample 中每个张量的规格，包括 name，shape，dtype
        for n, t in pytree_metadata.flatten_sample(sample).items():
            input_sample[n] = to_numpy(t, framework)

            # TODO: Remove this check once torch.bfloat16 is supported
            if not is_torch_available() or t.dtype != torch.bfloat16:
                input_dtypes[n] = input_sample[n].dtype
            else:
                input_dtypes[n] = torch.bfloat16
        input_names = list(input_sample.keys())

        input_ndims = [t.ndim for t in input_sample.values()]
        num_samples = len(dataloader)
        axes_shapes = _extract_axes_shapes(
            dataloader, pytree_metadata, input_names, input_ndims, num_samples, framework
        )
        # 获得 dataloader 中，所有 input sample 中 batch_dim 上最大的尺寸
        dataloader_max_batch_size = _extract_max_batch_size(axes_shapes, batch_dim)
        # 根据 dataloader 中所有 input sample 实际 shape
        # 总结出 tensorRT 需要的 dynamic shape
        dataloader_trt_profile = _get_trt_profile_from_axes_shapes(axes_shapes, batch_dim)
        input_metadata = _get_metadata_from_axes_shapes(pytree_metadata, axes_shapes, batch_dim, input_dtypes)

        # 如果 optimization profile 提供了自己的 dataloader
        # 则需要校验其是否符合 dataloader
        if optimization_profile.dataloader:
            pd_sample = next(iter(optimization_profile.dataloader))
            pd_input_sample = extract_sample(pd_sample, input_metadata, framework)
            pd_input_ndims = [t.ndim for t in pd_input_sample.values()]
            pd_input_dtypes = {n: t.dtype for n, t in pd_input_sample.items()}
            if pd_input_ndims != input_ndims:
                raise ModelNavigatorUserInputError(
                    "Provided performance dataloader does not match dataset dataloader size."
                )

            if pd_input_dtypes != input_dtypes:
                raise ModelNavigatorUserInputError(
                    "Provided performance dataloader does not match dataset dataloader data types."
                )

            self._validate_performance_dataloader_trt_profiles(
                optimization_profile=optimization_profile,
                pytree_metadata=input_metadata.pytree_metadata,
                input_names=input_names,
                input_ndims=pd_input_ndims,
                framework=framework,
                batch_dim=batch_dim,
                dataloader_trt_profile=dataloader_trt_profile,
            )

        return CommandOutput(
            status=CommandStatus.OK,
            output={
                "input_metadata": input_metadata,
                "dataloader_trt_profile": dataloader_trt_profile,
                "dataloader_max_batch_size": dataloader_max_batch_size,
            },
        )

    def _validate_performance_dataloader_trt_profiles(
        self,
        optimization_profile: OptimizationProfile,
        pytree_metadata,
        input_names,
        input_ndims,
        framework,
        batch_dim,
        dataloader_trt_profile,
    ):
        assert optimization_profile.dataloader != None
        axes_shapes = _extract_axes_shapes(
            dataloader=optimization_profile.dataloader,
            pytree_metadata=pytree_metadata,
            input_names=input_names,
            input_ndims=input_ndims,
            num_samples=1,
            framework=framework,
            check_len=False,
        )
        performance_dataloader_trt_profile = _get_trt_profile_from_axes_shapes(axes_shapes, batch_dim)
        for name, shape in performance_dataloader_trt_profile.items():
            dshape = dataloader_trt_profile[name]
            if shape.min < dshape.min or shape.max > dshape.max:
                raise ModelNavigatorUserInputError(
                    """Provided performance dataloader has invalid shape against the dataset dataloader."""
                    f""" Performance dataloader shape for input `{name}` is min: {shape.min}, max: {shape.max}."""
                    f""" Dataset dataloader shape for input `{name}` is min: {dshape.min}, max: {dshape.max}."""
                )


class InferOutputMetadata(Command, is_required=True):
    """Command to collect model outputs  metadata."""

    def _run(
        self,
        framework: Framework,
        model: Union[object, pathlib.Path],
        runner_cls: Type[NavigatorRunner],
        dataloader: SizedDataLoader,
        input_metadata: TensorMetadata,
        workspace: Workspace,
        verbose: bool,
        runner_config: Optional[RunnerConfig] = None,
        _output_names: Optional[Tuple[str, ...]] = None,
        batch_dim: Optional[int] = None,
    ) -> CommandOutput:
        """Execute the InferOutputMetadata command.

        功能：
            使用 model、input_metadata、runner_cls 等构建一个 runner
            runner 通过处理本地加载的 input sample 求的 outputs，其中：
            1. profiling_sample 得到的 output 用来求 output 的 pytreemetadata
            2. conversion_sample 因为覆盖 min 到 max 的动态尺寸范围，
               其 output 的尺寸用来构建 output 的 metadata
        Args:
            framework: Framework of model to run inference
            model: A model object or path to file
            runner_cls: Type of a runner to use with a model.
            dataloader: Dataloader for providing samples
            input_metadata: Model inputs metadata
            workspace: Working directory where command should be executed
            verbose: Enable verbose logging
            runner_config: Additional runner arguments.
            _output_names: Name of model outputs
            batch_dim: Location of batch dimension in data samples

        Returns:
            CommandOutput object
        """
        # 预备 output 的 Metadata
        # 主要是准备好所有张量的 name
        if framework == Framework.ONNX:
            if _output_names is not None:
                LOGGER.warning("ONNX output names are not supported yet. `output_names` will be ignored.")
            _, _output_names = get_onnx_io_names(model)
            temp_output_metadata = TensorMetadata({out_name: TensorSpec(out_name, ()) for out_name in _output_names})
        elif framework == Framework.TENSORRT:
            _, _output_names = get_tensorrt_io_names(model)
            temp_output_metadata = TensorMetadata({out_name: TensorSpec(out_name, ()) for out_name in _output_names})
        else:
            temp_output_metadata = None

        # 获得 runner 的配置数据，并初始化 runner
        runner_kwargs = runner_config.to_dict() if runner_config is not None else {}
        runner = runner_cls(
            model=model,
            input_metadata=input_metadata,
            output_metadata=temp_output_metadata,
            disable_fallback=False,
            **runner_kwargs,
        )

        # 从 workspace 加载 dataloader
        # FIXME 当时在 sample_as_npz 时，只保留了，张量数据的信息
        # 完整的 input 还需要 pytreeMetadata 才可以复原
        profiling_sample = load_samples("profiling_sample", workspace.path, batch_dim)[0]
        conversion_samples = load_samples("conversion_sample", workspace.path, batch_dim)

        with runner, ExecutionContext(workspace=workspace, verbose=verbose):
            # 使用 runner 对
            outputs = runner.infer(profiling_sample)
            # 计算 output 的 pytreeMetada
            pytree_metadata = PyTreeMetadata.from_sample(
                outputs, tensor_type=FRAMEWORK_TO_TENSOR_TYPE[framework], names=_output_names, prefix="output"
            )
            output_sample = {n: to_numpy(t, framework) for n, t in pytree_metadata.flatten_sample(outputs).items()}
            output_names = list(output_sample.keys())
            output_generator = (runner.infer(sample) for sample in conversion_samples)

            output_ndims = [t.ndim for t in output_sample.values()]
            output_dtypes = {n: t.dtype for n, t in output_sample.items()}
            num_samples = len(dataloader)
            axes_shapes = _extract_axes_shapes(
                output_generator, pytree_metadata, output_names, output_ndims, num_samples, framework, check_len=False
            )
        # 获得 output 的 metadata
        output_metadata = _get_metadata_from_axes_shapes(pytree_metadata, axes_shapes, batch_dim, output_dtypes)

        return CommandOutput(
            status=CommandStatus.OK,
            output={"output_metadata": output_metadata},
        )
