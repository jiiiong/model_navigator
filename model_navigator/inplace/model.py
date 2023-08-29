# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
"""Inplace models."""

import abc
import collections
import dataclasses
import inspect
import pathlib
import tempfile
from typing import Any, Callable, Dict, List

from model_navigator.api.config import TensorType
from model_navigator.api.package import load_from_workspace
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import PyTreeMetadata
from model_navigator.utils.module import lazy_import

from .config import OptimizeConfig, inplace_config
from .registry import module_registry
from .utils import TorchDataloader

torch = lazy_import("torch")

PYTREE_METADATA_PREFIX = "input"


class BaseModule(abc.ABC):
    """Base class for inplace Optimize modules."""

    def __init__(
        self,
        module,
        optimize_config: OptimizeConfig,
        name: str,
        input_mapping: Callable,
        output_mapping: Callable,
    ) -> None:
        """Initialize BaseModule.

        Args:
            module: module to be optimized.
            optimize_config: configuration for module optimization.
            name: name of the module.
            input_mapping: function mapping module input to runner input.
            output_mapping: function mapping runner output to module output.
        """
        self._module = module
        self._name = name
        self._signature = self._get_signature()
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._optimize_config = self._update_optimize_config(optimize_config)

    @property
    def name(self) -> str:
        """Module name."""
        return self._name

    @abc.abstractclassmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Module forward method."""
        pass

    @property
    def _workspace(self):
        return pathlib.Path(inplace_config.cache_dir) / f"{self._name}"

    def _update_optimize_config(self, optimize_config: OptimizeConfig) -> OptimizeConfig:
        config = dataclasses.replace(optimize_config)
        if config.workspace is None:
            config.workspace = self._workspace
            LOGGER.info(f"Setting workspace to {config.workspace}")

        return config

    def _get_signature(self) -> List[str]:
        """Get signature of the module forward method."""
        return inspect.getfullargspec(self._module.forward).args[1:]

    @property
    def is_ready_for_optimization(self) -> bool:
        """Check if the module is ready for optimization."""
        return False

    @property
    def is_optimized(self) -> bool:
        """Check if the module is optimized."""
        return False


class RecordModule(BaseModule):
    """Module that records samples for optimization."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize RecordModule."""
        super().__init__(*args, **kwargs)
        self._samples = []
        self._optimized = False
        self._temp_dir = tempfile.TemporaryDirectory(prefix=f"{self._name}_")
        self._samples_dir = pathlib.Path(self._temp_dir.name)
        self._indicies_groups: Dict[PyTreeMetadata, List[int]] = collections.defaultdict(list)

    def record_sample(self, *args: Any, **kwargs: Any) -> None:
        """Record a sample from the module."""
        sample = (*args, kwargs)
        sample = self._input_mapping(sample)
        ind = len(self._samples)
        sample_path = self._samples_dir / f"{ind}.pt"
        torch.save(sample, sample_path)  # change later to (sample,) to keep consistent with torch.onnx.export
        self._samples.append(sample_path)

        pytree_metadata = PyTreeMetadata.from_sample(sample, TensorType.TORCH, prefix=PYTREE_METADATA_PREFIX)
        self._indicies_groups[pytree_metadata].append(ind)

    def _filter_samples_by_pytree_metadata(self, pytree_metadata: PyTreeMetadata) -> List[pathlib.Path]:
        """Filter samples by PyTreeMetadata."""
        return [self._samples[ind] for ind in self._indicies_groups[pytree_metadata]]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Record a sample and run the module."""
        if len(self._samples) < inplace_config.max_num_samples:
            self.record_sample(*args, **kwargs)
        output = self._module(*args, **kwargs)
        return output

    def optimize(self):
        """Optimize the module using the recorded samples."""
        from model_navigator.api.torch import optimize

        config_dict = {k: v for k, v in self._optimize_config.to_dict().items() if k != "workspace"}

        for i, pytree_metadata in enumerate(self._indicies_groups):
            config_dict["workspace"] = self._optimize_config.workspace / str(i)
            samples = self._filter_samples_by_pytree_metadata(pytree_metadata)
            optimize(model=self._module, dataloader=TorchDataloader(samples), **config_dict)

        self._optimized = True

    @property
    def is_ready_for_optimization(self) -> bool:
        """Check if the module is ready for optimization."""
        return len(self._samples) >= inplace_config.min_num_samples

    @property
    def is_optimized(self) -> bool:
        """Check if the module is optimized."""
        return self._optimized


class RecordAndOptimizeModule(RecordModule):
    """Module that records samples for optimization and runs optimization."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Record a sample and run the module.

        If enough samples are collected for all registered modules, optimize them.
        """
        output = super().__call__(*args, **kwargs)
        if not self._optimized and module_registry.check_all_ready():
            module_registry.optimize()
        return output


class OptimizedModule(BaseModule):
    """Module that runs the optimized module."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize OptimizedModule.

        Load the optimized module from the Model Navigator workspace
        and get the runner according to the configured strategy.
        """
        super().__init__(*args, **kwargs)

        self._module.to("cpu")

        self._packages = []
        self._runners = {}
        for package_workspace in self._workspace.iterdir():
            package = load_from_workspace(package_workspace)
            package.load_source_model(self._module)

            runner = package.get_runner(return_type=TensorType.TORCH, strategy=inplace_config.strategy)
            runner.activate()
            pytree_metadata = package.status.input_metadata.pytree_metadata

            self._packages.append(package)
            self._runners[pytree_metadata] = runner

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the call through the optimized module."""
        sample = (*args, kwargs)
        sample = self._input_mapping(sample)
        pytree_metadata = PyTreeMetadata.from_sample(sample, TensorType.TORCH, prefix=PYTREE_METADATA_PREFIX)
        if pytree_metadata not in self._runners:
            raise ValueError(f"No runner found for {pytree_metadata}")
        runner = self._runners[pytree_metadata]
        if hasattr(
            runner, "inplace_infer"
        ):  # TODO this is a hack to avoid redundant flatten/unflatten for Torch runner
            output = runner.inplace_infer(*args, **kwargs)
        else:
            input_dict = runner.input_metadata.flatten_sample(sample)
            runner_output = runner.infer(input_dict)
            output = runner.output_metadata.unflatten_sample(runner_output)
        output = self._output_mapping(output)  # TODO better solution?
        return output

    @property
    def is_optimized(self) -> bool:
        """Check if the module is optimized."""
        return True


class PassthroughModule(BaseModule):
    """Module that passes through the original module."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Pass through the original module."""
        return self._module(*args, **kwargs)
