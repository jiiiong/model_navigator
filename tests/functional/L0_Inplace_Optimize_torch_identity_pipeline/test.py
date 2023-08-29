#!/usr/bin/env python3
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
"""e2e tests for exporting PyTorch identity model"""
import argparse
import logging
import pathlib

import yaml

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])
METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{version}-py3",
}

EXPECTED_PACKAGES = 3
EXPECTED_STATUSES_TEMPLATE = [
    "{name}.{ind}.onnx.OnnxCUDA",
    "{name}.{ind}.onnx.OnnxTensorRT",
    "{name}.{ind}.torch.TorchCUDA",
    "{name}.{ind}.torchscript-script.TorchScriptCUDA",
    "{name}.{ind}.torchscript-trace.TorchScriptCUDA",
    "{name}.{ind}.trt-fp16.TensorRT",
    "{name}.{ind}.trt-fp32.TensorRT",
]
EXPECTED_STATUSES = [
    status.format(name=name, ind=0)
    for status in EXPECTED_STATUSES_TEMPLATE
    for name in ("model_a", "model_b", "model_c")
]


def main():
    import numpy as np
    import torch  # pytype: disable=import-error

    import model_navigator as nav
    from model_navigator.inplace.registry import module_registry
    from tests import utils
    from tests.functional.common.utils import collect_optimize_statuses, validate_status

    nav.inplace_config.mode = nav.Mode.RECORDING

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status",
        type=pathlib.Path,
        required=True,
        help="Status file where per path result is stored.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Timeout for test.",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=utils.DEFAULT_LOG_FORMAT)
    LOGGER.debug(f"CLI args: {args}")

    class Pipeline:
        def __init__(self):
            self.model_a = torch.nn.Identity()
            self.model_b = torch.nn.Identity()
            self.model_c = torch.nn.Identity()

        def __call__(self, x):
            ret = self.model_a(x)
            ret = self.model_b(ret)
            ret = self.model_c(ret)
            return ret

    pipe = Pipeline()
    t = torch.randn(2, 3)
    dataloader = [t] * 5

    def verify_func(ys_runner, ys_expected):
        for y_runner, y_expected in zip(ys_runner, ys_expected):
            if not all(np.allclose(a, b) for a, b in zip(y_runner.values(), y_expected.values())):
                return False
        return True

    optimize_config = nav.OptimizeConfig(
        verbose=True,
        optimization_profile=nav.OptimizationProfile(batch_sizes=[1, 8, 16], stability_percentage=100),
        verify_func=verify_func,
        runners=(
            "OnnxCUDA",
            "OnnxTensorRT",
            "TorchCUDA",
            "TorchScriptCUDA",
            "TensorRT",
        ),
    )
    pipe.model_a = nav.Module(pipe.model_a, optimize_config=optimize_config, name="model_a")
    pipe.model_b = nav.Module(pipe.model_b, optimize_config=optimize_config, name="model_b")
    pipe.model_c = nav.Module(pipe.model_c, optimize_config=optimize_config, name="model_c")

    for batch in dataloader:
        pipe(batch)

    nav.optimize()

    for batch in dataloader:
        pipe(batch)

    names, packages = [], []
    for name, module in module_registry.items():
        for i, package in enumerate(getattr(module._wrapper, "_packages", [])):
            names.append(f"{name}.{i}")
            packages.append(package)
    assert len(packages) == EXPECTED_PACKAGES, "Wrong number of packages."

    status_file = args.status
    status = collect_optimize_statuses([package.status for package in packages], names)

    validate_status(status, expected_statuses=EXPECTED_STATUSES)

    with status_file.open("w") as fp:
        yaml.safe_dump(status, fp)

    LOGGER.info(f"Status saved to {status_file}")


if __name__ == "__main__":
    main()
