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
"""Convert ONNX model to TensorRT engine."""

import pathlib
from typing import Any, Dict, List, Optional, Tuple

import fire
from polygraphy.backend.trt import CreateConfig, Profile, engine_from_network, network_from_onnx_path, save_engine

from model_navigator.api.config import TensorRTPrecision, TensorRTPrecisionMode


def _get_precisions(precision, precision_mode):
    precision = TensorRTPrecision(precision)
    precision_mode = TensorRTPrecisionMode(precision_mode)
    if precision_mode == TensorRTPrecisionMode.HIERARCHY:
        tf32, fp16, int8 = {
            TensorRTPrecision.FP32: [True, False, False],
            TensorRTPrecision.FP16: [True, True, False],
            TensorRTPrecision.INT8: [True, True, True],
        }[precision]
    elif precision_mode == TensorRTPrecisionMode.SINGLE:
        tf32, fp16, int8 = {
            TensorRTPrecision.FP32: [True, False, False],
            TensorRTPrecision.FP16: [False, True, False],
            TensorRTPrecision.INT8: [False, False, True],
        }[precision]
    else:
        raise ValueError(
            f"Unsupported precision mode {precision_mode}. Only {TensorRTPrecisionMode.HIERARCHY} and "
            f"{TensorRTPrecisionMode.SINGLE} are allowed"
        )
    return tf32, fp16, int8


def convert(
    exported_model_path: str,
    converted_model_path: str,
    profiles: List[Dict[str, Dict[str, Tuple[int, ...]]]],
    max_workspace_size: int,
    precision: str,
    precision_mode: str,
    optimization_level: Optional[int] = None,
    compatibility_level: Optional[str] = None,
    workspace: Optional[str] = None,
    onnx_parser_flags: Optional[List[int]] = None,
    custom_args: Optional[Dict[str, Any]] = None,
) -> None:
    """Run conversion from TorchScript to Torch-TensorRT.

    Args:
        exported_model_path (str): TorchScript model path.
        converted_model_path (str): Output Torch-TensorRT model path.
        profiles (Dict[str, Dict[str, int]]): Dictionary with min, opt, max shapes of the inputs.
            The key is an input name and the value is a dictionary with keys ("min", "opt", "max")
            and respective values.
        max_workspace_size (int): Maximum workspace size in bytes.
        precision (str): TensorRT precision. Could be "fp16" or "fp32".
        precision_mode (str): TensorRT precision mode.
        target_device (str): _description_
        workspace (Optional[str], optional): Model Navigator workspace path.
            When None use current workdir. Defaults to None.
        optimization_level: Optimization level for TensorRT engine
        compatibility_level: Hardware compatibility level for generated engine
        onnx_parser_flags (Optional[List[trt.OnnxParserFlag]], optional): List of flags to set ONNX parser behavior.
        custom_args (Optional[Dict[str, str]], optional): Dictionary with passthrough parameters.
            For available arguments check PyTorch documentation: https://pytorch.org/TensorRT/py_api/torch_tensorrt.html
    """
    if not workspace:
        workspace = pathlib.Path.cwd()
    workspace = pathlib.Path(workspace)

    exported_model_path = pathlib.Path(exported_model_path)
    if not exported_model_path.is_absolute():
        exported_model_path = workspace / exported_model_path

    converted_model_path = pathlib.Path(converted_model_path)
    if not converted_model_path.is_absolute():
        converted_model_path = workspace / converted_model_path

    custom_args = custom_args or {}

    trt_profiles = []
    for profile in profiles:
        trt_profile = Profile()
        for name, dims in profile.items():
            trt_profile.add(name, **dims)
        trt_profiles.append(trt_profile)
    if not trt_profiles:
        trt_profiles = [Profile()]
    tf32, fp16, int8 = _get_precisions(precision, precision_mode)

    network = network_from_onnx_path(exported_model_path.as_posix(), flags=onnx_parser_flags)

    config_kwargs = {}
    if optimization_level:
        config_kwargs["builder_optimization_level"] = optimization_level
    if compatibility_level:
        config_kwargs["hardware_compatibility_level"] = compatibility_level
    engine = engine_from_network(
        network,
        config=CreateConfig(
            tf32=tf32,
            fp16=fp16,
            int8=int8,
            profiles=trt_profiles,
            **config_kwargs,
            **custom_args,
        ),
    )
    save_engine(engine, path=converted_model_path.as_posix())


if __name__ == "__main__":
    fire.Fire(convert)
