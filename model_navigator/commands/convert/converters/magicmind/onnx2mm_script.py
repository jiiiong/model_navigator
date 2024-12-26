import argparse
from typing import List, Optional

import fire
import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser
from magicmind.python.runtime import DataType
from calibrator import CalibData
import time
import sys
import numpy as np

# import common components
from framework_parser import OnnxParser
from common_calibrate import common_calibrate
from build_param import get_argparser
from model_process import (
    extract_params,
    config_network,
    get_builder_config,
    build_and_serialize,
)
from logger import Logger

log = Logger()

def get_network(args):
    parser = OnnxParser(args)
    network = parser.parse()
    return network

def get_args():
    # get common argparser,here is pytorch_parser
    arg_parser = get_argparser()

    # add custom args belonging to the current net.
    arg_parser.add_argument(
        "--image_dir",
        dest="image_dir",
        type=str,
        default="../../datasets/coco/val2017",
        help="coco2017 datasets",
    )
    return arg_parser.parse_args()

def calibrate(args, network : mm.Network, config : mm.BuilderConfig):
    # 创建量化工具并设置量化统计算法
    input_dims = mm.Dims(args.input_dims[0])
    custom_max_samples = 16
    max_samples = max(custom_max_samples, args.input_dims[0][0])
    calib_data = CalibData(
        shape=input_dims, max_samples=max_samples, img_dir=args.image_dir
    )
    common_calibrate(args, network, config, calib_data)


def convert(
    # Network Configuration
    input_dims: Optional[List[List[int]]] = None,
    batch_size: Optional[List[int]] = None,
    rgb2bgr: bool = False,

    # Model Builder Configuration
    mlu_arch: List[str] = ["mtp_372"], # "mtp_220", "mtp_270", "mtp_290", "mtp_372", "tp_322", "mtp_592"
    cluster_num: Optional[List[List[int]]] = None,
    precision: str = "force_float32",
    means: Optional[List[List[float]]] = None,
    vars: Optional[List[List[float]]] = None,
    toolchain_path: Optional[str] = None,
    input_layout: Optional[List[str]] = None, # choices=["NCHW", "NHWC", "NCT", "NTC", "NCDHW", "NDHWC"]
    output_layout: Optional[List[str]] = None,
    dim_range_min: Optional[List[List[int]]] = None,
    dim_range_max: Optional[List[List[int]]] = None,
    build_config: Optional[str] = None,
    dynamic_shape: bool = True, #############
    computation_preference: str = "auto", # "auto", "fast", "high_precision"
    type64to32_conversion: bool = False,##############
    conv_scale_fold: bool = False,###############

    # Model Calibration
    device_id: int = 0,
    random_calib_range: Optional[List[float]] = None,
    file_list: Optional[List[str]] = None,
    calibration_data_path: Optional[str] = None,
    calibration: bool = False,
    rpc_server: Optional[str] = None,
    calibration_algo: str = "linear", # "linear", "eqnm"
    weight_quant_granularity: str = "per_tensor", # "per_tensor", "per_axis"
    activation_quant_algo: str = "symmetric", # "symmetric", "asymmetric"

    # Build and Serialize
    input_dtypes: Optional[List[str]] = None,
    output_dtypes: Optional[List[str]] = None,
    magicmind_model: str = "./model",

    # Plugin Paths
    plugin_path: Optional[List[str]] = None,

    # Framework Parsing
    tf_pb: Optional[str] = None,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    pytorch_pt: Optional[str] = None,
    pt_input_dtypes: List[str] = ["FLOAT"],
    onnx: Optional[str] = None,
    prototxt: Optional[str] = None,
    caffemodel: Optional[str] = None
):
    begin_time = time.time()
    # get net args
    # args = get_args()
    args = {
        "input_dims": input_dims,
        "batch_size": batch_size,
        "rgb2bgr": rgb2bgr,
        "mlu_arch": mlu_arch,
        "cluster_num": cluster_num,
        "precision": precision,
        "means": means,
        "vars": vars,
        "toolchain_path": toolchain_path,
        "input_layout": input_layout,
        "output_layout": output_layout,
        "dim_range_min": dim_range_min,
        "dim_range_max": dim_range_max,
        "build_config": build_config,
        "dynamic_shape": dynamic_shape,
        "computation_preference": computation_preference,
        "type64to32_conversion": type64to32_conversion,
        "conv_scale_fold": conv_scale_fold,
        "device_id": device_id,
        "random_calib_range": random_calib_range,
        "file_list": file_list,
        "calibration_data_path": calibration_data_path,
        "calibration": calibration,
        "rpc_server": rpc_server,
        "calibration_algo": calibration_algo,
        "weight_quant_granularity": weight_quant_granularity,
        "activation_quant_algo": activation_quant_algo,
        "input_dtypes": input_dtypes,
        "output_dtypes": output_dtypes,
        "magicmind_model": magicmind_model,
        "plugin_path": plugin_path,
        "tf_pb": tf_pb,
        "input_names": input_names,
        "output_names": output_names,
        "pytorch_pt": pytorch_pt,
        "pt_input_dtypes": pt_input_dtypes,
        "onnx": onnx,
        "prototxt": prototxt,
        "caffemodel": caffemodel
    }

    network = get_network(args)
    # configure network, such as input_dim, batch_size ...
    config_args = extract_params("MODEL_CONFIG", args)
    config_network(network, config_args)
    # create build configuration
    builder_args = extract_params("MODEL_BUILDER", args)
    build_config = get_builder_config(builder_args)

    if args.precision.find("qint") != -1:
        log.info("Do calibrate...")
        calibrate(args, network, build_config)
    log.info("Build model...")
    # 生成模型并导出
    model_name = "network"
    build_and_serialize_args = extract_params("MODEL_BUILD_AND_SERIALIZE", args)
    build_and_serialize(network, build_config, build_and_serialize_args)

    end_time = time.time()
    log.info("gen_model time cost:{:.3f}s".format(end_time - begin_time))

if __name__ == "__main__":
    fire.Fire(convert)
