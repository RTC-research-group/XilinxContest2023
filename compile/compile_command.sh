#!/bin/bash

mkdir compile_result
python3 compile/vai_c_xir_cust_pytorch.py -x quantize_result/Sequential_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCAHX8H/U280/arch.json -o compile_result -n trt_pose --options '{"plugins": "vart_op_imp_tanh"}'