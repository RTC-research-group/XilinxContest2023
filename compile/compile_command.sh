#!/bin/bash

mkdir compile_result
python3 compile/vai_c_xir_cust_pytorch.py -x quantize_result/Sequential_int.xmodel -a $1 -o compile_result -n trt_pose