#!/bin/bash
#$1 -- data_config
#$2 -- model_dir


python3 vitis/quantize_trt_pose.py --model_dir $2 --quant_mode test --batch_size 1 --deploy \
                                    --data_config $1
