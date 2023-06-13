#!/bin/bash

python3 quantize/quantize_trt_pose.py --model_dir 'trt_pose/model/resnet18_baseline_att_224x224_A_epoch_249.pth' --quant_mode test --batch_size 1 --deploy \
                                    --data_config 'trt_pose/resnet_baseline_att_224x224_A.json'
