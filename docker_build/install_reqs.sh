#!/bin/bash

conda init bash
conda activate vitis-ai-pytorch
echo `which python`
/opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python -m pip install numpy>=1.20

/opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python -m pip install torchinfo

/opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python -m pip install pycocotools


cd /
git clone https://github.com/RTC-research-group/XilinxContest2023
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd XilinxContest2023 && git pull && chmod +x quantize/quantize_trt_pose_command.sh
chown -R vitis-ai-user .
#ls -la /
#ls -la /trt_pose
cd /trt_pose
/opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python -m pip install .