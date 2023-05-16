#!/bin/bash

conda init bash
conda activate vitis-ai-pytorch
echo `which python`
/opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python -m pip install numpy>=1.20

/opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python -m pip install torchinfo

/opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python -m pip install pycocotools


cd /
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
#ls -la /
#ls -la /trt_pose
cd /trt_pose
/opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python -m pip install .