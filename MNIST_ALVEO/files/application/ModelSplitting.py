import torch
from trt_pose import models
from Processing import preprocessingTransforms, postProcessBatch, \
    get_child_subgraph_dpu, runDPU, postProcessing, num_parts, num_links
from collections import OrderedDict

import torchvision
from trt_pose.models.common import *
from trt_pose.models.resnet import ResNetBackbone

cpuLayersWeights = [
    '1.cmap_conv.weight',
    '1.cmap_conv.bias',
    '1.paf_conv.weight',
    '1.paf_conv.bias'
]

def splitWeightsMatrix(modelFile):
    weightDict = torch.load(modelFile, map_location=torch.device('cpu'))

    resDpu = OrderedDict()
    resCpu = OrderedDict()

    for key, value in weightDict.items():
        if key in cpuLayersWeights:
            key_ = str(key)
            key_ = '0' + key_[1:] # Change from 2nd sequential subnetwork to 1st sequential subnetwork
            resCpu[key_] = value
        else:
            resDpu[key] = value

    return resDpu, resCpu


class CmapPafHeadAttentionDPU(torch.nn.Module):
    def __init__(self, input_channels, upsample_channels=256, num_upsample=0, num_flat=0):
        super(CmapPafHeadAttentionDPU, self).__init__()
        self.cmap_up = UpsampleCBR(input_channels, upsample_channels, num_upsample, num_flat)
        self.paf_up = UpsampleCBR(input_channels, upsample_channels, num_upsample, num_flat)
        self.cmap_att = torch.nn.Conv2d(upsample_channels, upsample_channels, kernel_size=3, stride=1, padding=1)
        self.paf_att = torch.nn.Conv2d(upsample_channels, upsample_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        xc = self.cmap_up(x)
        xp = self.paf_up(x)
        xc_att = self.cmap_att(xc)
        xp_att = self.paf_att(xp)
        ac = torch.sigmoid(xc_att)

        return xc, xp, ac, xp_att

class CmapPafHeadAttentionCPU(torch.nn.Module):
    def __init__(self, cmap_channels, paf_channels, upsample_channels=256):
        super(CmapPafHeadAttentionCPU, self).__init__()
        self.cmap_conv = torch.nn.Conv2d(upsample_channels, cmap_channels, kernel_size=1, stride=1, padding=0)
        self.paf_conv = torch.nn.Conv2d(upsample_channels, paf_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        xc, xp, ac, xp_att = x
        ap = torch.tanh(xp_att)
        mulCmap = xc * ac
        mulPaf = xp * ap

        return self.cmap_conv(mulCmap), self.paf_conv(mulPaf)


def buildDpuModel(upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    resnet = torchvision.models.resnet18(pretrained=pretrained)

    model = torch.nn.Sequential(
        ResNetBackbone(resnet),
        CmapPafHeadAttentionDPU(512, upsample_channels, num_upsample, num_flat)
    )
    return model

def buildCpuModel(cmap_channels, paf_channels, upsample_channels=256):
    model = torch.nn.Sequential(
        CmapPafHeadAttentionCPU(cmap_channels, paf_channels, upsample_channels)
    )
    return model