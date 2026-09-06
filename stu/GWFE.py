import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from WTConv import WTConv2d
import matplotlib.pyplot as plt
import torch.nn.functional as F


class GWFE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GWFE, self).__init__()
        self.groups = 4
        group_channels = in_channels // self.groups

        self.wtconv1 = WTConv2d(group_channels, group_channels, kernel_size=1)
        self.wtconv3 = WTConv2d(group_channels, group_channels, kernel_size=3)
        self.wtconv5 = WTConv2d(group_channels, group_channels, kernel_size=5)
        self.wtconv7 = WTConv2d(group_channels, group_channels, kernel_size=7)
        self.fusion_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x_groups = torch.chunk(x, self.groups, dim=1)
        group1 = F.leaky_relu(self.wtconv1(x_groups[0]), negative_slope=0.1)
        group2 = F.leaky_relu(self.wtconv3(x_groups[1]), negative_slope=0.1)
        group3 = F.leaky_relu(self.wtconv5(x_groups[2]), negative_slope=0.1)
        group4 = F.leaky_relu(self.wtconv7(x_groups[3]), negative_slope=0.1)
        x = torch.cat([group1, group2, group3, group4], dim=1)
        return self.fusion_conv(x)