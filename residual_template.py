# -*- coding: utf-8 -*-
# @Time : 2023/10/9 16:16
# @Author : yysgz
# @File : residual_template.py
# @Project : run_offline_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
"""
    当涉及到PyTorch中的高级残差连接时，通常是指在残差网络（ResNet）或其他深度卷积神经网络中引入一些变化或改进。
    以下是一个简单示例，展示了如何实现高级残差连接的一种方式，其中包括跳跃连接的条件性和卷积核大小的改进
"""

class AdvancedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_condition=False):
        super(AdvancedResidualBlock, self).__init__()

        # 主要卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 条件性跳跃连接
        self.use_condition = use_condition
        if self.use_condition:
            self.condition_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, condition=None):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        if self.use_condition and condition is not None:
            condition = self.condition_conv(condition)
            out = out + condition

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual  # 跳跃连接
        out = F.relu(out)

        return out


# 示例用法
input_channels = 64
output_channels = 128
block = AdvancedResidualBlock(input_channels, output_channels, stride=2, use_condition=True)
x = torch.randn(32, input_channels, 64, 64)
condition = torch.randn(32, input_channels, 64, 64)
output = block(x, condition)
print(output.shape)
