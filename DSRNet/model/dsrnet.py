from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.ops as ops
from model.DynamicConv import *


class Block(nn.Module):
    def __init__(self, ):
        super(Block, self).__init__()
        features = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=2, dilation=2, groups=1,
                      bias=False),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            DynamicConv(in_planes=features, out_planes=features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=2, dilation=2, groups=1,
                      bias=False),
            nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(
            DynamicConv(in_planes=features, out_planes=features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=2, dilation=2, groups=1,
                      bias=False),
            nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(
            DynamicConv(in_planes=features, out_planes=features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=2, dilation=2, groups=1,
                      bias=False),
            nn.ReLU(inplace=True))
        self.conv12 = nn.Sequential(
            DynamicConv(in_planes=features, out_planes=features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.conv13 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv14 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=2, dilation=2, groups=1,
                      bias=False),
            nn.ReLU(inplace=True))
        self.conv15 = nn.Sequential(
            DynamicConv(in_planes=features, out_planes=features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.conv16 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x4_1 = x4 + x1
        x5 = self.conv5(x4_1)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x7_1 = x7 + x4_1
        x8 = self.conv8(x7_1)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x10_1 = x10 + x7_1
        x11 = self.conv11(x10_1)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x13_1 = x13 + x10_1
        x14 = self.conv14(x13_1)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x16_1 = x16 + x13_1
        return x16_1


class Net(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(Net, self).__init__()
        self.scale = args.scale
        multi_scale = False
        features = 64

        '''
           in_channels, out_channels, kernel_size, stride, padding,dialation, groups,
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv13 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv14 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv15 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv16 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv17 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=3, kernel_size=3, padding=1, groups=1, bias=False))

        self.ReLU = nn.ReLU(inplace=True)
        self.upsample = ops.UpsampleBlock(64, scale=self.scale, multi_scale=multi_scale, group=1)

        self.Block = Block()
        self.sub_mean = common.MeanShift(255)
        self.add_mean = common.MeanShift(255, sign=1)

    def forward(self, x, scale):
        x = self.sub_mean(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x9_1 = x8 + x9
        x10 = self.conv10(x9_1)
        x10_1 = x10 + x7
        x11 = self.conv11(x10_1)
        x11_1 = x11 + x6
        x12 = self.conv12(x11_1)
        x12_1 = x12 + x5
        x13 = self.conv13(x12_1)
        x13_1 = x13 + x4
        x14 = self.conv14(x13_1)
        x14_1 = x14 + x3
        x15 = self.conv15(x14_1)
        x15_1 = x15 + x2
        x16 = self.conv16(x15_1)
        x16_1 = x16 + x1

        x16_2 = self.Block(x)

        x16_3 = x16_2 * x16_1
        temp = self.upsample(x16_3, scale)
        x17 = self.conv17(temp)
        out = self.add_mean(x17)

        return out

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, DynamicConv):
                m.update_temperature()