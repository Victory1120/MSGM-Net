import math

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight  # (64,64, 1, 1)
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)  # v = (64,1,1,1),m = (64,1,1,1)
        w = (w - m) / torch.sqrt(v + 1e-5)  # (64, 64, 1 , 1)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)

class resnet0(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, stride=1):
        super().__init__()
        cout = cout or cin

        self.gn1 = nn.GroupNorm(32, cin, eps=1e-6)
        self.conv1 = Conv2d(cin, cin, kernel_size=1, stride=stride)  # 64,64,没加strid,所以strid = 1??
        self.gn2 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv2 = Conv2d(cin, cout, kernel_size=3, stride=1, padding=1)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = Conv2d(cout, cout, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

        # Projection also with pre-activation according to paper.
        self.downsample = Conv2d(cin, cout, kernel_size=1, stride=1)  # 32, 64, strid = 2,
        self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x  # （8,32, 224,224）(8,256,55,55)
        # if hasattr(self, 'downsample'):
        #     residual = self.downsample(x)
        #     residual = self.gn_proj(residual)

        residual = self.downsample(x)
        residual = self.gn_proj(residual)  # (64,112,112)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))  # conv1*1降H与W
        y = self.relu(self.gn2(self.conv2(y)))  # conv3*3升通道
        y = self.gn3(self.conv3(y))  # 1*1 不变

        y = self.relu(residual + y)  # 残差连接
        return y

class resnet1(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, stride=1):
        super().__init__()
        cout = cout or cin

        self.gn1 = nn.GroupNorm(32, cin, eps=1e-6)
        self.conv1 = Conv2d(cin, cin, kernel_size=1, stride=stride)  # 64,64,没加strid,所以strid = 1??
        self.gn2 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv2 = Conv2d(cin, cout, kernel_size=3, stride=1, padding=1)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = Conv2d(cout, cout, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

        # Projection also with pre-activation according to paper.
        self.downsample = Conv2d(cin, cout, kernel_size=1, stride=2)  # 32, 64, strid = 2,
        self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x  # （8,32, 224,224）(8,256,55,55)
        # if hasattr(self, 'downsample'):
        #     residual = self.downsample(x)
        #     residual = self.gn_proj(residual)

        residual = self.downsample(x)
        residual = self.gn_proj(residual)  # (64,112,112)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))  # conv1*1降H与W
        y = self.relu(self.gn2(self.conv2(y)))  # conv3*3升通道
        y = self.gn3(self.conv3(y))  # 1*1 不变

        y = self.relu(residual + y)  # 残差连接
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))

class resnet2(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, stride=1):
        super().__init__()
        cout = cout or cin

        self.gn1 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv1 = Conv2d(cin, cout, kernel_size=1,stride=1)  # 64,64,没加strid,所以strid = 1??
        self.gn2 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv2 = Conv2d(cout, cout, kernel_size=3, stride=1, padding=1)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = Conv2d(cout, cout, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))  # y = (8,64,55,55)
        y = self.relu(self.gn2(self.conv2(y)))  # y = (8, 64, 55, 55)
        y = self.gn3(self.conv3(y))  # (8,256, 55, 55)

        y = self.relu(residual + y)  # (8, 256, 55, 55)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # print("b:",b)
        # print("c:",c)
        y = self.avg_pool(x).view(b, c)
        # print("y's shape:", y.shape)
        y = self.fc(y).view(b, c, 1, 1)
        # print("y2:",y.shape)
        return x * y.expand_as(x)


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):  # block_units = [3,3,6,9]
        super().__init__()
        width = int(64 * width_factor)  # width_factor = 1, width = 32
        self.width = width

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.body = nn.Sequential(OrderedDict([  # ResNet1,ResNet2
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', resnet0(cin=width, cout=width*2, stride=1))] +  # 64-->128
                [(f'unit{i:d}', resnet2(cin=width*2, cout=width*2)) for i in range(2, block_units[0] + 1)],  # 3 layers
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', resnet1(cin=width*2, cout=width*4, stride=2))] +  # 128-->256
                [(f'unit{i:d}', resnet2(cin=width*4, cout=width*4)) for i in range(2, block_units[1] + 1)],  # 4 layers
                ))),
            ('block3', nn.Sequential(OrderedDict(  # 256-->256
                [('unit1', resnet1(cin=width*4, cout=width*4, stride=2))] +
                [(f'unit{i:d}', resnet2(cin=width*4, cout=width*4)) for i in range(2, block_units[2] + 1)],  # 6 layers
                ))),
            ('block4', nn.Sequential(OrderedDict(
                [('unit1', resnet1(cin=width * 4, cout=width * 8, stride=2))] +
                [(f'unit{i:d}', resnet2(cin=width * 8, cout=width * 8)) for i in
                 range(2, block_units[2] + 1)],  # 6 layers
            ))),

        ]))
        self.SE = nn.Sequential(OrderedDict([
            ('SE1', Squeeze_Excite_Block(width * 2)),
            ('SE2', Squeeze_Excite_Block(width * 4)),
            ('SE3', Squeeze_Excite_Block(width * 4)),
        ]))

        self.lastse = Squeeze_Excite_Block(width * 8)


    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()  # b = batch数（8），c = 3(channels), in_size = 224
        # x = self.root(x)  # x = (8,3,224,224)--> x = (8, 32, 224, 224)
        # (b,64,h/2,w/2)  (b,64,112,112)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = x1 + x2 + x3  # (b,64,h/2,w/2)  (b,32,112,112)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x1)  #   (b,64,112,112)
        for i in range(len(self.body)-1):  # 0,1,2
            x = self.body[i](x)
            # right_size = int(in_size / 4 / (i+1))  # 56
            # if x.size()[2] != right_size:  # 56!=55,
            #     pad = right_size - x.size()[2]
            #     assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
            #     feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
            #     feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]  # 让特征值能够进行下去，这时候变成了(8,256,56,56)
            # else:
            x = self.SE[i](x)
            feat = x
            features.append(feat)  # (8,128, 112, 112) (8, 256, 56, 56) (8, 256, 28, 28)
        x = self.body[-1](x)  # 第四个block
        x = self.lastse(x)  # (8, 512, 14, 14)
        return x, features[::-1]
