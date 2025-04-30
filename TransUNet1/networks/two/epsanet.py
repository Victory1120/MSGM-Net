import torch
import torch.nn as nn
import math
from networks.z_model_v7.SE_weight_module import  SEWeightModule

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7], stride=1, conv_groups=[1, 4, 8]):
        super(PSAModule, self).__init__()
        self.mid = planes//2
        self.conv_1 = conv(inplans, self.mid, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, self.mid, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        # self.conv_3 = conv(inplans, self.mid*2, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
        #                     stride=stride, groups=conv_groups[2])
        # self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
        #                     stride=stride, groups=conv_groups[3])
        self.se1 = SEWeightModule(self.mid)
        self.se2 = SEWeightModule(self.mid)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)  # 32
        x2 = self.conv_2(x)  # 32
        # x3 = self.conv_3(x)  # 16

        feats = torch.cat((x1, x2), dim=1)
        feats = feats.view(batch_size, 2, self.mid, feats.shape[2], feats.shape[3])

        x1_se = self.se1(x1)
        x2_se = self.se2(x2)

        x_se = torch.cat((x1_se, x2_se), dim=1)  # 权重cat
        attention_vectors = x_se.view(batch_size, 2, self.mid, 1, 1)  # (b,4,channel/4,H,W)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(2):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out


class EPSABlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, conv_kernels=[3, 5, 7],
                 conv_groups=[1, 4, 8]):
        super(EPSABlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        # self.conv2 = conv3x3(planes, planes, stride=stride)
        self.conv2 = PSAModule(planes, planes, stride=stride, conv_kernels=conv_kernels, conv_groups=conv_groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # PSA模块,分组然后进行SE
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class EPSANet(nn.Module):
    def __init__(self,block, layers, num_classes=1000):
        super(EPSANet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(4, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layers(block, 64, layers[0], stride=1)  # (block,64,3,str=1)  64
        self.layer2 = self._make_layers(block, 128, layers[1], stride=2)  # (block,128,4,s=2)     128
        self.layer3 = self._make_layers(block, 128, layers[2], stride=2)  # (block,256.6,2)      128
        self.layer4 = self._make_layers(block, 256, layers[3], stride=2)                   #     256


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layers(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        features = []
        # (b,64,h/2,w/2)  (b,64,112,112)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = x1 + x2 + x3  # (b,64,h/2,w/2)  (b,64,112,112)
        features.append(x1)

        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)  # (b,64,56,56)

        x_1 = self.layer1(x)  # (b,256,56,56)  (b,256,112,112)
        features.append(x_1)
        x_2 = self.layer2(x_1)  # (b,512,28,28)  (b,256,56,56)
        features.append(x_2)
        x_3 = self.layer3(x_2)  # (b,1024,14,14) (b,512,28,28)
        features.append(x_3)
        x = self.layer4(x_3)  # (b,2048,7,7)  (b,1024,14,14)

        return x, features[::-1]


def epsanet50():
    model = EPSANet(EPSABlock, [3, 4, 6, 6], num_classes=9)
    return model

def epsanet101():
    model = EPSANet(EPSABlock, [3, 4, 23, 3], num_classes=1000)
    return model


if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((1,3, 224, 224), device=cuda0)
        model = epsanet50()
        model.cuda()
        y = model(x)
        # print(y.shape)
