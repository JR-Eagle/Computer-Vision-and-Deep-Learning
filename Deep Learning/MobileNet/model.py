"""
@author:Rai
MobileNet-V1 Neural Network
"""

import torch
import torch.nn as nn
from torchvision import datasets
import torch.nn.functional as F


class MobileNet(nn.Module):
    def __init__(self, num_classes=1000, init_weight=True):
        super(MobileNet, self).__init__()
        layers = [
            ConvBNReLU(3, 32, stride=2),
            ConvDW(32, 64, stride=1),
            ConvDW(64, 128, stride=2),
            ConvDW(128, 128, stride=1),
            ConvDW(128, 256, stride=2),
            ConvDW(256, 256, stride=1),
            ConvDW(256, 512, stride=2),
            ConvDW(512, 512, stride=1),
            ConvDW(512, 512, stride=1),
            ConvDW(512, 512, stride=1),
            ConvDW(512, 512, stride=1),
            ConvDW(512, 512, stride=1),
            ConvDW(512, 1024, stride=2),
            ConvDW(1024, 1024, stride=1)
        ]
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AvgPool2d(kernel_size=7)
        self.classifier = nn.Linear(1024, num_classes)

        if init_weight:
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0, std=0.01)
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(-1, 1024)
        out = self.classifier(out)

        return out


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, stride):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, groups=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

# class ConvBNReLU(nn.Module):
#     def __init__(self, in_ch, out_ch, stride):
#         super(ConvBNReLU, self).__init__()
#         self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False, groups=1)
#         self.bn = nn.BatchNorm2d(out_ch),
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         out = self.conv(x)
#         out = self.bn(out)
#         out = self.relu(out)
#         return out


class ConvDW(nn.Sequential):
    def __init__(self, in_ch, out_ch, stride):
        super(ConvDW, self).__init__(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

if __name__ == '__main__':
    x = torch.rand(2, 3, 224, 224)
    net = MobileNet(num_classes=10)
    y = net(x)
    print('output shape:', y.shape)
