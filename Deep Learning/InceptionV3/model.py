"""
@author:Rai
Inception_v3 Neural Network
"""

import torch
import torch.nn as nn
from torchvision import datasets
import torch.nn.functional as F


class InceptionV3(nn.Module):
    def __init__(self, num_classes, in_ch=3, aux_logits=True, drop_out=0, init_weight=True):
        super(InceptionV3, self).__init__()
        self.conv_1 = BasicConv2d(in_ch, 32, kernel_size=3, stride=2)  # [3, 299, 299] -> [32, 149, 149]
        self.conv_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)  # [32, 147, 147]
        self.conv_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)  # [64, 147, 147]

        self.max_pool_3a = nn.MaxPool2d(kernel_size=3, stride=2)  # [64, 73, 73]
        self.conv_3b = nn.Conv2d(64, 80, kernel_size=1, stride=1)  # [80, 73, 73]
        self.conv_4a = nn.Conv2d(80, 192, kernel_size=3, stride=1)  # [192, 71, 71]

        self.max_pool_5a = nn.MaxPool2d(kernel_size=3, stride=2)  # [192, 35, 35]
        self.incep_5b = InceptionA(192, pool_features=32)  # [256, 35, 35]
        self.incep_5c = InceptionA(256, pool_features=64)  # [288, 35, 35]
        self.incep_5d = InceptionA(288, pool_features=64)  # [288, 35, 35]

        self.incep_6a = InceptionB1(288)  # [768, 17, 17]
        self.incep_6b = InceptionB2(768, ch_7x7=128)  # [768, 17, 17]
        self.incep_6c = InceptionB2(768, ch_7x7=160)  # [768, 17, 17]
        self.incep_6d = InceptionB2(768, ch_7x7=160)  # [768, 17, 17]
        self.incep_6e = InceptionB2(768, ch_7x7=192)  # [768, 17, 17]

        if aux_logits:
            self.aux_logits = InceptionAux(768, num_classes=num_classes)
        else:
            self.aux_logits = None

        self.incep_7a = InceptionC(768)  # [1280, 8, 8]
        self.incep_7b = InceptionD(1280)  # [2048, 8, 8]
        self.incep_7c = InceptionD(2048)  # [2048, 8, 8]

        self.max_pool_8a = nn.MaxPool2d(kernel_size=8, stride=1)  # [2048, 1, 1]
        self.dropout = nn.Dropout(p=drop_out)
        self.fc = nn.Linear(2048 * 1 * 1, num_classes)

        if init_weight:
            for layer in self.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2a(x)
        x = self.conv_2b(x)

        x = self.max_pool_3a(x)
        x = self.conv_3b(x)
        x = self.conv_4a(x)

        x = self.max_pool_5a(x)
        x = self.incep_5b(x)
        x = self.incep_5c(x)
        x = self.incep_5d(x)

        x = self.incep_6a(x)
        x = self.incep_6b(x)
        x = self.incep_6c(x)
        x = self.incep_6d(x)
        x = self.incep_6e(x)

        if self.training and self.aux_logits:
            aux = self.aux_logits(x)

        x = self.incep_7a(x)
        x = self.incep_7b(x)
        x = self.incep_7c(x)

        x = self.max_pool_8a(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training:
            if self.aux_logits:
                return x, aux
            else:
                return x, None
        return x

class InceptionA(nn.Module):
    def __init__(self, ch_in, pool_features, conv_block=None):
        if conv_block is None:
            conv_block = BasicConv2d
        super(InceptionA, self).__init__()
        self.branch1x1 = conv_block(ch_in, 64, kernel_size=1, stride=1)

        self.branch5x5_1 = conv_block(ch_in, 48, kernel_size=1, stride=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, stride=1, padding=2)

        self.branch3x3_1 = conv_block(ch_in, 64, kernel_size=1, stride=1)
        self.branch3x3_2 = conv_block(64, 96, kernel_size=3, stride=1, padding=1)
        self.branch3x3_3 = conv_block(96, 96, kernel_size=3, stride=1, padding=1)

        self.branch_pool_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_2 = conv_block(ch_in, pool_features, kernel_size=1, stride=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = self.branch_pool_1(x)
        # branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool_2(branch_pool)

        output = [branch1x1, branch5x5, branch3x3, branch_pool]
        output = torch.cat(output, dim=1)  # [256, 35, 35]

        return output


class InceptionB1(nn.Module):
    def __init__(self, ch_in, conv_block=None):
        super(InceptionB1, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.branch3x3 = conv_block(ch_in, 384, kernel_size=3, stride=2)  # [384, 17, 17]

        self.branch3x3dbl_1 = conv_block(ch_in, 64, kernel_size=1, stride=1)  # [64, 35, 35]
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, stride=1, padding=1)  # [96, 35, 35]
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)  # [96, 17, 17]

        self.branch_pool = nn.AvgPool2d(kernel_size=3, stride=2)  # [288, 17, 17]

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.branch_pool(x)

        output = [branch3x3, branch3x3dbl, branch_pool]
        output = torch.cat(output, dim=1)

        return output


class InceptionB2(nn.Module):
    def __init__(self, ch_in, ch_7x7, conv_block=None):
        super(InceptionB2, self).__init__()

        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(ch_in, 192, kernel_size=1, stride=1)

        self.branch7x7_1 = conv_block(ch_in, ch_7x7, kernel_size=1, stride=1)
        self.branch7x7_2 = conv_block(ch_7x7, ch_7x7, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.branch7x7_3 = conv_block(ch_7x7, 192, kernel_size=(7, 1), stride=1, padding=(3, 0))

        self.branch7x7dbl_1 = conv_block(ch_in, ch_7x7, kernel_size=1, stride=1)
        self.branch7x7dbl_2 = conv_block(ch_7x7, ch_7x7, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(ch_7x7, ch_7x7, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(ch_7x7, ch_7x7, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(ch_7x7, 192, kernel_size=(1, 7), stride=1, padding=(0, 3))

        self.branch_pool_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_2 = conv_block(ch_in, 192, kernel_size=1, stride=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = self.branch_pool_1(x)
        branch_pool = self.branch_pool_2(branch_pool)

        output = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        output = torch.cat(output, dim=1)

        return output


class InceptionC(nn.Module):
    def __init__(self, ch_in, conv_block=None):
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.branch3x3_1 = conv_block(ch_in, 192, kernel_size=1, stride=1)  # [192, 17, 17]
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)  # [192, 8, 8]

        self.branch7x7x3_1 = conv_block(ch_in, 192, kernel_size=1, stride=1)  # [192, 17, 17]
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), stride=1, padding=(0, 3))  # [192, 17, 17]
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), stride=1, padding=(3, 0))  # [192, 17, 17]
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)  # [192, 8, 8]

        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7 = self.branch7x7x3_1(x)
        branch7x7 = self.branch7x7x3_2(branch7x7)
        branch7x7 = self.branch7x7x3_3(branch7x7)
        branch7x7 = self.branch7x7x3_4(branch7x7)

        branch_pool = self.branch_pool(x)

        output = [branch3x3, branch7x7, branch_pool]
        output = torch.cat(output, dim=1)

        return output


class InceptionD(nn.Module):
    def __init__(self, ch_in, conv_block=None):
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(ch_in, 320, kernel_size=1, stride=1)  # [320, 8, 8]

        self.branch3x3_1 = conv_block(ch_in, 384, kernel_size=1, stride=1)  # [384, 8, 8]
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), stride=1, padding=(0, 1))  # [384, 8, 8]
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), stride=1, padding=(1, 0))  # [384, 8, 8]

        self.branch3x3dbl_1 = conv_block(ch_in, 448, kernel_size=1, stride=1)  # [448, 8, 8]
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, stride=1, padding=1)  # [384, 8, 8]
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), stride=1, padding=(0, 1))  # [384, 8, 8]
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), stride=1, padding=(1, 0))  # [384, 8, 8]

        self.branch_pool_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)  # [1280, 8, 8]
        self.branch_pool_2 = conv_block(ch_in, 192, kernel_size=1, stride=1)  # [192, 8, 8]

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3),
                     self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, dim=1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl),
                        self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, dim=1)

        branch_pool = self.branch_pool_1(x)
        branch_pool = self.branch_pool_2(branch_pool)

        output = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        output = torch.cat(output, dim=1)

        return output


# Auxiliary Classifier
class InceptionAux(nn.Module):
    def __init__(self, ch_in, num_classes=1000):
        super(InceptionAux, self).__init__()
        self.avg_pool1 = nn.AvgPool2d(kernel_size=5, stride=3)  # output:[768, 5, 5]
        self.conv1 = BasicConv2d(ch_in, 128, kernel_size=1, stride=1)  # [128, 5, 5]
        self.conv2 = BasicConv2d(128, 768, kernel_size=5, stride=1)  # [768,1,1]
        self.avg_pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.avg_pool1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.avg_pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


if __name__ == '__main__':
    x = torch.rand(2, 3, 299, 299)
    net = InceptionV3(aux_logits=True, num_classes=10)
    net.train()
    y, aux = net(x)
    print('output shape:', y.shape)
    print('aux shapee:', aux.shape)
