"""
@author:Rai
GoogLeNet Neural Network
"""


import torch
import torch.nn as nn
from torchvision import  datasets

class GoogLeNet(nn.Module):
    def __init__(self, num_classes, aux_logit=True, init_weight=False):
        super(GoogLeNet, self).__init__()
        self.aux_logit = aux_logit
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # self.localRespNorm = nn.LocalResponseNorm(4)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1, stride=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpoo2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if aux_logit:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

        if init_weight:
            for layer in self.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant(layer.bias, 0)
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
                    nn.init.constant(layer.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpoo2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        if self.training and self.aux_logit:
            aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training and self.aux_logit:
            aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training and self.aux_logit:
            return x, aux2, aux1
        return x

class Inception(nn.Module):
    def __init__(self, in_ch, out_ch11, out_ch33_re, out_ch33, out_ch55_re, out_ch55, out_ch_pool):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_ch, out_ch11, kernel_size=1, stride=1)
        self.branch2 = nn.Sequential(
                       BasicConv2d(in_ch, out_ch33_re, kernel_size=1, stride=1),
                       BasicConv2d(out_ch33_re, out_ch33, kernel_size=3, stride=1, padding=1)
                    )
        self.branch3 = nn.Sequential(
                        BasicConv2d(in_ch, out_ch55_re, kernel_size=1, stride=1),
                        BasicConv2d(out_ch55_re, out_ch55, kernel_size=5, stride=1, padding=2)
                    )
        self.branch4 = nn.Sequential(
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                        BasicConv2d(in_ch, out_ch_pool, kernel_size=1, stride=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        output = [branch1, branch2, branch3, branch4]
        output = torch.cat(output, dim=1) # concatenate from dim 1(channel)
        return output


# Auxiliary Classifier
class InceptionAux(nn.Module):
    def __init__(self, in_ch, num_classes=1000):
        super(InceptionAux, self).__init__()
        self.avepool = nn.AvgPool2d(kernel_size=5, stride=3)  # output:[4,4,512]
        self.conv = BasicConv2d(in_ch, 128, kernel_size=1, stride=1) # output:[4,4,128]

        self.dropuout = nn.Dropout(p=0.7)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.avepool(x)
        x = self.conv(x)

        x = torch.flatten(x, 1)
        x = self.dropuout(x)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.dropuout(x)
        x = self.fc2(x)
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


if __name__ == '__main__':
    x = torch.rand(2, 3, 224, 224)
    googlenet = GoogLeNet(aux_logit=False, num_classes=10)
    y = googlenet(x)
    print('output shape:', y.shape)