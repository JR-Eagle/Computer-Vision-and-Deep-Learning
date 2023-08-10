import easyCNN
import torch.nn as nn
import torchvision
from torchvision import models

def get_model(model_name, num_classes=4, use_pretrained=True):

    if model_name in ['easycnn']:
            model = easyCNN.EasyCNN(num_classes)

    elif model_name == 'alexnet':
       model = models.alexnet(pretrained=use_pretrained)
       num_ftrs = model.classifier[6].in_features
       model.classifier[6] = nn.Linear(4096, num_classes)

    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'resnet18_ft':
        model = models.resnet18(pretrained=use_pretrained)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'regnet_x_8gf':
        model = models.regnet_x_8gf(pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    return model
