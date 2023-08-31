"""
@author: Rai
Train AlexNet on the CIFAR10 dataset.
"""
import os

import torchvision.datasets as datasets
import torchvision.transforms as transfroms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from model import AlexNet
from tqdm import tqdm
import sys
import wandb
import datetime, time


def data_load(batch_size):
    data_transform = {
        'train': transfroms.Compose([
            transfroms.RandomHorizontalFlip(),
            transfroms.RandomResizedCrop(224),
            # Convert data to tensor [0.0,10.]
            transfroms.ToTensor(),
            transfroms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transfroms.Compose([
            transfroms.Resize((224, 224)),
            transfroms.ToTensor(),
            transfroms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }
    root = '../../data'
    train_set = datasets.CIFAR10(root=root, train=True,
                                 download=True, transform=data_transform['train'])
    test_set = datasets.CIFAR10(root=root, train=False,
                                download=True, transform=data_transform['test'])

    num_workers = min(os.cpu_count(), batch_size if batch_size > 1 else 0, 8)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_num, test_num = len(train_set), len(test_set)
    print('Using {} images for training, {} images for testing.'.format(train_num, test_num))
    return train_loader, test_loader


class Engine(object):
    def __init__(self, net, batch_size, lr, device):
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.optim = optim.Adam(net.parameters(), lr=lr)
        self.batch_size = batch_size
        self.device = device

    def train(self, train_loader):
        self.net.train()
        train_loss, train_acc = 0, 0
        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optim.zero_grad()
            outputs = self.net(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optim.step()
            pred = outputs.max(1)[1]

            train_loss += loss.item()
            train_acc += (pred == labels).sum().item()

        data_nums = len(train_loader.dataset)
        train_loss = train_loss * self.batch_size / data_nums
        train_acc = train_acc / data_nums
        return train_loss, train_acc

    def test(self, test_loader):
        self.net.eval()
        test_loss, test_acc = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(images)
                loss = self.criterion(outputs, labels)
                pred = outputs.max(1)[1]

                test_loss += loss.item()
                test_acc += (pred == labels).sum().item()
        data_nums = len(test_loader.dataset)
        test_loss = test_loss * self.batch_size / data_nums
        test_acc = test_acc / data_nums
        return test_loss, test_acc


def show_figure(history, epochs):
    unit = epochs / 10
    # Plot the loss curve
    fig1 = plt.figure()
    plt.plot(history[:, 0], history[:, 1], 'b', label='train_loss')
    plt.plot(history[:, 0], history[:, 3], 'g', label='test_loss')
    plt.title('Loss curve')
    plt.legend()
    plt.xticks(np.arange(0, epochs + 1, unit))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    fig1.savefig('Loss_Curve.png')

    # Plot the accuracy curve
    fig2 = plt.figure()
    plt.plot(history[:, 0], history[:, 2], 'b', label='train_acc')
    plt.plot(history[:, 0], history[:, 4], 'g', label='test_acc')
    plt.title('Accuracy curve')
    plt.legend()
    plt.xticks(np.arange(0, epochs + 1, unit))
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    fig2.savefig('Accuracy_Curve.png')

def main():
    dt_end = datetime.datetime.now()
    time_run_start = dt_end.strftime('%Y%m%d_%H%M%S')
    print("Start time:"+time_run_start+'\n')
    model_name = 'AlexNet'
    log_name = str(time_run_start[2:]) + '_' + model_name
    
    lr = 0.001
    epochs = 200
   
