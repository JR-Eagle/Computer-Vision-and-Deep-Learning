"""
@author: Rai
Training ResNet18 neural network using the CIFAR10 dataset.
"""

import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from model import resnet_18
import wandb
import datetime, time

def load_data(batch_size):
    """
    Load CIFAR10 dataset with necessary transformations.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    root_dir = '../../data'
    train_dataset = datasets.CIFAR10(root=root_dir, train=True, download=True, transform=data_transforms['train'])
    test_dataset = datasets.CIFAR10(root=root_dir, train=False, download=True, transform=data_transforms['test'])

    num_workers = min(os.cpu_count(), batch_size if batch_size > 1 else 0, 8)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f'Using {len(train_dataset)} images for training, {len(test_dataset)} images for testing.')
    return train_loader, test_loader


class Trainer:
    """
    Trainer class for handling the training and testing processes.
    """
    def __init__(self, model, batch_size, learning_rate, device):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.device = device

    def train_epoch(self, loader):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss, total_acc = 0, 0
        
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            predictions = outputs.max(1)[1]

            total_loss += loss.item()
            total_acc += (predictions == labels).sum().item()

        num_samples = len(loader.dataset)
        avg_loss = total_loss * self.batch_size / num_samples
        avg_acc = total_acc / num_samples
        return avg_loss, avg_acc

    def evaluate(self, loader):
        """
        Evaluate the model's performance on a dataset.
        """
        self.model.eval()
        total_loss, total_acc = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                predictions = outputs.max(1)[1]

                total_loss += loss.item()
                total_acc += (predictions == labels).sum().item()

        num_samples = len(loader.dataset)
        avg_loss = total_loss * self.batch_size / num_samples
        avg_acc = total_acc / num_samples
        return avg_loss, avg_acc


class Visualizer:
    """
    Visualizer class for plotting training curves.
    """
    @staticmethod
    def plot_curves(history, epochs):
        """
        Plot the training and testing curves for accuracy and loss.
        """
        unit = epochs // 10
        # Plotting the loss curve
        plt.figure()
        plt.plot(history[:, 0], history[:, 1], 'b', label='train_loss')
        plt.plot(history[:, 0], history[:, 3], 'g', label='test_loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.xticks(np.arange(0, epochs + 1, unit))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('loss_curve.png')
        plt.show()

        # Plotting the accuracy curve
        plt.figure()
        plt.plot(history[:, 0], history[:, 2], 'b', label='train_acc')
        plt.plot(history[:, 0], history[:, 4], 'g', label='test_acc')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.xticks(np.arange(0, epochs + 1, unit))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.savefig('accuracy_curve.png')
        plt.show()


def main():
    # Initialization
    batch_size = 128
    learning_rate = 0.0001
    epochs = 30
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = data_load(batch_size)
    net = resnet_18()
    
    # Finetuning
    model_weight_path = './resnet18-official.pth'
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 10)
    net.to(device)
    
    trainer = Trainer(net, batch_size, learning_rate, device)
    history = np.zeros((epochs, 5))
    
    # Training loop
    for epoch in range(epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader)
        test_loss, test_acc = trainer.evaluate(test_loader)
        
        print(f'Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        history[epoch] = [epoch+1, train_loss, train_acc, test_loss, test_acc]

    # Visualization
    Visualizer.plot_curves(history, epochs)


if __name__ == '__main__':
    main()


