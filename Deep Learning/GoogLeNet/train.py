"""
@author: Rai
Training script for GoogLeNet on CIFAR10 dataset.
"""
import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import GoogLeNet
import matplotlib.pyplot as plt
import wandb

def load_data(batch_size):
    """Load CIFAR10 dataset with transformations."""
    
    transformations = {
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
    
    root_directory = '../../data'
    train_dataset = datasets.CIFAR10(root=root_directory, train=True, download=True, transform=transformations['train'])
    test_dataset = datasets.CIFAR10(root=root_directory, train=False, download=True, transform=transformations['test'])

    num_workers = min(os.cpu_count(), batch_size if batch_size > 1 else 0, 8)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f'Using {len(train_dataset)} images for training and {len(test_dataset)} images for testing.')
    return train_loader, test_loader

class TrainingEngine:
    """Training and testing engine for GoogLeNet."""
    
    def __init__(self, model, batch_size, learning_rate, device):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.device = device

    def train_one_epoch(self, loader):
        """Train the model for one epoch."""
        self.model.train()
        total_loss, total_accuracy = 0, 0
        
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs, aux2_outputs, aux1_outputs = self.model(images)
            
            # Compute losses for main and auxiliary outputs
            main_loss = self.criterion(outputs, labels)
            aux1_loss = self.criterion(aux1_outputs, labels)
            aux2_loss = self.criterion(aux2_outputs, labels)
            
            # Combine the losses
            combined_loss = main_loss + 0.3 * (aux1_loss + aux2_loss)
            
            combined_loss.backward()
            self.optimizer.step()

            _, predictions = outputs.max(1)
            total_loss += combined_loss.item()
            total_accuracy += (predictions == labels).sum().item()

        average_loss = total_loss / len(loader)
        average_accuracy = total_accuracy / len(loader.dataset)
        return average_loss, average_accuracy

    def evaluate(self, loader):
        """Evaluate the model's performance on the validation set."""
        self.model.eval()
        total_loss, total_accuracy = 0, 0
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                _, predictions = outputs.max(1)

                total_loss += loss.item()
                total_accuracy += (predictions == labels).sum().item()

        average_loss = total_loss / len(loader)
        average_accuracy = total_accuracy / len(loader.dataset)
        return average_loss, average_accuracy

def plot_training_curves(history, epochs):
    """Plot training and validation loss and accuracy curves."""
    epoch_range = np.arange(0, epochs + 1, epochs // 10)
    
    # Plotting loss curve
    plt.figure()
    plt.plot(history[:, 0], history[:, 1], 'b', label='Training Loss')
    plt.plot(history[:, 0], history[:, 3], 'g', label='Validation Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.xticks(epoch_range)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss_curve.png')

    # Plotting accuracy curve
    plt.figure()
    plt.plot(history[:, 0], history[:, 2], 'b', label='Training Accuracy')
    plt.plot(history[:, 0], history[:, 4], 'g', label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.xticks(epoch_range)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy_curve.png')

def main():
    """Main function to handle training and evaluation."""
    
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = '
