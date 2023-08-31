"""
@author: Rai
Training VGG Neural Network with CIFAR10 dataset.
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
from model import vgg
import matplotlib.pyplot as plt
import wandb

def load_data(batch_size):
    """Load CIFAR10 dataset with transformations."""
    transformations = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
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


class ModelEngine:
    """Training and testing engine for neural network."""
    
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
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            _, predictions = outputs.max(1)
            total_loss += loss.item()
            total_accuracy += (predictions == labels).sum().item()

        average_loss = total_loss / len(loader)
        average_accuracy = total_accuracy / len(loader.dataset)
        return average_loss, average_accuracy

    def evaluate(self, loader):
        """Evaluate the model's performance on a dataset."""
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
    model_name = 'VGG'
    print(f"Start time: {current_time}")

    log_name = f"{current_time[2:]}_{model_name}"
    learning_rate = 0.0001
    epochs = 1
    batch_size = 128
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = load_data(batch_size)
    
    model = vgg(model_name='vgg11', num_classes=10)
    model.to(device)
    
    engine = ModelEngine(model=model, batch_size=batch_size, learning_rate=learning_rate, device=device)
    history = np.zeros((epochs, 5))
    best_accuracy = 0.0

    # Configuring wandb
    wandb.init(project="TrainingExercise", name=log_name, config={
        "model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate
    })

    for epoch in range(epochs):
        train_loss, train_acc = engine.train_one_epoch(train_loader)
        test_loss, test_acc = engine.evaluate(test_loader)
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss
