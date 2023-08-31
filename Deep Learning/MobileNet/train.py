"""
@author: Rai
Training script for MobileNet on CIFAR10 dataset
"""

import os
import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import MobileNet
import matplotlib.pyplot as plt
import numpy as np
import wandb

def load_data(batch_size):
    """Load CIFAR10 dataset with appropriate transformations."""
    
    # Define data transformations
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
    
    # Define the root directory for the dataset
    data_directory = '../../data'
    
    # Load training and test datasets
    train_dataset = datasets.CIFAR10(root=data_directory, train=True, download=True, transform=data_transforms['train'])
    test_dataset = datasets.CIFAR10(root=data_directory, train=False, download=True, transform=data_transforms['test'])
    
    # Create data loaders
    num_workers = min(os.cpu_count(), batch_size if batch_size > 1 else 0, 8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f'Using {len(train_dataset)} images for training and {len(test_dataset)} images for testing.')
    return train_loader, test_loader

class Trainer:
    """Class to handle training and testing of the model."""
    
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
        """Evaluate the model on the test set."""
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

def plot_results(history, epochs):
    """Plot training and validation loss and accuracy curves."""
    
    epoch_range = np.arange(0, epochs + 1, epochs // 10)
    
    # Plotting loss curves
    plt.figure()
    plt.plot(history[:, 0], history[:, 1], 'b', label='Training Loss')
    plt.plot(history[:, 0], history[:, 3], 'g', label='Validation Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.xticks(epoch_range)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss_curve.png')

    # Plotting accuracy curves
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
    
    # Initialize parameters and settings
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = 'MobileNetV1'
    print(f"Start time: {current_time}")
    log_name = f"{current_time[2:]}_{model_name}"
    
    learning_rate = 0.0003
    epochs = 50
    batch_size = 128
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load data and instantiate the model
    train_loader, test_loader = load_data(batch_size)
    model = MobileNet(num_classes=10).to(device)
    
    trainer = Trainer(model=model, batch_size=batch_size, learning_rate=learning_rate, device=device)
    
    history = np.zeros((0, 5))
    best_acc = 0.0
    save_path = 'MobileNet.pth'
    
    # wandb configuration
    wandb_config = {
        "log_name": log_name,
        'model': model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        'learning_rate': learning_rate
    }
    
    wandb.init(project="TrainingMobileNet", name=log_name, id=wandb.util.generate_id(), config=wandb_config, save_code=True)
    
    # Training loop
    for epoch in range(epochs):
        train_loss, train_acc = trainer.train_one_epoch(train_loader)
        test_loss, test_acc = trainer.evaluate(test_loader)
        
        print(f'Epoch: [{epoch + 1}/{epochs}], Training Loss: {train_loss:.5f}, Training Accuracy: {train_acc:.4f}, Validation Loss: {test_loss:.5f}, Validation Accuracy: {test_acc:.4f}')
        
        wandb.log({"Train_loss": train_loss, "Test_loss": test_loss}, step=epoch)
        wandb.log({"Train_acc": train_acc, "Test_acc": test_acc}, step=epoch)
        
        # Save best model
        if test_acc > best_acc:
            torch.save(model.state_dict(), save_path)
            best_acc = test_acc
        
        history_entry = np.array([epoch, train_loss, train_acc, test_loss, test_acc])
        history = np.vstack([history, history_entry])

    # Plot training and validation curves
    plot_results(history, epochs)
    
    # Display total training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"End time: {datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')}")
    print(f'Training duration: {total_time_str}')
    
if __name__ == '__main__':
    main()
