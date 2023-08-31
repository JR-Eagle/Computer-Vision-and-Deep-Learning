"""
@author: Rai
Train a network on the MNIST dataset.
"""
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import NeuralNet
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt


# Data loading and preprocessing
def data_load(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Convert the data range from [0,1] to [-1,1]
        transforms.Normalize(0.5, 0.5),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    data_root = '../data'
    train_set = datasets.MNIST(root=data_root, train=True,
                               download=True, transform=transform)

    test_set = datasets.MNIST(root=data_root, train=False,
                              download=False, transform=transform)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size)
    num_classes = 10
    return train_set, test_set, train_loader, test_loader, num_classes


# Define the training and validation models
class Model(object):
    def __init__(self, net, cost, optimizer, lr, device):
        self.net = net
        self.cost = self.create_loss(cost)
        self.optim = self.create_optim(optimizer, lr)
        self.device = device

    def create_loss(self, cost):
        f_cost_dict = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MST': nn.MSELoss()
        }
        return f_cost_dict[cost]

    def create_optim(self, optimizer, lr, **kwargs):
        f_optim_dict = {
            'SGD': optim.SGD(self.net.parameters(), lr=lr, **kwargs),
            'ADAM': optim.Adam(self.net.parameters(), lr=lr, **kwargs),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=lr, **kwargs)
        }
        return f_optim_dict[optimizer]

    def train(self, train_loader, batch_size):
        train_acc, train_loss= 0, 0
        for inputs, labels in train_loader:
            # Load data into GPU or CPU
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            # Predicted values
            outputs = self.net(inputs)
            # Calculate loss
            loss = self.cost(outputs, labels)
            # Prevent gradient accumulation
            self.optim.zero_grad()
            # Backpropagation
            loss.backward()
            # Update parameters
            self.optim.step()
            # The highest probability is the predicted value .max returns (maximum value, index)
            pred = torch.max(outputs, 1)[1]
            train_loss += loss.item()
            train_acc += (pred == labels).sum().item()

        data_nums = len(train_loader.dataset)
        train_acc = train_acc / data_nums
        train_loss = train_loss * batch_size / data_nums
        return train_acc, train_loss

    def test(self, test_loader, batch_size):
        test_acc, test_loss = 0, 0
        # Turn off the gradient to save memory
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                pred = torch.max(outputs, 1)[1]
                test_loss += loss.item()
                test_acc += (pred == labels).sum().item()
        data_nums = len(test_loader.dataset)
        test_acc = test_acc / data_nums
        test_loss = test_loss * batch_size / data_nums
        return test_acc, test_loss


def show_figure(history, epochs):
    unit = epochs / 5
    # Draw the loss curve
    plt.plot(history[:, 0], history[:, 1], 'b', label='train_loss')
    plt.plot(history[:, 0], history[:, 3], 'g', label='test_loss')
    plt.title('Loss curve')
    plt.legend()
    plt.xticks(np.arange(0, epochs + 1, unit))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('Loss_Curve.png')
    plt.show()

    # Draw the accuracy curve
    plt.plot(history[:, 0], history[:, 2], 'b', label='train_acc')
    plt.plot(history[:, 0], history[:, 4], 'g', label='test_acc')
    plt.title('Accuracy curve')
    plt.legend()
    plt.xticks(np.arange(0, epochs + 1, unit))
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('Accuracy_Curve.png')
    plt.show()


def main():
    batch_size = 128
    lr = 0.1
    epochs = 50
    # If the device supports GPU, use GPU for training, otherwise use CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_set, test_set, train_loader, test_loader, num_classes = data_load(batch_size)

    image, _ = train_set[0]
    ch_input = image.shape[0]
    net = NeuralNet(ch_input=ch_input, ch_output=num_classes, ch_hidden=128).to(device)
    net.to(device)
    model = Model(net=net, cost='CROSS_ENTROPY', optimizer='SGD', lr=lr, device=device)
    # Save the results of training and validation losses and accuracies
    history = np.zeros((0, 5))

    for epoch in range(epochs):
        train_acc, train_loss = model.train(train_loader, batch_size)
        test_acc, test_loss = model.test(test_loader, batch_size)
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(
                f'Epoch:[{epoch + 1}/{epochs}], train_loss:{train_loss:.5f}, train_acc:{train_acc:.4f}, test_loss:{test_loss:.5f}, test_acc:{test_acc:.4f}')
        item = np.array([epoch + 1, train_loss, train_acc, test_loss, test_acc])
        history = np.vstack([history, item])

    save_pth = './SimpleNet.pth'
    torch.save(net.state_dict(), save_pth)
    # Draw
