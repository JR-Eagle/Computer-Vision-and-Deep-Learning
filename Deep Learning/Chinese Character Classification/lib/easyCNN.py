import torch
import torch.nn.init as init
import torch.nn.functional as F

class EasyCNN(torch.nn.Module):
    def __init__(self, n_classes):
        super(EasyCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)

        self.pool = torch.nn.MaxPool2d(2, 2)  # カーネルサイズ, ストライド

        self.fc1 = torch.nn.Linear(16 * 62 * 62, 120)  # 入力サイズ, 出力サイズ
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, n_classes)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.kaiming_normal_(self.fc3.weight)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 62 * 62) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)

        return x