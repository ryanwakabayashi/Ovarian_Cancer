import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Define your model layers here (Currently using model from pytorch documentation)
        # self.conv1 = nn.Conv2d(3, 8, 3, stride=1)
        self.conv1 = nn.Conv2d(3, 8, 5, stride=1)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=1)
        self.conv3 = nn.Conv2d(16, 32, 5, stride=1)
        self.batch_norm = nn.BatchNorm2d(32)
        # Can I use a 1x1 convolution?
        self.pool = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(2032128, 64)
        self.fc1 = nn.Linear(2032128, 64)
        self.fc2 = nn.Linear(64, 6)


    def forward(self, x):
        # Define the forward pass (Currently using model from pytorch documentation)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = self.batch_norm(x)
        # x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
