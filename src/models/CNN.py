import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, stride=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=1)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1)
        self.conv4 = nn.Conv2d(32, 64, 3, stride=1)
        self.conv5 = nn.Conv2d(64, 128, 3, stride=1)

        self.batch_norm8 = nn.BatchNorm2d(8)
        self.batch_norm16 = nn.BatchNorm2d(16)
        self.batch_norm32 = nn.BatchNorm2d(32)
        self.batch_norm64 = nn.BatchNorm2d(64)
        self.batch_norm128 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(25088, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.batch_norm8(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.batch_norm16(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.batch_norm32(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.batch_norm64(x)
        x = self.pool(F.relu(self.conv5(x)))
        x = self.batch_norm128(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x
