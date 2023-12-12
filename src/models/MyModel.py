import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define your model layers here (Currently using model from pytorch documentation)
        self.conv1 = nn.Conv2d(3, 16, 7, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 7, stride=1)
        # self.conv3 = nn.Conv2d(32, 32, 1)

        # Can I use a 1x1 convolution?

        self.pool = nn.MaxPool2d(2, 2)
        # Adaptive Pooling Layer
        self.adap_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(1152, 32)
        self.fc2 = nn.Linear(32, 6)


    def forward(self, x):
        # Define the forward pass (Currently using model from pytorch documentation)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.adap_pool(F.relu(self.conv2(x)))

        # x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
