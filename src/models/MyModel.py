import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define your model layers here (Currently using model from pytorch documentation)
        self.conv1 = nn.Conv2d(3, 16, 9)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 9)
        self.conv3 = nn.Conv2d(32, 32, 1)
        # self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        self.fc1 = nn.Linear(744, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 6)


    def forward(self, x):
        # Define the forward pass (Currently using model from pytorch documentation)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = self.conv3(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
