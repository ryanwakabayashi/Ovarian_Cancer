import torch.nn as nn
import torch.nn.functional as F


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        # Encoder layers
        self.enc1 = nn.Conv2d(3, 16, 3, padding=1)
        self.enc2 = nn.Conv2d(16, 3, 3, padding=1)
        # self.enc3 = nn.Conv2d(32, 3, 3, padding=1)

        # Decoder layers
        self.dec1 = nn.ConvTranspose2d(3, 16, kernel_size=2, stride=2)
        # self.dec2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec3 = nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):

        # Encoding
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        # x = self.pool(x)
        # x = F.relu(self.enc3(x)) # problem lies here. How can i return the encoding with channels = 3
        encoding = self.pool(x)

        # Decoding
        x = F.relu(self.dec1(encoding))
        # x = F.relu(self.dec2(x))
        decoding = F.sigmoid(self.dec3(x))

        return encoding, decoding
