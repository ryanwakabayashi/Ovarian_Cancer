import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            for batch in self.train_loader:
                inputs, targets = batch
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                # Add logging and saving logic here

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                # Add evaluation logic here

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
