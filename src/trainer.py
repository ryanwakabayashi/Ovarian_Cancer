import torch
from tqdm.auto import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device="cpu"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, num_epochs):
        for epoch in tqdm(range(num_epochs), desc='Training Epochs', position=1):
            self.model.train()

            # Training loop
            with tqdm(self.train_loader, desc='Training Batch', position=0) as pbar:
                for batch in pbar:
                    # TODO: Cleaner way to do the following
                    inputs, targets = batch
                    inputs, targets = inputs.type(torch.float32).to(self.device), targets.to(self.device)
                    inputs = inputs.type(torch.float32)
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
                # TODO: Cleaner way to do the following
                inputs, targets = batch
                inputs, targets = inputs.type(torch.float32).to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                # Add evaluation logic here

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
