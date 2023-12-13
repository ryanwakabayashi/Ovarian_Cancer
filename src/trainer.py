import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device="cpu"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, num_epochs, writer):
        for epoch in tqdm(range(num_epochs), desc='Training Epochs', position=1):
            self.model.train()
            losses = []

            # Training loop
            with tqdm(self.train_loader, desc='Training Batch', position=0) as pbar:
                for batch_idx, batch in enumerate(pbar):
                    # TODO: Cleaner way to do the following
                    inputs, labels = batch
                    inputs, labels = inputs.type(torch.float32).to(self.device), labels.to(self.device)
                    inputs = inputs.type(torch.float32)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())
                    # TODO Add logging and saving logic here
                    writer.add_scalar('Loss/train', loss.item(), epoch * len(self.train_loader) + batch_idx)

            # TODO update test checkpoint save - might add date and time to filename - probably should have folder check/creation somewhere in the main file
            # if epoch % 5 == 0:
        self.save_checkpoint('src/checkpoints/checkpoint.pth', epoch, loss)

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                # TODO: Cleaner way to do the following
                inputs, labels = batch
                inputs, labels = inputs.type(torch.float32).to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                # Add evaluation logic here

    def save_checkpoint(self, path, epoch, loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        self.model.eval()
