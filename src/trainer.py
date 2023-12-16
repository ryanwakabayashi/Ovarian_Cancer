import numpy as np
import torch
from matplotlib.lines import Line2D
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    This method was found here: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/24'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


def save_plots():
    if not os.path.exists('gradient_flow'):
        os.makedirs('gradient_flow')
    plt.savefig('gradient_flow/grad_flow_vanishing.png')
    plt.ylim(bottom=-0.001, top=100)  # zoom out gradient regions
    plt.savefig('gradient_flow/grad_flow_exploding.png')


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, writer, device="cpu"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.writer = writer

    def train(self, num_epochs):
        for epoch in tqdm(range(num_epochs), desc='Training Epochs', position=1):
            self.model.train()
            losses = []

            # Training loop
            with tqdm(self.train_loader, desc='Training Batch', position=0) as pbar:
                for batch_idx, (inputs, labels) in enumerate(pbar):
                    inputs, labels = inputs.type(torch.float32).to(self.device), labels.to(self.device)
                    inputs = inputs.type(torch.float32)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    plot_grad_flow(self.model.named_parameters())
                    self.optimizer.step()
                    losses.append(loss.item())

                    # TODO Add logging and saving logic here
                    self.writer.add_scalar('Loss/Train', loss.item(), epoch * len(self.train_loader) + batch_idx)

            self.validate(epoch)

        save_plots()
            # TODO update test checkpoint save - might add date and time to filename - probably should have folder check/creation somewhere in the main file
            # if epoch % 5 == 0:
        self.save_checkpoint('src/checkpoints/checkpoint.pth', epoch, loss)

    def validate(self, epoch):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for (inputs, labels) in self.val_loader:
                inputs, labels = inputs.type(torch.float32).to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Add evaluation logic here
                self.writer.add_scalar('Loss/Validation', loss.item(), epoch)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        self.writer.add_scalar('Accuracy/Validation', accuracy, epoch)

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
