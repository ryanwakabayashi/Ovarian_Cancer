import torch
import argparse

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import CustomImageDataset
from torch.utils.tensorboard import SummaryWriter
from numpy import mean

from src.trainer import Trainer
from src.utils import get_device, model_loader


def main(model_name):
    writer = SummaryWriter()
    device = get_device()

    full_dataset = CustomImageDataset('data/train.csv', 'data/preprocessed_images')
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size],
                                                               generator=torch.Generator().manual_seed(0))

    num_epochs = 50

    path_to_models = 'src.models.'
    model = model_loader(path_to_models, model_name, device)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=12, pin_memory=True)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, writer, device)
    # trainer.train(num_epochs)

    for epoch in range(num_epochs):
        losses = []
        with tqdm(train_loader, desc='Training Batch', position=0) as pbar:
            for batch_idx, (images, labels) in enumerate(pbar):
                # Assuming the data loader returns only images (unsupervised learning)
                # If it returns images and labels, use: images, _ = data
                # images = inputs
                images = images.type(torch.float32).to(device)
                # Forward pass
                encodings, outputs = model(images)

                # Calculate the loss
                loss = criterion(outputs, images)
                losses.append(loss.item())

                # Zero the gradient buffers
                optimizer.zero_grad()

                # Backward pass
                loss.backward()

                # Update the weights
                optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {mean(losses):.4f}')

    torch.save({
        # 'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, 'src/checkpoints/autoencoder-checkpoint.pth')
    writer.flush()


def parse_args():
    parser = argparse.ArgumentParser(description='Runs training using a give model')
    parser.add_argument('--model', type=str, help='input model name', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.model)
