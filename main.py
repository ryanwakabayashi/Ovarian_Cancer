import torch

from src.models.MyModel import MyModel
from src.data.dataset import CustomImageDataset
from torch.utils.data import DataLoader
from src.trainer import Trainer
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


def print_hi(name):
    writer = SummaryWriter()
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} device")

    # Define the transformation for the image resizing
    # resize_image = transforms.Compose([
    #     # transforms.Resize((256, 256), antialias=True)
    #     transforms.Resize((2048, 2048), antialias=True)
    # ])

    # Get the full dataset and split it for train and validation
    full_dataset = CustomImageDataset('data/train.csv', 'data/preprocessed_images')
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))

    # training_data = CustomImageDataset('data/train.csv', 'data/train_images')
    num_epochs = 5

    model = MyModel().to(device)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=12)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)
    trainer.train(num_epochs, writer)

    writer.flush()

    # TODO move validation inside of trainer
    # trainer.validate()


if __name__ == '__main__':
    print_hi('PyCharm')
