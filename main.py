import torch

from src.models.MyModel import MyModel
from src.data.dataset import CustomImageDataset, CustomImageDatasetThumbnail
from torch.utils.data import DataLoader
from src.trainer import Trainer
from torchvision import transforms


def print_hi(name):
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} device")

    # Define the transformation for the image resizing
    resize_image = transforms.Compose([
        # transforms.Resize((256, 256), antialias=True)
        transforms.Resize((2048, 2048), antialias=True)
    ])

    # Get the full dataset and split it for train and validation
    # TODO change from small directory
    # full_dataset = CustomImageDataset('data/fake_train.csv', 'data/fake_train_thumbnails', transform=resize_image)
    # full_dataset = CustomImageDataset('data/fake_train.csv', 'data/fake_train_images')
    full_dataset = CustomImageDataset('data/train.csv', 'data/train_images', transform=resize_image)
    # full_dataset = CustomImageDatasetThumbnail('data/train.csv', 'data/train_thumbnails', transform=resize_image)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))




    # training_data = CustomImageDataset('data/train.csv', 'data/train_images')
    num_epochs = 10

    model = MyModel().to(device)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=24)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)
    trainer.train(num_epochs)
    trainer.validate()


if __name__ == '__main__':
    print_hi('PyCharm')
