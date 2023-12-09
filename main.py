import torch

from src.models.MyModel import MyModel
from src.utils.data.dataset import CustomImageDataset
from torch.utils.data import DataLoader
from src.trainer import Trainer


def print_hi(name):
    # Get the full dataset and split it for train and validation
    full_dataset = CustomImageDataset('data/train.csv', 'data/train_images')
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))


    # training_data = CustomImageDataset('data/train.csv', 'data/train_images')
    num_epochs = 1

    model = MyModel()
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer)
    trainer.train(num_epochs)
    trainer.validate()


if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
