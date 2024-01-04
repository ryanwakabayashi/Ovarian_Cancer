import torch
import argparse

from torchvision import transforms
from torch.utils.data import DataLoader
from src.data.dataset import CustomImageDataset
from torch.utils.tensorboard import SummaryWriter

from src.trainer import Trainer
from src.utils import get_device, model_loader


def main(model_name):
    writer = SummaryWriter()
    device = get_device()

    # TODO introduce the following to normalize the image
    transform = transforms.Compose([
        transforms.RandomResizedCrop((512, 512)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    full_dataset = CustomImageDataset('data/train.csv', 'data/preprocessed_images', transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size],
                                                               generator=torch.Generator().manual_seed(0))

    num_epochs = 35

    path_to_models = 'src.models.'
    model = model_loader(path_to_models, model_name, device)
    # model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
    # model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=12, pin_memory=False)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, writer, device)
    trainer.train(num_epochs)

    writer.flush()


def parse_args():
    parser = argparse.ArgumentParser(description='Runs training using a give model')
    parser.add_argument('--model', type=str, help='input model name', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.model)
