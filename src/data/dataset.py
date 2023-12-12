import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.target_map = {'CC': 0, 'EC': 1, 'HGSC': 2, 'LGSC': 3, 'MC': 4, 'Other': 5}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 0])) + ".png"
        image = read_image(img_path)
        label = self.target_map[self.img_labels.iloc[idx, 1]]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class CustomImageDatasetThumbnail(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.target_map = {'CC': 0, 'EC': 1, 'HGSC': 2, 'LGSC': 3, 'MC': 4, 'Other': 5}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 0])) + "_thumbnail.png"
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        label = self.target_map[label]
        if self.target_transform:
            label = self.target_transform(label)
        return image, label