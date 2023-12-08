from src.utils.data.dataset import CustomImageDataset
from torch.utils.data import DataLoader


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    training_data = CustomImageDataset('data/train.csv', 'data/train_images')
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
