import importlib
import torch


def get_device():
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} device")
    return device


def model_loader(path, class_name, device):
    module = importlib.import_module(path + class_name)
    try:
        my_class = getattr(module, class_name)
    except AttributeError:
        raise ValueError(f"Model class '{class_name}' not found in {path + class_name}.")
    return my_class().to(device)
