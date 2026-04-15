import torch
import random
import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataloader(dataset_name, batch_size=128, train=True, img_size=28):
    if dataset_name in ['mnist', 'fashion_mnist']:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        if dataset_name == 'mnist':
            ds = datasets.MNIST('./data', train=train, download=True, transform=transform)
        else:
            ds = datasets.FashionMNIST('./data', train=train, download=True, transform=transform)
        in_channels = 1
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        ds = datasets.CIFAR10('./data', train=train, download=True, transform=transform)
        in_channels = 3
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=0, pin_memory=False)
    return loader, in_channels


def save_checkpoint(model, optimizer, epoch, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt['epoch'], ckpt['loss']
