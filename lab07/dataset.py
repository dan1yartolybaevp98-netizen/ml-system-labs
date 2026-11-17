import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import DATA_DIR, BATCH_SIZE, NUM_WORKERS

def get_loaders(batch_size=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    train_set = datasets.FashionMNIST(
        root=str(DATA_DIR), train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        root=str(DATA_DIR), train=False, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=False
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False
    )
    return train_loader, test_loader

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
