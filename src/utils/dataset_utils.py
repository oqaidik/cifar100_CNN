from typing import Tuple
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms():
    """
    Returns train and test transforms for CIFAR-100.
    """
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, test_transform


def get_dataloaders(
    data_dir: str = "data",
    batch_size: int = 128,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates DataLoaders for CIFAR-100 train and test splits.
    """
    data_path = Path(data_dir)
    train_transform, test_transform = get_transforms()

    train_dataset = datasets.CIFAR100(
        root=str(data_path),
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR100(
        root=str(data_path),
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader
