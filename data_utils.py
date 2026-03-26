import json
import os
from typing import Dict, Tuple, Optional

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms():
    train_transforms = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    valid_test_transforms = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transforms, valid_test_transforms


def get_data_loaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """
    Expects:
      data_dir/
        train/
        valid/
        test/
    """
    train_transforms, valid_test_transforms = get_transforms()

    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(root=valid_dir, transform=valid_test_transforms)

    # Not required by the prompt, but keeping it is helpful for sanity checks.
    test_data = None
    if os.path.isdir(test_dir):
        test_data = datasets.ImageFolder(root=test_dir, transform=valid_test_transforms)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = None
    if test_data is not None:
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, valid_loader, test_loader, train_data.class_to_idx


def process_image(image_path: str) -> torch.Tensor:
    """
    Preprocess a PIL image to match training transforms.

    Returns:
      FloatTensor of shape [1, 3, 224, 224]
    """
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img)  # [3, 224, 224]
    return tensor.unsqueeze(0)  # [1, 3, 224, 224]


def load_category_names(category_names_path: str) -> Dict[str, str]:
    with open(category_names_path, "r") as f:
        return json.load(f)


def invert_class_to_idx(class_to_idx: Dict[str, int]) -> Dict[int, str]:
    return {idx: class_label for class_label, idx in class_to_idx.items()}

