#!/usr/bin/env python3
# pyright: reportMissingImports=false
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from checkpoint_utils import save_checkpoint
from data_utils import get_data_loaders
from modeling import build_model, get_classifier_head


def get_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(
    data_directory: str,
    *,
    save_dir: str,
    arch: str,
    learning_rate: float,
    hidden_units: int,
    epochs: int,
    batch_size: int = 64,
    use_gpu: bool = False,
):
    device = get_device(use_gpu)
    train_loader, valid_loader, _, class_to_idx = get_data_loaders(data_directory, batch_size=batch_size)

    num_classes = len(class_to_idx)

    model = build_model(arch=arch, num_classes=num_classes, hidden_units=hidden_units, pretrained=True)
    model.to(device)

    head = get_classifier_head(model, arch)
    optimizer = optim.Adam(head.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for e in range(epochs):
        model.train()
        train_loss_sum = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                log_ps = model(images)
                loss = criterion(log_ps, labels)

                val_loss_sum += loss.item()

                ps = torch.exp(log_ps)
                top_class = ps.argmax(dim=1)
                val_correct += (top_class == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss_sum / len(valid_loader)
        val_accuracy = val_correct / max(1, val_total)

        print(
            f"Epoch {e+1}/{epochs}.. "
            f"Train loss: {avg_train_loss:.3f}.. "
            f"Val loss: {avg_val_loss:.3f}.. "
            f"Val accuracy: {val_accuracy*100:.2f}%"
        )

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
    save_checkpoint(
        checkpoint_path,
        arch=arch,
        hidden_units=hidden_units,
        model=model,
        optimizer=optimizer,
        class_to_idx=class_to_idx,
        epochs=epochs,
    )
    print(f"Saved checkpoint to: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train an image classifier on a dataset.")
    parser.add_argument("data_directory", help="Path to dataset root containing train/valid/test folders.")
    parser.add_argument("--save_dir", default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--arch", default="vgg13", help='Choose architecture (e.g. "vgg13").')
    parser.add_argument("--learning_rate", type=float, default=0.003, help="Learning rate.")
    parser.add_argument("--hidden_units", type=int, default=2048, help="Hidden units in the classifier head.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training if available.")
    args = parser.parse_args()

    train(
        args.data_directory,
        save_dir=args.save_dir,
        arch=args.arch,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        epochs=args.epochs,
        use_gpu=args.gpu,
    )


if __name__ == "__main__":
    main()

