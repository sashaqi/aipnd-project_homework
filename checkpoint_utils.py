from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from data_utils import invert_class_to_idx
from modeling import build_model, get_classifier_head, infer_hidden_units_from_head, set_classifier_head


def save_checkpoint(
    checkpoint_path: str,
    *,
    arch: str,
    hidden_units: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    class_to_idx: Dict[str, int],
    epochs: int,
) -> None:
    head = get_classifier_head(model, arch)
    checkpoint = {
        "arch": arch,
        "hidden_units": hidden_units,
        "classifier": head,
        "state_dict": model.state_dict(),
        # Match the (slightly misspelled) key used in the notebook.
        "optim_stat_dict": optimizer.state_dict(),
        "class_to_idx": class_to_idx,
        "epochs": epochs,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location=device)

    arch = checkpoint.get("arch", "vgg16")
    class_to_idx = checkpoint["class_to_idx"]
    num_classes = len(class_to_idx)

    hidden_units = checkpoint.get("hidden_units")
    if hidden_units is None:
        # Try to infer from stored classifier module.
        head = checkpoint.get("classifier")
        if head is None:
            raise KeyError("Checkpoint missing both 'hidden_units' and 'classifier'.")
        hidden_units = infer_hidden_units_from_head(head)

    model = build_model(arch=arch, num_classes=num_classes, hidden_units=hidden_units, pretrained=False)

    # Restore weights. (state_dict should cover backbone + head.)
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = class_to_idx

    # Ensure head module matches the checkpoint's head structure if present.
    if "classifier" in checkpoint:
        try:
            set_classifier_head(model, arch, checkpoint["classifier"])
        except Exception:
            # Not critical if load_state_dict succeeded; ignore shape mismatch.
            pass

    model.to(device)
    model.eval()
    return model, checkpoint

