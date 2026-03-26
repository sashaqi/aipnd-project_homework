import torch
import torch.nn as nn
from torchvision import models


ARCH_TO_TORCHVISION = {
    "vgg13": "vgg13",
    "vgg16": "vgg16",
    "vgg19": "vgg19",
    "alexnet": "alexnet",
    # Udacity project commonly refers to these as "densenet" / "resnet".
    "densenet": "densenet121",
    "resnet": "resnet50",
}


def _get_pretrained_weights(tv_model_name: str):
    """
    Prefer the modern `weights=` API when available, but fall back to `pretrained=`.
    """
    weights_lookup = {
        "vgg13": getattr(models, "VGG13_Weights", None),
        "vgg16": getattr(models, "VGG16_Weights", None),
        "vgg19": getattr(models, "VGG19_Weights", None),
        "alexnet": getattr(models, "AlexNet_Weights", None),
        "densenet121": getattr(models, "DenseNet121_Weights", None),
        "resnet50": getattr(models, "ResNet50_Weights", None),
    }
    weights_enum = weights_lookup.get(tv_model_name)
    if weights_enum is not None and hasattr(weights_enum, "DEFAULT"):
        return weights_enum.DEFAULT
    return None


def build_model(arch: str, num_classes: int, hidden_units: int, pretrained: bool = True) -> nn.Module:
    """
    Build a pretrained backbone + a trainable classifier head.
    The head outputs log-probabilities via LogSoftmax (so training uses NLLLoss).
    """
    if arch not in ARCH_TO_TORCHVISION:
        raise ValueError(f"Unsupported architecture: {arch}. Choose from {sorted(ARCH_TO_TORCHVISION.keys())}")

    tv_model_name = ARCH_TO_TORCHVISION[arch]
    constructor = getattr(models, tv_model_name)

    if pretrained:
        weights = _get_pretrained_weights(tv_model_name)
        if weights is not None:
            base_model = constructor(weights=weights)
        else:
            base_model = constructor(pretrained=True)
    else:
        # Avoid downloads during rebuild by not using pretrained weights.
        try:
            base_model = constructor(weights=None)
        except TypeError:
            base_model = constructor(pretrained=False)

    # Freeze backbone parameters; train only the classifier head.
    for p in base_model.parameters():
        p.requires_grad = False

    dropout_p = 0.2  # matches the notebook's LogSoftmax head style

    if arch.startswith("vgg"):
        in_features = base_model.classifier[0].in_features
        new_classifier = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_units, num_classes),
            nn.LogSoftmax(dim=1),
        )
        base_model.classifier = new_classifier
    elif arch == "alexnet":
        # AlexNet has the first classifier Linear at index 1.
        in_features = base_model.classifier[1].in_features
        new_classifier = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_units, num_classes),
            nn.LogSoftmax(dim=1),
        )
        base_model.classifier = new_classifier
    elif arch == "densenet":
        in_features = base_model.classifier.in_features
        base_model.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_units, num_classes),
            nn.LogSoftmax(dim=1),
        )
    elif arch == "resnet":
        in_features = base_model.fc.in_features
        base_model.fc = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_units, num_classes),
            nn.LogSoftmax(dim=1),
        )

    # Unfreeze classifier head parameters only.
    head = get_classifier_head(base_model, arch)
    for p in head.parameters():
        p.requires_grad = True

    return base_model


def get_classifier_head(model: nn.Module, arch: str) -> nn.Module:
    if arch.startswith("vgg") or arch in {"alexnet", "densenet"}:
        return model.classifier
    if arch == "resnet":
        return model.fc
    raise ValueError(f"Unsupported architecture: {arch}")


def set_classifier_head(model: nn.Module, arch: str, head: nn.Module) -> None:
    if arch.startswith("vgg") or arch in {"alexnet", "densenet"}:
        model.classifier = head
        return
    if arch == "resnet":
        model.fc = head
        return
    raise ValueError(f"Unsupported architecture: {arch}")


def infer_hidden_units_from_head(head: nn.Module) -> int:
    """
    Try to infer the hidden_units from a classifier head by finding the first Linear layer.
    """
    if isinstance(head, nn.Sequential):
        for layer in head:
            if isinstance(layer, nn.Linear):
                return layer.out_features
    # Fall back: search for any Linear in the module.
    for m in head.modules():
        if isinstance(m, nn.Linear):
            return m.out_features
    raise ValueError("Could not infer hidden_units from classifier head.")

