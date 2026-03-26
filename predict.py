#!/usr/bin/env python3
# pyright: reportMissingImports=false
import argparse

from typing import Optional

import torch

from checkpoint_utils import load_checkpoint
from data_utils import load_category_names, process_image, invert_class_to_idx


def predict_image(
    image_path: str,
    checkpoint_path: str,
    *,
    top_k: int = 1,
    category_names_path: Optional[str] = None,
    use_gpu: bool = False,
):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    model, _ = load_checkpoint(checkpoint_path, device=device)

    if top_k < 1:
        raise ValueError("--top_k must be >= 1")

    model.eval()
    with torch.no_grad():
        image_tensor = process_image(image_path).to(device)  # [1, 3, 224, 224]
        log_ps = model(image_tensor)
        ps = torch.exp(log_ps)  # convert log-probs -> probs

        top_k = min(top_k, ps.shape[1])
        top_probs, top_idxs = ps.topk(top_k, dim=1)

    # [1, k] -> [k]
    top_probs = top_probs.squeeze(0).cpu()
    top_idxs = top_idxs.squeeze(0).cpu()

    idx_to_class = invert_class_to_idx(model.class_to_idx)
    top_classes = [idx_to_class[i.item()] for i in top_idxs]

    if category_names_path is not None:
        cat_to_name = load_category_names(category_names_path)
        top_labels = [cat_to_name[c] for c in top_classes]
    else:
        top_labels = top_classes

    return top_labels, top_probs.tolist()


def main():
    parser = argparse.ArgumentParser(description="Predict flower class from an image.")
    parser.add_argument("image_path", help="Path to an input image.")
    parser.add_argument("checkpoint_path", help="Path to a saved checkpoint (checkpoint.pth).")
    parser.add_argument("--top_k", type=int, default=1, help="Return top K most likely classes.")
    parser.add_argument(
        "--category_names",
        default=None,
        help="Path to cat_to_name.json mapping class labels to real names.",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference if available.")
    args = parser.parse_args()

    labels, probs = predict_image(
        args.image_path,
        args.checkpoint_path,
        top_k=args.top_k,
        category_names_path=args.category_names,
        use_gpu=args.gpu,
    )

    for label, prob in zip(labels, probs):
        print(f"{label}: {prob:.6f}")


if __name__ == "__main__":
    main()

