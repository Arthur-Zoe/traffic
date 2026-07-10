from pathlib import Path
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from train_gtsrb import (
    GTSRBDataset,
    build_model,
    compute_macro_f1,
    NUM_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total = 0
    correct = 0
    y_true = []
    y_pred = []

    for images, labels in tqdm(loader, desc="Test"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        preds = logits.argmax(dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    acc = correct / max(total, 1)
    macro_f1 = compute_macro_f1(y_true, y_pred, NUM_CLASSES)

    return acc, macro_f1, np.array(y_true), np.array(y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/gtsrb")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model_name = ckpt.get("model_name", "efficientnet_b0")
    img_size = ckpt.get("img_size", 224)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Checkpoint:", ckpt_path)
    print("Model:", model_name)
    print("Image size:", img_size)

    test_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    test_set = GTSRBDataset(
        root=args.data_dir,
        csv_name="Test.csv",
        transform=test_tf,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = build_model(model_name, NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)

    acc, macro_f1, y_true, y_pred = evaluate(model, test_loader, device)

    print(f"Test accuracy: {acc:.6f}")
    print(f"Test macro-F1: {macro_f1:.6f}")

    print("\n错误最多的类别：")
    errors = []
    for c in range(NUM_CLASSES):
        mask = y_true == c
        total_c = mask.sum()
        if total_c == 0:
            continue
        correct_c = ((y_true == c) & (y_pred == c)).sum()
        acc_c = correct_c / total_c
        errors.append((c, total_c, acc_c))

    errors = sorted(errors, key=lambda x: x[2])

    for c, total_c, acc_c in errors[:10]:
        print(f"Class {c:02d}: acc={acc_c:.4f}, support={total_c}")


if __name__ == "__main__":
    main()