from pathlib import Path
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from train_gtsrb import (
    GTSRBDataset,
    RandomBadWeather,
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

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = logits.argmax(dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    acc = correct / max(total, 1)
    f1 = compute_macro_f1(y_true, y_pred, NUM_CLASSES)
    return acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/gtsrb")
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--weather-prob", type=float, default=1.0)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_name = ckpt.get("model_name", "efficientnet_b0")
    img_size = ckpt.get("img_size", 224)

    tf = transforms.Compose([
        RandomBadWeather(p=args.weather_prob),
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_set = GTSRBDataset(args.data_dir, "Test.csv", transform=tf)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(model_name, NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)

    acc, f1 = evaluate(model, test_loader, device)

    print("Weather stress test")
    print(f"Accuracy: {acc:.6f}")
    print(f"Macro-F1: {f1:.6f}")


if __name__ == "__main__":
    main()