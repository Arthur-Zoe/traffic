from __future__ import annotations
from io import BytesIO

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm


NUM_CLASSES = 43
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RandomBadWeather:
    """
    更强的恶劣天气 / 成像退化增强：
    - 支持多种干扰叠加
    - 雨、雾、雪、暗光、模糊、噪声、压缩、低分辨率、阴影、眩光
    """

    def __init__(self, p: float = 0.45, max_ops: int = 3):
        self.p = p
        self.max_ops = max_ops

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")

        if random.random() > self.p:
            return img

        ops = [
            self.add_rain,
            self.add_fog,
            self.add_snow,
            self.add_dark,
            self.add_motion_blur,
            self.add_gaussian_blur,
            self.add_noise,
            self.add_jpeg_compression,
            self.add_low_resolution,
            self.add_shadow,
            self.add_glare,
        ]

        n_ops = random.randint(1, self.max_ops)
        chosen_ops = random.sample(ops, k=n_ops)

        for op in chosen_ops:
            if random.random() < 0.85:
                img = op(img)

        return img

    def add_rain(self, img: Image.Image) -> Image.Image:
        img = img.copy().convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        num_lines = random.randint(30, 90)
        angle = random.randint(-5, 8)

        for _ in range(num_lines):
            x = random.randint(-w // 4, w)
            y = random.randint(0, h)
            length = random.randint(8, 24)
            dx = angle
            dy = length
            color = random.choice([(170, 170, 170), (190, 190, 190), (210, 210, 210)])
            draw.line((x, y, x + dx, y + dy), fill=color, width=1)

        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.15, 0.45)))
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.65, 0.95))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.75, 1.05))
        return img

    def add_fog(self, img: Image.Image) -> Image.Image:
        img = img.copy().convert("RGB")
        w, h = img.size

        fog_color = random.randint(210, 245)
        fog = Image.new("RGB", (w, h), (fog_color, fog_color, fog_color))
        alpha = random.uniform(0.18, 0.45)
        img = Image.blend(img, fog, alpha)

        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.45, 0.85))
        img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.5, 0.9))
        return img

    def add_snow(self, img: Image.Image) -> Image.Image:
        img = img.copy().convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        num_dots = random.randint(60, 180)

        for _ in range(num_dots):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            r = random.choice([1, 1, 1, 2, 2, 3])
            gray = random.randint(220, 255)
            draw.ellipse((x, y, x + r, y + r), fill=(gray, gray, gray))

        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.15))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.65, 1.0))
        return img

    def add_dark(self, img: Image.Image) -> Image.Image:
        img = img.copy().convert("RGB")
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.28, 0.75))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.75, 1.25))
        img = ImageEnhance.Color(img).enhance(random.uniform(0.65, 1.05))
        return img

    def add_gaussian_blur(self, img: Image.Image) -> Image.Image:
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.8)))

    def add_motion_blur(self, img: Image.Image) -> Image.Image:
        # 简单水平运动模糊核
        k = random.choice([3, 5])
        kernel = [0] * (k * k)

        if random.random() < 0.5:
            # 水平方向
            for i in range(k):
                kernel[k * (k // 2) + i] = 1
        else:
            # 垂直方向
            for i in range(k):
                kernel[i * k + (k // 2)] = 1

        kernel = [v / k for v in kernel]
        return img.filter(ImageFilter.Kernel((k, k), kernel, scale=1.0))

    def add_noise(self, img: Image.Image) -> Image.Image:
        arr = np.array(img).astype(np.float32)
        sigma = random.uniform(4, 18)
        noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def add_jpeg_compression(self, img: Image.Image) -> Image.Image:
        buffer = BytesIO()
        quality = random.randint(25, 75)
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def add_low_resolution(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = random.uniform(0.45, 0.8)
        nw = max(8, int(w * scale))
        nh = max(8, int(h * scale))

        small = img.resize((nw, nh), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)

    def add_shadow(self, img: Image.Image) -> Image.Image:
        img = img.copy().convert("RGB")
        w, h = img.size

        overlay = Image.new("RGB", (w, h), (0, 0, 0))
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        x1 = random.randint(-w // 2, w // 2)
        x2 = random.randint(w // 2, int(w * 1.5))
        polygon = [
            (x1, 0),
            (x2, 0),
            (random.randint(w // 2, int(w * 1.5)), h),
            (random.randint(-w // 2, w // 2), h),
        ]

        shadow_alpha = random.randint(45, 110)
        draw.polygon(polygon, fill=shadow_alpha)

        return Image.composite(overlay, img, mask)

    def add_glare(self, img: Image.Image) -> Image.Image:
        img = img.copy().convert("RGB")
        w, h = img.size

        overlay = Image.new("RGB", (w, h), (255, 255, 255))
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        cx = random.randint(0, w)
        cy = random.randint(0, h)
        r = random.randint(max(8, w // 8), max(12, w // 3))
        alpha = random.randint(35, 95)

        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=alpha)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(6, 14)))

        img = Image.composite(overlay, img, mask)
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.05))
        return img


class GTSRBDataset(Dataset):
    def __init__(self, root: str | Path, csv_name: str, indices=None, transform=None):
        self.root = Path(root)
        self.csv_path = self.root / csv_name
        self.df = pd.read_csv(self.csv_path)

        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)

        self.paths = self.df["Path"].astype(str).tolist()
        self.labels = self.df["ClassId"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def _resolve_path(self, rel_path: str) -> Path:
        p = self.root / rel_path
        if p.exists():
            return p

        parts = Path(rel_path).parts
        if len(parts) > 0:
            first = parts[0]
            rest = parts[1:]

            candidates = [
                self.root / Path(first.lower(), *rest),
                self.root / Path(first.upper(), *rest),
                self.root / Path(first.capitalize(), *rest),
            ]

            for c in candidates:
                if c.exists():
                    return c

        return p

    def __getitem__(self, idx):
        img_path = self._resolve_path(self.paths[idx])
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def stratified_split(labels, val_ratio=0.15, seed=42):
    rng = random.Random(seed)
    by_class = defaultdict(list)

    for i, y in enumerate(labels):
        by_class[int(y)].append(i)

    train_idx = []
    val_idx = []

    for _, indices in by_class.items():
        rng.shuffle(indices)

        if len(indices) <= 1:
            train_idx.extend(indices)
            continue

        n_val = max(1, int(len(indices) * val_ratio))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return train_idx, val_idx


def build_transforms(img_size: int, weather_aug: bool, weather_prob: float):
    train_ops = []

    if weather_aug:
        train_ops.append(RandomBadWeather(p=weather_prob))

    train_ops.extend(
        [
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.72, 1.0),
                ratio=(0.85, 1.15),
            ),
            transforms.RandomRotation(degrees=12),
            transforms.RandomPerspective(distortion_scale=0.18, p=0.25),
            transforms.ColorJitter(
                brightness=0.35,
                contrast=0.35,
                saturation=0.25,
                hue=0.04,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.3))],
                p=0.25,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.12),
                ratio=(0.3, 3.3),
                value="random",
            ),
        ]
    )

    train_tf = transforms.Compose(train_ops)

    val_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_tf, val_tf


def build_model(model_name: str, num_classes: int, pretrained: bool):
    weights = None

    if model_name == "efficientnet_b0":
        if pretrained:
            try:
                weights = models.EfficientNet_B0_Weights.DEFAULT
                model = models.efficientnet_b0(weights=weights)
            except Exception:
                print("[WARNING] ImageNet 权重加载失败，改为从零训练。")
                model = models.efficientnet_b0(weights=None)
        else:
            model = models.efficientnet_b0(weights=None)

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "resnet18":
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
                model = models.resnet18(weights=weights)
            except Exception:
                print("[WARNING] ImageNet 权重加载失败，改为从零训练。")
                model = models.resnet18(weights=None)
        else:
            model = models.resnet18(weights=None)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if model_name == "mobilenet_v3_large":
        if pretrained:
            try:
                weights = models.MobileNet_V3_Large_Weights.DEFAULT
                model = models.mobilenet_v3_large(weights=weights)
            except Exception:
                print("[WARNING] ImageNet 权重加载失败，改为从零训练。")
                model = models.mobilenet_v3_large(weights=None)
        else:
            model = models.mobilenet_v3_large(weights=None)

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unknown model: {model_name}")


def compute_macro_f1(y_true, y_pred, num_classes: int):
    tp = np.zeros(num_classes, dtype=np.float64)
    fp = np.zeros(num_classes, dtype=np.float64)
    fn = np.zeros(num_classes, dtype=np.float64)

    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    f1_list = []
    for c in range(num_classes):
        precision = tp[c] / (tp[c] + fp[c] + 1e-12)
        recall = tp[c] / (tp[c] + fn[c] + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1_list.append(f1)

    return float(np.mean(f1_list))


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    model.eval()

    total = 0
    correct = 0
    y_true = []
    y_pred = []

    for images, labels in tqdm(loader, desc="Val", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        preds = logits.argmax(dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    acc = correct / max(total, 1)
    macro_f1 = compute_macro_f1(y_true, y_pred, num_classes)

    return acc, macro_f1


def make_class_weights(labels, mode: str):
    if mode == "none":
        return None

    counts = np.bincount(np.array(labels), minlength=NUM_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)

    if mode == "inverse":
        weights = 1.0 / counts
    elif mode == "sqrt_inverse":
        weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unknown class weight mode: {mode}")

    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train(args):
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_csv = data_dir / "Train.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"找不到 Train.csv: {train_csv}")

    df = pd.read_csv(train_csv)
    labels = df["ClassId"].astype(int).tolist()

    print("数据集路径:", data_dir)
    print("训练图片数:", len(df))
    print("类别数量:", len(set(labels)))
    print("类别范围:", min(labels), max(labels))

    train_idx, val_idx = stratified_split(labels, val_ratio=args.val_ratio, seed=args.seed)

    train_tf, val_tf = build_transforms(
        img_size=args.img_size,
        weather_aug=args.weather_aug,
        weather_prob=args.weather_prob,
    )

    train_set = GTSRBDataset(data_dir, "Train.csv", indices=train_idx, transform=train_tf)
    val_set = GTSRBDataset(data_dir, "Train.csv", indices=val_idx, transform=val_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Device:", device)

    model = build_model(args.model, NUM_CLASSES, args.pretrained)
    model = model.to(device)

    class_weights = make_class_weights([labels[i] for i in train_idx], args.class_weight)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.02,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        total = 0
        correct = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for images, labels_batch in pbar:
            images = images.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels_batch)

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            preds = logits.argmax(dim=1)
            bs = labels_batch.size(0)

            running_loss += loss.item() * bs
            total += bs
            correct += (preds == labels_batch).sum().item()

            pbar.set_postfix(
                loss=running_loss / max(total, 1),
                acc=correct / max(total, 1),
                lr=optimizer.param_groups[0]["lr"],
            )

        scheduler.step()

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_acc, val_f1 = evaluate(model, val_loader, device, NUM_CLASSES)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.5f} "
            f"train_acc={train_acc:.5f} "
            f"val_acc={val_acc:.5f} "
            f"val_macro_f1={val_f1:.5f}"
        )

        last_ckpt = {
            "model": model.state_dict(),
            "model_name": args.model,
            "num_classes": NUM_CLASSES,
            "img_size": args.img_size,
            "epoch": epoch,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        }

        torch.save(last_ckpt, output_dir / "last.pt")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            torch.save(last_ckpt, output_dir / "best.pt")
            print(f"[SAVE] best.pt  val_acc={best_acc:.5f}, val_macro_f1={best_f1:.5f}")

    print("训练完成")
    print("Best val_acc:", best_acc)
    print("Best val_macro_f1:", best_f1)
    print("Best checkpoint:", output_dir / "best.pt")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default="data/gtsrb")
    parser.add_argument("--output-dir", type=str, default="outputs/gtsrb_effb0_seed42")

    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet_b0",
        choices=["efficientnet_b0", "resnet18", "mobilenet_v3_large"],
    )

    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weight", type=str, default="sqrt_inverse", choices=["none", "inverse", "sqrt_inverse"])

    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--weather-aug", action="store_true")
    parser.add_argument("--weather-prob", type=float, default=0.35)

    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
