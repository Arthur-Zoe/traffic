from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from augmentations import RandomBadWeather
from datasets import CSVImageDataset, GTSRBDataset, build_class_mapping, export_class_mapping
from metrics import compute_macro_f1
from models import SUPPORTED_MODELS, build_model, checkpoint_state_dict, load_state_flexible
from presets import PRESETS, get_preset


NUM_CLASSES = 43
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


DEFAULT_CONFIG: dict[str, Any] = {
    "data_dir": "data/gtsrb",
    "train_csv": "Train.csv",
    "path_col": "Path",
    "label_col": "ClassId",
    "num_classes": 43,
    "auto_num_classes": False,
    "model": "efficientnet_b0",
    "img_size": 224,
    "epochs": 25,
    "batch_size": 32,
    "workers": 4,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "label_smoothing": 0.05,
    "class_weight": "sqrt_inverse",
    "val_ratio": 0.15,
    "seed": 42,
    "pretrained": False,
    "weather_aug": False,
    "weather_prob": 0.35,
    "weather_max_ops": 3,
    "weather_severity": "medium",
    "output_dir": "outputs/gtsrb_effb0_seed42",
    "resume": None,
    "init_checkpoint": None,
    "max_train_samples": None,
    "max_val_samples": None,
    "early_stopping_patience": None,
    "save_every_epoch": False,
    "grad_clip": 1.0,
    "cpu": False,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def stratified_split(labels: list[int], val_ratio: float = 0.15, seed: int = 42) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_class[int(label)].append(idx)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for indices in by_class.values():
        rng.shuffle(indices)
        if len(indices) <= 1 or val_ratio <= 0:
            train_idx.extend(indices)
            continue
        n_val = max(1, int(len(indices) * val_ratio))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def build_transforms(
    img_size: int,
    weather_aug: bool,
    weather_prob: float,
    weather_max_ops: int = 3,
    weather_severity: str = "medium",
) -> tuple[transforms.Compose, transforms.Compose]:
    train_ops: list[Any] = []
    if weather_aug:
        train_ops.append(
            RandomBadWeather(
                p=weather_prob,
                max_ops=weather_max_ops,
                severity=weather_severity,
            )
        )
    train_ops.extend(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.72, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomRotation(degrees=12),
            transforms.RandomPerspective(distortion_scale=0.18, p=0.25),
            transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.04),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.3))], p=0.25),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(train_ops), val_tf


def make_class_weights(labels: list[int], mode: str, num_classes: int = NUM_CLASSES) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    if mode == "inverse":
        weights = 1.0 / counts
    elif mode == "sqrt_inverse":
        weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unknown class weight mode: {mode}")
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in tqdm(loader, desc="Val", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)
        total += labels.size(0)
        correct += int((preds == labels).sum().item())
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
    acc = correct / max(total, 1)
    macro_f1 = compute_macro_f1(y_true, y_pred, num_classes, ignore_empty=True)
    return float(acc), float(macro_f1)


def _resolve_csv(data_dir: Path, train_csv: str | Path) -> Path:
    csv_path = Path(train_csv)
    if not csv_path.is_absolute():
        csv_path = data_dir / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到训练 CSV: {csv_path}")
    return csv_path


def _load_training_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, Path, dict[str, int], dict[int, str], int]:
    data_dir = Path(args.data_dir)
    csv_path = _resolve_csv(data_dir, args.train_csv)
    df = pd.read_csv(csv_path)
    if args.path_col not in df.columns:
        raise ValueError(f"CSV 缺少路径列 {args.path_col}: {csv_path}")
    if args.label_col not in df.columns:
        raise ValueError(f"CSV 缺少标签列 {args.label_col}: {csv_path}")
    labels_raw = df[args.label_col].astype(str).tolist()
    class_to_idx, idx_to_class = build_class_mapping(labels_raw, numeric_identity=True)
    mapped_labels = [class_to_idx[label] for label in labels_raw]
    inferred_num_classes = max(mapped_labels) + 1 if mapped_labels else 0
    num_classes = inferred_num_classes if args.auto_num_classes else int(args.num_classes)
    if num_classes < inferred_num_classes:
        raise ValueError(f"--num-classes={num_classes} 小于数据中推断类别数 {inferred_num_classes}")
    return df, csv_path, class_to_idx, idx_to_class, num_classes


def _subset_indices(indices: list[int], limit: int | None) -> list[int]:
    if limit is None or limit <= 0:
        return indices
    return indices[: min(limit, len(indices))]


def _checkpoint_dict(
    model: nn.Module,
    args: argparse.Namespace,
    epoch: int,
    best_f1: float,
    best_acc: float,
    val_acc: float,
    val_f1: float,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    class_to_idx: dict[str, int],
    idx_to_class: dict[int, str],
    num_classes: int,
) -> dict[str, Any]:
    train_config = vars(args).copy()
    return {
        "model": model.state_dict(),
        "model_name": args.model,
        "num_classes": num_classes,
        "img_size": args.img_size,
        "epoch": epoch,
        "best_macro_f1": best_f1,
        "best_acc": best_acc,
        "val_acc": val_acc,
        "val_macro_f1": val_f1,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
        "classes": [idx_to_class.get(i, str(i)) for i in range(num_classes)],
        "train_config": train_config,
        "args": train_config,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    df, csv_path, class_to_idx, idx_to_class, num_classes = _load_training_frame(args)
    labels = [class_to_idx[str(label)] for label in df[args.label_col].astype(str).tolist()]

    print("数据集路径:", Path(args.data_dir))
    print("训练 CSV:", csv_path)
    print("训练图片数:", len(df))
    print("类别数量:", num_classes)
    print("数据中类别数:", len(set(labels)))

    train_idx, val_idx = stratified_split(labels, val_ratio=args.val_ratio, seed=args.seed)
    train_idx = _subset_indices(train_idx, args.max_train_samples)
    val_idx = _subset_indices(val_idx, args.max_val_samples)

    train_tf, val_tf = build_transforms(
        img_size=args.img_size,
        weather_aug=args.weather_aug,
        weather_prob=args.weather_prob,
        weather_max_ops=args.weather_max_ops,
        weather_severity=args.weather_severity,
    )
    train_set = CSVImageDataset(csv_path, args.data_dir, args.path_col, args.label_col, train_tf, class_to_idx, train_idx)
    val_set = CSVImageDataset(csv_path, args.data_dir, args.path_col, args.label_col, val_tf, class_to_idx, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available() and not args.cpu,
        drop_last=len(train_set) >= args.batch_size,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available() and not args.cpu,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Device:", device)
    model = build_model(args.model, num_classes, args.pretrained).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.02)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 1
    best_f1 = -1.0
    best_acc = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        info = load_state_flexible(model, checkpoint_state_dict(ckpt))
        print("[RESUME] model:", info)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and ckpt["scaler"]:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_f1 = float(ckpt.get("best_macro_f1", ckpt.get("val_macro_f1", -1.0)))
        best_acc = float(ckpt.get("best_acc", ckpt.get("val_acc", -1.0)))
        print(f"[RESUME] start_epoch={start_epoch}, best_f1={best_f1:.6f}")
    elif args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        info = load_state_flexible(model, checkpoint_state_dict(ckpt))
        print("[INIT] loaded compatible weights:", info)
        if info["classifier_skipped"]:
            print("[INIT] 分类头维度不匹配，已跳过:", info["classifier_skipped"])

    class_weights = make_class_weights([labels[i] for i in train_idx], args.class_weight, num_classes)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_class_mapping(class_to_idx, output_dir)
    (output_dir / "train_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    no_improve = 0
    for epoch in range(start_epoch, args.epochs + 1):
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
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            preds = logits.argmax(dim=1)
            bs = labels_batch.size(0)
            running_loss += float(loss.item()) * bs
            total += bs
            correct += int((preds == labels_batch).sum().item())
            pbar.set_postfix(loss=running_loss / max(total, 1), acc=correct / max(total, 1), lr=optimizer.param_groups[0]["lr"])

        scheduler.step()
        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_acc, val_f1 = evaluate(model, val_loader, device, num_classes)
        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.5f} train_acc={train_acc:.5f} "
            f"val_acc={val_acc:.5f} val_macro_f1={val_f1:.5f}"
        )
        ckpt = _checkpoint_dict(
            model,
            args,
            epoch,
            best_f1,
            best_acc,
            val_acc,
            val_f1,
            optimizer,
            scheduler,
            scaler,
            class_to_idx,
            idx_to_class,
            num_classes,
        )
        torch.save(ckpt, output_dir / "last.pt")
        if args.save_every_epoch:
            torch.save(ckpt, output_dir / f"epoch_{epoch:03d}.pt")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            ckpt["best_macro_f1"] = best_f1
            ckpt["best_acc"] = best_acc
            torch.save(ckpt, output_dir / "best.pt")
            no_improve = 0
            print(f"[SAVE] best.pt  val_acc={best_acc:.5f}, val_macro_f1={best_f1:.5f}")
        else:
            no_improve += 1
            if args.early_stopping_patience is not None and no_improve >= args.early_stopping_patience:
                print(f"[EARLY STOP] patience={args.early_stopping_patience}, best_f1={best_f1:.6f}")
                break

    print("训练完成")
    print("Best val_acc:", best_acc)
    print("Best val_macro_f1:", best_f1)
    print("Best checkpoint:", output_dir / "best.pt")


def _add_arg(parser: argparse.ArgumentParser, name: str, **kwargs: Any) -> None:
    default = kwargs.pop("default", None)
    parser.add_argument(name, default=default, **kwargs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, default=None, choices=sorted(PRESETS.keys()))
    for key in [
        "data_dir",
        "train_csv",
        "path_col",
        "label_col",
        "model",
        "output_dir",
        "resume",
        "init_checkpoint",
        "class_weight",
        "weather_severity",
    ]:
        cli = "--" + key.replace("_", "-")
        choices = SUPPORTED_MODELS if key == "model" else (["none", "inverse", "sqrt_inverse"] if key == "class_weight" else None)
        parser.add_argument(cli, type=str, default=None, choices=choices)
    for key in [
        "num_classes",
        "img_size",
        "epochs",
        "batch_size",
        "workers",
        "seed",
        "weather_max_ops",
        "max_train_samples",
        "max_val_samples",
        "early_stopping_patience",
    ]:
        parser.add_argument("--" + key.replace("_", "-"), type=int, default=None)
    for key in ["lr", "weight_decay", "label_smoothing", "val_ratio", "weather_prob", "grad_clip"]:
        parser.add_argument("--" + key.replace("_", "-"), type=float, default=None)
    parser.add_argument("--auto-num-classes", action="store_true", default=None)
    parser.add_argument("--pretrained", action="store_true", default=None)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--weather-aug", action="store_true", default=None)
    parser.add_argument("--no-weather-aug", dest="weather_aug", action="store_false")
    parser.add_argument("--save-every-epoch", action="store_true", default=None)
    parser.add_argument("--cpu", action="store_true", default=None)
    parsed = parser.parse_args(argv)

    config = DEFAULT_CONFIG.copy()
    config.update(get_preset(parsed.preset))
    explicit = vars(parsed)
    for key, value in explicit.items():
        if key == "preset":
            continue
        if value is not None:
            config[key] = value
    config["preset"] = parsed.preset
    return argparse.Namespace(**config)


if __name__ == "__main__":
    train(parse_args())
