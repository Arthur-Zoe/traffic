from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from augmentations import CORRUPTION_NAMES, RandomBadWeather, apply_corruption
from datasets import build_class_mapping, resolve_image_path
from metrics import compute_macro_f1, confusion_matrix_np, per_class_prf, save_confusion_matrix_csv
from models import build_model, checkpoint_state_dict


DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


class EvaluationDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        csv_path: str | Path,
        path_col: str,
        label_col: str,
        img_size: int,
        mean: list[float],
        std: list[float],
        mode: str = "clean",
        severity: str = "medium",
        corruption: str | None = None,
        seed: int = 42,
        max_samples: int | None = None,
        class_to_idx: dict[str, int] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.csv_path = Path(csv_path)
        if not self.csv_path.is_absolute():
            self.csv_path = self.data_dir / self.csv_path
        self.df = pd.read_csv(self.csv_path)
        if path_col not in self.df.columns:
            raise ValueError(f"CSV 缺少路径列 {path_col}: {self.csv_path}")
        if label_col not in self.df.columns:
            raise ValueError(f"CSV 缺少标签列 {label_col}: {self.csv_path}")
        if max_samples is not None and max_samples > 0:
            self.df = self.df.iloc[:max_samples].reset_index(drop=True)
        self.paths = self.df[path_col].astype(str).tolist()
        labels_raw = self.df[label_col].astype(str).tolist()
        self.class_to_idx = class_to_idx or build_class_mapping(labels_raw, numeric_identity=True)[0]
        self.labels = [self.class_to_idx[str(label)] for label in labels_raw]
        self.mode = mode
        self.severity = severity
        self.corruption = corruption
        self.seed = seed
        self.tf = transforms.Compose(
            [
                transforms.Resize(int(img_size * 1.15)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def _corrupt(self, img: Image.Image, idx: int) -> Image.Image:
        if self.mode == "clean":
            return img.convert("RGB")
        local_seed = self.seed + idx * 1009
        if self.mode == "per-corruption":
            if self.corruption is None:
                raise ValueError("per-corruption mode requires corruption")
            return apply_corruption(img, self.corruption, self.severity, seed=local_seed)
        if self.mode == "random-stress":
            aug = RandomBadWeather(p=1.0, severity=self.severity, max_ops=3, seed=local_seed)
            return aug(img)
        raise ValueError(f"Unsupported mode: {self.mode}")

    def __getitem__(self, idx: int):
        rel_path = self.paths[idx]
        path = resolve_image_path(self.data_dir, rel_path)
        with Image.open(path) as img:
            image = img.convert("RGB")
        image = self._corrupt(image, idx)
        return self.tf(image), int(self.labels[idx]), rel_path


def checkpoint_metadata(checkpoint: str | Path) -> dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    idx_to_class_raw = ckpt.get("idx_to_class")
    if isinstance(idx_to_class_raw, dict):
        idx_to_class = {int(k): str(v) for k, v in idx_to_class_raw.items()}
    elif ckpt.get("classes") is not None:
        idx_to_class = {i: str(v) for i, v in enumerate(ckpt["classes"])}
    else:
        n = int(ckpt.get("num_classes", 43))
        idx_to_class = {i: str(i) for i in range(n)}
    num_classes = int(ckpt.get("num_classes", max(idx_to_class.keys(), default=42) + 1))
    return {
        "raw": ckpt,
        "model_name": ckpt.get("model_name", "efficientnet_b0"),
        "img_size": int(ckpt.get("img_size", 224)),
        "num_classes": num_classes,
        "mean": list(ckpt.get("mean", DEFAULT_MEAN)),
        "std": list(ckpt.get("std", DEFAULT_STD)),
        "idx_to_class": idx_to_class,
        "class_to_idx": {str(v): int(k) for k, v in idx_to_class.items()},
    }


def load_model_from_checkpoint(checkpoint: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    meta = checkpoint_metadata(checkpoint)
    model = build_model(meta["model_name"], meta["num_classes"], pretrained=False)
    state = checkpoint_state_dict(meta["raw"])
    clean_state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(clean_state, strict=True)
    model.to(device)
    model.eval()
    return model, meta


@torch.no_grad()
def predict_dataset(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int], list[str], list[float]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    paths: list[str] = []
    confidences: list[float] = []
    for images, labels, rel_paths in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)
        y_true.extend(labels.numpy().astype(int).tolist())
        y_pred.extend(preds.cpu().numpy().astype(int).tolist())
        confidences.extend(conf.cpu().numpy().astype(float).tolist())
        paths.extend(list(rel_paths))
    return y_true, y_pred, paths, confidences


def summarize_predictions(y_true: list[int], y_pred: list[int], num_classes: int) -> dict[str, Any]:
    total = len(y_true)
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    per_class = per_class_prf(y_true, y_pred, num_classes)
    errors = []
    for row in per_class:
        support = int(row["support"])
        recall = float(row["recall"])
        errors.append(
            {
                "class_id": int(row["class_id"]),
                "support": support,
                "errors": int(round(support * (1.0 - recall))),
                "recall": recall,
                "f1": float(row["f1"]),
            }
        )
    errors = sorted(errors, key=lambda x: (-x["errors"], x["recall"]))[:10]
    return {
        "accuracy": float(correct / total) if total else 0.0,
        "macro_f1": compute_macro_f1(y_true, y_pred, num_classes, ignore_empty=True),
        "total": total,
        "per_class": per_class,
        "worst_classes": errors,
    }


def write_predictions_csv(
    output: str | Path,
    paths: list[str],
    y_true: list[int],
    y_pred: list[int],
    confidences: list[float],
) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true", "pred", "confidence", "correct"])
        for path, true, pred, conf in zip(paths, y_true, y_pred, confidences):
            writer.writerow([path, true, pred, f"{conf:.8f}", int(true == pred)])


def evaluate_once(
    model: torch.nn.Module,
    meta: dict[str, Any],
    args: argparse.Namespace,
    mode: str,
    seed: int,
    corruption: str | None = None,
    output_prefix: Path | None = None,
) -> dict[str, Any]:
    dataset = EvaluationDataset(
        data_dir=args.data_dir,
        csv_path=args.csv,
        path_col=args.path_col,
        label_col=args.label_col,
        img_size=meta["img_size"],
        mean=meta["mean"],
        std=meta["std"],
        mode=mode,
        severity=args.severity,
        corruption=corruption,
        seed=seed,
        max_samples=args.max_samples,
        class_to_idx=meta.get("class_to_idx"),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device == "cuda",
    )
    y_true, y_pred, paths, confidences = predict_dataset(model, loader, torch.device(args.device))
    summary = summarize_predictions(y_true, y_pred, meta["num_classes"])
    summary.update({"mode": mode, "seed": seed, "severity": args.severity, "corruption": corruption})
    if output_prefix is not None:
        write_predictions_csv(output_prefix.with_name(output_prefix.name + "_predictions.csv"), paths, y_true, y_pred, confidences)
        cm = confusion_matrix_np(y_true, y_pred, meta["num_classes"])
        classes = [meta["idx_to_class"].get(i, str(i)) for i in range(meta["num_classes"])]
        save_confusion_matrix_csv(cm, output_prefix.with_name(output_prefix.name + "_confusion.csv"), classes)
    return summary


def aggregate_repeats(results: list[dict[str, Any]]) -> dict[str, Any]:
    acc = np.asarray([row["accuracy"] for row in results], dtype=np.float64)
    f1 = np.asarray([row["macro_f1"] for row in results], dtype=np.float64)
    return {
        "accuracy_mean": float(acc.mean()) if len(acc) else 0.0,
        "accuracy_std": float(acc.std(ddof=0)) if len(acc) else 0.0,
        "macro_f1_mean": float(f1.mean()) if len(f1) else 0.0,
        "macro_f1_std": float(f1.std(ddof=0)) if len(f1) else 0.0,
        "runs": results,
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    model, meta = load_model_from_checkpoint(args.checkpoint, device)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent / f"eval_{args.mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Model:", meta["model_name"])
    print("Image size:", meta["img_size"])
    print("Classes:", meta["num_classes"])

    if args.mode == "clean":
        result = evaluate_once(model, meta, args, "clean", args.seed, output_prefix=output_dir / "clean")
        summary = {"checkpoint": str(args.checkpoint), "metadata": {k: v for k, v in meta.items() if k != "raw"}, "clean": result}
    elif args.mode == "random-stress":
        runs = []
        for repeat in range(args.repeats):
            run_seed = args.seed + repeat * 10000
            prefix = output_dir / f"random_stress_seed{run_seed}"
            runs.append(evaluate_once(model, meta, args, "random-stress", run_seed, output_prefix=prefix))
        summary = {
            "checkpoint": str(args.checkpoint),
            "metadata": {k: v for k, v in meta.items() if k != "raw"},
            "random_stress": aggregate_repeats(runs),
        }
    elif args.mode == "per-corruption":
        results = {}
        for corruption in args.corruptions:
            prefix = output_dir / corruption
            results[corruption] = evaluate_once(model, meta, args, "per-corruption", args.seed, corruption, prefix)
        summary = {"checkpoint": str(args.checkpoint), "metadata": {k: v for k, v in meta.items() if k != "raw"}, "per_corruption": results}
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    output_json = Path(args.output_json) if args.output_json else output_dir / "summary.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Summary:", output_json)
    if args.mode == "clean":
        print(f"Accuracy: {summary['clean']['accuracy']:.6f}")
        print(f"Macro-F1: {summary['clean']['macro_f1']:.6f}")
    elif args.mode == "random-stress":
        agg = summary["random_stress"]
        print(f"Accuracy mean/std: {agg['accuracy_mean']:.6f} / {agg['accuracy_std']:.6f}")
        print(f"Macro-F1 mean/std: {agg['macro_f1_mean']:.6f} / {agg['macro_f1_std']:.6f}")
    else:
        for name, row in summary["per_corruption"].items():
            print(f"{name}: acc={row['accuracy']:.6f}, macro_f1={row['macro_f1']:.6f}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified evaluator for traffic sign checkpoints")
    parser.add_argument("--data-dir", default="data/gtsrb")
    parser.add_argument("--csv", default="Test.csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["clean", "random-stress", "per-corruption"], default="clean")
    parser.add_argument("--severity", choices=["light", "medium", "strong"], default="medium")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--path-col", default="Path")
    parser.add_argument("--label-col", default="ClassId")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--corruptions", nargs="+", default=CORRUPTION_NAMES)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_evaluation(parse_args())
