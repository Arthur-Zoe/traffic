from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from augmentations import CORRUPTION_NAMES
from evaluate import evaluate_once, load_model_from_checkpoint
from models import build_model, count_parameters


def benchmark_single_image(
    checkpoint: str | Path,
    model_name: str,
    num_classes: int,
    img_size: int,
    device: str,
    warmup: int = 10,
    runs: int = 50,
) -> float | None:
    if device == "cuda" and not torch.cuda.is_available():
        return None
    torch_device = torch.device(device)
    model, _ = load_model_from_checkpoint(checkpoint, torch_device)
    model.eval()
    image = torch.randn(1, 3, img_size, img_size, device=torch_device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(image)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = model(image)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return float(elapsed / runs)


def recommend(row: dict[str, Any]) -> str:
    model_name = str(row["model_name"])
    img_size = int(row["img_size"])
    clean_f1 = float(row["clean_macro_f1"])
    stress_f1 = float(row["strong_random_stress_macro_f1_mean"])
    cpu_ms = float(row["cpu_single_ms"])
    if "mobile" in model_name or cpu_ms < 12:
        return "fast-inference"
    if img_size >= 256:
        return "high-resolution"
    if stress_f1 >= clean_f1 - 0.08:
        return "severe-weather"
    if clean_f1 >= 0.99:
        return "clean"
    return "balanced"


def compare_checkpoint(checkpoint: str | Path, args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    eval_args = SimpleNamespace(
        data_dir=args.data_dir,
        csv=args.csv,
        path_col=args.path_col,
        label_col=args.label_col,
        severity="strong",
        batch_size=args.batch_size,
        workers=args.workers,
        max_samples=args.max_samples,
        device=args.device,
    )
    device = torch.device(args.device)
    model, meta = load_model_from_checkpoint(checkpoint, device)
    params = count_parameters(model)

    clean = evaluate_once(model, meta, eval_args, "clean", args.seed, output_prefix=None)
    per_corruption: dict[str, dict[str, Any]] = {}
    for corruption in args.corruptions:
        per_corruption[corruption] = evaluate_once(model, meta, eval_args, "per-corruption", args.seed, corruption, output_prefix=None)
    stress_runs = []
    for repeat in range(args.repeats):
        stress_runs.append(evaluate_once(model, meta, eval_args, "random-stress", args.seed + repeat * 10000, output_prefix=None))
    stress_acc = [row["accuracy"] for row in stress_runs]
    stress_f1 = [row["macro_f1"] for row in stress_runs]

    cpu_time = benchmark_single_image(checkpoint, meta["model_name"], meta["num_classes"], meta["img_size"], "cpu")
    gpu_time = benchmark_single_image(checkpoint, meta["model_name"], meta["num_classes"], meta["img_size"], "cuda")
    row: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "model_name": meta["model_name"],
        "img_size": meta["img_size"],
        "num_classes": meta["num_classes"],
        "parameters": params,
        "checkpoint_size_mb": checkpoint.stat().st_size / (1024 * 1024),
        "clean_accuracy": clean["accuracy"],
        "clean_macro_f1": clean["macro_f1"],
        "per_corruption": {
            name: {
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
            }
            for name, result in per_corruption.items()
        },
        "strong_random_stress_accuracy_mean": float(sum(stress_acc) / len(stress_acc)) if stress_acc else 0.0,
        "strong_random_stress_accuracy_std": float(torch.tensor(stress_acc).std(unbiased=False).item()) if stress_acc else 0.0,
        "strong_random_stress_macro_f1_mean": float(sum(stress_f1) / len(stress_f1)) if stress_f1 else 0.0,
        "strong_random_stress_macro_f1_std": float(torch.tensor(stress_f1).std(unbiased=False).item()) if stress_f1 else 0.0,
        "cpu_single_ms": None if cpu_time is None else cpu_time * 1000,
        "gpu_single_ms": None if gpu_time is None else gpu_time * 1000,
    }
    row["recommended_use"] = recommend({**row, "cpu_single_ms": row["cpu_single_ms"] or 9999})
    return row


def print_table(rows: list[dict[str, Any]]) -> None:
    header = [
        "checkpoint",
        "model",
        "size",
        "clean_acc",
        "clean_f1",
        "stress_f1",
        "cpu_ms",
        "gpu_ms",
        "recommend",
    ]
    print("\t".join(header))
    for row in rows:
        print(
            "\t".join(
                [
                    row["checkpoint"],
                    row["model_name"],
                    str(row["img_size"]),
                    f"{row['clean_accuracy']:.6f}",
                    f"{row['clean_macro_f1']:.6f}",
                    f"{row['strong_random_stress_macro_f1_mean']:.6f}",
                    "NA" if row["cpu_single_ms"] is None else f"{row['cpu_single_ms']:.3f}",
                    "NA" if row["gpu_single_ms"] is None else f"{row['gpu_single_ms']:.3f}",
                    row["recommended_use"],
                ]
            )
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare traffic sign checkpoints")
    parser.add_argument("--data-dir", default="data/gtsrb")
    parser.add_argument("--csv", default="Test.csv")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/model_comparison.json")
    parser.add_argument("--path-col", default="Path")
    parser.add_argument("--label-col", default="ClassId")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument("--corruptions", nargs="+", default=CORRUPTION_NAMES)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    rows = [compare_checkpoint(path, args) for path in args.checkpoints]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"models": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print_table(rows)
    print("Saved:", output)


if __name__ == "__main__":
    main()
