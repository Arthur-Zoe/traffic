from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from augmentations import CORRUPTION_NAMES
from evaluate import EvaluationDataset, checkpoint_metadata, load_model_from_checkpoint
from metrics import compute_metrics
from train_gtsrb import stratified_split


def fuse_probabilities(probabilities: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    if not probabilities:
        raise ValueError("At least one probability array is required.")
    if len(probabilities) != len(weights):
        raise ValueError("Probability arrays and weights must have the same length.")
    reference = probabilities[0].shape
    if any(array.shape != reference for array in probabilities):
        raise ValueError("All probability arrays must have identical shape.")
    weight_array = np.asarray(weights, dtype=np.float64)
    if np.any(weight_array < 0) or weight_array.sum() <= 0:
        raise ValueError("Fusion weights must be non-negative and sum to a positive value.")
    fused = sum(weight * array for weight, array in zip(weight_array / weight_array.sum(), probabilities))
    return fused / np.maximum(fused.sum(axis=1, keepdims=True), 1e-12)


def probability_metrics(y_true: np.ndarray, probabilities: np.ndarray, num_classes: int) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    return compute_metrics(y_true.tolist(), predictions.tolist(), labels=list(range(num_classes)))


@torch.no_grad()
def collect_probabilities(
    checkpoint: str | Path,
    data_dir: str | Path,
    csv_path: str | Path,
    path_col: str,
    label_col: str,
    mode: str,
    severity: str,
    seed: int,
    device: torch.device,
    batch_size: int,
    workers: int,
    indices: list[int] | None = None,
    corruption: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    model, meta = load_model_from_checkpoint(checkpoint, device)
    dataset = EvaluationDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        path_col=path_col,
        label_col=label_col,
        img_size=meta["img_size"],
        mean=meta["mean"],
        std=meta["std"],
        mode=mode,
        severity=severity,
        corruption=corruption,
        seed=seed,
        class_to_idx=meta["class_to_idx"],
    )
    loader = DataLoader(
        Subset(dataset, indices) if indices is not None else dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    labels: list[int] = []
    paths: list[str] = []
    probabilities: list[np.ndarray] = []
    for images, batch_labels, batch_paths in loader:
        logits = model(images.to(device, non_blocking=True))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.extend(batch_labels.numpy().astype(int).tolist())
        paths.extend(list(batch_paths))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(labels, dtype=np.int64), np.concatenate(probabilities), paths, meta


def aligned_probabilities(
    checkpoints: list[str],
    args: argparse.Namespace,
    csv_path: str,
    mode: str,
    seed: int,
    indices: list[int] | None = None,
    corruption: str | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[str], dict[str, Any]]:
    device = torch.device(args.device)
    labels_ref: np.ndarray | None = None
    paths_ref: list[str] | None = None
    probabilities: list[np.ndarray] = []
    meta_ref: dict[str, Any] | None = None
    for checkpoint in checkpoints:
        labels, probs, paths, meta = collect_probabilities(
            checkpoint, args.data_dir, csv_path, args.path_col, args.label_col, mode,
            args.severity, seed, device, args.batch_size, args.workers, indices, corruption,
        )
        if labels_ref is None:
            labels_ref, paths_ref, meta_ref = labels, paths, meta
        elif not np.array_equal(labels_ref, labels) or paths_ref != paths:
            raise RuntimeError("Models produced different sample order or labels; cannot safely fuse probabilities.")
        if int(meta["num_classes"]) != int(meta_ref["num_classes"]):
            raise RuntimeError("Checkpoints have different class counts; cannot fuse probabilities.")
        probabilities.append(probs)
    assert labels_ref is not None and paths_ref is not None and meta_ref is not None
    return labels_ref, probabilities, paths_ref, meta_ref


def val_indices(args: argparse.Namespace, checkpoint: str) -> list[int]:
    meta = checkpoint_metadata(checkpoint)
    csv_path = Path(args.selection_csv)
    if not csv_path.is_absolute():
        csv_path = Path(args.data_dir) / csv_path
    frame = pd.read_csv(csv_path)
    raw = frame[args.label_col].astype(str).tolist()
    labels = [meta["class_to_idx"][label] for label in raw]
    _, validation = stratified_split(labels, val_ratio=args.val_ratio, seed=args.seed)
    return validation


def candidate_weights() -> dict[str, list[float]]:
    candidates = {
        "v2_convnext_50_50": [0.5, 0.5, 0.0],
        "v2_convnext_60_40": [0.6, 0.4, 0.0],
        "v2_convnext_70_30": [0.7, 0.3, 0.0],
        "v2_b2_50_50": [0.5, 0.0, 0.5],
        "v2_b2_60_40": [0.6, 0.0, 0.4],
    }
    for weights in itertools.product([0.2, 0.3, 0.4, 0.5, 0.6], repeat=3):
        if abs(sum(weights) - 1.0) < 1e-9:
            candidates[f"three_{weights[0]:.1f}_{weights[1]:.1f}_{weights[2]:.1f}"] = list(weights)
    return candidates


def choose_weights(y_true: np.ndarray, probabilities: list[np.ndarray], num_classes: int) -> tuple[dict[str, list[float]], dict[str, Any]]:
    candidates = candidate_weights()
    scored: dict[str, Any] = {}
    for name, weights in candidates.items():
        active = [(prob, weight) for prob, weight in zip(probabilities, weights) if weight > 0]
        fused = fuse_probabilities([item[0] for item in active], [item[1] for item in active])
        scored[name] = {"weights": weights, **probability_metrics(y_true, fused, num_classes)}
    selected = {
        "v2_convnext_selected": max(
            (item for item in scored.items() if item[0].startswith("v2_convnext")), key=lambda item: item[1]["macro_f1"]
        )[1]["weights"],
        "v2_b2_selected": max(
            (item for item in scored.items() if item[0].startswith("v2_b2")), key=lambda item: item[1]["macro_f1"]
        )[1]["weights"],
        "three_selected": max(
            (item for item in scored.items() if item[0].startswith("three_")), key=lambda item: item[1]["macro_f1"]
        )[1]["weights"],
    }
    return selected, scored


def complementarity(y_true: np.ndarray, predictions: list[np.ndarray], names: list[str], num_classes: int) -> dict[str, Any]:
    correct = [set(np.flatnonzero(pred == y_true).tolist()) for pred in predictions]
    errors = [set(range(len(y_true))) - item for item in correct]
    all_correct = set.intersection(*correct)
    all_wrong = set.intersection(*errors)
    unique_correct = {name: len(item - set.union(*(correct[:idx] + correct[idx + 1 :]))) for idx, (name, item) in enumerate(zip(names, correct))}
    unique_wrong = {name: len(item - set.union(*(errors[:idx] + errors[idx + 1 :]))) for idx, (name, item) in enumerate(zip(names, errors))}
    disagreement = sum(len({int(pred[idx]) for pred in predictions}) > 1 for idx in range(len(y_true)))
    oracle = np.asarray([true if any(pred[idx] == true for pred in predictions) else predictions[0][idx] for idx, true in enumerate(y_true)])
    per_class: dict[str, Any] = {}
    for cls in range(num_classes):
        class_errors = [set(np.flatnonzero((y_true == cls) & (pred != cls)).tolist()) for pred in predictions]
        union = set.union(*class_errors)
        intersection = set.intersection(*class_errors)
        per_class[str(cls)] = {
            "all_error_overlap": len(intersection) / len(union) if union else 0.0,
            "error_union": len(union),
            "error_intersection": len(intersection),
        }
    return {
        "all_correct": len(all_correct),
        "all_wrong": len(all_wrong),
        "unique_correct": unique_correct,
        "unique_wrong": unique_wrong,
        "prediction_disagreements": disagreement,
        "oracle": probability_metrics(y_true, np.eye(num_classes)[oracle], num_classes),
        "per_class_error_overlap": per_class,
    }


def write_disagreements(output: Path, paths: list[str], y_true: np.ndarray, predictions: list[np.ndarray], names: list[str], mode: str) -> None:
    exists = output.exists()
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(["mode", "path", "true", *names])
        for idx, path in enumerate(paths):
            values = [int(pred[idx]) for pred in predictions]
            if len(set(values)) > 1:
                writer.writerow([mode, path, int(y_true[idx]), *values])


def evaluate_ensembles(y_true: np.ndarray, probabilities: list[np.ndarray], weights: dict[str, list[float]], num_classes: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, weight in weights.items():
        active = [(prob, value) for prob, value in zip(probabilities, weight) if value > 0]
        fused = fuse_probabilities([item[0] for item in active], [item[1] for item in active])
        result[name] = {"weights": weight, **probability_metrics(y_true, fused, num_classes)}
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse architecture complementarity and probability ensembles")
    parser.add_argument("--data-dir", default="data/gtsrb")
    parser.add_argument("--selection-csv", default="Train.csv")
    parser.add_argument("--report-csv", default="Test.csv")
    parser.add_argument("--path-col", default="Path")
    parser.add_argument("--label-col", default="ClassId")
    parser.add_argument("--checkpoints", nargs=3, required=True)
    parser.add_argument("--names", nargs=3, default=["v2", "convnext_tiny", "efficientnet_b2"])
    parser.add_argument("--severity", choices=["light", "medium", "strong"], default="strong")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/architecture_complementarity_v2_convnext_b2.json")
    parser.add_argument("--disagreements-csv", default="outputs/architecture_disagreements_v2_convnext_b2.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection_indices = val_indices(args, args.checkpoints[0])
    y_val, probs_val, _, meta = aligned_probabilities(args.checkpoints, args, args.selection_csv, "clean", args.seed, selection_indices)
    selected, selection_scores = choose_weights(y_val, probs_val, int(meta["num_classes"]))
    fixed = {name: weights for name, weights in candidate_weights().items() if not name.startswith("three_")}
    report_weights = {**fixed, **selected}
    output: dict[str, Any] = {
        "checkpoints": args.checkpoints,
        "names": args.names,
        "selection": {"split": "stratified_train_validation", "samples": len(selection_indices), "scores": selection_scores, "selected": selected},
        "report": {},
    }
    disagreement_path = Path(args.disagreements_csv)
    disagreement_path.parent.mkdir(parents=True, exist_ok=True)
    if disagreement_path.exists():
        disagreement_path.unlink()
    for mode, seed, corruption in [("clean", args.seed, None), ("random_stress", args.seed, None), *[(f"per_{name}", args.seed, name) for name in CORRUPTION_NAMES]]:
        actual_mode = "random-stress" if mode == "random_stress" else ("per-corruption" if corruption else "clean")
        y_true, probabilities, paths, meta = aligned_probabilities(args.checkpoints, args, args.report_csv, actual_mode, seed, None, corruption)
        predictions = [prob.argmax(axis=1) for prob in probabilities]
        output["report"][mode] = {
            "single_models": {name: probability_metrics(y_true, prob, int(meta["num_classes"])) for name, prob in zip(args.names, probabilities)},
            "ensembles": evaluate_ensembles(y_true, probabilities, report_weights, int(meta["num_classes"])),
            "complementarity": complementarity(y_true, predictions, args.names, int(meta["num_classes"])),
        }
        if mode in {"clean", "random_stress"}:
            write_disagreements(disagreement_path, paths, y_true, predictions, args.names, mode)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Selection samples:", len(selection_indices))
    print("Selected weights:", selected)
    print("Saved:", output_path)
    print("Disagreements:", disagreement_path)


if __name__ == "__main__":
    main()
