from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np


def confusion_matrix_np(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        true_i = int(true)
        pred_i = int(pred)
        if 0 <= true_i < num_classes and 0 <= pred_i < num_classes:
            matrix[true_i, pred_i] += 1
    return matrix


def per_class_prf(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> list[dict[str, float | int]]:
    cm = confusion_matrix_np(y_true, y_pred, num_classes)
    rows: list[dict[str, float | int]] = []
    for cls in range(num_classes):
        tp = float(cm[cls, cls])
        fp = float(cm[:, cls].sum() - cm[cls, cls])
        fn = float(cm[cls, :].sum() - cm[cls, cls])
        support = int(cm[cls, :].sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        rows.append(
            {
                "class_id": cls,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support,
            }
        )
    return rows


def compute_macro_f1(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int, ignore_empty: bool = True) -> float:
    rows = per_class_prf(y_true, y_pred, num_classes)
    values = [float(row["f1"]) for row in rows if (not ignore_empty or int(row["support"]) > 0)]
    return float(np.mean(values)) if values else 0.0


def compute_metrics(y_true: Iterable[int], y_pred: Iterable[int], labels: list[int] | None = None) -> dict[str, float]:
    y_true_list = [int(x) for x in y_true]
    y_pred_list = [int(x) for x in y_pred]
    if labels is None:
        num_classes = max(y_true_list + y_pred_list, default=-1) + 1
    else:
        num_classes = max(labels) + 1 if labels else 0
    correct = sum(1 for true, pred in zip(y_true_list, y_pred_list) if true == pred)
    total = len(y_true_list)
    return {
        "accuracy": float(correct / total) if total else 0.0,
        "macro_f1": compute_macro_f1(y_true_list, y_pred_list, num_classes, ignore_empty=True),
    }


def save_confusion_matrix_csv(matrix: np.ndarray, output: str | Path, classes: list[str] | None = None) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    names = classes if classes is not None else [str(i) for i in range(matrix.shape[0])]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *names])
        for name, row in zip(names, matrix.tolist()):
            writer.writerow([name, *row])


def save_report(y_true: Iterable[int], y_pred: Iterable[int], classes: list[str], out_txt: str | Path, out_csv: str | Path | None = None) -> None:
    y_true_list = [int(x) for x in y_true]
    y_pred_list = [int(x) for x in y_pred]
    rows = per_class_prf(y_true_list, y_pred_list, len(classes))
    metrics = compute_metrics(y_true_list, y_pred_list, labels=list(range(len(classes))))
    lines = [
        f"accuracy: {metrics['accuracy']:.6f}",
        f"macro_f1: {metrics['macro_f1']:.6f}",
        "",
        "class precision recall f1 support",
    ]
    for row, name in zip(rows, classes):
        lines.append(f"{name} {row['precision']:.6f} {row['recall']:.6f} {row['f1']:.6f} {row['support']}")
    out_txt = Path(out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if out_csv is not None:
        save_confusion_matrix_csv(confusion_matrix_np(y_true_list, y_pred_list, len(classes)), out_csv, classes)
