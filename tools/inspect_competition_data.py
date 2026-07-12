#!/usr/bin/env python3
"""Inspect a CSV-described competition image dataset before training.

The script deliberately reports bad samples instead of failing on them, so it is
safe to run on data received shortly before a competition deadline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError


PATH_CANDIDATES = ("Path", "path", "image", "image_path", "filename", "file")
LABEL_CANDIDATES = ("ClassId", "class_id", "label", "target", "class")


def _limited(values: Iterable[Any], maximum: int) -> list[Any]:
    return list(values)[:maximum]


def _is_integer(value: str) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return str(int(value)) == str(value).strip() or str(value).strip().lstrip("+") == str(int(value))


def _sorted_labels(labels: Iterable[str]) -> list[str]:
    labels = list(labels)
    if all(_is_integer(label) for label in labels):
        return sorted(labels, key=lambda value: int(value))
    return sorted(labels)


def _json_label(label: str) -> int | str:
    return int(label) if _is_integer(label) else label


def _resolve_path(data_dir: Path, csv_path: Path, value: str) -> Path:
    """Resolve relative paths against both documented supported roots."""
    path = Path(str(value).strip())
    if path.is_absolute():
        return path
    from_data_dir = data_dir / path
    if from_data_dir.exists():
        return from_data_dir
    from_csv_dir = csv_path.parent / path
    if from_csv_dir.exists():
        return from_csv_dir
    # Prefer data_dir in the report for missing paths, consistently.
    return from_data_dir


def _hash_file(item: tuple[Path, str]) -> tuple[str, Path, str]:
    path, label = item
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), path, label


def _find_csv(data_dir: Path, csv_value: str | None) -> Path | None:
    if csv_value:
        supplied = Path(csv_value)
        return supplied if supplied.is_absolute() else data_dir / supplied
    for name in ("Train.csv", "train.csv"):
        candidate = data_dir / name
        if candidate.is_file():
            return candidate
    return next((path for path in sorted(data_dir.glob("*.csv")) if path.is_file()), None)


def _empty_report(data_dir: Path, csv_path: Path | None) -> dict[str, Any]:
    return {
        "data_dir": str(data_dir), "csv_path": str(csv_path) if csv_path else "",
        "csv_columns": [], "path_column": "", "label_column": "", "row_count": 0,
        "image_count": 0, "readable_images": 0, "missing_images": 0,
        "missing_examples": [], "corrupt_images": 0, "corrupt_examples": [],
        "num_classes": 0, "class_counts": {}, "class_values": [],
        "classes_are_contiguous": False, "image_size_distribution": {}, "channel_modes": {},
        "duplicate_file_groups": 0, "duplicate_files": [],
        "duplicate_label_conflict_groups": 0, "duplicate_label_conflicts": [],
        "class_to_idx": {}, "idx_to_class": {}, "gtsrb_43_class_compatible": False,
        "recommended_num_classes": 0, "recommended_batch_size": 0,
        "recommended_img_size": 0, "warnings": [], "errors": [],
    }


def inspect_competition_data(
    data_dir: str | Path,
    csv_file: str | None = None,
    path_column: str | None = None,
    label_column: str | None = None,
    max_examples: int = 10,
    hash_workers: int = 4,
    check_duplicates: bool = True,
) -> dict[str, Any]:
    """Return a JSON-serialisable readiness report for a competition dataset."""
    data_dir = Path(data_dir).expanduser()
    csv_path = _find_csv(data_dir, csv_file) if data_dir.is_dir() else None
    report = _empty_report(data_dir, csv_path)
    if max_examples < 1:
        report["errors"].append("--max-examples must be at least 1.")
        return report
    if not data_dir.is_dir():
        report["errors"].append(f"Dataset root does not exist or is not a directory: {data_dir}")
        return report
    if csv_path is None or not csv_path.is_file():
        requested = csv_file or "Train.csv, train.csv, or another CSV in the dataset root"
        report["errors"].append(f"CSV file was not found: {requested}")
        return report

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        report["errors"].append(f"Could not read CSV {csv_path}: {exc}")
        return report

    report["csv_columns"] = columns
    report["row_count"] = len(rows)
    selected_path = path_column or next((name for name in PATH_CANDIDATES if name in columns), None)
    selected_label = label_column or next((name for name in LABEL_CANDIDATES if name in columns), None)
    report["path_column"] = selected_path or ""
    report["label_column"] = selected_label or ""
    if selected_path not in columns:
        source = "specified" if path_column else "automatically recognised"
        report["errors"].append(f"Image path column was not {source}. Available columns: {columns}")
    if selected_label not in columns:
        source = "specified" if label_column else "automatically recognised"
        report["errors"].append(f"Label column was not {source}. Available columns: {columns}")
    if report["errors"]:
        return report

    labels = [str(row.get(selected_label, "")).strip() for row in rows]
    if any(not label for label in labels):
        report["warnings"].append("One or more rows have an empty label; empty string is treated as a class.")
    ordered_labels = _sorted_labels(set(labels))
    counts = Counter(labels)
    report["num_classes"] = len(ordered_labels)
    report["class_counts"] = {label: counts[label] for label in ordered_labels}
    report["class_values"] = [_json_label(label) for label in ordered_labels]
    report["class_to_idx"] = {label: index for index, label in enumerate(ordered_labels)}
    report["idx_to_class"] = {str(index): label for index, label in enumerate(ordered_labels)}
    labels_are_numeric = bool(ordered_labels) and all(_is_integer(label) for label in ordered_labels)
    numeric_values = [int(label) for label in ordered_labels] if labels_are_numeric else []
    report["classes_are_contiguous"] = labels_are_numeric and numeric_values == list(range(len(numeric_values)))
    report["gtsrb_43_class_compatible"] = labels_are_numeric and numeric_values == list(range(43))
    report["recommended_num_classes"] = report["num_classes"]
    if not report["classes_are_contiguous"]:
        report["warnings"].append("Class values are not continuous 0..N-1; use the emitted class mapping during training.")
    if not report["gtsrb_43_class_compatible"]:
        report["warnings"].append("Dataset is not directly compatible with GTSRB's 43 classes (values 0..42).")
    if report["row_count"] < 100:
        report["warnings"].append("Dataset has fewer than 100 CSV rows; validation results may be unstable.")

    readable_for_hash: list[tuple[Path, str]] = []
    sizes: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    short_sides: list[int] = []
    for row, label in zip(rows, labels):
        raw_path = str(row.get(selected_path, "")).strip()
        image_path = _resolve_path(data_dir, csv_path, raw_path)
        if not raw_path or not image_path.is_file():
            report["missing_images"] += 1
            if len(report["missing_examples"]) < max_examples:
                report["missing_examples"].append(raw_path or "<empty path>")
            continue
        report["image_count"] += 1
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                width, height = image.size
                mode = image.mode
            report["readable_images"] += 1
            sizes[f"{width}x{height}"] += 1
            modes[mode] += 1
            short_sides.append(min(width, height))
            readable_for_hash.append((image_path, label))
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            report["corrupt_images"] += 1
            if len(report["corrupt_examples"]) < max_examples:
                report["corrupt_examples"].append({"path": str(image_path), "error": str(exc)})

    report["image_size_distribution"] = dict(sorted(sizes.items()))
    report["channel_modes"] = dict(sorted(modes.items()))
    if report["missing_images"]:
        report["warnings"].append(f"{report['missing_images']} image file(s) are missing.")
    if report["corrupt_images"]:
        report["warnings"].append(f"{report['corrupt_images']} image file(s) are corrupt or unreadable.")

    if short_sides:
        below_or_equal_160 = sum(side <= 160 for side in short_sides)
        report["recommended_img_size"] = 256 if below_or_equal_160 * 2 < len(short_sides) else 224
        report["recommended_batch_size"] = 8 if report["recommended_img_size"] == 256 else 12
    else:
        report["warnings"].append("No readable images were found; no input size or batch size can be recommended.")

    if check_duplicates and readable_for_hash:
        groups: dict[str, list[tuple[Path, str]]] = defaultdict(list)
        try:
            with ThreadPoolExecutor(max_workers=max(1, hash_workers)) as executor:
                for digest, path, label in executor.map(_hash_file, readable_for_hash):
                    groups[digest].append((path, label))
        except OSError as exc:
            report["warnings"].append(f"Could not complete duplicate hash check: {exc}")
        else:
            for digest, group in groups.items():
                if len(group) < 2:
                    continue
                paths = [str(path) for path, _ in group]
                group_labels = _sorted_labels({label for _, label in group})
                record = {"sha256": digest, "paths": _limited(paths, max_examples), "count": len(group)}
                if len(group_labels) == 1:
                    record["label"] = _json_label(group_labels[0])
                    report["duplicate_file_groups"] += 1
                    if len(report["duplicate_files"]) < max_examples:
                        report["duplicate_files"].append(record)
                else:
                    record["labels"] = [_json_label(label) for label in group_labels]
                    report["duplicate_label_conflict_groups"] += 1
                    if len(report["duplicate_label_conflicts"]) < max_examples:
                        report["duplicate_label_conflicts"].append(record)
    elif not check_duplicates:
        report["warnings"].append("Duplicate-image check was disabled.")

    return report


def _print_summary(report: dict[str, Any], output: Path) -> None:
    print(f"数据集路径: {report['data_dir']}")
    print(f"CSV: {report['csv_path']}")
    print(f"样本数: {report['row_count']}")
    print(f"可读取图片: {report['readable_images']}/{report['image_count']}")
    print(f"缺失图片: {report['missing_images']}")
    print(f"损坏图片: {report['corrupt_images']}")
    print(f"类别数量: {report['num_classes']}")
    print(f"类别范围: {report['class_values']}")
    print(f"重复组: {report['duplicate_file_groups']}")
    print(f"标签冲突组: {report['duplicate_label_conflict_groups']}")
    print(f"推荐 num_classes: {report['recommended_num_classes']}")
    print(f"推荐 img_size: {report['recommended_img_size']}")
    print(f"推荐 batch_size: {report['recommended_batch_size']}")
    print(f"GTSRB 43 类兼容: {report['gtsrb_43_class_compatible']}")
    for error in report["errors"]:
        print(f"[ERROR] {error}")
    for warning in report["warnings"]:
        print(f"[WARNING] {warning}")
    print(f"报告文件: {output}")


def _parse_bool(value: str) -> bool:
    if value.lower() in {"true", "1", "yes", "on"}:
        return True
    if value.lower() in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Dataset root directory.")
    parser.add_argument("--csv", default=None, help="CSV path relative to data-dir, or an absolute path.")
    parser.add_argument("--output", default="official_dataset_report.json", help="JSON report path.")
    parser.add_argument("--path-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--hash-workers", type=int, default=4)
    parser.add_argument("--check-duplicates", type=_parse_bool, default=True)
    args = parser.parse_args()
    report = inspect_competition_data(
        data_dir=args.data_dir,
        csv_file=args.csv,
        path_column=args.path_column,
        label_column=args.label_column,
        max_examples=args.max_examples,
        hash_workers=args.hash_workers,
        check_duplicates=args.check_duplicates,
    )
    output = Path(args.output).expanduser()
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"[ERROR] Could not write report {output}: {exc}")
        return 2
    _print_summary(report, output)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
