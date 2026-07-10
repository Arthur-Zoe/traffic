from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm")


def list_images(root: str | Path) -> list[str]:
    root = Path(root)
    return [str(path) for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in IMG_EXTS]


def resolve_image_path(data_dir: str | Path, value: str | Path) -> Path:
    data_dir = Path(data_dir)
    value_path = Path(str(value))
    if value_path.is_absolute():
        return value_path
    direct = data_dir / value_path
    if direct.exists():
        return direct
    parts = value_path.parts
    if parts:
        first, rest = parts[0], parts[1:]
        for alt in (first.lower(), first.upper(), first.capitalize()):
            candidate = data_dir / Path(alt, *rest)
            if candidate.exists():
                return candidate
    return direct


def build_class_mapping(labels: Iterable[Any], numeric_identity: bool = True) -> tuple[dict[str, int], dict[int, str]]:
    label_strings = [str(label) for label in labels]
    unique = sorted(set(label_strings), key=lambda x: (not x.lstrip("-").isdigit(), int(x) if x.lstrip("-").isdigit() else x))
    if numeric_identity and all(x.isdigit() for x in unique):
        ids = [int(x) for x in unique]
        if ids and min(ids) >= 0 and len(set(ids)) == len(ids):
            class_to_idx = {str(i): i for i in ids}
            idx_to_class = {i: str(i) for i in ids}
            return class_to_idx, idx_to_class
    class_to_idx = {name: idx for idx, name in enumerate(unique)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


def export_class_mapping(class_to_idx: dict[str, int], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    idx_to_class = {str(v): k for k, v in class_to_idx.items()}
    (output_dir / "class_to_idx.json").write_text(json.dumps(class_to_idx, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "idx_to_class.json").write_text(json.dumps(idx_to_class, ensure_ascii=False, indent=2), encoding="utf-8")


class CSVImageDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        data_dir: str | Path,
        path_col: str = "Path",
        label_col: str | None = "ClassId",
        transform: Any = None,
        class_to_idx: dict[str, int] | None = None,
        indices: list[int] | None = None,
        has_label: bool | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.data_dir = Path(data_dir)
        self.path_col = path_col
        self.label_col = label_col
        self.transform = transform
        self.df = pd.read_csv(self.csv_path)
        if path_col not in self.df.columns:
            raise ValueError(f"CSV 缺少路径列 {path_col}: {self.csv_path}")
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        self.paths = self.df[path_col].astype(str).tolist()
        label_exists = label_col is not None and label_col in self.df.columns
        self.has_label = label_exists if has_label is None else bool(has_label and label_exists)
        self.class_to_idx = class_to_idx
        self.idx_to_class: dict[int, str] | None = None
        self.targets: list[int] | None = None
        if self.has_label:
            labels = self.df[str(label_col)].astype(str).tolist()
            if class_to_idx is None:
                self.class_to_idx, self.idx_to_class = build_class_mapping(labels)
            else:
                self.idx_to_class = {idx: name for name, idx in class_to_idx.items()}
            missing = [label for label in labels if label not in self.class_to_idx]
            if missing:
                raise ValueError(f"CSV 中存在类别映射缺失的标签，例如: {missing[:5]}")
            self.targets = [self.class_to_idx[label] for label in labels]
        elif class_to_idx is not None:
            self.idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        rel_path = self.paths[idx]
        img_path = resolve_image_path(self.data_dir, rel_path)
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.has_label and self.targets is not None:
            return image, int(self.targets[idx])
        return image, rel_path


class FolderImageDataset(Dataset):
    def __init__(self, data_dir: str | Path, transform: Any = None, class_to_idx: dict[str, int] | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform
        classes = [path.name for path in sorted(self.data_dir.iterdir()) if path.is_dir()]
        self.class_to_idx = class_to_idx or {name: idx for idx, name in enumerate(classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.samples: list[tuple[Path, int]] = []
        for class_name, idx in self.class_to_idx.items():
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMG_EXTS:
                    self.samples.append((path, idx))
        self.targets = [idx for _, idx in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)


class UnlabeledImageDataset(Dataset):
    def __init__(self, data_dir: str | Path, transform: Any = None) -> None:
        self.data_dir = Path(data_dir)
        self.paths = [Path(path) for path in list_images(self.data_dir)]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, path.relative_to(self.data_dir).as_posix()


class GTSRBDataset(CSVImageDataset):
    def __init__(self, root: str | Path, csv_name: str, indices: list[int] | None = None, transform: Any = None):
        super().__init__(Path(root) / csv_name, root, path_col="Path", label_col="ClassId", transform=transform, indices=indices)


def build_train_dataset(
    data_dir: str | Path | None = None,
    csv_path: str | Path | None = None,
    image_dir: str | Path | None = None,
    image_col: str = "image",
    label_col: str = "label",
    transform: Any = None,
) -> Dataset:
    if csv_path is not None:
        root = image_dir if image_dir is not None else Path(csv_path).parent
        return CSVImageDataset(csv_path, root, path_col=image_col, label_col=label_col, transform=transform)
    if data_dir is None:
        raise ValueError("Either data_dir or csv_path must be provided.")
    return FolderImageDataset(data_dir, transform=transform)


def inspect_dataset(
    data_dir: str | Path,
    csv_path: str | Path | None = None,
    path_col: str = "Path",
    label_col: str | None = "ClassId",
    max_hash_bytes: int = 2_000_000,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    rows: list[tuple[str, str | None]] = []
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        if path_col not in df.columns:
            raise ValueError(f"CSV 缺少路径列 {path_col}: {csv_path}")
        labels = df[label_col].astype(str).tolist() if label_col and label_col in df.columns else [None] * len(df)
        rows = list(zip(df[path_col].astype(str).tolist(), labels))
    else:
        for path_str in list_images(data_dir):
            path = Path(path_str)
            label = path.parent.name if path.parent != data_dir else None
            rows.append((path.relative_to(data_dir).as_posix(), label))

    readable = 0
    corrupt: list[str] = []
    missing: list[str] = []
    sizes: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    hashes: defaultdict[str, list[tuple[str, str | None]]] = defaultdict(list)
    path_labels: defaultdict[str, set[str]] = defaultdict(set)

    for rel, label in rows:
        if label is not None:
            class_counts[str(label)] += 1
            path_labels[rel].add(str(label))
        path = resolve_image_path(data_dir, rel)
        if not path.exists():
            missing.append(rel)
            continue
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                sizes[f"{img.width}x{img.height}"] += 1
                modes[str(img.mode)] += 1
            readable += 1
            if path.stat().st_size <= max_hash_bytes:
                digest = hashlib.sha1(path.read_bytes()).hexdigest()
                hashes[digest].append((rel, str(label) if label is not None else None))
        except Exception:
            corrupt.append(rel)

    duplicate_files = [items for items in hashes.values() if len(items) > 1]
    label_conflicts = {path: sorted(labels) for path, labels in path_labels.items() if len(labels) > 1}
    class_to_idx, _ = build_class_mapping(class_counts.keys(), numeric_identity=True) if class_counts else ({}, {})
    return {
        "image_count": len(rows),
        "readable_images": readable,
        "corrupt_images": len(corrupt),
        "corrupt_examples": corrupt[:30],
        "missing_images": len(missing),
        "missing_examples": missing[:30],
        "num_classes": len(class_counts),
        "class_counts": dict(sorted(class_counts.items(), key=lambda x: (not x[0].isdigit(), int(x[0]) if x[0].isdigit() else x[0]))),
        "image_size_distribution": dict(sizes.most_common(30)),
        "channel_modes": dict(modes),
        "duplicate_files": duplicate_files[:30],
        "duplicate_file_groups": len(duplicate_files),
        "label_conflicts": label_conflicts,
        "recommended_num_classes": (max(class_to_idx.values()) + 1 if class_to_idx else len(class_counts)),
        "recommended_batch_size": 24 if readable and max((int(k.split('x')[0]) for k in sizes), default=224) >= 256 else 32,
        "recommended_img_size": 256 if sizes and np_median_size(list(sizes.elements())) >= 96 else 224,
    }


def np_median_size(size_strings: list[str]) -> float:
    if not size_strings:
        return 0.0
    values = []
    for item in size_strings:
        width, height = item.split("x")
        values.append(max(int(width), int(height)))
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2)
