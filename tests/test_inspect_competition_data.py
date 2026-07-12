from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from tools.inspect_competition_data import inspect_competition_data


def make_image(path: Path, mode: str = "RGB", size: tuple[int, int] = (32, 24), color=10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color).save(path)


def write_csv(root: Path, text: str, name: str = "Train.csv") -> Path:
    path = root / name
    path.write_text(text, encoding="utf-8")
    return path


def test_images_missing_corrupt_modes_duplicates_and_report_json(tmp_path: Path) -> None:
    make_image(tmp_path / "rgb.png", "RGB")
    make_image(tmp_path / "gray.png", "L")
    (tmp_path / "copy.png").write_bytes((tmp_path / "rgb.png").read_bytes())
    (tmp_path / "bad.png").write_bytes(b"not an image")
    write_csv(
        tmp_path,
        "Path,ClassId\n"
        "rgb.png,0\ncopy.png,0\ngray.png,1\nmissing.png,1\nbad.png,1\n",
    )
    report = inspect_competition_data(tmp_path)
    assert report["path_column"] == "Path"
    assert report["label_column"] == "ClassId"
    assert report["image_count"] == 4
    assert report["readable_images"] == 3
    assert report["missing_images"] == 1
    assert report["corrupt_images"] == 1
    assert report["channel_modes"] == {"L": 1, "RGB": 2}
    assert report["duplicate_file_groups"] == 1
    assert report["duplicate_label_conflict_groups"] == 0
    output = tmp_path / "reports" / "report.json"
    completed = subprocess.run(
        [sys.executable, "tools/inspect_competition_data.py", "--data-dir", str(tmp_path), "--output", str(output)],
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0
    assert "可读取图片:" in completed.stdout
    assert json.loads(output.read_text(encoding="utf-8"))["missing_images"] == 1


def test_duplicate_label_conflict_and_explicit_columns(tmp_path: Path) -> None:
    make_image(tmp_path / "same.png")
    (tmp_path / "same-copy.png").write_bytes((tmp_path / "same.png").read_bytes())
    write_csv(tmp_path, "file_name,kind\nsame.png,stop\nsame-copy.png,yield\n")
    report = inspect_competition_data(tmp_path, path_column="file_name", label_column="kind")
    assert report["path_column"] == "file_name"
    assert report["label_column"] == "kind"
    assert report["duplicate_label_conflict_groups"] == 1
    assert report["duplicate_file_groups"] == 0
    assert report["classes_are_contiguous"] is False


def test_csv_relative_path_resolution_and_auto_csv_fallback(tmp_path: Path) -> None:
    csv_dir = tmp_path / "metadata"
    make_image(csv_dir / "images" / "a.png")
    csv_dir.mkdir(exist_ok=True)
    (csv_dir / "labels.csv").write_text("image,label\nimages/a.png,0\n", encoding="utf-8")
    report = inspect_competition_data(tmp_path, csv_file="metadata/labels.csv")
    assert report["readable_images"] == 1

    # With no Train.csv/train.csv, the first root CSV is selected automatically.
    make_image(tmp_path / "b.png")
    write_csv(tmp_path, "image,label\nb.png,0\n", name="anything.csv")
    automatic = inspect_competition_data(tmp_path)
    assert automatic["csv_path"].endswith("anything.csv")


def test_gtsrb_compatibility_and_non_contiguous_classes(tmp_path: Path) -> None:
    rows = ["image,label"]
    for label in range(43):
        image = tmp_path / f"{label}.png"
        make_image(image)
        rows.append(f"{image.name},{label}")
    write_csv(tmp_path, "\n".join(rows) + "\n")
    report = inspect_competition_data(tmp_path, check_duplicates=False)
    assert report["num_classes"] == 43
    assert report["gtsrb_43_class_compatible"] is True
    assert report["classes_are_contiguous"] is True

    write_csv(tmp_path, "image,label\n0.png,0\n2.png,2\n")
    sparse = inspect_competition_data(tmp_path, check_duplicates=False)
    assert sparse["classes_are_contiguous"] is False
    assert sparse["recommended_num_classes"] == 2
