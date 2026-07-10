from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from augmentations import CORRUPTION_NAMES, RandomBadWeather, apply_corruption
from datasets import CSVImageDataset, build_class_mapping, resolve_image_path
from inference import TrafficSignClassifier, to_pil_image
from metrics import compute_macro_f1
from models import build_model, load_state_flexible


def make_image(path: Path, color: tuple[int, int, int] = (120, 30, 20)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), color).save(path)


def test_path_resolution_and_csv_dataset(tmp_path: Path) -> None:
    make_image(tmp_path / "Train" / "0" / "a.png")
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("Path,ClassId\nTrain/0/a.png,0\n", encoding="utf-8")
    assert resolve_image_path(tmp_path, "train/0/a.png").exists()
    ds = CSVImageDataset(csv_path, tmp_path, path_col="Path", label_col="ClassId")
    image, label = ds[0]
    assert image.mode == "RGB"
    assert label == 0


def test_class_mapping_string_and_numeric() -> None:
    class_to_idx, idx_to_class = build_class_mapping(["stop", "yield", "stop"], numeric_identity=True)
    assert class_to_idx == {"stop": 0, "yield": 1}
    assert idx_to_class[0] == "stop"
    numeric, _ = build_class_mapping(["0", "42", "3"], numeric_identity=True)
    assert numeric["42"] == 42


def test_all_augmentations_small_rgb_and_seed() -> None:
    img = Image.new("RGB", (8, 8), (100, 120, 140))
    for name in CORRUPTION_NAMES:
        out = apply_corruption(img, name, severity="strong", seed=123)
        assert isinstance(out, Image.Image)
        assert out.mode == "RGB"
        assert out.size == img.size
    aug1 = RandomBadWeather(p=1.0, severity="medium", max_ops=2, seed=7)
    aug2 = RandomBadWeather(p=1.0, severity="medium", max_ops=2, seed=7)
    assert np.array_equal(np.asarray(aug1(img)), np.asarray(aug2(img)))


def test_macro_f1_ignores_empty_support() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    value = compute_macro_f1(y_true, y_pred, num_classes=3, ignore_empty=True)
    assert value == pytest.approx((2 / 3 + 0.8) / 2)


def test_flexible_checkpoint_skips_classifier_mismatch() -> None:
    old = build_model("resnet18", 3, pretrained=False)
    new = build_model("resnet18", 4, pretrained=False)
    info = load_state_flexible(new, old.state_dict())
    assert info["skipped"] >= 2
    assert info["classifier_skipped"]


def test_inference_inputs_and_bad_type(tmp_path: Path) -> None:
    path = tmp_path / "img.png"
    make_image(path)
    pil = to_pil_image(path)
    assert pil.mode == "RGB"
    assert to_pil_image(np.zeros((10, 10), dtype=np.uint8)).mode == "RGB"
    with pytest.raises(TypeError):
        to_pil_image(123)  # type: ignore[arg-type]


def test_inference_predict_pil_numpy_path(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "mini.pt"
    model = build_model("resnet18", 2, pretrained=False)
    torch.save(
        {
            "model": model.state_dict(),
            "model_name": "resnet18",
            "num_classes": 2,
            "img_size": 32,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "idx_to_class": {"0": "a", "1": "b"},
        },
        ckpt_path,
    )
    img_path = tmp_path / "input.png"
    make_image(img_path)
    clf = TrafficSignClassifier(ckpt_path, device="cpu", topk=2)
    assert "index" in clf.predict_one(Image.open(img_path))
    assert "index" in clf.predict_one(np.zeros((32, 32, 3), dtype=np.uint8))
    assert "index" in clf.predict_one(img_path)
    with pytest.raises(TypeError):
        clf.predict({"bad": "input"})
