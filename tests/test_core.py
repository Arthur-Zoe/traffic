from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from augmentations import CORRUPTION_NAMES, RandomBadWeather, apply_corruption
from ensemble_analysis import fuse_probabilities, probability_metrics
from datasets import (
    CSVImageDataset,
    FolderImageDataset,
    UnlabeledImageDataset,
    build_class_mapping,
    inspect_dataset,
    resolve_image_path,
)
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
    numeric, _ = build_class_mapping(["0", "1", "2"], numeric_identity=True)
    assert numeric == {"0": 0, "1": 1, "2": 2}


def test_one_based_and_sparse_numeric_labels_are_compact_and_reversible() -> None:
    one_based, one_based_inverse = build_class_mapping(["1", "2", "3"], numeric_identity=True)
    assert one_based == {"1": 0, "2": 1, "3": 2}
    assert one_based_inverse == {0: "1", 1: "2", 2: "3"}
    sparse, sparse_inverse = build_class_mapping(["10", "30", "20"], numeric_identity=True)
    assert sparse == {"10": 0, "20": 1, "30": 2}
    assert sparse_inverse[2] == "30"


def test_folder_unlabeled_and_dataset_inspection(tmp_path: Path) -> None:
    make_image(tmp_path / "images" / "10" / "a.png")
    make_image(tmp_path / "images" / "30" / "b.png", color=(20, 80, 140))
    folder = FolderImageDataset(tmp_path / "images")
    unlabeled = UnlabeledImageDataset(tmp_path / "images")
    assert len(folder) == 2
    assert len(unlabeled) == 2

    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "image,label\nimages/10/a.png,10\nimages/10/a.png,30\nimages/30/b.png,30\nmissing.png,10\n",
        encoding="utf-8",
    )
    report = inspect_dataset(tmp_path, csv_path, path_col="image", label_col="label")
    assert report["recommended_num_classes"] == 2
    assert report["class_to_idx"] == {"10": 0, "30": 1}
    assert report["missing_images"] == 1
    assert "images/10/a.png" in report["label_conflicts"]
    assert report["duplicate_label_conflict_groups"] == 1


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


@pytest.mark.parametrize("model_name", ["convnext_tiny", "efficientnet_b2"])
def test_new_backbones_build_and_skip_classifier(model_name: str) -> None:
    model = build_model(model_name, 43, pretrained=False)
    logits = model(torch.zeros(1, 3, 64, 64))
    assert logits.shape == (1, 43)
    changed_head = build_model(model_name, 5, pretrained=False)
    info = load_state_flexible(model, changed_head.state_dict())
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
    batch = clf.predict([Image.open(img_path), np.zeros((32, 32, 3), dtype=np.uint8)], topk=2)
    assert isinstance(batch, list) and len(batch) == 2
    assert len(batch[0]["topk"]) == 2
    with pytest.raises(TypeError):
        clf.predict({"bad": "input"})


def test_ensemble_probability_alignment_and_metrics() -> None:
    first = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
    second = np.asarray([[0.4, 0.6], [0.9, 0.1]], dtype=np.float64)
    fused = fuse_probabilities([first, second], [0.8, 0.2])
    assert np.allclose(fused.sum(axis=1), 1.0)
    assert fused.argmax(axis=1).tolist() == [0, 1]
    assert probability_metrics(np.asarray([0, 1]), fused, 2)["macro_f1"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        fuse_probabilities([first, np.zeros((3, 2))], [0.5, 0.5])
