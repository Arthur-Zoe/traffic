from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance

import torch
from torchvision import transforms

from models import build_model, checkpoint_state_dict


DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def to_pil_image(item: str | Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    if isinstance(item, (str, Path)):
        with Image.open(item) as img:
            return img.convert("RGB")
    if isinstance(item, np.ndarray):
        arr = item
        if arr.ndim == 2:
            return Image.fromarray(arr).convert("RGB")
        if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
            if arr.shape[2] == 1:
                arr = arr[:, :, 0]
            return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        raise TypeError(f"Unsupported numpy image shape: {arr.shape}")
    raise TypeError(f"Unsupported input type: {type(item).__name__}. Expected path, PIL.Image, or numpy.ndarray.")


class TrafficSignClassifier:
    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        topk: int = 5,
        tta: bool = False,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        self.topk = topk
        self.tta = tta
        self.model: torch.nn.Module | None = None
        self.model_name = "efficientnet_b0"
        self.num_classes = 43
        self.img_size = 224
        self.mean = DEFAULT_MEAN
        self.std = DEFAULT_STD
        self.idx_to_class: dict[int, str] = {}
        self._base_tf: transforms.Compose | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        ckpt = torch.load(self.checkpoint, map_location=self.device)
        self.model_name = str(ckpt.get("model_name", "efficientnet_b0"))
        self.num_classes = int(ckpt.get("num_classes", 43))
        self.img_size = int(ckpt.get("img_size", 224))
        self.mean = list(ckpt.get("mean", DEFAULT_MEAN))
        self.std = list(ckpt.get("std", DEFAULT_STD))
        idx_to_class_raw = ckpt.get("idx_to_class")
        if isinstance(idx_to_class_raw, dict):
            self.idx_to_class = {int(k): str(v) for k, v in idx_to_class_raw.items()}
        elif ckpt.get("classes") is not None:
            self.idx_to_class = {i: str(v) for i, v in enumerate(ckpt["classes"])}
        else:
            self.idx_to_class = {i: str(i) for i in range(self.num_classes)}
        self.model = build_model(self.model_name, self.num_classes, pretrained=False)
        state = {key.removeprefix("module."): value for key, value in checkpoint_state_dict(ckpt).items()}
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()
        self._base_tf = transforms.Compose(
            [
                transforms.Resize(int(self.img_size * 1.15)),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def _variants(self, img: Image.Image) -> list[Image.Image]:
        img = img.convert("RGB")
        if not self.tta:
            return [img]
        return [
            img,
            ImageEnhance.Brightness(img).enhance(0.94),
            ImageEnhance.Brightness(img).enhance(1.06),
            img.resize((max(1, int(img.width * 1.08)), max(1, int(img.height * 1.08))), Image.Resampling.BICUBIC),
        ]

    @torch.no_grad()
    def _predict_images(self, images: list[Image.Image], topk: int | None = None) -> list[dict[str, Any]]:
        self.load()
        assert self.model is not None
        assert self._base_tf is not None
        if not images:
            return []
        k = min(topk or self.topk, self.num_classes)
        variants = [self._variants(image) for image in images]
        variants_per_image = len(variants[0])
        tensors = [self._base_tf(variant) for image_variants in variants for variant in image_variants]
        batch = torch.stack(tensors, dim=0).to(self.device)
        logits = self.model(batch)
        probabilities = torch.softmax(logits, dim=1).reshape(len(images), variants_per_image, self.num_classes).mean(dim=1)
        values, indices = torch.topk(probabilities, k=k, dim=1)
        results: list[dict[str, Any]] = []
        for row_values, row_indices in zip(values.cpu().tolist(), indices.cpu().tolist()):
            top = [
                {
                    "index": int(idx),
                    "class_name": self.idx_to_class.get(int(idx), str(int(idx))),
                    "confidence": float(value),
                }
                for value, idx in zip(row_values, row_indices)
            ]
            best = top[0]
            results.append(
                {
                    "index": best["index"],
                    "class_name": best["class_name"],
                    "confidence": best["confidence"],
                    "topk": top,
                }
            )
        return results

    @torch.no_grad()
    def predict_one(self, item: str | Path | Image.Image | np.ndarray, topk: int | None = None) -> dict[str, Any]:
        img = to_pil_image(item)
        return self._predict_images([img], topk=topk)[0]

    def predict(self, items: Any, topk: int | None = None) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(items, (str, Path, Image.Image, np.ndarray)):
            return self.predict_one(items, topk=topk)
        if isinstance(items, (list, tuple)):
            return self._predict_images([to_pil_image(item) for item in items], topk=topk)
        raise TypeError(f"Unsupported input type: {type(items).__name__}. Expected single image input or list/tuple batch.")


_DEFAULT_CLASSIFIER: TrafficSignClassifier | None = None


def configure(checkpoint: str | Path, device: str = "cpu", topk: int = 5, tta: bool = False) -> None:
    global _DEFAULT_CLASSIFIER
    _DEFAULT_CLASSIFIER = TrafficSignClassifier(checkpoint=checkpoint, device=device, topk=topk, tta=tta)


def classify(item: Any, checkpoint: str | Path | None = None, device: str = "cpu", topk: int = 5, tta: bool = False):
    global _DEFAULT_CLASSIFIER
    if checkpoint is not None:
        classifier = TrafficSignClassifier(checkpoint=checkpoint, device=device, topk=topk, tta=tta)
        return classifier.predict(item, topk=topk)
    if _DEFAULT_CLASSIFIER is None:
        raise RuntimeError("Classifier is not configured. Call configure(checkpoint=...) or pass checkpoint=... to classify().")
    return _DEFAULT_CLASSIFIER.predict(item, topk=topk)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single image inference helper")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--tta", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = classify(args.image, checkpoint=args.checkpoint, device=args.device, topk=args.topk, tta=args.tta)
    print(result)


if __name__ == "__main__":
    main()
