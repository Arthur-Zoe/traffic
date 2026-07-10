from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as tvm


SUPPORTED_MODELS = [
    "efficientnet_b0",
    "resnet18",
    "mobilenet_v3_large",
    "efficientnet_b2",
    "efficientnet_v2_s",
    "convnext_tiny",
    "resnet50",
]


def _get_weights(name: str, pretrained: bool) -> Any:
    if not pretrained:
        return None
    try:
        if name == "efficientnet_b0":
            return tvm.EfficientNet_B0_Weights.DEFAULT
        if name == "resnet18":
            return tvm.ResNet18_Weights.DEFAULT
        if name == "mobilenet_v3_large":
            return tvm.MobileNet_V3_Large_Weights.DEFAULT
        if name == "efficientnet_b2":
            return tvm.EfficientNet_B2_Weights.DEFAULT
        if name == "efficientnet_v2_s":
            return tvm.EfficientNet_V2_S_Weights.DEFAULT
        if name == "convnext_tiny":
            return tvm.ConvNeXt_Tiny_Weights.DEFAULT
        if name == "resnet50":
            return tvm.ResNet50_Weights.DEFAULT
    except AttributeError:
        return "OLD_TORCHVISION"
    return None


def create_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = _get_weights(model_name, pretrained)
    old = weights == "OLD_TORCHVISION"

    if model_name == "efficientnet_b0":
        model = tvm.efficientnet_b0(pretrained=pretrained) if old else tvm.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if model_name == "resnet18":
        model = tvm.resnet18(pretrained=pretrained) if old else tvm.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if model_name == "mobilenet_v3_large":
        model = tvm.mobilenet_v3_large(pretrained=pretrained) if old else tvm.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if model_name == "efficientnet_b2":
        model = tvm.efficientnet_b2(pretrained=pretrained) if old else tvm.efficientnet_b2(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if model_name == "efficientnet_v2_s":
        model = tvm.efficientnet_v2_s(pretrained=pretrained) if old else tvm.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if model_name == "convnext_tiny":
        model = tvm.convnext_tiny(pretrained=pretrained) if old else tvm.convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if model_name == "resnet50":
        model = tvm.resnet50(pretrained=pretrained) if old else tvm.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model_name: {model_name}")


def build_model(model_name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    return create_model(model_name, num_classes, pretrained=pretrained)


def count_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for param in model.parameters():
        param.requires_grad = trainable
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    if hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True


def checkpoint_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported checkpoint format.")


def load_state_flexible(model: nn.Module, state: dict[str, torch.Tensor]) -> dict[str, Any]:
    current = model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in state.items():
        clean_key = key.removeprefix("module.")
        if clean_key in current and current[clean_key].shape == value.shape:
            loaded[clean_key] = value
        else:
            skipped.append(key)
    current.update(loaded)
    model.load_state_dict(current)
    classifier_skipped = [
        key
        for key in skipped
        if "classifier" in key
        or key.removeprefix("module.") in {"fc.weight", "fc.bias"}
        or key.endswith(".fc.weight")
        or key.endswith(".fc.bias")
        or ".fc." in key
    ]
    return {
        "loaded": len(loaded),
        "skipped": len(skipped),
        "skipped_keys": skipped[:30],
        "classifier_skipped": classifier_skipped[:30],
    }


def load_flexible_checkpoint(model: nn.Module, checkpoint_path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    return load_state_flexible(model, checkpoint_state_dict(ckpt))
