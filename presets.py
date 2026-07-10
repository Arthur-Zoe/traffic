from __future__ import annotations

from copy import deepcopy


PRESETS: dict[str, dict[str, object]] = {
    "v1_clean": {
        "model": "efficientnet_b0",
        "img_size": 224,
        "weather_aug": True,
        "weather_prob": 0.35,
        "weather_max_ops": 1,
        "weather_severity": "light",
    },
    "v2_strong": {
        "model": "efficientnet_b0",
        "img_size": 224,
        "weather_aug": True,
        "weather_prob": 0.45,
        "weather_max_ops": 3,
        "weather_severity": "strong",
    },
    "v3_balanced": {
        "model": "efficientnet_b0",
        "img_size": 224,
        "weather_aug": True,
        "weather_prob": 0.40,
        "weather_max_ops": 2,
        "weather_severity": "medium",
    },
    "v4_extreme": {
        "model": "efficientnet_b0",
        "img_size": 224,
        "weather_aug": True,
        "weather_prob": 0.55,
        "weather_max_ops": 3,
        "weather_severity": "strong",
    },
    "v5_img256": {
        "model": "efficientnet_b0",
        "img_size": 256,
        "batch_size": 24,
        "weather_aug": True,
        "weather_prob": 0.40,
        "weather_max_ops": 2,
        "weather_severity": "medium",
    },
    "v6_mobile": {
        "model": "mobilenet_v3_large",
        "img_size": 224,
        "weather_aug": True,
        "weather_prob": 0.40,
        "weather_max_ops": 2,
        "weather_severity": "medium",
    },
    "v7_convnext_tiny": {
        "model": "convnext_tiny",
        "img_size": 224,
        "epochs": 20,
        "batch_size": 16,
        "lr": 2e-4,
        "weight_decay": 5e-2,
        "label_smoothing": 0.05,
        "class_weight": "sqrt_inverse",
        "pretrained": True,
        "weather_aug": True,
        "weather_prob": 0.45,
        "weather_max_ops": 3,
        "weather_severity": "strong",
        "seed": 42,
        "early_stopping_patience": 6,
    },
    "v8_efficientnet_b2": {
        "model": "efficientnet_b2",
        "img_size": 260,
        "epochs": 20,
        "batch_size": 16,
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "label_smoothing": 0.05,
        "class_weight": "sqrt_inverse",
        "pretrained": True,
        "weather_aug": True,
        "weather_prob": 0.45,
        "weather_max_ops": 3,
        "weather_severity": "strong",
        "seed": 42,
        "early_stopping_patience": 6,
    },
}


def get_preset(name: str | None) -> dict[str, object]:
    if not name:
        return {}
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {', '.join(PRESETS)}")
    return deepcopy(PRESETS[name])
