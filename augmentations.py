from __future__ import annotations

from io import BytesIO
import random
from typing import Callable, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


Severity = str
CorruptionFn = Callable[[Image.Image, random.Random, Severity], Image.Image]

CORRUPTION_NAMES = [
    "rain",
    "fog",
    "snow",
    "darkness",
    "brightness",
    "contrast",
    "gaussian_blur",
    "motion_blur",
    "gaussian_noise",
    "jpeg_compression",
    "low_resolution",
    "shadow",
    "glare",
]


def _rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB")


def _severity_value(severity: Severity, light: float, medium: float, strong: float) -> float:
    if severity == "light":
        return light
    if severity == "medium":
        return medium
    if severity == "strong":
        return strong
    raise ValueError(f"Unsupported severity: {severity}")


def _safe_int(rng: random.Random, low: int, high: int) -> int:
    if high < low:
        high = low
    return rng.randint(low, high)


def _avoid_degenerate(original: Image.Image, candidate: Image.Image) -> Image.Image:
    candidate = _rgb(candidate)
    arr = np.asarray(candidate, dtype=np.float32)
    mean = float(arr.mean())
    std = float(arr.std())
    if mean < 8.0 or mean > 247.0 or std < 3.0:
        return _rgb(original)
    return candidate


def add_rain(img: Image.Image, rng: random.Random, severity: Severity = "medium") -> Image.Image:
    original = _rgb(img)
    out = original.copy()
    w, h = out.size
    draw = ImageDraw.Draw(out)
    density = _severity_value(severity, 0.10, 0.18, 0.27)
    count = max(1, int((w + h) * density))
    max_len = max(4, int(min(w, h) * _severity_value(severity, 0.13, 0.18, 0.24)))
    angle = rng.randint(-7, 8)
    for _ in range(count):
        x = rng.randint(-max(1, w // 4), max(0, w - 1))
        y = rng.randint(0, max(0, h - 1))
        length = _safe_int(rng, max(3, max_len // 2), max_len)
        gray = rng.randint(170, 225)
        draw.line((x, y, x + angle, y + length), fill=(gray, gray, gray), width=1)
    out = out.filter(ImageFilter.GaussianBlur(radius=_severity_value(severity, 0.12, 0.25, 0.40)))
    out = ImageEnhance.Brightness(out).enhance(_severity_value(severity, 0.92, 0.82, 0.72))
    out = ImageEnhance.Contrast(out).enhance(_severity_value(severity, 0.96, 0.88, 0.80))
    return _avoid_degenerate(original, out)


def add_fog(img: Image.Image, rng: random.Random, severity: Severity = "medium") -> Image.Image:
    original = _rgb(img)
    w, h = original.size
    fog_color = rng.randint(220, 248)
    fog = Image.new("RGB", (w, h), (fog_color, fog_color, fog_color))
    alpha = rng.uniform(
        _severity_value(severity, 0.10, 0.22, 0.32),
        _severity_value(severity, 0.22, 0.38, 0.52),
    )
    out = Image.blend(original, fog, alpha)
    out = ImageEnhance.Contrast(out).enhance(_severity_value(severity, 0.82, 0.68, 0.55))
    out = ImageEnhance.Sharpness(out).enhance(_severity_value(severity, 0.92, 0.75, 0.60))
    return _avoid_degenerate(original, out)


def add_snow(img: Image.Image, rng: random.Random, severity: Severity = "medium") -> Image.Image:
    original = _rgb(img)
    out = original.copy()
    w, h = out.size
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    density = _severity_value(severity, 0.16, 0.28, 0.42)
    count = max(1, int((w + h) * density))
    for _ in range(count):
        x = rng.randint(0, max(0, w - 1))
        y = rng.randint(0, max(0, h - 1))
        r = rng.choice([1, 1, 1, 2] if severity != "strong" else [1, 1, 2, 2, 3])
        alpha = rng.randint(70, 165)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, alpha))
    out = Image.alpha_composite(out.convert("RGBA"), overlay).convert("RGB")
    out = ImageEnhance.Contrast(out).enhance(_severity_value(severity, 0.95, 0.85, 0.72))
    return _avoid_degenerate(original, out)


def add_darkness(img: Image.Image, rng: random.Random, severity: Severity = "medium") -> Image.Image:
    original = _rgb(img)
    factor = rng.uniform(
        _severity_value(severity, 0.70, 0.50, 0.34),
        _severity_value(severity, 0.88, 0.76, 0.62),
    )
    out = ImageEnhance.Brightness(original).enhance(factor)
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.85, 1.18))
    out = ImageEnhance.Color(out).enhance(rng.uniform(0.75, 1.05))
    return _avoid_degenerate(original, out)


def add_brightness(img: Image.Image, rng: random.Random, severity: Severity = "medium") -> Image.Image:
    original = _rgb(img)
    factor = rng.uniform(
        _severity_value(severity, 1.08, 1.15, 1.25),
        _severity_value(severity, 1.25, 1.42, 1.62),
    )
    out = ImageEnhance.Brightness(original).enhance(factor)
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.85, 1.08))
    return _avoid_degenerate(original, out)


def add_contrast(img: Image.Image, rng: random.Random, severity: Severity = "medium") -> Image.Image:
    original = _rgb(img)
    if rng.random() < 0.5:
        factor = rng.uniform(
            _severity_value(severity, 0.72, 0.55, 0.42),
            _severity_value(severity, 0.90, 0.75, 0.62),
        )
    else:
        factor = rng.uniform(
            _severity_value(severity, 1.10, 1.22, 1.35),
            _severity_value(severity, 1.35, 1.55, 1.80),
        )
    return _avoid_degenerate(original, ImageEnhance.Contrast(original).enhance(factor))


def add_gaussian_blur(img: Image.Image, rng: random.Random, severity: Severity = "medium") -> Image.Image:
    original = _rgb(img)
    radius = rng.uniform(
        _severity_value(severity, 0.20, 0.45, 0.75),
        _severity_value(severity, 0.75, 1.35, 2.00),
    )
    return _avoid_degenerate(original, original.filter(ImageFilter.GaussianBlur(radius=radius)))


def add_motion_blur(img: Image.Image, rng: random.Random, severity: Severity = "medium") -> Image.Image:
    original = _rgb(img)
    k = 3 if severity == "light" else rng.choice([3, 5])
    weights = [0.0] * (k * k)
    if rng.random() < 0.5:
        row = k // 2
        for col in range(k):
            weights[row * k + col] = 1.0 / k
    else:
        col = k // 2
        for row in range(k):
            weights[row * k + col] = 1.0 / k
    return _avoid_degenerate(original, original.filter(ImageFilter.Kernel((k, k), weights, scale=1.0)))


def add_gaussian_noise(img: Image.Image, rng: random.Random, severity: Severity = "medium") -> Image.Image:
    original = _rgb(img)
    arr = np.asarray(original, dtype=np.float32)
    sigma = rng.uniform(
        _severity_value(severity, 2.0, 5.0, 9.0),
        _severity_value(severity, 7.0, 14.0, 22.0),
    )
    np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
    noisy = np.clip(arr + np_rng.normal(0.0, sigma, arr.shape), 0, 255).astype(np.uint8)
    return _avoid_degenerate(original, Image.fromarray(noisy, mode="RGB"))


def add_jpeg_compression(img: Image.Image, rng: random.Random, severity: Severity = "medium") -> Image.Image:
    original = _rgb(img)
    quality = rng.randint(
        int(_severity_value(severity, 55, 35, 22)),
        int(_severity_value(severity, 85, 70, 55)),
    )
    with BytesIO() as buffer:
        original.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            out = decoded.convert("RGB").copy()
    return _avoid_degenerate(original, out)


def add_low_resolution(img: Image.Image, rng: random.Random, severity: Severity = "medium") -> Image.Image:
    original = _rgb(img)
    w, h = original.size
    scale = rng.uniform(
        _severity_value(severity, 0.68, 0.52, 0.38),
        _severity_value(severity, 0.88, 0.78, 0.66),
    )
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    small = original.resize((nw, nh), Image.Resampling.BILINEAR)
    out = small.resize((w, h), Image.Resampling.BILINEAR)
    return _avoid_degenerate(original, out)


def add_shadow(img: Image.Image, rng: random.Random, severity: Severity = "medium") -> Image.Image:
    original = _rgb(img)
    w, h = original.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    x1 = rng.randint(-max(1, w // 2), max(0, w // 2))
    x2 = rng.randint(max(0, w // 2), max(1, int(w * 1.5)))
    polygon = [
        (x1, 0),
        (x2, 0),
        (rng.randint(max(0, w // 2), max(1, int(w * 1.5))), h),
        (rng.randint(-max(1, w // 2), max(0, w // 2)), h),
    ]
    alpha = int(_severity_value(severity, 45, 75, 105))
    draw.polygon(polygon, fill=alpha)
    shadow = Image.new("RGB", (w, h), (0, 0, 0))
    out = Image.composite(shadow, original, mask)
    return _avoid_degenerate(original, out)


def add_glare(img: Image.Image, rng: random.Random, severity: Severity = "medium") -> Image.Image:
    original = _rgb(img)
    w, h = original.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    cx = rng.randint(0, max(0, w - 1))
    cy = rng.randint(0, max(0, h - 1))
    r_min = max(2, min(w, h) // 10)
    r_max = max(r_min, min(w, h) // int(_severity_value(severity, 4, 3, 2)))
    radius = rng.randint(r_min, r_max)
    alpha = int(_severity_value(severity, 42, 68, 92))
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=alpha)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(1.0, radius / 2.5)))
    out = Image.composite(Image.new("RGB", (w, h), (255, 255, 255)), original, mask)
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.86, 1.04))
    return _avoid_degenerate(original, out)


CORRUPTIONS: dict[str, CorruptionFn] = {
    "rain": add_rain,
    "fog": add_fog,
    "snow": add_snow,
    "darkness": add_darkness,
    "brightness": add_brightness,
    "contrast": add_contrast,
    "gaussian_blur": add_gaussian_blur,
    "motion_blur": add_motion_blur,
    "gaussian_noise": add_gaussian_noise,
    "jpeg_compression": add_jpeg_compression,
    "low_resolution": add_low_resolution,
    "shadow": add_shadow,
    "glare": add_glare,
}


def apply_corruption(
    img: Image.Image,
    corruption: str,
    severity: Severity = "medium",
    seed: int | None = None,
    rng: random.Random | None = None,
) -> Image.Image:
    if corruption not in CORRUPTIONS:
        raise ValueError(f"Unsupported corruption: {corruption}")
    local_rng = rng if rng is not None else random.Random(seed)
    return CORRUPTIONS[corruption](_rgb(img), local_rng, severity)


class RandomBadWeather:
    def __init__(
        self,
        p: float = 0.45,
        severity: Severity = "medium",
        max_ops: int = 3,
        seed: int | None = None,
        corruptions: Iterable[str] | None = None,
        fixed_corruption: str | None = None,
    ) -> None:
        self.p = p
        self.severity = severity
        self.max_ops = max(1, int(max_ops))
        # DataLoader seeds Python's global RNG independently per worker. Use it
        # when no explicit seed is requested; fixed seeds retain local state.
        self.rng = random.Random(seed) if seed is not None else random
        self.corruptions = list(corruptions) if corruptions is not None else list(CORRUPTION_NAMES)
        self.fixed_corruption = fixed_corruption
        for name in self.corruptions:
            if name not in CORRUPTIONS:
                raise ValueError(f"Unsupported corruption: {name}")
        if fixed_corruption is not None and fixed_corruption not in CORRUPTIONS:
            raise ValueError(f"Unsupported corruption: {fixed_corruption}")

    def __call__(self, img: Image.Image) -> Image.Image:
        out = _rgb(img)
        if self.rng.random() > self.p:
            return out
        if self.fixed_corruption is not None:
            return apply_corruption(out, self.fixed_corruption, self.severity, rng=self.rng)
        n_ops = self.rng.randint(1, min(self.max_ops, len(self.corruptions)))
        for name in self.rng.sample(self.corruptions, k=n_ops):
            out = apply_corruption(out, name, self.severity, rng=self.rng)
        return _rgb(out)
