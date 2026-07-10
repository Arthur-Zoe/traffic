import io
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import torchvision.transforms as T


class RandomJPEGCompression(object):
    def __init__(self, p=0.2, quality=(35, 90)):
        self.p = p
        self.quality = quality

    def __call__(self, img):
        if random.random() > self.p:
            return img
        q = random.randint(self.quality[0], self.quality[1])
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        return Image.open(buf).convert('RGB')


class RandomFog(object):
    def __init__(self, p=0.25, strength=(0.12, 0.38)):
        self.p = p
        self.strength = strength

    def __call__(self, img):
        if random.random() > self.p:
            return img
        alpha = random.uniform(self.strength[0], self.strength[1])
        white = Image.new('RGB', img.size, (245, 245, 245))
        out = Image.blend(img.convert('RGB'), white, alpha)
        out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.2)))
        return out


class RandomRain(object):
    def __init__(self, p=0.25, drops=(35, 120), length=(8, 22)):
        self.p = p
        self.drops = drops
        self.length = length

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img = img.convert('RGB')
        w, h = img.size
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        n = random.randint(self.drops[0], self.drops[1])
        angle = random.uniform(-0.5, 0.5)
        for _ in range(n):
            x = random.randint(0, max(0, w - 1))
            y = random.randint(0, max(0, h - 1))
            l = random.randint(self.length[0], self.length[1])
            x2 = int(x + l * angle)
            y2 = int(y + l)
            a = random.randint(45, 105)
            draw.line((x, y, x2, y2), fill=(220, 220, 220, a), width=1)
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=0.45))
        return Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')


class RandomSnow(object):
    def __init__(self, p=0.2, flakes=(50, 160)):
        self.p = p
        self.flakes = flakes

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img = img.convert('RGB')
        w, h = img.size
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        n = random.randint(self.flakes[0], self.flakes[1])
        for _ in range(n):
            x = random.randint(0, max(0, w - 1))
            y = random.randint(0, max(0, h - 1))
            r = random.choice([1, 1, 2])
            a = random.randint(70, 150)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, a))
        return Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')


class RandomMotionBlur(object):
    def __init__(self, p=0.2, kernel_size=(3, 7)):
        self.p = p
        self.kernel_size = kernel_size

    def __call__(self, img):
        if random.random() > self.p:
            return img
        k = random.choice([3, 5, 7])
        if k < self.kernel_size[0] or k > self.kernel_size[1]:
            k = self.kernel_size[0]
        # Horizontal or vertical motion blur. This avoids scipy/opencv dependency.
        weights = [0.0] * (k * k)
        if random.random() < 0.5:
            row = k // 2
            for col in range(k):
                weights[row * k + col] = 1.0 / k
        else:
            col = k // 2
            for row in range(k):
                weights[row * k + col] = 1.0 / k
        return img.filter(ImageFilter.Kernel((k, k), weights, scale=1.0))


class RandomLowLight(object):
    def __init__(self, p=0.25, factor=(0.35, 0.85)):
        self.p = p
        self.factor = factor

    def __call__(self, img):
        if random.random() > self.p:
            return img
        f = random.uniform(self.factor[0], self.factor[1])
        img = ImageEnhance.Brightness(img).enhance(f)
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.75, 1.2))
        return img


class RandomOverExposure(object):
    def __init__(self, p=0.15, factor=(1.25, 1.75)):
        self.p = p
        self.factor = factor

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img = ImageEnhance.Brightness(img).enhance(random.uniform(self.factor[0], self.factor[1]))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.15))
        return img


def build_transforms(img_size=224, train=True, strong_weather=False):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    if train:
        pre = [
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomResizedCrop(img_size, scale=(0.72, 1.0), ratio=(0.85, 1.15)),
            # Do not use RandomHorizontalFlip by default: traffic sign direction may change.
            T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.03),
            T.RandomAutocontrast(p=0.25),
            T.RandomAdjustSharpness(sharpness_factor=1.8, p=0.2),
            RandomLowLight(p=0.22),
            RandomOverExposure(p=0.12),
            RandomJPEGCompression(p=0.22),
        ]
        if strong_weather:
            pre += [
                RandomFog(p=0.28),
                RandomRain(p=0.24),
                RandomSnow(p=0.18),
                RandomMotionBlur(p=0.20),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.6)),
            ]
        return T.Compose(pre + [
            T.ToTensor(),
            normalize,
            T.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3), value='random'),
        ])

    return T.Compose([
        T.Resize((img_size + 32, img_size + 32)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        normalize,
    ])
