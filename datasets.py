import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def list_images(root):
    paths = []
    for base, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(base, f))
    return sorted(paths)


class CSVImageDataset(Dataset):
    def __init__(self, csv_path, image_dir, image_col='image', label_col='label', transform=None, class_to_idx=None, has_label=True):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform
        self.has_label = has_label and (label_col in self.df.columns)

        if self.has_label:
            labels = [str(x) for x in self.df[label_col].tolist()]
            if class_to_idx is None:
                classes = sorted(list(set(labels)))
                self.class_to_idx = {c: i for i, c in enumerate(classes)}
            else:
                self.class_to_idx = class_to_idx
            self.classes = [None] * len(self.class_to_idx)
            for k, v in self.class_to_idx.items():
                self.classes[v] = k
            self.targets = [self.class_to_idx[str(x)] for x in self.df[label_col].tolist()]
        else:
            self.class_to_idx = class_to_idx
            self.classes = None
            self.targets = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel = str(row[self.image_col])
        path = rel if os.path.isabs(rel) else os.path.join(self.image_dir, rel)
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.has_label:
            return img, self.targets[idx]
        return img, rel


def build_train_dataset(data_dir=None, csv_path=None, image_dir=None, image_col='image', label_col='label', transform=None):
    if csv_path is not None:
        if image_dir is None:
            image_dir = os.path.dirname(csv_path)
        ds = CSVImageDataset(csv_path, image_dir, image_col=image_col, label_col=label_col, transform=transform)
        return ds
    if data_dir is None:
        raise ValueError('Either data_dir or csv_path must be provided.')
    return ImageFolder(data_dir, transform=transform)
