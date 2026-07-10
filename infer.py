import argparse
import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from datasets import list_images
from models import create_model
from transforms import build_transforms


class TestImageDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.paths = list_images(test_dir)
        self.root = test_dir
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        rel = os.path.relpath(path, self.root).replace('\\', '/')
        return img, rel


def parse_args():
    p = argparse.ArgumentParser(description='Inference for RAICOM traffic sign baseline')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--test-dir', required=True)
    p.add_argument('--output', default='submission.csv')
    p.add_argument('--model', default=None, choices=[None, 'efficientnet_b0', 'efficientnet_b2', 'efficientnet_v2_s', 'convnext_tiny', 'resnet50'])
    p.add_argument('--img-size', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--filename-col', default='image')
    p.add_argument('--label-col', default='label')
    p.add_argument('--save-class-name', action='store_true')
    p.add_argument('--tta', action='store_true', help='Simple TTA: center crop + slightly larger resize center crop')
    return p.parse_args()


def predict_loader(model, loader, device):
    probs_all = []
    names_all = []
    model.eval()
    with torch.no_grad():
        for images, names in tqdm(loader):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu()
            probs_all.append(probs)
            names_all.extend(list(names))
    return torch.cat(probs_all, dim=0), names_all


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt.get('classes')
    if classes is None:
        raise RuntimeError('Checkpoint must contain classes. Use train.py checkpoint best.pt.')
    model_name = args.model or ckpt.get('model_name', 'efficientnet_b0')
    img_size = args.img_size or ckpt.get('img_size', 224)

    model = create_model(model_name, len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt['model'])
    tf = build_transforms(img_size, train=False)
    ds = TestImageDataset(args.test_dir, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=(device == 'cuda'))

    probs, names = predict_loader(model, loader, device)
    pred_idx = probs.argmax(dim=1).numpy().tolist()

    df = pd.DataFrame({args.filename_col: names, args.label_col: pred_idx})
    if args.save_class_name:
        df['class_name'] = [classes[i] for i in pred_idx]
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print('Saved:', args.output, 'Images:', len(df))


if __name__ == '__main__':
    main()
