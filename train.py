import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datasets import build_train_dataset
from metrics import compute_metrics, save_report
from models import create_model, load_flexible_checkpoint, set_backbone_trainable
from transforms import build_transforms


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def parse_args():
    p = argparse.ArgumentParser(description='RAICOM traffic sign classification baseline')
    # Folder mode: data_dir/class_x/*.jpg
    p.add_argument('--data-dir', default=None, help='Folder-format training dir. Example: data/gtsrb_folders/train')
    # CSV mode: csv contains image,label columns by default
    p.add_argument('--csv', default=None, help='CSV training labels path')
    p.add_argument('--image-dir', default=None, help='Image root for CSV mode')
    p.add_argument('--image-col', default='image')
    p.add_argument('--label-col', default='label')

    p.add_argument('--model', default='efficientnet_b0', choices=['efficientnet_b0', 'efficientnet_b2', 'efficientnet_v2_s', 'convnext_tiny', 'resnet50'])
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--label-smoothing', type=float, default=0.05)
    p.add_argument('--val-ratio', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', default='outputs/traffic_baseline')

    p.add_argument('--no-imagenet', action='store_true', help='Do not use ImageNet weights')
    p.add_argument('--init-checkpoint', default=None, help='Load compatible weights; classifier mismatch is skipped')
    p.add_argument('--freeze-backbone-epochs', type=int, default=0, help='Freeze backbone for first N epochs')

    p.add_argument('--class-weight', default='none', choices=['none', 'inverse', 'sqrt_inverse'])
    p.add_argument('--sampler', action='store_true', help='Use WeightedRandomSampler for train split')
    p.add_argument('--strong-weather', action='store_true', help='Enable rain/snow/fog/motion blur augmentations')
    p.add_argument('--amp', action='store_true', help='Use mixed precision on CUDA')
    p.add_argument('--patience', type=int, default=8)
    return p.parse_args()


def get_targets(dataset):
    if hasattr(dataset, 'targets') and dataset.targets is not None:
        return np.array(dataset.targets)
    raise RuntimeError('Dataset has no targets.')


def make_split(targets, val_ratio, seed):
    idx = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_ratio, random_state=seed, stratify=targets
    )
    return train_idx.tolist(), val_idx.tolist()


def make_class_weights(targets, num_classes, mode, device):
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    if mode == 'inverse':
        w = 1.0 / counts
    elif mode == 'sqrt_inverse':
        w = 1.0 / np.sqrt(counts)
    else:
        return None
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


def make_sampler(train_targets, num_classes):
    counts = np.bincount(train_targets, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    cls_w = 1.0 / np.sqrt(counts)
    sample_w = cls_w[train_targets]
    return WeightedRandomSampler(sample_w.tolist(), num_samples=len(sample_w), replacement=True)


def run_one_epoch(model, loader, criterion, optimizer, device, scaler=None, train=True):
    model.train(train)
    losses = []
    all_true = []
    all_pred = []

    pbar = tqdm(loader, leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            if scaler is not None and train:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, targets)
                if train:
                    loss.backward()
                    optimizer.step()

        pred = logits.argmax(dim=1).detach().cpu().numpy()
        all_pred.extend(pred.tolist())
        all_true.extend(targets.detach().cpu().numpy().tolist())
        losses.append(loss.item())
        pbar.set_description(('train' if train else 'val') + ' loss %.4f' % np.mean(losses))

    metrics = compute_metrics(all_true, all_pred, labels=list(range(logits.shape[1])))
    metrics['loss'] = float(np.mean(losses))
    return metrics, all_true, all_pred


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_tf = build_transforms(args.img_size, train=True, strong_weather=args.strong_weather)
    val_tf = build_transforms(args.img_size, train=False)

    # Build once for labels and classes, then build two datasets with different transforms.
    base_ds = build_train_dataset(data_dir=args.data_dir, csv_path=args.csv, image_dir=args.image_dir,
                                  image_col=args.image_col, label_col=args.label_col, transform=None)
    classes = list(base_ds.classes)
    num_classes = len(classes)
    targets = get_targets(base_ds)

    train_ds_full = build_train_dataset(data_dir=args.data_dir, csv_path=args.csv, image_dir=args.image_dir,
                                        image_col=args.image_col, label_col=args.label_col, transform=train_tf)
    val_ds_full = build_train_dataset(data_dir=args.data_dir, csv_path=args.csv, image_dir=args.image_dir,
                                      image_col=args.image_col, label_col=args.label_col, transform=val_tf)
    train_idx, val_idx = make_split(targets, args.val_ratio, args.seed)
    train_ds = Subset(train_ds_full, train_idx)
    val_ds = Subset(val_ds_full, val_idx)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)
    print('Classes:', num_classes)
    print('Train images:', len(train_ds), 'Val images:', len(val_ds))

    train_targets = targets[train_idx]
    sampler = make_sampler(train_targets, num_classes) if args.sampler else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
                              num_workers=args.workers, pin_memory=(device == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=(device == 'cuda'))

    model = create_model(args.model, num_classes, pretrained=not args.no_imagenet).to(device)
    if args.init_checkpoint:
        info = load_flexible_checkpoint(model, args.init_checkpoint, device=device)
        print('Loaded checkpoint:', info)

    if args.freeze_backbone_epochs > 0:
        set_backbone_trainable(model, False)
        print('Backbone frozen for first %d epoch(s).' % args.freeze_backbone_epochs)

    cw = make_class_weights(train_targets, num_classes, args.class_weight, device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler() if args.amp and device == 'cuda' else None

    with open(os.path.join(args.output_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
        for c in classes:
            f.write(str(c) + '\n')
    with open(os.path.join(args.output_dir, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    best_f1 = -1.0
    best_epoch = -1
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        if epoch == args.freeze_backbone_epochs + 1 and args.freeze_backbone_epochs > 0:
            set_backbone_trainable(model, True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - epoch + 1))
            print('Backbone unfrozen.')

        train_m, _, _ = run_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler, train=True)
        val_m, y_true, y_pred = run_one_epoch(model, val_loader, criterion, optimizer, device, scaler=None, train=False)
        scheduler.step()

        row = {
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            'train_loss': train_m['loss'],
            'train_acc': train_m['accuracy'],
            'train_macro_f1': train_m['macro_f1'],
            'val_loss': val_m['loss'],
            'val_acc': val_m['accuracy'],
            'val_macro_f1': val_m['macro_f1'],
            'seconds': time.time() - start,
        }
        history.append(row)
        print('[%03d/%03d] train_f1=%.6f val_f1=%.6f val_acc=%.6f loss=%.4f time=%.1fs' %
              (epoch, args.epochs, row['train_macro_f1'], row['val_macro_f1'], row['val_acc'], row['val_loss'], row['seconds']))

        with open(os.path.join(args.output_dir, 'history.json'), 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        if val_m['macro_f1'] > best_f1:
            best_f1 = val_m['macro_f1']
            best_epoch = epoch
            no_improve = 0
            ckpt = {
                'model': model.state_dict(),
                'classes': classes,
                'num_classes': num_classes,
                'model_name': args.model,
                'img_size': args.img_size,
                'best_f1': best_f1,
                'epoch': epoch,
                'args': vars(args),
            }
            torch.save(ckpt, os.path.join(args.output_dir, 'best.pt'))
            save_report(y_true, y_pred, classes,
                        os.path.join(args.output_dir, 'val_report.txt'),
                        os.path.join(args.output_dir, 'confusion_matrix.csv'))
            print('  saved best.pt, best val macro F1 = %.6f, score = %.3f' % (best_f1, best_f1 * 100))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print('Early stopping. Best epoch:', best_epoch, 'best_f1:', best_f1)
                break

    print('Finished. Best val macro F1 = %.6f, score = %.3f at epoch %d' % (best_f1, best_f1 * 100, best_epoch))


if __name__ == '__main__':
    main()
