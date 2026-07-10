import argparse
import os
from collections import Counter
from torchvision.datasets import ImageFolder


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    args = p.parse_args()
    ds = ImageFolder(args.data_dir)
    counts = Counter(ds.targets)
    print('Root:', args.data_dir)
    print('Images:', len(ds))
    print('Classes:', len(ds.classes))
    print('\nClass distribution:')
    for i, c in enumerate(ds.classes):
        print('%3d  %-40s  %6d' % (i, c, counts[i]))


if __name__ == '__main__':
    main()
