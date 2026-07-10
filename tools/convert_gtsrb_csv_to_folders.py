"""
Convert common GTSRB Kaggle/CSV format to ImageFolder format.
Expected columns usually include Path and ClassId.
Example:
python tools/convert_gtsrb_csv_to_folders.py \
  --csv data/gtsrb/Train.csv \
  --src-root data/gtsrb \
  --out-dir data/gtsrb_folders/train
"""
import argparse
import os
import shutil
import pandas as pd
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--src-root', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--path-col', default='Path')
    p.add_argument('--label-col', default='ClassId')
    p.add_argument('--copy', action='store_true', help='Copy instead of symlink')
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.out_dir, exist_ok=True)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        rel = str(row[args.path_col]).replace('\\', '/')
        label = str(row[args.label_col])
        src = os.path.join(args.src_root, rel)
        dst_dir = os.path.join(args.out_dir, label)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(rel))
        if os.path.exists(dst):
            continue
        if args.copy:
            shutil.copy2(src, dst)
        else:
            try:
                os.symlink(os.path.abspath(src), dst)
            except OSError:
                shutil.copy2(src, dst)
    print('Done:', args.out_dir)


if __name__ == '__main__':
    main()
