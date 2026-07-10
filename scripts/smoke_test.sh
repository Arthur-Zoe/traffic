#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-outputs/smoke_v3}"
OVERWRITE="${OVERWRITE:-0}"
if [[ -e "$OUT/best.pt" && "$OVERWRITE" != "1" ]]; then
  echo "Refusing to overwrite $OUT. Set OVERWRITE=1 to allow."
  exit 1
fi

python -m py_compile \
  train_gtsrb.py \
  augmentations.py \
  datasets.py \
  evaluate.py \
  compare_models.py \
  inspect_dataset.py \
  inference.py

python train_gtsrb.py \
  --data-dir data/gtsrb \
  --preset v3_balanced \
  --epochs 1 \
  --batch-size 8 \
  --workers 0 \
  --max-train-samples 256 \
  --max-val-samples 128 \
  --output-dir "$OUT" \
  "$@"
