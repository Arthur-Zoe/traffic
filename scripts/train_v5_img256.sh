#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-outputs/gtsrb_v5_img256_seed42}"
OVERWRITE="${OVERWRITE:-0}"
if [[ -e "$OUT/best.pt" && "$OVERWRITE" != "1" ]]; then
  echo "Refusing to overwrite $OUT. Set OVERWRITE=1 to allow."
  exit 1
fi

mkdir -p "$OUT"

python train_gtsrb.py \
  --data-dir data/gtsrb \
  --preset v5_img256 \
  --epochs "${EPOCHS:-30}" \
  --batch-size "${BATCH_SIZE:-24}" \
  --workers "${WORKERS:-4}" \
  --pretrained \
  --class-weight sqrt_inverse \
  --output-dir "$OUT" \
  "$@" 2>&1 | tee "$OUT/train.log"
