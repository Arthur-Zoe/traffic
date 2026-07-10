#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-outputs/gtsrb_v4_extreme_seed42}"
OVERWRITE="${OVERWRITE:-0}"
if [[ -e "$OUT/best.pt" && "$OVERWRITE" != "1" ]]; then
  echo "Refusing to overwrite $OUT. Set OVERWRITE=1 to allow."
  exit 1
fi

python train_gtsrb.py \
  --data-dir data/gtsrb \
  --preset v4_extreme \
  --epochs "${EPOCHS:-30}" \
  --batch-size "${BATCH_SIZE:-28}" \
  --workers "${WORKERS:-4}" \
  --pretrained \
  --class-weight sqrt_inverse \
  --output-dir "$OUT" \
  "$@"
