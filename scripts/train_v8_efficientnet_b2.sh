#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-outputs/gtsrb_v8_efficientnet_b2_seed42}"
OVERWRITE="${OVERWRITE:-0}"
RESUME="${RESUME:-0}"
EXTRA_ARGS=()
if [[ -e "$OUT/last.pt" && "$RESUME" == "1" ]]; then
  EXTRA_ARGS+=(--resume "$OUT/last.pt")
elif [[ ( -e "$OUT/best.pt" || -e "$OUT/last.pt" ) && "$OVERWRITE" != "1" ]]; then
  echo "Refusing to overwrite $OUT. Set OVERWRITE=1 to allow."
  echo "Set RESUME=1 to continue from $OUT/last.pt."
  exit 1
fi

mkdir -p "$OUT"
python train_gtsrb.py \
  --data-dir data/gtsrb \
  --preset v8_efficientnet_b2 \
  --epochs "${EPOCHS:-20}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --workers "${WORKERS:-4}" \
  --pretrained \
  --class-weight sqrt_inverse \
  --output-dir "$OUT" \
  "${EXTRA_ARGS[@]}" \
  "$@" 2>&1 | tee -a "$OUT/train.log"
