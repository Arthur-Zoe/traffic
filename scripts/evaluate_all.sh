#!/usr/bin/env bash
set -euo pipefail

CKPTS=(
  "outputs/gtsrb_effb0_weather_seed42/best.pt"
  "outputs/gtsrb_effb0_weather_v2_seed42/best.pt"
)

for ckpt in "${CKPTS[@]}"; do
  if [[ ! -f "$ckpt" ]]; then
    echo "Skipping missing checkpoint: $ckpt"
    continue
  fi
  name="$(basename "$(dirname "$ckpt")")"
  python evaluate.py \
    --data-dir data/gtsrb \
    --csv Test.csv \
    --checkpoint "$ckpt" \
    --mode clean \
    --output-dir "outputs/eval/${name}/clean" \
    "$@"
  python evaluate.py \
    --data-dir data/gtsrb \
    --csv Test.csv \
    --checkpoint "$ckpt" \
    --mode random-stress \
    --severity strong \
    --repeats 3 \
    --seed 42 \
    --output-dir "outputs/eval/${name}/random_stress" \
    "$@"
  python evaluate.py \
    --data-dir data/gtsrb \
    --csv Test.csv \
    --checkpoint "$ckpt" \
    --mode per-corruption \
    --severity strong \
    --output-dir "outputs/eval/${name}/per_corruption" \
    "$@"
done
