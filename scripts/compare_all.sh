#!/usr/bin/env bash
set -euo pipefail

python compare_models.py \
  --data-dir data/gtsrb \
  --csv Test.csv \
  --checkpoints \
    outputs/gtsrb_effb0_weather_seed42/best.pt \
    outputs/gtsrb_effb0_weather_v2_seed42/best.pt \
  --output outputs/model_comparison.json \
  "$@"
