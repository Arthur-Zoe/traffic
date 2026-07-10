#!/usr/bin/env bash
set -euo pipefail

python compare_models.py \
  --data-dir data/gtsrb \
  --csv Test.csv \
  --checkpoints \
    outputs/gtsrb_effb0_weather_seed42/best.pt \
    outputs/gtsrb_effb0_weather_v2_seed42/best.pt \
    outputs/gtsrb_v5_img256_seed42/best.pt \
    outputs/gtsrb_v6_mobile_seed42/best.pt \
  --output outputs/model_comparison_v1_v2_v5_v6.json \
  "$@"
