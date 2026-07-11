#!/usr/bin/env bash
set -euo pipefail

python compare_models.py \
  --data-dir data/gtsrb \
  --csv Test.csv \
  --checkpoints \
    outputs/gtsrb_effb0_weather_v2_seed42/best.pt \
    outputs/gtsrb_v7_convnext_tiny_seed42/best.pt \
    outputs/gtsrb_v8_efficientnet_b2_seed42/best.pt \
  --output outputs/architecture_comparison_v2_convnext_b2.json \
  "$@"
