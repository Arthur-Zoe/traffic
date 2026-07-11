#!/usr/bin/env bash
set -euo pipefail

NAMES=(v2 v7_convnext_tiny v8_efficientnet_b2)
CKPTS=(
  "outputs/gtsrb_effb0_weather_v2_seed42/best.pt"
  "outputs/gtsrb_v7_convnext_tiny_seed42/best.pt"
  "outputs/gtsrb_v8_efficientnet_b2_seed42/best.pt"
)

for index in "${!CKPTS[@]}"; do
  name="${NAMES[$index]}"
  checkpoint="${CKPTS[$index]}"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint"
    exit 1
  fi
  base="outputs/eval_architectures/${name}"
  python evaluate.py --data-dir data/gtsrb --csv Test.csv --checkpoint "$checkpoint" --mode clean --output-dir "$base/clean" "$@"
  python evaluate.py --data-dir data/gtsrb --csv Test.csv --checkpoint "$checkpoint" --mode random-stress --severity strong --repeats 3 --seed 42 --output-dir "$base/random_stress" "$@"
  python evaluate.py --data-dir data/gtsrb --csv Test.csv --checkpoint "$checkpoint" --mode per-corruption --severity strong --output-dir "$base/per_corruption" "$@"
done
