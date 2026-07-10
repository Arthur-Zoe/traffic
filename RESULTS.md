# Benchmark Results

## Run

- Date: 2026-07-10
- Code commit at run start: `1f126f0`
- Environment: Ubuntu 22.04, conda `raicom_weather`, Python 3.10.20
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6GB
- PyTorch: 2.5.1+cu124, CUDA runtime: 12.4
- Evaluation: `evaluate.py`, GTSRB `Test.csv`, strong random-stress with `repeats=3`, `seed=42`

All figures below are produced by this run. Evaluation artifacts are under `outputs/eval/`; the four-model JSON is `outputs/model_comparison_v1_v2_v5_v6.json`.

## Checkpoints And Training

| Model | Checkpoint | Training configuration | Training result |
|---|---|---|---|
| v1 | `outputs/gtsrb_effb0_weather_seed42/best.pt` | Existing EfficientNet-B0 baseline | Preserved existing checkpoint |
| v2 | `outputs/gtsrb_effb0_weather_v2_seed42/best.pt` | Existing strong-weather EfficientNet-B0 | Preserved existing checkpoint |
| v5 | `outputs/gtsrb_v5_img256_seed42/best.pt` | `v5_img256`, EfficientNet-B0, 256, batch 24, 30 epochs, medium weather, probability 0.40, max ops 2 | Best validation epoch 15, Acc 1.000000, Macro-F1 1.000000 |
| v6 | `outputs/gtsrb_v6_mobile_seed42/best.pt` | `v6_mobile`, MobileNetV3-Large, 224, batch 32, medium weather, probability 0.40, max ops 2, patience 8 | Best validation epoch 10, Acc 1.000000, Macro-F1 1.000000; normal early stop at epoch 18 |

## Core Comparison

| Model | Parameters | Checkpoint MB | Clean Acc | Clean Macro-F1 | Strong stress Acc mean +/- std | Strong stress Macro-F1 mean +/- std | CPU ms/image | GPU ms/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 4,062,631 | 15.76 | 0.996675 | 0.996366 | 0.917155 +/- 0.001533 | 0.899001 +/- 0.000889 | 14.713 | 2.546 |
| v2 | 4,062,631 | 15.76 | 0.997229 | 0.996303 | 0.947585 +/- 0.000360 | 0.943970 +/- 0.000597 | 20.328 | 2.447 |
| v5 | 4,062,631 | 46.91 | 0.995249 | 0.991154 | 0.934627 +/- 0.000878 | 0.927409 +/- 0.001713 | 15.638 | 2.526 |
| v6 | 4,257,115 | 49.03 | 0.995566 | 0.992696 | 0.910847 +/- 0.000807 | 0.896003 +/- 0.002407 | 12.374 | 1.990 |

## Per-Corruption Macro-F1

| Corruption | v1 | v2 | v5 | v6 |
|---|---:|---:|---:|---:|
| rain | 0.982950 | 0.987007 | 0.979574 | 0.969834 |
| fog | 0.995592 | 0.996722 | 0.992375 | 0.988561 |
| snow | 0.962659 | 0.964856 | 0.956276 | 0.910420 |
| darkness | 0.995954 | 0.997118 | 0.991418 | 0.991734 |
| brightness | 0.991142 | 0.991783 | 0.991573 | 0.989417 |
| contrast | 0.992660 | 0.992318 | 0.988072 | 0.987161 |
| gaussian_blur | 0.970049 | 0.973161 | 0.958610 | 0.949158 |
| motion_blur | 0.982479 | 0.986737 | 0.978677 | 0.977587 |
| gaussian_noise | 0.897676 | 0.969873 | 0.959666 | 0.933368 |
| jpeg_compression | 0.925089 | 0.960300 | 0.956297 | 0.943068 |
| low_resolution | 0.977583 | 0.979563 | 0.966210 | 0.968345 |
| shadow | 0.996370 | 0.995967 | 0.990653 | 0.990908 |
| glare | 0.988648 | 0.988677 | 0.986550 | 0.984683 |

## Inference Verification

The real v2 checkpoint was loaded through `inference.py` on CPU and GPU. Path, PIL, numpy array, batch, top-k, confidence, and safe TTA all succeeded. TTA used brightness and resize/center-crop only; no horizontal flip is used.

| Device | First load + predict | PIL single with TTA | numpy single with TTA | batch of 3 with TTA |
|---|---:|---:|---:|---:|
| CPU | 148.530 ms | 58.672 ms | 58.636 ms | 188.764 ms |
| GPU | 2262.553 ms | 8.716 ms | 11.851 ms | 46.192 ms |

The GPU first-call number includes CUDA context and checkpoint loading. Steady-state single-image timing is better represented by the comparison table, which runs without TTA.

## Decision

- Default model: v2. It has the best Test clean accuracy and the best strong-weather score, including clear gains on gaussian noise and JPEG compression.
- Fast-inference option: v6, only when platform latency is the dominant constraint. It has the fastest measured CPU/GPU inference but lower clean and weather robustness.
- v5 is retained as a high-resolution experiment, not the default. Its validation score was perfect but its held-out Test clean and stress scores did not exceed v2.
- Do not train v3 this round: v2 has no observable clean-class regression that a balanced B0 model needs to repair, and v5 did not establish a resolution advantage.
- Do not train v4 this round: v2 is already strongest on the checked fog, darkness, low-resolution, blur, noise, and JPEG stress cases. There is no measured signal that more extreme augmentation will improve the selected metric.

## Competition-Day Rule

1. Run `python inspect_dataset.py --data-dir official_data --csv train.csv --path-col image --label-col label --output outputs/official_dataset_report.json`.
2. Inspect the report before setting class count or submission labels. Internal classifier indices are continuous, while `idx_to_class.json` preserves the original official labels.
3. If the official class set differs from GTSRB, fine-tune from v2 with `--auto-num-classes --init-checkpoint outputs/gtsrb_effb0_weather_v2_seed42/best.pt`; do not treat GTSRB `ClassId` as an official label.
4. Evaluate official validation data with clean and corruption modes, then select v2, a fine-tuned model, or v6 according to measured accuracy and platform latency.

## Remaining Work

No v3/v4 training was run by design. No final platform `main.py` was created because the contest input/output interface and label contract are not known yet.
