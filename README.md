# RAICOM 2026 国赛：恶劣天气交通标识分类 baseline

这是一版可直接替换你原天气分类项目的通用图像分类框架。核心目标：

1. 先用国外交通标识数据集（如 GTSRB）预训练。
2. 训练时加入雨、雪、雾、低照度、过曝、模糊、JPEG 压缩等增强。
3. 比赛当天拿到官方 25 类数据后，替换分类头并快速微调。

代码避免使用 `from __future__ import annotations`，减少旧平台 Python 版本报错风险。

---

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

建议使用你原来的 conda 环境，例如：

```bash
conda activate raicom_weather
pip install -r requirements.txt
```

---

## 2. 数据格式

优先支持 ImageFolder 格式：

```text
data/gtsrb_folders/train/
├── 0/
├── 1/
├── 2/
└── ...
```

官方国赛数据如果也是文件夹分类，直接按这个格式放即可：

```text
data/official/train/
├── class_0/
├── class_1/
└── ... class_24/
```

如果官方是 CSV 标签，也支持：

```text
data/official/train_images/
data/official/train.csv    # 默认列名 image,label
```

---

## 3. GTSRB CSV 转文件夹格式

很多 GTSRB 下载版本包含 `Train.csv`，列名通常是 `Path` 和 `ClassId`。转换命令：

```bash
python tools/convert_gtsrb_csv_to_folders.py \
  --csv data/gtsrb/Train.csv \
  --src-root data/gtsrb \
  --out-dir data/gtsrb_folders/train
```

如果软链接失败，改用复制：

```bash
python tools/convert_gtsrb_csv_to_folders.py \
  --csv data/gtsrb/Train.csv \
  --src-root data/gtsrb \
  --out-dir data/gtsrb_folders/train \
  --copy
```

检查类别分布：

```bash
python tools/check_dataset.py --data-dir data/gtsrb_folders/train
```

---

## 4. 第一版训练：EfficientNet-B0

```bash
python train.py \
  --data-dir data/gtsrb_folders/train \
  --model efficientnet_b0 \
  --img-size 224 \
  --epochs 30 \
  --batch-size 32 \
  --workers 4 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --label-smoothing 0.05 \
  --class-weight sqrt_inverse \
  --strong-weather \
  --amp \
  --output-dir outputs/gtsrb_effb0_weather_seed42
```

输出：

```text
outputs/gtsrb_effb0_weather_seed42/
├── best.pt
├── classes.txt
├── args.json
├── history.json
├── val_report.txt
└── confusion_matrix.csv
```

---

## 5. 冲分训练：ConvNeXt-Tiny

显存够时用这个：

```bash
python train.py \
  --data-dir data/gtsrb_folders/train \
  --model convnext_tiny \
  --img-size 320 \
  --epochs 40 \
  --batch-size 16 \
  --workers 4 \
  --lr 5e-5 \
  --weight-decay 1e-4 \
  --label-smoothing 0.05 \
  --class-weight sqrt_inverse \
  --strong-weather \
  --amp \
  --output-dir outputs/gtsrb_convnext_tiny_320_weather_seed42
```

RTX 4050 6GB 如果 OOM，先把 batch size 改成 8：

```bash
--batch-size 8
```

---

## 6. 比赛当天：用 GTSRB 权重微调官方 25 类

如果官方数据是文件夹格式：

```bash
python train.py \
  --data-dir data/official/train \
  --model convnext_tiny \
  --img-size 320 \
  --epochs 40 \
  --batch-size 16 \
  --workers 4 \
  --lr 5e-5 \
  --weight-decay 1e-4 \
  --label-smoothing 0.05 \
  --class-weight sqrt_inverse \
  --strong-weather \
  --amp \
  --init-checkpoint outputs/gtsrb_convnext_tiny_320_weather_seed42/best.pt \
  --freeze-backbone-epochs 3 \
  --output-dir outputs/official_convnext_tiny_320_seed42
```

`--init-checkpoint` 会自动跳过形状不匹配的分类头，只加载 backbone 中能对上的权重。

如果官方数据是 CSV：

```bash
python train.py \
  --csv data/official/train.csv \
  --image-dir data/official/train_images \
  --image-col image \
  --label-col label \
  --model convnext_tiny \
  --img-size 320 \
  --epochs 40 \
  --batch-size 16 \
  --lr 5e-5 \
  --class-weight sqrt_inverse \
  --strong-weather \
  --amp \
  --init-checkpoint outputs/gtsrb_convnext_tiny_320_weather_seed42/best.pt \
  --freeze-backbone-epochs 3 \
  --output-dir outputs/official_convnext_tiny_320_seed42
```

---

## 7. 推理生成提交文件

```bash
python infer.py \
  --checkpoint outputs/official_convnext_tiny_320_seed42/best.pt \
  --test-dir data/official/test \
  --output outputs/submission.csv \
  --batch-size 64 \
  --filename-col image \
  --label-col label
```

如果官方要求类别名而不是数字，可以加：

```bash
--save-class-name
```

最终提交列名以官方样例为准。常见是：

```text
image,label
xxx.jpg,3
```

---

## 8. 重要策略说明

- 不默认使用水平翻转，因为交通标识有方向语义。
- 评分是 macro F1，不是 accuracy，所以启用了 `sqrt_inverse` 类别权重。
- `--strong-weather` 是为国赛题面准备的恶劣天气增强。
- 95 分左右需要后续做多 seed、多模型融合、TTA；这版先保证可跑通并形成高质量 baseline。
