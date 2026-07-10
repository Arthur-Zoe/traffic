# RAICOM 2026 交通标志分类工程

这是 2026 睿抗国赛交通标志分类项目的本地训练、评估和比赛适配工程。当前默认数据是 GTSRB 43 类，比赛当天拿到官方数据后，不应把 GTSRB 的 `ClassId` 直接当成官方标签；先检查官方数据结构和标签，再决定是否复用旧权重微调。

## 环境

- Ubuntu 22.04
- Conda 环境：`raicom_weather`
- Python 3.10
- PyTorch + CUDA 已在用户本机验证可用
- GPU：RTX 4050 Laptop，6GB 显存

安装普通依赖：

```bash
pip install -r requirements.txt
```

开发测试依赖只有 pytest：

```bash
pip install -r requirements-dev.txt
```

不要为了 Codex 沙箱或容器降级/重装 torch。若 Codex 普通命令看不到 GPU，只影响沙箱执行；用户本机终端可以直接用 CUDA。

## 数据结构

当前 GTSRB 数据目录：

```text
data/gtsrb/
├── Train.csv      # 默认路径列 Path，标签列 ClassId
├── Test.csv
├── Train/
└── Test/
```

通用数据模块 `datasets.py` 支持：

- CSV 数据集：自定义 `--path-col`、`--label-col`
- 文件夹按类别组织的数据集
- 无标签图片目录
- 字符串类别稳定映射
- 导出 `class_to_idx.json`、`idx_to_class.json`

## 已有模型

不要删除、移动或覆盖这些 checkpoint：

```text
outputs/gtsrb_effb0_weather_seed42/best.pt
outputs/gtsrb_effb0_weather_v2_seed42/best.pt
```

已验证结果：

| 模型 | Clean Acc | Clean Macro-F1 | 强压力 Acc 均值 | 强压力 Macro-F1 均值 |
|---|---:|---:|---:|---:|
| v1 简单天气增强 | 0.996675 | 0.996366 | 约 0.9216 | 约 0.9064 |
| v2 强天气增强 | 0.997229 | 0.996303 | 约 0.9479 | 约 0.9454 |

当前默认推荐 v2，但比赛当天必须重新用官方数据评估。

## 检查数据集

```bash
python inspect_dataset.py \
  --data-dir data/gtsrb \
  --csv Train.csv \
  --path-col Path \
  --label-col ClassId \
  --output outputs/gtsrb_dataset_report.json
```

官方数据示例：

```bash
python inspect_dataset.py \
  --data-dir official_data \
  --csv train.csv \
  --path-col image \
  --label-col label \
  --output outputs/official_dataset_report.json
```

报告包含图片数、可读数量、损坏/缺失图片、类别数、类别分布、尺寸分布、通道模式、重复文件、同图不同标签冲突和推荐配置。

## 训练 Preset

`train_gtsrb.py` 保持旧命令兼容，同时支持通用 CSV、类别数、resume、init-checkpoint 和 preset。

可用 preset：

- `v1_clean`：EfficientNet-B0，224，轻量天气增强，`weather_prob=0.35`
- `v2_strong`：EfficientNet-B0，224，强天气增强，`weather_prob=0.45`，`max_ops=3`
- `v3_balanced`：EfficientNet-B0，224，`weather_prob=0.40`，`max_ops=2`
- `v4_extreme`：EfficientNet-B0，224，`weather_prob=0.55`，`max_ops=3`
- `v5_img256`：EfficientNet-B0，256，默认 batch size 24
- `v6_mobile`：MobileNetV3-Large，224，用于推理速度受限平台

训练示例：

```bash
bash scripts/train_v3_balanced.sh
bash scripts/train_v4_extreme.sh
bash scripts/train_v5_img256.sh
bash scripts/train_v6_mobile.sh
```

直接命令示例：

```bash
python train_gtsrb.py \
  --data-dir data/gtsrb \
  --preset v3_balanced \
  --epochs 30 \
  --batch-size 32 \
  --workers 4 \
  --pretrained \
  --output-dir outputs/gtsrb_v3_balanced_seed42
```

显式命令行参数会覆盖 preset。脚本默认拒绝覆盖已有 `best.pt`，如确实要覆盖：

```bash
OVERWRITE=1 bash scripts/train_v3_balanced.sh
```

## 断点与迁移微调

完整恢复训练，包括 optimizer、scheduler、scaler 和 epoch：

```bash
python train_gtsrb.py \
  --data-dir data/gtsrb \
  --resume outputs/gtsrb_v3_balanced_seed42/last.pt \
  --output-dir outputs/gtsrb_v3_balanced_seed42
```

比赛当天如果官方类别数变化，使用 `--init-checkpoint` 只加载兼容权重，分类头维度不一致会自动跳过：

```bash
python train_gtsrb.py \
  --data-dir official_data \
  --train-csv train.csv \
  --path-col image \
  --label-col label \
  --auto-num-classes \
  --preset v3_balanced \
  --init-checkpoint outputs/gtsrb_effb0_weather_v2_seed42/best.pt \
  --output-dir outputs/official_v3_init_from_gtsrb
```

这样不会把 GTSRB 的 43 类标签强行套到官方类别上。

## Clean 评估

```bash
python evaluate.py \
  --data-dir data/gtsrb \
  --csv Test.csv \
  --checkpoint outputs/gtsrb_effb0_weather_v2_seed42/best.pt \
  --mode clean
```

输出包括 Accuracy、Macro-F1、每类 precision/recall/F1/support、混淆矩阵 CSV、预测明细 CSV、错误最多类别和统一 JSON。

## 恶劣天气评估

随机压力，多 seed 重复并输出 mean/std：

```bash
python evaluate.py \
  --data-dir data/gtsrb \
  --csv Test.csv \
  --checkpoint outputs/gtsrb_effb0_weather_v2_seed42/best.pt \
  --mode random-stress \
  --severity strong \
  --repeats 3 \
  --seed 42
```

逐 corruption 独立评估：

```bash
python evaluate.py \
  --data-dir data/gtsrb \
  --csv Test.csv \
  --checkpoint outputs/gtsrb_effb0_weather_v2_seed42/best.pt \
  --mode per-corruption \
  --severity strong
```

支持 rain、fog、snow、darkness、brightness、contrast、gaussian_blur、motion_blur、gaussian_noise、jpeg_compression、low_resolution、shadow、glare。

## 多模型对比

```bash
python compare_models.py \
  --data-dir data/gtsrb \
  --csv Test.csv \
  --checkpoints \
    outputs/gtsrb_effb0_weather_seed42/best.pt \
    outputs/gtsrb_effb0_weather_v2_seed42/best.pt \
  --output outputs/model_comparison.json
```

输出包括模型名、输入尺寸、参数量、checkpoint 大小、clean 指标、各 corruption 指标、strong random-stress mean/std、CPU/GPU 单张推理时间和推荐用途。

## 推理模块

`inference.py` 不依赖 pandas，导入时不加载模型，第一次调用时懒加载。支持路径、PIL.Image、numpy.ndarray、单张和 batch。

命令行：

```bash
python inference.py \
  --checkpoint outputs/gtsrb_effb0_weather_v2_seed42/best.pt \
  --image data/gtsrb/Test/00000.png \
  --device cpu \
  --topk 3
```

代码调用：

```python
from inference import TrafficSignClassifier

clf = TrafficSignClassifier("outputs/gtsrb_effb0_weather_v2_seed42/best.pt", device="cpu")
result = clf.predict_one("data/gtsrb/Test/00000.png")
```

TTA 只使用安全的中心裁剪/轻微亮度变化，不使用水平翻转。

## Smoke Test

```bash
bash scripts/smoke_test.sh
```

等价核心命令：

```bash
python train_gtsrb.py \
  --data-dir data/gtsrb \
  --preset v3_balanced \
  --epochs 1 \
  --batch-size 8 \
  --workers 0 \
  --max-train-samples 256 \
  --max-val-samples 128 \
  --output-dir outputs/smoke_v3
```

## 比赛当天流程

1. 不要先写死 `main.py` 接口，先看官方说明、样例提交和数据结构。
2. 用 `inspect_dataset.py` 检查官方训练/测试数据。
3. 确认官方类别数量、标签含义、CSV 字段、图片是裁剪标志还是完整道路图。
4. 如果官方类别和 GTSRB 不一致，用 `--auto-num-classes --init-checkpoint ...` 微调。
5. 用 `evaluate.py` 分别跑 clean、random-stress、per-corruption。
6. 用 `compare_models.py` 对比 v2、v3/v4/v5/v6 或官方微调模型。
7. 最后只写一层很薄的平台 `main.py`，调用 `inference.py`。

## Git 与大文件

`.gitignore` 已忽略：

```text
data/
outputs/
*.pt
__pycache__/
.pytest_cache/
```

不要提交数据集和大模型文件。
