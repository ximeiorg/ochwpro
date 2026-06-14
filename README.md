# ochwpro

使用 **Transformer** 编码器对笔画轨迹序列进行实时分类的轻量级手写输入引擎。

## 模型设计 — StrokeTransformer

### 整体架构

```
输入轨迹 (B, T, 5)                        输出 logits (B, 7356)
      │                                          ↑
      ▼                                          │
┌─────────────┐    ┌──────────────────┐    ┌──────────┐
│ Input Proj  │───►│  Transformer     │───►│ Classifier│
│ Linear(5→192)│    │  Encoder ×3      │    │  Head     │
└─────────────┘    └──────────────────┘    └──────────┘
      │                    ↑
      ▼                    │
┌─────────────┐    ┌──────────────┐
│ [CLS] Token │───►│  Positional  │
│  concat     │    │  Encoding    │
└─────────────┘    └──────────────┘
```

### 输入特征

每条笔画轨迹被编码为 5 维向量序列：

| 维度 | 含义 | 说明 |
|:---:|------|------|
| 0 | **x_norm** | 归一化 X 坐标 [0, 1] |
| 1 | **y_norm** | 归一化 Y 坐标 [0, 1] |
| 2 | **dx** | X 方向增量（当前点 - 前一点） |
| 3 | **dy** | Y 方向增量（当前点 - 前一点） |
| 4 | **pen_down** | 落笔标志（1=笔画中, 0=抬笔/笔画结束） |

### 模型参数

| 参数 | 值 | 说明 |
|------|:---:|------|
| d_model | 192 | Transformer 隐藏维度 |
| nhead | 4 | 注意力头数 |
| num_layers | 3 | Transformer 编码器层数 |
| dim_feedforward | 384 | FFN 隐藏层维度 (2×d_model) |
| dropout | 0.2 | Dropout 比率 |
| max_seq_len | 512 | 最大轨迹序列长度 |
| **参数量** | **~2.5M** | 轻量，适合移动端部署 |

### 关键设计决策

1. **[CLS] Token 分类** — 参考 BERT，在序列前拼接一个可学习的 [CLS] token，取其在 Transformer 输出中的对应向量作为分类依据，让模型自动学习关注最重要的轨迹信息。

2. **Pre-Norm 架构** — 使用 `norm_first=True`（LayerNorm 在 Attention/FFN 之前），相比 Post-Norm 训练更稳定，对学习率变化更鲁棒。

3. **可学习位置编码** — 参数化的位置编码向量（而非三角函数固定编码），让模型能自适应学习笔画轨迹的顺序关系。

4. **轻量设计** — 仅 ~2.5M 参数，可导出 TorchScript 用于手机端推理，适合替代传统输入法中的手写识别引擎。

### 训练配置

| 配置 | 值 |
|------|:---:|
| 优化器 | AdamW (lr=1e-3, β=(0.9,0.95), wd=1e-4) |
| 学习率调度 | CosineAnnealingLR |
| 损失函数 | CrossEntropyLoss + label_smoothing=0.1 |
| Batch 大小 | 128 |
| Epoch | 30 |
| 早停 | patience=10 (monitor=val_acc) |
| 数据增强 | 坐标归一化、序列截断/填充 |


## 使用方法

### 训练

```bash
# 使用默认参数训练
uv run python -m ochwpro.train

# 自定义参数
uv run python -m ochwpro.train \
    --data-root data \
    --batch-size 128 \
    --max-epochs 30 \
    --d-model 192 \
    --lr 1e-3
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|:------:|------|
| `--data-root` | `data` | 数据集根目录 |
| `--min-freq` | `10` | 字符最低出现次数 |
| `--max-samples` | `500` | 每字符最大样本数 |
| `--max-seq-len` | `512` | 最大序列长度 |
| `--d-model` | `192` | Transformer 隐藏维度 |
| `--nhead` | `4` | 注意力头数 |
| `--num-layers` | `3` | Transformer 层数 |
| `--batch-size` | `128` | 批次大小 |
| `--lr` | `1e-3` | 学习率 |
| `--max-epochs` | `30` | 最大训练轮数 |
| `--resume` | — | 从 checkpoint 恢复训练 |

### 启动手写输入演示

```bash
uv run python -m ochwpro.demo --model checkpoints/ochwpro-final.pt
```

打开 Tkinter 窗口后，用鼠标或触摸屏书写汉字，模型实时返回 Top-10 候选字。
