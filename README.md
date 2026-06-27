# ochwpro

使用 **Transformer** 编码器对笔画轨迹序列进行实时分类的轻量级手写输入引擎。

基于 CASIA-OLHWDB 数据集训练，支持 **7356** 类汉字单字识别。

![demo video](https://github.com/user-attachments/assets/1eafa234-4f1e-4f0b-8ccc-ccc93843d07e)

## 模型设计 — StrokeTransformer

### 整体架构

```
输入轨迹 (B, T, 5)                        输出 logits (B, 7356)
      │                                          ↑
      ▼                                          │
┌─────────────┐    ┌──────────────────┐    ┌──────────┐
│ Input Proj  │───►│  Transformer     │───►│ Classifier│
│ Linear(5→d)  │    │  Encoder ×3      │    │  Head     │
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
| dim_feedforward | 384 | FFN 隐藏层维度 |
| dropout | 0.2 | Dropout 比率 |
| max_seq_len | 512 | 最大轨迹序列长度 |
| **参数量** | **~2.5M** | 轻量，适合移动端部署 |

### 关键设计决策

1. **[CLS] Token 分类** — 参考 BERT，在序列前拼接一个可学习的 [CLS] token，取其在 Transformer 输出中的对应向量作为分类依据。

2. **Pre-Norm 架构** — `norm_first=True`，训练更稳定。

3. **可学习位置编码** — 让模型自适应学习笔画轨迹的顺序关系。

4. **2 层 MLP 分类头** — `Linear(d_model → d_model/2) → GELU → Dropout → Linear(d_model/2 → num_classes)`，增强分类能力。

### 数据增强

训练时对笔画轨迹做随机下采样（保留 15%~100% 的点），让模型适应不同采样密度（CASIA 数据 ~8-24 点/字 vs 鼠标输入 ~50+ 点/字）。

### 训练配置

| 配置 | 值 |
|------|:---:|
| 优化器 | AdamW (lr=1e-3, β=(0.9,0.95), wd=1e-4) |
| 学习率调度 | CosineAnnealingLR |
| 损失函数 | CrossEntropyLoss + label_smoothing=0.1 |
| Batch 大小 | 128 |
| Epoch | 30 |
| 早停 | patience=10 (monitor=val_acc) |
| 验证间隔 | 每 5000 步 |

### 当前结果

训练 30 个 epoch 后，在 CASIA-OLHWDB 测试集上达到 **88.86%** 的 Top-1 验证精度。

> 最佳权重: `checkpoints/ochwpro-epoch=28-val_acc=0.8886.ckpt`

## 使用方法

### 训练

```bash
# 训练
uv run python -m ochwpro.train

# 实时查看训练曲线（另开终端）
uv run tensorboard --logdir logs
# 浏览器打开 http://localhost:6006

# 快速验证（小数据集测试模型是否能学习）
uv run python -m ochwpro.quick_test
uv run python -m ochwpro.quick_test --augment         # 带增强
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
| `--fast-dev-run` | — | 快速开发模式（仅跑 1 batch） |

### 检查点目录结构

```
logs/ochwpro/version_X/
├── metrics.csv                  # 训练指标
├── events.out.tfevents.*        # TensorBoard 事件
└── checkpoints/
    ├── last.ckpt                         # 最后 epoch
    ├── ochwpro-epoch=28-val_acc=0.8886.ckpt  # Top-3
    └── ochwpro-best.ckpt                 # 最佳权重
```

### 导出 ONNX 移动端部署

```bash
# 导出 FP32 + FP16 + INT8 量化模型
uv run python -m ochwpro.export_onnx

# 指定模型和序列长度
uv run python -m ochwpro.export_onnx --model checkpoints/last.ckpt --seq-len 300
```

输出:
- `checkpoints/ochwpro.onnx` — FP32 (动态 batch)
- `checkpoints/ochwpro-fp16.onnx` — FP16 (精度无损)
- `checkpoints/ochwpro-int8.onnx` — INT8 量化 (体积缩小 4x)

### 启动手写输入演示

```bash
# 启动 (默认检测叠写引擎, GUI 上有"叠写"开关)
uv run python -m ochwpro.demo

# 指定模型
uv run python -m ochwpro.demo --model checkpoints/ochwpro-epoch=28-val_acc=0.8886.ckpt

# ONNX 推理
uv run python -m ochwpro.demo --model checkpoints/ochwpro-int8.onnx

# 回放日志
uv run python -m ochwpro.demo --replay logs/demo/stroke_20250615_123456.json
```

打开 Tkinter 窗口后，用鼠标或触摸屏书写汉字，模型实时返回 Top-10 候选字。

---

## 叠写模式 — DP 切分引擎

支持在同一区域连续书写多字，系统自动切分识别。

### 原理

不用额外训练任何模型，直接用单字分类模型做 DP 分割：

```
用户连续书写 (多字叠写)
       ↓
枚举所有可能的笔画分组 (j, i)
       ↓
每组用单字模型打分 → 置信度 P
       ↓
DP: dp[i] = max(dp[j] + log(P) + 切分惩罚)
       ↓
回溯最优切分路径 → 合并替代字 → 候选文本
```

- **DP 保证全局最优**：枚举所有切分方式，选总置信度最高的
- **切分惩罚** `-0.3`：每多切一个字扣分，防止过度切分
- **替代字组合**：每个切出来的字替换为 top-3 候选，生成更多候选文本
- **不需要训练数据**：直接用已有单字模型打分

## 项目结构

```
ochwpro/
├── model.py                # StrokeTransformer 模型定义
├── train.py                # 单字训练脚本 (Lightning)
├── demo.py                 # Tkinter 手写输入演示 (单字+叠写)
├── stroke_segmenter.py     # 叠写切分引擎 (纯 DP)
├── dataset.py              # 数据集加载 & 特征提取
├── char_index.py           # 字符↔标签索引映射
├── pot_parser.py           # .pot 二进制文件解析器
├── export_onnx.py          # ONNX 导出 + INT8 量化
└── quick_test.py           # 快速验证脚本
```

## 数据集

使用 CASIA-OLHWDB 离线手写汉字数据集：
- **300万+** 样本
- **1020** 位书写者
- **7356** 个汉字类别
- **POT 格式** 二进制笔画轨迹存储
