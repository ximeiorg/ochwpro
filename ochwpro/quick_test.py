"""
快速验证脚本 — 用少量数据测试模型能否正常学习。

用法:
  uv run python -m ochwpro.quick_test                    # 测试新模型 (无增强)
  uv run python -m ochwpro.quick_test --augment           # 测试新模型 (有增强)
  uv run python -m ochwpro.quick_test --model-size base   # base 模型
  uv run python -m ochwpro.quick_test --no-augment --no-model  # 仅测试数据加载
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .char_index import CharIndex
from .dataset import (
    StrokeSequenceDataset,
    AugmentedSubset,
    collate_sequences,
)
from .model import StrokeTransformer

# ── 配置 ──────────────────────────────────────────────────
SAMPLE_SIZE = 5000    # 总样本数（训练+验证）
VAL_RATIO = 0.2       # 验证集比例
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3


def build_small_dataset(
    data_root: str = 'data',
    char_index_path: str = 'data/char_index.json',
    max_seq_len: int = 512,
    sample_size: int = SAMPLE_SIZE,
    augment: bool = False,
):
    """加载少量数据进行快速验证."""
    char_index = CharIndex.load(char_index_path)
    full = StrokeSequenceDataset(
        data_root=data_root,
        char_index=char_index,
        max_samples_per_char=None,
        max_seq_len=max_seq_len,
        augment=False,
    )

    # 取前 sample_size 条
    n = min(sample_size, len(full))
    indices = list(range(n))
    val_n = int(n * VAL_RATIO)
    train_idx = indices[:-val_n]
    val_idx = indices[-val_n:]

    train_subset = Subset(full, train_idx)
    val_subset = Subset(full, val_idx)

    if augment:
        train_ds = AugmentedSubset(train_subset)
    else:
        train_ds = train_subset
    val_ds = val_subset

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=collate_sequences,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_sequences,
    )
    return train_loader, val_loader, char_index.size


def test_augmentation():
    """测试增强后的数据形状是否正常."""
    from .dataset import _read_one_sample_strokes, strokes_to_sequence
    from .pot_parser import scan_dataset
    pots = scan_dataset('data')
    if not pots:
        print("⚠️  没有找到 POT 文件")
        return
    strokes = _read_one_sample_strokes(str(pots[0]), 4)  # 跳过 sample_size(2)+tag(4)
    if not strokes:
        print("⚠️  无法读取样本")
        return
    print(f"原始笔画: {len(strokes)} 笔, {sum(len(s) for s in strokes)} 点")
    for _ in range(3):
        seq = strokes_to_sequence(strokes, augment=True)
        print(f"  增强后: shape={seq.shape}, "
              f"x=[{seq[:,0].min():.3f},{seq[:,0].max():.3f}], "
              f"dx=[{seq[:,2].min():.1f},{seq[:,2].max():.1f}]")
    print("✅ 数据增强形状正常")


def train_quick(model_size: str = 'small', augment: bool = False):
    """快速训练验证."""
    sizes = {
        'small': dict(d_model=192, nhead=4, num_layers=3, dim_feedforward=384),
        'base':  dict(d_model=256, nhead=8, num_layers=4, dim_feedforward=512),
    }
    cfg = sizes[model_size]

    print(f"\n{'='*50}")
    print(f"模型: {model_size} | 增强: {'✅' if augment else '❌'}")
    print(f"参数: {cfg}")
    print(f"{'='*50}\n")

    # 数据
    train_loader, val_loader, num_classes = build_small_dataset(augment=augment)
    print(f"数据: {len(train_loader.dataset)} 训练, {len(val_loader.dataset)} 验证, "
          f"{num_classes} 类\n")

    # 模型
    model = StrokeTransformer(num_classes=num_classes, **cfg)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4, betas=(0.9, 0.95))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}\n")

    for epoch in range(EPOCHS):
        # ── 训练 ──
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        n_batches = 0
        t0 = time.time()

        for x, mask, y in train_loader:
            opt.zero_grad()
            logits = model(x, mask)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            acc = (logits.argmax(dim=1) == y).float().mean()
            train_loss += loss.item()
            train_acc += acc.item()
            n_batches += 1

        train_loss /= n_batches
        train_acc /= n_batches

        # ── 验证 ──
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_val = 0

        with torch.no_grad():
            for x, mask, y in val_loader:
                logits = model(x, mask)
                loss = loss_fn(logits, y)
                acc = (logits.argmax(dim=1) == y).float().mean()
                val_loss += loss.item()
                val_acc += acc.item()
                n_val += 1

        val_loss /= n_val
        val_acc /= n_val
        elapsed = time.time() - t0

        print(f"Epoch {epoch:2d} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"{elapsed:.0f}s")

    print(f"\n{'='*50}")
    if val_acc > 0.01:
        print(f"✅ 模型成功学习! 验证精度 {val_acc:.2%}")
    else:
        print(f"❌ 模型未学习! 验证精度 {val_acc:.2%} (随机 ≈ 1/{num_classes} ≈ {1/num_classes:.4%})")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description='快速验证模型训练')
    parser.add_argument('--model-size', default='small', choices=['small', 'base'])
    parser.add_argument('--augment', action='store_true', help='启用数据增强')
    parser.add_argument('--no-augment', action='store_true', help='禁用数据增强')
    parser.add_argument('--test-data', action='store_true', help='仅测试数据加载')
    args = parser.parse_args()

    if args.test_data:
        test_augmentation()
        return

    augment = args.augment
    if args.no_augment:
        augment = False

    train_quick(model_size=args.model_size, augment=augment)


if __name__ == '__main__':
    main()
