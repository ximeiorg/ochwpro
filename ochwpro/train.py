"""
训练脚本 — 基于 Transformer 的笔画序列手写汉字识别。
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from .char_index import CharIndex
from .dataset import StrokeSequenceDataset, collate_sequences


class LitStrokeClassifier(L.LightningModule):
    """Lightning 封装的 StrokeTransformer."""

    def __init__(
        self,
        num_classes: int,
        d_model: int = 192,
        nhead: int = 4,
        num_layers: int = 3,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        from .model import StrokeTransformer
        self.model = StrokeTransformer(
            num_classes=num_classes,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.lr = lr

    def forward(self, x, mask=None):
        return self.model(x, mask)

    def training_step(self, batch, batch_idx):
        x, mask, y = batch
        logits = self(x, mask)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask, y = batch
        logits = self(x, mask)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr,
            weight_decay=1e-4, betas=(0.9, 0.95),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs or 30)
        return [optimizer], [scheduler]


def build_training_data(
    data_root: str | Path,
    char_index_path: str | Path,
    min_freq: int = 10,
    max_samples_per_char: int = 500,
    max_seq_len: int = 512,
    batch_size: int = 128,
    val_split: float = 0.1,
):
    """构建训练/验证数据集."""

    # 构建或加载字符索引
    if Path(char_index_path).exists():
        print(f"加载已有字符索引: {char_index_path}")
        char_index = CharIndex.load(char_index_path)
    else:
        print(f"扫描数据集构建字符索引 (min_freq={min_freq})...")
        from .pot_parser import scan_dataset, iter_pot_file
        from collections import Counter
        pot_files = scan_dataset(data_root)
        counter: Counter = Counter()
        for pot_path in pot_files:
            for sample in iter_pot_file(str(pot_path)):
                counter[sample['tag_code']] += 1
        chars = [ch for ch, cnt in counter.most_common() if cnt >= min_freq]
        char_index = CharIndex(chars)
        char_index.save(char_index_path)
        print(f"字符索引已保存: {char_index.size} 个字符")

    # 构建数据集
    full_dataset = StrokeSequenceDataset(
        data_root=data_root,
        char_index=char_index,
        max_samples_per_char=max_samples_per_char,
        max_seq_len=max_seq_len,
    )

    # 分割训练/验证
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    import os
    num_workers = min(os.cpu_count() or 4, 12)  # 充分利用多进程加载
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_sequences,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_sequences,
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_loader, val_loader, char_index


def train(args=None):
    """主训练入口."""
    import argparse

    parser = argparse.ArgumentParser(description='训练手写汉字识别模型 (Transformer)')
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--char-index', type=str, default='data/char_index.json')
    parser.add_argument('--min-freq', type=int, default=10,
                        help='最低出现次数过滤字符')
    parser.add_argument('--max-samples', type=int, default=500,
                        help='每个字符最大样本数')
    parser.add_argument('--max-seq-len', type=int, default=512,
                        help='最大序列长度')
    parser.add_argument('--d-model', type=int, default=192,
                        help='Transformer 隐藏维度')
    parser.add_argument('--nhead', type=int, default=4,
                        help='注意力头数')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Transformer 层数')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--fast-dev-run', action='store_true')

    opts = parser.parse_args(args)

    # 构建数据
    train_loader, val_loader, char_index = build_training_data(
        data_root=opts.data_root,
        char_index_path=opts.char_index,
        min_freq=opts.min_freq,
        max_samples_per_char=opts.max_samples,
        max_seq_len=opts.max_seq_len,
        batch_size=opts.batch_size,
        val_split=opts.val_split,
    )

    # 模型
    model = LitStrokeClassifier(
        num_classes=char_index.size,
        d_model=opts.d_model,
        nhead=opts.nhead,
        num_layers=opts.num_layers,
        lr=opts.lr,
    )

    # 回调
    callbacks = [
        ModelCheckpoint(
            dirpath='checkpoints',
            filename='ochwpro-{epoch:02d}-{val_acc:.4f}',
            monitor='val_acc', mode='max',
            save_top_k=3, save_last=True,
        ),
        EarlyStopping(monitor='val_acc', patience=10, mode='max'),
    ]

    # 训练器
    trainer = L.Trainer(
        max_epochs=opts.max_epochs,
        accelerator='auto',
        devices=opts.devices,
        callbacks=callbacks,
        logger=CSVLogger('logs', name='ochwpro'),
        fast_dev_run=opts.fast_dev_run,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=opts.resume)

    # 保存最终模型
    model_path = 'checkpoints/ochwpro-final.pt'
    Path('checkpoints').mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.model.state_dict(),
        'char_index': char_index.chars,
        'd_model': opts.d_model,
        'nhead': opts.nhead,
        'num_layers': opts.num_layers,
        'max_seq_len': opts.max_seq_len,
    }, model_path)

    # 同时导出 TorchScript 用于移动端
    script_path = 'checkpoints/ochwpro-script.pt'
    try:
        dummy_x = torch.zeros(1, 10, 5)
        dummy_mask = torch.ones(1, 10, dtype=torch.bool)
        model.model.eval()
        traced = torch.jit.trace(model.model, (dummy_x, dummy_mask))
        traced.save(script_path)
        print(f"TorchScript 已导出: {script_path}")
    except Exception as e:
        print(f"TorchScript 导出跳过: {e}")

    print(f"\n训练完成! 模型已保存: {model_path}")
    print(f"字符集大小: {char_index.size}")
    print(f"模型参数: {sum(p.numel() for p in model.model.parameters()):,}")


if __name__ == '__main__':
    train()
