"""
PyTorch Dataset — 将 POT 笔画序列转为 Transformer 输入特征。

核心设计：不将所有样本加载到内存，而是构建轻量级索引，
__getitem__ 时按需从磁盘读取单个样本并即时转换。

内存占用：索引 ≈ 路径(8B) + 偏移(8B) + 标签(4B) × 样本数
          147 万样本 ≈ 约 30MB 索引，而非数 GB 笔画数据。
"""

import pickle
import struct
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from .pot_parser import scan_dataset, decode_tag_code
from .char_index import CharIndex

# ── 索引缓存 ──────────────────────────────────────────────
INDEX_CACHE_DIR = Path('data/_index_cache')
INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _scan_one_pot_for_index(
    pot_path: str | Path,
    char_index: CharIndex,
    count_per_char: dict[str, int],
    max_samples_per_char: int | None,
) -> list[tuple[str, int, int]]:
    """扫描单个 .pot 文件，返回索引条目列表.

    每条: (pot_path_str, byte_offset, label_idx)
    不提取笔画数据，只记录偏移量。
    """
    index: list[tuple[str, int, int]] = []
    pot_str = str(pot_path)
    offset = 0

    with open(pot_path, 'rb') as fp:
        while True:
            data = fp.read(2)
            if len(data) != 2:
                break
            sample_size = struct.unpack('<H', data)[0]
            if sample_size == 0:
                break

            tag_data = fp.read(4)
            tag_code_bytes = bytes([tag_data[1], tag_data[0]])
            ch = decode_tag_code(tag_code_bytes)

            # 跳过笔画数据到下一个样本
            # 样本结构: sample_size(2) + tag(4) + stroke_num(2) + strokes + end_tag(4)
            fp.seek(sample_size - 6, 1)

            if ch is None or not char_index.contains(ch):
                offset += sample_size
                continue

            if max_samples_per_char is not None:
                cnt = count_per_char.get(ch, 0)
                if cnt >= max_samples_per_char:
                    offset += sample_size
                    continue
                count_per_char[ch] = cnt + 1

            label_idx = char_index.char_to_idx(ch)
            index.append((pot_str, offset, label_idx))
            offset += sample_size

    return index


def _read_one_sample_strokes(pot_path: str, byte_offset: int) -> list[list[tuple[int, int]]]:
    """从 .pot 文件指定偏移读取一个样本的笔画数据."""
    with open(pot_path, 'rb') as fp:
        fp.seek(byte_offset)
        sample_size = struct.unpack('<H', fp.read(2))[0]
        fp.read(4)  # tag code, skip
        stroke_number = struct.unpack('<H', fp.read(2))[0]
        strokes = []
        for _ in range(stroke_number):
            points: list[tuple[int, int]] = []
            while True:
                x, y = struct.unpack('<hh', fp.read(4))
                if (x, y) == (-1, 0):
                    break
                points.append((x, y))
            strokes.append(points)
    return strokes


def strokes_to_sequence(strokes: list[list[tuple[int, int]]]) -> np.ndarray:
    """将笔画数据转为 (seq_len, 5) 特征序列: [x, y, dx, dy, pen_down]."""
    all_points: list[tuple[int, int]] = []
    pen_down_flags: list[int] = []

    for stroke in strokes:
        for x, y in stroke:
            all_points.append((x, y))
            pen_down_flags.append(1)
        if pen_down_flags:
            pen_down_flags[-1] = 0

    if not all_points:
        return np.zeros((1, 5), dtype=np.float32)

    xs = np.array([p[0] for p in all_points], dtype=np.float32)
    ys = np.array([p[1] for p in all_points], dtype=np.float32)

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    range_x = max(max_x - min_x, 1.0)
    range_y = max(max_y - min_y, 1.0)

    seq = np.column_stack([
        (xs - min_x) / range_x,                    # x_norm
        (ys - min_y) / range_y,                    # y_norm
        np.diff(xs, prepend=xs[0:1]) / range_x,    # dx_norm (归一化!)
        np.diff(ys, prepend=ys[0:1]) / range_y,    # dy_norm (归一化!)
        np.array(pen_down_flags, dtype=np.float32),  # pen_down
    ])
    return seq.astype(np.float32)


def collate_sequences(batch: list[tuple[np.ndarray, int]]):
    """自定义 collate：padding 变长序列 + attention mask.

    Returns:
        padded: (batch, max_len, 5)
        mask: (batch, max_len) — True=有效位置
        labels: (batch,)
    """
    sequences, labels = zip(*batch)
    seqs = [torch.from_numpy(s) for s in sequences]
    labels = torch.tensor(labels, dtype=torch.long)

    padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)

    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = padded.size(1)
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    return padded, mask, labels


def _index_cache_path(data_root: str | Path, char_index: CharIndex,
                      max_samples_per_char: int | None) -> Path:
    """生成索引缓存文件路径."""
    import hashlib
    root_hash = hashlib.md5(str(data_root).encode()).hexdigest()[:8]
    samples_tag = f'_cap{max_samples_per_char}' if max_samples_per_char else ''
    return INDEX_CACHE_DIR / f'index_{root_hash}_{char_index.size}c{samples_tag}.pkl'


class StrokeSequenceDataset(Dataset):
    """POT 手写汉字笔画序列数据集 — 懒加载，按需读取.

    __init__ 仅构建轻量索引（不加载笔画数据），
    __getitem__ 时从磁盘读取单个样本并即时转换。
    """

    def __init__(
        self,
        data_root: str | Path,
        char_index: CharIndex,
        max_samples_per_char: int | None = None,
        max_seq_len: int = 512,
        rebuild_cache: bool = False,
    ):
        self.char_index = char_index
        self.max_seq_len = max_seq_len

        # 尝试加载缓存索引
        cache_path = _index_cache_path(data_root, char_index, max_samples_per_char)
        if not rebuild_cache and cache_path.exists():
            print(f"加载索引缓存: {cache_path.name}")
            with open(cache_path, 'rb') as f:
                self.index: list[tuple[str, int, int]] = pickle.load(f)
            print(f"StrokeSequenceDataset: {len(self.index)} 样本 (来自缓存)")
            return

        # 扫描构建索引（仅首次）
        print(f"扫描 POT 文件构建索引...")
        pot_files = scan_dataset(data_root)
        self.index: list[tuple[str, int, int]] = []
        count_per_char: dict[str, int] = {}

        for pot_path in pot_files:
            entries = _scan_one_pot_for_index(
                pot_path, char_index, count_per_char, max_samples_per_char,
            )
            self.index.extend(entries)

        # 保存缓存
        with open(cache_path, 'wb') as f:
            pickle.dump(self.index, f)
        print(f"StrokeSequenceDataset: {len(self.index)} 样本, "
              f"索引已缓存 -> {cache_path.name}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        pot_path, byte_offset, label_idx = self.index[idx]
        strokes = _read_one_sample_strokes(pot_path, byte_offset)
        seq = strokes_to_sequence(strokes)
        if len(seq) > self.max_seq_len:
            seq = seq[:self.max_seq_len]
            seq[-1, 4] = 0  # 截断点设为抬笔
        return seq, label_idx


# 别名
ChineseHandwritingDataset = StrokeSequenceDataset
