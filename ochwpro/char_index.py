"""
字符索引映射 — 将汉字映射到整数标签，管理字符集的构建。
"""

import json
from pathlib import Path
from collections import Counter
from .pot_parser import scan_dataset, iter_pot_file


class CharIndex:
    """字符 <=> 标签索引 双向映射。"""

    def __init__(self, chars: list[str] | None = None):
        self._char_to_idx: dict[str, int] = {}
        self._idx_to_char: dict[int, str] = {}
        if chars:
            for i, ch in enumerate(chars):
                self._char_to_idx[ch] = i
                self._idx_to_char[i] = ch

    @property
    def size(self) -> int:
        return len(self._char_to_idx)

    @property
    def chars(self) -> list[str]:
        return [self._idx_to_char[i] for i in range(self.size)]

    def char_to_idx(self, ch: str) -> int:
        return self._char_to_idx[ch]

    def idx_to_char(self, idx: int) -> str:
        return self._idx_to_char[idx]

    def add(self, ch: str) -> int:
        if ch not in self._char_to_idx:
            idx = len(self._char_to_idx)
            self._char_to_idx[ch] = idx
            self._idx_to_char[idx] = ch
        return self._char_to_idx[ch]

    def contains(self, ch: str) -> bool:
        return ch in self._char_to_idx

    def save(self, path: str | Path):
        """保存到 JSON 文件."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'chars': self.chars}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> 'CharIndex':
        """从 JSON 文件加载."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(data['chars'])

    @classmethod
    def build_from_datasets(cls, data_root: str | Path, min_freq: int = 5) -> 'CharIndex':
        """扫描数据集构建字符索引，过滤低频字符."""
        pot_files = scan_dataset(data_root)
        counter: Counter = Counter()
        for pot_path in pot_files:
            for sample in iter_pot_file(str(pot_path)):
                counter[sample['tag_code']] += 1

        chars = [ch for ch, cnt in counter.most_common() if cnt >= min_freq]
        print(f"扫描 {len(pot_files)} 个文件, {len(counter)} 不同字符, "
              f"过滤后 {len(chars)} 字符 (min_freq={min_freq})")
        return cls(chars)
