"""
CASIA Online Handwriting Database (.pot) 文件解析器

.pot 文件格式:
  - Sample size: unsigned short (2B) — 一个样本的总字节数
  - Tag code (GB): DWORD (4B) — GB2312/GBK 编码
  - Stroke number: unsigned short (2B) — 笔画数
  - Strokes (concatenated):
      - Coordinates (x, y): short (2B+2B)
      - Stroke end: (-1, 0) as signed short (2B+2B)
  - Character end: (-1, -1) as signed short (2B+2B)
"""

import struct
import zipfile
from pathlib import Path
from typing import Iterator


def decode_tag_code(tag_code_bytes: bytes) -> str | None:
    """从两个字节的 GB 标签码解码为中文字符."""
    tag_code_bytes = bytes(b for b in tag_code_bytes if b != 0x00)
    if len(tag_code_bytes) == 0:
        return None
    try:
        return tag_code_bytes.decode('gb2312')
    except UnicodeDecodeError:
        try:
            return tag_code_bytes.decode('gbk')
        except UnicodeDecodeError:
            return None


def read_pot_file(pot_filepath: str | Path) -> tuple[list[dict], dict[str, list]]:
    """读取单个 .pot 文件，返回 (samples, char_list).

    samples: [{'sample_size', 'tag_code', 'stroke_number', 'strokes'}, ...]
    char_list: {char: strokes} 每个字符只保留最后出现的笔画数据
    """
    samples: list[dict] = []
    char_list: dict[str, list] = {}

    with open(pot_filepath, 'rb') as fp:
        while True:
            # 样本大小
            data = fp.read(2)
            if len(data) != 2:
                break
            sample_size = struct.unpack('<H', data)[0]
            if sample_size == 0:
                break

            # 标签码 (4B, 低2位为GB编码)
            tag_data = fp.read(4)
            if len(tag_data) != 4:
                raise ValueError("Unexpected end of file while reading tag code")
            tag_code_bytes = bytes([tag_data[1], tag_data[0]])
            tag_code = decode_tag_code(tag_code_bytes)
            if tag_code is None:
                # 跳过整个样本
                fp.seek(sample_size - 6, 1)
                continue

            # 笔画数
            stroke_number = struct.unpack('<H', fp.read(2))[0]

            # 读取笔画
            strokes: list[list[tuple[int, int]]] = []
            for _ in range(stroke_number):
                points: list[tuple[int, int]] = []
                while True:
                    x, y = struct.unpack('<hh', fp.read(4))
                    if (x, y) == (-1, 0):
                        break
                    points.append((x, y))
                strokes.append(points)

            # 字符结束标志
            end_x, end_y = struct.unpack('<hh', fp.read(4))
            if (end_x, end_y) != (-1, -1):
                print(f"Warning: Invalid end tag ({end_x},{end_y}) at char {tag_code}")

            samples.append({
                'sample_size': sample_size,
                'tag_code': tag_code,
                'stroke_number': stroke_number,
                'strokes': strokes,
            })
            char_list[tag_code] = strokes

            # 跳到下一个样本
            bytes_consumed = 2 + 4 + 2 + sum(len(s) * 4 + 4 for s in strokes) + 4
            remaining = sample_size - bytes_consumed
            if remaining > 0:
                fp.seek(remaining, 1)

    return samples, char_list


def read_pot_zip_file(zip_path: str | Path, pot_name: str) -> tuple[list[dict], dict[str, list]]:
    """从 ZIP 包中直接读取 .pot 文件（不解压到磁盘）."""
    samples: list[dict] = []
    char_list: dict[str, list] = {}

    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open(pot_name) as fp:
            while True:
                data = fp.read(2)
                if len(data) != 2:
                    break
                sample_size = struct.unpack('<H', data)[0]
                if sample_size == 0:
                    break

                tag_data = fp.read(4)
                tag_code_bytes = bytes([tag_data[1], tag_data[0]])
                tag_code = decode_tag_code(tag_code_bytes)
                if tag_code is None:
                    fp.seek(sample_size - 6, 1)
                    continue

                stroke_number = struct.unpack('<H', fp.read(2))[0]
                strokes = []
                for _ in range(stroke_number):
                    points = []
                    while True:
                        x, y = struct.unpack('<hh', fp.read(4))
                        if (x, y) == (-1, 0):
                            break
                        points.append((x, y))
                    strokes.append(points)

                end_x, end_y = struct.unpack('<hh', fp.read(4))
                if (end_x, end_y) != (-1, -1):
                    print(f"Warning: Invalid end tag ({end_x},{end_y}) at char {tag_code}")

                samples.append({
                    'sample_size': sample_size,
                    'tag_code': tag_code,
                    'stroke_number': stroke_number,
                    'strokes': strokes,
                })
                char_list[tag_code] = strokes

                bytes_consumed = 2 + 4 + 2 + sum(len(s) * 4 + 4 for s in strokes) + 4
                remaining = sample_size - bytes_consumed
                if remaining > 0:
                    fp.seek(remaining, 1)

    return samples, char_list


def iter_pot_file(pot_filepath: str | Path) -> Iterator[dict]:
    """惰性迭代 .pot 文件中的每个样本，避免一次性加载全部到内存."""
    with open(pot_filepath, 'rb') as fp:
        while True:
            data = fp.read(2)
            if len(data) != 2:
                break
            sample_size = struct.unpack('<H', data)[0]
            if sample_size == 0:
                break

            tag_data = fp.read(4)
            tag_code_bytes = bytes([tag_data[1], tag_data[0]])
            tag_code = decode_tag_code(tag_code_bytes)
            if tag_code is None:
                fp.seek(sample_size - 6, 1)
                continue

            stroke_number = struct.unpack('<H', fp.read(2))[0]
            strokes = []
            for _ in range(stroke_number):
                points = []
                while True:
                    x, y = struct.unpack('<hh', fp.read(4))
                    if (x, y) == (-1, 0):
                        break
                    points.append((x, y))
                strokes.append(points)

            end_x, end_y = struct.unpack('<hh', fp.read(4))
            bytes_consumed = 2 + 4 + 2 + sum(len(s) * 4 + 4 for s in strokes) + 4
            remaining = sample_size - bytes_consumed
            if remaining > 0:
                fp.seek(remaining, 1)

            yield {
                'sample_size': sample_size,
                'tag_code': tag_code,
                'stroke_number': stroke_number,
                'strokes': strokes,
            }


def scan_dataset(data_root: str | Path) -> list[Path]:
    """扫描 data_root 下所有 .pot 文件，返回排序后的路径列表."""
    return sorted(Path(data_root).rglob('*.pot'))


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        samples, chars = read_pot_file(sys.argv[1])
        print(f"共 {len(samples)} 个样本，{len(chars)} 个不同字符")
        for ch, strokes in list(chars.items())[:5]:
            print(f"  '{ch}': {len(strokes)} 笔")
