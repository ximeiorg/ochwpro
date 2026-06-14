"""对比 CASIA 数据集中的"十"和用户的输入"""
import json
import torch
import numpy as np
from ochwpro.dataset import strokes_to_sequence, _read_one_sample_strokes, StrokeSequenceDataset
from ochwpro.char_index import CharIndex
from ochwpro.pot_parser import scan_dataset, decode_tag_code
from ochwpro.demo import load_model, is_cjk

# 加载字符索引
char_index = CharIndex.load('data/char_index.json')
target_idx = char_index.char_to_idx('十')
print(f'"十" index: {target_idx}')

# 从数据集中找到"十"的样本（不限每字符数量）
dataset = StrokeSequenceDataset('data', char_index, max_samples_per_char=None)
print(f'数据集大小: {len(dataset)}')

# 找到所有"十"的样本
shi_indices = []
for i, (path, offset, label) in enumerate(dataset.index):
    if label == target_idx:
        shi_indices.append((path, offset))
        if len(shi_indices) >= 20:
            break

print(f'找到 {len(shi_indices)} 个"十"样本\n')

# 统计笔画数分布
stroke_counts = {}
for idx, (path, offset) in enumerate(shi_indices):
    strokes = _read_one_sample_strokes(path, offset)
    n = len(strokes)
    stroke_counts[n] = stroke_counts.get(n, 0) + 1

print(f'"十"的笔画数分布: {dict(sorted(stroke_counts.items()))}')

for idx, (path, offset) in enumerate(shi_indices[:5]):
    # 读取原始坐标
    strokes = _read_one_sample_strokes(path, offset)
    seq = strokes_to_sequence(strokes)
    
    total_points = sum(len(s) for s in strokes)
    print(f'--- CASIA "十" #{idx+1} ---')
    print(f'  笔画: {len(strokes)} 笔, 轨迹点: {total_points}')
    print(f'  序列形状: {seq.shape}')
    
    # 坐标范围
    all_x = [p[0] for s in strokes for p in s]
    all_y = [p[1] for s in strokes for p in s]
    print(f'  原始 X: [{min(all_x)}, {max(all_x)}] (范围 {max(all_x)-min(all_x)})')
    print(f'  原始 Y: [{min(all_y)}, {max(all_y)}] (范围 {max(all_y)-min(all_y)})')
    print(f'  X/Y 范围比: {(max(all_x)-min(all_x))/(max(all_y)-min(all_y)):.2f}')
    print(f'  前3行特征:\n{seq[:3]}')
    print()

# 对比用户的输入
print('=== 用户画的"十" ===')
with open('logs/demo/stroke_20260615_040357_417050.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
user_strokes = [[tuple(pt) for pt in stroke] for stroke in data['strokes']]
user_seq = strokes_to_sequence(user_strokes)
all_x = [p[0] for s in user_strokes for p in s]
all_y = [p[1] for s in user_strokes for p in s]
print(f'  笔画: {len(user_strokes)} 笔, 轨迹点: {sum(len(s) for s in user_strokes)}')
print(f'  序列形状: {user_seq.shape}')
print(f'  原始 X: [{min(all_x)}, {max(all_x)}] (范围 {max(all_x)-min(all_x)})')
print(f'  原始 Y: [{min(all_y)}, {max(all_y)}] (范围 {max(all_y)-min(all_y)})')
print(f'  X/Y 范围比: {(max(all_x)-min(all_x))/(max(all_y)-min(all_y)):.2f}')
print(f'  前3行特征:\n{user_seq[:3]}')
