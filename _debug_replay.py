"""深度分析：回放日志 + 查看原始模型输出"""
import json
import torch
import numpy as np
from ochwpro.dataset import strokes_to_sequence
from ochwpro.demo import load_model, is_cjk

# 加载日志
with open('logs/demo/stroke_20260615_040357_417050.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

strokes = [[tuple(pt) for pt in stroke] for stroke in data['strokes']]
print(f'笔画: {len(strokes)} 笔, 轨迹点: {sum(len(s) for s in strokes)}')

# 特征序列
seq = strokes_to_sequence(strokes)
print(f'序列形状: {seq.shape}')
print(f'序列前5行:\n{seq[:5]}')

# 加载模型
model, chars = load_model('checkpoints/last.ckpt')

# 原始模型输出（不经过过滤）
seq_tensor = torch.from_numpy(seq).float().unsqueeze(0)
mask = torch.ones(1, seq.shape[0], dtype=torch.bool)

with torch.no_grad():
    logits = model(seq_tensor, mask)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    top100_probs, top100_indices = torch.topk(probs, 100)

print('\n=== 原始 Top-30（不过滤）===')
for i, (prob, idx) in enumerate(zip(top100_probs[:30], top100_indices[:30])):
    ch = chars[idx]
    cjk = '汉' if is_cjk(ch) else '非'
    print(f'  {i+1:2d}. [{idx:4d}] {ch} ({prob:.4%}) [{cjk}]')

# 检查 "十" 的排名
target_idx = chars.index('十')
target_prob = probs[target_idx].item()
# 计算排名
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
rank = (sorted_indices == target_idx).nonzero(as_tuple=True)[0].item()
print(f'\n=== "十" (index={target_idx}) ===')
print(f'  概率: {target_prob:.4%}')
print(f'  在全部 7356 个中排名第 {rank+1}')
print(f'  在前 100 中: {"是" if rank < 100 else "否"}')
