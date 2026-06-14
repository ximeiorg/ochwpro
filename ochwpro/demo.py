"""
触摸屏手写汉字输入演示 — 基于 StrokeTransformer 的实时轨迹识别。

输入法场景：用户书写笔画 → 实时提取轨迹序列 → Transformer 预测 Top-K 候选字。
可直接替换为手机输入法的识别引擎。

用法:
  python -m ochwpro.demo [--model checkpoints/ochwpro-final.pt]
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

from .dataset import strokes_to_sequence


# ── 日志目录 ──────────────────────────────────────────────
LOG_DIR = Path('logs/demo')
LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_model(model_path: str | Path):
    """加载训练好的 StrokeTransformer 和字符索引.

    支持两种格式:
      - ochwpro-final.pt: 训练脚本最终保存的格式
      - last.ckpt: Lightning ModelCheckpoint 保存的格式
    """
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

    # ── Lightning checkpoint (.ckpt) ──
    if 'hyper_parameters' in ckpt and 'state_dict' in ckpt:
        hp = ckpt['hyper_parameters']
        d_model = hp.get('d_model', 192)
        nhead = hp.get('nhead', 4)
        num_layers = hp.get('num_layers', 3)

        # 从 data/char_index.json 加载字符索引
        from .char_index import CharIndex
        char_index = CharIndex.load('data/char_index.json')
        chars = char_index.chars

        # 构建模型
        from .model import StrokeTransformer
        model = StrokeTransformer(
            num_classes=len(chars),
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
        )

        # 去掉 state_dict 中的 "model." 前缀
        state_dict = {k.removeprefix('model.'): v for k, v in ckpt['state_dict'].items()}
        model.load_state_dict(state_dict)
        model.eval()
        return model, chars

    # ── 最终模型格式 (.pt) ──
    chars = ckpt['char_index']
    d_model = ckpt.get('d_model', 192)
    nhead = ckpt.get('nhead', 4)
    num_layers = ckpt.get('num_layers', 3)
    max_seq_len = ckpt.get('max_seq_len', 512)

    from .model import StrokeTransformer
    model = StrokeTransformer(
        num_classes=len(chars),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    return model, chars


def is_cjk(ch: str) -> bool:
    """判断是否为 CJK 统一表意文字（汉字）。"""
    cp = ord(ch)
    return (
        (0x4E00 <= cp <= 0x9FFF) or      # 基本区
        (0x3400 <= cp <= 0x4DBF) or      # 扩展 A
        (0x20000 <= cp <= 0x2A6DF) or     # 扩展 B
        (0x2A700 <= cp <= 0x2B73F) or     # 扩展 C
        (0x2B740 <= cp <= 0x2B81F) or     # 扩展 D
        (0x2B820 <= cp <= 0x2CEAF) or     # 扩展 E
        (0xF900 <= cp <= 0xFAFF) or       # 兼容汉字
        (0x2F800 <= cp <= 0x2FA1F)        # 兼容扩展
    )


def predict(model, chars, strokes, top_k: int = 10):
    """直接用笔画轨迹预测汉字 — 无图像渲染.

    返回结果会过滤掉非汉字字符（如标点、字母、符号），
    确保候选列表只显示汉字。
    """
    # 笔画轨迹 -> 特征序列
    seq = strokes_to_sequence(strokes)  # (T, 5)
    seq_tensor = torch.from_numpy(seq).float().unsqueeze(0)  # (1, T, 5)
    mask = torch.ones(1, seq.shape[0], dtype=torch.bool)

    with torch.no_grad():
        logits = model(seq_tensor, mask)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        # 取更多候选，确保过滤后还有足够的汉字
        top_probs, top_indices = torch.topk(probs, min(top_k * 5, len(chars)))

    results = []
    for prob, idx in zip(top_probs, top_indices):
        ch = chars[idx]
        if is_cjk(ch):
            results.append((ch, prob.item()))
            if len(results) >= top_k:
                break

    # 如果过滤后不足 top_k，补上最高分的（防止全是非汉字的极端情况）
    if len(results) < top_k:
        top_probs, top_indices = torch.topk(probs, min(top_k, len(chars)))
        seen = {ch for ch, _ in results}
        for prob, idx in zip(top_probs, top_indices):
            ch = chars[idx]
            if ch not in seen:
                results.append((ch, prob.item()))
                seen.add(ch)
                if len(results) >= top_k:
                    break

    return results[:top_k]


class HandwritingApp:
    """Tkinter 触摸屏手写输入应用 — 直接轨迹序列识别."""

    def __init__(self, model, chars):
        import tkinter as tk
        self.root = tk.Tk()
        self.root.title("手写汉字输入 — StrokeTransformer ✍")
        self.root.geometry("620x520")

        self.model = model
        self.chars = chars
        self.top_k = 10

        # 当前笔画数据（原始屏幕坐标）
        self.strokes: list[list[tuple[int, int]]] = []
        self.current_stroke: list[tuple[int, int]] = []

        self._build_ui()

        # 鼠标事件（触摸屏在 Windows 下会自动转为鼠标事件）
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

    def _build_ui(self):
        import tkinter as tk
        from tkinter import ttk

        # 顶栏
        top = ttk.Frame(self.root)
        top.pack(fill='x', padx=5, pady=5)
        ttk.Label(top, text="✍ 手写输入（轨迹序列识别）",
                  font=('微软雅黑', 12, 'bold')).pack(side='left')
        ttk.Button(top, text="清空", command=self.clear).pack(side='right', padx=2)
        ttk.Button(top, text="撤销笔画", command=self.undo_stroke).pack(side='right', padx=2)

        # 画布
        cf = ttk.Frame(self.root)
        cf.pack(fill='both', expand=True, padx=5)
        self.canvas = tk.Canvas(cf, bg='white', cursor='crosshair', height=400)
        self.canvas.pack(fill='both', expand=True)

        # 主要候选（大字号显示）
        main_f = ttk.Frame(self.root)
        main_f.pack(fill='x', padx=5, pady=2)
        ttk.Label(main_f, text="➤", font=('微软雅黑', 14)).pack(side='left')
        self.main_result = tk.StringVar(value="等待书写...")
        ttk.Label(main_f, textvariable=self.main_result,
                  font=('微软雅黑', 28, 'bold'), foreground='#1a73e8').pack(side='left', padx=5)

        # Top-10 候选按钮
        btn_f = ttk.Frame(self.root)
        btn_f.pack(fill='x', padx=5, pady=2)
        self.candidate_buttons = []
        for i in range(self.top_k):
            btn = ttk.Button(
                btn_f, text='', width=5,
                command=lambda idx=i: self._select_candidate(idx),
            )
            btn.pack(side='left', padx=1)
            self.candidate_buttons.append(btn)

        # 状态栏
        self.status_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status_var,
                  font=('微软雅黑', 9), foreground='#666').pack(side='bottom', anchor='w', padx=5, pady=2)

    def _strokes_to_canvas_coords(self, stroke):
        """将画布坐标直接用于识别（无需转换）。"""
        return stroke  # 直接使用画布坐标

    def _on_mouse_down(self, event):
        self.current_stroke = [(event.x, event.y)]
        self._draw_point(event.x, event.y)

    def _on_mouse_move(self, event):
        if not self.current_stroke:
            return
        last = self.current_stroke[-1]
        self.current_stroke.append((event.x, event.y))
        self.canvas.create_line(last[0], last[1], event.x, event.y,
                                width=3, fill='black', capstyle='round', smooth=True)

    def _on_mouse_up(self, event):
        if self.current_stroke:
            self.current_stroke.append((event.x, event.y))
            self.strokes.append(self.current_stroke)
            self.current_stroke = []
            self._predict()

    def _on_touch_down(self, event):
        self.current_stroke = [(int(event.x), int(event.y))]

    def _on_touch_move(self, event):
        if not self.current_stroke:
            return
        last = self.current_stroke[-1]
        x, y = int(event.x), int(event.y)
        self.current_stroke.append((x, y))
        self.canvas.create_line(last[0], last[1], x, y,
                                width=3, fill='black', capstyle='round', smooth=True)

    def _on_touch_up(self, event):
        if self.current_stroke:
            self.strokes.append(self.current_stroke)
            self.current_stroke = []
            self._predict()

    def _predict(self):
        if not self.strokes:
            return
        results = predict(self.model, self.chars, self.strokes, self.top_k)
        self._candidates = results

        # 显示最佳候选
        best_ch, best_prob = results[0]
        self.main_result.set(f"{best_ch}")

        # 更新候选按钮
        for i, (ch, prob) in enumerate(results):
            self.candidate_buttons[i].config(text=f"{ch}\n{prob:.0%}")

        # 状态
        total_points = sum(len(s) for s in self.strokes)
        self.status_var.set(f"笔画: {len(self.strokes)} | 轨迹点: {total_points} | "
                            f"序列长度: {sum(len(s) for s in self.strokes)}")

        # 记录日志
        self._save_log(results)

    def _save_log(self, results):
        """将当前笔画和预测结果保存到日志文件."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        log_entry = {
            'timestamp': timestamp,
            'strokes': [[list(pt) for pt in stroke] for stroke in self.strokes],
            'predictions': [{'char': ch, 'prob': round(prob, 4)} for ch, prob in results],
        }
        log_path = LOG_DIR / f'stroke_{timestamp}.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)

    def _select_candidate(self, idx: int):
        if hasattr(self, '_candidates') and idx < len(self._candidates):
            ch, prob = self._candidates[idx]
            self.main_result.set(f"✔ {ch}")
            self.status_var.set(f"已选: {ch} (置信度: {prob:.1%})")

    def clear(self):
        self.canvas.delete('all')
        self.strokes = []
        self.current_stroke = []
        self.main_result.set("等待书写...")
        for btn in self.candidate_buttons:
            btn.config(text='')
        self._candidates = []
        self.status_var.set("")

    def undo_stroke(self):
        if self.strokes:
            self.strokes.pop()
            self.canvas.delete('all')
            for stroke in self.strokes:
                for i in range(len(stroke) - 1):
                    x1, y1 = stroke[i]
                    x2, y2 = stroke[i + 1]
                    self.canvas.create_line(x1, y1, x2, y2,
                                            width=3, fill='black', capstyle='round', smooth=True)
            if self.strokes:
                self._predict()
            else:
                self.main_result.set("等待书写...")
                for btn in self.candidate_buttons:
                    btn.config(text='')
                self.status_var.set("")

    def _draw_point(self, x, y):
        r = 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black', outline='')

    def run(self):
        self.root.mainloop()


def replay_log(log_path: str | Path, model_path: str | Path = 'checkpoints/last.ckpt'):
    """回放日志文件中的笔画，用模型重新预测。

    用法:
      uv run python -m ochwpro.demo --replay logs/demo/stroke_20250615_123456.json
    """
    log_path = Path(log_path)
    if not log_path.exists():
        print(f"错误: 日志文件不存在: {log_path}")
        return

    with open(log_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"回放: {log_path.name}")
    print(f"时间: {data['timestamp']}")
    print(f"笔画: {len(data['strokes'])} 笔, "
          f"轨迹点: {sum(len(s) for s in data['strokes'])}")

    # 加载模型
    model, chars = load_model(model_path)

    # 预测
    strokes = [[tuple(pt) for pt in stroke] for stroke in data['strokes']]
    results = predict(model, chars, strokes, top_k=10)

    print('\n预测结果:')
    for i, (ch, prob) in enumerate(results):
        mark = ' ★' if data['predictions'] and ch == data['predictions'][0]['char'] else ''
        print(f'  {i+1}. {ch} ({prob:.2%}){mark}')

    # 对比原结果
    print('\n原始结果:')
    for i, p in enumerate(data['predictions'][:10]):
        print(f'  {i+1}. {p["char"]} ({p["prob"]:.2%})')


def main():
    parser = argparse.ArgumentParser(description='手写汉字输入演示 (StrokeTransformer)')
    parser.add_argument('--model', type=str, default='checkpoints/last.ckpt')
    parser.add_argument('--replay', type=str, default=None,
                        help='回放日志文件中的笔画进行预测')
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        return

    # ── 回放模式 ──
    if args.replay:
        replay_log(args.replay, model_path)
        return

    # ── GUI 模式 ──
    print(f"加载模型: {model_path}")
    model, chars = load_model(model_path)
    print(f"字符集: {len(chars)} 个 | "
          f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"日志目录: {LOG_DIR}/")

    app = HandwritingApp(model, chars)
    app.run()


if __name__ == '__main__':
    main()
