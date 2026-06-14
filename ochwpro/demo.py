"""
触摸屏手写汉字输入演示 — 基于 StrokeTransformer 的实时轨迹识别。

输入法场景：用户书写笔画 → 实时提取轨迹序列 → Transformer 预测 Top-K 候选字。
可直接替换为手机输入法的识别引擎。

用法:
  python -m ochwpro.demo [--model checkpoints/ochwpro-final.pt]
"""

import argparse
from pathlib import Path

import torch
import numpy as np

from .dataset import strokes_to_sequence


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


def predict(model, chars, strokes, top_k: int = 10):
    """直接用笔画轨迹预测汉字 — 无图像渲染."""
    # 笔画轨迹 -> 特征序列
    seq = strokes_to_sequence(strokes)  # (T, 5)
    seq_tensor = torch.from_numpy(seq).float().unsqueeze(0)  # (1, T, 5)
    mask = torch.ones(1, seq.shape[0], dtype=torch.bool)

    with torch.no_grad():
        logits = model(seq_tensor, mask)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        top_probs, top_indices = torch.topk(probs, min(top_k, len(chars)))

    results = [(chars[idx], prob.item()) for prob, idx in zip(top_probs, top_indices)]
    return results


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


def main():
    parser = argparse.ArgumentParser(description='手写汉字输入演示 (StrokeTransformer)')
    parser.add_argument('--model', type=str, default='checkpoints/ochwpro-final.pt')
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行: python -m ochwpro.train")
        return

    print(f"加载模型: {model_path}")
    model, chars = load_model(model_path)
    print(f"字符集: {len(chars)} 个 | "
          f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    app = HandwritingApp(model, chars)
    app.run()


if __name__ == '__main__':
    main()
