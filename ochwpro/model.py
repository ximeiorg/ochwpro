"""
StrokeTransformer — 基于 Transformer 编码器的手写笔画序列分类模型。

专为手机端部署设计：
  - 轻量：3层 Transformer，d_model=192，4头注意力 (~2.5M 参数)
  - 输入：变长轨迹序列 (x, y, dx, dy, pen_down) + attention mask
  - 输出：汉字分类 logits

可导出为 TorchScript / ONNX 用于移动端推理。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """可学习位置编码."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self.pe[:, :x.size(1), :]


class StrokeTransformer(nn.Module):
    """笔画轨迹 Transformer 分类器.

    输入: (B, T, 5) — [x, y, dx, dy, pen_down]
    辅助: mask (B, T) — 有效位置 True/False
    输出: (B, num_classes) — logits
    """

    def __init__(
        self,
        num_classes: int,
        d_model: int = 192,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 384,
        dropout: float = 0.2,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model

        # 输入投影: 5 维特征 -> d_model
        self.input_proj = nn.Linear(5, d_model)

        # 位置编码 (+1 给 [CLS] token)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len + 1)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-Norm: 更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # LayerNorm + 分类头
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """前向传播.

        Args:
            x: (B, T, 5) 轨迹特征
            mask: (B, T) True=有效位置, 或 None

        Returns:
            logits: (B, num_classes)
        """
        B, T, _ = x.shape

        # 输入投影
        h = self.input_proj(x)  # (B, T, d_model)

        # 拼接 [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        h = torch.cat([cls_tokens, h], dim=1)  # (B, T+1, d_model)

        # 位置编码
        h = self.pos_encoder(h)

        # Transformer mask: 在最前面加一列给 CLS
        if mask is not None:
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
            tf_mask = torch.cat([cls_mask, mask], dim=1)
        else:
            tf_mask = None

        h = self.transformer(h, src_key_padding_mask=~tf_mask if tf_mask is not None else None)

        # 取 [CLS] 对应的输出
        cls_out = h[:, 0, :]
        cls_out = self.norm(cls_out)

        logits = self.classifier(cls_out)
        return logits

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, top_k: int = 10
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """推理接口: 返回 (top_k_indices, top_k_probs)."""
        logits = self(x, mask)
        probs = F.softmax(logits, dim=-1)
        return torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)
