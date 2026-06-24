"""
导出 ONNX 并量化为 INT8。

用法:
  uv run python -m ochwpro.export_onnx                                    # 默认导出最佳模型
  uv run python -m ochwpro.export_onnx --model checkpoints/last.ckpt      # 指定 checkpoint
  uv run python -m ochwpro.export_onnx --seq-len 300                      # 自定义序列长度
  uv run python -m ochwpro.export_onnx --no-quantize                      # 仅 FP32，不量化
"""

import argparse
from pathlib import Path

import torch

from .model import StrokeTransformer


def export(model_path: str, output_prefix: str = 'checkpoints/ochwpro',
           fixed_len: int = 200, quantize: bool = True):
    """导出 ONNX 并可选 INT8 量化.

    Args:
        model_path: Lightning checkpoint(.ckpt) 或 PyTorch(.pt) 路径
        output_prefix: 输出路径前缀 (默认 checkpoints/ochwpro)
        fixed_len: ONNX 固定序列长度 (默认 200)
        quantize: 是否 INT8 量化 (默认 True)
    """
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'hyper_parameters' in ckpt:
        hp = ckpt['hyper_parameters']
        sd = {k.removeprefix('model.'): v for k, v in ckpt['state_dict'].items()}
    else:
        hp = ckpt
        sd = ckpt['model_state_dict']

    model = StrokeTransformer(
        num_classes=7356,
        d_model=hp.get('d_model', 192),
        nhead=hp.get('nhead', 4),
        num_layers=hp.get('num_layers', 3),
        dim_feedforward=hp.get('dim_feedforward', 384),
    )
    model.load_state_dict(sd)
    model.eval()

    # 导出 FP32 ONNX
    output_prefix = Path(output_prefix)
    onnx_path = output_prefix.with_suffix('.onnx') if output_prefix.suffix else \
        output_prefix.parent / f'{output_prefix.stem}.onnx'

    dummy_x = torch.randn(1, fixed_len, 5)
    dummy_mask = torch.ones(1, fixed_len, dtype=torch.bool)

    print(f"导出 ONNX: 序列长度={fixed_len}, 参数量={sum(p.numel() for p in model.parameters()):,}")
    torch.onnx.export(
        model, (dummy_x, dummy_mask),
        str(onnx_path),
        input_names=['input', 'mask'],
        output_names=['logits'],
        dynamic_axes={
            'input': {0: 'batch'},
            'mask': {0: 'batch'},
            'logits': {0: 'batch'},
        },
        opset_version=17,
    )
    print(f"  FP32: {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # INT8 量化
    if quantize:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = onnx_path.with_stem(onnx_path.stem + '-int8')
        quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
        print(f"  INT8: {int8_path} ({int8_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # FP16 导出 (精度无损)
    model_fp16 = model.half()
    fp16_path = onnx_path.with_stem(onnx_path.stem + '-fp16')
    dummy_x_fp16 = torch.randn(1, fixed_len, 5).half()
    torch.onnx.export(
        model_fp16, (dummy_x_fp16, dummy_mask),
        str(fp16_path),
        input_names=['input', 'mask'],
        output_names=['logits'],
        dynamic_axes={
            'input': {0: 'batch'},
            'mask': {0: 'batch'},
            'logits': {0: 'batch'},
        },
        opset_version=17,
    )
    print(f"  FP16: {fp16_path} ({fp16_path.stat().st_size / 1024 / 1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description='导出 ONNX（FP32/FP16/INT8）')
    parser.add_argument('--model', type=str, default='checkpoints/ochwpro-epoch=29-val_acc=0.8886.ckpt',
                        help='模型路径')
    parser.add_argument('--output', type=str, default='checkpoints/ochwpro.onnx',
                        help='输出路径')
    parser.add_argument('--seq-len', type=int, default=200,
                        help='固定序列长度 (默认 200)')
    parser.add_argument('--no-quantize', action='store_true',
                        help='跳过 INT8 量化 (FP16 仍会导出)')
    args = parser.parse_args()

    export(args.model, args.output, args.seq_len, not args.no_quantize)


if __name__ == '__main__':
    main()
