/// StrokeTransformer — Rust + Candle 实现
///
/// 对应 Python ochwpro/model.py
/// 权重从 safetensors 加载 (PyTorch 命名格式)

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{self as nn, Module, VarBuilder};

#[derive(Clone)]
pub struct TransformerConfig {
    pub num_classes: usize,
    pub d_model: usize,
    pub nhead: usize,
    pub num_layers: usize,
    pub dim_feedforward: usize,
    pub dropout: f32,
    pub max_seq_len: usize,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            num_classes: 7356,
            d_model: 192,
            nhead: 4,
            num_layers: 3,
            dim_feedforward: 384,
            dropout: 0.2,
            max_seq_len: 512,
        }
    }
}

/// 加载 safetensors 权重并创建模型
pub fn load_model(
    weights: &[u8],
    config: &TransformerConfig,
    device: &Device,
) -> Result<StrokeTransformer> {
    let vb = VarBuilder::from_buffered_safetensors(weights.to_vec(), DType::F32, device)?;
    StrokeTransformer::load(vb, config)
}

// ── 位置编码 ──

pub(crate) struct PositionalEncoding {
    pub(crate) pe: Tensor,
}

impl PositionalEncoding {
    fn load(vb: VarBuilder) -> Result<Self> {
        let pe = vb.get((1, 513, 192), "pe")?;
        Ok(Self { pe })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let n = x.dims()[1];
        x.add(&self.pe.narrow(1, 0, n)?)
    }
}

// ── StrokeTransformer ──

pub struct StrokeTransformer {
    pub(crate) input_proj: nn::Linear,
    pub(crate) pos_encoder: PositionalEncoding,
    pub(crate) cls_token: Tensor,
    pub(crate) layers: Vec<TransformerLayer>,
    norm: nn::LayerNorm,
    classifier_0: nn::Dropout,
    classifier_1: nn::Linear,
    classifier_3: nn::Dropout,
    classifier_4: nn::Linear,
    config: TransformerConfig,
}

impl StrokeTransformer {
    pub fn load(vb: VarBuilder, config: &TransformerConfig) -> Result<Self> {
        let d_model = config.d_model;
        let d_half = d_model / 2;

        let input_proj = nn::linear(5, d_model, vb.pp("input_proj"))?;
        let pos_encoder = PositionalEncoding::load(vb.pp("pos_encoder"))?;
        let cls_token = vb.get((1, 1, d_model), "cls_token")?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer = TransformerLayer::load(vb.pp(&format!("transformer.layers.{i}")), config)?;
            layers.push(layer);
        }

        let norm = nn::layer_norm(d_model, 1e-5, vb.pp("norm"))?;
        let classifier_0 = nn::Dropout::new(config.dropout);
        let classifier_1 = nn::linear(d_model, d_half, vb.pp("classifier.1"))?;
        let classifier_3 = nn::Dropout::new(config.dropout * 0.5);
        let classifier_4 = nn::linear(d_half, config.num_classes, vb.pp("classifier.4"))?;

        Ok(Self {
            input_proj,
            pos_encoder,
            cls_token,
            layers,
            norm,
            classifier_0,
            classifier_1,
            classifier_3,
            classifier_4,
            config: config.clone(),
        })
    }

    pub fn forward(&self, x: &Tensor, seq_len: usize) -> Result<Tensor> {
        let device = x.device();
        let b = x.dims()[0];
        let d_model = self.config.d_model;
        let mut h = self.input_proj.forward(x)?;

        // [CLS] token
        let cls = self.cls_token.broadcast_as((b, 1, d_model))?;
        h = Tensor::cat(&[&cls, &h], 1)?;

        // 位置编码
        h = self.pos_encoder.forward(&h)?;

        // padding mask: (B, T+1), True=mask
        let src_mask = Tensor::zeros((b, seq_len + 1), DType::F32, device)?;

        // Transformer layers
        for layer in &self.layers {
            h = layer.forward(&h, &src_mask)?;
        }

        // [CLS] 输出
        let cls_out = h.narrow(1, 0, 1)?.squeeze(1)?;

        // Norm + 分类头 (分类头内部用 F32 避免精度损失)
        let h = self.norm.forward(&cls_out)?;
        let h = self.classifier_0.forward(&h, false)?;
        let h = self.classifier_1.forward(&h)?;
        let h = h.gelu_erf()?;
        let h = self.classifier_3.forward(&h, false)?;
        let h = self.classifier_4.forward(&h)?;

        Ok(h)
    }
}

// ── TransformerLayer (Pre-Norm) ──

pub(crate) struct TransformerLayer {
    self_attn: FusedSelfAttention,
    linear1: nn::Linear,
    linear2: nn::Linear,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    dropout: nn::Dropout,
}

impl TransformerLayer {
    fn load(vb: VarBuilder, config: &TransformerConfig) -> Result<Self> {
        let self_attn = FusedSelfAttention::load(vb.pp("self_attn"), config)?;
        let linear1 = nn::linear(config.d_model, config.dim_feedforward, vb.pp("linear1"))?;
        let linear2 = nn::linear(config.dim_feedforward, config.d_model, vb.pp("linear2"))?;
        let norm1 = nn::layer_norm(config.d_model, 1e-5, vb.pp("norm1"))?;
        let norm2 = nn::layer_norm(config.d_model, 1e-5, vb.pp("norm2"))?;
        let dropout = nn::Dropout::new(config.dropout);
        Ok(Self { self_attn, linear1, linear2, norm1, norm2, dropout })
    }

    pub(crate) fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // Pre-Norm: x -> norm1 -> attn -> + x -> norm2 -> ffn -> + x
        let residual = x;
        let h = self.norm1.forward(x)?;
        let h = self.self_attn.forward(&h, mask)?;
        let h = (h + residual)?;

        let residual = &h;
        let h = self.norm2.forward(&h)?;
        let h = self.linear1.forward(&h)?;
        let h = h.gelu_erf()?;
        let h = self.dropout.forward(&h, false)?;
        let h = self.linear2.forward(&h)?;
        h + residual
    }
}

// ── 融合 QKV 自注意力 ──

struct FusedSelfAttention {
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    out_proj: nn::Linear,
    nhead: usize,
    _d_model: usize,
}

impl FusedSelfAttention {
    fn load(vb: VarBuilder, config: &TransformerConfig) -> Result<Self> {
        let in_proj_weight = vb.get((3 * config.d_model, config.d_model), "in_proj_weight")?;
        let in_proj_bias = vb.get(3 * config.d_model, "in_proj_bias")?;
        let out_proj = nn::linear(config.d_model, config.d_model, vb.pp("out_proj"))?;
        Ok(Self { in_proj_weight, in_proj_bias, out_proj, nhead: config.nhead, _d_model: config.d_model })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let head_dim = d / self.nhead;

        // QKV 投影 (融合权重)
        let w = self.in_proj_weight.t()?;
        let x_2d = x.reshape((b * t, d))?;
        let qkv_2d = x_2d.matmul(&w)?;
        let qkv = qkv_2d.reshape((b, t, 3 * d))?;
        let qkv = qkv.broadcast_add(&self.in_proj_bias)?;

        // 拆分 Q, K, V
        let q = qkv.narrow(2, 0, d)?;
        let k = qkv.narrow(2, d, d)?;
        let v = qkv.narrow(2, 2 * d, d)?;

        // 多头: (B, nhead, T, head_dim)
        let q = q.reshape((b, t, self.nhead, head_dim))?.permute((0, 2, 1, 3))?;
        let k = k.reshape((b, t, self.nhead, head_dim))?.permute((0, 2, 1, 3))?;
        let v = v.reshape((b, t, self.nhead, head_dim))?.permute((0, 2, 1, 3))?;

        // Scaled Dot-Product Attention
        let scale = (head_dim as f64).sqrt();
        let attn = (q.matmul(&k.t()?)? / scale)?;

        // mask: (B, T+1) -> (B, 1, 1, T+1), mask=True → -inf
        let mask_f = mask.unsqueeze(1)?.unsqueeze(1)?.to_dtype(attn.dtype())?;
        let attn = attn.broadcast_add(&(mask_f * (-1e9_f64))?)?;

        let attn = candle_nn::ops::softmax(&attn, 3)?;
        let h = attn.matmul(&v)?;

        // 合并: (B, T, d_model)
        let h = h.permute((0, 2, 1, 3))?.reshape((b, t, d))?;

        // 输出投影
        self.out_proj.forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_default_config() {
        let config = TransformerConfig::default();
        assert_eq!(config.num_classes, 7356);
        assert_eq!(config.d_model, 192);
        assert_eq!(config.nhead, 4);
        assert_eq!(config.num_layers, 3);
        assert_eq!(config.dim_feedforward, 384);
        assert_eq!(config.max_seq_len, 512);
    }

    #[test]
    fn test_load_with_zeros() {
        let config = TransformerConfig::default();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = StrokeTransformer::load(vb, &config).unwrap();

        let input = Tensor::zeros((1, 10, 5), DType::F32, &device).unwrap();
        let output = model.forward(&input, 10).unwrap();
        assert_eq!(output.dims(), &[1, config.num_classes]);
    }

    #[test]
    fn test_forward_various_seq_lens() {
        let config = TransformerConfig::default();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = StrokeTransformer::load(vb, &config).unwrap();

        for &seq_len in &[1, 5, 50, 200] {
            let input = Tensor::zeros((1, seq_len, 5), DType::F32, &device).unwrap();
            let output = model.forward(&input, seq_len).unwrap();
            assert_eq!(output.dims(), &[1, config.num_classes],
                "seq_len={} should produce correct output shape", seq_len);
        }
    }

    #[test]
    fn test_softmax_output_sum() {
        let config = TransformerConfig::default();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = StrokeTransformer::load(vb, &config).unwrap();

        let input = Tensor::zeros((1, 10, 5), DType::F32, &device).unwrap();
        let output = model.forward(&input, 10).unwrap();
        let sm = candle_nn::ops::softmax(&output, 1).unwrap();
        let sum_val: f32 = sm.sum(1).unwrap().to_vec1::<f32>().unwrap()[0];
        assert!((sum_val - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_load_model_from_empty_weights() {
        let config = TransformerConfig::default();
        let device = Device::Cpu;
        // VarBuilder::zeros should produce a working model
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = StrokeTransformer::load(vb, &config).unwrap();
        assert_eq!(model.layers.len(), config.num_layers);
    }
}
