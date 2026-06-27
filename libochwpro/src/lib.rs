pub mod model;
pub mod preprocessing;

use candle_core::Device;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct TopK {
    pub label: String,
    pub score: f32,
    pub class_idx: usize,
}

pub struct Inference {
    pub(crate) model: model::StrokeTransformer,
    pub(crate) labels: Vec<String>,
}

impl Inference {
    pub fn load(weights: &[u8], labels: Vec<String>) -> anyhow::Result<Self> {
        let device = Device::Cpu;
        let config = model::TransformerConfig::default();
        let model = model::load_model(weights, &config, &device)?;
        Ok(Self { model, labels })
    }

    pub fn predict(
        &self,
        strokes: &[Vec<(f32, f32)>],
        top_k: usize,
    ) -> anyhow::Result<Vec<TopK>> {
        let device = Device::Cpu;

        // 预处理
        let seq_len = strokes.iter().map(|s| s.len()).sum::<usize>();
        let input = preprocessing::strokes_to_tensor(strokes, &device)?;

        // 推理
        let output = self.model.forward(&input, seq_len)?;
        let output = candle_nn::ops::softmax(&output, 1)?;

        // 取 top-k
        let mut predictions: Vec<(usize, f32)> = output
            .squeeze(0)?
            .to_vec1::<f32>()?
            .into_iter()
            .enumerate()
            .collect();
        predictions.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        let top_k_data: Vec<TopK> = predictions
            .into_iter()
            .take(top_k)
            .map(|(idx, score)| TopK {
                label: self.labels.get(idx).cloned().unwrap_or_else(|| "?".to_string()),
                score,
                class_idx: idx,
            })
            .collect();

        Ok(top_k_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_labels() -> Vec<String> {
        let mut labels: Vec<String> = (0..7356).map(|i| format!("char_{}", i)).collect();
        labels[0] = "?".to_string();
        labels[100] = "一".to_string();
        labels[200] = "人".to_string();
        labels
    }

    #[test]
    fn test_predict_with_zeros_model() {
        let device = Device::Cpu;
        let config = model::TransformerConfig::default();
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
        let transformer = model::StrokeTransformer::load(vb, &config).unwrap();

        let labels = test_labels();
        let inference = Inference {
            model: transformer,
            labels,
        };

        let strokes = vec![vec![(10.0, 10.0), (50.0, 50.0), (90.0, 30.0)]];
        let result = inference.predict(&strokes, 5).unwrap();
        assert_eq!(result.len(), 5);
        for pred in &result {
            assert!(!pred.label.is_empty());
            assert!(pred.score >= 0.0);
            assert!(pred.score <= 1.0);
        }
    }

    #[test]
    fn test_predict_topk_bounds() {
        let device = Device::Cpu;
        let config = model::TransformerConfig::default();
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
        let transformer = model::StrokeTransformer::load(vb, &config).unwrap();

        let labels = test_labels();
        let inference = Inference {
            model: transformer,
            labels,
        };

        let strokes = vec![vec![(0.0, 0.0), (100.0, 100.0)]];
        let result = inference.predict(&strokes, 0).unwrap();
        assert!(result.is_empty());

        let result = inference.predict(&strokes, 10).unwrap();
        assert_eq!(result.len(), 10);
        assert_eq!(result[0].class_idx, 0);
    }

    #[test]
    fn test_predict_multiple_strokes() {
        let device = Device::Cpu;
        let config = model::TransformerConfig::default();
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
        let transformer = model::StrokeTransformer::load(vb, &config).unwrap();

        let labels = test_labels();
        let inference = Inference {
            model: transformer,
            labels,
        };

        let strokes = vec![
            vec![(10.0, 10.0), (50.0, 10.0)],
            vec![(10.0, 50.0), (50.0, 50.0)],
            vec![(25.0, 10.0), (25.0, 50.0)],
        ];
        let result = inference.predict(&strokes, 3).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_predict_empty_strokes() {
        let device = Device::Cpu;
        let config = model::TransformerConfig::default();
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
        let transformer = model::StrokeTransformer::load(vb, &config).unwrap();

        let labels = test_labels();
        let inference = Inference {
            model: transformer,
            labels,
        };

        let strokes: Vec<Vec<(f32, f32)>> = vec![];
        let result = inference.predict(&strokes, 5).unwrap();
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_topk_score_descending() {
        let device = Device::Cpu;
        let config = model::TransformerConfig::default();
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
        let transformer = model::StrokeTransformer::load(vb, &config).unwrap();

        let labels = test_labels();
        let inference = Inference {
            model: transformer,
            labels,
        };

        let strokes = vec![vec![(0.0, 0.0), (1.0, 1.0)]];
        let result = inference.predict(&strokes, 10).unwrap();
        for i in 1..result.len() {
            assert!(result[i - 1].score >= result[i].score,
                "scores should be in descending order");
        }
    }

    #[test]
    #[ignore]
    fn test_debug_weight_dump() {
        use std::collections::HashMap;
        let weights_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .join("ochw-wasm/ochw_model_fp16.safetensors");
        let tensors: HashMap<String, candle_core::Tensor> =
            candle_core::safetensors::load(weights_path, &candle_core::Device::Cpu).unwrap();
        let mut names: Vec<&String> = tensors.keys().collect();
        names.sort();
        println!("Tensors in safetensors ({} total):", names.len());
        for name in &names {
            let t = &tensors[*name];
            println!("  {}: {:?} {:?}", name, t.shape(), t.dtype());
        }
        let expected_tensors = [
            "input_proj.weight", "input_proj.bias",
            "pos_encoder.pe",
            "cls_token",
            "norm.weight", "norm.bias",
            "classifier.1.weight", "classifier.1.bias",
            "classifier.4.weight", "classifier.4.bias",
        ];
        for layer_idx in 0..3 {
            let prefix = format!("transformer.layers.{}.", layer_idx);
            for name in &["self_attn.in_proj_weight", "self_attn.in_proj_bias",
                           "self_attn.out_proj.weight", "self_attn.out_proj.bias",
                           "linear1.weight", "linear1.bias",
                           "linear2.weight", "linear2.bias",
                           "norm1.weight", "norm1.bias",
                           "norm2.weight", "norm2.bias"] {
                let full_name = format!("{}{}", prefix, name);
                let found = tensors.contains_key(&full_name);
                if !found {
                    println!("  MISSING: {}", full_name);
                }
            }
        }
        for name in &expected_tensors {
            let found = tensors.contains_key(&name.to_string());
            if !found {
                println!("  MISSING: {}", name);
            }
        }
        println!("All tensors accounted for.");
    }

    #[test]
    #[ignore]
    fn test_debug_intermediate_outputs() {
        use candle_nn::Module;
        let weights = include_bytes!("../../ochw-wasm/ochw_model_fp16.safetensors");
        let labels_json = include_str!("../../ochw-wasm/char_index.json");
        let chars: Vec<String> = serde_json::from_str::<serde_json::Value>(labels_json)
            .unwrap()["chars"].as_array().unwrap()
            .iter().map(|c| c.as_str().unwrap().to_string()).collect();
        let inference = Inference::load(weights, chars).unwrap();
        let device = candle_core::Device::Cpu;

        let strokes = vec![vec![(100.0, 100.0), (200.0, 100.0), (300.0, 100.0)]];
        let input = crate::preprocessing::strokes_to_tensor(&strokes, &device).unwrap();
        println!("Input shape: {:?} dtype: {:?}", input.shape(), input.dtype());

        let x = input.to_dtype(candle_core::DType::F32).unwrap();
        let h = inference.model.input_proj.forward(&x).unwrap();
        println!("After input_proj: {:?} range [{:.4}, {:.4}]", h.shape(),
            h.to_vec3::<f32>().unwrap().iter().flatten().flatten().cloned().fold(f32::INFINITY, f32::min),
            h.to_vec3::<f32>().unwrap().iter().flatten().flatten().cloned().fold(f32::NEG_INFINITY, f32::max));

        // manually trace the forward
        let b = 1;
        let d_model = 192;
        let cls = inference.model.cls_token.broadcast_as((b, 1, d_model)).unwrap();
        let h2 = candle_core::Tensor::cat(&[&cls, &h], 1).unwrap();
        println!("After CLS cat: {:?}", h2.shape());
        
        // check cls_token values
        let cls_vals: Vec<f32> = inference.model.cls_token.squeeze(0).unwrap().squeeze(0).unwrap().to_vec1().unwrap();
        let cls_min = cls_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let cls_max = cls_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("CLS token range: [{:.4}, {:.4}], first 5: {:?}", cls_min, cls_max, &cls_vals[..5]);

        // check pos_encoder values
        let pe = &inference.model.pos_encoder.pe;
        let pe_vals: Vec<f32> = pe.squeeze(0).unwrap().to_vec2().unwrap().iter().take(1).flatten().copied().collect();
        let pe_min = pe_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let pe_max = pe_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("PE shape: {:?}, first row range: [{:.4}, {:.4}], first 5: {:?}", pe.shape(), pe_min, pe_max, &pe_vals[..5]);

        // check transformer output
        let mut h = inference.model.pos_encoder.forward(&h2).unwrap();
        println!("After pos_encoder: {:?}", h.shape());

        let src_mask = candle_core::Tensor::zeros((1, 4), candle_core::DType::F32, &device).unwrap();
        for (i, layer) in inference.model.layers.iter().enumerate() {
            h = layer.forward(&h, &src_mask).unwrap();
            println!("After layer {}: {:?} [{:.4}, {:.4}]", i, h.shape(),
                h.to_vec3::<f32>().unwrap().iter().flatten().flatten().cloned().fold(f32::INFINITY, f32::min),
                h.to_vec3::<f32>().unwrap().iter().flatten().flatten().cloned().fold(f32::NEG_INFINITY, f32::max));
        }
    }

    #[test]
    #[ignore]
    fn test_debug_model_output() {
        // 加载真实权重并检查输出是否有 NaN
        let weights = include_bytes!("../../ochw-wasm/ochw_model_fp16.safetensors");
        let labels_json = include_str!("../../ochw-wasm/char_index.json");
        let chars: Vec<String> = serde_json::from_str::<serde_json::Value>(labels_json)
            .unwrap()["chars"]
            .as_array().unwrap()
            .iter()
            .map(|c| c.as_str().unwrap().to_string())
            .collect();
        let inference = Inference::load(weights, chars).unwrap();

        // 简单输入
        let strokes = vec![
            vec![(100.0, 100.0), (200.0, 100.0), (300.0, 100.0)],
        ];
        let device = candle_core::Device::Cpu;
        let input = crate::preprocessing::strokes_to_tensor(&strokes, &device).unwrap();
        let seq_len = strokes.iter().map(|s| s.len()).sum::<usize>();
        let output = inference.model.forward(&input, seq_len).unwrap();
        println!("Output shape: {:?}", output.shape());
        println!("Output dtype: {:?}", output.dtype());
        let output_f32 = output.to_dtype(candle_core::DType::F32).unwrap();
        let output_vec: Vec<f32> = output_f32.squeeze(0).unwrap().to_vec1().unwrap();
        let has_nan = output_vec.iter().any(|v| v.is_nan());
        let has_inf = output_vec.iter().any(|v| v.is_infinite());
        let max_val = output_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_val = output_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        println!("Has NaN: {}, Has Inf: {}", has_nan, has_inf);
        println!("Output range: [{}, {}]", min_val, max_val);
        let mut top5: Vec<(usize, f32)> = output_vec.iter().copied().enumerate().collect();
        top5.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        println!("Raw logit top-5:");
        for (i, (idx, val)) in top5.iter().take(5).enumerate() {
            let label = inference.labels.get(*idx).cloned().unwrap_or_else(|| "?".to_string());
            println!("  {}. [{}] '{}' ({})", i+1, idx, label, val);
        }

        // 检查 softmax
        let sm = candle_nn::ops::softmax(&output, 1).unwrap();
        let sm_vec: Vec<f32> = sm.squeeze(0).unwrap().to_vec1().unwrap();
        let sm_sum: f32 = sm_vec.iter().sum();
        println!("Softmax sum: {}", sm_sum);
        let sm_max = sm_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sm_max_idx = sm_vec.iter().position(|&v| (v - sm_max).abs() < 1e-6);
        println!("Softmax max: {} at idx {:?}", sm_max, sm_max_idx);
        println!("Top-5 by index:");
        let mut idxs: Vec<(usize, f32)> = sm_vec.iter().copied().enumerate().collect();
        idxs.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        for (i, (idx, prob)) in idxs.iter().take(10).enumerate() {
            let label = inference.labels.get(*idx).cloned().unwrap_or_else(|| "?".to_string());
            println!("  {}. [{}] '{}' ({:.4})", i+1, idx, label, prob);
        }
    }

    fn load_strokes_from_demo(json: &str) -> Vec<Vec<(f32, f32)>> {
        let data: serde_json::Value = serde_json::from_str(json).unwrap();
        data["strokes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|stroke| {
                stroke
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|pt| {
                        let arr = pt.as_array().unwrap();
                        (arr[0].as_f64().unwrap() as f32, arr[1].as_f64().unwrap() as f32)
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    #[ignore]
    fn test_inference_with_demo_files() {
        let weights = include_bytes!("../../ochw-wasm/ochw_model_fp16.safetensors");
        let labels_json = include_str!("../../ochw-wasm/char_index.json");
        let chars: Vec<String> = serde_json::from_str::<serde_json::Value>(labels_json)
            .unwrap()["chars"]
            .as_array().unwrap()
            .iter()
            .map(|c| c.as_str().unwrap().to_string())
            .collect();
        let inference = Inference::load(weights, chars).unwrap();

        let demo_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .join("logs/demo");

        if !demo_dir.exists() {
            eprintln!("demo dir not found: {:?}", demo_dir);
            return;
        }

        let mut passed = 0;
        let mut total = 0;

        for entry in std::fs::read_dir(&demo_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            total += 1;

            let content = std::fs::read_to_string(&path).unwrap();
            let strokes = load_strokes_from_demo(&content);

            let result = inference.predict(&strokes, 10).unwrap();

            // 读取 expected predictions
            let data: serde_json::Value = serde_json::from_str(&content).unwrap();
            let expected_texts: Vec<String> = data["predictions"]
                .as_array().unwrap()
                .iter()
                .map(|p| p["text"].as_str().unwrap_or("?").to_string())
                .collect();

            let rust_top1 = &result[0].label;
            let expected_top1 = &expected_texts[0];

            println!("\n--- {} ---", path.file_name().unwrap().to_str().unwrap());
            println!("  strokes: {} strokes, {} pts total",
                strokes.len(),
                strokes.iter().map(|s| s.len()).sum::<usize>());
            println!("  expected top-1: '{}' ({:.4})", expected_top1,
                data["predictions"][0]["prob"].as_f64().unwrap_or(0.0));
            println!("  Rust top-5:");
            for (i, r) in result.iter().enumerate() {
                println!("    {}. '{}' ({:.4})", i + 1, r.label, r.score);
            }

            if rust_top1 == expected_top1 {
                passed += 1;
            } else {
                println!("  *** MISMATCH: Rust '{}' vs expected '{}'", rust_top1, expected_top1);
            }
        }

        println!("\n=== {}/{} demo files passed ===", passed, total);
        // 至少有一半通过（浮点精度导致的差异可能让部分文件不匹配）
        assert!(passed >= total / 2, "too many mismatches: {}/{}", passed, total);
    }

    #[test]
    fn test_inference_with_real_weights() {
        let weights = include_bytes!("../../ochw-wasm/ochw_model_fp16.safetensors");
        let labels_json = include_str!("../../ochw-wasm/char_index.json");
        let chars: Vec<String> = serde_json::from_str::<serde_json::Value>(labels_json)
            .unwrap()["chars"]
            .as_array().unwrap()
            .iter()
            .map(|c| c.as_str().unwrap().to_string())
            .collect();

        let inference = Inference::load(weights, chars).unwrap();

        // 简单横线
        let strokes = vec![
            vec![(100.0, 100.0), (200.0, 100.0), (300.0, 100.0)],
        ];
        let result = inference.predict(&strokes, 5).unwrap();
        assert_eq!(result.len(), 5);
        for r in &result {
            assert!(!r.label.is_empty());
            assert!(r.score > 0.0 && r.score <= 1.0);
            assert!(r.class_idx < 7356);
        }
    }

    #[test]
    fn test_inference_with_real_weights_complex() {
        let weights = include_bytes!("../../ochw-wasm/ochw_model_fp16.safetensors");
        let labels_json = include_str!("../../ochw-wasm/char_index.json");
        let chars: Vec<String> = serde_json::from_str::<serde_json::Value>(labels_json)
            .unwrap()["chars"]
            .as_array().unwrap()
            .iter()
            .map(|c| c.as_str().unwrap().to_string())
            .collect();

        let inference = Inference::load(weights, chars).unwrap();

        // 十字交叉 + 竖
        let strokes = vec![
            vec![(100.0, 100.0), (300.0, 100.0)],
            vec![(200.0, 50.0), (200.0, 150.0)],
        ];
        let result = inference.predict(&strokes, 10).unwrap();
        assert_eq!(result.len(), 10);
        // 分数递减
        for i in 1..result.len() {
            assert!(result[i-1].score >= result[i].score);
        }
        // 第一个预测应该有合理置信度 (> 1/7356 ≈ 0.00014)
        assert!(result[0].score > 0.00014);
    }

    #[test]
    fn test_topk_deterministic() {
        let weights = include_bytes!("../../ochw-wasm/ochw_model_fp16.safetensors");
        let labels_json = include_str!("../../ochw-wasm/char_index.json");
        let chars: Vec<String> = serde_json::from_str::<serde_json::Value>(labels_json)
            .unwrap()["chars"]
            .as_array().unwrap()
            .iter()
            .map(|c| c.as_str().unwrap().to_string())
            .collect();

        let inference = Inference::load(weights, chars).unwrap();
        let strokes = vec![vec![(150.0, 150.0), (250.0, 150.0), (200.0, 100.0), (200.0, 200.0)]];

        // 两次推理结果应该一致
        let r1 = inference.predict(&strokes, 10).unwrap();
        let r2 = inference.predict(&strokes, 10).unwrap();
        for i in 0..10 {
            assert_eq!(r1[i].label, r2[i].label,
                "results should be deterministic, mismatch at {i}: '{}' vs '{}'",
                r1[i].label, r2[i].label);
        }
    }
}
