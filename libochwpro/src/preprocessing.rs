/// 笔画预处理 — 对应 Python dataset.py 的 strokes_to_sequence
use candle_core::{Device, Result, Tensor};

/// 将原始笔画坐标转为模型输入特征序列 (B, T, 5)
/// strokes: Vec<Vec<(f32, f32)>> — 每笔的点坐标列表
pub fn strokes_to_tensor(
    strokes: &[Vec<(f32, f32)>],
    device: &Device,
) -> Result<Tensor> {
    // 展平所有点 + 落笔标志
    let mut xs: Vec<f32> = Vec::new();
    let mut ys: Vec<f32> = Vec::new();
    let mut pen_down: Vec<f32> = Vec::new();

    for stroke in strokes {
        for &(x, y) in stroke {
            xs.push(x);
            ys.push(y);
            pen_down.push(1.0);
        }
        if let Some(last) = pen_down.last_mut() {
            *last = 0.0; // 每笔最后一个点为抬笔
        }
    }

    if xs.is_empty() {
        return Tensor::zeros((1, 1, 5), candle_core::DType::F32, device);
    }

    // 归一化到 [0, 1]
    let min_x = xs.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_x = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_y = ys.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_y = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let range_x = (max_x - min_x).max(1.0);
    let range_y = (max_y - min_y).max(1.0);

    let n = xs.len();
    let mut features = Vec::with_capacity(n * 5);

    let mut prev_x = xs[0];
    let mut prev_y = ys[0];

    for i in 0..n {
        let x_norm = (xs[i] - min_x) / range_x;
        let y_norm = (ys[i] - min_y) / range_y;
        let dx = (xs[i] - prev_x) / range_x;
        let dy = (ys[i] - prev_y) / range_y;

        features.push(x_norm);
        features.push(y_norm);
        features.push(dx);
        features.push(dy);
        features.push(pen_down[i]);

        prev_x = xs[i];
        prev_y = ys[i];
    }

    Tensor::from_vec(features, (1, n, 5), device)
}

/// 生成 attention mask: (B, T), True = 有效位置
pub fn create_mask(n: usize, device: &Device) -> Result<Tensor> {
    let data: Vec<u8> = vec![1; n];
    Tensor::from_vec(data, (1, n), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_stroke() {
        let device = Device::Cpu;
        let strokes = vec![vec![(0.0, 0.0), (100.0, 100.0), (200.0, 50.0)]];
        let tensor = strokes_to_tensor(&strokes, &device).unwrap();
        assert_eq!(tensor.dims(), &[1, 3, 5]);
        let data = tensor.to_vec3::<f32>().unwrap();
        assert_eq!(data[0][0][4], 1.0);
        assert_eq!(data[0][1][4], 1.0);
        assert_eq!(data[0][2][4], 0.0);
        assert!(data[0][0][0] >= 0.0 && data[0][0][0] <= 1.0);
    }

    #[test]
    fn test_multiple_strokes() {
        let device = Device::Cpu;
        let strokes = vec![
            vec![(0.0, 0.0), (100.0, 0.0)],
            vec![(0.0, 100.0), (100.0, 100.0)],
        ];
        let tensor = strokes_to_tensor(&strokes, &device).unwrap();
        assert_eq!(tensor.dims(), &[1, 4, 5]);
        let data = tensor.to_vec3::<f32>().unwrap();
        assert_eq!(data[0][1][4], 0.0);
        assert_eq!(data[0][2][4], 1.0);
        assert_eq!(data[0][3][4], 0.0);
    }

    #[test]
    fn test_empty_strokes() {
        let device = Device::Cpu;
        let strokes: Vec<Vec<(f32, f32)>> = vec![];
        let tensor = strokes_to_tensor(&strokes, &device).unwrap();
        assert_eq!(tensor.dims(), &[1, 1, 5]);
    }

    #[test]
    fn test_normalization_identity() {
        let device = Device::Cpu;
        let strokes = vec![vec![(10.0, 20.0)]];
        let tensor = strokes_to_tensor(&strokes, &device).unwrap();
        let data = tensor.to_vec3::<f32>().unwrap();
        assert!((data[0][0][0] - 0.0).abs() < 1e-6);
        assert!((data[0][0][1] - 0.0).abs() < 1e-6);
        assert!((data[0][0][2] - 0.0).abs() < 1e-6);
        assert!((data[0][0][3] - 0.0).abs() < 1e-6);
        assert_eq!(data[0][0][4], 0.0);
    }

    #[test]
    fn test_create_mask() {
        let device = Device::Cpu;
        let mask = create_mask(5, &device).unwrap();
        assert_eq!(mask.dims(), &[1, 5]);
        let data = mask.to_vec2::<u8>().unwrap();
        assert_eq!(data[0], vec![1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_create_mask_zero() {
        let device = Device::Cpu;
        let mask = create_mask(0, &device).unwrap();
        assert_eq!(mask.dims(), &[1, 0]);
    }
}
