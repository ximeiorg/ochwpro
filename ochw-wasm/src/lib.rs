use libochwpro::Inference;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen]
pub struct Model {
    worker: Inference,
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<Model, JsError> {
        console_error_panic_hook::set_once();
        let weights = include_bytes!("../ochw_model_fp16.safetensors");
        let labels = load_labels()?;
        let worker = Inference::load(weights, labels)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { worker })
    }

    pub fn predict(&self, strokes: JsValue) -> Result<JsValue, JsError> {
        // 从 JS 接收笔画数据: [[[x,y],...],...]
        let strokes_js: Vec<Vec<Vec<f64>>> = serde_wasm_bindgen::from_value(strokes)?;
        let strokes: Vec<Vec<(f32, f32)>> = strokes_js
            .into_iter()
            .map(|stroke| stroke.into_iter().map(|p| (p[0] as f32, p[1] as f32)).collect())
            .collect();

        let result = self.worker.predict(&strokes, 10)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let json = serde_wasm_bindgen::to_value(&result)?;
        Ok(json)
    }
}

fn load_labels() -> Result<Vec<String>, JsError> {
    // 从 char_index.json 加载标签
    let json_str = include_str!("../char_index.json");
    let data: serde_json::Value = serde_json::from_str(json_str)?;
    let chars = data["chars"].as_array().ok_or(JsError::new("无效的标签文件"))?;
    let labels: Vec<String> = chars.iter().map(|c| c.as_str().unwrap_or("?").to_string()).collect();
    Ok(labels)
}
