import wasmUrl from "../node_modules/ochw-wasm/ochw_wasm_bg.wasm?url"
import init, { Model } from "ochw-wasm"

let model: Model | null = null

self.addEventListener('message', async (event: MessageEvent) => {
  try {
    const { type, strokes } = event.data

    if (type === 'init') {
      self.postMessage({ status: 'loading' })
      await init(wasmUrl)
      model = new Model()
      self.postMessage({ status: 'loaded' })
      return
    }

    if (type === 'predict' && model) {
      const result = model.predict(strokes)
      self.postMessage({ status: 'complete', output: result })
      return
    }
  } catch (e) {
    self.postMessage({ error: `worker error: ${e}` })
  }
})
