import { useState, useCallback, useRef, useEffect } from 'react'
import { CanvasBoard } from './components/CanvasBoard'
import './App.css'

interface TopK {
  label: string
  score: number
  class_idx: number
}

type StrokePoint = { x: number; y: number }
type Stroke = StrokePoint[]

function App() {
  const [loading, setLoading] = useState(true)
  const [modelReady, setModelReady] = useState(false)
  const [candidates, setCandidates] = useState<TopK[]>([])
  const [mainText, setMainText] = useState('等待书写...')
  const [statusText, setStatusText] = useState('加载模型中...')
  const [historyText, setHistoryText] = useState('')
  const [boardKey, setBoardKey] = useState(0)
  const strokesRef = useRef<Stroke[]>([])
  const workerRef = useRef<Worker | null>(null)

  useEffect(() => {
    const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
    workerRef.current = worker

    worker.onmessage = (e: MessageEvent) => {
      const data = e.data
      if (data.status === 'loaded') {
        setModelReady(true)
        setLoading(false)
        setStatusText('就绪 — 在画板上书写汉字')
      } else if (data.status === 'complete') {
        setCandidates(data.output || [])
        setLoading(false)
      } else if (data.status === 'loading') {
        setLoading(true)
      } else if (data.error) {
        console.error(data.error)
        setLoading(false)
        setStatusText('加载失败，请刷新页面重试')
      }
    }

    worker.postMessage({ type: 'init' })

    return () => {
      worker.terminate()
    }
  }, [])

  const handleStrokesChange = useCallback(
    (strokes: Stroke[], _times: number[]) => {
      strokesRef.current = strokes

      if (strokes.length === 0 || !modelReady) {
        setCandidates([])
        setMainText(historyText || '等待书写...')
        return
      }

      setLoading(true)
      setStatusText(`笔画: ${strokes.length} | 推理中...`)

      const strokesData = strokes.map((s) => s.map((p) => [p.x, p.y]))
      workerRef.current?.postMessage({ type: 'predict', strokes: strokesData })
    },
    [modelReady, historyText],
  )

  const selectCandidate = useCallback(
    (idx: number) => {
      if (idx >= candidates.length) return
      const word = candidates[idx].label
      const newHistory = historyText + word
      setHistoryText(newHistory)
      setMainText(newHistory)
      setCandidates([])
      strokesRef.current = []
    },
    [candidates, historyText],
  )

  const clearAll = () => {
    setHistoryText('')
    setMainText('等待书写...')
    setCandidates([])
    strokesRef.current = []
    setStatusText('')
    setBoardKey((c) => c + 1)
  }

  if (!modelReady && loading) {
    return (
      <div className="app-container">
        <div className="loading-overlay">
          <div className="spinner" />
          <p>加载模型中...</p>
          <p className="hint">首次加载约需 5-15 秒（模型权重 ~5MB）</p>
        </div>
      </div>
    )
  }

  return (
    <div className="app-container">
      <div className="toolbar">
        <span className="title">手写汉字识别</span>
        <span className="badge">WASM</span>
        <button onClick={clearAll} className="btn">
          清空
        </button>
      </div>

      <CanvasBoard key={boardKey} onStrokesChange={handleStrokesChange} />

      <div className="main-result">{mainText}</div>

      <div className="candidates">
        {candidates.map((cand, i) => (
          <button
            key={i}
            className="cand-btn"
            onClick={() => selectCandidate(i)}
          >
            {i + 1}.{cand.label}
          </button>
        ))}
        {candidates.length === 0 &&
          Array.from({ length: 5 }).map((_, i) => (
            <button key={i} className="cand-btn cand-btn-empty" disabled>
              {i + 1}.
            </button>
          ))}
      </div>

      <div className="status">{statusText}</div>

      <div className="history">
        <strong>已输入:</strong>
        {historyText || '\u2014'}
      </div>
    </div>
  )
}

export default App
