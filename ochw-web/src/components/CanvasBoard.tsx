import { useRef, useEffect } from 'react'

type StrokePoint = { x: number; y: number }
type Stroke = StrokePoint[]

interface Props {
  onStrokesChange: (strokes: Stroke[], strokeTimes: number[]) => void
}

export function CanvasBoard({ onStrokesChange }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const currentStrokeRef = useRef<StrokePoint[]>([])
  const allStrokesRef = useRef<Stroke[]>([])
  const timesRef = useRef<number[]>([])
  const isDrawing = useRef(false)

  const getPos = (e: MouseEvent | TouchEvent): StrokePoint => {
    const canvas = canvasRef.current!
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    let clientX: number, clientY: number
    if ('touches' in e) {
      const t = e.touches[0] || (e as TouchEvent).changedTouches[0]
      clientX = t.clientX
      clientY = t.clientY
    } else {
      clientX = e.clientX
      clientY = e.clientY
    }
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
    }
  }

  const redraw = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    ctx.fillStyle = '#fff'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    ctx.strokeStyle = '#000'
    ctx.lineWidth = 3
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    for (const stroke of allStrokesRef.current) {
      if (stroke.length < 2) continue
      ctx.beginPath()
      ctx.moveTo(stroke[0].x, stroke[0].y)
      for (let i = 1; i < stroke.length; i++) {
        ctx.lineTo(stroke[i].x, stroke[i].y)
      }
      ctx.stroke()
    }

    const cur = currentStrokeRef.current
    if (cur.length >= 2) {
      ctx.beginPath()
      ctx.moveTo(cur[0].x, cur[0].y)
      for (let i = 1; i < cur.length; i++) {
        ctx.lineTo(cur[i].x, cur[i].y)
      }
      ctx.stroke()
    }
  }

  const notifyChange = () => {
    onStrokesChange(allStrokesRef.current, timesRef.current)
  }

  const startStroke = (e: MouseEvent | TouchEvent) => {
    if ('button' in e && e.button !== 0) return
    e.preventDefault()
    isDrawing.current = true
    currentStrokeRef.current = [getPos(e)]
  }

  const moveStroke = (e: MouseEvent | TouchEvent) => {
    e.preventDefault()
    if (!isDrawing.current) return
    currentStrokeRef.current.push(getPos(e))
    redraw()
  }

  const endStroke = (e: MouseEvent | TouchEvent) => {
    e.preventDefault()
    if (!isDrawing.current) return
    isDrawing.current = false

    const cur = currentStrokeRef.current
    if (cur.length >= 2) {
      allStrokesRef.current = [...allStrokesRef.current, cur]
      timesRef.current = [...timesRef.current, Date.now()]
    }
    currentStrokeRef.current = []
    notifyChange()
    redraw()
  }

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    canvas.addEventListener('mousedown', startStroke)
    canvas.addEventListener('mousemove', moveStroke)
    canvas.addEventListener('mouseup', endStroke)
    canvas.addEventListener('mouseleave', endStroke)
    canvas.addEventListener('touchstart', startStroke, { passive: false })
    canvas.addEventListener('touchmove', moveStroke, { passive: false })
    canvas.addEventListener('touchend', endStroke, { passive: false })

    return () => {
      canvas.removeEventListener('mousedown', startStroke)
      canvas.removeEventListener('mousemove', moveStroke)
      canvas.removeEventListener('mouseup', endStroke)
      canvas.removeEventListener('mouseleave', endStroke)
      canvas.removeEventListener('touchstart', startStroke)
      canvas.removeEventListener('touchmove', moveStroke)
      canvas.removeEventListener('touchend', endStroke)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      width={600}
      height={350}
      style={{
        border: '2px solid #ccc',
        borderRadius: 8,
        cursor: 'crosshair',
        touchAction: 'none',
        width: '100%',
        maxWidth: 600,
        height: 350,
      }}
    />
  )
}
