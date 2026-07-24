import { useEffect, useRef, useState } from 'react'
import { useStore } from './store'

interface Props {
  overlaySrc: string | null
  overlayOpacity?: number
}

const MIN_SCALE = 0.1
const MAX_SCALE = 12

/** A lightweight, standalone pan/zoom map for viewing run results (ortho +
 * one server-rendered overlay PNG). Deliberately separate from MapView/its
 * design-editing view state - this one never edits anything. */
export default function ResultMap({ overlaySrc, overlayOpacity = 0.9 }: Props) {
  const meta = useStore((s) => s.meta)
  const containerRef = useRef<HTMLDivElement>(null)
  const [view, setView] = useState({ scale: 1, ox: 0, oy: 0 })
  const panState = useRef<{ x: number; y: number; ox: number; oy: number } | null>(null)

  useEffect(() => {
    if (!meta || !containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const scale = Math.min(rect.width / meta.width, rect.height / meta.height) * 0.95
    setView({ scale, ox: (rect.width - meta.width * scale) / 2, oy: (rect.height - meta.height * scale) / 2 })
  }, [meta])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top
      setView((v) => {
        const worldX = (mx - v.ox) / v.scale
        const worldY = (my - v.oy) / v.scale
        const factor = Math.exp(-e.deltaY * 0.0015)
        const newScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, v.scale * factor))
        return { scale: newScale, ox: mx - worldX * newScale, oy: my - worldY * newScale }
      })
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  function onPointerDown(e: React.PointerEvent) {
    panState.current = { x: e.clientX, y: e.clientY, ox: view.ox, oy: view.oy }
    ;(e.target as Element).setPointerCapture(e.pointerId)
  }
  function onPointerMove(e: React.PointerEvent) {
    if (!panState.current) return
    const dx = e.clientX - panState.current.x
    const dy = e.clientY - panState.current.y
    setView((v) => ({ ...v, ox: panState.current!.ox + dx, oy: panState.current!.oy + dy }))
  }
  function onPointerUp(e: React.PointerEvent) {
    panState.current = null
    ;(e.target as Element).releasePointerCapture(e.pointerId)
  }

  return (
    <div
      ref={containerRef}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      style={{ position: 'relative', width: '100%', height: '100%', overflow: 'hidden', background: '#111', cursor: 'grab' }}
    >
      {meta && (
        <div
          style={{
            position: 'absolute',
            left: 0,
            top: 0,
            width: meta.width,
            height: meta.height,
            transform: `translate(${view.ox}px, ${view.oy}px) scale(${view.scale})`,
            transformOrigin: '0 0',
          }}
        >
          <img
            src="/api/terrain/ortho.png"
            width={meta.width}
            height={meta.height}
            draggable={false}
            style={{ position: 'absolute', left: 0, top: 0 }}
            alt="orthophoto"
          />
          {overlaySrc && (
            <img
              src={overlaySrc}
              width={meta.width}
              height={meta.height}
              draggable={false}
              style={{ position: 'absolute', left: 0, top: 0, opacity: overlayOpacity }}
              alt="overlay"
            />
          )}
        </div>
      )}
    </div>
  )
}
