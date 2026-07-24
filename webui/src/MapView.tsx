import { useEffect, useMemo, useRef, useState } from 'react'
import { useStore } from './store'
import { computeNodataOverlay, computeZoneOverlay, fetchDem, fetchMasksRgba, fetchMeta, pixelToUtm } from './terrain'
import {
  buildEditableMask,
  materialColorTable,
  maxAbsDelta,
  renderElevationHeatmapRect,
  renderMaterialRect,
  renderSculptRect,
  Stroke,
  UndoManager,
} from './editor'
import { patchDesign } from './api'

const MIN_SCALE = 0.1
const MAX_SCALE = 12

export default function MapView() {
  const meta = useStore((s) => s.meta)
  const setMeta = useStore((s) => s.setMeta)
  const dem = useStore((s) => s.dem)
  const setDem = useStore((s) => s.setDem)
  const masksRgba = useStore((s) => s.masksRgba)
  const setMasksRgba = useStore((s) => s.setMasksRgba)
  const view = useStore((s) => s.view)
  const setView = useStore((s) => s.setView)

  const designName = useStore((s) => s.designName)
  const material = useStore((s) => s.material)
  const demDelta = useStore((s) => s.demDelta)
  const materials = useStore((s) => s.materials)
  const unlocked = useStore((s) => s.unlocked)
  const tool = useStore((s) => s.tool)
  const brushRadiusM = useStore((s) => s.brushRadiusM)
  const brushStrengthMps = useStore((s) => s.brushStrengthMps)
  const brushSoftness = useStore((s) => s.brushSoftness)
  const activeMaterialId = useStore((s) => s.activeMaterialId)
  const setDirty = useStore((s) => s.setDirty)
  const bumpEditVersion = useStore((s) => s.bumpEditVersion)
  const setUndoRedoCounts = useStore((s) => s.setUndoRedoCounts)
  const setUndoRedoFns = useStore((s) => s.setUndoRedoFns)
  const setUndoManager = useStore((s) => s.setUndoManager)
  const layerVersion = useStore((s) => s.layerVersion)
  const overlayMode = useStore((s) => s.overlayMode)
  const elevationRangeM = useStore((s) => s.elevationRangeM)
  const setElevationRangeM = useStore((s) => s.setElevationRangeM)

  const containerRef = useRef<HTMLDivElement>(null)
  const zoneCanvasRef = useRef<HTMLCanvasElement>(null)
  const nodataCanvasRef = useRef<HTMLCanvasElement>(null)
  const materialCanvasRef = useRef<HTMLCanvasElement>(null)
  const sculptCanvasRef = useRef<HTMLCanvasElement>(null)
  const spaceHeld = useRef(false)
  const panState = useRef<{ x: number; y: number; ox: number; oy: number } | null>(null)
  const strokeRef = useRef<Stroke | null>(null)
  const lastStampTime = useRef(0)
  const editableMaskRef = useRef<Uint8Array | null>(null)
  const undoManagerRef = useRef<UndoManager | null>(null)
  const pointerCellRef = useRef<{ row: number; col: number } | null>(null)
  const rafIdRef = useRef<number | null>(null)

  const [cursor, setCursor] = useState<{ row: number; col: number; x: number; y: number } | null>(null)
  const [brushCursor, setBrushCursor] = useState<{ sx: number; sy: number; editable: boolean } | null>(null)
  const [loading, setLoading] = useState('loading terrain...')
  const [panning, setPanning] = useState(false)
  const [sculptWarning, setSculptWarning] = useState<string | null>(null)

  const colorTable = useMemo(() => materialColorTable(materials), [materials])

  // ---- initial terrain load (independent of any design) ----
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      const m = await fetchMeta()
      if (cancelled) return
      setMeta(m)
      setLoading('loading elevation...')
      const [d, masks] = await Promise.all([fetchDem(m), fetchMasksRgba(m.width, m.height)])
      if (cancelled) return
      setDem(d)
      setMasksRgba(masks)
      setLoading('')
    })().catch((e) => setLoading('error: ' + e))
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ---- fit-to-container once meta is known ----
  useEffect(() => {
    if (!meta || !containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const scale = Math.min(rect.width / meta.width, rect.height / meta.height) * 0.95
    const ox = (rect.width - meta.width * scale) / 2
    const oy = (rect.height - meta.height * scale) / 2
    setView({ scale, ox, oy })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meta])

  // ---- zone overlay (base terrain layer, independent of designs) ----
  useEffect(() => {
    if (!meta || !masksRgba || !zoneCanvasRef.current) return
    const overlay = computeZoneOverlay(masksRgba, meta.width, meta.height)
    const canvas = zoneCanvasRef.current
    canvas.width = meta.width
    canvas.height = meta.height
    const ctx = canvas.getContext('2d')!
    ctx.putImageData(new ImageData(overlay, meta.width, meta.height), 0, 0)
  }, [meta, masksRgba])

  // ---- nodata overlay: hide the raw black export pixels outside the survey ----
  useEffect(() => {
    if (!meta || !masksRgba || !nodataCanvasRef.current) return
    const overlay = computeNodataOverlay(masksRgba, meta.width, meta.height)
    const canvas = nodataCanvasRef.current
    canvas.width = meta.width
    canvas.height = meta.height
    const ctx = canvas.getContext('2d')!
    ctx.putImageData(new ImageData(overlay, meta.width, meta.height), 0, 0)
  }, [meta, masksRgba])

  // ---- editable mask, recomputed when the base masks or the lock state change ----
  useEffect(() => {
    if (!meta || !masksRgba) return
    editableMaskRef.current = buildEditableMask(masksRgba, meta.width, meta.height, unlocked)
  }, [meta, masksRgba, unlocked])

  // ---- design canvases: full redraw on open / material-table change / mode switch ----
  useEffect(() => {
    if (!meta || !material || !demDelta) return
    const mc = materialCanvasRef.current
    const sc = sculptCanvasRef.current
    if (!mc || !sc) return
    mc.width = meta.width
    mc.height = meta.height
    sc.width = meta.width
    sc.height = meta.height
    const full = { minX: 0, minY: 0, maxX: meta.width - 1, maxY: meta.height - 1 }
    const range = maxAbsDelta(demDelta)
    setElevationRangeM(range)
    mc.getContext('2d')!.putImageData(renderMaterialRect(material, meta.width, colorTable, full), 0, 0)
    sc.getContext('2d')!.putImageData(
      overlayMode === 'elevation'
        ? renderElevationHeatmapRect(demDelta, meta.width, full, range)
        : renderSculptRect(demDelta, meta.width, full),
      0,
      0,
    )
    // layerVersion is intentionally a dependency: bulk edits (e.g. deleting a
    // material everywhere) mutate the typed arrays in place, so the array
    // *references* don't change — layerVersion is what forces this redraw.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meta, material, demDelta, colorTable, designName, layerVersion, overlayMode])

  // ---- undo manager: one per open design, shared with the 3D view via the store ----
  useEffect(() => {
    if (!designName || !meta) {
      undoManagerRef.current = null
      setUndoRedoFns(null, null)
      setUndoManager(null)
      return
    }
    undoManagerRef.current = new UndoManager(
      () => useStore.getState().designName,
      () => useStore.getState().material!,
      () => useStore.getState().demDelta!,
      () => meta.width,
      (rect) => {
        redrawRect(rect)
        refreshElevationRange()
        const counts = undoManagerRef.current!.counts()
        setUndoRedoCounts(counts.undo, counts.redo)
        setDirty(true)
        bumpEditVersion()
      },
    )
    setUndoRedoCounts(0, 0)
    setUndoRedoFns(
      () => undoManagerRef.current?.undo(),
      () => undoManagerRef.current?.redo(),
    )
    setUndoManager(undoManagerRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [designName, meta])

  function redrawRect(rect: { minX: number; minY: number; maxX: number; maxY: number }) {
    if (!meta) return
    const mat = useStore.getState().material
    const delta = useStore.getState().demDelta
    if (mat && materialCanvasRef.current) {
      materialCanvasRef.current.getContext('2d')!.putImageData(renderMaterialRect(mat, meta.width, colorTable, rect), rect.minX, rect.minY)
    }
    if (delta && sculptCanvasRef.current) {
      const img =
        useStore.getState().overlayMode === 'elevation'
          ? renderElevationHeatmapRect(delta, meta.width, rect, useStore.getState().elevationRangeM)
          : renderSculptRect(delta, meta.width, rect)
      sculptCanvasRef.current.getContext('2d')!.putImageData(img, rect.minX, rect.minY)
    }
  }

  /** Full-array rescan of |dem_delta|, called at "settle" points (stroke end,
   * undo/redo) so the heatmap legend/scale reflects the true current range. */
  function refreshElevationRange() {
    const delta = useStore.getState().demDelta
    if (delta) setElevationRangeM(maxAbsDelta(delta))
  }

  // ---- space bar toggles pan mode ----
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code === 'Space') spaceHeld.current = true
    }
    const up = (e: KeyboardEvent) => {
      if (e.code === 'Space') spaceHeld.current = false
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
    }
  }, [])

  // ---- wheel zoom (native listener: must be non-passive to preventDefault) ----
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const { scale, ox, oy } = useStore.getState().view
      const rect = el.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top
      const worldX = (mx - ox) / scale
      const worldY = (my - oy) / scale
      const factor = Math.exp(-e.deltaY * 0.0015)
      const newScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale * factor))
      setView({ scale: newScale, ox: mx - worldX * newScale, oy: my - worldY * newScale })
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function cellAtEvent(e: React.PointerEvent): { row: number; col: number } | null {
    if (!containerRef.current) return null
    const rect = containerRef.current.getBoundingClientRect()
    const mx = e.clientX - rect.left
    const my = e.clientY - rect.top
    const { scale, ox, oy } = useStore.getState().view
    const col = (mx - ox) / scale
    const row = (my - oy) / scale
    return { row, col }
  }

  // Keeps stamping at the last known cursor position every frame while a
  // stroke is active, so sculpt tools (raise/lower/flatten/smooth) accumulate
  // over time even when the pointer isn't moving ("m/s of hold").
  function tick() {
    if (!strokeRef.current || !pointerCellRef.current) {
      rafIdRef.current = null
      return
    }
    const now = performance.now()
    const dt = Math.min((now - lastStampTime.current) / 1000, 0.1)
    lastStampTime.current = now
    const rect = strokeRef.current.stampAt(pointerCellRef.current.row, pointerCellRef.current.col, dt)
    if (rect) redrawRect(rect)
    rafIdRef.current = requestAnimationFrame(tick)
  }

  function onPointerDown(e: React.PointerEvent) {
    // plain left-drag pans whenever there's nothing to paint (no design
    // open) - space+drag or middle-drag always pan, even mid-edit.
    if (e.button === 1 || (e.button === 0 && (spaceHeld.current || !designName))) {
      e.preventDefault()
      panState.current = { x: e.clientX, y: e.clientY, ox: view.ox, oy: view.oy }
      setPanning(true)
      ;(e.target as Element).setPointerCapture(e.pointerId)
      return
    }
    if (e.button === 0 && designName && material && demDelta && dem && meta && editableMaskRef.current) {
      e.preventDefault()
      const cell = cellAtEvent(e)
      if (!cell) return
      const radiusCells = brushRadiusM / meta.res
      strokeRef.current = new Stroke(
        tool,
        activeMaterialId,
        meta.width,
        meta.height,
        dem,
        material,
        demDelta,
        editableMaskRef.current,
        radiusCells,
        brushStrengthMps,
        brushSoftness,
      )
      lastStampTime.current = performance.now()
      pointerCellRef.current = cell
      const rect = strokeRef.current.stampAt(cell.row, cell.col, 0)
      if (rect) redrawRect(rect)
      ;(e.target as Element).setPointerCapture(e.pointerId)
      if (rafIdRef.current === null) rafIdRef.current = requestAnimationFrame(tick)
    }
  }

  function onPointerMove(e: React.PointerEvent) {
    if (panState.current) {
      const dx = e.clientX - panState.current.x
      const dy = e.clientY - panState.current.y
      setView({ ox: panState.current.ox + dx, oy: panState.current.oy + dy })
      return
    }
    if (!meta || !containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const cell = cellAtEvent(e)
    if (!cell) return
    const { row, col } = cell

    if (strokeRef.current) {
      pointerCellRef.current = { row, col }
      const now = performance.now()
      const dt = Math.min((now - lastStampTime.current) / 1000, 0.1)
      lastStampTime.current = now
      const dirty = strokeRef.current.moveTo(row, col, dt)
      if (dirty) redrawRect(dirty)
    }

    const flCol = Math.floor(col)
    const flRow = Math.floor(row)
    if (flCol < 0 || flCol >= meta.width || flRow < 0 || flRow >= meta.height) {
      setCursor(null)
      setBrushCursor(null)
      return
    }
    const { x, y } = pixelToUtm(meta, flCol + 0.5, flRow + 0.5)
    setCursor({ row: flRow, col: flCol, x, y })
    if (designName) {
      const editable = editableMaskRef.current ? editableMaskRef.current[flRow * meta.width + flCol] === 1 : false
      setBrushCursor({ sx: e.clientX - rect.left, sy: e.clientY - rect.top, editable })
    } else {
      setBrushCursor(null)
    }
  }

  async function onPointerUp(e: React.PointerEvent) {
    if (panState.current) {
      panState.current = null
      setPanning(false)
      ;(e.target as Element).releasePointerCapture(e.pointerId)
      return
    }
    if (strokeRef.current) {
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current)
        rafIdRef.current = null
      }
      pointerCellRef.current = null
      const rec = strokeRef.current.finish()
      strokeRef.current = null
      ;(e.target as Element).releasePointerCapture(e.pointerId)
      if (rec && designName) {
        undoManagerRef.current?.push(rec)
        const counts = undoManagerRef.current!.counts()
        setUndoRedoCounts(counts.undo, counts.redo)
        setDirty(true)
        if (rec.beforeDemDelta && rec.afterDemDelta) {
          refreshElevationRange()
          let maxChange = 0
          for (let i = 0; i < rec.afterDemDelta.length; i++) {
            const d = Math.abs(rec.afterDemDelta[i] - rec.beforeDemDelta[i])
            if (d > maxChange) maxChange = d
          }
          if (maxChange > 2.0) {
            setSculptWarning(`That stroke moved the ground up to ${maxChange.toFixed(1)} m in one go — double check this is intended.`)
            setTimeout(() => setSculptWarning(null), 6000)
          }
        }
        bumpEditVersion()
        await patchDesign(designName, { x: rec.x, y: rec.y, w: rec.w, h: rec.h, material: rec.afterMaterial, demDelta: rec.afterDemDelta })
      }
    }
  }

  const brushRadiusPx = meta ? (brushRadiusM / meta.res) * view.scale : 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div
        ref={containerRef}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={() => setBrushCursor(null)}
        style={{
          position: 'relative',
          flex: 1,
          overflow: 'hidden',
          background: '#111',
          cursor: panning ? 'grabbing' : spaceHeld.current ? 'grab' : designName ? 'none' : 'crosshair',
        }}
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
              style={{ position: 'absolute', left: 0, top: 0, userSelect: 'none' }}
              alt="orthophoto"
            />
            <img
              src="/api/terrain/hillshade.png"
              width={meta.width}
              height={meta.height}
              draggable={false}
              style={{ position: 'absolute', left: 0, top: 0, opacity: 0.35, mixBlendMode: 'multiply', userSelect: 'none' }}
              alt="hillshade"
            />
            <canvas ref={nodataCanvasRef} style={{ position: 'absolute', left: 0, top: 0, width: meta.width, height: meta.height }} />
            <canvas ref={zoneCanvasRef} style={{ position: 'absolute', left: 0, top: 0, width: meta.width, height: meta.height }} />
            <canvas
              ref={materialCanvasRef}
              style={{
                position: 'absolute',
                left: 0,
                top: 0,
                width: meta.width,
                height: meta.height,
                opacity: overlayMode === 'elevation' ? 0.15 : 1,
              }}
            />
            <canvas ref={sculptCanvasRef} style={{ position: 'absolute', left: 0, top: 0, width: meta.width, height: meta.height }} />
          </div>
        )}
        {brushCursor && (
          <div
            style={{
              position: 'absolute',
              left: brushCursor.sx - brushRadiusPx,
              top: brushCursor.sy - brushRadiusPx,
              width: brushRadiusPx * 2,
              height: brushRadiusPx * 2,
              borderRadius: '50%',
              border: `2px solid ${brushCursor.editable ? '#4ade80' : '#f87171'}`,
              pointerEvents: 'none',
            }}
          />
        )}
        {loading && (
          <div style={{ position: 'absolute', left: 12, top: 12, color: '#fff', font: '13px sans-serif' }}>{loading}</div>
        )}
        {sculptWarning && (
          <div
            style={{
              position: 'absolute',
              left: '50%',
              top: 12,
              transform: 'translateX(-50%)',
              background: 'rgba(120,53,15,0.9)',
              color: '#fff',
              font: '13px sans-serif',
              padding: '8px 14px',
              borderRadius: 4,
              border: '1px solid #f59e0b',
            }}
          >
            ⚠ {sculptWarning}
          </div>
        )}
        {designName && overlayMode === 'elevation' && (
          <div
            style={{
              position: 'absolute',
              left: 12,
              bottom: 12,
              background: 'rgba(0,0,0,0.6)',
              color: '#fff',
              font: '12px sans-serif',
              padding: '8px 10px',
              borderRadius: 4,
            }}
          >
            <div style={{ marginBottom: 4 }}>Elevation change (dem_delta)</div>
            <div
              style={{
                width: 180,
                height: 10,
                borderRadius: 2,
                background: 'linear-gradient(to right, rgb(20,80,190), #fff, rgb(195,60,30))',
              }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 2 }}>
              <span>-{elevationRangeM.toFixed(2)} m (dug)</span>
              <span>+{elevationRangeM.toFixed(2)} m (raised)</span>
            </div>
          </div>
        )}
      </div>
      <div
        style={{
          padding: '4px 12px',
          font: '12px ui-monospace, monospace',
          background: '#222',
          color: '#ddd',
          borderTop: '1px solid #333',
        }}
      >
        {cursor
          ? `row ${cursor.row}, col ${cursor.col}  |  x ${cursor.x.toFixed(1)}, y ${cursor.y.toFixed(1)} (EPSG:32636)`
          : 'move over the map to see coordinates'}
        {'   —   scroll to zoom, space+drag or middle-drag to pan'}
      </div>
    </div>
  )
}
