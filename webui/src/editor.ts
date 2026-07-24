import type { MaterialDef, Tool } from './store'
import { patchDesign } from './api'

/** Smoothstep falloff weight for a cell at distance `d` from the brush
 * centre, radius `r` (cells), softness `s` in [0,1]. */
export function brushWeight(d: number, r: number, s: number): number {
  if (d > r) return 0
  const soft = Math.max(s, 1e-6)
  const t = Math.max(0, Math.min(1, d / r - (1 - soft)) / soft)
  return 1 - t * t * (3 - 2 * t)
}

/** Precomputed per-cell editability, mirroring the server's zone lock so the
 * brush gives instant feedback; the server re-clamps authoritatively. */
export function buildEditableMask(masksRgba: Uint8ClampedArray, width: number, height: number, unlocked: boolean): Uint8Array {
  const n = width * height
  const out = new Uint8Array(n)
  for (let i = 0; i < n; i++) {
    const o = i * 4
    const building = masksRgba[o] > 127
    const zone = masksRgba[o + 1] > 127
    const water = masksRgba[o + 2] > 127
    const valid = masksRgba[o + 3] > 127
    out[i] = unlocked ? (valid && !building && !water ? 1 : 0) : (zone && valid && !building ? 1 : 0)
  }
  return out
}

interface Bbox {
  minX: number
  minY: number
  maxX: number
  maxY: number
}

export interface StrokeRecord {
  x: number
  y: number
  w: number
  h: number
  beforeMaterial?: Uint16Array
  afterMaterial?: Uint16Array
  beforeDemDelta?: Float32Array
  afterDemDelta?: Float32Array
}

const DEM_DELTA_LIMIT = 3.0

/** Tracks one paint/sculpt stroke: applies the brush in-place on the shared
 * `material`/`demDelta` typed arrays, records original values on first touch
 * of each cell (for undo), and reports the dirty rect after every stamp so
 * the caller can redraw just that rectangle. */
export class Stroke {
  private touched = new Map<number, { mat?: number; delta?: number }>()
  private bbox: Bbox | null = null
  private flattenTarget: number | null = null
  private lastPoint: { row: number; col: number } | null = null

  constructor(
    private tool: Tool,
    private activeMaterialId: number,
    private width: number,
    private height: number,
    private dem: Float32Array,
    private material: Uint16Array,
    private demDelta: Float32Array,
    private editable: Uint8Array,
    private radiusCells: number,
    private strengthPerSec: number,
    private softness: number,
  ) {}

  private touchesMaterial() {
    return this.tool === 'paint' || this.tool === 'eraser'
  }

  private touchesDem() {
    return this.tool === 'raise' || this.tool === 'lower' || this.tool === 'flatten' || this.tool === 'smooth'
  }

  private recordOriginal(idx: number) {
    if (this.touched.has(idx)) return
    const entry: { mat?: number; delta?: number } = {}
    if (this.touchesMaterial()) entry.mat = this.material[idx]
    if (this.touchesDem()) entry.delta = this.demDelta[idx]
    this.touched.set(idx, entry)
  }

  private growBbox(x0: number, y0: number, x1: number, y1: number) {
    if (!this.bbox) {
      this.bbox = { minX: x0, minY: y0, maxX: x1, maxY: y1 }
    } else {
      this.bbox.minX = Math.min(this.bbox.minX, x0)
      this.bbox.minY = Math.min(this.bbox.minY, y0)
      this.bbox.maxX = Math.max(this.bbox.maxX, x1)
      this.bbox.maxY = Math.max(this.bbox.maxY, y1)
    }
  }

  /** Stamp the brush once at (row, col). Returns the dirty rect touched (for
   * an incremental overlay redraw), or null if nothing changed. */
  stampAt(row: number, col: number, dtSeconds: number): Bbox | null {
    const R = this.radiusCells
    const x0 = Math.max(0, Math.floor(col - R))
    const x1 = Math.min(this.width - 1, Math.ceil(col + R))
    const y0 = Math.max(0, Math.floor(row - R))
    const y1 = Math.min(this.height - 1, Math.ceil(row + R))
    if (x1 < x0 || y1 < y0) return null

    if (this.tool === 'flatten' && this.flattenTarget === null) {
      let sum = 0
      let n = 0
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const d = Math.hypot(x - col, y - row)
          if (d > R) continue
          const idx = y * this.width + x
          sum += this.dem[idx] + this.demDelta[idx]
          n++
        }
      }
      this.flattenTarget = n > 0 ? sum / n : 0
    }

    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const d = Math.hypot(x - col, y - row)
        const wgt = brushWeight(d, R, this.softness)
        if (wgt <= 0) continue
        const idx = y * this.width + x
        if (!this.editable[idx]) continue

        this.recordOriginal(idx)

        switch (this.tool) {
          case 'paint':
            if (wgt > 0.5) this.material[idx] = this.activeMaterialId
            break
          case 'eraser':
            if (wgt > 0.5) this.material[idx] = 0
            break
          case 'raise':
            this.demDelta[idx] = clamp(this.demDelta[idx] + this.strengthPerSec * wgt * dtSeconds, -DEM_DELTA_LIMIT, DEM_DELTA_LIMIT)
            break
          case 'lower':
            this.demDelta[idx] = clamp(this.demDelta[idx] - this.strengthPerSec * wgt * dtSeconds, -DEM_DELTA_LIMIT, DEM_DELTA_LIMIT)
            break
          case 'flatten': {
            const target = this.flattenTarget ?? 0
            const cur = this.dem[idx] + this.demDelta[idx]
            this.demDelta[idx] = clamp(this.demDelta[idx] + (target - cur) * 0.15 * wgt, -DEM_DELTA_LIMIT, DEM_DELTA_LIMIT)
            break
          }
          case 'smooth': {
            let sum = 0
            let n = 0
            for (let ny = y - 1; ny <= y + 1; ny++) {
              for (let nx = x - 1; nx <= x + 1; nx++) {
                if (nx < 0 || nx >= this.width || ny < 0 || ny >= this.height) continue
                sum += this.demDelta[ny * this.width + nx]
                n++
              }
            }
            const avg = n > 0 ? sum / n : this.demDelta[idx]
            this.demDelta[idx] = clamp(this.demDelta[idx] * (1 - wgt) + avg * wgt, -DEM_DELTA_LIMIT, DEM_DELTA_LIMIT)
            break
          }
        }
      }
    }
    this.growBbox(x0, y0, x1, y1)
    return { minX: x0, minY: y0, maxX: x1, maxY: y1 }
  }

  /** Interpolates from the last stamped point to (row, col) so a fast drag
   * leaves no gaps, stepping at most radius/2 cells apart. */
  moveTo(row: number, col: number, dtSeconds: number): Bbox | null {
    if (!this.lastPoint) {
      this.lastPoint = { row, col }
      return this.stampAt(row, col, dtSeconds)
    }
    const dx = col - this.lastPoint.col
    const dy = row - this.lastPoint.row
    const dist = Math.hypot(dx, dy)
    const step = Math.max(this.radiusCells / 2, 0.5)
    const n = Math.max(1, Math.ceil(dist / step))
    let rect: Bbox | null = null
    for (let i = 1; i <= n; i++) {
      const t = i / n
      const r = this.stampAt(this.lastPoint.row + dy * t, this.lastPoint.col + dx * t, dtSeconds / n)
      if (r) rect = rect ? unionBbox(rect, r) : r
    }
    this.lastPoint = { row, col }
    return rect
  }

  /** Finalizes the stroke: returns the dense before/after patch (or null if
   * nothing was touched) covering just the bounding box of touched cells. */
  finish(): StrokeRecord | null {
    if (this.touched.size === 0 || !this.bbox) return null
    const { minX, minY, maxX, maxY } = this.bbox
    const w = maxX - minX + 1
    const h = maxY - minY + 1

    let afterMaterial: Uint16Array | undefined
    let beforeMaterial: Uint16Array | undefined
    if (this.touchesMaterial()) {
      afterMaterial = new Uint16Array(w * h)
      for (let y = 0; y < h; y++) {
        afterMaterial.set(this.material.subarray((minY + y) * this.width + minX, (minY + y) * this.width + minX + w), y * w)
      }
      beforeMaterial = afterMaterial.slice()
      for (const [idx, orig] of this.touched) {
        if (orig.mat === undefined) continue
        const x = idx % this.width
        const y = Math.floor(idx / this.width)
        beforeMaterial[(y - minY) * w + (x - minX)] = orig.mat
      }
    }

    let afterDemDelta: Float32Array | undefined
    let beforeDemDelta: Float32Array | undefined
    if (this.touchesDem()) {
      afterDemDelta = new Float32Array(w * h)
      for (let y = 0; y < h; y++) {
        afterDemDelta.set(this.demDelta.subarray((minY + y) * this.width + minX, (minY + y) * this.width + minX + w), y * w)
      }
      beforeDemDelta = afterDemDelta.slice()
      for (const [idx, orig] of this.touched) {
        if (orig.delta === undefined) continue
        const x = idx % this.width
        const y = Math.floor(idx / this.width)
        beforeDemDelta[(y - minY) * w + (x - minX)] = orig.delta
      }
    }

    return { x: minX, y: minY, w, h, beforeMaterial, afterMaterial, beforeDemDelta, afterDemDelta }
  }
}

function clamp(v: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, v))
}

function unionBbox(a: Bbox, b: Bbox): Bbox {
  return {
    minX: Math.min(a.minX, b.minX),
    minY: Math.min(a.minY, b.minY),
    maxX: Math.max(a.maxX, b.maxX),
    maxY: Math.max(a.maxY, b.maxY),
  }
}

/** Writes a StrokeRecord's before/after sub-rect into the live arrays
 * (material/demDelta), for undo/redo. Does not touch the server. */
function applyRecord(
  material: Uint16Array,
  demDelta: Float32Array,
  width: number,
  rec: StrokeRecord,
  which: 'before' | 'after',
) {
  const mat = which === 'before' ? rec.beforeMaterial : rec.afterMaterial
  const delta = which === 'before' ? rec.beforeDemDelta : rec.afterDemDelta
  for (let y = 0; y < rec.h; y++) {
    const rowOff = (rec.y + y) * width + rec.x
    if (mat) material.set(mat.subarray(y * rec.w, y * rec.w + rec.w), rowOff)
    if (delta) demDelta.set(delta.subarray(y * rec.w, y * rec.w + rec.w), rowOff)
  }
}

const MAX_HISTORY = 50

/** Client-side undo/redo stacks of StrokeRecords. Undo/redo both mutate the
 * live arrays locally AND replay through the normal patch endpoint, so the
 * server stays authoritative with no separate undo machinery server-side. */
export class UndoManager {
  private undoStack: StrokeRecord[] = []
  private redoStack: StrokeRecord[] = []

  constructor(
    private designName: () => string | null,
    private material: () => Uint16Array,
    private demDelta: () => Float32Array,
    private width: () => number,
    private onApplied: (rect: Bbox) => void,
  ) {}

  counts() {
    return { undo: this.undoStack.length, redo: this.redoStack.length }
  }

  push(rec: StrokeRecord) {
    this.undoStack.push(rec)
    if (this.undoStack.length > MAX_HISTORY) this.undoStack.shift()
    this.redoStack = []
  }

  async undo() {
    const rec = this.undoStack.pop()
    if (!rec) return
    applyRecord(this.material(), this.demDelta(), this.width(), rec, 'before')
    this.onApplied({ minX: rec.x, minY: rec.y, maxX: rec.x + rec.w - 1, maxY: rec.y + rec.h - 1 })
    this.redoStack.push(rec)
    const name = this.designName()
    if (name) await patchDesign(name, { x: rec.x, y: rec.y, w: rec.w, h: rec.h, material: rec.beforeMaterial, demDelta: rec.beforeDemDelta })
  }

  async redo() {
    const rec = this.redoStack.pop()
    if (!rec) return
    applyRecord(this.material(), this.demDelta(), this.width(), rec, 'after')
    this.onApplied({ minX: rec.x, minY: rec.y, maxX: rec.x + rec.w - 1, maxY: rec.y + rec.h - 1 })
    this.undoStack.push(rec)
    const name = this.designName()
    if (name) await patchDesign(name, { x: rec.x, y: rec.y, w: rec.w, h: rec.h, material: rec.afterMaterial, demDelta: rec.afterDemDelta })
  }

  clear() {
    this.undoStack = []
    this.redoStack = []
  }
}

// ---- overlay rendering ----

export function materialColorTable(materials: MaterialDef[]): Map<number, [number, number, number]> {
  const m = new Map<number, [number, number, number]>()
  for (const mat of materials) {
    const hex = mat.color.replace('#', '')
    const r = parseInt(hex.slice(0, 2), 16)
    const g = parseInt(hex.slice(2, 4), 16)
    const b = parseInt(hex.slice(4, 6), 16)
    m.set(mat.id, [r, g, b])
  }
  return m
}

export function renderMaterialRect(material: Uint16Array, width: number, colors: Map<number, [number, number, number]>, rect: Bbox): ImageData {
  const w = rect.maxX - rect.minX + 1
  const h = rect.maxY - rect.minY + 1
  const data = new Uint8ClampedArray(w * h * 4)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (rect.minY + y) * width + (rect.minX + x)
      const cls = material[idx]
      const o = (y * w + x) * 4
      if (cls === 0) {
        data[o + 3] = 0
        continue
      }
      const rgb = colors.get(cls)
      if (!rgb) {
        data[o + 3] = 0
        continue
      }
      data[o] = rgb[0]
      data[o + 1] = rgb[1]
      data[o + 2] = rgb[2]
      data[o + 3] = 140 // ~0.55 alpha
    }
  }
  return new ImageData(data, w, h)
}

const DELETE_BAND_ROWS = 300 // keeps each chunk's cell count under MAX_PATCH_CELLS on any grid width used here

/** Repaints every cell of `materialId` to 0 (base terrain), everywhere in
 * the grid. Mutates `material` in place and replays the change to the
 * server as a sequence of row-band patches, since a single request for the
 * whole grid would exceed the server's per-patch cell cap. */
export async function deleteMaterialEverywhere(designName: string, material: Uint16Array, width: number, height: number, materialId: number) {
  for (let y0 = 0; y0 < height; y0 += DELETE_BAND_ROWS) {
    const y1 = Math.min(height, y0 + DELETE_BAND_ROWS)
    const h = y1 - y0
    let touched = false
    const band = new Uint16Array(width * h)
    for (let y = 0; y < h; y++) {
      const rowOff = (y0 + y) * width
      for (let x = 0; x < width; x++) {
        const v = material[rowOff + x]
        if (v === materialId) {
          material[rowOff + x] = 0
          touched = true
        }
        band[y * width + x] = material[rowOff + x]
      }
    }
    if (touched) {
      await patchDesign(designName, { x: 0, y: y0, w: width, h, material: band })
    }
  }
}

export function renderSculptRect(demDelta: Float32Array, width: number, rect: Bbox): ImageData {
  const w = rect.maxX - rect.minX + 1
  const h = rect.maxY - rect.minY + 1
  const data = new Uint8ClampedArray(w * h * 4)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (rect.minY + y) * width + (rect.minX + x)
      const v = demDelta[idx]
      const o = (y * w + x) * 4
      if (v === 0) {
        data[o + 3] = 0
        continue
      }
      // Any nonzero edit gets at least ~24% opacity so small sculpts (a few
      // cm) stay visible, ramping to full opacity by 1 m.
      const mag = Math.min(Math.abs(v), 1.0)
      const alpha = 60 + mag * 195
      if (v < 0) {
        data[o] = 40
        data[o + 1] = 110
        data[o + 2] = 220
      } else {
        data[o] = 235
        data[o + 1] = 140
        data[o + 2] = 40
      }
      data[o + 3] = alpha
    }
  }
  return new ImageData(data, w, h)
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t
}

/** Blue (dug) -> white (unchanged) -> red (raised) diverging ramp, t in [-1, 1]. */
function divergingColor(t: number): [number, number, number] {
  const c = Math.max(-1, Math.min(1, t))
  if (c < 0) {
    const k = 1 + c // 0 at t=-1, 1 at t=0
    return [lerp(20, 255, k), lerp(80, 255, k), lerp(190, 255, k)]
  }
  const k = c // 0 at t=0, 1 at t=1
  return [lerp(255, 195, k), lerp(255, 60, k), lerp(255, 30, k)]
}

/** Dedicated elevation-change heatmap: a proper diverging colormap scaled to
 * `maxAbs` (the current largest |dem_delta| in the design), so the full
 * color range is always used regardless of how small the edits are. */
export function renderElevationHeatmapRect(demDelta: Float32Array, width: number, rect: Bbox, maxAbs: number): ImageData {
  const w = rect.maxX - rect.minX + 1
  const h = rect.maxY - rect.minY + 1
  const data = new Uint8ClampedArray(w * h * 4)
  const scale = Math.max(maxAbs, 0.02)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (rect.minY + y) * width + (rect.minX + x)
      const v = demDelta[idx]
      const o = (y * w + x) * 4
      if (v === 0) {
        data[o + 3] = 0
        continue
      }
      const [r, g, b] = divergingColor(v / scale)
      data[o] = r
      data[o + 1] = g
      data[o + 2] = b
      data[o + 3] = 235
    }
  }
  return new ImageData(data, w, h)
}

/** Full-array scan for the largest |dem_delta| — call at "settle" points
 * (design open, stroke end, undo/redo), not every animation frame. */
export function maxAbsDelta(demDelta: Float32Array): number {
  let m = 0
  for (let i = 0; i < demDelta.length; i++) {
    const a = Math.abs(demDelta[i])
    if (a > m) m = a
  }
  return m
}

/** Cheap scan of just a dirty rect, for live updates during an active drag
 * without re-scanning the whole grid on every frame. */
export function maxAbsDeltaInRect(demDelta: Float32Array, width: number, rect: Bbox): number {
  let m = 0
  for (let y = rect.minY; y <= rect.maxY; y++) {
    const rowOff = y * width
    for (let x = rect.minX; x <= rect.maxX; x++) {
      const a = Math.abs(demDelta[rowOff + x])
      if (a > m) m = a
    }
  }
  return m
}
