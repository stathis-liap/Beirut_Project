import { create } from 'zustand'
import type { UndoManager } from './editor'

export interface Transform {
  crs: string
  minx: number
  miny: number
  maxy: number
  res: number
  width: number
  height: number
}

export interface Gauge {
  name: string
  row: number
  col: number
}

export interface Meta {
  transform: Transform
  width: number
  height: number
  res: number
  gauges: Gauge[]
  storms: string[]
  dem_min: number
  dem_scale: number
  hazard_bounds: number[]
}

export interface ViewTransform {
  scale: number
  ox: number
  oy: number
}

export interface MaterialDef {
  id: number
  label: string
  color: string
  infil_mmh: number
  manning_n: number
  depression_m: number
  builtin: boolean
}

export interface DesignSummary {
  name: string
  modified: string
  notes: string
  unlocked: boolean
}

export type Tool = 'paint' | 'eraser' | 'raise' | 'lower' | 'flatten' | 'smooth'
export type OverlayMode = 'materials' | 'elevation'

export interface RunProgress {
  runId: string
  storm: string
  state: 'queued' | 'running' | 'done' | 'error'
  pct?: number
  t?: number
  duration?: number
  eta_s?: number | null
  error?: string
}

interface SandboxState {
  meta: Meta | null
  dem: Float32Array | null
  // raw decoded RGBA from /api/terrain/masks.png: R=building, G=zone, B=water, A=valid
  masksRgba: Uint8ClampedArray | null
  view: ViewTransform
  setMeta: (m: Meta) => void
  setDem: (d: Float32Array) => void
  setMasksRgba: (m: Uint8ClampedArray) => void
  setView: (v: Partial<ViewTransform>) => void

  // ---- design editing state ----
  designList: DesignSummary[]
  designName: string | null
  material: Uint16Array | null
  demDelta: Float32Array | null
  materials: MaterialDef[]
  unlocked: boolean
  dirty: boolean
  layerVersion: number // bumped on bulk edits (e.g. delete-material), to force a full 2D canvas redraw
  editVersion: number // bumped on every stroke/undo/redo, for the 3D view to resync (cheaper than layerVersion's full 2D redraw)
  activeMaterialId: number
  tool: Tool
  brushRadiusM: number
  brushStrengthMps: number
  brushSoftness: number
  undoCount: number
  redoCount: number
  undo: (() => void) | null
  redo: (() => void) | null
  // the live UndoManager instance, shared so both the 2D and 3D views can
  // push/undo/redo strokes against the same history for the open design.
  undoManager: UndoManager | null
  overlayMode: OverlayMode
  elevationRangeM: number // current max |dem_delta| in the open design, for the heatmap legend

  // Shared run-progress summary so a floating banner can show it regardless
  // of which tab (2D/3D/Results) is currently active.
  runProgress: RunProgress | null
  setRunProgress: (p: RunProgress | null) => void

  setDesignList: (l: DesignSummary[]) => void
  openDesign: (name: string, material: Uint16Array, demDelta: Float32Array, materials: MaterialDef[], unlocked: boolean) => void
  closeDesign: () => void
  setUnlockedLocal: (u: boolean) => void
  setMaterialsList: (m: MaterialDef[]) => void
  setDirty: (d: boolean) => void
  bumpLayerVersion: () => void
  bumpEditVersion: () => void
  setActiveMaterialId: (id: number) => void
  setTool: (t: Tool) => void
  setBrushRadiusM: (r: number) => void
  setBrushStrengthMps: (s: number) => void
  setBrushSoftness: (s: number) => void
  setUndoRedoCounts: (undo: number, redo: number) => void
  setUndoRedoFns: (undo: (() => void) | null, redo: (() => void) | null) => void
  setUndoManager: (m: UndoManager | null) => void
  setOverlayMode: (m: OverlayMode) => void
  setElevationRangeM: (r: number) => void
}

export const useStore = create<SandboxState>((set) => ({
  meta: null,
  dem: null,
  masksRgba: null,
  view: { scale: 1, ox: 0, oy: 0 },
  setMeta: (m) => set({ meta: m }),
  setDem: (d) => set({ dem: d }),
  setMasksRgba: (m) => set({ masksRgba: m }),
  setView: (v) => set((s) => ({ view: { ...s.view, ...v } })),

  designList: [],
  designName: null,
  material: null,
  demDelta: null,
  materials: [],
  unlocked: false,
  dirty: false,
  layerVersion: 0,
  editVersion: 0,
  activeMaterialId: 1,
  tool: 'paint',
  brushRadiusM: 4,
  brushStrengthMps: 0.15,
  brushSoftness: 0.6,
  undoCount: 0,
  redoCount: 0,
  undo: null,
  redo: null,
  undoManager: null,
  overlayMode: 'materials',
  elevationRangeM: 0,
  runProgress: null,

  setDesignList: (l) => set({ designList: l }),
  openDesign: (name, material, demDelta, materials, unlocked) =>
    set({
      designName: name,
      material,
      demDelta,
      materials,
      unlocked,
      dirty: false,
      layerVersion: 0,
      undoCount: 0,
      redoCount: 0,
    }),
  closeDesign: () =>
    set({
      designName: null,
      material: null,
      demDelta: null,
      materials: [],
      unlocked: false,
      dirty: false,
      undoCount: 0,
      redoCount: 0,
    }),
  setUnlockedLocal: (u) => set({ unlocked: u }),
  setMaterialsList: (m) => set({ materials: m }),
  setDirty: (d) => set({ dirty: d }),
  bumpLayerVersion: () => set((s) => ({ layerVersion: s.layerVersion + 1, editVersion: s.editVersion + 1 })),
  bumpEditVersion: () => set((s) => ({ editVersion: s.editVersion + 1 })),
  setActiveMaterialId: (id) => set({ activeMaterialId: id }),
  setTool: (t) => set({ tool: t }),
  setBrushRadiusM: (r) => set({ brushRadiusM: r }),
  setBrushStrengthMps: (s) => set({ brushStrengthMps: s }),
  setBrushSoftness: (s) => set({ brushSoftness: s }),
  setUndoRedoCounts: (undo, redo) => set({ undoCount: undo, redoCount: redo }),
  setUndoRedoFns: (undo, redo) => set({ undo, redo }),
  setUndoManager: (m) => set({ undoManager: m }),
  setOverlayMode: (m) => set({ overlayMode: m }),
  setElevationRangeM: (r) => set({ elevationRangeM: r }),
  setRunProgress: (p) => set({ runProgress: p }),
}))
