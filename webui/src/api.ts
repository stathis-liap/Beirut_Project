import type { MaterialDef } from './store'

export interface DesignJson {
  design: {
    name: string
    notes: string
    created: string
    modified: string
    base_terrain: string
    unlocked: boolean
  }
  materials: { materials: MaterialDef[] }
}

export interface DesignSummary {
  name: string
  modified: string
  notes: string
  unlocked: boolean
}

function toBase64(bytes: Uint8Array): string {
  let binary = ''
  const chunk = 0x8000
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk))
  }
  return btoa(binary)
}

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) {
    const detail = await r.json().catch(() => ({ detail: r.statusText }))
    throw new Error(detail.detail || `HTTP ${r.status}`)
  }
  return r.json()
}

export async function listDesigns(): Promise<DesignSummary[]> {
  return json(await fetch('/api/designs'))
}

export async function createDesign(name: string, template: string): Promise<DesignJson> {
  return json(
    await fetch('/api/designs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, template }),
    }),
  )
}

export async function getDesign(name: string): Promise<DesignJson> {
  return json(await fetch(`/api/designs/${encodeURIComponent(name)}`))
}

export async function getDesignMaterial(name: string): Promise<Uint16Array> {
  const r = await fetch(`/api/designs/${encodeURIComponent(name)}/material.bin`)
  if (!r.ok) throw new Error(`GET material.bin -> ${r.status}`)
  return new Uint16Array(await r.arrayBuffer())
}

export async function getDesignDemDelta(name: string): Promise<Float32Array> {
  const r = await fetch(`/api/designs/${encodeURIComponent(name)}/dem_delta.bin`)
  if (!r.ok) throw new Error(`GET dem_delta.bin -> ${r.status}`)
  return new Float32Array(await r.arrayBuffer())
}

export interface PatchRect {
  x: number
  y: number
  w: number
  h: number
  material?: Uint16Array
  demDelta?: Float32Array
}

export async function patchDesign(name: string, p: PatchRect): Promise<number> {
  const body: Record<string, unknown> = { x: p.x, y: p.y, w: p.w, h: p.h }
  if (p.material) body.material_b64 = toBase64(new Uint8Array(p.material.buffer, p.material.byteOffset, p.material.byteLength))
  if (p.demDelta) body.dem_delta_b64 = toBase64(new Uint8Array(p.demDelta.buffer, p.demDelta.byteOffset, p.demDelta.byteLength))
  const r = await json<{ applied_cells: number }>(
    await fetch(`/api/designs/${encodeURIComponent(name)}/patch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  )
  return r.applied_cells
}

export async function putMaterials(name: string, materials: MaterialDef[]): Promise<MaterialDef[]> {
  const r = await json<{ materials: MaterialDef[] }>(
    await fetch(`/api/designs/${encodeURIComponent(name)}/materials`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ materials }),
    }),
  )
  return r.materials
}

export async function putLock(name: string, unlocked: boolean): Promise<void> {
  await json(
    await fetch(`/api/designs/${encodeURIComponent(name)}/lock`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ unlocked }),
    }),
  )
}

export async function saveDesign(name: string): Promise<void> {
  await json(await fetch(`/api/designs/${encodeURIComponent(name)}/save`, { method: 'POST' }))
}

export async function deleteDesign(name: string): Promise<void> {
  const r = await fetch(`/api/designs/${encodeURIComponent(name)}`, { method: 'DELETE' })
  if (!r.ok && r.status !== 204) throw new Error(`DELETE design -> ${r.status}`)
}

export interface RunSummary {
  run_id: string
  design: string
  storm: string
  created: string
  label: string
  notes: string
  finished: boolean
  closure_rel?: number
  vol_rain_m3?: number
  vol_infiltrated_m3?: number
  vol_outflow_m3?: number
  vol_stored_end_m3?: number
}

export interface RunMetrics {
  flooded_streets_m2: number
  flooded_on_ribbon_m2: number
  flooded_0_25m_m2: number
  flooded_25_50m_m2: number
  flooded_50_100m_m2: number
  p99_depth_ribbon_cm: number
  mean_depth_ribbon_cm: number
  vol_rain_m3: number
  vol_infiltrated_m3: number
  infil_pct: number
  vol_outflow_m3: number
  vol_stored_end_m3: number
  closure_rel: number
  default_vmax_m: number
}

export interface RunDetail {
  run: RunSummary & { storm_json: unknown; design_sha256: string }
  meta: Record<string, unknown>
  metrics: RunMetrics
}

export async function listRuns(): Promise<RunSummary[]> {
  return json(await fetch('/api/runs'))
}

export async function getRun(runId: string): Promise<RunDetail> {
  return json(await fetch(`/api/runs/${encodeURIComponent(runId)}`))
}

export async function getRunFrames(runId: string): Promise<number[]> {
  return json(await fetch(`/api/runs/${encodeURIComponent(runId)}/frames`))
}

export async function patchRun(runId: string, patch: { label?: string; notes?: string }): Promise<void> {
  await json(
    await fetch(`/api/runs/${encodeURIComponent(runId)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    }),
  )
}

export async function deleteRun(runId: string): Promise<void> {
  const r = await fetch(`/api/runs/${encodeURIComponent(runId)}`, { method: 'DELETE' })
  if (!r.ok && r.status !== 204) throw new Error(`DELETE run -> ${r.status}`)
}

export interface CompareDelta {
  a: number
  b: number
  delta: number
  delta_pct: number | null
}

export interface CompareResult {
  a: RunMetrics
  b: RunMetrics
  deltas: Record<string, CompareDelta>
}

export async function getCompare(a: string, b: string): Promise<CompareResult> {
  return json(await fetch(`/api/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`))
}
