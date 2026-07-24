import type { MaterialDef, Meta } from './store'

export async function fetchMeta(): Promise<Meta> {
  const r = await fetch('/api/meta')
  if (!r.ok) throw new Error(`GET /api/meta -> ${r.status}`)
  return r.json()
}

export async function fetchDem(meta: Meta): Promise<Float32Array> {
  const r = await fetch('/api/terrain/dem.bin')
  if (!r.ok) throw new Error(`GET /api/terrain/dem.bin -> ${r.status}`)
  const buf = await r.arrayBuffer()
  const q = new Uint16Array(buf)
  const out = new Float32Array(q.length)
  for (let i = 0; i < q.length; i++) {
    out[i] = meta.dem_min + q[i] * meta.dem_scale
  }
  return out
}

export async function fetchMasksRgba(width: number, height: number): Promise<Uint8ClampedArray> {
  const r = await fetch('/api/terrain/masks.png')
  if (!r.ok) throw new Error(`GET /api/terrain/masks.png -> ${r.status}`)
  const blob = await r.blob()
  const bitmap = await createImageBitmap(blob)
  const canvas = new OffscreenCanvas(width, height)
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(bitmap, 0, 0)
  const data = ctx.getImageData(0, 0, width, height).data
  return data
}

/** Zone tint + 1px boundary outline, computed from the packed masks RGBA
 * (G channel = zone, A channel = valid). Returned as a straight RGBA buffer
 * ready for putImageData. */
export function computeZoneOverlay(masksRgba: Uint8ClampedArray, width: number, height: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(width * height * 4)
  const zoneAt = (x: number, y: number) => masksRgba[(y * width + x) * 4 + 1] > 127

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x
      const inZone = zoneAt(x, y)
      if (!inZone) continue
      const boundary =
        (x > 0 && !zoneAt(x - 1, y)) ||
        (x < width - 1 && !zoneAt(x + 1, y)) ||
        (y > 0 && !zoneAt(x, y - 1)) ||
        (y < height - 1 && !zoneAt(x, y + 1))
      const o = idx * 4
      out[o] = 46
      out[o + 1] = 139
      out[o + 2] = 87
      out[o + 3] = boundary ? 255 : 40
    }
  }
  return out
}

/** Covers the "no survey data" area (outside the actual cut polygon, inside
 * its bounding box) with a flat fill instead of the raw ortho export's baked-
 * in black pixels, so it reads as "nothing here" rather than a broken image.
 * A-channel of masksRgba is the valid mask. */
export function computeNodataOverlay(masksRgba: Uint8ClampedArray, width: number, height: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(width * height * 4)
  for (let i = 0; i < width * height; i++) {
    const valid = masksRgba[i * 4 + 3] > 127
    const o = i * 4
    if (valid) {
      out[o + 3] = 0
    } else {
      // must exactly match MapView's container background (#111) so the
      // nodata region blends in seamlessly with no visible box outline.
      out[o] = 17
      out[o + 1] = 17
      out[o + 2] = 17
      out[o + 3] = 255
    }
  }
  return out
}

export function utmToPixel(meta: Meta, x: number, y: number) {
  const col = (x - meta.transform.minx) / meta.transform.res
  const row = (meta.transform.maxy - y) / meta.transform.res
  return { col, row }
}

export function pixelToUtm(meta: Meta, col: number, row: number) {
  const x = meta.transform.minx + col * meta.transform.res
  const y = meta.transform.maxy - row * meta.transform.res
  return { x, y }
}

export function materialDepressionTable(materials: MaterialDef[]): Map<number, number> {
  const m = new Map<number, number>()
  for (const mat of materials) m.set(mat.id, mat.depression_m)
  return m
}

/** The surface the solver actually sees: base DEM minus each material's
 * detention depression, plus the user's sculpted delta — mirrors
 * scripts/bake_corridor.py's ordering (depression first, then dem_delta) so
 * the 3D view matches what a run will use. */
export function computeEffectiveZ(
  dem: Float32Array,
  demDelta: Float32Array,
  material: Uint16Array,
  depression: Map<number, number>,
): Float32Array {
  const n = dem.length
  const out = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const cls = material[i]
    const dep = cls === 0 ? 0 : depression.get(cls) ?? 0
    out[i] = dem[i] - dep + demDelta[i]
  }
  return out
}
