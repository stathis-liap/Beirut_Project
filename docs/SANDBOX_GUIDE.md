# Interactive Green-Corridor Sandbox — Implementation Guide

This document is a complete, self-contained build guide for an interactive
web sandbox on top of the existing flood pipeline. Follow it phase by phase,
in order. Every phase ends with a **checkpoint** — a concrete thing you can
run and see working. Do not start a phase before the previous checkpoint
passes.

## 0. What you are building

A browser tool where a **non-technical** city planner or civil engineer can:

1. open a 2D map (and 3D view) of the corridor terrain;
2. **sculpt** the ground with a brush — dig pits, raise mounds, flatten,
   smooth (i.e. add/remove material in the Z axis);
3. **paint materials** (porous sidewalk, bioswale, bioretention pond, …) with
   an adjustable brush, erase them, create **custom materials** and edit any
   material's properties (infiltration mm/h, Manning roughness, detention
   depth);
4. check the result in 2D and 3D;
5. save the design, choose rain (preset storms or a hand-drawn hyetograph),
   press **Run**, watch live progress;
6. view results (animated flood depth, max depth, hazard), save runs, and
   compare any two runs side by side.

The target user story: *"I tweak the corridor's shape and materials with
pick-and-place controls, confirm visually, run the rain, and compare against
my previous attempt."*

## 1. Ground rules

- **Never modify the base terrain files** in `output/terrain_cut_0.5/`. All
  edits live in a *design* (a delta on top of the base). The base DEM is
  read-only forever.
- **Reuse the existing pipeline code by importing it** — the solver, the bake
  logic, the coordinate transforms. Do not fork or copy-paste the solver.
- The Python interpreter is `/home/stathisliap/Work/.venv/bin/python`
  (plain `python` is NOT on PATH). numpy is 2.x (no `np.trapz`),
  `pkg_resources` is absent. torch + CUDA are available.
- Elevations are **ellipsoidal** (sea sits near +26 m). Never threshold
  elevation against 0.
- Only ONE simulation may run at a time (one GPU). Enforce it with a queue.
- Keep every server endpoint fast (< 200 ms) except the sim itself, which is
  a background job.
- All user-facing text must be plain language — the target user does not know
  what "Manning n" is; label it "surface roughness (higher = slower flow)".

## 2. Architecture

```
┌──────────────────────────────┐        REST + WebSocket        ┌──────────────────────────────┐
│  Browser (React + TS + Vite) │ <────────────────────────────> │  FastAPI backend (sandbox/)  │
│  - 2D canvas editor (paint/  │   GET terrain, PNG overlays    │  - state.py  designs + edits │
│    sculpt/erase, undo/redo)  │   POST stroke patches          │  - baking.py design→terrain  │
│  - three.js 3D viewer        │   POST run, WS progress        │  - jobs.py   GPU job queue   │
│  - rain editor, run panel,   │   GET result PNGs + metrics    │  - encode.py npy→PNG/binary  │
│    results & compare views   │                                │  - metrics.py run metrics    │
└──────────────────────────────┘                                └──────────────┬───────────────┘
                                                                               │ imports
                                                     scripts/flood_gpu.py  simulate()
                                                     scripts/bake logic, las_common, …
                                                                               │ reads/writes
                                                                output/sandbox/{designs,runs}/
```

Directory layout to create (repo root):

```
sandbox/                  # Python package (backend)
  __init__.py
  server.py               # FastAPI app; also serves the built frontend (webui/dist)
  state.py                # design load/save, patch application, edit-mask enforcement
  baking.py               # design -> effective terrain arrays (in memory)
  jobs.py                 # single-worker sim queue + progress relay
  encode.py               # numpy -> PNG / quantized binary for the browser
  metrics.py              # per-run and A-vs-B metrics
webui/                    # Vite + React + TypeScript frontend
output/sandbox/
  designs/<design-name>/  # saved designs (see §3.2)
  runs/<run-id>/          # sim outputs, same file format flood_gpu.py already writes
  storms/                 # user-created storm JSONs (same schema as storms/)
```

## 3. Data contracts

Get these right first; everything else hangs off them.

### 3.1 The base terrain (existing, read-only)

`output/terrain_cut_0.5/` contains, all on the same 2348 × 1142 grid at
0.5 m (row 0 = north):

| File | Contents |
|---|---|
| `dem.npy` | float32 ground elevation (ellipsoidal m); NaN outside survey |
| `dem_transform.json` | `{crs, minx, miny, maxy, res, width, height}` (EPSG:32636) |
| `masks.npz` | boolean `valid`, `building`, `water`, `courtyard` |
| `manning.npy` | float32 roughness |
| `infil_mmh.npy` | float32 infiltration (mm/h) |
| `rain_weight.npy` | float32 rain rerouting weights (roofs → streets) |
| `fillbound.npy` | static ponding depth bound (used for QA display only here) |
| `gauges.json` | `[{name, x, y}]` in UTM |
| `ortho.png`, `hillshade.png` | base imagery, same grid |

Load it with the existing `scripts/flood_gpu.py::load_terrain(dir)` which
returns `(dem, transform, masks, manning, infil, rain_weight, gauges)`.
Pixel↔UTM: `scripts/las_common.py::utm_to_pixel(t, x, y) -> (col, row)`.

### 3.2 A design (the sandbox's central object)

`output/sandbox/designs/<name>/`:

| File | Contents |
|---|---|
| `dem_delta.npy` | float32, same grid. Metres of material **added (+) / removed (−)** by sculpting. Starts all-zero. |
| `material.npy` | uint16, same grid. Material class per cell; **0 = untouched base terrain**. Starts from the official corridor raster (see below) or all-zero. |
| `materials.json` | the editable material table, §3.3 |
| `design.json` | `{name, notes, created, modified, base_terrain: "output/terrain_cut_0.5", unlocked: false}` |

The starting template "official corridor" is
`output/corridor_gi_cut/material.npy` (uint8 classes 1–7) cast to uint16.

### 3.3 materials.json

Seed from `scripts/bake_corridor.py::PROPS`
(`{class: (infil_mmh, manning_n, depression_m, label)}`):

```json
{
  "materials": [
    {"id": 1, "label": "vehicular lane",   "color": "#8a8a8a", "infil_mmh": 5.0,   "manning_n": 0.016, "depression_m": 0.00, "builtin": true},
    {"id": 2, "label": "porous bikelane",  "color": "#c26fd4", "infil_mmh": 150.0, "manning_n": 0.020, "depression_m": 0.00, "builtin": true},
    {"id": 3, "label": "bioswale",         "color": "#2e8b57", "infil_mmh": 200.0, "manning_n": 0.150, "depression_m": 0.15, "builtin": true},
    {"id": 4, "label": "porous sidewalk",  "color": "#d9b38c", "infil_mmh": 150.0, "manning_n": 0.020, "depression_m": 0.00, "builtin": true},
    {"id": 5, "label": "garden",           "color": "#7ec850", "infil_mmh": 100.0, "manning_n": 0.100, "depression_m": 0.00, "builtin": true},
    {"id": 6, "label": "bioretention pond","color": "#1f78b4", "infil_mmh": 250.0, "manning_n": 0.200, "depression_m": 0.40, "builtin": true},
    {"id": 7, "label": "terrace",          "color": "#b8860b", "infil_mmh": 100.0, "manning_n": 0.120, "depression_m": 0.10, "builtin": true}
  ]
}
```

Rules: all properties (including built-ins') are editable per design; custom
materials get `id >= 100` (never reuse a deleted id within a design);
deleting a material re-paints its cells to class 0. Clamp on save:
`0 <= infil_mmh <= 1000`, `0.01 <= manning_n <= 0.5`,
`0 <= depression_m <= 1.0`.

### 3.4 Storm schema (existing — reuse exactly)

`storms/*.json`:

```json
{"name": "t2", "steps": [[0, 300, 8.17], [300, 600, 9.33], ...],
 "total_mm": 25.9, "duration": 5400.0}
```

`steps` are `[t0_s, t1_s, mm_per_hour]` step functions. The custom-rain
editor must emit exactly this schema into `output/sandbox/storms/`.
`total_mm = sum((t1 - t0) * mmh for each step) / 3600`. Recompute it on
save; never trust the client.

### 3.5 A run

`output/sandbox/runs/<run_id>/` where
`run_id = f"{design}__{storm}__{YYYYmmdd-HHMMSS}"`. Contents are **exactly
what `flood_gpu.simulate()` already writes** (`depth_{t:06d}.npy` fp16
frames, `max_depth.npy`, `final_depth.npy`, `max_vel.npy`, `max_hazard.npy`,
`run_meta.json`, `gauges.csv`) **plus** a sandbox-written `run.json`:

```json
{"run_id": "...", "design": "myplan", "storm": "t2",
 "design_sha256": "<sha256 of dem_delta.npy + material.npy + materials.json bytes>",
 "storm_json": { ...full storm copied in... },
 "label": "my first try", "notes": "", "created": "..."}
```

The hash + embedded storm make every result reproducible and traceable even
if the design is later edited.

## 4. API reference

Build the backend to this table; the frontend codes against it. All binary
grids are **row-major, little-endian, row 0 = north**.

| Method + path | Body / params | Returns |
|---|---|---|
| `GET /api/meta` | — | `{transform, width, height, res, gauges:[{name,row,col}], storms:[...names], dem_min, dem_scale, hazard_bounds:[0.75,1.25,2.0]}` |
| `GET /api/terrain/ortho.png` | — | PNG (the file, as-is) |
| `GET /api/terrain/hillshade.png` | — | PNG |
| `GET /api/terrain/dem.bin` | — | uint16 quantized DEM (§6.2) |
| `GET /api/terrain/masks.png` | — | RGBA PNG: R=building, G=zone, B=water, A=valid (each 0/255) |
| `GET /api/designs` | — | `[{name, modified, notes, unlocked}]` |
| `POST /api/designs` | `{name, template: "official"\|"blank"\|<design>}` | created design.json |
| `GET /api/designs/{name}` | — | `{design, materials}` |
| `GET /api/designs/{name}/material.bin` | — | uint16 raster |
| `GET /api/designs/{name}/dem_delta.bin` | — | float32 raster |
| `POST /api/designs/{name}/patch` | §7.4 patch JSON | `{applied_cells}` |
| `PUT /api/designs/{name}/materials` | full materials.json | echo (clamped) |
| `PUT /api/designs/{name}/lock` | `{unlocked: bool}` | design.json |
| `POST /api/designs/{name}/save` | — | `{saved: true}` (flush in-memory → npy) |
| `DELETE /api/designs/{name}` | — | 204 |
| `GET /api/storms` / `GET /api/storms/{name}` | — | list / full storm JSON (built-in + custom) |
| `POST /api/storms` | full storm JSON | saved (custom dir, totals recomputed) |
| `POST /api/run` | `{design, storm, duration?, save_every?: 60}` | `{run_id, queued_behind}` |
| `WS /api/run/{run_id}/progress` | — | stream of `{t, duration, pct, storage_m3, outflow_m3, max_h, eta_s, state}` then `{state:"done", meta}` |
| `GET /api/runs` | — | `[run.json + {closure_rel, vol_*} from run_meta.json]` |
| `GET /api/runs/{id}` | — | `{run, meta, metrics}` (§10.2) |
| `GET /api/runs/{id}/frames` | — | `[t_seconds...]` available frame times |
| `GET /api/runs/{id}/frame/{t}.png` | `vmax?` | colormapped RGBA PNG of depth at t (§10.1) |
| `GET /api/runs/{id}/max_depth.png` / `hazard.png` | — | colormapped RGBA PNG |
| `GET /api/runs/{id}/gauges.csv` | — | the CSV |
| `PATCH /api/runs/{id}` | `{label?, notes?}` | run.json |
| `DELETE /api/runs/{id}` | — | 204 |
| `GET /api/compare` | `?a=<id>&b=<id>` | `{a, b, deltas}` metric table (§10.3) |
| `GET /api/compare/diff.png` | `?a&b` | diverging-colormap PNG of `max_depth_b − max_depth_a` |

Errors: JSON `{detail}` with 404 (unknown design/run), 409 (name exists /
run in progress on same design), 422 (bad payload).

## 5. Phase 0 — scaffold

1. Backend deps into the **existing** venv:
   `/home/stathisliap/Work/.venv/bin/pip install fastapi "uvicorn[standard]" pillow`
   (numpy/scipy/torch/matplotlib already there).
2. Frontend: `npm create vite@latest webui -- --template react-ts`, then in
   `webui/`: `npm i three zustand` and `npm i -D @types/three`.
3. `sandbox/server.py` minimal app:

```python
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Al-Masar Sandbox")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                   allow_methods=["*"], allow_headers=["*"])

@app.get("/api/health")
def health(): return {"ok": True}

dist = os.path.join(os.path.dirname(__file__), "..", "webui", "dist")
if os.path.isdir(dist):
    app.mount("/", StaticFiles(directory=dist, html=True), name="ui")
```

The `sys.path.insert` of `scripts/` is how the backend imports
`flood_gpu`, `las_common`, etc. — keep it at the top of `server.py` only.

4. Dev run (two terminals):
   `/home/stathisliap/Work/.venv/bin/uvicorn sandbox.server:app --reload --port 8008`
   and `cd webui && npm run dev`. In `webui/vite.config.ts` add a proxy so
   the frontend calls relative `/api/...`:

```ts
server: { proxy: { "/api": "http://localhost:8008" } }
```

5. Production run (the one command for end users):
   `cd webui && npm run build`, then
   `/home/stathisliap/Work/.venv/bin/uvicorn sandbox.server:app --port 8008`
   and open `http://localhost:8008`. Put both lines in `sandbox/run.sh`.

**Checkpoint 0:** `curl localhost:8008/api/health` → `{"ok":true}`; the Vite
page loads and can fetch `/api/health` through the proxy.

## 6. Phase 1 — terrain service (read-only)

### 6.1 Server-side terrain singleton

`sandbox/state.py`: load the base terrain **once at startup** into a module-
level object:

```python
from flood_gpu import load_terrain
from build_corridor_gi import rasterize, zone_rings

class Base:
    def __init__(self, terrain="output/terrain_cut_0.5"):
        (self.dem, self.t, self.masks, self.manning,
         self.infil, self.rain_w, self.gauges) = load_terrain(terrain)
        self.terrain = terrain
        rings = zone_rings("output/masar_zone_official.json")
        self.zone = rasterize(rings, self.t, self.dem.shape)
        self.editable_zone = self.zone & self.masks["valid"] & ~self.masks["building"]
        self.editable_full = self.masks["valid"] & ~self.masks["building"] & ~self.masks["water"]
```

(`zone_rings`/`rasterize` already exist in `scripts/build_corridor_gi.py`;
`rasterize` takes ~10–20 s on this grid — do it once, then cache
`zone.npy` under `output/sandbox/` and reload it on later startups.)

### 6.2 Quantized DEM for the browser

In `sandbox/encode.py`:

```python
def dem_bin(dem, valid):
    lo = float(np.nanmin(dem[valid])); hi = float(np.nanmax(dem[valid]))
    scale = (hi - lo) / 65535.0
    q = np.where(valid, np.round((dem - lo) / scale), 0).astype(np.uint16)
    return q.tobytes(), lo, scale
```

Serve the bytes with `media_type="application/octet-stream"`; expose
`dem_min = lo` and `dem_scale = scale` in `/api/meta`. Client reconstructs
`z = dem_min + q * dem_scale` (Float32Array). 2348 × 1142 × 2 B ≈ 5.4 MB —
fine as a single fetch. Add `Cache-Control: max-age=86400` (base never
changes).

### 6.3 masks.png

Pack the four booleans into RGBA channels with Pillow
(`Image.fromarray(rgba, "RGBA")`). The client reads it into an offscreen
canvas and extracts channels via `getImageData`.

### 6.4 Frontend map shell

- One zustand store (`webui/src/store.ts`) holding: meta, decoded arrays
  (`dem: Float32Array`, masks as Uint8Arrays), view transform
  `{scale, ox, oy}`, later the tool state and design arrays.
- A `MapView` component with a stack of absolutely-positioned canvases,
  all `width/height = grid` size, CSS-scaled by the view transform:
  1. ortho (drawn from `<img>`), 2. hillshade (globalAlpha ~0.35),
  3. zone outline (stroke the zone mask boundary; a simple way is
  drawing zone pixels to an offscreen canvas and using
  `ctx.filter`/edge detection — or just fill zone with 8% green tint plus
  a marching-squares outline if easy).
- Pan = drag with space or middle mouse; zoom = wheel around cursor
  (multiply `scale`, adjust `ox, oy`). Clamp scale to [0.1, 12].
- Coordinate helper used EVERYWHERE:
  `screenToCell(mx, my) = {col: (mx-ox)/scale, row: (my-oy)/scale}`.

**Checkpoint 1:** browser shows the orthophoto with hillshade shading and
the official zone tinted/outlined; pan and zoom feel smooth; a status bar
shows the cell + UTM coordinate under the cursor
(`x = minx + col*res + res/2`, `y = maxy − row*res − res/2`).

## 7. Phase 2 — the 2D editor

### 7.1 Design lifecycle (backend, `state.py`)

- In-memory registry `designs: dict[str, Design]`; a `Design` holds
  `material: np.uint16 array`, `dem_delta: np.float32 array`,
  `materials: dict`, `meta: dict`, `dirty: bool`.
- `POST /api/designs` template `"official"` → copy
  `output/corridor_gi_cut/material.npy` cast to uint16, zero `dem_delta`;
  `"blank"` → both zero; `<design>` → deep copy of that design.
- `POST .../save` writes the three files + `design.json` atomically (write
  to `*.tmp`, `os.replace`). Also autosave any dirty design every 60 s from
  a background thread, and on server shutdown.

### 7.2 Edit-mask enforcement (this is what makes the zone lock real)

Every patch is clamped **server-side**:

```python
mask = base.editable_full if design.meta["unlocked"] else base.editable_zone
sub = mask[y:y+h, x:x+w]
design.material[y:y+h, x:x+w][sub] = new_material[sub]        # only where allowed
design.dem_delta[y:y+h, x:x+w][sub] = new_delta[sub]
```

Also clamp `dem_delta` to ±3.0 m unconditionally. The client mirrors the
same mask for instant feedback (fetch it from `masks.png` G/A channels), but
the server is authoritative.

### 7.3 Client editing model

The client owns working copies of `material` (Uint16Array) and `dem_delta`
(Float32Array), fetched from `/api/designs/{name}/*.bin` on open. **All
brush math happens client-side at 60 fps**; the server only receives patches.

Brush stamp (per pointer-move, interpolate along the segment from the last
event so fast strokes leave no gaps — step at most `radius/2` cells):

```ts
// falloff weight for cell at distance d from brush centre, radius r, softness s∈[0,1]
const t = Math.max(0, (d / r - (1 - s)) / Math.max(s, 1e-6));
const wgt = d > r ? 0 : 1 - t * t * (3 - 2 * t);          // smoothstep
```

Tools (all respect the local editable mask):

| Tool | Effect per cell |
|---|---|
| **Paint material** | `material = activeClass` where `wgt > 0.5` (hard edge; "detail" = brush radius slider 0.5–30 m) |
| **Eraser** | `material = 0` where `wgt > 0.5` |
| **Raise / Lower** | `dem_delta += ±strength * wgt * dtFrame` (strength slider 0.05–0.5 m/s of hold) |
| **Flatten** | on stroke start capture `target = mean(dem_base + dem_delta)` under brush; then `dem_delta += (target − (base+delta)) * 0.15 * wgt` per event |
| **Smooth** | replace `dem_delta` with its 3×3 box blur, blended by `wgt` |

Redraw only the overlay canvases' dirty rectangle each frame:
- **material overlay**: color per class from `materials.json`, alpha 0.55;
- **sculpt tint**: negative delta → blue, positive → orange,
  `alpha = min(|delta|, 1.0) * 0.6`. Legend: "blue = dug, orange = filled".

### 7.4 Stroke patches

On `pointerup` (one patch per stroke), send the stroke's bounding rect:

```json
POST /api/designs/{name}/patch
{"x": 412, "y": 1180, "w": 55, "h": 40,
 "material_b64": "<base64 of uint16 LE w*h bytes>",     // optional
 "dem_delta_b64": "<base64 of float32 LE w*h bytes>"}    // optional
```

Include only the layer(s) the tool touched. Server decodes with
`np.frombuffer(base64.b64decode(s), dtype).reshape(h, w)`, applies §7.2,
marks dirty. Cap `w*h` at 1_000_000 cells (413).

### 7.5 Undo / redo

Client-side stack, max 50 entries. Before applying a stroke locally, copy
the stroke rect's previous `material` + `dem_delta` sub-arrays; undo =
restore them locally **and** send them as a normal patch (redo = the
inverse). No server-side undo machinery needed.

### 7.6 Material palette UI

Right-hand panel: swatch grid of materials (color + label), the active one
highlighted; below it, sliders/inputs for the active material's
`infiltration (mm per hour — how fast the ground drinks water)`,
`roughness (higher = slows water down)`,
`detention depth (how deep a hollow it forms, m)`; a "New material" button
(picks a free id ≥ 100, random distinct color, asks for a name); "Delete"
(with confirm; repaints its cells to 0 via one full-raster patch);
"Save properties" → `PUT .../materials`. A padlock toggle for the zone lock
(`PUT .../lock`) with the explanation text: *"Editing is limited to the
official Al-Masar zone. Unlock to experiment anywhere."*

**Checkpoint 2:** create a design from the official template; paint a
bioswale strip; dig a 1 m pit; erase part of the strip; undo/redo both;
press Save, hard-reload the page, reopen the design — everything persists.
Verify with
`python -c "import numpy as np; d=np.load('output/sandbox/designs/<n>/dem_delta.npy'); print(d.min(), d.max())"`.

## 8. Phase 3 — the 3D viewer

- `THREE.PlaneGeometry(width*res, height*res, width-1, height-1)`, rotate to
  XY-ground, set each vertex `z = dem_base + dem_delta − depression(material)`
  (the *effective* surface — compute a small `effectiveZ()` helper in the
  store shared with 2D; depression per class from `materials.json`).
  Full res is 2.68 M vertices — acceptable on the GPU machine; add a
  "performance mode" toggle that decimates by 2 (stride-2 sampling) for
  laptops.
- Invalid cells (`!valid`): set z = (min valid z − 2) so the mesh stays
  finite (same trick as `scripts/render_3d.py::fill_nodata`).
- Texture: ortho as the base map; the material overlay drawn to an offscreen
  canvas becomes a second texture blended in a small `ShaderMaterial`
  (or simpler: one merged canvas texture redrawn on edit — merged canvas is
  simpler; update via `texture.needsUpdate = true` on the dirty rect's
  containing redraw).
- `OrbitControls`, vertical exaggeration slider ×1–×3 (scale z only),
  sun-angle-ish `DirectionalLight` + ambient.
- Live sync: after each stroke, update the affected vertices' z
  (`geometry.attributes.position`, `needsUpdate = true`,
  `computeVertexNormals()` on the dirty region or throttled globally at
  ~4 Hz) and redraw the merged texture.
- Optional but specified: raycast the pointer against the mesh, convert the
  hit point to (row, col), and feed the SAME brush code as 2D — brushing
  directly on the 3D ground. Ship 2D-only first if time-boxed; add this
  after Checkpoint 5.

**Checkpoint 3:** dig a pit in the 2D view; switch to 3D (tab or split
view) — the pit is visible immediately, textured with the material colors;
orbit/zoom is smooth (> 30 fps full-res on the GPU machine).

## 9. Phase 4 — rain + run

### 9.1 The one solver modification

Add an optional progress callback to `scripts/flood_gpu.py::simulate` —
nothing else changes:

```python
def simulate(dem, res, steps, duration, out_dir, *, ..., limiter="scale",
             progress_cb=None):
```

and inside the existing per-save block (right where the `if progress:` print
happens, after `storage` is computed):

```python
if progress_cb is not None:
    progress_cb(t, duration, {"storage_m3": storage, "outflow_m3": vol_out,
                              "max_h": hmax, "wall_s": time.time() - t0_wall})
```

After the edit, run `/home/stathisliap/Work/.venv/bin/python
scripts/validate.py` — all analytic benchmarks must still pass (the change
is inert when `progress_cb is None`).

### 9.2 Baking (`sandbox/baking.py`)

Mirror `scripts/bake_corridor.py` exactly, in memory:

```python
def bake(base, design):
    dem   = base.dem.copy()
    man   = base.manning.copy()
    infil = base.infil.copy()
    mat   = design.material
    for m in design.materials["materials"]:
        cells = mat == m["id"]
        if not cells.any(): continue
        infil[cells] = m["infil_mmh"]
        man[cells]   = m["manning_n"]
        if m["depression_m"] > 0:
            dem[cells] -= m["depression_m"]
    dem = dem + design.dem_delta          # sculpting on top
    return dem, man, infil
```

Order matters: depression first, then `dem_delta` — identical totals to the
pipeline (`dem_eff = base − depression + delta`).

### 9.3 Job queue (`sandbox/jobs.py`)

- One `threading.Thread` worker consuming a `queue.Queue` of jobs; a
  registry `jobs: dict[run_id, Job]` where `Job` has
  `state ("queued"|"running"|"done"|"error")`, `latest: dict` (last
  progress payload), `error: str|None`.
- The worker: bake → write `run.json` (with the design sha256 and the full
  storm) → call

```python
simulate(dem, base.t["res"], storm["steps"], duration, run_dir,
         manning=man, infil_mmh=infil, valid=base.masks["valid"],
         water=base.masks["water"], rain_weight=base.rain_w,
         gauges=base.gauges, save_every=60.0, device="auto",
         save_frames=True, progress=False, progress_cb=job.update)
```

  `job.update(t, duration, stats)` just stores
  `{"t": t, "pct": 100*t/duration, "eta_s": stats["wall_s"]*(duration/max(t,1)-1), **stats}`
  into `job.latest` (plain dict assignment — thread-safe enough for a
  monotone status).
- The WebSocket handler (async side) **polls** `job.latest` every 0.5 s and
  sends it if changed; when `state == "done"` it sends the final
  `run_meta.json` and closes. Polling avoids all cross-thread asyncio
  hazards — do not try to call into the event loop from the worker thread.
- `POST /api/run` returns 409 if the same design is already queued/running;
  otherwise enqueues and returns the queue position.

### 9.4 Rain UI

- **Storm picker**: cards for the built-ins with plain labels — "Frequent
  storm (2-year, 26 mm)", "Observed 25 Nov 2025", "Severe (50-year)" — read
  names from `/api/storms`, details from each JSON.
- **Custom storm editor**: a bar chart of intensity (mm/h) per 5-minute
  step; drag a bar vertically to set it; inputs for duration (30 min–3 h)
  and total mm (rescales all bars proportionally); shape presets
  *uniform / front-loaded / peak in the middle* (the peak preset should
  imitate the built-in t2 shape: low shoulders, 2× blocks at ~1/3 in).
  "Save storm" → `POST /api/storms` (server recomputes `total_mm` and
  appends `"duration": last_t1 + 1800` — keep 30 min of post-storm drainage
  time like the built-ins).
- **Run panel**: design + storm summary, a Run button (disabled with reason
  while a job is queued/running), then a progress card: percent bar, sim
  clock vs storm duration, live storage and outflow numbers, ETA. On done:
  headline card with mass-balance closure (show
  `closure_rel` as "water accounting error: 0.00x %" — reassure the user it
  should be ~0).

**Checkpoint 4:** edit a design, run storm `t2` on it; the progress bar
moves, ETA is sane, and the finished run appears under `output/sandbox/runs/`
with `run_meta.json` closure ≤ 0.1 %. Also verify the **zero-edit
equivalence**: a design created from the "official" template with no edits,
run with the same storm/duration, must reproduce the pipeline's
`output/corridor_runs/after_t2` max depth to fp32 noise
(`np.abs(a-b).max() < 1e-3`, given identical solver settings). This proves
the sandbox bake equals `bake_corridor.py`.

## 10. Phase 5 — results & comparison

### 10.1 Colormapped PNGs (`sandbox/encode.py`)

```python
def depth_png(depth, vmax, alpha_below=0.01):
    x = np.clip(depth / vmax, 0, 1)
    rgba = (matplotlib.colormaps["viridis"](x) * 255).astype(np.uint8)  # or "Blues"
    rgba[..., 3] = np.where(depth > alpha_below, 230, 0)                # transparent when dry
    return png_bytes(rgba)   # Pillow, RGBA
```

- Per-run default `vmax` = 99th percentile of `max_depth` over street cells
  (min 0.3 m) — compute once, store in the run's cached metrics; the client
  can override with `?vmax=`.
- Hazard PNG: classify `max_hazard.npy` with the bounds and colors from
  `scripts/render2d.py` (`HAZ_BOUNDS = [0.75, 1.25, 2.0]`,
  `HAZ_COLORS = ["#ffe93d","#ff9a2e","#e8442e","#7d1a9e"]`, labels
  "caution / danger: children / danger: most adults / danger: all").
- Frames are fp16 `depth_{t:06d}.npy`; `GET .../frame/{t}.png` loads one,
  colormaps, returns; add an in-memory LRU (last 64 PNGs). Frame list
  endpoint globs the dir (`sorted`, exclude `final_depth`/`max_*` — note the
  existing naming already avoids collisions).

### 10.2 Per-run metrics (`sandbox/metrics.py`)

Generalize `scripts/analyze_corridor.py`:

```python
street = valid & ~building & ~courtyard & ~water          # from base masks
ribbon = design_material > 0                              # THIS design's footprint
bands  = distance_transform_edt(~ribbon) * res            # scipy.ndimage
```

Report (all at the 0.10 m threshold `analyze_corridor.flooded_area` uses):
flooded m² on streets total / on ribbon / 0–25 m / 25–50 m / 50–100 m;
p99 + mean max-depth on the ribbon; and from `run_meta.json`: rain,
infiltrated (+% of rain), outflow, stored, closure. Cache to
`<run>/metrics.json` on first request.

### 10.3 Results UI

- **Run library**: cards (label, design, storm, date, headline "flooded
  street area", notes; rename/notes/delete inline).
- **Run viewer**: map (reuse MapView) with an overlay selector —
  *Animation* (frame scrubber + play at 10 fps, sim-clock readout),
  *Max depth* (+ colorbar with vmax), *Hazard* (+ class legend) — plus a
  side panel with the §10.2 metrics and the gauge depth chart
  (parse `gauges.csv` client-side, draw with a tiny canvas line chart — no
  chart library needed).
- **Compare view**: pick run A ("before") and run B ("after") from
  dropdowns; shows `GET /api/compare/diff.png` (diverging colormap:
  blue = B shallower, red = B deeper, symmetric vmax = p99 |diff|) and a
  delta table (each §10.2 metric: A, B, Δ, Δ%). Add one plain-language
  headline the target user cares about, computed from the deltas:
  *"Your design reduces flooded street area on the corridor by 41 %."*

**Checkpoint 5:** run "official template, no edits" vs "official + your
pit + extra bioswales" on `t2` and open Compare — the diff map and table are
consistent with the maps, and the numbers are in the same family as the
published pipeline results in `output/corridor_runs/metrics.json`.

## 11. Phase 6 — polish for non-technical users

- **Guided flow**: a persistent 3-step header — ① Design ② Rain ③ Run &
  results — with the current step highlighted; empty states that say what to
  do ("Pick a material on the right and paint on the map").
- **Tooltips** on every material with one plain sentence (reuse the report's
  rationale, e.g. bioswale: "a planted channel that soaks up water and slows
  it down; it sits 15 cm below the pavement").
- **Guardrails**: confirm dialogs on delete/overwrite; warn when a single
  stroke moves more than 2 m of elevation; a visible "outside the editable
  zone" cursor state (red ring) instead of silently doing nothing; Run
  disabled with the reason shown; unsaved-changes indicator + save on
  Ctrl+S.
- **Defaults**: opening the app with no designs auto-creates
  "Official corridor" from the template; storm picker defaults to the
  frequent (t2) storm; everything runnable with three clicks.
- **Docs**: add a short "Sandbox" section to `README.md` — the two commands
  from Phase 0 step 5 and one screenshot.

**Checkpoint 6:** hand the app to someone who has never seen the project
with the single sentence "make the corridor flood less, then prove it" —
they get to a comparison screen without help.

## 12. Performance notes

- fp16 depth frames are ~5.4 MB each — never ship raw frames to the browser;
  server-side colormapped PNGs are 100–300 KB.
- DEM/material/delta transfers: quantized uint16 / raw typed arrays, fetched
  once per design open; strokes go up as small dirty-rect patches only.
- The sim itself: ~90 s of storm sims per run on the GPU for this domain
  (t2 ≈ 5400 s sim). Show ETA; never block an endpoint on it.
- `rasterize` of the zone polygon is slow (full-grid `contains_points`) —
  cache `output/sandbox/zone.npy`.
- Three.js: mutate the existing position attribute; never rebuild the
  geometry per stroke. Throttle `computeVertexNormals` to ~4 Hz.
- Keep every grid transfer row-major row-0-north and document it — mixed
  orientations are the most likely whole-class of bugs here.

## 13. Final end-to-end verification

1. `scripts/validate.py` passes (after the `progress_cb` edit).
2. Zero-edit equivalence (Checkpoint 4) passes — sandbox bake ≡
   `bake_corridor.py`.
3. Full user journey (Checkpoint 6) on a clean browser profile.
4. Kill the server mid-edit; restart; the autosaved design is intact.
5. Queue two runs back-to-back; both complete, in order, with closure
   ≤ 0.1 %.
6. `git status` shows no changes inside `output/terrain_cut_0.5/` (base
   terrain untouched — rule 1).

## 14. Explicitly out of v1 (later phases)

- **Drain optimizer button** — wrap `scripts/optimize_drains.py` as a job
  ("suggest inlets for this design"), show the proposed inlets as an
  editable point layer, pass them as `drains=` to `simulate`.
- **Cross-check button** — run SynxFlow (`scripts/synxflow_run.py`) on a
  design for independent confirmation of a headline result.
- **Report export** — one-click PDF/PNG pack of the compare view in the
  style of the report figures.
- **Multi-user / auth / concurrent designs**, larger domains via tiling,
  mobile/tablet support, live rain "paintbrush" (spatially varying storms).
