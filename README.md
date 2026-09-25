# Beirut Corridor Flood Simulation

Pipeline: huge drone point cloud → crop to corridor → DEM → 2D flood
simulation → what-if scenarios → 3D videos.

Data: `~/Work/Beirut_drone.las` — LAS 1.2 fmt 2, 1.5 B points, ~39 GB,
EPSG:32636 (UTM 36N). All scripts stream it; nothing loads it whole.

Plug and play: drop any cropped `.las` (LAS 1.2, point format 2) into
`data/` and the GUI/scripts pick it up automatically — no hardcoded
filename. With more than one file there, the most recently modified wins
(and every candidate found gets printed, so the choice is never silent).

```bash
source .venv/bin/activate   # Python 3.11 - Taichi 1.7 has no newer wheels
pip install -r requirements.txt
```

`.dwg` models additionally need the ODA File Converter, which is not
pip-installable: `winget install ODA.ODAFileConverter`.

The three full-map 5cm rasters (`DSM_5cm/`, `DTM_5cm/`, `RGB_5cm/`) are
found automatically under `3D/` or the project root; set `BEIRUT_MAP_DATA`
if they live anywhere else. They're only needed for "pick a point on the
map" — the 2D flood and CAD modes don't touch them.

## Quick start

Once the corridor has been cropped once (step 1 below), skip the manual
pipeline and just run:

```bash
python main.py
```

This opens the plug-and-play GUI: pick a rain intensity, storm duration,
and terrain quality, hit **Run simulation**, and it drives `build_dem.py`
-> `flood_sim.py` -> `render_3d.py` for you with a live progress bar. When
it's done you can open the results folder, the heatmap, the flyover video,
or an interactive 3D view (terrain only, or terrain + water with a
time/speed slider and play/pause). No command line needed after the initial
crop.

The manual steps below cover that one-time crop, plus scripting/batch use
(what-if scenarios, custom renders) that the GUI doesn't expose.

## 1. Preview + crop

```bash
# one pass over the LAS -> top-down RGB map + coverage (few minutes)
python scripts/make_preview.py ~/Work/Beirut_drone.las --res 0.5

# click the corridor polygon on the preview, streams + writes cropped LAS
python scripts/crop_cloud.py ~/Work/Beirut_drone.las --out data/corridor.las
# re-run later (e.g. after full download) without clicking:
python scripts/crop_cloud.py ~/Work/Beirut_drone.las --out data/corridor.las \
    --polygon output/crop_polygon.json
```

## 2. Terrain

```bash
python scripts/build_dem.py data/corridor.las --res 1.0
# outputs: output/dem.npy, ortho.png, dem_hillshade.png, flow_accum.png
# flow_accum.png already shows where the "river" forms - first demo image.
```

## 3. Flood simulation (baseline)

```bash
python scripts/flood_sim.py --rain 30 --duration 3600 --save-every 30 \
    --out output/run_baseline
```

Solver: Bates et al. 2010 inertial shallow-water scheme (LISFLOOD-FP),
rain-on-grid, Manning friction, optional infiltration + storm-drain sinks,
CFL-adaptive timestep. The per-step stencil runs as a single fused,
multi-threaded Numba kernel (~10x faster than a plain NumPy port at these
grid sizes, where per-op overhead dominates over raw FLOPs). Prints a mass
balance at the end as a sanity check.

## 4. What-if scenarios

```bash
# draw an edit polygon on the ortho (prints UTM coords, saves JSON)
python scripts/scenario.py draw --out output/edit1.json

# write a scenario file (see scenarios/ for examples), then:
python scripts/scenario.py run scenarios/escape_channel.json --rain 30
```

Edit ops: `raise`/`lower` (fill dirt / carve channel), `wall` (barrier),
`infiltrate` (permeable soil, mm/h), `sink` (storm drain).

## 5. Visualize

```bash
# 3D animated flyover video of one run
python scripts/render_3d.py video --run output/run_baseline --out output/baseline.mp4

# baseline vs scenario max-depth comparison
python scripts/render_3d.py compare \
    --runs output/run_baseline output/run_escape_channel --out output/compare.png
```

In the interactive viewer (`render_3d.py view`, or the GUI's "Open 3D View"
buttons), **Ctrl+Left-click** any point on the terrain to pop a focused,
lightweight 3D reconstruction of just that area (default 1.5 km radius) in
a new window — streamed directly and only from that region of the source
`.las`, not the corridor DEM, so it stays fast regardless of how big the
underlying point cloud is. See `scripts/reconstruct_area.py`.

## Running the 3D solver on a small GPU

Taichi sizes its device allocation from total VRAM, which overshoots on a
small laptop card and dies as `CUDA_ERROR_OUT_OF_MEMORY` before the first
step — with the GPU sitting idle in `nvidia-smi`, so the message points at
the wrong thing. `scripts/ti_init.py` caps the allocation and falls back to
CPU if the GPU still won't start. Two knobs if a run won't fit:

```bash
BEIRUT_TAICHI_ARCH=cpu     # force CPU (slower, same result)
BEIRUT_TAICHI_MEM_GB=0.25  # shrink the device-memory cap
BEIRUT_HASH_BUDGET_MB=96   # shrink the SPH spatial hash (coarser cells)
```

Note that under WDDM these are committed against system memory too, so a
machine whose pagefile can't grow (a full system drive) fails here even
with free VRAM. That failure looks identical to a GPU problem and isn't
one — check the commit charge before blaming the card.

## Notes / assumptions

- DEM = lowest-percentile Z per cell: streets at street level, buildings at
  roof height (act as flow obstacles). No storm-drain network data — the
  baseline assumes drains are absent/clogged, which matches the observed flooding.
- Rain scenarios are design storms (10/30/60 mm/h); no measured rainfall data.
- CAD models are not georeferenced — they're local scenes, so there's no
  `--utm-x/--utm-y` for them. `3D/3D Site.3dm` works as-is. `3D/3D Design.3dm`
  does not: it's tagged millimeters and mixes local-origin design geometry
  with GIS layers at real UTM coordinates, so no single `--unit-scale` fits
  both — the layers have to be separated in Rhino first.
- `3D/02. Beirut Project_3D Cad Model.dwg` converts fine via ODA but is
  23,450 ACIS 3DSOLID entities with no proxy graphics, which nothing in
  Python can tessellate. Re-export it as `.obj`/`.3dm` to use it.
- `3D/DSM_5cm` was salvaged from a CRC-corrupt zip and has one damaged LZW
  strip (see its `CORRUPT_REGION.json`). `map_point.py` interpolates across
  it automatically; a clean copy still needs re-requesting from AUB.
- The source LAS download must be COMPLETE before final crops — the file is
  flight-line ordered, so a partial file has patchy spatial coverage.
  `make_preview.py` writes a coverage mask and `crop_cloud.py` warns about gaps.
