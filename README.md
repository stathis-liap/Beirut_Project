# Beirut / Al-Masar Al-Akhdar — Pluvial Flood Simulation

GPU rain-on-grid flood simulation of the Fouad Boutros corridor (Mar
Mikhael ↔ Achrafieh) and its catchment, built from a 37 GB drone point
cloud. Answers: **where does rain water go today, and what would the
Al-Masar Al-Akhdar green corridor change?**

- Project plan & phase gates: [PLAN.md](PLAN.md)
- Research dossiers (IDF rainfall, engines, tooling): [docs/](docs/)
- Solver: Bates et al. 2010 inertial scheme, torch/CUDA, validated in
  `scripts/validate.py` (analytic benchmarks + reference-port equivalence)

## Environment

Use **bare `python`** everywhere — it resolves to `~/Work/.venv`
(Python 3.14: torch+CUDA, rasterio, shapely, geopandas, imageio). No
activation needed. The RTX 3050 is used automatically; add `--device cpu`
to any simulation command to force CPU. Every step is idempotent
(re-running overwrites its own outputs).

```bash
cd ~/Work/Beirut_Project
```

---

## The pipeline, step by step

### 0. Solver validation + storm library (run once, after any solver change)

```bash
python scripts/validate.py
```
Runs the physics gates: closed-box volume conservation, steady runoff on a
plane vs the kinematic/Manning analytic, lake-at-rest stillness, and
torch-port equivalence vs the legacy numpy solver (fp64, sub-mm) plus an
fp32-drift statistical check. **Everything must PASS** — output is
terminal text only.

```bash
python scripts/storms.py
```
Writes the design-storm library to `storms/*.json`: `t2/t10/t50` (2/10/50-yr
1-h alternating-block hyetographs from the Lebanese stormwater-code IDF
table), `t10cc` (10-yr +15 % climate uplift), `v1_nov2025` (replay of the
observed 25 Nov 2025 burst that flooded Sassine Sq / the Ring), `flat30`
(legacy demo storm).
**View:** `xdg-open storms/storms_preview.png` (all hyetographs side by side).

### 1. Point cloud → raster stack

```bash
python scripts/build_stack.py ~/Work/Beirut_drone.las --res 1.0 --out output/stack_1.0
```
Streams the whole 37 GB LAS once (multiprocess, ~2 min at 1 m) and grids
per cell: lowest Z (flow surface), highest Z (for tree/building height),
point count, mean RGB. Nothing is ever loaded whole.
Outputs: `output/stack_1.0/{zlow,zhigh,count,rgb}.npy + transform.json`.

### 2. OpenStreetMap context

```bash
python scripts/fetch_osm.py --transform output/stack_1.0/transform.json --out output/osm/osm.json
```
Downloads building footprints (post-2020 HOT mapping — near complete
here), streets, and river/canal water for the survey bbox via Overpass.

### 3. Terrain assembly (the heart of Phase A)

```bash
python scripts/build_terrain.py --stack output/stack_1.0 \
    --osm output/osm/osm.json --out output/terrain_1.0 --geotiff
```
Builds everything the solver needs:
- cleans the surface (hole fill, despeckle), removes tree canopy and
  re-interpolates the ground beneath (buildings stay as obstacles);
- classifies land cover: paved / building / canopy / grass / soil / water;
- **sea mask**: cells below +27.5 m ellipsoidal (sea ≈ +26 m) become
  outflow water — harbor basins and port ponds must never store runoff;
- finds enclosed courtyards; reroutes roof + courtyard rain to the nearest
  street cell (downspout model);
- Manning-n and infiltration rasters from land cover (cited values);
- static fill bound (max physically possible ponding per cell, on GPU) —
  the backbone of the sanity screens;
- auto-places 5 gauges on the main drainage lines (`--gauges-only`
  recomputes just these; you can also hand-edit `gauges.json`).

**View the QA — this is a required gate, not a formality:**
```bash
xdg-open output/terrain_1.0/landcover_qa.png    # classes + gauges over the ortho
xdg-open output/terrain_1.0/courtyards_qa.png   # detected enclosed courtyards
xdg-open output/terrain_1.0/hillshade.png       # terrain relief check
```
If a bridge/overpass dams a street: draw cut polygons with
`python scripts/scenario.py draw --data-dir output/terrain_1.0 --out output/cut1.json`,
collect them into a file `{"polygons": [[...], ...]}` and re-run
build_terrain with `--cuts cuts.json`.
`--geotiff` also writes `dem/landcover/manning/infil_mmh.tif`
(EPSG:32636) — drag into **QGIS**, or hand to the BUL ArcGIS hub.

### 4. Storm drains (scenario S1/S3 input)

```bash
python scripts/drains.py --terrain output/terrain_1.0 --osm output/osm/osm.json
```
Places an inlet every 27 m along OSM streets, 30 L/s each (an assumption —
no Beirut drainage data exists; that's why "drains working" is a scenario,
not the baseline). Options: `--spacing`, `--capacity`.
**View:** `xdg-open output/terrain_1.0/drains_qa.png` (cyan dots = inlets).

### 5. The Al-Masar corridor (scenario S2/S3 input)

```bash
python scripts/masar_corridor.py --terrain output/terrain_1.0
```
Builds the green-corridor edit file `scenarios/masar_corridor.json`:
a 16 m strip with rain-garden infiltration (150 mm/h, FAWB range) and
vegetated roughness (n = 0.05). By default it auto-traces the main
drainage channel as a proxy. **For the real report, draw the actual Fouad
Boutros right-of-way** (from the AUB maps) and use it instead:
```bash
python scripts/scenario.py draw --data-dir output/terrain_1.0 --out output/masar1.json
python scripts/masar_corridor.py --terrain output/terrain_1.0 --polygon output/masar1.json
```
**View:** `xdg-open output/terrain_1.0/masar_qa.png` (green outline = strip).

### 6. Run the scenario × storm matrix

```bash
# validation first: the Nov-2025 replay must flood the documented spots
python scripts/run_matrix.py --terrain output/terrain_1.0 --storms v1_nov2025 --animate

# then the design storms (4 scenarios each: S0 baseline, S1 drains,
# S2 corridor, S3 corridor+drains). ~20-30 min per run at 1 m on the GPU.
python scripts/run_matrix.py --terrain output/terrain_1.0 --storms t2 t10 t50 t10cc --animate
```
Each run simulates, then automatically produces its sanity report and
maps. To run long jobs detached and watch progress:
```bash
nohup python scripts/run_matrix.py ... > output/runs_matrix.log 2>&1 &
tail -f output/runs_matrix.log
```

---

## Where the results are & how to view them

### The matrix summary
```bash
cat output/runs/summary.md                       # scenario x storm metric table
libreoffice --calc output/runs/summary.csv       # same, spreadsheet
```
Columns: sanity verdict, p99 street depth, flooded area >10 cm / >30 cm,
volumes infiltrated / drained / discharged / stored, wall time.

### Per run — `output/runs/<scenario>__<storm>/`

| File | What it shows | View with |
|---|---|---|
| `flood.mp4` | animated flood over the ortho, clock + live hyetograph inset | `xdg-open .../flood.mp4` |
| `maxdepth.png` | max street depth map (color capped at p99 so artifacts can't hijack the scale; absolute max printed in the title) | `xdg-open` |
| `hazard.png` | DEFRA-style hazard classes h·(v+0.5): caution → danger-for-all | `xdg-open` |
| `diff_vs_baseline.png` | scenario minus baseline depth (blue = scenario is drier) — **the money shot for S2** | `xdg-open` |
| `sanity.png` + `sanity_report.json` | automated screens: mass-balance closure, static fill-bound check, every >1 m cell classified courtyard/numerical/genuine, velocity screen, rational-method outflow envelope | `xdg-open` / `python -m json.tool` |
| `gauges.csv` | depth + velocity time series at the 5 gauges | `column -s, -t .../gauges.csv \| less` |
| `run_meta.json` | full mass balance, outflow & storage series, solver settings | `python -m json.tool` |
| `depth_*.npy`, `max_*.npy`, `final_depth.npy` | raw fields (fp16 frames / fp32 maxima) for custom analysis | numpy |

**Trust rule: only publish runs whose `sanity_report.json` says `PASS`.**
The report explains every flag it raises.

### Extra visuals on demand

```bash
# 3D PyVista flyover video (camera settings pre-tuned)
python scripts/render_3d.py video --run output/runs/S0_baseline__t10 \
    --data-dir output/terrain_1.0 --out output/flyover.mp4

# any 2D product for any run pair
python scripts/render2d.py animate  --run <run> --terrain output/terrain_1.0 --out out.mp4
python scripts/render2d.py maxdepth --run <run> --terrain output/terrain_1.0 --out out.png
python scripts/render2d.py hazard   --run <run> --terrain output/terrain_1.0 --out out.png
python scripts/render2d.py diff --run-a <baseline> --run-b <scenario> \
    --terrain output/terrain_1.0 --out diff.png
```

### One-off simulation (outside the matrix)

```bash
python scripts/flood_gpu.py --terrain output/terrain_1.0 --storm storms/t10.json \
    --out output/runs/oneoff [--drains output/terrain_1.0/drains.npz] [--no-frames]
python scripts/sanity.py --run output/runs/oneoff --terrain output/terrain_1.0
```

---

## Finals at 0.5 m (overnight, for the report figures)

```bash
python scripts/build_stack.py ~/Work/Beirut_drone.las --res 0.5 --out output/stack_0.5
python scripts/build_terrain.py --stack output/stack_0.5 --osm output/osm/osm.json \
    --out output/terrain_0.5 --geotiff
python scripts/drains.py --terrain output/terrain_0.5 --osm output/osm/osm.json
python scripts/masar_corridor.py --terrain output/terrain_0.5
python scripts/run_matrix.py --terrain output/terrain_0.5 \
    --storms v1_nov2025 t10 --save-every 120        # frames are 24 MB each at 0.5 m
```

## Cross-engine verification (independent solvers, PLAN Phase D3)

```bash
python scripts/export_ascii.py --terrain output/terrain_1.0 \
    --storm storms/v1_nov2025.json --out output/export_lfp
```
Writes ESRI-ASCII `dem.asc`/`n.asc`, a rain file, and a ready LISFLOOD-FP
8.x `.par`. Engine choice, build notes, and EA Test-8A benchmark data
pointers: [docs/research_flood_engines.md](docs/research_flood_engines.md).

Three engines and the external benchmark run in one shot:

```bash
bash scripts/run_validation.sh 2>&1 | tee output/validation.log
```
It is idempotent (finished stages are skipped) and covers: our torch
solver, LISFLOOD-FP 8.2 (CPU, ~1-2 h), SynxFlow (GPU full-SWE, its own
`~/Work/.venv_synxflow`), the two pairwise comparisons, and the EA
Neelz-Pender **Test 8A** benchmark. Results and the environment pins:
[docs/crosscheck_lisflood.md](docs/crosscheck_lisflood.md).

## The cut domain (corridor at 0.5 m)

Smaller, higher-resolution domain around the corridor for validation and
final figures — the box is the corridor plus its D8 upslope watershed, so
cutting it out does not starve the corridor of runoff.

```bash
python scripts/cut_domain.py --terrain output/terrain_1.0 \
    --polygon output/masar_row.json --buffer 150      # -> output/cut_polygon.json
python scripts/crop_cloud.py ~/Work/Beirut_drone.las --out data/cut_masar.las \
    --polygon output/cut_polygon.json                 # 8.4 GB, ~1 min
python scripts/build_stack.py data/cut_masar.las --res 0.5 \
    --out output/stack_cut_0.5 --workers 4
python scripts/build_terrain.py --stack output/stack_cut_0.5 \
    --osm output/osm/osm.json --out output/terrain_cut_0.5 --geotiff
python scripts/drains.py --terrain output/terrain_cut_0.5 --osm output/osm/osm.json
python scripts/masar_corridor.py --terrain output/terrain_cut_0.5 \
    --polygon output/masar_row.json --width 16 --out scenarios/masar_corridor_cut.json
```
`--workers 4` is not optional on a 14 GB box: the LAS scripts hold a whole
chunk per worker.

---

## Current status (2026-07-13)

Done: validation suite PASS · storms built · 1 m stack + terrain + drains
+ corridor built and QA'd · test-block matrix (3 scenarios) PASS end-to-end.
The v1_nov2025 × 4-scenario matrix on the full domain runs detached —
`tail -f output/runs_matrix.log`; results land in `output/runs/`.

## Known facts & gotchas

- **Z is ellipsoidal — sea surface ≈ +26 m.** Never threshold against 0.
  The terrain builder masks everything below +27.5 m as outflow water
  (`--sea-level/--sea-margin`); without this, harbor basins swallow runoff
  39 m deep and collapse the timestep.
- **Friction bug fixed 2026-07-13**: the demo solver used `h^(10/3)` in
  the friction denominator instead of Bates-2010's `h^(7/3)` — friction
  overestimated ×1/h in shallow flow. Both solvers fixed;
  `validate.py` pins the physics (old demo outputs pre-date the fix).
- The legacy `/4` flux clip capped velocities unphysically; the production
  solver uses mass-conserving outflux scaling (`limiter="scale"`).
- fp32 runs are statistically equivalent to fp64 (<1 % volume, <5 %
  percentiles) but not cellwise identical — report percentiles/areas,
  never single cells.
- Roof/courtyard rain is rerouted to the nearest street cell (downspout
  model); courtyards are excluded from headline metrics.
- Drain capacity (30 L/s/inlet) is an assumption — present S0 vs S1 as a
  range (see docs/research_rainfall_beirut.md §4).
- Supercritical street flow of 6–8 m/s is *physical* on steep smooth
  streets here; the sanity screen only flags >10 m/s.

## Legacy demo (2-day sprint, Vendôme crop)

The original pipeline still works and shares this repo:
`make_preview.py`, `crop_cloud.py`, `build_dem.py`, `flood_sim.py`
(now the CPU reference for validate.py), `scenario.py`, `make_scenarios.py`,
`render_3d.py`, `city_flow.py`, `map_overlay.py`, `test_synthetic.py`.
Its outputs live in `output/run_*`, `output/flood_*.mp4`, etc.
