# Beirut Corridor Flood Simulation

Pipeline: huge drone point cloud → crop to corridor → DEM → 2D flood
simulation → what-if scenarios → 3D videos.

Data: `~/Work/Beirut_drone.las` — LAS 1.2 fmt 2, 1.5 B points, ~39 GB,
EPSG:32636 (UTM 36N). All scripts stream it; nothing loads it whole.

```bash
source .venv/bin/activate   # Python 3.12 venv
pip install -r requirements.txt
```

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

## Notes / assumptions

- DEM = lowest-percentile Z per cell: streets at street level, buildings at
  roof height (act as flow obstacles). No storm-drain network data — the
  baseline assumes drains are absent/clogged, which matches the observed flooding.
- Rain scenarios are design storms (10/30/60 mm/h); no measured rainfall data.
- The source LAS download must be COMPLETE before final crops — the file is
  flight-line ordered, so a partial file has patchy spatial coverage.
  `make_preview.py` writes a coverage mask and `crop_cloud.py` warns about gaps.
