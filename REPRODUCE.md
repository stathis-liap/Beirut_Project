# Reproducing the results

This repository contains the code, configuration, and the derived terrain
grids needed to reproduce the Al-Masar Al-Akhdar green-corridor flood study.

## What is and isn't in the repo

- **Included:** all pipeline code (`scripts/`), scenario and storm definitions
  (`scenarios/`, `storms/`), research/validation notes (`docs/`), the official
  corridor geometry and result JSONs (`output/*.json`, `output/crosscheck/`,
  `output/ea8/`), and the **terrain grids the solver runs on**
  (`output/terrain_cut_0.5/`, `output/corridor_gi_cut/`).
- **Not included (too large for GitHub):** the raw LiDAR point clouds
  (the source survey and the ~8.4 GB corridor cut `data/cut_masar.las`),
  the raster stacks, per-timestep run frames, and videos. These are all
  reproducible from the code plus the source survey, or from the terrain
  grids that *are* included.

You do **not** need the raw point cloud to reproduce the flood-simulation
results — the terrain grids in `output/terrain_cut_0.5/` are the direct
inputs to the solver.

## Environment

```bash
# Python 3.12 with the geospatial + torch stack (GPU optional but recommended)
# key deps: numpy, scipy, torch, rasterio, shapely, matplotlib, pillow
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy torch rasterio shapely matplotlib pillow
```

## Reproduce the corridor before/after results

```bash
# 1. bake the green-corridor materials into a copy of the terrain
python scripts/bake_corridor.py \
    --terrain output/terrain_cut_0.5 \
    --material output/corridor_gi_cut/material.npy \
    --out output/terrain_cut_corridor

# 2. run the before/after + drain scenarios (GPU; writes output/corridor_runs/)
bash scripts/run_corridor_study.sh

# 3. compute the corridor-focused metrics and figures
python scripts/analyze_corridor.py
```

`scripts/optimize_drains.py` produces the targeted inlet set; the exact set
used is provided at `output/terrain_cut_corridor/drains_opt.npz`.

## Regenerate the corridor geometry / materials (optional)

```bash
# corridor material raster from the official zone polygon + terrain
python scripts/build_corridor_gi.py \
    --terrain output/terrain_cut_0.5 \
    --zone output/masar_zone_official.json \
    --out output/corridor_gi_cut
```

## The corridor point cloud (optional download)

The 8.4 GB cropped corridor point cloud is too large for the git tree, so it
is published as a **GitHub Release asset** in compressed LAZ form
(~2.9 GB), split into two parts to fit the per-asset size limit. To obtain
it, download both parts from the repository's Releases page and reassemble:

```bash
cat cut_masar.laz.part00 cut_masar.laz.part01 > cut_masar.laz
# integrity check (expected sha256 of the reassembled .laz):
#   3cfbd75b71f08a6f03408145cc81fcbfd156c8bebeaa7a49e650e7cf3cd2d505
sha256sum cut_masar.laz
```

Reading the LAZ requires `laspy` with the `lazrs` backend
(`pip install laspy lazrs`). This raw cloud is only needed to rebuild the
terrain grids from scratch; the grids in `output/terrain_cut_0.5/` already
provide everything the simulations consume.

## Rebuild the terrain from the point cloud (needs the source survey or the LAZ above)

The terrain grids are rebuilt with:

```bash
python scripts/crop_cloud.py <survey.las> --out data/cut_masar.las \
    --polygon output/cut_polygon.json --workers 4
python scripts/build_stack.py data/cut_masar.las --res 0.5 \
    --out output/stack_cut_0.5 --workers 4
python scripts/build_terrain.py --stack output/stack_cut_0.5 \
    --out output/terrain_cut_0.5 --geotiff
```

## Validation

```bash
python scripts/validate.py                  # analytic benchmarks + mass balance
python scripts/benchmark_ea8.py run  --res 2m --out output/ea8   # EA Test 8A
python scripts/benchmark_ea8.py plot --out output/ea8
# cross-engine comparison (LISFLOOD-FP / SynxFlow) is driven by
# scripts/run_validation.sh; see docs/crosscheck_lisflood.md.
```
