# Research: point-cloud → flood-raster tooling & parameters (2026-07-12)

Feeds PLAN.md Phase A. Hardware context: 12 cores / 14 GB RAM / 340 GB
disk; input 37 GiB LAS 1.2 PF2 (XYZ+RGB only, no classes/returns/
intensity), 1.5 B pts over ~6 km², ~250 pts/m².

## PDAL (2.10.2, Jun 2026)

- In-memory cost ~45–60 B/pt ⇒ whole cloud ≈ 70–90 GB: **never load whole**.
- Streamable: `readers.las`, `readers.copc`, `writers.las`, `writers.gdal`.
  NOT streamable: `filters.smrf`, `filters.csf`, `filters.chipper`,
  `writers.copc`.
- **`pdal tile`** is the workhorse: streaming, `--length 250 --buffer 25`
  ⇒ ~100 buffered LAZ tiles of ~15–19 M pts (~1–1.2 GB each in PDAL) ⇒
  run 6–8 SMRF workers in parallel within 14 GB.
- COPC: use **Untwine 1.5.1** (GPLv3, conda-forge) — out-of-core octree,
  built for billions of points; needs ~40–80 GB temp disk; output ~8–12 GB
  COPC-LAZ. Payoff: spatial `bounds` queries + QGIS 3.26+ drag-and-drop.

## Ground classification on photogrammetric urban clouds

- **SMRF beats CSF/PMF on photogrammetric clouds** (Int. J. Digital Earth
  2020 comparison; PDAL tutorial also prefers SMRF). SMRF handles all-zero
  return numbers with just a warning (verified in SMRFilter.cpp) — our PF2
  case is fine.
- SMRF starting parameters for dense, steep, stepped Beirut fabric:
  `cell 0.5–1.0, slope 0.25–0.35` (default 0.15 would carve stairs/
  retaining walls out of ground), `threshold 0.45, scalar 1.2,
  window 30–50 m` (must exceed the largest contiguous roof block; default
  18 leaves big roofs as "ground"). Pingel's urban set (scalar 1.2 /
  slope 0.2 / threshold 0.45 / window 16) is the baseline to tune from.
- Pre-clean per tile: `filters.outlier` (mean_k 8–12, multiplier 2–2.5),
  thin to ~25 pts/m² (`filters.sample` r=0.2) — SMRF grids minima at
  `cell` anyway; 250 pts/m² only burns RAM/time.
- CSF as cross-check on misbehaving tiles: rigidness 1–2 (steep terrain),
  cloth res 0.5–1.0, smooth=true (Zhang et al. 2016). pyCSF
  (`cloth-simulation-filter` 1.1.7, Apache-2.0) if a Python API is wanted.
- Expected weaknesses regardless: no under-canopy ground (occlusion),
  melted courtyards ⇒ interpolation + report fraction (PLAN A3/A4).

## WhiteboxTools v2.4.0 (open core — all needed tools are free)

Rasters live in RAM as f64 (≥4× file size rule): 24–96 M cells ⇒ 0.2–0.8 GB
per grid — fine on 14 GB.
- `BreachDepressionsLeastCost --dist=100..200 --fill` — preferred urban
  hydro-conditioning (impact-minimizing breaching) for the *static* D8/QA
  products (dynamic solver still gets the raw surface).
- `RemoveOffTerrainObjects --filter=<px> --slope=15..25` — raster-side
  backstop for residual buildings.
- `FillMissingData --filter=11..25`, `FeaturePreservingSmoothing
  --filter=11 --norm_diff=8..15` (denoise without rounding curbs/steps).
- (`LidarGroundPointFilter`, `LidarTinGridding` exist too; paid extension
  NOT needed.) WBT batch-processes a LAS directory in parallel; keep tiles
  ≤ ~20 M pts.

## Vegetation from RGB (no NIR)

ExG = 2G − R − B on **chromaticity** coordinates (r=R/(R+G+B)…):
threshold ~0.05–0.10 (0.08 typical) or **Otsu per dataset**. Direct
point-cloud study: F ≈ 97.7 % (Remote Sens. 2023, 15:3254). Caveats:
misses dry/brown Mediterranean vegetation + shaded canopy; check whether
LAS RGB is 8-bit stretched to 16-bit before scaling. Trivial with
laspy/numpy per tile or PDAL `filters.assign` expressions.

## Buildings — OSM

Post-2020-blast HOT activation mapped everything within ~8 km of the port
(54k buildings, 516k edits in 4 weeks) — **Mar Mikhael/Achrafieh footprints
are effectively complete**; heights sparse (not needed for masks).
Fetch: Geofabrik `lebanon-latest.osm.pbf` (~50 MB, daily) → `osmium
tags-filter … a/building` → `ogr2ogr` GPKG; or osmnx/Overpass for the
small AOI. Burn: `gdal_rasterize -burn 1 -tr 0.5 0.5 -at` (all-touched, so
thin walls don't leak) or `rasterio.features.rasterize(all_touched=True)`.

## Manning's n (citable)

Chow (1959) via UN-SPIDER mirror; Schubert & Sanders (2012) AWR for
building treatments; Syme 2008 (TUFLOW) for practice values.

| Class | n | Notes |
|---|---|---|
| Asphalt | 0.013–0.016 | streets |
| Concrete (sidewalks) | 0.013–0.017 | |
| Stone paving / stairs | 0.015 (0.013–0.017) | keep low: 0.5 m grid resolves risers; only inflate n if steps get smoothed |
| Bare soil / rubble | 0.022–0.03 | |
| Grass / gardens | 0.030–0.050 | |
| Brush / urban trees | 0.05–0.10 (–0.16) | ground under canopy |
| **Buildings (if not blocked out)** | **0.3–0.5** | Syme 2008: n≈0.3–0.4 reproduces blocked-out afflux while keeping storage; alternative = block out (+5 m) as now |

## Infiltration (citable)

- Rawls, Brakensiek & Miller 1983 Green-Ampt: Ksat clay 0.3 / clay loam
  1.0 / silt loam 3.4 / loam 7.6 / sandy loam 10.9 / sand 117.8 mm/h;
  suction 316→50 mm; porosity 0.31–0.49.
- **Pitt et al. 2001 (compacted urban soils, 153 double-ring tests):**
  sandy non-compacted 330 → compacted 36 mm/h; clayey dry 249 →
  compacted/wet **5 mm/h**. Urban event infiltration ≈ constant-rate.
- Practical classes: impervious 0 mm/h + 1–2 mm initial loss; compacted
  urban soil 5–10 (clayey) to ~36 (sandy) mm/h; lawns/gardens GA Ksat
  7–11 mm/h, suction 89–110 mm; **bioretention/rain gardens 100–300 mm/h**
  (FAWB 2009 media guideline) — this parameterizes the Al-Masar strips.

## Recommended pipeline (fits the laptop)

1. Untwine → COPC (overnight, disk-bound; keep ≥80 GB temp free) — or skip
   straight to (2).
2. `pdal tile --length 250 --buffer 25` → ~100 buffered LAZ tiles.
3. ×6–8 parallel per tile: outlier → thin ~25 pts/m² → SMRF (params above)
   → drop buffer → ground + all-points LAZ.
4. ExG (Otsu, fallback 0.08) strips vegetation from the non-ground set.
5. Streaming `writers.gdal` over ground tiles → 0.5 m DTM (idw + count;
   set `bounds` explicitly in stream mode); WBT FillMissingData →
   FeaturePreservingSmoothing → RemoveOffTerrainObjects →
   BreachDepressionsLeastCost (QA/D8 products only).
6. OSM footprints → 0.5 m masks → building treatment + Manning +
   infiltration grids.

Existing custom streaming gridder (las_common/city_flow) stays as the
fast cross-check for the DTM and as the ortho/RGB source.
