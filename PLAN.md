# Al-Masar Al-Akhdar — Full Pluvial Flood Simulation: Project Plan

Goal: a defensible, GPU-accelerated rain-on-grid flood study of the Fouad
Boutros / Al-Masar Al-Akhdar corridor (Mar Mikhael ↔ Achrafieh) and its
contributing catchment, built from the 39 GB drone point cloud, validated at
multiple levels, and communicated so that AUB Beirut Urban Lab stakeholders —
including non-technical ones — can see where water goes today and what the
green corridor would change.

This extends the completed 2-day demo (Vendôme stairs crop, 370×835 m).
Everything below is phased A→F; each phase has explicit acceptance gates.

---

## 0. Where we are (verified 2026-07-12)

**Assets that work and are kept:**
- Streaming LAS toolkit (`las_common.py`) — raw-struct reader, ~10× faster
  than generic libs, partial-file safe. Full source now complete on disk
  (37 GiB, 1.5 B pts, EPSG:32636, extent 2916×2066 m).
- `city_flow.py` — multiprocess city-wide min-Z DEM + D8 at 2 m (done).
- `build_dem.py` — low-percentile flow-surface DEM + ortho + QA products.
- `flood_sim.py` — Bates/de Almeida-style inertial SWE solver (NumPy),
  rain-on-grid, Manning, infiltration, sinks, open boundaries, mass balance.
- Scenario engine (polygon edits), auto scenario generation, PyVista MP4
  renderer, OSM overlays, synthetic end-to-end test.
- Demo results on the Vendôme crop: 4 scenarios × 1 h @ 30 mm/h @ 0.5 m,
  mass balance closes (7 475 m³ rain = 6 417 stored + ~1 058 out, baseline).

**Verified hardware:** RTX 3050 Laptop 6 GB (5.6 GB free), CUDA 13.2 driver,
torch 2.12.1+cu130 in the venv **works on GPU**. Measured stencil probe:
50 diff/max passes over a 24.1 M-cell grid in 0.22 s. 12-core i5-13420H,
14 GB RAM, 340 GB free disk.

**Known defects to fix (measured today, baseline run):**
1. Headline "max depth 2.03 m" is an artifact: it sits in a ~5 m enclosed
   void between buildings (photogrammetric courtyard trap). Depth
   distribution over wet cells is otherwise plausible: p50 0.09 m,
   p90 0.23 m, p99 0.62 m; only 392 cells (0.04 %) exceed 1 m.
   → needs a courtyard/roof rain policy + automated artifact screens.
2. 86 % of rain volume still ponded at t = 1 h — no storm-drain model and
   courtyard traps inflate storage.
3. `escape_channel` scenario *increased* max depth (2.29 m) — scenario
   placement was automatic; needs hydraulic review with flow outputs.
4. The demo crop has open boundaries mid-slope: real uphill inflow from
   Achrafieh was cut off. The full task must simulate the whole catchment.
5. CPU solver: ~5 h per simulated hour on 1.2 M cells — unusable for the
   ~24 M-cell full domain and a multi-scenario matrix. → GPU (Phase B).
6. Point cloud has **no** classification / returns / intensity (all zero) —
   land cover must be derived from RGB + geometry + OSM priors.
7. `fill_sinks` is pure-Python heapq — fine at 2 m city scale, far too slow
   at 0.5 m. Replace with RichDEM / WhiteboxTools.

**Standing data gotchas (do not rediscover):**
- Z is **ellipsoidal**; sea surface sits at ≈ +26 m. Never threshold
  against 0 for "sea level". Mask sea/river by classification + location.
- The LAS is flight-line ordered; spatial crops need the full file (done).
- The survey footprint drains to two real outlets: the sea (port quays,
  north) and the Beirut River channel (east). These become model outflow
  boundaries.

---

## 1. Deliverables

| # | Deliverable | Audience |
|---|-------------|----------|
| D1 | Simulation-ready raster stack (COG GeoTIFFs, EPSG:32636): terrain, land cover, Manning n, infiltration, building/courtyard/water masks, ortho | us + BUL GIS hub |
| D2 | GPU flood engine (torch/CUDA port of the validated solver) + adopted external engine for cross-checks | us |
| D3 | Run matrix results: design storms × scenarios, each with depth/velocity/hazard rasters, hydrographs at gauges, run manifest + automated sanity report | us + report |
| D4 | Validation dossier: benchmarks, cross-engine comparison, convergence, sensitivity, ground-truth spot checks | reviewers |
| D5 | Visual pack: animated 2D maps, 3D flyover MP4s, A/B scenario videos, hazard maps, difference maps, 60–90 s narrated storyboard video, interactive HTML page | BUL + non-technical stakeholders |
| D6 | Final report (methods, assumptions, results, uncertainty, corridor recommendations) | BUL / AUB |

---

## 2. Phase A — Extract everything the cloud knows, then clean it

Target: one reproducible script pass from `Beirut_drone.las` to a tiled,
georeferenced raster stack at 0.5 m (analysis) and 1.0 m (iteration) over
the full 2 916 × 2 066 m survey.

**A0. (Recommended) one-time COPC conversion with Untwine** (not
`writers.copc`, which is non-streamable): out-of-core octree build,
~40–80 GB temp disk, output ~8–12 GB COPC-LAZ. Gives spatial `bounds`
queries for any future crop + drag-and-drop viewing in QGIS 3.26+. Keep
the raw-struct streaming path as fallback; it already works.
Tiling for classification: `pdal tile --length 250 --buffer 25`
(streaming) → ~100 LAZ tiles of ~15–19 M pts; 6–8 parallel workers fit in
14 GB. Details + parameters: `docs/research_pointcloud_tooling.md`.

**A1. Full-extent gridding (streaming, multiprocess — extend `city_flow.py`).**
Per 0.5 m cell, in one pass: low-percentile Z (flow surface candidate),
high-percentile Z (DSM), point count, mean RGB, RGB variance. Write as
tiled COGs via rasterio. Memory bound: ~24 M cells × a few float32 layers
≈ manageable; per-worker sparse accumulation as in `city_flow.py`.

**A2. Per-cell land-cover classification** (no LiDAR classes exist, so):
- ground/non-ground: **PDAL SMRF per tile** — proven best on
  photogrammetric urban clouds; urban-steep parameters: cell 0.5–1.0,
  slope 0.25–0.35 (default 0.15 would carve the stairs out of "ground"),
  threshold 0.45, scalar 1.2, window 30–50 m (> largest roof block);
  pre-clean with `filters.outlier`, thin to ~25 pts/m². CSF (rigidness
  1–2) as cross-check on tiles where SMRF misbehaves. The existing
  low-percentile gridder is the second, independent ground estimate;
- vegetation: excess-green ExG = 2g−r−b on **chromaticity** coords,
  Otsu threshold (fallback 0.08) + height-above-ground > 2 m ⇒ canopy;
  high ExG + low height ⇒ grass/shrub. Caveat: misses dry/brown
  Mediterranean vegetation — audit in QA;
- buildings: OSM footprints — **effectively complete here** (post-2020
  HOT activation mapped the blast radius: 54 k buildings) — fetched via
  Geofabrik Lebanon extract, rasterized `all_touched` so thin walls don't
  leak; intersect with height-above-ground > 3 m; manual patch layer for
  disagreements;
- water: sea + Beirut River by location polygon (NOT by z<0 — ellipsoidal);
- vehicles: small high-object blobs on streets (optional: ultralytics YOLO
  on the ortho — already installed); they are removed from terrain anyway;
- remainder: paved / bare soil split by color.
Output: `landcover.tif` (classes), QA: 200-cell random audit vs ortho.

**A3. Hydraulic surface assembly ("DTM + buildings"):**
- ground under canopy/vehicles: remove veg/vehicle cells, re-interpolate
  from neighboring ground (report interpolated fraction as uncertainty);
- buildings stamped as obstacles at roof height (current approach, kept);
- **bridge/overpass cuts**: the photogrammetric surface dams underpasses
  (Charles Helou overpasses, stair bridges). Detect + cut manually with the
  existing polygon editor; verify with flow arrows;
- stairs must survive: despeckle is edit-preserving (median only on
  outliers) — add cross-section QA plots on the three stair streets
  (Vendôme, St-Nicolas, Massad).

**A4. Courtyard & roof rain policy** (fixes defect 1):
- detect enclosed voids: cells not building, unreachable from the street
  network by flood-fill over non-building cells;
- default: buildings' and enclosed courtyards' rain is **rerouted to the
  nearest street-perimeter cell** (downspout assumption, mass-conserving);
  toggle to "roofs pond" for sensitivity;
- headline metrics exclude building-interior/courtyard cells by mask.

**A5. Parameter rasters** (all values + citations in
`docs/research_pointcloud_tooling.md`):
- `manning_n.tif`: asphalt 0.013–0.016, concrete 0.013–0.017, stone
  paving/stairs 0.015 (grid resolves the risers — don't inflate), soil
  0.022–0.03, grass/gardens 0.03–0.05, under-canopy 0.05–0.10 (Chow 1959);
  buildings either blocked out (+5 m, current approach) **or** n = 0.3–0.5
  (Syme 2008 shows n≈0.3–0.4 matches blocked-out afflux while keeping
  storage) — carried as a sensitivity pair;
- `infiltration.tif`: impervious 0 (+1–2 mm initial loss); compacted urban
  soil 5–10 mm/h clayey to ~36 mm/h sandy (Pitt et al. 2001); lawns/
  gardens Green-Ampt Ksat 7–11 mm/h (Rawls et al. 1983); **Al-Masar rain
  gardens / bioretention strips 100–300 mm/h** (FAWB 2009) — this is the
  scenario-S2 parameter.

**Gates for A:** DEM−demo-crop diff explained; stair cross-sections intact;
no phantom dams across known underpasses; land-cover audit ≥ 90 % on the
random sample; interpolated-ground fraction reported.

---

## 3. Phase B — Simulation engine: adopt what's proven, GPU-accelerate the rest

Strategy (two tracks, cheap insurance):

**B1. Primary: port `flood_sim.py` to torch CUDA.** The solver is already
vectorized NumPy implementing the same de Almeida/Bates inertial scheme
LISFLOOD-FP uses; torch is a near drop-in replacement and is already
installed with CUDA 13 support. fp32 state + fp64 mass accumulators,
`torch.compile` for kernel fusion. Measured probe ⇒ expected ~20–90 ms per
step on the 24 M-cell 0.5 m full domain ⇒ roughly 0.5–1.5 h wall per
simulated hour at 0.5 m, and minutes at 1.0 m (6 M cells, 2× dt). Keep the
NumPy path as reference (same function, backend switch); bit-compare fp64
CPU vs GPU on the test block.

**B2. Cross-check engines** (full comparison: `docs/research_flood_engines.md`),
run on our exported rasters for the finalist storms so every published
number has independent solvers behind it:
- **SERGHEI 2.x** (BSD-3, active, Kokkos/CUDA-13-friendly GPU, spatially
  varying rain/Horton-infiltration/Manning, ESRI-ASCII inputs) — a
  **full-SWE Roe solver, methodologically independent** of our inertial
  scheme: agreement is strong evidence. Build budget: half a day.
- **LISFLOOD-FP 8.2 ACC, CPU build** (trivial) — the *reference
  implementation of our exact scheme*: sharpest check of our torch
  kernels + strongest EA pedigree. License ambiguous for commercial use
  (see risks) — used for verification only, nothing shipped depends on it.
- Optional: **Itzï 26.6** (EA Test-8A-validated, SWMM coupling via pyswmm)
  if a real 1D drainage network is ever added; **Landlab OverlandFlow**
  (pip, MIT) for unit-level test-block checks.
Ruled out after source inspection: pypims/HiPIMS (dead), TorchSWE
(archived), TRITON (no infiltration), HEC-RAS (closed, Windows authoring).

**B3. Features to add (over the adopted base):**
- hyetograph rain input (time series, per §C design storms) instead of
  constant intensity; optional spatially varying rain;
- spatially varying Manning n + Green-Ampt infiltration from Phase A rasters;
- storm-drain inlets: point sinks with capacity caps (m³/s), placed along
  OSM streets at ~25–30 m spacing; capacity is a scenario parameter, not a
  point estimate — no per-inlet data exists for Beirut; the Lebanese code
  designs for a 10-yr storm, the aged combined network realistically
  intercepts ≤ 2–5-yr intensity, and the documented failure mode is
  inlets clogging within minutes (see docs/research_rainfall_beirut.md §4);
- roof/courtyard rain rerouting (Phase A4);
- gauge points: depth/velocity/discharge time series at named hotspots
  (the three stairways, Armenia St, corridor outlet, river/sea outlets);
- hazard accumulator: max h, max |v|, max h·(|v|+0.5) (DEFRA-style);
- checkpoint/resume + run manifest JSON (all inputs hashed) for
  reproducibility;
- review the ad-hoc `/4` flux clip against the published limiter.

**Gates for B:** GPU vs CPU reference agree (max |Δh| < 1 mm on test
block); mass balance closes < 1 % on all runs; 1 m full-domain storm ≤
15 min wall; 0.5 m ≤ 2 h wall.

---

## 4. Phase C — Rainfall forcing & the run matrix

Basis: published Beirut IDF table in the *Stormwater Network Code in
Lebanon* (2022; local copy `docs/stormwater_code_lebanon.pdf`),
cross-confirmed within ~10–25 % by a 2026 peer-reviewed satellite-derived
IDF study. Full table + provenance caveats:
`docs/research_rainfall_beirut.md`. Key values (mm/h):

| Duration | 2 yr | 10 yr | 50 yr |
|---|---|---|---|
| 10 min | 74.1 | 118.4 | 159.3 |
| 30 min | 41.9 | 66.8 | 91.6 |
| 60 min | 25.9 | 41.7 | 57.9 |

Storm set (alternating-block/Chicago hyetographs — peak intensity drives
pluvial response; constant-rate understates it):
- **V1 "25 Nov 2025 replay"** — observed: 25.4 mm/30 min with a
  22.2 mm/15 min (≈89 mm/h) peak, drains clogged. Primary validation
  storm: documented ponding at Sassine Sq + the Ring that day, from only
  a ~2–5-yr burst.
- **T2 / T10 / T50** — 1-h design storms, peak 10-min block from the
  table (74 / 118 / 159 mm/h), totals 25.9 / 41.7 / 57.9 mm.
- **T10+CC** — 10-yr storm ×1.15 (2050 horizon; Mediterranean literature
  supports +10–15 %, +20–30 % end-century — Zittis et al. 2021).
- Sanity long-duration check: 2-h 50-yr (35.3 mm/h avg) once.

The demo's flat 30 mm/h sits between T2 and T10 at 1 h — good continuity.

Run matrix (1 m grid for the full matrix, 0.5 m re-runs for finalists):

| Scenario | Drains | Corridor |
|---|---|---|
| S0 baseline-today | clogged (none) | as-is |
| S1 drains-working | capacity-capped inlets | as-is |
| S2 masar-built | clogged | green corridor edits |
| S3 masar+drains | capped inlets | green corridor edits |

Corridor edits follow the AUB design (PDF): infiltration strips / rain
gardens along the Fouad Boutros right-of-way, preserved vegetation,
permeable paving on the pedestrian spine — expressed with the existing
scenario polygon ops (`infiltrate`, `lower`, `raise`).

---

## 5. Phase D — Validation (first-class, automated)

Every run automatically produces `sanity_report.json` + a human-readable
page; a run that fails screens is flagged, not published.

**D1 Code-level:** keep `test_synthetic.py`; add analytic benchmarks:
(a) steady uniform-slope runoff → compare with kinematic q = i·L and
Manning normal depth; (b) still-lake test (no spurious currents, flat
water surface preserved); (c) volume conservation to machine precision
with closed boundaries.

**D2 Standard benchmark:** UK Environment Agency 2D benchmark (Néelz &
Pender 2013) **Test 8A** rainfall-on-urban-area (Glasgow, ~0.4 km² @ 2 m,
~97 k cells — runs in seconds). Inputs available in LISFLOOD-FP format on
Zenodo record 6907286 (official EA data is on-request); compare our torch
solver + SERGHEI against published depth/velocity curves (Itzï GMD 2017,
HEC RD-51).

**D3 Cross-engine:** finalist storms re-run in the adopted external engine
on identical rasters; compare depth maps (RMSE, hit/miss on >10 cm extent)
and hotspot rankings.

**D4 Convergence:** 2 m vs 1 m vs 0.5 m on the corridor; dt-halving check;
fp32 vs fp64.

**D5 Plausibility screens (the "2.5 m rule"):**
- static bound: per-cell max possible ponding = priority-flood fill depth
  of the terrain; dynamic max_depth must not exceed it (+ tolerance) —
  anything above is numerical, not physical;
- every cell > 1 m gets auto-classified: enclosed courtyard / DEM pit
  artifact / genuine depression (with ortho thumbnail in the report);
- velocity cap screen (flag |v| > 4 m/s off-stairs), wet-fraction and
  end-of-run storage fraction tracked across runs;
- outlet check: peak discharge at the two outlets vs rational-method
  Q = C·i·A envelope for the catchment.

**D6 Ground truth** (events compiled in `docs/research_rainfall_beirut.md`):
- **25 Nov 2025**: Sassine Sq area + Ring bridge ponded under a measured
  25.4 mm/30 min burst with drains "clogged within minutes" (minister
  quote) → V1 replay must show ponding at the corresponding low points;
- Dec 2023: Karantina/port flooding + **Beirut River overflow** (our east
  boundary) — checks the outlet framing;
- Dec 2019 / Jan 2019 (Norma) / Nov 2024: city-wide street flooding,
  underpasses — qualitative hotspot cross-check.
No news item names the stair streets specifically — rely on adjacent
documented points and ask BUL to annotate local hotspots on our map
(cheap, defensible, and engages the stakeholder).

**D7 Sensitivity:** Manning ±50 %, infiltration ±50 %, rain ±20 %, drain
capacity range, roof-policy toggle → tornado chart per headline metric, so
the report states uncertainty honestly.

---

## 6. Phase E — Visualization & communication

Everything georeferenced (COG + PNG in EPSG:32636 and 3857) so BUL can
drop layers into their ArcGIS hub.

- **2D animated maps:** depth over ortho and over OSM, timestamp + rain
  gauge inset; rendered to MP4 (existing matplotlib path, upgraded
  colormap/legend; depth colorbar capped at p99 with max annotated
  separately — never let 392 artifact cells set the scale again).
- **Flow, not just ponding:** velocity arrows / streamlines over the depth
  field at key moments; animated "particles" advected by (u,v) for the
  storyboard video — reads instantly for non-technical viewers.
- **3D flyovers:** PyVista (camera settings already tuned & memorized);
  A/B split-screen baseline-vs-scenario videos; camera path along the
  corridor from Mar Mikhael up to Achrafieh.
- **Hazard maps:** DEFRA-style h·(v+0.5) classes (safe / caution / danger
  for adults / danger for all) — the single most communicative static map.
- **Difference maps:** S2 − S0 depth deltas (blue = corridor helps).
- **Narrated storyboard (60–90 s MP4):** city context → rain begins →
  streets become rivers (particles) → hotspots pause with street names →
  corridor scenario replay → closing metrics. Assembled with ffmpeg from
  the pieces above; script written for a lay audience.
- **Interactive page:** single self-contained HTML (time slider + scenario
  toggle + hotspot gauges on downsampled rasters) — shareable as an
  Artifact link.

---

## 7. Phase F — Scenario analysis & report

- Metrics per hotspot and city-wide, per run: peak depth, time-to-10 cm,
  flooded area > 10 / > 30 cm, hazard-class areas, volume infiltrated /
  drained / discharged, gauge hydrographs.
- Answer the actual question: **how much flooding does Al-Masar Al-Akhdar
  remove, where, and under which storms** — with uncertainty ranges from
  D7.
- Report: methods, data lineage (manifests), assumptions (drains!, roofs,
  ellipsoidal datum), validation summary, results, recommendations
  (including where the corridor alone is insufficient, e.g. if S1 vs S2
  shows drains dominate).

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| 6 GB VRAM ceiling at 0.5 m full domain | fp32 state ≈ 1 GB — fits; if features push memory, tile with halo exchange or run 1 m |
| No storm-drain data exists | model capacity as scenario range S0↔S1, never a point claim |
| Photogrammetry: no ground under dense canopy, melted courtyards | interpolate + report interpolated fraction; courtyard mask + rain policy; manual patch layer |
| Bridges/overpasses dam the surface | explicit cut list with QA flow arrows (A3) |
| fp32 drift over ~50 k steps | fp64 mass accumulators + D4 fp32-vs-fp64 gate |
| External engine build pain on CUDA 13 | SERGHEI's Kokkos supports CUDA 13; LISFLOOD-FP used as CPU build only; SynxFlow (pip wheels, own venv) as backup; torch port is primary regardless |
| LISFLOOD-FP license ambiguity (GPL-2 vs "GPLv3 non-commercial", no LICENSE in zip) | verification-only role, nothing shipped depends on it; SERGHEI (BSD-3) is the quotable cross-check; get written clarification from Bristol if that changes |
| Laptop thermals on multi-hour GPU runs | checkpointing; run matrix at 1 m first; 0.5 m finals overnight |
| Scenario edits hurting (demo defect 3) | publish flow fields with every scenario; iterate placement using velocity maps |
| IDF table provenance undocumented in the Lebanese code | cross-checked against 2026 peer-reviewed satellite IDF (±10–25 %); both cited; sensitivity ±20 % on rain covers the gap |

## 9. Effort estimate (working days, solo)

A: 2–3 · B: 2–3 · C: 1 · D: 2–3 · E: 2–3 · F: 1–2 → **10–15 days**,
with E partially overlapping C/D (renders run while sims run).

## 10. Immediate next actions

1. `untwine` → COPC (background, overnight, needs ~80 GB temp) [A0]
2. Extend `city_flow.py` gridding to the multi-layer 0.5 m stack [A1]
3. Fetch OSM buildings/water/streets for the extent (pyrosm/overpass) [A2]
4. torch port of `simulate()` + test-block equivalence gate [B1]
5. Build SERGHEI + LISFLOOD-FP 8.2 (CPU); fetch Zenodo 6907286; run EA
   Test 8A on our solver and both engines [B2/D2]
