# Handover — Al-Masar Al-Akhdar Green Corridor Flood Study

Status as of **2026-08-21**. This document is meant to let someone (including a future
session with no memory of this one) pick the project up cold: what the simulation is,
what results exist and are trustworthy, what's mid-flight and not yet trustworthy, and
exactly what to do next.

**Read this first**: there are two separate layers of work in this repo right now.

1. **The published, validated pipeline** — everything under "Published Results" and
   "Pipeline Walkthrough" below. This is committed, tested, cross-validated against two
   independent flood engines, and is what `README.md`'s numbers describe. Nothing in
   this session touched it. It is safe to trust and cite.
2. **In-progress, uncommitted work** — an opt-in erosion sub-model (done, verified,
   safe), plus a richer land-cover classifier and a building-flattening fix for the
   corridor. All three are now technically complete and sit in scratch output
   directories (`output/terrain_cut_0.5_v3`, `output/corridor_gi_cut_v3`); what they are
   waiting on is human review, not more engineering. The long-running "buildings still
   look un-flattened in 3D" bug was a real DEM defect and is **fixed and verified** —
   see §9 for the root cause, the numbers, and what still needs sign-off. None of this
   has touched the live/published terrain or results yet.

---

## 1. What this project is

A GPU shallow-water (rain-on-grid) pluvial flood model for the **Al-Masar Al-Akhdar /
Fouad Boutros green corridor** in Beirut — an unbuilt highway right-of-way that Beirut
Urban Lab (AUB) has proposed converting into a green corridor instead. Built from a
39 GB airborne LiDAR survey of Beirut. The study answers: where does rainwater go in the
corridor and surrounding streets today, and how much does the proposed green corridor
change that (before vs. after), validated against independent flood-modelling software
and a published UK benchmark.

Primary deliverable: `output/report/almasar_combined_report.pdf` (20 pp).

## 2. Published results (trustworthy, committed, unaffected by this session)

Before/after the green corridor, 0.5 m resolution, across three storms (T2 frequent,
25-Nov-2025 observed, T50 severe):

| Metric | Result |
|---|---|
| Flooded street area on the corridor | ↓ 18–47% (47% T2, 36% observed, 18% T50) |
| Flooded area on walkable surfaces | ↓ 28–64% |
| Benefit radius | ~2 blocks (28–32% less flooding within 25 m, fades by ~100 m) |
| Infiltration | ~3×, absorbing 1000–2100 m³/storm |
| Peak discharge toward the port | ↓ 18–22%, delayed |
| Drain optimizer | 77 targeted inlets capture 86% of what a 600-inlet blanket removes |

**Validation**: LISFLOOD-FP agreement IoU 0.81 / RMSE 8.5 cm (test block); SynxFlow
(independent full-shallow-water solver) IoU 0.61 / corr 0.88 / RMSE 11 cm (full corridor
domain); passes the UK Environment Agency **Test 8A** benchmark (Néelz & Pender 2013).

These numbers live in `README.md`, `docs/crosscheck_lisflood.md`, and the report PDFs.
**They will change** once Part C (below) regenerates the pipeline with the richer
land-cover classifier and building-flattening fix — that is expected and must be called
out explicitly when it happens, not silently absorbed.

## 3. The simulation

**Physics**: Bates, Horritt & Fewtrell (2010) inertial shallow-water scheme, semi-implicit
Manning friction, donor-cell flux limiter. Ported to `torch` (CUDA or CPU), fp32 state.
Rain-on-grid (hyetograph forcing), spatially varying Manning n and infiltration, storm
drains with per-inlet capacity, roof/courtyard rain rerouting (downspout model), open
boundary at domain edge/water. Mass balance closes to ~1e-5 relative error.

**Solver**: `scripts/flood_gpu.py::simulate()` — the single most important function in
the repo. Takes a DEM + material rasters + a storm hyetograph, runs the CFL-limited
time-stepping loop, writes `depth_*.npy` frames (fp16), `max_depth.npy`, `final_depth.npy`,
`max_vel.npy`, `max_hazard.npy`, `run_meta.json` (full mass-balance breakdown: rain,
infiltrated, drained, outflow, stored, closure).

**Terrain a run consumes** (`load_terrain(dir)`): `dem.npy` (float32), `dem_transform.json`
(georeferencing), `masks.npz` (`valid`/`building`/`water`/`courtyard`/`eligible`),
`manning.npy`, `infil_mmh.npy`, `rain_weight.npy`, `gauges.json`, and (as of this session)
`erodible.npy`.

**Key non-obvious facts**:
- Z is **ellipsoidal**, not orthometric — sea level ≈ +26 m, never threshold depth against 0.
- The LAS has no classification/returns/intensity fields.
- Storm `t2` duration is 5400 s, not 3600 s.
- `flood_sim.py` (legacy numpy solver) is kept only as the port-correctness reference for
  `validate.py::test_equivalence` — the real solver is `flood_gpu.py`.

## 4. Pipeline walkthrough (script by script, in run order)

**Terrain, from the LiDAR survey**
- `build_stack.py` — streams the LAS once, grids per 0.5–1 m cell: `zlow`/`zhigh`
  (lowest/highest return), `count`, mean `rgb`. Writes `output/stack_*/`.
- `build_terrain.py` — the land-cover classifier + hydraulic-surface builder. DEM
  cleaning (hole-fill, despeckle), land-cover classification (see §7), building/water
  masks from OSM, canopy removal + ground re-interpolation, courtyard detection,
  roof/courtyard rain-rerouting raster, Manning/infiltration/erodibility rasters, static
  fill-bound (GPU morphological reconstruction), auto-placed gauges. Writes
  `output/terrain_<res>/`.
- `cut_domain.py` / `crop_cloud.py` — crop the high-resolution corridor domain (corridor
  strip + its D8 upslope catchment + buffer) out of the full city. One-time; the result
  (`output/cut_polygon.json`) is already committed, no need to rerun.

**Green corridor**
- `build_corridor_gi.py` — from the official 19 ha zone polygon
  (`output/masar_zone_official.json`), derives the ROW centreline + transverse material
  bands (vehicular lane, porous bikelane, bioswale, porous sidewalk, garden), places
  bioretention ponds at low points and terraces on steep spine segments. Writes
  `output/corridor_gi_cut/material.npy`.
- `bake_corridor.py` — applies each material class's infiltration/Manning/
  detention-depth/erodibility (`PROPS` table, literature-grounded) onto a **copy** of the
  base terrain. Writes `output/terrain_cut_corridor/`. This is the "after" scenario's
  terrain.

**Simulation & drainage**
- `flood_gpu.py` — the solver (§3).
- `drains.py` — uniform inlet network; `optimize_drains.py` — targeted minimal inlet set
  on the ponding hotspots (greedy, min-spacing).
- `run_corridor_study.sh` — runs the before/after/drains-only/after+drains scenarios
  across all 3 storms. **Idempotent**: skips any output dir that already has
  `max_depth.npy`, so delete stale dirs to force a rerun.
- `analyze_corridor.py` — corridor-focused metrics (flooded area by distance band,
  depth percentiles, water-balance %) → `output/corridor_runs/metrics.json` + figures.

**Validation**
- `validate.py` — 5 analytic tests (closed-box conservation, planar runoff vs. Manning,
  lake-at-rest, numpy/torch port equivalence, erosion sanity — see §8). Run this after
  *any* change to `flood_gpu.py`.
- `export_ascii.py`, `crosscheck.py`, `synxflow_run.py`, `lisflood_mass.py`,
  `run_validation.sh` — cross-engine comparison against LISFLOOD-FP and SynxFlow.
- `benchmark_ea8.py` — UK EA Test 8A benchmark (external, independent of Beirut terrain).

**Reproducibility note**: the 8.4 GB cropped corridor point cloud is too large for git;
published as a GitHub Release asset (LAZ, split in two parts). The terrain grids the
solver actually consumes are included directly in the repo, so the corridor study
reproduces without needing the point cloud at all — see `REPRODUCE.md`.

## 5. Tests

`scripts/validate.py` (run via `python scripts/validate.py`), 5 tests:

1. **`test_closed_box`** — rain into a walled basin; asserts volume conservation
   (`vol_rain == vol_stored + vol_outflow`) and no leakage over walls.
2. **`test_planar_runoff`** — steady rain on a sloped plane; asserts outflow → i·A at
   equilibrium and depth matches the analytic Manning normal-depth formula.
3. **`test_lake_at_rest`** — still water in a bowl; asserts no spurious currents (the
   well-balanced-scheme check).
4. **`test_equivalence`** — synthetic urban DEM: torch fp64+legacy-limiter vs. the numpy
   reference solver (port-correctness gate), then fp64 vs. fp32 production config
   (statistical equivalence under the production limiter).
5. **`test_erosion_sanity`** *(added this session)* — confirms `erosion=False` is
   bit-identical to passing no erosion kwarg at all; non-erodible cells stay exactly
   zero under erosion; mass balance still closes with erosion on (proves the "no new
   volume term needed" design claim empirically).

There is no formal test suite for the sandbox (`sandbox/`, `webui/`) — it's been verified
by hand each session via curl/API calls and headless-browser (Puppeteer) screenshots, not
automated. Worth building real tests if the sandbox becomes a long-term maintained tool.

## 6. Figures & QA outputs, and why they exist

- `*_qa.png` files inside each `output/terrain_*/` dir (`landcover_qa.png`,
  `courtyards_qa.png`, `hillshade.png`) — sanity-check the terrain build itself, no
  independent ground truth exists so these are eyeballed, not scored.
- `output/terrain_cut_0.5_v3/buildings_flatten_qa.png` *(scratch only, not published)*
  — colour-codes every building near the corridor as flattened (red) / kept-standing
  (blue) / kept-heritage (gold) / kept-institutional (teal) / kept-manual (purple), with
  the highway alignment buffer in green. Built specifically to let a human sanity-check
  the flatten logic against the real reference plan before it goes anywhere near
  published results — see §9.
- `output/corridor_runs/metrics.json` + `analyze_corridor.py`'s figures — the actual
  quantitative before/after comparison (flooded area by band, depth percentiles).
- `render2d.py` — colormap conventions (`turbo` for depth, DEFRA hazard classes,
  `RdBu_r` diverging for diffs) shared between the CLI figures and the sandbox's live PNG
  encoder (`sandbox/encode.py`) — kept in sync deliberately so a sandbox screenshot and a
  report figure read the same way.
- `output/report/almasar_combined_report.pdf` — the actual deliverable. Built from
  `greencorridor_report.pdf` (planner-facing) + `corridor_report.pdf` (validation
  write-up). **No report-build script exists** — figures are manually copied into
  `output/report/figs/` and the `.tex` prose is hand-authored/compiled. Keep this in mind
  before Part C: regenerating the pipeline means manually redoing this step too.

## 7. Benchmarks — done vs. not done

**Done, currently in the published report:**
- LISFLOOD-FP 8.2 (ACC scheme, built locally without sudo) — test-block agreement
  IoU 0.81, RMSE 8.5 cm, bias −2.5 cm. **Diverges on the corridor cut itself** (steep
  stepped terrain — LISFLOOD's ACC scheme fabricates volume there even with tuned
  theta/CFL; documented as a known engine limitation, not a bug in our solver — see
  `docs/crosscheck_lisflood.md`).
- SynxFlow (independent full-shallow-water Godunov scheme, GPU) — test-block IoU
  0.78/corr 0.898/RMSE 6.4 cm; **full corridor domain** (the actual hard-terrain case
  LISFLOOD couldn't handle) IoU 0.61/corr 0.875/RMSE 11.1 cm/bias +5 cm — the auto
  verdict prints DIVERGE only because the pass/fail thresholds were tuned on the gentle
  test block; the residual is the expected inertial-vs-full-SWE spreading, not a real
  disagreement.
- UK Environment Agency **Test 8A** (Néelz & Pender 2013) — `benchmark_ea8.py`, fully
  external/independent of the Beirut terrain, in-cluster with the published envelope
  (`docs/ea8_published_envelope.json`).

**Requested this session, NOT yet done — pending:**
> The user explicitly asked to "recreate the External benchmark: UK EA Test 8A and
> whatever other benchmark I can find" after visually confirming the fixed corridor.
> That visual confirmation never completed (see §9's open issue), so **this benchmark
> re-run never happened this session**. It should be step one whenever this picks back
> up, independent of whether Part C's terrain regeneration is ready:
> ```bash
> python scripts/validate.py
> python scripts/benchmark_ea8.py
> bash scripts/run_validation.sh   # LISFLOOD + SynxFlow legs
> ```
> Also worth actively researching: the UK EA published several other Neelz & Pender
> test cases beyond 8A (the full 2013 test suite has ~9 cases covering different
> hydraulic regimes — channel flow, floodplain flow, etc.). Only 8A has been attempted
> here. Check `~/Work/engines/ea_benchmark/` for what's already downloaded locally
> before assuming a fresh download is needed, and check whether the EA's test data
       licence permits redistributing results.

## 8. This session's work — what's actually done and verified

### 8a. Opt-in erosion sub-model — DONE, verified, safe to build on

`scripts/flood_gpu.py::simulate(erosion=False, ...)` — soil/soft-material cells
probabilistically get wetter and erode under local flow (driven by the solver's own
unit-discharge, no new physics tensor needed), lerping their live infiltration/Manning
toward a "mud" floor. Puddling on eroded cells is an emergent consequence of that (lower
infiltration + lower roughness under the unmodified SWE update) — no new mass-balance
term needed, `closure` stays the same identity as always.

- **Fully inert when off** (the default) — proven bit-for-bit identical to not passing
  any erosion kwarg at all, not just "numerically close." All 4 pre-existing
  `validate.py` tests pass completely unchanged.
- `bake_corridor.py`'s `PROPS` table and `sandbox/state.py`'s `default_materials()` both
  carry a 5th "erodibility" field per material (bioswale 0.5, garden 0.4, bioretention
  0.3, terrace 0.2, paved 0). New `erodible.npy` per terrain dir.
- Sandbox: "Enable erosion" checkbox in `RunPanel.tsx`, threaded through
  `RunBody`→`JobQueue`→`simulate()`; `sandbox/encode.py::erosion_png` + a new
  `GET /api/runs/{id}/erosion.png` endpoint; `ResultsView.tsx` shows an "Erosion" overlay
  button only on runs made with it on. Verified end-to-end via the live API.
- **Explicitly not independently calibrated** — say so wherever erosion results are
  shown, unlike the `PROPS` table's literature-grounded infil/Manning values.
- Not yet applied to any *published* run — the before/after study still runs with
  erosion off, by design, to keep the headline comparison apples-to-apples.

### 8b. Richer land-cover classifier — built, staged, awaiting review

`scripts/build_terrain.py`: 6 classes → 9 (added `gravel_unpaved`, `shrub`,
`paved_concrete`), using slope (`np.gradient(dem)`) and local height-roughness
(`scipy.ndimage.uniform_filter`, windowed std) derived from data already on disk — no
new LAS pass needed. Run into a **scratch directory**
(`output/terrain_cut_0.5_v3`, NOT the live `output/terrain_cut_0.5`) specifically so it
can be reviewed before being trusted.

One real bug found+fixed during this: the roughness threshold search first saturated at
its clamp because building roof/ground edges dominated the histogram and Otsu doesn't
suit its heavy-tailed shape — fixed by excluding a building halo and switching to an
adaptive top-percentile cut.

**Status**: technically complete, never got a final go/no-go from the user because
attention moved to the building-flattening bug below. `output/terrain_cut_0.5_v3/
landcover_qa.png` is the artifact to review.

### 8c. Building-flattening fix — the real unresolved item, see §9

`scripts/flatten_corridor_buildings.py` (new script) — identifies which buildings the
real Al-Masar Al-Akhdar design actually demolishes (today, `build_corridor_gi.py`
routes the corridor ribbon around *every* standing building alike — the real project
only clears the ones actually in the highway's path, keeping historic/institutional
buildings and repurposing them). This went through many rounds of user-reported bugs,
each traced to a genuine root cause (not guessed away) — full detail in §9.

## 9. RESOLVED — 3D view showed buildings as not flattened

**It was a real data bug, in the DEM, exactly as the user kept reporting.** Not a cache
problem, not a perception problem, not user error. The session that wrote the previous
version of this section had verified the fill on the buildings it happened to sample and
concluded the data was correct; it was correct for those, and wrong for others.

**Root cause.** `flatten_corridor_buildings.py` filled each demolished footprint from its
nearest open-ground cell, with `fill_sources = valid & ~building`. But `masks['building']`
comes from an OSM extract that is *demonstrably incomplete over this domain* — the same
systemic gap already flagged in §10. An unmapped building therefore reads as "open
ground" **at roof height**, and the nearest-neighbour fill happily flattened a demolished
building down onto its neighbour's rooftop.

Measured on `terrain_cut_0.5_v2` (the terrain under review when the bug was reported):

| | before fix | after fix |
|---|---|---|
| Flattened footprints still standing >3 m above their own surrounding ground | **14 of 35** | **0 of 35** |
| Worst footprint | **+23.1 m** | +2.4 m |
| Flattened cells >3 m above local ground | 38.1 % | 3.8 % |
| Worst single cell | **+51.2 m** | +8.3 m |

The two worst cases were flanked by tall blocks carrying no building mask at all —
54–79 % of their cells were filled from a source that was itself >3 m above real ground.

**The fix**: `unmapped_structure_mask()` in `flatten_corridor_buildings.py` — a
morphological white top-hat (flat structuring element wider than any real footprint,
default 60 m; `--structure-se-m` / `--structure-relief-m`). An opening is idempotent on
features wider than its element, so hillsides and terrace levels survive it and cancel to
~0, while a bounded building-sized plateau is erased and shows its full height. Those
cells are then vetoed as fill sources.

It is deliberately a **veto, not a classifier** (~80 % recall on known-building cells, and
it also flags some genuine high terrace ground). That asymmetry is the safe direction:
vetoing true ground only makes the fill pick the next-nearest ground cell, whereas missing
an unmapped roof puts a 20 m step back into the DEM. Correctness was therefore judged on
the *outcome*, in both directions — nothing left standing, and nothing sunk into a pit —
not on the detector's own accuracy. The result is insensitive to the parameters (SE
30/40/60 m × threshold 2/3 m all give 0 standing, 0 pits).

**Residual, stated honestly**: 3.8 % of flattened cells still sit >3 m above a 30 m-window
10th-percentile ground reference, max +8.3 m. Every one of them took its elevation from
real ground within ~5–8 m whose own top-hat relief is ≤2 m. This corridor genuinely is
terraced with multi-metre retaining walls (`build_corridor_gi.py` places terraces on these
same segments), so this is real terrain, not un-flattened roof. Lower `--structure-relief-m`
if a future review disagrees.

**Verification artifacts** (in `output/terrain_cut_0.5_v3/`):
- `flatten_fix_qa.png` — before/after hillshade + elevation section through the three worst
  footprints. The "before" trace shows a 20–25 m plateau inside the footprint; "after"
  tracks surrounding ground.
- `flatten_3d_ab.png` — the **sandbox's own 3D view**, same camera, old terrain vs new,
  captured through a real headless Chrome against the running server. The marker-coloured
  slabs standing at roof height on the left are flat ground on the right.
- `buildings_flatten_qa.png` — unchanged classification (37 flattened / 337 kept / 21
  heritage / 3 institutional / 2 manual); the fix changed the DEM fill, not which buildings
  are demolished.

**Terrain lineage**: `terrain_cut_0.5_v2` is superseded. The corrected terrain is
`output/terrain_cut_0.5_v3` (fresh `build_terrain.py` — verified to reproduce v2's
pre-flatten DEM and masks bit-for-bit — then the fixed flatten), with
`output/corridor_gi_cut_v3/` rebuilt against its masks (ROW ribbon 2.94 ha, up from
2.76 ha, since cleared footprints free up ribbon space).

**Two contributing robustness bugs fixed alongside**, both of which made this loop harder
to close than it needed to be:
1. `sandbox/server.py` served `webui/dist` through a bare `StaticFiles`, which sets no
   `Cache-Control` at all — leaving browsers free to apply heuristic freshness to
   `index.html`, the one file whose URL never changes while its contents do. A stale
   `index.html` pins an old hashed JS bundle indefinitely regardless of how fresh the API
   data is. Now served `no-cache` via `NoCacheIndex` (hashed assets still cache normally).
2. `sandbox/state.py` hardcoded `"base_terrain": "output/terrain_cut_0.5"` in every
   design's metadata. Under a `SANDBOX_TERRAIN_DIR` override that is the *one field*
   telling a reviewer which terrain they are looking at, and it reported the live terrain
   during a scratch-terrain review. Now reports the real `TERRAIN_DIR`.

**Still outstanding on this item**: user sign-off. The verification above is mine, not the
user's. Review server:

```bash
SANDBOX_TERRAIN_DIR=output/terrain_cut_0.5_v3 \
SANDBOX_DATA_DIR=output/sandbox_review_v3 \
SANDBOX_OFFICIAL_MATERIAL=output/corridor_gi_cut_v3/material.npy \
uvicorn sandbox.server:app --port 8008
```

Open it in a **new private window**, confirm the Design panel reads
`base_terrain: output/terrain_cut_0.5_v3`, and check the 3D view.

### 9b. Follow-up round — spikes, grading, nodata slab, deferral, corridor holes

Five further items, all from direct review of the 3D/2D views. Current terrain is
`output/terrain_cut_0.5_v3` + `output/corridor_gi_cut_v3`; both were rebuilt from
scratch for this round, so anything measured against the older grid is stale.

**1. Sharp spikes left standing inside cleared lots.** `build_terrain.py`'s courtyard
pass carves interior light-wells out of `masks['building']`, so they were not in
`flatten` and kept their original elevation — which for a roof-level interior void is
roof height. 29 cells in 6 clusters survived demolition as needles up to **+42 m** out
of the cleared lot. Anything fully enclosed by a cleared footprint (and not itself a
kept building) is now cleared with it. Enclosed-but-not-cleared cells: **0**.

**2. Grading over cleared lots.** `smooth_flattened()` — Jacobi smoothing with the
surrounding ground as a fixed boundary, default radius 3 m (`--flatten-smooth-m`).
Two bugs found while adding it, both the *same* root cause as §9's original defect one
layer up: the averaging support has to exclude roofs, and `valid & ~building` is not
enough because of the unmapped ones. With a naive support the smoothing put **12 of 41**
footprints back to standing, worst **+38 m** — it reintroduced exactly the defect the
fill fixes. Support is now the fill's own trusted-ground set plus the cleared lots.
Measured effect of smoothing at that point (radius 0 → 3 m): worst residual +8.3 → +6.3 m,
cells >3 m above local ground 4.0% → 2.2%, max single-cell step inside a lot
4.34 → 1.87 m, seam spikes 32 → 0. Nothing standing and nothing sunk, at every radius.

**3. The black slab.** The corridor cut is a polygon gridded onto its bounding box, so
the survey boundary left the southern **394 rows nodata edge to edge** — 9.9 ha, 17% of
the domain, rendering as a solid black slab in 2D and 3D. `build_terrain.py
--crop-to-valid` trims fully-empty border rows/columns and shifts the transform with
them. Grid 2348×1142 → 2001×1142; nodata 15.5% → 0.84%. Georeferencing verified
unchanged: 4000 random UTM points sample identical elevations on the old and new grids
(max difference 0.000000 m). Only fully-empty *border* rows go; interior nodata is
untouched.

**4. Clearing deferred to the corridor.** Previously the flatten was baked into the
terrain, so the "before" world was already cleared and before/after was not a real
comparison. `--defer-to-design` now leaves `dem.npy`/`masks.npz` as the before state —
every building standing, only marker-tinted in the ortho — and writes the clearing as
`flatten_delta.npy` + `flatten_mask.npy` + `rain_weight_cleared.npy`. Loading the
official green corridor is what demolishes them.

- `sandbox/state.py`: `Base` loads those layers; `Design.clears_buildings` records
  whether a design applies them. The clearing is deliberately **not** folded into
  `design.dem_delta` — that array is user sculpting and is clamped to
  ±`DEM_DELTA_LIMIT_M` per cell, so a 25 m demolition stored there would be clipped
  back to 3 m the moment anyone sculpted the lot and the building would visibly grow
  back. It is a terrain layer the flag switches on, served at
  `GET /api/terrain/flatten_delta.bin` (`/api/meta.has_clearing`).
- `sandbox/baking.py` and `scripts/bake_corridor.py` both apply it before the material
  bake, keeping the CLI bake and the sandbox design equivalent — the property
  `baking.py` exists to preserve.
- Rain rerouting moves with the buildings: cleared roofs stop feeding downspouts and
  the lots catch their own rain (`rain_weight_cleared.npy`, mirrors `build_terrain.py`
  §5). Rain volume is conserved exactly — 2 055 752 rain cells, equal to the base.
- Frontend: `clearDelta`/`clearsBuildings` in the store, added in every place that
  computes a rendered ground height (`Scene3D` mesh build, live sculpt region, picking
  mesh, `terrain.computeEffectiveZ`) and in `editor.ts`'s `groundAt()` so the flatten
  tool targets the cleared ground rather than the height of the demolished building.

**5. City-block-sized holes in the corridor.** `build_corridor_gi.py` lays the ribbon as
`zmask & ~building`, and the demolition criterion was overlap with the 2.5 m alignment
linework buffer — but the corridor that actually gets built is the ~36 m ROW ribbon
around it. A building can sit squarely inside the ribbon and miss the linework, and the
ribbon then routes around it. Of the 3.57 ha ROW band, **0.64 ha was blocked**; 0.23 ha
of that by ordinary buildings with no protection at all, the largest void 1080 m².
A building with ≥`--row-overlap-frac` (0.5) of its own footprint inside the ribbon is now
a demolition candidate on the same terms, with the same heritage/institutional overrides.
11 more candidates, 8 more demolished (37 → 45 buildings, 1.77 → 2.00 ha).
Ribbon 2.94 → 3.18 ha. `build_corridor_gi.py` reads `flatten_mask.npy` so it lays the
ribbon out for the cleared world.

**The remaining 0.41 ha of holes is protected buildings and was left alone on purpose** —
identified heritage and institutional/religious (schools, churches, the university). The
real Al-Masar design preserves and repurposes those; filling them with garden would
misrepresent the project. Two of the largest remaining voids (954 m², 587 m²) are in that
category. If any specific one should come out, it needs a decision, not a threshold.

**Verification of the after state** (41 cleared footprints): still standing >3 m: **0**;
sunk into a pit: **0**; seam spikes: **0**; max single-cell step inside a lot **1.87 m**;
per-cell height above local ground median +0.57 m, p99 +3.6 m, max +6.3 m. The 2.2% above
3 m is this corridor's genuine terracing (every one took its elevation from real ground
within ~5–8 m whose own top-hat relief is ≤2 m), not un-flattened roof.

**Artifacts**: `before_after_3d.png` (the sandbox's own 3D view, same camera, base terrain
vs official corridor loaded, captured through headless Chrome against the running server),
`corridor_2d_qa.png` (2D with the corridor loaded, no black slab), `buildings_flatten_qa.png`,
`landcover_qa.png`.

**Not verified by the user yet.** Same caveat as §9.

## 10. Other known loose ends

- **`output/terrain_cut_corridor/drains_opt.npz` drift**: rebaking the corridor for the
  erosion work forced this git-tracked file (backs the published "77 inlets, 86%
  capture" numbers) to want to regenerate as 84 inlets from today's local
  `before_v1_nov2025/final_depth.npy`. `optimize_drains.py` itself is unchanged since the
  initial commit — this is local input drift, not something this session caused, but
  it's real: **a fresh run of the documented reproduction steps does not currently
  reproduce the committed 77-inlet file bit-for-bit.** Reverted to the committed version
  rather than overwrite it; worth investigating properly at some point (diff exactly what
  changed about `before_v1_nov2025/final_depth.npy` versus whatever produced the
  original).
- **Missing-from-`masks['building']` gap is systemic, not isolated**: while chasing the
  "college gets flattened" report, found 11 of 126 heritage buildings and 1 of 17
  institutional buildings in the corridor domain are also missing from the OSM building
  extract despite having real elevation data. Fixed for the buildings this session's
  protection layers happen to cover; **not** fixed for the general case (any building
  outside those specific heritage/institutional lists that also happens to be missing
  from the OSM extract). If Part C ever needs a fully-correct citywide building mask,
  this needs a proper pass, not just patching the cases that happened to surface.
  **Update**: this gap was the root cause of §9's flattening bug, and
  `unmapped_structure_mask()` now detects unmapped structures from the DEM itself. That
  detector is scoped to vetoing *fill sources* — it does **not** write into
  `masks['building']`, so everything downstream of the mask (courtyards, rain rerouting,
  fillbound, the corridor ribbon's `~building` test) still sees the incomplete OSM mask.
  Folding it into the mask properly is the real fix and still belongs in Part C; it would
  change hydraulics, so it is deliberately not being smuggled in as part of a bug fix.
- **Design note already flagged for Part C**: `flatten_corridor_buildings.py` is
  currently a post-hoc raster patch (good enough to validate the concept). For the real
  pipeline regeneration, flattening should happen *earlier* — filtering `osm["buildings"]`
  against the highway buffer (minus heritage exceptions) in vector space, before
  `build_terrain.py`'s own courtyard/rain-rerouting/fillbound steps run — so those
  derived layers are computed correctly from the start instead of needing the same kind
  of after-the-fact patch this session did.

## 11. Code map

```
scripts/            the whole batch pipeline (Python, run via CLI, see §4)
sandbox/            FastAPI backend for the interactive web sandbox
  state.py            Base terrain (read-only, loaded once), Design/DesignStore
                       (per-design material.npy + dem_delta.npy, server-enforced
                       edit-zone locking), materials table (mirrors bake_corridor.PROPS)
  baking.py           in-memory bake, mirrors bake_corridor.py exactly (proves a
                       zero-edit sandbox design reproduces the pipeline bit-for-bit)
  jobs.py             single-worker GPU job queue, wraps flood_gpu.simulate with a
                       progress callback relayed over websocket
  encode.py           npy -> PNG colormap encoders (depth/hazard/diff/erosion),
                       kept in sync with render2d.py's conventions
  metrics.py          per-run metrics (generalizes analyze_corridor.py's logic)
  server.py           the whole REST + WebSocket API
webui/               React + TypeScript + Vite + three.js frontend
  store.ts             central zustand store (designs, materials, run progress, etc.)
  MapView.tsx          2D canvas editor (paint/sculpt tools, layered canvases)
  Scene3D.tsx           3D viewer (heightfield mesh + separate coarse "picking" mesh
                       for fast raycasting, live-updates during sculpting)
  editor.ts            Stroke/UndoManager (brush logic shared between 2D/3D),
                       material color table, elevation heatmap rendering
  terrain.ts           terrain data fetch + effective-Z computation + nearest-fill
                       (fillInvalidNearest - same pattern as this session's DEM fix)
  RunPanel.tsx          storm picker + custom hyetograph editor + run trigger
  ResultsView.tsx       results library, animated depth/hazard/erosion overlays,
                       before/after compare with diff maps
docs/                research notes, validation write-up, this file
output/              all generated data (gitignored except small reproducibility
                     artifacts like *.json zone/highway/heritage polygons and the
                     git-tracked drains_opt.npz / corridor_bake.json)
```

## 12. How to run things

```bash
# validate the solver after any flood_gpu.py change
python scripts/validate.py

# launch the interactive sandbox (production mode, port 8008)
bash sandbox/run.sh

# launch it pointed at scratch/review terrain instead of the live one
SANDBOX_TERRAIN_DIR=output/terrain_cut_0.5_v3 \
SANDBOX_DATA_DIR=output/sandbox_review_v3 \
SANDBOX_OFFICIAL_MATERIAL=output/corridor_gi_cut_v3/material.npy \
uvicorn sandbox.server:app --port 8008
# (env vars documented in sandbox/state.py; unset, behaves exactly as before)

# reproduce the published corridor study from included terrain grids
python scripts/bake_corridor.py --terrain output/terrain_cut_0.5 \
    --material output/corridor_gi_cut/material.npy --out output/terrain_cut_corridor
bash   scripts/run_corridor_study.sh
python scripts/analyze_corridor.py

# frontend dev loop
cd webui && npm run dev          # hot-reload, needs `uvicorn ... --reload` separately
cd webui && npm run build        # production build, served by sandbox/run.sh
```

## 13. Priority-ordered next steps

1. **Get user sign-off on the flattening fix** (§9). The bug itself is found, fixed and
   verified in the data, in a hillshade/section figure and in the sandbox's own 3D view —
   but every one of those checks is mine, not the user's, and the whole history of this
   item is verification that convinced me and not them. Sign-off is the remaining step,
   not more engineering.
2. **Get explicit sign-off** on `output/terrain_cut_0.5_v3/landcover_qa.png` (richer
   classifier), `buildings_flatten_qa.png` (flatten logic) and `flatten_fix_qa.png` (the
   fix itself) — review `_v3`, not the superseded `_v2`.
3. **Re-run the external benchmarks** — `validate.py`, `benchmark_ea8.py`, and
   `run_validation.sh` — this was explicitly requested and never completed this session.
   Independent of Part C; can happen against the *current published* terrain right now
   if useful as a fresh baseline, then again after Part C to show what changed.
4. **Investigate whether other EA Neelz & Pender test cases are worth adding**, beyond
   Test 8A.
5. **Part C** (only after 1–2 are resolved): rebake, rerun the full before/after/drains
   study, re-run cross-engine validation, regenerate figures, hand-update the report
   prose (no report-build script exists — this is manual), update `README.md`'s numbers.
   Expect the headline flooded-area-reduction numbers to shift; that's expected and must
   be stated plainly, not smoothed over.
6. Properly investigate the `drains_opt.npz` 77-vs-84 reproducibility drift (§10) at some
   point — not urgent, but a real gap in "the documented reproduction steps reproduce the
   committed artifacts exactly" that's currently just being worked around.
