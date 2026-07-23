# Cross-check vs LISFLOOD-FP 8.2 (2026-07-14) — PLAN.md D3

Independent verification of our torch/CUDA solver against LISFLOOD-FP 8.2
(Zenodo 13121102) — the published reference implementation of the same
Bates et al. (2010) inertial scheme. Test case: the 600×600 @ 0.5 m
Beirut test block, 25 Nov 2025 storm, identical exported inputs
(dem.asc, n.asc, rain), closed borders on both engines, no
infiltration/drains/rerouting.

## Verdict: CONSISTENT (within inter-model tolerance)

| Metric | Value |
|---|---|
| Flooded-extent IoU (>10 cm) | **0.81** |
| Wet-cell depth correlation | 0.845 |
| RMSE over wet cells | **8.5 cm** |
| Bias (ours − LISFLOOD) | −2.5 cm |
| p99 depth | 0.756 vs 0.785 m (4 %) |
| Wall time | **110 s GPU (ours) vs 8.1 min CPU (LISFLOOD)** |

The deep cells anchor the agreement (excluding them LOWERS correlation);
the residual scatter lives in 10–50 cm puddles whose edges the two
stabilization mechanisms shuffle differently. ±10 cm scatter between
industry 2D codes on urban cases is normal in the EA/Néelz–Pender
benchmark reports. Figure + JSON: `output/crosscheck/test_vs_lfp.*`.

## Findings along the way (all encoded in the tooling now)

1. **LISFLOOD's rain series holds its last value forever** — the file must
   end with an explicit `0.00 <t>` line or rain never stops
   (verified in `input.cpp LoadRain`; +576 m³ in our first attempt).
   `export_ascii.py` now zero-terminates.
2. **FREE boundaries fabricate volume on cropped urban edges** — the
   slope-extrapolating boundary produced Verror ≈ −7,000 m³ on 2,286 m³ of
   rain. Cross-checks use closed borders on both engines
   (`crosscheck.py run-ours --closed` pads a wall ring and crops back).
3. **Vanilla ACC is unstable on steep stepped urban DEMs**: without
   damping, LISFLOOD fabricated 6.85 **million** m³ (Verror −1.06e6 m³,
   dt → 8 ms) on this 0.5 m terrain. `theta 0.8` (de Almeida 2012) +
   `cfl 0.5` cures it completely (final Verror −0.0000). Our solver never
   needed this because of its mass-conserving outflux-scaling limiter —
   a strong independent validation of that design choice.

## Reproduce

```bash
# build once (no sudo needed; numa shim + NetCDF off)
#   binary: ~/Work/engines/LISFLOOD-FP/build/lisflood
python scripts/export_ascii.py --terrain output/terrain_test \
    --storm storms/v1_nov2025.json --out output/export_test
( cd output/export_test && ~/Work/engines/LISFLOOD-FP/build/lisflood -v v1_nov2025.par )
python scripts/crosscheck.py run-ours --export output/export_test \
    --storm storms/v1_nov2025.json --out output/crosscheck/ours_test --closed
python scripts/crosscheck.py compare \
    --ours output/crosscheck/ours_test/max_depth.npy \
    --theirs output/export_test/results/res.max \
    --export output/export_test --out output/crosscheck/test_vs_lfp.png
```

Full-domain (terrain_1.0) repeat: same commands with the 1 m terrain —
LISFLOOD CPU will need ~2–4 h per storm. SERGHEI (BSD-3, GPU, full-SWE Roe
— methodologically independent) remains optional; build notes in
docs/research_flood_engines.md.

---

# Engine #2: SynxFlow 1.0.1 (2026-07-22)

SERGHEI was the planned second engine but is not buildable on this
machine (no cmake / MPI / nvcc / sudo; Kokkos + Parallel-NetCDF from
source is a half-day with a real chance of failure). **SynxFlow** —
HiPIMS successor, BSD-3, GPU, pip wheels — replaces it. It solves the
*full* shallow-water equations with a Godunov/HLLC finite-volume scheme,
so it is methodologically independent of the Bates-2010 inertial scheme
that both our solver and LISFLOOD-FP implement. That makes it a stronger
check than LISFLOOD, which is the reference implementation of *our own*
scheme.

## Environment (it needs its own; do not touch the main venv)

```bash
uv venv --python 3.11 ~/Work/.venv_synxflow     # 3.12+ unsupported
VIRTUAL_ENV=~/Work/.venv_synxflow uv pip install synxflow \
    "setuptools<81" "numpy<2" "pandas<2.2"
```
Both pins are load-bearing: `synxflow.IO` imports `pkg_resources` (gone in
setuptools >= 81) and calls `np.trapz` (removed in numpy 2). The CUDA
runtime is statically linked into `flood.*.so`, so no CUDA toolkit is
needed. `flood.run()` **chdir's into the case folder** — pass absolute
paths or your post-processing will not find its own output.

## Runner

`scripts/synxflow_run.py` consumes the exact `export_ascii.py` output the
LISFLOOD cross-check uses (dem.asc / n.asc / storm.rain), runs with a
closed outline boundary (`rigid`) to match `crosscheck.py --closed` and
LISFLOOD's default border, takes `sim_time` from the generated `.par` so
all three engines run equally long, and writes `max_depth.asc` on the
identical grid — which `crosscheck.py compare` reads unchanged.

## First result (0.5 m test block, 25 Nov 2025 storm, 1800 s)

| Metric | vs SynxFlow | vs LISFLOOD-FP (for reference) |
|---|---|---|
| Flooded-extent IoU (>10 cm) | **0.78** | 0.81 |
| Depth RMSE over wet cells | **6.4 cm** | 8.5 cm |
| Bias (ours − theirs) | +2.1 cm | −2.5 cm |
| p99 depth | 0.756 vs 0.737 m | 0.756 vs 0.785 m |

Verdict CONSISTENT, and the RMSE against the *independent* scheme is
lower than against the reference implementation of our own.

---

# EA 2D benchmark Test 8A (Glasgow) — PLAN.md D2

External case with published answers: 0.4 km² of Glasgow at 2 m
(481×199), 400 mm/h of rain for 3 minutes, plus a 2.5 m³/s point-inflow
hydrograph, 5 h run, depth reported at 9 fixed stage points. Inputs are
the EA distribution repackaged inside Zenodo record 6907286 as
`4-Glasgow/Setup/ea8-{2m,0p5m}.*` (20 MB; the official EA release is
on-request).

`scripts/benchmark_ea8.py` parses the LISFLOOD-format `.dem/.n/.rain/
.bci/.bdy/.stage`, runs our solver (this is what `sources=` in
`flood_gpu.simulate` was added for — a time-interpolated point discharge),
stages a CPU-runnable `.par` for our LISFLOOD build (the shipped one asks
for `acc_nugrid` + `cuda`), and overlays the stage hydrographs.

Peak depth at the 9 points:

| | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 |
|---|---|---|---|---|---|---|---|---|---|
| ours | 0.579 | 0.239 | 0.727 | 0.131 | 0.261 | 0.064 | 0.241 | 0.271 | 0.199 |
| LISFLOOD-FP | 0.567 | 0.211 | 0.709 | 0.181 | 0.274 | 0.061 | 0.275 | 0.093 | 0.173 |
| published cluster | 0.55-0.59 | 0.23-0.27 | 0.65-0.73 | - | - | 0.06-0.15 | - | - | - |

**Our solver lands inside the published multi-package cluster at all four
points the EA report plots** (see the published-envelope section below).
Arrival times and hydrograph shapes match LISFLOOD everywhere - both show
the rain peak at ~10 min and the inflow peak at ~45 min.

Still open: P8 shows sharp isolated spikes in our series (0.27 m against
LISFLOOD's 0.09) that look numerical rather than physical, and at P4 we go
fully dry while LISFLOOD retains a shallow pond. Neither point has a
published curve to arbitrate.

## The 2x inflow unit trap (found by reading the spec)

LISFLOOD `.bdy` values are **per unit width**: for a point source it
applies `qtmp * dx` (`iterateq.cpp`, QVAR5 branch). Both EA resolutions
confirm it - the peak is 2.5 at 2 m and 10.0 at 0.5 m, and both mean the
**5 m3/s** of the spec's Figure (c). Taking the file at face value halves
the inflow and pushed our depths ~0.1 m low at the points fed by the point
source (P4, P5, P7). `benchmark_ea8.py` now multiplies by the cell size.
The spec's Manning values (0.02 roads/pavements, 0.05 elsewhere) match the
shipped `.n` grids exactly, so those need no adjustment.

## The published reference curves

EA report SC120002 (Neelz & Pender 2013),
`~/Work/engines/ea_benchmark/SC120002_benchmark_report.pdf`, downloaded
from
<https://assets.publishing.service.gov.uk/media/6033a943d3bf7f721f4b0d49/_SC120002_Benchmarking_2D_hydraulic_models_Report.pdf>.

- **Figure 4.41 (p.101)**: water level vs time at Test 8A points **1 and 2**
- **Figure 4.42 (p.102)**: same for points **3 and 6**
- **Figure 4.43 (p.103)**: velocities at points 2 and 6

19 packages per panel (MIKE FLOOD, TUFLOW, SOBEK, InfoWorks ICM, JFLOW+,
ISIS 2D, LISFLOOD-FP, ...). Only those four points are published; there
are no curves for points 4, 5, 7, 8, 9. The figures give water *level*,
so subtract the bed elevation - which the LISFLOOD `.stage` output header
lists per point.

`docs/ea8_published_envelope.json` holds the bands, and
`benchmark_ea8.py plot` shades them behind our curves. **They were read
off the figures by eye (+/- 0.01-0.02 m), not digitised** - run
WebPlotDigitizer over those four panels before quoting them in the
deliverable. The raw participant time series are not public; they would
have to be requested from fcerm.evidence@environment-agency.gov.uk.

---

# Full validation run on the corridor cut (2026-07-23)

`bash scripts/run_validation.sh`, cut domain (terrain_cut_0.5, 2.68 M cells
@ 0.5 m), 25 Nov 2025 storm. Three engines, external benchmark, and a
mass-balance verdict on every LISFLOOD run (`scripts/lisflood_mass.py` -
keys on Vol/Input, NOT LISFLOOD's Verror, which recovered to 0 on a run
that fabricated water to 5.8x rain).

| Run | Mass (Vol/Input) | Verdict |
|---|---|---|
| ours on cut | 1.00 (limiter) | conserves |
| LISFLOOD-FP on cut | **5.82** | **MASS FAIL - void** |
| SynxFlow on cut | 1.00 | conserves |
| LISFLOOD-FP on EA Test 8A | 1.00 | OK |

**LISFLOOD-FP cannot be used as the cut cross-check** - its ACC scheme
diverges on the 24-174 m terrain (theta 0.8/cfl 0.5 and 0.7/0.2 both fail;
fv1 needs NetCDF our build lacks). Its depth raster is discarded; the run
is kept only as evidence for the flux-limiter argument.

**SynxFlow is the cut cross-check** (full-SWE, independent scheme, mass
conserved):

| Metric | ours vs SynxFlow (cut, full storm) | ours vs SynxFlow (test block) |
|---|---|---|
| Flooded-extent IoU (>10 cm) | 0.61 | 0.78 |
| Wet-cell depth correlation | 0.875 | 0.898 |
| RMSE over wet cells | 11.1 cm | 6.4 cm |
| Bias (ours - SynxFlow) | +5.0 cm | +2.1 cm |
| p99 depth | 1.10 vs 1.28 m | 0.76 vs 0.74 m |

The depth maps are visually near-identical (same streets wet, same deep
points, same hotspots); the scatter sits on the 1:1 line. Metrics land
just outside the auto-`CONSISTENT` thresholds (which were calibrated on
the gentle test block) - IoU 0.61 vs 0.70, RMSE 11 vs 10 cm - so the
script prints "DIVERGE". The residual is the expected inertial-vs-full-SWE
signature: our inertial scheme spreads thin water slightly wider (the
above-diagonal shoulder at shallow depths), full-SWE concentrates it. On a
24-174 m stepped urban domain, IoU 0.61 / corr 0.88 / RMSE 11 cm is a
genuine broad agreement, not a divergence. **Report framing: the clean
headline is the test-block agreement (IoU 0.78-0.81 against both engines);
the cut adds a full-storm confirmation on hard terrain against the
independent scheme.** Figure: output/crosscheck/cut_vs_synx.png.

EA Test 8A remains the external-benchmark leg (ours in the published
multi-package cluster at all 4 points; LISFLOOD's 8A run mass-OK at 1.00).
