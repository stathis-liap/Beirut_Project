# Research: 2D pluvial flood engines for Linux + RTX 3050 6GB (2026-07-12)

Feeds PLAN.md Phase B/D2. The researcher downloaded and inspected actual
source (LISFLOOD-FP 8.1/8.2 from Zenodo, SERGHEI wiki, TRITON docs) —
"verified in source" claims below are from that inspection.

## Decision

- **Primary compute: torch-CUDA port of our own validated Bates-2010
  kernel** (torch 2.12+cu130 already in the venv; full control over roof
  routing, drains, hazard accumulators; fastest path).
- **External check #1: SERGHEI 2.x** — BSD-3 (safe commercially), very
  active (commits July 2026), Kokkos → CUDA 13 supported rather than
  fought, ESRI-ASCII inputs, spatially varying rain + Horton infiltration
  class maps + roughness rasters, **full-SWE augmented-Roe solver ⇒
  methodologically independent of our inertial scheme** — agreement means
  something. Build: CMake + Kokkos + MPI + Parallel-NetCDF, ~half a day.
  https://gitlab.com/serghei-model/serghei
- **External check #2: LISFLOOD-FP 8.2 ACC, CPU build** — the *reference
  implementation of our exact scheme* (sharpest check of our kernels) and
  the strongest EA-benchmark pedigree in the field. CPU build trivial
  (cmake + libnetcdf); GPU build needs small patches on CUDA 13 (arch 86 +
  C++17) or a side-by-side CUDA 12 toolkit — unnecessary, we have our own
  GPU. **License ambiguous** (Zenodo says GPL-2.0, site says "GPLv3
  non-commercial", zip has no LICENSE file — verified): fine for research;
  get written clarification before any commercial deliverable relies on it.
  Zenodo: https://zenodo.org/records/13121102
- **Optional check #3: Itzï 26.6** — revived (PyPI 2026-06-30), GPL-2+,
  same de Almeida family but independent code, **published EA Test 8A
  validation (GMD 2017)**, Green-Ampt rasters, and the only engine here
  with bidirectional SWMM drainage coupling (via pyswmm) if a real
  drainage-network model is ever wanted. CPU-only → subdomains/2–5 M cells.
  Needs GRASS ≥ 8.4.

## Ruled out

- **pypims/HiPIMS**: stale since 2022, will fight CUDA 12/13 — dead end.
- **TorchSWE**: archived read-only 2026, no rain — dead.
- **SynxFlow 1.0.2**: HiPIMS successor, pip wheels(!), GPU, rain +
  infiltration — viable alternate but Python 3.9–3.11 only (separate venv)
  and single-maintainer; keep as backup if SERGHEI's build annoys.
- **TRITON 2.0**: active, BSD-3, GPU, but no infiltration (expects
  rainfall-excess) — extra preprocessing for no benefit over SERGHEI.
- **SFINCS** (Deltares): excellent and active but CPU-only Fortran;
  subgrid approach interesting for future big domains, not needed here.
- **ANUGA / Landlab**: CPU, fine as unit-level references only (Landlab
  OverlandFlow = cheapest independent de Almeida implementation for
  test-block cross-checks; MIT, pip).
- **HEC-RAS 2D Linux**: headless possible via Docker but closed, CPU-only,
  authoring needs Windows GUI. Skip.

## EA benchmark (Néelz & Pender 2013) Test 8A data

Official datasets are on-request from the EA
(fcerm.evidence@environment-agency.gov.uk). Practical route: **Zenodo
record 6907286** (LISFLOOD-FP 8.1 paper data) contains the Glasgow
(Test-8A domain, ~0.4 km² @ 2 m, ~97 k cells) inputs + results in
LISFLOOD-FP format — small enough to run in seconds; validates
correctness, not scale. Published Test-8A depth/velocity curves to compare
against: Itzï GMD 2017 paper; HEC RD-51 report reproduces the full suite.

## VRAM reality check (matches our measurements)

5–25 M cells ≈ 1–3 GB of solver state — comfortable on 6 GB in fp32
(LISFLOOD-FP authors hit the wall near ~100 M cells on an 8 GB card).
Full survey @ 0.5 m = 24.1 M cells ⇒ fits; @ 0.25 m does not — don't go
there.
