#!/usr/bin/env bash
# Full validation night: three engines on the cut corridor domain + the EA
# Test 8A benchmark. Everything below is idempotent - rerunning skips work
# whose output already exists (delete the output dir to force a redo).
#
#   bash scripts/run_validation.sh 2>&1 | tee output/validation.log
#
# Budget: ~1-2 h LISFLOOD (CPU) + ~1-2 h SynxFlow (GPU) + ~3 min Test 8A.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/stathisliap/Work/.venv/bin/python
PY_SYNX=/home/stathisliap/Work/.venv_synxflow/bin/python
LISFLOOD=$HOME/Work/engines/LISFLOOD-FP/build/lisflood
EXPORT=output/export_cut
STORM=storms/v1_nov2025.json

step() { printf '\n=== %s  [%s] ===\n' "$1" "$(date +%H:%M:%S)"; }

# Stray stability probes must not be present while the engines run: a
# probe*.par with a short sim_time can hijack SynxFlow's run length.
rm -f $EXPORT/probe*.par $EXPORT/probe*.log
rm -rf $EXPORT/results_probe output/crosscheck/synx_probe

# --- 1. our solver on the cut (plain mode, closed borders) ----------------
if [ ! -f output/crosscheck/ours_cut/max_depth.npy ]; then
  step "ours: torch solver on the cut"
  $PY scripts/crosscheck.py run-ours --export $EXPORT --storm $STORM \
      --out output/crosscheck/ours_cut --closed
else
  echo "ours_cut already done, skipping"
fi

# --- 2. LISFLOOD-FP 8.2 on the same inputs --------------------------------
if [ ! -f $EXPORT/results/res.max ]; then
  step "LISFLOOD-FP on the cut (CPU, the slow one)"
  ( cd $EXPORT && $LISFLOOD -v v1_nov2025.par ) \
    || echo "WARNING: LISFLOOD exited non-zero - continuing with the other engines"
else
  echo "LISFLOOD cut result already present, skipping"
fi

# --- 3. SynxFlow (full-SWE GPU, independent scheme) -----------------------
if [ ! -f output/crosscheck/synx_cut/max_depth.asc ]; then
  step "SynxFlow on the cut (GPU)"
  $PY_SYNX scripts/synxflow_run.py --export $EXPORT \
      --out output/crosscheck/synx_cut
else
  echo "synx_cut already done, skipping"
fi

# --- 3b. did LISFLOOD conserve mass? -------------------------------------
# Its ACC solver fabricates volume on steep stepped terrain; on the cut it
# reached Vol/Rain 2.2 with theta 0.8 + cfl 0.5, and tightening to 0.7/0.2
# barely helped (docs/evidence/lisflood_cut_unstable_res.mass). A diverged
# raster must not be reported as a cross-check, so state the verdict here.
step "LISFLOOD mass balance check"
$PY scripts/lisflood_mass.py $EXPORT/results/res.mass \
    --json output/crosscheck/lisflood_cut_mass.json || true

# --- 4. pairwise comparisons ---------------------------------------------
step "compare: ours vs LISFLOOD-FP"
$PY scripts/crosscheck.py compare \
    --ours output/crosscheck/ours_cut/max_depth.npy \
    --theirs $EXPORT/results/res.max \
    --export $EXPORT --out output/crosscheck/cut_vs_lfp.png || true

step "compare: ours vs SynxFlow"
$PY scripts/crosscheck.py compare \
    --ours output/crosscheck/ours_cut/max_depth.npy \
    --theirs output/crosscheck/synx_cut/max_depth.asc \
    --export $EXPORT --out output/crosscheck/cut_vs_synx.png || true

# --- 5. EA 2D benchmark Test 8A (external, published case) ----------------
step "EA Test 8A: our solver"
$PY scripts/benchmark_ea8.py run --res 2m --out output/ea8 --save-every 20

step "EA Test 8A: LISFLOOD-FP"
$PY scripts/benchmark_ea8.py prep-lisflood --out output/ea8
( cd output/ea8/lisflood && $LISFLOOD -v ea8-2m.par ) || true

step "EA Test 8A: LISFLOOD mass balance check"
$PY scripts/lisflood_mass.py output/ea8/lisflood/results/ea8-2m.mass \
    --json output/ea8/lisflood_mass.json || true

step "EA Test 8A: comparison figure"
$PY scripts/benchmark_ea8.py plot --out output/ea8

rm -f $EXPORT/probe*.par $EXPORT/probe*.log
rm -rf $EXPORT/results_probe output/crosscheck/synx_probe

step "ALL DONE"
echo "figures:"
echo "  output/crosscheck/cut_vs_lfp.png    ours vs LISFLOOD-FP  (cut domain)"
echo "  output/crosscheck/cut_vs_synx.png   ours vs SynxFlow     (cut domain)"
echo "  output/ea8/ea8_stages.png           EA Test 8A stage hydrographs"
echo "  output/ea8/ea8_summary.json         EA Test 8A peak depths"
echo "mass-balance verdicts (read these before trusting a comparison):"
echo "  output/crosscheck/lisflood_cut_mass.json"
echo "  output/ea8/lisflood_mass.json"
