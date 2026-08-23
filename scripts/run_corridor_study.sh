#!/usr/bin/env bash
# Before/after green-corridor study on the 0.5 m corridor domain.
# Idempotent: existing runs are skipped. Detach-safe.
set -uo pipefail
cd /home/stathisliap/Work/Beirut_Project
PY=/home/stathisliap/Work/.venv/bin/python
# Overridable so a scratch/review terrain can be studied side by side with the
# published one instead of overwriting it; unset, these are the published paths
# and the script behaves exactly as before.
BEFORE=${CORRIDOR_BEFORE:-output/terrain_cut_0.5}
AFTER=${CORRIDOR_AFTER:-output/terrain_cut_corridor}
OUT=${CORRIDOR_OUT:-output/corridor_runs}
# extra flags passed straight to flood_gpu.py, e.g. CORRIDOR_FLAGS="--scheme hllc"
CORRIDOR_FLAGS=${CORRIDOR_FLAGS:-}
# limit to the before/after pairs (skip the drain scenarios) for expensive schemes
CORRIDOR_PAIRS_ONLY=${CORRIDOR_PAIRS_ONLY:-0}
mkdir -p $OUT
STORMS="t2 v1_nov2025 t50"

run(){ # terrain storm outdir drains
  local terr=$1 storm=$2 od=$3 drains=$4
  if [ -f $od/max_depth.npy ]; then echo "skip $od (done)"; return; fi
  echo "=== $od  [$(date +%H:%M:%S)] ==="
  if [ -n "$drains" ]; then
    $PY scripts/flood_gpu.py --terrain $terr --storm storms/$storm.json \
        --out $od --drains $drains --save-every 300 $CORRIDOR_FLAGS || echo "RUN FAILED: $od"
  else
    $PY scripts/flood_gpu.py --terrain $terr --storm storms/$storm.json \
        --out $od --save-every 300 $CORRIDOR_FLAGS || echo "RUN FAILED: $od"
  fi
}

# 1. BEFORE (existing surface, drains blocked)
for s in $STORMS; do run $BEFORE $s $OUT/before_$s ""; done
# 2. AFTER (green corridor, drains blocked)
for s in $STORMS; do run $AFTER $s $OUT/after_$s ""; done
if [ "$CORRIDOR_PAIRS_ONLY" = "1" ]; then
  echo "PAIRS ONLY: skipping the drain scenarios [$(date +%H:%M:%S)]"
  exit 0
fi
# 3. optimize drains from the observed-storm baseline ponding
if [ ! -f $AFTER/drains_opt.npz ]; then
  echo "=== optimize drains [$(date +%H:%M:%S)] ==="
  $PY scripts/optimize_drains.py --terrain $BEFORE \
      --ponding $OUT/before_v1_nov2025/final_depth.npy \
      --out $AFTER/drains_opt.npz || echo "OPTIMIZER FAILED"
fi
# 4. what-if: drains alone, and corridor + optimized drains
run $BEFORE v1_nov2025 $OUT/drainsonly_v1_nov2025 $AFTER/drains_opt.npz
for s in v1_nov2025 t50; do run $AFTER $s $OUT/afterdrains_$s $AFTER/drains_opt.npz; done

echo "ALL CORRIDOR RUNS DONE [$(date +%H:%M:%S)]"
