#!/usr/bin/env bash
# Attribution study: how much of the corridor's reported benefit is real, and
# how much is an artefact of modelling choices?
#
# The headline claim is a DIFFERENCE - flooded street area before minus after.
# Absolute depths are known to be scheme-sensitive (HLLC wets 21% less area
# than the inertial scheme on this same grid), but both scenarios carry the
# same bias, so the difference may be far more robust than either term. That
# is the question this answers, and it is the one a reviewer will ask first.
#
# Every variant therefore runs a MATCHED before/after pair and is scored on the
# reduction, never on an absolute. Variants are one-factor-at-a-time from the
# same baseline so each effect is attributable.
#
#   base    inertial | constant infiltration | fixed-capacity gullies
#   hllc    + shock-capturing Godunov scheme            (numerics)
#   ga      + Green-Ampt infiltration                   (soil physics)
#   hd      + head-discharge gullies                    (drainage physics)
#   ero     + probabilistic erosion, seeded ensemble    (surface degradation)
#
# Idempotent: any run whose max_depth.npy exists is skipped, so this can be
# re-entered after an interruption. Detach-safe.
#
#   bash scripts/run_sensitivity.sh 2>&1 | tee output/sensitivity.log

set -uo pipefail
cd /home/stathisliap/Work/Beirut_Project
PY=/home/stathisliap/Work/.venv/bin/python

BEFORE=${SENS_BEFORE:-output/terrain_cut_0.5_v3}
AFTER=${SENS_AFTER:-output/terrain_cut_corridor_v3}
OUT=${SENS_OUT:-output/sensitivity}
BASE_RUNS=${SENS_BASE_RUNS:-output/corridor_runs_v3}   # the base pair lives here
mkdir -p $OUT

step() { printf '\n=== %s  [%s] ===\n' "$1" "$(date +%H:%M:%S)"; }

run() { # terrain storm outdir  [extra flags...]
  local terr=$1 storm=$2 od=$3; shift 3
  if [ -f "$od/max_depth.npy" ]; then echo "skip $od (done)"; return; fi
  step "$od"
  $PY scripts/flood_gpu.py --terrain "$terr" --storm storms/$storm.json \
      --out "$od" --save-every 300 "$@" || echo "RUN FAILED: $od"
}

# --- wait for the baseline study, which supplies the base pair -------------
while pgrep -f "run_corridor_study.sh" > /dev/null; do
  echo "waiting for the baseline study to finish... [$(date +%H:%M:%S)]"
  sleep 120
done

# --- 1. numerics: does the REDUCTION survive shock capturing? --------------
# The expensive one (~11x), so a single storm: the observed 25 Nov 2025 event.
run $BEFORE v1_nov2025 $OUT/hllc_before --scheme hllc
run $AFTER  v1_nov2025 $OUT/hllc_after  --scheme hllc

# --- 2. soil physics: does the corridor still saturate in a T50? -----------
# Run on BOTH the observed storm and T50. T50 is the one that matters: the
# published claim is that the soils saturate and the corridor shifts from
# absorbing to detaining, which a constant infiltration rate cannot express.
for s in v1_nov2025 t50; do
  run $BEFORE $s $OUT/ga_before_$s --infil-mode green_ampt
  run $AFTER  $s $OUT/ga_after_$s  --infil-mode green_ampt
done

# --- 3. drainage physics: gullies that respond to head ---------------------
# Only meaningful with inlets present, so this is the after+drains scenario
# against its fixed-capacity twin in the baseline study.
if [ -f $AFTER/drains_opt.npz ]; then
  run $AFTER v1_nov2025 $OUT/hd_afterdrains --drains $AFTER/drains_opt.npz \
      --drain-mode head_discharge
  run $AFTER v1_nov2025 $OUT/cap_afterdrains --drains $AFTER/drains_opt.npz
else
  echo "WARNING: $AFTER/drains_opt.npz missing - skipping the drainage variant"
fi

# --- 4. erosion: seeded ensemble on the severe storm -----------------------
# Stochastic, so a single run says nothing; three seeds give a spread. Run on
# the AFTER case only - the corridor's bioswales, gardens and terraces are the
# erodible surfaces, and the question is whether they degrade under a T50.
for seed in 1 2 3; do
  run $AFTER t50 $OUT/ero_after_t50_s$seed --erosion --erosion-seed $seed
done

echo
echo "ALL SENSITIVITY RUNS DONE [$(date +%H:%M:%S)]"
echo "score them with: $PY scripts/analyze_sensitivity.py"
