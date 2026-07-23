#!/usr/bin/env bash
# One-shot progress report for a detached run_validation.sh.
#   bash scripts/validation_status.sh          # print once
#   watch -n 30 bash scripts/validation_status.sh
cd "$(dirname "$0")/.."

echo "=============================================================="
if pgrep -f "run_validation.sh" > /dev/null; then
  echo "STATUS: RUNNING   (started $(ps -o lstart= -p "$(pgrep -f run_validation.sh | head -1)" 2>/dev/null))"
else
  echo "STATUS: NOT RUNNING (finished, or stopped)"
fi
echo "now:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "--------------------------------------------------------------"

echo "current stage (last '===' line of the log):"
grep '^=== ' output/validation.log 2>/dev/null | tail -1 | sed 's/^/  /'
echo "last line:"
tail -1 output/validation.log 2>/dev/null | cut -c1-110 | sed 's/^/  /'
echo "--------------------------------------------------------------"

echo "which engine is on the CPU/GPU right now:"
pgrep -af "lisflood|synxflow_run|crosscheck.py|benchmark_ea8" \
  | grep -v pgrep | cut -c1-100 | sed 's/^/  /' || echo "  (none)"
echo "--------------------------------------------------------------"

if [ -f output/export_cut/results/res.mass ]; then
  echo "LISFLOOD on the cut - last mass line (Time Tstep .. Vol .. Verror Rain):"
  tail -1 output/export_cut/results/res.mass | sed 's/^/  /'
  echo "  (Vol should track Rain; a growing |Verror| means it is diverging)"
fi

echo "--------------------------------------------------------------"
echo "outputs:"
for f in output/crosscheck/ours_cut/max_depth.npy \
         output/export_cut/results/res.max \
         output/crosscheck/synx_cut/max_depth.asc \
         output/crosscheck/cut_vs_lfp.png \
         output/crosscheck/cut_vs_synx.png \
         output/crosscheck/lisflood_cut_mass.json \
         output/ea8/ea8_stages.png; do
  [ -e "$f" ] && echo "  [x] $f" || echo "  [ ] $f"
done
echo "=============================================================="
