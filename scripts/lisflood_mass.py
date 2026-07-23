#!/usr/bin/env python3
"""Verdict on a LISFLOOD-FP run's own mass balance (res.mass).

LISFLOOD's ACC solver fabricates volume on steep stepped urban terrain
(docs/crosscheck_lisflood.md). When it does, its depth raster is not worth
comparing against - so every cross-check should state explicitly whether
the external engine conserved mass, rather than quietly reporting an IoU
against a diverged field.

Columns (v8.x): Time Tstep MinTstep NumTsteps Area Vol Qin Hds Qout
Qerror Verror Rain-(Inf+Evap)

Usage:
  python scripts/lisflood_mass.py output/export_cut/results/res.mass \
      [--json output/crosscheck/lisflood_cut_mass.json]
"""

import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mass")
    ap.add_argument("--json")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="max |Verror| / rain volume before FAIL")
    ap.add_argument("--vol-tol", type=float, default=0.25,
                    help="max |Vol/Rain - 1| before FAIL (closed domain, no "
                         "infiltration, so retained volume must equal rain)")
    args = ap.parse_args()

    rows = []
    with open(args.mass) as f:
        f.readline()                       # header
        for line in f:
            p = line.split()
            if len(p) >= 12:
                rows.append([float(v) for v in p])
    if not rows:
        sys.exit(f"{args.mass}: no data rows")

    t, vol, verror, rain = (rows[-1][0], rows[-1][5], rows[-1][10],
                            rows[-1][11])
    # Total input = rain + any point-source inflow. Qin (col 7) is an
    # instantaneous discharge (m3/s), so integrate it (trapezoidal) over the
    # time column; without this the checker false-fails cases with a point
    # source, e.g. EA Test 8A's 5 m3/s culvert (Vol/Rain reads ~1.6 purely
    # because the inflow volume is missing from the denominator).
    qin_vol = 0.0
    for prev, cur in zip(rows, rows[1:]):
        qin_vol += 0.5 * (prev[6] + cur[6]) * (cur[0] - prev[0])
    inp = rain + qin_vol
    ratio = vol / inp if inp > 0 else float("inf")
    rel_err = abs(verror) / inp if inp > 0 else float("inf")
    # Vol/Rain is the load-bearing signal, NOT Verror. On the corridor cut
    # LISFLOOD's ACC scheme pumped the domain to 5.8x the rain input while
    # its own Verror column recovered to 0 - the fabrication is in the
    # (discretely self-consistent) unphysical fluxes, so Verror does not see
    # it. With closed borders and no infiltration the retained volume must
    # equal the rain, so |Vol/Rain - 1| is what catches divergence.
    ok = rel_err <= args.tol and abs(ratio - 1.0) <= args.vol_tol

    out = {"file": args.mass, "t_end_s": t, "volume_m3": vol,
           "rain_m3": rain, "inflow_m3": round(qin_vol, 1),
           "input_m3": round(inp, 1), "verror_m3": verror,
           "vol_over_input": round(ratio, 3),
           "rel_mass_error": round(rel_err, 4),
           "verdict": "MASS OK" if ok else "MASS FAIL (engine diverged)"}
    print(f"  t={t:.0f}s  Vol={vol:.0f} m3  Input={inp:.0f} m3 "
          f"(rain {rain:.0f} + inflow {qin_vol:.0f})  "
          f"Vol/Input={ratio:.2f}  Verror={verror:.0f} m3")
    print(f"  -> {out['verdict']}")
    if not ok:
        print("     the depth raster from this run is NOT comparable; treat "
              "any cross-check metric against it as void.")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  wrote {args.json}")


if __name__ == "__main__":
    main()
