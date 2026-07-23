#!/usr/bin/env python3
"""Export terrain + storm to ESRI ASCII / LISFLOOD-FP inputs (Phase B2/D3).

Produces <out>/dem.asc, n.asc, rain.rain and a LISFLOOD-FP 8.x .par file so
the same setup can be re-run in LISFLOOD-FP (our scheme's reference
implementation) or converted for SERGHEI (which also reads ESRI ASCII).

Usage:
  python scripts/export_ascii.py --terrain output/terrain_1.0 \
      --storm storms/v1_nov2025.json --out output/export_lfp
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from las_common import load_transform

NODATA = -9999.0


def write_asc(path, arr, t, nodata=NODATA):
    h, w = arr.shape
    a = np.where(np.isfinite(arr), arr, nodata)
    header = (f"ncols {w}\nnrows {h}\nxllcorner {t['minx']:.2f}\n"
              f"yllcorner {t['miny']:.2f}\ncellsize {t['res']}\n"
              f"NODATA_value {nodata:.0f}\n")
    with open(path, "w") as f:
        f.write(header)
        np.savetxt(f, a, fmt="%.3f")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", required=True)
    ap.add_argument("--storm", required=True)
    ap.add_argument("--out", default="output/export_lfp")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t = load_transform(os.path.join(args.terrain, "dem_transform.json"))
    dem = np.load(os.path.join(args.terrain, "dem.npy"))
    n = np.load(os.path.join(args.terrain, "manning.npy"))
    masks = np.load(os.path.join(args.terrain, "masks.npz"))
    # Sea / open-water cells must go out as NODATA, not as terrain. Our own
    # solver treats them as an outflow sink, but an exported grid has no way
    # to say that - so they arrive at the external engine as ordinary ground
    # tens of metres below the street beside them. On the corridor cut that
    # is a 38 m cliff at 0.5 m resolution, and LISFLOOD-FP's ACC scheme
    # fabricated water on it (Vol 2.2x rain, Verror growing; the log is kept
    # at docs/evidence/lisflood_cut_unstable_res.mass) even with theta 0.8 +
    # cfl 0.5. As NODATA both engines see a wall instead of a cliff, and
    # neither is asked to route water into a basin that is really the sea.
    keep = masks["valid"] & ~masks["water"]
    n_water = int((masks["valid"] & masks["water"]).sum())
    print(f"export domain: {int(keep.sum())} cells "
          f"({n_water} sea/water cells written as NODATA)")
    dem = np.where(keep, dem, np.nan)

    write_asc(os.path.join(args.out, "dem.asc"), dem, t)
    write_asc(os.path.join(args.out, "n.asc"),
              np.where(keep, n, np.nan), t)

    with open(args.storm) as f:
        storm = json.load(f)
    # LISFLOOD-FP rain file: the parser SKIPS the first line, then expects
    # "N units" and N lines of "value(mm/h) time(s)" with increasing times
    # (verified in input.cpp LoadRain/LoadTimeSeries, v8.2)
    steps = storm["steps"]
    with open(os.path.join(args.out, "storm.rain"), "w") as f:
        f.write(f"# {storm['name']} rainfall (value mm/h, time s)\n")
        f.write(f"{len(steps) * 2 + 1}    seconds\n")
        for t0, t1, i in steps:
            f.write(f"{i:.2f} {t0:.0f}\n{i:.2f} {t1 - 1:.0f}\n")
        # LISFLOOD holds the LAST value forever - terminate with zero rain
        f.write(f"0.00 {steps[-1][1]:.0f}\n")
    # NOTE: no bcifile -> closed borders. FREE boundaries on cropped urban
    # edges are unstable in v8.2 (slope extrapolation CREATES volume -
    # observed Verror ~ -7000 m3); for cross-checks run our solver with
    # --closed so both engines pond at the borders identically.
    # no 'routing': pure ACC dynamics = closest match to our inertial solver
    par = f"""# LISFLOOD-FP 8.x par - Beirut {storm['name']} (generated)
DEMfile      dem.asc
resroot      res
dirroot      results
saveint      300.0
sim_time     {storm['duration']:.0f}
initial_tstep 1.0
acceleration
cfl          0.5
theta        0.8
manningfile  n.asc
rainfall     storm.rain
depthoff
elevoff
"""
    # cfl+theta: without them LISFLOOD's ACC oscillates on steep stepped
    # urban DEMs and fabricates volume (observed Verror ~ -1e6 m3 on the
    # 0.5 m test block); theta<1 is the standard de Almeida (2012) damping
    with open(os.path.join(args.out, f"{storm['name']}.par"), "w") as f:
        f.write(par)
    print(f"wrote {args.out}/{storm['name']}.par")
    print("run:  lisflood -v " + os.path.join(args.out, f"{storm['name']}.par"))
    print("SERGHEI: use dem.asc/n.asc with a rain .input built from storm.rain "
          "(see docs/research_flood_engines.md)")


if __name__ == "__main__":
    main()
