#!/usr/bin/env python3
"""UK EA benchmark Test 4: speed of flood propagation over an extended floodplain.

Test 4 is the one benchmark besides 8A we can run without the EA's data files,
because its geometry is fully specified in the report text rather than shipped
as a DEM: Appendix A.4 says outright "No DEM is provided as the ground
elevation is uniformly 0". Everything else - domain, roughness, grid, inflow,
end time - is given numerically. So this is built from the specification, not
reconstructed from a figure.

  domain      1000 m (x) x 2000 m (y), horizontal, elevation 0
  inflow      20 m3/s peak, ~5 h time base, along a 20 m line in the middle
              of the western side (Figure A.8's trapezoid)
  Manning n   0.05 uniform
  grid        5 m (~80,000 cells)
  boundaries  all closed; dry bed initially
  end time    t = 5 h

What it tests: the celerity of the advancing front and the transient depths
and velocities at its leading edge - i.e. wetting and drying on a flat plain,
which is a different regime from 8A's urban rain-on-grid and from the steep
stepped terrain the Beirut corridor exercises.

IMPORTANT on the comparison: the EA distributes per-model result series with
the test data, which we do not have. The published values below were read off
Figure 4.13 of the benchmark report (cross-section of depth at t = 1 h) by
eye. They are good to maybe +/-0.02 m and are NOT a substitute for the real
data - treat "in cluster" here as weaker evidence than the 8A comparison,
which uses a proper digitised envelope.

Usage:
  python scripts/benchmark_ea4.py --out output/ea4 [--scheme hllc]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from flood_gpu import simulate

# Figure A.8: rises to the 20 m3/s peak by ~60 min, holds to ~240 min, back
# to zero by ~300 min.
HYDROGRAPH = [(0.0, 0.0), (3600.0, 20.0), (14400.0, 20.0), (18000.0, 0.0)]

# Figure 4.13, depth (m) along the centre line at t = 1 h, as the spread of
# the participating models. Read from the figure - see the module docstring.
PUBLISHED_XS_1H = {
    25: (0.32, 0.46), 50: (0.29, 0.36), 100: (0.25, 0.29),
    150: (0.21, 0.25), 200: (0.18, 0.21), 250: (0.15, 0.18),
    300: (0.11, 0.14), 350: (0.06, 0.10), 400: (0.00, 0.04),
}


def build(res=5.0):
    """Flat plain with a wall around it (Test 4's boundaries are all closed).

    simulate() treats the outermost ring as an open boundary, so the wall is
    one cell of high ground OUTSIDE the 1000x2000 m domain: the modelled area
    is exactly as specified and nothing can leave it."""
    nx = int(1000 / res)
    ny = int(2000 / res)
    dem = np.zeros((ny + 2, nx + 2), dtype=np.float32)
    dem[0, :] = dem[-1, :] = dem[:, 0] = dem[:, -1] = 10.0
    return dem, res, nx, ny


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/ea4")
    ap.add_argument("--res", type=float, default=5.0)
    ap.add_argument("--sim-time", type=float, default=18000.0)
    ap.add_argument("--scheme", default="inertial", choices=["inertial", "hllc"])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    dem, res, nx, ny = build(args.res)
    h, w = dem.shape
    # inflow: 20 m long line at x=0, centred on y=1000 (the domain mid-height)
    n_in = max(1, int(round(20.0 / res)))
    r_mid = 1 + ny // 2
    rows = [r_mid + i - n_in // 2 for i in range(n_in)]
    sources = [{"row": r, "col": 1,
                "series": [[t, q / n_in] for t, q in HYDROGRAPH]} for r in rows]

    save_every = 600.0
    print(f"EA Test 4: {nx * res:.0f} x {ny * res:.0f} m at {res} m "
          f"({nx * ny} cells), n=0.05, scheme={args.scheme}")
    print(f"  inflow over {n_in} cells ({n_in * res:.0f} m), peak 20 m3/s, "
          f"sim {args.sim_time:.0f}s")
    meta = simulate(dem, res, [], args.sim_time, args.out, manning=0.05,
                    sources=sources, save_every=save_every, device=args.device,
                    save_frames=True, progress=True, scheme=args.scheme)

    # --- cross-section along the centre line at t = 1 h ---------------------
    frames = sorted(f for f in os.listdir(args.out) if f.startswith("depth_"))
    idx_1h = int(round(3600.0 / save_every))
    d1 = np.load(os.path.join(args.out, frames[min(idx_1h, len(frames) - 1)])).astype(np.float32)
    idx_3h = int(round(10800.0 / save_every))
    d3 = np.load(os.path.join(args.out, frames[min(idx_3h, len(frames) - 1)])).astype(np.float32)

    line1 = d1[r_mid, 1:1 + nx]
    line3 = d3[r_mid, 1:1 + nx]
    xs = (np.arange(nx) + 0.5) * res

    print(f"\n  depth along the centre line at t = 1 h")
    print(f"  {'x (m)':>7} {'ours':>8} {'published spread':>18}   verdict")
    n_in_band = n_tot = 0
    rows_out = []
    for x, (lo, hi) in sorted(PUBLISHED_XS_1H.items()):
        i = int(x / res)
        if i >= nx:
            continue
        v = float(line1[i])
        ok = (lo - 0.02) <= v <= (hi + 0.02)     # allow the figure-reading error
        n_tot += 1
        n_in_band += ok
        print(f"  {x:7.0f} {v:8.3f}   {lo:6.2f} - {hi:5.2f}   "
              f"{'in cluster' if ok else 'OUTSIDE'}")
        rows_out.append({"x_m": x, "ours_m": v, "published": [lo, hi], "in_cluster": ok})

    def front(line, thresh):
        wet = np.flatnonzero(line >= thresh)
        return float(xs[wet[-1]]) if wet.size else 0.0

    f1, f3 = front(line1, 0.15), front(line3, 0.15)
    print(f"\n  0.15 m front position: {f1:.0f} m at 1 h, {f3:.0f} m at 3 h "
          f"(report Fig 4.12 shows ~200 m and ~400 m half-circles)")
    print(f"  mass balance: rain {meta['vol_rain_m3']:.0f} m3 "
          f"(inflow), closure {meta['closure_rel']:.2e}")
    print(f"\n  {n_in_band}/{n_tot} cross-section points inside the published spread")

    json.dump({"test": "EA 8-test suite, Test 4", "scheme": args.scheme,
               "res_m": res, "cross_section_1h": rows_out,
               "front_0p15m": {"t_1h": f1, "t_3h": f3},
               "closure_rel": meta["closure_rel"],
               "in_cluster": f"{n_in_band}/{n_tot}",
               "caveat": "published spread read from Figure 4.13 by eye, not "
                         "from the EA result files"},
              open(os.path.join(args.out, "ea4_result.json"), "w"), indent=2)
    print(f"  wrote {args.out}/ea4_result.json")


if __name__ == "__main__":
    main()
