#!/usr/bin/env python3
"""Dry-bed dam break against the Ritter analytic solution.

The one classical test that separates the two schemes cleanly. A dam break on
a dry frictionless bed has an exact solution (Ritter 1892), and it is the case
the inertial approximation cannot represent: dropping the advective term
removes exactly the momentum flux that sets the front celerity.

  python scripts/test_dambreak.py
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from flood_gpu import simulate

G = 9.81


def ritter(x, t, h0):
    """Exact dry-bed dam-break profile, dam at x = 0, water for x < 0."""
    c0 = np.sqrt(G * h0)
    xi = x / t
    h = np.where(xi <= -c0, h0,
        np.where(xi >= 2 * c0, 0.0, (2 * c0 - xi) ** 2 / (9 * G)))
    return h


def run(scheme, h0, res, ncols, t_end, device):
    nrows = 9
    dem = np.zeros((nrows, ncols), dtype=np.float32)
    dem[0, :] = dem[-1, :] = 50.0                      # side walls -> 1-D
    dem[:, 0] = dem[:, -1] = 50.0                      # end walls: a closed tank.
    # Without these the reservoir drains out of the open boundary and the
    # far field falls below h0, which scores as solver error when it is not.
    init = np.zeros((nrows, ncols), dtype=np.float32)
    init[1:-1, 1:ncols // 2] = h0                      # dam at the midpoint
    out = tempfile.mkdtemp(prefix="dambreak_")
    try:
        simulate(dem, res, [], t_end, out, manning=0.0, init_depth=init,
                 scheme=scheme, save_every=t_end, save_frames=False,
                 progress=False, device=device)
        return np.load(f"{out}/final_depth.npy")[nrows // 2].astype(float)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h0", type=float, default=10.0)
    ap.add_argument("--res", type=float, default=0.5,
                    help="resolution whose profile is stored for plotting")
    ap.add_argument("--length", type=float, default=800.0)
    ap.add_argument("--sweep", default="2.0,1.0,0.5,0.25",
                    help="resolutions to run, comma separated")
    ap.add_argument("--t-end", type=float, default=15.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="output/verification/dambreak.json")
    a = ap.parse_args()

    profiles_res = a.res
    sweep = [float(r) for r in a.sweep.split(",")] if a.sweep else [a.res]
    result = {"h0": a.h0, "t_end": a.t_end, "profile_res": profiles_res,
              "convergence": [], "schemes": {}}

    for res in sweep:
        ncols = int(round(a.length / res)) + 1
        x = (np.arange(ncols) - ncols // 2 + 0.5) * res
        exact = ritter(x, a.t_end, a.h0)
        # Score only where the analytic solution is meaningfully wet; the dry
        # tail is where a scheme either has a front or does not, scored by the
        # front position instead.
        wet = exact > 0.01
        front_exact = float(x[wet].max())
        row = {"res_m": res, "cells": ncols}
        for scheme in ("inertial", "hllc"):
            h = run(scheme, a.h0, res, ncols, a.t_end, a.device)
            rmse = float(np.sqrt(((h[wet] - exact[wet]) ** 2).mean()))
            got = x[h > 0.01]
            front = float(got.max()) if got.size else float("nan")
            row[scheme] = {"rmse_m": rmse, "rmse_pct_h0": 100 * rmse / a.h0,
                           "front_m": front,
                           "front_err_pct": 100 * (front - front_exact) / front_exact}
            print(f"res {res:5.2f}  {scheme:9s} RMSE {rmse*100:6.1f} cm "
                  f"({100*rmse/a.h0:4.1f}% of h0)   front {front:6.1f} m vs "
                  f"{front_exact:6.1f} m  ({100*(front-front_exact)/front_exact:+.1f}%)")
            if res == profiles_res:
                result["schemes"][scheme] = {"h": h.tolist()}
        row["front_exact_m"] = front_exact
        result["convergence"].append(row)
        if res == profiles_res:
            result["x"] = x.tolist()
            result["exact"] = exact.tolist()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(result, open(a.out, "w"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
