#!/usr/bin/env python3
"""Does resolution finer than 2 m change the answer? — a direct test of the EA's open question.

The Environment Agency benchmark report leaves this explicitly unresolved. Its
executive summary reports "large differences (up to 100%) in velocity
predictions" between packages at 2 m in shallow urban flow, and concludes:

    "it is not currently clear that grid resolutions finer than 2m will
     improve the quality of velocity predictions"

because DTM error and boundary conditions bite at the same order as the grid.
The report recommends refining Test 8 to 0.5 m to settle it (Section 5.2.1),
and nobody has.

We are in an unusually good position to answer it: a real dense city, surveyed
by LiDAR at 0.5 m, with a corridor whose flow regime (steep, supercritical,
stepped) is exactly where resolution should matter most.

Method - and the reason this is a fair test: coarsen the SAME terrain rather
than rebuild it per resolution. Block-mean the DEM and the parameter rasters,
block-OR the masks. That holds the survey, the land cover and the corridor
design fixed so the ONLY thing varying is the computational grid. Rebuilding
the terrain at each resolution would confound grid effects with classification
and rasterisation effects, and could not answer the question asked.

Reports the before/after REDUCTION at each resolution, since that is the
quantity the study claims, plus peak velocity - the variable the EA found
least reliable.

  python scripts/resolution_study.py --factors 1 2 4 8
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from flood_gpu import simulate, load_terrain, load_soil


def block_mean(a, f):
    h, w = a.shape
    a = a[:h // f * f, :w // f * f]
    return a.reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def block_any(a, f):
    h, w = a.shape
    a = a[:h // f * f, :w // f * f]
    return a.reshape(h // f, f, w // f, f).any(axis=(1, 3))


def block_all(a, f):
    h, w = a.shape
    a = a[:h // f * f, :w // f * f]
    return a.reshape(h // f, f, w // f, f).all(axis=(1, 3))


def coarsen(terrain, f, out):
    """Block-aggregate a terrain dir by an integer factor."""
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    dem = np.load(os.path.join(terrain, "dem.npy"))
    m = dict(np.load(os.path.join(terrain, "masks.npz")))
    np.save(os.path.join(out, "dem.npy"), block_mean(dem, f).astype(np.float32))
    cm = {}
    # `valid` needs every sub-cell surveyed, or a coarse cell straddling the
    # survey edge inherits a fabricated elevation. Everything else is "any":
    # a coarse cell containing a building is an obstacle.
    cm["valid"] = block_all(m["valid"], f)
    for k in ("building", "water", "courtyard"):
        if k in m:
            cm[k] = block_any(m[k], f) & cm["valid"]
    if "eligible" in m:
        cm["eligible"] = block_all(m["eligible"], f) & cm["valid"]
    np.savez_compressed(os.path.join(out, "masks.npz"), **cm)
    for name in ("manning.npy", "infil_mmh.npy", "erodible.npy",
                 "infil_psi_m.npy", "infil_dtheta.npy"):
        p = os.path.join(terrain, name)
        if os.path.exists(p):
            np.save(os.path.join(out, name), block_mean(np.load(p), f).astype(np.float32))
    # rain weight is a VOLUME weight (roof rain rerouted onto street cells), so
    # it must be summed, not averaged, or the coarse grid loses rainfall
    rw = np.load(os.path.join(terrain, "rain_weight.npy"))
    h, w = rw.shape
    rw = rw[:h // f * f, :w // f * f]
    np.save(os.path.join(out, "rain_weight.npy"),
            rw.reshape(h // f, f, w // f, f).sum(axis=(1, 3)).astype(np.float32))
    t = json.load(open(os.path.join(terrain, "dem_transform.json")))
    t2 = dict(t, res=t["res"] * f,
              width=int(cm["valid"].shape[1]), height=int(cm["valid"].shape[0]))
    t2["maxy"] = t["maxy"]
    t2["miny"] = t2["maxy"] - t2["height"] * t2["res"]
    json.dump(t2, open(os.path.join(out, "dem_transform.json"), "w"), indent=2)
    for name in ("gauges.json",):
        p = os.path.join(terrain, name)
        if os.path.exists(p):
            shutil.copy(p, out)
    return t2["res"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="output/terrain_cut_0.5_v3")
    ap.add_argument("--after", default="output/terrain_cut_corridor_v3")
    ap.add_argument("--storm", default="storms/v1_nov2025.json")
    ap.add_argument("--factors", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--out", default="output/resolution")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    storm = json.load(open(args.storm))
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for f in args.factors:
        for tag, src in (("before", args.before), ("after", args.after)):
            terr = src if f == 1 else os.path.join(args.out, f"terrain_{tag}_x{f}")
            res = (json.load(open(os.path.join(src, "dem_transform.json")))["res"]
                   if f == 1 else coarsen(src, f, terr))
            od = os.path.join(args.out, f"{tag}_x{f}")
            if not os.path.exists(os.path.join(od, "max_depth.npy")):
                dem, t, m, man, infil, rw, g, ero = load_terrain(terr)
                print(f"\n=== {tag} at {res:g} m ({dem.shape[1]}x{dem.shape[0]}) ===")
                simulate(dem, t["res"], storm["steps"], storm["duration"], od,
                         manning=man, infil_mmh=infil, valid=m["valid"],
                         water=m["water"], rain_weight=rw, gauges=g,
                         save_every=1e9, device=args.device, save_frames=False,
                         progress=True)
            else:
                print(f"skip {od} (done)")

    print("\nBEFORE/AFTER REDUCTION vs GRID RESOLUTION")
    print(f"{'res (m)':>8} {'cells':>10} {'before m2':>11} {'after m2':>10} "
          f"{'reduction':>10} {'peak |v|':>9}")
    for f in args.factors:
        b = os.path.join(args.out, f"before_x{f}")
        a = os.path.join(args.out, f"after_x{f}")
        if not (os.path.exists(os.path.join(b, "max_depth.npy"))
                and os.path.exists(os.path.join(a, "max_depth.npy"))):
            continue
        terr_b = args.before if f == 1 else os.path.join(args.out, f"terrain_before_x{f}")
        mm = np.load(os.path.join(terr_b, "masks.npz"))
        res = json.load(open(os.path.join(terr_b, "dem_transform.json")))["res"]
        street = mm["valid"] & ~mm["building"] & ~mm["water"]
        if "courtyard" in mm:
            street = street & ~mm["courtyard"]
        db = np.load(os.path.join(b, "max_depth.npy"))
        da = np.load(os.path.join(a, "max_depth.npy"))
        fb = float(((db > 0.05) & street).sum()) * res * res
        fa = float(((da > 0.05) & street).sum()) * res * res
        red = 100 * (fb - fa) / fb if fb > 0 else float("nan")
        vb = np.load(os.path.join(b, "max_vel.npy"))
        pv = float(np.percentile(vb[street & (db > 0.05)], 99)) if (street & (db > 0.05)).any() else 0.0
        print(f"{res:8.1f} {street.size:10d} {fb:11.0f} {fa:10.0f} {red:9.1f}% {pv:8.2f}")
        rows.append({"res_m": res, "cells": int(street.size), "before_m2": fb,
                     "after_m2": fa, "reduction_pct": red, "p99_velocity_ms": pv})

    json.dump({"rows": rows,
               "question": "EA SC120002 sec 6.1: is resolution finer than 2 m worth it?",
               "method": "same terrain block-aggregated, so only the grid varies"},
              open(os.path.join(args.out, "resolution_summary.json"), "w"), indent=2)
    print(f"\nwrote {args.out}/resolution_summary.json")


if __name__ == "__main__":
    main()
