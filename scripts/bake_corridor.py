#!/usr/bin/env python3
"""Bake the green-corridor materials into a terrain directory.

Reads a material-class raster (scripts/build_corridor_gi.py) and applies,
per class, an infiltration rate, a Manning roughness, and a surface
depression (detention storage) to a COPY of the source terrain. The copy is
then a drop-in `--terrain` for flood_gpu.py, so the "after" scenario is a
full high-fidelity terrain, not a polygon edit.

Material properties (literature-grounded; see report methods):
  infiltration mm/h | Manning n | depression m (lowers DEM to detain)
"""
import argparse, json, os, shutil, sys
import numpy as np

# class: (infil mm/h, manning n, depression m, label)
PROPS = {
    1: (5.0,   0.016, 0.00, "vehicular lane"),
    2: (150.0, 0.020, 0.00, "porous bikelane"),
    3: (200.0, 0.150, 0.15, "bioswale"),
    4: (150.0, 0.020, 0.00, "porous sidewalk"),
    5: (100.0, 0.100, 0.00, "garden"),
    6: (250.0, 0.200, 0.40, "bioretention pond / rain garden"),
    7: (100.0, 0.120, 0.10, "terrace"),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default="output/terrain_cut_0.5")
    ap.add_argument("--material", default="output/corridor_gi_cut/material.npy")
    ap.add_argument("--out", default="output/terrain_cut_corridor")
    args = ap.parse_args()

    if os.path.abspath(args.terrain) == os.path.abspath(args.out):
        sys.exit("refusing to overwrite the source terrain")
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    shutil.copytree(args.terrain, args.out)

    mat = np.load(args.material)
    dem = np.load(os.path.join(args.out, "dem.npy"))
    man = np.load(os.path.join(args.out, "manning.npy"))
    infil = np.load(os.path.join(args.out, "infil_mmh.npy"))
    if mat.shape != dem.shape:
        sys.exit(f"material {mat.shape} != terrain {dem.shape}")

    changed = 0
    for cls, (inf, n, dep, lab) in PROPS.items():
        m = mat == cls
        if not m.any():
            continue
        infil[m] = inf
        man[m] = n
        if dep > 0:
            dem[m] = dem[m] - dep      # detention: lower the cell
        changed += int(m.sum())
        print(f"  class {cls} {lab:32} {m.sum():6d} cells  "
              f"infil {inf:.0f} n {n:.3f} depr {dep:.2f} m")

    np.save(os.path.join(args.out, "dem.npy"), dem.astype(np.float32))
    np.save(os.path.join(args.out, "manning.npy"), man.astype(np.float32))
    np.save(os.path.join(args.out, "infil_mmh.npy"), infil.astype(np.float32))
    res = json.load(open(os.path.join(args.out, "dem_transform.json")))["res"]
    json.dump({"source": args.terrain, "material": args.material,
               "props": {k: v for k, v in PROPS.items()},
               "cells_changed": changed,
               "area_m2": round(changed * res * res)},
              open(os.path.join(args.out, "corridor_bake.json"), "w"), indent=2)
    print(f"baked {changed} cells ({changed*res*res/1e4:.2f} ha) -> {args.out}")

if __name__ == "__main__":
    main()
