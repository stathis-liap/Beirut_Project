#!/usr/bin/env python3
"""Bake the green-corridor materials into a terrain directory.

Reads a material-class raster (scripts/build_corridor_gi.py) and applies,
per class, an infiltration rate, a Manning roughness, and a surface
depression (detention storage) to a COPY of the source terrain. The copy is
then a drop-in `--terrain` for flood_gpu.py, so the "after" scenario is a
full high-fidelity terrain, not a polygon edit.

Material properties (literature-grounded; see report methods):
  infiltration mm/h | Manning n | depression m (lowers DEM to detain)
  | erodibility 0-1 (opt-in erosion sub-model susceptibility - illustrative/
  tunable, NOT independently calibrated like the other three columns; see
  flood_gpu.py's EROSION_DEFAULTS docstring)
"""
import argparse, json, os, shutil, sys
import numpy as np

# Green-Ampt parameters for the engineered GI media (see the note at the
# psi/dtheta write below). Literature values for a sand/gravel filter build-up,
# uncalibrated like everything else in PROPS.
GI_PSI_M = 0.05
GI_DTHETA = 0.30

# class: (infil mm/h, manning n, depression m, erodibility 0-1, label)
PROPS = {
    1: (5.0,   0.016, 0.00, 0.00, "vehicular lane"),
    2: (150.0, 0.020, 0.00, 0.00, "porous bikelane"),
    3: (200.0, 0.150, 0.15, 0.50, "bioswale"),
    4: (150.0, 0.020, 0.00, 0.00, "porous sidewalk"),
    5: (100.0, 0.100, 0.00, 0.40, "garden"),
    6: (250.0, 0.200, 0.40, 0.30, "bioretention pond / rain garden"),
    7: (100.0, 0.120, 0.10, 0.20, "terrace"),
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
    epath = os.path.join(args.out, "erodible.npy")
    # the source terrain may predate the erosion sub-model (no city-wide
    # erodible.npy yet) - start from all-zero rather than failing, since
    # the corridor materials below still get their own erodibility overlaid.
    erodible = np.load(epath) if os.path.exists(epath) else np.zeros_like(dem)

    # Deferred building clearing (flatten_corridor_buildings.py
    # --defer-to-design): the source terrain is the "before" world with every
    # building standing, and building the corridor is what demolishes the ones
    # in its path. Applying it here keeps this bake equivalent to the sandbox's
    # official-corridor design, which carries the same delta - the property
    # sandbox/baking.py exists to preserve.
    fd_path = os.path.join(args.out, "flatten_delta.npy")
    if os.path.exists(fd_path):
        delta = np.load(fd_path)
        cleared = int((delta < 0).sum())
        dem = dem + delta
        masks = dict(np.load(os.path.join(args.out, "masks.npz")))
        fm_path = os.path.join(args.out, "flatten_mask.npy")
        if os.path.exists(fm_path):
            fmask = np.load(fm_path)
            masks["building"] = masks["building"] & ~fmask
            if "courtyard" in masks:
                masks["courtyard"] = masks["courtyard"] & ~fmask
            if "eligible" in masks:
                masks["eligible"] = masks["eligible"] | (fmask & masks["valid"])
            np.savez_compressed(os.path.join(args.out, "masks.npz"), **masks)
        # roofs that no longer exist must stop feeding downspouts
        rwc_path = os.path.join(args.out, "rain_weight_cleared.npy")
        if os.path.exists(rwc_path):
            np.save(os.path.join(args.out, "rain_weight.npy"),
                    np.load(rwc_path).astype(np.float32))
        print(f"corridor clears {cleared} building cells "
              f"(mean drop {float(delta[delta < 0].mean()):.1f} m)")

    changed = 0
    for cls, (inf, n, dep, erod, lab) in PROPS.items():
        m = mat == cls
        if not m.any():
            continue
        infil[m] = inf
        man[m] = n
        erodible[m] = erod
        if dep > 0:
            dem[m] = dem[m] - dep      # detention: lower the cell
        changed += int(m.sum())
        print(f"  class {cls} {lab:32} {m.sum():6d} cells  "
              f"infil {inf:.0f} n {n:.3f} depr {dep:.2f} m erod {erod:.2f}")

    # Green-Ampt soil parameters for the corridor build-up. The GI media is an
    # engineered sand/gravel filter, not the natural soil underneath: high
    # conductivity but LOW capillary suction and high available porosity, so it
    # cannot simply inherit the land-cover values build_terrain.py assigned.
    psi_p = os.path.join(args.out, "infil_psi_m.npy")
    dth_p = os.path.join(args.out, "infil_dtheta.npy")
    if os.path.exists(psi_p) and os.path.exists(dth_p):
        gi = mat > 0
        psi = np.load(psi_p); psi[gi] = GI_PSI_M; np.save(psi_p, psi.astype(np.float32))
        dth = np.load(dth_p); dth[gi] = GI_DTHETA; np.save(dth_p, dth.astype(np.float32))
        print(f"  green-ampt: {int(gi.sum())} corridor cells set to engineered media "
              f"(psi {GI_PSI_M} m, dtheta {GI_DTHETA})")

    np.save(os.path.join(args.out, "dem.npy"), dem.astype(np.float32))
    np.save(os.path.join(args.out, "manning.npy"), man.astype(np.float32))
    np.save(os.path.join(args.out, "infil_mmh.npy"), infil.astype(np.float32))
    np.save(os.path.join(args.out, "erodible.npy"), erodible.astype(np.float32))
    res = json.load(open(os.path.join(args.out, "dem_transform.json")))["res"]
    json.dump({"source": args.terrain, "material": args.material,
               "props": {k: v for k, v in PROPS.items()},
               "cells_changed": changed,
               "area_m2": round(changed * res * res)},
              open(os.path.join(args.out, "corridor_bake.json"), "w"), indent=2)
    print(f"baked {changed} cells ({changed*res*res/1e4:.2f} ha) -> {args.out}")

if __name__ == "__main__":
    main()
