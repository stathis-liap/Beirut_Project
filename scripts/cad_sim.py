#!/usr/bin/env python3
"""Same idea as area_sim.py (one terrain feeding both the 2D grid solver and
the capped 3D SPH solver, viewable in one 2D/3D toggle window), but sourced
from a CAD model - .3dm via load_cad_model.py, .dwg via load_dwg_model.py
(picked from the --model extension) - instead of streamed from a .las.

Not UTM-addressable - this is a "local scene" (own coordinate origin, no
real-world georeference), so there's no --utm-x/--utm-y here, just a model
file and a resolution.

Usage:
  python scripts/cad_sim.py --model "3D/3D Site.3dm" --res 0.4 \
      --rain 30 --duration 600 --particle-duration 12 --view
  python scripts/cad_sim.py --model "3D/some_model.dwg" --res 0.4 --view
"""

import argparse
import os
import sys

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

DEFAULT_PARTICLE_DURATION_S = 12.0


def run(model_path, res, rain_mmh, duration_s, particle_duration_s, out_dir, unit_scale=1.0):
    ext = os.path.splitext(model_path)[1].lower()
    terrain_dir = os.path.join(out_dir, "_terrain")
    if os.path.exists(os.path.join(terrain_dir, "dem.npy")):
        print(f"reusing already-built terrain at {terrain_dir}")
    elif ext == ".dwg":
        import load_dwg_model
        load_dwg_model.build(model_path, res, terrain_dir, unit_scale=unit_scale)
    else:
        import load_cad_model
        load_cad_model.build(model_path, res, terrain_dir, unit_scale=unit_scale)
    dem = np.load(os.path.join(terrain_dir, "dem.npy")).astype(np.float64)
    print(f"terrain: {dem.shape[1] * res:.0f} x {dem.shape[0] * res:.0f} m @ {res:.2f} m/cell")

    dir_2d = os.path.join(out_dir, "grid2d")
    print(f"\n=== 2D grid solver: {rain_mmh:.0f} mm/h for {duration_s:.0f}s (full storm) ===")
    import flood_sim
    save_every_2d = max(2.0, duration_s / 60.0)
    flood_sim.simulate(dem, rain_mmh, duration_s, save_every_2d, dir_2d, res=res)

    dir_3d = os.path.join(out_dir, "particles3d")
    print(f"\n=== 3D SPH solver: {rain_mmh:.0f} mm/h for {particle_duration_s:.0f}s (capped) ===")
    import ti_init
    ti_init.init()
    import particle_sim
    particle_sim.run(particle_duration_s, dir_3d, scene="cad_model",
                     rain_mmh=rain_mmh, terrain_dir=terrain_dir)

    print(f"\ndone -> {out_dir}/ (grid2d/, particles3d/, _terrain/)")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to a .3dm file")
    ap.add_argument("--res", type=float, default=0.4, help="m/cell (see load_cad_model.py)")
    ap.add_argument("--unit-scale", type=float, default=1.0)
    ap.add_argument("--rain", type=float, default=30.0, help="mm/h")
    ap.add_argument("--duration", type=float, default=600.0,
                    help="2D grid solver duration, seconds (default 600 = 10 min)")
    ap.add_argument("--particle-duration", type=float, default=DEFAULT_PARTICLE_DURATION_S)
    ap.add_argument("--out", help="default: output/cad_<model filename>_sim")
    ap.add_argument("--view", action="store_true", help="open the 2D/3D toggle viewer when done")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(
        os.path.dirname(SCRIPTS_DIR), "output",
        "cad_" + os.path.splitext(os.path.basename(args.model))[0].replace(" ", "_") + "_sim")

    run(args.model, args.res, args.rain, args.duration, args.particle_duration,
        out_dir, unit_scale=args.unit_scale)

    if args.view:
        import render_3d
        render_3d.cmd_area_view(argparse.Namespace(dir=out_dir, z_exagg=1.0))


if __name__ == "__main__":
    main()
