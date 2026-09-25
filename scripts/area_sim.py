#!/usr/bin/env python3
"""Run BOTH solvers on the exact same small, user-chosen area, so they can
be toggled/compared in one viewer instead of living as two disconnected
demos (the 2D grid citywide, a 3D particle peek only reachable via
Ctrl+Click on a much bigger default radius).

Terrain is streamed from the .las and reconstructed ONCE, then handed to:
  - flood_sim.py's grid solver, for the FULL configured storm duration -
    cheap regardless of duration, since cost scales with grid cells and
    this area's grid is tiny.
  - particle_sim.py's SPH engine, for a much shorter, separately-capped
    duration - SPH cost is dominated by particle count and timestep, not
    grid size, so it can never do a full storm; see --particle-duration.

Both write into the same --out directory, so render_3d.py's area-view
command can load them side by side with a 2D/3D toggle.

Usage:
  python scripts/area_sim.py --utm-x 733224 --utm-y 3753476 --radius 25 \
      --rain 50 --duration 900 --particle-duration 15 \
      --out output/area_733224_3753476 --view
"""

import argparse
import os
import sys

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

DEFAULT_RADIUS_M = 25.0        # small on purpose - this is what actually
                                # lets the 3D particle budget (15,000, see
                                # particle_sim.MAX_PARTICLES) read as a
                                # coherent body of water instead of dust
                                # scattered across a whole city block
DEFAULT_PARTICLE_DURATION_S = 15.0


def build_terrain(las_path, utm_x, utm_y, radius_m, out_dir):
    import reconstruct_area
    from las_common import resolve_las_path

    resolved = resolve_las_path(las_path, os.path.join(
        os.path.dirname(SCRIPTS_DIR), "data"))
    if not resolved:
        raise ValueError(f"no .las file found (looked for {las_path!r} and in data/)")
    res = reconstruct_area.auto_res(radius_m)
    terrain_dir = os.path.join(out_dir, "_terrain")
    if not os.path.exists(os.path.join(terrain_dir, "dem.npy")):
        reconstruct_area.build(resolved, utm_x, utm_y, radius_m, res, terrain_dir)
    else:
        print(f"reusing already-built terrain at {terrain_dir}")
    dem = np.load(os.path.join(terrain_dir, "dem.npy")).astype(np.float64)
    return dem, res, terrain_dir, resolved


def run(utm_x, utm_y, radius_m, rain_mmh, duration_s, particle_duration_s,
        out_dir, las_path=None):
    dem, res, terrain_dir, resolved_las = build_terrain(las_path, utm_x, utm_y, radius_m, out_dir)
    print(f"area: {dem.shape[1] * res:.0f} x {dem.shape[0] * res:.0f} m @ {res:.2f} m/cell "
          f"(radius {radius_m:.0f} m)")

    dir_2d = os.path.join(out_dir, "grid2d")
    print(f"\n=== 2D grid solver: {rain_mmh:.0f} mm/h for {duration_s:.0f}s (full storm) ===")
    import flood_sim
    save_every_2d = max(5.0, duration_s / 60.0)
    flood_sim.simulate(dem, rain_mmh, duration_s, save_every_2d, dir_2d, res=res)

    dir_3d = os.path.join(out_dir, "particles3d")
    print(f"\n=== 3D SPH solver: {rain_mmh:.0f} mm/h for {particle_duration_s:.0f}s "
          f"(capped - SPH can't run a full storm at any usable frame rate) ===")
    import ti_init
    ti_init.init()
    import particle_sim
    particle_sim.run(particle_duration_s, dir_3d, scene="area_rain",
                     utm_x=utm_x, utm_y=utm_y, radius_m=radius_m, las_path=resolved_las,
                     rain_mmh=rain_mmh, terrain_dir=terrain_dir)

    print(f"\ndone -> {out_dir}/ (grid2d/, particles3d/, _terrain/)")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utm-x", type=float, required=True)
    ap.add_argument("--utm-y", type=float, required=True)
    ap.add_argument("--radius", type=float, default=DEFAULT_RADIUS_M,
                    help=f"crop radius, meters (default {DEFAULT_RADIUS_M:.0f} - kept small "
                         "so the 3D particle budget still looks like water, not dust)")
    ap.add_argument("--rain", type=float, default=30.0, help="mm/h")
    ap.add_argument("--duration", type=float, default=900.0,
                    help="2D grid solver duration, seconds (default 900 = 15 min)")
    ap.add_argument("--particle-duration", type=float, default=DEFAULT_PARTICLE_DURATION_S,
                    help=f"3D SPH solver duration, seconds (default {DEFAULT_PARTICLE_DURATION_S:.0f} "
                         "- independent of --duration, SPH is real-time-cost regardless of area size)")
    ap.add_argument("--las", help=".las path (default: auto-discover in data/)")
    ap.add_argument("--out", help="default: output/area_<x>_<y>")
    ap.add_argument("--view", action="store_true", help="open the 2D/3D toggle viewer when done")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(
        os.path.dirname(SCRIPTS_DIR), "output", f"area_{int(args.utm_x)}_{int(args.utm_y)}")

    run(args.utm_x, args.utm_y, args.radius, args.rain, args.duration,
        args.particle_duration, out_dir, las_path=args.las)

    if args.view:
        import render_3d
        render_3d.cmd_area_view(argparse.Namespace(dir=out_dir, z_exagg=1.0))


if __name__ == "__main__":
    main()
