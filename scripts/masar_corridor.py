#!/usr/bin/env python3
"""Build the Al-Masar Al-Akhdar scenario (S2) edit file.

The green corridor is modeled as an infiltration strip (bioretention /
rain-garden soils, FAWB 2009: 100-300 mm/h — default 150) with rougher
vegetated surface (n = 0.05) along the corridor path.

Path source, in order of preference:
  --polygon file(s): polygons drawn with `scenario.py draw` on the ortho
     (use this to trace the real Fouad Boutros right-of-way);
  otherwise: auto-trace of the main drainage channel (proxy for the
     corridor, which follows the valley line).

Writes scenarios/masar_corridor.json + a preview PNG.

Usage:
  python scripts/masar_corridor.py --terrain output/terrain_1.0 \
      [--polygon output/masar1.json output/masar2.json] [--width 16]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from las_common import load_transform, utm_to_pixel
from make_scenarios import trace_main_channel, buffer_polygon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default="output/terrain_1.0")
    ap.add_argument("--polygon", nargs="*", help="drawn polygon JSONs")
    ap.add_argument("--width", type=float, default=16.0, help="strip width, m")
    ap.add_argument("--infil-mmh", type=float, default=150.0)
    ap.add_argument("--manning", type=float, default=0.05)
    ap.add_argument("--out", default="scenarios/masar_corridor.json")
    args = ap.parse_args()

    t = load_transform(os.path.join(args.terrain, "dem_transform.json"))
    dem = np.load(os.path.join(args.terrain, "dem.npy"))

    polys = []
    if args.polygon:
        for p in args.polygon:
            with open(p) as f:
                polys.append(json.load(f)["polygon"])
        print(f"{len(polys)} drawn polygons")
    else:
        target = 400_000
        factor = max(2, int(np.ceil(np.sqrt(dem.size / target))))
        print(f"auto-tracing main channel (coarsen factor {factor})...")
        path = trace_main_channel(dem, t["res"], factor=factor)
        print(f"channel: {len(path)} cells, "
              f"{dem[tuple(path[0])]:.1f} -> {dem[tuple(path[-1])]:.1f} m")
        polys = [buffer_polygon(path, dem.shape, t, args.width).tolist()]

    edits = []
    for poly in polys:
        edits.append({"op": "infiltrate", "mmh": args.infil_mmh, "polygon": poly})
        edits.append({"op": "manning", "n": args.manning, "polygon": poly})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"name": "masar_corridor", "edits": edits}, f, indent=2)
    print(f"wrote {args.out}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    ortho = np.asarray(Image.open(os.path.join(args.terrain, "ortho.png")))
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(ortho)
    for poly in polys:
        p = np.asarray(poly)
        px, py = utm_to_pixel(t, p[:, 0], p[:, 1])
        ax.plot(np.append(px, px[0]), np.append(py, py[0]), color="lime", lw=2)
    ax.set_title(f"Al-Masar corridor strip: infiltrate {args.infil_mmh:.0f} mm/h, "
                 f"n={args.manning}")
    ax.axis("off")
    fig.savefig(os.path.join(args.terrain, "masar_qa.png"),
                dpi=130, bbox_inches="tight")
    print(f"wrote {os.path.join(args.terrain, 'masar_qa.png')}")


if __name__ == "__main__":
    main()
