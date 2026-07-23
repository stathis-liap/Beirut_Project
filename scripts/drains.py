#!/usr/bin/env python3
"""Storm-drain inlets along OSM streets (scenario S1/S3 input).

No measured drainage data exists for Beirut (PLAN.md risks), so drains are
a scenario parameter: inlets every --spacing m along mapped streets, each
capped at --capacity m3/s (default 0.03 = a typical double grate). The
baseline scenario simply omits this file (drains clogged — the documented
failure mode).

Usage:
  python scripts/drains.py --terrain output/terrain_1.0 \
      --osm output/osm/osm.json [--spacing 27] [--capacity 0.03]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from las_common import load_transform, utm_to_pixel

SKIP_HIGHWAYS = {"footway", "steps", "path", "cycleway", "pedestrian",
                 "corridor", "track"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default="output/terrain_1.0")
    ap.add_argument("--osm", default="output/osm/osm.json")
    ap.add_argument("--spacing", type=float, default=27.0)
    ap.add_argument("--capacity", type=float, default=0.03, help="m3/s per inlet")
    args = ap.parse_args()

    from shapely.geometry import LineString
    t = load_transform(os.path.join(args.terrain, "dem_transform.json"))
    masks = np.load(os.path.join(args.terrain, "masks.npz"))
    eligible = masks["eligible"]
    h, w = eligible.shape

    with open(args.osm) as f:
        roads = json.load(f)["roads"]

    pts = []
    for rd in roads:
        if rd.get("highway") in SKIP_HIGHWAYS or len(rd["coords"]) < 2:
            continue
        line = LineString(rd["coords"])
        n = max(1, int(line.length // args.spacing))
        for k in range(n + 1):
            p = line.interpolate(min(k * args.spacing, line.length))
            pts.append((p.x, p.y))
    pts = np.array(pts)
    if len(pts) == 0:
        sys.exit("no road points - check the OSM file")

    cols, rows = utm_to_pixel(t, pts[:, 0], pts[:, 1])
    rows = np.round(rows).astype(int)
    cols = np.round(cols).astype(int)
    ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    rows, cols = rows[ok], cols[ok]
    ok = eligible[rows, cols]
    rows, cols = rows[ok], cols[ok]
    # dedupe cells
    flat = np.unique(rows.astype(np.int64) * w + cols)
    rows, cols = (flat // w).astype(np.int32), (flat % w).astype(np.int32)
    cap = np.full(len(rows), args.capacity, dtype=np.float32)

    out = os.path.join(args.terrain, "drains.npz")
    np.savez_compressed(out, rows=rows, cols=cols, cap=cap)
    print(f"{len(rows)} inlets @ {args.capacity} m3/s "
          f"(total {cap.sum():.1f} m3/s) -> {out}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    ortho = np.asarray(Image.open(os.path.join(args.terrain, "ortho.png")))
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(ortho)
    ax.plot(cols, rows, ".", color="cyan", ms=1.5)
    ax.set_title(f"{len(rows)} storm-drain inlets, {args.capacity} m3/s each")
    ax.axis("off")
    fig.savefig(os.path.join(args.terrain, "drains_qa.png"),
                dpi=130, bbox_inches="tight")
    print("wrote drains_qa.png")


if __name__ == "__main__":
    main()
