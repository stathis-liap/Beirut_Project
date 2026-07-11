#!/usr/bin/env python3
"""Overlay the drone preview on an OpenStreetMap basemap for geographic context.

Fetches OSM tiles (cached locally), warps preview_rgb.png from EPSG:32636
into the map projection, and writes a two-panel figure: wide city context
with the survey footprint outlined, and a close-up with the drone imagery
draped on the map.

Usage:
  python scripts/map_overlay.py [--data-dir output] [--out output/preview_on_map.png]
"""

import argparse
import math
import os
import sys
import time
import urllib.request

import numpy as np
from PIL import Image
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(__file__))
from las_common import load_transform

TILE = 256
HEADERS = {"User-Agent": "BeirutFloodProject/1.0 (research demo)"}


def fetch_tile(z, x, y, cache):
    path = os.path.join(cache, f"{z}_{x}_{y}.png")
    if not os.path.exists(path):
        url = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
        time.sleep(0.1)  # be polite to the tile server
    return np.asarray(Image.open(path).convert("RGB"))


def lonlat_to_tilef(lon, lat, z):
    n = 2 ** z
    tx = (lon + 180.0) / 360.0 * n
    ty = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return tx, ty


def build_mosaic(lon0, lat0, lon1, lat1, z, cache):
    """Mosaic of all tiles covering the lon/lat box; returns image + the
    mosaic's own lon/lat bounds and a lon/lat -> pixel mapper."""
    tx0, ty1 = lonlat_to_tilef(lon0, lat0, z)
    tx1, ty0 = lonlat_to_tilef(lon1, lat1, z)
    x0, x1 = int(tx0), int(tx1)
    y0, y1 = int(ty0), int(ty1)
    img = np.zeros(((y1 - y0 + 1) * TILE, (x1 - x0 + 1) * TILE, 3), np.uint8)
    n_tiles = (x1 - x0 + 1) * (y1 - y0 + 1)
    k = 0
    for xi in range(x0, x1 + 1):
        for yi in range(y0, y1 + 1):
            img[(yi - y0) * TILE:(yi - y0 + 1) * TILE,
                (xi - x0) * TILE:(xi - x0 + 1) * TILE] = fetch_tile(z, xi, yi, cache)
            k += 1
            print(f"\r  z{z}: tile {k}/{n_tiles}", end="", flush=True)
    print()

    def to_px(lon, lat):
        tx, ty = np.vectorize(lambda lo, la: lonlat_to_tilef(lo, la, z))(lon, lat)
        return (tx - x0) * TILE, (ty - y0) * TILE

    return img, to_px


def warp_preview(mosaic, to_px, z, x_off_tiles=None, *, t, preview, covered,
                 alpha=0.8):
    """Blend the preview onto the mosaic where the survey has data."""
    h, w = mosaic.shape[:2]
    # mosaic pixel centers -> lon/lat -> UTM -> preview pixel
    # invert to_px analytically: build lon/lat per pixel via tile math
    # (vectorized without np.vectorize for speed)
    # reconstruct tile-space coords from to_px(0,0)? simpler: pass z and
    # figure global tile offset from a reference point
    lon_ref, lat_ref = 35.5, 33.9
    px_ref, py_ref = to_px(lon_ref, lat_ref)
    tx_ref, ty_ref = lonlat_to_tilef(lon_ref, lat_ref, z)
    n = 2 ** z
    cols = (np.arange(w) + 0.5 - float(px_ref)) / TILE + tx_ref
    rows = (np.arange(h) + 0.5 - float(py_ref)) / TILE + ty_ref
    lon = cols / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * rows / n))))
    lon2d, lat2d = np.meshgrid(lon, lat)

    tr = Transformer.from_crs(4326, 32636, always_xy=True)
    ux, uy = tr.transform(lon2d, lat2d)
    pc = ((ux - t["minx"]) / t["res"]).astype(np.int64)
    pr = ((t["maxy"] - uy) / t["res"]).astype(np.int64)
    inside = ((pc >= 0) & (pc < t["width"]) & (pr >= 0) & (pr < t["height"]))
    pc = np.clip(pc, 0, t["width"] - 1)
    pr = np.clip(pr, 0, t["height"] - 1)
    mask = inside & covered[pr, pc]

    out = mosaic.astype(np.float32)
    src = preview[pr, pc].astype(np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * src[mask]
    return out.astype(np.uint8), mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="output")
    ap.add_argument("--out", default="output/preview_on_map.png")
    ap.add_argument("--alpha", type=float, default=0.8)
    args = ap.parse_args()

    t = load_transform(os.path.join(args.data_dir, "preview_transform.json"))
    preview = np.asarray(Image.open(
        os.path.join(args.data_dir, "preview_rgb.png")))
    covered = np.load(os.path.join(args.data_dir, "preview_count.npy")) > 0

    tr = Transformer.from_crs(32636, 4326, always_xy=True)
    corners_x = [t["minx"], t["minx"] + t["width"] * t["res"]]
    corners_y = [t["miny"], t["maxy"]]
    lons, lats = [], []
    for cx in corners_x:
        for cy in corners_y:
            lo, la = tr.transform(cx, cy)
            lons.append(lo)
            lats.append(la)
    lon0, lon1, lat0, lat1 = min(lons), max(lons), min(lats), max(lats)
    print(f"survey footprint: lon {lon0:.4f}..{lon1:.4f}  "
          f"lat {lat0:.4f}..{lat1:.4f}")

    cache = os.path.join(args.data_dir, "osm_tiles")
    os.makedirs(cache, exist_ok=True)

    # wide context: ~2.5x padding around the survey
    dlon, dlat = lon1 - lon0, lat1 - lat0
    wide, wide_px = build_mosaic(lon0 - 1.2 * dlon, lat0 - 1.2 * dlat,
                                 lon1 + 1.2 * dlon, lat1 + 1.2 * dlat,
                                 13, cache)
    wide_ov, _ = warp_preview(wide, wide_px, 13, t=t, preview=preview,
                              covered=covered, alpha=0.55)

    # close-up: small padding, overlay
    close, close_px = build_mosaic(lon0 - 0.12 * dlon, lat0 - 0.12 * dlat,
                                   lon1 + 0.12 * dlon, lat1 + 0.12 * dlat,
                                   16, cache)
    close_ov, _ = warp_preview(close, close_px, 16, t=t, preview=preview,
                               covered=covered, alpha=args.alpha)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11))
    ax1.imshow(wide_ov)
    bx = [lon0, lon1, lon1, lon0]
    by = [lat0, lat0, lat1, lat1]
    px, py = wide_px(np.array(bx), np.array(by))
    ax1.add_patch(MplPolygon(np.column_stack([px, py]), closed=True,
                             fill=False, edgecolor="red", linewidth=2.5))
    ax1.set_title("Beirut — survey footprint (red) on OpenStreetMap", fontsize=14)
    ax1.axis("off")

    ax2.imshow(close_ov)
    ax2.set_title("Drone preview draped on the map", fontsize=14)
    ax2.axis("off")

    fig.text(0.5, 0.02, "Basemap © OpenStreetMap contributors",
             ha="center", fontsize=9, color="gray")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
