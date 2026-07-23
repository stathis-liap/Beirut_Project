#!/usr/bin/env python3
"""Assemble the simulation-ready terrain from a raster stack + OSM.

Steps (PLAN.md Phase A):
  1. clean the low-Z surface (hole fill inside coverage, despeckle)
  2. land cover per cell: building (OSM), tree canopy (ExG + relief),
     grass, soil (color heuristic), water (OSM), paved (default)
  3. hydraulic surface: remove canopy -> re-interpolate ground from
     non-building neighbors; buildings stay as obstacles at roof height;
     optional bridge/underpass cuts (polygons re-interpolated as ground)
  4. courtyards: walkable cells not connected to the open street network
  5. roof/courtyard rain rerouting -> rain_weight raster (downspout model)
  6. Manning n + infiltration rasters from land cover
  7. static fill bound (GPU morphological reconstruction) for sanity screens
  8. auto gauges at the highest flow-accumulation points

Outputs in --out (names compatible with render_3d.py / scenario.py):
  dem.npy dem_transform.json ortho.png landcover.npy manning.npy
  infil_mmh.npy rain_weight.npy fillbound.npy masks.npz gauges.json
  QA: hillshade.png landcover_qa.png courtyards_qa.png

Usage:
  python scripts/build_terrain.py --stack output/stack_1.0 \
      --osm output/osm/osm.json --out output/terrain_1.0
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
from las_common import load_transform, save_transform, utm_to_pixel
from build_dem import despeckle, hillshade, fill_sinks, d8_flow_accumulation

# land-cover classes
PAVED, BUILDING, CANOPY, GRASS, SOIL, WATER = 0, 1, 2, 3, 4, 5
CLASS_NAMES = ["paved", "building", "canopy", "grass", "soil", "water"]
MANNING = {PAVED: 0.016, BUILDING: 0.05, CANOPY: 0.10,
           GRASS: 0.035, SOIL: 0.025, WATER: 0.03}
INFIL_MMH = {PAVED: 0.0, BUILDING: 0.0, CANOPY: 11.0,
             GRASS: 11.0, SOIL: 8.0, WATER: 0.0}


def fill_holes_masked(dem, fillable, sources=None, max_iters=300):
    """Fill NaN cells inside `fillable` from neighbor means; only cells in
    `sources` (default: all finite) seed values, so e.g. roofs don't leak
    into ground interpolation."""
    filled = dem.copy()
    if sources is not None:
        filled[~sources & np.isfinite(filled)] = np.nan
    k = np.ones((3, 3), dtype=np.float32)
    for _ in range(max_iters):
        nan = np.isnan(filled) & fillable
        if not nan.any():
            break
        v = np.where(np.isnan(filled), 0.0, filled).astype(np.float32)
        m = (~np.isnan(filled)).astype(np.float32)
        vs = ndimage.convolve(v, k, mode="nearest")
        ms = ndimage.convolve(m, k, mode="nearest")
        can = nan & (ms > 0)
        filled[can] = vs[can] / ms[can]
    out = dem.copy()
    take = fillable & np.isfinite(filled)
    out[take] = filled[take]
    return out


def rasterize_polys(polys, t, shape, all_touched=True):
    from rasterio import features
    from affine import Affine
    if not polys:
        return np.zeros(shape, dtype=bool)
    from shapely.geometry import Polygon
    shapes = []
    for p in polys:
        try:
            g = Polygon(p)
            if g.is_valid and g.area > 0:
                shapes.append(g)
            elif not g.is_valid:
                g = g.buffer(0)
                if g.area > 0:
                    shapes.append(g)
        except Exception:
            continue
    if not shapes:
        return np.zeros(shape, dtype=bool)
    tr = Affine(t["res"], 0, t["minx"], 0, -t["res"], t["maxy"])
    m = features.rasterize(((g, 1) for g in shapes), out_shape=shape,
                           transform=tr, fill=0, all_touched=all_touched,
                           dtype=np.uint8)
    return m.astype(bool)


def rasterize_lines(lines, t, shape, buffer_m=6.0):
    from shapely.geometry import LineString
    polys = []
    for c in lines:
        if len(c) >= 2:
            try:
                polys.append(list(LineString(c).buffer(buffer_m).exterior.coords))
            except Exception:
                continue
    return rasterize_polys(polys, t, shape)


def otsu_threshold(vals, bins=256):
    hist, edges = np.histogram(vals, bins=bins)
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = np.divide(m0, w0, out=np.zeros_like(m0), where=w0 > 0)
    mu1 = np.divide(m0[-1] - m0, w1, out=np.zeros_like(m0), where=w1 > 0)
    between = w0 * w1 * (mu0 - mu1) ** 2
    return centers[np.argmax(between)]


def fill_bound_gpu(dem, valid, water=None, eps_stop=1e-4, max_iters=40000):
    """Priority-flood-equivalent depression filling by morphological
    reconstruction (erosion) on the GPU. Returns filled elevation.
    Open boundary = grid border + cells next to no-data + water outlets
    (the solver zeroes depth there, so the static bound must too)."""
    import torch
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    low = float(np.nanmin(dem[valid])) - 10.0
    d = np.where(valid, dem, low).astype(np.float32)
    seed = np.zeros_like(valid)
    seed[0, :] = seed[-1, :] = seed[:, 0] = seed[:, -1] = True
    seed |= ndimage.binary_dilation(~valid) & valid
    if water is not None:
        seed |= water
        d = np.where(water, low, d)
    dt = torch.tensor(d, device=dev)
    st = torch.tensor(seed | ~valid, device=dev)
    big = torch.tensor(float(np.nanmax(dem[valid])) + 100.0, device=dev)
    Ft = torch.where(st, dt, big.expand_as(dt)).clone()
    for i in range(max_iters):
        er = -F.max_pool2d(-Ft.unsqueeze(0).unsqueeze(0), 3, 1, 1)[0, 0]
        Fn = torch.maximum(dt, er)
        if i % 256 == 255:
            if (Ft - Fn).max().item() < eps_stop:
                Ft = Fn
                break
        Ft = Fn
    out = Ft.cpu().numpy()
    out[~valid] = np.nan
    return out


def auto_gauges(dem, valid, t, n_gauges=5, min_sep_m=150.0):
    """Probe points at the strongest interior flow-accumulation cells.

    Restricted to the p20-p85 elevation band of the survey so gauges land
    on the drainage paths of the built slope (the corridor), not on the
    harbor quays / sea-noise area where D8 accumulation is largest."""
    target = 400_000
    factor = max(1, int(np.ceil(np.sqrt(dem.size / target))))
    hc, wc = dem.shape[0] // factor, dem.shape[1] // factor
    d = dem[:hc * factor, :wc * factor].reshape(hc, factor, wc, factor)
    with np.errstate(all="ignore"):
        coarse = np.nanmin(np.nanmin(d, axis=3), axis=1)
    finite = np.isfinite(coarse)
    coarse = np.where(finite, coarse, np.nanmax(coarse) + 50.0)
    acc = d8_flow_accumulation(fill_sinks(coarse))
    interior = ndimage.distance_transform_edt(finite) > max(3, 20 / (t["res"] * factor))
    lo, hi = np.nanpercentile(dem[valid], [20, 85])
    band = finite & (coarse >= lo) & (coarse <= hi)
    acc = np.where(interior & band, acc, 0)
    gauges, taken = [], []
    order = np.argsort(acc, axis=None)[::-1]
    rows, cols = np.unravel_index(order, acc.shape)
    min_sep_px = min_sep_m / (t["res"] * factor)
    for r, c in zip(rows, cols):
        if acc[r, c] <= 0:
            break
        if all((r - rr) ** 2 + (c - cc) ** 2 >= min_sep_px ** 2 for rr, cc in taken):
            taken.append((r, c))
            fr, fc = r * factor + factor // 2, c * factor + factor // 2
            x = t["minx"] + (fc + 0.5) * t["res"]
            y = t["maxy"] - (fr + 0.5) * t["res"]
            gauges.append({"name": f"g{len(gauges) + 1}",
                           "x": round(x, 1), "y": round(y, 1)})
            if len(gauges) >= n_gauges:
                break
    return gauges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="output/stack_1.0")
    ap.add_argument("--osm", default="output/osm/osm.json")
    ap.add_argument("--out", default=None, help="default output/terrain_<res>")
    ap.add_argument("--exg", default="otsu", help="'otsu' or a float threshold")
    ap.add_argument("--canopy-relief", type=float, default=2.0)
    ap.add_argument("--sea-level", type=float, default=26.0,
                    help="ellipsoidal sea-surface height (Beirut LAS: ~+26 m)")
    ap.add_argument("--sea-margin", type=float, default=1.5,
                    help="cells below sea-level+margin become outflow water")
    ap.add_argument("--cuts", help="JSON {'polygons': [[[x,y],...],...]} "
                                   "bridge/underpass cuts, re-interpolated as ground")
    ap.add_argument("--no-fillbound", action="store_true")
    ap.add_argument("--geotiff", action="store_true", help="also write COG-ish GeoTIFFs")
    ap.add_argument("--gauges-only", action="store_true",
                    help="recompute gauges.json on an existing terrain dir and exit")
    args = ap.parse_args()

    if args.gauges_only:
        out = args.out or f"output/terrain_x"
        t = load_transform(os.path.join(out, "dem_transform.json"))
        dem = np.load(os.path.join(out, "dem.npy"))
        valid = np.load(os.path.join(out, "masks.npz"))["valid"]
        gauges = auto_gauges(np.where(valid, dem, np.nan), valid, t)
        with open(os.path.join(out, "gauges.json"), "w") as f:
            json.dump(gauges, f, indent=2)
        print(f"gauges rewritten: {gauges}")
        return

    t = load_transform(os.path.join(args.stack, "transform.json"))
    res = t["res"]
    out = args.out or f"output/terrain_{res:g}"
    os.makedirs(out, exist_ok=True)

    zlow = np.load(os.path.join(args.stack, "zlow.npy"))
    zhigh = np.load(os.path.join(args.stack, "zhigh.npy"))
    rgb = np.load(os.path.join(args.stack, "rgb.npy"))
    count = np.load(os.path.join(args.stack, "count.npy"))
    h, w = zlow.shape
    print(f"stack {w} x {h} at {res} m")

    # --- 1. clean surface --------------------------------------------------
    covered = count > 0
    valid = ndimage.binary_fill_holes(covered)
    print(f"coverage {100 * covered.mean():.1f}%, valid (holes filled) "
          f"{100 * valid.mean():.1f}%")
    ground = fill_holes_masked(zlow, valid & ~covered)
    ground = despeckle(ground)
    relief = np.where(np.isfinite(zhigh) & np.isfinite(ground),
                      zhigh - ground, 0.0)

    # --- 2. land cover -----------------------------------------------------
    osm = {"buildings": [], "roads": [], "water_polys": [], "water_lines": []}
    if args.osm and os.path.exists(args.osm):
        with open(args.osm) as f:
            osm = json.load(f)
    else:
        print("WARNING: no OSM file - buildings/water masks will be empty")
    building = rasterize_polys(osm["buildings"], t, (h, w)) & valid
    water = (rasterize_polys(osm["water_polys"], t, (h, w)) |
             rasterize_lines(osm["water_lines"], t, (h, w), buffer_m=8.0)) & valid
    # Sea / harbor basins / flooded excavations: OSM maps the sea as a
    # coastline (no polygons), but Z is ellipsoidal with the sea surface at
    # ~+26 m, and nothing in this city stands below that + margin. The
    # photogrammetric "bottom" of enclosed water bodies is noise (down to
    # -17 m) - unmasked, they swallow runoff 39 m deep and crush the CFL dt.
    below = valid & (ground < args.sea_level + args.sea_margin)
    sea = ndimage.binary_fill_holes(
        ndimage.binary_closing(below, iterations=2)) & valid
    print(f"sea/low-water mask: {below.sum()} cells below "
          f"{args.sea_level + args.sea_margin:.1f} m -> {sea.sum()} after closing")
    water |= sea
    water &= ~building

    s = rgb.astype(np.float32).sum(axis=2) + 1e-6
    rc, gc, bc = (rgb[..., i].astype(np.float32) / s for i in range(3))
    exg = 2 * gc - rc - bc
    if args.exg == "otsu":
        thr = float(otsu_threshold(exg[valid & (s > 30)]))
        thr = float(np.clip(thr, 0.03, 0.15))
    else:
        thr = float(args.exg)
    print(f"ExG threshold {thr:.3f}")
    veg = (exg > thr) & valid
    canopy = veg & (relief > args.canopy_relief) & ~building
    grass = veg & ~canopy & ~building & ~water
    mx = rgb.max(axis=2).astype(np.float32)
    mn = rgb.min(axis=2).astype(np.float32)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)
    soil = (~veg & ~building & ~water & valid & (sat > 0.12) &
            (rgb[..., 0] > rgb[..., 1]) & (rgb[..., 1] > rgb[..., 2]))

    landcover = np.full((h, w), PAVED, dtype=np.uint8)
    landcover[soil] = SOIL
    landcover[grass] = GRASS
    landcover[canopy] = CANOPY
    landcover[water] = WATER
    landcover[building] = BUILDING
    landcover[~valid] = PAVED
    for ci, name in enumerate(CLASS_NAMES):
        print(f"  {name:9s} {100 * ((landcover == ci) & valid).sum() / valid.sum():5.1f}%")

    # --- 3. hydraulic surface ----------------------------------------------
    dem = ground.copy()
    ground_src = valid & ~building & ~canopy
    dem = fill_holes_masked(dem, canopy, sources=ground_src)
    interp_frac = canopy.sum() / valid.sum()
    print(f"canopy removed & re-interpolated: {100 * interp_frac:.1f}% of valid cells")

    if args.cuts and os.path.exists(args.cuts):
        from matplotlib.path import Path as MplPath
        with open(args.cuts) as f:
            cuts = json.load(f)["polygons"]
        cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        ux = t["minx"] + cols * res
        uy = t["maxy"] - rows * res
        pts = np.column_stack([ux.ravel(), uy.ravel()])
        for poly in cuts:
            m = MplPath(np.asarray(poly)).contains_points(pts).reshape(h, w)
            dem = fill_holes_masked(dem, m, sources=valid & ~m & ~building)
            print(f"  cut applied: {m.sum()} cells")

    # --- 4. courtyards -----------------------------------------------------
    walkable = valid & ~building
    lbl, nlbl = ndimage.label(walkable)
    frame = np.zeros((h, w), dtype=bool)
    frame[0, :] = frame[-1, :] = frame[:, 0] = frame[:, -1] = True
    open_cells = walkable & (frame | ndimage.binary_dilation(~valid))
    open_labels = np.unique(lbl[open_cells])
    courtyard = walkable & ~np.isin(lbl, open_labels)
    print(f"courtyards: {courtyard.sum()} cells "
          f"({100 * courtyard.sum() / valid.sum():.2f}% of valid)")

    # --- 5. rain rerouting (downspout model) -------------------------------
    eligible = walkable & ~courtyard & ~water
    rain_weight = eligible.astype(np.float32)
    src = (building | courtyard) & valid
    if src.any() and eligible.any():
        _, (ir, ic) = ndimage.distance_transform_edt(~eligible, return_indices=True)
        dst = ir[src].astype(np.int64) * w + ic[src].astype(np.int64)
        np.add.at(rain_weight.reshape(-1), dst, 1.0)
    print(f"rain rerouted from {src.sum()} roof/courtyard cells; "
          f"total rain cells {rain_weight.sum():.0f} "
          f"(= {int((valid & ~water).sum())} expected)")

    # --- 6. parameter rasters ----------------------------------------------
    manning = np.full((h, w), MANNING[PAVED], dtype=np.float32)
    infil = np.zeros((h, w), dtype=np.float32)
    for ci in range(6):
        manning[landcover == ci] = MANNING[ci]
        infil[landcover == ci] = INFIL_MMH[ci]

    # --- write core outputs -------------------------------------------------
    np.save(os.path.join(out, "dem.npy"),
            np.where(valid, dem, np.nan).astype(np.float32))
    Image.fromarray(rgb).save(os.path.join(out, "ortho.png"))
    save_transform(os.path.join(out, "dem_transform.json"),
                   t["minx"], t["miny"], res, w, h,
                   extra={"stack": os.path.abspath(args.stack)})
    np.save(os.path.join(out, "landcover.npy"), landcover)
    np.save(os.path.join(out, "manning.npy"), manning)
    np.save(os.path.join(out, "infil_mmh.npy"), infil)
    np.save(os.path.join(out, "rain_weight.npy"), rain_weight)
    np.savez_compressed(os.path.join(out, "masks.npz"),
                        valid=valid, building=building, courtyard=courtyard,
                        water=water, eligible=eligible)

    # --- 7. static fill bound (GPU) -----------------------------------------
    if not args.no_fillbound:
        print("fill bound (GPU morphological reconstruction)...")
        dem_f = np.where(valid, dem, np.nan)
        filled = fill_bound_gpu(np.nan_to_num(dem_f, nan=0.0), valid, water=water)
        fb = np.clip(filled - dem, 0, None)
        fb[~valid] = np.nan
        np.save(os.path.join(out, "fillbound.npy"), fb.astype(np.float32))
        print(f"  max static ponding depth {np.nanmax(fb):.2f} m, "
              f"p99 {np.nanpercentile(fb[valid], 99):.2f} m")

    # --- 8. gauges -----------------------------------------------------------
    gauges = auto_gauges(np.where(valid, dem, np.nan), valid, t)
    with open(os.path.join(out, "gauges.json"), "w") as f:
        json.dump(gauges, f, indent=2)
    print(f"gauges: {[g['name'] for g in gauges]}")

    # --- QA figures -----------------------------------------------------------
    hs = (hillshade(np.where(valid, dem, np.nanmin(dem)), res) * 255).astype(np.uint8)
    Image.fromarray(hs).save(os.path.join(out, "hillshade.png"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#bdbdbd", "#e05555", "#1a7a2e",
                           "#8fd06e", "#c8a44b", "#3d8bd4"])
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(rgb)
    lc = np.ma.masked_where(~valid, landcover)
    ax.imshow(lc, cmap=cmap, vmin=-0.5, vmax=5.5, alpha=0.45, interpolation="nearest")
    for g in gauges:
        px, py = utm_to_pixel(t, g["x"], g["y"])
        ax.plot(px, py, "wo", ms=8, mec="k")
        ax.annotate(g["name"], (px, py), color="w", fontsize=11,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_title(f"land cover ({' / '.join(CLASS_NAMES)}) + gauges")
    ax.axis("off")
    fig.savefig(os.path.join(out, "landcover_qa.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(rgb)
    ax.imshow(np.ma.masked_where(~courtyard, np.ones_like(landcover)),
              cmap="autumn", alpha=0.8, interpolation="nearest")
    ax.set_title("enclosed courtyards (rain rerouted to nearest street)")
    ax.axis("off")
    fig.savefig(os.path.join(out, "courtyards_qa.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    if args.geotiff:
        import rasterio
        from affine import Affine
        tr = Affine(res, 0, t["minx"], 0, -res, t["maxy"])
        for name, arr, dtype in [("dem", dem, "float32"),
                                 ("landcover", landcover, "uint8"),
                                 ("manning", manning, "float32"),
                                 ("infil_mmh", infil, "float32")]:
            with rasterio.open(
                    os.path.join(out, f"{name}.tif"), "w", driver="GTiff",
                    height=h, width=w, count=1, dtype=dtype, crs="EPSG:32636",
                    transform=tr, compress="deflate", tiled=True) as dst:
                dst.write(arr.astype(dtype), 1)
        print("wrote GeoTIFFs")

    print(f"terrain ready in {out}/")


if __name__ == "__main__":
    main()
