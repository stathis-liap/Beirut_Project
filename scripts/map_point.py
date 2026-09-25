"""Pick any point on the full 2.95 x 2.1 km map and crop a local DEM
around it, for the SPH stairs solver (particle_sim.py --scene map_point)
to run on - generalizes what real_stairs.py did for one hardcoded
location (Saint Nicolas Stairs) to anywhere the 5cm DSM covers.

Two pieces:
  - build_thumbnail(): a cached low-res RGB preview of the whole map (for
    a GUI click-to-pick widget - the full rasters are multi-GB, far too
    big to display directly) plus its pixel<->UTM transform.
  - crop_local(): a windowed DSM read around one UTM point, cleaned up
    (canopy/noise rejected, buildings auto-detected as flow obstacles),
    with an automatically-placed water source at the highest walkable
    point in the crop.

Usage (standalone):
  python scripts/map_point.py thumbnail
  python scripts/map_point.py crop --x 732685.33 --y 3753131.31 --size 30
"""

import argparse
import json
import os

import numpy as np
from scipy import ndimage

import map_data

THUMB_DIR = os.path.join(map_data.PROJECT_ROOT, "terrain", "map_overview")
THUMB_PNG = os.path.join(THUMB_DIR, "overview.png")
THUMB_META = os.path.join(THUMB_DIR, "overview_meta.json")
THUMB_MAX_DIM = 1800

CROP_SIZE_M = 30.0          # default crop footprint, meters
CANOPY_FOOTPRINT_M = 0.5     # small-window low-percentile "ground snap"
CANOPY_PERCENTILE = 15
BUILDING_WINDOW_M = 20.0     # local-min filter footprint for obstacle detection -
                              # must be bigger than a typical building footprint,
                              # or the filter can't "see" street level nearby
BUILDING_MARGIN_M = 2.5      # cell counts as a building if this much above local min
BUILDING_WALL_BOOST_M = 6.0  # extra height added to detected obstacles


def build_thumbnail(force=False):
    if os.path.exists(THUMB_PNG) and os.path.exists(THUMB_META) and not force:
        with open(THUMB_META) as f:
            return THUMB_PNG, json.load(f)

    map_data.require_map_data()
    import rasterio
    from rasterio.enums import Resampling
    from PIL import Image

    os.makedirs(THUMB_DIR, exist_ok=True)
    with rasterio.open(map_data.RGB_PATH) as src:
        scale = THUMB_MAX_DIM / max(src.width, src.height)
        out_h, out_w = max(1, int(src.height * scale)), max(1, int(src.width * scale))
        data = src.read([1, 2, 3], out_shape=(3, out_h, out_w), resampling=Resampling.average)
        t = src.transform

    img = np.transpose(data, (1, 2, 0)).astype(np.uint8)
    Image.fromarray(img).save(THUMB_PNG)
    meta = {
        "minx": t.c, "maxy": t.f,
        "res_x": src.res[0] / scale, "res_y": src.res[1] / scale,
        "width": out_w, "height": out_h, "crs": "EPSG:32636",
    }
    with open(THUMB_META, "w") as f:
        json.dump(meta, f, indent=2)
    return THUMB_PNG, meta


def thumb_pixel_to_utm(meta, px, py):
    x = meta["minx"] + px * meta["res_x"]
    y = meta["maxy"] - py * meta["res_y"]
    return x, y


def utm_to_thumb_pixel(meta, x, y):
    px = (x - meta["minx"]) / meta["res_x"]
    py = (meta["maxy"] - y) / meta["res_y"]
    return px, py


def _mask_corrupt_band(dsm, minx, maxy, native_res):
    """Replaces the salvaged DSM's damaged LZW strip with interpolated
    values. The band is ~95 rows (~4.75 m) of garbage spanning the full
    raster width - narrow enough to bridge from the clean rows on either
    side, which beats leaving NaNs that would propagate through the
    percentile/minimum filters below and blow a hole in the crop.
    """
    band = map_data.corrupt_utm_y_band()
    if band is None:
        return dsm
    y_lo, y_hi = band
    h = dsm.shape[0]
    rows = np.arange(h)
    row_y_top = maxy - rows * native_res
    row_y_bot = row_y_top - native_res
    bad = (row_y_bot <= y_hi) & (row_y_top >= y_lo)
    if not bad.any():
        return dsm

    good = ~bad
    if not good.any():
        raise ValueError(
            f"({minx:.1f}, {maxy:.1f}) crop falls entirely inside the corrupt "
            f"DSM band (UTM y {y_lo}-{y_hi}); pick a point away from it")
    print(f"  masking {int(bad.sum())} corrupt DSM row(s) (UTM y {y_lo}-{y_hi})")
    dsm = dsm.copy()
    for col in range(dsm.shape[1]):
        dsm[bad, col] = np.interp(rows[bad], rows[good], dsm[good, col])
    return dsm


def crop_local(utm_x, utm_y, size_m=CROP_SIZE_M, res=map_data.MAP_RES_M):
    """Windowed DSM crop around (utm_x, utm_y). Rejects tree-canopy/noise
    spikes with a small local low-percentile filter (same idea as
    real_stairs.py's profile extraction, generalized to 2D) and flags
    buildings as flow obstacles via a local-minimum-filter height check
    (same trick as find_real_stairs.py's street_mask).

    Returns (dem, obstacle_mask, transform_dict).
    """
    map_data.require_map_data()
    import rasterio
    from rasterio.windows import Window

    half = size_m / 2
    with rasterio.open(map_data.DSM_PATH) as src:
        row0, col0 = src.index(utm_x - half, utm_y + half)
        row1, col1 = src.index(utm_x + half, utm_y - half)
        row0, col0 = max(0, row0), max(0, col0)
        row1, col1 = min(src.height, row1), min(src.width, col1)
        if row1 <= row0 or col1 <= col0:
            raise ValueError(f"({utm_x}, {utm_y}) falls outside the map coverage")
        win = Window(col0, row0, col1 - col0, row1 - row0)
        dsm = src.read(1, window=win).astype(np.float64)
        win_transform = src.window_transform(win)
    minx, maxy = win_transform.c, win_transform.f

    dsm = _mask_corrupt_band(dsm, minx, maxy, native_res=map_data.MAP_RES_M)

    native_res = map_data.MAP_RES_M
    if abs(res - native_res) > 1e-6:
        dsm = ndimage.zoom(dsm, native_res / res, order=1)

    footprint = max(3, int(round(CANOPY_FOOTPRINT_M / res)))
    ground = ndimage.percentile_filter(dsm, CANOPY_PERCENTILE, size=footprint)

    roof_window = max(5, int(round(BUILDING_WINDOW_M / res)))
    local_min = ndimage.minimum_filter(ground, size=roof_window)
    obstacle = ground > (local_min + BUILDING_MARGIN_M)

    dem = ground.copy()
    dem[obstacle] += BUILDING_WALL_BOOST_M

    h, w = dem.shape
    transform = {"minx": minx, "maxy": maxy, "res": res, "width": w, "height": h}
    return dem, obstacle, transform


def find_source(dem, obstacle, res, margin_m=2.0):
    """A high walkable cell, a bit in from the crop edge, as the water
    source - plus a small downhill-pointing initial velocity computed from
    the local gradient there, so emitted particles start moving the right
    way regardless of the crop's orientation.

    Ranks by a *smoothed* copy of the DEM, not the raw cell max: a single
    unfiltered spike (a rooftop AC unit, railing post, whatever survived
    the canopy filter in crop_local) is narrower than genuine high ground,
    so smoothing over ~1m washes it out before ranking - picking the raw
    max here previously placed the source on top of such spikes, dropping
    water from several meters above the real surface.
    """
    walkable = ~obstacle
    h, w = dem.shape
    margin = max(1, int(round(margin_m / res)))
    interior = np.zeros_like(walkable)
    interior[margin:max(margin + 1, h - margin), margin:max(margin + 1, w - margin)] = True
    candidates = walkable & interior
    if not candidates.any():
        candidates = walkable
    if not candidates.any():
        raise ValueError("no walkable (non-obstacle) cells in this crop")

    smooth_footprint = max(3, int(round(1.0 / res)))
    smoothed = ndimage.uniform_filter(dem, size=smooth_footprint)

    target = np.percentile(smoothed[candidates], 95)
    high_enough = candidates & (smoothed >= target)
    masked = np.where(high_enough, smoothed, np.inf)
    row, col = np.unravel_index(np.argmin(masked), dem.shape)  # closest to target from above

    r0, r1 = max(0, row - 1), min(h - 1, row + 1)
    c0, c1 = max(0, col - 1), min(w - 1, col + 1)
    dzdx = (dem[row, c1] - dem[row, c0]) / max((c1 - c0) * res, 1e-6)
    dzdy = (dem[r1, col] - dem[r0, col]) / max((r1 - r0) * res, 1e-6)
    grad = np.array([-dzdx, -dzdy])
    norm = np.linalg.norm(grad)
    direction = grad / norm if norm > 1e-6 else np.array([0.0, 1.0])
    speed = 0.1
    vx, vy = float(direction[0] * speed), float(direction[1] * speed)

    x, y = col * res, row * res
    z = float(dem[row, col]) + 0.05
    return x, y, z, vx, vy


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("thumbnail")

    c = sub.add_parser("crop")
    c.add_argument("--x", type=float, required=True)
    c.add_argument("--y", type=float, required=True)
    c.add_argument("--size", type=float, default=CROP_SIZE_M)

    args = ap.parse_args()
    if args.cmd == "thumbnail":
        path, meta = build_thumbnail(force=True)
        print(f"wrote {path}\n{json.dumps(meta, indent=2)}")
    elif args.cmd == "crop":
        dem, obstacle, t = crop_local(args.x, args.y, size_m=args.size)
        source = find_source(dem, obstacle, t["res"])
        print(f"dem shape {dem.shape}, elev range {dem.min():.2f}-{dem.max():.2f} m, "
              f"{100*obstacle.mean():.1f}% flagged as obstacles")
        print(f"source (x,y,z,vx,vy) = {tuple(round(v, 3) for v in source)}")


if __name__ == "__main__":
    main()
