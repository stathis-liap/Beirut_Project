#!/usr/bin/env python3
"""Build the Al-Masar green-corridor green-infrastructure representation
from the official AUB ArcGIS geometry, with distinct materials.

Sources (already reprojected to EPSG:32636):
  output/masar_zone_official.json      - the 19 ha Green Path zone polygon
  output/masar_fb_highway_official.json - Fouad Boutros ROW linework

Approach (confirmed with the client 2026-07-23):
  * the ZONE polygon is the corridor footprint / extent;
  * a SPINE is derived down the zone (its per-northing medial column,
    smoothed) as the Fouad Boutros right-of-way centreline;
  * transverse MATERIAL BANDS follow the design cross-section
    (2 m porous sidewalk | 1.5 m porous bikelane | 1.5 m bioswale |
     3 m vehicular lane | 1.5 m bikelane | 2 m sidewalk), i.e. an 11.5 m
     street ribbon centred on the spine;
  * the zone area OUTSIDE the street ribbon is green nodes -
     bioretention ponds / rain gardens (placed at terrain low points, where
     water accumulates), terraces (on steep spine segments) and gardens
     (the rest).

Writes a material-class raster on the given terrain grid + a QA overlay.
This script only defines geometry+materials; it does NOT run the sim.
"""
import argparse, json, os, sys
import numpy as np
from matplotlib.path import Path as MplPath
sys.path.insert(0, os.path.dirname(__file__))
from las_common import load_transform, utm_to_pixel

# material classes (id -> label)
MAT = {0:"(none)", 1:"vehicular lane", 2:"porous bikelane", 3:"bioswale",
       4:"porous sidewalk", 5:"garden", 6:"bioretention pond / rain garden",
       7:"terrace"}
# half-width (m) of each transverse band from the spine, cumulative
BANDS = [(1.5, 1),   # 0.0-1.5  vehicular (3 m lane, +/-1.5)
         (3.0, 2),   # 1.5-3.0  bikelane
         (4.5, 3),   # 3.0-4.5  bioswale
         (6.5, 4)]   # 4.5-6.5  sidewalk  (=> 13 m street ribbon)

def zone_rings(path):
    return json.load(open(path))["features"][0]["geometry"]["rings"]

def rasterize(rings, t, shape):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    mask = np.zeros(h*w, bool)
    for r in rings:
        p = np.asarray(r)
        px, py = utm_to_pixel(t, p[:,0], p[:,1])
        inside = MplPath(np.column_stack([px, py])).contains_points(pts)
        mask ^= inside   # xor handles holes
    return mask.reshape(h, w)

def spine_from_zone(zone_mask, res, smooth_m=25.0):
    """Per-row median column of the zone => a N-S centreline, smoothed."""
    h, w = zone_mask.shape
    rows, cols = [], []
    for r in range(h):
        xs = np.nonzero(zone_mask[r])[0]
        if len(xs) >= 2:
            rows.append(r); cols.append(np.median(xs))
    rows = np.array(rows); cols = np.array(cols, float)
    k = max(3, int(smooth_m/res) | 1)
    pad = np.pad(cols, k//2, mode="edge")
    cols_s = np.convolve(pad, np.ones(k)/k, "valid")
    return rows, cols_s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default="output/terrain_1.0")
    ap.add_argument("--zone", default="output/masar_zone_official.json")
    ap.add_argument("--out", default="output/corridor_gi")
    ap.add_argument("--row-halfwidth", type=float, default=18.0,
                    help="half-width (m) of the green right-of-way ribbon "
                         "around the spine; the study ZONE is much wider "
                         "(surrounding blocks) and is not all permeable")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t = load_transform(os.path.join(args.terrain,"dem_transform.json"))
    res = t["res"]
    dem = np.load(os.path.join(args.terrain,"dem.npy"))
    masks = np.load(os.path.join(args.terrain,"masks.npz"))
    fillbound = None
    fb_path = os.path.join(args.terrain,"fillbound.npy")
    if os.path.exists(fb_path):
        fillbound = np.load(fb_path)
    h, w = dem.shape

    zone = zone_rings(args.zone)
    zmask = rasterize(zone, t, (h, w))
    building = masks["building"]; valid = masks["valid"]
    # buildings inside the zone stay as buildings (not permeable)
    zmask &= valid
    print(f"zone cells (valid): {zmask.sum()} = {zmask.sum()*res*res/1e4:.1f} ha")

    rows, cols = spine_from_zone(zmask, res)
    spine = np.zeros((h, w), bool)
    rr = np.clip(rows.astype(int), 0, h-1)
    cc = np.clip(cols.astype(int), 0, w-1)
    spine[rr, cc] = True

    from scipy import ndimage
    dist = ndimage.distance_transform_edt(~spine) * res   # m from spine
    # longitudinal slope of the spine (for terraces)
    slope = np.hypot(*np.gradient(np.where(valid, dem, np.nan)))
    slope = np.nan_to_num(slope)

    mat = np.zeros((h, w), np.uint8)
    # The green infrastructure occupies the right-of-way ribbon around the
    # spine, NOT the full study zone (which spans surrounding blocks).
    ribbon = zmask & (~building) & (dist <= args.row_halfwidth)
    print(f"ROW ribbon (<= {args.row_halfwidth:.0f} m from spine): "
          f"{ribbon.sum()*res*res/1e4:.2f} ha of the {zmask.sum()*res*res/1e4:.1f} ha zone")
    # transverse street bands
    prev = 0.0
    for hw_, cls in BANDS:
        band = ribbon & (dist > prev) & (dist <= hw_)
        mat[band] = cls
        prev = hw_
    # ribbon area beyond the street cross-section = flanking green nodes
    green = ribbon & (dist > BANDS[-1][0])
    mat[green] = 5   # garden default
    # bioretention ponds: low points within the green area (where water sits)
    if fillbound is not None:
        pond_depth = np.where(green, fillbound, 0)
        thr = np.percentile(pond_depth[green & (pond_depth>0)], 80) if (green&(pond_depth>0)).any() else 0.3
        mat[green & (pond_depth >= max(thr,0.25))] = 6
    # terraces: green cells on the steep spine segments
    steep = green & (slope > np.percentile(slope[green] if green.any() else [0], 85))
    mat[steep & (mat==5)] = 7

    np.save(os.path.join(args.out,"material.npy"), mat)
    counts = {MAT[i]: int((mat==i).sum()) for i in MAT if (mat==i).any()}
    areas  = {k: round(v*res*res) for k,v in counts.items()}
    json.dump({"class_labels":MAT, "cell_area_m2":res*res, "areas_m2":areas},
              open(os.path.join(args.out,"material_summary.json"),"w"), indent=2)
    print("material areas (m2):")
    for k,v in areas.items(): print(f"  {k:32} {v:8d}")
    # save spine polyline in UTM
    sx = t["minx"] + cc*res + res/2
    sy = t["maxy"] - rr*res - res/2
    json.dump({"crs":"EPSG:32636","spine":[[float(a),float(b)] for a,b in zip(sx,sy)]},
              open(os.path.join(args.out,"spine.json"),"w"))
    print(f"wrote {args.out}/material.npy, spine.json, material_summary.json")

if __name__ == "__main__":
    main()
