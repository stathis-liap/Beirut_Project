#!/usr/bin/env python3
"""Turns a Rhino .3dm CAD model into the same (dem.npy, ortho.png,
dem_transform.json) triplet reconstruct_area.py/build_dem.py produce from a
.las point cloud - every downstream consumer (flood_sim.py, particle_sim.py,
render_3d.py) only ever reads that triplet, so once it exists here the CAD
model is a drop-in terrain source, no changes needed anywhere else.

Only .3dm is supported. The other files in 3D/ can't be read the same way:
  - .dwg needs Autodesk's ODA File Converter (a separate, manually-installed
    GUI/CLI tool - not pip-installable, not something to silently fetch and
    run) to get to DXF first; ezdxf reads DXF but not binary DWG directly.
  - .skp needs Trimble's SketchUp C++ SDK, which has no official or
    functional unofficial Python binding.
  Workaround for either: export to DXF or OBJ from the tool that made them -
  both of those ARE readable here (DXF via ezdxf, OBJ via trimesh).

Coordinate system: this model's file metadata claims centimeters, but the
raw coordinate span (~226 x 516 units) only makes sense as a real site if
those units are actually meters - a very common CAD unit-tag/scale mismatch.
Defaults to treating 1 model unit = 1 meter (--unit-scale to override). This
is NOT georeferenced to real UTM (the model is built near a local origin,
not at Beirut's real coordinates) - it lands in the same "local scene"
category as particle_sim.py's synthetic/real_stairs scenes, not the
UTM-addressable las/map_point ones. See cad_model scene in particle_sim.py.

Usage:
  python scripts/load_cad_model.py --model "3D/3D Site.3dm" --res 0.5 \
      --out output/cad_site
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
from build_dem import fill_holes, despeckle, hillshade
from las_common import save_transform

BUILDING_LAYER_HINTS = ("bldg", "building", "struct")  # case-insensitive
                                                        # substring match -
                                                        # covers this
                                                        # project's "G-Bldgs"
                                                        # without hardcoding
                                                        # one exact name


MAX_BLOCK_DEPTH = 8   # blocks nest; this stops a self-referencing definition
                       # (rare but survivable in a working file) from looping


def _xform_matrix(xf):
    return np.array([[xf.M00, xf.M01, xf.M02, xf.M03],
                     [xf.M10, xf.M11, xf.M12, xf.M13],
                     [xf.M20, xf.M21, xf.M22, xf.M23],
                     [xf.M30, xf.M31, xf.M32, xf.M33]], dtype=np.float64)


def _mesh_vertices(mesh, xform):
    n = len(mesh.Vertices)
    if n == 0:
        return None
    a = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        v = mesh.Vertices[i]
        a[i, 0], a[i, 1], a[i, 2] = v.X, v.Y, v.Z
    if xform is not None:
        a = a @ xform[:3, :3].T + xform[:3, 3]
    return a


def geometry_points(model, geo, xform, stats, depth):
    """Yields (N,3) arrays of world-space vertices for one object.

    Surfaces are tessellated with openNURBS's own mesher (the one Rhino uses
    for the viewport), and InstanceReferences - CAD block instances - are
    resolved against the file's InstanceDefinition table and walked
    recursively with their transform composed in. Skipping blocks is not
    viable on a real architectural file: this project's "3D Design.3dm"
    carries 16,766 of them over 4,419 definitions, so ignoring them drops
    most of the slabs and building components and leaves a DEM with nothing
    in it. Curves/text/hatches carry no surface and are still skipped.
    """
    import rhino3dm as r3d

    tname = type(geo).__name__
    meshes = []

    if tname == "Mesh":
        meshes = [geo]
    elif tname == "Brep":
        meshes = [f.GetMesh(r3d.MeshType.Any) for f in geo.Faces]
    elif tname == "Extrusion":
        brep = geo.ToBrep(True)
        if brep is not None:
            meshes = [f.GetMesh(r3d.MeshType.Any) for f in brep.Faces]
    elif tname.endswith("Surface") or tname == "Surface":
        brep = geo.ToBrep() if hasattr(geo, "ToBrep") else None
        if brep is not None:
            meshes = [f.GetMesh(r3d.MeshType.Any) for f in brep.Faces]
    elif tname == "InstanceReference" and depth < MAX_BLOCK_DEPTH:
        idef = model.InstanceDefinitions.FindId(geo.ParentIdefId)
        if idef is not None:
            child = _xform_matrix(geo.Xform)
            if xform is not None:
                child = xform @ child
            stats["blocks"] += 1
            for mid in idef.GetObjectIds():
                member = model.Objects.FindId(mid)
                if member is not None:
                    yield from geometry_points(model, member.Geometry, child,
                                               stats, depth + 1)
        return

    for m in meshes:
        if m is None:
            continue
        a = _mesh_vertices(m, xform)
        if a is not None:
            stats["faces"] += 1
            yield a


def extract_points(path):
    """Tessellates every surface in the .3dm into a flat (N,3) point array,
    tagged with a parallel is_building bool array from each object's layer
    name. Breps (the vast majority of objects in these files) have no
    ready-made vertex list - GetMesh() on each BrepFace is openNURBS's own
    tessellator, the same one Rhino uses for the viewport, not something
    this script reimplements.
    """
    import rhino3dm as r3d

    print(f"opening {path} ...")
    t0 = time.time()
    model = r3d.File3dm.Read(path)
    if model is None:
        sys.exit(f"rhino3dm couldn't read {path} (corrupt, or not a .3dm file)")
    print(f"  read in {time.time() - t0:.1f}s, {len(model.Objects)} objects, "
         f"{len(model.Layers)} layers, unit system {model.Settings.ModelUnitSystem}")

    layer_is_building = {}
    for i, layer in enumerate(model.Layers):
        name = (layer.Name or "").lower()
        layer_is_building[i] = any(h in name for h in BUILDING_LAYER_HINTS)
    print("  layers:", {model.Layers[i].Name: layer_is_building[i] for i in layer_is_building})

    chunks, bldg_flags = [], []
    n_obj = len(model.Objects)
    stats = {"faces": 0, "blocks": 0}
    for oi, obj in enumerate(model.Objects):
        building = layer_is_building.get(obj.Attributes.LayerIndex, False)
        for arr in geometry_points(model, obj.Geometry, None, stats, 0):
            chunks.append(arr)
            bldg_flags.append(np.full(len(arr), building, dtype=bool))

        if oi % 2000 == 0:
            so_far = sum(len(c) for c in chunks)
            print(f"\r  tessellating: {100 * oi / n_obj:5.1f}%  "
                 f"({so_far:,} verts so far)", end="", flush=True)

    if not chunks:
        return (np.empty(0), np.empty(0), np.empty(0), np.empty(0, dtype=bool))

    pts = np.concatenate(chunks)
    is_bldg = np.concatenate(bldg_flags)
    print(f"\r  tessellating: 100.0%  ({len(pts):,} verts from {stats['faces']:,} "
         f"faces, {stats['blocks']:,} block instances expanded)")
    return pts[:, 0], pts[:, 1], pts[:, 2], is_bldg


TOP_PERCENTILE = 92.0  # per cell, not a strict max - a detailed architectural
                       # model tessellates into thousands of small features
                       # (railings, mullions, trim) that are only a handful
                       # of points each; np.maximum.at picks up every single
                       # one of those as a full-height spike, which reads as
                       # jagged noise rather than a building. A high
                       # percentile needs a real cluster of points near the
                       # top of a cell to register - one stray railing post
                       # among hundreds of roof/wall points gets outvoted,
                       # same idea as map_point.py's percentile-based canopy
                       # rejection, just aimed at CAD tessellation noise
                       # instead of LIDAR canopy noise.


MAX_GRID_CELLS = 200_000_000   # ~800 MB as float32, already far past any
                                # plausible site. A CAD file whose units
                                # don't match --unit-scale (a millimeter
                                # model read as meters is 1000x too wide, and
                                # one that mixes local design geometry with
                                # GIS layers at real UTM coordinates is worse)
                                # produces a coordinate span that turns into
                                # billions of cells here. Without this check
                                # the first thing that happens is numpy being
                                # asked for tens of GB, which on a machine
                                # with a tight commit charge takes far more
                                # down with it than this one script.


DENSITY_BINS = 512         # histogram resolution used to locate the built-up core
DENSITY_TARGET = 0.95      # keep the densest bins covering this much of the
                            # geometry; the rest is stray/scattered. Not higher:
                            # on this project's design file the last few percent
                            # of vertices are themselves spread over the full
                            # 220,000-unit extent, so a 0.99 target re-admits
                            # every outlier it was meant to exclude. Clean
                            # models are protected by DENSITY_SPAN_RATIO
                            # instead of by a timid target.
DENSITY_PASSES = 3         # re-bin inside the box found by the previous pass.
                            # One pass over a hugely outlier-stretched extent
                            # only resolves the core to within a bin (a bin is
                            # 428 units wide when the raw span is 220,000), so
                            # the box comes back far looser than the real site;
                            # each further pass bins the survivors and tightens.
DENSITY_SPAN_RATIO = 1.5   # only override the raw bbox when it's at least
                            # this much bigger than the dense core, so a
                            # genuinely wide, evenly-covered site is left alone
DENSITY_MARGIN = 0.02


def robust_bounds(x, y):
    """(minx, maxx, miny, maxy, keep_mask) for the grid, ignoring stray
    geometry far from the built-up area.

    Working CAD files routinely carry geometry nowhere near the site - a
    line left at the old origin, annotation dragged off into space, blocks
    placed against a georeferenced import's coordinates. min/max is
    maximally sensitive to exactly that: in this project's "3D Design.3dm"
    the real site is a few hundred units across, while strays stretch the
    raw bbox past 200,000 - which then reads as "the units must be wrong"
    and blows up the grid allocation.

    A percentile trim was tried first and isn't enough: once block
    instances are expanded there are tens of thousands of scattered points,
    far more than any fixed percentile clips. So the extent is chosen by
    DENSITY instead - bin coarsely, keep the fullest bins until they account
    for DENSITY_TARGET of all vertices, and take their bounding box. That
    holds up regardless of how many outliers there are, as long as they're
    sparse relative to the site, which is what "stray" means.
    """
    raw = (x.min(), x.max(), y.min(), y.max())
    raw_x, raw_y = raw[1] - raw[0], raw[3] - raw[2]
    full = np.ones(len(x), dtype=bool)
    if raw_x <= 0 or raw_y <= 0:
        return (*raw, full)

    lo_x, hi_x, lo_y, hi_y = raw
    sel = full
    for _ in range(DENSITY_PASSES):
        span_x, span_y = hi_x - lo_x, hi_y - lo_y
        if span_x <= 0 or span_y <= 0 or sel.sum() == 0:
            break
        sx, sy = x[sel], y[sel]
        ix = np.clip(((sx - lo_x) / span_x * DENSITY_BINS).astype(np.int32), 0, DENSITY_BINS - 1)
        iy = np.clip(((sy - lo_y) / span_y * DENSITY_BINS).astype(np.int32), 0, DENSITY_BINS - 1)
        counts = np.bincount(iy.astype(np.int64) * DENSITY_BINS + ix,
                             minlength=DENSITY_BINS * DENSITY_BINS)

        order = np.argsort(counts)[::-1]
        keep_bins = order[np.cumsum(counts[order]) <= DENSITY_TARGET * len(sx)]
        if len(keep_bins) == 0:
            keep_bins = order[:1]

        by, bx = keep_bins // DENSITY_BINS, keep_bins % DENSITY_BINS
        n_lo_x = lo_x + bx.min() * span_x / DENSITY_BINS
        n_hi_x = lo_x + (bx.max() + 1) * span_x / DENSITY_BINS
        n_lo_y = lo_y + by.min() * span_y / DENSITY_BINS
        n_hi_y = lo_y + (by.max() + 1) * span_y / DENSITY_BINS
        if (n_hi_x - n_lo_x) >= span_x * 0.98 and (n_hi_y - n_lo_y) >= span_y * 0.98:
            break  # already tight, further passes would only nibble real edges
        lo_x, hi_x, lo_y, hi_y = n_lo_x, n_hi_x, n_lo_y, n_hi_y
        sel = (x >= lo_x) & (x <= hi_x) & (y >= lo_y) & (y <= hi_y)

    core_x, core_y = hi_x - lo_x, hi_y - lo_y
    if core_x <= 0 or core_y <= 0 or (raw_x < DENSITY_SPAN_RATIO * core_x and
                                      raw_y < DENSITY_SPAN_RATIO * core_y):
        return (*raw, full)

    lo_x -= core_x * DENSITY_MARGIN; hi_x += core_x * DENSITY_MARGIN
    lo_y -= core_y * DENSITY_MARGIN; hi_y += core_y * DENSITY_MARGIN
    keep = (x >= lo_x) & (x <= hi_x) & (y >= lo_y) & (y <= hi_y)
    print(f"  raw bbox {raw_x:,.0f} x {raw_y:,.0f} units is outlier-driven; "
          f"using the dense core {hi_x - lo_x:,.0f} x {hi_y - lo_y:,.0f} "
          f"({100 * keep.mean():.2f}% of verts, {(~keep).sum():,} strays dropped)")
    return lo_x, hi_x, lo_y, hi_y, keep


def points_to_dem(x, y, z, is_building, res):
    """Bins points per grid cell (same idea as reconstruct_area.grid_local)
    and takes a high percentile of each cell's z-values - see
    TOP_PERCENTILE for why not a strict max.
    """
    import pandas as pd

    minx, maxx, miny, maxy, keep = robust_bounds(x, y)
    if not keep.all():
        x, y, z, is_building = x[keep], y[keep], z[keep], is_building[keep]

    w = int(np.ceil((maxx - minx) / res)) + 1
    h = int(np.ceil((maxy - miny) / res)) + 1

    if w * h > MAX_GRID_CELLS:
        sys.exit(
            f"model spans {maxx - minx:,.0f} x {maxy - miny:,.0f} units, which at "
            f"--res {res} would need a {w:,} x {h:,} grid ({w * h / 1e9:.1f} billion "
            "cells) - refusing to allocate that.\n"
            "That span almost certainly means the model's units aren't meters. "
            "Pass --unit-scale to convert (0.001 for a millimeter model, 0.01 for "
            "centimeters), or --res for a coarser grid if the extent really is "
            "that large.\n"
            "If a sane --unit-scale still leaves the grid mostly empty, the model "
            "likely mixes local-origin geometry with georeferenced (UTM) layers; "
            "those have to be separated in the CAD tool first - no single scale "
            "fits both.")

    col = np.clip(((x - minx) / res).astype(np.int64), 0, w - 1)
    row = np.clip(((maxy - y) / res).astype(np.int64), 0, h - 1)
    idx = row * w + col

    df = pd.DataFrame({"idx": idx, "z": z, "is_building": is_building})
    grouped = df.groupby("idx")
    cell_z = grouped["z"].quantile(TOP_PERCENTILE / 100.0)
    cell_bldg_frac = grouped["is_building"].mean()

    dem = np.full(h * w, np.nan, dtype=np.float32)
    dem[cell_z.index.to_numpy()] = cell_z.to_numpy()
    dem = dem.reshape(h, w)

    building_frac = np.zeros(h * w, dtype=np.float32)
    building_frac[cell_bldg_frac.index.to_numpy()] = cell_bldg_frac.to_numpy()
    obstacle = (building_frac > 0.5).reshape(h, w)
    return dem, obstacle, minx, maxy, w, h


def suppress_spikes(dem, res, smooth_m=0.9):
    """Gaussian-blurs the DEM to remove tessellation-detail noise (railings,
    mullions, trim - a heavily-detailed architectural model tessellates into
    thousands of small features whose per-cell height can dominate their own
    cell regardless of point density, so a percentile-based per-cell/
    neighborhood filter alone doesn't reliably separate "small loud detail"
    from "broad roof surface" - tried that first, it didn't work).
    Direct spatial smoothing trades away some genuine sharp edges (a real
    parapet line softens a bit) but that's the right tradeoff here: this
    DEM feeds a flood/SPH simulation and a terrain render, not an
    architectural drawing - a smooth, coherent building mass beats a
    razor-precise forest of spikes.
    """
    sigma = max(0.5, smooth_m / res)
    return ndimage.gaussian_filter(dem, sigma=sigma)


def synth_ortho(dem, obstacle, res):
    """No real photography for a CAD model - a hillshade tinted by the
    building mask (warm gray for roofs, cool gray-green for ground) reads
    fine as a terrain texture and matches this project's existing
    dem_hillshade.png convention (see build_dem.hillshade)."""
    hs = hillshade(np.nan_to_num(dem, nan=np.nanmin(dem)), res)
    hs3 = np.stack([hs, hs, hs], axis=-1).astype(np.float64)
    ground_tint = np.array([0.85, 0.88, 0.82])
    bldg_tint = np.array([0.90, 0.78, 0.68])
    tint = np.where(obstacle[..., None], bldg_tint, ground_tint)
    rgb = np.clip(hs3 * tint, 0, 1)
    return (rgb * 255).astype(np.uint8)


def build(model_path, res, out_dir, unit_scale=1.0):
    x, y, z, is_building = extract_points(model_path)
    if len(x) == 0:
        sys.exit("no tessellatable surface geometry found in this model")
    x, y, z = x * unit_scale, y * unit_scale, z * unit_scale

    dem, obstacle, minx, maxy, w, h = points_to_dem(x, y, z, is_building, res)
    coverage = 1.0 - np.isnan(dem).mean()
    print(f"  grid: {w} x {h} @ {res}m ({w * res:.0f} x {h * res:.0f} m), "
         f"{100 * (1 - coverage):.1f}% empty before filling")
    if coverage < 0.02:
        sys.exit(f"only {100 * coverage:.1f}% of the grid has real data - too sparse to "
                 "be a real terrain (fill_holes would mostly be extrapolating guesswork, "
                 "not reconstructing surface). Check the model actually has surface "
                 "geometry on the layers being read, or try --unit-scale if the model's "
                 "real-world scale doesn't match its coordinate values.")

    print("filling holes / despeckling...")
    dem = fill_holes(dem)
    dem = despeckle(dem)
    dem = suppress_spikes(dem, res)
    ortho = synth_ortho(dem, obstacle, res)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "dem.npy"), dem.astype(np.float32))
    np.save(os.path.join(out_dir, "obstacle.npy"), obstacle)
    Image.fromarray(ortho).save(os.path.join(out_dir, "ortho.png"))
    save_transform(os.path.join(out_dir, "dem_transform.json"), float(minx), float(maxy - h * res),
                   res, w, h, extra={"crs": "LOCAL", "source_model": os.path.abspath(model_path),
                                     "unit_scale": unit_scale})
    print(f"wrote {out_dir}/dem.npy, obstacle.npy, ortho.png, dem_transform.json")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to a .3dm file")
    ap.add_argument("--res", type=float, default=0.4,
                    help="m/cell (default 0.4 - fine resolution is fine now that "
                         "suppress_spikes() removes tessellation-detail noise "
                         "spatially instead of needing a coarse grid to average it "
                         "away)")
    ap.add_argument("--unit-scale", type=float, default=1.0,
                    help="multiply raw model coordinates by this to get meters "
                         "(default 1.0 - see module docstring on why the file's "
                         "own 'centimeters' unit tag is not used blindly)")
    ap.add_argument("--out", help="default: output/cad_<model filename>")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(
        os.path.dirname(SCRIPTS_DIR), "output",
        "cad_" + os.path.splitext(os.path.basename(args.model))[0].replace(" ", "_"))
    build(args.model, args.res, out_dir, unit_scale=args.unit_scale)


if __name__ == "__main__":
    main()
