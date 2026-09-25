#!/usr/bin/env python3
"""Turns an AutoCAD .dwg into the same (dem.npy, ortho.png,
dem_transform.json) triplet load_cad_model.py produces from a .3dm - see
that module's docstring for why that triplet is the actual integration
point with the rest of the pipeline (flood_sim.py/particle_sim.py/
render_3d.py only ever read it, agnostic to source format).

Two real steps, not one:
  1. DWG -> DXF via ODA File Converter (ezdxf.addons.odafc). This part
     works and needs nothing beyond `winget install ODA.ODAFileConverter`
     (a real, standard package-manager install - not a manually-fetched
     installer). ezdxf reads DXF directly but not binary DWG.
  2. Extracting actual vertex data from the DXF. This is the part that may
     come back thin or empty, and there's no way around it: AutoCAD-
     authored buildings commonly model solids as 3DSOLID entities, which
     store an opaque ACIS/ASM binary blob (Autodesk's proprietary B-rep
     kernel), not a mesh. ezdxf exposes that blob (`entity.acis_data`) but
     cannot tessellate it - doing so needs a real ACIS/ASM geometry kernel,
     which has no free/open Python binding. Confirmed (on the file this was
     built against) that AutoCAD's "proxy graphic" fallback tessellation -
     the one legitimate escape hatch DXF offers for exactly this situation -
     isn't present either (`entity.proxy_graphic` is None on all of them).
     What DOES extract cleanly: LINE, LWPOLYLINE, 3DFACE, MESH, POLYFACE/
     POLYMESH, and INSERT block instances of any of those (block
     transforms are applied recursively). If a file's buildings are
     genuinely modeled as 3DSOLID with no proxy graphic, this will report
     a low coverage number and mostly empty terrain - that's an honest
     result, not a bug, and the fix is exporting the solids to a mesh
     format (.obj, .3dm) from the CAD tool that authored them, not this
     script.

Usage:
  python scripts/load_dwg_model.py --model "3D/some_model.dwg" --res 0.4 \
      --out output/cad_dwg
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
from build_dem import fill_holes, despeckle
from las_common import save_transform
from load_cad_model import BUILDING_LAYER_HINTS, points_to_dem, suppress_spikes, synth_ortho

MESH_TYPES = {"LINE", "LWPOLYLINE", "3DFACE", "MESH", "POLYFACE", "POLYMESH", "POLYLINE"}

# common install locations winget/the ODA installer use - tried in order so
# this doesn't hardcode one version number
ODA_CANDIDATE_PATHS = [
    r"C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe",
]


def _find_oda_exe():
    import glob
    for p in ODA_CANDIDATE_PATHS:
        if os.path.exists(p):
            return p
    for p in glob.glob(r"C:\Program Files\ODA\ODAFileConverter*\ODAFileConverter.exe"):
        return p
    return None


def _entity_points(e, xform=None):
    """Yields (x, y, z) world-space points from one DXF entity's own
    geometry (LINE endpoints, LWPOLYLINE/POLYLINE vertices, 3DFACE/MESH/
    POLYFACE vertices), transformed by `xform` (an ezdxf Matrix44, for
    entities reached through a block INSERT) if given."""
    t = e.dxftype()
    pts = []
    if t == "LINE":
        pts = [e.dxf.start, e.dxf.end]
    elif t == "LWPOLYLINE":
        elev = e.dxf.elevation
        pts = [(p[0], p[1], elev) for p in e.get_points("xy")]
    elif t == "POLYLINE":
        pts = [v.dxf.location for v in e.vertices]
    elif t == "3DFACE":
        pts = [e.dxf.vtx0, e.dxf.vtx1, e.dxf.vtx2, e.dxf.vtx3]
    elif t in ("MESH", "POLYFACE", "POLYMESH"):
        try:
            pts = list(e.vertices)
        except Exception:
            pts = []
    for p in pts:
        x, y, z = p[0], p[1], p[2] if len(p) > 2 else 0.0
        if xform is not None:
            x, y, z = xform.transform((x, y, z))
        yield (x, y, z)


def extract_points(dxf_path):
    import ezdxf

    print(f"reading {dxf_path} ...")
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    layer_is_building = {}
    for layer in doc.layers:
        name = (layer.dxf.name or "").lower()
        layer_is_building[layer.dxf.name] = any(h in name for h in BUILDING_LAYER_HINTS)

    xs, ys, zs, is_bldg = [], [], [], []
    n_mesh_entities = 0
    n_solid_entities = 0  # counted, not extracted - see module docstring

    def walk(entities, xform, depth=0):
        nonlocal n_mesh_entities, n_solid_entities
        if depth > 12:  # guard against pathological/circular block nesting
            return
        for e in entities:
            t = e.dxftype()
            if t == "INSERT":
                block = doc.blocks.get(e.dxf.name)
                insert_xform = e.matrix44() if hasattr(e, "matrix44") else None
                combined = insert_xform if xform is None else (
                    insert_xform @ xform if insert_xform is not None else xform)
                walk(block, combined, depth + 1)
                continue
            if t == "3DSOLID":
                n_solid_entities += 1
                continue
            if t not in MESH_TYPES:
                continue
            building = layer_is_building.get(e.dxf.layer, False)
            got_any = False
            for x, y, z in _entity_points(e, xform):
                xs.append(x); ys.append(y); zs.append(z)
                is_bldg.append(building)
                got_any = True
            if got_any:
                n_mesh_entities += 1

    walk(msp, None)
    print(f"  extracted {len(xs):,} points from {n_mesh_entities} mesh-like entities; "
         f"{n_solid_entities} ACIS 3DSOLID entities skipped (no tessellation available - "
         "see module docstring)")
    if n_solid_entities:
        total = n_mesh_entities + n_solid_entities
        print(f"  coverage: {100 * n_mesh_entities / total:.1f}% of geometry was usable "
             f"({n_solid_entities} solids excluded) - a low number here means most of "
             "this model's real content is ACIS solids this can't read; the terrain "
             "below will be correspondingly sparse")

    return (np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64),
            np.asarray(zs, dtype=np.float64), np.asarray(is_bldg, dtype=bool))


def dwg_to_dxf(dwg_path, out_dir, oda_path=None):
    import ezdxf
    from ezdxf.addons import odafc

    oda_path = oda_path or _find_oda_exe()
    if not oda_path:
        sys.exit("ODA File Converter not found - install it with:\n"
                 "  winget install --id ODA.ODAFileConverter\n"
                 "(free, from the Open Design Alliance; needed to read DWG at all)")
    ezdxf.options.set("odafc-addon", "win_exec_path", oda_path)
    if not odafc.is_installed():
        sys.exit(f"ODA File Converter not usable at {oda_path}")

    print(f"converting {dwg_path} -> DXF via ODA File Converter...")
    t0 = time.time()
    doc = odafc.readfile(dwg_path)
    print(f"  converted in {time.time() - t0:.1f}s, DXF version {doc.dxfversion}")
    os.makedirs(out_dir, exist_ok=True)
    dxf_path = os.path.join(out_dir, "converted.dxf")
    doc.saveas(dxf_path)
    return dxf_path


def build(model_path, res, out_dir, unit_scale=1.0, oda_path=None):
    dxf_path = dwg_to_dxf(model_path, out_dir, oda_path=oda_path)
    x, y, z, is_building = extract_points(dxf_path)
    if len(x) == 0:
        sys.exit("no usable (non-ACIS) geometry found in this .dwg - see module "
                 "docstring; export solids to a mesh format (.obj/.3dm) instead")
    x, y, z = x * unit_scale, y * unit_scale, z * unit_scale

    dem, obstacle, minx, maxy, w, h = points_to_dem(x, y, z, is_building, res)
    coverage = 1.0 - np.isnan(dem).mean()
    print(f"  grid: {w} x {h} @ {res}m ({w * res:.0f} x {h * res:.0f} m), "
         f"{100 * (1 - coverage):.1f}% empty before filling")
    if coverage < 0.02:
        sys.exit(f"only {100 * coverage:.1f}% of the grid has real data - the extracted "
                 "points (scattered reference lines/boundaries, most likely) don't "
                 "actually describe a surface, they're too sparse relative to this "
                 "extent. fill_holes would just be extrapolating near-total guesswork "
                 "across the empty majority, not reconstructing real terrain - refusing "
                 "to write a terrain that would look plausible but isn't. This means "
                 "this file's real geometry is ACIS solids with no usable fallback (see "
                 "module docstring) - export them to a mesh format from the CAD tool "
                 "that authored them.")

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
    ap.add_argument("--model", required=True, help="path to a .dwg file")
    ap.add_argument("--res", type=float, default=0.4, help="m/cell")
    ap.add_argument("--unit-scale", type=float, default=1.0)
    ap.add_argument("--oda-path", help="path to ODAFileConverter.exe (default: auto-detect)")
    ap.add_argument("--out", help="default: output/cad_<model filename>")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(
        os.path.dirname(SCRIPTS_DIR), "output",
        "cad_" + os.path.splitext(os.path.basename(args.model))[0].replace(" ", "_"))
    build(args.model, args.res, out_dir, unit_scale=args.unit_scale, oda_path=args.oda_path)


if __name__ == "__main__":
    main()
