#!/usr/bin/env python3
"""Flatten the buildings the real Al-Masar Al-Akhdar / Fouad Boutros project
actually demolishes.

build_corridor_gi.py never modeled this: `ribbon = zmask & (~building) & ...`
just routes the material band around every standing building inside the zone
alike, so the corridor material ends up appearing to run past/behind
buildings the real design clears away. This script identifies which
buildings that actually is: any building footprint overlapping the OFFICIAL
Fouad Boutros highway design linework (output/masar_fb_highway_official.json,
pulled from AUB's ArcGIS - a road literally cannot be built through a
standing building, so overlap with the as-designed alignment is a
directly data-driven "gets demolished" criterion, no manual list needed) -
EXCEPT buildings that are identified heritage, or institutional/religious/
industrial (schools, hospitals, churches/mosques, government offices,
utilities, the port...) per Beirut Urban Lab's own building survey: real
cities don't demolish the school or the water reservoir to build a park
around it, they route around it, same as the corridor already does for
every OTHER standing building outside its path. Buildings that don't
overlap the alignment at all are left exactly as they are today regardless.

For each flagged building: removed from masks['building'] (so it becomes
normal, paintable/floodable ground) and its footprint in dem.npy is filled
flat to the median elevation of its own immediately-touching perimeter (see
local_dem_fill) - not left at roof height, and not diffused/interpolated
from a wider neighbourhood, which this corridor's real stepped/terraced
character (retaining walls between levels, not smooth slopes) makes
unreliable.

Also fixes the reverse gap: some heritage/institutional buildings (found via
user report - e.g. Beirut Annonciation Orthodox College) have real elevation
data but were never rasterized into masks['building'] at all by the OSM
extract build_terrain.py used, meaning the corridor ribbon would already
paint straight over them today with no flattening step involved. Any
protected footprint landing on open, non-water ground gets recognized as a
real building before anything else runs.

Usage:
  python scripts/flatten_corridor_buildings.py --terrain output/terrain_cut_0.5_v2 \
      --out output/terrain_cut_0.5_v2
"""
import argparse
import os
import shutil
import sys
import json

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
from las_common import load_transform
from build_terrain import BUILDING, GRAVEL, MANNING, INFIL_MMH, ERODIBILITY, rasterize_lines, rasterize_polys
from build_corridor_gi import rasterize as rasterize_zone, zone_rings, spine_from_zone


def unmapped_structure_mask(dem, valid, res, se_m, thr_m):
    """Cells that stand up like a roof, whether or not OSM knows about them.

    masks['building'] comes from an OSM extract that is demonstrably
    incomplete here: 11 of 126 heritage and 1 of 17 institutional buildings
    in this domain carry real roof elevations but were never rasterized as
    buildings at all. That gap is not just cosmetic for this script - it
    silently poisons `fill_sources` below. An unmapped building adjacent to a
    flattened footprint reads as "open ground" at ROOF height, and
    nearest-neighbour fill will happily flatten a demolished building down to
    its neighbour's rooftop. Measured on this domain before this mask
    existed: 14 of 35 flattened footprints ended up still standing >3 m above
    their own surrounding ground, the worst by +23 m - 54-79% of those cells
    filled from a source that was itself >3 m above real ground. That is
    exactly the "the buildings are still there" symptom reported from the 3D
    view, and it was a genuine data defect, not a rendering or cache one.

    Detection is a morphological white top-hat: open the surface with a flat
    structuring element wider than any building footprint, and subtract. An
    opening is idempotent on features wider than the element, so a hillside
    or a terrace level survives it and cancels to ~0, while a bounded,
    building-sized plateau is erased by the opening and shows up as its full
    height above surroundings. `se_m` therefore wants to exceed the widest
    real footprint (95th-percentile minimum-width here is 47 m, so 60 m).

    This is deliberately a veto, not a classifier - it recovers ~80% of
    known-building cells and also flags some genuine high terrace ground.
    That asymmetry is the safe direction: vetoing a true ground cell only
    makes the fill choose the next-nearest ground cell, while missing an
    unmapped roof puts a 20 m step back into the DEM. The outcome is checked
    directly in both directions (nothing left standing, nothing sunk into a
    pit) rather than trusted from the detector's own accuracy.
    """
    k = int(round(se_m / res)) | 1
    filled = np.where(valid, dem, float(np.median(dem[valid])))
    tophat = filled - ndimage.grey_opening(filled, size=(k, k))
    return valid & (tophat > thr_m)


def local_dem_fill(dem, flatten, fill_sources):
    """Fills every flattened cell with the elevation of its single nearest
    open-ground cell (true Euclidean nearest, via distance_transform_edt) -
    the same nearest-fill technique build_terrain.py's own rain-rerouting
    downspout model already uses (`distance_transform_edt(~eligible,
    return_indices=True)`), just applied to elevation instead of a routing
    target.

    Two earlier approaches both broke on this corridor's real character - it
    is explicitly stepped/terraced (retaining walls between levels, per
    build_corridor_gi.py's own terrace placement), not a smooth slope:
    - A single fill_holes_masked() diffusion call over the WHOLE domain's
      flatten mask at once let unrelated, distant terraces blend together
      (some cells came out filled from an elevation the building never
      remotely touched, up to +53 m nonsense).
    - A single flat fill per building (median of its touching perimeter)
      was more stable in aggregate, but a handful of large/oddly-shaped
      footprints genuinely straddle a real multi-terrace hillside (one
      building's OWN original roof spanned 66-99 m - not a bug, that's a
      real stepped structure), and forcing one flat number across that
      discards real structure the nearest-source elevation itself never
      claimed to know.
    Per-cell nearest-neighbour fill has neither failure mode: it can never
    reach past a retaining wall to an unrelated terrace (nearest is nearest),
    and it reproduces a naturally terraced result across a sloped footprint
    by construction, rather than averaging or flattening it away.
    """
    dist_input = ~fill_sources
    _, (iy, ix) = ndimage.distance_transform_edt(dist_input, return_indices=True)
    out = dem.copy()
    out[flatten] = dem[iy[flatten], ix[flatten]]
    # Flattening should only ever remove height, never add it - a handful of
    # building footprints in the source data span an implausible internal
    # range (one is 80.7-135.6 m across a single "building" polygon, clearly
    # an OSM-digitization/courtyard-gap artifact, not a real roof), and for
    # those, even a nearest true-ground source can occasionally sit higher
    # than a cell's own original height. Clamping to "never taller than it
    # already was" can't make anything look less flattened than it started.
    out = np.minimum(out, dem)
    return out


def smooth_flattened(dem, flatten, support, res, radius_m, iterations=12):
    """Grade the cleared lots instead of leaving the raw nearest-source fill.

    `local_dem_fill` gives every cleared cell the elevation of its nearest real
    ground cell. That is the right *value*, but as a surface it is a Voronoi
    mosaic: neighbouring cells can take their elevation from sources on
    different sides of the footprint and meet at a hard seam, and a demolition
    site is not a stepped mosaic - it is graded. This smooths the fill toward
    its surroundings while holding everything outside the footprint fixed
    (Jacobi iterations with Dirichlet boundaries), so the cleared lot joins the
    street continuously and no seam survives inside it.

    Only cleared cells move; real terrain is never touched.

    `support` is what the average is allowed to see, and it must exclude the
    roofs of buildings that stay standing - a cleared lot is usually adjacent
    to one. Averaging those in drags the lot back up toward roof height, and
    the "never raise" clamp cannot catch it, because the cell's own original
    value is that same roof. Measured with a plain valid-cell support: 12 of
    41 cleared footprints came back to standing >3 m, the worst +38 m, i.e.
    the smoothing on its own reintroduced the exact defect the fill fixes.
    Ground plus the cleared lots themselves (which hold ground values by
    construction) is the correct support.
    """
    if radius_m <= 0 or not flatten.any():
        return dem
    k = max(1, int(round(radius_m / res)))
    size = 2 * k + 1
    out = dem.copy()
    den = ndimage.uniform_filter(support.astype(np.float32), size=size)
    for _ in range(iterations):
        num = ndimage.uniform_filter(np.where(support, out, 0.0).astype(np.float32), size=size)
        sm = np.divide(num, den, out=out.astype(np.float32), where=den > 0)
        out[flatten] = sm[flatten]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default="output/terrain_cut_0.5")
    ap.add_argument("--highway", default="output/masar_fb_highway_official.json")
    ap.add_argument("--road-buffer-m", type=float, default=2.5,
                    help="half-width buffer around each highway design line - the "
                         "linework is CAD road-edge/curb/median detail, not a single "
                         "centerline, so only a small buffer is needed to merge nearby "
                         "parallel edges into one paved swath; too large falsely flags "
                         "buildings that merely face the road rather than sit on it "
                         "(7m over-flagged in review), too small fragments the buffer "
                         "between edge pairs (1.5m started leaving gaps) - 2.5m was the "
                         "best-observed balance, tune per corridor if needed")
    ap.add_argument("--heritage", default="output/masar_heritage_buildings_official.json",
                    help="Beirut Urban Lab's own identified-heritage-building polygons for "
                         "the Masar corridor - buildings overlapping these are kept standing "
                         "even if they're inside the highway alignment (matches the reference "
                         "design's 'preserve historic buildings, repurpose as cultural "
                         "facilities' principle), set to '' to disable")
    ap.add_argument("--institutional", default="output/masar_institutional_buildings_official.json",
                    help="Beirut Urban Lab's building survey, filtered to Building_Use in "
                         "(Institutional, Religious, Industrial) - schools, hospitals, "
                         "churches/mosques, government offices, utilities, the port. Kept "
                         "standing for the same reason heritage buildings are: a working "
                         "school or a water reservoir doesn't get torn out for a park, the "
                         "corridor routes around it. Set to '' to disable")
    ap.add_argument("--institutional-osm", default="output/masar_institutional_buildings_osm.json",
                    help="same protection, from OSM amenity tags (school/college/university/"
                         "hospital/clinic/place_of_worship/fire_station/police/government) "
                         "instead of the AUB survey - perfectly aligned with masks['building'] "
                         "since it's the same OSM source, and catches named landmarks (e.g. "
                         "Sagesse University, Beirut Annonciation Orthodox College) the AUB "
                         "survey's Building_Use field left unpopulated for this corridor. "
                         "Set to '' to disable")
    ap.add_argument("--manual-protect", default="output/masar_manual_protected_buildings.json",
                    help="explicit user-confirmed exceptions not covered by any dataset "
                         "queried so far (e.g. buildings identified by on-the-ground local "
                         "knowledge) - same protection as heritage/institutional. Set to "
                         "'' to disable")
    ap.add_argument("--structure-se-m", type=float, default=60.0,
                    help="width of the morphological opening used to spot roofs "
                         "the OSM building mask missed (unmapped_structure_mask). "
                         "Must exceed the widest real footprint or a large roof's "
                         "interior survives the opening and stays a fill source; "
                         "95th-percentile building minimum-width here is 47 m")
    ap.add_argument("--structure-relief-m", type=float, default=2.0,
                    help="how far a cell must stand above its own surroundings, "
                         "after that opening, to be vetoed as a fill source")
    ap.add_argument("--zone", default="output/masar_zone_official.json",
                    help="official Green Path zone polygon, used to derive the "
                         "right-of-way ribbon a demolition candidate must sit in; "
                         "set to '' to select on the alignment linework alone")
    ap.add_argument("--row-halfwidth", type=float, default=18.0,
                    help="half-width (m) of the ROW ribbon around the zone spine - "
                         "keep equal to build_corridor_gi.py's --row-halfwidth, since "
                         "the point is to clear what would otherwise punch holes in it")
    ap.add_argument("--row-overlap-frac", type=float, default=0.5,
                    help="fraction of a building's own footprint that must lie inside "
                         "the ROW ribbon for it to count as in the corridor's path")
    ap.add_argument("--marker-tint", action="store_true",
                    help="paint cleared footprints a synthetic marker colour in "
                         "ortho.png. Only useful when the clearing is baked into the "
                         "terrain; with --defer-to-design the before/after geometry "
                         "already distinguishes them and the tint is false colour")
    ap.add_argument("--flatten-pad-m", type=float, default=4.0,
                    help="clear this far beyond each demolished footprint, to catch "
                         "the roof rim the OSM polygon does not cover (eaves, parapets, "
                         "facade returns) which otherwise survives as a ring of "
                         "full-height spikes around the cleared lot; never takes cells "
                         "from a building that stays standing. 0 disables")
    ap.add_argument("--flatten-smooth-m", type=float, default=3.0,
                    help="smoothing radius (m) applied to cleared lots so they grade "
                         "into the surrounding street instead of keeping the raw "
                         "nearest-source fill's seams; 0 disables")
    ap.add_argument("--defer-to-design", action="store_true",
                    help="do not modify dem.npy/masks - leave every building standing "
                         "(still marker-tinted in the ortho) and write the clearing as "
                         "flatten_delta.npy + flatten_mask.npy + rain_weight_cleared.npy "
                         "for the green-corridor design to apply, so the sandbox has a "
                         "real before/after rather than a pre-cleared 'before'")
    ap.add_argument("--out", default=None, help="default: overwrite --terrain in place")
    args = ap.parse_args()

    t = load_transform(os.path.join(args.terrain, "dem_transform.json"))
    dem = np.load(os.path.join(args.terrain, "dem.npy"))
    masks = dict(np.load(os.path.join(args.terrain, "masks.npz")))
    valid, building = masks["valid"], masks["building"]
    h, w = dem.shape
    res = t["res"]

    with open(args.highway) as f:
        hwy = json.load(f)
    lines = [p for feat in hwy["features"] for p in feat["geometry"]["paths"]]
    print(f"{len(lines)} highway design polylines, {args.road_buffer_m:.1f} m half-width buffer")
    highway_mask = rasterize_lines(lines, t, (h, w), buffer_m=args.road_buffer_m)

    def load_poly_mask(path, label):
        if not path or not os.path.exists(path):
            return np.zeros((h, w), dtype=bool)
        with open(path) as f:
            data = json.load(f)
        polys = [feat["geometry"]["rings"][0] for feat in data["features"]
                if feat.get("geometry", {}).get("rings")]
        print(f"{len(polys)} {label} polygons loaded")
        return rasterize_polys(polys, t, (h, w))

    heritage_mask = load_poly_mask(args.heritage, "identified heritage-building")
    institutional_mask = (load_poly_mask(args.institutional, "institutional/religious/industrial (AUB survey)")
                          | load_poly_mask(args.institutional_osm, "institutional/religious (OSM amenity tags)"))
    manual_mask = load_poly_mask(args.manual_protect, "user-confirmed manual protection")

    # Some protected buildings (e.g. Beirut Annonciation Orthodox College)
    # aren't in masks['building'] at all - real elevation data, but never
    # rasterized as a building by the OSM extract build_terrain.py used. That
    # means the corridor ribbon (`~building`) would already paint straight
    # over them today, with no flattening step involved. Recognize any
    # heritage/institutional/manual footprint landing on open ground as a
    # building before doing anything else, so it's a real standing obstacle
    # like any other - not just excluded from this script's own flatten set.
    water = masks.get("water", np.zeros((h, w), dtype=bool))
    missing = (heritage_mask | institutional_mask | manual_mask) & valid & ~water & ~building
    if missing.any():
        print(f"{int(missing.sum())} cells of protected buildings were missing from "
              f"masks['building'] entirely (present in the source survey, absent from the "
              f"OSM extract) - adding them as buildings before anything else runs")
    building = building | missing   # material rasters patched below, once the output dir exists

    # The highway linework is a narrow CAD alignment, but the corridor the
    # project actually builds is the ~36 m right-of-way ribbon around it. A
    # building can sit squarely inside that ribbon and still miss the 2.5 m
    # linework buffer, and build_corridor_gi.py's `ribbon = zmask & ~building`
    # then routes the green corridor around it - which is what leaves
    # block-sized voids punched through the finished corridor. Measured here
    # before this criterion existed: 0.64 ha of the 3.57 ha ROW band was
    # blocked, 0.23 ha of it by ordinary buildings with no protection at all
    # (the largest single void 1080 m2). A building substantially inside the
    # right-of-way is in the corridor's path whether or not the alignment
    # linework happens to clip it, so it is a demolition candidate on the same
    # terms - and the same heritage/institutional protections still override.
    ribbon_band = np.zeros((h, w), dtype=bool)
    if args.zone and os.path.exists(args.zone):
        zmask = rasterize_zone(zone_rings(args.zone), t, (h, w)) & valid
        rows_s, cols_s = spine_from_zone(zmask, res)
        spine = np.zeros((h, w), dtype=bool)
        spine[np.clip(rows_s.astype(int), 0, h - 1), np.clip(cols_s.astype(int), 0, w - 1)] = True
        band_dist = ndimage.distance_transform_edt(~spine) * res
        ribbon_band = zmask & (band_dist <= args.row_halfwidth)
        print(f"ROW ribbon band: {ribbon_band.sum() * res * res / 1e4:.2f} ha "
              f"(<= {args.row_halfwidth:.0f} m from the zone spine)")

    lbl, n = ndimage.label(building)
    # a building counts as in the corridor's path if enough of IT sits in the
    # band - not merely if the band clips its corner
    in_band = np.array(ndimage.sum(ribbon_band & building, lbl, np.arange(1, n + 1)))
    area = np.array(ndimage.sum(building, lbl, np.arange(1, n + 1)))
    overlaps_ribbon = set((np.flatnonzero(
        (area > 0) & (in_band / np.maximum(area, 1) >= args.row_overlap_frac)) + 1).tolist())
    overlaps_linework = set(np.unique(lbl[highway_mask & building])) - {0}
    overlaps_highway = overlaps_linework | overlaps_ribbon
    overlaps_heritage = set(np.unique(lbl[heritage_mask & building])) - {0}
    overlaps_institutional = set(np.unique(lbl[institutional_mask & building])) - {0}
    overlaps_manual = set(np.unique(lbl[manual_mask & building])) - {0}
    kept_heritage_ids = sorted(overlaps_highway & overlaps_heritage)
    # later protections win ties if a building is somehow flagged more than
    # one way - any reason keeps it standing, the QA figure just needs one bucket.
    kept_institutional_ids = sorted((overlaps_highway & overlaps_institutional) - set(kept_heritage_ids))
    kept_manual_ids = sorted((overlaps_highway & overlaps_manual) - set(kept_heritage_ids) - set(kept_institutional_ids))
    flat_ids = sorted(overlaps_highway - overlaps_heritage - overlaps_institutional - overlaps_manual)
    flatten = np.isin(lbl, flat_ids)
    kept = building & ~flatten
    # Courtyards and light-wells inside a demolished footprint are carved out
    # of masks['building'] by build_terrain.py's courtyard pass, so they are
    # not in `flatten` and keep their original elevation - which for an
    # interior roof-level void is roof height. Left alone they survive
    # demolition as isolated needles standing up to 42 m out of the cleared
    # lot: the "sharp spikes" reported in the 3D view. Anything fully enclosed
    # by a cleared footprint goes with it, unless it is a kept building.
    # The OSM footprint is a plan-view polygon and the DEM is a surface, so the
    # rasterized footprint is routinely a little smaller than the roof it
    # covers - eaves, parapets, the facade's own returns. Whatever of the roof
    # falls outside the polygon is not in `flatten`, and survives demolition as
    # a rim of full-height cells right around the cleared lot. Measured before
    # this pad: the one-cell ring around cleared footprints had 37% of its
    # cells standing >3 m above local ground, the worst +50 m - roof height.
    # The excess decays with distance (37% at 0.5 m, 26% at 1.0 m, 21% at
    # 1.5 m) and then plateaus near 18%, which is the genuine neighbouring
    # structures and terrain the pad must NOT eat; hence a short pad, and
    # `~kept` so it can never take a bite out of a building that stays.
    if args.flatten_pad_m > 0:
        k = max(1, int(round(args.flatten_pad_m / res)))
        pad = (ndimage.binary_dilation(flatten, np.ones((2 * k + 1, 2 * k + 1), bool))
               & ~flatten & valid & ~kept & ~water)
        if pad.any():
            print(f"{int(pad.sum())} cells within {args.flatten_pad_m:.1f} m of a cleared "
                  f"footprint cleared with it (roof rim outside the OSM polygon)")
            flatten |= pad
            kept = building & ~flatten
    # after padding, so a pad that closes a gap is filled too
    enclosed = ndimage.binary_fill_holes(flatten) & ~flatten & ~kept
    if enclosed.any():
        print(f"{int(enclosed.sum())} cells enclosed inside cleared footprints "
              f"(courtyards/light-wells at roof height) cleared with them")
        flatten |= enclosed
        kept = building & ~flatten
    print(f"{n} buildings total; {len(overlaps_highway)} in the corridor's path "
          f"({len(overlaps_linework)} from the alignment linework, "
          f"{len(overlaps_ribbon - overlaps_linework)} more from the ROW-ribbon test), of "
          f"which {len(kept_heritage_ids)} are identified heritage, {len(kept_institutional_ids)} "
          f"are institutional/religious/industrial, and {len(kept_manual_ids)} are manually "
          f"confirmed (all kept standing) -> flattening {len(flat_ids)} buildings / "
          f"{int(flatten.sum())} cells ({flatten.sum() * res * res / 1e4:.2f} ha), "
          f"{int(kept.sum())} cells left standing ({kept.sum() * res * res / 1e4:.2f} ha)")

    new_building = building & ~flatten
    # only genuinely open ground seeds the fill - neither the flattened
    # footprint itself nor any OTHER (kept) building's roof should leak in,
    # including the roofs of buildings the OSM extract never mapped (see
    # unmapped_structure_mask - `~building` alone is not enough).
    unmapped = unmapped_structure_mask(dem, valid, res, args.structure_se_m,
                                       args.structure_relief_m)
    fill_sources = valid & ~building & ~unmapped
    print(f"unmapped-structure veto: {int((unmapped & ~building).sum())} cells "
          f"outside masks['building'] look like roofs and are excluded as fill "
          f"sources ({100 * (unmapped & ~building).sum() / max(int((valid & ~building).sum()), 1):.0f}% "
          f"of non-building ground)")
    new_dem = local_dem_fill(dem, flatten, fill_sources)
    # exactly the fill's own trusted-ground set, plus the cleared lots (whose
    # values came from it). Anything that stands up like a roof is excluded
    # here for the same reason it is excluded as a fill source - `~building`
    # alone leaves the unmapped ones in, and the average drags the cleared lot
    # back up onto them.
    smooth_support = valid & (flatten | (~building & ~unmapped))
    new_dem = smooth_flattened(new_dem, flatten, smooth_support, res, args.flatten_smooth_m)
    # smoothing is an averaging pass, so re-assert the one hard invariant the
    # fill carries: clearing a building may lower a cell, never raise it.
    new_dem = np.minimum(new_dem, dem)

    # The ortho photo is a real aerial image: it still shows the actual,
    # physically-standing roof on a cleared cell, because nobody has demolished
    # it to re-photograph. A synthetic marker tint used to be painted here so a
    # reviewer could not mistake a cleared lot for a standing building. That
    # was only needed while the clearing was baked into the base terrain and
    # the two states looked alike; with --defer-to-design the "before" view
    # shows the buildings standing at full height and the "after" view drops
    # them, so the geometry itself carries the distinction and the tint is just
    # false colour over the whole corridor. Off by default; --marker-tint
    # brings it back for a review that wants the cleared set called out.
    ortho_img = Image.open(os.path.join(args.terrain, "ortho.png"))
    new_ortho = np.array(ortho_img)
    if args.marker_tint:
        # a tone with near-zero occurrence in the real imagery (0.02% of valid
        # pixels within a generous colour distance), chosen so it reads as an
        # unmistakable synthetic marker rather than a plausible roof - an
        # earlier tan matched ~10% of Beirut's real tile/gravel roofs.
        new_ortho = new_ortho.astype(np.float32)
        new_ortho[flatten] = np.array([180, 90, 150], dtype=np.float32)
        new_ortho = np.clip(new_ortho, 0, 255).astype(np.uint8)

    out = args.out or args.terrain
    if os.path.abspath(out) != os.path.abspath(args.terrain):
        if os.path.exists(out):
            shutil.rmtree(out)
        shutil.copytree(args.terrain, out)

    # Rain rerouting has to move with the buildings. build_terrain.py's
    # downspout model sends a roof's/courtyard's rain to its nearest eligible
    # street cell; a cleared lot must both stop being a source and start
    # catching its own rain, or the "after" case quietly rains the corridor
    # through a downspout that no longer exists. This mirrors build_terrain.py
    # section 5 with the cleared set moved from sources to targets.
    courtyard0 = masks.get("courtyard", np.zeros((h, w), dtype=bool))
    src_new = ((building | courtyard0) & ~flatten) & valid
    elig_new = ((masks.get("eligible", valid & ~building) | (flatten & valid))
                & ~src_new & valid & ~water)
    rw_cleared = elig_new.astype(np.float32)
    if src_new.any() and elig_new.any():
        _, (ir, ic) = ndimage.distance_transform_edt(~elig_new, return_indices=True)
        dst = ir[src_new].astype(np.int64) * w + ic[src_new].astype(np.int64)
        np.add.at(rw_cleared.reshape(-1), dst, 1.0)
    expected = int((valid & ~water).sum())
    print(f"rain weight with the corridor cleared: {rw_cleared.sum():.0f} rain cells "
          f"(= {expected} expected)")

    Image.fromarray(new_ortho, ortho_img.mode).save(os.path.join(out, "ortho.png"))

    if args.defer_to_design:
        # Everything about the "before" terrain stays as it is - the buildings
        # stand, at their real height, and only the marker tint says they are
        # slated for clearing. The clearing itself ships as a delta the green
        # corridor applies, so loading the corridor is what demolishes them and
        # before/after is a real comparison instead of two views of a terrain
        # that was already cleared.
        delta = (new_dem - dem).astype(np.float32)
        delta[~flatten] = 0.0
        np.save(os.path.join(out, "flatten_delta.npy"), delta)
        np.save(os.path.join(out, "flatten_mask.npy"), flatten)
        np.save(os.path.join(out, "rain_weight_cleared.npy"), rw_cleared)
        print(f"deferred: wrote flatten_delta.npy ({float(delta[flatten].mean()):.1f} m mean "
              f"drop over {int(flatten.sum())} cells), flatten_mask.npy, "
              f"rain_weight_cleared.npy - dem.npy and masks.npz left as the 'before' state")

    if not args.defer_to_design:
        np.save(os.path.join(out, "dem.npy"), new_dem.astype(np.float32))
        masks["building"] = new_building
        # courtyard/eligible were derived from the OLD building mask in
        # build_terrain.py and are now stale over the flattened footprint -
        # patch them directly rather than require a full terrain rebuild.
        if "courtyard" in masks:
            masks["courtyard"] = masks["courtyard"] & ~flatten
        if "eligible" in masks:
            masks["eligible"] = masks["eligible"] | (flatten & valid)
        np.savez_compressed(os.path.join(out, "masks.npz"), **masks)

        # manning/infil/erodible/landcover/rain_weight were computed from the OLD
        # building mask too. `flatten` cells would otherwise still hydraulically
        # behave like a building (n=0.05, infil=0, no direct rain) despite no
        # longer being one - build_corridor_gi.py's material bake will overwrite
        # whatever of this ends up inside the actual ribbon/green-node footprint
        # anyway, but give every flattened cell a sane "freshly cleared ground"
        # default regardless, so nothing outside that footprint is left stale.
        # `missing` cells are the opposite direction: newly recognized as
        # buildings, so they need real building hydraulics instead of whatever
        # generic landcover class they'd been misclassified as.
        for name, table, dtype in [("manning.npy", MANNING, np.float32),
                                   ("infil_mmh.npy", INFIL_MMH, np.float32),
                                   ("erodible.npy", ERODIBILITY, np.float32)]:
            path = os.path.join(out, name)
            if os.path.exists(path):
                arr = np.load(path)
                arr[flatten] = table[GRAVEL]
                arr[missing] = table[BUILDING]
                np.save(path, arr.astype(dtype))
        lc_path = os.path.join(out, "landcover.npy")
        if os.path.exists(lc_path):
            lc = np.load(lc_path)
            lc[flatten] = GRAVEL
            lc[missing] = BUILDING
            np.save(lc_path, lc.astype(np.uint8))
        rw_path = os.path.join(out, "rain_weight.npy")
        if os.path.exists(rw_path):
            rw = np.load(rw_path)
            rw[flatten] = 1.0   # ordinary open ground now, not a roof redirected elsewhere
            # `missing` is left as-is deliberately: it was already receiving rain
            # directly (never having been recognized as a roof to reroute), and
            # zeroing it out now without adding that weight to a downspout target
            # would silently delete rain volume from the domain. Not rerouting a
            # newly-recognized roof's rain is a minor simplification; losing
            # volume from the mass balance would not be.
            np.save(rw_path, rw.astype(np.float32))

    # QA figure: which buildings got flagged vs kept, and the highway buffer
    # footprint that decided it - review this against the reference image
    # before this ever lands in the official corridor template for real.
    ortho = np.array(Image.open(os.path.join(out, "ortho.png")))
    kept_heritage = np.isin(lbl, kept_heritage_ids)
    kept_institutional = np.isin(lbl, kept_institutional_ids)
    kept_manual = np.isin(lbl, kept_manual_ids)
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[highway_mask] = [76, 175, 80, 70]
    overlay[kept] = [59, 130, 246, 150]
    overlay[kept_heritage] = [250, 204, 21, 210]
    overlay[kept_institutional] = [45, 212, 191, 220]
    overlay[kept_manual] = [168, 85, 247, 220]
    overlay[flatten] = [239, 68, 68, 190]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(ortho)
    ax.imshow(overlay)
    n_generic_kept = n - len(flat_ids) - len(kept_heritage_ids) - len(kept_institutional_ids) - len(kept_manual_ids)
    ax.set_title(f"flattened (red, {len(flat_ids)}) / kept-standing (blue, {n_generic_kept}) / "
                 f"kept-heritage (gold, {len(kept_heritage_ids)}) / kept-institutional "
                 f"(teal, {len(kept_institutional_ids)}) / kept-manual (purple, "
                 f"{len(kept_manual_ids)}) - green = highway alignment buffer")
    ax.axis("off")
    fig.savefig(os.path.join(out, "buildings_flatten_qa.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out}/ (+ buildings_flatten_qa.png)")


if __name__ == "__main__":
    main()
