"""Real-measured DEM for one flight of Saint Nicolas Stairs, Beirut.

Saint Nicolas Stairs (Escalier Saint-Nicolas / l'Escalier de l'Art), Rmeil,
Achrafieh: the longest stairway in the Middle East, ~500 m of pedestrian
route and 125 steps total, connecting Rue Gouraud (Gemmayzeh, downhill) to
Rue Sursock (uphill, by the Sursock Museum). It climbs in several distinct
flights separated by street landings, not one continuous run.

This used to be a geometric reconstruction from documented figures (the
original 5cm-resolution drone point cloud didn't cover this location).
It now doesn't need to guess: `terrain/DSM_5cm/...tiff` (a 2.95 x 2.1 km,
5cm drone DSM covering the whole area, obtained separately from the
original corridor crop) does cover Saint Nicolas Stairs. The flight was
located by intersecting two signatures in that raster - narrow+steep (DSM
slope) and visually confirmed against the RGB ortho (regular riser-shadow
striping over ~30 m, ~4.9 m wide, between buildings, matching the known
Gouraud<->Sursock endpoints) - then a clean ~1m-wide center-line transect
was extracted at native 5cm resolution and saved to
`terrain/saint_nicolas_profile.npy` (see saint_nicolas_profile_meta.json
for extraction details and the source UTM anchor).

Honest caveat: the measured profile shows alternating ~2m flat landings
and ~2m sloped ramps rather than crisp box-steps. That's most likely a
photogrammetry limitation, not the true geometry - nadir/oblique drone
capture reconstructs near-vertical riser faces poorly, so individual stone
risers blur into short ramps. Rather than assume idealized steps to
compensate, this uses the real measured surface as-is: it's still real
elevation gain over a real, confirmed staircase, just smoothed at
sub-riser scale.
"""

import json
import os

import numpy as np

TERRAIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "terrain")
PROFILE_PATH = os.path.join(TERRAIN_DIR, "saint_nicolas_profile.npy")
PROFILE_RES_M = 0.05   # native resolution of the extracted profile

WIDTH_M = 4.5          # measured walkway width (conservative - stayed inside
                       # the confirmed flat zone; the true width may be larger,
                       # the far edge wasn't captured in the cross-section check)
SIDEWALK_M = 1.0       # flanking sidewalk strip, flat, each side
WALL_HEIGHT_M = 6.0     # flanking buildings, tall enough to act as flow walls
BASE_ELEV_M = 80.0      # arbitrary local datum (Achrafieh hillside, ~80-100 m ASL)


def saint_nicolas_flight_dem(res=0.05):
    """One real flight of Saint Nicolas Stairs, gridded at `res` m/cell.

    Same row convention as test_synthetic.synthetic_dem: row 0 is the
    uphill (source) end, row increases downhill - so particle_sim's
    existing source_position (emits near row 3) and emit velocity (+y)
    work unchanged.

    Returns (dem, street_mask), both (rows, cols) arrays, street_mask True
    over the walkable surface (not the flanking sidewalks/buildings).
    """
    profile = np.load(PROFILE_PATH).astype(np.float64)
    if res != PROFILE_RES_M:
        src_s = np.arange(len(profile)) * PROFILE_RES_M
        n_rows = int(round(src_s[-1] / res)) + 1
        dst_s = np.arange(n_rows) * res
        profile = np.interp(dst_s, src_s, profile)
    n_rows = len(profile)

    n_cols_street = int(round(WIDTH_M / res))
    n_cols_side = int(round(SIDEWALK_M / res))
    n_cols = n_cols_side * 2 + n_cols_street

    dem = BASE_ELEV_M + np.tile(profile[:, None], (1, n_cols))
    street = np.zeros((n_rows, n_cols), dtype=bool)
    street[:, n_cols_side:n_cols_side + n_cols_street] = True
    dem[~street] += WALL_HEIGHT_M

    return dem, street


def profile_meta():
    with open(os.path.join(TERRAIN_DIR, "saint_nicolas_profile_meta.json")) as f:
        return json.load(f)
