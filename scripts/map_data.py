"""Paths to the full-map 5cm drone rasters (DSM/DTM/RGB), covering a
2.95 x 2.1 km area of Beirut - much bigger than the data/*.las crop the
main flood_sim.py pipeline uses. See real_stairs.py's docstring for how
these were found and used for the Saint Nicolas Stairs scene.

The three rasters have moved around between machines (they arrived as
separate multi-GB zips, so they get unpacked wherever there was room), and
a hardcoded root silently broke every map_point consumer with a
FileNotFoundError that read like missing data rather than a wrong path.
So the root is searched for instead: BEIRUT_MAP_DATA wins if set, otherwise
the first CANDIDATE_ROOTS entry that actually contains the DSM.
"""

import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)

DSM_REL = os.path.join("DSM_5cm", "Beirut_drone_40-150m_DSM_5cm_epsg32636.tiff")
DTM_REL = os.path.join("DTM_5cm", "Beirut_drone_40-150m_DTM_5cm_epsg32636.tiff")
RGB_REL = os.path.join("RGB_5cm", "Beirut_drone_40-150m_RGB_5cm_epsg32636.tiff")

CANDIDATE_ROOTS = (
    os.path.join(PROJECT_ROOT, "3D"),      # where the AUB zips currently unpack
    PROJECT_ROOT,
    os.path.dirname(PROJECT_ROOT),         # the original layout, sibling of terrain/
)

MAP_RES_M = 0.05  # native resolution of all three rasters


def _discover_root():
    env = os.environ.get("BEIRUT_MAP_DATA")
    if env:
        return env
    for root in CANDIDATE_ROOTS:
        if os.path.exists(os.path.join(root, DSM_REL)):
            return root
    return CANDIDATE_ROOTS[0]


MAP_DATA_ROOT = _discover_root()

DSM_PATH = os.path.join(MAP_DATA_ROOT, DSM_REL)
DTM_PATH = os.path.join(MAP_DATA_ROOT, DTM_REL)
RGB_PATH = os.path.join(MAP_DATA_ROOT, RGB_REL)

# One LZW strip of the DSM decoded as garbage when it was salvaged from a
# CRC-corrupt zip; see DSM_5cm/CORRUPT_REGION.json, which is the source of
# truth for the band and is read at runtime rather than duplicated here.
# Values inside it can look plausible while being wrong, so there is no
# detecting this from the data - it has to be masked by position.
CORRUPT_REGION_JSON = os.path.join(MAP_DATA_ROOT, "DSM_5cm", "CORRUPT_REGION.json")


def corrupt_utm_y_band():
    """(y_min, y_max) UTM band of the damaged DSM strip, or None if the
    sidecar isn't present (a clean re-request from AUB would drop it)."""
    if not os.path.exists(CORRUPT_REGION_JSON):
        return None
    import json
    try:
        with open(CORRUPT_REGION_JSON) as f:
            band = json.load(f).get("bad_utm_y_band")
    except (OSError, ValueError):
        return None
    if not band or len(band) != 2:
        return None
    return float(min(band)), float(max(band))


def require_map_data():
    missing = [p for p in (DSM_PATH, RGB_PATH) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Full-map rasters not found:\n" + "\n".join(missing) +
            f"\n(looked under {MAP_DATA_ROOT}; also tried "
            + ", ".join(CANDIDATE_ROOTS) +
            "\nset the BEIRUT_MAP_DATA env var if they live elsewhere)")
