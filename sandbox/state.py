"""Loads the base (read-only) terrain once at server startup, and manages
the sandbox's designs (edits layered on top of the base terrain).

Never mutate anything on `base` after construction — each design holds its
own copies of the mutable layers (material, dem_delta).
"""
import copy
import dataclasses
import datetime
import glob
import json
import os
import shutil
import threading
import time

import numpy as np

from flood_gpu import load_terrain
from build_corridor_gi import rasterize, zone_rings
from bake_corridor import PROPS as BAKE_PROPS

TERRAIN_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "terrain_cut_0.5")
ZONE_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "masar_zone_official.json")
SANDBOX_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "sandbox")
STORMS_DIR = os.path.join(os.path.dirname(__file__), "..", "storms")
DESIGNS_DIR = os.path.join(SANDBOX_DIR, "designs")
OFFICIAL_MATERIAL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "output", "corridor_gi_cut", "material.npy")

# Swatch colors for the seven literature-grounded materials in bake_corridor.PROPS.
MATERIAL_COLORS = {
    1: "#8a8a8a", 2: "#c26fd4", 3: "#2e8b57", 4: "#d9b38c",
    5: "#7ec850", 6: "#1f78b4", 7: "#b8860b",
}

MAX_PATCH_CELLS = 1_000_000
DEM_DELTA_LIMIT_M = 3.0


class Base:
    def __init__(self, terrain=TERRAIN_DIR, zone_path=ZONE_PATH, cache_dir=SANDBOX_DIR):
        (self.dem, self.t, self.masks, self.manning,
         self.infil, self.rain_w, self.gauges) = load_terrain(terrain)
        self.terrain = terrain
        self.h, self.w = self.dem.shape

        os.makedirs(cache_dir, exist_ok=True)
        zone_cache = os.path.join(cache_dir, "zone.npy")
        if os.path.exists(zone_cache):
            self.zone = np.load(zone_cache)
        else:
            rings = zone_rings(zone_path)
            self.zone = rasterize(rings, self.t, self.dem.shape)
            np.save(zone_cache, self.zone)

        valid, building, water = self.masks["valid"], self.masks["building"], self.masks["water"]
        self.editable_zone = self.zone & valid & ~building
        self.editable_full = valid & ~building & ~water

        lo = float(np.nanmin(self.dem[valid]))
        hi = float(np.nanmax(self.dem[valid]))
        self.dem_min = lo
        self.dem_scale = (hi - lo) / 65535.0


SANDBOX_STORMS_DIR = os.path.join(SANDBOX_DIR, "storms")


def list_storms():
    names = []
    for pattern_dir in (STORMS_DIR, SANDBOX_STORMS_DIR):
        if os.path.isdir(pattern_dir):
            for p in sorted(glob.glob(os.path.join(pattern_dir, "*.json"))):
                names.append(os.path.splitext(os.path.basename(p))[0])
    return names


def get_storm(name):
    for pattern_dir in (STORMS_DIR, SANDBOX_STORMS_DIR):
        path = os.path.join(pattern_dir, f"{name}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise KeyError(name)


def save_storm(storm):
    """Recomputes total_mm/duration server-side (never trust the client) and
    writes the storm to the sandbox's custom-storm directory."""
    os.makedirs(SANDBOX_STORMS_DIR, exist_ok=True)
    steps = storm["steps"]
    total_mm = sum((t1 - t0) * mmh for t0, t1, mmh in steps) / 3600.0
    duration = max(t1 for _, t1, _ in steps) + 1800.0
    out = {"name": storm["name"], "steps": steps,
           "total_mm": round(total_mm, 2), "duration": duration}
    with open(os.path.join(SANDBOX_STORMS_DIR, f"{storm['name']}.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def design_sha256(design):
    import hashlib
    h = hashlib.sha256()
    h.update(design.material.tobytes())
    h.update(design.dem_delta.tobytes())
    h.update(json.dumps(design.materials, sort_keys=True).encode())
    return h.hexdigest()


def default_materials():
    """Fresh copy of the seven built-in materials, seeded from bake_corridor.PROPS."""
    materials = []
    for cls, (infil, n, depr, label) in BAKE_PROPS.items():
        materials.append({
            "id": cls, "label": label, "color": MATERIAL_COLORS.get(cls, "#999999"),
            "infil_mmh": float(infil), "manning_n": float(n), "depression_m": float(depr),
            "builtin": True,
        })
    return {"materials": materials}


def clamp_materials(materials_list):
    """Validate + clamp a materials table before it is trusted (§3.3 rules)."""
    out = []
    for m in materials_list:
        out.append({
            "id": int(m["id"]),
            "label": str(m["label"])[:60],
            "color": str(m["color"]),
            "infil_mmh": float(min(max(m["infil_mmh"], 0.0), 1000.0)),
            "manning_n": float(min(max(m["manning_n"], 0.01), 0.5)),
            "depression_m": float(min(max(m["depression_m"], 0.0), 1.0)),
            "builtin": bool(m.get("builtin", False)),
        })
    return {"materials": out}


@dataclasses.dataclass
class Design:
    name: str
    material: np.ndarray      # uint16 (h, w); 0 = untouched base terrain
    dem_delta: np.ndarray     # float32 (h, w); metres added(+)/removed(-)
    materials: dict           # {"materials": [...]}
    notes: str = ""
    unlocked: bool = False
    created: str = ""
    modified: str = ""
    dirty: bool = False

    def to_json(self):
        return {
            "design": {
                "name": self.name, "notes": self.notes,
                "created": self.created, "modified": self.modified,
                "base_terrain": "output/terrain_cut_0.5", "unlocked": self.unlocked,
            },
            "materials": self.materials,
        }


def _atomic_save_npy(path, arr):
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _save_design_to_disk(root, design):
    d = os.path.join(root, design.name)
    os.makedirs(d, exist_ok=True)
    _atomic_save_npy(os.path.join(d, "material.npy"), design.material)
    _atomic_save_npy(os.path.join(d, "dem_delta.npy"), design.dem_delta)
    _atomic_write_json(os.path.join(d, "materials.json"), design.materials)
    _atomic_write_json(os.path.join(d, "design.json"), design.to_json()["design"])
    design.dirty = False


def apply_patch(base, design, x, y, w, h, material=None, dem_delta=None):
    """Clamp an incoming stroke patch to the editable mask (server-side —
    this is what makes the zone lock real) and write it into the design."""
    H, W = base.h, base.w
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return 0

    mask = base.editable_full if design.unlocked else base.editable_zone
    sub_mask = mask[y0:y1, x0:x1]
    applied = int(sub_mask.sum())

    if material is not None:
        m_sub = material[y0 - y: y1 - y, x0 - x: x1 - x]
        design.material[y0:y1, x0:x1][sub_mask] = m_sub[sub_mask]
    if dem_delta is not None:
        d_sub = np.clip(dem_delta[y0 - y: y1 - y, x0 - x: x1 - x],
                         -DEM_DELTA_LIMIT_M, DEM_DELTA_LIMIT_M)
        design.dem_delta[y0:y1, x0:x1][sub_mask] = d_sub[sub_mask]

    if material is not None or dem_delta is not None:
        design.dirty = True
        design.modified = now_iso()
    return applied


class DesignStore:
    def __init__(self, base, root=DESIGNS_DIR):
        self.base = base
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self.designs: dict[str, Design] = {}
        self._lock = threading.Lock()
        self._load_existing()

    def _load_existing(self):
        for name in sorted(os.listdir(self.root)):
            d = os.path.join(self.root, name)
            if not os.path.isdir(d):
                continue
            try:
                material = np.load(os.path.join(d, "material.npy"))
                dem_delta = np.load(os.path.join(d, "dem_delta.npy"))
                with open(os.path.join(d, "materials.json")) as f:
                    materials = json.load(f)
                with open(os.path.join(d, "design.json")) as f:
                    meta = json.load(f)
                ts = now_iso()
                self.designs[name] = Design(
                    name=name, material=material, dem_delta=dem_delta,
                    materials=materials, notes=meta.get("notes", ""),
                    unlocked=meta.get("unlocked", False),
                    created=meta.get("created", ts), modified=meta.get("modified", ts))
            except Exception as e:
                print(f"[sandbox] skipping broken design '{name}': {e}")

    def list(self):
        with self._lock:
            return [
                {"name": d.name, "modified": d.modified, "notes": d.notes,
                 "unlocked": d.unlocked}
                for d in self.designs.values()
            ]

    def create(self, name, template):
        with self._lock:
            if name in self.designs:
                raise KeyError(f"design '{name}' already exists")
            h, w = self.base.h, self.base.w
            if template == "blank":
                material = np.zeros((h, w), np.uint16)
                dem_delta = np.zeros((h, w), np.float32)
                materials = default_materials()
            elif template == "official":
                material = np.load(OFFICIAL_MATERIAL_PATH).astype(np.uint16)
                dem_delta = np.zeros((h, w), np.float32)
                materials = default_materials()
            elif template in self.designs:
                src = self.designs[template]
                material = src.material.copy()
                dem_delta = src.dem_delta.copy()
                materials = copy.deepcopy(src.materials)
            else:
                raise ValueError(f"unknown template '{template}'")
            ts = now_iso()
            design = Design(name=name, material=material, dem_delta=dem_delta,
                             materials=materials, created=ts, modified=ts)
            self.designs[name] = design
            _save_design_to_disk(self.root, design)
            return design

    def get(self, name) -> Design:
        if name not in self.designs:
            raise KeyError(name)
        return self.designs[name]

    def save(self, name):
        with self._lock:
            _save_design_to_disk(self.root, self.get(name))

    def delete(self, name):
        with self._lock:
            design = self.designs.pop(name)  # KeyError if missing
            d = os.path.join(self.root, design.name)
            if os.path.isdir(d):
                shutil.rmtree(d)

    def patch(self, name, x, y, w, h, material=None, dem_delta=None):
        with self._lock:
            return apply_patch(self.base, self.get(name), x, y, w, h, material, dem_delta)

    def set_materials(self, name, materials_json):
        with self._lock:
            design = self.get(name)
            design.materials = materials_json
            design.dirty = True
            design.modified = now_iso()

    def set_lock(self, name, unlocked):
        with self._lock:
            design = self.get(name)
            design.unlocked = unlocked
            design.dirty = True
            design.modified = now_iso()

    def flush_dirty(self):
        with self._lock:
            for design in self.designs.values():
                if design.dirty:
                    try:
                        _save_design_to_disk(self.root, design)
                    except Exception as e:
                        print(f"[sandbox] autosave failed for '{design.name}': {e}")


def _autosave_loop(store, interval=60):
    while True:
        time.sleep(interval)
        store.flush_dirty()


base = Base()
design_store = DesignStore(base)
threading.Thread(target=_autosave_loop, args=(design_store,), daemon=True).start()
