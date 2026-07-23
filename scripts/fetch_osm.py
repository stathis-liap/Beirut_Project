#!/usr/bin/env python3
"""Fetch OSM buildings / roads / water for a raster extent via Overpass.

Beirut's building footprints near the port are effectively complete in OSM
(post-2020 HOT activation), so OSM is the authority for building masks.

Reads a stack/terrain transform.json for the bbox, queries Overpass (with
mirror fallback), converts way geometries to EPSG:32636 and writes a
self-contained JSON:

  {"buildings": [[[x,y],...], ...],            closed polygons
   "roads":     [{"coords": [...], "highway": "..."}, ...],
   "water_polys": [...], "water_lines": [...]}

Usage:
  python scripts/fetch_osm.py --transform output/stack_1.0/transform.json \
      --out output/osm/osm.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import requests
from pyproj import Transformer

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
]
HEADERS = {"User-Agent": "beirut-flood-study/1.0 (research; contact via AUB BUL)"}


def query_overpass(q):
    last = None
    for url in ENDPOINTS:
        try:
            r = requests.post(url, data={"data": q}, headers=HEADERS, timeout=300)
            if r.status_code == 200:
                return r.json()
            last = f"{url}: HTTP {r.status_code}"
        except requests.RequestException as e:
            last = f"{url}: {e}"
        print(f"  retrying ({last})")
        time.sleep(3)
    sys.exit(f"all Overpass endpoints failed: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transform", default="output/stack_1.0/transform.json")
    ap.add_argument("--out", default="output/osm/osm.json")
    ap.add_argument("--pad", type=float, default=50.0, help="bbox pad, m")
    args = ap.parse_args()

    with open(args.transform) as f:
        t = json.load(f)
    minx, miny = t["minx"] - args.pad, t["miny"] - args.pad
    maxx = t["minx"] + t["width"] * t["res"] + args.pad
    maxy = t["maxy"] + args.pad

    to_wgs = Transformer.from_crs("EPSG:32636", "EPSG:4326", always_xy=True)
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True)
    w_lon, s_lat = to_wgs.transform(minx, miny)
    e_lon, n_lat = to_wgs.transform(maxx, maxy)
    bbox = f"{s_lat:.6f},{w_lon:.6f},{n_lat:.6f},{e_lon:.6f}"
    print(f"bbox (s,w,n,e): {bbox}")

    q = f"""[out:json][timeout:240];
(
  way["building"]({bbox});
  way["highway"]({bbox});
  way["natural"="water"]({bbox});
  way["water"]({bbox});
  way["waterway"]({bbox});
);
out geom;"""
    print("querying Overpass...")
    data = query_overpass(q)
    els = data.get("elements", [])
    print(f"{len(els)} ways")

    out = {"buildings": [], "roads": [], "water_polys": [], "water_lines": []}
    for el in els:
        geom = el.get("geometry")
        if not geom or len(geom) < 2:
            continue
        lons = np.array([g["lon"] for g in geom])
        lats = np.array([g["lat"] for g in geom])
        xs, ys = to_utm.transform(lons, lats)
        coords = np.round(np.column_stack([xs, ys]), 2).tolist()
        tags = el.get("tags", {})
        closed = geom[0] == geom[-1] and len(geom) >= 4
        if "building" in tags:
            if closed:
                out["buildings"].append(coords)
        elif "highway" in tags:
            out["roads"].append({"coords": coords, "highway": tags["highway"]})
        elif tags.get("natural") == "water" or "water" in tags:
            (out["water_polys"] if closed else out["water_lines"]).append(coords)
        elif "waterway" in tags:
            ww = tags.get("waterway")
            if ww == "riverbank" and closed:
                out["water_polys"].append(coords)
            elif ww in ("river", "canal"):   # not drains/culverts/streams
                out["water_lines"].append(coords)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}: {len(out['buildings'])} buildings, "
          f"{len(out['roads'])} roads, {len(out['water_polys'])} water polys, "
          f"{len(out['water_lines'])} water lines")


if __name__ == "__main__":
    main()
