#!/usr/bin/env python3
"""What-if scenario engine: apply polygon edits to the DEM, then hand the
result to flood_sim.py.

A scenario is a JSON file:
{
  "name": "escape_channel",
  "edits": [
    {"op": "lower",      "meters": 0.5,  "polygon": [[x,y],...]},   # carve channel
    {"op": "raise",      "meters": 1.0,  "polygon": [...]},          # fill dirt
    {"op": "wall",       "meters": 1.2,  "polygon": [...]},          # barrier
    {"op": "infiltrate", "mmh": 50,      "polygon": [...]},          # permeable soil
    {"op": "sink",                        "polygon": [...]}          # storm drain
  ]
}
Polygons are UTM (EPSG:32636) coordinates.

Draw polygons interactively:
  python scripts/scenario.py draw --out output/my_edit.json
    (click on the ortho; Enter finishes ONE polygon, prints it for pasting
     into a scenario file, and also saves it)

Apply a scenario and run the simulation:
  python scripts/scenario.py run scenarios/escape_channel.json \
      --rain 30 --duration 3600
Outputs land in output/run_<scenario name>/.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from matplotlib.path import Path as MplPath

sys.path.insert(0, os.path.dirname(__file__))
from las_common import load_transform, pixel_to_utm


def polygon_mask(t, dem_shape, poly):
    h, w = dem_shape
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    ux, uy = pixel_to_utm(t, cols.ravel(), rows.ravel())
    m = MplPath(np.asarray(poly)).contains_points(np.column_stack([ux, uy]))
    return m.reshape(h, w)


def apply_scenario(scn, dem, t):
    """Returns (modified dem, infiltration map mm/h or None, sink mask or None)."""
    dem = dem.copy()
    infil = None
    sink = None
    for e in scn.get("edits", []):
        m = polygon_mask(t, dem.shape, e["polygon"])
        op = e["op"]
        if op == "raise":
            dem[m] += e["meters"]
        elif op == "lower":
            dem[m] -= e["meters"]
        elif op == "wall":
            dem[m] += e["meters"]  # same as raise; kept separate for reporting
        elif op == "infiltrate":
            if infil is None:
                infil = np.zeros(dem.shape, dtype=np.float64)
            infil[m] = e["mmh"]
        elif op == "sink":
            if sink is None:
                sink = np.zeros(dem.shape, dtype=bool)
            sink[m] = True
        else:
            sys.exit(f"unknown op: {op}")
        print(f"  {op:10s} {m.sum():7d} cells"
              + (f"  {e.get('meters', e.get('mmh', ''))}" if op != "sink" else ""))
    return dem, infil, sink


def cmd_draw(args):
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from PIL import Image

    t = load_transform(os.path.join(args.data_dir, "dem_transform.json"))
    img = np.asarray(Image.open(os.path.join(args.data_dir, "ortho.png")))
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img)
    ax.set_title("Click ONE polygon (right-click undo, Enter = done)")
    xs, ys = [], []
    line, = ax.plot([], [], "-o", color="red", lw=1.5, ms=4)

    def on_click(ev):
        if ev.inaxes != ax:
            return
        if ev.button == 1:
            xs.append(ev.xdata); ys.append(ev.ydata)
        elif ev.button == 3 and xs:
            xs.pop(); ys.pop()
        line.set_data(xs + xs[:1], ys + ys[:1])
        fig.canvas.draw_idle()

    def on_key(ev):
        if ev.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    if len(xs) < 3:
        sys.exit("need >= 3 vertices")
    ux, uy = pixel_to_utm(t, np.array(xs), np.array(ys))
    poly = [[round(float(x), 2), round(float(y), 2)] for x, y in zip(ux, uy)]
    print(json.dumps(poly))
    with open(args.out, "w") as f:
        json.dump({"polygon": poly}, f, indent=2)
    print(f"saved to {args.out}")


def cmd_run(args):
    with open(args.scenario) as f:
        scn = json.load(f)
    name = scn.get("name") or os.path.splitext(os.path.basename(args.scenario))[0]
    t = load_transform(os.path.join(args.data_dir, "dem_transform.json"))
    dem = np.load(os.path.join(args.data_dir, "dem.npy")).astype(np.float64)

    print(f"scenario '{name}':")
    dem2, infil, sink = apply_scenario(scn, dem, t)

    scen_dir = os.path.join(args.data_dir, f"scenario_{name}")
    os.makedirs(scen_dir, exist_ok=True)
    dem_p = os.path.join(scen_dir, "dem_mod.npy")
    np.save(dem_p, dem2.astype(np.float32))
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "flood_sim.py"),
           "--dem-override", dem_p,
           "--transform", os.path.join(args.data_dir, "dem_transform.json"),
           "--rain", str(args.rain), "--duration", str(args.duration),
           "--save-every", str(args.save_every),
           "--out", os.path.join(args.data_dir, f"run_{name}")]
    if args.rain_stop:
        cmd += ["--rain-stop", str(args.rain_stop)]
    if infil is not None:
        p = os.path.join(scen_dir, "infil.npy")
        np.save(p, infil)
        cmd += ["--infil", p]
    if sink is not None:
        p = os.path.join(scen_dir, "sink.npy")
        np.save(p, sink)
        cmd += ["--sink", p]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("draw")
    d.add_argument("--out", default="output/edit_polygon.json")
    d.add_argument("--data-dir", default="output")
    d.set_defaults(func=cmd_draw)

    r = sub.add_parser("run")
    r.add_argument("scenario")
    r.add_argument("--data-dir", default="output")
    r.add_argument("--rain", type=float, default=30.0)
    r.add_argument("--rain-stop", type=float, default=None)
    r.add_argument("--duration", type=float, default=3600.0)
    r.add_argument("--save-every", type=float, default=30.0)
    r.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
