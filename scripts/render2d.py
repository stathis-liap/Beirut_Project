#!/usr/bin/env python3
"""2D visual products from a run (PLAN.md Phase E).

  animate   MP4 of depth frames over the ortho, with clock + hyetograph
  maxdepth  static max-depth map (color capped at street p99, max annotated)
  hazard    DEFRA-style hazard classes from max h*(|v|+0.5)
  diff      max-depth difference map between two runs (scenario - baseline)

Usage:
  python scripts/render2d.py animate  --run <run> --terrain <terrain> --out x.mp4
  python scripts/render2d.py maxdepth --run <run> --terrain <terrain> --out x.png
  python scripts/render2d.py hazard   --run <run> --terrain <terrain> --out x.png
  python scripts/render2d.py diff --run-a <base> --run-b <scen> --terrain <t> --out x.png
"""

import argparse
import glob
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

HAZ_BOUNDS = [0.75, 1.25, 2.0]
HAZ_COLORS = ["#ffe93d", "#ff9a2e", "#e8442e", "#7d1a9e"]
HAZ_LABELS = ["caution", "danger: children", "danger: most adults", "danger: all"]


def load_ctx(terrain, scale=None):
    ortho = np.asarray(Image.open(os.path.join(terrain, "ortho.png")))
    if scale is None:
        scale = max(1, int(np.ceil(ortho.shape[1] / 2200)))
    return ortho[::scale, ::scale], scale


def street_mask(terrain, scale):
    m = np.load(os.path.join(terrain, "masks.npz"))
    return (m["valid"] & ~m["courtyard"] & ~m["water"])[::scale, ::scale]


def depth_clim(run, terrain, scale):
    md = np.load(os.path.join(run, "max_depth.npy"))[::scale, ::scale]
    wet = md[street_mask(terrain, scale) & (md > 0.05)]
    return (0.0, float(np.clip(np.percentile(wet, 99), 0.25, 1.0))
            if wet.size else 0.5)


def cmd_animate(args):
    import imageio.v2 as imageio
    ortho, scale = load_ctx(args.terrain, args.scale)
    with open(os.path.join(args.run, "run_meta.json")) as f:
        meta = json.load(f)
    frames = sorted(glob.glob(os.path.join(args.run, "depth_*.npy")))
    if not frames:
        raise SystemExit(f"no frames in {args.run} (run without --no-frames)")
    clim = depth_clim(args.run, args.terrain, scale)
    steps = meta["steps_mmh"]
    t_end = meta["duration"]
    i_max = max((s[2] for s in steps), default=1.0)

    fig = plt.figure(figsize=(12.8, 9.2), dpi=100)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(ortho)
    ax.axis("off")
    water_im = ax.imshow(np.full(ortho.shape[:2], np.nan), cmap="Blues",
                         vmin=clim[0], vmax=clim[1], alpha=0.85)
    clock = ax.text(0.015, 0.975, "", transform=ax.transAxes, color="w",
                    fontsize=16, va="top",
                    bbox=dict(facecolor="k", alpha=0.6, pad=6))
    # hyetograph inset
    hx = fig.add_axes([0.70, 0.87, 0.28, 0.11])
    for t0, t1, i in steps:
        hx.bar((t0 + t1) / 120, i, width=(t1 - t0) / 60, color="#9ecbff")
    cursor = hx.axvline(0, color="r", lw=1.5)
    hx.set_xlim(0, t_end / 60)
    hx.set_ylim(0, i_max * 1.15)
    hx.tick_params(labelsize=7, colors="w")
    hx.patch.set_alpha(0.55)
    hx.set_title("rain mm/h", fontsize=8, color="w")

    writer = imageio.get_writer(args.out, fps=args.fps, quality=7,
                                macro_block_size=None)
    for k, fp in enumerate(frames):
        d = np.load(fp).astype(np.float32)[::scale, ::scale]
        water_im.set_data(np.where(d > 0.02, d, np.nan))
        tsec = int(os.path.basename(fp).split("_")[1].split(".")[0])
        clock.set_text(f"t = {tsec // 60:02d}:{tsec % 60:02d}")
        cursor.set_xdata([tsec / 60])
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        # libx264 yuv420p needs even dimensions
        buf = buf[:buf.shape[0] // 2 * 2, :buf.shape[1] // 2 * 2]
        writer.append_data(buf)
        print(f"\r  frame {k + 1}/{len(frames)}", end="", flush=True)
    writer.close()
    plt.close(fig)
    print(f"\nwrote {args.out}")


def cmd_maxdepth(args):
    ortho, scale = load_ctx(args.terrain, args.scale)
    md = np.load(os.path.join(args.run, "max_depth.npy"))[::scale, ::scale]
    st = street_mask(args.terrain, scale)
    clim = depth_clim(args.run, args.terrain, scale)
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(ortho)
    im = ax.imshow(np.where((md > 0.05) & st, md, np.nan), cmap="turbo",
                   vmin=clim[0], vmax=clim[1], alpha=0.8)
    ax.set_title(f"{os.path.basename(args.run)} - max street depth "
                 f"(cap {clim[1]:.2f} m = p99; absolute max {md[st].max():.2f} m)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.7, label="m")
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


def cmd_hazard(args):
    ortho, scale = load_ctx(args.terrain, args.scale)
    hz = np.load(os.path.join(args.run, "max_hazard.npy"))[::scale, ::scale]
    st = street_mask(args.terrain, scale)
    md = np.load(os.path.join(args.run, "max_depth.npy"))[::scale, ::scale]
    cls = np.digitize(hz, HAZ_BOUNDS).astype(float)
    cls[(md < 0.10) | ~st] = np.nan
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(ortho)
    ax.imshow(cls, cmap=ListedColormap(HAZ_COLORS), vmin=-0.5, vmax=3.5,
              alpha=0.85, interpolation="nearest")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in HAZ_COLORS]
    ax.legend(handles, HAZ_LABELS, loc="lower right", fontsize=11,
              title="flood hazard h(v+0.5)")
    ax.set_title(f"{os.path.basename(args.run)} - hazard classes (DEFRA-style)")
    ax.axis("off")
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


def cmd_diff(args):
    ortho, scale = load_ctx(args.terrain, args.scale)
    a = np.load(os.path.join(args.run_a, "max_depth.npy"))[::scale, ::scale]
    b = np.load(os.path.join(args.run_b, "max_depth.npy"))[::scale, ::scale]
    st = street_mask(args.terrain, scale)
    d = np.where(st, b - a, np.nan)
    sig = np.abs(d) > 0.02
    vmax = float(np.percentile(np.abs(d[sig & ~np.isnan(d)]), 99)) if sig.any() else 0.1
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(ortho)
    im = ax.imshow(np.where(sig, d, np.nan), cmap="RdBu_r", vmin=-vmax,
                   vmax=vmax, alpha=0.85)
    helped = float((d < -0.02).sum()) * (scale ** 2)
    hurt = float((d > 0.02).sum()) * (scale ** 2)
    ax.set_title(f"{os.path.basename(args.run_b)} minus "
                 f"{os.path.basename(args.run_a)}   "
                 f"(blue = shallower: {helped:.0f} px improved, {hurt:.0f} worse)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.7, label="depth change (m)")
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, extra in [
            ("animate", cmd_animate, [("--run", True)]),
            ("maxdepth", cmd_maxdepth, [("--run", True)]),
            ("hazard", cmd_hazard, [("--run", True)]),
            ("diff", cmd_diff, [("--run-a", True), ("--run-b", True)])]:
        p = sub.add_parser(name)
        for arg, req in extra:
            p.add_argument(arg, required=req)
        p.add_argument("--terrain", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--scale", type=int, default=None)
        if name == "animate":
            p.add_argument("--fps", type=int, default=12)
        p.set_defaults(func=fn)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
