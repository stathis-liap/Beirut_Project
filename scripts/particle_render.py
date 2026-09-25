#!/usr/bin/env python3
"""PyVista rendering for a particle_sim.py run - terrain plus the particle
cloud colored by speed, same oblique-view convention as render_3d.py.

Usage:
  # offscreen snapshots (usually invoked via particle_sim.py --render)
  python scripts/particle_render.py --run output/particles

  # interactive on-screen viewer, time/speed slider + play/pause, same
  # controls as render_3d.py's flood viewer
  python scripts/particle_render.py --run output/particles --view
"""

import argparse
import glob
import json
import os
import time

import numpy as np


def _bilinear_height(dem, x, y, res):
    """Same bilinear terrain sample as particle_sim.py's sample_height() -
    matters here because the depth grid's own cells are only used to BIN
    particles, not to measure their height above ground; measuring against
    the nearest (truncated) cell instead of the same smooth interpolation
    the physics itself used produces huge spurious depths right at steep
    DEM discontinuities (building walls), where a particle resting exactly
    at the physically-correct interpolated ground level can be several
    cells' worth of height away from its nearest-cell neighbor's raw value.
    """
    h, w = dem.shape
    fx = np.clip(x / res, 0.0, w - 1.001)
    fy = np.clip(y / res, 0.0, h - 1.001)
    c0 = fx.astype(np.int64)
    r0 = fy.astype(np.int64)
    tx, ty = fx - c0, fy - r0
    z00, z10 = dem[r0, c0], dem[r0, c0 + 1]
    z01, z11 = dem[r0 + 1, c0], dem[r0 + 1, c0 + 1]
    z0 = z00 * (1 - tx) + z10 * tx
    z1 = z01 * (1 - tx) + z11 * tx
    return z0 * (1 - ty) + z1 * ty


MAX_RENDER_DEPTH_M = 1.0  # visualization-only safety clip - a small fraction
                          # of particles occasionally get launched several
                          # meters into the air near steep terrain
                          # discontinuities (building walls in this
                          # heightfield-as-mesh representation), a real SPH
                          # instability not yet root-caused. Left unclipped,
                          # ONE such particle spikes the whole rasterized
                          # surface upward at its cell (particles_to_depth
                          # takes a per-cell MAX), which reads as a bizarre
                          # tower rather than water. Clipping here only
                          # affects this rendering path, not the physics or
                          # the saved particle data.


AIRBORNE_HEIGHT_M = 0.08  # a particle higher than this above the LOCAL
                          # terrain is still genuinely falling rain, not
                          # part of the ground-hugging flood sheet - a bit
                          # more than 2x the SPH kernel radius (H=0.044m),
                          # generous enough that a thin flowing layer
                          # bouncing under its own pressure doesn't get
                          # mistaken for still-falling rain. Below this,
                          # "settled" - see particles_to_depth/airborne_particles.


def airborne_particles(pos, dem, res, threshold=AIRBORNE_HEIGHT_M):
    """Splits out the particles still genuinely in flight (see
    AIRBORNE_HEIGHT_M) - lumping these into the same rasterized surface as
    the settled water (particles_to_depth) is what made freshly-falling
    rain look like random spikes poking up out of an otherwise flat
    puddle. Rendered separately as individual drops on top of the flood
    surface, they actually read as "rain still falling into a flood
    that's forming," not as noise.
    """
    if len(pos) == 0:
        return pos
    terrain_z = _bilinear_height(dem, pos[:, 0], pos[:, 1], res)
    return pos[(pos[:, 2] - terrain_z) > threshold]


def particles_to_depth(pos, dem, res):
    """Rasterizes the SETTLED portion of a particle cloud (see
    airborne_particles - still-falling rain is excluded here) onto the
    terrain's own grid as a depth field (per cell: the highest a particle
    reaches above the terrain there), so it can be rendered with
    render_3d.add_water()'s shaded, continuous water-surface mesh instead
    of a scatter of individual points. A few thousand SPH particles read
    as "dust" as discrete dots at any area bigger than a small room, but
    as a rasterized surface they read as what they physically represent:
    an uneven, thin sheet of water - the same visual language the 2D
    solver's depth grid already uses, so toggling between the two doesn't
    also change the rendering style along with the physics.
    """
    h, w = dem.shape
    depth = np.zeros(h * w, dtype=np.float32)
    if len(pos) == 0:
        return depth.reshape(h, w)
    terrain_z = _bilinear_height(dem, pos[:, 0], pos[:, 1], res)
    height = pos[:, 2] - terrain_z
    settled = height <= AIRBORNE_HEIGHT_M
    pos, height = pos[settled], height[settled]
    if len(pos) == 0:
        return depth.reshape(h, w)
    local_depth = np.clip(height, 0.0, MAX_RENDER_DEPTH_M).astype(np.float32)
    col = np.clip((pos[:, 0] / res).astype(np.int64), 0, w - 1)
    row = np.clip((pos[:, 1] / res).astype(np.int64), 0, h - 1)
    idx = row * w + col
    np.maximum.at(depth, idx, local_depth)
    return depth.reshape(h, w)


def render_snapshots(dem, run_dir, frame_names, n_snapshots=6, res=1.0):
    import pyvista as pv
    if not frame_names:
        print("no frames to render")
        return

    h, w = dem.shape
    x, y = np.arange(w) * res, np.arange(h) * res
    xx, yy = np.meshgrid(x, y)
    terrain = pv.StructuredGrid(xx, yy, dem)

    # Frame the camera on where the water actually goes, not the whole
    # terrain - a shallow flow can be a tiny fraction of a large crop's
    # footprint, invisible at a whole-scene zoom level.
    all_pts = [np.load(os.path.join(run_dir, f))["pos"] for f in frame_names]
    all_pts = [p for p in all_pts if len(p)]
    if all_pts:
        pts_all = np.concatenate(all_pts)
        margin = 5.0
        cx = (pts_all[:, 0].min() + pts_all[:, 0].max()) / 2
        cy = (pts_all[:, 1].min() + pts_all[:, 1].max()) / 2
        cz = float(dem[np.clip(int(cy / res), 0, h - 1), np.clip(int(cx / res), 0, w - 1)])
        span = max(np.ptp(pts_all[:, 0]), np.ptp(pts_all[:, 1])) + 2 * margin
        cam_pos = (cx, cy - span * 0.9, cz + span * 0.7)
        cam_focal = (cx, cy, cz)
    else:
        cam_pos, cam_focal = None, None

    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=[1200, 900])
    pl.add_mesh(terrain, color="lightgray", opacity=0.6, name="terrain")
    pl.set_background("black")
    if cam_pos:
        pl.camera_position = [cam_pos, cam_focal, (0, 0, 1)]
    else:
        pl.camera_position = "xy"
        pl.camera.elevation = -55

    idxs = np.linspace(0, len(frame_names) - 1, min(n_snapshots, len(frame_names))).astype(int)
    for k, i in enumerate(idxs):
        data = np.load(os.path.join(run_dir, frame_names[i]))
        pts, speed = data["pos"], data["speed"]
        try:
            pl.remove_actor("particles", render=False)
        except Exception:
            pass
        if len(pts):
            cloud = pv.PolyData(pts)
            cloud["speed"] = speed
            pl.add_mesh(cloud, scalars="speed", cmap="turbo", point_size=3,
                        render_points_as_spheres=True, name="particles",
                        clim=(0.0, max(1.0, float(speed.max()))),
                        show_scalar_bar=True,
                        scalar_bar_args={"title": "speed (m/s)", "color": "white"})
        out_png = os.path.join(run_dir, f"snapshot_{k:02d}.png")
        pl.screenshot(out_png)
        print(f"wrote {out_png}")
    pl.close()


def _frame_time(fname):
    # particles_000175.npz -> t=1.75s (particle_sim.py names frames by
    # round(t * 100), i.e. centiseconds)
    return int(os.path.basename(fname).split("_")[1].split(".")[0]) / 100.0


def interactive_view(dem, run_dir, frame_names, res=1.0, z_exagg=1.0):
    """On-screen PyVista viewer with a continuous time slider, speed slider
    and play/pause - same controls as render_3d.py's flood viewer.

    particle_sim.py only saves a frame every `save_every` seconds (0.25s by
    default, i.e. 4 fps) - stepping straight to the nearest saved frame
    would look visibly choppy for water moving several m/s. Particle
    indices are stable across frames (particle_sim.py only ever appends,
    never reorders or removes), so instead this linearly interpolates each
    particle's position between the two bracketing saved frames for smooth
    playback without having to save every physics substep.
    """
    import pyvista as pv

    if not frame_names:
        print("no frames to view")
        return

    times = np.array([_frame_time(f) for f in frame_names], dtype=np.float64)
    frames = [np.load(os.path.join(run_dir, f)) for f in frame_names]
    positions = [f["pos"] for f in frames]
    speeds = [f["speed"] for f in frames]

    h, w = dem.shape
    x, y = np.arange(w) * res, np.arange(h) * res
    xx, yy = np.meshgrid(x, y)
    terrain = pv.StructuredGrid(xx, yy, dem * z_exagg)

    all_pts = [p for p in positions if len(p)]
    if all_pts:
        pts_all = np.concatenate(all_pts)
        margin = 2.0
        cx = (pts_all[:, 0].min() + pts_all[:, 0].max()) / 2
        cy = (pts_all[:, 1].min() + pts_all[:, 1].max()) / 2
        cz = float(dem[np.clip(int(cy / res), 0, h - 1), np.clip(int(cx / res), 0, w - 1)])
        span = max(np.ptp(pts_all[:, 0]), np.ptp(pts_all[:, 1])) + 2 * margin
        cam_pos = (cx, cy - span * 0.9, cz + span * 0.7)
        cam_focal = (cx, cy, cz)
    else:
        cam_pos, cam_focal = None, None
    speed_max = max(1.0, float(max((s.max() for s in speeds if len(s)), default=1.0)))

    pl = pv.Plotter()
    pl.add_mesh(terrain, color="lightgray", opacity=0.7, name="terrain")
    pl.set_background("black")
    if cam_pos:
        pl.camera_position = [cam_pos, cam_focal, (0, 0, 1)]
    else:
        pl.camera_position = "xy"
        pl.camera.elevation = -55

    state = {"t": float(times[0]), "playing": False, "speed": 1.0}

    def interpolate(t):
        t = min(max(t, times[0]), times[-1])
        i = int(np.searchsorted(times, t, side="right") - 1)
        i = min(max(i, 0), len(times) - 2) if len(times) > 1 else 0
        t0, t1 = times[i], times[min(i + 1, len(times) - 1)]
        alpha = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        p0, p1 = positions[i], positions[min(i + 1, len(positions) - 1)]
        s0, s1 = speeds[i], speeds[min(i + 1, len(speeds) - 1)]
        n_shared = min(len(p0), len(p1))  # particle indices are stable/append-only
        pos = p0[:n_shared] * (1 - alpha) + p1[:n_shared] * alpha
        spd = s0[:n_shared] * (1 - alpha) + s1[:n_shared] * alpha
        n_new = len(p1) - n_shared
        if n_new > 0:
            # particles born during this interval: index order == birth
            # order (emission only ever appends), so stagger their
            # appearance across the interval instead of popping all of
            # them in at once at t0
            born = (np.arange(n_new) + 1) / n_new <= alpha
            if born.any():
                pos = np.concatenate([pos, p1[n_shared:][born]])
                spd = np.concatenate([spd, s1[n_shared:][born]])
        return pos, spd

    def show_at(t):
        state["t"] = t
        pos, spd = interpolate(t)
        try:
            pl.remove_actor("particles", render=False)
        except Exception:
            pass
        if len(pos):
            cloud = pv.PolyData(pos)
            cloud["speed"] = spd
            pl.add_mesh(cloud, scalars="speed", cmap="turbo", point_size=3,
                        render_points_as_spheres=True, name="particles",
                        clim=(0.0, speed_max), show_scalar_bar=True,
                        scalar_bar_args={"title": "speed (m/s)", "color": "white"},
                        render=False)
        pl.add_text(f"t = {t:5.2f}s", name="clock", color="white",
                    font_size=14, position="upper_right", render=False)
        pl.render()

    def on_time_slider(value):
        show_at(value)

    def on_speed_slider(value):
        state["speed"] = value

    def set_playstate(playing):
        state["playing"] = playing
        pl.add_text("PLAYING" if playing else "PAUSED",
                    name="playstate", color="yellow" if playing else "white",
                    font_size=10, position="lower_right")

    time_slider = pl.add_slider_widget(on_time_slider, rng=[times[0], times[-1]], value=times[0],
                                       title="Time (s)", pointa=(0.25, 0.9), pointb=(0.8, 0.9),
                                       style="modern", interaction_event="always")
    pl.add_slider_widget(on_speed_slider, rng=[0.25, 4.0], value=1.0, title="Speed",
                        pointa=(0.03, 0.9), pointb=(0.2, 0.9), style="modern",
                        interaction_event="always")
    pl.add_checkbox_button_widget(set_playstate, value=False, position=(10, 10),
                                  size=40, color_on="yellow", color_off="grey")
    pl.add_text("Play/Pause", position=(58, 20), font_size=10, color="white")
    pl.add_text("Click button: play/pause    drag sliders: scrub time / speed",
               position="lower_left", font_size=10, color="white")
    set_playstate(False)
    show_at(times[0])

    # same wall-clock-driven loop as render_3d.py's flood viewer (VTK's own
    # timer machinery proved unreliable there) - here it's simpler still
    # since playback time is continuous, not a frame index, so advancing it
    # is just state["t"] += dt * speed and interpolate() handles the rest.
    pl.show(auto_close=False, interactive_update=True)
    last_t = time.time()
    while not pl._closed:
        now = time.time()
        dt = now - last_t
        last_t = now
        if state["playing"]:
            t = state["t"] + dt * state["speed"]
            if t >= times[-1]:
                t = times[0]  # loop back to the start
            show_at(t)
            time_slider.GetRepresentation().SetValue(t)
        pl.update()
        time.sleep(0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=6, help="snapshot mode: number of PNGs")
    ap.add_argument("--view", action="store_true",
                    help="interactive on-screen viewer instead of offscreen snapshots")
    args = ap.parse_args()

    dem = np.load(os.path.join(args.run, "dem_crop.npy"))
    frames = sorted(os.path.basename(p)
                     for p in glob.glob(os.path.join(args.run, "particles_*.npz")))
    meta_path = os.path.join(args.run, "meta.json")
    res = 1.0
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            res = json.load(f).get("res", 1.0)

    if args.view:
        interactive_view(dem, args.run, frames, res=res)
    else:
        render_snapshots(dem, args.run, frames, n_snapshots=args.n, res=res)


if __name__ == "__main__":
    main()
