"""Render flood simulation results in 3D with PyVista (offscreen).

  # interactive on-screen viewer (rotate/zoom with the mouse) - true ortho
  # view: straight-down camera + parallel projection, a map-like look
  python scripts/render_3d.py view --data-dir terrain
  # + water, with a time slider to scrub the storm and space to play/pause
  python scripts/render_3d.py view --data-dir terrain --run sim/run_baseline
  # Ctrl+Click anywhere on the terrain: pop a rain-drop SPH simulation of
  # that area in a new window (see enable_click_to_reconstruct)

  # animated MP4 of one run
  python scripts/render_3d.py video --run output/run_baseline --out output/baseline.mp4

  # static max-depth comparison (baseline vs scenario)
  python scripts/render_3d.py compare --runs output/run_baseline output/run_channel \
      --out output/compare.png

  # 2D max-depth heatmap over the ortho (fast fallback)
  python scripts/render_3d.py heatmap --run output/run_baseline --out output/heat.png
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def load_common(data_dir):
    from PIL import Image
    dem = np.load(os.path.join(data_dir, "dem.npy")).astype(np.float32)
    ortho = np.asarray(Image.open(os.path.join(data_dir, "ortho.png")))
    with open(os.path.join(data_dir, "dem_transform.json")) as f:
        t = json.load(f)
    return dem, ortho, t


def make_terrain(dem, res, z_exagg=1.0):
    import pyvista as pv
    h, w = dem.shape
    x = np.arange(w) * res
    y = np.arange(h) * res
    xx, yy = np.meshgrid(x, y)
    grid = pv.StructuredGrid(xx, yy, dem * z_exagg)
    # texture coordinates for draping the ortho
    grid.active_texture_coordinates = np.column_stack([
        (xx.ravel(order="F") / x.max()),
        1.0 - (yy.ravel(order="F") / y.max()),
    ]).astype(np.float32)
    return grid


def water_mesh(dem, depth, res, z_exagg=1.0, min_depth=0.02):
    import pyvista as pv
    h, w = dem.shape
    x = np.arange(w) * res
    y = np.arange(h) * res
    xx, yy = np.meshgrid(x, y)
    surf = np.where(depth > min_depth, dem + depth, np.nan)
    grid = pv.StructuredGrid(xx, yy, surf * z_exagg)
    grid["depth"] = depth.ravel(order="F")
    return grid.threshold(min_depth, scalars="depth")


def setup_plotter(dem, ortho, res, z_exagg, window=(1600, 1000)):
    import pyvista as pv
    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=list(window))
    terrain = make_terrain(dem, res, z_exagg)
    tex = pv.numpy_to_texture(np.ascontiguousarray(ortho[::1]))
    pl.add_mesh(terrain, texture=tex, name="terrain")
    pl.set_background("black")
    return pl


def add_water(pl, dem, depth, res, z_exagg, clim=(0.0, 1.0), render=True):
    wm = water_mesh(dem, depth, res, z_exagg)
    if wm.n_points > 0:
        pl.add_mesh(wm, scalars="depth", cmap="Blues", clim=clim,
                    opacity=0.75, name="water", show_scalar_bar=True,
                    scalar_bar_args={"title": "depth (m)", "color": "white"},
                    render=render)
    return wm


RECONSTRUCT_RADIUS_KM = 0.75    # calibrate via --reconstruct-radius (CLI) or
                                 # the GUI's "Reconstruction radius" field -
                                 # 1.5km (the original default) was too much
                                 # for most use; this is just a starting point
RECONSTRUCT_RAIN_MMH = 30.0      # rain intensity fed to the clicked-area SPH sim
RECONSTRUCT_DURATION_S = 10.0    # simulated seconds of rain for that sim


def write_seed_file(t, flood, utm_x, utm_y, radius_m, out_dir):
    """Crop whatever water depth is CURRENTLY on screen in this viewer
    (flood["state"]["idx"] - wherever the time slider/playback happens to
    be, not necessarily frame 0) around the click point, and save it as a
    seed file particle_sim.py's area_rain scene can resample onto its own,
    much finer local grid (see build_seed_particles there).

    Without this, every Ctrl+Click restarts from bone-dry ground regardless
    of how much the 2D macro solver has already flooded that exact spot -
    the two simulations agreeing on "how wet is it here right now" is the
    actual point of feeding one into the other, not just a nice-to-have.
    Returns None (no file written) if there's nothing worth seeding, so the
    caller can fall back to the old dry-start behavior.
    """
    from las_common import utm_to_pixel

    idx = flood["state"]["idx"]
    depth_now = flood["depths"][idx]
    res = t["res"]
    col0, row0 = utm_to_pixel(t, utm_x - radius_m, utm_y + radius_m)
    col1, row1 = utm_to_pixel(t, utm_x + radius_m, utm_y - radius_m)
    row0, row1 = int(max(0, row0)), int(min(depth_now.shape[0], round(row1)))
    col0, col1 = int(max(0, col0)), int(min(depth_now.shape[1], round(col1)))
    if row1 <= row0 or col1 <= col0:
        return None
    crop = depth_now[row0:row1, col0:col1]
    if not np.any(crop > 0.02):
        print("[click] no standing water at this point/time in the macro run - starting dry")
        return None

    os.makedirs(out_dir, exist_ok=True)
    seed_path = os.path.join(out_dir, "seed.npz")
    np.savez(seed_path, depth=crop.astype(np.float32), res=res,
             minx=t["minx"] + col0 * res, maxy=t["maxy"] - row0 * res)
    print(f"[click] seeding local sim from the macro run's water at t={flood['times'][idx]}s "
          f"(max depth in crop: {float(crop.max()):.2f} m)")
    return seed_path


def enable_click_to_reconstruct(pl, t, radius_km=RECONSTRUCT_RADIUS_KM,
                                rain_mmh=RECONSTRUCT_RAIN_MMH,
                                rain_duration_s=RECONSTRUCT_DURATION_S,
                                flood=None):
    """Ctrl+Left-click anywhere on the terrain: pop a focused, lightweight
    rain/SPH particle simulation of a small area (a `radius_km`-around-the-
    click crop, streamed straight from the source .las - see
    reconstruct_area.py + particle_sim.py's area_rain scene) in a brand new
    window. This viewer stays showing the coarse overview it already
    loaded; the close-up is a separate process so neither one has to hold
    both the overview and a fine-grained point-cloud crop in memory at once.

    `flood`, if given (see cmd_view), is {"times", "depths", "state"} from
    the SAME 2D flood run currently animating in this same window - it's
    what lets the click hand the local particle sim a real starting water
    state instead of always starting from dry ground (see write_seed_file).
    """
    import subprocess
    import vtk

    radius_m = radius_km * 1000.0
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    from las_common import resolve_las_path

    picker = vtk.vtkCellPicker()
    picker.SetTolerance(0.005)

    def on_left_press(obj, event):
        interactor = pl.iren.interactor
        ctrl = bool(interactor.GetControlKey())
        print(f"[click] left-button press, ctrl={ctrl}")
        if not ctrl:
            return
        x, y = interactor.GetEventPosition()
        picker.Pick(x, y, 0, pl.renderer)
        if picker.GetCellId() == -1:
            print("[click] ctrl+click registered but missed the terrain (no cell under cursor)")
            return
        wx, wy, _wz = picker.GetPickPosition()
        utm_x, utm_y = t["minx"] + wx, t["maxy"] - wy

        las_path = resolve_las_path(t.get("source_las"), data_dir)
        if not las_path:
            print(f"[click] no .las file found to reconstruct from "
                  f"(checked {t.get('source_las')!r} and {data_dir})")
            return
        print(f"[click] Ctrl+click at UTM ({utm_x:.1f}, {utm_y:.1f}) - running a rain "
              f"simulation on the {radius_km:.2f}km area around it in a new window...")
        # each click needs its own --out - without one, every click would
        # overwrite the same default output/particles dir
        import datetime
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(project_root, "output", f"area_rain_{stamp}")

        cmd = [sys.executable,
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "particle_sim.py"),
              "--scene", "area_rain", "--utm-x", str(utm_x), "--utm-y", str(utm_y),
              "--radius", str(radius_m), "--rain", str(rain_mmh),
              "--duration", str(rain_duration_s), "--out", out_dir, "--view"]
        if flood is not None:
            seed_path = write_seed_file(t, flood, utm_x, utm_y, radius_m, out_dir)
            if seed_path:
                cmd += ["--seed", seed_path]
        subprocess.Popen(cmd)

    # pl.iren.add_observer (not the raw pl.iren.interactor.AddObserver) matters
    # here: it wraps the callback so exceptions actually surface as a visible
    # warning instead of vanishing into VTK's C++/Python boundary - this is
    # exactly the mechanism PyVista's own enable_point_picking(left_clicking=True)
    # uses internally for LeftButtonPressEvent, so it's the well-tested path.
    pl.iren.add_observer("LeftButtonPressEvent", on_left_press)
    pl.add_text(f"Ctrl+Click: rain simulation on this area ({radius_km:.2g}km radius, new window)",
               name="reconstruct_hint", position="upper_left", font_size=9, color="white")


def cmd_view(args):
    import pyvista as pv
    dem, ortho, t = load_common(args.data_dir)
    res = t["res"]

    pl = pv.Plotter()
    terrain = make_terrain(dem, res, args.z_exagg)
    tex = pv.numpy_to_texture(np.ascontiguousarray(ortho))
    pl.add_mesh(terrain, texture=tex, name="terrain")
    pl.set_background("black")
    # true ortho view: straight-down camera (no oblique .elevation tilt) +
    # parallel/orthographic projection (no perspective foreshortening) - a
    # real map-like view matching the ortho.png texture it's draped with,
    # instead of the previous tilted perspective flyover angle
    pl.camera_position = "xy"
    pl.enable_parallel_projection()

    # load the flood run's playback data (if any) BEFORE wiring up the click
    # handler, not after - enable_click_to_reconstruct needs live access to
    # "what water depth is on screen right now" (state["idx"]) so Ctrl+Click
    # can seed the local particle sim from the macro solver's actual state
    # instead of always starting from dry ground (see write_seed_file).
    flood = None
    if args.run:
        frame_paths = sorted(glob.glob(os.path.join(args.run, "depth_*.npy")))
        if not frame_paths:
            sys.exit(f"no depth_*.npy in {args.run}")
        times = [int(os.path.basename(fp).split("_")[1].split(".")[0]) for fp in frame_paths]
        depths = [np.load(fp).astype(np.float32) for fp in frame_paths]
        max_depth = np.load(os.path.join(args.run, "max_depth.npy")).astype(np.float32)
        clim = (0.0, max(0.5, float(max_depth.max()) * 0.8))
        state = {"idx": 0, "playing": False, "speed": 1.0, "progress": 0.0}
        flood = {"times": times, "depths": depths, "state": state}

    enable_click_to_reconstruct(pl, t, radius_km=args.reconstruct_radius,
                                rain_mmh=args.reconstruct_rain,
                                rain_duration_s=args.reconstruct_duration,
                                flood=flood)

    if not flood:
        pl.show()
        return

    times, depths, state = flood["times"], flood["depths"], flood["state"]

    def show_frame(idx):
        # remove_actor() and add_mesh() each render by default - doing both
        # per frame flashed an empty-water frame in between (the "restarts
        # from scratch" flicker). Suppress those and render once at the end.
        idx = max(0, min(len(depths) - 1, idx))
        state["idx"] = idx
        try:
            pl.remove_actor("water", render=False)
        except Exception:
            pass
        add_water(pl, dem, depths[idx], res, args.z_exagg, clim, render=False)
        mm, ss = divmod(times[idx], 60)
        pl.add_text(f"t = {mm:02d}:{ss:02d}", name="clock", color="white",
                    font_size=14, position="upper_right", render=False)
        pl.render()

    def on_time_slider(value):
        idx = min(range(len(times)), key=lambda i: abs(times[i] - value))
        show_frame(idx)

    def on_speed_slider(value):
        state["speed"] = value

    def set_playstate(playing):
        state["playing"] = playing
        pl.add_text("PLAYING" if playing else "PAUSED",
                    name="playstate", color="yellow" if playing else "white",
                    font_size=10, position="lower_right")

    # both sliders live in the header strip (top of the window) so they
    # don't collide with the depth color-scale bar PyVista draws along
    # the bottom for the water mesh's scalar_bar.
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
    show_frame(0)

    # VTK's own timer/animation machinery (add_timer_event, repeating
    # CreateRepeatingTimer via TimerEvent observers) proved unreliable here -
    # either never firing or badly delayed depending on the interactor.
    # Driving the animation from a plain Python loop around interactive_update
    # sidesteps that: pl.update() both redraws and processes mouse/button
    # input each iteration, and frame advancement is paced by real elapsed
    # wall-clock time, so it self-corrects if a render happens to be slow
    # instead of silently falling behind.
    FRAMES_PER_SEC = 12.0
    pl.show(auto_close=False, interactive_update=True)
    last_t = time.time()
    while not pl._closed:
        now = time.time()
        dt = now - last_t
        last_t = now
        if state["playing"]:
            state["progress"] += dt * state["speed"] * FRAMES_PER_SEC
            if state["progress"] >= 1.0:
                advance = int(state["progress"])
                state["progress"] -= advance
                nxt = state["idx"] + advance
                if nxt >= len(depths):
                    nxt = 0  # loop back to the start of the storm
                show_frame(nxt)
                time_slider.GetRepresentation().SetValue(times[nxt])
        pl.update()
        time.sleep(0.02)


def cmd_area_view(args):
    """Unified viewer for an area_sim.py output dir: one terrain, one time/
    speed control, and a toggle between the 2D grid solver's depth surface
    and the 3D SPH solver's particle cloud (rasterized into the same kind
    of shaded surface via particle_render.particles_to_depth - see there
    for why, not just rendered as raw points).

    The time slider is a 0..1 fraction of whichever mode is currently
    active, not raw seconds - the two runs cover very different real
    durations (the 2D grid can run the whole storm, the SPH run is capped
    to a short window, see area_sim.DEFAULT_PARTICLE_DURATION_S) so one
    slider can't sensibly share a single seconds axis between them. The
    actual elapsed time for whichever mode is showing is in the on-screen
    clock text instead.
    """
    import pyvista as pv
    from particle_render import particles_to_depth, airborne_particles

    out_dir = args.dir
    dem, ortho, t = load_common(os.path.join(out_dir, "_terrain"))
    res = t["res"]

    dir_2d = os.path.join(out_dir, "grid2d")
    frame_paths_2d = sorted(glob.glob(os.path.join(dir_2d, "depth_*.npy")))
    if not frame_paths_2d:
        sys.exit(f"no depth_*.npy in {dir_2d}")
    times_2d = [int(os.path.basename(fp).split("_")[1].split(".")[0]) for fp in frame_paths_2d]
    depths_2d = [np.load(fp).astype(np.float32) for fp in frame_paths_2d]
    max_depth_2d = np.load(os.path.join(dir_2d, "max_depth.npy")).astype(np.float32)
    clim_2d = (0.0, max(0.3, float(max_depth_2d.max()) * 0.8))

    dir_3d = os.path.join(out_dir, "particles3d")
    frame_paths_3d = sorted(glob.glob(os.path.join(dir_3d, "particles_*.npz")))
    if not frame_paths_3d:
        sys.exit(f"no particles_*.npz in {dir_3d}")
    times_3d = [int(os.path.basename(fp).split("_")[1].split(".")[0]) / 100.0
               for fp in frame_paths_3d]
    # each 3D frame renders as TWO layers, not one: the settled/flowing
    # water as a shaded surface (particles_to_depth), and whatever's still
    # genuinely falling as individual drops (airborne_particles) - lumping
    # both into one surface is what made falling rain look like random
    # spikes; this split is what actually reads as "rain forming a flood"
    # rather than either raw confetti or a mysteriously-appearing puddle.
    pos_3d = [np.load(fp)["pos"] for fp in frame_paths_3d]
    depths_3d = [particles_to_depth(p, dem, res) for p in pos_3d]
    rain_3d = [airborne_particles(p, dem, res) for p in pos_3d]
    max_3d = max((float(d.max()) for d in depths_3d if d.size), default=0.3)
    clim_3d = (0.0, max(0.1, max_3d * 0.8))

    # NOT an explicit window_size - the other two interactive (on-screen)
    # viewers in this project (cmd_view here, particle_render.interactive_view)
    # both use a bare pv.Plotter() and work fine; this one crashed with
    # "Hardware does not support the number of textures defined" / "Could not
    # create shader object" the one time it asked for an explicit large
    # on-screen size. Offscreen renders elsewhere DO pass window_size safely
    # (see render_snapshots, setup_plotter) - that's a different code path
    # (renders to an off-screen framebuffer), not evidence this is safe for
    # an actual on-screen window too.
    pl = pv.Plotter()
    terrain = make_terrain(dem, res, args.z_exagg)
    tex = pv.numpy_to_texture(np.ascontiguousarray(ortho))
    pl.add_mesh(terrain, texture=tex, name="terrain")
    pl.set_background("black")
    pl.camera_position = "xy"
    pl.camera.elevation = -55
    # camera_position="xy" auto-fits the terrain for a straight-down view;
    # tilting it afterward via .elevation rotates around that same fitted
    # distance, which can push part of the terrain outside the viewport at
    # the new, more foreshortened oblique angle - reset_camera() re-fits the
    # zoom/distance for the CURRENT view direction so the whole local area
    # is actually visible, not just whatever fit the top-down preset.
    pl.reset_camera()

    state = {"mode": "2d", "idx2d": 0, "idx3d": 0, "playing": False,
            "speed": 1.0, "progress": 0.0}

    def show_2d(idx, render=True):
        idx = max(0, min(len(depths_2d) - 1, idx))
        state["idx2d"] = idx
        try:
            pl.remove_actor("water", render=False)
            pl.remove_actor("rain", render=False)  # in case 3D mode left this showing
        except Exception:
            pass
        add_water(pl, dem, depths_2d[idx], res, args.z_exagg, clim_2d, render=False)
        mm, ss = divmod(times_2d[idx], 60)
        pl.add_text(f"2D grid solver   t = {mm:02d}:{ss:02d}", name="clock", color="white",
                    font_size=13, position="upper_right", render=False)
        if render:
            pl.render()

    def show_3d(idx, render=True):
        idx = max(0, min(len(depths_3d) - 1, idx))
        state["idx3d"] = idx
        try:
            pl.remove_actor("water", render=False)
            pl.remove_actor("rain", render=False)
        except Exception:
            pass
        add_water(pl, dem, depths_3d[idx], res, args.z_exagg, clim_3d, render=False)
        rain_pts = rain_3d[idx]
        if len(rain_pts):
            cloud = pv.PolyData((rain_pts * [1, 1, args.z_exagg]).astype(np.float32))
            pl.add_mesh(cloud, color="#bfe3ff", point_size=4, opacity=0.85,
                       render_points_as_spheres=True, name="rain", render=False)
        pl.add_text(f"3D particle solver   t = {times_3d[idx]:5.2f}s", name="clock", color="white",
                    font_size=13, position="upper_right", render=False)
        if render:
            pl.render()

    def show_active(render=True):
        if state["mode"] == "2d":
            show_2d(state["idx2d"], render=render)
        else:
            show_3d(state["idx3d"], render=render)

    def on_time_slider(value):
        if state["mode"] == "2d":
            show_2d(int(round(value * (len(depths_2d) - 1))))
        else:
            show_3d(int(round(value * (len(depths_3d) - 1))))

    def on_speed_slider(value):
        state["speed"] = value

    def set_mode(is_3d):
        state["mode"] = "3d" if is_3d else "2d"
        state["progress"] = 0.0
        show_active()

    def set_playstate(playing):
        state["playing"] = playing
        pl.add_text("PLAYING" if playing else "PAUSED",
                    name="playstate", color="yellow" if playing else "white",
                    font_size=10, position="lower_right")

    time_slider = pl.add_slider_widget(on_time_slider, rng=[0.0, 1.0], value=0.0,
                                       title="Time", pointa=(0.25, 0.9), pointb=(0.8, 0.9),
                                       style="modern", interaction_event="always")
    pl.add_slider_widget(on_speed_slider, rng=[0.25, 4.0], value=1.0, title="Speed",
                        pointa=(0.03, 0.9), pointb=(0.2, 0.9), style="modern",
                        interaction_event="always")
    pl.add_checkbox_button_widget(set_playstate, value=False, position=(10, 10),
                                  size=40, color_on="yellow", color_off="grey")
    pl.add_text("Play/Pause", position=(58, 20), font_size=10, color="white")
    pl.add_checkbox_button_widget(set_mode, value=False, position=(10, 60),
                                  size=40, color_on="#4fc3f7", color_off="grey")
    pl.add_text("Toggle: 2D grid (off) / 3D particles (on)",
               position=(58, 70), font_size=10, color="white")
    pl.add_text("Click button: play/pause    2nd button: 2D/3D solver    "
               "drag sliders: scrub time / speed",
               position="lower_left", font_size=10, color="white")
    set_playstate(False)
    show_active()

    FRAMES_PER_SEC = 8.0
    pl.show(auto_close=False, interactive_update=True)
    last_t = time.time()
    while not pl._closed:
        now = time.time()
        dt = now - last_t
        last_t = now
        if state["playing"]:
            n = len(depths_2d) if state["mode"] == "2d" else len(depths_3d)
            state["progress"] += dt * state["speed"] * FRAMES_PER_SEC
            if state["progress"] >= 1.0:
                advance = int(state["progress"])
                state["progress"] -= advance
                cur = state["idx2d"] if state["mode"] == "2d" else state["idx3d"]
                idx = cur + advance
                if idx >= n:
                    idx = 0
                if state["mode"] == "2d":
                    show_2d(idx)
                else:
                    show_3d(idx)
                time_slider.GetRepresentation().SetValue(idx / max(1, n - 1))
        pl.update()
        time.sleep(0.02)


def cmd_video(args):
    import imageio.v2 as imageio
    dem, ortho, t = load_common(args.data_dir)
    if args.dem_override and os.path.exists(args.dem_override):
        dem = np.load(args.dem_override).astype(np.float32)
    res = t["res"]
    frames = sorted(glob.glob(os.path.join(args.run, "depth_*.npy")))
    if not frames:
        sys.exit(f"no depth_*.npy in {args.run}")
    print(f"{len(frames)} frames")

    clim = (0.0, max(0.5, float(np.load(
        os.path.join(args.run, "max_depth.npy")).max()) * 0.8))

    pl = setup_plotter(dem, ortho, res, args.z_exagg)
    pl.camera_position = "xy"
    pl.camera.elevation = -55  # oblique view
    pl.camera.zoom(args.zoom)

    writer = imageio.get_writer(args.out, fps=args.fps, quality=8)
    for i, fp in enumerate(frames):
        depth = np.load(fp).astype(np.float32)
        try:
            pl.remove_actor("water")
        except Exception:
            pass
        add_water(pl, dem, depth, res, args.z_exagg, clim)
        tsec = int(os.path.basename(fp).split("_")[1].split(".")[0])
        pl.add_text(f"t = {tsec // 60:02d}:{tsec % 60:02d}", name="clock",
                    color="white", font_size=14)
        img = pl.screenshot(return_img=True)
        writer.append_data(img)
        print(f"\r  frame {i + 1}/{len(frames)}", end="", flush=True)
    writer.close()
    pl.close()
    print(f"\nwrote {args.out}")


def cmd_compare(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dem, ortho, t = load_common(args.data_dir)
    n = len(args.runs)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 8))
    axes = np.atleast_1d(axes)
    vmax = max(float(np.load(os.path.join(r, "max_depth.npy")).max())
               for r in args.runs)
    vmax = min(vmax, 2.0)
    for ax, r in zip(axes, args.runs):
        md = np.load(os.path.join(r, "max_depth.npy"))
        ax.imshow(ortho)
        im = ax.imshow(np.where(md > 0.05, md, np.nan), cmap="turbo",
                       vmin=0, vmax=vmax, alpha=0.8)
        with open(os.path.join(r, "run_meta.json")) as f:
            meta = json.load(f)
        ax.set_title(f"{os.path.basename(r)}  (rain {meta['rain_mmh']:.0f} mm/h)\n"
                     f"max depth {md.max():.2f} m")
        ax.axis("off")
    fig.colorbar(im, ax=axes.tolist(), label="max water depth (m)", shrink=0.7)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


def cmd_heatmap(args):
    args.runs = [args.run]
    cmd_compare(args)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    vw = sub.add_parser("view")
    vw.add_argument("--data-dir", default="output")
    vw.add_argument("--run", help="optional: overlay max_depth water surface from a sim run")
    vw.add_argument("--z-exagg", type=float, default=1.0)
    vw.add_argument("--reconstruct-radius", type=float, default=RECONSTRUCT_RADIUS_KM,
                    help="km - Ctrl+Click reconstruction crop radius (default "
                         f"{RECONSTRUCT_RADIUS_KM}km)")
    vw.add_argument("--reconstruct-rain", type=float, default=RECONSTRUCT_RAIN_MMH,
                    help=f"mm/h - Ctrl+Click rain intensity (default {RECONSTRUCT_RAIN_MMH})")
    vw.add_argument("--reconstruct-duration", type=float, default=RECONSTRUCT_DURATION_S,
                    help=f"seconds of simulated rain (default {RECONSTRUCT_DURATION_S})")
    vw.set_defaults(func=cmd_view)

    v = sub.add_parser("video")
    v.add_argument("--run", required=True)
    v.add_argument("--data-dir", default="output")
    v.add_argument("--dem-override", help="scenario dem_mod.npy for correct terrain")
    v.add_argument("--out", required=True)
    v.add_argument("--fps", type=int, default=12)
    v.add_argument("--zoom", type=float, default=1.3)
    v.add_argument("--z-exagg", type=float, default=1.0)
    v.set_defaults(func=cmd_video)

    c = sub.add_parser("compare")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--data-dir", default="output")
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_compare)

    hm = sub.add_parser("heatmap")
    hm.add_argument("--run", required=True)
    hm.add_argument("--data-dir", default="output")
    hm.add_argument("--out", required=True)
    hm.set_defaults(func=cmd_heatmap)

    av = sub.add_parser("area-view", help="2D grid / 3D particle toggle viewer for an "
                        "area_sim.py output dir")
    av.add_argument("--dir", required=True, help="area_sim.py --out directory "
                    "(contains _terrain/, grid2d/, particles3d/)")
    av.add_argument("--z-exagg", type=float, default=1.0)
    av.set_defaults(func=cmd_area_view)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
