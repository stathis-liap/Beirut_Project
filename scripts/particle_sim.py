#!/usr/bin/env python3
"""Standalone particle (SPH) flood prototype - proof of concept.

Weakly-compressible SPH (Bates/Monaghan-style momentum equation, Tait
equation of state), full 3D, GPU/CPU-portable via Taichi. A source pours
water down one flight of stairs from the synthetic test corridor
(test_synthetic.py's synthetic_dem()) - this validates the engine on a
known, controllable scene before it ever touches real Beirut data or the
GUI/pipeline.

Neighbor search uses a uniform-grid spatial hash (bin particles into H-sized
XY cells each step, only test the 9 neighboring columns), not brute-force
O(N^2). Cost per particle is bounded by *local* density, not total particle
count, so it stays roughly linear in N instead of quadratic - this is what
lets SPACING go small (lots of fine particles) without wall-clock time
exploding. The grid is 2D (XY only, ignoring Z when binning): water here is
always a thin sheet hugging the terrain, so within one XY column the Z
spread rarely exceeds H anyway, and the real 3D distance check inside the
inner loop (`if r < H`) still filters out any false candidates a column
happens to contain - it's just a candidate list, not the correctness check.

Terrain collision reads the DEM heightfield directly (bilinear sampled),
the same data structure flood_sim.py's grid solver uses - no separate
terrain representation needed.

Usage:
  python scripts/particle_sim.py --duration 10 --out output/particles
  python scripts/particle_sim.py --duration 10 --out output/particles --render
"""

import argparse
import glob
import json
import math
import os
import subprocess
import sys

import numpy as np

try:
    import taichi as ti
except ImportError:
    print("Installing taichi (not in requirements.txt - only this prototype needs it)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "taichi"])
    import taichi as ti

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_synthetic import synthetic_dem
from real_stairs import saint_nicolas_flight_dem

# ---------------------------------------------------------------- physics --
REST_DENSITY = 1000.0       # kg/m^3, water
SPACING = 0.02               # m, rest particle spacing - smaller = more, finer
                              # particles for the same water volume, closer to
                              # how real water actually looks (was 0.03, was
                              # 0.08 before that). The O(N) spatial hash (see
                              # module docstring) removed the cubic blow-up
                              # with PARTICLE COUNT, but SPACING has a second,
                              # separate cost: DT is CFL-tied to H = 2.2*SPACING,
                              # so halving SPACING roughly halves DT too - more
                              # steps needed for the same simulated duration,
                              # regardless of particle count or neighbor-search
                              # algorithm. Measured 0.01 (an earlier, more
                              # aggressive attempt) as impractical: DT dropped
                              # to ~22us and a 1s sim stayed stuck at t=0.00s
                              # for a minute-plus. 0.02 is the practical floor
                              # found by testing, not a guess.
PARTICLE_MASS = REST_DENSITY * SPACING ** 3
H = 2.2 * SPACING            # m, SPH smoothing radius (kernel support)
EMIT_JITTER = 10.0 * H        # m, side length of the square patch new point-
                              # source particles (emit(), not emit_rain()) are
                              # scattered over. Calibrated empirically, not a
                              # guess: with the old H*2 box, ~1 particle/step
                              # arriving into a patch barely bigger than the
                              # kernel radius - while freshly-spawned particles
                              # still have near-zero velocity and haven't had
                              # time to move away - supersaturated local
                              # density (73% of particles pinned at the
                              # MAX_DENSITY_RATIO cap within 22ms of simulated
                              # time) and launched particles via the resulting
                              # pressure spike straight to the MAX_SPEED clamp,
                              # not via gravity. Sweeping this from 2H to 16H
                              # (see particle_sim's git history / dev notes)
                              # found 10H the point where density stays
                              # essentially at rest (<0.3% of particles ever
                              # cap out) and the emitted speed distribution
                              # becomes gravity-graded (median ~2 m/s, rising
                              # smoothly with distance travelled) instead of
                              # slammed against the ceiling from frame one.
                              # Still comfortably inside the narrowest source
                              # scene's walkway (Saint Nicolas Stairs, 4.5 m).
PARTICLE_RADIUS = SPACING / 2
GRAVITY = 9.81
SOUND_SPEED = 20.0            # m/s, numerical - tuned for expected flow speeds
TAIT_GAMMA = 7
TAIT_B = SOUND_SPEED ** 2 * REST_DENSITY / TAIT_GAMMA
VISCOSITY = 2.0                # numerical/artificial viscosity - well above
                                # real water's physical value (~0.001), used
                                # here purely to damp instability, standard
                                # practice in real-time WCSPH
RESTITUTION = 0.0              # inelastic normal bounce off terrain
FRICTION = 0.7                 # tangential velocity retained PER SECOND of
                                # continuous ground contact (not per step -
                                # a shallow film is in contact almost every
                                # step, so a per-step fraction would compound
                                # into total damping within a couple of ms)
DT = 0.02 * H / SOUND_SPEED    # CFL-ish: a fraction of h / speed of sound
FRICTION_STEP = FRICTION ** DT  # the actual per-step multiplier
MAX_DENSITY_RATIO = 2.0        # cap rho/rho0 before the Tait EOS - real water
                                # doesn't compress much; without this cap a
                                # transient local pile-up explodes through
                                # the density^7 term into an unphysical burst
MAX_SPEED = 6.0                # m/s hard clamp - a blunt final safety valve;
                                # if particles are pinned at this cap it's a
                                # sign the emitter/kernel still needs tuning
MAX_ACCEL = 25.0 * GRAVITY     # safety valve: the spiky pressure kernel's
                                # gradient still peaks sharply as two
                                # particles' separation r -> 0, which (even
                                # with the density cap above) can inject a
                                # huge single-step velocity kick from one
                                # too-close pair. Capping |accel| is the
                                # standard real-time-SPH fix for this.

POLY6_COEF = 315.0 / (64.0 * math.pi * H ** 9)
SPIKY_GRAD_COEF = 45.0 / (math.pi * H ** 6)
VISC_LAP_COEF = 45.0 / (math.pi * H ** 6)

MAX_PARTICLES = 15000         # kept at the same cap as before SPACING dropped
                              # to 0.02 - see SPACING's comment on why a
                              # smaller SPACING alone (independent of N)
                              # already costs more, via a smaller DT, not
                              # just via more particles. Raising this further
                              # is possible (the O(N) spatial hash means it
                              # won't blow up catastrophically) but measured
                              # cost grows faster than linearly with N in
                              # practice (60s of wall-clock per simulated
                              # second at N~10500, 103s/simulated-second at
                              # N~23000) - likely GPU occupancy/launch-overhead
                              # effects at this scale, not the neighbor search
                              # itself. Raise with a short calibration run
                              # first, not blindly.
MAX_PER_CELL = 512             # candidates per XY cell. At cell size H
                                # (small domains) occupancy is set by
                                # physical density (H/SPACING is fixed at
                                # 2.2), independent of MAX_PARTICLES - this
                                # just needs margin for local pileups. At
                                # cell size > H (large domains, see
                                # MAX_HASH_CELLS_PER_AXIS) cells cover more
                                # area so can hold more particles even at
                                # low overall density; the grid stays small
                                # either way so the memory cost of a
                                # generous cap here is trivial.
MAX_HASH_CELLS_PER_AXIS = 1000  # hard cap on the spatial hash grid's shape,
                                # regardless of domain size. Cell size = H
                                # would be ideal (tightest neighbor search),
                                # but a fixed H over a large domain (e.g. a
                                # 1km area_rain crop) means hash_nx*hash_ny
                                # explodes - grid_particles alone hit ~30GB
                                # at 1km with cell=H and blew out CUDA memory.
                                # build_solver grows the cell size above H
                                # instead, once domain/H would exceed this
                                # cap - see its GRID_CELL computation. Total
                                # particles stay MAX_PARTICLES-capped either
                                # way, so bigger cells just mean more (still
                                # bounded) candidates per cell, not incorrect
                                # results.
                                # Was 400 - too tight once SPACING dropped to
                                # 0.02 (H=0.044): the 30m stairs domain then
                                # hit domain/400=0.075m > H, silently forcing
                                # ~1.7x oversized cells (each covering ~2.8x
                                # the area, so ~2.8x the wasted neighbor
                                # candidates) on a scene the cap was never
                                # meant to affect. 1000 keeps cell size == H
                                # up to a ~44m domain at this SPACING, while
                                # a 10km area_rain crop still only costs
                                # ~2GB (1000*1000*512*4 bytes), far under
                                # what OOM'd before.
HASH_MEMORY_BUDGET_MB = 192    # ceiling on the grid_particles allocation.
                                # MAX_HASH_CELLS_PER_AXIS alone bounds the
                                # axis COUNT, not the bytes, and the bytes are
                                # what actually fails: the 30 m / 5 cm
                                # map_point crop sits far under the 1000-cell
                                # axis cap yet still asks for 666*699*512*4 =
                                # ~950 MB to track at most MAX_PARTICLES
                                # (15k) particles - ~238 M slots for 15 k
                                # occupants. On a 4 GB laptop GPU (or a box
                                # whose commit limit is already stretched)
                                # that dies as CUDA_ERROR_OUT_OF_MEMORY
                                # before the first step, which reads as "the
                                # GPU is too small" when it's really an
                                # allocation ~40x larger than the problem.
                                # Cells are kept as fine as this budget
                                # allows and coarsened only when needed;
                                # coarser cells mean more candidates per
                                # neighbor query (slower) but never a wrong
                                # answer, since every candidate is still
                                # distance-checked. Override with
                                # BEIRUT_HASH_BUDGET_MB.
RES = 1.0                     # m/cell, matches synthetic_dem's default
REAL_STAIRS_RES = 0.05        # m/cell for the real_stairs scene - fine enough
                               # to resolve a 16 cm riser / 30 cm tread
RAIN_HEIGHT_M = 0.6            # area_rain scene: drop spawn height above the
                                # local terrain - enough to see a short visible
                                # fall/splash without wasting steps on freefall

MIN_SEED_DEPTH_M = 0.03        # area_rain --seed: ignore macro-grid noise/
                                # film depths - only seed cells the 2D
                                # solver considers meaningfully wet
MAX_SEED_PARTICLES = 4000      # area_rain --seed: cap on pre-existing-water
                                # particles, kept well under MAX_PARTICLES so
                                # there's still headroom left for the rain
                                # this sim adds on top. These approximate
                                # where the 2D macro solver already has
                                # standing water - not a mass-exact
                                # transcription of its depth field (that
                                # would need far more particles than the
                                # budget allows for anything but a tiny
                                # pond) - just enough that clicking an
                                # already-flooded spot starts as a visible
                                # puddle instead of dry ground.


def load_seed_depth(seed_path):
    """Loads a seed file written by render_3d.py's write_seed_file: a crop
    of the 2D macro solver's CURRENTLY DISPLAYED depth frame - whatever the
    time slider/playback was showing at click time - on its own coarse
    macro-resolution grid and UTM transform."""
    z = np.load(seed_path)
    return z["depth"], float(z["res"]), float(z["minx"]), float(z["maxy"])


def build_seed_particles(dem, res, minx, maxy, seed_depth, seed_res, seed_minx, seed_maxy):
    """Resamples `seed_depth` (the 2D macro solver's depth field, on its own
    coarser grid) onto this scene's local (dem, res) grid via UTM-aligned
    bilinear interpolation, then returns (pos, vel) resting-particle arrays
    approximating where the macro sim already has standing water - see
    MAX_SEED_PARTICLES for why this is a capped visual approximation, not an
    exact transcription of the macro solver's water volume.
    """
    from scipy.ndimage import map_coordinates

    h, w = dem.shape
    rows, cols = np.mgrid[0:h, 0:w]
    x = minx + cols * res
    y = maxy - rows * res
    src_row = (seed_maxy - y) / seed_res
    src_col = (x - seed_minx) / seed_res
    depth_local = map_coordinates(seed_depth, [src_row, src_col], order=1,
                                  mode="constant", cval=0.0)

    wet = np.argwhere(depth_local > MIN_SEED_DEPTH_M)
    if len(wet) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    if len(wet) > MAX_SEED_PARTICLES:
        idx = np.random.default_rng(0).choice(len(wet), MAX_SEED_PARTICLES, replace=False)
        wet = wet[idx]
    r, c = wet[:, 0], wet[:, 1]
    px, py = c * res, r * res
    pz = dem[r, c] + np.minimum(depth_local[r, c], 0.15) * 0.5
    pos = np.stack([px, py, pz], axis=1).astype(np.float32)
    vel = np.zeros_like(pos)
    return pos, vel


def crop_stairs_scene(flight=(75, 140), margin=10):
    """One flight of stairs from the synthetic corridor, tightly cropped to
    the street footprint (+ margin) so the particle count stays small."""
    dem_full, street_full = synthetic_dem()  # default h=400, w=300, res=1.0
    r0, r1 = flight
    cols = np.where(street_full[r0:r1].any(axis=0))[0]
    c0 = max(int(cols.min()) - margin, 0)
    c1 = min(int(cols.max()) + margin + 1, dem_full.shape[1])
    dem = np.ascontiguousarray(dem_full[r0:r1, c0:c1], dtype=np.float32)
    street = np.ascontiguousarray(street_full[r0:r1, c0:c1])
    return dem, street


def source_position(dem, street, res=RES):
    """Street centerline a few rows in from the (uphill) top edge."""
    row = 3
    cols = np.where(street[row])[0]
    col = int(cols.mean()) if len(cols) else dem.shape[1] // 2
    x, y = col * res, row * res
    z = float(dem[row, col]) + 0.05
    return x, y, z


def hash_grid_dims(domain_x, domain_y):
    """Cell size and cell counts for the spatial hash: as fine as H, then
    coarsened until the grid_particles allocation fits both
    MAX_HASH_CELLS_PER_AXIS and HASH_MEMORY_BUDGET_MB.
    """
    budget_bytes = float(os.environ.get(
        "BEIRUT_HASH_BUDGET_MB", HASH_MEMORY_BUDGET_MB)) * 1024 * 1024

    cell = max(H, domain_x / MAX_HASH_CELLS_PER_AXIS,
               domain_y / MAX_HASH_CELLS_PER_AXIS)

    def dims(c):
        return (max(1, int(domain_x / c) + 3), max(1, int(domain_y / c) + 3))

    nx, ny = dims(cell)
    while nx * ny * MAX_PER_CELL * 4 > budget_bytes and (nx > 4 or ny > 4):
        cell *= 1.25
        nx, ny = dims(cell)

    mb = nx * ny * MAX_PER_CELL * 4 / 1024 / 1024
    print(f"  spatial hash: {nx} x {ny} cells @ {cell:.3f} m "
          f"({mb:.0f} MB, budget {budget_bytes / 1024 / 1024:.0f} MB)")
    return cell, nx, ny


def build_solver(dem, res=RES):
    grid_h, grid_w = dem.shape
    dem_field = ti.field(dtype=ti.f32, shape=(grid_h, grid_w))
    dem_field.from_numpy(dem)

    pos = ti.Vector.field(3, dtype=ti.f32, shape=MAX_PARTICLES)
    vel = ti.Vector.field(3, dtype=ti.f32, shape=MAX_PARTICLES)
    accel = ti.Vector.field(3, dtype=ti.f32, shape=MAX_PARTICLES)
    density = ti.field(dtype=ti.f32, shape=MAX_PARTICLES)
    pressure = ti.field(dtype=ti.f32, shape=MAX_PARTICLES)

    domain_x = (grid_w - 1) * res
    domain_y = (grid_h - 1) * res

    # spatial hash: 2D grid of XY columns, padded by one cell on each side
    # so a particle sitting exactly on the domain edge still has full 3x3
    # neighbor coverage. Cell size is H when the domain is small (tightest
    # neighbor search, e.g. the stairs scenes), but grows beyond H for a
    # large domain (e.g. a wide-radius area_rain crop) so the grid's cell
    # COUNT stays bounded - see MAX_HASH_CELLS_PER_AXIS's comment for why
    # (an unbounded hash_nx*hash_ny*MAX_PER_CELL blew past available GPU
    # memory at 1km domains).
    grid_cell, hash_nx, hash_ny = hash_grid_dims(domain_x, domain_y)
    grid_count = ti.field(dtype=ti.i32, shape=(hash_nx, hash_ny))
    grid_particles = ti.field(dtype=ti.i32, shape=(hash_nx, hash_ny, MAX_PER_CELL))

    @ti.func
    def cell_of(p):
        cx = ti.min(ti.max(int(p[0] / grid_cell) + 1, 0), hash_nx - 1)
        cy = ti.min(ti.max(int(p[1] / grid_cell) + 1, 0), hash_ny - 1)
        return cx, cy

    @ti.kernel
    def clear_grid():
        for i, j in grid_count:
            grid_count[i, j] = 0

    @ti.kernel
    def build_grid(n: ti.i32):
        for i in range(n):
            cx, cy = cell_of(pos[i])
            slot = ti.atomic_add(grid_count[cx, cy], 1)
            if slot < MAX_PER_CELL:
                grid_particles[cx, cy, slot] = i

    @ti.func
    def sample_height(x, y):
        fx = ti.min(ti.max(x / res, 0.0), grid_w - 1.001)
        fy = ti.min(ti.max(y / res, 0.0), grid_h - 1.001)
        c0, r0 = int(fx), int(fy)
        tx, ty = fx - c0, fy - r0
        z00, z10 = dem_field[r0, c0], dem_field[r0, c0 + 1]
        z01, z11 = dem_field[r0 + 1, c0], dem_field[r0 + 1, c0 + 1]
        z0 = z00 * (1 - tx) + z10 * tx
        z1 = z01 * (1 - tx) + z11 * tx
        return z0 * (1 - ty) + z1 * ty

    @ti.func
    def terrain_normal(x, y):
        eps = res * 0.5
        dzdx = (sample_height(x + eps, y) - sample_height(x - eps, y)) / (2 * eps)
        dzdy = (sample_height(x, y + eps) - sample_height(x, y - eps)) / (2 * eps)
        return ti.Vector([-dzdx, -dzdy, 1.0]).normalized()

    @ti.func
    def poly6(r):
        out = 0.0
        if 0.0 <= r <= H:
            t = H * H - r * r
            out = POLY6_COEF * t * t * t
        return out

    @ti.func
    def grad_spiky(rij, r):
        out = ti.Vector([0.0, 0.0, 0.0])
        if 0.0 < r <= H:
            t = H - r
            out = -SPIKY_GRAD_COEF * t * t * (rij / r)
        return out

    @ti.func
    def lap_visc(r):
        out = 0.0
        if 0.0 <= r <= H:
            out = VISC_LAP_COEF * (H - r)
        return out

    @ti.kernel
    def emit(n0: ti.i32, n_new: ti.i32, sx: ti.f32, sy: ti.f32, sz: ti.f32,
             vx: ti.f32, vy: ti.f32):
        for k in range(n_new):
            idx = n0 + k
            # Jitter box for the new particle's spawn point, sized to EMIT_JITTER
            # (see that constant's comment) rather than a fixed "H*2" - a box
            # only slightly bigger than the kernel radius sounds safe but isn't:
            # emitted particles start at ~zero velocity, so for the first
            # several hundred steps they barely move away from where they
            # spawned, while ~1 new particle keeps arriving into that same
            # patch every step. Measured with a 2H box: 73% of particles were
            # already pinned at the density cap within 22ms, and the resulting
            # pressure spike (not gravity) was what launched them, straight to
            # the MAX_SPEED safety clamp - unphysical, and confirmed unphysical
            # by comparing displacement against free-fall energy (particles
            # that had barely dropped in elevation were already near max
            # speed). Spreading emission over EMIT_JITTER instead gives
            # incoming water room to sit near rest density while gravity - not
            # an artificial compression spike - accelerates it downhill.
            jx = (ti.random(ti.f32) - 0.5) * EMIT_JITTER
            jy = (ti.random(ti.f32) - 0.5) * EMIT_JITTER
            pos[idx] = ti.Vector([sx + jx, sy + jy, sz])
            vel[idx] = ti.Vector([vx, vy, -0.1])

    @ti.kernel
    def emit_rain(n0: ti.i32, n_new: ti.i32, rain_height: ti.f32):
        # scattered across the whole domain instead of one point source -
        # each drop starts just above the LOCAL terrain height (not one
        # fixed altitude), so drops over a hilltop and drops over a low
        # spot both fall about the same short distance before landing
        for k in range(n_new):
            idx = n0 + k
            rx = ti.random(ti.f32) * domain_x
            ry = ti.random(ti.f32) * domain_y
            gz = sample_height(rx, ry)
            pos[idx] = ti.Vector([rx, ry, gz + rain_height])
            vel[idx] = ti.Vector([0.0, 0.0, -1.5])

    @ti.kernel
    def compute_density_pressure(n: ti.i32):
        for i in range(n):
            rho = 0.0
            pi = pos[i]
            cx, cy = cell_of(pi)
            for dx, dy in ti.ndrange((-1, 2), (-1, 2)):
                ncx, ncy = cx + dx, cy + dy
                if 0 <= ncx < hash_nx and 0 <= ncy < hash_ny:
                    cnt = ti.min(grid_count[ncx, ncy], MAX_PER_CELL)
                    for k in range(cnt):
                        j = grid_particles[ncx, ncy, k]
                        r = (pi - pos[j]).norm()
                        if r < H:
                            rho += PARTICLE_MASS * poly6(r)
            rho = ti.min(ti.max(rho, REST_DENSITY * 0.05), REST_DENSITY * MAX_DENSITY_RATIO)
            density[i] = rho
            pressure[i] = ti.max(TAIT_B * (ti.pow(rho / REST_DENSITY, TAIT_GAMMA) - 1.0), 0.0)

    @ti.kernel
    def compute_forces(n: ti.i32):
        for i in range(n):
            pi, vi = pos[i], vel[i]
            rho_i, p_i = density[i], pressure[i]
            press_acc = ti.Vector([0.0, 0.0, 0.0])
            visc_acc = ti.Vector([0.0, 0.0, 0.0])
            cx, cy = cell_of(pi)
            for dx, dy in ti.ndrange((-1, 2), (-1, 2)):
                ncx, ncy = cx + dx, cy + dy
                if 0 <= ncx < hash_nx and 0 <= ncy < hash_ny:
                    cnt = ti.min(grid_count[ncx, ncy], MAX_PER_CELL)
                    for k in range(cnt):
                        j = grid_particles[ncx, ncy, k]
                        if j != i:
                            rij = pi - pos[j]
                            r = rij.norm()
                            if 1e-6 < r < H:
                                rho_j, p_j = density[j], pressure[j]
                                press_acc += -PARTICLE_MASS * (p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j)) * grad_spiky(rij, r)
                                visc_acc += PARTICLE_MASS * (vel[j] - vi) / rho_j * lap_visc(r)
            acc = ti.Vector([0.0, 0.0, -GRAVITY]) + press_acc + VISCOSITY * visc_acc / rho_i
            mag = acc.norm()
            if mag > MAX_ACCEL:
                acc = acc / mag * MAX_ACCEL
            accel[i] = acc

    @ti.kernel
    def integrate(n: ti.i32, dt: ti.f32):
        for i in range(n):
            vel[i] += accel[i] * dt
            speed = vel[i].norm()
            if speed > MAX_SPEED:
                vel[i] = vel[i] / speed * MAX_SPEED
            pos[i] += vel[i] * dt

            x, y, z = pos[i][0], pos[i][1], pos[i][2]
            gz = sample_height(x, y)
            if z < gz + PARTICLE_RADIUS:
                nrm = terrain_normal(x, y)
                vn = vel[i].dot(nrm)
                if vn < 0:
                    vel[i] -= (1.0 + RESTITUTION) * vn * nrm
                vel[i] *= FRICTION_STEP
                pos[i][2] = gz + PARTICLE_RADIUS

            # domain edges: simple reflective walls (open-boundary outflow
            # is the natural next step, skipped here - the crop's margin
            # keeps particles from reaching the edges within this demo)
            if pos[i][0] < 0.0:
                pos[i][0], vel[i][0] = 0.0, -vel[i][0] * FRICTION_STEP
            if pos[i][0] > domain_x:
                pos[i][0], vel[i][0] = domain_x, -vel[i][0] * FRICTION_STEP
            if pos[i][1] < 0.0:
                pos[i][1], vel[i][1] = 0.0, -vel[i][1] * FRICTION_STEP
            if pos[i][1] > domain_y:
                pos[i][1], vel[i][1] = domain_y, -vel[i][1] * FRICTION_STEP

    return dict(pos=pos, vel=vel, density=density,
                emit=emit, emit_rain=emit_rain, clear_grid=clear_grid, build_grid=build_grid,
                compute_density_pressure=compute_density_pressure,
                compute_forces=compute_forces, integrate=integrate)


def run(duration, out_dir, rate_m3s=0.15, save_every=0.25, scene="synthetic",
        utm_x=None, utm_y=None, crop_size=30.0, radius_m=None, las_path=None,
        rain_mmh=None, seed_path=None, terrain_dir=None):
    vx0, vy0 = 0.0, 0.05  # default emit nudge, matches the row-0-is-uphill
                          # convention shared by the synthetic and real_stairs scenes
    rain_mode = False
    sx = sy = sz = None
    seed_pos = seed_vel = None
    if scene == "synthetic":
        dem, street = crop_stairs_scene()
        res = RES
        sx, sy, sz = source_position(dem, street, res=res)
    elif scene == "real_stairs":
        dem, street = saint_nicolas_flight_dem(res=REAL_STAIRS_RES)
        res = REAL_STAIRS_RES
        sx, sy, sz = source_position(dem, street, res=res)
    elif scene == "map_point":
        if utm_x is None or utm_y is None:
            raise ValueError("map_point scene needs --utm-x and --utm-y")
        import map_point
        import map_data
        res = map_data.MAP_RES_M
        dem, obstacle, _t = map_point.crop_local(utm_x, utm_y, size_m=crop_size, res=res)
        sx, sy, sz, vx0, vy0 = map_point.find_source(dem, obstacle, res)
    elif scene == "area_rain":
        if utm_x is None or utm_y is None:
            raise ValueError("area_rain scene needs --utm-x and --utm-y")
        import reconstruct_area
        from las_common import resolve_las_path, load_transform
        radius_m = radius_m or reconstruct_area.DEFAULT_RADIUS_M
        rain_mmh = 30.0 if rain_mmh is None else rain_mmh
        resolved_las = resolve_las_path(las_path, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
        if not resolved_las:
            raise ValueError(f"no .las file found (looked for {las_path!r} and in data/)")
        res = reconstruct_area.auto_res(radius_m)
        if terrain_dir is None:
            terrain_dir = os.path.join(out_dir, "_terrain")
        if os.path.exists(os.path.join(terrain_dir, "dem.npy")):
            # a caller (e.g. area_sim.py, running both the 2D grid and this
            # 3D solver on the same footprint) may have already built this
            # exact terrain - reuse it instead of re-streaming the whole
            # .las again, which costs ~15-20s regardless of crop size (the
            # scan is bounded by the SOURCE file's size, not the crop).
            print(f"  reusing already-built terrain at {terrain_dir}")
        else:
            reconstruct_area.build(resolved_las, utm_x, utm_y, radius_m, res, terrain_dir)
        dem = np.load(os.path.join(terrain_dir, "dem.npy")).astype(np.float64)
        rain_mode = True

        if seed_path and os.path.exists(seed_path):
            local_t = load_transform(os.path.join(terrain_dir, "dem_transform.json"))
            seed_depth, seed_res, seed_minx, seed_maxy = load_seed_depth(seed_path)
            seed_pos, seed_vel = build_seed_particles(
                dem, res, local_t["minx"], local_t["maxy"],
                seed_depth, seed_res, seed_minx, seed_maxy)
    elif scene == "cad_model":
        # rain over a terrain built by load_cad_model.py from a .3dm CAD
        # model instead of a .las point cloud - not UTM-addressable (the
        # model sits at its own local origin, see that script's docstring),
        # so this is a "local scene" like synthetic/real_stairs, just
        # rained on like area_rain instead of fed from one point source.
        if not terrain_dir or not os.path.exists(os.path.join(terrain_dir, "dem.npy")):
            raise ValueError("cad_model scene needs --terrain-dir pointing at a "
                             "load_cad_model.py output (dem.npy/dem_transform.json)")
        from las_common import load_transform
        local_t = load_transform(os.path.join(terrain_dir, "dem_transform.json"))
        res = local_t["res"]
        dem = np.load(os.path.join(terrain_dir, "dem.npy")).astype(np.float64)
        rain_mmh = 30.0 if rain_mmh is None else rain_mmh
        rain_mode = True
    else:
        raise ValueError(f"unknown scene {scene!r}")

    if rain_mode:
        domain_area = ((dem.shape[1] - 1) * res) * ((dem.shape[0] - 1) * res)
        rate_m3s = (rain_mmh / 1000.0 / 3600.0) * domain_area
        print(f"scene: {scene!r}, {dem.shape[1] * res:.1f} x {dem.shape[0] * res:.1f} m "
              f"@ {res:.2f} m/cell, rain {rain_mmh:.0f} mm/h over {domain_area:.0f} m^2 "
              f"-> {rate_m3s:.4f} m^3/s")
    else:
        print(f"scene: {scene!r}, {dem.shape[1] * res:.1f} x {dem.shape[0] * res:.1f} m "
              f"@ {res:.2f} m/cell, source at ({sx:.1f}, {sy:.1f}, {sz:.2f})")
    print(f"H={H:.3f} m  dt={DT * 1000:.2f} ms  particle_mass={PARTICLE_MASS * 1000:.1f} g  "
          f"TAIT_B={TAIT_B:.0f}")

    os.makedirs(out_dir, exist_ok=True)
    solver = build_solver(dem, res=res)
    pos, vel, density = solver["pos"], solver["vel"], solver["density"]
    # local refs, not dict lookups, in the hot loop - and n as a plain
    # Python int, NOT a ti.field read back every step. n_active used to be
    # a ti.field(shape=()) re-read via `int(n_active[None])` every single
    # iteration - that forces a GPU->CPU sync (flush the command queue,
    # transfer one int back) on EVERY step. With DT in the tens of
    # microseconds, a 10s sim is 100,000+ steps, so that was 100,000+
    # forced syncs killing GPU pipelining - the dominant cost, well beyond
    # the actual SPH compute. n_active was never read inside a kernel, so
    # host-side tracking loses nothing.
    emit_fn = solver["emit_rain"] if rain_mode else solver["emit"]
    clear_grid = solver["clear_grid"]
    build_grid = solver["build_grid"]
    compute_density_pressure = solver["compute_density_pressure"]
    compute_forces = solver["compute_forces"]
    integrate = solver["integrate"]

    particles_per_step = rate_m3s / SPACING ** 3 * DT
    carry = 0.0
    t, next_save, it = 0.0, 0.0, 0
    n = 0
    if seed_pos is not None and len(seed_pos):
        # upload into the field's first n_seed slots by writing a full
        # MAX_PARTICLES-sized (zero-padded) array - taichi's from_numpy()
        # requires the array to match the field's full shape, not a slice.
        # Nothing else has written to pos/vel yet at this point, so this is
        # a plain initialization, not an overwrite of anything live.
        n = min(len(seed_pos), MAX_PARTICLES)
        full_pos = np.zeros((MAX_PARTICLES, 3), dtype=np.float32)
        full_vel = np.zeros((MAX_PARTICLES, 3), dtype=np.float32)
        full_pos[:n] = seed_pos[:n]
        full_vel[:n] = seed_vel[:n]
        pos.from_numpy(full_pos)
        vel.from_numpy(full_vel)
        print(f"  seeded {n} resting particles from the macro run's existing water")
    saved = []

    while t < duration:
        carry += particles_per_step
        n_new = int(carry)
        if n_new > 0 and n < MAX_PARTICLES:
            n_new = min(n_new, MAX_PARTICLES - n)
            if rain_mode:
                emit_fn(n, n_new, RAIN_HEIGHT_M)
            else:
                emit_fn(n, n_new, sx, sy, sz, vx0, vy0)
            carry -= n_new
            n += n_new

        if n > 0:
            clear_grid()
            build_grid(n)
            compute_density_pressure(n)
            compute_forces(n)
            integrate(n, DT)

        if t >= next_save:
            p = pos.to_numpy()[:n]
            v = vel.to_numpy()[:n]
            speed = np.linalg.norm(v, axis=1)
            fn = os.path.join(out_dir, f"particles_{int(round(t * 100)):06d}.npz")
            np.savez(fn, pos=p, speed=speed)
            saved.append(os.path.basename(fn))
            max_speed = float(speed.max()) if n else 0.0
            print(f"  t={t:6.2f}s n={n:5d} max_speed={max_speed:5.2f} m/s")
            next_save += save_every
        t += DT
        it += 1

    np.save(os.path.join(out_dir, "dem_crop.npy"), dem)
    meta = {"scene": scene, "res": res}
    if scene == "map_point":
        meta.update({"utm_x": utm_x, "utm_y": utm_y, "crop_size_m": crop_size})
    elif scene == "area_rain":
        meta.update({"utm_x": utm_x, "utm_y": utm_y, "radius_m": radius_m, "rain_mmh": rain_mmh})
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done: {len(saved)} frames -> {out_dir}/")
    return dem, out_dir, saved, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=10.0, help="seconds of sim time")
    ap.add_argument("--rate", type=float, default=0.15,
                    help="source flow rate, m^3/s (synthetic/real_stairs/map_point scenes)")
    ap.add_argument("--out", default="output/particles")
    ap.add_argument("--scene",
                    choices=["synthetic", "real_stairs", "map_point", "area_rain", "cad_model"],
                    default="synthetic",
                    help="synthetic: test_synthetic.py's S-shaped corridor (default). "
                         "real_stairs: one real flight of Beirut's Saint Nicolas Stairs, "
                         "geolocated and measured from the 5cm drone DSM (see real_stairs.py). "
                         "map_point: any point on the full 5cm map (see map_point.py), "
                         "needs --utm-x/--utm-y. "
                         "area_rain: rain falling over a .las-cropped area around a point "
                         "(see reconstruct_area.py), needs --utm-x/--utm-y. "
                         "cad_model: rain over a terrain built from a CAD model (see "
                         "load_cad_model.py), needs --terrain-dir")
    ap.add_argument("--terrain-dir", help="cad_model scene (required): a load_cad_model.py "
                    "output dir. area_rain scene (optional): reuse an already-built crop "
                    "instead of re-streaming the .las")
    ap.add_argument("--utm-x", type=float, help="map_point/area_rain: UTM easting (EPSG:32636)")
    ap.add_argument("--utm-y", type=float, help="map_point/area_rain: UTM northing (EPSG:32636)")
    ap.add_argument("--crop-size", type=float, default=30.0,
                    help="map_point scene: crop footprint, meters")
    ap.add_argument("--radius", type=float,
                    help="area_rain scene: crop radius, meters (default: reconstruct_area.py's, 1500m)")
    ap.add_argument("--las", help="area_rain scene: .las path (default: auto-discover in data/)")
    ap.add_argument("--rain", type=float, help="area_rain scene: rain intensity, mm/h (default 30)")
    ap.add_argument("--seed", help="area_rain scene: path to a seed file (see render_3d.py's "
                    "write_seed_file) - pre-fills standing water matching the 2D macro solver's "
                    "current depth at this spot instead of starting from dry ground")
    ap.add_argument("--render", action="store_true", help="also dump PyVista snapshots")
    ap.add_argument("--view", action="store_true",
                    help="after the run, open the interactive on-screen viewer "
                         "(same process, see particle_render.interactive_view)")
    args = ap.parse_args()

    import ti_init
    ti_init.init()

    dem, out_dir, saved, res = run(args.duration, args.out, rate_m3s=args.rate, scene=args.scene,
                                   utm_x=args.utm_x, utm_y=args.utm_y, crop_size=args.crop_size,
                                   radius_m=args.radius, las_path=args.las, rain_mmh=args.rain,
                                   seed_path=args.seed, terrain_dir=args.terrain_dir)

    if args.render:
        from particle_render import render_snapshots
        render_snapshots(dem, out_dir, saved, res=res)

    if args.view:
        from particle_render import interactive_view
        frames = sorted(os.path.basename(p) for p in
                        glob.glob(os.path.join(out_dir, "particles_*.npz")))
        interactive_view(dem, out_dir, frames, res=res)


if __name__ == "__main__":
    main()
