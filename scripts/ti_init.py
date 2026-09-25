"""One place to start Taichi, shared by particle_sim.py / area_sim.py /
cad_sim.py.

They each used to call a bare `ti.init(arch=ti.gpu)`. Two things go wrong
with that on a modest machine, and both surface as the same unhelpful
CUDA_ERROR_OUT_OF_MEMORY:

  - Taichi sizes its device allocation from total VRAM, which overshoots on
    a small laptop GPU (this project's dev box is a 4 GB card). The SPH
    scenes are budgeted at MAX_PARTICLES (15k) with a memory-capped spatial
    hash, so a small explicit cap is plenty.
  - Under WDDM those device allocations are committed against system
    memory, so a box whose pagefile is pinned (a full system drive, say)
    fails here even with the GPU sitting idle - nvidia-smi shows free VRAM
    while the allocation still can't be backed.

`ti.init()` is also lazy: it succeeds, and the allocation only actually
happens when the first kernel materializes the field tree. Wrapping just
the init call therefore catches nothing, which is why the fallback below
forces materialization with a probe kernel before declaring the GPU usable.

A GPU that fails for any reason falls back to CPU with a visible notice
rather than taking the run down - these scenes are small enough that x64 is
slower but usable, which beats no simulation at all.

Env overrides:
  BEIRUT_TAICHI_ARCH=cpu|gpu   force a backend, skip the fallback logic
  BEIRUT_TAICHI_MEM_GB=<float> change the device-memory cap
"""

import os

DEFAULT_DEVICE_MEMORY_GB = 0.5  # comfortably over what the scenes ask for
                                # (a HASH_MEMORY_BUDGET_MB-capped hash plus
                                # MAX_PARTICLES-sized particle fields), and
                                # small enough to still be grantable when
                                # the system commit charge is tight


def _probe(ti):
    """Forces the field tree to materialize, so a device allocation that is
    going to fail fails here - inside the caller's try - instead of midway
    through the first simulation step."""
    f = ti.field(ti.f32, shape=(64, 64))

    @ti.kernel
    def touch():
        for i, j in f:
            f[i, j] = 1.0

    touch()
    return float(f.to_numpy().sum())


def init(verbose=True):
    """Starts Taichi and returns the arch actually in use."""
    import taichi as ti

    want = os.environ.get("BEIRUT_TAICHI_ARCH", "").strip().lower()
    mem_gb = float(os.environ.get("BEIRUT_TAICHI_MEM_GB", DEFAULT_DEVICE_MEMORY_GB))

    if want == "cpu":
        ti.init(arch=ti.cpu)
    elif want == "gpu":
        ti.init(arch=ti.gpu, device_memory_GB=mem_gb)
        _probe(ti)
    else:
        try:
            ti.init(arch=ti.gpu, device_memory_GB=mem_gb)
            _probe(ti)
        except Exception as exc:
            print(f"Taichi GPU init failed ({type(exc).__name__}) - falling back "
                  "to CPU, which is slower but produces the same result. "
                  "Set BEIRUT_TAICHI_ARCH=gpu to see the raw error.")
            if verbose:
                print(f"  reason: {str(exc).strip().splitlines()[-1][:200]}")
            ti.reset()
            ti.init(arch=ti.cpu)

    arch = ti.lang.impl.current_cfg().arch
    if verbose:
        print("Taichi backend:", arch)
    return arch
