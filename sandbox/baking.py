"""Design -> effective terrain arrays, in memory.

Mirrors scripts/bake_corridor.py exactly (depression lowers the DEM, then
infiltration/roughness are overlaid), so a sandbox design is provably
equivalent to the pipeline's baked terrain for the same material layout.
"""


def bake(base, design):
    """Returns (dem, manning, infil) float32 arrays ready for flood_gpu.simulate.
    Never mutates `base` or `design` - always works on copies."""
    dem = base.dem.copy()
    man = base.manning.copy()
    infil = base.infil.copy()
    mat = design.material

    for m in design.materials["materials"]:
        cells = mat == m["id"]
        if not cells.any():
            continue
        infil[cells] = m["infil_mmh"]
        man[cells] = m["manning_n"]
        if m["depression_m"] > 0:
            dem[cells] -= m["depression_m"]

    dem = dem + design.dem_delta
    return dem, man, infil
