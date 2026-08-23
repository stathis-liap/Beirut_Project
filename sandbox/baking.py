"""Design -> effective terrain arrays, in memory.

Mirrors scripts/bake_corridor.py exactly (depression lowers the DEM, then
infiltration/roughness are overlaid), so a sandbox design is provably
equivalent to the pipeline's baked terrain for the same material layout.
"""


def bake(base, design):
    """Returns (dem, manning, infil, erodible) float32 arrays ready for
    flood_gpu.simulate. Never mutates `base` or `design` - always works on
    copies."""
    dem = base.dem.copy()
    man = base.manning.copy()
    infil = base.infil.copy()
    erodible = base.erodible.copy()
    mat = design.material

    for m in design.materials["materials"]:
        cells = mat == m["id"]
        if not cells.any():
            continue
        infil[cells] = m["infil_mmh"]
        man[cells] = m["manning_n"]
        erodible[cells] = m.get("erodible_frac", 0.0)
        if m["depression_m"] > 0:
            dem[cells] -= m["depression_m"]

    # The corridor's own demolition, applied before the user's sculpting for
    # the same reason bake_corridor.py applies it before the material bake:
    # it changes the ground the design is drawn on, it is not an edit to it.
    if design.clears_buildings and base.flatten_delta is not None:
        dem = dem + base.flatten_delta

    dem = dem + design.dem_delta
    return dem, man, infil, erodible
