"""numpy -> PNG / quantized binary encoders for the browser."""
import io

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
from matplotlib import colormaps

# same conventions as scripts/render2d.py, so sandbox figures match the
# pipeline's published ones.
HAZ_BOUNDS = [0.75, 1.25, 2.0]
HAZ_COLORS_RGB = [(255, 233, 61), (255, 154, 46), (232, 68, 46), (125, 26, 158)]


def dem_bin(dem, valid, lo, scale):
    """float32 DEM -> uint16-quantized bytes, row-major, row 0 = north.

    z = lo + q * scale to reconstruct; invalid cells encode as 0.
    """
    safe_scale = max(scale, 1e-9)
    q = np.where(valid, np.round((dem - lo) / safe_scale), 0)
    q = np.clip(q, 0, 65535).astype(np.uint16)
    return q.tobytes()


def masks_png(masks, zone):
    """Pack valid/building/water/zone into one RGBA PNG.

    R=building, G=zone, B=water, A=valid (each channel 0 or 255).
    """
    h, w = zone.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 0] = masks["building"].astype(np.uint8) * 255
    rgba[..., 1] = zone.astype(np.uint8) * 255
    rgba[..., 2] = masks["water"].astype(np.uint8) * 255
    rgba[..., 3] = masks["valid"].astype(np.uint8) * 255
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _png_bytes(rgba):
    buf = io.BytesIO()
    Image.fromarray(rgba.astype(np.uint8), "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def depth_png(depth, vmax, alpha_below=0.01):
    """Colormapped flood-depth RGBA PNG (turbo, matching render2d.py)."""
    x = np.clip(depth / max(vmax, 1e-6), 0, 1)
    rgba = (colormaps["turbo"](x) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(depth > alpha_below, 230, 0).astype(np.uint8)
    return _png_bytes(rgba)


def hazard_png(max_hazard, max_depth, depth_thr=0.10):
    """DEFRA-style hazard classes, same bounds/colors as render2d.py."""
    cls = np.digitize(max_hazard, HAZ_BOUNDS)  # 0..3
    show = max_depth >= depth_thr
    rgba = np.zeros((*max_hazard.shape, 4), np.uint8)
    for i, (r, g, b) in enumerate(HAZ_COLORS_RGB):
        m = show & (cls == i)
        rgba[m, 0] = r
        rgba[m, 1] = g
        rgba[m, 2] = b
        rgba[m, 3] = 220
    return _png_bytes(rgba)


def diff_png(depth_b, depth_a, vmax, thr=0.02):
    """Diverging depth_b - depth_a map (RdBu_r, matching render2d.py): blue =
    shallower in b, red = deeper in b."""
    d = depth_b - depth_a
    sig = np.abs(d) > thr
    t = np.clip(d / max(vmax, 1e-6), -1, 1)
    rgba = (colormaps["RdBu_r"]((t + 1) / 2) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(sig, 220, 0).astype(np.uint8)
    return _png_bytes(rgba)
