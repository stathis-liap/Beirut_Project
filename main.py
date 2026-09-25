"""Beirut Corridor Flood Simulator - main entry point.

Launches the plug-and-play GUI: pick a rain amount, storm duration, and
terrain quality, hit Run, and it drives the whole pipeline (terrain build ->
flood simulation -> heatmap/video render) for you.

Usage:
  python main.py
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# import name -> pip name, for the packages actually imported under scripts/.
# The GUI exposes every mode in one window, so this covers all of them, not
# just the 2D flood pipeline - a missing taichi/rasterio/rhino3dm used to
# surface only once someone clicked into that tab and got a traceback.
REQUIRED = {
    "numpy": "numpy", "scipy": "scipy", "PIL": "pillow",
    "matplotlib": "matplotlib", "numba": "numba", "pyvista": "pyvista",
    "imageio": "imageio", "pyproj": "pyproj",
    "taichi": "taichi",        # 3D SPH solver
    "rasterio": "rasterio",    # "pick a point on the map" raster reads
    "rhino3dm": "rhino3dm",    # .3dm CAD models
    "ezdxf": "ezdxf",          # .dxf / converted .dwg
    "pandas": "pandas",        # CAD vert -> DEM binning
}


def ensure_dependencies():
    missing = [pip_name for mod, pip_name in REQUIRED.items()
               if not _can_import(mod)]
    if not missing:
        return
    print(f"Missing packages: {', '.join(missing)} - installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)


def _can_import(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def ensure_data():
    data_dir = os.path.join(ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    from scripts.las_common import find_las_file
    if find_las_file(data_dir):
        return
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "Missing point cloud",
        f"Can't find any .las file in {data_dir}.\n\n"
        "Drop a .las point cloud there - any file works, plug and play "
        "(e.g. a corridor already cropped with scripts/crop_cloud.py, or "
        "any other LAS 1.2 format-2 point cloud).")
    sys.exit(1)


if __name__ == "__main__":
    ensure_dependencies()
    ensure_data()

    import tkinter as tk
    from scripts.gui_demo import App

    root = tk.Tk()
    App(root)
    root.mainloop()
