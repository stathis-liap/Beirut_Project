"""Beirut Corridor Flood Simulator - main entry point.

Launches the plug-and-play GUI: pick a rain amount, storm duration, and
terrain quality, hit Run, and it drives the whole pipeline (terrain build ->
flood simulation -> heatmap/video render) for you.

Usage:
  python main.py
"""

import os
import sys
import tkinter as tk

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

from gui_demo import App

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
