"""Shared helpers for streaming huge uncompressed LAS files.

The Beirut drone cloud is LAS 1.2, point format 2 (26-byte records:
XYZ int32 + intensity + flags + classification + ... + RGB uint16).
Reading it as a raw structured array is ~10x faster than a generic
LAS library path and works on partially downloaded files.
"""

import json
import os
import struct

import numpy as np

# LAS point format 2 record layout (26 bytes)
POINT_DTYPE = np.dtype([
    ("X", "<i4"), ("Y", "<i4"), ("Z", "<i4"),
    ("intensity", "<u2"),
    ("flags", "u1"),
    ("classification", "u1"),
    ("scan_angle", "i1"),
    ("user_data", "u1"),
    ("point_source_id", "<u2"),
    ("red", "<u2"), ("green", "<u2"), ("blue", "<u2"),
])


class LasHeader:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            h = f.read(227)
        if h[0:4] != b"LASF":
            raise ValueError(f"{path} is not a LAS file")
        self.version = (h[24], h[25])
        self.header_size = struct.unpack("<H", h[94:96])[0]
        self.offset_to_points = struct.unpack("<I", h[96:100])[0]
        self.n_vlrs = struct.unpack("<I", h[100:104])[0]
        self.point_format = h[104]
        self.record_length = struct.unpack("<H", h[105:107])[0]
        self.n_points_header = struct.unpack("<I", h[107:111])[0]
        (self.sx, self.sy, self.sz,
         self.ox, self.oy, self.oz) = struct.unpack("<6d", h[131:179])
        (self.maxx, self.minx, self.maxy,
         self.miny, self.maxz, self.minz) = struct.unpack("<6d", h[179:227])
        self.raw_header_bytes = None  # filled by read_raw_prefix

        if self.point_format != 2 or self.record_length != POINT_DTYPE.itemsize:
            raise ValueError(
                f"expected point format 2 / 26-byte records, got "
                f"format {self.point_format} / {self.record_length} bytes"
            )

        file_size = os.path.getsize(path)
        self.n_points_in_file = (file_size - self.offset_to_points) // self.record_length
        self.is_partial = self.n_points_in_file < self.n_points_header

    def read_raw_prefix(self):
        """Header + VLRs, reusable verbatim when writing a cropped copy."""
        with open(self.path, "rb") as f:
            self.raw_header_bytes = f.read(self.offset_to_points)
        return self.raw_header_bytes

    def scale_xy(self, pts):
        """Structured array -> float64 UTM x, y."""
        x = pts["X"] * self.sx + self.ox
        y = pts["Y"] * self.sy + self.oy
        return x, y

    def describe(self):
        pct = 100.0 * self.n_points_in_file / self.n_points_header
        return (
            f"{self.path}\n"
            f"  LAS {self.version[0]}.{self.version[1]}, format {self.point_format}, "
            f"{self.record_length} B/point\n"
            f"  points: {self.n_points_in_file:,} in file / "
            f"{self.n_points_header:,} in header ({pct:.1f}%"
            f"{', PARTIAL FILE' if self.is_partial else ''})\n"
            f"  X {self.minx:.1f}..{self.maxx:.1f}  "
            f"Y {self.miny:.1f}..{self.maxy:.1f}  "
            f"Z {self.minz:.1f}..{self.maxz:.1f}"
        )


def iter_chunks(header, chunk_points=20_000_000, start=0, stop=None):
    """Yield (index_of_first_point, structured_array) over [start, stop)."""
    stop = header.n_points_in_file if stop is None else min(stop, header.n_points_in_file)
    rl = header.record_length
    with open(header.path, "rb") as f:
        i = start
        while i < stop:
            n = min(chunk_points, stop - i)
            f.seek(header.offset_to_points + i * rl)
            buf = f.read(n * rl)
            pts = np.frombuffer(buf, dtype=POINT_DTYPE, count=len(buf) // rl)
            yield i, pts
            i += len(pts)


def save_transform(path, minx, miny, res, width, height, extra=None):
    """Pixel<->UTM affine: col = (x-minx)/res, row = (maxy-y)/res (row 0 = north)."""
    d = {
        "crs": "EPSG:32636",
        "minx": minx, "miny": miny, "res": res,
        "width": width, "height": height,
        "maxy": miny + height * res,
    }
    if extra:
        d.update(extra)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    return d


def load_transform(path):
    with open(path) as f:
        return json.load(f)


def utm_to_pixel(t, x, y):
    col = (np.asarray(x) - t["minx"]) / t["res"]
    row = (t["maxy"] - np.asarray(y)) / t["res"]
    return col, row


def pixel_to_utm(t, col, row):
    x = t["minx"] + np.asarray(col) * t["res"]
    y = t["maxy"] - np.asarray(row) * t["res"]
    return x, y
