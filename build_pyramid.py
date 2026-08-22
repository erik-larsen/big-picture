#!/usr/bin/env python3
"""Step 2: Build a tiled image pyramid from a (H, W, 3) uint8 .npy image.

Works for any rectangular image (e.g. the square 16k test image or a 16:9
mosaic). Levels are numbered 0 (full resolution) .. N (coarsest, a single
tile). Dimensions that stop dividing evenly are handled with partial edge
tiles, padded out by edge replication.

Each stored tile is TILE x TILE payload plus a BORDER-pixel skirt on every
side, baked from neighboring pixels (edge-clamped at image boundaries), so
the viewer can bilinear-filter inside a tile without seams.

Layout:
  pyramid/meta.json
  pyramid/L{level}/{y}_{x}.png   (stored size = TILE + 2*BORDER per side)
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

TILE = 256
BORDER = 2


def ceil_div(a, b):
    return -(-a // b)


def downsample_to(src, dst):
    """2x2 box-filter src memmap into dst memmap, strip-wise.

    Odd dimensions are edge-replicated by one row/column first, so dst is
    ceil(src/2) along each axis."""
    h, w = src.shape[:2]
    strip = 2048
    for y0 in range(0, h, strip):
        s = np.asarray(src[y0:y0 + strip]).astype(np.uint16)
        if s.shape[0] % 2:
            s = np.concatenate([s, s[-1:]], axis=0)
        if w % 2:
            s = np.concatenate([s, s[:, -1:]], axis=1)
        d = (s[0::2, 0::2] + s[0::2, 1::2] + s[1::2, 0::2] + s[1::2, 1::2] + 2) >> 2
        dst[y0 // 2:(y0 + strip) // 2] = d.astype(np.uint8)


def cut_tiles(level_img, level, out_dir, fmt):
    h, w = level_img.shape[:2]
    ny, nx = ceil_div(h, TILE), ceil_div(w, TILE)
    ldir = out_dir / f"L{level}"
    ldir.mkdir(parents=True, exist_ok=True)
    for ty in range(ny):
        y0, y1 = ty * TILE - BORDER, (ty + 1) * TILE + BORDER
        rows = level_img[max(y0, 0):min(y1, h)]
        for tx in range(nx):
            x0, x1 = tx * TILE - BORDER, (tx + 1) * TILE + BORDER
            t = rows[:, max(x0, 0):min(x1, w)]
            pad = ((max(-y0, 0), max(y1 - h, 0)),
                   (max(-x0, 0), max(x1 - w, 0)), (0, 0))
            if any(p != (0, 0) for p in pad[:2]):
                t = np.pad(t, pad, mode="edge")
            im = Image.fromarray(np.ascontiguousarray(t))
            if fmt == "png":
                im.save(ldir / f"{ty}_{tx}.png", compress_level=1)
            else:
                im.save(ldir / f"{ty}_{tx}.jpg", quality=95)
        print(f"L{level}: row {ty + 1}/{ny}", end="\r", flush=True)
    print()
    return nx * ny


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src_pos", nargs="?", metavar="SRC",
                    help="source .npy (same as --src)")
    ap.add_argument("--src", default="test_image_16k.npy")
    ap.add_argument("--out", default=None,
                    help="output dir (default: <src stem>_pyramid)")
    ap.add_argument("--format", choices=["png", "jpg"], default="png")
    args = ap.parse_args()
    if args.src_pos:                       # allow a bare positional path
        args.src = args.src_pos
    if args.out is None:
        args.out = f"{Path(args.src).stem}_pyramid"

    src = np.load(args.src, mmap_mode="r")
    H, W = src.shape[:2]
    assert src.ndim == 3 and src.shape[2] == 3

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    dims, w, h = [(W, H)], W, H
    while max(w, h) > TILE:
        w, h = ceil_div(w, 2), ceil_div(h, 2)
        dims.append((w, h))
    n_levels = len(dims)
    print(f"{W}x{H}, {n_levels} levels (L0 full res .. "
          f"L{n_levels - 1} single {TILE}px tile)")

    tmp = out / "_levels"
    tmp.mkdir()
    levels = [src]
    for l in range(1, n_levels):
        lw, lh = dims[l]
        mm = np.lib.format.open_memmap(
            tmp / f"L{l}.npy", mode="w+", dtype=np.uint8, shape=(lh, lw, 3))
        downsample_to(levels[-1], mm)
        levels.append(mm)

    total = 0
    for l, img in enumerate(levels):
        total += cut_tiles(img, l, out, args.format)

    del levels
    shutil.rmtree(tmp)

    meta = {
        "image_width": W,
        "image_height": H,
        "tile_size": TILE,
        "border": BORDER,
        "levels": n_levels,
        "format": args.format,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {total} tiles to {out}/ + meta.json")


if __name__ == "__main__":
    main()
