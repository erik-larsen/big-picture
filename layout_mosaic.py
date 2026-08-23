#!/usr/bin/env python3
"""Lay out a folder tree of images into one giant mosaic (16:9 default).

Images are gathered recursively (sorted by path, so folders stay grouped),
then placed with a justified-rows layout: every row is scaled to span the
full canvas width while preserving each image's aspect ratio, and the row
height is chosen so the rows collectively fill the canvas height. The
canvas aspect is configurable (--aspect 16:9, 3:1, 2.35, ...) and the
canvas is auto-sized so every image fits at ~native resolution.

Output is a (H, W, 3) uint8 .npy memmap ready for build_pyramid.py, plus
a small preview PNG and a layout.json manifest mapping each source image
to its rectangle in the mosaic.
"""
import argparse
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def default_workers():
    return max(1, (os.cpu_count() or 2) // 2)


def gather(src_dir):
    items = []
    for p in sorted(Path(src_dir).rglob("*")):
        if p.suffix.lower() not in EXTS or not p.is_file():
            continue
        try:
            with Image.open(p) as im:
                w, h = im.size
            items.append({"path": p, "w": w, "h": h, "aspect": w / h})
        except Exception as e:
            print(f"skipping {p}: {e}")
    return items


def layout_rows(items, W, H):
    """Justified rows: returns rows as (row_height, [items]) filling W x H."""
    total_aspect = sum(it["aspect"] for it in items)
    row_h = math.sqrt(W * H / total_aspect)      # ideal row height
    for _ in range(6):                           # converge on canvas height
        rows, cur, cur_a = [], [], 0.0
        for it in items:
            cur.append(it)
            cur_a += it["aspect"]
            if cur_a * row_h >= W:
                rows.append((W / cur_a, cur))    # justify to full width
                cur, cur_a = [], 0.0
        if cur:
            rows.append((min(row_h, W / cur_a), cur))
        total_h = sum(h for h, _ in rows)
        row_h *= H / total_h
    scale = min(H / total_h, 1.0)                # never overflow the canvas
    return [(h * scale, row) for h, row in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="root folder of images")
    ap.add_argument("--out", default=None,
                    help="output .npy (default: <src folder name>_mosaic.npy)")
    ap.add_argument("--width", type=int, default=None,
                    help="canvas width in px; default: auto-sized so every "
                         "image fits at its native resolution")
    ap.add_argument("--aspect", default="16:9",
                    help="canvas aspect ratio as W:H or a float "
                         "(e.g. 16:9, 3:1, 2.35)")
    ap.add_argument("--gap", type=int, default=8,
                    help="visual gap between images, in pixels")
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help="decode/resize threads (default: half the cores)")
    ap.add_argument("--preview", default=None,
                    help="preview PNG (default: <out stem>_preview.png)")
    ap.add_argument("--manifest", default=None,
                    help="manifest JSON (default: <out stem>_layout.json)")
    args = ap.parse_args()

    if args.out is None:
        src = Path(args.src).resolve()   # keep outputs beside their input
        args.out = str(src.parent / f"{src.name}_mosaic.npy")
    stem = Path(args.out).with_suffix("")
    if args.preview is None:
        args.preview = f"{stem}_preview.png"
    if args.manifest is None:
        args.manifest = f"{stem}_layout.json"

    items = gather(args.src)
    if not items:
        raise SystemExit(f"no images found under {args.src}")

    if ":" in args.aspect:
        aw, ah = args.aspect.split(":")
        aspect = float(aw) / float(ah)
    else:
        aspect = float(args.aspect)

    # The viewer's feedback pass packs page coordinates in 12 bits,
    # limiting the pyramid to 4096 tiles (1048576 px) along an axis;
    # in practice disk space runs out well before that.
    MAX_W = 4096 * 256
    if args.width:
        W = args.width
    else:
        # canvas of the chosen aspect whose area holds every image at
        # native size (each cell also carries the inter-image gap)
        g = args.gap
        area = sum((it["w"] + g) * (it["h"] + g) for it in items)
        W = math.ceil(math.sqrt(area * aspect) / 256) * 256
        if W > MAX_W:
            print(f"auto width {W} exceeds the {MAX_W}px limit, capping "
                  f"(images will be scaled down ~{MAX_W / W:.2f}x)")
            W = MAX_W
    H = max(round(W / aspect), 256)
    scale = math.sqrt(W * H / sum(it["w"] * it["h"] for it in items))
    print(f"{len(items)} images -> {W}x{H} mosaic "
          f"({args.aspect}, ~{scale:.2f}x native scale, "
          f"{args.workers} workers)")

    canvas = np.lib.format.open_memmap(
        args.out, mode="w+", dtype=np.uint8, shape=(H, W, 3))
    canvas[:] = (18, 18, 20)

    # 1. solve every placement first, so the manifest order is stable
    #    and the decode work becomes an embarrassingly parallel list
    manifest, g = [], args.gap // 2
    y = 0.0
    for rh, row in layout_rows(items, W, H):
        x, y0, y1 = 0.0, round(y), round(y + rh)
        for it in row:
            x0, x1 = round(x), round(x + it["aspect"] * rh)
            x += it["aspect"] * rh
            ix0, ix1 = min(x0 + g, W), min(x1 - g, W)
            iy0, iy1 = y0 + g, min(y1 - g, H)
            if ix1 - ix0 >= 2 and iy1 - iy0 >= 2:
                manifest.append({"path": str(it["path"]),
                                 "x": ix0, "y": iy0,
                                 "w": ix1 - ix0, "h": iy1 - iy0})
        y += rh

    # 2. decode/resize/blit in parallel. Each image owns a disjoint
    #    rectangle of the memmap, so the writes need no locking, and
    #    Pillow drops the GIL for the decode and resample.
    done, lock = [0], threading.Lock()

    def place(m):
        with Image.open(m["path"]) as im:
            im.draft("RGB", (m["w"] * 2, m["h"] * 2))
            tile = im.convert("RGB").resize((m["w"], m["h"]), Image.LANCZOS)
        canvas[m["y"]:m["y"] + m["h"], m["x"]:m["x"] + m["w"]] = \
            np.asarray(tile)
        with lock:
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == len(manifest):
                print(f"placed {done[0]}/{len(manifest)}",
                      end="\r", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(place, manifest):
            pass

    canvas.flush()
    prev = canvas[::8, ::8]
    Image.fromarray(np.ascontiguousarray(prev)).save(args.preview)
    Path(args.manifest).write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {args.out} ({canvas.nbytes / 1e6:.0f} MB), "
          f"{args.preview}, {args.manifest} ({len(manifest)} placed)")


if __name__ == "__main__":
    main()
