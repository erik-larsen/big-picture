#!/usr/bin/env python3
"""Generate a synthetic folder tree of "photos" to exercise the pipeline.

Real photo collections need an API key (fetch_photos.py) or your own
pictures. This makes a stand-in out of thin air: gradient cards, each
labelled with its folder path, index, and pixel size, filed into nested
subfolders. Because every card announces where it came from, the mosaic
doubles as a readable map of the layout algorithm — you can see which
folder a picture belongs to while flying around it.

Deterministic: the same --seed and --count always produce the same tree.

    ./gen_test_image_tree.py                      # -> pics/phototree/
    ./layout_mosaic.py pics/phototree --aspect 3:1
    ./build_pyramid.py pics/phototree_mosaic.npy
    ./vt_sphere_viewer.py pics/phototree_mosaic_pyramid
"""
import argparse
import colorsys
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

FOLDERS = ["vacation/beach", "vacation/city", "vacation/mountains",
           "pets", "food/breakfast", "food/dinner",
           "art/abstract", "art/photos", "family/2024", "family/2025"]


def default_workers():
    return max(1, (os.cpu_count() or 2) // 2)


def make_card(folder, index, w, h, hue1, hue2):
    """A two-tone diagonal gradient with the card's identity written on it."""
    c1 = np.array(colorsys.hsv_to_rgb(hue1, 0.62, 0.95)) * 255
    c2 = np.array(colorsys.hsv_to_rgb(hue2, 0.75, 0.55)) * 255
    gy = np.linspace(0, 1, h)[:, None, None]
    gx = np.linspace(0, 1, w)[None, :, None]
    t = gy * 0.75 + gx * 0.25
    im = Image.fromarray((c1 * (1 - t) + c2 * t).astype(np.uint8))

    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w - 1, h - 1], outline=(255, 255, 255),
                width=max(2, max(w, h) // 180))
    d.multiline_text((w // 2, h // 2), f"{folder}\n#{index:03d}\n{w}x{h}",
                     fill=(255, 255, 255), anchor="mm", align="center",
                     font_size=min(w, h) // 9,
                     stroke_width=max(2, min(w, h) // 120),
                     stroke_fill=(0, 0, 0))
    return im


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="pics/phototree",
                    help="output folder (default: pics/phototree)")
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--seed", type=int, default=11,
                    help="same seed + count reproduces the same tree")
    ap.add_argument("--min-size", type=int, default=480)
    ap.add_argument("--max-size", type=int, default=1600)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--workers", type=int, default=default_workers())
    args = ap.parse_args()

    # draw every card's parameters up front, so the tree depends only on
    # the seed and not on the order threads happen to finish in
    rng = random.Random(args.seed)
    cards = [{
        "folder": rng.choice(FOLDERS),
        "index": i,
        "w": rng.randint(args.min_size, args.max_size),
        "h": rng.randint(args.min_size, args.max_size),
        "hue1": (h1 := rng.random()),
        "hue2": (h1 + rng.uniform(0.08, 0.3)) % 1.0,
    } for i in range(args.count)]

    root = Path(args.out)
    for f in {c["folder"] for c in cards}:
        (root / f).mkdir(parents=True, exist_ok=True)

    done, lock = [0], threading.Lock()

    def write(c):
        im = make_card(c["folder"], c["index"], c["w"], c["h"],
                       c["hue1"], c["hue2"])
        im.save(root / c["folder"] / f"img_{c['index']:03d}.jpg",
                quality=args.quality)
        with lock:
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == len(cards):
                print(f"{done[0]}/{len(cards)} cards", end="\r", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(write, cards):
            pass

    px = sum(c["w"] * c["h"] for c in cards)
    print(f"\nwrote {len(cards)} images ({px / 1e6:.0f} Mpx) under {root}/ "
          f"in {len({c['folder'] for c in cards})} folders\n\n"
          f"Next:\n"
          f"    ./layout_mosaic.py {root} --aspect 3:1\n"
          f"    ./build_pyramid.py {root}_mosaic.npy\n"
          f"    ./vt_sphere_viewer.py {root}_mosaic_pyramid\n")


if __name__ == "__main__":
    main()
