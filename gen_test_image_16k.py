#!/usr/bin/env python3
"""Step 1: Generate a 16384x16384 synthetic UV/RGB test image.

The image encodes U in red, V in green, and a mix of a checkerboard and a
zone plate (aliasing test) in blue, with tile-grid lines overlaid so that
tile seams and LOD transitions are easy to spot in the viewer.

Output is a numpy .npy file (written as a memmap, so peak RAM stays low)
plus a small PNG preview.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIZE = 16384
STRIP = 1024  # rows generated per pass
CELL = 256    # grid cell / tile size

FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def load_font(size):
    for path in FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    raise SystemExit("no TrueType font found; add one to FONTS")


def col_name(i):
    """0 -> A, 25 -> Z, 26 -> AA ... (spreadsheet column naming)."""
    name = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        name = chr(ord("A") + r) + name
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="pics/test_image_16k.npy")
    ap.add_argument("--preview", default=None,
                    help="preview PNG (default: beside --out)")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.preview is None:
        args.preview = str(out.parent / "test_image_preview.png")

    img = np.lib.format.open_memmap(
        args.out, mode="w+", dtype=np.uint8, shape=(SIZE, SIZE, 3)
    )
    preview = np.zeros((SIZE // 8, SIZE // 8, 3), np.uint8)

    x = np.arange(SIZE, dtype=np.float32)
    u8_u = (x / (SIZE - 1) * 255.0).astype(np.uint8)  # U ramp, reused per strip
    font = load_font(72)

    for y0 in range(0, SIZE, STRIP):
        ys = np.arange(y0, y0 + STRIP, dtype=np.float32)
        strip = np.empty((STRIP, SIZE, 3), np.uint8)

        # R = U, G = V
        strip[:, :, 0] = u8_u[None, :]
        strip[:, :, 1] = (ys / (SIZE - 1) * 255.0).astype(np.uint8)[:, None]

        # B = 64px checkerboard blended with a zone plate centered on the image
        xi = np.arange(SIZE)
        checker = (((xi[None, :] // 64) ^ (ys.astype(np.int64)[:, None] // 64)) & 1)
        dx = x[None, :] - SIZE / 2
        dy = (ys - SIZE / 2)[:, None]
        r2 = dx * dx + dy * dy
        zone = np.sin(np.pi * r2 / 32768.0)  # Nyquist at r=8192
        b = 96.0 + 48.0 * checker + 64.0 * zone
        strip[:, :, 2] = np.clip(b, 0, 255).astype(np.uint8)

        # Tile grid: brighten 256px tile lines, stronger every 1024px
        gx = (xi[None, :] % 256 < 2) | (xi[None, :] % 256 >= 254)
        gy = (ys.astype(np.int64)[:, None] % 256 < 2) | (ys.astype(np.int64)[:, None] % 256 >= 254)
        grid = gx | gy
        Gx = (xi[None, :] % 1024 < 3) | (xi[None, :] % 1024 >= 1021)
        Gy = (ys.astype(np.int64)[:, None] % 1024 < 3) | (ys.astype(np.int64)[:, None] % 1024 >= 1021)
        grid2 = Gx | Gy
        f = strip.astype(np.int16)
        f[grid] += 48
        f[grid2] = 255 - (255 - f[grid2]) // 4  # near-white major grid
        strip = np.clip(f, 0, 255).astype(np.uint8)

        # cell addresses (A1-style: alpha column, numeric row) at the
        # center of every 256px grid cell, white with a black outline so
        # the glyph edges stay legible over any background
        pim = Image.fromarray(strip)
        d = ImageDraw.Draw(pim)
        for cy in range(y0 // CELL, (y0 + STRIP) // CELL):
            for cx in range(SIZE // CELL):
                d.text((cx * CELL + CELL // 2, cy * CELL + CELL // 2 - y0),
                       f"{col_name(cx)}{cy + 1}", font=font, anchor="mm",
                       fill=(255, 255, 255), stroke_width=4,
                       stroke_fill=(0, 0, 0))
        strip = np.asarray(pim)

        img[y0:y0 + STRIP] = strip
        # 8x8 box-filter preview
        p = strip.reshape(STRIP // 8, 8, SIZE // 8, 8, 3).mean(axis=(1, 3))
        preview[y0 // 8:(y0 + STRIP) // 8] = p.astype(np.uint8)
        print(f"rows {y0 + STRIP}/{SIZE}", end="\r", flush=True)

    img.flush()
    Image.fromarray(preview).save(args.preview)
    print(f"\nwrote {args.out} ({SIZE}x{SIZE}x3, "
          f"{img.nbytes / 1e6:.0f} MB) and {args.preview}")


if __name__ == "__main__":
    main()
