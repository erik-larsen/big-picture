#!/usr/bin/env python3
"""Package a tile pyramid for the web viewer (web/index.html).

Writes a self-contained folder any static host can serve:

  <out>/meta.json      pyramid metadata, as build_pyramid.py wrote it
  <out>/layout.json    each image's rectangle, paths cut to basenames
  <out>/credits.json   basename -> photographer, when the photos came
                       from fetch_photos.py (it writes photos.json)
  <out>/L{level}/...   the tiles, hard-linked where possible (no copy)

Build the pyramid as JPEG for the web -- PNG tiles run 5-10x larger:

    ./build_pyramid.py pics/photos_mountains_mosaic.npy --format jpg --quality 80
    ./export_web.py pics/photos_mountains_mosaic_pyramid
    python3 -m http.server -d web   # with web/tiles -> the export folder

Serve the folder beside web/ as tiles/, or anywhere else and point the page
at it with ?tiles=URL.
"""
import argparse
import json
import os
import shutil
from pathlib import Path


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:                     # another filesystem
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pyramid", help="pyramid dir from build_pyramid.py")
    ap.add_argument("--out", default=None,
                    help="output dir (default: <stem>_web beside the pyramid)")
    args = ap.parse_args()

    pyr = Path(args.pyramid)
    meta = json.loads((pyr / "meta.json").read_text())
    if meta.get("format") != "jpg":
        print("note: PNG tiles -- rebuild with --format jpg for the web")
    stem = pyr.name.removesuffix("_pyramid")
    out = Path(args.out or pyr.parent / f"{stem}_web")
    out.mkdir(parents=True, exist_ok=True)

    shutil.copy2(pyr / "meta.json", out / "meta.json")

    layout_path = pyr.parent / f"{stem}_layout.json"
    credits = {}
    if layout_path.exists():
        layout = json.loads(layout_path.read_text())
        photo_dirs = {Path(r["path"]).parent for r in layout}
        for r in layout:
            r["path"] = Path(r["path"]).name
        (out / "layout.json").write_text(json.dumps(layout, separators=(",", ":")))
        for d in photo_dirs:
            if (d / "photos.json").exists():
                for p in json.loads((d / "photos.json").read_text()):
                    credits[p["file"]] = {k: p[k] for k in
                                          ("author", "author_url", "page")}
            if (d / "CREDITS.md").exists():
                shutil.copy2(d / "CREDITS.md", out / "CREDITS.md")
        print(f"layout: {len(layout)} images, {len(credits)} credited")
    else:
        print(f"no {layout_path.name}: no hover outlines or credits")
    if credits:
        (out / "credits.json").write_text(json.dumps(credits, indent=0))

    n, size = 0, 0
    for level in range(meta["levels"]):
        src, dst = pyr / f"L{level}", out / f"L{level}"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir()
        for f in src.iterdir():
            link_or_copy(f, dst / f.name)
            n += 1
            size += f.stat().st_size
    print(f"{n} tiles, {size / 1e6:.0f} MB -> {out}/")


if __name__ == "__main__":
    main()
