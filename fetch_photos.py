#!/usr/bin/env python3
"""Fetch royalty-free photographs to build a real-world mosaic.

Three sources, all via their official APIs (each needs a free key):

  unsplash  https://unsplash.com/developers   UNSPLASH_ACCESS_KEY
  pexels    https://www.pexels.com/api/       PEXELS_API_KEY
  pixabay   https://pixabay.com/api/docs/     PIXABAY_API_KEY

All three are asked for photographs only, so illustrations, vector art,
and AI artwork are filtered out server-side.

Licence note: Unsplash grants the right to distribute images (the only
carve-outs are selling without significant modification, and compiling a
competing service), and Pexels restricts redistribution to "other stock
photo or wallpaper platforms" — so a fetched set from either may be
committed alongside a project like this one. Pixabay is stricter: it
forbids distributing content on a standalone basis, collections
included, so keep Pixabay downloads out of version control.

Attribution isn't required by any of the three licences, but both the
Unsplash and Pexels API terms ask for it, and it's the decent thing to
do — CREDITS.md credits every photographer, with a link.

    ./fetch_photos.py --source unsplash --count 512
    ./layout_mosaic.py pics/photos_scenery --aspect 3:1
    ./build_pyramid.py pics/photos_scenery_mosaic.npy
    ./vt_sphere_viewer.py pics/photos_scenery_mosaic_pyramid
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UA = "big-picture/1.0 (https://github.com/erik-larsen/big-picture)"


def get(url, headers=None, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def get_json(url, headers=None):
    return json.loads(get(url, headers))


# ------------------------------------------------------------- sources
# Each returns (records, extra_headers). A record is
# {url, id, page, author, author_url}.

def from_unsplash(key, query, count, width, quality):
    hdr = {"Authorization": f"Client-ID {key}"}
    out, page, per = [], 1, 30                    # 30 is the API maximum
    while len(out) < count:
        u = ("https://api.unsplash.com/search/photos?"
             + urllib.parse.urlencode({"query": query, "per_page": per,
                                       "page": page,
                                       "orientation": "landscape",
                                       "content_filter": "high"}))
        res = get_json(u, hdr).get("results", [])
        if not res:
            break
        for p in res:
            out.append({
                # raw is Imgix-backed, so ask for an exact width
                "url": (p["urls"]["raw"] + f"&w={width}&q={quality}"
                        "&fm=jpg&fit=max"),
                "id": p["id"],
                "page": p["links"]["html"],
                "author": p["user"]["name"],
                "author_url": p["user"]["links"]["html"],
                # Unsplash API terms: ping this when a photo is downloaded
                "track": p["links"].get("download_location"),
            })
        print(f"  page {page}: {len(out)} listed")
        page += 1
        time.sleep(0.3)
    return out[:count], hdr


def from_pexels(key, query, count, width, quality):
    hdr = {"Authorization": key}
    out, page, per = [], 1, 80                    # 80 is the API maximum
    while len(out) < count:
        u = ("https://api.pexels.com/v1/search?"
             + urllib.parse.urlencode({"query": query, "per_page": per,
                                       "page": page,
                                       "orientation": "landscape"}))
        res = get_json(u, hdr).get("photos", [])
        if not res:
            break
        for p in res:
            out.append({
                "url": (p["src"]["original"] + "?auto=compress&cs=tinysrgb"
                        f"&fm=jpg&w={width}&q={quality}"),
                "id": str(p["id"]),
                "page": p["url"],
                "author": p.get("photographer", ""),
                "author_url": p.get("photographer_url", ""),
                "track": None,
            })
        print(f"  page {page}: {len(out)} listed")
        page += 1
        time.sleep(0.3)
    return out[:count], hdr


def from_pixabay(key, query, count, width, quality):   # fixed sizes only
    out, page, per = [], 1, 200                   # 200 is the API maximum
    while len(out) < count:
        u = ("https://pixabay.com/api/?"
             + urllib.parse.urlencode({"key": key, "q": query,
                                       "image_type": "photo",
                                       "orientation": "horizontal",
                                       "per_page": per, "page": page,
                                       "safesearch": "true"}))
        data = get_json(u)
        res = data.get("hits", [])
        if not res:
            break
        for p in res:
            out.append({
                "url": p.get("largeImageURL") or p["webformatURL"],
                "id": str(p["id"]),
                "page": p.get("pageURL", ""),
                "author": p.get("user", ""),
                "author_url": "",
                "track": None,
            })
        print(f"  page {page}: {len(out)} listed")
        if len(out) >= data.get("totalHits", 0):
            break
        page += 1
        time.sleep(0.5)
    return out[:count], {}


SOURCES = {
    "unsplash": (from_unsplash, "UNSPLASH_ACCESS_KEY",
                 "https://unsplash.com/developers"),
    "pexels": (from_pexels, "PEXELS_API_KEY",
               "https://www.pexels.com/api/"),
    "pixabay": (from_pixabay, "PIXABAY_API_KEY",
                "https://pixabay.com/api/docs/"),
}


def write_credits(out, source, records):
    """One line per photographer rather than per file: attribution is owed
    to the people, and a 1000-line file nobody reads serves no one."""
    by_author = {}
    for r in records:
        by_author.setdefault((r["author"] or "unknown", r["author_url"]),
                             []).append(r)
    lines = ["# Photo credits", "",
             f"{len(records)} photographs from {source.capitalize()}, "
             f"fetched with `fetch_photos.py`.", "",
             f"Attribution is not required by the {source.capitalize()} "
             "licence, but is offered here with thanks — and their API "
             "terms ask for it.", ""]
    for (name, url), rs in sorted(by_author.items()):
        who = f"[{name}]({url})" if url else name
        lines.append(f"- {who} — {len(rs)} photo"
                     + ("s" if len(rs) != 1 else ""))
    (out / "CREDITS.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=sorted(SOURCES), default="unsplash")
    ap.add_argument("--key", default=None,
                    help="API key (or use the source's env var)")
    ap.add_argument("--query", default="scenery")
    ap.add_argument("--count", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1080,
                    help="requested width in px (unsplash/pexels)")
    ap.add_argument("--quality", type=int, default=78,
                    help="JPEG quality (unsplash/pexels)")
    ap.add_argument("--budget-mb", type=float, default=0,
                    help="stop downloading once this many MB are on disk")
    ap.add_argument("--out", default=None,
                    help="output folder (default: pics/photos_<query>)")
    args = ap.parse_args()

    fetch, env_var, signup = SOURCES[args.source]
    key = args.key or os.environ.get(env_var)
    if not key:
        raise SystemExit(
            f"error: no {args.source} API key\n\n"
            f"Get a free one at {signup}, then either:\n\n"
            f"    export {env_var}=...\n"
            f"    ./fetch_photos.py --source {args.source}\n\n"
            f"or pass it directly:\n\n"
            f"    ./fetch_photos.py --source {args.source} --key YOUR_KEY\n")

    stem = args.query.replace(" ", "_")
    out = Path(args.out or Path("pics") / (f"pixabay_{stem}"
                                           if args.source == "pixabay"
                                           else f"photos_{stem}"))
    out.mkdir(parents=True, exist_ok=True)

    print(f"searching {args.source} for '{args.query}' (photographs only)...")
    try:
        records, hdr = fetch(key, args.query, args.count,
                             args.width, args.quality)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"API request failed: HTTP {e.code} {e.reason}\n"
                         f"(check the key, and the source's rate limit)")
    if not records:
        raise SystemExit("no results")

    print(f"downloading {len(records)} photos to {out}/")
    got, warned, total = 0, False, 0
    for i, r in enumerate(records):
        r["file"] = f"{i:04d}_{r['id']}.jpg"
        dest = out / r["file"]
        if not dest.exists():
            try:
                dest.write_bytes(get(r["url"]))
            except Exception as e:
                print(f"  failed {r['url']}: {e}")
                continue
            if r.get("track"):        # Unsplash download-tracking ping
                try:
                    get(r["track"], hdr, timeout=15)
                except Exception:
                    if not warned:
                        print("  (download tracking rate-limited; "
                              "continuing without it)")
                        warned = True
            time.sleep(0.05)
        got += 1
        total += dest.stat().st_size if dest.exists() else 0
        if got % 50 == 0:
            print(f"  {got}/{len(records)}  ({total / 1e6:.0f} MB)")
        if args.budget_mb and total >= args.budget_mb * 1e6:
            print(f"  reached the {args.budget_mb:.0f} MB budget "
                  f"after {got} photos")
            records = records[:got]
            break

    write_credits(out, args.source, records)
    print(f"\n{got} photos, {total / 1e6:.0f} MB in {out}/  "
          f"(+ CREDITS.md)\n\n"
          f"Next:\n"
          f"    ./layout_mosaic.py {out} --aspect 3:1\n"
          f"    ./build_pyramid.py {out}_mosaic.npy\n"
          f"    ./vt_sphere_viewer.py {out}_mosaic_pyramid\n")


if __name__ == "__main__":
    main()
