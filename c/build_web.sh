#!/bin/sh
# Build the sphere viewer for the browser (WebAssembly + WebGL 1).
# Usage: ./build_web.sh   (needs emcc on PATH: source <emsdk>/emsdk_env.sh)
# Output: ../web/vt_sphere_viewer.{js,wasm}, loaded by ../web/index.html
set -e

if ! command -v emcc >/dev/null 2>&1; then
    echo "emcc not found: install emsdk, then source <emsdk>/emsdk_env.sh" >&2
    exit 1
fi

# FS and run dependencies: the page writes meta.json and layout.json into
# the in-memory filesystem before main(), so vt_init reads them unchanged.
emcc -O2 -Wall vt_sphere_viewer.c vt_core.c -o ../web/vt_sphere_viewer.js \
    -sUSE_SDL=2 -sALLOW_MEMORY_GROWTH -sENVIRONMENT=web \
    -sEXPORTED_RUNTIME_METHODS=FS,addRunDependency,removeRunDependency

echo "built: ../web/vt_sphere_viewer.js + .wasm"
