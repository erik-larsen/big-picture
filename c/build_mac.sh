#!/bin/sh
# Build the C viewers against SDL2 (brew) and ANGLE GLES2 (opengl-for-mac).
# Usage: OGL_FOR_MAC=/path/to/opengl-for-mac ./build_mac.sh
set -e

OGL_FOR_MAC="${OGL_FOR_MAC:-$HOME/Github/opengl-for-mac}"
export DYLD_FALLBACK_LIBRARY_PATH="$OGL_FOR_MAC/lib"

CFLAGS="-O2 -Wall $(sdl2-config --cflags) -I$OGL_FOR_MAC/include -DVT_ANGLE_LIB_DIR=\"$OGL_FOR_MAC/lib\""
LIBS="$(sdl2-config --libs) -L$OGL_FOR_MAC/lib -lGLESv2 -lEGL -lm -Wl,-headerpad_max_install_names"

clang $CFLAGS -c vt_core.c -o vt_core.o
clang $CFLAGS vt_viewer.c vt_core.o $LIBS -o vt_viewer
clang $CFLAGS vt_sphere_viewer.c vt_core.o $LIBS -o vt_sphere_viewer

# The ANGLE dylibs carry cwd-relative install names (./libGLESv2.dylib);
# rewrite them to absolute paths so the viewers run from anywhere without
# DYLD_FALLBACK_LIBRARY_PATH, then re-sign (required on arm64).
for exe in vt_viewer vt_sphere_viewer; do
    install_name_tool \
        -change ./libGLESv2.dylib "$OGL_FOR_MAC/lib/libGLESv2.dylib" \
        -change ./libEGL.dylib "$OGL_FOR_MAC/lib/libEGL.dylib" \
        "$exe"
    codesign -f -s - "$exe" 2>/dev/null
done
echo "built: vt_viewer, vt_sphere_viewer"
