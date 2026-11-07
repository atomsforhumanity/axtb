#!/usr/bin/env bash
# Build AMD AOCL (default 5.3) from AMD's GitHub sources (BSD-3-Clause), no root needed:
#   AOCL-Utils (CPU detection), AOCL-BLAS = BLIS "amdzen" (Zen 1-5 kernels + runtime dispatch,
#   OpenMP threading -> libblis-mt.so), AOCL-LAPACK = libFLAME (-DENABLE_AMD_FLAGS, linked to
#   AOCL-BLAS, OpenMP -> libflame.so). LP64 (32-bit integers), like OpenBLAS/MKL in the harness.
# We build from source instead of using AMD's prebuilt tarball: the tarball comes under AMD's
# evaluation EULA (confidentiality clause, use only with AMD products), the sources do not.
#
# Usage: build_aocl.sh <prefix> [srcdir] [jobs]
# Then: linalg-bench --backend lapack:<prefix>/lib/libflame.so --blas <prefix>/lib/libblis-mt.so
set -euo pipefail
PREFIX=$(realpath -m "$1")
SRC=$(realpath -m "${2:-$PREFIX-src}")
JOBS=${3:-8}
TAG=${AOCL_TAG:-5.3}
mkdir -p "$SRC" "$PREFIX"
for r in aocl-utils blis libflame; do
  [ -d "$SRC/$r" ] || git clone -q --depth 1 --branch "$TAG" "https://github.com/amd/$r.git" "$SRC/$r"
done

cmake -S "$SRC/aocl-utils" -B "$SRC/build-utils" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DBUILD_SHARED_LIBS=ON
cmake --build "$SRC/build-utils" -j "$JOBS"
cmake --install "$SRC/build-utils"

(cd "$SRC/blis" && ./configure --prefix="$PREFIX" --enable-cblas --enable-threading=openmp --enable-shared --disable-static amdzen \
  && make -j "$JOBS" && make install)

# libFLAME links libflame.so with "-shared ... -pie"; GCC 15 lets the later -pie win and tries to
# link an executable (undefined main/cexp). -pie means nothing for a shared library: drop it.
sed -i 's/set(LINKER_FLAGS "-pie /set(LINKER_FLAGS "/' "$SRC/libflame/cmake/CompilerFlags.cmake"
cmake -S "$SRC/libflame" -B "$SRC/build-flame" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DENABLE_AMD_FLAGS=ON -DENABLE_AOCL_BLAS=ON -DAOCL_ROOT="$PREFIX" \
  -DLIBAOCLUTILS_LIBRARY_PATH="$PREFIX/lib/libaoclutils.so" -DLIBAOCLUTILS_INCLUDE_PATH="$PREFIX/include"
cmake --build "$SRC/build-flame" -j "$JOBS"
cmake --install "$SRC/build-flame"

for r in aocl-utils blis libflame; do echo "$r $TAG $(git -C "$SRC/$r" rev-parse HEAD)"; done > "$PREFIX/SOURCES.txt"
ls -l "$PREFIX"/lib/*.so* >&2
