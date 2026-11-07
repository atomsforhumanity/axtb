#!/usr/bin/env bash
# Assemble the build context for bench/cluster/Dockerfile: the repo at REV (default HEAD) with
# the working tree's bench/ harness on top (uncommitted benchmark changes are included; the
# axtb sources are exactly REV, recorded in AXTB_REV).
# Usage: make_context.sh <outdir> [rev]
#   then: docker build --build-arg BASE_IMAGE=<image> -t <registry>/axtb-bench:<tag> <outdir>
set -euo pipefail
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
OUT=$1; REV=${2:-HEAD}
rm -rf "$OUT"; mkdir -p "$OUT"
git -C "$ROOT" archive "$REV" | tar -x -C "$OUT"
(cd "$ROOT" && git ls-files -co --exclude-standard bench | grep -v -e '^bench/results/' -e '^bench/linalg/results/') \
  | rsync -a --files-from=- "$ROOT/" "$OUT/"
cp "$ROOT/bench/cluster/Dockerfile" "$OUT/Dockerfile"
printf 'target\n**/target\n**/target-v4\n.git\n' > "$OUT/.dockerignore"
echo "$REV $(git -C "$ROOT" rev-parse "$REV")$(git -C "$ROOT" diff --quiet "$REV" -- bench || echo ' + working-tree bench/')" > "$OUT/AXTB_REV"
du -sh "$OUT"
