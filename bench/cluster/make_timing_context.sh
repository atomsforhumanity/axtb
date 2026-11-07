#!/usr/bin/env bash
# Build context for bench/cluster/Dockerfile.timing: axtb at REV (+ the working tree's bench/,
# via make_context.sh), optionally axtb at BASE_REV (its CLI only, for the overhead check), and
# the reference programs: xtb 6.8.0's release tarball ($XTB680_TARBALL:
# xtb-6.8.0-linux-x86_64-ifx-mkl.tar.xz from the grimme-lab/xtb v6.8.0 release) and the g-xtb
# 1.1.0 program ($AXTB_GXTB11_DIR, default reference/gxtb-v1.1.0 of the main checkout).
# Usage: make_timing_context.sh <outdir> [rev] [base_rev]
#        then: docker build -t $REGISTRY/axtb-timing:<tag> <outdir>
# REGISTRY: the image registry (host:port) that holds the bench image, put into the Dockerfile's FROM.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(git -C "$HERE" rev-parse --show-toplevel)
MAIN=$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")
OUT=$1; REV=${2:-HEAD}; BASE=${3:-}
TARBALL=${XTB680_TARBALL:?set XTB680_TARBALL to xtb-6.8.0-linux-x86_64-ifx-mkl.tar.xz}
G11=${AXTB_GXTB11_DIR:-$MAIN/reference/gxtb-v1.1.0}
rm -rf "$OUT"; mkdir -p "$OUT/programs/gxtb-1.1.0" "$OUT/axtb-base"
"$HERE/make_context.sh" "$OUT/axtb" "$REV" > /dev/null
rm -f "$OUT/axtb/Dockerfile" "$OUT/axtb/.dockerignore"
if [ -n "$BASE" ]; then
  git -C "$ROOT" archive "$BASE" | tar -x -C "$OUT/axtb-base"
  echo "$BASE $(git -C "$ROOT" rev-parse "$BASE")" > "$OUT/axtb-base/AXTB_REV"
else
  echo "none" > "$OUT/axtb-base/AXTB_REV"
fi
cp "$TARBALL" "$OUT/programs/"
cp "$G11/gxtb" "$G11/gxtb_parameters" "$G11/basisq" "$G11/eeq" "$OUT/programs/gxtb-1.1.0/"
sed "s#<registry>#${REGISTRY:?set REGISTRY to the image registry}#" "$HERE/Dockerfile.timing" > "$OUT/Dockerfile"
printf 'axtb/target\naxtb-base/target\n**/.git\n' > "$OUT/.dockerignore"
cat "$OUT/axtb/AXTB_REV" "$OUT/axtb-base/AXTB_REV"
du -sh "$OUT"
