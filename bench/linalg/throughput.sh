#!/usr/bin/env bash
# Throughput mode: NPROC concurrent single-threaded processes, one per physical core
# (what thousands of independent jobs on a node look like), vs. one process alone.
# Usage: ./throughput.sh <matdir> <mkl_lib_dir> [nproc] [sizes]
# Optional environment: BACKENDS="faer22 openblas mkl" (also faer24, faer24v4 with BV4=<binary>,
# aocl with AOCL=<prefix>), OB, OB_PREFIX (see run.sh), OUT=results/throughput.tsv
set -euo pipefail
cd "$(dirname "$0")"
MAT=$1; MKLDIR=$2; NPROC=${3:-8}; SIZES=${4:-266,500,1000}
B=target/release/linalg-bench
# OpenBLAS: the OpenMP build by default (Debian/Ubuntu libopenblas0-openmp), the kind axtb's
# AXTB_LAPACK accepts (OpenMP or sequential builds; it refuses the pthreads build, whose thread
# count is process-wide). The recorded laptop results used the pthreads build.
OB=${OB:-/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so.0}
OB_PREFIX=${OB_PREFIX:-}
MK=$MKLDIR/libmkl_rt.so.3
AOCL=${AOCL:-}
BV4=${BV4:-}
BACKENDS=${BACKENDS:-faer22 openblas mkl}
case " $BACKENDS " in *" openblas "*) [ -e "$OB" ] || { echo "OpenBLAS not found at $OB: set OB=<libopenblas.so>" >&2; exit 2; };; esac
export LD_LIBRARY_PATH=$MKLDIR${AOCL:+:$AOCL/lib}
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS RAYON_NUM_THREADS || true
OUT=${OUT:-results/throughput.tsv}
TMP=$(mktemp -d)
echo -e "mode\tbackend\tthreads\tn\top\tmedian_ms\tmin_ms\treps\tcpu_per_wall\tresidual\tchecksum" > $OUT
for be in $BACKENDS; do
  BIN=$B; EXTRA=()
  case $be in
    faer24v4) ARG=faer24; BIN=$BV4;;
    faer*) ARG=$be;;
    openblas) ARG=lapack:$OB; EXTRA=(${OB_PREFIX:+--prefix $OB_PREFIX});;
    mkl) ARG=lapack:$MK;;
    aocl) ARG=lapack:$AOCL/lib/libflame.so; EXTRA=(--blas $AOCL/lib/libblis-mt.so);;
  esac
  echo "throughput $be x$NPROC $(date +%T)" >&2
  for i in $(seq 0 $((NPROC-1))); do
    taskset -c $i $BIN --backend $ARG "${EXTRA[@]}" --threads 1 --sizes $SIZES --ops evd,gen --budget 3.0 --nocheck --mat $MAT 2>/dev/null \
      | sed "s/^faer24\t/$be\t/; s/^/concurrent$NPROC\t/" > $TMP/tp_${be}_$i.tmp &
  done
  wait
  cat $TMP/tp_${be}_*.tmp >> $OUT; rm -f $TMP/tp_${be}_*.tmp
done
rmdir $TMP
