#!/usr/bin/env bash
# Interleaved benchmark rounds: every backend x thread count, repeated ROUNDS times.
# Usage: ./run.sh <matdir> <mkl_lib_dir> <avx512_binary> [rounds] [sizes]
# <avx512_binary>: this crate built with `--features faer-avx512` (needs Rust >= 1.89)
# Optional environment (defaults = the laptop run):
#   THREADS="1 4 8"   BACKENDS="faer22 faer24 faer24v4 openblas mkl"   (add aocl for AMD AOCL)
#   OB=<libopenblas.so>  OB_PREFIX=<symbol prefix, scipy_ for the scipy-openblas32 wheel>
#   AOCL=<prefix with lib/libflame.so and lib/libblis-mt.so, from build_aocl.sh>
#   OUT=results/raw.tsv  ENV=results/env.txt  B=<linalg-bench binary>
#   PIN=1: 1 thread -> CPU 2; t threads -> CPUs 0..t-1 (one per physical core where SMT siblings
#   are numbered i and i+ncores, as on the laptop and the cluster nodes). PIN=0: no taskset.
#   Use PIN=0 for faer above 8 threads on the 64-core nodes: t rayon workers pinned to exactly
#   t CPUs run 10-1000x slower there (spin-waiting), unpinned they do not.
set -euo pipefail
cd "$(dirname "$0")"
MAT=$1; MKLDIR=$2; BV4=$3; ROUNDS=${4:-3}; SIZES=${5:-100,266,500,1000,2000,3000}
B=${B:-target/release/linalg-bench}
PIN=${PIN:-1}
# OpenBLAS: the OpenMP build by default (Debian/Ubuntu libopenblas0-openmp), the kind axtb's
# AXTB_LAPACK accepts (OpenMP or sequential builds; it refuses the pthreads build, whose thread
# count is process-wide). The recorded laptop results used the pthreads build.
OB=${OB:-/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so.0}
OB_PREFIX=${OB_PREFIX:-}
MK=$MKLDIR/libmkl_rt.so.3
AOCL=${AOCL:-}
THREADS=${THREADS:-1 4 8}
BACKENDS=${BACKENDS:-faer22 faer24 faer24v4 openblas mkl}
case " $BACKENDS " in *" openblas "*) [ -e "$OB" ] || { echo "OpenBLAS not found at $OB: set OB=<libopenblas.so>" >&2; exit 2; };; esac
export LD_LIBRARY_PATH=$MKLDIR${AOCL:+:$AOCL/lib}
# no thread env vars: every library is set per process through its API
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS RAYON_NUM_THREADS || true
OUT=${OUT:-results/raw.tsv}; ENV=${ENV:-results/env.txt}
{ date; uname -r; lscpu | grep 'Model name'; rustc --version || true; } > $ENV
echo -e "round\tbackend\tthreads\tn\top\tmedian_ms\tmin_ms\treps\tcpu_per_wall\tresidual\tchecksum" > $OUT
for r in $(seq 1 $ROUNDS); do
  for t in $THREADS; do
    case $t in 1) CPUS=2;; *) CPUS=0-$((t-1));; esac
    TS=(taskset -c $CPUS); [ "$PIN" = 1 ] || TS=()
    for be in $BACKENDS; do
      BIN=$B; EXTRA=()
      case $be in
        faer24v4) ARG=faer24; BIN=$BV4; OPS=evd,gen,genmm,gemm;;
        faer*) ARG=$be; OPS=evd,gen,genmm,gemm;;
        openblas) ARG=lapack:$OB; EXTRA=(${OB_PREFIX:+--prefix $OB_PREFIX}); OPS=evd,gen,gen2t,evr40,mr40,genr40,gemm;;
        mkl) ARG=lapack:$MK; OPS=evd,gen,gen2t,evr40,mr40,genr40,gemm;;
        aocl) ARG=lapack:$AOCL/lib/libflame.so; EXTRA=(--blas $AOCL/lib/libblis-mt.so); OPS=evd,gen,gen2t,evr40,mr40,genr40,gemm;;
      esac
      echo "round $r threads $t $be $(date +%T)" >&2
      "${TS[@]}" $BIN --backend $ARG "${EXTRA[@]}" --threads $t --sizes $SIZES --ops $OPS --budget 1.0 --mat $MAT \
        2>>$ENV | sed "s/^faer24\t/$be\t/; s/^/$r\t/" >> $OUT
    done
  done
done
echo done >&2
