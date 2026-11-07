#!/usr/bin/env bash
# One benchmark task on a whole cluster node (see run.yaml). ./task.json: {"id": ..., "kind": K}
#   linalg     bench/linalg/run.sh: 6 backends x 1/8/32/64 threads x 3 interleaved rounds
#   linalg-unpinned  the same at 32/64 threads without taskset
#   linalg-tp  bench/linalg/throughput.sh: 64 and 128 concurrent single-thread processes
#   suite      bench/run_benchmarks.py: axtb vs xtb 6.7.1 vs tblite 0.8.0, 1/8/32/64 threads
# Thread counts are explicit; the pod must reserve the node ($NTHREADS >= 64). Never nproc here.
set -euo pipefail
KIND=$(python3 -c 'import json; print(json.load(open("task.json"))["kind"])')
D=$PWD/out   # results dir (not OUT: run.sh and throughput.sh read OUT)
[ "${NTHREADS:-0}" -ge 64 ] || { echo "needs threads_per_task >= 64 (NTHREADS=${NTHREADS:-unset})"; exit 2; }
{
  date -u; echo "node: ${NODE_NAME:-$(hostname)}  NTHREADS=$NTHREADS  kind=$KIND"
  cat /opt/axtb/AXTB_REV /opt/axtb/RUST_VERSIONS /opt/aocl-5.3/SOURCES.txt
  lscpu; echo "cpu0 SMT siblings: $(cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list)"
  echo "cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)"
  echo "load before: $(cat /proc/loadavg)"
} > "$D/node.txt"
L=/opt/axtb/bench/linalg
OB=$(python3 -c 'import scipy_openblas32 as s; print(s.get_lib_dir())')/libscipy_openblas.so
export OB OB_PREFIX=scipy_ AOCL=/opt/aocl-5.3
MKLDIR=/usr/local/lib
ALL="faer22 faer24 faer24v4 openblas mkl aocl"
case $KIND in
  linalg)
    cp -r "$RUN_DIR/mats" ./mats   # GFN2 H0/S of 11 taxol copies (n=3289): the laptop run's files
    sha256sum mats/*.bin >> "$D/node.txt"
    THREADS="1 8 32 64" BACKENDS="$ALL" OUT=$D/raw.tsv ENV=$D/env.txt \
      $L/run.sh "$PWD/mats" $MKLDIR $L/target-v4/release/linalg-bench 3 2> "$D/progress.log"
    python3 $L/summarize.py "$D/raw.tsv" > "$D/summary_tables.txt"
    ;;
  linalg-unpinned)
    # 32 and 64 threads without taskset (run.sh PIN=0): pinned faer (rayon) spin-waits pathologically
    # with exactly t CPUs. Uses $RUN_DIR/run.sh when present (image 2026-11-04c predates PIN).
    RS=$L/run.sh; [ -f "$RUN_DIR/run.sh" ] && RS=$RUN_DIR/run.sh
    cp -r "$RUN_DIR/mats" ./mats
    sha256sum mats/*.bin >> "$D/node.txt"
    B=$L/target/release/linalg-bench PIN=0 THREADS="32 64" BACKENDS="$ALL" OUT=$D/raw_unpinned.tsv ENV=$D/env_unpinned.txt \
      bash "$RS" "$PWD/mats" $MKLDIR $L/target-v4/release/linalg-bench 3 2> "$D/progress.log"
    python3 $L/summarize.py "$D/raw_unpinned.tsv" > "$D/summary_tables_unpinned.txt"
    ;;
  linalg-tp)
    cp -r "$RUN_DIR/mats" ./mats
    sha256sum mats/*.bin >> "$D/node.txt"
    for np in 64 128; do
      BACKENDS="$ALL" BV4=$L/target-v4/release/linalg-bench OUT=$D/throughput$np.tsv \
        $L/throughput.sh "$PWD/mats" $MKLDIR $np 2>> "$D/progress.log"
    done
    python3 $L/summarize.py "$D/throughput64.tsv,$D/throughput128.tsv" evd,gen > "$D/throughput_tables.txt"
    ;;
  suite)
    mkdir -p tmp; export AXTB_SCRATCH=$PWD/tmp
    python3 /opt/axtb/bench/run_benchmarks.py --threads 1,8,32,64 --hess-threads 1,8,32,64 \
      --throughput 128x64,512x64,1024x128 --out "$D/results.json" 2>&1 | tee "$D/run.log"
    ;;
  *) echo "unknown kind $KIND"; exit 2;;
esac
echo "load after: $(cat /proc/loadavg)  $(date -u)" >> "$D/node.txt"
