#!/usr/bin/env bash
# One task of the final timings (bench/final_timings.py) in a cluster pod. ./task.json holds the
# task; the harness runs from the run dir when it holds a copy ($RUN_DIR/final_timings.py, so a
# fix needs no image rebuild), else from the image. Node-identifying information (node.txt)
# stays in the raw output, not in result.json.
set -euo pipefail
D=$PWD/out
export AXTB_ROOT=/opt/axtb AXTB_BIN=/opt/axtb/target/release/axtb AXTB_REV_FILE=/opt/axtb/AXTB_REV \
       AXTB_XTB680=/opt/xtb-6.8.0/bin/xtb AXTB_GXTB2=/opt/gxtb/bin/xtb AXTB_GXTB11_DIR=/opt/gxtb-1.1.0 PYTHON=python3
[ -x /opt/axtb-base/target/release/axtb ] && export AXTB_BASE_BIN=/opt/axtb-base/target/release/axtb
H=/opt/axtb/bench/final_timings.py; [ -f "$RUN_DIR/final_timings.py" ] && H=$RUN_DIR/final_timings.py
{
  date -u; echo "node: ${NODE_NAME:-$(hostname)}  NTHREADS=${NTHREADS:-}  harness: $H"
  cat /opt/axtb/AXTB_REV /opt/axtb-base/AXTB_REV; sha256sum "$H"
  lscpu | grep -E "Model name|Socket|Core|Thread"
  echo "cpu0 SMT siblings: $(cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list)"
  echo "cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)"
  echo "load before: $(cat /proc/loadavg)"
} > "$D/node.txt"
mkdir -p tmp
python3 "$H" --task task.json --scratch "$PWD/tmp" --out "$D/result.json" 2>&1 | tee "$D/run.log"
echo "load after: $(cat /proc/loadavg)  $(date -u)" >> "$D/node.txt"
