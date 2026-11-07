#!/usr/bin/env bash
# Hand-run spot checks of bench/final_timings.py: each binary run directly with bash's `time`,
# alternating, 3 times per cell, on the inputs of spot_check_inputs.py (the harness's coord files
# and JSON requests). usage: spot_check.sh INPUT_DIR [XTB GXTB AXTB]; the programs default to
# $AXTB_XTB680 and $AXTB_GXTB2 (tests/programs.py), the CLI to $AXTB_BIN or target/release/axtb.
set -u
D=$1
XTB=${2:-${AXTB_XTB680:?set AXTB_XTB680 to xtb 6.8.0}}
GX=${3:-${AXTB_GXTB2:?set AXTB_GXTB2 to the g-xTB 2.0.1 build of xtb}}
AX=$(readlink -f "${4:-${AXTB_BIN:-target/release/axtb}}")
W=$(mktemp -d); cd "$W"
TIMEFORMAT='%R'
sha256sum "$XTB" "$GX" "$AX" | awk '{print substr($1,1,16), $2}'
prog() {  # binary, args, threads, coord file
  rm -rf p && mkdir p && cp "$D/$4" p/coord && cd p
  local share; share=$(dirname "$(readlink -f "$1")")/../share/xtb
  { time env OMP_NUM_THREADS=$3 MKL_NUM_THREADS=$3 OMP_STACKSIZE=1G HOME=. XTBPATH="$share" "$1" coord $2 > out 2>&1; } 2>&1
  grep -q "normal termination" out || echo "FAILED $1"
  cd ..
}
ax() {  # request file
  { time env -u OMP_NUM_THREADS -u AXTB_LAPACK -u AXTB_LAPACK_BLAS "$AX" --json < "$D/$1" > ax.json; } 2>&1
  python3 -c "import json; d=json.load(open('ax.json')); print('   axtb E', d['energy'], 'it', d['iterations'])"
}
for r in 1 2 3; do
  echo "gfn2 (H2O)100 1 thread round $r: xtb 6.8.0 $(prog "$XTB" "--gfn 2 --grad" 1 w100.coord) s, axtb $(ax gfn2-w100-t1.json | tr '\n' ' ')"
  echo "gfn2 (H2O)200 8 threads round $r: xtb 6.8.0 $(prog "$XTB" "--gfn 2 --grad" 8 w200.coord) s, axtb $(ax gfn2-w200-t8.json | tr '\n' ' ')"
  echo "gxtb2 taxol 4 threads round $r: program $(prog "$GX" "--gxtb --grad" 4 taxol.coord) s, axtb $(ax gxtb2-taxol-t4.json | tr '\n' ' ')"
done
rm -rf "$W"
