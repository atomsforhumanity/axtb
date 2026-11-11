#!/usr/bin/env bash
# axtb's checks, in two tiers.
#
#   scripts/check.sh fast                 no external programs: formatting, lints, docs, the Rust
#                                         tests, the LGPL-only build, the Python tests from a wheel
#                                         in a fresh venv, the privacy grep and the crates' copies
#                                         of the license files
#   scripts/check.sh programs [GATE ...]  the validation gates against xtb, tblite and the g-xTB
#                                         programs (all, or the named ones); a gate whose programs
#                                         are not set is reported as skipped
#   scripts/check.sh all                  both
#   scripts/check.sh list                 the gates of the programs tier: needs and limits
#
# The programs come from AXTB_XTB680, AXTB_GXTB2, AXTB_XTB671 and AXTB_GXTB11_DIR
# (tests/programs.py); xtb's atomic masses for the Hessian comparisons from the xtb source in
# $AXTB_REFERENCE/xtb-gxtb (default reference/, see reference/README.md).
#
# Other settings:
#   AXTB_CHECK_DIR        scratch directory: the venv, the wheel, the gates' outputs and logs
#                         (default target/check; the gates never write into the repository)
#   AXTB_CHECK_PYTHON     the Python of the venv: a version for uv (default 3.12) or, without uv,
#                         an interpreter (default python3); tblite 0.8.0 needs a Python with wheels
#   AXTB_CHECK_JOBS       parallel program runs of the gates that take --jobs (default 4)
#   AXTB_PRIVATE_PATTERN  an extra regular expression for the privacy grep (site host names)
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 2
CHECK_DIR=${AXTB_CHECK_DIR:-$ROOT/target/check}
JOBS=${AXTB_CHECK_JOBS:-4}
VENV=$CHECK_DIR/venv
PY=$VENV/bin/python
export AXTB_BIN=$ROOT/target/release/axtb
export AXTB_REFERENCE=${AXTB_REFERENCE:-$ROOT/reference}
PROGRAM_VARS=(AXTB_XTB680 AXTB_GXTB2 AXTB_XTB671 AXTB_GXTB11_DIR MEPD_DIR MEPD_PYTHON)
RESULTS=()
FAILED=0

say() { printf '\n== %s\n' "$*"; }

# step NAME COMMAND...: run, record PASS or FAIL
step() {
  local name=$1
  shift
  say "$name"
  if "$@"; then
    RESULTS+=("PASS  $name")
  else
    RESULTS+=("FAIL  $name")
    FAILED=1
  fi
}

# ------------------------------------------------------------------------------------- fast

privacy() {
  # home directories, private IPv4 ranges, the shared range of overlay networks (100.64/10),
  # tailnet names; plus $AXTB_PRIVATE_PATTERN
  local pat='/home/[A-Za-z0-9_]|/Users/[A-Za-z0-9_]|~/(dev|\.local|Downloads)/|192\.168\.[0-9]'
  pat+='|\b10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\b|\b172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}\b'
  pat+='|\b100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}\b|\.ts\.net\b'
  if [ -n "${AXTB_PRIVATE_PATTERN:-}" ]; then
    pat+="|$AXTB_PRIVATE_PATTERN"
  else
    echo "privacy: AXTB_PRIVATE_PATTERN is not set, so site host names are not checked"
  fi
  local hits
  hits=$(git ls-files -z | xargs -0 grep -nIE "$pat" 2>/dev/null)
  if [ -n "$hits" ]; then
    printf '%s\n' "$hits" | head -50
    echo "privacy: $(printf '%s\n' "$hits" | wc -l) lines with home paths, private addresses or host names"
    return 1
  fi
  echo "privacy: no home paths, private addresses or host names in $(git ls-files | wc -l) tracked files"
}

license_files() {
  # cargo packages only the files inside a crate's directory, and the wheel only those next to
  # pyproject.toml, so each crate keeps copies of the texts it needs; they must equal the top level
  local need=(
    "base axtb d4 solvation gfnff driver axtb-cli axtb-py|COPYING COPYING.LESSER NOTICE"
    "gxtb|COPYING NOTICE"
    "axtb-py|LICENSE"
  )
  local entry crates files c f bad=0
  for entry in "${need[@]}"; do
    IFS='|' read -r crates files <<< "$entry"
    for c in $crates; do
      for f in $files; do
        if ! cmp -s "$f" "crates/$c/$f"; then
          echo "crates/$c/$f is missing or differs from ./$f"
          bad=1
        fi
      done
    done
  done
  [ $bad = 0 ] && echo "license files: the crate copies match COPYING, COPYING.LESSER, LICENSE and NOTICE"
  return $bad
}

lgpl_build() {
  # the build without g-xTB (LGPL-3.0-or-later) compiles cleanly and does not contain axtb-gxtb
  cargo clippy -p axtb-cli -p axtb-py --no-default-features -- -D warnings || return 1
  if cargo tree -p axtb-cli -p axtb-py --no-default-features -e normal | grep -q axtb-gxtb; then
    echo "axtb-gxtb is in the --no-default-features dependency tree"
    return 1
  fi
}

make_venv() {
  rm -rf "$VENV"
  mkdir -p "$CHECK_DIR"
  if command -v uv > /dev/null; then
    uv venv -q -p "${AXTB_CHECK_PYTHON:-3.12}" "$VENV" || return 1
    PIP=(uv pip install -q --python "$PY")
    REINSTALL=--reinstall
  else
    "${AXTB_CHECK_PYTHON:-python3}" -m venv "$VENV" || return 1
    PIP=("$PY" -m pip install -q)
    REINSTALL=--force-reinstall
  fi
  "${PIP[@]}" maturin numpy pytest "tblite==0.8.0" "dftd4==4.3.1" qccompute
}

wheel() {
  rm -rf "$CHECK_DIR/wheels"
  # a wheel for this host (the release's manylinux_2_28 wheels need the manylinux container)
  "$VENV/bin/maturin" build -q --release --compatibility linux -m crates/axtb-py/Cargo.toml -i "$PY" \
    -o "$CHECK_DIR/wheels" || return 1
  "${PIP[@]}" "$REINSTALL" "$CHECK_DIR"/wheels/axtb-*.whl
}

python_tests() {
  # without the reference programs: the cases that need them skip
  local unset=()
  for v in "${PROGRAM_VARS[@]}"; do unset+=(-u "$v"); done
  env "${unset[@]}" "$PY" -m pytest -q -p no:cacheprovider crates/axtb-py/tests tests/reference_matrix
}

fast() {
  step "cargo fmt --check" cargo fmt --all --check
  step "clippy -D warnings" cargo clippy --workspace --all-targets -- -D warnings
  step "rustdoc -D warnings" env RUSTDOCFLAGS=-Dwarnings cargo doc --workspace --no-deps -q
  step "cargo test --release" cargo test --workspace --release -q
  step "LGPL-only build (--no-default-features)" lgpl_build
  step "release CLI" cargo build --release -q -p axtb-cli
  step "fresh venv ($CHECK_DIR/venv)" make_venv
  step "wheel (maturin build) installed in the venv" wheel
  step "Python tests (crates/axtb-py/tests, tests/reference_matrix)" python_tests
  step "privacy grep" privacy
  step "license files in the crates" license_files
}

# ---------------------------------------------------------------------------------- programs

# The gates: name | programs (all needed; "a/b": one of them is enough; "-": none) | command.
# Every command writes its results into $OUT (the gate's directory), never into the repository.
# The matrix runs start the different-SCF-solution diagnoses from the other xtb version's
# moments too (--xtb-other), as the recorded results did.
GATES=(
  "validate-tblite|-|tests/validate_tblite.py"
  "validate-xtb|AXTB_XTB680|tests/validate_xtb.py"
  "validate-xtb-alpb|AXTB_XTB680|tests/validate_xtb.py --alpb water"
  "validate-hessian|AXTB_GXTB2|tests/validate_hessian.py"
  "matrix-xtb-6.8.0|AXTB_XTB680,AXTB_GXTB2|tests/reference_matrix/run.py --xtb-other \$AXTB_GXTB2 --fresh --jobs \$JOBS --outdir \$OUT"
  "matrix-xtb-6.7.1|AXTB_GXTB2,AXTB_XTB680|tests/reference_matrix/run.py --xtb \$AXTB_GXTB2 --xtb-other \$AXTB_XTB680 --fresh --jobs \$JOBS --outdir \$OUT"
  "pytest-programs|AXTB_XTB680,AXTB_GXTB2|-m pytest -q -p no:cacheprovider tests/reference_matrix crates/axtb-py/tests/test_molden.py"
  "xtb680-changes|AXTB_XTB680|tests/xtb680_changes.py --out \$OUT/xtb680_changes.json"
  "xtb-hess-acc|AXTB_XTB680|tests/xtb_hess_acc.py --out \$OUT/xtb_gfn2_hess_acc.json"
  "program-differences|AXTB_XTB680|tests/program_differences.py --out \$OUT/program_diffs.json"
  "scf-hard|AXTB_GXTB2|tests/scf_hard_set.py --jobs \$JOBS --out \$OUT/scf_hard.json"
  "vibrational-xtb-6.8.0|AXTB_XTB680|tests/vibrational_check.py --out \$OUT/vibrational.json"
  "vibrational-xtb-6.7.1|AXTB_GXTB2|tests/vibrational_check.py --xtb \$AXTB_GXTB2 --out \$OUT/vibrational.json"
  "gfnff-xtb-6.8.0|AXTB_XTB680|gfnff"
  "gfnff-xtb-6.7.1|AXTB_GXTB2|gfnff --xtb \$AXTB_GXTB2"
  "gxtb-reproduction|-|tests/gxtb_reproduction.py --out \$OUT/gxtb_reproduction.json"
  "gxtb2-reproduction|-|tests/gxtb2_reproduction.py --out \$OUT/gxtb2_reproduction.json"
  "gxtb-etemp|-|tests/gxtb_etemp_reproduction.py --out \$OUT/gxtb_etemp.json"
  "gxtb-energy-differences|-|tests/gxtb_energy_differences.py --out \$OUT/gxtb_energy_differences.json"
  "gxtb-elements|AXTB_GXTB2/AXTB_GXTB11_DIR|tests/gxtb_elements.py --jobs \$JOBS --out \$OUT/gxtb_elements.json"
  "gxtb-parity|AXTB_GXTB2/AXTB_XTB680/AXTB_XTB671/AXTB_GXTB11_DIR|tests/gxtb_parity_features.py --only properties,restart --out \$OUT/gxtb_parity_features.json"
  "solvation-composite|AXTB_GXTB2|tests/solvation_composite.py --repeat 1 --out \$OUT/solvation_composite.json"
  "molden|AXTB_GXTB2/AXTB_GXTB11_DIR|tests/molden_check.py --out \$OUT/molden.json"
  "ts-example|-|bench/ts_example.py"
)

# the GFN-FF report without its scaling part (timings up to 9900 atoms); every part runs, and
# the gate fails if any part fails
gfnff() {
  local part rc=0
  for part in accuracy hessian topology; do
    "$PY" tests/gfnff_reproduction.py --part "$part" --jobs "$JOBS" --out "$OUT/gfnff_reproduction.json" "$@" || rc=$?
  done
  return $rc
}

have() {  # "A,B" all of them, "A/B" one of them, "-" nothing
  local need=$1 v
  [ "$need" = "-" ] && return 0
  if [[ $need == */* ]]; then
    for v in ${need//\// }; do "$PY" tests/programs.py --have "$v" && return 0; done
    return 1
  fi
  for v in ${need//,/ }; do "$PY" tests/programs.py --have "$v" || return 1; done
}

list_gates() {  # with the limits of each gate's script (LIMITS and KNOWN_DIFFERENCES, tests/gate.py)
  printf '%-26s %-52s %s\n' gate programs command
  local g name need cmd script
  for g in "${GATES[@]}"; do
    IFS='|' read -r name need cmd <<< "$g"
    printf '%-26s %-52s %s\n' "$name" "$need" "$cmd"
    script=${cmd%% *}
    [[ $cmd == gfnff* ]] && script=tests/gfnff_reproduction.py
    if [[ $script == *.py ]]; then
      python3 tests/gate.py "$script"
    else
      echo "    limits: the pytest assertions"
    fi
  done
}

build_examples() {
  cargo build --release -q -p axtb-cli \
    && cargo build --release -q -p axtb-gfnff --examples \
    && cargo build --release -q -p axtb --example conventions_ablation
}

programs_tier() {
  local only=("$@")
  if [ ! -x "$PY" ] || ! "$PY" -c "import axtb" 2> /dev/null; then
    step "fresh venv ($CHECK_DIR/venv)" make_venv
    step "wheel (maturin build) installed in the venv" wheel
  fi
  step "release CLI and examples" build_examples
  say "programs (tests/programs.py)"
  "$PY" tests/programs.py
  local g name need cmd log rc
  for g in "${GATES[@]}"; do
    IFS='|' read -r name need cmd <<< "$g"
    if [ ${#only[@]} -gt 0 ] && [[ " ${only[*]} " != *" $name "* ]]; then continue; fi
    OUT=$CHECK_DIR/programs/$name
    rm -rf "$OUT"
    mkdir -p "$OUT"
    log=$OUT/log.txt
    if ! have "$need"; then
      RESULTS+=("SKIP  $name (needs ${need//\// or })")
      continue
    fi
    say "$name: $cmd"
    local start=$SECONDS
    if [[ $cmd == gfnff* ]]; then
      eval "$cmd" > "$log" 2>&1
    else
      eval "\"\$PY\" $cmd" > "$log" 2>&1
    fi
    rc=$?
    tail -n 5 "$log"
    case $rc in
      0) RESULTS+=("PASS  $name ($((SECONDS - start)) s)") ;;
      77) RESULTS+=("SKIP  $name: $(grep -m1 '^SKIPPED' "$log")") ;;
      *)
        RESULTS+=("FAIL  $name (exit $rc, log $log)")
        FAILED=1
        ;;
    esac
  done
}

# ------------------------------------------------------------------------------------- main

tier=${1:-fast}
shift || true
case $tier in
  fast) fast ;;
  programs) programs_tier "$@" ;;
  all)
    fast
    programs_tier "$@"
    ;;
  list)
    list_gates
    exit 0
    ;;
  *)
    sed -n '2,12p' "$0"
    exit 2
    ;;
esac

say "summary"
printf '%s\n' "${RESULTS[@]}"
exit $FAILED
