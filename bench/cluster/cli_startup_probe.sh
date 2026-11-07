#!/usr/bin/env bash
# Why the axtb CLI was slow in throughput on a 128-thread node: builds axtb-cli variants in the image
# (A = eeq.rs sequential LU; AC = A + gemm-common/private-gemm-x86 reading only cpu0 cache info via patch_cpu0.py),
# then: energies, syscalls/CPU of one process, 64x8 concurrent runs, and cli_throughput.py.
# Run in a container of the bench image on an idle node (126 CPUs, 200 GiB): env SM=<dir> bash <dir>/cli_startup_probe.sh
# with SM = a run dir holding eeq.rs (the fixed crates/axtb/src/eeq.rs), patch_cpu0.py, cli_throughput.py.
set -e
SM=${SM:?set SM}; OUT=$SM/out; mkdir -p $OUT
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
export AXTB_SCRATCH=/tmp/w/tmp; mkdir -p $AXTB_SCRATCH
(apt-get update -qq && apt-get install -y -qq strace) >/dev/null 2>&1 || echo "no strace"
W=/tmp/w; cp -r /opt/axtb $W/ax; cd $W/ax
cp $SM/eeq.rs crates/axtb/src/eeq.rs
cargo build --release --offline -p axtb-cli --target-dir $W/tA -j 64 2>&1 | tail -1
for c in gemm-common-0.18.2 private-gemm-x86-0.1.22; do cp -r /opt/cargo/registry/src/*/$c $W/$c; python3 $SM/patch_cpu0.py $W/$c/src/cache.rs; done
printf '\n[patch.crates-io]\ngemm-common = { path = "%s" }\nprivate-gemm-x86 = { path = "%s" }\n' $W/gemm-common-0.18.2 $W/private-gemm-x86-0.1.22 >> Cargo.toml
cargo build --release --offline -p axtb-cli --target-dir $W/tAC -j 64 2>&1 | grep -E "Adding|Finished|warning: Patch"
declare -A BIN=([orig]=/opt/axtb/target/release/axtb [A]=$W/tA/release/axtb [AC]=$W/tAC/release/axtb)
X=/opt/axtb/tests/structures
echo "== energies (Eh) and SCF iterations"
for v in orig A AC; do for f in caffeine taxol large/water100; do ${BIN[$v]} $X/$f.xyz --grad | python3 -c "import json,sys; d=json.load(sys.stdin); print('$v', '$f', repr(d['energy']), d['iterations'])"; done; done
echo "== one process"
for v in orig A AC; do
  strace -f -c -o /tmp/st ${BIN[$v]} $X/caffeine.xyz --grad >/dev/null
  echo "$v openat=$(awk '$NF=="openat"{print $4}' /tmp/st) clone3=$(awk '$NF=="clone3"{print $4}' /tmp/st)"
  python3 - ${BIN[$v]} $X <<'PY'
import subprocess, resource, time, sys
b, X = sys.argv[1], sys.argv[2]
for f in ["water", "caffeine", "taxol"]:
    best = None
    for _ in range(5):
        r0 = resource.getrusage(resource.RUSAGE_CHILDREN); t = time.perf_counter()
        subprocess.run([b, f"{X}/{f}.xyz", "--grad"], capture_output=True, check=True)
        r1 = resource.getrusage(resource.RUSAGE_CHILDREN); w = time.perf_counter() - t
        cur = (w, r1.ru_utime - r0.ru_utime, r1.ru_stime - r0.ru_stime)
        best = cur if best is None or cur[0] < best[0] else best
    print(f"   {f}: wall {best[0]*1e3:.1f} ms  user {best[1]*1e3:.0f} ms  sys {best[2]*1e3:.0f} ms (best of 5)")
PY
done
echo "== 64 concurrent workers x 8 runs (bash loop, caffeine)"
for v in orig A AC; do s=$EPOCHREALTIME; for w in $(seq 64); do (for i in $(seq 8); do ${BIN[$v]} $X/caffeine.xyz --grad >/dev/null; done) & done; wait; e=$EPOCHREALTIME; echo "$v: $(awk -v a=$s -v b=$e 'BEGIN{printf "%.2f", b-a}') s"; done
echo "== run_benchmarks throughput rows, rerun (3 repeats)"
python3 $SM/cli_throughput.py --root /opt/axtb --specs 128x64,512x64,1024x128 --bin orig=${BIN[orig]} --bin eeq-fix=${BIN[A]} \
  --bin eeq-fix+cache-scan-cpu0=${BIN[AC]} --out $OUT/cli_throughput.json > $OUT/cli_throughput.log 2>&1
cat $OUT/cli_throughput.log
echo "node load at end: $(cat /proc/loadavg)"
