# probe only: make a vendored copy of gemm-common / private-gemm-x86 read cpu0's cache files
# instead of every CPU's (128 CPUs x 4 caches x 11 files = 5770 sysfs opens per process)
import sys
p = sys.argv[1]; s = open(p).read()
old = 'for cpu_x in fs::read_dir("/sys/devices/system/cpu")? {'
new = 'for cpu_x in fs::read_dir("/sys/devices/system/cpu")?.filter(|e| e.as_ref().map_or(true, |e| e.file_name() == "cpu0")) {'
assert s.count(old) == 1, p
open(p, "w").write(s.replace(old, new)); print("patched", p)
