// SPDX-License-Identifier: LGPL-3.0-or-later
//! Records in `python/axtb/_build.py` whether this build of the extension module assumes an
//! x86-64-v3 CPU (AVX2, FMA, ...; `.cargo/config.toml`), so that `axtb/__init__.py` checks the
//! CPU before it loads the module only when the build needs it: a build with
//! `RUSTFLAGS="-C target-cpu=x86-64"` imports on any x86-64 CPU.

fn main() {
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let features = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let v3 = arch == "x86_64" && features.split(',').any(|f| f == "avx2" || f == "fma");
    let text = format!(
        "# Written by crates/axtb-py/build.rs (not tracked): whether the extension module was built for\n\
         # x86-64-v3 CPUs (AVX2, FMA, ...), which axtb/__init__.py then checks for before loading it.\n\
         X86_64_V3 = {}\n",
        if v3 { "True" } else { "False" }
    );
    let dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    // python/ next to this file in the repository; in maturin's source distribution the python
    // source sits at the top, next to the workspace's Cargo.toml
    let package = ["python/axtb", "../../python/axtb"]
        .iter()
        .map(|p| dir.join(p))
        .find(|p| p.is_dir())
        .expect("python/axtb next to crates/axtb-py/Cargo.toml or two directories up");
    let path = package.join("_build.py");
    // rewritten only when it changes, so that an unchanged build leaves the file alone
    if std::fs::read_to_string(&path).ok().as_deref() != Some(text.as_str()) {
        std::fs::write(&path, text).expect("write python/axtb/_build.py");
    }
    // rerun for a new target CPU (RUSTFLAGS) or a deleted file, not for every edit in the crate
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", path.display());
    println!("cargo:rerun-if-env-changed=RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
}
