# axtb benchmark image for whole cluster nodes (the recorded runs: AMD Threadripper PRO 9985WX).
#   xtb 6.7.1 (g-xTB 2.0.1 build) from the base image, as `xtb`
#   tblite 0.8.0 + dftd4 4.3.1 (PyPI), axtb (this repo: CLI + Python module)
#   bench/linalg harness: faer 0.22/0.24 (+ AVX-512 build), OpenBLAS (scipy-openblas32 wheel),
#   Intel MKL (mkl wheel), AMD AOCL 5.3 (BLIS + libFLAME built from AMD's GitHub sources)
# Build context: bench/cluster/make_context.sh (a snapshot of the repo with this file at its root).
# BASE_IMAGE: a Debian/Ubuntu image with Python 3 and pip, and the g-xTB 2.0.1 build of xtb
# (grimme-lab/g-xtb 2.0.1) installed as /opt/gxtb/bin/xtb, e.g.
#   docker build --build-arg BASE_IMAGE=<registry>/<g-xtb image> -t <registry>/axtb-bench:<tag> <context>
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates curl util-linux procps \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /opt/gxtb/bin/xtb /usr/local/bin/xtb

# AMD AOCL 5.3 (LP64, OpenMP): /opt/aocl-5.3/lib/{libflame.so,libblis-mt.so,libaoclutils.so}
COPY bench/linalg/build_aocl.sh /tmp/build_aocl.sh
RUN bash /tmp/build_aocl.sh /opt/aocl-5.3 /tmp/aocl-src 16 && rm -rf /tmp/aocl-src

# Same Python package versions as the laptop run (bench/results/results.json)
RUN pip install --no-cache-dir tblite==0.8.0 dftd4==4.3.1 maturin==1.16.0 \
        mkl==2026.2.0 scipy-openblas32==0.3.34.481.3

# Rust stable for axtb, the harness and faer's AVX-512 build
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable

COPY . /opt/axtb
WORKDIR /opt/axtb
RUN cargo build --release -p axtb-cli \
    && (cd crates/axtb-py && maturin build --release --compatibility linux -o /tmp/wheels) \
    && pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels \
    && (cd bench/linalg && cargo build --release \
        && cargo build --release --features faer-avx512 --target-dir target-v4) \
    && rm -rf target/release/deps target/release/build target/release/incremental \
              bench/linalg/target*/release/deps bench/linalg/target*/release/build \
    && rustc --version > /opt/axtb/RUST_VERSIONS

RUN python3 -c "import axtb, tblite, dftd4; print('axtb', axtb.__version__)" && xtb --version | grep version
