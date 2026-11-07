//! Dense linear-algebra backend benchmark for axtb.
//!
//! One process = one backend at one thread count. Backends:
//!   faer22 / faer24       pure Rust (faer 0.22.6 / 0.24.4), `Par` + an explicit rayon pool
//!   lapack:<path-to-.so>  any LP64 BLAS/LAPACK loaded with dlopen (OpenBLAS, MKL's mkl_rt, ...)
//!                         AOCL: lapack:<aocl>/lib/libflame.so --blas <aocl>/lib/libblis-mt.so
//!
//! Ops (column-major, f64):
//!   evd      full symmetric eigendecomposition, all vectors (dsyevd)
//!   gen      F C = S C e with the Cholesky factor of S reused:
//!              LAPACK: dsygst + dsyevd + dtrsm   (what dsygvd does after dpotrf)
//!              faer:   2x trsm + evd + trsm      (axtb's GenEigSolver::solve)
//!   gen2t    LAPACK only: 2x dtrsm + dsyevd + dtrsm (same algorithm as the faer path)
//!   evr40    dsyevr, lowest 40% of eigenpairs (RANGE='I')
//!   mr40     dsytrd + dstemr(RANGE='I', lowest 40%) + dormtr on those columns
//!   genr40   dsygst + dsyevr(lowest 40%) + dtrsm on those columns
//!   gemm     C = A B, n x n x n
//!
//! Output: one TSV line per (n, op): backend threads n op median_ms min_ms reps cpu/wall residual checksum

use std::ffi::{c_char, c_int, CStr};
use std::time::Instant;

// ---------------------------------------------------------------- matrices

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        // xorshift64*, uniform in [-1, 1)
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545F4914F6CDD1D);
        (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

/// Symmetric "Fock-like" matrix: random symmetric with a spread diagonal.
fn make_f(n: usize) -> Vec<f64> {
    let mut r = Rng(0x9E3779B97F4A7C15 ^ n as u64);
    let mut a = vec![0.0; n * n];
    for j in 0..n {
        for i in j..n {
            let v = if i == j { -1.0 + 2.0 * (i as f64) / n as f64 + 0.1 * r.next() } else { 0.3 * r.next() / (1.0 + (i - j) as f64).sqrt() };
            a[i + j * n] = v;
            a[j + i * n] = v;
        }
    }
    a
}

/// SPD "overlap-like" matrix: unit diagonal, decaying off-diagonal, diagonally dominant.
fn make_s(n: usize) -> Vec<f64> {
    let mut r = Rng(0xD1B54A32D192ED03 ^ n as u64);
    let mut a = vec![0.0; n * n];
    for j in 0..n {
        for i in j..n {
            let v = if i == j { 1.0 } else { 0.5 * r.next().abs() * (-0.05 * (i - j) as f64).exp() / 10.0 };
            a[i + j * n] = v;
            a[j + i * n] = v;
        }
    }
    // make strictly diagonally dominant -> SPD
    for i in 0..n {
        let s: f64 = (0..n).filter(|&k| k != i).map(|k| a[i + k * n].abs()).sum();
        a[i + i * n] = 1.0 + s;
    }
    a
}

/// Leading n x n principal submatrix of a stored N x N column-major f64 matrix (real GFN2 H0 / S).
fn load_sub(dir: &str, name: &str, n: usize) -> Vec<f64> {
    let big: usize = std::fs::read_to_string(format!("{dir}/dim.txt")).unwrap().trim().parse().unwrap();
    assert!(n <= big, "requested n={n} > stored {big}");
    let bytes = std::fs::read(format!("{dir}/{name}")).unwrap();
    let mut out = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            let o = 8 * (i + j * big);
            out[i + j * n] = f64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
        }
    }
    out
}

fn cpu_time() -> f64 {
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut ru);
        let t = |tv: libc::timeval| tv.tv_sec as f64 + tv.tv_usec as f64 * 1e-6;
        t(ru.ru_utime) + t(ru.ru_stime)
    }
}

/// max_ij |A X - B X diag(w)| / max|A| for the first `k` columns of X (B = I if None).
fn residual(n: usize, a: &[f64], b: Option<&[f64]>, x: &[f64], w: &[f64], k: usize) -> f64 {
    use faer24::{Accum, MatRef, Par};
    let am = MatRef::from_column_major_slice(a, n, n);
    let xm = MatRef::from_column_major_slice(&x[..n * k], n, k);
    let mut ax = faer24::Mat::<f64>::zeros(n, k);
    faer24::linalg::matmul::matmul(ax.as_mut(), Accum::Replace, am, xm, 1.0, Par::Seq);
    let mut bx = faer24::Mat::<f64>::zeros(n, k);
    match b {
        Some(b) => faer24::linalg::matmul::matmul(bx.as_mut(), Accum::Replace, MatRef::from_column_major_slice(b, n, n), xm, 1.0, Par::Seq),
        None => bx.copy_from(xm),
    }
    let amax = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let mut r = 0.0f64;
    for j in 0..k {
        for i in 0..n {
            r = r.max((ax[(i, j)] - bx[(i, j)] * w[j]).abs());
        }
    }
    r / amax
}

// ---------------------------------------------------------------- backend trait

trait Backend {
    fn prepare(&mut self, n: usize, f: &[f64], s: &[f64]);
    fn has(&self, op: &str) -> bool;
    fn run(&mut self, op: &str, n: usize, f: &[f64]);
    /// (eigenvalues, eigenvectors column-major, number of valid columns) of the last run
    fn result(&mut self, op: &str, n: usize) -> Option<(Vec<f64>, Vec<f64>, usize)>;
    fn describe(&self) -> String;
}

fn k40(n: usize) -> usize {
    (2 * n).div_ceil(5)
}

// ---------------------------------------------------------------- faer (both versions)

macro_rules! faer_backend {
    ($modname:ident, $faer:ident, $ver:expr, $bsfn:ident) => {
        mod $modname {
            use $faer as faer;
            use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
            use faer::linalg::cholesky::llt::factor::{cholesky_in_place, cholesky_in_place_scratch, LltRegularization};
            use faer::linalg::evd::{self_adjoint_evd, self_adjoint_evd_scratch, ComputeEigenvectors};
            use faer::linalg::triangular_solve::{solve_lower_triangular_in_place, solve_upper_triangular_in_place};
            use faer::{Accum, Mat, MatRef, Par};

            pub struct Faer {
                pub par: Par,
                pub threads: usize,
                l: Mat<f64>,
                linv: Mat<f64>,
                work: Mat<f64>,
                c: Mat<f64>,
                b: Mat<f64>,
                w: Vec<f64>,
                mem: MemBuffer,
            }

            impl Faer {
                pub fn new(threads: usize) -> Self {
                    let par = if threads > 1 { Par::rayon(threads) } else { Par::Seq };
                    Faer { par, threads, l: Mat::zeros(0, 0), linv: Mat::zeros(0, 0), work: Mat::zeros(0, 0), c: Mat::zeros(0, 0), b: Mat::zeros(0, 0), w: vec![], mem: MemBuffer::new(StackReq::EMPTY) }
                }
            }

            impl super::Backend for Faer {
                fn describe(&self) -> String {
                    format!("faer {} par={:?} rayon_threads_in_pool={}", $ver, self.par, rayon::current_num_threads())
                }
                fn has(&self, op: &str) -> bool {
                    matches!(op, "evd" | "gen" | "genmm" | "gemm" | "trd")
                }
                fn prepare(&mut self, n: usize, _f: &[f64], s: &[f64]) {
                    let par = self.par;
                    let req = StackReq::any_of(&[
                        cholesky_in_place_scratch::<f64>(n, par, Default::default()),
                        self_adjoint_evd_scratch::<f64>(n, ComputeEigenvectors::Yes, par, Default::default()),
                    ]);
                    self.mem = MemBuffer::new(req);
                    let mut l = MatRef::from_column_major_slice(s, n, n).to_owned();
                    cholesky_in_place(l.as_mut(), LltRegularization::default(), par, MemStack::new(&mut self.mem), Default::default()).expect("S not SPD");
                    for j in 0..n {
                        for i in 0..j {
                            l[(i, j)] = 0.0;
                        }
                    }
                    // explicit L^-1 once per geometry (for "genmm")
                    let mut linv = Mat::<f64>::identity(n, n);
                    solve_lower_triangular_in_place(l.as_ref(), linv.as_mut(), par);
                    self.linv = linv;
                    self.l = l;
                    self.work = Mat::zeros(n, n);
                    self.c = Mat::zeros(n, n);
                    // gemm second operand: a different matrix
                    self.b = Mat::from_fn(n, n, |i, j| ((i * 7 + j * 13) % 17) as f64 / 17.0 - 0.5);
                    self.w = vec![0.0; n];
                }
                fn run(&mut self, op: &str, n: usize, f: &[f64]) {
                    let par = self.par;
                    let fm = MatRef::from_column_major_slice(f, n, n);
                    match op {
                        "evd" => {
                            let mut s = faer::diag::Diag::<f64>::zeros(n);
                            self_adjoint_evd(fm, s.as_mut(), Some(self.c.as_mut()), par, MemStack::new(&mut self.mem), Default::default()).expect("evd");
                            for i in 0..n {
                                self.w[i] = s[i];
                            }
                        }
                        "gen" => {
                            // identical to axtb::linalg::GenEigSolver::solve
                            self.work.copy_from(fm);
                            solve_lower_triangular_in_place(self.l.as_ref(), self.work.as_mut(), par);
                            self.c.copy_from(self.work.transpose());
                            solve_lower_triangular_in_place(self.l.as_ref(), self.c.as_mut(), par);
                            let mut s = faer::diag::Diag::<f64>::zeros(n);
                            self_adjoint_evd(self.c.as_ref(), s.as_mut(), Some(self.work.as_mut()), par, MemStack::new(&mut self.mem), Default::default()).expect("evd");
                            for i in 0..n {
                                self.w[i] = s[i];
                            }
                            self.c.copy_from(&self.work);
                            solve_upper_triangular_in_place(self.l.transpose(), self.c.as_mut(), par);
                        }
                        "trd" => {
                            // diagnostics: tridiagonalization alone (same as LAPACK "trd" = dsytrd)
                            use faer::linalg::evd::tridiag::{tridiag_in_place, tridiag_in_place_scratch};
                            let bs = faer::linalg::qr::no_pivoting::factor::$bsfn::<f64>(n, n);
                            let mut hh = Mat::<f64>::zeros(bs, n - 1);
                            let mut mem = MemBuffer::new(tridiag_in_place_scratch::<f64>(n, par, Default::default()));
                            self.work.copy_from(fm);
                            tridiag_in_place(self.work.as_mut(), hh.as_mut(), par, MemStack::new(&mut mem), Default::default());
                        }
                        "genmm" => {
                            // work = L^-1 F ; c = work L^-T ; evd ; c = L^-T U  (3 GEMMs instead of 3 TRSMs)
                            faer::linalg::matmul::matmul(self.work.as_mut(), Accum::Replace, self.linv.as_ref(), fm, 1.0, par);
                            faer::linalg::matmul::matmul(self.c.as_mut(), Accum::Replace, self.work.as_ref(), self.linv.transpose(), 1.0, par);
                            let mut s = faer::diag::Diag::<f64>::zeros(n);
                            self_adjoint_evd(self.c.as_ref(), s.as_mut(), Some(self.work.as_mut()), par, MemStack::new(&mut self.mem), Default::default()).expect("evd");
                            for i in 0..n {
                                self.w[i] = s[i];
                            }
                            faer::linalg::matmul::matmul(self.c.as_mut(), Accum::Replace, self.linv.transpose(), self.work.as_ref(), 1.0, par);
                        }
                        "gemm" => {
                            faer::linalg::matmul::matmul(self.c.as_mut(), Accum::Replace, fm, self.b.as_ref(), 1.0, par);
                        }
                        _ => unreachable!(),
                    }
                }
                fn result(&mut self, op: &str, n: usize) -> Option<(Vec<f64>, Vec<f64>, usize)> {
                    match op {
                        "evd" | "gen" | "genmm" => Some((self.w.clone(), (0..n * n).map(|k| self.c[(k % n, k / n)]).collect(), n)),
                        _ => None,
                    }
                }
            }
        }
    };
}

faer_backend!(faer22_backend, faer22, "0.22.6", recommended_blocksize);
faer_backend!(faer24_backend, faer24, "0.24.4", recommended_block_size);

// ---------------------------------------------------------------- nalgebra (pure Rust, single-threaded)

struct Nalgebra {
    w: Vec<f64>,
    v: Vec<f64>,
}

impl Backend for Nalgebra {
    fn describe(&self) -> String {
        "nalgebra 0.33 SymmetricEigen (sequential)".into()
    }
    fn has(&self, op: &str) -> bool {
        op == "evd"
    }
    fn prepare(&mut self, _n: usize, _f: &[f64], _s: &[f64]) {}
    fn run(&mut self, _op: &str, n: usize, f: &[f64]) {
        let m = nalgebra::DMatrix::from_column_slice(n, n, f);
        let e = nalgebra::linalg::SymmetricEigen::new(m);
        // nalgebra returns unsorted eigenvalues; sorting is not part of the timing-relevant work
        self.w = e.eigenvalues.as_slice().to_vec();
        self.v = e.eigenvectors.as_slice().to_vec();
    }
    fn result(&mut self, _op: &str, n: usize) -> Option<(Vec<f64>, Vec<f64>, usize)> {
        Some((self.w.clone(), self.v.clone(), n))
    }
}

// ---------------------------------------------------------------- LAPACK via dlopen

type FInt = c_int;
type L = usize; // hidden Fortran character-length argument

#[allow(non_camel_case_types)]
mod sig {
    use super::*;
    pub type dsyevd = unsafe extern "C" fn(*const c_char, *const c_char, *const FInt, *mut f64, *const FInt, *mut f64, *mut f64, *const FInt, *mut FInt, *const FInt, *mut FInt, L, L);
    pub type dsyevr = unsafe extern "C" fn(
        *const c_char, *const c_char, *const c_char, *const FInt, *mut f64, *const FInt, *const f64, *const f64, *const FInt, *const FInt, *const f64, *mut FInt, *mut f64, *mut f64, *const FInt, *mut FInt, *mut f64,
        *const FInt, *mut FInt, *const FInt, *mut FInt, L, L, L,
    );
    pub type dsygst = unsafe extern "C" fn(*const FInt, *const c_char, *const FInt, *mut f64, *const FInt, *const f64, *const FInt, *mut FInt, L);
    pub type dpotrf = unsafe extern "C" fn(*const c_char, *const FInt, *mut f64, *const FInt, *mut FInt, L);
    pub type dtrsm = unsafe extern "C" fn(*const c_char, *const c_char, *const c_char, *const c_char, *const FInt, *const FInt, *const f64, *const f64, *const FInt, *mut f64, *const FInt, L, L, L, L);
    pub type dgemm = unsafe extern "C" fn(*const c_char, *const c_char, *const FInt, *const FInt, *const FInt, *const f64, *const f64, *const FInt, *const f64, *const FInt, *const f64, *mut f64, *const FInt, L, L);
    pub type dsytrd = unsafe extern "C" fn(*const c_char, *const FInt, *mut f64, *const FInt, *mut f64, *mut f64, *mut f64, *mut f64, *const FInt, *mut FInt, L);
    pub type dstemr = unsafe extern "C" fn(
        *const c_char, *const c_char, *const FInt, *mut f64, *mut f64, *const f64, *const f64, *const FInt, *const FInt, *mut FInt, *mut f64, *mut f64, *const FInt, *const FInt, *mut FInt, *mut FInt, *mut f64, *const FInt,
        *mut FInt, *const FInt, *mut FInt, L, L,
    );
    pub type dormtr = unsafe extern "C" fn(*const c_char, *const c_char, *const c_char, *const FInt, *const FInt, *const f64, *const FInt, *const f64, *mut f64, *const FInt, *mut f64, *const FInt, *mut FInt, L, L, L);
}

struct Lapack {
    _lib: libloading::Library,
    _blas: Option<libloading::Library>,
    name: String,
    info: String,
    dsyevd: sig::dsyevd,
    dsyevr: sig::dsyevr,
    dsygst: sig::dsygst,
    dpotrf: sig::dpotrf,
    dtrsm: sig::dtrsm,
    dgemm: sig::dgemm,
    dsytrd: sig::dsytrd,
    dstemr: sig::dstemr,
    dormtr: sig::dormtr,
    // buffers
    a: Vec<f64>,
    lfac: Vec<f64>,
    b: Vec<f64>,
    z: Vec<f64>,
    w: Vec<f64>,
    d: Vec<f64>,
    e: Vec<f64>,
    tau: Vec<f64>,
    work: Vec<f64>,
    iwork: Vec<FInt>,
    isuppz: Vec<FInt>,
    last_k: usize,
}

const C: fn(u8) -> c_char = |b| b as c_char;

impl Lapack {
    /// `path`: the LAPACK library; `blas`: optional separate BLAS library (e.g. AOCL-BLIS next to
    /// AOCL-libFLAME), loaded first with RTLD_GLOBAL so the LAPACK library resolves against it;
    /// `prefix`: symbol prefix (e.g. "scipy_" for the scipy-openblas32 wheel).
    unsafe fn load(path: &str, threads: usize, prefix: &str, blas: Option<&str>) -> Self {
        let blas_lib: Option<libloading::Library> = blas.map(|p| {
            libloading::os::unix::Library::open(Some(p), libc::RTLD_NOW | libc::RTLD_GLOBAL).unwrap_or_else(|e| panic!("dlopen {p}: {e}")).into()
        });
        let lib = libloading::Library::new(path).unwrap_or_else(|e| panic!("dlopen {path}: {e}"));
        let raw = |name: &str| -> Option<*const ()> {
            let full = format!("{prefix}{name}\0");
            for l in std::iter::once(&lib).chain(blas_lib.iter()) {
                if let Ok(s) = l.get::<*const ()>(full.as_bytes()) {
                    return Some(*s);
                }
            }
            None
        };
        macro_rules! sym {
            ($t:ty, $n:expr) => {
                std::mem::transmute::<*const (), $t>(raw($n).unwrap_or_else(|| panic!("missing symbol {prefix}{}", $n)))
            };
        }
        macro_rules! opt {
            ($t:ty, $n:expr) => {
                raw($n).map(|p| std::mem::transmute::<*const (), $t>(p))
            };
        }
        let mut info = String::new();
        let name;
        // --- MKL: pick the threading layer before any computational call
        if let Some(set_layer) = opt!(unsafe extern "C" fn(c_int) -> c_int, "MKL_Set_Threading_Layer") {
            name = "mkl".to_string();
            let set_iface = sym!(unsafe extern "C" fn(c_int) -> c_int, "MKL_Set_Interface_Layer");
            set_iface(0); // LP64
            // 0 = INTEL (libiomp5), 1 = SEQUENTIAL
            let layer = if threads == 1 { 1 } else { 0 };
            set_layer(layer);
            let set_n = sym!(unsafe extern "C" fn(c_int), "MKL_Set_Num_Threads");
            set_n(threads as c_int);
            let get_n = sym!(unsafe extern "C" fn() -> c_int, "MKL_Get_Max_Threads");
            let ver = sym!(unsafe extern "C" fn(*mut c_char, c_int), "MKL_Get_Version_String");
            let mut buf = vec![0 as c_char; 256];
            ver(buf.as_mut_ptr(), 255);
            info = format!(
                "{} | layer={} | MKL_Get_Max_Threads={}",
                CStr::from_ptr(buf.as_ptr()).to_string_lossy(),
                if layer == 1 { "SEQUENTIAL" } else { "INTEL(libiomp5)" },
                get_n()
            );
        } else if let Some(set_n) = opt!(unsafe extern "C" fn(c_int), "openblas_set_num_threads") {
            name = "openblas".to_string();
            set_n(threads as c_int);
            let get_n = sym!(unsafe extern "C" fn() -> c_int, "openblas_get_num_threads");
            let cfg = sym!(unsafe extern "C" fn() -> *const c_char, "openblas_get_config");
            let core = sym!(unsafe extern "C" fn() -> *const c_char, "openblas_get_corename");
            info = format!(
                "{} | core={} | openblas_get_num_threads={}",
                CStr::from_ptr(cfg()).to_string_lossy().trim(),
                CStr::from_ptr(core()).to_string_lossy(),
                get_n()
            );
        } else if let Some(set_n) = opt!(unsafe extern "C" fn(c_int), "bli_thread_set_num_threads") {
            set_n(threads as c_int);
            if let Some(get_n) = opt!(unsafe extern "C" fn() -> c_int, "bli_thread_get_num_threads") {
                info = format!("bli_thread_get_num_threads={}", get_n());
            }
            if let Some(v) = opt!(unsafe extern "C" fn() -> *const c_char, "bli_info_get_version_str") {
                info = format!("BLIS {} | {info}", CStr::from_ptr(v()).to_string_lossy());
            }
            // code path chosen by BLIS' runtime dispatch (amdzen builds: zen..zen5, also on Intel)
            if let (Some(q), Some(s)) = (opt!(unsafe extern "C" fn() -> c_int, "bli_arch_query_id"), opt!(unsafe extern "C" fn(c_int) -> *const c_char, "bli_arch_string")) {
                info = format!("{info} | blis_arch={}", CStr::from_ptr(s(q())).to_string_lossy());
            }
            // AOCL-LAPACK (libFLAME) has its own OpenMP regions (thread-local count): set it too,
            // and the OpenMP default, so nothing falls back to "all CPUs"
            if let Some(v) = opt!(unsafe extern "C" fn() -> *const c_char, "FLA_Get_AOCL_Version") {
                name = "aocl".to_string();
                info = format!("{} | {info}", CStr::from_ptr(v()).to_string_lossy().trim());
            } else {
                name = "blis".to_string();
            }
            if let Some(f) = opt!(unsafe extern "C" fn(c_int), "fla_thread_set_num_threads") {
                f(threads as c_int);
                if let Some(g) = opt!(unsafe extern "C" fn() -> c_int, "fla_thread_get_num_threads") {
                    info = format!("{info} | fla_thread_get_num_threads={}", g());
                }
            }
            if let Some(f) = opt!(unsafe extern "C" fn(c_int), "omp_set_num_threads") {
                f(threads as c_int);
            }
        } else {
            name = "lapack".to_string();
        }
        Lapack {
            dsyevd: sym!(sig::dsyevd, "dsyevd_"),
            dsyevr: sym!(sig::dsyevr, "dsyevr_"),
            dsygst: sym!(sig::dsygst, "dsygst_"),
            dpotrf: sym!(sig::dpotrf, "dpotrf_"),
            dtrsm: sym!(sig::dtrsm, "dtrsm_"),
            dgemm: sym!(sig::dgemm, "dgemm_"),
            dsytrd: sym!(sig::dsytrd, "dsytrd_"),
            dstemr: sym!(sig::dstemr, "dstemr_"),
            dormtr: sym!(sig::dormtr, "dormtr_"),
            _lib: lib,
            _blas: blas_lib,
            name,
            info,
            a: vec![],
            lfac: vec![],
            b: vec![],
            z: vec![],
            w: vec![],
            d: vec![],
            e: vec![],
            tau: vec![],
            work: vec![],
            iwork: vec![],
            isuppz: vec![],
            last_k: 0,
        }
    }

    fn syevd(&mut self, n: usize) {
        let ni = n as FInt;
        let lwork = (self.work.len()) as FInt;
        let liwork = self.iwork.len() as FInt;
        let mut info = 0;
        unsafe { (self.dsyevd)(&C(b'V'), &C(b'L'), &ni, self.a.as_mut_ptr(), &ni, self.w.as_mut_ptr(), self.work.as_mut_ptr(), &lwork, self.iwork.as_mut_ptr(), &liwork, &mut info, 1, 1) };
        assert_eq!(info, 0, "dsyevd info");
    }
    fn syevr_k(&mut self, n: usize, k: usize) {
        let ni = n as FInt;
        let (il, iu) = (1 as FInt, k as FInt);
        let (vl, vu, abstol) = (0.0, 0.0, 0.0);
        let mut m = 0;
        let lwork = self.work.len() as FInt;
        let liwork = self.iwork.len() as FInt;
        let mut info = 0;
        unsafe {
            (self.dsyevr)(
                &C(b'V'), &C(b'I'), &C(b'L'), &ni, self.a.as_mut_ptr(), &ni, &vl, &vu, &il, &iu, &abstol, &mut m, self.w.as_mut_ptr(), self.z.as_mut_ptr(), &ni, self.isuppz.as_mut_ptr(), self.work.as_mut_ptr(), &lwork,
                self.iwork.as_mut_ptr(), &liwork, &mut info, 1, 1, 1,
            )
        };
        assert_eq!(info, 0, "dsyevr info");
        assert_eq!(m as usize, k);
    }
    fn sygst(&mut self, n: usize) {
        let ni = n as FInt;
        let mut info = 0;
        unsafe { (self.dsygst)(&1, &C(b'L'), &ni, self.a.as_mut_ptr(), &ni, self.lfac.as_ptr(), &ni, &mut info, 1) };
        assert_eq!(info, 0);
    }
    /// B := op(L)^-1 B (side L) or B op(L)^-1 (side R), L = lower Cholesky factor
    fn trsm(&self, side: u8, trans: u8, n: usize, ncols: usize, bmat: *mut f64) {
        let ni = n as FInt;
        let (m, nn) = if side == b'L' { (ni, ncols as FInt) } else { (ncols as FInt, ni) };
        unsafe { (self.dtrsm)(&C(side), &C(b'L'), &C(trans), &C(b'N'), &m, &nn, &1.0, self.lfac.as_ptr(), &ni, bmat, &m, 1, 1, 1, 1) };
    }
}

impl Backend for Lapack {
    fn describe(&self) -> String {
        format!("{} | {}", self.name, self.info)
    }
    fn has(&self, _op: &str) -> bool {
        true
    }
    fn prepare(&mut self, n: usize, _f: &[f64], s: &[f64]) {
        let ni = n as FInt;
        self.a = vec![0.0; n * n];
        self.z = vec![0.0; n * n];
        self.w = vec![0.0; n];
        self.d = vec![0.0; n];
        self.e = vec![0.0; n];
        self.tau = vec![0.0; n];
        self.isuppz = vec![0; 2 * n];
        self.b = (0..n * n).map(|k| ((k % n * 7 + k / n * 13) % 17) as f64 / 17.0 - 0.5).collect();
        // Cholesky of S once (outside timing)
        self.lfac = s.to_vec();
        let mut info = 0;
        unsafe { (self.dpotrf)(&C(b'L'), &ni, self.lfac.as_mut_ptr(), &ni, &mut info, 1) };
        assert_eq!(info, 0, "dpotrf");
        // workspace queries (once per n, outside timing)
        let mut wq = [0.0f64; 1];
        let mut iwq: [FInt; 1] = [0];
        let (mut lw, mut liw) = (1usize, 1usize);
        let m1: FInt = -1;
        unsafe {
            (self.dsyevd)(&C(b'V'), &C(b'L'), &ni, self.a.as_mut_ptr(), &ni, self.w.as_mut_ptr(), wq.as_mut_ptr(), &m1, iwq.as_mut_ptr(), &m1, &mut info, 1, 1);
            lw = lw.max(wq[0] as usize);
            liw = liw.max(iwq[0] as usize);
            let (il, iu) = (1 as FInt, k40(n) as FInt);
            let mut m = 0;
            (self.dsyevr)(
                &C(b'V'), &C(b'I'), &C(b'L'), &ni, self.a.as_mut_ptr(), &ni, &0.0, &0.0, &il, &iu, &0.0, &mut m, self.w.as_mut_ptr(), self.z.as_mut_ptr(), &ni, self.isuppz.as_mut_ptr(), wq.as_mut_ptr(), &m1,
                iwq.as_mut_ptr(), &m1, &mut info, 1, 1, 1,
            );
            lw = lw.max(wq[0] as usize);
            liw = liw.max(iwq[0] as usize);
            (self.dsytrd)(&C(b'L'), &ni, self.a.as_mut_ptr(), &ni, self.d.as_mut_ptr(), self.e.as_mut_ptr(), self.tau.as_mut_ptr(), wq.as_mut_ptr(), &m1, &mut info, 1);
            lw = lw.max(wq[0] as usize);
            let k = k40(n) as FInt;
            let mut tryrac: FInt = 1;
            (self.dstemr)(
                &C(b'V'), &C(b'I'), &ni, self.d.as_mut_ptr(), self.e.as_mut_ptr(), &0.0, &0.0, &il, &iu, &mut m, self.w.as_mut_ptr(), self.z.as_mut_ptr(), &ni, &k, self.isuppz.as_mut_ptr(), &mut tryrac, wq.as_mut_ptr(), &m1,
                iwq.as_mut_ptr(), &m1, &mut info, 1, 1,
            );
            lw = lw.max(wq[0] as usize);
            liw = liw.max(iwq[0] as usize);
            (self.dormtr)(&C(b'L'), &C(b'L'), &C(b'N'), &ni, &k, self.a.as_ptr(), &ni, self.tau.as_ptr(), self.z.as_mut_ptr(), &ni, wq.as_mut_ptr(), &m1, &mut info, 1, 1, 1);
            lw = lw.max(wq[0] as usize);
        }
        self.work = vec![0.0; lw + 64];
        self.iwork = vec![0; liw + 64];
    }
    fn run(&mut self, op: &str, n: usize, f: &[f64]) {
        let ni = n as FInt;
        let k = k40(n);
        match op {
            "evd" => {
                self.a.copy_from_slice(f);
                self.syevd(n);
                self.last_k = n;
            }
            "gen" => {
                self.a.copy_from_slice(f);
                self.sygst(n);
                self.syevd(n);
                let p = self.a.as_mut_ptr();
                self.trsm(b'L', b'T', n, n, p);
                self.last_k = n;
            }
            "gen2t" => {
                self.a.copy_from_slice(f);
                let p = self.a.as_mut_ptr();
                self.trsm(b'L', b'N', n, n, p); // L^-1 F
                self.trsm(b'R', b'T', n, n, p); // (L^-1 F) L^-T
                self.syevd(n);
                self.trsm(b'L', b'T', n, n, p);
                self.last_k = n;
            }
            "evr40" => {
                self.a.copy_from_slice(f);
                self.syevr_k(n, k);
                self.last_k = k;
            }
            "evrall" => {
                self.a.copy_from_slice(f);
                self.syevr_k(n, n);
                self.last_k = n;
            }
            "genr40" => {
                self.a.copy_from_slice(f);
                self.sygst(n);
                self.syevr_k(n, k);
                let p = self.z.as_mut_ptr();
                self.trsm(b'L', b'T', n, k, p);
                self.last_k = k;
            }
            "mr40" => {
                self.a.copy_from_slice(f);
                let mut info = 0;
                let lwork = self.work.len() as FInt;
                let liwork = self.iwork.len() as FInt;
                let ki = k as FInt;
                let mut m = 0;
                let mut tryrac: FInt = 1;
                unsafe {
                    (self.dsytrd)(&C(b'L'), &ni, self.a.as_mut_ptr(), &ni, self.d.as_mut_ptr(), self.e.as_mut_ptr(), self.tau.as_mut_ptr(), self.work.as_mut_ptr(), &lwork, &mut info, 1);
                    assert_eq!(info, 0);
                    (self.dstemr)(
                        &C(b'V'), &C(b'I'), &ni, self.d.as_mut_ptr(), self.e.as_mut_ptr(), &0.0, &0.0, &1, &ki, &mut m, self.w.as_mut_ptr(), self.z.as_mut_ptr(), &ni, &ki, self.isuppz.as_mut_ptr(), &mut tryrac,
                        self.work.as_mut_ptr(), &lwork, self.iwork.as_mut_ptr(), &liwork, &mut info, 1, 1,
                    );
                    assert_eq!(info, 0, "dstemr");
                    assert_eq!(m as usize, k);
                    (self.dormtr)(&C(b'L'), &C(b'L'), &C(b'N'), &ni, &ki, self.a.as_ptr(), &ni, self.tau.as_ptr(), self.z.as_mut_ptr(), &ni, self.work.as_mut_ptr(), &lwork, &mut info, 1, 1, 1);
                    assert_eq!(info, 0);
                }
                self.last_k = k;
            }
            "trd" | "trd_mr" => {
                // diagnostics: dsytrd alone / dsytrd + dstemr(subset), no back-transform
                self.a.copy_from_slice(f);
                let mut info = 0;
                let lwork = self.work.len() as FInt;
                let liwork = self.iwork.len() as FInt;
                let ki = k as FInt;
                let mut m = 0;
                let mut tryrac: FInt = 1;
                unsafe {
                    (self.dsytrd)(&C(b'L'), &ni, self.a.as_mut_ptr(), &ni, self.d.as_mut_ptr(), self.e.as_mut_ptr(), self.tau.as_mut_ptr(), self.work.as_mut_ptr(), &lwork, &mut info, 1);
                    if op == "trd_mr" {
                        (self.dstemr)(
                            &C(b'V'), &C(b'I'), &ni, self.d.as_mut_ptr(), self.e.as_mut_ptr(), &0.0, &0.0, &1, &ki, &mut m, self.w.as_mut_ptr(), self.z.as_mut_ptr(), &ni, &ki, self.isuppz.as_mut_ptr(), &mut tryrac,
                            self.work.as_mut_ptr(), &lwork, self.iwork.as_mut_ptr(), &liwork, &mut info, 1, 1,
                        );
                    }
                }
                self.last_k = 0;
            }
            "gemm" => unsafe {
                (self.dgemm)(&C(b'N'), &C(b'N'), &ni, &ni, &ni, &1.0, f.as_ptr(), &ni, self.b.as_ptr(), &ni, &0.0, self.z.as_mut_ptr(), &ni, 1, 1);
            },
            _ => panic!("unknown op {op}"),
        }
    }
    fn result(&mut self, op: &str, _n: usize) -> Option<(Vec<f64>, Vec<f64>, usize)> {
        match op {
            "evd" | "gen" | "gen2t" => Some((self.w.clone(), self.a.clone(), self.last_k)),
            "evr40" | "genr40" | "mr40" | "evrall" => Some((self.w.clone(), self.z.clone(), self.last_k)),
            "trd" | "trd_mr" => None,
            _ => None,
        }
    }
}

// ---------------------------------------------------------------- driver

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = v.len() / 2;
    if v.len() % 2 == 1 { v[m] } else { 0.5 * (v[m - 1] + v[m]) }
}

/// Nested-parallelism check: `k` rayon workers each call dsyevd on their own n x n matrix at
/// the same time while the library's global thread count is `threads`. With `local`, every
/// worker first sets its own library thread count to 1 through the per-thread API
/// (openblas_set_num_threads_local / mkl_set_num_threads_local).
fn nested(path: &str, threads: usize, k: usize, n: usize, local: bool, f: &[f64]) {
    use rayon::prelude::*;
    unsafe {
        let lib = libloading::Library::new(path).unwrap();
        let is_mkl = lib.get::<*const ()>(b"MKL_Set_Num_Threads_Local\0").is_ok();
        if is_mkl {
            let set_layer = *lib.get::<unsafe extern "C" fn(c_int) -> c_int>(b"MKL_Set_Threading_Layer\0").unwrap();
            set_layer(0);
            (*lib.get::<unsafe extern "C" fn(c_int)>(b"MKL_Set_Num_Threads\0").unwrap())(threads as c_int);
        } else {
            (*lib.get::<unsafe extern "C" fn(c_int)>(b"openblas_set_num_threads\0").unwrap())(threads as c_int);
        }
        let set_local: unsafe extern "C" fn(c_int) -> c_int = if is_mkl {
            *lib.get(b"MKL_Set_Num_Threads_Local\0").unwrap()
        } else {
            *lib.get(b"openblas_set_num_threads_local\0").unwrap()
        };
        let dsyevd: sig::dsyevd = *lib.get(b"dsyevd_\0").unwrap();
        let ni = n as FInt;
        let pool = rayon::ThreadPoolBuilder::new().num_threads(k).build().unwrap();
        let mut times = vec![];
        let c0 = cpu_time();
        let w0 = Instant::now();
        for _ in 0..5 {
            let t = Instant::now();
            pool.install(|| {
                (0..k).into_par_iter().for_each(|_| {
                    if local {
                        set_local(1);
                    }
                    let mut a = f.to_vec();
                    let mut w = vec![0.0; n];
                    let lwork = (1 + 6 * n + 2 * n * n) as FInt;
                    let liwork = (3 + 5 * n) as FInt;
                    let mut work = vec![0.0; lwork as usize];
                    let mut iwork = vec![0 as FInt; liwork as usize];
                    let mut info = 0;
                    dsyevd(&C(b'V'), &C(b'L'), &ni, a.as_mut_ptr(), &ni, w.as_mut_ptr(), work.as_mut_ptr(), &lwork, iwork.as_mut_ptr(), &liwork, &mut info, 1, 1);
                    assert_eq!(info, 0);
                });
            });
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let ratio = (cpu_time() - c0) / w0.elapsed().as_secs_f64();
        println!("nested\t{}\tglobal_threads={threads}\tworkers={k}\tn={n}\tlocal1={local}\tbatch_ms={:.1}\tcpu/wall={ratio:.1}", if is_mkl { "mkl" } else { "openblas" }, median(&mut times));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let backend = get("--backend").expect("--backend faer22|faer24|lapack:<lib.so>");
    let threads: usize = get("--threads").map(|s| s.parse().unwrap()).unwrap_or(1);
    let sizes: Vec<usize> = get("--sizes").unwrap_or("100,266,500,1000,2000,3000".into()).split(',').map(|s| s.parse().unwrap()).collect();
    let ops: Vec<String> = get("--ops").unwrap_or("evd,gen,gen2t,evr40,mr40,genr40,gemm".into()).split(',').map(String::from).collect();
    let budget: f64 = get("--budget").map(|s| s.parse().unwrap()).unwrap_or(2.0);
    let check = !args.iter().any(|a| a == "--nocheck");
    let matdir = get("--mat");

    if let Some(k) = get("--nested") {
        let n = sizes[0];
        let f = match &matdir {
            Some(d) => load_sub(d, "h0.bin", n),
            None => make_f(n),
        };
        nested(&backend[7..], threads, k.parse().unwrap(), n, args.iter().any(|a| a == "--local"), &f);
        return;
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    pool.install(|| {
        let mut be: Box<dyn Backend> = match backend.as_str() {
            "faer22" => Box::new(faer22_backend::Faer::new(threads)),
            "faer24" => Box::new(faer24_backend::Faer::new(threads)),
            "nalgebra" => Box::new(Nalgebra { w: vec![], v: vec![] }),
            s if s.starts_with("lapack:") => {
                let prefix = get("--prefix").unwrap_or_default();
                let blas = get("--blas");
                Box::new(unsafe { Lapack::load(&s[7..], threads, &prefix, blas.as_deref()) })
            }
            _ => panic!("unknown backend"),
        };
        let label = match backend.as_str() {
            s if s.starts_with("lapack:") => be.describe().split(' ').next().unwrap().to_string(),
            s => s.to_string(),
        };
        eprintln!("# {}", be.describe());
        for &n in &sizes {
            let (f, s) = match &matdir {
                Some(d) => (load_sub(d, "h0.bin", n), load_sub(d, "s.bin", n)),
                None => (make_f(n), make_s(n)),
            };
            be.prepare(n, &f, &s);
            for op in &ops {
                if !be.has(op) {
                    continue;
                }
                // warm-up (also gives a time estimate)
                let t0 = Instant::now();
                be.run(op, n, &f);
                let t_est = t0.elapsed().as_secs_f64();
                // correctness check on the warm-up result
                let (mut resid, mut csum) = (f64::NAN, f64::NAN);
                if let Some((w, x, k)) = be.result(op, n) {
                    csum = w[..k].iter().sum::<f64>();
                    if check && n <= 1000 {
                        let bmat = if op.starts_with("gen") { Some(&s[..]) } else { None };
                        resid = residual(n, &f, bmat, &x, &w, k);
                    }
                }
                let reps = ((budget / t_est.max(1e-6)) as usize).clamp(if n >= 2000 { 3 } else { 5 }, 101);
                let mut times = Vec::with_capacity(reps);
                let c0 = cpu_time();
                let w0 = Instant::now();
                for _ in 0..reps {
                    let t = Instant::now();
                    be.run(op, n, &f);
                    times.push(t.elapsed().as_secs_f64() * 1e3);
                }
                let wall = w0.elapsed().as_secs_f64();
                let cpu = cpu_time() - c0;
                let tmin = times.iter().cloned().fold(f64::INFINITY, f64::min);
                let med = median(&mut times);
                println!("{label}\t{threads}\t{n}\t{op}\t{med:.3}\t{tmin:.3}\t{reps}\t{:.2}\t{resid:.1e}\t{csum:.10e}", cpu / wall);
            }
        }
        // never dlclose a LAPACK: AOCL's OpenMP (libgomp) workers are still alive at this point,
        // and unloading libblis-mt/libflame under them segfaults at exit
        std::mem::forget(be);
    });
}
