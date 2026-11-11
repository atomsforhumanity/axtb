//! Optional LAPACK eigensolver loaded at run time (Intel MKL, OpenBLAS or AMD AOCL), for the
//! SCF's symmetric eigenproblems: tridiagonal reduction (`dsytrd`), divide and conquer on the
//! tridiagonal matrix (`dstedc`) and the back-transformation of the needed eigenvectors
//! (`dormtr`), the sequence of xtb 6.8.0 (grimme-lab/xtb#1466). Nothing is linked into the
//! build: on Linux the library is opened with `dlopen` when the environment variable
//! `AXTB_LAPACK` names it (a path or a soname on the loader path, e.g. `libmkl_rt.so.2`); for
//! AOCL `AXTB_LAPACK_BLAS` names the BLIS library that libFLAME needs, loaded first. The
//! variables are read once per process, at the first SCF of 200 or more basis functions; later
//! changes have no effect. faer stays the default, and the fallback when the library cannot be
//! used ([`crate::linalg::runtime_lapack`] and `SinglePoint::eigensolver` say why; the CLI and
//! the Python module warn). Other platforms always use faer.
//!
//! Threads: every call sets the library's thread count for the calling thread only
//! (`mkl_set_num_threads_local`, `openblas_set_num_threads_local`,
//! `bli_thread_set_num_threads_local` with `omp_set_num_threads`) to the calculator's thread
//! count and restores it afterwards. The library runs its own OpenMP threads while the
//! calculator's pool waits. A library without a per-thread thread count is refused, since it
//! would choose its own (all cores): OpenBLAS must be an OpenMP build (Debian and Ubuntu:
//! `libopenblas0-openmp`) or a sequential one; the pthreads build (`libopenblas0-pthread`) has
//! `openblas_set_num_threads_local` too, but there it changes nothing, its thread count being
//! process-wide (`openblas_get_parallel` tells them apart). Only LP64 libraries (32-bit
//! integers) work: MKL is set to
//! LP64, an OpenBLAS built with 64-bit integers is refused, and an AOCL library must be an LP64
//! build. MKL's Intel OpenMP runs without thread affinity (`kmp_set_defaults`, which applies to
//! every user of Intel OpenMP in the process): its topology detection cost ~0.1 s per process
//! on the 128-CPU nodes.
//!
//! Reproducibility: MKL runs in strict conditional numerical reproducibility mode
//! (`MKL_CBWR_Set(AUTO | STRICT)`, before its first call), so that its results repeat bit for
//! bit on a given thread count; if MKL cannot be put in that mode (it was initialised in the
//! process before, e.g. by an MKL build of NumPy) or its threading or interface layer cannot be
//! set, it is refused. `dsytrd` (a threaded symmetric matrix-vector product with one partial
//! sum per thread in every library) makes the results differ between thread counts by
//! round-off, as faer's parallel tridiagonalization does in [`crate::linalg`] and xtb's MKL
//! does. Against faer the results differ by round-off (energies ~1e-13 Eh, gradients ~1e-15
//! Eh/bohr).

use std::ffi::{c_char, c_int};
#[cfg(target_os = "linux")]
use std::ffi::{c_void, CStr, CString};
use std::sync::OnceLock;

type FInt = c_int;
type L = usize; // hidden Fortran character-length argument

/// `dlopen` and its flags (glibc's values: Linux only).
#[cfg(target_os = "linux")]
mod dl {
    use std::ffi::{c_char, c_int, c_void, CStr};

    extern "C" {
        pub fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlerror() -> *const c_char;
    }
    pub const RTLD_NOW: c_int = 2;
    pub const RTLD_GLOBAL: c_int = 0x100;

    pub fn last_error() -> String {
        // SAFETY: dlerror returns null or a NUL-terminated string valid until the next call
        unsafe {
            let e = dlerror();
            if e.is_null() {
                "unknown error".into()
            } else {
                CStr::from_ptr(e).to_string_lossy().into_owned()
            }
        }
    }
}

/// MKL's settings (`mkl_service.h`, `mkl_cbwr.h`).
#[cfg(target_os = "linux")]
mod mkl {
    use std::ffi::c_int;
    pub const CBWR_AUTO: c_int = 2;
    pub const CBWR_STRICT: c_int = 0x10000;
    pub const CBWR_SUCCESS: c_int = 0;
    pub const THREADING_INTEL: c_int = 0;
    pub const INTERFACE_LP64: c_int = 0;
    pub const INTERFACE_ILP64: c_int = 1;
}

#[allow(non_camel_case_types)]
type dsytrd_t = unsafe extern "C" fn(
    *const c_char,
    *const FInt,
    *mut f64,
    *const FInt,
    *mut f64,
    *mut f64,
    *mut f64,
    *mut f64,
    *const FInt,
    *mut FInt,
    L,
);
#[allow(non_camel_case_types)]
type dstedc_t = unsafe extern "C" fn(
    *const c_char,
    *const FInt,
    *mut f64,
    *mut f64,
    *mut f64,
    *const FInt,
    *mut f64,
    *const FInt,
    *mut FInt,
    *const FInt,
    *mut FInt,
    L,
);
#[allow(non_camel_case_types)]
type dormtr_t = unsafe extern "C" fn(
    *const c_char,
    *const c_char,
    *const c_char,
    *const FInt,
    *const FInt,
    *const f64,
    *const FInt,
    *const f64,
    *mut f64,
    *const FInt,
    *mut f64,
    *const FInt,
    *mut FInt,
    L,
    L,
    L,
);

#[allow(non_camel_case_types)]
type dsygst_t = unsafe extern "C" fn(
    *const FInt,
    *const c_char,
    *const FInt,
    *mut f64,
    *const FInt,
    *const f64,
    *const FInt,
    *mut FInt,
    L,
);
#[allow(non_camel_case_types)]
type dtrsm_t = unsafe extern "C" fn(
    *const c_char,
    *const c_char,
    *const c_char,
    *const c_char,
    *const FInt,
    *const FInt,
    *const f64,
    *const f64,
    *const FInt,
    *mut f64,
    *const FInt,
    L,
    L,
    L,
    L,
);

/// The thread-count setter of a sequential library (always one thread).
#[cfg(target_os = "linux")]
unsafe extern "C" fn sequential(_threads: c_int) -> c_int {
    1
}

/// A loaded LAPACK library.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) struct Lapack {
    /// "mkl", "openblas" or "aocl".
    pub name: &'static str,
    /// Version and threading information for logs.
    pub info: String,
    dsytrd: dsytrd_t,
    dstedc: dstedc_t,
    dormtr: dormtr_t,
    dsygst: dsygst_t,
    dtrsm: dtrsm_t,
    /// The library's thread count for the calling thread.
    set_threads: unsafe extern "C" fn(c_int) -> c_int,
    omp_set: Option<unsafe extern "C" fn(c_int)>,
    omp_get: Option<unsafe extern "C" fn() -> c_int>,
}

// SAFETY: the function pointers are plain C entry points of a library that stays loaded
unsafe impl Send for Lapack {}
unsafe impl Sync for Lapack {}

impl Lapack {
    /// Run-time libraries are opened on Linux only.
    #[cfg(not(target_os = "linux"))]
    pub fn open(path: &str, _blas: Option<&str>) -> Result<Self, String> {
        Err(format!("{path}: run-time LAPACK libraries are supported on Linux only"))
    }

    /// Open `path` (and first `blas`, globally, when given), refusing a library whose thread
    /// count cannot be set per thread, an ILP64 OpenBLAS, and an MKL that cannot be set up for
    /// LP64, Intel OpenMP and strict reproducibility.
    #[cfg(target_os = "linux")]
    pub fn open(path: &str, blas: Option<&str>) -> Result<Self, String> {
        use dl::{dlopen, dlsym, last_error, RTLD_GLOBAL, RTLD_NOW};
        // SAFETY: dlopen/dlsym with NUL-terminated names; every symbol is transmuted to the
        // signature its library documents, and the libraries stay loaded for the process
        unsafe {
            let mut handles = Vec::new();
            if let Some(b) = blas {
                let c = CString::new(b).map_err(|e| e.to_string())?;
                let h = dlopen(c.as_ptr(), RTLD_NOW | RTLD_GLOBAL);
                if h.is_null() {
                    return Err(format!("dlopen {b}: {}", last_error()));
                }
                handles.push(h);
            }
            let c = CString::new(path).map_err(|e| e.to_string())?;
            let h = dlopen(c.as_ptr(), RTLD_NOW | RTLD_GLOBAL);
            if h.is_null() {
                return Err(format!("dlopen {path}: {}", last_error()));
            }
            handles.insert(0, h);
            let raw = |name: &str| -> Option<*mut c_void> {
                let n = CString::new(name).ok()?;
                handles.iter().map(|&h| dlsym(h, n.as_ptr())).find(|p| !p.is_null())
            };
            macro_rules! sym {
                ($t:ty, $n:expr) => {
                    std::mem::transmute::<*mut c_void, $t>(raw($n).ok_or_else(|| format!("{path}: no symbol {}", $n))?)
                };
            }
            macro_rules! opt {
                ($t:ty, $n:expr) => {
                    raw($n).map(|p| std::mem::transmute::<*mut c_void, $t>(p))
                };
            }
            let (name, info, set_threads) = if let Some(layer) =
                opt!(unsafe extern "C" fn(c_int) -> c_int, "MKL_Set_Threading_Layer")
            {
                // Intel OpenMP works out the CPU topology for thread affinity on its first
                // parallel region: ~0.1 s per process on a 128-CPU node (3 ms on 16). axtb pins
                // nothing, so affinity is off (`kmp_set_defaults`, before the runtime starts).
                let iomp = std::path::Path::new(path).parent().map(|d| d.join("libiomp5.so")).filter(|p| p.exists());
                let iomp = iomp
                    .and_then(|p| CString::new(p.to_string_lossy().as_bytes()).ok())
                    .unwrap_or_else(|| c"libiomp5.so".into());
                let h = dlopen(iomp.as_ptr(), RTLD_NOW | RTLD_GLOBAL);
                if !h.is_null() {
                    let f = dlsym(h, c"kmp_set_defaults".as_ptr());
                    if !f.is_null() {
                        std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*const c_char)>(f)(
                            c"KMP_AFFINITY=disabled".as_ptr(),
                        );
                    }
                }
                // LP64, Intel OpenMP threading, strict CNR (results repeat bit for bit on a
                // given thread count), all before the first computation; each setter returns
                // the setting in effect, which differs when MKL was set up before
                let iface = sym!(unsafe extern "C" fn(c_int) -> c_int, "MKL_Set_Interface_Layer");
                let got = iface(mkl::INTERFACE_LP64);
                if got & mkl::INTERFACE_ILP64 != 0 {
                    return Err(format!("{path}: MKL uses the ILP64 interface ({got}), axtb needs LP64"));
                }
                let got = layer(mkl::THREADING_INTEL);
                if got != mkl::THREADING_INTEL {
                    return Err(format!("{path}: MKL's threading layer is {got}, not Intel OpenMP"));
                }
                let cbwr =
                    sym!(unsafe extern "C" fn(c_int) -> c_int, "MKL_CBWR_Set")(mkl::CBWR_AUTO | mkl::CBWR_STRICT);
                if cbwr != mkl::CBWR_SUCCESS {
                    return Err(format!(
                        "{path}: MKL_CBWR_Set(AUTO | STRICT) returned {cbwr} (MKL was initialised before axtb \
                         loaded it?); without it MKL's results would not repeat bit for bit"
                    ));
                }
                let mut buf = vec![0 as c_char; 256];
                if let Some(v) = opt!(unsafe extern "C" fn(*mut c_char, c_int), "MKL_Get_Version_String") {
                    v(buf.as_mut_ptr(), 255);
                }
                let ver = CStr::from_ptr(buf.as_ptr()).to_string_lossy().trim().to_string();
                (
                    "mkl",
                    format!("{ver} | CNR AUTO,STRICT"),
                    opt!(unsafe extern "C" fn(c_int) -> c_int, "MKL_Set_Num_Threads_Local"),
                )
            } else if let Some(cfg) = opt!(unsafe extern "C" fn() -> *const c_char, "openblas_get_config") {
                let info = CStr::from_ptr(cfg()).to_string_lossy().trim().to_string();
                if info.contains("USE64BITINT") {
                    return Err(format!("{path}: OpenBLAS with 64-bit integers ({info}), axtb needs LP64"));
                }
                // 0: sequential, 1: pthreads, 2: OpenMP. The pthreads build has
                // openblas_set_num_threads_local too, but its thread count is process-wide and
                // the setter changes nothing: it runs on all cores.
                let parallel = sym!(unsafe extern "C" fn() -> c_int, "openblas_get_parallel")();
                let set = match parallel {
                    0 => Some(sequential as unsafe extern "C" fn(c_int) -> c_int),
                    2 => opt!(unsafe extern "C" fn(c_int) -> c_int, "openblas_set_num_threads_local"),
                    _ => None,
                };
                if set.is_none() {
                    return Err(format!(
                        "{path}: OpenBLAS without a per-thread thread count (threading model {parallel}, 1 = pthreads; \
                         {info}), so it would run on all cores: use an OpenMP build (Debian/Ubuntu: libopenblas0-openmp)"
                    ));
                }
                ("openblas", info, set)
            } else if let Some(set) = opt!(unsafe extern "C" fn(c_int) -> c_int, "bli_thread_set_num_threads_local") {
                let mut info = String::new();
                if let Some(v) = opt!(unsafe extern "C" fn() -> *const c_char, "FLA_Get_AOCL_Version") {
                    info = CStr::from_ptr(v()).to_string_lossy().trim().to_string();
                }
                if let Some(v) = opt!(unsafe extern "C" fn() -> *const c_char, "bli_info_get_version_str") {
                    info = format!("{info} | BLIS {}", CStr::from_ptr(v()).to_string_lossy());
                }
                ("aocl", info, Some(set))
            } else {
                return Err(format!("{path}: not MKL, OpenBLAS or AOCL, so its thread count cannot be set per thread"));
            };
            let set_threads = set_threads.ok_or_else(|| format!("{path}: MKL without MKL_Set_Num_Threads_Local"))?;
            Ok(Lapack {
                name,
                info,
                dsytrd: sym!(dsytrd_t, "dsytrd_"),
                dstedc: sym!(dstedc_t, "dstedc_"),
                dormtr: sym!(dormtr_t, "dormtr_"),
                dsygst: sym!(dsygst_t, "dsygst_"),
                dtrsm: sym!(dtrsm_t, "dtrsm_"),
                set_threads,
                omp_set: opt!(unsafe extern "C" fn(c_int), "omp_set_num_threads"),
                omp_get: opt!(unsafe extern "C" fn() -> c_int, "omp_get_max_threads"),
            })
        }
    }

    /// The library named by `AXTB_LAPACK` (and `AXTB_LAPACK_BLAS`), loaded once per process:
    /// `Ok(None)` without the variable, `Err` with the reason when it names a library that
    /// cannot be used. Nothing is printed: the reason is in `SinglePoint::eigensolver` and
    /// [`crate::linalg::runtime_lapack`], and the front ends warn with it.
    pub fn from_env() -> Result<Option<&'static Lapack>, &'static str> {
        static LIB: OnceLock<Result<Option<Lapack>, String>> = OnceLock::new();
        let lib = LIB.get_or_init(|| {
            let Some(path) = std::env::var("AXTB_LAPACK").ok().filter(|s| !s.is_empty()) else {
                return Ok(None);
            };
            let blas = std::env::var("AXTB_LAPACK_BLAS").ok().filter(|s| !s.is_empty());
            Lapack::open(&path, blas.as_deref()).map(Some)
        });
        match lib {
            Ok(l) => Ok(l.as_ref()),
            Err(e) => Err(e.as_str()),
        }
    }

    /// Name and version for the results, e.g. `"mkl (Intel(R) oneAPI Math Kernel Library ...)"`.
    pub fn describe(&self) -> String {
        format!("{} ({})", self.name, self.info)
    }

    /// Run `f` with the library's thread count set to `threads` for the calling thread.
    fn with_threads<T>(&self, threads: usize, f: impl FnOnce() -> T) -> T {
        let t = threads.max(1) as c_int;
        // SAFETY: the libraries' documented thread-count setters and getters
        unsafe {
            let prev = (self.set_threads)(t);
            let prev_omp = match (self.omp_set, self.omp_get) {
                (Some(set), Some(get)) if self.name == "aocl" => {
                    let p = get();
                    set(t);
                    Some((set, p))
                }
                _ => None,
            };
            let r = f();
            // MKL's and OpenBLAS's local setters return the previous value (0: none set)
            if self.name != "aocl" {
                (self.set_threads)(prev);
            }
            if let Some((set, p)) = prev_omp {
                set(p);
            }
            r
        }
    }
}

/// Work space of the library eigensolver for one matrix size.
pub struct LapackEig {
    n: usize,
    /// The matrix of [`LapackEig::solve_generalized`] (n x n, allocated on first use).
    a: Vec<f64>,
    tau: Vec<f64>,
    d: Vec<f64>,
    e: Vec<f64>,
    z: Vec<f64>,
    work: Vec<f64>,
    iwork: Vec<FInt>,
}

impl LapackEig {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            a: Vec::new(),
            tau: vec![0.0; n.max(1)],
            d: vec![0.0; n],
            e: vec![0.0; n.max(1)],
            z: vec![0.0; n * n],
            work: Vec::new(),
            iwork: Vec::new(),
        }
    }

    /// Eigenvalues (ascending, into `w`) of the symmetric matrix whose lower triangle is in the
    /// column-major `a` (leading dimension `lda`, overwritten), and the eigenvectors of the
    /// lowest `k(w)` of them into the column-major `v` (n x k, leading dimension n). Returns k.
    pub fn solve(
        &mut self,
        lib: &Lapack,
        threads: usize,
        a: &mut [f64],
        lda: usize,
        w: &mut [f64],
        v: &mut Vec<f64>,
        k: impl FnOnce(&[f64]) -> usize,
    ) -> Result<usize, String> {
        let kk = lib.with_threads(threads, || self.eig(lib, a, lda, w, k))?;
        v.clear();
        v.extend_from_slice(&self.z[..self.n * kk]);
        Ok(kk)
    }

    /// The generalized problem F C = S C ε with S = L Lᵀ, all in the library: the reduction
    /// A = L⁻¹ F L⁻ᵀ by `dsygst` (lower triangle of `f` read), the eigenvalues of A (ascending,
    /// into `w`) by `dsytrd` + `dstedc`, its eigenvectors U of the lowest `k(w)` by `dormtr`,
    /// and C = L⁻ᵀ U by `dtrsm`, left in [`vectors`](Self::vectors). `l` is the lower Cholesky
    /// factor (column-major; its strict upper triangle is not read). Returns k.
    pub fn solve_generalized(
        &mut self,
        lib: &Lapack,
        threads: usize,
        f: faer::MatRef<'_, f64>,
        l: faer::MatRef<'_, f64>,
        w: &mut [f64],
        k: impl FnOnce(&[f64]) -> usize,
    ) -> Result<usize, String> {
        let n = self.n;
        assert!(f.nrows() == n && f.ncols() == n && l.nrows() == n && l.ncols() == n);
        let l = l.try_as_col_major().ok_or("the Cholesky factor is not column-major")?;
        let (ni, ldl) = (n as FInt, l.col_stride() as FInt);
        let mut a = std::mem::take(&mut self.a);
        a.resize(n * n, 0.0);
        faer::MatMut::from_column_major_slice_mut(&mut a, n, n).copy_from_triangular_lower(f);
        let c = |b: u8| b as c_char;
        let r = lib.with_threads(threads, || {
            let mut info: FInt = 0;
            // SAFETY: `a` is n x n at leading dimension n, `l` n x n at leading dimension
            // ldl >= n (column-major)
            unsafe { (lib.dsygst)(&1, &c(b'L'), &ni, a.as_mut_ptr(), &ni, l.as_ptr(), &ldl, &mut info, 1) };
            if info != 0 {
                return Err(format!("dsygst info {info}"));
            }
            let kk = self.eig(lib, &mut a, n, w, k)?;
            if kk > 0 {
                let ki = kk as FInt;
                // SAFETY: `l` as above, the first k <= n columns of `z` (n x n, leading
                // dimension n), set by `eig`
                unsafe {
                    (lib.dtrsm)(
                        &c(b'L'),
                        &c(b'L'),
                        &c(b'T'),
                        &c(b'N'),
                        &ni,
                        &ki,
                        &1.0,
                        l.as_ptr(),
                        &ldl,
                        self.z.as_mut_ptr(),
                        &ni,
                        1,
                        1,
                        1,
                        1,
                    );
                }
            }
            Ok(kk)
        });
        self.a = a;
        r
    }

    /// The k eigenvectors of the last [`solve_generalized`](Self::solve_generalized) call
    /// (column-major n x k, leading dimension n).
    pub fn vectors(&self, k: usize) -> &[f64] {
        &self.z[..self.n * k]
    }

    /// `dsytrd` + `dstedc` + `dormtr` on the symmetric matrix whose lower triangle is in the
    /// column-major `a` (leading dimension `lda`, overwritten): eigenvalues into `w`, the
    /// eigenvectors of the lowest `k(w)` into `z`; returns k. The caller sets the library's
    /// thread count.
    fn eig(
        &mut self,
        lib: &Lapack,
        a: &mut [f64],
        lda: usize,
        w: &mut [f64],
        k: impl FnOnce(&[f64]) -> usize,
    ) -> Result<usize, String> {
        let n = self.n;
        let ni = n as FInt;
        let ldai = lda as FInt;
        let c = |b: u8| b as c_char;
        assert!(lda >= n.max(1) && a.len() >= lda * n.saturating_sub(1) + n && w.len() >= n);
        // SAFETY: the library's documented routines on `a` (n x n at leading dimension lda,
        // checked above), `d`, `e`, `tau` and `z` (n x n) of this size, and work arrays of the
        // sizes the routines asked for
        unsafe {
            let mut info: FInt = 0;
            // work space: the largest of the three queries
            let mut q = [0.0f64; 1];
            let mut qi: [FInt; 1] = [0];
            let m1: FInt = -1;
            (lib.dsytrd)(
                &c(b'L'),
                &ni,
                a.as_mut_ptr(),
                &ldai,
                self.d.as_mut_ptr(),
                self.e.as_mut_ptr(),
                self.tau.as_mut_ptr(),
                q.as_mut_ptr(),
                &m1,
                &mut info,
                1,
            );
            let mut lw = q[0] as usize;
            (lib.dstedc)(
                &c(b'I'),
                &ni,
                self.d.as_mut_ptr(),
                self.e.as_mut_ptr(),
                self.z.as_mut_ptr(),
                &ni,
                q.as_mut_ptr(),
                &m1,
                qi.as_mut_ptr(),
                &m1,
                &mut info,
                1,
            );
            lw = lw.max(q[0] as usize);
            let liw = (qi[0] as usize).max(1);
            (lib.dormtr)(
                &c(b'L'),
                &c(b'L'),
                &c(b'N'),
                &ni,
                &ni,
                a.as_ptr(),
                &ldai,
                self.tau.as_ptr(),
                self.z.as_mut_ptr(),
                &ni,
                q.as_mut_ptr(),
                &m1,
                &mut info,
                1,
                1,
                1,
            );
            lw = lw.max(q[0] as usize).max(1);
            if self.work.len() < lw {
                self.work.resize(lw, 0.0);
            }
            if self.iwork.len() < liw {
                self.iwork.resize(liw, 0);
            }
            let lwi = self.work.len() as FInt;
            let liwi = self.iwork.len() as FInt;
            (lib.dsytrd)(
                &c(b'L'),
                &ni,
                a.as_mut_ptr(),
                &ldai,
                self.d.as_mut_ptr(),
                self.e.as_mut_ptr(),
                self.tau.as_mut_ptr(),
                self.work.as_mut_ptr(),
                &lwi,
                &mut info,
                1,
            );
            if info != 0 {
                return Err(format!("dsytrd info {info}"));
            }
            (lib.dstedc)(
                &c(b'I'),
                &ni,
                self.d.as_mut_ptr(),
                self.e.as_mut_ptr(),
                self.z.as_mut_ptr(),
                &ni,
                self.work.as_mut_ptr(),
                &lwi,
                self.iwork.as_mut_ptr(),
                &liwi,
                &mut info,
                1,
            );
            if info != 0 {
                return Err(format!("dstedc info {info}"));
            }
            w[..n].copy_from_slice(&self.d);
            let kk = k(&w[..n]).min(n);
            if kk > 0 {
                let ki = kk as FInt;
                (lib.dormtr)(
                    &c(b'L'),
                    &c(b'L'),
                    &c(b'N'),
                    &ni,
                    &ki,
                    a.as_ptr(),
                    &ldai,
                    self.tau.as_ptr(),
                    self.z.as_mut_ptr(),
                    &ni,
                    self.work.as_mut_ptr(),
                    &lwi,
                    &mut info,
                    1,
                    1,
                    1,
                );
                if info != 0 {
                    return Err(format!("dormtr info {info}"));
                }
            }
            Ok(kk)
        }
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::Lapack;

    fn refusal(path: &str) -> String {
        match Lapack::open(path, None) {
            Ok(lib) => panic!("{path} was accepted as {}", lib.describe()),
            Err(e) => e,
        }
    }

    #[test]
    fn missing_and_unknown_libraries_are_refused() {
        let e = refusal("/nonexistent/liblapack.so");
        assert!(e.starts_with("dlopen /nonexistent/liblapack.so:"), "{e}");
        // a library without a per-thread thread-count setter (and here without LAPACK)
        let e = refusal("libc.so.6");
        assert!(e.contains("not MKL, OpenBLAS or AOCL"), "{e}");
    }

    /// The pthreads build of OpenBLAS (Debian/Ubuntu `libopenblas0-pthread`) has no per-thread
    /// thread count and would run on every core. Loads the library into the test process, so
    /// on request only: `cargo test --release -p axtb-base --features linalg pthreads_openblas -- --ignored`.
    #[test]
    #[ignore = "needs Debian/Ubuntu's libopenblas0-pthread"]
    fn pthreads_openblas_is_refused() {
        let e = refusal("/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0");
        assert!(e.contains("threading model 1"), "{e}");
    }
}
