//! `axtb` command-line driver: one calculation per call, the result as JSON on stdout.
//!
//! `axtb mol.xyz [options]` runs the structure file (XYZ, Ångström); `axtb --json` reads a
//! JSON request (`axtb_driver::Request`, positions in bohr) on stdin instead. Both become one
//! `axtb_driver::Request`, which the driver checks and runs; the response
//! (`axtb_driver::Response`) is the one Python returns too. `axtb --help` lists the options.
//!
//! Errors: in `--json` mode `{"error": {"kind", "message", ...}}` on stdout, otherwise
//! `error: <message>` on stderr; exit code 1. Invalid arguments and unreadable files are usage
//! errors (`error: ...` on stderr, exit code 2).

use axtb_driver::{Method, Request, Temperatures};
use clap::parser::ValueSource;
use clap::{CommandFactory, FromArgMatches, Parser};
use std::io::Read;
use std::path::PathBuf;
use std::process::ExitCode;

const ABOUT: &str = "Extended tight binding (GFN2-xTB, g-xTB v1.1 and 2.0.1) and GFN-FF: energies, gradients, \
Hessians with vibrational analysis. Prints the result as JSON on stdout.";

const LONG_ABOUT: &str = "Extended tight binding (GFN2-xTB, g-xTB v1.1 and 2.0.1) and GFN-FF: energies, gradients, \
Hessians with vibrational analysis. Prints the result as JSON on stdout.

The structure file is XYZ in Ångström, converted to bohr as the reproduced program does in the program \
conventions (mctc-lib's factor; g-xTB 1.1.0's own) and with CODATA 2022 in axtb's. `axtb --json` reads a \
JSON request on stdin instead: the same settings under the names of the response (`numbers`, `positions` in \
bohr, `charge`, `uhf`, `method`, `gradient`, `hessian`, ...; see the README), unknown names refused. Settings \
that do not apply to the method are refused, not ignored. Names (methods, conventions, ...) ignore case and \
anything but letters and digits.

Errors: with --json `{\"error\": {\"kind\", \"message\", ...}}` on stdout, else `error: ...` on stderr; exit \
code 1 (2 for invalid arguments and unreadable files).";

/// The Molden file of `--molden` without a name, as `xtb --molden` names it.
const MOLDEN_FILE: &str = "molden.input";

#[derive(Parser, Debug)]
#[command(name = "axtb", version, about = ABOUT, long_about = LONG_ABOUT, allow_negative_numbers = true)]
struct Cli {
    /// Structure file (XYZ, Ångström)
    #[arg(value_name = "STRUCTURE")]
    structure: Option<PathBuf>,

    /// Read a JSON request on stdin instead of a structure file and options
    #[arg(long)]
    json: bool,

    /// Method: gfn2 (GFN2-xTB, default), gfnff (GFN-FF), gxtb-v1.1 or gxtb-v2.0.1 (g-xTB)
    #[arg(long, value_name = "METHOD", help_heading = "Method")]
    method: Option<String>,

    /// Conventions: axtb (default; CODATA 2022, fixes) or a program's: GFN2-xTB xtb (6.8.0),
    /// xtb-6.7.1, tblite; g-xTB gxtb; GFN-FF xtb, xtb-6.7.1
    #[arg(long, value_name = "NAME", help_heading = "Method")]
    conventions: Option<String>,

    /// Total charge
    #[arg(long = "chrg", value_name = "Q", help_heading = "Method")]
    charge: Option<f64>,

    /// Unpaired electrons (default: 0 or 1 by the electron count; GFN-FF ignores it)
    #[arg(long, value_name = "N", help_heading = "Method")]
    uhf: Option<i64>,

    /// Threads (default 1; more than 8 rarely help one SCF single point)
    #[arg(long, value_name = "N", help_heading = "Method")]
    threads: Option<usize>,

    /// Compute the gradient
    #[arg(long = "grad", help_heading = "Method")]
    gradient: bool,

    /// SCF accuracy multiplier (default 1)
    #[arg(long = "acc", value_name = "X", help_heading = "SCF (GFN2-xTB, g-xTB)")]
    accuracy: Option<f64>,

    /// Electronic temperature in K (default: GFN2-xTB 300, g-xTB 0)
    #[arg(long, value_name = "K", help_heading = "SCF (GFN2-xTB, g-xTB)")]
    etemp: Option<f64>,

    /// SCF iteration limit (default: GFN2-xTB and g-xTB 2.0.1 250, g-xTB v1.1 1000)
    #[arg(long = "max-iter", value_name = "N", help_heading = "SCF (GFN2-xTB, g-xTB)")]
    max_iter: Option<usize>,

    /// GFN2-xTB SCF mixer: auto, broyden or diis (default: the conventions')
    #[arg(long, value_name = "MIXER", help_heading = "SCF (GFN2-xTB, g-xTB)")]
    mixer: Option<String>,

    /// Print every GFN2-xTB SCF iteration to stderr
    #[arg(long = "scf-trace", help_heading = "SCF (GFN2-xTB, g-xTB)")]
    scf_trace: bool,

    /// g-xTB singlets: a broken-symmetry unrestricted solution where it lies lower (an axtb
    /// extension; uhf 0, axtb conventions)
    #[arg(long = "broken-symmetry", help_heading = "SCF (GFN2-xTB, g-xTB)")]
    broken_symmetry: bool,

    /// Leave out the dispersion (GFN2-xTB, for validation)
    #[arg(long, help_heading = "SCF (GFN2-xTB, g-xTB)")]
    nodisp: bool,

    /// Write the converged orbitals as a Molden file (default molden.input; FILE is the argument
    /// after --molden unless it is the only structure file; --molden=FILE is unambiguous)
    #[arg(
        long,
        value_name = "FILE",
        num_args = 0..=1,
        default_missing_value = MOLDEN_FILE,
        help_heading = "SCF (GFN2-xTB, g-xTB)"
    )]
    molden: Option<String>,

    /// Cartesian instead of spherical functions in the Molden file
    #[arg(long = "molden-cartesian", help_heading = "SCF (GFN2-xTB, g-xTB)")]
    molden_cartesian: bool,

    /// Add the Mayer-Wiberg bond orders
    #[arg(long = "wbo", help_heading = "SCF (GFN2-xTB, g-xTB)")]
    bond_orders: bool,

    /// Implicit solvent ALPB (GFN2-xTB; g-xTB with --solvation-correction)
    #[arg(long, value_name = "SOLVENT", conflicts_with = "gbsa", help_heading = "Implicit solvent")]
    alpb: Option<String>,

    /// Implicit solvent GBSA (GFN2-xTB; g-xTB with --solvation-correction)
    #[arg(long, value_name = "SOLVENT", help_heading = "Implicit solvent")]
    gbsa: Option<String>,

    /// Reference state of the solvation energy: gsolv (default), bar1M or reference
    #[arg(long = "solvation-state", value_name = "STATE", help_heading = "Implicit solvent")]
    solvation_state: Option<String>,

    /// g-xTB in solvent: the composite E(g-xTB) + [E(GFN2-xTB + solvent) - E(GFN2-xTB)] (gfn2)
    #[arg(long = "solvation-correction", value_name = "METHOD", help_heading = "Implicit solvent")]
    solvation_correction: Option<String>,

    /// Conventions of the composite's GFN2-xTB halves (default: axtb for g-xTB's axtb,
    /// xtb-6.7.1 for gxtb)
    #[arg(long = "correction-conventions", value_name = "NAME", help_heading = "Implicit solvent")]
    correction_conventions: Option<String>,

    /// Compute the Hessian (and the gradient) with its vibrational analysis
    #[arg(long = "hess", help_heading = "Hessian")]
    hessian: bool,

    /// Hessian method: analytic (default; g-xTB v1.1 numerical), response or numerical
    #[arg(long = "hess-method", value_name = "METHOD", help_heading = "Hessian")]
    hessian_method: Option<String>,

    /// Finite-difference step in bohr (numerical default 0.005; response 5e-4, g-xTB 2.0.1 2e-4)
    #[arg(long = "hess-step", value_name = "H", help_heading = "Hessian")]
    hessian_step: Option<f64>,

    /// SCF accuracy of the displacements (numerical, default 0.01) or of the reference
    #[arg(long = "hess-acc", value_name = "X", help_heading = "Hessian")]
    hessian_accuracy: Option<f64>,

    /// Skip the vibrational analysis of the Hessian
    #[arg(long, help_heading = "Vibrational analysis")]
    novib: bool,

    /// Skip the thermochemistry
    #[arg(long, help_heading = "Vibrational analysis")]
    nothermo: bool,

    /// Analysis conventions: axtb, xtb (6.8.0) or xtb-6.7.1 (default: the calculation's)
    #[arg(long = "analysis-conventions", value_name = "NAME", help_heading = "Vibrational analysis")]
    analysis_conventions: Option<String>,

    /// Temperatures in K (default 298.15; results at the last)
    #[arg(long = "temp", value_name = "T[,T...]", value_delimiter = ',', help_heading = "Vibrational analysis")]
    temperature: Option<Vec<f64>>,

    /// Pressure in Pa (default 101325)
    #[arg(long, value_name = "PA", help_heading = "Vibrational analysis")]
    pressure: Option<f64>,

    /// Rotor cutoff in cm⁻¹ (xtb's sthr, default 50)
    #[arg(long, value_name = "X", help_heading = "Vibrational analysis")]
    sthr: Option<f64>,

    /// Imaginary modes smaller than this in cm⁻¹ are inverted, larger ones left out (xtb's
    /// ithr, default 20)
    #[arg(long, value_name = "X", help_heading = "Vibrational analysis")]
    ithr: Option<f64>,

    /// Frequency scale of the thermochemistry (default 1; GFN-FF in xtb conventions 1.03)
    #[arg(long, value_name = "X", help_heading = "Vibrational analysis")]
    scale: Option<f64>,

    /// Rotational symmetry number (default: from the point group)
    #[arg(long, value_name = "N", help_heading = "Vibrational analysis")]
    symnum: Option<u32>,

    /// Symmetry threshold in bohr (xtb's desy, default 0.1)
    #[arg(long, value_name = "X", help_heading = "Vibrational analysis")]
    desy: Option<f64>,

    /// GFN-FF: set the topology up from this structure (XYZ, same atoms)
    #[arg(long = "topology-ref", value_name = "FILE", help_heading = "GFN-FF")]
    topology_ref: Option<PathBuf>,

    /// GFN-FF: charges of the molecular fragments, in the order of their first atoms
    #[arg(long = "frag-charges", value_name = "Q1,Q2,...", value_delimiter = ',', help_heading = "GFN-FF")]
    fragment_charges: Option<Vec<f64>>,
}

/// The CPU must have what the build assumes (x86-64-v3: AVX2 and FMA, `.cargo/config.toml`).
/// Kept out of line and free of anything but std's feature detection and a static message,
/// so that no AVX instruction runs before it (checked in the disassembly).
#[inline(never)]
fn check_cpu() {
    if let Some(msg) = axtb_driver::cpu_mismatch() {
        let mut err = std::io::stderr();
        let _ = std::io::Write::write_all(&mut err, b"error: ");
        let _ = std::io::Write::write_all(&mut err, msg.to_bytes());
        let _ = std::io::Write::write_all(&mut err, b"\n");
        std::process::exit(1);
    }
}

fn main() -> ExitCode {
    check_cpu();
    run()
}

/// A usage error (exit code 2) with clap's formatting.
fn usage(msg: impl std::fmt::Display) -> ! {
    Cli::command().error(clap::error::ErrorKind::ValueValidation, msg).exit()
}

/// The request of the command line: from stdin with `--json`, else from the structure file
/// and the options.
fn request(mut cli: Cli) -> Result<Request, axtb_driver::Error> {
    if cli.json {
        let mut text = String::new();
        if let Err(e) = std::io::stdin().read_to_string(&mut text) {
            usage(format!("cannot read the request on stdin: {e}"));
        }
        return serde_json::from_str(&text).map_err(|e| axtb_driver::Error::Input(format!("bad request: {e}")));
    }
    // `--molden FILE` takes the next argument, which is the structure when there is no other
    if cli.structure.is_none() {
        if let Some(m) = cli.molden.take_if(|m| m != MOLDEN_FILE) {
            cli.structure = Some(m.into());
            cli.molden = Some(MOLDEN_FILE.into());
        }
    }
    let Some(path) = &cli.structure else { usage("a structure file is required (or --json)") };
    let method = Method::parse(cli.method.as_deref().unwrap_or("gfn2"))?;
    let read = |p: &PathBuf| match std::fs::read_to_string(p) {
        Ok(t) => axtb_driver::read_xyz(&t, method, cli.conventions.as_deref()).map_err(|e| match e {
            axtb_driver::Error::Input(m) => axtb_driver::Error::Input(format!("{}: {m}", p.display())),
            other => other,
        }),
        Err(e) => usage(format!("cannot read {}: {e}", p.display())),
    };
    let (numbers, positions) = read(path)?;
    let topology_positions = match &cli.topology_ref {
        Some(p) => {
            let (n, r) = read(p)?;
            if n != numbers {
                return Err(axtb_driver::Error::Input(format!(
                    "the topology reference {} has other atoms than the structure",
                    p.display()
                )));
            }
            Some(r)
        }
        None => None,
    };
    let molden_file = cli.molden.or_else(|| cli.molden_cartesian.then(|| MOLDEN_FILE.into()));
    Ok(Request {
        numbers,
        positions,
        charge: cli.charge.unwrap_or(0.0),
        uhf: cli.uhf,
        method: cli.method,
        conventions: cli.conventions,
        accuracy: cli.accuracy,
        etemp: cli.etemp,
        max_iter: cli.max_iter,
        threads: cli.threads,
        dispersion: cli.nodisp.then_some(false),
        mixer: cli.mixer,
        scf_trace: cli.scf_trace,
        broken_symmetry: cli.broken_symmetry,
        alpb: cli.alpb,
        gbsa: cli.gbsa,
        solvation_state: cli.solvation_state,
        solvation_correction: cli.solvation_correction,
        correction_conventions: cli.correction_conventions,
        fragment_charges: cli.fragment_charges,
        topology_positions,
        gradient: cli.gradient,
        hessian: cli.hessian,
        hessian_method: cli.hessian_method,
        hessian_step: cli.hessian_step,
        hessian_accuracy: cli.hessian_accuracy,
        molden_file,
        molden_cartesian: cli.molden_cartesian,
        bond_orders: cli.bond_orders,
        vibrations: cli.novib.then_some(false),
        thermo: cli.nothermo.then_some(false),
        analysis_conventions: cli.analysis_conventions,
        temperature: cli.temperature.map(Temperatures::Many),
        pressure: cli.pressure,
        rotor_cutoff: cli.sthr,
        imaginary_cutoff: cli.ithr,
        frequency_scale: cli.scale,
        symmetry_number: cli.symnum,
        symmetry_threshold: cli.desy,
        ..Request::default()
    })
}

/// Report a failed request: as JSON on stdout with `--json`, else on stderr.
fn fail(json: bool, error: &impl serde::Serialize, message: &str) {
    if json {
        println!("{}", serde_json::json!({ "error": error }));
    } else {
        eprintln!("error: {message}");
    }
}

fn run() -> ExitCode {
    let matches = Cli::command().get_matches();
    let cli = Cli::from_arg_matches(&matches).unwrap_or_else(|e| e.exit());
    let json = cli.json;
    if json {
        let command = Cli::command();
        let given: Vec<String> = command
            .get_arguments()
            .filter(|a| {
                a.get_id() != "json" && matches.value_source(a.get_id().as_str()) == Some(ValueSource::CommandLine)
            })
            .map(|a| a.get_long().map_or_else(|| a.get_id().to_string(), |l| format!("--{l}")))
            .collect();
        if !given.is_empty() {
            usage(format!("--json reads everything from the request on stdin; remove {}", given.join(", ")));
        }
    }
    let req = match request(cli) {
        Ok(r) => r,
        Err(e) => {
            fail(json, &e, &e.to_string());
            return ExitCode::from(1);
        }
    };
    if let Ok(method) = Method::parse(req.method.as_deref().unwrap_or("gfn2")) {
        if let Some(w) = axtb_driver::threads_warning(method, req.threads.unwrap_or(1), req.hessian) {
            eprintln!("warning: {w}");
        }
        if let Some(w) = axtb_driver::uhf_warning(method, req.uhf) {
            eprintln!("warning: {w}");
        }
    }
    match std::panic::catch_unwind(|| axtb_driver::run(&req)) {
        Ok(Ok(response)) => {
            if let Some(w) = axtb_driver::eigensolver_warning(&response) {
                eprintln!("warning: {w}");
            }
            match serde_json::to_string(&response) {
                Ok(text) => {
                    println!("{text}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    let e = axtb_driver::Error::Numerical(format!("cannot write the response: {e}"));
                    fail(json, &e, &e.to_string());
                    ExitCode::from(1)
                }
            }
        }
        Ok(Err(e)) => {
            if std::env::var_os("AXTB_LAPACK").is_some_and(|v| !v.is_empty()) {
                if let Some(w) = axtb_driver::lapack_warning() {
                    eprintln!("warning: {w}");
                }
            }
            fail(json, &e, &e.to_string());
            ExitCode::from(1)
        }
        Err(panic) => {
            let what = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_default();
            let message = format!("internal error (a bug in axtb; please report it): {what}");
            fail(json, &serde_json::json!({"kind": "internal", "message": message}), &message);
            ExitCode::from(101)
        }
    }
}
