//! Warm starts for the golden corpus (`scripts/golden.py`; a development tool, not part of the
//! published package): an SCF started from the previous geometry's converged wavefunction, which
//! one CLI request cannot express.
//!
//! Reads `{"request": {...}, "trajectory": [[[x, y, z], ...], ...]}` on stdin (a CLI request and
//! the geometries in bohr), builds one [`axtb_driver::Calculator`] from the request, runs the
//! request's task at every geometry in order (each SCF starts as the request's `guess` and the
//! method's conventions say: `"previous"` carries the wavefunction from one geometry to the next)
//! and prints one line per geometry: the response as the CLI writes it with `--json`. The first
//! failure prints `{"error": ...}` as the CLI does and ends the run with exit code 1.
//!
//! `cargo build --release -p axtb-driver --example golden_trajectory`

use axtb_driver::{Calculator, Request, Task};
use std::io::Read;
use std::process::ExitCode;

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    request: Request,
    trajectory: Vec<Vec<[f64; 3]>>,
}

fn run(input: &Input) -> axtb_driver::Result<()> {
    let req = &input.request;
    let task = Task::from_request(req)?;
    let mut calc = Calculator::new(req)?;
    for positions in &input.trajectory {
        let response = calc.compute(positions, &task)?;
        let text = serde_json::to_string(&response)
            .map_err(|e| axtb_driver::Error::Numerical(format!("cannot write the response: {e}")))?;
        println!("{text}");
    }
    Ok(())
}

fn main() -> ExitCode {
    let mut text = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut text) {
        eprintln!("cannot read the input on stdin: {e}");
        return ExitCode::from(2);
    }
    let result = serde_json::from_str::<Input>(&text)
        .map_err(|e| axtb_driver::Error::Input(format!("bad input: {e}")))
        .and_then(|input| run(&input));
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            println!("{}", serde_json::json!({ "error": e }));
            ExitCode::from(1)
        }
    }
}
