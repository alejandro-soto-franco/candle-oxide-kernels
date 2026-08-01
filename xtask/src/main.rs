//! Maintainer-only PTX generation. The published crate ships what this
//! writes, so this is never run by a consumer and never by `build.rs`.
//!
//! # Contract
//!
//! Each family's file is written atomically: [`write_family`] writes to a
//! process-unique temporary path and renames it onto the destination, so no
//! reader ever observes a partially written PTX file. This is per-file
//! atomicity only. Building both families before writing either means a
//! build failure leaves every committed artifact untouched, but the two
//! writes themselves are independent: if writing `binary.ptx` succeeds and
//! writing `unary.ptx` then fails, `binary.ptx` is left updated while
//! `unary.ptx` is not. There is no cross-family transaction. A torn pair
//! like that is caught by the reproducibility check a later task adds, not
//! by this tool.

use candle_oxide_codegen::build_family;
use candle_oxide_codegen::desc::{binary_ops, unary_ops, Family};
use cuda_oxide_codegen::experimental::Target;
use std::path::{Path, PathBuf};

/// Write `ptx` to `path` atomically: a process-unique temporary file next
/// to `path`, then a rename onto `path`.
///
/// No reader ever observes a partially written file at `path`, because the
/// rename is the only step that touches the destination name. This says
/// nothing about any other file: a caller writing several files in
/// sequence gets per-file atomicity for each, not an all-or-nothing update
/// across the set. See the module doc comment.
fn write_family(path: &Path, ptx: &str) -> std::io::Result<()> {
    let tmp = path.with_extension(format!("ptx.tmp.{}", std::process::id()));
    std::fs::write(&tmp, ptx)?;
    match std::fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(error) => {
            let _ = std::fs::remove_file(&tmp);
            Err(error)
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_default();
    if command != "gen-ptx" {
        eprintln!("usage: cargo xtask gen-ptx [--arch sm_75] [--out <dir>]");
        std::process::exit(2);
    }

    // sm_75 is the shipped baseline (candle-oxide-kernels/ptx/*.ptx); an
    // unqualified run must reproduce what is committed, not silently drop
    // Turing support by defaulting somewhere else.
    let mut arch = "sm_75".to_string();
    let mut out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../candle-oxide-kernels/ptx");
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--arch" => match args.next() {
                Some(value) => arch = value,
                None => {
                    eprintln!("--arch takes a value");
                    std::process::exit(2);
                }
            },
            "--out" => match args.next() {
                Some(value) => out = PathBuf::from(value),
                None => {
                    eprintln!("--out takes a value");
                    std::process::exit(2);
                }
            },
            other => {
                eprintln!("unknown flag {other}");
                std::process::exit(2);
            }
        }
    }

    // Validated before either family is built, so an unparseable target
    // exits cleanly and never reaches build_family's own panic path.
    if let Err(error) = Target::parse(&arch) {
        eprintln!("invalid arch {arch}: {error}");
        std::process::exit(1);
    }

    // Build both families before writing either, so a failure in the second
    // leaves the first's committed artifact untouched.
    let families = [(Family::Binary, binary_ops()), (Family::Unary, unary_ops())];
    let mut built = Vec::new();
    for (family, ops) in families {
        match build_family(family, &ops, &arch) {
            Ok(ptx) => built.push((family, ptx)),
            Err(error) => {
                eprintln!("generation failed for {family:?} at {arch}: {error}");
                std::process::exit(1);
            }
        }
    }

    std::fs::create_dir_all(&out).expect("the output directory is writable");
    for (family, ptx) in built {
        let path = out.join(family.artifact());
        write_family(&path, &ptx).expect("the artifact is writable");
        println!("wrote {} ({} bytes)", path.display(), ptx.len());
    }
}

#[cfg(test)]
mod tests {
    use super::write_family;

    /// Exercises the case the design exists for, rather than the total
    /// failure a build-time error produces: one write already landed, then
    /// a second write fails. Forces the failure by pre-occupying the
    /// destination with a directory, so `write_family`'s rename step
    /// cannot succeed.
    #[test]
    fn a_later_write_failure_does_not_touch_an_earlier_successful_write() {
        let dir = std::env::temp_dir().join(format!(
            "xtask_write_family_{}_{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let binary = dir.join("binary.ptx");
        write_family(&binary, "BINARY PTX").expect("the first write succeeds");

        let unary = dir.join("unary.ptx");
        // A directory already sits at the destination name, so the rename
        // in write_family cannot land.
        std::fs::create_dir_all(&unary).unwrap();

        let result = write_family(&unary, "UNARY PTX");

        assert!(
            result.is_err(),
            "a write whose destination is occupied cannot silently succeed"
        );
        assert_eq!(
            std::fs::read_to_string(&binary).unwrap(),
            "BINARY PTX",
            "an earlier successful write must survive a later write's failure; \
             write_family gives no cross-file guarantee, only that this one is untouched"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
