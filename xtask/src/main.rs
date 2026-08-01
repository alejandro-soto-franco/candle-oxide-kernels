//! Maintainer-only PTX generation. The published crate ships what this
//! writes, so this is never run by a consumer and never by `build.rs`.

use candle_oxide_codegen::build_family;
use candle_oxide_codegen::desc::{binary_ops, unary_ops, Family};
use cuda_oxide_codegen::experimental::Target;
use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_default();
    if command != "gen-ptx" {
        eprintln!("usage: cargo xtask gen-ptx [--arch sm_80] [--out <dir>]");
        std::process::exit(2);
    }

    let mut arch = "sm_80".to_string();
    let mut out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../candle-oxide-kernels/ptx");
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--arch" => arch = args.next().expect("--arch takes a value"),
            "--out" => out = PathBuf::from(args.next().expect("--out takes a value")),
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
        let tmp = path.with_extension("ptx.tmp");
        std::fs::write(&tmp, &ptx).expect("the artifact is writable");
        std::fs::rename(&tmp, &path).expect("the rename is atomic");
        println!("wrote {} ({} bytes)", path.display(), ptx.len());
    }
}
