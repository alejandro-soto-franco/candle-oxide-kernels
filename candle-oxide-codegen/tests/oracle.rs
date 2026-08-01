//! Differential oracle: the standalone MIR builder against the `#[kernel]`
//! frontend.
//!
//! `candle-oxide-kernel` writes the same 30 elementwise entries in Rust and
//! lowers them through cuda-oxide's rustc codegen backend. Its PTX is checked
//! in under `oracle/`, one file per entry, so this test needs no device
//! toolchain and drift shows up as a diff. `oracle/README.md` records the
//! command, the toolchain and the cuda-oxide revision behind those files, and
//! what agreement does and does not establish: the two routes share LLVM and,
//! for every libdevice-backed op, the libdevice bitcode, so a match on
//! `usin`/`ucos`/`uexp`/`ulog` confirms symbol selection, leaving the
//! arithmetic behind those symbols unchecked.
//!
//! `utanh`, `urecip` and the twelve comparison entries have no oracle: Rust
//! routes `f32::tanh` through libm rather than a `core::intrinsics`
//! placeholder, Candle computes the reciprocal in double for both dtypes, and
//! the comparisons' byte-store form has no natural Rust surface. Those rest on
//! the nvcc diff in `abi_diff.rs` and on `binary_bodies.rs`.

use candle_oxide_codegen::build_family;
use candle_oxide_codegen::desc::{all_ops, Family};
use candle_oxide_codegen::testing::entry_body;
use std::collections::BTreeSet;

/// Mnemonic heads that carry floating-point computation. Loads, stores and
/// moves are excluded: they say where a value came from, never what was
/// computed, and their type suffix is spelled differently by different code
/// generators, which would make comparing them noise rather than signal.
const ARITHMETIC: &[&str] = &[
    "abs", "add", "copysign", "cvt", "div", "ex2", "fma", "lg2", "mad", "max", "min", "mul", "neg",
    "rcp", "rsqrt", "selp", "setp", "sqrt", "sub", "testp",
];

/// The floating-point instructions one entry uses, deduplicated and sorted.
///
/// Deduplication is the tolerance this comparison applies. The two frontends
/// emit a different number of copies of the same arithmetic, because the
/// builder unrolls Candle's four layout paths into four loops while the Rust
/// source selects the operand index inside one loop. Ordering and repetition
/// therefore carry no shared meaning; membership does.
fn semantic_ops(ptx: &str, entry: &str) -> Vec<String> {
    entry_body(ptx, entry)
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .map(|mnemonic| mnemonic.trim_end_matches(';'))
        .filter(|mnemonic| {
            let mut parts = mnemonic.split('.');
            let head = parts.next().unwrap_or_default();
            let rest: Vec<&str> = parts.collect();
            ARITHMETIC.contains(&head) && (rest.contains(&"f32") || rest.contains(&"f64"))
        })
        .map(str::to_string)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

#[test]
fn the_builder_agrees_with_the_kernel_frontend() {
    let oracle_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("oracle");
    let mut compared = 0;

    for family in [Family::Binary, Family::Unary] {
        let ops: Vec<_> = all_ops()
            .into_iter()
            .filter(|op| op.family() == family)
            .collect();
        let ours = build_family(family, &ops, "sm_80").unwrap();

        for op in &ops {
            let path = oracle_dir.join(format!("{}.ptx", op.name));
            let Ok(reference) = std::fs::read_to_string(&path) else {
                continue; // No Rust surface, so nvcc is this entry's reference.
            };
            // A truncated oracle file would slice to a short body and could
            // agree by accident, so require the entry to run to its return.
            assert!(
                entry_body(&reference, &op.name).contains("ret;"),
                "{} is truncated in the oracle directory",
                op.name
            );

            let want = semantic_ops(&reference, &op.name);
            let got = semantic_ops(&ours, &op.name);
            let dtype = if op.name.ends_with("_f64") {
                "f64"
            } else {
                "f32"
            };

            match op.name.trim_end_matches("_f32").trim_end_matches("_f64") {
                // The one recorded disagreement, and it is not a defect on
                // either side. Candle's `bmaximum` is `fmaxf`, which nvcc
                // lowers to a single `max.f32` (checked directly against
                // `binary.cu` at sm_80), and the builder matches that through
                // `__nv_fmaxf`. Rust's `f32::max` carries a different NaN and
                // signed-zero contract, so cuda-oxide's #391 deliberately
                // lowers the `maximum_number_nsz` placeholder to an explicit
                // compare and select rather than to libdevice. Both compute
                // IEEE maxNum; they differ only in how a signed zero may be
                // chosen, which Candle does not rely on. The assertions below
                // keep the pair discriminating: a max/min swap on either side
                // still fails.
                stem @ ("bmaximum" | "bminimum") => {
                    let native = if stem == "bmaximum" { "max" } else { "min" };
                    let opposite = if stem == "bmaximum" { "min" } else { "max" };
                    assert_eq!(
                        got,
                        vec![format!("{native}.{dtype}")],
                        "{} should lower to one {native}.{dtype}, as nvcc does",
                        op.name
                    );
                    let predicate = if stem == "bmaximum" { "ge" } else { "le" };
                    let other = if stem == "bmaximum" { "le" } else { "ge" };
                    assert!(
                        want.contains(&format!("setp.{predicate}.{dtype}"))
                            && want.contains(&format!("selp.{dtype}"))
                            && !want.contains(&format!("setp.{other}.{dtype}"))
                            && !want.contains(&format!("{opposite}.{dtype}")),
                        "{} should select on setp.{predicate}.{dtype} in the oracle, found {want:?}",
                        op.name
                    );
                    // The NaN guard carries the whole verdict above. The oracle
                    // tests `setp.nan.{dtype} %p, %r, %r` on each operand before
                    // the compare, so a NaN lhs selects rhs, a NaN rhs leaves
                    // `setp.{predicate}` unordered-false and selects lhs, and two
                    // NaNs return rhs. That is IEEE maxNum, which is what
                    // `fmaxf` and the builder's `{native}.{dtype}` compute. A
                    // bare compare and select would keep every mnemonic asserted
                    // above and still disagree with nvcc on NaN inputs, so the
                    // guard is required explicitly.
                    assert!(
                        want.contains(&format!("setp.nan.{dtype}")),
                        "{} lost its NaN guard in the oracle, so the compare and \
                         select no longer computes maxNum: {want:?}",
                        op.name
                    );
                }
                _ => assert_eq!(got, want, "{} disagrees with the #[kernel] oracle", op.name),
            }
            compared += 1;
        }
    }

    assert!(
        compared >= 30,
        "only {compared} entries had an oracle for the comparison"
    );
}
