//! Structural assertions on the fused family: the ABI each entry declares,
//! the libdevice symbol the GELU chain resolves to, and the FMA contraction
//! gate.
//!
//! The consumer side is written in a separate repository against the same
//! interface, so a parameter word that moves here breaks a caller that
//! cannot be changed in the same commit. These assertions are the ABI's
//! only mechanical check: the fused entries have no nvcc counterpart, so
//! `abi_diff.rs` has nothing to compare them against.

use candle_oxide_codegen::build_family;
use candle_oxide_codegen::desc::{fused_ops, Family};
use candle_oxide_codegen::testing::{entry_body, entry_params};

/// The whole family, built at the architecture the other builder tests use.
fn fused_ptx() -> String {
    build_family(Family::Fused, &fused_ops(), "sm_80").unwrap()
}

/// The declaration shape of each parameter, with the identifier dropped.
///
/// Only the shape is an ABI concern; the parameter's name is a spelling the
/// backend chooses.
fn param_shapes(ptx: &str, entry: &str) -> Vec<String> {
    entry_params(ptx, entry)
        .iter()
        .map(|line| {
            let words: Vec<&str> = line.split_whitespace().collect();
            words[..words.len() - 1].join(" ")
        })
        .collect()
}

/// A `const T *` or `T *` parameter, as `MirPtrType::get_global` stamps it.
const PTR: &str = ".param .u64 .ptr .global .align 1";
/// A `size_t` parameter.
const WORD: &str = ".param .u64";

#[test]
fn the_family_holds_exactly_the_three_chains() {
    let ptx = fused_ptx();
    for name in [
        "fused_scale_mask_f32",
        "fused_bias_gelu_erf_f32",
        "fused_bias_residual_f32",
    ] {
        assert!(
            ptx.contains(&format!(".visible .entry {name}(")),
            "{name} missing:\n{ptx}"
        );
    }
    assert_eq!(
        ptx.matches(".visible .entry ").count(),
        3,
        "the fused family declares three entries and no others"
    );
}

#[test]
fn scale_mask_declares_seven_words_with_scale_last() {
    let ptx = fused_ptx();
    assert_eq!(
        param_shapes(&ptx, "fused_scale_mask_f32"),
        vec![
            WORD,          // numel
            WORD,          // hss
            WORD,          // ss
            PTR,           // x
            PTR,           // mask
            PTR,           // out
            ".param .f32", // scale
        ],
        "fused_scale_mask_f32 ABI:\n{ptx}"
    );
}

#[test]
fn bias_gelu_erf_declares_five_words() {
    let ptx = fused_ptx();
    assert_eq!(
        param_shapes(&ptx, "fused_bias_gelu_erf_f32"),
        vec![
            WORD, // numel
            WORD, // d
            PTR,  // x
            PTR,  // bias
            PTR,  // out
        ],
        "fused_bias_gelu_erf_f32 ABI:\n{ptx}"
    );
}

#[test]
fn bias_residual_declares_six_words() {
    let ptx = fused_ptx();
    assert_eq!(
        param_shapes(&ptx, "fused_bias_residual_f32"),
        vec![
            WORD, // numel
            WORD, // d
            PTR,  // x
            PTR,  // bias
            PTR,  // residual
            PTR,  // out
        ],
        "fused_bias_residual_f32 ABI:\n{ptx}"
    );
}

/// libdevice is linked and its bodies are inlined, so the symbol survives
/// into the entry as the label the inliner names the merged block after,
/// `%__nv_normcdff.exit`. That is what witnesses the call: a formulation
/// written out by hand, or a different libdevice symbol, leaves no such
/// label. Any other route to the Gaussian CDF loses bit-exactness against
/// `ugelu_erf_f32`, which reaches `normcdff` through `normcdfg`
/// (`cuda_utils.cuh:140`).
#[test]
fn the_gelu_chain_resolves_normcdff() {
    let ptx = fused_ptx();
    let body = entry_body(&ptx, "fused_bias_gelu_erf_f32");
    assert!(
        body.contains("__nv_normcdff"),
        "fused_bias_gelu_erf_f32 does not reach __nv_normcdff:\n{body}"
    );
}

/// The contraction gate.
///
/// `fused_scale_mask_f32` replaces `affine_f32` followed by `badd_f32`,
/// which rounds the product to memory before it adds the mask. Contracting
/// the multiply and the add into one FMA drops that rounding and gives a
/// more accurate answer than the path being replaced, so the fused kernel
/// stops being bit-exact. This assertion is what catches a future change to
/// `Family::fma_contraction`.
#[test]
fn scale_mask_keeps_the_multiply_and_the_add_apart() {
    let ptx = fused_ptx();
    let body = entry_body(&ptx, "fused_scale_mask_f32");
    assert!(
        !body.contains("fma.rn.f32"),
        "fused_scale_mask_f32 contracted x * scale + mask into an FMA, \
         which rounds once where the reference rounds twice:\n{body}"
    );
    // The pair the FMA would have replaced must still be there, or the
    // assertion above passes on a kernel that computes something else.
    assert!(
        body.contains("mul.rn.f32"),
        "fused_scale_mask_f32 lost its multiply:\n{body}"
    );
    assert!(
        body.contains("add.rn.f32"),
        "fused_scale_mask_f32 lost its mask add:\n{body}"
    );
}

/// The setting behind the assertion above, pinned directly.
///
/// The elementwise families cannot witness their own contraction setting in
/// PTX: no elementwise body pairs a floating-point multiply with an add, so
/// there is no contraction site for the flag to act on and flipping it
/// changes nothing in `binary.ptx` or `unary.ptx`. Asserting the flag here
/// is what keeps a change to the fused family from taking them with it.
#[test]
fn only_the_fused_family_switches_contraction_off() {
    assert!(Family::Binary.fma_contraction());
    assert!(Family::Unary.fma_contraction());
    assert!(!Family::Fused.fma_contraction());
}

/// A fused set without the GELU chain names no libdevice symbol, so it
/// compiles self-contained. This is the first op set in the workspace to
/// reach that path in `build_family`.
#[test]
fn a_chain_set_without_the_gelu_entry_needs_no_libdevice() {
    let ops: Vec<_> = fused_ops()
        .into_iter()
        .filter(|op| op.libdevice_name().is_none())
        .collect();
    assert_eq!(ops.len(), 2, "only the GELU chain calls libdevice");
    let ptx = build_family(Family::Fused, &ops, "sm_80").unwrap();
    assert!(ptx.contains(".visible .entry fused_scale_mask_f32("));
    assert!(ptx.contains(".visible .entry fused_bias_residual_f32("));
}
