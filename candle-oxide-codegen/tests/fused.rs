//! Structural assertions on the fused family: the ABI each entry declares,
//! the libdevice symbol the GELU entry resolves to, the FMA contraction gate
//! and the division gate.
//!
//! The consumer side is written in a separate repository against the same
//! interface, so a parameter word that moves here breaks a caller that
//! cannot be changed in the same commit. These assertions are the ABI's
//! only mechanical check: the fused entries have no nvcc counterpart, so
//! `abi_diff.rs` has nothing to compare them against.

use candle_oxide_codegen::build_family;
use candle_oxide_codegen::desc::{fused_ops, Family};
use candle_oxide_codegen::testing::{entry_body, entry_params};

/// Every entry the family declares, read from the op table rather than
/// listed here, so an entry added later is covered by the gates below
/// without an edit.
fn fused_names() -> Vec<String> {
    fused_ops().into_iter().map(|op| op.name).collect()
}

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
fn the_family_holds_exactly_the_entries_the_table_names() {
    let ptx = fused_ptx();
    let names = fused_names();
    assert_eq!(
        names,
        vec![
            "fused_scale_mask_f32",
            "fused_bias_gelu_erf_f32",
            "fused_bias_residual_f32",
            "fused_bias_f32",
            "fused_permute12_f32",
        ],
        "the family's entry names are part of the interface"
    );
    for name in &names {
        assert!(
            ptx.contains(&format!(".visible .entry {name}(")),
            "{name} missing:\n{ptx}"
        );
    }
    assert_eq!(
        ptx.matches(".visible .entry ").count(),
        names.len(),
        "the fused family declares the table's entries and no others"
    );
}

#[test]
fn scale_mask_declares_five_words_with_scale_last() {
    let ptx = fused_ptx();
    assert_eq!(
        param_shapes(&ptx, "fused_scale_mask_f32"),
        vec![
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
fn bias_gelu_erf_declares_four_words() {
    let ptx = fused_ptx();
    assert_eq!(
        param_shapes(&ptx, "fused_bias_gelu_erf_f32"),
        vec![
            WORD, // d
            PTR,  // x
            PTR,  // bias
            PTR,  // out
        ],
        "fused_bias_gelu_erf_f32 ABI:\n{ptx}"
    );
}

#[test]
fn bias_residual_declares_five_words() {
    let ptx = fused_ptx();
    assert_eq!(
        param_shapes(&ptx, "fused_bias_residual_f32"),
        vec![
            WORD, // d
            PTR,  // x
            PTR,  // bias
            PTR,  // residual
            PTR,  // out
        ],
        "fused_bias_residual_f32 ABI:\n{ptx}"
    );
}

#[test]
fn bias_declares_four_words() {
    let ptx = fused_ptx();
    assert_eq!(
        param_shapes(&ptx, "fused_bias_f32"),
        vec![
            WORD, // d
            PTR,  // x
            PTR,  // bias
            PTR,  // out
        ],
        "fused_bias_f32 ABI:\n{ptx}"
    );
}

#[test]
fn permute12_declares_three_words() {
    let ptx = fused_ptx();
    assert_eq!(
        param_shapes(&ptx, "fused_permute12_f32"),
        vec![
            WORD, // d
            PTR,  // src
            PTR,  // out
        ],
        "fused_permute12_f32 ABI:\n{ptx}"
    );
}

/// The permutation is entirely in the two row bases, so the loop body loads
/// and stores and computes nothing else. A floating-point instruction in
/// there would mean the copy stopped being a copy.
#[test]
fn permute12_only_copies() {
    let ptx = fused_ptx();
    let body = entry_body(&ptx, "fused_permute12_f32");
    for arithmetic in [".f32", "fma.rn", "mul.rn", "add.rn"] {
        assert_eq!(
            body.matches(arithmetic).count(),
            0,
            "fused_permute12_f32 carries a {arithmetic}; a copy computes \
             nothing:\n{body}"
        );
    }
    assert_eq!(
        body.matches("ld.global").count(),
        1,
        "fused_permute12_f32 loads once an element:\n{body}"
    );
    assert_eq!(
        body.matches("st.global").count(),
        1,
        "fused_permute12_f32 stores once an element:\n{body}"
    );
}

/// `fused_bias_f32` is the GELU entry with the activation removed, so its
/// body is the same two loads and one add, and nothing more.
#[test]
fn bias_adds_once_and_calls_nothing() {
    let ptx = fused_ptx();
    let body = entry_body(&ptx, "fused_bias_f32");
    assert_eq!(
        body.matches("add.rn.f32").count(),
        1,
        "fused_bias_f32 declares one add:\n{body}"
    );
    assert_eq!(
        body.matches("mul.rn.f32").count(),
        0,
        "fused_bias_f32 has no activation to multiply by:\n{body}"
    );
    assert!(
        !body.contains("__nv_"),
        "fused_bias_f32 calls no libdevice symbol:\n{body}"
    );
}

/// The division gate.
///
/// The launch is shaped like the tensor so that every index term comes from
/// a special register. Recovering a row and a column from a flat counter
/// instead needs `i / hss`, `i % ss` and `i % d`, and a division by a
/// runtime value expands to roughly twenty instructions, which is what held
/// the two chains moving eight bytes an element to 46 per cent of the device
/// bandwidth ceiling while the one moving twelve reached 74. An index
/// expression that reintroduces a division gives that back silently, so it
/// fails here instead.
#[test]
fn no_fused_entry_divides() {
    let ptx = fused_ptx();
    for name in fused_names() {
        let body = entry_body(&ptx, &name);
        for divide in ["div.u32", "rem.u32", "div.u64", "rem.u64"] {
            assert_eq!(
                body.matches(divide).count(),
                0,
                "{name} carries a {divide}; the grid is shaped like the tensor \
                 so no index term needs one:\n{body}"
            );
        }
    }
}

/// The grid coordinates each entry reads, which is the other half of the
/// division gate: an entry could drop its divisions by taking the row as a
/// parameter again, and this pins that the row comes from the launch.
#[test]
fn each_chain_reads_the_grid_coordinates_its_geometry_names() {
    let ptx = fused_ptx();

    // `(ceil(ss / block), H, B)`: the batch index is `ctaid.z` and the head
    // count is `nctaid.y`, which is what replaced the `hss` parameter.
    let scale_mask = entry_body(&ptx, "fused_scale_mask_f32");
    for sreg in ["%ctaid.x", "%ctaid.y", "%ctaid.z", "%nctaid.y"] {
        assert!(
            scale_mask.contains(sreg),
            "fused_scale_mask_f32 does not read {sreg}:\n{scale_mask}"
        );
    }

    // `(D2, D1, B)`: both permuted dimensions are grid extents, so the
    // entry reads `nctaid.x` for `D2` as well as `nctaid.y` for `D1`, and
    // neither is a parameter.
    let permute = entry_body(&ptx, "fused_permute12_f32");
    for sreg in ["%ctaid.x", "%ctaid.y", "%ctaid.z", "%nctaid.x", "%nctaid.y"] {
        assert!(
            permute.contains(sreg),
            "fused_permute12_f32 does not read {sreg}:\n{permute}"
        );
    }

    // `(ceil(d / block), rows, 1)`: one row per `ctaid.y`, and the z
    // dimension is unused, so reading it would mean the geometry drifted.
    for name in [
        "fused_bias_gelu_erf_f32",
        "fused_bias_residual_f32",
        "fused_bias_f32",
    ] {
        let body = entry_body(&ptx, name);
        for sreg in ["%ctaid.x", "%ctaid.y"] {
            assert!(body.contains(sreg), "{name} does not read {sreg}:\n{body}");
        }
        assert!(
            !body.contains("%ctaid.z"),
            "{name} reads %ctaid.z, but its grid is one deep:\n{body}"
        );
    }
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

/// The contraction gate, and the signed-zero gate alongside it.
///
/// `fused_scale_mask_f32` replaces `affine_f32` followed by `badd_f32`.
/// That path rounds the product to memory before it adds the mask, so
/// contracting the multiply and the add into one FMA drops a rounding and
/// gives a more accurate answer than the path being replaced, and the fused
/// kernel stops being bit-exact. This catches a future change to
/// `Family::fma_contraction`.
///
/// The instruction counts are pinned rather than only their presence.
/// `affine_f32` computes `x * mul + add` with `add` zero, so the entry adds
/// twice: once for that zero, which turns a `-0.0` product into `+0.0`, and
/// once for the mask. `x + 0.0 -> x` is an invalid fold in IEEE without
/// `nsz`, so both survive today; a fast-math or `nsz` flag arriving later
/// would delete the first one and drift the entry off bit-exactness with no
/// other outward sign. A count is what surfaces that.
#[test]
fn scale_mask_rounds_the_multiply_the_zero_add_and_the_mask_add_apart() {
    let ptx = fused_ptx();
    let body = entry_body(&ptx, "fused_scale_mask_f32");
    assert_eq!(
        body.matches("fma.rn.f32").count(),
        0,
        "fused_scale_mask_f32 contracted a multiply and an add into an FMA, \
         which rounds once where the reference rounds twice:\n{body}"
    );
    assert_eq!(
        body.matches("mul.rn.f32").count(),
        1,
        "fused_scale_mask_f32 declares exactly one multiply, `x * scale`:\n{body}"
    );
    assert_eq!(
        body.matches("add.rn.f32").count(),
        2,
        "fused_scale_mask_f32 declares two adds, `+ 0.0f` from affine_f32 and \
         then the mask; a count of one means the zero add was folded away and \
         a `-0.0` product now survives into the mask add:\n{body}"
    );
    // The zero must be the literal `affine_f32` carries, so the entry cannot
    // pass the count above with an add of something else.
    assert!(
        body.contains("0f00000000"),
        "fused_scale_mask_f32 lost affine_f32's `+ 0.0f` operand:\n{body}"
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

/// A fused set without the GELU entry names no libdevice symbol, so it
/// compiles self-contained. This is the first op set in the workspace to
/// reach that path in `build_family`.
#[test]
fn a_fused_set_without_the_gelu_entry_needs_no_libdevice() {
    let ops: Vec<_> = fused_ops()
        .into_iter()
        .filter(|op| op.libdevice_name().is_none())
        .collect();
    assert_eq!(ops.len(), 4, "only the GELU entry calls libdevice");
    let ptx = build_family(Family::Fused, &ops, "sm_80").unwrap();
    for op in &ops {
        assert!(
            ptx.contains(&format!(".visible .entry {}(", op.name)),
            "{} missing from the self-contained build:\n{ptx}",
            op.name
        );
    }
}
