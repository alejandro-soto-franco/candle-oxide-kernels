use candle_oxide_codegen::build_family;
use candle_oxide_codegen::desc::{binary_ops, Family};
use candle_oxide_codegen::testing::assert_ptxas_accepts;

/// A SINGLE-entry module. Every assertion below uses a bare `ptx.contains`
/// or a bare count, which is sound only because the module holds one entry.
/// Widening this filter silently turns those assertions module-scoped, the
/// defect Task 4 fixed by introducing `testing::entry_body`.
fn badd_only() -> Vec<candle_oxide_codegen::desc::OpDesc> {
    binary_ops()
        .into_iter()
        .filter(|op| op.name == "badd_f32")
        .collect()
}

#[test]
fn each_strided_path_decomposes_its_own_index() {
    let ptx = build_family(Family::Binary, &badd_only(), "sm_80").unwrap();
    // Every strided operand walks the dimensions once, dividing the running
    // index by `dims[dim_idx]` per step. Counting those divisions counts the
    // strided operands across the four layout paths:
    //
    //   both contiguous       0
    //   lhs contiguous only   1  (the rhs is decomposed)
    //   rhs contiguous only   1  (the lhs is decomposed)
    //   neither contiguous    2  (both are decomposed)
    //
    // The remainder never appears as `rem.u32`: LLVM strength-reduces
    // `idx % dim` into this same `div.u32` followed by `mul.lo.s32` and
    // `sub.s32`, so the division is the site that survives.
    //
    // Verified by mutation: forcing all four paths contiguous drops the count
    // to 0. The count is invariant under swapping the two single-strided
    // paths, so it does not pin which operand each path decomposes; that
    // pairing is covered by the GPU parity comparison.
    let divisions = ptx.matches("div.u32").count();
    assert_eq!(
        divisions, 4,
        "expected one index decomposition per strided operand across the four \
         layout paths, saw {divisions}:\n{ptx}"
    );
}

#[test]
fn the_null_strides_pointer_is_tested() {
    let ptx = build_family(Family::Binary, &badd_only(), "sm_80").unwrap();
    // `dims_and_strides == nullptr` selects the contiguous path without
    // dereferencing, so the parameter is compared against zero.
    assert!(
        ptx.contains("setp.eq.s64") || ptx.contains("setp.eq.b64") || ptx.contains("setp.ne.s64"),
        "the null check on dims_and_strides is missing:\n{ptx}"
    );
}

#[test]
fn the_strided_module_still_compiles_and_assembles() {
    let ops = binary_ops();
    let ptx = build_family(Family::Binary, &ops, "sm_80").unwrap();
    assert_ptxas_accepts(ptx.as_bytes(), "binary_strided_all", "sm_80");
}
