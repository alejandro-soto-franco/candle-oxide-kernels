use candle_oxide_codegen::build_family;
use candle_oxide_codegen::desc::{binary_ops, Family};
use candle_oxide_codegen::testing::assert_ptxas_accepts;

fn badd_only() -> Vec<candle_oxide_codegen::desc::OpDesc> {
    binary_ops().into_iter().filter(|op| op.name == "badd_f32").collect()
}

#[test]
fn the_entry_has_four_layout_paths() {
    let ptx = build_family(Family::Binary, &badd_only(), "sm_80").unwrap();
    // Four grid-stride loops means at least four backward branches, one per
    // layout path, plus the two is_contiguous loops.
    let branches = ptx.matches("bra").count();
    assert!(branches >= 6, "expected the four layout paths and two is_contiguous loops, saw {branches} branches:\n{ptx}");
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
