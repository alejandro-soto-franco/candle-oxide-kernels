use candle_oxide_codegen::build_family;
use candle_oxide_codegen::desc::{binary_ops, Family};
use candle_oxide_codegen::testing::assert_ptxas_accepts;

#[test]
fn ptxas_accepts_the_arithmetic_binary_entries() {
    let ops: Vec<_> = binary_ops()
        .into_iter()
        .filter(|op| op.name.starts_with("badd"))
        .collect();
    let ptx = build_family(Family::Binary, &ops, "sm_80").unwrap();
    assert_ptxas_accepts(ptx.as_bytes(), "binary_badd", "sm_80");
}

#[test]
fn ptxas_accepts_at_the_deployment_arch_too() {
    let ops: Vec<_> = binary_ops()
        .into_iter()
        .filter(|op| op.name.starts_with("badd"))
        .collect();
    let ptx = build_family(Family::Binary, &ops, "sm_80").unwrap();
    // The baseline artifact is JIT'd upward by the driver, so sm_120 must
    // also accept the sm_80 PTX.
    assert_ptxas_accepts(ptx.as_bytes(), "binary_badd_sm120", "sm_120");
}
