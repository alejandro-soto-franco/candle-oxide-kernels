use candle_oxide_codegen::build_family;
use candle_oxide_codegen::desc::{binary_ops, Family};
use candle_oxide_codegen::testing::{assert_ptxas_accepts, declared_target};

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

/// The tests above assemble PTX this crate builds fresh at `sm_80`. Neither
/// they nor `candle-oxide-kernels/tests/artifacts.rs` (which only checks for
/// `.entry` strings) ever runs `ptxas` over the PTX actually committed under
/// `candle-oxide-kernels/ptx/`. This assembles the shipped files themselves,
/// both at the architecture they declare and at `sm_120`, the
/// forward-compatibility path the driver relies on.
#[test]
fn ptxas_accepts_the_committed_artifacts() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../candle-oxide-kernels/ptx");
    for name in ["binary.ptx", "unary.ptx"] {
        let ptx = std::fs::read_to_string(root.join(name)).unwrap();
        let arch = declared_target(&ptx).to_string();
        assert_ptxas_accepts(ptx.as_bytes(), &format!("committed_{name}_own_arch"), &arch);
        assert_ptxas_accepts(ptx.as_bytes(), &format!("committed_{name}_sm120"), "sm_120");
    }
}
