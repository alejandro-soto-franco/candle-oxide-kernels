use candle_oxide_codegen::build_family;
use candle_oxide_codegen::desc::{binary_ops, Family};
use candle_oxide_codegen::testing::assert_ptxas_accepts;

#[test]
fn every_binary_entry_compiles_into_one_module() {
    let ops = binary_ops();
    let ptx = build_family(Family::Binary, &ops, "sm_80").unwrap();
    for op in &ops {
        assert!(
            ptx.contains(&format!(".visible .entry {}(", op.name)),
            "{} missing from the binary module",
            op.name
        );
    }
    assert_ptxas_accepts(ptx.as_bytes(), "binary_all", "sm_80");
}

#[test]
fn arithmetic_lowers_to_the_expected_instructions() {
    let ops = binary_ops();
    let ptx = build_family(Family::Binary, &ops, "sm_80").unwrap();
    for want in ["add.f32", "sub.f32", "mul.f32", "div.rn.f32", "add.f64", "div.rn.f64"] {
        assert!(ptx.contains(want), "{want} missing:\n{ptx}");
    }
}

#[test]
fn libdevice_min_and_max_resolved_at_link_time() {
    let ops: Vec<_> = binary_ops()
        .into_iter()
        .filter(|op| op.name.starts_with("bmaximum") || op.name.starts_with("bminimum"))
        .collect();
    let ptx = build_family(Family::Binary, &ops, "sm_80").unwrap();
    // Linking::Libdevice resolves __nv_* at the IR level before llc runs, so
    // no libdevice symbol survives as an unresolved external declaration.
    let unresolved: Vec<&str> = ptx
        .lines()
        .filter(|line| line.contains(".extern .func") && line.contains("__nv_"))
        .collect();
    assert!(
        unresolved.is_empty(),
        "libdevice symbols survived to the artifact: {unresolved:?}\n{ptx}"
    );
    assert_ptxas_accepts(ptx.as_bytes(), "binary_minmax", "sm_80");
}

#[test]
fn comparisons_store_one_byte() {
    let ops: Vec<_> = binary_ops()
        .into_iter()
        .filter(|op| op.name.starts_with("lt_") || op.name.starts_with("ge_"))
        .collect();
    let ptx = build_family(Family::Binary, &ops, "sm_80").unwrap();
    // Candle's BINARY_OP_OUT writes `uint8_t`, so the store must be one byte
    // wide. nvcc spells that `st.global.u8` and the LLVM NVPTX backend spells
    // it `st.global.b8`; PTX treats the two identically for `st`.
    assert!(
        ptx.contains("st.global.u8") || ptx.contains("st.global.b8"),
        "comparison writes a byte:\n{ptx}"
    );
    assert_ptxas_accepts(ptx.as_bytes(), "binary_cmp", "sm_80");
}
