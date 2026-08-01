use candle_oxide_codegen::build_family;
use candle_oxide_codegen::desc::{binary_ops, Family};
use candle_oxide_codegen::testing::{assert_ptxas_accepts, entry_body};

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
fn each_arithmetic_entry_lowers_to_its_own_instruction() {
    let ops = binary_ops();
    let ptx = build_family(Family::Binary, &ops, "sm_80").unwrap();
    for (entry, mnemonic) in [
        ("badd_f32", "add.f32"),
        ("bsub_f32", "sub.f32"),
        ("bmul_f32", "mul.f32"),
        ("bdiv_f32", "div.rn.f32"),
        ("badd_f64", "add.f64"),
        ("bsub_f64", "sub.f64"),
        ("bmul_f64", "mul.f64"),
        ("bdiv_f64", "div.rn.f64"),
    ] {
        let body = entry_body(&ptx, entry);
        assert!(body.contains(mnemonic), "{entry} must lower to {mnemonic}:\n{body}");
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
fn bmaximum_and_bminimum_lower_to_distinct_instructions() {
    let ops = binary_ops();
    let ptx = build_family(Family::Binary, &ops, "sm_80").unwrap();
    // The resolved libdevice call surfaces as `max.fN`/`min.fN` after
    // Linking::Libdevice inlines it; verified against the actual PTX rather
    // than assumed.
    for (max_entry, min_entry, max_mnemonic, min_mnemonic) in [
        ("bmaximum_f32", "bminimum_f32", "max.f32", "min.f32"),
        ("bmaximum_f64", "bminimum_f64", "max.f64", "min.f64"),
    ] {
        let max_body = entry_body(&ptx, max_entry);
        let min_body = entry_body(&ptx, min_entry);
        assert!(
            max_body.contains(max_mnemonic) && !max_body.contains(min_mnemonic),
            "{max_entry} must lower to {max_mnemonic}, not {min_mnemonic}:\n{max_body}"
        );
        assert!(
            min_body.contains(min_mnemonic) && !min_body.contains(max_mnemonic),
            "{min_entry} must lower to {min_mnemonic}, not {max_mnemonic}:\n{min_body}"
        );
    }
}

#[test]
fn comparisons_store_one_byte() {
    let stems = ["eq", "ne", "lt", "le", "gt", "ge"];
    let ops: Vec<_> = binary_ops()
        .into_iter()
        .filter(|op| stems.iter().any(|stem| op.name.starts_with(&format!("{stem}_"))))
        .collect();
    assert_eq!(ops.len(), 12, "expected all six comparison kinds at both dtypes");
    let ptx = build_family(Family::Binary, &ops, "sm_80").unwrap();
    // Candle's BINARY_OP_OUT writes `uint8_t`, so the store must be one byte
    // wide. nvcc spells that `st.global.u8` and the LLVM NVPTX backend spells
    // it `st.global.b8`; PTX treats the two identically for `st`.
    for op in &ops {
        let body = entry_body(&ptx, &op.name);
        assert!(
            body.contains("st.global.u8") || body.contains("st.global.b8"),
            "{} must store one byte:\n{body}",
            op.name
        );
    }
    assert_ptxas_accepts(ptx.as_bytes(), "binary_cmp", "sm_80");
}
