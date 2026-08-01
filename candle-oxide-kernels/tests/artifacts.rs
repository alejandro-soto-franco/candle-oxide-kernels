use candle_oxide_kernels as k;

#[test]
fn the_artifacts_are_present_and_non_empty() {
    assert!(k::BINARY.len() > 1000, "binary.ptx looks truncated");
    assert!(k::UNARY.len() > 1000, "unary.ptx looks truncated");
}

#[test]
fn every_declared_entry_is_in_its_artifact() {
    for name in k::BINARY_ENTRIES {
        assert!(
            k::BINARY.contains(&format!(".entry {name}(")),
            "{name} missing from binary.ptx"
        );
    }
    for name in k::UNARY_ENTRIES {
        assert!(
            k::UNARY.contains(&format!(".entry {name}(")),
            "{name} missing from unary.ptx"
        );
    }
}

#[test]
fn every_artifact_entry_is_declared() {
    // Count entries in binary.ptx
    let binary_count = k::BINARY.matches(".visible .entry ").count();
    assert_eq!(
        binary_count,
        k::BINARY_ENTRIES.len(),
        "number of entries in binary.ptx does not match BINARY_ENTRIES"
    );

    // Count entries in unary.ptx
    let unary_count = k::UNARY.matches(".visible .entry ").count();
    assert_eq!(
        unary_count,
        k::UNARY_ENTRIES.len(),
        "number of entries in unary.ptx does not match UNARY_ENTRIES"
    );
}

#[test]
fn the_module_names_are_namespaced() {
    assert_eq!(k::BINARY_MODULE, "oxide::binary");
    assert_eq!(k::UNARY_MODULE, "oxide::unary");
    assert!(k::has_entry(k::BINARY_MODULE, "badd_f32"));
    assert!(k::has_entry(k::UNARY_MODULE, "usqrt_f64"));
    assert!(
        !k::has_entry(k::UNARY_MODULE, "badd_f32"),
        "families do not overlap"
    );
}
