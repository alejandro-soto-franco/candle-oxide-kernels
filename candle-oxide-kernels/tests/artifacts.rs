use candle_oxide_kernels as k;

#[test]
fn the_artifacts_are_present_and_non_empty() {
    assert!(k::BINARY.len() > 1000, "binary.ptx looks truncated");
    assert!(k::UNARY.len() > 1000, "unary.ptx looks truncated");
    assert!(k::FUSED.len() > 1000, "fused.ptx looks truncated");
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
    for name in k::FUSED_ENTRIES {
        assert!(
            k::FUSED.contains(&format!(".entry {name}(")),
            "{name} missing from fused.ptx"
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

    // Count entries in fused.ptx
    let fused_count = k::FUSED.matches(".visible .entry ").count();
    assert_eq!(
        fused_count,
        k::FUSED_ENTRIES.len(),
        "number of entries in fused.ptx does not match FUSED_ENTRIES"
    );
}

#[test]
fn the_module_names_are_namespaced() {
    assert_eq!(k::BINARY_MODULE, "oxide::binary");
    assert_eq!(k::UNARY_MODULE, "oxide::unary");
    assert_eq!(k::FUSED_MODULE, "oxide::fused");
    assert!(k::has_entry(k::BINARY_MODULE, "badd_f32"));
    assert!(k::has_entry(k::UNARY_MODULE, "usqrt_f64"));
    assert!(k::has_entry(k::FUSED_MODULE, "fused_scale_mask_f32"));
    assert!(
        !k::has_entry(k::UNARY_MODULE, "badd_f32"),
        "families do not overlap"
    );
    assert!(
        !k::has_entry(k::BINARY_MODULE, "fused_bias_residual_f32"),
        "families do not overlap"
    );
}

/// The body of one entry, from its `.visible .entry` line to the closing
/// brace at column zero.
fn entry_body<'a>(ptx: &'a str, entry: &str) -> &'a str {
    let needle = format!(".visible .entry {entry}(");
    let start = ptx
        .find(&needle)
        .unwrap_or_else(|| panic!("{entry} is absent from the artifact"));
    let rest = &ptx[start..];
    let end = rest
        .find("\n}")
        .map(|at| start + at + 2)
        .unwrap_or(ptx.len());
    &ptx[start..end]
}

/// The contraction gate, applied to the bytes this crate actually ships.
///
/// `candle-oxide-codegen/tests/fused.rs` asserts the same thing on PTX built
/// fresh at `sm_80`. This one reads `fused.ptx` as committed, needs no CUDA
/// toolkit and no LLVM, and so runs in the job that proves the published
/// crate stands alone. `fused_scale_mask_f32` replaces `affine_f32` followed
/// by `badd_f32`, which rounds the product to memory before it adds the
/// mask, so a contracted multiply-add rounds once where the reference rounds
/// twice and the entry stops being bit-exact.
#[test]
fn the_shipped_scale_mask_entry_carries_no_contracted_multiply_add() {
    let body = entry_body(k::FUSED, "fused_scale_mask_f32");
    assert!(
        !body.contains("fma.rn.f32"),
        "fused_scale_mask_f32 ships a contracted multiply-add:\n{body}"
    );
    assert!(body.contains("mul.rn.f32"), "the multiply is gone:\n{body}");
    assert!(body.contains("add.rn.f32"), "the mask add is gone:\n{body}");
}

/// Routing is asserted by content rather than by identity: two `&'static
/// str` consts holding the same literal can share one allocation, so a
/// pointer comparison would pass on a `ptx_for` that returned the wrong
/// family's PTX. Each artifact is identified here by the entries it holds
/// and by the entries it must not.
#[test]
fn every_module_key_resolves_to_its_own_ptx() {
    let families = [
        (k::BINARY_MODULE, k::BINARY_ENTRIES),
        (k::UNARY_MODULE, k::UNARY_ENTRIES),
        (k::FUSED_MODULE, k::FUSED_ENTRIES),
    ];
    for (module, entries) in families {
        let ptx = k::ptx_for(module).unwrap_or_else(|| panic!("{module} resolves to no PTX"));
        for (other, other_entries) in families {
            for name in other_entries {
                let present = ptx.contains(&format!(".entry {name}("));
                assert_eq!(
                    present,
                    other == module,
                    "{module} resolved to PTX that {} {name}, which belongs to {other}",
                    if present { "holds" } else { "lacks" }
                );
            }
        }
        assert_eq!(entries.len(), ptx.matches(".visible .entry ").count());
    }
    assert!(k::ptx_for("oxide::nothing").is_none());
    assert!(!k::has_entry("oxide::nothing", "fused_scale_mask_f32"));
}
