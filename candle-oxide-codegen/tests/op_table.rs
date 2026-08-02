use candle_oxide_codegen::desc::{all_ops, Arity, DType, Family, FusedChain, OutType};

#[test]
fn the_table_covers_the_v1_surface() {
    let ops = all_ops();
    let names: Vec<&str> = ops.iter().map(|op| op.name.as_str()).collect();

    // 6 binary arithmetic ops x 2 dtypes.
    for stem in ["badd", "bsub", "bmul", "bdiv", "bmaximum", "bminimum"] {
        for suffix in ["f32", "f64"] {
            let want = format!("{stem}_{suffix}");
            assert!(names.contains(&want.as_str()), "missing {want}");
        }
    }
    // 6 comparisons x 2 dtypes, all writing u8.
    for stem in ["eq", "ne", "lt", "le", "gt", "ge"] {
        for suffix in ["f32", "f64"] {
            let want = format!("{stem}_{suffix}");
            let op = ops
                .iter()
                .find(|op| op.name == want)
                .unwrap_or_else(|| panic!("missing {want}"));
            assert_eq!(op.out, OutType::U8, "{want} writes uint8_t");
            assert_eq!(op.arity, Arity::Binary);
        }
    }
    // 11 unary ops x 2 dtypes.
    for stem in [
        "ucopy", "uneg", "usqr", "urecip", "uabs", "usqrt", "uexp", "ulog", "usin", "ucos", "utanh",
    ] {
        for suffix in ["f32", "f64"] {
            let want = format!("{stem}_{suffix}");
            assert!(names.contains(&want.as_str()), "missing {want}");
        }
    }
    // 3 fused chains, f32 only. Each replaces a run of two elementwise
    // kernels, so each carries its own parameter list and its own body
    // rather than a stem crossed with a dtype.
    for (stem, chain) in [
        ("fused_scale_mask", FusedChain::ScaleMask),
        ("fused_bias_gelu_erf", FusedChain::BiasGeluErf),
        ("fused_bias_residual", FusedChain::BiasResidual),
    ] {
        let want = format!("{stem}_f32");
        let op = ops
            .iter()
            .find(|op| op.name == want)
            .unwrap_or_else(|| panic!("missing {want}"));
        assert_eq!(op.arity, Arity::Fused);
        assert_eq!(op.family(), Family::Fused, "{want} ships in fused.ptx");
        assert_eq!(op.chain(), Some(chain));
        assert_eq!(op.dtype, DType::F32, "{want} has no f64 form");
        assert_eq!(op.out, OutType::Same);
        assert!(
            !names.contains(&format!("{stem}_f64").as_str()),
            "{stem} must not ship an f64 form nothing calls"
        );
    }
    // Only the GELU chain calls libdevice, which is what decides whether a
    // fused build links against it.
    assert_eq!(
        ops.iter()
            .filter(|op| op.family() == Family::Fused && op.libdevice_name().is_some())
            .map(|op| op.name.as_str())
            .collect::<Vec<_>>(),
        vec!["fused_bias_gelu_erf_f32"]
    );
    assert_eq!(
        FusedChain::BiasGeluErf.libdevice_name(),
        Some("__nv_normcdff"),
        "any other Gaussian CDF loses bit-exactness against ugelu_erf_f32"
    );

    assert_eq!(
        ops.len(),
        49,
        "12 binary + 12 comparison + 22 unary + 3 fused"
    );
}

#[test]
fn every_name_is_unique() {
    let mut names: Vec<String> = all_ops().into_iter().map(|op| op.name).collect();
    names.sort();
    let before = names.len();
    names.dedup();
    assert_eq!(
        names.len(),
        before,
        "entry-point names collide inside a module"
    );
}
