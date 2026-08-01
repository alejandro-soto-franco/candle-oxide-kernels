use candle_oxide_codegen::build_family;
use candle_oxide_codegen::desc::{all_ops, Family};
use candle_oxide_codegen::testing::{entry_params, nvcc_ptx_for};

/// Candle's kernel sources, the reference this crate replaces.
fn candle_kernels_src() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("CANDLE_KERNELS_SRC").unwrap_or_else(|_| {
        format!("{}/candle/candle-kernels/src", std::env::var("HOME").unwrap())
    }))
}

#[test]
fn every_entry_matches_nvcc_parameter_shape() {
    let src = candle_kernels_src();
    let reference = [
        (Family::Binary, src.join("binary.cu")),
        (Family::Unary, src.join("unary.cu")),
    ];

    for (family, cu) in reference {
        let nvcc_ptx = nvcc_ptx_for(&cu, &src, "sm_80");
        let ops: Vec<_> = all_ops().into_iter().filter(|op| op.family() == family).collect();
        let ours = build_family(family, &ops, "sm_80").unwrap();

        for op in &ops {
            let want = entry_params(&nvcc_ptx, &op.name);
            assert!(!want.is_empty(), "nvcc did not emit {}", op.name);
            let got = entry_params(&ours, &op.name);
            assert_eq!(
                got.len(),
                want.len(),
                "{} takes {} words from nvcc and {} from the builder",
                op.name,
                want.len(),
                got.len()
            );
            for (index, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                // Drop the trailing identifier from each side; only the
                // declaration's shape (type, address space, alignment) is
                // an ABI concern, never the parameter's name.
                let got_shape: Vec<&str> = g.split_whitespace().collect();
                let want_shape: Vec<&str> = w.split_whitespace().collect();
                let got_shape = &got_shape[..got_shape.len() - 1];
                let want_shape = &want_shape[..want_shape.len() - 1];

                if let Some(rest) = got_shape.strip_prefix([".param", ".u64", ".ptr"].as_slice()) {
                    // The one accepted divergence: MirPtrType::get_global
                    // stamps every pointer parameter with an explicit
                    // address space and pointee alignment; nvcc's frontend
                    // does not carry either into the .param declaration.
                    // Every pointer cudarc passes is a global allocation of
                    // natural alignment, so this is an address-space and
                    // optimiser-hint annotation, not a calling-convention
                    // change: the word underneath is still one .u64.
                    assert_eq!(
                        rest.first().copied(),
                        Some(".global"),
                        "{} word {index}: unexpected pointer annotation {got_shape:?}",
                        op.name
                    );
                    assert_eq!(
                        rest.get(1).copied(),
                        Some(".align"),
                        "{} word {index}: unexpected pointer annotation {got_shape:?}",
                        op.name
                    );
                    assert!(
                        rest.get(2).is_some_and(|n| n.parse::<u32>().is_ok()),
                        "{} word {index}: `.align` was not followed by a number: {got_shape:?}",
                        op.name
                    );
                    assert_eq!(
                        rest.len(),
                        3,
                        "{} word {index}: unexpected trailing tokens after `.align N`: {got_shape:?}",
                        op.name
                    );
                    // The divergence is a checked fact, not prose: nvcc must
                    // not have emitted `.ptr` for the corresponding word, or
                    // this is no longer the known, accepted divergence.
                    assert!(
                        !want_shape.contains(&".ptr"),
                        "{} word {index}: nvcc unexpectedly emitted a `.ptr` annotation too: {want_shape:?}",
                        op.name
                    );
                    assert_eq!(
                        &got_shape[..2],
                        want_shape,
                        "{} word {index}: builder {got_shape:?} against nvcc {want_shape:?}",
                        op.name
                    );
                } else {
                    // No pointer annotation on our side: nothing to
                    // normalise, the two declarations must agree exactly.
                    assert_eq!(
                        got_shape,
                        want_shape,
                        "{} word {index}: builder {got_shape:?} against nvcc {want_shape:?}",
                        op.name
                    );
                }
            }
        }
    }
}
