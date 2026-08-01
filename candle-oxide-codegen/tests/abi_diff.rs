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
fn every_entry_matches_nvcc_word_for_word() {
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
                // The single accepted divergence is `.ptr .global`, an
                // address-space assertion from MirPtrType::get_global against
                // nvcc's bare `.ptr`. Every pointer cudarc passes is a global
                // allocation, so it holds.
                let normalised = g.replace(".ptr .global", ".ptr");
                assert_eq!(
                    normalised.split_whitespace().take(2).collect::<Vec<_>>(),
                    w.split_whitespace().take(2).collect::<Vec<_>>(),
                    "{} word {index}: builder {g:?} against nvcc {w:?}",
                    op.name
                );
            }
        }
    }
}
