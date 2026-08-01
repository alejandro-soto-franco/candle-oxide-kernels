//! Test helpers shared by the tiers that need a CUDA toolkit but no GPU.

use std::path::PathBuf;

/// Locate `ptxas`, matching `cuda-oxide/crates/cuda-oxide-codegen/tests/spine_kernel_ptx.rs`.
pub fn find_ptxas() -> PathBuf {
    ["CUDA_TOOLKIT_PATH", "CUDA_HOME"]
        .iter()
        .filter_map(|var| std::env::var(var).ok())
        .filter(|root| !root.trim().is_empty())
        .map(|root| PathBuf::from(root).join("bin/ptxas"))
        .chain(std::iter::once(PathBuf::from("/usr/local/cuda/bin/ptxas")))
        .find(|candidate| candidate.exists())
        .unwrap_or_else(|| PathBuf::from("ptxas"))
}

/// Write `ptx` to a scratch file and require `ptxas -arch=<arch>` to accept it.
pub fn assert_ptxas_accepts(ptx: &[u8], label: &str, arch: &str) {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("candle_oxide_{label}_{}_{}", std::process::id(), unique));
    std::fs::create_dir_all(&dir).unwrap();
    let ptx_path = dir.join("kernel.ptx");
    let cubin_path = dir.join("kernel.cubin");
    std::fs::write(&ptx_path, ptx).unwrap();

    let ptxas = find_ptxas();
    let result = std::process::Command::new(&ptxas)
        .arg(format!("-arch={arch}"))
        .arg("--compile-only")
        .arg(&ptx_path)
        .arg("-o")
        .arg(&cubin_path)
        .output();
    let _ = std::fs::remove_dir_all(&dir);

    let out = result.unwrap_or_else(|error| {
        panic!(
            "could not run {}: {error}\nThis test needs `ptxas` from a CUDA toolkit \
             (no GPU required). Point CUDA_TOOLKIT_PATH or CUDA_HOME at the install \
             root, or put `ptxas` on PATH.",
            ptxas.display()
        )
    });
    assert!(
        out.status.success(),
        "ptxas rejected the {label} PTX at {arch}:\nstderr:\n{}\n\nPTX:\n{}",
        String::from_utf8_lossy(&out.stderr),
        String::from_utf8_lossy(ptx)
    );
}
