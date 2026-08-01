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
    let dir = std::env::temp_dir().join(format!(
        "candle_oxide_{label}_{}_{}",
        std::process::id(),
        unique
    ));
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

/// Locate `nvcc`, same search order as `find_ptxas`.
pub fn find_nvcc() -> PathBuf {
    ["CUDA_TOOLKIT_PATH", "CUDA_HOME"]
        .iter()
        .filter_map(|var| std::env::var(var).ok())
        .filter(|root| !root.trim().is_empty())
        .map(|root| PathBuf::from(root).join("bin/nvcc"))
        .chain(std::iter::once(PathBuf::from("/usr/local/cuda/bin/nvcc")))
        .find(|candidate| candidate.exists())
        .unwrap_or_else(|| PathBuf::from("nvcc"))
}

/// Compile a `.cu` to PTX with nvcc and return the text. Mirrors the flags
/// `candle-kernels/build.rs` passes on this branch: GCC 16's frontend trips
/// on a builtin pulled in by system headers, so a checkout that carries
/// `gcc16_cuda_compat.h` needs the compatibility pre-include and
/// `--allow-unsupported-compiler` to accept the host compiler at all. A
/// checkout without that file, such as upstream Candle, has no GCC 16
/// trouble to work around, so the pre-include is added only when the file
/// is present.
pub fn nvcc_ptx_for(
    cu_path: &std::path::Path,
    include_dir: &std::path::Path,
    arch: &str,
) -> String {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!(
        "candle_oxide_nvcc_{}_{}",
        std::process::id(),
        unique
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("reference.ptx");
    let compat_header = include_dir.join("gcc16_cuda_compat.h");
    let mut command = std::process::Command::new(find_nvcc());
    command
        .arg("--ptx")
        .arg(format!("-arch={arch}"))
        .arg("--allow-unsupported-compiler");
    if compat_header.exists() {
        command.arg("--pre-include").arg(&compat_header);
    }
    let result = command
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-I")
        .arg(include_dir)
        .arg(cu_path)
        .arg("-o")
        .arg(&out)
        .output();
    let out_result = result.unwrap_or_else(|error| {
        panic!(
            "could not run {}: {error}\nThis test needs `nvcc` from a CUDA toolkit \
             (no GPU required). Point CUDA_TOOLKIT_PATH or CUDA_HOME at the install \
             root, or put `nvcc` on PATH.",
            find_nvcc().display()
        )
    });
    assert!(
        out_result.status.success(),
        "nvcc failed on {}:\n{}",
        cu_path.display(),
        String::from_utf8_lossy(&out_result.stderr)
    );
    let read = std::fs::read_to_string(&out);
    let _ = std::fs::remove_dir_all(&dir);
    read.unwrap_or_else(|error| panic!("could not read {}: {error}", out.display()))
}

/// The `.param` lines of one entry, in declaration order.
pub fn entry_params(ptx: &str, entry: &str) -> Vec<String> {
    let needle = format!(".entry {entry}(");
    ptx.lines()
        .skip_while(|line| !line.contains(&needle))
        .take_while(|line| !line.contains(')'))
        .filter(|line| line.contains(".param"))
        .map(|line| line.trim().trim_end_matches(',').to_string())
        .collect()
}

/// The architecture declared on a PTX file's own `.target` line, e.g.
/// `sm_75` from `.target sm_75`.
///
/// Reads the baseline from the PTX text itself rather than from a literal
/// held by the caller, so a caller comparing against or reassembling a
/// committed artifact cannot drift from what is actually declared in it.
pub fn declared_target(ptx: &str) -> &str {
    ptx.lines()
        .find_map(|line| line.strip_prefix(".target "))
        .unwrap_or_else(|| panic!("no `.target` line found in:\n{ptx}"))
        .trim()
}

/// The PTX text of one entry point, from its `.visible .entry` line to the
/// closing brace of its body. Panics if the entry is absent, since every
/// caller is asserting something about an entry it expects to exist.
pub fn entry_body<'a>(ptx: &'a str, entry: &str) -> &'a str {
    let needle = format!(".visible .entry {entry}(");
    let start = ptx
        .find(&needle)
        .unwrap_or_else(|| panic!("entry {entry} is absent from the module"));
    let rest = &ptx[start..];
    // Entry bodies close with a brace at column 0.
    let end = rest
        .find("\n}")
        .map(|offset| start + offset + 2)
        .unwrap_or(ptx.len());
    &ptx[start..end]
}
