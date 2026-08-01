//! Guards against a committed artifact drifting from the source that
//! generates it. Regenerates both families into a scratch directory and
//! diffs the result against `candle-oxide-kernels/ptx/`.
//!
//! This only proves anything because generation is deterministic: two runs
//! produce byte-identical output (verified by hand while writing this test,
//! `sha256sum` matching across two separate `--out` directories and against
//! the committed files). If a future change to the codegen makes output
//! depend on wall-clock time, an absolute path, or hash-ordered iteration,
//! this test starts failing intermittently rather than on genuine drift;
//! treat that as a codegen bug to fix, not a reason to loosen the
//! comparison here.

use std::process::Command;

#[test]
fn regenerating_reproduces_the_committed_artifacts() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let committed = root.join("candle-oxide-kernels/ptx");
    let fresh = std::env::temp_dir().join(format!("xtask_repro_{}", std::process::id()));

    let out = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["gen-ptx", "--arch", "sm_80", "--out"])
        .arg(&fresh)
        .output()
        .expect("xtask runs");
    assert!(
        out.status.success(),
        "generation failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    for name in ["binary.ptx", "unary.ptx"] {
        let want = std::fs::read_to_string(committed.join(name)).unwrap();
        let got = std::fs::read_to_string(fresh.join(name)).unwrap();
        assert_eq!(
            got, want,
            "{name} drifted from its source. Run `cargo xtask gen-ptx` and commit the result."
        );
    }
    let _ = std::fs::remove_dir_all(&fresh);
}
