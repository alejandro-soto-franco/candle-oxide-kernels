use std::process::Command;

/// A total build failure (both families fail, since the arch string is
/// shared) must exit non-zero before the write loop is ever entered. This
/// proves "a failed build writes nothing", not cross-family atomicity: the
/// build loop exits on `Family::Binary`'s failure, so `Family::Unary` is
/// never attempted and neither file is written. The write-vs-write case
/// (one family's file already landed, the other's write then fails) is
/// exercised directly against `write_family` in `xtask/src/main.rs`'s own
/// unit tests, since that is a case a shared, single-arch CLI invocation
/// cannot provoke through the binary's stdin/stdout surface.
///
/// `sm_00` parses as a syntactically valid `Target` and fails later, inside
/// `compiler.compile(...)`, with a `CompileError` ("compute capability must
/// contain at least two digits"). It does not exercise the xtask's
/// `Target::parse` pre-check; `--arch bogus` does that (verified manually,
/// see task-9-report.md).
#[test]
fn a_failed_build_writes_nothing() {
    let dir = std::env::temp_dir().join(format!("xtask_atomicity_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let binary = dir.join("binary.ptx");
    std::fs::write(&binary, "SENTINEL").unwrap();

    // An arch nvcc and ptxas both reject stops the run before any write.
    let status = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["gen-ptx", "--arch", "sm_00", "--out"])
        .arg(&dir)
        .output()
        .expect("xtask runs");

    assert!(!status.status.success(), "an invalid arch must fail the run");
    assert_eq!(
        std::fs::read_to_string(&binary).unwrap(),
        "SENTINEL",
        "a failed run must not touch the committed artifact"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
