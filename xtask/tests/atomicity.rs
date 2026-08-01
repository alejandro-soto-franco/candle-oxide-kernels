use std::process::Command;

#[test]
fn a_failed_run_leaves_the_committed_artifacts_untouched() {
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
