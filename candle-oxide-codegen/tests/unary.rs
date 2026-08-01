use candle_oxide_codegen::build_family;
use candle_oxide_codegen::desc::{unary_ops, Family, OpDesc};
use candle_oxide_codegen::testing::assert_ptxas_accepts;

fn pick(names: &[&str]) -> Vec<OpDesc> {
    unary_ops().into_iter().filter(|op| names.contains(&op.name.as_str())).collect()
}

#[test]
fn uneg_f32_declares_candle_five_word_abi() {
    let ptx = build_family(Family::Unary, &pick(&["uneg_f32"]), "sm_80").unwrap();
    assert!(ptx.contains(".visible .entry uneg_f32("), "entry point missing:\n{ptx}");
    let params = ptx
        .lines()
        .skip_while(|line| !line.contains(".visible .entry uneg_f32("))
        .take_while(|line| !line.contains(')'))
        .filter(|line| line.contains(".param"))
        .count();
    assert_eq!(params, 5, "uneg_f32 takes five positional words:\n{ptx}");
}

#[test]
fn the_in_place_branch_reads_the_output_buffer() {
    let ptx = build_family(Family::Unary, &pick(&["ucopy_f32"]), "sm_80").unwrap();
    // `inp ? inp[i] : out[i]` means the kernel loads from the output pointer
    // on one path, so a global load and a global store both name it.
    assert!(ptx.contains("ld.global"), "the in-place branch loads:\n{ptx}");
    assert_ptxas_accepts(ptx.as_bytes(), "unary_ucopy", "sm_80");
}

#[test]
fn every_unary_entry_compiles_into_one_module() {
    let ops = unary_ops();
    let ptx = build_family(Family::Unary, &ops, "sm_80").unwrap();
    for op in &ops {
        assert!(
            ptx.contains(&format!(".visible .entry {}(", op.name)),
            "{} missing from the unary module",
            op.name
        );
    }
    assert_ptxas_accepts(ptx.as_bytes(), "unary_all", "sm_80");
}

#[test]
fn libdevice_unary_ops_resolve_and_sqrt_is_correctly_rounded() {
    let ptx = build_family(Family::Unary, &pick(&["usqrt_f32", "usqrt_f64"]), "sm_80").unwrap();
    // #596 passes the reflect variable to llc, so NVVMReflectPass resolves
    // __CUDA_PREC_SQRT to 1 and libdevice sqrt gives the correctly rounded
    // form, matching nvcc. `sqrt.approx.f32` was the pre-#596 defect.
    assert!(ptx.contains("sqrt.rn.f32"), "f32 sqrt is correctly rounded:\n{ptx}");
    assert!(!ptx.contains("sqrt.approx.f32"), "the approximate form must not appear:\n{ptx}");
    assert_ptxas_accepts(ptx.as_bytes(), "unary_sqrt", "sm_80");
}

#[test]
fn urecip_f32_divides_in_double() {
    let ptx = build_family(Family::Unary, &pick(&["urecip_f32"]), "sm_80").unwrap();
    // `recipg(float a) { return 1.0 / a; }` promotes to double, so the f32
    // entry contains a double division and a narrowing conversion.
    assert!(ptx.contains("div.rn.f64"), "the promotion to double is missing:\n{ptx}");
    assert!(ptx.contains("cvt.rn.f32.f64"), "the narrowing on return is missing:\n{ptx}");
}
