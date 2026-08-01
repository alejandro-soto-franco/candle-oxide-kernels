use candle_oxide_codegen::build_family;
use candle_oxide_codegen::desc::{Arity, Body, DType, Family, OpDesc, OutType};

fn badd(dtype: DType) -> OpDesc {
    OpDesc {
        name: format!("badd_{}", dtype.suffix()),
        dtype,
        arity: Arity::Binary,
        out: OutType::Same,
        body: Body::Add,
    }
}

#[test]
fn badd_f32_declares_candle_six_word_abi() {
    let ptx = build_family(Family::Binary, &[badd(DType::F32)], "sm_80").unwrap();
    assert!(ptx.contains(".visible .entry badd_f32("), "entry point missing:\n{ptx}");
    let params = ptx
        .lines()
        .skip_while(|line| !line.contains(".visible .entry badd_f32("))
        .take_while(|line| !line.contains(')'))
        .filter(|line| line.contains(".param"))
        .count();
    assert_eq!(params, 6, "badd_f32 takes six positional words:\n{ptx}");
}

#[test]
fn badd_f64_compiles_too() {
    let ptx = build_family(Family::Binary, &[badd(DType::F64)], "sm_80").unwrap();
    assert!(ptx.contains(".visible .entry badd_f64("), "entry point missing:\n{ptx}");
    assert!(ptx.contains("add.f64"), "f64 add lowers to add.f64:\n{ptx}");
}

#[test]
fn the_grid_stride_loop_survives_as_a_loop() {
    let ptx = build_family(Family::Binary, &[badd(DType::F32)], "sm_80").unwrap();
    assert!(ptx.contains("bra"), "the loop lowers to a branch:\n{ptx}");
}

#[test]
fn one_module_holds_every_entry_it_was_given() {
    let ptx = build_family(
        Family::Binary,
        &[badd(DType::F32), badd(DType::F64)],
        "sm_80",
    )
    .unwrap();
    assert!(ptx.contains(".visible .entry badd_f32("));
    assert!(ptx.contains(".visible .entry badd_f64("));
}
