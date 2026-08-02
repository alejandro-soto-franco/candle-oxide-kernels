//! Low-level pliron construction helpers, factored out of the `badd_f32`
//! spike at `cuda-oxide/crates/cuda-oxide-codegen/tests/candle_badd_spike.rs`.

use crate::desc::{DType, OutType};
use dialect_mir::attributes::MirCastKindAttr;
use dialect_mir::ops::{MirCastOp, MirCondBranchOp, MirConstantOp, MirFloatConstantOp, MirGotoOp};
use dialect_mir::types::MirPtrType;
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{FPDoubleAttr, FPSingleAttr, IntegerAttr},
        op_interfaces::OperandSegmentInterface,
        types::{FP32Type, FP64Type, IntegerType, Signedness},
    },
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
    region::Region,
    utils::apint::APInt,
    value::Value,
};

/// Append an operation with no successors to `block`, returning its first
/// result if it has one.
pub fn emit(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    info: (fn(Ptr<Operation>) -> pliron::op::OpObj, std::any::TypeId),
    results: Vec<TypeHandle>,
    operands: Vec<Value>,
) -> Option<Value> {
    let op = Operation::new(ctx, info, results.clone(), operands, vec![], 0);
    let res = if results.is_empty() {
        None
    } else {
        Some(op.deref(ctx).get_result(0))
    };
    op.insert_at_back(block, ctx);
    res
}

/// Widen a signless 32-bit value to 64 bits. Signless integers carry
/// unsigned semantics, which is what `size_t` and Candle's `unsigned int`
/// indices need, so no signedness attribute is set.
pub fn widen_u32_to_u64(ctx: &mut Context, block: Ptr<BasicBlock>, v: Value) -> Value {
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![v],
        vec![],
        0,
    );
    MirCastOp::new(op).set_attr_cast_kind(ctx, MirCastKindAttr::IntToInt);
    let res = op.deref(ctx).get_result(0);
    op.insert_at_back(block, ctx);
    res
}

/// Narrow a signless 64-bit value to 32 bits, the truncation Candle performs
/// implicitly when a `size_t` stride lands in an `unsigned int` accumulator.
pub fn narrow_u64_to_u32(ctx: &mut Context, block: Ptr<BasicBlock>, v: Value) -> Value {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![v],
        vec![],
        0,
    );
    MirCastOp::new(op).set_attr_cast_kind(ctx, MirCastKindAttr::IntToInt);
    let res = op.deref(ctx).get_result(0);
    op.insert_at_back(block, ctx);
    res
}

fn const_int(ctx: &mut Context, block: Ptr<BasicBlock>, value: u64, width: u32) -> Value {
    let ty = IntegerType::get(ctx, width, Signedness::Signless);
    let op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![ty.into()],
        vec![],
        vec![],
        0,
    );
    let attr = IntegerAttr::new(
        ty,
        APInt::from_u64(value, std::num::NonZeroUsize::new(width as usize).unwrap()),
    );
    MirConstantOp::new(op).set_attr_value(ctx, attr);
    let res = op.deref(ctx).get_result(0);
    op.insert_at_back(block, ctx);
    res
}

/// A 1-bit constant, the `true` and `false` that `is_contiguous` returns.
pub fn const_i1(ctx: &mut Context, block: Ptr<BasicBlock>, value: bool) -> Value {
    const_int(ctx, block, value as u64, 1)
}

pub fn const_u32(ctx: &mut Context, block: Ptr<BasicBlock>, value: u32) -> Value {
    const_int(ctx, block, value as u64, 32)
}

pub fn const_u64(ctx: &mut Context, block: Ptr<BasicBlock>, value: u64) -> Value {
    const_int(ctx, block, value, 64)
}

/// A 32-bit float constant, the width `affine.cu`'s `add` operand carries at
/// f32 and so the width `FusedKind::ScaleMask` needs for its literal `0.0f`.
pub fn const_f32(ctx: &mut Context, block: Ptr<BasicBlock>, value: f32) -> Value {
    let f32_ty: TypeHandle = FP32Type::get(ctx).into();
    let op = Operation::new(
        ctx,
        MirFloatConstantOp::get_concrete_op_info(),
        vec![f32_ty],
        vec![],
        vec![],
        0,
    );
    MirFloatConstantOp::new(op).set_attr_float_value(ctx, FPSingleAttr::from(value));
    let res = op.deref(ctx).get_result(0);
    op.insert_at_back(block, ctx);
    res
}

/// A 64-bit float constant, the width `Body::Recip` needs for the literal
/// `1.0` in `recipg`'s `1.0 / a` (`cuda_utils.cuh:120`).
pub fn const_f64(ctx: &mut Context, block: Ptr<BasicBlock>, value: f64) -> Value {
    let f64_ty: TypeHandle = FP64Type::get(ctx).into();
    let op = Operation::new(
        ctx,
        MirFloatConstantOp::get_concrete_op_info(),
        vec![f64_ty],
        vec![],
        vec![],
        0,
    );
    MirFloatConstantOp::new(op).set_attr_float_value_f64(ctx, FPDoubleAttr::from(value));
    let res = op.deref(ctx).get_result(0);
    op.insert_at_back(block, ctx);
    res
}

/// Every type one entry needs.
pub struct Tys {
    /// Element type of the operands.
    pub f: TypeHandle,
    /// Element type of the output buffer, `f` unless the entry compares.
    pub out_elem: TypeHandle,
    pub i1: TypeHandle,
    pub i32: TypeHandle,
    pub i64: TypeHandle,
    /// `const size_t *`
    pub usize_ptr: TypeHandle,
    /// `const T *`
    pub in_ptr: TypeHandle,
    /// `OUT_T *`
    pub out_ptr: TypeHandle,
}

impl Tys {
    pub fn new(ctx: &mut Context, dtype: DType, out: OutType) -> Self {
        let f: TypeHandle = match dtype {
            DType::F32 => FP32Type::get(ctx).into(),
            DType::F64 => FP64Type::get(ctx).into(),
        };
        let i1 = IntegerType::get(ctx, 1, Signedness::Signless).into();
        let i8: TypeHandle = IntegerType::get(ctx, 8, Signedness::Signless).into();
        let i32 = IntegerType::get(ctx, 32, Signedness::Signless).into();
        let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
        let out_elem = match out {
            OutType::Same => f,
            OutType::U8 => i8,
        };
        Tys {
            f,
            out_elem,
            i1,
            i32,
            i64: i64_ty.into(),
            usize_ptr: MirPtrType::get_global(ctx, i64_ty.into(), false).into(),
            in_ptr: MirPtrType::get_global(ctx, f, false).into(),
            out_ptr: MirPtrType::get_global(ctx, out_elem, true).into(),
        }
    }
}

/// Append an empty basic block with arguments of type `args` to `region`.
pub fn new_block(ctx: &mut Context, region: Ptr<Region>, args: Vec<TypeHandle>) -> Ptr<BasicBlock> {
    let block = BasicBlock::new(ctx, None, args);
    block.insert_at_back(region, ctx);
    block
}

/// Terminate `block` with an unconditional branch to `target`, passing `args`
/// as the target's block arguments.
pub fn goto(ctx: &mut Context, block: Ptr<BasicBlock>, target: Ptr<BasicBlock>, args: Vec<Value>) {
    let op = Operation::new(
        ctx,
        MirGotoOp::get_concrete_op_info(),
        vec![],
        args,
        vec![target],
        0,
    );
    op.insert_at_back(block, ctx);
}

/// Terminate `block` with a conditional branch. `true_args` and `false_args`
/// supply each successor's block arguments.
pub fn cond_br(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    cond: Value,
    true_target: Ptr<BasicBlock>,
    true_args: Vec<Value>,
    false_target: Ptr<BasicBlock>,
    false_args: Vec<Value>,
) {
    let (flat, segs) =
        MirCondBranchOp::compute_segment_sizes(vec![vec![cond], true_args, false_args]);
    let op = Operation::new(
        ctx,
        MirCondBranchOp::get_concrete_op_info(),
        vec![],
        flat,
        vec![true_target, false_target],
        0,
    );
    Operation::get_op::<MirCondBranchOp>(op, ctx)
        .unwrap()
        .set_operand_segment_sizes(ctx, segs);
    op.insert_at_back(block, ctx);
}

/// The module's single top-level block, created if the module has none.
pub fn module_block(ctx: &mut Context, module: &impl Op) -> Ptr<BasicBlock> {
    let module_region = module.get_operation().deref(ctx).get_region(0);
    let existing = {
        let region = module_region.deref(ctx);
        region.iter(ctx).next()
    };
    match existing {
        Some(block) => block,
        None => {
            let block = BasicBlock::new(ctx, None, vec![]);
            block.insert_at_back(module_region, ctx);
            block
        }
    }
}
