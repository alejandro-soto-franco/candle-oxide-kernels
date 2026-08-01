//! Turns loaded operand values into the value stored to `out`, mirroring the
//! `FUNC` argument of Candle's `BINARY_OP` and `UNARY_OP` macros.

use crate::desc::{Body, CmpKind, DType, OpDesc, OutType};
use crate::emit::{emit, Tys};
use dialect_mir::attributes::MirCastKindAttr;
use dialect_mir::ops::{
    MirAddOp, MirCallOp, MirCastOp, MirDivOp, MirEqOp, MirGeOp, MirGtOp, MirLeOp, MirLtOp,
    MirMulOp, MirNeOp, MirNegOp, MirSubOp,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::attributes::StringAttr,
    builtin::types::FP64Type,
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
    value::Value,
};

/// Emit a call to `callee` with `args`, returning its single result.
fn call(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    callee: &str,
    result_ty: TypeHandle,
    args: Vec<Value>,
) -> Value {
    let op = Operation::new(
        ctx,
        MirCallOp::get_concrete_op_info(),
        vec![result_ty],
        args,
        vec![],
        0,
    );
    MirCallOp::new(op).set_attr_callee(ctx, StringAttr::new(callee.to_string()));
    let res = op.deref(ctx).get_result(0);
    op.insert_at_back(block, ctx);
    res
}

/// Cast a float value to another float width.
fn fp_cast(ctx: &mut Context, block: Ptr<BasicBlock>, v: Value, to: TypeHandle) -> Value {
    let op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![to],
        vec![v],
        vec![],
        0,
    );
    MirCastOp::new(op).set_attr_cast_kind(ctx, MirCastKindAttr::FloatToFloat);
    let res = op.deref(ctx).get_result(0);
    op.insert_at_back(block, ctx);
    res
}

/// Compute the stored value from the loaded operands.
///
/// `xs` holds one value for a unary entry and two for a binary entry. The
/// returned value has type `tys.out_elem`.
pub fn emit_body(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    tys: &Tys,
    op: &OpDesc,
    xs: &[Value],
) -> Value {
    let f = tys.f;
    match op.body {
        Body::Add => emit(
            ctx,
            block,
            MirAddOp::get_concrete_op_info(),
            vec![f],
            vec![xs[0], xs[1]],
        )
        .unwrap(),
        Body::Sub => emit(
            ctx,
            block,
            MirSubOp::get_concrete_op_info(),
            vec![f],
            vec![xs[0], xs[1]],
        )
        .unwrap(),
        Body::Mul => emit(
            ctx,
            block,
            MirMulOp::get_concrete_op_info(),
            vec![f],
            vec![xs[0], xs[1]],
        )
        .unwrap(),
        Body::Div => emit(
            ctx,
            block,
            MirDivOp::get_concrete_op_info(),
            vec![f],
            vec![xs[0], xs[1]],
        )
        .unwrap(),
        Body::Copy => xs[0],
        Body::Neg => emit(
            ctx,
            block,
            MirNegOp::get_concrete_op_info(),
            vec![f],
            vec![xs[0]],
        )
        .unwrap(),
        Body::Sqr => emit(
            ctx,
            block,
            MirMulOp::get_concrete_op_info(),
            vec![f],
            vec![xs[0], xs[0]],
        )
        .unwrap(),
        // `recipg(float a) { return 1.0 / a; }` at cuda_utils.cuh:120. The
        // literal is a double, so the f32 form divides in double and rounds
        // once on return. `1.0f / a` would be a different function.
        Body::Recip => {
            let f64_ty: TypeHandle = FP64Type::get(ctx).into();
            let one = crate::emit::const_f64(ctx, block, 1.0);
            let x = match op.dtype {
                DType::F32 => fp_cast(ctx, block, xs[0], f64_ty),
                DType::F64 => xs[0],
            };
            let q = emit(
                ctx,
                block,
                MirDivOp::get_concrete_op_info(),
                vec![f64_ty],
                vec![one, x],
            )
            .unwrap();
            match op.dtype {
                DType::F32 => fp_cast(ctx, block, q, f),
                DType::F64 => q,
            }
        }
        Body::Libdevice(_) => {
            let callee = op.libdevice_name().expect("Body::Libdevice carries a name");
            call(ctx, block, callee, f, xs.to_vec())
        }
        Body::Cmp(kind) => {
            let info = match kind {
                CmpKind::Eq => MirEqOp::get_concrete_op_info(),
                CmpKind::Ne => MirNeOp::get_concrete_op_info(),
                CmpKind::Lt => MirLtOp::get_concrete_op_info(),
                CmpKind::Le => MirLeOp::get_concrete_op_info(),
                CmpKind::Gt => MirGtOp::get_concrete_op_info(),
                CmpKind::Ge => MirGeOp::get_concrete_op_info(),
            };
            let bit = emit(ctx, block, info, vec![tys.i1], vec![xs[0], xs[1]]).unwrap();
            debug_assert_eq!(op.out, OutType::U8);
            // Widen the predicate to the `uint8_t` the output buffer holds.
            let widened = Operation::new(
                ctx,
                MirCastOp::get_concrete_op_info(),
                vec![tys.out_elem],
                vec![bit],
                vec![],
                0,
            );
            MirCastOp::new(widened).set_attr_cast_kind(ctx, MirCastKindAttr::IntToInt);
            let res = widened.deref(ctx).get_result(0);
            widened.insert_at_back(block, ctx);
            res
        }
    }
}
