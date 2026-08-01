//! Candle's two layout helpers, `cuda_utils.cuh:9-23` and `:25-38`, built as
//! pliron control flow.
//!
//! Both helpers append their own loop blocks to the function region and hand
//! back the block where the caller carries on, so no caller assumes its work
//! stays inside a single basic block. Every block a helper appends is
//! dominated by the block the caller passed in, so values live at the call
//! site are still live at the continuation and need no threading through
//! block arguments.

use crate::emit::{
    cond_br, const_i1, const_u32, const_u64, emit, goto, narrow_u64_to_u32, new_block,
    widen_u32_to_u64, Tys,
};
use dialect_mir::ops::{
    MirAddOp, MirDivOp, MirGtOp, MirLoadOp, MirLtOp, MirMulOp, MirNeOp, MirPtrOffsetOp, MirRemOp,
    MirSubOp,
};
use pliron::{
    basic_block::BasicBlock,
    context::{Context, Ptr},
    op::Op,
    region::Region,
    value::Value,
};

/// `dims[dim_idx]`, where `dim_idx` is a 64-bit index into a `const size_t *`.
fn load_usize(ctx: &mut Context, block: Ptr<BasicBlock>, tys: &Tys, base: Value, idx: Value) -> Value {
    let p = emit(ctx, block, MirPtrOffsetOp::get_concrete_op_info(), vec![tys.usize_ptr], vec![base, idx])
        .unwrap();
    emit(ctx, block, MirLoadOp::get_concrete_op_info(), vec![tys.i64], vec![p]).unwrap()
}

/// `dim_idx = num_dims - 1 - d`, computed in 64 bits and then round-tripped
/// through 32 bits, matching C's `unsigned int dim_idx = num_dims - 1 - d`.
fn dim_index(ctx: &mut Context, block: Ptr<BasicBlock>, tys: &Tys, num_dims: Value, d: Value) -> Value {
    let one = const_u64(ctx, block, 1);
    let last = emit(ctx, block, MirSubOp::get_concrete_op_info(), vec![tys.i64], vec![num_dims, one]).unwrap();
    let wide = emit(ctx, block, MirSubOp::get_concrete_op_info(), vec![tys.i64], vec![last, d]).unwrap();
    let narrow = narrow_u64_to_u32(ctx, block, wide);
    widen_u32_to_u64(ctx, block, narrow)
}

/// `is_contiguous(num_dims, dims, strides)` from `cuda_utils.cuh:9-23`.
///
/// The accumulator is 64-bit, as `size_t acc` is, and it is compared against
/// the 64-bit stride without narrowing. Dimensions of extent 1 are skipped.
/// Returns the `i1` verdict and the block where the caller carries on.
pub fn emit_is_contiguous(
    ctx: &mut Context,
    region: Ptr<Region>,
    tys: &Tys,
    num_dims: Value,
    dims: Value,
    strides: Value,
    entry: Ptr<BasicBlock>,
) -> (Value, Ptr<BasicBlock>) {
    let header = new_block(ctx, region, vec![tys.i64, tys.i64]); // (acc, d)
    let step = new_block(ctx, region, vec![]);
    let check_acc = new_block(ctx, region, vec![]);
    let advance = new_block(ctx, region, vec![]);
    let yes = new_block(ctx, region, vec![]);
    let no = new_block(ctx, region, vec![]);
    let join = new_block(ctx, region, vec![tys.i1]);

    let acc0 = const_u64(ctx, entry, 1);
    let d0 = const_u64(ctx, entry, 0);
    goto(ctx, entry, header, vec![acc0, d0]);

    // header(acc, d): `d < num_dims` on unsigned 64-bit values.
    let (acc, d) = {
        let b = header.deref(ctx);
        (b.get_argument(0), b.get_argument(1))
    };
    let more = emit(ctx, header, MirLtOp::get_concrete_op_info(), vec![tys.i1], vec![d, num_dims]).unwrap();
    cond_br(ctx, header, more, step, vec![], yes, vec![]);

    // step: load dims[dim_idx] and strides[dim_idx], then test `dim > 1`.
    let dim_idx = dim_index(ctx, step, tys, num_dims, d);
    let dim = load_usize(ctx, step, tys, dims, dim_idx);
    let strd = load_usize(ctx, step, tys, strides, dim_idx);
    let one = const_u64(ctx, step, 1);
    let gt1 = emit(ctx, step, MirGtOp::get_concrete_op_info(), vec![tys.i1], vec![dim, one]).unwrap();
    cond_br(ctx, step, gt1, check_acc, vec![], advance, vec![]);

    // check_acc: `acc != strides[dim_idx]` fails the test. The `&&` in C
    // short-circuits, which the two branches reproduce.
    let ne = emit(ctx, check_acc, MirNeOp::get_concrete_op_info(), vec![tys.i1], vec![acc, strd]).unwrap();
    cond_br(ctx, check_acc, ne, no, vec![], advance, vec![]);

    // advance: acc *= dim; d += 1.
    let acc_next = emit(ctx, advance, MirMulOp::get_concrete_op_info(), vec![tys.i64], vec![acc, dim]).unwrap();
    let one = const_u64(ctx, advance, 1);
    let d_next = emit(ctx, advance, MirAddOp::get_concrete_op_info(), vec![tys.i64], vec![d, one]).unwrap();
    goto(ctx, advance, header, vec![acc_next, d_next]);

    let t = const_i1(ctx, yes, true);
    goto(ctx, yes, join, vec![t]);
    let f = const_i1(ctx, no, false);
    goto(ctx, no, join, vec![f]);

    let verdict = join.deref(ctx).get_argument(0);
    (verdict, join)
}

/// `get_strided_index(idx, num_dims, dims, strides)` from `cuda_utils.cuh:25-38`,
/// in the inlined form `binary_op_macros.cuh` uses.
///
/// The arithmetic is 32-bit and it wraps, matching Candle. `dims[d]` and
/// `strides[d]` are 64-bit loads narrowed to 32 bits; the remainder, the
/// multiply, the accumulation and the division all run at 32 bits. Only the
/// dimension counter stays 64-bit, since it indexes the `size_t` arrays.
///
/// Returns the 32-bit index and the block where the caller carries on.
#[allow(clippy::too_many_arguments)]
pub fn emit_strided_index(
    ctx: &mut Context,
    region: Ptr<Region>,
    tys: &Tys,
    i: Value,
    num_dims: Value,
    dims: Value,
    strides: Value,
    entry: Ptr<BasicBlock>,
) -> (Value, Ptr<BasicBlock>) {
    let header = new_block(ctx, region, vec![tys.i32, tys.i32, tys.i64]); // (strided_i, idx, d)
    let step = new_block(ctx, region, vec![]);
    let exit = new_block(ctx, region, vec![tys.i32]);

    let zero = const_u32(ctx, entry, 0);
    let d0 = const_u64(ctx, entry, 0);
    goto(ctx, entry, header, vec![zero, i, d0]);

    let (strided_i, idx, d) = {
        let b = header.deref(ctx);
        (b.get_argument(0), b.get_argument(1), b.get_argument(2))
    };
    let more = emit(ctx, header, MirLtOp::get_concrete_op_info(), vec![tys.i1], vec![d, num_dims]).unwrap();
    cond_br(ctx, header, more, step, vec![], exit, vec![strided_i]);

    let dim_idx = dim_index(ctx, step, tys, num_dims, d);
    let dim64 = load_usize(ctx, step, tys, dims, dim_idx);
    let strd64 = load_usize(ctx, step, tys, strides, dim_idx);
    // The truncations Candle performs implicitly on its way into `unsigned int`.
    let dim = narrow_u64_to_u32(ctx, step, dim64);
    let strd = narrow_u64_to_u32(ctx, step, strd64);

    let i_dim = emit(ctx, step, MirRemOp::get_concrete_op_info(), vec![tys.i32], vec![idx, dim]).unwrap();
    let contrib = emit(ctx, step, MirMulOp::get_concrete_op_info(), vec![tys.i32], vec![i_dim, strd]).unwrap();
    let strided_next =
        emit(ctx, step, MirAddOp::get_concrete_op_info(), vec![tys.i32], vec![strided_i, contrib]).unwrap();
    let idx_next = emit(ctx, step, MirDivOp::get_concrete_op_info(), vec![tys.i32], vec![idx, dim]).unwrap();
    let one = const_u64(ctx, step, 1);
    let d_next = emit(ctx, step, MirAddOp::get_concrete_op_info(), vec![tys.i64], vec![d, one]).unwrap();
    goto(ctx, step, header, vec![strided_next, idx_next, d_next]);

    let result = exit.deref(ctx).get_argument(0);
    (result, exit)
}
