//! The fused family's entry builders.
//!
//! What the family's entries share is that each takes its index from the
//! launch geometry rather than computing it. Three of the five also collapse
//! a run of two Candle kernels into one pass; the other two are a single
//! traversal Candle already performs, run without an index computation. The
//! scaffolding is `binary.rs`'s with the layout branching removed: every
//! operand is contiguous by precondition, which the caller checks before it
//! reaches these kernels, so there is one loop rather than four and no
//! stride walk.
//!
//! # The grid carries the index
//!
//! The launch is shaped like the tensor: one grid row per tensor row, and
//! the `x` dimension covering the last dimension alone. Every term of an
//! operand index is then a special register or the induction variable, and
//! no term is computed from a flat element counter.
//!
//! That is what the geometry buys. A flat counter needs a division and a
//! modulo per dimension to recover the coordinates, against divisors loaded
//! from memory, and a division by a runtime value expands to roughly twenty
//! instructions. The measurement that prompted this ordered the first three
//! entries by achieved bandwidth exactly as a fixed per-element cost
//! predicts: the one moving twelve bytes an element reached 74 per cent of
//! the device ceiling while the two moving eight reached 46, because a fixed
//! cost hides behind memory traffic in proportion to how much traffic there
//! is. Removing the division took the worst of them to 92 per cent.
//!
//! [`FusedKind::Permute12`] is the same argument applied to a copy Candle
//! routes through `get_strided_index`, which walks `num_dims` deep, and
//! [`FusedKind::Bias`] to a broadcast add Candle sends down the same path
//! because a broadcast carries a zero stride.
//!
//! The row is loop-invariant, so each entry computes its row bases once in
//! the entry block and the loop adds the column to them.
//!
//! Index arithmetic runs at 64 bits. `ss` and `d` are `size_t` parameters,
//! and a row index reaching the 65535 CUDA caps `gridDim.y` and `gridDim.z`
//! at, multiplied by a row length in the thousands, leaves 32 bits well
//! behind. Only the column stays 32-bit, matching the induction variable of
//! Candle's own elementwise kernels.

use crate::body::{call, fp_cast};
use crate::desc::{FusedKind, OpDesc};
use crate::emit::{
    cond_br, const_f32, const_f64, const_u32, const_u64, emit, goto, module_block, new_block,
    widen_u32_to_u64, Tys,
};
use cuda_oxide_codegen::experimental::CodegenModule;
use dialect_mir::ops::{
    MirAddOp, MirDivOp, MirFuncOp, MirLoadOp, MirLtOp, MirMulOp, MirPtrOffsetOp, MirReturnOp,
    MirStoreOp, MirSubOp,
};
use dialect_nvvm::ops::{
    ReadPtxSregCtaidXOp, ReadPtxSregCtaidYOp, ReadPtxSregCtaidZOp, ReadPtxSregNctaidXOp,
    ReadPtxSregNctaidYOp, ReadPtxSregNtidXOp, ReadPtxSregTidXOp, ShflSyncBflyF32Op,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::TypeAttr, op_interfaces::SymbolOpInterface, types::FP64Type,
        types::FunctionType,
    },
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
    region::Region,
    value::Value,
};

/// The parameter list one entry declares, in the order the interface fixes.
///
/// Every word-sized parameter comes first and `scale` sits last, so a caller
/// pushing arguments walks a contiguous run of `.u64` words before it reaches
/// the single `.f32`.
///
/// No entry takes an element count. The loop bound is the row length, and
/// how many rows there are is what the launch already says.
fn params(tys: &Tys, kind: FusedKind) -> Vec<TypeHandle> {
    match kind {
        FusedKind::ScaleMask => vec![
            tys.i64,     // ss, S*S
            tys.in_ptr,  // x
            tys.in_ptr,  // mask
            tys.out_ptr, // out
            tys.f,       // scale
        ],
        FusedKind::BiasGeluErf | FusedKind::Bias => vec![
            tys.i64,     // d, the last dimension and the bias length
            tys.in_ptr,  // x
            tys.in_ptr,  // bias
            tys.out_ptr, // out
        ],
        FusedKind::BiasResidual => vec![
            tys.i64,     // d, the last dimension and the bias length
            tys.in_ptr,  // x
            tys.in_ptr,  // bias
            tys.in_ptr,  // residual
            tys.out_ptr, // out
        ],
        FusedKind::Permute12 => vec![
            tys.i64,     // d, the last dimension
            tys.in_ptr,  // src
            tys.out_ptr, // out
        ],
        // The column count is `S`, which is the grid's x extent, so this
        // kind takes no word-sized parameter at all.
        FusedKind::ScaleMaskSoftmax => vec![
            tys.in_ptr,  // scores
            tys.in_ptr,  // mask
            tys.out_ptr, // out
            tys.f,       // scale
        ],
    }
}

/// Which parameter holds the output buffer, per kind.
fn out_position(kind: FusedKind) -> usize {
    match kind {
        FusedKind::ScaleMask | FusedKind::BiasGeluErf | FusedKind::Bias => 3,
        FusedKind::BiasResidual => 4,
        FusedKind::Permute12 | FusedKind::ScaleMaskSoftmax => 2,
    }
}

/// Where one entry's loop starts and what it addresses, all computed in the
/// entry block, which dominates the loop.
struct Launch {
    /// The loop bound, which is also the length of one row: `ss` or `d`.
    bound: Value,
    /// The first column this thread visits.
    col0: Value,
    /// The distance between the columns one thread visits.
    step: Value,
    /// Where this block's row starts in the operands the read index
    /// addresses: `x`, `residual` and `src`.
    read_row: Value,
    /// Where this block's row starts in `out`. It differs from `read_row`
    /// only for [`FusedKind::Permute12`], which is what the permutation is.
    write_row: Value,
    /// Where this block's row starts in the broadcast operand. `mask` is
    /// shared across heads, so the scale-mask entry reads it at `b * ss`;
    /// `bias` is shared across every row, so the bias entries read it at the
    /// column alone and their base is zero. The permutation has no broadcast
    /// operand.
    broadcast: Option<Value>,
}

/// Read one 32-bit special register.
fn sreg(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    tys: &Tys,
    info: (fn(Ptr<Operation>) -> pliron::op::OpObj, std::any::TypeId),
) -> Value {
    emit(ctx, block, info, vec![tys.i32], vec![]).unwrap()
}

/// A 32-bit integer operation on two grid coordinates.
fn grid_op(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    tys: &Tys,
    info: (fn(Ptr<Operation>) -> pliron::op::OpObj, std::any::TypeId),
    a: Value,
    b: Value,
) -> Value {
    emit(ctx, block, info, vec![tys.i32], vec![a, b]).unwrap()
}

/// A 64-bit integer operation on two index values.
fn index_op(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    tys: &Tys,
    info: (fn(Ptr<Operation>) -> pliron::op::OpObj, std::any::TypeId),
    a: Value,
    b: Value,
) -> Value {
    emit(ctx, block, info, vec![tys.i64], vec![a, b]).unwrap()
}

/// `a * b + c` on 64-bit index values, the shape every row base takes.
fn row_base(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    tys: &Tys,
    a: Value,
    b: Value,
    c: Value,
) -> Value {
    let product = index_op(ctx, block, tys, MirMulOp::get_concrete_op_info(), a, b);
    index_op(
        ctx,
        block,
        tys,
        MirAddOp::get_concrete_op_info(),
        product,
        c,
    )
}

/// `col = ctaid.x * ntid.x + tid.x`, stepping by `ntid.x * nctaid.x`: the
/// grid's x dimension and the block together cover one row.
fn strided_column(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    tys: &Tys,
    tid: Value,
    ntid: Value,
    ctaid: Value,
    nctaid: Value,
) -> (Value, Value) {
    let base = grid_op(
        ctx,
        block,
        tys,
        MirMulOp::get_concrete_op_info(),
        ctaid,
        ntid,
    );
    let col0 = grid_op(ctx, block, tys, MirAddOp::get_concrete_op_info(), base, tid);
    let step = grid_op(
        ctx,
        block,
        tys,
        MirMulOp::get_concrete_op_info(),
        ntid,
        nctaid,
    );
    (col0, step)
}

/// Compute one entry's launch description from the grid coordinates.
fn emit_launch(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    tys: &Tys,
    args: &[Value],
    kind: FusedKind,
) -> Launch {
    let bound = args[0];
    let tid = sreg(ctx, block, tys, ReadPtxSregTidXOp::get_concrete_op_info());
    let ntid = sreg(ctx, block, tys, ReadPtxSregNtidXOp::get_concrete_op_info());
    let ctaid = sreg(ctx, block, tys, ReadPtxSregCtaidXOp::get_concrete_op_info());
    let nctaid = sreg(
        ctx,
        block,
        tys,
        ReadPtxSregNctaidXOp::get_concrete_op_info(),
    );

    match kind {
        // Grid `(ceil(ss / block), H, B)`. The head count arrives as
        // `nctaid.y` rather than as a parameter, which is what lets `hss`
        // leave the signature: `hss` was only ever `H * ss`.
        FusedKind::ScaleMask => {
            let (col0, step) = strided_column(ctx, block, tys, tid, ntid, ctaid, nctaid);
            let b = sreg(ctx, block, tys, ReadPtxSregCtaidZOp::get_concrete_op_info());
            let h = sreg(ctx, block, tys, ReadPtxSregCtaidYOp::get_concrete_op_info());
            let heads = sreg(
                ctx,
                block,
                tys,
                ReadPtxSregNctaidYOp::get_concrete_op_info(),
            );
            let b64 = widen_u32_to_u64(ctx, block, b);
            let h64 = widen_u32_to_u64(ctx, block, h);
            let heads64 = widen_u32_to_u64(ctx, block, heads);

            let row_index = row_base(ctx, block, tys, b64, heads64, h64);
            let row = index_op(
                ctx,
                block,
                tys,
                MirMulOp::get_concrete_op_info(),
                row_index,
                bound,
            );
            Launch {
                bound,
                col0,
                step,
                read_row: row,
                write_row: row,
                // `mask` is `[B, 1, S, S]`, so every head of one batch
                // element reads the same rows of it.
                broadcast: Some(index_op(
                    ctx,
                    block,
                    tys,
                    MirMulOp::get_concrete_op_info(),
                    b64,
                    bound,
                )),
            }
        }
        // Grid `(ceil(d / block), rows, 1)`, `rows` being every dimension
        // but the last flattened into one.
        FusedKind::BiasGeluErf | FusedKind::BiasResidual | FusedKind::Bias => {
            let (col0, step) = strided_column(ctx, block, tys, tid, ntid, ctaid, nctaid);
            let row = sreg(ctx, block, tys, ReadPtxSregCtaidYOp::get_concrete_op_info());
            let row64 = widen_u32_to_u64(ctx, block, row);
            let row = index_op(
                ctx,
                block,
                tys,
                MirMulOp::get_concrete_op_info(),
                row64,
                bound,
            );
            Launch {
                bound,
                col0,
                step,
                read_row: row,
                write_row: row,
                broadcast: Some(const_u64(ctx, block, 0)),
            }
        }
        // Grid `(D2, D1, B)` over a logical shape `[B, D2, D1, d]`.
        //
        // `ctaid.x` names the dimension being permuted, so it cannot also
        // index the column: `k` runs from `tid.x` alone, stepping by
        // `ntid.x`. The caller sizes the block to cover `d` where it can,
        // and the loop carries whatever is left.
        FusedKind::Permute12 => {
            let b = sreg(ctx, block, tys, ReadPtxSregCtaidZOp::get_concrete_op_info());
            let i = sreg(ctx, block, tys, ReadPtxSregCtaidYOp::get_concrete_op_info());
            let d1 = sreg(
                ctx,
                block,
                tys,
                ReadPtxSregNctaidYOp::get_concrete_op_info(),
            );
            let b64 = widen_u32_to_u64(ctx, block, b);
            let i64 = widen_u32_to_u64(ctx, block, i);
            let j64 = widen_u32_to_u64(ctx, block, ctaid);
            let d1_64 = widen_u32_to_u64(ctx, block, d1);
            let d2_64 = widen_u32_to_u64(ctx, block, nctaid);

            // `src[((b * D1 + i) * D2 + j) * d + k]`, the contiguous source.
            let src_plane = row_base(ctx, block, tys, b64, d1_64, i64);
            let src_row = row_base(ctx, block, tys, src_plane, d2_64, j64);
            // `out[((b * D2 + j) * D1 + i) * d + k]`, the permuted result.
            let out_plane = row_base(ctx, block, tys, b64, d2_64, j64);
            let out_row = row_base(ctx, block, tys, out_plane, d1_64, i64);

            Launch {
                bound,
                col0: tid,
                step: ntid,
                read_row: index_op(
                    ctx,
                    block,
                    tys,
                    MirMulOp::get_concrete_op_info(),
                    src_row,
                    bound,
                ),
                write_row: index_op(
                    ctx,
                    block,
                    tys,
                    MirMulOp::get_concrete_op_info(),
                    out_row,
                    bound,
                ),
                broadcast: None,
            }
        }
        // `emit_softmax` builds its own control flow and never asks for a
        // single-loop launch, so `build_fused` branches away before here.
        FusedKind::ScaleMaskSoftmax => {
            unreachable!("the softmax fold lays out its own passes")
        }
    }
}

/// Append `op`'s entry point to `module`.
pub fn build_fused(module: &mut CodegenModule, op: &OpDesc) {
    let name = op.name.clone();
    let kind = op
        .kind()
        .unwrap_or_else(|| panic!("{name} reached the fused builder without a kind"));
    module.edit(|ctx, module| {
        let module_block = module_block(ctx, module);
        let tys = Tys::new(ctx, op.dtype, op.out);

        let params = params(&tys, kind);
        let func_type = FunctionType::get(ctx, params.clone(), vec![]);
        let func = {
            let fop = Operation::new(
                ctx,
                MirFuncOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                1,
            );
            let func = MirFuncOp::new(ctx, fop, TypeAttr::new(func_type.into()));
            func.set_symbol_name(ctx, name.as_str().try_into().unwrap());
            func
        };
        let region = func.get_operation().deref(ctx).get_region(0);

        let arity = params.len();
        let entry = BasicBlock::new(ctx, None, params);
        entry.insert_at_front(region, ctx);

        let args: Vec<Value> = {
            let b = entry.deref(ctx);
            (0..arity).map(|index| b.get_argument(index)).collect()
        };

        // Every other kind is one traversal, so one loop carries it. The
        // softmax fold is three passes over the row with a warp reduction
        // between each, so it builds its own control flow.
        if kind == FusedKind::ScaleMaskSoftmax {
            emit_softmax(ctx, region, entry, &tys, &args);
            func.get_operation().insert_at_back(module_block, ctx);
            return;
        }

        let out = args[out_position(kind)];

        let launch = emit_launch(ctx, entry, &tys, &args, kind);

        let header = new_block(ctx, region, vec![tys.i32]);
        let body = new_block(ctx, region, vec![tys.i32]);
        let exit = new_block(ctx, region, vec![]);

        // There is one loop, so the entry block enters it directly and the
        // pre-header the elementwise builders need for their branch selection
        // has nothing left to do.
        goto(ctx, entry, header, vec![launch.col0]);

        // header(col): `col < bound` compares a 32-bit column against a
        // 64-bit row length, so widen first, matching C's integer promotion.
        // This test is what retires the threads left over when the block
        // size does not divide the row.
        let col_hdr = header.deref(ctx).get_argument(0);
        let col_wide = widen_u32_to_u64(ctx, header, col_hdr);
        let more = emit(
            ctx,
            header,
            MirLtOp::get_concrete_op_info(),
            vec![tys.i1],
            vec![col_wide, launch.bound],
        )
        .unwrap();
        cond_br(ctx, header, more, body, vec![col_hdr], exit, vec![]);

        // body(col): the entry computes the stored value, then the column
        // advances by the launch's stride.
        let col = body.deref(ctx).get_argument(0);
        let col_off = widen_u32_to_u64(ctx, body, col);
        let read_index = index_op(
            ctx,
            body,
            &tys,
            MirAddOp::get_concrete_op_info(),
            launch.read_row,
            col_off,
        );
        let write_index = index_op(
            ctx,
            body,
            &tys,
            MirAddOp::get_concrete_op_info(),
            launch.write_row,
            col_off,
        );
        let value = emit_value(ctx, body, &tys, &args, &launch, kind, read_index, col_off);
        let store_ptr = emit(
            ctx,
            body,
            MirPtrOffsetOp::get_concrete_op_info(),
            vec![tys.out_ptr],
            vec![out, write_index],
        )
        .unwrap();
        emit(
            ctx,
            body,
            MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![store_ptr, value],
        );

        let col_next = emit(
            ctx,
            body,
            MirAddOp::get_concrete_op_info(),
            vec![tys.i32],
            vec![col, launch.step],
        )
        .unwrap();
        goto(ctx, body, header, vec![col_next]);

        emit(
            ctx,
            exit,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
        );

        func.get_operation().insert_at_back(module_block, ctx);
    });
    module.mark_kernel_entry(&op.name).unwrap();
}

/// The warp Candle's softmax reduces over. `reduce.cu` launches block
/// `(1, 32, 1)` and this launches `(32, 1, 1)`, so both give a linear thread
/// id equal to the lane and the butterfly sees the same partners.
const WARP: u32 = 32;

/// The lanes Candle's butterfly exchanges with, in order
/// (`reduce.cu:246-249` and `:265-268`).
const BUTTERFLY: [u32; 5] = [16, 8, 4, 2, 1];

/// How a warp reduction folds a lane's value into the accumulator.
#[derive(Clone, Copy)]
enum Combine {
    /// `maxg(acc, partner)`, which is `fmaxf` at f32.
    Max,
    /// `acc += partner`.
    Add,
}

/// One butterfly exchange with the lane `delta` bits away, over the full
/// warp.
fn shfl_bfly(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    member_mask: Value,
    value: Value,
    delta: u32,
) -> Value {
    let lane = const_u32(ctx, block, delta);
    let op = ShflSyncBflyF32Op::build(ctx, member_mask, value, lane);
    let res = op.deref(ctx).get_result(0);
    op.insert_at_back(block, ctx);
    res
}

/// Candle's warp-wide reduction: five butterfly exchanges over lanes 16, 8,
/// 4, 2 and 1, folding each partner into the accumulator as it arrives.
///
/// The order is the result. A tree of a different shape, or the same shape
/// with the operands the other way round, reduces the same numbers to a
/// different float, so this is written out rather than rolled into a loop
/// the optimiser might reassociate.
fn emit_warp_reduce(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    tys: &Tys,
    member_mask: Value,
    seed: Value,
    combine: Combine,
) -> Value {
    let mut acc = seed;
    for delta in BUTTERFLY {
        let partner = shfl_bfly(ctx, block, member_mask, acc, delta);
        acc = match combine {
            Combine::Max => call(ctx, block, "__nv_fmaxf", tys.f, vec![acc, partner]),
            Combine::Add => emit(
                ctx,
                block,
                MirAddOp::get_concrete_op_info(),
                vec![tys.f],
                vec![acc, partner],
            )
            .unwrap(),
        };
    }
    acc
}

/// `(scores[row + col] * scale + 0.0f) + mask[mrow + col]`, the value
/// `fused_scale_mask_f32` would have written and `softmax_f32` would have
/// read straight back.
///
/// Returns the value and the score index, which the exponential pass stores
/// through as well.
#[allow(clippy::too_many_arguments)]
fn emit_scale_mask_value(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    tys: &Tys,
    args: &[Value],
    row: Value,
    mrow: Value,
    col: Value,
) -> (Value, Value) {
    let (scores, mask, scale) = (args[0], args[1], args[3]);
    let index = index_op(ctx, block, tys, MirAddOp::get_concrete_op_info(), row, col);
    let mask_index = index_op(ctx, block, tys, MirAddOp::get_concrete_op_info(), mrow, col);
    let xv = load_in(ctx, block, tys, scores, index);
    let mv = load_in(ctx, block, tys, mask, mask_index);
    let scaled = emit(
        ctx,
        block,
        MirMulOp::get_concrete_op_info(),
        vec![tys.f],
        vec![xv, scale],
    )
    .unwrap();
    // `affine_f32`'s `add` operand, kept for the reason round one added it:
    // it turns a `-0.0` product into `+0.0`.
    let zero = const_f32(ctx, block, 0.0);
    let affine = emit(
        ctx,
        block,
        MirAddOp::get_concrete_op_info(),
        vec![tys.f],
        vec![scaled, zero],
    )
    .unwrap();
    let value = emit(
        ctx,
        block,
        MirAddOp::get_concrete_op_info(),
        vec![tys.f],
        vec![affine, mv],
    )
    .unwrap();
    (value, index)
}

/// Candle's softmax over the scaled and masked scores, in place of the
/// `fused_scale_mask_f32` write and the `softmax_f32` read that followed it.
///
/// Three passes over the row, each a grid-stride walk from `tid.x` by one
/// warp, with a warp reduction between them:
///
/// 1. the maximum of `t(col)`, reduced across the warp;
/// 2. `exp(t(col) - max)`, stored, and summed across the warp;
/// 3. every stored element scaled by the reciprocal of that sum.
///
/// `t(col)` is recomputed in the first two rather than kept, which is the
/// whole point: keeping it is the round trip through memory this removes.
///
/// The stride is the literal [`WARP`] rather than `ntid.x`. The block is one
/// warp and the caller enforces it, so 32 is the true constant, and reading
/// it from the launch would leave the trip count as a runtime division.
/// Candle's own softmax carries three `div.u32` for exactly that reason.
fn emit_softmax(
    ctx: &mut Context,
    region: Ptr<Region>,
    entry: Ptr<BasicBlock>,
    tys: &Tys,
    args: &[Value],
) {
    let out = args[2];

    let tid = sreg(ctx, entry, tys, ReadPtxSregTidXOp::get_concrete_op_info());
    let s = sreg(ctx, entry, tys, ReadPtxSregCtaidXOp::get_concrete_op_info());
    let h = sreg(ctx, entry, tys, ReadPtxSregCtaidYOp::get_concrete_op_info());
    let b = sreg(ctx, entry, tys, ReadPtxSregCtaidZOp::get_concrete_op_info());
    let s_extent = sreg(
        ctx,
        entry,
        tys,
        ReadPtxSregNctaidXOp::get_concrete_op_info(),
    );
    let h_extent = sreg(
        ctx,
        entry,
        tys,
        ReadPtxSregNctaidYOp::get_concrete_op_info(),
    );

    // `ncols` is `S`, the grid's own x extent.
    let ncols = s_extent;
    let step = const_u32(ctx, entry, WARP);
    let member_mask = const_u32(ctx, entry, u32::MAX);

    let b64 = widen_u32_to_u64(ctx, entry, b);
    let h64 = widen_u32_to_u64(ctx, entry, h);
    let s64 = widen_u32_to_u64(ctx, entry, s);
    let s_extent64 = widen_u32_to_u64(ctx, entry, s_extent);
    let h_extent64 = widen_u32_to_u64(ctx, entry, h_extent);

    // `row = ((b * H + h) * S + s) * S`, the score row this block owns.
    let bh = row_base(ctx, entry, tys, b64, h_extent64, h64);
    let bhs = row_base(ctx, entry, tys, bh, s_extent64, s64);
    let row = index_op(
        ctx,
        entry,
        tys,
        MirMulOp::get_concrete_op_info(),
        bhs,
        s_extent64,
    );
    // `mrow = (b * S + s) * S`. The mask is `[B, 1, S, S]`, so every head of
    // one batch element reads the same row of it.
    let bs = row_base(ctx, entry, tys, b64, s_extent64, s64);
    let mrow = index_op(
        ctx,
        entry,
        tys,
        MirMulOp::get_concrete_op_info(),
        bs,
        s_extent64,
    );

    let max_header = new_block(ctx, region, vec![tys.i32, tys.f]);
    let max_body = new_block(ctx, region, vec![tys.i32, tys.f]);
    let max_done = new_block(ctx, region, vec![tys.f]);
    let sum_header = new_block(ctx, region, vec![tys.i32, tys.f]);
    let sum_body = new_block(ctx, region, vec![tys.i32, tys.f]);
    let sum_done = new_block(ctx, region, vec![tys.f]);
    let scale_header = new_block(ctx, region, vec![tys.i32]);
    let scale_body = new_block(ctx, region, vec![tys.i32]);
    let exit = new_block(ctx, region, vec![]);

    // Pass one: `max_val = -INFINITY`, then the maximum over this lane's
    // columns. A lane past the end contributes `-INFINITY`, which is the
    // identity, so a column count that is not a multiple of the warp needs
    // no special case.
    let neg_inf = const_f32(ctx, entry, f32::NEG_INFINITY);
    goto(ctx, entry, max_header, vec![tid, neg_inf]);

    let (max_col, max_acc) = {
        let blk = max_header.deref(ctx);
        (blk.get_argument(0), blk.get_argument(1))
    };
    let more = emit(
        ctx,
        max_header,
        MirLtOp::get_concrete_op_info(),
        vec![tys.i1],
        vec![max_col, ncols],
    )
    .unwrap();
    cond_br(
        ctx,
        max_header,
        more,
        max_body,
        vec![max_col, max_acc],
        max_done,
        vec![max_acc],
    );

    let (col, acc) = {
        let blk = max_body.deref(ctx);
        (blk.get_argument(0), blk.get_argument(1))
    };
    let col64 = widen_u32_to_u64(ctx, max_body, col);
    let (t, _) = emit_scale_mask_value(ctx, max_body, tys, args, row, mrow, col64);
    let folded = call(ctx, max_body, "__nv_fmaxf", tys.f, vec![acc, t]);
    let next = emit(
        ctx,
        max_body,
        MirAddOp::get_concrete_op_info(),
        vec![tys.i32],
        vec![col, step],
    )
    .unwrap();
    goto(ctx, max_body, max_header, vec![next, folded]);

    let lane_max = max_done.deref(ctx).get_argument(0);
    let max_val = emit_warp_reduce(ctx, max_done, tys, member_mask, lane_max, Combine::Max);

    // Pass two: `exp(t - max_val)`, stored and accumulated. A lane past the
    // end contributes `0.0f`, the identity for the sum.
    let zero = const_f32(ctx, max_done, 0.0);
    goto(ctx, max_done, sum_header, vec![tid, zero]);

    let (sum_col, sum_acc) = {
        let blk = sum_header.deref(ctx);
        (blk.get_argument(0), blk.get_argument(1))
    };
    let more = emit(
        ctx,
        sum_header,
        MirLtOp::get_concrete_op_info(),
        vec![tys.i1],
        vec![sum_col, ncols],
    )
    .unwrap();
    cond_br(
        ctx,
        sum_header,
        more,
        sum_body,
        vec![sum_col, sum_acc],
        sum_done,
        vec![sum_acc],
    );

    let (col, acc) = {
        let blk = sum_body.deref(ctx);
        (blk.get_argument(0), blk.get_argument(1))
    };
    let col64 = widen_u32_to_u64(ctx, sum_body, col);
    let (t, index) = emit_scale_mask_value(ctx, sum_body, tys, args, row, mrow, col64);
    let shifted = emit(
        ctx,
        sum_body,
        MirSubOp::get_concrete_op_info(),
        vec![tys.f],
        vec![t, max_val],
    )
    .unwrap();
    let val = call(ctx, sum_body, "__nv_expf", tys.f, vec![shifted]);
    let running = emit(
        ctx,
        sum_body,
        MirAddOp::get_concrete_op_info(),
        vec![tys.f],
        vec![acc, val],
    )
    .unwrap();
    store_out(ctx, sum_body, tys, out, index, val);
    let next = emit(
        ctx,
        sum_body,
        MirAddOp::get_concrete_op_info(),
        vec![tys.i32],
        vec![col, step],
    )
    .unwrap();
    goto(ctx, sum_body, sum_header, vec![next, running]);

    let lane_sum = sum_done.deref(ctx).get_argument(0);
    let total = emit_warp_reduce(ctx, sum_done, tys, member_mask, lane_sum, Combine::Add);

    // `const ACC inv_tmp = 1. / tmp`, with `ACC` a float and `1.` a double,
    // so the divide widens and the quotient narrows back. Both nvcc and this
    // pipeline fold that to one correctly rounded single-precision
    // reciprocal, which is the same number: a double quotient carries more
    // than twice the significand of a float, so rounding it twice lands
    // where rounding once would. `urecip_f32` takes the same shape and
    // lowers the same way.
    let inv_total = emit_double_reciprocal(ctx, sum_done, tys, total);
    goto(ctx, sum_done, scale_header, vec![tid]);

    // Pass three: scale every element this lane stored.
    let scale_col = scale_header.deref(ctx).get_argument(0);
    let more = emit(
        ctx,
        scale_header,
        MirLtOp::get_concrete_op_info(),
        vec![tys.i1],
        vec![scale_col, ncols],
    )
    .unwrap();
    cond_br(
        ctx,
        scale_header,
        more,
        scale_body,
        vec![scale_col],
        exit,
        vec![],
    );

    let col = scale_body.deref(ctx).get_argument(0);
    let col64 = widen_u32_to_u64(ctx, scale_body, col);
    let index = index_op(
        ctx,
        scale_body,
        tys,
        MirAddOp::get_concrete_op_info(),
        row,
        col64,
    );
    let stored = load_out(ctx, scale_body, tys, out, index);
    let scaled = emit(
        ctx,
        scale_body,
        MirMulOp::get_concrete_op_info(),
        vec![tys.f],
        vec![stored, inv_total],
    )
    .unwrap();
    store_out(ctx, scale_body, tys, out, index, scaled);
    let next = emit(
        ctx,
        scale_body,
        MirAddOp::get_concrete_op_info(),
        vec![tys.i32],
        vec![col, step],
    )
    .unwrap();
    goto(ctx, scale_body, scale_header, vec![next]);

    emit(
        ctx,
        exit,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
    );
}

/// `(float)(1.0 / (double)x)`, the shape `1. / tmp` takes when the
/// accumulator is a float and the literal is a double.
fn emit_double_reciprocal(ctx: &mut Context, block: Ptr<BasicBlock>, tys: &Tys, x: Value) -> Value {
    let f64_ty: TypeHandle = FP64Type::get(ctx).into();
    let one = const_f64(ctx, block, 1.0);
    let wide = fp_cast(ctx, block, x, f64_ty);
    let quotient = emit(
        ctx,
        block,
        MirDivOp::get_concrete_op_info(),
        vec![f64_ty],
        vec![one, wide],
    )
    .unwrap();
    fp_cast(ctx, block, quotient, tys.f)
}

/// `out[off] = value`.
fn store_out(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    tys: &Tys,
    out: Value,
    off: Value,
    value: Value,
) {
    let p = emit(
        ctx,
        block,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![tys.out_ptr],
        vec![out, off],
    )
    .unwrap();
    emit(
        ctx,
        block,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![p, value],
    );
}

/// `out[off]`, one element read back through the output pointer.
fn load_out(ctx: &mut Context, block: Ptr<BasicBlock>, tys: &Tys, out: Value, off: Value) -> Value {
    let p = emit(
        ctx,
        block,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![tys.out_ptr],
        vec![out, off],
    )
    .unwrap();
    emit(
        ctx,
        block,
        MirLoadOp::get_concrete_op_info(),
        vec![tys.f],
        vec![p],
    )
    .unwrap()
}

/// `base[off]`, one element loaded from a `const T *` parameter.
fn load_in(ctx: &mut Context, block: Ptr<BasicBlock>, tys: &Tys, base: Value, off: Value) -> Value {
    let p = emit(
        ctx,
        block,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![tys.in_ptr],
        vec![base, off],
    )
    .unwrap();
    emit(
        ctx,
        block,
        MirLoadOp::get_concrete_op_info(),
        vec![tys.f],
        vec![p],
    )
    .unwrap()
}

/// The index of the broadcast operand: its row base plus the column.
fn broadcast_index(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    tys: &Tys,
    launch: &Launch,
    col: Value,
) -> Value {
    let base = launch
        .broadcast
        .expect("this kind reads a broadcast operand");
    index_op(ctx, block, tys, MirAddOp::get_concrete_op_info(), base, col)
}

/// The value one entry stores, given the entry block's arguments, its launch
/// description, the read index and the column within the row.
///
/// Every load happens here, so the caller only has to know where the output
/// buffer sits and where the write index points.
#[allow(clippy::too_many_arguments)]
fn emit_value(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    tys: &Tys,
    args: &[Value],
    launch: &Launch,
    kind: FusedKind,
    read_index: Value,
    col: Value,
) -> Value {
    match kind {
        // `out[row + col] = (x[row + col] * scale + 0.0f) + mask[b * ss + col]`.
        //
        // The multiply and the mask add stay two roundings.
        // `Family::fma_contraction` is what keeps them apart; contraction
        // would fold the pair into one FMA and diverge from the `affine_f32`
        // then `badd_f32` pair this replaces, which rounds the product to
        // memory before it adds.
        //
        // The `+ 0.0f` is the `add` operand of `affine_f32`, which Candle
        // reaches through `Tensor::affine(scale, 0.0)` and whose body is
        // `x * mul + add` (`affine.cu:19`). It is dead for every value
        // except one: a product of `-0.0` becomes `+0.0`, since `-0.0 + 0.0`
        // is `+0.0` under round-to-nearest. Dropping it would leave the
        // `-0.0` in place and disagree with the reference whenever the mask
        // element is also `-0.0`. The fold `x + 0.0 -> x` is invalid in IEEE
        // without `nsz`, so the instruction survives; `tests/fused.rs` pins
        // the count rather than trusting that.
        FusedKind::ScaleMask => {
            let (x, mask, scale) = (args[1], args[2], args[4]);
            let mask_index = broadcast_index(ctx, block, tys, launch, col);
            let xv = load_in(ctx, block, tys, x, read_index);
            let mv = load_in(ctx, block, tys, mask, mask_index);
            let scaled = emit(
                ctx,
                block,
                MirMulOp::get_concrete_op_info(),
                vec![tys.f],
                vec![xv, scale],
            )
            .unwrap();
            let zero = const_f32(ctx, block, 0.0);
            let affine = emit(
                ctx,
                block,
                MirAddOp::get_concrete_op_info(),
                vec![tys.f],
                vec![scaled, zero],
            )
            .unwrap();
            emit(
                ctx,
                block,
                MirAddOp::get_concrete_op_info(),
                vec![tys.f],
                vec![affine, mv],
            )
            .unwrap()
        }
        // `t = x[row + col] + bias[col]; out[row + col] = t * normcdf(t)`.
        FusedKind::BiasGeluErf => {
            let (x, bias) = (args[1], args[2]);
            let bias_index = broadcast_index(ctx, block, tys, launch, col);
            let xv = load_in(ctx, block, tys, x, read_index);
            let bv = load_in(ctx, block, tys, bias, bias_index);
            let t = emit(
                ctx,
                block,
                MirAddOp::get_concrete_op_info(),
                vec![tys.f],
                vec![xv, bv],
            )
            .unwrap();
            let callee = kind
                .libdevice_names()
                .first()
                .expect("the GELU entry names a libdevice symbol");
            let cdf = call(ctx, block, callee, tys.f, vec![t]);
            emit(
                ctx,
                block,
                MirMulOp::get_concrete_op_info(),
                vec![tys.f],
                vec![t, cdf],
            )
            .unwrap()
        }
        // `out[row + col] = x[row + col] + bias[col]`, the same load pair as
        // the GELU entry with nothing applied to the sum.
        FusedKind::Bias => {
            let (x, bias) = (args[1], args[2]);
            let bias_index = broadcast_index(ctx, block, tys, launch, col);
            let xv = load_in(ctx, block, tys, x, read_index);
            let bv = load_in(ctx, block, tys, bias, bias_index);
            emit(
                ctx,
                block,
                MirAddOp::get_concrete_op_info(),
                vec![tys.f],
                vec![xv, bv],
            )
            .unwrap()
        }
        // `out[row + col] = (x[row + col] + bias[col]) + residual[row + col]`,
        // left-associated.
        FusedKind::BiasResidual => {
            let (x, bias, residual) = (args[1], args[2], args[3]);
            let bias_index = broadcast_index(ctx, block, tys, launch, col);
            let xv = load_in(ctx, block, tys, x, read_index);
            let bv = load_in(ctx, block, tys, bias, bias_index);
            let rv = load_in(ctx, block, tys, residual, read_index);
            let biased = emit(
                ctx,
                block,
                MirAddOp::get_concrete_op_info(),
                vec![tys.f],
                vec![xv, bv],
            )
            .unwrap();
            emit(
                ctx,
                block,
                MirAddOp::get_concrete_op_info(),
                vec![tys.f],
                vec![biased, rv],
            )
            .unwrap()
        }
        // A copy. The permutation is entirely in the two row bases, so the
        // value is the loaded element and there is no arithmetic to round.
        FusedKind::Permute12 => {
            let src = args[1];
            load_in(ctx, block, tys, src, read_index)
        }
        // Three passes rather than one value, so `emit_softmax` emits the
        // whole body and `build_fused` branches away before here.
        FusedKind::ScaleMaskSoftmax => {
            unreachable!("the softmax fold emits its own loop bodies")
        }
    }
}
