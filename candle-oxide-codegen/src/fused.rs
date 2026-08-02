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

use crate::body::call;
use crate::desc::{FusedKind, OpDesc};
use crate::emit::{
    cond_br, const_f32, const_u64, emit, goto, module_block, new_block, widen_u32_to_u64, Tys,
};
use cuda_oxide_codegen::experimental::CodegenModule;
use dialect_mir::ops::{
    MirAddOp, MirFuncOp, MirLoadOp, MirLtOp, MirMulOp, MirPtrOffsetOp, MirReturnOp, MirStoreOp,
};
use dialect_nvvm::ops::{
    ReadPtxSregCtaidXOp, ReadPtxSregCtaidYOp, ReadPtxSregCtaidZOp, ReadPtxSregNctaidXOp,
    ReadPtxSregNctaidYOp, ReadPtxSregNtidXOp, ReadPtxSregTidXOp,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{attributes::TypeAttr, op_interfaces::SymbolOpInterface, types::FunctionType},
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
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
    }
}

/// Which parameter holds the output buffer, per kind.
fn out_position(kind: FusedKind) -> usize {
    match kind {
        FusedKind::ScaleMask | FusedKind::BiasGeluErf | FusedKind::Bias => 3,
        FusedKind::BiasResidual => 4,
        FusedKind::Permute12 => 2,
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
                .libdevice_name()
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
    }
}
