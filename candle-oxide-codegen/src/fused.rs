//! The three fused chain entry builders.
//!
//! Each chain collapses a run of two Candle elementwise kernels into one
//! pass, so the scaffolding here is `binary.rs`'s with the layout branching
//! removed. Every operand is contiguous by precondition, which the caller
//! checks before it reaches this kernel, so there is one loop rather than
//! four and no stride walk.
//!
//! # The grid carries the index
//!
//! The launch is shaped like the tensor: one grid row per tensor row, and
//! the `x` dimension covering the last dimension alone. Every term of an
//! operand index is then a special register or the induction variable, and
//! no term is computed from a flat element counter.
//!
//! That is what the geometry buys. A flat counter needs `i / hss`, `i % ss`
//! and `i % d` to recover the row and the column, and a division by a
//! runtime value expands to roughly twenty instructions. The measurement
//! that prompted this ordered the three chains by achieved bandwidth exactly
//! as a fixed per-element cost predicts: the chain moving twelve bytes an
//! element reached 74 per cent of the device ceiling while the two moving
//! eight reached 46, because a fixed cost hides behind memory traffic in
//! proportion to how much traffic there is.
//!
//! The row is loop-invariant, so each chain computes its row bases once in
//! the entry block and the loop adds the column to them.
//!
//! Index arithmetic runs at 64 bits. `ss` and `d` are `size_t` parameters,
//! and a row index reaching the 65535 CUDA caps `gridDim.y` and `gridDim.z`
//! at, multiplied by a row length in the thousands, leaves 32 bits well
//! behind. Only the column stays 32-bit, matching the induction variable of
//! Candle's own elementwise kernels.

use crate::body::call;
use crate::desc::{FusedChain, OpDesc};
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

/// The parameter list one chain declares, in the order the interface fixes.
///
/// Every word-sized parameter comes first and `scale` sits last, so a caller
/// pushing arguments walks a contiguous run of `.u64` words before it reaches
/// the single `.f32`.
///
/// No chain takes an element count. The loop bound is the row length, and
/// how many rows there are is `gridDim.y` times `gridDim.z`, which the
/// launch already carries.
fn params(tys: &Tys, chain: FusedChain) -> Vec<TypeHandle> {
    match chain {
        FusedChain::ScaleMask => vec![
            tys.i64,     // ss, S*S
            tys.in_ptr,  // x
            tys.in_ptr,  // mask
            tys.out_ptr, // out
            tys.f,       // scale
        ],
        FusedChain::BiasGeluErf => vec![
            tys.i64,     // d, the last dimension and the bias length
            tys.in_ptr,  // x
            tys.in_ptr,  // bias
            tys.out_ptr, // out
        ],
        FusedChain::BiasResidual => vec![
            tys.i64,     // d, the last dimension and the bias length
            tys.in_ptr,  // x
            tys.in_ptr,  // bias
            tys.in_ptr,  // residual
            tys.out_ptr, // out
        ],
    }
}

/// Which parameter holds the output buffer, per chain.
fn out_position(chain: FusedChain) -> usize {
    match chain {
        FusedChain::ScaleMask => 3,
        FusedChain::BiasGeluErf => 3,
        FusedChain::BiasResidual => 4,
    }
}

/// The loop-invariant offsets one chain reads from the grid, all computed in
/// the entry block, which dominates the loop.
struct Rows {
    /// The loop bound, which is also the length of one row: `ss` or `d`.
    bound: Value,
    /// Where this block's row starts in the operands the full index
    /// addresses: `x`, `out`, and `residual` where a chain has one.
    row: Value,
    /// Where this block's row starts in the broadcast operand. `mask` is
    /// shared across heads, so the scale-mask chain reads it at `b * ss`;
    /// `bias` is shared across every row, so the other two read it at the
    /// column alone and their base is zero.
    broadcast: Value,
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

/// Compute one chain's row bases from the grid coordinates.
fn emit_rows(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    tys: &Tys,
    args: &[Value],
    chain: FusedChain,
) -> Rows {
    let bound = args[0];
    match chain {
        // Grid `(ceil(ss / block), H, B)`. The head count arrives as
        // `nctaid.y` rather than as a parameter, which is what lets `hss`
        // leave the signature: `hss` was only ever `H * ss`.
        FusedChain::ScaleMask => {
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

            let batch_rows = index_op(
                ctx,
                block,
                tys,
                MirMulOp::get_concrete_op_info(),
                b64,
                heads64,
            );
            let row_index = index_op(
                ctx,
                block,
                tys,
                MirAddOp::get_concrete_op_info(),
                batch_rows,
                h64,
            );
            Rows {
                bound,
                row: index_op(
                    ctx,
                    block,
                    tys,
                    MirMulOp::get_concrete_op_info(),
                    row_index,
                    bound,
                ),
                // `mask` is `[B, 1, S, S]`, so every head of one batch
                // element reads the same rows of it.
                broadcast: index_op(
                    ctx,
                    block,
                    tys,
                    MirMulOp::get_concrete_op_info(),
                    b64,
                    bound,
                ),
            }
        }
        // Grid `(ceil(d / block), rows, 1)`, `rows` being every dimension
        // but the last flattened into one.
        FusedChain::BiasGeluErf | FusedChain::BiasResidual => {
            let row = sreg(ctx, block, tys, ReadPtxSregCtaidYOp::get_concrete_op_info());
            let row64 = widen_u32_to_u64(ctx, block, row);
            Rows {
                bound,
                row: index_op(
                    ctx,
                    block,
                    tys,
                    MirMulOp::get_concrete_op_info(),
                    row64,
                    bound,
                ),
                broadcast: const_u64(ctx, block, 0),
            }
        }
    }
}

/// Append `op`'s entry point to `module`.
pub fn build_fused(module: &mut CodegenModule, op: &OpDesc) {
    let name = op.name.clone();
    let chain = op
        .chain()
        .unwrap_or_else(|| panic!("{name} reached the fused builder without a chain"));
    module.edit(|ctx, module| {
        let module_block = module_block(ctx, module);
        let tys = Tys::new(ctx, op.dtype, op.out);

        let params = params(&tys, chain);
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
        let out = args[out_position(chain)];

        let tid = sreg(ctx, entry, &tys, ReadPtxSregTidXOp::get_concrete_op_info());
        let ctaid = sreg(
            ctx,
            entry,
            &tys,
            ReadPtxSregCtaidXOp::get_concrete_op_info(),
        );
        let ntid = sreg(ctx, entry, &tys, ReadPtxSregNtidXOp::get_concrete_op_info());
        let nctaid = sreg(
            ctx,
            entry,
            &tys,
            ReadPtxSregNctaidXOp::get_concrete_op_info(),
        );

        let base = emit(
            ctx,
            entry,
            MirMulOp::get_concrete_op_info(),
            vec![tys.i32],
            vec![ctaid, ntid],
        )
        .unwrap();
        let col0 = emit(
            ctx,
            entry,
            MirAddOp::get_concrete_op_info(),
            vec![tys.i32],
            vec![base, tid],
        )
        .unwrap();
        let step = emit(
            ctx,
            entry,
            MirMulOp::get_concrete_op_info(),
            vec![tys.i32],
            vec![ntid, nctaid],
        )
        .unwrap();

        let rows = emit_rows(ctx, entry, &tys, &args, chain);

        let header = new_block(ctx, region, vec![tys.i32]);
        let body = new_block(ctx, region, vec![tys.i32]);
        let exit = new_block(ctx, region, vec![]);

        // There is one loop, so the entry block enters it directly and the
        // pre-header the elementwise builders need for their branch selection
        // has nothing left to do.
        goto(ctx, entry, header, vec![col0]);

        // header(col): `col < bound` compares a 32-bit column against a
        // 64-bit row length, so widen first, matching C's integer promotion.
        // The x dimension of the grid covers the row, so this test is what
        // retires the threads of the last block in a row that does not
        // divide by the block size.
        let col_hdr = header.deref(ctx).get_argument(0);
        let col_wide = widen_u32_to_u64(ctx, header, col_hdr);
        let more = emit(
            ctx,
            header,
            MirLtOp::get_concrete_op_info(),
            vec![tys.i1],
            vec![col_wide, rows.bound],
        )
        .unwrap();
        cond_br(ctx, header, more, body, vec![col_hdr], exit, vec![]);

        // body(col): the chain computes the stored value, then the column
        // advances by the grid stride.
        let col = body.deref(ctx).get_argument(0);
        let col_off = widen_u32_to_u64(ctx, body, col);
        let index = index_op(
            ctx,
            body,
            &tys,
            MirAddOp::get_concrete_op_info(),
            rows.row,
            col_off,
        );
        let value = emit_chain(ctx, body, &tys, &args, &rows, chain, index, col_off);
        let store_ptr = emit(
            ctx,
            body,
            MirPtrOffsetOp::get_concrete_op_info(),
            vec![tys.out_ptr],
            vec![out, index],
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
            vec![col, step],
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

/// The value one chain stores at `index`, given the entry block's arguments,
/// its row bases, the full element index and the column within the row.
///
/// Every load happens here, so the caller only has to know where the output
/// buffer sits.
#[allow(clippy::too_many_arguments)]
fn emit_chain(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    tys: &Tys,
    args: &[Value],
    rows: &Rows,
    chain: FusedChain,
    index: Value,
    col: Value,
) -> Value {
    // The broadcast operand is read at its own row base plus the column.
    let broadcast_index = index_op(
        ctx,
        block,
        tys,
        MirAddOp::get_concrete_op_info(),
        rows.broadcast,
        col,
    );
    match chain {
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
        FusedChain::ScaleMask => {
            let (x, mask, scale) = (args[1], args[2], args[4]);
            let xv = load_in(ctx, block, tys, x, index);
            let mv = load_in(ctx, block, tys, mask, broadcast_index);
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
        FusedChain::BiasGeluErf => {
            let (x, bias) = (args[1], args[2]);
            let xv = load_in(ctx, block, tys, x, index);
            let bv = load_in(ctx, block, tys, bias, broadcast_index);
            let t = emit(
                ctx,
                block,
                MirAddOp::get_concrete_op_info(),
                vec![tys.f],
                vec![xv, bv],
            )
            .unwrap();
            let callee = chain
                .libdevice_name()
                .expect("the GELU chain names a libdevice symbol");
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
        // `out[row + col] = (x[row + col] + bias[col]) + residual[row + col]`,
        // left-associated.
        FusedChain::BiasResidual => {
            let (x, bias, residual) = (args[1], args[2], args[3]);
            let xv = load_in(ctx, block, tys, x, index);
            let bv = load_in(ctx, block, tys, bias, broadcast_index);
            let rv = load_in(ctx, block, tys, residual, index);
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
    }
}
