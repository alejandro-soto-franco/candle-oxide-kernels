//! The three fused chain entry builders.
//!
//! Each chain collapses a run of two Candle elementwise kernels into one
//! grid-stride pass, so the scaffolding here is `binary.rs`'s with the layout
//! branching removed. Every operand is contiguous by precondition, which the
//! caller checks before it reaches this kernel, so an operand index is an
//! explicit expression on the induction variable and there is one loop rather
//! than four.
//!
//! The index arithmetic runs at 64 bits throughout. `hss`, `ss` and `d` are
//! `size_t` parameters, so C would promote the `unsigned int` induction
//! variable to `size_t` for every one of `i / hss`, `i % ss` and `i % d`, and
//! computing at 32 bits would wrap on a tensor these chains are meant to
//! carry: the attention scores in the profile are 64.6 million elements, and
//! `hss` alone reaches into the millions.

use crate::body::call;
use crate::desc::{FusedChain, OpDesc};
use crate::emit::{cond_br, emit, goto, module_block, new_block, widen_u32_to_u64, Tys};
use cuda_oxide_codegen::experimental::CodegenModule;
use dialect_mir::ops::{
    MirAddOp, MirDivOp, MirFuncOp, MirLoadOp, MirLtOp, MirMulOp, MirPtrOffsetOp, MirRemOp,
    MirReturnOp, MirStoreOp,
};
use dialect_nvvm::ops::{
    ReadPtxSregCtaidXOp, ReadPtxSregNctaidXOp, ReadPtxSregNtidXOp, ReadPtxSregTidXOp,
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
fn params(tys: &Tys, chain: FusedChain) -> Vec<TypeHandle> {
    match chain {
        FusedChain::ScaleMask => vec![
            tys.i64,     // numel
            tys.i64,     // hss, H*S*S
            tys.i64,     // ss, S*S
            tys.in_ptr,  // x
            tys.in_ptr,  // mask
            tys.out_ptr, // out
            tys.f,       // scale
        ],
        FusedChain::BiasGeluErf => vec![
            tys.i64,     // numel
            tys.i64,     // d, the last dimension and the bias length
            tys.in_ptr,  // x
            tys.in_ptr,  // bias
            tys.out_ptr, // out
        ],
        FusedChain::BiasResidual => vec![
            tys.i64,     // numel
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
        FusedChain::ScaleMask => 5,
        FusedChain::BiasGeluErf => 4,
        FusedChain::BiasResidual => 5,
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
        let numel = args[0];
        let out = args[out_position(chain)];

        let tid = emit(
            ctx,
            entry,
            ReadPtxSregTidXOp::get_concrete_op_info(),
            vec![tys.i32],
            vec![],
        )
        .unwrap();
        let ctaid = emit(
            ctx,
            entry,
            ReadPtxSregCtaidXOp::get_concrete_op_info(),
            vec![tys.i32],
            vec![],
        )
        .unwrap();
        let ntid = emit(
            ctx,
            entry,
            ReadPtxSregNtidXOp::get_concrete_op_info(),
            vec![tys.i32],
            vec![],
        )
        .unwrap();
        let nctaid = emit(
            ctx,
            entry,
            ReadPtxSregNctaidXOp::get_concrete_op_info(),
            vec![tys.i32],
            vec![],
        )
        .unwrap();

        let base = emit(
            ctx,
            entry,
            MirMulOp::get_concrete_op_info(),
            vec![tys.i32],
            vec![ctaid, ntid],
        )
        .unwrap();
        let i0 = emit(
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

        let header = new_block(ctx, region, vec![tys.i32]);
        let body = new_block(ctx, region, vec![tys.i32]);
        let exit = new_block(ctx, region, vec![]);

        // There is one loop, so the entry block enters it directly and the
        // pre-header the elementwise builders need for their branch selection
        // has nothing left to do.
        goto(ctx, entry, header, vec![i0]);

        // header(i): `i < numel` compares a 32-bit index against a 64-bit
        // count, so widen first, matching C's integer promotion.
        let i_hdr = header.deref(ctx).get_argument(0);
        let i_wide = widen_u32_to_u64(ctx, header, i_hdr);
        let more = emit(
            ctx,
            header,
            MirLtOp::get_concrete_op_info(),
            vec![tys.i1],
            vec![i_wide, numel],
        )
        .unwrap();
        cond_br(ctx, header, more, body, vec![i_hdr], exit, vec![]);

        // body(i): the chain computes the stored value, then the induction
        // variable advances by the grid stride.
        let i = body.deref(ctx).get_argument(0);
        let i_off = widen_u32_to_u64(ctx, body, i);
        let value = emit_chain(ctx, body, &tys, &args, chain, i_off);
        let store_ptr = emit(
            ctx,
            body,
            MirPtrOffsetOp::get_concrete_op_info(),
            vec![tys.out_ptr],
            vec![out, i_off],
        )
        .unwrap();
        emit(
            ctx,
            body,
            MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![store_ptr, value],
        );

        let i_next = emit(
            ctx,
            body,
            MirAddOp::get_concrete_op_info(),
            vec![tys.i32],
            vec![i, step],
        )
        .unwrap();
        goto(ctx, body, header, vec![i_next]);

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

/// The value one chain stores at element `i`, given the entry block's
/// arguments and the 64-bit element index.
///
/// Every load and every index computation happens here, so the caller only
/// has to know where the output buffer sits.
fn emit_chain(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    tys: &Tys,
    args: &[Value],
    chain: FusedChain,
    i: Value,
) -> Value {
    match chain {
        // `out[i] = x[i] * scale + mask[(i / hss) * ss + (i % ss)]`.
        //
        // The multiply and the add stay two roundings. `Family::fma_contraction`
        // is what keeps them apart; contraction would fold the pair into one
        // FMA and diverge from the `affine_f32` then `badd_f32` pair this
        // replaces, which rounds the product to memory before it adds.
        FusedChain::ScaleMask => {
            let (hss, ss, x, mask, scale) = (args[1], args[2], args[3], args[4], args[6]);
            let xv = load_in(ctx, block, tys, x, i);
            let row = index_op(ctx, block, tys, MirDivOp::get_concrete_op_info(), i, hss);
            let row_base = index_op(ctx, block, tys, MirMulOp::get_concrete_op_info(), row, ss);
            let within = index_op(ctx, block, tys, MirRemOp::get_concrete_op_info(), i, ss);
            let mask_i = index_op(
                ctx,
                block,
                tys,
                MirAddOp::get_concrete_op_info(),
                row_base,
                within,
            );
            let mv = load_in(ctx, block, tys, mask, mask_i);
            let scaled = emit(
                ctx,
                block,
                MirMulOp::get_concrete_op_info(),
                vec![tys.f],
                vec![xv, scale],
            )
            .unwrap();
            emit(
                ctx,
                block,
                MirAddOp::get_concrete_op_info(),
                vec![tys.f],
                vec![scaled, mv],
            )
            .unwrap()
        }
        // `t = x[i] + bias[i % d]; out[i] = t * normcdf(t)`.
        FusedChain::BiasGeluErf => {
            let (d, x, bias) = (args[1], args[2], args[3]);
            let xv = load_in(ctx, block, tys, x, i);
            let bias_i = index_op(ctx, block, tys, MirRemOp::get_concrete_op_info(), i, d);
            let bv = load_in(ctx, block, tys, bias, bias_i);
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
        // `out[i] = (x[i] + bias[i % d]) + residual[i]`, left-associated.
        FusedChain::BiasResidual => {
            let (d, x, bias, residual) = (args[1], args[2], args[3], args[4]);
            let xv = load_in(ctx, block, tys, x, i);
            let bias_i = index_op(ctx, block, tys, MirRemOp::get_concrete_op_info(), i, d);
            let bv = load_in(ctx, block, tys, bias, bias_i);
            let rv = load_in(ctx, block, tys, residual, i);
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
