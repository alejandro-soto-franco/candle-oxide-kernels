//! The five-word unary entry builder, with Candle's two layout paths and the
//! null-`inp` in-place branch.
//!
//! `unary.cu`'s `UNARY_OP` macro selects between two grid-stride loops on the
//! `info == nullptr || is_contiguous(...)` verdict, and within each loop
//! chooses its load address on the independent `inp == nullptr` verdict:
//! `inp ? inp[i] : out[i]` in the contiguous loop, `inp ? inp[strided_i] :
//! out[i]` in the strided one. The `inp` test is emitted once in the entry
//! block; both loops take the resulting `i1` as a block argument and branch
//! on it themselves, so the load choice is not duplicated across four bodies.

use crate::body::emit_body;
use crate::desc::OpDesc;
use crate::emit::{cond_br, emit, goto, module_block, new_block, widen_u32_to_u64, Tys};
use crate::index::{emit_is_contiguous, emit_strided_index};
use cuda_oxide_codegen::experimental::CodegenModule;
use dialect_mir::attributes::MirCastKindAttr;
use dialect_mir::ops::{
    MirAddOp, MirCastOp, MirEqOp, MirFuncOp, MirLoadOp, MirLtOp, MirMulOp, MirPtrOffsetOp,
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
    region::Region,
    value::Value,
};

/// Everything both loops read, all of it defined in blocks that dominate
/// both loops.
struct Env {
    numel: Value,
    num_dims: Value,
    dims: Value,
    strides: Value,
    inp: Value,
    out: Value,
    /// `blockIdx.x * blockDim.x + threadIdx.x`
    i0: Value,
    /// `blockDim.x * gridDim.x`
    step: Value,
    /// `inp == nullptr`, computed once in the entry block.
    inp_is_null: Value,
}

/// Append `op`'s entry point to `module`.
pub fn build_unary(module: &mut CodegenModule, op: &OpDesc) {
    let name = op.name.clone();
    module.edit(|ctx, module| {
        let module_block = module_block(ctx, module);
        let tys = Tys::new(ctx, op.dtype, op.out);

        let params = vec![
            tys.i64,       // numel
            tys.i64,       // num_dims
            tys.usize_ptr, // info
            tys.in_ptr,    // inp
            tys.out_ptr,   // out
        ];
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

        let entry = BasicBlock::new(ctx, None, params);
        entry.insert_at_front(region, ctx);

        let (numel, num_dims, info, inp, out) = {
            let b = entry.deref(ctx);
            (
                b.get_argument(0),
                b.get_argument(1),
                b.get_argument(2),
                b.get_argument(3),
                b.get_argument(4),
            )
        };

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

        // `info == nullptr`. Same pointer-expose-address idiom as binary's
        // `dims_and_strides == nullptr` test: pliron has no int-to-pointer
        // cast kind, so the address is exposed and compared against zero.
        let info_addr = expose_address(ctx, entry, &tys, info);
        let zero = crate::emit::const_u64(ctx, entry, 0);
        let info_is_null = emit(
            ctx,
            entry,
            MirEqOp::get_concrete_op_info(),
            vec![tys.i1],
            vec![info_addr, zero],
        )
        .unwrap();

        // `inp == nullptr`, tested once here and carried into both loops.
        let inp_addr = expose_address(ctx, entry, &tys, inp);
        let inp_is_null = emit(
            ctx,
            entry,
            MirEqOp::get_concrete_op_info(),
            vec![tys.i1],
            vec![inp_addr, zero],
        )
        .unwrap();

        // The layout test lives behind the null check, so a null `info` is
        // never dereferenced.
        let check = new_block(ctx, region, vec![]);
        let strides = emit(
            ctx,
            check,
            MirPtrOffsetOp::get_concrete_op_info(),
            vec![tys.usize_ptr],
            vec![info, num_dims],
        )
        .unwrap();

        let (is_cont, after_cont) =
            emit_is_contiguous(ctx, region, &tys, num_dims, info, strides, check);

        let env = Env {
            numel,
            num_dims,
            dims: info,
            strides,
            inp,
            out,
            i0,
            step,
            inp_is_null,
        };

        let contiguous = emit_contiguous_loop(ctx, region, &tys, op, &env);
        let strided = emit_strided_loop(ctx, region, &tys, op, &env);

        // A null `info` takes the contiguous loop directly.
        cond_br(ctx, entry, info_is_null, contiguous, vec![], check, vec![]);
        cond_br(
            ctx,
            after_cont,
            is_cont,
            contiguous,
            vec![],
            strided,
            vec![],
        );

        func.get_operation().insert_at_back(module_block, ctx);
    });
    module.mark_kernel_entry(&op.name).unwrap();
}

/// The contiguous grid-stride loop: `TYPENAME x = inp ? inp[i] : out[i];`.
///
/// Returns the pre-header, the block a caller branches to in order to enter
/// this loop.
fn emit_contiguous_loop(
    ctx: &mut Context,
    region: Ptr<Region>,
    tys: &Tys,
    op: &OpDesc,
    env: &Env,
) -> Ptr<BasicBlock> {
    let preheader = new_block(ctx, region, vec![]);
    let header = new_block(ctx, region, vec![tys.i32]);
    let body = new_block(ctx, region, vec![tys.i32]);
    let load_inp = new_block(ctx, region, vec![]);
    let load_out = new_block(ctx, region, vec![]);
    let join = new_block(ctx, region, vec![tys.f]);
    let exit = new_block(ctx, region, vec![]);

    goto(ctx, preheader, header, vec![env.i0]);

    let i_hdr = header.deref(ctx).get_argument(0);
    let i_wide = widen_u32_to_u64(ctx, header, i_hdr);
    let more = emit(
        ctx,
        header,
        MirLtOp::get_concrete_op_info(),
        vec![tys.i1],
        vec![i_wide, env.numel],
    )
    .unwrap();
    cond_br(ctx, header, more, body, vec![i_hdr], exit, vec![]);

    let i = body.deref(ctx).get_argument(0);
    let i_off = widen_u32_to_u64(ctx, body, i);
    cond_br(
        ctx,
        body,
        env.inp_is_null,
        load_out,
        vec![],
        load_inp,
        vec![],
    );

    let ip = emit(
        ctx,
        load_inp,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![tys.in_ptr],
        vec![env.inp, i_off],
    )
    .unwrap();
    let x_inp = emit(
        ctx,
        load_inp,
        MirLoadOp::get_concrete_op_info(),
        vec![tys.f],
        vec![ip],
    )
    .unwrap();
    goto(ctx, load_inp, join, vec![x_inp]);

    let op_ptr = emit(
        ctx,
        load_out,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![tys.out_ptr],
        vec![env.out, i_off],
    )
    .unwrap();
    let x_out = emit(
        ctx,
        load_out,
        MirLoadOp::get_concrete_op_info(),
        vec![tys.f],
        vec![op_ptr],
    )
    .unwrap();
    goto(ctx, load_out, join, vec![x_out]);

    let x = join.deref(ctx).get_argument(0);
    let value = emit_body(ctx, join, tys, op, &[x]);
    let store_ptr = emit(
        ctx,
        join,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![tys.out_ptr],
        vec![env.out, i_off],
    )
    .unwrap();
    emit(
        ctx,
        join,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![store_ptr, value],
    );

    let i_next = emit(
        ctx,
        join,
        MirAddOp::get_concrete_op_info(),
        vec![tys.i32],
        vec![i, env.step],
    )
    .unwrap();
    goto(ctx, join, header, vec![i_next]);

    emit(
        ctx,
        exit,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
    );
    preheader
}

/// The strided grid-stride loop:
/// `unsigned strided_i = get_strided_index(...); TYPENAME x = inp ? inp[strided_i] : out[i];`.
///
/// The in-place branch reads `out` at the contiguous `i`, not `strided_i`,
/// reproducing `unary.cu`'s asymmetry exactly.
///
/// Returns the pre-header, the block a caller branches to in order to enter
/// this loop.
fn emit_strided_loop(
    ctx: &mut Context,
    region: Ptr<Region>,
    tys: &Tys,
    op: &OpDesc,
    env: &Env,
) -> Ptr<BasicBlock> {
    let preheader = new_block(ctx, region, vec![]);
    let header = new_block(ctx, region, vec![tys.i32]);
    let body = new_block(ctx, region, vec![tys.i32]);
    let load_inp = new_block(ctx, region, vec![]);
    let load_out = new_block(ctx, region, vec![]);
    let join = new_block(ctx, region, vec![tys.f]);
    let exit = new_block(ctx, region, vec![]);

    goto(ctx, preheader, header, vec![env.i0]);

    let i_hdr = header.deref(ctx).get_argument(0);
    let i_wide = widen_u32_to_u64(ctx, header, i_hdr);
    let more = emit(
        ctx,
        header,
        MirLtOp::get_concrete_op_info(),
        vec![tys.i1],
        vec![i_wide, env.numel],
    )
    .unwrap();
    cond_br(ctx, header, more, body, vec![i_hdr], exit, vec![]);

    let i = body.deref(ctx).get_argument(0);
    let (strided_i, blk) = emit_strided_index(
        ctx,
        region,
        tys,
        i,
        env.num_dims,
        env.dims,
        env.strides,
        body,
    );

    let i_off = widen_u32_to_u64(ctx, blk, i);
    cond_br(
        ctx,
        blk,
        env.inp_is_null,
        load_out,
        vec![],
        load_inp,
        vec![],
    );

    // `inp[strided_i]`: the strided index, only reached when `inp` is present.
    let strided_off = widen_u32_to_u64(ctx, load_inp, strided_i);
    let ip = emit(
        ctx,
        load_inp,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![tys.in_ptr],
        vec![env.inp, strided_off],
    )
    .unwrap();
    let x_inp = emit(
        ctx,
        load_inp,
        MirLoadOp::get_concrete_op_info(),
        vec![tys.f],
        vec![ip],
    )
    .unwrap();
    goto(ctx, load_inp, join, vec![x_inp]);

    // `out[i]`: the contiguous index, even on the strided path.
    let op_ptr = emit(
        ctx,
        load_out,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![tys.out_ptr],
        vec![env.out, i_off],
    )
    .unwrap();
    let x_out = emit(
        ctx,
        load_out,
        MirLoadOp::get_concrete_op_info(),
        vec![tys.f],
        vec![op_ptr],
    )
    .unwrap();
    goto(ctx, load_out, join, vec![x_out]);

    let x = join.deref(ctx).get_argument(0);
    let value = emit_body(ctx, join, tys, op, &[x]);
    let store_ptr = emit(
        ctx,
        join,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![tys.out_ptr],
        vec![env.out, i_off],
    )
    .unwrap();
    emit(
        ctx,
        join,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![store_ptr, value],
    );

    let i_next = emit(
        ctx,
        join,
        MirAddOp::get_concrete_op_info(),
        vec![tys.i32],
        vec![i, env.step],
    )
    .unwrap();
    goto(ctx, join, header, vec![i_next]);

    emit(
        ctx,
        exit,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
    );
    preheader
}

/// `(size_t)ptr`, the only pointer-to-integer conversion pliron's MIR dialect
/// carries.
fn expose_address(ctx: &mut Context, block: Ptr<BasicBlock>, tys: &Tys, ptr: Value) -> Value {
    let cast = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![tys.i64],
        vec![ptr],
        vec![],
        0,
    );
    MirCastOp::new(cast).set_attr_cast_kind(ctx, MirCastKindAttr::PointerExposeAddress);
    let res = cast.deref(ctx).get_result(0);
    cast.insert_at_back(block, ctx);
    res
}
