//! The six-word binary entry builder, with Candle's four layout paths.
//!
//! `binary_op_macros.cuh` selects between four grid-stride loops on the two
//! `is_contiguous` verdicts, and a null `dims_and_strides` forces both to
//! true without a dereference. Each loop differs only in how it turns the
//! induction variable into an operand index, so one Rust routine emits all
//! four, parameterised by a pair of `IndexMode`s.

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

/// How one operand turns the induction variable into an element index.
#[derive(Clone, Copy, PartialEq, Eq)]
enum IndexMode {
    /// `lhs[i]`, the operand is contiguous.
    Contiguous,
    /// `lhs[lhs_i]`, the index walks the dimensions.
    Strided,
}

/// Everything the four loops read, all of it defined in blocks that dominate
/// every loop that uses it.
struct Env {
    numel: Value,
    num_dims: Value,
    dims: Value,
    lhs: Value,
    rhs: Value,
    out: Value,
    /// `blockIdx.x * blockDim.x + threadIdx.x`
    i0: Value,
    /// `blockDim.x * gridDim.x`
    step: Value,
    lhs_strides: Value,
    rhs_strides: Value,
}

/// Append `op`'s entry point to `module`.
pub fn build_binary(module: &mut CodegenModule, op: &OpDesc) {
    let name = op.name.clone();
    module.edit(|ctx, module| {
        let module_block = module_block(ctx, module);
        let tys = Tys::new(ctx, op.dtype, op.out);

        let params = vec![
            tys.i64,       // numel
            tys.i64,       // num_dims
            tys.usize_ptr, // dims_and_strides
            tys.in_ptr,    // lhs
            tys.in_ptr,    // rhs
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

        let (numel, num_dims, dims, lhs, rhs, out) = {
            let b = entry.deref(ctx);
            (
                b.get_argument(0),
                b.get_argument(1),
                b.get_argument(2),
                b.get_argument(3),
                b.get_argument(4),
                b.get_argument(5),
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

        // `dims_and_strides == nullptr`. pliron offers no int-to-pointer cast
        // kind, so the address is exposed and compared against zero, which
        // lowers to `ptrtoint` and an integer `icmp eq`.
        let ds_addr = expose_address(ctx, entry, &tys, dims);
        let zero = crate::emit::const_u64(ctx, entry, 0);
        let is_null = emit(
            ctx,
            entry,
            MirEqOp::get_concrete_op_info(),
            vec![tys.i1],
            vec![ds_addr, zero],
        )
        .unwrap();

        // The layout tests live behind the null check, so a null pointer is
        // never dereferenced.
        let check = new_block(ctx, region, vec![]);
        let one = crate::emit::const_u64(ctx, check, 1);
        let two = crate::emit::const_u64(ctx, check, 2);
        let nd1 = emit(
            ctx,
            check,
            MirMulOp::get_concrete_op_info(),
            vec![tys.i64],
            vec![num_dims, one],
        )
        .unwrap();
        let nd2 = emit(
            ctx,
            check,
            MirMulOp::get_concrete_op_info(),
            vec![tys.i64],
            vec![num_dims, two],
        )
        .unwrap();
        let lhs_strides = emit(
            ctx,
            check,
            MirPtrOffsetOp::get_concrete_op_info(),
            vec![tys.usize_ptr],
            vec![dims, nd1],
        )
        .unwrap();
        let rhs_strides = emit(
            ctx,
            check,
            MirPtrOffsetOp::get_concrete_op_info(),
            vec![tys.usize_ptr],
            vec![dims, nd2],
        )
        .unwrap();

        let (lhs_cont, after_lhs) =
            emit_is_contiguous(ctx, region, &tys, num_dims, dims, lhs_strides, check);
        let (rhs_cont, after_rhs) =
            emit_is_contiguous(ctx, region, &tys, num_dims, dims, rhs_strides, after_lhs);

        let env = Env {
            numel,
            num_dims,
            dims,
            lhs,
            rhs,
            out,
            i0,
            step,
            lhs_strides,
            rhs_strides,
        };

        // Named for which operand is contiguous, matching the C's `lhs_cont`
        // and `rhs_cont`. The `IndexMode` on the same line is the opposite
        // reading: an operand that is contiguous is indexed contiguously.
        let both = emit_loop(
            ctx,
            region,
            &tys,
            op,
            &env,
            IndexMode::Contiguous,
            IndexMode::Contiguous,
        );
        let lhs_cont_only = emit_loop(
            ctx,
            region,
            &tys,
            op,
            &env,
            IndexMode::Contiguous,
            IndexMode::Strided,
        );
        let rhs_cont_only = emit_loop(
            ctx,
            region,
            &tys,
            op,
            &env,
            IndexMode::Strided,
            IndexMode::Contiguous,
        );
        let neither = emit_loop(
            ctx,
            region,
            &tys,
            op,
            &env,
            IndexMode::Strided,
            IndexMode::Strided,
        );

        // A null `dims_and_strides` takes the both-contiguous loop directly.
        cond_br(ctx, entry, is_null, both, vec![], check, vec![]);

        // The two verdicts select one of the four loops.
        let on_lhs_cont = new_block(ctx, region, vec![]);
        let on_lhs_strided = new_block(ctx, region, vec![]);
        cond_br(
            ctx,
            after_rhs,
            lhs_cont,
            on_lhs_cont,
            vec![],
            on_lhs_strided,
            vec![],
        );
        cond_br(
            ctx,
            on_lhs_cont,
            rhs_cont,
            both,
            vec![],
            lhs_cont_only,
            vec![],
        );
        cond_br(
            ctx,
            on_lhs_strided,
            rhs_cont,
            rhs_cont_only,
            vec![],
            neither,
            vec![],
        );

        func.get_operation().insert_at_back(module_block, ctx);
    });
    module.mark_kernel_entry(&op.name).unwrap();
}

/// One grid-stride loop over `numel`, with each operand indexed per its mode.
///
/// Returns the pre-header, the block a caller branches to in order to enter
/// this loop.
fn emit_loop(
    ctx: &mut Context,
    region: Ptr<Region>,
    tys: &Tys,
    op: &OpDesc,
    env: &Env,
    lhs_mode: IndexMode,
    rhs_mode: IndexMode,
) -> Ptr<BasicBlock> {
    let preheader = new_block(ctx, region, vec![]);
    let header = new_block(ctx, region, vec![tys.i32]);
    let body = new_block(ctx, region, vec![tys.i32]);
    let exit = new_block(ctx, region, vec![]);

    goto(ctx, preheader, header, vec![env.i0]);

    // header(i): `i < numel` compares a 32-bit index against a 64-bit count,
    // so widen first, matching C's integer promotion.
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

    // body(i): compute each operand's 32-bit index, then load, apply and store.
    let i = body.deref(ctx).get_argument(0);
    let (lhs_i, blk) = operand_index(ctx, region, tys, env, lhs_mode, i, env.lhs_strides, body);
    let (rhs_i, blk) = operand_index(ctx, region, tys, env, rhs_mode, i, env.rhs_strides, blk);

    // Widen to 64 bits only here, at the pointer arithmetic.
    let lhs_off = widen_u32_to_u64(ctx, blk, lhs_i);
    let rhs_off = widen_u32_to_u64(ctx, blk, rhs_i);
    let out_off = widen_u32_to_u64(ctx, blk, i);

    let lp = emit(
        ctx,
        blk,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![tys.in_ptr],
        vec![env.lhs, lhs_off],
    )
    .unwrap();
    let x = emit(
        ctx,
        blk,
        MirLoadOp::get_concrete_op_info(),
        vec![tys.f],
        vec![lp],
    )
    .unwrap();
    let rp = emit(
        ctx,
        blk,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![tys.in_ptr],
        vec![env.rhs, rhs_off],
    )
    .unwrap();
    let y = emit(
        ctx,
        blk,
        MirLoadOp::get_concrete_op_info(),
        vec![tys.f],
        vec![rp],
    )
    .unwrap();
    let value = emit_body(ctx, blk, tys, op, &[x, y]);
    let op_ptr = emit(
        ctx,
        blk,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![tys.out_ptr],
        vec![env.out, out_off],
    )
    .unwrap();
    emit(
        ctx,
        blk,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![op_ptr, value],
    );

    let i_next = emit(
        ctx,
        blk,
        MirAddOp::get_concrete_op_info(),
        vec![tys.i32],
        vec![i, env.step],
    )
    .unwrap();
    goto(ctx, blk, header, vec![i_next]);

    emit(
        ctx,
        exit,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
    );
    preheader
}

/// The 32-bit element index for one operand, and the block where the caller
/// carries on.
///
/// The contiguous mode is the identity on the induction variable and appends
/// no blocks, so the caller carries on where it was.
#[allow(clippy::too_many_arguments)]
fn operand_index(
    ctx: &mut Context,
    region: Ptr<Region>,
    tys: &Tys,
    env: &Env,
    mode: IndexMode,
    i: Value,
    strides: Value,
    block: Ptr<BasicBlock>,
) -> (Value, Ptr<BasicBlock>) {
    match mode {
        IndexMode::Contiguous => (i, block),
        IndexMode::Strided => {
            emit_strided_index(ctx, region, tys, i, env.num_dims, env.dims, strides, block)
        }
    }
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
