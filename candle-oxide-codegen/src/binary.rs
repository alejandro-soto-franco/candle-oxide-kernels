//! The six-word binary entry builder.

use crate::body::emit_body;
use crate::desc::OpDesc;
use crate::emit::{emit, module_block, widen_u32_to_u64, Tys};
use cuda_oxide_codegen::experimental::CodegenModule;
use dialect_mir::ops::{
    MirAddOp, MirCondBranchOp, MirFuncOp, MirGotoOp, MirLoadOp, MirLtOp, MirMulOp,
    MirPtrOffsetOp, MirReturnOp, MirStoreOp,
};
use dialect_nvvm::ops::{
    ReadPtxSregCtaidXOp, ReadPtxSregNctaidXOp, ReadPtxSregNtidXOp, ReadPtxSregTidXOp,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::TypeAttr,
        op_interfaces::{OperandSegmentInterface, SymbolOpInterface},
        types::FunctionType,
    },
    op::Op,
    operation::Operation,
};

/// Append `op`'s entry point to `module`, contiguous path only.
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
            let fop = Operation::new(ctx, MirFuncOp::get_concrete_op_info(), vec![], vec![], vec![], 1);
            let func = MirFuncOp::new(ctx, fop, TypeAttr::new(func_type.into()));
            func.set_symbol_name(ctx, name.as_str().try_into().unwrap());
            func
        };
        let func_region = func.get_operation().deref(ctx).get_region(0);

        let entry = BasicBlock::new(ctx, None, params);
        // The induction variable is 32-bit, matching Candle's `unsigned int i`.
        let header = BasicBlock::new(ctx, None, vec![tys.i32]);
        let body = BasicBlock::new(ctx, None, vec![tys.i32]);
        let exit = BasicBlock::new(ctx, None, vec![]);
        entry.insert_at_front(func_region, ctx);
        header.insert_at_back(func_region, ctx);
        body.insert_at_back(func_region, ctx);
        exit.insert_at_back(func_region, ctx);

        let (numel, lhs, rhs, out) = {
            let b = entry.deref(ctx);
            (b.get_argument(0), b.get_argument(3), b.get_argument(4), b.get_argument(5))
        };

        let tid = emit(ctx, entry, ReadPtxSregTidXOp::get_concrete_op_info(), vec![tys.i32], vec![]).unwrap();
        let ctaid = emit(ctx, entry, ReadPtxSregCtaidXOp::get_concrete_op_info(), vec![tys.i32], vec![]).unwrap();
        let ntid = emit(ctx, entry, ReadPtxSregNtidXOp::get_concrete_op_info(), vec![tys.i32], vec![]).unwrap();
        let nctaid = emit(ctx, entry, ReadPtxSregNctaidXOp::get_concrete_op_info(), vec![tys.i32], vec![]).unwrap();

        let base = emit(ctx, entry, MirMulOp::get_concrete_op_info(), vec![tys.i32], vec![ctaid, ntid]).unwrap();
        let i0 = emit(ctx, entry, MirAddOp::get_concrete_op_info(), vec![tys.i32], vec![base, tid]).unwrap();
        let stride = emit(ctx, entry, MirMulOp::get_concrete_op_info(), vec![tys.i32], vec![ntid, nctaid]).unwrap();

        let goto = Operation::new(ctx, MirGotoOp::get_concrete_op_info(), vec![], vec![i0], vec![header], 0);
        goto.insert_at_back(entry, ctx);

        // header(i): `i < numel` compares a 32-bit index against a 64-bit
        // count, so widen first, matching C's integer promotion.
        let i_hdr = header.deref(ctx).get_argument(0);
        let i_wide = widen_u32_to_u64(ctx, header, i_hdr);
        let cond = emit(ctx, header, MirLtOp::get_concrete_op_info(), vec![tys.i1], vec![i_wide, numel]).unwrap();
        let (flat, segs) = MirCondBranchOp::compute_segment_sizes(vec![vec![cond], vec![i_hdr], vec![]]);
        let br = Operation::new(ctx, MirCondBranchOp::get_concrete_op_info(), vec![], flat, vec![body, exit], 0);
        Operation::get_op::<MirCondBranchOp>(br, ctx).unwrap().set_operand_segment_sizes(ctx, segs);
        br.insert_at_back(header, ctx);

        // body(i): out[i] = FUNC(lhs[i], rhs[i]); goto header(i + stride).
        let i_body = body.deref(ctx).get_argument(0);
        let idx = widen_u32_to_u64(ctx, body, i_body);
        let lp = emit(ctx, body, MirPtrOffsetOp::get_concrete_op_info(), vec![tys.in_ptr], vec![lhs, idx]).unwrap();
        let x = emit(ctx, body, MirLoadOp::get_concrete_op_info(), vec![tys.f], vec![lp]).unwrap();
        let rp = emit(ctx, body, MirPtrOffsetOp::get_concrete_op_info(), vec![tys.in_ptr], vec![rhs, idx]).unwrap();
        let y = emit(ctx, body, MirLoadOp::get_concrete_op_info(), vec![tys.f], vec![rp]).unwrap();
        let value = emit_body(ctx, body, &tys, op, &[x, y]);
        let op_ptr = emit(ctx, body, MirPtrOffsetOp::get_concrete_op_info(), vec![tys.out_ptr], vec![out, idx]).unwrap();
        emit(ctx, body, MirStoreOp::get_concrete_op_info(), vec![], vec![op_ptr, value]);
        let i_next = emit(ctx, body, MirAddOp::get_concrete_op_info(), vec![tys.i32], vec![i_body, stride]).unwrap();
        let back = Operation::new(ctx, MirGotoOp::get_concrete_op_info(), vec![], vec![i_next], vec![header], 0);
        back.insert_at_back(body, ctx);

        emit(ctx, exit, MirReturnOp::get_concrete_op_info(), vec![], vec![]);
        func.get_operation().insert_at_back(module_block, ctx);
    });
    module.mark_kernel_entry(&op.name).unwrap();
}
