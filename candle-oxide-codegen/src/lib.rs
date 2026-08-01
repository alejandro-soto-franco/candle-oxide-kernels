//! Builds Candle's elementwise CUDA kernels through cuda-oxide's standalone
//! PTX API. Maintainer-only: the published artifact is the PTX this crate
//! writes, shipped by `candle-oxide-kernels`.

pub mod binary;
pub mod body;
pub mod desc;
pub mod emit;

use cuda_oxide_codegen::experimental::{
    CodegenModule, CompileError, CompileOptions, Compiler, Linking, Target,
};
use desc::{Arity, Family, OpDesc};

/// Compile every entry in `ops` into one PTX module for `family`.
///
/// Linking is opted in whenever an entry names a libdevice symbol, which
/// #596 made possible. Contraction is set explicitly to nvcc's default so
/// parity comparisons are against like, never inherited.
pub fn build_family(family: Family, ops: &[OpDesc], target: &str) -> Result<String, CompileError> {
    let mut module = CodegenModule::new(match family {
        Family::Binary => "candle_oxide_binary",
        Family::Unary => "candle_oxide_unary",
    })
    .expect("module name is valid");

    for op in ops {
        assert_eq!(
            op.family(),
            family,
            "{} belongs to {:?}, not {family:?}",
            op.name,
            op.family()
        );
        match op.arity {
            Arity::Binary => binary::build_binary(&mut module, op),
            Arity::Unary => unimplemented!("unary arrives in Task 6"),
        }
    }

    let needs_libdevice = ops.iter().any(|op| op.libdevice_name().is_some());
    let compiler = Compiler::discover().expect("LLVM 21+ llc and opt are installed");
    let options = CompileOptions::new(Target::parse(target).expect("valid target"))
        .with_fma_contraction(true)
        .with_linking(if needs_libdevice { Linking::Libdevice } else { Linking::SelfContained });

    let ptx = compiler.compile(&mut module, &options)?.into_ptx();
    Ok(String::from_utf8(ptx).expect("PTX is utf-8"))
}
