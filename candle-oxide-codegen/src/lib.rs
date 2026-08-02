//! Builds Candle's elementwise CUDA kernels through cuda-oxide's standalone
//! PTX API. Maintainer-only: the published artifact is the PTX this crate
//! writes, shipped by `candle-oxide-kernels`.

pub mod binary;
pub mod body;
pub mod desc;
pub mod emit;
pub mod fused;
pub mod index;
pub mod testing;
pub mod unary;

use cuda_oxide_codegen::experimental::{
    CodegenModule, CompileError, CompileOptions, Compiler, Linking, Target,
};
use desc::{Arity, Family, OpDesc};

/// Compile every entry in `ops` into one PTX module for `family`.
///
/// Linking is opted in whenever an entry names a libdevice symbol, which
/// #596 made possible. Contraction is set explicitly per family rather than
/// inherited, so parity comparisons are against like: see
/// [`Family::fma_contraction`].
pub fn build_family(family: Family, ops: &[OpDesc], target: &str) -> Result<String, CompileError> {
    let mut module = CodegenModule::new(match family {
        Family::Binary => "candle_oxide_binary",
        Family::Unary => "candle_oxide_unary",
        Family::Fused => "candle_oxide_fused",
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
            Arity::Unary => unary::build_unary(&mut module, op),
            Arity::Fused => fused::build_fused(&mut module, op),
        }
    }

    let needs_libdevice = ops.iter().any(|op| !op.libdevice_symbols().is_empty());
    let compiler = Compiler::discover().expect("LLVM 21+ llc and opt are installed");
    let options = CompileOptions::new(Target::parse(target).expect("valid target"))
        .with_fma_contraction(family.fma_contraction())
        .with_linking(if needs_libdevice {
            Linking::Libdevice
        } else {
            // Reached by a fused set holding neither
            // `fused_bias_gelu_erf_f32` nor `fused_scale_mask_softmax_f32`,
            // the only fused entries naming a symbol. The two elementwise
            // families always carry one (`bmaximum`/`bminimum` in binary;
            // `uexp`, `ulog`, `usin`, `ucos`, `usqrt` in unary), so a
            // `binary_ops()` or `unary_ops()` set never lands here.
            Linking::SelfContained
        });

    let ptx = compiler.compile(&mut module, &options)?.into_ptx();
    Ok(String::from_utf8(ptx).expect("PTX is utf-8"))
}
