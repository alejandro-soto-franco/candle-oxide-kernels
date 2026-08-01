# candle-oxide-kernels

Elementwise CUDA kernels for Candle, generated as PTX by cuda-oxide.

This crate ships precompiled PTX for 46 entry points across two modules: `oxide::binary` (24 kernels for binary operations) and `oxide::unary` (22 kernels for unary operations). The PTX is generated offline and committed, so this crate carries no build dependencies and requires no nightly toolchain.

## Architecture

The committed PTX targets `sm_75` (Turing). Every load goes through the CUDA driver's PTX-to-SASS JIT regardless of GPU architecture, since this crate ships PTX text as source. PTX is forward-compatible, so the driver JITs the same module up to whatever architecture the running GPU implements.

`sm_75` and `sm_80` produce byte-identical instruction streams for this kernel set; the two PTX files differ only in their `.version` and `.target` header lines. The baseline is therefore a compatibility declaration, making no code-generation difference, and `sm_75` was picked for hardware coverage back to Turing at no cost. `cargo xtask gen-ptx --arch <target>` regenerates the PTX for any architecture without changing this crate's shape.

Every pointer parameter carries a `.ptr .global .align 1` annotation, the most conservative alignment declaration possible. Neither this pipeline nor nvcc vectorises loads for these entries, so the annotation costs nothing: there is no wider load for it to inhibit.

## Limitations

`ucopy_f32` and `ucopy_f64` are declared in `UNARY_ENTRIES` and present in `unary.ptx`, but Candle's tensor API never reaches them: `copy_strided_src` loads `ucopy` directly from Candle's stock kernel module. These two entries are unreachable through the tensor API today.

## Loading

Load a module through `CudaDevice::get_or_load_custom_func`, keyed by the module name:

```rust
use candle_oxide_kernels::{BINARY_MODULE, BINARY};

let module = device.get_or_load_custom_func(BINARY_MODULE, BINARY)?;
let kernel = module.get_function("badd_f32")?;
```

Do not route through `candle_kernels::Module`, whose index is positional into a fixed array and would collide.

## Generated from

cuda-oxide revision `0810ba95ff9dcf1ca379b779d3ebe33a8899b15d`.
