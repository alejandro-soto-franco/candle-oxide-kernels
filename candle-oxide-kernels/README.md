# candle-oxide-kernels

Elementwise CUDA kernels for Candle, generated as PTX by cuda-oxide.

This crate ships precompiled PTX for 46 entry points across two modules: `oxide::binary` (24 kernels for binary operations) and `oxide::unary` (22 kernels for unary operations). The PTX is generated offline and committed, so this crate carries no build dependencies and requires no nightly toolchain.

## Architecture

The committed PTX targets `sm_75` (Turing). Every load goes through the CUDA driver's PTX-to-SASS JIT regardless of GPU architecture, since this crate ships PTX text rather than a cubin; PTX is forward-compatible, so the driver JITs the same module up to whatever architecture the running GPU implements.

Measurement in `docs/arch-baseline.md` found no measurable difference in first-load JIT cost between `sm_75`, `sm_80` and the running GPU's own architecture, so `sm_75` was chosen for hardware coverage back to Turing at the same cost. `cargo xtask gen-ptx --arch <target>` regenerates the PTX for any architecture without changing this crate's shape.

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
