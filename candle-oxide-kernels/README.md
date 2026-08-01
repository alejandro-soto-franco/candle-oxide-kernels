# candle-oxide-kernels

Elementwise CUDA kernels for Candle, generated as PTX by cuda-oxide.

This crate ships precompiled PTX for 46 entry points across two modules: `oxide::binary` (24 kernels for binary operations) and `oxide::unary` (22 kernels for unary operations). The PTX is generated offline and committed, so this crate carries no build dependencies and requires no nightly toolchain.

## Architecture

The committed PTX targets `sm_80` (Ampere). The CUDA driver automatically JITs to newer architectures when loaded on more recent devices, and the oldest supported Ampere hardware runs the binary natively.

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
