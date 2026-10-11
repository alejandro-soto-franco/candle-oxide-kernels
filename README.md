# candle-oxide-kernels

Candle's elementwise CUDA kernels as PTX, generated through
[cuda-oxide](https://github.com/NVIDIA/cuda-rust/tree/main/cuda-oxide) and checked bit-exact against
Candle's own nvcc-built kernels.

The published crate ships 52 precompiled entry points across three PTX modules.
It carries no build dependencies, needs no nightly toolchain, and needs neither a
CUDA toolkit nor a GPU to build.

## Usage

Each module is a `&'static str` of PTX text, loaded through Candle's custom
module path and keyed by module name:

```rust
use candle_oxide_kernels::{BINARY, BINARY_MODULE};

let module = device.get_or_load_custom_func(BINARY_MODULE, BINARY)?;
let kernel = module.get_function("badd_f32")?;
```

`BINARY_ENTRIES`, `UNARY_ENTRIES` and `FUSED_ENTRIES` list the entry points of
each module. `has_entry` and `ptx_for` resolve a module and entry by name.

Routing through `candle_kernels::Module` collides, since that index is positional
into a fixed array.

## Verification

- 63 parity cases against Candle's stock kernels at difference exactly 0.0, with
  no tolerance, covering both dtypes, all four binary layout paths and both unary
  ones, contiguous and strided.
- All 46 elementwise entries compared against their nvcc-built counterparts on
  hardware over 64 special values, covering NaN, both infinities, signed zeros,
  subnormals, the extremes of the float range and the Payne-Hanek argument
  reduction trigger points. Zero bit differences.
- Candle's own test suite passes with these kernels substituted for its own.
- Routing confirmed by mutation: corrupting a single entry in the committed PTX
  fails exactly the cases that exercise it, at each of the three dispatch sites
  Candle loads through.

Measured against Candle's stock kernels, these run about 4 per cent slower on the
contiguous path and about 4 per cent faster on the strided one. See
`candle-oxide-kernels/README.md` for the measurement and for what accounts for
each figure.

## Crates

| Crate | Role |
| --- | --- |
| `candle-oxide-kernels` | the published crate, shipping the committed PTX |
| `candle-oxide-codegen` | builds the PTX from an op table through cuda-oxide's standalone MIR API. Maintainer-only |
| `xtask` | regenerates the committed PTX. Maintainer-only |

## Regenerating the PTX

```bash
cargo xtask gen-ptx
```

Writes `candle-oxide-kernels/ptx/binary.ptx`, `unary.ptx` and `fused.ptx` at the
shipped `sm_75` baseline. `--arch <target>` regenerates for another architecture.

`sm_75` is a compatibility declaration and makes no code-generation difference:
`sm_75` and `sm_80` produce byte-identical instruction streams for this kernel
set, and the CUDA driver JITs the shipped PTX up to whatever architecture the
running GPU implements.

Building `candle-oxide-codegen` and `xtask` needs LLVM 21 or later (`llc`,
`opt`). The reproducibility, parameter-shape and `ptxas` assembly checks under
`cargo test --workspace` additionally need a CUDA toolkit, though no GPU.

## Licence

Licensed under either of Apache License, Version 2.0 or MIT license, at your
option.
