# candle-oxide-kernels

Elementwise CUDA kernels for Candle, generated as PTX by cuda-oxide.

This crate ships precompiled PTX for 49 entry points across three modules: `oxide::binary` (24 kernels for binary operations), `oxide::unary` (22 kernels for unary operations) and `oxide::fused` (3 fused chains). The PTX is generated offline and committed, so this crate carries no build dependencies and requires no nightly toolchain.

## Fused chains

`oxide::fused` holds three f32 entries, each collapsing a run of two elementwise kernels into one grid-stride pass:

| entry | computes |
| --- | --- |
| `fused_scale_mask_f32` | `out[i] = (x[i] * scale + 0.0f) + mask[(i / hss) * ss + (i % ss)]` |
| `fused_bias_gelu_erf_f32` | `t = x[i] + bias[i % d]; out[i] = t * normcdf(t)` |
| `fused_bias_residual_f32` | `out[i] = (x[i] + bias[i % d]) + residual[i]` |

Every operand must be contiguous and f32. There is no layout branching and no stride walk, which is what the contiguity precondition buys; a caller that cannot establish it evaluates the reference expression instead. `FUSED_ENTRIES` carries the full parameter list of each entry.

This family compiles with FMA contraction off, where the elementwise families compile with it on to match nvcc's default. `fused_scale_mask_f32` replaces `affine_f32` followed by `badd_f32`, which rounds the product to memory before it adds the mask, so contracting the multiply and the add into one FMA would round once where the path being replaced rounds twice. The other two chains carry no contraction site.

`fused_scale_mask_f32` carries a `+ 0.0f` because `affine_f32`'s body is `x * mul + add`, which Candle reaches with `add` zero through `Tensor::affine(scale, 0.0)`. That add changes one value: a product of `-0.0` becomes `+0.0`, since `-0.0 + 0.0` is `+0.0` under round-to-nearest. Omitting it would disagree with the reference whenever the mask element is `-0.0` as well. The entry therefore rounds three times, and the test suite pins the instruction counts so a fold enabled by a later `nsz` or relaxed floating-point flag surfaces as a failure rather than as numeric drift.

## Architecture

The committed PTX targets `sm_75` (Turing). Every load goes through the CUDA driver's PTX-to-SASS JIT regardless of GPU architecture, since this crate ships PTX text as source. PTX is forward-compatible, so the driver JITs the same module up to whatever architecture the running GPU implements.

`sm_75` and `sm_80` produce byte-identical instruction streams for this kernel set; the two PTX files differ only in their `.version` and `.target` header lines. The baseline is therefore a compatibility declaration, making no code-generation difference, and `sm_75` was picked for hardware coverage back to Turing at no cost. `cargo xtask gen-ptx --arch <target>` regenerates the PTX for any architecture without changing this crate's shape.

Every pointer parameter carries a `.ptr .global .align 1` annotation, the most conservative alignment declaration possible. Neither this pipeline nor nvcc vectorises loads for these entries, so the annotation costs nothing: there is no wider load for it to inhibit.

## Verification

The figures below cover the 46 elementwise entries. The three fused chains have no nvcc
counterpart to diff against, so their host-side checks are the declared ABI, the libdevice
symbol the GELU chain resolves to, the absence of a contracted multiply-add in
`fused_scale_mask_f32`, and `ptxas` acceptance at the shipped baseline and at `sm_120`.
Numeric parity for those three runs on a GPU, in the consuming Candle checkout.

These kernels are bit-exact against Candle's own nvcc-built kernels. The parity suite in the
consuming Candle checkout runs 63 cases at difference exactly 0.0, with no tolerance, covering
both dtypes, all four binary layout paths and both unary ones, contiguous and strided, at 4096
and at 4194304 elements, including a deliberately undersized launch grid so the grid-stride
loop wraps.

Every one of the 46 shipped entries was additionally compared against its nvcc-built
counterpart on hardware over 64 special values: NaN, positive and negative infinity, signed
zeros, subnormals, the extremes of the float range, and the Payne-Hanek argument-reduction
trigger points for the trigonometric entries. Zero bit differences.

Candle's own test suite passes with these kernels substituted for its own.

Routing was confirmed by mutation rather than by inspection. Corrupting a single entry in the
committed PTX fails exactly the cases that exercise it, at each of the three dispatch sites
Candle loads through.

## Performance

Measured on an RTX 5060 Laptop against Candle's stock kernels, negating 4194304 f32 elements,
three interleaved runs, medians:

| Layout | Stock | These kernels |
| --- | --- | --- |
| Contiguous | 59.9 us | 62.3 us |
| Strided | 251.5 us | 239.2 us |

About 4 per cent slower on the contiguous path and about 4 per cent faster on the strided one.
Both figures reproduce across runs.

The contiguous difference comes from the launch geometry rather than from the loop. Candle
launches one thread per element, so each thread executes the loop body once and the prologue
dominates: 36 instructions against 35 for the stock kernel, with 31 registers against 26 and no
occupancy consequence. The strided gain comes from a leaner index decomposition.

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
