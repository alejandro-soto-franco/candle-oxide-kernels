# candle-oxide-kernels

Elementwise CUDA kernels for Candle, generated as PTX by cuda-oxide.

This crate ships precompiled PTX for 51 entry points across three modules: `oxide::binary` (24 kernels for binary operations), `oxide::unary` (22 kernels for unary operations) and `oxide::fused` (5 entries whose index comes from the launch geometry). The PTX is generated offline and committed, so this crate carries no build dependencies and requires no nightly toolchain.

## The fused family

What `oxide::fused`'s entries share is that each takes its index from the launch geometry rather than computing it. Three also collapse a run of two elementwise kernels into one pass; the other two are a single traversal Candle already performs, run without an index computation. `row` below is the element offset of the block's row and `col` the column within it:

| entry | computes |
| --- | --- |
| `fused_scale_mask_f32` | `out[row + col] = (x[row + col] * scale + 0.0f) + mask[b * ss + col]` |
| `fused_bias_gelu_erf_f32` | `t = x[row + col] + bias[col]; out[row + col] = t * normcdf(t)` |
| `fused_bias_residual_f32` | `out[row + col] = (x[row + col] + bias[col]) + residual[row + col]` |
| `fused_bias_f32` | `out[row + col] = x[row + col] + bias[col]` |
| `fused_permute12_f32` | `out[((b * D2 + j) * D1 + i) * d + k] = src[((b * D1 + i) * D2 + j) * d + k]` |

Every operand must be contiguous and f32. There is no layout branching and no stride walk, which is what the contiguity precondition buys; a caller that cannot establish it evaluates the reference expression instead. `FUSED_ENTRIES` carries the full parameter list, the launch geometry and the index expression of each entry.

The launch is shaped like the tensor rather than flat: `ctaid.x` walks the last dimension, and `ctaid.y` and `ctaid.z` carry the row. Every index term is then a special register or the induction variable, and no entry divides. A flat element counter would need a division and a modulo per dimension to recover the coordinates, against divisors loaded from memory, and a division by a runtime value expands to roughly twenty instructions. That fixed per-element cost is what held the two chains moving eight bytes an element to 46 per cent of the measured 317 GB/s device ceiling while the one moving twelve reached 74; removing it took the worst of them to 92 per cent. `fused_scale_mask_f32` reads the head count from `nctaid.y`, so it takes neither an element count nor `hss`, and `fused_permute12_f32` reads both permuted extents from `nctaid.x` and `nctaid.y`.

`fused_permute12_f32` copies a rank-4 tensor whose logical shape is `[B, D2, D1, d]` over a contiguous source of shape `[B, D1, D2, d]`, and `fused_bias_f32` is the last-dimension broadcast add. Both displace a kernel Candle routes through `get_strided_index`, which walks `num_dims` deep, in the copy's case because the layout is permuted and in the add's because a broadcast carries a zero stride. `fused_permute12_f32` takes its column from `tid.x` alone, stepping by `ntid.x`, since `ctaid.x` names a permuted dimension and cannot also index the column.

CUDA caps `gridDim.y` and `gridDim.z` at 65535, so a caller whose head count, batch size, `D1` or row count exceeds that falls back. `ss` and `d` must be non-zero, as loop bounds.

This family compiles with FMA contraction off, where the elementwise families compile with it on to match nvcc's default. `fused_scale_mask_f32` replaces `affine_f32` followed by `badd_f32`, which rounds the product to memory before it adds the mask, so contracting the multiply and the add into one FMA would round once where the path being replaced rounds twice. The other four entries carry no contraction site.

`fused_scale_mask_f32` carries a `+ 0.0f` because `affine_f32`'s body is `x * mul + add`, which Candle reaches with `add` zero through `Tensor::affine(scale, 0.0)`. That add changes one value: a product of `-0.0` becomes `+0.0`, since `-0.0 + 0.0` is `+0.0` under round-to-nearest. Omitting it would disagree with the reference whenever the mask element is `-0.0` as well. The entry therefore rounds three times, and the test suite pins the instruction counts so a fold enabled by a later `nsz` or relaxed floating-point flag surfaces as a failure rather than as numeric drift.

## Architecture

The committed PTX targets `sm_75` (Turing). Every load goes through the CUDA driver's PTX-to-SASS JIT regardless of GPU architecture, since this crate ships PTX text as source. PTX is forward-compatible, so the driver JITs the same module up to whatever architecture the running GPU implements.

`sm_75` and `sm_80` produce byte-identical instruction streams for this kernel set; the two PTX files differ only in their `.version` and `.target` header lines. The baseline is therefore a compatibility declaration, making no code-generation difference, and `sm_75` was picked for hardware coverage back to Turing at no cost. `cargo xtask gen-ptx --arch <target>` regenerates the PTX for any architecture without changing this crate's shape.

Every pointer parameter carries a `.ptr .global .align 1` annotation, the most conservative alignment declaration possible. Neither this pipeline nor nvcc vectorises loads for these entries, so the annotation costs nothing: there is no wider load for it to inhibit.

## Verification

The figures below cover the 46 elementwise entries. The five fused entries have no nvcc
counterpart to diff against, so their host-side checks are the declared ABI, the libdevice
symbol the GELU entry resolves to, the absence of a contracted multiply-add in
`fused_scale_mask_f32`, the absence of any division in any of them, the grid coordinates
each reads, and `ptxas` acceptance at the shipped baseline and at `sm_120`. Numeric parity
for those five runs on a GPU, in the consuming Candle checkout.

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
