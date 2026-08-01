# PTX baseline architecture

The committed PTX under `candle-oxide-kernels/ptx/` targets `sm_80`, a
placeholder set before this measurement existed. `sm_75` was the other
candidate. PTX is forward-compatible: the CUDA driver JITs a module built at
an older baseline up to whatever architecture the running GPU implements, so
the question is not correctness but the JIT cost paid on first load of a
module built below the GPU's own architecture.

## Method

Measured on the development box: an RTX 5060 Laptop GPU, compute capability
12.0 (`sm_120`), CUDA 13.3.

Three PTX sets were generated with `cargo xtask gen-ptx` into scratch
directories, one per candidate baseline plus the GPU's own architecture as a
no-JIT reference:

```
cargo xtask gen-ptx --arch sm_75  --out <scratch>/ptx-75
cargo xtask gen-ptx --arch sm_80  --out <scratch>/ptx-80
cargo xtask gen-ptx --arch sm_120 --out <scratch>/ptx-120
```

None of these writes touched `candle-oxide-kernels/ptx/`.

A standalone binary (`jit-bench`, cudarc 0.19.8 only, no Candle dependency)
calls `CudaContext::new` once, then times two calls to
`CudaContext::load_module` with `Instant`: one for `binary.ptx`, one for
`unary.ptx`. This is the exact call `candle-core`'s
`get_or_load_custom_func` makes (`self.context.load_module(ptx.into())` in
`candle-core/src/cuda_backend/device.rs`), reached without pulling in
Candle's own build. Context creation happens before the timer starts, so the
measurement isolates module load, not device or driver initialisation.

Each sample is a fresh process invocation with `CUDA_CACHE_DISABLE=1`, so
every sample is a genuine JIT rather than a hit against the on-disk ptxas
JIT cache (`~/.nv/ComputeCache`), which would otherwise serve every run
after the first from disk and hide the cost the measurement exists to find.
Three samples per baseline; medians reported.

## Results (median of 3, ms)

| baseline | binary.ptx load | unary.ptx load |
|---|---|---|
| `sm_75`  | 763.7 | 275.3 |
| `sm_80`  | 760.6 | 273.2 |
| `sm_120` | 774.2 | 272.2 |

Raw samples:

```
sm_75:  binary 749.7 774.4 763.7   unary 282.1 275.3 264.0
sm_80:  binary 760.6 768.6 760.2   unary 283.1 265.5 273.2
sm_120: binary 768.3 774.2 796.7   unary 272.2 267.6 272.4
```

## Reading

The spread between baselines (roughly 10-15 ms on a ~1035 ms combined
median) is smaller than the run-to-run spread within a single baseline
(roughly 25-30 ms for `sm_75` binary alone). There is no measurable JIT
cost difference between `sm_75`, `sm_80` and `sm_120` on this driver and
this GPU, including against the GPU's own architecture. The driver still
runs ptxas over the PTX text on every load regardless of the baseline
recorded in the PTX header; the shipped artefact holds no cubin, so there
is no "no JIT" case to compare against here, only JIT-from-different-headers,
and the header does not move the cost.

Given a negligible measured difference, breadth of hardware support decides
the question rather than a fabricated performance argument.

## Decision

`sm_75` (Turing and later) is the better baseline: it covers three more
hardware generations than `sm_80` (Ampere and later) at no measured JIT
cost. `candle-oxide-kernels/ptx/` still ships `sm_80` PTX as of this
measurement; adopting `sm_75` is a one-command regeneration,
`cargo xtask gen-ptx --arch sm_75 --out candle-oxide-kernels/ptx`, followed
by the full test and CI reproducibility check. That regeneration is left
for a separate commit so this task's diff stays documentation-only against
the artefact Task 13 already validated (31 tests, GPU parity at difference
0.0, dispatch-site mutation coverage). The xtask can emit any architecture
on demand, so nothing about the crate's shape depends on which baseline is
currently committed.

## The `.align 1` pointer annotation

Every pointer `.param` the builder emits carries `.ptr .global .align 1`;
nvcc emits no pointer annotation at all for the same parameter. `.align 1`
declares the pointee alignment as one byte, the most conservative
declaration possible. It cannot produce an incorrect result; the parity
suite already proves bit-exactness. The open question was whether it
inhibits wider vectorised loads that nvcc's unannotated form would permit.

Compared for `ucopy_f32` and `badd_f32` at `sm_80`, both compiled with
`--allow-unsupported-compiler` against the same `gcc16_cuda_compat.h`
pre-include used by the existing ABI diff test:

- **PTX load width.** Both the builder's PTX and nvcc's PTX load the f32
  data operand with a scalar instruction (`ld.global.b32` / `ld.global.f32`)
  in both entries. Neither side ever emits a `.v2` or `.v4` vectorised load
  for the data pointer. (The `.b64`/`.u64` loads present on both sides load
  the `usize` count and stride parameters, not element data.)
- **SASS.** `ptxas -arch=sm_80` on both PTX files, followed by
  `cuobjdump --dump-sass`, shows the same picture: `ucopy_f32` uses `LDG.E`
  (32-bit) and `LDG.E.64` on both sides, never `LDG.E.128`.

nvcc's own frontend, with no alignment annotation on the pointer parameter
at all, does not vectorise these loads either. The `.align 1` annotation
cannot be responsible for a vectorisation nvcc itself never performs on
these entries: there is nothing to inhibit. No cost is demonstrated. The
question is closed as measured, not fixed; the builder is unmodified, and
the committed PTX is unmodified.

This does not rule out a cost on a kernel where nvcc does vectorise (a
wider, more heavily unrolled elementwise op might cross that threshold).
That would need its own measurement against its own nvcc baseline and is
not something this investigation's two-entry sample can speak to.
