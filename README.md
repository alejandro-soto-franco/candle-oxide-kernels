# candle-oxide

Candle's elementwise CUDA kernels as PTX, generated through
[cuda-oxide](https://github.com/NVlabs/cuda-oxide) and checked bit-exact
against Candle's own nvcc-built kernels.

## Crates

- **`candle-oxide-kernels`** is the published crate. It ships 49 precompiled
  entry points across three PTX modules (`oxide::binary`, `oxide::unary`,
  `oxide::fused`), carries no build dependencies, and needs no nightly
  toolchain. See `candle-oxide-kernels/README.md`.
- **`candle-oxide-codegen`** builds the PTX from an op table through
  cuda-oxide's standalone MIR API. Maintainer-only; not published.
- **`xtask`** regenerates the committed PTX (`cargo xtask gen-ptx`).
  Maintainer-only; not published.

## Regenerating the PTX

```bash
cargo xtask gen-ptx
```

Writes `candle-oxide-kernels/ptx/binary.ptx`, `unary.ptx` and `fused.ptx` at
the shipped baseline (`sm_75`). Building `candle-oxide-codegen` and `xtask` needs LLVM
21+ (`llc`, `opt`) but no CUDA toolkit or GPU; the reproducibility,
parameter-shape and `ptxas` assembly checks under `cargo test --workspace`
do need a CUDA toolkit (no GPU) and are what CI's `codegen` job runs.

## License

Licensed under either of Apache License, Version 2.0 or MIT license, at
your option.
