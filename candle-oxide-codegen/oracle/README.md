# Oracle PTX

One file per entry, sliced from the module `candle-oxide-kernel` emits. Each
file holds that entry's `.visible .entry` body verbatim, preceded by the
module's `.version` / `.target` / `.address_size` lines. `tests/oracle.rs`
reads them, so regenerating is a diff.

These entries are written in Rust and lowered through cuda-oxide's rustc
`#[kernel]` frontend, a different route from the standalone MIR builder this
crate uses. Agreement on the arithmetic is therefore evidence from two
independent frontends, not one pipeline checked against itself.

## Provenance

- Source: `$ORACLE_SRC`, branch `feat/oracle-elementwise`,
  `src/main.rs`.
- Toolchain: `nightly-2026-04-03` (`rustc 1.96.0-nightly 55e86c996`,
  LLVM 22.1.2), pinned by that crate's `rust-toolchain.toml`.
- cuda-oxide revision: `0810ba95`, the same revision this crate builds
  against. `candle-oxide-kernel` takes `cuda-device`, `cuda-host` and
  `cuda-core` by path from `$CUDA_OXIDE`, a detached worktree at that
  revision.
- Codegen backend: `librustc_codegen_cuda.so` built from
  `$CUDA_OXIDE/crates/rustc-codegen-cuda` at the same revision and
  passed through `CUDA_OXIDE_BACKEND`. The shared cache at
  `~/.cargo/cuda-oxide/` holds a build from an unrelated branch and must not
  be used for this: the two revisions disagree on five of the thirty entries,
  which would make any comparison against them unreadable.
- Target: `sm_80`, matching `build_family`'s target in the test.

## Command

```bash
cd $CUDA_OXIDE/crates/rustc-codegen-cuda
CARGO_TARGET_DIR=<backend target directory> \
LIBRARY_PATH="$(rustc --print sysroot)/lib" \
LD_LIBRARY_PATH="$(rustc --print sysroot)/lib" \
  cargo build --lib --target x86_64-unknown-linux-gnu

cd $ORACLE_SRC
CUDA_OXIDE_BACKEND=<backend target directory>/x86_64-unknown-linux-gnu/debug/librustc_codegen_cuda.so \
  cargo oxide build --arch sm_80
# writes candle_oxide_kernel.ptx at the crate root; scripts/split_oracle.py
# slices it into this directory.
python3 $ORACLE_SRC/scripts/split_oracle.py
```

## Coverage

Thirty entries: `ucopy`, `uneg`, `usqr`, `uabs`, `usqrt`, `uexp`, `ulog`,
`usin`, `ucos`, `badd`, `bsub`, `bmul`, `bdiv`, `bmaximum` and `bminimum`, at
f32 and f64.

`utanh`, `urecip` and the twelve comparison entries have no oracle. Rust
routes `f32::tanh` through libm rather than a `core::intrinsics` placeholder,
Candle computes the reciprocal in double for both dtypes, and the
comparisons' byte-store form has no natural Rust surface. Those entries rest
on the nvcc parameter diff in `tests/abi_diff.rs` and on the body assertions
in `tests/binary_bodies.rs`.
