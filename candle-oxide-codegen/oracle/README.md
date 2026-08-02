# Oracle PTX

One file per entry, sliced from the module `candle-oxide-oracle` emits. Each
file holds that entry's `.visible .entry` body verbatim, preceded by the
module's `.version` / `.target` / `.address_size` lines. `tests/oracle.rs`
reads them, so regenerating is a diff.

These entries are written in Rust and lowered through cuda-oxide's rustc
`#[kernel]` frontend, a different route from the standalone MIR builder this
crate uses.

What agreement establishes is narrower than "two independent derivations", and
it varies by op. The two routes share LLVM 22 and, for every libdevice-backed
op, they share the libdevice bitcode itself: the builder names `__nv_expf`,
`__nv_logf`, `__nv_sinf`, `__nv_cosf`, `__nv_sqrtf` and `__nv_fabsf` directly,
the oracle reaches the same symbols through Rust intrinsic placeholders, and
the same LLVM inlines the same bitcode on both sides. A 22-to-35 instruction
match on `usin`, `ucos`, `uexp` or `ulog` therefore confirms that both routes
picked the same libdevice symbol, which is the real risk in the op table, and
carries no information about the polynomial itself, which is one shared
artefact. `ulog_f32.ptx` holds no `lg2` on either side, only that shared
inlined expansion.

Agreement is evidence about symbol selection, about the operation each entry
performs, and about the surrounding grid-stride loop and ABI. For the ops
whose arithmetic the builder emits itself (`badd`, `bsub`, `bmul`, `bdiv`,
`ucopy`, `uneg`, `usqr`) it is evidence about the arithmetic too.

## Provenance

- Source: `$ORACLE_SRC` (a checkout of `candle-oxide-oracle`), `src/main.rs`,
  at commit `91334cc` on branch `feat/oracle-elementwise`.
- Toolchain: `nightly-2026-04-03` (`rustc 1.96.0-nightly 55e86c996`,
  LLVM 22.1.2), pinned by that crate's `rust-toolchain.toml`.
- cuda-oxide revision: `0810ba95`, the same revision this crate builds
  against. `candle-oxide-oracle` takes `cuda-device`, `cuda-host` and
  `cuda-core` by path from `$CUDA_OXIDE` (a checkout of cuda-oxide at that
  revision, kept outside this repository).
- Codegen backend: `librustc_codegen_cuda.so` built from
  `$CUDA_OXIDE/crates/rustc-codegen-cuda` at the same revision and passed
  through `CUDA_OXIDE_BACKEND`. A backend cache shared with another branch
  must not be used for this: the two revisions disagree on five of the
  thirty entries, which would make any comparison against them unreadable.
  Build the backend fresh into a target directory dedicated to this
  revision.
- Target: `sm_80`, matching `build_family`'s target in the test.

## Command

```bash
cd "$CUDA_OXIDE/crates/rustc-codegen-cuda"
CARGO_TARGET_DIR=<a target directory dedicated to this cuda-oxide revision> \
LIBRARY_PATH="$(rustc --print sysroot)/lib" \
LD_LIBRARY_PATH="$(rustc --print sysroot)/lib" \
  cargo build --lib --target x86_64-unknown-linux-gnu

cd "$ORACLE_SRC"
CUDA_OXIDE_BACKEND=<that target directory>/x86_64-unknown-linux-gnu/debug/librustc_codegen_cuda.so \
  cargo oxide build --arch sm_80
# writes candle_oxide_oracle.ptx at the crate root; scripts/split_oracle.py
# slices it into this directory.
python3 "$ORACLE_SRC/scripts/split_oracle.py"
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
