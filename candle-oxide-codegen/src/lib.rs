//! Builds Candle's elementwise CUDA kernels through cuda-oxide's standalone
//! PTX API. Maintainer-only: the published artifact is the PTX this crate
//! writes, shipped by `candle-oxide-kernels`.

pub mod desc;
