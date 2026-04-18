//! `lib/common/` translation: shared primitives (bits, mem, errors, entropy).

pub mod bits;
pub mod bitstream;
pub mod debug;
pub mod entropy_common;
pub mod error;
pub mod fse_decompress;
pub mod mem;
#[cfg(feature = "std")]
pub mod pool;
#[cfg(feature = "std")]
pub mod threading;
pub mod xxhash;
pub mod zstd_common;
pub mod zstd_internal;
