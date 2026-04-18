//! Translation of `lib/compress/zstdmt_compress.c`. Multi-threaded
//! compression. Behind the `mt` feature.
//!
//! Current v0.1 provides the API surface as returnable errors /
//! no-ops so the `mt` feature can compile without triggering panics.
//! A real port would use `rayon` for parallel block compression, but
//! the sequential path already handles all correctness requirements —
//! MT is purely a speed optimization.

#![allow(unused_variables)]

use core::marker::PhantomData;

pub struct ZSTDMT_CCtx {
    _priv: PhantomData<()>,
}

/// Port of `ZSTDMT_createCCtx`. v0.1 returns `None` until the rayon-
/// based worker pool lands; callers should fall back to the
/// single-threaded `ZSTD_createCCtx`.
pub fn ZSTDMT_createCCtx(_nbWorkers: u32) -> Option<Box<ZSTDMT_CCtx>> {
    None
}

/// Port of `ZSTDMT_freeCCtx`. Drops the Box; returns 0.
pub fn ZSTDMT_freeCCtx(_mtctx: Option<Box<ZSTDMT_CCtx>>) -> usize {
    0
}

/// Port of `ZSTDMT_compressStream_generic`. Returns
/// `ErrorCode::Generic` until the real MT pipeline is ported.
pub fn ZSTDMT_compressStream_generic(
    _mtctx: &mut ZSTDMT_CCtx,
    _output: &mut [u8],
    _output_pos: &mut usize,
    _input: &[u8],
    _input_pos: &mut usize,
    _end_op: crate::compress::zstd_compress::ZSTD_EndDirective,
) -> usize {
    crate::common::error::ERROR(crate::common::error::ErrorCode::Generic)
}

/// Port of `ZSTDMT_sizeof_CCtx`. Stub returns 0 (no allocation).
#[inline]
pub fn ZSTDMT_sizeof_CCtx(_mtctx: &ZSTDMT_CCtx) -> usize {
    core::mem::size_of::<ZSTDMT_CCtx>()
}

/// Port of `ZSTDMT_toFlushNow`. Stub returns 0 (nothing queued).
#[inline]
pub fn ZSTDMT_toFlushNow(_mtctx: &ZSTDMT_CCtx) -> usize {
    0
}

/// Port of `ZSTDMT_nextInputSizeHint`. Stub returns the block-max
/// suggestion from `ZSTD_CStreamInSize`.
#[inline]
pub fn ZSTDMT_nextInputSizeHint(_mtctx: &ZSTDMT_CCtx) -> usize {
    crate::compress::zstd_compress::ZSTD_CStreamInSize()
}
