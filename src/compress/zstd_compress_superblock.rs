//! Translation of `lib/compress/zstd_compress_superblock.c`.
//!
//! Superblock compression splits a single block into multiple small
//! sub-blocks with independent entropy tables to better adapt to
//! local statistics. Used when `cctx.appliedParams.targetCBlockSize`
//! is set. Not yet ported — current v0.1 always emits single blocks
//! per `ZSTD_BLOCKSIZE_MAX` stride.

#![allow(unused_variables)]

/// Port of `ZSTD_compressSuperBlock`. Skeletal — returns
/// `ErrorCode::Generic` so callers that opt into the superblock path
/// get a proper error rather than a panic.
///
/// Upstream takes `ZSTD_CCtx*`; our port uses the real type so the
/// API signature shape matches.
pub fn ZSTD_compressSuperBlock(
    _zc: &mut crate::compress::zstd_compress::ZSTD_CCtx,
    _dst: &mut [u8],
    _src: &[u8],
    _lastBlock: u32,
) -> usize {
    crate::common::error::ERROR(crate::common::error::ErrorCode::Generic)
}
