//! Translation of `lib/compress/zstd_compress_superblock.c`.
//!
//! Superblock compression splits a single block into multiple small
//! sub-blocks with independent entropy tables to better adapt to
//! local statistics. Used when `cctx.appliedParams.targetCBlockSize`
//! is set. Not yet ported — current v0.1 always emits single blocks
//! per `ZSTD_BLOCKSIZE_MAX` stride.

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::zstd_compress::ZSTD_createCCtx;

    #[test]
    fn compressSuperBlock_stub_returns_Generic_error() {
        // Superblock compression is stubbed in v0.1 pending the
        // sub-block entropy re-emission port. Contract: callers that
        // opt into the superblock path get a proper zstd error code,
        // not a panic.
        use crate::common::error::{ERR_getErrorCode, ERR_isError, ErrorCode};
        let mut cctx = ZSTD_createCCtx().unwrap();
        let mut dst = [0u8; 64];
        let src = b"superblock-test";
        let rc = ZSTD_compressSuperBlock(&mut cctx, &mut dst, src, 1);
        assert!(ERR_isError(rc));
        assert_eq!(ERR_getErrorCode(rc), ErrorCode::Generic);
    }
}
