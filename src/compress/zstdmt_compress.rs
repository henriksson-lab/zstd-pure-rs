//! Translation of `lib/compress/zstdmt_compress.c`. Multi-threaded
//! compression. Behind the `mt` feature.
//!
//! Current v0.1 provides the API surface as returnable errors /
//! no-ops so the `mt` feature can compile without triggering panics.
//! A real port would use `rayon` for parallel block compression, but
//! the sequential path already handles all correctness requirements —
//! MT is purely a speed optimization.

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zstdmt_api_surface_stubs_behave_safely() {
        // v0.1 has no MT backend — the API surface exists for
        // callers to compile-check against, but every entry must
        // return a safe default (None / 0 / Generic) rather than
        // panicking. Guards against an accidental `todo!()` sneaking
        // in during the future rayon port.
        use crate::common::error::{ERR_getErrorCode, ERR_isError, ErrorCode};
        use crate::compress::zstd_compress::ZSTD_EndDirective;

        // Creator always returns None in v0.1.
        assert!(ZSTDMT_createCCtx(1).is_none());
        assert!(ZSTDMT_createCCtx(0).is_none());

        // Free accepts None without panicking.
        assert_eq!(ZSTDMT_freeCCtx(None), 0);

        // Size / flush-now queries work on a dangling PhantomData
        // stub ctx without allocating.
        let stub = ZSTDMT_CCtx { _priv: core::marker::PhantomData };
        // `ZSTDMT_CCtx` is PhantomData-only → size is 0; just ensure
        // the accessor doesn't panic.
        let _ = ZSTDMT_sizeof_CCtx(&stub);
        assert_eq!(ZSTDMT_toFlushNow(&stub), 0);
        assert!(ZSTDMT_nextInputSizeHint(&stub) > 0);

        // compressStream_generic returns Generic.
        let mut dst = [0u8; 32];
        let mut dp = 0usize;
        let mut sp = 0usize;
        let mut stub = ZSTDMT_CCtx { _priv: core::marker::PhantomData };
        let rc = ZSTDMT_compressStream_generic(
            &mut stub, &mut dst, &mut dp, b"x", &mut sp,
            ZSTD_EndDirective::ZSTD_e_end,
        );
        assert!(ERR_isError(rc));
        assert_eq!(ERR_getErrorCode(rc), ErrorCode::Generic);
    }
}
