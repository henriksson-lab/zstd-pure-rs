//! Pure-Rust port of the Zstandard (zstd) compression library.
//!
//! See `README.md` for status and the overall plan.
//!
//! # Quickstart
//!
//! ```
//! use zstd_pure_rs::prelude::*;
//!
//! let src = b"Pure-Rust zstd can compress this.".to_vec();
//!
//! // Compress at level 3 (upstream default).
//! let mut compressed = vec![0u8; ZSTD_compressBound(src.len())];
//! let cSize = ZSTD_compress(&mut compressed, &src, 3);
//! assert!(!ERR_isError(cSize));
//! compressed.truncate(cSize);
//!
//! // Decompress.
//! let mut decoded = vec![0u8; src.len()];
//! let dSize = ZSTD_decompress(&mut decoded, &compressed);
//! assert_eq!(dSize, src.len());
//! assert_eq!(decoded, src);
//! ```
//!
//! # Skippable frames
//!
//! ```
//! use zstd_pure_rs::prelude::*;
//!
//! // Skippable frames embed arbitrary user data that decoders skip.
//! let payload = b"custom-metadata-goes-here";
//! let mut frame = vec![0u8; payload.len() + 8];
//! let wn = ZSTD_writeSkippableFrame(&mut frame, payload, 7);
//! assert_eq!(wn, payload.len() + 8);
//!
//! // Read back the payload + magic variant.
//! let mut out = vec![0u8; payload.len()];
//! let mut variant = 0u32;
//! let rn = ZSTD_readSkippableFrame(&mut out, Some(&mut variant), &frame);
//! assert_eq!(rn, payload.len());
//! assert_eq!(variant, 7);
//! assert_eq!(&out, payload);
//! ```
//!
//! # Dictionary-based compression
//!
//! ```
//! use zstd_pure_rs::prelude::*;
//!
//! let dict = b"common prefix shared across messages ".to_vec();
//! let src = b"common prefix shared across messages \
//!             plus per-message content".to_vec();
//!
//! // One-shot dict compression via raw content.
//! let mut cctx = ZSTD_createCCtx().expect("alloc");
//! let mut dst = vec![0u8; 512];
//! let n = ZSTD_compress_usingDict(&mut cctx, &mut dst, &src, &dict, 3);
//! assert!(!ERR_isError(n));
//! dst.truncate(n);
//!
//! // Decompress with the same dict.
//! let mut dctx = zstd_pure_rs::decompress::zstd_decompress_block::ZSTD_DCtx::default();
//! let mut out = vec![0u8; src.len() + 64];
//! let d = ZSTD_decompress_usingDict(&mut dctx, &mut out, &dst, &dict);
//! assert_eq!(&out[..d], &src[..]);
//! ```
//!
//! # Streaming compression
//!
//! ```
//! use zstd_pure_rs::prelude::*;
//!
//! let src = b"streaming compression demo ".repeat(30);
//! let mut cctx = ZSTD_createCCtx().expect("alloc");
//! ZSTD_initCStream(&mut cctx, 3);
//!
//! // Stage input, then drive endStream until fully drained.
//! let mut dst = vec![0u8; 2048];
//! let mut dst_pos = 0usize;
//! let mut src_pos = 0usize;
//! ZSTD_compressStream(&mut cctx, &mut dst, &mut dst_pos, &src, &mut src_pos);
//! loop {
//!     let rc = ZSTD_endStream(&mut cctx, &mut dst, &mut dst_pos);
//!     if rc == 0 || ERR_isError(rc) { break; }
//! }
//!
//! // Decompress via the public one-shot entry.
//! let mut out = vec![0u8; src.len() + 64];
//! let d = ZSTD_decompress(&mut out, &dst[..dst_pos]);
//! assert_eq!(&out[..d], &src[..]);
//! ```
//!
//! # Parametric API (advanced)
//!
//! ```
//! use zstd_pure_rs::prelude::*;
//!
//! let src = b"parametric api demo".repeat(20);
//! let mut cctx = ZSTD_createCCtx().expect("alloc");
//! // Opt into XXH64 checksum + exact frame-header FCS.
//! ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_compressionLevel, 5);
//! ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_checksumFlag, 1);
//! ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_contentSizeFlag, 1);
//!
//! let mut dst = vec![0u8; 1024];
//! let n = ZSTD_compress2(&mut cctx, &mut dst, &src);
//! assert!(!ERR_isError(n));
//! dst.truncate(n);
//!
//! let mut out = vec![0u8; src.len() + 64];
//! let d = ZSTD_decompress(&mut out, &dst);
//! assert_eq!(&out[..d], &src[..]);
//! ```
//!
//! See upstream's `zstd.h` for the full API — the same function names
//! are available here (prefixed with `ZSTD_` / `HUF_` / `FSE_` as in
//! upstream) under the `zstd_pure_rs::prelude` re-exports.

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(clippy::missing_safety_doc)]
// Upstream zstd functions often take many arguments; we preserve 1:1
// signatures so code-complexity-comparator stays meaningful.
#![allow(clippy::too_many_arguments)]

// v0.1 is std-only — the port uses `Vec`, `format!`, and `std::thread`
// throughout. A proper `no_std` variant is a future effort.
#[cfg(not(feature = "std"))]
compile_error!(
    "zstd-pure-rs v0.1 requires the `std` feature (enabled by default). \
     no_std support is a future goal; see TODO.md."
);

pub mod common;
pub mod compress;
pub mod decompress;

pub use common::error::{ErrorCode, ZstdError};
pub use common::mem::is_little_endian;
pub use common::zstd_common::{
    ZSTD_isError, ZSTD_versionNumber, ZSTD_versionString, ZSTD_getErrorName,
};

/// Re-exports of the most commonly-used public API, so callers can
/// write `use zstd_pure_rs::prelude::*;` instead of drilling into
/// `compress::zstd_compress` / `decompress::zstd_decompress`.
pub mod prelude {
    // One-shot compression.
    pub use crate::compress::zstd_compress::{
        ZSTD_CCtx, ZSTD_CCtx_loadDictionary, ZSTD_CCtx_loadDictionary_advanced,
        ZSTD_CCtx_loadDictionary_byReference, ZSTD_CCtx_refCDict,
        ZSTD_CCtx_refPrefix, ZSTD_CCtx_refPrefix_advanced, ZSTD_CCtx_refThreadPool,
        ZSTD_CDict, ZSTD_Sequence, ZSTD_compress, ZSTD_compress2, ZSTD_compressBegin,
        ZSTD_compressBegin_usingCDict, ZSTD_compressBegin_usingDict, ZSTD_compressBound,
        ZSTD_compressCCtx, ZSTD_compressSequences, ZSTD_compress_usingCDict,
        ZSTD_compress_usingDict, ZSTD_createCCtx, ZSTD_createCCtx_advanced,
        ZSTD_createCDict, ZSTD_createCDict_advanced, ZSTD_createCDict_byReference,
        ZSTD_customMem, ZSTD_freeCCtx, ZSTD_freeCDict, ZSTD_generateSequences,
        ZSTD_getDictID_fromCDict, ZSTD_getFrameProgression, ZSTD_frameProgression,
        ZSTD_mergeBlockDelimiters, ZSTD_sequenceBound, ZSTD_toFlushNow,
        ZSTD_writeSkippableFrame,
    };
    // Streaming compression.
    pub use crate::compress::zstd_compress::{
        ZSTD_CCtx_setPledgedSrcSize, ZSTD_CStream, ZSTD_CStreamInSize,
        ZSTD_CStreamOutSize, ZSTD_EndDirective, ZSTD_compressStream,
        ZSTD_compressStream2, ZSTD_compressStream2_simpleArgs, ZSTD_createCStream,
        ZSTD_createCStream_advanced, ZSTD_endStream, ZSTD_estimateCCtxSize_usingCCtxParams,
        ZSTD_estimateCStreamSize_usingCCtxParams, ZSTD_estimateCStreamSize_usingCParams,
        ZSTD_estimateCCtxSize_usingCParams, ZSTD_flushStream, ZSTD_freeCStream,
        ZSTD_initCStream, ZSTD_initCStream_srcSize, ZSTD_initCStream_usingDict,
        ZSTD_resetCStream,
    };
    // Parametric + level info.
    pub use crate::compress::zstd_compress::{
        ZSTD_CCtxParams_getParameter, ZSTD_CCtxParams_init, ZSTD_CCtxParams_init_advanced,
        ZSTD_CCtxParams_reset, ZSTD_CCtxParams_setParameter,
        ZSTD_CCtx_getParameter, ZSTD_CCtx_params, ZSTD_CCtx_reset, ZSTD_CCtx_setCParams,
        ZSTD_CCtx_setFParams, ZSTD_CCtx_setParameter, ZSTD_CCtx_setParams,
        ZSTD_CCtx_setParametersUsingCCtxParams,
        ZSTD_FrameParameters, ZSTD_ResetDirective, ZSTD_adjustCParams, ZSTD_bounds,
        ZSTD_cParam_getBounds, ZSTD_cParam_withinBounds, ZSTD_cParameter, ZSTD_checkCParams,
        ZSTD_createCCtxParams, ZSTD_defaultCLevel, ZSTD_freeCCtxParams, ZSTD_getCParams,
        ZSTD_getParams, ZSTD_maxCLevel, ZSTD_minCLevel, ZSTD_parameters,
        ZSTD_CLEVEL_DEFAULT, ZSTD_MAX_CLEVEL, ZSTD_NO_CLEVEL,
    };
    // Memory estimation.
    pub use crate::compress::zstd_compress::{
        ZSTD_estimateCCtxSize, ZSTD_estimateCDictSize, ZSTD_estimateCDictSize_advanced,
        ZSTD_estimateCStreamSize, ZSTD_initStaticCCtx, ZSTD_initStaticCDict,
        ZSTD_initStaticCStream, ZSTD_sizeof_CCtx, ZSTD_sizeof_CDict, ZSTD_sizeof_CStream,
        ZSTD_sizeof_mtctx,
    };
    // Legacy block-level API stubs.
    pub use crate::compress::zstd_compress::{
        ZSTD_compressContinue, ZSTD_compressEnd,
    };
    // One-shot + streaming decompression.
    pub use crate::decompress::zstd_decompress::{
        ZSTD_DCtx_getParameter, ZSTD_DCtx_loadDictionary,
        ZSTD_DCtx_loadDictionary_advanced, ZSTD_DCtx_loadDictionary_byReference,
        ZSTD_DCtx_refDDict, ZSTD_DCtx_refPrefix, ZSTD_DCtx_refPrefix_advanced,
        ZSTD_DCtx_reset, ZSTD_DCtx_setMaxWindowSize, ZSTD_DCtx_setParameter,
        ZSTD_DResetDirective, ZSTD_DStream, ZSTD_DStreamInSize, ZSTD_DStreamOutSize,
        ZSTD_FrameHeader, ZSTD_copyDCtx, ZSTD_createDStream, ZSTD_dParameter,
        ZSTD_dParam_getBounds, ZSTD_decompress, ZSTD_decompressDCtx,
        ZSTD_decompressStream, ZSTD_decompress_usingDDict, ZSTD_decompress_usingDict,
        ZSTD_estimateDCtxSize, ZSTD_estimateDStreamSize, ZSTD_findDecompressedSize,
        ZSTD_findFrameCompressedSize, ZSTD_format_e, ZSTD_frameHeaderSize, ZSTD_freeDStream,
        ZSTD_getDictID_fromFrame, ZSTD_getFrameHeader,
        ZSTD_decodingBufferSize_min, ZSTD_decompressBound, ZSTD_decompressionMargin,
        ZSTD_getDecompressedSize, ZSTD_getFrameContentSize, ZSTD_initDStream,
        ZSTD_initDStream_usingDDict, ZSTD_initDStream_usingDict,
        ZSTD_nextSrcSizeToDecompress,
        ZSTD_isFrame, ZSTD_isSkippableFrame, ZSTD_readSkippableFrame, ZSTD_resetDStream,
        ZSTD_sizeof_DCtx, ZSTD_sizeof_DStream,
        ZSTD_CONTENTSIZE_ERROR, ZSTD_CONTENTSIZE_UNKNOWN, ZSTD_MAGICNUMBER,
        ZSTD_MAGIC_SKIPPABLE_START,
    };
    // Decompress-side legacy block-level + static-init API stubs.
    pub use crate::decompress::zstd_decompress::{
        ZSTD_createDCtx_advanced, ZSTD_createDStream_advanced,
        ZSTD_decompressBegin, ZSTD_decompressBegin_usingDDict,
        ZSTD_decompressBegin_usingDict, ZSTD_decompressContinue,
        ZSTD_initStaticDCtx, ZSTD_initStaticDDict, ZSTD_initStaticDStream,
        ZSTD_nextInputType, ZSTD_nextInputType_e,
    };
    pub use crate::decompress::zstd_ddict::{
        ZSTD_DDict, ZSTD_DDict_dictContent, ZSTD_DDict_dictSize, ZSTD_createDDict,
        ZSTD_createDDict_advanced, ZSTD_createDDict_byReference,
        ZSTD_dictContentType_e, ZSTD_dictLoadMethod_e, ZSTD_freeDDict,
        ZSTD_getDictID_fromDDict, ZSTD_estimateDDictSize, ZSTD_getDictID_fromDict,
        ZSTD_sizeof_DDict,
    };
    // Error handling.
    pub use crate::common::error::{ERR_getErrorName, ERR_isError, ErrorCode, ZstdError};
    // Thread pool API stubs.
    pub use crate::common::pool::{
        POOL_ctx, ZSTD_createThreadPool, ZSTD_freeThreadPool,
    };
    // Version + determinism helpers.
    pub use crate::common::zstd_common::{
        ZSTD_VERSION_MAJOR, ZSTD_VERSION_MINOR, ZSTD_VERSION_NUMBER,
        ZSTD_VERSION_RELEASE, ZSTD_VERSION_STRING, ZSTD_getErrorCode,
        ZSTD_getErrorName, ZSTD_getErrorString, ZSTD_isDeterministicBuild,
        ZSTD_isError, ZSTD_versionNumber, ZSTD_versionString,
    };
}

#[cfg(test)]
mod prelude_tests {
    use crate::prelude::*;

    #[test]
    fn prelude_exposes_common_compress_decompress_flow() {
        let src = b"prelude smoke test. ".repeat(50);
        let mut dst = vec![0u8; 1024];
        let n = ZSTD_compress(&mut dst, &src, 1);
        assert!(!ERR_isError(n));
        dst.truncate(n);
        let mut out = vec![0u8; src.len() + 64];
        let d = ZSTD_decompress(&mut out, &dst);
        assert_eq!(&out[..d], &src[..]);
    }

    #[test]
    fn prelude_exposes_api_types_and_version_helpers() {
        // Version + error helpers reachable through prelude.
        assert_eq!(ZSTD_versionString(), "1.6.0");
        assert_eq!(ZSTD_versionNumber(), ZSTD_VERSION_NUMBER);
        assert_eq!(ZSTD_isDeterministicBuild(), 0);
        assert!(!ZSTD_isError(0));

        // Error-code round-trip.
        use crate::common::error::ERROR;
        let e = ERROR(ErrorCode::DstSizeTooSmall);
        assert_eq!(ZSTD_getErrorCode(e), ErrorCode::DstSizeTooSmall);
        let s1 = ZSTD_getErrorString(ErrorCode::DstSizeTooSmall);
        let s2 = ZSTD_getErrorName(e);
        assert_eq!(s1, s2);

        // Type names reachable.
        let _: Option<ZSTD_nextInputType_e> = None;
        let _: Option<ZSTD_dictContentType_e> = None;
        let _: Option<ZSTD_dictLoadMethod_e> = None;
        let _: Option<ZSTD_format_e> = None;
        let _: Option<ZSTD_EndDirective> = None;
        let _: Option<ZSTD_ResetDirective> = None;
        let _: Option<ZSTD_DResetDirective> = None;
        let _: Option<ZSTD_FrameParameters> = None;
        let _: Option<ZSTD_parameters> = None;
        let _: Option<ZSTD_FrameHeader> = None;
        let _: Option<ZSTD_frameProgression> = None;
        let _: Option<ZSTD_Sequence> = None;
        let _: Option<ZSTD_customMem> = None;
        let _: Option<ErrorCode> = None;
    }

    #[test]
    fn prelude_exposes_new_helpers() {
        // Touch the recently-added entry points to make sure they're
        // reachable from the public prelude.
        let src = b"prelude extended touch".to_vec();
        let mut dst = vec![0u8; 256];
        let mut cctx = ZSTD_createCCtx().unwrap();
        // No level set — ZSTD_compress2 must default to CLEVEL_DEFAULT.
        let n = ZSTD_compress2(&mut cctx, &mut dst, &src);
        assert!(!ERR_isError(n), "compress2 err={n:#x}");
        dst.truncate(n);

        // Frame-level queries
        let b = ZSTD_decompressBound(&dst);
        assert!(b >= src.len() as u64);
        let _ = ZSTD_decompressionMargin(&dst);
        let _ = ZSTD_getDecompressedSize(&dst);
        let _ = ZSTD_sequenceBound(1024);

        // Skippable frames
        let mut skip = vec![0u8; 16];
        ZSTD_writeSkippableFrame(&mut skip, b"meta", 3);
        let mut out = vec![0u8; 16];
        let mut v = 0u32;
        ZSTD_readSkippableFrame(&mut out, Some(&mut v), &skip);
        assert_eq!(v, 3);
    }
}
