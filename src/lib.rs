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
//! let mut dst = vec![0u8; ZSTD_compressBound(src.len())];
//! let n = ZSTD_compress_usingDict(&mut cctx, &mut dst, &src, &dict, 3);
//! assert!(!ERR_isError(n));
//! dst.truncate(n);
//!
//! // Decompress with the same dict.
//! let mut dctx = ZSTD_createDCtx();
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
//! let mut dst = vec![0u8; ZSTD_compressBound(src.len())];
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
//! let mut dst = vec![0u8; ZSTD_compressBound(src.len())];
//! let n = ZSTD_compress2(&mut cctx, &mut dst, &src);
//! assert!(!ERR_isError(n));
//! dst.truncate(n);
//!
//! let mut out = vec![0u8; src.len() + 64];
//! let d = ZSTD_decompress(&mut out, &dst);
//! assert_eq!(&out[..d], &src[..]);
//! ```
//!
//! See upstream's `zstd.h` for API semantics. The main public `ZSTD_`
//! entry points are available under the `zstd_pure_rs::prelude`
//! re-exports. Lower-level `HUF_` and `FSE_` helpers remain in their
//! module namespaces unless explicitly re-exported there; unsupported
//! experimental parameters return the corresponding error codes.

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
// Upstream zstd functions often take many arguments; we preserve 1:1
// signatures so code-complexity-comparator stays meaningful.
#![allow(clippy::too_many_arguments)]
// This crate intentionally preserves upstream C control flow and test
// structure closely, so we suppress style-oriented lints that would
// otherwise demand less faithful Rust rewrites.
#![allow(clippy::collapsible_if)]
#![allow(clippy::default_constructed_unit_structs)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::implicit_saturating_sub)]
#![allow(clippy::needless_late_init)]
#![allow(clippy::needless_option_as_deref)]
#![allow(clippy::needless_return)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::type_complexity)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::unnecessary_mut_passed)]

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
    ZSTD_getErrorCode, ZSTD_getErrorName, ZSTD_getErrorString, ZSTD_isDeterministicBuild,
    ZSTD_isError, ZSTD_versionNumber, ZSTD_versionString, ZSTD_VERSION_MAJOR, ZSTD_VERSION_MINOR,
    ZSTD_VERSION_NUMBER, ZSTD_VERSION_RELEASE, ZSTD_VERSION_STRING,
};

/// Upstream-compatible alias for `ZSTD_frameParameters`.
pub type ZSTD_frameParameters = compress::zstd_compress::ZSTD_FrameParameters;
/// Upstream-compatible legacy alias for `ZSTD_FrameHeader`.
pub type ZSTD_frameHeader = decompress::zstd_decompress::ZSTD_FrameHeader;
/// Upstream-compatible legacy alias for `ZSTD_FrameType_e`.
pub type ZSTD_frameType_e = decompress::zstd_decompress::ZSTD_FrameType_e;
/// Upstream-compatible legacy alias for `ZSTD_ParamSwitch_e`.
pub type ZSTD_paramSwitch_e = compress::zstd_ldm::ZSTD_ParamSwitch_e;
/// Upstream-compatible legacy alias for `ZSTD_SequenceFormat_e`.
pub type ZSTD_sequenceFormat_e = compress::zstd_compress::ZSTD_SequenceFormat_e;
/// Upstream-compatible alias for `ZSTD_threadPool`.
#[cfg(feature = "std")]
pub type ZSTD_threadPool = common::pool::POOL_ctx;

/// Upstream `ZSTD_BLOCKSPLITTER_LEVEL_MAX`.
pub const ZSTD_BLOCKSPLITTER_LEVEL_MAX: i32 = 6;
/// Upstream `ZSTD_SEQUENCE_PRODUCER_ERROR`.
pub const ZSTD_SEQUENCE_PRODUCER_ERROR: usize = usize::MAX;

/// Upstream macro aliases for implemented experimental parameter names.
pub const ZSTD_c_format: compress::zstd_compress::ZSTD_cParameter =
    compress::zstd_compress::ZSTD_cParameter::ZSTD_c_format;
pub const ZSTD_c_rsyncable: compress::zstd_compress::ZSTD_cParameter =
    compress::zstd_compress::ZSTD_cParameter::ZSTD_c_rsyncable;
pub const ZSTD_c_forceMaxWindow: compress::zstd_compress::ZSTD_cParameter =
    compress::zstd_compress::ZSTD_cParameter::ZSTD_c_forceMaxWindow;
pub const ZSTD_c_forceAttachDict: compress::zstd_compress::ZSTD_cParameter =
    compress::zstd_compress::ZSTD_cParameter::ZSTD_c_forceAttachDict;
pub const ZSTD_c_literalCompressionMode: compress::zstd_compress::ZSTD_cParameter =
    compress::zstd_compress::ZSTD_cParameter::ZSTD_c_literalCompressionMode;
pub const ZSTD_c_srcSizeHint: compress::zstd_compress::ZSTD_cParameter =
    compress::zstd_compress::ZSTD_cParameter::ZSTD_c_srcSizeHint;
pub const ZSTD_c_enableDedicatedDictSearch: compress::zstd_compress::ZSTD_cParameter =
    compress::zstd_compress::ZSTD_cParameter::ZSTD_c_enableDedicatedDictSearch;
pub const ZSTD_c_stableInBuffer: compress::zstd_compress::ZSTD_cParameter =
    compress::zstd_compress::ZSTD_cParameter::ZSTD_c_stableInBuffer;
pub const ZSTD_c_stableOutBuffer: compress::zstd_compress::ZSTD_cParameter =
    compress::zstd_compress::ZSTD_cParameter::ZSTD_c_stableOutBuffer;
pub const ZSTD_c_blockDelimiters: compress::zstd_compress::ZSTD_cParameter =
    compress::zstd_compress::ZSTD_cParameter::ZSTD_c_blockDelimiters;
pub const ZSTD_c_validateSequences: compress::zstd_compress::ZSTD_cParameter =
    compress::zstd_compress::ZSTD_cParameter::ZSTD_c_validateSequences;
pub const ZSTD_c_splitAfterSequences: compress::zstd_compress::ZSTD_cParameter =
    compress::zstd_compress::ZSTD_cParameter::ZSTD_c_splitAfterSequences;
pub const ZSTD_c_useRowMatchFinder: compress::zstd_compress::ZSTD_cParameter =
    compress::zstd_compress::ZSTD_cParameter::ZSTD_c_useRowMatchFinder;
pub const ZSTD_c_deterministicRefPrefix: compress::zstd_compress::ZSTD_cParameter =
    compress::zstd_compress::ZSTD_cParameter::ZSTD_c_deterministicRefPrefix;
pub const ZSTD_c_prefetchCDictTables: compress::zstd_compress::ZSTD_cParameter =
    compress::zstd_compress::ZSTD_cParameter::ZSTD_c_prefetchCDictTables;
pub const ZSTD_c_blockSplitterLevel: compress::zstd_compress::ZSTD_cParameter =
    compress::zstd_compress::ZSTD_cParameter::ZSTD_c_blockSplitterLevel;
pub const ZSTD_c_enableSeqProducerFallback: compress::zstd_compress::ZSTD_cParameter =
    compress::zstd_compress::ZSTD_cParameter::ZSTD_c_enableSeqProducerFallback;
pub const ZSTD_c_maxBlockSize: compress::zstd_compress::ZSTD_cParameter =
    compress::zstd_compress::ZSTD_cParameter::ZSTD_c_maxBlockSize;
pub const ZSTD_c_repcodeResolution: compress::zstd_compress::ZSTD_cParameter =
    compress::zstd_compress::ZSTD_cParameter::ZSTD_c_repcodeResolution;
pub const ZSTD_c_searchForExternalRepcodes: compress::zstd_compress::ZSTD_cParameter =
    compress::zstd_compress::ZSTD_cParameter::ZSTD_c_repcodeResolution;
pub const ZSTD_d_format: decompress::zstd_decompress::ZSTD_dParameter =
    decompress::zstd_decompress::ZSTD_dParameter::ZSTD_d_format;
pub const ZSTD_d_forceIgnoreChecksum: decompress::zstd_decompress::ZSTD_dParameter =
    decompress::zstd_decompress::ZSTD_dParameter::ZSTD_d_forceIgnoreChecksum;

/// Re-exports of the most commonly-used public API, so callers can
/// write `use zstd_pure_rs::prelude::*;` instead of drilling into
/// `compress::zstd_compress` / `decompress::zstd_decompress`.
pub mod prelude {
    // One-shot compression.
    #[cfg(feature = "mt")]
    pub use crate::compress::zstd_compress::ZSTD_CCtx_refRayonThreadPool;
    #[cfg(feature = "std")]
    pub use crate::compress::zstd_compress::ZSTD_CCtx_refThreadPool;
    pub use crate::compress::zstd_compress::{
        ZSTD_CCtx, ZSTD_CCtx_loadDictionary, ZSTD_CCtx_loadDictionary_advanced,
        ZSTD_CCtx_loadDictionary_byReference, ZSTD_CCtx_refCDict, ZSTD_CCtx_refPrefix,
        ZSTD_CCtx_refPrefix_advanced, ZSTD_CDict, ZSTD_Sequence, ZSTD_SequenceFormat_e,
        ZSTD_SequencePosition, ZSTD_compress, ZSTD_compress2, ZSTD_compressBegin,
        ZSTD_compressBegin_usingCDict, ZSTD_compressBegin_usingDict, ZSTD_compressBound,
        ZSTD_compressCCtx, ZSTD_compressSequences, ZSTD_compressSequencesAndLiterals,
        ZSTD_compress_usingCDict, ZSTD_compress_usingCDict_advanced, ZSTD_compress_usingDict,
        ZSTD_createCCtx, ZSTD_createCCtx_advanced, ZSTD_createCDict, ZSTD_createCDict_advanced,
        ZSTD_createCDict_advanced2, ZSTD_createCDict_byReference, ZSTD_customMem,
        ZSTD_frameProgression, ZSTD_freeCCtx, ZSTD_freeCDict, ZSTD_generateSequences,
        ZSTD_getDictID_fromCDict, ZSTD_getFrameProgression, ZSTD_mergeBlockDelimiters,
        ZSTD_sequenceBound, ZSTD_toFlushNow, ZSTD_writeSkippableFrame,
    };
    // Streaming compression.
    pub use crate::compress::zstd_compress::{
        ZSTD_CCtx_setPledgedSrcSize, ZSTD_CStream, ZSTD_CStreamInSize, ZSTD_CStreamOutSize,
        ZSTD_EndDirective, ZSTD_compressBegin_advanced, ZSTD_compressBegin_usingCDict_advanced,
        ZSTD_compressStream, ZSTD_compressStream2, ZSTD_compressStream2_simpleArgs,
        ZSTD_createCStream, ZSTD_createCStream_advanced, ZSTD_endStream,
        ZSTD_estimateCCtxSize_usingCCtxParams, ZSTD_estimateCCtxSize_usingCParams,
        ZSTD_estimateCStreamSize_usingCCtxParams, ZSTD_estimateCStreamSize_usingCParams,
        ZSTD_flushStream, ZSTD_freeCStream, ZSTD_inBuffer, ZSTD_initCStream,
        ZSTD_initCStream_advanced, ZSTD_initCStream_srcSize, ZSTD_initCStream_usingCDict,
        ZSTD_initCStream_usingCDict_advanced, ZSTD_initCStream_usingDict, ZSTD_outBuffer,
        ZSTD_resetCStream,
    };
    // Parametric + level info.
    pub use crate::compress::match_state::ZSTD_compressionParameters;
    pub use crate::compress::zstd_compress::ZSTD_dictAttachPref_e;
    pub use crate::compress::zstd_compress::{
        ZSTD_CCtxParams_getParameter, ZSTD_CCtxParams_init, ZSTD_CCtxParams_init_advanced,
        ZSTD_CCtxParams_registerSequenceProducer, ZSTD_CCtxParams_reset,
        ZSTD_CCtxParams_setParameter, ZSTD_CCtxParams_setZstdParams, ZSTD_CCtx_getParameter,
        ZSTD_CCtx_params, ZSTD_CCtx_reset, ZSTD_CCtx_setCParams, ZSTD_CCtx_setFParams,
        ZSTD_CCtx_setFormat, ZSTD_CCtx_setParameter, ZSTD_CCtx_setParametersUsingCCtxParams,
        ZSTD_CCtx_setParams, ZSTD_FrameParameters, ZSTD_ResetDirective, ZSTD_adjustCParams,
        ZSTD_allocFunction, ZSTD_bounds, ZSTD_cParam_clampBounds, ZSTD_cParam_getBounds,
        ZSTD_cParam_withinBounds, ZSTD_cParameter, ZSTD_checkCParams,
        ZSTD_compressFrame_fast_advanced, ZSTD_compressFrame_fast_with_prefix_advanced,
        ZSTD_createCCtxParams, ZSTD_defaultCLevel, ZSTD_forceIgnoreChecksum_e, ZSTD_freeCCtxParams,
        ZSTD_freeFunction, ZSTD_getCParams, ZSTD_getParams, ZSTD_literalCompressionMode_e,
        ZSTD_maxCLevel, ZSTD_minCLevel, ZSTD_parameters, ZSTD_refMultipleDDicts_e,
        ZSTD_registerSequenceProducer, ZSTD_sequenceProducer_F, ZSTD_writeFrameHeader_advanced,
        ZSTD_BLOCKSIZE_MAX_MIN, ZSTD_CHAINLOG_MAX, ZSTD_CHAINLOG_MAX_32, ZSTD_CHAINLOG_MAX_64,
        ZSTD_CHAINLOG_MIN, ZSTD_CLEVEL_DEFAULT, ZSTD_COMPRESSBOUND, ZSTD_FRAMEHEADERSIZE_MAX,
        ZSTD_MAX_CLEVEL, ZSTD_MAX_INPUT_SIZE, ZSTD_MINMATCH_MAX, ZSTD_MINMATCH_MIN, ZSTD_NO_CLEVEL,
        ZSTD_OVERLAPLOG_MAX, ZSTD_OVERLAPLOG_MIN, ZSTD_SEARCHLOG_MAX, ZSTD_SEARCHLOG_MIN,
        ZSTD_SRCSIZEHINT_MAX, ZSTD_SRCSIZEHINT_MIN, ZSTD_STRATEGY_MAX, ZSTD_STRATEGY_MIN,
        ZSTD_TARGETCBLOCKSIZE_MAX, ZSTD_TARGETCBLOCKSIZE_MIN, ZSTD_TARGETLENGTH_MAX,
        ZSTD_TARGETLENGTH_MIN, ZSTD_WINDOWLOG_MAX, ZSTD_WINDOWLOG_MIN,
    };
    pub use crate::compress::zstd_compress::{
        ZSTD_LDM_BUCKETSIZELOG_MIN, ZSTD_LDM_HASHLOG_MAX, ZSTD_LDM_HASHLOG_MIN,
        ZSTD_LDM_HASHRATELOG_MAX, ZSTD_LDM_HASHRATELOG_MIN, ZSTD_LDM_MINMATCH_MAX,
        ZSTD_LDM_MINMATCH_MIN,
    };
    pub use crate::compress::zstd_compress_literals::ZSTD_strategy;
    pub use crate::compress::zstd_compress_sequences::{
        ZSTD_btlazy2, ZSTD_btopt, ZSTD_btultra, ZSTD_btultra2, ZSTD_dfast, ZSTD_fast, ZSTD_greedy,
        ZSTD_lazy, ZSTD_lazy2,
    };
    pub use crate::compress::zstd_ldm::{
        ZSTD_ParamSwitch_e, ZSTD_HASHLOG_MAX, ZSTD_HASHLOG_MIN, ZSTD_LDM_BUCKETSIZELOG_MAX,
        ZSTD_LDM_DEFAULT_WINDOW_LOG,
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
        ZSTD_compressBegin_usingCDict_deprecated, ZSTD_compressBegin_usingDict_deprecated,
        ZSTD_compressBlock, ZSTD_compressBlock_deprecated, ZSTD_compressContinue,
        ZSTD_compressContinue_public, ZSTD_compressEnd, ZSTD_compressEnd_public, ZSTD_getBlockSize,
    };
    // Advanced + ctx-copy helpers.
    pub use crate::compress::zstd_compress::{
        ZSTD_compress_advanced, ZSTD_copyCCtx, ZSTD_cycleLog, ZSTD_dedicatedDictSearch_getCParams,
        ZSTD_dedicatedDictSearch_isSupported, ZSTD_dedicatedDictSearch_revertCParams,
        ZSTD_dictAndWindowLog,
    };
    // One-shot + streaming decompression.
    pub use crate::decompress::zstd_decompress::{
        ZSTD_DCtx, ZSTD_DCtx_getParameter, ZSTD_DCtx_loadDictionary,
        ZSTD_DCtx_loadDictionary_advanced, ZSTD_DCtx_loadDictionary_byReference,
        ZSTD_DCtx_refDDict, ZSTD_DCtx_refPrefix, ZSTD_DCtx_refPrefix_advanced, ZSTD_DCtx_reset,
        ZSTD_DCtx_setFormat, ZSTD_DCtx_setMaxWindowSize, ZSTD_DCtx_setParameter,
        ZSTD_DResetDirective, ZSTD_DStream, ZSTD_DStreamInSize, ZSTD_DStreamOutSize,
        ZSTD_FrameHeader, ZSTD_FrameType_e, ZSTD_copyDCtx, ZSTD_createDCtx, ZSTD_createDStream,
        ZSTD_dParam_getBounds, ZSTD_dParam_withinBounds, ZSTD_dParameter,
        ZSTD_decodingBufferSize_min, ZSTD_decompress, ZSTD_decompressBound, ZSTD_decompressDCtx,
        ZSTD_decompressStream, ZSTD_decompressStream_simpleArgs, ZSTD_decompress_usingDDict,
        ZSTD_decompress_usingDict, ZSTD_decompressionMargin, ZSTD_dictUses_e,
        ZSTD_estimateDCtxSize, ZSTD_estimateDStreamSize, ZSTD_estimateDStreamSize_fromFrame,
        ZSTD_findDecompressedSize, ZSTD_findFrameCompressedSize,
        ZSTD_findFrameCompressedSize_advanced, ZSTD_findFrameSizeInfo, ZSTD_format_e,
        ZSTD_frameHeaderSize, ZSTD_frameSizeInfo, ZSTD_freeDCtx, ZSTD_freeDStream,
        ZSTD_getDecompressedSize, ZSTD_getDictID_fromFrame, ZSTD_getFrameContentSize,
        ZSTD_getFrameHeader, ZSTD_getFrameHeader_advanced, ZSTD_initDStream,
        ZSTD_initDStream_usingDDict, ZSTD_initDStream_usingDict, ZSTD_isFrame,
        ZSTD_isSkippableFrame, ZSTD_nextSrcSizeToDecompress, ZSTD_readSkippableFrame,
        ZSTD_resetDStream, ZSTD_sizeof_DCtx, ZSTD_sizeof_DStream, ZSTD_CONTENTSIZE_ERROR,
        ZSTD_CONTENTSIZE_UNKNOWN, ZSTD_DECOMPRESSION_MARGIN, ZSTD_FRAMEHEADERSIZE_MIN,
        ZSTD_FRAMEHEADERSIZE_PREFIX, ZSTD_FRAMEIDSIZE, ZSTD_MAGICNUMBER, ZSTD_MAGIC_DICTIONARY,
        ZSTD_MAGIC_SKIPPABLE_MASK, ZSTD_MAGIC_SKIPPABLE_START, ZSTD_SKIPPABLEHEADERSIZE,
        ZSTD_WINDOWLOG_ABSOLUTEMIN, ZSTD_WINDOWLOG_LIMIT_DEFAULT, ZSTD_WINDOWLOG_MAX_32,
        ZSTD_WINDOWLOG_MAX_64,
    };
    pub use crate::decompress::zstd_decompress_block::{
        ZSTD_BLOCKHEADERSIZE, ZSTD_BLOCKSIZELOG_MAX, ZSTD_BLOCKSIZE_MAX,
    };
    // Decompress-side legacy block-level + static-init API stubs.
    pub use crate::decompress::zstd_ddict::{
        ZSTD_DDict, ZSTD_DDict_dictContent, ZSTD_DDict_dictSize, ZSTD_copyDDictParameters,
        ZSTD_createDDict, ZSTD_createDDict_advanced, ZSTD_createDDict_byReference,
        ZSTD_dictContentType_e, ZSTD_dictLoadMethod_e, ZSTD_estimateDDictSize, ZSTD_freeDDict,
        ZSTD_getDictID_fromDDict, ZSTD_getDictID_fromDict, ZSTD_sizeof_DDict,
    };
    pub use crate::decompress::zstd_decompress::{
        ZSTD_createDCtx_advanced, ZSTD_createDStream_advanced, ZSTD_decompressBegin,
        ZSTD_decompressBegin_usingDDict, ZSTD_decompressBegin_usingDict, ZSTD_decompressBlock,
        ZSTD_decompressBlock_deprecated, ZSTD_decompressContinue, ZSTD_initStaticDCtx,
        ZSTD_initStaticDDict, ZSTD_initStaticDStream, ZSTD_insertBlock, ZSTD_nextInputType,
        ZSTD_nextInputType_e,
    };
    // Error handling.
    pub use crate::common::error::{ERR_getErrorName, ERR_isError, ErrorCode, ZstdError};
    // Thread pool API.
    #[cfg(feature = "std")]
    pub use crate::common::pool::{POOL_ctx, ZSTD_createThreadPool, ZSTD_freeThreadPool};
    #[cfg(feature = "std")]
    pub use crate::ZSTD_threadPool;
    pub use crate::{
        ZSTD_c_blockDelimiters, ZSTD_c_blockSplitterLevel, ZSTD_c_deterministicRefPrefix,
        ZSTD_c_enableDedicatedDictSearch, ZSTD_c_enableSeqProducerFallback, ZSTD_c_forceAttachDict,
        ZSTD_c_forceMaxWindow, ZSTD_c_format, ZSTD_c_literalCompressionMode, ZSTD_c_maxBlockSize,
        ZSTD_c_prefetchCDictTables, ZSTD_c_repcodeResolution, ZSTD_c_rsyncable,
        ZSTD_c_searchForExternalRepcodes, ZSTD_c_splitAfterSequences, ZSTD_c_srcSizeHint,
        ZSTD_c_stableInBuffer, ZSTD_c_stableOutBuffer, ZSTD_c_useRowMatchFinder,
        ZSTD_c_validateSequences, ZSTD_d_forceIgnoreChecksum, ZSTD_d_format, ZSTD_frameHeader,
        ZSTD_frameParameters, ZSTD_frameType_e, ZSTD_paramSwitch_e, ZSTD_sequenceFormat_e,
        ZSTD_BLOCKSPLITTER_LEVEL_MAX, ZSTD_SEQUENCE_PRODUCER_ERROR,
    };
    // Version + determinism helpers.
    pub use crate::common::zstd_common::{
        ZSTD_getErrorCode, ZSTD_getErrorName, ZSTD_getErrorString, ZSTD_isDeterministicBuild,
        ZSTD_isError, ZSTD_versionNumber, ZSTD_versionString, ZSTD_VERSION_MAJOR,
        ZSTD_VERSION_MINOR, ZSTD_VERSION_NUMBER, ZSTD_VERSION_RELEASE, ZSTD_VERSION_STRING,
    };
}

#[cfg(test)]
mod prelude_tests {
    use crate::prelude::*;

    #[test]
    fn prelude_exposes_common_compress_decompress_flow() {
        let src = b"prelude smoke test. ".repeat(50);
        let mut dst = vec![0u8; ZSTD_compressBound(src.len())];
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
        // The concrete string literal is pinned in `ZSTD_VERSION_STRING`;
        // see the unit test in `zstd_common` for that gate.
        assert_eq!(ZSTD_versionString(), ZSTD_VERSION_STRING);
        assert_eq!(ZSTD_versionNumber(), ZSTD_VERSION_NUMBER);
        assert_eq!(ZSTD_isDeterministicBuild(), 1);
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
        let _: Option<ZSTD_dictUses_e> = None;
        let _: Option<ZSTD_dictContentType_e> = None;
        let _: Option<ZSTD_dictLoadMethod_e> = None;
        let _: Option<ZSTD_format_e> = None;
        let _: Option<ZSTD_EndDirective> = None;
        let _: Option<ZSTD_ResetDirective> = None;
        let _: Option<ZSTD_DResetDirective> = None;
        let _: Option<ZSTD_FrameParameters> = None;
        let _: Option<ZSTD_parameters> = None;
        let _: Option<ZSTD_FrameHeader> = None;
        let _: Option<ZSTD_frameHeader> = None;
        let _: Option<ZSTD_frameProgression> = None;
        let _: Option<ZSTD_Sequence> = None;
        let _: Option<ZSTD_sequenceFormat_e> = None;
        let _: Option<ZSTD_DCtx> = None;
        let _: Option<ZSTD_inBuffer<'static>> = None;
        let _: Option<ZSTD_outBuffer<'static>> = None;
        let _: Option<ZSTD_customMem> = None;
        let _: Option<ZSTD_allocFunction> = None;
        let _: Option<ZSTD_freeFunction> = None;
        let _: Option<ZSTD_sequenceProducer_F> = None;
        let _: Option<ZSTD_forceIgnoreChecksum_e> = None;
        let _: Option<ZSTD_refMultipleDDicts_e> = None;
        let _: Option<ZSTD_dictAttachPref_e> = None;
        let _: Option<ZSTD_literalCompressionMode_e> = None;
        let _: Option<ZSTD_paramSwitch_e> = None;
        let _: Option<ZSTD_strategy> = None;
        let _: Option<ErrorCode> = None;

        // Upstream's public block-level helper takes a compression context.
        let _: fn(&ZSTD_CCtx) -> usize = ZSTD_getBlockSize;
    }

    #[test]
    fn crate_root_exposes_common_public_helpers() {
        use crate::common::error::ERROR;

        assert_eq!(crate::ZSTD_versionString(), crate::ZSTD_VERSION_STRING);
        assert_eq!(crate::ZSTD_versionNumber(), crate::ZSTD_VERSION_NUMBER);
        assert_eq!(crate::ZSTD_VERSION_MAJOR, 1);
        assert_eq!(crate::ZSTD_VERSION_MINOR, 6);
        assert_eq!(crate::ZSTD_VERSION_RELEASE, 0);
        assert_eq!(crate::ZSTD_isDeterministicBuild(), 1);
        assert!(!crate::ZSTD_isError(0));

        let e = ERROR(crate::ErrorCode::DstSizeTooSmall);
        assert_eq!(
            crate::ZSTD_getErrorCode(e),
            crate::ErrorCode::DstSizeTooSmall
        );
        assert_eq!(
            crate::ZSTD_getErrorName(e),
            crate::ZSTD_getErrorString(crate::ErrorCode::DstSizeTooSmall)
        );
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
        let _compress_sequences_and_literals: fn(
            &mut ZSTD_CCtx,
            &mut [u8],
            &[ZSTD_Sequence],
            &[u8],
            usize,
            usize,
        ) -> usize = ZSTD_compressSequencesAndLiterals;

        // Skippable frames
        let mut skip = vec![0u8; 16];
        ZSTD_writeSkippableFrame(&mut skip, b"meta", 3);
        let mut out = vec![0u8; 16];
        let mut v = 0u32;
        ZSTD_readSkippableFrame(&mut out, Some(&mut v), &skip);
        assert_eq!(v, 3);
    }

    #[test]
    fn prelude_exposes_upstream_static_aliases_and_constants() {
        assert_eq!(ZSTD_frameType_e::ZSTD_frame as u32, 0);
        assert_eq!(ZSTD_FrameType_e::ZSTD_skippableFrame as u32, 1);
        assert_eq!(ZSTD_fast, 1);
        assert_eq!(ZSTD_dfast, 2);
        assert_eq!(ZSTD_greedy, 3);
        assert_eq!(ZSTD_lazy, 4);
        assert_eq!(ZSTD_lazy2, 5);
        assert_eq!(ZSTD_btlazy2, 6);
        assert_eq!(ZSTD_btopt, 7);
        assert_eq!(ZSTD_btultra, 8);
        assert_eq!(ZSTD_btultra2, 9);
        assert_eq!(ZSTD_STRATEGY_MIN, ZSTD_fast);
        assert_eq!(ZSTD_STRATEGY_MAX, ZSTD_btultra2);
        assert_eq!(ZSTD_BLOCKSPLITTER_LEVEL_MAX, 6);
        assert_eq!(ZSTD_SEQUENCE_PRODUCER_ERROR, usize::MAX);
        assert_eq!(ZSTD_MAGIC_DICTIONARY, 0xEC30A437);
        assert_eq!(ZSTD_MAGIC_SKIPPABLE_MASK, 0xFFFFFFF0);
        assert_eq!(ZSTD_COMPRESSBOUND(1024), ZSTD_compressBound(1024));
        assert_eq!(ZSTD_BLOCKSIZE_MAX_MIN, 1 << 10);
        assert_eq!(ZSTD_DECOMPRESSION_MARGIN(0, 1024), 1046);
        assert_eq!(ZSTD_FRAMEHEADERSIZE_PREFIX(ZSTD_format_e::ZSTD_f_zstd1), 5);
        assert_eq!(ZSTD_FRAMEHEADERSIZE_MIN(ZSTD_format_e::ZSTD_f_zstd1), 6);
        assert_eq!(ZSTD_c_format, ZSTD_cParameter::ZSTD_c_format);
        assert_eq!(ZSTD_c_rsyncable, ZSTD_cParameter::ZSTD_c_rsyncable);
        assert_eq!(
            ZSTD_c_forceMaxWindow,
            ZSTD_cParameter::ZSTD_c_forceMaxWindow
        );
        assert_eq!(
            ZSTD_c_forceAttachDict,
            ZSTD_cParameter::ZSTD_c_forceAttachDict
        );
        assert_eq!(
            ZSTD_c_literalCompressionMode,
            ZSTD_cParameter::ZSTD_c_literalCompressionMode
        );
        assert_eq!(ZSTD_c_srcSizeHint, ZSTD_cParameter::ZSTD_c_srcSizeHint);
        assert_eq!(
            ZSTD_c_enableDedicatedDictSearch,
            ZSTD_cParameter::ZSTD_c_enableDedicatedDictSearch
        );
        assert_eq!(
            ZSTD_c_stableInBuffer,
            ZSTD_cParameter::ZSTD_c_stableInBuffer
        );
        assert_eq!(
            ZSTD_c_stableOutBuffer,
            ZSTD_cParameter::ZSTD_c_stableOutBuffer
        );
        assert_eq!(
            ZSTD_c_blockDelimiters,
            ZSTD_cParameter::ZSTD_c_blockDelimiters
        );
        assert_eq!(
            ZSTD_c_validateSequences,
            ZSTD_cParameter::ZSTD_c_validateSequences
        );
        assert_eq!(
            ZSTD_c_splitAfterSequences,
            ZSTD_cParameter::ZSTD_c_splitAfterSequences
        );
        assert_eq!(
            ZSTD_c_useRowMatchFinder,
            ZSTD_cParameter::ZSTD_c_useRowMatchFinder
        );
        assert_eq!(
            ZSTD_c_deterministicRefPrefix,
            ZSTD_cParameter::ZSTD_c_deterministicRefPrefix
        );
        assert_eq!(
            ZSTD_c_prefetchCDictTables,
            ZSTD_cParameter::ZSTD_c_prefetchCDictTables
        );
        assert_eq!(
            ZSTD_c_blockSplitterLevel,
            ZSTD_cParameter::ZSTD_c_blockSplitterLevel
        );
        assert_eq!(
            ZSTD_c_enableSeqProducerFallback,
            ZSTD_cParameter::ZSTD_c_enableSeqProducerFallback
        );
        assert_eq!(ZSTD_c_maxBlockSize, ZSTD_cParameter::ZSTD_c_maxBlockSize);
        assert_eq!(
            ZSTD_c_repcodeResolution,
            ZSTD_cParameter::ZSTD_c_repcodeResolution
        );
        assert_eq!(
            ZSTD_c_searchForExternalRepcodes,
            ZSTD_cParameter::ZSTD_c_repcodeResolution
        );
        assert_eq!(ZSTD_d_format, ZSTD_dParameter::ZSTD_d_format);
        assert_eq!(
            ZSTD_d_forceIgnoreChecksum,
            ZSTD_dParameter::ZSTD_d_forceIgnoreChecksum
        );
    }

    #[test]
    fn prelude_exposes_format_helpers() {
        // The format-aware surface lives behind `*_advanced` + the new
        // `ZSTD_{C,D}Ctx_setFormat` helpers; verify the prelude
        // re-exports them so downstream crates don't have to reach
        // into module paths.
        let mut cctx = ZSTD_createCCtx().unwrap();
        let mut dctx = ZSTD_createDCtx();
        assert_eq!(
            ZSTD_CCtx_setFormat(&mut cctx, ZSTD_format_e::ZSTD_f_zstd1_magicless),
            ZSTD_format_e::ZSTD_f_zstd1_magicless as usize,
        );
        assert_eq!(
            ZSTD_DCtx_setFormat(&mut dctx, ZSTD_format_e::ZSTD_f_zstd1_magicless),
            0,
        );
        // Parametric bounds getter reachable.
        let cb = ZSTD_cParam_getBounds(ZSTD_cParameter::ZSTD_c_format);
        assert_eq!(cb.upperBound, ZSTD_format_e::ZSTD_f_zstd1_magicless as i32);
        let db = ZSTD_dParam_getBounds(ZSTD_dParameter::ZSTD_d_format);
        assert_eq!(db.upperBound, ZSTD_format_e::ZSTD_f_zstd1_magicless as i32);
        // `_advanced` variants for measurement + header parsing.
        let mut frame = vec![0u8; 64];
        let hdrSize = ZSTD_writeFrameHeader_advanced(
            &mut frame,
            &ZSTD_FrameParameters {
                contentSizeFlag: 0,
                checksumFlag: 0,
                noDictIDFlag: 1,
            },
            15,
            0,
            0,
            ZSTD_format_e::ZSTD_f_zstd1_magicless,
        );
        assert!(!ERR_isError(hdrSize));
    }
}
