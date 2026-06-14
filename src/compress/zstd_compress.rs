//! Translation of `lib/compress/zstd_compress.c`. Top-level compression
//! API and orchestration.

#![allow(non_snake_case)]
#![allow(clippy::default_constructed_unit_structs)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::implicit_saturating_sub)]
#![allow(clippy::needless_late_init)]
#![allow(clippy::needless_return)]
#![allow(clippy::needless_option_as_deref)]
#![allow(clippy::type_complexity)]

use crate::common::bits::ZSTD_highbit32;
use crate::common::error::{ERR_isError, ErrorCode, ERROR};
use crate::common::mem::{
    MEM_32bits, MEM_64bits, MEM_read64, MEM_write64, MEM_writeLE16, MEM_writeLE24, MEM_writeLE32,
    MEM_writeLE64,
};
use crate::compress::fse_compress::{FSE_CTable, FSE_CTABLE_SIZE_U32};
use crate::compress::hist::HIST_countFast_wksp;
use crate::compress::huf_compress::HUF_CElt;
use crate::compress::seq_store::{
    Repcodes_t, SeqStore_t, ZSTD_countSeqStoreLiteralsBytes, ZSTD_countSeqStoreMatchBytes,
    ZSTD_deriveSeqStoreChunkInto, ZSTD_longLengthType_e, ZSTD_seqStore_resolveOffCodes,
};
use crate::compress::zstd_compress_literals::{
    HUF_repeat, ZSTD_compressLiterals, ZSTD_literalsCompressionIsDisabled, ZSTD_minGain,
};
use crate::compress::zstd_compress_sequences::{
    FSE_repeat, ZSTD_DefaultPolicy_e, ZSTD_buildCTable, ZSTD_encodeSequences,
    ZSTD_selectEncodingType,
};
use crate::decompress::zstd_decompress_block::{
    DefaultMaxOff, LLFSELog, LL_defaultNorm, LL_defaultNormLog, MLFSELog, ML_defaultNorm,
    ML_defaultNormLog, MaxLL, MaxML, MaxOff, MaxSeq, OF_defaultNorm, OF_defaultNormLog, OffFSELog,
    SymbolEncodingType_e, ZSTD_blockHeaderSize, LONGNBSEQ, MIN_CBLOCK_SIZE, ZSTD_BLOCKSIZE_MAX,
};

/// Port of `ZSTD_CCtx`. Holds a reusable match state, seqStore, and
/// entropy tables so successive `ZSTD_compressCCtx` calls amortize
/// allocation. Also backs the simple streaming API — `initCStream`
/// captures the level, `compressStream` buffers input, `endStream`
/// compresses the buffered input and drains the result.
#[derive(Debug, Clone)]
pub struct ZSTD_CCtx {
    /// Match state carried across blocks inside a single frame. Lazy
    /// initialized on first `ZSTD_compressCCtx` call.
    pub ms: Option<crate::compress::match_state::ZSTD_MatchState_t>,
    /// Seq store reused across frames.
    pub seqStore: Option<SeqStore_t>,
    /// Per-frame entropy tables (literals huf + sequence fse). `prev`
    /// and `next` slots rotate each block.
    pub prevEntropy: ZSTD_entropyCTables_t,
    pub nextEntropy: ZSTD_entropyCTables_t,
    /// Upstream `blockState.prevCBlock.rep` / `nextCBlock.rep`. The
    /// three-slot repcode history that tracks which offsets the
    /// compressor last emitted — rotated each block alongside
    /// `prev/nextEntropy`.
    pub prev_rep: [u32; 3],
    pub next_rep: [u32; 3],
    /// Scratch context for the post-block splitter.
    pub blockSplitCtx: Option<ZSTD_blockSplitCtx>,
    /// Reusable entropy scratch matching upstream's CCtx workspace
    /// carving for per-block HUF/FSE statistics.
    pub entropyScratch: ZSTD_entropyScratch,
    /// Upstream `externSeqStore`. Internal raw-sequence stream used by
    /// MT/LDM orchestration paths to feed pre-generated matches into
    /// the regular block compressor.
    pub externalMatchStore: Option<crate::compress::zstd_ldm::RawSeqStore_t>,
    /// Upstream `ldmState`. Persistent long-distance matcher state
    /// carried across blocks within a frame.
    pub ldmState: Option<crate::compress::zstd_ldm::ldmState_t>,
    /// Upstream `ldmSequences` / `maxNbLdmSequences`. Scratch raw
    /// sequence buffer reused by LDM generation.
    pub ldmSequences: crate::compress::zstd_ldm::RawSeqStore_t,
    /// Attached external thread pool reference for MT-capable builds.
    /// Stored as a raw pointer-sized token because the public API
    /// borrows the pool rather than transferring ownership.
    pub threadPoolRef: usize,
    /// Attached external rayon thread pool reference for MT-capable builds.
    /// Stored separately from `threadPoolRef` because the existing pool API
    /// uses zstd's `POOL_ctx` type.
    pub rayonThreadPoolRef: usize,
    /// Cached size hint for the attached thread pool / MT surface.
    pub mtctxSizeHint: usize,

    // ---- Streaming-mode state (initCStream / compressStream / endStream) ----
    /// Compression level set by `ZSTD_initCStream`. `None` until init.
    pub stream_level: Option<i32>,
    /// Optional pledged source size (via `ZSTD_CCtx_setPledgedSrcSize`).
    /// When set, `endStream` emits a frame header with an exact
    /// content-size field and can pick tighter cParams.
    pub pledged_src_size: Option<u64>,
    /// Optional dictionary bytes set by `ZSTD_initCStream_usingDict` /
    /// `loadDictionary`. `endStream` uses them as history when
    /// non-empty; magic-prefixed zstd-format dictionaries also retain
    /// their parsed `dictID`.
    pub stream_dict: Vec<u8>,
    /// Original dictionary bytes provided by the caller. For full
    /// dictionaries, `stream_dict` stores only parsed content while
    /// this keeps the magic-prefixed bytes for later session init.
    pub stream_dict_original: Vec<u8>,
    /// Caller-selected interpretation for `stream_dict_original`.
    pub stream_dict_content_type: crate::decompress::zstd_ddict::ZSTD_dictContentType_e,
    /// Referenced CDict installed via `ZSTD_CCtx_refCDict`. When set,
    /// stream/session init routes through the CDict-backed begin path
    /// instead of degrading to a raw dictionary.
    pub stream_cdict: Option<ZSTD_CDict>,
    /// `ZSTD_c_checksumFlag`: emit an XXH64 trailer when true.
    pub param_checksum: bool,
    /// `ZSTD_c_contentSizeFlag`: emit the frame content-size field
    /// when true (default true, matching upstream).
    pub param_contentSize: bool,
    /// `ZSTD_c_dictIDFlag`: emit dictID in header when a dict is in
    /// use (default true, matching upstream).
    pub param_dictID: bool,
    /// Pending input bytes staged by `ZSTD_compressStream`, awaiting
    /// `ZSTD_endStream` to be compressed in a single frame.
    pub stream_in_buffer: Vec<u8>,
    /// Effective parameter snapshot captured when this buffered
    /// stream first accepts input. Since this port doesn't yet emit
    /// blocks before `endStream`, the final one-shot compression must
    /// not observe later CCtx parameter edits for already accepted
    /// bytes.
    pub stream_params_snapshot: Option<ZSTD_streamParamsSnapshot>,
    /// Compressed payload produced by `ZSTD_endStream`, awaiting drain
    /// through successive `endStream` calls.
    pub stream_out_buffer: Vec<u8>,
    /// Bytes already copied out of `stream_out_buffer` into caller's
    /// output buffer.
    pub stream_out_drained: usize,
    /// Upstream `streamStage`: load caller input into the bounded
    /// input window, then flush any pending compressed output.
    pub stream_stage: ZSTD_cStreamStage,
    /// Upstream `inToCompress`: start offset of the next block inside
    /// `stream_in_buffer`.
    pub stream_in_to_compress: usize,
    /// Upstream `inBuffTarget`: target end offset before emitting the
    /// next non-final block.
    pub stream_in_target: usize,
    /// Upstream `frameEnded`: set after the final block/epilogue has
    /// been produced and only pending output remains to drain.
    pub stream_frame_ended: bool,
    /// Absolute match-state index represented by `stream_in_buffer[0]`.
    pub stream_window_base: u32,
    /// Internal escape hatch for CLI routes that must preserve the
    /// buffered output shape while still using the streaming API.
    pub stream_disable_windowed: bool,
    /// Stable-buffer expectation state for the streaming API.
    pub expected_in_src: usize,
    pub expected_in_size: usize,
    pub expected_in_pos: usize,
    pub expected_out_buffer_size: usize,
    pub buffer_expectations_set: bool,
    /// Flag: `endStream` has produced output (frame is final).
    pub stream_closed: bool,
    /// The previous streaming frame fully drained and no new input has
    /// started yet. Used to make repeated `endStream()` calls no-op while
    /// still allowing init-stage setters.
    pub stream_frame_completed: bool,
    /// cParams requested via `ZSTD_CCtx_setCParams`. Stored so API
    /// getters can round-trip and so `ZSTD_getBlockSize` can fall
    /// back to the requested `windowLog` before the CCtx has been
    /// initialized. The compressor path resolves the final cParams
    /// via `resolveCParams` → `ZSTD_overrideCParams`, which layers
    /// `requestedParams.cParams` on top of level-derived defaults.
    pub requested_cParams: Option<crate::compress::match_state::ZSTD_compressionParameters>,
    /// Upstream `cctx.requestedParams`. Full CCtx_params struct the
    /// user has configured; `appliedParams` is derived from this at
    /// `ZSTD_compressBegin_internal` time.
    pub requestedParams: ZSTD_CCtx_params,
    /// Upstream `cctx.appliedParams`. The active parameter set for
    /// the current compression session — set by `ZSTD_resetCCtx_internal`
    /// after resolving knobs against the concrete cParams.
    pub appliedParams: ZSTD_CCtx_params,
    /// Upstream `cctx.stage`. Lifecycle marker for the block-level
    /// compression API.
    pub stage: ZSTD_compressionStage_e,
    /// Upstream `cctx.dictID`. The dictID of the active dictionary;
    /// zero when no dict is in use.
    pub dictID: u32,
    /// Upstream `cctx.dictContentSize`. Size in bytes of the active
    /// dict's content.
    pub dictContentSize: usize,
    /// Upstream `cctx.consumedSrcSize`. Running total of source
    /// bytes consumed this frame.
    pub consumedSrcSize: u64,
    /// Upstream `cctx.producedCSize`. Running total of compressed
    /// bytes emitted this frame.
    pub producedCSize: u64,
    /// Upstream `cctx.pledgedSrcSizePlusOne`. One plus the pledged
    /// content size, or 0 if unknown. The `+1` encoding lets zero
    /// disambiguate "unknown" from "pledged 0".
    pub pledgedSrcSizePlusOne: u64,
    /// Upstream `cctx.isFirstBlock`. True until the first block has
    /// been emitted; used to gate the RLE-downgrade path.
    pub isFirstBlock: i32,
    /// Upstream `cctx.bmi2`. Cached runtime CPU feature probe used
    /// by entropy/literal encoders to select BMI2-specialized paths.
    pub bmi2: i32,
    /// Upstream `cctx.initialized`. Tracks whether the CCtx has gone
    /// through a successful `ZSTD_resetCCtx_internal()` at least once.
    pub initialized: bool,
    /// Upstream `cctx.blockSizeMax`. Max block size for the current
    /// session; derived from `params.cParams.windowLog` capped at
    /// `ZSTD_BLOCKSIZE_MAX`.
    pub blockSizeMax: usize,
    /// Upstream `cctx.xxhState`. XXH64 accumulator for the optional
    /// frame-end checksum.
    pub xxhState: crate::common::xxhash::XXH64_state_t,
    /// Upstream `cctx.appliedParams.format`. Active frame format —
    /// `ZSTD_f_zstd1` (default, magic-prefixed) or
    /// `ZSTD_f_zstd1_magicless`. Set via `ZSTD_CCtx_setFormat`;
    /// `ZSTD_compressFrame_fast` / `_with_prefix` thread it through
    /// `ZSTD_writeFrameHeader_advanced`.
    pub format: crate::decompress::zstd_decompress::ZSTD_format_e,

    /// Upstream-equivalent of `cctx.prefixDict` being non-empty
    /// (`zstd_compress.c:6378`). When set, `stream_dict` was
    /// attached via `ZSTD_CCtx_refPrefix` and is single-use — the
    /// stream compressor auto-clears it after the next frame.
    /// `loadDictionary` / `refCDict` leave this `false` so the
    /// dict persists (matches upstream's `localDict` / `cdict`
    /// fields which aren't zeroed at compress start).
    pub prefix_is_single_use: bool,
    /// Upstream custom allocator bundle requested for this context.
    pub customMem: ZSTD_customMem,
    /// Upstream `cctx.staticSize`. Non-zero when initialized from a
    /// caller-provided static workspace.
    pub staticSize: usize,
}

impl Default for ZSTD_CCtx {
    fn default() -> Self {
        let mut requestedParams = ZSTD_CCtx_params::default();
        let _ = ZSTD_CCtxParams_init(&mut requestedParams, ZSTD_CLEVEL_DEFAULT);
        let appliedParams = requestedParams;
        Self {
            ms: None,
            seqStore: None,
            prevEntropy: ZSTD_entropyCTables_t::default(),
            nextEntropy: ZSTD_entropyCTables_t::default(),
            // Upstream's `repStartValue = {1, 4, 8}` — the canonical
            // initial repcode history at frame start.
            prev_rep: [1, 4, 8],
            next_rep: [1, 4, 8],
            blockSplitCtx: None,
            entropyScratch: ZSTD_entropyScratch::default(),
            externalMatchStore: None,
            ldmState: None,
            ldmSequences: crate::compress::zstd_ldm::RawSeqStore_t::default(),
            threadPoolRef: 0,
            rayonThreadPoolRef: 0,
            mtctxSizeHint: 0,
            stream_level: None,
            pledged_src_size: None,
            stream_dict: Vec::new(),
            stream_dict_original: Vec::new(),
            stream_dict_content_type:
                crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_auto,
            stream_cdict: None,
            param_checksum: false,
            param_contentSize: true, // upstream default
            param_dictID: true,      // upstream default
            stream_in_buffer: Vec::new(),
            stream_params_snapshot: None,
            stream_out_buffer: Vec::new(),
            stream_out_drained: 0,
            stream_stage: ZSTD_cStreamStage::zcss_init,
            stream_in_to_compress: 0,
            stream_in_target: 0,
            stream_frame_ended: false,
            stream_window_base: crate::compress::match_state::ZSTD_WINDOW_START_INDEX,
            stream_disable_windowed: false,
            expected_in_src: 0,
            expected_in_size: 0,
            expected_in_pos: 0,
            expected_out_buffer_size: 0,
            buffer_expectations_set: false,
            stream_closed: false,
            stream_frame_completed: false,
            requested_cParams: None,
            requestedParams,
            appliedParams,
            stage: ZSTD_compressionStage_e::default(),
            dictID: 0,
            dictContentSize: 0,
            consumedSrcSize: 0,
            producedCSize: 0,
            pledgedSrcSizePlusOne: 0,
            isFirstBlock: 1,
            bmi2: crate::common::zstd_internal::ZSTD_cpuSupportsBmi2(),
            initialized: false,
            blockSizeMax: 0,
            xxhState: crate::common::xxhash::XXH64_state_t::default(),
            format: crate::decompress::zstd_decompress::ZSTD_format_e::ZSTD_f_zstd1,
            prefix_is_single_use: false,
            customMem: ZSTD_customMem::default(),
            staticSize: 0,
        }
    }
}

/// Port of `ZSTD_blockSplitCtx` (`zstd_compress_internal.h:463`).
/// Keeps reusable seq-store scratch for recursive block-split
/// estimation plus the final partition table.
#[derive(Debug, Clone)]
pub struct ZSTD_blockSplitCtx {
    pub fullSeqStoreChunk: SeqStore_t,
    pub firstHalfSeqStore: SeqStore_t,
    pub secondHalfSeqStore: SeqStore_t,
    pub currSeqStore: SeqStore_t,
    pub nextSeqStore: SeqStore_t,
    pub partitions: [u32; ZSTD_MAX_NB_BLOCK_SPLITS],
    pub entropyMetadata: ZSTD_entropyCTablesMetadata_t,
    pub entropyScratch: ZSTD_entropyScratch,
}

impl Default for ZSTD_blockSplitCtx {
    fn default() -> Self {
        let mk = || SeqStore_t::with_capacity(ZSTD_BLOCKSIZE_MAX / 3, ZSTD_BLOCKSIZE_MAX);
        Self {
            fullSeqStoreChunk: mk(),
            firstHalfSeqStore: mk(),
            secondHalfSeqStore: mk(),
            currSeqStore: mk(),
            nextSeqStore: mk(),
            partitions: [0; ZSTD_MAX_NB_BLOCK_SPLITS],
            entropyMetadata: ZSTD_entropyCTablesMetadata_t::default(),
            entropyScratch: ZSTD_entropyScratch::default(),
        }
    }
}

/// Upstream `STREAM_ACCUMULATOR_MIN` — lifted here to avoid a circular
/// `zstd_compress_sequences` import when `ZSTD_seqToCodes` checks
/// whether an offCode crosses the long-offset split threshold.
#[inline]
const fn STREAM_ACCUMULATOR_MIN() -> u32 {
    if MEM_32bits() != 0 {
        25
    } else {
        57
    }
}

/// Port of `ZSTD_LLcode`. Maps a literal length to its FSE-table
/// symbol code. For `litLength <= 63` we index a 64-entry lookup table;
/// for larger values we use `highbit32 + 19`.
#[inline]
pub fn ZSTD_LLcode(litLength: u32) -> u32 {
    const LL_CODE: [u8; 64] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20,
        20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23,
        24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    ];
    const LL_DELTA: u32 = 19;
    if litLength > 63 {
        ZSTD_highbit32(litLength) + LL_DELTA
    } else {
        LL_CODE[litLength as usize] as u32
    }
}

/// Port of `ZSTD_MLcode`. `mlBase` = matchLength - MINMATCH.
#[inline]
pub fn ZSTD_MLcode(mlBase: u32) -> u32 {
    const ML_CODE: [u8; 128] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37,
        38, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 40, 40,
        40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
        41, 41, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
        42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    ];
    const ML_DELTA: u32 = 36;
    if mlBase > 127 {
        ZSTD_highbit32(mlBase) + ML_DELTA
    } else {
        ML_CODE[mlBase as usize] as u32
    }
}

/// Port of `ZSTD_seqToCodes`. Converts the SeqStore's sequences into
/// the three per-sequence code tables (llCode / ofCode / mlCode) in
/// place. Sets `MaxLL` / `MaxML` at the `longLengthPos` when the
/// store flagged a long-length overflow. Returns 1 if any offCode
/// crosses the 32-bit long-offset threshold (always 0 on 64-bit
/// targets, matching upstream's assert).
pub fn ZSTD_seqToCodes(seqStore: &mut SeqStore_t) -> i32 {
    let nbSeq = seqStore.sequences.len();
    seqStore.llCode.resize(nbSeq, 0);
    seqStore.ofCode.resize(nbSeq, 0);
    seqStore.mlCode.resize(nbSeq, 0);

    let mut longOffsets = 0i32;
    for (u, seq) in seqStore.sequences.iter().enumerate() {
        let ofCode = ZSTD_highbit32(seq.offBase);
        seqStore.llCode[u] = ZSTD_LLcode(seq.litLength as u32) as u8;
        seqStore.ofCode[u] = ofCode as u8;
        seqStore.mlCode[u] = ZSTD_MLcode(seq.mlBase as u32) as u8;
        debug_assert!(!(MEM_64bits() != 0 && ofCode >= STREAM_ACCUMULATOR_MIN()));
        if MEM_32bits() != 0 && ofCode >= STREAM_ACCUMULATOR_MIN() {
            longOffsets = 1;
        }
    }
    if seqStore.longLengthType == ZSTD_longLengthType_e::ZSTD_llt_literalLength {
        seqStore.llCode[seqStore.longLengthPos as usize] = MaxLL as u8;
    }
    if seqStore.longLengthType == ZSTD_longLengthType_e::ZSTD_llt_matchLength {
        seqStore.mlCode[seqStore.longLengthPos as usize] = MaxML as u8;
    }
    longOffsets
}

/// Port of `ZSTD_createCCtx`. Returns an empty context — all heap
/// allocations happen lazily on the first `ZSTD_compressCCtx` call.
pub fn ZSTD_createCCtx() -> Option<Box<ZSTD_CCtx>> {
    ZSTD_createCCtx_advanced(ZSTD_customMem::default())
}

/// Port of `ZSTD_freeCCtx`. Drop the context; Rust's `Box` handles
/// deallocation. Returns 0 (upstream returns 0 on success).
pub fn ZSTD_freeCCtx(cctx: Option<Box<ZSTD_CCtx>>) -> usize {
    if let Some(cctx) = cctx {
        let customMem = cctx.customMem;
        unsafe {
            ZSTD_customFreeBox(cctx, customMem);
        }
    }
    0
}

/// Port of `ZSTD_bounds`. Returned by `ZSTD_cParam_getBounds` /
/// `ZSTD_dParam_getBounds`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZSTD_bounds {
    /// Error code (0 = success).
    pub error: usize,
    /// Lower bound (inclusive).
    pub lowerBound: i32,
    /// Upper bound (inclusive).
    pub upperBound: i32,
}

/// Port of `ZSTD_cParam_getBounds`. Returns the valid range for a
/// compression parameter.
pub fn ZSTD_cParam_getBounds(param: ZSTD_cParameter) -> ZSTD_bounds {
    let mut bounds = ZSTD_bounds {
        error: 0,
        lowerBound: 0,
        upperBound: 0,
    };

    match param {
        ZSTD_cParameter::ZSTD_c_compressionLevel => {
            bounds.lowerBound = ZSTD_minCLevel();
            bounds.upperBound = ZSTD_maxCLevel();
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_windowLog => {
            bounds.lowerBound = ZSTD_WINDOWLOG_MIN as i32;
            bounds.upperBound = ZSTD_WINDOWLOG_MAX() as i32;
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_hashLog => {
            bounds.lowerBound = crate::compress::zstd_ldm::ZSTD_HASHLOG_MIN as i32;
            bounds.upperBound = crate::compress::zstd_ldm::ZSTD_HASHLOG_MAX as i32;
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_chainLog => {
            bounds.lowerBound = ZSTD_CHAINLOG_MIN as i32;
            bounds.upperBound = ZSTD_CHAINLOG_MAX() as i32;
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_searchLog => {
            bounds.lowerBound = ZSTD_SEARCHLOG_MIN as i32;
            bounds.upperBound = ZSTD_SEARCHLOG_MAX() as i32;
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_minMatch => {
            bounds.lowerBound = ZSTD_MINMATCH_MIN as i32;
            bounds.upperBound = ZSTD_MINMATCH_MAX as i32;
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_targetLength => {
            bounds.lowerBound = ZSTD_TARGETLENGTH_MIN as i32;
            bounds.upperBound = ZSTD_TARGETLENGTH_MAX as i32;
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_strategy => {
            bounds.lowerBound = ZSTD_STRATEGY_MIN as i32;
            bounds.upperBound = ZSTD_STRATEGY_MAX as i32;
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_targetCBlockSize => {
            bounds.lowerBound = ZSTD_TARGETCBLOCKSIZE_MIN as i32;
            bounds.upperBound = ZSTD_TARGETCBLOCKSIZE_MAX as i32;
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_enableLongDistanceMatching
        | ZSTD_cParameter::ZSTD_c_literalCompressionMode
        | ZSTD_cParameter::ZSTD_c_splitAfterSequences
        | ZSTD_cParameter::ZSTD_c_useRowMatchFinder
        | ZSTD_cParameter::ZSTD_c_prefetchCDictTables
        | ZSTD_cParameter::ZSTD_c_repcodeResolution => {
            use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
            bounds.lowerBound = ZSTD_ParamSwitch_e::ZSTD_ps_auto as i32;
            bounds.upperBound = ZSTD_ParamSwitch_e::ZSTD_ps_disable as i32;
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_ldmHashLog => {
            bounds.lowerBound = ZSTD_LDM_HASHLOG_MIN as i32;
            bounds.upperBound = ZSTD_LDM_HASHLOG_MAX as i32;
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_ldmMinMatch => {
            bounds.lowerBound = ZSTD_LDM_MINMATCH_MIN as i32;
            bounds.upperBound = ZSTD_LDM_MINMATCH_MAX as i32;
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_ldmBucketSizeLog => {
            bounds.lowerBound = ZSTD_LDM_BUCKETSIZELOG_MIN as i32;
            bounds.upperBound = crate::compress::zstd_ldm::ZSTD_LDM_BUCKETSIZELOG_MAX as i32;
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_ldmHashRateLog => {
            bounds.lowerBound = ZSTD_LDM_HASHRATELOG_MIN as i32;
            bounds.upperBound = ZSTD_LDM_HASHRATELOG_MAX() as i32;
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_checksumFlag => {
            bounds.lowerBound = 0;
            bounds.upperBound = 1;
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_contentSizeFlag => {
            bounds.lowerBound = 0;
            bounds.upperBound = 1;
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_dictIDFlag => {
            bounds.lowerBound = 0;
            bounds.upperBound = 1;
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_format => {
            use crate::decompress::zstd_decompress::ZSTD_format_e;
            bounds.lowerBound = ZSTD_format_e::ZSTD_f_zstd1 as i32;
            bounds.upperBound = ZSTD_format_e::ZSTD_f_zstd1_magicless as i32;
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_nbWorkers => {
            bounds.lowerBound = 0;
            bounds.upperBound = {
                #[cfg(feature = "mt")]
                {
                    200
                }
                #[cfg(not(feature = "mt"))]
                {
                    0
                }
            };
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_rsyncable
        | ZSTD_cParameter::ZSTD_c_forceMaxWindow
        | ZSTD_cParameter::ZSTD_c_enableDedicatedDictSearch
        | ZSTD_cParameter::ZSTD_c_deterministicRefPrefix
        | ZSTD_cParameter::ZSTD_c_validateSequences
        | ZSTD_cParameter::ZSTD_c_enableSeqProducerFallback => {
            bounds.lowerBound = 0;
            bounds.upperBound = 1;
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_forceAttachDict => {
            bounds.lowerBound = ZSTD_dictAttachPref_e::ZSTD_dictDefaultAttach as i32;
            bounds.upperBound = ZSTD_dictAttachPref_e::ZSTD_dictForceLoad as i32;
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_stableInBuffer | ZSTD_cParameter::ZSTD_c_stableOutBuffer => {
            bounds.lowerBound = ZSTD_bufferMode_e::ZSTD_bm_buffered as i32;
            bounds.upperBound = ZSTD_bufferMode_e::ZSTD_bm_stable as i32;
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_blockDelimiters => {
            bounds.lowerBound = ZSTD_SequenceFormat_e::ZSTD_sf_noBlockDelimiters as i32;
            bounds.upperBound = ZSTD_SequenceFormat_e::ZSTD_sf_explicitBlockDelimiters as i32;
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_blockSplitterLevel => {
            bounds.lowerBound = 0;
            bounds.upperBound = 6;
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_jobSize => {
            bounds.lowerBound = 0;
            bounds.upperBound = {
                #[cfg(feature = "mt")]
                {
                    crate::compress::zstdmt_compress::ZSTDMT_JOBSIZE_MAX as i32
                }
                #[cfg(not(feature = "mt"))]
                {
                    0
                }
            };
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_overlapLog => {
            bounds.lowerBound = 0;
            bounds.upperBound = {
                #[cfg(feature = "mt")]
                {
                    9
                }
                #[cfg(not(feature = "mt"))]
                {
                    0
                }
            };
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_srcSizeHint => {
            bounds.lowerBound = ZSTD_SRCSIZEHINT_MIN;
            bounds.upperBound = ZSTD_SRCSIZEHINT_MAX;
            return bounds;
        }
        ZSTD_cParameter::ZSTD_c_maxBlockSize => {
            bounds.lowerBound = ZSTD_BLOCKSIZE_MAX_MIN as i32;
            bounds.upperBound = ZSTD_BLOCKSIZE_MAX as i32;
            return bounds;
        }
    }
}

/// Port of `ZSTD_WINDOWLOG_MAX`. Compile-time constant in upstream
/// via `sizeof(size_t) == 4 ? 30 : 31`. Rust port follows suit.
pub const fn ZSTD_WINDOWLOG_MAX() -> u32 {
    if crate::common::mem::MEM_32bits() != 0 {
        30
    } else {
        31
    }
}

/// Re-export of `crate::decompress::zstd_decompress::ZSTD_WINDOWLOG_ABSOLUTEMIN`
/// — same upstream constant, previously duplicated on both sides.
pub use crate::decompress::zstd_decompress::ZSTD_WINDOWLOG_ABSOLUTEMIN;

/// Upstream `ZSTD_WINDOWLOG_MIN` (`zstd.h:1266`). Minimum windowLog
/// accepted by `ZSTD_c_windowLog` — below this the compressor rejects
/// the param.
pub const ZSTD_WINDOWLOG_MIN: u32 = 10;

/// Upstream `ZSTD_CHAINLOG_MAX_32` / `ZSTD_CHAINLOG_MAX_64`
/// (`zstd.h:1269-1270`). Max chain-table log, differs by target
/// width.
pub const ZSTD_CHAINLOG_MAX_32: u32 = 29;
pub const ZSTD_CHAINLOG_MAX_64: u32 = 30;

/// Upstream `ZSTD_CHAINLOG_MAX`. Resolved at runtime against the
/// current pointer width.
pub const fn ZSTD_CHAINLOG_MAX() -> u32 {
    if crate::common::mem::MEM_32bits() != 0 {
        ZSTD_CHAINLOG_MAX_32
    } else {
        ZSTD_CHAINLOG_MAX_64
    }
}

/// Upstream `ZSTD_CHAINLOG_MIN` (= `ZSTD_HASHLOG_MIN`).
pub const ZSTD_CHAINLOG_MIN: u32 = 6;

/// Upstream `ZSTD_SEARCHLOG_MAX` — `windowLogMax - 1`.
pub const fn ZSTD_SEARCHLOG_MAX() -> u32 {
    ZSTD_WINDOWLOG_MAX() - 1
}

/// Upstream `ZSTD_SEARCHLOG_MIN`.
pub const ZSTD_SEARCHLOG_MIN: u32 = 1;

/// Upstream `ZSTD_MINMATCH_MIN` / `ZSTD_MINMATCH_MAX` (`zstd.h:1275-76`).
pub const ZSTD_MINMATCH_MIN: u32 = 3;
pub const ZSTD_MINMATCH_MAX: u32 = 7;

/// Upstream `ZSTD_TARGETLENGTH_MIN` / `ZSTD_TARGETLENGTH_MAX`
/// (`zstd.h:1277-78`).
pub const ZSTD_TARGETLENGTH_MIN: u32 = 0;
pub const ZSTD_TARGETLENGTH_MAX: u32 =
    crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX as u32;

/// Upstream `ZSTD_STRATEGY_MIN` / `MAX` (`zstd.h:1279-80`).
pub const ZSTD_STRATEGY_MIN: u32 = 1; // ZSTD_fast
pub const ZSTD_STRATEGY_MAX: u32 = 9; // ZSTD_btultra2

/// Upstream `ZSTD_BLOCKSIZE_MAX_MIN` (`zstd.h:1281`). Smallest valid
/// `maxBlockSize` param — below this, `ZSTD_compressBound` becomes
/// inaccurate.
pub const ZSTD_BLOCKSIZE_MAX_MIN: usize = 1 << 10;

/// Upstream `ZSTD_OVERLAPLOG_MIN` / `MAX` (`zstd.h:1284-85`). Controls
/// the LDM `overlapLog` param — overlap size between MT worker jobs.
pub const ZSTD_OVERLAPLOG_MIN: u32 = 0;
pub const ZSTD_OVERLAPLOG_MAX: u32 = 9;

/// Upstream `ZSTD_LDM_HASHLOG_MIN` / `MAX` (`zstd.h:1295-96`). Aliased
/// onto the ZSTD-wide hashLog bounds so the LDM parametric API can
/// share validation.
pub const ZSTD_LDM_HASHLOG_MIN: u32 = 6; // ZSTD_HASHLOG_MIN
pub const ZSTD_LDM_HASHLOG_MAX: u32 = 30; // ZSTD_HASHLOG_MAX

/// Upstream `ZSTD_LDM_MINMATCH_MIN` / `MAX` (`zstd.h:1297-98`).
pub const ZSTD_LDM_MINMATCH_MIN: u32 = 4;
pub const ZSTD_LDM_MINMATCH_MAX: u32 = 4096;

/// Upstream `ZSTD_LDM_BUCKETSIZELOG_MIN` (`zstd.h:1299`).
pub const ZSTD_LDM_BUCKETSIZELOG_MIN: u32 = 1;

/// Upstream `ZSTD_LDM_HASHRATELOG_MIN` / `MAX` (`zstd.h:1301-02`).
pub const ZSTD_LDM_HASHRATELOG_MIN: u32 = 0;
pub const fn ZSTD_LDM_HASHRATELOG_MAX() -> u32 {
    use crate::compress::zstd_ldm::ZSTD_HASHLOG_MIN;
    ZSTD_WINDOWLOG_MAX() - ZSTD_HASHLOG_MIN
}

/// Upstream `ZSTD_TARGETCBLOCKSIZE_MIN` / `MAX` (`zstd.h:1305-06`).
/// Target per-block compressed size for the block splitter; 1340
/// fits an Ethernet/wifi/4G transport frame.
pub const ZSTD_TARGETCBLOCKSIZE_MIN: u32 = 1340;
pub const ZSTD_TARGETCBLOCKSIZE_MAX: u32 =
    crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX as u32;

/// Port of `cmpgtz_any_s8`.
///
/// Upstream uses SVE predicates; the Rust scalar fallback answers the
/// same question for a plain byte slice.
pub fn cmpgtz_any_s8(bytes: &[i8]) -> i32 {
    bytes.iter().any(|&b| b > 0) as i32
}

/// Upstream `ZSTD_SRCSIZEHINT_MIN` / `MAX` (`zstd.h:1307-08`).
pub const ZSTD_SRCSIZEHINT_MIN: i32 = 0;
pub const ZSTD_SRCSIZEHINT_MAX: i32 = i32::MAX;

/// Port of `ZSTD_cycleLog`. The btlazy2 family stores 2 chains per
/// `chainLog` entry (left + right), so its effective cycle length is
/// `chainLog - 1`. Other strategies use the full chainLog.
#[inline]
pub fn ZSTD_cycleLog(hashLog: u32, strat: u32) -> u32 {
    // ZSTD_btlazy2 = 6.
    let btScale: u32 = if strat >= 6 { 1 } else { 0 };
    hashLog - btScale
}

/// Port of `ZSTD_checkCParams`. Validates every cParam field against
/// the upstream-documented bounds. Returns 0 on success, or a zstd
/// error code when any field is out of range. Like upstream, every
/// field is checked through `ZSTD_cParam_getBounds()`.
pub fn ZSTD_checkCParams(
    cParams: crate::compress::match_state::ZSTD_compressionParameters,
) -> usize {
    use crate::common::error::{ErrorCode, ERROR};

    let check_bound = |param: ZSTD_cParameter, value: i32| -> usize {
        let bounds = ZSTD_cParam_getBounds(param);
        if ERR_isError(bounds.error) {
            return bounds.error;
        }
        if value < bounds.lowerBound || value > bounds.upperBound {
            return ERROR(ErrorCode::ParameterOutOfBound);
        }
        0
    };

    let rc = check_bound(ZSTD_cParameter::ZSTD_c_windowLog, cParams.windowLog as i32);
    if ERR_isError(rc) {
        return rc;
    }
    let rc = check_bound(ZSTD_cParameter::ZSTD_c_chainLog, cParams.chainLog as i32);
    if ERR_isError(rc) {
        return rc;
    }
    let rc = check_bound(ZSTD_cParameter::ZSTD_c_hashLog, cParams.hashLog as i32);
    if ERR_isError(rc) {
        return rc;
    }
    let rc = check_bound(ZSTD_cParameter::ZSTD_c_searchLog, cParams.searchLog as i32);
    if ERR_isError(rc) {
        return rc;
    }
    let rc = check_bound(ZSTD_cParameter::ZSTD_c_minMatch, cParams.minMatch as i32);
    if ERR_isError(rc) {
        return rc;
    }
    let rc = check_bound(
        ZSTD_cParameter::ZSTD_c_targetLength,
        cParams.targetLength as i32,
    );
    if ERR_isError(rc) {
        return rc;
    }
    let rc = check_bound(ZSTD_cParameter::ZSTD_c_strategy, cParams.strategy as i32);
    if ERR_isError(rc) {
        return rc;
    }
    0
}

/// Port of `ZSTD_clampCParams`. Clamps each compression field through
/// the same `ZSTD_cParam_getBounds()` ranges that
/// `ZSTD_checkCParams()` validates against.
fn ZSTD_clampCParams(
    mut cPar: crate::compress::match_state::ZSTD_compressionParameters,
) -> crate::compress::match_state::ZSTD_compressionParameters {
    fn clamp_field(param: ZSTD_cParameter, value: &mut u32) {
        let mut signed = *value as i32;
        let rc = ZSTD_cParam_clampBounds(param, &mut signed);
        debug_assert!(!ERR_isError(rc));
        *value = signed as u32;
    }

    clamp_field(ZSTD_cParameter::ZSTD_c_windowLog, &mut cPar.windowLog);
    clamp_field(ZSTD_cParameter::ZSTD_c_chainLog, &mut cPar.chainLog);
    clamp_field(ZSTD_cParameter::ZSTD_c_hashLog, &mut cPar.hashLog);
    clamp_field(ZSTD_cParameter::ZSTD_c_searchLog, &mut cPar.searchLog);
    clamp_field(ZSTD_cParameter::ZSTD_c_minMatch, &mut cPar.minMatch);
    clamp_field(ZSTD_cParameter::ZSTD_c_targetLength, &mut cPar.targetLength);
    clamp_field(ZSTD_cParameter::ZSTD_c_strategy, &mut cPar.strategy);
    cPar
}

/// Port of `ZSTD_adjustCParams`. Public wrapper around
/// `ZSTD_adjustCParams_internal`. Upstream first clamps the cParams
/// via `ZSTD_clampCParams`; the Rust port now does the same through
/// the shared bounds table instead of hand-coded per-field limits.
pub fn ZSTD_adjustCParams(
    cPar: crate::compress::match_state::ZSTD_compressionParameters,
    srcSize: u64,
    dictSize: u64,
) -> crate::compress::match_state::ZSTD_compressionParameters {
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;

    ZSTD_adjustCParams_internal(
        ZSTD_clampCParams(cPar),
        srcSize,
        dictSize,
        ZSTD_CParamMode_e::ZSTD_cpm_unknown,
        ZSTD_ParamSwitch_e::ZSTD_ps_auto,
    )
}

/// Port of `ZSTD_adjustCParams_internal`. Shrinks windowLog / hashLog
/// / chainLog so they're right-sized for the actual data volume —
/// big caps waste allocation time on tiny inputs, and the hash
/// family can't span more than 32 bits of state total.
///
/// Upstream's `ZSTD_EXCLUDE_*` compile-time strategy-cascade block is
/// skipped here; we never define those exclusion macros.
pub fn ZSTD_adjustCParams_internal(
    mut cPar: crate::compress::match_state::ZSTD_compressionParameters,
    mut srcSize: u64,
    mut dictSize: u64,
    mode: ZSTD_CParamMode_e,
    mut useRowMatchFinder: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
) -> crate::compress::match_state::ZSTD_compressionParameters {
    use crate::common::bits::ZSTD_highbit32;
    use crate::compress::match_state::{ZSTD_CDictIndicesAreTagged, ZSTD_SHORT_CACHE_TAG_BITS};
    use crate::compress::zstd_ldm::{ZSTD_ParamSwitch_e, ZSTD_HASHLOG_MIN};
    use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;

    const MIN_SRC_SIZE: u64 = 513; // (1<<9) + 1
    let max_wlog = ZSTD_WINDOWLOG_MAX();
    let maxWindowResize: u64 = 1u64 << (max_wlog - 1);

    match mode {
        ZSTD_CParamMode_e::ZSTD_cpm_unknown | ZSTD_CParamMode_e::ZSTD_cpm_noAttachDict => {}
        ZSTD_CParamMode_e::ZSTD_cpm_createCDict => {
            if dictSize != 0 && srcSize == ZSTD_CONTENTSIZE_UNKNOWN {
                srcSize = MIN_SRC_SIZE;
            }
        }
        ZSTD_CParamMode_e::ZSTD_cpm_attachDict => {
            dictSize = 0;
        }
    }

    // Resize windowLog if total input fits comfortably.
    if srcSize <= maxWindowResize && dictSize <= maxWindowResize {
        let tSize = srcSize.wrapping_add(dictSize) as u32;
        let hashSizeMin: u32 = 1u32 << ZSTD_HASHLOG_MIN;
        let srcLog = if tSize < hashSizeMin {
            ZSTD_HASHLOG_MIN
        } else {
            ZSTD_highbit32(tSize - 1) + 1
        };
        if cPar.windowLog > srcLog {
            cPar.windowLog = srcLog;
        }
    }

    if srcSize != ZSTD_CONTENTSIZE_UNKNOWN {
        let dictAndWindowLog = ZSTD_dictAndWindowLog(cPar.windowLog, srcSize, dictSize);
        let cycleLog = ZSTD_cycleLog(cPar.chainLog, cPar.strategy);
        if cPar.hashLog > dictAndWindowLog + 1 {
            cPar.hashLog = dictAndWindowLog + 1;
        }
        if cycleLog > dictAndWindowLog {
            cPar.chainLog -= cycleLog - dictAndWindowLog;
        }
    }

    if cPar.windowLog < ZSTD_WINDOWLOG_ABSOLUTEMIN {
        cPar.windowLog = ZSTD_WINDOWLOG_ABSOLUTEMIN;
    }

    // Tag-bit budget: fast/dfast CDicts pack an 8-bit tag into
    // hashTable entries, so hashLog + chainLog must leave 8 bits free.
    if mode == ZSTD_CParamMode_e::ZSTD_cpm_createCDict && ZSTD_CDictIndicesAreTagged(&cPar) {
        let maxShortCacheHashLog = 32 - ZSTD_SHORT_CACHE_TAG_BITS;
        if cPar.hashLog > maxShortCacheHashLog {
            cPar.hashLog = maxShortCacheHashLog;
        }
        if cPar.chainLog > maxShortCacheHashLog {
            cPar.chainLog = maxShortCacheHashLog;
        }
    }

    // Row match finder default: conservatively assume enabled so we
    // shrink hashLog preemptively.
    if useRowMatchFinder == ZSTD_ParamSwitch_e::ZSTD_ps_auto {
        useRowMatchFinder = ZSTD_ParamSwitch_e::ZSTD_ps_enable;
    }
    /* We can't hash more than 32-bits in total. So that means that we require:
     * (hashLog - rowLog + 8) <= 32
     */
    if crate::compress::match_state::ZSTD_rowMatchFinderUsed(cPar.strategy, useRowMatchFinder) {
        let rowLog = cPar.searchLog.clamp(4, 6);
        let maxRowHashLog = 32 - crate::compress::zstd_lazy::ZSTD_ROW_HASH_TAG_BITS;
        let maxHashLog = maxRowHashLog + rowLog;
        debug_assert!(cPar.hashLog >= rowLog);
        if cPar.hashLog > maxHashLog {
            cPar.hashLog = maxHashLog;
        }
    }

    cPar
}

/// Port of `ZSTD_dictAndWindowLog`. Picks the smallest windowLog
/// that covers both the dict and the (known) src, capped at
/// `ZSTD_WINDOWLOG_MAX`. No-op when there's no dict.
pub fn ZSTD_dictAndWindowLog(windowLog: u32, srcSize: u64, dictSize: u64) -> u32 {
    if dictSize == 0 {
        return windowLog;
    }
    let max_wlog = ZSTD_WINDOWLOG_MAX();
    debug_assert!(windowLog <= max_wlog);
    let maxWindowSize: u64 = 1u64 << max_wlog;
    let windowSize: u64 = 1u64 << windowLog;
    let dictAndWindowSize = dictSize + windowSize;
    if windowSize >= dictSize + srcSize {
        windowLog
    } else if dictAndWindowSize >= maxWindowSize {
        max_wlog
    } else {
        // highbit32((u32)(dictAndWindowSize - 1)) + 1
        crate::common::bits::ZSTD_highbit32((dictAndWindowSize - 1) as u32) + 1
    }
}

/// Port of `ZSTD_cParam_clampBounds`. Clamps `*value` into the param's
/// valid range, returning 0 on success or a zstd error code on
/// `ZSTD_cParam_getBounds` failure.
pub fn ZSTD_cParam_clampBounds(cParam: ZSTD_cParameter, value: &mut i32) -> usize {
    let bounds = ZSTD_cParam_getBounds(cParam);
    if crate::common::error::ERR_isError(bounds.error) {
        return bounds.error;
    }
    if *value < bounds.lowerBound {
        *value = bounds.lowerBound;
    }
    if *value > bounds.upperBound {
        *value = bounds.upperBound;
    }
    0
}

/// Port of `ZSTD_maxNbSeq`. Upper bound on the number of sequences a
/// block of `blockSize` bytes can emit, given `minMatch` and whether
/// an external sequence producer is active.
#[inline]
pub fn ZSTD_maxNbSeq(blockSize: usize, minMatch: u32, useSequenceProducer: bool) -> usize {
    let divider = if minMatch == 3 || useSequenceProducer {
        3
    } else {
        4
    };
    blockSize / divider
}

/// Port of `ZSTD_CParamMode_e`. Tells cParam selection helpers how to
/// weigh `dictSize` vs. `srcSize` when picking a row from the default
/// cParam tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZSTD_CParamMode_e {
    /// Regular or extDict compression — both sizes count.
    #[default]
    ZSTD_cpm_noAttachDict = 0,
    /// dictMatchState / dedicatedDictSearch — only srcSize counts.
    ZSTD_cpm_attachDict = 1,
    /// Creating a CDict — both sizes count (like noAttachDict).
    ZSTD_cpm_createCDict = 2,
    /// Public `getCParams` / `getParams` / `adjustParams` — treat as
    /// legacy (both sizes count).
    ZSTD_cpm_unknown = 3,
}

/// Port of `ZSTD_getCParamRowSize`. Combines `srcSizeHint` and
/// `dictSize` into a single "effective size" used to pick a tableID
/// for the default-cParams lookup.
///
/// `attachDict` mode disables `dictSize` accumulation since the dict
/// is attached rather than copied into the stream.
pub fn ZSTD_getCParamRowSize(srcSizeHint: u64, dictSize: usize, mode: ZSTD_CParamMode_e) -> u64 {
    use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;
    let effectiveDictSize = if mode == ZSTD_CParamMode_e::ZSTD_cpm_attachDict {
        0
    } else {
        dictSize
    };
    let unknown = srcSizeHint == ZSTD_CONTENTSIZE_UNKNOWN;
    let addedSize: u64 = if unknown && effectiveDictSize > 0 {
        500
    } else {
        0
    };
    if unknown && effectiveDictSize == 0 {
        ZSTD_CONTENTSIZE_UNKNOWN
    } else {
        // Upstream relies on U64 wrap when `srcSizeHint ==
        // ZSTD_CONTENTSIZE_UNKNOWN` and `dictSize > 0`; reproduce
        // with `wrapping_add` so rSize ends up as a tiny value
        // (→ smallest tableID), matching upstream's behavior exactly.
        srcSizeHint
            .wrapping_add(effectiveDictSize as u64)
            .wrapping_add(addedSize)
    }
}

/// Upstream `ZSTD_LAZY_DDSS_BUCKET_LOG`. DDSS hash bucket holds
/// `2^LOG` entries rather than a single index, letting the DDS search
/// scan more candidates per probe.
pub const ZSTD_LAZY_DDSS_BUCKET_LOG: u32 = 2;

/// Port of `ZSTD_dedicatedDictSearch_getCParams` (`zstd_compress.c:8204`).
/// Produces cParams for a CDict being created with the Dedicated
/// Dictionary Search strategy: derives baseline cParams via
/// `ZSTD_getCParams` (cpm_createCDict mode is folded into our
/// `ZSTD_getCParams` via its dictSize handling), then inflates
/// `hashLog` by `ZSTD_LAZY_DDSS_BUCKET_LOG` for lazy-family
/// strategies (greedy/lazy/lazy2). Fast/dfast and bt-family
/// strategies pass through unchanged.
pub fn ZSTD_dedicatedDictSearch_getCParams(
    compressionLevel: i32,
    dictSize: usize,
) -> crate::compress::match_state::ZSTD_compressionParameters {
    use crate::compress::zstd_compress_sequences::{ZSTD_greedy, ZSTD_lazy, ZSTD_lazy2};
    let mut cParams = ZSTD_getCParams(compressionLevel, 0, dictSize);
    if cParams.strategy == ZSTD_greedy
        || cParams.strategy == ZSTD_lazy
        || cParams.strategy == ZSTD_lazy2
    {
        cParams.hashLog += ZSTD_LAZY_DDSS_BUCKET_LOG;
    }
    cParams
}

/// Port of `ZSTD_dedicatedDictSearch_revertCParams`. After DDSS has
/// artificially inflated `hashLog` for bucket expansion, reverse the
/// adjustment so the baseline cParams are suitable for a plain search.
pub fn ZSTD_dedicatedDictSearch_revertCParams(
    cParams: &mut crate::compress::match_state::ZSTD_compressionParameters,
) {
    // Only the lazy family (3..=5) applies the DDSS bucket inflation.
    if (3..=5).contains(&cParams.strategy) {
        use crate::compress::zstd_ldm::ZSTD_HASHLOG_MIN;
        cParams.hashLog = cParams
            .hashLog
            .saturating_sub(ZSTD_LAZY_DDSS_BUCKET_LOG)
            .max(ZSTD_HASHLOG_MIN);
    }
}

/// Port of `ZSTD_dedicatedDictSearch_isSupported`. True when cParams
/// allow the "dedicated dictionary search" optimization — a lazy-
/// family strategy with a hashLog bigger than chainLog and chainLog
/// capped at 24.
#[inline]
pub fn ZSTD_dedicatedDictSearch_isSupported(
    cParams: &crate::compress::match_state::ZSTD_compressionParameters,
) -> bool {
    // ZSTD_greedy = 3, ZSTD_lazy2 = 5.
    (3..=5).contains(&cParams.strategy)
        && cParams.hashLog > cParams.chainLog
        && cParams.chainLog <= 24
}

/// Port of `ZSTD_cParam_withinBounds`. True when `value` lies inside
/// the parameter's valid range. Returns false on any error bound
/// (unknown param) so callers can use this as a `validate-or-reject`
/// gate.
#[inline]
pub fn ZSTD_cParam_withinBounds(cParam: ZSTD_cParameter, value: i32) -> bool {
    let bounds = ZSTD_cParam_getBounds(cParam);
    if crate::common::error::ERR_isError(bounds.error) {
        return false;
    }
    value >= bounds.lowerBound && value <= bounds.upperBound
}

/// Port of `ZSTD_CStream`. In upstream this is `typedef ZSTD_CCtx
/// ZSTD_CStream` — one struct for both one-shot and streaming. Rust
/// type alias matches.
pub type ZSTD_CStream = ZSTD_CCtx;

#[derive(Debug, Clone, Copy)]
pub struct ZSTD_streamParamsSnapshot {
    pub requestedParams: ZSTD_CCtx_params,
    pub stream_level: i32,
    pub param_checksum: bool,
    pub param_contentSize: bool,
    pub param_dictID: bool,
    pub format: crate::decompress::zstd_decompress::ZSTD_format_e,
}

/// Port of `ZSTD_createCStream`. Alias for `ZSTD_createCCtx`.
pub fn ZSTD_createCStream() -> Option<Box<ZSTD_CStream>> {
    ZSTD_createCCtx()
}

/// Port of `ZSTD_freeCStream`. Alias for `ZSTD_freeCCtx`.
pub fn ZSTD_freeCStream(zcs: Option<Box<ZSTD_CStream>>) -> usize {
    ZSTD_freeCCtx(zcs)
}

/// Proxy for upstream's `streamStage == zcss_init` check. Returns true
/// when the CCtx is at a point where dict / prefix / pledge setters
/// are legal — i.e. nothing has been staged or emitted for the
/// current frame yet.
#[inline]
fn cctx_is_in_init_stage(cctx: &ZSTD_CCtx) -> bool {
    cctx.stream_stage == ZSTD_cStreamStage::zcss_init
        && cctx.stream_in_buffer.is_empty()
        && cctx.stream_out_buffer.is_empty()
        && !cctx.stream_closed
}

fn cctx_mark_stream_frame_completed(cctx: &mut ZSTD_CCtx) {
    cctx.stream_in_buffer.clear();
    cctx.stream_params_snapshot = None;
    cctx.stream_out_buffer.clear();
    cctx.stream_out_drained = 0;
    cctx.stream_stage = ZSTD_cStreamStage::zcss_init;
    cctx.stream_in_to_compress = 0;
    cctx.stream_in_target = 0;
    cctx.stream_frame_ended = false;
    cctx.stream_window_base = crate::compress::match_state::ZSTD_WINDOW_START_INDEX;
    cctx.stream_closed = false;
    cctx.stream_frame_completed = true;
    cctx.pledgedSrcSizePlusOne = 0;
    cctx.pledged_src_size = None;
    cctx.consumedSrcSize = 0;
    cctx.initialized = false;
}

#[inline]
fn zstd_snapshot_stream_params(cctx: &ZSTD_CCtx) -> ZSTD_streamParamsSnapshot {
    ZSTD_streamParamsSnapshot {
        requestedParams: cctx.requestedParams,
        stream_level: cctx.stream_level.unwrap_or(ZSTD_CLEVEL_DEFAULT),
        param_checksum: cctx.param_checksum,
        param_contentSize: cctx.param_contentSize,
        param_dictID: cctx.param_dictID,
        format: cctx.format,
    }
}

#[inline]
fn zstd_apply_stream_params_snapshot(cctx: &mut ZSTD_CCtx, snapshot: ZSTD_streamParamsSnapshot) {
    cctx.requestedParams = snapshot.requestedParams;
    cctx.stream_level = Some(snapshot.stream_level);
    cctx.param_checksum = snapshot.param_checksum;
    cctx.param_contentSize = snapshot.param_contentSize;
    cctx.param_dictID = snapshot.param_dictID;
    cctx.format = snapshot.format;
}

/// Port of `ZSTD_CCtx_loadDictionary`. Stashes the dict on the CCtx
/// so subsequent streaming / one-shot compressions use it. Upstream
/// also supports pre-digested dicts via `ZSTD_CCtx_refCDict`; this
/// entry point stores raw bytes, but it does preserve the embedded
/// dictID for magic-prefixed zstd-format dictionaries.
///
/// Upstream (zstd_compress.c:1306) rejects with `StageWrong` when
/// called mid-session — silently swapping dicts between
/// `compressStream` calls would leave the already-buffered bytes
/// compressed against the wrong back-reference substrate.
pub fn ZSTD_CCtx_loadDictionary(cctx: &mut ZSTD_CCtx, dict: &[u8]) -> usize {
    ZSTD_CCtx_loadDictionary_advanced(
        cctx,
        dict,
        crate::decompress::zstd_ddict::ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy,
        crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_auto,
    )
}

/// Port of `ZSTD_CCtx_refPrefix`. Stores `prefix` as raw-content dict
/// on the cctx and flags it as single-use — the next `endStream` /
/// `compress2` auto-clears it so the prefix doesn't persist into a
/// subsequent frame. Matches upstream's `ZSTD_memset(&cctx->prefixDict)`
/// consumption in `ZSTD_CCtx_init_compressStream2`
/// (`zstd_compress.c:6381`). Same init-stage gate as the other
/// dict-family setters.
pub fn ZSTD_CCtx_refPrefix(cctx: &mut ZSTD_CCtx, prefix: &[u8]) -> usize {
    if !cctx_is_in_init_stage(cctx) {
        return ERROR(ErrorCode::StageWrong);
    }
    // Upstream (zstd_compress.c:1372) clears ALL prior dicts first,
    // then installs the new prefix only if non-empty. Mirror that so
    // `refPrefix(&[])` behaves as "clear any bound dict" rather than
    // silently leaving `prefix_is_single_use = true` with an empty
    // stream_dict (which would be an inconsistent state).
    ZSTD_clearAllDicts(cctx);
    if !prefix.is_empty() {
        cctx.stream_dict = prefix.to_vec();
        cctx.stream_dict_original = prefix.to_vec();
        cctx.stream_dict_content_type =
            crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_rawContent;
        cctx.dictContentSize = prefix.len();
        // Prefixes are treated as raw content — `dictID` was just
        // zeroed by `clearAllDicts` so no explicit re-write needed.
        // Upstream (zstd_compress.c:6381): the prefix is single-use
        // — it gets cleared by the next compress. Flag so
        // `endStream` knows to wipe `stream_dict` after emitting.
        cctx.prefix_is_single_use = true;
    }
    0
}

/// Port of `ZSTD_compressBegin`. Legacy "begin / continue / end"
/// entry — starts a new compression session at the given level.
pub fn ZSTD_compressBegin(cctx: &mut ZSTD_CCtx, compressionLevel: i32) -> usize {
    use crate::decompress::zstd_ddict::ZSTD_dictContentType_e;
    use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;

    let params = ZSTD_getParams_internal(
        compressionLevel,
        ZSTD_CONTENTSIZE_UNKNOWN,
        0,
        ZSTD_CParamMode_e::ZSTD_cpm_noAttachDict,
    );
    let effective_level = if compressionLevel == 0 {
        ZSTD_CLEVEL_DEFAULT
    } else {
        compressionLevel
    };
    let mut cctxParams = ZSTD_CCtx_params::default();
    ZSTD_CCtxParams_init_internal(&mut cctxParams, &params, effective_level);
    ZSTD_compressBegin_advanced_internal(
        cctx,
        &[],
        ZSTD_dictContentType_e::ZSTD_dct_auto,
        None,
        &cctxParams,
        ZSTD_CONTENTSIZE_UNKNOWN,
    )
}

/// Port of `ZSTD_compressBegin_usingDict`. Variant that begins a
/// session from caller-supplied dictionary bytes; magic-prefixed
/// zstd-format dictionaries keep their dictID, while raw-content
/// dictionaries remain raw history.
pub fn ZSTD_compressBegin_usingDict(
    cctx: &mut ZSTD_CCtx,
    dict: &[u8],
    compressionLevel: i32,
) -> usize {
    use crate::decompress::zstd_ddict::ZSTD_dictContentType_e;
    use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;
    // Match upstream: pick cParams via getParams_internal with
    // cpm_noAttachDict, then initialize params_internal with the
    // caller's level (mapping 0 → CLEVEL_DEFAULT).
    let params = ZSTD_getParams_internal(
        compressionLevel,
        ZSTD_CONTENTSIZE_UNKNOWN,
        dict.len(),
        ZSTD_CParamMode_e::ZSTD_cpm_noAttachDict,
    );
    let effective_level = if compressionLevel == 0 {
        ZSTD_CLEVEL_DEFAULT
    } else {
        compressionLevel
    };
    let mut cctxParams = ZSTD_CCtx_params::default();
    ZSTD_CCtxParams_init_internal(&mut cctxParams, &params, effective_level);
    ZSTD_compressBegin_advanced_internal(
        cctx,
        dict,
        ZSTD_dictContentType_e::ZSTD_dct_auto,
        None,
        &cctxParams,
        ZSTD_CONTENTSIZE_UNKNOWN,
    )
}

/// Port of `ZSTD_compressContinue` (`zstd_compress.c:4886`). Legacy
/// begin/continue/end API entry — thin wrapper over
/// `ZSTD_compressContinue_public`.
pub fn ZSTD_compressContinue(cctx: &mut ZSTD_CCtx, dst: &mut [u8], src: &[u8]) -> usize {
    ZSTD_compressContinue_public(cctx, dst, src)
}

/// Port of `ZSTD_compressEnd` (`zstd_compress.c:5458`). Legacy
/// begin/continue/end API entry — thin wrapper over `ZSTD_compressEnd_public`.
pub fn ZSTD_compressEnd(cctx: &mut ZSTD_CCtx, dst: &mut [u8], src: &[u8]) -> usize {
    ZSTD_compressEnd_public(cctx, dst, src)
}

/// Port of `ZSTD_BlockCompressor_f` (`zstd_compress_internal.h:580`).
/// Function-pointer type for a single-block match-finder — takes
/// match state, seq store, repcode array, source bytes, and returns
/// the bytes processed (or an error).
pub type ZSTD_BlockCompressor_f = fn(
    ms: &mut crate::compress::match_state::ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; crate::compress::seq_store::ZSTD_REP_NUM],
    src: &[u8],
) -> usize;

/// Port of `ZSTD_selectBlockCompressor` (`zstd_compress.c:3093`).
/// Picks the match-finder entry point for a given
/// (strategy × dictMode × useRowMatchFinder) tuple. Greedy/lazy/lazy2
/// dispatch directly to their row-hash wrappers when row mode is
/// enabled; other strategies ignore `useRowMatchFinder`, matching
/// upstream's strategy matrix.
///
/// Strategy 0 is treated as "fast" per upstream's table-slot default.
pub fn ZSTD_selectBlockCompressor(
    strat: u32,
    useRowMatchFinder: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
    dictMode: crate::compress::match_state::ZSTD_dictMode_e,
) -> ZSTD_BlockCompressor_f {
    use crate::compress::match_state::{ZSTD_dictMode_e, ZSTD_rowMatchFinderUsed};
    use crate::compress::zstd_compress_sequences::{
        ZSTD_btlazy2, ZSTD_btopt, ZSTD_btultra, ZSTD_btultra2, ZSTD_dfast, ZSTD_fast, ZSTD_greedy,
        ZSTD_lazy, ZSTD_lazy2,
    };
    use crate::compress::zstd_double_fast::{
        ZSTD_compressBlock_doubleFast, ZSTD_compressBlock_doubleFast_dictMatchState,
        ZSTD_compressBlock_doubleFast_extDict,
    };
    use crate::compress::zstd_fast::{
        ZSTD_compressBlock_fast, ZSTD_compressBlock_fast_dictMatchState,
        ZSTD_compressBlock_fast_extDict,
    };
    use crate::compress::zstd_lazy::{
        ZSTD_compressBlock_btlazy2, ZSTD_compressBlock_btlazy2_dictMatchState,
        ZSTD_compressBlock_btlazy2_extDict, ZSTD_compressBlock_greedy,
        ZSTD_compressBlock_greedy_dedicatedDictSearch,
        ZSTD_compressBlock_greedy_dedicatedDictSearch_row,
        ZSTD_compressBlock_greedy_dictMatchState, ZSTD_compressBlock_greedy_dictMatchState_row,
        ZSTD_compressBlock_greedy_extDict, ZSTD_compressBlock_greedy_extDict_row,
        ZSTD_compressBlock_greedy_row, ZSTD_compressBlock_lazy, ZSTD_compressBlock_lazy2,
        ZSTD_compressBlock_lazy2_dedicatedDictSearch,
        ZSTD_compressBlock_lazy2_dedicatedDictSearch_row, ZSTD_compressBlock_lazy2_dictMatchState,
        ZSTD_compressBlock_lazy2_dictMatchState_row, ZSTD_compressBlock_lazy2_extDict,
        ZSTD_compressBlock_lazy2_extDict_row, ZSTD_compressBlock_lazy2_row,
        ZSTD_compressBlock_lazy_dedicatedDictSearch,
        ZSTD_compressBlock_lazy_dedicatedDictSearch_row, ZSTD_compressBlock_lazy_dictMatchState,
        ZSTD_compressBlock_lazy_dictMatchState_row, ZSTD_compressBlock_lazy_extDict,
        ZSTD_compressBlock_lazy_extDict_row, ZSTD_compressBlock_lazy_row,
    };
    use crate::compress::zstd_opt::{
        ZSTD_compressBlock_btopt, ZSTD_compressBlock_btopt_dictMatchState,
        ZSTD_compressBlock_btopt_extDict, ZSTD_compressBlock_btultra, ZSTD_compressBlock_btultra2,
        ZSTD_compressBlock_btultra_dictMatchState, ZSTD_compressBlock_btultra_extDict,
    };

    let useRow = ZSTD_rowMatchFinderUsed(strat, useRowMatchFinder);
    // Strategy 0 defaults to "fast" (upstream table slot 0).
    let effective_strat = if strat == 0 { ZSTD_fast } else { strat };

    // Row-hash matchers only exist for greedy/lazy/lazy2; other
    // strategies are unaffected by `useRowMatchFinder`.
    if useRow
        && (effective_strat == ZSTD_greedy
            || effective_strat == ZSTD_lazy
            || effective_strat == ZSTD_lazy2)
    {
        return match dictMode {
            ZSTD_dictMode_e::ZSTD_noDict => match effective_strat {
                s if s == ZSTD_greedy => ZSTD_compressBlock_greedy_row,
                s if s == ZSTD_lazy => ZSTD_compressBlock_lazy_row,
                _ => ZSTD_compressBlock_lazy2_row,
            },
            ZSTD_dictMode_e::ZSTD_extDict => match effective_strat {
                s if s == ZSTD_greedy => ZSTD_compressBlock_greedy_extDict_row,
                s if s == ZSTD_lazy => ZSTD_compressBlock_lazy_extDict_row,
                _ => ZSTD_compressBlock_lazy2_extDict_row,
            },
            ZSTD_dictMode_e::ZSTD_dictMatchState => match effective_strat {
                s if s == ZSTD_greedy => ZSTD_compressBlock_greedy_dictMatchState_row,
                s if s == ZSTD_lazy => ZSTD_compressBlock_lazy_dictMatchState_row,
                _ => ZSTD_compressBlock_lazy2_dictMatchState_row,
            },
            ZSTD_dictMode_e::ZSTD_dedicatedDictSearch => match effective_strat {
                s if s == ZSTD_greedy => ZSTD_compressBlock_greedy_dedicatedDictSearch_row,
                s if s == ZSTD_lazy => ZSTD_compressBlock_lazy_dedicatedDictSearch_row,
                _ => ZSTD_compressBlock_lazy2_dedicatedDictSearch_row,
            },
        };
    }

    match dictMode {
        ZSTD_dictMode_e::ZSTD_noDict => match effective_strat {
            s if s == ZSTD_fast => ZSTD_compressBlock_fast,
            s if s == ZSTD_dfast => ZSTD_compressBlock_doubleFast,
            s if s == ZSTD_greedy => ZSTD_compressBlock_greedy,
            s if s == ZSTD_lazy => ZSTD_compressBlock_lazy,
            s if s == ZSTD_lazy2 => ZSTD_compressBlock_lazy2,
            s if s == ZSTD_btlazy2 => ZSTD_compressBlock_btlazy2,
            s if s == ZSTD_btopt => ZSTD_compressBlock_btopt,
            s if s == ZSTD_btultra => ZSTD_compressBlock_btultra,
            s if s == ZSTD_btultra2 => ZSTD_compressBlock_btultra2,
            _ => panic!(
                "unsupported noDict block compressor strategy {}",
                effective_strat
            ),
        },
        ZSTD_dictMode_e::ZSTD_extDict => match effective_strat {
            s if s == ZSTD_fast => ZSTD_compressBlock_fast_extDict,
            s if s == ZSTD_dfast => ZSTD_compressBlock_doubleFast_extDict,
            s if s == ZSTD_greedy => ZSTD_compressBlock_greedy_extDict,
            s if s == ZSTD_lazy => ZSTD_compressBlock_lazy_extDict,
            s if s == ZSTD_lazy2 => ZSTD_compressBlock_lazy2_extDict,
            s if s == ZSTD_btlazy2 => ZSTD_compressBlock_btlazy2_extDict,
            s if s == ZSTD_btopt => ZSTD_compressBlock_btopt_extDict,
            s if s == ZSTD_btultra => ZSTD_compressBlock_btultra_extDict,
            s if s == ZSTD_btultra2 => ZSTD_compressBlock_btultra_extDict,
            _ => panic!(
                "unsupported extDict block compressor strategy {}",
                effective_strat
            ),
        },
        ZSTD_dictMode_e::ZSTD_dictMatchState => match effective_strat {
            s if s == ZSTD_fast => ZSTD_compressBlock_fast_dictMatchState,
            s if s == ZSTD_dfast => ZSTD_compressBlock_doubleFast_dictMatchState,
            s if s == ZSTD_greedy => ZSTD_compressBlock_greedy_dictMatchState,
            s if s == ZSTD_lazy => ZSTD_compressBlock_lazy_dictMatchState,
            s if s == ZSTD_lazy2 => ZSTD_compressBlock_lazy2_dictMatchState,
            s if s == ZSTD_btlazy2 => ZSTD_compressBlock_btlazy2_dictMatchState,
            s if s == ZSTD_btopt => ZSTD_compressBlock_btopt_dictMatchState,
            s if s == ZSTD_btultra => ZSTD_compressBlock_btultra_dictMatchState,
            s if s == ZSTD_btultra2 => ZSTD_compressBlock_btultra_dictMatchState,
            _ => panic!(
                "unsupported dictMatchState block compressor strategy {}",
                effective_strat
            ),
        },
        ZSTD_dictMode_e::ZSTD_dedicatedDictSearch => match effective_strat {
            s if s == ZSTD_greedy => ZSTD_compressBlock_greedy_dedicatedDictSearch,
            s if s == ZSTD_lazy => ZSTD_compressBlock_lazy_dedicatedDictSearch,
            s if s == ZSTD_lazy2 => ZSTD_compressBlock_lazy2_dedicatedDictSearch,
            _ => panic!(
                "unsupported dedicatedDictSearch block compressor strategy {}",
                effective_strat
            ),
        },
    }
}

/// Upstream `MIN_SEQUENCES_BLOCK_SPLITTING` (`zstd_compress.c:4204`).
/// Minimum seq count before the block splitter considers bisecting.
pub const MIN_SEQUENCES_BLOCK_SPLITTING: usize = 300;

/// Upstream `ZSTD_MAX_NB_BLOCK_SPLITS` (`zstd_compress_internal.h:460`).
/// Absolute cap on how many splits the block splitter can produce
/// within a single block.
pub const ZSTD_MAX_NB_BLOCK_SPLITS: usize = 196;

/// Port of `ZSTD_writeEpilogue` (`zstd_compress.c:5368`). Writes a
/// final empty `bt_raw` block and the optional XXH64 checksum to
/// close a frame. If the compression session never emitted a block
/// (`stage == ZSTDcs_init`), writes the frame header first.
///
/// Rust-port status: uses our CCtx's `stage`, `xxhState`, and
/// `appliedParams` fields. Upstream's `ZSTD_writeFrameHeader` takes
/// the params pointer + pledgedSrcSize + dictID; our port gets those
/// from `cctx.appliedParams.fParams`, `cctx.pledgedSrcSizePlusOne`,
/// and `cctx.dictID` respectively.
pub fn ZSTD_writeEpilogue(cctx: &mut ZSTD_CCtx, dst: &mut [u8]) -> usize {
    use crate::common::error::{ErrorCode, ERROR};
    use crate::common::mem::{MEM_writeLE24, MEM_writeLE32};
    use crate::common::xxhash::XXH64_digest;
    use crate::decompress::zstd_decompress_block::{blockType_e, ZSTD_blockHeaderSize};

    if cctx.stage == ZSTD_compressionStage_e::ZSTDcs_created {
        return ERROR(ErrorCode::StageWrong);
    }

    let mut op = 0usize;

    // Empty-frame path: haven't emitted a block yet.
    if cctx.stage == ZSTD_compressionStage_e::ZSTDcs_init {
        let pledged = ZSTD_getPledgedSrcSize(cctx);
        let fhSize = ZSTD_writeFrameHeader(
            dst,
            &cctx.appliedParams.fParams,
            cctx.appliedParams.cParams.windowLog,
            pledged,
            cctx.dictID,
        );
        if crate::common::error::ERR_isError(fhSize) {
            return fhSize;
        }
        op += fhSize;
        cctx.stage = ZSTD_compressionStage_e::ZSTDcs_ongoing;
    }

    if cctx.stage != ZSTD_compressionStage_e::ZSTDcs_ending {
        if dst.len() - op < ZSTD_blockHeaderSize {
            return ERROR(ErrorCode::DstSizeTooSmall);
        }
        // Last-block marker: lastBlock=1, bt_raw, size=0.
        let cBlockHeader24: u32 = 1u32.wrapping_add((blockType_e::bt_raw as u32) << 1);
        MEM_writeLE24(&mut dst[op..], cBlockHeader24);
        op += ZSTD_blockHeaderSize;
    }

    if cctx.appliedParams.fParams.checksumFlag != 0 {
        if dst.len() - op < 4 {
            return ERROR(ErrorCode::DstSizeTooSmall);
        }
        let checksum = XXH64_digest(&cctx.xxhState) as u32;
        MEM_writeLE32(&mut dst[op..], checksum);
        op += 4;
    }

    cctx.stage = ZSTD_compressionStage_e::ZSTDcs_created;
    op
}

/// Port of `ZSTD_optimalBlockSize` (`zstd_compress.c:4576`). Picks the
/// next block size for the outer compressor loop. Full blocks can be
/// pre-split at a content-shift boundary when savings justify it.
pub fn ZSTD_optimalBlockSize(
    src: &[u8],
    blockSizeMax: usize,
    mut splitLevel: i32,
    strat: u32,
    savings: i64,
) -> usize {
    use crate::compress::zstd_presplit::ZSTD_splitBlock;

    const FULL_BLOCK: usize = 128 << 10;
    // Match upstream's `splitLevels` table from `zstd_compress.c:4579`.
    // (Note: a previous workaround forced `[0; 10]` (fromBorders for all
    // strategies) thinking byChunks corrupted output. Bisection showed
    // the actual bug is a separate L5+ greedy/lazy corruption that
    // triggers on specific content past ~80 MB silesia and fires
    // regardless of splitter choice — see TODO.md.)
    const SPLIT_LEVELS: [i32; 10] = [0, 0, 1, 2, 2, 3, 3, 4, 4, 4];

    if src.len() < FULL_BLOCK || blockSizeMax < FULL_BLOCK {
        return src.len().min(blockSizeMax);
    }
    if savings < 3 {
        return FULL_BLOCK;
    }
    if splitLevel == 1 {
        return FULL_BLOCK;
    }
    if splitLevel == 0 {
        splitLevel = SPLIT_LEVELS[strat as usize];
    } else {
        debug_assert!((2..=6).contains(&splitLevel));
        splitLevel -= 2;
    }
    ZSTD_splitBlock(&src[..FULL_BLOCK], splitLevel)
}

/// Port of `ZSTD_compressContinue_public` (`zstd_compress.c:4877`).
/// Non-deprecated alias of `ZSTD_compressContinue`.
#[inline]
pub fn ZSTD_compressContinue_public(cctx: &mut ZSTD_CCtx, dst: &mut [u8], src: &[u8]) -> usize {
    ZSTD_compressContinue_internal(cctx, dst, src, 1, 0)
}

/// Port of `ZSTD_compressEnd_public` (`zstd_compress.c:5431`). Public
/// entry point that finalizes the last block and writes the frame
/// epilogue. `ZSTD_compressEnd()` is just the legacy wrapper around
/// this symbol, matching upstream.
pub fn ZSTD_compressEnd_public(cctx: &mut ZSTD_CCtx, dst: &mut [u8], src: &[u8]) -> usize {
    let cSize = ZSTD_compressContinue_internal(cctx, dst, src, 1, 1);
    if ERR_isError(cSize) {
        return cSize;
    }
    let endSize = ZSTD_writeEpilogue(cctx, &mut dst[cSize..]);
    if ERR_isError(endSize) {
        return endSize;
    }
    if cctx.pledgedSrcSizePlusOne != 0 && cctx.pledgedSrcSizePlusOne != cctx.consumedSrcSize + 1 {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    ZSTD_CCtx_trace(cctx, endSize);
    cSize + endSize
}

/// Port of `ZSTD_compress_frameChunk` (`zstd_compress.c:4615`).
/// Compresses `src` into one or more framed blocks, consuming all
/// input and setting the last-block bit on the final block when
/// `lastFrameChunk != 0`.
///
/// Current scope covers the normal single-thread block loop used by
/// the existing Rust compressor:
///   - upstream pre-block splitter via `ZSTD_optimalBlockSize`
///   - no targetCBlockSize/superblock mode
///   - optional post-block-splitter mode
fn ZSTD_compressBlock_internal(
    cctx: &mut ZSTD_CCtx,
    dst: &mut [u8],
    src: &[u8],
    frame: bool,
) -> usize {
    use crate::compress::zstd_compress_sequences::FSE_repeat;

    let bss = ZSTD_buildSeqStore(cctx, src);
    if ERR_isError(bss) {
        return bss;
    }
    if bss == ZSTD_BuildSeqStore_e::ZSTDbss_noCompress as usize {
        if cctx.prevEntropy.fse.offcode_repeatMode == FSE_repeat::FSE_repeat_valid {
            cctx.prevEntropy.fse.offcode_repeatMode = FSE_repeat::FSE_repeat_check;
        }
        return 0;
    }

    let disableLiteralCompression = ZSTD_literalsCompressionIsDisabled(
        cctx.appliedParams.literalCompressionMode,
        cctx.appliedParams.cParams.strategy,
        cctx.appliedParams.cParams.targetLength,
    ) as i32;
    let mut cSize = {
        let seqStore = cctx.seqStore.as_mut().unwrap();
        ZSTD_entropyCompressSeqStore_wksp(
            dst,
            seqStore,
            &cctx.prevEntropy,
            &mut cctx.nextEntropy,
            cctx.appliedParams.cParams.strategy,
            disableLiteralCompression,
            src.len(),
            cctx.bmi2,
            &mut cctx.entropyScratch,
        )
    };
    if ERR_isError(cSize) {
        return cSize;
    }

    if frame && cctx.isFirstBlock == 0 && cSize < 25 && ZSTD_isRLE(src) != 0 {
        cSize = 1;
        if let Some(first) = dst.first_mut() {
            *first = src[0];
        } else {
            return ERROR(ErrorCode::DstSizeTooSmall);
        }
    }

    if !ERR_isError(cSize) && cSize > 1 {
        ZSTD_blockState_confirmRepcodesAndEntropyTables(cctx);
    }
    if cctx.prevEntropy.fse.offcode_repeatMode == FSE_repeat::FSE_repeat_valid {
        cctx.prevEntropy.fse.offcode_repeatMode = FSE_repeat::FSE_repeat_check;
    }
    cSize
}

/// Port of `ZSTD_compress_frameChunk` (`zstd_compress.c:4615`).
/// Compresses a chunk of data into one or multiple blocks. All blocks are
/// terminated and all input is consumed. The frame header must already
/// have been emitted by the caller. Errors if `dstCapacity` is too small
/// to hold the compressed output.
pub fn ZSTD_compress_frameChunk(
    cctx: &mut ZSTD_CCtx,
    dst: &mut [u8],
    src: &[u8],
    lastFrameChunk: u32,
) -> usize {
    use crate::common::xxhash::XXH64_update;
    use crate::compress::match_state::{
        ZSTD_matchState_checkDictValidity, ZSTD_matchState_enforceMaxDist,
        ZSTD_overflowCorrectIfNeeded,
    };

    cctx.ms.get_or_insert_with(|| {
        crate::compress::match_state::ZSTD_MatchState_t::new(cctx.appliedParams.cParams)
    });
    cctx.seqStore.get_or_insert_with(|| {
        SeqStore_t::with_capacity(ZSTD_BLOCKSIZE_MAX / 3, ZSTD_BLOCKSIZE_MAX)
    });

    let mut remaining = src.len();
    let mut ip = 0usize;
    let mut op = 0usize;
    let blockSizeMax = if cctx.blockSizeMax != 0 {
        cctx.blockSizeMax
    } else if cctx.appliedParams.maxBlockSize != 0 {
        cctx.appliedParams.maxBlockSize.min(ZSTD_BLOCKSIZE_MAX)
    } else {
        (1usize << cctx.appliedParams.cParams.windowLog).min(ZSTD_BLOCKSIZE_MAX)
    };
    let maxDist: u32 = 1u32 << cctx.appliedParams.cParams.windowLog;
    let mut savings = cctx.consumedSrcSize as i64 - cctx.producedCSize as i64;
    let dict_prefix = if cctx.consumedSrcSize == 0 {
        cctx.ms
            .as_ref()
            .and_then(|ms| (!ms.dictContent.is_empty()).then(|| ms.dictContent.clone()))
    } else {
        None
    };
    let history_prefix_len = dict_prefix.as_ref().map_or(0, Vec::len);
    let history_window = dict_prefix.map(|dict| {
        let mut window = Vec::with_capacity(dict.len() + src.len());
        window.extend_from_slice(&dict);
        window.extend_from_slice(src);
        window
    });
    let window_buf = history_window.as_deref().unwrap_or(src);

    if cctx.appliedParams.fParams.checksumFlag != 0 && !src.is_empty() {
        XXH64_update(&mut cctx.xxhState, src);
    }

    while remaining != 0 {
        let blockSize = ZSTD_optimalBlockSize(
            &src[ip..ip + remaining],
            blockSizeMax,
            cctx.appliedParams.preBlockSplitter_level,
            cctx.appliedParams.cParams.strategy,
            savings,
        );
        if blockSize == 0 {
            return ERROR(ErrorCode::Generic);
        }
        let lastBlock = lastFrameChunk & ((blockSize == remaining) as u32);
        // blockStartAbs is the absolute index of THIS block's start.
        // Earlier we read `window.nextSrc`, but `nextSrc` was advanced
        // to `src_abs + src.len()` (the end of the whole input) by the
        // `ZSTD_window_update` call at the top of `compressContinue_internal`,
        // so it points at end-of-input, not block start. Using that
        // bogus value made `ZSTD_overflowCorrectIfNeeded` and
        // `ZSTD_window_enforceMaxDist` think the window had already
        // exceeded `maxDist` on the very first block, shoving
        // `lowLimit` (and therefore `nextToUpdate`) far past the
        // current block, producing garbage matches and bloated output.
        let blockStartAbs = cctx
            .ms
            .as_ref()
            .unwrap()
            .window
            .base_offset
            .wrapping_add(ip as u32);
        let blockEndAbs = blockStartAbs.wrapping_add(blockSize as u32);

        if dst.len() - op < ZSTD_blockHeaderSize + MIN_CBLOCK_SIZE + 1 {
            return ERROR(ErrorCode::DstSizeTooSmall);
        }

        {
            let ms = cctx.ms.as_mut().unwrap();
            ZSTD_overflowCorrectIfNeeded(
                ms,
                cctx.appliedParams.useRowMatchFinder,
                0,
                cctx.appliedParams.cParams.windowLog,
                cctx.appliedParams.cParams.chainLog,
                cctx.appliedParams.cParams.strategy,
                blockStartAbs,
                blockEndAbs,
            );
            ZSTD_matchState_checkDictValidity(ms, blockEndAbs, maxDist);
            ZSTD_matchState_enforceMaxDist(ms, blockStartAbs, maxDist);
            if ms.nextToUpdate < ms.window.lowLimit {
                ms.nextToUpdate = ms.window.lowLimit;
            }
        }

        let cSize = if ZSTD_useTargetCBlockSize(&cctx.appliedParams) {
            let bss = ZSTD_buildSeqStore_with_window(
                cctx,
                window_buf,
                history_prefix_len + ip,
                history_prefix_len + ip + blockSize,
            );
            if ERR_isError(bss) {
                return bss;
            }

            if bss == ZSTD_BuildSeqStore_e::ZSTDbss_compress as usize
                && cctx.isFirstBlock == 0
                && ZSTD_maybeRLE(cctx.seqStore.as_ref().unwrap())
                && ZSTD_isRLE(&src[ip..ip + blockSize]) != 0
            {
                ZSTD_rleCompressBlock(&mut dst[op..], src[ip], blockSize, lastBlock)
            } else {
                let cSize = if bss == ZSTD_BuildSeqStore_e::ZSTDbss_compress as usize {
                    crate::compress::zstd_compress_superblock::ZSTD_compressSuperBlock(
                        cctx,
                        &mut dst[op..],
                        &src[ip..ip + blockSize],
                        lastBlock,
                    )
                } else {
                    0
                };

                if cSize != ERROR(ErrorCode::DstSizeTooSmall) {
                    let maxCSize = blockSize.saturating_sub(ZSTD_minGain(
                        blockSize,
                        cctx.appliedParams.cParams.strategy,
                    ));
                    if ERR_isError(cSize) {
                        return cSize;
                    }
                    if cSize != 0 && cSize < maxCSize + ZSTD_blockHeaderSize {
                        ZSTD_blockState_confirmRepcodesAndEntropyTables(cctx);
                        cSize
                    } else {
                        ZSTD_noCompressBlock(&mut dst[op..], &src[ip..ip + blockSize], lastBlock)
                    }
                } else {
                    ZSTD_noCompressBlock(&mut dst[op..], &src[ip..ip + blockSize], lastBlock)
                }
            }
        } else if ZSTD_blockSplitterEnabled(&cctx.appliedParams) {
            ZSTD_compressBlock_splitBlock_with_window(
                cctx,
                &mut dst[op..],
                window_buf,
                history_prefix_len + ip,
                history_prefix_len + ip + blockSize,
                lastBlock,
            )
        } else {
            let cBodySize = ZSTD_compressBlock_internal_with_window(
                cctx,
                &mut dst[op + ZSTD_blockHeaderSize..],
                window_buf,
                history_prefix_len + ip,
                history_prefix_len + ip + blockSize,
                true,
            );
            if ERR_isError(cBodySize) {
                cBodySize
            } else if cBodySize == 0 {
                ZSTD_noCompressBlock(&mut dst[op..], &src[ip..ip + blockSize], lastBlock)
            } else if cBodySize == 1 {
                let header = lastBlock
                    .wrapping_add(
                        (crate::decompress::zstd_decompress_block::blockType_e::bt_rle as u32) << 1,
                    )
                    .wrapping_add((blockSize as u32) << 3);
                MEM_writeLE24(&mut dst[op..], header);
                ZSTD_blockHeaderSize + 1
            } else {
                let header = lastBlock
                    .wrapping_add(
                        (crate::decompress::zstd_decompress_block::blockType_e::bt_compressed
                            as u32)
                            << 1,
                    )
                    .wrapping_add((cBodySize as u32) << 3);
                MEM_writeLE24(&mut dst[op..], header);
                ZSTD_blockHeaderSize + cBodySize
            }
        };
        if ERR_isError(cSize) {
            return cSize;
        }

        savings += blockSize as i64 - cSize as i64;
        remaining -= blockSize;
        ip += blockSize;
        op += cSize;
        cctx.isFirstBlock = 0;
    }

    if lastFrameChunk != 0 && op != 0 {
        cctx.stage = ZSTD_compressionStage_e::ZSTDcs_ending;
    }
    op
}

fn ZSTD_compress_frameChunk_windowed(
    cctx: &mut ZSTD_CCtx,
    dst: &mut [u8],
    window_buf: &[u8],
    src_pos: usize,
    src_len: usize,
    lastFrameChunk: u32,
) -> usize {
    use crate::common::xxhash::XXH64_update;
    use crate::compress::match_state::{
        ZSTD_matchState_checkDictValidity, ZSTD_matchState_enforceMaxDist,
        ZSTD_overflowCorrectIfNeeded,
    };

    if src_pos > window_buf.len() || src_len > window_buf.len().saturating_sub(src_pos) {
        return ERROR(ErrorCode::SrcSizeWrong);
    }

    cctx.ms.get_or_insert_with(|| {
        crate::compress::match_state::ZSTD_MatchState_t::new(cctx.appliedParams.cParams)
    });
    cctx.seqStore.get_or_insert_with(|| {
        SeqStore_t::with_capacity(ZSTD_BLOCKSIZE_MAX / 3, ZSTD_BLOCKSIZE_MAX)
    });

    let src = &window_buf[src_pos..src_pos + src_len];
    let mut remaining = src_len;
    let mut ip = src_pos;
    let mut op = 0usize;
    let blockSizeMax = if cctx.blockSizeMax != 0 {
        cctx.blockSizeMax
    } else if cctx.appliedParams.maxBlockSize != 0 {
        cctx.appliedParams.maxBlockSize.min(ZSTD_BLOCKSIZE_MAX)
    } else {
        (1usize << cctx.appliedParams.cParams.windowLog).min(ZSTD_BLOCKSIZE_MAX)
    };
    let maxDist: u32 = 1u32 << cctx.appliedParams.cParams.windowLog;
    let mut savings = cctx.consumedSrcSize as i64 - cctx.producedCSize as i64;

    if cctx.appliedParams.fParams.checksumFlag != 0 && !src.is_empty() {
        XXH64_update(&mut cctx.xxhState, src);
    }

    while remaining != 0 {
        let blockSize = ZSTD_optimalBlockSize(
            &window_buf[ip..ip + remaining],
            blockSizeMax,
            cctx.appliedParams.preBlockSplitter_level,
            cctx.appliedParams.cParams.strategy,
            savings,
        );
        if blockSize == 0 {
            return ERROR(ErrorCode::Generic);
        }
        let lastBlock = lastFrameChunk & ((blockSize == remaining) as u32);
        let blockStartAbs = cctx
            .ms
            .as_ref()
            .unwrap()
            .window
            .base_offset
            .wrapping_add(ip as u32);
        let blockEndAbs = blockStartAbs.wrapping_add(blockSize as u32);

        if dst.len() - op < ZSTD_blockHeaderSize + MIN_CBLOCK_SIZE + 1 {
            return ERROR(ErrorCode::DstSizeTooSmall);
        }

        {
            let ms = cctx.ms.as_mut().unwrap();
            ZSTD_overflowCorrectIfNeeded(
                ms,
                cctx.appliedParams.useRowMatchFinder,
                0,
                cctx.appliedParams.cParams.windowLog,
                cctx.appliedParams.cParams.chainLog,
                cctx.appliedParams.cParams.strategy,
                blockStartAbs,
                blockEndAbs,
            );
            ZSTD_matchState_checkDictValidity(ms, blockEndAbs, maxDist);
            ZSTD_matchState_enforceMaxDist(ms, blockStartAbs, maxDist);
            if ms.nextToUpdate < ms.window.lowLimit {
                ms.nextToUpdate = ms.window.lowLimit;
            }
        }

        let cSize = if ZSTD_useTargetCBlockSize(&cctx.appliedParams) {
            let bss = ZSTD_buildSeqStore_with_window(cctx, window_buf, ip, ip + blockSize);
            if ERR_isError(bss) {
                return bss;
            }

            if bss == ZSTD_BuildSeqStore_e::ZSTDbss_compress as usize
                && cctx.isFirstBlock == 0
                && ZSTD_maybeRLE(cctx.seqStore.as_ref().unwrap())
                && ZSTD_isRLE(&window_buf[ip..ip + blockSize]) != 0
            {
                ZSTD_rleCompressBlock(&mut dst[op..], window_buf[ip], blockSize, lastBlock)
            } else {
                let cSize = if bss == ZSTD_BuildSeqStore_e::ZSTDbss_compress as usize {
                    crate::compress::zstd_compress_superblock::ZSTD_compressSuperBlock(
                        cctx,
                        &mut dst[op..],
                        &window_buf[ip..ip + blockSize],
                        lastBlock,
                    )
                } else {
                    0
                };

                if cSize != ERROR(ErrorCode::DstSizeTooSmall) {
                    let maxCSize = blockSize.saturating_sub(ZSTD_minGain(
                        blockSize,
                        cctx.appliedParams.cParams.strategy,
                    ));
                    if ERR_isError(cSize) {
                        return cSize;
                    }
                    if cSize != 0 && cSize < maxCSize + ZSTD_blockHeaderSize {
                        ZSTD_blockState_confirmRepcodesAndEntropyTables(cctx);
                        cSize
                    } else {
                        ZSTD_noCompressBlock(
                            &mut dst[op..],
                            &window_buf[ip..ip + blockSize],
                            lastBlock,
                        )
                    }
                } else {
                    ZSTD_noCompressBlock(&mut dst[op..], &window_buf[ip..ip + blockSize], lastBlock)
                }
            }
        } else if ZSTD_blockSplitterEnabled(&cctx.appliedParams) {
            ZSTD_compressBlock_splitBlock_with_window(
                cctx,
                &mut dst[op..],
                window_buf,
                ip,
                ip + blockSize,
                lastBlock,
            )
        } else {
            let cBodySize = ZSTD_compressBlock_internal_with_window(
                cctx,
                &mut dst[op + ZSTD_blockHeaderSize..],
                window_buf,
                ip,
                ip + blockSize,
                true,
            );
            if ERR_isError(cBodySize) {
                cBodySize
            } else if cBodySize == 0 {
                ZSTD_noCompressBlock(&mut dst[op..], &window_buf[ip..ip + blockSize], lastBlock)
            } else if cBodySize == 1 {
                let header = lastBlock
                    .wrapping_add(
                        (crate::decompress::zstd_decompress_block::blockType_e::bt_rle as u32) << 1,
                    )
                    .wrapping_add((blockSize as u32) << 3);
                MEM_writeLE24(&mut dst[op..], header);
                ZSTD_blockHeaderSize + 1
            } else {
                let header = lastBlock
                    .wrapping_add(
                        (crate::decompress::zstd_decompress_block::blockType_e::bt_compressed
                            as u32)
                            << 1,
                    )
                    .wrapping_add((cBodySize as u32) << 3);
                MEM_writeLE24(&mut dst[op..], header);
                ZSTD_blockHeaderSize + cBodySize
            }
        };
        if ERR_isError(cSize) {
            return cSize;
        }

        savings += blockSize as i64 - cSize as i64;
        remaining -= blockSize;
        ip += blockSize;
        op += cSize;
        cctx.isFirstBlock = 0;
    }

    if lastFrameChunk != 0 && op != 0 {
        cctx.stage = ZSTD_compressionStage_e::ZSTDcs_ending;
    }
    op
}

/// Port of `ZSTD_compressContinue_internal` (`zstd_compress.c:4816`).
/// Writes a frame header on the first framed call, updates window
/// bookkeeping, then dispatches to `ZSTD_compress_frameChunk`.
///
/// Current scope supports both the normal frame path and the legacy
/// headerless block path through the already-ported block-body
/// compressor.
pub fn ZSTD_compressContinue_internal(
    cctx: &mut ZSTD_CCtx,
    dst: &mut [u8],
    src: &[u8],
    frame: u32,
    lastFrameChunk: u32,
) -> usize {
    use crate::compress::match_state::{ZSTD_overflowCorrectIfNeeded, ZSTD_window_update};

    if cctx.stage == ZSTD_compressionStage_e::ZSTDcs_created {
        return ERROR(ErrorCode::StageWrong);
    }

    let mut fhSize = 0usize;
    let mut out = dst;
    if frame != 0 && cctx.stage == ZSTD_compressionStage_e::ZSTDcs_init {
        let pledged = ZSTD_getPledgedSrcSize(cctx);
        fhSize = if cctx.appliedParams.format
            == crate::decompress::zstd_decompress::ZSTD_format_e::ZSTD_f_zstd1
        {
            ZSTD_writeFrameHeader(
                out,
                &cctx.appliedParams.fParams,
                cctx.appliedParams.cParams.windowLog,
                pledged,
                cctx.dictID,
            )
        } else {
            ZSTD_writeFrameHeader_advanced(
                out,
                &cctx.appliedParams.fParams,
                cctx.appliedParams.cParams.windowLog,
                pledged,
                cctx.dictID,
                cctx.appliedParams.format,
            )
        };
        if ERR_isError(fhSize) {
            return fhSize;
        }
        out = &mut out[fhSize..];
        cctx.stage = ZSTD_compressionStage_e::ZSTDcs_ongoing;
    }

    if src.is_empty() {
        return fhSize;
    }

    if cctx.ms.is_none() {
        cctx.ms = Some(crate::compress::match_state::ZSTD_MatchState_t::new(
            cctx.appliedParams.cParams,
        ));
    }
    let rc = ZSTD_initLocalDict(cctx);
    if ERR_isError(rc) {
        return rc;
    }
    let ms = cctx.ms.as_mut().unwrap();
    let srcAbs = ms.window.nextSrc;
    if !ZSTD_window_update(&mut ms.window, srcAbs, src.len(), ms.forceNonContiguous) {
        ms.forceNonContiguous = false;
        ms.nextToUpdate = ms.window.dictLimit;
    }
    if cctx.appliedParams.ldmEnable == crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_enable
    {
        if let Some(ldmState) = cctx.ldmState.as_mut() {
            ZSTD_window_update(&mut ldmState.window, srcAbs, src.len(), false);
        }
    }

    if frame == 0 {
        let srcEndAbs = srcAbs.wrapping_add(src.len() as u32);
        ZSTD_overflowCorrectIfNeeded(
            ms,
            cctx.appliedParams.useRowMatchFinder,
            0,
            cctx.appliedParams.cParams.windowLog,
            cctx.appliedParams.cParams.chainLog,
            cctx.appliedParams.cParams.strategy,
            srcAbs,
            srcEndAbs,
        );
        let cSize = ZSTD_compressBlock_internal(cctx, out, src, false);
        if ERR_isError(cSize) {
            return cSize;
        }
        cctx.stage = ZSTD_compressionStage_e::ZSTDcs_ongoing;
        cctx.consumedSrcSize += src.len() as u64;
        cctx.producedCSize += cSize as u64;
        return cSize;
    }

    let cSize = ZSTD_compress_frameChunk(cctx, out, src, lastFrameChunk);
    if ERR_isError(cSize) {
        return cSize;
    }
    cctx.consumedSrcSize += src.len() as u64;
    cctx.producedCSize += (cSize + fhSize) as u64;
    if cctx.pledgedSrcSizePlusOne != 0 && cctx.consumedSrcSize + 1 > cctx.pledgedSrcSizePlusOne {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    cSize + fhSize
}

/// Port of `ZSTD_compressBlock_deprecated` (`zstd_compress.c:4907`).
/// Legacy single-block compressor that emits a headerless block body.
pub fn ZSTD_compressBlock_deprecated(cctx: &mut ZSTD_CCtx, dst: &mut [u8], src: &[u8]) -> usize {
    let blockSizeMax = ZSTD_getBlockSize_deprecated(cctx);
    if src.len() > blockSizeMax {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    ZSTD_compressContinue_internal(cctx, dst, src, 0, 0)
}

/// Port of `ZSTD_compressBlock` (`zstd_compress.c:4917`).
/// Public alias of `ZSTD_compressBlock_deprecated`.
#[inline]
pub fn ZSTD_compressBlock(cctx: &mut ZSTD_CCtx, dst: &mut [u8], src: &[u8]) -> usize {
    ZSTD_compressBlock_deprecated(cctx, dst, src)
}

/// Port of `ZSTD_getBlockSize_deprecated` (`zstd_compress.c:4893`).
#[inline]
pub fn ZSTD_getBlockSize_deprecated(cctx: &ZSTD_CCtx) -> usize {
    cctx.appliedParams.maxBlockSize.min(
        1usize
            .checked_shl(cctx.appliedParams.cParams.windowLog)
            .unwrap_or(usize::MAX),
    )
}

/// Port of `ZSTD_compressBegin_usingCDict`. Legacy begin/end
/// session initializer that wires a pre-built CDict.
pub fn ZSTD_compressBegin_usingCDict(cctx: &mut ZSTD_CCtx, cdict: &ZSTD_CDict) -> usize {
    ZSTD_compressBegin_usingCDict_internal(
        cctx,
        cdict,
        cctx.requestedParams.fParams,
        crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN,
    )
}

/// Port of `ZSTD_compressBegin_usingCDict_internal`
/// (`zstd_compress.c:5847`). Full variant: picks cParams based on the
/// CDict + pledgedSrcSize heuristic (attach-mode vs derive-from-level),
/// composes `ZSTD_CCtx_params`, and routes through
/// `ZSTD_compressBegin_internal` with the cdict.
pub fn ZSTD_compressBegin_usingCDict_internal(
    cctx: &mut ZSTD_CCtx,
    cdict: &ZSTD_CDict,
    fParams: ZSTD_FrameParameters,
    pledgedSrcSize: u64,
) -> usize {
    use crate::decompress::zstd_ddict::ZSTD_dictContentType_e;
    use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;
    // Upstream: pick between CDict's cParams verbatim (attach mode)
    // and `ZSTD_getCParams(cdict->compressionLevel, pledged, dictSize)`
    // (re-derive). Threshold: pledged < 128KB, pledged < dictSize*6,
    // pledged == UNKNOWN, or cdict.compressionLevel == 0.
    let dictSize = cdict.dictContent.len();
    let use_cdict_params = pledgedSrcSize < ZSTD_USE_CDICT_PARAMS_SRCSIZE_CUTOFF
        || pledgedSrcSize < dictSize as u64 * ZSTD_USE_CDICT_PARAMS_DICTSIZE_MULTIPLIER
        || pledgedSrcSize == ZSTD_CONTENTSIZE_UNKNOWN
        || cdict.compressionLevel == 0;

    let cParams = if use_cdict_params {
        ZSTD_getCParamsFromCDict(cdict)
    } else {
        ZSTD_getCParams(cdict.compressionLevel, pledgedSrcSize, dictSize)
    };

    let params = ZSTD_parameters { cParams, fParams };
    let mut cctxParams = ZSTD_CCtx_params::default();
    ZSTD_CCtxParams_init_internal(&mut cctxParams, &params, cdict.compressionLevel);
    cctxParams.format = cctx.format;
    ZSTD_compressBegin_internal(
        cctx,
        &[],
        ZSTD_dictContentType_e::ZSTD_dct_auto,
        Some(cdict),
        &cctxParams,
        pledgedSrcSize,
        ZSTD_buffered_policy_e::ZSTDb_not_buffered,
    )
}

/// Port of `ZSTD_compressBegin_usingCDict_advanced`. Legacy
/// begin/end-style initializer that wires a pre-built CDict with
/// explicit `fParams` + `pledgedSrcSize`. Upstream
/// (zstd_compress.c:5888) routes through
/// `ZSTD_compressBegin_usingCDict_internal`.
pub fn ZSTD_compressBegin_usingCDict_advanced(
    cctx: &mut ZSTD_CCtx,
    cdict: &ZSTD_CDict,
    fParams: ZSTD_FrameParameters,
    pledgedSrcSize: u64,
) -> usize {
    ZSTD_compressBegin_usingCDict_internal(cctx, cdict, fParams, pledgedSrcSize)
}

/// Port of `ZSTD_compressBegin_advanced`. Legacy begin/end-style
/// initializer that takes an explicit `ZSTD_parameters` bundle + a
/// raw-content dict + pledged src size. Upstream
/// (zstd_compress.c:5329) validates cParams, then initializes the
/// session.
pub fn ZSTD_compressBegin_advanced(
    cctx: &mut ZSTD_CCtx,
    dict: &[u8],
    params: ZSTD_parameters,
    pledgedSrcSize: u64,
) -> usize {
    let mut cctxParams = ZSTD_CCtx_params::default();
    ZSTD_CCtxParams_init_internal(&mut cctxParams, &params, ZSTD_NO_CLEVEL);
    ZSTD_compressBegin_advanced_internal(
        cctx,
        dict,
        crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_auto,
        None,
        &cctxParams,
        pledgedSrcSize,
    )
}

/// Port of `ZSTD_loadCEntropy` (`zstd_compress.c:5085`). Parses the
/// entropy-tables section of a zstd-format dictionary into the
/// caller's entropy tables + `rep[]`. Starts reading at `dict[8..]`
/// (skipping magic + dictID, which the caller has already validated).
///
/// Layout: HUF CTable → FSE OF table → FSE ML table → FSE LL table
/// → 3 × 4-byte rep values. Returns total bytes consumed, or a
/// `DictionaryCorrupted` error on any parse failure.
///
/// Sets `repeatMode` for each table based on `ZSTD_dictNCountRepeat` —
/// tables with every symbol present become `FSE_repeat_valid`;
/// tables with zero-weights become `FSE_repeat_check`. HUF similarly
/// becomes `HUF_repeat_valid` only if no zero-weight symbols AND the
/// full 255-symbol alphabet is present.
pub fn ZSTD_loadCEntropy(
    entropy: &mut ZSTD_entropyCTables_t,
    rep: &mut [u32; 3],
    dict: &[u8],
) -> usize {
    use crate::common::error::{ERR_isError, ErrorCode, ERROR};
    use crate::common::mem::MEM_readLE32;
    use crate::compress::fse_compress::FSE_buildCTable_wksp;
    use crate::compress::huf_compress::{HUF_readCTable, HUF_WORKSPACE_SIZE};
    use crate::compress::zstd_compress_literals::HUF_repeat;
    use crate::compress::zstd_compress_sequences::{FSE_repeat, ZSTD_dictNCountRepeat};
    use crate::decompress::zstd_decompress_block::{
        LLFSELog, MLFSELog, MaxLL, MaxML, MaxOff, OffFSELog,
    };

    if dict.len() < 8 {
        return ERROR(ErrorCode::DictionaryCorrupted);
    }
    let mut pos = 8usize; // skip magic + dictID
    let mut workspace = vec![0u8; HUF_WORKSPACE_SIZE];

    // --- HUF literals CTable ---
    entropy.huf.repeatMode = HUF_repeat::HUF_repeat_check;
    let mut maxSymbolValue: u32 = 255;
    let mut hasZeroWeights: u32 = 1;
    let hufHeaderSize = HUF_readCTable(
        &mut entropy.huf.CTable,
        &mut maxSymbolValue,
        &dict[pos..],
        &mut hasZeroWeights,
    );
    if ERR_isError(hufHeaderSize) {
        return ERROR(ErrorCode::DictionaryCorrupted);
    }
    if hasZeroWeights == 0 && maxSymbolValue == 255 {
        entropy.huf.repeatMode = HUF_repeat::HUF_repeat_valid;
    }
    pos += hufHeaderSize;

    // --- FSE offset table ---
    let mut offcodeNCount = [0i16; (MaxOff + 1) as usize];
    let mut offcodeMaxValue: u32 = MaxOff;
    let mut offcodeLog: u32 = 0;
    let offcodeHeaderSize = crate::common::entropy_common::FSE_readNCount(
        &mut offcodeNCount,
        &mut offcodeMaxValue,
        &mut offcodeLog,
        &dict[pos..],
    );
    if ERR_isError(offcodeHeaderSize) || offcodeLog > OffFSELog {
        return ERROR(ErrorCode::DictionaryCorrupted);
    }
    let rc = FSE_buildCTable_wksp(
        &mut entropy.fse.offcodeCTable,
        &offcodeNCount,
        MaxOff,
        offcodeLog,
        &mut workspace,
    );
    if ERR_isError(rc) {
        return ERROR(ErrorCode::DictionaryCorrupted);
    }
    pos += offcodeHeaderSize;

    // --- FSE match-length table ---
    let mut mlNCount = [0i16; (MaxML + 1) as usize];
    let mut mlMaxValue: u32 = MaxML;
    let mut mlLog: u32 = 0;
    let mlHeaderSize = crate::common::entropy_common::FSE_readNCount(
        &mut mlNCount,
        &mut mlMaxValue,
        &mut mlLog,
        &dict[pos..],
    );
    if ERR_isError(mlHeaderSize) || mlLog > MLFSELog {
        return ERROR(ErrorCode::DictionaryCorrupted);
    }
    let rc = FSE_buildCTable_wksp(
        &mut entropy.fse.matchlengthCTable,
        &mlNCount,
        mlMaxValue,
        mlLog,
        &mut workspace,
    );
    if ERR_isError(rc) {
        return ERROR(ErrorCode::DictionaryCorrupted);
    }
    entropy.fse.matchlength_repeatMode = ZSTD_dictNCountRepeat(&mlNCount, mlMaxValue, MaxML);
    pos += mlHeaderSize;

    // --- FSE literal-length table ---
    let mut llNCount = [0i16; (MaxLL + 1) as usize];
    let mut llMaxValue: u32 = MaxLL;
    let mut llLog: u32 = 0;
    let llHeaderSize = crate::common::entropy_common::FSE_readNCount(
        &mut llNCount,
        &mut llMaxValue,
        &mut llLog,
        &dict[pos..],
    );
    if ERR_isError(llHeaderSize) || llLog > LLFSELog {
        return ERROR(ErrorCode::DictionaryCorrupted);
    }
    let rc = FSE_buildCTable_wksp(
        &mut entropy.fse.litlengthCTable,
        &llNCount,
        llMaxValue,
        llLog,
        &mut workspace,
    );
    if ERR_isError(rc) {
        return ERROR(ErrorCode::DictionaryCorrupted);
    }
    entropy.fse.litlength_repeatMode = ZSTD_dictNCountRepeat(&llNCount, llMaxValue, MaxLL);
    pos += llHeaderSize;

    // --- 3 × 4-byte rep values ---
    if pos + 12 > dict.len() {
        return ERROR(ErrorCode::DictionaryCorrupted);
    }
    rep[0] = MEM_readLE32(&dict[pos..pos + 4]);
    rep[1] = MEM_readLE32(&dict[pos + 4..pos + 8]);
    rep[2] = MEM_readLE32(&dict[pos + 8..pos + 12]);
    pos += 12;

    // --- Validate ---
    let dictContentSize = dict.len() - pos;
    let offcodeMax = if dictContentSize as u64 <= u32::MAX as u64 - 128 * 1024 {
        let maxOffset = (dictContentSize as u32).wrapping_add(128 * 1024);
        crate::common::bits::ZSTD_highbit32(maxOffset)
    } else {
        MaxOff
    };
    entropy.fse.offcode_repeatMode =
        ZSTD_dictNCountRepeat(&offcodeNCount, offcodeMaxValue, offcodeMax.min(MaxOff));

    // All repcodes must be nonzero and within dict content.
    for r in rep.iter() {
        if *r == 0 || *r as usize > dictContentSize {
            return ERROR(ErrorCode::DictionaryCorrupted);
        }
    }

    let _ = FSE_repeat::FSE_repeat_valid; // type anchor
    let _ = ErrorCode::DictionaryCorrupted;
    pos
}

/// Port of `ZSTD_compress_insertDictionary` (`zstd_compress.c:5217`).
/// Routes between raw-content and zstd-format dictionary paths,
/// seeding the live match state for raw/full dictionaries and
/// resetting the compressed-block entropy state up front.
pub fn ZSTD_compress_insertDictionary(
    cctx: &mut ZSTD_CCtx,
    params: &ZSTD_CCtx_params,
    dict: &[u8],
    dictContentType: crate::decompress::zstd_ddict::ZSTD_dictContentType_e,
) -> usize {
    use crate::common::error::{ERR_isError, ErrorCode, ERROR};
    use crate::common::mem::MEM_readLE32;
    use crate::compress::zstd_fast::{ZSTD_dictTableLoadMethod_e, ZSTD_tableFillPurpose_e};
    use crate::decompress::zstd_ddict::ZSTD_dictContentType_e;
    use crate::decompress::zstd_decompress::ZSTD_MAGICNUMBER_DICTIONARY;

    if dict.is_empty() || dict.len() < 8 {
        if dictContentType == ZSTD_dictContentType_e::ZSTD_dct_fullDict {
            return ERROR(ErrorCode::DictionaryWrong);
        }
        // Short or absent dict: no-op.
        return 0;
    }

    ZSTD_reset_compressedBlockState(&mut cctx.prev_rep, &mut cctx.prevEntropy);

    if dictContentType == ZSTD_dictContentType_e::ZSTD_dct_rawContent {
        let (ms_opt, ldm_opt) = (&mut cctx.ms, &mut cctx.ldmState);
        if let Some(ms) = ms_opt.as_mut() {
            let dictID = ZSTD_loadDictionaryContent(
                ms,
                ldm_opt.as_mut(),
                params,
                dict,
                ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_fast,
                ZSTD_tableFillPurpose_e::ZSTD_tfp_forCCtx,
            );
            if ERR_isError(dictID) {
                return dictID;
            }
        }
        cctx.stream_dict = dict.to_vec();
        cctx.stream_dict_original = dict.to_vec();
        cctx.stream_dict_content_type = ZSTD_dictContentType_e::ZSTD_dct_rawContent;
        cctx.dictContentSize = dict.len();
        cctx.dictID = 0;
        return 0;
    }

    let magic = MEM_readLE32(&dict[..4]);
    if magic != ZSTD_MAGICNUMBER_DICTIONARY {
        if dictContentType == ZSTD_dictContentType_e::ZSTD_dct_fullDict {
            return ERROR(ErrorCode::DictionaryWrong);
        }
        let (ms_opt, ldm_opt) = (&mut cctx.ms, &mut cctx.ldmState);
        if let Some(ms) = ms_opt.as_mut() {
            let dictID = ZSTD_loadDictionaryContent(
                ms,
                ldm_opt.as_mut(),
                params,
                dict,
                ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_fast,
                ZSTD_tableFillPurpose_e::ZSTD_tfp_forCCtx,
            );
            if ERR_isError(dictID) {
                return dictID;
            }
        }
        cctx.stream_dict = dict.to_vec();
        cctx.stream_dict_original = dict.to_vec();
        cctx.stream_dict_content_type = dictContentType;
        cctx.dictContentSize = dict.len();
        cctx.dictID = 0;
        return 0;
    }

    let dictID = if let Some(ms) = cctx.ms.as_mut() {
        ZSTD_loadZstdDictionary(
            &mut cctx.prevEntropy,
            &mut cctx.prev_rep,
            ms,
            params,
            dict,
            ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_fast,
            ZSTD_tableFillPurpose_e::ZSTD_tfp_forCCtx,
        )
    } else {
        let eSize = ZSTD_loadCEntropy(&mut cctx.prevEntropy, &mut cctx.prev_rep, dict);
        if ERR_isError(eSize) {
            return eSize;
        }
        MEM_readLE32(&dict[4..8]) as usize
    };
    if ERR_isError(dictID) {
        return dictID;
    }
    let mut entropy = ZSTD_entropyCTables_t::default();
    let mut rep = [0u32; 3];
    let eSize = ZSTD_loadCEntropy(&mut entropy, &mut rep, dict);
    if ERR_isError(eSize) {
        return eSize;
    }
    cctx.dictID = dictID as u32;
    cctx.stream_dict = dict[eSize..].to_vec();
    cctx.stream_dict_original = dict.to_vec();
    cctx.stream_dict_content_type = dictContentType;
    cctx.dictContentSize = dict.len();
    dictID
}

/// Port of `ZSTD_compressBegin_internal` (`zstd_compress.c:5262`).
/// Internal workhorse of the compressBegin family. Mirrors upstream's
/// reset + dictionary-routing shape: reset the CCtx, then immediately
/// seed the working match state from either a raw/full dictionary or a
/// CDict.
pub fn ZSTD_compressBegin_internal(
    cctx: &mut ZSTD_CCtx,
    dict: &[u8],
    dictContentType: crate::decompress::zstd_ddict::ZSTD_dictContentType_e,
    cdict: Option<&ZSTD_CDict>,
    params: &ZSTD_CCtx_params,
    pledgedSrcSize: u64,
    zbuff: ZSTD_buffered_policy_e,
) -> usize {
    debug_assert!(!ERR_isError(ZSTD_checkCParams(params.cParams)));
    debug_assert!(cdict.is_none() || dict.is_empty());
    if let Some(cd) = cdict {
        if ZSTD_shouldAttachDict(cd, params, pledgedSrcSize) {
            return ZSTD_resetCCtx_byAttachingCDict(cctx, cd, *params, pledgedSrcSize, zbuff);
        }
        return ZSTD_resetCCtx_byCopyingCDict(cctx, cd, *params, pledgedSrcSize, zbuff);
    }

    let rc = ZSTD_resetCCtx_internal(
        cctx,
        params,
        pledgedSrcSize,
        dict.len(),
        ZSTD_compResetPolicy_e::ZSTDcrp_makeClean,
        zbuff,
    );
    if ERR_isError(rc) {
        return rc;
    }

    cctx.stream_level = Some(params.compressionLevel);
    cctx.format = params.format;
    cctx.externalMatchStore = None;

    if dict.is_empty() {
        ZSTD_clearAllDicts(cctx);
        return 0;
    }

    let dictID = ZSTD_compress_insertDictionary(cctx, params, dict, dictContentType);
    if ERR_isError(dictID) {
        return dictID;
    }

    cctx.stream_dict.clear();
    cctx.stream_dict_original.clear();
    cctx.stream_dict_content_type =
        crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_auto;
    cctx.prefix_is_single_use = false;
    cctx.dictID = dictID as u32;
    cctx.dictContentSize = dict.len();
    0
}

/// Port of `ZSTD_initCStream_internal` (`zstd_compress.c:6010`).
/// Lower-level variant taking `ZSTD_CCtx_params*`. Resets the
/// session, sets pledged size, stashes the params, and wires in
/// either a raw-content dict or a CDict (exclusive — upstream
/// asserts both aren't set). Delegates to existing Rust API methods.
pub fn ZSTD_initCStream_internal(
    zcs: &mut ZSTD_CCtx,
    dict: &[u8],
    cdict: Option<&ZSTD_CDict>,
    params: &ZSTD_CCtx_params,
    pledgedSrcSize: u64,
) -> usize {
    let rc = ZSTD_CCtx_reset(zcs, ZSTD_ResetDirective::ZSTD_reset_session_only);
    if ERR_isError(rc) {
        return rc;
    }
    let rc = ZSTD_CCtx_setPledgedSrcSize(zcs, pledgedSrcSize);
    if ERR_isError(rc) {
        return rc;
    }
    debug_assert!(!ERR_isError(ZSTD_checkCParams(params.cParams)));
    // "either dict or cdict, not both" — upstream assertion.
    debug_assert!(cdict.is_none() || dict.is_empty());
    // Upstream's `zcs->requestedParams = *params` — keep the full
    // param bundle so subsequent `ZSTD_CCtx_getParameter` reads
    // reflect what the caller actually requested.
    zcs.requestedParams = *params;
    if !dict.is_empty() {
        ZSTD_CCtx_loadDictionary(zcs, dict)
    } else if let Some(cd) = cdict {
        ZSTD_CCtx_refCDict(zcs, cd)
    } else {
        0
    }
}

/// Port of `ZSTD_compressBegin_advanced_internal`
/// (`zstd_compress.c:5309`). Lower-level variant that takes a
/// pre-built `ZSTD_CCtx_params*` + optional CDict. This now routes
/// straight through `ZSTD_compressBegin_internal()`, so the same
/// dict/CDict and session-reset plumbing is used as the rest of the
/// public begin surfaces.
pub fn ZSTD_compressBegin_advanced_internal(
    cctx: &mut ZSTD_CCtx,
    dict: &[u8],
    dictContentType: crate::decompress::zstd_ddict::ZSTD_dictContentType_e,
    cdict: Option<&ZSTD_CDict>,
    params: &ZSTD_CCtx_params,
    pledgedSrcSize: u64,
) -> usize {
    let rc = ZSTD_checkCParams(params.cParams);
    if ERR_isError(rc) {
        return rc;
    }
    ZSTD_compressBegin_internal(
        cctx,
        dict,
        dictContentType,
        cdict,
        params,
        pledgedSrcSize,
        ZSTD_buffered_policy_e::ZSTDb_not_buffered,
    )
}

/// Port of `ZSTD_compressBegin_usingDict_deprecated`
/// (`zstd_compress.c:5341`). Legacy name; forwards to the modern
/// `ZSTD_compressBegin_usingDict`.
#[inline]
pub fn ZSTD_compressBegin_usingDict_deprecated(
    cctx: &mut ZSTD_CCtx,
    dict: &[u8],
    compressionLevel: i32,
) -> usize {
    ZSTD_compressBegin_usingDict(cctx, dict, compressionLevel)
}

/// Port of `ZSTD_compressBegin_usingCDict_deprecated`
/// (`zstd_compress.c:5897`). Legacy name; forwards to modern
/// `ZSTD_compressBegin_usingCDict`.
#[inline]
pub fn ZSTD_compressBegin_usingCDict_deprecated(cctx: &mut ZSTD_CCtx, cdict: &ZSTD_CDict) -> usize {
    ZSTD_compressBegin_usingCDict(cctx, cdict)
}

/// Port of `ZSTD_CCtx_refThreadPool`. Wires an external thread pool
/// into the compressor.
#[cfg(feature = "std")]
#[inline]
pub fn ZSTD_CCtx_refThreadPool(
    cctx: &mut ZSTD_CCtx,
    pool: Option<&crate::common::pool::POOL_ctx>,
) -> usize {
    // Upstream (zstd_compress.c:1356) stage-gates thread-pool binding
    // the same way as the dict-family setters. Attaching a pool mid-
    // stream is undefined in upstream and could race with an active
    // worker dispatch once MT support lands. Gate now so the contract
    // is consistent across the CCtx API surface.
    if !cctx_is_in_init_stage(cctx) {
        return ERROR(ErrorCode::StageWrong);
    }
    cctx.threadPoolRef = pool.map_or(0, |p| p as *const crate::common::pool::POOL_ctx as usize);
    cctx.rayonThreadPoolRef = 0;
    cctx.mtctxSizeHint = pool.map_or(0, crate::common::pool::POOL_sizeof);
    0
}

/// Attaches an external rayon thread pool for multi-threaded compression.
///
/// The caller owns the rayon pool and must keep it alive while operations
/// using this context run. Passing `None` clears the rayon pool reference.
#[cfg(feature = "mt")]
#[inline]
pub fn ZSTD_CCtx_refRayonThreadPool(
    cctx: &mut ZSTD_CCtx,
    pool: Option<&rayon::ThreadPool>,
) -> usize {
    if !cctx_is_in_init_stage(cctx) {
        return ERROR(ErrorCode::StageWrong);
    }
    cctx.rayonThreadPoolRef = pool.map_or(0, |p| p as *const rayon::ThreadPool as usize);
    if pool.is_some() {
        cctx.threadPoolRef = 0;
    }
    cctx.mtctxSizeHint = if pool.is_some() {
        core::mem::size_of::<rayon::ThreadPool>()
    } else if cctx.threadPoolRef == 0 {
        0
    } else {
        cctx.mtctxSizeHint
    };
    0
}

/// Port of `ZSTD_CCtx_refPrefix_advanced`. Extends `refPrefix` with
/// an explicit `ZSTD_dictContentType_e`, routing through the same
/// dictionary parser used by `compressBegin_internal()` before marking
/// the prefix single-use.
pub fn ZSTD_CCtx_refPrefix_advanced(
    cctx: &mut ZSTD_CCtx,
    prefix: &[u8],
    dictContentType: crate::decompress::zstd_ddict::ZSTD_dictContentType_e,
) -> usize {
    if !cctx_is_in_init_stage(cctx) {
        return ERROR(ErrorCode::StageWrong);
    }
    ZSTD_clearAllDicts(cctx);
    if prefix.is_empty() {
        return 0;
    }
    let params = cctx.requestedParams;
    let rc = ZSTD_compress_insertDictionary(cctx, &params, prefix, dictContentType);
    if ERR_isError(rc) {
        return rc;
    }
    cctx.prefix_is_single_use = true;
    0
}

/// Port of `ZSTD_CCtx_loadDictionary_advanced`. Upstream takes
/// `dictLoadMethod` and `dictContentType`. The Rust port matches
/// upstream's lazy load behavior: stash the dictionary bytes and parse
/// them when the next compression session is initialized.
pub fn ZSTD_CCtx_loadDictionary_advanced(
    cctx: &mut ZSTD_CCtx,
    dict: &[u8],
    dictLoadMethod: crate::decompress::zstd_ddict::ZSTD_dictLoadMethod_e,
    dictContentType: crate::decompress::zstd_ddict::ZSTD_dictContentType_e,
) -> usize {
    use crate::decompress::zstd_ddict::ZSTD_dictLoadMethod_e;

    if !cctx_is_in_init_stage(cctx) {
        return ERROR(ErrorCode::StageWrong);
    }
    ZSTD_clearAllDicts(cctx);
    if dict.is_empty() {
        return 0;
    }
    if cctx.staticSize != 0 && dictLoadMethod == ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy {
        return ERROR(ErrorCode::MemoryAllocation);
    }
    cctx.stream_dict = dict.to_vec();
    cctx.stream_dict_original = dict.to_vec();
    cctx.stream_dict_content_type = dictContentType;
    cctx.dictContentSize = dict.len();
    cctx.dictID = 0;
    0
}

/// Port of `ZSTD_CCtx_loadDictionary_byReference`. Forwards to the
/// core loader — there's no shared ownership savings to claim
/// until we surface a non-owning CDict variant.
#[inline]
pub fn ZSTD_CCtx_loadDictionary_byReference(cctx: &mut ZSTD_CCtx, dict: &[u8]) -> usize {
    ZSTD_CCtx_loadDictionary(cctx, dict)
}

/// Port of `ZSTD_CCtx_refCDict`. Wires a pre-built CDict into the
/// CCtx for subsequent session initialization.
pub fn ZSTD_CCtx_refCDict(cctx: &mut ZSTD_CCtx, cdict: &ZSTD_CDict) -> usize {
    // Upstream (zstd_compress.c:1346) stage-gates the same way as
    // loadDictionary / setPledgedSrcSize — swapping a CDict mid-
    // session would leave already-buffered bytes compressed against
    // mismatched state.
    if !cctx_is_in_init_stage(cctx) {
        return ERROR(ErrorCode::StageWrong);
    }
    // Upstream (zstd_compress.c:1348) clears any prior dict state
    // before installing the new CDict. Mirror so a `refCDict` after
    // an earlier `refPrefix` / `loadDictionary` doesn't leave stale
    // fields (e.g. `prefix_is_single_use`) set.
    ZSTD_clearAllDicts(cctx);
    cctx.stream_cdict = Some(cdict.clone());
    cctx.stream_level = Some(cdict.compressionLevel);
    cctx.dictID = cdict.dictID;
    cctx.dictContentSize = cdict.dictContent.len();
    // refCDict attaches persistently — the single-use flag stays
    // at the `clearAllDicts` default (`false`).
    0
}

/// Port of `ZSTD_minCLevel`. Returns the minimum valid (most
/// negative) level supported. Upstream: 1 - (1 << 17) = -131071.
pub fn ZSTD_minCLevel() -> i32 {
    1 - (1 << 17)
}

/// Port of `ZSTD_maxCLevel`. Returns `ZSTD_MAX_CLEVEL = 22`.
pub fn ZSTD_maxCLevel() -> i32 {
    ZSTD_MAX_CLEVEL
}

/// Port of `ZSTD_defaultCLevel`. Upstream default is 3.
pub fn ZSTD_defaultCLevel() -> i32 {
    ZSTD_CLEVEL_DEFAULT
}

/// Port of `ZSTD_CStreamInSize`. Suggested input-buffer size for
/// streaming compression — upstream returns `ZSTD_BLOCKSIZE_MAX`.
pub fn ZSTD_CStreamInSize() -> usize {
    crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX
}

/// Port of `ZSTD_CStreamOutSize`. Suggested output-buffer size —
/// `ZSTD_compressBound(ZSTD_BLOCKSIZE_MAX) + header/trailer margin`.
pub fn ZSTD_CStreamOutSize() -> usize {
    // Upstream (zstd_compress.c:5980):
    //   ZSTD_compressBound(ZSTD_BLOCKSIZE_MAX) + ZSTD_blockHeaderSize + 4
    // — the 4 bytes cover the optional XXH64 checksum trailer.
    // Previously used `ZSTD_FRAMEHEADERSIZE_MAX` (18) which over-
    // estimated the stream-end margin by 15 bytes.
    use crate::decompress::zstd_decompress_block::{ZSTD_blockHeaderSize, ZSTD_BLOCKSIZE_MAX};
    ZSTD_compressBound(ZSTD_BLOCKSIZE_MAX) + ZSTD_blockHeaderSize + 4
}

/// Upstream `ZSTD_MAX_CLEVEL` — maximum supported positive level.
pub const ZSTD_MAX_CLEVEL: i32 = 22;

/// Upstream `ZSTD_NO_CLEVEL` sentinel — used by advanced-init paths
/// to say "params did not come from a compression level".
pub const ZSTD_NO_CLEVEL: i32 = 0;

/// Upstream `ZSTD_CLEVEL_DEFAULT`.
pub const ZSTD_CLEVEL_DEFAULT: i32 = 3;

/// `(windowLog, chainLog, hashLog, searchLog, minMatch, targetLength,
/// strategy)` — one row of upstream's `ZSTD_defaultCParameters` table.
pub type CParamRow = (u32, u32, u32, u32, u32, u32, u32);

/// Port of upstream's `ZSTD_defaultCParameters[4][23]` table from
/// `lib/compress/clevels.h`. Indexed by `[tableID][row]` where
/// `tableID` is 0 (>256 KB src), 1 (≤256 KB), 2 (≤128 KB), or 3
/// (≤16 KB) and `row` is the compression level (row 0 is baseline
/// for negative levels).
pub const ZSTD_DEFAULT_CPARAMS: [[CParamRow; 23]; 4] = [
    // tableID 0 — "default" (> 256 KB)
    [
        (19, 12, 13, 1, 6, 1, 1), // base for negative
        (19, 13, 14, 1, 7, 0, 1),
        (20, 15, 16, 1, 6, 0, 1),
        (21, 16, 17, 1, 5, 0, 2),
        (21, 18, 18, 1, 5, 0, 2),
        (21, 18, 19, 3, 5, 2, 3),
        (21, 18, 19, 3, 5, 4, 4),
        (21, 19, 20, 4, 5, 8, 4),
        (21, 19, 20, 4, 5, 16, 5),
        (22, 20, 21, 4, 5, 16, 5),
        (22, 21, 22, 5, 5, 16, 5),
        (22, 21, 22, 6, 5, 16, 5),
        (22, 22, 23, 6, 5, 32, 5),
        (22, 22, 22, 4, 5, 32, 6),
        (22, 22, 23, 5, 5, 32, 6),
        (22, 23, 23, 6, 5, 32, 6),
        (22, 22, 22, 5, 5, 48, 7),
        (23, 23, 22, 5, 4, 64, 7),
        (23, 23, 22, 6, 3, 64, 8),
        (23, 24, 22, 7, 3, 256, 9),
        (25, 25, 23, 7, 3, 256, 9),
        (26, 26, 24, 7, 3, 512, 9),
        (27, 27, 25, 9, 3, 999, 9),
    ],
    // tableID 1 — ≤ 256 KB
    [
        (18, 12, 13, 1, 5, 1, 1),
        (18, 13, 14, 1, 6, 0, 1),
        (18, 14, 14, 1, 5, 0, 2),
        (18, 16, 16, 1, 4, 0, 2),
        (18, 16, 17, 3, 5, 2, 3),
        (18, 17, 18, 5, 5, 2, 3),
        (18, 18, 19, 3, 5, 4, 4),
        (18, 18, 19, 4, 4, 4, 4),
        (18, 18, 19, 4, 4, 8, 5),
        (18, 18, 19, 5, 4, 8, 5),
        (18, 18, 19, 6, 4, 8, 5),
        (18, 18, 19, 5, 4, 12, 6),
        (18, 19, 19, 7, 4, 12, 6),
        (18, 18, 19, 4, 4, 16, 7),
        (18, 18, 19, 4, 3, 32, 7),
        (18, 18, 19, 6, 3, 128, 7),
        (18, 19, 19, 6, 3, 128, 8),
        (18, 19, 19, 8, 3, 256, 8),
        (18, 19, 19, 6, 3, 128, 9),
        (18, 19, 19, 8, 3, 256, 9),
        (18, 19, 19, 10, 3, 512, 9),
        (18, 19, 19, 12, 3, 512, 9),
        (18, 19, 19, 13, 3, 999, 9),
    ],
    // tableID 2 — ≤ 128 KB
    [
        (17, 12, 12, 1, 5, 1, 1),
        (17, 12, 13, 1, 6, 0, 1),
        (17, 13, 15, 1, 5, 0, 1),
        (17, 15, 16, 2, 5, 0, 2),
        (17, 17, 17, 2, 4, 0, 2),
        (17, 16, 17, 3, 4, 2, 3),
        (17, 16, 17, 3, 4, 4, 4),
        (17, 16, 17, 3, 4, 8, 5),
        (17, 16, 17, 4, 4, 8, 5),
        (17, 16, 17, 5, 4, 8, 5),
        (17, 16, 17, 6, 4, 8, 5),
        (17, 17, 17, 5, 4, 8, 6),
        (17, 18, 17, 7, 4, 12, 6),
        (17, 18, 17, 3, 4, 12, 7),
        (17, 18, 17, 4, 3, 32, 7),
        (17, 18, 17, 6, 3, 256, 7),
        (17, 18, 17, 6, 3, 128, 8),
        (17, 18, 17, 8, 3, 256, 8),
        (17, 18, 17, 10, 3, 512, 8),
        (17, 18, 17, 5, 3, 256, 9),
        (17, 18, 17, 7, 3, 512, 9),
        (17, 18, 17, 9, 3, 512, 9),
        (17, 18, 17, 11, 3, 999, 9),
    ],
    // tableID 3 — ≤ 16 KB
    [
        (14, 12, 13, 1, 5, 1, 1),
        (14, 14, 15, 1, 5, 0, 1),
        (14, 14, 15, 1, 4, 0, 1),
        (14, 14, 15, 2, 4, 0, 2),
        (14, 14, 14, 4, 4, 2, 3),
        (14, 14, 14, 3, 4, 4, 4),
        (14, 14, 14, 4, 4, 8, 5),
        (14, 14, 14, 6, 4, 8, 5),
        (14, 14, 14, 8, 4, 8, 5),
        (14, 15, 14, 5, 4, 8, 6),
        (14, 15, 14, 9, 4, 8, 6),
        (14, 15, 14, 3, 4, 12, 7),
        (14, 15, 14, 4, 3, 24, 7),
        (14, 15, 14, 5, 3, 32, 8),
        (14, 15, 15, 6, 3, 64, 8),
        (14, 15, 15, 7, 3, 256, 8),
        (14, 15, 15, 5, 3, 48, 9),
        (14, 15, 15, 6, 3, 128, 9),
        (14, 15, 15, 7, 3, 256, 9),
        (14, 15, 15, 8, 3, 256, 9),
        (14, 15, 15, 8, 3, 512, 9),
        (14, 15, 15, 9, 3, 512, 9),
        (14, 15, 15, 10, 3, 999, 9),
    ],
];

/// Port of `ZSTD_getCParams_internal` (`zstd_compress.c:8286`).
/// Returns baseline cParams for a (level, srcSizeHint, dictSize,
/// mode) combination, then funnels through
/// `ZSTD_adjustCParams_internal` for final clamping. Picks the
/// tableID row via `ZSTD_getCParamRowSize`.
pub fn ZSTD_getCParams_internal(
    compressionLevel: i32,
    srcSizeHint: u64,
    dictSize: usize,
    mode: ZSTD_CParamMode_e,
) -> crate::compress::match_state::ZSTD_compressionParameters {
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
    let rSize = ZSTD_getCParamRowSize(srcSizeHint, dictSize, mode);
    let tableID: usize = (rSize <= 256 * 1024) as usize
        + (rSize <= 128 * 1024) as usize
        + (rSize <= 16 * 1024) as usize;

    let row = if compressionLevel == 0 {
        ZSTD_CLEVEL_DEFAULT as usize
    } else if compressionLevel < 0 {
        0
    } else if compressionLevel > ZSTD_MAX_CLEVEL {
        ZSTD_MAX_CLEVEL as usize
    } else {
        compressionLevel as usize
    };

    let (wl, cl, hl, sl, mm, tl, strat) = ZSTD_DEFAULT_CPARAMS[tableID][row];
    let mut cp = crate::compress::match_state::ZSTD_compressionParameters {
        windowLog: wl,
        chainLog: cl,
        hashLog: hl,
        searchLog: sl,
        minMatch: mm,
        targetLength: tl,
        strategy: strat,
    };
    if compressionLevel < 0 {
        cp.targetLength = (-compressionLevel) as u32;
    }
    ZSTD_adjustCParams_internal(
        cp,
        srcSizeHint,
        dictSize as u64,
        mode,
        ZSTD_ParamSwitch_e::ZSTD_ps_auto,
    )
}

/// Port of `ZSTD_getCParams` (`zstd_compress.c:8314`). Public wrapper
/// of `ZSTD_getCParams_internal` with `mode = ZSTD_cpm_unknown`.
/// `srcSizeHint == 0` maps to `ZSTD_CONTENTSIZE_UNKNOWN` per upstream.
pub fn ZSTD_getCParams(
    compressionLevel: i32,
    srcSizeHint: u64,
    dictSize: usize,
) -> crate::compress::match_state::ZSTD_compressionParameters {
    use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;
    let effective_hint = if srcSizeHint == 0 {
        ZSTD_CONTENTSIZE_UNKNOWN
    } else {
        srcSizeHint
    };
    ZSTD_getCParams_internal(
        compressionLevel,
        effective_hint,
        dictSize,
        ZSTD_CParamMode_e::ZSTD_cpm_unknown,
    )
}

/// Port of `ZSTD_cParameter` — the parameter id enum for the
/// parametric `ZSTD_CCtx_setParameter` / `ZSTD_CCtx_getParameter`
/// API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ZSTD_cParameter {
    ZSTD_c_compressionLevel = 100,
    ZSTD_c_windowLog = 101,
    ZSTD_c_hashLog = 102,
    ZSTD_c_chainLog = 103,
    ZSTD_c_searchLog = 104,
    ZSTD_c_minMatch = 105,
    ZSTD_c_targetLength = 106,
    ZSTD_c_strategy = 107,
    ZSTD_c_targetCBlockSize = 130,
    ZSTD_c_enableLongDistanceMatching = 160,
    ZSTD_c_ldmHashLog = 161,
    ZSTD_c_ldmMinMatch = 162,
    ZSTD_c_ldmBucketSizeLog = 163,
    ZSTD_c_ldmHashRateLog = 164,
    ZSTD_c_checksumFlag = 201,
    ZSTD_c_contentSizeFlag = 200,
    /// `ZSTD_c_dictIDFlag`: emit dictID into the frame header when
    /// applicable (default: 1).
    ZSTD_c_dictIDFlag = 202,
    /// Upstream `ZSTD_c_format` = `ZSTD_c_experimentalParam2` = 10
    /// (`zstd.h:523, 2050`). Toggles between `ZSTD_f_zstd1` (default)
    /// and `ZSTD_f_zstd1_magicless`. Sibling of decoder-side
    /// `ZSTD_d_format` (= `ZSTD_d_experimentalParam1` = 1000).
    ZSTD_c_format = 10,
    /// Upstream `ZSTD_c_nbWorkers` (400). Worker count for
    /// multithreaded compression. v0.1 is single-threaded so
    /// only 0 is accepted — any positive value returns
    /// `ParameterUnsupported` to match upstream's contract that
    /// MT-unsupported builds reject non-zero workers.
    ZSTD_c_nbWorkers = 400,
    ZSTD_c_jobSize = 401,
    ZSTD_c_overlapLog = 402,
    ZSTD_c_rsyncable = 500,
    ZSTD_c_forceMaxWindow = 1000,
    ZSTD_c_forceAttachDict = 1001,
    ZSTD_c_literalCompressionMode = 1002,
    ZSTD_c_srcSizeHint = 1004,
    ZSTD_c_enableDedicatedDictSearch = 1005,
    /// Upstream `ZSTD_c_stableInBuffer` = `ZSTD_c_experimentalParam9` = 1006.
    ZSTD_c_stableInBuffer = 1006,
    /// Upstream `ZSTD_c_stableOutBuffer` = `ZSTD_c_experimentalParam10` = 1007.
    ZSTD_c_stableOutBuffer = 1007,
    ZSTD_c_blockDelimiters = 1008,
    ZSTD_c_validateSequences = 1009,
    ZSTD_c_splitAfterSequences = 1010,
    ZSTD_c_useRowMatchFinder = 1011,
    ZSTD_c_deterministicRefPrefix = 1012,
    ZSTD_c_prefetchCDictTables = 1013,
    /// Upstream `ZSTD_c_enableSeqProducerFallback` = `ZSTD_c_experimentalParam17` = 1014.
    ZSTD_c_enableSeqProducerFallback = 1014,
    ZSTD_c_maxBlockSize = 1015,
    ZSTD_c_repcodeResolution = 1016,
    /// Upstream `ZSTD_c_blockSplitterLevel` = `ZSTD_c_experimentalParam20` = 1017.
    ZSTD_c_blockSplitterLevel = 1017,
}

#[inline]
fn ZSTD_cParam_acceptsDefaultZero(param: ZSTD_cParameter) -> bool {
    matches!(
        param,
        ZSTD_cParameter::ZSTD_c_windowLog
            | ZSTD_cParameter::ZSTD_c_hashLog
            | ZSTD_cParameter::ZSTD_c_chainLog
            | ZSTD_cParameter::ZSTD_c_searchLog
            | ZSTD_cParameter::ZSTD_c_minMatch
            | ZSTD_cParameter::ZSTD_c_targetLength
            | ZSTD_cParameter::ZSTD_c_strategy
            | ZSTD_cParameter::ZSTD_c_ldmHashLog
            | ZSTD_cParameter::ZSTD_c_ldmMinMatch
            | ZSTD_cParameter::ZSTD_c_ldmBucketSizeLog
            | ZSTD_cParameter::ZSTD_c_ldmHashRateLog
            | ZSTD_cParameter::ZSTD_c_targetCBlockSize
            | ZSTD_cParameter::ZSTD_c_srcSizeHint
            | ZSTD_cParameter::ZSTD_c_maxBlockSize
    )
}

#[inline]
fn ZSTD_validateCParamSetterValue(param: ZSTD_cParameter, value: i32) -> usize {
    if value == 0 && ZSTD_cParam_acceptsDefaultZero(param) {
        return 0;
    }
    let bounds = ZSTD_cParam_getBounds(param);
    if ERR_isError(bounds.error) {
        return bounds.error;
    }
    if value < bounds.lowerBound || value > bounds.upperBound {
        return ERROR(ErrorCode::ParameterOutOfBound);
    }
    0
}

/// Port of `ZSTD_CCtx_setParameter`. Stashes the value on the CCtx
/// for subsequent calls. For `compressionLevel`, behavior matches
/// `ZSTD_initCStream(level)`. For the frame flags, the next
/// compression call honors them.
pub fn ZSTD_CCtx_setParameter(cctx: &mut ZSTD_CCtx, param: ZSTD_cParameter, value: i32) -> usize {
    // Upstream (zstd_compress.c:727) stage-gates mid-session param
    // updates: only the `ZSTD_isUpdateAuthorized` subset can change
    // once input has been staged. Everything else — format, frame
    // flags, nbWorkers — rejects with `StageWrong`.
    if !cctx_is_in_init_stage(cctx) && !ZSTD_isUpdateAuthorized(param) {
        return ERROR(ErrorCode::StageWrong);
    }
    match param {
        ZSTD_cParameter::ZSTD_c_compressionLevel => {
            // Upstream (zstd_compress.c:797) clamps via
            // `ZSTD_cParam_clampBounds` and treats `value == 0` as
            // `ZSTD_CLEVEL_DEFAULT`. Rust port was previously a raw
            // stash — getParameter would return 0 or out-of-range
            // values unmodified.
            let mut clamped = value;
            let rc =
                ZSTD_cParam_clampBounds(ZSTD_cParameter::ZSTD_c_compressionLevel, &mut clamped);
            if ERR_isError(rc) {
                return rc;
            }
            let level = if clamped == 0 {
                ZSTD_CLEVEL_DEFAULT
            } else {
                clamped
            };
            cctx.stream_level = Some(level);
            cctx.requestedParams.compressionLevel = level;
            if level >= 0 {
                level as usize
            } else {
                0
            }
        }
        ZSTD_cParameter::ZSTD_c_windowLog
        | ZSTD_cParameter::ZSTD_c_hashLog
        | ZSTD_cParameter::ZSTD_c_chainLog
        | ZSTD_cParameter::ZSTD_c_searchLog
        | ZSTD_cParameter::ZSTD_c_minMatch
        | ZSTD_cParameter::ZSTD_c_targetLength
        | ZSTD_cParameter::ZSTD_c_strategy => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            let mut cParams = cctx.requestedParams.cParams;
            match param {
                ZSTD_cParameter::ZSTD_c_windowLog => cParams.windowLog = value as u32,
                ZSTD_cParameter::ZSTD_c_hashLog => cParams.hashLog = value as u32,
                ZSTD_cParameter::ZSTD_c_chainLog => cParams.chainLog = value as u32,
                ZSTD_cParameter::ZSTD_c_searchLog => cParams.searchLog = value as u32,
                ZSTD_cParameter::ZSTD_c_minMatch => cParams.minMatch = value as u32,
                ZSTD_cParameter::ZSTD_c_targetLength => cParams.targetLength = value as u32,
                ZSTD_cParameter::ZSTD_c_strategy => cParams.strategy = value as u32,
                _ => unreachable!(),
            }
            cctx.requested_cParams = Some(cParams);
            cctx.requestedParams.cParams = cParams;
            value as usize
        }
        ZSTD_cParameter::ZSTD_c_enableLongDistanceMatching => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            cctx.requestedParams.ldmEnable = match value {
                1 => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_enable,
                2 => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_disable,
                _ => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_auto,
            };
            cctx.requestedParams.ldmParams.enableLdm = cctx.requestedParams.ldmEnable;
            cctx.requestedParams.ldmEnable as usize
        }
        ZSTD_cParameter::ZSTD_c_ldmHashLog
        | ZSTD_cParameter::ZSTD_c_ldmMinMatch
        | ZSTD_cParameter::ZSTD_c_ldmBucketSizeLog
        | ZSTD_cParameter::ZSTD_c_ldmHashRateLog => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            match param {
                ZSTD_cParameter::ZSTD_c_ldmHashLog => {
                    cctx.requestedParams.ldmParams.hashLog = value as u32
                }
                ZSTD_cParameter::ZSTD_c_ldmMinMatch => {
                    cctx.requestedParams.ldmParams.minMatchLength = value as u32
                }
                ZSTD_cParameter::ZSTD_c_ldmBucketSizeLog => {
                    cctx.requestedParams.ldmParams.bucketSizeLog = value as u32
                }
                ZSTD_cParameter::ZSTD_c_ldmHashRateLog => {
                    cctx.requestedParams.ldmParams.hashRateLog = value as u32
                }
                _ => unreachable!(),
            }
            value as usize
        }
        ZSTD_cParameter::ZSTD_c_targetCBlockSize => {
            let mut adjusted = value;
            if adjusted != 0 && adjusted < ZSTD_TARGETCBLOCKSIZE_MIN as i32 {
                adjusted = ZSTD_TARGETCBLOCKSIZE_MIN as i32;
            }
            let rc = ZSTD_validateCParamSetterValue(param, adjusted);
            if ERR_isError(rc) {
                return rc;
            }
            cctx.requestedParams.targetCBlockSize = adjusted as usize;
            cctx.requestedParams.targetCBlockSize
        }
        ZSTD_cParameter::ZSTD_c_stableInBuffer => {
            let bounds = ZSTD_cParam_getBounds(ZSTD_cParameter::ZSTD_c_stableInBuffer);
            if value < bounds.lowerBound || value > bounds.upperBound {
                return ERROR(ErrorCode::ParameterOutOfBound);
            }
            cctx.requestedParams.inBufferMode = if value == ZSTD_bufferMode_e::ZSTD_bm_stable as i32
            {
                ZSTD_bufferMode_e::ZSTD_bm_stable
            } else {
                ZSTD_bufferMode_e::ZSTD_bm_buffered
            };
            cctx.requestedParams.inBufferMode as usize
        }
        ZSTD_cParameter::ZSTD_c_stableOutBuffer => {
            let bounds = ZSTD_cParam_getBounds(ZSTD_cParameter::ZSTD_c_stableOutBuffer);
            if value < bounds.lowerBound || value > bounds.upperBound {
                return ERROR(ErrorCode::ParameterOutOfBound);
            }
            cctx.requestedParams.outBufferMode =
                if value == ZSTD_bufferMode_e::ZSTD_bm_stable as i32 {
                    ZSTD_bufferMode_e::ZSTD_bm_stable
                } else {
                    ZSTD_bufferMode_e::ZSTD_bm_buffered
                };
            cctx.requestedParams.outBufferMode as usize
        }
        ZSTD_cParameter::ZSTD_c_rsyncable => {
            #[cfg(not(feature = "mt"))]
            {
                if value != 0 {
                    return ERROR(ErrorCode::ParameterUnsupported);
                }
                return 0;
            }
            #[cfg(feature = "mt")]
            {
                let rc = ZSTD_validateCParamSetterValue(param, value);
                if ERR_isError(rc) {
                    return rc;
                }
                cctx.requestedParams.rsyncable = value;
                cctx.requestedParams.rsyncable as usize
            }
        }
        ZSTD_cParameter::ZSTD_c_enableSeqProducerFallback => {
            let bounds = ZSTD_cParam_getBounds(ZSTD_cParameter::ZSTD_c_enableSeqProducerFallback);
            if value < bounds.lowerBound || value > bounds.upperBound {
                return ERROR(ErrorCode::ParameterOutOfBound);
            }
            cctx.requestedParams.enableMatchFinderFallback = value;
            cctx.requestedParams.enableMatchFinderFallback as usize
        }
        ZSTD_cParameter::ZSTD_c_blockSplitterLevel => {
            let bounds = ZSTD_cParam_getBounds(ZSTD_cParameter::ZSTD_c_blockSplitterLevel);
            if value < bounds.lowerBound || value > bounds.upperBound {
                return ERROR(ErrorCode::ParameterOutOfBound);
            }
            cctx.requestedParams.preBlockSplitter_level = value;
            cctx.requestedParams.preBlockSplitter_level as usize
        }
        ZSTD_cParameter::ZSTD_c_forceMaxWindow => {
            cctx.requestedParams.forceWindow = (value != 0) as i32;
            cctx.requestedParams.forceWindow as usize
        }
        ZSTD_cParameter::ZSTD_c_forceAttachDict => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            cctx.requestedParams.attachDictPref = match value {
                1 => ZSTD_dictAttachPref_e::ZSTD_dictForceAttach,
                2 => ZSTD_dictAttachPref_e::ZSTD_dictForceCopy,
                3 => ZSTD_dictAttachPref_e::ZSTD_dictForceLoad,
                _ => ZSTD_dictAttachPref_e::ZSTD_dictDefaultAttach,
            };
            cctx.requestedParams.attachDictPref as usize
        }
        ZSTD_cParameter::ZSTD_c_literalCompressionMode => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            cctx.requestedParams.literalCompressionMode = match value {
                1 => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_enable,
                2 => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_disable,
                _ => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_auto,
            };
            cctx.requestedParams.literalCompressionMode as usize
        }
        ZSTD_cParameter::ZSTD_c_srcSizeHint => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            cctx.requestedParams.srcSizeHint = value;
            cctx.requestedParams.srcSizeHint as usize
        }
        ZSTD_cParameter::ZSTD_c_enableDedicatedDictSearch => {
            cctx.requestedParams.enableDedicatedDictSearch = (value != 0) as i32;
            cctx.requestedParams.enableDedicatedDictSearch as usize
        }
        ZSTD_cParameter::ZSTD_c_blockDelimiters => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            cctx.requestedParams.blockDelimiters = match value {
                1 => ZSTD_SequenceFormat_e::ZSTD_sf_explicitBlockDelimiters,
                _ => ZSTD_SequenceFormat_e::ZSTD_sf_noBlockDelimiters,
            };
            cctx.requestedParams.blockDelimiters as usize
        }
        ZSTD_cParameter::ZSTD_c_validateSequences => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            cctx.requestedParams.validateSequences = value;
            cctx.requestedParams.validateSequences as usize
        }
        ZSTD_cParameter::ZSTD_c_splitAfterSequences => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            cctx.requestedParams.postBlockSplitter = match value {
                1 => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_enable,
                2 => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_disable,
                _ => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_auto,
            };
            cctx.requestedParams.postBlockSplitter as usize
        }
        ZSTD_cParameter::ZSTD_c_useRowMatchFinder => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            cctx.requestedParams.useRowMatchFinder = match value {
                1 => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_enable,
                2 => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_disable,
                _ => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_auto,
            };
            cctx.requestedParams.useRowMatchFinder as usize
        }
        ZSTD_cParameter::ZSTD_c_deterministicRefPrefix => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            cctx.requestedParams.deterministicRefPrefix = (value != 0) as i32;
            cctx.requestedParams.deterministicRefPrefix as usize
        }
        ZSTD_cParameter::ZSTD_c_prefetchCDictTables => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            cctx.requestedParams.prefetchCDictTables = match value {
                1 => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_enable,
                2 => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_disable,
                _ => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_auto,
            };
            cctx.requestedParams.prefetchCDictTables as usize
        }
        ZSTD_cParameter::ZSTD_c_maxBlockSize => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            cctx.requestedParams.maxBlockSize = value as usize;
            cctx.requestedParams.maxBlockSize
        }
        ZSTD_cParameter::ZSTD_c_repcodeResolution => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            cctx.requestedParams.searchForExternalRepcodes = match value {
                1 => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_enable,
                2 => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_disable,
                _ => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_auto,
            };
            cctx.requestedParams.searchForExternalRepcodes as usize
        }
        ZSTD_cParameter::ZSTD_c_checksumFlag => {
            cctx.param_checksum = value != 0;
            cctx.requestedParams.fParams.checksumFlag = (value != 0) as u32;
            cctx.requestedParams.fParams.checksumFlag as usize
        }
        ZSTD_cParameter::ZSTD_c_contentSizeFlag => {
            cctx.param_contentSize = value != 0;
            cctx.requestedParams.fParams.contentSizeFlag = (value != 0) as u32;
            cctx.requestedParams.fParams.contentSizeFlag as usize
        }
        ZSTD_cParameter::ZSTD_c_dictIDFlag => {
            cctx.param_dictID = value != 0;
            // Upstream `noDictIDFlag = !dictIDFlag`.
            cctx.requestedParams.fParams.noDictIDFlag = (value == 0) as u32;
            if cctx.requestedParams.fParams.noDictIDFlag != 0 {
                0
            } else {
                1
            }
        }
        ZSTD_cParameter::ZSTD_c_format => {
            // Upstream bounds-checks against `[zstd1, magicless]`.
            let bounds = ZSTD_cParam_getBounds(ZSTD_cParameter::ZSTD_c_format);
            if value < bounds.lowerBound || value > bounds.upperBound {
                return ERROR(ErrorCode::ParameterOutOfBound);
            }
            let new_format = match value {
                v if v
                    == crate::decompress::zstd_decompress::ZSTD_format_e::ZSTD_f_zstd1_magicless
                        as i32 =>
                {
                    crate::decompress::zstd_decompress::ZSTD_format_e::ZSTD_f_zstd1_magicless
                }
                _ => crate::decompress::zstd_decompress::ZSTD_format_e::ZSTD_f_zstd1,
            };
            cctx.format = new_format;
            // Keep the parametric shadow in sync so
            // `ZSTD_CCtxParams_getParameter(cctx.requestedParams, ...)`
            // reports the same value — matches upstream where
            // `cctx->requestedParams.format` is the canonical slot.
            cctx.requestedParams.format = new_format;
            cctx.requestedParams.format as usize
        }
        ZSTD_cParameter::ZSTD_c_nbWorkers => {
            if value != 0 && cctx.staticSize != 0 {
                return ERROR(ErrorCode::ParameterUnsupported);
            }
            #[cfg(not(feature = "mt"))]
            {
                if value != 0 {
                    return ERROR(ErrorCode::ParameterUnsupported);
                }
                cctx.requestedParams.nbWorkers = 0;
                return 0;
            }
            #[cfg(feature = "mt")]
            {
                let bounds = ZSTD_cParam_getBounds(ZSTD_cParameter::ZSTD_c_nbWorkers);
                if value < bounds.lowerBound || value > bounds.upperBound {
                    return ERROR(ErrorCode::ParameterOutOfBound);
                }
                cctx.requestedParams.nbWorkers = value;
                cctx.requestedParams.nbWorkers as usize
            }
        }
        ZSTD_cParameter::ZSTD_c_jobSize => {
            #[cfg(not(feature = "mt"))]
            {
                if value != 0 {
                    return ERROR(ErrorCode::ParameterUnsupported);
                }
                return 0;
            }
            #[cfg(feature = "mt")]
            {
                let mut adjusted = value;
                if adjusted != 0
                    && adjusted < crate::compress::zstdmt_compress::ZSTDMT_JOBSIZE_MIN as i32
                {
                    adjusted = crate::compress::zstdmt_compress::ZSTDMT_JOBSIZE_MIN as i32;
                }
                let bounds = ZSTD_cParam_getBounds(ZSTD_cParameter::ZSTD_c_jobSize);
                if adjusted < bounds.lowerBound || adjusted > bounds.upperBound {
                    return ERROR(ErrorCode::ParameterOutOfBound);
                }
                cctx.requestedParams.jobSize = adjusted as usize;
                cctx.requestedParams.jobSize
            }
        }
        ZSTD_cParameter::ZSTD_c_overlapLog => {
            #[cfg(not(feature = "mt"))]
            {
                if value != 0 {
                    return ERROR(ErrorCode::ParameterUnsupported);
                }
                return 0;
            }
            #[cfg(feature = "mt")]
            {
                let bounds = ZSTD_cParam_getBounds(ZSTD_cParameter::ZSTD_c_overlapLog);
                if value < bounds.lowerBound || value > bounds.upperBound {
                    return ERROR(ErrorCode::ParameterOutOfBound);
                }
                cctx.requestedParams.overlapLog = value;
                cctx.requestedParams.overlapLog as usize
            }
        }
    }
}

/// Port of `ZSTD_CCtx_getParameter`. Reads a previously-set parameter.
pub fn ZSTD_CCtx_getParameter(cctx: &ZSTD_CCtx, param: ZSTD_cParameter, value: &mut i32) -> usize {
    ZSTD_CCtxParams_getParameter(&cctx.requestedParams, param, value)
}

/// Port of `ZSTD_CCtx_reset`. Matches upstream's three modes:
///   * `session_only` — clear per-frame stream state (in/out
///     buffers, drain cursor, closed flag, pledged size). Keeps
///     parameters and dicts so the CCtx is ready for the next
///     compression with the same settings.
///   * `parameters` — restore default parameters and clear the
///     configured dict; preserves any in-flight stream state.
///   * `session_and_parameters` — do both.
pub fn ZSTD_CCtx_reset(cctx: &mut ZSTD_CCtx, reset: ZSTD_ResetDirective) -> usize {
    let clear_session = matches!(
        reset,
        ZSTD_ResetDirective::ZSTD_reset_session_only
            | ZSTD_ResetDirective::ZSTD_reset_session_and_parameters,
    );
    let clear_params = matches!(
        reset,
        ZSTD_ResetDirective::ZSTD_reset_parameters
            | ZSTD_ResetDirective::ZSTD_reset_session_and_parameters,
    );
    // Upstream (zstd_compress.c:1392) rejects a parameters-only reset
    // when `streamStage != zcss_init`. The `session_and_parameters`
    // variant clears session first (which lands in init), so the gate
    // only applies to pure `reset_parameters`. Rejecting here prevents
    // a caller from dropping appliedParams mid-frame while bytes are
    // still buffered against them.
    if clear_params && !clear_session && !cctx_is_in_init_stage(cctx) {
        return ERROR(ErrorCode::StageWrong);
    }
    if clear_session {
        cctx.stream_in_buffer.clear();
        cctx.stream_params_snapshot = None;
        cctx.stream_out_buffer.clear();
        cctx.stream_out_drained = 0;
        cctx.stream_stage = ZSTD_cStreamStage::zcss_init;
        cctx.stream_in_to_compress = 0;
        cctx.stream_in_target = 0;
        cctx.stream_frame_ended = false;
        cctx.stream_window_base = crate::compress::match_state::ZSTD_WINDOW_START_INDEX;
        cctx.expected_in_src = 0;
        cctx.expected_in_size = 0;
        cctx.expected_in_pos = 0;
        cctx.expected_out_buffer_size = 0;
        cctx.buffer_expectations_set = false;
        cctx.stream_closed = false;
        cctx.stream_frame_completed = false;
        cctx.pledged_src_size = None;
        cctx.externalMatchStore = None;
        // Restore the repcode history + entropy repeatModes to
        // frame-start defaults, matching upstream's
        // `ZSTD_reset_compressedBlockState` call path during session
        // reset.
        ZSTD_reset_compressedBlockState(&mut cctx.prev_rep, &mut cctx.prevEntropy);
        ZSTD_reset_compressedBlockState(&mut cctx.next_rep, &mut cctx.nextEntropy);
        // Upstream detects a frame boundary inside `ZSTD_window_update`
        // via pointer comparison (new src pointer != old `nextSrc`
        // pivots into the non-contiguous branch). Our u32-index port
        // always passes `src_abs = window.nextSrc`, so that detection
        // can't fire — without an explicit reset here the next frame
        // would inherit the previous frame's `dictLimit` / `nextSrc`
        // and `getLowestPrefixIndex` would pin the prefix start well
        // above the new block. Re-seed the window sentinels so a
        // reused CCtx behaves byte-for-byte like a freshly created one.
        if let Some(ms) = cctx.ms.as_mut() {
            crate::compress::match_state::ZSTD_window_init(&mut ms.window);
            ms.nextToUpdate = ms.window.dictLimit;
            ms.loadedDictEnd = 0;
        }
        // Reset session counters + stage tracker.
        cctx.stage = ZSTD_compressionStage_e::ZSTDcs_created;
        cctx.consumedSrcSize = 0;
        cctx.producedCSize = 0;
        cctx.pledgedSrcSizePlusOne = 0;
        cctx.isFirstBlock = 1;
        crate::common::xxhash::XXH64_reset(&mut cctx.xxhState, 0);
    }
    if clear_params {
        let customMem = cctx.customMem;
        // Upstream (zstd_compress.c:1394) routes through
        // `ZSTD_clearAllDicts` + `ZSTD_CCtxParams_reset` so every
        // dict-related slot is wiped uniformly. Delegate for parity —
        // a direct field-by-field reset could miss `dictID` or
        // `dictContentSize` if they're ever re-examined by a future
        // path.
        ZSTD_clearAllDicts(cctx);
        cctx.stream_level = None;
        cctx.param_checksum = false;
        cctx.param_contentSize = true;
        cctx.param_dictID = true;
        cctx.expected_in_src = 0;
        cctx.expected_in_size = 0;
        cctx.expected_in_pos = 0;
        cctx.expected_out_buffer_size = 0;
        cctx.buffer_expectations_set = false;
        cctx.requested_cParams = None;
        cctx.requestedParams = ZSTD_CCtx_params::default();
        let rc = ZSTD_CCtxParams_reset(&mut cctx.requestedParams);
        if ERR_isError(rc) {
            return rc;
        }
        cctx.appliedParams = ZSTD_CCtx_params::default();
        let rc = ZSTD_CCtxParams_reset(&mut cctx.appliedParams);
        if ERR_isError(rc) {
            return rc;
        }
        cctx.customMem = customMem;
        cctx.requestedParams.customMem = customMem;
        cctx.appliedParams.customMem = customMem;
        // Upstream's `CCtxParams_reset` re-initializes `format` to
        // `ZSTD_f_zstd1` — a param-level reset must clear the
        // magicless-mode knob too so a subsequent caller doesn't
        // silently inherit it.
        cctx.format = crate::decompress::zstd_decompress::ZSTD_format_e::ZSTD_f_zstd1;
    }
    0
}

/// Port of `ZSTD_ResetDirective` (`zstd.h:589`). Upstream fixes the
/// discriminants at 1/2/3 — not the Rust default 0/1/2. FFI callers
/// passing the numeric values directly rely on this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ZSTD_ResetDirective {
    ZSTD_reset_session_only = 1,
    ZSTD_reset_parameters = 2,
    ZSTD_reset_session_and_parameters = 3,
}

/// Port of `ZSTD_estimateCCtxSize_usingCParams`. Returns a
/// conservative upper bound on the heap memory needed to compress
/// with the given cParams. Accounts for the hash table
/// (`1<<hashLog` u32s), chain table (`1<<chainLog` u32s), seq store
/// (~block-size worth of literals + sequences), and entropy tables
/// (~32 KB). This is an estimate — real allocation is done lazily
/// via `Vec`, and Rust's allocator may round up.
fn ZSTD_estimateCCtxSize_usingCCtxParams_internal(
    cParams: &crate::compress::match_state::ZSTD_compressionParameters,
    params: &ZSTD_CCtx_params,
    isStatic: bool,
    buffInSize: usize,
    buffOutSize: usize,
    pledgedSrcSize: u64,
) -> usize {
    use crate::common::zstd_internal::WILDCOPY_OVERLENGTH;
    use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;

    let windowSize = if pledgedSrcSize == ZSTD_CONTENTSIZE_UNKNOWN {
        1usize << cParams.windowLog
    } else {
        (1usize << cParams.windowLog).min(pledgedSrcSize.max(1) as usize)
    };
    let blockSize =
        crate::compress::match_state::ZSTD_resolveMaxBlockSize(params.maxBlockSize).min(windowSize);
    let maxNbSeq = ZSTD_maxNbSeq(blockSize, cParams.minMatch, ZSTD_hasExtSeqProd(params));
    let tokenSpace = WILDCOPY_OVERLENGTH
        + blockSize
        + maxNbSeq * core::mem::size_of::<crate::compress::seq_store::SeqDef>()
        + 3 * maxNbSeq * core::mem::size_of::<u8>();
    let blockStateSpace =
        2 * (core::mem::size_of::<[u32; 3]>() + core::mem::size_of::<ZSTD_entropyCTables_t>());
    let useRowMatchFinder = crate::compress::match_state::ZSTD_resolveRowMatchFinderMode(
        params.useRowMatchFinder,
        cParams,
    );
    let matchStateSize = ZSTD_sizeof_matchState(cParams, useRowMatchFinder, false, true);
    let bufferSpace = buffInSize + buffOutSize;
    let cctxSpace = if isStatic {
        core::mem::size_of::<ZSTD_CCtx>()
    } else {
        0
    };
    let externalSeqSpace = if ZSTD_hasExtSeqProd(params) {
        ZSTD_sequenceBound(blockSize) * core::mem::size_of::<ZSTD_Sequence>()
    } else {
        0
    };
    cctxSpace + blockStateSpace + matchStateSize + tokenSpace + bufferSpace + externalSeqSpace
}

pub(crate) const ZSTD_COMPRESS_SEQUENCES_WORKSPACE_SIZE: usize =
    core::mem::size_of::<u32>() * (MaxSeq as usize + 2);
pub(crate) const ZSTD_ENTROPY_WORKSPACE_SIZE: usize =
    crate::compress::huf_compress::HUF_WORKSPACE_SIZE + ZSTD_COMPRESS_SEQUENCES_WORKSPACE_SIZE;
const ZSTD_SEQ_ENCODE_WORKSPACE_SIZE: usize = 16 * 1024;
const ZSTD_SLIPBLOCK_WORKSPACESIZE: usize = 8208;
const ZSTD_TMP_WORKSPACE_SIZE: usize = if ZSTD_ENTROPY_WORKSPACE_SIZE > ZSTD_SLIPBLOCK_WORKSPACESIZE
{
    ZSTD_ENTROPY_WORKSPACE_SIZE
} else {
    ZSTD_SLIPBLOCK_WORKSPACESIZE
};

#[derive(Debug, Clone)]
pub struct ZSTD_entropyScratch {
    pub blockWorkspaceU32: Vec<u32>,
    pub entropyWorkspace: Vec<u8>,
    pub hufCount: Vec<u32>,
    pub hufHistWorkspace: Vec<u32>,
    pub hufBuildWorkspace: Vec<u32>,
    pub hufWriteWorkspace: Vec<u8>,
    pub seqCountWorkspace: Vec<u32>,
    pub seqEntropyWorkspace: Vec<u8>,
    pub seqHistWorkspace: Vec<u32>,
}

impl Default for ZSTD_entropyScratch {
    fn default() -> Self {
        Self {
            blockWorkspaceU32: vec![
                0u32;
                crate::compress::hist::HIST_WKSP_SIZE_U32
                    .max(MaxSeq as usize + 1)
            ],
            entropyWorkspace: vec![0u8; ZSTD_ENTROPY_WORKSPACE_SIZE],
            hufCount: vec![0u32; 256],
            hufHistWorkspace: vec![0u32; crate::compress::hist::HIST_WKSP_SIZE_U32],
            hufBuildWorkspace: vec![
                0u32;
                crate::compress::huf_compress::HUF_CTABLE_WORKSPACE_SIZE_U32
            ],
            hufWriteWorkspace: vec![
                0u8;
                crate::compress::huf_compress::HUF_WRITE_CTABLE_WORKSPACE_SIZE
            ],
            seqCountWorkspace: vec![
                0u32;
                crate::compress::hist::HIST_WKSP_SIZE_U32
                    .max(MaxSeq as usize + 1)
            ],
            seqEntropyWorkspace: vec![0u8; ZSTD_SEQ_ENCODE_WORKSPACE_SIZE],
            seqHistWorkspace: vec![0u32; 1024],
        }
    }
}

#[inline]
fn ZSTD_static_compressedBlockState_size() -> usize {
    fn align_up(size: usize, align: usize) -> usize {
        (size + align - 1) & !(align - 1)
    }

    let huf_size = align_up(
        257 * core::mem::size_of::<HUF_CElt>() + core::mem::size_of::<u32>(),
        core::mem::align_of::<usize>(),
    );
    let fse_size = (FSE_CTABLE_SIZE_U32(OffFSELog, MaxOff)
        + FSE_CTABLE_SIZE_U32(MLFSELog, MaxML)
        + FSE_CTABLE_SIZE_U32(LLFSELog, MaxLL))
        * core::mem::size_of::<FSE_CTable>()
        + 3 * core::mem::size_of::<u32>();
    align_up(
        huf_size + fse_size + 3 * core::mem::size_of::<u32>(),
        core::mem::align_of::<usize>(),
    )
}

#[inline]
fn ZSTD_minStaticCCtxSize() -> usize {
    core::mem::size_of::<ZSTD_CCtx>()
        + ZSTD_TMP_WORKSPACE_SIZE
        + 2 * ZSTD_static_compressedBlockState_size()
}

/// Port of `ZSTD_estimateCCtxSize_usingCParams` (`zstd_compress.c:1789`).
/// Estimates CCtx footprint for one-shot compression using the supplied
/// cParams. Wraps `ZSTD_estimateCCtxSize_usingCCtxParams_internal` with
/// defaults appropriate for non-streaming use.
pub fn ZSTD_estimateCCtxSize_usingCParams(
    cParams: crate::compress::match_state::ZSTD_compressionParameters,
) -> usize {
    let params = ZSTD_makeCCtxParamsFromCParams(cParams);
    ZSTD_estimateCCtxSize_usingCCtxParams_internal(
        &cParams,
        &params,
        true,
        0,
        0,
        crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN,
    )
}

/// Port of `ZSTD_estimateCCtxSize_internal` (`zstd_compress.c:1806`).
/// Picks the max footprint across the 4 upstream srcSize tiers
/// (16KB / 128KB / 256KB / UNKNOWN) for a single compressionLevel.
fn ZSTD_estimateCCtxSize_internal(compressionLevel: i32) -> usize {
    let tiers = [16u64 * 1024, 128 * 1024, 256 * 1024, u64::MAX];
    let mut largest = 0usize;
    for tier in &tiers {
        let cp = ZSTD_getCParams(compressionLevel, *tier, 0);
        let sz = ZSTD_estimateCCtxSize_usingCParams(cp);
        if sz > largest {
            largest = sz;
        }
    }
    largest
}

/// Port of `ZSTD_estimateCCtxSize` (`zstd_compress.c:1819`). Sweeps
/// levels from `min(compressionLevel, 1)` up to `compressionLevel`
/// and takes the max — matching upstream's monotonicity guarantee
/// (higher level never reports a smaller footprint than any lower
/// level).
pub fn ZSTD_estimateCCtxSize(compressionLevel: i32) -> usize {
    let start = compressionLevel.min(1);
    let mut memBudget = 0usize;
    for level in start..=compressionLevel {
        let newMB = ZSTD_estimateCCtxSize_internal(level);
        if newMB > memBudget {
            memBudget = newMB;
        }
    }
    memBudget
}

/// Port of `ZSTD_estimateCStreamSize_usingCParams`. Roughly the same
/// as CCtx size + 2×ZSTD_BLOCKSIZE_MAX for the streaming in/out
/// buffers (upstream accounts for them explicitly).
pub fn ZSTD_estimateCStreamSize_usingCParams(
    cParams: crate::compress::match_state::ZSTD_compressionParameters,
) -> usize {
    use crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX;
    ZSTD_estimateCCtxSize_usingCParams(cParams) + 2 * ZSTD_BLOCKSIZE_MAX
}

/// Port of `ZSTD_estimateCStreamSize_internal` (`zstd_compress.c:1872`).
/// Single-level estimate: CParams at UNKNOWN srcSize →
/// `ZSTD_estimateCStreamSize_usingCParams`.
fn ZSTD_estimateCStreamSize_internal(compressionLevel: i32) -> usize {
    let cp = ZSTD_getCParams(compressionLevel, u64::MAX, 0);
    ZSTD_estimateCStreamSize_usingCParams(cp)
}

/// Port of `ZSTD_estimateCStreamSize` (`zstd_compress.c:1878`). Sweeps
/// levels from `min(compressionLevel, 1)` up to `compressionLevel`
/// and takes the max — matches upstream's monotonicity guarantee.
pub fn ZSTD_estimateCStreamSize(compressionLevel: i32) -> usize {
    let start = compressionLevel.min(1);
    let mut memBudget = 0usize;
    for level in start..=compressionLevel {
        let newMB = ZSTD_estimateCStreamSize_internal(level);
        if newMB > memBudget {
            memBudget = newMB;
        }
    }
    memBudget
}

/// Port of `ZSTD_estimateCCtxSize_usingCCtxParams`. v0.1 doesn't
/// track cwksp rounding / dedicated-dict-space exactly, but this keeps
/// the upstream control-flow shape: resolve effective cParams from the
/// params struct, then size the match state / token buffers against the
/// resolved block size.
pub fn ZSTD_estimateCCtxSize_usingCCtxParams(params: &ZSTD_CCtx_params) -> usize {
    use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;

    let cParams = ZSTD_getCParamsFromCCtxParams(
        params,
        ZSTD_CONTENTSIZE_UNKNOWN,
        0,
        ZSTD_CParamMode_e::ZSTD_cpm_noAttachDict,
    );
    ZSTD_estimateCCtxSize_usingCCtxParams_internal(
        &cParams,
        params,
        true,
        0,
        0,
        ZSTD_CONTENTSIZE_UNKNOWN,
    )
}

/// Port of `ZSTD_estimateCStreamSize_usingCCtxParams`. Resolves the
/// effective cParams from the params bundle, derives the working block
/// size, then adds streaming input/output buffers on top of the CCtx
/// core estimate.
pub fn ZSTD_estimateCStreamSize_usingCCtxParams(params: &ZSTD_CCtx_params) -> usize {
    use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;

    let cParams = ZSTD_getCParamsFromCCtxParams(
        params,
        ZSTD_CONTENTSIZE_UNKNOWN,
        0,
        ZSTD_CParamMode_e::ZSTD_cpm_noAttachDict,
    );
    let blockSize = crate::compress::match_state::ZSTD_resolveMaxBlockSize(params.maxBlockSize)
        .min(1usize << cParams.windowLog);
    let inBuffSize = (1usize << cParams.windowLog) + blockSize;
    let outBuffSize = ZSTD_compressBound(blockSize) + 1;
    ZSTD_estimateCCtxSize_usingCCtxParams_internal(
        &cParams,
        params,
        true,
        inBuffSize,
        outBuffSize,
        ZSTD_CONTENTSIZE_UNKNOWN,
    )
}

std::thread_local! {
    static ZSTD_COMPRESS_CCTX: std::cell::RefCell<Option<Box<ZSTD_CCtx>>> =
        const { std::cell::RefCell::new(None) };
}

/// Port of upstream's one-shot `ZSTD_compress` (`zstd.h:160`).
/// Compresses `src` at the given level into `dst` and returns the
/// compressed byte count (or an `ErrorCode`-bearing return value if
/// `ZSTD_isError`). No dctx, no streaming, no dict — the simplest
/// entry point. For context-managed, streaming, or dict-bearing
/// flows use `ZSTD_compressCCtx` / `ZSTD_compressStream2` /
/// `ZSTD_compress_usingDict` instead.
///
/// The frame emitted declares the content size in the header
/// (`contentSizeFlag = 1`) and suppresses both the dictID field
/// (`noDictIDFlag = 1`) and the XXH64 trailer (`checksumFlag = 0`).
/// Level is clamped into `[ZSTD_minCLevel, ZSTD_maxCLevel]`; higher
/// strategies route through the current `ZSTD_selectBlockCompressor()`
/// table, including the shared optimal-parser entries.
pub fn ZSTD_compress(dst: &mut [u8], src: &[u8], compressionLevel: i32) -> usize {
    ZSTD_COMPRESS_CCTX.with(|slot| {
        let mut cctx_slot = slot.borrow_mut();
        if cctx_slot.is_none() {
            *cctx_slot = ZSTD_createCCtx();
            if cctx_slot.is_none() {
                return ERROR(ErrorCode::MemoryAllocation);
            }
        }
        ZSTD_compressCCtx(
            cctx_slot.as_deref_mut().expect("checked initialized"),
            dst,
            src,
            compressionLevel,
        )
    })
}

/// Port of `ZSTD_compressCCtx`. Context-managed one-shot compression
/// at the given level. Reuses the context's seqStore + entropy tables
/// across calls. Match state is re-seeded per call (no cross-call
/// history yet — upstream has the same default for `ZSTD_compressCCtx`
/// unless the caller sets pledgedSrcSize + uses the streaming API).
pub fn ZSTD_compressCCtx(
    cctx: &mut ZSTD_CCtx,
    dst: &mut [u8],
    src: &[u8],
    compressionLevel: i32,
) -> usize {
    ZSTD_compress_usingDict(cctx, dst, src, &[], compressionLevel)
}

/// Port of `ZSTD_CDict`. Pre-digested compression dictionary. In
/// upstream, this caches a `ZSTD_MatchState_t` plus entropy / repcode
/// seeds so successive `ZSTD_compress_usingCDict` calls can attach or
/// copy the prebuilt dictionary state instead of rescanning the input
/// bytes from scratch.
#[derive(Debug, Clone)]
pub struct ZSTD_CDict {
    pub dictContent: Vec<u8>,
    pub compressionLevel: i32,
    /// Parsed dictID (0 for raw-content dicts / any dict without the
    /// `ZSTD_MAGIC_DICTIONARY` prefix).
    pub dictID: u32,
    /// cParams derived from level + dictSize at creation time.
    /// Mirrors upstream's `cdict->matchState.cParams` — exposed via
    /// `ZSTD_getCParamsFromCDict` so callers can inspect the CDict's
    /// strategy without re-deriving from the level.
    pub cParams: crate::compress::match_state::ZSTD_compressionParameters,
    /// Resolved row-matchfinder mode captured at CDict creation.
    pub useRowMatchFinder: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
    /// Cached entropy tables parsed from a zstd-format dictionary.
    pub entropy: ZSTD_entropyCTables_t,
    /// Cached repcode history parsed from a zstd-format dictionary.
    pub rep: [u32; 3],
    /// Upstream `matchState.dedicatedDictSearch`.
    pub dedicatedDictSearch: i32,
    /// Cached match-state tables seeded from the dictionary content.
    /// These back both the attach-CDict path (`dictMatchState`) and
    /// the copy-CDict path that clones pre-filled tables into the
    /// working CCtx.
    pub matchState: crate::compress::match_state::ZSTD_MatchState_t,
    /// Upstream custom allocator bundle requested for this CDict.
    pub customMem: ZSTD_customMem,
}

/// Port of `ZSTD_createCDict`. By-copy dict load — stores a clone.
/// Routes through the real advanced-create path so the CDict's match
/// state and entropy tables are initialized like upstream.
pub fn ZSTD_createCDict(dict: &[u8], compressionLevel: i32) -> Option<Box<ZSTD_CDict>> {
    use crate::decompress::zstd_ddict::{ZSTD_dictContentType_e, ZSTD_dictLoadMethod_e};
    use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;
    // Upstream `ZSTD_createCDict_advanced` routes through the
    // level-clamp / default-mapping before deriving cParams. Match
    // here so callers passing 0 (= default) or out-of-range values
    // get the same cParams as `ZSTD_compress` at the equivalent
    // level.
    let mut clamped = compressionLevel;
    let _ = ZSTD_cParam_clampBounds(ZSTD_cParameter::ZSTD_c_compressionLevel, &mut clamped);
    let effective_level = if clamped == 0 {
        ZSTD_CLEVEL_DEFAULT
    } else {
        clamped
    };
    let params = ZSTD_getParams_internal(
        effective_level,
        ZSTD_CONTENTSIZE_UNKNOWN,
        dict.len(),
        ZSTD_CParamMode_e::ZSTD_cpm_createCDict,
    );
    let mut cctxParams = ZSTD_CCtx_params::default();
    ZSTD_CCtxParams_init_internal(&mut cctxParams, &params, effective_level);
    ZSTD_createCDict_advanced2(
        dict,
        ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy,
        ZSTD_dictContentType_e::ZSTD_dct_auto,
        &cctxParams,
    )
}

/// Port of `ZSTD_getDictID_fromCDict`. Reads the dictID parsed at
/// CDict creation time — 0 for raw-content dicts.
#[inline]
pub fn ZSTD_getDictID_fromCDict(cdict: &ZSTD_CDict) -> u32 {
    cdict.dictID
}

/// Port of `ZSTD_createCDict_byReference`. By-reference load — Rust
/// can't store a borrow without a lifetime parameter on `ZSTD_CDict`,
/// so this still owns the bytes. It does, however, route through the
/// real advanced-create path with `ZSTD_dlm_byRef`, so sizing and
/// parameter selection follow the same upstream branch.
pub fn ZSTD_createCDict_byReference(dict: &[u8], compressionLevel: i32) -> Option<Box<ZSTD_CDict>> {
    use crate::decompress::zstd_ddict::{ZSTD_dictContentType_e, ZSTD_dictLoadMethod_e};
    use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;
    let mut clamped = compressionLevel;
    let _ = ZSTD_cParam_clampBounds(ZSTD_cParameter::ZSTD_c_compressionLevel, &mut clamped);
    let effective_level = if clamped == 0 {
        ZSTD_CLEVEL_DEFAULT
    } else {
        clamped
    };
    let params = ZSTD_getParams_internal(
        effective_level,
        ZSTD_CONTENTSIZE_UNKNOWN,
        dict.len(),
        ZSTD_CParamMode_e::ZSTD_cpm_createCDict,
    );
    let mut cctxParams = ZSTD_CCtx_params::default();
    ZSTD_CCtxParams_init_internal(&mut cctxParams, &params, effective_level);
    ZSTD_createCDict_advanced2(
        dict,
        ZSTD_dictLoadMethod_e::ZSTD_dlm_byRef,
        ZSTD_dictContentType_e::ZSTD_dct_auto,
        &cctxParams,
    )
}

/// Port of `ZSTD_createCDict_advanced`. Accepts explicit
/// `dictLoadMethod`, `dictContentType`, and `cParams` knobs.
pub fn ZSTD_createCDict_advanced(
    dict: &[u8],
    dictLoadMethod: crate::decompress::zstd_ddict::ZSTD_dictLoadMethod_e,
    dictContentType: crate::decompress::zstd_ddict::ZSTD_dictContentType_e,
    cParams: crate::compress::match_state::ZSTD_compressionParameters,
) -> Option<Box<ZSTD_CDict>> {
    let mut params = ZSTD_CCtx_params::default();
    ZSTD_CCtxParams_init(&mut params, ZSTD_NO_CLEVEL);
    params.cParams = cParams;
    ZSTD_createCDict_advanced2(dict, dictLoadMethod, dictContentType, &params)
}

/// Upstream `ZSTD_USE_CDICT_PARAMS_SRCSIZE_CUTOFF` (`zstd_compress.c:5256`).
/// Threshold below which `ZSTD_compressBegin_usingCDict_*` uses the
/// CDict's cParams verbatim instead of re-deriving from level+size.
pub const ZSTD_USE_CDICT_PARAMS_SRCSIZE_CUTOFF: u64 = 128 * 1024;

/// Upstream `ZSTD_USE_CDICT_PARAMS_DICTSIZE_MULTIPLIER` (`zstd_compress.c:5257`).
/// Keeps CDict's cParams when `pledgedSrcSize < dictSize * MULTIPLIER`.
pub const ZSTD_USE_CDICT_PARAMS_DICTSIZE_MULTIPLIER: u64 = 6;

/// Upstream `attachDictSizeCutoffs` (`zstd_compress.c:2320`).
/// Per-strategy cutoff: below this source size, upstream prefers to
/// attach the CDict's match state in place; above, it prefers to
/// copy. Indexed by strategy 1..=9 (index 0 is unused).
pub const attachDictSizeCutoffs: [usize; 10] = [
    8 * 1024,  // unused (strategy = 0)
    8 * 1024,  // ZSTD_fast
    16 * 1024, // ZSTD_dfast
    32 * 1024, // ZSTD_greedy
    32 * 1024, // ZSTD_lazy
    32 * 1024, // ZSTD_lazy2
    32 * 1024, // ZSTD_btlazy2
    32 * 1024, // ZSTD_btopt
    8 * 1024,  // ZSTD_btultra
    8 * 1024,  // ZSTD_btultra2
];

/// Port of `ZSTD_getCParamMode` (`zstd_compress.c:5983`). Returns
/// `cpm_attachDict` when a CDict is present and
/// `ZSTD_shouldAttachDict` would attach it; otherwise `cpm_noAttachDict`.
/// Used by `ZSTD_compressBegin_*` to pick the right cParam-derivation
/// table row.
pub fn ZSTD_getCParamMode(
    cdict: Option<&ZSTD_CDict>,
    params: &ZSTD_CCtx_params,
    pledgedSrcSize: u64,
) -> ZSTD_CParamMode_e {
    match cdict {
        Some(c) if ZSTD_shouldAttachDict(c, params, pledgedSrcSize) => {
            ZSTD_CParamMode_e::ZSTD_cpm_attachDict
        }
        _ => ZSTD_CParamMode_e::ZSTD_cpm_noAttachDict,
    }
}

/// Port of `ZSTD_compressionStage_e` (`zstd_compress_internal.h:46`).
/// Lifecycle stage of a CCtx: `created` → `init` (post-begin) →
/// `ongoing` (after first block) → `ending` (epilogue pending).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZSTD_compressionStage_e {
    #[default]
    ZSTDcs_created = 0,
    ZSTDcs_init = 1,
    ZSTDcs_ongoing = 2,
    ZSTDcs_ending = 3,
}

/// Port of `ZSTD_cStreamStage` (`zstd_compress_internal.h:47`). CStream
/// ingest sub-stage: `init` → `load` (consuming caller input) →
/// `flush` (draining output to caller).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZSTD_cStreamStage {
    #[default]
    zcss_init = 0,
    zcss_load = 1,
    zcss_flush = 2,
}

/// Port of `ZSTD_buffered_policy_e` (`zstd_compress_internal.h`). Tells
/// `ZSTD_resetCCtx_internal` whether to allocate the streaming
/// in/out buffers or skip them (one-shot compression).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZSTD_buffered_policy_e {
    #[default]
    ZSTDb_not_buffered = 0,
    ZSTDb_buffered = 1,
}

/// Port of upstream `ZSTD_bufferMode_e` (`zstd_internal.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZSTD_bufferMode_e {
    #[default]
    ZSTD_bm_buffered = 0,
    ZSTD_bm_stable = 1,
}

/// Port of `ZSTD_copyBlockSequences` (`zstd_compress.c:3453`). After
/// the match finder has populated `seqStore`, this function translates
/// the internal `SeqDef[]` representation (offBase + optional long-
/// length flag) into caller-visible `ZSTD_Sequence[]` entries and
/// appends them to `outSeqs[seqIndex..]`. Also appends a final
/// `{ll=lastLiterals, ml=0, off=0, rep=0}` block-delimiter entry.
///
/// `prevRepcodes` is the repcode history *before* this block; it's
/// walked forward as each sequence is emitted to resolve
/// repcode-flavored offBases to raw offsets.
///
/// Returns `0` on success or a zstd error when `outSeqs` is too small.
pub fn ZSTD_copyBlockSequences(
    seqCollector: &mut SeqCollector,
    outSeqs: &mut [ZSTD_Sequence],
    seqStore: &SeqStore_t,
    prevRepcodes: &[u32; crate::compress::seq_store::ZSTD_REP_NUM],
) -> usize {
    use crate::common::error::{ErrorCode, ERROR};
    use crate::compress::seq_store::{
        Repcodes_t, ZSTD_longLengthType_e, ZSTD_updateRep, MINMATCH, OFFBASE_IS_REPCODE,
        OFFBASE_TO_OFFSET, OFFBASE_TO_REPCODE,
    };
    let nbInSequences = seqStore.sequences.len();
    let nbInLiterals = seqStore.literals.len();
    let nbOutSequences = nbInSequences + 1;
    let dst_start = seqCollector.seqIndex;
    if nbOutSequences > seqCollector.maxSequences.saturating_sub(dst_start) {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }

    let mut repcodes = Repcodes_t { rep: *prevRepcodes };
    let mut nbOutLiterals: u32 = 0;
    for i in 0..nbInSequences {
        let inSeq = seqStore.sequences[i];
        let out = &mut outSeqs[dst_start + i];
        out.litLength = inSeq.litLength as u32;
        out.matchLength = (inSeq.mlBase as u32).wrapping_add(MINMATCH);
        out.rep = 0;

        if i as u32 == seqStore.longLengthPos {
            match seqStore.longLengthType {
                ZSTD_longLengthType_e::ZSTD_llt_literalLength => out.litLength += 0x10000,
                ZSTD_longLengthType_e::ZSTD_llt_matchLength => out.matchLength += 0x10000,
                ZSTD_longLengthType_e::ZSTD_llt_none => {}
            }
        }

        let rawOffset = if OFFBASE_IS_REPCODE(inSeq.offBase) {
            let repcode = OFFBASE_TO_REPCODE(inSeq.offBase);
            out.rep = repcode;
            if out.litLength != 0 {
                repcodes.rep[(repcode - 1) as usize]
            } else if repcode == 3 {
                repcodes.rep[0] - 1
            } else {
                repcodes.rep[repcode as usize]
            }
        } else {
            OFFBASE_TO_OFFSET(inSeq.offBase)
        };
        out.offset = rawOffset;

        ZSTD_updateRep(
            &mut repcodes.rep,
            inSeq.offBase,
            (inSeq.litLength == 0) as u32,
        );

        nbOutLiterals += out.litLength;
    }

    // Block delimiter: trailing literals with ml=0, off=0.
    let lastLLSize = (nbInLiterals as u32).saturating_sub(nbOutLiterals);
    let tail = &mut outSeqs[dst_start + nbInSequences];
    tail.litLength = lastLLSize;
    tail.matchLength = 0;
    tail.offset = 0;
    tail.rep = 0;

    seqCollector.seqIndex += nbOutSequences;
    0
}

/// Rust-only helper: mirrors `ZSTD_copyBlockSequences` for the
/// superblock/targetCBlockSize path, which subdivides a single block's
/// `seqStore` into multiple sub-block delimiter entries.
fn ZSTD_copySuperBlockSequences(
    seqCollector: &mut SeqCollector,
    outSeqs: &mut [ZSTD_Sequence],
    seqStore: &SeqStore_t,
    prevRepcodes: &[u32; crate::compress::seq_store::ZSTD_REP_NUM],
    prevEntropy: &ZSTD_entropyCTables_t,
    nextEntropy: &mut ZSTD_entropyCTables_t,
    appliedParams: &ZSTD_CCtx_params,
    scratch: &mut ZSTD_entropyScratch,
) -> usize {
    use crate::compress::seq_store::{ZSTD_deriveSeqStoreChunk, ZSTD_updateRep};
    use crate::compress::zstd_compress_superblock::{
        sizeBlockSequences, ZSTD_estimateSubBlockSize,
    };
    const BYTESCALE: usize = 256;

    let nbSeqs = seqStore.sequences.len();
    if nbSeqs == 0 {
        return ZSTD_copyBlockSequences(seqCollector, outSeqs, seqStore, prevRepcodes);
    }

    let mut entropyMetadata = ZSTD_entropyCTablesMetadata_t::default();
    let rc = ZSTD_buildBlockEntropyStats(
        &mut ZSTD_deriveSeqStoreChunk(seqStore, 0, nbSeqs),
        prevEntropy,
        nextEntropy,
        appliedParams,
        &mut entropyMetadata,
        scratch,
    );
    if ERR_isError(rc) {
        return rc;
    }

    let mut currentRepcodes = *prevRepcodes;
    let mut sp = 0usize;
    let minTarget = ZSTD_TARGETCBLOCKSIZE_MIN as usize;
    let targetCBlockSize = minTarget.max(appliedParams.targetCBlockSize);
    let writeLitEntropy = (entropyMetadata.hufMetadata.hType
        == crate::decompress::zstd_decompress_block::SymbolEncodingType_e::set_compressed)
        as i32;
    let writeSeqEntropy = 1i32;

    let ebs = ZSTD_estimateSubBlockSize(
        &seqStore.literals,
        &seqStore.ofCode,
        &seqStore.llCode,
        &seqStore.mlCode,
        nbSeqs,
        nextEntropy,
        &entropyMetadata,
        &mut scratch.hufCount,
        &mut scratch.seqCountWorkspace,
        &mut scratch.blockWorkspaceU32,
        writeLitEntropy != 0,
        writeSeqEntropy != 0,
    );
    let avgLitCost = if !seqStore.literals.is_empty() {
        (ebs.estLitSize * BYTESCALE) / seqStore.literals.len()
    } else {
        BYTESCALE
    };
    let avgSeqCost = ((ebs.estBlockSize - ebs.estLitSize) * BYTESCALE) / nbSeqs;
    let nbSubBlocks = ((ebs.estBlockSize + (targetCBlockSize / 2)) / targetCBlockSize).max(1);
    let avgBlockBudget = (ebs.estBlockSize * BYTESCALE) / nbSubBlocks;

    for n in 0..(nbSubBlocks.saturating_sub(1)) {
        let seqCount = sizeBlockSequences(
            &seqStore.sequences[sp..],
            seqStore.sequences.len() - sp,
            avgBlockBudget,
            avgLitCost,
            avgSeqCost,
            (n == 0) as i32,
        );
        if sp + seqCount == seqStore.sequences.len() || seqCount == 0 {
            break;
        }

        let chunk = ZSTD_deriveSeqStoreChunk(seqStore, sp, sp + seqCount);
        let rc = ZSTD_copyBlockSequences(seqCollector, outSeqs, &chunk, &currentRepcodes);
        if ERR_isError(rc) {
            return rc;
        }

        for seqIdx in sp..sp + seqCount {
            let ll0 = (seqStore.sequences[seqIdx].litLength == 0) as u32;
            ZSTD_updateRep(
                &mut currentRepcodes,
                seqStore.sequences[seqIdx].offBase,
                ll0,
            );
        }
        sp += seqCount;
    }

    let chunk = ZSTD_deriveSeqStoreChunk(seqStore, sp, seqStore.sequences.len());
    ZSTD_copyBlockSequences(seqCollector, outSeqs, &chunk, &currentRepcodes)
}

/// Port of `ZSTD_inBuffer` (`zstd.h:701`). Public-API cursor struct
/// used by the streaming compression / decompression APIs — the
/// caller hands the compressor a buffer and the compressor advances
/// `pos` to reflect how much it consumed.
///
/// Rust-native streaming ports (`ZSTD_compressStream` etc.) accept
/// slices directly, but this struct is exposed so callers porting
/// C code can keep their idiom.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZSTD_inBuffer<'a> {
    pub src: Option<&'a [u8]>,
    pub size: usize,
    pub pos: usize,
}

/// Port of `ZSTD_outBuffer` (`zstd.h:707`). Symmetric with
/// `ZSTD_inBuffer` for the output side.
#[derive(Debug, Default)]
pub struct ZSTD_outBuffer<'a> {
    pub dst: Option<&'a mut [u8]>,
    pub size: usize,
    pub pos: usize,
}

/// Rust-port equivalent of upstream `ZSTD_sequenceProducer_F`
/// (`zstd.h:2931`). The callback emits a block-local parse into
/// `outSeqs` and returns the number of populated entries.
///
/// `sequenceProducerState` remains the upstream-style opaque user
/// pointer slot, represented here as a `usize`.
pub type ZSTD_sequenceProducer_F = fn(
    sequenceProducerState: usize,
    outSeqs: &mut [ZSTD_Sequence],
    src: &[u8],
    dict: &[u8],
    compressionLevel: i32,
    windowSize: usize,
) -> usize;

/// Port of `ZSTD_postProcessSequenceProducerResult` (`zstd_compress.c:3198`).
/// Validates sequences returned by an external sequence producer
/// (`ZSTD_registerSequenceProducer` API) and appends a block-delimiter
/// sentinel if the caller didn't already emit one. Returns the new
/// sequence count, or a zstd error on invalid input.
///
/// Contract:
///   - `nbExternalSeqs == 0 && srcSize > 0` ⇒ producer failed.
///   - `nbExternalSeqs > outSeqs.len()` ⇒ producer overshot capacity.
///   - `srcSize == 0` ⇒ write a single zero-delimiter, return 1.
///   - Otherwise append `{0, 0, 0, 0}` if the tail sequence isn't
///     already a delimiter.
pub fn ZSTD_postProcessSequenceProducerResult(
    outSeqs: &mut [ZSTD_Sequence],
    nbExternalSeqs: usize,
    srcSize: usize,
) -> usize {
    use crate::common::error::{ErrorCode, ERROR};
    let outSeqsCapacity = outSeqs.len();
    if nbExternalSeqs > outSeqsCapacity {
        return ERROR(ErrorCode::Generic);
    }
    if nbExternalSeqs == 0 && srcSize > 0 {
        return ERROR(ErrorCode::Generic);
    }
    if srcSize == 0 {
        if outSeqsCapacity == 0 {
            return ERROR(ErrorCode::DstSizeTooSmall);
        }
        outSeqs[0] = ZSTD_Sequence::default();
        return 1;
    }
    let lastSeq = outSeqs[nbExternalSeqs - 1];
    if lastSeq.offset == 0 && lastSeq.matchLength == 0 {
        return nbExternalSeqs;
    }
    if nbExternalSeqs == outSeqsCapacity {
        return ERROR(ErrorCode::Generic);
    }
    outSeqs[nbExternalSeqs] = ZSTD_Sequence::default();
    nbExternalSeqs + 1
}

/// Port of `SeqCollector` (`zstd_compress_internal.h:355`). Toggles a
/// shadow-collection mode on the compressor: when `collectSequences`
/// is set, the block-compress path emits discovered sequences into
/// the caller's `outSeqs` buffer instead of (or in addition to)
/// serializing them. Used by `ZSTD_generateSequences`.
///
/// Rust-port note: upstream stores `ZSTD_Sequence*` directly — our
/// port defers the slice until the call site via an index range into
/// a separate `outSeqs` buffer the caller owns.
#[derive(Debug, Clone, Copy, Default)]
pub struct SeqCollector {
    /// Non-zero when `ZSTD_compressBlock_internal` should divert
    /// match-finder output into the caller's `outSeqs` slice.
    pub collectSequences: i32,
    /// How many sequences have been written so far.
    pub seqIndex: usize,
    /// Max count the caller's buffer can hold.
    pub maxSequences: usize,
}

/// Port of `ZSTD_SequencePosition` (`zstd_compress_internal.h:1517`).
/// Cursor tracking for caller-supplied `ZSTD_Sequence` arrays —
/// `(idx, posInSequence, posInSrc)` identifies where the next block
/// boundary starts when transferring sequences across block splits.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZSTD_SequencePosition {
    /// Index in the caller's `ZSTD_Sequence[]`.
    pub idx: u32,
    /// Byte position within sequence `idx` — nonzero when a sequence
    /// straddles a block boundary and was partly consumed by the
    /// previous block.
    pub posInSequence: u32,
    /// Running count of source bytes consumed by sequences so far.
    pub posInSrc: usize,
}

/// Port of `blockSize_explicitDelimiter` (`zstd_compress.c:6920`).
/// Walks a caller-supplied `ZSTD_Sequence[]` starting at `seqPos.idx`
/// summing litLength + matchLength until it hits a delimiter entry
/// (`offset == 0`). Returns the total uncompressed bytes the next
/// block will cover, or a zstd error if the sequence array ends
/// before a delimiter, or a delimiter carries a non-zero matchLength.
pub fn blockSize_explicitDelimiter(
    inSeqs: &[ZSTD_Sequence],
    seqPos: ZSTD_SequencePosition,
) -> usize {
    use crate::common::error::{ErrorCode, ERROR};
    let mut spos = seqPos.idx as usize;
    let mut blockSize: usize = 0;
    let mut end = false;
    while spos < inSeqs.len() {
        let seq = inSeqs[spos];
        end = seq.offset == 0;
        blockSize += seq.litLength as usize + seq.matchLength as usize;
        if end {
            if seq.matchLength != 0 {
                return ERROR(ErrorCode::ExternalSequencesInvalid);
            }
            break;
        }
        spos += 1;
    }
    if !end {
        return ERROR(ErrorCode::ExternalSequencesInvalid);
    }
    blockSize
}

/// Port of `determine_blockSize` (`zstd_compress.c:6942`). Dispatches
/// on `ZSTD_SequenceFormat_e`: for `noBlockDelimiters` returns
/// `min(remaining, blockSize)` (a "target" size), for
/// `explicitBlockDelimiters` returns the explicit-delimiter sum,
/// validating against `blockSize` and `remaining` upper bounds.
pub fn determine_blockSize(
    mode: ZSTD_SequenceFormat_e,
    blockSize: usize,
    remaining: usize,
    inSeqs: &[ZSTD_Sequence],
    seqPos: ZSTD_SequencePosition,
) -> usize {
    use crate::common::error::{ERR_isError, ErrorCode, ERROR};
    if mode == ZSTD_SequenceFormat_e::ZSTD_sf_noBlockDelimiters {
        return remaining.min(blockSize);
    }
    let explicit = blockSize_explicitDelimiter(inSeqs, seqPos);
    if ERR_isError(explicit) {
        return explicit;
    }
    if explicit > blockSize {
        return ERROR(ErrorCode::ExternalSequencesInvalid);
    }
    if explicit > remaining {
        return ERROR(ErrorCode::ExternalSequencesInvalid);
    }
    explicit
}

/// Port of `ZSTD_SequenceFormat_e` (`zstd.h:1581`). Tells the
/// caller-supplied-sequences path whether block boundaries are
/// explicit `{0,0,X,0}` delimiters or implicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZSTD_SequenceFormat_e {
    #[default]
    ZSTD_sf_noBlockDelimiters = 0,
    ZSTD_sf_explicitBlockDelimiters = 1,
}

/// Port of `ZSTD_sizeof_matchState` (`zstd_compress.c:1669`). Byte
/// count needed to hold the hash + chain + hashLog3 tables plus
/// optional opt-parser scratch when the strategy is `btopt`+. Used
/// by `ZSTD_estimateCDictSize_advanced` and
/// `ZSTD_estimateCCtxSize_*`.
///
/// Our port skips the cwksp alignment wrappers (our tables are
/// `Vec<u32>` and take only the raw element count × 4 bytes) and the
/// `slackSpace` reserve (an ASAN-redzone allowance), and keeps the
/// lazy row-tag table + opt-parser tables as their byte counts.
pub fn ZSTD_sizeof_matchState(
    cParams: &crate::compress::match_state::ZSTD_compressionParameters,
    useRowMatchFinder: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
    enableDedicatedDictSearch: bool,
    forCCtx: bool,
) -> usize {
    use crate::compress::match_state::{ZSTD_allocateChainTable, ZSTD_rowMatchFinderUsed};
    use crate::compress::zstd_compress_sequences::ZSTD_btopt;

    let chainSize = if ZSTD_allocateChainTable(
        cParams.strategy,
        useRowMatchFinder,
        (enableDedicatedDictSearch && !forCCtx) as u32,
    ) {
        1usize << cParams.chainLog
    } else {
        0
    };
    let hSize: usize = 1 << cParams.hashLog;
    let hashLog3 = if forCCtx && cParams.minMatch == 3 {
        ZSTD_HASHLOG3_MAX.min(cParams.windowLog)
    } else {
        0
    };
    let h3Size: usize = if hashLog3 != 0 { 1 << hashLog3 } else { 0 };
    let tableSpace = (chainSize + hSize + h3Size) * core::mem::size_of::<u32>();

    let optSpace = if forCCtx && cParams.strategy >= ZSTD_btopt {
        ((MaxML + 1) as usize
            + (MaxLL + 1) as usize
            + (MaxOff + 1) as usize
            + (1 << Litbits) as usize)
            * core::mem::size_of::<u32>()
            + ZSTD_OPT_SIZE * 16
            + ZSTD_OPT_SIZE * 16
    } else {
        0
    };

    let lazyAdditionalSpace = if ZSTD_rowMatchFinderUsed(cParams.strategy, useRowMatchFinder) {
        hSize
    } else {
        0
    };

    tableSpace + optSpace + lazyAdditionalSpace
}

/// Upstream `ZSTD_HASHLOG3_MAX` (`zstd_compress_internal.h`). Max
/// log size of the auxiliary 3-byte hash table used when `minMatch=3`.
pub const ZSTD_HASHLOG3_MAX: u32 = 17;

/// Re-export of `zstd_opt::ZSTD_OPT_SIZE` — the canonical
/// `ZSTD_OPT_NUM + 3` match-table capacity for the optimal parser's
/// forward-DP candidate buffer. Previously duplicated here as
/// `1 << 12`, dropping the +3 margin that upstream requires.
pub use crate::compress::zstd_opt::ZSTD_OPT_SIZE;

/// Upstream `Litbits`. Log2 of the literal alphabet size (256 symbols
/// = 8 bits).
pub const Litbits: u32 = 8;

/// Port of `ZSTD_compResetPolicy_e` (`zstd_compress.c:1974`). Tells
/// `ZSTD_reset_matchState` whether to scrub the hash/chain tables
/// (`makeClean`) or leave them with whatever stale entries the
/// previous frame left behind (`leaveDirty`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZSTD_compResetPolicy_e {
    #[default]
    ZSTDcrp_makeClean = 0,
    ZSTDcrp_leaveDirty = 1,
}

/// Port of `ZSTD_indexResetPolicy_e` (`zstd_compress.c:1984`).
/// Controls whether the window's `nextSrc` index continues from where
/// the previous block left off (`continue`) or resets to the sentinel
/// start (`reset`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZSTD_indexResetPolicy_e {
    #[default]
    ZSTDirp_continue = 0,
    ZSTDirp_reset = 1,
}

/// Port of `ZSTD_resetTarget_e` (`zstd_compress.c:1989`). Disambiguates
/// `ZSTD_reset_matchState`'s client — a CDict being constructed vs.
/// an active CCtx being re-initialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZSTD_resetTarget_e {
    ZSTD_resetTarget_CDict = 0,
    #[default]
    ZSTD_resetTarget_CCtx = 1,
}

/// Port of `ZSTD_reset_matchState` (`zstd_compress.c:2011`).
/// Re-sizes and clears the match-state tables for a fresh CCtx or
/// CDict initialization while preserving the Rust port's owned-`Vec`
/// storage model.
pub fn ZSTD_reset_matchState(
    ms: &mut crate::compress::match_state::ZSTD_MatchState_t,
    cParams: &crate::compress::match_state::ZSTD_compressionParameters,
    useRowMatchFinder: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
    crp: ZSTD_compResetPolicy_e,
    forceResetIndex: ZSTD_indexResetPolicy_e,
    forWho: ZSTD_resetTarget_e,
) -> usize {
    use crate::compress::match_state::{
        ZSTD_advanceHashSalt, ZSTD_allocateChainTable, ZSTD_invalidateMatchState,
        ZSTD_rowMatchFinderUsed, ZSTD_window_init,
    };

    let chainSize = if ZSTD_allocateChainTable(cParams.strategy, useRowMatchFinder, 0) {
        1usize << cParams.chainLog
    } else {
        0
    };
    let hashSize = 1usize << cParams.hashLog;
    let hashLog3 = if forWho == ZSTD_resetTarget_e::ZSTD_resetTarget_CCtx && cParams.minMatch == 3 {
        ZSTD_HASHLOG3_MAX.min(cParams.windowLog)
    } else {
        0
    };
    let hash3Size = if hashLog3 != 0 { 1usize << hashLog3 } else { 0 };
    let rowLog = cParams.searchLog.clamp(4, 6);
    let useRow = ZSTD_rowMatchFinderUsed(cParams.strategy, useRowMatchFinder);
    let rowHashLog = if useRow {
        cParams.hashLog.saturating_sub(rowLog)
    } else {
        0
    };
    let tagTableSize = if useRow { hashSize } else { 0 };

    if forceResetIndex == ZSTD_indexResetPolicy_e::ZSTDirp_reset {
        ZSTD_window_init(&mut ms.window);
    }

    ms.hashLog3 = hashLog3;
    ZSTD_invalidateMatchState(ms);

    ms.cParams = *cParams;
    ms.rowHashLog = rowHashLog;
    if ms.hashTable.len() != hashSize {
        ms.hashTable.resize(hashSize, 0);
    }
    if ms.chainTable.len() != chainSize {
        ms.chainTable.resize(chainSize, 0);
    }
    if ms.hashTable3.len() != hash3Size {
        ms.hashTable3.resize(hash3Size, 0);
    }
    if ms.tagTable.len() != tagTableSize {
        ms.tagTable.resize(tagTableSize, 0);
    }

    if crp != ZSTD_compResetPolicy_e::ZSTDcrp_leaveDirty
        || forceResetIndex == ZSTD_indexResetPolicy_e::ZSTDirp_reset
    {
        ms.hashTable.fill(0);
        ms.chainTable.fill(0);
        ms.hashTable3.fill(0);
        ms.tagTable.fill(0);
    }

    if useRow {
        if forWho == ZSTD_resetTarget_e::ZSTD_resetTarget_CCtx {
            ZSTD_advanceHashSalt(ms);
        } else {
            ms.hashSalt = 0;
        }
    } else {
        ms.hashSalt = 0;
    }
    0
}

/// Port of `ZSTD_initLocalDict` (`zstd_compress.c:1268`).
/// The current Rust CCtx doesn't retain a separate `localDict`
/// cache object, so the caller-visible equivalent is that any
/// already-loaded `stream_dict` is ready for reuse as-is.
#[inline]
pub fn ZSTD_initLocalDict(cctx: &mut ZSTD_CCtx) -> usize {
    if cctx.stream_dict.is_empty() {
        return 0;
    }

    if cctx.dictContentSize == cctx.stream_dict.len() {
        return 0;
    }

    cctx.dictContentSize = cctx.stream_dict.len();
    if let Some(ms) = cctx.ms.as_mut() {
        ms.dictMatchState = None;
        ms.dictContent = cctx.stream_dict.clone();
        ms.loadedDictEnd = cctx.stream_dict.len() as u32;
    }
    if cctx.prefix_is_single_use {
        cctx.prefix_is_single_use = false;
    }
    0
}

/// Port of `ZSTD_BuildSeqStore_e` (`zstd_compress.c:3286`). Reports
/// whether `ZSTD_buildSeqStore` wants the caller to emit a compressed
/// block (`ZSTDbss_compress`) or fall through to a raw/rle block
/// (`ZSTDbss_noCompress` — e.g. when the input is tiny or the match
/// finder found nothing worth emitting).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZSTD_BuildSeqStore_e {
    ZSTDbss_compress = 0,
    ZSTDbss_noCompress = 1,
}

enum ZSTD_buildSeqStore_select_e {
    Final(usize),
    LastLiterals(usize),
}

/// Rust-only helper: shared core for `ZSTD_buildSeqStore` /
/// `ZSTD_buildSeqStore_with_window`. Resolves the dict mode and row
/// matchfinder mode, runs LDM pre-pass when enabled, then invokes the
/// selected block compressor over `window_buf[src_pos..src_end]`.
/// Returns either a `Final` result (e.g. noCompress short-circuit) or
/// the trailing literal count to drop into the seq store.
fn ZSTD_buildSeqStore_selectMatches_with_window(
    cctx: &mut ZSTD_CCtx,
    window_buf: &[u8],
    src_pos: usize,
    src_end: usize,
) -> ZSTD_buildSeqStore_select_e {
    use crate::compress::match_state::{
        ZSTD_dictMode_e, ZSTD_matchState_dictMode, ZSTD_resolveRowMatchFinderMode,
    };
    use crate::compress::zstd_ldm::{
        ZSTD_ParamSwitch_e, ZSTD_ldm_blockCompress, ZSTD_ldm_generateSequences,
    };
    let src = &window_buf[src_pos..src_end];
    let window_to_block_end = &window_buf[..src_end];
    let ms = cctx.ms.as_mut().expect("match state must be initialized");
    let seqStore = cctx
        .seqStore
        .as_mut()
        .expect("seqStore must be initialized");
    let resolvedUseRowMatchFinder = ZSTD_resolveRowMatchFinderMode(
        cctx.appliedParams.useRowMatchFinder,
        &cctx.appliedParams.cParams,
    );

    cctx.next_rep = cctx.prev_rep;
    let lastLLSize = if let Some(rawSeqStore) = cctx.externalMatchStore.as_mut() {
        if ZSTD_hasExtSeqProd(&cctx.appliedParams) {
            return ZSTD_buildSeqStore_select_e::Final(ERROR(
                ErrorCode::ParameterCombinationUnsupported,
            ));
        }
        ZSTD_ldm_blockCompress(
            rawSeqStore,
            ms,
            seqStore,
            &mut cctx.next_rep,
            src,
            cctx.appliedParams.useRowMatchFinder,
        )
    } else if cctx.appliedParams.ldmEnable == ZSTD_ParamSwitch_e::ZSTD_ps_enable {
        if ZSTD_hasExtSeqProd(&cctx.appliedParams) {
            return ZSTD_buildSeqStore_select_e::Final(ERROR(
                ErrorCode::ParameterCombinationUnsupported,
            ));
        }
        let ldmParams = cctx.appliedParams.ldmParams;
        let ldmState = cctx
            .ldmState
            .as_mut()
            .expect("ldmState must be initialized when LDM is enabled");
        let rawSeqStore = &mut cctx.ldmSequences;
        rawSeqStore.size = 0;
        rawSeqStore.pos = 0;
        rawSeqStore.posInSequence = 0;
        let lowestIndex = ldmState.window.dictLimit;
        let _ = ZSTD_ldm_generateSequences(
            ldmState,
            rawSeqStore,
            &ldmParams,
            window_buf,
            src_pos,
            src_end,
            lowestIndex,
        );
        ZSTD_ldm_blockCompress(
            rawSeqStore,
            ms,
            seqStore,
            &mut cctx.next_rep,
            src,
            cctx.appliedParams.useRowMatchFinder,
        )
    } else if let Some(sequenceProducer) = cctx.appliedParams.extSeqProdFunc {
        ms.ldmSeqStore = None;
        let mut extSeqs = vec![ZSTD_Sequence::default(); ZSTD_sequenceBound(src.len()).max(1)];
        let windowSize = 1usize << cctx.appliedParams.cParams.windowLog;
        let nbExternalSeqs = sequenceProducer(
            cctx.appliedParams.extSeqProdState,
            &mut extSeqs,
            src,
            &[],
            cctx.appliedParams.compressionLevel,
            windowSize,
        );
        let nbPostProcessedSeqs =
            ZSTD_postProcessSequenceProducerResult(&mut extSeqs, nbExternalSeqs, src.len());
        if !ERR_isError(nbPostProcessedSeqs) {
            let extSeqs = &extSeqs[..nbPostProcessedSeqs];
            let mut seqPos = ZSTD_SequencePosition::default();
            let seqLenSum: usize = extSeqs
                .iter()
                .map(|seq| seq.litLength as usize + seq.matchLength as usize)
                .sum();
            if seqLenSum > src.len() {
                return ZSTD_buildSeqStore_select_e::Final(ERROR(
                    ErrorCode::ExternalSequencesInvalid,
                ));
            }
            let blockSize = ZSTD_transferSequences_wBlockDelim(
                cctx,
                &mut seqPos,
                extSeqs,
                src,
                src.len(),
                cctx.appliedParams.searchForExternalRepcodes,
            );
            if ERR_isError(blockSize) {
                return ZSTD_buildSeqStore_select_e::Final(blockSize);
            }
            return ZSTD_buildSeqStore_select_e::Final(
                ZSTD_BuildSeqStore_e::ZSTDbss_compress as usize,
            );
        }
        if cctx.appliedParams.enableMatchFinderFallback == 0 {
            return ZSTD_buildSeqStore_select_e::Final(nbPostProcessedSeqs);
        }
        ms.ldmSeqStore = None;
        let dictMode = ZSTD_matchState_dictMode(ms);
        let blockCompressor = ZSTD_selectBlockCompressor(
            cctx.appliedParams.cParams.strategy,
            resolvedUseRowMatchFinder,
            dictMode,
        );
        blockCompressor(ms, seqStore, &mut cctx.next_rep, src)
    } else {
        ms.ldmSeqStore = None;
        let dictMode = ZSTD_matchState_dictMode(ms);
        match (cctx.appliedParams.cParams.strategy, dictMode) {
            (s, ZSTD_dictMode_e::ZSTD_noDict)
                if s == crate::compress::zstd_compress_sequences::ZSTD_fast =>
            {
                crate::compress::zstd_fast::ZSTD_compressBlock_fast_noDict_generic(
                    ms,
                    seqStore,
                    &mut cctx.next_rep,
                    window_to_block_end,
                    src_pos,
                    ms.cParams.minMatch,
                )
            }
            (s, ZSTD_dictMode_e::ZSTD_noDict)
                if s == crate::compress::zstd_compress_sequences::ZSTD_dfast =>
            {
                crate::compress::zstd_double_fast::ZSTD_compressBlock_doubleFast_noDict_generic(
                    ms,
                    seqStore,
                    &mut cctx.next_rep,
                    window_to_block_end,
                    src_pos,
                    ms.cParams.minMatch,
                )
            }
            (s, ZSTD_dictMode_e::ZSTD_noDict)
                if s == crate::compress::zstd_compress_sequences::ZSTD_greedy =>
            {
                let searchMethod = if crate::compress::match_state::ZSTD_rowMatchFinderUsed(
                    ms.cParams.strategy,
                    resolvedUseRowMatchFinder,
                ) {
                    crate::compress::zstd_lazy::searchMethod_e::search_rowHash
                } else {
                    crate::compress::zstd_lazy::searchMethod_e::search_hashChain
                };
                crate::compress::zstd_lazy::ZSTD_compressBlock_lazy_generic_with_istart(
                    ms,
                    seqStore,
                    &mut cctx.next_rep,
                    window_to_block_end,
                    src_pos,
                    searchMethod,
                    0,
                    ZSTD_dictMode_e::ZSTD_noDict,
                )
            }
            (s, ZSTD_dictMode_e::ZSTD_noDict)
                if s == crate::compress::zstd_compress_sequences::ZSTD_lazy =>
            {
                let searchMethod = if crate::compress::match_state::ZSTD_rowMatchFinderUsed(
                    ms.cParams.strategy,
                    resolvedUseRowMatchFinder,
                ) {
                    crate::compress::zstd_lazy::searchMethod_e::search_rowHash
                } else {
                    crate::compress::zstd_lazy::searchMethod_e::search_hashChain
                };
                crate::compress::zstd_lazy::ZSTD_compressBlock_lazy_generic_with_istart(
                    ms,
                    seqStore,
                    &mut cctx.next_rep,
                    window_to_block_end,
                    src_pos,
                    searchMethod,
                    1,
                    ZSTD_dictMode_e::ZSTD_noDict,
                )
            }
            (s, ZSTD_dictMode_e::ZSTD_noDict)
                if s == crate::compress::zstd_compress_sequences::ZSTD_lazy2 =>
            {
                let searchMethod = if crate::compress::match_state::ZSTD_rowMatchFinderUsed(
                    ms.cParams.strategy,
                    resolvedUseRowMatchFinder,
                ) {
                    crate::compress::zstd_lazy::searchMethod_e::search_rowHash
                } else {
                    crate::compress::zstd_lazy::searchMethod_e::search_hashChain
                };
                crate::compress::zstd_lazy::ZSTD_compressBlock_lazy_generic_with_istart(
                    ms,
                    seqStore,
                    &mut cctx.next_rep,
                    window_to_block_end,
                    src_pos,
                    searchMethod,
                    2,
                    ZSTD_dictMode_e::ZSTD_noDict,
                )
            }
            (s, ZSTD_dictMode_e::ZSTD_noDict)
                if s == crate::compress::zstd_compress_sequences::ZSTD_btlazy2 =>
            {
                crate::compress::zstd_lazy::ZSTD_compressBlock_btlazy2_window(
                    ms,
                    seqStore,
                    &mut cctx.next_rep,
                    window_to_block_end,
                    src_pos,
                    src_end,
                )
            }
            (s, ZSTD_dictMode_e::ZSTD_noDict)
                if s == crate::compress::zstd_compress_sequences::ZSTD_btopt =>
            {
                crate::compress::zstd_opt::ZSTD_compressBlock_btopt_window(
                    ms,
                    seqStore,
                    &mut cctx.next_rep,
                    window_to_block_end,
                    src_pos,
                    src_end,
                )
            }
            (s, ZSTD_dictMode_e::ZSTD_noDict)
                if s == crate::compress::zstd_compress_sequences::ZSTD_btultra =>
            {
                crate::compress::zstd_opt::ZSTD_compressBlock_btultra_window(
                    ms,
                    seqStore,
                    &mut cctx.next_rep,
                    window_to_block_end,
                    src_pos,
                    src_end,
                )
            }
            (s, ZSTD_dictMode_e::ZSTD_noDict)
                if s == crate::compress::zstd_compress_sequences::ZSTD_btultra2 =>
            {
                crate::compress::zstd_opt::ZSTD_compressBlock_btultra2_window(
                    ms,
                    seqStore,
                    &mut cctx.next_rep,
                    window_to_block_end,
                    src_pos,
                    src_end,
                )
            }
            _ => {
                ms.ldmSeqStore = None;
                let blockCompressor = ZSTD_selectBlockCompressor(
                    cctx.appliedParams.cParams.strategy,
                    resolvedUseRowMatchFinder,
                    dictMode,
                );
                blockCompressor(ms, seqStore, &mut cctx.next_rep, src)
            }
        }
    };
    ZSTD_buildSeqStore_select_e::LastLiterals(lastLLSize)
}

/// Port of `ZSTD_buildSeqStore` (`zstd_compress.c:3288`). Runs the
/// configured match finder over `src` and populates `cctx.seqStore`.
/// Returns one of the `ZSTD_BuildSeqStore_e` discriminants
/// (`ZSTDbss_compress` / `ZSTDbss_noCompress`) or a zstd error code.
pub fn ZSTD_buildSeqStore(cctx: &mut ZSTD_CCtx, src: &[u8]) -> usize {
    use crate::compress::match_state::{
        ZSTD_dictMode_e, ZSTD_matchState_dictMode, ZSTD_resolveRowMatchFinderMode,
    };
    use crate::compress::seq_store::{
        ZSTD_resetSeqStore, ZSTD_storeLastLiterals, ZSTD_validateSeqStore,
    };
    use crate::compress::zstd_ldm::{
        ZSTD_ParamSwitch_e, ZSTD_ldm_blockCompress, ZSTD_ldm_generateSequences,
        ZSTD_ldm_skipRawSeqStoreBytes,
    };

    if src.len() < MIN_CBLOCK_SIZE + ZSTD_blockHeaderSize + 2 {
        if let Some(rawSeqStore) = cctx.externalMatchStore.as_mut() {
            if cctx.appliedParams.cParams.strategy
                >= crate::compress::zstd_compress_sequences::ZSTD_btopt
            {
                ZSTD_ldm_skipRawSeqStoreBytes(rawSeqStore, src.len());
            } else {
                crate::compress::zstd_ldm::ZSTD_ldm_skipSequences(
                    rawSeqStore,
                    src.len(),
                    cctx.appliedParams.cParams.minMatch,
                );
            }
        }
        return ZSTD_BuildSeqStore_e::ZSTDbss_noCompress as usize;
    }

    {
        let ms = cctx.ms.get_or_insert_with(|| {
            crate::compress::match_state::ZSTD_MatchState_t::new(cctx.appliedParams.cParams)
        });
        ZSTD_assertEqualCParams(cctx.appliedParams.cParams, ms.cParams);
        ZSTD_setEntropySeed_if_needed(ms, &cctx.prevEntropy, cctx.appliedParams.cParams.strategy);
        ms.opt.literalCompressionMode = cctx.appliedParams.literalCompressionMode;
        let curr = ms.window.base_offset;
        if curr > ms.nextToUpdate.wrapping_add(384) {
            ms.nextToUpdate =
                curr.wrapping_sub(192u32.min(curr.wrapping_sub(ms.nextToUpdate).wrapping_sub(384)));
        }
    }

    {
        let seqStore = cctx.seqStore.get_or_insert_with(|| {
            SeqStore_t::with_capacity(ZSTD_BLOCKSIZE_MAX / 3, ZSTD_BLOCKSIZE_MAX)
        });
        ZSTD_resetSeqStore(seqStore);
    }

    let ms = cctx.ms.as_mut().expect("match state must be initialized");
    let seqStore = cctx
        .seqStore
        .as_mut()
        .expect("seqStore must be initialized");
    let resolvedUseRowMatchFinder = ZSTD_resolveRowMatchFinderMode(
        cctx.appliedParams.useRowMatchFinder,
        &cctx.appliedParams.cParams,
    );

    cctx.next_rep = cctx.prev_rep;
    let lastLLSize = if let Some(rawSeqStore) = cctx.externalMatchStore.as_mut() {
        if ZSTD_hasExtSeqProd(&cctx.appliedParams) {
            return ERROR(ErrorCode::ParameterCombinationUnsupported);
        }
        ZSTD_ldm_blockCompress(
            rawSeqStore,
            ms,
            seqStore,
            &mut cctx.next_rep,
            src,
            cctx.appliedParams.useRowMatchFinder,
        )
    } else if cctx.appliedParams.ldmEnable == ZSTD_ParamSwitch_e::ZSTD_ps_enable {
        if ZSTD_hasExtSeqProd(&cctx.appliedParams) {
            return ERROR(ErrorCode::ParameterCombinationUnsupported);
        }
        let ldmParams = cctx.appliedParams.ldmParams;
        let ldmState = cctx
            .ldmState
            .as_mut()
            .expect("ldmState must be initialized when LDM is enabled");
        let rawSeqStore = &mut cctx.ldmSequences;
        rawSeqStore.size = 0;
        rawSeqStore.pos = 0;
        rawSeqStore.posInSequence = 0;
        let lowestIndex = ldmState.window.dictLimit;
        let _ = ZSTD_ldm_generateSequences(
            ldmState,
            rawSeqStore,
            &ldmParams,
            src,
            0,
            src.len(),
            lowestIndex,
        );
        ZSTD_ldm_blockCompress(
            rawSeqStore,
            ms,
            seqStore,
            &mut cctx.next_rep,
            src,
            cctx.appliedParams.useRowMatchFinder,
        )
    } else if let Some(sequenceProducer) = cctx.appliedParams.extSeqProdFunc {
        ms.ldmSeqStore = None;
        let mut extSeqs = vec![ZSTD_Sequence::default(); ZSTD_sequenceBound(src.len()).max(1)];
        let windowSize = 1usize << cctx.appliedParams.cParams.windowLog;
        let nbExternalSeqs = sequenceProducer(
            cctx.appliedParams.extSeqProdState,
            &mut extSeqs,
            src,
            &[],
            cctx.appliedParams.compressionLevel,
            windowSize,
        );
        let nbPostProcessedSeqs =
            ZSTD_postProcessSequenceProducerResult(&mut extSeqs, nbExternalSeqs, src.len());
        if !ERR_isError(nbPostProcessedSeqs) {
            let extSeqs = &extSeqs[..nbPostProcessedSeqs];
            let mut seqPos = ZSTD_SequencePosition::default();
            let seqLenSum: usize = extSeqs
                .iter()
                .map(|seq| seq.litLength as usize + seq.matchLength as usize)
                .sum();
            if seqLenSum > src.len() {
                return ERROR(ErrorCode::ExternalSequencesInvalid);
            }
            let blockSize = ZSTD_transferSequences_wBlockDelim(
                cctx,
                &mut seqPos,
                extSeqs,
                src,
                src.len(),
                cctx.appliedParams.searchForExternalRepcodes,
            );
            if ERR_isError(blockSize) {
                return blockSize;
            }
            return ZSTD_BuildSeqStore_e::ZSTDbss_compress as usize;
        }
        if cctx.appliedParams.enableMatchFinderFallback == 0 {
            return nbPostProcessedSeqs;
        }
        ms.ldmSeqStore = None;
        let dictMode = ZSTD_matchState_dictMode(ms);
        let blockCompressor = ZSTD_selectBlockCompressor(
            cctx.appliedParams.cParams.strategy,
            resolvedUseRowMatchFinder,
            dictMode,
        );
        blockCompressor(ms, seqStore, &mut cctx.next_rep, src)
    } else {
        ms.ldmSeqStore = None;
        let dictMode = ZSTD_matchState_dictMode(ms);
        match (cctx.appliedParams.cParams.strategy, dictMode) {
            (s, ZSTD_dictMode_e::ZSTD_noDict)
                if s == crate::compress::zstd_compress_sequences::ZSTD_fast =>
            {
                crate::compress::zstd_fast::ZSTD_compressBlock_fast_noDict_generic(
                    ms,
                    seqStore,
                    &mut cctx.next_rep,
                    src,
                    0,
                    ms.cParams.minMatch,
                )
            }
            (s, ZSTD_dictMode_e::ZSTD_noDict)
                if s == crate::compress::zstd_compress_sequences::ZSTD_dfast =>
            {
                crate::compress::zstd_double_fast::ZSTD_compressBlock_doubleFast_noDict_generic(
                    ms,
                    seqStore,
                    &mut cctx.next_rep,
                    src,
                    0,
                    ms.cParams.minMatch,
                )
            }
            (s, ZSTD_dictMode_e::ZSTD_noDict)
                if s == crate::compress::zstd_compress_sequences::ZSTD_btopt =>
            {
                crate::compress::zstd_opt::ZSTD_compressBlock_btopt_window(
                    ms,
                    seqStore,
                    &mut cctx.next_rep,
                    src,
                    0,
                    src.len(),
                )
            }
            (s, ZSTD_dictMode_e::ZSTD_noDict)
                if s == crate::compress::zstd_compress_sequences::ZSTD_btultra =>
            {
                crate::compress::zstd_opt::ZSTD_compressBlock_btultra_window(
                    ms,
                    seqStore,
                    &mut cctx.next_rep,
                    src,
                    0,
                    src.len(),
                )
            }
            (s, ZSTD_dictMode_e::ZSTD_noDict)
                if s == crate::compress::zstd_compress_sequences::ZSTD_btultra2 =>
            {
                crate::compress::zstd_opt::ZSTD_compressBlock_btultra2_window(
                    ms,
                    seqStore,
                    &mut cctx.next_rep,
                    src,
                    0,
                    src.len(),
                )
            }
            _ => {
                let blockCompressor = ZSTD_selectBlockCompressor(
                    cctx.appliedParams.cParams.strategy,
                    resolvedUseRowMatchFinder,
                    dictMode,
                );
                blockCompressor(ms, seqStore, &mut cctx.next_rep, src)
            }
        }
    };

    if ERR_isError(lastLLSize) {
        return lastLLSize;
    }
    if lastLLSize > src.len() {
        return ERROR(ErrorCode::Generic);
    }
    let lastLiterals = &src[src.len() - lastLLSize..];
    ZSTD_storeLastLiterals(seqStore, lastLiterals);
    ZSTD_validateSeqStore(seqStore, cctx.appliedParams.cParams.minMatch);
    ZSTD_BuildSeqStore_e::ZSTDbss_compress as usize
}

/// Rust-only helper: window-aware variant of `ZSTD_buildSeqStore`. Operates
/// on a contiguous `[dict_prefix .. src]` window buffer with `src_pos`/
/// `src_end` indices, so the match finder can address dict history via
/// the same offsets used by the C port.
fn ZSTD_buildSeqStore_with_window(
    cctx: &mut ZSTD_CCtx,
    window_buf: &[u8],
    src_pos: usize,
    src_end: usize,
) -> usize {
    use crate::compress::seq_store::{
        ZSTD_resetSeqStore, ZSTD_storeLastLiterals, ZSTD_validateSeqStore,
    };
    use crate::compress::zstd_ldm::ZSTD_ldm_skipRawSeqStoreBytes;

    let src = &window_buf[src_pos..src_end];
    if src.len() < MIN_CBLOCK_SIZE + ZSTD_blockHeaderSize + 2 {
        if let Some(rawSeqStore) = cctx.externalMatchStore.as_mut() {
            if cctx.appliedParams.cParams.strategy
                >= crate::compress::zstd_compress_sequences::ZSTD_btopt
            {
                ZSTD_ldm_skipRawSeqStoreBytes(rawSeqStore, src.len());
            } else {
                crate::compress::zstd_ldm::ZSTD_ldm_skipSequences(
                    rawSeqStore,
                    src.len(),
                    cctx.appliedParams.cParams.minMatch,
                );
            }
        }
        return ZSTD_BuildSeqStore_e::ZSTDbss_noCompress as usize;
    }

    {
        let ms = cctx.ms.get_or_insert_with(|| {
            crate::compress::match_state::ZSTD_MatchState_t::new(cctx.appliedParams.cParams)
        });
        ZSTD_assertEqualCParams(cctx.appliedParams.cParams, ms.cParams);
        ZSTD_setEntropySeed_if_needed(ms, &cctx.prevEntropy, cctx.appliedParams.cParams.strategy);
        ms.opt.literalCompressionMode = cctx.appliedParams.literalCompressionMode;
        let curr = ms.window.base_offset.wrapping_add(src_pos as u32);
        if curr > ms.nextToUpdate.wrapping_add(384) {
            ms.nextToUpdate =
                curr.wrapping_sub(192u32.min(curr.wrapping_sub(ms.nextToUpdate).wrapping_sub(384)));
        }
    }

    {
        let seqStore = cctx.seqStore.get_or_insert_with(|| {
            SeqStore_t::with_capacity(ZSTD_BLOCKSIZE_MAX / 3, ZSTD_BLOCKSIZE_MAX)
        });
        ZSTD_resetSeqStore(seqStore);
    }

    match ZSTD_buildSeqStore_selectMatches_with_window(cctx, window_buf, src_pos, src_end) {
        ZSTD_buildSeqStore_select_e::Final(rc) => rc,
        ZSTD_buildSeqStore_select_e::LastLiterals(lastLLSize) => {
            if ERR_isError(lastLLSize) {
                return lastLLSize;
            }
            if lastLLSize > src.len() {
                return ERROR(ErrorCode::Generic);
            }
            let lastLiterals = &src[src.len() - lastLLSize..];
            let seqStore = cctx
                .seqStore
                .as_mut()
                .expect("seqStore must be initialized");
            ZSTD_storeLastLiterals(seqStore, lastLiterals);
            ZSTD_validateSeqStore(seqStore, cctx.appliedParams.cParams.minMatch);
            ZSTD_BuildSeqStore_e::ZSTDbss_compress as usize
        }
    }
}

#[inline]
fn ZSTD_setEntropySeed_if_needed(
    ms: &mut crate::compress::match_state::ZSTD_MatchState_t,
    prevEntropy: &ZSTD_entropyCTables_t,
    strategy: u32,
) {
    if strategy >= crate::compress::zstd_compress_sequences::ZSTD_btopt {
        ms.entropySeed = Some(prevEntropy.clone());
    } else {
        ms.entropySeed = None;
    }
}

/// Port of `ZSTD_copyCCtx` (`zstd_compress.c:2615`). Deep-copies
/// `src` state into `dst` for starting a new compression at the
/// given `pledgedSrcSize`. Upstream's implementation walks its
/// cwksp-allocated state; our Rust port owns each field and uses
/// `Clone::clone_from` to do a byte-faithful copy. Sets
/// `pledged_src_size` from the argument (0 → UNKNOWN per upstream).
pub fn ZSTD_copyCCtx_internal(
    dstCCtx: &mut ZSTD_CCtx,
    srcCCtx: &ZSTD_CCtx,
    fParams: ZSTD_FrameParameters,
    pledgedSrcSize: u64,
    zbuff: ZSTD_buffered_policy_e,
) -> usize {
    use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;

    if srcCCtx.stage != ZSTD_compressionStage_e::ZSTDcs_init
        && srcCCtx.stage != ZSTD_compressionStage_e::ZSTDcs_created
    {
        return ERROR(ErrorCode::StageWrong);
    }
    if srcCCtx.stage == ZSTD_compressionStage_e::ZSTDcs_created {
        dstCCtx.clone_from(srcCCtx);
        dstCCtx.appliedParams.fParams = fParams;
        dstCCtx.pledged_src_size =
            if pledgedSrcSize == 0 || pledgedSrcSize == ZSTD_CONTENTSIZE_UNKNOWN {
                None
            } else {
                Some(pledgedSrcSize)
            };
        dstCCtx.pledgedSrcSizePlusOne = pledgedSrcSize.wrapping_add(1);
        return 0;
    }

    let mut params = dstCCtx.requestedParams;
    params.cParams = srcCCtx.appliedParams.cParams;
    params.useRowMatchFinder = srcCCtx.appliedParams.useRowMatchFinder;
    params.postBlockSplitter = srcCCtx.appliedParams.postBlockSplitter;
    params.ldmEnable = srcCCtx.appliedParams.ldmEnable;
    params.maxBlockSize = srcCCtx.appliedParams.maxBlockSize;
    params.fParams = fParams;

    let rc = ZSTD_resetCCtx_internal(
        dstCCtx,
        &params,
        pledgedSrcSize,
        0,
        ZSTD_compResetPolicy_e::ZSTDcrp_leaveDirty,
        zbuff,
    );
    if ERR_isError(rc) {
        return rc;
    }

    dstCCtx.clone_from(srcCCtx);
    dstCCtx.appliedParams.fParams = fParams;
    dstCCtx.pledged_src_size = if pledgedSrcSize == 0 || pledgedSrcSize == ZSTD_CONTENTSIZE_UNKNOWN
    {
        None
    } else {
        Some(pledgedSrcSize)
    };
    dstCCtx.pledgedSrcSizePlusOne = pledgedSrcSize.wrapping_add(1);
    0
}

/// Port of `ZSTD_copyCCtx` (`zstd_compress.c:2615`). Duplicates `srcCCtx`
/// into `dstCCtx`. Only valid during `ZSTDcs_init` (after creation, before
/// the first `ZSTD_compressContinue` call). `pledgedSrcSize == 0` means
/// unknown. Returns 0 on success or a zstd error code.
pub fn ZSTD_copyCCtx(dst: &mut ZSTD_CCtx, src: &ZSTD_CCtx, pledgedSrcSize: u64) -> usize {
    let pledged = if pledgedSrcSize == 0 {
        crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN
    } else {
        pledgedSrcSize
    };
    let mut fParams = ZSTD_FrameParameters {
        contentSizeFlag: (pledged != crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN)
            as u32,
        checksumFlag: 0,
        noDictIDFlag: 0,
    };
    fParams.noDictIDFlag = (!src.param_dictID) as u32;
    ZSTD_copyCCtx_internal(
        dst,
        src,
        fParams,
        pledged,
        ZSTD_buffered_policy_e::ZSTDb_not_buffered,
    )
}

/// Port of `ZSTD_shouldAttachDict` (`zstd_compress.c:2333`). Decides
/// whether the CDict's pre-digested match state should be attached
/// in place (cheap reuse) versus copied into the CCtx (better for
/// large inputs). Returns true when: the pledged source size fits
/// within the strategy-specific cutoff, or the size is unknown. DDSS
/// CDicts always attach.
pub fn ZSTD_shouldAttachDict(
    cdict: &ZSTD_CDict,
    params: &ZSTD_CCtx_params,
    pledgedSrcSize: u64,
) -> bool {
    use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;
    let strategy = cdict.cParams.strategy as usize;
    let cutoff = attachDictSizeCutoffs
        .get(strategy)
        .copied()
        .unwrap_or(attachDictSizeCutoffs[1]);
    cdict.dedicatedDictSearch != 0 || {
        let fits = pledgedSrcSize <= cutoff as u64
            || pledgedSrcSize == ZSTD_CONTENTSIZE_UNKNOWN
            || params.attachDictPref == ZSTD_dictAttachPref_e::ZSTD_dictForceAttach;
        fits && params.attachDictPref != ZSTD_dictAttachPref_e::ZSTD_dictForceCopy
            && params.forceWindow == 0
    }
}

/// Port of `ZSTD_resetCCtx_internal` (`zstd_compress.c:2145`).
/// Re-initializes the owned CCtx state for a fresh compression
/// session, reusing existing allocations when possible.
pub fn ZSTD_resetCCtx_internal(
    zc: &mut ZSTD_CCtx,
    params: &ZSTD_CCtx_params,
    pledgedSrcSize: u64,
    loadedDictSize: usize,
    crp: ZSTD_compResetPolicy_e,
    zbuff: ZSTD_buffered_policy_e,
) -> usize {
    use crate::compress::match_state::{
        ZSTD_allocateChainTable, ZSTD_dictTooBig, ZSTD_indexTooCloseToMax,
        ZSTD_resolveRowMatchFinderMode, ZSTD_rowMatchFinderUsed,
    };
    use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;

    debug_assert!(!ERR_isError(ZSTD_checkCParams(params.cParams)));
    zc.isFirstBlock = 1;
    zc.appliedParams = *params;
    zc.appliedParams.useRowMatchFinder = ZSTD_resolveRowMatchFinderMode(
        zc.appliedParams.useRowMatchFinder,
        &zc.appliedParams.cParams,
    );
    if zc.appliedParams.ldmEnable == crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_enable {
        crate::compress::zstd_ldm::ZSTD_ldm_adjustParameters(
            &mut zc.appliedParams.ldmParams,
            &params.cParams,
        );
    }

    let windowSize = core::cmp::max(
        1usize,
        ((1u64 << params.cParams.windowLog).min(pledgedSrcSize)) as usize,
    );
    let blockSize =
        crate::compress::match_state::ZSTD_resolveMaxBlockSize(params.maxBlockSize).min(windowSize);
    let maxNbSeq = ZSTD_maxNbSeq(
        blockSize,
        params.cParams.minMatch,
        ZSTD_hasExtSeqProd(params),
    );
    let buffOutSize = if zbuff == ZSTD_buffered_policy_e::ZSTDb_buffered
        && params.outBufferMode == ZSTD_bufferMode_e::ZSTD_bm_buffered
    {
        ZSTD_compressBound(blockSize).wrapping_add(1)
    } else {
        0
    };
    let buffInSize = if zbuff == ZSTD_buffered_policy_e::ZSTDb_buffered
        && params.inBufferMode == ZSTD_bufferMode_e::ZSTD_bm_buffered
    {
        windowSize.wrapping_add(blockSize)
    } else {
        0
    };
    let maxNbLdmSeq =
        crate::compress::zstd_ldm::ZSTD_ldm_getMaxNbSeq(zc.appliedParams.ldmParams, blockSize)
            .max(1);

    if zc.staticSize != 0 {
        let required = ZSTD_estimateCCtxSize_usingCCtxParams_internal(
            &params.cParams,
            params,
            true,
            buffInSize,
            buffOutSize,
            pledgedSrcSize,
        );
        if zc.staticSize < required {
            return ERROR(ErrorCode::WorkSpaceTooSmall);
        }
    }

    let needsIndexReset = if crp == ZSTD_compResetPolicy_e::ZSTDcrp_makeClean {
        ZSTD_indexResetPolicy_e::ZSTDirp_reset
    } else {
        match zc.ms.as_ref() {
            None => ZSTD_indexResetPolicy_e::ZSTDirp_reset,
            Some(ms) => {
                let chainSize = if ZSTD_allocateChainTable(
                    params.cParams.strategy,
                    zc.appliedParams.useRowMatchFinder,
                    0,
                ) {
                    1usize << params.cParams.chainLog
                } else {
                    0
                };
                let hashSize = 1usize << params.cParams.hashLog;
                let hashLog3 = if params.cParams.minMatch == 3 {
                    ZSTD_HASHLOG3_MAX.min(params.cParams.windowLog)
                } else {
                    0
                };
                let hash3Size = if hashLog3 != 0 { 1usize << hashLog3 } else { 0 };
                let tagTableSize = if ZSTD_rowMatchFinderUsed(
                    params.cParams.strategy,
                    zc.appliedParams.useRowMatchFinder,
                ) {
                    hashSize
                } else {
                    0
                };
                let tableLayoutChanged = ms.hashTable.len() != hashSize
                    || ms.hashTable3.len() != hash3Size
                    || ms.chainTable.len() != chainSize
                    || ms.tagTable.len() != tagTableSize;
                if ZSTD_indexTooCloseToMax(&ms.window)
                    || ZSTD_dictTooBig(loadedDictSize)
                    || !zc.initialized
                    || tableLayoutChanged
                {
                    ZSTD_indexResetPolicy_e::ZSTDirp_reset
                } else {
                    ZSTD_indexResetPolicy_e::ZSTDirp_continue
                }
            }
        }
    };

    let ms = zc.ms.get_or_insert_with(|| {
        crate::compress::match_state::ZSTD_MatchState_t::new(params.cParams)
    });
    let rc = ZSTD_reset_matchState(
        ms,
        &params.cParams,
        zc.appliedParams.useRowMatchFinder,
        crp,
        needsIndexReset,
        ZSTD_resetTarget_e::ZSTD_resetTarget_CCtx,
    );
    if ERR_isError(rc) {
        return rc;
    }

    match zc.seqStore.as_mut() {
        Some(store) if store.maxNbSeq >= maxNbSeq && store.maxNbLit >= blockSize => store.reset(),
        _ => zc.seqStore = Some(SeqStore_t::with_capacity(maxNbSeq.max(1), blockSize.max(1))),
    }
    if zc.appliedParams.ldmEnable == crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_enable {
        match zc.ldmState.as_mut() {
            Some(ldmState)
                if ldmState.hashTable.len() == (1usize << zc.appliedParams.ldmParams.hashLog)
                    && ldmState.bucketOffsets.len()
                        == (1usize
                            << (zc.appliedParams.ldmParams.hashLog
                                - zc.appliedParams.ldmParams.bucketSizeLog)) =>
            {
                crate::compress::match_state::ZSTD_window_init(&mut ldmState.window);
                ldmState.loadedDictEnd = 0;
                ldmState
                    .hashTable
                    .fill(crate::compress::zstd_ldm::ldmEntry_t::default());
                ldmState.bucketOffsets.fill(0);
            }
            _ => {
                zc.ldmState = Some(crate::compress::zstd_ldm::ldmState_t::new(
                    &zc.appliedParams.ldmParams,
                ));
            }
        }
        if zc.ldmSequences.capacity < maxNbLdmSeq {
            zc.ldmSequences = crate::compress::zstd_ldm::RawSeqStore_t::with_capacity(maxNbLdmSeq);
        } else {
            zc.ldmSequences.size = 0;
            zc.ldmSequences.pos = 0;
            zc.ldmSequences.posInSequence = 0;
        }
    } else {
        zc.ldmState = None;
        zc.ldmSequences = crate::compress::zstd_ldm::RawSeqStore_t::default();
    }

    zc.pledgedSrcSizePlusOne = pledgedSrcSize.wrapping_add(1);
    zc.pledged_src_size = if pledgedSrcSize == ZSTD_CONTENTSIZE_UNKNOWN {
        None
    } else {
        Some(pledgedSrcSize)
    };
    zc.externalMatchStore = None;
    zc.stream_in_buffer.clear();
    zc.stream_params_snapshot = None;
    zc.stream_out_buffer.clear();
    zc.stream_out_drained = 0;
    zc.stream_stage = ZSTD_cStreamStage::zcss_init;
    zc.stream_in_to_compress = 0;
    zc.stream_in_target = 0;
    zc.stream_frame_ended = false;
    zc.stream_window_base = crate::compress::match_state::ZSTD_WINDOW_START_INDEX;
    zc.expected_in_src = 0;
    zc.expected_in_size = 0;
    zc.expected_in_pos = 0;
    zc.expected_out_buffer_size = 0;
    zc.buffer_expectations_set = false;
    zc.stream_closed = false;
    zc.stream_frame_completed = false;
    if zc.stream_in_buffer.capacity() < buffInSize {
        zc.stream_in_buffer
            .reserve(buffInSize - zc.stream_in_buffer.capacity());
    }
    if zc.stream_out_buffer.capacity() < buffOutSize {
        zc.stream_out_buffer
            .reserve(buffOutSize - zc.stream_out_buffer.capacity());
    }
    zc.consumedSrcSize = 0;
    zc.producedCSize = 0;
    zc.stage = ZSTD_compressionStage_e::ZSTDcs_init;
    zc.dictID = 0;
    zc.dictContentSize = 0;
    zc.blockSizeMax = blockSize;
    crate::common::xxhash::XXH64_reset(&mut zc.xxhState, 0);
    ZSTD_reset_compressedBlockState(&mut zc.prev_rep, &mut zc.prevEntropy);
    ZSTD_reset_compressedBlockState(&mut zc.next_rep, &mut zc.nextEntropy);
    zc.initialized = true;
    0
}

/// Port of `ZSTD_resetCCtx_byAttachingCDict` (`zstd_compress.c:2351`).
/// Reuses the CDict's preseeded match-state tables by attaching them
/// as a live `dictMatchState` on the working CCtx.
pub fn ZSTD_resetCCtx_byAttachingCDict(
    cctx: &mut ZSTD_CCtx,
    cdict: &ZSTD_CDict,
    mut params: ZSTD_CCtx_params,
    pledgedSrcSize: u64,
    zbuff: ZSTD_buffered_policy_e,
) -> usize {
    let windowLog = params.cParams.windowLog;
    params.cParams = ZSTD_adjustCParams_internal(
        cdict.cParams,
        pledgedSrcSize,
        cdict.dictContent.len() as u64,
        ZSTD_CParamMode_e::ZSTD_cpm_attachDict,
        params.useRowMatchFinder,
    );
    params.cParams.windowLog = windowLog;
    params.useRowMatchFinder = cdict.useRowMatchFinder;
    let rc = ZSTD_resetCCtx_internal(
        cctx,
        &params,
        pledgedSrcSize,
        0,
        ZSTD_compResetPolicy_e::ZSTDcrp_makeClean,
        zbuff,
    );
    if ERR_isError(rc) {
        return rc;
    }
    if let Some(ms) = cctx.ms.as_mut() {
        let cdict_end = cdict
            .matchState
            .window
            .nextSrc
            .saturating_sub(cdict.matchState.window.base_offset);
        let cdict_len = cdict_end.saturating_sub(cdict.matchState.window.dictLimit);
        if cdict_len != 0 {
            ms.dictMatchState = Some(Box::new(cdict.matchState.clone()));
            ms.dictContent.clear();
            let cdict_end_index = cdict.matchState.window.nextSrc;
            if ms.window.dictLimit < cdict_end_index {
                ms.window.nextSrc = cdict_end_index;
                crate::compress::match_state::ZSTD_window_clear(&mut ms.window);
                ms.window.base_offset = cdict_end_index;
            }
            ms.loadedDictEnd = ms.window.dictLimit;
        }
    }
    cctx.stream_dict.clear();
    cctx.stream_dict_original.clear();
    cctx.stream_dict_content_type =
        crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_auto;
    cctx.stream_level = Some(cdict.compressionLevel);
    cctx.dictID = cdict.dictID;
    cctx.dictContentSize = cdict.dictContent.len();
    cctx.prevEntropy = cdict.entropy.clone();
    cctx.prev_rep = cdict.rep;
    0
}

/// Port of `ZSTD_resetCCtx_byCopyingCDict` (`zstd_compress.c:2417`).
/// Reuses the CDict's pre-seeded tables by cloning them into the
/// working match state.
pub fn ZSTD_resetCCtx_byCopyingCDict(
    cctx: &mut ZSTD_CCtx,
    cdict: &ZSTD_CDict,
    mut params: ZSTD_CCtx_params,
    pledgedSrcSize: u64,
    zbuff: ZSTD_buffered_policy_e,
) -> usize {
    let windowLog = params.cParams.windowLog;
    params.cParams = cdict.cParams;
    params.cParams.windowLog = windowLog;
    params.useRowMatchFinder = cdict.useRowMatchFinder;
    let rc = ZSTD_resetCCtx_internal(
        cctx,
        &params,
        pledgedSrcSize,
        0,
        ZSTD_compResetPolicy_e::ZSTDcrp_leaveDirty,
        zbuff,
    );
    if ERR_isError(rc) {
        return rc;
    }
    cctx.ms = Some(cdict.matchState.clone());
    if let Some(ms) = cctx.ms.as_mut() {
        ms.dictMatchState = None;
        ms.cParams.windowLog = params.cParams.windowLog;
        ms.dictContent = cdict.dictContent.clone();
        ms.loadedDictEnd = cdict.dictContent.len() as u32;
    }
    cctx.stream_dict = cdict.dictContent.clone();
    cctx.stream_dict_original = cdict.dictContent.clone();
    cctx.stream_dict_content_type =
        crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_rawContent;
    cctx.stream_level = Some(cdict.compressionLevel);
    cctx.dictID = cdict.dictID;
    cctx.dictContentSize = cdict.dictContent.len();
    cctx.prevEntropy = cdict.entropy.clone();
    cctx.prev_rep = cdict.rep;
    0
}

/// Port of `ZSTD_referenceExternalSequences` (`zstd_compress.c:4804`).
/// Stores a copied raw-sequence stream on the CCtx so subsequent
/// `ZSTD_buildSeqStore()` calls can consume it block-by-block.
#[inline]
pub fn ZSTD_referenceExternalSequences(
    cctx: &mut ZSTD_CCtx,
    seq: Option<&[crate::compress::zstd_ldm::rawSeq]>,
) -> usize {
    debug_assert!(
        cctx.stage == ZSTD_compressionStage_e::ZSTDcs_init
            || cctx.stage == ZSTD_compressionStage_e::ZSTDcs_created
    );
    cctx.externalMatchStore = seq.map(|seqs| crate::compress::zstd_ldm::RawSeqStore_t {
        seq: seqs.to_vec(),
        pos: 0,
        posInSequence: 0,
        size: seqs.len(),
        capacity: seqs.len(),
    });
    0
}

/// Port of `ZSTD_loadZstdDictionary` (`zstd_compress.c:5185`).
/// Parses the entropy tables from a full zstd-format dictionary and
/// seeds a match state with the remaining raw-content bytes.
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_loadDictionaryContent(
    ms: &mut crate::compress::match_state::ZSTD_MatchState_t,
    mut ls: Option<&mut crate::compress::zstd_ldm::ldmState_t>,
    params: &ZSTD_CCtx_params,
    src: &[u8],
    dtlm: crate::compress::zstd_fast::ZSTD_dictTableLoadMethod_e,
    tfp: crate::compress::zstd_fast::ZSTD_tableFillPurpose_e,
) -> usize {
    use crate::compress::match_state::{
        ZSTD_CDictIndicesAreTagged, ZSTD_overflowCorrectIfNeeded, ZSTD_window_update,
        ZSTD_SHORT_CACHE_TAG_BITS, ZSTD_WINDOW_START_INDEX,
    };
    use crate::compress::zstd_compress_sequences::{
        ZSTD_btlazy2, ZSTD_btopt, ZSTD_btultra, ZSTD_btultra2, ZSTD_dfast, ZSTD_fast, ZSTD_greedy,
        ZSTD_lazy, ZSTD_lazy2,
    };
    use crate::compress::zstd_double_fast::ZSTD_fillDoubleHashTable;
    use crate::compress::zstd_fast::{ZSTD_fillHashTable, HASH_READ_SIZE};
    use crate::compress::zstd_lazy::{ZSTD_insertAndFindFirstIndex, ZSTD_row_update};
    use crate::compress::zstd_ldm::{
        ZSTD_ParamSwitch_e, ZSTD_ldm_fillHashTable, ZSTD_ldm_getMaxNbSeq,
    };
    use crate::compress::zstd_opt::ZSTD_updateTree;

    debug_assert_eq!(ms.cParams.strategy, params.cParams.strategy);
    debug_assert_eq!(ms.cParams.hashLog, params.cParams.hashLog);
    debug_assert_eq!(ms.cParams.chainLog, params.cParams.chainLog);

    let mut dict = src;
    let mut maxDictSize = crate::compress::match_state::ZSTD_CURRENT_MAX
        .saturating_sub(ZSTD_WINDOW_START_INDEX) as usize;
    if ZSTD_CDictIndicesAreTagged(&params.cParams)
        && tfp == crate::compress::zstd_fast::ZSTD_tableFillPurpose_e::ZSTD_tfp_forCDict
    {
        maxDictSize = ((1usize << (32 - ZSTD_SHORT_CACHE_TAG_BITS))
            - ZSTD_WINDOW_START_INDEX as usize)
            .min(maxDictSize);
        debug_assert!(ls.is_none());
    }
    if dict.len() > maxDictSize {
        dict = &dict[dict.len() - maxDictSize..];
    }

    let dictStart = ZSTD_WINDOW_START_INDEX;
    let forceNonContiguous = params.deterministicRefPrefix != 0
        || crate::compress::match_state::ZSTD_window_isEmpty(&ms.window);
    ZSTD_window_update(&mut ms.window, dictStart, dict.len(), forceNonContiguous);
    ms.dictContent.clear();
    ms.dictContent.extend_from_slice(dict);
    ms.forceNonContiguous = params.deterministicRefPrefix != 0;

    if let Some(ldmState) = ls.as_deref_mut() {
        if params.ldmParams.enableLdm == ZSTD_ParamSwitch_e::ZSTD_ps_enable {
            ZSTD_window_update(&mut ldmState.window, dictStart, dict.len(), false);
            ldmState.loadedDictEnd = if params.forceWindow != 0 {
                0
            } else {
                dictStart.wrapping_add(dict.len() as u32)
            };
            if ldmState.hashTable.is_empty() || ldmState.bucketOffsets.is_empty() {
                *ldmState = crate::compress::zstd_ldm::ldmState_t::new(&params.ldmParams);
            }
            let neededSeq = ZSTD_ldm_getMaxNbSeq(params.ldmParams, dict.len());
            let _ = neededSeq;
            ZSTD_ldm_fillHashTable(ldmState, dict, 0, &params.ldmParams);
        }
    }

    let hashSizedMaxDict = 1usize
        << ((params.cParams.hashLog + 3)
            .max(params.cParams.chainLog + 1)
            .min(31) as usize);
    if dict.len() > hashSizedMaxDict {
        dict = &dict[dict.len() - hashSizedMaxDict..];
        ms.dictContent.clear();
        ms.dictContent.extend_from_slice(dict);
    }

    ms.nextToUpdate = ZSTD_WINDOW_START_INDEX;
    ms.loadedDictEnd = if params.forceWindow != 0 {
        0
    } else {
        ZSTD_WINDOW_START_INDEX.wrapping_add(dict.len() as u32)
    };

    if dict.len() <= HASH_READ_SIZE {
        return 0;
    }

    ZSTD_overflowCorrectIfNeeded(
        ms,
        params.useRowMatchFinder,
        0,
        params.cParams.windowLog,
        params.cParams.chainLog,
        params.cParams.strategy,
        ZSTD_WINDOW_START_INDEX,
        ZSTD_WINDOW_START_INDEX.wrapping_add(dict.len() as u32),
    );

    let endForInsert = dict.len() - HASH_READ_SIZE;
    match params.cParams.strategy {
        s if s == ZSTD_fast => ZSTD_fillHashTable(ms, dict, dtlm, tfp),
        s if s == ZSTD_dfast => ZSTD_fillDoubleHashTable(ms, dict, dtlm, tfp),
        s if s == ZSTD_greedy || s == ZSTD_lazy || s == ZSTD_lazy2 => {
            if params.useRowMatchFinder == ZSTD_ParamSwitch_e::ZSTD_ps_enable {
                ms.tagTable.fill(0);
                ZSTD_row_update(ms, endForInsert, dict);
            } else {
                ZSTD_insertAndFindFirstIndex(ms, dict, endForInsert);
            }
        }
        s if s == ZSTD_btlazy2 || s == ZSTD_btopt || s == ZSTD_btultra || s == ZSTD_btultra2 => {
            ZSTD_updateTree(
                ms,
                dict,
                ZSTD_WINDOW_START_INDEX.wrapping_add(endForInsert as u32),
                dict.len(),
            );
        }
        _ => return ERROR(ErrorCode::ParameterUnsupported),
    }
    ms.nextToUpdate = ZSTD_WINDOW_START_INDEX.wrapping_add(dict.len() as u32);
    0
}

/// Port of `ZSTD_loadZstdDictionary` (`zstd_compress.c:5185`).
/// Parses the entropy tables from a full zstd-format dictionary and
/// seeds a match state with the remaining raw-content bytes.
pub fn ZSTD_loadZstdDictionary(
    entropy: &mut ZSTD_entropyCTables_t,
    rep: &mut [u32; 3],
    ms: &mut crate::compress::match_state::ZSTD_MatchState_t,
    params: &ZSTD_CCtx_params,
    dict: &[u8],
    dtlm: crate::compress::zstd_fast::ZSTD_dictTableLoadMethod_e,
    tfp: crate::compress::zstd_fast::ZSTD_tableFillPurpose_e,
) -> usize {
    use crate::common::mem::MEM_readLE32;
    use crate::compress::match_state::ZSTD_window_init;
    use crate::decompress::zstd_decompress::ZSTD_MAGICNUMBER_DICTIONARY;

    if dict.len() < 8 || MEM_readLE32(&dict[..4]) != ZSTD_MAGICNUMBER_DICTIONARY {
        return ERROR(ErrorCode::DictionaryWrong);
    }
    let eSize = ZSTD_loadCEntropy(entropy, rep, dict);
    if ERR_isError(eSize) {
        return eSize;
    }
    let dictContent = &dict[eSize..];
    ZSTD_window_init(&mut ms.window);
    let load = ZSTD_loadDictionaryContent(ms, None, params, dictContent, dtlm, tfp);
    if ERR_isError(load) {
        return load;
    }
    if params.fParams.noDictIDFlag != 0 {
        0
    } else {
        MEM_readLE32(&dict[4..8]) as usize
    }
}

/// Port of `ZSTD_initCDict_internal` (`zstd_compress.c:5575`).
pub fn ZSTD_initCDict_internal(
    cdict: &mut ZSTD_CDict,
    dictBuffer: &[u8],
    _dictLoadMethod: crate::decompress::zstd_ddict::ZSTD_dictLoadMethod_e,
    dictContentType: crate::decompress::zstd_ddict::ZSTD_dictContentType_e,
    params: ZSTD_CCtx_params,
) -> usize {
    use crate::common::mem::MEM_readLE32;
    use crate::compress::zstd_fast::{
        ZSTD_dictTableLoadMethod_e, ZSTD_fillHashTable, ZSTD_tableFillPurpose_e,
    };
    use crate::decompress::zstd_ddict::ZSTD_dictContentType_e;
    use crate::decompress::zstd_decompress::ZSTD_MAGICNUMBER_DICTIONARY;

    cdict.cParams = params.cParams;
    cdict.useRowMatchFinder = params.useRowMatchFinder;
    cdict.dedicatedDictSearch = params.enableDedicatedDictSearch;
    cdict.dictContent = dictBuffer.to_vec();
    cdict.matchState = crate::compress::match_state::ZSTD_MatchState_t::new(params.cParams);
    cdict.matchState.dedicatedDictSearch = cdict.dedicatedDictSearch as u32;
    cdict.matchState.dictContent.clear();
    let rc = ZSTD_reset_matchState(
        &mut cdict.matchState,
        &params.cParams,
        params.useRowMatchFinder,
        ZSTD_compResetPolicy_e::ZSTDcrp_makeClean,
        ZSTD_indexResetPolicy_e::ZSTDirp_reset,
        ZSTD_resetTarget_e::ZSTD_resetTarget_CDict,
    );
    if ERR_isError(rc) {
        return rc;
    }
    cdict.matchState.dedicatedDictSearch = cdict.dedicatedDictSearch as u32;
    ZSTD_reset_compressedBlockState(&mut cdict.rep, &mut cdict.entropy);

    if dictBuffer.len() >= 8
        && dictContentType != ZSTD_dictContentType_e::ZSTD_dct_rawContent
        && MEM_readLE32(&dictBuffer[..4]) == ZSTD_MAGICNUMBER_DICTIONARY
    {
        let dictID = ZSTD_loadZstdDictionary(
            &mut cdict.entropy,
            &mut cdict.rep,
            &mut cdict.matchState,
            &params,
            dictBuffer,
            ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_full,
            ZSTD_tableFillPurpose_e::ZSTD_tfp_forCDict,
        );
        if ERR_isError(dictID) {
            return dictID;
        }
        cdict.dictID = dictID as u32;
        cdict.dictContent = dictBuffer
            [ZSTD_loadCEntropy(&mut cdict.entropy, &mut cdict.rep, dictBuffer)..]
            .to_vec();
        cdict.matchState.dictContent = cdict.dictContent.clone();
    } else {
        cdict.dictID = 0;
        if !cdict.dictContent.is_empty() {
            cdict.matchState.dictContent = cdict.dictContent.clone();
            cdict.matchState.window.nextSrc = crate::compress::match_state::ZSTD_WINDOW_START_INDEX
                .wrapping_add(cdict.dictContent.len() as u32);
            ZSTD_fillHashTable(
                &mut cdict.matchState,
                &cdict.dictContent,
                ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_full,
                ZSTD_tableFillPurpose_e::ZSTD_tfp_forCDict,
            );
        }
    }
    0
}

/// Port of `ZSTD_createCDict_advanced_internal` (`zstd_compress.c:5629`).
pub fn ZSTD_createCDict_advanced_internal(
    _dictSize: usize,
    _dictLoadMethod: crate::decompress::zstd_ddict::ZSTD_dictLoadMethod_e,
    cParams: crate::compress::match_state::ZSTD_compressionParameters,
    useRowMatchFinder: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
    _enableDedicatedDictSearch: i32,
    customMem: ZSTD_customMem,
) -> Option<Box<ZSTD_CDict>> {
    if !ZSTD_customMem_validate(customMem) {
        return None;
    }
    unsafe {
        ZSTD_customAllocBox(
            ZSTD_CDict {
                dictContent: Vec::new(),
                compressionLevel: ZSTD_NO_CLEVEL,
                dictID: 0,
                cParams,
                useRowMatchFinder,
                entropy: ZSTD_entropyCTables_t::default(),
                rep: ZSTD_REP_START_VALUE,
                dedicatedDictSearch: _enableDedicatedDictSearch,
                matchState: {
                    let mut ms = crate::compress::match_state::ZSTD_MatchState_t::new(cParams);
                    ms.dedicatedDictSearch = _enableDedicatedDictSearch as u32;
                    ms
                },
                customMem,
            },
            customMem,
        )
    }
}

/// Port of `ZSTD_getCParamsFromCDict` (`zstd_compress.c:5828`). Returns
/// the cParams the CDict was created with. Used by
/// `ZSTD_compressBegin_usingCDict_*` to seed the CCtx with the CDict's
/// strategy before any compression begins.
#[inline]
pub fn ZSTD_getCParamsFromCDict(
    cdict: &ZSTD_CDict,
) -> crate::compress::match_state::ZSTD_compressionParameters {
    cdict.cParams
}

/// Port of `ZSTD_freeCDict`. Drops the Box; returns 0.
pub fn ZSTD_freeCDict(_cdict: Option<Box<ZSTD_CDict>>) -> usize {
    if let Some(cdict) = _cdict {
        let customMem = cdict.customMem;
        unsafe {
            ZSTD_customFreeBox(cdict, customMem);
        }
    }
    0
}

/// Port of `ZSTD_createCDict_advanced2` (`zstd_compress.c:5684`). Full
/// variant: accepts explicit load method, content type, and an
/// originating `ZSTD_CCtx_params` whose `enableDedicatedDictSearch`
/// flag picks between plain and DDSS cParams.
pub fn ZSTD_createCDict_advanced2(
    dict: &[u8],
    dictLoadMethod: crate::decompress::zstd_ddict::ZSTD_dictLoadMethod_e,
    dictContentType: crate::decompress::zstd_ddict::ZSTD_dictContentType_e,
    originalCctxParams: &ZSTD_CCtx_params,
) -> Option<Box<ZSTD_CDict>> {
    let mut cctxParams = *originalCctxParams;
    let mut cParams = if cctxParams.enableDedicatedDictSearch != 0 {
        let mut cp = ZSTD_dedicatedDictSearch_getCParams(cctxParams.compressionLevel, dict.len());
        ZSTD_overrideCParams(&mut cp, &cctxParams.cParams);
        cp
    } else {
        use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;
        ZSTD_getCParamsFromCCtxParams(
            &cctxParams,
            ZSTD_CONTENTSIZE_UNKNOWN,
            dict.len(),
            ZSTD_CParamMode_e::ZSTD_cpm_createCDict,
        )
    };

    if !ZSTD_dedicatedDictSearch_isSupported(&cParams) {
        cctxParams.enableDedicatedDictSearch = 0;
        use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;
        cParams = ZSTD_getCParamsFromCCtxParams(
            &cctxParams,
            ZSTD_CONTENTSIZE_UNKNOWN,
            dict.len(),
            ZSTD_CParamMode_e::ZSTD_cpm_createCDict,
        );
    }

    cctxParams.cParams = cParams;
    use crate::compress::match_state::ZSTD_resolveRowMatchFinderMode;
    cctxParams.useRowMatchFinder =
        ZSTD_resolveRowMatchFinderMode(cctxParams.useRowMatchFinder, &cParams);
    let mut cdict = ZSTD_createCDict_advanced_internal(
        dict.len(),
        dictLoadMethod,
        cctxParams.cParams,
        cctxParams.useRowMatchFinder,
        cctxParams.enableDedicatedDictSearch,
        cctxParams.customMem,
    )?;
    cdict.compressionLevel = cctxParams.compressionLevel;
    let rc = ZSTD_initCDict_internal(
        &mut cdict,
        dict,
        dictLoadMethod,
        dictContentType,
        cctxParams,
    );
    if ERR_isError(rc) {
        return None;
    }
    Some(cdict)
}

/// Port of `ZSTD_estimateCDictSize_advanced`. Upstream sums the
/// workspace needed for the CDict struct, HUF scratch, a sized match
/// state, and optionally an owned dictionary copy.
pub fn ZSTD_estimateCDictSize_advanced(
    dictSize: usize,
    cParams: crate::compress::match_state::ZSTD_compressionParameters,
    dictLoadMethod: crate::decompress::zstd_ddict::ZSTD_dictLoadMethod_e,
) -> usize {
    use crate::compress::huf_compress::HUF_WORKSPACE_SIZE;
    use crate::compress::match_state::ZSTD_resolveRowMatchFinderMode;
    use crate::decompress::zstd_ddict::ZSTD_dictLoadMethod_e;
    let useRowMatchFinder = ZSTD_resolveRowMatchFinderMode(
        crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_auto,
        &cParams,
    );
    let matchStateSize = ZSTD_sizeof_matchState(&cParams, useRowMatchFinder, true, false);
    let owned_dict = if dictLoadMethod == ZSTD_dictLoadMethod_e::ZSTD_dlm_byRef {
        0
    } else {
        dictSize
    };
    core::mem::size_of::<ZSTD_CDict>() + HUF_WORKSPACE_SIZE + matchStateSize + owned_dict
}

/// Port of `ZSTD_estimateCDictSize`. Wrapper: picks cParams for
/// `(level, dictSize)` in create-CDict mode, then calls the advanced
/// variant.
pub fn ZSTD_estimateCDictSize(dictSize: usize, compressionLevel: i32) -> usize {
    use crate::decompress::zstd_ddict::ZSTD_dictLoadMethod_e;
    use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;
    let cParams = ZSTD_getCParams_internal(
        compressionLevel,
        ZSTD_CONTENTSIZE_UNKNOWN,
        dictSize,
        ZSTD_CParamMode_e::ZSTD_cpm_createCDict,
    );
    ZSTD_estimateCDictSize_advanced(dictSize, cParams, ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy)
}

/// Rust-only helper: deeply sums the heap-allocated bytes owned by a
/// `ZSTD_MatchState_t` (tables, dict content, recursively the attached
/// dictMatchState). Used by `ZSTD_sizeof_CCtx` / `ZSTD_sizeof_CDict`.
fn ZSTD_sizeof_matchState_allocated(ms: &crate::compress::match_state::ZSTD_MatchState_t) -> usize {
    let mut sz = 0;
    sz += ms.dictContent.capacity();
    sz += ms.hashTable.capacity() * core::mem::size_of::<u32>();
    sz += ms.hashTable3.capacity() * core::mem::size_of::<u32>();
    sz += ms.tagTable.capacity();
    sz += ms.chainTable.capacity() * core::mem::size_of::<u32>();
    sz += ZSTD_sizeof_optState_allocated(&ms.opt);
    if let Some(ls) = ms.ldmSeqStore.as_ref() {
        sz += ZSTD_sizeof_rawSeqStore_allocated(ls);
    }
    if let Some(dms) = ms.dictMatchState.as_ref() {
        sz += core::mem::size_of::<crate::compress::match_state::ZSTD_MatchState_t>();
        sz += ZSTD_sizeof_matchState_allocated(dms);
    }
    sz
}

fn ZSTD_sizeof_optState_allocated(opt: &crate::compress::zstd_opt::optState_t) -> usize {
    opt.litFreq.capacity() * core::mem::size_of::<u32>()
        + opt.litLengthFreq.capacity() * core::mem::size_of::<u32>()
        + opt.matchLengthFreq.capacity() * core::mem::size_of::<u32>()
        + opt.offCodeFreq.capacity() * core::mem::size_of::<u32>()
        + opt.matchTable.capacity()
            * core::mem::size_of::<crate::compress::zstd_opt::ZSTD_match_t>()
        + opt.priceTable.capacity()
            * core::mem::size_of::<crate::compress::zstd_opt::ZSTD_optimal_t>()
}

fn ZSTD_sizeof_seqStore_allocated(ss: &SeqStore_t) -> usize {
    ss.literals.capacity()
        + ss.sequences.capacity() * core::mem::size_of::<crate::compress::seq_store::SeqDef>()
        + ss.llCode.capacity()
        + ss.mlCode.capacity()
        + ss.ofCode.capacity()
}

fn ZSTD_sizeof_rawSeqStore_allocated(ss: &crate::compress::zstd_ldm::RawSeqStore_t) -> usize {
    ss.seq.capacity() * core::mem::size_of::<crate::compress::zstd_ldm::rawSeq>()
}

fn ZSTD_sizeof_ldmState_allocated(st: &crate::compress::zstd_ldm::ldmState_t) -> usize {
    st.hashTable.capacity() * core::mem::size_of::<crate::compress::zstd_ldm::ldmEntry_t>()
        + st.bucketOffsets.capacity()
}

fn ZSTD_sizeof_fseCTables_allocated(fse: &ZSTD_fseCTables_t) -> usize {
    (fse.offcodeCTable.capacity()
        + fse.matchlengthCTable.capacity()
        + fse.litlengthCTable.capacity())
        * core::mem::size_of::<FSE_CTable>()
}

fn ZSTD_sizeof_entropyCTables_allocated(entropy: &ZSTD_entropyCTables_t) -> usize {
    entropy.huf.CTable.capacity() * core::mem::size_of::<HUF_CElt>()
        + ZSTD_sizeof_fseCTables_allocated(&entropy.fse)
}

fn ZSTD_sizeof_blockSplitCtx_allocated(ctx: &ZSTD_blockSplitCtx) -> usize {
    ZSTD_sizeof_seqStore_allocated(&ctx.fullSeqStoreChunk)
        + ZSTD_sizeof_seqStore_allocated(&ctx.firstHalfSeqStore)
        + ZSTD_sizeof_seqStore_allocated(&ctx.secondHalfSeqStore)
        + ZSTD_sizeof_seqStore_allocated(&ctx.currSeqStore)
        + ZSTD_sizeof_seqStore_allocated(&ctx.nextSeqStore)
}

fn ZSTD_sizeof_CDict_allocated(cdict: &ZSTD_CDict) -> usize {
    cdict.dictContent.capacity()
        + ZSTD_sizeof_entropyCTables_allocated(&cdict.entropy)
        + ZSTD_sizeof_matchState_allocated(&cdict.matchState)
}

/// Port of `ZSTD_sizeof_CDict`. Reports the currently allocated CDict
/// object plus its owned dictionary bytes and seeded match-state
/// tables.
pub fn ZSTD_sizeof_CDict(cdict: &ZSTD_CDict) -> usize {
    core::mem::size_of::<ZSTD_CDict>() + ZSTD_sizeof_CDict_allocated(cdict)
}

/// Port of `ZSTD_sizeof_mtctx`. Returns the currently attached MT
/// thread-pool footprint when one is referenced through the CCtx,
/// otherwise 0.
#[inline]
pub fn ZSTD_sizeof_mtctx(cctx: &ZSTD_CCtx) -> usize {
    cctx.mtctxSizeHint
}

/// Port of `ZSTD_sizeof_CCtx`. Walks the CCtx's owned `Vec`s and
/// returns a total byte footprint. Approximate — `Vec::capacity()`
/// may exceed actual usage due to allocator rounding.
pub fn ZSTD_sizeof_CCtx(cctx: &ZSTD_CCtx) -> usize {
    let mut sz = core::mem::size_of::<ZSTD_CCtx>();
    if let Some(ms) = cctx.ms.as_ref() {
        sz += ZSTD_sizeof_matchState_allocated(ms);
    }
    if let Some(ss) = cctx.seqStore.as_ref() {
        sz += ZSTD_sizeof_seqStore_allocated(ss);
    }
    sz += ZSTD_sizeof_entropyCTables_allocated(&cctx.prevEntropy);
    sz += ZSTD_sizeof_entropyCTables_allocated(&cctx.nextEntropy);
    if let Some(ctx) = cctx.blockSplitCtx.as_ref() {
        sz += ZSTD_sizeof_blockSplitCtx_allocated(ctx);
    }
    if let Some(es) = cctx.externalMatchStore.as_ref() {
        sz += ZSTD_sizeof_rawSeqStore_allocated(es);
    }
    if let Some(st) = cctx.ldmState.as_ref() {
        sz += ZSTD_sizeof_ldmState_allocated(st);
    }
    sz += ZSTD_sizeof_rawSeqStore_allocated(&cctx.ldmSequences);
    if let Some(cdict) = cctx.stream_cdict.as_ref() {
        sz += ZSTD_sizeof_CDict_allocated(cdict);
    }
    sz += cctx.stream_in_buffer.capacity();
    sz += cctx.stream_out_buffer.capacity();
    sz += cctx.stream_dict.capacity();
    sz += cctx.stream_dict_original.capacity();
    sz += ZSTD_sizeof_mtctx(cctx);
    sz
}

/// Port of `ZSTD_sizeof_CStream`. Alias for `ZSTD_sizeof_CCtx`.
pub fn ZSTD_sizeof_CStream(zcs: &ZSTD_CStream) -> usize {
    ZSTD_sizeof_CCtx(zcs)
}

/// Port of `ZSTD_compress_usingCDict`. Uses the pre-loaded dict +
/// level to compress. Functionally equivalent to
/// `ZSTD_compress_usingDict` with the CDict's captured level.
pub fn ZSTD_compress_usingCDict(
    cctx: &mut ZSTD_CCtx,
    dst: &mut [u8],
    src: &[u8],
    cdict: &ZSTD_CDict,
) -> usize {
    ZSTD_compress_usingCDict_advanced(cctx, dst, src, cdict, cctx.requestedParams.fParams)
}

/// Port of `ZSTD_compress_usingCDict_internal` (`zstd_compress.c:5911`).
/// Internal helper: runs
/// `ZSTD_compressBegin_usingCDict_internal(..., pledgedSrcSize=src.len())`
/// then `ZSTD_compressEnd_public`.
pub fn ZSTD_compress_usingCDict_internal(
    cctx: &mut ZSTD_CCtx,
    dst: &mut [u8],
    src: &[u8],
    cdict: &ZSTD_CDict,
    fParams: ZSTD_FrameParameters,
) -> usize {
    let saved_format = cctx.format;
    cctx.format = crate::decompress::zstd_decompress::ZSTD_format_e::ZSTD_f_zstd1;
    let rc = ZSTD_compressBegin_usingCDict_internal(cctx, cdict, fParams, src.len() as u64);
    if ERR_isError(rc) {
        cctx.format = saved_format;
        return rc;
    }
    let result = ZSTD_compressEnd_public(cctx, dst, src);
    cctx.format = saved_format;
    result
}

/// Port of `ZSTD_compress_usingCDict_advanced`. Compress with a
/// pre-built CDict using caller-supplied `fParams`. Upstream
/// (zstd_compress.c:5923) forwards to
/// `ZSTD_compress_usingCDict_internal`.
pub fn ZSTD_compress_usingCDict_advanced(
    cctx: &mut ZSTD_CCtx,
    dst: &mut [u8],
    src: &[u8],
    cdict: &ZSTD_CDict,
    fParams: ZSTD_FrameParameters,
) -> usize {
    ZSTD_compress_usingCDict_internal(cctx, dst, src, cdict, fParams)
}

/// Port of `ZSTD_compress_usingDict`. Compresses `src` using the same
/// dictionary-loading path as the rest of the begin/end API surface:
/// raw-content dicts provide history, while magic-prefixed
/// zstd-format dicts also seed entropy / repcodes through
/// `ZSTD_compressBegin_internal()`.
///
/// The resulting frame omits the dictionary bytes and declares
/// `src.len()` as the content size, so the decoder side still needs
/// the same dict via `ZSTD_decompress_usingDict` or an equivalent
/// DCtx dictionary-loading path.
pub fn ZSTD_compress_usingDict(
    cctx: &mut ZSTD_CCtx,
    dst: &mut [u8],
    src: &[u8],
    dict: &[u8],
    compressionLevel: i32,
) -> usize {
    let params = ZSTD_getParams_internal(
        compressionLevel,
        src.len() as u64,
        dict.len(),
        ZSTD_CParamMode_e::ZSTD_cpm_noAttachDict,
    );
    let effective_level = if compressionLevel == 0 {
        ZSTD_CLEVEL_DEFAULT
    } else {
        compressionLevel
    };
    let mut cctx_params = ZSTD_CCtx_params::default();
    ZSTD_CCtxParams_init_internal(&mut cctx_params, &params, effective_level);
    ZSTD_compress_advanced_internal(cctx, dst, src, dict, &cctx_params)
}

/// Port of `ZSTD_compress_advanced` (`zstd_compress.c:5465`). Public
/// caller-driven variant of `ZSTD_compress` that takes a full
/// `ZSTD_parameters` bundle. Validates cParams, then wraps them in a
/// `ZSTD_CCtx_params` and routes through
/// `ZSTD_compress_advanced_internal`.
pub fn ZSTD_compress_advanced(
    cctx: &mut ZSTD_CCtx,
    dst: &mut [u8],
    src: &[u8],
    dict: &[u8],
    params: ZSTD_parameters,
) -> usize {
    let rc = ZSTD_checkCParams(params.cParams);
    if ERR_isError(rc) {
        return rc;
    }
    let mut cctxParams = ZSTD_CCtx_params::default();
    ZSTD_CCtxParams_init_internal(&mut cctxParams, &params, ZSTD_NO_CLEVEL);
    ZSTD_compress_advanced_internal(cctx, dst, src, dict, &cctxParams)
}

/// Port of `ZSTD_compress_advanced_internal` (`zstd_compress.c:5482`).
/// Upstream variant that takes a `ZSTD_CCtx_params*` instead of a
/// `ZSTD_parameters`. The Rust port routes directly through
/// `ZSTD_compressBegin_internal()` with the caller-provided
/// `ZSTD_CCtx_params`.
pub fn ZSTD_compress_advanced_internal(
    cctx: &mut ZSTD_CCtx,
    dst: &mut [u8],
    src: &[u8],
    dict: &[u8],
    params: &ZSTD_CCtx_params,
) -> usize {
    let rc = ZSTD_compressBegin_internal(
        cctx,
        dict,
        crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_auto,
        None,
        params,
        src.len() as u64,
        ZSTD_buffered_policy_e::ZSTDb_not_buffered,
    );
    if ERR_isError(rc) {
        return rc;
    }
    ZSTD_compressEnd_public(cctx, dst, src)
}

/// Port of upstream's `ZSTD_MAX_INPUT_SIZE` (see `lib/zstd.h`).
///
/// Values beyond this are not supported by `ZSTD_compressBound` and
/// the helper returns an error code. Exact constants match upstream's
/// 32- vs. 64-bit limits.
pub const ZSTD_MAX_INPUT_SIZE: usize = if core::mem::size_of::<usize>() == 8 {
    0xFF00FF00FF00FF00
} else {
    0xFF00FF00
};

/// Port of the `ZSTD_COMPRESSBOUND` macro. Returns 0 when `srcSize >=
/// ZSTD_MAX_INPUT_SIZE` (i.e. the caller's input would blow the
/// upper-bound formula).
#[inline]
pub const fn ZSTD_COMPRESSBOUND(srcSize: usize) -> usize {
    if srcSize >= ZSTD_MAX_INPUT_SIZE {
        0
    } else {
        let block_margin = if srcSize < (128 << 10) {
            ((128 << 10) - srcSize) >> 11
        } else {
            0
        };
        srcSize + (srcSize >> 8) + block_margin
    }
}

/// Port of `ZSTD_compressBound`. The returned value is the maximum
/// possible size of the output buffer that a one-pass compression of
/// `srcSize` bytes can produce. Returns an error code (see
/// `ERR_isError`) when `srcSize` exceeds `ZSTD_MAX_INPUT_SIZE`.
pub fn ZSTD_compressBound(srcSize: usize) -> usize {
    let r = ZSTD_COMPRESSBOUND(srcSize);
    if r == 0 {
        return crate::common::error::ERROR(crate::common::error::ErrorCode::SrcSizeWrong);
    }
    r
}

/// Port of `ZSTD_fseCTables_t`. Three owned FSE CTables + their
/// repeat-modes. Sized identically to upstream (computed from the
/// FSELog constants + max-symbol ceilings of each stream).
#[derive(Debug, Clone)]
pub struct ZSTD_fseCTables_t {
    pub offcodeCTable: Vec<FSE_CTable>,
    pub matchlengthCTable: Vec<FSE_CTable>,
    pub litlengthCTable: Vec<FSE_CTable>,
    pub offcode_repeatMode: FSE_repeat,
    pub matchlength_repeatMode: FSE_repeat,
    pub litlength_repeatMode: FSE_repeat,
}

impl Default for ZSTD_fseCTables_t {
    fn default() -> Self {
        Self {
            offcodeCTable: vec![0u32; FSE_CTABLE_SIZE_U32(OffFSELog, MaxOff)],
            matchlengthCTable: vec![0u32; FSE_CTABLE_SIZE_U32(MLFSELog, MaxML)],
            litlengthCTable: vec![0u32; FSE_CTABLE_SIZE_U32(LLFSELog, MaxLL)],
            offcode_repeatMode: FSE_repeat::FSE_repeat_none,
            matchlength_repeatMode: FSE_repeat::FSE_repeat_none,
            litlength_repeatMode: FSE_repeat::FSE_repeat_none,
        }
    }
}

/// Upstream `ZSTD_MAX_HUF_HEADER_SIZE` (`zstd_internal.h:115`).
/// 128 = header (≤1 byte) + ≤127 byte Huffman tree description.
pub const ZSTD_MAX_HUF_HEADER_SIZE: usize = 128;

/// Upstream `ZSTD_MAX_FSE_HEADERS_SIZE` (`zstd_internal.h:117`).
/// Tight bound on serialized FSE-table bytes for LL + ML + OF
/// streams combined, rounded up to a byte.
pub const ZSTD_MAX_FSE_HEADERS_SIZE: usize = {
    use crate::decompress::zstd_decompress_block::{MaxLL, MaxML, MaxOff};
    const LLFSELog: usize = 9;
    const MLFSELog: usize = 9;
    const OffFSELog: usize = 8;
    ((MaxML as usize + 1) * MLFSELog
        + (MaxLL as usize + 1) * LLFSELog
        + (MaxOff as usize + 1) * OffFSELog)
        .div_ceil(8)
};

/// Port of `ZSTD_hufCTablesMetadata_t` (`zstd_compress_internal.h:154`).
/// Populated by `ZSTD_buildBlockEntropyStats_literals`. Carries the
/// chosen literals-block type + serialized HUF tree description.
#[derive(Debug, Clone, Copy)]
pub struct ZSTD_hufCTablesMetadata_t {
    pub hType: SymbolEncodingType_e,
    pub hufDesBuffer: [u8; ZSTD_MAX_HUF_HEADER_SIZE],
    pub hufDesSize: usize,
}

impl Default for ZSTD_hufCTablesMetadata_t {
    fn default() -> Self {
        Self {
            hType: SymbolEncodingType_e::default(),
            hufDesBuffer: [0u8; ZSTD_MAX_HUF_HEADER_SIZE],
            hufDesSize: 0,
        }
    }
}

/// Port of `ZSTD_fseCTablesMetadata_t` (`zstd_compress_internal.h:165`).
/// Populated by `ZSTD_buildBlockEntropyStats_sequences`. Holds
/// per-symbol encoding modes for LL/OF/ML + serialized FSE tables.
#[derive(Debug, Clone, Copy)]
pub struct ZSTD_fseCTablesMetadata_t {
    pub llType: SymbolEncodingType_e,
    pub ofType: SymbolEncodingType_e,
    pub mlType: SymbolEncodingType_e,
    pub fseTablesBuffer: [u8; ZSTD_MAX_FSE_HEADERS_SIZE],
    pub fseTablesSize: usize,
    /// Carries upstream's 1.3.4 accounting hack — see
    /// `ZSTD_entropyCompressSeqStore_internal` for context.
    pub lastCountSize: usize,
}

impl Default for ZSTD_fseCTablesMetadata_t {
    fn default() -> Self {
        Self {
            llType: SymbolEncodingType_e::default(),
            ofType: SymbolEncodingType_e::default(),
            mlType: SymbolEncodingType_e::default(),
            fseTablesBuffer: [0u8; ZSTD_MAX_FSE_HEADERS_SIZE],
            fseTablesSize: 0,
            lastCountSize: 0,
        }
    }
}

/// Port of `ZSTD_entropyCTablesMetadata_t` (`zstd_compress_internal.h:174`).
/// Bundles HUF + FSE metadata together.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZSTD_entropyCTablesMetadata_t {
    pub hufMetadata: ZSTD_hufCTablesMetadata_t,
    pub fseMetadata: ZSTD_fseCTablesMetadata_t,
}

/// Port of `ZSTD_symbolEncodingTypeStats_t` — the return type of
/// `ZSTD_buildSequencesStatistics`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZSTD_symbolEncodingTypeStats_t {
    pub LLtype: SymbolEncodingType_e,
    pub Offtype: SymbolEncodingType_e,
    pub MLtype: SymbolEncodingType_e,
    /// Bytes written into the caller-supplied `dst` (NCount headers +
    /// RLE-symbol byte, summed across the 3 streams). On error, holds
    /// the zstd error code instead.
    pub size: usize,
    /// Size of the **last** NCount written, or 0 if none used a
    /// `set_compressed` encoding (upstream's "1.3.4 bug" accounting).
    pub lastCountSize: usize,
    pub longOffsets: i32,
}

/// Port of `ZSTD_estimateBlockSize_literal` (`zstd_compress.c:3853`).
/// Given a literals buffer + HUF state + block-metadata, returns the
/// byte count the encoder would emit for the literals block.
///
/// Dispatches on `hufMetadata.hType`:
///   - `set_basic` → raw bytes, size = litSize
///   - `set_rle` → 1 byte
///   - `set_compressed` / `set_repeat` → histogram + HUF estimate +
///     optional tree description bytes + optional 6-byte jump table
///     for the 4-stream case.
pub fn ZSTD_estimateBlockSize_literal(
    literals: &[u8],
    huf: &ZSTD_hufCTables_t,
    hufMetadata: &ZSTD_hufCTablesMetadata_t,
    count: &mut [u32],
    workspace: &mut [u32],
    writeEntropy: bool,
) -> usize {
    use crate::compress::hist::HIST_count_wksp;
    use crate::compress::huf_compress::{HUF_estimateCompressedSize, HUF_SYMBOLVALUE_MAX};
    use crate::decompress::zstd_decompress_block::SymbolEncodingType_e;

    let litSize = literals.len();
    let literalSectionHeaderSize = 3 + (litSize >= 1024) as usize + (litSize >= 16 * 1024) as usize;
    let singleStream = litSize < 256;

    match hufMetadata.hType {
        SymbolEncodingType_e::set_basic => litSize,
        SymbolEncodingType_e::set_rle => 1,
        SymbolEncodingType_e::set_compressed | SymbolEncodingType_e::set_repeat => {
            let mut maxSymbolValue = HUF_SYMBOLVALUE_MAX;
            let count_len = maxSymbolValue as usize + 1;
            if count.len() < count_len {
                return litSize;
            }
            let count = &mut count[..count_len];
            count.fill(0);
            let largest = HIST_count_wksp(count, &mut maxSymbolValue, literals, workspace);
            if crate::common::error::ERR_isError(largest) {
                return litSize;
            }
            let mut cLitSizeEstimate =
                HUF_estimateCompressedSize(&huf.CTable, count, maxSymbolValue);
            if writeEntropy {
                cLitSizeEstimate += hufMetadata.hufDesSize;
            }
            if !singleStream {
                cLitSizeEstimate += 6;
            }
            cLitSizeEstimate + literalSectionHeaderSize
        }
    }
}

/// Port of `ZSTD_estimateBlockSize_symbolType` (`zstd_compress.c:3880`).
/// Estimates the byte size of one of the three FSE-compressed code
/// streams (of / ml / ll). Dispatches on the encoding mode:
///   - `set_basic` → cross-entropy against default distribution
///   - `set_rle` → 0 extra bits (1 symbol byte in the header)
///   - `set_compressed` / `set_repeat` → FSE cost using the table
///
/// Adds the per-symbol extra-bits cost from `additionalBits`, or
/// treats the code itself as the bit count when `additionalBits` is
/// `None` (offset streams).
pub fn ZSTD_estimateBlockSize_symbolType(
    encType: crate::decompress::zstd_decompress_block::SymbolEncodingType_e,
    codeTable: &[u8],
    maxCode: u32,
    fseCTable: &[crate::compress::fse_compress::FSE_CTable],
    additionalBits: Option<&[u8]>,
    defaultNorm: &[i16],
    defaultNormLog: u32,
    count: &mut [u32],
    workspace: &mut [u32],
) -> usize {
    use crate::compress::hist::HIST_countFast_wksp;
    use crate::compress::zstd_compress_sequences::{ZSTD_crossEntropyCost, ZSTD_fseBitCost};
    use crate::decompress::zstd_decompress_block::SymbolEncodingType_e;

    let nbSeq = codeTable.len();
    let mut max = maxCode;
    let count_len = maxCode as usize + 1;
    if count.len() < count_len {
        return nbSeq * 10;
    }
    let count = &mut count[..count_len];
    count.fill(0);
    let _ = HIST_countFast_wksp(count, &mut max, codeTable, workspace);

    let mut cSymbolTypeSizeEstimateInBits: usize = match encType {
        SymbolEncodingType_e::set_basic => {
            ZSTD_crossEntropyCost(defaultNorm, defaultNormLog, count, max)
        }
        SymbolEncodingType_e::set_rle => 0,
        SymbolEncodingType_e::set_compressed | SymbolEncodingType_e::set_repeat => {
            ZSTD_fseBitCost(fseCTable, count, max)
        }
    };
    if crate::common::error::ERR_isError(cSymbolTypeSizeEstimateInBits) {
        return nbSeq * 10;
    }
    for &c in codeTable {
        if let Some(bits) = additionalBits {
            cSymbolTypeSizeEstimateInBits += bits[c as usize] as usize;
        } else {
            cSymbolTypeSizeEstimateInBits += c as usize;
        }
    }
    cSymbolTypeSizeEstimateInBits >> 3
}

/// Port of `ZSTD_estimateBlockSize_sequences` (`zstd_compress.c:3918`).
/// Estimates the byte size of the full sequences section: fixed-size
/// header (1-4 bytes) + per-stream estimates for OF/LL/ML + optional
/// serialized FSE tables.
pub fn ZSTD_estimateBlockSize_sequences(
    ofCodeTable: &[u8],
    llCodeTable: &[u8],
    mlCodeTable: &[u8],
    nbSeq: usize,
    fseTables: &ZSTD_fseCTables_t,
    fseMetadata: &ZSTD_fseCTablesMetadata_t,
    countWorkspace: &mut [u32],
    workspace: &mut [u32],
    writeEntropy: bool,
) -> usize {
    use crate::decompress::zstd_decompress_block::{
        DefaultMaxOff, LL_bits, LL_defaultNorm, LL_defaultNormLog, ML_bits, ML_defaultNorm,
        ML_defaultNormLog, MaxLL, MaxML, MaxOff, OF_defaultNorm, OF_defaultNormLog,
    };

    const LONGNBSEQ: usize = 0x7F00;
    let seqHeaderSize = 1 + 1 + (nbSeq >= 128) as usize + (nbSeq >= LONGNBSEQ) as usize;

    let mut total: usize = 0;
    total += ZSTD_estimateBlockSize_symbolType(
        fseMetadata.ofType,
        ofCodeTable,
        MaxOff,
        &fseTables.offcodeCTable,
        None,
        &OF_defaultNorm,
        OF_defaultNormLog,
        countWorkspace,
        workspace,
    );
    total += ZSTD_estimateBlockSize_symbolType(
        fseMetadata.llType,
        llCodeTable,
        MaxLL,
        &fseTables.litlengthCTable,
        Some(&LL_bits),
        &LL_defaultNorm,
        LL_defaultNormLog,
        countWorkspace,
        workspace,
    );
    total += ZSTD_estimateBlockSize_symbolType(
        fseMetadata.mlType,
        mlCodeTable,
        MaxML,
        &fseTables.matchlengthCTable,
        Some(&ML_bits),
        &ML_defaultNorm,
        ML_defaultNormLog,
        countWorkspace,
        workspace,
    );
    let _ = DefaultMaxOff;
    if writeEntropy {
        total += fseMetadata.fseTablesSize;
    }
    total + seqHeaderSize
}

/// Port of `ZSTD_estimateBlockSize` (`zstd_compress.c:3947`). Combines
/// literal + sequences estimates + the 3-byte block header into a
/// full compressed-block size estimate.
pub fn ZSTD_estimateBlockSize(
    literals: &[u8],
    ofCodeTable: &[u8],
    llCodeTable: &[u8],
    mlCodeTable: &[u8],
    nbSeq: usize,
    entropy: &ZSTD_entropyCTables_t,
    entropyMetadata: &ZSTD_entropyCTablesMetadata_t,
    countWorkspace: &mut [u32],
    literalCountWorkspace: &mut [u32],
    workspace: &mut [u32],
    writeLitEntropy: bool,
    writeSeqEntropy: bool,
) -> usize {
    use crate::decompress::zstd_decompress_block::ZSTD_blockHeaderSize;
    let literalsSize = ZSTD_estimateBlockSize_literal(
        literals,
        &entropy.huf,
        &entropyMetadata.hufMetadata,
        literalCountWorkspace,
        workspace,
        writeLitEntropy,
    );
    let seqSize = ZSTD_estimateBlockSize_sequences(
        ofCodeTable,
        llCodeTable,
        mlCodeTable,
        nbSeq,
        &entropy.fse,
        &entropyMetadata.fseMetadata,
        countWorkspace,
        workspace,
        writeSeqEntropy,
    );
    literalsSize + seqSize + ZSTD_blockHeaderSize
}

/// Port of `ZSTD_buildEntropyStatisticsAndEstimateSubBlockSize`
/// (`zstd_compress.c:3972`). Builds fresh entropy metadata for the
/// provided `seqStore`, then immediately feeds it into
/// `ZSTD_estimateBlockSize`.
///
/// Upstream stores the metadata in `zc->blockSplitCtx.entropyMetadata`
/// and carves two workspaces out of `tmpWorkspace`. The Rust port
/// does carry `blockSplitCtx`, but this helper still computes the
/// entropy metadata locally and uses owned scratch buffers for the
/// already-ported histogram / FSE builders.
pub fn ZSTD_buildEntropyStatisticsAndEstimateSubBlockSize(
    seqStore: &mut SeqStore_t,
    zc: &mut ZSTD_CCtx,
) -> usize {
    let blockSplitCtx = zc
        .blockSplitCtx
        .get_or_insert_with(ZSTD_blockSplitCtx::default);
    blockSplitCtx.entropyMetadata = ZSTD_entropyCTablesMetadata_t::default();
    let rc = ZSTD_buildBlockEntropyStats(
        seqStore,
        &zc.prevEntropy,
        &mut zc.nextEntropy,
        &zc.appliedParams,
        &mut blockSplitCtx.entropyMetadata,
        &mut blockSplitCtx.entropyScratch,
    );
    if ERR_isError(rc) {
        return rc;
    }
    let entropyMetadata = &blockSplitCtx.entropyMetadata;
    ZSTD_estimateBlockSize(
        &seqStore.literals,
        &seqStore.ofCode,
        &seqStore.llCode,
        &seqStore.mlCode,
        seqStore.sequences.len(),
        &zc.nextEntropy,
        entropyMetadata,
        &mut blockSplitCtx.entropyScratch.seqCountWorkspace,
        &mut blockSplitCtx.entropyScratch.hufCount,
        &mut blockSplitCtx.entropyScratch.blockWorkspaceU32,
        entropyMetadata.hufMetadata.hType == SymbolEncodingType_e::set_compressed,
        true,
    )
}

#[derive(Debug)]
struct seqStoreSplits<'a> {
    splitLocations: &'a mut [u32; ZSTD_MAX_NB_BLOCK_SPLITS],
    idx: usize,
}

/// Port of `ZSTD_deriveBlockSplitsHelper` (`zstd_compress.c:4218`).
fn ZSTD_deriveBlockSplitsHelper(
    splits: &mut seqStoreSplits<'_>,
    startIdx: usize,
    endIdx: usize,
    zc: &mut ZSTD_CCtx,
    origSeqStore: &SeqStore_t,
) {
    let midIdx = (startIdx + endIdx) / 2;

    if endIdx - startIdx < MIN_SEQUENCES_BLOCK_SPLITTING || splits.idx >= ZSTD_MAX_NB_BLOCK_SPLITS {
        return;
    }

    zc.blockSplitCtx
        .get_or_insert_with(ZSTD_blockSplitCtx::default);
    let zc_ptr = zc as *mut ZSTD_CCtx;
    let (estimatedOriginalSize, estimatedFirstHalfSize, estimatedSecondHalfSize) = unsafe {
        let ctx = (*zc_ptr).blockSplitCtx.as_mut().unwrap();
        let fullSeqStoreChunk = &mut ctx.fullSeqStoreChunk as *mut SeqStore_t;
        let firstHalfSeqStore = &mut ctx.firstHalfSeqStore as *mut SeqStore_t;
        let secondHalfSeqStore = &mut ctx.secondHalfSeqStore as *mut SeqStore_t;

        ZSTD_deriveSeqStoreChunkInto(&mut *fullSeqStoreChunk, origSeqStore, startIdx, endIdx);
        let estimatedOriginalSize = ZSTD_buildEntropyStatisticsAndEstimateSubBlockSize(
            &mut *fullSeqStoreChunk,
            &mut *zc_ptr,
        );

        ZSTD_deriveSeqStoreChunkInto(&mut *firstHalfSeqStore, origSeqStore, startIdx, midIdx);
        let estimatedFirstHalfSize = ZSTD_buildEntropyStatisticsAndEstimateSubBlockSize(
            &mut *firstHalfSeqStore,
            &mut *zc_ptr,
        );

        ZSTD_deriveSeqStoreChunkInto(&mut *secondHalfSeqStore, origSeqStore, midIdx, endIdx);
        let estimatedSecondHalfSize = ZSTD_buildEntropyStatisticsAndEstimateSubBlockSize(
            &mut *secondHalfSeqStore,
            &mut *zc_ptr,
        );

        (
            estimatedOriginalSize,
            estimatedFirstHalfSize,
            estimatedSecondHalfSize,
        )
    };
    if ERR_isError(estimatedOriginalSize)
        || ERR_isError(estimatedFirstHalfSize)
        || ERR_isError(estimatedSecondHalfSize)
    {
        return;
    }
    if estimatedFirstHalfSize + estimatedSecondHalfSize < estimatedOriginalSize {
        ZSTD_deriveBlockSplitsHelper(splits, startIdx, midIdx, zc, origSeqStore);
        splits.splitLocations[splits.idx] = midIdx as u32;
        splits.idx += 1;
        ZSTD_deriveBlockSplitsHelper(splits, midIdx, endIdx, zc, origSeqStore);
    }
}

/// Port of `ZSTD_deriveBlockSplits` (`zstd_compress.c:4264`).
pub fn ZSTD_deriveBlockSplits(zc: &mut ZSTD_CCtx, nbSeq: u32) -> usize {
    let mut partitions = [0u32; ZSTD_MAX_NB_BLOCK_SPLITS];
    if nbSeq <= 4 {
        return 0;
    }
    zc.blockSplitCtx
        .get_or_insert_with(ZSTD_blockSplitCtx::default);
    let origSeqStore = match zc.seqStore.as_ref() {
        Some(seqStore) => seqStore as *const SeqStore_t,
        None => return 0,
    };
    let idx = {
        let mut splits = seqStoreSplits {
            splitLocations: &mut partitions,
            idx: 0,
        };
        unsafe {
            ZSTD_deriveBlockSplitsHelper(&mut splits, 0, nbSeq as usize, zc, &*origSeqStore);
        }
        splits.splitLocations[splits.idx] = nbSeq;
        splits.idx
    };
    zc.blockSplitCtx
        .get_or_insert_with(ZSTD_blockSplitCtx::default)
        .partitions = partitions;
    idx
}

/// Port of `ZSTD_compressSeqStore_singleBlock`
/// (`zstd_compress.c:4130`).
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_compressSeqStore_singleBlock(
    zc: &mut ZSTD_CCtx,
    seqStore: &mut SeqStore_t,
    dRep: &mut Repcodes_t,
    cRep: &mut Repcodes_t,
    dst: &mut [u8],
    src: &[u8],
    lastBlock: u32,
    isPartition: u32,
) -> usize {
    const RLE_MAX_LENGTH: usize = 25;

    let dRepOriginal = *dRep;
    if isPartition != 0 {
        ZSTD_seqStore_resolveOffCodes(dRep, cRep, seqStore, seqStore.sequences.len() as u32);
    }
    if dst.len() < ZSTD_blockHeaderSize {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }

    let disableLiteralCompression = ZSTD_literalsCompressionIsDisabled(
        zc.appliedParams.literalCompressionMode,
        zc.appliedParams.cParams.strategy,
        zc.appliedParams.cParams.targetLength,
    ) as i32;
    let mut cSeqsSize = ZSTD_entropyCompressSeqStore_wksp(
        &mut dst[ZSTD_blockHeaderSize..],
        seqStore,
        &zc.prevEntropy,
        &mut zc.nextEntropy,
        zc.appliedParams.cParams.strategy,
        disableLiteralCompression,
        src.len(),
        zc.bmi2,
        &mut zc.entropyScratch,
    );
    if ERR_isError(cSeqsSize) {
        return cSeqsSize;
    }

    if zc.isFirstBlock == 0 && cSeqsSize < RLE_MAX_LENGTH && ZSTD_isRLE(src) != 0 {
        cSeqsSize = 1;
    }

    let cSize = if cSeqsSize == 0 {
        let cSize = ZSTD_noCompressBlock(dst, src, lastBlock);
        if ERR_isError(cSize) {
            return cSize;
        }
        *dRep = dRepOriginal;
        cSize
    } else if cSeqsSize == 1 {
        if src.is_empty() {
            return ERROR(ErrorCode::Generic);
        }
        let cSize = ZSTD_rleCompressBlock(dst, src[0], src.len(), lastBlock);
        if ERR_isError(cSize) {
            return cSize;
        }
        *dRep = dRepOriginal;
        cSize
    } else {
        ZSTD_blockState_confirmRepcodesAndEntropyTables(zc);
        let cBlockHeader = lastBlock
            .wrapping_add(
                (crate::decompress::zstd_decompress_block::blockType_e::bt_compressed as u32) << 1,
            )
            .wrapping_add((cSeqsSize as u32) << 3);
        MEM_writeLE24(dst, cBlockHeader);
        ZSTD_blockHeaderSize + cSeqsSize
    };

    if zc.prevEntropy.fse.offcode_repeatMode == FSE_repeat::FSE_repeat_valid {
        zc.prevEntropy.fse.offcode_repeatMode = FSE_repeat::FSE_repeat_check;
    }
    cSize
}

/// Port of `ZSTD_compressBlock_splitBlock_internal`
/// (`zstd_compress.c:4281`).
pub fn ZSTD_compressBlock_splitBlock_internal(
    zc: &mut ZSTD_CCtx,
    dst: &mut [u8],
    src: &[u8],
    lastBlock: u32,
    nbSeq: u32,
) -> usize {
    let mut cSize = 0usize;
    let mut ip = 0usize;
    let mut op = 0usize;
    let mut srcBytesTotal = 0usize;
    let partitions;
    let numSplits = ZSTD_deriveBlockSplits(zc, nbSeq);
    partitions = zc
        .blockSplitCtx
        .as_ref()
        .map(|ctx| ctx.partitions)
        .unwrap_or([0; ZSTD_MAX_NB_BLOCK_SPLITS]);

    let mut dRep = Repcodes_t { rep: zc.prev_rep };
    let mut cRep = Repcodes_t { rep: zc.prev_rep };

    if numSplits == 0 {
        let origSeqStore = match zc.seqStore.as_ref() {
            Some(seqStore) => seqStore as *const SeqStore_t,
            None => return ERROR(ErrorCode::Generic),
        };
        zc.blockSplitCtx
            .get_or_insert_with(ZSTD_blockSplitCtx::default);
        let zc_ptr = zc as *mut ZSTD_CCtx;
        return unsafe {
            let ctx = (*zc_ptr).blockSplitCtx.as_mut().unwrap();
            ZSTD_deriveSeqStoreChunkInto(&mut ctx.currSeqStore, &*origSeqStore, 0, nbSeq as usize);
            ZSTD_compressSeqStore_singleBlock(
                &mut *zc_ptr,
                &mut ctx.currSeqStore,
                &mut dRep,
                &mut cRep,
                dst,
                src,
                lastBlock,
                0,
            )
        };
    }

    let origSeqStore = match zc.seqStore.as_ref() {
        Some(seqStore) => seqStore as *const SeqStore_t,
        None => return ERROR(ErrorCode::Generic),
    };
    zc.blockSplitCtx
        .get_or_insert_with(ZSTD_blockSplitCtx::default);
    let zc_ptr = zc as *mut ZSTD_CCtx;
    let mut currSeqStore;
    let mut nextSeqStore;
    unsafe {
        let ctx = (*zc_ptr).blockSplitCtx.as_mut().unwrap();
        currSeqStore = &mut ctx.currSeqStore as *mut SeqStore_t;
        nextSeqStore = &mut ctx.nextSeqStore as *mut SeqStore_t;
        ZSTD_deriveSeqStoreChunkInto(
            &mut *currSeqStore,
            &*origSeqStore,
            0,
            partitions[0] as usize,
        );
    }

    for i in 0..=numSplits {
        let lastPartition = (i == numSplits) as u32;
        let mut lastBlockEntireSrc = 0u32;
        let mut srcBytes = unsafe {
            ZSTD_countSeqStoreLiteralsBytes(&*currSeqStore)
                + ZSTD_countSeqStoreMatchBytes(&*currSeqStore)
        };

        srcBytesTotal += srcBytes;
        if lastPartition != 0 {
            srcBytes += src.len().saturating_sub(srcBytesTotal);
            lastBlockEntireSrc = lastBlock;
        } else {
            unsafe {
                ZSTD_deriveSeqStoreChunkInto(
                    &mut *nextSeqStore,
                    &*origSeqStore,
                    partitions[i] as usize,
                    partitions[i + 1] as usize,
                );
            }
        }

        let cSizeChunk = unsafe {
            ZSTD_compressSeqStore_singleBlock(
                &mut *zc_ptr,
                &mut *currSeqStore,
                &mut dRep,
                &mut cRep,
                &mut dst[op..],
                &src[ip..ip + srcBytes],
                lastBlockEntireSrc,
                1,
            )
        };
        if ERR_isError(cSizeChunk) {
            return cSizeChunk;
        }

        ip += srcBytes;
        op += cSizeChunk;
        cSize += cSizeChunk;
        if lastPartition == 0 {
            core::mem::swap(&mut currSeqStore, &mut nextSeqStore);
        }
    }

    zc.prev_rep = dRep.rep;
    cSize
}

/// Port of `ZSTD_compressBlock_splitBlock` (`zstd_compress.c:4358`).
pub fn ZSTD_compressBlock_splitBlock(
    zc: &mut ZSTD_CCtx,
    dst: &mut [u8],
    src: &[u8],
    lastBlock: u32,
) -> usize {
    let bss = ZSTD_buildSeqStore(zc, src);
    if ERR_isError(bss) {
        return bss;
    }
    if bss == ZSTD_BuildSeqStore_e::ZSTDbss_noCompress as usize {
        if zc.prevEntropy.fse.offcode_repeatMode == FSE_repeat::FSE_repeat_valid {
            zc.prevEntropy.fse.offcode_repeatMode = FSE_repeat::FSE_repeat_check;
        }
        return ZSTD_noCompressBlock(dst, src, lastBlock);
    }

    let nbSeq = match zc.seqStore.as_ref() {
        Some(seqStore) => seqStore.sequences.len() as u32,
        None => return ERROR(ErrorCode::Generic),
    };
    ZSTD_compressBlock_splitBlock_internal(zc, dst, src, lastBlock, nbSeq)
}

/// Rust-only helper: window-aware variant of
/// `ZSTD_compressBlock_splitBlock`. Operates on a `[dict_prefix .. src]`
/// window buffer with explicit `src_start`/`src_end` indices so the match
/// finder can address dict history without copying.
pub fn ZSTD_compressBlock_splitBlock_with_window(
    zc: &mut ZSTD_CCtx,
    dst: &mut [u8],
    window_buf: &[u8],
    src_start: usize,
    src_end: usize,
    lastBlock: u32,
) -> usize {
    let src = &window_buf[src_start..src_end];
    let bss = ZSTD_buildSeqStore_with_window(zc, window_buf, src_start, src_end);
    if ERR_isError(bss) {
        return bss;
    }
    if bss == ZSTD_BuildSeqStore_e::ZSTDbss_noCompress as usize {
        if zc.prevEntropy.fse.offcode_repeatMode == FSE_repeat::FSE_repeat_valid {
            zc.prevEntropy.fse.offcode_repeatMode = FSE_repeat::FSE_repeat_check;
        }
        return ZSTD_noCompressBlock(dst, src, lastBlock);
    }

    let nbSeq = match zc.seqStore.as_ref() {
        Some(seqStore) => seqStore.sequences.len() as u32,
        None => return ERROR(ErrorCode::Generic),
    };
    ZSTD_compressBlock_splitBlock_internal(zc, dst, src, lastBlock, nbSeq)
}

/// Port of `ZSTD_buildBlockEntropyStats_literals`
/// (`zstd_compress.c:3657`). Builds / selects the literals entropy
/// mode for a block and fills `hufMetadata`.
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_buildBlockEntropyStats_literals(
    src: &[u8],
    prevHuf: &ZSTD_hufCTables_t,
    nextHuf: &mut ZSTD_hufCTables_t,
    hufMetadata: &mut ZSTD_hufCTablesMetadata_t,
    literalsCompressionIsDisabled: bool,
    count: &mut [u32],
    histWksp: &mut [u32],
    hufBuildWksp: &mut [u32],
    hufWriteWksp: &mut [u8],
    _hufFlags: i32,
) -> usize {
    use crate::compress::hist::HIST_count_wksp;
    use crate::compress::huf_compress::{
        HUF_buildCTable_wksp, HUF_estimateCompressedSize, HUF_optimalTableLog, HUF_validateCTable,
        HUF_writeCTable_wksp, HUF_SYMBOLVALUE_MAX,
    };
    use crate::compress::zstd_compress_literals::{HUF_repeat, LitHufLog};

    const COMPRESS_LITERALS_SIZE_MIN: usize = 63;

    let srcSize = src.len();
    let mut maxSymbolValue = HUF_SYMBOLVALUE_MAX;
    let mut huffLog = LitHufLog;
    let mut repeat = prevHuf.repeatMode;
    count.fill(0);
    histWksp.fill(0);

    nextHuf.clone_from(prevHuf);

    if literalsCompressionIsDisabled {
        hufMetadata.hType = SymbolEncodingType_e::set_basic;
        return 0;
    }

    let minLitSize = if prevHuf.repeatMode == HUF_repeat::HUF_repeat_valid {
        6
    } else {
        COMPRESS_LITERALS_SIZE_MIN
    };
    if srcSize <= minLitSize {
        hufMetadata.hType = SymbolEncodingType_e::set_basic;
        return 0;
    }

    let largest = HIST_count_wksp(count, &mut maxSymbolValue, src, histWksp);
    if ERR_isError(largest) {
        return largest;
    }
    if largest == srcSize {
        hufMetadata.hType = SymbolEncodingType_e::set_rle;
        return 0;
    }
    if largest <= (srcSize >> 7) + 4 {
        hufMetadata.hType = SymbolEncodingType_e::set_basic;
        return 0;
    }

    if repeat == HUF_repeat::HUF_repeat_check
        && !HUF_validateCTable(&prevHuf.CTable, count, maxSymbolValue)
    {
        repeat = HUF_repeat::HUF_repeat_none;
    }

    nextHuf.CTable.fill(0);
    huffLog = HUF_optimalTableLog(huffLog, srcSize, maxSymbolValue);
    let maxBits = HUF_buildCTable_wksp(
        &mut nextHuf.CTable,
        count,
        maxSymbolValue,
        huffLog,
        hufBuildWksp,
    );
    if ERR_isError(maxBits) {
        return maxBits;
    }
    huffLog = maxBits as u32;

    let newCSize = HUF_estimateCompressedSize(&nextHuf.CTable, count, maxSymbolValue);
    let hSize = HUF_writeCTable_wksp(
        &mut hufMetadata.hufDesBuffer,
        &nextHuf.CTable,
        maxSymbolValue,
        huffLog,
        hufWriteWksp,
    );
    if ERR_isError(hSize) {
        return hSize;
    }

    if repeat != HUF_repeat::HUF_repeat_none {
        let oldCSize = HUF_estimateCompressedSize(&prevHuf.CTable, count, maxSymbolValue);
        if oldCSize < srcSize && (oldCSize <= hSize + newCSize || hSize + 12 >= srcSize) {
            nextHuf.clone_from(prevHuf);
            hufMetadata.hType = SymbolEncodingType_e::set_repeat;
            return 0;
        }
    }
    if newCSize + hSize >= srcSize {
        nextHuf.clone_from(prevHuf);
        hufMetadata.hType = SymbolEncodingType_e::set_basic;
        return 0;
    }

    hufMetadata.hType = SymbolEncodingType_e::set_compressed;
    nextHuf.repeatMode = HUF_repeat::HUF_repeat_check;
    hufMetadata.hufDesSize = hSize;
    hSize
}

/// Port of `ZSTD_buildDummySequencesStatistics` (`zstd_compress.c:3767`).
/// Fast-path that skips `ZSTD_buildSequencesStatistics` when the block
/// has no sequences — emits `set_basic` for all three streams and
/// clears `nextEntropy` repeatModes so the next block must re-emit
/// fresh tables. Zero output bytes because default tables are
/// implicit in the zstd format.
pub fn ZSTD_buildDummySequencesStatistics(
    nextEntropy: &mut ZSTD_fseCTables_t,
) -> ZSTD_symbolEncodingTypeStats_t {
    use crate::compress::zstd_compress_sequences::FSE_repeat;
    nextEntropy.litlength_repeatMode = FSE_repeat::FSE_repeat_none;
    nextEntropy.offcode_repeatMode = FSE_repeat::FSE_repeat_none;
    nextEntropy.matchlength_repeatMode = FSE_repeat::FSE_repeat_none;
    ZSTD_symbolEncodingTypeStats_t {
        LLtype: SymbolEncodingType_e::set_basic,
        Offtype: SymbolEncodingType_e::set_basic,
        MLtype: SymbolEncodingType_e::set_basic,
        size: 0,
        lastCountSize: 0,
        longOffsets: 0,
    }
}

/// Port of `ZSTD_buildBlockEntropyStats_sequences`
/// (`zstd_compress.c:3786`).
pub fn ZSTD_buildBlockEntropyStats_sequences(
    seqStorePtr: &mut SeqStore_t,
    prevEntropy: &ZSTD_fseCTables_t,
    nextEntropy: &mut ZSTD_fseCTables_t,
    cctxParams: &ZSTD_CCtx_params,
    fseMetadata: &mut ZSTD_fseCTablesMetadata_t,
    workspace_u32: &mut [u32],
    entropyWorkspace: &mut [u8],
    histWorkspace: &mut [u32],
) -> usize {
    let strategy = cctxParams.cParams.strategy;
    let nbSeq = seqStorePtr.sequences.len();
    let stats = if nbSeq != 0 {
        ZSTD_buildSequencesStatistics_wksp(
            seqStorePtr,
            nbSeq,
            prevEntropy,
            nextEntropy,
            &mut fseMetadata.fseTablesBuffer,
            strategy,
            workspace_u32,
            entropyWorkspace,
            histWorkspace,
        )
    } else {
        ZSTD_buildDummySequencesStatistics(nextEntropy)
    };
    if ERR_isError(stats.size) {
        return stats.size;
    }
    fseMetadata.llType = stats.LLtype;
    fseMetadata.ofType = stats.Offtype;
    fseMetadata.mlType = stats.MLtype;
    fseMetadata.lastCountSize = stats.lastCountSize;
    fseMetadata.fseTablesSize = stats.size;
    stats.size
}

/// Port of `ZSTD_buildBlockEntropyStats`
/// (`zstd_compress.c:3822`).
pub fn ZSTD_buildBlockEntropyStats(
    seqStorePtr: &mut SeqStore_t,
    prevEntropy: &ZSTD_entropyCTables_t,
    nextEntropy: &mut ZSTD_entropyCTables_t,
    cctxParams: &ZSTD_CCtx_params,
    entropyMetadata: &mut ZSTD_entropyCTablesMetadata_t,
    scratch: &mut ZSTD_entropyScratch,
) -> usize {
    use crate::compress::zstd_compress_literals::ZSTD_literalsCompressionIsDisabled;
    use crate::compress::zstd_compress_sequences::ZSTD_btultra;
    use crate::decompress::huf_decompress::HUF_flags_optimalDepth;

    let huf_useOptDepth = cctxParams.cParams.strategy >= ZSTD_btultra;
    let hufFlags = if huf_useOptDepth {
        HUF_flags_optimalDepth
    } else {
        0
    };

    let hufDesSize = ZSTD_buildBlockEntropyStats_literals(
        &seqStorePtr.literals,
        &prevEntropy.huf,
        &mut nextEntropy.huf,
        &mut entropyMetadata.hufMetadata,
        ZSTD_literalsCompressionIsDisabled(
            cctxParams.literalCompressionMode,
            cctxParams.cParams.strategy,
            cctxParams.cParams.targetLength,
        ),
        &mut scratch.hufCount,
        &mut scratch.hufHistWorkspace,
        &mut scratch.hufBuildWorkspace,
        &mut scratch.hufWriteWorkspace,
        hufFlags,
    );
    if ERR_isError(hufDesSize) {
        return hufDesSize;
    }
    entropyMetadata.hufMetadata.hufDesSize = hufDesSize;

    let fseTableSize = ZSTD_buildBlockEntropyStats_sequences(
        seqStorePtr,
        &prevEntropy.fse,
        &mut nextEntropy.fse,
        cctxParams,
        &mut entropyMetadata.fseMetadata,
        &mut scratch.blockWorkspaceU32,
        &mut scratch.entropyWorkspace,
        &mut scratch.seqHistWorkspace,
    );
    if ERR_isError(fseTableSize) {
        return fseTableSize;
    }
    entropyMetadata.fseMetadata.fseTablesSize = fseTableSize;
    0
}

/// Port of `ZSTD_buildSequencesStatistics`. Runs histogram →
/// `ZSTD_selectEncodingType` → `ZSTD_buildCTable` for each of the
/// three sequence streams (LL → Off → ML), writing NCount headers /
/// RLE bytes into `dst` and populating `nextEntropy`'s CTables +
/// repeat-modes in place. `countWorkspace` is the histogram target
/// (≥ MaxSeq+1 u32s); `entropyWorkspace` is the FSE-build workspace
/// bytes (sized per upstream's ENTROPY_WORKSPACE_SIZE).
///
/// Rust signature note: upstream takes `prevEntropy` as a `const*`;
/// we take `&ZSTD_fseCTables_t` and reuse by-value clones of the
/// repeatModes (they're `Copy`-able enums).
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_buildSequencesStatistics_wksp(
    seqStore: &mut SeqStore_t,
    nbSeq: usize,
    prevEntropy: &ZSTD_fseCTables_t,
    nextEntropy: &mut ZSTD_fseCTables_t,
    dst: &mut [u8],
    strategy: u32,
    countWorkspace: &mut [u32],
    entropyWorkspace: &mut [u8],
    histWorkspace: &mut [u32],
) -> ZSTD_symbolEncodingTypeStats_t {
    let mut stats = ZSTD_symbolEncodingTypeStats_t {
        longOffsets: ZSTD_seqToCodes(seqStore),
        ..Default::default()
    };
    debug_assert!(nbSeq != 0);

    histWorkspace.fill(0);
    let mut op = 0usize;

    // --- Literal Lengths ---
    {
        let mut max = MaxLL;
        let mostFrequent = HIST_countFast_wksp(
            countWorkspace,
            &mut max,
            &seqStore.llCode[..nbSeq],
            histWorkspace,
        );
        nextEntropy.litlength_repeatMode = prevEntropy.litlength_repeatMode;
        stats.LLtype = ZSTD_selectEncodingType(
            &mut nextEntropy.litlength_repeatMode,
            countWorkspace,
            max,
            mostFrequent,
            nbSeq,
            LLFSELog,
            &prevEntropy.litlengthCTable,
            &LL_defaultNorm,
            LL_defaultNormLog,
            ZSTD_DefaultPolicy_e::ZSTD_defaultAllowed,
            strategy,
        );
        let countSize = ZSTD_buildCTable(
            &mut dst[op..],
            &mut nextEntropy.litlengthCTable,
            LLFSELog,
            stats.LLtype,
            countWorkspace,
            max,
            &seqStore.llCode[..nbSeq],
            nbSeq,
            &LL_defaultNorm,
            LL_defaultNormLog,
            MaxLL,
            &prevEntropy.litlengthCTable,
            prevEntropy.litlengthCTable.len() * 4,
            entropyWorkspace,
        );
        if ERR_isError(countSize) {
            stats.size = countSize;
            return stats;
        }
        if stats.LLtype == SymbolEncodingType_e::set_compressed {
            stats.lastCountSize = countSize;
        }
        op += countSize;
    }

    // --- Offsets ---
    {
        let mut max = MaxOff;
        let mostFrequent = HIST_countFast_wksp(
            countWorkspace,
            &mut max,
            &seqStore.ofCode[..nbSeq],
            histWorkspace,
        );
        let defaultPolicy = if max <= DefaultMaxOff {
            ZSTD_DefaultPolicy_e::ZSTD_defaultAllowed
        } else {
            ZSTD_DefaultPolicy_e::ZSTD_defaultDisallowed
        };
        nextEntropy.offcode_repeatMode = prevEntropy.offcode_repeatMode;
        stats.Offtype = ZSTD_selectEncodingType(
            &mut nextEntropy.offcode_repeatMode,
            countWorkspace,
            max,
            mostFrequent,
            nbSeq,
            OffFSELog,
            &prevEntropy.offcodeCTable,
            &OF_defaultNorm,
            OF_defaultNormLog,
            defaultPolicy,
            strategy,
        );
        let countSize = ZSTD_buildCTable(
            &mut dst[op..],
            &mut nextEntropy.offcodeCTable,
            OffFSELog,
            stats.Offtype,
            countWorkspace,
            max,
            &seqStore.ofCode[..nbSeq],
            nbSeq,
            &OF_defaultNorm,
            OF_defaultNormLog,
            DefaultMaxOff,
            &prevEntropy.offcodeCTable,
            prevEntropy.offcodeCTable.len() * 4,
            entropyWorkspace,
        );
        if ERR_isError(countSize) {
            stats.size = countSize;
            return stats;
        }
        if stats.Offtype == SymbolEncodingType_e::set_compressed {
            stats.lastCountSize = countSize;
        }
        op += countSize;
    }

    // --- Match Lengths ---
    {
        let mut max = MaxML;
        let mostFrequent = HIST_countFast_wksp(
            countWorkspace,
            &mut max,
            &seqStore.mlCode[..nbSeq],
            histWorkspace,
        );
        nextEntropy.matchlength_repeatMode = prevEntropy.matchlength_repeatMode;
        stats.MLtype = ZSTD_selectEncodingType(
            &mut nextEntropy.matchlength_repeatMode,
            countWorkspace,
            max,
            mostFrequent,
            nbSeq,
            MLFSELog,
            &prevEntropy.matchlengthCTable,
            &ML_defaultNorm,
            ML_defaultNormLog,
            ZSTD_DefaultPolicy_e::ZSTD_defaultAllowed,
            strategy,
        );
        let countSize = ZSTD_buildCTable(
            &mut dst[op..],
            &mut nextEntropy.matchlengthCTable,
            MLFSELog,
            stats.MLtype,
            countWorkspace,
            max,
            &seqStore.mlCode[..nbSeq],
            nbSeq,
            &ML_defaultNorm,
            ML_defaultNormLog,
            MaxML,
            &prevEntropy.matchlengthCTable,
            prevEntropy.matchlengthCTable.len() * 4,
            entropyWorkspace,
        );
        if ERR_isError(countSize) {
            stats.size = countSize;
            return stats;
        }
        if stats.MLtype == SymbolEncodingType_e::set_compressed {
            stats.lastCountSize = countSize;
        }
        op += countSize;
    }

    stats.size = op;
    stats
}

#[allow(clippy::too_many_arguments)]
pub fn ZSTD_buildSequencesStatistics(
    seqStore: &mut SeqStore_t,
    nbSeq: usize,
    prevEntropy: &ZSTD_fseCTables_t,
    nextEntropy: &mut ZSTD_fseCTables_t,
    dst: &mut [u8],
    strategy: u32,
    countWorkspace: &mut [u32],
    entropyWorkspace: &mut [u8],
) -> ZSTD_symbolEncodingTypeStats_t {
    let mut histWorkspace = [0u32; 1024];
    ZSTD_buildSequencesStatistics_wksp(
        seqStore,
        nbSeq,
        prevEntropy,
        nextEntropy,
        dst,
        strategy,
        countWorkspace,
        entropyWorkspace,
        &mut histWorkspace,
    )
}

/// Port of `ZSTD_hufCTables_t`. Literal-Huffman CTable + its repeat
/// mode. Sized for a 255-symbol alphabet (upstream
/// `HUF_CTABLE_SIZE_ST(255) = 257`).
#[derive(Debug, Clone)]
pub struct ZSTD_hufCTables_t {
    pub CTable: Vec<HUF_CElt>,
    pub repeatMode: HUF_repeat,
}

impl Default for ZSTD_hufCTables_t {
    fn default() -> Self {
        Self {
            CTable: vec![HUF_CElt::default(); 257],
            repeatMode: HUF_repeat::HUF_repeat_none,
        }
    }
}

/// Port of `ZSTD_entropyCTables_t`.
#[derive(Debug, Clone, Default)]
pub struct ZSTD_entropyCTables_t {
    pub huf: ZSTD_hufCTables_t,
    pub fse: ZSTD_fseCTables_t,
}

const SUSPECT_UNCOMPRESSIBLE_LITERAL_RATIO: usize = 20;

/// Port of `ZSTD_entropyCompressSeqStore_internal`. Emits a full
/// compressed block body:
///   1. Literal section via `ZSTD_compressLiterals`
///   2. Sequence-count header (1/2/3 bytes)
///   3. 3 CTables + NCount headers via `ZSTD_buildSequencesStatistics`
///   4. FSE-coded sequence stream via `ZSTD_encodeSequences`
///
/// Returns total bytes written, or `0` if the caller should fall back
/// to an uncompressed block (upstream's "1.3.4 bug" avoidance), or a
/// zstd error code.
///
/// Rust signature note: upstream takes `literals: *const void` pointing
/// at the seq-store's literal buffer. We take the full `SeqStore_t`
/// (which owns `literals`) + the `disableLiteralCompression` flag that
/// upstream computes from `cctxParams`.
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_entropyCompressSeqStore_internal_wksp(
    dst: &mut [u8],
    seqStore: &mut SeqStore_t,
    prevEntropy: &ZSTD_entropyCTables_t,
    nextEntropy: &mut ZSTD_entropyCTables_t,
    strategy: u32,
    disableLiteralCompression: i32,
    bmi2: i32,
    countWorkspace: &mut [u32],
    entropyWorkspace: &mut [u8],
    histWorkspace: &mut [u32],
) -> usize {
    let nbSeq = seqStore.sequences.len();
    let litSize = seqStore.literals.len();
    let dstCapacity = dst.len();
    let mut op = 0usize;

    // --- 1. Compress literals ---
    let suspectUncompressible =
        if nbSeq == 0 || litSize / nbSeq >= SUSPECT_UNCOMPRESSIBLE_LITERAL_RATIO {
            1
        } else {
            0
        };
    // Upstream passes both `prevHuf` and `nextHuf` to
    // `ZSTD_compressLiterals()`, and that helper starts by copying the
    // full previous HUF state into `nextHuf`. We need the same state
    // here so repeat/check mode validates and possibly reuses the
    // previous table, not a stale table already sitting in `nextEntropy`.
    nextEntropy.huf.clone_from(&prevEntropy.huf);
    let cSize = ZSTD_compressLiterals(
        &mut dst[op..],
        &seqStore.literals,
        disableLiteralCompression,
        strategy,
        Some(&mut nextEntropy.huf.CTable),
        Some(&mut nextEntropy.huf.repeatMode),
        suspectUncompressible,
        bmi2,
    );
    if ERR_isError(cSize) {
        return cSize;
    }
    op += cSize;

    // --- 2. Sequence-count header (1/2/3 bytes) ---
    if dstCapacity - op < 3 + 1 {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    if nbSeq < 128 {
        dst[op] = nbSeq as u8;
        op += 1;
    } else if (nbSeq as i32) < LONGNBSEQ {
        dst[op] = ((nbSeq >> 8) + 0x80) as u8;
        dst[op + 1] = nbSeq as u8;
        op += 2;
    } else {
        dst[op] = 0xFF;
        MEM_writeLE16(&mut dst[op + 1..], (nbSeq as i32 - LONGNBSEQ) as u16);
        op += 3;
    }

    if nbSeq == 0 {
        // Carry prev tables over (upstream: memcpy(&next.fse, &prev.fse)).
        nextEntropy.fse = prevEntropy.fse.clone();
        return op;
    }

    // --- 3. Sequence statistics (NCount headers + CTables) ---
    let seqHead_pos = op;
    op += 1;
    let stats = ZSTD_buildSequencesStatistics_wksp(
        seqStore,
        nbSeq,
        &prevEntropy.fse,
        &mut nextEntropy.fse,
        &mut dst[op..],
        strategy,
        countWorkspace,
        entropyWorkspace,
        histWorkspace,
    );
    if ERR_isError(stats.size) {
        return stats.size;
    }
    let seqHeadByte =
        ((stats.LLtype as u8) << 6) | ((stats.Offtype as u8) << 4) | ((stats.MLtype as u8) << 2);
    dst[seqHead_pos] = seqHeadByte;
    op += stats.size;

    // --- 4. FSE-coded sequence bit-stream ---
    let bitstreamSize = ZSTD_encodeSequences(
        &mut dst[op..],
        &nextEntropy.fse.matchlengthCTable,
        &seqStore.mlCode[..nbSeq],
        &nextEntropy.fse.offcodeCTable,
        &seqStore.ofCode[..nbSeq],
        &nextEntropy.fse.litlengthCTable,
        &seqStore.llCode[..nbSeq],
        &seqStore.sequences[..nbSeq],
        nbSeq,
        stats.longOffsets,
        bmi2,
    );
    if ERR_isError(bitstreamSize) {
        return bitstreamSize;
    }
    op += bitstreamSize;

    // Upstream 1.3.4-bug avoidance: if the last NCount was 2 bytes and
    // the bitstream is 1 byte (total 3), emit uncompressed instead.
    if stats.lastCountSize != 0 && (stats.lastCountSize + bitstreamSize) < 4 {
        return 0;
    }
    op
}

pub fn ZSTD_entropyCompressSeqStore_internal(
    dst: &mut [u8],
    seqStore: &mut SeqStore_t,
    prevEntropy: &ZSTD_entropyCTables_t,
    nextEntropy: &mut ZSTD_entropyCTables_t,
    strategy: u32,
    disableLiteralCompression: i32,
    bmi2: i32,
) -> usize {
    let mut scratch = ZSTD_entropyScratch::default();
    ZSTD_entropyCompressSeqStore_internal_wksp(
        dst,
        seqStore,
        prevEntropy,
        nextEntropy,
        strategy,
        disableLiteralCompression,
        bmi2,
        &mut scratch.seqCountWorkspace,
        &mut scratch.seqEntropyWorkspace,
        &mut scratch.seqHistWorkspace,
    )
}

/// Port of `ZSTD_entropyCompressSeqStore_wExtLitBuffer` + the
/// `ZSTD_entropyCompressSeqStore` wrapper — checks the ratio-vs-raw
/// gate via `ZSTD_minGain` and falls back to "emit uncompressed"
/// (return 0) when the compressed payload doesn't beat the source by
/// the strategy-driven threshold.
pub fn ZSTD_entropyCompressSeqStore(
    dst: &mut [u8],
    seqStore: &mut SeqStore_t,
    prevEntropy: &ZSTD_entropyCTables_t,
    nextEntropy: &mut ZSTD_entropyCTables_t,
    strategy: u32,
    disableLiteralCompression: i32,
    blockSize: usize,
    bmi2: i32,
) -> usize {
    let mut scratch = ZSTD_entropyScratch::default();
    ZSTD_entropyCompressSeqStore_wksp(
        dst,
        seqStore,
        prevEntropy,
        nextEntropy,
        strategy,
        disableLiteralCompression,
        blockSize,
        bmi2,
        &mut scratch,
    )
}

pub fn ZSTD_entropyCompressSeqStore_wksp(
    dst: &mut [u8],
    seqStore: &mut SeqStore_t,
    prevEntropy: &ZSTD_entropyCTables_t,
    nextEntropy: &mut ZSTD_entropyCTables_t,
    strategy: u32,
    disableLiteralCompression: i32,
    blockSize: usize,
    bmi2: i32,
    scratch: &mut ZSTD_entropyScratch,
) -> usize {
    let dstCapacity = dst.len();
    let cSize = ZSTD_entropyCompressSeqStore_internal_wksp(
        dst,
        seqStore,
        prevEntropy,
        nextEntropy,
        strategy,
        disableLiteralCompression,
        bmi2,
        &mut scratch.seqCountWorkspace,
        &mut scratch.seqEntropyWorkspace,
        &mut scratch.seqHistWorkspace,
    );
    if cSize == 0 {
        return 0;
    }
    // If internal ran out of space but a raw block would fit, fall back.
    if cSize == ERROR(ErrorCode::DstSizeTooSmall) && blockSize <= dstCapacity {
        return 0;
    }
    if ERR_isError(cSize) {
        return cSize;
    }
    let minGain = ZSTD_minGain(blockSize, strategy);
    let maxCSize = blockSize.saturating_sub(minGain);
    if cSize >= maxCSize {
        return 0; // not compressible enough
    }
    debug_assert!(cSize < ZSTD_BLOCKSIZE_MAX);
    cSize
}

/// Port of `ZSTD_noCompressBlock`. Writes a 3-byte block header
/// signalling a raw (uncompressed) block, then copies `src` verbatim.
/// `lastBlock` is the last-block bit (0 or 1). Returns the total
/// number of bytes written (header + payload) or an error code.
pub fn ZSTD_noCompressBlock(dst: &mut [u8], src: &[u8], lastBlock: u32) -> usize {
    use crate::decompress::zstd_decompress_block::{blockType_e, ZSTD_blockHeaderSize};
    let cBlockHeader24 = lastBlock
        .wrapping_add((blockType_e::bt_raw as u32) << 1)
        .wrapping_add((src.len() as u32) << 3);
    if src.len() + ZSTD_blockHeaderSize > dst.len() {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    MEM_writeLE24(dst, cBlockHeader24);
    dst[ZSTD_blockHeaderSize..ZSTD_blockHeaderSize + src.len()].copy_from_slice(src);
    ZSTD_blockHeaderSize + src.len()
}

/// Port of `ZSTD_rleCompressBlock`. Writes a 3-byte block header + 1
/// RLE symbol byte (total 4 bytes).
pub fn ZSTD_rleCompressBlock(dst: &mut [u8], rleByte: u8, srcSize: usize, lastBlock: u32) -> usize {
    use crate::decompress::zstd_decompress_block::blockType_e;
    let cBlockHeader = lastBlock
        .wrapping_add((blockType_e::bt_rle as u32) << 1)
        .wrapping_add((srcSize as u32) << 3);
    if dst.len() < 4 {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    MEM_writeLE24(dst, cBlockHeader);
    dst[3] = rleByte;
    4
}

/// Strategy-dispatching match-finder + entropy stage over a
/// cumulative-src buffer. Scans `src[istart..]`, treats `src[..istart]`
/// as prior-block content.
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_compressBlock_any_then_entropy_with_history_wksp(
    dst: &mut [u8],
    src: &[u8],
    istart: usize,
    ms: &mut crate::compress::match_state::ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; crate::compress::seq_store::ZSTD_REP_NUM],
    prevEntropy: &ZSTD_entropyCTables_t,
    nextEntropy: &mut ZSTD_entropyCTables_t,
    strategy: u32,
    useRowMatchFinder: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
    disableLiteralCompression: i32,
    bmi2: i32,
    allowRle: bool,
    scratch: &mut ZSTD_entropyScratch,
) -> usize {
    const RLE_MAX_LENGTH: usize = 25;
    use crate::compress::zstd_compress_sequences::{
        ZSTD_btlazy2, ZSTD_btopt, ZSTD_btultra, ZSTD_btultra2, ZSTD_dfast, ZSTD_fast, ZSTD_greedy,
        ZSTD_lazy, ZSTD_lazy2,
    };

    ZSTD_setEntropySeed_if_needed(ms, prevEntropy, strategy);

    let lastLits = match strategy {
        s if s == ZSTD_fast => crate::compress::zstd_fast::ZSTD_compressBlock_fast_noDict_generic(
            ms,
            seqStore,
            rep,
            src,
            istart,
            ms.cParams.minMatch,
        ),
        s if s == ZSTD_dfast => {
            crate::compress::zstd_double_fast::ZSTD_compressBlock_doubleFast_noDict_generic(
                ms,
                seqStore,
                rep,
                src,
                istart,
                ms.cParams.minMatch,
            )
        }
        s if s == ZSTD_greedy => {
            let searchMethod = if crate::compress::match_state::ZSTD_rowMatchFinderUsed(
                strategy,
                useRowMatchFinder,
            ) {
                crate::compress::zstd_lazy::searchMethod_e::search_rowHash
            } else {
                crate::compress::zstd_lazy::searchMethod_e::search_hashChain
            };
            crate::compress::zstd_lazy::ZSTD_compressBlock_lazy_generic_with_istart(
                ms,
                seqStore,
                rep,
                src,
                istart,
                searchMethod,
                0,
                crate::compress::match_state::ZSTD_dictMode_e::ZSTD_noDict,
            )
        }
        s if s == ZSTD_lazy => {
            let searchMethod = if crate::compress::match_state::ZSTD_rowMatchFinderUsed(
                strategy,
                useRowMatchFinder,
            ) {
                crate::compress::zstd_lazy::searchMethod_e::search_rowHash
            } else {
                crate::compress::zstd_lazy::searchMethod_e::search_hashChain
            };
            crate::compress::zstd_lazy::ZSTD_compressBlock_lazy_generic_with_istart(
                ms,
                seqStore,
                rep,
                src,
                istart,
                searchMethod,
                1,
                crate::compress::match_state::ZSTD_dictMode_e::ZSTD_noDict,
            )
        }
        s if s == ZSTD_lazy2 || s == ZSTD_btlazy2 => {
            let searchMethod = if crate::compress::match_state::ZSTD_rowMatchFinderUsed(
                strategy,
                useRowMatchFinder,
            ) {
                crate::compress::zstd_lazy::searchMethod_e::search_rowHash
            } else {
                crate::compress::zstd_lazy::searchMethod_e::search_hashChain
            };
            crate::compress::zstd_lazy::ZSTD_compressBlock_lazy_generic_with_istart(
                ms,
                seqStore,
                rep,
                src,
                istart,
                searchMethod,
                2,
                crate::compress::match_state::ZSTD_dictMode_e::ZSTD_noDict,
            )
        }
        s if s == ZSTD_btopt => crate::compress::zstd_opt::ZSTD_compressBlock_btopt_window(
            ms,
            seqStore,
            rep,
            src,
            istart,
            src.len(),
        ),
        s if s == ZSTD_btultra => crate::compress::zstd_opt::ZSTD_compressBlock_btultra_window(
            ms,
            seqStore,
            rep,
            src,
            istart,
            src.len(),
        ),
        s if s == ZSTD_btultra2 => crate::compress::zstd_opt::ZSTD_compressBlock_btultra2_window(
            ms,
            seqStore,
            rep,
            src,
            istart,
            src.len(),
        ),
        _ => crate::compress::zstd_fast::ZSTD_compressBlock_fast_noDict_generic(
            ms,
            seqStore,
            rep,
            src,
            istart,
            ms.cParams.minMatch,
        ),
    };

    let tail_start = src.len() - lastLits;
    seqStore.literals.extend_from_slice(&src[tail_start..]);
    ZSTD_seqToCodes(seqStore);

    let blockSize = src.len() - istart;
    let cSize = ZSTD_entropyCompressSeqStore_wksp(
        dst,
        seqStore,
        prevEntropy,
        nextEntropy,
        strategy,
        disableLiteralCompression,
        blockSize,
        bmi2,
        scratch,
    );

    if allowRle
        && !ERR_isError(cSize)
        && cSize < RLE_MAX_LENGTH
        && ZSTD_isRLE(&src[istart..]) != 0
        && !dst.is_empty()
    {
        dst[0] = src[istart];
        return 1;
    }
    cSize
}

/// Frame-wrapped history compressor.
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_compressBlock_any_framed_with_history_wksp(
    dst: &mut [u8],
    src: &[u8],
    istart: usize,
    ms: &mut crate::compress::match_state::ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; crate::compress::seq_store::ZSTD_REP_NUM],
    prevEntropy: &ZSTD_entropyCTables_t,
    nextEntropy: &mut ZSTD_entropyCTables_t,
    strategy: u32,
    useRowMatchFinder: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
    disableLiteralCompression: i32,
    bmi2: i32,
    lastBlock: u32,
    isFirstBlock: bool,
    scratch: &mut ZSTD_entropyScratch,
) -> usize {
    use crate::decompress::zstd_decompress_block::{blockType_e, ZSTD_blockHeaderSize};
    if dst.len() < ZSTD_blockHeaderSize {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    let (header_slot, body_slot) = dst.split_at_mut(ZSTD_blockHeaderSize);
    let cBodySize = ZSTD_compressBlock_any_then_entropy_with_history_wksp(
        body_slot,
        src,
        istart,
        ms,
        seqStore,
        rep,
        prevEntropy,
        nextEntropy,
        strategy,
        useRowMatchFinder,
        disableLiteralCompression,
        bmi2,
        !isFirstBlock,
        scratch,
    );
    if ERR_isError(cBodySize) {
        return cBodySize;
    }
    let block_src_len = src.len() - istart;
    if cBodySize == 1 && block_src_len > 0 {
        let rleByte = body_slot[0];
        return ZSTD_rleCompressBlock(dst, rleByte, block_src_len, lastBlock);
    }
    if cBodySize == 0 {
        return ZSTD_noCompressBlock(dst, &src[istart..], lastBlock);
    }
    let header = lastBlock
        .wrapping_add((blockType_e::bt_compressed as u32) << 1)
        .wrapping_add((cBodySize as u32) << 3);
    MEM_writeLE24(header_slot, header);
    ZSTD_blockHeaderSize + cBodySize
}

/// Simplified port of `ZSTD_compressBlock_internal`'s control flow.
/// Dispatches on `cParams.strategy`:
///   - `ZSTD_fast`  → `ZSTD_compressBlock_fast`
///   - `ZSTD_dfast` → `ZSTD_compressBlock_doubleFast`
///   - `ZSTD_greedy`/`ZSTD_lazy`/`ZSTD_lazy2`/`ZSTD_btlazy2` →
///     the corresponding lazy-family block compressors selected by
///     `ZSTD_selectBlockCompressor()`.
///   - `ZSTD_btopt`/`ZSTD_btultra`/`ZSTD_btultra2` →
///     the shared optimal-parser entries in `zstd_opt.rs`.
///
/// After the match finder runs:
///   - Append the tail literals to the seq-store buffer.
///   - `ZSTD_seqToCodes` materializes ll/of/ml code tables.
///   - `ZSTD_entropyCompressSeqStore` emits the compressed body.
///   - Late `ZSTD_isRLE` check downgrades a trivial result (≤25 B)
///     to a 1-byte RLE payload.
///
/// Returns compressed size, `0` to signal "fall back to raw", or an
/// error code. The seqStore is left in its post-encode state — the
/// caller must `reset()` it before the next block.
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_compressBlock_fast_then_entropy(
    dst: &mut [u8],
    src: &[u8],
    ms: &mut crate::compress::match_state::ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; crate::compress::seq_store::ZSTD_REP_NUM],
    prevEntropy: &ZSTD_entropyCTables_t,
    nextEntropy: &mut ZSTD_entropyCTables_t,
    strategy: u32,
    disableLiteralCompression: i32,
    bmi2: i32,
) -> usize {
    let mut scratch = ZSTD_entropyScratch::default();
    ZSTD_compressBlock_fast_then_entropy_wksp(
        dst,
        src,
        ms,
        seqStore,
        rep,
        prevEntropy,
        nextEntropy,
        strategy,
        disableLiteralCompression,
        bmi2,
        &mut scratch,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn ZSTD_compressBlock_fast_then_entropy_wksp(
    dst: &mut [u8],
    src: &[u8],
    ms: &mut crate::compress::match_state::ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; crate::compress::seq_store::ZSTD_REP_NUM],
    prevEntropy: &ZSTD_entropyCTables_t,
    nextEntropy: &mut ZSTD_entropyCTables_t,
    strategy: u32,
    disableLiteralCompression: i32,
    bmi2: i32,
    scratch: &mut ZSTD_entropyScratch,
) -> usize {
    const RLE_MAX_LENGTH: usize = 25;

    ZSTD_setEntropySeed_if_needed(ms, prevEntropy, strategy);

    // Phase 1: run the strategy-appropriate matcher. Append the tail
    // literals so the seq store's literals buffer is complete.
    let lastLits = match strategy {
        crate::compress::zstd_compress_sequences::ZSTD_greedy => {
            crate::compress::zstd_lazy::ZSTD_compressBlock_greedy(ms, seqStore, rep, src)
        }
        crate::compress::zstd_compress_sequences::ZSTD_lazy => {
            crate::compress::zstd_lazy::ZSTD_compressBlock_lazy(ms, seqStore, rep, src)
        }
        crate::compress::zstd_compress_sequences::ZSTD_lazy2
        | crate::compress::zstd_compress_sequences::ZSTD_btlazy2 => {
            crate::compress::zstd_lazy::ZSTD_compressBlock_lazy2(ms, seqStore, rep, src)
        }
        crate::compress::zstd_compress_sequences::ZSTD_dfast => {
            crate::compress::zstd_double_fast::ZSTD_compressBlock_doubleFast(ms, seqStore, rep, src)
        }
        _ => crate::compress::zstd_fast::ZSTD_compressBlock_fast(ms, seqStore, rep, src),
    };
    let tail_start = src.len() - lastLits;
    seqStore.literals.extend_from_slice(&src[tail_start..]);

    // Phase 2: materialize the three code tables.
    ZSTD_seqToCodes(seqStore);

    // Phase 3: entropy-compress the block body.
    let cSize = ZSTD_entropyCompressSeqStore_wksp(
        dst,
        seqStore,
        prevEntropy,
        nextEntropy,
        strategy,
        disableLiteralCompression,
        src.len(),
        bmi2,
        scratch,
    );

    // Phase 4: upstream's late RLE downgrade.
    if !ERR_isError(cSize) && cSize < RLE_MAX_LENGTH && ZSTD_isRLE(src) != 0 && !dst.is_empty() {
        dst[0] = src[0];
        return 1;
    }
    cSize
}

/// Emits a fully-framed block (3-byte header + body). Runs the
/// `fast`-strategy match finder + entropy stage; on any of the three
/// fall-back conditions (raw-prefered, RLE-downgrade, or no savings)
/// emits the appropriate block type via `ZSTD_noCompressBlock` /
/// `ZSTD_rleCompressBlock`.
///
/// Rust signature note: upstream `ZSTD_compressBlock_internal` doesn't
/// emit the block header itself — the caller in `ZSTD_compress_frameChunk`
/// does so with knowledge of `lastBlock`. We keep that separation by
/// returning the body size + the selected block type; callers write
/// the header themselves.
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_compressBlock_fast_framed(
    dst: &mut [u8],
    src: &[u8],
    ms: &mut crate::compress::match_state::ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; crate::compress::seq_store::ZSTD_REP_NUM],
    prevEntropy: &ZSTD_entropyCTables_t,
    nextEntropy: &mut ZSTD_entropyCTables_t,
    strategy: u32,
    disableLiteralCompression: i32,
    bmi2: i32,
    lastBlock: u32,
) -> usize {
    let mut scratch = ZSTD_entropyScratch::default();
    ZSTD_compressBlock_fast_framed_wksp(
        dst,
        src,
        ms,
        seqStore,
        rep,
        prevEntropy,
        nextEntropy,
        strategy,
        disableLiteralCompression,
        bmi2,
        lastBlock,
        &mut scratch,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn ZSTD_compressBlock_fast_framed_wksp(
    dst: &mut [u8],
    src: &[u8],
    ms: &mut crate::compress::match_state::ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; crate::compress::seq_store::ZSTD_REP_NUM],
    prevEntropy: &ZSTD_entropyCTables_t,
    nextEntropy: &mut ZSTD_entropyCTables_t,
    strategy: u32,
    disableLiteralCompression: i32,
    bmi2: i32,
    lastBlock: u32,
    scratch: &mut ZSTD_entropyScratch,
) -> usize {
    use crate::decompress::zstd_decompress_block::ZSTD_blockHeaderSize;

    // Try compressed; may return 0 (fall-back to raw) or 1 (RLE).
    // Reserve 3 bytes at dst[0..3] for the block header — encode body
    // into dst[ZSTD_blockHeaderSize..].
    if dst.len() < ZSTD_blockHeaderSize {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    let (header_slot, body_slot) = dst.split_at_mut(ZSTD_blockHeaderSize);
    let cBodySize = ZSTD_compressBlock_fast_then_entropy_wksp(
        body_slot,
        src,
        ms,
        seqStore,
        rep,
        prevEntropy,
        nextEntropy,
        strategy,
        disableLiteralCompression,
        bmi2,
        scratch,
    );
    if ERR_isError(cBodySize) {
        return cBodySize;
    }
    // cBodySize == 1 → late-RLE downgrade from compressBlock. Emit RLE block.
    if cBodySize == 1 && !src.is_empty() {
        let rleByte = body_slot[0];
        return ZSTD_rleCompressBlock(dst, rleByte, src.len(), lastBlock);
    }
    // cBodySize == 0 → "fall back to raw".
    if cBodySize == 0 {
        return ZSTD_noCompressBlock(dst, src, lastBlock);
    }
    // Otherwise emit a compressed block: write the header into the
    // reserved slot.
    use crate::decompress::zstd_decompress_block::blockType_e;
    let header = lastBlock
        .wrapping_add((blockType_e::bt_compressed as u32) << 1)
        .wrapping_add((cBodySize as u32) << 3);
    MEM_writeLE24(header_slot, header);
    ZSTD_blockHeaderSize + cBodySize
}

/// Rust-only helper variant of `ZSTD_compressBlock_internal` that operates
/// on a cumulative `window_buf` slice plus `(src_pos, src_end)` indices —
/// lets the caller pass prior-block history without copying. Builds the
/// seq store via `ZSTD_buildSeqStore_with_window`, entropy-compresses, then
/// applies upstream's late RLE downgrade + repcode/entropy table confirm.
#[allow(clippy::too_many_arguments)]
fn ZSTD_compressBlock_internal_with_window(
    cctx: &mut ZSTD_CCtx,
    dst: &mut [u8],
    window_buf: &[u8],
    src_pos: usize,
    src_end: usize,
    frame: bool,
) -> usize {
    use crate::compress::zstd_compress_sequences::FSE_repeat;

    let src = &window_buf[src_pos..src_end];
    let blockSize = src.len();
    let bss = ZSTD_buildSeqStore_with_window(cctx, window_buf, src_pos, src_end);
    if ERR_isError(bss) {
        return bss;
    }
    if bss == ZSTD_BuildSeqStore_e::ZSTDbss_noCompress as usize {
        if cctx.prevEntropy.fse.offcode_repeatMode == FSE_repeat::FSE_repeat_valid {
            cctx.prevEntropy.fse.offcode_repeatMode = FSE_repeat::FSE_repeat_check;
        }
        return 0;
    }

    let disableLiteralCompression = ZSTD_literalsCompressionIsDisabled(
        cctx.appliedParams.literalCompressionMode,
        cctx.appliedParams.cParams.strategy,
        cctx.appliedParams.cParams.targetLength,
    ) as i32;
    let compressedSeqsSize = {
        let seqStore = cctx.seqStore.as_mut().unwrap();
        ZSTD_entropyCompressSeqStore_wksp(
            dst,
            seqStore,
            &cctx.prevEntropy,
            &mut cctx.nextEntropy,
            cctx.appliedParams.cParams.strategy,
            disableLiteralCompression,
            blockSize,
            cctx.bmi2,
            &mut cctx.entropyScratch,
        )
    };
    if ERR_isError(compressedSeqsSize) {
        return compressedSeqsSize;
    }

    let compressedSeqsSize =
        if frame && cctx.isFirstBlock == 0 && compressedSeqsSize < 25 && ZSTD_isRLE(src) != 0 {
            dst[0] = src[0];
            1
        } else {
            compressedSeqsSize
        };

    if compressedSeqsSize > 1 {
        ZSTD_blockState_confirmRepcodesAndEntropyTables(cctx);
    }
    if cctx.prevEntropy.fse.offcode_repeatMode == FSE_repeat::FSE_repeat_valid {
        cctx.prevEntropy.fse.offcode_repeatMode = FSE_repeat::FSE_repeat_check;
    }
    compressedSeqsSize
}

/// Port of `ZSTD_CCtxParams_setZstdParams` (zstd_compress.c:423).
/// Copies a `ZSTD_parameters` bundle into a `ZSTD_CCtx_params`
/// struct. Upstream asserts cParams validity; we match that contract
/// by calling `ZSTD_checkCParams` and propagating any error as a
/// return value (upstream's `assert` relies on callers having already
/// validated). Sets `compressionLevel` to `ZSTD_NO_CLEVEL` per
/// upstream, since cParams are assumed fully defined.
pub fn ZSTD_CCtxParams_setZstdParams(
    cctxParams: &mut ZSTD_CCtx_params,
    params: &ZSTD_parameters,
) -> usize {
    let rc = ZSTD_checkCParams(params.cParams);
    if ERR_isError(rc) {
        return rc;
    }
    cctxParams.cParams = params.cParams;
    cctxParams.fParams = params.fParams;
    cctxParams.compressionLevel = ZSTD_NO_CLEVEL;
    0
}

/// Port of `BlockSummary` (zstd_compress_internal.h:1528).
/// Summary of a single block's sequence list — returned by
/// `ZSTD_get1BlockSummary` so the ZSTD_compressSequences driver can
/// size the output block's header before emitting bytes.
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockSummary {
    pub nbSequences: usize,
    pub blockSize: usize,
    pub litSize: usize,
}

/// Port of `ZSTD_get1BlockSummary` (zstd_compress.c:7919 scalar).
/// Upstream's scalar fallback reads `litLength` + `matchLength` as a
/// packed `U64`, sums those packed halves independently, and uses the
/// `matchLengthHalfIsZero` helper to identify the end-of-block
/// terminator. This preserves the original helper structure and the
/// packed-half arithmetic.
#[inline]
pub fn matchLengthHalfIsZero(litMatchLength: u64) -> bool {
    if crate::common::mem::MEM_isLittleEndian() != 0 {
        litMatchLength <= 0xFFFF_FFFF
    } else {
        (litMatchLength as u32) == 0
    }
}

/// Port of `ZSTD_get1BlockSummary` scalar variant (`zstd_compress.c:7919`).
/// Walks `seqs` until the block-delimiter terminator (offset==0 &&
/// matchLength==0), using four packed-half accumulators for throughput.
/// Returns the per-block sequence count, total `blockSize`, and `litSize`.
/// On a missing terminator returns `nbSequences` set to an
/// `ExternalSequencesInvalid` error code.
pub fn ZSTD_get1BlockSummary(seqs: &[ZSTD_Sequence]) -> BlockSummary {
    #[inline]
    fn packed_lit_match_length(seq: &ZSTD_Sequence) -> u64 {
        if crate::common::mem::MEM_isLittleEndian() != 0 {
            (seq.litLength as u64) | ((seq.matchLength as u64) << 32)
        } else {
            ((seq.litLength as u64) << 32) | (seq.matchLength as u64)
        }
    }

    let mut litMatchSize0: u64 = 0;
    let mut litMatchSize1: u64 = 0;
    let mut litMatchSize2: u64 = 0;
    let mut litMatchSize3: u64 = 0;
    let mut n = 0usize;
    let mut found_terminator = false;

    if seqs.len() > 3 {
        while n < seqs.len() - 3 {
            let mut litMatchLength = packed_lit_match_length(&seqs[n]);
            litMatchSize0 = litMatchSize0.wrapping_add(litMatchLength);
            if matchLengthHalfIsZero(litMatchLength) {
                debug_assert_eq!(seqs[n].offset, 0);
                found_terminator = true;
                break;
            }

            litMatchLength = packed_lit_match_length(&seqs[n + 1]);
            litMatchSize1 = litMatchSize1.wrapping_add(litMatchLength);
            if matchLengthHalfIsZero(litMatchLength) {
                n += 1;
                debug_assert_eq!(seqs[n].offset, 0);
                found_terminator = true;
                break;
            }

            litMatchLength = packed_lit_match_length(&seqs[n + 2]);
            litMatchSize2 = litMatchSize2.wrapping_add(litMatchLength);
            if matchLengthHalfIsZero(litMatchLength) {
                n += 2;
                debug_assert_eq!(seqs[n].offset, 0);
                found_terminator = true;
                break;
            }

            litMatchLength = packed_lit_match_length(&seqs[n + 3]);
            litMatchSize3 = litMatchSize3.wrapping_add(litMatchLength);
            if matchLengthHalfIsZero(litMatchLength) {
                n += 3;
                debug_assert_eq!(seqs[n].offset, 0);
                found_terminator = true;
                break;
            }

            n += 4;
        }
    }

    while !found_terminator && n < seqs.len() {
        let litMatchLength = packed_lit_match_length(&seqs[n]);
        litMatchSize0 = litMatchSize0.wrapping_add(litMatchLength);
        if matchLengthHalfIsZero(litMatchLength) {
            debug_assert_eq!(seqs[n].offset, 0);
            found_terminator = true;
            break;
        }
        n += 1;
    }

    if !found_terminator {
        return BlockSummary {
            nbSequences: ERROR(ErrorCode::ExternalSequencesInvalid),
            ..BlockSummary::default()
        };
    }

    let total = litMatchSize0
        .wrapping_add(litMatchSize1)
        .wrapping_add(litMatchSize2)
        .wrapping_add(litMatchSize3);
    let (litSize, blockSize) = if crate::common::mem::MEM_isLittleEndian() != 0 {
        let lit = total as u32 as usize;
        let block = lit.wrapping_add((total >> 32) as usize);
        (lit, block)
    } else {
        let lit = (total >> 32) as usize;
        let block = lit.wrapping_add(total as u32 as usize);
        (lit, block)
    };
    BlockSummary {
        nbSequences: n + 1,
        blockSize,
        litSize,
    }
}

/// Port of `ZSTD_initCCtx` (zstd_compress.c:110). Upstream zeros the
/// whole struct and applies default parameters via
/// `ZSTD_CCtx_reset(_parameters)`. Our port's CCtx is already
/// default-constructed via `ZSTD_CCtx::default()`; this helper simply
/// resets the struct to defaults while preserving the requested
/// `customMem` bundle on the live CCtx.
pub fn ZSTD_initCCtx(cctx: &mut ZSTD_CCtx, _customMem: ZSTD_customMem) {
    let preservedCustomMem = if ZSTD_customMem_isNull(_customMem) {
        cctx.customMem
    } else {
        _customMem
    };
    *cctx = ZSTD_CCtx::default();
    cctx.customMem = preservedCustomMem;
    cctx.requestedParams.customMem = preservedCustomMem;
    cctx.appliedParams.customMem = preservedCustomMem;
    let _ = ZSTD_CCtx_reset(cctx, ZSTD_ResetDirective::ZSTD_reset_parameters);
}

/// Port of `ZSTD_useTargetCBlockSize` (zstd_compress.c:2749). True
/// when `targetCBlockSize != 0` — enables the superblock compressor.
#[inline]
pub fn ZSTD_useTargetCBlockSize(cctxParams: &ZSTD_CCtx_params) -> bool {
    cctxParams.targetCBlockSize != 0
}

/// Port of `ZSTD_blockSplitterEnabled` (zstd_compress.c:2760). True
/// when `postBlockSplitter` is explicitly enabled. Upstream asserts
/// `postBlockSplitter != ZSTD_ps_auto` here; `ps_auto` resolves via
/// `ZSTD_resolveBlockSplitterMode` before this is called.
#[inline]
pub fn ZSTD_blockSplitterEnabled(cctxParams: &ZSTD_CCtx_params) -> bool {
    cctxParams.postBlockSplitter == crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_enable
}

/// Port of `ZSTD_isUpdateAuthorized` (zstd_compress.c:674). Returns
/// true if a given `ZSTD_cParameter` is safe to update mid-session
/// (i.e. after `compressStream` has started), false if changing it
/// would require resetting the CCtx.
#[inline]
pub fn ZSTD_isUpdateAuthorized(param: ZSTD_cParameter) -> bool {
    match param {
        ZSTD_cParameter::ZSTD_c_compressionLevel
        | ZSTD_cParameter::ZSTD_c_hashLog
        | ZSTD_cParameter::ZSTD_c_chainLog
        | ZSTD_cParameter::ZSTD_c_searchLog
        | ZSTD_cParameter::ZSTD_c_minMatch
        | ZSTD_cParameter::ZSTD_c_targetLength
        | ZSTD_cParameter::ZSTD_c_strategy
        | ZSTD_cParameter::ZSTD_c_blockSplitterLevel => true,

        ZSTD_cParameter::ZSTD_c_format
        | ZSTD_cParameter::ZSTD_c_windowLog
        | ZSTD_cParameter::ZSTD_c_contentSizeFlag
        | ZSTD_cParameter::ZSTD_c_checksumFlag
        | ZSTD_cParameter::ZSTD_c_dictIDFlag
        | ZSTD_cParameter::ZSTD_c_forceMaxWindow
        | ZSTD_cParameter::ZSTD_c_nbWorkers
        | ZSTD_cParameter::ZSTD_c_jobSize
        | ZSTD_cParameter::ZSTD_c_overlapLog
        | ZSTD_cParameter::ZSTD_c_rsyncable
        | ZSTD_cParameter::ZSTD_c_enableDedicatedDictSearch
        | ZSTD_cParameter::ZSTD_c_enableLongDistanceMatching
        | ZSTD_cParameter::ZSTD_c_ldmHashLog
        | ZSTD_cParameter::ZSTD_c_ldmMinMatch
        | ZSTD_cParameter::ZSTD_c_ldmBucketSizeLog
        | ZSTD_cParameter::ZSTD_c_ldmHashRateLog
        | ZSTD_cParameter::ZSTD_c_forceAttachDict
        | ZSTD_cParameter::ZSTD_c_literalCompressionMode
        | ZSTD_cParameter::ZSTD_c_targetCBlockSize
        | ZSTD_cParameter::ZSTD_c_srcSizeHint
        | ZSTD_cParameter::ZSTD_c_stableInBuffer
        | ZSTD_cParameter::ZSTD_c_stableOutBuffer
        | ZSTD_cParameter::ZSTD_c_blockDelimiters
        | ZSTD_cParameter::ZSTD_c_validateSequences
        | ZSTD_cParameter::ZSTD_c_splitAfterSequences
        | ZSTD_cParameter::ZSTD_c_useRowMatchFinder
        | ZSTD_cParameter::ZSTD_c_deterministicRefPrefix
        | ZSTD_cParameter::ZSTD_c_prefetchCDictTables
        | ZSTD_cParameter::ZSTD_c_enableSeqProducerFallback
        | ZSTD_cParameter::ZSTD_c_maxBlockSize
        | ZSTD_cParameter::ZSTD_c_repcodeResolution => false,
    }
}

/// Port of `ZSTD_freeCCtxContent` (zstd_compress.c:178). Drops the
/// CCtx's heap-owned content — dicts, match-state tables, seq store,
/// streaming buffers — leaving the outer struct itself intact.
/// Upstream uses this when recycling a CCtx in-place; our port
/// achieves the same via field-level resets. Kept as a named entry
/// so C-compat call sites compile.
pub fn ZSTD_freeCCtxContent(cctx: &mut ZSTD_CCtx) {
    ZSTD_clearAllDicts(cctx);
    cctx.ms = None;
    cctx.seqStore = None;
    cctx.stream_in_buffer.clear();
    cctx.stream_params_snapshot = None;
    cctx.stream_out_buffer.clear();
    cctx.stream_out_drained = 0;
}

/// Port of `ZSTD_CCtx_trace` (zstd_compress.c:5407). Upstream
/// conditionally emits a trace event at the end of compression when
/// `ZSTD_TRACE` is compiled in. Our port has no tracing infrastructure
/// so this is an intentional no-op — kept for API surface parity.
#[inline]
pub fn ZSTD_CCtx_trace(cctx: &mut ZSTD_CCtx, extraCSize: usize) {
    let _streaming = !cctx.stream_in_buffer.is_empty()
        || !cctx.stream_out_buffer.is_empty()
        || cctx.appliedParams.nbWorkers > 0;
    let _dictionaryID = cctx.dictID;
    let _dictionarySize = cctx.dictContentSize;
    let _uncompressedSize = cctx.consumedSrcSize;
    let _compressedSize = cctx.producedCSize + extraCSize as u64;
}

/// Port of `ZSTD_hasExtSeqProd` (zstd_compress_internal.h:1613).
/// Upstream returns true iff the CCtxParams struct carries an
/// external sequence producer callback.
#[inline]
pub fn ZSTD_hasExtSeqProd(params: &ZSTD_CCtx_params) -> bool {
    params.extSeqProdFunc.is_some()
}

/// Port of `ZSTD_getSeqStore` (zstd_compress.c:230). Returns the
/// CCtx's seq store, if it's been allocated. Upstream always has a
/// `seqStore` field; our port defers allocation so the caller may
/// see `None` on a freshly-created CCtx before the first compression
/// call seeds it.
#[inline]
pub fn ZSTD_getSeqStore(ctx: &ZSTD_CCtx) -> Option<&crate::compress::seq_store::SeqStore_t> {
    ctx.seqStore.as_ref()
}

/// Port of `ZSTD_maybeRLE`. Heuristic used by the block-splitter:
/// a block with very few sequences and very few literals is a
/// candidate for the `bt_rle` path. Upstream (zstd_compress.c:3622)
/// returns true when `nbSeqs < 4 && nbLits < 10`.
pub fn ZSTD_maybeRLE(seqStore: &crate::compress::seq_store::SeqStore_t) -> bool {
    seqStore.sequences.len() < 4 && seqStore.literals.len() < 10
}

/// Port of `ZSTD_isRLE`. Returns `1` iff every byte of `src` equals
/// `src[0]`. Upstream uses a size_t-strided unrolled loop; the Rust
/// port uses `Iterator::all` which the compiler auto-vectorizes.
pub fn ZSTD_isRLE(src: &[u8]) -> i32 {
    if src.is_empty() {
        return 1;
    }
    let first = src[0];
    if src.iter().all(|&b| b == first) {
        1
    } else {
        0
    }
}

/// Port of `ZSTD_FrameParameters`. Controls frame-header optionals:
/// content-size, dictID-inclusion, and XXH64 checksum trailer.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZSTD_FrameParameters {
    /// Emit the frame content size in the header (ON by default).
    pub contentSizeFlag: u32,
    /// Emit an XXH64 checksum trailer.
    pub checksumFlag: u32,
    /// Suppress dictID in the header (upstream default: 0 = include).
    pub noDictIDFlag: u32,
}

/// Upstream `ZSTD_FRAMEHEADERSIZE_MAX` — worst-case header size.
pub const ZSTD_FRAMEHEADERSIZE_MAX: usize = 18;

/// Port of `ZSTD_forceIgnoreChecksum_e`. Controls
/// `ZSTD_d_forceIgnoreChecksum` — whether the decompressor validates
/// the XXH64 trailer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ZSTD_forceIgnoreChecksum_e {
    #[default]
    ZSTD_d_validateChecksum = 0,
    ZSTD_d_ignoreChecksum = 1,
}

/// Port of `ZSTD_refMultipleDDicts_e`. Controls
/// `ZSTD_d_refMultipleDDicts`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ZSTD_refMultipleDDicts_e {
    #[default]
    ZSTD_rmd_refSingleDDict = 0,
    ZSTD_rmd_refMultipleDDicts = 1,
}

/// Port of `ZSTD_dictAttachPref_e`. Dictionary attach/copy/load
/// preference — normally driven by an internal heuristic.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ZSTD_dictAttachPref_e {
    #[default]
    ZSTD_dictDefaultAttach = 0,
    ZSTD_dictForceAttach = 1,
    ZSTD_dictForceCopy = 2,
    ZSTD_dictForceLoad = 3,
}

/// Port of `ZSTD_literalCompressionMode_e`. Deprecated in upstream in
/// favor of `ZSTD_ParamSwitch_e` — kept for API parity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ZSTD_literalCompressionMode_e {
    #[default]
    ZSTD_lcm_auto = 0,
    ZSTD_lcm_huffman = 1,
    ZSTD_lcm_uncompressed = 2,
}

/// Port of `ZSTD_frameProgression`. Tracks input / output counters so
/// long-running streaming compressions can report progress. MT-only
/// fields (`currentJobID`, `nbActiveWorkers`) are always zero in v0.1
/// since the compressor is single-threaded.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZSTD_frameProgression {
    pub ingested: u64,
    pub consumed: u64,
    pub produced: u64,
    pub flushed: u64,
    pub currentJobID: u32,
    pub nbActiveWorkers: u32,
}

/// Port of `ZSTD_getFrameProgression`. Upstream walks the CCtx's
/// `consumedSrcSize` / `producedCSize` counters; v0.1 reports
/// `stream_in_buffer` ingest + `stream_out_buffer` produced so
/// `ingested ≥ consumed ≥ produced` holds during a streaming session.
pub fn ZSTD_getFrameProgression(cctx: &ZSTD_CCtx) -> ZSTD_frameProgression {
    let ingested = cctx.stream_in_buffer.len() as u64;
    let produced = cctx.stream_out_buffer.len() as u64;
    let flushed = cctx.stream_out_drained as u64;
    ZSTD_frameProgression {
        ingested,
        // Single-threaded: either the buffer has been drained through
        // endStream (consumed = ingested) or we haven't started.
        consumed: if cctx.stream_closed { ingested } else { 0 },
        produced,
        flushed,
        currentJobID: 0,
        nbActiveWorkers: 0,
    }
}

/// Port of `ZSTD_getBlockSize` (compressor side, `zstd_compress.c:4901`).
/// Upstream keeps this as a direct wrapper over the deprecated helper.
#[inline]
pub fn ZSTD_getBlockSize(cctx: &ZSTD_CCtx) -> usize {
    ZSTD_getBlockSize_deprecated(cctx)
}

/// Port of `ZSTD_writeSkippableFrame`. Writes a skippable frame header
/// (magic + userData) followed by `src`. `magicVariant` is the low
/// nibble tag — must fit in 4 bits.
///
/// Returns total bytes written (header + payload) or an error code
/// for dst-too-small / src-too-large / variant-out-of-range.
pub fn ZSTD_writeSkippableFrame(dst: &mut [u8], src: &[u8], magicVariant: u32) -> usize {
    use crate::common::error::{ErrorCode, ERROR};
    use crate::decompress::zstd_decompress::{
        ZSTD_MAGIC_SKIPPABLE_START, ZSTD_SKIPPABLEHEADERSIZE,
    };
    if dst.len() < src.len() + ZSTD_SKIPPABLEHEADERSIZE {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    if src.len() > u32::MAX as usize {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    if magicVariant > 15 {
        return ERROR(ErrorCode::ParameterOutOfBound);
    }
    MEM_writeLE32(&mut dst[..4], ZSTD_MAGIC_SKIPPABLE_START + magicVariant);
    MEM_writeLE32(&mut dst[4..8], src.len() as u32);
    dst[ZSTD_SKIPPABLEHEADERSIZE..ZSTD_SKIPPABLEHEADERSIZE + src.len()].copy_from_slice(src);
    src.len() + ZSTD_SKIPPABLEHEADERSIZE
}

/// Port of `ZSTD_Sequence`. Advanced-API sequence entry used by
/// `ZSTD_generateSequences` / `ZSTD_compressSequences`. A
/// `(0, 0, 0, _)` tuple marks a block boundary.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZSTD_Sequence {
    pub offset: u32,
    pub litLength: u32,
    pub matchLength: u32,
    /// Repcode slot id (0..=3). When non-zero, `offset` represents
    /// the corresponding rep — see upstream `zstd.h` for the exact
    /// litLength-dependent lookup rules.
    pub rep: u32,
}

/// Port of `convertSequences_noRepcodes` scalar fallback
/// (`zstd_compress.c:7674`). Transcodes a `ZSTD_Sequence[]` stream
/// into the lower-level `SeqDef[]` — applying `OFFSET_TO_OFFBASE` and
/// subtracting `MINMATCH` from each match length. Returns a "long
/// length" index sentinel: 0 for none, `n+1` for a match-length
/// overflow at position n, or `n+nbSequences+1` for a litLength
/// overflow. `rep` field on input is ignored (noRepcodes contract).
pub fn convertSequences_noRepcodes(
    dstSeqs: &mut [crate::compress::seq_store::SeqDef],
    inSeqs: &[ZSTD_Sequence],
) -> usize {
    use crate::compress::seq_store::{MINMATCH, OFFSET_TO_OFFBASE};
    let nbSequences = inSeqs.len();
    debug_assert!(dstSeqs.len() >= nbSequences);
    let mut longLen: usize = 0;
    for n in 0..nbSequences {
        dstSeqs[n].offBase = OFFSET_TO_OFFBASE(inSeqs[n].offset);
        dstSeqs[n].litLength = inSeqs[n].litLength as u16;
        dstSeqs[n].mlBase = (inSeqs[n].matchLength - MINMATCH) as u16;
        if inSeqs[n].matchLength > 65535 + MINMATCH {
            debug_assert_eq!(longLen, 0);
            longLen = n + 1;
        }
        if inSeqs[n].litLength > 65535 {
            debug_assert_eq!(longLen, 0);
            longLen = n + nbSequences + 1;
        }
    }
    longLen
}

/// Port of `ZSTD_convertBlockSequences`. Converts one externally
/// supplied block of public `ZSTD_Sequence` records into the internal
/// `SeqStore_t` representation, updating the outgoing rep history in
/// `cctx.next_rep`.
///
/// Preconditions mirror upstream:
///   - `nbSequences >= 1`
///   - the last sequence is an explicit block delimiter
///     (`matchLength == 0 && offset == 0`)
///
/// Returns `0` on success or `ExternalSequencesInvalid` on malformed
/// input / insufficient seqStore capacity.
pub fn ZSTD_convertBlockSequences(
    cctx: &mut ZSTD_CCtx,
    inSeqs: &[ZSTD_Sequence],
    repcodeResolution: bool,
) -> usize {
    use crate::compress::seq_store::{
        Repcodes_t, ZSTD_longLengthType_e, ZSTD_storeSeqOnly, ZSTD_updateRep, MINMATCH,
    };

    let nbSequences = inSeqs.len();
    if nbSequences == 0 {
        return ERROR(ErrorCode::ExternalSequencesInvalid);
    }

    if cctx.seqStore.is_none() {
        cctx.seqStore = Some(SeqStore_t::with_capacity(
            ZSTD_BLOCKSIZE_MAX / 3,
            ZSTD_BLOCKSIZE_MAX,
        ));
    }
    let seqStore = cctx.seqStore.as_mut().unwrap();

    if nbSequences >= seqStore.maxNbSeq {
        return ERROR(ErrorCode::ExternalSequencesInvalid);
    }
    seqStore.reset();

    if inSeqs[nbSequences - 1].matchLength != 0 || inSeqs[nbSequences - 1].offset != 0 {
        return ERROR(ErrorCode::ExternalSequencesInvalid);
    }

    let mut updatedRepcodes = Repcodes_t { rep: cctx.prev_rep };

    for seq in &inSeqs[..nbSequences - 1] {
        if seq.matchLength != 0 && seq.offset == 0 {
            return ERROR(ErrorCode::ExternalSequencesInvalid);
        }
        if seq.matchLength != 0 && seq.matchLength < MINMATCH {
            return ERROR(ErrorCode::ExternalSequencesInvalid);
        }
    }

    if !repcodeResolution {
        seqStore.sequences.resize(
            nbSequences - 1,
            crate::compress::seq_store::SeqDef::default(),
        );
        let longl =
            convertSequences_noRepcodes(&mut seqStore.sequences, &inSeqs[..nbSequences - 1]);
        if longl != 0 {
            debug_assert_eq!(
                seqStore.longLengthType,
                ZSTD_longLengthType_e::ZSTD_llt_none
            );
            if longl < nbSequences {
                seqStore.longLengthType = ZSTD_longLengthType_e::ZSTD_llt_matchLength;
                seqStore.longLengthPos = (longl - 1) as u32;
            } else {
                debug_assert!(longl <= 2 * (nbSequences - 1));
                seqStore.longLengthType = ZSTD_longLengthType_e::ZSTD_llt_literalLength;
                seqStore.longLengthPos = (longl - (nbSequences - 1) - 1) as u32;
            }
        }
    } else {
        for seq in &inSeqs[..nbSequences - 1] {
            let litLength = seq.litLength;
            let matchLength = seq.matchLength;
            let ll0 = (litLength == 0) as u32;
            let offBase = ZSTD_finalizeOffBase(seq.offset, &updatedRepcodes.rep, ll0);
            ZSTD_storeSeqOnly(seqStore, litLength as usize, offBase, matchLength as usize);
            ZSTD_updateRep(&mut updatedRepcodes.rep, offBase, ll0);
        }
    }

    if !repcodeResolution && nbSequences > 1 {
        let rep = &mut updatedRepcodes.rep;
        if nbSequences >= 4 {
            let lastSeqIdx = nbSequences - 2;
            rep[2] = inSeqs[lastSeqIdx - 2].offset;
            rep[1] = inSeqs[lastSeqIdx - 1].offset;
            rep[0] = inSeqs[lastSeqIdx].offset;
        } else if nbSequences == 3 {
            rep[2] = rep[0];
            rep[1] = inSeqs[0].offset;
            rep[0] = inSeqs[1].offset;
        } else {
            debug_assert_eq!(nbSequences, 2);
            rep[2] = rep[1];
            rep[1] = rep[0];
            rep[0] = inSeqs[0].offset;
        }
    }

    cctx.next_rep = updatedRepcodes.rep;
    0
}

/// Port of `ZSTD_transferSequences_wBlockDelim` (`zstd_compress.c:6669`).
/// Scans an external `ZSTD_Sequence` array, storing each sequence into the
/// CCtx seqStore until it hits a block delimiter (which also carries the
/// block's last literals). `blockSize` must equal the sum of sequence
/// lengths. Returns `blockSize` on success or a ZSTD error code.
fn ZSTD_transferSequences_wBlockDelim(
    cctx: &mut ZSTD_CCtx,
    seqPos: &mut ZSTD_SequencePosition,
    inSeqs: &[ZSTD_Sequence],
    src: &[u8],
    blockSize: usize,
    externalRepSearch: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
) -> usize {
    use crate::compress::seq_store::{
        Repcodes_t, ZSTD_storeLastLiterals, ZSTD_storeSeq, ZSTD_updateRep, MINMATCH,
        OFFSET_TO_OFFBASE,
    };
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;

    let mut idx = seqPos.idx as usize;
    let startIdx = idx;
    let mut ip = 0usize;
    let iend = blockSize;
    let mut updatedRepcodes = Repcodes_t { rep: cctx.prev_rep };
    let dictSize = cctx.dictContentSize;
    let seqStore = cctx.seqStore.get_or_insert_with(|| {
        SeqStore_t::with_capacity(ZSTD_BLOCKSIZE_MAX / 3, ZSTD_BLOCKSIZE_MAX)
    });

    while idx < inSeqs.len() && (inSeqs[idx].matchLength != 0 || inSeqs[idx].offset != 0) {
        let litLength = inSeqs[idx].litLength as usize;
        let matchLength = inSeqs[idx].matchLength as usize;
        if matchLength != 0 && inSeqs[idx].offset == 0 {
            return ERROR(ErrorCode::ExternalSequencesInvalid);
        }
        if matchLength != 0 && matchLength < MINMATCH as usize {
            return ERROR(ErrorCode::ExternalSequencesInvalid);
        }
        let offBase = if externalRepSearch == ZSTD_ParamSwitch_e::ZSTD_ps_disable {
            OFFSET_TO_OFFBASE(inSeqs[idx].offset)
        } else {
            let ll0 = (litLength == 0) as u32;
            let offBase = ZSTD_finalizeOffBase(inSeqs[idx].offset, &updatedRepcodes.rep, ll0);
            ZSTD_updateRep(&mut updatedRepcodes.rep, offBase, ll0);
            offBase
        };

        if cctx.appliedParams.validateSequences != 0 {
            let posInSrc = seqPos.posInSrc + litLength + matchLength;
            let rc = ZSTD_validateSequence(
                offBase,
                matchLength as u32,
                cctx.appliedParams.cParams.minMatch,
                posInSrc,
                cctx.appliedParams.cParams.windowLog,
                dictSize,
                ZSTD_hasExtSeqProd(&cctx.appliedParams),
            );
            if ERR_isError(rc) {
                return rc;
            }
        }
        if idx - seqPos.idx as usize >= seqStore.maxNbSeq {
            return ERROR(ErrorCode::ExternalSequencesInvalid);
        }
        if ip + litLength + matchLength > iend || ip + litLength > src.len() {
            return ERROR(ErrorCode::ExternalSequencesInvalid);
        }
        ZSTD_storeSeq(seqStore, litLength, &src[ip..], offBase, matchLength);
        ip += litLength + matchLength;
        seqPos.posInSrc += litLength + matchLength;
        idx += 1;
    }

    if idx == inSeqs.len() {
        return ERROR(ErrorCode::ExternalSequencesInvalid);
    }

    if externalRepSearch == ZSTD_ParamSwitch_e::ZSTD_ps_disable && idx != startIdx {
        let rep = &mut updatedRepcodes.rep;
        let lastSeqIdx = idx - 1;
        if lastSeqIdx >= startIdx + 2 {
            rep[2] = inSeqs[lastSeqIdx - 2].offset;
            rep[1] = inSeqs[lastSeqIdx - 1].offset;
            rep[0] = inSeqs[lastSeqIdx].offset;
        } else if lastSeqIdx == startIdx + 1 {
            rep[2] = rep[0];
            rep[1] = inSeqs[lastSeqIdx - 1].offset;
            rep[0] = inSeqs[lastSeqIdx].offset;
        } else {
            rep[2] = rep[1];
            rep[1] = rep[0];
            rep[0] = inSeqs[lastSeqIdx].offset;
        }
    }

    cctx.next_rep = updatedRepcodes.rep;
    if inSeqs[idx].litLength != 0 {
        let lastLL = inSeqs[idx].litLength as usize;
        if ip + lastLL != iend || ip + lastLL > src.len() {
            return ERROR(ErrorCode::ExternalSequencesInvalid);
        }
        ZSTD_storeLastLiterals(seqStore, &src[ip..ip + lastLL]);
        seqPos.posInSrc += lastLL;
        ip += lastLL;
    }
    if ip != iend {
        return ERROR(ErrorCode::ExternalSequencesInvalid);
    }
    seqPos.idx = (idx as u32).wrapping_add(1);
    blockSize
}

/// Port of `ZSTD_transferSequences_noDelim` (`zstd_compress.c:6769`).
/// No-delimiter mode: scans `blockSize` bytes of `src` worth of sequences,
/// splitting the trailing sequence if needed (with min-match guard) to
/// avoid producing a sub-minMatch tail match. Returns the number of bytes
/// consumed from `src` (≤ `blockSize`), or a ZSTD error code.
fn ZSTD_transferSequences_noDelim(
    cctx: &mut ZSTD_CCtx,
    seqPos: &mut ZSTD_SequencePosition,
    inSeqs: &[ZSTD_Sequence],
    src: &[u8],
    blockSize: usize,
    _externalRepSearch: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
) -> usize {
    use crate::compress::seq_store::{
        Repcodes_t, ZSTD_storeLastLiterals, ZSTD_storeSeq, ZSTD_updateRep, MINMATCH,
    };

    let mut idx = seqPos.idx as usize;
    let mut startPosInSequence = seqPos.posInSequence;
    let mut endPosInSequence = seqPos.posInSequence.wrapping_add(blockSize as u32);
    let mut ip = 0usize;
    let mut iend = blockSize;
    let mut updatedRepcodes = Repcodes_t { rep: cctx.prev_rep };
    let dictSize = cctx.dictContentSize;
    let seqStore = cctx.seqStore.get_or_insert_with(|| {
        SeqStore_t::with_capacity(ZSTD_BLOCKSIZE_MAX / 3, ZSTD_BLOCKSIZE_MAX)
    });
    let mut bytesAdjustment = 0u32;
    let mut finalMatchSplit = false;

    while endPosInSequence != 0 && idx < inSeqs.len() && !finalMatchSplit {
        let currSeq = inSeqs[idx];
        let mut litLength = currSeq.litLength;
        let mut matchLength = currSeq.matchLength;
        let rawOffset = currSeq.offset;

        if endPosInSequence >= currSeq.litLength.wrapping_add(currSeq.matchLength) {
            if startPosInSequence >= litLength {
                startPosInSequence = startPosInSequence.wrapping_sub(litLength);
                litLength = 0;
                matchLength = matchLength.wrapping_sub(startPosInSequence);
            } else {
                litLength = litLength.wrapping_sub(startPosInSequence);
            }
            endPosInSequence =
                endPosInSequence.wrapping_sub(currSeq.litLength.wrapping_add(currSeq.matchLength));
            startPosInSequence = 0;
        } else if endPosInSequence > litLength {
            litLength = if startPosInSequence >= litLength {
                0
            } else {
                litLength.wrapping_sub(startPosInSequence)
            };
            let mut firstHalfMatchLength = endPosInSequence
                .wrapping_sub(startPosInSequence)
                .wrapping_sub(litLength);
            if matchLength as usize > blockSize
                && firstHalfMatchLength >= cctx.appliedParams.cParams.minMatch
            {
                let secondHalfMatchLength = currSeq
                    .matchLength
                    .wrapping_add(currSeq.litLength)
                    .wrapping_sub(endPosInSequence);
                if secondHalfMatchLength < cctx.appliedParams.cParams.minMatch {
                    let adjust = cctx
                        .appliedParams
                        .cParams
                        .minMatch
                        .wrapping_sub(secondHalfMatchLength);
                    endPosInSequence = endPosInSequence.wrapping_sub(adjust);
                    bytesAdjustment = adjust;
                    firstHalfMatchLength = firstHalfMatchLength.wrapping_sub(adjust);
                }
                matchLength = firstHalfMatchLength;
                finalMatchSplit = true;
            } else {
                bytesAdjustment = endPosInSequence.wrapping_sub(currSeq.litLength);
                endPosInSequence = currSeq.litLength;
                break;
            }
        } else {
            break;
        }

        if matchLength != 0 && rawOffset == 0 {
            return ERROR(ErrorCode::ExternalSequencesInvalid);
        }
        if matchLength == 0 {
            return ERROR(ErrorCode::ExternalSequencesInvalid);
        }
        if matchLength < MINMATCH {
            return ERROR(ErrorCode::ExternalSequencesInvalid);
        }
        let ll0 = (litLength == 0) as u32;
        let offBase = ZSTD_finalizeOffBase(rawOffset, &updatedRepcodes.rep, ll0);
        ZSTD_updateRep(&mut updatedRepcodes.rep, offBase, ll0);
        if cctx.appliedParams.validateSequences != 0 {
            let posInSrc = seqPos.posInSrc + litLength as usize + matchLength as usize;
            let rc = ZSTD_validateSequence(
                offBase,
                matchLength,
                cctx.appliedParams.cParams.minMatch,
                posInSrc,
                cctx.appliedParams.cParams.windowLog,
                dictSize,
                ZSTD_hasExtSeqProd(&cctx.appliedParams),
            );
            if ERR_isError(rc) {
                return rc;
            }
        }
        if idx - seqPos.idx as usize >= seqStore.maxNbSeq {
            return ERROR(ErrorCode::ExternalSequencesInvalid);
        }
        let litLengthU = litLength as usize;
        let matchLengthU = matchLength as usize;
        if ip + litLengthU + matchLengthU > src.len() || ip + litLengthU + matchLengthU > blockSize
        {
            return ERROR(ErrorCode::ExternalSequencesInvalid);
        }
        ZSTD_storeSeq(seqStore, litLengthU, &src[ip..], offBase, matchLengthU);
        ip += litLengthU + matchLengthU;
        seqPos.posInSrc += litLengthU + matchLengthU;
        if !finalMatchSplit {
            idx += 1;
        }
    }

    seqPos.idx = idx as u32;
    seqPos.posInSequence = endPosInSequence;
    cctx.next_rep = updatedRepcodes.rep;

    iend = iend.saturating_sub(bytesAdjustment as usize);
    if ip > iend || iend > src.len() {
        return ERROR(ErrorCode::ExternalSequencesInvalid);
    }
    if ip != iend {
        ZSTD_storeLastLiterals(seqStore, &src[ip..iend]);
        seqPos.posInSrc += iend - ip;
    }
    iend
}

/// Port of `ZSTD_selectSequenceCopier` (`zstd_compress.c:6905`). Picks the
/// `wBlockDelim` vs `noDelim` sequence-transfer function based on the
/// caller-selected `ZSTD_SequenceFormat_e`.
fn ZSTD_selectSequenceCopier(
    mode: ZSTD_SequenceFormat_e,
) -> fn(
    &mut ZSTD_CCtx,
    &mut ZSTD_SequencePosition,
    &[ZSTD_Sequence],
    &[u8],
    usize,
    crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
) -> usize {
    match mode {
        ZSTD_SequenceFormat_e::ZSTD_sf_explicitBlockDelimiters => {
            ZSTD_transferSequences_wBlockDelim
        }
        ZSTD_SequenceFormat_e::ZSTD_sf_noBlockDelimiters => ZSTD_transferSequences_noDelim,
    }
}

/// Port of `ZSTD_compressSequences_internal` (`zstd_compress.c:6968`).
/// Compresses every caller-supplied sequence block-by-block, emitting raw /
/// RLE / compressed block bodies and their headers into `dst`. Returns the
/// cumulative compressed size or a ZSTD error code.
fn ZSTD_compressSequences_internal(
    cctx: &mut ZSTD_CCtx,
    dst: &mut [u8],
    inSeqs: &[ZSTD_Sequence],
    src: &[u8],
) -> usize {
    let mut cSize = 0usize;
    let mut remaining = src.len();
    let mut seqPos = ZSTD_SequencePosition::default();
    let mut ip = 0usize;
    let mut op = 0usize;
    let sequenceCopier = ZSTD_selectSequenceCopier(cctx.appliedParams.blockDelimiters);

    if remaining == 0 {
        if dst.len() < ZSTD_blockHeaderSize {
            return ERROR(ErrorCode::DstSizeTooSmall);
        }
        MEM_writeLE32(&mut dst[op..], 1);
        op += ZSTD_blockHeaderSize;
        cSize += ZSTD_blockHeaderSize;
    }

    while remaining != 0 {
        let targetBlockSize = determine_blockSize(
            cctx.appliedParams.blockDelimiters,
            cctx.blockSizeMax,
            remaining,
            inSeqs,
            seqPos,
        );
        if ERR_isError(targetBlockSize) {
            return targetBlockSize;
        }
        let lastBlock = (targetBlockSize == remaining) as u32;
        let seqStore = cctx.seqStore.get_or_insert_with(|| {
            SeqStore_t::with_capacity(ZSTD_BLOCKSIZE_MAX / 3, ZSTD_BLOCKSIZE_MAX)
        });
        seqStore.reset();
        let blockSize = sequenceCopier(
            cctx,
            &mut seqPos,
            inSeqs,
            &src[ip..ip + targetBlockSize],
            targetBlockSize,
            cctx.appliedParams.searchForExternalRepcodes,
        );
        if ERR_isError(blockSize) {
            return blockSize;
        }

        if blockSize < MIN_CBLOCK_SIZE + ZSTD_blockHeaderSize + 2 {
            let cBlockSize =
                ZSTD_noCompressBlock(&mut dst[op..], &src[ip..ip + blockSize], lastBlock);
            if ERR_isError(cBlockSize) {
                return cBlockSize;
            }
            cSize += cBlockSize;
            op += cBlockSize;
            ip += blockSize;
            remaining -= blockSize;
            continue;
        }

        if dst.len() - op < ZSTD_blockHeaderSize {
            return ERROR(ErrorCode::DstSizeTooSmall);
        }
        let compressedSeqsSize = {
            let seqStore = cctx.seqStore.as_mut().unwrap();
            let disableLiteralCompression = ZSTD_literalsCompressionIsDisabled(
                cctx.appliedParams.literalCompressionMode,
                cctx.appliedParams.cParams.strategy,
                cctx.appliedParams.cParams.targetLength,
            ) as i32;
            ZSTD_entropyCompressSeqStore_wksp(
                &mut dst[op + ZSTD_blockHeaderSize..],
                seqStore,
                &cctx.prevEntropy,
                &mut cctx.nextEntropy,
                cctx.appliedParams.cParams.strategy,
                disableLiteralCompression,
                blockSize,
                cctx.bmi2,
                &mut cctx.entropyScratch,
            )
        };
        if ERR_isError(compressedSeqsSize) {
            return compressedSeqsSize;
        }

        let compressedSeqsSize = if cctx.isFirstBlock == 0
            && ZSTD_maybeRLE(cctx.seqStore.as_ref().unwrap())
            && ZSTD_isRLE(&src[ip..ip + blockSize]) != 0
        {
            1
        } else {
            compressedSeqsSize
        };

        let cBlockSize = if compressedSeqsSize == 0 {
            ZSTD_noCompressBlock(&mut dst[op..], &src[ip..ip + blockSize], lastBlock)
        } else if compressedSeqsSize == 1 {
            ZSTD_rleCompressBlock(&mut dst[op..], src[ip], blockSize, lastBlock)
        } else {
            ZSTD_blockState_confirmRepcodesAndEntropyTables(cctx);
            if cctx.prevEntropy.fse.offcode_repeatMode == FSE_repeat::FSE_repeat_valid {
                cctx.prevEntropy.fse.offcode_repeatMode = FSE_repeat::FSE_repeat_check;
            }
            let cBlockHeader = lastBlock
                .wrapping_add(
                    (crate::decompress::zstd_decompress_block::blockType_e::bt_compressed as u32)
                        << 1,
                )
                .wrapping_add((compressedSeqsSize as u32) << 3);
            MEM_writeLE24(&mut dst[op..], cBlockHeader);
            ZSTD_blockHeaderSize + compressedSeqsSize
        };
        if ERR_isError(cBlockSize) {
            return cBlockSize;
        }
        cSize += cBlockSize;
        op += cBlockSize;
        ip += blockSize;
        remaining -= blockSize;
        cctx.isFirstBlock = 0;
    }

    cSize
}

/// Port of `ZSTD_compressSequences`. Compresses a caller-provided
/// sequence stream into a frame in `dst`.
pub fn ZSTD_compressSequences(
    cctx: &mut ZSTD_CCtx,
    dst: &mut [u8],
    sequences: &[ZSTD_Sequence],
    src: &[u8],
) -> usize {
    use crate::common::xxhash::{XXH64_digest, XXH64_update};

    let mut op = 0usize;
    let mut cSize = 0usize;
    let rc = ZSTD_CCtx_init_compressStream2(cctx, ZSTD_EndDirective::ZSTD_e_end, src.len());
    if ERR_isError(rc) {
        return rc;
    }

    let frameHeaderSize = ZSTD_writeFrameHeader_advanced(
        &mut dst[op..],
        &cctx.appliedParams.fParams,
        cctx.appliedParams.cParams.windowLog,
        src.len() as u64,
        cctx.dictID,
        cctx.appliedParams.format,
    );
    if ERR_isError(frameHeaderSize) {
        return frameHeaderSize;
    }
    op += frameHeaderSize;
    cSize += frameHeaderSize;

    if cctx.appliedParams.fParams.checksumFlag != 0 && !src.is_empty() {
        XXH64_update(&mut cctx.xxhState, src);
    }

    let cBlocksSize = ZSTD_compressSequences_internal(cctx, &mut dst[op..], sequences, src);
    if ERR_isError(cBlocksSize) {
        return cBlocksSize;
    }
    op += cBlocksSize;
    cSize += cBlocksSize;

    if cctx.appliedParams.fParams.checksumFlag != 0 {
        if dst.len() - op < 4 {
            return ERROR(ErrorCode::DstSizeTooSmall);
        }
        MEM_writeLE32(&mut dst[op..], XXH64_digest(&cctx.xxhState) as u32);
        cSize += 4;
    }
    cSize
}

/// Port of `ZSTD_mergeBlockDelimiters`. Filters block-boundary
/// sentinels (offset=0, matchLength=0) out of a sequence array,
/// rolling their `litLength` onto the next real sequence. Returns
/// the new trimmed length.
pub fn ZSTD_mergeBlockDelimiters(sequences: &mut [ZSTD_Sequence]) -> usize {
    let seqsSize = sequences.len();
    let mut out = 0usize;
    for input in 0..seqsSize {
        if sequences[input].offset == 0 && sequences[input].matchLength == 0 {
            if input != seqsSize - 1 {
                let carry = sequences[input].litLength;
                sequences[input + 1].litLength += carry;
            }
        } else {
            sequences[out] = sequences[input];
            out += 1;
        }
    }
    out
}

/// Port of `ZSTD_generateSequences`. Scans `src` block-by-block via
/// the configured CCtx match finder and projects each populated
/// `SeqStore_t` into caller-visible `ZSTD_Sequence` entries.
pub fn ZSTD_generateSequences(
    zc: &mut ZSTD_CCtx,
    outSeqs: &mut [ZSTD_Sequence],
    src: &[u8],
) -> usize {
    use crate::compress::match_state::{
        ZSTD_matchState_checkDictValidity, ZSTD_matchState_enforceMaxDist,
        ZSTD_overflowCorrectIfNeeded, ZSTD_window_update,
    };

    if src.is_empty() {
        if outSeqs.is_empty() {
            return ERROR(ErrorCode::DstSizeTooSmall);
        }
        outSeqs[0] = ZSTD_Sequence::default();
        return 1;
    }

    let rc = ZSTD_CCtx_init_compressStream2(zc, ZSTD_EndDirective::ZSTD_e_end, src.len());
    if ERR_isError(rc) {
        return rc;
    }

    let mut seqCollector = SeqCollector {
        collectSequences: 1,
        seqIndex: 0,
        maxSequences: outSeqs.len(),
    };

    let srcAbs = {
        let ms = zc.ms.get_or_insert_with(|| {
            crate::compress::match_state::ZSTD_MatchState_t::new(zc.appliedParams.cParams)
        });
        let srcAbs = ms.window.nextSrc;
        if !ZSTD_window_update(&mut ms.window, srcAbs, src.len(), false) {
            ms.nextToUpdate = ms.window.dictLimit;
        }
        srcAbs
    };

    let mut remaining = src.len();
    let mut ip = 0usize;
    let maxDist: u32 = 1u32 << zc.appliedParams.cParams.windowLog;

    while remaining != 0 {
        let blockSize = remaining.min(zc.blockSizeMax);
        let blockStartAbs = srcAbs.wrapping_add(ip as u32);
        let blockEndAbs = blockStartAbs.wrapping_add(blockSize as u32);

        {
            let ms = zc.ms.as_mut().unwrap();
            ZSTD_overflowCorrectIfNeeded(
                ms,
                zc.appliedParams.useRowMatchFinder,
                0,
                zc.appliedParams.cParams.windowLog,
                zc.appliedParams.cParams.chainLog,
                zc.appliedParams.cParams.strategy,
                blockStartAbs,
                blockEndAbs,
            );
            ZSTD_matchState_checkDictValidity(ms, blockEndAbs, maxDist);
            ZSTD_matchState_enforceMaxDist(ms, blockStartAbs, maxDist);
            if ms.nextToUpdate < ms.window.lowLimit {
                ms.nextToUpdate = ms.window.lowLimit;
            }
        }

        let prevRepcodes = zc.prev_rep;
        let bss = ZSTD_buildSeqStore_with_window(zc, src, ip, ip + blockSize);
        if ERR_isError(bss) {
            return bss;
        }

        if bss == ZSTD_BuildSeqStore_e::ZSTDbss_noCompress as usize {
            if seqCollector.seqIndex >= seqCollector.maxSequences {
                return ERROR(ErrorCode::DstSizeTooSmall);
            }
            outSeqs[seqCollector.seqIndex] = ZSTD_Sequence {
                offset: 0,
                litLength: blockSize as u32,
                matchLength: 0,
                rep: 0,
            };
            seqCollector.seqIndex += 1;
        } else {
            let seqStore = zc.seqStore.as_ref().unwrap();
            let rc = if zc.requestedParams.targetCBlockSize != 0 {
                ZSTD_copySuperBlockSequences(
                    &mut seqCollector,
                    outSeqs,
                    seqStore,
                    &prevRepcodes,
                    &zc.prevEntropy,
                    &mut zc.nextEntropy,
                    &zc.appliedParams,
                    &mut zc.entropyScratch,
                )
            } else {
                ZSTD_copyBlockSequences(&mut seqCollector, outSeqs, seqStore, &prevRepcodes)
            };
            if ERR_isError(rc) {
                return rc;
            }
            zc.prev_rep = zc.next_rep;
        }

        ip += blockSize;
        remaining -= blockSize;
        zc.isFirstBlock = 0;
    }

    if zc.appliedParams.blockDelimiters == ZSTD_SequenceFormat_e::ZSTD_sf_noBlockDelimiters {
        ZSTD_mergeBlockDelimiters(&mut outSeqs[..seqCollector.seqIndex])
    } else {
        seqCollector.seqIndex
    }
}

/// Port of `ZSTD_clearAllDicts`. Drops any cached dict state from
/// the CCtx — the raw-content dict, any ref-CDict linkage, and the
/// prefix-dict shadow — leaving the caller ready for a dict-free
/// session.
pub fn ZSTD_clearAllDicts(cctx: &mut ZSTD_CCtx) {
    cctx.stream_cdict = None;
    cctx.stream_dict.clear();
    cctx.stream_dict_original.clear();
    cctx.stream_dict_content_type =
        crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_auto;
    cctx.dictID = 0;
    cctx.dictContentSize = 0;
    if let Some(ms) = cctx.ms.as_mut() {
        ms.dictMatchState = None;
        ms.dictContent.clear();
        ms.loadedDictEnd = 0;
    }
    // Also clear the single-usage marker — after clearAllDicts the
    // cctx has no dict at all, so the flag shouldn't report "there's
    // a prefix waiting to be consumed" to the next compress.
    cctx.prefix_is_single_use = false;
}

/// Port of `ZSTD_overrideCParams`. For each non-zero field in
/// `overrides`, write it into `cParams`. Used to apply caller-
/// supplied cParam overrides on top of level-derived defaults.
pub fn ZSTD_overrideCParams(
    cParams: &mut crate::compress::match_state::ZSTD_compressionParameters,
    overrides: &crate::compress::match_state::ZSTD_compressionParameters,
) {
    if overrides.windowLog != 0 {
        cParams.windowLog = overrides.windowLog;
    }
    if overrides.hashLog != 0 {
        cParams.hashLog = overrides.hashLog;
    }
    if overrides.chainLog != 0 {
        cParams.chainLog = overrides.chainLog;
    }
    if overrides.searchLog != 0 {
        cParams.searchLog = overrides.searchLog;
    }
    if overrides.minMatch != 0 {
        cParams.minMatch = overrides.minMatch;
    }
    if overrides.targetLength != 0 {
        cParams.targetLength = overrides.targetLength;
    }
    if overrides.strategy != 0 {
        cParams.strategy = overrides.strategy;
    }
}

/// Port of `ZSTD_assertEqualCParams`. Debug-only sanity check — panics
/// in debug builds if every cParams field doesn't match.
#[inline]
pub fn ZSTD_assertEqualCParams(
    a: crate::compress::match_state::ZSTD_compressionParameters,
    b: crate::compress::match_state::ZSTD_compressionParameters,
) {
    debug_assert_eq!(a.windowLog, b.windowLog);
    debug_assert_eq!(a.chainLog, b.chainLog);
    debug_assert_eq!(a.hashLog, b.hashLog);
    debug_assert_eq!(a.searchLog, b.searchLog);
    debug_assert_eq!(a.minMatch, b.minMatch);
    debug_assert_eq!(a.targetLength, b.targetLength);
    debug_assert_eq!(a.strategy, b.strategy);
}

/// Port of `ZSTD_fastSequenceLengthSum`. Sums `litLength +
/// matchLength` over a `ZSTD_Sequence` array — used by upstream to
/// compute total input bytes the sequence stream describes.
/// Upstream's simd-friendly no-early-exit variant; we mirror the
/// shape for code-complexity-comparator.
pub fn ZSTD_fastSequenceLengthSum(seqs: &[ZSTD_Sequence]) -> usize {
    let mut lit = 0usize;
    let mut ml = 0usize;
    for s in seqs {
        lit += s.litLength as usize;
        ml += s.matchLength as usize;
    }
    lit + ml
}

/// Port of `ZSTD_safecopyLiterals` (`zstd_compress_internal.h:706`).
/// Copies `src[..iend]` into `dst` without assuming the caller can
/// read past `ilimit_w`, mirroring upstream's "wildcopy until the safe
/// boundary, then byte-copy the tail" contract.
pub fn ZSTD_safecopyLiterals(dst: &mut [u8], src: &[u8], iend: usize, ilimit_w: usize) -> usize {
    debug_assert!(iend > ilimit_w);
    let mut copied = 0usize;
    if copied <= ilimit_w {
        let n = ilimit_w - copied;
        zstd_safecopy_copy16(dst, copied, src, copied);
        if 16 < n {
            let mut wild_op = copied + 16;
            let mut wild_ip = copied + 16;
            let wild_oend = copied + n;
            while wild_op < wild_oend {
                zstd_safecopy_copy16(dst, wild_op, src, wild_ip);
                wild_op += 16;
                wild_ip += 16;
                zstd_safecopy_copy16(dst, wild_op, src, wild_ip);
                wild_op += 16;
                wild_ip += 16;
            }
        }
        copied += n;
    }
    while copied < iend {
        dst[copied] = src[copied];
        copied += 1;
    }
    copied
}

#[inline]
fn zstd_safecopy_copy16(dst: &mut [u8], op: usize, src: &[u8], ip: usize) {
    let a = MEM_read64(&src[ip..ip + 8]);
    let b = MEM_read64(&src[ip + 8..ip + 16]);
    MEM_write64(&mut dst[op..op + 8], a);
    MEM_write64(&mut dst[op + 8..op + 16], b);
}

/// Port of `ZSTD_validateSequence`. Checks that a sequence's offBase
/// fits within the effective window + dict bound, and that its match
/// length clears the configured `minMatch` floor.
///
/// Returns `0` on success or an `externalSequences_invalid` error.
pub fn ZSTD_validateSequence(
    offBase: u32,
    matchLength: u32,
    minMatch: u32,
    posInSrc: usize,
    windowLog: u32,
    dictSize: usize,
    useSequenceProducer: bool,
) -> usize {
    let windowSize: usize = 1usize << windowLog;
    // Once we've decoded more than `windowSize` bytes, we can't
    // reference anything further back than that. Before that, dict
    // bytes are also reachable.
    let offsetBound = if posInSrc > windowSize {
        windowSize
    } else {
        posInSrc + dictSize
    };
    let matchLenLowerBound = if minMatch == 3 || useSequenceProducer {
        3
    } else {
        4
    };
    let maxOffBase = (offsetBound as u32)
        .wrapping_add(crate::decompress::zstd_decompress_block::ZSTD_REP_NUM as u32);
    if offBase > maxOffBase {
        return ERROR(ErrorCode::ExternalSequencesInvalid);
    }
    if matchLength < matchLenLowerBound {
        return ERROR(ErrorCode::ExternalSequencesInvalid);
    }
    0
}

/// Port of `ZSTD_finalizeOffBase`. Given a raw offset value, the
/// current repcode history, and the litLength-zero flag, returns the
/// offBase sumtype code to store — collapsing to `REPCODE1/2/3` when
/// the offset matches the corresponding `rep[]` slot.
pub fn ZSTD_finalizeOffBase(
    rawOffset: u32,
    rep: &[u32; crate::decompress::zstd_decompress_block::ZSTD_REP_NUM],
    ll0: u32,
) -> u32 {
    use crate::compress::seq_store::{OFFSET_TO_OFFBASE, REPCODE_TO_OFFBASE};
    if ll0 == 0 && rawOffset == rep[0] {
        return REPCODE_TO_OFFBASE(1);
    }
    if rawOffset == rep[1] {
        return REPCODE_TO_OFFBASE(2 - ll0);
    }
    if rawOffset == rep[2] {
        return REPCODE_TO_OFFBASE(3 - ll0);
    }
    if ll0 != 0 && rawOffset == rep[0].saturating_sub(1) && rawOffset != 0 {
        return REPCODE_TO_OFFBASE(3);
    }
    OFFSET_TO_OFFBASE(rawOffset)
}

/// Port of `ZSTD_sequenceBound`. Upper bound on the number of
/// `ZSTD_Sequence` entries an `srcSize`-byte input can emit —
/// `(srcSize / MINMATCH_MIN) + 1 + (srcSize / BLOCKSIZE_MAX_MIN) + 1`.
#[inline]
pub fn ZSTD_sequenceBound(srcSize: usize) -> usize {
    const ZSTD_MINMATCH_MIN: usize = 3;
    const ZSTD_BLOCKSIZE_MAX_MIN: usize = 1 << 10;
    let maxNbSeq = (srcSize / ZSTD_MINMATCH_MIN) + 1;
    let maxNbDelims = (srcSize / ZSTD_BLOCKSIZE_MAX_MIN) + 1;
    maxNbSeq + maxNbDelims
}

/// Port of `ZSTD_toFlushNow`. Reports how many bytes are still buffered
/// inside the compressor awaiting a flush call. v0.1 is single-thread
/// and the streaming path doesn't buffer past `endStream`, so this
/// returns 0 — upstream does the same on non-MT builds.
#[inline]
pub fn ZSTD_toFlushNow(_cctx: &ZSTD_CCtx) -> usize {
    0
}

/// Port of `ZSTD_CCtx_params`. Cut-down version exposing the public
/// API surface. Upstream carries many more fields (LDM params, MT
/// job size, etc.) — v0.1 stores the policy knobs whose resolvers
/// the init helpers consult.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZSTD_CCtx_params {
    pub compressionLevel: i32,
    pub cParams: crate::compress::match_state::ZSTD_compressionParameters,
    pub fParams: ZSTD_FrameParameters,
    /// Upstream `params.format`. Matches `ZSTD_CCtx.format`. Only
    /// `CCtxParams_setParameter` / `_getParameter` consult this slot;
    /// the compressor path reads from `cctx.format`.
    pub format: crate::decompress::zstd_decompress::ZSTD_format_e,
    /// `ZSTD_c_useRowMatchFinder` — picks the row-hash matcher family.
    pub useRowMatchFinder: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
    /// `ZSTD_c_useBlockSplitter` — controls the post-compress
    /// block-splitter pass.
    pub postBlockSplitter: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
    /// Upstream `preBlockSplitter_level`. Controls the cheaper
    /// content-shift splitter that runs before sequence production.
    /// 0 = auto, 1 = disable, 2..=6 = increasing CPU budget.
    pub preBlockSplitter_level: i32,
    /// `ZSTD_c_enableLongDistanceMatching` — turns on the LDM
    /// long-distance matcher.
    pub ldmEnable: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
    /// `ZSTD_c_validateSequences` — 1 to validate caller-supplied
    /// sequences; 0 to trust them.
    pub validateSequences: i32,
    /// `ZSTD_c_maxBlockSize` — upper bound on block size in bytes;
    /// 0 means "default to `ZSTD_BLOCKSIZE_MAX`".
    pub maxBlockSize: usize,
    /// `ZSTD_c_searchForExternalRepcodes` — auto / enable / disable.
    pub searchForExternalRepcodes: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
    /// `ZSTD_c_targetCBlockSize` — when nonzero, aim for this compressed
    /// block size (in bytes). Enables the superblock compressor path.
    pub targetCBlockSize: usize,
    /// `ZSTD_c_attachDictPref` — controls attach-vs-copy behavior for
    /// CDicts. Default is `ZSTD_dictDefaultAttach` (auto).
    pub attachDictPref: ZSTD_dictAttachPref_e,
    /// `ZSTD_c_forceMaxWindow` — when set, the compressor forces its
    /// window size to the frame's windowLog rather than shrinking to
    /// the actual content size.
    pub forceWindow: i32,
    /// `ZSTD_c_srcSizeHint` — advisory source size passed to
    /// `ZSTD_getCParams_internal` when the exact pledged size is
    /// unknown.
    pub srcSizeHint: i32,
    /// `ZSTD_c_stableInBuffer` — whether the caller guarantees the
    /// input buffer pointer/consumed prefix remains stable across
    /// streaming calls.
    pub inBufferMode: ZSTD_bufferMode_e,
    /// `ZSTD_c_stableOutBuffer` — whether the caller guarantees the
    /// remaining output capacity never grows across streaming calls.
    pub outBufferMode: ZSTD_bufferMode_e,
    /// `ZSTD_c_blockDelimiters` — picks between implicit and explicit
    /// block delimiters in caller-supplied sequence mode.
    pub blockDelimiters: ZSTD_SequenceFormat_e,
    /// `ZSTD_c_literalCompressionMode` — auto / force-on / force-off
    /// for HUF literal compression. Upstream deprecated; kept for
    /// API parity.
    pub literalCompressionMode: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
    /// `ZSTD_c_nbWorkers` — multithreading worker count. v0.1 is
    /// single-threaded so this is an API-only knob.
    pub nbWorkers: i32,
    /// `ZSTD_c_enableDedicatedDictSearch` — opt-in to DDSS hash
    /// inflation when building a CDict.
    pub enableDedicatedDictSearch: i32,
    /// `ZSTD_c_deterministicRefPrefix` — forces the `refPrefix` path
    /// to use the same window-setup rules regardless of frame state.
    pub deterministicRefPrefix: i32,
    /// `ZSTD_c_prefetchCDictTables` — auto / enable / disable prefetch
    /// policy for attached CDict tables.
    pub prefetchCDictTables: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
    /// Upstream `overlapLog` for MT job overlap sizing.
    pub overlapLog: i32,
    /// Upstream `jobSize` for MT section sizing.
    pub jobSize: usize,
    /// Upstream `rsyncable` toggle for synchronization-point splitting.
    pub rsyncable: i32,
    /// Upstream embedded LDM parameter bundle used by MT.
    pub ldmParams: crate::compress::zstd_ldm::ldmParams_t,
    /// Upstream external sequence producer opaque state pointer.
    pub extSeqProdState: usize,
    /// Upstream external sequence producer callback.
    pub extSeqProdFunc: Option<ZSTD_sequenceProducer_F>,
    /// Upstream `enableMatchFinderFallback` / `ZSTD_c_enableSeqProducerFallback`.
    /// When nonzero, an external sequence producer error falls back to the
    /// internal matchfinder on a block-by-block basis.
    pub enableMatchFinderFallback: i32,
    /// Upstream custom allocator bundle.
    pub customMem: ZSTD_customMem,
}

/// Port of `ZSTD_createCCtxParams`. Allocates + initializes with the
/// default level. Returns `None` only on allocation failure.
pub fn ZSTD_createCCtxParams() -> Option<Box<ZSTD_CCtx_params>> {
    let mut p =
        unsafe { ZSTD_customAllocBox(ZSTD_CCtx_params::default(), ZSTD_customMem::default())? };
    ZSTD_CCtxParams_init(&mut p, ZSTD_CLEVEL_DEFAULT);
    Some(p)
}

/// Port of `ZSTD_createCCtxParams_advanced`.
pub fn ZSTD_createCCtxParams_advanced(customMem: ZSTD_customMem) -> Option<Box<ZSTD_CCtx_params>> {
    if !ZSTD_customMem_validate(customMem) {
        return None;
    }
    let mut p = unsafe { ZSTD_customAllocBox(ZSTD_CCtx_params::default(), customMem)? };
    ZSTD_CCtxParams_init(&mut p, ZSTD_CLEVEL_DEFAULT);
    p.customMem = customMem;
    Some(p)
}

/// Port of `ZSTD_freeCCtxParams`. Drops the Box.
#[inline]
pub fn ZSTD_freeCCtxParams(_params: Option<Box<ZSTD_CCtx_params>>) -> usize {
    if let Some(params) = _params {
        let customMem = params.customMem;
        unsafe {
            ZSTD_customFreeBox(params, customMem);
        }
    }
    0
}

/// Port of `ZSTD_CCtxParams_reset`. Reinitialize to default level.
#[inline]
pub fn ZSTD_CCtxParams_reset(params: &mut ZSTD_CCtx_params) -> usize {
    ZSTD_CCtxParams_init(params, ZSTD_CLEVEL_DEFAULT)
}

/// Port of `ZSTD_CCtxParams_init`. Zeros the struct, sets the level,
/// and turns on `contentSizeFlag` (upstream default).
pub fn ZSTD_CCtxParams_init(params: &mut ZSTD_CCtx_params, compressionLevel: i32) -> usize {
    let customMem = params.customMem;
    *params = ZSTD_CCtx_params::default();
    params.customMem = customMem;
    params.compressionLevel = compressionLevel;
    params.fParams.contentSizeFlag = 1;
    0
}

/// Port of upstream's `repStartValue` (`zstd_internal.h:65`). The
/// canonical initial repcode history at frame start — also the value
/// that `ZSTD_reset_compressedBlockState` restores.
pub const ZSTD_REP_START_VALUE: [u32; 3] = [1, 4, 8];

/// Rust-only helper: decodes the `pledgedSrcSizePlusOne` slot into the
/// effective pledged source size, mapping `0` to `ZSTD_CONTENTSIZE_UNKNOWN`.
#[inline]
fn ZSTD_getPledgedSrcSize(cctx: &ZSTD_CCtx) -> u64 {
    if cctx.pledgedSrcSizePlusOne == 0 {
        crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN
    } else {
        cctx.pledgedSrcSizePlusOne - 1
    }
}

/// Port of `ZSTD_reset_compressedBlockState` (`zstd_compress.c:1942`).
/// Restores the upstream-default repcode history and marks every
/// entropy table's repeatMode as `none` so the next block must
/// re-emit fresh tables rather than reusing stale ones.
///
/// Our CCtx carries `prev_rep` + `prevEntropy` directly (no shadowed
/// `next` fields to rotate), so this helper operates on those.
/// Caller should also reset `next_rep` + `nextEntropy` if they're
/// about to start a fresh frame.
pub fn ZSTD_reset_compressedBlockState(rep: &mut [u32; 3], entropy: &mut ZSTD_entropyCTables_t) {
    *rep = ZSTD_REP_START_VALUE;
    entropy.huf.repeatMode = crate::compress::zstd_compress_literals::HUF_repeat::HUF_repeat_none;
    entropy.fse.offcode_repeatMode =
        crate::compress::zstd_compress_sequences::FSE_repeat::FSE_repeat_none;
    entropy.fse.matchlength_repeatMode =
        crate::compress::zstd_compress_sequences::FSE_repeat::FSE_repeat_none;
    entropy.fse.litlength_repeatMode =
        crate::compress::zstd_compress_sequences::FSE_repeat::FSE_repeat_none;
}

/// Port of `writeBlockHeader` (`zstd_compress.c:3638`). Writes the
/// 3-byte little-endian block header. `cSize == 1` signals an RLE
/// block (`blockSize` carries the uncompressed length); otherwise
/// signals a compressed block (`cSize` carries the compressed length).
/// `lastBlock` is a bool encoded as 0/1 in the LSB.
#[inline]
pub fn writeBlockHeader(op: &mut [u8], cSize: usize, blockSize: usize, lastBlock: u32) {
    use crate::common::mem::MEM_writeLE24;
    use crate::decompress::zstd_decompress_block::blockType_e;
    let cBlockHeader: u32 = if cSize == 1 {
        lastBlock
            .wrapping_add((blockType_e::bt_rle as u32) << 1)
            .wrapping_add((blockSize as u32) << 3)
    } else {
        lastBlock
            .wrapping_add((blockType_e::bt_compressed as u32) << 1)
            .wrapping_add((cSize as u32) << 3)
    };
    MEM_writeLE24(op, cBlockHeader);
}

/// Port of `ZSTD_blockState_confirmRepcodesAndEntropyTables`
/// (`zstd_compress.c:3630`). After a block is accepted upstream
/// swaps the `prev`/`next` pointers so the newly-emitted entropy
/// tables and repcodes become the baseline for the next block. Our
/// CCtx owns `prev_*` + `next_*` directly (not as pointers), so we
/// swap the contents with `std::mem::swap`.
pub fn ZSTD_blockState_confirmRepcodesAndEntropyTables(cctx: &mut ZSTD_CCtx) {
    core::mem::swap(&mut cctx.prev_rep, &mut cctx.next_rep);
    core::mem::swap(&mut cctx.prevEntropy, &mut cctx.nextEntropy);
}

/// Port of `ZSTD_invalidateRepCodes` (`zstd_compress.c:2310`). Zeros
/// every entry of the CCtx's live repcode history (the "previous
/// block" view that the next block's rep-check will consult).
pub fn ZSTD_invalidateRepCodes(cctx: &mut ZSTD_CCtx) {
    for slot in cctx.prev_rep.iter_mut() {
        *slot = 0;
    }
}

/// Port of `ZSTD_getCParamsFromCCtxParams` (`zstd_compress.c:1653`).
/// Derives the effective `ZSTD_compressionParameters` from a params
/// struct + srcSizeHint + dictSize + mode. Flow:
///   1. If srcSizeHint is UNKNOWN and `params.srcSizeHint > 0`, use
///      that stored hint (v0.1 doesn't carry `srcSizeHint` on our
///      params struct, so this branch is inert).
///   2. Pick baseline cParams via `ZSTD_getCParams`.
///   3. Upstream widens windowLog to `ZSTD_LDM_DEFAULT_WINDOW_LOG`
///      when LDM is enabled.
///   4. Apply `ZSTD_overrideCParams` to fold in any explicitly-set
///      fields from `params.cParams`.
///   5. Final `ZSTD_adjustCParams_internal` clamp against window /
///      dict / source constraints.
pub fn ZSTD_getCParamsFromCCtxParams(
    CCtxParams: &ZSTD_CCtx_params,
    srcSizeHint: u64,
    dictSize: usize,
    mode: ZSTD_CParamMode_e,
) -> crate::compress::match_state::ZSTD_compressionParameters {
    use crate::compress::zstd_ldm::{ZSTD_ParamSwitch_e, ZSTD_LDM_DEFAULT_WINDOW_LOG};
    use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;
    // Upstream: if srcSizeHint is UNKNOWN and params.srcSizeHint > 0,
    // use the stored hint.
    let effective_srcSizeHint =
        if srcSizeHint == ZSTD_CONTENTSIZE_UNKNOWN && CCtxParams.srcSizeHint > 0 {
            CCtxParams.srcSizeHint as u64
        } else {
            srcSizeHint
        };
    let mut cParams = ZSTD_getCParams(CCtxParams.compressionLevel, effective_srcSizeHint, dictSize);
    // LDM widens windowLog when enabled.
    if CCtxParams.ldmParams.enableLdm == ZSTD_ParamSwitch_e::ZSTD_ps_enable {
        cParams.windowLog = ZSTD_LDM_DEFAULT_WINDOW_LOG;
    }
    ZSTD_overrideCParams(&mut cParams, &CCtxParams.cParams);
    ZSTD_adjustCParams_internal(
        cParams,
        effective_srcSizeHint,
        dictSize as u64,
        mode,
        CCtxParams.useRowMatchFinder,
    )
}

/// Port of `ZSTD_makeCCtxParamsFromCParams` (`zstd_compress.c:319`).
/// Produces a `ZSTD_CCtx_params` struct seeded with the default level
/// but overriding `cParams` to the caller-supplied value. The
/// resolver calls (`ZSTD_resolveEnableLdm`, `ZSTD_resolveBlockSplitterMode`,
/// …) mirror upstream's advanced-parameter normalization.
pub fn ZSTD_makeCCtxParamsFromCParams(
    cParams: crate::compress::match_state::ZSTD_compressionParameters,
) -> ZSTD_CCtx_params {
    use crate::compress::match_state::{
        ZSTD_resolveBlockSplitterMode, ZSTD_resolveEnableLdm, ZSTD_resolveExternalRepcodeSearch,
        ZSTD_resolveExternalSequenceValidation, ZSTD_resolveMaxBlockSize,
        ZSTD_resolveRowMatchFinderMode,
    };
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;

    let mut cctxParams = ZSTD_CCtx_params::default();
    ZSTD_CCtxParams_init(&mut cctxParams, ZSTD_CLEVEL_DEFAULT);
    cctxParams.cParams = cParams;

    cctxParams.ldmParams.enableLdm =
        ZSTD_resolveEnableLdm(cctxParams.ldmParams.enableLdm, &cParams);
    cctxParams.ldmEnable = cctxParams.ldmParams.enableLdm;
    cctxParams.postBlockSplitter =
        ZSTD_resolveBlockSplitterMode(ZSTD_ParamSwitch_e::ZSTD_ps_auto, &cParams);
    cctxParams.useRowMatchFinder =
        ZSTD_resolveRowMatchFinderMode(ZSTD_ParamSwitch_e::ZSTD_ps_auto, &cParams);
    cctxParams.validateSequences = ZSTD_resolveExternalSequenceValidation(0);
    cctxParams.maxBlockSize = ZSTD_resolveMaxBlockSize(0);
    cctxParams.searchForExternalRepcodes = ZSTD_resolveExternalRepcodeSearch(
        ZSTD_ParamSwitch_e::ZSTD_ps_auto,
        cctxParams.compressionLevel,
    );

    debug_assert!(!crate::common::error::ERR_isError(ZSTD_checkCParams(
        cParams
    )));
    cctxParams
}

/// Port of `ZSTD_CCtxParams_init_internal` (`zstd_compress.c:388`).
/// Zeroes the params, installs the caller's cParams / fParams, sets
/// `compressionLevel`, and folds auto-resolvable policy knobs through
/// their resolvers (`ZSTD_resolveRowMatchFinderMode`,
/// `ZSTD_resolveBlockSplitterMode`, `ZSTD_resolveEnableLdm`,
/// `ZSTD_resolveExternalSequenceValidation`, `ZSTD_resolveMaxBlockSize`,
/// `ZSTD_resolveExternalRepcodeSearch`).
pub fn ZSTD_CCtxParams_init_internal(
    cctxParams: &mut ZSTD_CCtx_params,
    zstdParams: &ZSTD_parameters,
    compressionLevel: i32,
) {
    debug_assert!(!crate::common::error::ERR_isError(ZSTD_checkCParams(
        zstdParams.cParams
    )));
    use crate::compress::match_state::{
        ZSTD_resolveBlockSplitterMode, ZSTD_resolveEnableLdm, ZSTD_resolveExternalRepcodeSearch,
        ZSTD_resolveExternalSequenceValidation, ZSTD_resolveMaxBlockSize,
        ZSTD_resolveRowMatchFinderMode,
    };
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;

    *cctxParams = ZSTD_CCtx_params::default();
    cctxParams.compressionLevel = compressionLevel;
    cctxParams.cParams = zstdParams.cParams;
    cctxParams.fParams = zstdParams.fParams;
    cctxParams.useRowMatchFinder =
        ZSTD_resolveRowMatchFinderMode(ZSTD_ParamSwitch_e::ZSTD_ps_auto, &zstdParams.cParams);
    cctxParams.postBlockSplitter =
        ZSTD_resolveBlockSplitterMode(ZSTD_ParamSwitch_e::ZSTD_ps_auto, &zstdParams.cParams);
    cctxParams.ldmParams.enableLdm =
        ZSTD_resolveEnableLdm(cctxParams.ldmParams.enableLdm, &zstdParams.cParams);
    cctxParams.ldmEnable = cctxParams.ldmParams.enableLdm;
    cctxParams.validateSequences = ZSTD_resolveExternalSequenceValidation(0);
    cctxParams.maxBlockSize = ZSTD_resolveMaxBlockSize(0);
    cctxParams.searchForExternalRepcodes =
        ZSTD_resolveExternalRepcodeSearch(ZSTD_ParamSwitch_e::ZSTD_ps_auto, compressionLevel);
}

/// Port of `ZSTD_CCtxParams_init_advanced`. Seeds from a full
/// `ZSTD_parameters` struct after a bounds check.
pub fn ZSTD_CCtxParams_init_advanced(
    params: &mut ZSTD_CCtx_params,
    zstdParams: ZSTD_parameters,
) -> usize {
    let rc = ZSTD_checkCParams(zstdParams.cParams);
    if ERR_isError(rc) {
        return rc;
    }
    ZSTD_CCtxParams_init_internal(params, &zstdParams, ZSTD_NO_CLEVEL);
    0
}

/// Port of `ZSTD_CCtxParams_setParameter`. Thin forwarder — mirrors
/// the `ZSTD_CCtx_setParameter` shape but writes into the caller's
/// standalone params struct rather than a CCtx.
pub fn ZSTD_CCtxParams_setParameter(
    params: &mut ZSTD_CCtx_params,
    param: ZSTD_cParameter,
    value: i32,
) -> usize {
    match param {
        ZSTD_cParameter::ZSTD_c_compressionLevel => {
            // Upstream clamps via `ZSTD_cParam_clampBounds` + maps 0
            // to `ZSTD_CLEVEL_DEFAULT`. Mirror here so CCtxParams-
            // level callers see the same readback as CCtx-level ones.
            let mut clamped = value;
            let rc =
                ZSTD_cParam_clampBounds(ZSTD_cParameter::ZSTD_c_compressionLevel, &mut clamped);
            if ERR_isError(rc) {
                return rc;
            }
            params.compressionLevel = if clamped == 0 {
                ZSTD_CLEVEL_DEFAULT
            } else {
                clamped
            };
            if params.compressionLevel >= 0 {
                params.compressionLevel as usize
            } else {
                0
            }
        }
        ZSTD_cParameter::ZSTD_c_windowLog
        | ZSTD_cParameter::ZSTD_c_hashLog
        | ZSTD_cParameter::ZSTD_c_chainLog
        | ZSTD_cParameter::ZSTD_c_searchLog
        | ZSTD_cParameter::ZSTD_c_minMatch
        | ZSTD_cParameter::ZSTD_c_targetLength
        | ZSTD_cParameter::ZSTD_c_strategy => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            match param {
                ZSTD_cParameter::ZSTD_c_windowLog => params.cParams.windowLog = value as u32,
                ZSTD_cParameter::ZSTD_c_hashLog => params.cParams.hashLog = value as u32,
                ZSTD_cParameter::ZSTD_c_chainLog => params.cParams.chainLog = value as u32,
                ZSTD_cParameter::ZSTD_c_searchLog => params.cParams.searchLog = value as u32,
                ZSTD_cParameter::ZSTD_c_minMatch => params.cParams.minMatch = value as u32,
                ZSTD_cParameter::ZSTD_c_targetLength => params.cParams.targetLength = value as u32,
                ZSTD_cParameter::ZSTD_c_strategy => params.cParams.strategy = value as u32,
                _ => unreachable!(),
            }
            value as usize
        }
        ZSTD_cParameter::ZSTD_c_enableLongDistanceMatching => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            params.ldmEnable = match value {
                1 => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_enable,
                2 => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_disable,
                _ => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_auto,
            };
            params.ldmParams.enableLdm = params.ldmEnable;
            params.ldmEnable as usize
        }
        ZSTD_cParameter::ZSTD_c_ldmHashLog
        | ZSTD_cParameter::ZSTD_c_ldmMinMatch
        | ZSTD_cParameter::ZSTD_c_ldmBucketSizeLog
        | ZSTD_cParameter::ZSTD_c_ldmHashRateLog => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            match param {
                ZSTD_cParameter::ZSTD_c_ldmHashLog => params.ldmParams.hashLog = value as u32,
                ZSTD_cParameter::ZSTD_c_ldmMinMatch => {
                    params.ldmParams.minMatchLength = value as u32
                }
                ZSTD_cParameter::ZSTD_c_ldmBucketSizeLog => {
                    params.ldmParams.bucketSizeLog = value as u32
                }
                ZSTD_cParameter::ZSTD_c_ldmHashRateLog => {
                    params.ldmParams.hashRateLog = value as u32
                }
                _ => unreachable!(),
            }
            value as usize
        }
        ZSTD_cParameter::ZSTD_c_targetCBlockSize => {
            let mut adjusted = value;
            if adjusted != 0 && adjusted < ZSTD_TARGETCBLOCKSIZE_MIN as i32 {
                adjusted = ZSTD_TARGETCBLOCKSIZE_MIN as i32;
            }
            let rc = ZSTD_validateCParamSetterValue(param, adjusted);
            if ERR_isError(rc) {
                return rc;
            }
            params.targetCBlockSize = adjusted as usize;
            params.targetCBlockSize
        }
        ZSTD_cParameter::ZSTD_c_stableInBuffer => {
            let bounds = ZSTD_cParam_getBounds(ZSTD_cParameter::ZSTD_c_stableInBuffer);
            if value < bounds.lowerBound || value > bounds.upperBound {
                return ERROR(ErrorCode::ParameterOutOfBound);
            }
            params.inBufferMode = if value == ZSTD_bufferMode_e::ZSTD_bm_stable as i32 {
                ZSTD_bufferMode_e::ZSTD_bm_stable
            } else {
                ZSTD_bufferMode_e::ZSTD_bm_buffered
            };
            params.inBufferMode as usize
        }
        ZSTD_cParameter::ZSTD_c_stableOutBuffer => {
            let bounds = ZSTD_cParam_getBounds(ZSTD_cParameter::ZSTD_c_stableOutBuffer);
            if value < bounds.lowerBound || value > bounds.upperBound {
                return ERROR(ErrorCode::ParameterOutOfBound);
            }
            params.outBufferMode = if value == ZSTD_bufferMode_e::ZSTD_bm_stable as i32 {
                ZSTD_bufferMode_e::ZSTD_bm_stable
            } else {
                ZSTD_bufferMode_e::ZSTD_bm_buffered
            };
            params.outBufferMode as usize
        }
        ZSTD_cParameter::ZSTD_c_rsyncable => {
            #[cfg(not(feature = "mt"))]
            {
                if value != 0 {
                    return ERROR(ErrorCode::ParameterUnsupported);
                }
                return 0;
            }
            #[cfg(feature = "mt")]
            {
                let rc = ZSTD_validateCParamSetterValue(param, value);
                if ERR_isError(rc) {
                    return rc;
                }
                params.rsyncable = value;
                params.rsyncable as usize
            }
        }
        ZSTD_cParameter::ZSTD_c_enableSeqProducerFallback => {
            let bounds = ZSTD_cParam_getBounds(ZSTD_cParameter::ZSTD_c_enableSeqProducerFallback);
            if value < bounds.lowerBound || value > bounds.upperBound {
                return ERROR(ErrorCode::ParameterOutOfBound);
            }
            params.enableMatchFinderFallback = value;
            params.enableMatchFinderFallback as usize
        }
        ZSTD_cParameter::ZSTD_c_blockSplitterLevel => {
            let bounds = ZSTD_cParam_getBounds(ZSTD_cParameter::ZSTD_c_blockSplitterLevel);
            if value < bounds.lowerBound || value > bounds.upperBound {
                return ERROR(ErrorCode::ParameterOutOfBound);
            }
            params.preBlockSplitter_level = value;
            params.preBlockSplitter_level as usize
        }
        ZSTD_cParameter::ZSTD_c_forceMaxWindow => {
            params.forceWindow = (value != 0) as i32;
            params.forceWindow as usize
        }
        ZSTD_cParameter::ZSTD_c_forceAttachDict => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            params.attachDictPref = match value {
                1 => ZSTD_dictAttachPref_e::ZSTD_dictForceAttach,
                2 => ZSTD_dictAttachPref_e::ZSTD_dictForceCopy,
                3 => ZSTD_dictAttachPref_e::ZSTD_dictForceLoad,
                _ => ZSTD_dictAttachPref_e::ZSTD_dictDefaultAttach,
            };
            params.attachDictPref as usize
        }
        ZSTD_cParameter::ZSTD_c_literalCompressionMode => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            params.literalCompressionMode = match value {
                1 => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_enable,
                2 => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_disable,
                _ => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_auto,
            };
            params.literalCompressionMode as usize
        }
        ZSTD_cParameter::ZSTD_c_srcSizeHint => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            params.srcSizeHint = value;
            params.srcSizeHint as usize
        }
        ZSTD_cParameter::ZSTD_c_enableDedicatedDictSearch => {
            params.enableDedicatedDictSearch = (value != 0) as i32;
            params.enableDedicatedDictSearch as usize
        }
        ZSTD_cParameter::ZSTD_c_blockDelimiters => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            params.blockDelimiters = match value {
                1 => ZSTD_SequenceFormat_e::ZSTD_sf_explicitBlockDelimiters,
                _ => ZSTD_SequenceFormat_e::ZSTD_sf_noBlockDelimiters,
            };
            params.blockDelimiters as usize
        }
        ZSTD_cParameter::ZSTD_c_validateSequences => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            params.validateSequences = value;
            params.validateSequences as usize
        }
        ZSTD_cParameter::ZSTD_c_splitAfterSequences => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            params.postBlockSplitter = match value {
                1 => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_enable,
                2 => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_disable,
                _ => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_auto,
            };
            params.postBlockSplitter as usize
        }
        ZSTD_cParameter::ZSTD_c_useRowMatchFinder => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            params.useRowMatchFinder = match value {
                1 => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_enable,
                2 => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_disable,
                _ => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_auto,
            };
            params.useRowMatchFinder as usize
        }
        ZSTD_cParameter::ZSTD_c_deterministicRefPrefix => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            params.deterministicRefPrefix = (value != 0) as i32;
            params.deterministicRefPrefix as usize
        }
        ZSTD_cParameter::ZSTD_c_prefetchCDictTables => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            params.prefetchCDictTables = match value {
                1 => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_enable,
                2 => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_disable,
                _ => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_auto,
            };
            params.prefetchCDictTables as usize
        }
        ZSTD_cParameter::ZSTD_c_maxBlockSize => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            params.maxBlockSize = value as usize;
            params.maxBlockSize
        }
        ZSTD_cParameter::ZSTD_c_repcodeResolution => {
            let rc = ZSTD_validateCParamSetterValue(param, value);
            if ERR_isError(rc) {
                return rc;
            }
            params.searchForExternalRepcodes = match value {
                1 => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_enable,
                2 => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_disable,
                _ => crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_auto,
            };
            params.searchForExternalRepcodes as usize
        }
        ZSTD_cParameter::ZSTD_c_checksumFlag => {
            params.fParams.checksumFlag = (value != 0) as u32;
            params.fParams.checksumFlag as usize
        }
        ZSTD_cParameter::ZSTD_c_contentSizeFlag => {
            params.fParams.contentSizeFlag = (value != 0) as u32;
            params.fParams.contentSizeFlag as usize
        }
        ZSTD_cParameter::ZSTD_c_dictIDFlag => {
            // Upstream stores noDictIDFlag, inverted — we mirror.
            params.fParams.noDictIDFlag = if value != 0 { 0 } else { 1 };
            if params.fParams.noDictIDFlag != 0 {
                0
            } else {
                1
            }
        }
        ZSTD_cParameter::ZSTD_c_format => {
            let bounds = ZSTD_cParam_getBounds(ZSTD_cParameter::ZSTD_c_format);
            if value < bounds.lowerBound || value > bounds.upperBound {
                return ERROR(ErrorCode::ParameterOutOfBound);
            }
            params.format = match value {
                v if v
                    == crate::decompress::zstd_decompress::ZSTD_format_e::ZSTD_f_zstd1_magicless
                        as i32 =>
                {
                    crate::decompress::zstd_decompress::ZSTD_format_e::ZSTD_f_zstd1_magicless
                }
                _ => crate::decompress::zstd_decompress::ZSTD_format_e::ZSTD_f_zstd1,
            };
            params.format as usize
        }
        ZSTD_cParameter::ZSTD_c_nbWorkers => {
            #[cfg(not(feature = "mt"))]
            {
                if value != 0 {
                    return ERROR(ErrorCode::ParameterUnsupported);
                }
                params.nbWorkers = 0;
                return 0;
            }
            #[cfg(feature = "mt")]
            {
                let bounds = ZSTD_cParam_getBounds(ZSTD_cParameter::ZSTD_c_nbWorkers);
                if value < bounds.lowerBound || value > bounds.upperBound {
                    return ERROR(ErrorCode::ParameterOutOfBound);
                }
                params.nbWorkers = value;
                params.nbWorkers as usize
            }
        }
        ZSTD_cParameter::ZSTD_c_jobSize => {
            #[cfg(not(feature = "mt"))]
            {
                if value != 0 {
                    return ERROR(ErrorCode::ParameterUnsupported);
                }
                return 0;
            }
            #[cfg(feature = "mt")]
            {
                let mut adjusted = value;
                if adjusted != 0
                    && adjusted < crate::compress::zstdmt_compress::ZSTDMT_JOBSIZE_MIN as i32
                {
                    adjusted = crate::compress::zstdmt_compress::ZSTDMT_JOBSIZE_MIN as i32;
                }
                let bounds = ZSTD_cParam_getBounds(ZSTD_cParameter::ZSTD_c_jobSize);
                if adjusted < bounds.lowerBound || adjusted > bounds.upperBound {
                    return ERROR(ErrorCode::ParameterOutOfBound);
                }
                params.jobSize = adjusted as usize;
                params.jobSize
            }
        }
        ZSTD_cParameter::ZSTD_c_overlapLog => {
            #[cfg(not(feature = "mt"))]
            {
                if value != 0 {
                    return ERROR(ErrorCode::ParameterUnsupported);
                }
                return 0;
            }
            #[cfg(feature = "mt")]
            {
                let bounds = ZSTD_cParam_getBounds(ZSTD_cParameter::ZSTD_c_overlapLog);
                if value < bounds.lowerBound || value > bounds.upperBound {
                    return ERROR(ErrorCode::ParameterOutOfBound);
                }
                params.overlapLog = value;
                params.overlapLog as usize
            }
        }
    }
}

/// Port of `ZSTD_CCtxParams_getParameter`. Symmetric getter.
pub fn ZSTD_CCtxParams_getParameter(
    params: &ZSTD_CCtx_params,
    param: ZSTD_cParameter,
    value: &mut i32,
) -> usize {
    *value = match param {
        ZSTD_cParameter::ZSTD_c_compressionLevel => params.compressionLevel,
        ZSTD_cParameter::ZSTD_c_windowLog => params.cParams.windowLog as i32,
        ZSTD_cParameter::ZSTD_c_hashLog => params.cParams.hashLog as i32,
        ZSTD_cParameter::ZSTD_c_chainLog => params.cParams.chainLog as i32,
        ZSTD_cParameter::ZSTD_c_searchLog => params.cParams.searchLog as i32,
        ZSTD_cParameter::ZSTD_c_minMatch => params.cParams.minMatch as i32,
        ZSTD_cParameter::ZSTD_c_targetLength => params.cParams.targetLength as i32,
        ZSTD_cParameter::ZSTD_c_strategy => params.cParams.strategy as i32,
        ZSTD_cParameter::ZSTD_c_targetCBlockSize => params.targetCBlockSize as i32,
        ZSTD_cParameter::ZSTD_c_enableLongDistanceMatching => params.ldmParams.enableLdm as i32,
        ZSTD_cParameter::ZSTD_c_ldmHashLog => params.ldmParams.hashLog as i32,
        ZSTD_cParameter::ZSTD_c_ldmMinMatch => params.ldmParams.minMatchLength as i32,
        ZSTD_cParameter::ZSTD_c_ldmBucketSizeLog => params.ldmParams.bucketSizeLog as i32,
        ZSTD_cParameter::ZSTD_c_ldmHashRateLog => params.ldmParams.hashRateLog as i32,
        ZSTD_cParameter::ZSTD_c_stableInBuffer => params.inBufferMode as i32,
        ZSTD_cParameter::ZSTD_c_stableOutBuffer => params.outBufferMode as i32,
        ZSTD_cParameter::ZSTD_c_rsyncable => {
            #[cfg(not(feature = "mt"))]
            {
                return ERROR(ErrorCode::ParameterUnsupported);
            }
            #[cfg(feature = "mt")]
            {
                params.rsyncable
            }
        }
        ZSTD_cParameter::ZSTD_c_forceMaxWindow => params.forceWindow,
        ZSTD_cParameter::ZSTD_c_forceAttachDict => params.attachDictPref as i32,
        ZSTD_cParameter::ZSTD_c_literalCompressionMode => params.literalCompressionMode as i32,
        ZSTD_cParameter::ZSTD_c_srcSizeHint => params.srcSizeHint,
        ZSTD_cParameter::ZSTD_c_enableDedicatedDictSearch => params.enableDedicatedDictSearch,
        ZSTD_cParameter::ZSTD_c_blockDelimiters => params.blockDelimiters as i32,
        ZSTD_cParameter::ZSTD_c_validateSequences => params.validateSequences,
        ZSTD_cParameter::ZSTD_c_splitAfterSequences => params.postBlockSplitter as i32,
        ZSTD_cParameter::ZSTD_c_useRowMatchFinder => params.useRowMatchFinder as i32,
        ZSTD_cParameter::ZSTD_c_deterministicRefPrefix => params.deterministicRefPrefix,
        ZSTD_cParameter::ZSTD_c_prefetchCDictTables => params.prefetchCDictTables as i32,
        ZSTD_cParameter::ZSTD_c_enableSeqProducerFallback => params.enableMatchFinderFallback,
        ZSTD_cParameter::ZSTD_c_maxBlockSize => params.maxBlockSize as i32,
        ZSTD_cParameter::ZSTD_c_repcodeResolution => params.searchForExternalRepcodes as i32,
        ZSTD_cParameter::ZSTD_c_blockSplitterLevel => params.preBlockSplitter_level,
        ZSTD_cParameter::ZSTD_c_checksumFlag => params.fParams.checksumFlag as i32,
        ZSTD_cParameter::ZSTD_c_contentSizeFlag => params.fParams.contentSizeFlag as i32,
        ZSTD_cParameter::ZSTD_c_format => params.format as i32,
        ZSTD_cParameter::ZSTD_c_nbWorkers => params.nbWorkers,
        ZSTD_cParameter::ZSTD_c_jobSize => {
            #[cfg(not(feature = "mt"))]
            {
                return ERROR(ErrorCode::ParameterUnsupported);
            }
            #[cfg(feature = "mt")]
            {
                params.jobSize as i32
            }
        }
        ZSTD_cParameter::ZSTD_c_overlapLog => {
            #[cfg(not(feature = "mt"))]
            {
                return ERROR(ErrorCode::ParameterUnsupported);
            }
            #[cfg(feature = "mt")]
            {
                params.overlapLog
            }
        }
        ZSTD_cParameter::ZSTD_c_dictIDFlag => {
            if params.fParams.noDictIDFlag != 0 {
                0
            } else {
                1
            }
        }
    };
    0
}

/// Port of `ZSTD_parameters`. Bundles the active `cParams` and
/// `fParams` — returned by the public `ZSTD_getParams` entry point so
/// callers can seed a CCtx with one struct.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZSTD_parameters {
    pub cParams: crate::compress::match_state::ZSTD_compressionParameters,
    pub fParams: ZSTD_FrameParameters,
}

/// Port of `ZSTD_getParams_internal` (`zstd_compress.c:8324`). Picks
/// baseline cParams via `ZSTD_getCParams_internal` (mode-aware) and
/// pairs them with upstream's default fParams.
pub fn ZSTD_getParams_internal(
    compressionLevel: i32,
    srcSizeHint: u64,
    dictSize: usize,
    mode: ZSTD_CParamMode_e,
) -> ZSTD_parameters {
    ZSTD_parameters {
        cParams: ZSTD_getCParams_internal(compressionLevel, srcSizeHint, dictSize, mode),
        fParams: ZSTD_FrameParameters {
            contentSizeFlag: 1,
            checksumFlag: 0,
            noDictIDFlag: 0,
        },
    }
}

/// Port of `ZSTD_getParams`. Convenience wrapper:
/// `srcSizeHint == 0` → UNKNOWN, mode = `ZSTD_cpm_unknown`.
pub fn ZSTD_getParams(compressionLevel: i32, srcSizeHint: u64, dictSize: usize) -> ZSTD_parameters {
    use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;
    let effective = if srcSizeHint == 0 {
        ZSTD_CONTENTSIZE_UNKNOWN
    } else {
        srcSizeHint
    };
    ZSTD_getParams_internal(
        compressionLevel,
        effective,
        dictSize,
        ZSTD_CParamMode_e::ZSTD_cpm_unknown,
    )
}

/// Port of `ZSTD_writeFrameHeader`. Emits the 4-byte magic number
/// followed by the frame header descriptor byte, optional window
/// descriptor, optional dictID, and optional frame content size — per
/// the Zstandard format spec.
///
/// Rust signature note: upstream takes `ZSTD_CCtx_params*`; we accept
/// the raw knobs (windowLog, frame params, dictID) since the larger
/// params struct hasn't landed yet.
pub fn ZSTD_writeFrameHeader(
    dst: &mut [u8],
    fParams: &ZSTD_FrameParameters,
    windowLog: u32,
    pledgedSrcSize: u64,
    dictID: u32,
) -> usize {
    ZSTD_writeFrameHeader_advanced(
        dst,
        fParams,
        windowLog,
        pledgedSrcSize,
        dictID,
        crate::decompress::zstd_decompress::ZSTD_format_e::ZSTD_f_zstd1,
    )
}

/// Port of upstream's format-aware `ZSTD_writeFrameHeader`
/// (`zstd_compress.c:4719`). When `format == ZSTD_f_zstd1_magicless` the
/// 4-byte magic prefix is elided — the rest of the header layout
/// (FHD, windowLog, dictID, FCS) is identical. Callers that don't
/// need magicless frames should use the plain `ZSTD_writeFrameHeader`
/// wrapper. Decoder side honors the symmetric `dctx.format` via
/// `ZSTD_DCtx_setFormat`.
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_writeFrameHeader_advanced(
    dst: &mut [u8],
    fParams: &ZSTD_FrameParameters,
    windowLog: u32,
    pledgedSrcSize: u64,
    dictID: u32,
    format: crate::decompress::zstd_decompress::ZSTD_format_e,
) -> usize {
    use crate::decompress::zstd_decompress::{ZSTD_format_e, ZSTD_MAGICNUMBER};
    if dst.len() < ZSTD_FRAMEHEADERSIZE_MAX {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    let dictIDSizeCodeLength = ((dictID > 0) as u32)
        .wrapping_add((dictID >= 256) as u32)
        .wrapping_add((dictID >= 65536) as u32);
    let dictIDSizeCode = if fParams.noDictIDFlag != 0 {
        0
    } else {
        dictIDSizeCodeLength
    };
    let checksumFlag = (fParams.checksumFlag > 0) as u32;
    let windowSize: u32 = 1u32 << windowLog;
    let singleSegment =
        (fParams.contentSizeFlag != 0 && windowSize as u64 >= pledgedSrcSize) as u32;
    let windowLogByte: u8 =
        ((windowLog - crate::decompress::zstd_decompress::ZSTD_WINDOWLOG_ABSOLUTEMIN) << 3) as u8;
    let fcsCode: u32 = if fParams.contentSizeFlag != 0 {
        ((pledgedSrcSize >= 256) as u32)
            .wrapping_add((pledgedSrcSize >= 65536 + 256) as u32)
            .wrapping_add((pledgedSrcSize >= 0xFFFFFFFFu64) as u32)
    } else {
        0
    };
    let frameHeaderDescriptorByte: u8 =
        (dictIDSizeCode + (checksumFlag << 2) + (singleSegment << 5) + (fcsCode << 6)) as u8;

    let mut pos = 0usize;
    if format == ZSTD_format_e::ZSTD_f_zstd1 {
        MEM_writeLE32(dst, ZSTD_MAGICNUMBER);
        pos = 4;
    }
    dst[pos] = frameHeaderDescriptorByte;
    pos += 1;
    if singleSegment == 0 {
        dst[pos] = windowLogByte;
        pos += 1;
    }
    match dictIDSizeCode {
        0 => {}
        1 => {
            dst[pos] = dictID as u8;
            pos += 1;
        }
        2 => {
            MEM_writeLE16(&mut dst[pos..], dictID as u16);
            pos += 2;
        }
        _ => {
            MEM_writeLE32(&mut dst[pos..], dictID);
            pos += 4;
        }
    }
    match fcsCode {
        0 => {
            if singleSegment != 0 {
                dst[pos] = pledgedSrcSize as u8;
                pos += 1;
            }
        }
        1 => {
            MEM_writeLE16(&mut dst[pos..], (pledgedSrcSize - 256) as u16);
            pos += 2;
        }
        2 => {
            MEM_writeLE32(&mut dst[pos..], pledgedSrcSize as u32);
            pos += 4;
        }
        _ => {
            MEM_writeLE64(&mut dst[pos..], pledgedSrcSize);
            pos += 8;
        }
    }
    pos
}

/// Port of `ZSTD_writeLastEmptyBlock`. 3-byte header with lastBlock=1,
/// blockType=bt_raw, size=0 — used when a frame needs an explicit
/// end-of-frame marker (e.g. after the final compressed block was
/// emitted without `lastBlock=1`, or for an empty source).
pub fn ZSTD_writeLastEmptyBlock(dst: &mut [u8]) -> usize {
    use crate::decompress::zstd_decompress_block::{blockType_e, ZSTD_blockHeaderSize};
    if dst.len() < ZSTD_blockHeaderSize {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    let cBlockHeader24 = 1u32.wrapping_add((blockType_e::bt_raw as u32) << 1);
    MEM_writeLE24(dst, cBlockHeader24);
    ZSTD_blockHeaderSize
}

/// Single-frame top-level compressor over the fast strategy. Emits:
///   magic + header → sequence of block bodies (one per
///   ZSTD_BLOCKSIZE_MAX slice) → optional XXH64 trailer.
///
/// Stateless helper: allocates its own match state + seq store +
/// entropy state each call (no CCtx re-use). Unlike the CCtx entry
/// points, this stays on the dedicated one-shot fast-frame helper
/// path instead of the normal `compressBegin_internal` session flow.
/// The resulting frame is bitwise-identical to what upstream
/// produces for the same input when using strategy=fast with matched
/// `cParams`.
#[inline]
pub fn ZSTD_compressFrame_fast(
    dst: &mut [u8],
    src: &[u8],
    cParams: crate::compress::match_state::ZSTD_compressionParameters,
    fParams: ZSTD_FrameParameters,
) -> usize {
    ZSTD_compressFrame_fast_advanced(
        dst,
        src,
        cParams,
        fParams,
        crate::decompress::zstd_decompress::ZSTD_format_e::ZSTD_f_zstd1,
    )
}

/// Format-aware variant of `ZSTD_compressFrame_fast`. When
/// `format == ZSTD_f_zstd1_magicless` the emitted frame starts with
/// the 1-byte Frame Header Descriptor instead of the 4-byte magic +
/// FHD. Everything else (block bodies, optional XXH64 trailer) is
/// identical. Pairs with the decoder's `ZSTD_DCtx_setFormat`.
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_compressFrame_fast_advanced(
    dst: &mut [u8],
    src: &[u8],
    cParams: crate::compress::match_state::ZSTD_compressionParameters,
    fParams: ZSTD_FrameParameters,
    format: crate::decompress::zstd_decompress::ZSTD_format_e,
) -> usize {
    use crate::common::xxhash::{XXH64_digest, XXH64_reset, XXH64_state_t, XXH64_update};
    use crate::compress::seq_store::{SeqStore_t, ZSTD_REP_NUM};
    use crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX;

    let mut op = 0usize;
    // 1. Frame header.
    let hdrSize = ZSTD_writeFrameHeader_advanced(
        &mut dst[op..],
        &fParams,
        cParams.windowLog,
        src.len() as u64,
        0,
        format,
    );
    if ERR_isError(hdrSize) {
        return hdrSize;
    }
    op += hdrSize;

    // 2. Per-frame state. We now keep a single match state across
    //    blocks for ALL supported strategies (fast, dfast, greedy,
    //    lazy, lazy2) so hash-table + chain-table entries seeded by
    //    earlier blocks are visible as history for later blocks.
    let mut seqStore = SeqStore_t::with_capacity(ZSTD_BLOCKSIZE_MAX / 3, ZSTD_BLOCKSIZE_MAX);
    let mut carry_ms = crate::compress::match_state::ZSTD_MatchState_t::new(cParams);
    let useRowMatchFinder = crate::compress::match_state::ZSTD_resolveRowMatchFinderMode(
        crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_auto,
        &cParams,
    );
    let mut rep: [u32; ZSTD_REP_NUM] = [1, 4, 8];
    let mut prevEntropy = ZSTD_entropyCTables_t::default();
    let mut nextEntropy = ZSTD_entropyCTables_t::default();
    let mut entropyScratch = ZSTD_entropyScratch::default();

    // 3. XXH64 accumulator.
    let mut xxh = XXH64_state_t::default();
    XXH64_reset(&mut xxh, 0);

    // 4. Block loop.
    let mut ip = 0usize;
    let mut emitted_any_block = false;
    while ip < src.len() {
        let remaining = src.len() - ip;
        let blockSize = remaining.min(ZSTD_BLOCKSIZE_MAX);
        let is_last = ip + blockSize == src.len();
        if fParams.checksumFlag != 0 {
            XXH64_update(&mut xxh, &src[ip..ip + blockSize]);
        }
        seqStore.reset();
        // Pass the cumulative src prefix (0..=ip+blockSize) so hash
        // entries from earlier blocks are valid back-references. The
        // per-strategy with-history match finder scans [ip..ip+blockSize).
        let bodySize = ZSTD_compressBlock_any_framed_with_history_wksp(
            &mut dst[op..],
            &src[..ip + blockSize],
            ip,
            &mut carry_ms,
            &mut seqStore,
            &mut rep,
            &prevEntropy,
            &mut nextEntropy,
            cParams.strategy,
            useRowMatchFinder,
            0,
            0,
            if is_last { 1 } else { 0 },
            ip == 0,
            &mut entropyScratch,
        );
        if ERR_isError(bodySize) {
            return bodySize;
        }
        op += bodySize;
        emitted_any_block = true;
        // Standalone one-shot helper equivalent of upstream's
        // blockState confirm: entropy tables advance via
        // prev/next swap, while repcodes already live in `rep`.
        std::mem::swap(&mut prevEntropy, &mut nextEntropy);
        // offcode_repeatMode "valid" → "check" carryover.
        if prevEntropy.fse.offcode_repeatMode
            == crate::compress::zstd_compress_sequences::FSE_repeat::FSE_repeat_valid
        {
            prevEntropy.fse.offcode_repeatMode =
                crate::compress::zstd_compress_sequences::FSE_repeat::FSE_repeat_check;
        }
        ip += blockSize;
    }

    // 5. Empty-source edge case: frame must still contain at least a
    //    last-block marker.
    if !emitted_any_block {
        let n = ZSTD_writeLastEmptyBlock(&mut dst[op..]);
        if ERR_isError(n) {
            return n;
        }
        op += n;
    }

    // 6. Optional XXH64 checksum trailer (4 low bytes).
    if fParams.checksumFlag != 0 {
        if dst.len() - op < 4 {
            return ERROR(ErrorCode::DstSizeTooSmall);
        }
        let digest = XXH64_digest(&xxh);
        MEM_writeLE32(&mut dst[op..], digest as u32);
        op += 4;
    }

    op
}

/// Variant of `ZSTD_compressFrame_fast` that treats an initial
/// `prefix` as history for back-references but does NOT include the
/// prefix bytes in the frame's content size or in the output. Used by
/// `ZSTD_compress_usingDict` for raw-content dictionaries.
///
/// `seed_entropy` / `seed_rep` optionally seed the compressor's
/// per-frame entropy tables and repcode history — used when the caller
/// has already parsed a magic-prefixed zstd-format dict via
/// `ZSTD_compress_insertDictionary`. Upstream threads these through
/// `cctx->blockState.prevCBlock->entropy` and `cctx->blockState.prevCBlock->rep`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn ZSTD_compressFrame_fast_with_prefix(
    dst: &mut [u8],
    src: &[u8],
    prefix: &[u8],
    cParams: crate::compress::match_state::ZSTD_compressionParameters,
    fParams: ZSTD_FrameParameters,
    seed_entropy: Option<&ZSTD_entropyCTables_t>,
    seed_rep: Option<[u32; crate::compress::seq_store::ZSTD_REP_NUM]>,
) -> usize {
    ZSTD_compressFrame_fast_with_prefix_advanced(
        dst,
        src,
        prefix,
        cParams,
        fParams,
        seed_entropy,
        seed_rep,
        crate::decompress::zstd_decompress::ZSTD_format_e::ZSTD_f_zstd1,
    )
}

/// Format-aware variant of `_with_prefix`. Emits a magicless frame
/// when `format == ZSTD_f_zstd1_magicless`. The prefix's back-ref
/// history mechanics are unchanged — only the outer frame header's
/// magic-prefix bytes are affected.
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_compressFrame_fast_with_prefix_advanced(
    dst: &mut [u8],
    src: &[u8],
    prefix: &[u8],
    cParams: crate::compress::match_state::ZSTD_compressionParameters,
    fParams: ZSTD_FrameParameters,
    seed_entropy: Option<&ZSTD_entropyCTables_t>,
    seed_rep: Option<[u32; crate::compress::seq_store::ZSTD_REP_NUM]>,
    format: crate::decompress::zstd_decompress::ZSTD_format_e,
) -> usize {
    use crate::common::xxhash::{XXH64_digest, XXH64_reset, XXH64_state_t, XXH64_update};
    use crate::compress::seq_store::ZSTD_REP_NUM;
    use crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX;

    let mut op = 0usize;
    // If the prefix is a magic-tagged dict, propagate its dictID into
    // the frame header so `ZSTD_getDictID_fromFrame` round-trips.
    // Upstream does the same through `ZSTD_getDictID_fromDict` during
    // `ZSTD_CCtx_loadDictionary_advanced`; with `fParams.noDictIDFlag`
    // set the writer suppresses the field regardless.
    let dictID = crate::decompress::zstd_ddict::ZSTD_getDictID_fromDict(prefix);
    let hdrSize = ZSTD_writeFrameHeader_advanced(
        &mut dst[op..],
        &fParams,
        cParams.windowLog,
        src.len() as u64,
        dictID,
        format,
    );
    if ERR_isError(hdrSize) {
        return hdrSize;
    }
    op += hdrSize;

    // Build the combined scan buffer `prefix || src`. The match state
    // scans positions [prefix.len()..] with history in [0..prefix.len()).
    let mut combined = Vec::with_capacity(prefix.len() + src.len());
    combined.extend_from_slice(prefix);
    combined.extend_from_slice(src);
    let prefix_len = prefix.len();

    let mut seqStore = SeqStore_t::with_capacity(ZSTD_BLOCKSIZE_MAX / 3, ZSTD_BLOCKSIZE_MAX);
    let mut carry_ms = crate::compress::match_state::ZSTD_MatchState_t::new(cParams);
    let useRowMatchFinder = crate::compress::match_state::ZSTD_resolveRowMatchFinderMode(
        crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_auto,
        &cParams,
    );
    let mut rep: [u32; ZSTD_REP_NUM] = seed_rep.unwrap_or([1, 4, 8]);
    let mut prevEntropy = match seed_entropy {
        Some(s) => s.clone(),
        None => ZSTD_entropyCTables_t::default(),
    };
    let mut nextEntropy = ZSTD_entropyCTables_t::default();

    // Pre-seed the match state's hash tables from the prefix bytes by
    // running the match finder over `combined[..prefix_len]` with a
    // throwaway seq store. The resulting sequences are discarded, but
    // the hash table entries remain and become valid back-references
    // for subsequent `src` blocks.
    if prefix_len > 8 {
        let mut throwaway = SeqStore_t::with_capacity(
            prefix_len.div_ceil(4).min(ZSTD_BLOCKSIZE_MAX / 3),
            prefix_len.min(ZSTD_BLOCKSIZE_MAX),
        );
        let mut throwaway_rep: [u32; ZSTD_REP_NUM] = [1, 4, 8];
        // Scan in block-sized chunks so we don't exceed throwaway cap.
        let mut p = 0usize;
        while p < prefix_len {
            let end = (p + ZSTD_BLOCKSIZE_MAX).min(prefix_len);
            let minMatch = carry_ms.cParams.minMatch;
            throwaway.reset();
            let _ = match cParams.strategy {
                crate::compress::zstd_compress_sequences::ZSTD_greedy => {
                    let searchMethod = if crate::compress::match_state::ZSTD_rowMatchFinderUsed(
                        carry_ms.cParams.strategy,
                        useRowMatchFinder,
                    ) {
                        crate::compress::zstd_lazy::searchMethod_e::search_rowHash
                    } else {
                        crate::compress::zstd_lazy::searchMethod_e::search_hashChain
                    };
                    crate::compress::zstd_lazy::ZSTD_compressBlock_lazy_generic_with_istart(
                        &mut carry_ms,
                        &mut throwaway,
                        &mut throwaway_rep,
                        &combined[..end],
                        p,
                        searchMethod,
                        0,
                        crate::compress::match_state::ZSTD_dictMode_e::ZSTD_noDict,
                    )
                }
                crate::compress::zstd_compress_sequences::ZSTD_lazy => {
                    let searchMethod = if crate::compress::match_state::ZSTD_rowMatchFinderUsed(
                        carry_ms.cParams.strategy,
                        useRowMatchFinder,
                    ) {
                        crate::compress::zstd_lazy::searchMethod_e::search_rowHash
                    } else {
                        crate::compress::zstd_lazy::searchMethod_e::search_hashChain
                    };
                    crate::compress::zstd_lazy::ZSTD_compressBlock_lazy_generic_with_istart(
                        &mut carry_ms,
                        &mut throwaway,
                        &mut throwaway_rep,
                        &combined[..end],
                        p,
                        searchMethod,
                        1,
                        crate::compress::match_state::ZSTD_dictMode_e::ZSTD_noDict,
                    )
                }
                crate::compress::zstd_compress_sequences::ZSTD_lazy2
                | crate::compress::zstd_compress_sequences::ZSTD_btlazy2 => {
                    let searchMethod = if crate::compress::match_state::ZSTD_rowMatchFinderUsed(
                        carry_ms.cParams.strategy,
                        useRowMatchFinder,
                    ) {
                        crate::compress::zstd_lazy::searchMethod_e::search_rowHash
                    } else {
                        crate::compress::zstd_lazy::searchMethod_e::search_hashChain
                    };
                    crate::compress::zstd_lazy::ZSTD_compressBlock_lazy_generic_with_istart(
                        &mut carry_ms,
                        &mut throwaway,
                        &mut throwaway_rep,
                        &combined[..end],
                        p,
                        searchMethod,
                        2,
                        crate::compress::match_state::ZSTD_dictMode_e::ZSTD_noDict,
                    )
                }
                crate::compress::zstd_compress_sequences::ZSTD_dfast => {
                    crate::compress::zstd_double_fast::ZSTD_compressBlock_doubleFast_noDict_generic(
                        &mut carry_ms,
                        &mut throwaway,
                        &mut throwaway_rep,
                        &combined[..end],
                        p,
                        minMatch,
                    )
                }
                _ => crate::compress::zstd_fast::ZSTD_compressBlock_fast_noDict_generic(
                    &mut carry_ms,
                    &mut throwaway,
                    &mut throwaway_rep,
                    &combined[..end],
                    p,
                    minMatch,
                ),
            };
            p = end;
        }
        // The rep array is NOT carried over — upstream keeps the
        // default [1,4,8] for dict-compression start (rep offsets
        // would need dict-relative validity checking).
    }

    let mut xxh = XXH64_state_t::default();
    XXH64_reset(&mut xxh, 0);
    let mut entropyScratch = ZSTD_entropyScratch::default();

    let mut ip = 0usize; // offset within `src`, not `combined`
    let mut emitted_any_block = false;
    while ip < src.len() {
        let remaining = src.len() - ip;
        let blockSize = remaining.min(ZSTD_BLOCKSIZE_MAX);
        let is_last = ip + blockSize == src.len();
        if fParams.checksumFlag != 0 {
            XXH64_update(&mut xxh, &src[ip..ip + blockSize]);
        }
        seqStore.reset();
        // Scan buffer = combined[..prefix_len + ip + blockSize], with
        // istart = prefix_len + ip (covers the prefix as history).
        let scan_end = prefix_len + ip + blockSize;
        let scan_istart = prefix_len + ip;
        let bodySize = ZSTD_compressBlock_any_framed_with_history_wksp(
            &mut dst[op..],
            &combined[..scan_end],
            scan_istart,
            &mut carry_ms,
            &mut seqStore,
            &mut rep,
            &prevEntropy,
            &mut nextEntropy,
            cParams.strategy,
            useRowMatchFinder,
            0,
            0,
            if is_last { 1 } else { 0 },
            ip == 0,
            &mut entropyScratch,
        );
        if ERR_isError(bodySize) {
            return bodySize;
        }
        op += bodySize;
        emitted_any_block = true;
        std::mem::swap(&mut prevEntropy, &mut nextEntropy);
        if prevEntropy.fse.offcode_repeatMode
            == crate::compress::zstd_compress_sequences::FSE_repeat::FSE_repeat_valid
        {
            prevEntropy.fse.offcode_repeatMode =
                crate::compress::zstd_compress_sequences::FSE_repeat::FSE_repeat_check;
        }
        ip += blockSize;
    }
    if !emitted_any_block {
        let n = ZSTD_writeLastEmptyBlock(&mut dst[op..]);
        if ERR_isError(n) {
            return n;
        }
        op += n;
    }
    if fParams.checksumFlag != 0 {
        if dst.len() - op < 4 {
            return ERROR(ErrorCode::DstSizeTooSmall);
        }
        let digest = XXH64_digest(&xxh);
        MEM_writeLE32(&mut dst[op..], digest as u32);
        op += 4;
    }
    op
}

/// Port of `ZSTD_initCStream`. Resets the CCtx's streaming state and
/// captures the compression level for subsequent `compressStream` /
/// `endStream` calls. Returns 0 on success.
pub fn ZSTD_initCStream(zcs: &mut ZSTD_CCtx, compressionLevel: i32) -> usize {
    // Upstream (zstd_compress.c:6101) routes through
    // `ZSTD_CCtx_setParameter(c_compressionLevel, value)`, picking up
    // that setter's clamp + `value == 0 → CLEVEL_DEFAULT` behavior.
    // Previously we raw-stored the level — left a C-compat caller
    // that called `initCStream(cctx, 0)` with `getParameter` readback
    // returning 0 (upstream returns 3).
    let rc = ZSTD_CCtx_reset(zcs, ZSTD_ResetDirective::ZSTD_reset_session_only);
    if ERR_isError(rc) {
        return rc;
    }
    zcs.stream_in_buffer.clear();
    zcs.stream_params_snapshot = None;
    zcs.stream_out_buffer.clear();
    zcs.stream_out_drained = 0;
    zcs.stream_stage = ZSTD_cStreamStage::zcss_init;
    zcs.stream_in_to_compress = 0;
    zcs.stream_in_target = 0;
    zcs.stream_frame_ended = false;
    zcs.stream_window_base = crate::compress::match_state::ZSTD_WINDOW_START_INDEX;
    zcs.stream_closed = false;
    zcs.stream_frame_completed = false;
    ZSTD_clearAllDicts(zcs);
    let rc = ZSTD_CCtx_setParameter(
        zcs,
        ZSTD_cParameter::ZSTD_c_compressionLevel,
        compressionLevel,
    );
    if ERR_isError(rc) {
        return rc;
    }
    0
}

/// Port of `ZSTD_EndDirective`. Tells `ZSTD_compressStream2` whether
/// the current call should flush / end the frame / just continue
/// buffering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ZSTD_EndDirective {
    ZSTD_e_continue = 0,
    ZSTD_e_flush = 1,
    ZSTD_e_end = 2,
}

/// Port of `ZSTD_compressStream2`. Unified streaming compression
/// entry point that dispatches based on `end_op`:
///   - `ZSTD_e_continue`: buffer input, drain any pending output.
///   - `ZSTD_e_flush`: as continue + flush any buffered output.
///   - `ZSTD_e_end`: as flush + finalize frame.
///
/// Returns the recommended next input size, or 0 when the current
/// frame is fully delivered (after `ZSTD_e_end`).
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_compressStream2(
    cctx: &mut ZSTD_CCtx,
    output: &mut [u8],
    output_pos: &mut usize,
    input: &[u8],
    input_pos: &mut usize,
    end_op: ZSTD_EndDirective,
) -> usize {
    if *output_pos > output.len() {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    if *input_pos > input.len() {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    let output_len = output.len();
    let input_len = input.len();
    let mut output_buffer = ZSTD_outBuffer {
        dst: Some(output),
        size: output_len,
        pos: *output_pos,
    };
    let mut input_buffer = ZSTD_inBuffer {
        src: Some(input),
        size: input_len,
        pos: *input_pos,
    };
    if cctx.stream_level.is_none() {
        cctx.stream_level = Some(ZSTD_CLEVEL_DEFAULT);
    }
    let result = ZSTD_compressStream_generic(cctx, &mut output_buffer, &mut input_buffer, end_op);
    *output_pos = output_buffer.pos;
    *input_pos = input_buffer.pos;
    result
}

/// Port of `ZSTD_compressStream_generic` (`zstd_compress.c:6127`).
/// In the buffered Rust stream model this is the main directive
/// dispatcher once the public `(slice, pos)` API has been unpacked
/// into `ZSTD_inBuffer` / `ZSTD_outBuffer`.
pub fn ZSTD_compressStream_generic(
    zcs: &mut ZSTD_CCtx,
    output: &mut ZSTD_outBuffer<'_>,
    input: &mut ZSTD_inBuffer<'_>,
    flushMode: ZSTD_EndDirective,
) -> usize {
    use crate::compress::zstd_compress::ZSTD_EndDirective::{
        ZSTD_e_continue, ZSTD_e_end, ZSTD_e_flush,
    };
    if output.pos > output.size {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    if input.pos > input.size {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    let stable_rc = ZSTD_checkBufferStability(zcs, output, input, flushMode);
    if ERR_isError(stable_rc) {
        return stable_rc;
    }
    let dst = match output.dst.as_deref_mut() {
        Some(dst) => dst,
        None => &mut [],
    };
    let src = match input.src {
        Some(src) => src,
        None => &[],
    };
    let src_end = input.size.min(src.len());
    let dst_end = output.size;
    if input.pos > src_end {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    let remaining_src = src_end - input.pos;
    if flushMode == ZSTD_e_end
        && remaining_src == 0
        && zcs.stream_frame_completed
        && zcs.stream_in_buffer.is_empty()
        && zcs.stream_out_buffer.is_empty()
    {
        ZSTD_setBufferExpectations(zcs, output, input);
        return 0;
    }
    let first_call_end_with_known_size = flushMode == ZSTD_e_end
        && zcs.stage == ZSTD_compressionStage_e::ZSTDcs_created
        && zcs.stream_in_buffer.is_empty()
        && zcs.stream_out_buffer.is_empty()
        && !zcs.stream_closed;
    if first_call_end_with_known_size {
        let pledged = remaining_src as u64;
        zcs.pledged_src_size = Some(pledged);
        zcs.pledgedSrcSizePlusOne = pledged.wrapping_add(1);
    }
    if flushMode == ZSTD_e_end
        && zcs.stream_in_buffer.is_empty()
        && zcs.stream_dict.is_empty()
        && zcs.stream_cdict.is_none()
        && !zcs.prefix_is_single_use
        && (dst_end.saturating_sub(output.pos) >= ZSTD_compressBound(remaining_src)
            || zcs.requestedParams.outBufferMode == ZSTD_bufferMode_e::ZSTD_bm_stable)
    {
        let rc = zstd_init_stream_from_snapshot(zcs, flushMode, remaining_src);
        if ERR_isError(rc) {
            return rc;
        }
        let c_size =
            ZSTD_compressEnd_public(zcs, &mut dst[output.pos..dst_end], &src[input.pos..src_end]);
        if ERR_isError(c_size) {
            return c_size;
        }
        input.pos = src_end;
        output.pos += c_size;
        cctx_mark_stream_frame_completed(zcs);
        ZSTD_setBufferExpectations(zcs, output, input);
        return 0;
    }
    if zstd_stream_can_use_windowed(zcs) {
        let result = zstd_compressStream_windowed(
            zcs,
            &mut dst[..dst_end],
            &mut output.pos,
            &src[..src_end],
            &mut input.pos,
            flushMode,
            false,
        );
        if ERR_isError(result) {
            return result;
        }
        ZSTD_setBufferExpectations(zcs, output, input);
        return result;
    }
    let cont_rc = zstd_compressStream_continue_buffered(
        zcs,
        &mut dst[..dst_end],
        &mut output.pos,
        &src[..src_end],
        &mut input.pos,
    );
    if ERR_isError(cont_rc) {
        return cont_rc;
    }
    let result = match flushMode {
        ZSTD_e_continue => cont_rc,
        ZSTD_e_flush => zstd_flushStream_buffered(zcs, &mut dst[..dst_end], &mut output.pos),
        ZSTD_e_end => zstd_endStream_buffered(zcs, &mut dst[..dst_end], &mut output.pos),
    };
    if ERR_isError(result) {
        return result;
    }
    ZSTD_setBufferExpectations(zcs, output, input);
    result
}

/// Port of `ZSTD_compressStream2_simpleArgs`. Thin wrapper that
/// unpacks a `(buf, size, pos)` tuple into the slice form we use.
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_compressStream2_simpleArgs(
    cctx: &mut ZSTD_CCtx,
    dst: &mut [u8],
    dst_pos: &mut usize,
    src: &[u8],
    src_pos: &mut usize,
    end_op: ZSTD_EndDirective,
) -> usize {
    ZSTD_compressStream2(cctx, dst, dst_pos, src, src_pos, end_op)
}

pub type ZSTD_allocFunction = fn(opaque: usize, size: usize) -> *mut core::ffi::c_void;
pub type ZSTD_freeFunction = fn(opaque: usize, address: *mut core::ffi::c_void);

/// Port of `ZSTD_customMem`. Upstream passes `alloc`/`free` function
/// pointers plus an opaque pointer. The Rust port preserves that
/// descriptor and threads it through advanced creator surfaces, but
/// top-level object allocation for `Box<T>`-returning APIs still uses
/// Rust's global allocator.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZSTD_customMem {
    pub customAlloc: Option<ZSTD_allocFunction>,
    pub customFree: Option<ZSTD_freeFunction>,
    pub opaque: usize,
}

impl PartialEq for ZSTD_customMem {
    fn eq(&self, other: &Self) -> bool {
        self.opaque == other.opaque
            && match (self.customAlloc, other.customAlloc) {
                (Some(a), Some(b)) => core::ptr::fn_addr_eq(a, b),
                (None, None) => true,
                _ => false,
            }
            && match (self.customFree, other.customFree) {
                (Some(a), Some(b)) => core::ptr::fn_addr_eq(a, b),
                (None, None) => true,
                _ => false,
            }
    }
}

impl Eq for ZSTD_customMem {}

/// Port of `ZSTD_customMem_isNull` — true iff the caller passed the
/// all-zeros sentinel (no custom alloc/free, zero opaque) signalling
/// "use the default allocator".
#[inline]
pub fn ZSTD_customMem_isNull(customMem: ZSTD_customMem) -> bool {
    customMem.customAlloc.is_none() && customMem.customFree.is_none() && customMem.opaque == 0
}

/// Rust-only helper: enforces upstream's invariant that `customAlloc` and
/// `customFree` are either both set or both null — rejects half-configured
/// allocator bundles.
#[inline]
pub(crate) fn ZSTD_customMem_validate(customMem: ZSTD_customMem) -> bool {
    matches!(
        (
            customMem.customAlloc.is_some(),
            customMem.customFree.is_some()
        ),
        (false, false) | (true, true)
    )
}

pub(crate) unsafe fn ZSTD_customAllocBox<T>(value: T, customMem: ZSTD_customMem) -> Option<Box<T>> {
    use core::ptr;

    if ZSTD_customMem_isNull(customMem) {
        return Some(Box::new(value));
    }

    let raw =
        (customMem.customAlloc?)(customMem.opaque, core::mem::size_of::<T>().max(1)) as *mut T;
    if raw.is_null() {
        return None;
    }
    unsafe {
        ptr::write(raw, value);
        Some(Box::from_raw(raw))
    }
}

pub(crate) unsafe fn ZSTD_customFreeBox<T>(value: Box<T>, customMem: ZSTD_customMem) {
    use core::ptr;

    if ZSTD_customMem_isNull(customMem) {
        drop(value);
        return;
    }

    let raw = Box::into_raw(value);
    unsafe {
        ptr::drop_in_place(raw);
        if let Some(customFree) = customMem.customFree {
            customFree(customMem.opaque, raw.cast());
        }
    }
}

/// Port of `ZSTD_createCCtx_advanced`.
pub fn ZSTD_createCCtx_advanced(customMem: ZSTD_customMem) -> Option<Box<ZSTD_CCtx>> {
    if !ZSTD_customMem_validate(customMem) {
        return None;
    }
    let mut cctx = unsafe { ZSTD_customAllocBox(ZSTD_CCtx::default(), customMem)? };
    cctx.customMem = customMem;
    cctx.requestedParams.customMem = customMem;
    cctx.appliedParams.customMem = customMem;
    Some(cctx)
}

/// Port of `ZSTD_createCStream_advanced`.
pub fn ZSTD_createCStream_advanced(customMem: ZSTD_customMem) -> Option<Box<ZSTD_CStream>> {
    ZSTD_createCCtx_advanced(customMem)
}

/// Port of `ZSTD_initStaticCCtx`. Places a `ZSTD_CCtx` header inside
/// the caller's workspace when alignment and size allow it.
pub fn ZSTD_initStaticCCtx(workspace: &mut [u8]) -> Option<&mut ZSTD_CCtx> {
    use core::mem::align_of;
    use core::ptr;

    if (workspace.as_mut_ptr() as usize) & (align_of::<u64>() - 1) != 0 {
        return None;
    }
    if workspace.len() < ZSTD_minStaticCCtxSize() {
        return None;
    }

    let cctx = unsafe { &mut *(workspace.as_mut_ptr() as *mut ZSTD_CCtx) };
    unsafe {
        ptr::write(cctx, ZSTD_CCtx::default());
    }
    ZSTD_initCCtx(cctx, ZSTD_customMem::default());
    cctx.staticSize = workspace.len();
    Some(cctx)
}

/// Port of `ZSTD_initStaticCStream`. Alias for `ZSTD_initStaticCCtx`.
pub fn ZSTD_initStaticCStream(workspace: &mut [u8]) -> Option<&mut ZSTD_CStream> {
    ZSTD_initStaticCCtx(workspace)
}

/// Port of `ZSTD_initStaticCDict`. Places a `ZSTD_CDict` header inside
/// the caller's workspace and initializes it through the normal CDict
/// builder path.
pub fn ZSTD_initStaticCDict<'a>(
    workspace: &'a mut [u8],
    dict: &[u8],
    cParams: crate::compress::match_state::ZSTD_compressionParameters,
) -> Option<&'a mut ZSTD_CDict> {
    use crate::compress::match_state::ZSTD_resolveRowMatchFinderMode;
    use crate::decompress::zstd_ddict::{ZSTD_dictContentType_e, ZSTD_dictLoadMethod_e};
    use core::mem::{align_of, size_of};
    use core::ptr;

    if (workspace.as_mut_ptr() as usize) & (align_of::<u64>() - 1) != 0 {
        return None;
    }
    if workspace.len() < size_of::<ZSTD_CDict>() {
        return None;
    }

    let level = (cParams.strategy as i32).clamp(1, ZSTD_MAX_CLEVEL);
    let mut params = ZSTD_CCtx_params::default();
    ZSTD_CCtxParams_init(&mut params, level);
    params.cParams = cParams;
    params.useRowMatchFinder =
        ZSTD_resolveRowMatchFinderMode(params.useRowMatchFinder, &params.cParams);

    let cdict = unsafe { &mut *(workspace.as_mut_ptr() as *mut ZSTD_CDict) };
    unsafe {
        ptr::write(
            cdict,
            ZSTD_CDict {
                dictContent: Vec::new(),
                compressionLevel: params.compressionLevel,
                dictID: 0,
                cParams: params.cParams,
                useRowMatchFinder: params.useRowMatchFinder,
                entropy: ZSTD_entropyCTables_t::default(),
                rep: ZSTD_REP_START_VALUE,
                dedicatedDictSearch: params.enableDedicatedDictSearch,
                matchState: {
                    let mut ms =
                        crate::compress::match_state::ZSTD_MatchState_t::new(params.cParams);
                    ms.dedicatedDictSearch = params.enableDedicatedDictSearch as u32;
                    ms
                },
                customMem: params.customMem,
            },
        );
    }
    let rc = ZSTD_initCDict_internal(
        cdict,
        dict,
        ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy,
        ZSTD_dictContentType_e::ZSTD_dct_auto,
        params,
    );
    if ERR_isError(rc) {
        return None;
    }
    Some(cdict)
}

/// Port of `ZSTD_CCtx_setCParams`. Validates cParams and writes each
/// field through `requestedParams.cParams` so later cParam-reading
/// code sees the updates.
pub fn ZSTD_CCtx_setCParams(
    cctx: &mut ZSTD_CCtx,
    cParams: crate::compress::match_state::ZSTD_compressionParameters,
) -> usize {
    let rc = ZSTD_checkCParams(cParams);
    if ERR_isError(rc) {
        return rc;
    }
    // Authorized cParam updates are allowed mid-stream. The buffered
    // stream path snapshots parameters when input is first staged, so
    // this only affects future work and doesn't retarget already-
    // buffered input.
    cctx.requested_cParams = Some(cParams);
    cctx.requestedParams.cParams = cParams;
    0
}

/// Port of `ZSTD_CCtx_setParams`. Applies a full `ZSTD_parameters`
/// bundle: cParams check first, then fParams via `setFParams`, then
/// the cParams themselves. Matches the upstream ordering so a
/// mid-frame stage error doesn't leave the CCtx half-updated.
pub fn ZSTD_CCtx_setParams(cctx: &mut ZSTD_CCtx, params: ZSTD_parameters) -> usize {
    let rc = ZSTD_checkCParams(params.cParams);
    if ERR_isError(rc) {
        return rc;
    }
    let rc = ZSTD_CCtx_setFParams(cctx, params.fParams);
    if ERR_isError(rc) {
        return rc;
    }
    ZSTD_CCtx_setCParams(cctx, params.cParams)
}

/// Sibling of `ZSTD_DCtx_setFormat` on the compressor side. Flips the
/// CCtx between `ZSTD_f_zstd1` (default, magic-prefixed frames) and
/// `ZSTD_f_zstd1_magicless` (no 4-byte magic prefix). The stream
/// compressor's `endStream` path (and `ZSTD_compress2`, which routes
/// through the streaming compressor) threads this through
/// `ZSTD_compressFrame_fast_advanced` / `_with_prefix_advanced` so
/// emitted frames — dict-less and dict-bearing alike — respect the
/// setting. `ZSTD_compress` / `ZSTD_compress_usingDict` stay on the
/// zstd1 format since they don't accept a cctx-scoped format knob.
pub fn ZSTD_CCtx_setFormat(
    cctx: &mut ZSTD_CCtx,
    format: crate::decompress::zstd_decompress::ZSTD_format_e,
) -> usize {
    // Sibling of `ZSTD_DCtx_setFormat`. Upstream routes through
    // `ZSTD_CCtx_setParameter(ZSTD_c_format, value)` so bounds
    // checking + enum casting lives in one spot.
    ZSTD_CCtx_setParameter(cctx, ZSTD_cParameter::ZSTD_c_format, format as i32)
}

/// Port of `ZSTD_CCtx_setParametersUsingCCtxParams`. Applies a
/// previously-prepared `ZSTD_CCtx_params` to the CCtx — pulls the
/// level + cParams + fParams through the existing setters.
pub fn ZSTD_CCtx_setParametersUsingCCtxParams(
    cctx: &mut ZSTD_CCtx,
    params: &ZSTD_CCtx_params,
) -> usize {
    // Upstream (zstd_compress.c:1198) rejects wholesale params
    // replacement mid-session — that would desync appliedParams
    // from buffered input. Gate matches the dict-family setters.
    if !cctx_is_in_init_stage(cctx) {
        return ERROR(ErrorCode::StageWrong);
    }
    if cctx.stream_cdict.is_some() {
        return ERROR(ErrorCode::StageWrong);
    }
    cctx.stream_level = Some(params.compressionLevel);
    cctx.requestedParams = *params;
    // Sync the direct `cctx.format` slot so the compressor path
    // (which reads `cctx.format`) picks up params-level magicless
    // mode. Without this, wholesale params replacement here would
    // silently revert the active format to zstd1 on the cctx.
    cctx.format = params.format;
    let rc = ZSTD_CCtx_setParams(
        cctx,
        ZSTD_parameters {
            cParams: params.cParams,
            fParams: params.fParams,
        },
    );
    if ERR_isError(rc) {
        return rc;
    }
    0
}

/// Port of `ZSTD_CCtx_setFParams`. Writes each frame-parameter flag
/// through `ZSTD_CCtx_setParameter`.
///
/// Note: upstream flips `noDictIDFlag` when recording — if it's 0
/// (dictID NOT suppressed) we set `dictIDFlag = 1`; if 1 we set 0.
pub fn ZSTD_CCtx_setFParams(cctx: &mut ZSTD_CCtx, fparams: ZSTD_FrameParameters) -> usize {
    cctx.requestedParams.fParams = fparams;
    let rc = ZSTD_CCtx_setParameter(
        cctx,
        ZSTD_cParameter::ZSTD_c_contentSizeFlag,
        (fparams.contentSizeFlag != 0) as i32,
    );
    if ERR_isError(rc) {
        return rc;
    }
    let rc = ZSTD_CCtx_setParameter(
        cctx,
        ZSTD_cParameter::ZSTD_c_checksumFlag,
        (fparams.checksumFlag != 0) as i32,
    );
    if ERR_isError(rc) {
        return rc;
    }
    let rc = ZSTD_CCtx_setParameter(
        cctx,
        ZSTD_cParameter::ZSTD_c_dictIDFlag,
        (fparams.noDictIDFlag == 0) as i32,
    );
    if ERR_isError(rc) {
        return rc;
    }
    0
}

/// Port of `ZSTD_compress2`. Public one-shot entry that honors any
/// parameters previously set via `ZSTD_CCtx_setParameter` — preferred
/// modern API over `ZSTD_compressCCtx` (which takes a level argument
/// and resets everything else).
///
/// Internally: reset the session, then drive the streaming compressor
/// to end-of-frame with stable in/out buffers.
pub fn ZSTD_compress2(cctx: &mut ZSTD_CCtx, dst: &mut [u8], src: &[u8]) -> usize {
    let old_in_buffer_mode = cctx.requestedParams.inBufferMode;
    let old_out_buffer_mode = cctx.requestedParams.outBufferMode;
    let rc = ZSTD_CCtx_reset(cctx, ZSTD_ResetDirective::ZSTD_reset_session_only);
    debug_assert!(!ERR_isError(rc));
    cctx.requestedParams.inBufferMode = ZSTD_bufferMode_e::ZSTD_bm_stable;
    cctx.requestedParams.outBufferMode = ZSTD_bufferMode_e::ZSTD_bm_stable;
    let mut dst_pos = 0usize;
    let mut src_pos = 0usize;
    let result = ZSTD_compressStream2_simpleArgs(
        cctx,
        dst,
        &mut dst_pos,
        src,
        &mut src_pos,
        ZSTD_EndDirective::ZSTD_e_end,
    );
    cctx.requestedParams.inBufferMode = old_in_buffer_mode;
    cctx.requestedParams.outBufferMode = old_out_buffer_mode;
    if ERR_isError(result) {
        return result;
    }
    if result != 0 {
        // Compression didn't finish — caller's dst was too small.
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    debug_assert_eq!(src_pos, src.len());
    dst_pos
}

/// Port of `ZSTD_initCStream_srcSize`. Initializes streaming
/// compression with a known source size and compression level —
/// equivalent to `ZSTD_initCStream` + `ZSTD_CCtx_setPledgedSrcSize`.
pub fn ZSTD_initCStream_srcSize(
    zcs: &mut ZSTD_CCtx,
    compressionLevel: i32,
    pledgedSrcSize: u64,
) -> usize {
    let rc = ZSTD_initCStream(zcs, compressionLevel);
    if ERR_isError(rc) {
        return rc;
    }
    let pledgedSrcSize = if pledgedSrcSize == 0 {
        crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN
    } else {
        pledgedSrcSize
    };
    ZSTD_CCtx_setPledgedSrcSize(zcs, pledgedSrcSize)
}

/// Port of `ZSTD_initCStream_usingDict`. Initializes streaming
/// compression that will use dictionary bytes supplied through the
/// regular loadDictionary path, including dictID tracking for
/// magic-prefixed zstd-format dictionaries.
pub fn ZSTD_initCStream_usingDict(
    zcs: &mut ZSTD_CCtx,
    dict: &[u8],
    compressionLevel: i32,
) -> usize {
    let rc = ZSTD_initCStream(zcs, compressionLevel);
    if ERR_isError(rc) {
        return rc;
    }
    // Use the full loadDictionary path so dictID / dictContentSize
    // are tracked alongside the bytes.
    ZSTD_CCtx_loadDictionary(zcs, dict)
}

/// Port of `ZSTD_initCStream_usingCDict`. Streaming-init variant that
/// links a pre-built CDict. Upstream (zstd_compress.c:6046): resets
/// session state, then `refCDict`. Our `refCDict` binds the CDict on
/// `stream_cdict`, records its level, and preserves the CDict-backed
/// session path rather than degrading to raw bytes.
pub fn ZSTD_initCStream_usingCDict(zcs: &mut ZSTD_CCtx, cdict: &ZSTD_CDict) -> usize {
    let rc = ZSTD_CCtx_reset(zcs, ZSTD_ResetDirective::ZSTD_reset_session_only);
    if ERR_isError(rc) {
        return rc;
    }
    ZSTD_CCtx_refCDict(zcs, cdict)
}

/// Port of `ZSTD_initCStream_usingCDict_advanced`. Streaming-init
/// that pairs a CDict with explicit `fParams` + pledged src size.
/// Upstream (zstd_compress.c:6032): reset → setPledgedSrcSize →
/// copy fParams → refCDict.
pub fn ZSTD_initCStream_usingCDict_advanced(
    zcs: &mut ZSTD_CCtx,
    cdict: &ZSTD_CDict,
    fParams: ZSTD_FrameParameters,
    pledgedSrcSize: u64,
) -> usize {
    let rc = ZSTD_CCtx_reset(zcs, ZSTD_ResetDirective::ZSTD_reset_session_only);
    if ERR_isError(rc) {
        return rc;
    }
    let rc = ZSTD_CCtx_setPledgedSrcSize(zcs, pledgedSrcSize);
    if ERR_isError(rc) {
        return rc;
    }
    let rc = ZSTD_CCtx_setFParams(zcs, fParams);
    if ERR_isError(rc) {
        return rc;
    }
    ZSTD_CCtx_refCDict(zcs, cdict)
}

/// Port of `ZSTD_initCStream_advanced`. Streaming-init with explicit
/// `ZSTD_parameters` bundle + caller-supplied dictionary bytes +
/// pledged src size. The dictionary still goes through the regular
/// loadDictionary path, so magic-prefixed zstd-format dictionaries
/// preserve dictID.
/// Upstream (zstd_compress.c:6059) applies a legacy back-compat rule:
/// `pss==0 && fParams.contentSizeFlag==0 → treat as CONTENTSIZE_UNKNOWN`.
pub fn ZSTD_initCStream_advanced(
    zcs: &mut ZSTD_CCtx,
    dict: &[u8],
    params: ZSTD_parameters,
    pledgedSrcSize: u64,
) -> usize {
    use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;
    let effective_pledge = if pledgedSrcSize == 0 && params.fParams.contentSizeFlag == 0 {
        ZSTD_CONTENTSIZE_UNKNOWN
    } else {
        pledgedSrcSize
    };
    let rc = ZSTD_CCtx_reset(zcs, ZSTD_ResetDirective::ZSTD_reset_session_only);
    if ERR_isError(rc) {
        return rc;
    }
    let rc = ZSTD_CCtx_setPledgedSrcSize(zcs, effective_pledge);
    if ERR_isError(rc) {
        return rc;
    }
    let rc = ZSTD_checkCParams(params.cParams);
    if ERR_isError(rc) {
        return rc;
    }
    let rc = ZSTD_CCtx_setParams(zcs, params);
    if ERR_isError(rc) {
        return rc;
    }
    ZSTD_CCtx_loadDictionary(zcs, dict)
}

/// Port of `ZSTD_resetCStream` (`zstd_compress.c:5993`). Resets the
/// stream for a new frame with the given pledged size. The
/// compression level + dict configured via the last init call are
/// preserved.
///
pub fn ZSTD_resetCStream(zcs: &mut ZSTD_CCtx, pss: u64) -> usize {
    use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;
    let pledgedSrcSize = if pss == 0 {
        ZSTD_CONTENTSIZE_UNKNOWN
    } else {
        pss
    };
    let rc = ZSTD_CCtx_reset(zcs, ZSTD_ResetDirective::ZSTD_reset_session_only);
    if ERR_isError(rc) {
        return rc;
    }
    ZSTD_CCtx_setPledgedSrcSize(zcs, pledgedSrcSize)
}

/// Port of `ZSTD_CCtx_setPledgedSrcSize`. Declares the exact number
/// of bytes the caller intends to feed through `compressStream` before
/// `endStream`. When set, the emitted frame header includes the
/// content-size field (enabling some decoder optimizations) and
/// `ZSTD_getCParams` uses the hint to pick tighter cParams.
///
/// Must be called after `ZSTD_initCStream` but before the first
/// `ZSTD_compressStream` call. Returns 0 on success. `u64::MAX` (=
/// `ZSTD_CONTENTSIZE_UNKNOWN`) clears any prior pledge so the
/// endStream size-match check is skipped.
pub fn ZSTD_CCtx_setPledgedSrcSize(zcs: &mut ZSTD_CCtx, pledgedSrcSize: u64) -> usize {
    // Upstream (zstd_compress.c:1249) rejects with `StageWrong` if the
    // stream is past the init stage. Without this gate a caller who
    // re-pledged mid-frame would silently desync
    // `pledgedSrcSizePlusOne` from the already-buffered input,
    // producing a frame header that lies about content size.
    if !cctx_is_in_init_stage(zcs) {
        return ERROR(ErrorCode::StageWrong);
    }
    zcs.pledged_src_size = if pledgedSrcSize == u64::MAX {
        None
    } else {
        Some(pledgedSrcSize)
    };
    // Upstream stores `pledgedSrcSize + 1` so 0 disambiguates
    // "unknown" (stored as 0) from "pledged exactly 0 bytes" (stored
    // as 1). UNKNOWN sentinel `u64::MAX` wraps via wrapping_add.
    zcs.pledgedSrcSizePlusOne = pledgedSrcSize.wrapping_add(1);
    0
}

/// Buffered continue-stage backend used by `ZSTD_compressStream_generic`.
/// Buffers `input[input_pos..]` into the CCtx and drains any pending
/// output to `output[output_pos..]`. Advances both cursors in place.
///
/// v0.1 scope: this buffers the ENTIRE input in the CCtx until
/// `endStream` finalizes the frame. True block-by-block streaming
/// (where `compressStream` can emit a non-last block when
/// `stream_in_buffer` reaches `ZSTD_BLOCKSIZE_MAX`) is a later
/// refinement — it requires splitting the frame header from the
/// block loop in `ZSTD_compressFrame_fast`.
fn zstd_compressStream_continue_buffered(
    zcs: &mut ZSTD_CCtx,
    output: &mut [u8],
    output_pos: &mut usize,
    input: &[u8],
    input_pos: &mut usize,
) -> usize {
    if *output_pos > output.len() {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    if *input_pos > input.len() {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    if zcs.stream_level.is_none() {
        return ERROR(ErrorCode::InitMissing);
    }
    // Drain any already-produced output first.
    let drained = stream_drain(zcs, output, output_pos);
    if ERR_isError(drained) {
        return drained;
    }
    // Stage new input.
    if !zcs.stream_closed {
        let new_input = &input[*input_pos..];
        if !new_input.is_empty() {
            zcs.stream_frame_completed = false;
            if zcs.stream_in_buffer.is_empty() && zcs.stream_params_snapshot.is_none() {
                zcs.stream_params_snapshot = Some(zstd_snapshot_stream_params(zcs));
            }
        }
        zcs.stream_in_buffer.extend_from_slice(new_input);
        *input_pos = input.len();
    }
    // Hint: any value > 0 means "more input expected". Return 1 when
    // nothing is staged yet — upstream's convention signals "call
    // endStream to finalize" via `endStream` returning 0.
    if zcs.stream_closed && zcs.stream_out_drained == zcs.stream_out_buffer.len() {
        0
    } else {
        1
    }
}

/// Port of `ZSTD_compressStream`.
pub fn ZSTD_compressStream(
    zcs: &mut ZSTD_CCtx,
    output: &mut [u8],
    output_pos: &mut usize,
    input: &[u8],
    input_pos: &mut usize,
) -> usize {
    let rc = ZSTD_compressStream2(
        zcs,
        output,
        output_pos,
        input,
        input_pos,
        ZSTD_EndDirective::ZSTD_e_continue,
    );
    if ERR_isError(rc) {
        return rc;
    }
    ZSTD_nextInputSizeHint_MTorST(zcs)
}

/// Drain as much of `zcs.stream_out_buffer[stream_out_drained..]` as
/// fits into `output[output_pos..]`. Returns 0 on success or error.
fn stream_drain(zcs: &mut ZSTD_CCtx, output: &mut [u8], output_pos: &mut usize) -> usize {
    if *output_pos > output.len() {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    let avail = output.len() - *output_pos;
    let pending = zcs.stream_out_buffer.len() - zcs.stream_out_drained;
    let n = avail.min(pending);
    if n > 0 {
        output[*output_pos..*output_pos + n].copy_from_slice(
            &zcs.stream_out_buffer[zcs.stream_out_drained..zcs.stream_out_drained + n],
        );
        zcs.stream_out_drained += n;
        *output_pos += n;
    }
    0
}

fn zstd_init_stream_from_snapshot(
    zcs: &mut ZSTD_CCtx,
    end_op: ZSTD_EndDirective,
    in_size: usize,
) -> usize {
    if zcs.stage != ZSTD_compressionStage_e::ZSTDcs_created {
        return 0;
    }

    let snapshot = zcs
        .stream_params_snapshot
        .unwrap_or_else(|| zstd_snapshot_stream_params(zcs));
    let future_requestedParams = zcs.requestedParams;
    let future_requested_cParams = zcs.requested_cParams;
    let future_stream_level = zcs.stream_level;
    let future_param_checksum = zcs.param_checksum;
    let future_param_contentSize = zcs.param_contentSize;
    let future_param_dictID = zcs.param_dictID;
    let future_format = zcs.format;

    zstd_apply_stream_params_snapshot(zcs, snapshot);
    let rc = ZSTD_CCtx_init_compressStream2(zcs, end_op, in_size);

    zcs.requestedParams = future_requestedParams;
    zcs.requested_cParams = future_requested_cParams;
    zcs.stream_level = future_stream_level;
    zcs.param_checksum = future_param_checksum;
    zcs.param_contentSize = future_param_contentSize;
    zcs.param_dictID = future_param_dictID;
    zcs.format = future_format;
    zcs.stream_params_snapshot = if end_op == ZSTD_EndDirective::ZSTD_e_end {
        None
    } else {
        Some(snapshot)
    };
    rc
}

fn zstd_stream_window_capacity(zcs: &ZSTD_CCtx) -> usize {
    let window_size = (1usize << zcs.appliedParams.cParams.windowLog)
        .min(zcs.pledged_src_size.unwrap_or(u64::MAX) as usize)
        .max(1);
    window_size.saturating_add(zcs.blockSizeMax.max(1))
}

fn zstd_stream_can_use_windowed(zcs: &ZSTD_CCtx) -> bool {
    zstd_stream_can_use_windowed_with_stable_one_shot(zcs, false)
}

fn zstd_stream_can_use_windowed_with_stable_one_shot(
    zcs: &ZSTD_CCtx,
    allow_stable_one_shot: bool,
) -> bool {
    use crate::compress::zstd_compress_sequences::ZSTD_btopt;

    let level = zcs
        .stream_level
        .unwrap_or(zcs.requestedParams.compressionLevel);
    let strategy = zcs
        .appliedParams
        .cParams
        .strategy
        .max(zcs.requestedParams.cParams.strategy);
    let buffered_modes = zcs.requestedParams.inBufferMode == ZSTD_bufferMode_e::ZSTD_bm_buffered
        && zcs.requestedParams.outBufferMode == ZSTD_bufferMode_e::ZSTD_bm_buffered;
    let stable_one_shot_modes = allow_stable_one_shot
        && zcs.requestedParams.inBufferMode == ZSTD_bufferMode_e::ZSTD_bm_stable
        && zcs.requestedParams.outBufferMode == ZSTD_bufferMode_e::ZSTD_bm_stable;
    !zcs.stream_disable_windowed
        && zcs.stream_cdict.is_none()
        && zcs.stream_dict.is_empty()
        && !zcs.prefix_is_single_use
        && (1..=22).contains(&level)
        && level < 16
        && strategy < ZSTD_btopt
        && zcs.requestedParams.nbWorkers == 0
        && zcs.appliedParams.nbWorkers == 0
        && (buffered_modes || stable_one_shot_modes)
}

fn zstd_stream_compress_windowed_block(
    zcs: &mut ZSTD_CCtx,
    output: &mut [u8],
    output_pos: &mut usize,
    src_pos: usize,
    src_len: usize,
    last_block: bool,
) -> usize {
    use crate::compress::match_state::ZSTD_window_update;

    let bound = if src_len == 0 {
        ZSTD_FRAMEHEADERSIZE_MAX + ZSTD_blockHeaderSize + 8
    } else {
        ZSTD_compressBound(src_len).saturating_add(ZSTD_FRAMEHEADERSIZE_MAX + 8)
    };
    if ERR_isError(bound) {
        return bound;
    }
    let compressed_capacity = bound.max(32);
    let direct_to_output = output.len().saturating_sub(*output_pos) >= compressed_capacity;
    let mut compressed = Vec::new();
    let dst = if direct_to_output {
        unsafe { output.as_mut_ptr().add(*output_pos) }
    } else {
        compressed = std::mem::take(&mut zcs.stream_out_buffer);
        compressed.clear();
        if compressed.capacity() < compressed_capacity {
            compressed.reserve_exact(compressed_capacity);
        }
        debug_assert!(
            compressed.capacity() >= compressed_capacity,
            "stream out capacity {} < requested {} for src_len {} bound {}",
            compressed.capacity(),
            compressed_capacity,
            src_len,
            bound
        );
        compressed.as_mut_ptr()
    };
    let mut op = 0usize;

    if zcs.stage == ZSTD_compressionStage_e::ZSTDcs_created {
        return ERROR(ErrorCode::StageWrong);
    }

    if zcs.stage == ZSTD_compressionStage_e::ZSTDcs_init {
        let pledged = ZSTD_getPledgedSrcSize(zcs);
        let fhSize = if zcs.appliedParams.format
            == crate::decompress::zstd_decompress::ZSTD_format_e::ZSTD_f_zstd1
        {
            ZSTD_writeFrameHeader(
                unsafe { core::slice::from_raw_parts_mut(dst.add(op), compressed_capacity - op) },
                &zcs.appliedParams.fParams,
                zcs.appliedParams.cParams.windowLog,
                pledged,
                zcs.dictID,
            )
        } else {
            ZSTD_writeFrameHeader_advanced(
                unsafe { core::slice::from_raw_parts_mut(dst.add(op), compressed_capacity - op) },
                &zcs.appliedParams.fParams,
                zcs.appliedParams.cParams.windowLog,
                pledged,
                zcs.dictID,
                zcs.appliedParams.format,
            )
        };
        if ERR_isError(fhSize) {
            return fhSize;
        }
        op += fhSize;
        zcs.stage = ZSTD_compressionStage_e::ZSTDcs_ongoing;
    }

    if zcs.ms.is_none() {
        zcs.ms = Some(crate::compress::match_state::ZSTD_MatchState_t::new(
            zcs.appliedParams.cParams,
        ));
    }
    let rc = ZSTD_initLocalDict(zcs);
    if ERR_isError(rc) {
        return rc;
    }

    let src_abs = zcs.stream_window_base.wrapping_add(src_pos as u32);
    {
        let ms = zcs.ms.as_mut().unwrap();
        ms.window.base_offset = zcs.stream_window_base;
        if !ZSTD_window_update(&mut ms.window, src_abs, src_len, ms.forceNonContiguous) {
            ms.forceNonContiguous = false;
            ms.nextToUpdate = ms.window.dictLimit;
        }
    }

    if zcs.appliedParams.ldmEnable == crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_enable
    {
        if let Some(ldmState) = zcs.ldmState.as_mut() {
            ZSTD_window_update(&mut ldmState.window, src_abs, src_len, false);
        }
    }

    let ptr = zcs.stream_in_buffer.as_ptr();
    let len = zcs.stream_in_buffer.len();
    let window = unsafe { core::slice::from_raw_parts(ptr, len) };
    let cSize = ZSTD_compress_frameChunk_windowed(
        zcs,
        unsafe { core::slice::from_raw_parts_mut(dst.add(op), compressed_capacity - op) },
        window,
        src_pos,
        src_len,
        last_block as u32,
    );
    if ERR_isError(cSize) {
        return cSize;
    }
    op += cSize;
    zcs.consumedSrcSize = zcs.consumedSrcSize.saturating_add(src_len as u64);
    zcs.producedCSize = zcs.producedCSize.saturating_add(op as u64);
    if zcs.pledgedSrcSizePlusOne != 0 && zcs.consumedSrcSize + 1 > zcs.pledgedSrcSizePlusOne {
        return ERROR(ErrorCode::SrcSizeWrong);
    }

    if last_block {
        let endSize = ZSTD_writeEpilogue(zcs, unsafe {
            core::slice::from_raw_parts_mut(dst.add(op), compressed_capacity - op)
        });
        if ERR_isError(endSize) {
            return endSize;
        }
        op += endSize;
        if zcs.pledgedSrcSizePlusOne != 0 && zcs.pledgedSrcSizePlusOne != zcs.consumedSrcSize + 1 {
            return ERROR(ErrorCode::SrcSizeWrong);
        }
        zcs.stream_frame_ended = true;
        zcs.stream_closed = true;
    }

    if direct_to_output {
        *output_pos += op;
        zcs.stream_out_buffer.clear();
    } else {
        unsafe {
            compressed.set_len(op);
        }
        zcs.stream_out_buffer = compressed;
    }
    zcs.stream_out_drained = 0;
    zcs.stream_stage = ZSTD_cStreamStage::zcss_flush;
    0
}

fn zstd_stream_trim_window(zcs: &mut ZSTD_CCtx) {
    let level = zcs
        .stream_level
        .unwrap_or(zcs.appliedParams.compressionLevel);
    if level >= 8 && zcs.stream_in_buffer.capacity() < (64 << 20) {
        return;
    }
    let keep = zstd_stream_window_capacity(zcs).saturating_sub(zcs.blockSizeMax);
    let current_start = zcs.stream_in_to_compress.min(zcs.stream_in_buffer.len());
    let drain = current_start.saturating_sub(keep);
    if drain != 0 {
        zcs.stream_in_buffer.drain(..drain);
        zcs.stream_window_base = zcs.stream_window_base.wrapping_add(drain as u32);
        zcs.stream_in_to_compress = zcs.stream_in_to_compress.saturating_sub(drain);
        zcs.stream_in_target = zcs.stream_in_target.saturating_sub(drain);
        if let Some(ms) = zcs.ms.as_mut() {
            ms.window.base_offset = zcs.stream_window_base;
        }
    }
}

fn zstd_compressStream_windowed(
    zcs: &mut ZSTD_CCtx,
    output: &mut [u8],
    output_pos: &mut usize,
    input: &[u8],
    input_pos: &mut usize,
    flushMode: ZSTD_EndDirective,
    allow_stable_one_shot: bool,
) -> usize {
    use crate::compress::zstd_compress::ZSTD_EndDirective::{
        ZSTD_e_continue, ZSTD_e_end, ZSTD_e_flush,
    };

    loop {
        let rc = stream_drain(zcs, output, output_pos);
        if ERR_isError(rc) {
            return rc;
        }
        if zcs.stream_out_drained != zcs.stream_out_buffer.len() {
            return zcs.stream_out_buffer.len() - zcs.stream_out_drained;
        }
        if zcs.stream_stage == ZSTD_cStreamStage::zcss_flush {
            zcs.stream_out_buffer.clear();
            zcs.stream_out_drained = 0;
            if zcs.stream_frame_ended {
                cctx_mark_stream_frame_completed(zcs);
                return 0;
            }
            zcs.stream_stage = ZSTD_cStreamStage::zcss_load;
        }

        if zcs.stream_stage == ZSTD_cStreamStage::zcss_init {
            if zcs.stream_params_snapshot.is_none() {
                zcs.stream_params_snapshot = Some(zstd_snapshot_stream_params(zcs));
            }
            let remaining = input.len().saturating_sub(*input_pos);
            let init_size = if flushMode == ZSTD_e_end {
                remaining
            } else {
                0
            };
            let rc = zstd_init_stream_from_snapshot(zcs, flushMode, init_size);
            if ERR_isError(rc) {
                return rc;
            }
            if !zstd_stream_can_use_windowed_with_stable_one_shot(zcs, allow_stable_one_shot) {
                return ERROR(ErrorCode::ParameterUnsupported);
            }
        }

        let target = if zcs.stream_in_target == 0 {
            zcs.blockSizeMax
        } else {
            zcs.stream_in_target
        };
        if zcs.stream_in_buffer.len() < target && *input_pos < input.len() {
            let capacity = zstd_stream_window_capacity(zcs).max(target);
            let to_load = (target - zcs.stream_in_buffer.len())
                .min(input.len().saturating_sub(*input_pos))
                .min(capacity.saturating_sub(zcs.stream_in_buffer.len()));
            if to_load != 0 {
                zcs.stream_in_buffer
                    .extend_from_slice(&input[*input_pos..*input_pos + to_load]);
                *input_pos += to_load;
                zcs.stream_frame_completed = false;
            }
        }

        let available = zcs
            .stream_in_buffer
            .len()
            .saturating_sub(zcs.stream_in_to_compress);
        if flushMode == ZSTD_e_continue && available < zcs.blockSizeMax {
            return zcs.blockSizeMax - available;
        }
        if flushMode == ZSTD_e_flush && available == 0 {
            return 0;
        }
        if flushMode == ZSTD_e_end && available == 0 && *input_pos == input.len() {
            let rc = zstd_stream_compress_windowed_block(
                zcs,
                output,
                output_pos,
                zcs.stream_in_to_compress,
                0,
                true,
            );
            if ERR_isError(rc) {
                return rc;
            }
            continue;
        }
        if available == 0 {
            return zcs.blockSizeMax;
        }

        let last = flushMode == ZSTD_e_end && *input_pos == input.len();
        let block_len = if last || flushMode == ZSTD_e_flush {
            available
        } else {
            available.min(zcs.blockSizeMax)
        };
        let block_start = zcs.stream_in_to_compress;
        let rc = zstd_stream_compress_windowed_block(
            zcs,
            output,
            output_pos,
            block_start,
            block_len,
            last,
        );
        if ERR_isError(rc) {
            return rc;
        }
        zcs.stream_in_to_compress += block_len;
        zcs.stream_in_target = zcs.stream_in_to_compress + zcs.blockSizeMax;
        zstd_stream_trim_window(zcs);
    }
}

/// Buffered flush-stage backend used by `ZSTD_compressStream_generic`.
/// Flush initializes the frame when needed and emits all currently
/// staged input as non-final block(s), then drains pending output.
fn zstd_flushStream_buffered(
    zcs: &mut ZSTD_CCtx,
    output: &mut [u8],
    output_pos: &mut usize,
) -> usize {
    let rc = stream_drain(zcs, output, output_pos);
    if ERR_isError(rc) {
        return rc;
    }
    let remaining = zcs.stream_out_buffer.len() - zcs.stream_out_drained;
    if remaining != 0 {
        return remaining;
    }

    if !zcs.stream_closed && !zcs.stream_in_buffer.is_empty() {
        let src = std::mem::take(&mut zcs.stream_in_buffer);
        let rc = zstd_init_stream_from_snapshot(zcs, ZSTD_EndDirective::ZSTD_e_flush, src.len());
        if ERR_isError(rc) {
            zcs.stream_in_buffer = src;
            return rc;
        }
        let bound = ZSTD_compressBound(src.len());
        if ERR_isError(bound) {
            zcs.stream_in_buffer = src;
            return bound;
        }
        let mut compressed = vec![0u8; bound.max(32)];
        let n = ZSTD_compressContinue_public(zcs, &mut compressed, &src);
        if ERR_isError(n) {
            return n;
        }
        compressed.truncate(n);
        zcs.stream_out_buffer = compressed;
        zcs.stream_out_drained = 0;
        let rc = stream_drain(zcs, output, output_pos);
        if ERR_isError(rc) {
            return rc;
        }
    }

    zcs.stream_out_buffer.len() - zcs.stream_out_drained
}

/// Port of `ZSTD_flushStream`.
pub fn ZSTD_flushStream(zcs: &mut ZSTD_CCtx, output: &mut [u8], output_pos: &mut usize) -> usize {
    if *output_pos > output.len() {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    let output_len = output.len();
    let mut output_buffer = ZSTD_outBuffer {
        dst: Some(output),
        size: output_len,
        pos: *output_pos,
    };
    let mut input_buffer = inBuffer_forEndFlush(zcs);
    input_buffer.size = input_buffer.pos;
    let result = ZSTD_compressStream_generic(
        zcs,
        &mut output_buffer,
        &mut input_buffer,
        ZSTD_EndDirective::ZSTD_e_flush,
    );
    *output_pos = output_buffer.pos;
    result
}

#[inline]
fn zstd_effective_in_buffer_mode(cctx: &ZSTD_CCtx) -> ZSTD_bufferMode_e {
    if cctx.stage == ZSTD_compressionStage_e::ZSTDcs_created {
        cctx.requestedParams.inBufferMode
    } else {
        cctx.appliedParams.inBufferMode
    }
}

#[inline]
fn zstd_effective_out_buffer_mode(cctx: &ZSTD_CCtx) -> ZSTD_bufferMode_e {
    if cctx.stage == ZSTD_compressionStage_e::ZSTDcs_created {
        cctx.requestedParams.outBufferMode
    } else {
        cctx.appliedParams.outBufferMode
    }
}

/// Port of `inBuffer_forEndFlush`. Upstream returns the last stable
/// input buffer when `ZSTD_bm_stable` is enabled, else a null input.
/// The Rust port preserves the size/pos metadata used by stable-input
/// validation, but doesn't reconstruct a borrowed slice from the saved
/// raw pointer because the stream path still eagerly copies input.
pub fn inBuffer_forEndFlush<'a>(zcs: &ZSTD_CCtx) -> ZSTD_inBuffer<'a> {
    if zstd_effective_in_buffer_mode(zcs) == ZSTD_bufferMode_e::ZSTD_bm_stable {
        return ZSTD_inBuffer {
            src: None,
            size: zcs.expected_in_size,
            pos: zcs.expected_in_pos,
        };
    }
    ZSTD_inBuffer {
        src: None,
        size: 0,
        pos: 0,
    }
}

/// Port of `ZSTD_nextInputSizeHint`. In the v0.1 buffered streaming
/// model, the next useful chunk is simply one block.
pub fn ZSTD_nextInputSizeHint(_zcs: &ZSTD_CCtx) -> usize {
    ZSTD_CStreamInSize()
}

/// Port of `ZSTD_nextInputSizeHint_MTorST`. Multithreaded mode isn't
/// implemented, so this defers to the single-thread hint.
pub fn ZSTD_nextInputSizeHint_MTorST(cctx: &ZSTD_CCtx) -> usize {
    ZSTD_nextInputSizeHint(cctx)
}

/// Port of `ZSTD_registerSequenceProducer` (`zstd_compress.c:8346`).
/// Stores the opaque callback/state pair on `requestedParams`.
pub fn ZSTD_registerSequenceProducer(
    zc: &mut ZSTD_CCtx,
    extSeqProdState: usize,
    extSeqProdFunc: Option<ZSTD_sequenceProducer_F>,
) {
    ZSTD_CCtxParams_registerSequenceProducer(
        &mut zc.requestedParams,
        extSeqProdState,
        extSeqProdFunc,
    );
}

/// Port of `ZSTD_CCtxParams_registerSequenceProducer`
/// (`zstd_compress.c:8357`). Stores the opaque callback/state pair for
/// later init-time propagation through `requestedParams`/`appliedParams`.
pub fn ZSTD_CCtxParams_registerSequenceProducer(
    params: &mut ZSTD_CCtx_params,
    extSeqProdState: usize,
    extSeqProdFunc: Option<ZSTD_sequenceProducer_F>,
) {
    if extSeqProdFunc.is_some() {
        params.extSeqProdState = extSeqProdState;
        params.extSeqProdFunc = extSeqProdFunc;
    } else {
        params.extSeqProdState = 0;
        params.extSeqProdFunc = None;
    }
}

/// Port of `ZSTD_setBufferExpectations`.
pub fn ZSTD_setBufferExpectations(
    cctx: &mut ZSTD_CCtx,
    output: &ZSTD_outBuffer<'_>,
    input: &ZSTD_inBuffer<'_>,
) {
    if zstd_effective_in_buffer_mode(cctx) == ZSTD_bufferMode_e::ZSTD_bm_stable {
        cctx.expected_in_src = input.src.map_or(0, |src| src.as_ptr() as usize);
        cctx.expected_in_size = input.size;
        cctx.expected_in_pos = input.pos;
    }
    if zstd_effective_out_buffer_mode(cctx) == ZSTD_bufferMode_e::ZSTD_bm_stable {
        cctx.expected_out_buffer_size = output.size.saturating_sub(output.pos);
    }
    cctx.buffer_expectations_set = true;
}

/// Port of `ZSTD_checkBufferStability`.
pub fn ZSTD_checkBufferStability(
    cctx: &ZSTD_CCtx,
    output: &ZSTD_outBuffer<'_>,
    input: &ZSTD_inBuffer<'_>,
    endOp: ZSTD_EndDirective,
) -> usize {
    if !cctx.buffer_expectations_set {
        return 0;
    }
    if zstd_effective_in_buffer_mode(cctx) == ZSTD_bufferMode_e::ZSTD_bm_stable {
        let same_src = cctx.expected_in_src == input.src.map_or(0, |src| src.as_ptr() as usize);
        let synthetic_end_flush = input.src.is_none()
            && endOp != ZSTD_EndDirective::ZSTD_e_continue
            && cctx.expected_in_size == input.size
            && cctx.expected_in_pos == input.pos;
        if (!same_src && !synthetic_end_flush) || cctx.expected_in_pos != input.pos {
            return ERROR(ErrorCode::StabilityConditionNotRespected);
        }
    }
    if zstd_effective_out_buffer_mode(cctx) == ZSTD_bufferMode_e::ZSTD_bm_stable
        && cctx.expected_out_buffer_size != output.size.saturating_sub(output.pos)
    {
        return ERROR(ErrorCode::StabilityConditionNotRespected);
    }
    0
}

/// Buffered end-of-frame implementation used by the public
/// `ZSTD_endStream()` wrapper and `ZSTD_compressStream2(..., e_end)`.
/// On the first end call it compresses the staged input into
/// `stream_out_buffer`, then drains that buffer into the caller.
fn zstd_endStream_buffered(
    zcs: &mut ZSTD_CCtx,
    output: &mut [u8],
    output_pos: &mut usize,
) -> usize {
    if zcs.stream_frame_completed
        && zcs.stream_in_buffer.is_empty()
        && zcs.stream_out_buffer.is_empty()
    {
        return 0;
    }
    if zcs.stream_level.is_none() {
        return ERROR(ErrorCode::InitMissing);
    }
    let stream_params = zcs
        .stream_params_snapshot
        .unwrap_or_else(|| zstd_snapshot_stream_params(zcs));
    // First endStream call on this frame: compress the staged input
    // into stream_out_buffer.
    if !zcs.stream_closed {
        let src = std::mem::take(&mut zcs.stream_in_buffer);
        // Validate pledged size matches everything emitted so far plus
        // the final staged tail. Prior `flushStream()` calls may have
        // already advanced `consumedSrcSize`.
        if let Some(pledged) = zcs.pledged_src_size {
            if pledged != zcs.consumedSrcSize.saturating_add(src.len() as u64) {
                return ERROR(ErrorCode::SrcSizeWrong);
            }
        }
        let bound = ZSTD_compressBound(src.len());
        if ERR_isError(bound) {
            return bound;
        }
        // Upstream (zstd_compress.c:6381) zeroes `prefixDict`
        // BEFORE running the compress: a local copy carries the
        // prefix into this frame, but the cctx slot is wiped
        // eagerly so an error path still leaves the flag clean.
        // Mirror by snapshotting the dict + flag, then wiping the
        // cctx-side single-use state. The compress below consumes
        // the snapshot; success-or-failure the flag is already gone.
        let prefix_snapshot: Option<(
            Vec<u8>,
            Vec<u8>,
            crate::decompress::zstd_ddict::ZSTD_dictContentType_e,
        )> = if zcs.prefix_is_single_use {
            let content = std::mem::take(&mut zcs.stream_dict);
            let original = std::mem::take(&mut zcs.stream_dict_original);
            let content_type = zcs.stream_dict_content_type;
            zcs.stream_dict_content_type =
                crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_auto;
            zcs.dictID = 0;
            zcs.dictContentSize = 0;
            zcs.prefix_is_single_use = false;
            Some((content, original, content_type))
        } else {
            None
        };
        #[cfg(feature = "mt")]
        let maybe_mt = if stream_params.requestedParams.nbWorkers > 0
            && src.len() > crate::compress::zstdmt_compress::ZSTDMT_JOBSIZE_MIN
        {
            let pledged_is_known = zcs.pledged_src_size.is_some();
            let effective_prefix: &[u8] = prefix_snapshot
                .as_ref()
                .map(|(content, _, _)| content.as_slice())
                .unwrap_or(zcs.stream_dict.as_slice());
            let fp = ZSTD_FrameParameters {
                contentSizeFlag: if stream_params.param_contentSize && pledged_is_known {
                    1
                } else {
                    0
                },
                checksumFlag: if stream_params.param_checksum { 1 } else { 0 },
                noDictIDFlag: if stream_params.param_dictID { 0 } else { 1 },
            };
            Some(zstd_endstream_mt_compress(
                stream_params.requestedParams,
                stream_params.format,
                zcs.threadPoolRef,
                zcs.rayonThreadPoolRef,
                stream_params.stream_level,
                zcs.pledged_src_size
                    .unwrap_or(crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN),
                &src,
                effective_prefix,
                fp,
            ))
        } else {
            None
        };

        #[cfg(not(feature = "mt"))]
        let maybe_mt: Option<Result<Vec<u8>, usize>> = None;

        zcs.stream_out_buffer = if let Some(result) = maybe_mt {
            match result {
                Ok(buf) => buf,
                Err(code) => return code,
            }
        } else {
            use crate::compress::match_state::{
                ZSTD_resolveBlockSplitterMode, ZSTD_resolveEnableLdm,
                ZSTD_resolveExternalRepcodeSearch, ZSTD_resolveExternalSequenceValidation,
                ZSTD_resolveMaxBlockSize, ZSTD_resolveRowMatchFinderMode,
            };
            use crate::decompress::zstd_ddict::ZSTD_dictContentType_e;

            let persistent_raw_dict = if prefix_snapshot.is_none() && zcs.stream_cdict.is_none() {
                Some((
                    zcs.stream_dict.clone(),
                    zcs.stream_dict_original.clone(),
                    zcs.stream_dict_content_type,
                    zcs.dictID,
                    zcs.dictContentSize,
                    zcs.prefix_is_single_use,
                ))
            } else {
                None
            };

            if let Some((prefix_content, prefix_original, prefix_content_type)) = prefix_snapshot {
                zcs.stream_dict = prefix_content;
                zcs.stream_dict_original = prefix_original;
                zcs.stream_dict_content_type = prefix_content_type;
                zcs.dictContentSize = if zcs.stream_dict_original.is_empty() {
                    zcs.stream_dict.len()
                } else {
                    zcs.stream_dict_original.len()
                };
            }

            let pledged = ZSTD_getPledgedSrcSize(zcs);
            let pledged_is_known = zcs.pledged_src_size.is_some();
            let cdict_snapshot = zcs.stream_cdict.clone();
            let cdict = cdict_snapshot.as_ref();
            let raw_dict_snapshot = if cdict.is_none() && !zcs.stream_dict.is_empty() {
                let original = if zcs.stream_dict_original.is_empty() {
                    zcs.stream_dict.clone()
                } else {
                    zcs.stream_dict_original.clone()
                };
                Some((original, zcs.stream_dict_content_type))
            } else {
                None
            };
            let (dict, dict_content_type) = raw_dict_snapshot
                .as_ref()
                .map(|(dict, content_type)| (dict.as_slice(), *content_type))
                .unwrap_or((&[][..], ZSTD_dictContentType_e::ZSTD_dct_auto));
            let raw_cdict_compat = cdict.is_some();
            let dictSize = cdict.map_or_else(
                || {
                    if zcs.dictContentSize != 0 {
                        zcs.dictContentSize
                    } else {
                        dict.len()
                    }
                },
                |cd| cd.dictContent.len(),
            );
            let mode = ZSTD_getCParamMode(
                if raw_cdict_compat { None } else { cdict },
                &stream_params.requestedParams,
                pledged,
            );
            let mut params = stream_params.requestedParams;
            params.compressionLevel = stream_params.stream_level;
            params.format = stream_params.format;
            params.fParams = ZSTD_FrameParameters {
                contentSizeFlag: if stream_params.param_contentSize && pledged_is_known {
                    1
                } else {
                    0
                },
                checksumFlag: if stream_params.param_checksum { 1 } else { 0 },
                noDictIDFlag: if stream_params.param_dictID { 0 } else { 1 },
            };
            params.cParams = ZSTD_getCParamsFromCCtxParams(&params, pledged, dictSize, mode);
            params.postBlockSplitter =
                ZSTD_resolveBlockSplitterMode(params.postBlockSplitter, &params.cParams);
            params.ldmParams.enableLdm =
                ZSTD_resolveEnableLdm(params.ldmParams.enableLdm, &params.cParams);
            params.ldmEnable = params.ldmParams.enableLdm;
            params.useRowMatchFinder =
                ZSTD_resolveRowMatchFinderMode(params.useRowMatchFinder, &params.cParams);
            params.validateSequences =
                ZSTD_resolveExternalSequenceValidation(params.validateSequences);
            params.maxBlockSize = ZSTD_resolveMaxBlockSize(params.maxBlockSize);
            params.searchForExternalRepcodes = ZSTD_resolveExternalRepcodeSearch(
                params.searchForExternalRepcodes,
                params.compressionLevel,
            );

            let future_requestedParams = zcs.requestedParams;
            let future_requested_cParams = zcs.requested_cParams;
            let future_stream_level = zcs.stream_level;
            zcs.requestedParams = params;
            let direct_to_output = bound <= output.len().saturating_sub(*output_pos);
            let mut compressed = if direct_to_output {
                Vec::new()
            } else {
                vec![0u8; bound.max(32)]
            };
            let n = if let Some(cd) = cdict {
                let init = ZSTD_compressBegin_usingCDict_internal(zcs, cd, params.fParams, pledged);
                if ERR_isError(init) {
                    init
                } else if direct_to_output {
                    ZSTD_compressEnd_public(zcs, &mut output[*output_pos..], &src)
                } else {
                    ZSTD_compressEnd_public(zcs, &mut compressed, &src)
                }
            } else if !dict.is_empty() {
                let init = ZSTD_compressBegin_internal(
                    zcs,
                    dict,
                    dict_content_type,
                    None,
                    &params,
                    pledged,
                    ZSTD_buffered_policy_e::ZSTDb_buffered,
                );
                if ERR_isError(init) {
                    init
                } else if direct_to_output {
                    ZSTD_compressEnd_public(zcs, &mut output[*output_pos..], &src)
                } else {
                    ZSTD_compressEnd_public(zcs, &mut compressed, &src)
                }
            } else if zcs.stage != ZSTD_compressionStage_e::ZSTDcs_created {
                if direct_to_output {
                    ZSTD_compressEnd_public(zcs, &mut output[*output_pos..], &src)
                } else {
                    ZSTD_compressEnd_public(zcs, &mut compressed, &src)
                }
            } else if direct_to_output {
                ZSTD_compress_advanced_internal(
                    zcs,
                    &mut output[*output_pos..],
                    &src,
                    dict,
                    &params,
                )
            } else {
                ZSTD_compress_advanced_internal(zcs, &mut compressed, &src, dict, &params)
            };
            if let Some((
                stream_dict,
                stream_dict_original,
                stream_dict_content_type,
                dictID,
                dictContentSize,
                prefix_is_single_use,
            )) = persistent_raw_dict.as_ref()
            {
                zcs.stream_dict = stream_dict.clone();
                zcs.stream_dict_original = stream_dict_original.clone();
                zcs.stream_dict_content_type = *stream_dict_content_type;
                zcs.dictID = *dictID;
                zcs.dictContentSize = *dictContentSize;
                zcs.prefix_is_single_use = *prefix_is_single_use;
            } else if cdict.is_none() {
                zcs.stream_dict.clear();
                zcs.stream_dict_original.clear();
                zcs.stream_dict_content_type = ZSTD_dictContentType_e::ZSTD_dct_auto;
                zcs.dictID = 0;
                zcs.dictContentSize = 0;
            }
            zcs.requestedParams = future_requestedParams;
            zcs.requested_cParams = future_requested_cParams;
            zcs.stream_level = future_stream_level;
            if ERR_isError(n) {
                return n;
            }
            if direct_to_output {
                *output_pos += n;
                Vec::new()
            } else {
                compressed.truncate(n);
                compressed
            }
        };
        zcs.stream_out_drained = 0;
        zcs.stream_closed = true;
    }
    stream_drain(zcs, output, output_pos);
    let remaining = zcs.stream_out_buffer.len() - zcs.stream_out_drained;
    if remaining == 0 {
        cctx_mark_stream_frame_completed(zcs);
    }
    remaining
}

/// Port of `ZSTD_endStream`. Upstream reaches the end-of-frame logic
/// through `compressStream2(..., ZSTD_e_end)` using the synthetic
/// `inBuffer_forEndFlush()` metadata. Mirror that wrapper-level
/// stability/expectation handling here, while keeping the current
/// buffered finalization backend.
pub fn ZSTD_endStream(zcs: &mut ZSTD_CCtx, output: &mut [u8], output_pos: &mut usize) -> usize {
    if *output_pos > output.len() {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    let mut output_buffer = ZSTD_outBuffer {
        dst: None,
        size: output.len(),
        pos: *output_pos,
    };
    let input_buffer = inBuffer_forEndFlush(zcs);
    let stable_rc = ZSTD_checkBufferStability(
        zcs,
        &output_buffer,
        &input_buffer,
        ZSTD_EndDirective::ZSTD_e_end,
    );
    if ERR_isError(stable_rc) {
        return stable_rc;
    }
    if zcs.stream_stage != ZSTD_cStreamStage::zcss_init && zstd_stream_can_use_windowed(zcs) {
        let mut empty_pos = 0usize;
        let result = zstd_compressStream_windowed(
            zcs,
            output,
            output_pos,
            &[],
            &mut empty_pos,
            ZSTD_EndDirective::ZSTD_e_end,
            false,
        );
        if ERR_isError(result) {
            return result;
        }
        output_buffer.pos = *output_pos;
        ZSTD_setBufferExpectations(zcs, &output_buffer, &input_buffer);
        return result;
    }
    let result = zstd_endStream_buffered(zcs, output, output_pos);
    if ERR_isError(result) {
        return result;
    }
    output_buffer.pos = *output_pos;
    ZSTD_setBufferExpectations(zcs, &output_buffer, &input_buffer);
    result
}

/// Rust-only helper: builds a `ZSTDMT_CCtx`, primes it with the
/// caller-supplied params + optional prefix dict + attached pool, and
/// drains the MT compressor to a fresh `Vec` in one shot. Used by the
/// streaming `endStream` path when `nbWorkers > 0`.
#[cfg(feature = "mt")]
fn zstd_endstream_mt_compress(
    requested_params: ZSTD_CCtx_params,
    format: crate::decompress::zstd_decompress::ZSTD_format_e,
    thread_pool_ref: usize,
    rayon_thread_pool_ref: usize,
    level: i32,
    pledged_src_size: u64,
    src: &[u8],
    effective_prefix: &[u8],
    fp: ZSTD_FrameParameters,
) -> Result<Vec<u8>, usize> {
    use crate::compress::zstdmt_compress::{
        Range, ZSTDMT_compressStream_generic, ZSTDMT_createCCtx, ZSTDMT_freeCCtx,
        ZSTDMT_initCStream_internal, ZSTDMT_setRayonThreadPool, ZSTDMT_setThreadPool,
    };

    let nb_workers = requested_params.nbWorkers.max(1) as u32;
    let mut mtctx = match ZSTDMT_createCCtx(nb_workers) {
        Some(mtctx) => mtctx,
        None => return Err(ERROR(ErrorCode::MemoryAllocation)),
    };
    let mut params = requested_params;
    params.compressionLevel = level;
    params.fParams = fp;
    params.format = format;
    params.nbWorkers = nb_workers as i32;
    let init = ZSTDMT_initCStream_internal(&mut mtctx, params, pledged_src_size);
    if ERR_isError(init) {
        let _ = ZSTDMT_freeCCtx(Some(mtctx));
        return Err(init);
    }
    if rayon_thread_pool_ref != 0 {
        let pool = unsafe { &*(rayon_thread_pool_ref as *const rayon::ThreadPool) };
        ZSTDMT_setRayonThreadPool(&mut mtctx, Some(pool));
    } else if thread_pool_ref != 0 {
        let pool = unsafe { &*(thread_pool_ref as *const crate::common::pool::POOL_ctx) };
        ZSTDMT_setThreadPool(&mut mtctx, Some(pool));
    }
    if !effective_prefix.is_empty() {
        mtctx.inBuff.prefix = Range {
            start: effective_prefix.as_ptr() as usize,
            size: effective_prefix.len(),
        };
    }

    let bound = ZSTD_compressBound(src.len()).max(32);
    if ERR_isError(bound) {
        let _ = ZSTDMT_freeCCtx(Some(mtctx));
        return Err(bound);
    }
    let mut compressed = vec![0u8; bound];
    let mut output_pos = 0usize;
    let mut input_pos = 0usize;
    loop {
        let rem = ZSTDMT_compressStream_generic(
            &mut mtctx,
            &mut compressed,
            &mut output_pos,
            src,
            &mut input_pos,
            ZSTD_EndDirective::ZSTD_e_end,
        );
        if ERR_isError(rem) {
            let _ = ZSTDMT_freeCCtx(Some(mtctx));
            return Err(rem);
        }
        if rem == 0 {
            break;
        }
        compressed.resize(output_pos + rem.max(32), 0);
    }
    compressed.truncate(output_pos);
    let _ = ZSTDMT_freeCCtx(Some(mtctx));
    Ok(compressed)
}

/// Port of `ZSTD_CCtx_init_compressStream2` (`zstd_compress.c:6373`).
/// Transparent initialization stage entered on the first streaming call:
/// snapshots `cctx.requestedParams`, derives effective `cParams` via the
/// resolvers (LDM / row matcher / block splitter / external repcode
/// search / max block size / sequence validation), threads them into
/// `ZSTD_compressBegin_internal`. When `endOp == ZSTD_e_end`, `inSize`
/// becomes the pledged source size; otherwise it's ignored.
pub fn ZSTD_CCtx_init_compressStream2(
    cctx: &mut ZSTD_CCtx,
    endOp: ZSTD_EndDirective,
    inSize: usize,
) -> usize {
    use crate::compress::match_state::{
        ZSTD_resolveBlockSplitterMode, ZSTD_resolveEnableLdm, ZSTD_resolveExternalRepcodeSearch,
        ZSTD_resolveExternalSequenceValidation, ZSTD_resolveMaxBlockSize,
        ZSTD_resolveRowMatchFinderMode,
    };
    use crate::decompress::zstd_ddict::ZSTD_dictContentType_e;

    let mut params = cctx.requestedParams;
    if let Some(level) = cctx.stream_level {
        params.compressionLevel = level;
    }
    if endOp == ZSTD_EndDirective::ZSTD_e_end {
        cctx.pledgedSrcSizePlusOne = inSize as u64 + 1;
        cctx.pledged_src_size = Some(inSize as u64);
    }
    let pledged = ZSTD_getPledgedSrcSize(cctx);
    let pledged_is_known = cctx.pledged_src_size.is_some();
    params.fParams = ZSTD_FrameParameters {
        contentSizeFlag: if cctx.param_contentSize && pledged_is_known {
            1
        } else {
            0
        },
        checksumFlag: if cctx.param_checksum { 1 } else { 0 },
        noDictIDFlag: if cctx.param_dictID { 0 } else { 1 },
    };
    let cdict_snapshot = cctx.stream_cdict.clone();
    let cdict = cdict_snapshot.as_ref();
    let dict_snapshot = if cdict.is_none() && !cctx.stream_dict.is_empty() {
        let dict = if cctx.stream_dict_original.is_empty() {
            cctx.stream_dict.clone()
        } else {
            cctx.stream_dict_original.clone()
        };
        Some((
            dict,
            cctx.stream_dict_content_type,
            cctx.dictContentSize,
            !cctx.prefix_is_single_use,
        ))
    } else {
        None
    };
    let dictSize = cdict.map_or_else(
        || {
            dict_snapshot
                .as_ref()
                .map(|(dict, _, content_size, _)| {
                    if *content_size != 0 {
                        *content_size
                    } else {
                        dict.len()
                    }
                })
                .unwrap_or(0)
        },
        |cd| cd.dictContent.len(),
    );
    let mode = ZSTD_getCParamMode(cdict, &params, pledged);
    params.cParams = ZSTD_getCParamsFromCCtxParams(&params, pledged, dictSize, mode);
    params.postBlockSplitter =
        ZSTD_resolveBlockSplitterMode(params.postBlockSplitter, &params.cParams);
    params.ldmParams.enableLdm = ZSTD_resolveEnableLdm(params.ldmParams.enableLdm, &params.cParams);
    params.ldmEnable = params.ldmParams.enableLdm;
    params.useRowMatchFinder =
        ZSTD_resolveRowMatchFinderMode(params.useRowMatchFinder, &params.cParams);
    params.validateSequences = ZSTD_resolveExternalSequenceValidation(params.validateSequences);
    params.maxBlockSize = ZSTD_resolveMaxBlockSize(params.maxBlockSize);
    params.searchForExternalRepcodes = ZSTD_resolveExternalRepcodeSearch(
        params.searchForExternalRepcodes,
        params.compressionLevel,
    );
    #[cfg(feature = "mt")]
    if params.extSeqProdFunc.is_some() && params.nbWorkers >= 1 {
        return ERROR(ErrorCode::ParameterCombinationUnsupported);
    }

    let (dict, dict_content_type) = dict_snapshot
        .as_ref()
        .map(|(dict, content_type, _, _)| (dict.as_slice(), *content_type))
        .unwrap_or((&[][..], ZSTD_dictContentType_e::ZSTD_dct_auto));

    let rc = ZSTD_compressBegin_internal(
        cctx,
        dict,
        dict_content_type,
        cdict,
        &params,
        pledged,
        ZSTD_buffered_policy_e::ZSTDb_buffered,
    );
    if ERR_isError(rc) {
        return rc;
    }
    if let Some((dict, content_type, content_size, persistent)) = dict_snapshot {
        if persistent {
            cctx.stream_dict = dict.clone();
            cctx.stream_dict_original = dict;
            cctx.stream_dict_content_type = content_type;
            cctx.dictContentSize = if content_size != 0 {
                content_size
            } else {
                cctx.stream_dict.len()
            };
        } else {
            cctx.stream_dict.clear();
            cctx.stream_dict_original.clear();
            cctx.stream_dict_content_type = ZSTD_dictContentType_e::ZSTD_dct_auto;
            cctx.dictID = 0;
            cctx.dictContentSize = 0;
            cctx.prefix_is_single_use = false;
        }
    }
    cctx.requestedParams = params;
    cctx.blockSizeMax = if params.maxBlockSize != 0 {
        params.maxBlockSize.min(ZSTD_BLOCKSIZE_MAX)
    } else {
        (1usize << params.cParams.windowLog).min(ZSTD_BLOCKSIZE_MAX)
    };
    if params.compressionLevel >= 8 && params.inBufferMode == ZSTD_bufferMode_e::ZSTD_bm_buffered {
        let pledged = cctx.pledgedSrcSizePlusOne.wrapping_sub(1);
        if pledged != crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN {
            let reserve = (pledged as usize).min(64 << 20);
            if cctx.stream_in_buffer.capacity() < reserve {
                cctx.stream_in_buffer
                    .reserve_exact(reserve - cctx.stream_in_buffer.capacity());
            }
        }
    }
    cctx.stream_in_to_compress = 0;
    cctx.stream_in_target = if params.inBufferMode == ZSTD_bufferMode_e::ZSTD_bm_buffered {
        let pledged = cctx.pledgedSrcSizePlusOne.wrapping_sub(1);
        cctx.blockSizeMax + ((pledged as usize == cctx.blockSizeMax) as usize)
    } else {
        0
    };
    cctx.stream_out_drained = 0;
    cctx.stream_frame_ended = false;
    cctx.stream_stage = ZSTD_cStreamStage::zcss_load;
    cctx.stream_window_base = cctx
        .ms
        .as_ref()
        .map(|ms| ms.window.base_offset)
        .unwrap_or(crate::compress::match_state::ZSTD_WINDOW_START_INDEX);
    0
}

/// Port of `ZSTD_compressSequencesAndLiterals_internal` (`zstd_compress.c:8005`).
/// Block-by-block compresses caller-supplied `(inSeqs, literals)` into `dst`.
/// Unlike the `_internal` sequence-only path, the original source is not
/// available — so the function cannot fall back to raw / RLE blocks and
/// errors with `cannotProduce_uncompressedBlock` if entropy compression
/// fails.
fn ZSTD_compressSequencesAndLiterals_internal(
    cctx: &mut ZSTD_CCtx,
    dst: &mut [u8],
    inSeqs: &[ZSTD_Sequence],
    literals: &[u8],
    srcSize: usize,
) -> usize {
    let mut remaining = srcSize;
    let mut cSize = 0usize;
    let mut op = 0usize;
    let mut seqs = inSeqs;
    let mut lits = literals;
    let repcodeResolution = cctx.appliedParams.searchForExternalRepcodes
        == crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_enable;

    if seqs.is_empty() {
        return ERROR(ErrorCode::ExternalSequencesInvalid);
    }
    if seqs.len() == 1 && seqs[0].litLength == 0 {
        if dst.len() < ZSTD_blockHeaderSize {
            return ERROR(ErrorCode::DstSizeTooSmall);
        }
        MEM_writeLE24(&mut dst[op..], 1);
        op += ZSTD_blockHeaderSize;
        cSize += ZSTD_blockHeaderSize;
    }

    while !seqs.is_empty() {
        let block = ZSTD_get1BlockSummary(seqs);
        if ERR_isError(block.nbSequences) {
            return block.nbSequences;
        }
        let lastBlock = (block.nbSequences == seqs.len()) as u32;
        if block.litSize > lits.len() {
            return ERROR(ErrorCode::ExternalSequencesInvalid);
        }
        if block.blockSize > remaining {
            return ERROR(ErrorCode::ExternalSequencesInvalid);
        }
        let block_literals = &lits[..block.litSize];
        let seqStore = cctx.seqStore.get_or_insert_with(|| {
            SeqStore_t::with_capacity(ZSTD_BLOCKSIZE_MAX / 3, ZSTD_BLOCKSIZE_MAX)
        });
        seqStore.reset();

        let conversionStatus =
            ZSTD_convertBlockSequences(cctx, &seqs[..block.nbSequences], repcodeResolution);
        if ERR_isError(conversionStatus) {
            return conversionStatus;
        }
        seqs = &seqs[block.nbSequences..];
        lits = &lits[block_literals.len()..];
        remaining -= block.blockSize;

        if dst.len() - op < ZSTD_blockHeaderSize {
            return ERROR(ErrorCode::DstSizeTooSmall);
        }
        let compressedSeqsSize = {
            let seqStore = cctx.seqStore.as_mut().unwrap();
            let saved_literals = seqStore.literals.clone();
            seqStore.literals.clear();
            seqStore.literals.extend_from_slice(block_literals);
            let disableLiteralCompression = ZSTD_literalsCompressionIsDisabled(
                cctx.appliedParams.literalCompressionMode,
                cctx.appliedParams.cParams.strategy,
                cctx.appliedParams.cParams.targetLength,
            ) as i32;
            let rc = ZSTD_entropyCompressSeqStore_internal_wksp(
                &mut dst[op + ZSTD_blockHeaderSize..],
                seqStore,
                &cctx.prevEntropy,
                &mut cctx.nextEntropy,
                cctx.appliedParams.cParams.strategy,
                disableLiteralCompression,
                0,
                &mut cctx.entropyScratch.seqCountWorkspace,
                &mut cctx.entropyScratch.seqEntropyWorkspace,
                &mut cctx.entropyScratch.seqHistWorkspace,
            );
            seqStore.literals = saved_literals;
            rc
        };
        if ERR_isError(compressedSeqsSize) {
            return compressedSeqsSize;
        }
        let compressedSeqsSize = if compressedSeqsSize > cctx.blockSizeMax {
            0
        } else {
            compressedSeqsSize
        };
        if compressedSeqsSize == 0 {
            return ERROR(ErrorCode::CannotProduceUncompressedBlock);
        }

        ZSTD_blockState_confirmRepcodesAndEntropyTables(cctx);
        if cctx.prevEntropy.fse.offcode_repeatMode == FSE_repeat::FSE_repeat_valid {
            cctx.prevEntropy.fse.offcode_repeatMode = FSE_repeat::FSE_repeat_check;
        }
        let cBlockHeader = lastBlock
            .wrapping_add(
                (crate::decompress::zstd_decompress_block::blockType_e::bt_compressed as u32) << 1,
            )
            .wrapping_add((compressedSeqsSize as u32) << 3);
        MEM_writeLE24(&mut dst[op..], cBlockHeader);
        let cBlockSize = ZSTD_blockHeaderSize + compressedSeqsSize;
        cSize += cBlockSize;
        op += cBlockSize;
        cctx.isFirstBlock = 0;

        if lastBlock != 0 {
            break;
        }
    }

    if !lits.is_empty() || remaining != 0 {
        return ERROR(ErrorCode::ExternalSequencesInvalid);
    }
    cSize
}

/// Port of `ZSTD_compressSequencesAndLiterals` (`zstd_compress.c:8112`).
/// Compresses an externally-prepared `(inSeqs, literals)` pair into a
/// complete zstd frame. Requires explicit block delimiters and disables
/// sequence validation; rejects with an error otherwise. The literals
/// capacity must cover `literals.len()`, matching the upstream C API
/// check.
pub fn ZSTD_compressSequencesAndLiterals(
    cctx: &mut ZSTD_CCtx,
    dst: &mut [u8],
    inSeqs: &[ZSTD_Sequence],
    literals: &[u8],
    litCapacity: usize,
    decompressedSize: usize,
) -> usize {
    let mut op = 0usize;
    let mut cSize = 0usize;

    if litCapacity < literals.len() {
        return ERROR(ErrorCode::WorkSpaceTooSmall);
    }
    let rc = ZSTD_CCtx_init_compressStream2(cctx, ZSTD_EndDirective::ZSTD_e_end, decompressedSize);
    if ERR_isError(rc) {
        return rc;
    }
    if cctx.appliedParams.blockDelimiters == ZSTD_SequenceFormat_e::ZSTD_sf_noBlockDelimiters {
        return ERROR(ErrorCode::FrameParameterUnsupported);
    }
    if cctx.appliedParams.validateSequences != 0 {
        return ERROR(ErrorCode::ParameterUnsupported);
    }
    if cctx.appliedParams.fParams.checksumFlag != 0 {
        return ERROR(ErrorCode::FrameParameterUnsupported);
    }

    let frameHeaderSize = ZSTD_writeFrameHeader_advanced(
        &mut dst[op..],
        &cctx.appliedParams.fParams,
        cctx.appliedParams.cParams.windowLog,
        decompressedSize as u64,
        cctx.dictID,
        cctx.appliedParams.format,
    );
    if ERR_isError(frameHeaderSize) {
        return frameHeaderSize;
    }
    op += frameHeaderSize;
    cSize += frameHeaderSize;

    let cBlocksSize = ZSTD_compressSequencesAndLiterals_internal(
        cctx,
        &mut dst[op..],
        inSeqs,
        literals,
        decompressedSize,
    );
    if ERR_isError(cBlocksSize) {
        return cBlocksSize;
    }
    cSize + cBlocksSize
}

#[cfg(test)]
mod local_streaming_static_tests {
    use super::*;

    #[test]
    fn compress_stream_returns_next_input_size_hint() {
        let mut cctx = ZSTD_CCtx::default();
        assert_eq!(ZSTD_initCStream(&mut cctx, 1), 0);

        let input = b"small streaming payload";
        let mut input_pos = 0usize;
        let mut output = [];
        let mut output_pos = 0usize;
        let hint = ZSTD_compressStream(
            &mut cctx,
            &mut output,
            &mut output_pos,
            input,
            &mut input_pos,
        );

        assert_eq!(input_pos, input.len());
        assert_eq!(output_pos, 0);
        assert_eq!(hint, ZSTD_nextInputSizeHint_MTorST(&cctx));
        assert!(!ERR_isError(hint));
    }

    #[test]
    fn init_static_cctx_rejects_exact_struct_size_workspace() {
        let cctx_size = core::mem::size_of::<ZSTD_CCtx>();
        let mut workspace = vec![0u8; cctx_size + 8];
        let base = workspace.as_mut_ptr() as usize;
        let offset = (8 - (base & 7)) & 7;
        let exact = &mut workspace[offset..offset + cctx_size];

        assert!(ZSTD_initStaticCCtx(exact).is_none());
    }

    #[test]
    fn init_static_cctx_rejects_workspace_below_static_header_reserve() {
        let cctx_size = core::mem::size_of::<ZSTD_CCtx>();
        let min_static = ZSTD_minStaticCCtxSize();
        let workspace_len = min_static - 1;
        let mut workspace = vec![0u64; workspace_len.div_ceil(core::mem::size_of::<u64>())];
        let bytes_len = workspace.len() * core::mem::size_of::<u64>();
        let bytes_all = unsafe {
            core::slice::from_raw_parts_mut(workspace.as_mut_ptr() as *mut u8, bytes_len)
        };
        let bytes = &mut bytes_all[..workspace_len];

        assert!(bytes.len() > cctx_size);
        assert!(bytes.len() < min_static);
        assert!(ZSTD_initStaticCCtx(bytes).is_none());
    }

    #[test]
    fn init_static_cctx_accepts_workspace_with_static_header_reserve() {
        let workspace_len = ZSTD_minStaticCCtxSize();
        let mut workspace = vec![0u64; workspace_len.div_ceil(core::mem::size_of::<u64>())];
        let bytes_len = workspace.len() * core::mem::size_of::<u64>();
        let bytes_all = unsafe {
            core::slice::from_raw_parts_mut(workspace.as_mut_ptr() as *mut u8, bytes_len)
        };
        let bytes = &mut bytes_all[..workspace_len];

        let cctx = ZSTD_initStaticCCtx(bytes).expect("static cctx with static reserve");
        assert_eq!(cctx.staticSize, workspace_len);
    }

    #[test]
    fn static_cctx_load_dictionary_by_copy_rejects_internal_allocation() {
        use crate::decompress::zstd_ddict::{ZSTD_dictContentType_e, ZSTD_dictLoadMethod_e};

        let workspace_len = ZSTD_minStaticCCtxSize();
        let mut workspace = vec![0u64; workspace_len.div_ceil(core::mem::size_of::<u64>())];
        let bytes_len = workspace.len() * core::mem::size_of::<u64>();
        let bytes_all = unsafe {
            core::slice::from_raw_parts_mut(workspace.as_mut_ptr() as *mut u8, bytes_len)
        };
        let bytes = &mut bytes_all[..workspace_len];
        let cctx = ZSTD_initStaticCCtx(bytes).expect("static cctx");

        let dict = b"non-empty dictionary bytes";
        let by_copy = ZSTD_CCtx_loadDictionary_advanced(
            cctx,
            dict,
            ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy,
            ZSTD_dictContentType_e::ZSTD_dct_auto,
        );
        assert_eq!(by_copy, ERROR(ErrorCode::MemoryAllocation));

        let by_ref = ZSTD_CCtx_loadDictionary_advanced(
            cctx,
            dict,
            ZSTD_dictLoadMethod_e::ZSTD_dlm_byRef,
            ZSTD_dictContentType_e::ZSTD_dct_auto,
        );
        assert_eq!(by_ref, 0);
    }

    #[test]
    fn init_static_cctx_rejects_bad_alignment() {
        let min_static = ZSTD_minStaticCCtxSize();
        let mut workspace = vec![0u8; min_static + 16];
        let base = workspace.as_mut_ptr() as usize;
        let aligned_offset = (8 - (base & 7)) & 7;
        let misaligned_offset = if aligned_offset == 0 { 1 } else { 0 };
        let misaligned = &mut workspace[misaligned_offset..misaligned_offset + min_static];

        assert_ne!((misaligned.as_mut_ptr() as usize) & 7, 0);
        assert!(ZSTD_initStaticCCtx(misaligned).is_none());
    }

    #[test]
    fn compress_sequences_and_literals_accepts_exact_literal_capacity_check() {
        let mut cctx = ZSTD_CCtx::default();
        cctx.requestedParams.blockDelimiters =
            ZSTD_SequenceFormat_e::ZSTD_sf_explicitBlockDelimiters;
        let seqs = [ZSTD_Sequence {
            offset: 0,
            litLength: 1,
            matchLength: 0,
            rep: 0,
        }];
        let literals = b"x";
        let mut dst = vec![0u8; 128];

        let rc = ZSTD_compressSequencesAndLiterals(
            &mut cctx,
            &mut dst,
            &seqs,
            literals,
            literals.len(),
            literals.len(),
        );

        assert_ne!(rc, ERROR(ErrorCode::WorkSpaceTooSmall));
    }

    #[test]
    fn compress_sequences_and_literals_internal_rejects_overlong_block_total() {
        let mut cctx = ZSTD_CCtx::default();
        let seqs = [ZSTD_Sequence {
            offset: 0,
            litLength: 2,
            matchLength: 0,
            rep: 0,
        }];
        let literals = b"xx";
        let mut dst = vec![0u8; 128];

        let rc =
            ZSTD_compressSequencesAndLiterals_internal(&mut cctx, &mut dst, &seqs, literals, 1);

        assert_eq!(rc, ERROR(ErrorCode::ExternalSequencesInvalid));
    }

    #[test]
    fn validate_sequence_matches_upstream_zero_offset_bound_edge() {
        use crate::compress::seq_store::REPCODE_TO_OFFBASE;

        assert_eq!(
            ZSTD_validateSequence(REPCODE_TO_OFFBASE(1), 3, 3, 0, 10, 0, false),
            0
        );
        assert_eq!(ZSTD_validateSequence(0, 3, 3, 0, 10, 0, false), 0);
    }

    #[test]
    fn reset_cstream_zero_pledge_maps_to_unknown() {
        use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;

        let mut cctx = ZSTD_CCtx::default();
        assert_eq!(ZSTD_initCStream(&mut cctx, 1), 0);
        assert_eq!(ZSTD_resetCStream(&mut cctx, 0), 0);

        assert_eq!(ZSTD_getPledgedSrcSize(&cctx), ZSTD_CONTENTSIZE_UNKNOWN);
        assert!(cctx.pledged_src_size.is_none());
    }

    #[test]
    fn ldm_cparams_derivation_uses_embedded_ldm_enable() {
        use crate::compress::zstd_ldm::{ZSTD_ParamSwitch_e, ZSTD_LDM_DEFAULT_WINDOW_LOG};
        use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;

        let mut params = ZSTD_CCtx_params::default();
        ZSTD_CCtxParams_init(&mut params, 1);
        params.ldmParams.enableLdm = ZSTD_ParamSwitch_e::ZSTD_ps_enable;

        let cparams = ZSTD_getCParamsFromCCtxParams(
            &params,
            ZSTD_CONTENTSIZE_UNKNOWN,
            0,
            ZSTD_CParamMode_e::ZSTD_cpm_unknown,
        );

        assert_eq!(cparams.windowLog, ZSTD_LDM_DEFAULT_WINDOW_LOG);
    }

    #[test]
    fn buffer_stability_uses_applied_params_not_future_requested_params() {
        use crate::common::error::ERR_getErrorCode;

        let mut cctx = ZSTD_CCtx::default();
        cctx.stage = ZSTD_compressionStage_e::ZSTDcs_init;
        cctx.appliedParams.inBufferMode = ZSTD_bufferMode_e::ZSTD_bm_stable;
        cctx.appliedParams.outBufferMode = ZSTD_bufferMode_e::ZSTD_bm_stable;

        let src = b"stable source bytes";
        let moved_src = b"different backing storage";
        let input = ZSTD_inBuffer {
            src: Some(src),
            size: src.len(),
            pos: 4,
        };
        let output = ZSTD_outBuffer {
            dst: None,
            size: 64,
            pos: 9,
        };
        ZSTD_setBufferExpectations(&mut cctx, &output, &input);

        cctx.requestedParams.inBufferMode = ZSTD_bufferMode_e::ZSTD_bm_buffered;
        cctx.requestedParams.outBufferMode = ZSTD_bufferMode_e::ZSTD_bm_buffered;

        let moved_input = ZSTD_inBuffer {
            src: Some(moved_src),
            size: moved_src.len(),
            pos: 4,
        };
        let rc = ZSTD_checkBufferStability(
            &cctx,
            &output,
            &moved_input,
            ZSTD_EndDirective::ZSTD_e_continue,
        );
        assert_eq!(
            ERR_getErrorCode(rc),
            ErrorCode::StabilityConditionNotRespected
        );

        let grown_output = ZSTD_outBuffer {
            dst: None,
            size: 72,
            pos: 9,
        };
        let rc = ZSTD_checkBufferStability(
            &cctx,
            &grown_output,
            &input,
            ZSTD_EndDirective::ZSTD_e_continue,
        );
        assert_eq!(
            ERR_getErrorCode(rc),
            ErrorCode::StabilityConditionNotRespected
        );
    }

    #[test]
    fn buffer_stability_ignores_requested_stable_until_applied() {
        let mut cctx = ZSTD_CCtx::default();
        cctx.stage = ZSTD_compressionStage_e::ZSTDcs_init;
        cctx.appliedParams.inBufferMode = ZSTD_bufferMode_e::ZSTD_bm_buffered;
        cctx.appliedParams.outBufferMode = ZSTD_bufferMode_e::ZSTD_bm_buffered;
        cctx.requestedParams.inBufferMode = ZSTD_bufferMode_e::ZSTD_bm_stable;
        cctx.requestedParams.outBufferMode = ZSTD_bufferMode_e::ZSTD_bm_stable;

        let input = ZSTD_inBuffer {
            src: Some(b"source one"),
            size: 10,
            pos: 2,
        };
        let output = ZSTD_outBuffer {
            dst: None,
            size: 32,
            pos: 3,
        };
        ZSTD_setBufferExpectations(&mut cctx, &output, &input);

        let moved_input = ZSTD_inBuffer {
            src: Some(b"source two"),
            size: 10,
            pos: 2,
        };
        let grown_output = ZSTD_outBuffer {
            dst: None,
            size: 64,
            pos: 3,
        };
        assert_eq!(
            ZSTD_checkBufferStability(
                &cctx,
                &grown_output,
                &moved_input,
                ZSTD_EndDirective::ZSTD_e_continue,
            ),
            0
        );
    }

    #[test]
    fn cctx_set_parameter_returns_upstream_set_values() {
        let mut cctx = ZSTD_CCtx::default();
        assert_eq!(
            ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_compressionLevel, 0),
            ZSTD_CLEVEL_DEFAULT as usize
        );
        assert_eq!(
            ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_compressionLevel, -5),
            0
        );
        assert_eq!(
            ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_stableInBuffer, 1),
            1
        );
        assert_eq!(
            ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_targetCBlockSize, 1),
            ZSTD_TARGETCBLOCKSIZE_MIN as usize
        );
        assert_eq!(
            ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_forceAttachDict, 3),
            3
        );
    }

    #[test]
    fn cctx_params_set_parameter_returns_upstream_set_values() {
        let mut params = ZSTD_CCtx_params::default();
        assert_eq!(
            ZSTD_CCtxParams_setParameter(
                &mut params,
                ZSTD_cParameter::ZSTD_c_compressionLevel,
                9999,
            ),
            ZSTD_maxCLevel() as usize
        );
        assert_eq!(
            ZSTD_CCtxParams_setParameter(&mut params, ZSTD_cParameter::ZSTD_c_checksumFlag, 1),
            1
        );
        assert_eq!(
            ZSTD_CCtxParams_setParameter(&mut params, ZSTD_cParameter::ZSTD_c_dictIDFlag, 0),
            0
        );
        assert_eq!(
            ZSTD_CCtxParams_setParameter(
                &mut params,
                ZSTD_cParameter::ZSTD_c_enableLongDistanceMatching,
                2,
            ),
            2
        );
    }

    #[cfg(not(feature = "mt"))]
    #[test]
    fn non_mt_nbworkers_setter_returns_parameter_unsupported() {
        use crate::common::error::ERR_getErrorCode;

        let mut cctx = ZSTD_CCtx::default();
        let rc = ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_nbWorkers, 1);
        assert_eq!(ERR_getErrorCode(rc), ErrorCode::ParameterUnsupported);

        let mut params = ZSTD_CCtx_params::default();
        let rc = ZSTD_CCtxParams_setParameter(&mut params, ZSTD_cParameter::ZSTD_c_nbWorkers, 1);
        assert_eq!(ERR_getErrorCode(rc), ErrorCode::ParameterUnsupported);
    }

    #[test]
    fn flush_stream_emits_staged_input_as_non_final_blocks() {
        use crate::decompress::zstd_decompress::ZSTD_decompress;

        let first = b"flush should emit this staged input ".repeat(32);
        let second = b"and end should append the final tail ".repeat(19);
        let mut expected = Vec::new();
        expected.extend_from_slice(&first);
        expected.extend_from_slice(&second);

        let mut cctx = ZSTD_CCtx::default();
        assert_eq!(ZSTD_initCStream(&mut cctx, 1), 0);

        let mut dst = vec![0u8; ZSTD_compressBound(expected.len()).max(128)];
        let mut dp = 0usize;
        let mut ip = 0usize;
        let rc = ZSTD_compressStream(&mut cctx, &mut dst, &mut dp, &first, &mut ip);
        assert!(!ERR_isError(rc), "compressStream errored: {rc:#x}");
        assert_eq!(ip, first.len());
        assert_eq!(dp, 0);

        let rc = ZSTD_flushStream(&mut cctx, &mut dst, &mut dp);
        assert!(!ERR_isError(rc), "flushStream errored: {rc:#x}");
        assert_eq!(rc, 0);
        assert!(dp > 0, "flushStream must emit staged frame bytes");

        let mut ip2 = 0usize;
        let rc = ZSTD_compressStream(&mut cctx, &mut dst, &mut dp, &second, &mut ip2);
        assert!(!ERR_isError(rc), "second compressStream errored: {rc:#x}");
        assert_eq!(ip2, second.len());
        loop {
            let rc = ZSTD_endStream(&mut cctx, &mut dst, &mut dp);
            assert!(!ERR_isError(rc), "endStream errored: {rc:#x}");
            if rc == 0 {
                break;
            }
            if dp == dst.len() {
                dst.resize(dst.len() * 2, 0);
            }
        }

        let mut decoded = vec![0u8; expected.len()];
        let d = ZSTD_decompress(&mut decoded, &dst[..dp]);
        assert_eq!(d, expected.len());
        assert_eq!(decoded, expected);
    }

    #[test]
    fn unpledged_multi_call_stream_omits_frame_content_size() {
        use crate::decompress::zstd_decompress::{
            ZSTD_decompress, ZSTD_getFrameContentSize, ZSTD_CONTENTSIZE_UNKNOWN,
        };

        let first = b"unpledged first chunk ".repeat(17);
        let second = b"unpledged second chunk ".repeat(23);
        let mut expected = Vec::new();
        expected.extend_from_slice(&first);
        expected.extend_from_slice(&second);

        let mut cctx = ZSTD_CCtx::default();
        assert_eq!(ZSTD_initCStream(&mut cctx, 1), 0);

        let mut dst = vec![0u8; ZSTD_compressBound(expected.len()).max(128)];
        let mut dp = 0usize;
        let mut ip = 0usize;
        let rc = ZSTD_compressStream(&mut cctx, &mut dst, &mut dp, &first, &mut ip);
        assert!(!ERR_isError(rc), "first compressStream errored: {rc:#x}");
        let mut ip2 = 0usize;
        let rc = ZSTD_compressStream(&mut cctx, &mut dst, &mut dp, &second, &mut ip2);
        assert!(!ERR_isError(rc), "second compressStream errored: {rc:#x}");
        loop {
            let rc = ZSTD_endStream(&mut cctx, &mut dst, &mut dp);
            assert!(!ERR_isError(rc), "endStream errored: {rc:#x}");
            if rc == 0 {
                break;
            }
            if dp == dst.len() {
                dst.resize(dst.len() * 2, 0);
            }
        }

        assert_eq!(
            ZSTD_getFrameContentSize(&dst[..dp]),
            ZSTD_CONTENTSIZE_UNKNOWN
        );
        let mut decoded = vec![0u8; expected.len()];
        let d = ZSTD_decompress(&mut decoded, &dst[..dp]);
        assert_eq!(d, expected.len());
        assert_eq!(decoded, expected);
    }

    #[cfg(feature = "mt")]
    #[test]
    fn endstream_mt_helper_reuses_source_while_draining_sections() {
        use crate::decompress::zstd_decompress::{
            ZSTD_decompress, ZSTD_format_e::ZSTD_f_zstd1, ZSTD_getFrameContentSize,
            ZSTD_CONTENTSIZE_UNKNOWN,
        };

        let mut params = ZSTD_CCtx_params::default();
        ZSTD_CCtxParams_init(&mut params, 3);
        params.nbWorkers = 1;

        let src_len = crate::compress::zstdmt_compress::ZSTDMT_JOBSIZE_MIN + 17;
        let mut src = Vec::with_capacity(src_len);
        let pattern = b"mt endstream helper source block ";
        while src.len() < src_len {
            src.extend_from_slice(pattern);
        }
        src.truncate(src_len);

        let compressed = zstd_endstream_mt_compress(
            params,
            ZSTD_f_zstd1,
            0,
            0,
            3,
            ZSTD_CONTENTSIZE_UNKNOWN,
            &src,
            &[],
            ZSTD_FrameParameters {
                contentSizeFlag: 0,
                checksumFlag: 0,
                noDictIDFlag: 0,
            },
        )
        .expect("unpledged MT endstream helper should drain to completion");

        assert!(!compressed.is_empty());
        assert_eq!(
            ZSTD_getFrameContentSize(&compressed),
            ZSTD_CONTENTSIZE_UNKNOWN
        );

        let pledged_compressed = zstd_endstream_mt_compress(
            params,
            ZSTD_f_zstd1,
            0,
            0,
            3,
            src.len() as u64,
            &src,
            &[],
            ZSTD_FrameParameters {
                contentSizeFlag: 1,
                checksumFlag: 0,
                noDictIDFlag: 0,
            },
        )
        .expect("pledged MT endstream helper should drain to completion");

        assert!(!pledged_compressed.is_empty());
        assert_eq!(
            ZSTD_getFrameContentSize(&pledged_compressed),
            src.len() as u64
        );
        let mut pledged_decoded = vec![0u8; src.len() + ZSTD_BLOCKSIZE_MAX];
        let pledged_decoded_size = ZSTD_decompress(&mut pledged_decoded, &pledged_compressed);
        assert_eq!(pledged_decoded_size, src.len());
        assert_eq!(&pledged_decoded[..pledged_decoded_size], src.as_slice());
    }
}

#[cfg(test)]
#[path = "zstd_compress_tests.rs"]
mod tests;
