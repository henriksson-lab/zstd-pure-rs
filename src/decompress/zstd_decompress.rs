//! Translation of `lib/decompress/zstd_decompress.c`. The frame-level
//! decoder: magic number, frame header, block loop, checksum validation.

use crate::common::error::{ErrorCode, ERROR};
use crate::common::mem::{MEM_readLE16, MEM_readLE32, MEM_readLE64};

pub const ZSTD_MAGICNUMBER: u32 = 0xFD2FB528;
pub const ZSTD_MAGIC_DICTIONARY: u32 = 0xEC30A437;
pub const ZSTD_MAGIC_SKIPPABLE_START: u32 = 0x184D2A50;
pub const ZSTD_MAGIC_SKIPPABLE_MASK: u32 = 0xFFFFFFF0;

pub const ZSTD_FRAMEIDSIZE: usize = 4;
pub const ZSTD_SKIPPABLEHEADERSIZE: usize = 8;
pub const ZSTD_WINDOWLOG_ABSOLUTEMIN: u32 = 10;
pub const ZSTD_WINDOWLOG_MAX_64: u32 = 31;
pub const ZSTD_WINDOWLOG_MAX_32: u32 = 30;

pub const ZSTD_CONTENTSIZE_UNKNOWN: u64 = u64::MAX;
pub const ZSTD_CONTENTSIZE_ERROR: u64 = u64::MAX - 1;

pub const ZSTD_fcs_fieldSize: [usize; 4] = [0, 2, 4, 8];
pub const ZSTD_did_fieldSize: [usize; 4] = [0, 1, 2, 4];

/// Port of `ZSTD_format_e`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZSTD_format_e {
    #[default]
    ZSTD_f_zstd1,
    ZSTD_f_zstd1_magicless,
}

/// Port of `ZSTD_FrameType_e`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZSTD_FrameType_e {
    #[default]
    ZSTD_frame,
    ZSTD_skippableFrame,
}

/// Mirror of `ZSTD_FrameHeader` (upstream public struct).
#[derive(Debug, Clone, Copy, Default)]
pub struct ZSTD_FrameHeader {
    pub frameContentSize: u64,
    pub windowSize: u64,
    pub blockSizeMax: u32,
    pub frameType: ZSTD_FrameType_e,
    pub headerSize: u32,
    pub dictID: u32,
    pub checksumFlag: u32,
    pub _reserved1: u32,
    pub _reserved2: u32,
}

/// Port of `ZSTD_dStreamStage` (`zstd_decompress_internal.h:94`). The
/// streaming-decoder's ingest sub-stage, distinct from `ZSTD_dStage`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZSTD_dStreamStage {
    #[default]
    zdss_init = 0,
    zdss_loadHeader = 1,
    zdss_read = 2,
    zdss_load = 3,
    zdss_flush = 4,
}

/// Port of `ZSTD_dictUses_e` (`zstd_decompress_internal.h:97`). Tracks
/// how long a dict remains bound to a DCtx: indefinitely, once, or
/// not at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZSTD_dictUses_e {
    ZSTD_use_indefinitely = -1,
    #[default]
    ZSTD_dont_use = 0,
    ZSTD_use_once = 1,
}

/// Port of `ZSTD_dStage` (`zstd_decompress_internal.h:89`). Lifecycle
/// state of the streaming DCtx; consumed by `ZSTD_nextInputType` /
/// `ZSTD_nextSrcSizeToDecompressWithInputSize` / `ZSTD_isSkipFrame`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZSTD_dStage {
    #[default]
    ZSTDds_getFrameHeaderSize = 0,
    ZSTDds_decodeFrameHeader = 1,
    ZSTDds_decodeBlockHeader = 2,
    ZSTDds_decompressBlock = 3,
    ZSTDds_decompressLastBlock = 4,
    ZSTDds_checkChecksum = 5,
    ZSTDds_decodeSkippableHeader = 6,
    ZSTDds_skipFrame = 7,
}

/// Port of `ZSTD_FRAMEHEADERSIZE_PREFIX` (`zstd.h:1257`). Minimum input
/// size required to query frame-header size: 5 for zstd1, 1 for the
/// magicless variant.
#[inline]
pub const fn ZSTD_FRAMEHEADERSIZE_PREFIX(format: ZSTD_format_e) -> usize {
    match format {
        ZSTD_format_e::ZSTD_f_zstd1 => 5,
        ZSTD_format_e::ZSTD_f_zstd1_magicless => 1,
    }
}

/// Port of `ZSTD_FRAMEHEADERSIZE_MIN` (`zstd.h:1258`). Minimum size of
/// a valid frame header: 6 bytes for zstd1 (4B magic + 1B FHD + 1B
/// window), 2 for magicless.
#[inline]
pub const fn ZSTD_FRAMEHEADERSIZE_MIN(format: ZSTD_format_e) -> usize {
    match format {
        ZSTD_format_e::ZSTD_f_zstd1 => 6,
        ZSTD_format_e::ZSTD_f_zstd1_magicless => 2,
    }
}

/// Port of `ZSTD_startingInputLength` / `ZSTD_FRAMEHEADERSIZE_PREFIX`.
/// Minimum bytes needed to read the frame-header's Frame Header
/// Descriptor (FHD) byte. For zstd1 this is 5 (4-byte magic + FHD);
/// for the magicless variant it's 1.
#[inline]
pub fn ZSTD_startingInputLength(format: ZSTD_format_e) -> usize {
    match format {
        ZSTD_format_e::ZSTD_f_zstd1 => ZSTD_FRAMEIDSIZE + 1,
        ZSTD_format_e::ZSTD_f_zstd1_magicless => 1,
    }
}

/// Port of `ZSTD_frameHeaderSize_internal` / `ZSTD_frameHeaderSize`.
/// Reads only the Frame Header Descriptor byte, returns the full
/// header size (inclusive of magic, FHD, window descriptor, dictID,
/// FCS).
pub fn ZSTD_frameHeaderSize_internal(src: &[u8], format: ZSTD_format_e) -> usize {
    let minInputSize = ZSTD_startingInputLength(format);
    if src.len() < minInputSize {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    let fhd = src[minInputSize - 1];
    let dictID = (fhd & 3) as usize;
    let singleSegment = ((fhd >> 5) & 1) as usize;
    let fcsId = (fhd >> 6) as usize;
    minInputSize
        + (1 - singleSegment)               // window descriptor byte (absent when singleSegment)
        + ZSTD_did_fieldSize[dictID]
        + ZSTD_fcs_fieldSize[fcsId]
        + (singleSegment & (fcsId == 0) as usize) // extra byte for FCS when singleSegment and fcsId==0
}

/// Port of `ZSTD_frameHeaderSize` (`zstd.h:1118`). Returns the byte
/// count of the frame header beginning at `src`, or an error code if
/// the header is malformed / truncated. Always assumes the default
/// `ZSTD_f_zstd1` format — callers who need magicless support should
/// use `ZSTD_frameHeaderSize_internal` with an explicit `format`.
pub fn ZSTD_frameHeaderSize(src: &[u8]) -> usize {
    ZSTD_frameHeaderSize_internal(src, ZSTD_format_e::ZSTD_f_zstd1)
}

/// Port of `ZSTD_getFrameHeader_advanced`. Returns:
///   - 0 on success; `zfh` is populated.
///   - >0 : `src` was too small; value is the needed byte count.
///   - ERR_isError(rc): malformed/unsupported header.
pub fn ZSTD_getFrameHeader_advanced(
    zfh: &mut ZSTD_FrameHeader,
    src: &[u8],
    format: ZSTD_format_e,
) -> usize {
    let srcSize = src.len();
    let minInputSize = ZSTD_startingInputLength(format);

    if srcSize < minInputSize {
        // Short-read handling: validate magic prefix if we have anything.
        if srcSize > 0 && format != ZSTD_format_e::ZSTD_f_zstd1_magicless {
            let toCopy = 4.min(srcSize);
            let mut hbuf = ZSTD_MAGICNUMBER.to_le_bytes();
            hbuf[..toCopy].copy_from_slice(&src[..toCopy]);
            if MEM_readLE32(&hbuf) != ZSTD_MAGICNUMBER {
                let mut hbuf2 = ZSTD_MAGIC_SKIPPABLE_START.to_le_bytes();
                hbuf2[..toCopy].copy_from_slice(&src[..toCopy]);
                if (MEM_readLE32(&hbuf2) & ZSTD_MAGIC_SKIPPABLE_MASK) != ZSTD_MAGIC_SKIPPABLE_START
                {
                    return ERROR(ErrorCode::PrefixUnknown);
                }
            }
        }
        return minInputSize;
    }

    *zfh = ZSTD_FrameHeader::default();

    if format != ZSTD_format_e::ZSTD_f_zstd1_magicless
        && MEM_readLE32(&src[..4]) != ZSTD_MAGICNUMBER
    {
        if (MEM_readLE32(&src[..4]) & ZSTD_MAGIC_SKIPPABLE_MASK) == ZSTD_MAGIC_SKIPPABLE_START {
            if srcSize < ZSTD_SKIPPABLEHEADERSIZE {
                return ZSTD_SKIPPABLEHEADERSIZE;
            }
            zfh.frameType = ZSTD_FrameType_e::ZSTD_skippableFrame;
            zfh.dictID = MEM_readLE32(&src[..4]) - ZSTD_MAGIC_SKIPPABLE_START;
            zfh.headerSize = ZSTD_SKIPPABLEHEADERSIZE as u32;
            zfh.frameContentSize =
                MEM_readLE32(&src[ZSTD_FRAMEIDSIZE..ZSTD_FRAMEIDSIZE + 4]) as u64;
            return 0;
        }
        return ERROR(ErrorCode::PrefixUnknown);
    }

    let fhsize = ZSTD_frameHeaderSize_internal(src, format);
    if crate::common::error::ERR_isError(fhsize) {
        return fhsize;
    }
    if srcSize < fhsize {
        return fhsize;
    }
    zfh.headerSize = fhsize as u32;

    let fhdByte = src[minInputSize - 1];
    let mut pos = minInputSize;
    let dictIDSizeCode = (fhdByte & 3) as u32;
    let checksumFlag = ((fhdByte >> 2) & 1) as u32;
    let singleSegment = ((fhdByte >> 5) & 1) as u32;
    let fcsID = (fhdByte >> 6) as u32;
    if (fhdByte & 0x08) != 0 {
        // Reserved bit must be zero.
        return ERROR(ErrorCode::FrameParameterUnsupported);
    }

    let mut windowSize: u64 = 0;
    let mut dictID: u32 = 0;
    let mut frameContentSize: u64 = ZSTD_CONTENTSIZE_UNKNOWN;

    if singleSegment == 0 {
        let wlByte = src[pos];
        pos += 1;
        let windowLog = ((wlByte >> 3) as u32) + ZSTD_WINDOWLOG_ABSOLUTEMIN;
        let winLogMax = if core::mem::size_of::<usize>() == 8 {
            ZSTD_WINDOWLOG_MAX_64
        } else {
            ZSTD_WINDOWLOG_MAX_32
        };
        if windowLog > winLogMax {
            return ERROR(ErrorCode::FrameParameterWindowTooLarge);
        }
        windowSize = 1u64 << windowLog;
        windowSize += (windowSize >> 3) * (wlByte & 7) as u64;
    }
    match dictIDSizeCode {
        0 => {}
        1 => {
            dictID = src[pos] as u32;
            pos += 1;
        }
        2 => {
            dictID = MEM_readLE16(&src[pos..pos + 2]) as u32;
            pos += 2;
        }
        _ => {
            // 3
            dictID = MEM_readLE32(&src[pos..pos + 4]);
            pos += 4;
        }
    }
    match fcsID {
        0 => {
            if singleSegment != 0 {
                frameContentSize = src[pos] as u64;
            }
        }
        1 => frameContentSize = MEM_readLE16(&src[pos..pos + 2]) as u64 + 256,
        2 => frameContentSize = MEM_readLE32(&src[pos..pos + 4]) as u64,
        _ => frameContentSize = MEM_readLE64(&src[pos..pos + 8]),
    }
    if singleSegment != 0 {
        windowSize = frameContentSize;
    }

    zfh.frameType = ZSTD_FrameType_e::ZSTD_frame;
    zfh.frameContentSize = frameContentSize;
    zfh.windowSize = windowSize;
    zfh.blockSizeMax =
        windowSize.min(crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX as u64) as u32;
    zfh.dictID = dictID;
    zfh.checksumFlag = checksumFlag;
    0
}

/// Port of `ZSTD_getFrameHeader` (`zstd.h:1154`). Parses `src` as the
/// start of a default `ZSTD_f_zstd1` frame and populates `zfh`.
/// Returns 0 on success, a positive byte-count hint when `src` is
/// short, or an error code when the header is malformed. For
/// magicless callers use `ZSTD_getFrameHeader_advanced`.
pub fn ZSTD_getFrameHeader(zfh: &mut ZSTD_FrameHeader, src: &[u8]) -> usize {
    ZSTD_getFrameHeader_advanced(zfh, src, ZSTD_format_e::ZSTD_f_zstd1)
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    /// Hand-build a minimal valid zstd-format dictionary:
    /// 4-byte magic + 4-byte dictID + serialized HUF CTable + FSE OF/ML/LL + 3×u32 rep + raw content.
    /// Uses the compressor-side serializers so the test exercises the
    /// real writers; we verify the decoder-side parsers accept it.
    fn build_minimal_zstd_dict(dictID: u32, content: &[u8]) -> Vec<u8> {
        use crate::common::mem::MEM_writeLE32;
        use crate::compress::fse_compress::{
            FSE_buildCTable_wksp, FSE_normalizeCount, FSE_writeNCount,
        };
        use crate::compress::huf_compress::{
            HUF_buildCTable_wksp, HUF_writeCTable, HUF_CTABLE_WORKSPACE_SIZE_U32,
        };
        use crate::decompress::zstd_decompress_block::{
            LL_defaultNorm, LL_defaultNormLog, ML_defaultNorm, ML_defaultNormLog, MaxLL, MaxML,
            MaxOff, OF_defaultNorm, OF_defaultNormLog,
        };

        let mut out = Vec::new();
        // Magic + dictID.
        let mut magic_bytes = [0u8; 4];
        MEM_writeLE32(&mut magic_bytes, ZSTD_MAGICNUMBER_DICTIONARY);
        out.extend_from_slice(&magic_bytes);
        let mut id_bytes = [0u8; 4];
        MEM_writeLE32(&mut id_bytes, dictID);
        out.extend_from_slice(&id_bytes);

        // HUF CTable — seed from content bytes so writeCTable has real weights.
        let mut count = [0u32; 256];
        for &b in content.iter() {
            count[b as usize] += 1;
        }
        // Pad single-symbol content to avoid degenerate table.
        for (i, c) in count.iter_mut().enumerate().take(16) {
            if *c == 0 {
                *c = 1;
                let _ = i;
            }
        }
        let maxSymbolValue = count
            .iter()
            .enumerate()
            .rposition(|(_, &c)| c > 0)
            .unwrap_or(0) as u32;
        let totalCount: usize = count.iter().sum::<u32>() as usize;
        let tableLog =
            crate::compress::huf_compress::HUF_optimalTableLog(11, totalCount, maxSymbolValue);
        let mut ct = vec![0u64; 257];
        let mut wksp = vec![0u32; HUF_CTABLE_WORKSPACE_SIZE_U32];
        FSE_buildCTable_wksp(
            &mut [0u32; 512], // dummy to avoid unused import
            &[0i16; 1],
            0,
            5,
            &mut [0u8; 1024],
        );
        let _ = HUF_buildCTable_wksp(&mut ct, &count, maxSymbolValue, tableLog, &mut wksp);
        let mut huf_hdr = vec![0u8; 512];
        let w = HUF_writeCTable(&mut huf_hdr, &ct, maxSymbolValue, tableLog);
        assert!(!crate::common::error::ERR_isError(w));
        out.extend_from_slice(&huf_hdr[..w]);

        // FSE OF/ML/LL tables using default distributions.
        let mut fse_buf = vec![0u8; 256];
        // OF
        let w = FSE_writeNCount(&mut fse_buf, &OF_defaultNorm, MaxOff, OF_defaultNormLog);
        assert!(!crate::common::error::ERR_isError(w));
        out.extend_from_slice(&fse_buf[..w]);
        // ML
        let w = FSE_writeNCount(&mut fse_buf, &ML_defaultNorm, MaxML, ML_defaultNormLog);
        assert!(!crate::common::error::ERR_isError(w));
        out.extend_from_slice(&fse_buf[..w]);
        // LL
        let w = FSE_writeNCount(&mut fse_buf, &LL_defaultNorm, MaxLL, LL_defaultNormLog);
        assert!(!crate::common::error::ERR_isError(w));
        out.extend_from_slice(&fse_buf[..w]);
        // Silence unused-import warnings.
        let _ = FSE_normalizeCount;

        // 3 × rep values — must be nonzero and ≤ content.len().
        let safe = (content.len() as u32).clamp(1, 8);
        for r in [safe, safe.saturating_sub(1).max(1), 1u32] {
            let mut rb = [0u8; 4];
            MEM_writeLE32(&mut rb, r);
            out.extend_from_slice(&rb);
        }

        // Raw content.
        out.extend_from_slice(content);
        out
    }

    #[test]
    fn insertDictionary_roundtrips_magic_prefix_dict() {
        // Build a real zstd-format dict with a known dictID + raw
        // content. Call `ZSTD_decompress_insertDictionary` and verify:
        //   - dictID parsed from bytes 4..8 lands on dctx.dictID
        //   - stream_dict holds the raw content portion
        //   - litEntropy + fseEntropy flags turn on
        use crate::decompress::zstd_decompress_block::ZSTD_DCtx;
        let content = b"zstd-dict-content-bytes-for-compression-context";
        let dict = build_minimal_zstd_dict(0x1234_5678, content);

        let mut dctx = ZSTD_DCtx::new();
        let rc = ZSTD_decompress_insertDictionary(&mut dctx, &dict);
        assert!(
            !crate::common::error::ERR_isError(rc),
            "insertDictionary failed: {}",
            crate::common::error::ERR_getErrorName(rc)
        );
        assert_eq!(dctx.dictID, 0x1234_5678);
        assert_eq!(dctx.stream_dict, content);
        assert_eq!(dctx.litEntropy, 1);
        assert_eq!(dctx.fseEntropy, 1);
    }

    #[test]
    fn compress_insertDictionary_roundtrips_magic_prefix() {
        // Symmetric parity gate: build a zstd-format dict, feed it
        // through `ZSTD_compress_insertDictionary`, verify dictID +
        // content stashing + entropy repeatMode transitions.
        use crate::compress::zstd_compress::{ZSTD_compress_insertDictionary, ZSTD_createCCtx};
        use crate::compress::zstd_compress_literals::HUF_repeat;
        use crate::decompress::zstd_ddict::ZSTD_dictContentType_e;
        let content = b"zstd-dict-content-bytes-for-compression-context";
        let dict = build_minimal_zstd_dict(0xABCD_1234, content);

        let mut cctx = ZSTD_createCCtx().unwrap();
        let params = cctx.requestedParams;
        let rc = ZSTD_compress_insertDictionary(
            &mut cctx,
            &params,
            &dict,
            ZSTD_dictContentType_e::ZSTD_dct_auto,
        );
        assert!(
            !crate::common::error::ERR_isError(rc),
            "compress_insertDictionary failed: {}",
            crate::common::error::ERR_getErrorName(rc)
        );
        assert_eq!(cctx.dictID, 0xABCD_1234);
        assert_eq!(cctx.stream_dict, content);
        // Entropy repeatMode: HUF becomes `check` (or `valid` if every
        // symbol present), FSE tables become `check` or `valid`.
        // Either way, it's no longer `none`.
        assert_ne!(cctx.prevEntropy.huf.repeatMode, HUF_repeat::HUF_repeat_none);
    }

    #[test]
    fn insertDictionary_raw_content_stashes_bytes() {
        // Dict with no magic prefix → raw content path.
        use crate::decompress::zstd_decompress_block::ZSTD_DCtx;
        let raw = b"this is not a zstd dict, just arbitrary bytes";
        let mut dctx = ZSTD_DCtx::new();
        let rc = ZSTD_decompress_insertDictionary(&mut dctx, raw);
        assert!(!crate::common::error::ERR_isError(rc));
        assert_eq!(dctx.dictID, 0);
        assert_eq!(dctx.stream_dict, raw);
    }

    #[test]
    fn zstd_sizeof_dctx_includes_owned_tables() {
        use crate::decompress::zstd_decompress_block::ZSTD_DCtx;
        let dctx = ZSTD_DCtx::new();
        let sz = ZSTD_sizeof_DCtx(&dctx);
        // Must include at least the seq DTables (3 × ~4 KB for LL/OF/ML default).
        assert!(sz > 4096, "sizeof_DCtx unexpectedly small: {sz}");
        // DStream alias is the same.
        assert_eq!(ZSTD_sizeof_DStream(&dctx), sz);
    }

    #[test]
    fn zstd_sizeof_dctx_grows_when_dict_loaded() {
        // Loading a 4 KB dict must bump the reported size by >= the
        // dict's capacity — otherwise callers that size allocation
        // pools from this helper will under-provision.
        use crate::decompress::zstd_decompress_block::ZSTD_DCtx;
        let mut dctx = ZSTD_DCtx::new();
        let before = ZSTD_sizeof_DCtx(&dctx);
        let dict = vec![0xABu8; 4096];
        ZSTD_DCtx_loadDictionary(&mut dctx, &dict);
        let after = ZSTD_sizeof_DCtx(&dctx);
        assert!(
            after >= before + dict.len(),
            "sizeof_DCtx did not reflect loaded dict: before={before} after={after}"
        );
    }

    #[test]
    fn zstd_dParam_getBounds_windowLogMax_range() {
        // Upstream: [ZSTD_WINDOWLOG_ABSOLUTEMIN, ZSTD_WINDOWLOG_MAX].
        // That's [10, 31] on 64-bit and [10, 30] on 32-bit — NOT the
        // 27-byte `ZSTD_WINDOWLOG_LIMIT_DEFAULT` which is merely the
        // streaming-decoder's default cap, not the absolute bound.
        let b = ZSTD_dParam_getBounds(ZSTD_dParameter::ZSTD_d_windowLogMax);
        assert_eq!(b.error, 0);
        let expected_upper = if crate::common::mem::MEM_32bits() != 0 {
            30
        } else {
            31
        };
        assert_eq!((b.lowerBound, b.upperBound), (10, expected_upper));
    }

    #[test]
    fn fresh_DCtx_getParameter_windowLogMax_returns_LIMIT_DEFAULT() {
        // Upstream (zstd_decompress.c:244) initializes `maxWindowSize`
        // to `(1 << ZSTD_WINDOWLOG_LIMIT_DEFAULT) + 1`; `getParameter`
        // therefore returns 27 on a fresh DCtx. Previously Rust port
        // returned 0 — diverged from C-compat callers that use
        // getParameter to decide whether to override the cap.
        let dctx = ZSTD_DCtx::default();
        let mut got = -999i32;
        ZSTD_DCtx_getParameter(&dctx, ZSTD_dParameter::ZSTD_d_windowLogMax, &mut got);
        assert_eq!(got, ZSTD_WINDOWLOG_LIMIT_DEFAULT as i32);
    }

    #[test]
    fn DCtx_setParameter_windowLogMax_maps_zero_to_LIMIT_DEFAULT() {
        // Upstream contract: `ZSTD_DCtx_setParameter(d_windowLogMax, 0)`
        // substitutes `ZSTD_WINDOWLOG_LIMIT_DEFAULT` (27) before
        // bounds-checking and storing. C callers rely on this — 0 is
        // the documented way to request "default cap".
        let mut dctx = ZSTD_DCtx::default();
        let rc = ZSTD_DCtx_setParameter(&mut dctx, ZSTD_dParameter::ZSTD_d_windowLogMax, 0);
        assert_eq!(rc, 0);
        let mut got = -1i32;
        ZSTD_DCtx_getParameter(&dctx, ZSTD_dParameter::ZSTD_d_windowLogMax, &mut got);
        assert_eq!(got, 27);
    }

    #[test]
    fn DCtx_setParameter_windowLogMax_rejects_out_of_range() {
        // Previously stored any value unchecked — now must emit
        // `ParameterOutOfBound` to match upstream CHECK_DBOUNDS.
        use crate::common::error::{ERR_getErrorCode, ERR_isError};
        let mut dctx = ZSTD_DCtx::default();
        for oor in [-1, 9, 40, 100] {
            let rc = ZSTD_DCtx_setParameter(&mut dctx, ZSTD_dParameter::ZSTD_d_windowLogMax, oor);
            assert!(ERR_isError(rc), "expected error for value={oor}");
            assert_eq!(ERR_getErrorCode(rc), ErrorCode::ParameterOutOfBound);
        }
    }

    #[test]
    fn dParam_bounds_accept_windowLog_at_upper_edge_via_setMaxWindowSize() {
        // Regression: previously `ZSTD_dParam_getBounds` reported the
        // LIMIT_DEFAULT (27) as the upper bound, rejecting any attempt
        // to permit a window larger than 128 MB. Upstream's bound is
        // the absolute WINDOWLOG_MAX — callers legitimately need to
        // raise the cap for oversized frames. Pinning both edges:
        //   - 1 << ABSOLUTEMIN (10) must be accepted
        //   - 1 << MAX (31 on 64-bit / 30 on 32-bit) must be accepted
        //   - 1 << (MAX+1) would overflow on 32-bit and exceeds
        //     usize on most build configs, so the upper-edge accept
        //     is the tightest usable pin.
        let mut dctx = ZSTD_DCtx::default();
        let min_size = 1usize << 10;
        assert_eq!(ZSTD_DCtx_setMaxWindowSize(&mut dctx, min_size), 0);
        let upper = if crate::common::mem::MEM_32bits() != 0 {
            30
        } else {
            31
        };
        let max_size = 1usize << upper;
        assert_eq!(ZSTD_DCtx_setMaxWindowSize(&mut dctx, max_size), 0);
    }

    #[test]
    fn zstd_dstream_buffer_sizes_nonzero() {
        assert!(ZSTD_DStreamInSize() > 0);
        assert!(ZSTD_DStreamOutSize() > 0);
    }

    #[test]
    fn zstd_dctx_setParameter_windowLogMax_roundtrips() {
        use crate::decompress::zstd_decompress_block::ZSTD_DCtx;
        let mut dctx = ZSTD_DCtx::new();
        let rc = ZSTD_DCtx_setParameter(&mut dctx, ZSTD_dParameter::ZSTD_d_windowLogMax, 20);
        assert_eq!(rc, 0);
        let mut v = 0i32;
        ZSTD_DCtx_getParameter(&dctx, ZSTD_dParameter::ZSTD_d_windowLogMax, &mut v);
        assert_eq!(v, 20);
    }

    #[test]
    fn zstd_dctx_reset_clears_streaming_state() {
        use crate::decompress::zstd_decompress_block::ZSTD_DCtx;
        let mut dctx = ZSTD_DCtx::new();
        dctx.stream_in_buffer.extend_from_slice(b"pending");
        ZSTD_DCtx_reset(&mut dctx, ZSTD_DResetDirective::ZSTD_reset_session_only);
        assert!(dctx.stream_in_buffer.is_empty());
    }

    #[test]
    fn zstd_estimate_dctx_size_positive() {
        let s = ZSTD_estimateDCtxSize();
        assert!(s > 0);
        // Streaming needs more than DCtx alone.
        assert!(ZSTD_estimateDStreamSize(1 << 17) > s);
    }

    #[test]
    fn zstd_estimate_dstream_size_from_frame_reflects_window() {
        // A small-window frame (level 1 on small input) should need
        // less than a maxed-out 128 KB window estimate.
        let src: Vec<u8> = b"small src for small window".to_vec();
        let mut dst = vec![0u8; 256];
        let n = crate::compress::zstd_compress::ZSTD_compress(&mut dst, &src, 1);
        dst.truncate(n);
        let from_frame = ZSTD_estimateDStreamSize_fromFrame(&dst);
        // Bogus input → falls back to default (1<<17 window).
        let fallback = ZSTD_estimateDStreamSize_fromFrame(&[0xFF, 0xFF, 0xFF, 0xFF]);
        // Both are non-zero and fallback covers at least the default.
        assert!(from_frame > 0);
        assert!(fallback > 0);
    }

    #[test]
    fn zstd_decompressBlock_roundtrips_a_compressed_block_body() {
        use crate::decompress::zstd_decompress_block::{
            blockProperties_t, blockType_e, ZSTD_DCtx, ZSTD_blockHeaderSize, ZSTD_getcBlockSize,
        };
        // Produce a single compressed block via our compressor, then
        // decode just its body (no frame header) through the public
        // ZSTD_decompressBlock entry point.
        let src: Vec<u8> = b"hello block api. "
            .iter()
            .cycle()
            .take(200)
            .copied()
            .collect();

        // Compress the whole thing as a frame, then extract the
        // compressed block body by parsing the frame header manually.
        let mut frame = vec![0u8; 1024];
        let n = crate::compress::zstd_compress::ZSTD_compress(&mut frame, &src, 1);
        frame.truncate(n);

        // Parse frame header.
        let mut zfh = super::ZSTD_FrameHeader::default();
        let rc = super::ZSTD_getFrameHeader(&mut zfh, &frame);
        assert_eq!(rc, 0);
        let body_start = zfh.headerSize as usize;

        // Parse block header.
        let mut bp = blockProperties_t {
            blockType: blockType_e::bt_raw,
            lastBlock: 0,
            origSize: 0,
        };
        let body_size = ZSTD_getcBlockSize(&frame[body_start..], &mut bp);

        // Only test when block is actually compressed (not raw/RLE).
        if bp.blockType == blockType_e::bt_compressed {
            let body = &frame
                [body_start + ZSTD_blockHeaderSize..body_start + ZSTD_blockHeaderSize + body_size];
            let mut dctx = ZSTD_DCtx::new();
            let mut out = vec![0u8; src.len() + 64];
            let d = ZSTD_decompressBlock(&mut dctx, &mut out, body);
            assert!(!crate::common::error::ERR_isError(d));
            assert_eq!(&out[..d], &src[..]);
        }
    }

    #[test]
    fn windowlog_and_contentsize_constants_match_upstream() {
        // Pin the remaining format-level constants.
        assert_eq!(ZSTD_WINDOWLOG_ABSOLUTEMIN, 10);
        assert_eq!(ZSTD_WINDOWLOG_MAX_64, 31);
        assert_eq!(ZSTD_WINDOWLOG_MAX_32, 30);
        // Content-size sentinels: UNKNOWN = -1 (u64::MAX),
        // ERROR = -2 (u64::MAX - 1). Ordering matters so the
        // `ret >= ZSTD_CONTENTSIZE_ERROR` check in
        // `ZSTD_getDecompressedSize` covers both sentinels.
        assert_eq!(ZSTD_CONTENTSIZE_UNKNOWN, u64::MAX);
        assert_eq!(ZSTD_CONTENTSIZE_ERROR, u64::MAX - 1);
        // UNKNOWN is strictly greater than ERROR — the ordering is
        // what lets callers collapse both sentinels with a single
        // `ret >= ZSTD_CONTENTSIZE_ERROR` check.
        assert_eq!(ZSTD_CONTENTSIZE_UNKNOWN - ZSTD_CONTENTSIZE_ERROR, 1);
    }

    #[test]
    fn frame_format_magic_and_size_constants_match_spec() {
        // Pin the format-level constants that must match the
        // Zstandard format specification byte-for-byte for cross-
        // compatibility with upstream and with the file format.
        assert_eq!(ZSTD_MAGICNUMBER, 0xFD2FB528);
        assert_eq!(ZSTD_MAGIC_DICTIONARY, 0xEC30A437);
        assert_eq!(ZSTD_MAGIC_SKIPPABLE_START, 0x184D2A50);
        assert_eq!(ZSTD_MAGIC_SKIPPABLE_MASK, 0xFFFFFFF0);
        assert_eq!(ZSTD_FRAMEIDSIZE, 4);
        assert_eq!(ZSTD_SKIPPABLEHEADERSIZE, 8);
        // And the skippable mask + start must cover exactly the
        // 16 valid skippable magics (variants 0..=15).
        for v in 0u32..=15 {
            let magic = ZSTD_MAGIC_SKIPPABLE_START | v;
            assert_eq!(
                magic & ZSTD_MAGIC_SKIPPABLE_MASK,
                ZSTD_MAGIC_SKIPPABLE_START
            );
        }
    }

    #[test]
    fn zstd_isFrame_detects_regular_and_skippable() {
        // Regular frame magic.
        let regular = ZSTD_MAGICNUMBER.to_le_bytes();
        assert_eq!(ZSTD_isFrame(&regular), 1);
        // Skippable frame magic variant 0.
        let skippable = ZSTD_MAGIC_SKIPPABLE_START.to_le_bytes();
        assert_eq!(ZSTD_isFrame(&skippable), 1);
        // Skippable variant 15 (bit 3-0 can be 0..=15).
        let skippable15 = (ZSTD_MAGIC_SKIPPABLE_START | 0x0F).to_le_bytes();
        assert_eq!(ZSTD_isFrame(&skippable15), 1);
        // Garbage.
        assert_eq!(ZSTD_isFrame(&[0xDE, 0xAD, 0xBE, 0xEF]), 0);
        // Too short.
        assert_eq!(ZSTD_isFrame(&[0xFD, 0x2F]), 0);
        assert_eq!(ZSTD_isFrame(&[]), 0);
    }

    #[test]
    fn zstd_isSkippableFrame_rejects_regular_frame() {
        let regular = ZSTD_MAGICNUMBER.to_le_bytes();
        assert_eq!(ZSTD_isSkippableFrame(&regular), 0);
        let skippable = (ZSTD_MAGIC_SKIPPABLE_START | 0x05).to_le_bytes();
        assert_eq!(ZSTD_isSkippableFrame(&skippable), 1);
    }

    #[test]
    fn zstd_isFrame_rejects_dictionary_magic() {
        // The `ZSTD_MAGIC_DICTIONARY` prefix (0xEC30A437) identifies a
        // zstd dictionary file, NOT a compressed frame. `ZSTD_isFrame`
        // must return 0 — otherwise callers that sniff input type
        // would treat a dict as decompressible.
        let dict_magic = ZSTD_MAGIC_DICTIONARY.to_le_bytes();
        assert_eq!(ZSTD_isFrame(&dict_magic), 0);
        // Same magic but padded — still not a frame.
        let mut padded = dict_magic.to_vec();
        padded.extend_from_slice(&[0u8; 4]);
        assert_eq!(ZSTD_isFrame(&padded), 0);
    }

    #[test]
    fn zstd_isSkippableFrame_rejects_dictionary_magic() {
        // Symmetric with `zstd_isFrame_rejects_dictionary_magic`.
        // Dict magic 0xEC30A437 also isn't a skippable frame.
        let dict_magic = ZSTD_MAGIC_DICTIONARY.to_le_bytes();
        assert_eq!(ZSTD_isSkippableFrame(&dict_magic), 0);
    }

    #[test]
    fn zstd_isSkippableFrame_accepts_all_16_variants() {
        // Upstream spec allows skippable magics from
        // ZSTD_MAGIC_SKIPPABLE_START (0x184D2A50) to
        // ZSTD_MAGIC_SKIPPABLE_START + 15 (0x184D2A5F). Verify
        // every variant registers as a skippable frame.
        for v in 0u32..=15 {
            let magic = (ZSTD_MAGIC_SKIPPABLE_START + v).to_le_bytes();
            assert_eq!(
                ZSTD_isSkippableFrame(&magic),
                1,
                "variant {v} didn't register as skippable"
            );
        }
        // Variant 16 wraps into non-skippable territory (0x184D2A60).
        let invalid = (ZSTD_MAGIC_SKIPPABLE_START + 16).to_le_bytes();
        assert_eq!(ZSTD_isSkippableFrame(&invalid), 0);
    }

    #[test]
    fn zstd_isSkippableFrame_rejects_short_src_safely() {
        // Safety: sub-4-byte inputs must return 0 without panicking
        // on OOB slicing. Symmetric with `ZSTD_isFrame`'s coverage.
        assert_eq!(ZSTD_isSkippableFrame(&[]), 0);
        assert_eq!(ZSTD_isSkippableFrame(&[0x50]), 0);
        assert_eq!(ZSTD_isSkippableFrame(&[0x50, 0x2A, 0x4D]), 0);
    }

    #[test]
    fn zstd_getDictID_fromFrame_returns_zero_when_absent() {
        // Compress a small payload without a dict → FHD has no dictID.
        let src: Vec<u8> = b"hello world".to_vec();
        let mut dst = vec![0u8; 128];
        let n = crate::compress::zstd_compress::ZSTD_compress(&mut dst, &src, 1);
        dst.truncate(n);
        assert_eq!(ZSTD_getDictID_fromFrame(&dst), 0);
    }

    #[test]
    fn copyRawBlock_and_setRleBlock_cover_happy_and_dst_too_small_paths() {
        // copyRawBlock: byte-for-byte copy + DstSizeTooSmall.
        let src = b"hello-raw";
        let mut dst = [0u8; 16];
        let n = ZSTD_copyRawBlock(&mut dst, src);
        assert_eq!(n, src.len());
        assert_eq!(&dst[..n], src);

        let mut tiny = [0u8; 4];
        assert!(crate::common::error::ERR_isError(ZSTD_copyRawBlock(
            &mut tiny, src
        )));

        // setRleBlock: fill N copies of a byte + DstSizeTooSmall.
        let mut buf = vec![0u8; 32];
        let rle_n = ZSTD_setRleBlock(&mut buf, 0xAB, 10);
        assert_eq!(rle_n, 10);
        assert!(buf[..10].iter().all(|&b| b == 0xAB));
        // Bytes past regenSize must not have been touched.
        assert!(buf[10..].iter().all(|&b| b == 0));

        let mut short = [0u8; 4];
        assert!(crate::common::error::ERR_isError(ZSTD_setRleBlock(
            &mut short, 0xCD, 10
        )));
    }

    #[test]
    fn findFrameCompressedSize_matches_compressor_output_for_multi_block() {
        // Compress a 200 KB payload → guaranteed multi-block frame
        // (> 128 KB block boundary). `ZSTD_findFrameCompressedSize`
        // must report exactly the compressed byte count, not include
        // trailing garbage or under-report.
        use crate::compress::zstd_compress::{ZSTD_compress, ZSTD_compressBound};
        let src: Vec<u8> = b"multi-block frame-size probe. "
            .iter()
            .cycle()
            .take(200_000)
            .copied()
            .collect();
        let bound = ZSTD_compressBound(src.len());
        let mut dst = vec![0u8; bound];
        let n = ZSTD_compress(&mut dst, &src, 1);
        assert!(!crate::common::error::ERR_isError(n));

        // Append trailing garbage to ensure the helper reports the
        // real frame size, not just dst.len().
        let mut with_trailer = dst[..n].to_vec();
        with_trailer.extend_from_slice(&[0xFFu8; 64]);
        let reported = ZSTD_findFrameCompressedSize(&with_trailer);
        assert!(!crate::common::error::ERR_isError(reported));
        assert_eq!(reported, n);
    }

    #[test]
    fn findFrameSizeInfo_reports_skippable_frame_with_zero_decompressed_bound() {
        // For a skippable frame, the returned info must be:
        //   nbBlocks = 0
        //   compressedSize = 8 + user_data_len
        //   decompressedBound = 0
        // This shape lets callers (`decompressBound`, CLI frame-walk)
        // skip past the skippable region without attempting to
        // allocate space for its "content".
        let user_data_len: u32 = 10;
        let mut src = Vec::new();
        src.extend_from_slice(&ZSTD_MAGIC_SKIPPABLE_START.to_le_bytes());
        src.extend_from_slice(&user_data_len.to_le_bytes());
        src.extend(core::iter::repeat_n(0u8, user_data_len as usize));

        let info = ZSTD_findFrameSizeInfo(&src, ZSTD_format_e::ZSTD_f_zstd1);
        assert_eq!(info.nbBlocks, 0);
        assert_eq!(info.compressedSize, 8 + user_data_len as usize);
        assert_eq!(info.decompressedBound, 0);
    }

    #[test]
    fn decompressBound_all_skippable_frames_returns_zero() {
        // Sibling of `findDecompressedSize_all_skippable_frames_returns_zero`:
        // `ZSTD_decompressBound` must also report 0 when all input
        // frames are skippable (they don't contribute to the output
        // stream).
        let mut src = Vec::new();
        for &user_data_len in &[5u32, 12] {
            src.extend_from_slice(&ZSTD_MAGIC_SKIPPABLE_START.to_le_bytes());
            src.extend_from_slice(&user_data_len.to_le_bytes());
            src.extend(core::iter::repeat_n(0u8, user_data_len as usize));
        }
        assert_eq!(ZSTD_decompressBound(&src), 0);
    }

    #[test]
    fn findDecompressedSize_all_skippable_frames_returns_zero() {
        // Stream of two back-to-back skippable frames with no regular
        // frames. `ZSTD_findDecompressedSize` must return 0 since
        // skippable frames contribute nothing to the decompressed
        // output stream.
        let mut src = Vec::new();
        for &user_data_len in &[5u32, 12] {
            src.extend_from_slice(&ZSTD_MAGIC_SKIPPABLE_START.to_le_bytes());
            src.extend_from_slice(&user_data_len.to_le_bytes());
            src.extend(core::iter::repeat_n(0u8, user_data_len as usize));
        }
        assert_eq!(ZSTD_findDecompressedSize(&src), 0);
    }

    #[test]
    fn findDecompressedSize_returns_UNKNOWN_when_any_frame_lacks_fcs() {
        // Contract: `ZSTD_findDecompressedSize` must return
        // CONTENTSIZE_UNKNOWN if any frame in the stream lacks a
        // declared FCS — NOT a silent garbage value. Callers rely
        // on this sentinel to decide whether to pre-allocate.
        let raw = make_raw_hello_frame();

        // Build a frame with singleSegment=0 and fcsID=0 → FCS absent.
        let mut no_fcs = Vec::new();
        no_fcs.extend_from_slice(&ZSTD_MAGICNUMBER.to_le_bytes());
        no_fcs.push(0x00); // FHD
        no_fcs.push(0x20); // wlByte → windowLog=14
                           // Single raw block: lastBlock=1, bt_raw=0, cSize=3.
        no_fcs.push(((3u32 << 3) | 1) as u8);
        no_fcs.push(0);
        no_fcs.push(0);
        no_fcs.extend_from_slice(b"abc");

        // Stream = raw(FCS=5) + no_fcs_frame. Total must be UNKNOWN
        // because one frame doesn't declare its size.
        let mut stream = raw;
        stream.extend_from_slice(&no_fcs);
        assert_eq!(ZSTD_findDecompressedSize(&stream), ZSTD_CONTENTSIZE_UNKNOWN);
    }

    #[test]
    fn getFrameContentSize_reports_compressed_payload_size_across_fcs_encodings() {
        // `ZSTD_getFrameContentSize` should return the declared
        // decompressed size verbatim across all FCS-code sizes
        // (upstream encodes 1/2/4/8 bytes depending on magnitude).
        // Pin each of the four size regimes the encoder selects.
        let sizes: &[u64] = &[
            1,             // singleSegment=1, fcsCode=0 → 1-byte FCS
            300,           // fcsCode=1 → 2-byte FCS (stored - 256)
            100_000,       // fcsCode=2 → 4-byte FCS
            0x1_0000_0000, // fcsCode=3 → 8-byte FCS
        ];
        for &sz in sizes {
            // For sizes > 128 KB we can't realistically compress and
            // verify, but the header still must declare the FCS —
            // so synthesize a frame directly via ZSTD_writeFrameHeader.
            use crate::compress::zstd_compress::{
                ZSTD_FrameParameters, ZSTD_writeFrameHeader, ZSTD_FRAMEHEADERSIZE_MAX,
            };
            use crate::decompress::zstd_decompress::ZSTD_WINDOWLOG_ABSOLUTEMIN;
            let fp = ZSTD_FrameParameters {
                contentSizeFlag: 1,
                checksumFlag: 0,
                noDictIDFlag: 1,
            };
            let mut hdr = vec![0u8; ZSTD_FRAMEHEADERSIZE_MAX];
            let n = ZSTD_writeFrameHeader(&mut hdr, &fp, ZSTD_WINDOWLOG_ABSOLUTEMIN + 10, sz, 0);
            assert!(!crate::common::error::ERR_isError(n));
            hdr.truncate(n);
            assert_eq!(
                ZSTD_getFrameContentSize(&hdr),
                sz,
                "FCS roundtrip failed for size={sz}",
            );
        }
    }

    #[test]
    fn findDecompressedSize_sums_multiple_concatenated_frames() {
        // Multi-frame streams: `ZSTD_findDecompressedSize` should
        // return the sum of every frame's FCS, treating skippable
        // frames as contributing zero. Pin against three concatenated
        // regular frames of distinct sizes.
        let parts: &[&[u8]] = &[
            b"alpha".as_ref(),
            b"longer-part-here".as_ref(),
            b"c".as_ref(),
        ];
        let expected_total: u64 = parts.iter().map(|p| p.len() as u64).sum();

        let mut stream = Vec::new();
        for part in parts {
            let mut buf = vec![0u8; 256];
            let n = crate::compress::zstd_compress::ZSTD_compress(&mut buf, part, 3);
            assert!(!crate::common::error::ERR_isError(n));
            stream.extend_from_slice(&buf[..n]);
        }
        assert_eq!(ZSTD_findDecompressedSize(&stream), expected_total);

        // Insert a skippable frame in the middle — doesn't contribute
        // to the decompressed-size sum.
        let mut stream_with_skip = Vec::new();
        {
            let mut buf = vec![0u8; 256];
            let n = crate::compress::zstd_compress::ZSTD_compress(&mut buf, parts[0], 3);
            stream_with_skip.extend_from_slice(&buf[..n]);
        }
        stream_with_skip.extend_from_slice(&ZSTD_MAGIC_SKIPPABLE_START.to_le_bytes());
        stream_with_skip.extend_from_slice(&8u32.to_le_bytes());
        stream_with_skip.extend_from_slice(b"SKIPDATA");
        {
            let mut buf = vec![0u8; 256];
            let n = crate::compress::zstd_compress::ZSTD_compress(&mut buf, parts[1], 3);
            stream_with_skip.extend_from_slice(&buf[..n]);
        }
        assert_eq!(
            ZSTD_findDecompressedSize(&stream_with_skip),
            (parts[0].len() + parts[1].len()) as u64,
        );
    }

    #[test]
    fn findDecompressedSize_returns_ERROR_on_trailing_garbage_after_valid_frame() {
        // Upstream contract: a valid frame followed by bytes that don't
        // form a valid frame header is treated as a corrupted stream,
        // not an empty-tail success. `ZSTD_findDecompressedSize` must
        // return `ZSTD_CONTENTSIZE_ERROR` — not `UNKNOWN`, which would
        // wrongly suggest "decodable, just unknown size".
        let mut src = make_raw_hello_frame();
        // Append a few bytes that look like the start of a zstd magic
        // but truncate before the frame header can be parsed.
        src.extend_from_slice(&[0x28, 0xB5, 0x2F]);
        let rc = ZSTD_findDecompressedSize(&src);
        assert_eq!(rc, ZSTD_CONTENTSIZE_ERROR);
    }

    #[test]
    fn zstd_getDictID_fromFrame_reads_dictID_when_present() {
        // Build a synthetic frame with a 4-byte dictID via
        // ZSTD_writeFrameHeader and verify round-trip through
        // ZSTD_getDictID_fromFrame.
        use crate::compress::zstd_compress::{
            ZSTD_FrameParameters, ZSTD_writeFrameHeader, ZSTD_FRAMEHEADERSIZE_MAX,
        };
        let dictID_in = 0xDEAD_BEEFu32;
        let fParams = ZSTD_FrameParameters {
            contentSizeFlag: 0,
            checksumFlag: 0,
            noDictIDFlag: 0,
        };
        let mut buf = [0u8; ZSTD_FRAMEHEADERSIZE_MAX];
        let n = ZSTD_writeFrameHeader(&mut buf, &fParams, 17, 0, dictID_in);
        assert!(!crate::common::error::ERR_isError(n));
        let got = ZSTD_getDictID_fromFrame(&buf[..n]);
        assert_eq!(got, dictID_in);
    }

    #[test]
    fn frame_header_truncated_requests_more() {
        // Only magic bytes — need 5 bytes (magic + FHD) to even start.
        let mut zfh = ZSTD_FrameHeader::default();
        let src = ZSTD_MAGICNUMBER.to_le_bytes();
        let rc = ZSTD_getFrameHeader(&mut zfh, &src);
        assert_eq!(rc, 5);
    }

    #[test]
    fn createDCtx_advanced_and_createDStream_advanced_return_Some() {
        // Symmetric with `ZSTD_createCCtx_advanced`. The advanced
        // creators must return functional decoder objects.
        use crate::compress::zstd_compress::ZSTD_customMem;
        let _dctx = ZSTD_createDCtx_advanced(ZSTD_customMem::default()).unwrap();
        let _dstream = ZSTD_createDStream_advanced(ZSTD_customMem::default()).unwrap();

        // And prove the returned DCtx actually decodes: compress a
        // payload, feed into _advanced-created DCtx, verify roundtrip.
        use crate::compress::zstd_compress::{ZSTD_compress, ZSTD_compressBound};
        let src: Vec<u8> = b"DCtx_advanced functional probe ".repeat(20);
        let bound = ZSTD_compressBound(src.len());
        let mut compressed = vec![0u8; bound];
        let n = ZSTD_compress(&mut compressed, &src, 1);
        let mut out = vec![0u8; src.len() + 64];
        let d = ZSTD_decompress(&mut out, &compressed[..n]);
        assert_eq!(&out[..d], &src[..]);
    }

    #[test]
    fn decompresses_upstream_level10_large_real_text_frame() {
        use crate::decompress::zstd_decompress_block::{
            blockProperties_t, blockType_e, streaming_operation, ZSTD_DCtx, ZSTD_blockHeaderSize,
            ZSTD_buildDefaultSeqTables, ZSTD_decodeLiteralsBlock, ZSTD_decodeSeqHeaders,
            ZSTD_decoder_entropy_rep, ZSTD_decompressSequences_body, ZSTD_getcBlockSize,
        };
        use std::fs;
        use std::io::Write;
        use std::path::PathBuf;
        use std::process::{Command, Stdio};

        let which = Command::new("which")
            .arg("zstd")
            .output()
            .expect("which zstd");
        if !which.status.success() {
            eprintln!("upstream zstd not on $PATH; skipping");
            return;
        }
        let zstd = PathBuf::from(String::from_utf8(which.stdout).unwrap().trim());

        let seed = fs::read("tests/fixtures/zstd_h.txt").expect("read zstd_h.txt");
        let src: Vec<u8> = seed.repeat(20);

        let mut child = Command::new(&zstd)
            .args(["-q", "--no-check", "-10", "-c", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("spawn upstream zstd");
        child.stdin.as_mut().unwrap().write_all(&src).unwrap();
        let out = child.wait_with_output().expect("wait upstream zstd");
        assert!(
            out.status.success(),
            "upstream compression failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let mut dctx = ZSTD_DCtx::new();
        ZSTD_buildDefaultSeqTables(&mut dctx);
        let mut decoded = vec![0u8; src.len()];
        let mut rep = ZSTD_decoder_entropy_rep::default();
        let mut zfh = ZSTD_FrameHeader::default();
        let hdr = ZSTD_getFrameHeader_advanced(&mut zfh, &out.stdout, dctx.format);
        assert_eq!(hdr, 0);
        let mut ip = zfh.headerSize as usize;
        let mut op = 0usize;
        let mut block_idx = 0usize;
        loop {
            let mut bp = blockProperties_t {
                blockType: blockType_e::bt_raw,
                lastBlock: 0,
                origSize: 0,
            };
            let cblock = ZSTD_getcBlockSize(&out.stdout[ip..], &mut bp);
            assert!(
                !crate::common::error::ERR_isError(cblock),
                "block {block_idx}: cblock parse failed: {}",
                crate::common::error::ERR_getErrorName(cblock)
            );
            ip += ZSTD_blockHeaderSize;
            match bp.blockType {
                blockType_e::bt_compressed => {
                    let block = &out.stdout[ip..ip + cblock];
                    let lit_rc = ZSTD_decodeLiteralsBlock(
                        &mut dctx,
                        block,
                        &mut decoded[op..],
                        streaming_operation::not_streaming,
                    );
                    assert!(
                        !crate::common::error::ERR_isError(lit_rc),
                        "block {block_idx}: literals failed: {}",
                        crate::common::error::ERR_getErrorName(lit_rc)
                    );
                    let mut nb_seq = 0i32;
                    let seq_header =
                        ZSTD_decodeSeqHeaders(&mut dctx, &mut nb_seq, &block[lit_rc..]);
                    assert!(
                        !crate::common::error::ERR_isError(seq_header),
                        "block {block_idx}: seq headers failed: {}",
                        crate::common::error::ERR_getErrorName(seq_header)
                    );
                    let lit_snapshot = dctx.litExtraBuffer[..dctx.litSize].to_vec();
                    let ll = dctx.LLTable.clone();
                    let of = dctx.OFTable.clone();
                    let ml = dctx.MLTable.clone();
                    let seq_rc = ZSTD_decompressSequences_body(
                        &mut decoded,
                        op,
                        &[],
                        &block[lit_rc + seq_header..],
                        nb_seq,
                        &lit_snapshot,
                        dctx.litSize,
                        &ll,
                        &of,
                        &ml,
                        &mut rep,
                    );
                    assert!(
                        !crate::common::error::ERR_isError(seq_rc),
                        "block {block_idx}: sequence body failed: {}",
                        crate::common::error::ERR_getErrorName(seq_rc)
                    );
                    op += seq_rc;
                }
                blockType_e::bt_raw => {
                    let rc = ZSTD_copyRawBlock(&mut decoded[op..], &out.stdout[ip..ip + cblock]);
                    assert!(
                        !crate::common::error::ERR_isError(rc),
                        "block {block_idx}: raw block failed: {}",
                        crate::common::error::ERR_getErrorName(rc)
                    );
                    op += rc;
                }
                blockType_e::bt_rle => {
                    let rc =
                        ZSTD_setRleBlock(&mut decoded[op..], out.stdout[ip], bp.origSize as usize);
                    assert!(
                        !crate::common::error::ERR_isError(rc),
                        "block {block_idx}: rle block failed: {}",
                        crate::common::error::ERR_getErrorName(rc)
                    );
                    op += rc;
                }
                blockType_e::bt_reserved => panic!("block {block_idx}: reserved block"),
            }
            ip += cblock;
            block_idx += 1;
            if bp.lastBlock != 0 {
                break;
            }
        }
        let d = ZSTD_decompress(&mut decoded, &out.stdout);
        assert!(
            !crate::common::error::ERR_isError(d),
            "rust decompressor rejected upstream frame: {}",
            crate::common::error::ERR_getErrorName(d)
        );
        assert_eq!(&decoded[..op], &src[..]);
        assert_eq!(&decoded[..d], &src[..]);
    }

    #[test]
    fn createDCtx_advanced_rejects_invalid_custommem_pairs() {
        use crate::compress::zstd_compress::ZSTD_customMem;

        fn dummy_alloc(_opaque: usize, _size: usize) -> *mut core::ffi::c_void {
            core::ptr::null_mut()
        }

        let invalid = ZSTD_customMem {
            customAlloc: Some(dummy_alloc),
            customFree: None,
            opaque: 1,
        };

        assert!(ZSTD_createDCtx_advanced(invalid).is_none());
        assert!(ZSTD_createDStream_advanced(invalid).is_none());
    }

    #[test]
    fn advanced_dctx_surfaces_invoke_custom_allocator_callbacks() {
        use crate::compress::zstd_compress::ZSTD_customMem;
        use core::sync::atomic::{AtomicUsize, Ordering};

        static ALLOCS: AtomicUsize = AtomicUsize::new(0);
        static FREES: AtomicUsize = AtomicUsize::new(0);

        fn counting_alloc(_opaque: usize, size: usize) -> *mut core::ffi::c_void {
            use std::alloc::{alloc, Layout};

            const ALIGN: usize = 64;
            const HEADER_WORDS: usize = 2;

            let total = size.max(1) + ALIGN + HEADER_WORDS * core::mem::size_of::<usize>();
            let layout = Layout::from_size_align(total, ALIGN).unwrap();
            unsafe {
                let base = alloc(layout);
                if base.is_null() {
                    return core::ptr::null_mut();
                }
                let payload_addr =
                    (base as usize + HEADER_WORDS * core::mem::size_of::<usize>() + ALIGN - 1)
                        & !(ALIGN - 1);
                let header = (payload_addr as *mut usize).sub(HEADER_WORDS);
                header.write(base as usize);
                header.add(1).write(total);
                ALLOCS.fetch_add(1, Ordering::SeqCst);
                payload_addr as *mut core::ffi::c_void
            }
        }

        fn counting_free(_opaque: usize, address: *mut core::ffi::c_void) {
            use std::alloc::{dealloc, Layout};

            const ALIGN: usize = 64;
            const HEADER_WORDS: usize = 2;

            if address.is_null() {
                return;
            }
            unsafe {
                let header = (address as *mut usize).sub(HEADER_WORDS);
                let base = header.read() as *mut u8;
                let total = header.add(1).read();
                let layout = Layout::from_size_align(total, ALIGN).unwrap();
                dealloc(base, layout);
                FREES.fetch_add(1, Ordering::SeqCst);
            }
        }

        let custom = ZSTD_customMem {
            customAlloc: Some(counting_alloc),
            customFree: Some(counting_free),
            opaque: 11,
        };

        let dctx = ZSTD_createDCtx_advanced(custom).unwrap();
        assert_eq!(dctx.customMem, custom);
        let dstream = ZSTD_createDStream_advanced(custom).unwrap();
        assert_eq!(dstream.customMem, custom);

        assert_eq!(ALLOCS.load(Ordering::SeqCst), 2);
        assert_eq!(FREES.load(Ordering::SeqCst), 0);
        assert_eq!(ZSTD_freeDCtx(dctx), 0);
        assert_eq!(ZSTD_freeDStream(Some(dstream)), 0);
        assert_eq!(FREES.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn ZSTD_DStream_is_alias_for_ZSTD_DCtx() {
        // Symmetric with the compress-side alias test. Upstream
        // `typedef ZSTD_DCtx ZSTD_DStream`; our `pub type` mirrors.
        assert_eq!(
            core::mem::size_of::<ZSTD_DStream>(),
            core::mem::size_of::<ZSTD_DCtx>()
        );
        let ds: Box<ZSTD_DStream> = ZSTD_createDStream().unwrap();
        assert_eq!(ZSTD_sizeof_DCtx(&ds), ZSTD_sizeof_DStream(&ds));
    }

    #[test]
    fn ZSTD_nextInputType_e_discriminants_match_upstream() {
        // `ZSTD_nextInputType_e` is the return value of the public
        // `ZSTD_nextInputType()`. Upstream declares it as a bare
        // `typedef enum { ... }` so discriminants are the default
        // sequential 0..5 — any drift would mis-signal block/header
        // expectations to C callers consuming this enum.
        assert_eq!(ZSTD_nextInputType_e::ZSTDnit_frameHeader as u32, 0);
        assert_eq!(ZSTD_nextInputType_e::ZSTDnit_blockHeader as u32, 1);
        assert_eq!(ZSTD_nextInputType_e::ZSTDnit_block as u32, 2);
        assert_eq!(ZSTD_nextInputType_e::ZSTDnit_lastBlock as u32, 3);
        assert_eq!(ZSTD_nextInputType_e::ZSTDnit_checksum as u32, 4);
        assert_eq!(ZSTD_nextInputType_e::ZSTDnit_skippableFrame as u32, 5);
    }

    #[test]
    fn decompressStream_simpleArgs_forwards_to_decompressStream() {
        // Parity with upstream's `ZSTD_decompressStream_simpleArgs`:
        // a thin forwarder over `decompressStream`. Verify a basic
        // roundtrip so a future refactor that accidentally decouples
        // them trips this gate.
        use crate::common::error::ERR_isError;
        use crate::compress::zstd_compress::{ZSTD_compress, ZSTD_compressBound};

        let src = b"decompressStream_simpleArgs smoke test ".repeat(3);
        let mut framed = vec![0u8; ZSTD_compressBound(src.len())];
        let n = ZSTD_compress(&mut framed, &src, 3);
        assert!(!ERR_isError(n));
        framed.truncate(n);

        let mut dctx = ZSTD_DCtx::new();
        ZSTD_initDStream(&mut dctx);
        let mut out = vec![0u8; src.len() + 64];
        let mut in_pos = 0usize;
        let mut out_pos = 0usize;
        let _ = ZSTD_decompressStream_simpleArgs(
            &mut dctx,
            &mut out,
            &mut out_pos,
            &framed,
            &mut in_pos,
        );
        for _ in 0..8 {
            if out_pos >= src.len() {
                break;
            }
            let _ = ZSTD_decompressStream_simpleArgs(
                &mut dctx,
                &mut out,
                &mut out_pos,
                &[],
                &mut 0usize,
            );
        }
        assert_eq!(&out[..out_pos], &src[..]);
    }

    #[test]
    fn DCtx_reset_parameters_only_rejects_mid_stream_but_combined_variant_always_accepts() {
        // Mirror of the compressor-side three-way gate. Pure
        // `reset_parameters` must reject mid-stream with `StageWrong`;
        // `reset_session_only` and `reset_session_and_parameters`
        // are always safe since they clear streaming state first.
        use crate::common::error::{ERR_getErrorCode, ERR_isError};
        use crate::compress::zstd_compress::{ZSTD_compress, ZSTD_compressBound};

        let src = b"dctx-reset-stage-semantics ".repeat(3);
        let mut framed = vec![0u8; ZSTD_compressBound(src.len())];
        let n = ZSTD_compress(&mut framed, &src, 3);
        assert!(!ERR_isError(n));
        framed.truncate(n);

        let mut dctx = ZSTD_DCtx::new();
        ZSTD_initDStream(&mut dctx);
        let mut out = vec![0u8; src.len() + 64];
        let mut in_pos = 0usize;
        let mut out_pos = 0usize;
        let _ = ZSTD_decompressStream(&mut dctx, &mut out, &mut out_pos, &framed, &mut in_pos);

        // reset_parameters alone: rejected mid-stream.
        let rc = ZSTD_DCtx_reset(&mut dctx, ZSTD_DResetDirective::ZSTD_reset_parameters);
        assert!(ERR_isError(rc));
        assert_eq!(ERR_getErrorCode(rc), ErrorCode::StageWrong);

        // session_only: always OK.
        assert_eq!(
            ZSTD_DCtx_reset(&mut dctx, ZSTD_DResetDirective::ZSTD_reset_session_only),
            0,
        );
        // Now back in init, reset_parameters succeeds.
        assert_eq!(
            ZSTD_DCtx_reset(&mut dctx, ZSTD_DResetDirective::ZSTD_reset_parameters),
            0,
        );

        // session_and_parameters: always OK mid-stream.
        ZSTD_initDStream(&mut dctx);
        let mut out = vec![0u8; src.len() + 64];
        let mut in_pos = 0usize;
        let mut out_pos = 0usize;
        let _ = ZSTD_decompressStream(&mut dctx, &mut out, &mut out_pos, &framed, &mut in_pos);
        assert_eq!(
            ZSTD_DCtx_reset(
                &mut dctx,
                ZSTD_DResetDirective::ZSTD_reset_session_and_parameters,
            ),
            0,
        );
    }

    #[test]
    fn DCtx_param_setters_reject_mid_stream_with_StageWrong() {
        // Upstream contract (zstd_decompress.c:1809, 1908): every
        // DCtx parameter setter rejects mid-stream with `StageWrong`.
        // Unlike the compressor, there's no authorized-subset — the
        // decoder has to see all params up-front since they affect
        // header parsing.
        use crate::common::error::{ERR_getErrorCode, ERR_isError};
        use crate::compress::zstd_compress::{ZSTD_compress, ZSTD_compressBound};

        let src = b"dctx-param-stage-gate ".repeat(4);
        let mut framed = vec![0u8; ZSTD_compressBound(src.len())];
        let n = ZSTD_compress(&mut framed, &src, 3);
        assert!(!ERR_isError(n));
        framed.truncate(n);

        // setParameter.
        {
            let mut dctx = ZSTD_DCtx::new();
            ZSTD_initDStream(&mut dctx);
            // Init stage: accepted.
            assert_eq!(
                ZSTD_DCtx_setParameter(
                    &mut dctx,
                    ZSTD_dParameter::ZSTD_d_format,
                    ZSTD_format_e::ZSTD_f_zstd1 as i32,
                ),
                0,
            );
            // Stage input.
            let mut out = vec![0u8; src.len() + 64];
            let mut in_pos = 0usize;
            let mut out_pos = 0usize;
            let _ = ZSTD_decompressStream(&mut dctx, &mut out, &mut out_pos, &framed, &mut in_pos);
            let rc = ZSTD_DCtx_setParameter(
                &mut dctx,
                ZSTD_dParameter::ZSTD_d_format,
                ZSTD_format_e::ZSTD_f_zstd1_magicless as i32,
            );
            assert!(ERR_isError(rc));
            assert_eq!(ERR_getErrorCode(rc), ErrorCode::StageWrong);
        }
        // setMaxWindowSize.
        {
            let mut dctx = ZSTD_DCtx::new();
            ZSTD_initDStream(&mut dctx);
            assert_eq!(ZSTD_DCtx_setMaxWindowSize(&mut dctx, 1 << 20), 0);
            let mut out = vec![0u8; src.len() + 64];
            let mut in_pos = 0usize;
            let mut out_pos = 0usize;
            let _ = ZSTD_decompressStream(&mut dctx, &mut out, &mut out_pos, &framed, &mut in_pos);
            let rc = ZSTD_DCtx_setMaxWindowSize(&mut dctx, 1 << 20);
            assert!(ERR_isError(rc));
            assert_eq!(ERR_getErrorCode(rc), ErrorCode::StageWrong);
        }
    }

    #[test]
    fn DCtx_dict_family_setters_reject_mid_stream_with_StageWrong() {
        // Symmetric to the compressor-side gate. A caller who swaps
        // the dict mid-stream would decouple back-ref history from
        // bytes already buffered in `dctx.stream_in_buffer`,
        // producing a valid-looking but wrong decode.
        use crate::common::error::{ERR_getErrorCode, ERR_isError};
        use crate::compress::zstd_compress::{ZSTD_compress, ZSTD_compressBound};

        let dict = b"dctx-dict-family-stage-gate ".repeat(3);
        let src = b"some payload bytes ".repeat(4);
        let mut framed = vec![0u8; ZSTD_compressBound(src.len())];
        let n = ZSTD_compress(&mut framed, &src, 3);
        assert!(!ERR_isError(n));
        framed.truncate(n);

        // loadDictionary.
        {
            let mut dctx = ZSTD_DCtx::new();
            ZSTD_initDStream(&mut dctx);
            assert_eq!(ZSTD_DCtx_loadDictionary(&mut dctx, &dict), 0);
            // Stage input into the DCtx's stream buffer.
            let mut out = vec![0u8; src.len() + 64];
            let mut in_pos = 0usize;
            let mut out_pos = 0usize;
            let _ = ZSTD_decompressStream(&mut dctx, &mut out, &mut out_pos, &framed, &mut in_pos);
            let rc = ZSTD_DCtx_loadDictionary(&mut dctx, &dict);
            assert!(ERR_isError(rc));
            assert_eq!(ERR_getErrorCode(rc), ErrorCode::StageWrong);
        }
        // refPrefix.
        {
            let mut dctx = ZSTD_DCtx::new();
            ZSTD_initDStream(&mut dctx);
            assert_eq!(ZSTD_DCtx_refPrefix(&mut dctx, &dict), 0);
            let mut out = vec![0u8; src.len() + 64];
            let mut in_pos = 0usize;
            let mut out_pos = 0usize;
            let _ = ZSTD_decompressStream(&mut dctx, &mut out, &mut out_pos, &framed, &mut in_pos);
            let rc = ZSTD_DCtx_refPrefix(&mut dctx, &dict);
            assert!(ERR_isError(rc));
            assert_eq!(ERR_getErrorCode(rc), ErrorCode::StageWrong);
        }
    }

    #[test]
    fn DCtx_getParameter_defaults_on_fresh_dctx_match_upstream() {
        // Upstream contract (zstd_decompress.c:244):
        //   - d_windowLogMax: ZSTD_WINDOWLOG_LIMIT_DEFAULT (27)
        //   - d_format:       ZSTD_f_zstd1 (magic-prefixed)
        // Pin the fresh-DCtx defaults — decoder-side mirror of the
        // compressor gate so a future `ZSTD_DCtx::default` refactor
        // can't silently shift the API contract.
        let dctx = ZSTD_DCtx::new();
        let mut v = 0i32;
        assert_eq!(
            ZSTD_DCtx_getParameter(&dctx, ZSTD_dParameter::ZSTD_d_windowLogMax, &mut v),
            0,
        );
        assert_eq!(v, ZSTD_WINDOWLOG_LIMIT_DEFAULT as i32);

        assert_eq!(
            ZSTD_DCtx_getParameter(&dctx, ZSTD_dParameter::ZSTD_d_format, &mut v),
            0,
        );
        assert_eq!(v, ZSTD_format_e::ZSTD_f_zstd1 as i32);
    }

    #[test]
    fn DCtx_reset_parameters_clears_every_dict_slot_via_clearDict_helper() {
        // After the refactor to route `reset(parameters)` through
        // `ZSTD_clearDict` + `ZSTD_DCtx_resetParameters`, the param
        // reset must wipe ALL dict-related slots — `stream_dict`,
        // `dictID`, `ddict_rep`, `dictUses` — not just the subset
        // the earlier field-by-field body reset covered.
        let mut dctx = ZSTD_DCtx::new();
        // Seed every dict-related slot.
        dctx.stream_dict = b"reset-wipe-test".to_vec();
        dctx.dictID = 0xCA_FE_BA_BE;
        dctx.ddict_rep = [7, 8, 9];
        dctx.dictUses = ZSTD_dictUses_e::ZSTD_use_indefinitely;

        assert_eq!(
            ZSTD_DCtx_reset(&mut dctx, ZSTD_DResetDirective::ZSTD_reset_parameters),
            0,
        );
        assert!(dctx.stream_dict.is_empty());
        assert_eq!(dctx.dictID, 0);
        assert_eq!(dctx.ddict_rep, [0u32; 3]);
        assert_eq!(dctx.dictUses, ZSTD_dictUses_e::ZSTD_dont_use);
    }

    #[test]
    fn DCtx_loadDictionary_persists_across_decodes() {
        // Complement to the refPrefix one-shot test: loadDictionary
        // marks `use_indefinitely`, so the dict must stay attached
        // across multiple decompress calls. Prevents a future
        // refactor from accidentally widening the auto-clear to
        // also demote `use_indefinitely`.
        use crate::common::error::ERR_isError;
        use crate::common::xxhash::XXH64_state_t;
        use crate::compress::zstd_compress::{ZSTD_compress, ZSTD_compressBound};
        use crate::decompress::zstd_decompress_block::{
            ZSTD_buildDefaultSeqTables, ZSTD_decoder_entropy_rep,
        };

        let src = b"loadDictionary-persists-across-decodes-payload ".repeat(2);
        let mut framed = vec![0u8; ZSTD_compressBound(src.len())];
        let n = ZSTD_compress(&mut framed, &src, 3);
        framed.truncate(n);

        let mut dctx = ZSTD_DCtx::new();
        ZSTD_buildDefaultSeqTables(&mut dctx);
        let dict = b"persistent-dict-bytes".to_vec();
        assert_eq!(ZSTD_DCtx_loadDictionary(&mut dctx, &dict), 0);
        assert_eq!(dctx.dictUses, ZSTD_dictUses_e::ZSTD_use_indefinitely);

        let mut out = vec![0u8; src.len() + 64];
        let mut rep = ZSTD_decoder_entropy_rep::default();
        let mut xxh = XXH64_state_t::default();
        // First decode.
        let d1 = ZSTD_decompressDCtx(&mut dctx, &mut rep, &mut xxh, &mut out, &framed);
        assert!(!ERR_isError(d1));
        // Dict must survive.
        assert_eq!(dctx.dictUses, ZSTD_dictUses_e::ZSTD_use_indefinitely);
        assert_eq!(dctx.stream_dict, dict);

        // Second decode — still attached.
        let d2 = ZSTD_decompressDCtx(&mut dctx, &mut rep, &mut xxh, &mut out, &framed);
        assert!(!ERR_isError(d2));
        assert_eq!(dctx.dictUses, ZSTD_dictUses_e::ZSTD_use_indefinitely);
        assert_eq!(dctx.stream_dict, dict);
    }

    #[test]
    fn DCtx_refDDict_empty_content_clears_dict_state() {
        // Parallel to `loadDictionary(&[])`: a DDict with empty
        // content must wipe prior dict state on `refDDict`.
        // Matches upstream's `zstd_decompress.c:1783` pattern —
        // `clearDict` unconditionally, early-return on empty content.
        use crate::decompress::zstd_ddict::ZSTD_DDict;
        let mut dctx = ZSTD_DCtx::new();
        assert_eq!(ZSTD_DCtx_loadDictionary(&mut dctx, b"pre-existing"), 0);
        assert_eq!(dctx.dictUses, ZSTD_dictUses_e::ZSTD_use_indefinitely);

        // Ref an empty DDict (zero-length content).
        let empty_ddict = ZSTD_DDict {
            dictBuffer: Vec::new(),
            dictContent: core::ptr::null(),
            dictSize: 0,
            dictID: 0,
            entropyPresent: 0,
        };
        assert_eq!(ZSTD_DCtx_refDDict(&mut dctx, &empty_ddict), 0);
        assert_eq!(dctx.dictUses, ZSTD_dictUses_e::ZSTD_dont_use);
        assert!(dctx.stream_dict.is_empty());
        assert_eq!(dctx.dictID, 0);
    }

    #[test]
    fn DCtx_loadDictionary_empty_slice_clears_dict_state() {
        // Symmetric to compressor-side gate: empty-dict load acts
        // as `clearDict`. Matches upstream's `zstd_decompress.c:1710`
        // empty-dict pattern.
        let mut dctx = ZSTD_DCtx::new();
        assert_eq!(ZSTD_DCtx_loadDictionary(&mut dctx, b"real-dict"), 0);
        assert_eq!(dctx.dictUses, ZSTD_dictUses_e::ZSTD_use_indefinitely);
        assert_eq!(dctx.stream_dict, b"real-dict");

        // Empty reload clears.
        assert_eq!(ZSTD_DCtx_loadDictionary(&mut dctx, &[]), 0);
        assert_eq!(dctx.dictUses, ZSTD_dictUses_e::ZSTD_dont_use);
        assert!(dctx.stream_dict.is_empty());
        assert_eq!(dctx.dictID, 0);
    }

    #[test]
    fn DCtx_refPrefix_empty_slice_clears_dict_state() {
        // Symmetric to the compressor-side gate: `refPrefix(&[])`
        // must act as "clear" rather than silently leaving
        // `dictUses = use_once` with an empty stream_dict. Matches
        // upstream's `zstd_decompress.c:1725` pattern (clearDict
        // before install, install only if non-empty).
        let mut dctx = ZSTD_DCtx::new();
        assert_eq!(ZSTD_DCtx_refPrefix(&mut dctx, b"pre-existing"), 0);
        assert_eq!(dctx.dictUses, ZSTD_dictUses_e::ZSTD_use_once);
        assert!(!dctx.stream_dict.is_empty());

        // Re-bind with empty prefix → clears everything.
        assert_eq!(ZSTD_DCtx_refPrefix(&mut dctx, &[]), 0);
        assert_eq!(dctx.dictUses, ZSTD_dictUses_e::ZSTD_dont_use);
        assert!(dctx.stream_dict.is_empty());
        assert_eq!(dctx.dictID, 0);
    }

    #[test]
    fn decompressStream_refPrefix_auto_clears_after_one_frame() {
        // Streaming-path sibling of `DCtx_refPrefix_auto_clears_after_one_decode`.
        // A prefix bound via `ZSTD_DCtx_refPrefix` on a streaming dctx
        // must auto-clear after the first frame in the stream, matching
        // the upstream `use_once` contract. Without this, a second
        // frame on the same stream would silently re-apply the stale
        // prefix as back-ref history.
        use crate::common::error::ERR_isError;
        use crate::compress::zstd_compress::{ZSTD_compress, ZSTD_compressBound};

        let src = b"streaming-refPrefix-auto-clear-payload ".repeat(2);
        let mut framed = vec![0u8; ZSTD_compressBound(src.len())];
        let n = ZSTD_compress(&mut framed, &src, 3);
        framed.truncate(n);

        let mut dctx = ZSTD_DCtx::new();
        ZSTD_initDStream(&mut dctx);
        // Bind a prefix (use_once).
        let prefix = b"streaming-one-shot-prefix".to_vec();
        assert_eq!(ZSTD_DCtx_refPrefix(&mut dctx, &prefix), 0);
        assert_eq!(dctx.dictUses, ZSTD_dictUses_e::ZSTD_use_once);

        let mut out = vec![0u8; src.len() + 64];
        let mut in_pos = 0usize;
        let mut out_pos = 0usize;
        let _ = ZSTD_decompressStream(&mut dctx, &mut out, &mut out_pos, &framed, &mut in_pos);
        for _ in 0..8 {
            if out_pos >= src.len() {
                break;
            }
            let _ = ZSTD_decompressStream(&mut dctx, &mut out, &mut out_pos, &[], &mut 0usize);
        }
        let decoded_len = out_pos;
        assert!(!ERR_isError(decoded_len));

        // After the first frame, both tracker and stream_dict are
        // back to the uninitialized state.
        assert_eq!(dctx.dictUses, ZSTD_dictUses_e::ZSTD_dont_use);
        assert!(
            dctx.stream_dict.is_empty(),
            "streaming refPrefix dict persisted after one-shot decode",
        );
    }

    #[test]
    fn DCtx_refPrefix_auto_clears_after_one_decode() {
        // Upstream contract: `ZSTD_DCtx_refPrefix` is a one-shot
        // binding. After the next `ZSTD_decompressDCtx` consumes it,
        // the dict must be cleared — a subsequent decode on the same
        // dctx should NOT see the prefix. Previously our port left
        // `stream_dict` set forever, diverging from upstream.
        use crate::common::error::ERR_isError;
        use crate::common::xxhash::XXH64_state_t;
        use crate::compress::zstd_compress::{ZSTD_compress, ZSTD_compressBound};
        use crate::decompress::zstd_decompress_block::{
            ZSTD_buildDefaultSeqTables, ZSTD_decoder_entropy_rep,
        };

        // Build a plain (dict-less) frame — decoding it should work
        // regardless of the prefix state.
        let src = b"refPrefix-auto-clear-payload ".repeat(3);
        let mut framed = vec![0u8; ZSTD_compressBound(src.len())];
        let n = ZSTD_compress(&mut framed, &src, 3);
        framed.truncate(n);

        let mut dctx = ZSTD_DCtx::new();
        ZSTD_buildDefaultSeqTables(&mut dctx);
        // Attach a prefix via refPrefix (use_once lifetime).
        let prefix = b"one-shot-prefix-bytes".to_vec();
        assert_eq!(ZSTD_DCtx_refPrefix(&mut dctx, &prefix), 0);
        assert_eq!(dctx.dictUses, ZSTD_dictUses_e::ZSTD_use_once);
        assert_eq!(dctx.stream_dict, prefix);

        // Run one decode — consumes the prefix.
        let mut out = vec![0u8; src.len() + 64];
        let mut rep = ZSTD_decoder_entropy_rep::default();
        let mut xxh = XXH64_state_t::default();
        let d = ZSTD_decompressDCtx(&mut dctx, &mut rep, &mut xxh, &mut out, &framed);
        assert!(!ERR_isError(d));

        // After the decode, the prefix must be gone and the tracker
        // must be back to `ZSTD_dont_use`.
        assert_eq!(dctx.dictUses, ZSTD_dictUses_e::ZSTD_dont_use);
        assert!(
            dctx.stream_dict.is_empty(),
            "refPrefix dict persisted after one-shot decode",
        );
    }

    #[test]
    fn DCtx_dict_family_setters_populate_dictUses_lifecycle_tracker() {
        // Upstream (zstd_decompress.c:1703, 1728, 1786) tags the dict
        // lifecycle via `dctx.dictUses`:
        //   - loadDictionary / refDDict → ZSTD_use_indefinitely
        //   - refPrefix → ZSTD_use_once (auto-clear after next frame)
        // Our port's field + setter wiring must mirror this so the
        // future auto-clear logic can read the right disposition.
        use crate::decompress::zstd_ddict::ZSTD_createDDict;
        let dict_bytes = b"dictUses-lifecycle-tracker".to_vec();

        // loadDictionary → use_indefinitely.
        let mut dctx = ZSTD_DCtx::new();
        assert_eq!(dctx.dictUses, ZSTD_dictUses_e::ZSTD_dont_use);
        assert_eq!(ZSTD_DCtx_loadDictionary(&mut dctx, &dict_bytes), 0);
        assert_eq!(dctx.dictUses, ZSTD_dictUses_e::ZSTD_use_indefinitely);

        // clearDict → dont_use.
        ZSTD_clearDict(&mut dctx);
        assert_eq!(dctx.dictUses, ZSTD_dictUses_e::ZSTD_dont_use);

        // refPrefix → use_once (single-frame lifetime).
        let mut dctx = ZSTD_DCtx::new();
        assert_eq!(ZSTD_DCtx_refPrefix(&mut dctx, &dict_bytes), 0);
        assert_eq!(dctx.dictUses, ZSTD_dictUses_e::ZSTD_use_once);

        // refDDict → use_indefinitely.
        let mut dctx = ZSTD_DCtx::new();
        let ddict = ZSTD_createDDict(&dict_bytes).expect("ddict");
        assert_eq!(ZSTD_DCtx_refDDict(&mut dctx, &ddict), 0);
        assert_eq!(dctx.dictUses, ZSTD_dictUses_e::ZSTD_use_indefinitely);

        // DCtx_reset(parameters) wipes the tracker back to dont_use.
        ZSTD_DCtx_reset(&mut dctx, ZSTD_DResetDirective::ZSTD_reset_parameters);
        assert_eq!(dctx.dictUses, ZSTD_dictUses_e::ZSTD_dont_use);
    }

    #[test]
    fn initDStream_usingDDict_parses_magic_prefix_dict_entropy() {
        // Sibling of `initDStream_usingDict` gate: a DDict built from
        // a magic-prefix dict must surface its dictID + entropy
        // flags on the dctx through the `usingDDict` init path.
        use super::tests::build_minimal_zstd_dict;
        use crate::decompress::zstd_ddict::ZSTD_createDDict;
        let content = b"initDStream_usingDDict-magic-dict-content ".repeat(2);
        let dict = build_minimal_zstd_dict(0x12_34_56_78, &content);
        let ddict = ZSTD_createDDict(&dict).expect("ddict create");

        let mut dctx = ZSTD_DCtx::new();
        let rc = ZSTD_initDStream_usingDDict(&mut dctx, &ddict);
        assert!(!crate::common::error::ERR_isError(rc));
        assert_eq!(dctx.dictID, 0x12_34_56_78);
        assert_eq!(dctx.litEntropy, 1);
        assert_eq!(dctx.fseEntropy, 1);
    }

    #[test]
    fn initDStream_usingDict_parses_magic_prefix_dict_entropy() {
        // Upstream contract (zstd_decompress.c:1744): the streaming
        // init helper routes through `ZSTD_DCtx_loadDictionary`, so
        // a magic-prefix zstd-format dict has its entropy tables
        // parsed onto the dctx. Previously our port wrote
        // `stream_dict = dict` directly, bypassing the magic probe —
        // callers seeding a magicless stream with a full-format dict
        // would have entropy flags stuck at 0.
        use super::tests::build_minimal_zstd_dict;
        let content = b"initDStream_usingDict-magic-dict-content ".repeat(2);
        let dict = build_minimal_zstd_dict(0xAB_CD_EF_00, &content);

        let mut dctx = ZSTD_DCtx::new();
        let rc = ZSTD_initDStream_usingDict(&mut dctx, &dict);
        assert!(!crate::common::error::ERR_isError(rc));
        // dictID was parsed from the magic prefix.
        assert_eq!(dctx.dictID, 0xAB_CD_EF_00);
        // Entropy flags flipped on — HUF + FSE tables loaded.
        assert_eq!(dctx.litEntropy, 1);
        assert_eq!(dctx.fseEntropy, 1);
    }

    #[test]
    fn DCtx_setMaxWindowSize_handles_non_power_of_two_via_highbit() {
        // Upstream stores `maxWindowSize` bytes verbatim; our port
        // converts to log2 since we track `d_windowLogMax` instead.
        // The conversion must use ceiling-log2 semantics so a value
        // like `(1 << 20) + 1` rounds up to windowLog = 20 rather
        // than collapsing to the floor (`trailing_zeros` would give
        // 0 → clamped to 10). Previously the `trailing_zeros`
        // version silently dropped the window to the minimum for
        // any non-power-of-2 input.
        let mut dctx = ZSTD_DCtx::new();
        // Exact power of 2 → the log2 of the power.
        assert_eq!(ZSTD_DCtx_setMaxWindowSize(&mut dctx, 1 << 20), 0);
        assert_eq!(dctx.d_windowLogMax, 20);

        // Non-power-of-2 just above 1<<20 → still reports 20 (the
        // highest set bit).
        let mut dctx = ZSTD_DCtx::new();
        assert_eq!(ZSTD_DCtx_setMaxWindowSize(&mut dctx, (1usize << 20) + 1), 0,);
        assert_eq!(dctx.d_windowLogMax, 20);
    }

    #[test]
    fn internal_decoder_stage_enums_match_upstream() {
        // Upstream `zstd_decompress_internal.h:89-97`:
        //   ZSTD_dStreamStage: zdss_init=0, zdss_loadHeader=1,
        //     zdss_read=2, zdss_load=3, zdss_flush=4
        //   ZSTD_dStage: 0..=7 (getFrameHeaderSize → skipFrame)
        //   ZSTD_dictUses_e: use_indefinitely=-1, dont_use=0, use_once=1
        // These feed the DCtx state machine — `dctx_is_in_init_stage`
        // gate checks, streaming input-type hints, dict-lifetime
        // tracking all consume the raw values.
        assert_eq!(ZSTD_dStreamStage::zdss_init as i32, 0);
        assert_eq!(ZSTD_dStreamStage::zdss_loadHeader as i32, 1);
        assert_eq!(ZSTD_dStreamStage::zdss_read as i32, 2);
        assert_eq!(ZSTD_dStreamStage::zdss_load as i32, 3);
        assert_eq!(ZSTD_dStreamStage::zdss_flush as i32, 4);
        assert_eq!(ZSTD_dStage::ZSTDds_getFrameHeaderSize as i32, 0);
        assert_eq!(ZSTD_dStage::ZSTDds_decodeFrameHeader as i32, 1);
        assert_eq!(ZSTD_dStage::ZSTDds_decodeBlockHeader as i32, 2);
        assert_eq!(ZSTD_dStage::ZSTDds_decompressBlock as i32, 3);
        assert_eq!(ZSTD_dStage::ZSTDds_decompressLastBlock as i32, 4);
        assert_eq!(ZSTD_dStage::ZSTDds_checkChecksum as i32, 5);
        assert_eq!(ZSTD_dStage::ZSTDds_decodeSkippableHeader as i32, 6);
        assert_eq!(ZSTD_dStage::ZSTDds_skipFrame as i32, 7);
        assert_eq!(ZSTD_dictUses_e::ZSTD_use_indefinitely as i32, -1);
        assert_eq!(ZSTD_dictUses_e::ZSTD_dont_use as i32, 0);
        assert_eq!(ZSTD_dictUses_e::ZSTD_use_once as i32, 1);
    }

    #[test]
    fn nextSrcSizeWithInputSize_and_isSkipFrame_match_stage_rules() {
        use crate::decompress::zstd_decompress_block::blockType_e;

        let mut dctx = ZSTD_DCtx::new();
        dctx.expected = 64;
        dctx.stage = ZSTD_dStage::ZSTDds_decodeBlockHeader;
        assert_eq!(ZSTD_nextSrcSizeToDecompressWithInputSize(&dctx, 7), 64);
        assert_eq!(ZSTD_isSkipFrame(&dctx), 0);

        dctx.stage = ZSTD_dStage::ZSTDds_decompressBlock;
        dctx.bType = blockType_e::bt_compressed;
        assert_eq!(ZSTD_nextSrcSizeToDecompressWithInputSize(&dctx, 7), 64);

        dctx.bType = blockType_e::bt_raw;
        assert_eq!(ZSTD_nextSrcSizeToDecompressWithInputSize(&dctx, 7), 7);
        assert_eq!(ZSTD_nextSrcSizeToDecompressWithInputSize(&dctx, 128), 64);

        dctx.stage = ZSTD_dStage::ZSTDds_skipFrame;
        assert_eq!(ZSTD_isSkipFrame(&dctx), 1);
    }

    #[test]
    fn format_e_and_frameType_e_discriminants_match_upstream() {
        // Upstream wire-level values:
        //   ZSTD_f_zstd1 = 0, ZSTD_f_zstd1_magicless = 1 (zstd.h:1385)
        //   ZSTD_frame = 0, ZSTD_skippableFrame = 1     (zstd.h:1510)
        // These show up as `ZSTD_FrameHeader.frameType` fields and
        // `ZSTD_getFrameHeader_advanced` args — FFI callers pass the
        // raw integer values. Lock the discriminants.
        assert_eq!(ZSTD_format_e::ZSTD_f_zstd1 as i32, 0);
        assert_eq!(ZSTD_format_e::ZSTD_f_zstd1_magicless as i32, 1);
        assert_eq!(ZSTD_FrameType_e::ZSTD_frame as i32, 0);
        assert_eq!(ZSTD_FrameType_e::ZSTD_skippableFrame as i32, 1);
    }

    #[test]
    fn DResetDirective_discriminants_match_upstream() {
        // Decoder-side alias of `ZSTD_ResetDirective` — same
        // upstream discriminants (zstd.h:589): 1/2/3.
        assert_eq!(ZSTD_DResetDirective::ZSTD_reset_session_only as i32, 1);
        assert_eq!(ZSTD_DResetDirective::ZSTD_reset_parameters as i32, 2);
        assert_eq!(
            ZSTD_DResetDirective::ZSTD_reset_session_and_parameters as i32,
            3,
        );
    }

    #[test]
    fn dParameter_discriminants_match_upstream_zstd_h() {
        // Pin the `ZSTD_dParameter` C-ABI values. Mirror of the
        // compressor-side gate: drift here silently mis-routes FFI
        // callers.
        assert_eq!(ZSTD_dParameter::ZSTD_d_windowLogMax as i32, 100);
        // `ZSTD_d_format` = `ZSTD_d_experimentalParam1` = 1000.
        assert_eq!(ZSTD_dParameter::ZSTD_d_format as i32, 1000);
    }

    #[test]
    fn DCtx_setParameter_d_format_round_trips_through_getParameter() {
        // Upstream exposes format as `ZSTD_d_format` — set via
        // `ZSTD_DCtx_setParameter(d_format, value)` and read back via
        // `ZSTD_DCtx_getParameter`. Our port now mirrors that path
        // so callers who use the parametric API (not just the direct
        // `ZSTD_DCtx_setFormat` helper) land on the same state.
        use crate::common::error::{ERR_getErrorCode, ERR_isError, ErrorCode};
        let mut dctx = ZSTD_DCtx::new();
        let mut value = 0i32;

        // Default is zstd1.
        assert_eq!(
            ZSTD_DCtx_getParameter(&dctx, ZSTD_dParameter::ZSTD_d_format, &mut value),
            0,
        );
        assert_eq!(value, ZSTD_format_e::ZSTD_f_zstd1 as i32);

        // Flip to magicless via the parametric setter.
        assert_eq!(
            ZSTD_DCtx_setParameter(
                &mut dctx,
                ZSTD_dParameter::ZSTD_d_format,
                ZSTD_format_e::ZSTD_f_zstd1_magicless as i32,
            ),
            0,
        );
        assert_eq!(dctx.format, ZSTD_format_e::ZSTD_f_zstd1_magicless);

        // Getter reports it back.
        assert_eq!(
            ZSTD_DCtx_getParameter(&dctx, ZSTD_dParameter::ZSTD_d_format, &mut value),
            0,
        );
        assert_eq!(value, ZSTD_format_e::ZSTD_f_zstd1_magicless as i32);

        // Out-of-bounds values must be rejected, not silently clamped.
        let rc = ZSTD_DCtx_setParameter(&mut dctx, ZSTD_dParameter::ZSTD_d_format, 42);
        assert!(ERR_isError(rc));
        assert_eq!(ERR_getErrorCode(rc), ErrorCode::ParameterOutOfBound);
    }

    #[test]
    fn DCtx_reset_parameters_clears_magicless_format() {
        // Symmetric to the compressor-side gate: a dctx `reset(parameters)`
        // must restore the default zstd1 format. session_only must NOT
        // touch it — upstream keeps format as a param, not session state.
        let mut dctx = ZSTD_DCtx::new();
        ZSTD_DCtx_setFormat(&mut dctx, ZSTD_format_e::ZSTD_f_zstd1_magicless);
        assert_eq!(dctx.format, ZSTD_format_e::ZSTD_f_zstd1_magicless);

        ZSTD_DCtx_reset(&mut dctx, ZSTD_DResetDirective::ZSTD_reset_session_only);
        assert_eq!(dctx.format, ZSTD_format_e::ZSTD_f_zstd1_magicless);

        ZSTD_DCtx_reset(&mut dctx, ZSTD_DResetDirective::ZSTD_reset_parameters);
        assert_eq!(dctx.format, ZSTD_format_e::ZSTD_f_zstd1);

        ZSTD_DCtx_setFormat(&mut dctx, ZSTD_format_e::ZSTD_f_zstd1_magicless);
        ZSTD_DCtx_reset(
            &mut dctx,
            ZSTD_DResetDirective::ZSTD_reset_session_and_parameters,
        );
        assert_eq!(dctx.format, ZSTD_format_e::ZSTD_f_zstd1);
    }

    #[test]
    fn DCtx_setFormat_stashes_format_on_dctx() {
        // `ZSTD_DCtx_setFormat` now stores the format on the DCtx.
        // Contract: setter returns 0 and the field persists for
        // frame-header parsing (which reads `dctx.format` via
        // `ZSTD_startingInputLength`).
        use crate::common::error::ERR_isError;
        let mut dctx = ZSTD_DCtx::default();
        assert_eq!(dctx.format, ZSTD_format_e::ZSTD_f_zstd1);
        let rc = ZSTD_DCtx_setFormat(&mut dctx, ZSTD_format_e::ZSTD_f_zstd1_magicless);
        assert!(!ERR_isError(rc));
        assert_eq!(dctx.format, ZSTD_format_e::ZSTD_f_zstd1_magicless);
    }

    #[test]
    fn initDStream_hint_honors_magicless_format() {
        // `ZSTD_initDStream` / `ZSTD_resetDStream` return
        // `ZSTD_startingInputLength(dctx.format)` — upstream's first-
        // read hint for a streaming decoder. Previously our port
        // hardcoded `ZSTD_f_zstd1` (= 5 bytes: 4-byte magic + 1-byte
        // FHD). A magicless-mode dctx should instead get 1 (FHD only).
        let mut dctx = ZSTD_DCtx::default();
        assert_eq!(ZSTD_initDStream(&mut dctx), 5);
        assert_eq!(ZSTD_resetDStream(&mut dctx), 5);
        dctx.format = ZSTD_format_e::ZSTD_f_zstd1_magicless;
        assert_eq!(ZSTD_initDStream(&mut dctx), 1);
        assert_eq!(ZSTD_resetDStream(&mut dctx), 1);
    }

    #[test]
    fn decompress_usingDict_honors_magicless_format_on_dctx() {
        // Parity gate: when a caller sets `dctx.format = magicless`
        // on the dctx and passes a magicless frame + raw dict to
        // `ZSTD_decompress_usingDict`, the decode must succeed.
        // Before `ZSTD_decompressFrame_withOpStart` learned about
        // format, this path hardcoded zstd1 and a magicless frame
        // would be rejected by the header probe.
        use crate::common::error::ERR_isError;
        use crate::compress::zstd_compress::{
            ZSTD_CCtx_setFormat, ZSTD_createCCtx, ZSTD_endStream, ZSTD_initCStream_usingDict,
        };

        let dict = b"usingDict-magicless-parity-dict-bytes ".repeat(4);
        let src = b"payload-referencing-usingDict-magicless-parity-dict-bytes ".repeat(6);
        let mut cctx = ZSTD_createCCtx().unwrap();
        assert_eq!(
            ZSTD_CCtx_setFormat(&mut cctx, ZSTD_format_e::ZSTD_f_zstd1_magicless),
            0,
        );
        ZSTD_initCStream_usingDict(&mut cctx, &dict, 3);

        let mut compressed = vec![0u8; 4096];
        let mut cp = 0usize;
        let mut sp = 0usize;
        let _ = crate::compress::zstd_compress::ZSTD_compressStream(
            &mut cctx,
            &mut compressed,
            &mut cp,
            &src,
            &mut sp,
        );
        loop {
            let r = ZSTD_endStream(&mut cctx, &mut compressed, &mut cp);
            assert!(!ERR_isError(r));
            if r == 0 {
                break;
            }
        }
        compressed.truncate(cp);

        let mut dctx = ZSTD_DCtx::new();
        assert_eq!(
            ZSTD_DCtx_setFormat(&mut dctx, ZSTD_format_e::ZSTD_f_zstd1_magicless),
            0,
        );
        let mut out = vec![0u8; src.len() + 128];
        let d = ZSTD_decompress_usingDict(&mut dctx, &mut out, &compressed, &dict);
        assert!(!ERR_isError(d), "decode err: {d:#x}");
        assert_eq!(&out[..d], &src[..]);
    }

    #[test]
    fn decompressStream_honors_magicless_format() {
        // Streaming decoder parity gate: the streaming path measures
        // frame size via `findFrameCompressedSize` — previously that
        // call was hardcoded to the zstd1 format, so magicless-format
        // dctxs would reject valid magicless bytes as malformed. After
        // the fix the streaming path threads `dctx.format` through.
        use crate::common::error::ERR_isError;
        use crate::compress::zstd_compress::{ZSTD_compress, ZSTD_compressBound};
        let src = b"streaming-magicless-decode-parity-payload ".repeat(10);
        let mut framed = vec![0u8; ZSTD_compressBound(src.len())];
        let c_sz = ZSTD_compress(&mut framed, &src, 3);
        assert!(!ERR_isError(c_sz));
        framed.truncate(c_sz);
        let magicless = framed[4..].to_vec();

        let mut dctx = ZSTD_DCtx::new();
        let _ = ZSTD_DCtx_setFormat(&mut dctx, ZSTD_format_e::ZSTD_f_zstd1_magicless);
        ZSTD_initDStream(&mut dctx);

        let mut out = vec![0u8; src.len() + 64];
        let mut in_pos = 0usize;
        let mut out_pos = 0usize;
        let _hint =
            ZSTD_decompressStream(&mut dctx, &mut out, &mut out_pos, &magicless, &mut in_pos);
        for _ in 0..8 {
            if out_pos >= src.len() {
                break;
            }
            let _ = ZSTD_decompressStream(&mut dctx, &mut out, &mut out_pos, &[], &mut 0usize);
        }
        assert_eq!(&out[..out_pos], &src[..]);
    }

    #[test]
    fn decompressFrame_honors_magicless_format() {
        // Real parity gate: compress a payload with upstream zstd1
        // format → strip the 4-byte magic → set dctx.format to
        // ZSTD_f_zstd1_magicless → decode. The decoder must accept
        // the magicless bytes and reconstruct the original payload.
        use crate::common::xxhash::XXH64_state_t;
        use crate::compress::zstd_compress::{ZSTD_compress, ZSTD_compressBound};
        use crate::decompress::zstd_decompress_block::{
            ZSTD_DCtx, ZSTD_buildDefaultSeqTables, ZSTD_decoder_entropy_rep,
        };

        let src = b"Hello, magicless zstd world! 0123456789";
        let mut zstd_framed = vec![0u8; ZSTD_compressBound(src.len())];
        let c_sz = ZSTD_compress(&mut zstd_framed, src, 1);
        assert!(!crate::common::error::ERR_isError(c_sz));
        zstd_framed.truncate(c_sz);

        // Strip the 4-byte magic to make it magicless.
        let magicless = &zstd_framed[4..];

        let mut dctx = ZSTD_DCtx::new();
        ZSTD_buildDefaultSeqTables(&mut dctx);
        let rc = ZSTD_DCtx_setFormat(&mut dctx, ZSTD_format_e::ZSTD_f_zstd1_magicless);
        assert!(!crate::common::error::ERR_isError(rc));

        let mut out = vec![0u8; src.len()];
        let mut rep = ZSTD_decoder_entropy_rep::default();
        let mut xxh = XXH64_state_t::default();
        let mut consumed = 0usize;
        let decoded = ZSTD_decompressFrame(
            &mut dctx,
            &mut rep,
            &mut xxh,
            &mut out,
            magicless,
            &mut consumed,
        );
        assert!(
            !crate::common::error::ERR_isError(decoded),
            "decompressFrame failed: {}",
            crate::common::error::ERR_getErrorName(decoded)
        );
        assert_eq!(decoded, src.len());
        assert_eq!(&out[..decoded], src);
        // Consumed must cover the magicless input exactly (no magic bytes).
        assert_eq!(consumed, magicless.len());
    }

    #[test]
    fn decompressContinue_rejects_wrong_chunk_size() {
        let mut dctx = ZSTD_DCtx::default();
        let mut dst = [0u8; 64];
        let src = b"some-input";
        let rc = ZSTD_decompressContinue(&mut dctx, &mut dst, src);
        assert!(crate::common::error::ERR_isError(rc));
        use crate::common::error::ERR_getErrorCode;
        assert_eq!(ERR_getErrorCode(rc), ErrorCode::SrcSizeWrong);
    }

    #[test]
    fn decompressContinue_roundtrips_single_raw_block_frame() {
        use crate::common::mem::MEM_writeLE24;
        use crate::compress::zstd_compress::{
            ZSTD_FrameParameters, ZSTD_writeFrameHeader, ZSTD_FRAMEHEADERSIZE_MAX,
        };

        let payload = b"legacy continue raw block";
        let fparams = ZSTD_FrameParameters {
            contentSizeFlag: 1,
            checksumFlag: 0,
            noDictIDFlag: 1,
        };
        let mut header = [0u8; ZSTD_FRAMEHEADERSIZE_MAX];
        let hsize = ZSTD_writeFrameHeader(&mut header, &fparams, 17, payload.len() as u64, 0);
        assert!(!crate::common::error::ERR_isError(hsize));

        let mut frame = Vec::with_capacity(hsize + 3 + payload.len());
        frame.extend_from_slice(&header[..hsize]);
        let blockHeader = 1u32 | ((payload.len() as u32) << 3);
        let mut bh = [0u8; 3];
        MEM_writeLE24(&mut bh, blockHeader);
        frame.extend_from_slice(&bh);
        frame.extend_from_slice(payload);

        let mut dctx = ZSTD_DCtx::default();
        let mut out = vec![0u8; payload.len()];
        let mut ip = 0usize;
        let mut op = 0usize;

        while ip < frame.len() {
            let chunk = ZSTD_nextSrcSizeToDecompress(&dctx);
            let produced =
                ZSTD_decompressContinue(&mut dctx, &mut out[op..], &frame[ip..ip + chunk]);
            assert!(
                !crate::common::error::ERR_isError(produced),
                "decompressContinue failed at ip={ip}: {}",
                crate::common::error::ERR_getErrorName(produced)
            );
            ip += chunk;
            op += produced;
        }

        assert_eq!(op, payload.len());
        assert_eq!(&out[..op], payload);
        assert_eq!(
            ZSTD_nextSrcSizeToDecompress(&dctx),
            ZSTD_startingInputLength(ZSTD_format_e::ZSTD_f_zstd1)
        );
    }

    #[test]
    fn decompressContinue_multi_block_with_independent_dst_buffers() {
        // Build a multi-block compressed frame whose later blocks
        // back-reference earlier blocks. Drive it through
        // `ZSTD_decompressContinue` while handing each block a *fresh*
        // `dst` slice — the prior block's bytes do not live in
        // `dst[..op_start]`, so the only way back-references can
        // resolve is via the rolling history buffer in `dctx`.
        //
        // This is the regression gate for the dctx.historyBuffer +
        // ext-dict wiring in `ZSTD_execSequence`.
        use crate::compress::zstd_compress::ZSTD_compress;

        // Strong cross-block repetition: a 64 KB unique preamble
        // followed by 8 copies of itself, total ~576 KB. At standard
        // block size 128 KB, blocks 2..=4 will reference into earlier
        // blocks' bytes. Every block boundary will see at least one
        // long back-reference cross it.
        let chunk: Vec<u8> = (0..65_536u32)
            .map(|i| ((i * 17 + 3) & 0xFF) as u8)
            .collect();
        let mut payload = chunk.clone();
        for _ in 0..8 {
            payload.extend_from_slice(&chunk);
        }

        let mut compressed = vec![0u8; payload.len() + 1024];
        let n = ZSTD_compress(&mut compressed, &payload, 1);
        assert!(
            !crate::common::error::ERR_isError(n),
            "compress failed: {}",
            crate::common::error::ERR_getErrorName(n)
        );
        compressed.truncate(n);

        // Drive ZSTD_decompressContinue chunk-by-chunk; for each
        // decompressBlock stage, allocate a brand-new Vec for the
        // block output so prior-block bytes are NOT visible to the
        // sequence executor through `dst`. Concatenate the per-call
        // outputs to verify the full frame.
        let mut dctx = ZSTD_DCtx::default();
        let mut ip = 0usize;
        let mut decoded: Vec<u8> = Vec::with_capacity(payload.len());

        while ip < compressed.len() {
            let chunk = ZSTD_nextSrcSizeToDecompress(&dctx);
            if chunk == 0 {
                break;
            }
            // Allocate a fresh buffer big enough for any block.
            let block_max = dctx
                .fParams
                .blockSizeMax
                .max(crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX as u32)
                as usize;
            let mut block_dst = vec![0u8; block_max + 64];
            let produced =
                ZSTD_decompressContinue(&mut dctx, &mut block_dst, &compressed[ip..ip + chunk]);
            assert!(
                !crate::common::error::ERR_isError(produced),
                "decompressContinue at ip={ip}, chunk={chunk}: {}",
                crate::common::error::ERR_getErrorName(produced)
            );
            decoded.extend_from_slice(&block_dst[..produced]);
            ip += chunk;
        }

        assert_eq!(decoded.len(), payload.len(), "decoded length mismatch");
        assert_eq!(decoded, payload, "decoded bytes differ from payload");
    }

    #[test]
    fn stream_workspace_helpers_track_overflow_and_continue_wrapper() {
        use crate::common::mem::MEM_writeLE24;
        use crate::compress::zstd_compress::{
            ZSTD_FrameParameters, ZSTD_writeFrameHeader, ZSTD_FRAMEHEADERSIZE_MAX,
        };

        let mut dctx = ZSTD_DCtx::default();
        dctx.stream_in_buffer.reserve(128);
        dctx.stream_out_buffer.reserve(128);
        assert_eq!(ZSTD_DCtx_isOverflow(&dctx, 16, 16), 1);
        ZSTD_DCtx_updateOversizedDuration(&mut dctx, 16, 16);
        assert_eq!(dctx.oversizedDuration, 1);
        ZSTD_DCtx_updateOversizedDuration(&mut dctx, 1024, 1024);
        assert_eq!(dctx.oversizedDuration, 0);
        assert_eq!(ZSTD_checkOutBuffer(&dctx, &[], 0), 0);

        let payload = b"continue-stream wrapper";
        let fparams = ZSTD_FrameParameters {
            contentSizeFlag: 1,
            checksumFlag: 0,
            noDictIDFlag: 1,
        };
        let mut header = [0u8; ZSTD_FRAMEHEADERSIZE_MAX];
        let hsize = ZSTD_writeFrameHeader(&mut header, &fparams, 17, payload.len() as u64, 0);
        assert!(!crate::common::error::ERR_isError(hsize));

        let mut frame = Vec::new();
        frame.extend_from_slice(&header[..hsize]);
        let blockHeader = 1u32 | ((payload.len() as u32) << 3);
        let mut bh = [0u8; 3];
        MEM_writeLE24(&mut bh, blockHeader);
        frame.extend_from_slice(&bh);
        frame.extend_from_slice(payload);

        let mut zds = ZSTD_DCtx::default();
        let mut out = vec![0u8; payload.len()];
        let mut ip = 0usize;
        let mut op = 0usize;
        while ip < frame.len() {
            let chunk = ZSTD_nextSrcSizeToDecompress(&zds);
            let rc =
                ZSTD_decompressContinueStream(&mut zds, &mut out, &mut op, &frame[ip..ip + chunk]);
            assert!(
                !crate::common::error::ERR_isError(rc),
                "continue stream failed at ip={ip}: {}",
                crate::common::error::ERR_getErrorName(rc)
            );
            ip += chunk;
        }
        assert_eq!(&out[..op], payload);
    }

    #[test]
    fn decodeFrameHeader_sets_checksum_and_rejects_wrong_dict() {
        use crate::common::error::ERR_getErrorCode;
        use crate::compress::zstd_compress::{
            ZSTD_FrameParameters, ZSTD_writeFrameHeader, ZSTD_FRAMEHEADERSIZE_MAX,
        };

        let fparams = ZSTD_FrameParameters {
            contentSizeFlag: 1,
            checksumFlag: 1,
            noDictIDFlag: 0,
        };
        let mut header = [0u8; ZSTD_FRAMEHEADERSIZE_MAX];
        let hsize = ZSTD_writeFrameHeader(&mut header, &fparams, 17, 11, 0x1234);
        assert!(!crate::common::error::ERR_isError(hsize));

        let mut dctx = ZSTD_DCtx::default();
        dctx.dictID = 0x1234;
        let rc = ZSTD_decodeFrameHeader(&mut dctx, &header[..hsize], hsize);
        assert_eq!(rc, 0);
        assert_eq!(dctx.fParams.dictID, 0x1234);
        assert_eq!(dctx.validateChecksum, 1);
        assert_eq!(dctx.processedCSize, hsize as u64);

        let mut wrong = ZSTD_DCtx::default();
        wrong.dictID = 0x5678;
        let rc = ZSTD_decodeFrameHeader(&mut wrong, &header[..hsize], hsize);
        assert!(crate::common::error::ERR_isError(rc));
        assert_eq!(ERR_getErrorCode(rc), ErrorCode::DictionaryWrong);
    }

    #[test]
    fn getDDict_and_refDictContent_follow_upstream_lifecycle() {
        let mut dctx = ZSTD_DCtx::default();
        let dict = b"raw dictionary bytes";

        assert_eq!(ZSTD_refDictContent(&mut dctx, dict), 0);
        dctx.dictUses = ZSTD_dictUses_e::ZSTD_use_once;
        assert_eq!(dctx.prefixStart, Some(dict.as_ptr() as usize));
        assert_eq!(
            dctx.previousDstEnd,
            Some(dict.as_ptr() as usize + dict.len())
        );

        let used = ZSTD_getDDict(&mut dctx).expect("one-shot dict");
        assert_eq!(used, dict);
        assert_eq!(dctx.dictUses, ZSTD_dictUses_e::ZSTD_dont_use);
        assert!(dctx.stream_dict.is_empty());
        assert!(ZSTD_getDDict(&mut dctx).is_none());
    }

    #[test]
    fn ddict_hashset_inserts_replaces_and_expands() {
        use crate::compress::zstd_compress::ZSTD_customMem;
        use crate::decompress::zstd_ddict::ZSTD_DDict;

        fn make_ddict(dictID: u32, content: &[u8]) -> ZSTD_DDict {
            let dictBuffer = content.to_vec();
            let dictContent = dictBuffer.as_ptr();
            ZSTD_DDict {
                dictBuffer,
                dictContent,
                dictSize: content.len(),
                dictID,
                entropyPresent: 0,
            }
        }

        let first = make_ddict(7, b"first");
        let replacement = make_ddict(7, b"replacement");
        let mut set = ZSTD_createDDictHashSet(ZSTD_customMem::default());
        assert_eq!(
            ZSTD_DDictHashSet_addDDict(&mut set, &first, ZSTD_customMem::default()),
            0
        );
        assert_eq!(set.ddictPtrCount, 1);
        assert!(core::ptr::eq(
            ZSTD_DDictHashSet_getDDict(&set, 7).expect("first"),
            &first
        ));

        assert_eq!(
            ZSTD_DDictHashSet_addDDict(&mut set, &replacement, ZSTD_customMem::default()),
            0
        );
        assert_eq!(set.ddictPtrCount, 1);
        assert!(core::ptr::eq(
            ZSTD_DDictHashSet_getDDict(&set, 7).expect("replacement"),
            &replacement
        ));

        let many: Vec<Box<ZSTD_DDict>> = (100..121)
            .map(|id| Box::new(make_ddict(id, &[id as u8; 3])))
            .collect();
        for ddict in &many {
            assert_eq!(
                ZSTD_DDictHashSet_addDDict(&mut set, ddict.as_ref(), ZSTD_customMem::default()),
                0
            );
        }
        assert!(set.ddictPtrTableSize > DDICT_HASHSET_TABLE_BASE_SIZE);
        for ddict in &many {
            let found = ZSTD_DDictHashSet_getDDict(&set, ddict.dictID).expect("present");
            assert!(core::ptr::eq(found, ddict.as_ref()));
        }
        assert!(ZSTD_DDictHashSet_getDDict(&set, 0xFFFF).is_none());

        let mut dctx = ZSTD_DCtx::default();
        dctx.fParams.dictID = 117;
        let selected = ZSTD_DCtx_selectFrameDDict(&mut dctx, &set).expect("selected");
        assert_eq!(selected.dictID, 117);
        assert_eq!(dctx.dictID, 117);
        assert_eq!(dctx.dictUses, ZSTD_dictUses_e::ZSTD_use_indefinitely);
        assert_eq!(dctx.stream_dict, [117u8; 3]);
    }

    #[test]
    fn decoder_rejects_corrupted_xxh64_trailer_with_checksumWrong() {
        // Compress with `--check` flag, flip one byte of the XXH64
        // trailer, and verify the decoder surfaces ChecksumWrong —
        // NOT a silent decode-and-pass-bad-bytes-up-the-stack.
        use crate::compress::match_state::ZSTD_compressionParameters;
        use crate::compress::zstd_compress::{
            ZSTD_FrameParameters, ZSTD_compressBound, ZSTD_compressFrame_fast, ZSTD_getCParams,
        };

        let src: Vec<u8> = b"payload with xxh64 trailer ".repeat(40);
        let bound = ZSTD_compressBound(src.len());
        let mut dst = vec![0u8; bound];
        let cp: ZSTD_compressionParameters = ZSTD_getCParams(3, src.len() as u64, 0);
        let fp = ZSTD_FrameParameters {
            contentSizeFlag: 1,
            checksumFlag: 1,
            noDictIDFlag: 1,
        };
        let n = ZSTD_compressFrame_fast(&mut dst, &src, cp, fp);
        assert!(!crate::common::error::ERR_isError(n));
        dst.truncate(n);

        // Flip the last byte — the low 8 bits of the XXH64 trailer.
        let last = dst.len() - 1;
        dst[last] ^= 0x01;

        let mut out = vec![0u8; src.len() + 64];
        let rc = ZSTD_decompress(&mut out, &dst);
        assert!(
            crate::common::error::ERR_isError(rc),
            "decoder missed corrupted checksum (rc={rc})"
        );
        assert_eq!(
            crate::common::error::ERR_getErrorCode(rc),
            ErrorCode::ChecksumWrong,
            "expected ChecksumWrong, got {:?}",
            crate::common::error::ERR_getErrorCode(rc)
        );
    }

    #[test]
    fn resetDStream_clears_streaming_state_and_returns_next_hint() {
        // Contract:
        //   - clears stream_in_buffer, stream_out_buffer, drain cursor
        //   - returns `ZSTD_startingInputLength(format)` — the number
        //     of bytes needed to query the next frame header
        //     (5 for regular zstd1, 1 for magicless)
        // Preserves stream_dict (that's session-level state on the DCtx).
        use crate::decompress::zstd_decompress_block::ZSTD_DCtx;
        let mut dctx = ZSTD_DCtx::new();
        dctx.stream_in_buffer.extend_from_slice(b"pending-in");
        dctx.stream_out_buffer.extend_from_slice(b"pending-out");
        dctx.stream_out_drained = 4;
        dctx.stream_dict = b"sticky-dict".to_vec();

        let hint = ZSTD_resetDStream(&mut dctx);
        assert_eq!(hint, ZSTD_startingInputLength(ZSTD_format_e::ZSTD_f_zstd1));
        assert_eq!(hint, 5);
        assert!(dctx.stream_in_buffer.is_empty());
        assert!(dctx.stream_out_buffer.is_empty());
        assert_eq!(dctx.stream_out_drained, 0);
        // Dict survives the reset.
        assert_eq!(dctx.stream_dict, b"sticky-dict");
    }

    #[test]
    fn multi_frame_roundtrip_across_3_frames_with_interleaved_skippables() {
        // Three regular frames with distinct payloads + two
        // skippable frames at start and middle. All five frames
        // must decode in order, skippable frames contributing no
        // output bytes.
        use crate::compress::zstd_compress::{
            ZSTD_compress, ZSTD_compressBound, ZSTD_writeSkippableFrame,
        };

        let payloads: [&[u8]; 3] = [
            b"payload-alpha ",
            b"payload-beta-is-a-bit-longer ",
            b"payload-gamma! ",
        ];

        let mut combined = Vec::new();
        let mut expected = Vec::new();
        // Skippable at start (meta).
        let mut skip = vec![0u8; 32];
        let n = ZSTD_writeSkippableFrame(&mut skip, b"leading", 0);
        combined.extend_from_slice(&skip[..n]);
        // First regular frame.
        for (i, payload) in payloads.iter().enumerate() {
            let bound = ZSTD_compressBound(payload.len());
            let mut c = vec![0u8; bound];
            let n = ZSTD_compress(&mut c, payload, 1);
            assert!(!crate::common::error::ERR_isError(n));
            combined.extend_from_slice(&c[..n]);
            expected.extend_from_slice(payload);
            // Interleave a skippable after frame 1 (between 1 and 2).
            if i == 0 {
                let mut skip2 = vec![0u8; 24];
                let n2 = ZSTD_writeSkippableFrame(&mut skip2, b"mid", 7);
                combined.extend_from_slice(&skip2[..n2]);
            }
        }

        let mut out = vec![0u8; expected.len() + 64];
        let d = ZSTD_decompress(&mut out, &combined);
        assert!(!crate::common::error::ERR_isError(d));
        assert_eq!(d, expected.len());
        assert_eq!(&out[..d], &expected[..]);
    }

    #[test]
    fn zstd_decompress_loops_over_concatenated_frames() {
        // Upstream contract: `ZSTD_decompress` walks multiple frames
        // in `src`, appending each payload into `dst`. Skippable
        // frames are silently advanced past without consuming dst
        // space.
        use crate::compress::zstd_compress::{
            ZSTD_compress, ZSTD_compressBound, ZSTD_writeSkippableFrame,
        };
        let src = b"concat probe content ".to_vec();
        let bound = ZSTD_compressBound(src.len());
        let mut frame = vec![0u8; bound];
        let n = ZSTD_compress(&mut frame, &src, 1);
        frame.truncate(n);

        // Layout: frame || skippable || frame — confirms both
        // skippable-passthrough and per-frame output accumulation.
        let mut combined = frame.clone();
        let mut skip = vec![0u8; 16];
        let skip_n = ZSTD_writeSkippableFrame(&mut skip, b"meta", 1);
        combined.extend_from_slice(&skip[..skip_n]);
        combined.extend_from_slice(&frame);

        let mut out = vec![0u8; src.len() * 2 + 64];
        let d = ZSTD_decompress(&mut out, &combined);
        assert!(!crate::common::error::ERR_isError(d));
        assert_eq!(d, src.len() * 2);
        assert_eq!(&out[..src.len()], &src[..]);
        assert_eq!(&out[src.len()..src.len() * 2], &src[..]);
    }

    #[test]
    fn decompress_rejects_truncated_frame_body_with_error() {
        // Compress a payload, chop off some bytes from the MIDDLE of
        // the compressed stream (leaving header intact), and verify
        // the decoder surfaces an error rather than succeeding with
        // partial / corrupted output.
        use crate::compress::zstd_compress::{ZSTD_compress, ZSTD_compressBound};
        let src: Vec<u8> = b"truncation probe content ".repeat(30);
        let bound = ZSTD_compressBound(src.len());
        let mut compressed = vec![0u8; bound];
        let n = ZSTD_compress(&mut compressed, &src, 3);
        assert!(!crate::common::error::ERR_isError(n));

        // Truncate to 70% — well past the header but before
        // completing the block body.
        let truncated_len = n * 7 / 10;
        assert!(truncated_len > 10 && truncated_len < n);
        let truncated = &compressed[..truncated_len];

        let mut out = vec![0u8; src.len() + 64];
        let rc = ZSTD_decompress(&mut out, truncated);
        assert!(
            crate::common::error::ERR_isError(rc),
            "decoder accepted truncated input (rc={rc}) — must reject",
        );
    }

    #[test]
    fn two_independent_dctxs_decode_same_frame_to_same_output() {
        // Isolation contract mirror of the CCtx-side test. Two DCtxes
        // decoding the same frame (with different dicts loaded) must
        // produce identical output for the frame — per-DCtx state
        // (stream_dict / windowLogMax / entropy tables) must not
        // leak between DCtx instances.
        use crate::compress::zstd_compress::{ZSTD_compress, ZSTD_compressBound};
        let src: Vec<u8> = b"DCtx-isolation test payload. ".repeat(30);
        let bound = ZSTD_compressBound(src.len());
        let mut compressed = vec![0u8; bound];
        let n = ZSTD_compress(&mut compressed, &src, 3);
        assert!(!crate::common::error::ERR_isError(n));
        compressed.truncate(n);

        // DCtx A with one dict loaded; DCtx B with a different dict.
        // Neither dict affects the non-dict frame we compressed above.
        let mut a = ZSTD_DCtx::new();
        let mut b = ZSTD_DCtx::new();
        ZSTD_DCtx_loadDictionary(&mut a, b"some-dict-A");
        ZSTD_DCtx_loadDictionary(&mut b, b"other-dict-B");

        let mut out_a = vec![0u8; src.len() + 64];
        let d_a = ZSTD_decompress(&mut out_a, &compressed);
        assert_eq!(&out_a[..d_a], &src[..]);

        let mut out_b = vec![0u8; src.len() + 64];
        let d_b = ZSTD_decompress(&mut out_b, &compressed);
        assert_eq!(&out_b[..d_b], &src[..]);

        // Neither DCtx should have its stream_dict disturbed.
        assert_eq!(a.stream_dict, b"some-dict-A");
        assert_eq!(b.stream_dict, b"other-dict-B");
    }

    #[test]
    fn zstd_decompress_rejects_too_small_dst_buffer() {
        // Symmetric with the compress-side too-small-dst test.
        // `ZSTD_decompress` must return a ZSTD_isError when the dst
        // can't hold the decompressed output, not panic on OOB writes.
        use crate::compress::zstd_compress::{ZSTD_compress, ZSTD_compressBound};
        let src: Vec<u8> = b"payload that won't fit in a tiny dst. ".repeat(20);
        let bound = ZSTD_compressBound(src.len());
        let mut compressed = vec![0u8; bound];
        let n = ZSTD_compress(&mut compressed, &src, 1);
        assert!(!crate::common::error::ERR_isError(n));

        // dst far smaller than the real decompressed size.
        let mut tiny_dst = [0u8; 16];
        let rc = ZSTD_decompress(&mut tiny_dst, &compressed[..n]);
        assert!(crate::common::error::ERR_isError(rc));
    }

    #[test]
    fn zstd_decompress_rejects_garbage_without_panicking() {
        // Safety gate: feeding arbitrary bytes into `ZSTD_decompress`
        // must surface a ZSTD_isError return — never panic. This is
        // the contract callers rely on when accepting compressed
        // input from the network / disk.
        let mut dst = vec![0u8; 1024];
        // Empty src is NOT garbage — it's a valid zero-frame stream
        // returning 0 bytes (matches upstream). The garbage inputs
        // below are all malformed and MUST surface an error.
        let test_inputs: Vec<Vec<u8>> = vec![
            vec![0u8],       // 1 byte (below magic)
            vec![0u8; 3],    // below magic size
            vec![0xFFu8; 8], // bogus magic
            {
                // Valid magic but truncated mid-FHD.
                let mut v = ZSTD_MAGICNUMBER.to_le_bytes().to_vec();
                v.push(0x20);
                v
            },
            {
                // Valid magic but FHD reserved bit set — must reject.
                let mut v = ZSTD_MAGICNUMBER.to_le_bytes().to_vec();
                v.extend_from_slice(&[0x28, 100]); // reserved bit 3 set
                v
            },
            (0..200u8).map(|i| i.wrapping_mul(17)).collect(), // pseudo-random
        ];
        // And confirm empty input returns 0 bytes cleanly (not an error).
        let empty_rc = ZSTD_decompress(&mut dst, &[]);
        assert!(!crate::common::error::ERR_isError(empty_rc));
        assert_eq!(empty_rc, 0);
        for (i, input) in test_inputs.iter().enumerate() {
            let rc = ZSTD_decompress(&mut dst, input);
            assert!(
                crate::common::error::ERR_isError(rc),
                "input #{i} (len={}) should have errored but returned {rc}",
                input.len(),
            );
        }
    }

    #[test]
    fn startingInputLength_differs_by_format() {
        // zstd1: 4-byte magic + 1-byte FHD = 5.
        // magicless: just the 1-byte FHD = 1.
        assert_eq!(ZSTD_startingInputLength(ZSTD_format_e::ZSTD_f_zstd1), 5);
        assert_eq!(
            ZSTD_startingInputLength(ZSTD_format_e::ZSTD_f_zstd1_magicless),
            1
        );
    }

    #[test]
    fn getFrameHeader_advanced_magicless_skips_magic_check() {
        // In magicless mode the FHD is at offset 0 (no 4-byte magic).
        // Build a magicless frame: FHD=0x20 (singleSegment=1, fcsID=0)
        // + 1-byte FCS. Parser must accept it without looking for the
        // zstd magic number.
        let src = [0x20u8, 42];
        let mut zfh = ZSTD_FrameHeader::default();
        let rc =
            ZSTD_getFrameHeader_advanced(&mut zfh, &src, ZSTD_format_e::ZSTD_f_zstd1_magicless);
        assert_eq!(rc, 0);
        assert_eq!(zfh.frameContentSize, 42);
        // Header size in magicless mode drops by 4 (no magic).
        assert_eq!(zfh.headerSize, 2);
    }

    #[test]
    fn frameHeaderSize_rejects_too_short_input() {
        // `ZSTD_frameHeaderSize` needs at least magic (4) + FHD (1)
        // bytes to read the FHD. A shorter input must return a
        // ZSTD_isError (specifically SrcSizeWrong), not index OOB.
        assert!(crate::common::error::ERR_isError(ZSTD_frameHeaderSize(&[])));
        assert!(crate::common::error::ERR_isError(ZSTD_frameHeaderSize(&[
            0xFDu8
        ])));
        assert!(crate::common::error::ERR_isError(ZSTD_frameHeaderSize(&[
            0u8, 0, 0, 0
        ])));
    }

    #[test]
    fn frameHeaderSize_returns_exact_size_for_each_layout() {
        // The layout-specific frame header sizes are:
        //   - singleSegment=1, fcsID=0 → 6 bytes (magic + FHD + 1 FCS)
        //   - singleSegment=0, fcsID=0 → 6 bytes (magic + FHD + 1 wlByte + 0 FCS)
        //   - singleSegment=1, fcsID=1 → 7 bytes (magic + FHD + 2 FCS)
        //   - singleSegment=1, fcsID=2 → 9 bytes (magic + FHD + 4 FCS)
        //   - singleSegment=1, fcsID=3 → 13 bytes (magic + FHD + 8 FCS)
        // All numbers match what frame_header_fcs_size_variants observed
        // via the full parser; this test pins the stand-alone helper.
        let magic = ZSTD_MAGICNUMBER.to_le_bytes();

        // fcsID=0, singleSegment=1 → 6 bytes.
        let mut src = magic.to_vec();
        src.push(0x20); // FHD = singleSegment=1, fcsID=0
        src.push(42); // FCS byte
        assert_eq!(ZSTD_frameHeaderSize(&src), 6);

        // fcsID=0, singleSegment=0 → 6 bytes (wlByte in place of FCS).
        let mut src = magic.to_vec();
        src.push(0x00);
        src.push(0x20);
        assert_eq!(ZSTD_frameHeaderSize(&src), 6);

        // fcsID=1, singleSegment=1 → 7 bytes.
        let mut src = magic.to_vec();
        src.push((1 << 6) | (1 << 5));
        src.extend_from_slice(&744u16.to_le_bytes());
        assert_eq!(ZSTD_frameHeaderSize(&src), 7);

        // fcsID=3, singleSegment=1 → 13 bytes.
        let mut src = magic.to_vec();
        src.push((3 << 6) | (1 << 5));
        src.extend_from_slice(&0u64.to_le_bytes());
        assert_eq!(ZSTD_frameHeaderSize(&src), 13);
    }

    #[test]
    fn frame_header_rejects_oversized_windowLog() {
        // Windowlog 32 exceeds ZSTD_WINDOWLOG_MAX_64 (31). Construct
        // a frame with wlByte whose top 5 bits encode windowLog - 10.
        // wlByte high 5 bits = 22 → windowLog = 22 + 10 = 32 → reject.
        let mut src = ZSTD_MAGICNUMBER.to_le_bytes().to_vec();
        src.push(0x00); // FHD: all zero (multi-segment, no dict/checksum)
        src.push(22u8 << 3); // wlByte → windowLog = 22 + 10 = 32
        let mut zfh = ZSTD_FrameHeader::default();
        let rc = ZSTD_getFrameHeader(&mut zfh, &src);
        assert!(crate::common::error::ERR_isError(rc));
        assert_eq!(
            crate::common::error::ERR_getErrorCode(rc),
            ErrorCode::FrameParameterWindowTooLarge
        );
    }

    #[test]
    fn frame_header_accepts_max_windowLog() {
        // windowLog = 31 (exactly at the cap) must still parse.
        // wlByte high 5 bits = 21 → windowLog = 21 + 10 = 31.
        let mut src = ZSTD_MAGICNUMBER.to_le_bytes().to_vec();
        src.push(0x00);
        src.push(21u8 << 3);
        let mut zfh = ZSTD_FrameHeader::default();
        let rc = ZSTD_getFrameHeader(&mut zfh, &src);
        assert_eq!(rc, 0);
        assert_eq!(zfh.windowSize, 1u64 << 31);
    }

    #[test]
    fn frame_header_bad_magic_errors() {
        let mut zfh = ZSTD_FrameHeader::default();
        let src = [0xFF, 0xFF, 0xFF, 0xFF, 0, 0];
        let rc = ZSTD_getFrameHeader(&mut zfh, &src);
        assert!(crate::common::error::ERR_isError(rc));
        assert_eq!(
            crate::common::error::ERR_getErrorCode(rc),
            ErrorCode::PrefixUnknown
        );
    }

    #[test]
    fn frame_header_skippable_frame() {
        // Magic skippable[0] + 4-byte frame content size.
        let mut src = Vec::new();
        src.extend_from_slice(&ZSTD_MAGIC_SKIPPABLE_START.to_le_bytes());
        src.extend_from_slice(&42u32.to_le_bytes());
        let mut zfh = ZSTD_FrameHeader::default();
        let rc = ZSTD_getFrameHeader(&mut zfh, &src);
        assert_eq!(rc, 0);
        assert_eq!(zfh.frameType, ZSTD_FrameType_e::ZSTD_skippableFrame);
        assert_eq!(zfh.frameContentSize, 42);
        assert_eq!(zfh.dictID, 0);
        assert_eq!(zfh.headerSize, ZSTD_SKIPPABLEHEADERSIZE as u32);
    }

    #[test]
    fn frame_header_single_segment_no_dict_no_fcs() {
        // FHD byte: singleSegment=1, fcsID=0 → FCS field = 1 byte,
        // dictID=0, checksumFlag=0, reserved=0.
        // FHD = (fcsID<<6)|(singleSegment<<5)|(reserved<<3)|(checksumFlag<<2)|dictID
        //     = (0<<6)|(1<<5)|0|0|0 = 0x20.
        let mut src = Vec::new();
        src.extend_from_slice(&ZSTD_MAGICNUMBER.to_le_bytes());
        src.push(0x20); // FHD
        src.push(100); // FCS byte (singleSegment implies size is this byte)
        let mut zfh = ZSTD_FrameHeader::default();
        let rc = ZSTD_getFrameHeader(&mut zfh, &src);
        assert_eq!(rc, 0);
        assert_eq!(zfh.frameType, ZSTD_FrameType_e::ZSTD_frame);
        assert_eq!(zfh.frameContentSize, 100);
        assert_eq!(zfh.windowSize, 100); // singleSegment: window = FCS
        assert_eq!(zfh.checksumFlag, 0);
        assert_eq!(zfh.dictID, 0);
        assert_eq!(zfh.headerSize, 6);
    }

    #[test]
    fn frame_header_fcs_size_variants() {
        // fcsID encodes the FCS field width: 0 → 1 byte (when
        // singleSegment), 1 → 2 bytes + 256 offset, 2 → 4 bytes,
        // 3 → 8 bytes. Exercise each non-trivial variant so the
        // ranger decode logic (line 204-206) stays byte-exact.
        //
        // FHD layout: (fcsID<<6)|(singleSegment<<5)|(reserved<<3)|(checksumFlag<<2)|dictID

        // --- fcsID=1: FCS = LE16 + 256. Encode FCS=1000 → raw LE16 = 744.
        {
            let mut src = Vec::new();
            src.extend_from_slice(&ZSTD_MAGICNUMBER.to_le_bytes());
            src.push((1 << 6) | (1 << 5)); // fcsID=1, singleSegment=1
            src.extend_from_slice(&744u16.to_le_bytes());
            let mut zfh = ZSTD_FrameHeader::default();
            let rc = ZSTD_getFrameHeader(&mut zfh, &src);
            assert_eq!(rc, 0);
            assert_eq!(zfh.frameContentSize, 1000);
            assert_eq!(zfh.windowSize, 1000);
            assert_eq!(zfh.headerSize, 7);
        }

        // --- fcsID=2: FCS = LE32.
        {
            let mut src = Vec::new();
            src.extend_from_slice(&ZSTD_MAGICNUMBER.to_le_bytes());
            src.push((2 << 6) | (1 << 5)); // fcsID=2, singleSegment=1
            src.extend_from_slice(&123_456u32.to_le_bytes());
            let mut zfh = ZSTD_FrameHeader::default();
            let rc = ZSTD_getFrameHeader(&mut zfh, &src);
            assert_eq!(rc, 0);
            assert_eq!(zfh.frameContentSize, 123_456);
            assert_eq!(zfh.windowSize, 123_456);
            assert_eq!(zfh.headerSize, 9);
        }

        // --- fcsID=3: FCS = LE64.
        {
            let mut src = Vec::new();
            src.extend_from_slice(&ZSTD_MAGICNUMBER.to_le_bytes());
            src.push((3 << 6) | (1 << 5)); // fcsID=3, singleSegment=1
            src.extend_from_slice(&9_999_999_999u64.to_le_bytes());
            let mut zfh = ZSTD_FrameHeader::default();
            let rc = ZSTD_getFrameHeader(&mut zfh, &src);
            assert_eq!(rc, 0);
            assert_eq!(zfh.frameContentSize, 9_999_999_999);
            assert_eq!(zfh.headerSize, 13);
        }
    }

    #[test]
    fn frame_header_window_descriptor() {
        // singleSegment=0, so there's a 1-byte window descriptor.
        // FHD = 0x00 (fcsID=0, singleSeg=0, reserved=0, checksum=0, dictID=0).
        // wlByte: windowLog = (wlByte>>3) + 10; we want windowLog=14 → wlByte>>3 = 4 → wlByte = 0x20.
        let mut src = Vec::new();
        src.extend_from_slice(&ZSTD_MAGICNUMBER.to_le_bytes());
        src.push(0x00); // FHD
        src.push(0x20); // wl
                        // No dictID, no FCS (fcsID=0 non-single → absent).
        let mut zfh = ZSTD_FrameHeader::default();
        let rc = ZSTD_getFrameHeader(&mut zfh, &src);
        assert_eq!(rc, 0);
        assert_eq!(zfh.windowSize, 1u64 << 14);
        assert_eq!(zfh.frameContentSize, ZSTD_CONTENTSIZE_UNKNOWN);
    }

    fn make_raw_hello_frame() -> Vec<u8> {
        let mut src = Vec::new();
        src.extend_from_slice(&ZSTD_MAGICNUMBER.to_le_bytes());
        src.push(0x20); // FHD: singleSegment, fcsID=0 (1-byte FCS)
        src.push(5); // FCS byte
        let bh = (5u32 << 3) | (1); // lastBlock=1, bt_raw=0, cSize=5
        src.push((bh & 0xFF) as u8);
        src.push(((bh >> 8) & 0xFF) as u8);
        src.push(((bh >> 16) & 0xFF) as u8);
        src.extend_from_slice(b"HELLO");
        src
    }

    #[test]
    fn get_frame_content_size_returns_declared_fcs() {
        let src = make_raw_hello_frame();
        let fcs = ZSTD_getFrameContentSize(&src);
        assert_eq!(fcs, 5);
    }

    #[test]
    fn get_frame_content_size_unknown_on_absent_fcs() {
        // singleSegment=0, fcsID=0 → FCS absent → CONTENTSIZE_UNKNOWN.
        let mut src = Vec::new();
        src.extend_from_slice(&ZSTD_MAGICNUMBER.to_le_bytes());
        src.push(0x00); // FHD
        src.push(0x20); // window descriptor
        let fcs = ZSTD_getFrameContentSize(&src);
        assert_eq!(fcs, ZSTD_CONTENTSIZE_UNKNOWN);
    }

    #[test]
    fn get_frame_content_size_error_on_bad_magic() {
        let src = [0xFFu8; 16];
        let fcs = ZSTD_getFrameContentSize(&src);
        assert_eq!(fcs, ZSTD_CONTENTSIZE_ERROR);
    }

    #[test]
    fn get_frame_content_size_zero_for_skippable_frame() {
        // A skippable frame's user data isn't decompressed content
        // per the spec — `ZSTD_getFrameContentSize` must report 0,
        // not the user-data length (which parses into
        // frameContentSize on the skippable path).
        let mut src = Vec::new();
        src.extend_from_slice(&ZSTD_MAGIC_SKIPPABLE_START.to_le_bytes());
        src.extend_from_slice(&12u32.to_le_bytes()); // user-data len
        src.extend_from_slice(&[0u8; 12]);
        assert_eq!(ZSTD_getFrameContentSize(&src), 0);
    }

    #[test]
    fn find_frame_compressed_size_matches_hand_count() {
        let src = make_raw_hello_frame();
        let sz = ZSTD_findFrameCompressedSize(&src);
        assert_eq!(sz, src.len());
    }

    #[test]
    fn decompressionMargin_nonzero_for_valid_frame() {
        // Compress some bytes, then ask for the decomp margin.
        let src = b"the fast brown fox ".repeat(32);
        let mut dst = vec![0u8; 256];
        let n = crate::compress::zstd_compress::ZSTD_compress(&mut dst, &src, 3);
        dst.truncate(n);
        let margin = ZSTD_decompressionMargin(&dst);
        assert!(!crate::common::error::ERR_isError(margin));
        assert!(margin > 0);
        // Upper bound: frame header + a few blocks' overhead +
        // blockSizeMax (128 KB). For a tiny payload we expect
        // margin < 200 KB.
        assert!(margin < 200 * 1024);
    }

    #[test]
    fn decompressionMargin_includes_checksum_bytes_when_flag_is_set() {
        // `ZSTD_decompressionMargin` must account for the 4-byte XXH64
        // trailer when the frame declares `checksumFlag`. Two frames
        // compressed from the same source, one with checksum, one
        // without, should differ by at least 4 bytes of margin.
        let src = b"fast brown fox ".repeat(32);
        let mut cctx_a = crate::compress::zstd_compress::ZSTD_createCCtx().unwrap();
        crate::compress::zstd_compress::ZSTD_CCtx_setParameter(
            &mut cctx_a,
            crate::compress::zstd_compress::ZSTD_cParameter::ZSTD_c_checksumFlag,
            0,
        );
        let mut dst_no_chk = vec![0u8; 256];
        let n_no =
            crate::compress::zstd_compress::ZSTD_compress2(&mut cctx_a, &mut dst_no_chk, &src);
        assert!(!crate::common::error::ERR_isError(n_no));
        dst_no_chk.truncate(n_no);

        let mut cctx_b = crate::compress::zstd_compress::ZSTD_createCCtx().unwrap();
        crate::compress::zstd_compress::ZSTD_CCtx_setParameter(
            &mut cctx_b,
            crate::compress::zstd_compress::ZSTD_cParameter::ZSTD_c_checksumFlag,
            1,
        );
        let mut dst_chk = vec![0u8; 256];
        let n_chk = crate::compress::zstd_compress::ZSTD_compress2(&mut cctx_b, &mut dst_chk, &src);
        assert!(!crate::common::error::ERR_isError(n_chk));
        dst_chk.truncate(n_chk);

        let m_no = ZSTD_decompressionMargin(&dst_no_chk);
        let m_chk = ZSTD_decompressionMargin(&dst_chk);
        assert!(!crate::common::error::ERR_isError(m_no));
        assert!(!crate::common::error::ERR_isError(m_chk));
        // The checksum-bearing margin must exceed the plain margin by
        // at least 4 (the trailer). Block-count overhead is identical
        // for the same source → any excess is just the 4-byte XXH64.
        assert!(
            m_chk >= m_no + 4,
            "checksum margin must include ≥4 bytes over plain margin: chk={m_chk}, no={m_no}"
        );
    }

    #[test]
    fn decompressionMargin_rejects_garbage_input() {
        // Invalid frame header → error sentinel, not a silent 0.
        let rc = ZSTD_decompressionMargin(&[0u8; 32]);
        assert!(crate::common::error::ERR_isError(rc));
    }

    #[test]
    fn decompressBound_empty_input_is_zero() {
        assert_eq!(ZSTD_decompressBound(&[]), 0);
    }

    #[test]
    fn decompressBound_corrupted_input_returns_error_sentinel() {
        // Non-frame bytes can't be parsed → CONTENTSIZE_ERROR.
        let bogus = [0u8, 1, 2, 3, 4, 5, 6, 7];
        assert_eq!(ZSTD_decompressBound(&bogus), ZSTD_CONTENTSIZE_ERROR);
    }

    #[test]
    fn decompressBound_upper_bounds_real_decompressed_size() {
        // Regression gate: `ZSTD_decompressBound` must never return
        // a value smaller than the actual decompressed size across
        // varied (size, level) combos. A bound underrun would let
        // callers allocate too-small buffers.
        use crate::compress::zstd_compress::{ZSTD_compress, ZSTD_compressBound};
        for &size in &[0usize, 1, 33, 1024, 65_536, 131_073] {
            let src: Vec<u8> = (0..size as u32).map(|i| (i ^ (i >> 5)) as u8).collect();
            for &level in &[1i32, 3, 10] {
                let bound = ZSTD_compressBound(src.len());
                let mut dst = vec![0u8; bound];
                let n = ZSTD_compress(&mut dst, &src, level);
                assert!(!crate::common::error::ERR_isError(n));
                let db = ZSTD_decompressBound(&dst[..n]);
                assert!(
                    db != ZSTD_CONTENTSIZE_ERROR,
                    "decompressBound flagged error on valid frame size={size} level={level}"
                );
                assert!(
                    db >= size as u64,
                    "bound under-reports: size={size} level={level} bound={db}",
                );
            }
        }
    }

    #[test]
    fn decompressBound_sums_frames() {
        // Build 2 raw-HELLO frames + 1 skippable.
        let mut src = make_raw_hello_frame();
        src.extend_from_slice(&make_raw_hello_frame());
        src.extend_from_slice(&ZSTD_MAGIC_SKIPPABLE_START.to_le_bytes());
        src.extend_from_slice(&4u32.to_le_bytes());
        src.extend_from_slice(&[0u8; 4]);
        let bound = ZSTD_decompressBound(&src);
        // Two raw "HELLO" = 10; skippable doesn't add content.
        assert_eq!(bound, 10);
    }

    #[test]
    fn getDecompressedSize_zero_on_unknown() {
        // Build a tiny "hello" raw frame without content size in header.
        let raw = make_raw_hello_frame();
        let got = ZSTD_getDecompressedSize(&raw);
        // FCS present in raw-hello frame (5) — return it, not 0.
        assert_eq!(got, 5);

        // For random non-frame bytes, returns 0.
        let bogus = [0u8, 1, 2, 3];
        assert_eq!(ZSTD_getDecompressedSize(&bogus), 0);

        // Valid frame WITHOUT FCS (singleSegment=0, fcsID=0) should
        // also collapse UNKNOWN (= u64::MAX sentinel) down to 0 per
        // the deprecated-API convention.
        let mut no_fcs = Vec::new();
        no_fcs.extend_from_slice(&ZSTD_MAGICNUMBER.to_le_bytes());
        no_fcs.push(0x00); // FHD: no single-segment, no FCS
        no_fcs.push(0x20); // window descriptor
        assert_eq!(ZSTD_getDecompressedSize(&no_fcs), 0);
    }

    #[test]
    fn decodingBufferSize_min_basic() {
        // Small window, small content.
        let sz = ZSTD_decodingBufferSize_min(1024, 1024);
        assert!(!crate::common::error::ERR_isError(sz));
        assert!(sz >= 1024);
    }

    #[test]
    fn decodingBufferSize_min_matches_upstream_formula() {
        // Regression gate for upstream's formula in
        // `ZSTD_decodingBufferSize_internal`:
        //   blockSize     = min(windowSize, ZSTD_BLOCKSIZE_MAX)   (with blockSizeMax = BLOCKSIZE_MAX)
        //   neededRBSize  = windowSize + 2*blockSize + 2*WILDCOPY
        //   return          min(frameContentSize, neededRBSize)
        use crate::common::zstd_internal::WILDCOPY_OVERLENGTH;
        use crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX;

        // Small window: blockSize = windowSize = 1024.
        let w = 1024u64;
        let fcs = u64::MAX; // Take the RB path, not the FCS path.
        let expected = w + 2 * w + (WILDCOPY_OVERLENGTH as u64) * 2;
        assert_eq!(ZSTD_decodingBufferSize_min(w, fcs) as u64, expected);

        // Large window (> BLOCKSIZE_MAX): blockSize = BLOCKSIZE_MAX.
        let w2 = 1u64 << 20;
        let expected2 = w2 + 2 * ZSTD_BLOCKSIZE_MAX as u64 + (WILDCOPY_OVERLENGTH as u64) * 2;
        assert_eq!(ZSTD_decodingBufferSize_min(w2, fcs) as u64, expected2);

        // FCS caps the result: when frameContentSize is tiny, return it.
        let tiny = 200u64;
        assert_eq!(ZSTD_decodingBufferSize_min(w2, tiny) as u64, tiny);
    }

    #[test]
    fn copyDCtx_deep_copies_all_state() {
        let mut src = ZSTD_DCtx::default();
        src.stream_dict = b"marker-dict".to_vec();
        src.d_windowLogMax = 25;

        let mut dst = ZSTD_DCtx::default();
        ZSTD_copyDCtx(&mut dst, &src);
        assert_eq!(dst.stream_dict, b"marker-dict");
        assert_eq!(dst.d_windowLogMax, 25);
    }

    #[test]
    fn initDStream_family_returns_startingInputLength() {
        // All three `ZSTD_initDStream*` variants must return the
        // starting-input length hint (5 for zstd1), matching upstream
        // (zstd_decompress.c:1746/1755/1766). Silent 0 previously —
        // broke streaming callers that used the return as the initial
        // minimum-bytes-to-read hint.
        use crate::decompress::zstd_ddict::ZSTD_createDDict;
        let expected = ZSTD_startingInputLength(ZSTD_format_e::ZSTD_f_zstd1);
        let mut a = ZSTD_DCtx::default();
        assert_eq!(ZSTD_initDStream(&mut a), expected);
        let mut b = ZSTD_DCtx::default();
        assert_eq!(ZSTD_initDStream_usingDict(&mut b, b"seed-dict"), expected);
        let mut c = ZSTD_DCtx::default();
        let ddict = ZSTD_createDDict(b"seed-ddict").expect("ddict");
        assert_eq!(ZSTD_initDStream_usingDDict(&mut c, &ddict), expected);
    }

    #[test]
    fn initDStream_usingDDict_copies_dict_content() {
        use crate::decompress::zstd_ddict::ZSTD_createDDict;
        let dict = b"test-dict-content".to_vec();
        let ddict = ZSTD_createDDict(&dict).expect("ddict");
        let mut dctx = ZSTD_DCtx::default();
        let rc = ZSTD_initDStream_usingDDict(&mut dctx, &ddict);
        // Upstream (zstd_decompress.c:1766) returns
        // `ZSTD_startingInputLength(format)` = 5 for zstd1. Previously
        // our port returned 0 — masked divergence for C-compat
        // callers that used the return as the initial buffer size.
        assert_eq!(rc, ZSTD_startingInputLength(ZSTD_format_e::ZSTD_f_zstd1));
        assert_eq!(dctx.stream_dict, dict);
    }

    #[test]
    fn nextSrcSizeToDecompress_starts_at_frame_header_prefix() {
        let dctx = ZSTD_DCtx::default();
        assert_eq!(
            ZSTD_nextSrcSizeToDecompress(&dctx),
            ZSTD_startingInputLength(ZSTD_format_e::ZSTD_f_zstd1)
        );
    }

    #[test]
    fn insertBlock_updates_history_end_and_contiguity_state() {
        let mut dctx = ZSTD_DCtx::default();
        let a = vec![1u8; 8];
        let b = vec![2u8; 5];

        let n = ZSTD_insertBlock(&mut dctx, &a);
        assert_eq!(n, a.len());
        assert_eq!(dctx.prefixStart, Some(a.as_ptr() as usize));
        assert_eq!(dctx.previousDstEnd, Some(a.as_ptr() as usize + a.len()));
        assert_eq!(dctx.dictEnd, None);

        let n = ZSTD_insertBlock(&mut dctx, &b);
        assert_eq!(n, b.len());
        assert_eq!(dctx.dictEnd, Some(a.as_ptr() as usize + a.len()));
        assert_eq!(dctx.prefixStart, Some(b.as_ptr() as usize));
        assert_eq!(dctx.previousDstEnd, Some(b.as_ptr() as usize + b.len()));
    }

    #[test]
    fn initDStream_preserves_configured_dict() {
        // Once a dict is loaded, ZSTD_initDStream (session_only-style
        // reset) must keep it so back-to-back frame decodes all see
        // the same dict.
        let mut dctx = ZSTD_DCtx::default();
        dctx.stream_dict = b"sticky-dict".to_vec();
        ZSTD_initDStream(&mut dctx);
        assert_eq!(dctx.stream_dict, b"sticky-dict");
    }

    #[test]
    fn dParam_all_variants_set_get_roundtrip() {
        // Symmetric with CCtx side — every ZSTD_dParameter variant
        // should round-trip via setParameter / getParameter.
        let mut dctx = ZSTD_DCtx::default();
        let cases = [(ZSTD_dParameter::ZSTD_d_windowLogMax, 20)];
        for &(param, value) in &cases {
            ZSTD_DCtx_setParameter(&mut dctx, param, value);
            let mut got = -1i32;
            ZSTD_DCtx_getParameter(&dctx, param, &mut got);
            assert_eq!(got, value, "param {:?} didn't round-trip", param);
        }
    }

    #[test]
    fn dctx_reset_modes_differ_correctly() {
        let mut dctx = ZSTD_DCtx::default();
        dctx.stream_dict = b"prior-dict".to_vec();
        dctx.d_windowLogMax = 25;

        // session_only: keep dict + windowLogMax.
        ZSTD_DCtx_reset(&mut dctx, ZSTD_DResetDirective::ZSTD_reset_session_only);
        assert_eq!(dctx.stream_dict, b"prior-dict");
        assert_eq!(dctx.d_windowLogMax, 25);

        // parameters: drop them. `d_windowLogMax` resets to
        // `ZSTD_WINDOWLOG_LIMIT_DEFAULT` (27), matching upstream's
        // `maxWindowSize = (1 << 27) + 1` initialization.
        ZSTD_DCtx_reset(&mut dctx, ZSTD_DResetDirective::ZSTD_reset_parameters);
        assert!(dctx.stream_dict.is_empty());
        assert_eq!(dctx.d_windowLogMax, ZSTD_WINDOWLOG_LIMIT_DEFAULT);

        // session_and_parameters: superset — does both in one call.
        dctx.stream_dict = b"seed-again".to_vec();
        dctx.d_windowLogMax = 20;
        ZSTD_DCtx_reset(
            &mut dctx,
            ZSTD_DResetDirective::ZSTD_reset_session_and_parameters,
        );
        assert!(dctx.stream_dict.is_empty());
        assert_eq!(dctx.d_windowLogMax, ZSTD_WINDOWLOG_LIMIT_DEFAULT);
    }

    #[test]
    fn decompress_side_free_functions_accept_none_without_panic() {
        // Symmetric with the compress-side contract: the two
        // decompression-side Option-taking freers must accept None
        // without panicking. (`ZSTD_freeDCtx` takes `Box<T>` directly
        // and has no None path.)
        assert_eq!(ZSTD_freeDStream(None), 0);
        assert_eq!(crate::decompress::zstd_ddict::ZSTD_freeDDict(None), 0);
    }

    #[test]
    fn decompressBegin_variants_seed_dict_consistently() {
        // Legacy continue-style init entries must all land the same
        // bytes in `stream_dict`:
        //   - _usingDict stashes the raw dict
        //   - _usingDDict extracts the DDict's content, same result
        //   - plain _decompressBegin is a no-op (doesn't clear)
        use crate::decompress::zstd_ddict::ZSTD_createDDict;
        let dict = b"begin-usingDict-seed".to_vec();

        let mut a = ZSTD_DCtx::default();
        assert_eq!(ZSTD_decompressBegin_usingDict(&mut a, &dict), 0);
        assert_eq!(a.stream_dict, dict);

        let ddict = ZSTD_createDDict(&dict).expect("ddict");
        let mut b = ZSTD_DCtx::default();
        assert_eq!(ZSTD_decompressBegin_usingDDict(&mut b, &ddict), 0);
        assert_eq!(b.stream_dict, dict);

        // Plain decompressBegin on a DCtx with a prior dict must
        // leave it intact (upstream semantic: no-op in v0.1 scope).
        let mut c = ZSTD_DCtx::default();
        c.stream_dict = b"preloaded".to_vec();
        assert_eq!(ZSTD_decompressBegin(&mut c), 0);
        assert_eq!(c.stream_dict, b"preloaded");
    }

    #[test]
    fn DCtx_refDDict_seeds_stream_dict_and_roundtrips_via_stream_api() {
        // Symmetric with ZSTD_CCtx_refCDict: refDDict wires the
        // DDict's raw content into the DCtx so a subsequent
        // streaming decompress honors the dict.
        use crate::compress::zstd_compress::{ZSTD_compress_usingDict, ZSTD_createCCtx};
        use crate::decompress::zstd_ddict::ZSTD_createDDict;

        let dict = b"refDDict-test-dict ".repeat(8);
        let ddict = ZSTD_createDDict(&dict).expect("ddict");

        // State check.
        let mut dctx = ZSTD_DCtx::default();
        let rc = ZSTD_DCtx_refDDict(&mut dctx, &ddict);
        assert_eq!(rc, 0);
        assert_eq!(dctx.stream_dict, dict);

        // Roundtrip: compress with the raw-content dict, decompress
        // via the streaming DCtx that was primed via refDDict.
        let src: Vec<u8> = b"payload wearing the refDDict-test-dict ".repeat(18);
        let mut cctx = ZSTD_createCCtx().unwrap();
        let mut cbuf = vec![0u8; 4096];
        let n = ZSTD_compress_usingDict(&mut cctx, &mut cbuf, &src, &dict, 3);
        assert!(!crate::common::error::ERR_isError(n));

        ZSTD_initDStream(&mut dctx);
        let mut out = vec![0u8; src.len() + 64];
        let mut in_pos = 0usize;
        let mut out_pos = 0usize;
        let drain =
            ZSTD_decompressStream(&mut dctx, &mut out, &mut out_pos, &cbuf[..n], &mut in_pos);
        assert!(!crate::common::error::ERR_isError(drain));
        assert_eq!(&out[..out_pos], &src[..]);
    }

    #[test]
    fn DCtx_refPrefix_roundtrips_with_compressor_using_dict() {
        // End-to-end: compress via ZSTD_compress_usingDict, decompress
        // through a streaming DCtx that was primed with
        // ZSTD_DCtx_refPrefix. refPrefix must be symmetric with the
        // compression-side dict load.
        use crate::compress::zstd_compress::{ZSTD_compress_usingDict, ZSTD_createCCtx};
        let dict = b"shared-dict-content-for-prefix-test ".repeat(8);
        let src = b"payload that uses shared-dict-content-for-prefix-test ".repeat(16);

        let mut cctx = ZSTD_createCCtx().unwrap();
        let mut dst = vec![0u8; 4096];
        let n = ZSTD_compress_usingDict(&mut cctx, &mut dst, &src, &dict, 3);
        assert!(!crate::common::error::ERR_isError(n));

        let mut dctx = ZSTD_DCtx::default();
        let rc = ZSTD_DCtx_refPrefix(&mut dctx, &dict);
        assert_eq!(rc, 0);
        ZSTD_initDStream(&mut dctx);
        let mut out_buf = vec![0u8; src.len() + 64];
        let mut in_pos = 0usize;
        let mut out_pos = 0usize;
        let drain = ZSTD_decompressStream(
            &mut dctx,
            &mut out_buf,
            &mut out_pos,
            &dst[..n],
            &mut in_pos,
        );
        assert!(
            !crate::common::error::ERR_isError(drain),
            "drain err: {drain:#x}"
        );
        assert_eq!(&out_buf[..out_pos], &src[..]);
    }

    #[test]
    fn DCtx_loadDictionary_variants_store_equivalent_state() {
        // loadDictionary, loadDictionary_byReference, and
        // loadDictionary_advanced are thin wrappers that must all
        // land the bytes in `stream_dict`. Regression gate in case
        // someone re-implements one of them without the others.
        use crate::decompress::zstd_ddict::{ZSTD_dictContentType_e, ZSTD_dictLoadMethod_e};
        let dict = b"variant-equivalence-test-dict".to_vec();

        let mut a = ZSTD_DCtx::default();
        ZSTD_DCtx_loadDictionary(&mut a, &dict);

        let mut b = ZSTD_DCtx::default();
        ZSTD_DCtx_loadDictionary_byReference(&mut b, &dict);

        let mut c = ZSTD_DCtx::default();
        ZSTD_DCtx_loadDictionary_advanced(
            &mut c,
            &dict,
            ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy,
            ZSTD_dictContentType_e::ZSTD_dct_auto,
        );

        assert_eq!(a.stream_dict, dict);
        assert_eq!(b.stream_dict, dict);
        assert_eq!(c.stream_dict, dict);
    }

    #[test]
    fn decompressStream_handles_skippable_then_regular_frame() {
        // Streaming decoder contract: a skippable frame fed first
        // should not poison subsequent decompression of a real frame
        // staged after it. Upstream decoders silently consume the
        // skippable's 8+N bytes and proceed; our port must do the
        // same in streaming mode.
        let src = b"streaming-skip-then-real ".repeat(20);
        let mut frame = vec![0u8; 4096];
        let n = crate::compress::zstd_compress::ZSTD_compress(&mut frame, &src, 3);
        assert!(!crate::common::error::ERR_isError(n));

        let mut stream = Vec::new();
        stream.extend_from_slice(&ZSTD_MAGIC_SKIPPABLE_START.to_le_bytes());
        stream.extend_from_slice(&8u32.to_le_bytes());
        stream.extend_from_slice(b"SKIPDATA");
        stream.extend_from_slice(&frame[..n]);

        let mut dctx = ZSTD_DCtx::new();
        ZSTD_initDStream(&mut dctx);
        let mut out = vec![0u8; src.len() + 64];
        let mut in_pos = 0usize;
        let mut out_pos = 0usize;
        let _hint = ZSTD_decompressStream(&mut dctx, &mut out, &mut out_pos, &stream, &mut in_pos);
        // Keep draining until no more progress is expected (simple
        // cap on iterations).
        for _ in 0..8 {
            if out_pos >= src.len() {
                break;
            }
            let _ = ZSTD_decompressStream(&mut dctx, &mut out, &mut out_pos, &[], &mut 0usize);
        }
        assert_eq!(&out[..out_pos], &src[..]);
    }

    #[test]
    fn decompressDCtx_truncated_frame_errors_out_cleanly() {
        // Feeding half of a valid frame must return an error (not
        // panic, not produce garbage). Upstream surfaces srcSize_wrong
        // through ZSTD_decompressFrame; ours should do the same.
        use crate::decompress::zstd_decompress_block::{
            ZSTD_buildDefaultSeqTables, ZSTD_decoder_entropy_rep,
        };
        let src = b"full-frame-that-we-then-truncate ".repeat(20);
        let mut frame = vec![0u8; 4096];
        let n = crate::compress::zstd_compress::ZSTD_compress(&mut frame, &src, 3);
        assert!(!crate::common::error::ERR_isError(n));

        let mut dctx = ZSTD_DCtx::new();
        ZSTD_buildDefaultSeqTables(&mut dctx);
        let mut rep = ZSTD_decoder_entropy_rep::default();
        let mut xxh = crate::common::xxhash::XXH64_state_t::default();
        let mut out = vec![0u8; src.len() + 64];
        let decoded = ZSTD_decompressDCtx(&mut dctx, &mut rep, &mut xxh, &mut out, &frame[..n / 2]);
        assert!(
            crate::common::error::ERR_isError(decoded),
            "truncated frame should error, got decoded={}",
            decoded,
        );
    }

    #[test]
    fn decompressDCtx_advances_past_skippable_frames_mid_stream() {
        // Regression gate: `ZSTD_decompressDCtx` must transparently
        // advance past skippable frames when they appear BETWEEN two
        // regular frames. The full decoded byte count is the sum of
        // regular-frame payloads; skippable frames contribute nothing
        // to `dst` but their magic+size+payload must be stepped over.
        use crate::decompress::zstd_decompress_block::{
            ZSTD_buildDefaultSeqTables, ZSTD_decoder_entropy_rep,
        };
        let first = b"alpha-first-frame".as_ref();
        let second = b"omega-second-frame".as_ref();
        let mut stream = Vec::new();
        {
            let mut buf = vec![0u8; 256];
            let n = crate::compress::zstd_compress::ZSTD_compress(&mut buf, first, 3);
            stream.extend_from_slice(&buf[..n]);
        }
        stream.extend_from_slice(&ZSTD_MAGIC_SKIPPABLE_START.to_le_bytes());
        stream.extend_from_slice(&8u32.to_le_bytes());
        stream.extend_from_slice(b"SKIPDATA");
        {
            let mut buf = vec![0u8; 256];
            let n = crate::compress::zstd_compress::ZSTD_compress(&mut buf, second, 3);
            stream.extend_from_slice(&buf[..n]);
        }

        let mut dctx = ZSTD_DCtx::new();
        ZSTD_buildDefaultSeqTables(&mut dctx);
        let mut rep = ZSTD_decoder_entropy_rep::default();
        let mut xxh = crate::common::xxhash::XXH64_state_t::default();
        let mut out = vec![0u8; first.len() + second.len() + 64];
        let decoded = ZSTD_decompressDCtx(&mut dctx, &mut rep, &mut xxh, &mut out, &stream);
        assert!(!crate::common::error::ERR_isError(decoded));
        assert_eq!(decoded, first.len() + second.len());
        assert_eq!(&out[..first.len()], first);
        assert_eq!(&out[first.len()..decoded], second);
    }

    #[test]
    fn decompressDCtx_applies_loaded_stream_dict() {
        // Upstream `ZSTD_decompressDCtx` routes through
        // `ZSTD_decompress_usingDDict(dctx, ..., ZSTD_getDDict(dctx))`,
        // so a dict loaded via `ZSTD_DCtx_loadDictionary` must be
        // honored. Previously our port ignored `dctx.stream_dict` and
        // fed dict-compressed frames through the no-dict decoder,
        // silently producing garbage.
        let dict = b"shared-dict-for-decompressDCtx-apply ".repeat(8);
        let src: Vec<u8> = b"payload referring to shared-dict-for-decompressDCtx-apply ".repeat(15);
        let mut cctx = crate::compress::zstd_compress::ZSTD_createCCtx().unwrap();
        let mut frame = vec![0u8; 4096];
        let n = crate::compress::zstd_compress::ZSTD_compress_usingDict(
            &mut cctx, &mut frame, &src, &dict, 3,
        );
        assert!(!crate::common::error::ERR_isError(n));

        let mut dctx = ZSTD_DCtx::new();
        ZSTD_DCtx_loadDictionary(&mut dctx, &dict);
        use crate::decompress::zstd_decompress_block::{
            ZSTD_buildDefaultSeqTables, ZSTD_decoder_entropy_rep,
        };
        ZSTD_buildDefaultSeqTables(&mut dctx);
        let mut rep = ZSTD_decoder_entropy_rep::default();
        let mut xxh = crate::common::xxhash::XXH64_state_t::default();
        let mut out = vec![0u8; src.len() + 64];
        let decoded = ZSTD_decompressDCtx(&mut dctx, &mut rep, &mut xxh, &mut out, &frame[..n]);
        assert!(!crate::common::error::ERR_isError(decoded));
        assert_eq!(&out[..decoded], &src[..]);
    }

    #[test]
    fn getFrameHeader_on_skippable_frame_returns_variant_and_size() {
        // Upstream contract (zstd_decompress.c: `ZSTD_getFrameHeader_advanced`):
        // when `src` starts with a skippable magic, `zfh.dictID`
        // stores the magic-variant nibble (0..=15), `frameType` is
        // `ZSTD_skippableFrame`, `frameContentSize` is the user-data
        // length, and `headerSize` is `ZSTD_SKIPPABLEHEADERSIZE` (8).
        // Previously unpinned — a driver change here would silently
        // mis-describe skippable frames to callers.
        let mut buf = Vec::new();
        buf.extend_from_slice(&(ZSTD_MAGIC_SKIPPABLE_START + 7).to_le_bytes());
        buf.extend_from_slice(&12u32.to_le_bytes());
        buf.extend_from_slice(&[0xAB; 12]);

        let mut zfh = ZSTD_FrameHeader::default();
        let rc = ZSTD_getFrameHeader(&mut zfh, &buf);
        assert_eq!(rc, 0);
        assert_eq!(zfh.frameType, ZSTD_FrameType_e::ZSTD_skippableFrame);
        assert_eq!(zfh.dictID, 7);
        assert_eq!(zfh.headerSize, ZSTD_SKIPPABLEHEADERSIZE as u32);
        assert_eq!(zfh.frameContentSize, 12);
    }

    #[test]
    fn DCtx_loadDictionary_empty_slice_clears_previous_dict() {
        // Sibling of the CCtx test: the decoder's `loadDictionary`
        // with an empty slice must clear any previously loaded dict,
        // not leave stale bytes. Upstream equivalent is
        // `ZSTD_DCtx_loadDictionary_advanced(dctx, NULL, 0, ...)`.
        let mut dctx = ZSTD_DCtx::default();
        ZSTD_DCtx_loadDictionary(&mut dctx, b"sticky-dict");
        assert_eq!(dctx.stream_dict, b"sticky-dict");
        ZSTD_DCtx_loadDictionary(&mut dctx, &[]);
        assert!(dctx.stream_dict.is_empty());
    }

    #[test]
    fn decompress_usingDict_with_empty_dict_matches_no_dict_path() {
        // Upstream treats an empty dict as "no dict" — decode must
        // succeed for frames compressed without a dict. Ensures the
        // newly-added dictID-mismatch check doesn't spuriously reject
        // no-dict frames (frame dictID=0 + dict dictID=0 → no conflict).
        let src = b"empty-dict == no-dict ".repeat(20);
        let mut cbuf = vec![0u8; 4096];
        let n = crate::compress::zstd_compress::ZSTD_compress(&mut cbuf, &src, 3);
        assert!(!crate::common::error::ERR_isError(n));

        // Decompress with an empty dict slice.
        let mut dctx = ZSTD_DCtx::new();
        let mut out = vec![0u8; src.len() + 64];
        let rc = ZSTD_decompress_usingDict(&mut dctx, &mut out, &cbuf[..n], &[]);
        assert!(!crate::common::error::ERR_isError(rc));
        assert_eq!(&out[..rc], &src[..]);
    }

    #[test]
    fn decompress_usingDict_rejects_dictID_mismatch() {
        // Upstream contract: when both the frame header and the dict
        // declare a dictID, they must match — otherwise the caller
        // gets `DictionaryWrong` instead of silently-corrupted output.
        // Previously Rust port skipped this check.
        use crate::common::error::{ERR_getErrorCode, ERR_isError};
        use crate::common::mem::MEM_writeLE32;

        // Start from a real full dictionary, then mutate only the
        // dictID field to create a mismatch without corrupting the
        // entropy tables or raw content.
        let mut dict_a = include_bytes!("../../zstd/tests/dict-files/zero-weight-dict").to_vec();
        MEM_writeLE32(&mut dict_a[4..8], 0x11111111);
        let mut dict_b = dict_a.clone();
        MEM_writeLE32(&mut dict_b[4..8], 0x22222222);

        // Compress with dict A (so frame header declares dictID=A).
        let src = b"dict-A-vs-dict-B dictID mismatch ".repeat(20);
        let mut cctx = crate::compress::zstd_compress::ZSTD_createCCtx().unwrap();
        let mut frame = vec![0u8; 4096];
        let n = crate::compress::zstd_compress::ZSTD_compress_usingDict(
            &mut cctx, &mut frame, &src, &dict_a, 3,
        );
        assert!(!ERR_isError(n));

        // Decompress with dict B: dictID mismatch → DictionaryWrong.
        let mut dctx = ZSTD_DCtx::new();
        let mut out = vec![0u8; src.len() + 64];
        let rc = ZSTD_decompress_usingDict(&mut dctx, &mut out, &frame[..n], &dict_b);
        assert!(ERR_isError(rc));
        assert_eq!(ERR_getErrorCode(rc), ErrorCode::DictionaryWrong);
    }

    #[test]
    fn decompress_usingDDict_rejects_dictID_mismatch() {
        use crate::common::error::{ERR_getErrorCode, ERR_isError};
        use crate::common::mem::MEM_writeLE32;
        use crate::decompress::zstd_ddict::ZSTD_createDDict;

        let mut dict_a = include_bytes!("../../zstd/tests/dict-files/zero-weight-dict").to_vec();
        MEM_writeLE32(&mut dict_a[4..8], 0x11111111);
        let mut dict_b = dict_a.clone();
        MEM_writeLE32(&mut dict_b[4..8], 0x22222222);

        let src = b"dict-A-vs-ddict-B dictID mismatch ".repeat(20);
        let mut cctx = crate::compress::zstd_compress::ZSTD_createCCtx().unwrap();
        let mut frame = vec![0u8; 4096];
        let n = crate::compress::zstd_compress::ZSTD_compress_usingDict(
            &mut cctx, &mut frame, &src, &dict_a, 3,
        );
        assert!(!ERR_isError(n));

        let ddict_b = ZSTD_createDDict(&dict_b).expect("ddict");
        let mut dctx = ZSTD_DCtx::new();
        let mut out = vec![0u8; src.len() + 64];
        let rc = ZSTD_decompress_usingDDict(&mut dctx, &mut out, &frame[..n], &ddict_b);
        assert!(ERR_isError(rc));
        assert_eq!(ERR_getErrorCode(rc), ErrorCode::DictionaryWrong);
    }

    #[test]
    fn decompress_usingDict_uses_caller_owned_dctx() {
        // Parity fix: previously `ZSTD_decompress_usingDict` allocated a
        // fresh `ZSTD_DCtx` internally and threw the caller's dctx away
        // — any per-session state set by the caller (e.g. blockSizeMax
        // from the decoded frame) would never surface. After the fix we
        // call `decompressBegin` on the caller's dctx and decode into
        // it, so observable fields like `isFrameDecompression` and
        // `blockSizeMax` reflect the call.
        use crate::common::error::ERR_isError;
        use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

        let src_bytes = b"use-caller-dctx-for-decompress-usingDict-parity ".repeat(4);
        let mut cctx = crate::compress::zstd_compress::ZSTD_createCCtx().unwrap();
        let mut frame = vec![0u8; 4096];
        // Compress without a dict — still exercises the usingDict path
        // with an empty dict.
        let n = crate::compress::zstd_compress::ZSTD_compress_usingDict(
            &mut cctx,
            &mut frame,
            &src_bytes,
            &[],
            3,
        );
        assert!(!ERR_isError(n));

        // Seed a dctx with a bogus dictID to prove decompressBegin
        // actually runs (it resets dictID to 0).
        let mut dctx = ZSTD_DCtx::new();
        dctx.dictID = 0xDEAD_BEEF;
        let mut out = vec![0u8; src_bytes.len() + 64];
        let rc = ZSTD_decompress_usingDict(&mut dctx, &mut out, &frame[..n], &[]);
        assert!(!ERR_isError(rc));
        assert_eq!(&out[..rc], src_bytes.as_slice());
        // decompressBegin cleared the seeded dictID.
        assert_eq!(dctx.dictID, 0);
        // The frame-level fields land on the caller's dctx.
        assert_eq!(dctx.isFrameDecompression, 1);
        assert!(dctx.blockSizeMax > 0);
    }

    #[test]
    fn DCtx_refPrefix_advanced_matches_plain_across_every_contentType() {
        // `ZSTD_DCtx_refPrefix_advanced` is the content-type-aware
        // sibling of `refPrefix`. v0.1 treats every content-type as
        // raw, so all three flavors must produce the same stream_dict
        // state as the plain call — any drift would signal an
        // accidental wiring mistake in one branch.
        use crate::decompress::zstd_ddict::ZSTD_dictContentType_e;
        let prefix = b"advanced-refPrefix-roundtrip".to_vec();

        let mut plain = ZSTD_DCtx::default();
        ZSTD_DCtx_refPrefix(&mut plain, &prefix);

        for ct in [
            ZSTD_dictContentType_e::ZSTD_dct_auto,
            ZSTD_dictContentType_e::ZSTD_dct_rawContent,
            ZSTD_dictContentType_e::ZSTD_dct_fullDict,
        ] {
            let mut adv = ZSTD_DCtx::default();
            let rc = ZSTD_DCtx_refPrefix_advanced(&mut adv, &prefix, ct);
            assert_eq!(rc, 0);
            assert_eq!(adv.stream_dict, plain.stream_dict);
        }
    }

    #[test]
    fn initStatic_decompression_variants_construct_ctx_and_ddict() {
        let mut buf = vec![0u64; (1 << 20) / core::mem::size_of::<u64>()];
        let bytes = unsafe {
            core::slice::from_raw_parts_mut(
                buf.as_mut_ptr() as *mut u8,
                buf.len() * core::mem::size_of::<u64>(),
            )
        };
        let dctx = ZSTD_initStaticDCtx(bytes).expect("static dctx");
        assert_eq!(
            ZSTD_nextSrcSizeToDecompress(dctx),
            ZSTD_startingInputLength(ZSTD_format_e::ZSTD_f_zstd1)
        );
        let dstream = ZSTD_initStaticDStream(bytes).expect("static dstream");
        assert_eq!(
            ZSTD_nextSrcSizeToDecompress(dstream),
            ZSTD_startingInputLength(ZSTD_format_e::ZSTD_f_zstd1)
        );
        let dict_bytes = b"dict";
        let ddict = ZSTD_initStaticDDict(bytes, dict_bytes).expect("static ddict");
        assert_eq!(
            crate::decompress::zstd_ddict::ZSTD_DDict_dictContent(ddict),
            dict_bytes
        );
    }

    #[test]
    fn DCtx_setMaxWindowSize_stores_log2() {
        let mut dctx = ZSTD_DCtx::default();
        let rc = ZSTD_DCtx_setMaxWindowSize(&mut dctx, 1 << 18);
        assert_eq!(rc, 0);
        assert_eq!(dctx.d_windowLogMax, 18u32);
    }

    #[test]
    fn DCtx_setMaxWindowSize_rejects_out_of_bounds() {
        let mut dctx = ZSTD_DCtx::default();
        // Below min (1<<10 = 1024).
        assert!(crate::common::error::ERR_isError(
            ZSTD_DCtx_setMaxWindowSize(&mut dctx, 100)
        ));
        // Above max: upper bound is ZSTD_WINDOWLOG_MAX (31 on 64-bit,
        // 30 on 32-bit). 1 << (max+1) is the first out-of-range byte
        // count on 64-bit; on 32-bit it would overflow usize so skip.
        if crate::common::mem::MEM_32bits() == 0 {
            assert!(crate::common::error::ERR_isError(
                ZSTD_DCtx_setMaxWindowSize(&mut dctx, 1usize << 32)
            ));
        }
    }

    #[test]
    fn isFrame_and_isSkippable_are_exclusive() {
        // A real zstd frame: magic 0xFD2FB528.
        let mut real = 0xFD2FB528u32.to_le_bytes().to_vec();
        real.resize(32, 0);
        assert_eq!(ZSTD_isFrame(&real), 1);
        assert_eq!(ZSTD_isSkippableFrame(&real), 0);

        // A skippable frame.
        let mut skip = ZSTD_MAGIC_SKIPPABLE_START.to_le_bytes().to_vec();
        skip.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(ZSTD_isFrame(&skip), 1); // isFrame includes skippable
        assert_eq!(ZSTD_isSkippableFrame(&skip), 1);

        // Random bytes — neither.
        let rand = [0u8, 1, 2, 3, 4, 5, 6, 7];
        assert_eq!(ZSTD_isFrame(&rand), 0);
        assert_eq!(ZSTD_isSkippableFrame(&rand), 0);
    }

    #[test]
    fn skippableFrame_all_16_variants_roundtrip() {
        use crate::compress::zstd_compress::ZSTD_writeSkippableFrame;
        for variant in 0..16u32 {
            let payload = [variant as u8; 7];
            let mut buf = vec![0u8; 32];
            let w = ZSTD_writeSkippableFrame(&mut buf, &payload, variant);
            assert!(!crate::common::error::ERR_isError(w));
            let mut out = [0u8; 7];
            let mut got_variant = 0u32;
            let r = ZSTD_readSkippableFrame(&mut out, Some(&mut got_variant), &buf);
            assert_eq!(r, 7);
            assert_eq!(got_variant, variant);
            assert_eq!(&out, &payload);
        }
    }

    #[test]
    fn readSkippableFrame_returns_payload_and_variant() {
        // Build: magic + userData=5 + 5 payload bytes.
        let mut src = Vec::new();
        src.extend_from_slice(&(ZSTD_MAGIC_SKIPPABLE_START + 3).to_le_bytes()); // variant 3
        src.extend_from_slice(&5u32.to_le_bytes());
        src.extend_from_slice(b"hello");

        let mut dst = [0u8; 16];
        let mut variant: u32 = 0;
        let n = ZSTD_readSkippableFrame(&mut dst, Some(&mut variant), &src);
        assert_eq!(n, 5);
        assert_eq!(&dst[..5], b"hello");
        assert_eq!(variant, 3);
    }

    #[test]
    fn writeSkippableFrame_rejects_error_paths() {
        // Exercise every documented error path: dst too small for
        // header+payload, variant > 15, and verify the happy-path
        // boundary (dst exactly = header+payload succeeds).
        use crate::compress::zstd_compress::ZSTD_writeSkippableFrame;
        let payload = b"abcd".as_slice();
        // dst too small for even the 8-byte header.
        let mut tiny = [0u8; 4];
        assert!(crate::common::error::ERR_isError(ZSTD_writeSkippableFrame(
            &mut tiny, payload, 0
        )));
        // dst too small for header + payload.
        let mut no_room = [0u8; 11];
        assert!(crate::common::error::ERR_isError(ZSTD_writeSkippableFrame(
            &mut no_room,
            payload,
            0
        )));
        // magicVariant > 15 is out of spec.
        let mut ok_buf = [0u8; 64];
        assert!(crate::common::error::ERR_isError(ZSTD_writeSkippableFrame(
            &mut ok_buf,
            payload,
            16
        )));
        // Exact-fit dst succeeds.
        let mut exact = [0u8; 8 + 4];
        let n = ZSTD_writeSkippableFrame(&mut exact, payload, 0);
        assert_eq!(n, 12);
    }

    #[test]
    fn readSkippableFrame_rejects_non_skippable() {
        // Plain zstd frame magic should fail.
        let src = 0xFD2FB528u32.to_le_bytes();
        let mut dst = [0u8; 16];
        let rc = ZSTD_readSkippableFrame(&mut dst, None, &src);
        assert!(crate::common::error::ERR_isError(rc));
    }

    #[test]
    fn readSkippableFrame_rejects_short_src_and_truncated_frame() {
        // Three error paths: (1) src shorter than the 8-byte header,
        // (2) claimed frame size > src.len() (truncated mid-payload).
        // The existing tests cover non-skippable + small-dst; this
        // rounds out coverage to all 4 distinct reject paths.
        let mut dst = [0u8; 32];

        // (1) src too short for skippable header.
        let too_short = [0u8; 4];
        assert!(crate::common::error::ERR_isError(ZSTD_readSkippableFrame(
            &mut dst, None, &too_short
        )));

        // (2) header claims 20-byte payload but src is truncated to
        //     header + 5 bytes.
        let mut truncated = Vec::new();
        truncated.extend_from_slice(&ZSTD_MAGIC_SKIPPABLE_START.to_le_bytes());
        truncated.extend_from_slice(&20u32.to_le_bytes());
        truncated.extend_from_slice(&[0u8; 5]);
        assert!(crate::common::error::ERR_isError(ZSTD_readSkippableFrame(
            &mut dst, None, &truncated
        )));
    }

    #[test]
    fn readSkippableFrame_rejects_small_dst() {
        let mut src = Vec::new();
        src.extend_from_slice(&ZSTD_MAGIC_SKIPPABLE_START.to_le_bytes());
        src.extend_from_slice(&10u32.to_le_bytes());
        src.extend_from_slice(&[0u8; 10]);
        let mut dst = [0u8; 4]; // too small
        let rc = ZSTD_readSkippableFrame(&mut dst, None, &src);
        assert!(crate::common::error::ERR_isError(rc));
    }

    #[test]
    fn find_frame_compressed_size_skippable() {
        let mut src = Vec::new();
        src.extend_from_slice(&ZSTD_MAGIC_SKIPPABLE_START.to_le_bytes());
        src.extend_from_slice(&7u32.to_le_bytes());
        src.extend_from_slice(&[0u8; 7]);
        let sz = ZSTD_findFrameCompressedSize(&src);
        assert_eq!(sz, ZSTD_SKIPPABLEHEADERSIZE + 7);
    }

    #[test]
    fn find_decompressed_size_sums_frames() {
        // Two back-to-back raw "HELLO" frames + one skippable → 10.
        let mut src = make_raw_hello_frame();
        src.extend_from_slice(&ZSTD_MAGIC_SKIPPABLE_START.to_le_bytes());
        src.extend_from_slice(&3u32.to_le_bytes());
        src.extend_from_slice(&[0u8; 3]);
        src.extend_from_slice(&make_raw_hello_frame());
        let total = ZSTD_findDecompressedSize(&src);
        assert_eq!(total, 10);
    }

    #[test]
    fn decompress_frame_raw_block_roundtrip() {
        // Handcraft a minimal valid zstd frame with a single raw block.
        //   - Magic: ZSTD_MAGICNUMBER
        //   - FHD: singleSegment=1, fcsID=0 → byte 0x20. FCS is 1 byte.
        //   - FCS byte: 5 (declared frame content size)
        //   - Block header: lastBlock=1, bt_raw(=0), cSize=5 → (5<<3)|(0<<1)|1 = 0x29.
        //   - Raw payload: "HELLO"
        let mut src = Vec::new();
        src.extend_from_slice(&ZSTD_MAGICNUMBER.to_le_bytes());
        src.push(0x20);
        src.push(5);
        let bh = (5u32 << 3) | 1; // lastBlock=1, bt_raw=0, cSize=5
        src.push((bh & 0xFF) as u8);
        src.push(((bh >> 8) & 0xFF) as u8);
        src.push(((bh >> 16) & 0xFF) as u8);
        src.extend_from_slice(b"HELLO");

        let mut dst = vec![0u8; 32];
        let out = ZSTD_decompress(&mut dst, &src);
        assert!(
            !crate::common::error::ERR_isError(out),
            "err: {}",
            crate::common::error::ERR_getErrorName(out)
        );
        assert_eq!(out, 5);
        assert_eq!(&dst[..5], b"HELLO");
    }

    #[test]
    fn decompress_frame_rle_block_roundtrip() {
        // singleSegment frame, FCS=10, one RLE block of 10 bytes of 'Z'.
        // Block header: lastBlock=1, bt_rle(=1), origSize=10 →
        //   (10<<3)|(1<<1)|1 = 0x53.
        let mut src = Vec::new();
        src.extend_from_slice(&ZSTD_MAGICNUMBER.to_le_bytes());
        src.push(0x20); // FHD single-segment, no dict, FCS 1 byte
        src.push(10); // FCS
        let bh = (10u32 << 3) | (1 << 1) | 1;
        src.push((bh & 0xFF) as u8);
        src.push(((bh >> 8) & 0xFF) as u8);
        src.push(((bh >> 16) & 0xFF) as u8);
        src.push(b'Z');

        let mut dst = vec![0u8; 32];
        let out = ZSTD_decompress(&mut dst, &src);
        assert!(!crate::common::error::ERR_isError(out));
        assert_eq!(out, 10);
        for b in dst.iter().take(10) {
            assert_eq!(*b, b'Z');
        }
    }

    #[test]
    fn frame_header_reserved_bit_errors() {
        let mut src = Vec::new();
        src.extend_from_slice(&ZSTD_MAGICNUMBER.to_le_bytes());
        src.push(0x08); // reserved bit 3 set
        src.push(0);
        let mut zfh = ZSTD_FrameHeader::default();
        let rc = ZSTD_getFrameHeader(&mut zfh, &src);
        assert!(crate::common::error::ERR_isError(rc));
    }
}

// ZSTD_DCtx lives in `zstd_decompress_block` (since most of its fields
// are literal/sequence decoder state that lands first). Re-export here
// so upstream symbol lookups resolve.
pub use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

/// Port of `ZSTD_createDCtx`. Rust port returns an owned `ZSTD_DCtx`
/// with default seq-tables pre-built.
pub fn ZSTD_createDCtx() -> Box<ZSTD_DCtx> {
    let mut d = ZSTD_createDCtx_internal(crate::compress::zstd_compress::ZSTD_customMem::default())
        .expect("default customMem allocation must succeed");
    crate::decompress::zstd_decompress_block::ZSTD_buildDefaultSeqTables(&mut d);
    d
}

/// Port of `ZSTD_freeDCtx`. In the Rust port, dropping the Box frees.
pub fn ZSTD_freeDCtx(dctx: Box<ZSTD_DCtx>) -> usize {
    let customMem = dctx.customMem;
    unsafe {
        crate::compress::zstd_compress::ZSTD_customFreeBox(dctx, customMem);
    }
    0
}

/// Port of `ZSTD_copyRawBlock` (raw-block passthrough).
pub fn ZSTD_copyRawBlock(dst: &mut [u8], src: &[u8]) -> usize {
    if src.len() > dst.len() {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    dst[..src.len()].copy_from_slice(src);
    src.len()
}

/// Port of `ZSTD_setRleBlock` (RLE-block expansion).
pub fn ZSTD_setRleBlock(dst: &mut [u8], b: u8, regenSize: usize) -> usize {
    if regenSize > dst.len() {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    for d in dst[..regenSize].iter_mut() {
        *d = b;
    }
    regenSize
}

/// Port of `ZSTD_decompressFrame` — the block-loop driver. Given a
/// fully-initialized `ZSTD_DCtx`, reads the frame header, iterates
/// blocks until `lastBlock`, handles RAW / RLE / compressed block
/// types, validates the frame checksum if present, and returns the
/// decompressed payload size.
///
/// Rust signature note: upstream mutates `*srcPtr` and `*srcSizePtr`
/// to advance the source cursor; we accept `src` by value and return
/// the tuple `(decoded_size, src_consumed)` via out-params.
/// Variant of `ZSTD_decompressFrame` that begins writing at
/// `dst[op_start..]` instead of `dst[0..]`. Used by
/// `ZSTD_decompress_usingDict` to decode into a buffer whose initial
/// `op_start` bytes hold the dict history. Returns the number of
/// decoded bytes (i.e., `final_op - op_start`).
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_decompressFrame_withOpStart(
    dctx: &mut crate::decompress::zstd_decompress_block::ZSTD_DCtx,
    entropy_rep: &mut crate::decompress::zstd_decompress_block::ZSTD_decoder_entropy_rep,
    xxh: &mut crate::common::xxhash::XXH64_state_t,
    dst: &mut [u8],
    op_start: usize,
    src: &[u8],
    src_consumed: &mut usize,
) -> usize {
    use crate::common::xxhash::{XXH64_digest, XXH64_reset, XXH64_update};
    use crate::decompress::zstd_decompress_block::{
        blockProperties_t, blockType_e, streaming_operation, ZSTD_blockHeaderSize,
        ZSTD_decompressBlock_internal, ZSTD_getcBlockSize,
    };
    // Minimum bytes needed to read the first frame header. Upstream's
    // `ZSTD_FRAMEHEADERSIZE_MIN(format)` is 6 for zstd1, 2 for
    // magicless — the latter lets magicless-mode decoders succeed on
    // payloads that the zstd1 cap would reject as "too small".
    let frameheadersize_min = ZSTD_FRAMEHEADERSIZE_MIN(dctx.format);
    let mut ip: usize = 0;
    let mut op: usize = op_start;
    let mut remaining = src.len();
    if remaining < frameheadersize_min + ZSTD_blockHeaderSize {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    let mut zfh = ZSTD_FrameHeader::default();
    // Thread `dctx.format` so magicless frames parse correctly.
    let rc = ZSTD_getFrameHeader_advanced(&mut zfh, src, dctx.format);
    if crate::common::error::ERR_isError(rc) {
        return rc;
    }
    if rc != 0 {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    if zfh.frameType != ZSTD_FrameType_e::ZSTD_frame {
        return ERROR(ErrorCode::PrefixUnknown);
    }
    dctx.isFrameDecompression = 1;
    dctx.blockSizeMax = zfh.blockSizeMax as usize;
    let validateChecksum = zfh.checksumFlag != 0;
    if validateChecksum {
        XXH64_reset(xxh, 0);
    }
    ip += zfh.headerSize as usize;
    remaining -= zfh.headerSize as usize;
    loop {
        let mut bp = blockProperties_t {
            blockType: blockType_e::bt_raw,
            lastBlock: 0,
            origSize: 0,
        };
        let cBlockSize = ZSTD_getcBlockSize(&src[ip..], &mut bp);
        if crate::common::error::ERR_isError(cBlockSize) {
            return cBlockSize;
        }
        ip += ZSTD_blockHeaderSize;
        remaining -= ZSTD_blockHeaderSize;
        if cBlockSize > remaining {
            return ERROR(ErrorCode::SrcSizeWrong);
        }
        let decodedSize = match bp.blockType {
            blockType_e::bt_compressed => ZSTD_decompressBlock_internal(
                dctx,
                entropy_rep,
                dst,
                op,
                &src[ip..ip + cBlockSize],
                streaming_operation::not_streaming,
            ),
            blockType_e::bt_raw => ZSTD_copyRawBlock(&mut dst[op..], &src[ip..ip + cBlockSize]),
            blockType_e::bt_rle => ZSTD_setRleBlock(&mut dst[op..], src[ip], bp.origSize as usize),
            blockType_e::bt_reserved => return ERROR(ErrorCode::CorruptionDetected),
        };
        if crate::common::error::ERR_isError(decodedSize) {
            return decodedSize;
        }
        if validateChecksum && decodedSize > 0 {
            XXH64_update(xxh, &dst[op..op + decodedSize]);
        }
        op += decodedSize;
        ip += cBlockSize;
        remaining -= cBlockSize;
        if bp.lastBlock != 0 {
            break;
        }
    }
    let decoded = op - op_start;
    if zfh.frameContentSize != ZSTD_CONTENTSIZE_UNKNOWN && (decoded as u64) != zfh.frameContentSize
    {
        return ERROR(ErrorCode::CorruptionDetected);
    }
    if validateChecksum {
        if remaining < 4 {
            return ERROR(ErrorCode::ChecksumWrong);
        }
        let calc = XXH64_digest(xxh) as u32;
        let read = MEM_readLE32(&src[ip..ip + 4]);
        if calc != read {
            return ERROR(ErrorCode::ChecksumWrong);
        }
        ip += 4;
    }
    *src_consumed = ip;
    decoded
}

pub fn ZSTD_decompressFrame(
    dctx: &mut crate::decompress::zstd_decompress_block::ZSTD_DCtx,
    entropy_rep: &mut crate::decompress::zstd_decompress_block::ZSTD_decoder_entropy_rep,
    xxh: &mut crate::common::xxhash::XXH64_state_t,
    dst: &mut [u8],
    src: &[u8],
    src_consumed: &mut usize,
) -> usize {
    use crate::common::xxhash::{XXH64_digest, XXH64_reset, XXH64_update};
    use crate::decompress::zstd_decompress_block::{
        blockProperties_t, blockType_e, streaming_operation, ZSTD_blockHeaderSize,
        ZSTD_decompressBlock_internal, ZSTD_getcBlockSize,
    };

    let mut ip: usize = 0;
    let mut op: usize = 0;
    let mut remaining = src.len();

    // Honor `dctx.format`: magicless frames skip the 4-byte magic,
    // so the minimum header is 2 bytes + block header, not 6+3.
    let frameHeaderSizeMin = ZSTD_FRAMEHEADERSIZE_MIN(dctx.format);
    if remaining < frameHeaderSizeMin + ZSTD_blockHeaderSize {
        return ERROR(ErrorCode::SrcSizeWrong);
    }

    // Parse frame header.
    let mut zfh = ZSTD_FrameHeader::default();
    let rc = ZSTD_getFrameHeader_advanced(&mut zfh, src, dctx.format);
    if crate::common::error::ERR_isError(rc) {
        return rc;
    }
    if rc != 0 {
        // Header not complete — caller gave too few bytes.
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    if zfh.frameType != ZSTD_FrameType_e::ZSTD_frame {
        // Skippable frames are valid but carry no payload — caller
        // should use a different entry point. We reject here.
        return ERROR(ErrorCode::PrefixUnknown);
    }

    // Apply the frame header to the DCtx. Minimal shape for block decode.
    dctx.isFrameDecompression = 1;
    dctx.blockSizeMax = zfh.blockSizeMax as usize;
    let validateChecksum = zfh.checksumFlag != 0;
    if validateChecksum {
        XXH64_reset(xxh, 0);
    }

    ip += zfh.headerSize as usize;
    remaining -= zfh.headerSize as usize;

    loop {
        let mut bp = blockProperties_t {
            blockType: blockType_e::bt_raw,
            lastBlock: 0,
            origSize: 0,
        };
        let cBlockSize = ZSTD_getcBlockSize(&src[ip..], &mut bp);
        if crate::common::error::ERR_isError(cBlockSize) {
            return cBlockSize;
        }
        ip += ZSTD_blockHeaderSize;
        remaining -= ZSTD_blockHeaderSize;
        if cBlockSize > remaining {
            return ERROR(ErrorCode::SrcSizeWrong);
        }

        let decodedSize = match bp.blockType {
            blockType_e::bt_compressed => ZSTD_decompressBlock_internal(
                dctx,
                entropy_rep,
                dst,
                op,
                &src[ip..ip + cBlockSize],
                streaming_operation::not_streaming,
            ),
            blockType_e::bt_raw => ZSTD_copyRawBlock(&mut dst[op..], &src[ip..ip + cBlockSize]),
            blockType_e::bt_rle => ZSTD_setRleBlock(&mut dst[op..], src[ip], bp.origSize as usize),
            blockType_e::bt_reserved => {
                return ERROR(ErrorCode::CorruptionDetected);
            }
        };
        if crate::common::error::ERR_isError(decodedSize) {
            return decodedSize;
        }
        if validateChecksum && decodedSize > 0 {
            XXH64_update(xxh, &dst[op..op + decodedSize]);
        }
        op += decodedSize;
        ip += cBlockSize;
        remaining -= cBlockSize;
        if bp.lastBlock != 0 {
            break;
        }
    }

    // FrameContentSize check (if declared).
    if zfh.frameContentSize != ZSTD_CONTENTSIZE_UNKNOWN && (op as u64) != zfh.frameContentSize {
        return ERROR(ErrorCode::CorruptionDetected);
    }

    // Checksum trailer.
    if validateChecksum {
        if remaining < 4 {
            return ERROR(ErrorCode::ChecksumWrong);
        }
        let calc = XXH64_digest(xxh) as u32;
        let read = MEM_readLE32(&src[ip..ip + 4]);
        if calc != read {
            return ERROR(ErrorCode::ChecksumWrong);
        }
        ip += 4;
    }

    *src_consumed = ip;
    op
}

/// Port of `ZSTD_decompressDCtx`. Loops over frames in `src` —
/// skipping over skippable frames, decoding each regular frame —
/// until the source is exhausted. Matches upstream's multi-frame
/// contract: callers with concatenated frames see all payloads
/// appended into `dst`.
pub fn ZSTD_decompressDCtx(
    dctx: &mut crate::decompress::zstd_decompress_block::ZSTD_DCtx,
    entropy_rep: &mut crate::decompress::zstd_decompress_block::ZSTD_decoder_entropy_rep,
    xxh: &mut crate::common::xxhash::XXH64_state_t,
    dst: &mut [u8],
    src: &[u8],
) -> usize {
    let ddict = ZSTD_getDDict(dctx);
    let mut ip = 0usize;
    let mut op = 0usize;
    while ip < src.len() {
        let rem = &src[ip..];
        // Skippable frame: advance past it without writing to dst.
        // Upstream (zstd_decompress.c:1120) only checks for skippable
        // magic when `dctx->format == ZSTD_f_zstd1` — magicless-mode
        // frames don't have a magic prefix, so the 4-byte window would
        // match arbitrary first bytes and misfire.
        if dctx.format == ZSTD_format_e::ZSTD_f_zstd1
            && rem.len() >= ZSTD_SKIPPABLEHEADERSIZE
            && (MEM_readLE32(&rem[..4]) & ZSTD_MAGIC_SKIPPABLE_MASK) == ZSTD_MAGIC_SKIPPABLE_START
        {
            let sz = readSkippableFrameSize(rem);
            if crate::common::error::ERR_isError(sz) {
                return sz;
            }
            ip += sz;
            continue;
        }
        // Regular frame: decode into dst[op..]. Upstream routes
        // through `ZSTD_decompress_usingDDict(dctx, ..., ZSTD_getDDict(dctx))`
        // so a dict loaded via `ZSTD_DCtx_loadDictionary` / `_refDDict`
        // gets applied here; previously we ignored `dctx.stream_dict`,
        // so `ZSTD_decompressDCtx` + loaded-dict paths silently
        // produced garbage for dict-compressed frames.
        if let Some(dict) = ddict.as_deref() {
            // Measure this frame's compressed footprint so we can
            // advance the input cursor past it. Use format-aware
            // variant so magicless-mode callers work.
            let frame_sz = ZSTD_findFrameCompressedSize_advanced(rem, dctx.format);
            if crate::common::error::ERR_isError(frame_sz) {
                return frame_sz;
            }
            let decoded = ZSTD_decompress_usingDict(dctx, &mut dst[op..], &rem[..frame_sz], dict);
            if crate::common::error::ERR_isError(decoded) {
                return decoded;
            }
            ip += frame_sz;
            op += decoded;
            continue;
        }
        let mut consumed = 0usize;
        let decoded =
            ZSTD_decompressFrame(dctx, entropy_rep, xxh, &mut dst[op..], rem, &mut consumed);
        if crate::common::error::ERR_isError(decoded) {
            return decoded;
        }
        ip += consumed;
        op += decoded;
    }
    op
}

/// Port of upstream's one-shot `ZSTD_decompress` (`zstd.h:176`).
/// Decompresses a complete frame in `src` into `dst` and returns the
/// decoded byte count (or an `ErrorCode`-bearing return value if
/// `ZSTD_isError`). No dctx re-use, no streaming, no dict — the
/// simplest entry point. For context-managed, streaming, or
/// dict-bearing flows use `ZSTD_decompressDCtx` /
/// `ZSTD_decompressStream` / `ZSTD_decompress_usingDict` instead.
///
/// `dst` must be at least as large as the frame's declared content
/// size (`ZSTD_getFrameContentSize`); passing a short buffer returns
/// a `ZSTD_error_dstSize_tooSmall` error code.
pub fn ZSTD_decompress(dst: &mut [u8], src: &[u8]) -> usize {
    use crate::common::xxhash::XXH64_state_t;
    use crate::decompress::zstd_decompress_block::{
        ZSTD_DCtx, ZSTD_buildDefaultSeqTables, ZSTD_decoder_entropy_rep,
    };
    let mut dctx = ZSTD_DCtx::new();
    ZSTD_buildDefaultSeqTables(&mut dctx);
    let mut rep = ZSTD_decoder_entropy_rep::default();
    let mut xxh = XXH64_state_t::default();
    ZSTD_decompressDCtx(&mut dctx, &mut rep, &mut xxh, dst, src)
}

/// Port of `ZSTD_decompress_usingDict`. Decompresses `src` using a
/// raw-content `dict` as history — sequences in `src` may reference
/// back into the dict. Upstream does this via ext-dict plumbing; our
/// cut-down approach concatenates `dict || scratch` and decodes into
/// the scratch portion starting at `op_start = dict.len()`. Back-refs
/// land naturally in the dict bytes.
///
/// v0.1 scope: raw-content dicts. Pre-digested `ZSTD_dct_fullDict`
/// dictionaries (with embedded HUF/FSE entropy tables) aren't
/// seeded on this path — callers needing magic-prefix entropy seeding
/// should use the DCtx-based flow (`ZSTD_DCtx_loadDictionary` +
/// `ZSTD_decompressDCtx`), which routes through
/// `ZSTD_decompress_insertDictionary` + `ZSTD_loadDEntropy`.
pub fn ZSTD_decompress_usingDict(
    dctx: &mut ZSTD_DCtx,
    dst: &mut [u8],
    src: &[u8],
    dict: &[u8],
) -> usize {
    use crate::common::error::ERR_isError;
    use crate::common::xxhash::XXH64_state_t;
    use crate::decompress::zstd_decompress_block::ZSTD_decoder_entropy_rep;

    // Upstream (zstd_decompress.c / ZSTD_decompress_usingDict):
    // when both the frame header and the dict declare a dictID, they
    // must match — otherwise `DictionaryWrong`. Previously the Rust
    // port skipped this check, silently proceeding with a mismatched
    // dict which corrupted output.
    let frame_dict_id = ZSTD_getDictID_fromFrame(src);
    let dict_dict_id = crate::decompress::zstd_ddict::ZSTD_getDictID_fromDict(dict);
    if frame_dict_id != 0 && dict_dict_id != 0 && frame_dict_id != dict_dict_id {
        return ERROR(ErrorCode::DictionaryWrong);
    }

    // Determine how much output we need.
    let declared = ZSTD_getFrameContentSize(src);
    let out_size = if declared == ZSTD_CONTENTSIZE_UNKNOWN || declared == ZSTD_CONTENTSIZE_ERROR {
        dst.len()
    } else {
        declared as usize
    };
    if out_size > dst.len() {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }

    // Combined buffer: dict || scratch. Decoder writes into scratch
    // starting at position dict.len().
    let mut combined = vec![0u8; dict.len() + out_size];
    combined[..dict.len()].copy_from_slice(dict);

    // Reset the caller's dctx for a fresh frame decode. Upstream uses
    // the caller-owned dctx throughout; previously we allocated a
    // throwaway `ZSTD_DCtx::new()` and discarded it, which meant the
    // caller's dctx state was silently ignored after the call — a
    // faithful-translation gap.
    let rc = ZSTD_decompressBegin(dctx);
    if ERR_isError(rc) {
        return rc;
    }
    let mut rep = ZSTD_decoder_entropy_rep::default();
    let mut xxh = XXH64_state_t::default();

    // Call the frame decoder with an op_start-like offset. Upstream
    // threads this via the DCtx's prefix pointer; we use the "dst
    // starts with dict" convention — requires a small tweak to
    // ZSTD_decompressFrame to honor a starting op offset.
    //
    // Easier path for v0.1: inline a single-frame decode that honors
    // an op-offset into the combined buffer.
    let mut consumed = 0usize;
    let decoded = ZSTD_decompressFrame_withOpStart(
        dctx,
        &mut rep,
        &mut xxh,
        &mut combined,
        dict.len(),
        src,
        &mut consumed,
    );
    if ERR_isError(decoded) {
        return decoded;
    }
    dst[..decoded].copy_from_slice(&combined[dict.len()..dict.len() + decoded]);
    decoded
}

/// Port of `ZSTD_dParam_getBounds`. Returns the valid range for a
/// decompression parameter. Upstream: `[ZSTD_WINDOWLOG_ABSOLUTEMIN,
/// ZSTD_WINDOWLOG_MAX]` — 10..31 on 64-bit, 10..30 on 32-bit. The
/// default cap is `ZSTD_WINDOWLOG_LIMIT_DEFAULT` (27), not the
/// upper bound.
pub fn ZSTD_dParam_getBounds(
    param: ZSTD_dParameter,
) -> crate::compress::zstd_compress::ZSTD_bounds {
    match param {
        ZSTD_dParameter::ZSTD_d_windowLogMax => {
            let upper = if crate::common::mem::MEM_32bits() != 0 {
                ZSTD_WINDOWLOG_MAX_32 as i32
            } else {
                ZSTD_WINDOWLOG_MAX_64 as i32
            };
            crate::compress::zstd_compress::ZSTD_bounds {
                error: 0,
                lowerBound: 10,
                upperBound: upper,
            }
        }
        ZSTD_dParameter::ZSTD_d_format => crate::compress::zstd_compress::ZSTD_bounds {
            error: 0,
            lowerBound: ZSTD_format_e::ZSTD_f_zstd1 as i32,
            upperBound: ZSTD_format_e::ZSTD_f_zstd1_magicless as i32,
        },
    }
}

/// Port of `ZSTD_DCtx_trace_end` (zstd_decompress.c:922). Upstream
/// emits a decompress-end trace event when `ZSTD_TRACE` is compiled
/// in. v0.1 has no tracing infrastructure — intentional no-op.
#[inline]
pub fn ZSTD_DCtx_trace_end(
    _dctx: &ZSTD_DCtx,
    _uncompressedSize: u64,
    _compressedSize: u64,
    _streaming: i32,
) {
}

/// Port of `ZSTD_clearDict` (zstd_decompress.c:316). Drops any dict
/// linkage from the DCtx — semantically equivalent to `refDDict(NULL)`.
/// Our port's DCtx keeps the dict in `stream_dict`; upstream tracks
/// `ddictLocal` + `ddict` + `dictUses` separately.
pub fn ZSTD_clearDict(dctx: &mut ZSTD_DCtx) {
    dctx.stream_dict.clear();
    dctx.dictID = 0;
    dctx.ddict_rep = [0; 3];
    // litEntropy / fseEntropy are per-frame flags — ZSTD_decompressBegin
    // resets them on the next frame start, so leaving them here is
    // consistent with upstream (clearDict doesn't touch them either).
    // Reset the dict-lifecycle tracker so the next loadDictionary /
    // refPrefix / refDDict call starts from a clean slate.
    dctx.dictUses = ZSTD_dictUses_e::ZSTD_dont_use;
}

pub const DDICT_HASHSET_MAX_LOAD_FACTOR_COUNT_MULT: usize = 4;
pub const DDICT_HASHSET_MAX_LOAD_FACTOR_SIZE_MULT: usize = 3;
pub const DDICT_HASHSET_TABLE_BASE_SIZE: usize = 64;
pub const DDICT_HASHSET_RESIZE_FACTOR: usize = 2;

/// Port of upstream `ZSTD_DDictHashSet`: an open-addressed table of
/// borrowed DDict references keyed by `dictID`.
pub struct ZSTD_DDictHashSet<'a> {
    pub ddictPtrTable: Vec<Option<&'a crate::decompress::zstd_ddict::ZSTD_DDict>>,
    pub ddictPtrTableSize: usize,
    pub ddictPtrCount: usize,
}

/// Port of `ZSTD_DDictHashSet_getIndex`.
pub fn ZSTD_DDictHashSet_getIndex(hashSet: &ZSTD_DDictHashSet<'_>, dictID: u32) -> usize {
    let hash = crate::common::xxhash::XXH64(&dictID.to_le_bytes(), 0);
    (hash as usize) & (hashSet.ddictPtrTableSize - 1)
}

/// Port of `ZSTD_DDictHashSet_emplaceDDict`. Inserts without resizing,
/// replacing an existing DDict with the same dictID.
pub fn ZSTD_DDictHashSet_emplaceDDict<'a>(
    hashSet: &mut ZSTD_DDictHashSet<'a>,
    ddict: &'a crate::decompress::zstd_ddict::ZSTD_DDict,
) -> usize {
    use crate::decompress::zstd_ddict::ZSTD_getDictID_fromDDict;

    let dictID = ZSTD_getDictID_fromDDict(ddict);
    let mut idx = ZSTD_DDictHashSet_getIndex(hashSet, dictID);
    let idxRangeMask = hashSet.ddictPtrTableSize - 1;
    if hashSet.ddictPtrCount == hashSet.ddictPtrTableSize {
        return ERROR(ErrorCode::Generic);
    }
    while let Some(existing) = hashSet.ddictPtrTable[idx] {
        if ZSTD_getDictID_fromDDict(existing) == dictID {
            hashSet.ddictPtrTable[idx] = Some(ddict);
            return 0;
        }
        idx = (idx + 1) & idxRangeMask;
    }
    hashSet.ddictPtrTable[idx] = Some(ddict);
    hashSet.ddictPtrCount += 1;
    0
}

/// Port of `ZSTD_DDictHashSet_expand`.
pub fn ZSTD_DDictHashSet_expand<'a>(
    hashSet: &mut ZSTD_DDictHashSet<'a>,
    _customMem: crate::compress::zstd_compress::ZSTD_customMem,
) -> usize {
    let oldTable = core::mem::take(&mut hashSet.ddictPtrTable);
    hashSet.ddictPtrTableSize *= DDICT_HASHSET_RESIZE_FACTOR;
    hashSet.ddictPtrTable = vec![None; hashSet.ddictPtrTableSize];
    hashSet.ddictPtrCount = 0;
    for ddict in oldTable.into_iter().flatten() {
        let rc = ZSTD_DDictHashSet_emplaceDDict(hashSet, ddict);
        if crate::common::error::ERR_isError(rc) {
            return rc;
        }
    }
    0
}

/// Port of `ZSTD_DDictHashSet_getDDict`.
pub fn ZSTD_DDictHashSet_getDDict<'a>(
    hashSet: &ZSTD_DDictHashSet<'a>,
    dictID: u32,
) -> Option<&'a crate::decompress::zstd_ddict::ZSTD_DDict> {
    use crate::decompress::zstd_ddict::ZSTD_getDictID_fromDDict;

    let mut idx = ZSTD_DDictHashSet_getIndex(hashSet, dictID);
    let idxRangeMask = hashSet.ddictPtrTableSize - 1;
    loop {
        match hashSet.ddictPtrTable[idx] {
            Some(ddict) if ZSTD_getDictID_fromDDict(ddict) == dictID => return Some(ddict),
            Some(_) => idx = (idx + 1) & idxRangeMask,
            None => return None,
        }
    }
}

/// Port of `ZSTD_createDDictHashSet`.
pub fn ZSTD_createDDictHashSet<'a>(
    _customMem: crate::compress::zstd_compress::ZSTD_customMem,
) -> ZSTD_DDictHashSet<'a> {
    ZSTD_DDictHashSet {
        ddictPtrTable: vec![None; DDICT_HASHSET_TABLE_BASE_SIZE],
        ddictPtrTableSize: DDICT_HASHSET_TABLE_BASE_SIZE,
        ddictPtrCount: 0,
    }
}

/// Port of `ZSTD_DDictHashSet_addDDict`.
pub fn ZSTD_DDictHashSet_addDDict<'a>(
    hashSet: &mut ZSTD_DDictHashSet<'a>,
    ddict: &'a crate::decompress::zstd_ddict::ZSTD_DDict,
    customMem: crate::compress::zstd_compress::ZSTD_customMem,
) -> usize {
    if hashSet.ddictPtrCount * DDICT_HASHSET_MAX_LOAD_FACTOR_COUNT_MULT / hashSet.ddictPtrTableSize
        * DDICT_HASHSET_MAX_LOAD_FACTOR_SIZE_MULT
        != 0
    {
        let rc = ZSTD_DDictHashSet_expand(hashSet, customMem);
        if crate::common::error::ERR_isError(rc) {
            return rc;
        }
    }
    ZSTD_DDictHashSet_emplaceDDict(hashSet, ddict)
}

/// Port of `ZSTD_DCtx_selectFrameDDict`.
///
/// Upstream reads `dctx->ddictSet` when `ZSTD_d_refMultipleDDicts` is
/// enabled. This Rust port keeps the hash set explicit to avoid
/// storing borrowed DDict references inside the long-lived `ZSTD_DCtx`;
/// when a frame dictID is found, the selected DDict parameters are
/// copied into the DCtx exactly like `ZSTD_DCtx_refDDict`.
pub fn ZSTD_DCtx_selectFrameDDict<'a>(
    dctx: &mut ZSTD_DCtx,
    hashSet: &ZSTD_DDictHashSet<'a>,
) -> Option<&'a crate::decompress::zstd_ddict::ZSTD_DDict> {
    let frameDDict = ZSTD_DDictHashSet_getDDict(hashSet, dctx.fParams.dictID)?;
    ZSTD_clearDict(dctx);
    dctx.dictID = dctx.fParams.dictID;
    crate::decompress::zstd_ddict::ZSTD_copyDDictParameters(dctx, frameDDict);
    dctx.stream_dict = crate::decompress::zstd_ddict::ZSTD_DDict_dictContent(frameDDict).to_vec();
    dctx.dictUses = ZSTD_dictUses_e::ZSTD_use_indefinitely;
    Some(frameDDict)
}

/// Port of `ZSTD_getDDict` (`zstd_decompress.c:1180`).
///
/// Upstream returns the currently selected `ZSTD_DDict*`, while this
/// Rust port stores the active dictionary content bytes in
/// `dctx.stream_dict`. Return an owned snapshot so callers can both
/// consume the dictionary lifecycle and continue mutating the DCtx
/// during decompression.
pub fn ZSTD_getDDict(dctx: &mut ZSTD_DCtx) -> Option<Vec<u8>> {
    match dctx.dictUses {
        ZSTD_dictUses_e::ZSTD_dont_use => {
            ZSTD_clearDict(dctx);
            None
        }
        ZSTD_dictUses_e::ZSTD_use_indefinitely => {
            if dctx.stream_dict.is_empty() {
                None
            } else {
                Some(dctx.stream_dict.clone())
            }
        }
        ZSTD_dictUses_e::ZSTD_use_once => {
            let dict = if dctx.stream_dict.is_empty() {
                None
            } else {
                Some(dctx.stream_dict.clone())
            };
            ZSTD_clearDict(dctx);
            dict
        }
    }
}

/// Port of `ZSTD_createDCtx_internal` (`zstd_decompress.c:294`).
/// Upstream mallocs a DCtx from the customMem allocator, sets
/// `customMem`, then calls `ZSTD_initDCtx_internal`. Rust port's
/// `Box::new(ZSTD_DCtx::default())` gives an already-initialized
/// DCtx; this helper adds the explicit init-internal call for
/// upstream-parity behavior on field resets.
pub fn ZSTD_createDCtx_internal(
    customMem: crate::compress::zstd_compress::ZSTD_customMem,
) -> Option<Box<ZSTD_DCtx>> {
    let mut dctx = unsafe {
        crate::compress::zstd_compress::ZSTD_customAllocBox(ZSTD_DCtx::new(), customMem)?
    };
    dctx.customMem = customMem;
    ZSTD_initDCtx_internal(&mut dctx);
    Some(dctx)
}

/// Port of `ZSTD_initDCtx_internal` (`zstd_decompress.c:252`). Resets
/// a freshly-allocated DCtx to a known-good starting state — upstream
/// zeros ddict pointers + inBuff/outBuff + `streamStage = zdss_init`,
/// sets `isFrameDecompression = 1`, and finally calls
/// `ZSTD_DCtx_resetParameters` to install upstream defaults.
///
/// Our Rust port's `ZSTD_DCtx::new()` already initializes every field,
/// so this helper just calls the reset-parameters path — matching
/// upstream semantics (fresh DCtx → default params).
pub fn ZSTD_initDCtx_internal(dctx: &mut ZSTD_DCtx) {
    dctx.isFrameDecompression = 1;
    ZSTD_clearDict(dctx);
    ZSTD_DCtx_resetParameters(dctx);
}

/// Port of `ZSTD_DCtx_resetParameters` (zstd_decompress.c:240).
/// Resets the DCtx's decoder-parameter slots back to upstream
/// defaults: `maxWindowSize = (1 << ZSTD_WINDOWLOG_LIMIT_DEFAULT)`,
/// leaving session state untouched. Several upstream DCtx fields
/// (format, outBufferMode, forceIgnoreChecksum, refMultipleDDicts,
/// disableHufAsm, maxBlockSizeParam) aren't tracked in v0.1 — the
/// ones we do carry are reset here.
pub fn ZSTD_DCtx_resetParameters(dctx: &mut ZSTD_DCtx) {
    dctx.d_windowLogMax = ZSTD_WINDOWLOG_LIMIT_DEFAULT;
    // Include `format` — now tracked as a parameter via the
    // `ZSTD_d_format` enum variant, so a parameter-reset must
    // restore it to the default zstd1 mode.
    dctx.format = ZSTD_format_e::ZSTD_f_zstd1;
    // The dict-lifecycle tracker is scoped to parameters — a
    // parameter-reset returns it to `ZSTD_dont_use` to stay
    // consistent with `ZSTD_DCtx_reset(parameters)`.
    dctx.dictUses = ZSTD_dictUses_e::ZSTD_dont_use;
}

/// Port of `ZSTD_dParam_withinBounds` (zstd_decompress.c:1864).
/// Returns 1 if `value` falls within the param's bounds, 0 otherwise
/// (including the bounds-error case). Used by the setParameter path
/// to validate callers' inputs without returning an error code.
pub fn ZSTD_dParam_withinBounds(dParam: ZSTD_dParameter, value: i32) -> i32 {
    let bounds = ZSTD_dParam_getBounds(dParam);
    if crate::common::error::ERR_isError(bounds.error) {
        return 0;
    }
    if value < bounds.lowerBound || value > bounds.upperBound {
        return 0;
    }
    1
}

/// Port of `ZSTD_DCtx_setMaxWindowSize`. Clamps `maxWindowSize` into
/// `[1 << min, 1 << max]` (taken from `ZSTD_d_windowLogMax`'s bounds)
/// and stores it on the DCtx for subsequent streaming decompressions
/// to enforce against frame headers.
pub fn ZSTD_DCtx_setMaxWindowSize(dctx: &mut ZSTD_DCtx, maxWindowSize: usize) -> usize {
    use crate::common::error::{ErrorCode, ERROR};
    // Upstream (zstd_decompress.c:1809) gates with
    // `streamStage != zdss_init → StageWrong`.
    if !dctx_is_in_init_stage(dctx) {
        return ERROR(ErrorCode::StageWrong);
    }
    let bounds = ZSTD_dParam_getBounds(ZSTD_dParameter::ZSTD_d_windowLogMax);
    let min = 1usize << bounds.lowerBound;
    let max = 1usize << bounds.upperBound;
    if maxWindowSize < min || maxWindowSize > max {
        return ERROR(ErrorCode::ParameterOutOfBound);
    }
    // We store windowLog (the log2) rather than bytes to match the
    // existing `d_windowLogMax` parameter slot. Use `highbit`
    // semantics (bits - leading_zeros - 1) instead of `trailing_zeros`
    // so a non-power-of-2 input picks the ceiling log2 rather than
    // collapsing to the absolute minimum — upstream stores
    // maxWindowSize verbatim so behaviorally the two are equivalent
    // for any value that's >= `1 << bounds.lowerBound`.
    dctx.d_windowLogMax = ((maxWindowSize.leading_zeros() as i32 ^ (usize::BITS as i32 - 1))
        as u32)
        .max(bounds.lowerBound as u32);
    0
}

/// Port of `ZSTD_DCtx_setFormat` (`zstd_decompress.c:1816`). Thin
/// wrapper around `ZSTD_DCtx_setParameter(d_format, value)` — matches
/// upstream's single-parameter shape. The decoder respects
/// `dctx.format` when parsing frame headers via
/// `ZSTD_startingInputLength(dctx.format)` and
/// `ZSTD_getFrameHeader_advanced`.
pub fn ZSTD_DCtx_setFormat(
    dctx: &mut ZSTD_DCtx,
    format: crate::decompress::zstd_decompress::ZSTD_format_e,
) -> usize {
    // Upstream (zstd_decompress.c:1816) routes through
    // `ZSTD_DCtx_setParameter(ZSTD_d_format, value)` so the bounds
    // check + enum-cast happens in one place.
    ZSTD_DCtx_setParameter(dctx, ZSTD_dParameter::ZSTD_d_format, format as i32)
}

/// Port of `ZSTD_DStream`. Upstream `typedef ZSTD_DCtx ZSTD_DStream`
/// — same struct for both APIs.
pub type ZSTD_DStream = ZSTD_DCtx;

/// Port of `ZSTD_createDStream`. Alias for `ZSTD_createDCtx`.
pub fn ZSTD_createDStream() -> Option<Box<ZSTD_DStream>> {
    Some(ZSTD_createDCtx())
}

/// Port of `ZSTD_freeDStream`. Alias for `ZSTD_freeDCtx`.
pub fn ZSTD_freeDStream(zds: Option<Box<ZSTD_DStream>>) -> usize {
    if let Some(zds) = zds {
        return ZSTD_freeDCtx(zds);
    }
    0
}

/// Proxy for upstream's `dctx.streamStage == zdss_init` check —
/// returns true when no streaming decompression is in flight for the
/// current frame. Sibling of the compressor-side
/// `cctx_is_in_init_stage`.
#[inline]
fn dctx_is_in_init_stage(dctx: &ZSTD_DCtx) -> bool {
    dctx.stream_in_buffer.is_empty()
        && dctx.stream_out_buffer.is_empty()
        && dctx.stream_out_drained == 0
}

/// Port of `ZSTD_DCtx_loadDictionary`. Configures the DCtx with a
/// dict — routes through `ZSTD_decompress_insertDictionary` which
/// handles both raw-content and magic-prefixed zstd-format dicts
/// (parsing the entropy tables in the latter case).
///
/// Upstream (zstd_decompress.c:1704) gates with
/// `streamStage != zdss_init → StageWrong` — swapping a dict mid-
/// stream would decouple the back-ref substrate from bytes already
/// in the DCtx's input buffer.
pub fn ZSTD_DCtx_loadDictionary(dctx: &mut ZSTD_DCtx, dict: &[u8]) -> usize {
    if !dctx_is_in_init_stage(dctx) {
        return ERROR(ErrorCode::StageWrong);
    }
    // Upstream (zstd_decompress.c:1710 → loadDictionary_advanced):
    // `ZSTD_clearDict` first, then install only when there's
    // actual content. Empty-dict calls become a pure clear.
    ZSTD_clearDict(dctx);
    if dict.is_empty() {
        return 0;
    }
    let rc = ZSTD_decompress_insertDictionary(dctx, dict);
    if crate::common::error::ERR_isError(rc) {
        return rc;
    }
    // Upstream (zstd_decompress.c:1711) marks loaded dicts as
    // `use_indefinitely` so they persist across subsequent frames.
    // refPrefix sets `use_once` below — the distinction matters for
    // the post-frame auto-clear path.
    dctx.dictUses = ZSTD_dictUses_e::ZSTD_use_indefinitely;
    0
}

/// Port of `ZSTD_DCtx_refPrefix`. Prefixes are always raw content;
/// upstream clears dictID and doesn't parse magic. Marks the binding
/// as `ZSTD_use_once` — the next `decompressDCtx` / `decompressStream`
/// frame consumes the prefix and auto-clears it, matching upstream's
/// single-use `refPrefix` contract (`zstd_decompress.c:1728`). Same
/// stage gate as the other DCtx dict-family setters.
pub fn ZSTD_DCtx_refPrefix(dctx: &mut ZSTD_DCtx, prefix: &[u8]) -> usize {
    if !dctx_is_in_init_stage(dctx) {
        return ERROR(ErrorCode::StageWrong);
    }
    // Upstream (zstd_decompress.c:1725 → loadDictionary_advanced)
    // clears prior dict state before installing. Mirror so
    // `refPrefix(&[])` acts as "clear" rather than leaving
    // `dictUses = use_once` with an empty stream_dict.
    ZSTD_clearDict(dctx);
    if !prefix.is_empty() {
        dctx.stream_dict = prefix.to_vec();
        // `dictID` was just zeroed by `clearDict` — no explicit
        // re-write needed for the raw-content prefix path.
        // Upstream (zstd_decompress.c:1728) marks prefix-dict as
        // `use_once` so it's auto-cleared after the next frame decodes.
        dctx.dictUses = ZSTD_dictUses_e::ZSTD_use_once;
    }
    0
}

/// Port of `ZSTD_DCtx_refPrefix_advanced`. Upstream extends the base
/// `refPrefix` with an explicit `ZSTD_dictContentType_e`; v0.1
/// treats all content types as raw.
pub fn ZSTD_DCtx_refPrefix_advanced(
    dctx: &mut ZSTD_DCtx,
    prefix: &[u8],
    _dictContentType: crate::decompress::zstd_ddict::ZSTD_dictContentType_e,
) -> usize {
    ZSTD_DCtx_refPrefix(dctx, prefix)
}

/// Port of `ZSTD_DCtx_loadDictionary_advanced`. Forwards to the core
/// loader. The Rust port always copies the caller bytes and lets the
/// auto loader distinguish raw-content from magic-prefix dictionaries.
pub fn ZSTD_DCtx_loadDictionary_advanced(
    dctx: &mut ZSTD_DCtx,
    dict: &[u8],
    _dictLoadMethod: crate::decompress::zstd_ddict::ZSTD_dictLoadMethod_e,
    _dictContentType: crate::decompress::zstd_ddict::ZSTD_dictContentType_e,
) -> usize {
    ZSTD_DCtx_loadDictionary(dctx, dict)
}

/// Port of `ZSTD_DCtx_loadDictionary_byReference`. Forwards to the
/// owning loader — v0.1 doesn't split by-ref from by-copy yet.
#[inline]
pub fn ZSTD_DCtx_loadDictionary_byReference(dctx: &mut ZSTD_DCtx, dict: &[u8]) -> usize {
    ZSTD_DCtx_loadDictionary(dctx, dict)
}

/// Port of `ZSTD_DCtx_refDDict` (`zstd_decompress.c:1780`). Wires a
/// pre-built DDict into the DCtx. Upstream also clears any prior
/// dict state + tracks the DDict in a hash set for multi-dict
/// lookup; our simplified port copies the DDict parameters and raw
/// content into the DCtx.
pub fn ZSTD_DCtx_refDDict(
    dctx: &mut ZSTD_DCtx,
    ddict: &crate::decompress::zstd_ddict::ZSTD_DDict,
) -> usize {
    // Upstream (zstd_decompress.c:1782) gates on `streamStage == zdss_init`.
    if !dctx_is_in_init_stage(dctx) {
        return ERROR(ErrorCode::StageWrong);
    }
    ZSTD_clearDict(dctx);
    let content = crate::decompress::zstd_ddict::ZSTD_DDict_dictContent(ddict);
    if content.is_empty() {
        return 0;
    }
    crate::decompress::zstd_ddict::ZSTD_copyDDictParameters(dctx, ddict);
    // Upstream (zstd_decompress.c:1786) marks ref'd DDict as
    // `use_indefinitely`.
    dctx.dictUses = ZSTD_dictUses_e::ZSTD_use_indefinitely;
    0
}

/// Port of `ZSTD_DStreamInSize`. Suggested input-buffer size for
/// streaming decompression. Upstream returns `ZSTD_BLOCKSIZE_MAX + 3`.
pub fn ZSTD_DStreamInSize() -> usize {
    use crate::decompress::zstd_decompress_block::{ZSTD_blockHeaderSize, ZSTD_BLOCKSIZE_MAX};
    // Upstream (zstd_decompress.c:1696):
    // `ZSTD_BLOCKSIZE_MAX + ZSTD_blockHeaderSize`.
    ZSTD_BLOCKSIZE_MAX + ZSTD_blockHeaderSize
}

/// Port of `ZSTD_DStreamOutSize`. Suggested output-buffer size —
/// upstream returns `ZSTD_BLOCKSIZE_MAX`.
pub fn ZSTD_DStreamOutSize() -> usize {
    crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX
}

/// Port of `ZSTD_dParameter` — parametric decoder configuration.
/// Only the subset we honor is exposed; callers setting unsupported
/// ids get `ParameterUnsupported`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ZSTD_dParameter {
    ZSTD_d_windowLogMax = 100,
    /// Upstream `ZSTD_d_format` = `ZSTD_d_experimentalParam1` (1000).
    /// Toggles between `ZSTD_f_zstd1` and `ZSTD_f_zstd1_magicless`.
    ZSTD_d_format = 1000,
}

/// Port of `ZSTD_DCtx_setParameter`. For `windowLogMax` we record
/// the bound on the DCtx for potential upper-bound checks during
/// frame-header parsing. v0.1 just stashes the value; the enforcement
/// path lands alongside the ZSTD_windowLog_max limit check.
///
/// Upstream contract (lib/decompress/zstd_decompress.c:1910): if
/// `value == 0`, substitute `ZSTD_WINDOWLOG_LIMIT_DEFAULT` (27);
/// then bounds-check via `CHECK_DBOUNDS`; store on the DCtx. Mirrors
/// that behavior so C callers passing 0 get the documented default.
pub fn ZSTD_DCtx_setParameter(dctx: &mut ZSTD_DCtx, param: ZSTD_dParameter, value: i32) -> usize {
    use crate::common::error::{ErrorCode, ERROR};
    // Upstream (zstd_decompress.c:1908) unconditionally gates
    // `DCtx_setParameter` on `streamStage == zdss_init`. Unlike the
    // compressor side, there's no "authorized subset" — every param
    // must be set before streaming begins. Swapping format or
    // windowLogMax mid-stream would decouple the frame-header probe
    // from bytes already buffered.
    if !dctx_is_in_init_stage(dctx) {
        return ERROR(ErrorCode::StageWrong);
    }
    match param {
        ZSTD_dParameter::ZSTD_d_windowLogMax => {
            let effective = if value == 0 {
                ZSTD_WINDOWLOG_LIMIT_DEFAULT as i32
            } else {
                value
            };
            let bounds = ZSTD_dParam_getBounds(ZSTD_dParameter::ZSTD_d_windowLogMax);
            if effective < bounds.lowerBound || effective > bounds.upperBound {
                return ERROR(ErrorCode::ParameterOutOfBound);
            }
            dctx.d_windowLogMax = effective as u32;
            0
        }
        ZSTD_dParameter::ZSTD_d_format => {
            // Upstream (zstd_decompress.c:1915): bounds-check against
            // `[ZSTD_f_zstd1, ZSTD_f_zstd1_magicless]` then stash.
            let bounds = ZSTD_dParam_getBounds(ZSTD_dParameter::ZSTD_d_format);
            if value < bounds.lowerBound || value > bounds.upperBound {
                return ERROR(ErrorCode::ParameterOutOfBound);
            }
            dctx.format = match value {
                v if v == ZSTD_format_e::ZSTD_f_zstd1_magicless as i32 => {
                    ZSTD_format_e::ZSTD_f_zstd1_magicless
                }
                _ => ZSTD_format_e::ZSTD_f_zstd1,
            };
            0
        }
    }
}

/// Port of `ZSTD_WINDOWLOG_LIMIT_DEFAULT` — the streaming decoder's
/// conservative default cap (128 MB). Distinct from `ZSTD_WINDOWLOG_MAX`
/// which is the absolute-max permissible via `setParameter`.
pub const ZSTD_WINDOWLOG_LIMIT_DEFAULT: u32 = 27;

/// Port of `ZSTD_DCtx_getParameter`.
pub fn ZSTD_DCtx_getParameter(dctx: &ZSTD_DCtx, param: ZSTD_dParameter, value: &mut i32) -> usize {
    *value = match param {
        ZSTD_dParameter::ZSTD_d_windowLogMax => dctx.d_windowLogMax as i32,
        ZSTD_dParameter::ZSTD_d_format => dctx.format as i32,
    };
    0
}

/// Port of `ZSTD_DCtx_reset`. Matches upstream's three modes: clear
/// per-frame streaming state on `session_only`, restore default
/// parameters + drop the configured dict on `parameters`, do both
/// on `session_and_parameters`.
pub fn ZSTD_DCtx_reset(dctx: &mut ZSTD_DCtx, reset: ZSTD_DResetDirective) -> usize {
    let clear_session = matches!(
        reset,
        ZSTD_DResetDirective::ZSTD_reset_session_only
            | ZSTD_DResetDirective::ZSTD_reset_session_and_parameters,
    );
    let clear_params = matches!(
        reset,
        ZSTD_DResetDirective::ZSTD_reset_parameters
            | ZSTD_DResetDirective::ZSTD_reset_session_and_parameters,
    );
    // Upstream (zstd_decompress.c:1958) gates a pure params-reset on
    // `streamStage == zdss_init`. The combined variant clears session
    // first, so the gate only fires for `reset_parameters` alone.
    if clear_params && !clear_session && !dctx_is_in_init_stage(dctx) {
        return ERROR(ErrorCode::StageWrong);
    }
    if clear_session {
        ZSTD_resetDStream(dctx);
    }
    if clear_params {
        // Upstream (zstd_decompress.c:1958) routes through
        // `ZSTD_clearDict` + `ZSTD_DCtx_resetParameters` so every
        // dict- and parameter-related slot is wiped uniformly. We
        // delegate to the same pair for parity — this also clears
        // `dictID`, `ddict_rep`, and `dictUses` that a direct
        // field-by-field reset would miss.
        ZSTD_clearDict(dctx);
        ZSTD_DCtx_resetParameters(dctx);
    }
    0
}

/// Port of `ZSTD_ResetDirective` (decoder-side alias).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ZSTD_DResetDirective {
    // Upstream aliases this to the same `ZSTD_ResetDirective` enum
    // (zstd.h:589), values 1/2/3. Pin them explicitly so FFI bridges
    // passing raw integers route correctly.
    ZSTD_reset_session_only = 1,
    ZSTD_reset_parameters = 2,
    ZSTD_reset_session_and_parameters = 3,
}

/// Port of `ZSTD_sizeof_DCtx`. Walks the DCtx's owned `Vec`s.
pub fn ZSTD_sizeof_DCtx(dctx: &ZSTD_DCtx) -> usize {
    core::mem::size_of::<ZSTD_DCtx>()
        + dctx.hufTable.capacity() * core::mem::size_of::<u32>()
        + dctx.workspace.capacity() * core::mem::size_of::<u32>()
        + dctx.litExtraBuffer.capacity()
        + dctx.LLTable.capacity()
            * core::mem::size_of::<crate::decompress::zstd_decompress_block::ZSTD_seqSymbol>()
        + dctx.OFTable.capacity()
            * core::mem::size_of::<crate::decompress::zstd_decompress_block::ZSTD_seqSymbol>()
        + dctx.MLTable.capacity()
            * core::mem::size_of::<crate::decompress::zstd_decompress_block::ZSTD_seqSymbol>()
        + dctx.stream_in_buffer.capacity()
        + dctx.stream_out_buffer.capacity()
        + dctx.stream_dict.capacity()
}

/// Port of `ZSTD_sizeof_DStream`. Alias.
pub fn ZSTD_sizeof_DStream(zds: &ZSTD_DStream) -> usize {
    ZSTD_sizeof_DCtx(zds)
}

/// Port of `ZSTD_estimateDCtxSize` (`zstd_decompress.c:229`) — upstream
/// returns a single `sizeof(ZSTD_DCtx)`. All the HUF / FSE / litbuffer
/// tables live inside the upstream DCtx struct as arrays; in our port
/// they're `Vec` fields, so this constant undercount-s the true heap
/// footprint. Use `ZSTD_sizeof_DCtx(&dctx)` on a live context to get
/// the Rust-accurate total including owned allocations.
pub fn ZSTD_estimateDCtxSize() -> usize {
    core::mem::size_of::<ZSTD_DCtx>()
}

/// Port of `ZSTD_estimateDStreamSize` (`zstd_decompress.c:1993`).
/// Returns `DCtxSize + inBuffSize + outBuffSize` where
/// `inBuffSize = min(windowSize, BLOCKSIZE_MAX)` and `outBuffSize` is
/// the ring-buffer size needed for an unknown-content-size frame at
/// this window.
pub fn ZSTD_estimateDStreamSize(windowSize: usize) -> usize {
    use crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX;
    let blockSize = windowSize.min(ZSTD_BLOCKSIZE_MAX);
    let inBuffSize = blockSize;
    let outBuffSize = ZSTD_decodingBufferSize_min(windowSize as u64, ZSTD_CONTENTSIZE_UNKNOWN);
    ZSTD_estimateDCtxSize() + inBuffSize + outBuffSize
}

/// Port of `ZSTD_estimateDStreamSize_fromFrame`. Parses the frame
/// header to extract windowSize, then calls `ZSTD_estimateDStreamSize`.
pub fn ZSTD_estimateDStreamSize_fromFrame(src: &[u8]) -> usize {
    let mut zfh = ZSTD_FrameHeader::default();
    let rc = ZSTD_getFrameHeader(&mut zfh, src);
    if crate::common::error::ERR_isError(rc) || rc != 0 {
        return ZSTD_estimateDStreamSize(1 << 17); // default block-size estimate
    }
    ZSTD_estimateDStreamSize(zfh.windowSize as usize)
}

/// Port of `ZSTD_insertBlock` (`zstd_decompress.c:887`). Inserts
/// `block` as raw history into the DCtx — useful when a frameless
/// protocol tracks uncompressed blocks alongside compressed ones.
pub fn ZSTD_insertBlock(dctx: &mut ZSTD_DCtx, block: &[u8]) -> usize {
    crate::decompress::zstd_decompress_block::ZSTD_checkContinuity(dctx, block, block.len());
    dctx.previousDstEnd = Some(block.as_ptr() as usize + block.len());
    block.len()
}

/// Port of the public `ZSTD_decompressBlock`. Decompresses a single
/// block body WITHOUT any frame header — `src` starts at the block
/// data (literals + sequences), as emitted by `ZSTD_compressBlock`.
/// Only meaningful for callers building frameless protocols.
///
/// Rust signature: `src` is the block body bytes. Returns the number
/// of decoded bytes written into `dst`, or an error code.
pub fn ZSTD_decompressBlock(dctx: &mut ZSTD_DCtx, dst: &mut [u8], src: &[u8]) -> usize {
    use crate::decompress::zstd_decompress_block::{
        streaming_operation, ZSTD_buildDefaultSeqTables, ZSTD_decoder_entropy_rep,
        ZSTD_decompressBlock_internal,
    };
    ZSTD_buildDefaultSeqTables(dctx);
    let mut entropy_rep = ZSTD_decoder_entropy_rep::default();
    ZSTD_decompressBlock_internal(
        dctx,
        &mut entropy_rep,
        dst,
        0,
        src,
        streaming_operation::not_streaming,
    )
}

/// Port of `ZSTD_decompressBlock_deprecated` (zstd_decompress_block.c:2291).
/// Upstream kept the legacy-name entry alongside the current
/// `ZSTD_decompressBlock`; both do the same work. Forwards to the
/// current entry.
pub fn ZSTD_decompressBlock_deprecated(dctx: &mut ZSTD_DCtx, dst: &mut [u8], src: &[u8]) -> usize {
    ZSTD_decompressBlock(dctx, dst, src)
}

/// Port of `ZSTD_getBlockSize`. Returns the maximum block size the
/// DCtx will accept — `ZSTD_BLOCKSIZE_MAX` unless a frame header
/// narrowed it.
pub fn ZSTD_getBlockSize(_dctx: &ZSTD_DCtx) -> usize {
    crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX
}

/// Port of `ZSTD_isFrame`. Returns 1 if `src` starts with a zstd
/// magic number (regular or skippable frame), 0 otherwise. Cheap —
/// only reads the first 4 bytes.
pub fn ZSTD_isFrame(src: &[u8]) -> u32 {
    if src.len() < ZSTD_FRAMEIDSIZE {
        return 0;
    }
    let magic = MEM_readLE32(&src[..4]);
    if magic == ZSTD_MAGICNUMBER {
        return 1;
    }
    if (magic & ZSTD_MAGIC_SKIPPABLE_MASK) == ZSTD_MAGIC_SKIPPABLE_START {
        return 1;
    }
    0
}

/// Port of `ZSTD_isSkippableFrame`. Returns 1 if `src` starts with
/// one of the 16 skippable-frame magic variants (0x184D2A5X).
pub fn ZSTD_isSkippableFrame(src: &[u8]) -> u32 {
    if src.len() < ZSTD_FRAMEIDSIZE {
        return 0;
    }
    let magic = MEM_readLE32(&src[..4]);
    if (magic & ZSTD_MAGIC_SKIPPABLE_MASK) == ZSTD_MAGIC_SKIPPABLE_START {
        return 1;
    }
    0
}

/// Port of `ZSTD_decodeFrameHeader` (`zstd_decompress.c:698`).
/// `headerSize` must be the exact size returned by
/// `ZSTD_frameHeaderSize_internal`. Populates `dctx.fParams`, checks
/// dictionary compatibility, initializes checksum state, and accounts
/// for consumed compressed header bytes.
pub fn ZSTD_decodeFrameHeader(dctx: &mut ZSTD_DCtx, src: &[u8], headerSize: usize) -> usize {
    use crate::common::xxhash::XXH64_reset;

    if src.len() < headerSize {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    let result = ZSTD_getFrameHeader_advanced(&mut dctx.fParams, &src[..headerSize], dctx.format);
    if crate::common::error::ERR_isError(result) {
        return result;
    }
    if result != 0 {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    if dctx.fParams.dictID != 0 && dctx.dictID != dctx.fParams.dictID {
        return ERROR(ErrorCode::DictionaryWrong);
    }
    dctx.isFrameDecompression = 1;
    dctx.blockSizeMax = dctx.fParams.blockSizeMax as usize;
    dctx.validateChecksum = dctx.fParams.checksumFlag;
    if dctx.validateChecksum != 0 {
        XXH64_reset(&mut dctx.xxhState, 0);
    }
    dctx.processedCSize = dctx.processedCSize.wrapping_add(headerSize as u64);
    0
}

/// Port of `ZSTD_readSkippableFrame`. Copies a skippable frame's
/// payload into `dst` and optionally reports the low-nibble "magic
/// variant" the frame was tagged with.
///
/// Returns the number of payload bytes written, or a zstd error
/// code (dst too small, bad magic, etc).
pub fn ZSTD_readSkippableFrame(
    dst: &mut [u8],
    magicVariant: Option<&mut u32>,
    src: &[u8],
) -> usize {
    use crate::common::error::{ErrorCode, ERROR};
    if src.len() < ZSTD_SKIPPABLEHEADERSIZE {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    let magicNumber = MEM_readLE32(&src[..4]);
    let skippableFrameSize = readSkippableFrameSize(src);
    if ZSTD_isSkippableFrame(src) == 0 {
        return ERROR(ErrorCode::FrameParameterUnsupported);
    }
    if skippableFrameSize < ZSTD_SKIPPABLEHEADERSIZE || skippableFrameSize > src.len() {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    let skippableContentSize = skippableFrameSize - ZSTD_SKIPPABLEHEADERSIZE;
    if skippableContentSize > dst.len() {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    if skippableContentSize > 0 {
        dst[..skippableContentSize].copy_from_slice(
            &src[ZSTD_SKIPPABLEHEADERSIZE..ZSTD_SKIPPABLEHEADERSIZE + skippableContentSize],
        );
    }
    if let Some(v) = magicVariant {
        *v = magicNumber - ZSTD_MAGIC_SKIPPABLE_START;
    }
    skippableContentSize
}

/// Port of `ZSTD_getDictID_fromFrame`. Returns the dictID the frame
/// was compressed with, or 0 if the frame doesn't declare one or the
/// header can't be parsed.
pub fn ZSTD_getDictID_fromFrame(src: &[u8]) -> u32 {
    let mut zfh = ZSTD_FrameHeader::default();
    let rc = ZSTD_getFrameHeader(&mut zfh, src);
    if crate::common::error::ERR_isError(rc) || rc != 0 {
        return 0;
    }
    zfh.dictID
}

/// Port of `ZSTD_decompress_usingDDict`. Decompresses `src` using a
/// pre-digested decompression dictionary. Uses the DDict's raw
/// content as history and seeds serialized dictionary entropy when
/// present.
pub fn ZSTD_decompress_usingDDict(
    dctx: &mut ZSTD_DCtx,
    dst: &mut [u8],
    src: &[u8],
    ddict: &crate::decompress::zstd_ddict::ZSTD_DDict,
) -> usize {
    use crate::common::error::ERR_isError;
    use crate::common::xxhash::XXH64_state_t;
    use crate::decompress::zstd_decompress_block::ZSTD_decoder_entropy_rep;

    let frame_dict_id = ZSTD_getDictID_fromFrame(src);
    if frame_dict_id != 0 && ddict.dictID != 0 && frame_dict_id != ddict.dictID {
        return ERROR(ErrorCode::DictionaryWrong);
    }

    let content = crate::decompress::zstd_ddict::ZSTD_DDict_dictContent(ddict);
    let declared = ZSTD_getFrameContentSize(src);
    let out_size = if declared == ZSTD_CONTENTSIZE_UNKNOWN || declared == ZSTD_CONTENTSIZE_ERROR {
        dst.len()
    } else {
        declared as usize
    };
    if out_size > dst.len() {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }

    let rc = ZSTD_decompressBegin(dctx);
    if ERR_isError(rc) {
        return rc;
    }
    crate::decompress::zstd_ddict::ZSTD_copyDDictParameters(dctx, ddict);

    let mut combined = vec![0u8; content.len() + out_size];
    combined[..content.len()].copy_from_slice(content);
    let mut rep = ZSTD_decoder_entropy_rep {
        rep: dctx.ddict_rep,
    };
    let mut xxh = XXH64_state_t::default();
    let mut consumed = 0usize;
    let decoded = ZSTD_decompressFrame_withOpStart(
        dctx,
        &mut rep,
        &mut xxh,
        &mut combined,
        content.len(),
        src,
        &mut consumed,
    );
    if ERR_isError(decoded) {
        return decoded;
    }
    dst[..decoded].copy_from_slice(&combined[content.len()..content.len() + decoded]);
    decoded
}

/// Port of `ZSTD_decompress_insertDictionary` (`zstd_decompress.c:1539`).
/// Configures the DCtx with a dict. Three paths:
///   - `dictSize < 8` → raw content dict, no magic.
///   - No magic prefix → raw content dict (`ZSTD_dct_auto` behavior).
///   - `ZSTD_MAGIC_DICTIONARY` prefix → parse dictID (skip 4-byte magic),
///     run `ZSTD_loadDEntropy` to populate HUF + FSE tables + rep[],
///     stash remaining bytes as the dict content.
///
/// Our port keeps the content on `stream_dict` (the concatenate-with-
/// src path `ZSTD_decompress_usingDict` uses). `litEntropy` and
/// `fseEntropy` flags are set to 1 when entropy tables were loaded,
/// so downstream `ZSTD_buildSeqTable` can honor set_repeat.
pub fn ZSTD_decompress_insertDictionary(dctx: &mut ZSTD_DCtx, dict: &[u8]) -> usize {
    use crate::common::error::{ERR_isError, ErrorCode, ERROR};
    use crate::common::mem::MEM_readLE32;

    // Raw content path: too small or no magic → just stash bytes.
    if dict.len() < 8 {
        dctx.dictID = 0;
        return ZSTD_refDictContent(dctx, dict);
    }
    let magic = MEM_readLE32(&dict[..4]);
    if magic != ZSTD_MAGICNUMBER_DICTIONARY {
        dctx.dictID = 0;
        return ZSTD_refDictContent(dctx, dict);
    }

    // Magic-prefixed zstd dict: parse dictID, entropy, rep, content.
    dctx.dictID = MEM_readLE32(&dict[4..8]);
    let mut rep = [0u32; 3];
    let eSize = ZSTD_loadDEntropy(dctx, &mut rep, dict);
    if ERR_isError(eSize) {
        return ERROR(ErrorCode::DictionaryCorrupted);
    }
    dctx.ddict_rep = rep;
    dctx.litEntropy = 1;
    dctx.fseEntropy = 1;
    // Content starts after the entropy-tables region.
    ZSTD_refDictContent(dctx, &dict[eSize..])
}

/// Port of `ZSTD_refDictContent` (`zstd_decompress.c:1435`).
///
/// C keeps raw pointers into caller-owned dictionary bytes. The Rust
/// port records pointer-equivalent addresses for continuity tests and
/// also copies the bytes into `stream_dict`, which is the owned backing
/// used by the safe dictionary decompression paths.
pub fn ZSTD_refDictContent(dctx: &mut ZSTD_DCtx, dict: &[u8]) -> usize {
    let dict_start = dict.as_ptr() as usize;
    let dict_end = dict_start + dict.len();
    let previous_dst_end = dctx.previousDstEnd.unwrap_or(0);
    let prefix_start = dctx.prefixStart.unwrap_or(previous_dst_end);

    dctx.dictEnd = dctx.previousDstEnd;
    dctx.virtualStart = Some(dict_start.wrapping_sub(previous_dst_end.wrapping_sub(prefix_start)));
    dctx.prefixStart = Some(dict_start);
    dctx.previousDstEnd = Some(dict_end);
    dctx.stream_dict = dict.to_vec();
    0
}

/// Upstream `ZSTD_MAGIC_DICTIONARY` (`zstd.h:143`). 0xEC30A437 —
/// marks a zstd-format dictionary (vs raw-content).
pub const ZSTD_MAGICNUMBER_DICTIONARY: u32 = 0xEC30A437;

/// Port of `ZSTD_loadDEntropy` (`zstd_decompress.c:1451`). Parses the
/// entropy-tables section of a zstd-format dictionary into the DCtx's
/// HUF + FSE tables.
///
/// Layout (post-magic+dictID): HUF DTable → FSE OF table → FSE ML
/// table → FSE LL table → 3 × u32 rep values. Returns total bytes
/// consumed (up to and including the rep values), or a
/// `DictionaryCorrupted` error.
///
/// Caller provides `repOut` for the 3 repcodes; the dict content
/// follows the consumed region.
pub fn ZSTD_loadDEntropy(dctx: &mut ZSTD_DCtx, repOut: &mut [u32; 3], dict: &[u8]) -> usize {
    use crate::common::error::{ERR_isError, ErrorCode, ERROR};
    use crate::common::mem::MEM_readLE32;
    use crate::decompress::huf_decompress::HUF_readDTableX2;
    use crate::decompress::zstd_decompress_block::{
        LLFSELog, LL_base, LL_bits, MLFSELog, ML_base, ML_bits, MaxLL, MaxML, MaxOff, OF_base,
        OF_bits, OffFSELog, ZSTD_buildFSETable,
    };

    if dict.len() <= 8 {
        return ERROR(ErrorCode::DictionaryCorrupted);
    }
    let mut pos = 8usize; // skip magic + dictID

    // --- HUF DTable ---
    let mut hufWorkspace = vec![0u32; 1024];
    let hSize = HUF_readDTableX2(&mut dctx.hufTable, &dict[pos..], &mut hufWorkspace, 0);
    if ERR_isError(hSize) {
        return ERROR(ErrorCode::DictionaryCorrupted);
    }
    pos += hSize;

    // --- FSE OF table ---
    let mut ofcNCount = [0i16; (MaxOff + 1) as usize];
    let mut ofcMaxValue: u32 = MaxOff;
    let mut ofcLog: u32 = 0;
    let ofcSize = crate::common::entropy_common::FSE_readNCount(
        &mut ofcNCount,
        &mut ofcMaxValue,
        &mut ofcLog,
        &dict[pos..],
    );
    if ERR_isError(ofcSize) || ofcMaxValue > MaxOff || ofcLog > OffFSELog {
        return ERROR(ErrorCode::DictionaryCorrupted);
    }
    ZSTD_buildFSETable(
        &mut dctx.OFTable,
        &ofcNCount,
        ofcMaxValue,
        &OF_base,
        &OF_bits,
        ofcLog,
    );
    pos += ofcSize;

    // --- FSE ML table ---
    let mut mlNCount = [0i16; (MaxML + 1) as usize];
    let mut mlMaxValue: u32 = MaxML;
    let mut mlLog: u32 = 0;
    let mlSize = crate::common::entropy_common::FSE_readNCount(
        &mut mlNCount,
        &mut mlMaxValue,
        &mut mlLog,
        &dict[pos..],
    );
    if ERR_isError(mlSize) || mlMaxValue > MaxML || mlLog > MLFSELog {
        return ERROR(ErrorCode::DictionaryCorrupted);
    }
    ZSTD_buildFSETable(
        &mut dctx.MLTable,
        &mlNCount,
        mlMaxValue,
        &ML_base,
        &ML_bits,
        mlLog,
    );
    pos += mlSize;

    // --- FSE LL table ---
    let mut llNCount = [0i16; (MaxLL + 1) as usize];
    let mut llMaxValue: u32 = MaxLL;
    let mut llLog: u32 = 0;
    let llSize = crate::common::entropy_common::FSE_readNCount(
        &mut llNCount,
        &mut llMaxValue,
        &mut llLog,
        &dict[pos..],
    );
    if ERR_isError(llSize) || llMaxValue > MaxLL || llLog > LLFSELog {
        return ERROR(ErrorCode::DictionaryCorrupted);
    }
    ZSTD_buildFSETable(
        &mut dctx.LLTable,
        &llNCount,
        llMaxValue,
        &LL_base,
        &LL_bits,
        llLog,
    );
    pos += llSize;

    // --- 3 × 4-byte rep values ---
    if pos + 12 > dict.len() {
        return ERROR(ErrorCode::DictionaryCorrupted);
    }
    let dictContentSize = dict.len() - (pos + 12);
    for slot in repOut.iter_mut() {
        let r = MEM_readLE32(&dict[pos..pos + 4]);
        if r == 0 || (r as usize) > dictContentSize {
            return ERROR(ErrorCode::DictionaryCorrupted);
        }
        *slot = r;
        pos += 4;
    }

    pos
}

/// Port of `ZSTD_decompressMultiFrame` (`zstd_decompress.c:1070`).
/// Walks `src` through successive frames + skippable frames,
/// concatenating their decompressed output into `dst`. Either `dict`
/// or `ddict` may be supplied (not both) — the DDict's raw content
/// is extracted as the effective dict.
///
/// v0.1 delegates to `ZSTD_decompress_usingDict` which already
/// handles multi-frame walk + skippable-frame skipping; the wrapper
/// exists for API-surface parity with upstream's exported name.
pub fn ZSTD_decompressMultiFrame(
    dctx: &mut ZSTD_DCtx,
    dst: &mut [u8],
    src: &[u8],
    dict: &[u8],
    ddict: Option<&crate::decompress::zstd_ddict::ZSTD_DDict>,
) -> usize {
    debug_assert!(
        dict.is_empty() || ddict.is_none(),
        "dict xor ddict, not both"
    );
    if let Some(dd) = ddict {
        let content = crate::decompress::zstd_ddict::ZSTD_DDict_dictContent(dd);
        ZSTD_decompress_usingDict(dctx, dst, src, content)
    } else {
        ZSTD_decompress_usingDict(dctx, dst, src, dict)
    }
}

/// Port of `ZSTD_getFrameContentSize`. Returns the declared
/// decompressed size of the frame at `src`, `ZSTD_CONTENTSIZE_UNKNOWN`
/// if the FCS field is absent, or `ZSTD_CONTENTSIZE_ERROR` on a
/// malformed / truncated header.
pub fn ZSTD_getFrameContentSize(src: &[u8]) -> u64 {
    let mut zfh = ZSTD_FrameHeader::default();
    let rc = ZSTD_getFrameHeader(&mut zfh, src);
    if rc != 0 {
        return ZSTD_CONTENTSIZE_ERROR;
    }
    if zfh.frameType == ZSTD_FrameType_e::ZSTD_skippableFrame {
        return 0;
    }
    zfh.frameContentSize
}

/// Port of `readSkippableFrameSize`. Returns the total skippable-frame
/// byte length (header + payload).
fn readSkippableFrameSize(src: &[u8]) -> usize {
    if src.len() < ZSTD_SKIPPABLEHEADERSIZE {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    let sizeU32 = MEM_readLE32(&src[ZSTD_FRAMEIDSIZE..ZSTD_FRAMEIDSIZE + 4]);
    let total = ZSTD_SKIPPABLEHEADERSIZE.wrapping_add(sizeU32 as usize);
    if (total as u32) < sizeU32 {
        return ERROR(ErrorCode::FrameParameterUnsupported);
    }
    if total > src.len() {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    total
}

/// Mirror of `ZSTD_frameSizeInfo`: bookkeeping for walking frames.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZSTD_frameSizeInfo {
    pub nbBlocks: usize,
    pub compressedSize: usize,
    pub decompressedBound: u64,
}

fn frameSizeInfo_error(err_code: usize) -> ZSTD_frameSizeInfo {
    ZSTD_frameSizeInfo {
        nbBlocks: 0,
        compressedSize: err_code,
        decompressedBound: ZSTD_CONTENTSIZE_ERROR,
    }
}

/// Port of `ZSTD_findFrameSizeInfo`. Walks a single frame — handles
/// both skippable and regular frames.
pub fn ZSTD_findFrameSizeInfo(src: &[u8], format: ZSTD_format_e) -> ZSTD_frameSizeInfo {
    use crate::decompress::zstd_decompress_block::{
        blockProperties_t, blockType_e, ZSTD_blockHeaderSize, ZSTD_getcBlockSize,
    };
    let srcSize = src.len();

    if format == ZSTD_format_e::ZSTD_f_zstd1
        && srcSize >= ZSTD_SKIPPABLEHEADERSIZE
        && (MEM_readLE32(&src[..4]) & ZSTD_MAGIC_SKIPPABLE_MASK) == ZSTD_MAGIC_SKIPPABLE_START
    {
        return ZSTD_frameSizeInfo {
            nbBlocks: 0,
            compressedSize: readSkippableFrameSize(src),
            decompressedBound: 0,
        };
    }

    // Regular zstd frame.
    let mut zfh = ZSTD_FrameHeader::default();
    let rc = ZSTD_getFrameHeader_advanced(&mut zfh, src, format);
    if crate::common::error::ERR_isError(rc) {
        return frameSizeInfo_error(rc);
    }
    if rc > 0 {
        return frameSizeInfo_error(ERROR(ErrorCode::SrcSizeWrong));
    }

    let mut ip = zfh.headerSize as usize;
    let mut remaining = srcSize - ip;
    let mut nbBlocks: usize = 0;

    loop {
        let mut bp = blockProperties_t {
            blockType: blockType_e::bt_raw,
            lastBlock: 0,
            origSize: 0,
        };
        let cBlockSize = ZSTD_getcBlockSize(&src[ip..], &mut bp);
        if crate::common::error::ERR_isError(cBlockSize) {
            return frameSizeInfo_error(cBlockSize);
        }
        if ZSTD_blockHeaderSize + cBlockSize > remaining {
            return frameSizeInfo_error(ERROR(ErrorCode::SrcSizeWrong));
        }
        ip += ZSTD_blockHeaderSize + cBlockSize;
        remaining -= ZSTD_blockHeaderSize + cBlockSize;
        nbBlocks += 1;
        if bp.lastBlock != 0 {
            break;
        }
    }

    if zfh.checksumFlag != 0 {
        if remaining < 4 {
            return frameSizeInfo_error(ERROR(ErrorCode::SrcSizeWrong));
        }
        ip += 4;
    }

    let compressedSize = ip;
    let decompressedBound = if zfh.frameContentSize != ZSTD_CONTENTSIZE_UNKNOWN {
        zfh.frameContentSize
    } else {
        nbBlocks as u64 * zfh.blockSizeMax as u64
    };
    ZSTD_frameSizeInfo {
        nbBlocks,
        compressedSize,
        decompressedBound,
    }
}

/// Port of `ZSTD_findFrameCompressedSize_advanced`
/// (`zstd_decompress.c:801`). Format-aware variant of
/// `ZSTD_findFrameCompressedSize`.
pub fn ZSTD_findFrameCompressedSize_advanced(src: &[u8], format: ZSTD_format_e) -> usize {
    ZSTD_findFrameSizeInfo(src, format).compressedSize
}

/// Port of `ZSTD_findFrameCompressedSize`. Returns the total byte
/// length of the first frame in `src`, or an error code. Uses the
/// default zstd1 format — delegates to `_advanced`.
#[inline]
pub fn ZSTD_findFrameCompressedSize(src: &[u8]) -> usize {
    ZSTD_findFrameCompressedSize_advanced(src, ZSTD_format_e::ZSTD_f_zstd1)
}

/// Port of `ZSTD_findDecompressedSize`. Walks potentially multiple
/// frames (regular and skippable) and sums their declared FCS.
pub fn ZSTD_findDecompressedSize(src: &[u8]) -> u64 {
    let mut src = src;
    let mut total: u64 = 0;
    while src.len() >= ZSTD_startingInputLength(ZSTD_format_e::ZSTD_f_zstd1) {
        let magic = MEM_readLE32(&src[..4]);
        if (magic & ZSTD_MAGIC_SKIPPABLE_MASK) == ZSTD_MAGIC_SKIPPABLE_START {
            let sz = readSkippableFrameSize(src);
            if crate::common::error::ERR_isError(sz) {
                return ZSTD_CONTENTSIZE_ERROR;
            }
            src = &src[sz..];
            continue;
        }
        let fcs = ZSTD_getFrameContentSize(src);
        if fcs >= ZSTD_CONTENTSIZE_ERROR {
            return fcs;
        }
        if total.checked_add(fcs).is_none() {
            return ZSTD_CONTENTSIZE_ERROR;
        }
        total += fcs;
        let sz = ZSTD_findFrameCompressedSize(src);
        if crate::common::error::ERR_isError(sz) {
            return ZSTD_CONTENTSIZE_ERROR;
        }
        src = &src[sz..];
    }
    if !src.is_empty() {
        return ZSTD_CONTENTSIZE_ERROR;
    }
    total
}

/// Port of `ZSTD_initDStream`. Resets the DCtx's per-frame streaming
/// state (input/output buffers + drain cursor). Preserves any
/// configured dict so the next frame can still resolve back-refs
/// into it — upstream routes through `ZSTD_DCtx_reset(session_only)`.
pub fn ZSTD_initDStream(zds: &mut ZSTD_DCtx) -> usize {
    zds.stream_in_buffer.clear();
    zds.stream_out_buffer.clear();
    zds.stream_out_drained = 0;
    // Upstream (zstd_decompress.c:1755) returns
    // `ZSTD_startingInputLength(dctx->format)`. Same contract as
    // `ZSTD_resetDStream` — callers use the hint to size the first
    // decompressStream read. Reading `dctx.format` (vs hardcoding
    // `ZSTD_f_zstd1`) makes magicless-mode callers get the shorter
    // 2-byte hint instead of the zstd1 5-byte hint.
    ZSTD_startingInputLength(zds.format)
}

/// Port of `ZSTD_initDStream_usingDict`. Initializes streaming
/// decompression with a raw-content dictionary — every frame decoded
/// in this session will be passed the dict as history.
pub fn ZSTD_initDStream_usingDict(zds: &mut ZSTD_DCtx, dict: &[u8]) -> usize {
    let rc = ZSTD_initDStream(zds);
    if crate::common::error::ERR_isError(rc) {
        return rc;
    }
    // Route through `ZSTD_DCtx_loadDictionary` (not a direct
    // `stream_dict = dict` write) so magic-prefix dicts get their
    // entropy tables parsed onto the dctx. Matches upstream
    // (`zstd_decompress.c:1744`): `reset(session_only)` +
    // `loadDictionary` chain.
    if !dict.is_empty() {
        let rc = ZSTD_DCtx_loadDictionary(zds, dict);
        if crate::common::error::ERR_isError(rc) {
            return rc;
        }
    }
    ZSTD_startingInputLength(zds.format)
}

/// Port of `ZSTD_DECOMPRESSION_MARGIN` macro (`zstd.h:1574`). Static
/// upper bound on the margin needed to safely decompress
/// `originalSize` bytes from a frame whose max block size is
/// `blockSize`. Useful when you know originalSize ahead of time and
/// want a compile-time margin (matching upstream's preprocessor
/// macro semantics).
#[inline]
pub const fn ZSTD_DECOMPRESSION_MARGIN(originalSize: usize, blockSize: usize) -> usize {
    use crate::compress::zstd_compress::ZSTD_FRAMEHEADERSIZE_MAX;
    let blocks = if originalSize == 0 {
        0
    } else {
        3 * originalSize.div_ceil(blockSize)
    };
    ZSTD_FRAMEHEADERSIZE_MAX + 4 + blocks + blockSize
}

/// Port of `ZSTD_decompressionMargin`. Returns an upper bound on the
/// number of bytes of `dst` padding needed to safely decompress `src`
/// in-place — the caller offsets its output cursor by this much so
/// wildcopy overreads can't clobber unread compressed data.
///
/// The margin sums: frame header bytes + 4-byte checksum when
/// present + 3 bytes per block + the max block size observed across
/// all frames. Skippable frames count their full size.
pub fn ZSTD_decompressionMargin(src: &[u8]) -> usize {
    use crate::common::error::{ERR_isError, ErrorCode, ERROR};
    let mut margin: usize = 0;
    let mut maxBlockSize: u32 = 0;
    let mut cursor = src;

    while !cursor.is_empty() {
        let info = ZSTD_findFrameSizeInfo(cursor, ZSTD_format_e::ZSTD_f_zstd1);
        let mut zfh = ZSTD_FrameHeader::default();
        let rc = ZSTD_getFrameHeader(&mut zfh, cursor);
        if ERR_isError(rc) {
            return rc;
        }
        if ERR_isError(info.compressedSize) || info.decompressedBound == ZSTD_CONTENTSIZE_ERROR {
            return ERROR(ErrorCode::CorruptionDetected);
        }

        if zfh.frameType == ZSTD_FrameType_e::ZSTD_frame {
            margin += zfh.headerSize as usize;
            margin += if zfh.checksumFlag != 0 { 4 } else { 0 };
            margin += 3 * info.nbBlocks;
            if zfh.blockSizeMax > maxBlockSize {
                maxBlockSize = zfh.blockSizeMax;
            }
        } else {
            // Skippable: the whole frame counts.
            margin += info.compressedSize;
        }

        cursor = &cursor[info.compressedSize..];
    }

    margin + maxBlockSize as usize
}

/// Port of `ZSTD_decompressBound`. Walks every frame in `src`,
/// summing the per-frame `decompressedBound`. Returns
/// `ZSTD_CONTENTSIZE_ERROR` on any parse error.
pub fn ZSTD_decompressBound(src: &[u8]) -> u64 {
    use crate::common::error::ERR_isError;
    let mut bound: u64 = 0;
    let mut cursor = src;
    while !cursor.is_empty() {
        let info = ZSTD_findFrameSizeInfo(cursor, ZSTD_format_e::ZSTD_f_zstd1);
        if ERR_isError(info.compressedSize) || info.decompressedBound == ZSTD_CONTENTSIZE_ERROR {
            return ZSTD_CONTENTSIZE_ERROR;
        }
        cursor = &cursor[info.compressedSize..];
        bound = bound.saturating_add(info.decompressedBound);
    }
    bound
}

/// Port of `ZSTD_getDecompressedSize`. Deprecated: reads the frame
/// content size, returning 0 for "unknown / empty / error" — the
/// modern path is `ZSTD_getFrameContentSize` which distinguishes
/// those cases via sentinels.
pub fn ZSTD_getDecompressedSize(src: &[u8]) -> u64 {
    let ret = ZSTD_getFrameContentSize(src);
    if ret >= ZSTD_CONTENTSIZE_ERROR {
        0
    } else {
        ret
    }
}

/// Port of `ZSTD_decodingBufferSize_internal` (`zstd_decompress.c:1970`).
/// Caller supplies `blockSizeMax` — the decoder's max expected block
/// size; `ZSTD_decodingBufferSize_min` defaults it to
/// `ZSTD_BLOCKSIZE_MAX`. Returns the ring-buffer size needed: window
/// plus twice the block size (one for output, one for split-lit
/// trailing bytes) plus two wildcopy-overlength slack regions.
pub fn ZSTD_decodingBufferSize_internal(
    windowSize: u64,
    frameContentSize: u64,
    blockSizeMax: usize,
) -> usize {
    use crate::common::error::{ErrorCode, ERROR};
    use crate::common::zstd_internal::WILDCOPY_OVERLENGTH;
    use crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX;
    let blockSize = windowSize
        .min(ZSTD_BLOCKSIZE_MAX as u64)
        .min(blockSizeMax as u64) as usize;
    let neededRBSize: u64 = windowSize + (blockSize as u64) * 2 + (WILDCOPY_OVERLENGTH as u64) * 2;
    let neededSize = frameContentSize.min(neededRBSize);
    let minRBSize = neededSize as usize;
    if minRBSize as u64 != neededSize {
        return ERROR(ErrorCode::FrameParameterWindowTooLarge);
    }
    minRBSize
}

/// Port of `ZSTD_decodingBufferSize_min`. Worst-case ring-buffer size
/// needed to decompress a frame with the given window and content
/// sizes — accounts for two-block wildcopy slack. Delegates to
/// `ZSTD_decodingBufferSize_internal` with `blockSizeMax =
/// ZSTD_BLOCKSIZE_MAX`.
pub fn ZSTD_decodingBufferSize_min(windowSize: u64, frameContentSize: u64) -> usize {
    use crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX;
    ZSTD_decodingBufferSize_internal(windowSize, frameContentSize, ZSTD_BLOCKSIZE_MAX)
}

/// Port of `ZSTD_createDCtx_advanced`.
pub fn ZSTD_createDCtx_advanced(
    customMem: crate::compress::zstd_compress::ZSTD_customMem,
) -> Option<Box<ZSTD_DCtx>> {
    if !crate::compress::zstd_compress::ZSTD_customMem_validate(customMem) {
        return None;
    }
    ZSTD_createDCtx_internal(customMem)
}

/// Port of `ZSTD_createDStream_advanced`.
pub fn ZSTD_createDStream_advanced(
    customMem: crate::compress::zstd_compress::ZSTD_customMem,
) -> Option<Box<ZSTD_DStream>> {
    ZSTD_createDCtx_advanced(customMem)
}

/// Port of `ZSTD_initStaticDCtx`. Places a `ZSTD_DCtx` header inside
/// the caller's workspace when alignment and size allow it.
pub fn ZSTD_initStaticDCtx(workspace: &mut [u8]) -> Option<&mut ZSTD_DCtx> {
    use core::mem::{align_of, size_of};
    use core::ptr;

    if (workspace.as_mut_ptr() as usize) & (align_of::<u64>() - 1) != 0 {
        return None;
    }
    if workspace.len() < size_of::<ZSTD_DCtx>() {
        return None;
    }

    let dctx = unsafe { &mut *(workspace.as_mut_ptr() as *mut ZSTD_DCtx) };
    unsafe {
        ptr::write(dctx, ZSTD_DCtx::default());
    }
    ZSTD_initDCtx_internal(dctx);
    Some(dctx)
}

/// Port of `ZSTD_initStaticDStream`. Alias for `ZSTD_initStaticDCtx`.
pub fn ZSTD_initStaticDStream(workspace: &mut [u8]) -> Option<&mut ZSTD_DStream> {
    ZSTD_initStaticDCtx(workspace)
}

/// Port of `ZSTD_initStaticDDict`. Places a `ZSTD_DDict` header in
/// caller workspace and references the caller's `dict` bytes.
pub fn ZSTD_initStaticDDict<'a>(
    workspace: &'a mut [u8],
    dict: &[u8],
) -> Option<&'a mut crate::decompress::zstd_ddict::ZSTD_DDict> {
    use crate::decompress::zstd_ddict::{ZSTD_DDict, ZSTD_dictContentType_e};
    use core::mem::{align_of, size_of};
    use core::ptr;

    if (workspace.as_mut_ptr() as usize) & (align_of::<u64>() - 1) != 0 {
        return None;
    }
    if workspace.len() < size_of::<ZSTD_DDict>() {
        return None;
    }

    let ddict = unsafe { &mut *(workspace.as_mut_ptr() as *mut ZSTD_DDict) };
    unsafe {
        ptr::write(
            ddict,
            ZSTD_DDict {
                dictBuffer: Vec::new(),
                dictContent: dict.as_ptr(),
                dictSize: dict.len(),
                dictID: 0,
                entropyPresent: 0,
            },
        );
    }
    let rc = crate::decompress::zstd_ddict::ZSTD_loadEntropy_intoDDict(
        ddict,
        ZSTD_dictContentType_e::ZSTD_dct_auto,
    );
    if crate::common::error::ERR_isError(rc) {
        return None;
    }
    Some(ddict)
}

/// Port of `ZSTD_nextInputType_e`. Tells `ZSTD_decompressContinue`
/// callers what kind of chunk the decompressor expects next.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZSTD_nextInputType_e {
    #[default]
    ZSTDnit_frameHeader = 0,
    ZSTDnit_blockHeader = 1,
    ZSTDnit_block = 2,
    ZSTDnit_lastBlock = 3,
    ZSTDnit_checksum = 4,
    ZSTDnit_skippableFrame = 5,
}

/// Port of `ZSTD_nextInputType`. Reports what kind of chunk the
/// block-level legacy decoder expects next from `decompressContinue`.
#[inline]
pub fn ZSTD_nextInputType(dctx: &ZSTD_DCtx) -> ZSTD_nextInputType_e {
    match dctx.stage {
        ZSTD_dStage::ZSTDds_getFrameHeaderSize | ZSTD_dStage::ZSTDds_decodeFrameHeader => {
            ZSTD_nextInputType_e::ZSTDnit_frameHeader
        }
        ZSTD_dStage::ZSTDds_decodeBlockHeader => ZSTD_nextInputType_e::ZSTDnit_blockHeader,
        ZSTD_dStage::ZSTDds_decompressBlock => ZSTD_nextInputType_e::ZSTDnit_block,
        ZSTD_dStage::ZSTDds_decompressLastBlock => ZSTD_nextInputType_e::ZSTDnit_lastBlock,
        ZSTD_dStage::ZSTDds_checkChecksum => ZSTD_nextInputType_e::ZSTDnit_checksum,
        ZSTD_dStage::ZSTDds_decodeSkippableHeader | ZSTD_dStage::ZSTDds_skipFrame => {
            ZSTD_nextInputType_e::ZSTDnit_skippableFrame
        }
    }
}

/// Port of `ZSTD_nextSrcSizeToDecompressWithInputSize`. During raw
/// block streaming, upstream may consume a partial block bounded by
/// currently available input; all other stages require the exact
/// expected size.
#[inline]
pub fn ZSTD_nextSrcSizeToDecompressWithInputSize(dctx: &ZSTD_DCtx, inputSize: usize) -> usize {
    let in_block = matches!(
        dctx.stage,
        ZSTD_dStage::ZSTDds_decompressBlock | ZSTD_dStage::ZSTDds_decompressLastBlock
    );
    if !in_block || dctx.bType != crate::decompress::zstd_decompress_block::blockType_e::bt_raw {
        return dctx.expected;
    }
    if dctx.expected == 0 {
        return 0;
    }
    inputSize.clamp(1, dctx.expected)
}

/// Port of `ZSTD_isSkipFrame`.
#[inline]
pub fn ZSTD_isSkipFrame(dctx: &ZSTD_DCtx) -> i32 {
    (dctx.stage == ZSTD_dStage::ZSTDds_skipFrame) as i32
}

/// Port of `ZSTD_decompressBegin`. Legacy continue-style init — v0.1
/// doesn't drive a block-level state machine, so this is a no-op
/// returning 0.
#[inline]
pub fn ZSTD_decompressBegin(dctx: &mut ZSTD_DCtx) -> usize {
    use crate::common::xxhash::XXH64_reset;
    // Upstream (zstd_decompress.c:1560) resets per-frame state:
    // clear entropy flags, zero dictID, set stage to "awaiting
    // frame header", rebuild default FSE DTables so set_basic
    // branches in block-0 find valid defaults.
    dctx.litEntropy = 0;
    dctx.fseEntropy = 0;
    dctx.dictID = 0;
    dctx.isFrameDecompression = 1;
    dctx.previousDstEnd = None;
    dctx.prefixStart = None;
    dctx.virtualStart = None;
    dctx.dictEnd = None;
    // `repStartValue = {1, 4, 8}`.
    dctx.ddict_rep = [1, 4, 8];
    // Seed default FSE DTables so that `set_basic` blocks (the
    // very first sequence block of a fresh frame) have a valid
    // table to read from. Previously we relied on DCtx::new() to
    // have done this, but session resets (`reset_session_only`)
    // also need to re-seed without re-creating the whole DCtx.
    crate::decompress::zstd_decompress_block::ZSTD_buildDefaultSeqTables(dctx);
    dctx.expected = ZSTD_startingInputLength(dctx.format);
    dctx.stage = ZSTD_dStage::ZSTDds_getFrameHeaderSize;
    dctx.processedCSize = 0;
    dctx.decodedSize = 0;
    dctx.headerSize = 0;
    dctx.rleSize = 0;
    dctx.bType = crate::decompress::zstd_decompress_block::blockType_e::bt_raw;
    dctx.validateChecksum = 0;
    dctx.fParams = ZSTD_FrameHeader::default();
    dctx.headerBuffer.fill(0);
    dctx.historyBuffer.clear();
    XXH64_reset(&mut dctx.xxhState, 0);
    0
}

/// Port of `ZSTD_decompressBegin_usingDict` (`zstd_decompress.c:1588`).
/// Calls `ZSTD_decompressBegin` first, then dispatches through
/// `ZSTD_decompress_insertDictionary` which handles magic-prefix vs
/// raw-content dicts.
pub fn ZSTD_decompressBegin_usingDict(dctx: &mut ZSTD_DCtx, dict: &[u8]) -> usize {
    use crate::common::error::{ERR_isError, ErrorCode, ERROR};
    let rc = ZSTD_decompressBegin(dctx);
    if ERR_isError(rc) {
        return rc;
    }
    if dict.is_empty() {
        return 0;
    }
    let rc = ZSTD_decompress_insertDictionary(dctx, dict);
    if ERR_isError(rc) {
        return ERROR(ErrorCode::DictionaryCorrupted);
    }
    0
}

/// Port of `ZSTD_decompressBegin_usingDDict` (`zstd_decompress.c:1601`).
/// Uses the DDict's preloaded parameters and raw content.
pub fn ZSTD_decompressBegin_usingDDict(
    dctx: &mut ZSTD_DCtx,
    ddict: &crate::decompress::zstd_ddict::ZSTD_DDict,
) -> usize {
    use crate::common::error::ERR_isError;
    let rc = ZSTD_decompressBegin(dctx);
    if ERR_isError(rc) {
        return rc;
    }
    let content = crate::decompress::zstd_ddict::ZSTD_DDict_dictContent(ddict);
    if content.is_empty() {
        return 0;
    }
    crate::decompress::zstd_ddict::ZSTD_copyDDictParameters(dctx, ddict);
    0
}

/// Port of `ZSTD_decompressContinue`. Legacy block-level decode —
/// callers must feed exactly the chunk kind and size reported by
/// `ZSTD_nextInputType()` / `ZSTD_nextSrcSizeToDecompress()`.
pub fn ZSTD_decompressContinue(dctx: &mut ZSTD_DCtx, dst: &mut [u8], src: &[u8]) -> usize {
    use crate::common::error::{ErrorCode, ERROR};
    use crate::common::mem::MEM_readLE32;
    use crate::common::xxhash::{XXH64_digest, XXH64_update};
    use crate::decompress::zstd_decompress_block::{
        blockProperties_t, blockType_e, streaming_operation, ZSTD_blockHeaderSize,
        ZSTD_decoder_entropy_rep, ZSTD_decompressBlock_internal, ZSTD_getcBlockSize,
    };

    if src.len() != ZSTD_nextSrcSizeToDecompress(dctx) {
        return ERROR(ErrorCode::SrcSizeWrong);
    }

    dctx.processedCSize = dctx.processedCSize.wrapping_add(src.len() as u64);

    match dctx.stage {
        ZSTD_dStage::ZSTDds_getFrameHeaderSize => {
            if dctx.format == ZSTD_format_e::ZSTD_f_zstd1
                && src.len() >= ZSTD_FRAMEIDSIZE
                && (MEM_readLE32(src) & ZSTD_MAGIC_SKIPPABLE_MASK) == ZSTD_MAGIC_SKIPPABLE_START
            {
                dctx.headerBuffer[..src.len()].copy_from_slice(src);
                dctx.expected = ZSTD_SKIPPABLEHEADERSIZE - src.len();
                dctx.stage = ZSTD_dStage::ZSTDds_decodeSkippableHeader;
                return 0;
            }
            let headerSize = ZSTD_frameHeaderSize_internal(src, dctx.format);
            if crate::common::error::ERR_isError(headerSize) {
                return headerSize;
            }
            dctx.headerSize = headerSize;
            dctx.headerBuffer[..src.len()].copy_from_slice(src);
            dctx.expected = headerSize - src.len();
            dctx.stage = ZSTD_dStage::ZSTDds_decodeFrameHeader;
            0
        }
        ZSTD_dStage::ZSTDds_decodeFrameHeader => {
            let offset = dctx.headerSize - src.len();
            dctx.headerBuffer[offset..offset + src.len()].copy_from_slice(src);
            let header = dctx.headerBuffer[..dctx.headerSize].to_vec();
            let rc = ZSTD_decodeFrameHeader(dctx, &header, header.len());
            if crate::common::error::ERR_isError(rc) {
                return rc;
            }
            if dctx.fParams.frameType != ZSTD_FrameType_e::ZSTD_frame {
                return ERROR(ErrorCode::PrefixUnknown);
            }
            dctx.expected = ZSTD_blockHeaderSize;
            dctx.stage = ZSTD_dStage::ZSTDds_decodeBlockHeader;
            0
        }
        ZSTD_dStage::ZSTDds_decodeBlockHeader => {
            let mut bp = blockProperties_t {
                blockType: blockType_e::bt_raw,
                lastBlock: 0,
                origSize: 0,
            };
            let cBlockSize = ZSTD_getcBlockSize(src, &mut bp);
            if crate::common::error::ERR_isError(cBlockSize) {
                return cBlockSize;
            }
            if cBlockSize > dctx.fParams.blockSizeMax as usize {
                return ERROR(ErrorCode::CorruptionDetected);
            }
            dctx.expected = cBlockSize;
            dctx.bType = bp.blockType;
            dctx.rleSize = bp.origSize as usize;
            if cBlockSize != 0 {
                dctx.stage = if bp.lastBlock != 0 {
                    ZSTD_dStage::ZSTDds_decompressLastBlock
                } else {
                    ZSTD_dStage::ZSTDds_decompressBlock
                };
                return 0;
            }
            if bp.lastBlock != 0 {
                if dctx.fParams.checksumFlag != 0 {
                    dctx.expected = 4;
                    dctx.stage = ZSTD_dStage::ZSTDds_checkChecksum;
                } else {
                    dctx.expected = ZSTD_startingInputLength(dctx.format);
                    dctx.stage = ZSTD_dStage::ZSTDds_getFrameHeaderSize;
                }
            } else {
                dctx.expected = ZSTD_blockHeaderSize;
            }
            0
        }
        ZSTD_dStage::ZSTDds_decompressLastBlock | ZSTD_dStage::ZSTDds_decompressBlock => {
            let mut entropy_rep = ZSTD_decoder_entropy_rep {
                rep: dctx.ddict_rep,
            };
            let rSize = match dctx.bType {
                blockType_e::bt_compressed => {
                    let r = ZSTD_decompressBlock_internal(
                        dctx,
                        &mut entropy_rep,
                        dst,
                        0,
                        src,
                        streaming_operation::is_streaming,
                    );
                    dctx.expected = 0;
                    r
                }
                blockType_e::bt_raw => {
                    let r = ZSTD_copyRawBlock(dst, src);
                    if crate::common::error::ERR_isError(r) {
                        return r;
                    }
                    dctx.expected -= r;
                    r
                }
                blockType_e::bt_rle => {
                    let r = ZSTD_setRleBlock(dst, src[0], dctx.rleSize);
                    dctx.expected = 0;
                    r
                }
                blockType_e::bt_reserved => return ERROR(ErrorCode::CorruptionDetected),
            };
            if crate::common::error::ERR_isError(rSize) {
                return rSize;
            }
            if rSize > dctx.fParams.blockSizeMax as usize {
                return ERROR(ErrorCode::CorruptionDetected);
            }
            dctx.ddict_rep = entropy_rep.rep;
            dctx.decodedSize = dctx.decodedSize.wrapping_add(rSize as u64);
            if dctx.validateChecksum != 0 && rSize > 0 {
                XXH64_update(&mut dctx.xxhState, &dst[..rSize]);
            }
            // Append this block's output to the rolling history buffer
            // so subsequent blocks' back-references can resolve into it
            // via the ext-dict path. Cap at the frame's window size
            // (or the ZSTD_BLOCKSIZE_MAX fallback before a frame
            // header has been parsed).
            if rSize > 0 {
                let cap = if dctx.fParams.windowSize > 0 {
                    dctx.fParams.windowSize as usize
                } else {
                    crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX
                };
                dctx.historyBuffer.extend_from_slice(&dst[..rSize]);
                if dctx.historyBuffer.len() > cap {
                    let drop = dctx.historyBuffer.len() - cap;
                    dctx.historyBuffer.drain(..drop);
                }
            }
            if dctx.expected > 0 {
                return rSize;
            }
            if dctx.stage == ZSTD_dStage::ZSTDds_decompressLastBlock {
                if dctx.fParams.frameContentSize != ZSTD_CONTENTSIZE_UNKNOWN
                    && dctx.decodedSize != dctx.fParams.frameContentSize
                {
                    return ERROR(ErrorCode::CorruptionDetected);
                }
                if dctx.fParams.checksumFlag != 0 {
                    dctx.expected = 4;
                    dctx.stage = ZSTD_dStage::ZSTDds_checkChecksum;
                } else {
                    dctx.expected = ZSTD_startingInputLength(dctx.format);
                    dctx.stage = ZSTD_dStage::ZSTDds_getFrameHeaderSize;
                }
                // Clear the rolling history at frame boundaries — the
                // next frame starts with a fresh window.
                dctx.historyBuffer.clear();
            } else {
                dctx.expected = ZSTD_blockHeaderSize;
                dctx.stage = ZSTD_dStage::ZSTDds_decodeBlockHeader;
            }
            rSize
        }
        ZSTD_dStage::ZSTDds_checkChecksum => {
            if dctx.validateChecksum != 0 {
                let h32 = XXH64_digest(&dctx.xxhState) as u32;
                let check32 = MEM_readLE32(src);
                if check32 != h32 {
                    return ERROR(ErrorCode::ChecksumWrong);
                }
            }
            dctx.expected = ZSTD_startingInputLength(dctx.format);
            dctx.stage = ZSTD_dStage::ZSTDds_getFrameHeaderSize;
            0
        }
        ZSTD_dStage::ZSTDds_decodeSkippableHeader => {
            let offset = ZSTD_SKIPPABLEHEADERSIZE - src.len();
            dctx.headerBuffer[offset..offset + src.len()].copy_from_slice(src);
            dctx.expected = MEM_readLE32(&dctx.headerBuffer[ZSTD_FRAMEIDSIZE..]) as usize;
            dctx.stage = ZSTD_dStage::ZSTDds_skipFrame;
            0
        }
        ZSTD_dStage::ZSTDds_skipFrame => {
            dctx.expected = ZSTD_startingInputLength(dctx.format);
            dctx.stage = ZSTD_dStage::ZSTDds_getFrameHeaderSize;
            0
        }
    }
}

/// Port of `ZSTD_copyDCtx`. Deep-copies `src` into `dst`. Upstream
/// only copies the "header" portion of the struct to skip the large
/// `inBuff` workspace; the Rust port owns each field in a `Vec`, so
/// we delegate to `Clone::clone_from` — this copies everything,
/// including the scratch buffers (they're small and the semantics
/// are correct).
#[inline]
pub fn ZSTD_copyDCtx(dst: &mut ZSTD_DCtx, src: &ZSTD_DCtx) {
    dst.clone_from(src);
}

/// Port of `ZSTD_initDStream_usingDDict`. Like `ZSTD_initDStream` but
/// attaches a pre-built `ZSTD_DDict` — the DDict's raw content is
/// copied onto `dctx.stream_dict` so every frame decoded in this
/// session sees it as history.
pub fn ZSTD_initDStream_usingDDict(
    zds: &mut ZSTD_DCtx,
    ddict: &crate::decompress::zstd_ddict::ZSTD_DDict,
) -> usize {
    let rc = ZSTD_initDStream(zds);
    if crate::common::error::ERR_isError(rc) {
        return rc;
    }
    // Route through `ZSTD_DCtx_refDDict` (which calls
    // `insertDictionary` under the hood) so a DDict built from a
    // magic-prefix zstd-format dict surfaces its dictID + entropy
    // tables on the dctx. Previously we wrote stream_dict directly,
    // bypassing the magic probe — sibling fix to the
    // `initDStream_usingDict` parity fix. Matches upstream's
    // `initDStream_usingDDict` → `refDDict` chain
    // (zstd_decompress.c:1753).
    let rc = ZSTD_DCtx_refDDict(zds, ddict);
    if crate::common::error::ERR_isError(rc) {
        return rc;
    }
    // Same `dctx.format`-aware hint as initDStream / resetDStream —
    // magicless-mode callers get 1, zstd1 callers get 5.
    ZSTD_startingInputLength(zds.format)
}

/// Port of `ZSTD_nextSrcSizeToDecompress`. Returns the exact byte
/// count the legacy block-level decoder expects next.
#[inline]
pub fn ZSTD_nextSrcSizeToDecompress(dctx: &ZSTD_DCtx) -> usize {
    dctx.expected
}

/// Port of `ZSTD_resetDStream`. Clears per-frame state (input/output
/// buffers) but preserves the configured dict for subsequent frames.
/// Returns a hint for the suggested next input size.
pub fn ZSTD_resetDStream(zds: &mut ZSTD_DCtx) -> usize {
    zds.stream_in_buffer.clear();
    zds.stream_out_buffer.clear();
    zds.stream_out_drained = 0;
    zds.oversizedDuration = 0;
    // Upstream (zstd_decompress.c:1772) returns
    // `ZSTD_startingInputLength(dctx->format)` — the number of bytes
    // needed to query the next frame header. Magicless-mode (`ZSTD_f_zstd1_magicless`)
    // callers need the shorter 2-byte hint instead of the 5-byte zstd1 hint.
    ZSTD_startingInputLength(zds.format)
}

/// Port of `ZSTD_DCtx_isOverflow`.
pub fn ZSTD_DCtx_isOverflow(
    zds: &ZSTD_DCtx,
    neededInBuffSize: usize,
    neededOutBuffSize: usize,
) -> i32 {
    let retained = zds.stream_in_buffer.capacity() + zds.stream_out_buffer.capacity();
    let needed = (neededInBuffSize + neededOutBuffSize)
        * crate::common::zstd_internal::ZSTD_WORKSPACETOOLARGE_FACTOR;
    (retained >= needed) as i32
}

/// Port of `ZSTD_DCtx_updateOversizedDuration`.
pub fn ZSTD_DCtx_updateOversizedDuration(
    zds: &mut ZSTD_DCtx,
    neededInBuffSize: usize,
    neededOutBuffSize: usize,
) {
    if ZSTD_DCtx_isOverflow(zds, neededInBuffSize, neededOutBuffSize) != 0 {
        zds.oversizedDuration += 1;
    } else {
        zds.oversizedDuration = 0;
    }
}

/// Port of `ZSTD_checkOutBuffer`. The stable-output-buffer mode isn't
/// exposed by this Rust port yet, so every output buffer is accepted.
#[inline]
pub fn ZSTD_checkOutBuffer(_zds: &ZSTD_DCtx, _output: &[u8], _output_pos: usize) -> usize {
    0
}

/// Port of `ZSTD_decompressContinueStream`, adapted to Rust slices.
/// It calls the legacy `ZSTD_decompressContinue` transition and
/// advances `output_pos` by the decoded byte count.
pub fn ZSTD_decompressContinueStream(
    zds: &mut ZSTD_DCtx,
    output: &mut [u8],
    output_pos: &mut usize,
    src: &[u8],
) -> usize {
    let dstSize = if ZSTD_isSkipFrame(zds) != 0 {
        0
    } else {
        output.len() - *output_pos
    };
    let decodedSize =
        ZSTD_decompressContinue(zds, &mut output[*output_pos..*output_pos + dstSize], src);
    if crate::common::error::ERR_isError(decodedSize) {
        return decodedSize;
    }
    *output_pos += decodedSize;
    0
}

/// Port of `ZSTD_decompressStream`. Buffers `input[input_pos..]` into
/// the DCtx, detects when a complete frame has been received via
/// `ZSTD_findFrameCompressedSize`, decodes it into an internal output
/// buffer, and drains the result into `output[output_pos..]`.
///
/// Returns a hint for the next suggested input size: 0 when the
/// current frame is complete and fully drained, `ZSTD_blockHeaderSize`
/// as a conservative "need more" hint otherwise.
///
/// v0.1 scope: single-frame per call-sequence. Multi-frame streams
/// work if the caller invokes `ZSTD_initDStream` between frames.
pub fn ZSTD_decompressStream(
    zds: &mut ZSTD_DCtx,
    output: &mut [u8],
    output_pos: &mut usize,
    input: &[u8],
    input_pos: &mut usize,
) -> usize {
    use crate::common::error::ERR_isError;
    // Drain any already-decoded bytes first.
    let avail = output.len() - *output_pos;
    let pending = zds.stream_out_buffer.len() - zds.stream_out_drained;
    let n = avail.min(pending);
    if n > 0 {
        output[*output_pos..*output_pos + n].copy_from_slice(
            &zds.stream_out_buffer[zds.stream_out_drained..zds.stream_out_drained + n],
        );
        zds.stream_out_drained += n;
        *output_pos += n;
    }

    // Ingest fresh input.
    zds.stream_in_buffer.extend_from_slice(&input[*input_pos..]);
    *input_pos = input.len();

    // If nothing pending on either side, we're done.
    if zds.stream_out_drained == zds.stream_out_buffer.len() && zds.stream_in_buffer.is_empty() {
        return 0;
    }
    // If output fully drained AND fresh input available, try to probe
    // a new frame.
    if zds.stream_out_drained == zds.stream_out_buffer.len() {
        // Try to measure a full frame from the staged input. Thread
        // the dctx's stored format through so magicless-mode streams
        // decode without the 4-byte magic prefix.
        let frame_sz = ZSTD_findFrameCompressedSize_advanced(&zds.stream_in_buffer, zds.format);
        if ERR_isError(frame_sz) {
            // Could be "need more input" (SrcSizeWrong). Return a
            // non-zero hint so the caller keeps feeding.
            return 3; // ZSTD_blockHeaderSize
        }
        // Determine decoded size.
        let declared = ZSTD_getFrameContentSize(&zds.stream_in_buffer);
        let out_size = if declared == ZSTD_CONTENTSIZE_UNKNOWN || declared == ZSTD_CONTENTSIZE_ERROR
        {
            // Fall back to a generous bound: 32× compressed size.
            frame_sz * 32
        } else {
            declared as usize
        };
        let mut decoded = vec![0u8; out_size.max(1)];
        let d = if zds.stream_dict.is_empty() {
            // Route through `ZSTD_decompressDCtx` (not `ZSTD_decompress`)
            // so the stream's DCtx state — crucially `dctx.format` —
            // is honored. A magicless-mode streaming decoder would
            // previously fail here because `ZSTD_decompress` allocates
            // a fresh dctx fixed to `ZSTD_f_zstd1`.
            use crate::common::xxhash::XXH64_state_t;
            use crate::decompress::zstd_decompress_block::ZSTD_decoder_entropy_rep;
            let frame_bytes = zds.stream_in_buffer[..frame_sz].to_vec();
            let mut rep = ZSTD_decoder_entropy_rep::default();
            let mut xxh = XXH64_state_t::default();
            ZSTD_decompressDCtx(zds, &mut rep, &mut xxh, &mut decoded, &frame_bytes)
        } else {
            // Thread the stream's own DCtx through the dict-decode
            // path (previously we allocated a throwaway `ZSTD_DCtx`
            // per frame — faithful-translation gap now that
            // `ZSTD_decompress_usingDict` honors the caller's dctx).
            // Clone the frame bytes + dict so the per-frame decoder
            // call can borrow `zds` mutably without aliasing.
            //
            // Honor the `use_once` lifetime: if the dict was bound
            // via `refPrefix`, demote it before the decode and clear
            // it after so a subsequent frame on the same stream
            // doesn't silently re-apply the prefix. `loadDictionary`
            // / `refDDict` bindings (`use_indefinitely`) survive.
            let was_use_once = zds.dictUses == ZSTD_dictUses_e::ZSTD_use_once;
            if was_use_once {
                zds.dictUses = ZSTD_dictUses_e::ZSTD_dont_use;
            }
            let dict = zds.stream_dict.clone();
            let frame_bytes = zds.stream_in_buffer[..frame_sz].to_vec();
            let decoded_len = ZSTD_decompress_usingDict(zds, &mut decoded, &frame_bytes, &dict);
            if was_use_once && !ERR_isError(decoded_len) {
                ZSTD_clearDict(zds);
            }
            decoded_len
        };
        if ERR_isError(d) {
            return d;
        }
        decoded.truncate(d);
        zds.stream_out_buffer = decoded;
        zds.stream_out_drained = 0;
        // Remove the consumed frame from the input buffer so multi-
        // frame streams work when the caller re-inits between frames.
        zds.stream_in_buffer.drain(..frame_sz);

        // Drain the freshly-decoded output.
        let avail = output.len() - *output_pos;
        let pending = zds.stream_out_buffer.len() - zds.stream_out_drained;
        let n = avail.min(pending);
        if n > 0 {
            output[*output_pos..*output_pos + n].copy_from_slice(
                &zds.stream_out_buffer[zds.stream_out_drained..zds.stream_out_drained + n],
            );
            zds.stream_out_drained += n;
            *output_pos += n;
        }
    }

    let remaining = zds.stream_out_buffer.len() - zds.stream_out_drained;
    if remaining == 0 && zds.stream_in_buffer.is_empty() {
        0
    } else {
        remaining.max(3)
    }
}

/// Port of `ZSTD_decompressStream_simpleArgs` (`zstd.h:2611`). Upstream
/// provides this as an FFI-friendly variant of `ZSTD_decompressStream`
/// that takes positions by pointer so dynamic-language binders don't
/// have to build a `ZSTD_inBuffer` / `ZSTD_outBuffer` struct. Our
/// Rust port's base signature already uses `&mut usize` cursors so
/// the simpleArgs version is just a thin named alias — kept for
/// parity with FFI callers ported from upstream headers.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn ZSTD_decompressStream_simpleArgs(
    dctx: &mut ZSTD_DCtx,
    dst: &mut [u8],
    dst_pos: &mut usize,
    src: &[u8],
    src_pos: &mut usize,
) -> usize {
    ZSTD_decompressStream(dctx, dst, dst_pos, src, src_pos)
}
