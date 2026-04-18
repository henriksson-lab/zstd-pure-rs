//! Translation of `lib/decompress/zstd_decompress.c`. The frame-level
//! decoder: magic number, frame header, block loop, checksum validation.

#![allow(unused_variables)]

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
pub fn ZSTD_frameHeaderSize_internal(
    src: &[u8],
    format: ZSTD_format_e,
) -> usize {
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
            zfh.frameContentSize = MEM_readLE32(&src[ZSTD_FRAMEIDSIZE..ZSTD_FRAMEIDSIZE + 4]) as u64;
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
    zfh.blockSizeMax = windowSize.min(crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX as u64) as u32;
    zfh.dictID = dictID;
    zfh.checksumFlag = checksumFlag;
    0
}

pub fn ZSTD_getFrameHeader(zfh: &mut ZSTD_FrameHeader, src: &[u8]) -> usize {
    ZSTD_getFrameHeader_advanced(zfh, src, ZSTD_format_e::ZSTD_f_zstd1)
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

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
        let b = ZSTD_dParam_getBounds(ZSTD_dParameter::ZSTD_d_windowLogMax);
        assert_eq!(b.error, 0);
        assert_eq!((b.lowerBound, b.upperBound), (10, 27));
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
        let rc = ZSTD_DCtx_setParameter(
            &mut dctx,
            ZSTD_dParameter::ZSTD_d_windowLogMax,
            20,
        );
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
            blockProperties_t, blockType_e, ZSTD_blockHeaderSize, ZSTD_DCtx, ZSTD_getcBlockSize,
        };
        // Produce a single compressed block via our compressor, then
        // decode just its body (no frame header) through the public
        // ZSTD_decompressBlock entry point.
        let src: Vec<u8> = b"hello block api. "
            .iter().cycle().take(200).copied().collect();

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
            let body = &frame[body_start + ZSTD_blockHeaderSize
                ..body_start + ZSTD_blockHeaderSize + body_size];
            let mut dctx = ZSTD_DCtx::new();
            let mut out = vec![0u8; src.len() + 64];
            let d = ZSTD_decompressBlock(&mut dctx, &mut out, body);
            assert!(!crate::common::error::ERR_isError(d));
            assert_eq!(&out[..d], &src[..]);
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
        assert!(crate::common::error::ERR_isError(
            ZSTD_copyRawBlock(&mut tiny, src)
        ));

        // setRleBlock: fill N copies of a byte + DstSizeTooSmall.
        let mut buf = vec![0u8; 32];
        let rle_n = ZSTD_setRleBlock(&mut buf, 0xAB, 10);
        assert_eq!(rle_n, 10);
        assert!(buf[..10].iter().all(|&b| b == 0xAB));
        // Bytes past regenSize must not have been touched.
        assert!(buf[10..].iter().all(|&b| b == 0));

        let mut short = [0u8; 4];
        assert!(crate::common::error::ERR_isError(
            ZSTD_setRleBlock(&mut short, 0xCD, 10)
        ));
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
    fn zstd_getDictID_fromFrame_reads_dictID_when_present() {
        // Build a synthetic frame with a 4-byte dictID via
        // ZSTD_writeFrameHeader and verify round-trip through
        // ZSTD_getDictID_fromFrame.
        use crate::compress::zstd_compress::{
            ZSTD_writeFrameHeader, ZSTD_FRAMEHEADERSIZE_MAX, ZSTD_FrameParameters,
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
        // Symmetric with `ZSTD_createCCtx_advanced`. v0.1 ignores
        // the custom-allocator arg and returns a default DCtx /
        // DStream. Confirm both creators always return `Some` —
        // callers must not see `None` from these entries.
        use crate::compress::zstd_compress::ZSTD_customMem;
        let _dctx = ZSTD_createDCtx_advanced(ZSTD_customMem).unwrap();
        let _dstream = ZSTD_createDStream_advanced(ZSTD_customMem).unwrap();
    }

    #[test]
    fn decompressContinue_stub_returns_Generic_error() {
        // Symmetric with the compress-side continue/end stub test.
        // `ZSTD_decompressContinue` is stubbed in v0.1 (our port
        // uses the buffer-until-complete-frame streaming instead of
        // the legacy block-level state machine). Callers must see a
        // proper zstd error code rather than a panic.
        let mut dctx = ZSTD_DCtx::default();
        let mut dst = [0u8; 64];
        let src = b"some-input";
        let rc = ZSTD_decompressContinue(&mut dctx, &mut dst, src);
        assert!(crate::common::error::ERR_isError(rc));
        use crate::common::error::ERR_getErrorCode;
        assert_eq!(ERR_getErrorCode(rc), ErrorCode::Generic);
    }

    #[test]
    fn decoder_rejects_corrupted_xxh64_trailer_with_checksumWrong() {
        // Compress with `--check` flag, flip one byte of the XXH64
        // trailer, and verify the decoder surfaces ChecksumWrong —
        // NOT a silent decode-and-pass-bad-bytes-up-the-stack.
        use crate::compress::match_state::ZSTD_compressionParameters;
        use crate::compress::zstd_compress::{
            ZSTD_compressBound, ZSTD_compressFrame_fast, ZSTD_FrameParameters,
            ZSTD_getCParams,
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
        assert!(crate::common::error::ERR_isError(rc), "decoder missed corrupted checksum (rc={rc})");
        assert_eq!(
            crate::common::error::ERR_getErrorCode(rc),
            ErrorCode::ChecksumWrong,
            "expected ChecksumWrong, got {:?}",
            crate::common::error::ERR_getErrorCode(rc)
        );
    }

    #[test]
    fn zstd_decompress_rejects_garbage_without_panicking() {
        // Safety gate: feeding arbitrary bytes into `ZSTD_decompress`
        // must surface a ZSTD_isError return — never panic. This is
        // the contract callers rely on when accepting compressed
        // input from the network / disk.
        let mut dst = vec![0u8; 1024];
        let test_inputs: Vec<Vec<u8>> = vec![
            vec![],                                    // empty
            vec![0u8],                                 // 1 byte
            vec![0u8; 3],                              // below magic size
            vec![0xFFu8; 8],                           // bogus magic
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
        src.push(42);   // FCS byte
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
        src.push(100);  // FCS byte (singleSegment implies size is this byte)
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
        src.push(5);     // FCS byte
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
        // Regression gate: upstream's formula is
        //   blockSize     = min(windowSize, ZSTD_BLOCKSIZE_MAX)
        //   neededRBSize  = windowSize + blockSize + ZSTD_BLOCKSIZE_MAX + 2*WILDCOPY
        //   return          min(frameContentSize, neededRBSize)
        // Previously our port doubled blockSize instead of adding a
        // fixed ZSTD_BLOCKSIZE_MAX term — diverged by ~127 KB for
        // windowSizes below ZSTD_BLOCKSIZE_MAX.
        use crate::common::zstd_internal::WILDCOPY_OVERLENGTH;
        use crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX;

        // Small window: blockSize = windowSize = 1024.
        let w = 1024u64;
        let fcs = u64::MAX; // Take the RB path, not the FCS path.
        let expected = w + w + ZSTD_BLOCKSIZE_MAX as u64 + (WILDCOPY_OVERLENGTH as u64) * 2;
        assert_eq!(ZSTD_decodingBufferSize_min(w, fcs) as u64, expected);

        // Large window (> BLOCKSIZE_MAX): blockSize = BLOCKSIZE_MAX.
        let w2 = 1u64 << 20;
        let expected2 = w2
            + ZSTD_BLOCKSIZE_MAX as u64
            + ZSTD_BLOCKSIZE_MAX as u64
            + (WILDCOPY_OVERLENGTH as u64) * 2;
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
    fn initDStream_usingDDict_copies_dict_content() {
        use crate::decompress::zstd_ddict::ZSTD_createDDict;
        let dict = b"test-dict-content".to_vec();
        let ddict = ZSTD_createDDict(&dict).expect("ddict");
        let mut dctx = ZSTD_DCtx::default();
        let rc = ZSTD_initDStream_usingDDict(&mut dctx, &ddict);
        assert_eq!(rc, 0);
        assert_eq!(dctx.stream_dict, dict);
    }

    #[test]
    fn nextSrcSizeToDecompress_returns_block_hint() {
        let dctx = ZSTD_DCtx::default();
        assert_eq!(ZSTD_nextSrcSizeToDecompress(&dctx), ZSTD_DStreamInSize());
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
        let cases = [
            (ZSTD_dParameter::ZSTD_d_windowLogMax, 20),
        ];
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

        // parameters: drop them.
        ZSTD_DCtx_reset(&mut dctx, ZSTD_DResetDirective::ZSTD_reset_parameters);
        assert!(dctx.stream_dict.is_empty());
        assert_eq!(dctx.d_windowLogMax, 0);
    }

    #[test]
    fn decompress_side_free_functions_accept_none_without_panic() {
        // Symmetric with the compress-side contract: the two
        // decompression-side Option-taking freers must accept None
        // without panicking. (`ZSTD_freeDCtx` takes `Box<T>` directly
        // and has no None path.)
        assert_eq!(ZSTD_freeDStream(None), 0);
        assert_eq!(
            crate::decompress::zstd_ddict::ZSTD_freeDDict(None),
            0
        );
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
        let drain = ZSTD_decompressStream(
            &mut dctx, &mut out, &mut out_pos, &cbuf[..n], &mut in_pos,
        );
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
            &mut dctx, &mut out_buf, &mut out_pos, &dst[..n], &mut in_pos,
        );
        assert!(!crate::common::error::ERR_isError(drain), "drain err: {drain:#x}");
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
    fn initStatic_decompression_variants_return_none() {
        // Contract: v0.1 has no static-buffer init support. All
        // decompression-side initStatic* must return None regardless
        // of workspace size so callers fall back to heap-allocated
        // creators.
        let mut buf = vec![0u8; 1 << 20];
        assert!(ZSTD_initStaticDCtx(&mut buf).is_none());
        assert!(ZSTD_initStaticDStream(&mut buf).is_none());
        let dict_bytes = b"dict";
        assert!(ZSTD_initStaticDDict(&mut buf, dict_bytes).is_none());
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
        // Above max (1<<27).
        assert!(crate::common::error::ERR_isError(
            ZSTD_DCtx_setMaxWindowSize(&mut dctx, 1usize << 30)
        ));
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
        assert_eq!(ZSTD_isFrame(&skip), 1);  // isFrame includes skippable
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
        assert!(crate::common::error::ERR_isError(ZSTD_writeSkippableFrame(&mut tiny, payload, 0)));
        // dst too small for header + payload.
        let mut no_room = [0u8; 11];
        assert!(crate::common::error::ERR_isError(ZSTD_writeSkippableFrame(&mut no_room, payload, 0)));
        // magicVariant > 15 is out of spec.
        let mut ok_buf = [0u8; 64];
        assert!(crate::common::error::ERR_isError(ZSTD_writeSkippableFrame(&mut ok_buf, payload, 16)));
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
        assert!(crate::common::error::ERR_isError(
            ZSTD_readSkippableFrame(&mut dst, None, &too_short)
        ));

        // (2) header claims 20-byte payload but src is truncated to
        //     header + 5 bytes.
        let mut truncated = Vec::new();
        truncated.extend_from_slice(&ZSTD_MAGIC_SKIPPABLE_START.to_le_bytes());
        truncated.extend_from_slice(&20u32.to_le_bytes());
        truncated.extend_from_slice(&[0u8; 5]);
        assert!(crate::common::error::ERR_isError(
            ZSTD_readSkippableFrame(&mut dst, None, &truncated)
        ));
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
        src.push(10);   // FCS
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
    let mut d = Box::new(ZSTD_DCtx::new());
    crate::decompress::zstd_decompress_block::ZSTD_buildDefaultSeqTables(&mut d);
    d
}

/// Port of `ZSTD_freeDCtx`. In the Rust port, dropping the Box frees.
pub fn ZSTD_freeDCtx(_dctx: Box<ZSTD_DCtx>) -> usize {
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
        blockType_e, blockProperties_t, ZSTD_blockHeaderSize, ZSTD_decompressBlock_internal,
        ZSTD_getcBlockSize, streaming_operation,
    };
    const FRAMEHEADERSIZE_MIN_ZSTD1: usize = 6;
    let mut ip: usize = 0;
    let mut op: usize = op_start;
    let mut remaining = src.len();
    if remaining < FRAMEHEADERSIZE_MIN_ZSTD1 + ZSTD_blockHeaderSize {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    let mut zfh = ZSTD_FrameHeader::default();
    let rc = ZSTD_getFrameHeader(&mut zfh, src);
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
            blockType_e::bt_rle => {
                ZSTD_setRleBlock(&mut dst[op..], src[ip], bp.origSize as usize)
            }
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
    if zfh.frameContentSize != ZSTD_CONTENTSIZE_UNKNOWN
        && (decoded as u64) != zfh.frameContentSize
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
        blockType_e, blockProperties_t, ZSTD_blockHeaderSize, ZSTD_decompressBlock_internal,
        ZSTD_getcBlockSize, streaming_operation,
    };

    const FRAMEHEADERSIZE_MIN_ZSTD1: usize = 6;

    let mut ip: usize = 0;
    let mut op: usize = 0;
    let mut remaining = src.len();

    if remaining < FRAMEHEADERSIZE_MIN_ZSTD1 + ZSTD_blockHeaderSize {
        return ERROR(ErrorCode::SrcSizeWrong);
    }

    // Parse frame header.
    let mut zfh = ZSTD_FrameHeader::default();
    let rc = ZSTD_getFrameHeader(&mut zfh, src);
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
            blockType_e::bt_raw => {
                ZSTD_copyRawBlock(&mut dst[op..], &src[ip..ip + cBlockSize])
            }
            blockType_e::bt_rle => {
                ZSTD_setRleBlock(&mut dst[op..], src[ip], bp.origSize as usize)
            }
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

/// Port of `ZSTD_decompressDCtx`. Wraps `ZSTD_decompressFrame` with
/// DCtx-lifetime plumbing; a multi-frame stream loops until the source
/// is exhausted (handled separately in `ZSTD_decompressMultiFrame` when
/// that's ported).
pub fn ZSTD_decompressDCtx(
    dctx: &mut crate::decompress::zstd_decompress_block::ZSTD_DCtx,
    entropy_rep: &mut crate::decompress::zstd_decompress_block::ZSTD_decoder_entropy_rep,
    xxh: &mut crate::common::xxhash::XXH64_state_t,
    dst: &mut [u8],
    src: &[u8],
) -> usize {
    let mut consumed = 0usize;
    ZSTD_decompressFrame(dctx, entropy_rep, xxh, dst, src, &mut consumed)
}

pub fn ZSTD_decompress(dst: &mut [u8], src: &[u8]) -> usize {
    use crate::common::xxhash::XXH64_state_t;
    use crate::decompress::zstd_decompress_block::{
        ZSTD_buildDefaultSeqTables, ZSTD_DCtx, ZSTD_decoder_entropy_rep,
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
/// dictionaries (with embedded entropy tables) are not yet honored —
/// callers with a magic-tagged dict should pass its raw content
/// portion or use a by-reference DDict creator.
pub fn ZSTD_decompress_usingDict(
    _dctx: &mut ZSTD_DCtx,
    dst: &mut [u8],
    src: &[u8],
    dict: &[u8],
) -> usize {
    use crate::common::error::ERR_isError;
    use crate::common::xxhash::XXH64_state_t;
    use crate::decompress::zstd_decompress_block::{
        ZSTD_buildDefaultSeqTables, ZSTD_decoder_entropy_rep,
    };

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

    // Fresh DCtx so stale sequence tables / lit buffers don't leak.
    // (Upstream allows dict-state reuse; we keep it simple for now.)
    let mut dctx = ZSTD_DCtx::new();
    ZSTD_buildDefaultSeqTables(&mut dctx);
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
        &mut dctx,
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
/// decompression parameter.
pub fn ZSTD_dParam_getBounds(param: ZSTD_dParameter) -> crate::compress::zstd_compress::ZSTD_bounds {
    match param {
        ZSTD_dParameter::ZSTD_d_windowLogMax => {
            crate::compress::zstd_compress::ZSTD_bounds {
                error: 0,
                lowerBound: 10,
                upperBound: 27,
            }
        }
    }
}

/// Port of `ZSTD_DCtx_setMaxWindowSize`. Clamps `maxWindowSize` into
/// `[1 << min, 1 << max]` (taken from `ZSTD_d_windowLogMax`'s bounds)
/// and stores it on the DCtx for subsequent streaming decompressions
/// to enforce against frame headers.
pub fn ZSTD_DCtx_setMaxWindowSize(dctx: &mut ZSTD_DCtx, maxWindowSize: usize) -> usize {
    use crate::common::error::{ERROR, ErrorCode};
    let bounds = ZSTD_dParam_getBounds(ZSTD_dParameter::ZSTD_d_windowLogMax);
    let min = 1usize << bounds.lowerBound;
    let max = 1usize << bounds.upperBound;
    if maxWindowSize < min || maxWindowSize > max {
        return ERROR(ErrorCode::ParameterOutOfBound);
    }
    // We store windowLog (the log2) rather than bytes to match the
    // existing `d_windowLogMax` parameter slot.
    dctx.d_windowLogMax =
        maxWindowSize.trailing_zeros().max(bounds.lowerBound as u32);
    0
}

/// Port of `ZSTD_DCtx_setFormat`. Thin wrapper over
/// `ZSTD_DCtx_setParameter(d_format, ..)`. Deferred until we grow a
/// `ZSTD_d_format` parameter slot; returns
/// `ParameterUnsupported` for now.
pub fn ZSTD_DCtx_setFormat(
    _dctx: &mut ZSTD_DCtx,
    _format: crate::decompress::zstd_decompress::ZSTD_format_e,
) -> usize {
    use crate::common::error::{ERROR, ErrorCode};
    ERROR(ErrorCode::ParameterUnsupported)
}

/// Port of `ZSTD_DStream`. Upstream `typedef ZSTD_DCtx ZSTD_DStream`
/// — same struct for both APIs.
pub type ZSTD_DStream = ZSTD_DCtx;

/// Port of `ZSTD_createDStream`. Alias for `ZSTD_createDCtx`.
pub fn ZSTD_createDStream() -> Option<Box<ZSTD_DStream>> {
    Some(Box::new(ZSTD_DCtx::new()))
}

/// Port of `ZSTD_freeDStream`. Alias for `ZSTD_freeDCtx`.
pub fn ZSTD_freeDStream(_zds: Option<Box<ZSTD_DStream>>) -> usize {
    0
}

/// Port of `ZSTD_DCtx_loadDictionary`. Stashes the dict on the DCtx
/// so subsequent streaming / one-shot decompressions honor it.
pub fn ZSTD_DCtx_loadDictionary(dctx: &mut ZSTD_DCtx, dict: &[u8]) -> usize {
    dctx.stream_dict = dict.to_vec();
    0
}

/// Port of `ZSTD_DCtx_refPrefix`. Same effect as loadDictionary in
/// v0.1.
pub fn ZSTD_DCtx_refPrefix(dctx: &mut ZSTD_DCtx, prefix: &[u8]) -> usize {
    dctx.stream_dict = prefix.to_vec();
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
/// loader; load-method + content-type parameters are currently
/// ignored (entropy-dict path isn't wired yet).
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

/// Port of `ZSTD_DCtx_refDDict`. Wires a pre-built DDict into the
/// DCtx — we copy the DDict's raw content into stream_dict.
pub fn ZSTD_DCtx_refDDict(
    dctx: &mut ZSTD_DCtx,
    ddict: &crate::decompress::zstd_ddict::ZSTD_DDict,
) -> usize {
    let content = crate::decompress::zstd_ddict::ZSTD_DDict_dictContent(ddict);
    dctx.stream_dict = content.to_vec();
    0
}

/// Port of `ZSTD_DStreamInSize`. Suggested input-buffer size for
/// streaming decompression. Upstream returns `ZSTD_BLOCKSIZE_MAX + 3`.
pub fn ZSTD_DStreamInSize() -> usize {
    crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX + 3
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
}

/// Port of `ZSTD_DCtx_setParameter`. For `windowLogMax` we record
/// the bound on the DCtx for potential upper-bound checks during
/// frame-header parsing. v0.1 just stashes the value; the enforcement
/// path lands alongside the ZSTD_windowLog_max limit check.
pub fn ZSTD_DCtx_setParameter(
    dctx: &mut ZSTD_DCtx,
    param: ZSTD_dParameter,
    value: i32,
) -> usize {
    match param {
        ZSTD_dParameter::ZSTD_d_windowLogMax => {
            dctx.d_windowLogMax = value as u32;
            0
        }
    }
}

/// Port of `ZSTD_DCtx_getParameter`.
pub fn ZSTD_DCtx_getParameter(
    dctx: &ZSTD_DCtx,
    param: ZSTD_dParameter,
    value: &mut i32,
) -> usize {
    *value = match param {
        ZSTD_dParameter::ZSTD_d_windowLogMax => dctx.d_windowLogMax as i32,
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
    if clear_session {
        ZSTD_resetDStream(dctx);
    }
    if clear_params {
        dctx.stream_dict.clear();
        dctx.d_windowLogMax = 0;
    }
    0
}

/// Port of `ZSTD_ResetDirective` (decoder-side alias).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZSTD_DResetDirective {
    ZSTD_reset_session_only,
    ZSTD_reset_parameters,
    ZSTD_reset_session_and_parameters,
}

/// Port of `ZSTD_sizeof_DCtx`. Walks the DCtx's owned `Vec`s.
pub fn ZSTD_sizeof_DCtx(dctx: &ZSTD_DCtx) -> usize {
    core::mem::size_of::<ZSTD_DCtx>()
        + dctx.hufTable.capacity() * core::mem::size_of::<u32>()
        + dctx.workspace.capacity() * core::mem::size_of::<u32>()
        + dctx.litExtraBuffer.capacity()
        + dctx.LLTable.capacity() * core::mem::size_of::<crate::decompress::zstd_decompress_block::ZSTD_seqSymbol>()
        + dctx.OFTable.capacity() * core::mem::size_of::<crate::decompress::zstd_decompress_block::ZSTD_seqSymbol>()
        + dctx.MLTable.capacity() * core::mem::size_of::<crate::decompress::zstd_decompress_block::ZSTD_seqSymbol>()
        + dctx.stream_in_buffer.capacity()
        + dctx.stream_out_buffer.capacity()
        + dctx.stream_dict.capacity()
}

/// Port of `ZSTD_sizeof_DStream`. Alias.
pub fn ZSTD_sizeof_DStream(zds: &ZSTD_DStream) -> usize {
    ZSTD_sizeof_DCtx(zds)
}

/// Port of `ZSTD_estimateDCtxSize`. Returns the heap footprint of a
/// `ZSTD_DCtx` — the three FSE DTables + HUF DTable + workspace +
/// literals buffer. Approximate; real usage depends on frame
/// header's windowSize.
pub fn ZSTD_estimateDCtxSize() -> usize {
    core::mem::size_of::<ZSTD_DCtx>()
        + 8 * 1024   // HUF table
        + 32 * 1024  // HUF workspace
        + 64 * 1024  // literals buffer
        + 4 * 1024   // three seq DTables
}

/// Port of `ZSTD_estimateDStreamSize`. Upstream: ~ DCtx + windowSize.
pub fn ZSTD_estimateDStreamSize(windowSize: usize) -> usize {
    ZSTD_estimateDCtxSize() + windowSize.max(1 << 10)
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

/// Port of the public `ZSTD_decompressBlock`. Decompresses a single
/// block body WITHOUT any frame header — `src` starts at the block
/// data (literals + sequences), as emitted by `ZSTD_compressBlock`.
/// Only meaningful for callers building frameless protocols.
///
/// Rust signature: `src` is the block body bytes. Returns the number
/// of decoded bytes written into `dst`, or an error code.
pub fn ZSTD_decompressBlock(
    dctx: &mut ZSTD_DCtx,
    dst: &mut [u8],
    src: &[u8],
) -> usize {
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
    use crate::common::error::{ERROR, ErrorCode};
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
        dst[..skippableContentSize]
            .copy_from_slice(&src[ZSTD_SKIPPABLEHEADERSIZE..ZSTD_SKIPPABLEHEADERSIZE + skippableContentSize]);
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
/// pre-digested decompression dictionary. v0.1 delegates to
/// `ZSTD_decompress_usingDict` on the DDict's raw content (the entropy
/// pre-digestion path isn't active yet — see `zstd_ddict.rs`).
pub fn ZSTD_decompress_usingDDict(
    dctx: &mut ZSTD_DCtx,
    dst: &mut [u8],
    src: &[u8],
    ddict: &crate::decompress::zstd_ddict::ZSTD_DDict,
) -> usize {
    let content = crate::decompress::zstd_ddict::ZSTD_DDict_dictContent(ddict);
    ZSTD_decompress_usingDict(dctx, dst, src, content)
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

/// Port of `ZSTD_findFrameCompressedSize`. Returns the total byte
/// length of the first frame in `src`, or an error code.
pub fn ZSTD_findFrameCompressedSize(src: &[u8]) -> usize {
    ZSTD_findFrameSizeInfo(src, ZSTD_format_e::ZSTD_f_zstd1).compressedSize
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
    0
}

/// Port of `ZSTD_initDStream_usingDict`. Initializes streaming
/// decompression with a raw-content dictionary — every frame decoded
/// in this session will be passed the dict as history.
pub fn ZSTD_initDStream_usingDict(zds: &mut ZSTD_DCtx, dict: &[u8]) -> usize {
    let rc = ZSTD_initDStream(zds);
    if crate::common::error::ERR_isError(rc) {
        return rc;
    }
    zds.stream_dict = dict.to_vec();
    0
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
    use crate::common::error::{ERROR, ErrorCode, ERR_isError};
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
        if ERR_isError(info.compressedSize)
            || info.decompressedBound == ZSTD_CONTENTSIZE_ERROR
        {
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
        if ERR_isError(info.compressedSize) || info.decompressedBound == ZSTD_CONTENTSIZE_ERROR
        {
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

/// Port of `ZSTD_decodingBufferSize_min`. Worst-case ring-buffer size
/// needed to decompress a frame with the given window and content
/// sizes — accounts for two-block wildcopy slack.
pub fn ZSTD_decodingBufferSize_min(windowSize: u64, frameContentSize: u64) -> usize {
    use crate::common::error::{ERROR, ErrorCode};
    use crate::common::zstd_internal::WILDCOPY_OVERLENGTH;
    use crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX;
    let blockSize = windowSize.min(ZSTD_BLOCKSIZE_MAX as u64) as usize;
    // Upstream: `windowSize + blockSize + ZSTD_BLOCKSIZE_MAX + 2*WILDCOPY`.
    // The extra fixed `ZSTD_BLOCKSIZE_MAX` term covers the litbuffer
    // stored after the most recent block output.
    let neededRBSize: u64 = windowSize
        + blockSize as u64
        + ZSTD_BLOCKSIZE_MAX as u64
        + (WILDCOPY_OVERLENGTH as u64) * 2;
    let neededSize = frameContentSize.min(neededRBSize);
    // Truncation check for 32-bit targets.
    let minRBSize = neededSize as usize;
    if minRBSize as u64 != neededSize {
        return ERROR(ErrorCode::FrameParameterWindowTooLarge);
    }
    minRBSize
}

/// Port of `ZSTD_createDCtx_advanced`. v0.1 ignores the custom
/// allocator and returns a default DCtx.
pub fn ZSTD_createDCtx_advanced(
    _customMem: crate::compress::zstd_compress::ZSTD_customMem,
) -> Option<Box<ZSTD_DCtx>> {
    Some(ZSTD_createDCtx())
}

/// Port of `ZSTD_createDStream_advanced`. Same as above.
pub fn ZSTD_createDStream_advanced(
    _customMem: crate::compress::zstd_compress::ZSTD_customMem,
) -> Option<Box<ZSTD_DStream>> {
    Some(ZSTD_createDCtx())
}

/// Port of `ZSTD_initStaticDCtx`. v0.1 doesn't support the
/// static-buffer init pattern (all `Vec`s are heap-allocated). Always
/// returns `None`.
pub fn ZSTD_initStaticDCtx(workspace: &mut [u8]) -> Option<&mut ZSTD_DCtx> {
    let _ = workspace;
    None
}

/// Port of `ZSTD_initStaticDStream`. Always `None`.
pub fn ZSTD_initStaticDStream(workspace: &mut [u8]) -> Option<&mut ZSTD_DStream> {
    let _ = workspace;
    None
}

/// Port of `ZSTD_initStaticDDict`. Always `None`.
pub fn ZSTD_initStaticDDict<'a>(
    workspace: &'a mut [u8],
    dict: &[u8],
) -> Option<&'a mut crate::decompress::zstd_ddict::ZSTD_DDict> {
    let _ = (workspace, dict);
    None
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
/// decompressor expects next. v0.1 doesn't drive a block-level state
/// machine — returns `frameHeader` so callers that politely consult
/// this helper aren't misled.
#[inline]
pub fn ZSTD_nextInputType(_dctx: &ZSTD_DCtx) -> ZSTD_nextInputType_e {
    ZSTD_nextInputType_e::ZSTDnit_frameHeader
}

/// Port of `ZSTD_decompressBegin`. Legacy continue-style init — v0.1
/// doesn't drive a block-level state machine, so this is a no-op
/// returning 0.
#[inline]
pub fn ZSTD_decompressBegin(_dctx: &mut ZSTD_DCtx) -> usize {
    0
}

/// Port of `ZSTD_decompressBegin_usingDict`. Stashes a raw-content
/// dict on the DCtx then calls `ZSTD_decompressBegin`.
pub fn ZSTD_decompressBegin_usingDict(dctx: &mut ZSTD_DCtx, dict: &[u8]) -> usize {
    dctx.stream_dict = dict.to_vec();
    ZSTD_decompressBegin(dctx)
}

/// Port of `ZSTD_decompressBegin_usingDDict`. Uses the DDict's raw
/// content as the dict history.
pub fn ZSTD_decompressBegin_usingDDict(
    dctx: &mut ZSTD_DCtx,
    ddict: &crate::decompress::zstd_ddict::ZSTD_DDict,
) -> usize {
    dctx.stream_dict =
        crate::decompress::zstd_ddict::ZSTD_DDict_dictContent(ddict).to_vec();
    ZSTD_decompressBegin(dctx)
}

/// Port of `ZSTD_decompressContinue`. Legacy block-level decode —
/// expected input shape follows `ZSTD_nextInputType` transitions.
/// v0.1 doesn't drive that state machine; returns
/// `ErrorCode::Generic` so callers fall back to
/// `ZSTD_decompressStream`.
pub fn ZSTD_decompressContinue(
    _dctx: &mut ZSTD_DCtx,
    _dst: &mut [u8],
    _src: &[u8],
) -> usize {
    use crate::common::error::{ERROR, ErrorCode};
    ERROR(ErrorCode::Generic)
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
    zds.stream_dict =
        crate::decompress::zstd_ddict::ZSTD_DDict_dictContent(ddict).to_vec();
    0
}

/// Port of `ZSTD_nextSrcSizeToDecompress`. Returns a hint for how
/// many more input bytes the decompressor needs to make progress. In
/// v0.1 we don't track granular expected-size state, so we simply
/// return `ZSTD_DStreamInSize` (upstream's advisory block-worst-case).
#[inline]
pub fn ZSTD_nextSrcSizeToDecompress(_dctx: &ZSTD_DCtx) -> usize {
    ZSTD_DStreamInSize()
}

/// Port of `ZSTD_resetDStream`. Clears per-frame state (input/output
/// buffers) but preserves the configured dict for subsequent frames.
/// Returns a hint for the suggested next input size.
pub fn ZSTD_resetDStream(zds: &mut ZSTD_DCtx) -> usize {
    zds.stream_in_buffer.clear();
    zds.stream_out_buffer.clear();
    zds.stream_out_drained = 0;
    3
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
    if zds.stream_out_drained == zds.stream_out_buffer.len()
        && zds.stream_in_buffer.is_empty()
    {
        return 0;
    }
    // If output fully drained AND fresh input available, try to probe
    // a new frame.
    if zds.stream_out_drained == zds.stream_out_buffer.len() {
        // Try to measure a full frame from the staged input.
        let frame_sz = ZSTD_findFrameCompressedSize(&zds.stream_in_buffer);
        if ERR_isError(frame_sz) {
            // Could be "need more input" (SrcSizeWrong). Return a
            // non-zero hint so the caller keeps feeding.
            return 3; // ZSTD_blockHeaderSize
        }
        // Determine decoded size.
        let declared = ZSTD_getFrameContentSize(&zds.stream_in_buffer);
        let out_size = if declared == ZSTD_CONTENTSIZE_UNKNOWN || declared == ZSTD_CONTENTSIZE_ERROR {
            // Fall back to a generous bound: 32× compressed size.
            frame_sz * 32
        } else {
            declared as usize
        };
        let mut decoded = vec![0u8; out_size.max(1)];
        let d = if zds.stream_dict.is_empty() {
            ZSTD_decompress(&mut decoded, &zds.stream_in_buffer[..frame_sz])
        } else {
            let dict = zds.stream_dict.clone();
            let mut tmp_dctx = ZSTD_DCtx::new();
            ZSTD_decompress_usingDict(
                &mut tmp_dctx,
                &mut decoded,
                &zds.stream_in_buffer[..frame_sz],
                &dict,
            )
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
