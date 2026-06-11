//! Translation of `lib/decompress/zstd_decompress_block.c`.
//!
//! This file is the core of the decoder: literals decoding (raw / RLE /
//! HUF-compressed) and sequence execution. The full `ZSTD_DCtx` struct
//! (defined in `zstd_decompress_internal.h`) is ported here, along
//! with the FSE seq-table builders, bitstream state, and the full
//! `ZSTD_decompressBlock_internal` pipeline — every fixture roundtrip,
//! CLI roundtrip, and magicless/refPrefix lifecycle test exercises
//! this module.
//!
//! Types + helpers: block-header (`blockType_e`, `blockProperties_t`),
//! `ZSTD_getcBlockSize`, `ZSTD_copy4` / `ZSTD_copy8`, `ZSTD_overlapCopy8`
//! (overlap-to-8 spreader), sequence-decoder value types (`seq_t`,
//! `ZSTD_fseState`).
//!
//! Entry points: `ZSTD_decodeLiteralsBlock`, `ZSTD_decodeSeqHeaders`,
//! `ZSTD_buildSeqTable`, `ZSTD_decompressBlock_internal`,
//! `ZSTD_decompressBlock`.

use crate::common::error::{ErrorCode, ERROR};
use crate::common::mem::{MEM_readLE16, MEM_readLE24, MEM_readLE32};
use crate::decompress::huf_decompress::{
    HUF_DTable, HUF_decompress1X1_DCtx_wksp, HUF_decompress1X_usingDTable,
    HUF_decompress4X_hufOnly_wksp, HUF_decompress4X_usingDTable, HUF_flags_bmi2,
    HUF_flags_disableAsm, HUF_DTABLE_SIZE_U32, HUF_TABLELOG_MAX,
};

/// Mirror of upstream `blockType_e`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum blockType_e {
    bt_raw = 0,
    bt_rle = 1,
    bt_compressed = 2,
    bt_reserved = 3,
}

impl blockType_e {
    fn from_bits(bits: u32) -> Self {
        match bits & 3 {
            0 => Self::bt_raw,
            1 => Self::bt_rle,
            2 => Self::bt_compressed,
            _ => Self::bt_reserved,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct blockProperties_t {
    pub blockType: blockType_e,
    pub lastBlock: u32,
    pub origSize: u32,
}

pub const ZSTD_BLOCKHEADERSIZE: usize = 3;
pub const ZSTD_blockHeaderSize: usize = ZSTD_BLOCKHEADERSIZE;
pub const ZSTD_BLOCKSIZELOG_MAX: u32 = 17;
pub const ZSTD_BLOCKSIZE_MAX: usize = 1 << ZSTD_BLOCKSIZELOG_MAX;
pub const ZSTD_REP_NUM: usize = 3;
pub const MIN_CBLOCK_SIZE: usize = 2;
pub use crate::common::zstd_internal::{WILDCOPY_OVERLENGTH, WILDCOPY_VECLEN};
pub const ZSTD_LITBUFFEREXTRASIZE: usize = 65536; // BLOCKSIZE_MAX / 2
const STREAM_ACCUMULATOR_MIN_32: u32 = 25;
const LONG_OFFSETS_MAX_EXTRA_BITS_32: u32 = 5;

/// Port of `ZSTD_longOffset_e` (`zstd_decompress_block.c:1227`). Tells
/// the sequence-decode inner loop whether to take the long-offset
/// path (extra reload of the bitstream accumulator before reading
/// offset extra bits). Relevant only on 32-bit targets where the
/// accumulator is 25 bits wide; on 64-bit, the regular path handles
/// every legal offset without a split.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZSTD_longOffset_e {
    #[default]
    ZSTD_lo_isRegularOffset = 0,
    ZSTD_lo_isLongOffset = 1,
}

/// Port of `ZSTD_checkContinuity` (`zstd_decompress_block.c:2280`).
/// Rotates the DCtx's history-segment pointer-equivalents when the
/// caller switches to a non-contiguous destination buffer.
#[inline]
pub fn ZSTD_checkContinuity(dctx: &mut ZSTD_DCtx, dst: &[u8], dstSize: usize) {
    if dstSize == 0 {
        return;
    }
    let dst_start = dst.as_ptr() as usize;
    if Some(dst_start) != dctx.previousDstEnd {
        dctx.dictEnd = dctx.previousDstEnd;
        let previous_end = dctx.previousDstEnd.unwrap_or(dst_start);
        let prefix_start = dctx.prefixStart.unwrap_or(previous_end);
        dctx.virtualStart = Some(dst_start.wrapping_sub(previous_end.wrapping_sub(prefix_start)));
        dctx.prefixStart = Some(dst_start);
        dctx.previousDstEnd = Some(dst_start);
    }
}

/// Port of `ZSTD_DCtx_get_bmi2` (`zstd_decompress_internal.h:213`).
/// Reads back the DCtx's `bmi2` detection flag. Upstream guards this
/// behind `DYNAMIC_BMI2`; our port always returns the stored flag
/// (which is `0` in v0.1 — we don't ship a BMI2 fast path yet).
#[inline]
pub fn ZSTD_DCtx_get_bmi2(dctx: &ZSTD_DCtx) -> i32 {
    dctx.bmi2
}

/// Port of `ZSTD_totalHistorySize` (`zstd_decompress_block.c:2098`).
/// Size of the decompressed history available before `curPtr` —
/// expressed as bytes between a virtual start and the current write
/// cursor. Upstream's formula is pointer-subtraction; our port mirrors
/// it with index-based inputs so callers stay in safe Rust.
#[inline]
pub fn ZSTD_totalHistorySize(curIdx: usize, virtualStartIdx: usize) -> usize {
    curIdx - virtualStartIdx
}

/// Port of `ZSTD_maxShortOffset` (`zstd_decompress_block.c:2147`).
/// Largest offset that can be decoded in a single bitstream read
/// without mid-offset refill. On 64-bit targets the 57-bit accumulator
/// can hold any legal zstd offset (windowLog ≤ 31), so the cap is
/// `usize::MAX`. On 32-bit, the 25-bit accumulator caps `offBase` at
/// `(1 << 26) - 1`; subtract `ZSTD_REP_NUM` to convert to a raw offset.
#[inline]
pub fn ZSTD_maxShortOffset() -> usize {
    use crate::common::mem::MEM_64bits;
    if MEM_64bits() != 0 {
        usize::MAX
    } else {
        let stream_accumulator_min: u32 = 25;
        let maxOffbase: usize = (1usize << (stream_accumulator_min + 1)) - 1;
        maxOffbase - ZSTD_REP_NUM
    }
}

/// Port of `ZSTD_safecopyDstBeforeSrc` (`zstd_decompress_block.c:883`).
/// Copies `length` bytes starting at `src_idx` in `buf` to `dst_idx`,
/// where upstream's contract says `dst` comes *before* `src` in memory
/// (no aliasing concerns for a forward copy, but short overlaps are
/// still special-cased). Upstream mixes the short-length fallback with
/// a wildcopy fast-path when `dst` and `src` are far apart; the Rust
/// port uses a byte-by-byte forward copy since the aliasing-safety of
/// our wildcopy helpers is already centralized elsewhere.
///
/// Operates on a single mutable buffer with two indices to stay safe
/// in the face of borrow-checker constraints.
pub fn ZSTD_safecopyDstBeforeSrc(buf: &mut [u8], dst_idx: usize, src_idx: usize, length: usize) {
    debug_assert!(dst_idx <= src_idx || src_idx + length <= dst_idx);
    for i in 0..length {
        buf[dst_idx + i] = buf[src_idx + i];
    }
}

/// Port of `ZSTD_safecopy`. Copies `length` bytes inside one buffer,
/// selecting the overlap-safe byte-copy path for short tails.
pub fn ZSTD_safecopy(
    buf: &mut [u8],
    dst_idx: usize,
    src_idx: usize,
    length: usize,
    ovtype: crate::common::zstd_internal::ZSTD_overlap_e,
) {
    if length == 0 {
        return;
    }
    if ovtype == crate::common::zstd_internal::ZSTD_overlap_e::ZSTD_overlap_src_before_dst
        && src_idx < dst_idx
        && src_idx + length > dst_idx
    {
        for i in 0..length {
            buf[dst_idx + i] = buf[src_idx + i];
        }
    } else {
        buf.copy_within(src_idx..src_idx + length, dst_idx);
    }
}

/// Port of `ZSTD_getcBlockSize`. Reads the 3-byte block header from
/// `src` and fills `bpPtr`. Returns the compressed block size (except
/// for `bt_rle`, where it returns 1 — one source byte holds the value
/// to repeat).
pub fn ZSTD_getcBlockSize(src: &[u8], bpPtr: &mut blockProperties_t) -> usize {
    if src.len() < ZSTD_blockHeaderSize {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    let hdr = MEM_readLE24(&src[..3]);
    let cSize = (hdr >> 3) as usize;
    bpPtr.lastBlock = hdr & 1;
    bpPtr.blockType = blockType_e::from_bits(hdr >> 1);
    bpPtr.origSize = cSize as u32;
    if bpPtr.blockType == blockType_e::bt_rle {
        return 1;
    }
    if bpPtr.blockType == blockType_e::bt_reserved {
        return ERROR(ErrorCode::CorruptionDetected);
    }
    cSize
}

/// Port of `ZSTD_copy4`: 4-byte memcpy.
#[inline]
pub fn ZSTD_copy4(dst: &mut [u8], src: &[u8]) {
    dst[..4].copy_from_slice(&src[..4]);
}

/// Port of `ZSTD_copy8`: 8-byte memcpy.
#[inline]
pub fn ZSTD_copy8(dst: &mut [u8], src: &[u8]) {
    dst[..8].copy_from_slice(&src[..8]);
}

/// Port of `ZSTD_overlapCopy8`. Copies 8 bytes from `buf[ip..]` to
/// `buf[op..]`, handling the case where `offset < 8` (self-overlap, so
/// we have to "spread" the pattern until offset >= 8). Advances `*op`
/// and `*ip` by 8.
///
/// Upstream uses pointer pairs into a single buffer; the Rust port
/// takes one `&mut [u8]` plus two indices to keep the aliasing
/// single-borrow.
#[inline(always)]
pub fn ZSTD_overlapCopy8(buf: &mut [u8], op: &mut usize, ip: &mut usize, offset: usize) {
    debug_assert!(*ip <= *op);
    debug_assert!(*op + 8 <= buf.len());
    if offset < 8 {
        // Spread the low bytes until op-ip >= 8. Upstream runs
        // `size_t` arithmetic that can transiently underflow the
        // source pointer; we use `wrapping_{add,sub}` for identical
        // modular behaviour. `DEC32[offset] + (8 - DEC64[offset])` is
        // always positive, so the net delta applied to `ip` is safe.
        const DEC32: [usize; 8] = [0, 1, 2, 1, 4, 4, 4, 4];
        const DEC64: [usize; 8] = [8, 8, 8, 7, 8, 9, 10, 11];
        let sub2 = DEC64[offset];
        unsafe {
            let ptr = buf.as_mut_ptr();
            let op0 = *op;
            let ip0 = *ip;
            *ptr.add(op0) = *ptr.add(ip0);
            *ptr.add(op0 + 1) = *ptr.add(ip0 + 1);
            *ptr.add(op0 + 2) = *ptr.add(ip0 + 2);
            *ptr.add(op0 + 3) = *ptr.add(ip0 + 3);
            *ip = ip.wrapping_add(DEC32[offset]);
            let ip1 = *ip;
            let s0 = *ptr.add(ip1);
            let s1 = *ptr.add(ip1 + 1);
            let s2 = *ptr.add(ip1 + 2);
            let s3 = *ptr.add(ip1 + 3);
            *ptr.add(op0 + 4) = s0;
            *ptr.add(op0 + 5) = s1;
            *ptr.add(op0 + 6) = s2;
            *ptr.add(op0 + 7) = s3;
            *ip = ip.wrapping_sub(sub2);
        }
    } else {
        // Non-overlapping 8-byte copy.
        unsafe {
            let ptr = buf.as_mut_ptr();
            let v = (ptr.add(*ip) as *const [u8; 8]).read_unaligned();
            (ptr.add(*op) as *mut [u8; 8]).write_unaligned(v);
        }
    }
    *ip = ip.wrapping_add(8);
    *op += 8;
}

// ---- Sequence-decoder value types -------------------------------------

/// Mirror of upstream `seq_t`: one decoded (litLength, matchLength,
/// offset) triplet.
#[derive(Debug, Clone, Copy, Default)]
pub struct seq_t {
    pub litLength: usize,
    pub matchLength: usize,
    pub offset: usize,
}

/// Mirror of upstream `ZSTD_fseState` (the `table` pointer is resolved
/// to an index offset into the caller's table array to keep borrows
/// lifetime-free).
#[derive(Debug, Clone, Copy, Default)]
pub struct ZSTD_fseState {
    pub state: usize,
    pub table_offset: usize,
}

/// Port of `ZSTD_initFseState`. Reads `tableLog` bits from `bitD`, then
/// reloads. The table is stored in the caller's slice — we record an
/// offset of 1 (header sits at slot 0).
pub fn ZSTD_initFseState(
    dsp: &mut ZSTD_fseState,
    bitD: &mut crate::common::bitstream::BIT_DStream_t,
    dt: &[ZSTD_seqSymbol],
) {
    let header = seq_header_read(dt);
    dsp.state = crate::common::bitstream::BIT_readBits(bitD, header.tableLog);
    crate::common::bitstream::BIT_reloadDStream(bitD);
    dsp.table_offset = 1;
}

/// Port of `ZSTD_updateFseStateWithDInfo`.
#[inline(always)]
pub fn ZSTD_updateFseStateWithDInfo(
    dsp: &mut ZSTD_fseState,
    bitD: &mut crate::common::bitstream::BIT_DStream_t,
    nextState: u16,
    nbBits: u32,
) {
    let lowBits = crate::common::bitstream::BIT_readBits(bitD, nbBits);
    dsp.state = nextState as usize + lowBits;
}

/// BMI2 variant of `BIT_readBitsFast` used by the BMI2-targeted
/// sequence decoder. Reads `nbBits` from the container, then skips.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[target_feature(enable = "bmi1,bmi2,lzcnt")]
unsafe fn BIT_readBitsFast_bmi2(
    bitD: &mut crate::common::bitstream::BIT_DStream_t,
    nbBits: u32,
) -> usize {
    debug_assert!(nbBits >= 1);
    let reg_mask = 63u32;
    let v = ((bitD.bitContainer as u64) << (bitD.bitsConsumed & reg_mask))
        >> ((64u32 - nbBits) & reg_mask);
    crate::common::bitstream::BIT_skipBits(bitD, nbBits);
    v as usize
}

/// BMI2 variant of `BIT_readBits` used by the BMI2-targeted FSE state
/// update path. Uses `BEXTR` when the request fits in the container.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[target_feature(enable = "bmi1,bmi2,lzcnt")]
unsafe fn BIT_readBits_bmi2(
    bitD: &mut crate::common::bitstream::BIT_DStream_t,
    nbBits: u32,
) -> usize {
    let v = if bitD.bitsConsumed + nbBits <= 64 {
        let start = 64 - bitD.bitsConsumed - nbBits;
        core::arch::x86_64::_bextr_u64(bitD.bitContainer as u64, start, nbBits) as usize
    } else {
        crate::common::bitstream::BIT_lookBits(bitD, nbBits)
    };
    crate::common::bitstream::BIT_skipBits(bitD, nbBits);
    v
}

/// BMI2 specialization of `ZSTD_updateFseStateWithDInfo`.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[target_feature(enable = "bmi1,bmi2,lzcnt")]
unsafe fn ZSTD_updateFseStateWithDInfo_bmi2(
    dsp: &mut ZSTD_fseState,
    bitD: &mut crate::common::bitstream::BIT_DStream_t,
    nextState: u16,
    nbBits: u32,
) {
    let lowBits = BIT_readBits_bmi2(bitD, nbBits);
    dsp.state = nextState as usize + lowBits;
}

/// Holds the three FSE states plus the rolling repcode offsets used by
/// the sequence decoder. `'a` ties the bit-stream to its source slice.
pub struct seqState_t<'a> {
    pub DStream: crate::common::bitstream::BIT_DStream_t<'a>,
    pub stateLL: ZSTD_fseState,
    pub stateOffb: ZSTD_fseState,
    pub stateML: ZSTD_fseState,
    pub prevOffset: [usize; ZSTD_REP_NUM],
}

/// Port of `ZSTD_decodeSequence` (portable path — no aarch64-specific
/// local copy). Decodes one sequence triplet (litLength, matchLength,
/// offset), updates repcodes, and (when not the last sequence) advances
/// the FSE states.
#[inline]
pub fn ZSTD_decodeSequence(
    seqState: &mut seqState_t,
    longOffsets: ZSTD_longOffset_e,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    isLastSeq: bool,
) -> seq_t {
    use crate::common::bitstream::{BIT_readBitsFast, BIT_reloadDStream};

    let ll_idx = seqState.stateLL.table_offset + seqState.stateLL.state;
    let ml_idx = seqState.stateML.table_offset + seqState.stateML.state;
    let of_idx = seqState.stateOffb.table_offset + seqState.stateOffb.state;
    debug_assert!(ll_idx < LLTable.len());
    debug_assert!(ml_idx < MLTable.len());
    debug_assert!(of_idx < OFTable.len());
    let llDInfo = unsafe { *LLTable.get_unchecked(ll_idx) };
    let mlDInfo = unsafe { *MLTable.get_unchecked(ml_idx) };
    let ofDInfo = unsafe { *OFTable.get_unchecked(of_idx) };

    let mut seq = seq_t {
        litLength: llDInfo.baseValue as usize,
        matchLength: mlDInfo.baseValue as usize,
        offset: 0,
    };

    let ofBase = ofDInfo.baseValue as usize;
    let llBits = llDInfo.nbAdditionalBits;
    let mlBits = mlDInfo.nbAdditionalBits;
    let ofBits = ofDInfo.nbAdditionalBits;
    let totalBits = llBits as u32 + mlBits as u32 + ofBits as u32;

    // --- decode offset (with repcode handling) ---
    let mut prev0 = seqState.prevOffset[0];
    let mut prev1 = seqState.prevOffset[1];
    let mut prev2 = seqState.prevOffset[2];
    let mut offset;
    if ofBits > 1 {
        if crate::common::mem::MEM_32bits() != 0
            && longOffsets == ZSTD_longOffset_e::ZSTD_lo_isLongOffset
            && (ofBits as u32) >= STREAM_ACCUMULATOR_MIN_32
        {
            let extraBits = LONG_OFFSETS_MAX_EXTRA_BITS_32;
            offset = ofBase
                + (BIT_readBitsFast(&mut seqState.DStream, ofBits as u32 - extraBits) << extraBits);
            BIT_reloadDStream(&mut seqState.DStream);
            offset += BIT_readBitsFast(&mut seqState.DStream, extraBits);
        } else {
            offset = ofBase + BIT_readBitsFast(&mut seqState.DStream, ofBits as u32);
            if crate::common::mem::MEM_32bits() != 0 {
                BIT_reloadDStream(&mut seqState.DStream);
            }
        }
        prev2 = prev1;
        prev1 = prev0;
        prev0 = offset;
    } else {
        let ll0 = (llDInfo.baseValue == 0) as usize;
        if ofBits == 0 {
            if ll0 != 0 {
                offset = prev1;
                prev1 = prev0;
                prev0 = offset;
            } else {
                offset = prev0;
            }
        } else {
            let raw = ofBase + ll0 + BIT_readBitsFast(&mut seqState.DStream, 1);
            let mut temp = match raw {
                1 => prev1,
                3 => prev0.wrapping_sub(1),
                r if r >= 2 => prev2,
                _ => prev0,
            };
            if temp == 0 {
                temp = temp.wrapping_sub(1);
            }
            prev2 = if raw == 1 { prev2 } else { prev1 };
            prev1 = prev0;
            prev0 = temp;
            offset = temp;
        }
    }
    seq.offset = offset;

    if mlBits > 0 {
        seq.matchLength += BIT_readBitsFast(&mut seqState.DStream, mlBits as u32);
    }

    if crate::common::mem::MEM_32bits() != 0
        && mlBits as u32 + llBits as u32
            >= STREAM_ACCUMULATOR_MIN_32 - LONG_OFFSETS_MAX_EXTRA_BITS_32
    {
        BIT_reloadDStream(&mut seqState.DStream);
    }
    if crate::common::mem::MEM_64bits() != 0
        && totalBits
            >= crate::common::bitstream::BIT_DStream_unfinished + 57
                - (LLFSELog + MLFSELog + OffFSELog)
    {
        // Upstream reloads when the remaining bit budget gets tight.
        // Condition is a static macro; the portable-path check below
        // tracks that effect.
        BIT_reloadDStream(&mut seqState.DStream);
    }

    if llBits > 0 {
        seq.litLength += BIT_readBitsFast(&mut seqState.DStream, llBits as u32);
    }
    if crate::common::mem::MEM_32bits() != 0 {
        BIT_reloadDStream(&mut seqState.DStream);
    }

    if !isLastSeq {
        ZSTD_updateFseStateWithDInfo(
            &mut seqState.stateLL,
            &mut seqState.DStream,
            llDInfo.nextState,
            llDInfo.nbBits as u32,
        );
        ZSTD_updateFseStateWithDInfo(
            &mut seqState.stateML,
            &mut seqState.DStream,
            mlDInfo.nextState,
            mlDInfo.nbBits as u32,
        );
        if crate::common::mem::MEM_32bits() != 0 {
            BIT_reloadDStream(&mut seqState.DStream);
        }
        ZSTD_updateFseStateWithDInfo(
            &mut seqState.stateOffb,
            &mut seqState.DStream,
            ofDInfo.nextState,
            ofDInfo.nbBits as u32,
        );
        BIT_reloadDStream(&mut seqState.DStream);
    }

    seqState.prevOffset[0] = prev0;
    seqState.prevOffset[1] = prev1;
    seqState.prevOffset[2] = prev2;
    seq
}

/// BMI2 specialization of `ZSTD_decodeSequence`. Same control flow as
/// the portable path, with `BIT_readBitsFast_bmi2` substituted in.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[target_feature(enable = "bmi1,bmi2,lzcnt")]
unsafe fn ZSTD_decodeSequence_bmi2(
    seqState: &mut seqState_t,
    longOffsets: ZSTD_longOffset_e,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    isLastSeq: bool,
) -> seq_t {
    use crate::common::bitstream::BIT_reloadDStream;

    let ll_idx = seqState.stateLL.table_offset + seqState.stateLL.state;
    let ml_idx = seqState.stateML.table_offset + seqState.stateML.state;
    let of_idx = seqState.stateOffb.table_offset + seqState.stateOffb.state;
    debug_assert!(ll_idx < LLTable.len());
    debug_assert!(ml_idx < MLTable.len());
    debug_assert!(of_idx < OFTable.len());
    let llDInfo = *LLTable.get_unchecked(ll_idx);
    let mlDInfo = *MLTable.get_unchecked(ml_idx);
    let ofDInfo = *OFTable.get_unchecked(of_idx);

    let mut seq = seq_t {
        litLength: llDInfo.baseValue as usize,
        matchLength: mlDInfo.baseValue as usize,
        offset: 0,
    };

    let ofBase = ofDInfo.baseValue as usize;
    let llBits = llDInfo.nbAdditionalBits;
    let mlBits = mlDInfo.nbAdditionalBits;
    let ofBits = ofDInfo.nbAdditionalBits;
    let totalBits = llBits as u32 + mlBits as u32 + ofBits as u32;

    let mut prev0 = seqState.prevOffset[0];
    let mut prev1 = seqState.prevOffset[1];
    let mut prev2 = seqState.prevOffset[2];
    let mut offset;
    if ofBits > 1 {
        if crate::common::mem::MEM_32bits() != 0
            && longOffsets == ZSTD_longOffset_e::ZSTD_lo_isLongOffset
            && (ofBits as u32) >= STREAM_ACCUMULATOR_MIN_32
        {
            let extraBits = LONG_OFFSETS_MAX_EXTRA_BITS_32;
            offset = ofBase
                + (BIT_readBitsFast_bmi2(&mut seqState.DStream, ofBits as u32 - extraBits)
                    << extraBits);
            BIT_reloadDStream(&mut seqState.DStream);
            offset += BIT_readBitsFast_bmi2(&mut seqState.DStream, extraBits);
        } else {
            offset = ofBase + BIT_readBitsFast_bmi2(&mut seqState.DStream, ofBits as u32);
            if crate::common::mem::MEM_32bits() != 0 {
                BIT_reloadDStream(&mut seqState.DStream);
            }
        }
        prev2 = prev1;
        prev1 = prev0;
        prev0 = offset;
    } else {
        let ll0 = (llDInfo.baseValue == 0) as usize;
        if ofBits == 0 {
            if ll0 != 0 {
                offset = prev1;
                prev1 = prev0;
                prev0 = offset;
            } else {
                offset = prev0;
            }
        } else {
            let raw = ofBase + ll0 + BIT_readBitsFast_bmi2(&mut seqState.DStream, 1);
            let mut temp = match raw {
                1 => prev1,
                3 => prev0.wrapping_sub(1),
                r if r >= 2 => prev2,
                _ => prev0,
            };
            if temp == 0 {
                temp = temp.wrapping_sub(1);
            }
            prev2 = if raw == 1 { prev2 } else { prev1 };
            prev1 = prev0;
            prev0 = temp;
            offset = temp;
        }
    }
    seq.offset = offset;

    if mlBits > 0 {
        seq.matchLength += BIT_readBitsFast_bmi2(&mut seqState.DStream, mlBits as u32);
    }

    if crate::common::mem::MEM_32bits() != 0
        && mlBits as u32 + llBits as u32
            >= STREAM_ACCUMULATOR_MIN_32 - LONG_OFFSETS_MAX_EXTRA_BITS_32
    {
        BIT_reloadDStream(&mut seqState.DStream);
    }
    if crate::common::mem::MEM_64bits() != 0
        && totalBits
            >= crate::common::bitstream::BIT_DStream_unfinished + 57
                - (LLFSELog + MLFSELog + OffFSELog)
    {
        BIT_reloadDStream(&mut seqState.DStream);
    }

    if llBits > 0 {
        seq.litLength += BIT_readBitsFast_bmi2(&mut seqState.DStream, llBits as u32);
    }
    if crate::common::mem::MEM_32bits() != 0 {
        BIT_reloadDStream(&mut seqState.DStream);
    }

    if !isLastSeq {
        ZSTD_updateFseStateWithDInfo_bmi2(
            &mut seqState.stateLL,
            &mut seqState.DStream,
            llDInfo.nextState,
            llDInfo.nbBits as u32,
        );
        ZSTD_updateFseStateWithDInfo_bmi2(
            &mut seqState.stateML,
            &mut seqState.DStream,
            mlDInfo.nextState,
            mlDInfo.nbBits as u32,
        );
        if crate::common::mem::MEM_32bits() != 0 {
            BIT_reloadDStream(&mut seqState.DStream);
        }
        ZSTD_updateFseStateWithDInfo_bmi2(
            &mut seqState.stateOffb,
            &mut seqState.DStream,
            ofDInfo.nextState,
            ofDInfo.nbBits as u32,
        );
        BIT_reloadDStream(&mut seqState.DStream);
    }

    seqState.prevOffset[0] = prev0;
    seqState.prevOffset[1] = prev1;
    seqState.prevOffset[2] = prev2;
    seq
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
macro_rules! ZSTD_decodeSequence_bmi2_inline {
    (
        $DStream:expr,
        $stateLL:expr,
        $stateOffb:expr,
        $stateML:expr,
        $prev0:expr,
        $prev1:expr,
        $prev2:expr,
        $LLTablePtr:expr,
        $OFTablePtr:expr,
        $MLTablePtr:expr,
        $longOffsets:expr,
        $isLastSeq:expr
    ) => {{
        use crate::common::bitstream::BIT_reloadDStream;
        macro_rules! update_fse_state {
            ($state:expr, $info:expr) => {{
                let lowBits = BIT_readBits_bmi2(&mut $DStream, $info.nbBits as u32);
                $state.state = $info.nextState as usize + lowBits;
            }};
        }

        let ll_idx = $stateLL.table_offset + $stateLL.state;
        let ml_idx = $stateML.table_offset + $stateML.state;
        let of_idx = $stateOffb.table_offset + $stateOffb.state;
        let llDInfo = *$LLTablePtr.add(ll_idx);
        let mlDInfo = *$MLTablePtr.add(ml_idx);
        let ofDInfo = *$OFTablePtr.add(of_idx);

        let ofBase = ofDInfo.baseValue as usize;
        let llBits = llDInfo.nbAdditionalBits;
        let mlBits = mlDInfo.nbAdditionalBits;
        let ofBits = ofDInfo.nbAdditionalBits;
        let totalBits = llBits as u32 + mlBits as u32 + ofBits as u32;

        let mut litLength = llDInfo.baseValue as usize;
        let mut matchLength = mlDInfo.baseValue as usize;

        let mut offset;
        if ofBits > 1 {
            if crate::common::mem::MEM_32bits() != 0
                && $longOffsets == ZSTD_longOffset_e::ZSTD_lo_isLongOffset
                && (ofBits as u32) >= STREAM_ACCUMULATOR_MIN_32
            {
                let extraBits = LONG_OFFSETS_MAX_EXTRA_BITS_32;
                offset = ofBase
                    + (BIT_readBitsFast_bmi2(&mut $DStream, ofBits as u32 - extraBits)
                        << extraBits);
                BIT_reloadDStream(&mut $DStream);
                offset += BIT_readBitsFast_bmi2(&mut $DStream, extraBits);
            } else {
                offset = ofBase + BIT_readBitsFast_bmi2(&mut $DStream, ofBits as u32);
                if crate::common::mem::MEM_32bits() != 0 {
                    BIT_reloadDStream(&mut $DStream);
                }
            }
            $prev2 = $prev1;
            $prev1 = $prev0;
            $prev0 = offset;
        } else {
            let ll0 = (llDInfo.baseValue == 0) as usize;
            if ofBits == 0 {
                if ll0 != 0 {
                    offset = $prev1;
                    $prev1 = $prev0;
                    $prev0 = offset;
                } else {
                    offset = $prev0;
                }
            } else {
                let raw = ofBase + ll0 + BIT_readBitsFast_bmi2(&mut $DStream, 1);
                let mut temp = match raw {
                    1 => $prev1,
                    3 => $prev0.wrapping_sub(1),
                    r if r >= 2 => $prev2,
                    _ => $prev0,
                };
                if temp == 0 {
                    temp = temp.wrapping_sub(1);
                }
                $prev2 = if raw == 1 { $prev2 } else { $prev1 };
                $prev1 = $prev0;
                $prev0 = temp;
                offset = temp;
            }
        }
        if mlBits > 0 {
            matchLength += BIT_readBitsFast_bmi2(&mut $DStream, mlBits as u32);
        }

        if crate::common::mem::MEM_32bits() != 0
            && mlBits as u32 + llBits as u32
                >= STREAM_ACCUMULATOR_MIN_32 - LONG_OFFSETS_MAX_EXTRA_BITS_32
        {
            BIT_reloadDStream(&mut $DStream);
        }
        if crate::common::mem::MEM_64bits() != 0
            && totalBits
                >= crate::common::bitstream::BIT_DStream_unfinished + 57
                    - (LLFSELog + MLFSELog + OffFSELog)
        {
            BIT_reloadDStream(&mut $DStream);
        }

        if llBits > 0 {
            litLength += BIT_readBitsFast_bmi2(&mut $DStream, llBits as u32);
        }
        if crate::common::mem::MEM_32bits() != 0 {
            BIT_reloadDStream(&mut $DStream);
        }

        if !$isLastSeq {
            update_fse_state!($stateLL, llDInfo);
            update_fse_state!($stateML, mlDInfo);
            if crate::common::mem::MEM_32bits() != 0 {
                BIT_reloadDStream(&mut $DStream);
            }
            update_fse_state!($stateOffb, ofDInfo);
            BIT_reloadDStream(&mut $DStream);
        }

        (litLength, matchLength, offset)
    }};
}

/// Port of `ZSTD_execSequence`. Executes a single (litLength,
/// matchLength, offset) sequence: copies literals from `litBuf[litPtr..]`
/// to `dst[op..]`, then back-references `matchLength` bytes starting
/// `offset` bytes earlier in the virtual stream `[ext_history][dst]`.
/// Returns the total bytes written (`litLength + matchLength`) on
/// success.
///
/// Rust signature note: upstream takes raw `BYTE**` pointers so state
/// can mutate across calls. Here we take &mut slices + indices and
/// return the updated `litPtr_offset` via the `out_litPtr` parameter.
///
/// `ext_history` carries any history bytes that conceptually live
/// *before* `dst[0]` — typically the loaded raw-content dictionary or
/// the prior streaming-block tail — so cross-segment back-references
/// (the upstream `extDict` path) can land. Pass `&[]` when no
/// out-of-buffer history is reachable; one-shot decode paths that
/// concatenate `dict || scratch` into `dst` already have everything
/// in-buffer and can leave `ext_history` empty.
#[inline(always)]
pub fn ZSTD_execSequence(
    dst: &mut [u8],
    op: usize,
    sequence: seq_t,
    ext_history: &[u8],
    litBuf: &[u8],
    litPtr_offset: usize,
    out_litPtr: &mut usize,
) -> usize {
    unsafe {
        ZSTD_execSequence_rawLit(
            dst,
            op,
            sequence,
            ext_history,
            litBuf.as_ptr(),
            litBuf.len(),
            litPtr_offset,
            out_litPtr,
        )
    }
}

/// Rust-only helper: raw-pointer literal-base variant of
/// `ZSTD_execSequence`. Lets the caller hand off the literal buffer
/// as a `(base, total_len)` pair so a single `*const u8` covers both
/// the in-DCtx `litBuffer` and the in-`dst` literal regions used by
/// the split-literal-buffer fast path.
#[inline(always)]
unsafe fn ZSTD_execSequence_rawLit(
    dst: &mut [u8],
    op: usize,
    sequence: seq_t,
    ext_history: &[u8],
    lit_base: *const u8,
    lit_len_total: usize,
    litPtr_offset: usize,
    out_litPtr: &mut usize,
) -> usize {
    let oend = dst.len();
    let oLitEnd = op + sequence.litLength;
    let sequenceLength = sequence.litLength + sequence.matchLength;
    let oMatchEnd = op + sequenceLength;
    if oMatchEnd > oend {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    let iLitEnd = litPtr_offset + sequence.litLength;
    if iLitEnd > lit_len_total {
        return ERROR(ErrorCode::CorruptionDetected);
    }
    // Offset reaches before dst[0] — must come from ext_history.
    // Reject if even the virtual `[ext_history][dst]` view doesn't
    // have enough history.
    if sequence.offset > oLitEnd + ext_history.len() {
        return ERROR(ErrorCode::CorruptionDetected);
    }
    if sequence.matchLength == 0 {
        // Upstream asserts matchLength >= 1; corrupt input otherwise.
        return ERROR(ErrorCode::CorruptionDetected);
    }

    // Copy literals. Hot path: most literal runs are short (avg 5-10
    // bytes for typical text). `copy_from_slice` lowers to a memmove
    // call (~30 cycles call overhead) regardless of length. When we
    // have 16 bytes of slack on both source and destination, do an
    // inline 16-byte unaligned copy via raw pointers — saves the call
    // overhead. This mirrors upstream's `ZSTD_copy16` shortcut at the
    // top of `ZSTD_execSequence`. `litBuf` is always a separate
    // allocation (a Vec clone of the literal block), so the 16-byte
    // write to `dst[op..op+16]` cannot corrupt litBuf even if
    // `lit_len < 16`.
    let lit_len = sequence.litLength;
    if lit_len <= 16 && iLitEnd + 16 <= lit_len_total && oLitEnd + 16 <= oend {
        // SAFETY: bounds verified above. Read of 16 bytes from
        // `litBuf[litPtr_offset..]` is in-range; write of 16 bytes to
        // `dst[op..op+16]` is in-range (op + 16 ≤ oLitEnd + 16 ≤ oend).
        // Bytes [op + lit_len .. op + 16] contain garbage future-literal
        // bytes but get overwritten by the immediately-following match
        // copy (when ml + lit_len ≥ 16) or by the next sequence's
        // literal/match writes.
        let src_p = lit_base.add(litPtr_offset);
        let dst_p = dst.as_mut_ptr().add(op);
        let v = (src_p as *const [u8; 16]).read_unaligned();
        (dst_p as *mut [u8; 16]).write_unaligned(v);
    } else if lit_len > 16 && iLitEnd + 16 <= lit_len_total && oLitEnd + 16 <= oend {
        // Wildcopy 16-byte stamps when both sides have slack. The
        // 16-byte overshoot past oLitEnd is bounded at ≤ 15 bytes
        // (since end = op + ceil(lit_len/16)*16 ≤ oLitEnd + 15).
        // SAFETY: each stamp's read/write is bounded by the slack guard.
        let mut copied = 0usize;
        let src_p = lit_base.add(litPtr_offset);
        let dst_p = dst.as_mut_ptr().add(op);
        loop {
            let v = (src_p.add(copied) as *const [u8; 16]).read_unaligned();
            (dst_p.add(copied) as *mut [u8; 16]).write_unaligned(v);
            copied += 16;
            if copied >= lit_len {
                break;
            }
        }
    } else {
        std::ptr::copy_nonoverlapping(
            lit_base.add(litPtr_offset),
            dst.as_mut_ptr().add(op),
            lit_len,
        );
    }
    *out_litPtr = iLitEnd;

    // ext-dict path: offset reaches into ext_history. Mirrors the
    // upstream `if (sequence.offset > (size_t)(oLitEnd - prefixStart))`
    // branch — copy the leading segment from ext_history's tail, then
    // continue from the start of dst if the match spans both regions.
    if sequence.offset > oLitEnd {
        let into_ext = sequence.offset - oLitEnd;
        let ext_start = ext_history.len() - into_ext;
        let from_ext = sequence.matchLength.min(into_ext);
        dst[oLitEnd..oLitEnd + from_ext]
            .copy_from_slice(&ext_history[ext_start..ext_start + from_ext]);
        if sequence.matchLength > from_ext {
            // Spill into the prefix portion of dst (starting at index 0).
            // This is a *forward* wildcopy: each write may be read back
            // by a later iteration when the spill length exceeds the
            // initial offset, which is the ZSTD repcode-style pattern
            // expansion. `copy_within` is memmove-semantics — for
            // overlapping forward copies it preserves the original src
            // bytes, breaking the expansion. Do it byte-by-byte.
            let remaining = sequence.matchLength - from_ext;
            let dst_off = oLitEnd + from_ext;
            for i in 0..remaining {
                dst[dst_off + i] = dst[i];
            }
        }
        return sequenceLength;
    }

    // In-buffer match: copy from dst[match_src..] to dst[oLitEnd..oMatchEnd].
    // If offset >= WILDCOPY_VECLEN, the regions don't overlap within
    // the 16-byte wildcopy window, so a plain byte-by-byte forward
    // copy is correct (and matches upstream's `ZSTD_wildcopy`
    // semantics on non-overlap). Otherwise we use the overlap-spread
    // primitive `ZSTD_overlapCopy8` plus a src-before-dst wildcopy.
    let match_src = oLitEnd - sequence.offset;
    let mut match_idx = match_src;
    let mut out_idx = oLitEnd;

    if sequence.offset >= WILDCOPY_VECLEN {
        // True non-overlap fast path: when offset > matchLength,
        // the source and destination ranges don't overlap at all —
        // safe to use `copy_within` (memmove) which lowers to a
        // SIMD/SSE memcpy on x86.
        if sequence.offset >= sequence.matchLength {
            // Inline 16-byte stamp for short matches (typical: 5-15
            // bytes). `copy_within` lowers to a memmove call (~30 cycles
            // call overhead per match) — for ml ≤ 16 we can do one
            // unaligned 16-byte read+write instead. The over-write
            // beyond oMatchEnd into [oMatchEnd, oLitEnd+16] is fine: the
            // next sequence's literal copy at op_next == oMatchEnd will
            // overwrite the garbage, OR the tail-literals copy or
            // caller-side fill will. Gated on slack so dst doesn't
            // overflow at end-of-block.
            let ml = sequence.matchLength;
            if ml <= 16 && oLitEnd + 16 <= oend {
                // SAFETY: offset >= ml AND offset >= 16, so match_src + 16
                // = oLitEnd - offset + 16 ≤ oLitEnd ≤ oend, and the read
                // and write ranges are disjoint (source is fully behind
                // the destination by at least `offset` bytes).
                unsafe {
                    let dst_ptr = dst.as_mut_ptr();
                    let src_p = dst_ptr.add(match_src);
                    let dst_p = dst_ptr.add(oLitEnd);
                    let v = (src_p as *const [u8; 16]).read_unaligned();
                    (dst_p as *mut [u8; 16]).write_unaligned(v);
                }
            } else if oLitEnd + ml + 16 <= oend {
                // Wildcopy fast path for longer matches with full slack.
                // Stamps end at oLitEnd + ((ml-1) | 15) + 1 ≤ oLitEnd +
                // ml + 15 < oLitEnd + ml + 16 ≤ oend. Reads similarly
                // bounded since match_src + ml ≤ oLitEnd by `offset >= ml`.
                // SAFETY: per-stamp bounds (write at `oLitEnd + copied`,
                // read at `match_src + copied`, both for 16 bytes); copied
                // < ml + 16 in the worst case. Source/dest disjoint
                // because offset ≥ matchLength.
                unsafe {
                    let dst_ptr = dst.as_mut_ptr();
                    let src_p = dst_ptr.add(match_src);
                    let dst_p = dst_ptr.add(oLitEnd);
                    let mut copied = 0usize;
                    loop {
                        let v = (src_p.add(copied) as *const [u8; 16]).read_unaligned();
                        (dst_p.add(copied) as *mut [u8; 16]).write_unaligned(v);
                        copied += 16;
                        if copied >= ml {
                            break;
                        }
                    }
                }
            } else {
                unsafe {
                    let dst_ptr = dst.as_mut_ptr();
                    std::ptr::copy_nonoverlapping(dst_ptr.add(match_src), dst_ptr.add(oLitEnd), ml);
                }
            }
        } else {
            // Wildcopy-style overlapping case: offset >= 16 but the
            // match wraps around so the destination overlaps the
            // source. The repcode-pattern semantics require bytes
            // written by an earlier iteration to be read by a later
            // one — that's how a `(offset, matchLength)` pair
            // expands to a repeating pattern of length `matchLength`
            // bytes. Since `offset >= 16`, we can safely copy in
            // 16-byte chunks: each chunk reads bytes that are at
            // least 16 positions behind the current write head, so
            // the chunk read never aliases the chunk write. The tail
            // is finished byte-by-byte to handle any leftover bytes
            // and the final < 16-byte stride. Mirrors upstream's
            // `ZSTD_wildcopy(..., ZSTD_overlap_src_before_dst)` for
            // the medium-offset overlap case.
            let mut copied = 0usize;
            unsafe {
                let dst_ptr = dst.as_mut_ptr();
                while copied + 16 <= sequence.matchLength {
                    let v = (dst_ptr.add(match_src + copied) as *const [u8; 16]).read_unaligned();
                    (dst_ptr.add(oLitEnd + copied) as *mut [u8; 16]).write_unaligned(v);
                    copied += 16;
                }
            }
            if copied < sequence.matchLength {
                if oLitEnd + copied + 16 <= oend {
                    unsafe {
                        let dst_ptr = dst.as_mut_ptr();
                        let v =
                            (dst_ptr.add(match_src + copied) as *const [u8; 16]).read_unaligned();
                        (dst_ptr.add(oLitEnd + copied) as *mut [u8; 16]).write_unaligned(v);
                    }
                } else {
                    while copied < sequence.matchLength {
                        dst[oLitEnd + copied] = dst[match_src + copied];
                        copied += 1;
                    }
                }
            }
        }
    } else {
        // Overlap-spread up to 8 bytes, then wildcopy-style tail.
        // We need at least 8 bytes written so overlapCopy8 can spread.
        // Guard against buffers too short for the 8-byte stamp.
        if oMatchEnd < out_idx + 8 {
            // Tail too short for spread path; fall back to byte-by-byte.
            for i in 0..sequence.matchLength {
                dst[oLitEnd + i] = dst[match_src + i];
            }
        } else {
            ZSTD_overlapCopy8(dst, &mut out_idx, &mut match_idx, sequence.offset);
            // out_idx is now oLitEnd + 8. The overlap spreader has moved
            // match_idx far enough behind out_idx that 8-byte stamps are
            // overlap-safe while preserving pattern expansion semantics.
            let mut copied = 0usize;
            let remaining = sequence.matchLength - 8;
            unsafe {
                let dst_ptr = dst.as_mut_ptr();
                while copied + 8 <= remaining {
                    let v = (dst_ptr.add(match_idx + copied) as *const [u8; 8]).read_unaligned();
                    (dst_ptr.add(out_idx + copied) as *mut [u8; 8]).write_unaligned(v);
                    copied += 8;
                }
            }
            while copied < remaining {
                dst[out_idx + copied] = dst[match_idx + copied];
                copied += 1;
            }
        }
    }
    sequenceLength
}

/// Rust-only helper: cold fallback path for `ZSTD_execSequence_rawLit`.
/// Marked `#[cold]` so the optimizer keeps the inline fast path lean.
#[cold]
#[inline(never)]
unsafe fn ZSTD_execSequence_rawLit_fallback(
    dst: &mut [u8],
    op: usize,
    sequence: seq_t,
    ext_history: &[u8],
    lit_base: *const u8,
    lit_len_total: usize,
    litPtr_offset: usize,
    out_litPtr: &mut usize,
) -> usize {
    ZSTD_execSequence_rawLit(
        dst,
        op,
        sequence,
        ext_history,
        lit_base,
        lit_len_total,
        litPtr_offset,
        out_litPtr,
    )
}

/// Rust-only helper: bounds-precheck-free fast variant of
/// `ZSTD_execSequence_rawLit`. Called only after the body decoder has
/// already verified that 16-byte slack exists on both sides.
#[inline(always)]
unsafe fn ZSTD_execSequence_rawLit_fast(
    dst: &mut [u8],
    op: usize,
    litLength: usize,
    matchLength: usize,
    offset: usize,
    lit_base: *const u8,
    litPtr_offset: usize,
) -> usize {
    let oLitEnd = op + litLength;
    let sequenceLength = litLength + matchLength;

    let dst_ptr = dst.as_mut_ptr();
    let lit_src = lit_base.add(litPtr_offset);
    let lit_dst = dst_ptr.add(op);
    let v = (lit_src as *const [u8; 16]).read_unaligned();
    (lit_dst as *mut [u8; 16]).write_unaligned(v);
    if litLength > 16 {
        let mut copied = 16usize;
        while copied < litLength {
            let v = (lit_src.add(copied) as *const [u8; 16]).read_unaligned();
            (lit_dst.add(copied) as *mut [u8; 16]).write_unaligned(v);
            copied += 16;
        }
    }
    let match_src = oLitEnd - offset;
    let mut match_idx = match_src;
    let mut out_idx = oLitEnd;
    if offset >= WILDCOPY_VECLEN {
        let v = (dst_ptr.add(match_src) as *const [u8; 16]).read_unaligned();
        (dst_ptr.add(oLitEnd) as *mut [u8; 16]).write_unaligned(v);
        if matchLength > 16 {
            let mut copied = 16usize;
            while copied < matchLength {
                let v = (dst_ptr.add(match_src + copied) as *const [u8; 16]).read_unaligned();
                (dst_ptr.add(oLitEnd + copied) as *mut [u8; 16]).write_unaligned(v);
                copied += 16;
            }
        }
    } else {
        ZSTD_overlapCopy8(dst, &mut out_idx, &mut match_idx, offset);
        if matchLength > 8 {
            let mut copied = 0usize;
            let remaining = matchLength - 8;
            while copied < remaining {
                let v = (dst_ptr.add(match_idx + copied) as *const [u8; 8]).read_unaligned();
                (dst_ptr.add(out_idx + copied) as *mut [u8; 8]).write_unaligned(v);
                copied += 8;
            }
        }
    }
    sequenceLength
}

/// Rust-only helper: raw-pointer twin of `ZSTD_overlapCopy8`. Used by
/// the bounds-precheck-free fast sequence executor where the buffer is
/// already pinned to a `*mut u8`.
#[inline(always)]
unsafe fn ZSTD_overlapCopy8_ptr(ptr: *mut u8, op: &mut usize, ip: &mut usize, offset: usize) {
    debug_assert!(*ip <= *op);
    if offset < 8 {
        const DEC32: [usize; 8] = [0, 1, 2, 1, 4, 4, 4, 4];
        const DEC64: [usize; 8] = [8, 8, 8, 7, 8, 9, 10, 11];
        let sub2 = DEC64[offset];
        let op0 = *op;
        let ip0 = *ip;
        *ptr.add(op0) = *ptr.add(ip0);
        *ptr.add(op0 + 1) = *ptr.add(ip0 + 1);
        *ptr.add(op0 + 2) = *ptr.add(ip0 + 2);
        *ptr.add(op0 + 3) = *ptr.add(ip0 + 3);
        *ip = ip.wrapping_add(DEC32[offset]);
        let ip1 = *ip;
        let s0 = *ptr.add(ip1);
        let s1 = *ptr.add(ip1 + 1);
        let s2 = *ptr.add(ip1 + 2);
        let s3 = *ptr.add(ip1 + 3);
        *ptr.add(op0 + 4) = s0;
        *ptr.add(op0 + 5) = s1;
        *ptr.add(op0 + 6) = s2;
        *ptr.add(op0 + 7) = s3;
        *ip = ip.wrapping_sub(sub2);
    } else {
        let v = (ptr.add(*ip) as *const [u8; 8]).read_unaligned();
        (ptr.add(*op) as *mut [u8; 8]).write_unaligned(v);
    }
    *ip = ip.wrapping_add(8);
    *op = op.wrapping_add(8);
}

/// Rust-only helper: raw-pointer twin of
/// `ZSTD_execSequence_rawLit_fast`. Avoids the `&mut [u8]` borrow so
/// the split-literal-buffer decoder can write through one pointer.
#[inline(always)]
unsafe fn ZSTD_execSequence_rawLit_fast_ptr(
    dst_ptr: *mut u8,
    op: usize,
    litLength: usize,
    matchLength: usize,
    offset: usize,
    lit_base: *const u8,
    litPtr_offset: usize,
) -> usize {
    let oLitEnd = op + litLength;
    let sequenceLength = litLength + matchLength;

    let lit_src = lit_base.add(litPtr_offset);
    let lit_dst = dst_ptr.add(op);
    let v = (lit_src as *const [u8; 16]).read_unaligned();
    (lit_dst as *mut [u8; 16]).write_unaligned(v);
    if litLength > 16 {
        let mut copied = 16usize;
        while copied < litLength {
            let v = (lit_src.add(copied) as *const [u8; 16]).read_unaligned();
            (lit_dst.add(copied) as *mut [u8; 16]).write_unaligned(v);
            copied += 16;
        }
    }
    let match_src = oLitEnd - offset;
    let mut match_idx = match_src;
    let mut out_idx = oLitEnd;
    if offset >= WILDCOPY_VECLEN {
        let v = (dst_ptr.add(match_src) as *const [u8; 16]).read_unaligned();
        (dst_ptr.add(oLitEnd) as *mut [u8; 16]).write_unaligned(v);
        if matchLength > 16 {
            let mut copied = 16usize;
            while copied < matchLength {
                let v = (dst_ptr.add(match_src + copied) as *const [u8; 16]).read_unaligned();
                (dst_ptr.add(oLitEnd + copied) as *mut [u8; 16]).write_unaligned(v);
                copied += 16;
            }
        }
    } else {
        ZSTD_overlapCopy8_ptr(dst_ptr, &mut out_idx, &mut match_idx, offset);
        if matchLength > 8 {
            let mut copied = 0usize;
            let remaining = matchLength - 8;
            while copied < remaining {
                let v = (dst_ptr.add(match_idx + copied) as *const [u8; 8]).read_unaligned();
                (dst_ptr.add(out_idx + copied) as *mut [u8; 8]).write_unaligned(v);
                copied += 8;
            }
        }
    }
    sequenceLength
}

/// Slow-tail wrapper for `ZSTD_execSequenceEnd`. The safe Rust
/// `ZSTD_execSequence` already performs bounds checks before copying,
/// so the end-path entry forwards to the same implementation.
pub fn ZSTD_execSequenceEnd(
    dst: &mut [u8],
    op: usize,
    sequence: seq_t,
    ext_history: &[u8],
    litBuf: &[u8],
    litPtr_offset: usize,
    out_litPtr: &mut usize,
) -> usize {
    ZSTD_execSequence(
        dst,
        op,
        sequence,
        ext_history,
        litBuf,
        litPtr_offset,
        out_litPtr,
    )
}

/// Slow-tail wrapper for `ZSTD_execSequenceEndSplitLitBuffer`.
pub fn ZSTD_execSequenceEndSplitLitBuffer(
    dst: &mut [u8],
    op: usize,
    sequence: seq_t,
    ext_history: &[u8],
    litBuf: &[u8],
    litPtr_offset: usize,
    out_litPtr: &mut usize,
) -> usize {
    ZSTD_execSequence(
        dst,
        op,
        sequence,
        ext_history,
        litBuf,
        litPtr_offset,
        out_litPtr,
    )
}

/// Split-literal wrapper for `ZSTD_execSequenceSplitLitBuffer`.
pub fn ZSTD_execSequenceSplitLitBuffer(
    dst: &mut [u8],
    op: usize,
    sequence: seq_t,
    ext_history: &[u8],
    litBuf: &[u8],
    litPtr_offset: usize,
    out_litPtr: &mut usize,
) -> usize {
    ZSTD_execSequence(
        dst,
        op,
        sequence,
        ext_history,
        litBuf,
        litPtr_offset,
        out_litPtr,
    )
}

/// Fuzz-assert helper name. The production Rust port doesn't keep the
/// fuzz-only dictionary sentinels, so this reports whether an ext-dict
/// span is currently recorded on the DCtx.
#[inline]
pub fn ZSTD_dictionaryIsActive(dctx: &ZSTD_DCtx, _prefixStart: usize, _oLitEnd: usize) -> i32 {
    dctx.dictEnd.is_some() as i32
}

/// Fuzz-only validation hook in upstream. Kept as a no-op entry point
/// because normal decoding paths already return zstd errors instead
/// of relying on debug assertions.
#[inline]
pub fn ZSTD_assertValidSequence(
    _dctx: &ZSTD_DCtx,
    _op: usize,
    _oend: usize,
    _seq: seq_t,
    _prefixStart: usize,
    _virtualStart: usize,
) {
}

// ---- Literals-block types -------------------------------------------

/// Mirror of `SymbolEncodingType_e`. Shared between literals-block
/// and sequences-block headers — named "symbol encoding" because it
/// reuses the same 2-bit discriminator across both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum SymbolEncodingType_e {
    #[default]
    set_basic = 0,
    set_rle = 1,
    set_compressed = 2,
    set_repeat = 3,
}

impl SymbolEncodingType_e {
    fn from_bits(bits: u32) -> Self {
        match bits & 3 {
            0 => Self::set_basic,
            1 => Self::set_rle,
            2 => Self::set_compressed,
            _ => Self::set_repeat,
        }
    }
}

/// Mirror of `ZSTD_litLocation_e`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum ZSTD_litLocation_e {
    #[default]
    ZSTD_not_in_dst = 0,
    ZSTD_in_dst = 1,
    ZSTD_split = 2,
}

/// Mirror of `streaming_operation`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum streaming_operation {
    not_streaming = 0,
    is_streaming = 1,
}

pub const MIN_LITERALS_FOR_4_STREAMS: usize = 6;

/// Minimal `ZSTD_DCtx` — this crate's cut-down mirror of upstream's
/// mammoth struct. We only surface the fields that the already-ported
/// decoder pieces touch; the rest will arrive incrementally.
///
/// Upstream packs many decoder-internal caches into one struct; the
/// Rust port keeps them in the same place so the 1:1 mapping is still
/// useful to code-complexity-comparator.
#[derive(Clone)]
pub struct ZSTD_DCtx {
    /// Huffman decoding table, sized for `HUF_TABLELOG_MAX`.
    pub hufTable: Vec<HUF_DTable>,
    /// Scratch space for `HUF_read*DTable*_wksp`.
    pub workspace: Vec<u32>,
    /// Literal-buffer scratch; owned when `litBufferLocation ==
    /// ZSTD_not_in_dst`.
    pub litExtraBuffer: Vec<u8>,

    /// Pointer-equivalent into `litExtraBuffer` or `dst` (as a byte
    /// index plus a tag). A real port will use an `enum` over
    /// `&mut [u8]` slices once DCtx lifetimes are settled.
    pub litBuffer_offset: usize,
    pub litBufferEnd_offset: usize,
    pub litBufferLocation: ZSTD_litLocation_e,

    pub litPtr_offset: usize,
    pub litPtr_from_dst: bool, // if true, offset is into dst; else into litExtraBuffer
    pub litSize: usize,
    pub litEntropy: u32,

    /// Upstream `dctx->dictID`. Parsed from a zstd-format dict's
    /// 4-byte ID field when one is loaded; 0 for raw-content dicts.
    pub dictID: u32,
    /// Upstream `dctx->ddict_rep[ZSTD_REP_NUM]`. Rep values parsed
    /// from a zstd-format dict's trailing 12-byte rep section.
    pub ddict_rep: [u32; 3],
    /// End pointer-equivalent of the prior decode/insert destination.
    pub previousDstEnd: Option<usize>,
    /// Start pointer-equivalent of the current history segment.
    pub prefixStart: Option<usize>,
    /// Virtual start pointer-equivalent spanning the current segment
    /// plus the immediately-previous contiguous one.
    pub virtualStart: Option<usize>,
    /// End pointer-equivalent of the previous segment once continuity
    /// is broken and it becomes an external dictionary.
    pub dictEnd: Option<usize>,

    /// Upstream `dctx->isFrameDecompression`. Fresh DCtx defaults to
    /// frame mode; standalone block wrappers temporarily force this to
    /// 0 so `ZSTD_blockSizeMax()` ignores the frame header cap.
    pub isFrameDecompression: i32,
    /// Most fParams fields are unused in the literal-block decode;
    /// keep just the block-size cap we read in frame mode.
    pub blockSizeMax: usize,
    pub disableHufAsm: i32,
    pub bmi2: i32,
    pub ddictIsCold: i32,

    // ---- sequences state ----
    pub LLTable: Vec<ZSTD_seqSymbol>,
    pub OFTable: Vec<ZSTD_seqSymbol>,
    pub MLTable: Vec<ZSTD_seqSymbol>,
    pub ll_default_active: bool,
    pub of_default_active: bool,
    pub ml_default_active: bool,
    pub fseEntropy: u32,
    /// Per-table repeat breadcrumbs retained for compatibility with
    /// older tests. `set_repeat` validity is gated by `fseEntropy`,
    /// matching upstream's single `flagRepeatTable` argument.
    pub fse_ll_fresh: bool,
    pub fse_of_fresh: bool,
    pub fse_ml_fresh: bool,

    // ---- Streaming-mode state (initDStream / decompressStream) ----
    /// Pending compressed bytes staged by `ZSTD_decompressStream`.
    pub stream_in_buffer: Vec<u8>,
    /// Decoded output awaiting drain into the caller's output buffer.
    pub stream_out_buffer: Vec<u8>,
    /// Bytes already drained from `stream_out_buffer` into the caller.
    pub stream_out_drained: usize,
    /// Upstream `oversizedDuration`: number of consecutive streaming
    /// calls whose retained buffers are much larger than the needed
    /// input/output workspace.
    pub oversizedDuration: usize,
    /// Raw-content dict set by `ZSTD_initDStream_usingDict` — applied
    /// to every frame decoded until reset / re-init.
    pub stream_dict: Vec<u8>,
    /// Sliding history of prior-block output, retained across
    /// `ZSTD_decompressContinue` calls so back-references in later
    /// blocks can resolve into earlier-block bytes that no longer live
    /// in the caller's `dst`. Capped at `fParams.windowSize` (or
    /// `ZSTD_BLOCKSIZE_MAX` before a frame header has been parsed).
    /// Reset on frame boundaries and on `ZSTD_decompressBegin`.
    pub historyBuffer: Vec<u8>,
    /// `ZSTD_d_windowLogMax`: public log-form upper bound on the
    /// window size the decoder will accept. Set via
    /// `ZSTD_DCtx_setParameter`.
    pub d_windowLogMax: u32,
    /// Byte-form max window size used for streaming enforcement.
    /// `ZSTD_DCtx_setMaxWindowSize()` accepts byte sizes that are not
    /// powers of two, so the effective cap must not be reconstructed
    /// from `d_windowLogMax`.
    pub d_maxWindowSize: u64,
    /// Tracks whether the max-window cap came from an explicit public
    /// setter rather than the fresh-DCtx default.
    pub d_maxWindowSizeSet: bool,

    /// Upstream `dctx->format`. Selected decoder format — zstd1 (with
    /// magic) or zstd1_magicless (for embedded / streaming contexts
    /// where the 4-byte magic is known redundantly).
    pub format: crate::decompress::zstd_decompress::ZSTD_format_e,
    /// Upstream `dctx->forceIgnoreChecksum`. User-specified override
    /// for skipping frame checksum validation even when the frame
    /// header advertises a checksum. Default is validate (0).
    pub forceIgnoreChecksum: crate::compress::zstd_compress::ZSTD_forceIgnoreChecksum_e,
    /// Legacy `decompressContinue()` expected compressed-byte count
    /// for the next state-machine transition.
    pub expected: usize,
    /// Legacy block/frame decode stage.
    pub stage: crate::decompress::zstd_decompress::ZSTD_dStage,
    /// Running compressed bytes consumed by the legacy continue API.
    pub processedCSize: u64,
    /// Running decompressed bytes produced for the current frame.
    pub decodedSize: u64,
    /// Frame header decoded for the current frame.
    pub fParams: crate::decompress::zstd_decompress::ZSTD_FrameHeader,
    /// Block type latched from the most recent block header.
    pub bType: blockType_e,
    /// Whether to validate the checksum trailer for the current frame.
    pub validateChecksum: u32,
    /// XXH64 state for checksum validation in the legacy continue API.
    pub xxhState: crate::common::xxhash::XXH64_state_t,
    /// Full frame/skippable header bytes staged across split calls.
    pub headerBuffer: [u8; crate::compress::zstd_compress::ZSTD_FRAMEHEADERSIZE_MAX],
    /// Full header size expected for `headerBuffer`.
    pub headerSize: usize,
    /// Current RLE block's regenerated size.
    pub rleSize: usize,

    /// Upstream `dctx->dictUses` (`zstd_decompress_internal.h:97`).
    /// Tracks the dict lifecycle: `ZSTD_dont_use` = no dict attached;
    /// `ZSTD_use_once` = `refPrefix`-bound dict that auto-clears
    /// after one frame; `ZSTD_use_indefinitely` = persistent dict
    /// from `loadDictionary` / `refDDict`.
    pub dictUses: crate::decompress::zstd_decompress::ZSTD_dictUses_e,
    /// Upstream custom allocator bundle requested for this DCtx.
    pub customMem: crate::compress::zstd_compress::ZSTD_customMem,
}

/// Rust-only helper: build the canonical default LL FSE decode table
/// from the upstream-spec `LL_defaultNorm` / `LL_defaultNormLog`.
fn build_default_ll_dtable_fresh() -> Vec<ZSTD_seqSymbol> {
    let mut t = vec![ZSTD_seqSymbol::default(); SEQSYMBOL_TABLE_SIZE(LLFSELog)];
    ZSTD_buildFSETable(
        &mut t,
        &LL_defaultNorm,
        MaxLL,
        &LL_base,
        &LL_bits,
        LL_defaultNormLog,
    );
    t
}

/// Rust-only helper: build the canonical default OF FSE decode table.
fn build_default_of_dtable_fresh() -> Vec<ZSTD_seqSymbol> {
    let mut t = vec![ZSTD_seqSymbol::default(); SEQSYMBOL_TABLE_SIZE(OffFSELog)];
    ZSTD_buildFSETable(
        &mut t,
        &OF_defaultNorm,
        DefaultMaxOff,
        &OF_base,
        &OF_bits,
        OF_defaultNormLog,
    );
    t
}

/// Rust-only helper: build the canonical default ML FSE decode table.
fn build_default_ml_dtable_fresh() -> Vec<ZSTD_seqSymbol> {
    let mut t = vec![ZSTD_seqSymbol::default(); SEQSYMBOL_TABLE_SIZE(MLFSELog)];
    ZSTD_buildFSETable(
        &mut t,
        &ML_defaultNorm,
        MaxML,
        &ML_base,
        &ML_bits,
        ML_defaultNormLog,
    );
    t
}

/// Rust-only helper: cached canonical LL decode table, built lazily.
pub(crate) fn default_ll_dtable() -> &'static [ZSTD_seqSymbol] {
    static TABLE: std::sync::OnceLock<Vec<ZSTD_seqSymbol>> = std::sync::OnceLock::new();
    TABLE.get_or_init(build_default_ll_dtable_fresh).as_slice()
}

/// Rust-only helper: cached canonical OF decode table, built lazily.
pub(crate) fn default_of_dtable() -> &'static [ZSTD_seqSymbol] {
    static TABLE: std::sync::OnceLock<Vec<ZSTD_seqSymbol>> = std::sync::OnceLock::new();
    TABLE.get_or_init(build_default_of_dtable_fresh).as_slice()
}

/// Rust-only helper: cached canonical ML decode table, built lazily.
pub(crate) fn default_ml_dtable() -> &'static [ZSTD_seqSymbol] {
    static TABLE: std::sync::OnceLock<Vec<ZSTD_seqSymbol>> = std::sync::OnceLock::new();
    TABLE.get_or_init(build_default_ml_dtable_fresh).as_slice()
}

/// Rust-only helper: allocate a zero-initialized FSE decode table of
/// size `SEQSYMBOL_TABLE_SIZE(table_log)`.
fn alloc_seq_table(table_log: u32) -> Vec<ZSTD_seqSymbol> {
    vec![ZSTD_seqSymbol::default(); SEQSYMBOL_TABLE_SIZE(table_log)]
}

impl Default for ZSTD_DCtx {
    fn default() -> Self {
        // Initialize the HUF DTable's DTableDesc header exactly like
        // upstream's `HUF_CREATE_STATIC_DTABLEX1` macro:
        //   first_u32 = (maxTableLog - 1) * 0x01000001
        // which lays `maxTableLog-1` into byte 0 (the maxTableLog
        // field) and byte 3 (the reserved field). Without this the
        // header reads `maxTableLog=0` and `HUF_readDTableX1` rejects
        // every input as TableLogTooLarge.
        let mut hufTable = vec![0u32; HUF_DTABLE_SIZE_U32(HUF_TABLELOG_MAX)];
        let mtl_minus_1 = HUF_TABLELOG_MAX - 1;
        hufTable[0] = mtl_minus_1 * 0x01_00_00_01;
        Self {
            hufTable,
            workspace: vec![
                0u32;
                crate::decompress::huf_decompress::HUF_DECOMPRESS_WORKSPACE_SIZE_U32
            ],
            litExtraBuffer: vec![0u8; ZSTD_LITBUFFEREXTRASIZE + WILDCOPY_OVERLENGTH],
            litBuffer_offset: 0,
            litBufferEnd_offset: 0,
            litBufferLocation: ZSTD_litLocation_e::ZSTD_not_in_dst,
            litPtr_offset: 0,
            litPtr_from_dst: false,
            litSize: 0,
            litEntropy: 0,
            dictID: 0,
            ddict_rep: [0; 3],
            previousDstEnd: None,
            prefixStart: None,
            virtualStart: None,
            dictEnd: None,
            isFrameDecompression: 1,
            blockSizeMax: ZSTD_BLOCKSIZE_MAX,
            disableHufAsm: 0,
            bmi2: crate::common::zstd_internal::ZSTD_cpuSupportsBmi2(),
            ddictIsCold: 0,
            LLTable: alloc_seq_table(LLFSELog),
            OFTable: alloc_seq_table(OffFSELog),
            MLTable: alloc_seq_table(MLFSELog),
            ll_default_active: true,
            of_default_active: true,
            ml_default_active: true,
            fseEntropy: 0,
            fse_ll_fresh: false,
            fse_of_fresh: false,
            fse_ml_fresh: false,
            stream_in_buffer: Vec::new(),
            stream_out_buffer: Vec::new(),
            stream_out_drained: 0,
            oversizedDuration: 0,
            stream_dict: Vec::new(),
            historyBuffer: Vec::new(),
            d_windowLogMax: crate::decompress::zstd_decompress::ZSTD_WINDOWLOG_LIMIT_DEFAULT,
            d_maxWindowSize: (1u64
                << crate::decompress::zstd_decompress::ZSTD_WINDOWLOG_LIMIT_DEFAULT)
                + 1,
            d_maxWindowSizeSet: false,
            format: crate::decompress::zstd_decompress::ZSTD_format_e::ZSTD_f_zstd1,
            forceIgnoreChecksum:
                crate::compress::zstd_compress::ZSTD_forceIgnoreChecksum_e::ZSTD_d_validateChecksum,
            expected: crate::decompress::zstd_decompress::ZSTD_startingInputLength(
                crate::decompress::zstd_decompress::ZSTD_format_e::ZSTD_f_zstd1,
            ),
            stage: crate::decompress::zstd_decompress::ZSTD_dStage::ZSTDds_getFrameHeaderSize,
            processedCSize: 0,
            decodedSize: 0,
            fParams: crate::decompress::zstd_decompress::ZSTD_FrameHeader::default(),
            bType: blockType_e::bt_raw,
            validateChecksum: 0,
            xxhState: crate::common::xxhash::XXH64_state_t::default(),
            headerBuffer: [0u8; crate::compress::zstd_compress::ZSTD_FRAMEHEADERSIZE_MAX],
            headerSize: 0,
            rleSize: 0,
            dictUses: crate::decompress::zstd_decompress::ZSTD_dictUses_e::ZSTD_dont_use,
            customMem: crate::compress::zstd_compress::ZSTD_customMem::default(),
        }
    }
}

impl ZSTD_DCtx {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Port of `ZSTD_blockSizeMax`. In standalone block decode mode it's
/// capped at `ZSTD_BLOCKSIZE_MAX`; during frame decoding the frame
/// header's `blockSizeMax` can lower it further.
fn ZSTD_blockSizeMax(dctx: &ZSTD_DCtx) -> usize {
    if dctx.isFrameDecompression != 0 {
        dctx.blockSizeMax
    } else {
        ZSTD_BLOCKSIZE_MAX
    }
}

/// Port of `ZSTD_allocateLiteralsBuffer`. Selects the same literal
/// storage location as upstream: excess room after `dst`, the DCtx
/// extra buffer, or a split between `dst` and the extra buffer.
fn ZSTD_allocateLiteralsBuffer(
    dctx: &mut ZSTD_DCtx,
    _dst_len: usize,
    dst_capacity: usize,
    litSize: usize,
    streaming: streaming_operation,
    expectedWriteSize: usize,
    splitImmediately: u32,
) {
    let blockSizeMax = ZSTD_blockSizeMax(dctx);
    if streaming == streaming_operation::not_streaming
        && dst_capacity > blockSizeMax + WILDCOPY_OVERLENGTH + litSize + WILDCOPY_OVERLENGTH
    {
        dctx.litBuffer_offset = blockSizeMax + WILDCOPY_OVERLENGTH;
        dctx.litBufferEnd_offset = dctx.litBuffer_offset + litSize;
        dctx.litBufferLocation = ZSTD_litLocation_e::ZSTD_in_dst;
        return;
    }

    let needed = litSize.saturating_add(WILDCOPY_OVERLENGTH);
    if dctx.litExtraBuffer.len() < needed {
        dctx.litExtraBuffer.resize(needed, 0);
    }
    if litSize > ZSTD_LITBUFFEREXTRASIZE {
        if splitImmediately != 0 {
            dctx.litBuffer_offset =
                expectedWriteSize - litSize + ZSTD_LITBUFFEREXTRASIZE - WILDCOPY_OVERLENGTH;
            dctx.litBufferEnd_offset = dctx.litBuffer_offset + litSize - ZSTD_LITBUFFEREXTRASIZE;
        } else {
            dctx.litBuffer_offset = expectedWriteSize - litSize;
            dctx.litBufferEnd_offset = expectedWriteSize;
        }
        dctx.litBufferLocation = ZSTD_litLocation_e::ZSTD_split;
        return;
    }
    dctx.litBuffer_offset = 0;
    dctx.litBufferEnd_offset = litSize;
    dctx.litBufferLocation = ZSTD_litLocation_e::ZSTD_not_in_dst;
}

// ---- Top-level block API ----------------------------------------------

/// Decoder entropy state retained across blocks — mirrors the subset
/// of `ZSTD_entropyDTables_t` that this port touches today.
#[derive(Debug, Clone)]
pub struct ZSTD_decoder_entropy_rep {
    /// Repeated-offset history (zstd spec: 1, 4, 8 at frame start).
    pub rep: [u32; ZSTD_REP_NUM],
}

impl Default for ZSTD_decoder_entropy_rep {
    fn default() -> Self {
        Self { rep: [1, 4, 8] }
    }
}

/// Port of `ZSTD_decompressSequences_body`. Consumes the FSE-encoded
/// `seqSrc` stream, regenerating sequences into `dst` by calling
/// `ZSTD_decodeSequence` + `ZSTD_execSequence` in a loop. Emits the
/// trailing literals segment once `nbSeq` is exhausted.
///
/// Rust signature note: upstream reads literals from `dctx->litPtr`
/// and repcodes from `dctx->entropy.rep`. We accept them explicitly as
/// parameters so `dctx` doesn't need to be borrowed both immutably
/// (for tables / litBuf) and mutably (for rep[]) at once.
pub fn ZSTD_decompressSequences_body(
    dst: &mut [u8],
    op_start: usize,
    ext_history: &[u8],
    seqSrc: &[u8],
    nbSeq: i32,
    litBuf: &[u8],
    litSize: usize,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
) -> usize {
    unsafe {
        ZSTD_decompressSequences_body_rawLit(
            dst,
            op_start,
            ext_history,
            seqSrc,
            nbSeq,
            litBuf.as_ptr(),
            litSize,
            ZSTD_longOffset_e::ZSTD_lo_isRegularOffset,
            LLTable,
            OFTable,
            MLTable,
            entropy_rep,
        )
    }
}

/// Rust-only helper: raw-literal-pointer variant of
/// `ZSTD_decompressSequences_body`. Forms the unsafe core that the
/// safe `_body` entry forwards to after splatting `litBuf.as_ptr()`.
unsafe fn ZSTD_decompressSequences_body_rawLit(
    dst: &mut [u8],
    op_start: usize,
    ext_history: &[u8],
    seqSrc: &[u8],
    nbSeq: i32,
    lit_base: *const u8,
    litSize: usize,
    isLongOffset: ZSTD_longOffset_e,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
) -> usize {
    use crate::common::bitstream::{BIT_DStream_t, BIT_endOfDStream, BIT_initDStream};

    // `op_start` is the byte index into `dst` where the current block's
    // output begins. Prior-block output lives in `dst[..op_start]` and
    // must remain visible for cross-block back-references (offsets
    // larger than the current block's offset relative to op_start).
    // `ext_history` carries any further history that lives outside
    // `dst` — see `ZSTD_execSequence` for details.
    let mut op: usize = op_start;
    let mut litPtr: usize = 0;
    let litEnd = litSize;

    if nbSeq > 0 {
        let mut seqState = seqState_t {
            DStream: BIT_DStream_t::default(),
            stateLL: ZSTD_fseState::default(),
            stateOffb: ZSTD_fseState::default(),
            stateML: ZSTD_fseState::default(),
            prevOffset: [
                entropy_rep.rep[0] as usize,
                entropy_rep.rep[1] as usize,
                entropy_rep.rep[2] as usize,
            ],
        };

        let rc = BIT_initDStream(&mut seqState.DStream, seqSrc, seqSrc.len());
        if crate::common::error::ERR_isError(rc) {
            return ERROR(ErrorCode::CorruptionDetected);
        }

        ZSTD_initFseState(&mut seqState.stateLL, &mut seqState.DStream, LLTable);
        ZSTD_initFseState(&mut seqState.stateOffb, &mut seqState.DStream, OFTable);
        ZSTD_initFseState(&mut seqState.stateML, &mut seqState.DStream, MLTable);

        let mut remaining = nbSeq;
        while remaining > 0 {
            let sequence = ZSTD_decodeSequence(
                &mut seqState,
                isLongOffset,
                LLTable,
                OFTable,
                MLTable,
                remaining == 1,
            );
            let oneSeqSize = ZSTD_execSequence_rawLit(
                dst,
                op,
                sequence,
                ext_history,
                lit_base,
                litSize,
                litPtr,
                &mut litPtr,
            );
            if crate::common::error::ERR_isError(oneSeqSize) {
                return oneSeqSize;
            }
            op += oneSeqSize;
            remaining -= 1;
        }

        // Must land exactly at the end of the bit-stream.
        if BIT_endOfDStream(&seqState.DStream) == 0 {
            return ERROR(ErrorCode::CorruptionDetected);
        }

        // Save repcodes for the next block.
        entropy_rep.rep[0] = seqState.prevOffset[0] as u32;
        entropy_rep.rep[1] = seqState.prevOffset[1] as u32;
        entropy_rep.rep[2] = seqState.prevOffset[2] as u32;
    }

    // Tail literals segment: copy any remaining literals directly into dst.
    let lastLLSize = litEnd - litPtr;
    if op + lastLLSize > dst.len() {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    if lastLLSize > 0 {
        std::ptr::copy_nonoverlapping(lit_base.add(litPtr), dst.as_mut_ptr().add(op), lastLLSize);
        op += lastLLSize;
    }
    // Return bytes emitted by this call (not total dst position).
    op - op_start
}

/// Rust-only helper: split-literal-buffer sequence decoder body. The
/// const-generic `BMI2` flag picks between the portable and BMI2
/// per-sequence decoders. Handles the seam where literals straddle
/// the in-dst buffer and the extra-literal scratch.
unsafe fn ZSTD_decompressSequences_bodySplitLitBuffer_raw_impl<const BMI2: bool>(
    dst: &mut [u8],
    op_start: usize,
    dst_limit: usize,
    ext_history: &[u8],
    seqSrc: &[u8],
    nbSeq: i32,
    lit_dst_base: *const u8,
    lit_dst_len: usize,
    lit_extra: &[u8],
    litSize: usize,
    isLongOffset: ZSTD_longOffset_e,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
) -> usize {
    use crate::common::bitstream::{BIT_DStream_t, BIT_endOfDStream, BIT_initDStream};

    let mut op = op_start;
    let mut litPtr = 0usize;
    let mut extraPtr = 0usize;
    let mut using_extra = false;
    let dst_ptr = dst.as_mut_ptr();

    if nbSeq > 0 {
        let mut seqState = seqState_t {
            DStream: BIT_DStream_t::default(),
            stateLL: ZSTD_fseState::default(),
            stateOffb: ZSTD_fseState::default(),
            stateML: ZSTD_fseState::default(),
            prevOffset: [
                entropy_rep.rep[0] as usize,
                entropy_rep.rep[1] as usize,
                entropy_rep.rep[2] as usize,
            ],
        };

        let rc = BIT_initDStream(&mut seqState.DStream, seqSrc, seqSrc.len());
        if crate::common::error::ERR_isError(rc) {
            return ERROR(ErrorCode::CorruptionDetected);
        }
        ZSTD_initFseState(&mut seqState.stateLL, &mut seqState.DStream, LLTable);
        ZSTD_initFseState(&mut seqState.stateOffb, &mut seqState.DStream, OFTable);
        ZSTD_initFseState(&mut seqState.stateML, &mut seqState.DStream, MLTable);

        let mut remaining = nbSeq;
        while remaining > 0 {
            let mut sequence = if BMI2 {
                #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
                {
                    ZSTD_decodeSequence_bmi2(
                        &mut seqState,
                        isLongOffset,
                        LLTable,
                        OFTable,
                        MLTable,
                        remaining == 1,
                    )
                }
                #[cfg(not(all(target_arch = "x86_64", target_pointer_width = "64")))]
                {
                    unreachable!("BMI2 split decoder is x86_64-only")
                }
            } else {
                ZSTD_decodeSequence(
                    &mut seqState,
                    isLongOffset,
                    LLTable,
                    OFTable,
                    MLTable,
                    remaining == 1,
                )
            };
            if !using_extra && litPtr + sequence.litLength > lit_dst_len {
                let leftover = lit_dst_len.saturating_sub(litPtr);
                if leftover > dst.len().saturating_sub(op) {
                    return ERROR(ErrorCode::DstSizeTooSmall);
                }
                if leftover != 0 {
                    std::ptr::copy(lit_dst_base.add(litPtr), dst_ptr.add(op), leftover);
                    sequence.litLength -= leftover;
                    op += leftover;
                }
                using_extra = true;
                extraPtr = 0;
            }

            let (lit_base, lit_total, lit_ptr_ref) = if using_extra {
                (lit_extra.as_ptr(), lit_extra.len(), &mut extraPtr)
            } else {
                let dst_limited = std::slice::from_raw_parts_mut(dst_ptr, dst_limit);
                let oneSeqSize = ZSTD_execSequence_rawLit(
                    dst_limited,
                    op,
                    sequence,
                    ext_history,
                    lit_dst_base,
                    lit_dst_len,
                    litPtr,
                    &mut litPtr,
                );
                if crate::common::error::ERR_isError(oneSeqSize) {
                    return oneSeqSize;
                }
                op += oneSeqSize;
                remaining -= 1;
                continue;
            };
            let oneSeqSize = ZSTD_execSequence_rawLit(
                dst,
                op,
                sequence,
                ext_history,
                lit_base,
                lit_total,
                *lit_ptr_ref,
                lit_ptr_ref,
            );
            if crate::common::error::ERR_isError(oneSeqSize) {
                return oneSeqSize;
            }
            op += oneSeqSize;
            remaining -= 1;
        }

        if BIT_endOfDStream(&seqState.DStream) == 0 {
            return ERROR(ErrorCode::CorruptionDetected);
        }
        entropy_rep.rep[0] = seqState.prevOffset[0] as u32;
        entropy_rep.rep[1] = seqState.prevOffset[1] as u32;
        entropy_rep.rep[2] = seqState.prevOffset[2] as u32;
    }

    if !using_extra {
        let last = lit_dst_len - litPtr;
        if op + last > dst.len() {
            return ERROR(ErrorCode::DstSizeTooSmall);
        }
        if last != 0 {
            std::ptr::copy(lit_dst_base.add(litPtr), dst_ptr.add(op), last);
            op += last;
        }
        extraPtr = 0;
    }
    let last = litSize - lit_dst_len - extraPtr;
    if op + last > dst.len() {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    if last != 0 {
        std::ptr::copy_nonoverlapping(lit_extra.as_ptr().add(extraPtr), dst_ptr.add(op), last);
        op += last;
    }
    op - op_start
}

/// Rust-only helper: split-literal-buffer variant of `ZSTD_execSequence`.
/// Walks the seam between the in-dst literal region and the extra
/// scratch buffer when a single sequence's literals straddle them.
#[inline(always)]
unsafe fn ZSTD_execSequenceSplitLitBuffer_raw(
    dst: &mut [u8],
    dst_limit: usize,
    op: usize,
    sequence: &mut seq_t,
    ext_history: &[u8],
    lit_dst_base: *const u8,
    lit_dst_len: usize,
    lit_extra: &[u8],
    litPtr: &mut usize,
    extraPtr: &mut usize,
    using_extra: &mut bool,
) -> usize {
    let op_start = op;
    let mut op = op;
    if !*using_extra && *litPtr + sequence.litLength > lit_dst_len {
        let leftover = lit_dst_len.saturating_sub(*litPtr);
        if leftover > dst.len().saturating_sub(op) {
            return ERROR(ErrorCode::DstSizeTooSmall);
        }
        if leftover != 0 {
            std::ptr::copy(
                lit_dst_base.add(*litPtr),
                dst.as_mut_ptr().add(op),
                leftover,
            );
            sequence.litLength -= leftover;
            op += leftover;
        }
        *using_extra = true;
        *extraPtr = 0;
    }

    let oneSeqSize = if *using_extra {
        ZSTD_execSequence_rawLit(
            dst,
            op,
            *sequence,
            ext_history,
            lit_extra.as_ptr(),
            lit_extra.len(),
            *extraPtr,
            extraPtr,
        )
    } else {
        let dst_ptr = dst.as_mut_ptr();
        let dst_limited = std::slice::from_raw_parts_mut(dst_ptr, dst_limit);
        ZSTD_execSequence_rawLit(
            dst_limited,
            op,
            *sequence,
            ext_history,
            lit_dst_base,
            lit_dst_len,
            *litPtr,
            litPtr,
        )
    };
    if crate::common::error::ERR_isError(oneSeqSize) {
        return oneSeqSize;
    }
    oneSeqSize + (op - op_start)
}

/// Rust-only helper: portable specialization of
/// `ZSTD_decompressSequences_bodySplitLitBuffer_raw_impl::<false>`.
unsafe fn ZSTD_decompressSequences_bodySplitLitBuffer_raw(
    dst: &mut [u8],
    op_start: usize,
    dst_limit: usize,
    ext_history: &[u8],
    seqSrc: &[u8],
    nbSeq: i32,
    lit_dst_base: *const u8,
    lit_dst_len: usize,
    lit_extra: &[u8],
    litSize: usize,
    isLongOffset: ZSTD_longOffset_e,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
) -> usize {
    ZSTD_decompressSequences_bodySplitLitBuffer_raw_impl::<false>(
        dst,
        op_start,
        dst_limit,
        ext_history,
        seqSrc,
        nbSeq,
        lit_dst_base,
        lit_dst_len,
        lit_extra,
        litSize,
        isLongOffset,
        LLTable,
        OFTable,
        MLTable,
        entropy_rep,
    )
}

/// Rust-only helper: BMI2 specialization of
/// `ZSTD_decompressSequences_bodySplitLitBuffer_raw_impl::<true>`.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[target_feature(enable = "bmi1,bmi2,lzcnt")]
unsafe fn ZSTD_decompressSequences_bodySplitLitBuffer_bmi2_raw(
    dst: &mut [u8],
    op_start: usize,
    dst_limit: usize,
    ext_history: &[u8],
    seqSrc: &[u8],
    nbSeq: i32,
    lit_dst_base: *const u8,
    lit_dst_len: usize,
    lit_extra: &[u8],
    litSize: usize,
    isLongOffset: ZSTD_longOffset_e,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
) -> usize {
    ZSTD_decompressSequences_bodySplitLitBuffer_raw_impl::<true>(
        dst,
        op_start,
        dst_limit,
        ext_history,
        seqSrc,
        nbSeq,
        lit_dst_base,
        lit_dst_len,
        lit_extra,
        litSize,
        isLongOffset,
        LLTable,
        OFTable,
        MLTable,
        entropy_rep,
    )
}

/// Rust-only helper: BMI2-targeted body of `ZSTD_decompressSequences`.
/// Forwards to `_body_bmi2_rawLit` after splatting the literal pointer.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[target_feature(enable = "bmi1,bmi2,lzcnt")]
unsafe fn ZSTD_decompressSequences_body_bmi2_impl(
    dst: &mut [u8],
    op_start: usize,
    ext_history: &[u8],
    seqSrc: &[u8],
    nbSeq: i32,
    litBuf: &[u8],
    litSize: usize,
    isLongOffset: ZSTD_longOffset_e,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
) -> usize {
    ZSTD_decompressSequences_body_bmi2_rawLit(
        dst,
        op_start,
        ext_history,
        seqSrc,
        nbSeq,
        litBuf.as_ptr(),
        litSize,
        isLongOffset,
        LLTable,
        OFTable,
        MLTable,
        entropy_rep,
    )
}

/// Rust-only helper: raw-literal-pointer BMI2 body. Splits the path
/// based on whether `ext_history` is empty so the hot no-ext-dict path
/// gets its own monomorphized routine.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[target_feature(enable = "bmi1,bmi2,lzcnt")]
unsafe fn ZSTD_decompressSequences_body_bmi2_rawLit(
    dst: &mut [u8],
    op_start: usize,
    ext_history: &[u8],
    seqSrc: &[u8],
    nbSeq: i32,
    lit_base: *const u8,
    litSize: usize,
    isLongOffset: ZSTD_longOffset_e,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
) -> usize {
    if ext_history.is_empty() {
        return ZSTD_decompressSequences_body_bmi2_noExt_rawLit(
            dst,
            op_start,
            seqSrc,
            nbSeq,
            lit_base,
            litSize,
            LLTable.as_ptr(),
            OFTable.as_ptr(),
            MLTable.as_ptr(),
            isLongOffset,
            entropy_rep,
        );
    }

    use crate::common::bitstream::{BIT_DStream_t, BIT_endOfDStream, BIT_initDStream};

    let mut op: usize = op_start;
    let mut litPtr: usize = 0;
    let litEnd = litSize;

    if nbSeq > 0 {
        let mut DStream = BIT_DStream_t::default();
        let mut stateLL = ZSTD_fseState::default();
        let mut stateOffb = ZSTD_fseState::default();
        let mut stateML = ZSTD_fseState::default();
        let mut prevOffset0 = entropy_rep.rep[0] as usize;
        let mut prevOffset1 = entropy_rep.rep[1] as usize;
        let mut prevOffset2 = entropy_rep.rep[2] as usize;

        let rc = BIT_initDStream(&mut DStream, seqSrc, seqSrc.len());
        if crate::common::error::ERR_isError(rc) {
            return ERROR(ErrorCode::CorruptionDetected);
        }

        ZSTD_initFseState(&mut stateLL, &mut DStream, LLTable);
        ZSTD_initFseState(&mut stateOffb, &mut DStream, OFTable);
        ZSTD_initFseState(&mut stateML, &mut DStream, MLTable);

        let mut remaining = nbSeq;
        #[cfg(target_arch = "x86_64")]
        core::arch::asm!(
            ".p2align 6",
            "nop",
            ".p2align 5",
            "nop",
            ".p2align 3",
            options(nomem, nostack, preserves_flags)
        );
        while remaining > 0 {
            let (litLength, matchLength, offset) = ZSTD_decodeSequence_bmi2_inline!(
                DStream,
                stateLL,
                stateOffb,
                stateML,
                prevOffset0,
                prevOffset1,
                prevOffset2,
                LLTable.as_ptr(),
                OFTable.as_ptr(),
                MLTable.as_ptr(),
                isLongOffset,
                remaining == 1
            );
            let oLitEnd = op + litLength;
            let sequenceLength = litLength + matchLength;
            let oneSeqSize = if matchLength != 0
                && litPtr + litLength + 16 <= litSize
                && op + sequenceLength + WILDCOPY_OVERLENGTH <= dst.len()
                && offset <= oLitEnd
            {
                let oneSeqSize = ZSTD_execSequence_rawLit_fast(
                    dst,
                    op,
                    litLength,
                    matchLength,
                    offset,
                    lit_base,
                    litPtr,
                );
                litPtr += litLength;
                oneSeqSize
            } else {
                ZSTD_execSequence_rawLit_fallback(
                    dst,
                    op,
                    seq_t {
                        litLength,
                        matchLength,
                        offset,
                    },
                    ext_history,
                    lit_base,
                    litSize,
                    litPtr,
                    &mut litPtr,
                )
            };
            if crate::common::error::ERR_isError(oneSeqSize) {
                return oneSeqSize;
            }
            op += oneSeqSize;
            remaining -= 1;
        }

        if BIT_endOfDStream(&DStream) == 0 {
            return ERROR(ErrorCode::CorruptionDetected);
        }

        entropy_rep.rep[0] = prevOffset0 as u32;
        entropy_rep.rep[1] = prevOffset1 as u32;
        entropy_rep.rep[2] = prevOffset2 as u32;
    }

    let lastLLSize = litEnd - litPtr;
    if op + lastLLSize > dst.len() {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    if lastLLSize > 0 {
        std::ptr::copy_nonoverlapping(lit_base.add(litPtr), dst.as_mut_ptr().add(op), lastLLSize);
        op += lastLLSize;
    }
    op - op_start
}

/// Rust-only helper: no-ext-dict specialization of the BMI2 sequence
/// decoder body. Hot path for one-shot frames that don't reach into a
/// dictionary or prior streaming-block tail.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[inline(never)]
#[target_feature(enable = "bmi1,bmi2,lzcnt")]
unsafe fn ZSTD_decompressSequences_body_bmi2_noExt_rawLit(
    dst: &mut [u8],
    op_start: usize,
    seqSrc: &[u8],
    nbSeq: i32,
    lit_base: *const u8,
    litSize: usize,
    LLTable: *const ZSTD_seqSymbol,
    OFTable: *const ZSTD_seqSymbol,
    MLTable: *const ZSTD_seqSymbol,
    isLongOffset: ZSTD_longOffset_e,
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
) -> usize {
    use crate::common::bitstream::{BIT_DStream_t, BIT_endOfDStream, BIT_initDStream};

    let mut op: usize = op_start;
    let mut litPtr: usize = 0;
    let dst_ptr = dst.as_mut_ptr();
    let dst_len = dst.len();

    if nbSeq > 0 {
        let mut DStream = BIT_DStream_t::default();
        let mut stateLL = ZSTD_fseState::default();
        let mut stateOffb = ZSTD_fseState::default();
        let mut stateML = ZSTD_fseState::default();
        let mut prevOffset0 = entropy_rep.rep[0] as usize;
        let mut prevOffset1 = entropy_rep.rep[1] as usize;
        let mut prevOffset2 = entropy_rep.rep[2] as usize;

        let rc = BIT_initDStream(&mut DStream, seqSrc, seqSrc.len());
        if crate::common::error::ERR_isError(rc) {
            return ERROR(ErrorCode::CorruptionDetected);
        }

        ZSTD_initFseState(
            &mut stateLL,
            &mut DStream,
            core::slice::from_raw_parts(LLTable, 1),
        );
        ZSTD_initFseState(
            &mut stateOffb,
            &mut DStream,
            core::slice::from_raw_parts(OFTable, 1),
        );
        ZSTD_initFseState(
            &mut stateML,
            &mut DStream,
            core::slice::from_raw_parts(MLTable, 1),
        );

        let mut remaining = nbSeq;
        #[cfg(target_arch = "x86_64")]
        core::arch::asm!(
            ".p2align 6",
            "nop",
            ".p2align 5",
            "nop",
            ".p2align 3",
            options(nomem, nostack, preserves_flags)
        );
        while remaining > 0 {
            let (litLength, matchLength, offset) = ZSTD_decodeSequence_bmi2_inline!(
                DStream,
                stateLL,
                stateOffb,
                stateML,
                prevOffset0,
                prevOffset1,
                prevOffset2,
                LLTable,
                OFTable,
                MLTable,
                isLongOffset,
                remaining == 1
            );
            let oLitEnd = op + litLength;
            let sequenceLength = litLength + matchLength;
            let oneSeqSize = if matchLength != 0
                && litPtr + litLength + 16 <= litSize
                && op + sequenceLength + WILDCOPY_OVERLENGTH <= dst_len
                && offset <= oLitEnd
            {
                let oneSeqSize = ZSTD_execSequence_rawLit_fast_ptr(
                    dst_ptr,
                    op,
                    litLength,
                    matchLength,
                    offset,
                    lit_base,
                    litPtr,
                );
                litPtr += litLength;
                oneSeqSize
            } else {
                ZSTD_execSequence_rawLit_fallback(
                    dst,
                    op,
                    seq_t {
                        litLength,
                        matchLength,
                        offset,
                    },
                    &[],
                    lit_base,
                    litSize,
                    litPtr,
                    &mut litPtr,
                )
            };
            if crate::common::error::ERR_isError(oneSeqSize) {
                return oneSeqSize;
            }
            op += oneSeqSize;
            remaining -= 1;
        }

        if BIT_endOfDStream(&DStream) == 0 {
            return ERROR(ErrorCode::CorruptionDetected);
        }

        entropy_rep.rep[0] = prevOffset0 as u32;
        entropy_rep.rep[1] = prevOffset1 as u32;
        entropy_rep.rep[2] = prevOffset2 as u32;
    }

    let lastLLSize = litSize - litPtr;
    if op + lastLLSize > dst.len() {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    if lastLLSize > 0 {
        std::ptr::copy_nonoverlapping(lit_base.add(litPtr), dst.as_mut_ptr().add(op), lastLLSize);
        op += lastLLSize;
    }
    op - op_start
}

/// Port of `ZSTD_decompressSequences_body_bmi2`. Runtime-dispatches to
/// the BMI2-targeted impl when the CPU advertises the required
/// features, otherwise falls back to the portable
/// `ZSTD_decompressSequences_body`.
fn ZSTD_decompressSequences_body_bmi2(
    dst: &mut [u8],
    op_start: usize,
    ext_history: &[u8],
    seqSrc: &[u8],
    nbSeq: i32,
    litBuf: &[u8],
    litSize: usize,
    isLongOffset: ZSTD_longOffset_e,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
) -> usize {
    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    {
        if std::is_x86_feature_detected!("bmi1")
            && std::is_x86_feature_detected!("bmi2")
            && std::is_x86_feature_detected!("lzcnt")
        {
            return unsafe {
                ZSTD_decompressSequences_body_bmi2_impl(
                    dst,
                    op_start,
                    ext_history,
                    seqSrc,
                    nbSeq,
                    litBuf,
                    litSize,
                    isLongOffset,
                    LLTable,
                    OFTable,
                    MLTable,
                    entropy_rep,
                )
            };
        }
    }

    unsafe {
        ZSTD_decompressSequences_body_rawLit(
            dst,
            op_start,
            ext_history,
            seqSrc,
            nbSeq,
            litBuf.as_ptr(),
            litSize,
            isLongOffset,
            LLTable,
            OFTable,
            MLTable,
            entropy_rep,
        )
    }
}

fn ZSTD_decompressSequences_body_withLongOffset(
    dst: &mut [u8],
    op_start: usize,
    ext_history: &[u8],
    seqSrc: &[u8],
    nbSeq: i32,
    litBuf: &[u8],
    litSize: usize,
    isLongOffset: ZSTD_longOffset_e,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
) -> usize {
    unsafe {
        ZSTD_decompressSequences_body_rawLit(
            dst,
            op_start,
            ext_history,
            seqSrc,
            nbSeq,
            litBuf.as_ptr(),
            litSize,
            isLongOffset,
            LLTable,
            OFTable,
            MLTable,
            entropy_rep,
        )
    }
}

fn ZSTD_decompressSequences_body_bmi2_withLongOffset(
    dst: &mut [u8],
    op_start: usize,
    ext_history: &[u8],
    seqSrc: &[u8],
    nbSeq: i32,
    litBuf: &[u8],
    litSize: usize,
    isLongOffset: ZSTD_longOffset_e,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
) -> usize {
    ZSTD_decompressSequences_body_bmi2(
        dst,
        op_start,
        ext_history,
        seqSrc,
        nbSeq,
        litBuf,
        litSize,
        isLongOffset,
        LLTable,
        OFTable,
        MLTable,
        entropy_rep,
    )
}

/// Default wrapper for `ZSTD_decompressSequences_default`.
pub fn ZSTD_decompressSequences_default(
    dst: &mut [u8],
    op_start: usize,
    ext_history: &[u8],
    seqSrc: &[u8],
    nbSeq: i32,
    litBuf: &[u8],
    litSize: usize,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
) -> usize {
    ZSTD_decompressSequences_body(
        dst,
        op_start,
        ext_history,
        seqSrc,
        nbSeq,
        litBuf,
        litSize,
        LLTable,
        OFTable,
        MLTable,
        entropy_rep,
    )
}

/// Split-literal-buffer body wrapper. The current Rust literal buffer
/// layout doesn't split literals, so it forwards to the common body.
pub fn ZSTD_decompressSequencesSplitLitBuffer_default(
    dst: &mut [u8],
    op_start: usize,
    ext_history: &[u8],
    seqSrc: &[u8],
    nbSeq: i32,
    litBuf: &[u8],
    litSize: usize,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
) -> usize {
    ZSTD_decompressSequences_body(
        dst,
        op_start,
        ext_history,
        seqSrc,
        nbSeq,
        litBuf,
        litSize,
        LLTable,
        OFTable,
        MLTable,
        entropy_rep,
    )
}

/// Port of `ZSTD_prefetchMatch`.
pub fn ZSTD_prefetchMatch(
    mut prefetchPos: usize,
    sequence: seq_t,
    _prefixStart: usize,
    _dictEnd: usize,
) -> usize {
    prefetchPos += sequence.litLength;
    prefetchPos + sequence.matchLength
}

/// Rust-only helper: `_mm_prefetch`-emitting variant of
/// `ZSTD_prefetchMatch`. Hints the match's source bytes into L1 ahead
/// of execution; no-op on non-x86_64 targets.
#[inline]
fn ZSTD_prefetchMatch_rust(
    mut prefetchPos: usize,
    sequence: seq_t,
    dst: &[u8],
    ext_history: &[u8],
) -> usize {
    prefetchPos += sequence.litLength;

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    unsafe {
        use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

        let match_ptr = if sequence.offset > prefetchPos {
            let into_ext = sequence.offset - prefetchPos;
            if into_ext <= ext_history.len() {
                Some(ext_history.as_ptr().add(ext_history.len() - into_ext))
            } else {
                None
            }
        } else {
            let idx = prefetchPos - sequence.offset;
            if idx < dst.len() {
                Some(dst.as_ptr().add(idx))
            } else {
                None
            }
        };

        if let Some(ptr) = match_ptr {
            _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
            _mm_prefetch(ptr.add(64) as *const i8, _MM_HINT_T0);
        }
    }

    prefetchPos + sequence.matchLength
}

/// Long-offset body wrapper. On 64-bit Rust targets the normal decoder
/// can read all legal zstd offsets without a mid-offset refill, but
/// upstream also uses this body as the "prefetch" decoder. It decodes
/// a small ring of sequences ahead of execution to give long-distance
/// matches more time to reach cache.
fn ZSTD_decompressSequencesLong_body_impl<const BMI2: bool>(
    dst: &mut [u8],
    op_start: usize,
    ext_history: &[u8],
    seqSrc: &[u8],
    nbSeq: i32,
    litBuf: &[u8],
    litSize: usize,
    isLongOffset: ZSTD_longOffset_e,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
) -> usize {
    use crate::common::bitstream::{BIT_DStream_t, BIT_endOfDStream, BIT_initDStream};

    const STORED_SEQS: usize = 8;
    const STORED_SEQS_MASK: usize = STORED_SEQS - 1;
    let mut op: usize = op_start;
    let mut litPtr: usize = 0;
    let litEnd = litSize;

    if nbSeq > 0 {
        let mut seqState = seqState_t {
            DStream: BIT_DStream_t::default(),
            stateLL: ZSTD_fseState::default(),
            stateOffb: ZSTD_fseState::default(),
            stateML: ZSTD_fseState::default(),
            prevOffset: [
                entropy_rep.rep[0] as usize,
                entropy_rep.rep[1] as usize,
                entropy_rep.rep[2] as usize,
            ],
        };

        let rc = BIT_initDStream(&mut seqState.DStream, seqSrc, seqSrc.len());
        if crate::common::error::ERR_isError(rc) {
            return ERROR(ErrorCode::CorruptionDetected);
        }

        ZSTD_initFseState(&mut seqState.stateLL, &mut seqState.DStream, LLTable);
        ZSTD_initFseState(&mut seqState.stateOffb, &mut seqState.DStream, OFTable);
        ZSTD_initFseState(&mut seqState.stateML, &mut seqState.DStream, MLTable);

        let total = nbSeq as usize;
        let seqAdvance = total.min(STORED_SEQS);
        let mut sequences = [seq_t::default(); STORED_SEQS];
        let mut seqNb = 0usize;
        let mut prefetchPos = op_start;

        while seqNb < seqAdvance {
            let sequence = if BMI2 {
                #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
                unsafe {
                    ZSTD_decodeSequence_bmi2(
                        &mut seqState,
                        isLongOffset,
                        LLTable,
                        OFTable,
                        MLTable,
                        seqNb + 1 == total,
                    )
                }
                #[cfg(not(all(target_arch = "x86_64", target_pointer_width = "64")))]
                {
                    unreachable!("BMI2 long decoder is x86_64-only")
                }
            } else {
                ZSTD_decodeSequence(
                    &mut seqState,
                    isLongOffset,
                    LLTable,
                    OFTable,
                    MLTable,
                    seqNb + 1 == total,
                )
            };
            prefetchPos = ZSTD_prefetchMatch_rust(prefetchPos, sequence, dst, ext_history);
            sequences[seqNb] = sequence;
            seqNb += 1;
        }

        while seqNb < total {
            let sequence = if BMI2 {
                #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
                unsafe {
                    ZSTD_decodeSequence_bmi2(
                        &mut seqState,
                        isLongOffset,
                        LLTable,
                        OFTable,
                        MLTable,
                        seqNb + 1 == total,
                    )
                }
                #[cfg(not(all(target_arch = "x86_64", target_pointer_width = "64")))]
                {
                    unreachable!("BMI2 long decoder is x86_64-only")
                }
            } else {
                ZSTD_decodeSequence(
                    &mut seqState,
                    isLongOffset,
                    LLTable,
                    OFTable,
                    MLTable,
                    seqNb + 1 == total,
                )
            };
            let execute_idx = (seqNb - STORED_SEQS) & STORED_SEQS_MASK;
            let oneSeqSize = ZSTD_execSequence(
                dst,
                op,
                sequences[execute_idx],
                ext_history,
                litBuf,
                litPtr,
                &mut litPtr,
            );
            if crate::common::error::ERR_isError(oneSeqSize) {
                return oneSeqSize;
            }
            prefetchPos = ZSTD_prefetchMatch_rust(prefetchPos, sequence, dst, ext_history);
            sequences[seqNb & STORED_SEQS_MASK] = sequence;
            op += oneSeqSize;
            seqNb += 1;
        }

        let mut finish = total - seqAdvance;
        while finish < total {
            let sequence = sequences[finish & STORED_SEQS_MASK];
            let oneSeqSize =
                ZSTD_execSequence(dst, op, sequence, ext_history, litBuf, litPtr, &mut litPtr);
            if crate::common::error::ERR_isError(oneSeqSize) {
                return oneSeqSize;
            }
            op += oneSeqSize;
            finish += 1;
        }

        if BIT_endOfDStream(&seqState.DStream) == 0 {
            return ERROR(ErrorCode::CorruptionDetected);
        }

        entropy_rep.rep[0] = seqState.prevOffset[0] as u32;
        entropy_rep.rep[1] = seqState.prevOffset[1] as u32;
        entropy_rep.rep[2] = seqState.prevOffset[2] as u32;
    }

    let lastLLSize = litEnd - litPtr;
    if op + lastLLSize > dst.len() {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    if lastLLSize > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(
                litBuf.as_ptr().add(litPtr),
                dst.as_mut_ptr().add(op),
                lastLLSize,
            );
        }
        op += lastLLSize;
    }

    op - op_start
}

/// Port of `ZSTD_decompressSequencesLong_body`. Portable wrapper that
/// delegates to the const-generic `_body_impl::<false>` long decoder.
pub fn ZSTD_decompressSequencesLong_body(
    dst: &mut [u8],
    op_start: usize,
    ext_history: &[u8],
    seqSrc: &[u8],
    nbSeq: i32,
    litBuf: &[u8],
    litSize: usize,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
) -> usize {
    ZSTD_decompressSequencesLong_body_impl::<false>(
        dst,
        op_start,
        ext_history,
        seqSrc,
        nbSeq,
        litBuf,
        litSize,
        ZSTD_longOffset_e::ZSTD_lo_isRegularOffset,
        LLTable,
        OFTable,
        MLTable,
        entropy_rep,
    )
}

/// Rust-only helper: split-literal-buffer long-prefetch body. The
/// const-generic `BMI2` flag picks between the portable and BMI2
/// per-sequence decoders; ring-buffers 8 sequences ahead so the match
/// loads have time to reach L1 before execution.
fn ZSTD_decompressSequencesLong_bodySplitLitBuffer_impl<const BMI2: bool>(
    dst: &mut [u8],
    op_start: usize,
    dst_limit: usize,
    ext_history: &[u8],
    seqSrc: &[u8],
    nbSeq: i32,
    lit_dst_base: *const u8,
    lit_dst_len: usize,
    lit_extra: &[u8],
    litSize: usize,
    isLongOffset: ZSTD_longOffset_e,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
) -> usize {
    use crate::common::bitstream::{BIT_DStream_t, BIT_endOfDStream, BIT_initDStream};

    const STORED_SEQS: usize = 8;
    const STORED_SEQS_MASK: usize = STORED_SEQS - 1;
    let mut op = op_start;
    let mut litPtr = 0usize;
    let mut extraPtr = 0usize;
    let mut using_extra = false;

    if nbSeq > 0 {
        let mut seqState = seqState_t {
            DStream: BIT_DStream_t::default(),
            stateLL: ZSTD_fseState::default(),
            stateOffb: ZSTD_fseState::default(),
            stateML: ZSTD_fseState::default(),
            prevOffset: [
                entropy_rep.rep[0] as usize,
                entropy_rep.rep[1] as usize,
                entropy_rep.rep[2] as usize,
            ],
        };

        let rc = BIT_initDStream(&mut seqState.DStream, seqSrc, seqSrc.len());
        if crate::common::error::ERR_isError(rc) {
            return ERROR(ErrorCode::CorruptionDetected);
        }
        ZSTD_initFseState(&mut seqState.stateLL, &mut seqState.DStream, LLTable);
        ZSTD_initFseState(&mut seqState.stateOffb, &mut seqState.DStream, OFTable);
        ZSTD_initFseState(&mut seqState.stateML, &mut seqState.DStream, MLTable);

        let total = nbSeq as usize;
        let seqAdvance = total.min(STORED_SEQS);
        let mut sequences = [seq_t::default(); STORED_SEQS];
        let mut seqNb = 0usize;
        let mut prefetchPos = op_start;

        while seqNb < seqAdvance {
            let sequence = if BMI2 {
                #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
                unsafe {
                    ZSTD_decodeSequence_bmi2(
                        &mut seqState,
                        isLongOffset,
                        LLTable,
                        OFTable,
                        MLTable,
                        seqNb + 1 == total,
                    )
                }
                #[cfg(not(all(target_arch = "x86_64", target_pointer_width = "64")))]
                {
                    unreachable!("BMI2 long split decoder is x86_64-only")
                }
            } else {
                ZSTD_decodeSequence(
                    &mut seqState,
                    isLongOffset,
                    LLTable,
                    OFTable,
                    MLTable,
                    seqNb + 1 == total,
                )
            };
            prefetchPos = ZSTD_prefetchMatch_rust(prefetchPos, sequence, dst, ext_history);
            sequences[seqNb] = sequence;
            seqNb += 1;
        }

        while seqNb < total {
            let sequence = if BMI2 {
                #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
                unsafe {
                    ZSTD_decodeSequence_bmi2(
                        &mut seqState,
                        isLongOffset,
                        LLTable,
                        OFTable,
                        MLTable,
                        seqNb + 1 == total,
                    )
                }
                #[cfg(not(all(target_arch = "x86_64", target_pointer_width = "64")))]
                {
                    unreachable!("BMI2 long split decoder is x86_64-only")
                }
            } else {
                ZSTD_decodeSequence(
                    &mut seqState,
                    isLongOffset,
                    LLTable,
                    OFTable,
                    MLTable,
                    seqNb + 1 == total,
                )
            };
            let execute_idx = (seqNb - STORED_SEQS) & STORED_SEQS_MASK;
            let mut queued = sequences[execute_idx];
            let oneSeqSize = unsafe {
                ZSTD_execSequenceSplitLitBuffer_raw(
                    dst,
                    dst_limit,
                    op,
                    &mut queued,
                    ext_history,
                    lit_dst_base,
                    lit_dst_len,
                    lit_extra,
                    &mut litPtr,
                    &mut extraPtr,
                    &mut using_extra,
                )
            };
            if crate::common::error::ERR_isError(oneSeqSize) {
                return oneSeqSize;
            }
            prefetchPos = ZSTD_prefetchMatch_rust(prefetchPos, sequence, dst, ext_history);
            sequences[seqNb & STORED_SEQS_MASK] = sequence;
            op += oneSeqSize;
            seqNb += 1;
        }

        let mut finish = total - seqAdvance;
        while finish < total {
            let mut sequence = sequences[finish & STORED_SEQS_MASK];
            let oneSeqSize = unsafe {
                ZSTD_execSequenceSplitLitBuffer_raw(
                    dst,
                    dst_limit,
                    op,
                    &mut sequence,
                    ext_history,
                    lit_dst_base,
                    lit_dst_len,
                    lit_extra,
                    &mut litPtr,
                    &mut extraPtr,
                    &mut using_extra,
                )
            };
            if crate::common::error::ERR_isError(oneSeqSize) {
                return oneSeqSize;
            }
            op += oneSeqSize;
            finish += 1;
        }

        if BIT_endOfDStream(&seqState.DStream) == 0 {
            return ERROR(ErrorCode::CorruptionDetected);
        }
        entropy_rep.rep[0] = seqState.prevOffset[0] as u32;
        entropy_rep.rep[1] = seqState.prevOffset[1] as u32;
        entropy_rep.rep[2] = seqState.prevOffset[2] as u32;
    }

    unsafe {
        if !using_extra {
            let last = lit_dst_len - litPtr;
            if op + last > dst.len() {
                return ERROR(ErrorCode::DstSizeTooSmall);
            }
            if last != 0 {
                std::ptr::copy(lit_dst_base.add(litPtr), dst.as_mut_ptr().add(op), last);
                op += last;
            }
            extraPtr = 0;
        }
        let last = litSize - lit_dst_len - extraPtr;
        if op + last > dst.len() {
            return ERROR(ErrorCode::DstSizeTooSmall);
        }
        if last != 0 {
            std::ptr::copy_nonoverlapping(
                lit_extra.as_ptr().add(extraPtr),
                dst.as_mut_ptr().add(op),
                last,
            );
            op += last;
        }
    }

    op - op_start
}

/// Rust-only helper: BMI2-targeted long-prefetch body. Forwards to
/// `_body_impl::<true>`.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[target_feature(enable = "bmi1,bmi2,lzcnt")]
unsafe fn ZSTD_decompressSequencesLong_body_bmi2_impl(
    dst: &mut [u8],
    op_start: usize,
    ext_history: &[u8],
    seqSrc: &[u8],
    nbSeq: i32,
    litBuf: &[u8],
    litSize: usize,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
) -> usize {
    ZSTD_decompressSequencesLong_body_impl::<true>(
        dst,
        op_start,
        ext_history,
        seqSrc,
        nbSeq,
        litBuf,
        litSize,
        ZSTD_longOffset_e::ZSTD_lo_isRegularOffset,
        LLTable,
        OFTable,
        MLTable,
        entropy_rep,
    )
}

/// Port of `ZSTD_decompressSequencesLong_default`. Portable
/// long-prefetch entry; forwards to `ZSTD_decompressSequencesLong_body`.
pub fn ZSTD_decompressSequencesLong_default(
    dst: &mut [u8],
    op_start: usize,
    ext_history: &[u8],
    seqSrc: &[u8],
    nbSeq: i32,
    litBuf: &[u8],
    litSize: usize,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
) -> usize {
    ZSTD_decompressSequencesLong_body(
        dst,
        op_start,
        ext_history,
        seqSrc,
        nbSeq,
        litBuf,
        litSize,
        LLTable,
        OFTable,
        MLTable,
        entropy_rep,
    )
}

/// Port of `ZSTD_decompressSequences_bmi2`. BMI2 entry point that runs
/// the BMI2-targeted body when the CPU supports it.
pub fn ZSTD_decompressSequences_bmi2(
    dst: &mut [u8],
    op_start: usize,
    ext_history: &[u8],
    seqSrc: &[u8],
    nbSeq: i32,
    litBuf: &[u8],
    litSize: usize,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
) -> usize {
    ZSTD_decompressSequences_body_bmi2(
        dst,
        op_start,
        ext_history,
        seqSrc,
        nbSeq,
        litBuf,
        litSize,
        ZSTD_longOffset_e::ZSTD_lo_isRegularOffset,
        LLTable,
        OFTable,
        MLTable,
        entropy_rep,
    )
}

/// Port of `ZSTD_decompressSequencesSplitLitBuffer_bmi2`. The Rust
/// literal layout doesn't actually split, so it forwards to the
/// regular BMI2 body.
pub fn ZSTD_decompressSequencesSplitLitBuffer_bmi2(
    dst: &mut [u8],
    op_start: usize,
    ext_history: &[u8],
    seqSrc: &[u8],
    nbSeq: i32,
    litBuf: &[u8],
    litSize: usize,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
) -> usize {
    ZSTD_decompressSequences_body_bmi2(
        dst,
        op_start,
        ext_history,
        seqSrc,
        nbSeq,
        litBuf,
        litSize,
        ZSTD_longOffset_e::ZSTD_lo_isRegularOffset,
        LLTable,
        OFTable,
        MLTable,
        entropy_rep,
    )
}

/// Port of `ZSTD_decompressSequencesLong_bmi2`. BMI2 long-prefetch
/// entry point; runtime-dispatches to the BMI2 impl when supported.
pub fn ZSTD_decompressSequencesLong_bmi2(
    dst: &mut [u8],
    op_start: usize,
    ext_history: &[u8],
    seqSrc: &[u8],
    nbSeq: i32,
    litBuf: &[u8],
    litSize: usize,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
) -> usize {
    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    {
        if std::is_x86_feature_detected!("bmi1")
            && std::is_x86_feature_detected!("bmi2")
            && std::is_x86_feature_detected!("lzcnt")
        {
            return unsafe {
                ZSTD_decompressSequencesLong_body_bmi2_impl(
                    dst,
                    op_start,
                    ext_history,
                    seqSrc,
                    nbSeq,
                    litBuf,
                    litSize,
                    LLTable,
                    OFTable,
                    MLTable,
                    entropy_rep,
                )
            };
        }
    }
    ZSTD_decompressSequencesLong_body(
        dst,
        op_start,
        ext_history,
        seqSrc,
        nbSeq,
        litBuf,
        litSize,
        LLTable,
        OFTable,
        MLTable,
        entropy_rep,
    )
}

/// Port of `ZSTD_decompressSequences`. Public-facing portable entry
/// point — forwards to `ZSTD_decompressSequences_default`.
pub fn ZSTD_decompressSequences(
    dst: &mut [u8],
    op_start: usize,
    ext_history: &[u8],
    seqSrc: &[u8],
    nbSeq: i32,
    litBuf: &[u8],
    litSize: usize,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
) -> usize {
    ZSTD_decompressSequences_default(
        dst,
        op_start,
        ext_history,
        seqSrc,
        nbSeq,
        litBuf,
        litSize,
        LLTable,
        OFTable,
        MLTable,
        entropy_rep,
    )
}

/// Port of `ZSTD_decompressSequencesSplitLitBuffer`. Public split-buffer
/// entry; the Rust layout doesn't split, so forwards to the default.
pub fn ZSTD_decompressSequencesSplitLitBuffer(
    dst: &mut [u8],
    op_start: usize,
    ext_history: &[u8],
    seqSrc: &[u8],
    nbSeq: i32,
    litBuf: &[u8],
    litSize: usize,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
) -> usize {
    ZSTD_decompressSequencesSplitLitBuffer_default(
        dst,
        op_start,
        ext_history,
        seqSrc,
        nbSeq,
        litBuf,
        litSize,
        LLTable,
        OFTable,
        MLTable,
        entropy_rep,
    )
}

/// Port of `ZSTD_decompressSequencesLong`. Public long-prefetch entry
/// point — forwards to `ZSTD_decompressSequencesLong_default`.
pub fn ZSTD_decompressSequencesLong(
    dst: &mut [u8],
    op_start: usize,
    ext_history: &[u8],
    seqSrc: &[u8],
    nbSeq: i32,
    litBuf: &[u8],
    litSize: usize,
    LLTable: &[ZSTD_seqSymbol],
    OFTable: &[ZSTD_seqSymbol],
    MLTable: &[ZSTD_seqSymbol],
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
) -> usize {
    ZSTD_decompressSequencesLong_default(
        dst,
        op_start,
        ext_history,
        seqSrc,
        nbSeq,
        litBuf,
        litSize,
        LLTable,
        OFTable,
        MLTable,
        entropy_rep,
    )
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ZSTD_OffsetInfo {
    pub longOffsetShare: u32,
    pub maxNbAdditionalBits: u32,
}

/// Port of `ZSTD_getOffsetInfo`.
pub fn ZSTD_getOffsetInfo(offTable: &[ZSTD_seqSymbol], nbSeq: i32) -> ZSTD_OffsetInfo {
    let mut info = ZSTD_OffsetInfo::default();
    if nbSeq != 0 {
        let tableLog = seq_header_read(offTable).tableLog;
        let max = 1usize << tableLog;
        for entry in offTable.iter().skip(1).take(max) {
            info.maxNbAdditionalBits = info.maxNbAdditionalBits.max(entry.nbAdditionalBits as u32);
            if entry.nbAdditionalBits > 22 {
                info.longOffsetShare += 1;
            }
        }
        info.longOffsetShare <<= OffFSELog - tableLog;
    }
    info
}

/// Port of `ZSTD_decompressBlock_internal`. Decompresses one
/// `bt_compressed` zstd block from `src` into `dst`, returning the
/// number of bytes written.
///
/// Flow: 1) parse literals-block header + payload via
/// `ZSTD_decodeLiteralsBlock`; 2) parse sequence-block header + build
/// LL/OFF/ML FSE tables via `ZSTD_decodeSeqHeaders`; 3) run the
/// decode-exec loop via `ZSTD_decompressSequences_body`.
///
/// Upstream supports additional code paths here (prefetch decoder,
/// split-lit-buffer path, long-offsets 32-bit path) — we dispatch
/// only to the body-variant decoder for v0.1.
pub fn ZSTD_decompressBlock_internal(
    dctx: &mut ZSTD_DCtx,
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
    dst: &mut [u8],
    op_start: usize,
    src: &[u8],
    streaming: streaming_operation,
) -> usize {
    ZSTD_decompressBlock_internal_with_ext_history(
        dctx,
        entropy_rep,
        dst,
        op_start,
        src,
        streaming,
        None,
    )
}

/// Rust-only variant of `ZSTD_decompressBlock_internal` that lets a
/// caller provide the prior contiguous history segment explicitly.
/// This is used by the CLI output ring: the current prefix lives in
/// `dst[..op_start]`, while the previous wrapped segment is borrowed
/// as `ext_history_override`, matching upstream's virtual
/// `[dict][prefix]` layout without copying the ring tail into
/// `dctx.historyBuffer`.
pub fn ZSTD_decompressBlock_internal_with_ext_history(
    dctx: &mut ZSTD_DCtx,
    entropy_rep: &mut ZSTD_decoder_entropy_rep,
    dst: &mut [u8],
    op_start: usize,
    src: &[u8],
    streaming: streaming_operation,
    ext_history_override: Option<&[u8]>,
) -> usize {
    #[inline]
    fn finish_block(dctx: &mut ZSTD_DCtx, nbSeq: i32, rc: usize) -> usize {
        if nbSeq > 0 {
            dctx.fseEntropy = 1;
        }
        rc
    }

    let dstCapacity = dst.len().saturating_sub(op_start);
    let mut srcSize = src.len();
    if srcSize > ZSTD_blockSizeMax(dctx) {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    if srcSize < MIN_CBLOCK_SIZE {
        return ERROR(ErrorCode::CorruptionDetected);
    }

    // 1. Decode literals block header + payload. Literals are buffered
    //    in the DCtx's owned scratch; the `dst` slice is only used by
    //    the future spill-into-dst optimization, so we feed it the
    //    remaining-capacity slice starting at op_start.
    let (_, dst_tail) = dst.split_at_mut(op_start);
    let litEncType = SymbolEncodingType_e::from_bits(src[0] as u32);
    let litCSize = if litEncType == SymbolEncodingType_e::set_basic {
        ZSTD_decodeBasicLiteralsBlock_direct(dctx, src, dst_tail.len())
    } else {
        ZSTD_decodeLiteralsBlock(dctx, src, dst_tail, streaming)
    };
    if crate::common::error::ERR_isError(litCSize) {
        return litCSize;
    }
    let direct_lit_start = if litEncType == SymbolEncodingType_e::set_basic {
        litCSize - dctx.litSize
    } else {
        0
    };
    let mut ip: usize = litCSize;
    srcSize -= litCSize;

    // 2. Decode sequence-section header + build LL/OFF/ML FSE tables.
    let mut nbSeq: i32 = 0;
    let seqHSize = ZSTD_decodeSeqHeaders(dctx, &mut nbSeq, &src[ip..ip + srcSize]);
    if crate::common::error::ERR_isError(seqHSize) {
        return seqHSize;
    }
    ip += seqHSize;
    srcSize -= seqHSize;

    if nbSeq > 0 && (dst.is_empty() || dstCapacity == 0) {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }

    let blockSizeMax = dstCapacity.min(ZSTD_blockSizeMax(dctx));
    let override_history_len = ext_history_override.map_or(0, |h| h.len());
    let totalHistorySize = dctx.stream_dict.len()
        + override_history_len.max(dctx.historyBuffer.len())
        + op_start
        + blockSizeMax;
    let mut isLongOffset =
        if crate::common::mem::MEM_32bits() != 0 && totalHistorySize > ZSTD_maxShortOffset() {
            ZSTD_longOffset_e::ZSTD_lo_isLongOffset
        } else {
            ZSTD_longOffset_e::ZSTD_lo_isRegularOffset
        };
    let mut usePrefetchDecoder = dctx.ddictIsCold != 0;
    if isLongOffset == ZSTD_longOffset_e::ZSTD_lo_isLongOffset
        || (!usePrefetchDecoder && totalHistorySize > (1usize << 24) && nbSeq > 8)
    {
        let info = ZSTD_getOffsetInfo(
            active_of_table_from(&dctx.OFTable, dctx.of_default_active),
            nbSeq,
        );
        if isLongOffset == ZSTD_longOffset_e::ZSTD_lo_isLongOffset
            && info.maxNbAdditionalBits <= STREAM_ACCUMULATOR_MIN_32
        {
            isLongOffset = ZSTD_longOffset_e::ZSTD_lo_isRegularOffset;
        }
        if !usePrefetchDecoder {
            let minShare = if crate::common::mem::MEM_64bits() != 0 {
                7
            } else {
                20
            };
            usePrefetchDecoder = info.longOffsetShare >= minShare;
        }
    }
    dctx.ddictIsCold = 0;

    let LL = active_ll_table_from(&dctx.LLTable, dctx.ll_default_active);
    let OF = active_of_table_from(&dctx.OFTable, dctx.of_default_active);
    let ML = active_ml_table_from(&dctx.MLTable, dctx.ml_default_active);

    // Build the ext-dict history slice. The virtual layout the
    // sequence decoder reasons about is:
    //
    //     [ stream_dict ][ historyBuffer ][ dst[..op_start] ][ this block ]
    //
    // - `stream_dict` is the loaded raw-content dictionary (if any).
    // - `historyBuffer` is the rolling output of prior blocks decoded
    //   via `ZSTD_decompressContinue`, retained outside `dst`.
    // - `dst[..op_start]` is what the caller has put in front of us
    //   (e.g. the dict concatenated by `ZSTD_decompress_usingDict`,
    //   or earlier blocks of the same frame for `ZSTD_decompressFrame`).
    //
    let cap = if dctx.fParams.windowSize > 0 {
        dctx.fParams.windowSize as usize
    } else {
        ZSTD_BLOCKSIZE_MAX
    };
    let ext_history_joined;
    let ext_history = if let Some(history) = ext_history_override {
        if dctx.stream_dict.is_empty() {
            history
        } else {
            let history_len = history.len().min(cap);
            let dict_len = dctx.stream_dict.len().min(cap.saturating_sub(history_len));
            let mut v = Vec::with_capacity(dict_len + history_len);
            v.extend_from_slice(&dctx.stream_dict[dctx.stream_dict.len() - dict_len..]);
            v.extend_from_slice(&history[history.len() - history_len..]);
            ext_history_joined = v;
            ext_history_joined.as_slice()
        }
    } else if dctx.stream_dict.is_empty() {
        // Hot streaming path: historyBuffer is not mutated until this
        // block has finished regenerating, so avoid cloning it for each
        // block. Expose only the valid window suffix; the backing Vec
        // may keep older bytes to avoid shifting on every block. The
        // raw slice sidesteps an immutable borrow of `dctx` while the
        // decoder updates entropy tables below.
        let start = dctx.historyBuffer.len().saturating_sub(cap);
        unsafe {
            std::slice::from_raw_parts(
                dctx.historyBuffer.as_ptr().add(start),
                dctx.historyBuffer.len() - start,
            )
        }
    } else {
        let history_len = dctx.historyBuffer.len().min(cap);
        let dict_len = dctx.stream_dict.len().min(cap.saturating_sub(history_len));
        let mut v = Vec::with_capacity(dict_len + history_len);
        v.extend_from_slice(&dctx.stream_dict[dctx.stream_dict.len() - dict_len..]);
        v.extend_from_slice(&dctx.historyBuffer[dctx.historyBuffer.len() - history_len..]);
        ext_history_joined = v;
        ext_history_joined.as_slice()
    };

    // 3. Regenerate sequences into dst. Keep literal and FSE table
    // snapshots detached from `dctx` so the hot sequence loop gets
    // non-aliasing inputs without per-block heap table allocation.
    let litSize = dctx.litSize;
    let lit_snapshot = if litEncType == SymbolEncodingType_e::set_basic {
        Some(&src[direct_lit_start..direct_lit_start + litSize])
    } else if dctx.litPtr_from_dst {
        None
    } else {
        Some(&dctx.litExtraBuffer[..litSize])
    };
    if dctx.litBufferLocation == ZSTD_litLocation_e::ZSTD_split {
        let lit_abs = op_start + dctx.litPtr_offset;
        let lit_dst_len = dctx.litBufferEnd_offset - dctx.litBuffer_offset;
        let out_limit = op_start + dctx.litBuffer_offset;
        let dst_ptr = dst.as_mut_ptr();
        let lit_base = unsafe { dst_ptr.add(lit_abs) as *const u8 };
        if usePrefetchDecoder {
            #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
            if ZSTD_DCtx_get_bmi2(dctx) != 0
                && std::is_x86_feature_detected!("bmi1")
                && std::is_x86_feature_detected!("bmi2")
                && std::is_x86_feature_detected!("lzcnt")
            {
                let rc = ZSTD_decompressSequencesLong_bodySplitLitBuffer_impl::<true>(
                    dst,
                    op_start,
                    out_limit,
                    ext_history,
                    &src[ip..ip + srcSize],
                    nbSeq,
                    lit_base,
                    lit_dst_len,
                    &dctx.litExtraBuffer[..ZSTD_LITBUFFEREXTRASIZE],
                    litSize,
                    isLongOffset,
                    LL,
                    OF,
                    ML,
                    entropy_rep,
                );
                return finish_block(dctx, nbSeq, rc);
            }
            let rc = ZSTD_decompressSequencesLong_bodySplitLitBuffer_impl::<false>(
                dst,
                op_start,
                out_limit,
                ext_history,
                &src[ip..ip + srcSize],
                nbSeq,
                lit_base,
                lit_dst_len,
                &dctx.litExtraBuffer[..ZSTD_LITBUFFEREXTRASIZE],
                litSize,
                isLongOffset,
                LL,
                OF,
                ML,
                entropy_rep,
            );
            return finish_block(dctx, nbSeq, rc);
        }
        #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
        if ZSTD_DCtx_get_bmi2(dctx) != 0
            && !usePrefetchDecoder
            && std::is_x86_feature_detected!("bmi1")
            && std::is_x86_feature_detected!("bmi2")
            && std::is_x86_feature_detected!("lzcnt")
        {
            let rc = unsafe {
                ZSTD_decompressSequences_bodySplitLitBuffer_bmi2_raw(
                    dst,
                    op_start,
                    out_limit,
                    ext_history,
                    &src[ip..ip + srcSize],
                    nbSeq,
                    lit_base,
                    lit_dst_len,
                    &dctx.litExtraBuffer[..ZSTD_LITBUFFEREXTRASIZE],
                    litSize,
                    isLongOffset,
                    LL,
                    OF,
                    ML,
                    entropy_rep,
                )
            };
            return finish_block(dctx, nbSeq, rc);
        }
        let rc = unsafe {
            ZSTD_decompressSequences_bodySplitLitBuffer_raw(
                dst,
                op_start,
                out_limit,
                ext_history,
                &src[ip..ip + srcSize],
                nbSeq,
                lit_base,
                lit_dst_len,
                &dctx.litExtraBuffer[..ZSTD_LITBUFFEREXTRASIZE],
                litSize,
                isLongOffset,
                LL,
                OF,
                ML,
                entropy_rep,
            )
        };
        return finish_block(dctx, nbSeq, rc);
    }

    if dctx.litPtr_from_dst {
        let lit_abs = op_start + dctx.litPtr_offset;
        let out_limit = op_start + dctx.litBuffer_offset;
        let dst_ptr = dst.as_mut_ptr();
        let lit_base = unsafe { dst_ptr.add(lit_abs) as *const u8 };
        let dst_limited = unsafe { std::slice::from_raw_parts_mut(dst_ptr, out_limit) };
        if usePrefetchDecoder {
            debug_assert_eq!(ZSTD_DCtx_get_bmi2(dctx), 0);
            // Raw-literal long-body dispatch is still scalar here; keep
            // prefetch sequencing for correctness of the chosen control
            // flow, but avoid manufacturing a Rust slice alias into dst.
        }
        #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
        if ZSTD_DCtx_get_bmi2(dctx) != 0
            && !usePrefetchDecoder
            && std::is_x86_feature_detected!("bmi1")
            && std::is_x86_feature_detected!("bmi2")
            && std::is_x86_feature_detected!("lzcnt")
        {
            let rc = unsafe {
                ZSTD_decompressSequences_body_bmi2_rawLit(
                    dst_limited,
                    op_start,
                    ext_history,
                    &src[ip..ip + srcSize],
                    nbSeq,
                    lit_base,
                    litSize,
                    isLongOffset,
                    LL,
                    OF,
                    ML,
                    entropy_rep,
                )
            };
            return finish_block(dctx, nbSeq, rc);
        }
        let rc = unsafe {
            ZSTD_decompressSequences_body_rawLit(
                dst_limited,
                op_start,
                ext_history,
                &src[ip..ip + srcSize],
                nbSeq,
                lit_base,
                litSize,
                isLongOffset,
                LL,
                OF,
                ML,
                entropy_rep,
            )
        };
        return finish_block(dctx, nbSeq, rc);
    }

    let lit_snapshot = lit_snapshot.expect("non-dst literal buffer must be available");
    if usePrefetchDecoder {
        #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
        if ZSTD_DCtx_get_bmi2(dctx) != 0
            && std::is_x86_feature_detected!("bmi1")
            && std::is_x86_feature_detected!("bmi2")
            && std::is_x86_feature_detected!("lzcnt")
        {
            let rc = ZSTD_decompressSequencesLong_body_impl::<true>(
                dst,
                op_start,
                ext_history,
                &src[ip..ip + srcSize],
                nbSeq,
                lit_snapshot,
                litSize,
                isLongOffset,
                LL,
                OF,
                ML,
                entropy_rep,
            );
            return finish_block(dctx, nbSeq, rc);
        }
        let rc = ZSTD_decompressSequencesLong_body_impl::<false>(
            dst,
            op_start,
            ext_history,
            &src[ip..ip + srcSize],
            nbSeq,
            lit_snapshot,
            litSize,
            isLongOffset,
            LL,
            OF,
            ML,
            entropy_rep,
        );
        return finish_block(dctx, nbSeq, rc);
    }
    if ZSTD_DCtx_get_bmi2(dctx) != 0 {
        let rc = ZSTD_decompressSequences_body_bmi2_withLongOffset(
            dst,
            op_start,
            ext_history,
            &src[ip..ip + srcSize],
            nbSeq,
            lit_snapshot,
            litSize,
            isLongOffset,
            LL,
            OF,
            ML,
            entropy_rep,
        );
        return finish_block(dctx, nbSeq, rc);
    }
    let rc = ZSTD_decompressSequences_body_withLongOffset(
        dst,
        op_start,
        ext_history,
        &src[ip..ip + srcSize],
        nbSeq,
        lit_snapshot,
        litSize,
        isLongOffset,
        LL,
        OF,
        ML,
        entropy_rep,
    );
    finish_block(dctx, nbSeq, rc)
}

/// Port of `ZSTD_decodeLiteralsBlock_wrapper` (`zstd_decompress_block.c:342`).
/// Standalone-block variant: sets `isFrameDecompression = 0` and
/// forwards to `ZSTD_decodeLiteralsBlock` with `streaming =
/// not_streaming`.
pub fn ZSTD_decodeLiteralsBlock_wrapper(dctx: &mut ZSTD_DCtx, src: &[u8], dst: &mut [u8]) -> usize {
    dctx.isFrameDecompression = 0;
    ZSTD_decodeLiteralsBlock(dctx, src, dst, streaming_operation::not_streaming)
}

/// Rust-only helper: directly materializes a raw or RLE literals block
/// into the DCtx's `litExtraBuffer`. Split out of
/// `ZSTD_decodeLiteralsBlock` so the slow-path branches stay readable.
fn ZSTD_decodeBasicLiteralsBlock_direct(
    dctx: &mut ZSTD_DCtx,
    src: &[u8],
    dstCapacity: usize,
) -> usize {
    let srcSize = src.len();
    if srcSize < MIN_CBLOCK_SIZE {
        return ERROR(ErrorCode::CorruptionDetected);
    }
    if SymbolEncodingType_e::from_bits(src[0] as u32) != SymbolEncodingType_e::set_basic {
        return ERROR(ErrorCode::CorruptionDetected);
    }

    let lhlCode = (src[0] >> 2) & 3;
    let (lhSize, litSize) = match lhlCode {
        0 | 2 => (1usize, (src[0] >> 3) as usize),
        1 => {
            if srcSize < 2 {
                return ERROR(ErrorCode::CorruptionDetected);
            }
            (2usize, (MEM_readLE16(&src[..2]) >> 4) as usize)
        }
        _ => {
            if srcSize < 3 {
                return ERROR(ErrorCode::CorruptionDetected);
            }
            (3usize, (MEM_readLE24(&src[..3]) >> 4) as usize)
        }
    };

    if litSize > ZSTD_blockSizeMax(dctx) {
        return ERROR(ErrorCode::CorruptionDetected);
    }
    if ZSTD_blockSizeMax(dctx).min(dstCapacity) < litSize {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    if litSize + lhSize > srcSize {
        return ERROR(ErrorCode::CorruptionDetected);
    }

    dctx.litPtr_from_dst = false;
    dctx.litPtr_offset = 0;
    dctx.litSize = litSize;
    dctx.litBufferEnd_offset = litSize;
    dctx.litBufferLocation = ZSTD_litLocation_e::ZSTD_not_in_dst;
    lhSize + litSize
}

/// Port of `ZSTD_decodeLiteralsBlock`. Decodes a literals-block header,
/// populates `dctx.litBuffer` / `litSize` / `litPtr` and returns the
/// number of bytes consumed from `src`.
///
/// Scope note: `set_repeat` and `set_compressed` (Huffman-compressed)
/// branches are implemented; the literal buffer is always placed in
/// `dctx.litExtraBuffer` for now (the spill-into-dst optimization will
/// land with `ZSTD_execSequence`).
pub fn ZSTD_decodeLiteralsBlock(
    dctx: &mut ZSTD_DCtx,
    src: &[u8],
    dst: &mut [u8],
    streaming: streaming_operation,
) -> usize {
    let srcSize = src.len();
    let dstCapacity = dst.len();
    if srcSize < MIN_CBLOCK_SIZE {
        return ERROR(ErrorCode::CorruptionDetected);
    }

    let litEncType = SymbolEncodingType_e::from_bits(src[0] as u32);
    let blockSizeMax = ZSTD_blockSizeMax(dctx);

    match litEncType {
        SymbolEncodingType_e::set_repeat | SymbolEncodingType_e::set_compressed => {
            if litEncType == SymbolEncodingType_e::set_repeat && dctx.litEntropy == 0 {
                return ERROR(ErrorCode::DictionaryCorrupted);
            }
            if srcSize < 5 {
                return ERROR(ErrorCode::CorruptionDetected);
            }

            let lhlCode = (src[0] >> 2) & 3;
            let (lhSize, litSize, litCSize, singleStream) = match lhlCode {
                0 | 1 => {
                    // 2 - 2 - 10 - 10
                    if srcSize < 3 {
                        return ERROR(ErrorCode::CorruptionDetected);
                    }
                    let lhc = MEM_readLE24(&src[..3]);
                    let singleStream = lhlCode == 0;
                    let lhSize = 3usize;
                    let litSize = ((lhc >> 4) & 0x3FF) as usize;
                    let litCSize = ((lhc >> 14) & 0x3FF) as usize;
                    (lhSize, litSize, litCSize, singleStream)
                }
                2 => {
                    // 2 - 2 - 14 - 14
                    if srcSize < 4 {
                        return ERROR(ErrorCode::CorruptionDetected);
                    }
                    let lhc = MEM_readLE32(&src[..4]);
                    let lhSize = 4usize;
                    let litSize = ((lhc >> 4) & 0x3FFF) as usize;
                    let litCSize = (lhc >> 18) as usize;
                    (lhSize, litSize, litCSize, false)
                }
                _ => {
                    // 3: 2 - 2 - 18 - 18
                    if srcSize < 5 {
                        return ERROR(ErrorCode::CorruptionDetected);
                    }
                    let lhc = MEM_readLE32(&src[..4]);
                    let lhSize = 5usize;
                    let litSize = ((lhc >> 4) & 0x3FFFF) as usize;
                    let litCSize = ((lhc >> 22) as usize) + ((src[4] as usize) << 10);
                    (lhSize, litSize, litCSize, false)
                }
            };

            if litSize > blockSizeMax {
                return ERROR(ErrorCode::CorruptionDetected);
            }
            if !singleStream && litSize < MIN_LITERALS_FOR_4_STREAMS {
                return ERROR(ErrorCode::LiteralsHeaderWrong);
            }
            if litCSize + lhSize > srcSize {
                return ERROR(ErrorCode::CorruptionDetected);
            }
            let expectedWriteSize = blockSizeMax.min(dstCapacity);
            if expectedWriteSize < litSize {
                return ERROR(ErrorCode::DstSizeTooSmall);
            }

            ZSTD_allocateLiteralsBuffer(
                dctx,
                dst.len(),
                dstCapacity,
                litSize,
                streaming,
                expectedWriteSize,
                0,
            );

            let huf_flags = if ZSTD_DCtx_get_bmi2(dctx) != 0 {
                HUF_flags_bmi2
            } else {
                0
            } | if dctx.disableHufAsm != 0 {
                HUF_flags_disableAsm
            } else {
                0
            };

            // Split HUFptr borrow from dctx borrow: decompress into
            // litExtraBuffer. Upstream uses `dctx->HUFptr` (set by the
            // previous `set_compressed` block) for `set_repeat`, and
            // `dctx->entropy.hufTable` for the fresh-header path. Here
            // the two live in the same `hufTable` field.
            let in_dst_lit = dctx.litBufferLocation == ZSTD_litLocation_e::ZSTD_in_dst
                || dctx.litBufferLocation == ZSTD_litLocation_e::ZSTD_split;
            let hufSuccess = if litEncType == SymbolEncodingType_e::set_repeat {
                let huf_table = &dctx.hufTable.clone();
                let lit_dst: &mut [u8] = if in_dst_lit {
                    let start = dctx.litBuffer_offset;
                    &mut dst[start..start + litSize]
                } else {
                    &mut dctx.litExtraBuffer[..litSize]
                };
                if singleStream {
                    if (huf_flags & HUF_flags_bmi2) != 0 {
                        HUF_decompress1X_usingDTable::<true>(
                            lit_dst,
                            &src[lhSize..lhSize + litCSize],
                            huf_table,
                        )
                    } else {
                        HUF_decompress1X_usingDTable::<false>(
                            lit_dst,
                            &src[lhSize..lhSize + litCSize],
                            huf_table,
                        )
                    }
                } else if (huf_flags & HUF_flags_bmi2) != 0 {
                    HUF_decompress4X_usingDTable::<true>(
                        lit_dst,
                        &src[lhSize..lhSize + litCSize],
                        huf_table,
                    )
                } else {
                    HUF_decompress4X_usingDTable::<false>(
                        lit_dst,
                        &src[lhSize..lhSize + litCSize],
                        huf_table,
                    )
                }
            } else if singleStream {
                let lit_dst: &mut [u8] = if in_dst_lit {
                    let start = dctx.litBuffer_offset;
                    &mut dst[start..start + litSize]
                } else {
                    &mut dctx.litExtraBuffer[..litSize]
                };
                HUF_decompress1X1_DCtx_wksp(
                    &mut dctx.hufTable,
                    lit_dst,
                    &src[lhSize..lhSize + litCSize],
                    &mut dctx.workspace,
                    huf_flags,
                )
            } else {
                let lit_dst: &mut [u8] = if in_dst_lit {
                    let start = dctx.litBuffer_offset;
                    &mut dst[start..start + litSize]
                } else {
                    &mut dctx.litExtraBuffer[..litSize]
                };
                HUF_decompress4X_hufOnly_wksp(
                    &mut dctx.hufTable,
                    lit_dst,
                    &src[lhSize..lhSize + litCSize],
                    &mut dctx.workspace,
                    huf_flags,
                )
            };

            if crate::common::error::ERR_isError(hufSuccess) {
                return ERROR(ErrorCode::CorruptionDetected);
            }

            if dctx.litBufferLocation == ZSTD_litLocation_e::ZSTD_split {
                let initial_start = dctx.litBuffer_offset;
                let final_start = initial_start + ZSTD_LITBUFFEREXTRASIZE - WILDCOPY_OVERLENGTH;
                let first_len = litSize - ZSTD_LITBUFFEREXTRASIZE;
                dctx.litExtraBuffer[..ZSTD_LITBUFFEREXTRASIZE].copy_from_slice(
                    &dst[initial_start + first_len
                        ..initial_start + first_len + ZSTD_LITBUFFEREXTRASIZE],
                );
                dst.copy_within(initial_start..initial_start + first_len, final_start);
                dctx.litBuffer_offset = final_start;
                dctx.litBufferEnd_offset = final_start + first_len;
            }

            dctx.litPtr_from_dst = in_dst_lit;
            dctx.litPtr_offset = if in_dst_lit { dctx.litBuffer_offset } else { 0 };
            dctx.litSize = litSize;
            dctx.litEntropy = 1;
            litCSize + lhSize
        }

        SymbolEncodingType_e::set_basic => {
            let lhlCode = (src[0] >> 2) & 3;
            let (lhSize, litSize) = match lhlCode {
                0 | 2 => (1usize, (src[0] >> 3) as usize),
                1 => (2usize, (MEM_readLE16(&src[..2]) >> 4) as usize),
                _ => {
                    if srcSize < 3 {
                        return ERROR(ErrorCode::CorruptionDetected);
                    }
                    (3usize, (MEM_readLE24(&src[..3]) >> 4) as usize)
                }
            };
            if litSize > blockSizeMax {
                return ERROR(ErrorCode::CorruptionDetected);
            }
            let expectedWriteSize = blockSizeMax.min(dstCapacity);
            if expectedWriteSize < litSize {
                return ERROR(ErrorCode::DstSizeTooSmall);
            }
            ZSTD_allocateLiteralsBuffer(
                dctx,
                dst.len(),
                dstCapacity,
                litSize,
                streaming,
                expectedWriteSize,
                1,
            );

            if lhSize + litSize + WILDCOPY_OVERLENGTH > srcSize {
                if litSize + lhSize > srcSize {
                    return ERROR(ErrorCode::CorruptionDetected);
                }
                if dctx.litBufferLocation == ZSTD_litLocation_e::ZSTD_in_dst {
                    let start = dctx.litBuffer_offset;
                    dst[start..start + litSize].copy_from_slice(&src[lhSize..lhSize + litSize]);
                    dctx.litPtr_from_dst = true;
                    dctx.litPtr_offset = start;
                } else if dctx.litBufferLocation == ZSTD_litLocation_e::ZSTD_split {
                    let first_len = litSize - ZSTD_LITBUFFEREXTRASIZE;
                    let start = dctx.litBuffer_offset;
                    dst[start..start + first_len].copy_from_slice(&src[lhSize..lhSize + first_len]);
                    dctx.litExtraBuffer[..ZSTD_LITBUFFEREXTRASIZE].copy_from_slice(
                        &src[lhSize + first_len..lhSize + first_len + ZSTD_LITBUFFEREXTRASIZE],
                    );
                    dctx.litPtr_from_dst = true;
                    dctx.litPtr_offset = start;
                } else {
                    dctx.litExtraBuffer[..litSize].copy_from_slice(&src[lhSize..lhSize + litSize]);
                    dctx.litPtr_from_dst = false;
                    dctx.litPtr_offset = 0;
                }
                dctx.litSize = litSize;
                return lhSize + litSize;
            }

            // Direct reference into compressed stream. Upstream sets
            // litPtr = istart + lhSize. The Rust port copies into the
            // owned extraBuffer so downstream code doesn't need to
            // hold `src`'s borrow.
            if dctx.litExtraBuffer.len() < litSize {
                dctx.litExtraBuffer.resize(litSize, 0);
            }
            dctx.litExtraBuffer[..litSize].copy_from_slice(&src[lhSize..lhSize + litSize]);
            dctx.litPtr_from_dst = false;
            dctx.litPtr_offset = 0;
            dctx.litSize = litSize;
            dctx.litBufferEnd_offset = litSize;
            dctx.litBufferLocation = ZSTD_litLocation_e::ZSTD_not_in_dst;
            lhSize + litSize
        }

        SymbolEncodingType_e::set_rle => {
            let lhlCode = (src[0] >> 2) & 3;
            let (lhSize, litSize) = match lhlCode {
                0 | 2 => (1usize, (src[0] >> 3) as usize),
                1 => {
                    if srcSize < 3 {
                        return ERROR(ErrorCode::CorruptionDetected);
                    }
                    (2usize, (MEM_readLE16(&src[..2]) >> 4) as usize)
                }
                _ => {
                    if srcSize < 4 {
                        return ERROR(ErrorCode::CorruptionDetected);
                    }
                    (3usize, (MEM_readLE24(&src[..3]) >> 4) as usize)
                }
            };
            if litSize > blockSizeMax {
                return ERROR(ErrorCode::CorruptionDetected);
            }
            let expectedWriteSize = blockSizeMax.min(dstCapacity);
            if expectedWriteSize < litSize {
                return ERROR(ErrorCode::DstSizeTooSmall);
            }
            ZSTD_allocateLiteralsBuffer(
                dctx,
                dst.len(),
                dstCapacity,
                litSize,
                streaming,
                expectedWriteSize,
                1,
            );

            let rle_byte = src[lhSize];
            if dctx.litBufferLocation == ZSTD_litLocation_e::ZSTD_in_dst {
                let start = dctx.litBuffer_offset;
                for b in dst[start..start + litSize].iter_mut() {
                    *b = rle_byte;
                }
                dctx.litPtr_from_dst = true;
                dctx.litPtr_offset = start;
            } else if dctx.litBufferLocation == ZSTD_litLocation_e::ZSTD_split {
                let first_len = litSize - ZSTD_LITBUFFEREXTRASIZE;
                let start = dctx.litBuffer_offset;
                for b in dst[start..start + first_len].iter_mut() {
                    *b = rle_byte;
                }
                for b in dctx.litExtraBuffer[..ZSTD_LITBUFFEREXTRASIZE].iter_mut() {
                    *b = rle_byte;
                }
                dctx.litPtr_from_dst = true;
                dctx.litPtr_offset = start;
            } else {
                for b in dctx.litExtraBuffer[..litSize].iter_mut() {
                    *b = rle_byte;
                }
                dctx.litPtr_from_dst = false;
                dctx.litPtr_offset = 0;
            }
            dctx.litSize = litSize;
            lhSize + 1
        }
    }
}

pub const MIN_SEQUENCES_SIZE: usize = 1;
pub const LONGNBSEQ: i32 = 0x7F00;

// Sequence symbol alphabets (from lib/common/zstd_internal.h).
pub const MaxLL: u32 = 35;
pub const MaxML: u32 = 52;
pub const MaxOff: u32 = 31;
pub const DefaultMaxOff: u32 = 28;
pub const MaxSeq: u32 = 52; // max(MaxLL, MaxML)
pub const LLFSELog: u32 = 9;
pub const MLFSELog: u32 = 9;
pub const OffFSELog: u32 = 8;
pub const MaxFSELog: u32 = 9;

/// Upstream `LL_bits` — extra literal-length bits read from the stream
/// per LL code.
pub const LL_bits: [u8; (MaxLL + 1) as usize] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16,
];

/// Upstream `LL_base`.
pub const LL_base: [u32; (MaxLL + 1) as usize] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28, 32, 40, 48, 64,
    0x80, 0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000, 0x10000,
];

/// Upstream `ML_bits`.
pub const ML_bits: [u8; (MaxML + 1) as usize] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
];

/// Upstream `ML_base`.
pub const ML_base: [u32; (MaxML + 1) as usize] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 0x83, 0x103, 0x203,
    0x403, 0x803, 0x1003, 0x2003, 0x4003, 0x8003, 0x10003,
];

/// Upstream `OF_bits`. Offset code `c` reads `c` additional bits.
pub const OF_bits: [u8; (MaxOff + 1) as usize] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31,
];

/// Upstream `OF_base`.
pub const OF_base: [u32; (MaxOff + 1) as usize] = [
    0, 1, 1, 5, 0xD, 0x1D, 0x3D, 0x7D, 0xFD, 0x1FD, 0x3FD, 0x7FD, 0xFFD, 0x1FFD, 0x3FFD, 0x7FFD,
    0xFFFD, 0x1FFFD, 0x3FFFD, 0x7FFFD, 0xFFFFD, 0x1FFFFD, 0x3FFFFD, 0x7FFFFD, 0xFFFFFD, 0x1FFFFFD,
    0x3FFFFFD, 0x7FFFFFD, 0xFFFFFFD, 0x1FFFFFFD, 0x3FFFFFFD, 0x7FFFFFFD,
];

// Default normalized counts (from lib/common/zstd_internal.h). The
// decoder uses these to build the "basic" FSE default DTables.
pub const LL_defaultNorm: [i16; (MaxLL + 1) as usize] = [
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
    -1, -1, -1, -1,
];
pub const LL_defaultNormLog: u32 = 6;

pub const ML_defaultNorm: [i16; (MaxML + 1) as usize] = [
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
];
pub const ML_defaultNormLog: u32 = 6;

pub const OF_defaultNorm: [i16; (DefaultMaxOff + 1) as usize] = [
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
];
pub const OF_defaultNormLog: u32 = 5;

/// Mirror of upstream `ZSTD_seqSymbol_header`. Two u32 fields, stored
/// in the DTable's slot-0 descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ZSTD_seqSymbol_header {
    pub fastMode: u32,
    pub tableLog: u32,
}

/// Mirror of upstream `ZSTD_seqSymbol` (decode entry).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ZSTD_seqSymbol {
    pub nextState: u16,
    pub nbAdditionalBits: u8,
    pub nbBits: u8,
    pub baseValue: u32,
}

/// Upstream macro. DTable capacity = 1 header + (1 << tableLog) entries.
#[inline]
pub const fn SEQSYMBOL_TABLE_SIZE(tableLog: u32) -> usize {
    1 + (1 << tableLog) as usize
}

/// Port of `ZSTD_buildSeqTable_rle` (inlined upstream helper). A
/// single entry that always emits `baseValue` with no additional bits.
pub fn ZSTD_buildSeqTable_rle(dt: &mut [ZSTD_seqSymbol], baseline: u32, nbBits: u8) {
    // Write header slot.
    dt[0] = ZSTD_seqSymbol {
        nextState: 0,
        nbAdditionalBits: 0,
        nbBits: 0,
        baseValue: 0,
    };
    // Abuse the Rust struct as upstream's header — we piggyback on
    // `tableLog`/`fastMode` via explicit fields elsewhere, but for the
    // RLE case only the decode slot matters and the FSE decoder keys
    // off tableLog=0 from the header.
    // Store tableLog=0 and fastMode=0 via baseValue=0 trick: upstream
    // uses ZSTD_seqSymbol_header-style packing; to keep the layout
    // simple we add a parallel header_of accessor that reads these
    // fields. For now we encode tableLog in the `nbBits` byte of slot 0.
    let header = ZSTD_seqSymbol {
        nextState: 0, // fastMode
        nbAdditionalBits: 0,
        nbBits: 0, // tableLog
        baseValue: 0,
    };
    dt[0] = header;
    dt[1] = ZSTD_seqSymbol {
        nextState: 0,
        nbAdditionalBits: nbBits,
        nbBits: 0,
        baseValue: baseline,
    };
}

/// Encode an `(fastMode, tableLog)` header into slot 0 of a seqSymbol
/// DTable. We reuse two fields of `ZSTD_seqSymbol` as the header:
/// `nextState` holds `fastMode`, `baseValue` holds `tableLog`.
#[inline]
fn seq_header_write(dt: &mut [ZSTD_seqSymbol], fastMode: u32, tableLog: u32) {
    dt[0] = ZSTD_seqSymbol {
        nextState: fastMode as u16,
        nbAdditionalBits: 0,
        nbBits: 0,
        baseValue: tableLog,
    };
}

/// Rust-only helper: read the `(fastMode, tableLog)` header out of slot
/// 0 of an FSE decode table.
#[inline]
pub fn seq_header_read(dt: &[ZSTD_seqSymbol]) -> ZSTD_seqSymbol_header {
    ZSTD_seqSymbol_header {
        fastMode: dt[0].nextState as u32,
        tableLog: dt[0].baseValue,
    }
}

/// Rust-only helper: number of FSE decode entries actually populated
/// (header slot + `1 << tableLog` entries).
#[inline]
fn seq_table_used_len(dt: &[ZSTD_seqSymbol]) -> usize {
    1 + (1usize << dt[0].baseValue)
}

/// Test-only helper: returns the LL FSE table the decoder would
/// currently dispatch through (canonical default vs. parsed table).
#[cfg(test)]
pub(crate) fn active_ll_table(dctx: &ZSTD_DCtx) -> &[ZSTD_seqSymbol] {
    active_ll_table_from(&dctx.LLTable, dctx.ll_default_active)
}

/// Test-only helper: see `active_ll_table` — same pattern for OF.
#[inline]
#[cfg(test)]
pub(crate) fn active_of_table(dctx: &ZSTD_DCtx) -> &[ZSTD_seqSymbol] {
    active_of_table_from(&dctx.OFTable, dctx.of_default_active)
}

/// Test-only helper: see `active_ll_table` — same pattern for ML.
#[inline]
#[cfg(test)]
pub(crate) fn active_ml_table(dctx: &ZSTD_DCtx) -> &[ZSTD_seqSymbol] {
    active_ml_table_from(&dctx.MLTable, dctx.ml_default_active)
}

/// Rust-only helper: pick between the canonical LL default table and
/// the DCtx-held parsed table, based on the `default_active` flag.
#[inline]
fn active_ll_table_from(table: &[ZSTD_seqSymbol], default_active: bool) -> &[ZSTD_seqSymbol] {
    if default_active {
        default_ll_dtable()
    } else {
        &table[..seq_table_used_len(table)]
    }
}

/// Rust-only helper: pick between the canonical OF default table and
/// the DCtx-held parsed table.
#[inline]
fn active_of_table_from(table: &[ZSTD_seqSymbol], default_active: bool) -> &[ZSTD_seqSymbol] {
    if default_active {
        default_of_dtable()
    } else {
        &table[..seq_table_used_len(table)]
    }
}

/// Rust-only helper: pick between the canonical ML default table and
/// the DCtx-held parsed table.
#[inline]
fn active_ml_table_from(table: &[ZSTD_seqSymbol], default_active: bool) -> &[ZSTD_seqSymbol] {
    if default_active {
        default_ml_dtable()
    } else {
        &table[..seq_table_used_len(table)]
    }
}

/// Port of `ZSTD_buildFSETable_body` / `ZSTD_buildFSETable`. Writes
/// `1 + (1<<tableLog)` `ZSTD_seqSymbol` entries into `dt`.
pub fn ZSTD_buildFSETable(
    dt: &mut [ZSTD_seqSymbol],
    normalizedCounter: &[i16],
    maxSymbolValue: u32,
    baseValue: &[u32],
    nbAdditionalBits: &[u8],
    tableLog: u32,
) {
    let maxSV1 = (maxSymbolValue + 1) as usize;
    let tableSize = 1usize << tableLog;

    // Rust: stack-allocate the scratch that C takes via workspace.
    // Both buffers are fully written in their used ranges before being read.
    let mut symbolNext: [core::mem::MaybeUninit<u16>; (MaxSeq + 1) as usize] =
        unsafe { core::mem::MaybeUninit::uninit().assume_init() };
    let mut spread: [core::mem::MaybeUninit<u8>; 1 << 9] =
        unsafe { core::mem::MaybeUninit::uninit().assume_init() }; // MaxFSELog
    let dt_ptr = dt.as_mut_ptr();
    let nc_ptr = normalizedCounter.as_ptr();
    let base_value_ptr = baseValue.as_ptr();
    let nb_additional_bits_ptr = nbAdditionalBits.as_ptr();
    let symbol_next_ptr = symbolNext.as_mut_ptr() as *mut u16;
    let spread_ptr = spread.as_mut_ptr() as *mut u8;
    let mut highThreshold: i64 = tableSize as i64 - 1;
    let mut fastMode: u32 = 1;
    let largeLimit: i16 = 1 << (tableLog - 1);

    for s in 0..maxSV1 {
        let nc = unsafe { *nc_ptr.add(s) };
        if nc == -1 {
            // Lowprob symbol: place it at `highThreshold` and walk the
            // sink index down.
            unsafe {
                (*dt_ptr.add(1 + highThreshold as usize)).baseValue = s as u32;
            }
            highThreshold -= 1;
            unsafe {
                *symbol_next_ptr.add(s) = 1;
            }
        } else {
            if nc >= largeLimit {
                fastMode = 0;
            }
            unsafe {
                *symbol_next_ptr.add(s) = nc as u16;
            }
        }
    }
    seq_header_write(dt, fastMode, tableLog);

    // Spread symbols across positions.
    let tableMask = tableSize - 1;
    let step = crate::common::fse_decompress::FSE_TABLESTEP(tableSize as u32) as usize;

    if highThreshold == tableSize as i64 - 1 {
        // Fast path: lay down in order, then spread.
        let mut pos: usize = 0;
        for s in 0..maxSV1 {
            let nc = unsafe { *nc_ptr.add(s) };
            let n = nc as i32;
            for i in 0..n as usize {
                unsafe {
                    *spread_ptr.add(pos + i) = s as u8;
                }
            }
            if n > 0 {
                pos += n as usize;
            }
        }
        let mut position: usize = 0;
        let mut s = 0;
        while s < tableSize {
            for u in 0..2 {
                let uPos = (position + (u * step)) & tableMask;
                unsafe {
                    (*dt_ptr.add(1 + uPos)).baseValue = *spread_ptr.add(s + u) as u32;
                }
            }
            position = (position + (2 * step)) & tableMask;
            s += 2;
        }
    } else {
        // Slow path with lowprob region skipping.
        let mut position: usize = 0;
        for s in 0..maxSV1 {
            let nc = unsafe { *nc_ptr.add(s) };
            for _ in 0..nc {
                unsafe {
                    (*dt_ptr.add(1 + position)).baseValue = s as u32;
                }
                position = (position + step) & tableMask;
                while position as i64 > highThreshold {
                    position = (position + step) & tableMask;
                }
            }
        }
    }

    // Build decoding entries.
    for u in 0..tableSize {
        let entry = unsafe { &mut *dt_ptr.add(1 + u) };
        let symbol = entry.baseValue as usize;
        let nextState = unsafe { *symbol_next_ptr.add(symbol) };
        unsafe {
            *symbol_next_ptr.add(symbol) = nextState + 1;
        }
        let hb = crate::common::bits::ZSTD_highbit32(nextState as u32);
        let nbBits = (tableLog - hb) as u8;
        let newState = (((nextState as u32) << nbBits) - tableSize as u32) as u16;
        entry.nbBits = nbBits;
        entry.nextState = newState;
        unsafe {
            entry.nbAdditionalBits = *nb_additional_bits_ptr.add(symbol);
            entry.baseValue = *base_value_ptr.add(symbol);
        }
    }
}

/// Port of `ZSTD_buildFSETable_body`. The Rust port doesn't need the
/// C workspace argument because the scratch arrays are stack-owned in
/// `ZSTD_buildFSETable`.
pub fn ZSTD_buildFSETable_body(
    dt: &mut [ZSTD_seqSymbol],
    normalizedCounter: &[i16],
    maxSymbolValue: u32,
    baseValue: &[u32],
    nbAdditionalBits: &[u8],
    tableLog: u32,
) {
    ZSTD_buildFSETable(
        dt,
        normalizedCounter,
        maxSymbolValue,
        baseValue,
        nbAdditionalBits,
        tableLog,
    );
}

/// Portable wrapper for `ZSTD_buildFSETable_body_default`.
pub fn ZSTD_buildFSETable_body_default(
    dt: &mut [ZSTD_seqSymbol],
    normalizedCounter: &[i16],
    maxSymbolValue: u32,
    baseValue: &[u32],
    nbAdditionalBits: &[u8],
    tableLog: u32,
) {
    ZSTD_buildFSETable_body(
        dt,
        normalizedCounter,
        maxSymbolValue,
        baseValue,
        nbAdditionalBits,
        tableLog,
    );
}

/// BMI2 wrapper for `ZSTD_buildFSETable_body_bmi2`. The generic Rust
/// implementation is already bit-identical for the table contents.
pub fn ZSTD_buildFSETable_body_bmi2(
    dt: &mut [ZSTD_seqSymbol],
    normalizedCounter: &[i16],
    maxSymbolValue: u32,
    baseValue: &[u32],
    nbAdditionalBits: &[u8],
    tableLog: u32,
) {
    ZSTD_buildFSETable_body(
        dt,
        normalizedCounter,
        maxSymbolValue,
        baseValue,
        nbAdditionalBits,
        tableLog,
    );
}

/// Port of `ZSTD_buildSeqTable`. Returns bytes consumed from `src`, or
/// an error code. `default_table` is the precomputed "basic" DTable
/// (used when `type == set_basic`).
///
/// Rust signature note: upstream uses `const ZSTD_seqSymbol**
/// DTablePtr` so the output table pointer can be either the DCtx's
/// scratch (`set_rle`/`set_compressed`) or a static default
/// (`set_basic`) or "last used" (`set_repeat`). Rust can't return a
/// reference that covers both borrowed cases, so we copy the default
/// table into `DTableSpace` in the `set_basic` case. Correctness is
/// identical; there's a one-memcpy-per-block overhead.
pub fn ZSTD_buildSeqTable(
    DTableSpace: &mut [ZSTD_seqSymbol],
    type_: SymbolEncodingType_e,
    max: u32,
    maxLog: u32,
    src: &[u8],
    baseValue: &[u32],
    nbAdditionalBits: &[u8],
    defaultTable: Option<&[ZSTD_seqSymbol]>,
    flagRepeatTable: bool,
    _ddictIsCold: i32,
    _nbSeq: i32,
) -> usize {
    match type_ {
        SymbolEncodingType_e::set_rle => {
            if src.is_empty() {
                return ERROR(ErrorCode::SrcSizeWrong);
            }
            let symbol = src[0] as u32;
            if symbol > max {
                return ERROR(ErrorCode::CorruptionDetected);
            }
            let baseline = baseValue[symbol as usize];
            let nbBits = nbAdditionalBits[symbol as usize];
            ZSTD_buildSeqTable_rle(DTableSpace, baseline, nbBits);
            1
        }
        SymbolEncodingType_e::set_basic => {
            if let Some(t) = defaultTable {
                // Copy the default table into DTableSpace so callers
                // can read it through the same `&[ZSTD_seqSymbol]`.
                let len = t.len().min(DTableSpace.len());
                if len > 0 {
                    unsafe {
                        std::ptr::copy_nonoverlapping(t.as_ptr(), DTableSpace.as_mut_ptr(), len);
                    }
                }
            }
            0
        }
        SymbolEncodingType_e::set_repeat => {
            if !flagRepeatTable {
                return ERROR(ErrorCode::CorruptionDetected);
            }
            // Upstream prefetches the repeated table when cold; Rust
            // port skips the prefetch hint.
            0
        }
        SymbolEncodingType_e::set_compressed => {
            let mut max_mut = max;
            let mut tableLog: u32 = 0;
            let mut norm = [0i16; (MaxSeq + 1) as usize];
            let headerSize = crate::common::entropy_common::FSE_readNCount_no_clear(
                &mut norm,
                &mut max_mut,
                &mut tableLog,
                src,
            );
            if crate::common::error::ERR_isError(headerSize) {
                return ERROR(ErrorCode::CorruptionDetected);
            }
            if tableLog > maxLog {
                return ERROR(ErrorCode::CorruptionDetected);
            }
            ZSTD_buildFSETable(
                DTableSpace,
                &norm,
                max_mut,
                baseValue,
                nbAdditionalBits,
                tableLog,
            );
            headerSize
        }
    }
}

/// Decoded sequence-block header descriptor. The FSE-table builder
/// `ZSTD_buildSeqTable` (ported) consumes this to populate the DCtx's
/// LL / OF / ML FSE tables for the body's bitstream.
#[derive(Debug, Clone, Copy)]
pub struct SeqBlockHeader {
    pub nbSeq: i32,
    pub LLtype: SymbolEncodingType_e,
    pub OFtype: SymbolEncodingType_e,
    pub MLtype: SymbolEncodingType_e,
    /// Number of bytes of `src` consumed so far (after the nbSeq
    /// varint and the encoding-type byte, before the FSE table payloads).
    pub headerSize: usize,
}

/// Port of the opening of `ZSTD_decodeSeqHeaders` that parses only the
/// `nbSeq` varint and the `LL / OF / ML` encoding-type byte. Returns
/// either an error code or a `SeqBlockHeader` (through `out`) and the
/// number of bytes consumed so far.
///
/// The FSE-table building that follows (three `ZSTD_buildSeqTable`
/// calls) is the next step — `ZSTD_buildSeqTable` itself is already
/// ported and used by the full `ZSTD_decodeSeqHeaders` entry.
pub fn ZSTD_decodeSeqHeaders_probe(src: &[u8], out: &mut SeqBlockHeader) -> usize {
    let iend = src.len();
    if iend < MIN_SEQUENCES_SIZE {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    let mut ip = 0usize;
    let mut nbSeq: i32 = src[ip] as i32;
    ip += 1;

    if nbSeq > 0x7F {
        if nbSeq == 0xFF {
            if ip + 2 > iend {
                return ERROR(ErrorCode::SrcSizeWrong);
            }
            nbSeq = MEM_readLE16(&src[ip..ip + 2]) as i32 + LONGNBSEQ;
            ip += 2;
        } else {
            if ip >= iend {
                return ERROR(ErrorCode::SrcSizeWrong);
            }
            nbSeq = ((nbSeq - 0x80) << 8) + src[ip] as i32;
            ip += 1;
        }
    }
    out.nbSeq = nbSeq;

    if nbSeq == 0 {
        // No sequences: section must end immediately.
        if ip != iend {
            return ERROR(ErrorCode::CorruptionDetected);
        }
        out.LLtype = SymbolEncodingType_e::set_basic;
        out.OFtype = SymbolEncodingType_e::set_basic;
        out.MLtype = SymbolEncodingType_e::set_basic;
        out.headerSize = ip;
        return ip;
    }

    if ip + 1 > iend {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    if src[ip] & 3 != 0 {
        // "Reserved" bits must be zero.
        return ERROR(ErrorCode::CorruptionDetected);
    }
    out.LLtype = SymbolEncodingType_e::from_bits(src[ip] as u32 >> 6);
    out.OFtype = SymbolEncodingType_e::from_bits(src[ip] as u32 >> 4);
    out.MLtype = SymbolEncodingType_e::from_bits(src[ip] as u32 >> 2);
    ip += 1;
    out.headerSize = ip;
    ip
}

/// Build the three default seqSymbol DTables (LL / OF / ML) from their
/// canonical normalized counts. Called once per DCtx init — the output
/// is constant, derived solely from the default-norm tables.
pub fn ZSTD_buildDefaultSeqTables(dctx: &mut ZSTD_DCtx) {
    ZSTD_buildFSETable(
        &mut dctx.LLTable,
        &LL_defaultNorm,
        MaxLL,
        &LL_base,
        &LL_bits,
        LL_defaultNormLog,
    );
    // OF defaults use `DefaultMaxOff` (28), not `MaxOff` (31); upstream
    // keeps the full 32-entry base/bits tables but only the first 29
    // positions are populated for defaults.
    ZSTD_buildFSETable(
        &mut dctx.OFTable,
        &OF_defaultNorm,
        DefaultMaxOff,
        &OF_base,
        &OF_bits,
        OF_defaultNormLog,
    );
    ZSTD_buildFSETable(
        &mut dctx.MLTable,
        &ML_defaultNorm,
        MaxML,
        &ML_base,
        &ML_bits,
        ML_defaultNormLog,
    );
    dctx.ll_default_active = true;
    dctx.of_default_active = true;
    dctx.ml_default_active = true;
}

/// Port of `ZSTD_decodeSeqHeaders`. Full form: header probe + three
/// `ZSTD_buildSeqTable` invocations for LL / OF / ML. Returns bytes
/// consumed from `src`.
pub fn ZSTD_decodeSeqHeaders(dctx: &mut ZSTD_DCtx, nbSeqPtr: &mut i32, src: &[u8]) -> usize {
    let mut hdr = SeqBlockHeader {
        nbSeq: 0,
        LLtype: SymbolEncodingType_e::set_basic,
        OFtype: SymbolEncodingType_e::set_basic,
        MLtype: SymbolEncodingType_e::set_basic,
        headerSize: 0,
    };
    let consumed = ZSTD_decodeSeqHeaders_probe(src, &mut hdr);
    if crate::common::error::ERR_isError(consumed) {
        return consumed;
    }
    *nbSeqPtr = hdr.nbSeq;
    if hdr.nbSeq == 0 {
        return consumed;
    }

    let mut ip = hdr.headerSize;

    // LL
    {
        let ll_sz = if hdr.LLtype == SymbolEncodingType_e::set_basic {
            dctx.ll_default_active = true;
            0
        } else {
            let sz = ZSTD_buildSeqTable(
                &mut dctx.LLTable,
                hdr.LLtype,
                MaxLL,
                LLFSELog,
                &src[ip..],
                &LL_base,
                &LL_bits,
                Some(default_ll_dtable()),
                dctx.fseEntropy != 0,
                dctx.ddictIsCold,
                hdr.nbSeq,
            );
            if !crate::common::error::ERR_isError(sz)
                && hdr.LLtype != SymbolEncodingType_e::set_repeat
            {
                dctx.ll_default_active = false;
            }
            sz
        };
        if crate::common::error::ERR_isError(ll_sz) {
            return ERROR(ErrorCode::CorruptionDetected);
        }
        ip += ll_sz;
        dctx.fse_ll_fresh = hdr.LLtype == SymbolEncodingType_e::set_compressed
            || hdr.LLtype == SymbolEncodingType_e::set_repeat
            || dctx.fse_ll_fresh;
    }

    // OF
    {
        let of_sz = if hdr.OFtype == SymbolEncodingType_e::set_basic {
            dctx.of_default_active = true;
            0
        } else {
            let sz = ZSTD_buildSeqTable(
                &mut dctx.OFTable,
                hdr.OFtype,
                MaxOff,
                OffFSELog,
                &src[ip..],
                &OF_base,
                &OF_bits,
                Some(default_of_dtable()),
                dctx.fseEntropy != 0,
                dctx.ddictIsCold,
                hdr.nbSeq,
            );
            if !crate::common::error::ERR_isError(sz)
                && hdr.OFtype != SymbolEncodingType_e::set_repeat
            {
                dctx.of_default_active = false;
            }
            sz
        };
        if crate::common::error::ERR_isError(of_sz) {
            return ERROR(ErrorCode::CorruptionDetected);
        }
        ip += of_sz;
        dctx.fse_of_fresh = hdr.OFtype == SymbolEncodingType_e::set_compressed
            || hdr.OFtype == SymbolEncodingType_e::set_repeat
            || dctx.fse_of_fresh;
    }

    // ML
    {
        let ml_sz = if hdr.MLtype == SymbolEncodingType_e::set_basic {
            dctx.ml_default_active = true;
            0
        } else {
            let sz = ZSTD_buildSeqTable(
                &mut dctx.MLTable,
                hdr.MLtype,
                MaxML,
                MLFSELog,
                &src[ip..],
                &ML_base,
                &ML_bits,
                Some(default_ml_dtable()),
                dctx.fseEntropy != 0,
                dctx.ddictIsCold,
                hdr.nbSeq,
            );
            if !crate::common::error::ERR_isError(sz)
                && hdr.MLtype != SymbolEncodingType_e::set_repeat
            {
                dctx.ml_default_active = false;
            }
            sz
        };
        if crate::common::error::ERR_isError(ml_sz) {
            return ERROR(ErrorCode::CorruptionDetected);
        }
        ip += ml_sz;
        dctx.fse_ml_fresh = hdr.MLtype == SymbolEncodingType_e::set_compressed
            || hdr.MLtype == SymbolEncodingType_e::set_repeat
            || dctx.fse_ml_fresh;
    }

    ip
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dctx_default_max_window_size_matches_upstream_byte_cap() {
        let dctx = ZSTD_DCtx::default();
        assert_eq!(
            dctx.d_maxWindowSize,
            (1u64 << crate::decompress::zstd_decompress::ZSTD_WINDOWLOG_LIMIT_DEFAULT) + 1
        );
        assert!(!dctx.d_maxWindowSizeSet);
        assert_eq!(dctx.isFrameDecompression, 1);
    }

    #[test]
    fn decode_literals_block_wrapper_forces_standalone_block_cap() {
        let mut src = vec![56u8];
        src.extend_from_slice(b"ABCDEFG");
        let mut dst = [0u8; 16];
        let mut dctx = ZSTD_DCtx::new();
        dctx.isFrameDecompression = 1;
        dctx.blockSizeMax = 1;

        let rc = ZSTD_decodeLiteralsBlock_wrapper(&mut dctx, &src, &mut dst);

        assert_eq!(rc, 8);
        assert_eq!(dctx.isFrameDecompression, 0);
        assert_eq!(dctx.litSize, 7);
    }

    #[test]
    fn checkContinuity_rotates_history_when_destination_changes() {
        let mut dctx = ZSTD_DCtx::default();
        let a = vec![0u8; 8];
        let b = vec![0u8; 4];

        ZSTD_checkContinuity(&mut dctx, &a, a.len());
        assert_eq!(dctx.prefixStart, Some(a.as_ptr() as usize));
        assert_eq!(dctx.previousDstEnd, Some(a.as_ptr() as usize));
        assert_eq!(dctx.dictEnd, None);

        ZSTD_checkContinuity(&mut dctx, &b, b.len());
        assert_eq!(dctx.dictEnd, Some(a.as_ptr() as usize));
        assert_eq!(dctx.prefixStart, Some(b.as_ptr() as usize));
        assert_eq!(dctx.previousDstEnd, Some(b.as_ptr() as usize));
        assert!(dctx.virtualStart.is_some());
    }

    #[test]
    fn decodeSeqHeaders_basic_restores_canonical_default_tables() {
        let mut dctx = ZSTD_DCtx::default();
        seq_header_write(&mut dctx.LLTable, 0, LLFSELog);
        seq_header_write(&mut dctx.OFTable, 0, OffFSELog);
        seq_header_write(&mut dctx.MLTable, 0, MLFSELog);
        dctx.ll_default_active = false;
        dctx.of_default_active = false;
        dctx.ml_default_active = false;

        let mut nb_seq = 0i32;
        let rc = ZSTD_decodeSeqHeaders(&mut dctx, &mut nb_seq, &[1, 0]);
        assert_eq!(rc, 2);
        assert_eq!(nb_seq, 1);
        assert!(dctx.ll_default_active);
        assert!(dctx.of_default_active);
        assert!(dctx.ml_default_active);
        assert_eq!(
            seq_header_read(active_ll_table(&dctx)).tableLog,
            LL_defaultNormLog
        );
        assert_eq!(
            seq_header_read(active_of_table(&dctx)).tableLog,
            OF_defaultNormLog
        );
        assert_eq!(
            seq_header_read(active_ml_table(&dctx)).tableLog,
            ML_defaultNormLog
        );
    }

    #[test]
    fn blockType_e_discriminants_match_upstream_spec() {
        // Block-type discriminants are baked into the 2-bit block
        // header field — spec-critical, cross-compat-critical.
        assert_eq!(blockType_e::bt_raw as u8, 0);
        assert_eq!(blockType_e::bt_rle as u8, 1);
        assert_eq!(blockType_e::bt_compressed as u8, 2);
        assert_eq!(blockType_e::bt_reserved as u8, 3);
        // from_bits should round-trip through each value.
        assert_eq!(blockType_e::from_bits(0), blockType_e::bt_raw);
        assert_eq!(blockType_e::from_bits(1), blockType_e::bt_rle);
        assert_eq!(blockType_e::from_bits(2), blockType_e::bt_compressed);
        assert_eq!(blockType_e::from_bits(3), blockType_e::bt_reserved);
        // Higher bits are masked off — 7 still resolves to reserved.
        assert_eq!(blockType_e::from_bits(7), blockType_e::bt_reserved);
    }

    #[test]
    fn symbolEncodingType_e_discriminants_match_upstream_spec() {
        // Same 2-bit discriminator is reused in the literal section
        // header and in each of LL/OF/ML encoding-type fields of the
        // sequences section header — all spec-critical.
        assert_eq!(SymbolEncodingType_e::set_basic as u8, 0);
        assert_eq!(SymbolEncodingType_e::set_rle as u8, 1);
        assert_eq!(SymbolEncodingType_e::set_compressed as u8, 2);
        assert_eq!(SymbolEncodingType_e::set_repeat as u8, 3);
        // from_bits should round-trip through each value.
        assert_eq!(
            SymbolEncodingType_e::from_bits(0),
            SymbolEncodingType_e::set_basic
        );
        assert_eq!(
            SymbolEncodingType_e::from_bits(1),
            SymbolEncodingType_e::set_rle
        );
        assert_eq!(
            SymbolEncodingType_e::from_bits(2),
            SymbolEncodingType_e::set_compressed
        );
        assert_eq!(
            SymbolEncodingType_e::from_bits(3),
            SymbolEncodingType_e::set_repeat
        );
        // Higher bits are masked off — 7 still resolves to set_repeat.
        assert_eq!(
            SymbolEncodingType_e::from_bits(7),
            SymbolEncodingType_e::set_repeat
        );
    }

    #[test]
    fn block_format_constants_match_upstream() {
        // Format-level constants must match the Zstandard format spec.
        // Drift here would break cross-compatibility with upstream.
        assert_eq!(ZSTD_BLOCKHEADERSIZE, 3);
        assert_eq!(ZSTD_blockHeaderSize, 3);
        assert_eq!(ZSTD_BLOCKSIZELOG_MAX, 17);
        assert_eq!(ZSTD_BLOCKSIZE_MAX, 128 * 1024);
        assert_eq!(ZSTD_REP_NUM, 3);
        assert_eq!(MIN_CBLOCK_SIZE, 2);
        assert_eq!(WILDCOPY_OVERLENGTH, 32);
        assert_eq!(WILDCOPY_VECLEN, 16);
        assert_eq!(STREAM_ACCUMULATOR_MIN_32, 25);
        assert_eq!(LONG_OFFSETS_MAX_EXTRA_BITS_32, 5);
    }

    #[test]
    fn sequences_default_norms_match_spec_anchors() {
        // Default normalized-count tables + their tableLogs form the
        // "basic" encoding mode's implicit FSE tables — every frame
        // compressed via default entropy references these values.
        // Drift here would silently flip to incompatible default
        // distributions. Pinned against upstream's zstd_internal.h.

        // Log values and array lengths.
        assert_eq!(LL_defaultNormLog, 6);
        assert_eq!(ML_defaultNormLog, 6);
        assert_eq!(OF_defaultNormLog, 5);
        assert_eq!(LL_defaultNorm.len(), (MaxLL + 1) as usize);
        assert_eq!(ML_defaultNorm.len(), (MaxML + 1) as usize);
        assert_eq!(OF_defaultNorm.len(), (DefaultMaxOff + 1) as usize);

        // Anchor entries (first, specific interior, last -1 marker).
        assert_eq!(LL_defaultNorm[0], 4);
        assert_eq!(LL_defaultNorm[13], 1);
        assert_eq!(LL_defaultNorm[25], 3);
        assert_eq!(LL_defaultNorm[35], -1);
        assert_eq!(ML_defaultNorm[0], 1);
        assert_eq!(ML_defaultNorm[1], 4);
        assert_eq!(ML_defaultNorm[52], -1);
        assert_eq!(OF_defaultNorm[0], 1);
        assert_eq!(OF_defaultNorm[6], 2);
        assert_eq!(OF_defaultNorm[28], -1);

        // Normalization sum invariant: signed non-default entries sum
        // to `1 << tableLog` plus the count of -1 "less-probable"
        // entries (upstream expects each -1 to represent 1/(1<<log)).
        fn verify_norm_sum(norm: &[i16], log: u32) {
            let total: i32 = norm
                .iter()
                .map(|&v| if v == -1 { 1 } else { v as i32 })
                .sum();
            assert_eq!(total as u32, 1 << log);
        }
        verify_norm_sum(&LL_defaultNorm, LL_defaultNormLog);
        verify_norm_sum(&ML_defaultNorm, ML_defaultNormLog);
        verify_norm_sum(&OF_defaultNorm, OF_defaultNormLog);
    }

    #[test]
    fn sequences_extra_bits_tables_match_spec_anchors() {
        // Spot-check the spec-anchored entries of LL_bits, ML_bits,
        // OF_bits against the Zstandard format spec. These tables
        // drive "number of extra bits to read after the code" during
        // sequence decode — a single wrong entry corrupts output.
        // Full arrays sit in `zstd_internal.h` upstream.

        // LL_bits: codes 0..15 → 0; 16..19 → 1; 20,21 → 2; 22,23 → 3;
        // 24 → 4; 25 → 6; 26..35 increment monotonically 7..16.
        assert_eq!(LL_bits[0], 0);
        assert_eq!(LL_bits[15], 0);
        assert_eq!(LL_bits[16], 1);
        assert_eq!(LL_bits[19], 1);
        assert_eq!(LL_bits[24], 4);
        assert_eq!(LL_bits[25], 6);
        assert_eq!(LL_bits[35], 16);

        // ML_bits: codes 0..31 → 0; 32..35 → 1; 36,37 → 2; 38,39 → 3;
        // 40,41 → 4; 42 → 5; 43 → 7; last (52) → 16.
        assert_eq!(ML_bits[0], 0);
        assert_eq!(ML_bits[31], 0);
        assert_eq!(ML_bits[32], 1);
        assert_eq!(ML_bits[42], 5);
        assert_eq!(ML_bits[43], 7);
        assert_eq!(ML_bits[52], 16);

        // OF_bits: pure identity OF_bits[i] == i over the 0..=31 range.
        for (i, b) in OF_bits.iter().enumerate() {
            assert_eq!(*b as usize, i);
        }

        // Anchor values of the _base tables.
        assert_eq!(LL_base[0], 0);
        assert_eq!(LL_base[15], 15);
        assert_eq!(LL_base[16], 16);
        assert_eq!(LL_base[35], 0x10000);
        assert_eq!(ML_base[0], 3);
        assert_eq!(ML_base[52], 0x10003);

        // OF_base has a closed-form invariant: for code >= 2,
        // `OF_base[c] == (1 << c) - 3`. Codes 0,1 are special (0,1).
        assert_eq!(OF_base[0], 0);
        assert_eq!(OF_base[1], 1);
        for (c, &got) in OF_base.iter().enumerate().skip(2) {
            let expected: u32 = (1u32 << c) - 3;
            assert_eq!(
                got, expected,
                "OF_base[{c}] should be (1<<{c})-3 = {expected}, got {got}",
            );
        }
    }

    #[test]
    fn sequences_FSE_format_constants_match_upstream() {
        // Sequences-section FSE log constants are baked into every
        // frame's entropy tables; drift here would make us read/write
        // the wrong number of bits per state and silently corrupt the
        // bitstream. Pinned against `lib/common/zstd_internal.h`.
        assert_eq!(MaxLL, 35);
        assert_eq!(MaxML, 52);
        assert_eq!(MaxOff, 31);
        assert_eq!(DefaultMaxOff, 28);
        assert_eq!(MaxSeq, core::cmp::max(MaxLL, MaxML));
        assert_eq!(LLFSELog, 9);
        assert_eq!(MLFSELog, 9);
        assert_eq!(OffFSELog, 8);
        assert_eq!(
            MaxFSELog,
            core::cmp::max(core::cmp::max(MLFSELog, LLFSELog), OffFSELog),
        );
        // LL_bits table must have MaxLL+1 entries.
        assert_eq!(LL_bits.len(), (MaxLL + 1) as usize);
    }

    #[test]
    fn get_cblock_size_parses_header_raw() {
        // Header: blockType=bt_raw(0), lastBlock=1, cSize=10.
        // Encoded as 24-bit LE: (cSize<<3) | (type<<1) | last.
        let hdr_val: u32 = (10u32 << 3) | 1; // bt_raw + lastBlock=1
        let src = [
            (hdr_val & 0xFF) as u8,
            ((hdr_val >> 8) & 0xFF) as u8,
            ((hdr_val >> 16) & 0xFF) as u8,
        ];
        let mut bp = blockProperties_t {
            blockType: blockType_e::bt_reserved,
            lastBlock: 0,
            origSize: 0,
        };
        let rc = ZSTD_getcBlockSize(&src, &mut bp);
        assert_eq!(rc, 10);
        assert_eq!(bp.lastBlock, 1);
        assert_eq!(bp.blockType, blockType_e::bt_raw);
        assert_eq!(bp.origSize, 10);
    }

    #[test]
    fn get_cblock_size_rle_returns_one() {
        let hdr_val: u32 = (5u32 << 3) | (1 << 1); // bt_rle, lastBlock=0
        let src = [
            (hdr_val & 0xFF) as u8,
            ((hdr_val >> 8) & 0xFF) as u8,
            ((hdr_val >> 16) & 0xFF) as u8,
        ];
        let mut bp = blockProperties_t {
            blockType: blockType_e::bt_reserved,
            lastBlock: 0,
            origSize: 0,
        };
        let rc = ZSTD_getcBlockSize(&src, &mut bp);
        assert_eq!(rc, 1, "RLE header returns 1 regardless of cSize field");
        assert_eq!(bp.blockType, blockType_e::bt_rle);
        assert_eq!(bp.origSize, 5);
    }

    #[test]
    fn get_cblock_size_reserved_is_error() {
        let hdr_val: u32 = (1u32 << 3) | (3 << 1); // bt_reserved
        let src = [
            (hdr_val & 0xFF) as u8,
            ((hdr_val >> 8) & 0xFF) as u8,
            ((hdr_val >> 16) & 0xFF) as u8,
        ];
        let mut bp = blockProperties_t {
            blockType: blockType_e::bt_raw,
            lastBlock: 0,
            origSize: 0,
        };
        let rc = ZSTD_getcBlockSize(&src, &mut bp);
        assert!(crate::common::error::ERR_isError(rc));
    }

    #[test]
    fn get_cblock_size_rejects_truncated() {
        let mut bp = blockProperties_t {
            blockType: blockType_e::bt_raw,
            lastBlock: 0,
            origSize: 0,
        };
        let rc = ZSTD_getcBlockSize(&[0, 0], &mut bp);
        assert!(crate::common::error::ERR_isError(rc));
    }

    #[test]
    fn overlap_copy8_spreads_with_small_offset() {
        // Pattern at buf[0..1] = 'A'; op=1; offset=1. After the spread,
        // buf[1..=8] should all be 'A'.
        let mut buf = [0u8; 32];
        buf[0] = b'A';
        let mut op = 1usize;
        let mut ip = 0usize;
        ZSTD_overlapCopy8(&mut buf, &mut op, &mut ip, 1);
        for i in 0..8 {
            assert_eq!(buf[1 + i], b'A', "byte {i} should have been spread");
        }
        assert_eq!(op, 9);
    }

    #[test]
    fn overlap_copy8_non_overlap_is_memcpy() {
        // Distinct regions; op-ip=8. Plain 8-byte copy.
        let mut buf = [0u8; 32];
        for (i, slot) in buf.iter_mut().enumerate().take(8) {
            *slot = (i as u8) + 1; // 1..=8
        }
        let mut op = 16usize;
        let mut ip = 0usize;
        ZSTD_overlapCopy8(&mut buf, &mut op, &mut ip, 16);
        for i in 0..8 {
            assert_eq!(buf[16 + i], (i as u8) + 1);
        }
        assert_eq!(op, 24);
        assert_eq!(ip, 8);
    }

    #[test]
    fn decode_literals_block_rle_size1() {
        // Header: set_rle, lhlCode=0 → lhSize=1, litSize = src[0]>>3.
        // src[0] = (litSize << 3) | (set_rle << 0) = (5<<3)|1 = 41.
        // src[1] = rle byte = 0x5A.
        let src = [41u8, 0x5A];
        let mut dst = [0u8; 32];
        let mut dctx = ZSTD_DCtx::new();
        let rc = ZSTD_decodeLiteralsBlock(
            &mut dctx,
            &src,
            &mut dst,
            streaming_operation::not_streaming,
        );
        assert!(
            !crate::common::error::ERR_isError(rc),
            "got error: {}",
            crate::common::error::ERR_getErrorName(rc)
        );
        assert_eq!(rc, 2, "lhSize(1) + 1 RLE byte consumed");
        assert_eq!(dctx.litSize, 5);
        for i in 0..5 {
            assert_eq!(dctx.litExtraBuffer[i], 0x5A);
        }
    }

    #[test]
    fn decode_literals_block_rle_can_use_dst_literal_buffer() {
        let src = [41u8, 0x5A];
        let mut dst = vec![0u8; ZSTD_BLOCKSIZE_MAX + WILDCOPY_OVERLENGTH * 2 + 64];
        let mut dctx = ZSTD_DCtx::new();
        dctx.bmi2 = 0;
        let rc = ZSTD_decodeLiteralsBlock(
            &mut dctx,
            &src,
            &mut dst,
            streaming_operation::not_streaming,
        );
        assert_eq!(rc, 2);
        assert_eq!(dctx.litSize, 5);
        assert_eq!(dctx.litBufferLocation, ZSTD_litLocation_e::ZSTD_in_dst);
        assert!(dctx.litPtr_from_dst);
        assert_eq!(
            &dst[dctx.litPtr_offset..dctx.litPtr_offset + dctx.litSize],
            &[0x5A; 5]
        );
    }

    #[test]
    fn decode_literals_block_rle_can_use_split_literal_buffer() {
        let lit_size = ZSTD_LITBUFFEREXTRASIZE + 128;
        let header = ((lit_size as u32) << 4) | (3 << 2) | 1;
        let src = [
            (header & 0xFF) as u8,
            ((header >> 8) & 0xFF) as u8,
            ((header >> 16) & 0xFF) as u8,
            0xA5,
        ];
        let mut dst = vec![0u8; ZSTD_BLOCKSIZE_MAX];
        let mut dctx = ZSTD_DCtx::new();
        dctx.bmi2 = 0;
        let rc = ZSTD_decodeLiteralsBlock(
            &mut dctx,
            &src,
            &mut dst,
            streaming_operation::not_streaming,
        );
        assert_eq!(rc, 4);
        assert_eq!(dctx.litSize, lit_size);
        assert_eq!(dctx.litBufferLocation, ZSTD_litLocation_e::ZSTD_split);
        assert!(dctx.litPtr_from_dst);
        assert_eq!(dctx.litBufferEnd_offset - dctx.litBuffer_offset, 128);
        assert!(dst[dctx.litBuffer_offset..dctx.litBufferEnd_offset]
            .iter()
            .all(|&b| b == 0xA5));
        assert!(dctx.litExtraBuffer[..ZSTD_LITBUFFEREXTRASIZE]
            .iter()
            .all(|&b| b == 0xA5));
    }

    #[test]
    fn decode_literals_block_basic_size1() {
        // Header: set_basic, lhlCode=0 → lhSize=1, litSize = src[0]>>3.
        // Make litSize=7 so the raw bytes follow directly: 7<<3 | 0 = 56.
        let mut src = vec![56u8];
        src.extend_from_slice(b"ABCDEFG");
        let mut dst = [0u8; 64];
        let mut dctx = ZSTD_DCtx::new();
        let rc = ZSTD_decodeLiteralsBlock(
            &mut dctx,
            &src,
            &mut dst,
            streaming_operation::not_streaming,
        );
        assert!(
            !crate::common::error::ERR_isError(rc),
            "got error: {}",
            crate::common::error::ERR_getErrorName(rc)
        );
        assert_eq!(rc, 1 + 7);
        assert_eq!(dctx.litSize, 7);
        assert_eq!(&dctx.litExtraBuffer[..7], b"ABCDEFG");
    }

    #[test]
    fn decode_literals_block_basic_large_direct_reference_resizes_copy_buffer() {
        let lit_size = ZSTD_LITBUFFEREXTRASIZE + 1024;
        let header = ((lit_size as u32) << 4) | (3 << 2);
        let mut src = vec![
            (header & 0xFF) as u8,
            ((header >> 8) & 0xFF) as u8,
            ((header >> 16) & 0xFF) as u8,
        ];
        src.extend((0..lit_size).map(|i| (i & 0xFF) as u8));
        src.extend(std::iter::repeat(0).take(WILDCOPY_OVERLENGTH + 1));

        let mut dst = vec![0u8; ZSTD_BLOCKSIZE_MAX + lit_size + WILDCOPY_OVERLENGTH * 2 + 1];
        let mut dctx = ZSTD_DCtx::new();
        let rc = ZSTD_decodeLiteralsBlock(
            &mut dctx,
            &src,
            &mut dst,
            streaming_operation::not_streaming,
        );

        assert!(
            !crate::common::error::ERR_isError(rc),
            "got error: {}",
            crate::common::error::ERR_getErrorName(rc)
        );
        assert_eq!(rc, 3 + lit_size);
        assert_eq!(dctx.litSize, lit_size);
        assert_eq!(dctx.litBufferLocation, ZSTD_litLocation_e::ZSTD_not_in_dst);
        assert!(!dctx.litPtr_from_dst);
        assert_eq!(dctx.litExtraBuffer.len(), lit_size);
        assert_eq!(&dctx.litExtraBuffer[..16], &src[3..19]);
        assert_eq!(
            &dctx.litExtraBuffer[lit_size - 16..lit_size],
            &src[3 + lit_size - 16..3 + lit_size]
        );
    }

    #[test]
    fn decode_literals_block_rejects_short_input() {
        let mut dctx = ZSTD_DCtx::new();
        let mut dst = [0u8; 16];
        let rc = ZSTD_decodeLiteralsBlock(
            &mut dctx,
            &[0u8],
            &mut dst,
            streaming_operation::not_streaming,
        );
        assert!(crate::common::error::ERR_isError(rc));
    }

    #[test]
    fn decode_literals_block_compressed_short_header_matches_upstream_1_5_7() {
        // set_compressed, lhlCode=0 uses a 3-byte header:
        // litSize=20, litCSize=0. Upstream 1.5.7 still requires
        // srcSize >= 5 before parsing compressed/repeat literal headers.
        let lhc = (20u32 << 4) | 2;
        let src = [
            (lhc & 0xFF) as u8,
            ((lhc >> 8) & 0xFF) as u8,
            ((lhc >> 16) & 0xFF) as u8,
        ];
        let mut dst = [0u8; 1];
        let mut dctx = ZSTD_DCtx::new();

        let rc = ZSTD_decodeLiteralsBlock(
            &mut dctx,
            &src,
            &mut dst,
            streaming_operation::not_streaming,
        );

        assert_eq!(rc, ERROR(ErrorCode::CorruptionDetected));
    }

    #[test]
    fn decode_sequence_rep0_minus_one_zero_forces_invalid_offset() {
        let mut seq_state = seqState_t {
            DStream: crate::common::bitstream::BIT_DStream_t {
                bitContainer: 1usize << (usize::BITS - 1),
                bitsConsumed: 0,
                ptr: 0,
                start: 0,
                limitPtr: 0,
                src: &[],
            },
            stateLL: ZSTD_fseState::default(),
            stateOffb: ZSTD_fseState::default(),
            stateML: ZSTD_fseState::default(),
            prevOffset: [1, 7, 9],
        };
        let ll_table = [ZSTD_seqSymbol {
            baseValue: 0,
            ..ZSTD_seqSymbol::default()
        }];
        let of_table = [ZSTD_seqSymbol {
            baseValue: 1,
            nbAdditionalBits: 1,
            ..ZSTD_seqSymbol::default()
        }];
        let ml_table = [ZSTD_seqSymbol::default()];

        let seq = ZSTD_decodeSequence(
            &mut seq_state,
            ZSTD_longOffset_e::ZSTD_lo_isRegularOffset,
            &ll_table,
            &of_table,
            &ml_table,
            true,
        );

        assert_eq!(seq.offset, usize::MAX);
        assert_eq!(seq_state.prevOffset[0], usize::MAX);
        assert_eq!(seq_state.prevOffset[1], 1);
        assert_eq!(seq_state.prevOffset[2], 7);
    }

    #[test]
    fn seqheaders_probe_nbseq_zero() {
        // nbSeq == 0, single byte, ends immediately.
        let src = [0u8];
        let mut out = SeqBlockHeader {
            nbSeq: -1,
            LLtype: SymbolEncodingType_e::set_basic,
            OFtype: SymbolEncodingType_e::set_basic,
            MLtype: SymbolEncodingType_e::set_basic,
            headerSize: 0,
        };
        let consumed = ZSTD_decodeSeqHeaders_probe(&src, &mut out);
        assert_eq!(consumed, 1);
        assert_eq!(out.nbSeq, 0);
    }

    #[test]
    fn seqheaders_probe_nbseq_small() {
        // nbSeq = 0x42 (66, fits in one byte), then encoding byte.
        // Encoding: LL=compressed(2), OF=rle(1), ML=repeat(3), reserved=0.
        // Layout: (LL<<6) | (OF<<4) | (ML<<2) | reserved
        //       = (2<<6) | (1<<4) | (3<<2) | 0 = 0x9C
        let src = [0x42u8, 0x9C];
        let mut out = SeqBlockHeader {
            nbSeq: 0,
            LLtype: SymbolEncodingType_e::set_basic,
            OFtype: SymbolEncodingType_e::set_basic,
            MLtype: SymbolEncodingType_e::set_basic,
            headerSize: 0,
        };
        let consumed = ZSTD_decodeSeqHeaders_probe(&src, &mut out);
        assert_eq!(consumed, 2);
        assert_eq!(out.nbSeq, 0x42);
        assert_eq!(out.LLtype, SymbolEncodingType_e::set_compressed);
        assert_eq!(out.OFtype, SymbolEncodingType_e::set_rle);
        assert_eq!(out.MLtype, SymbolEncodingType_e::set_repeat);
    }

    #[test]
    fn seqheaders_probe_nbseq_two_byte() {
        // nbSeq between 0x80 and 0xFE: `((b0-0x80)<<8) + b1`.
        // Pick nbSeq = 0x155 = (0x81 - 0x80) << 8 | 0x55 = 0x155.
        // Need encoding byte too.
        let src = [0x81, 0x55, 0x00];
        let mut out = SeqBlockHeader {
            nbSeq: 0,
            LLtype: SymbolEncodingType_e::set_basic,
            OFtype: SymbolEncodingType_e::set_basic,
            MLtype: SymbolEncodingType_e::set_basic,
            headerSize: 0,
        };
        let consumed = ZSTD_decodeSeqHeaders_probe(&src, &mut out);
        assert_eq!(consumed, 3);
        assert_eq!(out.nbSeq, 0x155);
    }

    #[test]
    fn seqheaders_probe_long_varint() {
        // nbSeq = LONGNBSEQ + MEM_readLE16(next 2 bytes). b0 = 0xFF.
        let src = [0xFFu8, 0x78, 0x56, 0x00];
        let mut out = SeqBlockHeader {
            nbSeq: 0,
            LLtype: SymbolEncodingType_e::set_basic,
            OFtype: SymbolEncodingType_e::set_basic,
            MLtype: SymbolEncodingType_e::set_basic,
            headerSize: 0,
        };
        let consumed = ZSTD_decodeSeqHeaders_probe(&src, &mut out);
        assert_eq!(consumed, 4);
        assert_eq!(out.nbSeq, LONGNBSEQ + 0x5678);
    }

    #[test]
    fn seqheaders_probe_rejects_reserved_bits() {
        let src = [0x10, 0x03]; // nbSeq=0x10, encoding byte sets reserved=0b11
        let mut out = SeqBlockHeader {
            nbSeq: 0,
            LLtype: SymbolEncodingType_e::set_basic,
            OFtype: SymbolEncodingType_e::set_basic,
            MLtype: SymbolEncodingType_e::set_basic,
            headerSize: 0,
        };
        let rc = ZSTD_decodeSeqHeaders_probe(&src, &mut out);
        assert!(crate::common::error::ERR_isError(rc));
    }

    #[test]
    fn seqheaders_probe_rejects_extraneous_after_nbSeq_zero() {
        let src = [0u8, 0xAB]; // nbSeq=0 but trailing byte
        let mut out = SeqBlockHeader {
            nbSeq: 0,
            LLtype: SymbolEncodingType_e::set_basic,
            OFtype: SymbolEncodingType_e::set_basic,
            MLtype: SymbolEncodingType_e::set_basic,
            headerSize: 0,
        };
        let rc = ZSTD_decodeSeqHeaders_probe(&src, &mut out);
        assert!(crate::common::error::ERR_isError(rc));
    }

    #[test]
    fn decompress_block_internal_raw_literals_zero_sequences() {
        // Build a minimal compressed block payload by hand:
        //   - Literals block: set_basic, lhlCode=0, litSize=5 → header
        //     byte = (5<<3)|(0<<2)|(0) = 40. Payload: "HELLO".
        //   - Sequence block: single zero byte (nbSeq=0).
        let mut src = vec![40u8];
        src.extend_from_slice(b"HELLO");
        src.push(0u8);

        let mut dst = vec![0u8; 32];
        let mut dctx = ZSTD_DCtx::new();
        ZSTD_buildDefaultSeqTables(&mut dctx);
        let mut entropy = ZSTD_decoder_entropy_rep::default();
        let out = ZSTD_decompressBlock_internal(
            &mut dctx,
            &mut entropy,
            &mut dst,
            0,
            &src,
            streaming_operation::not_streaming,
        );
        assert!(
            !crate::common::error::ERR_isError(out),
            "err: {}",
            crate::common::error::ERR_getErrorName(out)
        );
        assert_eq!(out, 5);
        assert_eq!(&dst[..5], b"HELLO");
    }

    #[test]
    fn decompress_block_internal_nonempty_sequences_set_fse_entropy_on_body_error() {
        // Upstream sequence bodies set dctx->fseEntropy as soon as
        // nbSeq is non-zero, before BIT_initDStream can fail. This
        // block has a valid literal section and sequence header, but
        // no sequence bitstream payload, so the body errors after the
        // entropy flag is made repeatable.
        let src = [0u8, 1u8, 0u8];
        let mut dst = vec![0u8; 32];
        let mut dctx = ZSTD_DCtx::new();
        ZSTD_buildDefaultSeqTables(&mut dctx);
        let mut entropy = ZSTD_decoder_entropy_rep::default();

        let out = ZSTD_decompressBlock_internal(
            &mut dctx,
            &mut entropy,
            &mut dst,
            0,
            &src,
            streaming_operation::not_streaming,
        );

        assert!(crate::common::error::ERR_isError(out));
        assert_eq!(dctx.fseEntropy, 1);
    }

    #[test]
    fn decompress_block_internal_rle_literals_zero_sequences() {
        // Literals block: set_rle, lhlCode=0, litSize=10 → header
        // byte = (10<<3)|(0<<2)|1 = 81. Payload: 1 RLE byte.
        // Sequence block: nbSeq=0.
        let src = [81u8, b'Q', 0u8];
        let mut dst = vec![0u8; 32];
        let mut dctx = ZSTD_DCtx::new();
        ZSTD_buildDefaultSeqTables(&mut dctx);
        let mut entropy = ZSTD_decoder_entropy_rep::default();
        let out = ZSTD_decompressBlock_internal(
            &mut dctx,
            &mut entropy,
            &mut dst,
            0,
            &src,
            streaming_operation::not_streaming,
        );
        assert!(!crate::common::error::ERR_isError(out));
        assert_eq!(out, 10);
        for b in dst.iter().take(10) {
            assert_eq!(*b, b'Q');
        }
    }

    #[test]
    fn decompress_sequences_nb_zero_emits_only_literals() {
        // nbSeq = 0: the function just copies the entire lit buffer
        // into dst and returns its size.
        let lit = b"literalpayload";
        let mut dst = vec![0u8; 32];
        let mut ll_tbl = vec![ZSTD_seqSymbol::default(); SEQSYMBOL_TABLE_SIZE(LLFSELog)];
        let mut of_tbl = vec![ZSTD_seqSymbol::default(); SEQSYMBOL_TABLE_SIZE(OffFSELog)];
        let mut ml_tbl = vec![ZSTD_seqSymbol::default(); SEQSYMBOL_TABLE_SIZE(MLFSELog)];
        let mut dctx = ZSTD_DCtx::new();
        ZSTD_buildDefaultSeqTables(&mut dctx);
        // Use DCtx's default tables as the concrete LL/OF/ML tables
        // (they're valid and ready for use).
        ll_tbl.clone_from(&dctx.LLTable);
        of_tbl.clone_from(&dctx.OFTable);
        ml_tbl.clone_from(&dctx.MLTable);

        let mut rep = ZSTD_decoder_entropy_rep::default();
        let out = ZSTD_decompressSequences_body(
            &mut dst,
            0,
            &[], // empty ext_history
            &[], // no FSE stream
            0,   // no sequences
            lit,
            lit.len(),
            &ll_tbl,
            &of_tbl,
            &ml_tbl,
            &mut rep,
        );
        assert!(
            !crate::common::error::ERR_isError(out),
            "error: {}",
            crate::common::error::ERR_getErrorName(out)
        );
        assert_eq!(out, lit.len());
        assert_eq!(&dst[..out], lit);
    }

    #[test]
    fn decompress_sequences_dst_too_small_errors() {
        let lit = b"way too many literals";
        let mut dst = vec![0u8; 4]; // smaller than lit
        let mut dctx = ZSTD_DCtx::new();
        ZSTD_buildDefaultSeqTables(&mut dctx);
        let mut rep = ZSTD_decoder_entropy_rep::default();
        let rc = ZSTD_decompressSequences_body(
            &mut dst,
            0,
            &[],
            &[],
            0,
            lit,
            lit.len(),
            &dctx.LLTable,
            &dctx.OFTable,
            &dctx.MLTable,
            &mut rep,
        );
        assert!(crate::common::error::ERR_isError(rc));
    }

    #[test]
    fn exec_sequence_large_offset_copies_literals_and_match() {
        // dst[0..16] prefix = "0123456789ABCDEF"; execute one sequence
        // at op=16 with litLength=5 ("HELLO"), offset=16, matchLength=16.
        // After literal copy, oLitEnd=21. match_src = 21-16 = 5, so
        // the match reads dst[5..21] = "56789ABCDEFHELLO" and writes
        // it to dst[21..37]. Verifies offset >= WILDCOPY_VECLEN path.
        let mut dst = vec![0u8; 64];
        dst[..16].copy_from_slice(b"0123456789ABCDEF");
        let litBuf = b"HELLO";
        let seq = seq_t {
            litLength: 5,
            matchLength: 16,
            offset: 16,
        };
        let mut litPtr = 0usize;
        let written = ZSTD_execSequence(&mut dst, 16, seq, &[], litBuf, 0, &mut litPtr);
        assert!(!crate::common::error::ERR_isError(written));
        assert_eq!(written, 21);
        assert_eq!(&dst[16..21], b"HELLO");
        assert_eq!(&dst[21..37], b"56789ABCDEFHELLO");
        assert_eq!(litPtr, 5);
    }

    #[test]
    fn exec_sequence_fast_boundary_uses_single_wildcopy_stamps() {
        // The fast helper mirrors upstream's copy16 + wildcopy shape.
        // At length 17 it may use exactly one extra 16-byte stamp, but
        // must not issue the second unrolled stamp that would read
        // beyond the caller-provided WILDCOPY_OVERLENGTH slack.
        let op = 32usize;
        let lit_len = 17usize;
        let match_len = 17usize;
        let offset = 32usize;
        let mut dst = vec![0u8; op + lit_len + match_len + WILDCOPY_OVERLENGTH];
        for (i, b) in dst.iter_mut().take(op).enumerate() {
            *b = b'a' + (i % 26) as u8;
        }
        let lit = vec![b'0'; lit_len + 16];

        let written = unsafe {
            ZSTD_execSequence_rawLit_fast(&mut dst, op, lit_len, match_len, offset, lit.as_ptr(), 0)
        };

        assert_eq!(written, lit_len + match_len);
        assert_eq!(&dst[op..op + lit_len], &lit[..lit_len]);
        let expected_match = {
            let mut v = Vec::new();
            v.extend_from_slice(&dst[op + lit_len - offset..op]);
            v.extend_from_slice(&lit[..match_len - v.len()]);
            v
        };
        assert_eq!(
            &dst[op + lit_len..op + lit_len + match_len],
            expected_match.as_slice()
        );
    }

    #[test]
    fn exec_sequence_small_offset_spreads_pattern() {
        // Prefix "AB" at op=0..2; lit="CD" (2 bytes); then
        // matchLength=8 with offset=2 → copies "ABAB..." pattern 4
        // times after the literals. Total written = 2+8 = 10.
        let mut dst = vec![0u8; 32];
        dst[..2].copy_from_slice(b"AB");
        let litBuf = b"CD";
        let seq = seq_t {
            litLength: 2,
            matchLength: 8,
            offset: 2,
        };
        let mut litPtr = 0usize;
        let written = ZSTD_execSequence(&mut dst, 2, seq, &[], litBuf, 0, &mut litPtr);
        assert!(!crate::common::error::ERR_isError(written));
        assert_eq!(written, 10);
        assert_eq!(&dst[2..4], b"CD");
        // After "ABCD", match-copy offset 2 reads "CD" → spreads "CDCDCDCD".
        // With overlapCopy8 on offset=2: first 8 bytes of match = "CDCDCDCD".
        assert_eq!(&dst[4..12], b"CDCDCDCD");
    }

    #[test]
    fn exec_sequence_offset_beyond_prefix_errors() {
        let mut dst = vec![0u8; 16];
        let litBuf = b"X";
        let seq = seq_t {
            litLength: 1,
            matchLength: 4,
            offset: 10, // larger than op(0)+litLength(1) = 1
        };
        let mut litPtr = 0;
        let rc = ZSTD_execSequence(&mut dst, 0, seq, &[], litBuf, 0, &mut litPtr);
        assert!(crate::common::error::ERR_isError(rc));
    }

    #[test]
    fn exec_sequence_ext_history_resolves_within_history() {
        // ext_history holds the prior history segment; offset reaches
        // back into it without spilling into dst.
        // ext_history = "0123456789", dst is empty for op=0,
        // litBuf = "L"; offset=8 reaches back 8 bytes from oLitEnd=1
        // → into_ext=7, ext_start = 10 - 7 = 3.
        // matchLength=5 fits entirely in ext_history → reads
        // ext_history[3..8] = "34567" into dst[1..6].
        let mut dst = vec![0u8; 16];
        let ext_history = b"0123456789";
        let litBuf = b"L";
        let seq = seq_t {
            litLength: 1,
            matchLength: 5,
            offset: 8,
        };
        let mut litPtr = 0usize;
        let written = ZSTD_execSequence(&mut dst, 0, seq, ext_history, litBuf, 0, &mut litPtr);
        assert!(!crate::common::error::ERR_isError(written));
        assert_eq!(written, 6);
        assert_eq!(&dst[..1], b"L");
        assert_eq!(&dst[1..6], b"34567");
    }

    #[test]
    fn exec_sequence_ext_history_spills_into_dst_prefix() {
        // ext_history = "AB" (2 bytes); dst[0..2] starts as "MN".
        // op=2, litLength=0, offset=4, matchLength=6.
        // oLitEnd=2, into_ext = offset - oLitEnd = 2; ext_start = 0.
        // First 2 bytes of match come from ext_history[0..2] = "AB".
        // Remaining 4 bytes come from dst[0..4] forward: "MN" then
        // the freshly-written "AB". Final layout dst[..8] = "MNABMNAB".
        let mut dst = vec![0u8; 16];
        dst[..2].copy_from_slice(b"MN");
        let ext_history = b"AB";
        let seq = seq_t {
            litLength: 0,
            matchLength: 6,
            offset: 4,
        };
        let mut litPtr = 0usize;
        let written = ZSTD_execSequence(&mut dst, 2, seq, ext_history, b"", 0, &mut litPtr);
        assert!(!crate::common::error::ERR_isError(written));
        assert_eq!(written, 6);
        assert_eq!(&dst[..8], b"MNABMNAB");
    }

    #[test]
    fn build_default_seq_tables_populates_all_three() {
        let mut dctx = ZSTD_DCtx::new();
        ZSTD_buildDefaultSeqTables(&mut dctx);
        // tableLog stored in slot-0 header baseValue.
        assert_eq!(seq_header_read(&dctx.LLTable).tableLog, LL_defaultNormLog);
        assert_eq!(seq_header_read(&dctx.OFTable).tableLog, OF_defaultNormLog);
        assert_eq!(seq_header_read(&dctx.MLTable).tableLog, ML_defaultNormLog);
        // Decoding entries must each have non-zero nbBits (every slot
        // in a valid FSE DTable consumes at least 1 bit).
        for i in 1..=(1usize << LL_defaultNormLog) {
            assert!(dctx.LLTable[i].nbBits > 0, "LL slot {i}");
        }
    }

    #[test]
    fn decode_seq_headers_full_nb_seq_zero() {
        // nbSeq == 0: single byte consumed, no FSE build triggered.
        let mut dctx = ZSTD_DCtx::new();
        ZSTD_buildDefaultSeqTables(&mut dctx);
        let mut nbSeq: i32 = -1;
        let rc = ZSTD_decodeSeqHeaders(&mut dctx, &mut nbSeq, &[0u8]);
        assert!(!crate::common::error::ERR_isError(rc));
        assert_eq!(rc, 1);
        assert_eq!(nbSeq, 0);
    }

    #[test]
    fn decode_seq_headers_full_all_basic() {
        // nbSeq=1, LL=basic, OF=basic, ML=basic (encoding-byte 0x00).
        // No FSE table payload needed when every table type is basic.
        let src = [0x01u8, 0x00];
        let mut dctx = ZSTD_DCtx::new();
        ZSTD_buildDefaultSeqTables(&mut dctx);
        let mut nbSeq: i32 = 0;
        let rc = ZSTD_decodeSeqHeaders(&mut dctx, &mut nbSeq, &src);
        assert!(
            !crate::common::error::ERR_isError(rc),
            "got: {}",
            crate::common::error::ERR_getErrorName(rc)
        );
        assert_eq!(rc, 2);
        assert_eq!(nbSeq, 1);
    }

    #[test]
    fn decode_seq_headers_repeat_uses_global_fse_entropy_flag() {
        // Upstream passes dctx->fseEntropy as flagRepeatTable for all
        // three sequence tables. A repeat OF table is valid even if the
        // previous active OF table was the default table.
        let src = [0x01u8, 0x30]; // nbSeq=1, LL=basic, OF=repeat, ML=basic.
        let mut dctx = ZSTD_DCtx::new();
        dctx.fseEntropy = 1;
        dctx.of_default_active = true;
        let mut nbSeq = 0;

        let rc = ZSTD_decodeSeqHeaders(&mut dctx, &mut nbSeq, &src);

        assert_eq!(rc, 2);
        assert_eq!(nbSeq, 1);
        assert!(dctx.of_default_active);
    }

    #[test]
    fn decode_seq_headers_repeat_rejects_without_global_fse_entropy() {
        // Per-table breadcrumbs must not make set_repeat valid when
        // upstream's single dctx->fseEntropy flag is unset.
        let src = [0x01u8, 0x30]; // nbSeq=1, LL=basic, OF=repeat, ML=basic.
        let mut dctx = ZSTD_DCtx::new();
        dctx.fseEntropy = 0;
        dctx.fse_of_fresh = true;
        let mut nbSeq = 0;

        let rc = ZSTD_decodeSeqHeaders(&mut dctx, &mut nbSeq, &src);

        assert!(crate::common::error::ERR_isError(rc));
    }

    #[test]
    fn decode_seq_headers_full_rle_triple() {
        // nbSeq=1, LL=rle, OF=rle, ML=rle → encoding byte = (1<<6)|(1<<4)|(1<<2) = 0x54.
        // Each RLE type consumes 1 byte after the header.
        let src = [0x01, 0x54, 0, 0, 0]; // nbSeq, encByte, LL RLE symbol, OF RLE symbol, ML RLE symbol
        let mut dctx = ZSTD_DCtx::new();
        let mut nbSeq: i32 = 0;
        let rc = ZSTD_decodeSeqHeaders(&mut dctx, &mut nbSeq, &src);
        assert!(!crate::common::error::ERR_isError(rc), "err");
        // 1 (nbSeq) + 1 (encByte) + 3 (three rle symbol bytes) = 5.
        assert_eq!(rc, 5);
        assert_eq!(nbSeq, 1);
    }

    #[test]
    fn build_seq_table_rle_writes_single_entry() {
        let mut dt = vec![ZSTD_seqSymbol::default(); 4];
        let consumed = ZSTD_buildSeqTable(
            &mut dt,
            SymbolEncodingType_e::set_rle,
            MaxLL,
            LLFSELog,
            &[5u8], // symbol 5
            &LL_base,
            &LL_bits,
            None,
            false,
            0,
            10,
        );
        assert_eq!(consumed, 1);
        assert_eq!(dt[1].baseValue, LL_base[5]);
        assert_eq!(dt[1].nbAdditionalBits, LL_bits[5]);
    }

    #[test]
    fn build_seq_table_rle_rejects_symbol_over_max() {
        let mut dt = vec![ZSTD_seqSymbol::default(); 4];
        let rc = ZSTD_buildSeqTable(
            &mut dt,
            SymbolEncodingType_e::set_rle,
            5,
            LLFSELog,
            &[10u8],
            &LL_base,
            &LL_bits,
            None,
            false,
            0,
            10,
        );
        assert!(crate::common::error::ERR_isError(rc));
    }

    #[test]
    fn build_seq_table_basic_returns_zero_consumed() {
        let mut dt = vec![ZSTD_seqSymbol::default(); SEQSYMBOL_TABLE_SIZE(LLFSELog)];
        let default = vec![
            ZSTD_seqSymbol {
                baseValue: 99,
                nbBits: 7,
                nbAdditionalBits: 1,
                nextState: 3,
            };
            SEQSYMBOL_TABLE_SIZE(LLFSELog)
        ];
        let consumed = ZSTD_buildSeqTable(
            &mut dt,
            SymbolEncodingType_e::set_basic,
            MaxLL,
            LLFSELog,
            &[],
            &LL_base,
            &LL_bits,
            Some(&default),
            false,
            0,
            10,
        );
        assert_eq!(consumed, 0);
        // Default table copied in.
        assert_eq!(dt[5].baseValue, 99);
    }

    #[test]
    fn build_seq_table_repeat_requires_flag() {
        let mut dt = vec![ZSTD_seqSymbol::default(); 16];
        let rc = ZSTD_buildSeqTable(
            &mut dt,
            SymbolEncodingType_e::set_repeat,
            MaxLL,
            LLFSELog,
            &[],
            &LL_base,
            &LL_bits,
            None,
            false, // flag NOT set → error
            0,
            10,
        );
        assert!(crate::common::error::ERR_isError(rc));

        let ok = ZSTD_buildSeqTable(
            &mut dt,
            SymbolEncodingType_e::set_repeat,
            MaxLL,
            LLFSELog,
            &[],
            &LL_base,
            &LL_bits,
            None,
            true, // flag set → ok, 0 bytes consumed
            0,
            10,
        );
        assert_eq!(ok, 0);
    }

    #[test]
    fn copy4_copy8_sanity() {
        let src = [1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut dst = [0u8; 8];
        ZSTD_copy4(&mut dst, &src);
        assert_eq!(&dst[..4], &[1, 2, 3, 4]);
        ZSTD_copy8(&mut dst, &src);
        assert_eq!(&dst, &[1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
