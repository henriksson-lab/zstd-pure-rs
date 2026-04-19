//! Translation of `lib/common/bitstream.h`.
//!
//! Both encoder (`BIT_CStream_*`) and decoder (`BIT_DStream_*`) are
//! implemented. Upstream stores raw pointers into the caller's buffer;
//! in Rust we carry an explicit borrowed slice plus byte-offset indices
//! so the same per-function mapping holds (`start` / `ptr` / `limitPtr`
//! / `endPtr` are byte indices relative to the slice start).

use crate::common::bits::ZSTD_highbit32;
use crate::common::error::{ErrorCode, ERROR};
use crate::common::mem::{MEM_readLEST, MEM_writeLEST};

// ---- status codes (mirror upstream `BIT_DStream_status`) ----
pub const BIT_DStream_unfinished: u32 = 0;
pub const BIT_DStream_endOfBuffer: u32 = 1;
pub const BIT_DStream_completed: u32 = 2;
pub const BIT_DStream_overflow: u32 = 3;

/// Bit-container width in bytes. `size_t` upstream; `usize` here.
const CONTAINER_BYTES: usize = core::mem::size_of::<usize>();
const CONTAINER_BITS: u32 = (CONTAINER_BYTES as u32) * 8;

// ========================================================================
// Encoder (BIT_CStream_t)
// ========================================================================

/// Mirrors the upstream `BIT_CStream_t` struct. Rust adds a borrow of
/// the destination buffer so the `startPtr` / `ptr` / `endPtr` fields —
/// kept as byte offsets — can actually drive writes. `bitPos` is the
/// count of bits currently held in `bitContainer`.
#[derive(Debug)]
pub struct BIT_CStream_t<'a> {
    pub bitContainer: usize,
    pub bitPos: u32,
    pub startPtr: usize,
    pub ptr: usize,
    pub endPtr: usize,
    pub dst: &'a mut [u8],
}

/// Port of `BIT_initCStream`.
///
/// Rust signature note: upstream C takes a pre-allocated `BIT_CStream_t*`
/// plus `(ptr, cap)` and writes the pointer into the struct. Rust's
/// borrow checker won't let us reseat a `&'a mut [u8]` field, so we
/// return `(BIT_CStream_t, status)` instead. Logic complexity and
/// observable state transitions are unchanged.
pub fn BIT_initCStream<'a>(
    startPtr: &'a mut [u8],
    dstCapacity: usize,
) -> (BIT_CStream_t<'a>, usize) {
    let (endPtr, rc) = if dstCapacity <= CONTAINER_BYTES {
        (0, ERROR(ErrorCode::DstSizeTooSmall))
    } else {
        (dstCapacity - CONTAINER_BYTES, 0)
    };
    let bitC = BIT_CStream_t {
        bitContainer: 0,
        bitPos: 0,
        startPtr: 0,
        ptr: 0,
        endPtr,
        dst: startPtr,
    };
    (bitC, rc)
}

/// Port of `BIT_addBits` — safe variant; masks the high bits before OR-ing.
#[inline]
pub fn BIT_addBits(bitC: &mut BIT_CStream_t, value: usize, nbBits: u32) {
    debug_assert!(nbBits < 32);
    debug_assert!(nbBits + bitC.bitPos < CONTAINER_BITS);
    bitC.bitContainer |= BIT_getLowerBits(value, nbBits) << bitC.bitPos;
    bitC.bitPos += nbBits;
}

/// Port of `BIT_addBitsFast` — caller guarantees `value>>nbBits == 0`.
#[inline]
pub fn BIT_addBitsFast(bitC: &mut BIT_CStream_t, value: usize, nbBits: u32) {
    debug_assert!(value >> nbBits == 0);
    debug_assert!(nbBits + bitC.bitPos < CONTAINER_BITS);
    bitC.bitContainer |= value << bitC.bitPos;
    bitC.bitPos += nbBits;
}

/// Port of `BIT_flushBits` — safe variant: caps `ptr` at `endPtr`.
#[inline]
pub fn BIT_flushBits(bitC: &mut BIT_CStream_t) {
    let nbBytes = (bitC.bitPos >> 3) as usize;
    debug_assert!(bitC.bitPos < CONTAINER_BITS);
    MEM_writeLEST(&mut bitC.dst[bitC.ptr..], bitC.bitContainer);
    bitC.ptr += nbBytes;
    if bitC.ptr > bitC.endPtr {
        bitC.ptr = bitC.endPtr;
    }
    bitC.bitPos &= 7;
    if nbBytes * 8 >= CONTAINER_BITS as usize {
        bitC.bitContainer = 0;
    } else {
        bitC.bitContainer >>= nbBytes * 8;
    }
}

/// Port of `BIT_flushBitsFast` — unsafe variant: no endPtr cap.
#[inline]
pub fn BIT_flushBitsFast(bitC: &mut BIT_CStream_t) {
    let nbBytes = (bitC.bitPos >> 3) as usize;
    debug_assert!(bitC.bitPos < CONTAINER_BITS);
    MEM_writeLEST(&mut bitC.dst[bitC.ptr..], bitC.bitContainer);
    bitC.ptr += nbBytes;
    bitC.bitPos &= 7;
    if nbBytes * 8 >= CONTAINER_BITS as usize {
        bitC.bitContainer = 0;
    } else {
        bitC.bitContainer >>= nbBytes * 8;
    }
}

/// Port of `BIT_closeCStream`.
pub fn BIT_closeCStream(bitC: &mut BIT_CStream_t) -> usize {
    BIT_addBitsFast(bitC, 1, 1); // end-marker bit
    BIT_flushBits(bitC);
    if bitC.ptr >= bitC.endPtr {
        return 0;
    }
    (bitC.ptr - bitC.startPtr) + (bitC.bitPos > 0) as usize
}

// ========================================================================
// Decoder (BIT_DStream_t)
// ========================================================================

/// Mirrors the upstream struct. `src` is the caller's buffer (Rust adds an
/// explicit borrow since we can't smuggle a raw pointer) and `ptr` /
/// `start` / `limitPtr` are byte offsets into `src`.
#[derive(Debug, Clone)]
pub struct BIT_DStream_t<'a> {
    pub bitContainer: usize,
    pub bitsConsumed: u32,
    pub ptr: usize,
    pub start: usize,
    pub limitPtr: usize,
    pub src: &'a [u8],
}

// Manual impl: derive needs `'a: 'static`, which we can't promise here.
#[allow(clippy::derivable_impls)]
impl Default for BIT_DStream_t<'_> {
    fn default() -> Self {
        Self {
            bitContainer: 0,
            bitsConsumed: 0,
            ptr: 0,
            start: 0,
            limitPtr: 0,
            src: &[],
        }
    }
}

/// Port of `BIT_initDStream`.
pub fn BIT_initDStream<'a>(bitD: &mut BIT_DStream_t<'a>, srcBuffer: &'a [u8], srcSize: usize) -> usize {
    if srcSize < 1 {
        *bitD = BIT_DStream_t::default();
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    let src = &srcBuffer[..srcSize];
    bitD.src = src;
    bitD.start = 0;
    bitD.limitPtr = CONTAINER_BYTES;

    if srcSize >= CONTAINER_BYTES {
        bitD.ptr = srcSize - CONTAINER_BYTES;
        bitD.bitContainer = MEM_readLEST(&src[bitD.ptr..]);
        let lastByte = src[srcSize - 1];
        if lastByte == 0 {
            return ERROR(ErrorCode::Generic);
        }
        bitD.bitsConsumed = 8 - ZSTD_highbit32(lastByte as u32);
    } else {
        bitD.ptr = 0;
        // Load src[0] and fill the rest of the container from the
        // trailing bytes, matching the fall-through switch upstream.
        let mut c = src[0] as usize;
        match srcSize {
            7 => {
                c += (src[6] as usize) << (CONTAINER_BITS as usize - 16);
                c += (src[5] as usize) << (CONTAINER_BITS as usize - 24);
                c += (src[4] as usize) << (CONTAINER_BITS as usize - 32);
                c += (src[3] as usize) << 24;
                c += (src[2] as usize) << 16;
                c += (src[1] as usize) << 8;
            }
            6 => {
                c += (src[5] as usize) << (CONTAINER_BITS as usize - 24);
                c += (src[4] as usize) << (CONTAINER_BITS as usize - 32);
                c += (src[3] as usize) << 24;
                c += (src[2] as usize) << 16;
                c += (src[1] as usize) << 8;
            }
            5 => {
                c += (src[4] as usize) << (CONTAINER_BITS as usize - 32);
                c += (src[3] as usize) << 24;
                c += (src[2] as usize) << 16;
                c += (src[1] as usize) << 8;
            }
            4 => {
                c += (src[3] as usize) << 24;
                c += (src[2] as usize) << 16;
                c += (src[1] as usize) << 8;
            }
            3 => {
                c += (src[2] as usize) << 16;
                c += (src[1] as usize) << 8;
            }
            2 => {
                c += (src[1] as usize) << 8;
            }
            _ => {}
        }
        bitD.bitContainer = c;
        let lastByte = src[srcSize - 1];
        if lastByte == 0 {
            return ERROR(ErrorCode::CorruptionDetected);
        }
        bitD.bitsConsumed = 8 - ZSTD_highbit32(lastByte as u32);
        bitD.bitsConsumed += (CONTAINER_BYTES - srcSize) as u32 * 8;
    }
    srcSize
}

/// Port of `BIT_getUpperBits`.
#[inline]
pub fn BIT_getUpperBits(bitContainer: usize, start: u32) -> usize {
    bitContainer >> start
}

/// Port of `BIT_getMiddleBits`.
#[inline]
pub fn BIT_getMiddleBits(bitContainer: usize, start: u32, nbBits: u32) -> usize {
    let reg_mask = CONTAINER_BITS - 1;
    let mask = (1usize << nbBits).wrapping_sub(1);
    (bitContainer >> (start & reg_mask)) & mask
}

/// Port of `BIT_getLowerBits`.
#[inline]
pub fn BIT_getLowerBits(bitContainer: usize, nbBits: u32) -> usize {
    let mask = (1usize << nbBits).wrapping_sub(1);
    bitContainer & mask
}

/// Port of `BIT_lookBits`. Upstream comment explicitly allows the
/// `start` computation to wrap — if `bitsConsumed + nbBits >
/// CONTAINER_BITS`, the caller is reading into already-consumed
/// territory (corrupt bitstream), and the value returned is
/// undefined-but-safe. Rust needs `wrapping_sub` to avoid trapping.
#[inline]
pub fn BIT_lookBits(bitD: &BIT_DStream_t, nbBits: u32) -> usize {
    let start = CONTAINER_BITS
        .wrapping_sub(bitD.bitsConsumed)
        .wrapping_sub(nbBits);
    BIT_getMiddleBits(bitD.bitContainer, start, nbBits)
}

/// Port of `BIT_lookBitsFast`. Requires `nbBits >= 1`.
#[inline]
pub fn BIT_lookBitsFast(bitD: &BIT_DStream_t, nbBits: u32) -> usize {
    debug_assert!(nbBits >= 1);
    let reg_mask = CONTAINER_BITS - 1;
    (bitD.bitContainer << (bitD.bitsConsumed & reg_mask))
        >> ((reg_mask + 1 - nbBits) & reg_mask)
}

/// Port of `BIT_skipBits`.
#[inline]
pub fn BIT_skipBits(bitD: &mut BIT_DStream_t, nbBits: u32) {
    bitD.bitsConsumed += nbBits;
}

/// Port of `BIT_readBits`.
#[inline]
pub fn BIT_readBits(bitD: &mut BIT_DStream_t, nbBits: u32) -> usize {
    let v = BIT_lookBits(bitD, nbBits);
    BIT_skipBits(bitD, nbBits);
    v
}

/// Port of `BIT_readBitsFast`.
#[inline]
pub fn BIT_readBitsFast(bitD: &mut BIT_DStream_t, nbBits: u32) -> usize {
    debug_assert!(nbBits >= 1);
    let v = BIT_lookBitsFast(bitD, nbBits);
    BIT_skipBits(bitD, nbBits);
    v
}

/// Port of `BIT_reloadDStream_internal`.
#[inline]
fn BIT_reloadDStream_internal(bitD: &mut BIT_DStream_t) -> u32 {
    bitD.ptr -= (bitD.bitsConsumed >> 3) as usize;
    bitD.bitsConsumed &= 7;
    bitD.bitContainer = MEM_readLEST(&bitD.src[bitD.ptr..]);
    BIT_DStream_unfinished
}

/// Port of `BIT_reloadDStreamFast`.
#[inline]
pub fn BIT_reloadDStreamFast(bitD: &mut BIT_DStream_t) -> u32 {
    if bitD.ptr < bitD.limitPtr {
        return BIT_DStream_overflow;
    }
    BIT_reloadDStream_internal(bitD)
}

/// Port of `BIT_reloadDStream`.
pub fn BIT_reloadDStream(bitD: &mut BIT_DStream_t) -> u32 {
    if bitD.bitsConsumed > CONTAINER_BITS {
        // Mirror upstream's behaviour: on overflow, force ptr to a
        // zero-filled region so subsequent reads return 0. In Rust we
        // just mark the state and avoid further progress.
        return BIT_DStream_overflow;
    }
    debug_assert!(bitD.ptr >= bitD.start);

    if bitD.ptr >= bitD.limitPtr {
        return BIT_reloadDStream_internal(bitD);
    }
    if bitD.ptr == bitD.start {
        if bitD.bitsConsumed < CONTAINER_BITS {
            return BIT_DStream_endOfBuffer;
        }
        return BIT_DStream_completed;
    }
    // start < ptr < limitPtr: cautious update. Use a subtract-vs-cap
    // compare rather than an unsigned `ptr - nbBytes` (which would
    // underflow in Rust when nbBytes > ptr - start).
    let mut nbBytes = (bitD.bitsConsumed >> 3) as usize;
    let mut result = BIT_DStream_unfinished;
    if nbBytes > bitD.ptr - bitD.start {
        nbBytes = bitD.ptr - bitD.start;
        result = BIT_DStream_endOfBuffer;
    }
    bitD.ptr -= nbBytes;
    bitD.bitsConsumed -= (nbBytes as u32) * 8;
    bitD.bitContainer = MEM_readLEST(&bitD.src[bitD.ptr..]);
    result
}

/// Port of `BIT_endOfDStream`.
#[inline]
pub fn BIT_endOfDStream(bitD: &BIT_DStream_t) -> u32 {
    (bitD.ptr == bitD.start && bitD.bitsConsumed == CONTAINER_BITS) as u32
}

#[cfg(test)]
mod tests {
    //! Sanity tests for the decoder. We construct a bitstream by hand —
    //! mirroring what `BIT_closeCStream` would produce — and decode.

    use super::*;

    /// Build a byte buffer that decodes (via BIT_reloadDStream) as a
    /// sequence of bit groups where each group is stored LSB-first in
    /// its byte, and the final byte carries a 1-bit end-marker in the
    /// MSB-most position used.
    ///
    /// For this test, we use a simple layout: fill a single byte with
    /// payload bits + an end-marker. Example: bits we want to read are
    /// "101" (3 bits). The stream byte is: payload 101 | endMark 1 | zeros.
    /// Stored as: 0b0000_1101  (bit 3 is endMark, bits 0..=2 are payload).
    fn single_byte_stream(payload: u8, payload_bits: u32) -> [u8; 1] {
        // endMark bit goes right above the payload.
        let mark = 1u8 << payload_bits;
        [payload | mark]
    }

    #[test]
    fn init_and_read_single_byte() {
        // Payload bits (read in reverse of how BIT_* stores): we push
        // bits 0b101 = 5 via BIT_lookBits returning from the top.
        let buf = single_byte_stream(0b101, 3);
        let mut d = BIT_DStream_t::default();
        let rc = BIT_initDStream(&mut d, &buf, buf.len());
        assert_eq!(rc, 1);
        // bitsConsumed should be set so that the end-marker is already
        // consumed.
        let v = BIT_readBits(&mut d, 3);
        assert_eq!(v, 0b101);
    }

    #[test]
    fn init_rejects_zero_last_byte() {
        let buf = [0u8, 0u8];
        let mut d = BIT_DStream_t::default();
        let rc = BIT_initDStream(&mut d, &buf, buf.len());
        assert!(crate::common::error::ERR_isError(rc));
    }

    #[test]
    fn init_rejects_empty() {
        let buf = [];
        let mut d = BIT_DStream_t::default();
        let rc = BIT_initDStream(&mut d, &buf, 0);
        assert!(crate::common::error::ERR_isError(rc));
    }

    #[test]
    fn get_lower_bits_matches_bitmask() {
        assert_eq!(BIT_getLowerBits(0xFFFF, 4), 0x0F);
        assert_eq!(BIT_getLowerBits(0xFFFF, 8), 0xFF);
        assert_eq!(BIT_getLowerBits(0, 16), 0);
    }

    #[test]
    fn get_upper_bits_shifts() {
        assert_eq!(BIT_getUpperBits(0xABCD, 8), 0xAB);
        assert_eq!(BIT_getUpperBits(0x1234_5678, 16), 0x1234);
    }

    #[test]
    fn get_middle_bits_extracts_window() {
        // Extract 4 bits starting at offset 4 from 0xABCD = 0b...AB_CD
        // → middle nibble = 0xC.
        assert_eq!(BIT_getMiddleBits(0xABCD, 4, 4), 0xC);
        // Extract 8 bits starting at offset 8 → 0xAB.
        assert_eq!(BIT_getMiddleBits(0xABCD, 8, 8), 0xAB);
        // start bits beyond CONTAINER_BITS mask correctly.
        let masked = BIT_getMiddleBits(0xABCD, CONTAINER_BITS + 4, 4);
        assert_eq!(masked, 0xC);
    }

    #[test]
    fn skipBits_advances_bitsConsumed_cursor() {
        // Build a minimal DStream and skip 5 bits — bitsConsumed must
        // move from 0 to 5 without touching bitContainer.
        let buf = [0x80u8];
        let mut d = BIT_DStream_t::default();
        let rc = BIT_initDStream(&mut d, &buf, 1);
        assert_eq!(rc, 1);
        let container_before = d.bitContainer;
        let consumed_before = d.bitsConsumed;
        BIT_skipBits(&mut d, 5);
        assert_eq!(d.bitsConsumed, consumed_before + 5);
        assert_eq!(d.bitContainer, container_before);
    }

    // ---- encoder ---------------------------------------------------

    #[test]
    fn init_cstream_rejects_tiny_capacity() {
        let mut buf = [0u8; 4];
        let len = buf.len();
        let (_bitC, rc) = BIT_initCStream(&mut buf, len);
        assert!(crate::common::error::ERR_isError(rc));
    }

    #[test]
    fn cstream_then_dstream_roundtrip() {
        // Encode a sequence of bit groups, close, then decode in
        // reverse (LIFO) order. Exercises init/add/flush/close and
        // init/look/read on the decoder.
        let mut buf = vec![0u8; 64];
        let cap = buf.len();
        let (mut bitC, rc) = BIT_initCStream(&mut buf, cap);
        assert_eq!(rc, 0);

        BIT_addBits(&mut bitC, 5, 3);
        BIT_flushBits(&mut bitC);
        BIT_addBits(&mut bitC, 0xF, 4);
        BIT_flushBits(&mut bitC);
        BIT_addBits(&mut bitC, 0x1A, 5);
        BIT_flushBits(&mut bitC);
        BIT_addBits(&mut bitC, 0x3FF, 10);
        BIT_flushBits(&mut bitC);
        let written = BIT_closeCStream(&mut bitC);
        assert!(written > 0, "closeCStream should report non-zero size");

        // Decode: values come out LIFO (last-pushed first).
        let mut bitD = BIT_DStream_t::default();
        let rc = BIT_initDStream(&mut bitD, &buf[..written], written);
        assert_eq!(rc, written, "init should accept the produced stream");

        assert_eq!(BIT_readBits(&mut bitD, 10), 0x3FF);
        assert_eq!(BIT_readBits(&mut bitD, 5), 0x1A);
        assert_eq!(BIT_readBits(&mut bitD, 4), 0xF);
        assert_eq!(BIT_readBits(&mut bitD, 3), 5);
    }

    #[test]
    fn reloadDStream_overflow_returns_overflow_sentinel() {
        // When bitsConsumed > CONTAINER_BITS (caller over-consumed),
        // `BIT_reloadDStream` must short-circuit to the overflow
        // sentinel — not try to rewind into negative territory.
        let mut buf = [0u8; 16];
        let cap = buf.len();
        let (mut bitC, _) = BIT_initCStream(&mut buf, cap);
        BIT_addBits(&mut bitC, 0xFF, 8);
        BIT_flushBits(&mut bitC);
        let written = BIT_closeCStream(&mut bitC);

        let mut bitD = BIT_DStream_t::default();
        BIT_initDStream(&mut bitD, &buf[..written], written);
        // Force an over-consumption state.
        bitD.bitsConsumed = CONTAINER_BITS + 1;
        assert_eq!(BIT_reloadDStream(&mut bitD), BIT_DStream_overflow);
    }

    #[test]
    fn reloadDStream_at_start_with_full_bits_returns_completed() {
        // When ptr already sits at start AND all container bits have
        // been consumed, reload reports `BIT_DStream_completed` —
        // the decoder should stop.
        let mut buf = [0u8; 16];
        let cap = buf.len();
        let (mut bitC, _) = BIT_initCStream(&mut buf, cap);
        BIT_addBits(&mut bitC, 1, 1);
        BIT_flushBits(&mut bitC);
        let written = BIT_closeCStream(&mut bitC);

        let mut bitD = BIT_DStream_t::default();
        BIT_initDStream(&mut bitD, &buf[..written], written);
        // Hand-force the completed state.
        bitD.ptr = bitD.start;
        bitD.bitsConsumed = CONTAINER_BITS;
        assert_eq!(BIT_reloadDStream(&mut bitD), BIT_DStream_completed);

        // Same position but bitsConsumed < CONTAINER_BITS → endOfBuffer.
        bitD.bitsConsumed = CONTAINER_BITS - 4;
        assert_eq!(BIT_reloadDStream(&mut bitD), BIT_DStream_endOfBuffer);
    }

    #[test]
    fn endOfDStream_reports_true_only_at_start_with_all_bits_consumed() {
        // Contract: returns 1 iff `ptr == start && bitsConsumed ==
        // CONTAINER_BITS`. Used by the FSE/HUF decoders as a
        // stream-termination signal. Regression: drift in the
        // condition would make decoders either stop early (losing
        // data) or loop past EOF (reading garbage).
        let mut buf = vec![0u8; 16];
        let cap = buf.len();
        let (mut bitC, rc) = BIT_initCStream(&mut buf, cap);
        assert_eq!(rc, 0);
        BIT_addBits(&mut bitC, 0xAB, 8);
        BIT_flushBits(&mut bitC);
        let written = BIT_closeCStream(&mut bitC);

        let mut bitD = BIT_DStream_t::default();
        BIT_initDStream(&mut bitD, &buf[..written], written);

        // Fresh stream — not at end.
        assert_eq!(BIT_endOfDStream(&bitD), 0);

        // Read the byte + drive ptr back to start.
        let _ = BIT_readBits(&mut bitD, 8);
        // Still not at end in general — we haven't rewound / consumed
        // the container fully.
        // Hand-set the condition that makes `endOfDStream` fire.
        bitD.ptr = bitD.start;
        bitD.bitsConsumed = CONTAINER_BITS;
        assert_eq!(BIT_endOfDStream(&bitD), 1);

        // Break either invariant → returns 0.
        bitD.bitsConsumed = CONTAINER_BITS - 1;
        assert_eq!(BIT_endOfDStream(&bitD), 0);
    }

    #[test]
    fn readBitsFast_produces_same_values_as_readBits_for_nbBits_ge_1() {
        // `BIT_readBitsFast` requires `nbBits >= 1` (debug_assert).
        // For that domain it MUST produce byte-identical results to
        // `BIT_readBits`. A regression where the Fast path's bit-mask
        // drifts would corrupt every decoded FSE/HUF symbol silently.
        let mut buf = vec![0u8; 64];
        let cap = buf.len();
        let (mut bitC, rc) = BIT_initCStream(&mut buf, cap);
        assert_eq!(rc, 0);
        BIT_addBits(&mut bitC, 0x0A, 4);
        BIT_flushBits(&mut bitC);
        BIT_addBits(&mut bitC, 0x15, 5);
        BIT_flushBits(&mut bitC);
        BIT_addBits(&mut bitC, 0x2B, 6);
        BIT_flushBits(&mut bitC);
        let written = BIT_closeCStream(&mut bitC);
        assert!(written > 0);

        // Decode with the Fast variant.
        let mut bitD = BIT_DStream_t::default();
        let rc = BIT_initDStream(&mut bitD, &buf[..written], written);
        assert_eq!(rc, written);
        assert_eq!(BIT_readBitsFast(&mut bitD, 6), 0x2B);
        assert_eq!(BIT_readBitsFast(&mut bitD, 5), 0x15);
        assert_eq!(BIT_readBitsFast(&mut bitD, 4), 0x0A);
    }

    #[test]
    fn close_cstream_reports_zero_on_overflow() {
        // Tiny capacity + many bytes written → ptr advances past endPtr
        // → close must report 0.
        let mut buf = vec![0u8; CONTAINER_BYTES + 2];
        let cap = buf.len();
        let (mut bitC, rc) = BIT_initCStream(&mut buf, cap);
        assert_eq!(rc, 0);

        for _ in 0..20 {
            BIT_addBits(&mut bitC, 0xFFFF, 16);
            BIT_flushBits(&mut bitC);
        }
        let out = BIT_closeCStream(&mut bitC);
        assert_eq!(out, 0, "overflow must be reported as 0");
    }

    // ---- decoder edge cases ---------------------------------------

    #[test]
    fn large_stream_reload_path() {
        // Build a stream > sizeof(usize) so the "normal case" init
        // branch is exercised. The content below has a non-zero last
        // byte (endMark present).
        let mut buf = [0u8; 32];
        buf[31] = 0b1000_0001; // endMark at bit 7, payload bit at position 0
        let mut d = BIT_DStream_t::default();
        let rc = BIT_initDStream(&mut d, &buf, buf.len());
        assert_eq!(rc, 32);
        // ptr should be at srcSize - sizeof(usize).
        assert_eq!(d.ptr, 32 - CONTAINER_BYTES);
        // bitsConsumed: last byte = 0x81 → highbit = 7 → consumed = 8-7 = 1.
        assert_eq!(d.bitsConsumed, 1);
    }
}
