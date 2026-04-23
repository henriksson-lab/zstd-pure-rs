//! Translation of `lib/compress/zstd_compress_literals.c`.
//!
//! **Fully implemented**: all three literal-block emitters
//! (`ZSTD_noCompressLiterals`, `ZSTD_compressRleLiteralsBlock`,
//! `ZSTD_compressLiterals` — the HUF-compressed path) plus the
//! `ZSTD_minLiteralsToCompress` / `ZSTD_minGain` heuristics.

use crate::common::error::{ErrorCode, ERROR};
use crate::common::mem::{MEM_writeLE16, MEM_writeLE32};
use crate::decompress::zstd_decompress_block::SymbolEncodingType_e;

/// Port of `ZSTD_literalsCompressionIsDisabled`. Answers the question
/// "should this frame emit raw literals blocks?" based on the
/// compression-mode setting plus a strategy/targetLength heuristic
/// for the `auto` case.
///
/// Upstream signature takes `ZSTD_CCtx_params*`; the Rust port takes
/// the three inputs explicitly so callers without a `CCtx_params`
/// struct yet can use it.
pub fn ZSTD_literalsCompressionIsDisabled(
    literalCompressionMode: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
    strategy: u32,
    targetLength: u32,
) -> bool {
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
    match literalCompressionMode {
        ZSTD_ParamSwitch_e::ZSTD_ps_enable => false,
        ZSTD_ParamSwitch_e::ZSTD_ps_disable => true,
        ZSTD_ParamSwitch_e::ZSTD_ps_auto => {
            // upstream: auto → disabled for fast strategy with non-zero
            // targetLength (flag set to accelerate compression).
            strategy == 1 && targetLength > 0
        }
    }
}

/// Upstream `ZSTD_strategy` is an enum `{ZSTD_fast=1, ..,
/// ZSTD_btultra2=9}`. The Rust port alias is a `u32` paired with
/// module-scoped `ZSTD_fast` / `ZSTD_dfast` / ... constants defined
/// in `zstd_compress_sequences` — keeps call sites ergonomic (plain
/// integer comparisons and arithmetic) without losing the named
/// constants.
pub type ZSTD_strategy = u32;

/// Upstream `HUF_repeat`. Mirrored here since this file is the first
/// compressor-side consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HUF_repeat {
    #[default]
    HUF_repeat_none,
    HUF_repeat_check,
    HUF_repeat_valid,
}

/// Port of `showHexa`. Debug-only upstream helper which returns the
/// walked size after dumping bytes. The Rust port keeps only the
/// return-value contract.
#[inline]
pub fn showHexa(src: &[u8]) -> usize {
    src.len()
}

/// Port of `ZSTD_noCompressLiterals`. Emits a `set_basic` literals
/// block — raw bytes prefixed by a 1-/2-/3-byte header. The returned
/// value is the total bytes written.
pub fn ZSTD_noCompressLiterals(dst: &mut [u8], src: &[u8]) -> usize {
    let srcSize = src.len();
    let flSize: usize = 1 + ((srcSize > 31) as usize) + ((srcSize > 4095) as usize);
    if srcSize + flSize > dst.len() {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    match flSize {
        1 => {
            // 2-bit set | 1-bit lhl | 5-bit size
            dst[0] = set_basic_bits() + ((srcSize as u8) << 3);
        }
        2 => {
            // 2 | 2 | 12
            let v = set_basic_bits() as u16 + (1u16 << 2) + ((srcSize as u16) << 4);
            MEM_writeLE16(&mut dst[..2], v);
        }
        _ => {
            // flSize == 3: 2 | 2 | 20
            let v = set_basic_bits() as u32 + (3u32 << 2) + ((srcSize as u32) << 4);
            MEM_writeLE32(&mut dst[..4], v);
        }
    }
    dst[flSize..flSize + srcSize].copy_from_slice(src);
    srcSize + flSize
}

/// Port of `allBytesIdentical` (file-private helper).
fn allBytesIdentical(src: &[u8]) -> bool {
    match src.split_first() {
        None => true,
        Some((first, rest)) => rest.iter().all(|b| b == first),
    }
}

/// Port of `ZSTD_compressRleLiteralsBlock`. Emits a `set_rle` literals
/// block — header + single byte to repeat. Caller guarantees all src
/// bytes are identical (the upstream function asserts this).
pub fn ZSTD_compressRleLiteralsBlock(dst: &mut [u8], src: &[u8]) -> usize {
    let srcSize = src.len();
    debug_assert!(
        !src.is_empty(),
        "ZSTD_compressRleLiteralsBlock requires srcSize >= 1"
    );
    debug_assert!(
        allBytesIdentical(src),
        "ZSTD_compressRleLiteralsBlock requires all bytes identical"
    );
    let flSize: usize = 1 + ((srcSize > 31) as usize) + ((srcSize > 4095) as usize);
    if dst.len() < flSize + 1 {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    match flSize {
        1 => dst[0] = set_rle_bits() + ((srcSize as u8) << 3),
        2 => {
            let v = set_rle_bits() as u16 + (1u16 << 2) + ((srcSize as u16) << 4);
            MEM_writeLE16(&mut dst[..2], v);
        }
        _ => {
            let v = set_rle_bits() as u32 + (3u32 << 2) + ((srcSize as u32) << 4);
            MEM_writeLE32(&mut dst[..4], v);
        }
    }
    dst[flSize] = src[0];
    flSize + 1
}

/// Port of `ZSTD_minLiteralsToCompress`. Returns the minimum input
/// size at which Huffman-compressed literals are worth attempting,
/// tightening as the strategy level rises.
pub fn ZSTD_minLiteralsToCompress(strategy: ZSTD_strategy, huf_repeat: HUF_repeat) -> usize {
    debug_assert!(strategy <= 9);
    let shift = (9i32 - strategy as i32).clamp(0, 3) as u32;
    if huf_repeat == HUF_repeat::HUF_repeat_valid {
        6
    } else {
        8usize << shift
    }
}

/// Minimum literals for a multi-stream (4X) Huffman block.
/// Canonical constant lives on the decoder side; re-exported here
/// for the symmetric `compress1X_vs_compress4X` selector.
pub use crate::decompress::zstd_decompress_block::MIN_LITERALS_FOR_4_STREAMS;

/// Upstream's `LitHufLog`.
pub const LitHufLog: u32 = 11;

/// Port of upstream's `ZSTD_minGain`. Minimum compressed-vs-raw
/// savings (in bytes) needed before Huffman compression is worth
/// keeping. Upstream (zstd_compress_internal.h:677-683):
///   `minlog = (strat >= ZSTD_btultra) ? strat - 1 : 6`
///   returns `(srcSize >> minlog) + 2`
///
/// Previously our port used `shift = 5 + strategy` with no `+ 2`
/// constant — a different formula that skewed the
/// literals-compression gate at every strategy.
#[inline]
pub fn ZSTD_minGain(srcSize: usize, strategy: ZSTD_strategy) -> usize {
    use crate::compress::zstd_compress_sequences::ZSTD_btultra;
    let minlog: u32 = if strategy >= ZSTD_btultra {
        strategy - 1
    } else {
        6
    };
    (srcSize >> minlog) + 2
}

/// Port of `ZSTD_compressLiterals`. Attempts to Huffman-compress
/// `src`; on failure or poor ratio, falls back to `ZSTD_noCompressLiterals`.
///
/// Rust signature note: upstream threads `prevHuf` / `nextHuf` state
/// (`ZSTD_hufCTables_t` plus a `HUF_repeat` flag) so repeated blocks
/// can reuse a previously-built Huffman tree. We accept optional
/// `&mut [HUF_CElt]` + `&mut HUF_repeat` so callers can manage that
/// state; passing `None` for both disables repeat mode.
pub fn ZSTD_compressLiterals(
    dst: &mut [u8],
    src: &[u8],
    disableLiteralCompression: i32,
    strategy: ZSTD_strategy,
    prevHufTable: Option<&mut [crate::compress::huf_compress::HUF_CElt]>,
    prevRepeat: Option<&mut HUF_repeat>,
    suspectUncompressible: i32,
    bmi2: i32,
) -> usize {
    use crate::common::error::{ErrorCode, ERROR};
    use crate::common::mem::{MEM_writeLE24, MEM_writeLE32};
    use crate::compress::huf_compress::{HUF_compress1X_repeat, HUF_compress4X_repeat};

    let srcSize = src.len();
    // Literal block header length: 3, 4, or 5 bytes depending on size.
    let lhSize = 3 + (srcSize >= 1024) as usize + (srcSize >= 16384) as usize;
    let mut singleStream = srcSize < 256;
    let mut hType = SymbolEncodingType_e::set_compressed;

    if disableLiteralCompression != 0 {
        return ZSTD_noCompressLiterals(dst, src);
    }

    // Too small to be worth compressing.
    let repeat_valid = prevRepeat
        .as_ref()
        .map(|r| **r == HUF_repeat::HUF_repeat_valid)
        .unwrap_or(false);
    let min_lits = ZSTD_minLiteralsToCompress(
        strategy,
        if repeat_valid {
            HUF_repeat::HUF_repeat_valid
        } else {
            HUF_repeat::HUF_repeat_none
        },
    );
    if srcSize < min_lits {
        return ZSTD_noCompressLiterals(dst, src);
    }
    if dst.len() < lhSize + 1 {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }

    // Upstream: flip to singleStream when repeat table is valid and
    // the header will fit in 3 bytes — tiny win that reuses cache.
    if repeat_valid && lhSize == 3 {
        singleStream = true;
    }

    // Flags bitmask (subset upstream uses — BMI2 / preferRepeat /
    // optimalDepth / suspectUncompressible).
    let flags: i32 = (if bmi2 != 0 {
        crate::common::entropy_common::HUF_flags_bmi2
    } else {
        0
    }) | (if suspectUncompressible != 0 {
        crate::decompress::huf_decompress::HUF_flags_suspectUncompressible
    } else {
        0
    });

    // Snapshot the caller's initial repeat flag before handing
    // `prevRepeat` to `HUF_compress*_repeat` — the inner function
    // overwrites it with the post-compression state (none / valid).
    // Upstream distinguishes `set_compressed` (fresh tree emitted)
    // from `set_repeat` (prior tree reused) by checking this final
    // state. We need both views: pre-call for gating, post-call for
    // the header's hType.
    let mut repeat_scratch = prevRepeat
        .as_deref()
        .copied()
        .unwrap_or(HUF_repeat::HUF_repeat_none);
    let cLitSize = {
        let (_, payload_dst) = dst.split_at_mut(lhSize);
        if singleStream {
            HUF_compress1X_repeat(
                payload_dst,
                src,
                crate::decompress::huf_decompress::HUF_SYMBOLVALUE_MAX,
                LitHufLog,
                prevHufTable,
                Some(&mut repeat_scratch),
                flags,
            )
        } else if srcSize >= MIN_LITERALS_FOR_4_STREAMS {
            HUF_compress4X_repeat(
                payload_dst,
                src,
                crate::decompress::huf_decompress::HUF_SYMBOLVALUE_MAX,
                LitHufLog,
                prevHufTable,
                Some(&mut repeat_scratch),
                flags,
            )
        } else {
            0
        }
    };
    // Propagate the final repeat state back to the caller's slot.
    if let Some(r) = prevRepeat {
        *r = repeat_scratch;
    }
    if crate::common::error::ERR_isError(cLitSize) {
        return ZSTD_noCompressLiterals(dst, src);
    }

    // If the compressor returned 1, upstream treats it as a single-symbol
    // RLE hint.
    if cLitSize == 1 && (srcSize >= 8 || allBytesIdentical(src)) {
        return ZSTD_compressRleLiteralsBlock(dst, src);
    }

    // Compressed output must yield at least `minGain` savings over raw.
    let minGain = ZSTD_minGain(srcSize, strategy);
    if cLitSize == 0 || cLitSize + minGain >= srcSize {
        return ZSTD_noCompressLiterals(dst, src);
    }

    // Differentiate set_repeat (HUF table was reused) vs set_compressed
    // by consulting the repeat flag the compressor left behind in
    // `repeat_scratch`. Upstream: when `HUF_compress*_repeat` chose to
    // reuse the prior CTable, it leaves the flag at HUF_repeat_valid and
    // skips emitting a tree description — the header then advertises
    // set_repeat so the decoder knows to reuse its prior table.
    let used_repeat = repeat_scratch == HUF_repeat::HUF_repeat_valid;
    if used_repeat {
        hType = SymbolEncodingType_e::set_repeat;
    }

    // Emit the literals-block header.
    let lhc: u32 = match lhSize {
        3 => {
            // 2-bit type | 1-bit singleStream flag | 10-bit srcSize | 10-bit cSize.
            hType as u32
                | ((!singleStream as u32) << 2)
                | ((srcSize as u32) << 4)
                | ((cLitSize as u32) << 14)
        }
        4 => hType as u32 | (2 << 2) | ((srcSize as u32) << 4) | ((cLitSize as u32) << 18),
        _ => {
            // lhSize == 5
            hType as u32 | (3 << 2) | ((srcSize as u32) << 4) | ((cLitSize as u32) << 22)
        }
    };
    match lhSize {
        3 => MEM_writeLE24(&mut dst[..3], lhc),
        4 => MEM_writeLE32(&mut dst[..4], lhc),
        _ => {
            MEM_writeLE32(&mut dst[..4], lhc);
            dst[4] = (cLitSize >> 10) as u8;
        }
    }
    lhSize + cLitSize
}

#[inline]
fn set_basic_bits() -> u8 {
    SymbolEncodingType_e::set_basic as u8
}
#[inline]
fn set_rle_bits() -> u8 {
    SymbolEncodingType_e::set_rle as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn HUF_repeat_discriminants_match_upstream() {
        // Upstream `HUF_repeat` order: none (0), check (1), valid (2).
        // The encoder's state-machine transitions rely on this exact
        // ordering — e.g., `HUF_compressCTable_internal` treats
        // values via `>= HUF_repeat_valid` and `== HUF_repeat_none`
        // checks. Pin the discriminants.
        assert_eq!(HUF_repeat::HUF_repeat_none as u32, 0);
        assert_eq!(HUF_repeat::HUF_repeat_check as u32, 1);
        assert_eq!(HUF_repeat::HUF_repeat_valid as u32, 2);
        // Default is "none" — matches upstream's zero-init convention.
        assert_eq!(HUF_repeat::default(), HUF_repeat::HUF_repeat_none);
    }

    #[test]
    fn minLiteralsToCompress_matches_upstream_formula() {
        // Upstream (zstd_compress_literals.c:115): shift=MIN(9-strat,3);
        // mintc = (huf_repeat==valid) ? 6 : (8 << shift). Regression
        // gate pinning both branches across the full strategy range.
        // `HUF_repeat` is already in scope via `use super::*` at the
        // top of this tests module.
        // No repeat: geometric 8 << min(9-strat, 3).
        for strat in 1u32..=9 {
            let shift = core::cmp::min(9 - strat as i32, 3) as u32;
            let expected: usize = 8usize << shift;
            assert_eq!(
                ZSTD_minLiteralsToCompress(strat, HUF_repeat::HUF_repeat_none),
                expected,
                "no-repeat minLiteralsToCompress({strat})",
            );
        }
        // Repeat valid: always 6.
        for strat in 1u32..=9 {
            assert_eq!(
                ZSTD_minLiteralsToCompress(strat, HUF_repeat::HUF_repeat_valid),
                6,
                "repeat-valid minLiteralsToCompress({strat})",
            );
        }
    }

    #[test]
    fn minGain_matches_upstream_formula_across_strategies() {
        // Pin the exact upstream formula:
        //   minlog = strat >= ZSTD_btultra(8) ? strat-1 : 6
        //   minGain = (srcSize >> minlog) + 2
        // Catches drift in either the shift selection or the
        // constant-2 term — both of which an earlier Rust port had
        // wrong (used `5 + strat` shift and no +2).
        let cases: &[(u32, usize)] = &[
            (1, 1024),  // fast: minlog=6 → (1024>>6)+2 = 16+2 = 18
            (3, 1024),  // greedy: still minlog=6 → 18
            (7, 1024),  // btopt: still minlog=6 → 18
            (8, 1024),  // btultra: minlog=7 → 8+2 = 10
            (9, 1024),  // btultra2: minlog=8 → 4+2 = 6
            (1, 65536), // larger src: minlog=6 → 1024+2 = 1026
            (9, 65536), // minlog=8 → 256+2 = 258
        ];
        for &(strat, sz) in cases {
            let minlog: u32 = if strat >= 8 { strat - 1 } else { 6 };
            let expected = (sz >> minlog) + 2;
            assert_eq!(
                ZSTD_minGain(sz, strat),
                expected,
                "minGain(size={sz}, strat={strat})",
            );
        }
    }

    // End-to-end: encoder's output is accepted by the decoder. Uses the
    // already-ported `ZSTD_decodeLiteralsBlock` so emit-then-decode
    // round-trips the same bytes.
    use crate::decompress::zstd_decompress_block::{
        streaming_operation, ZSTD_DCtx, ZSTD_buildDefaultSeqTables, ZSTD_decodeLiteralsBlock,
    };

    fn roundtrip(src: &[u8]) -> Vec<u8> {
        let mut buf = vec![0u8; src.len() + 16];
        let written = ZSTD_noCompressLiterals(&mut buf, src);
        assert!(!crate::common::error::ERR_isError(written));
        let mut dctx = ZSTD_DCtx::new();
        ZSTD_buildDefaultSeqTables(&mut dctx);
        let mut dst_junk = vec![0u8; src.len().max(16)];
        let consumed = ZSTD_decodeLiteralsBlock(
            &mut dctx,
            &buf[..written],
            &mut dst_junk,
            streaming_operation::not_streaming,
        );
        assert!(!crate::common::error::ERR_isError(consumed));
        assert_eq!(consumed, written);
        assert_eq!(dctx.litSize, src.len());
        dctx.litExtraBuffer[..dctx.litSize].to_vec()
    }

    #[test]
    fn literalsCompressionIsDisabled_respects_explicit_mode() {
        use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
        assert!(!ZSTD_literalsCompressionIsDisabled(
            ZSTD_ParamSwitch_e::ZSTD_ps_enable,
            1,
            32
        ));
        assert!(ZSTD_literalsCompressionIsDisabled(
            ZSTD_ParamSwitch_e::ZSTD_ps_disable,
            5,
            0
        ));
    }

    #[test]
    fn literalsCompressionIsDisabled_auto_fast_plus_targetLength_turns_off() {
        use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
        // auto + fast + targetLength > 0 → disabled.
        assert!(ZSTD_literalsCompressionIsDisabled(
            ZSTD_ParamSwitch_e::ZSTD_ps_auto,
            1,
            32
        ));
        // auto + fast + targetLength == 0 → enabled.
        assert!(!ZSTD_literalsCompressionIsDisabled(
            ZSTD_ParamSwitch_e::ZSTD_ps_auto,
            1,
            0
        ));
        // auto + non-fast → always enabled.
        assert!(!ZSTD_literalsCompressionIsDisabled(
            ZSTD_ParamSwitch_e::ZSTD_ps_auto,
            3,
            32
        ));
    }

    #[test]
    fn no_compress_short_literals_roundtrip() {
        // 1-byte header path: srcSize ≤ 31.
        let src = b"HELLO";
        let decoded = roundtrip(src);
        assert_eq!(decoded, src);
    }

    #[test]
    fn no_compress_medium_literals_roundtrip() {
        // 2-byte header path: 32 ≤ srcSize ≤ 4095.
        let src: Vec<u8> = (0..200u16).map(|i| (i as u8).wrapping_add(b'0')).collect();
        let decoded = roundtrip(&src);
        assert_eq!(decoded, src);
    }

    #[test]
    fn no_compress_large_literals_roundtrip() {
        // 3-byte header path: srcSize > 4095.
        let src: Vec<u8> = (0..8000u32).map(|i| (i as u8).wrapping_mul(31)).collect();
        let decoded = roundtrip(&src);
        assert_eq!(decoded, src);
    }

    #[test]
    fn no_compress_rejects_too_small_dst() {
        let src = [1u8; 10];
        let mut dst = [0u8; 5]; // needs at least 11 bytes
        let rc = ZSTD_noCompressLiterals(&mut dst, &src);
        assert!(crate::common::error::ERR_isError(rc));
    }

    #[test]
    fn rle_literals_roundtrip_short() {
        // Encode 20 × 'A' as RLE, decode, verify.
        let mut buf = [0u8; 4];
        let src = [b'A'; 20];
        let written = ZSTD_compressRleLiteralsBlock(&mut buf, &src);
        assert_eq!(written, 2); // 1-byte header + 1 RLE byte

        let mut dctx = ZSTD_DCtx::new();
        ZSTD_buildDefaultSeqTables(&mut dctx);
        let mut dst_junk = vec![0u8; src.len().max(16)];
        let consumed = ZSTD_decodeLiteralsBlock(
            &mut dctx,
            &buf[..written],
            &mut dst_junk,
            streaming_operation::not_streaming,
        );
        assert_eq!(consumed, written);
        assert_eq!(dctx.litSize, 20);
        assert_eq!(&dctx.litExtraBuffer[..20], &[b'A'; 20]);
    }

    #[test]
    fn all_bytes_identical_helper() {
        assert!(allBytesIdentical(b""));
        assert!(allBytesIdentical(b"A"));
        assert!(allBytesIdentical(b"AAAA"));
        assert!(!allBytesIdentical(b"AAAB"));
        assert!(!allBytesIdentical(b"AB"));
    }

    #[test]
    fn zstd_compress_literals_huf_roundtrip() {
        // Real vertical slice: compress → decode via the already-ported
        // decoder. Input is large enough to pass `ZSTD_minLiteralsToCompress`
        // at strategy=5 (64 bytes) and to yield real savings.
        let src: Vec<u8> = b"the quick brown fox jumps over the lazy dog. "
            .iter()
            .cycle()
            .take(500)
            .copied()
            .collect();

        let mut dst = vec![0u8; src.len() * 2 + 16];
        let written = ZSTD_compressLiterals(&mut dst, &src, 0, 5, None, None, 0, 0);
        assert!(
            !crate::common::error::ERR_isError(written),
            "compress err: {}",
            crate::common::error::ERR_getErrorName(written)
        );
        assert!(
            written > 0 && written < src.len(),
            "no savings: wrote {written} bytes for {}-byte src",
            src.len()
        );

        // Decode.
        let mut dctx = ZSTD_DCtx::new();
        ZSTD_buildDefaultSeqTables(&mut dctx);
        let mut dst_scratch = vec![0u8; src.len()];
        let consumed = ZSTD_decodeLiteralsBlock(
            &mut dctx,
            &dst[..written],
            &mut dst_scratch,
            streaming_operation::not_streaming,
        );
        assert!(
            !crate::common::error::ERR_isError(consumed),
            "decode err: {}",
            crate::common::error::ERR_getErrorName(consumed)
        );
        assert_eq!(consumed, written);
        assert_eq!(dctx.litSize, src.len());
        assert_eq!(&dctx.litExtraBuffer[..dctx.litSize], &src[..]);
    }

    #[test]
    fn zstd_compress_literals_disabled_falls_back_to_raw() {
        let src = b"some content that would compress";
        let mut dst = vec![0u8; src.len() + 16];
        let written = ZSTD_compressLiterals(&mut dst, src, 1, 5, None, None, 0, 0);
        assert!(!crate::common::error::ERR_isError(written));
        // Disabled literal compression → falls through to the raw emitter,
        // which always prefixes with a set_basic header.
        assert_eq!(dst[0] & 3, 0, "first byte should flag set_basic");
    }

    #[test]
    fn min_literals_heuristic_matches_upstream_formula() {
        // strategy 9 → shift=0 → 8 bytes; strategy 5 → shift=3 (cap) → 64.
        assert_eq!(
            ZSTD_minLiteralsToCompress(9, HUF_repeat::HUF_repeat_none),
            8
        );
        assert_eq!(
            ZSTD_minLiteralsToCompress(6, HUF_repeat::HUF_repeat_none),
            64
        );
        assert_eq!(
            ZSTD_minLiteralsToCompress(5, HUF_repeat::HUF_repeat_none),
            64
        );
        // Any strategy, if HUF table is valid (repeat available), drops to 6.
        assert_eq!(
            ZSTD_minLiteralsToCompress(1, HUF_repeat::HUF_repeat_valid),
            6
        );
    }
}
