//! Translation of `lib/compress/zstd_compress_sequences.c`.
//!
//! Ported: `ZSTD_encodeSequences` (per-block FSE sequence bit-stream
//! emitter), `ZSTD_NCountCost` / `ZSTD_fseBitCost` (encoding-cost
//! estimators), `ZSTD_selectEncodingType` (rle / raw / compressed /
//! repeat chooser), and `ZSTD_buildCTable` (symbol-count → FSE
//! encoding table).

#![allow(non_snake_case)]

use crate::common::bitstream::{BIT_addBits, BIT_closeCStream, BIT_flushBits, BIT_initCStream};
use crate::common::error::{ERR_isError, ErrorCode, ERROR};
use crate::common::mem::MEM_32bits;
use crate::compress::fse_compress::{
    ct_header_maxSV, FSE_CState_t, FSE_CTable, FSE_bitCost, FSE_buildCTable_rle,
    FSE_buildCTable_wksp, FSE_encodeSymbol, FSE_flushCState, FSE_initCState, FSE_initCState2,
    FSE_normalizeCount, FSE_optimalTableLog, FSE_writeNCount, FSE_NCOUNTBOUND,
};
use crate::compress::seq_store::SeqDef;
use crate::decompress::zstd_decompress_block::{LLFSELog, LL_bits, MLFSELog, ML_bits, OffFSELog};

/// Port of `FSE_repeat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FSE_repeat {
    /// Cannot use the previous table.
    FSE_repeat_none,
    /// Can use the previous table but it must be checked.
    FSE_repeat_check,
    /// Can use the previous table and it is assumed valid.
    FSE_repeat_valid,
}

/// Port of `ZSTD_dictNCountRepeat` (`zstd_compress.c:5071`). Classifies
/// whether a dictionary-supplied NCount is a `valid` ready-to-reuse
/// table (every symbol up to `maxSymbolValue` has a nonzero count) or
/// merely `check` (needs runtime validation). Returns `FSE_repeat_check`
/// when the dict's alphabet is smaller than the caller needs.
pub fn ZSTD_dictNCountRepeat(
    normalizedCounter: &[i16],
    dictMaxSymbolValue: u32,
    maxSymbolValue: u32,
) -> FSE_repeat {
    if dictMaxSymbolValue < maxSymbolValue {
        return FSE_repeat::FSE_repeat_check;
    }
    for &count in normalizedCounter.iter().take(maxSymbolValue as usize + 1) {
        if count == 0 {
            return FSE_repeat::FSE_repeat_check;
        }
    }
    FSE_repeat::FSE_repeat_valid
}

/// Port of `ZSTD_DefaultPolicy_e`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZSTD_DefaultPolicy_e {
    ZSTD_defaultDisallowed = 0,
    ZSTD_defaultAllowed = 1,
}

/// Port of `ZSTD_strategy` (1..=9). Kept as plain constants for `u32`
/// compatibility with `ZSTD_compressionParameters.strategy`.
pub const ZSTD_fast: u32 = 1;
pub const ZSTD_dfast: u32 = 2;
pub const ZSTD_greedy: u32 = 3;
pub const ZSTD_lazy: u32 = 4;
pub const ZSTD_lazy2: u32 = 5;
pub const ZSTD_btlazy2: u32 = 6;
pub const ZSTD_btopt: u32 = 7;
pub const ZSTD_btultra: u32 = 8;
pub const ZSTD_btultra2: u32 = 9;

use crate::decompress::zstd_decompress_block::SymbolEncodingType_e;

/// Upstream's `kInverseProbabilityLog256`: `ceil(-256 * log2(p/256))`
/// for p in 0..=255. Used by entropy-cost estimators.
pub const kInverseProbabilityLog256: [u32; 256] = [
    0, 2048, 1792, 1642, 1536, 1453, 1386, 1329, 1280, 1236, 1197, 1162, 1130, 1100, 1073, 1047,
    1024, 1001, 980, 960, 941, 923, 906, 889, 874, 859, 844, 830, 817, 804, 791, 779, 768, 756,
    745, 734, 724, 714, 704, 694, 685, 676, 667, 658, 650, 642, 633, 626, 618, 610, 603, 595, 588,
    581, 574, 567, 561, 554, 548, 542, 535, 529, 523, 517, 512, 506, 500, 495, 489, 484, 478, 473,
    468, 463, 458, 453, 448, 443, 438, 434, 429, 424, 420, 415, 411, 407, 402, 398, 394, 390, 386,
    382, 377, 373, 370, 366, 362, 358, 354, 350, 347, 343, 339, 336, 332, 329, 325, 322, 318, 315,
    311, 308, 305, 302, 298, 295, 292, 289, 286, 282, 279, 276, 273, 270, 267, 264, 261, 258, 256,
    253, 250, 247, 244, 241, 239, 236, 233, 230, 228, 225, 222, 220, 217, 215, 212, 209, 207, 204,
    202, 199, 197, 194, 192, 190, 187, 185, 182, 180, 178, 175, 173, 171, 168, 166, 164, 162, 159,
    157, 155, 153, 151, 149, 146, 144, 142, 140, 138, 136, 134, 132, 130, 128, 126, 123, 121, 119,
    117, 115, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96, 94, 93, 91, 89, 87, 85, 83, 82, 80,
    78, 76, 74, 73, 71, 69, 67, 66, 64, 62, 61, 59, 57, 55, 54, 52, 50, 49, 47, 46, 44, 42, 41, 39,
    37, 36, 34, 33, 31, 30, 28, 26, 25, 23, 22, 20, 19, 17, 16, 14, 13, 11, 10, 8, 7, 5, 4, 2, 1,
];

/// Port of `ZSTD_useLowProbCount`. Heuristic: for large sequence
/// counts (≥ 2048) upstream prefers encoding low-prob symbols via
/// ncount=-1; for smaller blocks it uses ncount=1. The threshold
/// mirrors upstream's comment.
#[inline]
pub fn ZSTD_useLowProbCount(nbSeq: usize) -> u32 {
    (nbSeq >= 2048) as u32
}

/// Port of `ZSTD_entropyCost`. Returns a cost estimate in bits for
/// encoding the distribution described by `count` under the entropy
/// bound (Shannon limit, scaled to the inverse-log256 table).
///
/// Upstream asserts `count[s] < total` for every `s`; we keep that
/// invariant as a `debug_assert!`.
pub fn ZSTD_entropyCost(count: &[u32], max: u32, total: usize) -> usize {
    debug_assert!(total > 0);
    let mut cost: u32 = 0;
    for &c_u32 in count.iter().take(max as usize + 1) {
        let c = c_u32 as usize;
        let mut norm = (256 * c) / total;
        if c != 0 && norm == 0 {
            norm = 1;
        }
        debug_assert!(c < total);
        cost = cost.wrapping_add(c_u32.wrapping_mul(kInverseProbabilityLog256[norm]));
    }
    (cost >> 8) as usize
}

/// Port of `ZSTD_crossEntropyCost`. Cost of encoding `count` under a
/// table described by `norm` with the given `accuracyLog`.
pub fn ZSTD_crossEntropyCost(norm: &[i16], accuracyLog: u32, count: &[u32], max: u32) -> usize {
    debug_assert!(accuracyLog <= 8);
    let shift = 8u32 - accuracyLog;
    let mut cost: u32 = 0;
    for (s, &n) in norm.iter().enumerate().take(max as usize + 1) {
        let norm_acc = if n != -1 { n as u32 } else { 1 };
        let norm256 = norm_acc << shift;
        debug_assert!(norm256 > 0 && norm256 < 256);
        cost =
            cost.wrapping_add(count[s].wrapping_mul(kInverseProbabilityLog256[norm256 as usize]));
    }
    (cost >> 8) as usize
}

// ---- Skeletons (need FSE_CTable + FSE_writeNCount etc.) ---------------

/// Port of the file-private `ZSTD_getFSEMaxSymbolValue` — peeks at
/// the high 16 bits of a CTable's header.
#[inline]
pub fn ZSTD_getFSEMaxSymbolValue(ctable: &[FSE_CTable]) -> u32 {
    ct_header_maxSV(ctable)
}

/// Port of `ZSTD_NCountCost`. Returns the **byte** cost of encoding
/// the normalized-count header for the given distribution.
pub fn ZSTD_NCountCost(count: &[u32], max: u32, nbSeq: usize, FSELog: u32) -> usize {
    let mut wksp = [0u8; FSE_NCOUNTBOUND];
    let mut norm = [0i16; (crate::decompress::zstd_decompress_block::MaxSeq + 1) as usize];
    let tableLog = FSE_optimalTableLog(FSELog, nbSeq, max);
    let rc = FSE_normalizeCount(
        &mut norm,
        tableLog,
        count,
        nbSeq,
        max,
        ZSTD_useLowProbCount(nbSeq),
    );
    if ERR_isError(rc) {
        return rc;
    }
    FSE_writeNCount(&mut wksp, &norm, max, tableLog)
}

/// Port of `ZSTD_fseBitCost`. Computes the cost in bits of encoding
/// `count` using the previous CTable. `ctable` is assumed to have
/// already been built via `FSE_buildCTable_wksp`.
pub fn ZSTD_fseBitCost(ctable: &[FSE_CTable], count: &[u32], max: u32) -> usize {
    let kAccuracyLog = 8u32;
    let mut cstate = FSE_CState_t {
        value: 0,
        stateLog: 0,
    };
    FSE_initCState(&mut cstate, ctable);
    if ZSTD_getFSEMaxSymbolValue(ctable) < max {
        return ERROR(ErrorCode::Generic);
    }
    let mut cost: u64 = 0;
    for (s, &c) in count.iter().enumerate().take(max as usize + 1) {
        let tableLog = cstate.stateLog;
        let badCost = (tableLog + 1) << kAccuracyLog;
        let bitCost = FSE_bitCost(ctable, tableLog, s as u32, kAccuracyLog);
        if c == 0 {
            continue;
        }
        if bitCost >= badCost {
            return ERROR(ErrorCode::Generic);
        }
        cost = cost.wrapping_add(c as u64 * bitCost as u64);
    }
    (cost >> kAccuracyLog) as usize
}

/// Port of `ZSTD_selectEncodingType`. Chooses between set_basic,
/// set_rle, set_repeat, and set_compressed for a given ll/ml/of
/// distribution. Updates `*repeatMode` in place. Mirrors upstream's
/// two-branch heuristic:
///   - `strategy < ZSTD_lazy`: simple sample-count + most-frequent
///     thresholds.
///   - otherwise: pick the cheapest of {basic, repeat, compressed}
///     via the explicit cost model.
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_selectEncodingType(
    repeatMode: &mut FSE_repeat,
    count: &[u32],
    max: u32,
    mostFrequent: usize,
    nbSeq: usize,
    FSELog: u32,
    prevCTable: &[FSE_CTable],
    defaultNorm: &[i16],
    defaultNormLog: u32,
    isDefaultAllowed: ZSTD_DefaultPolicy_e,
    strategy: u32,
) -> SymbolEncodingType_e {
    const UNAVAILABLE_COST: usize = usize::MAX / 4;
    let defaultsAllowed = isDefaultAllowed == ZSTD_DefaultPolicy_e::ZSTD_defaultAllowed;

    if mostFrequent == nbSeq {
        *repeatMode = FSE_repeat::FSE_repeat_none;
        if defaultsAllowed && nbSeq <= 2 {
            return SymbolEncodingType_e::set_basic;
        }
        return SymbolEncodingType_e::set_rle;
    }
    if strategy < ZSTD_lazy {
        if defaultsAllowed {
            let staticFse_nbSeq_max = 1000usize;
            let mult = (10u32 - strategy) as usize;
            let baseLog = 3u32;
            let dynamicFse_nbSeq_min = ((1usize << defaultNormLog) * mult) >> baseLog;
            debug_assert!((5..=6).contains(&defaultNormLog));
            debug_assert!((7..=9).contains(&mult));
            if *repeatMode == FSE_repeat::FSE_repeat_valid && nbSeq < staticFse_nbSeq_max {
                return SymbolEncodingType_e::set_repeat;
            }
            if nbSeq < dynamicFse_nbSeq_min || mostFrequent < (nbSeq >> (defaultNormLog - 1)) {
                *repeatMode = FSE_repeat::FSE_repeat_none;
                return SymbolEncodingType_e::set_basic;
            }
        }
    } else {
        let basicCost = if defaultsAllowed {
            ZSTD_crossEntropyCost(defaultNorm, defaultNormLog, count, max)
        } else {
            UNAVAILABLE_COST
        };
        let repeatCost = if *repeatMode != FSE_repeat::FSE_repeat_none {
            ZSTD_fseBitCost(prevCTable, count, max)
        } else {
            UNAVAILABLE_COST
        };
        let NCountCost = ZSTD_NCountCost(count, max, nbSeq, FSELog);
        let compressedCost = (NCountCost << 3) + ZSTD_entropyCost(count, max, nbSeq);

        if basicCost <= repeatCost && basicCost <= compressedCost {
            *repeatMode = FSE_repeat::FSE_repeat_none;
            return SymbolEncodingType_e::set_basic;
        }
        if repeatCost <= compressedCost {
            return SymbolEncodingType_e::set_repeat;
        }
    }
    *repeatMode = FSE_repeat::FSE_repeat_check;
    SymbolEncodingType_e::set_compressed
}

/// Port of `ZSTD_buildCTable`. Materializes `nextCTable` for the
/// chosen `type_` and, for `set_compressed`, writes the
/// `FSE_writeNCount` NCount header into `dst`. Returns the number of
/// bytes written to `dst` (0 for repeat/basic, 1 for RLE, NCountSize
/// for compressed), or an error code.
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_buildCTable(
    dst: &mut [u8],
    nextCTable: &mut [FSE_CTable],
    FSELog: u32,
    type_: SymbolEncodingType_e,
    count: &mut [u32],
    max: u32,
    codeTable: &[u8],
    nbSeq: usize,
    defaultNorm: &[i16],
    defaultNormLog: u32,
    defaultMax: u32,
    prevCTable: &[FSE_CTable],
    _prevCTableSize: usize,
    entropyWorkspace: &mut [u8],
) -> usize {
    match type_ {
        SymbolEncodingType_e::set_rle => {
            let rc = FSE_buildCTable_rle(nextCTable, max as u8);
            if ERR_isError(rc) {
                return rc;
            }
            if dst.is_empty() {
                return ERROR(ErrorCode::DstSizeTooSmall);
            }
            dst[0] = codeTable[0];
            1
        }
        SymbolEncodingType_e::set_repeat => {
            let n = nextCTable.len().min(prevCTable.len());
            nextCTable[..n].copy_from_slice(&prevCTable[..n]);
            0
        }
        SymbolEncodingType_e::set_basic => {
            let rc = FSE_buildCTable_wksp(
                nextCTable,
                defaultNorm,
                defaultMax,
                defaultNormLog,
                entropyWorkspace,
            );
            if ERR_isError(rc) {
                return rc;
            }
            0
        }
        SymbolEncodingType_e::set_compressed => {
            let mut nbSeq_1 = nbSeq;
            let tableLog = FSE_optimalTableLog(FSELog, nbSeq, max);
            if count[codeTable[nbSeq - 1] as usize] > 1 {
                count[codeTable[nbSeq - 1] as usize] -= 1;
                nbSeq_1 -= 1;
            }
            debug_assert!(nbSeq_1 > 1);
            let mut norm = [0i16; (crate::decompress::zstd_decompress_block::MaxSeq + 1) as usize];
            let rc = FSE_normalizeCount(
                &mut norm,
                tableLog,
                count,
                nbSeq_1,
                max,
                ZSTD_useLowProbCount(nbSeq_1),
            );
            if ERR_isError(rc) {
                return rc;
            }
            let NCountSize = FSE_writeNCount(dst, &norm, max, tableLog);
            if ERR_isError(NCountSize) {
                return NCountSize;
            }
            let rc2 = FSE_buildCTable_wksp(nextCTable, &norm, max, tableLog, entropyWorkspace);
            if ERR_isError(rc2) {
                return rc2;
            }
            NCountSize
        }
    }
}

/// Upstream `STREAM_ACCUMULATOR_MIN` — 57 on 64-bit targets, 25 on 32-bit.
/// Drives the long-offset split threshold in `ZSTD_encodeSequences`.
#[inline]
const fn STREAM_ACCUMULATOR_MIN() -> u32 {
    if MEM_32bits() != 0 {
        25
    } else {
        57
    }
}

/// Port of `ZSTD_encodeSequences_body`. Emits the FSE-coded sequence
/// bit-stream:
///   1. Init three FSE states from the **last** sequence's ll/ml/off
///      codes (upstream's `FSE_initCState2`).
///   2. Write that sequence's extra bits (ll extra, ml extra, off
///      extra — split when `longOffsets` and offset > `STREAM_ACCUMULATOR_MIN`).
///   3. Iterate remaining sequences in **reverse** order, emitting
///      (in order) ofCode, mlCode, llCode FSE states then the three
///      extra-bit payloads — with `BIT_flushBits` at the format-
///      prescribed boundaries.
///   4. `FSE_flushCState` each state (ml, off, ll) in order, then
///      `BIT_closeCStream`.
///
/// The returned `usize` is the size of the emitted bitstream, or an
/// error code (via `ERR_isError`).
pub fn ZSTD_encodeSequences_body(
    dst: &mut [u8],
    CTable_MatchLength: &[FSE_CTable],
    mlCodeTable: &[u8],
    CTable_OffsetBits: &[FSE_CTable],
    ofCodeTable: &[u8],
    CTable_LitLength: &[FSE_CTable],
    llCodeTable: &[u8],
    sequences: &[SeqDef],
    nbSeq: usize,
    longOffsets: i32,
) -> usize {
    let dstCapacity = dst.len();
    let (mut blockStream, rc) = BIT_initCStream(dst, dstCapacity);
    if rc != 0 {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    let mut stateMatchLength = FSE_CState_t {
        value: 0,
        stateLog: 0,
    };
    let mut stateOffsetBits = FSE_CState_t {
        value: 0,
        stateLog: 0,
    };
    let mut stateLitLength = FSE_CState_t {
        value: 0,
        stateLog: 0,
    };

    // --- first symbols (the last sequence in the array) ---
    let last = nbSeq - 1;
    FSE_initCState2(
        &mut stateMatchLength,
        CTable_MatchLength,
        mlCodeTable[last] as u32,
    );
    FSE_initCState2(
        &mut stateOffsetBits,
        CTable_OffsetBits,
        ofCodeTable[last] as u32,
    );
    FSE_initCState2(
        &mut stateLitLength,
        CTable_LitLength,
        llCodeTable[last] as u32,
    );
    BIT_addBits(
        &mut blockStream,
        sequences[last].litLength as usize,
        LL_bits[llCodeTable[last] as usize] as u32,
    );
    if MEM_32bits() != 0 {
        BIT_flushBits(&mut blockStream);
    }
    BIT_addBits(
        &mut blockStream,
        sequences[last].mlBase as usize,
        ML_bits[mlCodeTable[last] as usize] as u32,
    );
    if MEM_32bits() != 0 {
        BIT_flushBits(&mut blockStream);
    }
    if longOffsets != 0 {
        let ofBits = ofCodeTable[last] as u32;
        let extraBits = ofBits - ofBits.min(STREAM_ACCUMULATOR_MIN() - 1);
        if extraBits != 0 {
            BIT_addBits(
                &mut blockStream,
                sequences[last].offBase as usize,
                extraBits,
            );
            BIT_flushBits(&mut blockStream);
        }
        BIT_addBits(
            &mut blockStream,
            (sequences[last].offBase >> extraBits) as usize,
            ofBits - extraBits,
        );
    } else {
        BIT_addBits(
            &mut blockStream,
            sequences[last].offBase as usize,
            ofCodeTable[last] as u32,
        );
    }
    BIT_flushBits(&mut blockStream);

    // --- iterate remaining sequences in reverse order ---
    if nbSeq >= 2 {
        for n in (0..=nbSeq - 2).rev() {
            let llCode = llCodeTable[n];
            let ofCode = ofCodeTable[n];
            let mlCode = mlCodeTable[n];
            let llBits = LL_bits[llCode as usize] as u32;
            let ofBits = ofCode as u32;
            let mlBits = ML_bits[mlCode as usize] as u32;
            FSE_encodeSymbol(
                &mut blockStream,
                &mut stateOffsetBits,
                CTable_OffsetBits,
                ofCode as u32,
            );
            FSE_encodeSymbol(
                &mut blockStream,
                &mut stateMatchLength,
                CTable_MatchLength,
                mlCode as u32,
            );
            if MEM_32bits() != 0 {
                BIT_flushBits(&mut blockStream);
            }
            FSE_encodeSymbol(
                &mut blockStream,
                &mut stateLitLength,
                CTable_LitLength,
                llCode as u32,
            );
            if MEM_32bits() != 0
                || ofBits + mlBits + llBits >= 64 - 7 - (LLFSELog + MLFSELog + OffFSELog)
            {
                BIT_flushBits(&mut blockStream);
            }
            BIT_addBits(&mut blockStream, sequences[n].litLength as usize, llBits);
            if MEM_32bits() != 0 && (llBits + mlBits) > 24 {
                BIT_flushBits(&mut blockStream);
            }
            BIT_addBits(&mut blockStream, sequences[n].mlBase as usize, mlBits);
            if MEM_32bits() != 0 || ofBits + mlBits + llBits > 56 {
                BIT_flushBits(&mut blockStream);
            }
            if longOffsets != 0 {
                let extraBits = ofBits - ofBits.min(STREAM_ACCUMULATOR_MIN() - 1);
                if extraBits != 0 {
                    BIT_addBits(&mut blockStream, sequences[n].offBase as usize, extraBits);
                    BIT_flushBits(&mut blockStream);
                }
                BIT_addBits(
                    &mut blockStream,
                    (sequences[n].offBase >> extraBits) as usize,
                    ofBits - extraBits,
                );
            } else {
                BIT_addBits(&mut blockStream, sequences[n].offBase as usize, ofBits);
            }
            BIT_flushBits(&mut blockStream);
        }
    }

    FSE_flushCState(&mut blockStream, &stateMatchLength);
    FSE_flushCState(&mut blockStream, &stateOffsetBits);
    FSE_flushCState(&mut blockStream, &stateLitLength);

    let streamSize = BIT_closeCStream(&mut blockStream);
    if streamSize == 0 {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    streamSize
}

/// Port of `ZSTD_encodeSequences`. BMI2 variant is a no-op dispatch in
/// upstream's scalar build — we collapse to a single body call.
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_encodeSequences_default(
    dst: &mut [u8],
    CTable_MatchLength: &[FSE_CTable],
    mlCodeTable: &[u8],
    CTable_OffsetBits: &[FSE_CTable],
    ofCodeTable: &[u8],
    CTable_LitLength: &[FSE_CTable],
    llCodeTable: &[u8],
    sequences: &[SeqDef],
    nbSeq: usize,
    longOffsets: i32,
) -> usize {
    ZSTD_encodeSequences_body(
        dst,
        CTable_MatchLength,
        mlCodeTable,
        CTable_OffsetBits,
        ofCodeTable,
        CTable_LitLength,
        llCodeTable,
        sequences,
        nbSeq,
        longOffsets,
    )
}

/// Port of `ZSTD_encodeSequences_bmi2` (`zstd_compress_sequences.c:403`).
/// In this pure-Rust build there is no alternate BMI2 body, so this
/// is a one-shot alias of the default scalar path.
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_encodeSequences_bmi2(
    dst: &mut [u8],
    CTable_MatchLength: &[FSE_CTable],
    mlCodeTable: &[u8],
    CTable_OffsetBits: &[FSE_CTable],
    ofCodeTable: &[u8],
    CTable_LitLength: &[FSE_CTable],
    llCodeTable: &[u8],
    sequences: &[SeqDef],
    nbSeq: usize,
    longOffsets: i32,
) -> usize {
    ZSTD_encodeSequences_default(
        dst,
        CTable_MatchLength,
        mlCodeTable,
        CTable_OffsetBits,
        ofCodeTable,
        CTable_LitLength,
        llCodeTable,
        sequences,
        nbSeq,
        longOffsets,
    )
}

/// Port of `ZSTD_encodeSequences`. BMI2 variant is a no-op dispatch in
/// upstream's scalar build — we collapse to a single body call.
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_encodeSequences(
    dst: &mut [u8],
    CTable_MatchLength: &[FSE_CTable],
    mlCodeTable: &[u8],
    CTable_OffsetBits: &[FSE_CTable],
    ofCodeTable: &[u8],
    CTable_LitLength: &[FSE_CTable],
    llCodeTable: &[u8],
    sequences: &[SeqDef],
    nbSeq: usize,
    longOffsets: i32,
    bmi2: i32,
) -> usize {
    if bmi2 != 0 {
        ZSTD_encodeSequences_bmi2(
            dst,
            CTable_MatchLength,
            mlCodeTable,
            CTable_OffsetBits,
            ofCodeTable,
            CTable_LitLength,
            llCodeTable,
            sequences,
            nbSeq,
            longOffsets,
        )
    } else {
        ZSTD_encodeSequences_default(
            dst,
            CTable_MatchLength,
            mlCodeTable,
            CTable_OffsetBits,
            ofCodeTable,
            CTable_LitLength,
            llCodeTable,
            sequences,
            nbSeq,
            longOffsets,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn FSE_repeat_discriminants_match_upstream() {
        // Upstream `FSE_repeat`: none (0), check (1), valid (2).
        // Mirror of `HUF_repeat`. `selectEncodingType` cycles through
        // this enum when deciding set_repeat vs set_compressed, so
        // discriminant drift would silently swap repeat vs compressed.
        assert_eq!(FSE_repeat::FSE_repeat_none as u32, 0);
        assert_eq!(FSE_repeat::FSE_repeat_check as u32, 1);
        assert_eq!(FSE_repeat::FSE_repeat_valid as u32, 2);
    }

    #[test]
    fn ZSTD_strategy_constants_match_upstream_and_monotonic_ordering() {
        // Upstream `typedef enum { ZSTD_fast=1, ..., ZSTD_btultra2=9 }
        // ZSTD_strategy;` — strategies are ABI-exposed as integers
        // through `ZSTD_c_strategy` / `ZSTD_compressionParameters.strategy`.
        // Order is guaranteed (fast→strong) per zstd.h comment.
        assert_eq!(ZSTD_fast, 1);
        assert_eq!(ZSTD_dfast, 2);
        assert_eq!(ZSTD_greedy, 3);
        assert_eq!(ZSTD_lazy, 4);
        assert_eq!(ZSTD_lazy2, 5);
        assert_eq!(ZSTD_btlazy2, 6);
        assert_eq!(ZSTD_btopt, 7);
        assert_eq!(ZSTD_btultra, 8);
        assert_eq!(ZSTD_btultra2, 9);
        // Strict ordering (fast→strong) must hold.
        let order: [u32; 9] = [
            ZSTD_fast,
            ZSTD_dfast,
            ZSTD_greedy,
            ZSTD_lazy,
            ZSTD_lazy2,
            ZSTD_btlazy2,
            ZSTD_btopt,
            ZSTD_btultra,
            ZSTD_btultra2,
        ];
        for pair in order.windows(2) {
            assert!(pair[0] < pair[1]);
        }
    }

    #[test]
    fn ZSTD_DefaultPolicy_e_discriminants_match_upstream() {
        // `ZSTD_DefaultPolicy_e` drives `ZSTD_selectEncodingType` —
        // internal enum, but pinning the 0/1 values prevents an
        // accidental reorder from silently flipping default/allowed.
        assert_eq!(ZSTD_DefaultPolicy_e::ZSTD_defaultDisallowed as u32, 0);
        assert_eq!(ZSTD_DefaultPolicy_e::ZSTD_defaultAllowed as u32, 1);
    }

    #[test]
    fn inverse_prob_log_table_is_monotonic() {
        // The table is monotonically decreasing for indices > 1 (larger
        // p = lower inverse-log cost). Index 0 is a sentinel (0), so
        // start the check at 1.
        for i in 2..256 {
            assert!(
                kInverseProbabilityLog256[i] <= kInverseProbabilityLog256[i - 1],
                "table not monotonic at {i}"
            );
        }
        // Sanity anchors from upstream table.
        assert_eq!(kInverseProbabilityLog256[0], 0);
        assert_eq!(kInverseProbabilityLog256[1], 2048);
        assert_eq!(kInverseProbabilityLog256[255], 1);
    }

    #[test]
    fn use_low_prob_count_threshold() {
        assert_eq!(ZSTD_useLowProbCount(0), 0);
        assert_eq!(ZSTD_useLowProbCount(2047), 0);
        assert_eq!(ZSTD_useLowProbCount(2048), 1);
        assert_eq!(ZSTD_useLowProbCount(100_000), 1);
    }

    #[test]
    fn entropy_cost_uniform_distribution() {
        // Uniform distribution over 2 symbols: each has probability
        // 0.5 → inverse log = ceil(-256 * log2(128/256)) = 256 per
        // occurrence. Total cost (bits >> 8): for 2*128 = 256 counts,
        // cost_sum = 256*128 + 256*128 = 65536; >> 8 = 256.
        let mut count = [0u32; 256];
        count[b'a' as usize] = 128;
        count[b'b' as usize] = 128;
        let cost = ZSTD_entropyCost(&count, 255, 256);
        // Per the table at index 128 (p=128/256=0.5) → 256 (= 1 bit × 256).
        // Each of the 256 occurrences costs ~1 bit → ~256 bits total.
        assert!(
            (250..=260).contains(&cost),
            "uniform-2 entropy cost was {cost}"
        );
    }

    #[test]
    fn entropy_cost_skewed_distribution_lower_when_one_dominant() {
        // A concentrated distribution costs fewer bits per symbol than
        // a flat one of the same size — the hallmark of Shannon entropy.
        let total = 256;
        let mut flat = [0u32; 256];
        for c in flat.iter_mut().take(256) {
            *c = 1;
        }
        let mut skewed = [0u32; 256];
        skewed[0] = 250;
        skewed[1] = 6;
        let flat_cost = ZSTD_entropyCost(&flat, 255, total);
        let skewed_cost = ZSTD_entropyCost(&skewed, 1, total);
        assert!(
            skewed_cost < flat_cost,
            "skewed ({skewed_cost}) should cost less than flat ({flat_cost})"
        );
    }

    #[test]
    fn select_encoding_type_picks_rle_when_one_symbol_dominates_all() {
        // mostFrequent == nbSeq → format prefers RLE over basic when
        // nbSeq > 2 (or when defaults are disallowed).
        let mut rm = FSE_repeat::FSE_repeat_none;
        let count = [10u32, 0, 0, 0];
        let out = ZSTD_selectEncodingType(
            &mut rm,
            &count,
            3,
            10,
            10,
            6,
            &[],
            &[],
            0,
            ZSTD_DefaultPolicy_e::ZSTD_defaultAllowed,
            ZSTD_fast,
        );
        assert_eq!(out, SymbolEncodingType_e::set_rle);
        assert_eq!(rm, FSE_repeat::FSE_repeat_none);
    }

    #[test]
    fn select_encoding_type_picks_basic_when_nbSeq_below_two() {
        // nbSeq <= 2 and defaults allowed → prefer basic.
        let mut rm = FSE_repeat::FSE_repeat_none;
        let count = [2u32, 0, 0, 0];
        let out = ZSTD_selectEncodingType(
            &mut rm,
            &count,
            3,
            2,
            2,
            6,
            &[],
            &[],
            0,
            ZSTD_DefaultPolicy_e::ZSTD_defaultAllowed,
            ZSTD_fast,
        );
        assert_eq!(out, SymbolEncodingType_e::set_basic);
    }

    #[test]
    fn select_encoding_type_high_strategy_uses_cost_sentinels_not_error_codes() {
        use crate::compress::fse_compress::FSE_CTABLE_SIZE_U32;

        let mut rm = FSE_repeat::FSE_repeat_none;
        let count = [3u32, 3, 2, 2];
        let prev = vec![0u32; FSE_CTABLE_SIZE_U32(6, 3)];
        let default_norm = [4i16, 2, 1, 1];
        let out = ZSTD_selectEncodingType(
            &mut rm,
            &count,
            3,
            3,
            10,
            6,
            &prev,
            &default_norm,
            3,
            ZSTD_DefaultPolicy_e::ZSTD_defaultDisallowed,
            ZSTD_btopt,
        );
        assert_eq!(out, SymbolEncodingType_e::set_compressed);
        assert_eq!(rm, FSE_repeat::FSE_repeat_check);
    }

    #[test]
    fn build_ctable_rle_roundtrip() {
        use crate::compress::fse_compress::FSE_CTABLE_SIZE_U32;
        let mut dst = [0u8; 16];
        let mut ct = vec![0u32; FSE_CTABLE_SIZE_U32(0, 255)];
        let code_tbl = [7u8];
        let mut count = [0u32; 256];
        count[7] = 1;
        let mut wksp = vec![0u8; 4096];
        let n = ZSTD_buildCTable(
            &mut dst,
            &mut ct,
            8,
            SymbolEncodingType_e::set_rle,
            &mut count,
            7,
            &code_tbl,
            1,
            &[],
            0,
            0,
            &[],
            0,
            &mut wksp,
        );
        assert!(!crate::common::error::ERR_isError(n));
        assert_eq!(n, 1);
        assert_eq!(dst[0], 7);
    }

    #[test]
    fn encode_sequences_smoke_produces_nonzero_payload() {
        use crate::common::error::ERR_isError;
        use crate::compress::fse_compress::{FSE_buildCTable_wksp, FSE_CTABLE_SIZE_U32};
        use crate::compress::seq_store::SeqDef;
        use crate::decompress::zstd_decompress_block::{
            DefaultMaxOff, LL_defaultNorm, LL_defaultNormLog, ML_defaultNorm, ML_defaultNormLog,
            MaxLL, MaxML, OF_defaultNorm, OF_defaultNormLog,
        };

        // Build FSE CTables from the default norms (zstd's baseline —
        // always valid to encode with).
        let mut ll_ct = vec![0u32; FSE_CTABLE_SIZE_U32(LL_defaultNormLog, MaxLL)];
        let mut ml_ct = vec![0u32; FSE_CTABLE_SIZE_U32(ML_defaultNormLog, MaxML)];
        let mut of_ct = vec![0u32; FSE_CTABLE_SIZE_U32(OF_defaultNormLog, DefaultMaxOff)];
        let mut wksp = vec![0u8; 4096];
        assert!(!ERR_isError(FSE_buildCTable_wksp(
            &mut ll_ct,
            &LL_defaultNorm,
            MaxLL,
            LL_defaultNormLog,
            &mut wksp,
        )));
        assert!(!ERR_isError(FSE_buildCTable_wksp(
            &mut ml_ct,
            &ML_defaultNorm,
            MaxML,
            ML_defaultNormLog,
            &mut wksp,
        )));
        assert!(!ERR_isError(FSE_buildCTable_wksp(
            &mut of_ct,
            &OF_defaultNorm,
            DefaultMaxOff,
            OF_defaultNormLog,
            &mut wksp,
        )));

        // Tiny sequence stream: 3 entries with small in-table ll/ml/of
        // codes (pick indices with LL_defaultNorm != -1).
        let sequences = [
            SeqDef {
                offBase: 0b10,
                litLength: 3,
                mlBase: 4,
            },
            SeqDef {
                offBase: 0b100,
                litLength: 0,
                mlBase: 2,
            },
            SeqDef {
                offBase: 0b1000,
                litLength: 7,
                mlBase: 10,
            },
        ];
        let llCodes = [3u8, 0, 5];
        let mlCodes = [4u8, 2, 6];
        let ofCodes = [2u8, 3, 4]; // ofBase = ofCode (number of bits)
        let mut dst = vec![0u8; 512];
        let n = ZSTD_encodeSequences(
            &mut dst,
            &ml_ct,
            &mlCodes,
            &of_ct,
            &ofCodes,
            &ll_ct,
            &llCodes,
            &sequences,
            sequences.len(),
            0,
            0,
        );
        assert!(!ERR_isError(n), "encodeSequences returned error: {n:#x}");
        assert!(n > 0 && n < dst.len(), "unexpected payload length: {n}");
    }

    #[test]
    fn cross_entropy_cost_matches_entropy_cost_when_norm_matches_count() {
        // Construct a norm that exactly encodes a 2-symbol 50/50 split
        // at accuracyLog=1: norm[0]=1, norm[1]=1. Zero-prob symbols
        // beyond index 1 are encoded as norm=-1.
        let mut norm = [-1i16; 256];
        norm[0] = 1;
        norm[1] = 1;
        let mut count = [0u32; 256];
        count[0] = 128;
        count[1] = 128;
        let cost = ZSTD_crossEntropyCost(&norm, 1, &count, 1);
        // Expected: shift = 7; norm256 = 1<<7 = 128. Each of 256
        // occurrences costs table[128] = 256. Total >> 8 = 256.
        assert!((250..=260).contains(&cost), "cross entropy cost was {cost}");
    }
}
