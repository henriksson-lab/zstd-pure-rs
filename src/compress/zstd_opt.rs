//! Translation of `lib/compress/zstd_opt.c`. Strategies 7–9: `btopt`,
//! `btultra`, `btultra2`.
//!
//! **Implemented**: constants (`ZSTD_LITFREQ_ADD`, `ZSTD_MAX_PRICE`,
//! `ZSTD_PREDEF_THRESHOLD`, `BITCOST_ACCURACY`, `BITCOST_MULTIPLIER`,
//! `ZSTD_OPT_NUM`, `ZSTD_OPT_SIZE`, `MaxLit`,
//! `BASE_LL_FREQS`, `BASE_OF_FREQS`); types
//! (`ZSTD_match_t`, `ZSTD_optimal_t`, `ZSTD_OptPrice_e`,
//! `optState_t`, `ZSTD_optLdm_t`); pure helpers (`ZSTD_bitWeight`,
//! `ZSTD_fracWeight`, `WEIGHT`, `ZSTD_fCost`, `ZSTD_fWeight`,
//! `sum_u32`, `ZSTD_downscaleStats`, `ZSTD_scaleStats`,
//! `ZSTD_readMINMATCH`); pricing (`ZSTD_rawLiteralsCost`,
//! `ZSTD_litLengthPrice`, `ZSTD_getMatchPrice`); stats
//! (`ZSTD_updateStats`, `ZSTD_rescaleFreqs` no-dict variant);
//! 3-byte hash (`ZSTD_insertAndFindFirstIndexHash3`); binary-tree
//! build (`ZSTD_insertBt1`, `ZSTD_updateTree_internal`,
//! `ZSTD_updateTree`) and match-gatherer
//! (`ZSTD_insertBtAndGetAllMatches`,
//! `ZSTD_btGetAllMatches_{noDict,extDict,dictMatchState}`); optLdm helpers
//! (`ZSTD_optLdm_skipRawSeqStoreBytes`, `maybeAddMatch`,
//! `getNextMatchAndUpdateSeqStore`, `processMatchCandidate`).
//!
//! Dict-seeded `rescaleFreqs`, the `ZSTD_C_PREDICT` binary-tree fast
//! path, the shared forward-DP parser, and the `btopt` / `btultra` /
//! `btultra2` entries now cover `noDict`, `extDict`, and
//! `dictMatchState`.

#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]

use crate::common::bits::ZSTD_highbit32;
use crate::common::mem::{MEM_isLittleEndian, MEM_read32};
use crate::compress::fse_compress::{symbolTT_read, FSE_CState_t, FSE_initCState};
use crate::compress::huf_compress::HUF_getNbBitsFromCTable;
use crate::compress::match_state::{ZSTD_MatchState_t, ZSTD_dictMode_e};
use crate::compress::seq_store::{
    ZSTD_newRep, ZSTD_resetSeqStore, ZSTD_storeSeq, MINMATCH, OFFSET_TO_OFFBASE,
};
use crate::compress::zstd_compress::ZSTD_entropyCTables_t;
use crate::compress::zstd_compress::{ZSTD_LLcode, ZSTD_MLcode};
use crate::compress::zstd_compress_literals::HUF_repeat;
use crate::compress::zstd_fast::ZSTD_getLowestMatchIndex;
use crate::compress::zstd_hashes::{ZSTD_count, ZSTD_count_2segments, ZSTD_hash3Ptr, ZSTD_hashPtr};
use crate::compress::zstd_ldm::RawSeqStore_t;
use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
use crate::decompress::zstd_decompress_block::{
    LL_bits, ML_bits, MaxLL, MaxML, MaxOff, ZSTD_BLOCKSIZE_MAX, ZSTD_REP_NUM,
};

/// Upstream `ZSTD_OPT_NUM`. The optimal parser's search-depth limit.
pub const ZSTD_OPT_NUM: usize = 1 << 12;

/// Upstream `ZSTD_OPT_SIZE`. Allocation size for the opt tables.
pub const ZSTD_OPT_SIZE: usize = ZSTD_OPT_NUM + 3;

/// Upstream `MaxLit`. 255 — the max literal byte value.
pub const MaxLit: u32 = 255;

/// Port of `ZSTD_match_t`. A found match candidate: offBase-sumtype
/// code + raw match length.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZSTD_match_t {
    pub off: u32,
    pub len: u32,
}

/// Port of `ZSTD_optimal_t`. One position's Viterbi state: cost so far,
/// the preceding match's offset + length, pending litLength, and the
/// rolling repcode history.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZSTD_optimal_t {
    pub price: i32,
    pub off: u32,
    pub mlen: u32,
    pub litlen: u32,
    pub rep: [u32; ZSTD_REP_NUM],
}

/// Port of `ZSTD_OptPrice_e`. Dynamic stats vs. predefined cost mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ZSTD_OptPrice_e {
    #[default]
    zop_dynamic = 0,
    zop_predef = 1,
}

/// Port of `optState_t`. Upstream's tables live in a cwksp arena; the
/// Rust port owns `Vec<u32>`s directly. `matchTable` and `priceTable`
/// scratch arrays land with the parser proper.
///
#[derive(Debug, Clone)]
pub struct optState_t {
    pub litFreq: Vec<u32>,
    pub litLengthFreq: Vec<u32>,
    pub matchLengthFreq: Vec<u32>,
    pub offCodeFreq: Vec<u32>,
    pub matchTable: Vec<ZSTD_match_t>,
    pub priceTable: Vec<ZSTD_optimal_t>,

    pub litSum: u32,
    pub litLengthSum: u32,
    pub matchLengthSum: u32,
    pub offCodeSum: u32,
    pub litSumBasePrice: u32,
    pub litLengthSumBasePrice: u32,
    pub matchLengthSumBasePrice: u32,
    pub offCodeSumBasePrice: u32,
    pub priceType: ZSTD_OptPrice_e,
    pub literalCompressionMode: ZSTD_ParamSwitch_e,
}

impl Default for optState_t {
    fn default() -> Self {
        Self {
            litFreq: vec![0u32; (MaxLit + 1) as usize],
            litLengthFreq: vec![0u32; (MaxLL + 1) as usize],
            matchLengthFreq: vec![0u32; (MaxML + 1) as usize],
            offCodeFreq: vec![0u32; (MaxOff + 1) as usize],
            matchTable: vec![ZSTD_match_t::default(); ZSTD_OPT_SIZE],
            priceTable: vec![ZSTD_optimal_t::default(); ZSTD_OPT_SIZE],
            litSum: 0,
            litLengthSum: 0,
            matchLengthSum: 0,
            offCodeSum: 0,
            litSumBasePrice: 0,
            litLengthSumBasePrice: 0,
            matchLengthSumBasePrice: 0,
            offCodeSumBasePrice: 0,
            priceType: ZSTD_OptPrice_e::default(),
            literalCompressionMode: ZSTD_ParamSwitch_e::default(),
        }
    }
}

/// Port of `WEIGHT`. The active build uses fractional weight at
/// `optLevel >= 2`, falling back to whole-bit weight below that.
#[inline]
pub fn WEIGHT(stat: u32, optLevel: i32) -> u32 {
    if optLevel != 0 {
        ZSTD_fracWeight(stat)
    } else {
        ZSTD_bitWeight(stat)
    }
}

/// Port of `ZSTD_compressedLiterals`. True when literal blocks are
/// entropy-encoded rather than emitted raw.
#[inline]
pub fn ZSTD_compressedLiterals(optPtr: &optState_t) -> bool {
    optPtr.literalCompressionMode != ZSTD_ParamSwitch_e::ZSTD_ps_disable
}

/// Port of `ZSTD_setBasePrices`. Precompute the per-symbol "base price"
/// used as the floor against which weighted stats are subtracted when
/// pricing a sequence.
pub fn ZSTD_setBasePrices(optPtr: &mut optState_t, optLevel: i32) {
    if ZSTD_compressedLiterals(optPtr) {
        optPtr.litSumBasePrice = WEIGHT(optPtr.litSum, optLevel);
    }
    optPtr.litLengthSumBasePrice = WEIGHT(optPtr.litLengthSum, optLevel);
    optPtr.matchLengthSumBasePrice = WEIGHT(optPtr.matchLengthSum, optLevel);
    optPtr.offCodeSumBasePrice = WEIGHT(optPtr.offCodeSum, optLevel);
}

/// Upstream `ZSTD_LITFREQ_ADD`. Scaling constant so literal-frequency
/// stats adapt faster toward new distributions.
pub const ZSTD_LITFREQ_ADD: u32 = 2;

/// Upstream `ZSTD_MAX_PRICE`. Sentinel "infinite" price value.
pub const ZSTD_MAX_PRICE: i32 = 1 << 30;

/// Upstream `ZSTD_PREDEF_THRESHOLD`. If the source is smaller than
/// this, opt falls back to predefined (static) symbol costs.
pub const ZSTD_PREDEF_THRESHOLD: u32 = 8;

/// Upstream `BITCOST_ACCURACY`. The active configuration uses 8 bits
/// of fractional accuracy for price arithmetic.
pub const BITCOST_ACCURACY: u32 = 8;

/// Upstream `BITCOST_MULTIPLIER`. Precomputed `1 << BITCOST_ACCURACY`.
pub const BITCOST_MULTIPLIER: u32 = 1 << BITCOST_ACCURACY;

/// Port of `ZSTD_bitWeight`. Returns the whole-bit cost estimate of an
/// occurrence count — `ceil(log2(stat+1)) * BITCOST_MULTIPLIER`.
#[inline]
pub fn ZSTD_bitWeight(stat: u32) -> u32 {
    ZSTD_highbit32(stat + 1) * BITCOST_MULTIPLIER
}

/// Port of `ZSTD_fracWeight`. Fractional-bit cost approximation: uses
/// the high bit for the integer part and linear interpolation within
/// the next power-of-two for the fractional part.
#[inline]
pub fn ZSTD_fracWeight(rawStat: u32) -> u32 {
    let stat = rawStat + 1;
    let hb = ZSTD_highbit32(stat);
    let BWeight = hb * BITCOST_MULTIPLIER;
    let FWeight = (stat << BITCOST_ACCURACY) >> hb;
    debug_assert!(hb + BITCOST_ACCURACY < 31);
    BWeight + FWeight
}

/// Port of `sum_u32`. Straight sum over a slice.
#[inline]
pub fn sum_u32(table: &[u32]) -> u32 {
    table.iter().sum()
}

/// Port of the `base_directive_e` enum used by `ZSTD_downscaleStats`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum base_directive_e {
    /// Upstream `base_0possible` — floor value is 0 for zero-count
    /// slots, 1 otherwise.
    base_0possible = 0,
    /// Upstream `base_1guaranteed` — every slot floors at 1.
    base_1guaranteed = 1,
}

/// Port of `ZSTD_downscaleStats`. Divides every entry in `table` by
/// `1 << shift`, then adds a small floor (0 or 1 depending on `base1`
/// and whether the original was non-zero). Returns the new sum.
pub fn ZSTD_downscaleStats(
    table: &mut [u32],
    lastEltIndex: u32,
    shift: u32,
    base1: base_directive_e,
) -> u32 {
    debug_assert!(shift < 30);
    let mut sum: u32 = 0;
    let end = (lastEltIndex as usize) + 1;
    for slot in table.iter_mut().take(end) {
        let base = if base1 == base_directive_e::base_1guaranteed {
            1
        } else {
            u32::from(*slot > 0)
        };
        let newStat = base + (*slot >> shift);
        sum += newStat;
        *slot = newStat;
    }
    sum
}

/// Port of `ZSTD_scaleStats`. If the table's sum exceeds `2^logTarget`,
/// downscale every entry until the target is met. Returns the new
/// (post-scale) sum.
pub fn ZSTD_scaleStats(table: &mut [u32], lastEltIndex: u32, logTarget: u32) -> u32 {
    debug_assert!(logTarget < 30);
    let prevsum = sum_u32(&table[..(lastEltIndex as usize) + 1]);
    let factor = prevsum >> logTarget;
    if factor <= 1 {
        return prevsum;
    }
    ZSTD_downscaleStats(
        table,
        lastEltIndex,
        ZSTD_highbit32(factor),
        base_directive_e::base_1guaranteed,
    )
}

/// Port of `ZSTD_readMINMATCH`. Reads 3 or 4 bytes from `bytes` into a
/// u32 that's safe to compare byte-for-byte. For length 3, the unused
/// byte is masked out (shifted out on LE, shifted away on BE).
///
/// Upstream precondition: the buffer must be at least 4 bytes — the
/// Rust caller supplies a slice that starts at the read position.
#[inline]
pub fn ZSTD_readMINMATCH(bytes: &[u8], length: u32) -> u32 {
    match length {
        3 => {
            if MEM_isLittleEndian() != 0 {
                MEM_read32(bytes) << 8
            } else {
                MEM_read32(bytes) >> 8
            }
        }
        _ => MEM_read32(bytes), // includes default = 4
    }
}

/// Port of `ZSTD_rawLiteralsCost`. Cost of a literal run excluding
/// the litLength symbol itself.
///
/// - Raw literals: 8 bits per byte, no discount.
/// - Predef prices: flat 6 bits per byte.
/// - Dynamic: base price × litLength minus the weighted count of each
///   byte in the literal frequency table.
pub fn ZSTD_rawLiteralsCost(
    literals: &[u8],
    litLength: u32,
    optPtr: &optState_t,
    optLevel: i32,
) -> u32 {
    if litLength == 0 {
        return 0;
    }
    if !ZSTD_compressedLiterals(optPtr) {
        return (litLength << 3) * BITCOST_MULTIPLIER;
    }
    if optPtr.priceType == ZSTD_OptPrice_e::zop_predef {
        return litLength * 6 * BITCOST_MULTIPLIER;
    }
    // Dynamic — subtract the weighted frequency of each actual byte
    // from the per-literal base price.
    let mut price = optPtr.litSumBasePrice * litLength;
    let litPriceMax = optPtr.litSumBasePrice.saturating_sub(BITCOST_MULTIPLIER);
    debug_assert!(optPtr.litSumBasePrice >= BITCOST_MULTIPLIER);
    for &b in literals.iter().take(litLength as usize) {
        let mut litPrice = WEIGHT(optPtr.litFreq[b as usize], optLevel);
        if litPrice > litPriceMax {
            litPrice = litPriceMax;
        }
        price = price.saturating_sub(litPrice);
    }
    price
}

/// Port of `ZSTD_litLengthPrice`. Cost of the litLength symbol itself
/// (extra bits + table cost). The `litLength == ZSTD_BLOCKSIZE_MAX`
/// special case is upstream's trick for expressing an all-literals
/// block via a synthetic price.
pub fn ZSTD_litLengthPrice(litLength: u32, optPtr: &optState_t, optLevel: i32) -> u32 {
    debug_assert!(litLength as usize <= ZSTD_BLOCKSIZE_MAX);
    if optPtr.priceType == ZSTD_OptPrice_e::zop_predef {
        return WEIGHT(litLength, optLevel);
    }
    if litLength as usize == ZSTD_BLOCKSIZE_MAX {
        return BITCOST_MULTIPLIER
            + ZSTD_litLengthPrice(ZSTD_BLOCKSIZE_MAX as u32 - 1, optPtr, optLevel);
    }
    let llCode = ZSTD_LLcode(litLength) as usize;
    (LL_bits[llCode] as u32 * BITCOST_MULTIPLIER) + optPtr.litLengthSumBasePrice
        - WEIGHT(optPtr.litLengthFreq[llCode], optLevel)
}

/// Port of `ZSTD_getMatchPrice`. Cost of the match component of a
/// sequence (offset symbol + matchLength symbol + constants), to be
/// combined with literal cost for the full sequence price.
///
/// At `optLevel < 2` long offsets incur a handicap — favors decode
/// cache efficiency at small encode-ratio cost.
pub fn ZSTD_getMatchPrice(
    offBase: u32,
    matchLength: u32,
    optPtr: &optState_t,
    optLevel: i32,
) -> u32 {
    let offCode = ZSTD_highbit32(offBase);
    let mlBase = matchLength - MINMATCH;
    debug_assert!(matchLength >= MINMATCH);

    if optPtr.priceType == ZSTD_OptPrice_e::zop_predef {
        return WEIGHT(mlBase, optLevel) + (16 + offCode) * BITCOST_MULTIPLIER;
    }

    let mut price = (offCode * BITCOST_MULTIPLIER)
        + (optPtr.offCodeSumBasePrice - WEIGHT(optPtr.offCodeFreq[offCode as usize], optLevel));

    if optLevel < 2 && offCode >= 20 {
        price += (offCode - 19) * 2 * BITCOST_MULTIPLIER;
    }

    let mlCode = ZSTD_MLcode(mlBase) as usize;
    price += (ML_bits[mlCode] as u32 * BITCOST_MULTIPLIER) + optPtr.matchLengthSumBasePrice
        - WEIGHT(optPtr.matchLengthFreq[mlCode], optLevel);

    price + BITCOST_MULTIPLIER / 5
}

/// Port of `ZSTD_updateStats`. Increments every frequency counter
/// that a newly-accepted sequence touches. `litLength + literals.len()
/// must be ≥ litLength` — callers must pass enough literal bytes.
pub fn ZSTD_updateStats(
    optPtr: &mut optState_t,
    litLength: u32,
    literals: &[u8],
    offBase: u32,
    matchLength: u32,
) {
    if ZSTD_compressedLiterals(optPtr) {
        for &b in literals.iter().take(litLength as usize) {
            optPtr.litFreq[b as usize] += ZSTD_LITFREQ_ADD;
        }
        optPtr.litSum += litLength * ZSTD_LITFREQ_ADD;
    }
    let llCode = ZSTD_LLcode(litLength) as usize;
    optPtr.litLengthFreq[llCode] += 1;
    optPtr.litLengthSum += 1;

    let offCode = ZSTD_highbit32(offBase) as usize;
    debug_assert!(offCode <= MaxOff as usize);
    optPtr.offCodeFreq[offCode] += 1;
    optPtr.offCodeSum += 1;

    let mlBase = matchLength - MINMATCH;
    let mlCode = ZSTD_MLcode(mlBase) as usize;
    optPtr.matchLengthFreq[mlCode] += 1;
    optPtr.matchLengthSum += 1;
}

/// Port of `ZSTD_insertAndFindFirstIndexHash3`. Walks `hashTable3`
/// forward from `nextToUpdate3` to the absolute position at `ip_abs`,
/// inserting each position's 3-byte hash. Returns the previously
/// stored index at `ip`'s hash slot (0 if none).
///
/// `window_buf` spans the window; `ip_abs` is the absolute base-
/// relative index of the current position. `nextToUpdate3` is both
/// input and output (advanced to `ip_abs`).
///
/// Upstream asserts the table always lives in the contiguous prefix
/// (never extDict); the Rust port inherits that invariant.
pub fn ZSTD_insertAndFindFirstIndexHash3(
    ms: &mut ZSTD_MatchState_t,
    nextToUpdate3: &mut u32,
    window_buf: &[u8],
    ip_abs: u32,
) -> u32 {
    let hashLog3 = ms.hashLog3;
    let base_off = ms.window.base_offset;
    debug_assert!(hashLog3 > 0);
    debug_assert!(ip_abs >= base_off);
    // Hash of the current position — read from the input slice.
    let ip_pos = ip_abs.wrapping_sub(base_off) as usize;
    let hash3 = ZSTD_hash3Ptr(&window_buf[ip_pos..], hashLog3);

    let mut idx = *nextToUpdate3;
    while idx < ip_abs {
        let idx_pos = idx.wrapping_sub(base_off) as usize;
        let h = ZSTD_hash3Ptr(&window_buf[idx_pos..], hashLog3);
        ms.hashTable3[h] = idx;
        idx = idx.wrapping_add(1);
    }
    *nextToUpdate3 = ip_abs;
    ms.hashTable3[hash3]
}

/// Port of `ZSTD_optLdm_t`. Info the optimal parser needs to decide
/// whether to adopt an LDM-discovered match at the current block
/// position.
#[derive(Debug, Default)]
pub struct ZSTD_optLdm_t {
    pub seqStore: RawSeqStore_t,
    pub startPosInBlock: u32,
    pub endPosInBlock: u32,
    pub offset: u32,
}

#[inline]
fn ZSTD_cloneActiveRawSeqStore(src: Option<&RawSeqStore_t>) -> RawSeqStore_t {
    let Some(src) = src else {
        return RawSeqStore_t::default();
    };
    let end = src.size.min(src.seq.len());
    RawSeqStore_t {
        seq: src.seq[..end].to_vec(),
        pos: src.pos.min(end),
        posInSequence: src.posInSequence,
        size: end,
        capacity: end,
    }
}

/// Port of `ZSTD_optLdm_skipRawSeqStoreBytes`. Advances both `pos`
/// and `posInSequence` forward by `nbBytes`, popping full sequences
/// as they're consumed. Symmetric to LDM's `skipSequences` but treats
/// `litLength + matchLength` as a single block rather than partial
/// match/lit residuals.
pub fn ZSTD_optLdm_skipRawSeqStoreBytes(rawSeqStore: &mut RawSeqStore_t, nbBytes: usize) {
    let mut currPos = (rawSeqStore.posInSequence as u32).wrapping_add(nbBytes as u32);
    while currPos > 0 && rawSeqStore.pos < rawSeqStore.size {
        let currSeq = rawSeqStore.seq[rawSeqStore.pos];
        let fullLen = currSeq.litLength.wrapping_add(currSeq.matchLength);
        if currPos >= fullLen {
            currPos = currPos.wrapping_sub(fullLen);
            rawSeqStore.pos += 1;
        } else {
            rawSeqStore.posInSequence = currPos as usize;
            return;
        }
    }
    if currPos == 0 || rawSeqStore.pos == rawSeqStore.size {
        rawSeqStore.posInSequence = 0;
    }
}

/// Port of `ZSTD_optLdm_maybeAddMatch`. If the current block position
/// falls within the cached LDM candidate and the remaining match is
/// long enough, append it to `matches` (preserving ascending-length
/// ordering). Bumps `*nbMatches` on success.
pub fn ZSTD_optLdm_maybeAddMatch(
    matches: &mut [ZSTD_match_t],
    nbMatches: &mut u32,
    optLdm: &ZSTD_optLdm_t,
    currPosInBlock: u32,
    minMatch: u32,
) {
    // Reject out-of-range positions before doing any arithmetic —
    // upstream wraps around in unsigned math and relies on a later
    // check, but Rust's debug overflow checks make that a hard panic.
    if currPosInBlock < optLdm.startPosInBlock || currPosInBlock >= optLdm.endPosInBlock {
        return;
    }
    let posDiff = currPosInBlock - optLdm.startPosInBlock;
    let candidateMatchLength = optLdm.endPosInBlock - optLdm.startPosInBlock - posDiff;

    if candidateMatchLength < minMatch {
        return;
    }

    let n = *nbMatches as usize;
    if *nbMatches == 0
        || (n > 0
            && candidateMatchLength > matches[n - 1].len
            && (*nbMatches as usize) < ZSTD_OPT_NUM)
    {
        matches[n] = ZSTD_match_t {
            off: OFFSET_TO_OFFBASE(optLdm.offset),
            len: candidateMatchLength,
        };
        *nbMatches += 1;
    }
}

/// Port of `ZSTD_opt_getNextMatchAndUpdateSeqStore`. Populates
/// `optLdm.{startPosInBlock,endPosInBlock,offset}` for the next LDM
/// candidate and advances the seqStore cursor past consumed bytes.
///
/// Sets start/end to `u32::MAX` (upstream's `UINT_MAX`) when the
/// cursor is exhausted or when the remaining literal run overshoots
/// the block — callers treat this as "no LDM candidate this block".
pub fn ZSTD_opt_getNextMatchAndUpdateSeqStore(
    optLdm: &mut ZSTD_optLdm_t,
    currPosInBlock: u32,
    blockBytesRemaining: u32,
) {
    if optLdm.seqStore.size == 0 || optLdm.seqStore.pos >= optLdm.seqStore.size {
        optLdm.startPosInBlock = u32::MAX;
        optLdm.endPosInBlock = u32::MAX;
        return;
    }
    let currSeq = optLdm.seqStore.seq[optLdm.seqStore.pos];
    debug_assert!(
        optLdm.seqStore.posInSequence
            <= currSeq.litLength.wrapping_add(currSeq.matchLength) as usize
    );
    let currBlockEndPos = currPosInBlock.wrapping_add(blockBytesRemaining);
    let literalsBytesRemaining = currSeq
        .litLength
        .saturating_sub(optLdm.seqStore.posInSequence as u32);
    let matchBytesRemaining = if literalsBytesRemaining == 0 {
        currSeq
            .matchLength
            .wrapping_sub((optLdm.seqStore.posInSequence as u32).wrapping_sub(currSeq.litLength))
    } else {
        currSeq.matchLength
    };

    if literalsBytesRemaining >= blockBytesRemaining {
        optLdm.startPosInBlock = u32::MAX;
        optLdm.endPosInBlock = u32::MAX;
        ZSTD_optLdm_skipRawSeqStoreBytes(&mut optLdm.seqStore, blockBytesRemaining as usize);
        return;
    }

    optLdm.startPosInBlock = currPosInBlock.wrapping_add(literalsBytesRemaining);
    optLdm.endPosInBlock = optLdm.startPosInBlock.wrapping_add(matchBytesRemaining);
    optLdm.offset = currSeq.offset;

    if optLdm.endPosInBlock > currBlockEndPos {
        optLdm.endPosInBlock = currBlockEndPos;
        ZSTD_optLdm_skipRawSeqStoreBytes(
            &mut optLdm.seqStore,
            (currBlockEndPos - currPosInBlock) as usize,
        );
    } else {
        ZSTD_optLdm_skipRawSeqStoreBytes(
            &mut optLdm.seqStore,
            literalsBytesRemaining.wrapping_add(matchBytesRemaining) as usize,
        );
    }
}

/// Port of `ZSTD_optLdm_processMatchCandidate`. Thin wrapper that
/// reloads the next LDM candidate when the current position has
/// passed the cached endPos, then offers it to the match list.
pub fn ZSTD_optLdm_processMatchCandidate(
    optLdm: &mut ZSTD_optLdm_t,
    matches: &mut [ZSTD_match_t],
    nbMatches: &mut u32,
    currPosInBlock: u32,
    remainingBytes: u32,
    minMatch: u32,
) {
    if optLdm.seqStore.size == 0 || optLdm.seqStore.pos >= optLdm.seqStore.size {
        return;
    }
    if currPosInBlock >= optLdm.endPosInBlock {
        if currPosInBlock > optLdm.endPosInBlock {
            let posOvershoot = currPosInBlock - optLdm.endPosInBlock;
            ZSTD_optLdm_skipRawSeqStoreBytes(&mut optLdm.seqStore, posOvershoot as usize);
        }
        ZSTD_opt_getNextMatchAndUpdateSeqStore(optLdm, currPosInBlock, remainingBytes);
    }
    ZSTD_optLdm_maybeAddMatch(matches, nbMatches, optLdm, currPosInBlock, minMatch);
}

/// Port of `listStats`. Upstream keeps it in a disabled debug block;
/// the Rust port preserves the function surface and the element-count
/// behavior without emitting logs.
#[inline]
pub fn listStats(table: &[u32], lastEltID: i32) -> usize {
    let nbElts = (lastEltID + 1).max(0) as usize;
    table.len().min(nbElts)
}

/// Port of `ZSTD_insertBt1`.
///
/// Inserts position `curr` (= `ip_abs`) into the match-state's binary
/// search tree, re-sorting the tree by common-prefix length along the
/// walk. Returns the number of positions covered (upstream speedhack:
/// when we find a very long match, we can skip inserting the next
/// few positions since their substring is already reachable via the
/// long match).
///
/// Supports both the regular prefix tree and the ext-dict two-segment
/// match-counting path, including upstream's `ZSTD_C_PREDICT`
/// left/right shortcut.
pub fn ZSTD_insertBt1(
    ms: &mut ZSTD_MatchState_t,
    buf: &[u8],
    ip_abs: u32,
    iend_pos: usize,
    target: u32,
    mls: u32,
    extDict: bool,
) -> u32 {
    let hashLog = ms.cParams.hashLog;
    let base_off = ms.window.base_offset;
    debug_assert!(ip_abs >= base_off);
    let ip_pos = ip_abs.wrapping_sub(base_off) as usize;
    let h = ZSTD_hashPtr(&buf[ip_pos..], hashLog, mls);
    let btLog = ms.cParams.chainLog - 1;
    let btMask: u32 = (1u32 << btLog) - 1;
    let curr = ip_abs;
    let btLow: u32 = curr.saturating_sub(btMask);

    // Two chain-table slots at position `curr`, one for "smaller
    // subtree root", one for "larger subtree root". We track them by
    // index; `None` means "we hit btLow and stopped writing".
    let mut smaller_slot: Option<usize> = Some((2 * (curr & btMask)) as usize);
    let mut larger_slot: Option<usize> = Some((2 * (curr & btMask)) as usize + 1);

    let mut commonLengthSmaller = 0usize;
    let mut commonLengthLarger = 0usize;
    let windowLow = ZSTD_getLowestMatchIndex(ms, target, ms.cParams.windowLog);
    let mut matchEndIdx: u32 = curr.wrapping_add(8).wrapping_add(1);
    let mut bestLength: usize = 8;
    let mut nbCompares = 1u32 << ms.cParams.searchLog;
    let dictLimit = ms.window.dictLimit;
    let prefixStart = dictLimit.saturating_sub(ms.window.base_offset) as usize;
    let dict = &ms.dictContent;
    let dictEnd = dict.len();
    let dictBaseOffset = ms.window.dictBase_offset;

    // Update hash table to point at `curr`.
    let mut matchIndex = ms.hashTable[h];
    ms.hashTable[h] = curr;

    let mut predictedSmall = 0u32;
    let mut predictedLarge = 0u32;
    if curr > 0 {
        let prev_base = (2 * ((curr - 1) & btMask)) as usize;
        predictedSmall = ms.chainTable[prev_base];
        predictedLarge = ms.chainTable[prev_base + 1];
        predictedSmall += u32::from(predictedSmall > 0);
        predictedLarge += u32::from(predictedLarge > 0);
    }

    debug_assert!(windowLow > 0);
    while nbCompares > 0 && matchIndex >= windowLow {
        let next_base = (2 * (matchIndex & btMask)) as usize;
        let mut matchLength = commonLengthSmaller.min(commonLengthLarger);
        debug_assert!(matchIndex < curr);

        if curr > 0 {
            let predict_base = (2 * ((matchIndex - 1) & btMask)) as usize;
            if matchIndex == predictedSmall {
                if let Some(s) = smaller_slot {
                    ms.chainTable[s] = matchIndex;
                }
                if matchIndex <= btLow {
                    smaller_slot = None;
                    break;
                }
                smaller_slot = Some(next_base + 1);
                matchIndex = ms.chainTable[next_base + 1];
                predictedSmall = ms.chainTable[predict_base + 1];
                predictedSmall += u32::from(predictedSmall > 0);
                nbCompares -= 1;
                continue;
            }
            if matchIndex == predictedLarge {
                if let Some(l) = larger_slot {
                    ms.chainTable[l] = matchIndex;
                }
                if matchIndex <= btLow {
                    larger_slot = None;
                    break;
                }
                larger_slot = Some(next_base);
                matchIndex = ms.chainTable[next_base];
                predictedLarge = ms.chainTable[predict_base];
                predictedLarge += u32::from(predictedLarge > 0);
                nbCompares -= 1;
                continue;
            }
        }

        let mut match_pos = matchIndex.saturating_sub(ms.window.base_offset) as usize;
        if !extDict || matchIndex.wrapping_add(matchLength as u32) >= dictLimit {
            matchLength += ZSTD_count(buf, ip_pos + matchLength, match_pos + matchLength, iend_pos);
        } else {
            match_pos = matchIndex.saturating_sub(dictBaseOffset) as usize;
            matchLength += ZSTD_count_2segments(
                buf,
                ip_pos + matchLength,
                iend_pos,
                prefixStart,
                dict,
                match_pos + matchLength,
                dictEnd,
            );
            if matchIndex.wrapping_add(matchLength as u32) >= dictLimit {
                match_pos = matchIndex.saturating_sub(ms.window.base_offset) as usize;
            }
        }

        if matchLength > bestLength {
            bestLength = matchLength;
            if matchLength as u32 > matchEndIdx.wrapping_sub(matchIndex) {
                matchEndIdx = matchIndex.wrapping_add(matchLength as u32);
            }
        }

        if ip_pos + matchLength == iend_pos {
            // Equal to end → can't tell inf/sup; stop to preserve
            // tree consistency.
            break;
        }

        // Compare the byte after the common prefix to decide direction.
        if buf[match_pos + matchLength] < buf[ip_pos + matchLength] {
            // match is smaller than current
            if let Some(s) = smaller_slot {
                ms.chainTable[s] = matchIndex;
            }
            commonLengthSmaller = matchLength;
            if matchIndex <= btLow {
                smaller_slot = None;
                break;
            }
            smaller_slot = Some(next_base + 1);
            matchIndex = ms.chainTable[next_base + 1];
        } else {
            // match is larger than current
            if let Some(l) = larger_slot {
                ms.chainTable[l] = matchIndex;
            }
            commonLengthLarger = matchLength;
            if matchIndex <= btLow {
                larger_slot = None;
                break;
            }
            larger_slot = Some(next_base);
            matchIndex = ms.chainTable[next_base];
        }

        nbCompares -= 1;
    }

    // Nullify open ends.
    if let Some(s) = smaller_slot {
        ms.chainTable[s] = 0;
    }
    if let Some(l) = larger_slot {
        ms.chainTable[l] = 0;
    }

    let positions = if bestLength > 384 {
        192.min((bestLength - 384) as u32)
    } else {
        0
    };
    debug_assert!(matchEndIdx > curr.wrapping_add(8));
    positions.max(matchEndIdx.wrapping_sub(curr.wrapping_add(8)))
}

/// Upstream baseline litLength frequencies for first-block init —
/// `{4, 2, 1×34}` matches `baseLLfreqs` in `zstd_opt.c`.
const BASE_LL_FREQS: [u32; (MaxLL + 1) as usize] = {
    let mut a = [1u32; (MaxLL + 1) as usize];
    a[0] = 4;
    a[1] = 2;
    a
};

/// Upstream baseline offCode frequencies for first-block init —
/// `{6, 2, 1, 1, 2, 3, 4, 4, 4, 3, 2, 1×21}`.
const BASE_OF_FREQS: [u32; (MaxOff + 1) as usize] = {
    let mut a = [1u32; (MaxOff + 1) as usize];
    a[0] = 6;
    a[1] = 2;
    a[4] = 2;
    a[5] = 3;
    a[6] = 4;
    a[7] = 4;
    a[8] = 4;
    a[9] = 3;
    a[10] = 2;
    a
};

/// Port of `ZSTD_fCost`. Converts an internal `price` in
/// `BITCOST_MULTIPLIER` units back to a fractional-bytes value —
/// debug helper. Upstream gates this behind `DEBUGLEVEL >= 2`.
#[inline]
pub fn ZSTD_fCost(price: i32) -> f64 {
    price as f64 / (BITCOST_MULTIPLIER as f64 * 8.0)
}

/// Port of `ZSTD_fWeight`. Fractional-bit debug formatter, used by
/// upstream's `ZSTD_debugTable`. Mirrors `ZSTD_fracWeight` but divides
/// down to a `f64` bit-count.
#[inline]
pub fn ZSTD_fWeight(rawStat: u32) -> f64 {
    ZSTD_fracWeight(rawStat) as f64 / BITCOST_MULTIPLIER as f64
}

#[inline]
fn fse_max_nb_bits_from_ctable(ctable: &[u32], symbol: u32) -> u32 {
    let mut state = FSE_CState_t {
        value: 0,
        stateLog: 0,
    };
    FSE_initCState(&mut state, ctable);
    let tt = symbolTT_read(ctable, state.stateLog, symbol as usize);
    ((tt.deltaNbBits + ((1 << 16) - 1)) as u32) >> 16
}

/// Port of `ZSTD_rescaleFreqs`. Initializes the
/// per-block frequency tables:
///   * First block (`litLengthSum == 0`):
///       - when valid dictionary entropy tables are available, seed
///         literal/HUF and LL/ML/OF/FSE frequencies from them.
///       - otherwise, literals are counted directly from `src` via
///         `HIST_count_simple` and the sequence streams use baseline
///         priors (`BASE_LL_FREQS`, all-1s for ML, `BASE_OF_FREQS`).
///       - Predef price path for very small inputs.
///   * Subsequent blocks: scale existing accumulators down by
///     `ZSTD_scaleStats` so price drift stays bounded.
///
/// After either path, `ZSTD_setBasePrices` recomputes the per-symbol
/// floors.
pub fn ZSTD_rescaleFreqs(
    optPtr: &mut optState_t,
    src: &[u8],
    srcSize: usize,
    optLevel: i32,
    symbolCosts: Option<&ZSTD_entropyCTables_t>,
) {
    optPtr.priceType = ZSTD_OptPrice_e::zop_dynamic;
    let compressedLiterals = ZSTD_compressedLiterals(optPtr);

    if optPtr.litLengthSum == 0 {
        // First block — seed from baseline priors.
        if srcSize <= ZSTD_PREDEF_THRESHOLD as usize {
            optPtr.priceType = ZSTD_OptPrice_e::zop_predef;
        }

        if let Some(symbolCosts) =
            symbolCosts.filter(|s| s.huf.repeatMode == HUF_repeat::HUF_repeat_valid)
        {
            if compressedLiterals {
                optPtr.litSum = 0;
                for lit in 0..=MaxLit as usize {
                    let scaleLog = 11u32;
                    let bitCost = HUF_getNbBitsFromCTable(&symbolCosts.huf.CTable, lit as u32);
                    debug_assert!(bitCost <= scaleLog);
                    optPtr.litFreq[lit] = if bitCost != 0 {
                        1u32 << (scaleLog - bitCost)
                    } else {
                        1
                    };
                    optPtr.litSum += optPtr.litFreq[lit];
                }
            }

            optPtr.litLengthSum = 0;
            for ll in 0..=MaxLL as usize {
                let scaleLog = 10u32;
                let bitCost =
                    fse_max_nb_bits_from_ctable(&symbolCosts.fse.litlengthCTable, ll as u32);
                debug_assert!(bitCost < scaleLog);
                optPtr.litLengthFreq[ll] = if bitCost != 0 {
                    1u32 << (scaleLog - bitCost)
                } else {
                    1
                };
                optPtr.litLengthSum += optPtr.litLengthFreq[ll];
            }

            optPtr.matchLengthSum = 0;
            for ml in 0..=MaxML as usize {
                let scaleLog = 10u32;
                let bitCost =
                    fse_max_nb_bits_from_ctable(&symbolCosts.fse.matchlengthCTable, ml as u32);
                debug_assert!(bitCost < scaleLog);
                optPtr.matchLengthFreq[ml] = if bitCost != 0 {
                    1u32 << (scaleLog - bitCost)
                } else {
                    1
                };
                optPtr.matchLengthSum += optPtr.matchLengthFreq[ml];
            }

            optPtr.offCodeSum = 0;
            for of in 0..=MaxOff as usize {
                let scaleLog = 10u32;
                let bitCost =
                    fse_max_nb_bits_from_ctable(&symbolCosts.fse.offcodeCTable, of as u32);
                debug_assert!(bitCost < scaleLog);
                optPtr.offCodeFreq[of] = if bitCost != 0 {
                    1u32 << (scaleLog - bitCost)
                } else {
                    1
                };
                optPtr.offCodeSum += optPtr.offCodeFreq[of];
            }
        } else {
            if compressedLiterals {
                let mut lit = MaxLit;
                crate::compress::hist::HIST_count_simple(&mut optPtr.litFreq, &mut lit, src);
                optPtr.litSum = ZSTD_downscaleStats(
                    &mut optPtr.litFreq,
                    MaxLit,
                    8,
                    base_directive_e::base_0possible,
                );
            }

            optPtr.litLengthFreq.copy_from_slice(&BASE_LL_FREQS);
            optPtr.litLengthSum = sum_u32(&BASE_LL_FREQS);

            for f in optPtr.matchLengthFreq.iter_mut() {
                *f = 1;
            }
            optPtr.matchLengthSum = MaxML + 1;

            optPtr.offCodeFreq.copy_from_slice(&BASE_OF_FREQS);
            optPtr.offCodeSum = sum_u32(&BASE_OF_FREQS);
        }
    } else {
        // Subsequent block — downscale accumulated stats.
        if compressedLiterals {
            optPtr.litSum = ZSTD_scaleStats(&mut optPtr.litFreq, MaxLit, 12);
        }
        optPtr.litLengthSum = ZSTD_scaleStats(&mut optPtr.litLengthFreq, MaxLL, 11);
        optPtr.matchLengthSum = ZSTD_scaleStats(&mut optPtr.matchLengthFreq, MaxML, 11);
        optPtr.offCodeSum = ZSTD_scaleStats(&mut optPtr.offCodeFreq, MaxOff, 11);
    }

    ZSTD_setBasePrices(optPtr, optLevel);
}

/// Port of `ZSTD_insertBtAndGetAllMatches`.
///
/// Walks repcodes, the 3-byte hash (when `mls == 3`), and the binary
/// tree rooted at the current position. Any match longer than
/// `lengthToBeat - 1` is appended to `matches` (kept in ascending
/// length order — the tree walk discovers them that way). Returns
/// the number of matches written.
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_insertBtAndGetAllMatches(
    matches: &mut [ZSTD_match_t],
    ms: &mut ZSTD_MatchState_t,
    nextToUpdate3: &mut u32,
    buf: &[u8],
    ip_abs: u32,
    ilimit_pos: usize,
    dictMode: ZSTD_dictMode_e,
    rep: &[u32; ZSTD_REP_NUM],
    ll0: u32,
    lengthToBeat: u32,
    mls: u32,
) -> u32 {
    use crate::compress::match_state::ZSTD_index_overlap_check;
    use crate::compress::seq_store::{OFFSET_TO_OFFBASE, REPCODE_TO_OFFBASE};

    let sufficient_len = ms.cParams.targetLength.min(ZSTD_OPT_NUM as u32 - 1);
    let curr = ip_abs;
    let hashLog = ms.cParams.hashLog;
    let minMatch: u32 = if mls == 3 { 3 } else { 4 };
    let base_off = ms.window.base_offset;
    debug_assert!(ip_abs >= base_off);
    let ip_pos = ip_abs.wrapping_sub(base_off) as usize;
    let h = ZSTD_hashPtr(&buf[ip_pos..], hashLog, mls);
    let btLog = ms.cParams.chainLog - 1;
    let btMask: u32 = (1u32 << btLog) - 1;
    let mut commonLengthSmaller = 0usize;
    let mut commonLengthLarger = 0usize;
    let dictLimit = ms.window.dictLimit;
    let btLow: u32 = curr.saturating_sub(btMask);
    let windowLow = ZSTD_getLowestMatchIndex(ms, curr, ms.cParams.windowLog);
    let matchLow = if windowLow == 0 { 1 } else { windowLow };
    let prefixStart = dictLimit.saturating_sub(ms.window.base_offset) as usize;
    let dictBaseOffset = ms.window.dictBase_offset;
    let extDictPtr = ms.dictContent.as_ptr();
    let extDictEnd = ms.dictContent.len();
    let extDict = || unsafe { core::slice::from_raw_parts(extDictPtr, extDictEnd) };

    let dms_ptr = if dictMode == ZSTD_dictMode_e::ZSTD_dictMatchState {
        ms.dictMatchState
            .as_deref()
            .map_or(core::ptr::null(), |dms| dms as *const ZSTD_MatchState_t)
    } else {
        core::ptr::null()
    };
    // The current match state is mutated below, while the attached
    // dictionary match state is read-only. A raw pointer preserves
    // upstream's two-state model without cloning dictionary tables.
    let dms_ref = || unsafe { dms_ptr.as_ref() };
    let (dmsBaseOff, dmsLowLimit, dmsHighLimit, dmsIndexDelta, dmsHashLog, dmsBtMask, dmsBtLow) =
        if let Some(dms) = dms_ref() {
            let dmsHighLimit = dms.window.nextSrc;
            let dmsLowLimit = dms.window.lowLimit;
            let dmsIndexDelta = windowLow.wrapping_sub(dmsHighLimit);
            let dmsBtLog = dms.cParams.chainLog - 1;
            let dmsBtMask = (1u32 << dmsBtLog) - 1;
            let span = dmsHighLimit.saturating_sub(dmsLowLimit);
            let dmsBtLow = if dmsBtMask < span {
                dmsHighLimit - dmsBtMask
            } else {
                dmsLowLimit
            };
            (
                dms.window.base_offset,
                dmsLowLimit,
                dmsHighLimit,
                dmsIndexDelta,
                dms.cParams.hashLog,
                dmsBtMask,
                dmsBtLow,
            )
        } else {
            (0, 0, 0, 0, hashLog, 0, 0)
        };

    let mut smaller_slot: Option<usize> = Some((2 * (curr & btMask)) as usize);
    let mut larger_slot: Option<usize> = Some((2 * (curr & btMask)) as usize + 1);
    let mut matchEndIdx: u32 = curr.wrapping_add(8).wrapping_add(1);
    let mut mnum: u32 = 0;
    let mut nbCompares: u32 = 1u32 << ms.cParams.searchLog;
    let mut bestLength: usize = (lengthToBeat - 1) as usize;

    // ---- 1. Rep codes ----
    debug_assert!(ll0 <= 1);
    let lastR = (ZSTD_REP_NUM as u32).wrapping_add(ll0);
    for repCode in ll0..lastR {
        let repOffset = if repCode == ZSTD_REP_NUM as u32 {
            rep[0] - 1
        } else {
            rep[repCode as usize]
        };
        if repOffset == 0 {
            continue;
        }
        if repOffset.wrapping_sub(1) < curr.wrapping_sub(dictLimit) {
            let repIndex = curr.wrapping_sub(repOffset);
            if repIndex >= windowLow
                && ZSTD_readMINMATCH(&buf[ip_pos..], minMatch)
                    == ZSTD_readMINMATCH(&buf[ip_pos - repOffset as usize..], minMatch)
            {
                let fwd = ZSTD_count(
                    buf,
                    ip_pos + minMatch as usize,
                    ip_pos + minMatch as usize - repOffset as usize,
                    ilimit_pos,
                );
                let repLen = fwd + minMatch as usize;
                if repLen > bestLength {
                    bestLength = repLen;
                    matches[mnum as usize].off = REPCODE_TO_OFFBASE(repCode - ll0 + 1);
                    matches[mnum as usize].len = repLen as u32;
                    mnum += 1;
                    if repLen as u32 > sufficient_len || ip_pos + repLen == ilimit_pos {
                        return mnum;
                    }
                }
            }
        } else {
            let repIndex = curr.wrapping_sub(repOffset);
            let mut repLen = 0usize;
            if dictMode == ZSTD_dictMode_e::ZSTD_extDict {
                let repMatch = repIndex.saturating_sub(dictBaseOffset) as usize;
                if repOffset.wrapping_sub(1) < curr.wrapping_sub(windowLow)
                    && ZSTD_index_overlap_check(dictLimit, repIndex)
                    && ZSTD_minMatchEquals_2segments(
                        buf,
                        ip_pos,
                        prefixStart,
                        extDict(),
                        repMatch,
                        minMatch,
                    )
                {
                    repLen = ZSTD_count_2segments(
                        buf,
                        ip_pos + minMatch as usize,
                        ilimit_pos,
                        prefixStart,
                        extDict(),
                        repMatch + minMatch as usize,
                        extDictEnd,
                    ) + minMatch as usize;
                }
            } else if let Some(dms) = dms_ref() {
                let repMatch = repIndex
                    .wrapping_sub(dmsIndexDelta)
                    .wrapping_sub(dmsBaseOff) as usize;
                if repOffset.wrapping_sub(1)
                    < curr.wrapping_sub(dmsLowLimit.wrapping_add(dmsIndexDelta))
                    && ZSTD_index_overlap_check(dictLimit, repIndex)
                    && ZSTD_minMatchEquals_2segments(
                        buf,
                        ip_pos,
                        prefixStart,
                        &dms.dictContent,
                        repMatch,
                        minMatch,
                    )
                {
                    repLen = ZSTD_count_2segments(
                        buf,
                        ip_pos + minMatch as usize,
                        ilimit_pos,
                        prefixStart,
                        &dms.dictContent,
                        repMatch + minMatch as usize,
                        dms.dictContent.len(),
                    ) + minMatch as usize;
                }
            }
            if repLen > bestLength {
                bestLength = repLen;
                matches[mnum as usize].off = REPCODE_TO_OFFBASE(repCode - ll0 + 1);
                matches[mnum as usize].len = repLen as u32;
                mnum += 1;
                if repLen as u32 > sufficient_len || ip_pos + repLen == ilimit_pos {
                    return mnum;
                }
            }
        }
    }

    // ---- 2. Hash3 lookup (only when mls == 3) ----
    if mls == 3 && (bestLength as u32) < mls {
        let matchIndex3 = ZSTD_insertAndFindFirstIndexHash3(ms, nextToUpdate3, buf, ip_abs);
        if matchIndex3 >= matchLow && curr.wrapping_sub(matchIndex3) < (1u32 << 18) {
            let mlen = if dictMode == ZSTD_dictMode_e::ZSTD_extDict && matchIndex3 < dictLimit {
                let match_pos = matchIndex3.saturating_sub(dictBaseOffset) as usize;
                ZSTD_count_2segments(
                    buf,
                    ip_pos,
                    ilimit_pos,
                    prefixStart,
                    extDict(),
                    match_pos,
                    extDictEnd,
                )
            } else {
                let match_pos = matchIndex3.saturating_sub(ms.window.base_offset) as usize;
                ZSTD_count(buf, ip_pos, match_pos, ilimit_pos)
            };
            if mlen >= mls as usize {
                bestLength = mlen;
                matches[0].off = OFFSET_TO_OFFBASE(curr.wrapping_sub(matchIndex3));
                matches[0].len = mlen as u32;
                mnum = 1;
                if mlen as u32 > sufficient_len || ip_pos + mlen == ilimit_pos {
                    ms.nextToUpdate = curr.wrapping_add(1);
                    return 1;
                }
            }
        }
    }

    // ---- 3. Binary-tree walk ----
    let mut matchIndex = ms.hashTable[h];
    ms.hashTable[h] = curr;

    while nbCompares > 0 && matchIndex >= matchLow {
        let nextBase = (2 * (matchIndex & btMask)) as usize;
        let mut matchLength = commonLengthSmaller.min(commonLengthLarger);
        debug_assert!(curr > matchIndex);

        let mut match_pos = matchIndex.saturating_sub(ms.window.base_offset) as usize;
        if dictMode == ZSTD_dictMode_e::ZSTD_noDict
            || dictMode == ZSTD_dictMode_e::ZSTD_dictMatchState
            || matchIndex.wrapping_add(matchLength as u32) >= dictLimit
        {
            matchLength += ZSTD_count(
                buf,
                ip_pos + matchLength,
                match_pos + matchLength,
                ilimit_pos,
            );
        } else {
            match_pos = matchIndex.saturating_sub(dictBaseOffset) as usize;
            matchLength += ZSTD_count_2segments(
                buf,
                ip_pos + matchLength,
                ilimit_pos,
                prefixStart,
                extDict(),
                match_pos + matchLength,
                extDictEnd,
            );
            if matchIndex.wrapping_add(matchLength as u32) >= dictLimit {
                match_pos = matchIndex.saturating_sub(ms.window.base_offset) as usize;
            }
        }

        if matchLength > bestLength {
            if matchLength as u32 > matchEndIdx.wrapping_sub(matchIndex) {
                matchEndIdx = matchIndex.wrapping_add(matchLength as u32);
            }
            bestLength = matchLength;
            matches[mnum as usize].off = OFFSET_TO_OFFBASE(curr.wrapping_sub(matchIndex));
            matches[mnum as usize].len = matchLength as u32;
            mnum += 1;
            if matchLength as u32 > ZSTD_OPT_NUM as u32 || ip_pos + matchLength == ilimit_pos {
                if dictMode == ZSTD_dictMode_e::ZSTD_dictMatchState {
                    nbCompares = 0;
                }
                break;
            }
        }

        if buf[match_pos + matchLength] < buf[ip_pos + matchLength] {
            if let Some(s) = smaller_slot {
                ms.chainTable[s] = matchIndex;
            }
            commonLengthSmaller = matchLength;
            if matchIndex <= btLow {
                smaller_slot = None;
                break;
            }
            smaller_slot = Some(nextBase + 1);
            matchIndex = ms.chainTable[nextBase + 1];
        } else {
            if let Some(l) = larger_slot {
                ms.chainTable[l] = matchIndex;
            }
            commonLengthLarger = matchLength;
            if matchIndex <= btLow {
                larger_slot = None;
                break;
            }
            larger_slot = Some(nextBase);
            matchIndex = ms.chainTable[nextBase];
        }
        nbCompares -= 1;
    }

    if let Some(s) = smaller_slot {
        ms.chainTable[s] = 0;
    }
    if let Some(l) = larger_slot {
        ms.chainTable[l] = 0;
    }

    if dictMode == ZSTD_dictMode_e::ZSTD_dictMatchState && nbCompares > 0 {
        if let Some(dms) = dms_ref() {
            let dmsH = ZSTD_hashPtr(&buf[ip_pos..], dmsHashLog, mls);
            let mut dictMatchIndex = dms.hashTable[dmsH];
            commonLengthSmaller = 0;
            commonLengthLarger = 0;
            while nbCompares > 0 && dictMatchIndex > dmsLowLimit {
                let nextBase = (2 * (dictMatchIndex & dmsBtMask)) as usize;
                let mut matchLength = commonLengthSmaller.min(commonLengthLarger);
                let mut match_pos = dictMatchIndex.saturating_sub(dmsBaseOff) as usize;
                matchLength += ZSTD_count_2segments(
                    buf,
                    ip_pos + matchLength,
                    ilimit_pos,
                    prefixStart,
                    &dms.dictContent,
                    match_pos + matchLength,
                    dms.dictContent.len(),
                );
                if dictMatchIndex.wrapping_add(matchLength as u32) >= dmsHighLimit {
                    match_pos = dictMatchIndex
                        .wrapping_add(dmsIndexDelta)
                        .wrapping_sub(ms.window.base_offset)
                        as usize;
                }

                if matchLength > bestLength {
                    let translatedIndex = dictMatchIndex.wrapping_add(dmsIndexDelta);
                    if matchLength as u32 > matchEndIdx.wrapping_sub(translatedIndex) {
                        matchEndIdx = translatedIndex.wrapping_add(matchLength as u32);
                    }
                    bestLength = matchLength;
                    matches[mnum as usize].off =
                        OFFSET_TO_OFFBASE(curr.wrapping_sub(translatedIndex));
                    matches[mnum as usize].len = matchLength as u32;
                    mnum += 1;
                    if matchLength as u32 > ZSTD_OPT_NUM as u32
                        || ip_pos + matchLength == ilimit_pos
                    {
                        break;
                    }
                }

                if dictMatchIndex <= dmsBtLow {
                    break;
                }
                let crossedIntoPrefix =
                    dictMatchIndex.wrapping_add(matchLength as u32) >= dmsHighLimit;
                let matchByte = ZSTD_dmsOrderingByte(
                    buf,
                    match_pos,
                    &dms.dictContent,
                    dictMatchIndex.saturating_sub(dmsBaseOff) as usize,
                    matchLength,
                    crossedIntoPrefix,
                );
                if matchByte < buf[ip_pos + matchLength] {
                    commonLengthSmaller = matchLength;
                    dictMatchIndex = dms.chainTable[nextBase + 1];
                } else {
                    commonLengthLarger = matchLength;
                    dictMatchIndex = dms.chainTable[nextBase];
                }
                nbCompares -= 1;
            }
        }
    }

    debug_assert!(matchEndIdx > curr.wrapping_add(8));
    ms.nextToUpdate = matchEndIdx.wrapping_sub(8);
    mnum
}

#[inline]
fn ZSTD_dmsOrderingByte(
    prefix_buf: &[u8],
    prefix_match_pos: usize,
    dms_buf: &[u8],
    dms_match_pos: usize,
    matchLength: usize,
    crossedIntoPrefix: bool,
) -> u8 {
    if crossedIntoPrefix {
        prefix_buf[prefix_match_pos + matchLength]
    } else {
        dms_buf[dms_match_pos + matchLength]
    }
}

#[inline]
fn ZSTD_minMatchEquals_2segments(
    prefix_buf: &[u8],
    prefix_pos: usize,
    prefix_start: usize,
    dict_buf: &[u8],
    dict_pos: usize,
    min_match: u32,
) -> bool {
    let dict_remaining = dict_buf.len().saturating_sub(dict_pos);
    for i in 0..min_match as usize {
        let dict_byte = if i < dict_remaining {
            dict_buf[dict_pos + i]
        } else {
            prefix_buf[prefix_start + (i - dict_remaining)]
        };
        if prefix_buf[prefix_pos + i] != dict_byte {
            return false;
        }
    }
    true
}

/// Port of `ZSTD_btGetAllMatches_internal` — no-dict variant. Thin
/// wrapper: updates the binary tree up to `ip_abs`, then runs
/// `ZSTD_insertBtAndGetAllMatches_prefixOnly`. Returns the match
/// count (0 when `ip_abs` is already covered by prior tree updates).
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_btGetAllMatches_internal(
    matches: &mut [ZSTD_match_t],
    ms: &mut ZSTD_MatchState_t,
    nextToUpdate3: &mut u32,
    buf: &[u8],
    ip_abs: u32,
    ihigh_pos: usize,
    rep: &[u32; ZSTD_REP_NUM],
    ll0: u32,
    lengthToBeat: u32,
    dictMode: ZSTD_dictMode_e,
    mls: u32,
) -> u32 {
    debug_assert!((3..=6).contains(&mls));
    if ip_abs < ms.nextToUpdate {
        return 0;
    }
    ZSTD_updateTree_internal(ms, buf, ihigh_pos, ip_abs, mls, dictMode);
    ZSTD_insertBtAndGetAllMatches(
        matches,
        ms,
        nextToUpdate3,
        buf,
        ip_abs,
        ihigh_pos,
        dictMode,
        rep,
        ll0,
        lengthToBeat,
        mls,
    )
}

pub fn ZSTD_btGetAllMatches_noDict(
    matches: &mut [ZSTD_match_t],
    ms: &mut ZSTD_MatchState_t,
    nextToUpdate3: &mut u32,
    buf: &[u8],
    ip_abs: u32,
    ihigh_pos: usize,
    rep: &[u32; ZSTD_REP_NUM],
    ll0: u32,
    lengthToBeat: u32,
    mls: u32,
) -> u32 {
    ZSTD_btGetAllMatches_internal(
        matches,
        ms,
        nextToUpdate3,
        buf,
        ip_abs,
        ihigh_pos,
        rep,
        ll0,
        lengthToBeat,
        ZSTD_dictMode_e::ZSTD_noDict,
        mls,
    )
}

pub fn ZSTD_btGetAllMatches_extDict(
    matches: &mut [ZSTD_match_t],
    ms: &mut ZSTD_MatchState_t,
    nextToUpdate3: &mut u32,
    buf: &[u8],
    ip_abs: u32,
    ihigh_pos: usize,
    rep: &[u32; ZSTD_REP_NUM],
    ll0: u32,
    lengthToBeat: u32,
    mls: u32,
) -> u32 {
    ZSTD_btGetAllMatches_internal(
        matches,
        ms,
        nextToUpdate3,
        buf,
        ip_abs,
        ihigh_pos,
        rep,
        ll0,
        lengthToBeat,
        ZSTD_dictMode_e::ZSTD_extDict,
        mls,
    )
}

pub fn ZSTD_btGetAllMatches_dictMatchState(
    matches: &mut [ZSTD_match_t],
    ms: &mut ZSTD_MatchState_t,
    nextToUpdate3: &mut u32,
    buf: &[u8],
    ip_abs: u32,
    ihigh_pos: usize,
    rep: &[u32; ZSTD_REP_NUM],
    ll0: u32,
    lengthToBeat: u32,
    mls: u32,
) -> u32 {
    ZSTD_btGetAllMatches_internal(
        matches,
        ms,
        nextToUpdate3,
        buf,
        ip_abs,
        ihigh_pos,
        rep,
        ll0,
        lengthToBeat,
        ZSTD_dictMode_e::ZSTD_dictMatchState,
        mls,
    )
}

/// Mirrors upstream's macro family root `GEN_ZSTD_BT_GET_ALL_MATCHES`.
pub fn GEN_ZSTD_BT_GET_ALL_MATCHES() {
    type GetAllMatchesFn = fn(
        &mut [ZSTD_match_t],
        &mut ZSTD_MatchState_t,
        &mut u32,
        &[u8],
        u32,
        usize,
        &[u32; ZSTD_REP_NUM],
        u32,
        u32,
        u32,
    ) -> u32;

    let dispatchTable: [GetAllMatchesFn; 4] = [
        ZSTD_btGetAllMatches_noDict,
        ZSTD_btGetAllMatches_extDict,
        ZSTD_btGetAllMatches_dictMatchState,
        ZSTD_btGetAllMatches_noDict,
    ];
    let _ = dispatchTable;
}

/// Port of `ZSTD_updateTree_internal`. Walks the tree from
/// `ms.nextToUpdate` up to `target`, calling `ZSTD_insertBt1` at each
/// position. `dictMatchState` shares the regular prefix tree update;
/// `extDict` enables the two-segment compare path while building the
/// local tree.
///
/// `buf` spans the window; `iend_pos` is the end offset used for
/// match-length counting (`ZSTD_count` clamps at it).
pub fn ZSTD_updateTree_internal(
    ms: &mut ZSTD_MatchState_t,
    buf: &[u8],
    iend_pos: usize,
    target: u32,
    mls: u32,
    dictMode: ZSTD_dictMode_e,
) {
    let mut idx = ms.nextToUpdate;
    while idx < target {
        let forward = ZSTD_insertBt1(
            ms,
            buf,
            idx,
            iend_pos,
            target,
            mls,
            dictMode == ZSTD_dictMode_e::ZSTD_extDict,
        );
        debug_assert!(forward > 0, "insertBt1 must cover at least one position");
        idx += forward;
    }
    ms.nextToUpdate = target;
}

/// Port of `ZSTD_updateTree`. Thin wrapper over `_internal` using
/// `ms.cParams.minMatch` and the no-dict code path.
pub fn ZSTD_updateTree(ms: &mut ZSTD_MatchState_t, buf: &[u8], ip_abs: u32, iend_pos: usize) {
    let mls = ms.cParams.minMatch;
    ZSTD_updateTree_internal(ms, buf, iend_pos, ip_abs, mls, ZSTD_dictMode_e::ZSTD_noDict);
}

#[inline]
fn LIT_PRICE(literals: &[u8], optPtr: &optState_t, optLevel: i32) -> i32 {
    ZSTD_rawLiteralsCost(literals, 1, optPtr, optLevel) as i32
}

#[inline]
fn LL_PRICE(litLength: u32, optPtr: &optState_t, optLevel: i32) -> i32 {
    ZSTD_litLengthPrice(litLength, optPtr, optLevel) as i32
}

#[inline]
fn LL_INCPRICE(litLength: u32, optPtr: &optState_t, optLevel: i32) -> i32 {
    LL_PRICE(litLength, optPtr, optLevel) - LL_PRICE(litLength - 1, optPtr, optLevel)
}

/// Port of `ZSTD_compressBlock_opt_generic`.
#[allow(clippy::too_many_lines)]
fn ZSTD_compressBlock_opt_generic_window(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut crate::compress::seq_store::SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    window_buf: &[u8],
    src_pos: usize,
    src_end: usize,
    optLevel: i32,
    dictMode: ZSTD_dictMode_e,
) -> usize {
    let mut optState = core::mem::take(&mut ms.opt);
    let mut optLdm = ZSTD_optLdm_t {
        seqStore: ZSTD_cloneActiveRawSeqStore(ms.ldmSeqStore.as_ref()),
        ..Default::default()
    };
    let src = &window_buf[src_pos..src_end];
    let cParams = ms.cParams;
    let sufficient_len = cParams.targetLength.min(ZSTD_OPT_NUM as u32 - 1);
    let minMatch = if cParams.minMatch == 3 { 3 } else { 4 };
    let mls = cParams.minMatch.clamp(3, 6);
    let mut nextToUpdate3 = ms.nextToUpdate;
    let mut ip: usize = 0;
    let mut anchor: usize = 0;
    let iend = src.len();
    let ilimit = iend.saturating_sub(8);
    let base_off = ms.window.base_offset;
    let prefixStart = ms.window.dictLimit.saturating_sub(base_off) as usize;
    let mut lastStretch = ZSTD_optimal_t::default();
    let src_len = src.len();

    debug_assert!(optLevel <= 2);
    ZSTD_opt_getNextMatchAndUpdateSeqStore(&mut optLdm, ip as u32, (iend - ip) as u32);
    ZSTD_rescaleFreqs(
        &mut optState,
        src,
        src_len,
        optLevel,
        ms.entropySeed.as_ref(),
    );
    ip += usize::from(src_pos + ip == prefixStart);

    'match_loop: while ip < ilimit {
        let mut cur: u32;
        let mut last_pos: u32;
        let litlen = (ip - anchor) as u32;
        let ll0 = u32::from(litlen == 0);
        let ip_abs = base_off.wrapping_add(src_pos.wrapping_add(ip) as u32);
        let mut nbMatches = ZSTD_btGetAllMatches_internal(
            &mut optState.matchTable,
            ms,
            &mut nextToUpdate3,
            window_buf,
            ip_abs,
            src_end,
            rep,
            ll0,
            minMatch,
            dictMode,
            mls,
        );
        ZSTD_optLdm_processMatchCandidate(
            &mut optLdm,
            &mut optState.matchTable,
            &mut nbMatches,
            ip as u32,
            (iend - ip) as u32,
            minMatch,
        );
        if nbMatches == 0 {
            ip += 1;
            continue;
        }

        optState.priceTable[0].mlen = 0;
        optState.priceTable[0].litlen = litlen;
        optState.priceTable[0].price = LL_PRICE(litlen, &optState, optLevel);
        optState.priceTable[0].rep = *rep;

        {
            let max_last_pos = (iend - ip) as u32;
            let maxML = optState.matchTable[nbMatches as usize - 1].len;
            let maxOffBase = optState.matchTable[nbMatches as usize - 1].off;
            if maxML > sufficient_len {
                lastStretch.litlen = 0;
                lastStretch.mlen = maxML;
                lastStretch.off = maxOffBase;
                cur = 0;
                last_pos = maxML;
                goto_shortest_path(
                    &mut lastStretch,
                    &mut cur,
                    &mut last_pos,
                    &mut anchor,
                    &mut ip,
                    iend,
                    seqStore,
                    &mut optState,
                    optLevel,
                    src,
                    rep,
                );
                continue 'match_loop;
            }

            let mut pos = 1u32;
            while pos < minMatch {
                optState.priceTable[pos as usize].price = ZSTD_MAX_PRICE;
                optState.priceTable[pos as usize].mlen = 0;
                optState.priceTable[pos as usize].litlen = litlen + pos;
                pos += 1;
            }
            for matchNb in 0..nbMatches as usize {
                let offBase = optState.matchTable[matchNb].off;
                let end = optState.matchTable[matchNb].len;
                while pos <= end {
                    let matchPrice = ZSTD_getMatchPrice(offBase, pos, &optState, optLevel) as i32;
                    let sequencePrice = optState.priceTable[0].price + matchPrice;
                    optState.priceTable[pos as usize].mlen = pos;
                    optState.priceTable[pos as usize].off = offBase;
                    optState.priceTable[pos as usize].litlen = 0;
                    optState.priceTable[pos as usize].price =
                        sequencePrice + LL_PRICE(0, &optState, optLevel);
                    pos += 1;
                }
            }
            last_pos = (pos - 1).min(max_last_pos);
            optState.priceTable[last_pos as usize + 1].price = ZSTD_MAX_PRICE;
        }

        cur = 1;
        while cur <= last_pos {
            if ip + cur as usize > iend {
                last_pos = cur - 1;
                break;
            }
            let inr = ip + cur as usize;
            let litlen_here = optState.priceTable[cur as usize - 1].litlen + 1;
            let price = optState.priceTable[cur as usize - 1].price
                + LIT_PRICE(&src[ip + cur as usize - 1..], &optState, optLevel)
                + LL_INCPRICE(litlen_here, &optState, optLevel);
            if price <= optState.priceTable[cur as usize].price {
                let prevMatch = optState.priceTable[cur as usize];
                optState.priceTable[cur as usize] = optState.priceTable[cur as usize - 1];
                optState.priceTable[cur as usize].litlen = litlen_here;
                optState.priceTable[cur as usize].price = price;
                if optLevel >= 1
                    && prevMatch.litlen == 0
                    && LL_INCPRICE(1, &optState, optLevel) < 0
                    && ip + (cur as usize) < iend
                    && (cur as usize + 1) < optState.priceTable.len()
                {
                    let with1literal = prevMatch.price
                        + LIT_PRICE(&src[ip + cur as usize..], &optState, optLevel)
                        + LL_INCPRICE(1, &optState, optLevel);
                    let withMoreLiterals = price
                        + LIT_PRICE(&src[ip + cur as usize..], &optState, optLevel)
                        + LL_INCPRICE(litlen_here + 1, &optState, optLevel);
                    if with1literal < withMoreLiterals
                        && with1literal < optState.priceTable[cur as usize + 1].price
                    {
                        let prev = cur - prevMatch.mlen;
                        let newReps = ZSTD_newRep(
                            &optState.priceTable[prev as usize].rep,
                            prevMatch.off,
                            u32::from(optState.priceTable[prev as usize].litlen == 0),
                        );
                        optState.priceTable[cur as usize + 1] = prevMatch;
                        optState.priceTable[cur as usize + 1].rep = newReps.rep;
                        optState.priceTable[cur as usize + 1].litlen = 1;
                        optState.priceTable[cur as usize + 1].price = with1literal;
                        if last_pos < cur + 1 {
                            last_pos = cur + 1;
                        }
                    }
                }
            }

            if optState.priceTable[cur as usize].litlen == 0 {
                let prev = cur - optState.priceTable[cur as usize].mlen;
                let newReps = ZSTD_newRep(
                    &optState.priceTable[prev as usize].rep,
                    optState.priceTable[cur as usize].off,
                    u32::from(optState.priceTable[prev as usize].litlen == 0),
                );
                optState.priceTable[cur as usize].rep = newReps.rep;
            }
            if inr > ilimit {
                cur += 1;
                continue;
            }
            if cur == last_pos {
                break;
            }
            if optLevel == 0
                && (cur as usize + 1) < optState.priceTable.len()
                && optState.priceTable[cur as usize + 1].price
                    <= optState.priceTable[cur as usize].price + (BITCOST_MULTIPLIER / 2) as i32
            {
                cur += 1;
                continue;
            }

            let ll0 = u32::from(optState.priceTable[cur as usize].litlen == 0);
            let previousPrice = optState.priceTable[cur as usize].price;
            let basePrice = previousPrice + LL_PRICE(0, &optState, optLevel);
            let inr_abs = base_off.wrapping_add(src_pos.wrapping_add(inr) as u32);
            let mut nbMatches = ZSTD_btGetAllMatches_internal(
                &mut optState.matchTable,
                ms,
                &mut nextToUpdate3,
                window_buf,
                inr_abs,
                src_end,
                &optState.priceTable[cur as usize].rep,
                ll0,
                minMatch,
                dictMode,
                mls,
            );
            ZSTD_optLdm_processMatchCandidate(
                &mut optLdm,
                &mut optState.matchTable,
                &mut nbMatches,
                inr as u32,
                (iend - inr) as u32,
                minMatch,
            );
            if nbMatches == 0 {
                cur += 1;
                continue;
            }

            let longestML = optState.matchTable[nbMatches as usize - 1].len;
            if longestML > sufficient_len
                || (cur + longestML) as usize >= ZSTD_OPT_NUM
                || ip + cur as usize + longestML as usize >= iend
            {
                lastStretch.mlen = longestML;
                lastStretch.off = optState.matchTable[nbMatches as usize - 1].off;
                lastStretch.litlen = 0;
                last_pos = cur + longestML;
                goto_shortest_path(
                    &mut lastStretch,
                    &mut cur,
                    &mut last_pos,
                    &mut anchor,
                    &mut ip,
                    iend,
                    seqStore,
                    &mut optState,
                    optLevel,
                    src,
                    rep,
                );
                continue 'match_loop;
            }

            for matchNb in 0..nbMatches as usize {
                let offset = optState.matchTable[matchNb].off;
                let lastML = optState.matchTable[matchNb].len;
                let startML = if matchNb > 0 {
                    optState.matchTable[matchNb - 1].len + 1
                } else {
                    minMatch
                };
                let mut mlen = lastML;
                loop {
                    let pos = cur + mlen;
                    let price =
                        basePrice + ZSTD_getMatchPrice(offset, mlen, &optState, optLevel) as i32;
                    if pos > last_pos || price < optState.priceTable[pos as usize].price {
                        while last_pos < pos {
                            last_pos += 1;
                            optState.priceTable[last_pos as usize].price = ZSTD_MAX_PRICE;
                            optState.priceTable[last_pos as usize].litlen = !0;
                        }
                        optState.priceTable[pos as usize].mlen = mlen;
                        optState.priceTable[pos as usize].off = offset;
                        optState.priceTable[pos as usize].litlen = 0;
                        optState.priceTable[pos as usize].price = price;
                    } else if optLevel == 0 {
                        break;
                    }
                    if mlen == startML {
                        break;
                    }
                    mlen -= 1;
                }
            }
            optState.priceTable[last_pos as usize + 1].price = ZSTD_MAX_PRICE;
            cur += 1;
        }

        lastStretch = optState.priceTable[last_pos as usize];
        cur = last_pos - lastStretch.mlen;
        goto_shortest_path(
            &mut lastStretch,
            &mut cur,
            &mut last_pos,
            &mut anchor,
            &mut ip,
            iend,
            seqStore,
            &mut optState,
            optLevel,
            src,
            rep,
        );
    }

    ms.opt = optState;
    iend - anchor
}

/// Mirrors upstream's `FORCE_INLINE_TEMPLATE` declaration: each public
/// entry (btopt/btultra/btultra2/btopt_dictMatchState/...) calls this
/// with literal `optLevel` and `dictMode`, so `#[inline(always)]`
/// lets LLVM constant-propagate those values, eliminating per-byte
/// runtime branches on `optLevel == 0/1/2` and `dictMode == ...`.
#[inline(always)]
fn ZSTD_compressBlock_opt_generic(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut crate::compress::seq_store::SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
    optLevel: i32,
    dictMode: ZSTD_dictMode_e,
) -> usize {
    let mut optState = core::mem::take(&mut ms.opt);
    let mut optLdm = ZSTD_optLdm_t {
        seqStore: ZSTD_cloneActiveRawSeqStore(ms.ldmSeqStore.as_ref()),
        ..Default::default()
    };
    let cParams = ms.cParams;
    let sufficient_len = cParams.targetLength.min(ZSTD_OPT_NUM as u32 - 1);
    let minMatch = if cParams.minMatch == 3 { 3 } else { 4 };
    let mls = cParams.minMatch.clamp(3, 6);
    let mut nextToUpdate3 = ms.nextToUpdate;
    let mut ip: usize = 0;
    let mut anchor: usize = 0;
    let iend = src.len();
    let ilimit = iend.saturating_sub(8);
    let base_off = ms.window.base_offset;
    let prefixStart = ms.window.dictLimit.saturating_sub(base_off) as usize;
    let mut lastStretch = ZSTD_optimal_t::default();

    debug_assert!(optLevel <= 2);
    ZSTD_opt_getNextMatchAndUpdateSeqStore(&mut optLdm, ip as u32, (iend - ip) as u32);
    ZSTD_rescaleFreqs(
        &mut optState,
        src,
        src.len(),
        optLevel,
        ms.entropySeed.as_ref(),
    );
    ip += usize::from(ip == prefixStart);

    'match_loop: while ip < ilimit {
        let mut cur: u32;
        let mut last_pos: u32;
        let litlen = (ip - anchor) as u32;
        let ll0 = u32::from(litlen == 0);
        let ip_abs = base_off.wrapping_add(ip as u32);
        let mut nbMatches = ZSTD_btGetAllMatches_internal(
            &mut optState.matchTable,
            ms,
            &mut nextToUpdate3,
            src,
            ip_abs,
            iend,
            rep,
            ll0,
            minMatch,
            dictMode,
            mls,
        );
        ZSTD_optLdm_processMatchCandidate(
            &mut optLdm,
            &mut optState.matchTable,
            &mut nbMatches,
            ip as u32,
            (iend - ip) as u32,
            minMatch,
        );
        if nbMatches == 0 {
            ip += 1;
            continue;
        }

        optState.priceTable[0].mlen = 0;
        optState.priceTable[0].litlen = litlen;
        optState.priceTable[0].price = LL_PRICE(litlen, &optState, optLevel);
        optState.priceTable[0].rep = *rep;

        {
            let maxML = optState.matchTable[nbMatches as usize - 1].len;
            let maxOffBase = optState.matchTable[nbMatches as usize - 1].off;
            if maxML > sufficient_len {
                lastStretch.litlen = 0;
                lastStretch.mlen = maxML;
                lastStretch.off = maxOffBase;
                cur = 0;
                last_pos = maxML;
                goto_shortest_path(
                    &mut lastStretch,
                    &mut cur,
                    &mut last_pos,
                    &mut anchor,
                    &mut ip,
                    iend,
                    seqStore,
                    &mut optState,
                    optLevel,
                    src,
                    rep,
                );
                continue 'match_loop;
            }

            let mut pos = 1u32;
            while pos < minMatch {
                optState.priceTable[pos as usize].price = ZSTD_MAX_PRICE;
                optState.priceTable[pos as usize].mlen = 0;
                optState.priceTable[pos as usize].litlen = litlen + pos;
                pos += 1;
            }
            for matchNb in 0..nbMatches as usize {
                let offBase = optState.matchTable[matchNb].off;
                let end = optState.matchTable[matchNb].len;
                while pos <= end {
                    let matchPrice = ZSTD_getMatchPrice(offBase, pos, &optState, optLevel) as i32;
                    let sequencePrice = optState.priceTable[0].price + matchPrice;
                    optState.priceTable[pos as usize].mlen = pos;
                    optState.priceTable[pos as usize].off = offBase;
                    optState.priceTable[pos as usize].litlen = 0;
                    optState.priceTable[pos as usize].price =
                        sequencePrice + LL_PRICE(0, &optState, optLevel);
                    pos += 1;
                }
            }
            last_pos = pos - 1;
            optState.priceTable[pos as usize].price = ZSTD_MAX_PRICE;
        }

        cur = 1;
        while cur <= last_pos {
            if ip + cur as usize > iend {
                last_pos = cur - 1;
                break;
            }
            let inr = ip + cur as usize;
            let litlen_here = optState.priceTable[cur as usize - 1].litlen + 1;
            let price = optState.priceTable[cur as usize - 1].price
                + LIT_PRICE(&src[ip + cur as usize - 1..], &optState, optLevel)
                + LL_INCPRICE(litlen_here, &optState, optLevel);
            if price <= optState.priceTable[cur as usize].price {
                let prevMatch = optState.priceTable[cur as usize];
                optState.priceTable[cur as usize] = optState.priceTable[cur as usize - 1];
                optState.priceTable[cur as usize].litlen = litlen_here;
                optState.priceTable[cur as usize].price = price;
                if optLevel >= 1
                    && prevMatch.litlen == 0
                    && LL_INCPRICE(1, &optState, optLevel) < 0
                    && ip + (cur as usize) < iend
                    && (cur as usize + 1) < optState.priceTable.len()
                {
                    let with1literal = prevMatch.price
                        + LIT_PRICE(&src[ip + cur as usize..], &optState, optLevel)
                        + LL_INCPRICE(1, &optState, optLevel);
                    let withMoreLiterals = price
                        + LIT_PRICE(&src[ip + cur as usize..], &optState, optLevel)
                        + LL_INCPRICE(litlen_here + 1, &optState, optLevel);
                    if with1literal < withMoreLiterals
                        && with1literal < optState.priceTable[cur as usize + 1].price
                    {
                        let prev = cur - prevMatch.mlen;
                        let newReps = ZSTD_newRep(
                            &optState.priceTable[prev as usize].rep,
                            prevMatch.off,
                            u32::from(optState.priceTable[prev as usize].litlen == 0),
                        );
                        optState.priceTable[cur as usize + 1] = prevMatch;
                        optState.priceTable[cur as usize + 1].rep = newReps.rep;
                        optState.priceTable[cur as usize + 1].litlen = 1;
                        optState.priceTable[cur as usize + 1].price = with1literal;
                        if last_pos < cur + 1 {
                            last_pos = cur + 1;
                        }
                    }
                }
            }

            if optState.priceTable[cur as usize].litlen == 0 {
                let prev = cur - optState.priceTable[cur as usize].mlen;
                let newReps = ZSTD_newRep(
                    &optState.priceTable[prev as usize].rep,
                    optState.priceTable[cur as usize].off,
                    u32::from(optState.priceTable[prev as usize].litlen == 0),
                );
                optState.priceTable[cur as usize].rep = newReps.rep;
            }

            if inr > ilimit {
                cur += 1;
                continue;
            }
            if cur == last_pos {
                break;
            }
            if optLevel == 0
                && (cur as usize + 1) < optState.priceTable.len()
                && optState.priceTable[cur as usize + 1].price
                    <= optState.priceTable[cur as usize].price + (BITCOST_MULTIPLIER / 2) as i32
            {
                cur += 1;
                continue;
            }

            let ll0 = u32::from(optState.priceTable[cur as usize].litlen == 0);
            let previousPrice = optState.priceTable[cur as usize].price;
            let basePrice = previousPrice + LL_PRICE(0, &optState, optLevel);
            let inr_abs = base_off.wrapping_add(inr as u32);
            let mut nbMatches = ZSTD_btGetAllMatches_internal(
                &mut optState.matchTable,
                ms,
                &mut nextToUpdate3,
                src,
                inr_abs,
                iend,
                &optState.priceTable[cur as usize].rep,
                ll0,
                minMatch,
                dictMode,
                mls,
            );
            ZSTD_optLdm_processMatchCandidate(
                &mut optLdm,
                &mut optState.matchTable,
                &mut nbMatches,
                inr as u32,
                (iend - inr) as u32,
                minMatch,
            );
            if nbMatches == 0 {
                cur += 1;
                continue;
            }

            let longestML = optState.matchTable[nbMatches as usize - 1].len;
            if longestML > sufficient_len
                || (cur + longestML) as usize >= ZSTD_OPT_NUM
                || ip + cur as usize + longestML as usize >= iend
            {
                lastStretch.mlen = longestML;
                lastStretch.off = optState.matchTable[nbMatches as usize - 1].off;
                lastStretch.litlen = 0;
                last_pos = cur + longestML;
                goto_shortest_path(
                    &mut lastStretch,
                    &mut cur,
                    &mut last_pos,
                    &mut anchor,
                    &mut ip,
                    iend,
                    seqStore,
                    &mut optState,
                    optLevel,
                    src,
                    rep,
                );
                continue 'match_loop;
            }

            for matchNb in 0..nbMatches as usize {
                let offset = optState.matchTable[matchNb].off;
                let lastML = optState.matchTable[matchNb].len;
                let startML = if matchNb > 0 {
                    optState.matchTable[matchNb - 1].len + 1
                } else {
                    minMatch
                };
                let mut mlen = lastML;
                loop {
                    let pos = cur + mlen;
                    let price =
                        basePrice + ZSTD_getMatchPrice(offset, mlen, &optState, optLevel) as i32;
                    if pos > last_pos || price < optState.priceTable[pos as usize].price {
                        while last_pos < pos {
                            last_pos += 1;
                            optState.priceTable[last_pos as usize].price = ZSTD_MAX_PRICE;
                            optState.priceTable[last_pos as usize].litlen = !0;
                        }
                        optState.priceTable[pos as usize].mlen = mlen;
                        optState.priceTable[pos as usize].off = offset;
                        optState.priceTable[pos as usize].litlen = 0;
                        optState.priceTable[pos as usize].price = price;
                    } else if optLevel == 0 {
                        break;
                    }
                    if mlen == startML {
                        break;
                    }
                    mlen -= 1;
                }
            }
            optState.priceTable[last_pos as usize + 1].price = ZSTD_MAX_PRICE;
            cur += 1;
        }

        lastStretch = optState.priceTable[last_pos as usize];
        cur = last_pos - lastStretch.mlen;
        goto_shortest_path(
            &mut lastStretch,
            &mut cur,
            &mut last_pos,
            &mut anchor,
            &mut ip,
            iend,
            seqStore,
            &mut optState,
            optLevel,
            src,
            rep,
        );
    }

    ms.opt = optState;
    src.len() - anchor
}

#[allow(clippy::too_many_arguments)]
fn goto_shortest_path(
    lastStretch: &mut ZSTD_optimal_t,
    cur: &mut u32,
    last_pos: &mut u32,
    anchor: &mut usize,
    ip: &mut usize,
    iend: usize,
    seqStore: &mut crate::compress::seq_store::SeqStore_t,
    optStatePtr: &mut optState_t,
    optLevel: i32,
    src: &[u8],
    rep: &mut [u32; ZSTD_REP_NUM],
) {
    if lastStretch.mlen == 0 {
        *ip += *last_pos as usize;
        return;
    }

    if lastStretch.litlen == 0 {
        let reps = ZSTD_newRep(
            &optStatePtr.priceTable[*cur as usize].rep,
            lastStretch.off,
            u32::from(optStatePtr.priceTable[*cur as usize].litlen == 0),
        );
        *rep = reps.rep;
    } else {
        *rep = lastStretch.rep;
        *cur -= lastStretch.litlen;
    }

    let storeEnd = *cur + 2;
    let mut storeStart = storeEnd;
    let mut stretchPos = *cur;
    optStatePtr.priceTable[storeEnd as usize] = *lastStretch;
    loop {
        let nextStretch = optStatePtr.priceTable[stretchPos as usize];
        optStatePtr.priceTable[storeStart as usize].litlen = nextStretch.litlen;
        if nextStretch.mlen == 0 {
            break;
        }
        storeStart -= 1;
        optStatePtr.priceTable[storeStart as usize] = nextStretch;
        stretchPos -= nextStretch.litlen + nextStretch.mlen;
    }

    for storePos in storeStart..=storeEnd {
        let llen = optStatePtr.priceTable[storePos as usize].litlen as usize;
        let mlen = optStatePtr.priceTable[storePos as usize].mlen as usize;
        let offBase = optStatePtr.priceTable[storePos as usize].off;
        let advance = llen + mlen;

        if mlen == 0 {
            *ip = *anchor + llen;
            continue;
        }

        debug_assert!(*anchor + llen <= iend);
        ZSTD_updateStats(
            optStatePtr,
            llen as u32,
            &src[*anchor..],
            offBase,
            mlen as u32,
        );
        ZSTD_storeSeq(seqStore, llen, &src[*anchor..], offBase, mlen);
        *anchor += advance;
        *ip = *anchor;
    }
    ZSTD_setBasePrices(optStatePtr, optLevel);
}

fn ZSTD_compressBlock_opt0(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut crate::compress::seq_store::SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
    dictMode: ZSTD_dictMode_e,
) -> usize {
    ZSTD_compressBlock_opt_generic(ms, seqStore, rep, src, 0, dictMode)
}

fn ZSTD_compressBlock_opt2(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut crate::compress::seq_store::SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
    dictMode: ZSTD_dictMode_e,
) -> usize {
    ZSTD_compressBlock_opt_generic(ms, seqStore, rep, src, 2, dictMode)
}

#[allow(dead_code)]
fn ZSTD_initStats_ultra(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut crate::compress::seq_store::SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) {
    let mut tmpRep = *rep;
    debug_assert!(seqStore.sequences.is_empty());
    debug_assert_eq!(ms.window.dictLimit, ms.window.lowLimit);
    debug_assert!(ms.window.dictLimit.wrapping_sub(ms.nextToUpdate) <= 1);

    let _ = ZSTD_compressBlock_opt2(ms, seqStore, &mut tmpRep, src, ZSTD_dictMode_e::ZSTD_noDict);

    ZSTD_resetSeqStore(seqStore);
    let shift = src.len() as u32;
    ms.window.base_offset = ms.window.base_offset.wrapping_add(shift);
    ms.window.nextSrc = ms.window.nextSrc.wrapping_add(shift);
    ms.window.dictLimit = ms.window.dictLimit.wrapping_add(shift);
    ms.window.lowLimit = ms.window.dictLimit;
    ms.nextToUpdate = ms.window.dictLimit;
}

/// Port of `ZSTD_compressBlock_btopt`.
pub fn ZSTD_compressBlock_btopt(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut crate::compress::seq_store::SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_opt0(ms, seqStore, rep, src, ZSTD_dictMode_e::ZSTD_noDict)
}

pub fn ZSTD_compressBlock_btopt_window(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut crate::compress::seq_store::SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    window_buf: &[u8],
    src_pos: usize,
    src_end: usize,
) -> usize {
    if src_pos == 0 {
        return ZSTD_compressBlock_btopt(ms, seqStore, rep, &window_buf[..src_end]);
    }
    ZSTD_compressBlock_opt_generic_window(
        ms,
        seqStore,
        rep,
        window_buf,
        src_pos,
        src_end,
        0,
        ZSTD_dictMode_e::ZSTD_noDict,
    )
}

/// Port of `ZSTD_compressBlock_btultra`.
pub fn ZSTD_compressBlock_btultra(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut crate::compress::seq_store::SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_opt2(ms, seqStore, rep, src, ZSTD_dictMode_e::ZSTD_noDict)
}

pub fn ZSTD_compressBlock_btultra_window(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut crate::compress::seq_store::SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    window_buf: &[u8],
    src_pos: usize,
    src_end: usize,
) -> usize {
    if src_pos == 0 {
        return ZSTD_compressBlock_btultra(ms, seqStore, rep, &window_buf[..src_end]);
    }
    ZSTD_compressBlock_opt_generic_window(
        ms,
        seqStore,
        rep,
        window_buf,
        src_pos,
        src_end,
        2,
        ZSTD_dictMode_e::ZSTD_noDict,
    )
}

/// Port of `ZSTD_compressBlock_btultra2`.
pub fn ZSTD_compressBlock_btultra2(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut crate::compress::seq_store::SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    let curr = ms.window.base_offset;
    if opt_state_is_fresh(ms, seqStore)
        && curr == ms.window.dictLimit
        && src.len() > ZSTD_PREDEF_THRESHOLD as usize
    {
        ZSTD_initStats_ultra(ms, seqStore, rep, src);
    }
    ZSTD_compressBlock_opt2(ms, seqStore, rep, src, ZSTD_dictMode_e::ZSTD_noDict)
}

pub fn ZSTD_compressBlock_btultra2_window(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut crate::compress::seq_store::SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    window_buf: &[u8],
    src_pos: usize,
    src_end: usize,
) -> usize {
    if src_pos == 0 {
        return ZSTD_compressBlock_btultra2(ms, seqStore, rep, &window_buf[..src_end]);
    }
    ZSTD_compressBlock_opt_generic_window(
        ms,
        seqStore,
        rep,
        window_buf,
        src_pos,
        src_end,
        2,
        ZSTD_dictMode_e::ZSTD_noDict,
    )
}

#[inline]
fn opt_state_is_fresh(
    ms: &ZSTD_MatchState_t,
    seqStore: &crate::compress::seq_store::SeqStore_t,
) -> bool {
    ms.opt.litLengthSum == 0
        && seqStore.sequences.is_empty()
        && ms.window.dictLimit == ms.window.lowLimit
        && ms.window.dictLimit.wrapping_sub(ms.nextToUpdate) <= 1
}

/// Port of `ZSTD_compressBlock_btopt_dictMatchState` (`zstd_opt.c:1539`).
/// Routes through the shared optimal parser with `dictMatchState`
/// mode selected.
pub fn ZSTD_compressBlock_btopt_dictMatchState(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut crate::compress::seq_store::SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_opt0(ms, seqStore, rep, src, ZSTD_dictMode_e::ZSTD_dictMatchState)
}

/// Port of `ZSTD_compressBlock_btopt_extDict` (`zstd_opt.c:1546`).
/// Routes through the shared optimal parser with `extDict` mode
/// selected.
pub fn ZSTD_compressBlock_btopt_extDict(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut crate::compress::seq_store::SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_opt0(ms, seqStore, rep, src, ZSTD_dictMode_e::ZSTD_extDict)
}

/// Port of `ZSTD_compressBlock_btultra_dictMatchState` (`zstd_opt.c:1555`).
/// `btultra` entry for the shared optimal parser with
/// `dictMatchState` mode selected.
pub fn ZSTD_compressBlock_btultra_dictMatchState(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut crate::compress::seq_store::SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_opt2(ms, seqStore, rep, src, ZSTD_dictMode_e::ZSTD_dictMatchState)
}

/// Port of `ZSTD_compressBlock_btultra_extDict` (`zstd_opt.c:1562`).
/// `btultra` entry for the shared optimal parser with `extDict` mode
/// selected.
pub fn ZSTD_compressBlock_btultra_extDict(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut crate::compress::seq_store::SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_opt2(ms, seqStore, rep, src, ZSTD_dictMode_e::ZSTD_extDict)
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn bitWeight_powers_of_two() {
        // highbit32(stat+1) * 256.
        // stat=0 → highbit32(1) = 0, weight 0
        // stat=1 → highbit32(2) = 1, weight 256
        // stat=3 → highbit32(4) = 2, weight 512
        // stat=7 → highbit32(8) = 3, weight 768
        assert_eq!(ZSTD_bitWeight(0), 0);
        assert_eq!(ZSTD_bitWeight(1), 256);
        assert_eq!(ZSTD_bitWeight(3), 512);
        assert_eq!(ZSTD_bitWeight(7), 768);
    }

    #[test]
    fn fracWeight_matches_bitWeight_on_powers_of_two_minus_one() {
        // For stat = (1<<k)-1, stat+1 = 1<<k, hb=k, Fweight =
        // ((1<<k) << 8) >> k = 256. So fracWeight = k*256 + 256
        // = bitWeight + 256.
        for k in 1..10 {
            let stat = (1u32 << k) - 1;
            assert_eq!(
                ZSTD_fracWeight(stat),
                ZSTD_bitWeight(stat).wrapping_add(256)
            );
        }
    }

    #[test]
    fn fracWeight_monotonic_on_range() {
        let mut last = 0u32;
        for s in 0..1_000u32 {
            let w = ZSTD_fracWeight(s);
            assert!(w >= last, "fracWeight not monotonic at {s}: {last} -> {w}");
            last = w;
        }
    }

    #[test]
    fn sum_u32_matches_fold() {
        let t: [u32; 5] = [10, 20, 30, 40, 0];
        assert_eq!(sum_u32(&t), 100);
        assert_eq!(sum_u32(&[]), 0);
    }

    #[test]
    fn downscaleStats_base1_floors_every_slot() {
        let mut t = vec![0u32, 4, 7, 16, 0];
        let sum = ZSTD_downscaleStats(&mut t, 4, 1, base_directive_e::base_1guaranteed);
        // Each becomes 1 + (x>>1): 1, 3, 4, 9, 1.
        assert_eq!(t, vec![1u32, 3, 4, 9, 1]);
        assert_eq!(
            sum,
            1u32.wrapping_add(3)
                .wrapping_add(4)
                .wrapping_add(9)
                .wrapping_add(1)
        );
    }

    #[test]
    fn downscaleStats_base0_keeps_zeros_zero() {
        let mut t = vec![0u32, 4, 7, 16, 0];
        let sum = ZSTD_downscaleStats(&mut t, 4, 1, base_directive_e::base_0possible);
        // 0→0, 4→1+2=3, 7→1+3=4, 16→1+8=9, 0→0.
        assert_eq!(t, vec![0u32, 3, 4, 9, 0]);
        assert_eq!(sum, 3u32.wrapping_add(4).wrapping_add(9));
    }

    fn dynamic_opt_state() -> optState_t {
        let mut s = optState_t::default();
        // Populate plausible frequencies so base prices aren't zero.
        s.litSum = 1024;
        s.litLengthSum = 256;
        s.matchLengthSum = 256;
        s.offCodeSum = 128;
        for f in s.litFreq.iter_mut() {
            *f = 4;
        }
        for f in s.litLengthFreq.iter_mut() {
            *f = 8;
        }
        for f in s.matchLengthFreq.iter_mut() {
            *f = 4;
        }
        for f in s.offCodeFreq.iter_mut() {
            *f = 4;
        }
        ZSTD_setBasePrices(&mut s, 2);
        s
    }

    #[test]
    fn rawLiteralsCost_zero_for_empty_run() {
        let s = dynamic_opt_state();
        assert_eq!(ZSTD_rawLiteralsCost(b"", 0, &s, 2), 0);
    }

    #[test]
    fn rawLiteralsCost_raw_path_is_8bit_per_byte() {
        let mut s = optState_t::default();
        s.literalCompressionMode = ZSTD_ParamSwitch_e::ZSTD_ps_disable;
        let n = 7u32;
        let got = ZSTD_rawLiteralsCost(&[0u8; 7], n, &s, 2);
        assert_eq!(got, (n << 3) * BITCOST_MULTIPLIER);
    }

    #[test]
    fn rawLiteralsCost_predef_is_6bit_per_byte() {
        let mut s = optState_t::default();
        s.priceType = ZSTD_OptPrice_e::zop_predef;
        let got = ZSTD_rawLiteralsCost(&[0u8; 5], 5, &s, 2);
        assert_eq!(got, 5 * 6 * BITCOST_MULTIPLIER);
    }

    #[test]
    fn rawLiteralsCost_dynamic_non_negative_and_bounded() {
        let s = dynamic_opt_state();
        let lits: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        let cost = ZSTD_rawLiteralsCost(&lits, 8, &s, 2);
        // Cost per byte bounded by base price.
        assert!(cost <= s.litSumBasePrice * 8);
    }

    #[test]
    fn litLengthPrice_predef_matches_WEIGHT() {
        let mut s = optState_t::default();
        s.priceType = ZSTD_OptPrice_e::zop_predef;
        assert_eq!(ZSTD_litLengthPrice(42, &s, 2), WEIGHT(42, 2));
        assert_eq!(ZSTD_litLengthPrice(0, &s, 0), WEIGHT(0, 0));
    }

    #[test]
    fn litLengthPrice_dynamic_in_LL_range() {
        let s = dynamic_opt_state();
        // Spot check a small LL code — expect finite non-zero cost.
        let p = ZSTD_litLengthPrice(10, &s, 2);
        assert!(p > 0);
        assert!((p as i32) < ZSTD_MAX_PRICE);
    }

    #[test]
    fn getMatchPrice_penalizes_long_offsets_at_opt0() {
        let s = dynamic_opt_state();
        // offBase with highbit32 = 20 (≥20 triggers handicap).
        let offBase = 1u32 << 20;
        let lo = ZSTD_getMatchPrice(offBase, 10, &s, 2);
        let hi_penalty = ZSTD_getMatchPrice(offBase, 10, &s, 0);
        assert!(hi_penalty > lo, "opt0 should penalize offCode>=20");
    }

    #[test]
    fn updateStats_increments_all_slots() {
        let mut s = optState_t::default();
        s.literalCompressionMode = ZSTD_ParamSwitch_e::ZSTD_ps_enable;
        let lits = [10u8, 20, 30, 40];
        ZSTD_updateStats(&mut s, 4, &lits, 1 << 5, 8);
        // Four literal bytes → each litFreq slot bumps by ZSTD_LITFREQ_ADD.
        assert_eq!(s.litFreq[10], ZSTD_LITFREQ_ADD);
        assert_eq!(s.litFreq[20], ZSTD_LITFREQ_ADD);
        assert_eq!(s.litSum, 4 * ZSTD_LITFREQ_ADD);
        // LL code for 4 = 4 (literal length → direct code).
        assert_eq!(s.litLengthFreq[ZSTD_LLcode(4) as usize], 1);
        assert_eq!(s.litLengthSum, 1);
        // Match length 8 → mlBase=5 → ZSTD_MLcode(5).
        assert_eq!(s.matchLengthFreq[ZSTD_MLcode(5) as usize], 1);
        assert_eq!(s.matchLengthSum, 1);
        // Off code = highbit32(32) = 5.
        assert_eq!(s.offCodeFreq[5], 1);
        assert_eq!(s.offCodeSum, 1);
    }

    #[test]
    fn updateStats_skips_literal_freqs_when_raw_lits() {
        let mut s = optState_t::default();
        s.literalCompressionMode = ZSTD_ParamSwitch_e::ZSTD_ps_disable;
        let lits = [10u8, 20];
        ZSTD_updateStats(&mut s, 2, &lits, 1 << 5, 8);
        assert_eq!(s.litFreq[10], 0);
        assert_eq!(s.litSum, 0);
        // But sequence-symbol slots still bump.
        assert_eq!(s.litLengthSum, 1);
        assert_eq!(s.offCodeSum, 1);
        assert_eq!(s.matchLengthSum, 1);
    }

    fn store_with_seqs(seqs: &[(u32, u32, u32)]) -> RawSeqStore_t {
        use crate::compress::zstd_ldm::rawSeq;
        let mut s = RawSeqStore_t::with_capacity(seqs.len());
        for (i, &(off, ll, ml)) in seqs.iter().enumerate() {
            s.seq[i] = rawSeq {
                offset: off,
                litLength: ll,
                matchLength: ml,
            };
        }
        s.size = seqs.len();
        s
    }

    #[test]
    fn optLdm_skip_through_full_seqs() {
        // Three seqs of total length 6, 10, 14.
        let mut s = store_with_seqs(&[(100, 2, 4), (200, 3, 7), (300, 5, 9)]);
        ZSTD_optLdm_skipRawSeqStoreBytes(&mut s, 6);
        // 2+4 exactly consumes seq 0.
        assert_eq!(s.pos, 1);
        assert_eq!(s.posInSequence, 0);
    }

    #[test]
    fn optLdm_skip_into_middle_of_seq() {
        let mut s = store_with_seqs(&[(100, 2, 4), (200, 3, 7)]);
        ZSTD_optLdm_skipRawSeqStoreBytes(&mut s, 4);
        // Consumed 2 lits + 2 match bytes of seq 0; still inside.
        assert_eq!(s.pos, 0);
        assert_eq!(s.posInSequence, 4);
    }

    #[test]
    fn optLdm_skip_past_end_clears_posInSequence() {
        let mut s = store_with_seqs(&[(100, 2, 4)]);
        ZSTD_optLdm_skipRawSeqStoreBytes(&mut s, 1000);
        assert_eq!(s.pos, 1);
        assert_eq!(s.posInSequence, 0);
    }

    #[test]
    fn optLdm_maybeAddMatch_ignores_out_of_range() {
        let mut matches = vec![ZSTD_match_t::default(); 16];
        let mut n = 0u32;
        let ldm = ZSTD_optLdm_t {
            startPosInBlock: 100,
            endPosInBlock: 150,
            offset: 42,
            ..Default::default()
        };
        // Before start — ignored.
        ZSTD_optLdm_maybeAddMatch(&mut matches, &mut n, &ldm, 50, 4);
        assert_eq!(n, 0);
        // After end — ignored.
        ZSTD_optLdm_maybeAddMatch(&mut matches, &mut n, &ldm, 200, 4);
        assert_eq!(n, 0);
    }

    #[test]
    fn optLdm_maybeAddMatch_inside_range_appends() {
        let mut matches = vec![ZSTD_match_t::default(); 16];
        let mut n = 0u32;
        let ldm = ZSTD_optLdm_t {
            startPosInBlock: 100,
            endPosInBlock: 200,
            offset: 42,
            ..Default::default()
        };
        ZSTD_optLdm_maybeAddMatch(&mut matches, &mut n, &ldm, 120, 4);
        assert_eq!(n, 1);
        // candidate match length = endPos - startPos - (120-100) = 80.
        assert_eq!(matches[0].len, 80);
        assert_eq!(matches[0].off, OFFSET_TO_OFFBASE(42));
    }

    #[test]
    fn optLdm_maybeAddMatch_rejects_short_candidate() {
        let mut matches = vec![ZSTD_match_t::default(); 16];
        let mut n = 0u32;
        let ldm = ZSTD_optLdm_t {
            startPosInBlock: 100,
            endPosInBlock: 103,
            offset: 42,
            ..Default::default()
        };
        // At pos 102, remaining = 1 byte — too short for minMatch=4.
        ZSTD_optLdm_maybeAddMatch(&mut matches, &mut n, &ldm, 102, 4);
        assert_eq!(n, 0);
    }

    #[test]
    fn opt_getNextMatch_empty_store_yields_UINT_MAX() {
        let mut optLdm = ZSTD_optLdm_t::default();
        ZSTD_opt_getNextMatchAndUpdateSeqStore(&mut optLdm, 0, 1024);
        assert_eq!(optLdm.startPosInBlock, u32::MAX);
        assert_eq!(optLdm.endPosInBlock, u32::MAX);
    }

    #[test]
    fn opt_getNextMatch_populates_positions() {
        // seq: 5 literals then 10-byte match at offset 42. At block
        // pos 100 with 200 bytes remaining: start=105, end=115.
        let mut optLdm = ZSTD_optLdm_t {
            seqStore: store_with_seqs(&[(42, 5, 10)]),
            ..Default::default()
        };
        ZSTD_opt_getNextMatchAndUpdateSeqStore(&mut optLdm, 100, 200);
        assert_eq!(optLdm.startPosInBlock, 105);
        assert_eq!(optLdm.endPosInBlock, 115);
        assert_eq!(optLdm.offset, 42);
    }

    #[test]
    fn opt_getNextMatch_clips_to_block_end() {
        // Match extends past block end → end clamped.
        let mut optLdm = ZSTD_optLdm_t {
            seqStore: store_with_seqs(&[(42, 5, 100)]),
            ..Default::default()
        };
        ZSTD_opt_getNextMatchAndUpdateSeqStore(&mut optLdm, 100, 50);
        // literalsBytesRemaining = 5, matchBytesRemaining = 100;
        // block ends at 150; start = 105; end would be 205 but
        // clipped to 150.
        assert_eq!(optLdm.startPosInBlock, 105);
        assert_eq!(optLdm.endPosInBlock, 150);
    }

    #[test]
    fn opt_getNextMatch_literals_overshoot_block() {
        // 1000 literals, 10-match; block only has 200 bytes remaining.
        let mut optLdm = ZSTD_optLdm_t {
            seqStore: store_with_seqs(&[(42, 1000, 10)]),
            ..Default::default()
        };
        ZSTD_opt_getNextMatchAndUpdateSeqStore(&mut optLdm, 0, 200);
        // Literal run alone exceeds block → UINT_MAX.
        assert_eq!(optLdm.startPosInBlock, u32::MAX);
        assert_eq!(optLdm.endPosInBlock, u32::MAX);
    }

    #[test]
    fn optLdm_processMatchCandidate_reloads_past_end() {
        let mut matches = vec![ZSTD_match_t::default(); 16];
        let mut n = 0u32;
        let mut optLdm = ZSTD_optLdm_t {
            seqStore: store_with_seqs(&[(42, 5, 10), (100, 2, 8)]),
            startPosInBlock: 0,
            endPosInBlock: 0, // already past end — forces a reload
            offset: 0,
        };
        ZSTD_optLdm_processMatchCandidate(&mut optLdm, &mut matches, &mut n, 0, 1000, 4);
        // Should have loaded seq 0: start=5, end=15, then tried to
        // add. At pos=0 we're before start=5 — no match appended.
        assert_eq!(optLdm.startPosInBlock, 5);
        assert_eq!(optLdm.endPosInBlock, 15);
        assert_eq!(n, 0);
    }

    #[test]
    fn updateTree_advances_nextToUpdate_to_target() {
        use crate::compress::match_state::ZSTD_compressionParameters;
        let cp = ZSTD_compressionParameters {
            windowLog: 17,
            chainLog: 14,
            hashLog: 12,
            searchLog: 4,
            minMatch: 4,
            targetLength: 32,
            strategy: 7,
        };
        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.chainTable = vec![0u32; 1 << cp.chainLog];
        ms.nextToUpdate = 10;
        ms.window.lowLimit = 4;
        let data: Vec<u8> = (0..1024u32)
            .map(|i| (i.wrapping_mul(37) & 0xFF) as u8)
            .collect();
        ZSTD_updateTree(&mut ms, &data, 500, data.len());
        assert_eq!(ms.nextToUpdate, 500);
    }

    #[test]
    fn updateTree_noop_when_already_at_target() {
        use crate::compress::match_state::ZSTD_compressionParameters;
        let mut ms = ZSTD_MatchState_t::new(ZSTD_compressionParameters {
            windowLog: 17,
            chainLog: 14,
            hashLog: 12,
            searchLog: 4,
            minMatch: 4,
            strategy: 7,
            ..Default::default()
        });
        ms.chainTable = vec![0u32; 1 << 14];
        ms.nextToUpdate = 200;
        ms.window.lowLimit = 4;
        let data: Vec<u8> = vec![0u8; 1024];
        ZSTD_updateTree(&mut ms, &data, 200, data.len());
        // Unchanged, chainTable untouched.
        assert_eq!(ms.nextToUpdate, 200);
        assert!(ms.chainTable.iter().all(|&x| x == 0));
    }

    #[test]
    fn fCost_converts_price_to_bytes() {
        // BITCOST_MULTIPLIER * 8 bits per byte.
        // price = 8 * 256 → 1 byte.
        let b = ZSTD_fCost(8 * BITCOST_MULTIPLIER as i32);
        assert!((b - 1.0).abs() < 1e-9);
    }

    #[test]
    fn fWeight_matches_fracWeight_scaled() {
        let stat = 123u32;
        let bits = ZSTD_fWeight(stat);
        let expected = ZSTD_fracWeight(stat) as f64 / BITCOST_MULTIPLIER as f64;
        assert!((bits - expected).abs() < 1e-9);
    }

    #[test]
    fn rescaleFreqs_first_block_seeds_priors() {
        let mut s = optState_t::default();
        let src: Vec<u8> = (0..200u32).map(|i| (i & 0xFF) as u8).collect();
        ZSTD_rescaleFreqs(&mut s, &src, src.len(), 2, None);
        // Baseline litLength sum: 4 + 2 + 1×34 = 40.
        assert_eq!(s.litLengthSum, 4u32.wrapping_add(2).wrapping_add(34));
        // Baseline matchLength: all 1s over 53 slots.
        assert_eq!(s.matchLengthSum, (MaxML + 1));
        // Literals populated from src.
        assert!(s.litSum > 0);
        // Predef path only for srcSize <= 8; 200 bytes uses dynamic.
        assert_eq!(s.priceType, ZSTD_OptPrice_e::zop_dynamic);
    }

    #[test]
    fn rescaleFreqs_small_block_flips_to_predef() {
        let mut s = optState_t::default();
        let src = b"hi";
        ZSTD_rescaleFreqs(&mut s, src, src.len(), 2, None);
        // srcSize 2 ≤ threshold 8 → predef prices.
        assert_eq!(s.priceType, ZSTD_OptPrice_e::zop_predef);
    }

    #[test]
    fn rescaleFreqs_subsequent_block_recomputes_sums_from_freqs() {
        let mut s = optState_t::default();
        // Seed as if previous block ran — subsequent branch recomputes
        // sums from the freq tables via ZSTD_scaleStats.
        s.litLengthSum = 1; // non-zero → subsequent-block branch
        for f in s.litFreq.iter_mut() {
            *f = 40;
        }
        for f in s.litLengthFreq.iter_mut() {
            *f = 30;
        }
        for f in s.matchLengthFreq.iter_mut() {
            *f = 10;
        }
        for f in s.offCodeFreq.iter_mut() {
            *f = 15;
        }
        ZSTD_rescaleFreqs(&mut s, b"whatever", 8, 2, None);
        // Sums match the sum of each freq table (scaleStats is no-op
        // at these magnitudes: 30×36 = 1080, 1080>>11 = 0 → keep).
        assert_eq!(s.litLengthSum, 30 * (MaxLL + 1));
        assert_eq!(s.matchLengthSum, 10 * (MaxML + 1));
        assert_eq!(s.offCodeSum, 15 * (MaxOff + 1));
    }

    #[test]
    fn rescaleFreqs_first_block_can_seed_from_dictionary_entropy() {
        use crate::compress::huf_compress::{HUF_setNbBits, HUF_writeCTableHeader};

        let mut s = optState_t::default();
        let src = b"zzzzzzzzzzzzzzzz";
        let mut entropy = ZSTD_entropyCTables_t::default();
        entropy.huf.repeatMode = HUF_repeat::HUF_repeat_valid;
        HUF_writeCTableHeader(&mut entropy.huf.CTable, 11, MaxLit);
        HUF_setNbBits(&mut entropy.huf.CTable[1 + b'z' as usize], 1);

        ZSTD_rescaleFreqs(&mut s, src, src.len(), 2, Some(&entropy));

        assert_eq!(s.priceType, ZSTD_OptPrice_e::zop_dynamic);
        assert_eq!(s.litFreq[b'z' as usize], 1 << 10);
        assert_eq!(s.litLengthSum, MaxLL.wrapping_add(1));
        assert_eq!(s.matchLengthSum, MaxML.wrapping_add(1));
        assert_eq!(s.offCodeSum, MaxOff.wrapping_add(1));
    }

    #[test]
    fn btGetAllMatches_noDict_returns_zero_if_already_updated() {
        use crate::compress::match_state::ZSTD_compressionParameters;
        let cp = ZSTD_compressionParameters {
            windowLog: 17,
            chainLog: 14,
            hashLog: 12,
            searchLog: 4,
            minMatch: 4,
            targetLength: 32,
            strategy: 7,
        };
        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.chainTable = vec![0u32; 1 << cp.chainLog];
        // Pretend we've already processed past ip_abs=50.
        ms.nextToUpdate = 100;
        ms.window.dictLimit = 4;
        ms.window.lowLimit = 4;
        let data = vec![0u8; 1024];
        let rep: [u32; ZSTD_REP_NUM] = [1, 4, 8];
        let mut matches = vec![ZSTD_match_t::default(); 16];
        let mut n3 = 0u32;
        let n = ZSTD_btGetAllMatches_noDict(
            &mut matches,
            &mut ms,
            &mut n3,
            &data,
            50,
            data.len(),
            &rep,
            0,
            3,
            4,
        );
        assert_eq!(n, 0);
    }

    #[test]
    fn insertBtAndGetAllMatches_finds_repcode_hit() {
        use crate::compress::match_state::ZSTD_compressionParameters;
        let cp = ZSTD_compressionParameters {
            windowLog: 17,
            chainLog: 14,
            hashLog: 12,
            searchLog: 4,
            minMatch: 4,
            targetLength: 32,
            strategy: 7,
        };
        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.chainTable = vec![0u32; 1 << cp.chainLog];
        ms.window.dictLimit = 4;
        ms.window.lowLimit = 4;

        // Build a buffer where a 4-byte rep match is reachable at
        // offset = rep[0]. Use a simple repeating payload.
        let data: Vec<u8> = b"hello hello hello hello hello".to_vec();
        let ip_abs = 12u32; // at "hello" (third occurrence)
        let rep: [u32; ZSTD_REP_NUM] = [6, 1, 8]; // rep[0]=6 → "hello " cycle
        let mut matches = vec![ZSTD_match_t::default(); 32];
        let mut nextToUpdate3 = 0u32;

        let n = ZSTD_insertBtAndGetAllMatches(
            &mut matches,
            &mut ms,
            &mut nextToUpdate3,
            &data,
            ip_abs,
            data.len(),
            ZSTD_dictMode_e::ZSTD_noDict,
            &rep,
            0,
            3,
            4,
        );
        // At least the repcode match should be recorded.
        assert!(n >= 1, "expected at least one match");
        assert!(matches[0].len >= 4);
    }

    #[test]
    fn insertBt1_basic_insertion_advances_hashTable() {
        use crate::compress::match_state::ZSTD_compressionParameters;
        let cp = ZSTD_compressionParameters {
            windowLog: 17,
            chainLog: 14,
            hashLog: 12,
            searchLog: 4,
            minMatch: 4,
            targetLength: 32,
            strategy: 7,
        };
        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.chainTable = vec![0u32; 1 << cp.chainLog];
        ms.window.lowLimit = 4; // must be > 0
                                // Position 100 should insert its 4-byte hash into the hash
                                // table. Pre-existing matchIndex is zero (no prior), so the
                                // walk exits immediately and insertBt1 returns positions ≥ 1.
        let data: Vec<u8> = (0..256u32).map(|i| (i & 0xFF) as u8).collect();
        let ip_abs = ms.window.base_offset.wrapping_add(100);
        let positions = ZSTD_insertBt1(&mut ms, &data, ip_abs, data.len(), ip_abs, 4, false);
        assert!(positions >= 1);
        // Hash table entry for ip=100 should now be 100.
        let h = crate::compress::zstd_hashes::ZSTD_hashPtr(&data[100..], cp.hashLog, 4);
        assert_eq!(ms.hashTable[h], ip_abs);
    }

    #[test]
    fn insertBt1_uses_predicted_small_shortcut() {
        use crate::compress::match_state::ZSTD_compressionParameters;
        let cp = ZSTD_compressionParameters {
            windowLog: 17,
            chainLog: 14,
            hashLog: 12,
            searchLog: 1,
            minMatch: 4,
            targetLength: 32,
            strategy: 7,
        };
        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.chainTable = vec![0u32; 1 << cp.chainLog];
        ms.window.lowLimit = 4;

        let data = vec![b'a'; 256];
        let curr = 100u32;
        let match_index = 50u32;
        let curr_slot = (2 * (curr & ((1u32 << (cp.chainLog - 1)) - 1))) as usize;
        let prev_slot = (2 * ((curr - 1) & ((1u32 << (cp.chainLog - 1)) - 1))) as usize;
        let next_base = (2 * (match_index & ((1u32 << (cp.chainLog - 1)) - 1))) as usize;
        let h = crate::compress::zstd_hashes::ZSTD_hashPtr(&data[curr as usize..], cp.hashLog, 4);

        ms.hashTable[h] = match_index;
        ms.chainTable[prev_slot] = match_index - 1; // predictedSmall becomes match_index
        ms.chainTable[next_base + 1] = 0;

        let positions = ZSTD_insertBt1(&mut ms, &data, curr, data.len(), curr, 4, false);
        assert!(positions >= 1);
        assert_eq!(ms.chainTable[curr_slot], match_index);
        assert_eq!(ms.hashTable[h], curr);
    }

    fn ms_with_hash3(hashLog3: u32) -> ZSTD_MatchState_t {
        use crate::compress::match_state::ZSTD_compressionParameters;
        let cp = ZSTD_compressionParameters {
            windowLog: 20,
            chainLog: 16,
            hashLog: 17,
            searchLog: 4,
            minMatch: 3,
            targetLength: 32,
            strategy: 7,
        };
        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.hashLog3 = hashLog3;
        ms.hashTable3 = vec![0u32; 1 << hashLog3];
        ms
    }

    #[test]
    fn insertAndFindFirstIndexHash3_fills_range_up_to_ip() {
        let mut ms = ms_with_hash3(10);
        let data: Vec<u8> = (0..256u32).map(|i| (i & 0xFF) as u8).collect();
        let mut n = ms.window.base_offset;
        let ip_abs = ms.window.base_offset.wrapping_add(100);
        let ret = ZSTD_insertAndFindFirstIndexHash3(&mut ms, &mut n, &data, ip_abs);
        // No prior state → returned slot was zero.
        assert_eq!(ret, 0);
        // Cursor advanced.
        assert_eq!(n, ip_abs);
        // At least one non-zero slot recorded (hash collisions aside,
        // 100 insertions over 1024 slots leave many).
        let nonzero = ms.hashTable3.iter().filter(|&&x| x != 0).count();
        assert!(nonzero > 20);
    }

    #[test]
    fn insertAndFindFirstIndexHash3_returns_latest_match_for_same_hash() {
        let mut ms = ms_with_hash3(10);
        // Repeat a 3-byte pattern: same bytes → same hash → the most
        // recent index should be returned.
        let pat = [0xAB, 0xCD, 0xEF, 0x00, 0xAB, 0xCD, 0xEF, 0x00];
        let mut n = ms.window.base_offset;
        let ip_abs = ms.window.base_offset.wrapping_add(4);
        // First insert: positions 0..=3 seen.
        ZSTD_insertAndFindFirstIndexHash3(&mut ms, &mut n, &pat, ip_abs);
        // Second call at pos 4 should find the prior position 0 (same
        // 3-byte prefix).
        let found = ZSTD_insertAndFindFirstIndexHash3(&mut ms, &mut n, &pat, ip_abs);
        assert_eq!(
            found, ms.window.base_offset,
            "prior position of same 3-byte hash"
        );
    }

    #[test]
    fn dms_ordering_byte_uses_prefix_after_crossing_dms_high_limit() {
        let prefix = b"aaaaZsuffix";
        let dms = b"aaaaY";

        assert_eq!(ZSTD_dmsOrderingByte(prefix, 0, dms, 0, 4, true), b'Z');
        assert_eq!(ZSTD_dmsOrderingByte(prefix, 0, dms, 0, 4, false), b'Y');
    }

    #[test]
    fn minmatch_equals_2segments_allows_dict_prefix_boundary_match() {
        let prefix = b"abcdWXYZ";
        let dict = b"ab";

        assert!(ZSTD_minMatchEquals_2segments(prefix, 0, 2, dict, 0, 4));
        assert!(!ZSTD_minMatchEquals_2segments(prefix, 1, 2, dict, 0, 4));
    }

    #[test]
    fn dms_index_delta_uses_upstream_wrapping_arithmetic() {
        let window_low = 5u32;
        let dms_high = 11u32;
        let dms_low = 3u32;

        let dms_index_delta = window_low.wrapping_sub(dms_high);
        assert_eq!(dms_index_delta, u32::MAX - 5);
        assert_eq!(dms_low.wrapping_add(dms_index_delta), u32::MAX - 2);
        assert_eq!(7u32.wrapping_add(dms_index_delta), 1);
    }

    #[test]
    fn compressedLiterals_toggles_with_literalCompressionMode() {
        let mut s = optState_t::default();
        // default: ZSTD_ps_auto → treated as enabled.
        assert!(ZSTD_compressedLiterals(&s));
        s.literalCompressionMode = ZSTD_ParamSwitch_e::ZSTD_ps_disable;
        assert!(!ZSTD_compressedLiterals(&s));
        s.literalCompressionMode = ZSTD_ParamSwitch_e::ZSTD_ps_enable;
        assert!(ZSTD_compressedLiterals(&s));
    }

    #[test]
    fn setBasePrices_fills_all_four_slots() {
        let mut s = optState_t::default();
        s.litSum = 100;
        s.litLengthSum = 64;
        s.matchLengthSum = 32;
        s.offCodeSum = 16;
        // optLevel=0 → WEIGHT == ZSTD_bitWeight.
        ZSTD_setBasePrices(&mut s, 0);
        assert_eq!(s.litSumBasePrice, ZSTD_bitWeight(100));
        assert_eq!(s.litLengthSumBasePrice, ZSTD_bitWeight(64));
        assert_eq!(s.matchLengthSumBasePrice, ZSTD_bitWeight(32));
        assert_eq!(s.offCodeSumBasePrice, ZSTD_bitWeight(16));
    }

    #[test]
    fn setBasePrices_uses_fracWeight_at_opt2() {
        let mut s = optState_t::default();
        s.litSum = 100;
        s.litLengthSum = 64;
        s.matchLengthSum = 32;
        s.offCodeSum = 16;
        ZSTD_setBasePrices(&mut s, 2);
        assert_eq!(s.litSumBasePrice, ZSTD_fracWeight(100));
        assert_eq!(s.litLengthSumBasePrice, ZSTD_fracWeight(64));
    }

    #[test]
    fn setBasePrices_skips_litSum_when_raw_literals() {
        let mut s = optState_t::default();
        s.literalCompressionMode = ZSTD_ParamSwitch_e::ZSTD_ps_disable;
        s.litSum = 100;
        s.litLengthSum = 64;
        ZSTD_setBasePrices(&mut s, 0);
        // lit slot untouched when raw; others still filled.
        assert_eq!(s.litSumBasePrice, 0);
        assert_eq!(s.litLengthSumBasePrice, ZSTD_bitWeight(64));
    }

    #[test]
    fn optState_default_sizes_tables() {
        let s = optState_t::default();
        assert_eq!(s.litFreq.len(), 256);
        assert_eq!(s.litLengthFreq.len(), 36);
        assert_eq!(s.matchLengthFreq.len(), 53);
        assert_eq!(s.offCodeFreq.len(), 32);
        assert_eq!(s.matchTable.len(), ZSTD_OPT_SIZE);
        assert_eq!(s.priceTable.len(), ZSTD_OPT_SIZE);
        assert_eq!(s.priceType, ZSTD_OptPrice_e::zop_dynamic);
    }

    #[test]
    fn scaleStats_noop_when_under_target() {
        // prevsum = 10, logTarget=4 → factor = 10>>4 = 0 → no-op, return 10.
        let mut t = vec![1u32, 2, 3, 4];
        let s = ZSTD_scaleStats(&mut t, 3, 4);
        assert_eq!(t, vec![1, 2, 3, 4]);
        assert_eq!(s, 10);
    }

    #[test]
    fn scaleStats_halves_when_over_target() {
        // Build a table whose sum = 4096. logTarget=10 → factor = 4,
        // highbit32(4)=2, so each slot becomes 1 + (slot >> 2).
        let mut t = vec![256u32, 512, 1024, 2304];
        let s = ZSTD_scaleStats(&mut t, 3, 10);
        // 1 + (256>>2)=65, 1 + (512>>2)=129, 1 + (1024>>2)=257, 1 + (2304>>2)=577
        assert_eq!(t, vec![65u32, 129, 257, 577]);
        assert_eq!(
            s,
            65u32.wrapping_add(129).wrapping_add(257).wrapping_add(577)
        );
    }

    #[test]
    fn readMINMATCH_length4_full_word() {
        let bytes = [0x11u8, 0x22, 0x33, 0x44];
        // Whatever endianness, length=4 must round-trip MEM_read32.
        assert_eq!(ZSTD_readMINMATCH(&bytes, 4), MEM_read32(&bytes));
    }

    #[test]
    fn readMINMATCH_length3_masks_fourth_byte() {
        // Two buffers that share the first three bytes should compare
        // equal under length=3 even if their fourth bytes differ.
        let a = [0x11u8, 0x22, 0x33, 0xFF];
        let b = [0x11u8, 0x22, 0x33, 0x00];
        assert_eq!(ZSTD_readMINMATCH(&a, 3), ZSTD_readMINMATCH(&b, 3));
        // But differ under length=4.
        assert_ne!(ZSTD_readMINMATCH(&a, 4), ZSTD_readMINMATCH(&b, 4));
    }

    #[test]
    fn downscaleStats_respects_lastEltIndex() {
        let mut t = vec![16u32, 16, 16, 16];
        ZSTD_downscaleStats(&mut t, 1, 2, base_directive_e::base_1guaranteed);
        // Only first 2 are scaled: 1+4=5, 1+4=5. Last two untouched.
        assert_eq!(t, vec![5u32, 5, 16, 16]);
    }

    #[test]
    fn compressBlock_btopt_emits_sequences_on_repetitive_input() {
        use crate::compress::match_state::ZSTD_compressionParameters;
        use crate::compress::seq_store::SeqStore_t;

        let cp = ZSTD_compressionParameters {
            windowLog: 18,
            chainLog: 15,
            hashLog: 14,
            searchLog: 4,
            minMatch: 4,
            targetLength: 32,
            strategy: 7,
        };
        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.chainTable = vec![0u32; 1 << cp.chainLog];
        ms.hashLog3 = 12;
        ms.hashTable3 = vec![0u32; 1 << ms.hashLog3];
        let mut seqStore = SeqStore_t::with_capacity(4096, ZSTD_BLOCKSIZE_MAX);
        let mut rep = [1u32, 4, 8];
        let src = b"abcdefghabcdefghabcdefghabcdefghabcdefghabcdefgh";

        let last = ZSTD_compressBlock_btopt(&mut ms, &mut seqStore, &mut rep, src);

        assert!(last < src.len(), "expected at least one match");
        assert!(!seqStore.sequences.is_empty(), "expected emitted sequences");
        assert!(
            !seqStore.literals.is_empty(),
            "expected some leading literals"
        );
    }

    #[test]
    fn compressBlock_btopt_clamps_minmatch_above_6_for_bt_matchfinder() {
        use crate::compress::match_state::ZSTD_compressionParameters;
        use crate::compress::seq_store::SeqStore_t;

        let cp = ZSTD_compressionParameters {
            windowLog: 18,
            chainLog: 15,
            hashLog: 14,
            searchLog: 4,
            minMatch: 8,
            targetLength: 32,
            strategy: 7,
        };
        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.chainTable = vec![0u32; 1 << cp.chainLog];
        ms.hashLog3 = 12;
        ms.hashTable3 = vec![0u32; 1 << ms.hashLog3];
        let mut seqStore = SeqStore_t::with_capacity(4096, ZSTD_BLOCKSIZE_MAX);
        let mut rep = [1u32, 4, 8];
        let src = b"abcdefghabcdefghabcdefghabcdefghabcdefghabcdefgh";

        let last = ZSTD_compressBlock_btopt(&mut ms, &mut seqStore, &mut rep, src);

        assert!(last < src.len(), "expected at least one match");
        assert!(
            !seqStore.sequences.is_empty(),
            "expected the bounded bt matcher to emit sequences"
        );
    }

    #[test]
    fn compressBlock_btopt_uses_match_state_ldm_sequences_as_candidates() {
        use crate::compress::match_state::ZSTD_compressionParameters;
        use crate::compress::seq_store::{SeqStore_t, OFFSET_TO_OFFBASE};
        use crate::compress::zstd_ldm::{rawSeq, RawSeqStore_t};

        let cp = ZSTD_compressionParameters {
            windowLog: 18,
            chainLog: 15,
            hashLog: 14,
            searchLog: 4,
            minMatch: 4,
            targetLength: 32,
            strategy: 7,
        };
        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.chainTable = vec![0u32; 1 << cp.chainLog];
        ms.hashLog3 = 12;
        ms.hashTable3 = vec![0u32; 1 << ms.hashLog3];

        let mut ldm = RawSeqStore_t::with_capacity(2);
        ldm.seq[0] = rawSeq {
            litLength: 4,
            matchLength: 24,
            offset: 4,
        };
        ldm.seq[1] = rawSeq {
            litLength: 128,
            matchLength: 24,
            offset: 4,
        };
        ldm.size = 2;
        ms.ldmSeqStore = Some(ldm);

        let mut seqStore = SeqStore_t::with_capacity(4096, ZSTD_BLOCKSIZE_MAX);
        let mut rep = [1u32, 4, 8];
        let src = b"abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

        let last = ZSTD_compressBlock_btopt(&mut ms, &mut seqStore, &mut rep, src);

        assert!(
            last < src.len(),
            "expected the LDM candidate to be selected"
        );
        assert!(
            seqStore
                .sequences
                .iter()
                .any(|seq| seq.offBase == OFFSET_TO_OFFBASE(4) && seq.mlBase as usize + 4 >= 24),
            "expected opt parser to emit the injected LDM candidate"
        );
    }

    #[test]
    fn dict_and_ext_wrappers_route_through_live_opt_parser_entries() {
        use crate::compress::match_state::ZSTD_compressionParameters;
        use crate::compress::seq_store::SeqStore_t;

        fn build_ms(strategy: u32) -> ZSTD_MatchState_t {
            let cp = ZSTD_compressionParameters {
                windowLog: 18,
                chainLog: 15,
                hashLog: 14,
                searchLog: 4,
                minMatch: 4,
                targetLength: 32,
                strategy,
            };
            let mut ms = ZSTD_MatchState_t::new(cp);
            ms.chainTable = vec![0u32; 1 << cp.chainLog];
            ms.hashLog3 = 12;
            ms.hashTable3 = vec![0u32; 1 << ms.hashLog3];
            ms
        }

        let src = b"abcdefghabcdefghabcdefghabcdefghabcdefghabcdefgh";
        let variants: [fn(
            &mut ZSTD_MatchState_t,
            &mut SeqStore_t,
            &mut [u32; ZSTD_REP_NUM],
            &[u8],
        ) -> usize; 4] = [
            ZSTD_compressBlock_btopt_dictMatchState,
            ZSTD_compressBlock_btopt_extDict,
            ZSTD_compressBlock_btultra_dictMatchState,
            ZSTD_compressBlock_btultra_extDict,
        ];

        for (idx, f) in variants.into_iter().enumerate() {
            let strategy = if idx < 2 { 7 } else { 8 };
            let mut ms = build_ms(strategy);
            let mut seqStore = SeqStore_t::with_capacity(4096, ZSTD_BLOCKSIZE_MAX);
            let mut rep = [1u32, 4, 8];
            let last = f(&mut ms, &mut seqStore, &mut rep, src);
            assert!(last < src.len(), "variant {idx} emitted only literals");
            assert!(
                !seqStore.sequences.is_empty(),
                "variant {idx} regressed into a dead wrapper"
            );
        }
    }

    #[test]
    fn opt_state_is_fresh_requires_zero_litlengthsum_and_nearby_next_to_update() {
        use crate::compress::match_state::ZSTD_compressionParameters;
        use crate::compress::seq_store::SeqStore_t;

        let cp = ZSTD_compressionParameters {
            windowLog: 18,
            chainLog: 15,
            hashLog: 14,
            searchLog: 4,
            minMatch: 4,
            targetLength: 32,
            strategy: 9,
        };
        let mut ms = ZSTD_MatchState_t::new(cp);
        let seq_store = SeqStore_t::with_capacity(16, 256);

        ms.window.dictLimit = 1000;
        ms.window.lowLimit = 1000;
        ms.nextToUpdate = 999;
        ms.opt.litLengthSum = 0;
        assert!(opt_state_is_fresh(&ms, &seq_store));

        ms.opt.litLengthSum = 1;
        assert!(!opt_state_is_fresh(&ms, &seq_store));

        ms.opt.litLengthSum = 0;
        ms.nextToUpdate = 997;
        assert!(!opt_state_is_fresh(&ms, &seq_store));
    }

    #[test]
    fn init_stats_ultra_resets_seqstore_and_shifts_window_for_second_pass() {
        use crate::compress::match_state::ZSTD_compressionParameters;
        use crate::compress::seq_store::SeqStore_t;

        let cp = ZSTD_compressionParameters {
            windowLog: 18,
            chainLog: 15,
            hashLog: 14,
            searchLog: 4,
            minMatch: 4,
            targetLength: 32,
            strategy: 9,
        };
        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.chainTable = vec![0u32; 1 << cp.chainLog];
        ms.hashLog3 = 12;
        ms.hashTable3 = vec![0u32; 1 << ms.hashLog3];
        ms.window.base_offset = crate::compress::match_state::ZSTD_WINDOW_START_INDEX;
        ms.window.dictLimit = crate::compress::match_state::ZSTD_WINDOW_START_INDEX;
        ms.window.lowLimit = crate::compress::match_state::ZSTD_WINDOW_START_INDEX;
        ms.window.nextSrc = crate::compress::match_state::ZSTD_WINDOW_START_INDEX;
        ms.nextToUpdate = crate::compress::match_state::ZSTD_WINDOW_START_INDEX;

        let mut seq_store = SeqStore_t::with_capacity(4096, ZSTD_BLOCKSIZE_MAX);
        let mut rep = [1u32, 4, 8];
        let src = b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

        ZSTD_initStats_ultra(&mut ms, &mut seq_store, &mut rep, src);

        assert!(seq_store.sequences.is_empty());
        let shifted =
            (src.len() as u32).wrapping_add(crate::compress::match_state::ZSTD_WINDOW_START_INDEX);
        assert_eq!(ms.window.dictLimit, shifted);
        assert_eq!(ms.window.lowLimit, shifted);
        assert_eq!(ms.window.nextSrc, shifted);
        assert_eq!(ms.nextToUpdate, shifted);
        assert!(ms.opt.litLengthSum > 0);
    }
}
