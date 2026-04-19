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
//! (`ZSTD_insertBtAndGetAllMatches_prefixOnly`,
//! `ZSTD_btGetAllMatches_noDict`); optLdm helpers
//! (`ZSTD_optLdm_skipRawSeqStoreBytes`, `maybeAddMatch`,
//! `getNextMatchAndUpdateSeqStore`, `processMatchCandidate`).
//!
//! **Deferred**: the forward-DP price-based parser
//! (`ZSTD_compressBlock_opt_generic`), dict-seeded `rescaleFreqs`
//! branch, ext-dict / dictMatchState branches of the match-gatherer,
//! `ZSTD_initStats_ultra` 2-pass seeding. Current v0.1 clamps
//! strategies 7–9 down to `btlazy2` / `lazy2` at the `ZSTD_compress`
//! dispatcher so `compressBlock_bt*` here still return a clean error.

#![allow(non_snake_case)]

use crate::common::bits::ZSTD_highbit32;
use crate::common::mem::{MEM_isLittleEndian, MEM_read32};
use crate::compress::match_state::ZSTD_MatchState_t;
use crate::compress::seq_store::{MINMATCH, OFFSET_TO_OFFBASE};
use crate::compress::zstd_ldm::RawSeqStore_t;
use crate::compress::zstd_compress::{ZSTD_LLcode, ZSTD_MLcode};
use crate::compress::zstd_fast::ZSTD_getLowestMatchIndex;
use crate::compress::zstd_hashes::{ZSTD_count, ZSTD_hash3Ptr, ZSTD_hashPtr};
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
/// `symbolCosts` (upstream's `ZSTD_entropyCTables_t*`, used by the
/// dict-seeded `rescaleFreqs` path) is deferred until the entropy
/// table types are surfaced here.
#[derive(Debug)]
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
    if optLevel >= 2 {
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
    debug_assert!(hashLog3 > 0);
    // Hash of the current position — read from the input slice.
    let hash3 = ZSTD_hash3Ptr(&window_buf[ip_abs as usize..], hashLog3);

    let mut idx = *nextToUpdate3;
    while idx < ip_abs {
        let h = ZSTD_hash3Ptr(&window_buf[idx as usize..], hashLog3);
        ms.hashTable3[h] = idx;
        idx += 1;
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

/// Port of `ZSTD_optLdm_skipRawSeqStoreBytes`. Advances both `pos`
/// and `posInSequence` forward by `nbBytes`, popping full sequences
/// as they're consumed. Symmetric to LDM's `skipSequences` but treats
/// `litLength + matchLength` as a single block rather than partial
/// match/lit residuals.
pub fn ZSTD_optLdm_skipRawSeqStoreBytes(rawSeqStore: &mut RawSeqStore_t, nbBytes: usize) {
    let mut currPos = rawSeqStore.posInSequence as u32 + nbBytes as u32;
    while currPos > 0 && rawSeqStore.pos < rawSeqStore.size {
        let currSeq = rawSeqStore.seq[rawSeqStore.pos];
        let fullLen = currSeq.litLength + currSeq.matchLength;
        if currPos >= fullLen {
            currPos -= fullLen;
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
        optLdm.seqStore.posInSequence <= (currSeq.litLength + currSeq.matchLength) as usize
    );
    let currBlockEndPos = currPosInBlock + blockBytesRemaining;
    let literalsBytesRemaining = currSeq
        .litLength
        .saturating_sub(optLdm.seqStore.posInSequence as u32);
    let matchBytesRemaining = if literalsBytesRemaining == 0 {
        currSeq.matchLength - (optLdm.seqStore.posInSequence as u32 - currSeq.litLength)
    } else {
        currSeq.matchLength
    };

    if literalsBytesRemaining >= blockBytesRemaining {
        optLdm.startPosInBlock = u32::MAX;
        optLdm.endPosInBlock = u32::MAX;
        ZSTD_optLdm_skipRawSeqStoreBytes(&mut optLdm.seqStore, blockBytesRemaining as usize);
        return;
    }

    optLdm.startPosInBlock = currPosInBlock + literalsBytesRemaining;
    optLdm.endPosInBlock = optLdm.startPosInBlock + matchBytesRemaining;
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
            (literalsBytesRemaining + matchBytesRemaining) as usize,
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

/// Port of `ZSTD_insertBt1` — prefix-only variant.
///
/// Inserts position `curr` (= `ip_abs`) into the match-state's binary
/// search tree, re-sorting the tree by common-prefix length along the
/// walk. Returns the number of positions covered (upstream speedhack:
/// when we find a very long match, we can skip inserting the next
/// few positions since their substring is already reachable via the
/// long match).
///
/// The ext-dict branch of upstream (`extDict != 0`) is deferred —
/// callers currently only exercise the prefix-only path. Upstream's
/// `ZSTD_C_PREDICT` optimization block is also skipped (it's gated
/// behind a build-time define that's rarely turned on).
pub fn ZSTD_insertBt1(
    ms: &mut ZSTD_MatchState_t,
    buf: &[u8],
    ip_abs: u32,
    iend_pos: usize,
    target: u32,
    mls: u32,
) -> u32 {
    let hashLog = ms.cParams.hashLog;
    let h = ZSTD_hashPtr(&buf[ip_abs as usize..], hashLog, mls);
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
    let mut matchEndIdx: u32 = curr + 8 + 1;
    let mut bestLength: usize = 8;
    let mut nbCompares = 1u32 << ms.cParams.searchLog;

    // Update hash table to point at `curr`.
    let mut matchIndex = ms.hashTable[h];
    ms.hashTable[h] = curr;

    debug_assert!(windowLow > 0);
    while nbCompares > 0 && matchIndex >= windowLow {
        let next_base = (2 * (matchIndex & btMask)) as usize;
        let mut matchLength = commonLengthSmaller.min(commonLengthLarger);
        debug_assert!(matchIndex < curr);

        // Prefix-only: no extDict branch.
        let match_pos = matchIndex as usize;
        matchLength += ZSTD_count(
            buf,
            ip_abs as usize + matchLength,
            match_pos + matchLength,
            iend_pos,
        );

        if matchLength > bestLength {
            bestLength = matchLength;
            if matchLength as u32 > matchEndIdx - matchIndex {
                matchEndIdx = matchIndex + matchLength as u32;
            }
        }

        if ip_abs as usize + matchLength == iend_pos {
            // Equal to end → can't tell inf/sup; stop to preserve
            // tree consistency.
            break;
        }

        // Compare the byte after the common prefix to decide direction.
        if buf[match_pos + matchLength] < buf[ip_abs as usize + matchLength] {
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
    debug_assert!(matchEndIdx > curr + 8);
    positions.max(matchEndIdx - (curr + 8))
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

/// Port of `ZSTD_rescaleFreqs` — no-dict variant. Initializes the
/// per-block frequency tables:
///   * First block (`litLengthSum == 0`):
///       - literals: counted directly from `src` via `HIST_count_simple`,
///         then downscaled so accumulator stays bounded.
///       - litLength / matchLength / offCode: baseline priors
///         (`BASE_LL_FREQS`, all-1s for ML, `BASE_OF_FREQS`).
///       - Predef price path for very small inputs.
///   * Subsequent blocks: scale existing accumulators down by
///     `ZSTD_scaleStats` so price drift stays bounded.
///
/// After either path, `ZSTD_setBasePrices` recomputes the per-symbol
/// floors.
///
/// Upstream's dict-seeded branch (HUF CTable + FSE CTable costs) is
/// deferred — our callers currently only exercise the no-dict path.
pub fn ZSTD_rescaleFreqs(
    optPtr: &mut optState_t,
    src: &[u8],
    srcSize: usize,
    optLevel: i32,
) {
    optPtr.priceType = ZSTD_OptPrice_e::zop_dynamic;
    let compressedLiterals = ZSTD_compressedLiterals(optPtr);

    if optPtr.litLengthSum == 0 {
        // First block — seed from baseline priors.
        if srcSize <= ZSTD_PREDEF_THRESHOLD as usize {
            optPtr.priceType = ZSTD_OptPrice_e::zop_predef;
        }

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

/// Port of `ZSTD_insertBtAndGetAllMatches` — prefix-only (no extDict,
/// no dictMatchState) variant.
///
/// Walks repcodes, the 3-byte hash (when `mls == 3`), and the binary
/// tree rooted at the current position. Any match longer than
/// `lengthToBeat - 1` is appended to `matches` (kept in ascending
/// length order — the tree walk discovers them that way). Returns
/// the number of matches written.
///
/// Upstream's ext-dict + dictMatchState branches are deferred here;
/// callers currently only exercise the no-dict path (btopt/btultra
/// with plain frames).
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_insertBtAndGetAllMatches_prefixOnly(
    matches: &mut [ZSTD_match_t],
    ms: &mut ZSTD_MatchState_t,
    nextToUpdate3: &mut u32,
    buf: &[u8],
    ip_abs: u32,
    ilimit_pos: usize,
    rep: &[u32; ZSTD_REP_NUM],
    ll0: u32,
    lengthToBeat: u32,
    mls: u32,
) -> u32 {
    use crate::common::bits::ZSTD_highbit32;
    use crate::compress::match_state::ZSTD_index_overlap_check;
    use crate::compress::seq_store::{OFFSET_TO_OFFBASE, REPCODE_TO_OFFBASE};

    let sufficient_len = ms.cParams.targetLength.min(ZSTD_OPT_NUM as u32 - 1);
    let curr = ip_abs;
    let hashLog = ms.cParams.hashLog;
    let minMatch: u32 = if mls == 3 { 3 } else { 4 };
    let h = ZSTD_hashPtr(&buf[ip_abs as usize..], hashLog, mls);
    let btLog = ms.cParams.chainLog - 1;
    let btMask: u32 = (1u32 << btLog) - 1;
    let mut commonLengthSmaller = 0usize;
    let mut commonLengthLarger = 0usize;
    let dictLimit = ms.window.dictLimit;
    let btLow: u32 = curr.saturating_sub(btMask);
    let windowLow = ZSTD_getLowestMatchIndex(ms, curr, ms.cParams.windowLog);
    let matchLow = if windowLow == 0 { 1 } else { windowLow };

    let mut smaller_slot: Option<usize> = Some((2 * (curr & btMask)) as usize);
    let mut larger_slot: Option<usize> = Some((2 * (curr & btMask)) as usize + 1);
    let mut matchEndIdx: u32 = curr + 8 + 1;
    let mut mnum: u32 = 0;
    let mut nbCompares: u32 = 1u32 << ms.cParams.searchLog;
    let mut bestLength: usize = (lengthToBeat - 1) as usize;

    let ip_pos = ip_abs as usize;

    // ---- 1. Rep codes ----
    debug_assert!(ll0 <= 1);
    let lastR = ZSTD_REP_NUM as u32 + ll0;
    for repCode in ll0..lastR {
        let repOffset = if repCode == ZSTD_REP_NUM as u32 {
            rep[0] - 1
        } else {
            rep[repCode as usize]
        };
        if repOffset == 0 {
            continue;
        }
        // `curr > repIndex >= dictLimit` equivalent.
        if repOffset.wrapping_sub(1) < curr - dictLimit {
            let repIndex = curr - repOffset;
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
        }
    }

    // Silence unused-import warnings if the branches don't fire.
    let _ = ZSTD_index_overlap_check;
    let _ = ZSTD_highbit32;

    // ---- 2. Hash3 lookup (only when mls == 3) ----
    if mls == 3 && (bestLength as u32) < mls {
        let matchIndex3 = ZSTD_insertAndFindFirstIndexHash3(ms, nextToUpdate3, buf, ip_abs);
        if matchIndex3 >= matchLow && curr - matchIndex3 < (1u32 << 18) {
            let match_pos = matchIndex3 as usize;
            let mlen = ZSTD_count(buf, ip_pos, match_pos, ilimit_pos);
            if mlen >= mls as usize {
                bestLength = mlen;
                matches[0].off = OFFSET_TO_OFFBASE(curr - matchIndex3);
                matches[0].len = mlen as u32;
                mnum = 1;
                if mlen as u32 > sufficient_len || ip_pos + mlen == ilimit_pos {
                    ms.nextToUpdate = curr + 1;
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

        // Prefix-only: no extDict branch.
        let match_pos = matchIndex as usize;
        matchLength += ZSTD_count(buf, ip_pos + matchLength, match_pos + matchLength, ilimit_pos);

        if matchLength > bestLength {
            if matchLength as u32 > matchEndIdx - matchIndex {
                matchEndIdx = matchIndex + matchLength as u32;
            }
            bestLength = matchLength;
            matches[mnum as usize].off = OFFSET_TO_OFFBASE(curr - matchIndex);
            matches[mnum as usize].len = matchLength as u32;
            mnum += 1;
            if matchLength as u32 > ZSTD_OPT_NUM as u32 || ip_pos + matchLength == ilimit_pos {
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

    debug_assert!(matchEndIdx > curr + 8);
    ms.nextToUpdate = matchEndIdx - 8;
    mnum
}

/// Port of `ZSTD_btGetAllMatches_internal` — no-dict variant. Thin
/// wrapper: updates the binary tree up to `ip_abs`, then runs
/// `ZSTD_insertBtAndGetAllMatches_prefixOnly`. Returns the match
/// count (0 when `ip_abs` is already covered by prior tree updates).
#[allow(clippy::too_many_arguments)]
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
    debug_assert!((3..=6).contains(&mls));
    if ip_abs < ms.nextToUpdate {
        return 0;
    }
    ZSTD_updateTree_internal(ms, buf, ihigh_pos, ip_abs, mls);
    ZSTD_insertBtAndGetAllMatches_prefixOnly(
        matches, ms, nextToUpdate3, buf, ip_abs, ihigh_pos,
        rep, ll0, lengthToBeat, mls,
    )
}

/// Port of `ZSTD_updateTree_internal` — prefix-only (no extDict, no
/// dictMatchState) variant. Walks the tree from `ms.nextToUpdate` up
/// to `target`, calling `ZSTD_insertBt1` at each position.
///
/// `buf` spans the window; `iend_pos` is the end offset used for
/// match-length counting (`ZSTD_count` clamps at it).
pub fn ZSTD_updateTree_internal(
    ms: &mut ZSTD_MatchState_t,
    buf: &[u8],
    iend_pos: usize,
    target: u32,
    mls: u32,
) {
    let mut idx = ms.nextToUpdate;
    while idx < target {
        let forward = ZSTD_insertBt1(ms, buf, idx, iend_pos, target, mls);
        debug_assert!(forward > 0, "insertBt1 must cover at least one position");
        idx += forward;
    }
    ms.nextToUpdate = target;
}

/// Port of `ZSTD_updateTree`. Thin wrapper over `_internal` using
/// `ms.cParams.minMatch` and the no-dict code path.
pub fn ZSTD_updateTree(
    ms: &mut ZSTD_MatchState_t,
    buf: &[u8],
    ip_abs: u32,
    iend_pos: usize,
) {
    let mls = ms.cParams.minMatch;
    ZSTD_updateTree_internal(ms, buf, iend_pos, ip_abs, mls);
}

/// Port of `ZSTD_compressBlock_btopt`. Returns `ErrorCode::Generic`
/// until the full optimal parser wiring lands. Typed signature in
/// place so callers get real compile-time shape checking.
pub fn ZSTD_compressBlock_btopt(
    _ms: &mut ZSTD_MatchState_t,
    _seqStore: &mut crate::compress::seq_store::SeqStore_t,
    _rep: &mut [u32; ZSTD_REP_NUM],
    _src: &[u8],
) -> usize {
    crate::common::error::ERROR(crate::common::error::ErrorCode::Generic)
}

/// Port of `ZSTD_compressBlock_btultra`. Same error-until-ported pattern.
pub fn ZSTD_compressBlock_btultra(
    _ms: &mut ZSTD_MatchState_t,
    _seqStore: &mut crate::compress::seq_store::SeqStore_t,
    _rep: &mut [u32; ZSTD_REP_NUM],
    _src: &[u8],
) -> usize {
    crate::common::error::ERROR(crate::common::error::ErrorCode::Generic)
}

/// Port of `ZSTD_compressBlock_btultra2`. Same error-until-ported pattern.
pub fn ZSTD_compressBlock_btultra2(
    _ms: &mut ZSTD_MatchState_t,
    _seqStore: &mut crate::compress::seq_store::SeqStore_t,
    _rep: &mut [u32; ZSTD_REP_NUM],
    _src: &[u8],
) -> usize {
    crate::common::error::ERROR(crate::common::error::ErrorCode::Generic)
}

// ─── NOT YET PORTED: bt{opt,ultra}_{dictMatchState,extDict} variants.
// These would require the forward-DP optimal parser (currently also
// unported — base bt{opt,ultra,ultra2} return `Generic`) AND the
// dictMatchState / extDict linkage on `ZSTD_MatchState_t`. Returning
// `ErrorCode::Generic` so callers fail loudly. ───────────────────────

/// Port of `ZSTD_compressBlock_btopt_dictMatchState` (`zstd_opt.c:1539`).
/// **NOT YET PORTED** — returns `ErrorCode::Generic`.
pub fn ZSTD_compressBlock_btopt_dictMatchState(
    _ms: &mut ZSTD_MatchState_t,
    _seqStore: &mut crate::compress::seq_store::SeqStore_t,
    _rep: &mut [u32; ZSTD_REP_NUM],
    _src: &[u8],
) -> usize {
    crate::common::error::ERROR(crate::common::error::ErrorCode::Generic)
}

/// Port of `ZSTD_compressBlock_btopt_extDict` (`zstd_opt.c:1546`).
/// **NOT YET PORTED** — returns `ErrorCode::Generic`.
pub fn ZSTD_compressBlock_btopt_extDict(
    _ms: &mut ZSTD_MatchState_t,
    _seqStore: &mut crate::compress::seq_store::SeqStore_t,
    _rep: &mut [u32; ZSTD_REP_NUM],
    _src: &[u8],
) -> usize {
    crate::common::error::ERROR(crate::common::error::ErrorCode::Generic)
}

/// Port of `ZSTD_compressBlock_btultra_dictMatchState` (`zstd_opt.c:1555`).
/// **NOT YET PORTED** — returns `ErrorCode::Generic`.
pub fn ZSTD_compressBlock_btultra_dictMatchState(
    _ms: &mut ZSTD_MatchState_t,
    _seqStore: &mut crate::compress::seq_store::SeqStore_t,
    _rep: &mut [u32; ZSTD_REP_NUM],
    _src: &[u8],
) -> usize {
    crate::common::error::ERROR(crate::common::error::ErrorCode::Generic)
}

/// Port of `ZSTD_compressBlock_btultra_extDict` (`zstd_opt.c:1562`).
/// **NOT YET PORTED** — returns `ErrorCode::Generic`.
pub fn ZSTD_compressBlock_btultra_extDict(
    _ms: &mut ZSTD_MatchState_t,
    _seqStore: &mut crate::compress::seq_store::SeqStore_t,
    _rep: &mut [u32; ZSTD_REP_NUM],
    _src: &[u8],
) -> usize {
    crate::common::error::ERROR(crate::common::error::ErrorCode::Generic)
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
            assert_eq!(ZSTD_fracWeight(stat), ZSTD_bitWeight(stat) + 256);
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
        assert_eq!(sum, 1 + 3 + 4 + 9 + 1);
    }

    #[test]
    fn downscaleStats_base0_keeps_zeros_zero() {
        let mut t = vec![0u32, 4, 7, 16, 0];
        let sum = ZSTD_downscaleStats(&mut t, 4, 1, base_directive_e::base_0possible);
        // 0→0, 4→1+2=3, 7→1+3=4, 16→1+8=9, 0→0.
        assert_eq!(t, vec![0u32, 3, 4, 9, 0]);
        assert_eq!(sum, 3 + 4 + 9);
    }

    fn dynamic_opt_state() -> optState_t {
        let mut s = optState_t::default();
        // Populate plausible frequencies so base prices aren't zero.
        s.litSum = 1024;
        s.litLengthSum = 256;
        s.matchLengthSum = 256;
        s.offCodeSum = 128;
        for f in s.litFreq.iter_mut() { *f = 4; }
        for f in s.litLengthFreq.iter_mut() { *f = 8; }
        for f in s.matchLengthFreq.iter_mut() { *f = 4; }
        for f in s.offCodeFreq.iter_mut() { *f = 4; }
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
            s.seq[i] = rawSeq { offset: off, litLength: ll, matchLength: ml };
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
        let data: Vec<u8> = (0..1024u32).map(|i| (i.wrapping_mul(37) & 0xFF) as u8).collect();
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
        ZSTD_rescaleFreqs(&mut s, &src, src.len(), 2);
        // Baseline litLength sum: 4 + 2 + 1×34 = 40.
        assert_eq!(s.litLengthSum, 4 + 2 + 34);
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
        ZSTD_rescaleFreqs(&mut s, src, src.len(), 2);
        // srcSize 2 ≤ threshold 8 → predef prices.
        assert_eq!(s.priceType, ZSTD_OptPrice_e::zop_predef);
    }

    #[test]
    fn rescaleFreqs_subsequent_block_recomputes_sums_from_freqs() {
        let mut s = optState_t::default();
        // Seed as if previous block ran — subsequent branch recomputes
        // sums from the freq tables via ZSTD_scaleStats.
        s.litLengthSum = 1; // non-zero → subsequent-block branch
        for f in s.litFreq.iter_mut() { *f = 40; }
        for f in s.litLengthFreq.iter_mut() { *f = 30; }
        for f in s.matchLengthFreq.iter_mut() { *f = 10; }
        for f in s.offCodeFreq.iter_mut() { *f = 15; }
        ZSTD_rescaleFreqs(&mut s, b"whatever", 8, 2);
        // Sums match the sum of each freq table (scaleStats is no-op
        // at these magnitudes: 30×36 = 1080, 1080>>11 = 0 → keep).
        assert_eq!(s.litLengthSum, 30 * (MaxLL + 1));
        assert_eq!(s.matchLengthSum, 10 * (MaxML + 1));
        assert_eq!(s.offCodeSum, 15 * (MaxOff + 1));
    }

    #[test]
    fn btGetAllMatches_noDict_returns_zero_if_already_updated() {
        use crate::compress::match_state::ZSTD_compressionParameters;
        let cp = ZSTD_compressionParameters {
            windowLog: 17, chainLog: 14, hashLog: 12, searchLog: 4,
            minMatch: 4, targetLength: 32, strategy: 7,
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
            &mut matches, &mut ms, &mut n3,
            &data, 50, data.len(),
            &rep, 0, 3, 4,
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

        let n = ZSTD_insertBtAndGetAllMatches_prefixOnly(
            &mut matches, &mut ms, &mut nextToUpdate3,
            &data, ip_abs, data.len(),
            &rep, 0, 3, 4,
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
        let positions = ZSTD_insertBt1(&mut ms, &data, 100, data.len(), 100, 4);
        assert!(positions >= 1);
        // Hash table entry for ip=100 should now be 100.
        let h = crate::compress::zstd_hashes::ZSTD_hashPtr(&data[100..], cp.hashLog, 4);
        assert_eq!(ms.hashTable[h], 100);
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
        let mut n = 0u32;
        let ret = ZSTD_insertAndFindFirstIndexHash3(&mut ms, &mut n, &data, 100);
        // No prior state → returned slot was zero.
        assert_eq!(ret, 0);
        // Cursor advanced.
        assert_eq!(n, 100);
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
        let mut n = 0u32;
        // First insert: positions 0..=3 seen.
        ZSTD_insertAndFindFirstIndexHash3(&mut ms, &mut n, &pat, 4);
        // Second call at pos 4 should find the prior position 0 (same
        // 3-byte prefix).
        let found = ZSTD_insertAndFindFirstIndexHash3(&mut ms, &mut n, &pat, 4);
        assert_eq!(found, 0, "prior position of same 3-byte hash");
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
        assert_eq!(s, 65 + 129 + 257 + 577);
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
}
