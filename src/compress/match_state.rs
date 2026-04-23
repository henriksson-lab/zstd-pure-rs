//! Scaffolding for `ZSTD_MatchState_t` — the big compressor state
//! struct from `lib/compress/zstd_compress_internal.h`. Upstream packs
//! window-buffer bookkeeping, hash tables, chain tables, a per-row
//! cache, dictionary match state, the optimal-parser state, and the
//! active `ZSTD_compressionParameters` into one place.
//!
//! We port a cut-down shape first. Fields arrive as needed — the
//! `zstd_fast` port only touches `hashTable`, `nextToUpdate`,
//! `window.base`, and `cParams.{hashLog, minMatch}`.
//! Later ticks add rows / chains / optimal state / dictionary
//! linkage incrementally.

#![allow(non_snake_case)]

/// Port of the public `ZSTD_compressionParameters` from `zstd.h`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZSTD_compressionParameters {
    pub windowLog: u32,
    pub chainLog: u32,
    pub hashLog: u32,
    pub searchLog: u32,
    pub minMatch: u32,
    pub targetLength: u32,
    /// Upstream `ZSTD_strategy` enum (1..=9). Stored as u32 here; the
    /// full enum will arrive with the top-level `ZSTD_compress` port.
    pub strategy: u32,
}

/// Port of `ZSTD_window_t`. Upstream stores raw `BYTE*` base pointers
/// alongside u32 indices. The Rust port keeps the indices (they're
/// compared and used as offsets) and moves the byte buffer one level
/// up — the match state borrows `&[u8]` at call time rather than
/// stashing a pointer.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZSTD_window_t {
    /// Index one-past the last processed byte. Upstream's `nextSrc`
    /// is a pointer; we store the byte offset relative to `base`.
    pub nextSrc: u32,
    /// Index of the first byte of the current contiguous prefix. All
    /// regular indexes are relative to this position. In upstream
    /// this is a pointer and `index = offset - base`; we store the
    /// bookkeeping u32 directly.
    pub base_offset: u32,
    /// Start of the external-dictionary segment (mirrors upstream's
    /// `dictBase`). Indexes below `dictLimit` look in this region.
    pub dictBase_offset: u32,
    /// Below this index, the caller must consult the ext-dict.
    pub dictLimit: u32,
    /// Below this, no valid data exists.
    pub lowLimit: u32,
    /// Diagnostic counter.
    pub nbOverflowCorrections: u32,
}

pub const ZSTD_WINDOW_START_INDEX: u32 = 2;

/// Port of `ZSTD_dictMode_e`. Tells block compressors how to treat
/// the match state's dictionary linkage:
///   - `ZSTD_noDict`: no dictionary in play.
///   - `ZSTD_extDict`: ext-dict bytes sit below `window.lowLimit`.
///   - `ZSTD_dictMatchState`: a CDict's match state is attached via
///     `ms.dictMatchState` — search both the current window and the
///     dict's precomputed tables.
///   - `ZSTD_dedicatedDictSearch`: DDSS variant of dictMatchState —
///     dict tables use bucketed hash layout for wider probing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZSTD_dictMode_e {
    #[default]
    ZSTD_noDict = 0,
    ZSTD_extDict = 1,
    ZSTD_dictMatchState = 2,
    ZSTD_dedicatedDictSearch = 3,
}

/// Port of `ZSTD_matchState_dictMode`. Inspects the match state and
/// returns the mode the block compressor should use.
///
pub fn ZSTD_matchState_dictMode(ms: &ZSTD_MatchState_t) -> ZSTD_dictMode_e {
    if ms.dictMatchState.is_some() && ms.loadedDictEnd != 0 {
        return ZSTD_dictMode_e::ZSTD_dictMatchState;
    }
    if ZSTD_window_hasExtDict(&ms.window) {
        ZSTD_dictMode_e::ZSTD_extDict
    } else {
        ZSTD_dictMode_e::ZSTD_noDict
    }
}

/// Upstream `ZSTD_CURRENT_MAX` (zstd_compress_internal.h:1031). Upper
/// limit on `nextSrc` before we must fold the index space back to
/// zero via overflow correction. 64-bit target: `3500 * (1 << 20)`
/// bytes (~3.67 GB) — leaves 512 MB margin below 4 GB for the
/// maximum job size. 32-bit target: `2000 * (1 << 20)` (~2 GB) to
/// avoid crossing the signed-index midpoint. Previously our port used
/// `(3 << 29) + 1` (~1.6 GB) — wrong by ~2 GB; never visibly broke
/// anything because overflow correction rarely triggers in v0.1's
/// roundtrips, but would cut history window capacity by ~half.
pub const ZSTD_CURRENT_MAX: u32 = if crate::common::mem::MEM_32bits() != 0 {
    2000u32 * (1u32 << 20)
} else {
    3500u32 * (1u32 << 20)
};

/// Upstream `ZSTD_CHUNKSIZE_MAX` (zstd_compress_internal.h:1033).
/// Maximum chunk size before overflow correction needs to be called
/// again: `u32::MAX - ZSTD_CURRENT_MAX`.
pub const ZSTD_CHUNKSIZE_MAX: u32 = u32::MAX - ZSTD_CURRENT_MAX;

/// Port of `ZSTD_dictTooBig` (zstd_compress.c:2113). A loaded dict
/// bigger than one chunk-max can't be represented without triggering
/// overflow correction mid-load — upstream uses this gate to decide
/// whether to attach vs copy the dict tables.
#[inline]
pub const fn ZSTD_dictTooBig(loadedDictSize: usize) -> bool {
    loadedDictSize > ZSTD_CHUNKSIZE_MAX as usize
}

/// Port of `ZSTD_INDEXOVERFLOW_MARGIN` (zstd_compress.c:2102).
/// 16 MB safety margin below `ZSTD_CURRENT_MAX` — the threshold
/// below which we can ignore overflow-correction for small inputs.
pub const ZSTD_INDEXOVERFLOW_MARGIN: u32 = 16 * (1 << 20);

/// Port of `ZSTD_indexTooCloseToMax` (zstd_compress.c:2103). Returns
/// true when the window's `nextSrc` index is within
/// `ZSTD_INDEXOVERFLOW_MARGIN` of `ZSTD_CURRENT_MAX` — upstream uses
/// this gate to decide whether small-input compressions can skip the
/// normally-mandatory overflow check. In our port the window stores
/// indices directly (no base pointer), so we compute
/// `nextSrc - base` as the absolute index.
#[inline]
pub fn ZSTD_indexTooCloseToMax(w: &ZSTD_window_t) -> bool {
    // `nextSrc - base` in the pointer world maps to `nextSrc` itself
    // in the index world (both are offsets relative to base).
    w.nextSrc as u64 > (ZSTD_CURRENT_MAX - ZSTD_INDEXOVERFLOW_MARGIN) as u64
}

/// Upstream `ZSTD_DUBT_UNSORTED_MARK`. For `btlazy2`, an index of 1
/// means "still in unsorted queue" — the reducer must preserve it
/// across overflow correction so sorting picks back up correctly.
pub const ZSTD_DUBT_UNSORTED_MARK: u32 = 1;

/// Upstream `ZSTD_ROWSIZE`. Hash-table reducer walks in
/// `ZSTD_ROWSIZE`-wide strides for auto-vectorization; table sizes
/// must be a multiple.
pub const ZSTD_ROWSIZE: usize = 16;
pub const ZSTD_ROW_HASH_CACHE_SIZE: usize = 8;

/// Port of `ZSTD_reduceTable_internal`. Shifts every hash-table entry
/// down by `reducerValue` (floor 0), optionally preserving the
/// `ZSTD_DUBT_UNSORTED_MARK` sentinel intact.
///
/// Table length must be a multiple of `ZSTD_ROWSIZE` (upstream
/// enforces via `assert`).
pub fn ZSTD_reduceTable_internal(table: &mut [u32], reducerValue: u32, preserveMark: bool) {
    let size = table.len();
    debug_assert_eq!(size & (ZSTD_ROWSIZE - 1), 0);
    debug_assert!(size < (1usize << 31));

    // Protect index values below ZSTD_WINDOW_START_INDEX from wrapping.
    let reducerThreshold = reducerValue + ZSTD_WINDOW_START_INDEX;

    for cell in table.iter_mut() {
        *cell = if preserveMark && *cell == ZSTD_DUBT_UNSORTED_MARK {
            ZSTD_DUBT_UNSORTED_MARK
        } else if *cell < reducerThreshold {
            0
        } else {
            *cell - reducerValue
        };
    }
}

/// Port of `ZSTD_reduceTable`. Non-preserving variant — used by fast
/// / dfast / lazy strategies.
#[inline]
pub fn ZSTD_reduceTable(table: &mut [u32], reducerValue: u32) {
    ZSTD_reduceTable_internal(table, reducerValue, false);
}

/// Port of `ZSTD_reduceTable_btlazy2`. Preserves the unsorted-mark
/// sentinel (index 1) across the reduce — required so the btlazy2
/// sorter can still identify queued-but-unsorted nodes.
#[inline]
pub fn ZSTD_reduceTable_btlazy2(table: &mut [u32], reducerValue: u32) {
    ZSTD_reduceTable_internal(table, reducerValue, true);
}

/// Port of `ZSTD_rowMatchFinderSupported`. Only `greedy`/`lazy`/`lazy2`
/// can opt into the row-based matchfinder.
#[inline]
pub fn ZSTD_rowMatchFinderSupported(strategy: u32) -> bool {
    // ZSTD_greedy = 3, ZSTD_lazy2 = 5 in upstream enum.
    (3..=5).contains(&strategy)
}

/// Port of `ZSTD_resolveRowMatchFinderMode`. Converts an auto/enable/
/// disable setting plus cParams into a concrete enable/disable
/// decision. Non-Linux-kernel path (14-bit windowLog floor) only.
pub fn ZSTD_resolveRowMatchFinderMode(
    mode: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
    cParams: &ZSTD_compressionParameters,
) -> crate::compress::zstd_ldm::ZSTD_ParamSwitch_e {
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
    const K_WINDOW_LOG_LOWER_BOUND: u32 = 14;
    if mode != ZSTD_ParamSwitch_e::ZSTD_ps_auto {
        return mode;
    }
    if !ZSTD_rowMatchFinderSupported(cParams.strategy) {
        return ZSTD_ParamSwitch_e::ZSTD_ps_disable;
    }
    if cParams.windowLog > K_WINDOW_LOG_LOWER_BOUND {
        ZSTD_ParamSwitch_e::ZSTD_ps_enable
    } else {
        ZSTD_ParamSwitch_e::ZSTD_ps_disable
    }
}

/// Port of `ZSTD_resolveBlockSplitterMode`. Block splitter is auto-
/// enabled for `btopt` and up with windowLog ≥ 17.
pub fn ZSTD_resolveBlockSplitterMode(
    mode: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
    cParams: &ZSTD_compressionParameters,
) -> crate::compress::zstd_ldm::ZSTD_ParamSwitch_e {
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
    if mode != ZSTD_ParamSwitch_e::ZSTD_ps_auto {
        return mode;
    }
    // ZSTD_btopt = 7.
    if cParams.strategy >= 7 && cParams.windowLog >= 17 {
        ZSTD_ParamSwitch_e::ZSTD_ps_enable
    } else {
        ZSTD_ParamSwitch_e::ZSTD_ps_disable
    }
}

/// Port of `ZSTD_resolveEnableLdm`. Long-distance matcher auto-enables
/// for `btopt` and up with windowLog ≥ 27.
pub fn ZSTD_resolveEnableLdm(
    mode: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
    cParams: &ZSTD_compressionParameters,
) -> crate::compress::zstd_ldm::ZSTD_ParamSwitch_e {
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
    if mode != ZSTD_ParamSwitch_e::ZSTD_ps_auto {
        return mode;
    }
    // ZSTD_btopt = 7.
    if cParams.strategy >= 7 && cParams.windowLog >= 27 {
        ZSTD_ParamSwitch_e::ZSTD_ps_enable
    } else {
        ZSTD_ParamSwitch_e::ZSTD_ps_disable
    }
}

/// Port of `ZSTD_resolveMaxBlockSize`. Returns `ZSTD_BLOCKSIZE_MAX` on
/// a zero default, caller's value otherwise.
#[inline]
pub fn ZSTD_resolveMaxBlockSize(maxBlockSize: usize) -> usize {
    use crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX;
    if maxBlockSize == 0 {
        ZSTD_BLOCKSIZE_MAX
    } else {
        maxBlockSize
    }
}

/// Port of `ZSTD_resolveExternalSequenceValidation`. Identity — the
/// user-supplied validation flag is passed through unchanged.
#[inline]
pub fn ZSTD_resolveExternalSequenceValidation(mode: i32) -> i32 {
    mode
}

/// Port of `ZSTD_resolveExternalRepcodeSearch`. On `auto`, enables
/// external repcode search for high compression levels (≥ 10).
pub fn ZSTD_resolveExternalRepcodeSearch(
    value: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
    cLevel: i32,
) -> crate::compress::zstd_ldm::ZSTD_ParamSwitch_e {
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
    if value != ZSTD_ParamSwitch_e::ZSTD_ps_auto {
        return value;
    }
    if cLevel < 10 {
        ZSTD_ParamSwitch_e::ZSTD_ps_disable
    } else {
        ZSTD_ParamSwitch_e::ZSTD_ps_enable
    }
}

/// Port of `ZSTD_CDictIndicesAreTagged`. True for fast/dfast strategies
/// — CDict hashtables carry short-cache tags only for those.
#[inline]
pub fn ZSTD_CDictIndicesAreTagged(cParams: &ZSTD_compressionParameters) -> bool {
    // ZSTD_fast = 1, ZSTD_dfast = 2.
    cParams.strategy == 1 || cParams.strategy == 2
}

/// Port of `ZSTD_rowMatchFinderUsed`. True when the row matchfinder
/// is both supported and explicitly enabled — `mode` must be a
/// resolved enable/disable, never auto.
#[inline]
pub fn ZSTD_rowMatchFinderUsed(
    strategy: u32,
    mode: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
) -> bool {
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
    debug_assert!(mode != ZSTD_ParamSwitch_e::ZSTD_ps_auto);
    ZSTD_rowMatchFinderSupported(strategy) && mode == ZSTD_ParamSwitch_e::ZSTD_ps_enable
}

/// Port of `ZSTD_allocateChainTable`. Gatekeeper for which strategies
/// actually need a `chainTable`: DDS-dict mode always allocates;
/// `fast` never allocates; the row-based matchfinder replaces the
/// chain entirely, so row-enabled strategies don't need one either.
#[inline]
pub fn ZSTD_allocateChainTable(
    strategy: u32,
    useRowMatchFinder: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
    forDDSDict: u32,
) -> bool {
    // ZSTD_fast = 1.
    forDDSDict != 0 || (strategy != 1 && !ZSTD_rowMatchFinderUsed(strategy, useRowMatchFinder))
}

/// Port of `ZSTD_bitmix`. XXH3-style 64-bit bit mixer — used by
/// `ZSTD_advanceHashSalt` to blend the row-hash salt with the caller-
/// supplied entropy on each reset.
#[inline]
pub fn ZSTD_bitmix(mut val: u64, len: u64) -> u64 {
    use crate::common::bits::ZSTD_rotateRight_U64;
    val ^= ZSTD_rotateRight_U64(val, 49) ^ ZSTD_rotateRight_U64(val, 24);
    val = val.wrapping_mul(0x9FB21C651E98DF25u64);
    val ^= (val >> 35) + len;
    val = val.wrapping_mul(0x9FB21C651E98DF25u64);
    val ^ (val >> 28)
}

/// Port of `ZSTD_advanceHashSalt`. Reseeds `ms.hashSalt` by mixing in
/// `ms.hashSaltEntropy` — run on each CCtx reset so reused row-hash
/// tables don't collide across compressions.
#[inline]
pub fn ZSTD_advanceHashSalt(ms: &mut ZSTD_MatchState_t) {
    ms.hashSalt = ZSTD_bitmix(ms.hashSalt, 8) ^ ZSTD_bitmix(ms.hashSaltEntropy as u64, 4);
}

/// Port of `ZSTD_invalidateMatchState`. Clears window state, rewinds
/// `nextToUpdate` to the dictLimit, and invalidates any loaded dict
/// end. Upstream also resets `opt.litLengthSum`; this port clears the
/// attached `dictMatchState` as well.
pub fn ZSTD_invalidateMatchState(ms: &mut ZSTD_MatchState_t) {
    ZSTD_window_clear(&mut ms.window);
    ms.nextToUpdate = ms.window.dictLimit;
    ms.dictMatchState = None;
    ms.loadedDictEnd = 0;
    // Caller-owned btopt opt.litLengthSum reset lands with the
    // remaining opt-state plumbing.
}

/// Port of `ZSTD_reduceIndex`. Walks the match-state's owned hash /
/// chain / 3-byte-hash tables and applies `ZSTD_reduceTable` to each
/// after an overflow correction so every stored index is rebased.
///
/// The btlazy2 chain table is special-cased to preserve the
/// `ZSTD_DUBT_UNSORTED_MARK` sentinel.
///
/// Upstream takes `ZSTD_CCtx_params*` to look up `useRowMatchFinder`
/// and `dedicatedDictSearch`; the Rust port accepts those two as
/// explicit parameters.
pub fn ZSTD_reduceIndex(
    ms: &mut ZSTD_MatchState_t,
    useRowMatchFinder: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
    dedicatedDictSearch: u32,
    reducerValue: u32,
) {
    ZSTD_reduceTable(&mut ms.hashTable, reducerValue);

    if ZSTD_allocateChainTable(ms.cParams.strategy, useRowMatchFinder, dedicatedDictSearch)
        && !ms.chainTable.is_empty()
    {
        // ZSTD_btlazy2 = 6.
        if ms.cParams.strategy == 6 {
            ZSTD_reduceTable_btlazy2(&mut ms.chainTable, reducerValue);
        } else {
            ZSTD_reduceTable(&mut ms.chainTable, reducerValue);
        }
    }

    if ms.hashLog3 > 0 && !ms.hashTable3.is_empty() {
        ZSTD_reduceTable(&mut ms.hashTable3, reducerValue);
    }
}

/// Upstream `ZSTD_SHORT_CACHE_TAG_BITS`. Width of the tag packed into
/// the low bits of `hashTable` entries when dictionary matchers use
/// the short-cache optimization.
pub const ZSTD_SHORT_CACHE_TAG_BITS: u32 = 8;

/// Upstream `ZSTD_SHORT_CACHE_TAG_MASK`. Isolates the low-tag bits.
pub const ZSTD_SHORT_CACHE_TAG_MASK: u32 = (1u32 << ZSTD_SHORT_CACHE_TAG_BITS) - 1;

/// Port of `ZSTD_index_overlap_check`. Returns true (upstream returns 1)
/// when `repIndex` does NOT overlap the prefix-boundary guard band
/// immediately below `prefixLowestIndex`.
///
/// Upstream relies on intentional unsigned underflow — we replicate via
/// `wrapping_sub` in the same u32 arithmetic space.
#[inline]
pub fn ZSTD_index_overlap_check(prefixLowestIndex: u32, repIndex: u32) -> bool {
    prefixLowestIndex.wrapping_sub(1).wrapping_sub(repIndex) >= 3
}

/// Port of `ZSTD_writeTaggedIndex`. Unpacks `hashAndTag` into
/// `(hash, tag)`, then packs `(index, tag)` back into the hash table.
/// Used by the short-cache matchfinder in CDict table fills.
#[inline]
pub fn ZSTD_writeTaggedIndex(hashTable: &mut [u32], hashAndTag: usize, index: u32) {
    let hash = hashAndTag >> ZSTD_SHORT_CACHE_TAG_BITS;
    let tag = (hashAndTag as u32) & ZSTD_SHORT_CACHE_TAG_MASK;
    debug_assert!(
        index >> (32 - ZSTD_SHORT_CACHE_TAG_BITS) == 0,
        "index must fit below the tag bits",
    );
    hashTable[hash] = (index << ZSTD_SHORT_CACHE_TAG_BITS) | tag;
}

/// Port of `ZSTD_comparePackedTags`. Returns true when the low tag
/// bits of both packed values match — lets the short-cache matchfinder
/// skip the expensive memory fetch on a tag mismatch.
#[inline]
pub fn ZSTD_comparePackedTags(packedTag1: usize, packedTag2: usize) -> bool {
    let t1 = (packedTag1 as u32) & ZSTD_SHORT_CACHE_TAG_MASK;
    let t2 = (packedTag2 as u32) & ZSTD_SHORT_CACHE_TAG_MASK;
    t1 == t2
}

/// Port of `ZSTD_window_clear`. Collapses `dictLimit` / `lowLimit`
/// down to the current end-of-window — effectively wiping any
/// ext-dict / history knowledge while keeping the base index intact.
pub fn ZSTD_window_clear(window: &mut ZSTD_window_t) {
    let end = window.nextSrc;
    window.lowLimit = end;
    window.dictLimit = end;
}

/// Port of `ZSTD_window_update`. Appends a new segment onto the
/// logical window, rotating the previous prefix into ext-dict mode
/// when the segment is non-contiguous.
///
/// In the pointer-based C code `src` is a raw pointer; in this index-
/// based Rust port the caller supplies the new segment's absolute
/// start index in the same coordinate space as `window.nextSrc`.
///
/// Returns `true` when the segment is contiguous, `false` when the
/// window had to pivot into ext-dict mode.
pub fn ZSTD_window_update(
    window: &mut ZSTD_window_t,
    src_abs: u32,
    srcSize: usize,
    forceNonContiguous: bool,
) -> bool {
    let mut contiguous = true;
    if srcSize == 0 {
        return contiguous;
    }

    if src_abs != window.nextSrc || forceNonContiguous {
        let distanceFromBase = window.nextSrc.wrapping_sub(window.base_offset);
        window.lowLimit = window.dictLimit;
        window.dictLimit = distanceFromBase;
        window.dictBase_offset = window.base_offset;
        window.base_offset = src_abs.wrapping_sub(distanceFromBase);
        if window.dictLimit.wrapping_sub(window.lowLimit)
            < crate::compress::zstd_fast::HASH_READ_SIZE as u32
        {
            window.lowLimit = window.dictLimit;
        }
        contiguous = false;
    }

    let src_end_abs = src_abs.wrapping_add(srcSize as u32);
    window.nextSrc = src_end_abs;

    if src_end_abs > window.dictBase_offset.wrapping_add(window.lowLimit)
        && src_abs < window.dictBase_offset.wrapping_add(window.dictLimit)
    {
        let highInputIdx = src_end_abs.wrapping_sub(window.dictBase_offset);
        window.lowLimit = highInputIdx.min(window.dictLimit);
    }

    contiguous
}

/// Port of `ZSTD_window_isEmpty`. True on a freshly initialized
/// window — all three bookkeeping values still sit at their starting
/// sentinel.
///
/// In our Rust encoding `nextSrc` already stores upstream's
/// `nextSrc - base`, so the `nextSrc - base == ZSTD_WINDOW_START_INDEX`
/// check from upstream collapses to a direct comparison here.
#[inline]
pub fn ZSTD_window_isEmpty(window: &ZSTD_window_t) -> bool {
    window.dictLimit == ZSTD_WINDOW_START_INDEX
        && window.lowLimit == ZSTD_WINDOW_START_INDEX
        && window.nextSrc == ZSTD_WINDOW_START_INDEX
}

/// Port of `ZSTD_window_hasExtDict`. Non-zero when the window has a
/// non-empty ext-dict region (an earlier prefix that moved out of
/// direct reach).
#[inline]
pub fn ZSTD_window_hasExtDict(window: &ZSTD_window_t) -> bool {
    window.lowLimit < window.dictLimit
}

/// Port of `ZSTD_window_canOverflowCorrect`. Returns true when the
/// index space has grown large enough that overflow correction can
/// run without hurting compression ratio (dictionary already shifted
/// out of reach) AND we've pushed past the adjusted minimum threshold.
///
/// `src_abs` is the absolute-index value of the current cursor — in
/// upstream pointer arithmetic `(src - window.base)`, translated here
/// to an explicit u32 (since our Rust `window` stores only indices,
/// not pointers).
pub fn ZSTD_window_canOverflowCorrect(
    window: &ZSTD_window_t,
    cycleLog: u32,
    maxDist: u32,
    loadedDictEnd: u32,
    src_abs: u32,
) -> bool {
    let cycleSize = 1u32 << cycleLog;
    let curr = src_abs;
    let minIndexToOverflowCorrect = cycleSize + maxDist.max(cycleSize) + ZSTD_WINDOW_START_INDEX;

    // Wraparound is intentional — upstream relies on saturation via
    // u32 overflow to keep the threshold at at least the base value.
    let adjustment = window.nbOverflowCorrections + 1;
    let adjustedIndex = minIndexToOverflowCorrect
        .saturating_mul(adjustment)
        .max(minIndexToOverflowCorrect);
    let indexLargeEnough = curr > adjustedIndex;

    // Only worth correcting early if the dictionary is already
    // invalidated, so we don't lose useful history.
    let dictionaryInvalidated = curr > maxDist.saturating_add(loadedDictEnd);

    indexLargeEnough && dictionaryInvalidated
}

/// Port of `ZSTD_window_correctOverflow`. Reduces every bookkeeping
/// index by a single correction value so indices stay in range for
/// power-of-two hash tables without corrupting the `cycleLog`
/// alignment that chains / binary trees rely on.
///
/// `src_abs` is the current cursor's absolute index (upstream's
/// `src - window.base`). The returned correction value must be
/// subtracted from every hash-table / chain-table entry by the caller.
///
/// Preserves the invariant `(newCurrent & cycleMask) == (curr & cycleMask)`
/// so already-rooted tree nodes still address the same cycle slot.
pub fn ZSTD_window_correctOverflow(
    window: &mut ZSTD_window_t,
    cycleLog: u32,
    maxDist: u32,
    src_abs: u32,
) -> u32 {
    let cycleSize = 1u32 << cycleLog;
    let cycleMask = cycleSize - 1;
    let curr = src_abs;
    let currentCycle = curr & cycleMask;
    // Keep `newCurrent - maxDist >= ZSTD_WINDOW_START_INDEX` even
    // when currentCycle is tiny.
    let currentCycleCorrection = if currentCycle < ZSTD_WINDOW_START_INDEX {
        cycleSize.max(ZSTD_WINDOW_START_INDEX)
    } else {
        0
    };
    let newCurrent = currentCycle + currentCycleCorrection + maxDist.max(cycleSize);
    let correction = curr - newCurrent;

    debug_assert!(maxDist & (maxDist - 1) == 0, "maxDist must be power of two");
    debug_assert_eq!(curr & cycleMask, newCurrent & cycleMask);
    debug_assert!(curr > newCurrent);

    // base / dictBase would advance in upstream's pointer space; in
    // our Rust encoding they're u32 offsets, so subtract the
    // correction downward on the index-space side.
    window.base_offset = window.base_offset.wrapping_add(correction);
    window.dictBase_offset = window.dictBase_offset.wrapping_add(correction);
    window.lowLimit = if window.lowLimit < correction + ZSTD_WINDOW_START_INDEX {
        ZSTD_WINDOW_START_INDEX
    } else {
        window.lowLimit - correction
    };
    window.dictLimit = if window.dictLimit < correction + ZSTD_WINDOW_START_INDEX {
        ZSTD_WINDOW_START_INDEX
    } else {
        window.dictLimit - correction
    };

    debug_assert!(newCurrent >= maxDist);
    debug_assert!(newCurrent - maxDist >= ZSTD_WINDOW_START_INDEX);
    debug_assert!(window.lowLimit <= newCurrent);
    debug_assert!(window.dictLimit <= newCurrent);

    window.nbOverflowCorrections += 1;
    correction
}

/// Port of `ZSTD_window_init`. Sets up the starting-sentinel values
/// so a fresh window compares equal to every other fresh window and
/// so index 0/1 aren't used (they collide with DUBT sentinels).
pub fn ZSTD_window_init(window: &mut ZSTD_window_t) {
    *window = ZSTD_window_t::default();
    window.dictLimit = ZSTD_WINDOW_START_INDEX;
    window.lowLimit = ZSTD_WINDOW_START_INDEX;
    window.nextSrc = ZSTD_WINDOW_START_INDEX;
    // base_offset / dictBase_offset start at 0 — upstream points them
    // at a dummy byte, but in the u32-index encoding any shared value
    // will do.
}

/// Port of `ZSTD_window_enforceMaxDist`. Advances `window.lowLimit`
/// so every valid index stays within `maxDist` bytes of `blockEnd_abs`,
/// invalidating any loaded dictionary when input has moved past it.
///
/// `blockEnd_abs` is the absolute-index value of the end of the
/// current block (upstream's `blockEnd - window.base`).
///
/// `loadedDictEnd` is in/out: reset to 0 when the dictionary is
/// invalidated. Pass `&mut 0` when no dictionary is in use.
///
/// The `dictMatchState**` parameter from upstream is dropped here —
/// our Rust `ZSTD_MatchState_t` doesn't carry that linkage yet.
pub fn ZSTD_window_enforceMaxDist(
    window: &mut ZSTD_window_t,
    blockEnd_abs: u32,
    maxDist: u32,
    loadedDictEnd: &mut u32,
) {
    let blockEndIdx = blockEnd_abs;
    let loadedEnd = *loadedDictEnd;

    if blockEndIdx > maxDist + loadedEnd {
        let newLowLimit = blockEndIdx - maxDist;
        if window.lowLimit < newLowLimit {
            window.lowLimit = newLowLimit;
        }
        if window.dictLimit < window.lowLimit {
            window.dictLimit = window.lowLimit;
        }
        *loadedDictEnd = 0;
    }
}

/// Port of `ZSTD_checkDictValidity`. Invalidates `loadedDictEnd` once
/// the current block has moved beyond the window — either past
/// `loadedDictEnd + maxDist`, or after a non-contiguous jump flagged
/// by `loadedDictEnd != window.dictLimit`.
///
/// Upstream also nulls a `dictMatchState**`; we skip that since the
/// Rust port's `ZSTD_MatchState_t` doesn't yet carry that linkage.
pub fn ZSTD_checkDictValidity(
    window: &ZSTD_window_t,
    blockEnd_abs: u32,
    maxDist: u32,
    loadedDictEnd: &mut u32,
) {
    let loadedEnd = *loadedDictEnd;
    debug_assert!(blockEnd_abs >= loadedEnd);
    if blockEnd_abs > loadedEnd + maxDist || loadedEnd != window.dictLimit {
        *loadedDictEnd = 0;
    }
}

/// Port of `ZSTD_window_needOverflowCorrection`. Returns true when
/// indices have grown large enough that we must correct now — either
/// we've crossed `ZSTD_CURRENT_MAX`, or we're in "correct-frequently"
/// diagnostic mode (not exposed here) and canOverflowCorrect agrees.
///
/// `srcEnd_abs` is the absolute-index value of the end of the current
/// chunk (upstream `srcEnd - window.base`).
pub fn ZSTD_window_needOverflowCorrection(
    _window: &ZSTD_window_t,
    _cycleLog: u32,
    _maxDist: u32,
    _loadedDictEnd: u32,
    _src_abs: u32,
    srcEnd_abs: u32,
) -> bool {
    // We don't compile in `ZSTD_WINDOW_OVERFLOW_CORRECT_FREQUENTLY`
    // (fuzz-only diagnostic), so the canOverflowCorrect early-return
    // is inert — collapse to the ZSTD_CURRENT_MAX check.
    srcEnd_abs > ZSTD_CURRENT_MAX
}

/// Port of `ZSTD_overflowCorrectIfNeeded` (`zstd_compress.c:4550`).
/// Checks whether the window has accumulated enough absolute-index
/// distance to risk a u32 wrap (`> ZSTD_CURRENT_MAX`); if so, runs
/// `ZSTD_window_correctOverflow` to shift indices down and applies
/// the same correction to every hash/chain table entry via
/// `ZSTD_reduceIndex`. Also forces `nextToUpdate` and `loadedDictEnd`
/// to match the new index space.
///
/// Upstream takes `ZSTD_cwksp*` to mark tables dirty/clean around the
/// reduce; our Rust port owns tables as `Vec` and doesn't need that
/// transactional marking — the reduce is in-place.
///
/// `src_abs` is the current cursor's absolute index; `srcEnd_abs` is
/// the end of the current chunk (upstream's `iend - window.base`).
pub fn ZSTD_overflowCorrectIfNeeded(
    ms: &mut ZSTD_MatchState_t,
    useRowMatchFinder: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e,
    dedicatedDictSearch: u32,
    windowLog: u32,
    chainLog: u32,
    strategy: u32,
    src_abs: u32,
    srcEnd_abs: u32,
) {
    let cycleLog = crate::compress::zstd_compress::ZSTD_cycleLog(chainLog, strategy);
    let maxDist: u32 = 1u32 << windowLog;
    if ZSTD_window_needOverflowCorrection(
        &ms.window,
        cycleLog,
        maxDist,
        ms.loadedDictEnd,
        src_abs,
        srcEnd_abs,
    ) {
        let correction = ZSTD_window_correctOverflow(&mut ms.window, cycleLog, maxDist, src_abs);
        ZSTD_reduceIndex(ms, useRowMatchFinder, dedicatedDictSearch, correction);
        if ms.nextToUpdate < correction {
            ms.nextToUpdate = 0;
        } else {
            ms.nextToUpdate -= correction;
        }
        // Invalidate loaded dict on overflow correction — upstream
        // also nulls `ms.dictMatchState`, which we don't yet track.
        ms.loadedDictEnd = 0;
    }
}

/// Cut-down `ZSTD_MatchState_t`. Covers the primary hash table, the
/// `hashTable3` short-hash leg, and the `chainTable` used by
/// `zstd_lazy` / `zstd_opt`. Row-hash scratch fields are live for the
/// greedy/lazy/lazy2 matchfinders; only optimal-parser row mode
/// remains out of scope, matching upstream's strategy coverage.
#[derive(Debug, Clone)]
pub struct ZSTD_MatchState_t {
    pub window: ZSTD_window_t,

    /// Optional entropy-table seed for first-block optimal-parser
    /// pricing. Callers stash the prior block or dictionary entropy
    /// here before invoking `zstd_opt`.
    pub entropySeed: Option<crate::compress::zstd_compress::ZSTD_entropyCTables_t>,

    /// Upstream `ldmSeqStore`. When set, the optimal parser considers
    /// long-distance matcher sequences as candidate matches.
    pub ldmSeqStore: Option<crate::compress::zstd_ldm::RawSeqStore_t>,

    /// Upstream `dictMatchState`. When present, points at the loaded
    /// dictionary's match-state tables for DMS / DDSS search modes.
    pub dictMatchState: Option<Box<ZSTD_MatchState_t>>,
    /// Raw dictionary bytes associated with this match state. This is
    /// separate from `dictMatchState`: ext-dict modes dereference the
    /// local state's external dictionary bytes, while dictMatchState
    /// modes dereference the attached dictionary state's bytes.
    pub dictContent: Vec<u8>,

    /// Upstream `loadedDictEnd`. Non-zero while a dictionary is in
    /// use.
    pub loadedDictEnd: u32,

    /// Next index (into the `base`-relative position space) that
    /// `ZSTD_fillHashTable` still needs to insert.
    pub nextToUpdate: u32,

    /// Owned hash table sized `1 << (cParams.hashLog +
    /// ZSTD_SHORT_CACHE_TAG_BITS)` for the tagged-cache variant;
    /// `zstd_fast` uses the untagged `1 << cParams.hashLog`.
    pub hashTable: Vec<u32>,

    /// Secondary hash for 3-byte matches (used by `zstd_double_fast`
    /// and opt paths). Zero-sized until allocated.
    pub hashTable3: Vec<u32>,

    /// Upstream `hashLog3`. Log2 size of `hashTable3`; zero when the
    /// table isn't allocated.
    pub hashLog3: u32,

    /// Upstream `rowHashLog`. Log2 number of rows in the row-hash table.
    pub rowHashLog: u32,

    /// Upstream `tagTable`. Row-hash tags plus per-row head slot.
    pub tagTable: Vec<u8>,

    /// Upstream `hashCache`. Small rolling cache for row-hash updates.
    pub hashCache: [u32; ZSTD_ROW_HASH_CACHE_SIZE],

    /// Row-hash matcher salt — reseeded via `ZSTD_advanceHashSalt`
    /// so reused tables don't collide across compressions.
    pub hashSalt: u64,

    /// Entropy mixed into `hashSalt` on each reset.
    pub hashSaltEntropy: u32,

    /// Upstream `lazySkipping`. Non-zero tells lazy matchfinders to
    /// stop inserting every intermediate position.
    pub lazySkipping: u32,

    /// Chain table for `zstd_lazy` / `zstd_opt`. Zero-sized until
    /// allocated.
    pub chainTable: Vec<u32>,

    /// Active compression parameters.
    pub cParams: ZSTD_compressionParameters,
}

impl ZSTD_MatchState_t {
    /// Fresh state with the given compression parameters. Hash tables
    /// are sized from `cParams.hashLog`; chain / tag tables stay empty
    /// until the matcher actually needs them.
    pub fn new(cParams: ZSTD_compressionParameters) -> Self {
        let hashSize = 1usize << cParams.hashLog;
        Self {
            window: ZSTD_window_t {
                base_offset: ZSTD_WINDOW_START_INDEX,
                nextSrc: ZSTD_WINDOW_START_INDEX,
                dictBase_offset: 0,
                dictLimit: ZSTD_WINDOW_START_INDEX,
                lowLimit: ZSTD_WINDOW_START_INDEX,
                nbOverflowCorrections: 0,
            },
            entropySeed: None,
            ldmSeqStore: None,
            dictMatchState: None,
            dictContent: Vec::new(),
            loadedDictEnd: 0,
            nextToUpdate: ZSTD_WINDOW_START_INDEX,
            hashTable: vec![0u32; hashSize],
            hashTable3: Vec::new(),
            hashLog3: 0,
            rowHashLog: 0,
            tagTable: Vec::new(),
            hashCache: [0; ZSTD_ROW_HASH_CACHE_SIZE],
            hashSalt: 0,
            hashSaltEntropy: 0,
            lazySkipping: 0,
            chainTable: Vec::new(),
            cParams,
        }
    }

    /// Reset the match state, keeping allocations.
    pub fn reset(&mut self) {
        self.window = ZSTD_window_t {
            base_offset: ZSTD_WINDOW_START_INDEX,
            nextSrc: ZSTD_WINDOW_START_INDEX,
            dictBase_offset: 0,
            dictLimit: ZSTD_WINDOW_START_INDEX,
            lowLimit: ZSTD_WINDOW_START_INDEX,
            nbOverflowCorrections: 0,
        };
        self.entropySeed = None;
        self.ldmSeqStore = None;
        self.dictMatchState = None;
        self.dictContent.clear();
        self.loadedDictEnd = 0;
        self.nextToUpdate = ZSTD_WINDOW_START_INDEX;
        for c in self.hashTable.iter_mut() {
            *c = 0;
        }
        for c in self.hashTable3.iter_mut() {
            *c = 0;
        }
        for c in self.tagTable.iter_mut() {
            *c = 0;
        }
        for c in self.chainTable.iter_mut() {
            *c = 0;
        }
        self.rowHashLog = 0;
        self.hashCache = [0; ZSTD_ROW_HASH_CACHE_SIZE];
        self.hashSalt = 0;
        self.hashSaltEntropy = 0;
        self.lazySkipping = 0;
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn ZSTD_CURRENT_MAX_and_CHUNKSIZE_MAX_match_upstream_formula() {
        // Previously `(3<<29)+1 = 1.6 GB`; upstream is `3500*(1<<20)
        // = 3.67 GB` on 64-bit / `2000*(1<<20) = 2.0 GB` on 32-bit.
        // Pin the fix so the constant can't drift back.
        let expected = if crate::common::mem::MEM_32bits() != 0 {
            2000u32 * (1u32 << 20)
        } else {
            3500u32 * (1u32 << 20)
        };
        assert_eq!(ZSTD_CURRENT_MAX, expected);
        assert_eq!(ZSTD_CHUNKSIZE_MAX, u32::MAX - ZSTD_CURRENT_MAX);
        // And the dictTooBig gate flips at the chunk boundary.
        assert!(!ZSTD_dictTooBig(ZSTD_CHUNKSIZE_MAX as usize));
        assert!(ZSTD_dictTooBig(ZSTD_CHUNKSIZE_MAX as usize + 1));
    }

    #[test]
    fn ZSTD_dictMode_e_discriminants_match_upstream() {
        // Upstream (zstd_compress_internal.h:551-557):
        //   ZSTD_noDict=0, ZSTD_extDict=1, ZSTD_dictMatchState=2,
        //   ZSTD_dedicatedDictSearch=3
        // Block compressors dispatch on this enum via equality
        // checks; reordering would silently route to wrong strategy.
        assert_eq!(ZSTD_dictMode_e::ZSTD_noDict as u32, 0);
        assert_eq!(ZSTD_dictMode_e::ZSTD_extDict as u32, 1);
        assert_eq!(ZSTD_dictMode_e::ZSTD_dictMatchState as u32, 2);
        assert_eq!(ZSTD_dictMode_e::ZSTD_dedicatedDictSearch as u32, 3);
    }

    #[test]
    fn new_state_sizes_hash_table_from_hashlog() {
        let cp = ZSTD_compressionParameters {
            hashLog: 12,
            minMatch: 5,
            ..Default::default()
        };
        let ms = ZSTD_MatchState_t::new(cp);
        assert_eq!(ms.hashTable.len(), 1 << 12);
        assert_eq!(ms.cParams.minMatch, 5);
        // Initial window indices start at ZSTD_WINDOW_START_INDEX so
        // a zero-valued `hashTable[k] == 0` means "never inserted".
        assert_eq!(ms.nextToUpdate, ZSTD_WINDOW_START_INDEX);
        assert_eq!(ms.window.base_offset, ZSTD_WINDOW_START_INDEX);
    }

    #[test]
    fn window_isEmpty_true_at_init() {
        let w = ZSTD_window_t {
            nextSrc: ZSTD_WINDOW_START_INDEX,
            base_offset: ZSTD_WINDOW_START_INDEX,
            dictBase_offset: 0,
            dictLimit: ZSTD_WINDOW_START_INDEX,
            lowLimit: ZSTD_WINDOW_START_INDEX,
            nbOverflowCorrections: 0,
        };
        assert!(ZSTD_window_isEmpty(&w));
    }

    #[test]
    fn window_isEmpty_false_after_nextSrc_advances() {
        let mut w = ZSTD_window_t {
            nextSrc: ZSTD_WINDOW_START_INDEX,
            base_offset: ZSTD_WINDOW_START_INDEX,
            dictBase_offset: 0,
            dictLimit: ZSTD_WINDOW_START_INDEX,
            lowLimit: ZSTD_WINDOW_START_INDEX,
            nbOverflowCorrections: 0,
        };
        w.nextSrc += 1;
        assert!(!ZSTD_window_isEmpty(&w));
    }

    #[test]
    fn window_hasExtDict_tracks_lowLimit_vs_dictLimit() {
        let mut w = ZSTD_window_t::default();
        w.lowLimit = 5;
        w.dictLimit = 10;
        assert!(ZSTD_window_hasExtDict(&w));
        w.lowLimit = 10;
        assert!(!ZSTD_window_hasExtDict(&w));
        w.lowLimit = 11;
        assert!(!ZSTD_window_hasExtDict(&w));
    }

    #[test]
    fn window_clear_collapses_limits_to_nextSrc() {
        let mut w = ZSTD_window_t::default();
        w.nextSrc = 1000;
        w.dictLimit = 50;
        w.lowLimit = 10;
        ZSTD_window_clear(&mut w);
        assert_eq!(w.dictLimit, 1000);
        assert_eq!(w.lowLimit, 1000);
    }

    #[test]
    fn window_update_contiguous_only_advances_nextsrc() {
        let mut w = ZSTD_window_t {
            base_offset: 100,
            dictBase_offset: 80,
            dictLimit: 120,
            lowLimit: 90,
            nextSrc: 150,
            nbOverflowCorrections: 0,
        };
        let contiguous = ZSTD_window_update(&mut w, 150, 20, false);
        assert!(contiguous);
        assert_eq!(w.base_offset, 100);
        assert_eq!(w.dictBase_offset, 80);
        assert_eq!(w.dictLimit, 120);
        assert_eq!(w.lowLimit, 90);
        assert_eq!(w.nextSrc, 170);
    }

    #[test]
    fn window_update_non_contiguous_rotates_prefix_into_extdict() {
        let mut w = ZSTD_window_t {
            base_offset: 100,
            dictBase_offset: 50,
            dictLimit: 110,
            lowLimit: 70,
            nextSrc: 180,
            nbOverflowCorrections: 0,
        };
        let contiguous = ZSTD_window_update(&mut w, 300, 16, true);
        assert!(!contiguous);
        assert_eq!(w.dictBase_offset, 100);
        assert_eq!(w.dictLimit, 80);
        assert_eq!(w.base_offset, 220);
        assert_eq!(w.lowLimit, 110);
        assert_eq!(w.nextSrc, 316);
    }

    #[test]
    fn canOverflowCorrect_false_when_curr_too_small() {
        let w = ZSTD_window_t::default();
        // Tiny curr — far below minIndexToOverflowCorrect.
        assert!(!ZSTD_window_canOverflowCorrect(&w, 10, 1 << 20, 0, 1000));
    }

    #[test]
    fn canOverflowCorrect_requires_dict_invalidation() {
        let w = ZSTD_window_t::default();
        let cycleLog = 10;
        let cycleSize = 1u32 << cycleLog;
        let maxDist = 1u32 << 20;
        let minIdx = cycleSize + maxDist.max(cycleSize) + ZSTD_WINDOW_START_INDEX;
        let curr = minIdx + 100;
        // With loadedDictEnd = curr - 100, curr > maxDist + loadedDictEnd
        // only if curr - loadedDictEnd > maxDist, i.e. 100 > 1<<20 → false.
        let loaded = curr - 100;
        assert!(!ZSTD_window_canOverflowCorrect(
            &w, cycleLog, maxDist, loaded, curr
        ));
        // But with loadedDictEnd = 0 the dict is already invalidated.
        assert!(ZSTD_window_canOverflowCorrect(
            &w, cycleLog, maxDist, 0, curr
        ));
    }

    #[test]
    fn needOverflowCorrection_tracks_ZSTD_CURRENT_MAX() {
        let w = ZSTD_window_t::default();
        assert!(!ZSTD_window_needOverflowCorrection(
            &w,
            10,
            1 << 20,
            0,
            0,
            1000
        ));
        assert!(!ZSTD_window_needOverflowCorrection(
            &w,
            10,
            1 << 20,
            0,
            0,
            ZSTD_CURRENT_MAX
        ));
        assert!(ZSTD_window_needOverflowCorrection(
            &w,
            10,
            1 << 20,
            0,
            0,
            ZSTD_CURRENT_MAX + 1,
        ));
    }

    #[test]
    fn correctOverflow_preserves_cycle_alignment() {
        let mut w = ZSTD_window_t::default();
        w.dictLimit = 1 << 24;
        w.lowLimit = 1 << 20;
        let cycleLog = 16;
        let maxDist = 1u32 << 20;
        let src_abs: u32 = ZSTD_CURRENT_MAX + 1000;
        let correction = ZSTD_window_correctOverflow(&mut w, cycleLog, maxDist, src_abs);
        // newCurrent = curr - correction; they must share cycle bits.
        let cycleMask = (1u32 << cycleLog) - 1;
        let newCurrent = src_abs - correction;
        assert_eq!(src_abs & cycleMask, newCurrent & cycleMask);
        assert_eq!(w.nbOverflowCorrections, 1);
        assert!(newCurrent >= maxDist);
    }

    #[test]
    fn correctOverflow_floors_lowLimit_when_small() {
        let mut w = ZSTD_window_t::default();
        // lowLimit smaller than correction → gets pinned at START_INDEX.
        w.lowLimit = 5;
        w.dictLimit = 1 << 24;
        let src_abs: u32 = ZSTD_CURRENT_MAX + 500;
        ZSTD_window_correctOverflow(&mut w, 16, 1u32 << 20, src_abs);
        assert_eq!(w.lowLimit, ZSTD_WINDOW_START_INDEX);
    }

    #[test]
    fn correctOverflow_returns_positive_correction() {
        let mut w = ZSTD_window_t::default();
        w.dictLimit = 1 << 24;
        let src_abs: u32 = ZSTD_CURRENT_MAX + 42;
        let c = ZSTD_window_correctOverflow(&mut w, 16, 1u32 << 20, src_abs);
        // Upstream asserts correction > 1<<28 outside frequent-correct
        // mode — verify we land in that ballpark.
        assert!(c > (1u32 << 28));
    }

    #[test]
    fn window_init_sets_all_sentinels() {
        let mut w = ZSTD_window_t {
            nextSrc: 999,
            base_offset: 42,
            dictBase_offset: 17,
            dictLimit: 500,
            lowLimit: 400,
            nbOverflowCorrections: 5,
        };
        ZSTD_window_init(&mut w);
        assert_eq!(w.nextSrc, ZSTD_WINDOW_START_INDEX);
        assert_eq!(w.dictLimit, ZSTD_WINDOW_START_INDEX);
        assert_eq!(w.lowLimit, ZSTD_WINDOW_START_INDEX);
        assert_eq!(w.nbOverflowCorrections, 0);
        assert!(ZSTD_window_isEmpty(&w));
    }

    #[test]
    fn enforceMaxDist_noop_when_within_dist() {
        let mut w = ZSTD_window_t::default();
        w.lowLimit = 5;
        w.dictLimit = 10;
        let mut dictEnd = 0u32;
        // blockEnd 50, maxDist 100: still within range, no update.
        ZSTD_window_enforceMaxDist(&mut w, 50, 100, &mut dictEnd);
        assert_eq!(w.lowLimit, 5);
        assert_eq!(w.dictLimit, 10);
    }

    #[test]
    fn enforceMaxDist_advances_lowLimit_and_invalidates_dict() {
        let mut w = ZSTD_window_t::default();
        w.lowLimit = 5;
        w.dictLimit = 10;
        let mut dictEnd = 50u32;
        // blockEnd = 1200, maxDist = 100, loaded = 50 → 1200 > 150 → shift.
        // newLowLimit = 1100. dictLimit (10) < newLowLimit, so dictLimit = 1100.
        ZSTD_window_enforceMaxDist(&mut w, 1200, 100, &mut dictEnd);
        assert_eq!(w.lowLimit, 1100);
        assert_eq!(w.dictLimit, 1100);
        assert_eq!(dictEnd, 0);
    }

    #[test]
    fn checkDictValidity_invalidates_past_window() {
        let w = ZSTD_window_t {
            dictLimit: 100,
            ..Default::default()
        };
        let mut dictEnd = 100u32;
        // blockEnd = 500, maxDist = 100, loaded = 100 → 500 > 200 → invalidate.
        ZSTD_checkDictValidity(&w, 500, 100, &mut dictEnd);
        assert_eq!(dictEnd, 0);
    }

    #[test]
    fn checkDictValidity_invalidates_on_noncontig_jump() {
        let w = ZSTD_window_t {
            dictLimit: 100,
            ..Default::default()
        };
        let mut dictEnd = 80u32; // != dictLimit
        ZSTD_checkDictValidity(&w, 150, 1000, &mut dictEnd);
        assert_eq!(dictEnd, 0);
    }

    #[test]
    fn index_overlap_check_reports_non_overlap() {
        // prefixLowestIndex=100, repIndex=50 → (99 - 50) = 49 ≥ 3 → true.
        assert!(ZSTD_index_overlap_check(100, 50));
    }

    #[test]
    fn index_overlap_check_detects_guardband_overlap() {
        // repIndex inside the 3-byte guard below prefixLowestIndex.
        // prefixLowestIndex=100, repIndex=98 → (99 - 98) = 1 < 3 → false.
        assert!(!ZSTD_index_overlap_check(100, 98));
        assert!(!ZSTD_index_overlap_check(100, 99));
    }

    #[test]
    fn index_overlap_check_pass_when_in_prefix() {
        // repIndex >= prefixLowestIndex: (prefix-1 - repIndex) wraps
        // to a huge u32 ≥ 3 — upstream reads this as non-overlap
        // (repIndex sits in the prefix itself, always valid).
        assert!(ZSTD_index_overlap_check(100, 100));
        assert!(ZSTD_index_overlap_check(100, 500));
    }

    #[test]
    fn writeTaggedIndex_roundtrip() {
        let mut t = vec![0u32; 16];
        // hash=5, tag=0xAB → hashAndTag = (5<<8)|0xAB = 0x5AB.
        let hashAndTag: usize = (5 << ZSTD_SHORT_CACHE_TAG_BITS) | 0xAB;
        ZSTD_writeTaggedIndex(&mut t, hashAndTag, 0x1234);
        // Stored value should be (0x1234 << 8) | 0xAB.
        assert_eq!(t[5], (0x1234u32 << 8) | 0xAB);
    }

    #[test]
    fn comparePackedTags_matches_low_byte_only() {
        let a: usize = (0xDEADBEEF << 8) | 0x42;
        let b: usize = (0x12345678 << 8) | 0x42;
        let c: usize = (0x12345678 << 8) | 0x41;
        assert!(ZSTD_comparePackedTags(a, b));
        assert!(!ZSTD_comparePackedTags(a, c));
    }

    #[test]
    fn reduceTable_shifts_above_threshold_and_floors_below() {
        let mut t = vec![0u32; ZSTD_ROWSIZE];
        t[0] = 100;
        t[1] = 50;
        t[2] = 42;
        t[3] = 2; // just under threshold (reducer=40, threshold=40+2=42)
        ZSTD_reduceTable(&mut t, 40);
        assert_eq!(t[0], 60);
        assert_eq!(t[1], 10);
        assert_eq!(t[2], 2); // 42 >= 42 threshold → 42-40=2
        assert_eq!(t[3], 0); // 2 < threshold → floored
    }

    #[test]
    fn reduceTable_btlazy2_preserves_unsorted_mark() {
        let mut t = vec![0u32; ZSTD_ROWSIZE];
        t[0] = 100;
        t[1] = ZSTD_DUBT_UNSORTED_MARK;
        t[2] = ZSTD_DUBT_UNSORTED_MARK;
        t[3] = 50;
        ZSTD_reduceTable_btlazy2(&mut t, 40);
        assert_eq!(t[0], 60);
        assert_eq!(t[1], ZSTD_DUBT_UNSORTED_MARK); // preserved
        assert_eq!(t[2], ZSTD_DUBT_UNSORTED_MARK); // preserved
        assert_eq!(t[3], 10);
    }

    #[test]
    fn reduceTable_plain_does_not_preserve_unsorted_mark() {
        let mut t = vec![0u32; ZSTD_ROWSIZE];
        t[1] = ZSTD_DUBT_UNSORTED_MARK;
        ZSTD_reduceTable(&mut t, 40);
        // UNSORTED_MARK=1 < threshold=42 → squashed to 0.
        assert_eq!(t[1], 0);
    }

    #[test]
    fn rowMatchFinderSupported_only_for_greedy_lazy_lazy2() {
        assert!(!ZSTD_rowMatchFinderSupported(1)); // fast
        assert!(!ZSTD_rowMatchFinderSupported(2)); // dfast
        assert!(ZSTD_rowMatchFinderSupported(3)); // greedy
        assert!(ZSTD_rowMatchFinderSupported(4)); // lazy
        assert!(ZSTD_rowMatchFinderSupported(5)); // lazy2
        assert!(!ZSTD_rowMatchFinderSupported(6)); // btlazy2
        assert!(!ZSTD_rowMatchFinderSupported(7)); // btopt
    }

    #[test]
    fn allocateChainTable_rules() {
        use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
        // fast (1) → never allocate (except DDS).
        assert!(!ZSTD_allocateChainTable(
            1,
            ZSTD_ParamSwitch_e::ZSTD_ps_disable,
            0
        ));
        assert!(ZSTD_allocateChainTable(
            1,
            ZSTD_ParamSwitch_e::ZSTD_ps_disable,
            1
        ));
        // dfast (2) → always allocate.
        assert!(ZSTD_allocateChainTable(
            2,
            ZSTD_ParamSwitch_e::ZSTD_ps_disable,
            0
        ));
        // greedy (3) with row matchfinder enabled → skip chain.
        assert!(!ZSTD_allocateChainTable(
            3,
            ZSTD_ParamSwitch_e::ZSTD_ps_enable,
            0
        ));
        // greedy (3) with row matchfinder disabled → allocate.
        assert!(ZSTD_allocateChainTable(
            3,
            ZSTD_ParamSwitch_e::ZSTD_ps_disable,
            0
        ));
        // btlazy2 (6) → row matchfinder not supported, always allocate.
        assert!(ZSTD_allocateChainTable(
            6,
            ZSTD_ParamSwitch_e::ZSTD_ps_enable,
            0
        ));
    }

    #[test]
    fn reduceIndex_touches_hash_chain_and_hash3() {
        use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
        let cp = ZSTD_compressionParameters {
            hashLog: 4, // 16 entries → multiple of ZSTD_ROWSIZE
            chainLog: 4,
            strategy: 2, // dfast → chainTable used
            ..Default::default()
        };
        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.chainTable = vec![100u32; 16];
        ms.hashTable3 = vec![100u32; 16];
        ms.hashLog3 = 4;
        for h in ms.hashTable.iter_mut() {
            *h = 100;
        }

        ZSTD_reduceIndex(&mut ms, ZSTD_ParamSwitch_e::ZSTD_ps_disable, 0, 40);

        // All three tables shifted by 40 — entries that were 100 become 60.
        assert!(ms.hashTable.iter().all(|&x| x == 60));
        assert!(ms.chainTable.iter().all(|&x| x == 60));
        assert!(ms.hashTable3.iter().all(|&x| x == 60));
    }

    #[test]
    fn reduceIndex_skips_chain_table_for_fast_strategy() {
        use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
        let cp = ZSTD_compressionParameters {
            hashLog: 4,
            chainLog: 4,
            strategy: 1, // fast → no chain table
            ..Default::default()
        };
        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.chainTable = vec![100u32; 16];

        ZSTD_reduceIndex(&mut ms, ZSTD_ParamSwitch_e::ZSTD_ps_disable, 0, 40);
        // Fast strategy skips chain reduction.
        assert!(ms.chainTable.iter().all(|&x| x == 100));
    }

    #[test]
    fn resolveRowMatchFinderMode_respects_explicit_mode() {
        use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
        let cp = ZSTD_compressionParameters {
            strategy: 3,
            windowLog: 10,
            ..Default::default()
        };
        assert_eq!(
            ZSTD_resolveRowMatchFinderMode(ZSTD_ParamSwitch_e::ZSTD_ps_enable, &cp),
            ZSTD_ParamSwitch_e::ZSTD_ps_enable,
        );
    }

    #[test]
    fn resolveRowMatchFinderMode_auto_requires_windowLog_above_14() {
        use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
        let mut cp = ZSTD_compressionParameters {
            strategy: 3,
            windowLog: 14,
            ..Default::default()
        };
        assert_eq!(
            ZSTD_resolveRowMatchFinderMode(ZSTD_ParamSwitch_e::ZSTD_ps_auto, &cp),
            ZSTD_ParamSwitch_e::ZSTD_ps_disable,
        );
        cp.windowLog = 15;
        assert_eq!(
            ZSTD_resolveRowMatchFinderMode(ZSTD_ParamSwitch_e::ZSTD_ps_auto, &cp),
            ZSTD_ParamSwitch_e::ZSTD_ps_enable,
        );
        // Unsupported strategy → disable regardless.
        cp.strategy = 7;
        assert_eq!(
            ZSTD_resolveRowMatchFinderMode(ZSTD_ParamSwitch_e::ZSTD_ps_auto, &cp),
            ZSTD_ParamSwitch_e::ZSTD_ps_disable,
        );
    }

    #[test]
    fn resolveBlockSplitterMode_auto_gates_on_btopt_and_wlog_17() {
        use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
        let cp_lazy = ZSTD_compressionParameters {
            strategy: 3,
            windowLog: 20,
            ..Default::default()
        };
        assert_eq!(
            ZSTD_resolveBlockSplitterMode(ZSTD_ParamSwitch_e::ZSTD_ps_auto, &cp_lazy),
            ZSTD_ParamSwitch_e::ZSTD_ps_disable,
        );
        let cp_btopt_small = ZSTD_compressionParameters {
            strategy: 7,
            windowLog: 16,
            ..Default::default()
        };
        assert_eq!(
            ZSTD_resolveBlockSplitterMode(ZSTD_ParamSwitch_e::ZSTD_ps_auto, &cp_btopt_small),
            ZSTD_ParamSwitch_e::ZSTD_ps_disable,
        );
        let cp_btopt_big = ZSTD_compressionParameters {
            strategy: 7,
            windowLog: 17,
            ..Default::default()
        };
        assert_eq!(
            ZSTD_resolveBlockSplitterMode(ZSTD_ParamSwitch_e::ZSTD_ps_auto, &cp_btopt_big),
            ZSTD_ParamSwitch_e::ZSTD_ps_enable,
        );
    }

    #[test]
    fn resolveEnableLdm_auto_gates_on_wlog_27() {
        use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
        let cp_btopt_small = ZSTD_compressionParameters {
            strategy: 7,
            windowLog: 26,
            ..Default::default()
        };
        assert_eq!(
            ZSTD_resolveEnableLdm(ZSTD_ParamSwitch_e::ZSTD_ps_auto, &cp_btopt_small),
            ZSTD_ParamSwitch_e::ZSTD_ps_disable,
        );
        let cp_btopt_huge = ZSTD_compressionParameters {
            strategy: 9,
            windowLog: 27,
            ..Default::default()
        };
        assert_eq!(
            ZSTD_resolveEnableLdm(ZSTD_ParamSwitch_e::ZSTD_ps_auto, &cp_btopt_huge),
            ZSTD_ParamSwitch_e::ZSTD_ps_enable,
        );
    }

    #[test]
    fn resolveMaxBlockSize_zero_means_default() {
        use crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX;
        assert_eq!(ZSTD_resolveMaxBlockSize(0), ZSTD_BLOCKSIZE_MAX);
        assert_eq!(ZSTD_resolveMaxBlockSize(12345), 12345);
    }

    #[test]
    fn matchState_dictMode_reports_noDict_on_fresh() {
        let cp = ZSTD_compressionParameters::default();
        let ms = ZSTD_MatchState_t::new(cp);
        assert_eq!(ZSTD_matchState_dictMode(&ms), ZSTD_dictMode_e::ZSTD_noDict);
    }

    #[test]
    fn matchState_dictMode_reports_extDict_when_lowLimit_below_dictLimit() {
        let cp = ZSTD_compressionParameters::default();
        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.window.lowLimit = 5;
        ms.window.dictLimit = 10;
        assert_eq!(ZSTD_matchState_dictMode(&ms), ZSTD_dictMode_e::ZSTD_extDict);
    }

    #[test]
    fn matchState_dictMode_reports_dictMatchState_when_attached_and_live() {
        let cp = ZSTD_compressionParameters::default();
        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.dictMatchState = Some(Box::new(ZSTD_MatchState_t::new(cp)));
        ms.loadedDictEnd = 32;
        ms.window.lowLimit = 5;
        ms.window.dictLimit = 10;
        assert_eq!(
            ZSTD_matchState_dictMode(&ms),
            ZSTD_dictMode_e::ZSTD_dictMatchState
        );
    }

    #[test]
    fn resolveExternalRepcodeSearch_auto_flips_at_level_10() {
        use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
        assert_eq!(
            ZSTD_resolveExternalRepcodeSearch(ZSTD_ParamSwitch_e::ZSTD_ps_auto, 9),
            ZSTD_ParamSwitch_e::ZSTD_ps_disable,
        );
        assert_eq!(
            ZSTD_resolveExternalRepcodeSearch(ZSTD_ParamSwitch_e::ZSTD_ps_auto, 10),
            ZSTD_ParamSwitch_e::ZSTD_ps_enable,
        );
        // Explicit values pass through.
        assert_eq!(
            ZSTD_resolveExternalRepcodeSearch(ZSTD_ParamSwitch_e::ZSTD_ps_enable, 1),
            ZSTD_ParamSwitch_e::ZSTD_ps_enable,
        );
    }

    #[test]
    fn CDictIndicesAreTagged_fast_family() {
        assert!(ZSTD_CDictIndicesAreTagged(&ZSTD_compressionParameters {
            strategy: 1,
            ..Default::default()
        }));
        assert!(ZSTD_CDictIndicesAreTagged(&ZSTD_compressionParameters {
            strategy: 2,
            ..Default::default()
        }));
        assert!(!ZSTD_CDictIndicesAreTagged(&ZSTD_compressionParameters {
            strategy: 3,
            ..Default::default()
        }));
        assert!(!ZSTD_CDictIndicesAreTagged(&ZSTD_compressionParameters {
            strategy: 9,
            ..Default::default()
        }));
    }

    #[test]
    fn resolveExternalSequenceValidation_is_identity() {
        assert_eq!(ZSTD_resolveExternalSequenceValidation(0), 0);
        assert_eq!(ZSTD_resolveExternalSequenceValidation(1), 1);
        assert_eq!(ZSTD_resolveExternalSequenceValidation(42), 42);
    }

    #[test]
    fn bitmix_is_non_identity() {
        assert_ne!(ZSTD_bitmix(0xDEADBEEF, 8), 0xDEADBEEF);
        // Determinism.
        assert_eq!(ZSTD_bitmix(42, 8), ZSTD_bitmix(42, 8));
    }

    #[test]
    fn advanceHashSalt_changes_salt() {
        let cp = ZSTD_compressionParameters::default();
        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.hashSaltEntropy = 0x1234;
        let before = ms.hashSalt;
        ZSTD_advanceHashSalt(&mut ms);
        assert_ne!(ms.hashSalt, before);
    }

    #[test]
    fn invalidateMatchState_resets_window_and_nextToUpdate() {
        let cp = ZSTD_compressionParameters::default();
        let mut ms = ZSTD_MatchState_t::new(cp);
        // Pretend we processed a bunch of input.
        ms.window.nextSrc = 500;
        ms.window.dictLimit = 50;
        ms.window.lowLimit = 25;
        ms.nextToUpdate = 400;
        ms.loadedDictEnd = 99;
        ZSTD_invalidateMatchState(&mut ms);
        // window_clear collapses dictLimit/lowLimit to nextSrc = 500.
        assert_eq!(ms.window.dictLimit, 500);
        assert_eq!(ms.window.lowLimit, 500);
        assert_eq!(ms.nextToUpdate, 500); // tracks dictLimit
        assert_eq!(ms.loadedDictEnd, 0);
    }

    #[test]
    fn reset_preserves_allocation_but_zeros_table() {
        let cp = ZSTD_compressionParameters {
            hashLog: 10,
            minMatch: 4,
            ..Default::default()
        };
        let mut ms = ZSTD_MatchState_t::new(cp);
        for c in ms.hashTable.iter_mut().take(16) {
            *c = 0xdead;
        }
        ms.nextToUpdate = 100;
        ms.reset();
        assert!(ms.hashTable.iter().all(|&c| c == 0));
        assert_eq!(ms.nextToUpdate, ZSTD_WINDOW_START_INDEX);
        assert_eq!(ms.hashTable.len(), 1 << 10);
    }
}
