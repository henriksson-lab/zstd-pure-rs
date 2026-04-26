//! Translation of `lib/compress/zstd_fast.c` (strategy 1). Single-
//! hash-table match finder. Both phases are ported:
//!   - `ZSTD_fillHashTable` seeds the hash with the start of the
//!     block,
//!   - `ZSTD_compressBlock_fast` runs the main emit loop.
//!
//! Rust signature note: upstream's `end` argument is a `void*` into
//! the caller's source buffer; we accept an explicit `&[u8]` slice
//! representing the full source region.

#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]

use crate::common::mem::MEM_read32;
use crate::compress::match_state::{
    ZSTD_MatchState_t, ZSTD_comparePackedTags, ZSTD_index_overlap_check, ZSTD_window_hasExtDict,
    ZSTD_writeTaggedIndex, ZSTD_SHORT_CACHE_TAG_BITS,
};
use crate::compress::seq_store::{
    SeqStore_t, ZSTD_storeSeq, OFFSET_TO_OFFBASE, REPCODE_TO_OFFBASE, ZSTD_REP_NUM,
};
use crate::compress::zstd_hashes::{
    ZSTD_count, ZSTD_count_2segments, ZSTD_hashPtr, ZSTD_hashPtr_at_unchecked,
};

/// Upstream `kSearchStrength` — controls how aggressively the
/// match finder widens its step when mismatches are common.
pub const kSearchStrength: u32 = 8;

/// Port of `ZSTD_getLowestPrefixIndex`. Given the current index, the
/// match state, and the active windowLog, returns the smallest
/// back-reference index the encoder may emit without crossing out of
/// the window.
pub fn ZSTD_getLowestPrefixIndex(ms: &ZSTD_MatchState_t, curr: u32, windowLog: u32) -> u32 {
    let maxDistance = 1u32 << windowLog;
    let lowestValid = ms.window.dictLimit;
    let withinWindow = if curr.wrapping_sub(lowestValid) > maxDistance {
        curr.wrapping_sub(maxDistance)
    } else {
        lowestValid
    };
    let isDictionary = ms.loadedDictEnd != 0;
    if isDictionary {
        lowestValid
    } else {
        withinWindow
    }
}

/// Port of `ZSTD_getLowestMatchIndex`. Like `getLowestPrefixIndex`
/// but anchored at `lowLimit` rather than `dictLimit`, so ext-dict
/// matches below the prefix stay valid too. Used by matchers that
/// can reference the ext-dict (opt parser, lazy+dictMatchState).
pub fn ZSTD_getLowestMatchIndex(ms: &ZSTD_MatchState_t, curr: u32, windowLog: u32) -> u32 {
    let maxDistance = 1u32 << windowLog;
    let lowestValid = ms.window.lowLimit;
    let withinWindow = if curr.wrapping_sub(lowestValid) > maxDistance {
        curr.wrapping_sub(maxDistance)
    } else {
        lowestValid
    };
    if ms.loadedDictEnd != 0 {
        lowestValid
    } else {
        withinWindow
    }
}

/// Port of `ZSTD_match4Found_branch` — validates a candidate match
/// via an explicit early-out branch + 32-bit compare. Faster than the
/// CMOV-friendly variant when `match_idx >= idx_low_limit` is
/// well-predicted (typical case for large windows where matches
/// usually fall within window). Upstream picks this form when
/// `windowLog >= 19`.
#[inline(always)]
pub fn ZSTD_match4Found_branch(
    buf: &[u8],
    current_pos: usize,
    match_pos: usize,
    match_idx: u32,
    idx_low_limit: u32,
) -> bool {
    if match_idx < idx_low_limit {
        return false;
    }
    // SAFETY: when `match_idx >= idx_low_limit` the caller's loop
    // invariant guarantees `match_pos + 4 ≤ buf.len()` (positions
    // come from the matcher's hashTable for indices ≥ prefixStart).
    // `current_pos` is the matcher's `ip0` ≤ `ilimit < buf.len() - 4`.
    let m = unsafe { (buf.as_ptr().wrapping_add(match_pos) as *const u32).read_unaligned() };
    let c = unsafe { (buf.as_ptr().wrapping_add(current_pos) as *const u32).read_unaligned() };
    m == c
}

/// Port of `ZSTD_match4Found_cmov`. Mirrors upstream's CMOV-friendly
/// formulation: load 4 bytes either from `match_pos` (when in-range)
/// or from a fixed dummy buffer; then compare. The select-then-load
/// pattern compiles to CMOV on x86-64, replacing the branch on
/// `match_idx < idx_low_limit` with a predicated address swap. The
/// hot fast-path in `ZSTD_compressBlock_fast_noDict_generic` calls
/// this thousands of times per block; eliminating the unpredictable
/// branch is a measurable win when the matchTable returns stale
/// out-of-window indices.
#[inline(always)]
pub fn ZSTD_match4Found_cmov(
    buf: &[u8],
    current_pos: usize,
    match_pos: usize,
    match_idx: u32,
    idx_low_limit: u32,
) -> bool {
    static DUMMY: [u8; 4] = [0x12, 0x34, 0x56, 0x78];
    // Mirror upstream's `ZSTD_selectAddr(matchIdx, idxLowLimit, real,
    // dummy)`: pick the read source via a CMOV on the index-in-range
    // predicate, then unconditionally load 4 bytes from it. The slice-
    // bounds-check version emits a `cmp/jb` pair before each load,
    // breaking CMOV codegen. Caller ensures: when
    // `match_idx >= idx_low_limit`, `match_pos + 4 ≤ buf.len()`; the
    // L1 fast-matcher loop invariant `ip3 < ilimit = iend - 8` and the
    // matcher's prefix-check both uphold this. `current_pos` is `ip0`
    // which is similarly ≤ ilimit < buf.len() - 4.
    let real_ptr = buf.as_ptr().wrapping_add(match_pos);
    let dummy_ptr = DUMMY.as_ptr();
    let m_ptr = if match_idx >= idx_low_limit {
        real_ptr
    } else {
        dummy_ptr
    };
    let c_ptr = buf.as_ptr().wrapping_add(current_pos);
    // SAFETY: see comment above. `m_ptr` is either DUMMY (4 bytes
    // valid) or `buf[match_pos..match_pos+4]` (in-range when the
    // predicate is true). `c_ptr` is `buf[current_pos..current_pos+4]`
    // which is in-range by the caller's loop invariant.
    let m = unsafe { (m_ptr as *const u32).read_unaligned() };
    let c = unsafe { (c_ptr as *const u32).read_unaligned() };
    if m != c {
        return false;
    }
    match_idx >= idx_low_limit
}

/// Upstream `HASH_READ_SIZE` — the hash reads 8 bytes at a time.
pub const HASH_READ_SIZE: usize = 8;

/// Port of `ZSTD_dictTableLoadMethod_e`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZSTD_dictTableLoadMethod_e {
    ZSTD_dtlm_fast,
    ZSTD_dtlm_full,
}

/// Port of `ZSTD_tableFillPurpose_e`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZSTD_tableFillPurpose_e {
    ZSTD_tfp_forCCtx,
    ZSTD_tfp_forCDict,
}

/// Port of `ZSTD_fillHashTableForCDict`. Called when seeding a
/// digested dictionary; uses the short-cache tagged-index format and
/// typically runs with `ZSTD_dtlm_full` to load extras.
pub fn ZSTD_fillHashTableForCDict(
    ms: &mut ZSTD_MatchState_t,
    src: &[u8],
    dtlm: ZSTD_dictTableLoadMethod_e,
) {
    let hBits = ms.cParams.hashLog + ZSTD_SHORT_CACHE_TAG_BITS;
    let mls = ms.cParams.minMatch;
    let fastHashFillStep: u32 = 3;

    let src_end = src.len();
    if src_end < HASH_READ_SIZE {
        return;
    }
    let iend = src_end - HASH_READ_SIZE;

    let mut ip = ms.nextToUpdate.saturating_sub(ms.window.base_offset) as usize;
    while ip + fastHashFillStep as usize + 1 < iend + 2 {
        let curr = ms.window.base_offset.wrapping_add(ip as u32);
        let hash0 = ZSTD_hashPtr(&src[ip..], hBits, mls);
        ZSTD_writeTaggedIndex(&mut ms.hashTable, hash0, curr);
        if dtlm != ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_fast {
            for p in 1..fastHashFillStep as usize {
                let hash = ZSTD_hashPtr(&src[ip + p..], hBits, mls);
                let slot = hash >> ZSTD_SHORT_CACHE_TAG_BITS;
                if ms.hashTable[slot] == 0 {
                    ZSTD_writeTaggedIndex(&mut ms.hashTable, hash, curr.wrapping_add(p as u32));
                }
            }
        }
        ip += fastHashFillStep as usize;
    }
}

/// Port of `ZSTD_fillHashTableForCCtx`. Called for a fresh compression
/// context; always uses `ZSTD_dtlm_fast`.
pub fn ZSTD_fillHashTableForCCtx(
    ms: &mut ZSTD_MatchState_t,
    src: &[u8],
    _dtlm: ZSTD_dictTableLoadMethod_e,
) {
    let hBits = ms.cParams.hashLog;
    let mls = ms.cParams.minMatch;
    let fastHashFillStep: u32 = 3;

    let src_end = src.len();
    if src_end < HASH_READ_SIZE {
        return;
    }
    let iend = src_end - HASH_READ_SIZE;

    let mut ip = ms.nextToUpdate.saturating_sub(ms.window.base_offset) as usize;
    while ip + fastHashFillStep as usize + 1 < iend + 2 {
        let curr = ms.window.base_offset.wrapping_add(ip as u32);
        let hash0 = ZSTD_hashPtr(&src[ip..], hBits, mls);
        ms.hashTable[hash0] = curr;
        // Fast variant: no extra-slot fill.
        ip += fastHashFillStep as usize;
    }
}

/// Port of `ZSTD_compressBlock_fast_noDict_generic`. Mirrors
/// upstream's four-cursor (`ip0..ip3`) pipelined scan, including the
/// repcode-at-`ip2` fast path and the staggered hash-table updates
/// that keep the pipeline coherent across match exits.
///
/// Scans `src[istart..]`, treating `src[0..istart]` as already-
/// processed history (the hashTable entries for those positions
/// remain valid back-references). Emits sequences + literals into
/// `seqStore` from `anchor=istart`. `ms.window.base_offset` is the
/// absolute index offset: positions
/// `[base_off+0..base_off+src.len()]` identify bytes in the source.
///
/// Returns the length of the final "tail" literals segment (bytes
/// after the last emitted sequence that the entropy stage will
/// copy verbatim).
pub fn ZSTD_compressBlock_fast_noDict_generic(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
    istart: usize,
    mls: u32,
) -> usize {
    // Dispatch on (`mls`, `useCmov`). Upstream's `ZSTD_GEN_FAST_FN`
    // generates 4×2 = 8 specializations: 4 minMatch values × 2 match
    // predicates (CMOV-friendly select-then-load vs explicit branch).
    // The CMOV form is faster when the `matchIdx >= idx_low_limit`
    // predicate is unpredictable (small windows). The branch form is
    // faster when the predicate is well-predicted (large windows where
    // matches are typically in-window). Upstream picks the branch
    // variant when `windowLog >= 19`, which is the common case for
    // multi-MB inputs like silesia.
    let use_cmov = ms.cParams.windowLog < 19;
    if use_cmov {
        match mls {
            5 => ZSTD_compressBlock_fast_noDict_generic_mls::<5, true>(
                ms, seqStore, rep, src, istart,
            ),
            6 => ZSTD_compressBlock_fast_noDict_generic_mls::<6, true>(
                ms, seqStore, rep, src, istart,
            ),
            7 => ZSTD_compressBlock_fast_noDict_generic_mls::<7, true>(
                ms, seqStore, rep, src, istart,
            ),
            _ => ZSTD_compressBlock_fast_noDict_generic_mls::<4, true>(
                ms, seqStore, rep, src, istart,
            ),
        }
    } else {
        match mls {
            5 => ZSTD_compressBlock_fast_noDict_generic_mls::<5, false>(
                ms, seqStore, rep, src, istart,
            ),
            6 => ZSTD_compressBlock_fast_noDict_generic_mls::<6, false>(
                ms, seqStore, rep, src, istart,
            ),
            7 => ZSTD_compressBlock_fast_noDict_generic_mls::<7, false>(
                ms, seqStore, rep, src, istart,
            ),
            _ => ZSTD_compressBlock_fast_noDict_generic_mls::<4, false>(
                ms, seqStore, rep, src, istart,
            ),
        }
    }
}

#[inline(never)]
fn ZSTD_compressBlock_fast_noDict_generic_mls<const MLS: u32, const USE_CMOV: bool>(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
    istart: usize,
) -> usize {
    let hlog = ms.cParams.hashLog;
    let windowLog = ms.cParams.windowLog;
    let stepSize = ms
        .cParams
        .targetLength
        .wrapping_add((ms.cParams.targetLength == 0) as u32)
        .wrapping_add(1) as usize;

    let base_off = ms.window.base_offset;
    let srcSize = src.len();
    let endIndex = base_off.wrapping_add(srcSize as u32);
    let prefixStartIndex = ZSTD_getLowestPrefixIndex(ms, endIndex, windowLog);
    let prefixStart = prefixStartIndex.saturating_sub(base_off) as usize;
    let iend = srcSize;
    let ilimit = iend.saturating_sub(HASH_READ_SIZE);

    let mut anchor = istart;
    let mut ip0 = istart;
    // Upstream: skip byte 0 so the first match candidate can look back.
    if ip0 == prefixStart {
        ip0 += 1;
    }

    let mut rep_offset1 = rep[0];
    let mut rep_offset2 = rep[1];
    let mut offsetSaved1: u32 = 0;
    let mut offsetSaved2: u32 = 0;
    {
        let curr = base_off.wrapping_add(ip0 as u32);
        let windowLow = ZSTD_getLowestPrefixIndex(ms, curr, windowLog);
        let maxRep = curr.wrapping_sub(windowLow);
        if rep_offset2 > maxRep {
            offsetSaved2 = rep_offset2;
            rep_offset2 = 0;
        }
        if rep_offset1 > maxRep {
            offsetSaved1 = rep_offset1;
            rep_offset1 = 0;
        }
    }

    let kStepIncr = 1usize << (kSearchStrength - 1);

    // Hoist `ms.hashTable` into a local `&mut [u32]` for the duration
    // of the search+emit loop. Mirrors upstream's
    // `U32* const hashTable = ms->hashTable;` at function entry.
    // Without this, every per-byte `ms.hashTable[h]` chases the Vec
    // pointer through the matchState struct each iteration —
    // measurable overhead on the L1 inner loop.
    let hashTable = ms.hashTable.as_mut_ptr();

    'search: loop {
        let mut step = stepSize;
        let mut nextStep = ip0 + kStepIncr;
        let mut ip1 = ip0 + 1;
        let mut ip2 = ip0 + step;
        let mut ip3 = ip2 + 1;

        if ip3 >= ilimit {
            break 'search;
        }

        // SAFETY: outer-loop check `ip3 < ilimit = iend - 8` gives
        // `ip0 < ilimit - 1 < iend - 8`, so `ip0 + 8 < iend = src.len()`.
        // `ip1 = ip0 + 1` is similarly bounded. `MLS` is in {4,5,6,7}
        // — for MLS=4 we need 4 bytes, else 8.
        let mut hash0 = unsafe { ZSTD_hashPtr_at_unchecked(src, ip0, hlog, MLS) };
        let mut hash1 = unsafe { ZSTD_hashPtr_at_unchecked(src, ip1, hlog, MLS) };
        // SAFETY: every hash value is bounded by `1 << hlog`, and
        // `hashTable.len() == 1 << hlog` by allocation. The slice
        // bounds check is a per-iter `cmp 0x20(%rsp), %rax; jae`
        // pair (~7% of L1 cycles) that LLVM can't elide because it
        // doesn't track the post-shift hash value range.
        let mut matchIdx = unsafe { *hashTable.add(hash0) };
        let mut current0: u32;

        // Search-only inner loop. Mirrors upstream's structure where a
        // `goto _offset` / `goto _match` from the search body lands at a
        // single shared emit block outside the loop. By breaking out of
        // this loop with the match info as a tuple, the search loop body
        // is ~50 lines (vs ~250 lines with three inlined match-emit tails),
        // shrinking the I-cache footprint of the hot search path.
        let (match0, mut mLength, offcode) = loop {
            current0 = base_off.wrapping_add(ip0 as u32);
            // SAFETY: hash0 < (1 << hlog) == hashTable.len(), by hash
            // construction. See top-of-loop comment on `matchIdx`.
            unsafe {
                *hashTable.add(hash0) = current0;
            }

            // Repcode check at ip2 — branch-light: both reads issued
            // unconditionally via raw-pointer loads so the compiler
            // can CMOV. When `rep_offset1` is invalid we substitute
            // `ip2` for the read source; the compare-against-self is
            // masked off by `rep_safe`. Slice-indexed `MEM_read32`
            // would emit a bounds-check `cmp/jb` per read, defeating
            // CMOV codegen.
            // SAFETY: loop invariant `ip3 < ilimit = iend - 8` gives
            // `ip2 + 4 < iend - 4 ≤ src.len()`; `rep_read_pos ≤ ip2`.
            let rep_safe = rep_offset1 > 0 && ip2 >= rep_offset1 as usize;
            let rep_read_pos = if rep_safe {
                ip2 - rep_offset1 as usize
            } else {
                ip2
            };
            let rep_rval =
                unsafe { (src.as_ptr().wrapping_add(rep_read_pos) as *const u32).read_unaligned() };
            let rep_cval =
                unsafe { (src.as_ptr().wrapping_add(ip2) as *const u32).read_unaligned() };
            if rep_safe & (rep_rval == rep_cval) {
                ip0 = ip2;
                let mut match0 = ip0 - rep_offset1 as usize;
                let back = usize::from(ip0 > 0 && match0 > 0 && src[ip0 - 1] == src[match0 - 1]);
                ip0 -= back;
                match0 -= back;
                // SAFETY: hash1 < hashTable.len() as above.
                unsafe {
                    *hashTable.add(hash1) = base_off.wrapping_add(ip1 as u32);
                }
                break (match0, back + 4, REPCODE_TO_OFFBASE(1));
            }

            // First hash check at ip0. `USE_CMOV` selects between the
            // branchless (CMOV-friendly) and explicit-branch predicate
            // — upstream uses branch for large windows where the
            // `matchIdx >= prefixStartIndex` test is well-predicted.
            let match0_pos = matchIdx.saturating_sub(base_off) as usize;
            let match0_found = if USE_CMOV {
                ZSTD_match4Found_cmov(src, ip0, match0_pos, matchIdx, prefixStartIndex)
            } else {
                ZSTD_match4Found_branch(src, ip0, match0_pos, matchIdx, prefixStartIndex)
            };
            if match0_found && matchIdx < base_off.wrapping_add(ip0 as u32) {
                unsafe {
                    *hashTable.add(hash1) = base_off.wrapping_add(ip1 as u32);
                }
                let mut match0 = match0_pos;
                rep_offset2 = rep_offset1;
                rep_offset1 = (ip0 - match0) as u32;
                let offcode = OFFSET_TO_OFFBASE(rep_offset1);
                let mut mLength = 4;
                while ip0 > anchor && match0 > prefixStart && src[ip0 - 1] == src[match0 - 1] {
                    ip0 -= 1;
                    match0 -= 1;
                    mLength += 1;
                }
                break (match0, mLength, offcode);
            }

            // Shift positions for the second hash check.
            matchIdx = unsafe { *hashTable.add(hash1) };
            hash0 = hash1;
            // SAFETY: `ip2 < ilimit = iend - 8` by loop invariant, so
            // `ip2 + 8 < iend = src.len()`.
            hash1 = unsafe { ZSTD_hashPtr_at_unchecked(src, ip2, hlog, MLS) };
            ip0 = ip1;
            ip1 = ip2;
            ip2 = ip3;

            current0 = base_off.wrapping_add(ip0 as u32);
            unsafe {
                *hashTable.add(hash0) = current0;
            }

            // Second hash check at the shifted ip0.
            let match0_pos2 = matchIdx.saturating_sub(base_off) as usize;
            let match0_found2 = if USE_CMOV {
                ZSTD_match4Found_cmov(src, ip0, match0_pos2, matchIdx, prefixStartIndex)
            } else {
                ZSTD_match4Found_branch(src, ip0, match0_pos2, matchIdx, prefixStartIndex)
            };
            if match0_found2 && matchIdx < base_off.wrapping_add(ip0 as u32) {
                if step <= 4 {
                    unsafe {
                        *hashTable.add(hash1) = base_off.wrapping_add(ip1 as u32);
                    }
                }
                let mut match0 = matchIdx.saturating_sub(base_off) as usize;
                rep_offset2 = rep_offset1;
                rep_offset1 = (ip0 - match0) as u32;
                let offcode = OFFSET_TO_OFFBASE(rep_offset1);
                let mut mLength = 4;
                while ip0 > anchor && match0 > prefixStart && src[ip0 - 1] == src[match0 - 1] {
                    ip0 -= 1;
                    match0 -= 1;
                    mLength += 1;
                }
                break (match0, mLength, offcode);
            }

            // No match: step-advance and loop again. When we reach
            // `ilimit`, exit the outer search entirely (skipping emit).
            matchIdx = unsafe { *hashTable.add(hash1) };
            hash0 = hash1;
            // SAFETY: `ip2 < ilimit = iend - 8` by loop invariant.
            hash1 = unsafe { ZSTD_hashPtr_at_unchecked(src, ip2, hlog, MLS) };
            ip0 = ip1;
            ip1 = ip2;
            ip2 = ip0 + step;
            ip3 = ip1 + step;

            if ip2 >= nextStep {
                step += 1;
                crate::common::zstd_internal::prefetchSliceByte(src, ip1 + 64);
                crate::common::zstd_internal::prefetchSliceByte(src, ip1 + 128);
                nextStep += kStepIncr;
            }
            if ip3 >= ilimit {
                break 'search;
            }
        };

        // Shared emit tail (mirrors upstream's `_match` label).
        mLength += ZSTD_count(src, ip0 + mLength, match0 + mLength, iend);
        // `&src[anchor..]` (not `..ip0`) gives storeSeq's wildcopy a
        // 16-byte over-read window past the literal end.
        ZSTD_storeSeq(seqStore, ip0 - anchor, &src[anchor..], offcode, mLength);
        ip0 += mLength;
        anchor = ip0;
        if ip0 <= ilimit {
            // Hash table fill at curr0+2 (for next iter's first hash).
            // SAFETY: bounds-checked at the outer if; we keep that
            // check but skip the inner slice bounds check via the
            // unchecked hash variant.
            let curr_plus2_pos = current0.wrapping_sub(base_off).wrapping_add(2) as usize;
            if curr_plus2_pos + 8 <= src.len() {
                let h = unsafe { ZSTD_hashPtr_at_unchecked(src, curr_plus2_pos, hlog, MLS) };
                unsafe {
                    *hashTable.add(h) = current0.wrapping_add(2);
                }
            }
            if ip0 >= 2 && ip0 - 2 + 8 <= src.len() {
                let h = unsafe { ZSTD_hashPtr_at_unchecked(src, ip0 - 2, hlog, MLS) };
                unsafe {
                    *hashTable.add(h) = base_off.wrapping_add((ip0 - 2) as u32);
                }
            }
            while ip0 <= ilimit && rep_offset2 > 0 && ip0 >= rep_offset2 as usize {
                // SAFETY: `ip0 ≤ ilimit = iend - 8`, so `ip0 + 4 ≤ iend
                // = src.len()`; `ip0 - rep_offset2 ≤ ip0` so same bound.
                let cval =
                    unsafe { (src.as_ptr().wrapping_add(ip0) as *const u32).read_unaligned() };
                let rval = unsafe {
                    (src.as_ptr().wrapping_add(ip0 - rep_offset2 as usize) as *const u32)
                        .read_unaligned()
                };
                if cval != rval {
                    break;
                }
                let rLength = ZSTD_count(src, ip0 + 4, ip0 + 4 - rep_offset2 as usize, iend) + 4;
                std::mem::swap(&mut rep_offset1, &mut rep_offset2);
                // SAFETY: `ip0 ≤ ilimit < iend = src.len()`.
                let h = unsafe { ZSTD_hashPtr_at_unchecked(src, ip0, hlog, MLS) };
                // SAFETY: h < (1 << hlog) == hashTable.len(), by hash construction.
                unsafe {
                    *hashTable.add(h) = base_off.wrapping_add(ip0 as u32);
                }
                ZSTD_storeSeq(
                    seqStore,
                    0,
                    &src[anchor..anchor],
                    REPCODE_TO_OFFBASE(1),
                    rLength,
                );
                ip0 += rLength;
                anchor = ip0;
            }
        }
    }

    // Save repcodes for the next block, restoring any we zeroed in
    // the preamble.
    let offsetSaved2_final = if offsetSaved1 != 0 && rep_offset1 != 0 {
        offsetSaved1
    } else {
        offsetSaved2
    };
    rep[0] = if rep_offset1 != 0 {
        rep_offset1
    } else {
        offsetSaved1
    };
    rep[1] = if rep_offset2 != 0 {
        rep_offset2
    } else {
        offsetSaved2_final
    };

    iend - anchor
}

/// Port of `ZSTD_compressBlock_fast`. The public no-dict entry:
/// dispatches on `cParams.minMatch`, with dedicated dict-match-state
/// and ext-dict siblings implemented separately below.
pub fn ZSTD_compressBlock_fast(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    let mml = ms.cParams.minMatch;
    ZSTD_compressBlock_fast_noDict_generic(ms, seqStore, rep, src, 0, mml)
}

pub fn ZSTD_compressBlock_fast_dictMatchState(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_fast_dictMatchState_generic(ms, seqStore, rep, src)
}

/// Port of `ZSTD_compressBlock_fast_dictMatchState_generic`.
pub fn ZSTD_compressBlock_fast_dictMatchState_generic(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    let cParams = ms.cParams;
    let mls = cParams.minMatch;
    let stepSize = cParams.targetLength + u32::from(cParams.targetLength == 0);
    let prefixStartIndex = ms.window.dictLimit;
    let prefixStart = prefixStartIndex.saturating_sub(ms.window.base_offset) as usize;
    let endIndex = ms.window.base_offset.wrapping_add(src.len() as u32);
    let dms = match ms.dictMatchState.as_deref() {
        Some(dms) => dms,
        None => {
            return ZSTD_compressBlock_fast_noDict_generic(ms, seqStore, rep, src, 0, mls);
        }
    };
    let dictStartIndex = dms.window.dictLimit;
    let dictEndIndex = dms.window.nextSrc;
    let dictBaseOffset = dms.window.base_offset;
    let dict = &dms.dictContent;
    let dictStart = dictStartIndex.saturating_sub(dictBaseOffset) as usize;
    let dictSize = dictEndIndex.saturating_sub(dictBaseOffset);
    let dictIndexDelta = prefixStartIndex.saturating_sub(dictSize);
    let dictAndPrefixLength =
        (src.len() as u32).wrapping_add(dictEndIndex.saturating_sub(dictStartIndex));
    let dictHBits = dms.cParams.hashLog + ZSTD_SHORT_CACHE_TAG_BITS;
    let maxDistance = if cParams.windowLog >= 31 {
        u32::MAX
    } else {
        1u32 << cParams.windowLog
    };

    if endIndex.wrapping_sub(prefixStartIndex) > maxDistance {
        return ZSTD_compressBlock_fast_noDict_generic(ms, seqStore, rep, src, 0, mls);
    }
    if ZSTD_window_hasExtDict(&ms.window) {
        return ZSTD_compressBlock_fast_extDict_generic(ms, seqStore, rep, src);
    }
    if src.len() < HASH_READ_SIZE || stepSize == 0 {
        return src.len();
    }
    if dictAndPrefixLength == 0 {
        return ZSTD_compressBlock_fast_noDict_generic(ms, seqStore, rep, src, 0, mls);
    }
    if rep[0] > dictAndPrefixLength {
        rep[0] = dictAndPrefixLength;
    }
    if rep[1] > dictAndPrefixLength {
        rep[1] = dictAndPrefixLength;
    }

    if ms.prefetchCDictTables {
        crate::common::zstd_internal::ZSTD_prefetchArea(&dms.hashTable);
    }

    let hlog = cParams.hashLog;
    let base_off = ms.window.base_offset;
    let iend = src.len();
    let ilimit = iend.saturating_sub(HASH_READ_SIZE);
    let mut offset_1 = rep[0];
    let mut offset_2 = rep[1];
    let mut anchor = 0usize;
    let mut ip0 = usize::from(dictAndPrefixLength == 0);
    let mut ip1 = ip0 + stepSize as usize;

    while ip1 <= ilimit {
        let mut mLength: usize;
        let mut hash0 = ZSTD_hashPtr(&src[ip0..], hlog, mls);
        let dictHashAndTag0 = ZSTD_hashPtr(&src[ip0..], dictHBits, mls);
        let mut dictMatchIndexAndTag = dms.hashTable[dictHashAndTag0 >> ZSTD_SHORT_CACHE_TAG_BITS];
        let mut dictTagsMatch =
            ZSTD_comparePackedTags(dictMatchIndexAndTag as usize, dictHashAndTag0);
        let mut matchIndex = ms.hashTable[hash0];
        let mut curr = base_off.wrapping_add(ip0 as u32);
        let mut step = stepSize as usize;
        let kStepIncr = 1usize << kSearchStrength;
        let mut nextStep = ip0 + kStepIncr;

        loop {
            let match_pos = matchIndex.saturating_sub(base_off) as usize;
            let repIndex = curr.wrapping_add(1).wrapping_sub(offset_1);
            let repInDict = repIndex < prefixStartIndex;
            let repMatch = if repInDict {
                repIndex
                    .saturating_sub(dictIndexDelta)
                    .saturating_sub(dictBaseOffset) as usize
            } else {
                repIndex.saturating_sub(base_off) as usize
            };
            let hash1 = ZSTD_hashPtr(&src[ip1..], hlog, mls);
            let dictHashAndTag1 = ZSTD_hashPtr(&src[ip1..], dictHBits, mls);
            ms.hashTable[hash0] = curr;

            if ZSTD_index_overlap_check(prefixStartIndex, repIndex)
                && if repInDict {
                    repMatch + 4 <= dict.len()
                        && MEM_read32(&dict[repMatch..]) == MEM_read32(&src[ip0 + 1..])
                } else {
                    repMatch + 4 <= src.len()
                        && MEM_read32(&src[repMatch..]) == MEM_read32(&src[ip0 + 1..])
                }
            {
                let repMatchEnd = if repInDict { dict.len() } else { iend };
                mLength = ZSTD_count_2segments(
                    src,
                    ip0 + 1 + 4,
                    iend,
                    prefixStart,
                    if repInDict { dict } else { src },
                    repMatch + 4,
                    repMatchEnd,
                ) + 4;
                ip0 += 1;
                ZSTD_storeSeq(
                    seqStore,
                    ip0 - anchor,
                    &src[anchor..ip0],
                    REPCODE_TO_OFFBASE(1),
                    mLength,
                );
                break;
            }

            if dictTagsMatch {
                let dictMatchIndex = dictMatchIndexAndTag >> ZSTD_SHORT_CACHE_TAG_BITS;
                let dictMatch = dictMatchIndex.saturating_sub(dictBaseOffset) as usize;
                if dictMatchIndex > dictStartIndex
                    && dictMatch + 4 <= dict.len()
                    && MEM_read32(&dict[dictMatch..]) == MEM_read32(&src[ip0..])
                    && matchIndex <= prefixStartIndex
                {
                    let offset = curr
                        .wrapping_sub(dictMatchIndex)
                        .wrapping_sub(dictIndexDelta);
                    mLength = ZSTD_count_2segments(
                        src,
                        ip0 + 4,
                        iend,
                        prefixStart,
                        dict,
                        dictMatch + 4,
                        dict.len(),
                    ) + 4;
                    let mut catch_ip = ip0;
                    let mut catch_dict = dictMatch;
                    while catch_ip > anchor
                        && catch_dict > dictStart
                        && src[catch_ip - 1] == dict[catch_dict - 1]
                    {
                        catch_ip -= 1;
                        catch_dict -= 1;
                        mLength += 1;
                    }
                    ip0 = catch_ip;
                    offset_2 = offset_1;
                    offset_1 = offset;
                    ZSTD_storeSeq(
                        seqStore,
                        ip0 - anchor,
                        &src[anchor..ip0],
                        OFFSET_TO_OFFBASE(offset),
                        mLength,
                    );
                    break;
                }
            }

            if match_pos + 4 <= src.len()
                && ZSTD_match4Found_cmov(src, ip0, match_pos, matchIndex, prefixStartIndex)
            {
                let offset = curr.wrapping_sub(matchIndex);
                mLength = ZSTD_count(src, ip0 + 4, match_pos + 4, iend) + 4;
                let mut catch_ip = ip0;
                let mut catch_match = match_pos;
                while catch_ip > anchor
                    && catch_match > prefixStart
                    && src[catch_ip - 1] == src[catch_match - 1]
                {
                    catch_ip -= 1;
                    catch_match -= 1;
                    mLength += 1;
                }
                ip0 = catch_ip;
                offset_2 = offset_1;
                offset_1 = offset;
                ZSTD_storeSeq(
                    seqStore,
                    ip0 - anchor,
                    &src[anchor..ip0],
                    OFFSET_TO_OFFBASE(offset),
                    mLength,
                );
                break;
            }

            dictMatchIndexAndTag = dms.hashTable[dictHashAndTag1 >> ZSTD_SHORT_CACHE_TAG_BITS];
            dictTagsMatch = ZSTD_comparePackedTags(dictMatchIndexAndTag as usize, dictHashAndTag1);
            matchIndex = ms.hashTable[hash1];

            if ip1 >= nextStep {
                step += 1;
                nextStep += kStepIncr;
            }
            ip0 = ip1;
            ip1 += step;
            if ip1 > ilimit {
                rep[0] = offset_1;
                rep[1] = offset_2;
                return iend - anchor;
            }

            curr = base_off.wrapping_add(ip0 as u32);
            hash0 = hash1;
        }

        debug_assert!(mLength > 0);
        ip0 += mLength;
        anchor = ip0;

        if ip0 <= ilimit {
            ms.hashTable[ZSTD_hashPtr(
                &src[(curr.wrapping_add(2).wrapping_sub(base_off)) as usize..],
                hlog,
                mls,
            )] = curr.wrapping_add(2);
            ms.hashTable[ZSTD_hashPtr(&src[ip0 - 2..], hlog, mls)] =
                base_off.wrapping_add((ip0 - 2) as u32);

            while ip0 <= ilimit {
                let current2 = base_off.wrapping_add(ip0 as u32);
                let repIndex2 = current2.wrapping_sub(offset_2);
                let repInDict = repIndex2 < prefixStartIndex;
                let repMatch2 = if repInDict {
                    repIndex2
                        .saturating_sub(dictIndexDelta)
                        .saturating_sub(dictBaseOffset) as usize
                } else {
                    repIndex2.saturating_sub(base_off) as usize
                };
                if ZSTD_index_overlap_check(prefixStartIndex, repIndex2)
                    && if repInDict {
                        repMatch2 + 4 <= dict.len()
                            && MEM_read32(&dict[repMatch2..]) == MEM_read32(&src[ip0..])
                    } else {
                        repMatch2 + 4 <= src.len()
                            && MEM_read32(&src[repMatch2..]) == MEM_read32(&src[ip0..])
                    }
                {
                    let repEnd2 = if repInDict { dict.len() } else { iend };
                    let repLength2 = ZSTD_count_2segments(
                        src,
                        ip0 + 4,
                        iend,
                        prefixStart,
                        if repInDict { dict } else { src },
                        repMatch2 + 4,
                        repEnd2,
                    ) + 4;
                    std::mem::swap(&mut offset_2, &mut offset_1);
                    ZSTD_storeSeq(
                        seqStore,
                        0,
                        &src[anchor..anchor],
                        REPCODE_TO_OFFBASE(1),
                        repLength2,
                    );
                    ms.hashTable[ZSTD_hashPtr(&src[ip0..], hlog, mls)] = current2;
                    ip0 += repLength2;
                    anchor = ip0;
                    continue;
                }
                break;
            }
        }

        ip1 = ip0 + stepSize as usize;
    }

    rep[0] = offset_1;
    rep[1] = offset_2;
    iend - anchor
}

pub fn ZSTD_compressBlock_fast_extDict(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_fast_extDict_generic(ms, seqStore, rep, src)
}

/// Port of `ZSTD_compressBlock_fast_extDict_generic`.
pub fn ZSTD_compressBlock_fast_extDict_generic(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    let cParams = ms.cParams;
    let mls = cParams.minMatch;
    if !ZSTD_window_hasExtDict(&ms.window) {
        return ZSTD_compressBlock_fast_noDict_generic(ms, seqStore, rep, src, 0, mls);
    }
    let hlog = cParams.hashLog;
    let stepSize = (cParams.targetLength + u32::from(cParams.targetLength == 0) + 1) as usize;
    let prefixBaseIndex = ms.window.dictLimit.max(ms.loadedDictEnd);
    let base_off = prefixBaseIndex;
    let dict_base_off = prefixBaseIndex.wrapping_sub(ms.dictContent.len() as u32);
    let endIndex = prefixBaseIndex.wrapping_add(src.len() as u32);
    let lowLimit = ZSTD_getLowestMatchIndex(ms, endIndex, cParams.windowLog);
    let dictStartIndex = lowLimit.max(dict_base_off);
    let prefixStartIndex = prefixBaseIndex.max(lowLimit);
    let prefixStart = prefixStartIndex.wrapping_sub(base_off) as usize;
    let dict = &ms.dictContent;
    let dictStart = dictStartIndex.wrapping_sub(dict_base_off) as usize;
    let dictEnd = prefixStartIndex.wrapping_sub(dict_base_off) as usize;
    let iend = src.len();
    let ilimit = iend.saturating_sub(HASH_READ_SIZE);
    let mut anchor = 0usize;
    let mut offset_1 = rep[0];
    let mut offset_2 = rep[1];
    let mut offsetSaved1 = 0u32;
    let mut offsetSaved2 = 0u32;

    if prefixStartIndex == dictStartIndex {
        return ZSTD_compressBlock_fast_noDict_generic(ms, seqStore, rep, src, 0, mls);
    }
    if src.len() < HASH_READ_SIZE || stepSize == 0 {
        return src.len();
    }
    {
        let curr = base_off;
        let maxRep = curr.wrapping_sub(dictStartIndex);
        if offset_2 >= maxRep {
            offsetSaved2 = offset_2;
            offset_2 = 0;
        }
        if offset_1 >= maxRep {
            offsetSaved1 = offset_1;
            offset_1 = 0;
        }
    }
    debug_assert!(prefixStartIndex >= dictStartIndex);
    debug_assert!(endIndex >= prefixStartIndex);
    debug_assert!(dictEnd <= dict.len());
    debug_assert!(dictStart <= dictEnd);

    let mut ip0 = 0usize;
    loop {
        let mut step = stepSize;
        let kStepIncr = 1usize << (kSearchStrength - 1);
        let mut nextStep = ip0 + kStepIncr;
        let mut ip1 = ip0 + 1;
        let mut ip2 = ip0 + step;
        let mut ip3 = ip2 + 1;
        if ip3 >= ilimit {
            break;
        }

        let mut hash0 = ZSTD_hashPtr(&src[ip0..], hlog, mls);
        let mut hash1 = ZSTD_hashPtr(&src[ip1..], hlog, mls);
        let mut idx = ms.hashTable[hash0];

        'search: loop {
            let current2 = base_off.wrapping_add(ip2 as u32);
            let repIndex = current2.wrapping_sub(offset_1);
            let repInDict = repIndex < prefixStartIndex;
            let repLocal = if repInDict {
                repIndex.wrapping_sub(dict_base_off) as usize
            } else {
                repIndex.wrapping_sub(base_off) as usize
            };

            let current0 = base_off.wrapping_add(ip0 as u32);
            ms.hashTable[hash0] = current0;

            if offset_1 > 0
                && ZSTD_index_overlap_check(prefixStartIndex, repIndex)
                && if repInDict {
                    repLocal + 4 <= dictEnd
                        && MEM_read32(&dict[repLocal..]) == MEM_read32(&src[ip2..])
                } else {
                    repLocal + 4 <= src.len()
                        && MEM_read32(&src[repLocal..]) == MEM_read32(&src[ip2..])
                }
            {
                let mut match_local = repLocal;
                let mut mLength = usize::from(
                    ip2 > 0
                        && match_local > if repInDict { dictStart } else { prefixStart }
                        && src[ip2 - 1]
                            == if repInDict {
                                dict[match_local - 1]
                            } else {
                                src[match_local - 1]
                            },
                );
                ip0 = ip2 - mLength;
                match_local -= mLength;
                let matchEnd = if repInDict { dictEnd } else { iend };
                mLength += 4;
                mLength += ZSTD_count_2segments(
                    src,
                    ip0 + mLength,
                    iend,
                    prefixStart,
                    if repInDict { dict } else { src },
                    match_local + mLength,
                    matchEnd,
                );
                ZSTD_storeSeq(
                    seqStore,
                    ip0 - anchor,
                    &src[anchor..ip0],
                    REPCODE_TO_OFFBASE(1),
                    mLength,
                );
                ip0 += mLength;
                anchor = ip0;
                if ip1 < ip0 {
                    ms.hashTable[hash1] = base_off.wrapping_add(ip1 as u32);
                }
                if ip0 <= ilimit {
                    ms.hashTable[ZSTD_hashPtr(
                        &src[(current0.wrapping_add(2).wrapping_sub(base_off)) as usize..],
                        hlog,
                        mls,
                    )] = current0.wrapping_add(2);
                    ms.hashTable[ZSTD_hashPtr(&src[ip0 - 2..], hlog, mls)] =
                        base_off.wrapping_add((ip0 - 2) as u32);
                    while ip0 <= ilimit {
                        let repIndex2 = base_off.wrapping_add(ip0 as u32).wrapping_sub(offset_2);
                        let repInDict2 = repIndex2 < prefixStartIndex;
                        let repLocal2 = if repInDict2 {
                            repIndex2.wrapping_sub(dict_base_off) as usize
                        } else {
                            repIndex2.wrapping_sub(base_off) as usize
                        };
                        if offset_2 > 0
                            && ZSTD_index_overlap_check(prefixStartIndex, repIndex2)
                            && if repInDict2 {
                                repLocal2 + 4 <= dictEnd
                                    && MEM_read32(&dict[repLocal2..]) == MEM_read32(&src[ip0..])
                            } else {
                                repLocal2 + 4 <= src.len()
                                    && MEM_read32(&src[repLocal2..]) == MEM_read32(&src[ip0..])
                            }
                        {
                            let repEnd2 = if repInDict2 { dictEnd } else { iend };
                            let repLength2 = ZSTD_count_2segments(
                                src,
                                ip0 + 4,
                                iend,
                                prefixStart,
                                if repInDict2 { dict } else { src },
                                repLocal2 + 4,
                                repEnd2,
                            ) + 4;
                            std::mem::swap(&mut offset_2, &mut offset_1);
                            ZSTD_storeSeq(
                                seqStore,
                                0,
                                &src[anchor..anchor],
                                REPCODE_TO_OFFBASE(1),
                                repLength2,
                            );
                            ms.hashTable[ZSTD_hashPtr(&src[ip0..], hlog, mls)] =
                                base_off.wrapping_add(ip0 as u32);
                            ip0 += repLength2;
                            anchor = ip0;
                            continue;
                        }
                        break;
                    }
                }
                break 'search;
            }

            if idx >= dictStartIndex {
                let idxInDict = idx < prefixStartIndex;
                let idxLocal = if idxInDict {
                    idx.wrapping_sub(dict_base_off) as usize
                } else {
                    idx.wrapping_sub(base_off) as usize
                };
                if idxLocal + 4 <= if idxInDict { dictEnd } else { src.len() }
                    && MEM_read32(&src[ip0..])
                        == if idxInDict {
                            MEM_read32(&dict[idxLocal..])
                        } else {
                            MEM_read32(&src[idxLocal..])
                        }
                {
                    let offset = current0 - idx;
                    let mut match_local = idxLocal;
                    let lowMatchPtr = if idxInDict { dictStart } else { prefixStart };
                    let matchEnd = if idxInDict { dictEnd } else { iend };
                    let mut mLength = 4usize;
                    while ip0 > anchor
                        && match_local > lowMatchPtr
                        && src[ip0 - 1]
                            == if idxInDict {
                                dict[match_local - 1]
                            } else {
                                src[match_local - 1]
                            }
                    {
                        ip0 -= 1;
                        match_local -= 1;
                        mLength += 1;
                    }
                    offset_2 = offset_1;
                    offset_1 = offset;
                    mLength += ZSTD_count_2segments(
                        src,
                        ip0 + mLength,
                        iend,
                        prefixStart,
                        if idxInDict { dict } else { src },
                        match_local + mLength,
                        matchEnd,
                    );
                    ZSTD_storeSeq(
                        seqStore,
                        ip0 - anchor,
                        &src[anchor..ip0],
                        OFFSET_TO_OFFBASE(offset),
                        mLength,
                    );
                    ip0 += mLength;
                    anchor = ip0;
                    if ip1 < ip0 {
                        ms.hashTable[hash1] = base_off.wrapping_add(ip1 as u32);
                    }
                    if ip0 <= ilimit {
                        ms.hashTable[ZSTD_hashPtr(
                            &src[(current0.wrapping_add(2).wrapping_sub(base_off)) as usize..],
                            hlog,
                            mls,
                        )] = current0.wrapping_add(2);
                        ms.hashTable[ZSTD_hashPtr(&src[ip0 - 2..], hlog, mls)] =
                            base_off.wrapping_add((ip0 - 2) as u32);
                        while ip0 <= ilimit {
                            let repIndex2 =
                                base_off.wrapping_add(ip0 as u32).wrapping_sub(offset_2);
                            let repInDict2 = repIndex2 < prefixStartIndex;
                            let repLocal2 = if repInDict2 {
                                repIndex2.wrapping_sub(dict_base_off) as usize
                            } else {
                                repIndex2.wrapping_sub(base_off) as usize
                            };
                            if offset_2 > 0
                                && ZSTD_index_overlap_check(prefixStartIndex, repIndex2)
                                && if repInDict2 {
                                    repLocal2 + 4 <= dictEnd
                                        && MEM_read32(&dict[repLocal2..]) == MEM_read32(&src[ip0..])
                                } else {
                                    repLocal2 + 4 <= src.len()
                                        && MEM_read32(&src[repLocal2..]) == MEM_read32(&src[ip0..])
                                }
                            {
                                let repEnd2 = if repInDict2 { dictEnd } else { iend };
                                let repLength2 = ZSTD_count_2segments(
                                    src,
                                    ip0 + 4,
                                    iend,
                                    prefixStart,
                                    if repInDict2 { dict } else { src },
                                    repLocal2 + 4,
                                    repEnd2,
                                ) + 4;
                                std::mem::swap(&mut offset_2, &mut offset_1);
                                ZSTD_storeSeq(
                                    seqStore,
                                    0,
                                    &src[anchor..anchor],
                                    REPCODE_TO_OFFBASE(1),
                                    repLength2,
                                );
                                ms.hashTable[ZSTD_hashPtr(&src[ip0..], hlog, mls)] =
                                    base_off.wrapping_add(ip0 as u32);
                                ip0 += repLength2;
                                anchor = ip0;
                                continue;
                            }
                            break;
                        }
                    }
                    break 'search;
                }
            }

            idx = ms.hashTable[hash1];
            hash0 = hash1;
            hash1 = ZSTD_hashPtr(&src[ip2..], hlog, mls);
            ip0 = ip1;
            ip1 = ip2;
            ip2 = ip3;

            let current0 = base_off.wrapping_add(ip0 as u32);
            ms.hashTable[hash0] = current0;

            if idx >= dictStartIndex {
                let idxInDict = idx < prefixStartIndex;
                let idxLocal = if idxInDict {
                    idx.wrapping_sub(dict_base_off) as usize
                } else {
                    idx.wrapping_sub(base_off) as usize
                };
                if idxLocal + 4 <= if idxInDict { dictEnd } else { src.len() }
                    && MEM_read32(&src[ip0..])
                        == if idxInDict {
                            MEM_read32(&dict[idxLocal..])
                        } else {
                            MEM_read32(&src[idxLocal..])
                        }
                {
                    let offset = current0 - idx;
                    let mut match_local = idxLocal;
                    let lowMatchPtr = if idxInDict { dictStart } else { prefixStart };
                    let matchEnd = if idxInDict { dictEnd } else { iend };
                    let mut mLength = 4usize;
                    while ip0 > anchor
                        && match_local > lowMatchPtr
                        && src[ip0 - 1]
                            == if idxInDict {
                                dict[match_local - 1]
                            } else {
                                src[match_local - 1]
                            }
                    {
                        ip0 -= 1;
                        match_local -= 1;
                        mLength += 1;
                    }
                    offset_2 = offset_1;
                    offset_1 = offset;
                    mLength += ZSTD_count_2segments(
                        src,
                        ip0 + mLength,
                        iend,
                        prefixStart,
                        if idxInDict { dict } else { src },
                        match_local + mLength,
                        matchEnd,
                    );
                    ZSTD_storeSeq(
                        seqStore,
                        ip0 - anchor,
                        &src[anchor..ip0],
                        OFFSET_TO_OFFBASE(offset),
                        mLength,
                    );
                    ip0 += mLength;
                    anchor = ip0;
                    if ip1 < ip0 {
                        ms.hashTable[hash1] = base_off.wrapping_add(ip1 as u32);
                    }
                    if ip0 <= ilimit {
                        ms.hashTable[ZSTD_hashPtr(
                            &src[(current0.wrapping_add(2).wrapping_sub(base_off)) as usize..],
                            hlog,
                            mls,
                        )] = current0.wrapping_add(2);
                        ms.hashTable[ZSTD_hashPtr(&src[ip0 - 2..], hlog, mls)] =
                            base_off.wrapping_add((ip0 - 2) as u32);
                        while ip0 <= ilimit {
                            let repIndex2 =
                                base_off.wrapping_add(ip0 as u32).wrapping_sub(offset_2);
                            let repInDict2 = repIndex2 < prefixStartIndex;
                            let repLocal2 = if repInDict2 {
                                repIndex2.wrapping_sub(dict_base_off) as usize
                            } else {
                                repIndex2.wrapping_sub(base_off) as usize
                            };
                            if offset_2 > 0
                                && ZSTD_index_overlap_check(prefixStartIndex, repIndex2)
                                && if repInDict2 {
                                    repLocal2 + 4 <= dictEnd
                                        && MEM_read32(&dict[repLocal2..]) == MEM_read32(&src[ip0..])
                                } else {
                                    repLocal2 + 4 <= src.len()
                                        && MEM_read32(&src[repLocal2..]) == MEM_read32(&src[ip0..])
                                }
                            {
                                let repEnd2 = if repInDict2 { dictEnd } else { iend };
                                let repLength2 = ZSTD_count_2segments(
                                    src,
                                    ip0 + 4,
                                    iend,
                                    prefixStart,
                                    if repInDict2 { dict } else { src },
                                    repLocal2 + 4,
                                    repEnd2,
                                ) + 4;
                                std::mem::swap(&mut offset_2, &mut offset_1);
                                ZSTD_storeSeq(
                                    seqStore,
                                    0,
                                    &src[anchor..anchor],
                                    REPCODE_TO_OFFBASE(1),
                                    repLength2,
                                );
                                ms.hashTable[ZSTD_hashPtr(&src[ip0..], hlog, mls)] =
                                    base_off.wrapping_add(ip0 as u32);
                                ip0 += repLength2;
                                anchor = ip0;
                                continue;
                            }
                            break;
                        }
                    }
                    break 'search;
                }
            }

            idx = ms.hashTable[hash1];
            hash0 = hash1;
            hash1 = ZSTD_hashPtr(&src[ip2..], hlog, mls);
            ip0 = ip1;
            ip1 = ip2;
            ip2 = ip0 + step;
            ip3 = ip1 + step;
            if ip2 >= nextStep {
                step += 1;
                crate::common::zstd_internal::prefetchSliceByte(src, ip1 + 64);
                crate::common::zstd_internal::prefetchSliceByte(src, ip1 + 128);
                nextStep += kStepIncr;
            }
            if ip3 >= ilimit {
                break 'search;
            }
        }
    }

    offsetSaved2 = if offsetSaved1 != 0 && offset_1 != 0 {
        offsetSaved1
    } else {
        offsetSaved2
    };
    rep[0] = if offset_1 != 0 {
        offset_1
    } else {
        offsetSaved1
    };
    rep[1] = if offset_2 != 0 {
        offset_2
    } else {
        offsetSaved2
    };
    iend - anchor
}

/// Variant that scans `src[istart..]`, treating `src[0..istart]` as
/// valid history for back-references. Used for cross-block match
/// carry in `ZSTD_compressFrame_fast`.
pub fn ZSTD_compressBlock_fast_with_history(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
    istart: usize,
) -> usize {
    let mml = ms.cParams.minMatch;
    ZSTD_compressBlock_fast_noDict_generic(ms, seqStore, rep, src, istart, mml)
}

/// Port of `ZSTD_fillHashTable`. Dispatches on `tfp`.
pub fn ZSTD_fillHashTable(
    ms: &mut ZSTD_MatchState_t,
    src: &[u8],
    dtlm: ZSTD_dictTableLoadMethod_e,
    tfp: ZSTD_tableFillPurpose_e,
) {
    match tfp {
        ZSTD_tableFillPurpose_e::ZSTD_tfp_forCDict => ZSTD_fillHashTableForCDict(ms, src, dtlm),
        ZSTD_tableFillPurpose_e::ZSTD_tfp_forCCtx => ZSTD_fillHashTableForCCtx(ms, src, dtlm),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::match_state::ZSTD_compressionParameters;
    use crate::compress::match_state::ZSTD_WINDOW_START_INDEX;
    use crate::compress::seq_store::{OFFBASE_IS_OFFSET, OFFBASE_TO_OFFSET};

    #[test]
    fn dictTableLoadMethod_and_tableFillPurpose_discriminants_match_upstream() {
        // Upstream (zstd_compress_internal.h:548-549):
        //   ZSTD_dictTableLoadMethod_e: dtlm_fast=0, dtlm_full=1
        //   ZSTD_tableFillPurpose_e:    tfp_forCCtx=0, tfp_forCDict=1
        // These parameterize hash-table seeding strategy; drift would
        // mis-route the dict loader to the wrong fill mode.
        assert_eq!(ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_fast as u32, 0);
        assert_eq!(ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_full as u32, 1);
        assert_eq!(ZSTD_tableFillPurpose_e::ZSTD_tfp_forCCtx as u32, 0);
        assert_eq!(ZSTD_tableFillPurpose_e::ZSTD_tfp_forCDict as u32, 1);
    }

    fn test_state() -> ZSTD_MatchState_t {
        ZSTD_MatchState_t::new(ZSTD_compressionParameters {
            hashLog: 10,
            minMatch: 4,
            ..Default::default()
        })
    }

    #[test]
    fn match4Found_branch_rejects_below_low_limit_accepts_at_limit() {
        // Contract:
        //  - match_idx strictly below idx_low_limit → reject (false).
        //  - match_idx == idx_low_limit AND 4 bytes agree → accept.
        //  - match_idx in range but 4 bytes disagree → reject.
        let buf: Vec<u8> = (0..128u8).cycle().take(1024).collect();
        // The byte pattern has period 128, so positions 0 and 128
        // share the same 4 bytes; positions 0 and 1 do not.

        // Below the limit → reject regardless of content.
        assert!(!ZSTD_match4Found_branch(&buf, 128, 0, 5, 10));

        // At the limit with matching 4 bytes → accept.
        assert!(ZSTD_match4Found_branch(&buf, 128, 0, 10, 10));

        // In range but bytes disagree → reject.
        assert!(!ZSTD_match4Found_branch(&buf, 128, 1, 100, 10));
    }

    #[test]
    fn match4Found_cmov_matches_branch_variant() {
        let buf: Vec<u8> = (0..128u8).cycle().take(1024).collect();
        assert_eq!(
            ZSTD_match4Found_cmov(&buf, 128, 0, 10, 10),
            ZSTD_match4Found_branch(&buf, 128, 0, 10, 10)
        );
        assert_eq!(
            ZSTD_match4Found_cmov(&buf, 128, 1, 100, 10),
            ZSTD_match4Found_branch(&buf, 128, 1, 100, 10)
        );
    }

    #[test]
    fn getLowestMatchIndex_uses_lowLimit_not_dictLimit() {
        let mut ms = test_state();
        ms.window.lowLimit = 100;
        ms.window.dictLimit = 200;
        ms.loadedDictEnd = 0;
        // At curr=250: maxDist=1<<17. curr-lowLimit=150, curr-dictLimit=50 —
        // both ≤ maxDist, so "withinWindow" picks the respective lowest.
        assert_eq!(ZSTD_getLowestMatchIndex(&ms, 250, 17), 100);
        // Contrast with getLowestPrefixIndex on the same curr:
        assert_eq!(ZSTD_getLowestPrefixIndex(&ms, 250, 17), 200);
    }

    #[test]
    fn getLowestMatchIndex_clamps_to_curr_minus_maxDist() {
        let mut ms = test_state();
        ms.window.lowLimit = 10;
        ms.loadedDictEnd = 0;
        // curr - lowLimit = 1_000_000, maxDist = 1<<17. Clamp to
        // curr - maxDist.
        let got = ZSTD_getLowestMatchIndex(&ms, 1_000_010, 17);
        assert_eq!(got, 1_000_010 - (1 << 17));
    }

    #[test]
    fn getLowestMatchIndex_honors_loaded_dict() {
        let mut ms = test_state();
        ms.window.lowLimit = 10;
        ms.loadedDictEnd = 42;
        // With a dict loaded, we skip the "withinWindow" clamp.
        assert_eq!(ZSTD_getLowestMatchIndex(&ms, 1_000_000, 17), 10);
    }

    #[test]
    fn lowest_index_helpers_use_upstream_wrapping_subtraction() {
        let mut ms = test_state();
        ms.window.lowLimit = 0;
        ms.window.dictLimit = 0;
        ms.loadedDictEnd = 0;

        assert_eq!(ZSTD_getLowestPrefixIndex(&ms, 1, 17), 0);
        assert_eq!(ZSTD_getLowestMatchIndex(&ms, 1, 17), 0);
    }

    #[test]
    fn fill_hash_table_noop_when_source_too_small() {
        let mut ms = test_state();
        // src smaller than HASH_READ_SIZE → nothing is inserted.
        ZSTD_fillHashTable(
            &mut ms,
            &[1u8, 2, 3, 4],
            ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_fast,
            ZSTD_tableFillPurpose_e::ZSTD_tfp_forCCtx,
        );
        assert!(ms.hashTable.iter().all(|&c| c == 0));
    }

    #[test]
    fn fill_hash_table_cctx_inserts_every_third_position() {
        let mut ms = test_state();
        // Source with distinct bytes → each 4-byte window hashes to
        // a (mostly) distinct bucket; buckets should be non-zero after
        // fill.
        let src: Vec<u8> = (0..128u8).collect();
        ZSTD_fillHashTable(
            &mut ms,
            &src,
            ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_fast,
            ZSTD_tableFillPurpose_e::ZSTD_tfp_forCCtx,
        );
        let n_filled = ms.hashTable.iter().filter(|&&c| c != 0).count();
        // Inserts at positions 0, 3, 6, 9 ... up to roughly
        // (src.len() - HASH_READ_SIZE) / 3 — a sizeable fraction of
        // the 1024-entry table.
        assert!(
            n_filled > 10,
            "expected many buckets filled, got {n_filled}"
        );
    }

    #[test]
    fn compress_block_fast_with_history_uses_prior_block_content() {
        // A single match state scans 2 halves of a buffer where the
        // second half is an exact copy of the first. The with_history
        // variant should emit at least one full-offset sequence
        // referencing back into the first half.
        let half = b"The quick brown fox jumps over the lazy dog in the forest. ".repeat(20);
        let mut full = half.clone();
        full.extend_from_slice(&half);
        let boundary = half.len();

        let mut ms = ZSTD_MatchState_t::new(ZSTD_compressionParameters {
            windowLog: 17,
            hashLog: 12,
            minMatch: 4,
            ..Default::default()
        });
        let mut seqStore = SeqStore_t::with_capacity(1024, 131072);
        let mut rep: [u32; ZSTD_REP_NUM] = [1, 4, 8];

        // First pass: scan just the first half to populate the hash.
        ZSTD_compressBlock_fast(&mut ms, &mut seqStore, &mut rep, &full[..boundary]);
        let seqs_first = seqStore.sequences.clone();

        // Second pass: scan from the boundary with history.
        seqStore.reset();
        let last =
            ZSTD_compressBlock_fast_with_history(&mut ms, &mut seqStore, &mut rep, &full, boundary);
        let total_lits_second = seqStore.literals.len() + last;

        // With cross-block history, the second half should produce a
        // much smaller "unmatched" residue than if there were no
        // history at all — the text repeats verbatim. Expect literals
        // < 25% of the second-half size.
        let second_size = full.len() - boundary;
        assert!(
            total_lits_second < second_size / 4,
            "cross-block history not helping: lits={} of {} bytes",
            total_lits_second,
            second_size
        );
        assert!(!seqs_first.is_empty() || seqStore.sequences.is_empty());
    }

    #[test]
    fn compress_block_fast_finds_repetitive_matches() {
        // Heavy self-similar input: "abcdef..." repeated. The
        // match finder should emit multiple sequences.
        let src: Vec<u8> = b"the quick brown fox jumps over the lazy dog. "
            .iter()
            .cycle()
            .take(2000)
            .copied()
            .collect();

        let mut ms = ZSTD_MatchState_t::new(ZSTD_compressionParameters {
            windowLog: 17,
            hashLog: 12,
            minMatch: 4,
            ..Default::default()
        });
        // Non-dict block: the encoder fills the hash table
        // progressively as it scans; no pre-seeding.
        let mut seq = SeqStore_t::with_capacity(1024, 131072);
        let mut rep: [u32; ZSTD_REP_NUM] = [1, 4, 8];

        let last_lits = ZSTD_compressBlock_fast(&mut ms, &mut seq, &mut rep, &src);

        assert!(
            !seq.sequences.is_empty(),
            "match finder emitted no sequences"
        );
        // A reasonable amount of savings: sequence count should be a
        // small fraction of srcSize, and literals should be < srcSize.
        let total_lits = seq.literals.len() + last_lits;
        assert!(
            total_lits < src.len(),
            "no savings: total lits = {total_lits}, src = {}",
            src.len()
        );
    }

    #[test]
    fn dict_and_ext_wrappers_route_through_live_fast_entries() {
        let src: Vec<u8> = b"the quick brown fox jumps over the lazy dog. "
            .iter()
            .cycle()
            .take(1024)
            .copied()
            .collect();
        let variants: [fn(
            &mut ZSTD_MatchState_t,
            &mut SeqStore_t,
            &mut [u32; ZSTD_REP_NUM],
            &[u8],
        ) -> usize; 2] = [
            ZSTD_compressBlock_fast_dictMatchState,
            ZSTD_compressBlock_fast_extDict,
        ];

        for (idx, f) in variants.into_iter().enumerate() {
            let mut ms = ZSTD_MatchState_t::new(ZSTD_compressionParameters {
                windowLog: 17,
                hashLog: 12,
                minMatch: 4,
                ..Default::default()
            });
            let mut seq = SeqStore_t::with_capacity(1024, 131072);
            let mut rep = [1u32, 4, 8];
            let last_lits = f(&mut ms, &mut seq, &mut rep, &src);
            assert!(last_lits < src.len(), "variant {idx} emitted only literals");
            assert!(
                !seq.sequences.is_empty(),
                "variant {idx} regressed into a dead wrapper"
            );
        }
    }

    #[test]
    fn fill_hash_table_cdict_fills_additional_slots() {
        // Full-dict fill touches extra positions beyond the step.
        // On the same input, CDict fill should leave at least as many
        // buckets non-zero as CCtx fill (typically strictly more).
        let src: Vec<u8> = (0..200u8).collect();

        let mut ms_cctx = test_state();
        ZSTD_fillHashTable(
            &mut ms_cctx,
            &src,
            ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_fast,
            ZSTD_tableFillPurpose_e::ZSTD_tfp_forCCtx,
        );
        let n_cctx = ms_cctx.hashTable.iter().filter(|&&c| c != 0).count();

        let mut ms_cdict = test_state();
        ZSTD_fillHashTable(
            &mut ms_cdict,
            &src,
            ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_full,
            ZSTD_tableFillPurpose_e::ZSTD_tfp_forCDict,
        );
        let n_cdict = ms_cdict.hashTable.iter().filter(|&&c| c != 0).count();

        assert!(
            n_cdict >= n_cctx,
            "CDict fill ({n_cdict}) should match or exceed CCtx fill ({n_cctx})"
        );
    }

    #[test]
    fn fill_hash_table_cdict_packs_short_cache_tags() {
        let mut ms = test_state();
        let src: Vec<u8> = (0..11u8).collect();
        ZSTD_fillHashTableForCDict(&mut ms, &src, ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_fast);

        let hash_and_tag = ZSTD_hashPtr(
            &src,
            ms.cParams.hashLog + ZSTD_SHORT_CACHE_TAG_BITS,
            ms.cParams.minMatch,
        );
        let slot = hash_and_tag >> ZSTD_SHORT_CACHE_TAG_BITS;
        let expected = (ms.window.base_offset << ZSTD_SHORT_CACHE_TAG_BITS)
            | (hash_and_tag as u32 & ((1u32 << ZSTD_SHORT_CACHE_TAG_BITS) - 1));

        assert_eq!(ms.hashTable[slot], expected);
    }

    #[test]
    fn fast_ext_dict_uses_external_dictionary_bytes() {
        let dict = b"abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ".to_vec();
        let src = b"mnopqrstuvwxyz012345".to_vec();

        let cp = ZSTD_compressionParameters {
            windowLog: 17,
            hashLog: 12,
            minMatch: 4,
            strategy: 1,
            ..Default::default()
        };

        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.dictContent = dict.clone();
        ms.window.base_offset = ZSTD_WINDOW_START_INDEX;
        ms.nextToUpdate = ZSTD_WINDOW_START_INDEX;
        ZSTD_fillHashTableForCCtx(&mut ms, &dict, ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_fast);

        ms.window.dictBase_offset = ZSTD_WINDOW_START_INDEX;
        ms.window.lowLimit = ZSTD_WINDOW_START_INDEX;
        ms.window.dictLimit = ZSTD_WINDOW_START_INDEX.wrapping_add(dict.len() as u32);
        ms.window.base_offset = ms.window.dictLimit;
        ms.window.nextSrc = ms.window.base_offset.wrapping_add(src.len() as u32);
        ms.loadedDictEnd = dict.len() as u32;

        let mut seq = SeqStore_t::with_capacity(128, 4096);
        let mut rep = [1u32, 4, 8];
        let last_lits = ZSTD_compressBlock_fast_extDict(&mut ms, &mut seq, &mut rep, &src);

        assert!(last_lits < src.len(), "extDict path emitted only literals");
        assert!(
            !seq.sequences.is_empty(),
            "extDict path failed to emit any sequences"
        );
        assert!(OFFBASE_IS_OFFSET(seq.sequences[0].offBase));
        assert!(
            OFFBASE_TO_OFFSET(seq.sequences[0].offBase) > src.len() as u32,
            "first match offset {:?} did not reach into the external dictionary",
            seq.sequences[0]
        );
    }
}
