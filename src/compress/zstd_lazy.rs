//! Translation of `lib/compress/zstd_lazy.c` — strategies 3..=6
//! (greedy / lazy / lazy2 / btlazy2). Hash-chain infra
//! (`ZSTD_insertAndFindFirstIndex`, `ZSTD_HcFindBestMatch_noDict`) +
//! all four noDict entry points (`ZSTD_compressBlock_{greedy,lazy,lazy2,btlazy2}`)
//! are ported. btlazy2 currently falls through to lazy2 until the
//! true binary-tree matcher lands. dictMatchState / extDict / row-hash
//! variants remain skeletal — see the `stub_block_compressor!` macro.

#![allow(non_snake_case)]

use crate::common::mem::MEM_read32;
use crate::compress::match_state::ZSTD_MatchState_t;
use crate::compress::seq_store::{
    SeqStore_t, ZSTD_storeSeq, OFFSET_TO_OFFBASE, REPCODE_TO_OFFBASE, ZSTD_REP_NUM,
};
use crate::compress::zstd_hashes::{ZSTD_count, ZSTD_hashPtr};

/// Port of `ZSTD_insertAndFindFirstIndex_internal`. Catches the
/// chain table up from `nextToUpdate` to `target_idx` (the absolute
/// index of `ip`), then returns the head of `hashTable[hash(ip)]`.
///
/// Rust signature note: upstream takes raw `BYTE*` into the source.
/// Rust port takes the full `src` slice + the `ip` byte offset so
/// indexed reads can be bounds-checked.
pub fn ZSTD_insertAndFindFirstIndex_internal(
    ms: &mut ZSTD_MatchState_t,
    src: &[u8],
    ip: usize,
    mls: u32,
) -> u32 {
    let hashLog = ms.cParams.hashLog;
    let chainMask = (1u32 << ms.cParams.chainLog) - 1;
    let base_off = ms.window.base_offset;
    let target: u32 = base_off + ip as u32;
    // Ensure chainTable sized.
    let chainSize = 1usize << ms.cParams.chainLog;
    if ms.chainTable.len() < chainSize {
        ms.chainTable.resize(chainSize, 0);
    }
    let mut idx = ms.nextToUpdate;
    while idx < target {
        let rel = idx.saturating_sub(base_off) as usize;
        if rel + mls as usize > src.len() {
            break;
        }
        let h = ZSTD_hashPtr(&src[rel..], hashLog, mls);
        let slot = (idx & chainMask) as usize;
        ms.chainTable[slot] = ms.hashTable[h];
        ms.hashTable[h] = idx;
        idx += 1;
    }
    ms.nextToUpdate = target;
    // Lookup hash at ip itself.
    let h = ZSTD_hashPtr(&src[ip..], hashLog, mls);
    ms.hashTable[h]
}

/// Port of `ZSTD_HcFindBestMatch` (noDict path only). Walks the hash
/// chain at `ip` up to `searchLog` attempts, returns the longest
/// match length (in bytes) and fills `offBase_out` with the chosen
/// match's offBase. Returns 0 if no match ≥ 4 bytes was found.
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_HcFindBestMatch_noDict(
    ms: &mut ZSTD_MatchState_t,
    src: &[u8],
    ip: usize,
    iLimit: usize,
    offBase_out: &mut u32,
    mls: u32,
) -> usize {
    let chainMask = (1u32 << ms.cParams.chainLog) - 1;
    let chainSize = 1u32 << ms.cParams.chainLog;
    let base_off = ms.window.base_offset;
    let curr = base_off + ip as u32;
    let maxDistance = 1u32 << ms.cParams.windowLog;
    let lowestValid = ms.window.lowLimit;
    let withinMaxDistance = if curr.wrapping_sub(lowestValid) > maxDistance {
        curr - maxDistance
    } else {
        lowestValid
    };
    let isDictionary = ms.loadedDictEnd != 0;
    let lowLimit = if isDictionary {
        lowestValid
    } else {
        withinMaxDistance
    };
    let minChain = curr.saturating_sub(chainSize);
    let mut nbAttempts = 1u32 << ms.cParams.searchLog;
    let mut ml: usize = 3; // upstream: 4-1 (seeded below minMatch so any ≥ 4 wins)

    // Insert + find head.
    let mut matchIndex = ZSTD_insertAndFindFirstIndex_internal(ms, src, ip, mls);

    while matchIndex >= lowLimit && matchIndex < curr && nbAttempts > 0 {
        nbAttempts -= 1;
        let match_pos = matchIndex.saturating_sub(base_off) as usize;
        if match_pos + 4 > src.len() {
            if matchIndex <= minChain {
                break;
            }
            let next = ms.chainTable[(matchIndex & chainMask) as usize];
            if next >= matchIndex {
                break; // chain must strictly decrease
            }
            matchIndex = next;
            continue;
        }
        // Cheap pre-check: the 4 bytes ending at (match + ml - 3) must
        // equal those at (ip + ml - 3). Only compute full match length
        // if this passes — OR, if ml is at its seed (3), just do a
        // head-of-match u32 compare.
        let head_match = MEM_read32(&src[ip..]) == MEM_read32(&src[match_pos..]);
        let tail_match = ml >= 3
            && match_pos + ml < src.len()
            && ip + ml <= iLimit
            && MEM_read32(&src[match_pos + ml - 3..]) == MEM_read32(&src[ip + ml - 3..]);
        let currentMl = if head_match || tail_match {
            ZSTD_count(src, ip, match_pos, iLimit)
        } else {
            0
        };

        if currentMl > ml {
            ml = currentMl;
            *offBase_out = OFFSET_TO_OFFBASE(curr - matchIndex);
            if ip + currentMl == iLimit {
                break;
            }
        }
        if matchIndex <= minChain {
            break;
        }
        let next = ms.chainTable[(matchIndex & chainMask) as usize];
        if next >= matchIndex {
            break;
        }
        matchIndex = next;
    }
    if ml >= 4 { ml } else { 0 }
}

/// Port of `ZSTD_compressBlock_lazy_generic` (noDict path). A
/// single parser that handles depths 0 (greedy), 1 (lazy), and 2
/// (lazy2). Per scan position:
///   1. Check repcode at ip+1 (depth-0 solution candidate).
///   2. Chain-search best match at ip. If no match, step forward.
///   3. Depth ≥ 1: advance ip++, re-search; compare gain vs current
///      best. If strictly better, replace and re-evaluate at the new
///      ip (may loop back to step 3).
///   4. Depth == 2: same trick at ip+2.
///   5. Back-extend for full-offset matches, emit, then check
///      immediate repcode at the new anchor.
///
/// Gain formula (upstream): `gain = ml * W - highbit32(offBase) + k`,
/// where `W` is 3 at depth 0→1 transitions and 4 at depth 1→2. This
/// factors offset encoding cost into the comparison so a slightly
/// longer match with a much larger offset may still lose.
pub fn ZSTD_compressBlock_lazy_noDict_generic(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
    istart: usize,
    depth: u32,
) -> usize {
    let mls = ms.cParams.minMatch.clamp(4, 6);
    let windowLog = ms.cParams.windowLog;
    let base_off = ms.window.base_offset;
    let srcSize = src.len();
    let endIndex = base_off + srcSize as u32;
    let prefixStartIndex =
        crate::compress::zstd_fast::ZSTD_getLowestPrefixIndex(ms, endIndex, windowLog);
    let prefixStart = prefixStartIndex.saturating_sub(base_off) as usize;
    let iend = srcSize;
    let ilimit = iend.saturating_sub(8);

    let mut anchor = istart;
    let mut ip = istart;
    if ip == prefixStart {
        ip += 1;
    }

    let mut rep_offset1 = rep[0];
    let mut rep_offset2 = rep[1];
    let mut offsetSaved1: u32 = 0;
    let mut offsetSaved2: u32 = 0;
    {
        let curr = base_off + ip as u32;
        let windowLow =
            crate::compress::zstd_fast::ZSTD_getLowestPrefixIndex(ms, curr, windowLog);
        let maxRep = curr - windowLow;
        if rep_offset2 > maxRep {
            offsetSaved2 = rep_offset2;
            rep_offset2 = 0;
        }
        if rep_offset1 > maxRep {
            offsetSaved1 = rep_offset1;
            rep_offset1 = 0;
        }
    }

    let kSearchStrength: u32 = 8;

    while ip < ilimit {
        let mut matchLength: usize = 0;
        let mut offBase: u32 = REPCODE_TO_OFFBASE(1);
        let mut start = ip + 1;

        // 1. Repcode at ip+1 (depth-0 baseline).
        if rep_offset1 > 0
            && ip + 1 + 4 <= iend
            && ip + 1 >= rep_offset1 as usize
            && MEM_read32(&src[ip + 1..])
                == MEM_read32(&src[ip + 1 - rep_offset1 as usize..])
        {
            matchLength =
                ZSTD_count(src, ip + 1 + 4, ip + 1 + 4 - rep_offset1 as usize, iend) + 4;
            // Depth 0 could `goto storeSequence` here (upstream perf
            // shortcut). We fall through — the chain search at ip only
            // replaces if it strictly beats `matchLength`, so behavior
            // matches upstream for any realistic match.
        }

        // 2. Chain-search at ip (depth-0 candidate).
        {
            let mut cand_off: u32 = 0;
            let ml2 = ZSTD_HcFindBestMatch_noDict(ms, src, ip, iend, &mut cand_off, mls);
            if ml2 > matchLength {
                matchLength = ml2;
                offBase = cand_off;
                start = ip;
            }
        }

        if matchLength < 4 {
            // No usable match — step forward with ramp.
            let step = ((ip - anchor) >> kSearchStrength) + 1;
            ip += step;
            continue;
        }

        // 3. Depth-1 lookahead: try ip+1; if a better match exists
        //    there, swap it in. Depth-2 repeats at ip+2. We loop
        //    within the depth-1 block so that a chain of better
        //    matches at successive positions is followed through.
        if depth >= 1 {
            let mut probe = ip + 1;
            while probe < ilimit {
                // 3a. Repcode-at-probe check.
                if rep_offset1 > 0
                    && probe + 4 <= iend
                    && probe >= rep_offset1 as usize
                    && MEM_read32(&src[probe..])
                        == MEM_read32(&src[probe - rep_offset1 as usize..])
                {
                    let mlRep = ZSTD_count(
                        src,
                        probe + 4,
                        probe + 4 - rep_offset1 as usize,
                        iend,
                    ) + 4;
                    let gain2 = (mlRep as i32) * 3;
                    let gain1 = (matchLength as i32) * 3
                        - crate::common::bits::ZSTD_highbit32(offBase) as i32
                        + 1;
                    if mlRep >= 4 && gain2 > gain1 {
                        matchLength = mlRep;
                        offBase = REPCODE_TO_OFFBASE(1);
                        start = probe;
                    }
                }
                // 3b. Full chain search at probe.
                let mut cand_off: u32 = 0;
                let ml2 = ZSTD_HcFindBestMatch_noDict(
                    ms, src, probe, iend, &mut cand_off, mls,
                );
                if ml2 >= 4 && cand_off != 0 {
                    let gain2 = (ml2 as i32) * 4
                        - crate::common::bits::ZSTD_highbit32(cand_off) as i32;
                    let gain1 = (matchLength as i32) * 4
                        - crate::common::bits::ZSTD_highbit32(offBase) as i32
                        + 4;
                    if gain2 > gain1 {
                        matchLength = ml2;
                        offBase = cand_off;
                        start = probe;
                        if depth >= 2 {
                            probe += 1;
                            continue; // re-evaluate one more step ahead
                        }
                    }
                }
                // 3c. Depth-2: one more step.
                if depth == 2 && probe + 1 < ilimit {
                    let p2 = probe + 1;
                    if rep_offset1 > 0
                        && p2 + 4 <= iend
                        && p2 >= rep_offset1 as usize
                        && MEM_read32(&src[p2..])
                            == MEM_read32(&src[p2 - rep_offset1 as usize..])
                    {
                        let mlRep = ZSTD_count(
                            src,
                            p2 + 4,
                            p2 + 4 - rep_offset1 as usize,
                            iend,
                        ) + 4;
                        let gain2 = (mlRep as i32) * 4;
                        let gain1 = (matchLength as i32) * 4
                            - crate::common::bits::ZSTD_highbit32(offBase) as i32
                            + 1;
                        if mlRep >= 4 && gain2 > gain1 {
                            matchLength = mlRep;
                            offBase = REPCODE_TO_OFFBASE(1);
                            start = p2;
                        }
                    }
                    let mut cand2_off: u32 = 0;
                    let ml3 = ZSTD_HcFindBestMatch_noDict(
                        ms, src, p2, iend, &mut cand2_off, mls,
                    );
                    if ml3 >= 4 && cand2_off != 0 {
                        let gain2 = (ml3 as i32) * 4
                            - crate::common::bits::ZSTD_highbit32(cand2_off) as i32;
                        let gain1 = (matchLength as i32) * 4
                            - crate::common::bits::ZSTD_highbit32(offBase) as i32
                            + 7;
                        if gain2 > gain1 {
                            matchLength = ml3;
                            offBase = cand2_off;
                            start = p2;
                            probe = p2 + 1;
                            continue;
                        }
                    }
                }
                break; // nothing better found at this probe
            }
        }

        // Back-extend for full-offset matches.
        if offBase > ZSTD_REP_NUM as u32 {
            let offset = (offBase - ZSTD_REP_NUM as u32) as usize;
            while start > anchor
                && start > offset
                && start - offset > prefixStart
                && src[start - 1] == src[start - offset - 1]
            {
                start -= 1;
                matchLength += 1;
            }
            rep_offset2 = rep_offset1;
            rep_offset1 = offset as u32;
        }

        // Emit.
        let litLength = start - anchor;
        ZSTD_storeSeq(
            seqStore,
            litLength,
            &src[anchor..start],
            offBase,
            matchLength,
        );
        ip = start + matchLength;
        anchor = ip;

        // Immediate repcode at new anchor.
        while ip <= ilimit
            && rep_offset2 > 0
            && ip + 4 <= iend
            && ip >= rep_offset2 as usize
            && MEM_read32(&src[ip..]) == MEM_read32(&src[ip - rep_offset2 as usize..])
        {
            let r = ZSTD_count(src, ip + 4, ip + 4 - rep_offset2 as usize, iend) + 4;
            std::mem::swap(&mut rep_offset1, &mut rep_offset2);
            ZSTD_storeSeq(seqStore, 0, &src[ip..ip], REPCODE_TO_OFFBASE(1), r);
            ip += r;
            anchor = ip;
        }
    }

    // Save reps.
    let offsetSaved2_final = if offsetSaved1 != 0 && rep_offset1 != 0 {
        offsetSaved1
    } else {
        offsetSaved2
    };
    rep[0] = if rep_offset1 != 0 { rep_offset1 } else { offsetSaved1 };
    rep[1] = if rep_offset2 != 0 { rep_offset2 } else { offsetSaved2_final };

    iend - anchor
}

/// Public entry for strategy=greedy (3): depth=0.
pub fn ZSTD_compressBlock_greedy(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_noDict_generic(ms, seqStore, rep, src, 0, 0)
}

/// Public entry for strategy=lazy (4): depth=1.
pub fn ZSTD_compressBlock_lazy(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_noDict_generic(ms, seqStore, rep, src, 0, 1)
}

/// Public entry for strategy=lazy2 (5): depth=2.
pub fn ZSTD_compressBlock_lazy2(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_noDict_generic(ms, seqStore, rep, src, 0, 2)
}

// ─── NOT YET PORTED: row-hash / dictMatchState / extDict / DDSS
// variants of the greedy/lazy/lazy2/btlazy2 match-finders. Upstream
// uses `ms.tagTable` (row-hash), `ms.dictMatchState` (dict-attach
// mode), and `window.dictBase` as a distinct pointer (ext-dict); none
// of those are on our `ZSTD_MatchState_t` yet. Returning
// `ErrorCode::Generic` so callers dispatching via
// `ZSTD_selectBlockCompressor` fail loudly instead of silently
// emitting wrong output. ─────────────────────────────────────────────
macro_rules! stub_block_compressor {
    ($name:ident) => {
        pub fn $name(
            _ms: &mut ZSTD_MatchState_t,
            _seqStore: &mut SeqStore_t,
            _rep: &mut [u32; ZSTD_REP_NUM],
            _src: &[u8],
        ) -> usize {
            crate::common::error::ERROR(crate::common::error::ErrorCode::Generic)
        }
    };
}
stub_block_compressor!(ZSTD_compressBlock_greedy_row);
stub_block_compressor!(ZSTD_compressBlock_lazy_row);
stub_block_compressor!(ZSTD_compressBlock_lazy2_row);
stub_block_compressor!(ZSTD_compressBlock_greedy_dictMatchState_row);
stub_block_compressor!(ZSTD_compressBlock_lazy_dictMatchState_row);
stub_block_compressor!(ZSTD_compressBlock_lazy2_dictMatchState_row);
stub_block_compressor!(ZSTD_compressBlock_greedy_extDict_row);
stub_block_compressor!(ZSTD_compressBlock_lazy_extDict_row);
stub_block_compressor!(ZSTD_compressBlock_lazy2_extDict_row);
stub_block_compressor!(ZSTD_compressBlock_greedy_dedicatedDictSearch);
stub_block_compressor!(ZSTD_compressBlock_lazy_dedicatedDictSearch);
stub_block_compressor!(ZSTD_compressBlock_lazy2_dedicatedDictSearch);
stub_block_compressor!(ZSTD_compressBlock_greedy_dedicatedDictSearch_row);
stub_block_compressor!(ZSTD_compressBlock_lazy_dedicatedDictSearch_row);
stub_block_compressor!(ZSTD_compressBlock_lazy2_dedicatedDictSearch_row);
stub_block_compressor!(ZSTD_compressBlock_greedy_dictMatchState);
stub_block_compressor!(ZSTD_compressBlock_lazy_dictMatchState);
stub_block_compressor!(ZSTD_compressBlock_lazy2_dictMatchState);
stub_block_compressor!(ZSTD_compressBlock_btlazy2_dictMatchState);
stub_block_compressor!(ZSTD_compressBlock_greedy_extDict);
stub_block_compressor!(ZSTD_compressBlock_lazy_extDict);
stub_block_compressor!(ZSTD_compressBlock_lazy2_extDict);
stub_block_compressor!(ZSTD_compressBlock_btlazy2_extDict);

/// Cross-block-history variant — caller passes `src[..istart]` as
/// prior content. Shared by all three depth variants.
pub fn ZSTD_compressBlock_lazy_with_history(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
    istart: usize,
    depth: u32,
) -> usize {
    ZSTD_compressBlock_lazy_noDict_generic(ms, seqStore, rep, src, istart, depth)
}

/// btlazy2 (strategy 6) needs a binary-tree matcher — falls through
/// to lazy2 until the bt port lands.
pub fn ZSTD_compressBlock_btlazy2(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy2(ms, seqStore, rep, src)
}

/// Port of `ZSTD_updateDUBT`. Inserts positions `[nextToUpdate, target)`
/// into the btlazy2 chain table as an unsorted queue — each new entry
/// is prepended to the hash-bucket chain and its sort-mark slot is
/// set to `ZSTD_DUBT_UNSORTED_MARK`. A later sort pass
/// (`ZSTD_insertDUBT1`) consumes the queue during search.
///
/// `buf` is the window buffer, `target` is the absolute index to
/// fill up to.
pub fn ZSTD_updateDUBT(
    ms: &mut ZSTD_MatchState_t,
    buf: &[u8],
    target: u32,
    mls: u32,
) {
    use crate::compress::match_state::ZSTD_DUBT_UNSORTED_MARK;
    let hashLog = ms.cParams.hashLog;
    let btLog = ms.cParams.chainLog - 1;
    let btMask: u32 = (1u32 << btLog) - 1;

    debug_assert!(ms.nextToUpdate >= ms.window.dictLimit);
    let mut idx = ms.nextToUpdate;
    while idx < target {
        let h = ZSTD_hashPtr(&buf[idx as usize..], hashLog, mls);
        let matchIndex = ms.hashTable[h];

        let nextCandidateSlot = (2 * (idx & btMask)) as usize;
        let sortMarkSlot = nextCandidateSlot + 1;

        ms.hashTable[h] = idx;
        ms.chainTable[nextCandidateSlot] = matchIndex;
        ms.chainTable[sortMarkSlot] = ZSTD_DUBT_UNSORTED_MARK;

        idx += 1;
    }
    ms.nextToUpdate = target;
}

/// Port of `ZSTD_insertDUBT1` — prefix-only variant. Sorts one
/// already-queued unsorted DUBT entry at position `curr` into its
/// proper place in the binary tree (up to `nbCompares` comparisons).
///
/// Inputs:
///   - `buf`: the window buffer
///   - `curr`: absolute position being sorted
///   - `iend_pos`: end offset of input (clamp for `ZSTD_count`)
///   - `nbCompares`, `btLow`: caller-supplied search limits
///
/// Ext-dict branch is deferred (matches upstream's
/// `dictMode != ZSTD_extDict` path only).
pub fn ZSTD_insertDUBT1(
    ms: &mut ZSTD_MatchState_t,
    buf: &[u8],
    curr: u32,
    iend_pos: usize,
    mut nbCompares: u32,
    btLow: u32,
) {
    let btLog = ms.cParams.chainLog - 1;
    let btMask: u32 = (1u32 << btLog) - 1;
    let mut commonLengthSmaller: usize = 0;
    let mut commonLengthLarger: usize = 0;
    let ip_pos = curr as usize;

    let mut smaller_slot: Option<usize> = Some((2 * (curr & btMask)) as usize);
    let mut larger_slot: Option<usize> = Some((2 * (curr & btMask)) as usize + 1);

    // Read the head of the unsorted queue from the chain table at
    // `curr`'s smallerPtr slot — upstream starts iteration from there.
    let mut matchIndex = ms.chainTable[smaller_slot.unwrap()];
    let windowValid = ms.window.lowLimit;
    let maxDistance = 1u32 << ms.cParams.windowLog;
    let windowLow = if curr.wrapping_sub(windowValid) > maxDistance {
        curr - maxDistance
    } else {
        windowValid
    };

    debug_assert!(curr >= btLow);

    while nbCompares > 0 && matchIndex > windowLow {
        let nextBase = (2 * (matchIndex & btMask)) as usize;
        let mut matchLength = commonLengthSmaller.min(commonLengthLarger);
        debug_assert!(matchIndex < curr);

        let match_pos = matchIndex as usize;
        matchLength += ZSTD_count(buf, ip_pos + matchLength, match_pos + matchLength, iend_pos);

        if ip_pos + matchLength == iend_pos {
            break;
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
}

/// Public wrapper of `ZSTD_insertAndFindFirstIndex`.
pub fn ZSTD_insertAndFindFirstIndex(
    ms: &mut ZSTD_MatchState_t,
    src: &[u8],
    ip: usize,
) -> u32 {
    let mls = ms.cParams.minMatch;
    ZSTD_insertAndFindFirstIndex_internal(ms, src, ip, mls)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::match_state::ZSTD_compressionParameters;

    #[test]
    fn greedy_emits_sequences_on_repetitive_text() {
        let src: Vec<u8> = b"the quick brown fox jumps over the lazy dog. "
            .iter()
            .cycle()
            .take(2000)
            .copied()
            .collect();
        let mut ms = ZSTD_MatchState_t::new(ZSTD_compressionParameters {
            windowLog: 17,
            hashLog: 12,
            chainLog: 12,
            searchLog: 1,
            minMatch: 4,
            strategy: 3, // ZSTD_greedy
            ..Default::default()
        });
        let mut seq = SeqStore_t::with_capacity(1024, 131072);
        let mut rep: [u32; ZSTD_REP_NUM] = [1, 4, 8];
        let last_lits = ZSTD_compressBlock_greedy(&mut ms, &mut seq, &mut rep, &src);
        assert!(!seq.sequences.is_empty(), "greedy emitted 0 sequences");
        let total = seq.literals.len() + last_lits;
        assert!(total < src.len(), "no savings: {total} vs {}", src.len());
    }

    #[test]
    fn lazy_and_lazy2_roundtrip_through_decoder() {
        use crate::decompress::zstd_decompress_block::{
            streaming_operation, ZSTD_DCtx, ZSTD_decoder_entropy_rep,
            ZSTD_decompressBlock_internal,
        };

        let src: Vec<u8> = b"To be, or not to be, that is the question. "
            .iter()
            .cycle()
            .take(2000)
            .copied()
            .collect();

        for depth in 0..=2u32 {
            let mut ms = ZSTD_MatchState_t::new(ZSTD_compressionParameters {
                windowLog: 17,
                hashLog: 12,
                chainLog: 12,
                searchLog: 2,
                minMatch: 4,
                strategy: 3 + depth,
                ..Default::default()
            });
            let mut seq = SeqStore_t::with_capacity(1024, 131072);
            let mut rep: [u32; ZSTD_REP_NUM] = [1, 4, 8];
            let last = ZSTD_compressBlock_lazy_noDict_generic(
                &mut ms, &mut seq, &mut rep, &src, 0, depth,
            );
            // Push tail literals into the store so the decoder sees them.
            seq.literals.extend_from_slice(&src[src.len() - last..]);

            // Emit a full compressed block and decode it.
            crate::compress::zstd_compress::ZSTD_seqToCodes(&mut seq);
            let prev = crate::compress::zstd_compress::ZSTD_entropyCTables_t::default();
            let mut next = crate::compress::zstd_compress::ZSTD_entropyCTables_t::default();
            let mut body = vec![0u8; 4096];
            let body_n = crate::compress::zstd_compress::ZSTD_entropyCompressSeqStore(
                &mut body,
                &mut seq,
                &prev,
                &mut next,
                3 + depth,
                0,
                src.len(),
                0,
            );
            if body_n == 0 {
                continue; // this depth picked raw fallback — skip decode
            }
            assert!(!crate::common::error::ERR_isError(body_n), "depth {depth} compress err");
            body.truncate(body_n);

            let mut dctx = ZSTD_DCtx::new();
            let mut entropy = ZSTD_decoder_entropy_rep::default();
            let mut out = vec![0u8; src.len() + 64];
            let decoded = ZSTD_decompressBlock_internal(
                &mut dctx,
                &mut entropy,
                &mut out,
                0,
                &body,
                streaming_operation::not_streaming,
            );
            assert!(!crate::common::error::ERR_isError(decoded),
                "depth {depth} decode err: {decoded:#x}");
            assert_eq!(decoded, src.len());
            assert_eq!(&out[..decoded], &src[..], "depth {depth} mismatch");
        }
    }

    #[test]
    fn insertDUBT1_smoke_zero_head() {
        // When the chain table is all zeros, matchIndex starts at 0,
        // which is ≤ windowLow, so the loop exits immediately and the
        // slots are nullified. Main thing: no panic.
        let cp = ZSTD_compressionParameters {
            hashLog: 10,
            chainLog: 10,
            minMatch: 4,
            windowLog: 17,
            strategy: 6,
            ..Default::default()
        };
        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.chainTable = vec![0u32; 1 << cp.chainLog];
        ms.window.lowLimit = 4;
        ms.window.dictLimit = 4;
        let src: Vec<u8> = (0..1024u32).map(|i| (i & 0xFF) as u8).collect();
        ZSTD_insertDUBT1(&mut ms, &src, 500, src.len(), 16, 4);
        // Slots at curr=500 should end at 0 (nullified).
        let slot = (2 * (500u32 & ((1u32 << 9) - 1))) as usize;
        assert_eq!(ms.chainTable[slot], 0);
        assert_eq!(ms.chainTable[slot + 1], 0);
    }

    #[test]
    fn updateDUBT_marks_queued_positions_with_unsorted_mark() {
        use crate::compress::match_state::ZSTD_DUBT_UNSORTED_MARK;
        let cp = ZSTD_compressionParameters {
            hashLog: 10,
            chainLog: 10,
            minMatch: 4,
            strategy: 6, // btlazy2
            ..Default::default()
        };
        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.chainTable = vec![0u32; 1 << cp.chainLog];
        ms.window.dictLimit = 0;
        ms.nextToUpdate = 0;
        let src: Vec<u8> = (0..1024u32).map(|i| (i & 0xFF) as u8).collect();
        ZSTD_updateDUBT(&mut ms, &src, 100, 4);
        assert_eq!(ms.nextToUpdate, 100);

        // Every odd-slot (the sortMark slot at idx*2+1) within the
        // btMask range should now hold UNSORTED_MARK.
        let btLog = cp.chainLog - 1;
        let btMask = (1u32 << btLog) - 1;
        let mut marked = 0;
        for idx in 0..100u32 {
            let slot = (2 * (idx & btMask)) as usize + 1;
            if ms.chainTable[slot] == ZSTD_DUBT_UNSORTED_MARK {
                marked += 1;
            }
        }
        assert_eq!(marked, 100, "all 100 queued positions should have UNSORTED_MARK");
    }

    #[test]
    fn insert_and_find_first_index_updates_chain() {
        let mut ms = ZSTD_MatchState_t::new(ZSTD_compressionParameters {
            hashLog: 10,
            chainLog: 10,
            minMatch: 4,
            ..Default::default()
        });
        // Resize chain table.
        ms.chainTable.resize(1 << 10, 0);
        let src: Vec<u8> = (0..128u8).collect();
        // Insert at positions 0..30 then look up head at 30.
        let head = ZSTD_insertAndFindFirstIndex(&mut ms, &src, 30);
        assert_eq!(ms.nextToUpdate, ms.window.base_offset + 30);
        // head must be either 0 (no collision) or a valid prior position.
        assert!(head < ms.window.base_offset + 30);
    }
}
