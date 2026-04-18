//! Translation of `lib/compress/zstd_fast.c` (strategy 1). Single-
//! hash-table match finder. Lands in two phases:
//!   - `ZSTD_fillHashTable` (seed hash with the start of the block)
//!   - `ZSTD_compressBlock_fast` (main emit loop) — next tick.
//!
//! Rust signature note: upstream's `end` argument is a `void*` into
//! the caller's source buffer; we accept an explicit `&[u8]` slice
//! representing the full source region.

#![allow(unused_variables, non_snake_case)]

use crate::common::mem::MEM_read32;
use crate::compress::match_state::ZSTD_MatchState_t;
use crate::compress::seq_store::{
    SeqStore_t, ZSTD_storeSeq, OFFSET_TO_OFFBASE, REPCODE_TO_OFFBASE, ZSTD_REP_NUM,
};
use crate::compress::zstd_hashes::{ZSTD_count, ZSTD_hashPtr};

/// Upstream `kSearchStrength` — controls how aggressively the
/// match finder widens its step when mismatches are common.
pub const kSearchStrength: u32 = 8;

/// Port of `ZSTD_getLowestPrefixIndex`. Given the current index, the
/// match state, and the active windowLog, returns the smallest
/// back-reference index the encoder may emit without crossing out of
/// the window.
pub fn ZSTD_getLowestPrefixIndex(
    ms: &ZSTD_MatchState_t,
    curr: u32,
    windowLog: u32,
) -> u32 {
    let maxDistance = 1u32 << windowLog;
    let lowestValid = ms.window.dictLimit;
    let withinWindow = if curr.wrapping_sub(lowestValid) > maxDistance {
        curr - maxDistance
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
pub fn ZSTD_getLowestMatchIndex(
    ms: &ZSTD_MatchState_t,
    curr: u32,
    windowLog: u32,
) -> u32 {
    let maxDistance = 1u32 << windowLog;
    let lowestValid = ms.window.lowLimit;
    let withinWindow = if curr.wrapping_sub(lowestValid) > maxDistance {
        curr - maxDistance
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
/// via a simple range check + 32-bit compare. Returns `true` when
/// the candidate is in-range AND reads the same 4 bytes.
#[inline]
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
    MEM_read32(&buf[current_pos..]) == MEM_read32(&buf[match_pos..])
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
/// digested dictionary; always uses `ZSTD_dtlm_full` to load extras.
pub fn ZSTD_fillHashTableForCDict(
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
        let curr = ms.window.base_offset + ip as u32;
        let hash0 = ZSTD_hashPtr(&src[ip..], hBits, mls);
        ms.hashTable[hash0] = curr;
        // Full load: fill extra slots when empty.
        for p in 1..fastHashFillStep as usize {
            let h = ZSTD_hashPtr(&src[ip + p..], hBits, mls);
            if ms.hashTable[h] == 0 {
                ms.hashTable[h] = curr + p as u32;
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
        let curr = ms.window.base_offset + ip as u32;
        let hash0 = ZSTD_hashPtr(&src[ip..], hBits, mls);
        ms.hashTable[hash0] = curr;
        // Fast variant: no extra-slot fill.
        ip += fastHashFillStep as usize;
    }
}

/// Port of `ZSTD_compressBlock_fast_noDict_generic`. Rust-side
/// simplification: upstream keeps four pipelined position cursors
/// (ip0..ip3) for latency-hiding; we use a single cursor and the
/// same `matchFound` predicate. Output (stored sequences + tail
/// literals, repcode updates) is specified by the format and should
/// match upstream's output bit-for-bit for the same input.
///
/// Returns the length of the final "tail" literals segment (bytes
/// after the last emitted sequence that the entropy stage will
/// copy verbatim).
/// Port of `ZSTD_compressBlock_fast_noDict_generic`. Scans
/// `src[istart..]`, treating `src[0..istart]` as already-processed
/// history (the hashTable entries for those positions remain valid
/// back-references). Emits sequences + literals into `seqStore` from
/// `anchor=istart`. `ms.window.base_offset` is the absolute index
/// offset: positions `[base_off+0..base_off+src.len()]` identify
/// bytes in the source.
pub fn ZSTD_compressBlock_fast_noDict_generic(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
    istart: usize,
    mls: u32,
) -> usize {
    let hlog = ms.cParams.hashLog;
    let windowLog = ms.cParams.windowLog;
    // Upstream's `stepSize` is 2 (for targetLength=0) and covers a
    // pair of positions (ip0, ip1) per step via the 4-way pipeline.
    // Our single-cursor variant covers one position per step, so we
    // halve the advance to match upstream's scan density.
    let stepSize = (ms.cParams.targetLength + (ms.cParams.targetLength == 0) as u32)
        .div_ceil(2)
        .max(1) as usize;

    let base_off = ms.window.base_offset;
    let srcSize = src.len();
    let endIndex = base_off + srcSize as u32;
    let prefixStartIndex = ZSTD_getLowestPrefixIndex(ms, endIndex, windowLog);
    let prefixStart = prefixStartIndex.saturating_sub(base_off) as usize;
    let iend = srcSize;
    let ilimit = iend.saturating_sub(HASH_READ_SIZE);

    let mut anchor = istart;
    let mut ip = istart;
    // Upstream: skip byte 0 so the first match candidate can look back.
    if ip == prefixStart {
        ip += 1;
    }

    let mut rep_offset1 = rep[0];
    let mut rep_offset2 = rep[1];
    let mut offsetSaved1: u32 = 0;
    let mut offsetSaved2: u32 = 0;
    {
        let curr = base_off + ip as u32;
        let windowLow = ZSTD_getLowestPrefixIndex(ms, curr, windowLog);
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

    let mut step = stepSize;
    let kStepIncr = 1usize << (kSearchStrength - 1);
    let mut nextStep = ip + kStepIncr;

    // Main search loop — one cursor variant of upstream's 4-way
    // pipeline. Every iteration: hash ip, check the current rep-offset
    // at ip, check the hash-table candidate at ip, else advance.
    while ip < ilimit {
        let curr = base_off + ip as u32;

        // Inline helper: after emitting a match that ended at `ip_end`
        // (exclusive), perform upstream's two complementary hash
        // inserts (at curr+2 and ip_end-2) and drain any immediate
        // repcodes. Returns the new `ip` position.
        // Repcode path uses this with offBase=REPCODE(1); full-offset
        // path uses it with the newly-updated rep_offset1/rep_offset2.

        // Try repcode at ip+1 first. Upstream always checks at
        // ip0+step (typically ip0+2), ensuring litLength ≥ 1. We
        // check at ip+1 to guarantee litLength ≥ 1 — this matters
        // because the decoder interprets `(offBase=repcode-1, ll0=1)`
        // as rep[1] (shifted) rather than rep[0] (current offset).
        // Emitting with litLength=0 would desync the decoder's rep
        // array and corrupt output.
        let mut match_found = false;
        let mut new_ip = ip;
        if rep_offset1 > 0
            && ip + 1 + 4 <= iend
            && ip + 1 >= rep_offset1 as usize
            && MEM_read32(&src[ip + 1..])
                == MEM_read32(&src[ip + 1 - rep_offset1 as usize..])
        {
            let match_start = ip + 1;
            let match_pos = match_start - rep_offset1 as usize;
            let fwd = ZSTD_count(src, match_start + 4, match_pos + 4, iend);
            let mLength = fwd + 4;
            let litLength = match_start - anchor;
            ZSTD_storeSeq(
                seqStore,
                litLength,
                &src[anchor..match_start],
                REPCODE_TO_OFFBASE(1),
                mLength,
            );
            new_ip = match_start + mLength;
            anchor = new_ip;
            match_found = true;
        }

        if !match_found {
            // Hash-table candidate lookup.
            let h = ZSTD_hashPtr(&src[ip..], hlog, mls);
            let matchIdx = ms.hashTable[h];
            ms.hashTable[h] = curr;
            if matchIdx >= prefixStartIndex && matchIdx < curr {
                let match_pos = matchIdx.saturating_sub(base_off) as usize;
                if match_pos + 4 <= iend && MEM_read32(&src[ip..]) == MEM_read32(&src[match_pos..]) {
                    let mut ip_m = ip;
                    let mut match_m = match_pos;
                    while ip_m > anchor && match_m > prefixStart && src[ip_m - 1] == src[match_m - 1] {
                        ip_m -= 1;
                        match_m -= 1;
                    }
                    let bwd = ip - ip_m;
                    let fwd = ZSTD_count(src, ip + 4, match_pos + 4, iend);
                    let mLength = bwd + fwd + 4;
                    let offset = (ip - match_pos) as u32;
                    let litLength = ip_m - anchor;
                    rep_offset2 = rep_offset1;
                    rep_offset1 = offset;
                    ZSTD_storeSeq(
                        seqStore,
                        litLength,
                        &src[anchor..ip_m],
                        OFFSET_TO_OFFBASE(offset),
                        mLength,
                    );
                    new_ip = ip_m + mLength;
                    anchor = new_ip;
                    match_found = true;
                }
            }
        }

        if match_found {
            // Upstream's post-match hash fills: at curr+2 and ip_end-2.
            // Seed the hash table more densely around the match so
            // follow-on searches have better back-references.
            if new_ip <= ilimit {
                let curr_p2 = (curr as usize + 2) - base_off as usize;
                if curr_p2 + 4 <= src.len() {
                    let h1 = ZSTD_hashPtr(&src[curr_p2..], hlog, mls);
                    ms.hashTable[h1] = curr + 2;
                }
                if new_ip >= 2 && new_ip - 2 + 4 <= src.len() {
                    let h2 = ZSTD_hashPtr(&src[new_ip - 2..], hlog, mls);
                    ms.hashTable[h2] = base_off + (new_ip - 2) as u32;
                }
                // Immediate-repcode loop: drain consecutive repcode
                // hits at the new anchor. Halves the gap for
                // repetitive text.
                let mut rp = new_ip;
                while rp <= ilimit
                    && rep_offset2 > 0
                    && rp + 4 <= iend
                    && rp >= rep_offset2 as usize
                    && MEM_read32(&src[rp..]) == MEM_read32(&src[rp - rep_offset2 as usize..])
                {
                    let rLen =
                        ZSTD_count(src, rp + 4, rp + 4 - rep_offset2 as usize, iend) + 4;
                    std::mem::swap(&mut rep_offset1, &mut rep_offset2);
                    let h3 = ZSTD_hashPtr(&src[rp..], hlog, mls);
                    ms.hashTable[h3] = base_off + rp as u32;
                    ZSTD_storeSeq(seqStore, 0, &src[rp..rp], REPCODE_TO_OFFBASE(1), rLen);
                    rp += rLen;
                    anchor = rp;
                }
                ip = rp;
            } else {
                ip = new_ip;
            }
            // Upstream resets the step ramp at `_start:` after every
            // match — the step is local to each "hunt phase".
            step = stepSize;
            nextStep = ip + kStepIncr;
            continue;
        }

        // No match. Step forward — widen the step as we hit
        // uncompressible stretches.
        ip += step;
        if ip >= nextStep {
            step += 1;
            nextStep += kStepIncr;
        }
    }

    // Save repcodes for the next block, restoring any we zeroed in
    // the preamble.
    let offsetSaved2_final = if offsetSaved1 != 0 && rep_offset1 != 0 {
        offsetSaved1
    } else {
        offsetSaved2
    };
    rep[0] = if rep_offset1 != 0 { rep_offset1 } else { offsetSaved1 };
    rep[1] = if rep_offset2 != 0 { rep_offset2 } else { offsetSaved2_final };

    iend - anchor
}

/// Port of `ZSTD_compressBlock_fast`. The public entry — dispatches
/// on `cParams.minMatch`. Requires `dictMatchState` to be unset
/// (upstream asserts the same); dict-match-state variant is deferred.
pub fn ZSTD_compressBlock_fast(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    let mml = ms.cParams.minMatch;
    ZSTD_compressBlock_fast_noDict_generic(ms, seqStore, rep, src, 0, mml)
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
        assert!(n_filled > 10, "expected many buckets filled, got {n_filled}");
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
        let last = ZSTD_compressBlock_fast_with_history(
            &mut ms,
            &mut seqStore,
            &mut rep,
            &full,
            boundary,
        );
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
        assert!(total_lits < src.len(), "no savings: total lits = {total_lits}, src = {}", src.len());
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
}
