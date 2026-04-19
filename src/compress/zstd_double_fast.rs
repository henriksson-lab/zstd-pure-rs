//! Translation of `lib/compress/zstd_double_fast.c` (strategy 2).
//!
//! Two parallel hash tables: the "long" table hashed on 8 bytes
//! (stored in `ms.hashTable`) and the "short" table hashed on `mls`
//! bytes (stored in `ms.chainTable`, reused). Each scan position
//! checks repcode, then long match (preferred), then short match
//! (with a short-match-then-check-long-at-next-pos refinement).

#![allow(non_snake_case)]

use crate::common::mem::{MEM_read32, MEM_read64};
use crate::compress::match_state::ZSTD_MatchState_t;
use crate::compress::seq_store::{
    SeqStore_t, ZSTD_storeSeq, OFFSET_TO_OFFBASE, REPCODE_TO_OFFBASE, ZSTD_REP_NUM,
};
use crate::compress::zstd_fast::{
    kSearchStrength, ZSTD_getLowestPrefixIndex, HASH_READ_SIZE,
    ZSTD_dictTableLoadMethod_e, ZSTD_tableFillPurpose_e,
};
use crate::compress::zstd_hashes::{ZSTD_count, ZSTD_hashPtr};

/// Port of `ZSTD_fillDoubleHashTableForCCtx`. Seeds both the long
/// (`hashTable`, hashed on 8 bytes) and short (`chainTable`, hashed
/// on `minMatch` bytes) tables at stride `fastHashFillStep=3`.
/// `dtlm_full` additionally fills empty slots at intermediate
/// positions in the long table.
pub fn ZSTD_fillDoubleHashTableForCCtx(
    ms: &mut ZSTD_MatchState_t,
    src: &[u8],
    dtlm: ZSTD_dictTableLoadMethod_e,
) {
    let hBitsL = ms.cParams.hashLog;
    let hBitsS = ms.cParams.chainLog;
    let mls = ms.cParams.minMatch;
    let fastHashFillStep: usize = 3;

    if src.len() < HASH_READ_SIZE {
        return;
    }
    let iend = src.len() - HASH_READ_SIZE;

    // Ensure chainTable (short hash) is allocated.
    let chainSize = 1usize << hBitsS;
    if ms.chainTable.len() < chainSize {
        ms.chainTable.resize(chainSize, 0);
    }

    let mut ip = ms.nextToUpdate.saturating_sub(ms.window.base_offset) as usize;
    while ip + fastHashFillStep <= iend + 1 {
        let curr = ms.window.base_offset + ip as u32;
        for i in 0..fastHashFillStep {
            if ip + i + 8 > src.len() {
                break;
            }
            let smHash = ZSTD_hashPtr(&src[ip + i..], hBitsS, mls);
            let lgHash = ZSTD_hashPtr(&src[ip + i..], hBitsL, 8);
            if i == 0 {
                ms.chainTable[smHash] = curr + i as u32;
            }
            if i == 0 || ms.hashTable[lgHash] == 0 {
                ms.hashTable[lgHash] = curr + i as u32;
            }
            if dtlm == ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_fast {
                break;
            }
        }
        ip += fastHashFillStep;
    }
}

/// Port of `ZSTD_fillDoubleHashTable`. Dispatches on purpose (CCtx
/// vs CDict). The CDict variant is not yet specialized (upstream
/// uses `ZSTD_SHORT_CACHE_TAG_BITS` tagging that we haven't ported);
/// we fall back to the CCtx fill which is correct but omits the
/// tag-cache optimization.
pub fn ZSTD_fillDoubleHashTable(
    ms: &mut ZSTD_MatchState_t,
    src: &[u8],
    dtlm: ZSTD_dictTableLoadMethod_e,
    _tfp: ZSTD_tableFillPurpose_e,
) {
    ZSTD_fillDoubleHashTableForCCtx(ms, src, dtlm);
}

/// Port of `ZSTD_compressBlock_doubleFast_noDict_generic` — single-
/// cursor simplification of upstream's `ip`/`ip1` double-cursor
/// pipeline. Per scan position:
///   1. Check repcode at `ip+1`; take it if match.
///   2. Check long match at `ip` (8-byte hash); take it if in-range.
///   3. Check short match at `ip` (mls-byte hash); if match, also
///      probe long match at `ip+1` to see if a larger match lives
///      one byte ahead, prefer whichever is longer.
///   4. Advance by `step`, ramping step via `kSearchStrength`.
///
/// Uses the same `matchIdx < curr` + `match_pos + 4 <= iend` guards
/// as `ZSTD_compressBlock_fast_noDict_generic`.
pub fn ZSTD_compressBlock_doubleFast_noDict_generic(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
    istart: usize,
    mls: u32,
) -> usize {
    let hBitsL = ms.cParams.hashLog;
    let hBitsS = ms.cParams.chainLog;
    let windowLog = ms.cParams.windowLog;
    let stepSize = 2usize;

    // Ensure chainTable is sized for hBitsS.
    let chainSize = 1usize << hBitsS;
    if ms.chainTable.len() < chainSize {
        ms.chainTable.resize(chainSize, 0);
    }

    let base_off = ms.window.base_offset;
    let srcSize = src.len();
    let endIndex = base_off + srcSize as u32;
    let prefixStartIndex = ZSTD_getLowestPrefixIndex(ms, endIndex, windowLog);
    let prefixStart = prefixStartIndex.saturating_sub(base_off) as usize;
    let iend = srcSize;
    let ilimit = iend.saturating_sub(HASH_READ_SIZE);

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

    // Closure helper: after emitting a match that ended at `end_ip`,
    // apply upstream's complementary-insert pattern (hashLong at
    // curr+2 and end_ip-2; hashSmall at curr+2 and end_ip-1) plus
    // the immediate-repcode drain loop. Returns the new ip position.
    let complementary_insert_and_rep_drain =
        |ms: &mut ZSTD_MatchState_t,
         seqStore: &mut SeqStore_t,
         rep_offset1: &mut u32,
         rep_offset2: &mut u32,
         curr_at_match: u32,
         end_ip: usize,
         anchor: &mut usize|
         -> usize {
            if end_ip > ilimit {
                return end_ip;
            }
            let curr_p2 = (curr_at_match as usize + 2).saturating_sub(base_off as usize);
            if curr_p2 + 8 <= src.len() {
                let hL = ZSTD_hashPtr(&src[curr_p2..], hBitsL, 8);
                ms.hashTable[hL] = curr_at_match + 2;
                let hS = ZSTD_hashPtr(&src[curr_p2..], hBitsS, mls);
                ms.chainTable[hS] = curr_at_match + 2;
            }
            if end_ip >= 2 && end_ip - 2 + 8 <= src.len() {
                let hL = ZSTD_hashPtr(&src[end_ip - 2..], hBitsL, 8);
                ms.hashTable[hL] = base_off + (end_ip - 2) as u32;
            }
            if end_ip >= 1 && end_ip - 1 + mls as usize <= src.len() {
                let hS = ZSTD_hashPtr(&src[end_ip - 1..], hBitsS, mls);
                ms.chainTable[hS] = base_off + (end_ip - 1) as u32;
            }
            let mut rp = end_ip;
            while rp <= ilimit
                && *rep_offset2 > 0
                && rp + 4 <= iend
                && rp >= *rep_offset2 as usize
                && MEM_read32(&src[rp..]) == MEM_read32(&src[rp - *rep_offset2 as usize..])
            {
                let rLen =
                    ZSTD_count(src, rp + 4, rp + 4 - *rep_offset2 as usize, iend) + 4;
                std::mem::swap(rep_offset1, rep_offset2);
                let hS = ZSTD_hashPtr(&src[rp..], hBitsS, mls);
                ms.chainTable[hS] = base_off + rp as u32;
                let hL = ZSTD_hashPtr(&src[rp..], hBitsL, 8);
                ms.hashTable[hL] = base_off + rp as u32;
                ZSTD_storeSeq(seqStore, 0, &src[rp..rp], REPCODE_TO_OFFBASE(1), rLen);
                rp += rLen;
                *anchor = rp;
            }
            rp
        };

    while ip < ilimit {
        let curr = base_off + ip as u32;

        // 1. Repcode at ip+1.
        if rep_offset1 > 0
            && ip + 1 + 4 <= iend
            && ip + 1 >= rep_offset1 as usize
            && MEM_read32(&src[ip + 1..]) == MEM_read32(&src[ip + 1 - rep_offset1 as usize..])
        {
            let match_pos = ip + 1 - rep_offset1 as usize;
            let fwd = ZSTD_count(src, ip + 1 + 4, match_pos + 4, iend);
            let mLength = fwd + 4;
            let litLength = ip + 1 - anchor;
            ZSTD_storeSeq(
                seqStore,
                litLength,
                &src[anchor..ip + 1],
                REPCODE_TO_OFFBASE(1),
                mLength,
            );
            let end_ip = ip + 1 + mLength;
            anchor = end_ip;
            ip = complementary_insert_and_rep_drain(
                ms,
                seqStore,
                &mut rep_offset1,
                &mut rep_offset2,
                curr + 1,
                end_ip,
                &mut anchor,
            );
            step = stepSize;
            nextStep = ip + kStepIncr;
            continue;
        }

        // 2/3. Long then short hash-table lookups.
        let hl0 = ZSTD_hashPtr(&src[ip..], hBitsL, 8);
        let hs0 = ZSTD_hashPtr(&src[ip..], hBitsS, mls);
        let idxl0 = ms.hashTable[hl0];
        let idxs0 = ms.chainTable[hs0];
        // Update tables with current position.
        ms.hashTable[hl0] = curr;
        ms.chainTable[hs0] = curr;

        // Prefer long match.
        if idxl0 >= prefixStartIndex && idxl0 < curr {
            let match_pos = idxl0.saturating_sub(base_off) as usize;
            if match_pos + 8 <= iend && MEM_read64(&src[ip..]) == MEM_read64(&src[match_pos..]) {
                // Take the long match, extend.
                let fwd = ZSTD_count(src, ip + 8, match_pos + 8, iend);
                let mLength = fwd + 8;
                let mut ip_m = ip;
                let mut match_m = match_pos;
                let mut mLen = mLength;
                while ip_m > anchor
                    && match_m > prefixStart
                    && src[ip_m - 1] == src[match_m - 1]
                {
                    ip_m -= 1;
                    match_m -= 1;
                    mLen += 1;
                }
                let offset = (ip - match_pos) as u32;
                let litLength = ip_m - anchor;
                rep_offset2 = rep_offset1;
                rep_offset1 = offset;
                ZSTD_storeSeq(
                    seqStore,
                    litLength,
                    &src[anchor..ip_m],
                    OFFSET_TO_OFFBASE(offset),
                    mLen,
                );
                let end_ip = ip_m + mLen;
                anchor = end_ip;
                ip = complementary_insert_and_rep_drain(
                    ms,
                    seqStore,
                    &mut rep_offset1,
                    &mut rep_offset2,
                    curr,
                    end_ip,
                    &mut anchor,
                );
                step = stepSize;
                nextStep = ip + kStepIncr;
                continue;
            }
        }

        // Short match?
        if idxs0 >= prefixStartIndex && idxs0 < curr {
            let match_pos = idxs0.saturating_sub(base_off) as usize;
            if match_pos + 4 <= iend
                && MEM_read32(&src[ip..]) == MEM_read32(&src[match_pos..])
            {
                let mut offset = (ip - match_pos) as u32;
                let fwd = ZSTD_count(src, ip + 4, match_pos + 4, iend);
                let mut mLen = fwd + 4;
                let mut ip_m = ip;
                let mut match_m = match_pos;

                // Refinement: is there a longer long-match at ip+1?
                if ip + 1 < ilimit {
                    let hl1 = ZSTD_hashPtr(&src[ip + 1..], hBitsL, 8);
                    let idxl1 = ms.hashTable[hl1];
                    if idxl1 >= prefixStartIndex && idxl1 < base_off + (ip + 1) as u32 {
                        let mp1 = idxl1.saturating_sub(base_off) as usize;
                        if mp1 + 8 <= iend
                            && MEM_read64(&src[ip + 1..]) == MEM_read64(&src[mp1..])
                        {
                            let l1 = ZSTD_count(src, ip + 1 + 8, mp1 + 8, iend) + 8;
                            if l1 > mLen {
                                ip_m = ip + 1;
                                match_m = mp1;
                                mLen = l1;
                                offset = (ip_m - mp1) as u32;
                            }
                        }
                    }
                }
                // Back-extend.
                while ip_m > anchor
                    && match_m > prefixStart
                    && src[ip_m - 1] == src[match_m - 1]
                {
                    ip_m -= 1;
                    match_m -= 1;
                    mLen += 1;
                }
                let litLength = ip_m - anchor;
                rep_offset2 = rep_offset1;
                rep_offset1 = offset;
                ZSTD_storeSeq(
                    seqStore,
                    litLength,
                    &src[anchor..ip_m],
                    OFFSET_TO_OFFBASE(offset),
                    mLen,
                );
                let end_ip = ip_m + mLen;
                anchor = end_ip;
                ip = complementary_insert_and_rep_drain(
                    ms,
                    seqStore,
                    &mut rep_offset1,
                    &mut rep_offset2,
                    curr,
                    end_ip,
                    &mut anchor,
                );
                step = stepSize;
                nextStep = ip + kStepIncr;
                continue;
            }
        }

        // No match: advance with step ramp.
        ip += step;
        if ip >= nextStep {
            step += 1;
            nextStep += kStepIncr;
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

/// Port of `ZSTD_compressBlock_doubleFast` (public entry).
pub fn ZSTD_compressBlock_doubleFast(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    let mml = ms.cParams.minMatch;
    ZSTD_compressBlock_doubleFast_noDict_generic(ms, seqStore, rep, src, 0, mml)
}

/// Port of `ZSTD_compressBlock_doubleFast_dictMatchState`
/// (`zstd_double_fast.c:483`). **NOT YET PORTED** — requires
/// `ms.dictMatchState` linkage. Returns `ErrorCode::Generic`.
pub fn ZSTD_compressBlock_doubleFast_dictMatchState(
    _ms: &mut ZSTD_MatchState_t,
    _seqStore: &mut SeqStore_t,
    _rep: &mut [u32; ZSTD_REP_NUM],
    _src: &[u8],
) -> usize {
    crate::common::error::ERROR(crate::common::error::ErrorCode::Generic)
}

/// Port of `ZSTD_compressBlock_doubleFast_extDict` (`zstd_double_fast.c:709`).
/// **NOT YET PORTED** — requires ext-dict pointer tracking.
/// Returns `ErrorCode::Generic`.
pub fn ZSTD_compressBlock_doubleFast_extDict(
    _ms: &mut ZSTD_MatchState_t,
    _seqStore: &mut SeqStore_t,
    _rep: &mut [u32; ZSTD_REP_NUM],
    _src: &[u8],
) -> usize {
    crate::common::error::ERROR(crate::common::error::ErrorCode::Generic)
}

/// Cross-block-history variant. Treats `src[..istart]` as prior
/// content; hashTable / chainTable entries from earlier blocks stay
/// valid as back-references.
pub fn ZSTD_compressBlock_doubleFast_with_history(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
    istart: usize,
) -> usize {
    let mml = ms.cParams.minMatch;
    ZSTD_compressBlock_doubleFast_noDict_generic(ms, seqStore, rep, src, istart, mml)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::match_state::ZSTD_compressionParameters;

    #[test]
    fn fill_double_hash_table_populates_both_tables() {
        let mut ms = ZSTD_MatchState_t::new(ZSTD_compressionParameters {
            hashLog: 10,
            chainLog: 10,
            minMatch: 4,
            ..Default::default()
        });
        let src: Vec<u8> = (0..128u8).collect();
        ZSTD_fillDoubleHashTable(
            &mut ms,
            &src,
            ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_fast,
            ZSTD_tableFillPurpose_e::ZSTD_tfp_forCCtx,
        );
        let n_long = ms.hashTable.iter().filter(|&&c| c != 0).count();
        let n_short = ms.chainTable.iter().filter(|&&c| c != 0).count();
        assert!(n_long > 10, "long table empty: {n_long}");
        assert!(n_short > 10, "short table empty: {n_short}");
    }

    #[test]
    fn compress_block_double_fast_emits_sequences() {
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
            minMatch: 4,
            strategy: 2,
            ..Default::default()
        });
        let mut seq = SeqStore_t::with_capacity(1024, 131072);
        let mut rep: [u32; ZSTD_REP_NUM] = [1, 4, 8];
        let last_lits = ZSTD_compressBlock_doubleFast(&mut ms, &mut seq, &mut rep, &src);
        assert!(!seq.sequences.is_empty(), "match finder emitted 0 sequences");
        let total_lits = seq.literals.len() + last_lits;
        assert!(total_lits < src.len(), "no savings: lits={total_lits}, src={}", src.len());
    }
}
