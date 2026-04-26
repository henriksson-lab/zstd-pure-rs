//! Translation of `lib/compress/zstd_double_fast.c` (strategy 2).
//!
//! Two parallel hash tables: the "long" table hashed on 8 bytes
//! (stored in `ms.hashTable`) and the "short" table hashed on `mls`
//! bytes (stored in `ms.chainTable`, reused). Each scan position
//! checks repcode, then long match (preferred), then short match
//! (with a short-match-then-check-long-at-next-pos refinement).

#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]

use crate::common::mem::{MEM_read32, MEM_read64};
use crate::compress::match_state::{
    ZSTD_MatchState_t, ZSTD_comparePackedTags, ZSTD_index_overlap_check, ZSTD_window_hasExtDict,
    ZSTD_writeTaggedIndex, ZSTD_SHORT_CACHE_TAG_BITS,
};
use crate::compress::seq_store::{
    SeqStore_t, ZSTD_storeSeq, OFFSET_TO_OFFBASE, REPCODE_TO_OFFBASE, ZSTD_REP_NUM,
};
use crate::compress::zstd_fast::{
    kSearchStrength, ZSTD_dictTableLoadMethod_e, ZSTD_getLowestMatchIndex,
    ZSTD_getLowestPrefixIndex, ZSTD_tableFillPurpose_e, HASH_READ_SIZE,
};
use crate::compress::zstd_hashes::{ZSTD_count, ZSTD_count_2segments, ZSTD_hashPtr};

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
        let curr = ms.window.base_offset.wrapping_add(ip as u32);
        for i in 0..fastHashFillStep {
            if ip + i + 8 > src.len() {
                break;
            }
            let smHash = ZSTD_hashPtr(&src[ip + i..], hBitsS, mls);
            let lgHash = ZSTD_hashPtr(&src[ip + i..], hBitsL, 8);
            if i == 0 {
                ms.chainTable[smHash] = curr.wrapping_add(i as u32);
            }
            if i == 0 || ms.hashTable[lgHash] == 0 {
                ms.hashTable[lgHash] = curr.wrapping_add(i as u32);
            }
            if dtlm == ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_fast {
                break;
            }
        }
        ip += fastHashFillStep;
    }
}

/// Port of `ZSTD_fillDoubleHashTableForCDict`. CDict loading uses the
/// short-cache tagged-index format: the hash is computed with
/// `hashLog/chainLog + ZSTD_SHORT_CACHE_TAG_BITS`, then packed into
/// the table entry with `ZSTD_writeTaggedIndex()`.
pub fn ZSTD_fillDoubleHashTableForCDict(
    ms: &mut ZSTD_MatchState_t,
    src: &[u8],
    dtlm: ZSTD_dictTableLoadMethod_e,
) {
    let hBitsL = ms.cParams.hashLog + ZSTD_SHORT_CACHE_TAG_BITS;
    let hBitsS = ms.cParams.chainLog + ZSTD_SHORT_CACHE_TAG_BITS;
    let mls = ms.cParams.minMatch;
    let fastHashFillStep: usize = 3;

    if src.len() < HASH_READ_SIZE {
        return;
    }

    let chainSize = 1usize << ms.cParams.chainLog;
    if ms.chainTable.len() < chainSize {
        ms.chainTable.resize(chainSize, 0);
    }

    let iend = src.len() - HASH_READ_SIZE;
    let mut ip = ms.nextToUpdate.saturating_sub(ms.window.base_offset) as usize;
    while ip + fastHashFillStep <= iend + 1 {
        let curr = ms.window.base_offset.wrapping_add(ip as u32);
        for i in 0..fastHashFillStep {
            if ip + i + 8 > src.len() {
                break;
            }
            let smHashAndTag = ZSTD_hashPtr(&src[ip + i..], hBitsS, mls);
            let lgHashAndTag = ZSTD_hashPtr(&src[ip + i..], hBitsL, 8);
            if i == 0 {
                ZSTD_writeTaggedIndex(
                    &mut ms.chainTable,
                    smHashAndTag,
                    curr.wrapping_add(i as u32),
                );
            }
            let lgSlot = lgHashAndTag >> ZSTD_SHORT_CACHE_TAG_BITS;
            if i == 0 || ms.hashTable[lgSlot] == 0 {
                ZSTD_writeTaggedIndex(
                    &mut ms.hashTable,
                    lgHashAndTag,
                    curr.wrapping_add(i as u32),
                );
            }
            if dtlm == ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_fast {
                break;
            }
        }
        ip += fastHashFillStep;
    }
}

/// Port of `ZSTD_fillDoubleHashTable`. Dispatches on purpose (CCtx
/// vs CDict); the CDict path uses tagged cache entries, matching the
/// upstream fast/dfast dictionary loader.
pub fn ZSTD_fillDoubleHashTable(
    ms: &mut ZSTD_MatchState_t,
    src: &[u8],
    dtlm: ZSTD_dictTableLoadMethod_e,
    tfp: ZSTD_tableFillPurpose_e,
) {
    match tfp {
        ZSTD_tableFillPurpose_e::ZSTD_tfp_forCCtx => ZSTD_fillDoubleHashTableForCCtx(ms, src, dtlm),
        ZSTD_tableFillPurpose_e::ZSTD_tfp_forCDict => {
            ZSTD_fillDoubleHashTableForCDict(ms, src, dtlm)
        }
    }
}

/// Port of `ZSTD_compressBlock_doubleFast_noDict_generic`. Mirrors
/// upstream's two-cursor `ip` / `ip1` pipeline, including the
/// repcode-at-`ip+1` check, long-match preference, short-match then
/// `+1` long-match refinement, and the complementary insertion /
/// immediate-repcode drain after each emitted match.
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

    let chainSize = 1usize << hBitsS;
    if ms.chainTable.len() < chainSize {
        ms.chainTable.resize(chainSize, 0);
    }

    let base_off = ms.window.base_offset;
    let srcSize = src.len();
    let endIndex = base_off.wrapping_add(srcSize as u32);
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
        let curr = base_off.wrapping_add(ip as u32);
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
    let dummy = [0x12u8, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0xe2, 0xb4];

    'outer: loop {
        let mut step = 1usize;
        let mut nextStep = ip + kStepIncr;
        let mut ip1 = ip + step;

        if ip1 > ilimit {
            break 'outer;
        }

        let mut hl0 = ZSTD_hashPtr(&src[ip..], hBitsL, 8);
        let mut idxl0 = ms.hashTable[hl0];
        let mut matchl0 = idxl0.saturating_sub(base_off) as usize;

        let (mLength, offset, curr) = 'search: loop {
            let hs0 = ZSTD_hashPtr(&src[ip..], hBitsS, mls);
            let idxs0 = ms.chainTable[hs0];
            let curr = base_off.wrapping_add(ip as u32);
            let matchs0 = idxs0.saturating_sub(base_off) as usize;

            ms.hashTable[hl0] = curr;
            ms.chainTable[hs0] = curr;

            if rep_offset1 > 0
                && ip + 1 >= rep_offset1 as usize
                && ip + 1 + 4 <= iend
                && MEM_read32(&src[ip + 1 - rep_offset1 as usize..]) == MEM_read32(&src[ip + 1..])
            {
                let mLength =
                    ZSTD_count(src, ip + 1 + 4, ip + 1 + 4 - rep_offset1 as usize, iend) + 4;
                ip += 1;
                ZSTD_storeSeq(
                    seqStore,
                    ip - anchor,
                    &src[anchor..ip],
                    REPCODE_TO_OFFBASE(1),
                    mLength,
                );
                break 'search (mLength, 0, curr);
            }

            let hl1 = ZSTD_hashPtr(&src[ip1..], hBitsL, 8);

            let long_match_valid = idxl0 > prefixStartIndex && idxl0 < curr && matchl0 + 8 <= iend;
            let matchl0_safe = if long_match_valid {
                &src[matchl0..]
            } else {
                &dummy[..]
            };
            if long_match_valid && MEM_read64(matchl0_safe) == MEM_read64(&src[ip..]) {
                let mut mLength = ZSTD_count(src, ip + 8, matchl0 + 8, iend) + 8;
                let offset = (ip - matchl0) as u32;
                while ip > anchor && matchl0 > prefixStart && src[ip - 1] == src[matchl0 - 1] {
                    ip -= 1;
                    matchl0 -= 1;
                    mLength += 1;
                }
                break 'search (mLength, offset, curr);
            }

            let idxl1 = ms.hashTable[hl1];
            let matchl1 = idxl1.saturating_sub(base_off) as usize;

            let short_match_valid = idxs0 > prefixStartIndex && idxs0 < curr && matchs0 + 4 <= iend;
            let matchs0_safe = if short_match_valid {
                &src[matchs0..]
            } else {
                &dummy[..]
            };
            if short_match_valid && MEM_read32(matchs0_safe) == MEM_read32(&src[ip..]) {
                let mut mLength = ZSTD_count(src, ip + 4, matchs0 + 4, iend) + 4;
                let mut offset = (ip - matchs0) as u32;
                let mut match_pos = matchs0;

                if idxl1 > prefixStartIndex
                    && idxl1 < base_off.wrapping_add(ip1 as u32)
                    && matchl1 + 8 <= iend
                    && MEM_read64(&src[matchl1..]) == MEM_read64(&src[ip1..])
                {
                    let l1len = ZSTD_count(src, ip1 + 8, matchl1 + 8, iend) + 8;
                    if l1len > mLength {
                        ip = ip1;
                        mLength = l1len;
                        offset = (ip - matchl1) as u32;
                        match_pos = matchl1;
                    }
                }

                while ip > anchor && match_pos > prefixStart && src[ip - 1] == src[match_pos - 1]
                {
                    ip -= 1;
                    match_pos -= 1;
                    mLength += 1;
                }
                break 'search (mLength, offset, curr);
            }

            if ip1 >= nextStep {
                step += 1;
                crate::common::zstd_internal::prefetchSliceByte(src, ip1 + 64);
                crate::common::zstd_internal::prefetchSliceByte(src, ip1 + 128);
                nextStep += kStepIncr;
            }
            ip = ip1;
            ip1 += step;
            hl0 = hl1;
            idxl0 = idxl1;
            matchl0 = matchl1;

            if ip1 > ilimit {
                break 'outer;
            }
        };

        if offset != 0 {
            rep_offset2 = rep_offset1;
            rep_offset1 = offset;
            if step < 4 {
                let hl1 = ZSTD_hashPtr(&src[ip1..], hBitsL, 8);
                ms.hashTable[hl1] = base_off.wrapping_add(ip1 as u32);
            }
            ZSTD_storeSeq(
                seqStore,
                ip - anchor,
                &src[anchor..ip],
                OFFSET_TO_OFFBASE(offset),
                mLength,
            );
        }

        ip += mLength;
        anchor = ip;

        if ip <= ilimit {
            let indexToInsert = curr.wrapping_add(2);
            let indexToInsertPos = indexToInsert.saturating_sub(base_off) as usize;
            ms.hashTable[ZSTD_hashPtr(&src[indexToInsertPos..], hBitsL, 8)] = indexToInsert;
            ms.hashTable[ZSTD_hashPtr(&src[ip - 2..], hBitsL, 8)] =
                base_off.wrapping_add((ip - 2) as u32);
            ms.chainTable[ZSTD_hashPtr(&src[indexToInsertPos..], hBitsS, mls)] = indexToInsert;
            ms.chainTable[ZSTD_hashPtr(&src[ip - 1..], hBitsS, mls)] =
                base_off.wrapping_add((ip - 1) as u32);

            while ip <= ilimit
                && rep_offset2 > 0
                && MEM_read32(&src[ip..]) == MEM_read32(&src[ip - rep_offset2 as usize..])
            {
                let rLength = ZSTD_count(src, ip + 4, ip + 4 - rep_offset2 as usize, iend) + 4;
                std::mem::swap(&mut rep_offset1, &mut rep_offset2);
                ms.chainTable[ZSTD_hashPtr(&src[ip..], hBitsS, mls)] =
                    base_off.wrapping_add(ip as u32);
                ms.hashTable[ZSTD_hashPtr(&src[ip..], hBitsL, 8)] =
                    base_off.wrapping_add(ip as u32);
                ZSTD_storeSeq(seqStore, 0, &src[anchor..anchor], REPCODE_TO_OFFBASE(1), rLength);
                ip += rLength;
                anchor = ip;
            }
        }
    }

    offsetSaved2 = if offsetSaved1 != 0 && rep_offset1 != 0 {
        offsetSaved1
    } else {
        offsetSaved2
    };
    rep[0] = if rep_offset1 != 0 { rep_offset1 } else { offsetSaved1 };
    rep[1] = if rep_offset2 != 0 { rep_offset2 } else { offsetSaved2 };
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
/// (`zstd_double_fast.c:588`). The dedicated dict-match-state entry
/// dispatches into the specialized generic implementation below.
pub fn ZSTD_compressBlock_doubleFast_dictMatchState(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_doubleFast_dictMatchState_generic(ms, seqStore, rep, src)
}

/// Port of `ZSTD_compressBlock_doubleFast_dictMatchState_generic`.
pub fn ZSTD_compressBlock_doubleFast_dictMatchState_generic(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    let cParams = ms.cParams;
    let mls = cParams.minMatch;
    let endIndex = ms.window.base_offset.wrapping_add(src.len() as u32);
    let prefixLowestIndex = ZSTD_getLowestPrefixIndex(ms, endIndex, cParams.windowLog);
    let base_off = ms.window.base_offset;
    let prefixLowest = prefixLowestIndex.saturating_sub(base_off) as usize;
    let dms = match ms.dictMatchState.as_deref() {
        Some(dms) => dms,
        None => {
            return ZSTD_compressBlock_doubleFast_noDict_generic(ms, seqStore, rep, src, 0, mls);
        }
    };
    let dict = &dms.dictContent;
    let dictStartIndex = dms.window.dictLimit;
    let dictEndIndex = dms.window.nextSrc;
    let dictBaseOff = dms.window.base_offset;
    let dictStart = dictStartIndex.saturating_sub(dictBaseOff) as usize;
    let dictSize = dictEndIndex.saturating_sub(dictBaseOff);
    let dictIndexDelta = prefixLowestIndex.saturating_sub(dictSize);
    let dictHBitsL = dms.cParams.hashLog + ZSTD_SHORT_CACHE_TAG_BITS;
    let dictHBitsS = dms.cParams.chainLog + ZSTD_SHORT_CACHE_TAG_BITS;
    let dictAndPrefixLength =
        (src.len() as u32).wrapping_add(dictEndIndex.saturating_sub(dictStartIndex));
    let maxDistance = if cParams.windowLog >= 31 {
        u32::MAX
    } else {
        1u32 << cParams.windowLog
    };

    if ms.window.dictLimit.saturating_add(maxDistance) < endIndex {
        return ZSTD_compressBlock_doubleFast_noDict_generic(ms, seqStore, rep, src, 0, mls);
    }
    if ZSTD_window_hasExtDict(&ms.window) {
        return ZSTD_compressBlock_doubleFast_extDict_generic(ms, seqStore, rep, src);
    }
    if src.len() < HASH_READ_SIZE {
        return src.len();
    }
    if dictAndPrefixLength == 0 {
        return ZSTD_compressBlock_doubleFast_noDict_generic(ms, seqStore, rep, src, 0, mls);
    }
    if rep[0] > dictAndPrefixLength {
        rep[0] = dictAndPrefixLength;
    }
    if rep[1] > dictAndPrefixLength {
        rep[1] = dictAndPrefixLength;
    }
    debug_assert!(prefixLowestIndex <= endIndex);
    if ms.prefetchCDictTables {
        crate::common::zstd_internal::ZSTD_prefetchArea(&dms.hashTable);
        crate::common::zstd_internal::ZSTD_prefetchArea(&dms.chainTable);
    }
    let hBitsL = cParams.hashLog;
    let hBitsS = cParams.chainLog;
    let chainSize = 1usize << hBitsS;
    if ms.chainTable.len() < chainSize {
        ms.chainTable.resize(chainSize, 0);
    }
    let iend = src.len();
    let ilimit = iend.saturating_sub(HASH_READ_SIZE);
    let mut offset_1 = rep[0];
    let mut offset_2 = rep[1];
    let mut ip = usize::from(dictAndPrefixLength == 0);
    let mut anchor = 0usize;

    while ip < ilimit {
        let h2 = ZSTD_hashPtr(&src[ip..], hBitsL, 8);
        let h = ZSTD_hashPtr(&src[ip..], hBitsS, mls);
        let dictHashAndTagL = ZSTD_hashPtr(&src[ip..], dictHBitsL, 8);
        let dictHashAndTagS = ZSTD_hashPtr(&src[ip..], dictHBitsS, mls);
        let dictMatchIndexAndTagL = dms.hashTable[dictHashAndTagL >> ZSTD_SHORT_CACHE_TAG_BITS];
        let dictMatchIndexAndTagS = dms.chainTable[dictHashAndTagS >> ZSTD_SHORT_CACHE_TAG_BITS];
        let dictTagsMatchL =
            ZSTD_comparePackedTags(dictMatchIndexAndTagL as usize, dictHashAndTagL);
        let dictTagsMatchS =
            ZSTD_comparePackedTags(dictMatchIndexAndTagS as usize, dictHashAndTagS);
        let curr = base_off.wrapping_add(ip as u32);
        let matchIndexL = ms.hashTable[h2];
        let mut matchIndexS = ms.chainTable[h];
        let repIndex = curr.wrapping_add(1).wrapping_sub(offset_1);

        ms.hashTable[h2] = curr;
        ms.chainTable[h] = curr;

        let repInDict = repIndex < prefixLowestIndex;
        let repMatch = if repInDict {
            repIndex
                .saturating_sub(dictIndexDelta)
                .saturating_sub(dictBaseOff) as usize
        } else {
            repIndex.saturating_sub(base_off) as usize
        };
        if ZSTD_index_overlap_check(prefixLowestIndex, repIndex)
            && if repInDict {
                repMatch + 4 <= dict.len()
                    && MEM_read32(&dict[repMatch..]) == MEM_read32(&src[ip + 1..])
            } else {
                repMatch + 4 <= src.len()
                    && MEM_read32(&src[repMatch..]) == MEM_read32(&src[ip + 1..])
            }
        {
            let repMatchEnd = if repInDict { dict.len() } else { iend };
            let mLength = ZSTD_count_2segments(
                src,
                ip + 1 + 4,
                iend,
                prefixLowest,
                if repInDict { dict } else { src },
                repMatch + 4,
                repMatchEnd,
            ) + 4;
            ip += 1;
            ZSTD_storeSeq(
                seqStore,
                ip - anchor,
                &src[anchor..ip],
                REPCODE_TO_OFFBASE(1),
                mLength,
            );
            ip += mLength;
            anchor = ip;
            if ip <= ilimit {
                let indexToInsert = curr.wrapping_add(2);
                let ins = indexToInsert.wrapping_sub(base_off) as usize;
                if ins + 8 <= src.len() {
                    ms.hashTable[ZSTD_hashPtr(&src[ins..], hBitsL, 8)] = indexToInsert;
                    ms.chainTable[ZSTD_hashPtr(&src[ins..], hBitsS, mls)] = indexToInsert;
                }
                if ip >= 2 {
                    ms.hashTable[ZSTD_hashPtr(&src[ip - 2..], hBitsL, 8)] =
                        base_off.wrapping_add((ip - 2) as u32);
                }
                if ip >= 1 {
                    ms.chainTable[ZSTD_hashPtr(&src[ip - 1..], hBitsS, mls)] =
                        base_off.wrapping_add((ip - 1) as u32);
                }
                while ip <= ilimit {
                    let current2 = base_off.wrapping_add(ip as u32);
                    let repIndex2 = current2.wrapping_sub(offset_2);
                    let repInDict2 = repIndex2 < prefixLowestIndex;
                    let repMatch2 = if repInDict2 {
                        repIndex2
                            .saturating_sub(dictIndexDelta)
                            .saturating_sub(dictBaseOff) as usize
                    } else {
                        repIndex2.saturating_sub(base_off) as usize
                    };
                    if ZSTD_index_overlap_check(prefixLowestIndex, repIndex2)
                        && if repInDict2 {
                            repMatch2 + 4 <= dict.len()
                                && MEM_read32(&dict[repMatch2..]) == MEM_read32(&src[ip..])
                        } else {
                            repMatch2 + 4 <= src.len()
                                && MEM_read32(&src[repMatch2..]) == MEM_read32(&src[ip..])
                        }
                    {
                        let repEnd2 = if repInDict2 { dict.len() } else { iend };
                        let repLength2 = ZSTD_count_2segments(
                            src,
                            ip + 4,
                            iend,
                            prefixLowest,
                            if repInDict2 { dict } else { src },
                            repMatch2 + 4,
                            repEnd2,
                        ) + 4;
                        std::mem::swap(&mut offset_2, &mut offset_1);
                        ZSTD_storeSeq(seqStore, 0, &src[ip..ip], REPCODE_TO_OFFBASE(1), repLength2);
                        ms.chainTable[ZSTD_hashPtr(&src[ip..], hBitsS, mls)] = current2;
                        ms.hashTable[ZSTD_hashPtr(&src[ip..], hBitsL, 8)] = current2;
                        ip += repLength2;
                        anchor = ip;
                        continue;
                    }
                    break;
                }
            }
            continue;
        }

        if matchIndexL >= prefixLowestIndex {
            let matchLong = matchIndexL.saturating_sub(base_off) as usize;
            if matchLong + 8 <= src.len() && MEM_read64(&src[matchLong..]) == MEM_read64(&src[ip..])
            {
                let mut mLength = ZSTD_count(src, ip + 8, matchLong + 8, iend) + 8;
                let mut catch_ip = ip;
                let mut catch_match = matchLong;
                while catch_ip > anchor
                    && catch_match > prefixLowest
                    && src[catch_ip - 1] == src[catch_match - 1]
                {
                    catch_ip -= 1;
                    catch_match -= 1;
                    mLength += 1;
                }
                let offset = base_off
                    .wrapping_add(ip as u32)
                    .wrapping_sub(base_off.wrapping_add(matchLong as u32));
                ip = catch_ip;
                offset_2 = offset_1;
                offset_1 = offset;
                ZSTD_storeSeq(
                    seqStore,
                    ip - anchor,
                    &src[anchor..ip],
                    OFFSET_TO_OFFBASE(offset),
                    mLength,
                );
                ip += mLength;
                anchor = ip;
                continue;
            }
        } else if dictTagsMatchL {
            let dictMatchIndexL = dictMatchIndexAndTagL >> ZSTD_SHORT_CACHE_TAG_BITS;
            let mut dictMatchL = dictMatchIndexL.saturating_sub(dictBaseOff) as usize;
            if dictMatchL < dict.len()
                && dictMatchL > dictStart
                && dictMatchL + 8 <= dict.len()
                && MEM_read64(&dict[dictMatchL..]) == MEM_read64(&src[ip..])
            {
                let mut mLength = ZSTD_count_2segments(
                    src,
                    ip + 8,
                    iend,
                    prefixLowest,
                    dict,
                    dictMatchL + 8,
                    dict.len(),
                ) + 8;
                let offset = curr
                    .wrapping_sub(dictMatchIndexL)
                    .wrapping_sub(dictIndexDelta);
                while ip > anchor && dictMatchL > dictStart && src[ip - 1] == dict[dictMatchL - 1] {
                    ip -= 1;
                    dictMatchL -= 1;
                    mLength += 1;
                }
                offset_2 = offset_1;
                offset_1 = offset;
                ZSTD_storeSeq(
                    seqStore,
                    ip - anchor,
                    &src[anchor..ip],
                    OFFSET_TO_OFFBASE(offset),
                    mLength,
                );
                ip += mLength;
                anchor = ip;
                continue;
            }
        }

        let short_match = if matchIndexS > prefixLowestIndex {
            let match_pos = matchIndexS.saturating_sub(base_off) as usize;
            match_pos + 4 <= src.len() && MEM_read32(&src[match_pos..]) == MEM_read32(&src[ip..])
        } else if dictTagsMatchS {
            let dictMatchIndexS = dictMatchIndexAndTagS >> ZSTD_SHORT_CACHE_TAG_BITS;
            let match_pos = dictMatchIndexS.saturating_sub(dictBaseOff) as usize;
            matchIndexS = dictMatchIndexS + dictIndexDelta;
            match_pos > dictStart
                && match_pos + 4 <= dict.len()
                && MEM_read32(&dict[match_pos..]) == MEM_read32(&src[ip..])
        } else {
            false
        };

        if short_match {
            let hl3 = ZSTD_hashPtr(&src[ip + 1..], hBitsL, 8);
            let dictHashAndTagL3 = ZSTD_hashPtr(&src[ip + 1..], dictHBitsL, 8);
            let matchIndexL3 = ms.hashTable[hl3];
            let dictMatchIndexAndTagL3 =
                dms.hashTable[dictHashAndTagL3 >> ZSTD_SHORT_CACHE_TAG_BITS];
            let dictTagsMatchL3 =
                ZSTD_comparePackedTags(dictMatchIndexAndTagL3 as usize, dictHashAndTagL3);
            ms.hashTable[hl3] = curr.wrapping_add(1);

            if matchIndexL3 >= prefixLowestIndex {
                let mut matchL3 = matchIndexL3.saturating_sub(base_off) as usize;
                if matchL3 + 8 <= src.len()
                    && MEM_read64(&src[matchL3..]) == MEM_read64(&src[ip + 1..])
                {
                    let mut mLength = ZSTD_count(src, ip + 9, matchL3 + 8, iend) + 8;
                    ip += 1;
                    let offset = base_off
                        .wrapping_add(ip as u32)
                        .wrapping_sub(base_off.wrapping_add(matchL3 as u32));
                    while ip > anchor && matchL3 > prefixLowest && src[ip - 1] == src[matchL3 - 1] {
                        ip -= 1;
                        matchL3 -= 1;
                        mLength += 1;
                    }
                    offset_2 = offset_1;
                    offset_1 = offset;
                    ZSTD_storeSeq(
                        seqStore,
                        ip - anchor,
                        &src[anchor..ip],
                        OFFSET_TO_OFFBASE(offset),
                        mLength,
                    );
                    ip += mLength;
                    anchor = ip;
                    continue;
                }
            } else if dictTagsMatchL3 {
                let dictMatchIndexL3 = dictMatchIndexAndTagL3 >> ZSTD_SHORT_CACHE_TAG_BITS;
                let mut dictMatchL3 = dictMatchIndexL3.saturating_sub(dictBaseOff) as usize;
                if dictMatchL3 < dict.len()
                    && dictMatchL3 > dictStart
                    && dictMatchL3 + 8 <= dict.len()
                    && MEM_read64(&dict[dictMatchL3..]) == MEM_read64(&src[ip + 1..])
                {
                    let mut mLength = ZSTD_count_2segments(
                        src,
                        ip + 1 + 8,
                        iend,
                        prefixLowest,
                        dict,
                        dictMatchL3 + 8,
                        dict.len(),
                    ) + 8;
                    ip += 1;
                    let offset = curr
                        .wrapping_add(1)
                        .wrapping_sub(dictMatchIndexL3)
                        .wrapping_sub(dictIndexDelta);
                    while ip > anchor
                        && dictMatchL3 > dictStart
                        && src[ip - 1] == dict[dictMatchL3 - 1]
                    {
                        ip -= 1;
                        dictMatchL3 -= 1;
                        mLength += 1;
                    }
                    offset_2 = offset_1;
                    offset_1 = offset;
                    ZSTD_storeSeq(
                        seqStore,
                        ip - anchor,
                        &src[anchor..ip],
                        OFFSET_TO_OFFBASE(offset),
                        mLength,
                    );
                    ip += mLength;
                    anchor = ip;
                    continue;
                }
            }

            let mut mLength;
            let offset;
            if matchIndexS < prefixLowestIndex {
                let mut match_pos = matchIndexS
                    .saturating_sub(dictIndexDelta)
                    .saturating_sub(dictBaseOff) as usize;
                mLength = ZSTD_count_2segments(
                    src,
                    ip + 4,
                    iend,
                    prefixLowest,
                    dict,
                    match_pos + 4,
                    dict.len(),
                ) + 4;
                offset = curr.wrapping_sub(matchIndexS);
                while ip > anchor && match_pos > dictStart && src[ip - 1] == dict[match_pos - 1] {
                    ip -= 1;
                    match_pos -= 1;
                    mLength += 1;
                }
            } else {
                let mut match_pos = matchIndexS.saturating_sub(base_off) as usize;
                mLength = ZSTD_count(src, ip + 4, match_pos + 4, iend) + 4;
                offset = curr.wrapping_sub(matchIndexS);
                while ip > anchor && match_pos > prefixLowest && src[ip - 1] == src[match_pos - 1] {
                    ip -= 1;
                    match_pos -= 1;
                    mLength += 1;
                }
            }

            offset_2 = offset_1;
            offset_1 = offset;
            ZSTD_storeSeq(
                seqStore,
                ip - anchor,
                &src[anchor..ip],
                OFFSET_TO_OFFBASE(offset),
                mLength,
            );
            ip += mLength;
            anchor = ip;
            if ip <= ilimit {
                let indexToInsert = curr.wrapping_add(2);
                let ins = indexToInsert.saturating_sub(base_off) as usize;
                if ins + 8 <= src.len() {
                    ms.hashTable[ZSTD_hashPtr(&src[ins..], hBitsL, 8)] = indexToInsert;
                    ms.chainTable[ZSTD_hashPtr(&src[ins..], hBitsS, mls)] = indexToInsert;
                }
                if ip >= 2 {
                    ms.hashTable[ZSTD_hashPtr(&src[ip - 2..], hBitsL, 8)] =
                        base_off.wrapping_add((ip - 2) as u32);
                }
                if ip >= 1 {
                    ms.chainTable[ZSTD_hashPtr(&src[ip - 1..], hBitsS, mls)] =
                        base_off.wrapping_add((ip - 1) as u32);
                }
                while ip <= ilimit {
                    let current2 = base_off.wrapping_add(ip as u32);
                    let repIndex2 = current2.wrapping_sub(offset_2);
                    let repInDict2 = repIndex2 < prefixLowestIndex;
                    let repMatch2 = if repInDict2 {
                        repIndex2
                            .saturating_sub(dictIndexDelta)
                            .saturating_sub(dictBaseOff) as usize
                    } else {
                        repIndex2.saturating_sub(base_off) as usize
                    };
                    if ZSTD_index_overlap_check(prefixLowestIndex, repIndex2)
                        && if repInDict2 {
                            repMatch2 + 4 <= dict.len()
                                && MEM_read32(&dict[repMatch2..]) == MEM_read32(&src[ip..])
                        } else {
                            repMatch2 + 4 <= src.len()
                                && MEM_read32(&src[repMatch2..]) == MEM_read32(&src[ip..])
                        }
                    {
                        let repEnd2 = if repInDict2 { dict.len() } else { iend };
                        let repLength2 = ZSTD_count_2segments(
                            src,
                            ip + 4,
                            iend,
                            prefixLowest,
                            if repInDict2 { dict } else { src },
                            repMatch2 + 4,
                            repEnd2,
                        ) + 4;
                        std::mem::swap(&mut offset_2, &mut offset_1);
                        ZSTD_storeSeq(seqStore, 0, &src[ip..ip], REPCODE_TO_OFFBASE(1), repLength2);
                        ms.chainTable[ZSTD_hashPtr(&src[ip..], hBitsS, mls)] = current2;
                        ms.hashTable[ZSTD_hashPtr(&src[ip..], hBitsL, 8)] = current2;
                        ip += repLength2;
                        anchor = ip;
                        continue;
                    }
                    break;
                }
                continue;
            }
        }

        ip += ((ip - anchor) >> kSearchStrength) + 1;
        continue;
    }

    rep[0] = offset_1;
    rep[1] = offset_2;
    iend - anchor
}

pub fn ZSTD_compressBlock_doubleFast_extDict(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_doubleFast_extDict_generic(ms, seqStore, rep, src)
}

/// Port of `ZSTD_compressBlock_doubleFast_extDict_generic`.
pub fn ZSTD_compressBlock_doubleFast_extDict_generic(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    let cParams = ms.cParams;
    let mls = cParams.minMatch;
    if !ZSTD_window_hasExtDict(&ms.window) {
        return ZSTD_compressBlock_doubleFast_noDict_generic(ms, seqStore, rep, src, 0, mls);
    }
    let hBitsL = cParams.hashLog;
    let hBitsS = cParams.chainLog;
    let prefixBaseIndex = ms.window.dictLimit.max(ms.loadedDictEnd);
    let base_off = prefixBaseIndex;
    let dict_base_off = prefixBaseIndex.wrapping_sub(ms.dictContent.len() as u32);
    let iend = src.len();
    let ilimit = iend.saturating_sub(HASH_READ_SIZE);
    let endIndex = prefixBaseIndex.wrapping_add(src.len() as u32);
    let dictStartIndex = ZSTD_getLowestMatchIndex(ms, endIndex, cParams.windowLog).max(dict_base_off);
    let prefixStartIndex = prefixBaseIndex.max(dictStartIndex);
    let prefixStart = prefixStartIndex.wrapping_sub(base_off) as usize;
    let dict = &ms.dictContent;
    let dictStart = dictStartIndex.wrapping_sub(dict_base_off) as usize;
    let dictEnd = prefixStartIndex.wrapping_sub(dict_base_off) as usize;
    let mut offset_1 = rep[0];
    let mut offset_2 = rep[1];
    let mut ip = 0usize;
    let mut anchor = 0usize;

    if prefixStartIndex == dictStartIndex {
        return ZSTD_compressBlock_doubleFast_noDict_generic(ms, seqStore, rep, src, 0, mls);
    }
    if src.len() < HASH_READ_SIZE {
        return src.len();
    }
    debug_assert!(endIndex >= prefixStartIndex);
    debug_assert!(dictStart <= dictEnd);
    debug_assert!(dictEnd <= dict.len());

    while ip < ilimit {
        let hSmall = ZSTD_hashPtr(&src[ip..], hBitsS, mls);
        let matchIndex = ms.chainTable[hSmall];
        let matchInDict = matchIndex < prefixStartIndex;
        let mut match_pos = if matchInDict {
            matchIndex.wrapping_sub(dict_base_off) as usize
        } else {
            matchIndex.wrapping_sub(base_off) as usize
        };

        let hLong = ZSTD_hashPtr(&src[ip..], hBitsL, 8);
        let matchLongIndex = ms.hashTable[hLong];
        let matchLongInDict = matchLongIndex < prefixStartIndex;
        let mut matchLong = if matchLongInDict {
            matchLongIndex.wrapping_sub(dict_base_off) as usize
        } else {
            matchLongIndex.wrapping_sub(base_off) as usize
        };

        let curr = base_off.wrapping_add(ip as u32);
        let repIndex = curr.wrapping_add(1).wrapping_sub(offset_1);
        let repInDict = repIndex < prefixStartIndex;
        let repMatch = if repInDict {
            repIndex.wrapping_sub(dict_base_off) as usize
        } else {
            repIndex.wrapping_sub(base_off) as usize
        };
        ms.chainTable[hSmall] = curr;
        ms.hashTable[hLong] = curr;

        let mLength: usize;
        if offset_1 <= curr.wrapping_add(1).wrapping_sub(dictStartIndex)
            && ZSTD_index_overlap_check(prefixStartIndex, repIndex)
            && if repInDict {
                repMatch + 4 <= dictEnd
                    && MEM_read32(&dict[repMatch..]) == MEM_read32(&src[ip + 1..])
            } else {
                repMatch + 4 <= src.len()
                    && MEM_read32(&src[repMatch..]) == MEM_read32(&src[ip + 1..])
            }
        {
            let repMatchEnd = if repInDict { dictEnd } else { iend };
            mLength = ZSTD_count_2segments(
                src,
                ip + 1 + 4,
                iend,
                prefixStart,
                if repInDict { dict } else { src },
                repMatch + 4,
                repMatchEnd,
            ) + 4;
            ip += 1;
            ZSTD_storeSeq(
                seqStore,
                ip - anchor,
                &src[anchor..ip],
                REPCODE_TO_OFFBASE(1),
                mLength,
            );
        } else if matchLongIndex > dictStartIndex
            && matchLong + 8 <= if matchLongInDict { dictEnd } else { src.len() }
            && if matchLongInDict {
                MEM_read64(&dict[matchLong..]) == MEM_read64(&src[ip..])
            } else {
                MEM_read64(&src[matchLong..]) == MEM_read64(&src[ip..])
            }
        {
            let matchEnd = if matchLongInDict { dictEnd } else { iend };
            let lowMatchPtr = if matchLongInDict {
                dictStart
            } else {
                prefixStart
            };
            let offset = curr.wrapping_sub(matchLongIndex);
            let mut len = ZSTD_count_2segments(
                src,
                ip + 8,
                iend,
                prefixStart,
                if matchLongInDict { dict } else { src },
                matchLong + 8,
                matchEnd,
            ) + 8;
            while ip > anchor
                && matchLong > lowMatchPtr
                && src[ip - 1]
                    == if matchLongInDict {
                        dict[matchLong - 1]
                    } else {
                        src[matchLong - 1]
                    }
            {
                ip -= 1;
                matchLong -= 1;
                len += 1;
            }
            offset_2 = offset_1;
            offset_1 = offset;
            mLength = len;
            ZSTD_storeSeq(
                seqStore,
                ip - anchor,
                &src[anchor..ip],
                OFFSET_TO_OFFBASE(offset),
                mLength,
            );
        } else if matchIndex > dictStartIndex
            && match_pos + 4 <= if matchInDict { dictEnd } else { src.len() }
            && if matchInDict {
                MEM_read32(&dict[match_pos..]) == MEM_read32(&src[ip..])
            } else {
                MEM_read32(&src[match_pos..]) == MEM_read32(&src[ip..])
            }
        {
            let h3 = ZSTD_hashPtr(&src[ip + 1..], hBitsL, 8);
            let matchIndex3 = ms.hashTable[h3];
            let match3InDict = matchIndex3 < prefixStartIndex;
            let mut match3 = if match3InDict {
                matchIndex3.wrapping_sub(dict_base_off) as usize
            } else {
                matchIndex3.wrapping_sub(base_off) as usize
            };
            let offset;
            ms.hashTable[h3] = curr.wrapping_add(1);
            if matchIndex3 > dictStartIndex
                && match3 + 8 <= if match3InDict { dictEnd } else { src.len() }
                && if match3InDict {
                    MEM_read64(&dict[match3..]) == MEM_read64(&src[ip + 1..])
                } else {
                    MEM_read64(&src[match3..]) == MEM_read64(&src[ip + 1..])
                }
            {
                let matchEnd = if match3InDict { dictEnd } else { iend };
                let lowMatchPtr = if match3InDict { dictStart } else { prefixStart };
                let mut len = ZSTD_count_2segments(
                    src,
                    ip + 9,
                    iend,
                    prefixStart,
                    if match3InDict { dict } else { src },
                    match3 + 8,
                    matchEnd,
                ) + 8;
                ip += 1;
                offset = curr.wrapping_add(1).wrapping_sub(matchIndex3);
                while ip > anchor
                    && match3 > lowMatchPtr
                    && src[ip - 1]
                        == if match3InDict {
                            dict[match3 - 1]
                        } else {
                            src[match3 - 1]
                        }
                {
                    ip -= 1;
                    match3 -= 1;
                    len += 1;
                }
                mLength = len;
            } else {
                let matchEnd = if matchInDict { dictEnd } else { iend };
                let lowMatchPtr = if matchInDict { dictStart } else { prefixStart };
                let mut len = ZSTD_count_2segments(
                    src,
                    ip + 4,
                    iend,
                    prefixStart,
                    if matchInDict { dict } else { src },
                    match_pos + 4,
                    matchEnd,
                ) + 4;
                offset = curr.wrapping_sub(matchIndex);
                while ip > anchor
                    && match_pos > lowMatchPtr
                    && src[ip - 1]
                        == if matchInDict {
                            dict[match_pos - 1]
                        } else {
                            src[match_pos - 1]
                        }
                {
                    ip -= 1;
                    match_pos -= 1;
                    len += 1;
                }
                mLength = len;
            }
            offset_2 = offset_1;
            offset_1 = offset;
            ZSTD_storeSeq(
                seqStore,
                ip - anchor,
                &src[anchor..ip],
                OFFSET_TO_OFFBASE(offset),
                mLength,
            );
        } else {
            ip += ((ip - anchor) >> kSearchStrength) + 1;
            continue;
        }

        ip += mLength;
        anchor = ip;
        if ip <= ilimit {
            let indexToInsert = curr.wrapping_add(2);
            ms.hashTable[ZSTD_hashPtr(
                &src[indexToInsert.wrapping_sub(base_off) as usize..],
                hBitsL,
                8,
            )] = indexToInsert;
            ms.hashTable[ZSTD_hashPtr(&src[ip - 2..], hBitsL, 8)] =
                base_off.wrapping_add((ip - 2) as u32);
            ms.chainTable[ZSTD_hashPtr(
                &src[indexToInsert.wrapping_sub(base_off) as usize..],
                hBitsS,
                mls,
            )] = indexToInsert;
            ms.chainTable[ZSTD_hashPtr(&src[ip - 1..], hBitsS, mls)] =
                base_off.wrapping_add((ip - 1) as u32);

            while ip <= ilimit {
                let current2 = base_off.wrapping_add(ip as u32);
                let repIndex2 = current2.wrapping_sub(offset_2);
                let repInDict2 = repIndex2 < prefixStartIndex;
                let repMatch2 = if repInDict2 {
                    repIndex2.wrapping_sub(dict_base_off) as usize
                } else {
                    repIndex2.wrapping_sub(base_off) as usize
                };
                if offset_2 <= current2.wrapping_sub(dictStartIndex)
                    && ZSTD_index_overlap_check(prefixStartIndex, repIndex2)
                    && if repInDict2 {
                        repMatch2 + 4 <= dictEnd
                            && MEM_read32(&dict[repMatch2..]) == MEM_read32(&src[ip..])
                    } else {
                        repMatch2 + 4 <= src.len()
                            && MEM_read32(&src[repMatch2..]) == MEM_read32(&src[ip..])
                    }
                {
                    let repEnd2 = if repInDict2 { dictEnd } else { iend };
                    let repLength2 = ZSTD_count_2segments(
                        src,
                        ip + 4,
                        iend,
                        prefixStart,
                        if repInDict2 { dict } else { src },
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
                    ms.chainTable[ZSTD_hashPtr(&src[ip..], hBitsS, mls)] = current2;
                    ms.hashTable[ZSTD_hashPtr(&src[ip..], hBitsL, 8)] = current2;
                    ip += repLength2;
                    anchor = ip;
                    continue;
                }
                break;
            }
        }
    }

    rep[0] = offset_1;
    rep[1] = offset_2;
    iend - anchor
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
    use crate::compress::seq_store::{OFFBASE_IS_OFFSET, OFFBASE_TO_OFFSET};

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
    fn fill_double_hash_table_cdict_path_populates_tables() {
        let mut ms = ZSTD_MatchState_t::new(ZSTD_compressionParameters {
            hashLog: 10,
            chainLog: 10,
            minMatch: 4,
            ..Default::default()
        });
        let src: Vec<u8> = (0..128u8).cycle().take(256).collect();
        ZSTD_fillDoubleHashTable(
            &mut ms,
            &src,
            ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_full,
            ZSTD_tableFillPurpose_e::ZSTD_tfp_forCDict,
        );
        assert!(ms.hashTable.iter().any(|&c| c != 0));
        assert!(ms.chainTable.iter().any(|&c| c != 0));
    }

    #[test]
    fn fill_double_hash_table_cdict_path_packs_short_cache_tags() {
        let mut ms = ZSTD_MatchState_t::new(ZSTD_compressionParameters {
            hashLog: 10,
            chainLog: 10,
            minMatch: 4,
            ..Default::default()
        });
        let src: Vec<u8> = (0..10u8).collect();
        ZSTD_fillDoubleHashTableForCDict(&mut ms, &src, ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_fast);

        let large_hash_and_tag =
            ZSTD_hashPtr(&src, ms.cParams.hashLog + ZSTD_SHORT_CACHE_TAG_BITS, 8);
        let small_hash_and_tag = ZSTD_hashPtr(
            &src,
            ms.cParams.chainLog + ZSTD_SHORT_CACHE_TAG_BITS,
            ms.cParams.minMatch,
        );
        let large_slot = large_hash_and_tag >> ZSTD_SHORT_CACHE_TAG_BITS;
        let small_slot = small_hash_and_tag >> ZSTD_SHORT_CACHE_TAG_BITS;
        let expected_large = (ms.window.base_offset << ZSTD_SHORT_CACHE_TAG_BITS)
            | (large_hash_and_tag as u32 & ((1u32 << ZSTD_SHORT_CACHE_TAG_BITS) - 1));
        let expected_small = (ms.window.base_offset << ZSTD_SHORT_CACHE_TAG_BITS)
            | (small_hash_and_tag as u32 & ((1u32 << ZSTD_SHORT_CACHE_TAG_BITS) - 1));

        assert_eq!(ms.hashTable[large_slot], expected_large);
        assert_eq!(ms.chainTable[small_slot], expected_small);
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
        assert!(
            !seq.sequences.is_empty(),
            "match finder emitted 0 sequences"
        );
        let total_lits = seq.literals.len() + last_lits;
        assert!(
            total_lits < src.len(),
            "no savings: lits={total_lits}, src={}",
            src.len()
        );
    }

    #[test]
    fn dict_and_ext_wrappers_route_through_live_double_fast_entries() {
        fn build_ms() -> ZSTD_MatchState_t {
            ZSTD_MatchState_t::new(ZSTD_compressionParameters {
                windowLog: 17,
                hashLog: 12,
                chainLog: 12,
                minMatch: 4,
                strategy: 2,
                ..Default::default()
            })
        }

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
            ZSTD_compressBlock_doubleFast_dictMatchState,
            ZSTD_compressBlock_doubleFast_extDict,
        ];

        for (idx, f) in variants.into_iter().enumerate() {
            let mut ms = build_ms();
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
    fn double_fast_dict_match_state_uses_attached_dictionary_bytes() {
        let dict: Vec<u8> = b"dictionary phrase alpha beta gamma. "
            .iter()
            .cycle()
            .take(512)
            .copied()
            .collect();
        let src: Vec<u8> = b"dictionary phrase alpha beta gamma. payload payload payload. "
            .iter()
            .cycle()
            .take(1024)
            .copied()
            .collect();

        let cp = ZSTD_compressionParameters {
            windowLog: 17,
            hashLog: 12,
            chainLog: 12,
            minMatch: 4,
            strategy: 2,
            ..Default::default()
        };

        let mut dms = ZSTD_MatchState_t::new(cp);
        dms.dictContent = dict.clone();
        dms.window.nextSrc = dms.window.base_offset.wrapping_add(dict.len() as u32);
        ZSTD_fillDoubleHashTableForCDict(
            &mut dms,
            &dict,
            ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_full,
        );

        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.dictMatchState = Some(Box::new(dms));
        ms.window.base_offset = crate::compress::match_state::ZSTD_WINDOW_START_INDEX
            .wrapping_add(dict.len() as u32);
        ms.window.dictLimit = ms.window.base_offset;
        ms.window.lowLimit = ms.window.base_offset;
        ms.nextToUpdate = ms.window.base_offset;
        ms.loadedDictEnd = dict.len() as u32;

        let mut seq = SeqStore_t::with_capacity(1024, 131072);
        let mut rep = [1u32, 4, 8];
        let last_lits =
            ZSTD_compressBlock_doubleFast_dictMatchState(&mut ms, &mut seq, &mut rep, &src);
        assert!(
            last_lits < src.len(),
            "dictMatchState path emitted only literals"
        );
        assert!(
            !seq.sequences.is_empty(),
            "dictMatchState path failed to emit any sequences"
        );
    }

    #[test]
    fn double_fast_ext_dict_uses_external_dictionary_bytes() {
        let dict = b"abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ".to_vec();
        let src = b"mnopqrstuvwxyz012345".to_vec();

        let cp = ZSTD_compressionParameters {
            windowLog: 17,
            hashLog: 12,
            chainLog: 12,
            minMatch: 4,
            strategy: 2,
            ..Default::default()
        };

        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.dictContent = dict.clone();
        ms.window.base_offset = crate::compress::match_state::ZSTD_WINDOW_START_INDEX;
        ms.nextToUpdate = crate::compress::match_state::ZSTD_WINDOW_START_INDEX;
        ZSTD_fillDoubleHashTableForCCtx(&mut ms, &dict, ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_fast);

        ms.window.dictBase_offset = crate::compress::match_state::ZSTD_WINDOW_START_INDEX;
        ms.window.lowLimit = crate::compress::match_state::ZSTD_WINDOW_START_INDEX;
        ms.window.dictLimit = crate::compress::match_state::ZSTD_WINDOW_START_INDEX
            .wrapping_add(dict.len() as u32);
        ms.window.base_offset = ms.window.dictLimit;
        ms.window.nextSrc = ms.window.base_offset.wrapping_add(src.len() as u32);
        ms.loadedDictEnd = dict.len() as u32;

        let mut seq = SeqStore_t::with_capacity(128, 4096);
        let mut rep = [1u32, 4, 8];
        let last_lits = ZSTD_compressBlock_doubleFast_extDict(&mut ms, &mut seq, &mut rep, &src);

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
