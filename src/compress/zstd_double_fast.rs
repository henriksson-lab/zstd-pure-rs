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
use crate::compress::zstd_hashes::{
    ZSTD_count, ZSTD_count_2segments, ZSTD_hash4, ZSTD_hash5, ZSTD_hash6, ZSTD_hash7, ZSTD_hash8,
    ZSTD_hashPtr,
};

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

    debug_assert!(ms.nextToUpdate >= ms.window.base_offset);
    let mut ip = ms.nextToUpdate.wrapping_sub(ms.window.base_offset) as usize;
    while ip + fastHashFillStep <= iend + 1 {
        let curr = ms.window.base_offset.wrapping_add(ip as u32);
        for i in 0..fastHashFillStep {
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
    debug_assert!(ms.nextToUpdate >= ms.window.base_offset);
    let mut ip = ms.nextToUpdate.wrapping_sub(ms.window.base_offset) as usize;
    while ip + fastHashFillStep <= iend + 1 {
        let curr = ms.window.base_offset.wrapping_add(ip as u32);
        for i in 0..fastHashFillStep {
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
                ZSTD_writeTaggedIndex(&mut ms.hashTable, lgHashAndTag, curr.wrapping_add(i as u32));
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
    // Monomorphize on `mls` so `ZSTD_hashPtr`'s match dispatch folds
    // away in each specialization. Mirrors upstream's
    // `ZSTD_GEN_DOUBLEFAST_FN(noDict, mml)` macro expansion.
    match mls {
        5 => ZSTD_compressBlock_doubleFast_noDict_generic_mls::<5>(ms, seqStore, rep, src, istart),
        6 => ZSTD_compressBlock_doubleFast_noDict_generic_mls::<6>(ms, seqStore, rep, src, istart),
        7 => ZSTD_compressBlock_doubleFast_noDict_generic_mls::<7>(ms, seqStore, rep, src, istart),
        _ => ZSTD_compressBlock_doubleFast_noDict_generic_mls::<4>(ms, seqStore, rep, src, istart),
    }
}

/// MLS-monomorphized core of `ZSTD_compressBlock_doubleFast_noDict_generic`.
/// Mirrors upstream's `ZSTD_GEN_DOUBLEFAST_FN(noDict, mml)` template
/// expansions; the const generic lets LLVM fold the short-table hash
/// width for the same generated specialization set.
#[inline(never)]
fn ZSTD_compressBlock_doubleFast_noDict_generic_mls<const MLS: u32>(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
    istart: usize,
) -> usize {
    let hBitsL = ms.cParams.hashLog;
    let hBitsS = ms.cParams.chainLog;
    let windowLog = ms.cParams.windowLog;

    let chainSize = 1usize << hBitsS;
    if ms.chainTable.len() < chainSize {
        ms.chainTable.resize(chainSize, 0);
    }
    let hashTable = ms.hashTable.as_mut_ptr();
    let chainTable = ms.chainTable.as_mut_ptr();

    let base_off = ms.window.base_offset;
    let srcSize = src.len();
    let endIndex = base_off.wrapping_add(srcSize as u32);
    let prefixStartIndex = ZSTD_getLowestPrefixIndex(ms, endIndex, windowLog);
    debug_assert!(prefixStartIndex >= base_off);
    let prefixStart = prefixStartIndex.wrapping_sub(base_off) as usize;
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

    let kStepIncr = 1usize << kSearchStrength;
    let dummy = [0x12u8, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0xe2, 0xb4];

    'outer: loop {
        let mut step = 1usize;
        let mut nextStep = ip + kStepIncr;
        let mut ip1 = ip + step;

        if ip1 > ilimit {
            break 'outer;
        }

        // SAFETY: outer-loop check `ip1 > ilimit` (= iend - 8) gives
        // `ip < ilimit, so ip + 8 ≤ iend = src.len()`. Both hashes
        // need at most 8 bytes.
        let mut hl0 = unsafe {
            let p = src.as_ptr().wrapping_add(ip);
            ZSTD_hash8((p as *const u64).read_unaligned().to_le(), hBitsL, 0)
        };
        // SAFETY: hl0 < (1 << hBitsL) == ms.hashTable.len() by hash construction.
        let mut idxl0 = unsafe { *hashTable.add(hl0) };
        let mut matchl0 = idxl0.wrapping_sub(base_off) as usize;

        let (mLength, offset, curr) = 'search: loop {
            // SAFETY: same bound as above (`ip < ilimit`).
            let hs0 = unsafe {
                let p = src.as_ptr().wrapping_add(ip);
                match MLS {
                    5 => ZSTD_hash5((p as *const u64).read_unaligned().to_le(), hBitsS, 0),
                    6 => ZSTD_hash6((p as *const u64).read_unaligned().to_le(), hBitsS, 0),
                    7 => ZSTD_hash7((p as *const u64).read_unaligned().to_le(), hBitsS, 0),
                    8 => ZSTD_hash8((p as *const u64).read_unaligned().to_le(), hBitsS, 0),
                    _ => ZSTD_hash4((p as *const u32).read_unaligned().to_le(), hBitsS, 0) as usize,
                }
            };
            // SAFETY: hs0 < (1 << hBitsS) == ms.chainTable.len() (chainTable was resized
            // to at least chainSize=1<<hBitsS at function entry).
            let idxs0 = unsafe { *chainTable.add(hs0) };
            let curr = base_off.wrapping_add(ip as u32);
            let matchs0 = idxs0.wrapping_sub(base_off) as usize;

            unsafe {
                *hashTable.add(hl0) = curr;
                *chainTable.add(hs0) = curr;
            }

            if rep_offset1 > 0
                && unsafe {
                    (src.as_ptr()
                        .wrapping_add((ip + 1).wrapping_sub(rep_offset1 as usize))
                        as *const u32)
                        .read_unaligned()
                } == unsafe {
                    (src.as_ptr().wrapping_add(ip + 1) as *const u32).read_unaligned()
                }
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

            // SAFETY: `ip1 = ip + step ≤ ilimit = iend - 8`, so
            // `ip1 + 8 ≤ iend = src.len()`.
            let hl1 = unsafe {
                let p = src.as_ptr().wrapping_add(ip1);
                ZSTD_hash8((p as *const u64).read_unaligned().to_le(), hBitsL, 0)
            };

            // Long-match check via CMOV-friendly pointer select.
            // Mirrors upstream's `ZSTD_selectAddr(idxl0, prefixLow,
            // matchl0, dummy)` — the slice-indexing version emits a
            // bounds-check `cmp/jb` per read, defeating CMOV.
            let long_match_valid = idxl0 > prefixStartIndex;
            let real_long_ptr = src.as_ptr().wrapping_add(matchl0);
            let dummy_long_ptr = dummy.as_ptr();
            let m_long_ptr = if long_match_valid {
                real_long_ptr
            } else {
                dummy_long_ptr
            };
            // SAFETY: when `long_match_valid`, `matchl0 + 8 ≤ iend ≤
            // src.len()` so the 8-byte read is in-range; otherwise
            // we read from `dummy` (10 bytes, has 8 valid). The `ip`
            // read is bounded by `ilimit < iend - 8`, so safe.
            let m_long = unsafe { (m_long_ptr as *const u64).read_unaligned() };
            let ip_long = unsafe { (src.as_ptr().wrapping_add(ip) as *const u64).read_unaligned() };
            if long_match_valid && m_long == ip_long {
                debug_assert!(matchl0 + 8 <= iend);
                let mut mLength = ZSTD_count(src, ip + 8, matchl0 + 8, iend) + 8;
                let offset = (ip - matchl0) as u32;
                while ip > anchor && matchl0 > prefixStart && src[ip - 1] == src[matchl0 - 1] {
                    ip -= 1;
                    matchl0 -= 1;
                    mLength += 1;
                }
                break 'search (mLength, offset, curr);
            }

            // SAFETY: hl1 < (1 << hBitsL) == ms.hashTable.len() by hash construction.
            let idxl1 = unsafe { *hashTable.add(hl1) };
            let matchl1 = idxl1.wrapping_sub(base_off) as usize;

            // Short-match check via CMOV-friendly pointer select. Same
            // pattern as the long-match above — bypasses the slice
            // bounds check on `&src[matchs0..]` so both reads can issue
            // unconditionally.
            let short_match_valid = idxs0 > prefixStartIndex;
            let real_short_ptr = src.as_ptr().wrapping_add(matchs0);
            let dummy_short_ptr = dummy.as_ptr();
            let m_short_ptr = if short_match_valid {
                real_short_ptr
            } else {
                dummy_short_ptr
            };
            // SAFETY: when `short_match_valid`, `matchs0 + 4 ≤ iend ≤
            // src.len()` so the 4-byte read is in-range; otherwise
            // we read from `dummy` (10 bytes, has 4 valid).
            let m_short = unsafe { (m_short_ptr as *const u32).read_unaligned() };
            let ip_short =
                unsafe { (src.as_ptr().wrapping_add(ip) as *const u32).read_unaligned() };
            if short_match_valid && m_short == ip_short {
                debug_assert!(matchs0 + 4 <= iend);
                let mut mLength = ZSTD_count(src, ip + 4, matchs0 + 4, iend) + 4;
                let mut offset = (ip - matchs0) as u32;
                let mut match_pos = matchs0;

                if idxl1 > prefixStartIndex
                    && unsafe {
                        (src.as_ptr().wrapping_add(matchl1) as *const u64).read_unaligned()
                    } == unsafe {
                        (src.as_ptr().wrapping_add(ip1) as *const u64).read_unaligned()
                    }
                {
                    let l1len = ZSTD_count(src, ip1 + 8, matchl1 + 8, iend) + 8;
                    if l1len > mLength {
                        ip = ip1;
                        mLength = l1len;
                        offset = (ip - matchl1) as u32;
                        match_pos = matchl1;
                    }
                }

                while ip > anchor && match_pos > prefixStart && src[ip - 1] == src[match_pos - 1] {
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
                // SAFETY: hl1 < (1 << hBitsL) == ms.hashTable.len().
                unsafe {
                    *hashTable.add(hl1) = base_off.wrapping_add(ip1 as u32);
                }
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
            debug_assert!(indexToInsert >= base_off);
            let indexToInsertPos = indexToInsert.wrapping_sub(base_off) as usize;
            // SAFETY: `ip ≤ ilimit = iend - 8`, and `indexToInsertPos
            // = curr - base + 2 = ip_at_match + 2 ≤ ip ≤ ilimit`, so
            // both positions have ≥ 8 bytes available.
            let h = unsafe {
                let p = src.as_ptr().wrapping_add(indexToInsertPos);
                ZSTD_hash8((p as *const u64).read_unaligned().to_le(), hBitsL, 0)
            };
            // SAFETY: h < (1<<hBitsL) == ms.hashTable.len().
            unsafe {
                *hashTable.add(h) = indexToInsert;
            }
            let h = unsafe {
                let p = src.as_ptr().wrapping_add(indexToInsertPos);
                match MLS {
                    5 => ZSTD_hash5((p as *const u64).read_unaligned().to_le(), hBitsS, 0),
                    6 => ZSTD_hash6((p as *const u64).read_unaligned().to_le(), hBitsS, 0),
                    7 => ZSTD_hash7((p as *const u64).read_unaligned().to_le(), hBitsS, 0),
                    8 => ZSTD_hash8((p as *const u64).read_unaligned().to_le(), hBitsS, 0),
                    _ => ZSTD_hash4((p as *const u32).read_unaligned().to_le(), hBitsS, 0) as usize,
                }
            };
            // SAFETY: h < (1<<hBitsS) == ms.chainTable.len().
            unsafe {
                *chainTable.add(h) = indexToInsert;
            }
            let h = unsafe {
                let p = src.as_ptr().wrapping_add(ip - 2);
                ZSTD_hash8((p as *const u64).read_unaligned().to_le(), hBitsL, 0)
            };
            // SAFETY: `ip <= ilimit` implies `ip - 2 + 8 <= iend`.
            unsafe {
                *hashTable.add(h) = base_off.wrapping_add((ip - 2) as u32);
            }
            let h = unsafe {
                let p = src.as_ptr().wrapping_add(ip - 1);
                match MLS {
                    5 => ZSTD_hash5((p as *const u64).read_unaligned().to_le(), hBitsS, 0),
                    6 => ZSTD_hash6((p as *const u64).read_unaligned().to_le(), hBitsS, 0),
                    7 => ZSTD_hash7((p as *const u64).read_unaligned().to_le(), hBitsS, 0),
                    8 => ZSTD_hash8((p as *const u64).read_unaligned().to_le(), hBitsS, 0),
                    _ => ZSTD_hash4((p as *const u32).read_unaligned().to_le(), hBitsS, 0) as usize,
                }
            };
            // SAFETY: `ip <= ilimit` implies `ip - 1 + 8 <= iend`.
            unsafe {
                *chainTable.add(h) = base_off.wrapping_add((ip - 1) as u32);
            }

            while ip <= ilimit && rep_offset2 > 0 {
                // SAFETY: `ip ≤ ilimit = iend - 8` so `ip + 4 ≤ iend
                // = src.len()`; `ip - rep_offset2 ≤ ip` so safe.
                let cval =
                    unsafe { (src.as_ptr().wrapping_add(ip) as *const u32).read_unaligned() };
                let rval = unsafe {
                    (src.as_ptr()
                        .wrapping_add(ip.wrapping_sub(rep_offset2 as usize))
                        as *const u32)
                        .read_unaligned()
                };
                if cval != rval {
                    break;
                }
                let rLength = ZSTD_count(src, ip + 4, ip + 4 - rep_offset2 as usize, iend) + 4;
                std::mem::swap(&mut rep_offset1, &mut rep_offset2);
                let hs = unsafe {
                    let p = src.as_ptr().wrapping_add(ip);
                    match MLS {
                        5 => ZSTD_hash5((p as *const u64).read_unaligned().to_le(), hBitsS, 0),
                        6 => ZSTD_hash6((p as *const u64).read_unaligned().to_le(), hBitsS, 0),
                        7 => ZSTD_hash7((p as *const u64).read_unaligned().to_le(), hBitsS, 0),
                        8 => ZSTD_hash8((p as *const u64).read_unaligned().to_le(), hBitsS, 0),
                        _ => ZSTD_hash4((p as *const u32).read_unaligned().to_le(), hBitsS, 0)
                            as usize,
                    }
                };
                let hl = unsafe {
                    let p = src.as_ptr().wrapping_add(ip);
                    ZSTD_hash8((p as *const u64).read_unaligned().to_le(), hBitsL, 0)
                };
                // SAFETY: hs < ms.chainTable.len(), hl < ms.hashTable.len().
                unsafe {
                    *chainTable.add(hs) = base_off.wrapping_add(ip as u32);
                    *hashTable.add(hl) = base_off.wrapping_add(ip as u32);
                }
                ZSTD_storeSeq(
                    seqStore,
                    0,
                    &src[anchor..anchor],
                    REPCODE_TO_OFFBASE(1),
                    rLength,
                );
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
    rep[0] = if rep_offset1 != 0 {
        rep_offset1
    } else {
        offsetSaved1
    };
    rep[1] = if rep_offset2 != 0 {
        rep_offset2
    } else {
        offsetSaved2
    };
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
    let mls = match cParams.minMatch {
        5 => 5,
        6 => 6,
        7 => 7,
        _ => 4,
    };
    let endIndex = ms.window.base_offset.wrapping_add(src.len() as u32);
    let prefixLowestIndex = ZSTD_getLowestPrefixIndex(ms, endIndex, cParams.windowLog);
    let base_off = ms.window.base_offset;
    let prefixLowest = prefixLowestIndex.saturating_sub(base_off) as usize;
    let dms = ms
        .dictMatchState
        .as_deref()
        .expect("ZSTD_compressBlock_doubleFast_dictMatchState requires dictMatchState");
    let dict = &dms.dictContent;
    let dictStartIndex = dms.window.dictLimit;
    let dictEndIndex = dms.window.nextSrc;
    let dictBaseOff = dms.window.base_offset;
    debug_assert!(dictStartIndex >= dictBaseOff);
    debug_assert!(dictEndIndex >= dictBaseOff);
    let dictStart = dictStartIndex.wrapping_sub(dictBaseOff) as usize;
    let dictEnd = dictEndIndex.wrapping_sub(dictBaseOff) as usize;
    let dictIndexDelta = prefixLowestIndex.wrapping_sub(dictEndIndex);
    let dictHBitsL = dms.cParams.hashLog + ZSTD_SHORT_CACHE_TAG_BITS;
    let dictHBitsS = dms.cParams.chainLog + ZSTD_SHORT_CACHE_TAG_BITS;
    let parseStart = 0usize;
    let istartIndex = base_off.wrapping_add(parseStart as u32);
    let dictAndPrefixLength = istartIndex
        .wrapping_sub(prefixLowestIndex)
        .wrapping_add(dictEndIndex.wrapping_sub(dictStartIndex));
    let maxDistance = if cParams.windowLog >= 31 {
        u32::MAX
    } else {
        1u32 << cParams.windowLog
    };

    debug_assert!(ms.window.dictLimit.wrapping_add(maxDistance) >= endIndex);
    if ZSTD_window_hasExtDict(&ms.window) {
        return ZSTD_compressBlock_doubleFast_extDict_generic(ms, seqStore, rep, src);
    }
    if src.len() < HASH_READ_SIZE {
        return src.len();
    }
    debug_assert!(rep[0] <= dictAndPrefixLength);
    debug_assert!(rep[1] <= dictAndPrefixLength);
    debug_assert!(prefixLowestIndex <= endIndex);
    debug_assert!(dictStart <= dictEnd);
    debug_assert!(dictEnd <= dict.len());
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
    let mut ip = parseStart + usize::from(dictAndPrefixLength == 0);
    let mut anchor = 0usize;

    macro_rules! finish_match_stored {
        ($curr:expr, $mLength:expr) => {{
            ip += $mLength;
            anchor = ip;
            if ip <= ilimit {
                let indexToInsert = $curr.wrapping_add(2);
                let ins = indexToInsert.wrapping_sub(base_off) as usize;
                debug_assert!(ins + 8 <= src.len());
                ms.hashTable[ZSTD_hashPtr(&src[ins..], hBitsL, 8)] = indexToInsert;
                ms.chainTable[ZSTD_hashPtr(&src[ins..], hBitsS, mls)] = indexToInsert;
                ms.hashTable[ZSTD_hashPtr(&src[ip - 2..], hBitsL, 8)] =
                    base_off.wrapping_add((ip - 2) as u32);
                ms.chainTable[ZSTD_hashPtr(&src[ip - 1..], hBitsS, mls)] =
                    base_off.wrapping_add((ip - 1) as u32);
                while ip <= ilimit {
                    let current2 = base_off.wrapping_add(ip as u32);
                    let repIndex2 = current2.wrapping_sub(offset_2);
                    let repInDict2 = repIndex2 < prefixLowestIndex;
                    let repMatch2 = if repInDict2 {
                        repIndex2
                            .wrapping_sub(dictIndexDelta)
                            .wrapping_sub(dictBaseOff) as usize
                    } else {
                        repIndex2.wrapping_sub(base_off) as usize
                    };
                    if ZSTD_index_overlap_check(prefixLowestIndex, repIndex2)
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
                            prefixLowest,
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
        }};
    }

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
                .wrapping_sub(dictIndexDelta)
                .wrapping_sub(dictBaseOff) as usize
        } else {
            repIndex.wrapping_sub(base_off) as usize
        };
        if ZSTD_index_overlap_check(prefixLowestIndex, repIndex)
            && if repInDict {
                repMatch + 4 <= dictEnd
                    && MEM_read32(&dict[repMatch..]) == MEM_read32(&src[ip + 1..])
            } else {
                repMatch + 4 <= src.len()
                    && MEM_read32(&src[repMatch..]) == MEM_read32(&src[ip + 1..])
            }
        {
            let repMatchEnd = if repInDict { dictEnd } else { iend };
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
            finish_match_stored!(curr, mLength);
            continue;
        }

        if matchIndexL >= prefixLowestIndex {
            let matchLong = matchIndexL.wrapping_sub(base_off) as usize;
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
                finish_match_stored!(curr, mLength);
                continue;
            }
        } else if dictTagsMatchL {
            let dictMatchIndexL = dictMatchIndexAndTagL >> ZSTD_SHORT_CACHE_TAG_BITS;
            let mut dictMatchL = dictMatchIndexL.wrapping_sub(dictBaseOff) as usize;
            if dictMatchL < dictEnd
                && dictMatchL > dictStart
                && dictMatchL + 8 <= dictEnd
                && MEM_read64(&dict[dictMatchL..]) == MEM_read64(&src[ip..])
            {
                let mut mLength = ZSTD_count_2segments(
                    src,
                    ip + 8,
                    iend,
                    prefixLowest,
                    dict,
                    dictMatchL + 8,
                    dictEnd,
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
                finish_match_stored!(curr, mLength);
                continue;
            }
        }

        let short_match = if matchIndexS > prefixLowestIndex {
            let match_pos = matchIndexS.wrapping_sub(base_off) as usize;
            match_pos + 4 <= src.len() && MEM_read32(&src[match_pos..]) == MEM_read32(&src[ip..])
        } else if dictTagsMatchS {
            let dictMatchIndexS = dictMatchIndexAndTagS >> ZSTD_SHORT_CACHE_TAG_BITS;
            let match_pos = dictMatchIndexS.wrapping_sub(dictBaseOff) as usize;
            matchIndexS = dictMatchIndexS.wrapping_add(dictIndexDelta);
            match_pos > dictStart
                && match_pos + 4 <= dictEnd
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
                let mut matchL3 = matchIndexL3.wrapping_sub(base_off) as usize;
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
                    finish_match_stored!(curr, mLength);
                    continue;
                }
            } else if dictTagsMatchL3 {
                let dictMatchIndexL3 = dictMatchIndexAndTagL3 >> ZSTD_SHORT_CACHE_TAG_BITS;
                let mut dictMatchL3 = dictMatchIndexL3.wrapping_sub(dictBaseOff) as usize;
                if dictMatchL3 < dictEnd
                    && dictMatchL3 > dictStart
                    && dictMatchL3 + 8 <= dictEnd
                    && MEM_read64(&dict[dictMatchL3..]) == MEM_read64(&src[ip + 1..])
                {
                    let mut mLength = ZSTD_count_2segments(
                        src,
                        ip + 1 + 8,
                        iend,
                        prefixLowest,
                        dict,
                        dictMatchL3 + 8,
                        dictEnd,
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
                    finish_match_stored!(curr, mLength);
                    continue;
                }
            }

            let mut mLength;
            let offset;
            if matchIndexS < prefixLowestIndex {
                let mut match_pos = matchIndexS
                    .wrapping_sub(dictIndexDelta)
                    .wrapping_sub(dictBaseOff) as usize;
                mLength = ZSTD_count_2segments(
                    src,
                    ip + 4,
                    iend,
                    prefixLowest,
                    dict,
                    match_pos + 4,
                    dictEnd,
                ) + 4;
                offset = curr.wrapping_sub(matchIndexS);
                while ip > anchor && match_pos > dictStart && src[ip - 1] == dict[match_pos - 1] {
                    ip -= 1;
                    match_pos -= 1;
                    mLength += 1;
                }
            } else {
                let mut match_pos = matchIndexS.wrapping_sub(base_off) as usize;
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
            finish_match_stored!(curr, mLength);
            continue;
        }

        ip += ((ip - anchor) >> kSearchStrength) + 1;
        continue;
    }

    rep[0] = offset_1;
    rep[1] = offset_2;
    iend - anchor
}

/// Port of `ZSTD_compressBlock_doubleFast_extDict`. Public entry for
/// the double-fast strategy when the window holds an external
/// dictionary segment; dispatches to
/// `ZSTD_compressBlock_doubleFast_extDict_generic`.
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
    let mls = match cParams.minMatch {
        5 => 5,
        6 => 6,
        7 => 7,
        _ => 4,
    };
    if !ZSTD_window_hasExtDict(&ms.window) {
        return ZSTD_compressBlock_doubleFast_noDict_generic(ms, seqStore, rep, src, 0, mls);
    }
    let hBitsL = cParams.hashLog;
    let hBitsS = cParams.chainLog;
    let base_off = ms.window.base_offset;
    let dict_base_off = ms.window.dictBase_offset;
    let iend = src.len();
    let ilimit = iend.saturating_sub(HASH_READ_SIZE);
    let endIndex = base_off.wrapping_add(src.len() as u32);
    let dictStartIndex = ZSTD_getLowestMatchIndex(ms, endIndex, cParams.windowLog);
    let prefixStartIndex = ms.window.dictLimit.max(dictStartIndex);
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
    debug_assert!(dictStartIndex >= dict_base_off);
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
                debug_assert!(repMatch + 4 <= dictEnd);
                unsafe {
                    (dict.as_ptr().wrapping_add(repMatch) as *const u32).read_unaligned()
                        == (src.as_ptr().wrapping_add(ip + 1) as *const u32).read_unaligned()
                }
            } else {
                debug_assert!(repMatch + 4 <= src.len());
                unsafe {
                    (src.as_ptr().wrapping_add(repMatch) as *const u32).read_unaligned()
                        == (src.as_ptr().wrapping_add(ip + 1) as *const u32).read_unaligned()
                }
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
        } else if matchLongIndex > dictStartIndex && {
            if matchLongInDict {
                debug_assert!(matchLong + 8 <= dictEnd);
                unsafe {
                    (dict.as_ptr().wrapping_add(matchLong) as *const u64).read_unaligned()
                        == (src.as_ptr().wrapping_add(ip) as *const u64).read_unaligned()
                }
            } else {
                debug_assert!(matchLong + 8 <= src.len());
                unsafe {
                    (src.as_ptr().wrapping_add(matchLong) as *const u64).read_unaligned()
                        == (src.as_ptr().wrapping_add(ip) as *const u64).read_unaligned()
                }
            }
        } {
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
        } else if matchIndex > dictStartIndex && {
            if matchInDict {
                debug_assert!(match_pos + 4 <= dictEnd);
                unsafe {
                    (dict.as_ptr().wrapping_add(match_pos) as *const u32).read_unaligned()
                        == (src.as_ptr().wrapping_add(ip) as *const u32).read_unaligned()
                }
            } else {
                debug_assert!(match_pos + 4 <= src.len());
                unsafe {
                    (src.as_ptr().wrapping_add(match_pos) as *const u32).read_unaligned()
                        == (src.as_ptr().wrapping_add(ip) as *const u32).read_unaligned()
                }
            }
        } {
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
                && if match3InDict {
                    debug_assert!(match3 + 8 <= dictEnd);
                    unsafe {
                        (dict.as_ptr().wrapping_add(match3) as *const u64).read_unaligned()
                            == (src.as_ptr().wrapping_add(ip + 1) as *const u64).read_unaligned()
                    }
                } else {
                    debug_assert!(match3 + 8 <= src.len());
                    unsafe {
                        (src.as_ptr().wrapping_add(match3) as *const u64).read_unaligned()
                            == (src.as_ptr().wrapping_add(ip + 1) as *const u64).read_unaligned()
                    }
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
                        debug_assert!(repMatch2 + 4 <= dictEnd);
                        unsafe {
                            (dict.as_ptr().wrapping_add(repMatch2) as *const u32).read_unaligned()
                                == (src.as_ptr().wrapping_add(ip) as *const u32).read_unaligned()
                        }
                    } else {
                        debug_assert!(repMatch2 + 4 <= src.len());
                        unsafe {
                            (src.as_ptr().wrapping_add(repMatch2) as *const u32).read_unaligned()
                                == (src.as_ptr().wrapping_add(ip) as *const u32).read_unaligned()
                        }
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
        let dict = b"the quick brown fox jumps over the lazy dog. ".to_vec();
        let mut dms = build_ms();
        dms.dictContent = dict.clone();
        dms.window.nextSrc = dms.window.base_offset.wrapping_add(dict.len() as u32);
        ZSTD_fillDoubleHashTableForCDict(
            &mut dms,
            &dict,
            ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_full,
        );

        let mut dict_ms = build_ms();
        dict_ms.dictMatchState = Some(Box::new(dms));
        dict_ms.window.base_offset =
            crate::compress::match_state::ZSTD_WINDOW_START_INDEX.wrapping_add(dict.len() as u32);
        dict_ms.window.dictLimit = dict_ms.window.base_offset;
        dict_ms.window.lowLimit = dict_ms.window.base_offset;
        dict_ms.nextToUpdate = dict_ms.window.base_offset;
        dict_ms.loadedDictEnd = dict.len() as u32;

        let mut seq = SeqStore_t::with_capacity(1024, 131072);
        let mut rep = [1u32, 4, 8];
        let last_lits =
            ZSTD_compressBlock_doubleFast_dictMatchState(&mut dict_ms, &mut seq, &mut rep, &src);
        assert!(
            last_lits < src.len(),
            "dictMatchState emitted only literals"
        );
        assert!(
            !seq.sequences.is_empty(),
            "dictMatchState emitted no sequences"
        );

        let mut ext_ms = build_ms();
        let mut seq = SeqStore_t::with_capacity(1024, 131072);
        let mut rep = [1u32, 4, 8];
        let last_lits =
            ZSTD_compressBlock_doubleFast_extDict(&mut ext_ms, &mut seq, &mut rep, &src);
        assert!(last_lits < src.len(), "extDict emitted only literals");
        assert!(!seq.sequences.is_empty(), "extDict emitted no sequences");
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
        ms.window.base_offset =
            crate::compress::match_state::ZSTD_WINDOW_START_INDEX.wrapping_add(dict.len() as u32);
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
    fn double_fast_dict_match_state_accepts_repcodes_in_current_prefix_and_dict() {
        let dict: Vec<u8> = (0..64u8).collect();
        let src: Vec<u8> = (128..152u8).collect();
        let prefix_len = 32u32;

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
            .wrapping_add(dict.len() as u32)
            .wrapping_add(prefix_len);
        ms.window.dictLimit = ms.window.base_offset.wrapping_sub(prefix_len);
        ms.window.lowLimit = ms.window.dictLimit;
        ms.nextToUpdate = ms.window.base_offset;
        ms.loadedDictEnd = dict.len() as u32;

        let mut seq = SeqStore_t::with_capacity(16, 1024);
        let dict_and_prefix_len = dict.len() as u32 + prefix_len;
        let mut rep = [dict_and_prefix_len, dict.len() as u32 + 1, 8];
        let _last_lits =
            ZSTD_compressBlock_doubleFast_dictMatchState(&mut ms, &mut seq, &mut rep, &src);

        assert_eq!(rep[0], dict_and_prefix_len);
        assert_eq!(rep[1], dict.len() as u32 + 1);
    }

    #[test]
    fn double_fast_dict_match_state_wraps_dict_index_delta_like_c() {
        let dict: Vec<u8> = (0..96u8).collect();
        let src: Vec<u8> = dict[48..80].to_vec();
        let overlapped_prefix = 32u32;

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
        dms.hashTable.fill(0);

        let dict_base = dms.window.base_offset;
        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.dictMatchState = Some(Box::new(dms));
        ms.window.base_offset = dict_base.wrapping_add(dict.len() as u32);
        ms.window.dictLimit = ms.window.base_offset.wrapping_sub(overlapped_prefix);
        ms.window.lowLimit = ms.window.dictLimit;
        ms.nextToUpdate = ms.window.base_offset;
        ms.loadedDictEnd = dict.len() as u32;

        let mut seq = SeqStore_t::with_capacity(16, 1024);
        let mut rep = [1u32, 4, 8];
        let _last_lits =
            ZSTD_compressBlock_doubleFast_dictMatchState(&mut ms, &mut seq, &mut rep, &src);

        assert!(!seq.sequences.is_empty());
        assert_eq!(
            OFFBASE_TO_OFFSET(seq.sequences[0].offBase),
            dict.len() as u32 - (48 - overlapped_prefix)
        );
    }

    #[test]
    fn double_fast_dict_match_state_ignores_bytes_past_dict_end() {
        let valid_len = 32usize;
        let stale_pos = 40usize;
        let mut dict = b"abcdefghijklmnopqrstuvwxyz012345".to_vec();
        dict.extend_from_slice(b"ZZZZZZZZ");
        dict.extend_from_slice(b"stale-match-ABCDEFGH");
        let src = b"stale-match-ABCDEFGH".to_vec();

        let cp = ZSTD_compressionParameters {
            windowLog: 17,
            hashLog: 12,
            chainLog: 12,
            minMatch: 4,
            strategy: 2,
            ..Default::default()
        };

        let mut dms = ZSTD_MatchState_t::new(cp);
        dms.dictContent = dict;
        dms.chainTable.resize(1usize << dms.cParams.chainLog, 0);
        dms.window.dictLimit = dms.window.base_offset;
        dms.window.nextSrc = dms.window.base_offset.wrapping_add(valid_len as u32);
        let stale_index = dms.window.base_offset.wrapping_add(stale_pos as u32);
        let dict_hash_and_tag =
            ZSTD_hashPtr(&src, dms.cParams.hashLog + ZSTD_SHORT_CACHE_TAG_BITS, 8);
        crate::compress::match_state::ZSTD_writeTaggedIndex(
            &mut dms.hashTable,
            dict_hash_and_tag,
            stale_index,
        );

        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.dictMatchState = Some(Box::new(dms));
        ms.window.base_offset =
            crate::compress::match_state::ZSTD_WINDOW_START_INDEX.wrapping_add(valid_len as u32);
        ms.window.dictLimit = ms.window.base_offset;
        ms.window.lowLimit = ms.window.base_offset;
        ms.nextToUpdate = ms.window.base_offset;
        ms.loadedDictEnd = valid_len as u32;

        let mut seq = SeqStore_t::with_capacity(16, 1024);
        let mut rep = [1u32, 4, 8];
        let last_lits =
            ZSTD_compressBlock_doubleFast_dictMatchState(&mut ms, &mut seq, &mut rep, &src);

        assert_eq!(last_lits, src.len());
        assert!(seq.sequences.is_empty());
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
        ms.window.dictLimit =
            crate::compress::match_state::ZSTD_WINDOW_START_INDEX.wrapping_add(dict.len() as u32);
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

    #[test]
    fn double_fast_ext_dict_uses_window_dict_base_not_loaded_dict_end() {
        let dict = b"abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ".to_vec();
        let src = b"mnopqrstuvwxyz012345".to_vec();
        let dict_base = crate::compress::match_state::ZSTD_WINDOW_START_INDEX + 100;

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
        ms.window.base_offset = dict_base;
        ms.nextToUpdate = dict_base;
        ZSTD_fillDoubleHashTableForCCtx(&mut ms, &dict, ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_fast);

        ms.window.dictBase_offset = dict_base;
        ms.window.lowLimit = dict_base;
        ms.window.dictLimit = dict_base.wrapping_add(dict.len() as u32);
        ms.window.base_offset = ms.window.dictLimit;
        ms.window.nextSrc = ms.window.base_offset.wrapping_add(src.len() as u32);
        ms.loadedDictEnd = ms.window.dictLimit.wrapping_add(17);

        let mut seq = SeqStore_t::with_capacity(128, 4096);
        let mut rep = [1u32, 4, 8];
        let last_lits = ZSTD_compressBlock_doubleFast_extDict(&mut ms, &mut seq, &mut rep, &src);

        assert!(last_lits < src.len(), "extDict path emitted only literals");
        assert!(!seq.sequences.is_empty());
        assert_eq!(OFFBASE_TO_OFFSET(seq.sequences[0].offBase), 50);
    }
}
