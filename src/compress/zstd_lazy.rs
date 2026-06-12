//! Translation of `lib/compress/zstd_lazy.c` — strategies 3..=6
//! (greedy / lazy / lazy2 / btlazy2). Hash-chain infra
//! (`ZSTD_insertAndFindFirstIndex`, `ZSTD_HcFindBestMatch_noDict`) +
//! all four noDict entry points (`ZSTD_compressBlock_{greedy,lazy,lazy2,btlazy2}`)
//! are ported. The no-dict `btlazy2` binary-tree path is ported, and
//! the shared lazy parser now also drives the dictMatchState /
//! extDict / row-hash wrappers instead of routing them through the
//! old no-dict fallback.

#![allow(non_snake_case)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::type_complexity)]
#![allow(clippy::unnecessary_cast)]

use crate::common::bits::{
    ZSTD_countTrailingZeros64, ZSTD_highbit32, ZSTD_rotateRight_U16, ZSTD_rotateRight_U32,
    ZSTD_rotateRight_U64,
};
use crate::common::mem::MEM_read32;
use crate::compress::match_state::{
    ZSTD_MatchState_t, ZSTD_dictMode_e, ZSTD_index_overlap_check, ZSTD_ROW_HASH_CACHE_SIZE,
};
use crate::compress::seq_store::{
    SeqStore_t, ZSTD_storeSeq, OFFBASE_IS_OFFSET, OFFBASE_TO_OFFSET, OFFSET_TO_OFFBASE,
    REPCODE_TO_OFFBASE, ZSTD_REP_NUM,
};
use crate::compress::zstd_compress::ZSTD_LAZY_DDSS_BUCKET_LOG;
use crate::compress::zstd_fast::{kSearchStrength, ZSTD_getLowestMatchIndex};
use crate::compress::zstd_hashes::{
    ZSTD_count, ZSTD_count_2segments, ZSTD_hashPtr, ZSTD_hashPtrSalted,
    ZSTD_hashPtrSalted_at_unchecked,
};
use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
use core::mem::MaybeUninit;

pub const ZSTD_ROW_HASH_TAG_BITS: u32 = 8;
pub const ZSTD_ROW_HASH_TAG_MASK: u32 = (1u32 << ZSTD_ROW_HASH_TAG_BITS) - 1;
pub const ZSTD_ROW_HASH_MAX_ENTRIES: u32 = 64;
pub const ZSTD_ROW_HASH_CACHE_MASK: usize = ZSTD_ROW_HASH_CACHE_SIZE - 1;
pub const kLazySkippingStep: usize = 8;

pub type ZSTD_VecMask = u64;

/// Rust-only helper: unaligned 32-bit native-endian load from
/// `&src[pos..pos+4]`. `unsafe` because it bypasses bounds checks on
/// the hot lazy/row-hash path.
#[inline(always)]
unsafe fn read32_at(src: &[u8], pos: usize) -> u32 {
    debug_assert!(pos + 4 <= src.len());
    (src.as_ptr().add(pos) as *const u32).read_unaligned()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum searchMethod_e {
    search_hashChain = 0,
    search_binaryTree = 1,
    search_rowHash = 2,
}

/// Port of `ZSTD_VecMask_next`.
#[inline(always)]
pub fn ZSTD_VecMask_next(val: ZSTD_VecMask) -> u32 {
    ZSTD_countTrailingZeros64(val)
}

/// Port of `ZSTD_row_nextIndex`.
#[inline]
pub fn ZSTD_row_nextIndex(tagRow: &mut [u8], rowMask: u32) -> u32 {
    let mut next = (tagRow[0] as u32).wrapping_sub(1) & rowMask;
    next += ((next == 0) as u32) * rowMask;
    tagRow[0] = next as u8;
    next
}

/// Port of `ZSTD_isAligned`.
#[inline]
pub fn ZSTD_isAligned<T>(ptr: *const T, align: usize) -> bool {
    debug_assert!(align.is_power_of_two());
    ((ptr as usize) & (align - 1)) == 0
}

/// Port of `ZSTD_row_prefetch`. Mirrors upstream's `PREFETCH_L1` macro
/// — issues real `_mm_prefetch` / `prfm pldl1keep` hints rather than
/// hint-stubbed slice creations.
#[inline(always)]
pub fn ZSTD_row_prefetch(hashTable: &[u32], tagTable: &[u8], relRow: u32, rowLog: u32) {
    use crate::common::zstd_internal::ZSTD_prefetchL1;
    // `relRow` is always `< hashTable.len()` and `< tagTable.len()` by
    // construction (caller derived it as `(hash >> TAG_BITS) << rowLog`,
    // and the tables are sized to accommodate every possible value).
    // Use `wrapping_offset` to skip the per-call bounds checks — the
    // hot row-hash path showed `prefetcht0` and its surrounding
    // `cmp/jb` branches consuming ~9% of L7 cycles. A bad prefetch
    // address is silently ignored by the CPU; the wrapping_offset
    // keeps things sound by avoiding `.add`'s out-of-allocation UB.
    let relRow = relRow as isize;
    ZSTD_prefetchL1(hashTable.as_ptr().wrapping_offset(relRow));
    if rowLog >= 5 {
        ZSTD_prefetchL1(hashTable.as_ptr().wrapping_offset(relRow + 16));
    }
    ZSTD_prefetchL1(tagTable.as_ptr().wrapping_offset(relRow));
    if rowLog == 6 {
        ZSTD_prefetchL1(tagTable.as_ptr().wrapping_offset(relRow + 32));
    }
}

/// Port of `ZSTD_row_fillHashCache`. `idx` is the absolute window-space
/// index (= `base_off + ip` in caller terms) that the cache will start
/// pre-filling from; `iLimit` is the slice-offset upper bound on bytes
/// the matcher may still inspect.
///
#[inline(always)]
pub fn ZSTD_row_fillHashCache(
    ms: &mut ZSTD_MatchState_t,
    base: &[u8],
    rowLog: u32,
    mls: u32,
    mut idx: u32,
    iLimit: usize,
) {
    let hashLog = ms.rowHashLog;
    let base_off = ms.window.base_offset;
    if idx < base_off {
        idx = base_off;
    }
    let idx_slice = idx.saturating_sub(base_off) as usize;
    let maxElemsToPrefetch = if idx_slice > iLimit {
        0
    } else {
        (iLimit - idx_slice + 1) as u32
    };
    let lim = idx.wrapping_add(
        (crate::compress::match_state::ZSTD_ROW_HASH_CACHE_SIZE as u32).min(maxElemsToPrefetch),
    );
    while idx < lim {
        let off = idx.wrapping_sub(base_off) as usize;
        // SAFETY: `lim` is bounded by `iLimit + 1` and `iLimit ≤
        // base.len() - 8` (matcher invariant), so each iteration's
        // `off + 8 ≤ base.len()`.
        let hash = unsafe {
            ZSTD_hashPtrSalted_at_unchecked(
                base,
                off,
                hashLog + ZSTD_ROW_HASH_TAG_BITS,
                mls,
                ms.hashSalt,
            ) as u32
        };
        let row = (hash >> ZSTD_ROW_HASH_TAG_BITS) << rowLog;
        ZSTD_row_prefetch(&ms.hashTable, &ms.tagTable, row, rowLog);
        ms.hashCache[idx as usize & ZSTD_ROW_HASH_CACHE_MASK] = hash;
        idx = idx.wrapping_add(1);
    }
}

/// Port of `ZSTD_row_nextCachedHash`. `idx` is the absolute
/// window-space index. `base_off` lets us recover the slice offset
/// (= `idx - base_off`) for byte reads.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn ZSTD_row_nextCachedHash(
    cache: &mut [u32; crate::compress::match_state::ZSTD_ROW_HASH_CACHE_SIZE],
    hashTable: &[u32],
    tagTable: &[u8],
    base: &[u8],
    idx: u32,
    base_off: u32,
    hashLog: u32,
    rowLog: u32,
    mls: u32,
    hashSalt: u64,
) -> u32 {
    let off = idx.wrapping_sub(base_off) as usize
        + crate::compress::match_state::ZSTD_ROW_HASH_CACHE_SIZE;
    // SAFETY: caller's row-hash matcher invariant gives
    // `idx - base_off + CACHE_SIZE + 8 ≤ base.len()` — i.e.,
    // `ip + CACHE_SIZE + 8 ≤ iend` is implied by the matcher's
    // `ip ≤ ilimit = iend - 8 - CACHE_SIZE` bound.
    let newHash = unsafe {
        ZSTD_hashPtrSalted_at_unchecked(base, off, hashLog + ZSTD_ROW_HASH_TAG_BITS, mls, hashSalt)
            as u32
    };
    let row = (newHash >> ZSTD_ROW_HASH_TAG_BITS) << rowLog;
    ZSTD_row_prefetch(hashTable, tagTable, row, rowLog);
    let slot = idx as usize & ZSTD_ROW_HASH_CACHE_MASK;
    let mut hash = unsafe { *cache.get_unchecked(slot) };
    let hash_bits = hashLog + ZSTD_ROW_HASH_TAG_BITS;
    if hash_bits < u32::BITS {
        hash &= (1u32 << hash_bits) - 1;
    }
    unsafe {
        *cache.get_unchecked_mut(slot) = newHash;
    }
    hash
}

/// Port of `ZSTD_row_update_internalImpl`. `updateStartIdx`/`updateEndIdx`
/// are absolute indices (matching upstream's `(U32)(ip - base)` and the
/// chain-hash convention). The `base` slice is the matcher's input buffer;
/// we subtract `base_off` to recover the slice offset for byte reads.
#[inline(always)]
pub fn ZSTD_row_update_internalImpl(
    ms: &mut ZSTD_MatchState_t,
    mut updateStartIdx: u32,
    updateEndIdx: u32,
    mls: u32,
    rowLog: u32,
    rowMask: u32,
    useCache: bool,
    base: &[u8],
) {
    let hashLog = ms.rowHashLog;
    let base_off = ms.window.base_offset;
    if updateStartIdx < base_off {
        updateStartIdx = base_off;
    }
    while updateStartIdx < updateEndIdx {
        let slice_off = updateStartIdx.wrapping_sub(base_off) as usize;
        let hash = if useCache {
            ZSTD_row_nextCachedHash(
                &mut ms.hashCache,
                &ms.hashTable,
                &ms.tagTable,
                base,
                updateStartIdx,
                base_off,
                hashLog,
                rowLog,
                mls,
                ms.hashSalt,
            )
        } else {
            // SAFETY: caller passes `base = input_buf` (the source) and
            // `updateStartIdx ≤ target = base_off + ip`. `ip` is bounded
            // by the row-hash matcher's `ilimit < base.len() - 8`, so
            // `slice_off + 8 ≤ base.len()`.
            unsafe {
                ZSTD_hashPtrSalted_at_unchecked(
                    base,
                    slice_off,
                    hashLog + ZSTD_ROW_HASH_TAG_BITS,
                    mls,
                    ms.hashSalt,
                ) as u32
            }
        };
        let relRow = ((hash >> ZSTD_ROW_HASH_TAG_BITS) << rowLog) as usize;
        // SAFETY: `relRow + (1 << rowLog) ≤ tagTable.len()` by table
        // sizing — `relRow = (hash >> TAG_BITS) << rowLog` where the
        // shifted hash fits in `(rowHashLog + rowLog)` bits, matching
        // `tagTable.len() = 1 << (rowHashLog + rowLog)`. Same for the
        // hashTable. `pos < rowEntries = 1 << rowLog`. The bounds-
        // checked slice indexing emits per-call `cmp/jb` pairs that
        // dominate this per-byte hot loop (~17% of L7 cycles).
        let pos = unsafe {
            let head = ms.tagTable.get_unchecked_mut(relRow);
            let mut next = (*head as u32).wrapping_sub(1) & rowMask;
            next += ((next == 0) as u32) * rowMask;
            *head = next as u8;
            next as usize
        };
        unsafe {
            *ms.tagTable.get_unchecked_mut(relRow + pos) = (hash & ZSTD_ROW_HASH_TAG_MASK) as u8;
            *ms.hashTable.get_unchecked_mut(relRow + pos) = updateStartIdx;
        }
        updateStartIdx = updateStartIdx.wrapping_add(1);
    }
}

/// Port of `ZSTD_row_update_internal`. `ip` is the slice offset of the
/// current cursor; we convert it to the absolute window-space index
/// (`base_off + ip`) before storing it as `nextToUpdate` / hash-table
/// values, matching upstream's pointer-based `(U32)(ip - base)`
/// convention. Storing absolute indices keeps the "0 = uninitialized"
/// sentinel distinct from real entries (which start at `base_off >= 2`
/// for any window) and keeps row-hash compatible with the chain-hash
/// nextToUpdate convention so the same field can be shared and clamped
/// against `ms.window.lowLimit` between blocks.
#[inline(always)]
pub fn ZSTD_row_update_internal(
    ms: &mut ZSTD_MatchState_t,
    ip: usize,
    mls: u32,
    rowLog: u32,
    rowMask: u32,
    useCache: bool,
    base: &[u8],
) {
    let base_off = ms.window.base_offset;
    let mut idx = ms.nextToUpdate;
    let target = base_off.wrapping_add(ip as u32);
    const K_SKIP_THRESHOLD: u32 = 384;
    const K_MAX_MATCH_START_POSITIONS_TO_UPDATE: u32 = 96;
    const K_MAX_MATCH_END_POSITIONS_TO_UPDATE: u32 = 32;

    if useCache && target.wrapping_sub(idx) > K_SKIP_THRESHOLD {
        let bound = idx.wrapping_add(K_MAX_MATCH_START_POSITIONS_TO_UPDATE);
        ZSTD_row_update_internalImpl(ms, idx, bound, mls, rowLog, rowMask, useCache, base);
        idx = target.wrapping_sub(K_MAX_MATCH_END_POSITIONS_TO_UPDATE);
        ZSTD_row_fillHashCache(ms, base, rowLog, mls, idx, ip + 1);
    }
    ZSTD_row_update_internalImpl(ms, idx, target, mls, rowLog, rowMask, useCache, base);
    ms.nextToUpdate = target;
}

/// Port of `ZSTD_row_update`.
pub fn ZSTD_row_update(ms: &mut ZSTD_MatchState_t, ip: usize, base: &[u8]) {
    let rowLog = ms.cParams.searchLog.clamp(4, 6);
    let rowMask = (1u32 << rowLog) - 1;
    let mls = ms.cParams.minMatch.min(6);
    ZSTD_row_update_internal(ms, ip, mls, rowLog, rowMask, false, base);
}

/// Port of `ZSTD_row_matchMaskGroupWidth`. Always 1 in our scalar
/// SWAR formulation; marked `#[inline(always)]` so the constant
/// `1` propagates into the per-byte row-hash inner loop and the
/// `* groupWidth` / `/ groupWidth` patterns fold to no-ops.
#[inline(always)]
pub fn ZSTD_row_matchMaskGroupWidth(rowEntries: u32) -> u32 {
    debug_assert!(matches!(rowEntries, 16 | 32 | 64));
    1
}

/// Port of `ZSTD_row_getSSEMask`.
pub fn ZSTD_row_getSSEMask(nbChunks: i32, src: &[u8], tag: u8, head: u32) -> ZSTD_VecMask {
    let rowEntries = (nbChunks as usize) * 16;
    ZSTD_row_getMatchMask_scalar(&src[..rowEntries], tag, head, rowEntries as u32)
}

/// Port of `ZSTD_row_getNEONMask`.
pub fn ZSTD_row_getNEONMask(
    rowEntries: u32,
    src: &[u8],
    tag: u8,
    headGrouped: u32,
) -> ZSTD_VecMask {
    debug_assert!(matches!(rowEntries, 16 | 32 | 64));
    let rowEntries = rowEntries as usize;
    let slice = &src[..rowEntries];
    let scalarMask = ZSTD_row_getMatchMask_scalar(slice, tag, headGrouped, rowEntries as u32);
    let validMask = match rowEntries {
        16 => 0x1111_1111_1111_1111u64,
        32 => 0x5555_5555_5555_5555u64,
        _ => u64::MAX,
    };
    scalarMask & validMask
}

/// Port of `ZSTD_row_getRVVMask`.
pub fn ZSTD_row_getRVVMask(rowEntries: i32, src: &[u8], tag: u8, head: u32) -> ZSTD_VecMask {
    debug_assert!(matches!(rowEntries, 16 | 32 | 64));
    let rowEntries = rowEntries as usize;
    let slice = &src[..rowEntries];
    let scalarMask = ZSTD_row_getMatchMask_scalar(slice, tag, head, rowEntries as u32);
    match rowEntries {
        16 => scalarMask & 0xFFFF,
        32 => scalarMask & 0xFFFF_FFFF,
        _ => scalarMask,
    }
}

/// SWAR (SIMD-Within-A-Register) port of upstream's scalar
/// `ZSTD_row_getMatchMask` fallback. Processes `usize`-sized chunks at a
/// time, detecting matching bytes via `((x | 0x80) - 0x01) | x) & 0x80`
/// after `x ^= splat(tag)`. Per-byte high bits are then packed via the
/// `extractMagic` multiply trick. Matches upstream's little-endian path.
#[inline(always)]
fn ZSTD_row_getMatchMask_scalar(
    src: &[u8],
    tag: u8,
    headGrouped: u32,
    rowEntries: u32,
) -> ZSTD_VecMask {
    const CHUNK_SIZE: usize = core::mem::size_of::<usize>();
    let shift_amount: u32 = ((CHUNK_SIZE * 8) - CHUNK_SIZE) as u32;
    let x_ff: usize = !0usize;
    let x01: usize = x_ff / 0xFF;
    let x80: usize = x01 << 7;
    let splat_char: usize = (tag as usize).wrapping_mul(x01);
    let extract_magic: usize = (x_ff / 0x7F) >> CHUNK_SIZE;

    let mut matches: u64 = 0;
    let mut i: i32 = (rowEntries as i32) - (CHUNK_SIZE as i32);
    const {
        assert!(core::mem::size_of::<usize>() == 4 || core::mem::size_of::<usize>() == 8);
    }
    while i >= 0 {
        let off = i as usize;
        debug_assert!(off + CHUNK_SIZE <= src.len());
        let mut chunk = unsafe { (src.as_ptr().add(off) as *const usize).read_unaligned() };
        chunk = chunk.to_le();
        chunk ^= splat_char;
        chunk = (((chunk | x80).wrapping_sub(x01)) | chunk) & x80;
        matches <<= CHUNK_SIZE;
        matches |= (chunk.wrapping_mul(extract_magic) >> shift_amount) as u64;
        i -= CHUNK_SIZE as i32;
    }
    matches = !matches;
    match rowEntries {
        16 => ZSTD_rotateRight_U16(matches as u16, headGrouped) as u64,
        32 => ZSTD_rotateRight_U32(matches as u32, headGrouped) as u64,
        _ => ZSTD_rotateRight_U64(matches, headGrouped),
    }
}

/// Port of `ZSTD_row_getMatchMask`.
#[inline(always)]
pub fn ZSTD_row_getMatchMask(
    tagRow: &[u8],
    tag: u8,
    headGrouped: u32,
    rowEntries: u32,
) -> ZSTD_VecMask {
    debug_assert!(matches!(rowEntries, 16 | 32 | 64));
    debug_assert!(rowEntries <= ZSTD_ROW_HASH_MAX_ENTRIES);
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{
            _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_set1_epi8,
        };

        debug_assert!(tagRow.len() >= rowEntries as usize);
        let tagv = unsafe { _mm_set1_epi8(tag as i8) };
        let base = tagRow.as_ptr();
        let mut matches = 0u64;
        let chunks = rowEntries as usize / 16;
        for chunk in 0..chunks {
            let v = unsafe { _mm_loadu_si128(base.add(chunk * 16) as *const _) };
            let eq = unsafe { _mm_cmpeq_epi8(v, tagv) };
            matches |= (unsafe { _mm_movemask_epi8(eq) } as u32 as u64) << (chunk * 16);
        }
        return match rowEntries {
            16 => ZSTD_rotateRight_U16(matches as u16, headGrouped) as u64,
            32 => ZSTD_rotateRight_U32(matches as u32, headGrouped) as u64,
            _ => ZSTD_rotateRight_U64(matches, headGrouped),
        };
    }
    #[cfg(not(target_arch = "x86_64"))]
    ZSTD_row_getMatchMask_scalar(tagRow, tag, headGrouped, rowEntries)
}

/// Port of `ZSTD_dedicatedDictSearch_lazy_loadDictionary`.
pub fn ZSTD_dedicatedDictSearch_lazy_loadDictionary(
    ms: &mut ZSTD_MatchState_t,
    base: &[u8],
    ip: usize,
) {
    let target = ip as u32;
    let hashTable = &mut ms.hashTable;
    let chainTable = &mut ms.chainTable;
    let chainSize = 1u32 << ms.cParams.chainLog;
    let mut idx = ms.nextToUpdate;
    let minChain = if chainSize < target.saturating_sub(idx) {
        target - chainSize
    } else {
        idx
    };
    let bucketSize = 1u32 << ZSTD_LAZY_DDSS_BUCKET_LOG;
    let cacheSize = bucketSize - 1;
    let chainAttempts = (1u32 << ms.cParams.searchLog).saturating_sub(cacheSize);
    let chainLimit = chainAttempts.min(255);

    let hashLog = ms.cParams.hashLog - ZSTD_LAZY_DDSS_BUCKET_LOG;
    let tmpHashSize = 1usize << hashLog;
    let tmpChainSize = (((1u32 << ZSTD_LAZY_DDSS_BUCKET_LOG) - 1) << hashLog) as u32;
    let tmpMinChain = if tmpChainSize < target {
        target - tmpChainSize
    } else {
        idx
    };

    debug_assert!(ms.cParams.chainLog <= 24);
    debug_assert!(ms.cParams.hashLog > ms.cParams.chainLog);
    debug_assert!(idx != 0);
    debug_assert!(tmpMinChain <= minChain);

    if hashTable.len() < tmpHashSize + tmpChainSize as usize {
        return;
    }
    if chainTable.len() < chainSize as usize {
        chainTable.resize(chainSize as usize, 0);
    }

    {
        let (tmpHashTable, tmpChainTable) = hashTable.split_at_mut(tmpHashSize);

        while idx < target {
            let h = ZSTD_hashPtr(&base[idx as usize..], hashLog, ms.cParams.minMatch) as usize;
            if idx >= tmpMinChain {
                tmpChainTable[(idx - tmpMinChain) as usize] = tmpHashTable[h];
            }
            tmpHashTable[h] = idx;
            idx = idx.wrapping_add(1);
        }

        let mut chainPos = 0u32;
        for hashIdx in 0..(1u32 << hashLog) {
            let mut countBeyondMinChain = 0u32;
            let mut i = tmpHashTable[hashIdx as usize];
            let mut count = 0u32;
            while i >= tmpMinChain && count < cacheSize {
                if i < minChain {
                    countBeyondMinChain = countBeyondMinChain.wrapping_add(1);
                }
                i = tmpChainTable[(i - tmpMinChain) as usize];
                count = count.wrapping_add(1);
            }
            if count == cacheSize {
                count = 0;
                while count < chainLimit {
                    if i < minChain {
                        if i == 0 || {
                            countBeyondMinChain = countBeyondMinChain.wrapping_add(1);
                            countBeyondMinChain > cacheSize
                        } {
                            break;
                        }
                    }
                    chainTable[chainPos as usize] = i;
                    chainPos = chainPos.wrapping_add(1);
                    count = count.wrapping_add(1);
                    if i < tmpMinChain {
                        break;
                    }
                    i = tmpChainTable[(i - tmpMinChain) as usize];
                }
            } else {
                count = 0;
            }
            tmpHashTable[hashIdx as usize] = if count != 0 {
                (chainPos.wrapping_sub(count) << 8).wrapping_add(count)
            } else {
                0
            };
        }
        debug_assert!(chainPos <= chainSize);
    }

    for hashIdx in (0..(1u32 << hashLog)).rev() {
        let bucketIdx = (hashIdx << ZSTD_LAZY_DDSS_BUCKET_LOG) as usize;
        let chainPackedPointer = hashTable[hashIdx as usize];
        for slot in &mut hashTable[bucketIdx..bucketIdx + cacheSize as usize] {
            *slot = 0;
        }
        hashTable[bucketIdx + bucketSize as usize - 1] = chainPackedPointer;
    }

    for idx in ms.nextToUpdate..target {
        let h = ((ZSTD_hashPtr(&base[idx as usize..], hashLog, ms.cParams.minMatch) as u32)
            << ZSTD_LAZY_DDSS_BUCKET_LOG) as usize;
        for i in (1..cacheSize as usize).rev() {
            hashTable[h + i] = hashTable[h + i - 1];
        }
        hashTable[h] = idx;
    }

    ms.nextToUpdate = target;
}

/// Port of `ZSTD_dedicatedDictSearch_lazy_search`.
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_dedicatedDictSearch_lazy_search(
    offsetPtr: &mut u32,
    mut ml: usize,
    nbAttempts: u32,
    dms: &ZSTD_MatchState_t,
    input_buf: &[u8],
    dict_buf: &[u8],
    ip: usize,
    iLimit: usize,
    prefixStart: usize,
    curr: u32,
    dictLimit: u32,
    ddsIdx: usize,
) -> usize {
    let ddsLowestIndex = dms.window.dictLimit;
    let ddsSize = dict_buf.len() as u32;
    let ddsIndexDelta = dictLimit.wrapping_sub(ddsSize);
    let bucketSize = 1u32 << ZSTD_LAZY_DDSS_BUCKET_LOG;
    let bucketLimit = nbAttempts.min(bucketSize - 1);

    for ddsAttempt in 0..bucketLimit {
        let matchIndex = dms.hashTable[ddsIdx + ddsAttempt as usize];
        if matchIndex == 0 {
            return ml;
        }
        debug_assert!(matchIndex >= ddsLowestIndex);
        let matchPos = matchIndex as usize;
        let mut currentMl = 0usize;
        if matchPos + 4 <= dict_buf.len()
            && MEM_read32(&dict_buf[matchPos..]) == MEM_read32(&input_buf[ip..])
        {
            currentMl = ZSTD_count_2segments(
                input_buf,
                ip + 4,
                iLimit,
                prefixStart,
                dict_buf,
                matchPos + 4,
                dict_buf.len(),
            ) + 4;
        }
        if currentMl > ml {
            ml = currentMl;
            *offsetPtr =
                OFFSET_TO_OFFBASE(curr.wrapping_sub(matchIndex.wrapping_add(ddsIndexDelta)));
            if ip + currentMl == iLimit {
                return ml;
            }
        }
    }

    let chainPackedPointer = dms.hashTable[ddsIdx + bucketSize as usize - 1];
    let mut chainIndex = (chainPackedPointer >> 8) as usize;
    let chainLength = chainPackedPointer & 0xFF;
    let chainAttempts = nbAttempts.saturating_sub(bucketLimit);
    let chainLimit = chainAttempts.min(chainLength);
    for _ in 0..chainLimit {
        let matchIndex = dms.chainTable[chainIndex];
        let matchPos = matchIndex as usize;
        let mut currentMl = 0usize;
        debug_assert!(matchIndex >= ddsLowestIndex);
        if matchPos + 4 <= dict_buf.len()
            && MEM_read32(&dict_buf[matchPos..]) == MEM_read32(&input_buf[ip..])
        {
            currentMl = ZSTD_count_2segments(
                input_buf,
                ip + 4,
                iLimit,
                prefixStart,
                dict_buf,
                matchPos + 4,
                dict_buf.len(),
            ) + 4;
        }
        if currentMl > ml {
            ml = currentMl;
            *offsetPtr =
                OFFSET_TO_OFFBASE(curr.wrapping_sub(matchIndex.wrapping_add(ddsIndexDelta)));
            if ip + currentMl == iLimit {
                break;
            }
        }
        chainIndex += 1;
    }
    ml
}

/// Port of `ZSTD_RowFindBestMatch`. Mirrors upstream's
/// `FORCE_INLINE_TEMPLATE`: callers (the lazy/greedy/lazy2 row-hash
/// matchers via `ZSTD_searchMax`) pass per-block-constant `mls` /
/// `dictMode` / `rowLog`, so inlining lets LLVM constant-fold the
/// per-byte runtime branches inside.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn ZSTD_RowFindBestMatch(
    ms: &mut ZSTD_MatchState_t,
    input_buf: &[u8],
    ext_dict_buf: Option<&[u8]>,
    ip: usize,
    iLimit: usize,
    offsetPtr: &mut u32,
    mls: u32,
    dictMode: ZSTD_dictMode_e,
    rowLog: u32,
) -> usize {
    let hashLog = ms.rowHashLog;
    let cParams = ms.cParams;
    let base_off = ms.window.base_offset;
    // Match upstream's `(U32)(ip - base)` — absolute window-space index.
    // This is the value we'll compare against `lowLimit` and store in the
    // hash table (`base_off + 0` for the very first byte avoids the "0 =
    // never inserted" sentinel collision the previous ip-relative
    // convention had).
    let curr = base_off.wrapping_add(ip as u32);
    let dictLimit = ms.window.dictLimit;
    // Convert dictLimit (absolute) to a slice offset for byte-side bounds.
    let prefixStart = dictLimit.saturating_sub(base_off) as usize;
    let maxDistance = 1u32 << cParams.windowLog;
    let lowestValid = ms.window.lowLimit;
    let withinMaxDistance = if curr.wrapping_sub(lowestValid) > maxDistance {
        curr.wrapping_sub(maxDistance)
    } else {
        lowestValid
    };
    let isDictionary = ms.loadedDictEnd != 0;
    let lowLimit = if isDictionary {
        lowestValid
    } else {
        withinMaxDistance
    };
    let rowEntries = 1u32 << rowLog;
    let rowMask = rowEntries - 1;
    let cappedSearchLog = cParams.searchLog.min(rowLog);
    let groupWidth = ZSTD_row_matchMaskGroupWidth(rowEntries);
    let hashSalt = ms.hashSalt;
    let mut nbAttempts = 1u32 << cappedSearchLog;
    let mut ml = 3usize;

    let mut ddsIdx = 0usize;
    let mut ddsExtraAttempts = 0u32;
    let mut dmsTag = 0u32;
    let mut dmsRelRow = 0usize;

    if dictMode == ZSTD_dictMode_e::ZSTD_dedicatedDictSearch {
        if let Some(dms) = ms.dictMatchState.as_deref() {
            let ddsHashLog = dms.cParams.hashLog - ZSTD_LAZY_DDSS_BUCKET_LOG;
            ddsIdx = (ZSTD_hashPtr(&input_buf[ip..], ddsHashLog, mls) as usize)
                << ZSTD_LAZY_DDSS_BUCKET_LOG;
            ddsExtraAttempts = if cParams.searchLog > rowLog {
                1u32 << (cParams.searchLog - rowLog)
            } else {
                0
            };
        }
    }

    if dictMode == ZSTD_dictMode_e::ZSTD_dictMatchState {
        if let Some(dms) = ms.dictMatchState.as_deref() {
            let dmsHash = ZSTD_hashPtr(
                &input_buf[ip..],
                dms.rowHashLog + ZSTD_ROW_HASH_TAG_BITS,
                mls,
            ) as u32;
            dmsRelRow = ((dmsHash >> ZSTD_ROW_HASH_TAG_BITS) << rowLog) as usize;
            dmsTag = dmsHash & ZSTD_ROW_HASH_TAG_MASK;
            ZSTD_row_prefetch(&dms.hashTable, &dms.tagTable, dmsRelRow as u32, rowLog);
        }
    }

    let hash = if ms.lazySkipping == 0 {
        ZSTD_row_update_internal(ms, ip, mls, rowLog, rowMask, true, input_buf);
        ZSTD_row_nextCachedHash(
            &mut ms.hashCache,
            &ms.hashTable,
            &ms.tagTable,
            input_buf,
            curr,
            base_off,
            hashLog,
            rowLog,
            mls,
            hashSalt,
        )
    } else {
        ms.nextToUpdate = curr;
        ZSTD_hashPtrSalted(
            &input_buf[ip..],
            hashLog + ZSTD_ROW_HASH_TAG_BITS,
            mls,
            hashSalt,
        ) as u32
    };
    ms.hashSaltEntropy = ms.hashSaltEntropy.wrapping_add(hash);

    {
        let relRow = ((hash >> ZSTD_ROW_HASH_TAG_BITS) << rowLog) as usize;
        let tag = (hash & ZSTD_ROW_HASH_TAG_MASK) as u8;
        let tag_row = unsafe {
            let ptr = ms.tagTable.as_ptr().add(relRow);
            std::slice::from_raw_parts(ptr, rowEntries as usize)
        };
        let headGrouped = unsafe { *tag_row.get_unchecked(0) } as u32 & rowMask;
        let matches = ZSTD_row_getMatchMask(tag_row, tag, headGrouped, rowEntries);
        // 4-byte dummy for the CMOV-friendly select-address pattern in
        // the candidate prefilter below. Reused across every candidate
        // so the load goes to a hot constant when bounds fail.
        static PREFILTER_DUMMY: [u8; 4] = [0x12, 0x34, 0x56, 0x78];
        // The no-dict row-hash path is the dominant L10 hot loop. It
        // still mirrors upstream's collect-then-insert-then-score order
        // so row state and candidate attempt consumption stay identical.
        if dictMode == ZSTD_dictMode_e::ZSTD_noDict {
            let hash_row = unsafe { ms.hashTable.as_ptr().add(relRow) };
            let mut matchBuffer: [MaybeUninit<u32>; ZSTD_ROW_HASH_MAX_ENTRIES as usize] =
                unsafe { MaybeUninit::uninit().assume_init() };
            let mut numMatches = 0usize;
            let mut mask = matches;

            while mask > 0 && nbAttempts > 0 {
                let matchPos = (headGrouped + ZSTD_VecMask_next(mask)) & rowMask;
                if matchPos != 0 {
                    let matchIndex = unsafe { *hash_row.add(matchPos as usize) };
                    if matchIndex < lowLimit {
                        break;
                    }
                    let off = matchIndex.wrapping_sub(base_off) as usize;
                    if off < input_buf.len() {
                        crate::common::zstd_internal::prefetchSliceByte(input_buf, off);
                    }
                    unsafe {
                        matchBuffer.get_unchecked_mut(numMatches).write(matchIndex);
                    }
                    numMatches += 1;
                    nbAttempts -= 1;
                }
                mask &= mask - 1;
            }

            let pos = unsafe {
                let head = ms.tagTable.get_unchecked_mut(relRow);
                let mut next = (*head as u32).wrapping_sub(1) & rowMask;
                next += ((next == 0) as u32) * rowMask;
                *head = next as u8;
                next as usize
            };
            unsafe {
                *ms.tagTable.get_unchecked_mut(relRow + pos) = tag;
                *ms.hashTable.get_unchecked_mut(relRow + pos) = ms.nextToUpdate;
            }
            ms.nextToUpdate = ms.nextToUpdate.wrapping_add(1);

            for match_num in 0..numMatches {
                let matchIndex = unsafe { matchBuffer.get_unchecked(match_num).assume_init() };
                let matchBytePos = matchIndex.wrapping_sub(base_off) as usize;
                let prefilter_valid = matchBytePos + ml < input_buf.len() && ip + ml < iLimit;
                let m_ptr = if prefilter_valid {
                    input_buf.as_ptr().wrapping_add(matchBytePos + ml - 3)
                } else {
                    PREFILTER_DUMMY.as_ptr()
                };
                let c_ptr = if prefilter_valid {
                    input_buf.as_ptr().wrapping_add(ip + ml - 3)
                } else {
                    PREFILTER_DUMMY.as_ptr()
                };
                let m_val = unsafe { (m_ptr as *const u32).read_unaligned() };
                let c_val = unsafe { (c_ptr as *const u32).read_unaligned() };
                if prefilter_valid & (m_val == c_val) {
                    let currentMl = ZSTD_count(input_buf, ip, matchBytePos, iLimit);
                    if currentMl > ml {
                        ml = currentMl;
                        *offsetPtr = OFFSET_TO_OFFBASE(curr.wrapping_sub(matchIndex));
                        if ip + currentMl == iLimit {
                            break;
                        }
                    }
                }
            }
            return ml;
        }

        let mut matchBuffer: [MaybeUninit<u32>; ZSTD_ROW_HASH_MAX_ENTRIES as usize] =
            unsafe { MaybeUninit::uninit().assume_init() };
        let mut numMatches = 0usize;
        let mut mask = matches;
        // Hoist the row sub-slice and `dictBase_offset` read out of the
        // candidate loop so each iteration's hashTable access is a
        // bounded local index (skips the per-iter `ms.hashTable.len()`
        // bounds check) and we avoid re-reading `ms.window` for every
        // ext-dict candidate.
        let dict_base_off = ms.window.dictBase_offset;
        let hash_row = &ms.hashTable[relRow..relRow + rowEntries as usize];

        while mask > 0 && nbAttempts > 0 {
            let matchPos = (headGrouped + ZSTD_VecMask_next(mask)) & rowMask;
            let matchIndex = hash_row[matchPos as usize];
            if matchPos != 0 {
                if matchIndex < lowLimit {
                    break;
                }
                // Mirror upstream's `PREFETCH_L1(base + matchIndex)` from
                // zstd_lazy.c:1281 — issue a real cache-line hint for
                // the candidate before it's consumed below.
                if dictMode != ZSTD_dictMode_e::ZSTD_extDict || matchIndex >= dictLimit {
                    // Absolute matchIndex → slice offset.
                    let off = matchIndex.wrapping_sub(base_off) as usize;
                    if off < input_buf.len() {
                        crate::common::zstd_internal::prefetchSliceByte(input_buf, off);
                    }
                } else if let Some(dict_buf) = ext_dict_buf {
                    let off = matchIndex.wrapping_sub(dict_base_off) as usize;
                    if off < dict_buf.len() {
                        crate::common::zstd_internal::prefetchSliceByte(dict_buf, off);
                    }
                }
                unsafe {
                    matchBuffer.get_unchecked_mut(numMatches).write(matchIndex);
                }
                numMatches += 1;
                nbAttempts -= 1;
            }
            mask &= mask - 1;
        }
        // End of immutable borrow on ms.hashTable.
        let _ = hash_row;

        // SAFETY: `relRow + (1 << rowLog) ≤ tagTable.len()` by table
        // sizing (same invariant as in `ZSTD_row_update_internalImpl`).
        // `pos < rowEntries`. Inline `ZSTD_row_nextIndex` accessing
        // only `tagRow[0]` so we can target a single byte.
        let pos = unsafe {
            let head = ms.tagTable.get_unchecked_mut(relRow);
            let mut next = (*head as u32).wrapping_sub(1) & rowMask;
            next += ((next == 0) as u32) * rowMask;
            *head = next as u8;
            next as usize
        };
        unsafe {
            *ms.tagTable.get_unchecked_mut(relRow + pos) = tag;
            *ms.hashTable.get_unchecked_mut(relRow + pos) = ms.nextToUpdate;
        }
        ms.nextToUpdate = ms.nextToUpdate.wrapping_add(1);

        for match_num in 0..numMatches {
            let matchIndex = unsafe { matchBuffer.get_unchecked(match_num).assume_init() };
            let mut currentMl = 0usize;
            if dictMode != ZSTD_dictMode_e::ZSTD_extDict || matchIndex >= dictLimit {
                // Absolute matchIndex → slice offset.
                let matchPos = matchIndex.wrapping_sub(base_off) as usize;
                // `ml` is seeded at 3 and grows monotonically, so
                // `matchPos + ml >= 3` always — no underflow protection
                // needed. The bounds checks remain because matchIndex
                // can be from a prior block (saturating_sub won't catch
                // wraparound past `iLimit`).
                let prefilter_valid = matchPos + ml < input_buf.len() && ip + ml < iLimit;
                // CMOV-friendly select: read from real `matchPos+ml-3`
                // when valid, else from a dummy. Mirrors upstream's
                // `ZSTD_selectAddr`-style pointer pick. Slice-indexed
                // reads would emit per-call `cmp/jb` bounds checks that
                // defeat CMOV codegen on this hot inner prefilter.
                let m_ptr = if prefilter_valid {
                    input_buf.as_ptr().wrapping_add(matchPos + ml - 3)
                } else {
                    PREFILTER_DUMMY.as_ptr()
                };
                let c_ptr = if prefilter_valid {
                    input_buf.as_ptr().wrapping_add(ip + ml - 3)
                } else {
                    PREFILTER_DUMMY.as_ptr()
                };
                // SAFETY: when `prefilter_valid`, the loop invariants
                // give `matchPos + ml + 1 ≤ input_buf.len()` and
                // `ip + ml + 1 ≤ iLimit ≤ input_buf.len()`, so both
                // 4-byte reads are in-range. Otherwise we read from
                // `PREFILTER_DUMMY` (4 bytes valid).
                let m_val = unsafe { (m_ptr as *const u32).read_unaligned() };
                let c_val = unsafe { (c_ptr as *const u32).read_unaligned() };
                if prefilter_valid & (m_val == c_val) {
                    currentMl = ZSTD_count(input_buf, ip, matchPos, iLimit);
                }
            } else if let Some(dict_buf) = ext_dict_buf {
                // ext-dict path: the matchIndex still lives in the local
                // matchState's window-index space; the dict_buf however
                // is laid out so byte at absolute index `i` lives at
                // `dict_buf[i - dict_base_off]`. Reuse the per-matchstate
                // `dictBase_offset` for the conversion.
                let dict_base_off = ms.window.dictBase_offset;
                let matchPos = matchIndex.wrapping_sub(dict_base_off) as usize;
                if matchPos + 4 <= dict_buf.len()
                    && MEM_read32(&dict_buf[matchPos..]) == MEM_read32(&input_buf[ip..])
                {
                    currentMl = ZSTD_count_2segments(
                        input_buf,
                        ip + 4,
                        iLimit,
                        prefixStart,
                        dict_buf,
                        matchPos + 4,
                        dict_buf.len(),
                    ) + 4;
                }
            }
            if currentMl > ml {
                ml = currentMl;
                *offsetPtr = OFFSET_TO_OFFBASE(curr.wrapping_sub(matchIndex));
                if ip + currentMl == iLimit {
                    break;
                }
            }
        }
    }

    if dictMode == ZSTD_dictMode_e::ZSTD_dedicatedDictSearch {
        if let (Some(dms), Some(dict_buf)) = (ms.dictMatchState.as_deref(), ext_dict_buf) {
            ml = ZSTD_dedicatedDictSearch_lazy_search(
                offsetPtr,
                ml,
                nbAttempts + ddsExtraAttempts,
                dms,
                input_buf,
                dict_buf,
                ip,
                iLimit,
                prefixStart,
                curr,
                dictLimit,
                ddsIdx,
            );
        }
    } else if dictMode == ZSTD_dictMode_e::ZSTD_dictMatchState {
        if let (Some(dms), Some(dict_buf)) = (ms.dictMatchState.as_deref(), ext_dict_buf) {
            let dmsLowestIndex = dms.window.dictLimit;
            let dmsIndexDelta = dictLimit.wrapping_sub(dms.window.nextSrc);
            let headGrouped = (dms.tagTable[dmsRelRow] as u32 & rowMask) * groupWidth;
            let matches = ZSTD_row_getMatchMask(
                &dms.tagTable[dmsRelRow..dmsRelRow + rowEntries as usize],
                dmsTag as u8,
                headGrouped,
                rowEntries,
            );
            let mut matchBuffer: [MaybeUninit<u32>; ZSTD_ROW_HASH_MAX_ENTRIES as usize] =
                unsafe { MaybeUninit::uninit().assume_init() };
            let mut numMatches = 0usize;
            let mut mask = matches;
            while mask > 0 && nbAttempts > 0 {
                let matchPos = ((headGrouped + ZSTD_VecMask_next(mask)) / groupWidth) & rowMask;
                let matchIndex = dms.hashTable[dmsRelRow + matchPos as usize];
                if matchPos != 0 {
                    if matchIndex < dmsLowestIndex {
                        break;
                    }
                    unsafe {
                        matchBuffer.get_unchecked_mut(numMatches).write(matchIndex);
                    }
                    numMatches += 1;
                    nbAttempts -= 1;
                }
                mask &= mask - 1;
            }
            for match_num in 0..numMatches {
                let matchIndex = unsafe { matchBuffer.get_unchecked(match_num).assume_init() };
                let matchPos = matchIndex.saturating_sub(dms.window.base_offset) as usize;
                let mut currentMl = 0usize;
                if matchPos + 4 <= dict_buf.len()
                    && MEM_read32(&dict_buf[matchPos..]) == MEM_read32(&input_buf[ip..])
                {
                    currentMl = ZSTD_count_2segments(
                        input_buf,
                        ip + 4,
                        iLimit,
                        prefixStart,
                        dict_buf,
                        matchPos + 4,
                        dict_buf.len(),
                    ) + 4;
                }
                if currentMl > ml {
                    ml = currentMl;
                    *offsetPtr = OFFSET_TO_OFFBASE(
                        curr.wrapping_sub(matchIndex.wrapping_add(dmsIndexDelta)),
                    );
                    if ip + currentMl == iLimit {
                        break;
                    }
                }
            }
        }
    }
    ml
}

/// Port of `ZSTD_searchMax`. `FORCE_INLINE_TEMPLATE` upstream — the
/// dispatch on `searchMethod` / `dictMode` is fully constant-folded
/// when this is inlined into each templated public entry.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn ZSTD_searchMax(
    ms: &mut ZSTD_MatchState_t,
    input_buf: &[u8],
    ext_dict_buf: Option<&[u8]>,
    ip: usize,
    iend: usize,
    offsetPtr: &mut u32,
    mls: u32,
    rowLog: u32,
    searchMethod: searchMethod_e,
    dictMode: ZSTD_dictMode_e,
) -> usize {
    match searchMethod {
        searchMethod_e::search_hashChain => {
            ZSTD_HcFindBestMatch(ms, input_buf, ip, iend, offsetPtr, mls, dictMode)
        }
        searchMethod_e::search_binaryTree => match dictMode {
            ZSTD_dictMode_e::ZSTD_dedicatedDictSearch => 0,
            _ => ZSTD_BtFindBestMatch(ms, input_buf, ip, iend, offsetPtr, mls, dictMode),
        },
        searchMethod_e::search_rowHash => ZSTD_RowFindBestMatch(
            ms,
            input_buf,
            ext_dict_buf,
            ip,
            iend,
            offsetPtr,
            mls,
            dictMode,
            rowLog,
        ),
    }
}

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
    lazySkipping: u32,
) -> u32 {
    let hashLog = ms.cParams.hashLog;
    let chainMask = (1u32 << ms.cParams.chainLog) - 1;
    let base_off = ms.window.base_offset;
    let target: u32 = base_off.wrapping_add(ip as u32);
    // Ensure chainTable sized.
    let chainSize = 1usize << ms.cParams.chainLog;
    if ms.chainTable.len() < chainSize {
        ms.chainTable.resize(chainSize, 0);
    }
    let mut idx = ms.nextToUpdate;
    while idx < target {
        let rel = idx.saturating_sub(base_off) as usize;
        if rel + 8 > src.len() {
            break;
        }
        let h = ZSTD_hashPtr(&src[rel..], hashLog, mls);
        let slot = (idx & chainMask) as usize;
        ms.chainTable[slot] = ms.hashTable[h];
        ms.hashTable[h] = idx;
        idx += 1;
        if lazySkipping != 0 {
            break;
        }
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
#[inline]
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
    let curr = base_off.wrapping_add(ip as u32);
    let maxDistance = 1u32 << ms.cParams.windowLog;
    let lowestValid = ms.window.lowLimit;
    let withinMaxDistance = if curr.wrapping_sub(lowestValid) > maxDistance {
        curr.wrapping_sub(maxDistance)
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
    let mut matchIndex = ZSTD_insertAndFindFirstIndex_internal(ms, src, ip, mls, ms.lazySkipping);

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
        // Cheap pre-check, identical to upstream's
        // `ZSTD_HcFindBestMatch` HC4 inner loop:
        //     if (MEM_read32(match + ml - 3) == MEM_read32(ip + ml - 3))
        //         currentMl = ZSTD_count(ip, match, iLimit);
        // For the seed `ml = 3`, `ml - 3 == 0`, so this collapses to a
        // head-of-match u32 compare; for longer best-so-far it acts as
        // a "tail must match" filter that rejects shorter candidates
        // before the full byte-by-byte count.
        let tail_off = ml.wrapping_sub(3);
        let bytes_ok = match_pos + tail_off + 4 <= src.len() && ip + tail_off + 4 <= iLimit;
        let currentMl = if bytes_ok
            && MEM_read32(&src[match_pos + tail_off..]) == MEM_read32(&src[ip + tail_off..])
        {
            ZSTD_count(src, ip, match_pos, iLimit)
        } else {
            0
        };

        if currentMl > ml {
            ml = currentMl;
            *offBase_out = OFFSET_TO_OFFBASE(curr.wrapping_sub(matchIndex));
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
    ml
}

/// Port of `ZSTD_HcFindBestMatch`. Hash-chain match finder covering
/// all dict modes (`noDict`, `extDict`, `dictMatchState`,
/// `dedicatedDictSearch`); the `noDict` fast path is split out into
/// `ZSTD_HcFindBestMatch_noDict`.
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_HcFindBestMatch(
    ms: &mut ZSTD_MatchState_t,
    src: &[u8],
    ip: usize,
    iLimit: usize,
    offBase_out: &mut u32,
    mls: u32,
    dictMode: ZSTD_dictMode_e,
) -> usize {
    if dictMode == ZSTD_dictMode_e::ZSTD_noDict {
        return ZSTD_HcFindBestMatch_noDict(ms, src, ip, iLimit, offBase_out, mls);
    }

    let cParams = ms.cParams;
    let chainSize = 1u32 << cParams.chainLog;
    let chainMask = chainSize - 1;
    let base_off = ms.window.base_offset;
    let dict_base_off = ms.window.dictBase_offset;
    let dictLimit = ms.window.dictLimit;
    let prefixStart = dictLimit.saturating_sub(base_off) as usize;
    let dictEnd = ms.dictContent.len();
    let curr = base_off.wrapping_add(ip as u32);
    let maxDistance = 1u32 << cParams.windowLog;
    let lowestValid = ms.window.lowLimit;
    let withinMaxDistance = if curr.wrapping_sub(lowestValid) > maxDistance {
        curr.wrapping_sub(maxDistance)
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
    let mut nbAttempts = 1u32 << cParams.searchLog;
    let mut ml = 3usize;

    let ddsIdx = if dictMode == ZSTD_dictMode_e::ZSTD_dedicatedDictSearch {
        let dms = match ms.dictMatchState.as_deref() {
            Some(dms) => dms,
            None => return 0,
        };
        let ddsHashLog = dms.cParams.hashLog - ZSTD_LAZY_DDSS_BUCKET_LOG;
        (ZSTD_hashPtr(&src[ip..], ddsHashLog, mls) << ZSTD_LAZY_DDSS_BUCKET_LOG) as usize
    } else {
        0usize
    };

    let mut matchIndex = ZSTD_insertAndFindFirstIndex_internal(ms, src, ip, mls, ms.lazySkipping);

    while matchIndex >= lowLimit && nbAttempts > 0 {
        nbAttempts -= 1;
        let currentMl = if dictMode != ZSTD_dictMode_e::ZSTD_extDict || matchIndex >= dictLimit {
            let match_pos = matchIndex.saturating_sub(base_off) as usize;
            if match_pos + ml < src.len()
                && ip + ml <= iLimit
                && MEM_read32(&src[match_pos + ml - 3..]) == MEM_read32(&src[ip + ml - 3..])
            {
                ZSTD_count(src, ip, match_pos, iLimit)
            } else {
                0
            }
        } else {
            let match_pos = matchIndex.wrapping_sub(dict_base_off) as usize;
            if match_pos + 4 <= dictEnd
                && MEM_read32(&ms.dictContent[match_pos..]) == MEM_read32(&src[ip..])
            {
                ZSTD_count_2segments(
                    src,
                    ip + 4,
                    iLimit,
                    prefixStart,
                    &ms.dictContent,
                    match_pos + 4,
                    dictEnd,
                ) + 4
            } else {
                0
            }
        };

        if currentMl > ml {
            ml = currentMl;
            *offBase_out = OFFSET_TO_OFFBASE(curr.wrapping_sub(matchIndex));
            if ip + currentMl == iLimit {
                break;
            }
        }

        if matchIndex <= minChain {
            break;
        }
        matchIndex = ms.chainTable[(matchIndex & chainMask) as usize];
    }

    if dictMode == ZSTD_dictMode_e::ZSTD_dedicatedDictSearch {
        if let Some(dms) = ms.dictMatchState.as_deref() {
            ml = ZSTD_dedicatedDictSearch_lazy_search(
                offBase_out,
                ml,
                nbAttempts,
                dms,
                src,
                &dms.dictContent,
                ip,
                iLimit,
                prefixStart,
                curr,
                dictLimit,
                ddsIdx,
            );
        }
    } else if dictMode == ZSTD_dictMode_e::ZSTD_dictMatchState {
        if let Some(dms) = ms.dictMatchState.as_deref() {
            let dmsChainSize = 1u32 << dms.cParams.chainLog;
            let dmsChainMask = dmsChainSize - 1;
            let dmsLowestIndex = dms.window.dictLimit;
            let dmsEnd = dms.window.nextSrc;
            let dmsBaseOff = dms.window.base_offset;
            let dmsIndexDelta = dictLimit.wrapping_sub(dmsEnd);
            let dmsMinChain = if dmsEnd > dmsChainSize {
                dmsEnd.wrapping_sub(dmsChainSize)
            } else {
                0
            };
            matchIndex = dms.hashTable[ZSTD_hashPtr(&src[ip..], dms.cParams.hashLog, mls)];

            while matchIndex >= dmsLowestIndex && nbAttempts > 0 {
                nbAttempts -= 1;
                let match_pos = matchIndex.saturating_sub(dmsBaseOff) as usize;
                let currentMl = if match_pos + 4 <= dms.dictContent.len()
                    && MEM_read32(&dms.dictContent[match_pos..]) == MEM_read32(&src[ip..])
                {
                    ZSTD_count_2segments(
                        src,
                        ip + 4,
                        iLimit,
                        prefixStart,
                        &dms.dictContent,
                        match_pos + 4,
                        dms.dictContent.len(),
                    ) + 4
                } else {
                    0
                };

                if currentMl > ml {
                    ml = currentMl;
                    *offBase_out = OFFSET_TO_OFFBASE(
                        curr.wrapping_sub(matchIndex.wrapping_add(dmsIndexDelta)),
                    );
                    if ip + currentMl == iLimit {
                        break;
                    }
                }

                if matchIndex <= dmsMinChain {
                    break;
                }
                matchIndex = dms.chainTable[(matchIndex & dmsChainMask) as usize];
            }
        }
    }

    ml
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
    let endIndex = base_off.wrapping_add(srcSize as u32);
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
        let curr = base_off.wrapping_add(ip as u32);
        let windowLow = crate::compress::zstd_fast::ZSTD_getLowestPrefixIndex(ms, curr, windowLog);
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
    while ip < ilimit {
        let mut matchLength: usize = 0;
        let mut offBase: u32 = REPCODE_TO_OFFBASE(1);
        let mut start = ip + 1;
        let mut depth0_rep_match = false;

        // 1. Repcode at ip+1 (depth-0 baseline).
        if rep_offset1 > 0
            && ip + 1 + 4 <= iend
            && ip + 1 >= rep_offset1 as usize
            && MEM_read32(&src[ip + 1..]) == MEM_read32(&src[ip + 1 - rep_offset1 as usize..])
        {
            matchLength = ZSTD_count(src, ip + 1 + 4, ip + 1 + 4 - rep_offset1 as usize, iend) + 4;
            depth0_rep_match = depth == 0;
        }

        // 2. Chain-search at ip (depth-0 candidate).
        if !depth0_rep_match {
            let mut cand_off: u32 = 999_999_999;
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
            ms.lazySkipping = u32::from(step > kLazySkippingStep);
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
                    && MEM_read32(&src[probe..]) == MEM_read32(&src[probe - rep_offset1 as usize..])
                {
                    let mlRep =
                        ZSTD_count(src, probe + 4, probe + 4 - rep_offset1 as usize, iend) + 4;
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
                let mut cand_off: u32 = 999_999_999;
                let ml2 = ZSTD_HcFindBestMatch_noDict(ms, src, probe, iend, &mut cand_off, mls);
                if ml2 >= 4 && cand_off != 0 {
                    let gain2 =
                        (ml2 as i32) * 4 - crate::common::bits::ZSTD_highbit32(cand_off) as i32;
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
                        && MEM_read32(&src[p2..]) == MEM_read32(&src[p2 - rep_offset1 as usize..])
                    {
                        let mlRep =
                            ZSTD_count(src, p2 + 4, p2 + 4 - rep_offset1 as usize, iend) + 4;
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
                    let mut cand2_off: u32 = 999_999_999;
                    let ml3 = ZSTD_HcFindBestMatch_noDict(ms, src, p2, iend, &mut cand2_off, mls);
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
        ZSTD_storeSeq(seqStore, litLength, &src[anchor..], offBase, matchLength);
        ip = start + matchLength;
        anchor = ip;

        if ms.lazySkipping != 0 {
            ms.lazySkipping = 0;
        }

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

/// Rust-only helper: the binary-tree/row-hash slow-search counterpart
/// to `ZSTD_compressBlock_lazy_noDict_generic`. Used by btlazy2 and
/// the row-hash with-istart paths; `searchMethod` must not be
/// `search_hashChain`.
#[cfg(test)]
fn ZSTD_compressBlock_lazy_noDict_generic_search(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
    istart: usize,
    depth: u32,
    searchMethod: searchMethod_e,
) -> usize {
    debug_assert!(searchMethod != searchMethod_e::search_hashChain);

    let mls = ms.cParams.minMatch.clamp(4, 6);
    let windowLog = ms.cParams.windowLog;
    let base_off = ms.window.base_offset;
    let srcSize = src.len();
    let endIndex = base_off.wrapping_add(srcSize as u32);
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
        let curr = base_off.wrapping_add(ip as u32);
        let windowLow = crate::compress::zstd_fast::ZSTD_getLowestPrefixIndex(ms, curr, windowLog);
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
    while ip < ilimit {
        let mut matchLength: usize = 0;
        let mut offBase: u32 = REPCODE_TO_OFFBASE(1);
        let mut start = ip + 1;
        let mut depth0_rep_match = false;

        if rep_offset1 > 0
            && ip + 1 + 4 <= iend
            && ip + 1 >= rep_offset1 as usize
            && MEM_read32(&src[ip + 1..]) == MEM_read32(&src[ip + 1 - rep_offset1 as usize..])
        {
            matchLength = ZSTD_count(src, ip + 1 + 4, ip + 1 + 4 - rep_offset1 as usize, iend) + 4;
            depth0_rep_match = depth == 0;
        }

        if !depth0_rep_match {
            let mut cand_off: u32 = 999_999_999;
            let ml2 = match searchMethod {
                searchMethod_e::search_binaryTree => ZSTD_BtFindBestMatch(
                    ms,
                    src,
                    ip,
                    iend,
                    &mut cand_off,
                    mls,
                    ZSTD_dictMode_e::ZSTD_noDict,
                ),
                searchMethod_e::search_hashChain | searchMethod_e::search_rowHash => unreachable!(),
            };
            if ml2 > matchLength {
                matchLength = ml2;
                offBase = cand_off;
                start = ip;
            }
        }

        if matchLength < 4 {
            let step = ((ip - anchor) >> kSearchStrength) + 1;
            ip += step;
            ms.lazySkipping = u32::from(step > kLazySkippingStep);
            continue;
        }

        if depth >= 1 {
            let mut probe = ip + 1;
            while probe < ilimit {
                if rep_offset1 > 0
                    && probe + 4 <= iend
                    && probe >= rep_offset1 as usize
                    && MEM_read32(&src[probe..]) == MEM_read32(&src[probe - rep_offset1 as usize..])
                {
                    let mlRep =
                        ZSTD_count(src, probe + 4, probe + 4 - rep_offset1 as usize, iend) + 4;
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
                let mut cand_off: u32 = 999_999_999;
                let ml2 = match searchMethod {
                    searchMethod_e::search_binaryTree => ZSTD_BtFindBestMatch(
                        ms,
                        src,
                        probe,
                        iend,
                        &mut cand_off,
                        mls,
                        ZSTD_dictMode_e::ZSTD_noDict,
                    ),
                    searchMethod_e::search_hashChain | searchMethod_e::search_rowHash => {
                        unreachable!()
                    }
                };
                if ml2 >= 4 && cand_off != 0 {
                    let gain2 =
                        (ml2 as i32) * 4 - crate::common::bits::ZSTD_highbit32(cand_off) as i32;
                    let gain1 = (matchLength as i32) * 4
                        - crate::common::bits::ZSTD_highbit32(offBase) as i32
                        + 4;
                    if gain2 > gain1 {
                        matchLength = ml2;
                        offBase = cand_off;
                        start = probe;
                        if depth >= 2 {
                            probe += 1;
                            continue;
                        }
                    }
                }
                if depth == 2 && probe + 1 < ilimit {
                    let p2 = probe + 1;
                    if rep_offset1 > 0
                        && p2 + 4 <= iend
                        && p2 >= rep_offset1 as usize
                        && MEM_read32(&src[p2..]) == MEM_read32(&src[p2 - rep_offset1 as usize..])
                    {
                        let mlRep =
                            ZSTD_count(src, p2 + 4, p2 + 4 - rep_offset1 as usize, iend) + 4;
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
                    let mut cand2_off: u32 = 999_999_999;
                    let ml3 = match searchMethod {
                        searchMethod_e::search_binaryTree => ZSTD_BtFindBestMatch(
                            ms,
                            src,
                            p2,
                            iend,
                            &mut cand2_off,
                            mls,
                            ZSTD_dictMode_e::ZSTD_noDict,
                        ),
                        searchMethod_e::search_hashChain | searchMethod_e::search_rowHash => {
                            unreachable!()
                        }
                    };
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
                break;
            }
        }

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

        let litLength = start - anchor;
        ZSTD_storeSeq(seqStore, litLength, &src[anchor..], offBase, matchLength);
        ip = start + matchLength;
        anchor = ip;

        if ms.lazySkipping != 0 {
            ms.lazySkipping = 0;
        }

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

/// Mirrors upstream's `FORCE_INLINE_TEMPLATE` declaration on
/// `ZSTD_compressBlock_lazy_generic`: each public entry (greedy_row,
/// lazy_row, lazy2_row, ext_*, dictMatchState_*, etc.) passes literal
/// constants for `searchMethod` / `depth` / `dictMode`. With
/// `#[inline(always)]`, LLVM inlines this body into each public entry
/// and constant-propagates those values, eliminating the per-byte
/// runtime branches on `if dictMode == …` / `if searchMethod == …`.
/// Without this attribute the function stays as one runtime-dispatched
/// blob (~8KB) and every per-byte iteration pays for those branches.
#[inline(always)]
pub fn ZSTD_compressBlock_lazy_generic(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
    searchMethod: searchMethod_e,
    depth: u32,
    dictMode: ZSTD_dictMode_e,
) -> usize {
    ZSTD_compressBlock_lazy_generic_with_istart(
        ms,
        seqStore,
        rep,
        src,
        0,
        searchMethod,
        depth,
        dictMode,
    )
}

/// Generic lazy/row-hash matcher with a `istart` (first byte to match
/// from). Bytes in `src[..istart]` are treated as cross-block history
/// (matchable but not parsed for new sequences). Mirrors upstream's
/// `ZSTD_compressBlock_lazy_generic` template, which uses
/// `ip = ms->window.base + istart` implicitly through the
/// `(BYTE const*)src` parameter; in our slice-based encoding we make
/// `istart` an explicit parameter so callers like
/// `ZSTD_compressBlock_lazy_with_history` can avoid trimming `src` to
/// the current block (the trim would break the
/// `base_offset + ip = abs_index` invariant once base_offset is anchored
/// to the window start, not the block start).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn ZSTD_compressBlock_lazy_generic_with_istart(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
    istart: usize,
    searchMethod: searchMethod_e,
    depth: u32,
    dictMode: ZSTD_dictMode_e,
) -> usize {
    let iend = src.len();
    let rowLog = ms.cParams.searchLog.clamp(4, 6);
    let ilimit = if searchMethod == searchMethod_e::search_rowHash {
        iend.saturating_sub(8 + ZSTD_ROW_HASH_CACHE_SIZE)
    } else {
        iend.saturating_sub(8)
    };
    let base_off = ms.window.base_offset;
    let prefixLowestIndex = ms.window.dictLimit;
    let prefixLowest = prefixLowestIndex.saturating_sub(base_off) as usize;
    let mls = ms.cParams.minMatch.clamp(4, 6);

    let isDMS = dictMode == ZSTD_dictMode_e::ZSTD_dictMatchState;
    let isDDS = dictMode == ZSTD_dictMode_e::ZSTD_dedicatedDictSearch;
    let isDxS = isDMS || isDDS;
    let (dictLowestIndex, dms_base_offset, dictIndexDelta, dictAndPrefixLength, dictBase) = if isDxS
    {
        let dms = match ms.dictMatchState.as_deref() {
            Some(dms) => dms,
            None => return src.len(),
        };
        let dictLowestIndex = dms.window.dictLimit;
        let dictEndIndex = dms.window.nextSrc;
        let dictBase = dms.dictContent.clone();
        let dictIndexDelta = prefixLowestIndex.wrapping_sub(dictEndIndex);
        let curr = base_off.wrapping_add(istart as u32);
        let dictAndPrefixLength = curr
            .saturating_sub(prefixLowestIndex)
            .wrapping_add(dictEndIndex.saturating_sub(dictLowestIndex));
        (
            dictLowestIndex,
            dms.window.base_offset,
            dictIndexDelta,
            dictAndPrefixLength,
            dictBase,
        )
    } else {
        (0, 0, 0, 0, Vec::new())
    };
    let dictLowest = if isDxS {
        dictLowestIndex.saturating_sub(dms_base_offset) as usize
    } else {
        0
    };
    let extDictBaseOffset = ms.window.dictBase_offset;
    let extDictLowest = if dictMode == ZSTD_dictMode_e::ZSTD_extDict {
        ms.window.lowLimit.saturating_sub(extDictBaseOffset) as usize
    } else {
        0
    };
    let extDict = if dictMode == ZSTD_dictMode_e::ZSTD_extDict {
        ms.dictContent.clone()
    } else {
        Vec::new()
    };

    // Start matching at `istart`; bytes before that are cross-block
    // history. For a fresh window with no dict prefix, upstream skips
    // byte 0 to avoid colliding with the "uninitialized" sentinel; we
    // preserve that only at the very start of compression (istart == 0).
    let mut ip = if istart == 0 {
        usize::from(dictAndPrefixLength == 0)
    } else {
        istart
    };
    let mut anchor = istart;
    let mut offset_1 = rep[0];
    let mut offset_2 = rep[1];
    let mut offsetSaved1 = 0u32;
    let mut offsetSaved2 = 0u32;

    if dictMode == ZSTD_dictMode_e::ZSTD_noDict {
        let curr = base_off.wrapping_add(ip as u32);
        let windowLow = ZSTD_getLowestMatchIndex(ms, curr, ms.cParams.windowLog);
        let maxRep = curr.wrapping_sub(windowLow);
        if offset_2 > maxRep {
            offsetSaved2 = offset_2;
            offset_2 = 0;
        }
        if offset_1 > maxRep {
            offsetSaved1 = offset_1;
            offset_1 = 0;
        }
    }
    if isDxS {
        if offset_1 > dictAndPrefixLength || offset_2 > dictAndPrefixLength {
            return src.len();
        }
    }

    ms.lazySkipping = 0;
    if searchMethod == searchMethod_e::search_rowHash {
        // `fillHashCache` expects an absolute index now (matches
        // upstream's `ms->nextToUpdate` convention); the conversion to
        // slice frame happens inside the helper.
        ZSTD_row_fillHashCache(ms, src, rowLog, mls, ms.nextToUpdate, ilimit);
    }

    while ip < ilimit {
        let mut matchLength = 0usize;
        let mut offBase = REPCODE_TO_OFFBASE(1);
        let mut start = ip + 1;
        let mut depth0_rep_match = false;

        if isDxS {
            let repIndex = base_off
                .wrapping_add(ip as u32)
                .wrapping_add(1)
                .wrapping_sub(offset_1);
            let repInDict = repIndex < prefixLowestIndex;
            let repMatch = if repInDict {
                repIndex
                    .saturating_sub(dictIndexDelta)
                    .saturating_sub(dms_base_offset) as usize
            } else {
                repIndex.saturating_sub(base_off) as usize
            };
            if ZSTD_index_overlap_check(prefixLowestIndex, repIndex)
                && if repInDict {
                    repMatch + 4 <= dictBase.len()
                        && MEM_read32(&dictBase[repMatch..]) == MEM_read32(&src[ip + 1..])
                } else {
                    repMatch + 4 <= src.len()
                        && MEM_read32(&src[repMatch..]) == MEM_read32(&src[ip + 1..])
                }
            {
                let repMatchEnd = if repInDict { dictBase.len() } else { iend };
                matchLength = ZSTD_count_2segments(
                    src,
                    ip + 1 + 4,
                    iend,
                    prefixLowest,
                    if repInDict { dictBase.as_slice() } else { src },
                    repMatch + 4,
                    repMatchEnd,
                ) + 4;
                depth0_rep_match = depth == 0;
            }
        } else if dictMode == ZSTD_dictMode_e::ZSTD_extDict {
            let curr = base_off.wrapping_add(ip as u32);
            let windowLow =
                ZSTD_getLowestMatchIndex(ms, curr.wrapping_add(1), ms.cParams.windowLog);
            let repIndex = curr.wrapping_add(1).wrapping_sub(offset_1);
            let repInDict = repIndex < prefixLowestIndex;
            let repMatch = if repInDict {
                repIndex.saturating_sub(extDictBaseOffset) as usize
            } else {
                repIndex.saturating_sub(base_off) as usize
            };
            if ZSTD_index_overlap_check(prefixLowestIndex, repIndex)
                && offset_1 <= curr.wrapping_add(1).wrapping_sub(windowLow)
                && if repInDict {
                    repMatch + 4 <= extDict.len()
                        && MEM_read32(&extDict[repMatch..]) == MEM_read32(&src[ip + 1..])
                } else {
                    repMatch + 4 <= src.len()
                        && MEM_read32(&src[repMatch..]) == MEM_read32(&src[ip + 1..])
                }
            {
                let repMatchEnd = if repInDict { extDict.len() } else { iend };
                matchLength = ZSTD_count_2segments(
                    src,
                    ip + 1 + 4,
                    iend,
                    prefixLowest,
                    if repInDict { extDict.as_slice() } else { src },
                    repMatch + 4,
                    repMatchEnd,
                ) + 4;
                depth0_rep_match = depth == 0;
            }
        } else if offset_1 > 0
            && ip + 1 >= offset_1 as usize
            && unsafe { read32_at(src, ip + 1 - offset_1 as usize) }
                == unsafe { read32_at(src, ip + 1) }
        {
            matchLength = ZSTD_count(src, ip + 1 + 4, ip + 1 + 4 - offset_1 as usize, iend) + 4;
            depth0_rep_match = depth == 0;
        }

        if !depth0_rep_match {
            let mut offbaseFound = 999_999_999u32;
            let ml2 = ZSTD_searchMax(
                ms,
                src,
                if dictMode == ZSTD_dictMode_e::ZSTD_extDict {
                    Some(extDict.as_slice())
                } else if isDxS {
                    Some(dictBase.as_slice())
                } else {
                    None
                },
                ip,
                iend,
                &mut offbaseFound,
                mls,
                rowLog,
                searchMethod,
                dictMode,
            );
            if ml2 > matchLength {
                matchLength = ml2;
                start = ip;
                offBase = offbaseFound;
            }
        }

        if matchLength < 4 {
            let step = ((ip - anchor) >> kSearchStrength) + 1;
            ip += step;
            ms.lazySkipping = u32::from(if dictMode == ZSTD_dictMode_e::ZSTD_extDict {
                step > kLazySkippingStep + 1
            } else {
                step > kLazySkippingStep
            });
            continue;
        }

        if depth >= 1 {
            while ip < ilimit {
                ip += 1;
                if dictMode == ZSTD_dictMode_e::ZSTD_noDict {
                    if offBase != 0
                        && offset_1 > 0
                        && unsafe { read32_at(src, ip - offset_1 as usize) }
                            == unsafe { read32_at(src, ip) }
                    {
                        let mlRep = ZSTD_count(src, ip + 4, ip + 4 - offset_1 as usize, iend) + 4;
                        let gain2 = (mlRep * 3) as i32;
                        let gain1 = (matchLength * 3) as i32 - ZSTD_highbit32(offBase) as i32 + 1;
                        if mlRep >= 4 && gain2 > gain1 {
                            matchLength = mlRep;
                            offBase = REPCODE_TO_OFFBASE(1);
                            start = ip;
                        }
                    }
                }
                if dictMode == ZSTD_dictMode_e::ZSTD_extDict && offBase != 0 {
                    let curr = base_off.wrapping_add(ip as u32);
                    let windowLow = ZSTD_getLowestMatchIndex(ms, curr, ms.cParams.windowLog);
                    let repIndex = curr.wrapping_sub(offset_1);
                    let repInDict = repIndex < prefixLowestIndex;
                    let repMatch = if repInDict {
                        repIndex.saturating_sub(extDictBaseOffset) as usize
                    } else {
                        repIndex.saturating_sub(base_off) as usize
                    };
                    if ZSTD_index_overlap_check(prefixLowestIndex, repIndex)
                        && offset_1 <= curr.wrapping_sub(windowLow)
                        && if repInDict {
                            repMatch + 4 <= extDict.len()
                                && MEM_read32(&extDict[repMatch..]) == MEM_read32(&src[ip..])
                        } else {
                            repMatch + 4 <= src.len()
                                && MEM_read32(&src[repMatch..]) == MEM_read32(&src[ip..])
                        }
                    {
                        let repMatchEnd = if repInDict { extDict.len() } else { iend };
                        let mlRep = ZSTD_count_2segments(
                            src,
                            ip + 4,
                            iend,
                            prefixLowest,
                            if repInDict { extDict.as_slice() } else { src },
                            repMatch + 4,
                            repMatchEnd,
                        ) + 4;
                        let gain2 = (mlRep * 3) as i32;
                        let gain1 = (matchLength * 3) as i32 - ZSTD_highbit32(offBase) as i32 + 1;
                        if mlRep >= 4 && gain2 > gain1 {
                            matchLength = mlRep;
                            offBase = REPCODE_TO_OFFBASE(1);
                            start = ip;
                        }
                    }
                }
                if isDxS {
                    let repIndex = base_off.wrapping_add(ip as u32).wrapping_sub(offset_1);
                    let repInDict = repIndex < prefixLowestIndex;
                    let repMatch = if repInDict {
                        repIndex
                            .saturating_sub(dictIndexDelta)
                            .saturating_sub(dms_base_offset) as usize
                    } else {
                        repIndex.saturating_sub(base_off) as usize
                    };
                    if ZSTD_index_overlap_check(prefixLowestIndex, repIndex)
                        && if repInDict {
                            repMatch + 4 <= dictBase.len()
                                && MEM_read32(&dictBase[repMatch..]) == MEM_read32(&src[ip..])
                        } else {
                            repMatch + 4 <= src.len()
                                && MEM_read32(&src[repMatch..]) == MEM_read32(&src[ip..])
                        }
                    {
                        let repMatchEnd = if repInDict { dictBase.len() } else { iend };
                        let mlRep = ZSTD_count_2segments(
                            src,
                            ip + 4,
                            iend,
                            prefixLowest,
                            if repInDict { dictBase.as_slice() } else { src },
                            repMatch + 4,
                            repMatchEnd,
                        ) + 4;
                        let gain2 = (mlRep * 3) as i32;
                        let gain1 = (matchLength * 3) as i32 - ZSTD_highbit32(offBase) as i32 + 1;
                        if mlRep >= 4 && gain2 > gain1 {
                            matchLength = mlRep;
                            offBase = REPCODE_TO_OFFBASE(1);
                            start = ip;
                        }
                    }
                }
                {
                    let mut cand = 999_999_999u32;
                    let ml2 = ZSTD_searchMax(
                        ms,
                        src,
                        if dictMode == ZSTD_dictMode_e::ZSTD_extDict {
                            Some(extDict.as_slice())
                        } else if isDxS {
                            Some(dictBase.as_slice())
                        } else {
                            None
                        },
                        ip,
                        iend,
                        &mut cand,
                        mls,
                        rowLog,
                        searchMethod,
                        dictMode,
                    );
                    let gain2 = (ml2 * 4) as i32 - ZSTD_highbit32(cand) as i32;
                    let gain1 = (matchLength * 4) as i32 - ZSTD_highbit32(offBase) as i32 + 4;
                    if ml2 >= 4 && gain2 > gain1 {
                        matchLength = ml2;
                        offBase = cand;
                        start = ip;
                        continue;
                    }
                }
                if depth == 2 && ip < ilimit {
                    ip += 1;
                    if dictMode == ZSTD_dictMode_e::ZSTD_noDict {
                        if offBase != 0
                            && offset_1 > 0
                            && unsafe { read32_at(src, ip - offset_1 as usize) }
                                == unsafe { read32_at(src, ip) }
                        {
                            let mlRep =
                                ZSTD_count(src, ip + 4, ip + 4 - offset_1 as usize, iend) + 4;
                            let gain2 = (mlRep * 4) as i32;
                            let gain1 =
                                (matchLength * 4) as i32 - ZSTD_highbit32(offBase) as i32 + 1;
                            if mlRep >= 4 && gain2 > gain1 {
                                matchLength = mlRep;
                                offBase = REPCODE_TO_OFFBASE(1);
                                start = ip;
                            }
                        }
                    }
                    if dictMode == ZSTD_dictMode_e::ZSTD_extDict && offBase != 0 {
                        let curr = base_off.wrapping_add(ip as u32);
                        let windowLow = ZSTD_getLowestMatchIndex(ms, curr, ms.cParams.windowLog);
                        let repIndex = curr.wrapping_sub(offset_1);
                        let repInDict = repIndex < prefixLowestIndex;
                        let repMatch = if repInDict {
                            repIndex.saturating_sub(extDictBaseOffset) as usize
                        } else {
                            repIndex.saturating_sub(base_off) as usize
                        };
                        if ZSTD_index_overlap_check(prefixLowestIndex, repIndex)
                            && offset_1 <= curr.wrapping_sub(windowLow)
                            && if repInDict {
                                repMatch + 4 <= extDict.len()
                                    && MEM_read32(&extDict[repMatch..]) == MEM_read32(&src[ip..])
                            } else {
                                repMatch + 4 <= src.len()
                                    && MEM_read32(&src[repMatch..]) == MEM_read32(&src[ip..])
                            }
                        {
                            let repMatchEnd = if repInDict { extDict.len() } else { iend };
                            let mlRep = ZSTD_count_2segments(
                                src,
                                ip + 4,
                                iend,
                                prefixLowest,
                                if repInDict { extDict.as_slice() } else { src },
                                repMatch + 4,
                                repMatchEnd,
                            ) + 4;
                            let gain2 = (mlRep * 4) as i32;
                            let gain1 =
                                (matchLength * 4) as i32 - ZSTD_highbit32(offBase) as i32 + 1;
                            if mlRep >= 4 && gain2 > gain1 {
                                matchLength = mlRep;
                                offBase = REPCODE_TO_OFFBASE(1);
                                start = ip;
                            }
                        }
                    }
                    if isDxS {
                        let repIndex = base_off.wrapping_add(ip as u32).wrapping_sub(offset_1);
                        let repInDict = repIndex < prefixLowestIndex;
                        let repMatch = if repInDict {
                            repIndex
                                .saturating_sub(dictIndexDelta)
                                .saturating_sub(dms_base_offset)
                                as usize
                        } else {
                            repIndex.saturating_sub(base_off) as usize
                        };
                        if ZSTD_index_overlap_check(prefixLowestIndex, repIndex)
                            && if repInDict {
                                repMatch + 4 <= dictBase.len()
                                    && MEM_read32(&dictBase[repMatch..]) == MEM_read32(&src[ip..])
                            } else {
                                repMatch + 4 <= src.len()
                                    && MEM_read32(&src[repMatch..]) == MEM_read32(&src[ip..])
                            }
                        {
                            let repMatchEnd = if repInDict { dictBase.len() } else { iend };
                            let mlRep = ZSTD_count_2segments(
                                src,
                                ip + 4,
                                iend,
                                prefixLowest,
                                if repInDict { dictBase.as_slice() } else { src },
                                repMatch + 4,
                                repMatchEnd,
                            ) + 4;
                            let gain2 = (mlRep * 4) as i32;
                            let gain1 =
                                (matchLength * 4) as i32 - ZSTD_highbit32(offBase) as i32 + 1;
                            if mlRep >= 4 && gain2 > gain1 {
                                matchLength = mlRep;
                                offBase = REPCODE_TO_OFFBASE(1);
                                start = ip;
                            }
                        }
                    }
                    let mut cand = 999_999_999u32;
                    let ml2 = ZSTD_searchMax(
                        ms,
                        src,
                        if dictMode == ZSTD_dictMode_e::ZSTD_extDict {
                            Some(extDict.as_slice())
                        } else if isDxS {
                            Some(dictBase.as_slice())
                        } else {
                            None
                        },
                        ip,
                        iend,
                        &mut cand,
                        mls,
                        rowLog,
                        searchMethod,
                        dictMode,
                    );
                    let gain2 = (ml2 * 4) as i32 - ZSTD_highbit32(cand) as i32;
                    let gain1 = (matchLength * 4) as i32 - ZSTD_highbit32(offBase) as i32 + 7;
                    if ml2 >= 4 && gain2 > gain1 {
                        matchLength = ml2;
                        offBase = cand;
                        start = ip;
                        continue;
                    }
                }
                break;
            }
        }

        if OFFBASE_IS_OFFSET(offBase) {
            if dictMode == ZSTD_dictMode_e::ZSTD_noDict {
                while start > anchor
                    && start > OFFBASE_TO_OFFSET(offBase) as usize
                    && start - OFFBASE_TO_OFFSET(offBase) as usize > prefixLowest
                    && src[start - 1] == src[start - OFFBASE_TO_OFFSET(offBase) as usize - 1]
                {
                    start -= 1;
                    matchLength += 1;
                }
            }
            if isDxS {
                let matchIndex = base_off
                    .wrapping_add(start as u32)
                    .wrapping_sub(OFFBASE_TO_OFFSET(offBase));
                let repInDict = matchIndex < prefixLowestIndex;
                let mut match_pos = if repInDict {
                    matchIndex
                        .saturating_sub(dictIndexDelta)
                        .saturating_sub(dms_base_offset) as usize
                } else {
                    matchIndex.saturating_sub(base_off) as usize
                };
                let mStart = if repInDict { dictLowest } else { prefixLowest };
                while start > anchor
                    && match_pos > mStart
                    && src[start - 1]
                        == if repInDict {
                            dictBase[match_pos - 1]
                        } else {
                            src[match_pos - 1]
                        }
                {
                    start -= 1;
                    match_pos -= 1;
                    matchLength += 1;
                }
            }
            if dictMode == ZSTD_dictMode_e::ZSTD_extDict {
                let matchIndex = base_off
                    .wrapping_add(start as u32)
                    .wrapping_sub(OFFBASE_TO_OFFSET(offBase));
                let repInDict = matchIndex < prefixLowestIndex;
                let mut match_pos = if repInDict {
                    matchIndex.saturating_sub(extDictBaseOffset) as usize
                } else {
                    matchIndex.saturating_sub(base_off) as usize
                };
                let mStart = if repInDict {
                    extDictLowest
                } else {
                    prefixLowest
                };
                while start > anchor
                    && match_pos > mStart
                    && src[start - 1]
                        == if repInDict {
                            extDict[match_pos - 1]
                        } else {
                            src[match_pos - 1]
                        }
                {
                    start -= 1;
                    match_pos -= 1;
                    matchLength += 1;
                }
            }
            offset_2 = offset_1;
            offset_1 = OFFBASE_TO_OFFSET(offBase);
        }

        let litLength = start - anchor;
        ZSTD_storeSeq(seqStore, litLength, &src[anchor..], offBase, matchLength);
        anchor = start + matchLength;
        ip = anchor;

        if ms.lazySkipping != 0 {
            if searchMethod == searchMethod_e::search_rowHash {
                ZSTD_row_fillHashCache(ms, src, rowLog, mls, ms.nextToUpdate, ilimit);
            }
            ms.lazySkipping = 0;
        }

        if isDxS {
            while ip <= ilimit {
                let current2 = base_off.wrapping_add(ip as u32);
                let repIndex = current2.wrapping_sub(offset_2);
                let repInDict = repIndex < prefixLowestIndex;
                let repMatch = if repInDict {
                    repIndex
                        .saturating_sub(dictIndexDelta)
                        .saturating_sub(dms_base_offset) as usize
                } else {
                    repIndex.saturating_sub(base_off) as usize
                };
                if ZSTD_index_overlap_check(prefixLowestIndex, repIndex)
                    && if repInDict {
                        repMatch + 4 <= dictBase.len()
                            && MEM_read32(&dictBase[repMatch..]) == MEM_read32(&src[ip..])
                    } else {
                        repMatch + 4 <= src.len()
                            && MEM_read32(&src[repMatch..]) == MEM_read32(&src[ip..])
                    }
                {
                    matchLength = ZSTD_count_2segments(
                        src,
                        ip + 4,
                        iend,
                        prefixLowest,
                        if repInDict { dictBase.as_slice() } else { src },
                        repMatch + 4,
                        if repInDict { dictBase.len() } else { iend },
                    ) + 4;
                    offBase = offset_2;
                    offset_2 = offset_1;
                    offset_1 = offBase;
                    ZSTD_storeSeq(
                        seqStore,
                        0,
                        &src[anchor..anchor],
                        REPCODE_TO_OFFBASE(1),
                        matchLength,
                    );
                    ip += matchLength;
                    anchor = ip;
                    continue;
                }
                break;
            }
        }

        if dictMode == ZSTD_dictMode_e::ZSTD_noDict {
            while ip <= ilimit
                && offset_2 > 0
                && unsafe { read32_at(src, ip) }
                    == unsafe { read32_at(src, ip - offset_2 as usize) }
            {
                matchLength = ZSTD_count(src, ip + 4, ip + 4 - offset_2 as usize, iend) + 4;
                offBase = offset_2;
                offset_2 = offset_1;
                offset_1 = offBase;
                ZSTD_storeSeq(
                    seqStore,
                    0,
                    &src[anchor..anchor],
                    REPCODE_TO_OFFBASE(1),
                    matchLength,
                );
                ip += matchLength;
                anchor = ip;
            }
        }

        if dictMode == ZSTD_dictMode_e::ZSTD_extDict {
            while ip <= ilimit {
                let repCurrent = base_off.wrapping_add(ip as u32);
                let windowLow = ZSTD_getLowestMatchIndex(ms, repCurrent, ms.cParams.windowLog);
                let repIndex = repCurrent.wrapping_sub(offset_2);
                let repInDict = repIndex < prefixLowestIndex;
                let repMatch = if repInDict {
                    repIndex.saturating_sub(extDictBaseOffset) as usize
                } else {
                    repIndex.saturating_sub(base_off) as usize
                };
                if !ZSTD_index_overlap_check(prefixLowestIndex, repIndex)
                    || offset_2 > repCurrent.wrapping_sub(windowLow)
                {
                    break;
                }
                let matches = if repInDict {
                    repMatch + 4 <= extDict.len()
                        && MEM_read32(&extDict[repMatch..]) == MEM_read32(&src[ip..])
                } else {
                    repMatch + 4 <= src.len()
                        && MEM_read32(&src[repMatch..]) == MEM_read32(&src[ip..])
                };
                if !matches {
                    break;
                }
                matchLength = ZSTD_count_2segments(
                    src,
                    ip + 4,
                    iend,
                    prefixLowest,
                    if repInDict { extDict.as_slice() } else { src },
                    repMatch + 4,
                    if repInDict { extDict.len() } else { iend },
                ) + 4;
                offBase = offset_2;
                offset_2 = offset_1;
                offset_1 = offBase;
                ZSTD_storeSeq(
                    seqStore,
                    0,
                    &src[anchor..anchor],
                    REPCODE_TO_OFFBASE(1),
                    matchLength,
                );
                ip += matchLength;
                anchor = ip;
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

/// Public entry for strategy=greedy (3): depth=0.
pub fn ZSTD_compressBlock_greedy(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_hashChain,
        0,
        ZSTD_dictMode_e::ZSTD_noDict,
    )
}

/// Public entry for strategy=lazy (4): depth=1.
pub fn ZSTD_compressBlock_lazy(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_hashChain,
        1,
        ZSTD_dictMode_e::ZSTD_noDict,
    )
}

/// Public entry for strategy=lazy2 (5): depth=2.
pub fn ZSTD_compressBlock_lazy2(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_hashChain,
        2,
        ZSTD_dictMode_e::ZSTD_noDict,
    )
}

/// Port of `ZSTD_compressBlock_greedy_row`. Public entry: row-hash
/// search, greedy depth (0), no dictionary.
pub fn ZSTD_compressBlock_greedy_row(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_rowHash,
        0,
        ZSTD_dictMode_e::ZSTD_noDict,
    )
}

/// Port of `ZSTD_compressBlock_lazy_row`. Public entry: row-hash
/// search, lazy depth (1), no dictionary.
pub fn ZSTD_compressBlock_lazy_row(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_rowHash,
        1,
        ZSTD_dictMode_e::ZSTD_noDict,
    )
}

/// Port of `ZSTD_compressBlock_lazy2_row`. Public entry: row-hash
/// search, lazy2 depth (2), no dictionary.
pub fn ZSTD_compressBlock_lazy2_row(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_rowHash,
        2,
        ZSTD_dictMode_e::ZSTD_noDict,
    )
}

/// Port of `ZSTD_compressBlock_greedy_dictMatchState_row`. Row-hash,
/// depth=0, `dictMatchState`.
pub fn ZSTD_compressBlock_greedy_dictMatchState_row(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_rowHash,
        0,
        ZSTD_dictMode_e::ZSTD_dictMatchState,
    )
}

/// Port of `ZSTD_compressBlock_lazy_dictMatchState_row`. Row-hash,
/// depth=1, `dictMatchState`.
pub fn ZSTD_compressBlock_lazy_dictMatchState_row(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_rowHash,
        1,
        ZSTD_dictMode_e::ZSTD_dictMatchState,
    )
}

/// Port of `ZSTD_compressBlock_lazy2_dictMatchState_row`. Row-hash,
/// depth=2, `dictMatchState`.
pub fn ZSTD_compressBlock_lazy2_dictMatchState_row(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_rowHash,
        2,
        ZSTD_dictMode_e::ZSTD_dictMatchState,
    )
}

/// Port of `ZSTD_compressBlock_greedy_extDict_row`. Row-hash, depth=0,
/// `extDict`.
pub fn ZSTD_compressBlock_greedy_extDict_row(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_extDict_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_rowHash,
        0,
    )
}

/// Port of `ZSTD_compressBlock_lazy_extDict_row`. Row-hash, depth=1,
/// `extDict`.
pub fn ZSTD_compressBlock_lazy_extDict_row(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_extDict_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_rowHash,
        1,
    )
}

/// Port of `ZSTD_compressBlock_lazy2_extDict_row`. Row-hash, depth=2,
/// `extDict`.
pub fn ZSTD_compressBlock_lazy2_extDict_row(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_extDict_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_rowHash,
        2,
    )
}

/// Port of `ZSTD_compressBlock_greedy_dedicatedDictSearch`.
/// Hash-chain, depth=0, `dedicatedDictSearch`.
pub fn ZSTD_compressBlock_greedy_dedicatedDictSearch(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_hashChain,
        0,
        ZSTD_dictMode_e::ZSTD_dedicatedDictSearch,
    )
}

/// Port of `ZSTD_compressBlock_lazy_dedicatedDictSearch`. Hash-chain,
/// depth=1, `dedicatedDictSearch`.
pub fn ZSTD_compressBlock_lazy_dedicatedDictSearch(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_hashChain,
        1,
        ZSTD_dictMode_e::ZSTD_dedicatedDictSearch,
    )
}

/// Port of `ZSTD_compressBlock_lazy2_dedicatedDictSearch`. Hash-chain,
/// depth=2, `dedicatedDictSearch`.
pub fn ZSTD_compressBlock_lazy2_dedicatedDictSearch(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_hashChain,
        2,
        ZSTD_dictMode_e::ZSTD_dedicatedDictSearch,
    )
}

/// Port of `ZSTD_compressBlock_greedy_dedicatedDictSearch_row`.
/// Row-hash, depth=0, `dedicatedDictSearch`.
pub fn ZSTD_compressBlock_greedy_dedicatedDictSearch_row(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_rowHash,
        0,
        ZSTD_dictMode_e::ZSTD_dedicatedDictSearch,
    )
}

/// Port of `ZSTD_compressBlock_lazy_dedicatedDictSearch_row`. Row-hash,
/// depth=1, `dedicatedDictSearch`.
pub fn ZSTD_compressBlock_lazy_dedicatedDictSearch_row(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_rowHash,
        1,
        ZSTD_dictMode_e::ZSTD_dedicatedDictSearch,
    )
}

/// Port of `ZSTD_compressBlock_lazy2_dedicatedDictSearch_row`.
/// Row-hash, depth=2, `dedicatedDictSearch`.
pub fn ZSTD_compressBlock_lazy2_dedicatedDictSearch_row(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_rowHash,
        2,
        ZSTD_dictMode_e::ZSTD_dedicatedDictSearch,
    )
}

/// Port of `ZSTD_compressBlock_greedy_dictMatchState`. Hash-chain,
/// depth=0, `dictMatchState`.
pub fn ZSTD_compressBlock_greedy_dictMatchState(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_hashChain,
        0,
        ZSTD_dictMode_e::ZSTD_dictMatchState,
    )
}

/// Port of `ZSTD_compressBlock_lazy_dictMatchState`. Hash-chain,
/// depth=1, `dictMatchState`.
pub fn ZSTD_compressBlock_lazy_dictMatchState(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_hashChain,
        1,
        ZSTD_dictMode_e::ZSTD_dictMatchState,
    )
}

/// Port of `ZSTD_compressBlock_lazy2_dictMatchState`. Hash-chain,
/// depth=2, `dictMatchState`.
pub fn ZSTD_compressBlock_lazy2_dictMatchState(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_hashChain,
        2,
        ZSTD_dictMode_e::ZSTD_dictMatchState,
    )
}

/// Port of `ZSTD_compressBlock_btlazy2_dictMatchState`. Binary-tree
/// search at depth=2, `dictMatchState`.
pub fn ZSTD_compressBlock_btlazy2_dictMatchState(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_binaryTree,
        2,
        ZSTD_dictMode_e::ZSTD_dictMatchState,
    )
}

/// Port of `ZSTD_compressBlock_greedy_extDict`. Hash-chain, depth=0,
/// `extDict`.
pub fn ZSTD_compressBlock_greedy_extDict(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_extDict_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_hashChain,
        0,
    )
}

/// Port of `ZSTD_compressBlock_lazy_extDict`. Hash-chain, depth=1,
/// `extDict`.
pub fn ZSTD_compressBlock_lazy_extDict(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_extDict_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_hashChain,
        1,
    )
}

/// Port of `ZSTD_compressBlock_lazy2_extDict`. Hash-chain, depth=2,
/// `extDict`.
pub fn ZSTD_compressBlock_lazy2_extDict(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_extDict_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_hashChain,
        2,
    )
}

/// Port of `ZSTD_compressBlock_btlazy2_extDict`. Binary-tree search at
/// depth=2, `extDict`.
pub fn ZSTD_compressBlock_btlazy2_extDict(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_extDict_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_binaryTree,
        2,
    )
}

/// Port of `ZSTD_compressBlock_lazy_extDict_generic`.
pub fn ZSTD_compressBlock_lazy_extDict_generic(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
    searchMethod: searchMethod_e,
    depth: u32,
) -> usize {
    ZSTD_compressBlock_lazy_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod,
        depth,
        ZSTD_dictMode_e::ZSTD_extDict,
    )
}

/// Port of `ZSTD_DUBT_findBetterDictMatch`. Walks the dictionary's
/// pre-sorted binary tree looking for a longer match than `bestLength`
/// at `ip`. On improvement, updates `*offsetPtr` with the appropriate
/// offset-code and returns the new best length; otherwise returns the
/// input `bestLength` unchanged.
pub fn ZSTD_DUBT_findBetterDictMatch(
    ms: &mut ZSTD_MatchState_t,
    dictMS: &ZSTD_MatchState_t,
    src: &[u8],
    ip: usize,
    iLimit: usize,
    mut bestLength: usize,
    mut nbCompares: u32,
    offsetPtr: &mut u32,
    mls: u32,
) -> usize {
    let hashLog = dictMS.cParams.hashLog;
    let h = ZSTD_hashPtr(&src[ip..], hashLog, mls);
    let mut dictMatchIndex = dictMS.hashTable[h];

    let base_off = ms.window.base_offset;
    let prefixStart = ms.window.dictLimit.saturating_sub(base_off) as usize;
    let curr = base_off.wrapping_add(ip as u32);
    let dictBase = &dictMS.dictContent;
    let dictHighLimit = dictMS.window.nextSrc;
    let dictLowLimit = dictMS.window.lowLimit;
    let dictIndexDelta = ms.window.lowLimit.wrapping_sub(dictHighLimit);
    let dictBt = &dictMS.chainTable;
    let btLog = dictMS.cParams.chainLog - 1;
    let btMask = (1u32 << btLog) - 1;
    let btLow = if btMask >= dictHighLimit.saturating_sub(dictLowLimit) {
        dictLowLimit
    } else {
        dictHighLimit - btMask
    };

    let mut commonLengthSmaller = 0usize;
    let mut commonLengthLarger = 0usize;

    while nbCompares > 0 && dictMatchIndex > dictLowLimit {
        nbCompares -= 1;
        let nextPtr = (2 * (dictMatchIndex & btMask)) as usize;
        let mut matchLength = commonLengthSmaller.min(commonLengthLarger);
        let dictMatchPos = dictMatchIndex.saturating_sub(dictMS.window.base_offset) as usize;
        matchLength += ZSTD_count_2segments(
            src,
            ip + matchLength,
            iLimit,
            prefixStart,
            dictBase,
            dictMatchPos + matchLength,
            dictBase.len(),
        );

        if matchLength > bestLength {
            let matchIndex = dictMatchIndex.wrapping_add(dictIndexDelta);
            let currCost = ZSTD_highbit32(curr.wrapping_sub(matchIndex).wrapping_add(1)) as i32;
            let prevOff = (*offsetPtr).wrapping_add(1);
            let prevCost = if prevOff != 0 {
                ZSTD_highbit32(prevOff) as i32
            } else {
                i32::MAX
            };
            if 4 * (matchLength as i32 - bestLength as i32) > (currCost - prevCost) {
                bestLength = matchLength;
                *offsetPtr = OFFSET_TO_OFFBASE(curr.wrapping_sub(matchIndex));
            }
            if ip + matchLength == iLimit {
                break;
            }
        }

        let use_prefix_byte = dictMatchIndex.wrapping_add(matchLength as u32) >= dictHighLimit;
        let matchByte = if use_prefix_byte {
            let prefixPos = dictMatchIndex
                .wrapping_add(dictIndexDelta)
                .wrapping_sub(base_off) as usize;
            src[prefixPos + matchLength]
        } else {
            dictBase[dictMatchPos + matchLength]
        };

        if matchByte < src[ip + matchLength] {
            if dictMatchIndex <= btLow {
                break;
            }
            commonLengthSmaller = matchLength;
            dictMatchIndex = dictBt[nextPtr + 1];
        } else {
            if dictMatchIndex <= btLow {
                break;
            }
            commonLengthLarger = matchLength;
            dictMatchIndex = dictBt[nextPtr];
        }
    }

    bestLength
}

/// Port of `ZSTD_DUBT_findBestMatch`. Deferred-update binary-tree
/// match finder: first sorts any pending `ZSTD_DUBT_UNSORTED_MARK`
/// entries via `ZSTD_insertDUBT1`, then walks the freshly-sorted tree
/// at `ip` to find the best match. When `dictMode` is
/// `dictMatchState`, falls back to `ZSTD_DUBT_findBetterDictMatch`
/// against the attached dictionary tree. Writes the chosen offset
/// code to `*offBasePtr` and returns the match length.
pub fn ZSTD_DUBT_findBestMatch(
    ms: &mut ZSTD_MatchState_t,
    src: &[u8],
    ip: usize,
    iLimit: usize,
    offBasePtr: &mut u32,
    mls: u32,
    dictMode: ZSTD_dictMode_e,
) -> usize {
    use crate::compress::match_state::ZSTD_DUBT_UNSORTED_MARK;

    let cParams = ms.cParams;
    let curr = ms.window.base_offset.wrapping_add(ip as u32);
    let h = ZSTD_hashPtr(&src[ip..], cParams.hashLog, mls);
    let windowLow = ZSTD_getLowestMatchIndex(ms, curr, cParams.windowLog);
    let btLog = cParams.chainLog - 1;
    let btMask = (1u32 << btLog) - 1;
    let btLow = curr.saturating_sub(btMask);
    let unsortLimit = btLow.max(windowLow);
    let mut matchIndex = ms.hashTable[h];
    let mut nbCompares = 1u32 << cParams.searchLog;
    let mut nbCandidates = nbCompares;
    let mut previousCandidate = 0u32;

    debug_assert!(ip + 8 <= src.len());
    debug_assert!(dictMode != ZSTD_dictMode_e::ZSTD_dedicatedDictSearch);

    while matchIndex > unsortLimit
        && ms.chainTable[(2 * (matchIndex & btMask)) as usize + 1] == ZSTD_DUBT_UNSORTED_MARK
        && nbCandidates > 1
    {
        let next_slot = (2 * (matchIndex & btMask)) as usize;
        let unsorted_slot = next_slot + 1;
        ms.chainTable[unsorted_slot] = previousCandidate;
        previousCandidate = matchIndex;
        matchIndex = ms.chainTable[next_slot];
        nbCandidates -= 1;
    }

    if matchIndex > unsortLimit
        && ms.chainTable[(2 * (matchIndex & btMask)) as usize + 1] == ZSTD_DUBT_UNSORTED_MARK
    {
        let next_slot = (2 * (matchIndex & btMask)) as usize;
        ms.chainTable[next_slot] = 0;
        ms.chainTable[next_slot + 1] = 0;
    }

    matchIndex = previousCandidate;
    while matchIndex != 0 {
        let nextCandidateIdxPtr = (2 * (matchIndex & btMask)) as usize + 1;
        let nextCandidateIdx = ms.chainTable[nextCandidateIdxPtr];
        ZSTD_insertDUBT1(
            ms,
            src,
            matchIndex,
            iLimit,
            nbCandidates,
            unsortLimit,
            dictMode,
        );
        matchIndex = nextCandidateIdx;
        nbCandidates += 1;
    }

    let mut commonLengthSmaller = 0usize;
    let mut commonLengthLarger = 0usize;
    let mut smaller_slot: Option<usize> = Some((2 * (curr & btMask)) as usize);
    let mut larger_slot: Option<usize> = Some((2 * (curr & btMask)) as usize + 1);
    let mut matchEndIdx = curr.wrapping_add(9);
    let mut bestLength = 0usize;

    matchIndex = ms.hashTable[h];
    ms.hashTable[h] = curr;

    while nbCompares > 0 && matchIndex > windowLow {
        let nextPtr = (2 * (matchIndex & btMask)) as usize;
        let mut matchLength = commonLengthSmaller.min(commonLengthLarger);
        if dictMode != ZSTD_dictMode_e::ZSTD_extDict
            || matchIndex.wrapping_add(matchLength as u32) >= ms.window.dictLimit
        {
            let match_pos = matchIndex.saturating_sub(ms.window.base_offset) as usize;
            matchLength += ZSTD_count(src, ip + matchLength, match_pos + matchLength, iLimit);
        } else {
            let prefixStart = ms.window.dictLimit.saturating_sub(ms.window.base_offset) as usize;
            let dict_pos = matchIndex.saturating_sub(ms.window.dictBase_offset) as usize;
            matchLength += ZSTD_count_2segments(
                src,
                ip + matchLength,
                iLimit,
                prefixStart,
                &ms.dictContent,
                dict_pos + matchLength,
                ms.dictContent.len(),
            );
        }

        if matchLength > bestLength {
            if matchLength > matchEndIdx.wrapping_sub(matchIndex) as usize {
                matchEndIdx = matchIndex.wrapping_add(matchLength as u32);
            }
            let currCost = ZSTD_highbit32(curr.wrapping_sub(matchIndex).wrapping_add(1)) as i32;
            debug_assert!(*offBasePtr != 0);
            let prevCost = ZSTD_highbit32(*offBasePtr) as i32;
            if 4 * (matchLength - bestLength) as i32 > (currCost - prevCost) {
                bestLength = matchLength;
                *offBasePtr = OFFSET_TO_OFFBASE(curr.wrapping_sub(matchIndex));
            }
            if ip + matchLength == iLimit {
                if dictMode == ZSTD_dictMode_e::ZSTD_dictMatchState {
                    nbCompares = 0;
                }
                break;
            }
        }

        let match_byte = if dictMode == ZSTD_dictMode_e::ZSTD_extDict
            && matchIndex.wrapping_add(matchLength as u32) < ms.window.dictLimit
        {
            let dict_pos = matchIndex
                .saturating_sub(ms.window.dictBase_offset)
                .saturating_add(matchLength as u32) as usize;
            ms.dictContent[dict_pos]
        } else {
            let match_pos = matchIndex
                .saturating_add(matchLength as u32)
                .saturating_sub(ms.window.base_offset) as usize;
            src[match_pos]
        };

        if match_byte < src[ip + matchLength] {
            if let Some(slot) = smaller_slot {
                ms.chainTable[slot] = matchIndex;
            }
            commonLengthSmaller = matchLength;
            if matchIndex <= btLow {
                smaller_slot = None;
                break;
            }
            smaller_slot = Some(nextPtr + 1);
            matchIndex = ms.chainTable[nextPtr + 1];
        } else {
            if let Some(slot) = larger_slot {
                ms.chainTable[slot] = matchIndex;
            }
            commonLengthLarger = matchLength;
            if matchIndex <= btLow {
                larger_slot = None;
                break;
            }
            larger_slot = Some(nextPtr);
            matchIndex = ms.chainTable[nextPtr];
        }

        nbCompares -= 1;
    }

    if let Some(slot) = smaller_slot {
        ms.chainTable[slot] = 0;
    }
    if let Some(slot) = larger_slot {
        ms.chainTable[slot] = 0;
    }

    if dictMode == ZSTD_dictMode_e::ZSTD_dictMatchState && nbCompares > 0 {
        if let Some(dictMS) = ms.dictMatchState.clone() {
            bestLength = ZSTD_DUBT_findBetterDictMatch(
                ms, &dictMS, src, ip, iLimit, bestLength, nbCompares, offBasePtr, mls,
            );
        }
    }

    debug_assert!(matchEndIdx > curr.wrapping_add(8));
    ms.nextToUpdate = matchEndIdx.wrapping_sub(8);
    bestLength
}

/// Port of `ZSTD_BtFindBestMatch`. Tree-updater facade: brings the
/// binary tree up to `ip` via `ZSTD_updateDUBT`, then defers to
/// `ZSTD_DUBT_findBestMatch` for the actual search.
pub fn ZSTD_BtFindBestMatch(
    ms: &mut ZSTD_MatchState_t,
    src: &[u8],
    ip: usize,
    iLimit: usize,
    offBasePtr: &mut u32,
    mls: u32,
    dictMode: ZSTD_dictMode_e,
) -> usize {
    let curr = ms.window.base_offset.wrapping_add(ip as u32);
    if curr < ms.nextToUpdate {
        return 0;
    }
    ZSTD_updateDUBT(ms, src, curr, mls);
    ZSTD_DUBT_findBestMatch(ms, src, ip, iLimit, offBasePtr, mls, dictMode)
}

/// Cross-block-history variant — caller passes `src[..istart]` as
/// prior content. Shared by all three depth variants. When the
/// strategy is row-hash-eligible (greedy/lazy/lazy2 with row-hash
/// enabled — the auto-resolved default), routes through the row-hash
/// matcher via `lazy_generic_with_istart` for a substantial speed-up.
pub fn ZSTD_compressBlock_lazy_with_history(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
    istart: usize,
    depth: u32,
    useRowMatchFinder: ZSTD_ParamSwitch_e,
) -> usize {
    use crate::compress::match_state::ZSTD_rowMatchFinderUsed;

    if ZSTD_rowMatchFinderUsed(ms.cParams.strategy, useRowMatchFinder) {
        return ZSTD_compressBlock_lazy_generic_with_istart(
            ms,
            seqStore,
            rep,
            src,
            istart,
            searchMethod_e::search_rowHash,
            depth,
            ZSTD_dictMode_e::ZSTD_noDict,
        );
    }
    ZSTD_compressBlock_lazy_generic_with_istart(
        ms,
        seqStore,
        rep,
        src,
        istart,
        searchMethod_e::search_hashChain,
        depth,
        ZSTD_dictMode_e::ZSTD_noDict,
    )
}

/// Port of `ZSTD_compressBlock_btlazy2`. Binary-tree search at depth=2,
/// no dictionary.
pub fn ZSTD_compressBlock_btlazy2(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    src: &[u8],
) -> usize {
    ZSTD_compressBlock_lazy_generic(
        ms,
        seqStore,
        rep,
        src,
        searchMethod_e::search_binaryTree,
        2,
        ZSTD_dictMode_e::ZSTD_noDict,
    )
}

/// Cross-block-window variant of btlazy2 — caller passes
/// `window_buf[..src_end]` as the matchable buffer (so the BT walk can
/// reach into prior-block bytes via `ms.window`'s base_offset) and
/// `src_pos` is the offset where the current block starts. Without
/// this, multi-block compression at L13/L15 (btlazy2) re-trimmed `src`
/// to just the current block, the BT walked stale prior-block
/// matchIndex entries pointing past the trimmed slice end, the
/// bounds checks rejected those matches and cross-block matching
/// effectively didn't happen — visible as a 1.78× ratio on 10 MB
/// silesia at L15 vs upstream's 3.33×.
pub fn ZSTD_compressBlock_btlazy2_window(
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut SeqStore_t,
    rep: &mut [u32; ZSTD_REP_NUM],
    window_buf: &[u8],
    src_pos: usize,
    src_end: usize,
) -> usize {
    let block_end = &window_buf[..src_end];
    ZSTD_compressBlock_lazy_generic_with_istart(
        ms,
        seqStore,
        rep,
        block_end,
        src_pos,
        searchMethod_e::search_binaryTree,
        2,
        ZSTD_dictMode_e::ZSTD_noDict,
    )
}

/// Port of `ZSTD_updateDUBT`. Inserts positions `[nextToUpdate, target)`
/// into the btlazy2 chain table as an unsorted queue — each new entry
/// is prepended to the hash-bucket chain and its sort-mark slot is
/// set to `ZSTD_DUBT_UNSORTED_MARK`. A later sort pass
/// (`ZSTD_insertDUBT1`) consumes the queue during search.
///
/// `buf` is the window buffer, `target` is the absolute index to
/// fill up to.
pub fn ZSTD_updateDUBT(ms: &mut ZSTD_MatchState_t, buf: &[u8], target: u32, mls: u32) {
    use crate::compress::match_state::ZSTD_DUBT_UNSORTED_MARK;
    let base_off = ms.window.base_offset;
    let hashLog = ms.cParams.hashLog;
    let btLog = ms.cParams.chainLog - 1;
    let btMask: u32 = (1u32 << btLog) - 1;
    let chainSize = 1usize << ms.cParams.chainLog;
    if ms.chainTable.len() < chainSize {
        ms.chainTable.resize(chainSize, 0);
    }

    debug_assert!(ms.nextToUpdate >= ms.window.dictLimit);
    let mut idx = ms.nextToUpdate;
    while idx < target {
        let rel = idx.saturating_sub(base_off) as usize;
        if rel + 8 > buf.len() {
            break;
        }
        let h = ZSTD_hashPtr(&buf[rel..], hashLog, mls);
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

/// Port of `ZSTD_insertDUBT1`. Sorts one
/// already-queued unsorted DUBT entry at position `curr` into its
/// proper place in the binary tree (up to `nbCompares` comparisons).
///
/// Inputs:
///   - `buf`: the window buffer
///   - `curr`: absolute position being sorted
///   - `iend_pos`: end offset of input (clamp for `ZSTD_count`)
///   - `nbCompares`, `btLow`: caller-supplied search limits
pub fn ZSTD_insertDUBT1(
    ms: &mut ZSTD_MatchState_t,
    buf: &[u8],
    curr: u32,
    iend_pos: usize,
    mut nbCompares: u32,
    btLow: u32,
    dictMode: ZSTD_dictMode_e,
) {
    let base_off = ms.window.base_offset;
    let dict_base_off = ms.window.dictBase_offset;
    let dict_limit = ms.window.dictLimit;
    let btLog = ms.cParams.chainLog - 1;
    let btMask: u32 = (1u32 << btLog) - 1;
    let mut commonLengthSmaller: usize = 0;
    let mut commonLengthLarger: usize = 0;
    let curr_in_dict = dictMode == ZSTD_dictMode_e::ZSTD_extDict && curr < dict_limit;
    let ip_pos = if curr_in_dict {
        curr.saturating_sub(dict_base_off) as usize
    } else {
        curr.saturating_sub(base_off) as usize
    };
    let ip_buf: &[u8] = if curr_in_dict { &ms.dictContent } else { buf };
    let ip_limit = if curr_in_dict {
        ms.dictContent.len()
    } else {
        iend_pos
    };

    let mut smaller_slot: Option<usize> = Some((2 * (curr & btMask)) as usize);
    let mut larger_slot: Option<usize> = Some((2 * (curr & btMask)) as usize + 1);

    // Read the head of the unsorted queue from the chain table at
    // `curr`'s smallerPtr slot — upstream starts iteration from there.
    let mut matchIndex = ms.chainTable[smaller_slot.unwrap()];
    let windowValid = ms.window.lowLimit;
    let maxDistance = 1u32 << ms.cParams.windowLog;
    let windowLow = if curr.wrapping_sub(windowValid) > maxDistance {
        curr.wrapping_sub(maxDistance)
    } else {
        windowValid
    };

    debug_assert!(curr >= btLow);

    while nbCompares > 0 && matchIndex > windowLow {
        let nextBase = (2 * (matchIndex & btMask)) as usize;
        let mut matchLength = commonLengthSmaller.min(commonLengthLarger);
        debug_assert!(matchIndex < curr);

        if dictMode != ZSTD_dictMode_e::ZSTD_extDict
            || matchIndex.wrapping_add(matchLength as u32) >= dict_limit
            || curr_in_dict
        {
            let (match_buf, match_pos) = if dictMode == ZSTD_dictMode_e::ZSTD_extDict
                && matchIndex.wrapping_add(matchLength as u32) < dict_limit
            {
                (
                    &ms.dictContent[..],
                    matchIndex.saturating_sub(dict_base_off) as usize,
                )
            } else {
                (buf, matchIndex.saturating_sub(base_off) as usize)
            };
            debug_assert!(core::ptr::eq(ip_buf.as_ptr(), match_buf.as_ptr()));
            matchLength += ZSTD_count(
                match_buf,
                ip_pos + matchLength,
                match_pos + matchLength,
                ip_limit,
            );
        } else {
            let prefixStart = dict_limit.saturating_sub(base_off) as usize;
            let dict_pos = matchIndex.saturating_sub(dict_base_off) as usize;
            matchLength += ZSTD_count_2segments(
                buf,
                ip_pos + matchLength,
                iend_pos,
                prefixStart,
                &ms.dictContent,
                dict_pos + matchLength,
                ms.dictContent.len(),
            );
        }

        if ip_pos + matchLength == ip_limit {
            break;
        }

        let match_byte = if dictMode == ZSTD_dictMode_e::ZSTD_extDict
            && (matchIndex.wrapping_add(matchLength as u32) < dict_limit || curr_in_dict)
        {
            let dict_pos = matchIndex
                .saturating_sub(dict_base_off)
                .saturating_add(matchLength as u32) as usize;
            ms.dictContent[dict_pos]
        } else {
            let match_pos = matchIndex
                .saturating_add(matchLength as u32)
                .saturating_sub(base_off) as usize;
            buf[match_pos]
        };

        if match_byte < ip_buf[ip_pos + matchLength] {
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
pub fn ZSTD_insertAndFindFirstIndex(ms: &mut ZSTD_MatchState_t, src: &[u8], ip: usize) -> u32 {
    let mls = ms.cParams.minMatch;
    ZSTD_insertAndFindFirstIndex_internal(ms, src, ip, mls, 0)
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
            streaming_operation, ZSTD_DCtx, ZSTD_decoder_entropy_rep, ZSTD_decompressBlock_internal,
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
            let last =
                ZSTD_compressBlock_lazy_noDict_generic(&mut ms, &mut seq, &mut rep, &src, 0, depth);
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
            assert!(
                !crate::common::error::ERR_isError(body_n),
                "depth {depth} compress err"
            );
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
            assert!(
                !crate::common::error::ERR_isError(decoded),
                "depth {depth} decode err: {decoded:#x}"
            );
            assert_eq!(decoded, src.len());
            assert_eq!(&out[..decoded], &src[..], "depth {depth} mismatch");
        }
    }

    #[test]
    fn btlazy2_roundtrip_through_decoder() {
        use crate::decompress::zstd_decompress_block::{
            streaming_operation, ZSTD_DCtx, ZSTD_decoder_entropy_rep, ZSTD_decompressBlock_internal,
        };

        let src: Vec<u8> = b"To be, or not to be, that is the question. "
            .iter()
            .cycle()
            .take(2000)
            .copied()
            .collect();

        let mut ms = ZSTD_MatchState_t::new(ZSTD_compressionParameters {
            windowLog: 17,
            hashLog: 12,
            chainLog: 12,
            searchLog: 2,
            minMatch: 4,
            strategy: 6,
            ..Default::default()
        });
        let mut seq = SeqStore_t::with_capacity(1024, 131072);
        let mut rep: [u32; ZSTD_REP_NUM] = [1, 4, 8];
        let last = ZSTD_compressBlock_btlazy2(&mut ms, &mut seq, &mut rep, &src);
        seq.literals.extend_from_slice(&src[src.len() - last..]);

        crate::compress::zstd_compress::ZSTD_seqToCodes(&mut seq);
        let prev = crate::compress::zstd_compress::ZSTD_entropyCTables_t::default();
        let mut next = crate::compress::zstd_compress::ZSTD_entropyCTables_t::default();
        let mut body = vec![0u8; 4096];
        let body_n = crate::compress::zstd_compress::ZSTD_entropyCompressSeqStore(
            &mut body,
            &mut seq,
            &prev,
            &mut next,
            6,
            0,
            src.len(),
            0,
        );
        if body_n == 0 {
            return;
        }
        assert!(
            !crate::common::error::ERR_isError(body_n),
            "btlazy2 compress err"
        );
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
        assert!(
            !crate::common::error::ERR_isError(decoded),
            "btlazy2 decode err: {decoded:#x}"
        );
        assert_eq!(decoded, src.len());
        assert_eq!(&out[..decoded], &src[..]);
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
        ZSTD_insertDUBT1(
            &mut ms,
            &src,
            500,
            src.len(),
            16,
            4,
            ZSTD_dictMode_e::ZSTD_noDict,
        );
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
        assert_eq!(
            marked, 100,
            "all 100 queued positions should have UNSORTED_MARK"
        );
    }

    #[test]
    fn updateDUBT_uses_base_offset_relative_source_indexes() {
        let cp = ZSTD_compressionParameters {
            hashLog: 16,
            chainLog: 8,
            minMatch: 4,
            strategy: 6,
            ..Default::default()
        };
        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.chainTable = vec![0u32; 1 << cp.chainLog];
        ms.window.base_offset = 10_000;
        ms.window.dictLimit = 10_000;
        ms.window.lowLimit = 10_000;
        ms.nextToUpdate = 10_000;

        let src = b"abcdefghijklmnopqrstuvwxyz012345";
        let target = ms.window.base_offset.wrapping_add(8);
        ZSTD_updateDUBT(&mut ms, src, target, 4);

        let h0 = ZSTD_hashPtr(src, cp.hashLog, 4);
        let h1 = ZSTD_hashPtr(&src[1..], cp.hashLog, 4);
        assert_eq!(ms.hashTable[h0], ms.window.base_offset);
        assert_eq!(ms.hashTable[h1], ms.window.base_offset.wrapping_add(1));
        assert_eq!(ms.nextToUpdate, target);
    }

    #[test]
    fn lazy_dict_and_ext_wrappers_route_through_live_entries() {
        fn build_ms(strategy: u32) -> ZSTD_MatchState_t {
            let cp = ZSTD_compressionParameters {
                windowLog: 17,
                hashLog: 12,
                chainLog: 12,
                minMatch: 4,
                strategy,
                ..Default::default()
            };
            ZSTD_MatchState_t::new(cp)
        }

        let dict: Vec<u8> =
            b"abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ".to_vec();
        let src = b"mnopqrstuvwxyz012345mnopqrstuvwxyz012345".to_vec();

        let dict_variants: [fn(
            &mut ZSTD_MatchState_t,
            &mut SeqStore_t,
            &mut [u32; ZSTD_REP_NUM],
            &[u8],
        ) -> usize; 4] = [
            ZSTD_compressBlock_greedy_dictMatchState,
            ZSTD_compressBlock_lazy_dictMatchState,
            ZSTD_compressBlock_lazy2_dictMatchState,
            ZSTD_compressBlock_btlazy2_dictMatchState,
        ];
        let ext_variants: [fn(
            &mut ZSTD_MatchState_t,
            &mut SeqStore_t,
            &mut [u32; ZSTD_REP_NUM],
            &[u8],
        ) -> usize; 4] = [
            ZSTD_compressBlock_greedy_extDict,
            ZSTD_compressBlock_lazy_extDict,
            ZSTD_compressBlock_lazy2_extDict,
            ZSTD_compressBlock_btlazy2_extDict,
        ];

        for (idx, f) in dict_variants.into_iter().enumerate() {
            let strategy = match idx {
                0 => 3,
                1 => 4,
                2 => 5,
                _ => 6,
            };
            let mut ms = build_ms(strategy);
            let mut dms = build_ms(strategy);
            let dms_mls = dms.cParams.minMatch;
            dms.dictContent = dict.clone();
            dms.window.nextSrc = dms.window.base_offset.wrapping_add(dict.len() as u32);
            dms.chainTable.resize(1 << dms.cParams.chainLog, 0);
            for pos in 0..dict.len().saturating_sub(4) {
                let _ = ZSTD_insertAndFindFirstIndex_internal(&mut dms, &dict, pos, dms_mls, 0);
            }
            ms.dictMatchState = Some(Box::new(dms));
            ms.window.base_offset = crate::compress::match_state::ZSTD_WINDOW_START_INDEX
                .wrapping_add(dict.len() as u32);
            ms.window.dictLimit = ms.window.base_offset;
            ms.window.lowLimit = ms.window.base_offset;
            ms.nextToUpdate = ms.window.base_offset;
            ms.loadedDictEnd = dict.len() as u32;
            let mut seq = SeqStore_t::with_capacity(1024, 131072);
            let mut rep = [1u32, 4, 8];
            let last_lits = f(&mut ms, &mut seq, &mut rep, &src);
            assert!(
                last_lits < src.len(),
                "dict variant {idx} emitted only literals"
            );
            assert!(
                !seq.sequences.is_empty(),
                "dict variant {idx} regressed into a dead wrapper"
            );
        }

        for (idx, f) in ext_variants.into_iter().enumerate() {
            let strategy = match idx {
                0 => 3,
                1 => 4,
                2 => 5,
                _ => 6,
            };
            let mut ms = build_ms(strategy);
            let ms_mls = ms.cParams.minMatch;
            ms.chainTable.resize(1 << ms.cParams.chainLog, 0);
            ms.dictContent = dict.clone();
            ms.window.base_offset = crate::compress::match_state::ZSTD_WINDOW_START_INDEX;
            ms.nextToUpdate = ms.window.base_offset;
            for pos in 0..dict.len().saturating_sub(4) {
                let _ = ZSTD_insertAndFindFirstIndex_internal(&mut ms, &dict, pos, ms_mls, 0);
            }
            ms.window.dictBase_offset = crate::compress::match_state::ZSTD_WINDOW_START_INDEX;
            ms.window.lowLimit = crate::compress::match_state::ZSTD_WINDOW_START_INDEX;
            ms.window.dictLimit = crate::compress::match_state::ZSTD_WINDOW_START_INDEX
                .wrapping_add(dict.len() as u32);
            ms.window.base_offset = ms.window.dictLimit;
            ms.window.nextSrc = ms.window.base_offset.wrapping_add(src.len() as u32);
            ms.nextToUpdate = ms.window.base_offset;
            ms.loadedDictEnd = dict.len() as u32;

            let mut seq = SeqStore_t::with_capacity(1024, 131072);
            let mut rep = [1u32, 4, 8];
            let last_lits = f(&mut ms, &mut seq, &mut rep, &src);
            assert!(
                last_lits < src.len(),
                "ext variant {idx} emitted only literals"
            );
            assert!(
                !seq.sequences.is_empty(),
                "ext variant {idx} regressed into a dead wrapper"
            );
        }
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
        assert_eq!(ms.nextToUpdate, ms.window.base_offset.wrapping_add(30));
        // head must be either 0 (no collision) or a valid prior position.
        assert!(head < ms.window.base_offset.wrapping_add(30));
    }

    #[test]
    fn insert_and_find_first_index_lazy_skipping_inserts_one_position() {
        let mut ms = ZSTD_MatchState_t::new(ZSTD_compressionParameters {
            hashLog: 10,
            chainLog: 10,
            minMatch: 4,
            ..Default::default()
        });
        ms.chainTable.resize(1 << 10, 0);
        let src: Vec<u8> = (0..128u8).collect();
        ms.nextToUpdate = ms.window.base_offset.wrapping_add(10);

        let _ = ZSTD_insertAndFindFirstIndex_internal(&mut ms, &src, 30, 4, 1);

        assert_eq!(ms.nextToUpdate, ms.window.base_offset.wrapping_add(30));
        let inserted_hash = ZSTD_hashPtr(&src[10..], ms.cParams.hashLog, 4);
        let skipped_hash = ZSTD_hashPtr(&src[11..], ms.cParams.hashLog, 4);
        assert_eq!(ms.hashTable[inserted_hash], ms.window.base_offset + 10);
        assert_ne!(ms.hashTable[skipped_hash], ms.window.base_offset + 11);
    }

    #[test]
    fn hc_find_best_match_no_dict_returns_seeded_len_without_match() {
        let mut ms = ZSTD_MatchState_t::new(ZSTD_compressionParameters {
            windowLog: 17,
            hashLog: 12,
            chainLog: 12,
            searchLog: 4,
            minMatch: 4,
            ..Default::default()
        });
        ms.chainTable.resize(1 << ms.cParams.chainLog, 0);
        let src: Vec<u8> = (0..96u8).collect();
        let mut offbase = 0xDEAD_BEEFu32;

        let ml = ZSTD_HcFindBestMatch_noDict(&mut ms, &src, 32, src.len(), &mut offbase, 4);

        assert_eq!(ml, 3);
        assert_eq!(offbase, 0xDEAD_BEEF);
    }

    #[test]
    fn hc_dict_match_state_offsets_account_for_dict_base_offset() {
        let cp = ZSTD_compressionParameters {
            windowLog: 17,
            hashLog: 12,
            chainLog: 12,
            searchLog: 4,
            minMatch: 4,
            ..Default::default()
        };
        let dict = b"0123456789abcdef".to_vec();
        let dict_pos = 6usize;
        let src = b"6789abcd".to_vec();

        let mut dms = ZSTD_MatchState_t::new(cp);
        dms.dictContent = dict.clone();
        dms.window.nextSrc = dms.window.base_offset.wrapping_add(dict.len() as u32);
        dms.chainTable.resize(1 << cp.chainLog, 0);
        let h = ZSTD_hashPtr(&src, cp.hashLog, cp.minMatch);
        dms.hashTable[h] = dms.window.base_offset.wrapping_add(dict_pos as u32);

        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.window.base_offset = ms.window.base_offset.wrapping_add(dict.len() as u32);
        ms.window.dictLimit = ms.window.base_offset;
        ms.window.lowLimit = ms.window.base_offset;
        ms.nextToUpdate = ms.window.base_offset;
        ms.dictMatchState = Some(Box::new(dms));

        let mut offbase = 0u32;
        let ml = ZSTD_HcFindBestMatch(
            &mut ms,
            &src,
            0,
            src.len(),
            &mut offbase,
            cp.minMatch,
            ZSTD_dictMode_e::ZSTD_dictMatchState,
        );

        assert_eq!(ml, src.len());
        assert_eq!(OFFBASE_TO_OFFSET(offbase), (dict.len() - dict_pos) as u32);
    }

    #[test]
    fn hc_extdict_offsets_account_for_dict_base_offset() {
        let cp = ZSTD_compressionParameters {
            windowLog: 17,
            hashLog: 12,
            chainLog: 12,
            searchLog: 4,
            minMatch: 4,
            ..Default::default()
        };
        let dict = b"0123456789abcdef".to_vec();
        let dict_pos = 6usize;
        let src = b"6789abcd".to_vec();
        let dict_base = 100u32;
        let dict_limit = dict_base.wrapping_add(dict.len() as u32);

        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.dictContent = dict.clone();
        ms.window.dictBase_offset = dict_base;
        ms.window.lowLimit = dict_base;
        ms.window.dictLimit = dict_limit;
        ms.window.base_offset = dict_limit;
        ms.window.nextSrc = dict_limit.wrapping_add(src.len() as u32);
        ms.nextToUpdate = dict_limit;
        ms.loadedDictEnd = dict.len() as u32;
        ms.chainTable.resize(1 << cp.chainLog, 0);
        let h = ZSTD_hashPtr(&src, cp.hashLog, cp.minMatch);
        ms.hashTable[h] = dict_base.wrapping_add(dict_pos as u32);

        let mut offbase = 0u32;
        let ml = ZSTD_HcFindBestMatch(
            &mut ms,
            &src,
            0,
            src.len(),
            &mut offbase,
            cp.minMatch,
            ZSTD_dictMode_e::ZSTD_extDict,
        );

        assert_eq!(ml, src.len());
        assert_eq!(OFFBASE_TO_OFFSET(offbase), (dict.len() - dict_pos) as u32);
    }

    #[test]
    fn extdict_generic_without_extdict_preserves_rep_offsets() {
        let mut ms = ZSTD_MatchState_t::new(ZSTD_compressionParameters {
            windowLog: 17,
            hashLog: 12,
            chainLog: 12,
            searchLog: 2,
            minMatch: 4,
            strategy: 4,
            ..Default::default()
        });
        let src: Vec<u8> = (0..96u8).collect();
        let mut seq = SeqStore_t::with_capacity(1024, 131072);
        let mut rep = [1 << 20, 1 << 19, 8];

        let last = ZSTD_compressBlock_lazy_extDict_generic(
            &mut ms,
            &mut seq,
            &mut rep,
            &src,
            searchMethod_e::search_hashChain,
            1,
        );

        assert!(last <= src.len());
        assert_eq!(rep[0], 1 << 20);
        assert_eq!(rep[1], 1 << 19);
    }

    #[test]
    fn dubt_find_better_dict_match_preserves_upstream_zero_offset_cost_model() {
        let cp = ZSTD_compressionParameters {
            windowLog: 17,
            hashLog: 10,
            chainLog: 10,
            minMatch: 4,
            strategy: 6,
            ..Default::default()
        };

        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.window.base_offset = 128;
        ms.window.dictLimit = 128;
        ms.window.lowLimit = 128;

        let mut dict_ms = ZSTD_MatchState_t::new(cp);
        dict_ms.window.base_offset = 0;
        dict_ms.window.lowLimit = 0;
        dict_ms.window.dictLimit = 0;
        dict_ms.window.nextSrc = 1024;
        let mut dict = vec![b'Y'; 1024];
        dict[4..9].copy_from_slice(b"abcde");
        dict_ms.dictContent = dict;
        dict_ms.chainTable.resize(1 << cp.chainLog, 0);

        let src = b"abcdeZ";
        let h = ZSTD_hashPtr(src, cp.hashLog, cp.minMatch);
        dict_ms.hashTable[h] = 4;

        let mut offbase = 0u32;
        let best = ZSTD_DUBT_findBetterDictMatch(
            &mut ms,
            &dict_ms,
            src,
            0,
            5,
            4,
            1,
            &mut offbase,
            cp.minMatch,
        );

        assert_eq!(best, 4);
        assert_eq!(offbase, 0);
    }

    #[test]
    fn insert_dubt1_extdict_sorts_dictionary_current_against_dictionary_match() {
        let cp = ZSTD_compressionParameters {
            windowLog: 17,
            hashLog: 8,
            chainLog: 6,
            searchLog: 2,
            minMatch: 4,
            strategy: 6,
            ..Default::default()
        };
        let dict_base = 100u32;
        let curr = dict_base + 8;
        let match_index = dict_base + 4;
        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.window.dictBase_offset = dict_base;
        ms.window.lowLimit = dict_base;
        ms.window.dictLimit = dict_base + 20;
        ms.window.base_offset = ms.window.dictLimit;
        ms.dictContent = b"0000a000z00000000000".to_vec();
        ms.chainTable.resize(1 << cp.chainLog, 0);

        let bt_mask = (1u32 << (cp.chainLog - 1)) - 1;
        let curr_slot = (2 * (curr & bt_mask)) as usize;
        ms.chainTable[curr_slot] = match_index;

        let prefix = b"AAAAAAAAAAAAAAAA";
        ZSTD_insertDUBT1(
            &mut ms,
            prefix,
            curr,
            prefix.len(),
            1,
            dict_base,
            ZSTD_dictMode_e::ZSTD_extDict,
        );

        assert_eq!(ms.chainTable[curr_slot], match_index);
        assert_eq!(ms.chainTable[curr_slot + 1], 0);
    }

    fn lcg_bytes(len: usize) -> Vec<u8> {
        let mut x = 0x1234_5678u32;
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            out.extend_from_slice(&x.to_le_bytes());
        }
        out.truncate(len);
        out
    }

    #[test]
    fn no_dict_lazy_parser_enters_lazy_skipping_after_long_miss_run() {
        let cp = ZSTD_compressionParameters {
            windowLog: 17,
            hashLog: 12,
            chainLog: 12,
            searchLog: 4,
            minMatch: 4,
            strategy: 3,
            ..Default::default()
        };
        let src = lcg_bytes(4096);
        let mut ms = ZSTD_MatchState_t::new(cp);
        let mut seq = SeqStore_t::with_capacity(1024, src.len());
        let mut rep = [1u32, 4, 8];

        let last = ZSTD_compressBlock_lazy_noDict_generic(&mut ms, &mut seq, &mut rep, &src, 0, 0);

        assert_eq!(last, src.len());
        assert!(seq.sequences.is_empty());
        assert_eq!(ms.lazySkipping, 1);
    }

    #[test]
    fn no_dict_bt_parser_enters_lazy_skipping_after_long_miss_run() {
        let cp = ZSTD_compressionParameters {
            windowLog: 17,
            hashLog: 12,
            chainLog: 12,
            searchLog: 4,
            minMatch: 4,
            strategy: 6,
            ..Default::default()
        };
        let src = lcg_bytes(4096);
        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.chainTable.resize(1 << cp.chainLog, 0);
        let mut seq = SeqStore_t::with_capacity(1024, src.len());
        let mut rep = [1u32, 4, 8];

        let last = ZSTD_compressBlock_lazy_noDict_generic_search(
            &mut ms,
            &mut seq,
            &mut rep,
            &src,
            0,
            2,
            searchMethod_e::search_binaryTree,
        );

        assert_eq!(last, src.len());
        assert!(seq.sequences.is_empty());
        assert_eq!(ms.lazySkipping, 1);
    }
}
