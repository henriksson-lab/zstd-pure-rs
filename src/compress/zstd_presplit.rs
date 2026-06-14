//! Translation of `lib/compress/zstd_preSplit.c`.
//!
//! A compression-ratio heuristic that decides whether (and where) to
//! split a 128 KB block at a "content shift" boundary. Two strategies:
//!   - `splitBlock_fromBorders` (level 0): compare 512-byte
//!     fingerprints at the start and end of the block; if they differ
//!     enough, sample the middle and pick 32K / 64K / 96K.
//!   - `splitBlock_byChunks` (level 1-4): walk the block in 8 KB
//!     chunks, comparing each new chunk's fingerprint to the
//!     accumulated past-fingerprint until a threshold deviation is hit.
//!
//! A "fingerprint" is a histogram of 8/9/10-bit hashes of 2-byte
//! windows (or byte values when `hashLog == 8`).

#![allow(non_snake_case)]

use crate::compress::hist::HIST_add;

const THRESHOLD_PENALTY_RATE: u64 = 16;
const THRESHOLD_BASE: u64 = THRESHOLD_PENALTY_RATE - 2;
const THRESHOLD_PENALTY: i32 = 3;
const HASHLOG_MAX: u32 = 10;
const HASHTABLESIZE: usize = 1 << HASHLOG_MAX;
const KNUTH: u32 = 0x9e3779b9;

/// Port of `hash2`. For `hashLog==8` it returns the byte value; for
/// `hashLog > 8` it reads 2 bytes, multiplies by Knuth's constant,
/// and shifts down. Must be called with at least 2 bytes available.
#[inline(always)]
unsafe fn hash2(p: *const u8, hashLog: u32) -> usize {
    debug_assert!(hashLog >= 8);
    if hashLog == 8 {
        return unsafe { *p } as usize;
    }
    debug_assert!(hashLog <= HASHLOG_MAX);
    let v = unsafe { core::ptr::read_unaligned(p as *const u16) } as u32;
    ((v.wrapping_mul(KNUTH)) >> (32 - hashLog)) as usize
}

/// Port of `Fingerprint`.
#[derive(Debug, Clone)]
struct Fingerprint {
    events: [u32; HASHTABLESIZE],
    nbEvents: usize,
}

/// Port of `FPStats`.
#[derive(Debug, Clone)]
struct FPStats {
    pastEvents: Fingerprint,
    newEvents: Fingerprint,
}

/// Port of `initStats`.
fn initStats(fpstats: &mut FPStats) {
    *fpstats = FPStats {
        pastEvents: Fingerprint {
            events: [0u32; HASHTABLESIZE],
            nbEvents: 0,
        },
        newEvents: Fingerprint {
            events: [0u32; HASHTABLESIZE],
            nbEvents: 0,
        },
    }
}

/// Port of `addEvents` (zstd_preSplit.c). Walks `src` at stride
/// `samplingRate`, hashing each 2-byte window and incrementing the
/// matching bucket in `fp.events`.
#[inline(always)]
unsafe fn addEvents_generic(
    fp: &mut Fingerprint,
    src: *const u8,
    srcSize: usize,
    samplingRate: usize,
    hashLog: u32,
) {
    const HASHLENGTH: usize = 2;
    debug_assert!(srcSize >= HASHLENGTH);
    let limit = srcSize - HASHLENGTH + 1;
    let mut n = 0;
    while n < limit {
        let h = unsafe { hash2(src.add(n), hashLog) };
        fp.events[h] += 1;
        n += samplingRate;
    }
    fp.nbEvents += limit / samplingRate;
}

/// Port of `recordFingerprint`. Resets `fp` then populates it via
/// `addEvents_generic` — i.e. a fresh fingerprint of `src` under the
/// given sampling rate / hash width.
#[inline(always)]
unsafe fn recordFingerprint_generic(
    fp: &mut Fingerprint,
    src: *const u8,
    srcSize: usize,
    samplingRate: usize,
    hashLog: u32,
) {
    fp.events[..(1usize << hashLog)].fill(0);
    fp.nbEvents = 0;
    unsafe { addEvents_generic(fp, src, srcSize, samplingRate, hashLog) };
}

/// Port of `ZSTD_recordFingerprint_1`.
fn ZSTD_recordFingerprint_1(fp: &mut Fingerprint, src: *const u8, srcSize: usize) {
    unsafe { recordFingerprint_generic(fp, src, srcSize, 1, 10) }
}

/// Port of `ZSTD_recordFingerprint_5`.
fn ZSTD_recordFingerprint_5(fp: &mut Fingerprint, src: *const u8, srcSize: usize) {
    unsafe { recordFingerprint_generic(fp, src, srcSize, 5, 10) }
}

/// Port of `ZSTD_recordFingerprint_11`.
fn ZSTD_recordFingerprint_11(fp: &mut Fingerprint, src: *const u8, srcSize: usize) {
    unsafe { recordFingerprint_generic(fp, src, srcSize, 11, 9) }
}

/// Port of `ZSTD_recordFingerprint_43`.
fn ZSTD_recordFingerprint_43(fp: &mut Fingerprint, src: *const u8, srcSize: usize) {
    unsafe { recordFingerprint_generic(fp, src, srcSize, 43, 8) }
}

/// Port of `abs64`. Absolute value of a signed 64-bit difference,
/// returned as `u64` so wide histogram cross-products can sum without
/// risking signed overflow.
#[inline]
fn abs64(x: i64) -> u64 {
    x.unsigned_abs()
}

/// Port of `fpDistance`. The comparison metric is `|n1*e2 - n2*e1|`
/// summed per bucket — a cross-product that handles unequal event
/// counts without division.
fn fpDistance(fp1: &Fingerprint, fp2: &Fingerprint, hashLog: u32) -> u64 {
    let size = 1usize << hashLog;
    let mut distance: u64 = 0;
    for n in 0..size {
        let a = (fp1.events[n] as u64) * (fp2.nbEvents as u64);
        let b = (fp2.events[n] as u64) * (fp1.nbEvents as u64);
        distance = distance.wrapping_add(a.abs_diff(b));
    }
    distance
}

/// Port of `compareFingerprints`. Returns `true` when the new
/// fingerprint has deviated beyond threshold from the reference.
fn compareFingerprints(
    r#ref: &Fingerprint,
    newfp: &Fingerprint,
    penalty: i32,
    hashLog: u32,
) -> bool {
    debug_assert!(r#ref.nbEvents > 0);
    debug_assert!(newfp.nbEvents > 0);
    let p50 = (r#ref.nbEvents as u64) * (newfp.nbEvents as u64);
    let deviation = fpDistance(r#ref, newfp, hashLog);
    let threshold =
        p50.wrapping_mul((THRESHOLD_BASE as i32 + penalty) as u64) / THRESHOLD_PENALTY_RATE;
    deviation >= threshold
}

/// Port of `mergeEvents`.
#[allow(dead_code)]
fn mergeEvents(acc: &mut Fingerprint, newfp: &Fingerprint) {
    for n in 0..HASHTABLESIZE {
        acc.events[n] += newfp.events[n];
    }
    acc.nbEvents += newfp.nbEvents;
}

/// Port of `flushEvents`.
#[allow(dead_code)]
fn flushEvents(fpstats: &mut FPStats) {
    fpstats
        .pastEvents
        .events
        .copy_from_slice(&fpstats.newEvents.events);
    fpstats.pastEvents.nbEvents = fpstats.newEvents.nbEvents;
    fpstats.newEvents.events.fill(0);
    fpstats.newEvents.nbEvents = 0;
}

/// Port of `removeEvents`.
#[allow(dead_code)]
fn removeEvents(acc: &mut Fingerprint, slice: &Fingerprint) {
    for n in 0..HASHTABLESIZE {
        debug_assert!(acc.events[n] >= slice.events[n]);
        acc.events[n] -= slice.events[n];
    }
    acc.nbEvents -= slice.nbEvents;
}

const CHUNKSIZE: usize = 8 << 10;

type RecordEventsF = fn(&mut Fingerprint, *const u8, usize);

/// Port of `ZSTD_splitBlock_byChunks`. `level` selects the sampling
/// rate / hashLog pair:
///   - 0 → (rate=43, hashLog=8)   — cheapest
///   - 1 → (rate=11, hashLog=9)
///   - 2 → (rate=5,  hashLog=10)
///   - 3 → (rate=1,  hashLog=10)  — most accurate
fn ZSTD_splitBlock_byChunks(block: &[u8], level: i32) -> usize {
    const RECORDS_FS: [RecordEventsF; 4] = [
        ZSTD_recordFingerprint_43,
        ZSTD_recordFingerprint_11,
        ZSTD_recordFingerprint_5,
        ZSTD_recordFingerprint_1,
    ];
    const HASH_PARAMS: [u32; 4] = [8, 9, 10, 10];
    const FULL_BLOCK: usize = 128 << 10;

    debug_assert!((0..=3).contains(&level));
    debug_assert_eq!(block.len(), FULL_BLOCK);

    let level = level as usize;
    let record_f = RECORDS_FS[level];
    let hashLog = HASH_PARAMS[level];
    let mut fpstats = FPStats {
        pastEvents: Fingerprint {
            events: [0u32; HASHTABLESIZE],
            nbEvents: 0,
        },
        newEvents: Fingerprint {
            events: [0u32; HASHTABLESIZE],
            nbEvents: 0,
        },
    };
    let mut penalty = THRESHOLD_PENALTY;
    let block_ptr = block.as_ptr();

    initStats(&mut fpstats);
    record_f(&mut fpstats.pastEvents, block_ptr, CHUNKSIZE);

    let mut pos = CHUNKSIZE;
    while pos <= FULL_BLOCK - CHUNKSIZE {
        record_f(
            &mut fpstats.newEvents,
            unsafe { block_ptr.add(pos) },
            CHUNKSIZE,
        );
        if compareFingerprints(&fpstats.pastEvents, &fpstats.newEvents, penalty, hashLog) {
            return pos;
        }
        mergeEvents(&mut fpstats.pastEvents, &fpstats.newEvents);
        if penalty > 0 {
            penalty -= 1;
        }
        pos += CHUNKSIZE;
    }
    FULL_BLOCK
}

/// Port of `ZSTD_splitBlock_fromBorders`. Cheap heuristic: compare
/// 512-byte byte-histogram fingerprints at the block's start and
/// end; if they diverge, probe the middle and pick 32K/64K/96K split.
fn ZSTD_splitBlock_fromBorders(block: &[u8]) -> usize {
    const SEGMENT_SIZE: usize = 512;
    debug_assert_eq!(block.len(), 128 << 10);

    let mut fpstats = FPStats {
        pastEvents: Fingerprint {
            events: [0u32; HASHTABLESIZE],
            nbEvents: 0,
        },
        newEvents: Fingerprint {
            events: [0u32; HASHTABLESIZE],
            nbEvents: 0,
        },
    };
    initStats(&mut fpstats);
    HIST_add(&mut fpstats.pastEvents.events, &block[..SEGMENT_SIZE]);
    HIST_add(
        &mut fpstats.newEvents.events,
        &block[block.len() - SEGMENT_SIZE..],
    );
    fpstats.pastEvents.nbEvents = SEGMENT_SIZE;
    fpstats.newEvents.nbEvents = SEGMENT_SIZE;
    if !compareFingerprints(&fpstats.pastEvents, &fpstats.newEvents, 0, 8) {
        return block.len();
    }
    // Probe the middle.
    let mut middle = Fingerprint {
        events: [0u32; HASHTABLESIZE],
        nbEvents: 0,
    };
    let mid = block.len() / 2;
    HIST_add(
        &mut middle.events,
        &block[mid - SEGMENT_SIZE / 2..mid + SEGMENT_SIZE / 2],
    );
    middle.nbEvents = SEGMENT_SIZE;
    let distFromBegin = fpDistance(&fpstats.pastEvents, &middle, 8);
    let distFromEnd = fpDistance(&fpstats.newEvents, &middle, 8);
    let minDistance = (SEGMENT_SIZE as u64) * (SEGMENT_SIZE as u64) / 3;
    let delta = abs64(distFromBegin as i64 - distFromEnd as i64);
    if delta < minDistance {
        return 64 << 10;
    }
    if distFromBegin > distFromEnd {
        32 << 10
    } else {
        96 << 10
    }
}

/// Port of the public `ZSTD_splitBlock`. Dispatches on `level` (0..=4):
///   - level 0   → `splitBlock_fromBorders`
///   - level 1-4 → `splitBlock_byChunks` with internal level = level-1
pub fn ZSTD_splitBlock(block: &[u8], level: i32) -> usize {
    debug_assert!((0..=4).contains(&level));
    debug_assert_eq!(block.len(), 128 << 10);
    if level == 0 {
        ZSTD_splitBlock_fromBorders(block)
    } else {
        ZSTD_splitBlock_byChunks(block, level - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_block(fill: impl Fn(usize) -> u8) -> Vec<u8> {
        (0..128 << 10).map(fill).collect()
    }

    #[test]
    fn uniform_block_reports_no_split() {
        let block = mk_block(|_| 0x42);
        // Uniform content → similar fingerprints → no split.
        for level in 0..=4 {
            let split = ZSTD_splitBlock(&block, level);
            assert_eq!(
                split,
                block.len(),
                "level {level} suggested split at {split}"
            );
        }
    }

    #[test]
    fn two_halves_of_different_content_suggest_a_split() {
        // First half all zeros, second half pseudo-random. The
        // byChunks variant should detect the shift somewhere after
        // CHUNKSIZE (8 KB).
        let half = (128 << 10) / 2;
        let mut block = vec![0u8; 128 << 10];
        for (i, b) in block.iter_mut().enumerate().skip(half) {
            *b = (i as u8).wrapping_mul(31).wrapping_add(7);
        }
        // Level 3 is the most sensitive.
        let split = ZSTD_splitBlock(&block, 3);
        assert!(
            split < block.len(),
            "expected a split, got {split} == block.len()"
        );
        assert!(
            split >= CHUNKSIZE,
            "split {split} below first chunk boundary"
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn tiny_block_violates_upstream_assert_contract() {
        let block = vec![0u8; 1024];
        let _ = ZSTD_splitBlock(&block, 0);
    }

    #[test]
    fn hash2_passthrough_for_hashLog_8() {
        // At hashLog==8 the helper must return the raw byte value
        // without multiplying by KNUTH — this is a sampling fast
        // path that the presplit accuracy relies on.
        for b in 0u8..=255 {
            let buf = [b, 0u8];
            assert_eq!(unsafe { hash2(buf.as_ptr(), 8) }, b as usize);
        }
    }

    #[test]
    fn hash2_stable_across_invocations_for_same_input() {
        // Deterministic: same 2-byte input + same hashLog → same hash.
        let buf = [0xAB, 0xCD];
        let h1 = unsafe { hash2(buf.as_ptr(), 10) };
        let h2 = unsafe { hash2(buf.as_ptr(), 10) };
        assert_eq!(h1, h2);
        // Different hashLog within the valid range (8..=10) yields
        // a different output — KNUTH mult + right shift respond to
        // the shift amount.
        assert_ne!(unsafe { hash2(buf.as_ptr(), 10) }, unsafe {
            hash2(buf.as_ptr(), 9)
        });
        // Hash fits within the 1<<hashLog range.
        assert!(h1 < (1usize << 10));
    }

    #[test]
    fn addEvents_counts_partial_final_stride_like_c_loop() {
        let mut fp = Fingerprint {
            events: [0u32; HASHTABLESIZE],
            nbEvents: 0,
        };
        let src = [7; CHUNKSIZE];
        unsafe { addEvents_generic(&mut fp, src.as_ptr(), src.len(), 43, 8) };
        assert_eq!(fp.nbEvents, (CHUNKSIZE - 1) / 43);
        assert_eq!(fp.events[7] as usize, (CHUNKSIZE - 1).div_ceil(43));
    }

    #[test]
    fn flushEvents_copies_new_into_past_and_zeros_new() {
        let mut stats = FPStats {
            pastEvents: Fingerprint {
                events: [0u32; HASHTABLESIZE],
                nbEvents: 0,
            },
            newEvents: Fingerprint {
                events: [0u32; HASHTABLESIZE],
                nbEvents: 0,
            },
        };
        stats.newEvents.events[7] = 3;
        stats.newEvents.events[99] = 11;
        stats.newEvents.nbEvents = 14;

        flushEvents(&mut stats);

        assert_eq!(stats.pastEvents.events[7], 3);
        assert_eq!(stats.pastEvents.events[99], 11);
        assert_eq!(stats.pastEvents.nbEvents, 14);
        assert_eq!(stats.newEvents.nbEvents, 0);
        assert!(stats.newEvents.events.iter().all(|&x| x == 0));
    }

    #[test]
    fn removeEvents_subtracts_bucketwise() {
        let mut acc = Fingerprint {
            events: [0u32; HASHTABLESIZE],
            nbEvents: 0,
        };
        let mut slice = Fingerprint {
            events: [0u32; HASHTABLESIZE],
            nbEvents: 0,
        };
        acc.events[3] = 9;
        acc.events[8] = 5;
        acc.nbEvents = 14;
        slice.events[3] = 4;
        slice.events[8] = 1;
        slice.nbEvents = 5;

        removeEvents(&mut acc, &slice);

        assert_eq!(acc.events[3], 5);
        assert_eq!(acc.events[8], 4);
        assert_eq!(acc.nbEvents, 9);
    }
}
