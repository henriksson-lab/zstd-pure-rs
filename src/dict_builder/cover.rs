//! Translation of `lib/dictBuilder/cover.c` (the COVER dictionary
//! trainer). In progress — translated bottom-up.
//!
//! Shared structs mirror `cover.h`. The `ZDICT_*` parameter structs come
//! from [`super::zdict`].

#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]

use crate::common::bits::ZSTD_highbit32;

/* ======================================================================== */
/* Shared structs (cover.h)                                                  */
/* ======================================================================== */

/// Port of `COVER_segment_t`. A range in the source plus its score.
#[derive(Debug, Clone, Copy, Default)]
pub struct COVER_segment_t {
    pub begin: u32,
    pub end: u32,
    pub score: u32,
}

/// Port of `COVER_epoch_info_t`. Number of epochs and size of each.
#[derive(Debug, Clone, Copy, Default)]
pub struct COVER_epoch_info_t {
    pub num: u32,
    pub size: u32,
}

/// Port of `COVER_dictSelection_t`. C stores `BYTE* dictContent` (malloc'd,
/// freed via `COVER_dictSelectionFree`); the Rust port owns the buffer as a
/// `Vec<u8>` (auto-freed on drop). `dictContent.is_empty()` is the NULL
/// sentinel for the error case.
#[derive(Debug, Default, Clone)]
pub struct COVER_dictSelection_t {
    pub dictContent: Vec<u8>,
    pub dictSize: usize,
    pub totalCompressedSize: usize,
}

/* ======================================================================== */
/* Hash table — a small specialized map for storing activeDmers              */
/* ======================================================================== */
//
// The map does not resize, so if it becomes full it loops forever; thus it
// must be large enough to store every value. Linear probing, load < 0.5.

pub const MAP_EMPTY_VALUE: u32 = u32::MAX; // (U32)-1

/// Port of `COVER_map_pair_t`.
#[derive(Debug, Clone, Copy)]
pub struct COVER_map_pair_t {
    pub key: u32,
    pub value: u32,
}

/// Port of `COVER_map_t`.
pub struct COVER_map_t {
    pub data: Vec<COVER_map_pair_t>,
    pub sizeLog: u32,
    pub size: u32,
    pub sizeMask: u32,
}

impl Default for COVER_map_t {
    fn default() -> Self {
        COVER_map_t {
            data: Vec::new(),
            sizeLog: 0,
            size: 0,
            sizeMask: 0,
        }
    }
}

/// Port of `COVER_map_clear`.
pub fn COVER_map_clear(map: &mut COVER_map_t) {
    // memset(map->data, MAP_EMPTY_VALUE, map->size * sizeof(pair)) — every
    // byte 0xFF makes each U32 field == MAP_EMPTY_VALUE.
    for pair in map.data.iter_mut() {
        pair.key = MAP_EMPTY_VALUE;
        pair.value = MAP_EMPTY_VALUE;
    }
}

/// Port of `COVER_map_init`. Returns 1 on success, 0 on failure.
pub fn COVER_map_init(map: &mut COVER_map_t, size: u32) -> i32 {
    map.sizeLog = ZSTD_highbit32(size) + 2;
    map.size = 1u32 << map.sizeLog;
    map.sizeMask = map.size - 1;
    // malloc can't fail-soft in Rust; vec! aborts on OOM. Mirror success.
    map.data = vec![
        COVER_map_pair_t {
            key: MAP_EMPTY_VALUE,
            value: MAP_EMPTY_VALUE
        };
        map.size as usize
    ];
    1
}

/// Port of `COVER_prime4bytes`.
const COVER_prime4bytes: u32 = 2654435761;

/// Port of `COVER_map_hash`.
fn COVER_map_hash(map: &COVER_map_t, key: u32) -> u32 {
    key.wrapping_mul(COVER_prime4bytes) >> (32 - map.sizeLog)
}

/// Port of `COVER_map_index`. Index a key should be placed into.
fn COVER_map_index(map: &COVER_map_t, key: u32) -> u32 {
    let hash = COVER_map_hash(map, key);
    let mut i = hash;
    loop {
        let pos = &map.data[i as usize];
        if pos.value == MAP_EMPTY_VALUE {
            return i;
        }
        if pos.key == key {
            return i;
        }
        i = (i + 1) & map.sizeMask;
    }
}

/// Port of `COVER_map_at`. Returns the index of the slot for `key`,
/// inserting it (value 0) if absent. (C returns `U32*`; we return the
/// index since the map owns its storage — callers read/write
/// `map.data[idx].value`.) The map must not be full.
pub fn COVER_map_at(map: &mut COVER_map_t, key: u32) -> u32 {
    let idx = COVER_map_index(map, key);
    let pos = &mut map.data[idx as usize];
    if pos.value == MAP_EMPTY_VALUE {
        pos.key = key;
        pos.value = 0;
    }
    idx
}

/// Port of `COVER_map_remove`. Deletes `key` from the map if present.
pub fn COVER_map_remove(map: &mut COVER_map_t, key: u32) {
    let mut i = COVER_map_index(map, key);
    let mut del = i;
    let mut shift: u32 = 1;
    if map.data[del as usize].value == MAP_EMPTY_VALUE {
        return;
    }
    i = (i + 1) & map.sizeMask;
    loop {
        if map.data[i as usize].value == MAP_EMPTY_VALUE {
            map.data[del as usize].value = MAP_EMPTY_VALUE;
            return;
        }
        // If pos can be moved to del do so.
        let pos_key = map.data[i as usize].key;
        if ((i.wrapping_sub(COVER_map_hash(map, pos_key))) & map.sizeMask) >= shift {
            let pos_val = map.data[i as usize].value;
            map.data[del as usize].key = pos_key;
            map.data[del as usize].value = pos_val;
            del = i;
            shift = 1;
        } else {
            shift += 1;
        }
        i = (i + 1) & map.sizeMask;
    }
}

/// Port of `COVER_map_destroy`.
pub fn COVER_map_destroy(map: &mut COVER_map_t) {
    map.data = Vec::new();
    map.size = 0;
}

/* ======================================================================== */
/* Helper functions                                                          */
/* ======================================================================== */

/// Port of `COVER_sum`. Returns the sum of the sample sizes.
pub fn COVER_sum(samplesSizes: &[usize], nbSamples: u32) -> usize {
    let mut sum: usize = 0;
    for i in 0..nbSamples as usize {
        sum += samplesSizes[i];
    }
    sum
}

/* ======================================================================== */
/* Context                                                                   */
/* ======================================================================== */

use crate::common::error::{ErrorCode, ERROR};
use crate::dict_builder::zdict::ZDICT_cover_params_t;

/// `COVER_MAX_SAMPLES_SIZE` on 64-bit (`sizeof(size_t) == 8`): `(unsigned)-1`.
const COVER_MAX_SAMPLES_SIZE: usize = u32::MAX as usize;
/// `COVER_DEFAULT_SPLITPOINT`.
pub const COVER_DEFAULT_SPLITPOINT: f64 = 1.0;

/// Port of the anonymous `COVER_ctx_t` struct. Owned scratch arrays are
/// `Vec`s (C `malloc`s them); `samples` / `samplesSizes` are raw pointers
/// into the caller's buffers, as in C.
pub struct COVER_ctx_t {
    pub samples: *const u8,
    pub offsets: Vec<usize>,
    pub samplesSizes: *const usize,
    pub nbSamples: usize,
    pub nbTrainSamples: usize,
    pub nbTestSamples: usize,
    pub suffix: Vec<u32>,
    pub suffixSize: usize,
    pub freqs: Vec<u32>,
    pub dmerAt: Vec<u32>,
    pub d: u32,
    pub displayLevel: i32,
}

impl Default for COVER_ctx_t {
    fn default() -> Self {
        COVER_ctx_t {
            samples: core::ptr::null(),
            offsets: Vec::new(),
            samplesSizes: core::ptr::null(),
            nbSamples: 0,
            nbTrainSamples: 0,
            nbTestSamples: 0,
            suffix: Vec::new(),
            suffixSize: 0,
            freqs: Vec::new(),
            dmerAt: Vec::new(),
            d: 0,
            displayLevel: 0,
        }
    }
}

/// Rust-only helper: C `memcmp`.
#[inline]
unsafe fn memcmp(a: *const u8, b: *const u8, n: usize) -> i32 {
    let mut i = 0;
    while i < n {
        let x = *a.add(i);
        let y = *b.add(i);
        if x != y {
            return x as i32 - y as i32;
        }
        i += 1;
    }
    0
}

/// Rust-only helper: little-endian 8-byte read (`MEM_readLE64`).
#[inline]
unsafe fn mem_read_le64(p: *const u8) -> u64 {
    (p as *const u64).read_unaligned().to_le()
}

/// Port of `COVER_cmp`. -1/0/1 by the first `ctx.d` bytes of the dmers.
unsafe fn COVER_cmp(ctx: &COVER_ctx_t, lp: *const u8, rp: *const u8) -> i32 {
    let lhs = *(lp as *const u32);
    let rhs = *(rp as *const u32);
    memcmp(
        ctx.samples.add(lhs as usize),
        ctx.samples.add(rhs as usize),
        ctx.d as usize,
    )
}

/// Port of `COVER_cmp8`. Faster version for `d <= 8`.
unsafe fn COVER_cmp8(ctx: &COVER_ctx_t, lp: *const u8, rp: *const u8) -> i32 {
    let mask: u64 = if ctx.d == 8 {
        u64::MAX
    } else {
        (1u64 << (8 * ctx.d)) - 1
    };
    let lhs = mem_read_le64(ctx.samples.add(*(lp as *const u32) as usize)) & mask;
    let rhs = mem_read_le64(ctx.samples.add(*(rp as *const u32) as usize)) & mask;
    if lhs < rhs {
        -1
    } else {
        (lhs > rhs) as i32
    }
}

/// Port of `stableSort`. Upstream uses `qsort_r` with `COVER_strict_cmp`
/// (ties broken by element address); the documented intent (cover.c:695)
/// is a stable sort by dmer content. We use Rust's stable `sort_by` on the
/// content comparator — equal dmers keep their input-position order (the
/// suffix array is initialized to `[0, 1, 2, …]`).
fn stableSort(ctx: &mut COVER_ctx_t) {
    let samples = ctx.samples;
    let d = ctx.d;
    ctx.suffix.sort_by(|&a, &b| unsafe {
        if d <= 8 {
            let mask: u64 = if d == 8 { u64::MAX } else { (1u64 << (8 * d)) - 1 };
            let lhs = mem_read_le64(samples.add(a as usize)) & mask;
            let rhs = mem_read_le64(samples.add(b as usize)) & mask;
            lhs.cmp(&rhs)
        } else {
            let la = core::slice::from_raw_parts(samples.add(a as usize), d as usize);
            let lb = core::slice::from_raw_parts(samples.add(b as usize), d as usize);
            la.cmp(lb)
        }
    });
}

/// Port of `COVER_lower_bound`.
unsafe fn COVER_lower_bound(first: *const usize, last: *const usize, value: usize) -> *const usize {
    let mut first = first;
    let mut count = last.offset_from(first) as usize;
    while count != 0 {
        let step = count / 2;
        let mut ptr = first.add(step);
        if *ptr < value {
            ptr = ptr.add(1);
            first = ptr;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    first
}

/// Port of `COVER_groupBy`. Groups an array sorted by `cmp` into groups
/// with equivalent values, calling `grp` for each group.
unsafe fn COVER_groupBy(
    data: *const u8,
    count: usize,
    size: usize,
    ctx: &mut COVER_ctx_t,
    cmp: unsafe fn(&COVER_ctx_t, *const u8, *const u8) -> i32,
    grp: unsafe fn(&mut COVER_ctx_t, *const u8, *const u8),
) {
    let mut ptr = data;
    let mut num: usize = 0;
    while num < count {
        let mut grpEnd = ptr.add(size);
        num += 1;
        while num < count && cmp(ctx, ptr, grpEnd) == 0 {
            grpEnd = grpEnd.add(size);
            num += 1;
        }
        grp(ctx, ptr, grpEnd);
        ptr = grpEnd;
    }
}

/// Port of `COVER_group`. Called on each group of positions with the same
/// dmer; counts the frequency of each dmer and saves it in the suffix
/// array, filling `ctx.dmerAt`.
unsafe fn COVER_group(ctx: &mut COVER_ctx_t, group: *const u8, groupEnd: *const u8) {
    let suffix_base = ctx.suffix.as_mut_ptr();
    let dmerAt_base = ctx.dmerAt.as_mut_ptr();
    let offsets_base = ctx.offsets.as_ptr();
    let nbSamples = ctx.nbSamples;

    let mut grpPtr = group as *const u32;
    let grpEnd = groupEnd as *const u32;
    let dmerId: u32 = grpPtr.offset_from(suffix_base) as u32;
    let mut freq: u32 = 0;
    let mut curOffsetPtr = offsets_base;
    let offsetsEnd = offsets_base.add(nbSamples);
    let mut curSampleEnd = *offsets_base; // ctx->offsets[0]
    while grpPtr != grpEnd {
        let pos = *grpPtr;
        *dmerAt_base.add(pos as usize) = dmerId;
        if (pos as usize) < curSampleEnd {
            grpPtr = grpPtr.add(1);
            continue;
        }
        freq += 1;
        if grpPtr.add(1) != grpEnd {
            let sampleEndPtr = COVER_lower_bound(curOffsetPtr, offsetsEnd, pos as usize);
            curSampleEnd = *sampleEndPtr;
            curOffsetPtr = sampleEndPtr.add(1);
        }
        grpPtr = grpPtr.add(1);
    }
    *suffix_base.add(dmerId as usize) = freq;
}

/* ======================================================================== */
/* Cover functions                                                           */
/* ======================================================================== */

/// Port of `COVER_selectSegment`. Selects the best segment in an epoch.
/// `freqs` is the (mutable) per-dmer frequency array.
fn COVER_selectSegment(
    ctx: &COVER_ctx_t,
    freqs: &mut [u32],
    activeDmers: &mut COVER_map_t,
    begin: u32,
    end: u32,
    parameters: ZDICT_cover_params_t,
) -> COVER_segment_t {
    let k = parameters.k;
    let d = parameters.d;
    let dmersInK = k - d + 1;
    let mut bestSegment = COVER_segment_t {
        begin: 0,
        end: 0,
        score: 0,
    };
    let mut activeSegment = COVER_segment_t {
        begin,
        end: begin,
        score: 0,
    };
    /* Reset the activeDmers in the segment. */
    COVER_map_clear(activeDmers);
    while activeSegment.end < end {
        let newDmer = ctx.dmerAt[activeSegment.end as usize];
        let newDmerOcc = COVER_map_at(activeDmers, newDmer);
        if activeDmers.data[newDmerOcc as usize].value == 0 {
            activeSegment.score += freqs[newDmer as usize];
        }
        activeSegment.end += 1;
        activeDmers.data[newDmerOcc as usize].value += 1;

        if activeSegment.end - activeSegment.begin == dmersInK + 1 {
            let delDmer = ctx.dmerAt[activeSegment.begin as usize];
            let delDmerOcc = COVER_map_at(activeDmers, delDmer);
            activeSegment.begin += 1;
            activeDmers.data[delDmerOcc as usize].value -= 1;
            if activeDmers.data[delDmerOcc as usize].value == 0 {
                COVER_map_remove(activeDmers, delDmer);
                activeSegment.score -= freqs[delDmer as usize];
            }
        }

        if activeSegment.score > bestSegment.score {
            bestSegment = activeSegment;
        }
    }
    {
        /* Trim off the zero frequency head and tail from the segment. */
        let mut newBegin = bestSegment.end;
        let mut newEnd = bestSegment.begin;
        let mut pos = bestSegment.begin;
        while pos != bestSegment.end {
            let freq = freqs[ctx.dmerAt[pos as usize] as usize];
            if freq != 0 {
                newBegin = core::cmp::min(newBegin, pos);
                newEnd = pos + 1;
            }
            pos += 1;
        }
        bestSegment.begin = newBegin;
        bestSegment.end = newEnd;
    }
    {
        /* Zero out the frequency of each dmer covered by the chosen segment. */
        let mut pos = bestSegment.begin;
        while pos != bestSegment.end {
            freqs[ctx.dmerAt[pos as usize] as usize] = 0;
            pos += 1;
        }
    }
    bestSegment
}

/// Port of `COVER_checkParameters`. Returns non-zero if valid.
fn COVER_checkParameters(parameters: ZDICT_cover_params_t, maxDictSize: usize) -> i32 {
    if parameters.d == 0 || parameters.k == 0 {
        return 0;
    }
    if parameters.k as usize > maxDictSize {
        return 0;
    }
    if parameters.d > parameters.k {
        return 0;
    }
    if parameters.splitPoint <= 0.0 || parameters.splitPoint > 1.0 {
        return 0;
    }
    1
}

/// Port of `COVER_ctx_destroy`.
fn COVER_ctx_destroy(ctx: &mut COVER_ctx_t) {
    ctx.suffix = Vec::new();
    ctx.freqs = Vec::new();
    ctx.dmerAt = Vec::new();
    ctx.offsets = Vec::new();
}

/// Port of `COVER_ctx_init`. Returns 0 on success or an error code.
/// `samples` / `samplesSizes` are borrowed by raw pointer into `ctx` for
/// the lifetime of subsequent use (caller must keep them alive).
fn COVER_ctx_init(
    ctx: &mut COVER_ctx_t,
    samples: &[u8],
    samplesSizes: &[usize],
    nbSamples: u32,
    d: u32,
    splitPoint: f64,
    displayLevel: i32,
) -> usize {
    let totalSamplesSize = COVER_sum(samplesSizes, nbSamples);
    let nbTrainSamples = if splitPoint < 1.0 {
        (nbSamples as f64 * splitPoint) as u32
    } else {
        nbSamples
    };
    let nbTestSamples = if splitPoint < 1.0 {
        nbSamples - nbTrainSamples
    } else {
        nbSamples
    };
    let trainingSamplesSize = if splitPoint < 1.0 {
        COVER_sum(samplesSizes, nbTrainSamples)
    } else {
        totalSamplesSize
    };
    let _testSamplesSize = if splitPoint < 1.0 {
        COVER_sum(&samplesSizes[nbTrainSamples as usize..], nbTestSamples)
    } else {
        totalSamplesSize
    };
    ctx.displayLevel = displayLevel;
    /* Checks */
    if totalSamplesSize < core::cmp::max(d as usize, core::mem::size_of::<u64>())
        || totalSamplesSize >= COVER_MAX_SAMPLES_SIZE
    {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    if nbTrainSamples < 5 {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    if nbTestSamples < 1 {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    /* Zero the context (fresh fields). */
    *ctx = COVER_ctx_t::default();
    ctx.displayLevel = displayLevel;
    ctx.samples = samples.as_ptr();
    ctx.samplesSizes = samplesSizes.as_ptr();
    ctx.nbSamples = nbSamples as usize;
    ctx.nbTrainSamples = nbTrainSamples as usize;
    ctx.nbTestSamples = nbTestSamples as usize;
    /* Partial suffix array */
    ctx.suffixSize = trainingSamplesSize - core::cmp::max(d as usize, core::mem::size_of::<u64>()) + 1;
    ctx.suffix = vec![0u32; ctx.suffixSize];
    ctx.dmerAt = vec![0u32; ctx.suffixSize];
    ctx.offsets = vec![0usize; nbSamples as usize + 1];
    ctx.freqs = Vec::new();
    ctx.d = d;

    /* Fill offsets from the samplesSizes. */
    ctx.offsets[0] = 0;
    for i in 1..=nbSamples as usize {
        ctx.offsets[i] = ctx.offsets[i - 1] + samplesSizes[i - 1];
    }

    /* Construct the partial suffix array. */
    for i in 0..ctx.suffixSize {
        ctx.suffix[i] = i as u32;
    }
    stableSort(ctx);

    /* Compute frequencies via groupBy. */
    let cmp: unsafe fn(&COVER_ctx_t, *const u8, *const u8) -> i32 =
        if ctx.d <= 8 { COVER_cmp8 } else { COVER_cmp };
    unsafe {
        let data = ctx.suffix.as_ptr() as *const u8;
        COVER_groupBy(data, ctx.suffixSize, core::mem::size_of::<u32>(), ctx, cmp, COVER_group);
    }
    /* ctx->freqs = ctx->suffix; ctx->suffix = NULL; */
    ctx.freqs = core::mem::take(&mut ctx.suffix);
    0
}

/// Port of `COVER_warnOnSmallCorpus`. (Diagnostic print omitted.)
pub fn COVER_warnOnSmallCorpus(maxDictSize: usize, nbDmers: usize, displayLevel: i32) {
    let _ = displayLevel;
    let ratio = nbDmers as f64 / maxDictSize as f64;
    if ratio >= 10.0 {
        // ok
    }
    // else: upstream warns that the corpus is too small for the dict size.
}

/// Port of `COVER_computeEpochs`. Computes the number of epochs and the
/// size of each epoch (each epoch gets at least `10 * k` bytes).
pub fn COVER_computeEpochs(maxDictSize: u32, nbDmers: u32, k: u32, passes: u32) -> COVER_epoch_info_t {
    let minEpochSize = k * 10;
    let mut epochs = COVER_epoch_info_t { num: 0, size: 0 };
    epochs.num = core::cmp::max(1, maxDictSize / k / passes);
    epochs.size = nbDmers / epochs.num;
    if epochs.size >= minEpochSize {
        return epochs;
    }
    epochs.size = core::cmp::min(minEpochSize, nbDmers);
    epochs.num = nbDmers / epochs.size;
    epochs
}

/// Port of `COVER_checkTotalCompressedSize`. Compresses every check
/// sample with a CDict built from `dict` and returns `dict.len()` plus the
/// sum of compressed sizes (or a zstd error). `dict` is the finalized
/// dictionary content (its length is the C `dictBufferCapacity` argument).
#[allow(clippy::too_many_arguments)]
pub fn COVER_checkTotalCompressedSize(
    parameters: ZDICT_cover_params_t,
    samplesSizes: &[usize],
    samples: &[u8],
    offsets: &[usize],
    nbTrainSamples: usize,
    nbSamples: usize,
    dict: &[u8],
) -> usize {
    use crate::common::error::ERR_isError;
    use crate::compress::zstd_compress::{
        ZSTD_compressBound, ZSTD_compress_usingCDict, ZSTD_createCCtx, ZSTD_createCDict,
    };

    let mut totalCompressedSize = ERROR(ErrorCode::Generic);
    /* Allocate dst with enough space to compress the maximum sized sample. */
    let mut maxSampleSize: usize = 0;
    let mut i = if parameters.splitPoint < 1.0 { nbTrainSamples } else { 0 };
    while i < nbSamples {
        maxSampleSize = core::cmp::max(samplesSizes[i], maxSampleSize);
        i += 1;
    }
    let dstCapacity = ZSTD_compressBound(maxSampleSize);
    let mut dst = vec![0u8; dstCapacity];
    /* Create the cctx and cdict. (Box-owned; freed on drop — the C
     * `_compressCleanup` goto becomes early-return / scope exit.) */
    let cctx = ZSTD_createCCtx();
    let cdict = ZSTD_createCDict(dict, parameters.zParams.compressionLevel);
    let (mut cctx, cdict) = match (cctx, cdict) {
        (Some(c), Some(d)) => (c, d),
        _ => return totalCompressedSize,
    };
    /* Compress each sample and sum their sizes (or error). */
    totalCompressedSize = dict.len();
    i = if parameters.splitPoint < 1.0 { nbTrainSamples } else { 0 };
    while i < nbSamples {
        let src = &samples[offsets[i]..offsets[i] + samplesSizes[i]];
        let size = ZSTD_compress_usingCDict(&mut cctx, &mut dst, src, &cdict);
        if ERR_isError(size) {
            return size;
        }
        totalCompressedSize += size;
        i += 1;
    }
    totalCompressedSize
}

/// Port of `COVER_buildDictionary`. Given a prepared context, builds the
/// dictionary into `dict`; returns the `tail` offset (the dict content
/// occupies `dict[tail..]`).
fn COVER_buildDictionary(
    ctx: &COVER_ctx_t,
    freqs: &mut [u32],
    activeDmers: &mut COVER_map_t,
    dict: &mut [u8],
    dictBufferCapacity: usize,
    parameters: ZDICT_cover_params_t,
) -> usize {
    let mut tail = dictBufferCapacity;
    /* Divide the data into epochs; select one segment from each. */
    let epochs = COVER_computeEpochs(
        dictBufferCapacity as u32,
        ctx.suffixSize as u32,
        parameters.k,
        4,
    );
    let maxZeroScoreRun = core::cmp::max(10usize, core::cmp::min(100usize, (epochs.num >> 3) as usize));
    let mut zeroScoreRun: usize = 0;
    let mut epoch: usize = 0;
    while tail > 0 {
        let epochBegin = (epoch as u32).wrapping_mul(epochs.size);
        let epochEnd = epochBegin + epochs.size;
        /* Select a segment. */
        let segment = COVER_selectSegment(ctx, freqs, activeDmers, epochBegin, epochEnd, parameters);
        /* If the segment covers no dmers, we may be out of content. */
        if segment.score == 0 {
            zeroScoreRun += 1;
            if zeroScoreRun >= maxZeroScoreRun {
                break;
            }
            epoch = (epoch + 1) % epochs.num as usize;
            continue;
        }
        zeroScoreRun = 0;
        /* Trim the segment if necessary; if too small we are done. */
        let segmentSize = core::cmp::min(
            (segment.end - segment.begin + parameters.d - 1) as usize,
            tail,
        );
        if segmentSize < parameters.d as usize {
            break;
        }
        /* Fill the dictionary from the back so the best segments get the
         * smallest offsets. */
        tail -= segmentSize;
        // memcpy(dict + tail, ctx->samples + segment.begin, segmentSize)
        unsafe {
            let src = core::slice::from_raw_parts(ctx.samples.add(segment.begin as usize), segmentSize);
            dict[tail..tail + segmentSize].copy_from_slice(src);
        }
        epoch = (epoch + 1) % epochs.num as usize;
    }
    tail
}

/// Port of `COVER_best_t`. Saves the best parameters and dictionary across
/// the optimize loop. Upstream also uses it to synchronize worker threads
/// via a mutex/cond; this port runs the optimize loop sequentially (the
/// chosen dictionary is deterministic regardless of thread count), so the
/// mutex/cond and `liveJobs` bookkeeping collapse to plain fields.
pub struct COVER_best_t {
    pub liveJobs: usize,
    pub dict: Vec<u8>,
    pub dictSize: usize,
    pub parameters: ZDICT_cover_params_t,
    pub compressedSize: usize,
}

impl Default for COVER_best_t {
    fn default() -> Self {
        COVER_best_t {
            liveJobs: 0,
            dict: Vec::new(),
            dictSize: 0,
            parameters: ZDICT_cover_params_t::default(),
            compressedSize: usize::MAX, // (size_t)-1
        }
    }
}

/// Port of `COVER_best_init`.
pub fn COVER_best_init(best: &mut COVER_best_t) {
    best.liveJobs = 0;
    best.dict = Vec::new();
    best.dictSize = 0;
    best.compressedSize = usize::MAX;
    best.parameters = ZDICT_cover_params_t::default();
}

/// Port of `COVER_best_wait`. (No-op: optimize runs sequentially.)
pub fn COVER_best_wait(_best: &COVER_best_t) {}

/// Port of `COVER_best_destroy`. (No-op: `dict` drops with the struct.)
pub fn COVER_best_destroy(best: &mut COVER_best_t) {
    COVER_best_wait(best);
    best.dict = Vec::new();
}

/// Port of `COVER_best_start`.
pub fn COVER_best_start(best: &mut COVER_best_t) {
    best.liveJobs += 1;
}

/// Port of `COVER_best_finish`. If `selection` is the best so far, save its
/// dictionary and parameters. (Takes `selection` by ref; the caller frees it
/// afterwards, as in C.)
pub fn COVER_best_finish(
    best: &mut COVER_best_t,
    parameters: ZDICT_cover_params_t,
    selection: &COVER_dictSelection_t,
) {
    let compressedSize = selection.totalCompressedSize;
    let dictSize = selection.dictSize;
    if best.liveJobs > 0 {
        best.liveJobs -= 1;
    }
    /* If the new dictionary is better */
    if compressedSize < best.compressedSize {
        /* Save the dictionary, parameters, and size (skip if dict is NULL). */
        if !selection.dictContent.is_empty() {
            best.dict = selection.dictContent[..dictSize].to_vec();
            best.dictSize = dictSize;
            best.parameters = parameters;
            best.compressedSize = compressedSize;
        }
    }
}

/// Port of `setDictSelection`.
fn setDictSelection(buf: Vec<u8>, s: usize, csz: usize) -> COVER_dictSelection_t {
    COVER_dictSelection_t {
        dictContent: buf,
        dictSize: s,
        totalCompressedSize: csz,
    }
}

/// Port of `COVER_dictSelectionError`.
pub fn COVER_dictSelectionError(error: usize) -> COVER_dictSelection_t {
    setDictSelection(Vec::new(), 0, error)
}

/// Port of `COVER_dictSelectionIsError`.
pub fn COVER_dictSelectionIsError(selection: &COVER_dictSelection_t) -> u32 {
    (crate::common::error::ERR_isError(selection.totalCompressedSize)
        || selection.dictContent.is_empty()) as u32
}

/// Port of `COVER_dictSelectionFree`. (Owned `Vec` drops here.)
pub fn COVER_dictSelectionFree(selection: COVER_dictSelection_t) {
    drop(selection);
}

/// Port of `COVER_selectDict`. Finalizes the dictionary, and (if
/// `shrinkDict`) searches for the smallest dictionary within the allowed
/// regression of the largest dictionary's compressed size.
#[allow(clippy::too_many_arguments)]
pub fn COVER_selectDict(
    customDictContent: &[u8],
    dictBufferCapacity: usize,
    mut dictContentSize: usize,
    samplesBuffer: &[u8],
    samplesSizes: &[usize],
    nbFinalizeSamples: u32,
    nbCheckSamples: usize,
    nbSamples: usize,
    params: ZDICT_cover_params_t,
    offsets: &[usize],
    mut totalCompressedSize: usize,
) -> COVER_dictSelection_t {
    use crate::common::error::ERR_isError;
    use crate::dict_builder::zdict::{ZDICT_finalizeDictionary, ZDICT_DICTSIZE_MIN};

    let _ = totalCompressedSize; // overwritten below
    let largestDict;
    let largestCompressed;
    let origContentSize = dictContentSize;
    let mut largestDictbuffer = vec![0u8; dictBufferCapacity];
    let mut candidateDictBuffer = vec![0u8; dictBufferCapacity];
    let regressionTolerance = (params.shrinkDictMaxRegression as f64 / 100.0) + 1.00;

    /* Initial dictionary size and compressed size */
    largestDictbuffer[..dictContentSize].copy_from_slice(&customDictContent[..dictContentSize]);
    dictContentSize = ZDICT_finalizeDictionary(
        &mut largestDictbuffer,
        dictBufferCapacity,
        customDictContent,
        dictContentSize,
        samplesBuffer,
        samplesSizes,
        nbFinalizeSamples,
        params.zParams,
    );
    if ERR_isError(dictContentSize) {
        return COVER_dictSelectionError(dictContentSize);
    }

    totalCompressedSize = COVER_checkTotalCompressedSize(
        params,
        samplesSizes,
        samplesBuffer,
        offsets,
        nbCheckSamples,
        nbSamples,
        &largestDictbuffer[..dictContentSize],
    );
    if ERR_isError(totalCompressedSize) {
        return COVER_dictSelectionError(totalCompressedSize);
    }

    if params.shrinkDict == 0 {
        return setDictSelection(largestDictbuffer, dictContentSize, totalCompressedSize);
    }

    largestDict = dictContentSize;
    largestCompressed = totalCompressedSize;
    dictContentSize = ZDICT_DICTSIZE_MIN;

    /* Largest dict is initially at least ZDICT_DICTSIZE_MIN */
    while dictContentSize < largestDict {
        candidateDictBuffer[..largestDict].copy_from_slice(&largestDictbuffer[..largestDict]);
        // C: customDictContentEnd - dictContentSize == last `content_size` bytes
        // of the content. Clamp to the available content to avoid the C
        // underflow read when content_size > origContentSize (shrink edge).
        let content_size = core::cmp::min(dictContentSize, origContentSize);
        let content = &customDictContent[origContentSize - content_size..origContentSize];
        dictContentSize = ZDICT_finalizeDictionary(
            &mut candidateDictBuffer,
            dictBufferCapacity,
            content,
            content_size,
            samplesBuffer,
            samplesSizes,
            nbFinalizeSamples,
            params.zParams,
        );
        if ERR_isError(dictContentSize) {
            return COVER_dictSelectionError(dictContentSize);
        }

        totalCompressedSize = COVER_checkTotalCompressedSize(
            params,
            samplesSizes,
            samplesBuffer,
            offsets,
            nbCheckSamples,
            nbSamples,
            &candidateDictBuffer[..dictContentSize],
        );
        if ERR_isError(totalCompressedSize) {
            return COVER_dictSelectionError(totalCompressedSize);
        }

        if (totalCompressedSize as f64) <= (largestCompressed as f64) * regressionTolerance {
            return setDictSelection(candidateDictBuffer, dictContentSize, totalCompressedSize);
        }
        dictContentSize *= 2;
    }
    dictContentSize = largestDict;
    totalCompressedSize = largestCompressed;
    setDictSelection(largestDictbuffer, dictContentSize, totalCompressedSize)
}

/// Port of `ZDICT_trainFromBuffer_cover`. Single-threaded COVER training
/// entry point. Returns the dictionary size or an error code.
pub fn ZDICT_trainFromBuffer_cover(
    dictBuffer: &mut [u8],
    dictBufferCapacity: usize,
    samplesBuffer: &[u8],
    samplesSizes: &[usize],
    nbSamples: u32,
    mut parameters: ZDICT_cover_params_t,
) -> usize {
    use crate::common::error::ERR_isError;
    use crate::dict_builder::zdict::ZDICT_finalizeDictionary;

    let displayLevel = parameters.zParams.notificationLevel as i32;
    parameters.splitPoint = 1.0;
    /* Checks */
    if COVER_checkParameters(parameters, dictBufferCapacity) == 0 {
        return ERROR(ErrorCode::ParameterOutOfBound);
    }
    if nbSamples == 0 {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    if dictBufferCapacity < crate::dict_builder::zdict::ZDICT_DICTSIZE_MIN {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    /* Initialize context and activeDmers */
    let mut ctx = COVER_ctx_t::default();
    {
        let initVal = COVER_ctx_init(
            &mut ctx,
            samplesBuffer,
            samplesSizes,
            nbSamples,
            parameters.d,
            parameters.splitPoint,
            displayLevel,
        );
        if ERR_isError(initVal) {
            return initVal;
        }
    }
    COVER_warnOnSmallCorpus(dictBufferCapacity, ctx.suffixSize, displayLevel);
    let mut activeDmers = COVER_map_t::default();
    if COVER_map_init(&mut activeDmers, parameters.k - parameters.d + 1) == 0 {
        COVER_ctx_destroy(&mut ctx);
        return ERROR(ErrorCode::MemoryAllocation);
    }

    let dictionarySize;
    {
        // freqs aliases ctx.freqs in C; move it out so `&ctx` + `&mut freqs`
        // don't conflict, then restore it before destroy.
        let mut freqs = core::mem::take(&mut ctx.freqs);
        let tail = COVER_buildDictionary(
            &ctx,
            &mut freqs,
            &mut activeDmers,
            dictBuffer,
            dictBufferCapacity,
            parameters,
        );
        ctx.freqs = freqs;
        // ZDICT_finalizeDictionary(dict, cap, dict + tail, cap - tail, ...).
        // `customDictContent` overlaps `dictBuffer`; copy it out to break the
        // borrow (finalizeDictionary also copies internally to handle overlap).
        let content: Vec<u8> = dictBuffer[tail..dictBufferCapacity].to_vec();
        dictionarySize = ZDICT_finalizeDictionary(
            dictBuffer,
            dictBufferCapacity,
            &content,
            dictBufferCapacity - tail,
            samplesBuffer,
            samplesSizes,
            nbSamples,
            parameters.zParams,
        );
    }
    COVER_ctx_destroy(&mut ctx);
    COVER_map_destroy(&mut activeDmers);
    dictionarySize
}

/// Port of `COVER_tryParameters`. Tries one parameter set and updates
/// `best`. (C passes an owning opaque pointer for threading; here a plain
/// sequential call.)
fn COVER_tryParameters(
    ctx: &COVER_ctx_t,
    best: &mut COVER_best_t,
    dictBufferCapacity: usize,
    parameters: ZDICT_cover_params_t,
) {
    use crate::common::error::ERR_isError;
    let _ = ERR_isError;

    let totalCompressedSize = ERROR(ErrorCode::Generic);
    let mut activeDmers = COVER_map_t::default();
    let mut dict = vec![0u8; dictBufferCapacity];
    let mut selection = COVER_dictSelectionError(ERROR(ErrorCode::Generic));
    /* Copy the frequencies because we need to modify them */
    let mut freqs = ctx.freqs.clone();
    if COVER_map_init(&mut activeDmers, parameters.k - parameters.d + 1) != 0 {
        let tail = COVER_buildDictionary(
            ctx,
            &mut freqs,
            &mut activeDmers,
            &mut dict,
            dictBufferCapacity,
            parameters,
        );
        let content = dict[tail..dictBufferCapacity].to_vec();
        let total_samples_size = ctx.offsets[ctx.nbSamples];
        let samples = unsafe { core::slice::from_raw_parts(ctx.samples, total_samples_size) };
        let samplesSizes = unsafe { core::slice::from_raw_parts(ctx.samplesSizes, ctx.nbSamples) };
        selection = COVER_selectDict(
            &content,
            dictBufferCapacity,
            dictBufferCapacity - tail,
            samples,
            samplesSizes,
            ctx.nbTrainSamples as u32,
            ctx.nbTrainSamples,
            ctx.nbSamples,
            parameters,
            &ctx.offsets,
            totalCompressedSize,
        );
        let _ = COVER_dictSelectionIsError(&selection);
    }
    COVER_best_finish(best, parameters, &selection);
    COVER_map_destroy(&mut activeDmers);
    // `selection` drops here == COVER_dictSelectionFree.
}

/// Port of `ZDICT_optimizeTrainFromBuffer_cover`. Grid search over (d, k)
/// returning the best dictionary. POOL parallelism collapsed to a
/// deterministic sequential loop.
pub fn ZDICT_optimizeTrainFromBuffer_cover(
    dictBuffer: &mut [u8],
    dictBufferCapacity: usize,
    samplesBuffer: &[u8],
    samplesSizes: &[usize],
    nbSamples: u32,
    parameters: &mut ZDICT_cover_params_t,
) -> usize {
    use crate::common::error::ERR_isError;
    use crate::dict_builder::zdict::ZDICT_DICTSIZE_MIN;

    let _nbThreads = parameters.nbThreads; // POOL collapsed to sequential
    let splitPoint = if parameters.splitPoint <= 0.0 {
        COVER_DEFAULT_SPLITPOINT
    } else {
        parameters.splitPoint
    };
    let kMinD = if parameters.d == 0 { 6 } else { parameters.d };
    let kMaxD = if parameters.d == 0 { 8 } else { parameters.d };
    let kMinK = if parameters.k == 0 { 50 } else { parameters.k };
    let kMaxK = if parameters.k == 0 { 2000 } else { parameters.k };
    let kSteps = if parameters.steps == 0 { 40 } else { parameters.steps };
    let kStepSize = core::cmp::max((kMaxK - kMinK) / kSteps, 1);
    let shrinkDict: u32 = 0;
    let displayLevel = parameters.zParams.notificationLevel as i32;

    /* Checks */
    if splitPoint <= 0.0 || splitPoint > 1.0 {
        return ERROR(ErrorCode::ParameterOutOfBound);
    }
    if kMinK < kMaxD || kMaxK < kMinK {
        return ERROR(ErrorCode::ParameterOutOfBound);
    }
    if nbSamples == 0 {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    if dictBufferCapacity < ZDICT_DICTSIZE_MIN {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }

    /* Initialization */
    let mut best = COVER_best_t::default();
    COVER_best_init(&mut best);
    let mut warned = false;

    let mut d = kMinD;
    while d <= kMaxD {
        let mut ctx = COVER_ctx_t::default();
        {
            let childDisplayLevel = if displayLevel == 0 { 0 } else { displayLevel - 1 };
            let initVal = COVER_ctx_init(
                &mut ctx,
                samplesBuffer,
                samplesSizes,
                nbSamples,
                d,
                splitPoint,
                childDisplayLevel,
            );
            if ERR_isError(initVal) {
                COVER_best_destroy(&mut best);
                return initVal;
            }
        }
        if !warned {
            COVER_warnOnSmallCorpus(dictBufferCapacity, ctx.suffixSize, displayLevel);
            warned = true;
        }
        let mut k = kMinK;
        while k <= kMaxK {
            let mut p = *parameters;
            p.k = k;
            p.d = d;
            p.splitPoint = splitPoint;
            p.steps = kSteps;
            p.shrinkDict = shrinkDict;
            p.zParams.notificationLevel = ctx.displayLevel as u32;
            if COVER_checkParameters(p, dictBufferCapacity) == 0 {
                k += kStepSize;
                continue;
            }
            COVER_best_start(&mut best);
            COVER_tryParameters(&ctx, &mut best, dictBufferCapacity, p);
            k += kStepSize;
        }
        COVER_best_wait(&best);
        COVER_ctx_destroy(&mut ctx);
        d += 2;
    }

    let dictSize = best.dictSize;
    if ERR_isError(best.compressedSize) {
        let compressedSize = best.compressedSize;
        COVER_best_destroy(&mut best);
        return compressedSize;
    }
    *parameters = best.parameters;
    dictBuffer[..dictSize].copy_from_slice(&best.dict[..dictSize]);
    COVER_best_destroy(&mut best);
    dictSize
}
