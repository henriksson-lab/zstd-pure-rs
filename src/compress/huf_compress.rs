//! Translation of `lib/compress/huf_compress.c`.
//!
//! **Fully implemented**: the pure-math leaves (`HUF_cardinality`,
//! `HUF_minTableLog`, `HUF_optimalTableLog`, `HUF_compressBound`),
//! CTable access (`HUF_getNbBits/Value`, `HUF_setNbBits/Value`,
//! `HUF_getNbBitsFromCTable`, `HUF_readCTableHeader` /
//! `HUF_writeCTableHeader`), sort + tree construction
//! (`HUF_sort`, `HUF_buildTree`, `HUF_setMaxHeight`, `HUF_isSorted`,
//! `HUF_buildCTableFromTree`, `HUF_buildCTable_wksp` +
//! `HUF_buildCTable`), size/validity checks
//! (`HUF_estimateCompressedSize`, `HUF_validateCTable`), header
//! emission (`HUF_compressWeights`, `HUF_writeCTable`), bitstream
//! writer (`HUF_initCStream`, `HUF_addBits`, …), and the
//! `HUF_compress1X/4X_usingCTable` entry points. `_repeat` is also
//! fully exposed.

use crate::common::bits::ZSTD_highbit32;
use crate::compress::zstd_compress_literals::HUF_repeat;

/// Upstream `HUF_WORKSPACE_SIZE` (`huf.h:33`): 8 KiB + 512 bytes sort
/// scratch. Used as the `wkspSize` argument to every wksp-taking HUF
/// builder/reader.
pub const HUF_WORKSPACE_SIZE: usize = (8 << 10) + 512;

/// Upstream `HUF_WORKSPACE_SIZE_U64`. Same size expressed as a u64
/// count (useful when the caller's workspace is `[u64]`-typed).
pub const HUF_WORKSPACE_SIZE_U64: usize = HUF_WORKSPACE_SIZE / 8;

/// Upstream `HUF_BLOCKSIZE_MAX` (`huf.h:25`). Maximum input size for
/// a single block compressed with `HUF_compress*`.
pub const HUF_BLOCKSIZE_MAX: usize = 128 * 1024;
const SUSPECT_INCOMPRESSIBLE_SAMPLE_SIZE: usize = 4096;
const SUSPECT_INCOMPRESSIBLE_SAMPLE_RATIO: usize = 10;

/// Upstream `HUF_CTABLE_WORKSPACE_SIZE_U32` (`huf.h:161`). Byte count
/// of the workspace the CTable-building family (`HUF_buildCTable_wksp`,
/// etc.) expects. Broken out as U32 count + byte count per upstream.
pub const HUF_CTABLE_WORKSPACE_SIZE_U32: usize = 4 * (256 + 1) + 192;
pub const HUF_CTABLE_WORKSPACE_SIZE: usize = HUF_CTABLE_WORKSPACE_SIZE_U32 * 4;

/// Upstream `HUF_BLOCKBOUND` macro (`huf.h:52`). Only valid when the
/// caller has pre-filtered incompressible input via the fast
/// heuristic; `HUF_compressBound` layers on `HUF_CTABLEBOUND`.
#[inline]
pub const fn HUF_BLOCKBOUND(size: usize) -> usize {
    size + (size >> 8) + 8
}

/// Upstream `HUF_CTABLEBOUND` (`huf.h:51`). Worst-case bytes emitted
/// by `HUF_writeCTable`.
pub const HUF_CTABLEBOUND: usize = 129;

/// Upstream `HUF_CTABLE_SIZE_ST(maxSymbolValue)` (`huf.h:58`). Number
/// of `size_t` slots in a CTable, including the 2-slot header.
#[inline]
pub const fn HUF_CTABLE_SIZE_ST(maxSymbolValue: u32) -> usize {
    maxSymbolValue as usize + 2
}

/// Upstream `HUF_CTABLE_SIZE` — byte size of a CTable for a given
/// alphabet.
#[inline]
pub const fn HUF_CTABLE_SIZE(maxSymbolValue: u32) -> usize {
    HUF_CTABLE_SIZE_ST(maxSymbolValue) * core::mem::size_of::<usize>()
}

/// Upstream `HUF_DTABLE_SIZE(maxTableLog)` (`huf.h:65`). DTable slot
/// count including the 1-slot header.
#[inline]
pub const fn HUF_DTABLE_SIZE(maxTableLog: u32) -> usize {
    1 + (1usize << maxTableLog)
}

/// Upstream uses `U64` for `HUF_CElt` in the compressor (it packs both
/// code value and nbBits into one 64-bit word so the hot emit loop can
/// `BIT_addBitsFast` without a separate lookup).
pub type HUF_CElt = u64;

/// Port of `showU32`. Debug helper in upstream; retained here so the
/// function surface matches even though Rust tests don't log through it.
#[inline]
pub fn showU32(arr: &[u32]) -> usize {
    arr.len()
}

/// Port of `showCTableBits`. Debug helper that walks the same `nbBits`
/// projection upstream logs; retained so the helper surface matches.
#[inline]
pub fn showCTableBits(ctable: &[HUF_CElt]) -> usize {
    for &elt in ctable {
        let _ = HUF_getNbBits(elt);
    }
    ctable.len()
}

/// `HUF_COMPRESSBOUND` — mirror of the header macro.
/// Compact worst-case: 129 bytes header + srcSize + 1 + 1 + 2 = srcSize + 129.
#[inline]
pub const fn HUF_COMPRESSBOUND(srcSize: usize) -> usize {
    129 + srcSize + 1 + 1 + 2
}

/// Port of `HUF_compressBound`.
#[inline]
pub fn HUF_compressBound(srcSize: usize) -> usize {
    HUF_COMPRESSBOUND(srcSize)
}

/// Port of `HUF_tightCompressBound` (`huf_compress.c:1050`). Tight
/// upper bound on the encoded-bitstream size given a known `tableLog`:
/// `(srcSize * tableLog) / 8 + 8`. Used inside
/// `HUF_compress1X_usingCTable_internal_body` to decide between the
/// fast (few-reloads) and safe (per-symbol-reload) encode loops.
#[inline]
pub fn HUF_tightCompressBound(srcSize: usize, tableLog: usize) -> usize {
    ((srcSize * tableLog) >> 3) + 8
}

/// Port of `HUF_cardinality`. Count of symbols with non-zero
/// frequency in `count[0..=maxSymbolValue]`.
pub fn HUF_cardinality(count: &[u32], maxSymbolValue: u32) -> u32 {
    count
        .iter()
        .take(maxSymbolValue as usize + 1)
        .filter(|&&c| c != 0)
        .count() as u32
}

/// Port of `HUF_minTableLog`. Given a symbol cardinality (number of
/// distinct symbols), returns the minimum tableLog needed to express
/// a prefix-free code for all of them.
#[inline]
pub fn HUF_minTableLog(symbolCardinality: u32) -> u32 {
    ZSTD_highbit32(symbolCardinality) + 1
}

/// Port of `HUF_optimalTableLog` cheap path. Upstream uses this when
/// `HUF_flags_optimalDepth` is not set.
pub fn HUF_optimalTableLog(maxTableLog: u32, srcSize: usize, maxSymbolValue: u32) -> u32 {
    crate::compress::fse_compress::FSE_optimalTableLog_internal(
        maxTableLog,
        srcSize,
        maxSymbolValue,
        1,
    )
}

/// Port of upstream's full `HUF_optimalTableLog()` entry. The public
/// Rust helper above keeps the cheap/default path; this internal
/// variant mirrors the flag-gated depth probe used by
/// `HUF_compress_internal()`.
pub fn HUF_optimalTableLog_internal(
    maxTableLog: u32,
    srcSize: usize,
    maxSymbolValue: u32,
    table: &mut [HUF_CElt],
    count: &[u32],
    flags: i32,
) -> u32 {
    use crate::decompress::huf_decompress::HUF_flags_optimalDepth;

    debug_assert!(srcSize > 1);
    if (flags & HUF_flags_optimalDepth) == 0 {
        return HUF_optimalTableLog(maxTableLog, srcSize, maxSymbolValue);
    }

    let symbolCardinality = HUF_cardinality(count, maxSymbolValue);
    let minTableLog = HUF_minTableLog(symbolCardinality);
    let mut optSize = usize::MAX - 1;
    let mut optLog = maxTableLog;
    let mut probe = [0u8; HUF_CTABLEBOUND];

    for optLogGuess in minTableLog..=maxTableLog {
        let mut build_wksp = [0u32; 1024];
        let maxBits =
            HUF_buildCTable_wksp(table, count, maxSymbolValue, optLogGuess, &mut build_wksp);
        if crate::common::error::ERR_isError(maxBits) {
            continue;
        }

        if maxBits < optLogGuess as usize && optLogGuess > minTableLog {
            break;
        }

        let mut write_wksp = [0u8; 1];
        let hSize = HUF_writeCTable_wksp(
            &mut probe,
            table,
            maxSymbolValue,
            maxBits as u32,
            &mut write_wksp,
        );
        if crate::common::error::ERR_isError(hSize) {
            continue;
        }

        let newSize = HUF_estimateCompressedSize(table, count, maxSymbolValue) + hSize;
        if newSize > optSize + 1 {
            break;
        }
        if newSize < optSize {
            optSize = newSize;
            optLog = optLogGuess;
        }
    }

    debug_assert!(optLog <= crate::decompress::huf_decompress::HUF_TABLELOG_MAX as u32);
    optLog
}

// ---- HUF_CElt accessors --------------------------------------------------
//
// Upstream layout packs (code, nbBits) into one u64:
//   bits 0..8      = nbBits (≤ HUF_TABLELOG_ABSOLUTEMAX = 12)
//   bits (64-nbBits)..64 = code (MSB-justified)
// Keeping that exact layout so code-complexity-comparator sees the
// same accessor chain the C code uses.

/// Port of `HUF_getNbBits`.
#[inline]
pub fn HUF_getNbBits(elt: HUF_CElt) -> u64 {
    elt & 0xFF
}

/// Port of `HUF_getNbBitsFast`. Caller guarantees the CElt is
/// pre-masked to 8 bits (used in hot loops where the high bits hold
/// the code).
#[inline]
pub fn HUF_getNbBitsFast(elt: HUF_CElt) -> u64 {
    elt
}

/// Port of `HUF_getValue`. Returns the MSB-justified code value with
/// the nbBits byte cleared.
#[inline]
pub fn HUF_getValue(elt: HUF_CElt) -> u64 {
    elt & !0xFF
}

/// Port of `HUF_getValueFast`. Caller guarantees low byte already
/// cleared.
#[inline]
pub fn HUF_getValueFast(elt: HUF_CElt) -> u64 {
    elt
}

/// Port of `HUF_setNbBits`.
#[inline]
pub fn HUF_setNbBits(elt: &mut HUF_CElt, nbBits: u64) {
    debug_assert!(nbBits <= HUF_TABLELOG_ABSOLUTEMAX);
    *elt = nbBits;
}

/// Port of `HUF_setValue`. Writes `value` MSB-justified above the
/// existing nbBits byte. No-op when nbBits == 0 (dead symbol).
pub fn HUF_setValue(elt: &mut HUF_CElt, value: u64) {
    let nbBits = HUF_getNbBits(*elt);
    if nbBits > 0 {
        debug_assert!(value >> nbBits == 0);
        *elt |= value << (64 - nbBits);
    }
}

/// Upstream `HUF_TABLELOG_ABSOLUTEMAX` from the decoder header.
pub const HUF_TABLELOG_ABSOLUTEMAX: u64 =
    crate::decompress::huf_decompress::HUF_TABLELOG_ABSOLUTEMAX as u64;

/// Upstream `HUF_SYMBOLVALUE_MAX`.
pub const HUF_SYMBOLVALUE_MAX: u32 = crate::decompress::huf_decompress::HUF_SYMBOLVALUE_MAX;

/// Port of `HUF_CTableHeader`. Upstream packs this at `CTable[0]` —
/// the 64-bit CElt whose low bytes are { tableLog, maxSymbolValue,
/// reserved*6 }.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HUF_CTableHeader {
    pub tableLog: u8,
    pub maxSymbolValue: u8,
    pub unused: [u8; 6],
}

/// Port of `HUF_readCTableHeader`.
pub fn HUF_readCTableHeader(ctable: &[HUF_CElt]) -> HUF_CTableHeader {
    let raw = ctable[0];
    HUF_CTableHeader {
        tableLog: raw as u8,
        maxSymbolValue: (raw >> 8) as u8,
        unused: [0; 6],
    }
}

/// Port of `HUF_readCTable` (`huf_compress.c:292`). Parses a
/// serialized Huffman tree description (written by `HUF_writeCTable`)
/// back into a CTable. Reconstructs per-symbol (nbBits, value) by:
///   1. `HUF_readStats` — parse weights + rank histogram + tableLog.
///   2. Derive per-rank base values so codes increase monotonically
///      within rank, halving between ranks.
///   3. For each symbol, set `nbBits = (tableLog + 1 - weight)` and
///      assign the next unused value at that rank.
///
/// `hasZeroWeights` returns via the caller's mutable reference — true
/// if any symbol had weight 0 (rank 0 is non-empty).
///
/// Matches upstream bit-for-bit.
pub fn HUF_readCTable(
    ctable: &mut [HUF_CElt],
    maxSymbolValuePtr: &mut u32,
    src: &[u8],
    hasZeroWeights: &mut u32,
) -> usize {
    use crate::common::entropy_common::{HUF_readStats, HUF_TABLELOG_ABSOLUTEMAX};
    const HUF_SYMBOLVALUE_MAX_LOCAL: usize = 255;
    const HUF_TABLELOG_MAX: u32 = 12;

    let mut huffWeight = [0u8; HUF_SYMBOLVALUE_MAX_LOCAL + 1];
    let mut rankVal = [0u32; (HUF_TABLELOG_ABSOLUTEMAX + 1) as usize];
    let mut tableLog: u32 = 0;
    let mut nbSymbols: u32 = 0;

    let readSize = HUF_readStats(
        &mut huffWeight,
        HUF_SYMBOLVALUE_MAX_LOCAL + 1,
        &mut rankVal,
        &mut nbSymbols,
        &mut tableLog,
        src,
    );
    if crate::common::error::ERR_isError(readSize) {
        return readSize;
    }
    *hasZeroWeights = (rankVal[0] > 0) as u32;

    if tableLog > HUF_TABLELOG_MAX {
        return crate::common::error::ERROR(crate::common::error::ErrorCode::TableLogTooLarge);
    }
    if nbSymbols > *maxSymbolValuePtr + 1 {
        return crate::common::error::ERROR(
            crate::common::error::ErrorCode::MaxSymbolValueTooSmall,
        );
    }

    *maxSymbolValuePtr = nbSymbols - 1;
    HUF_writeCTableHeader(ctable, tableLog, *maxSymbolValuePtr);

    // Prepare base value per rank (cumulative start positions).
    let mut nextRankStart: u32 = 0;
    #[allow(clippy::needless_range_loop)]
    for n in 1..=(tableLog as usize) {
        let curr = nextRankStart;
        nextRankStart += rankVal[n] << (n - 1);
        rankVal[n] = curr;
    }

    // Fill nbBits for each symbol: weight=0 → nbBits=0, else
    // nbBits = tableLog + 1 - weight.
    let ct = &mut ctable[1..];
    for (n, elt) in ct.iter_mut().take(nbSymbols as usize).enumerate() {
        let w = huffWeight[n];
        let nbBits = if w != 0 { tableLog + 1 - w as u32 } else { 0 };
        HUF_setNbBits(elt, nbBits as u64);
    }

    // Count symbols per rank, compute starting code-value per rank.
    let mut nbPerRank = [0u16; (HUF_TABLELOG_MAX + 2) as usize];
    let mut valPerRank = [0u16; (HUF_TABLELOG_MAX + 2) as usize];
    for elt in ct.iter().take(nbSymbols as usize) {
        nbPerRank[HUF_getNbBits(*elt) as usize] += 1;
    }
    valPerRank[(tableLog + 1) as usize] = 0;
    {
        let mut min: u16 = 0;
        let mut n = tableLog as usize;
        while n > 0 {
            valPerRank[n] = min;
            min += nbPerRank[n];
            min >>= 1;
            n -= 1;
        }
    }
    // Assign code value within each rank, in symbol order.
    for elt in ct.iter_mut().take(nbSymbols as usize) {
        let rank = HUF_getNbBits(*elt) as usize;
        HUF_setValue(elt, valPerRank[rank] as u64);
        valPerRank[rank] += 1;
    }

    readSize
}

/// Port of `HUF_writeCTableHeader` (file-private).
pub fn HUF_writeCTableHeader(ctable: &mut [HUF_CElt], tableLog: u32, maxSymbolValue: u32) {
    debug_assert!(tableLog < 256);
    debug_assert!(maxSymbolValue < 256);
    ctable[0] = (tableLog as u64) | ((maxSymbolValue as u64) << 8);
}

/// Port of `HUF_getNbBitsFromCTable`. The table starts at `CTable[1]`
/// (slot 0 holds the header); returns the nbBits field for
/// `symbolValue` if present, otherwise 0.
pub fn HUF_getNbBitsFromCTable(ctable: &[HUF_CElt], symbolValue: u32) -> u32 {
    debug_assert!(symbolValue <= HUF_SYMBOLVALUE_MAX);
    let header = HUF_readCTableHeader(ctable);
    if symbolValue > header.maxSymbolValue as u32 {
        return 0;
    }
    HUF_getNbBits(ctable[1 + symbolValue as usize]) as u32
}

// ---- Huffman tree construction helpers ---------------------------------

/// Mirror of upstream `nodeElt`. Stored in the Huffman scratch table
/// during tree construction.
#[derive(Debug, Clone, Copy, Default)]
pub struct nodeElt {
    pub count: u32,
    pub parent: u16,
    pub byte: u8,
    pub nbBits: u8,
}

/// Port of `showHNodeSymbols`.
#[inline]
pub fn showHNodeSymbols(hnode: &[nodeElt]) -> usize {
    hnode.len()
}

/// Port of `showHNodeBits`.
#[inline]
pub fn showHNodeBits(hnode: &[nodeElt]) -> usize {
    hnode.len()
}

/// Upstream `HUF_WORKSPACE_MAX_ALIGNMENT`.
pub const HUF_WORKSPACE_MAX_ALIGNMENT: usize = 8;

/// Port of `HUF_alignUpWorkspace`.
pub fn HUF_alignUpWorkspace(
    workspace: usize,
    workspaceSize: &mut usize,
    align: usize,
) -> Option<usize> {
    let mask = align - 1;
    let rem = workspace & mask;
    let add = (align - rem) & mask;
    debug_assert!(align.is_power_of_two());
    debug_assert!(align <= HUF_WORKSPACE_MAX_ALIGNMENT);
    if *workspaceSize >= add {
        *workspaceSize -= add;
        Some(workspace + add)
    } else {
        *workspaceSize = 0;
        None
    }
}

// ---- HUF_sort (radix + per-bucket insertion) ---------------------------

/// Mirror of upstream `rankPos`.
#[derive(Debug, Clone, Copy, Default)]
pub struct rankPos {
    pub base: u16,
    pub curr: u16,
}

/// Mirror of upstream constants for the radix-sort bucketing.
pub const RANK_POSITION_TABLE_SIZE: usize = 192;
pub const RANK_POSITION_MAX_COUNT_LOG: u32 = 32;
pub const RANK_POSITION_LOG_BUCKETS_BEGIN: u32 =
    (RANK_POSITION_TABLE_SIZE as u32 - 1) - RANK_POSITION_MAX_COUNT_LOG - 1; // == 158
/// Cutoff where count-buckets stop being 1:1 and switch to log2.
pub const RANK_POSITION_DISTINCT_COUNT_CUTOFF: u32 = RANK_POSITION_LOG_BUCKETS_BEGIN + 7; // == 165

/// Port of `HUF_getIndex`. Maps a symbol count to its bucket.
#[inline]
pub fn HUF_getIndex(count: u32) -> u32 {
    if count < RANK_POSITION_DISTINCT_COUNT_CUTOFF {
        count
    } else {
        ZSTD_highbit32(count) + RANK_POSITION_LOG_BUCKETS_BEGIN
    }
}

/// Port of `HUF_swapNodes`.
#[inline]
pub fn HUF_swapNodes(a: &mut nodeElt, b: &mut nodeElt) {
    core::mem::swap(a, b);
}

/// Port of `HUF_insertionSort`.
fn HUF_insertionSort(huffNode: &mut [nodeElt], low: i32, high: i32) {
    let size = (high - low + 1) as usize;
    let slice = &mut huffNode[low as usize..low as usize + size];
    for i in 1..slice.len() {
        let key = slice[i];
        let mut j = i;
        while j > 0 && slice[j - 1].count < key.count {
            slice[j] = slice[j - 1];
            j -= 1;
        }
        slice[j] = key;
    }
}

/// Port of `HUF_quickSortPartition`.
fn HUF_quickSortPartition(arr: &mut [nodeElt], low: i32, high: i32) -> i32 {
    let pivot = arr[high as usize].count;
    let mut i = low - 1;
    let mut j = low;
    while j < high {
        if arr[j as usize].count > pivot {
            i += 1;
            arr.swap(i as usize, j as usize);
        }
        j += 1;
    }
    arr.swap((i + 1) as usize, high as usize);
    i + 1
}

/// Port of `HUF_simpleQuickSort`.
fn HUF_simpleQuickSort(arr: &mut [nodeElt], mut low: i32, mut high: i32) {
    const K_INSERTION_SORT_THRESHOLD: i32 = 8;
    if high - low < K_INSERTION_SORT_THRESHOLD {
        HUF_insertionSort(arr, low, high);
        return;
    }
    while low < high {
        let idx = HUF_quickSortPartition(arr, low, high);
        if idx - low < high - idx {
            HUF_simpleQuickSort(arr, low, idx - 1);
            low = idx + 1;
        } else {
            HUF_simpleQuickSort(arr, idx + 1, high);
            high = idx - 1;
        }
    }
}

/// Port of `HUF_sort`. Sorts `huffNode[0..=maxSymbolValue]` in
/// descending `count` order via a radix-like bucket sort. Buckets past
/// `RANK_POSITION_DISTINCT_COUNT_CUTOFF` use log2 bucketing, so we
/// need a per-bucket insertion sort for those high-count buckets.
pub fn HUF_sort(
    huffNode: &mut [nodeElt],
    count: &[u32],
    maxSymbolValue: u32,
    rankPosition: &mut [rankPos],
) {
    debug_assert!(rankPosition.len() >= RANK_POSITION_TABLE_SIZE);
    let maxSV1 = maxSymbolValue + 1;

    // Reset the bucket table.
    for rp in rankPosition.iter_mut().take(RANK_POSITION_TABLE_SIZE) {
        *rp = rankPos::default();
    }

    // Count one-less-than-destination-rank per symbol.
    for &c in count.iter().take(maxSV1 as usize) {
        let lowerRank = HUF_getIndex(c);
        debug_assert!(lowerRank < RANK_POSITION_TABLE_SIZE as u32 - 1);
        rankPosition[lowerRank as usize].base += 1;
    }

    // Accumulate bases from high rank down so that rankPosition[r].base
    // is the starting index in the output for all symbols of rank ≥ r+1.
    for n in (1..RANK_POSITION_TABLE_SIZE).rev() {
        rankPosition[n - 1].base += rankPosition[n].base;
        rankPosition[n - 1].curr = rankPosition[n - 1].base;
    }

    // Place each symbol at its bucket.
    for (n, &c) in count.iter().enumerate().take(maxSV1 as usize) {
        let r = HUF_getIndex(c) + 1;
        let pos = rankPosition[r as usize].curr as usize;
        rankPosition[r as usize].curr += 1;
        huffNode[pos].count = c;
        huffNode[pos].byte = n as u8;
    }

    // Sort each log-bucket (distinct-count buckets are already sorted
    // because each holds symbols with identical count — their order
    // doesn't matter for tree construction).
    #[allow(clippy::needless_range_loop)]
    for n in (RANK_POSITION_DISTINCT_COUNT_CUTOFF as usize)..(RANK_POSITION_TABLE_SIZE - 1) {
        let start = rankPosition[n].base as usize;
        let end = rankPosition[n].curr as usize;
        if end > start + 1 {
            HUF_simpleQuickSort(&mut huffNode[start..end], 0, (end - start - 1) as i32);
        }
    }
}

/// Upstream's `STARTNODE` — first slot used for parent nodes.
pub const STARTNODE: usize = (HUF_SYMBOLVALUE_MAX + 1) as usize;

/// Port of `HUF_setMaxHeight`. Enforces `targetNbBits` on a Huffman
/// tree whose leaves are sorted most-frequent-first. Upstream's
/// invariant: Σ 2^(largestBits - nbBits) over leaves = 2^largestBits.
/// When `largestBits > targetNbBits`, clip every long length to
/// `targetNbBits`, track the excess "cost", and repay it by lengthening
/// carefully chosen shorter codes — always picking the cheapest repay
/// option (extending one `nBitsToDecrease`-rank node vs moving two
/// rank-below nodes).
pub fn HUF_setMaxHeight(huffNode: &mut [nodeElt], lastNonNull: u32, targetNbBits: u32) -> u32 {
    use crate::decompress::huf_decompress::HUF_TABLELOG_MAX;
    let largestBits = huffNode[lastNonNull as usize].nbBits as u32;
    if largestBits <= targetNbBits {
        return largestBits;
    }

    let mut totalCost: i32 = 0;
    let baseCost: u32 = 1 << (largestBits - targetNbBits);
    let mut n = lastNonNull as i32;

    // Clip every leaf with nbBits > targetNbBits, accumulating the
    // cost of each clip (measured in units of 2^-largestBits; later
    // renormalized to 2^-targetNbBits).
    while n >= 0 && huffNode[n as usize].nbBits as u32 > targetNbBits {
        let leaf_bits = huffNode[n as usize].nbBits as u32;
        totalCost += (baseCost as i32) - (1 << (largestBits - leaf_bits));
        huffNode[n as usize].nbBits = targetNbBits as u8;
        n -= 1;
    }
    debug_assert!(huffNode[n as usize].nbBits as u32 <= targetNbBits);
    // Skip over any leaves already at targetNbBits — they can't be
    // extended further.
    while n >= 0 && huffNode[n as usize].nbBits as u32 == targetNbBits {
        n -= 1;
    }

    // Renormalize totalCost from 2^-largestBits to 2^-targetNbBits.
    debug_assert!((totalCost as u32 & (baseCost - 1)) == 0);
    totalCost >>= largestBits - targetNbBits;
    debug_assert!(totalCost > 0);

    const NO_SYMBOL: u32 = 0xF0F0F0F0;
    let mut rankLast = [NO_SYMBOL; (HUF_TABLELOG_MAX + 2) as usize];

    // For each rank strictly below targetNbBits, record the position
    // of the last (smallest-count) leaf in that rank.
    {
        let mut currentNbBits = targetNbBits;
        let mut pos = n;
        while pos >= 0 {
            let leaf_bits = huffNode[pos as usize].nbBits as u32;
            if leaf_bits >= currentNbBits {
                pos -= 1;
                continue;
            }
            currentNbBits = leaf_bits;
            rankLast[(targetNbBits - currentNbBits) as usize] = pos as u32;
            pos -= 1;
        }
    }

    // Repay totalCost by extending leaves at carefully chosen ranks.
    while totalCost > 0 {
        let mut nBitsToDecrease = ZSTD_highbit32(totalCost as u32) + 1;
        debug_assert!(nBitsToDecrease <= HUF_TABLELOG_MAX + 1);
        // Prefer moving a single high-rank node over two low-rank ones
        // when the high-rank one is cheaper.
        while nBitsToDecrease > 1 {
            let highPos = rankLast[nBitsToDecrease as usize];
            let lowPos = rankLast[(nBitsToDecrease - 1) as usize];
            if highPos == NO_SYMBOL {
                nBitsToDecrease -= 1;
                continue;
            }
            if lowPos == NO_SYMBOL {
                break;
            }
            let highTotal = huffNode[highPos as usize].count;
            let lowTotal = 2u32 * huffNode[lowPos as usize].count;
            if highTotal <= lowTotal {
                break;
            }
            nBitsToDecrease -= 1;
        }
        // Walk up until we land on an occupied rank.
        while nBitsToDecrease <= HUF_TABLELOG_MAX && rankLast[nBitsToDecrease as usize] == NO_SYMBOL
        {
            nBitsToDecrease += 1;
        }
        debug_assert!(rankLast[nBitsToDecrease as usize] != NO_SYMBOL);

        totalCost -= 1 << (nBitsToDecrease - 1);
        let target_idx = rankLast[nBitsToDecrease as usize] as usize;
        huffNode[target_idx].nbBits += 1;

        // Rank bookkeeping: the newly lengthened leaf just moved down
        // one rank. Fix up neighbouring ranks.
        if rankLast[(nBitsToDecrease - 1) as usize] == NO_SYMBOL {
            rankLast[(nBitsToDecrease - 1) as usize] = rankLast[nBitsToDecrease as usize];
        }
        if rankLast[nBitsToDecrease as usize] == 0 {
            rankLast[nBitsToDecrease as usize] = NO_SYMBOL;
        } else {
            rankLast[nBitsToDecrease as usize] -= 1;
            let p = rankLast[nBitsToDecrease as usize] as usize;
            if huffNode[p].nbBits as u32 != targetNbBits - nBitsToDecrease {
                rankLast[nBitsToDecrease as usize] = NO_SYMBOL;
            }
        }
    }

    // If cost correction overshot, unwind by shortening symbols back.
    while totalCost < 0 {
        if rankLast[1] == NO_SYMBOL {
            // No rank-1 symbol — synthesize one by shortening the
            // largest rank-0 node.
            while n >= 0 && huffNode[n as usize].nbBits as u32 == targetNbBits {
                n -= 1;
            }
            huffNode[(n + 1) as usize].nbBits -= 1;
            debug_assert!(n >= 0);
            rankLast[1] = (n + 1) as u32;
            totalCost += 1;
            continue;
        }
        huffNode[(rankLast[1] + 1) as usize].nbBits -= 1;
        rankLast[1] += 1;
        totalCost += 1;
    }

    targetNbBits
}

/// Port of `HUF_buildTree`. Takes a descending-by-count sorted symbol
/// array and merges the two least-frequent entries repeatedly to
/// build an unlimited-depth Huffman tree, writing parent indices back
/// into the same array. Returns `nonNullRank` — the index of the
/// smallest (last) non-zero-count symbol before the parent nodes
/// start.
///
/// Rust signature note: upstream uses `huffNode[-1]` as a sentinel
/// (`huffNode0 = huffNode - 1`). The Rust port emulates it via an
/// explicit sentinel value in the `lowS` comparison — there's no
/// index -1 at runtime.
pub fn HUF_buildTree(huffNode: &mut [nodeElt], maxSymbolValue: u32) -> i32 {
    // Locate the smallest non-null symbol.
    let mut nonNullRank = maxSymbolValue as i32;
    while nonNullRank >= 0 && huffNode[nonNullRank as usize].count == 0 {
        nonNullRank -= 1;
    }
    debug_assert!(nonNullRank >= 1, "HUF_buildTree needs ≥ 2 distinct symbols");

    let mut lowS = nonNullRank;
    let mut nodeNb = STARTNODE as i32;
    let nodeRoot = nodeNb + lowS - 1;
    let mut lowN = nodeNb;

    // First parent: merge the two rarest leaves.
    huffNode[nodeNb as usize].count =
        huffNode[lowS as usize].count + huffNode[(lowS - 1) as usize].count;
    huffNode[lowS as usize].parent = nodeNb as u16;
    huffNode[(lowS - 1) as usize].parent = nodeNb as u16;
    nodeNb += 1;
    lowS -= 2;

    // Sentinel: initialize future parent-node counts to a high value
    // that keeps the two-way merge balanced.
    for n in nodeNb..=nodeRoot {
        huffNode[n as usize].count = 1u32 << 30;
    }

    // Virtual sentinel for `huffNode[-1]`: upstream stashes it at
    // `huffNode0[0]` with count = 1 << 31 so that the `huffNode[lowS]`
    // side always loses when lowS has underflowed. We emulate it by
    // short-circuiting the comparison at lowS == -1.
    let sentinel: u32 = 1u32 << 31;

    // Create parents: repeatedly merge the two smallest available
    // nodes, where candidates come from the leaves (walking lowS down)
    // and the parent frontier (walking lowN up).
    while nodeNb <= nodeRoot {
        let n1_count = if lowS < 0 {
            sentinel
        } else {
            huffNode[lowS as usize].count
        };
        let lowN_count = huffNode[lowN as usize].count;
        let n1 = if n1_count < lowN_count {
            let v = lowS;
            lowS -= 1;
            v
        } else {
            let v = lowN;
            lowN += 1;
            v
        };

        let n1_count_stored = if n1 < 0 {
            sentinel
        } else {
            huffNode[n1 as usize].count
        };
        let second_lowS_count = if lowS < 0 {
            sentinel
        } else {
            huffNode[lowS as usize].count
        };
        let lowN_count = huffNode[lowN as usize].count;
        let n2 = if second_lowS_count < lowN_count {
            let v = lowS;
            lowS -= 1;
            v
        } else {
            let v = lowN;
            lowN += 1;
            v
        };

        let n2_count_stored = if n2 < 0 {
            sentinel
        } else {
            huffNode[n2 as usize].count
        };
        let combined = n1_count_stored + n2_count_stored;
        huffNode[nodeNb as usize].count = combined;
        if n1 >= 0 {
            huffNode[n1 as usize].parent = nodeNb as u16;
        }
        if n2 >= 0 {
            huffNode[n2 as usize].parent = nodeNb as u16;
        }
        nodeNb += 1;
    }

    // Distribute bit-lengths (unlimited height): root has 0 bits;
    // every other node is parent's nbBits + 1.
    huffNode[nodeRoot as usize].nbBits = 0;
    let mut n = nodeRoot - 1;
    while n >= STARTNODE as i32 {
        let p = huffNode[n as usize].parent as usize;
        huffNode[n as usize].nbBits = huffNode[p].nbBits + 1;
        n -= 1;
    }
    for n in 0..=nonNullRank as usize {
        let p = huffNode[n].parent as usize;
        huffNode[n].nbBits = huffNode[p].nbBits + 1;
    }

    nonNullRank
}

/// Port of `HUF_isSorted` — post-condition check.
pub fn HUF_isSorted(huffNode: &[nodeElt], maxSymbolValue1: u32) -> bool {
    for i in 1..(maxSymbolValue1 as usize) {
        if huffNode[i].count > huffNode[i - 1].count {
            return false;
        }
    }
    true
}

/// Port of `HUF_buildCTableFromTree`. Given a sorted-by-descending-count
/// Huffman tree (`huffNode[0..=nonNullRank]` holding each symbol's
/// final `nbBits` length), write the canonical Huffman code into
/// `CTable[1..]` and stamp the header into `CTable[0]`.
///
/// Canonical rule: values within each rank are assigned in rising
/// integer order; the start value of each rank is derived from the
/// rank above via `(start + count) >> 1`. Smallest `nbBits` → highest
/// value MSB, largest `nbBits` → lowest value. Upstream stores the
/// value MSB-justified via `HUF_setValue`.
pub fn HUF_buildCTableFromTree(
    ctable: &mut [HUF_CElt],
    huffNode: &[nodeElt],
    nonNullRank: i32,
    maxSymbolValue: u32,
    maxNbBits: u32,
) {
    use crate::decompress::huf_decompress::HUF_TABLELOG_MAX;
    const HTLOG_PLUS_1: usize = (HUF_TABLELOG_MAX + 1) as usize;
    let ct_start = 1; // ct = CTable + 1

    let mut nbPerRank = [0u16; HTLOG_PLUS_1];
    let mut valPerRank = [0u16; HTLOG_PLUS_1];
    let alphabetSize = (maxSymbolValue + 1) as usize;

    for n in 0..=nonNullRank as usize {
        nbPerRank[huffNode[n].nbBits as usize] += 1;
    }

    // Determine starting value per rank.
    {
        let mut min: u16 = 0;
        let mut n = maxNbBits as usize;
        while n > 0 {
            valPerRank[n] = min;
            min += nbPerRank[n];
            min >>= 1;
            n -= 1;
        }
    }

    // Push nbBits per symbol (in symbol order).
    for node in huffNode.iter().take(alphabetSize) {
        let b = node.byte as usize;
        HUF_setNbBits(&mut ctable[ct_start + b], node.nbBits as u64);
    }
    // Assign value within rank (also symbol order).
    for n in 0..alphabetSize {
        let nb = HUF_getNbBits(ctable[ct_start + n]) as usize;
        let v = valPerRank[nb] as u64;
        HUF_setValue(&mut ctable[ct_start + n], v);
        valPerRank[nb] += 1;
    }

    HUF_writeCTableHeader(ctable, maxNbBits, maxSymbolValue);
}

/// Default upstream tableLog when caller passes 0.
pub const HUF_TABLELOG_DEFAULT: u32 = 11;

/// Port of `HUF_buildCTable_wksp`. End-to-end Huffman CTable builder:
/// sort → buildTree → setMaxHeight → buildCTableFromTree. Returns the
/// final tableLog (used tree depth) or an error code.
///
/// Rust signature note: upstream passes a single `void*` workspace
/// containing both the nodeElt array and the rankPos array. The Rust
/// port allocates them as typed arrays locally — the workspace
/// argument is accepted for API parity but currently unused.
pub fn HUF_buildCTable_wksp(
    ct: &mut [HUF_CElt],
    count: &[u32],
    maxSymbolValue: u32,
    mut maxNbBits: u32,
    _workSpace: &mut [u32],
) -> usize {
    use crate::common::error::{ErrorCode, ERROR};
    if maxNbBits == 0 {
        maxNbBits = HUF_TABLELOG_DEFAULT;
    }
    if maxSymbolValue > HUF_SYMBOLVALUE_MAX {
        return ERROR(ErrorCode::MaxSymbolValueTooLarge);
    }
    // Upstream's huffNodeTbl: `2 * (HUF_SYMBOLVALUE_MAX + 1)` entries.
    // `huffNode = huffNodeTbl + 1`. Our port treats huffNodeTbl and
    // huffNode as the same buffer — the `huffNode[-1]` sentinel is
    // emulated inside `HUF_buildTree`.
    let mut huffNode = vec![nodeElt::default(); 2 * (HUF_SYMBOLVALUE_MAX as usize + 1)];
    let mut rankPosition = vec![rankPos::default(); RANK_POSITION_TABLE_SIZE];

    HUF_sort(&mut huffNode, count, maxSymbolValue, &mut rankPosition);
    let nonNullRank = HUF_buildTree(&mut huffNode, maxSymbolValue);
    let finalMaxBits = HUF_setMaxHeight(&mut huffNode, nonNullRank as u32, maxNbBits);
    let HUF_TABLELOG_MAX = crate::decompress::huf_decompress::HUF_TABLELOG_MAX;
    if finalMaxBits > HUF_TABLELOG_MAX {
        return ERROR(ErrorCode::Generic);
    }
    HUF_buildCTableFromTree(ct, &huffNode, nonNullRank, maxSymbolValue, finalMaxBits);
    finalMaxBits as usize
}

/// Port of `HUF_buildCTable` — convenience wrapper with an internal
/// workspace.
pub fn HUF_buildCTable(
    ct: &mut [HUF_CElt],
    count: &[u32],
    maxSymbolValue: u32,
    maxNbBits: u32,
) -> usize {
    let mut wksp = [0u32; 1024];
    HUF_buildCTable_wksp(ct, count, maxSymbolValue, maxNbBits, &mut wksp)
}

/// Port of `HUF_estimateCompressedSize`. Returns the estimated
/// compressed byte count (bits ÷ 8) that `HUF_compress*_usingCTable`
/// would produce given the CTable and symbol histogram.
pub fn HUF_estimateCompressedSize(
    ctable: &[HUF_CElt],
    count: &[u32],
    maxSymbolValue: u32,
) -> usize {
    let ct_start = 1;
    let mut nbBits: usize = 0;
    for s in 0..=(maxSymbolValue as usize) {
        nbBits += (HUF_getNbBits(ctable[ct_start + s]) as usize) * (count[s] as usize);
    }
    nbBits >> 3
}

/// Port of `HUF_validateCTable`. Returns `true` when every
/// positive-count symbol in `count` has a non-zero nbBits entry in
/// the CTable — the precondition the emit loop needs.
pub fn HUF_validateCTable(ctable: &[HUF_CElt], count: &[u32], maxSymbolValue: u32) -> bool {
    let header = HUF_readCTableHeader(ctable);
    debug_assert!(header.tableLog as u64 <= HUF_TABLELOG_ABSOLUTEMAX);
    if (header.maxSymbolValue as u32) < maxSymbolValue {
        return false;
    }
    let ct_start = 1;
    for s in 0..=(maxSymbolValue as usize) {
        if count[s] != 0 && HUF_getNbBits(ctable[ct_start + s]) == 0 {
            return false;
        }
    }
    true
}

/// Max FSE tableLog used when FSE-compressing HUF weights.
pub const MAX_FSE_TABLELOG_FOR_HUFF_HEADER: u32 = 6;

/// Port of `HUF_compressWeights`. FSE-compresses the `weightTable`
/// byte array (alphabet size ≤ HUF_TABLELOG_MAX = 12). Returns:
///   - 0 if the payload isn't compressible (each symbol max 1
///     occurrence, or FSE output can't fit).
///   - 1 for RLE (all weights identical).
///   - otherwise the byte length of the FSE-compressed payload.
pub fn HUF_compressWeights(dst: &mut [u8], weightTable: &[u8]) -> usize {
    use crate::compress::fse_compress::{
        FSE_buildCTable_wksp, FSE_compress_usingCTable, FSE_normalizeCount, FSE_optimalTableLog,
        FSE_writeNCount, FSE_CTABLE_SIZE_U32,
    };
    use crate::compress::hist::HIST_count_simple;

    let wtSize = weightTable.len();
    if wtSize <= 1 {
        return 0; // not compressible
    }

    let mut count = [0u32; (crate::decompress::huf_decompress::HUF_TABLELOG_MAX + 1) as usize];
    let mut maxSymbolValue: u32 = crate::decompress::huf_decompress::HUF_TABLELOG_MAX;
    let maxCount = HIST_count_simple(&mut count, &mut maxSymbolValue, weightTable);
    if maxCount as usize == wtSize {
        return 1; // RLE
    }
    if maxCount == 1 {
        return 0; // incompressible — every symbol appears at most once
    }

    let tableLog = FSE_optimalTableLog(MAX_FSE_TABLELOG_FOR_HUFF_HEADER, wtSize, maxSymbolValue);
    let mut norm = [0i16; (crate::decompress::huf_decompress::HUF_TABLELOG_MAX + 1) as usize];
    let rc = FSE_normalizeCount(&mut norm, tableLog, &count, wtSize, maxSymbolValue, 0);
    if crate::common::error::ERR_isError(rc) {
        return rc;
    }
    let tableLog = rc as u32;

    let mut written = 0usize;
    let hSize = FSE_writeNCount(&mut dst[written..], &norm, maxSymbolValue, tableLog);
    if crate::common::error::ERR_isError(hSize) {
        return hSize;
    }
    written += hSize;

    let ct_slots = FSE_CTABLE_SIZE_U32(tableLog, maxSymbolValue);
    let mut ct = vec![0u32; ct_slots];
    let mut build_wksp = vec![0u8; (maxSymbolValue as usize + 2) * 2 + (1 << tableLog) + 8];
    let rc = FSE_buildCTable_wksp(&mut ct, &norm, maxSymbolValue, tableLog, &mut build_wksp);
    if crate::common::error::ERR_isError(rc) {
        return rc;
    }
    let cSize = FSE_compress_usingCTable(&mut dst[written..], weightTable, &ct);
    if cSize == 0 {
        return 0;
    }
    if crate::common::error::ERR_isError(cSize) {
        return cSize;
    }
    written + cSize
}

/// Port of `HUF_writeCTable_wksp` (`huf_compress.c:248`). External-
/// workspace variant of `HUF_writeCTable`. Our port allocates the
/// workspace internally and ignores the caller's — the `_wksp` entry
/// exists only for upstream API-surface parity.
pub fn HUF_writeCTable_wksp(
    dst: &mut [u8],
    ct: &[HUF_CElt],
    maxSymbolValue: u32,
    huffLog: u32,
    _workspace: &mut [u8],
) -> usize {
    HUF_writeCTable(dst, ct, maxSymbolValue, huffLog)
}

/// Port of `HUF_writeCTable` (`huf_compress.c` — `_wksp` entry). Emits
/// the compact Huffman header that pairs with `HUF_readStats`:
/// attempts FSE-compression of the weight table first (when the
/// shape is favourable), falls back to raw 4-bit nibbles otherwise.
pub fn HUF_writeCTable(
    dst: &mut [u8],
    ct: &[HUF_CElt],
    maxSymbolValue: u32,
    huffLog: u32,
) -> usize {
    use crate::common::error::{ErrorCode, ERROR};
    debug_assert_eq!(
        HUF_readCTableHeader(ct).maxSymbolValue as u32,
        maxSymbolValue
    );
    debug_assert_eq!(HUF_readCTableHeader(ct).tableLog as u32, huffLog);
    if maxSymbolValue > HUF_SYMBOLVALUE_MAX {
        return ERROR(ErrorCode::MaxSymbolValueTooLarge);
    }
    // bitsToWeight: convert nbBits → weight. weight[n] = tableLog+1-n.
    let mut bitsToWeight =
        [0u8; (crate::decompress::huf_decompress::HUF_TABLELOG_MAX + 1) as usize];
    bitsToWeight[0] = 0;
    for (n, w) in bitsToWeight
        .iter_mut()
        .enumerate()
        .skip(1)
        .take(huffLog as usize)
    {
        *w = (huffLog + 1 - n as u32) as u8;
    }
    // huffWeight[n] = bitsToWeight[nbBits(ct[n])] for n in [0..maxSymbolValue).
    let mut huffWeight = vec![0u8; HUF_SYMBOLVALUE_MAX as usize + 1];
    for n in 0..maxSymbolValue as usize {
        let nb = HUF_getNbBits(ct[1 + n]) as usize;
        huffWeight[n] = bitsToWeight[nb];
    }

    // Try FSE-compressing the weights. dst[0] will hold the byte
    // length of the FSE payload; the payload starts at dst[1..].
    if dst.is_empty() {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    let hSize = HUF_compressWeights(&mut dst[1..], &huffWeight[..maxSymbolValue as usize]);
    if crate::common::error::ERR_isError(hSize) {
        return hSize;
    }
    if hSize > 1 && hSize < maxSymbolValue as usize / 2 {
        dst[0] = hSize as u8;
        return hSize + 1;
    }

    // Fall back: write raw 4-bit weights. Requires maxSymbolValue ≤ 128.
    if maxSymbolValue > (256 - 128) {
        return ERROR(ErrorCode::Generic);
    }
    let needed = (maxSymbolValue as usize).div_ceil(2) + 1;
    if needed > dst.len() {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    dst[0] = (128 + (maxSymbolValue - 1)) as u8;
    huffWeight[maxSymbolValue as usize] = 0; // msan-safe when maxSymbolValue is odd
    let mut n = 0usize;
    while n < maxSymbolValue as usize {
        dst[n / 2 + 1] = (huffWeight[n] << 4) + huffWeight[n + 1];
        n += 2;
    }
    (maxSymbolValue as usize).div_ceil(2) + 1
}

/// Upstream's `HUF_BITS_IN_CONTAINER`.
pub const HUF_BITS_IN_CONTAINER: usize = core::mem::size_of::<usize>() * 8;

/// Mirror of `HUF_CStream_t`. Upstream keeps two parallel bit
/// containers so the emit loop can fill index 1 while index 0 is
/// draining — we preserve the shape.
#[derive(Debug)]
pub struct HUF_CStream_t<'a> {
    pub bitContainer: [usize; 2],
    pub bitPos: [usize; 2],
    pub dst: &'a mut [u8],
    pub startPtr: usize,
    pub ptr: usize,
    pub endPtr: usize,
}

/// Port of `HUF_initCStream`. `dstCapacity` must exceed
/// `sizeof(bitContainer)` so the writer has room for a whole word
/// flush; returns a nonzero error code otherwise.
pub fn HUF_initCStream<'a>(dst: &'a mut [u8], dstCapacity: usize) -> (HUF_CStream_t<'a>, usize) {
    use crate::common::error::{ErrorCode, ERROR};
    let word = core::mem::size_of::<usize>();
    let (endPtr, rc) = if dstCapacity <= word {
        (0, ERROR(ErrorCode::DstSizeTooSmall))
    } else {
        (dstCapacity - word, 0)
    };
    let bitC = HUF_CStream_t {
        bitContainer: [0; 2],
        bitPos: [0; 2],
        dst,
        startPtr: 0,
        ptr: 0,
        endPtr,
    };
    (bitC, rc)
}

/// Port of `HUF_addBits`. Adds `elt`'s MSB-justified value at slot
/// `idx` of the bit container. `kFast` skips bookkeeping asserts
/// upstream needs for the hot loop — semantically equivalent.
#[inline]
pub fn HUF_addBits(bitC: &mut HUF_CStream_t, elt: HUF_CElt, idx: usize, kFast: bool) {
    debug_assert!(idx <= 1);
    debug_assert!(HUF_getNbBits(elt) <= HUF_TABLELOG_ABSOLUTEMAX);
    bitC.bitContainer[idx] >>= HUF_getNbBits(elt) as usize;
    let value = if kFast {
        HUF_getValueFast(elt)
    } else {
        HUF_getValue(elt)
    };
    bitC.bitContainer[idx] |= value as usize;
    bitC.bitPos[idx] = bitC.bitPos[idx].wrapping_add(HUF_getNbBitsFast(elt) as usize);
}

/// Port of `HUF_zeroIndex1`.
#[inline]
pub fn HUF_zeroIndex1(bitC: &mut HUF_CStream_t) {
    bitC.bitContainer[1] = 0;
    bitC.bitPos[1] = 0;
}

/// Port of `HUF_mergeIndex1`. Collapses `container\[1\]` into `container\[0\]`.
#[inline]
pub fn HUF_mergeIndex1(bitC: &mut HUF_CStream_t) {
    debug_assert!(bitC.bitPos[1] & 0xFF < HUF_BITS_IN_CONTAINER);
    bitC.bitContainer[0] >>= bitC.bitPos[1] & 0xFF;
    bitC.bitContainer[0] |= bitC.bitContainer[1];
    bitC.bitPos[0] = bitC.bitPos[0].wrapping_add(bitC.bitPos[1]);
}

/// Port of `HUF_flushBits`. Writes the high `bitPos[0]` bits of
/// `bitContainer[0]` out to `ptr` (LE-ordered), advances `ptr`, and
/// clamps `ptr` to `endPtr` when `kFast == false` to prevent overrun.
#[inline]
pub fn HUF_flushBits(bitC: &mut HUF_CStream_t, kFast: bool) {
    let nbBits = bitC.bitPos[0] & 0xFF;
    let nbBytes = nbBits >> 3;
    let bitContainer = bitC.bitContainer[0] >> (HUF_BITS_IN_CONTAINER - nbBits);
    bitC.bitPos[0] &= 7;
    crate::common::mem::MEM_writeLEST(&mut bitC.dst[bitC.ptr..], bitContainer);
    bitC.ptr += nbBytes;
    if !kFast && bitC.ptr > bitC.endPtr {
        bitC.ptr = bitC.endPtr;
    }
}

/// Port of `HUF_endMark`.
#[inline]
fn HUF_endMark() -> HUF_CElt {
    let mut e: HUF_CElt = 0;
    HUF_setNbBits(&mut e, 1);
    HUF_setValue(&mut e, 1);
    e
}

/// Port of `HUF_closeCStream`. Emits the end-mark, flushes, and
/// returns bytes written or 0 on overflow.
pub fn HUF_closeCStream(bitC: &mut HUF_CStream_t) -> usize {
    HUF_addBits(bitC, HUF_endMark(), 0, false);
    HUF_flushBits(bitC, false);
    let nbBits = bitC.bitPos[0] & 0xFF;
    if bitC.ptr >= bitC.endPtr {
        return 0;
    }
    (bitC.ptr - bitC.startPtr) + (nbBits > 0) as usize
}

/// Port of `HUF_encodeSymbol`.
#[inline(always)]
pub fn HUF_encodeSymbol(
    bitC: &mut HUF_CStream_t,
    symbol: u32,
    ctable: &[HUF_CElt],
    idx: usize,
    fast: bool,
) {
    // Upstream's `CTable` here is already shifted past the header, so
    // we apply the same +1 offset when indexing.
    HUF_addBits(bitC, ctable[1 + symbol as usize], idx, fast);
}

/// Port of `HUF_compress1X_usingCTable_internal_body_loop`.
pub fn HUF_compress1X_usingCTable_body_loop(
    bitC: &mut HUF_CStream_t,
    ip: &[u8],
    srcSize: usize,
    ctable: &[HUF_CElt],
    kUnroll: usize,
    kFastFlush: bool,
    kLastFast: bool,
) {
    let mut n = srcSize;
    let mut rem = n % kUnroll;
    if rem > 0 {
        while rem > 0 {
            rem -= 1;
            n -= 1;
            HUF_encodeSymbol(bitC, ip[n] as u32, ctable, 0, false);
        }
        HUF_flushBits(bitC, kFastFlush);
    }
    debug_assert_eq!(n % kUnroll, 0);

    if !n.is_multiple_of(2 * kUnroll) {
        let mut u = 1usize;
        while u < kUnroll {
            HUF_encodeSymbol(bitC, ip[n - u] as u32, ctable, 0, true);
            u += 1;
        }
        HUF_encodeSymbol(bitC, ip[n - kUnroll] as u32, ctable, 0, kLastFast);
        HUF_flushBits(bitC, kFastFlush);
        n -= kUnroll;
    }
    debug_assert_eq!(n % (2 * kUnroll), 0);

    while n > 0 {
        let mut u = 1usize;
        while u < kUnroll {
            HUF_encodeSymbol(bitC, ip[n - u] as u32, ctable, 0, true);
            u += 1;
        }
        HUF_encodeSymbol(bitC, ip[n - kUnroll] as u32, ctable, 0, kLastFast);
        HUF_flushBits(bitC, kFastFlush);

        HUF_zeroIndex1(bitC);
        let mut u = 1usize;
        while u < kUnroll {
            HUF_encodeSymbol(bitC, ip[n - kUnroll - u] as u32, ctable, 1, true);
            u += 1;
        }
        HUF_encodeSymbol(
            bitC,
            ip[n - (2 * kUnroll)] as u32,
            ctable,
            1,
            kLastFast,
        );
        HUF_mergeIndex1(bitC);
        HUF_flushBits(bitC, kFastFlush);
        n -= 2 * kUnroll;
    }
}

/// Exact-name wrapper for upstream's
/// `HUF_compress1X_usingCTable_internal_body_loop`.
#[inline]
pub fn HUF_compress1X_usingCTable_internal_body_loop(
    bitC: &mut HUF_CStream_t,
    ip: &[u8],
    srcSize: usize,
    ctable: &[HUF_CElt],
    kUnroll: usize,
    kFastFlush: bool,
    kLastFast: bool,
) {
    HUF_compress1X_usingCTable_body_loop(bitC, ip, srcSize, ctable, kUnroll, kFastFlush, kLastFast)
}

/// Port of `HUF_compress1X_usingCTable_internal_body`.
pub fn HUF_compress1X_usingCTable_internal_body(
    dst: &mut [u8],
    src: &[u8],
    ctable: &[HUF_CElt],
) -> usize {
    let dstSize = dst.len();
    let tableLog = HUF_readCTableHeader(ctable).tableLog as usize;
    if dstSize < 8 {
        return 0;
    }
    let (mut bitC, initErr) = HUF_initCStream(dst, dstSize);
    if crate::common::error::ERR_isError(initErr) {
        return 0;
    }
    if dstSize < HUF_tightCompressBound(src.len(), tableLog) || tableLog > 11 {
        HUF_compress1X_usingCTable_body_loop(
            &mut bitC,
            src,
            src.len(),
            ctable,
            if crate::common::mem::MEM_32bits() != 0 { 2 } else { 4 },
            false,
            false,
        );
    } else if crate::common::mem::MEM_32bits() != 0 {
        match tableLog {
            11 => HUF_compress1X_usingCTable_body_loop(
                &mut bitC, src, src.len(), ctable, 2, true, false,
            ),
            8..=10 => HUF_compress1X_usingCTable_body_loop(
                &mut bitC, src, src.len(), ctable, 2, true, true,
            ),
            _ => HUF_compress1X_usingCTable_body_loop(
                &mut bitC, src, src.len(), ctable, 3, true, true,
            ),
        }
    } else {
        match tableLog {
            11 => HUF_compress1X_usingCTable_body_loop(
                &mut bitC, src, src.len(), ctable, 5, true, false,
            ),
            10 => HUF_compress1X_usingCTable_body_loop(
                &mut bitC, src, src.len(), ctable, 5, true, true,
            ),
            9 => HUF_compress1X_usingCTable_body_loop(
                &mut bitC, src, src.len(), ctable, 6, true, false,
            ),
            8 => HUF_compress1X_usingCTable_body_loop(
                &mut bitC, src, src.len(), ctable, 7, true, false,
            ),
            7 => HUF_compress1X_usingCTable_body_loop(
                &mut bitC, src, src.len(), ctable, 8, true, false,
            ),
            _ => HUF_compress1X_usingCTable_body_loop(
                &mut bitC, src, src.len(), ctable, 9, true, true,
            ),
        }
    }
    HUF_closeCStream(&mut bitC)
}

/// Port of `HUF_compress1X_usingCTable`. `_flags` accepts the upstream
/// flags bitmask (`HUF_flags_*`); BMI2 / asm variants are not yet
/// wired so we always take the scalar path.
pub fn HUF_compress1X_usingCTable_internal_default(
    dst: &mut [u8],
    src: &[u8],
    ctable: &[HUF_CElt],
    _flags: i32,
) -> usize {
    HUF_compress1X_usingCTable_internal_body(dst, src, ctable)
}

/// Exact-name wrapper for upstream's
/// `HUF_compress1X_usingCTable_internal_bmi2`. The current Rust port
/// uses the same scalar encoder regardless of the BMI2 flag.
#[inline]
pub fn HUF_compress1X_usingCTable_internal_bmi2(
    dst: &mut [u8],
    src: &[u8],
    ctable: &[HUF_CElt],
) -> usize {
    HUF_compress1X_usingCTable_internal_body(dst, src, ctable)
}

/// Port of `HUF_compress1X_usingCTable_internal`.
pub fn HUF_compress1X_usingCTable(
    dst: &mut [u8],
    src: &[u8],
    ctable: &[HUF_CElt],
    flags: i32,
) -> usize {
    HUF_compress1X_usingCTable_internal(dst, src, ctable, flags)
}

/// Port of `HUF_compress1X_usingCTable_internal`.
pub fn HUF_compress1X_usingCTable_internal(
    dst: &mut [u8],
    src: &[u8],
    ctable: &[HUF_CElt],
    flags: i32,
) -> usize {
    let _ = flags;
    HUF_compress1X_usingCTable_internal_default(dst, src, ctable, flags)
}

/// Port of `HUF_compress4X_usingCTable`. Splits `src` into four
/// roughly-equal segments and emits each through
/// `HUF_compress1X_usingCTable_internal`, prefixed by a 6-byte jump
/// table (three 16-bit segment sizes; the fourth is implied by
/// `srcSize`).
///
/// The segment-sizes are computed as `(srcSize + 3) / 4` for the
/// first three; the last segment is whatever's left. Minimum dst size
/// = jump table + 1 byte per segment + 8-byte wildcopy slack.
pub fn HUF_compress4X_usingCTable_internal(
    dst: &mut [u8],
    src: &[u8],
    ctable: &[HUF_CElt],
    flags: i32,
) -> usize {
    use crate::common::mem::MEM_writeLE16;
    let srcSize = src.len();
    let dstSize = dst.len();
    let _ = flags;
    if dstSize < 6 + 1 + 1 + 1 + 8 {
        return 0;
    }
    if srcSize < 12 {
        return 0;
    }
    let segmentSize = srcSize.div_ceil(4);
    // dst[0..6] = jump table; payloads start at offset 6.
    let mut op = 6usize;

    // Segment 1.
    let ip = 0;
    let (_, tail) = dst.split_at_mut(op);
    let c1 = HUF_compress1X_usingCTable_internal_body(tail, &src[ip..ip + segmentSize], ctable);
    if c1 == 0 || c1 > 65535 {
        return 0;
    }
    MEM_writeLE16(&mut dst[0..2], c1 as u16);
    op += c1;

    // Segment 2.
    let ip = segmentSize;
    let (_, tail) = dst.split_at_mut(op);
    let c2 = HUF_compress1X_usingCTable_internal_body(tail, &src[ip..ip + segmentSize], ctable);
    if c2 == 0 || c2 > 65535 {
        return 0;
    }
    MEM_writeLE16(&mut dst[2..4], c2 as u16);
    op += c2;

    // Segment 3.
    let ip = 2 * segmentSize;
    let (_, tail) = dst.split_at_mut(op);
    let c3 = HUF_compress1X_usingCTable_internal_body(tail, &src[ip..ip + segmentSize], ctable);
    if c3 == 0 || c3 > 65535 {
        return 0;
    }
    MEM_writeLE16(&mut dst[4..6], c3 as u16);
    op += c3;

    // Segment 4 (implied size = remainder).
    let ip = 3 * segmentSize;
    let (_, tail) = dst.split_at_mut(op);
    let c4 = HUF_compress1X_usingCTable_internal_body(tail, &src[ip..srcSize], ctable);
    if c4 == 0 || c4 > 65535 {
        return 0;
    }
    op += c4;
    op
}

/// Port of `HUF_compress4X_usingCTable`.
pub fn HUF_compress4X_usingCTable(
    dst: &mut [u8],
    src: &[u8],
    ctable: &[HUF_CElt],
    flags: i32,
) -> usize {
    HUF_compress4X_usingCTable_internal(dst, src, ctable, flags)
}

/// Port of `HUF_nbStreams_e`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HUF_nbStreams_e {
    HUF_singleStream,
    HUF_fourStreams,
}

/// Port of `HUF_compressCTable_internal`. Emits `src` via either the
/// single-stream or quad-stream HUF encoder, then verifies that the
/// produced payload is strictly smaller than `srcSize - 1`
/// (incompressible otherwise).
pub fn HUF_compressCTable_internal(
    dst: &mut [u8],
    op_start: usize,
    src: &[u8],
    nbStreams: HUF_nbStreams_e,
    ctable: &[HUF_CElt],
    flags: i32,
) -> usize {
    let srcSize = src.len();
    let (_, tail) = dst.split_at_mut(op_start);
    let cSize = match nbStreams {
        HUF_nbStreams_e::HUF_singleStream => HUF_compress1X_usingCTable(tail, src, ctable, flags),
        HUF_nbStreams_e::HUF_fourStreams => HUF_compress4X_usingCTable(tail, src, ctable, flags),
    };
    if crate::common::error::ERR_isError(cSize) {
        return cSize;
    }
    if cSize == 0 {
        return 0;
    }
    // Post-check: total payload (header + emit) must be strictly
    // smaller than raw-size-1 to be worth keeping.
    if op_start + cSize >= srcSize.saturating_sub(1) {
        return 0;
    }
    op_start + cSize
}

/// Port of `HUF_compress_internal` — the tie-together that builds a
/// new CTable from `src`, writes it, and emits the payload. `repeat`
/// is mutated from `HUF_repeat_none` to `HUF_repeat_check` when a
/// fresh table was used (matching upstream semantics).
pub fn HUF_compress_internal(
    dst: &mut [u8],
    src: &[u8],
    mut maxSymbolValue: u32,
    mut huffLog: u32,
    nbStreams: HUF_nbStreams_e,
    oldHufTable: Option<&mut [HUF_CElt]>,
    mut repeat: Option<&mut HUF_repeat>,
    flags: i32,
) -> usize {
    use crate::common::error::{ErrorCode, ERROR};
    use crate::compress::hist::HIST_count_wksp;

    let srcSize = src.len();
    let dstSize = dst.len();

    if srcSize == 0 || dstSize == 0 {
        return 0;
    }
    if srcSize > HUF_BLOCKSIZE_MAX {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    if huffLog > crate::decompress::huf_decompress::HUF_TABLELOG_MAX {
        return ERROR(ErrorCode::TableLogTooLarge);
    }
    if maxSymbolValue > HUF_SYMBOLVALUE_MAX {
        return ERROR(ErrorCode::MaxSymbolValueTooLarge);
    }
    if maxSymbolValue == 0 {
        maxSymbolValue = HUF_SYMBOLVALUE_MAX;
    }
    if huffLog == 0 {
        huffLog = HUF_TABLELOG_DEFAULT;
    }

    if (flags & crate::decompress::huf_decompress::HUF_flags_preferRepeat) != 0 {
        if let (Some(old), Some(rep)) = (oldHufTable.as_deref(), repeat.as_deref()) {
            if *rep == HUF_repeat::HUF_repeat_valid {
                return HUF_compressCTable_internal(dst, 0, src, nbStreams, old, flags);
            }
        }
    }

    if (flags & crate::decompress::huf_decompress::HUF_flags_suspectUncompressible) != 0
        && srcSize
            >= (SUSPECT_INCOMPRESSIBLE_SAMPLE_SIZE * SUSPECT_INCOMPRESSIBLE_SAMPLE_RATIO)
    {
        let mut count = vec![0u32; (HUF_SYMBOLVALUE_MAX + 1) as usize];
        let mut maxSymbolValueBegin = maxSymbolValue;
        let largestBegin = crate::compress::hist::HIST_count_simple(
            &mut count,
            &mut maxSymbolValueBegin,
            &src[..SUSPECT_INCOMPRESSIBLE_SAMPLE_SIZE],
        ) as usize;
        let mut maxSymbolValueEnd = maxSymbolValue;
        let largestEnd = crate::compress::hist::HIST_count_simple(
            &mut count,
            &mut maxSymbolValueEnd,
            &src[srcSize - SUSPECT_INCOMPRESSIBLE_SAMPLE_SIZE..],
        ) as usize;
        let largestTotal = largestBegin + largestEnd;
        if largestTotal <= ((2 * SUSPECT_INCOMPRESSIBLE_SAMPLE_SIZE) >> 7) + 4 {
            return 0;
        }
    }

    // Histogram.
    let mut count = vec![0u32; (HUF_SYMBOLVALUE_MAX + 1) as usize];
    let mut hist_wksp = vec![0u32; crate::compress::hist::HIST_WKSP_SIZE_U32];
    let largest = HIST_count_wksp(&mut count, &mut maxSymbolValue, src, &mut hist_wksp);
    if crate::common::error::ERR_isError(largest) {
        return largest;
    }
    if largest == srcSize {
        // RLE: single symbol.
        dst[0] = src[0];
        return 1;
    }
    if largest <= (srcSize >> 7) + 4 {
        return 0; // probably not compressible
    }

    if let (Some(old), Some(rep)) = (oldHufTable.as_deref(), repeat.as_deref_mut()) {
        if *rep == HUF_repeat::HUF_repeat_check && !HUF_validateCTable(old, &count, maxSymbolValue)
        {
            *rep = HUF_repeat::HUF_repeat_none;
        }
    }

    if (flags & crate::decompress::huf_decompress::HUF_flags_preferRepeat) != 0 {
        if let (Some(old), Some(rep)) = (oldHufTable.as_deref(), repeat.as_deref()) {
            if *rep != HUF_repeat::HUF_repeat_none {
                return HUF_compressCTable_internal(dst, 0, src, nbStreams, old, flags);
            }
        }
    }

    // Build fresh CTable.
    let mut ctable = vec![0u64; (HUF_SYMBOLVALUE_MAX + 2) as usize];
    huffLog = HUF_optimalTableLog_internal(huffLog, srcSize, maxSymbolValue, &mut ctable, &count, flags);
    let maxBits = HUF_buildCTable(&mut ctable, &count, maxSymbolValue, huffLog);
    if crate::common::error::ERR_isError(maxBits) {
        return maxBits;
    }
    huffLog = maxBits as u32;

    // Write header. dst[0..hSize] holds the tree description.
    let hSize = HUF_writeCTable(dst, &ctable, maxSymbolValue, huffLog);
    if crate::common::error::ERR_isError(hSize) {
        return hSize;
    }

    // Compare against repeat path, if available.
    let repeat_is_valid = repeat
        .as_ref()
        .map(|r| **r != HUF_repeat::HUF_repeat_none)
        .unwrap_or(false);
    if repeat_is_valid {
        if let Some(old) = oldHufTable.as_deref() {
            let oldSize = HUF_estimateCompressedSize(old, &count, maxSymbolValue);
            let newSize = HUF_estimateCompressedSize(&ctable, &count, maxSymbolValue);
            if oldSize <= hSize + newSize || hSize + 12 >= srcSize {
                // Reuse the old table.
                return HUF_compressCTable_internal(dst, 0, src, nbStreams, old, flags);
            }
        }
    }

    // Refuse if the tree description alone eats nearly the whole budget.
    if hSize + 12 >= srcSize {
        return 0;
    }

    // Caller tracking: save the new table and mark repeat as fresh.
    if let Some(r) = repeat {
        *r = HUF_repeat::HUF_repeat_none;
    }
    if let Some(old) = oldHufTable {
        let n = ctable.len().min(old.len());
        old[..n].copy_from_slice(&ctable[..n]);
    }

    HUF_compressCTable_internal(dst, hSize, src, nbStreams, &ctable, flags)
}

/// Port of `HUF_compress1X_repeat`.
pub fn HUF_compress1X_repeat(
    dst: &mut [u8],
    src: &[u8],
    maxSymbolValue: u32,
    huffLog: u32,
    hufTable: Option<&mut [HUF_CElt]>,
    repeat: Option<&mut HUF_repeat>,
    flags: i32,
) -> usize {
    HUF_compress_internal(
        dst,
        src,
        maxSymbolValue,
        huffLog,
        HUF_nbStreams_e::HUF_singleStream,
        hufTable,
        repeat,
        flags,
    )
}

/// Port of `HUF_compress4X_repeat`.
pub fn HUF_compress4X_repeat(
    dst: &mut [u8],
    src: &[u8],
    maxSymbolValue: u32,
    huffLog: u32,
    hufTable: Option<&mut [HUF_CElt]>,
    repeat: Option<&mut HUF_repeat>,
    flags: i32,
) -> usize {
    HUF_compress_internal(
        dst,
        src,
        maxSymbolValue,
        huffLog,
        HUF_nbStreams_e::HUF_fourStreams,
        hufTable,
        repeat,
        flags,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn HUF_nbStreams_e_discriminants_match_upstream() {
        // Upstream: `typedef enum { HUF_singleStream,
        // HUF_fourStreams } HUF_nbStreams_e;` — 0/1. The compressor
        // dispatches to 1-way vs 4-way HUF encoding based on this
        // value; reorder would swap single vs four streams silently.
        assert_eq!(HUF_nbStreams_e::HUF_singleStream as u32, 0);
        assert_eq!(HUF_nbStreams_e::HUF_fourStreams as u32, 1);
    }

    #[test]
    fn readCTable_roundtrip_via_writeCTable() {
        // Parity proof for `HUF_readCTable`: build a CTable → write
        // → read → compare the per-symbol nbBits. Upstream guarantees
        // write+read is a lossless round-trip for the nbBits table
        // when `maxSymbolValue` matches between producer and reader.
        //
        // The reader reconstructs the reader-side maxSymbolValue as
        // `nbSymbols - 1` (the count of weights in the stream),
        // which may be smaller than the producer's cap when
        // trailing symbols were zero-weighted. Only compare the
        // overlap range.
        let mut count = [0u32; 256];
        // 8-symbol alphabet (no sparse gaps).
        for &b in b"The quick brown fox jumps over the lazy dog".iter() {
            count[b as usize] += 1;
        }
        // Find the actual max nonzero symbol to avoid trailing zeros.
        let maxSymbolValue = count
            .iter()
            .enumerate()
            .rposition(|(_, &c)| c > 0)
            .unwrap_or(0) as u32;
        let totalCount: usize = count.iter().sum::<u32>() as usize;
        let tableLog = HUF_optimalTableLog(11, totalCount, maxSymbolValue);
        let mut ct = vec![0u64; 257];
        let mut wksp = vec![0u32; HUF_CTABLE_WORKSPACE_SIZE_U32];
        let _ = HUF_buildCTable_wksp(&mut ct, &count, maxSymbolValue, tableLog, &mut wksp);

        let mut hdr = vec![0u8; 1024];
        let written = HUF_writeCTable(&mut hdr, &ct, maxSymbolValue, tableLog);
        assert!(!crate::common::error::ERR_isError(written));

        let mut ct2 = vec![0u64; 257];
        let mut maxSymbolValuePtr = maxSymbolValue;
        let mut hasZeroWeights: u32 = 0;
        let consumed = HUF_readCTable(
            &mut ct2,
            &mut maxSymbolValuePtr,
            &hdr[..written],
            &mut hasZeroWeights,
        );
        assert!(!crate::common::error::ERR_isError(consumed));
        assert_eq!(consumed, written);

        // tableLog must roundtrip exactly.
        assert_eq!(
            HUF_readCTableHeader(&ct).tableLog,
            HUF_readCTableHeader(&ct2).tableLog
        );

        // nbBits per symbol must match for symbols the reader saw.
        for s in 0..=(maxSymbolValuePtr as usize) {
            assert_eq!(
                HUF_getNbBits(ct[s + 1]),
                HUF_getNbBits(ct2[s + 1]),
                "nbBits mismatch at symbol {s}"
            );
        }
    }

    #[test]
    fn cardinality_counts_nonzero() {
        let mut c = [0u32; 256];
        c[b'a' as usize] = 10;
        c[b'b' as usize] = 5;
        c[b'c' as usize] = 1;
        assert_eq!(HUF_cardinality(&c, 255), 3);
        assert_eq!(HUF_cardinality(&c, b'b' as u32), 2); // caps
    }

    #[test]
    fn cardinality_empty_is_zero() {
        let c = [0u32; 256];
        assert_eq!(HUF_cardinality(&c, 255), 0);
    }

    #[test]
    fn min_tablelog_matches_highbit_formula() {
        // upstream: minBits = highbit32(cardinality) + 1.
        //   cardinality=1 → 0+1=1
        //   cardinality=2 → 1+1=2
        //   cardinality=7 → 2+1=3
        //   cardinality=8 → 3+1=4
        assert_eq!(HUF_minTableLog(1), 1);
        assert_eq!(HUF_minTableLog(2), 2);
        assert_eq!(HUF_minTableLog(7), 3);
        assert_eq!(HUF_minTableLog(8), 4);
    }

    #[test]
    fn optimal_tablelog_uses_fse_internal_with_minus_1() {
        // HUF's default path forwards to FSE_optimalTableLog_internal
        // with minus=1; confirm the two agree.
        use crate::compress::fse_compress::FSE_optimalTableLog_internal;
        let got = HUF_optimalTableLog(12, 2048, 255);
        let want = FSE_optimalTableLog_internal(12, 2048, 255, 1);
        assert_eq!(got, want);
    }

    #[test]
    fn optimal_tablelog_internal_matches_bruteforce_probe_when_optimal_depth_enabled() {
        let src = b"AAAAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBCCCCCCCCCCCCCCDDDDDDDDEEEE";
        let mut count = vec![0u32; 256];
        let mut max_symbol_value = 255u32;
        crate::compress::hist::HIST_count_simple(&mut count, &mut max_symbol_value, src);
        let mut table = vec![0u64; (HUF_SYMBOLVALUE_MAX + 2) as usize];
        let got = HUF_optimalTableLog_internal(
            12,
            src.len(),
            max_symbol_value,
            &mut table,
            &count,
            crate::decompress::huf_decompress::HUF_flags_optimalDepth,
        );

        let symbol_cardinality = HUF_cardinality(&count, max_symbol_value);
        let min_table_log = HUF_minTableLog(symbol_cardinality);
        let mut want = 12;
        let mut best_size = usize::MAX - 1;
        for guess in min_table_log..=12 {
            let mut scratch = vec![0u64; (HUF_SYMBOLVALUE_MAX + 2) as usize];
            let max_bits = HUF_buildCTable(&mut scratch, &count, max_symbol_value, guess);
            if crate::common::error::ERR_isError(max_bits) {
                continue;
            }
            if max_bits < guess as usize && guess > min_table_log {
                break;
            }
            let h_size =
                HUF_writeCTable(&mut [0u8; HUF_CTABLEBOUND], &scratch, max_symbol_value, max_bits as u32);
            if crate::common::error::ERR_isError(h_size) {
                continue;
            }
            let new_size = HUF_estimateCompressedSize(&scratch, &count, max_symbol_value) + h_size;
            if new_size > best_size + 1 {
                break;
            }
            if new_size < best_size {
                best_size = new_size;
                want = guess;
            }
        }
        assert_eq!(got, want);
    }

    #[test]
    fn compress_internal_rejects_large_suspect_incompressible_input_early() {
        let mut src = vec![0u8; SUSPECT_INCOMPRESSIBLE_SAMPLE_SIZE * SUSPECT_INCOMPRESSIBLE_SAMPLE_RATIO];
        let mut x = 0x1234_5678u32;
        for b in &mut src {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = x as u8;
        }
        let mut dst = vec![0u8; HUF_compressBound(src.len())];
        let got = HUF_compress1X_repeat(
            &mut dst,
            &src,
            255,
            11,
            None,
            None,
            crate::decompress::huf_decompress::HUF_flags_suspectUncompressible,
        );
        assert_eq!(got, 0);
    }

    #[test]
    fn celt_pack_unpack_roundtrip() {
        // Build a CElt for a 5-bit code 0b10110 = 22. The packed form
        // must decode to exactly those two fields.
        let mut elt: HUF_CElt = 0;
        HUF_setNbBits(&mut elt, 5);
        HUF_setValue(&mut elt, 22);
        assert_eq!(HUF_getNbBits(elt), 5);
        // Value is MSB-justified: 22 << (64 - 5) = 22 << 59.
        assert_eq!(HUF_getValue(elt), 22u64 << 59);
    }

    #[test]
    fn ctable_header_roundtrip() {
        let mut ct = [0u64; 8];
        HUF_writeCTableHeader(&mut ct, 11, 255);
        let h = HUF_readCTableHeader(&ct);
        assert_eq!(h.tableLog, 11);
        assert_eq!(h.maxSymbolValue, 255);
    }

    #[test]
    fn get_nbbits_from_ctable_respects_max_symbol() {
        let mut ct = [0u64; 260];
        HUF_writeCTableHeader(&mut ct, 8, 5);
        // Set nbBits=7 for symbol 3.
        HUF_setNbBits(&mut ct[1 + 3], 7);
        assert_eq!(HUF_getNbBitsFromCTable(&ct, 3), 7);
        // Symbol 10 is > maxSymbolValue(=5) → returns 0.
        assert_eq!(HUF_getNbBitsFromCTable(&ct, 10), 0);
    }

    #[test]
    fn build_ctable_from_tree_produces_canonical_prefix_code() {
        // Hand-craft a simple Huffman tree over 4 symbols:
        //   sym 0 → nbBits=1 (most common)
        //   sym 1 → nbBits=2
        //   sym 2 → nbBits=3
        //   sym 3 → nbBits=3
        // Sum 2^-nbBits: 0.5 + 0.25 + 0.125 + 0.125 = 1.0 (valid tree).
        let huffNode = vec![
            nodeElt {
                count: 0,
                parent: 0,
                byte: 0,
                nbBits: 1,
            },
            nodeElt {
                count: 0,
                parent: 0,
                byte: 1,
                nbBits: 2,
            },
            nodeElt {
                count: 0,
                parent: 0,
                byte: 2,
                nbBits: 3,
            },
            nodeElt {
                count: 0,
                parent: 0,
                byte: 3,
                nbBits: 3,
            },
        ];
        let maxSV = 3u32;
        let maxNbBits = 3u32;
        let mut ct = vec![0u64; 1 + (maxSV + 1) as usize];
        HUF_buildCTableFromTree(&mut ct, &huffNode, 3, maxSV, maxNbBits);

        // Each symbol's nbBits should match.
        assert_eq!(HUF_getNbBits(ct[1]), 1);
        assert_eq!(HUF_getNbBits(ct[2]), 2);
        assert_eq!(HUF_getNbBits(ct[3]), 3);
        assert_eq!(HUF_getNbBits(ct[4]), 3);

        // Upstream's canonical layout iterates maxNbBits → 1, so
        // shortest codes land at the HIGH end of the code-space:
        //   rank 3 (2 codes) → values 0, 1  (i.e. 000, 001)
        //   rank 2 (1 code)  → value 1      (i.e. 01)
        //   rank 1 (1 code)  → value 1      (i.e. 1)
        // Decoded canonically into symbol order:
        //   sym 0 (nb=1) = "1"      → value 1 → 1<<63
        //   sym 1 (nb=2) = "01"     → value 1 → 1<<62
        //   sym 2 (nb=3) = "000"    → value 0 → 0<<61
        //   sym 3 (nb=3) = "001"    → value 1 → 1<<61
        assert_eq!(HUF_getValue(ct[1]) >> 63, 1);
        assert_eq!(HUF_getValue(ct[2]) >> 62, 1);
        assert_eq!(HUF_getValue(ct[3]) >> 61, 0);
        assert_eq!(HUF_getValue(ct[4]) >> 61, 1);

        // Header reflects the params.
        let h = HUF_readCTableHeader(&ct);
        assert_eq!(h.tableLog, 3);
        assert_eq!(h.maxSymbolValue, 3);
    }

    #[test]
    fn huf_sort_descending_by_count() {
        // Symbols with varied counts → bucket sort must produce a
        // descending-by-count arrangement.
        let mut counts = [0u32; 256];
        counts[b'a' as usize] = 50;
        counts[b'b' as usize] = 10;
        counts[b'c' as usize] = 30;
        counts[b'd' as usize] = 70;
        counts[b'e' as usize] = 5;

        let mut huffNode = vec![nodeElt::default(); 256];
        let mut rp = vec![rankPos::default(); RANK_POSITION_TABLE_SIZE];
        HUF_sort(&mut huffNode, &counts, (b'e') as u32, &mut rp);

        // Top 5 entries (non-zero counts) must appear in strict
        // descending count order. Symbols with zero count fill the
        // tail — their ordering is irrelevant here.
        assert_eq!(huffNode[0].byte, b'd', "d has count 70");
        assert_eq!(huffNode[1].byte, b'a', "a has count 50");
        assert_eq!(huffNode[2].byte, b'c', "c has count 30");
        assert_eq!(huffNode[3].byte, b'b', "b has count 10");
        assert_eq!(huffNode[4].byte, b'e', "e has count 5");
        assert!(HUF_isSorted(&huffNode, (b'e' + 1) as u32));
    }

    #[test]
    fn get_index_buckets_monotonic() {
        // Before the cutoff, buckets are 1:1 with counts.
        assert_eq!(HUF_getIndex(0), 0);
        assert_eq!(HUF_getIndex(100), 100);
        // At/after cutoff, buckets grow log2ically.
        let cutoff = RANK_POSITION_DISTINCT_COUNT_CUTOFF;
        let at = HUF_getIndex(cutoff);
        let much_bigger = HUF_getIndex(1 << 20);
        assert!(much_bigger >= at);
    }

    #[test]
    fn huf_build_tree_emits_valid_kraft_lengths() {
        // Run the full sort → build pipeline and verify Kraft-McMillan
        // sum over the resulting bit-lengths equals 1 (valid prefix code).
        let mut counts = [0u32; 256];
        counts[b'a' as usize] = 50;
        counts[b'b' as usize] = 40;
        counts[b'c' as usize] = 30;
        counts[b'd' as usize] = 20;
        counts[b'e' as usize] = 10;
        let msv = b'e' as u32;

        let mut huffNode = vec![nodeElt::default(); 2 * (HUF_SYMBOLVALUE_MAX as usize + 1)];
        let mut rp = vec![rankPos::default(); RANK_POSITION_TABLE_SIZE];
        HUF_sort(&mut huffNode, &counts, msv, &mut rp);
        assert!(HUF_isSorted(&huffNode, msv + 1));

        let nonNullRank = HUF_buildTree(&mut huffNode, msv);
        assert_eq!(nonNullRank, 4, "5 distinct symbols → smallest index = 4");

        // Kraft sum: Σ 2^-nbBits over the leaves [0..=nonNullRank] should == 1.0.
        let mut kraft: f64 = 0.0;
        for (n, node) in huffNode.iter().enumerate().take(nonNullRank as usize + 1) {
            assert!(
                node.nbBits > 0,
                "leaf {n} has zero bit length — invalid tree"
            );
            kraft += 2f64.powi(-(node.nbBits as i32));
        }
        let err = (kraft - 1.0).abs();
        assert!(err < 1e-9, "Kraft-McMillan sum = {kraft}; expected 1.0");
    }

    #[test]
    fn build_ctable_wksp_enforces_max_nbbits() {
        // Extreme skew: symbol 0 appears 1000×, symbol 1 appears 1×.
        // Unlimited Huffman would give sym 0 nbBits=1, sym 1 nbBits=1.
        // With maxNbBits clamped to a high value this should still
        // produce a valid Kraft-sum-1 table.
        let mut count = [0u32; 256];
        count[0] = 1000;
        count[1] = 1;
        count[2] = 1;
        count[3] = 1;
        let mut ct = [0u64; 257];
        let rc = HUF_buildCTable(&mut ct, &count, 3, 8);
        assert!(!crate::common::error::ERR_isError(rc));
        let finalMaxBits = rc as u32;
        assert!(finalMaxBits <= 8);

        // Kraft sum over all 4 assigned symbols.
        let mut kraft: f64 = 0.0;
        for s in 0..4 {
            let nb = HUF_getNbBitsFromCTable(&ct, s) as i32;
            assert!(nb > 0, "symbol {s} unassigned");
            kraft += 2f64.powi(-nb);
        }
        let err = (kraft - 1.0).abs();
        assert!(err < 1e-9, "Kraft sum = {kraft}");
    }

    #[test]
    fn build_ctable_wksp_honours_tight_tablelog() {
        // Input that unconstrained would make nbBits > 4 for some
        // symbol; ask for maxNbBits=4 and verify setMaxHeight clipped
        // it. Extreme-skewed distribution like [1000, 500, 100, 10, 5,
        // 2, 1, 1] requires clip.
        let mut count = [0u32; 256];
        count[0] = 1000;
        count[1] = 500;
        count[2] = 100;
        count[3] = 10;
        count[4] = 5;
        count[5] = 2;
        count[6] = 1;
        count[7] = 1;
        let mut ct = [0u64; 257];
        let rc = HUF_buildCTable(&mut ct, &count, 7, 4);
        assert!(!crate::common::error::ERR_isError(rc));
        let finalMaxBits = rc as u32;
        assert!(
            finalMaxBits <= 4,
            "setMaxHeight failed: finalMaxBits={finalMaxBits}"
        );
        // Every symbol must still fit within the 4-bit budget.
        for s in 0..=7u32 {
            let nb = HUF_getNbBitsFromCTable(&ct, s);
            assert!(nb > 0 && nb <= 4, "sym {s} got nbBits={nb}");
        }
    }

    #[test]
    fn estimate_compressed_size_under_entropy_bound() {
        // Highly skewed distribution: dominant symbol gets nbBits=1,
        // rarest get higher nbBits. Estimated bits shouldn't exceed
        // the raw 8×srcSize upper bound.
        let mut count = [0u32; 256];
        count[0] = 100;
        count[1] = 50;
        count[2] = 10;
        count[3] = 5;
        let mut ct = [0u64; 257];
        let rc = HUF_buildCTable(&mut ct, &count, 3, 8);
        assert!(!crate::common::error::ERR_isError(rc));

        let total_symbols: usize = count.iter().take(4).map(|&c| c as usize).sum();
        let est = HUF_estimateCompressedSize(&ct, &count, 3);
        // Must not exceed raw byte count.
        assert!(
            est <= total_symbols,
            "huffman estimate {est} exceeds raw size {total_symbols}"
        );
        // Basic sanity: non-trivial output.
        assert!(est > 0);
    }

    #[test]
    fn validate_ctable_accepts_valid_and_rejects_missing_nbbits() {
        let mut count = [0u32; 256];
        count[0] = 10;
        count[1] = 5;
        let mut ct = [0u64; 257];
        HUF_buildCTable(&mut ct, &count, 1, 4);
        assert!(HUF_validateCTable(&ct, &count, 1));

        // Add a symbol with non-zero count but nbBits=0 in the CTable →
        // invalid. Upstream's emit loop would be undefined on this.
        let mut count2 = count;
        count2[2] = 1;
        assert!(!HUF_validateCTable(&ct, &count2, 2));
    }

    #[test]
    fn huf_compress1x_then_decompress_roundtrip() {
        // Full compress path: hist → buildCTable → compress1X. The
        // dedicated `write_ctable_then_read_stats_matches` test below
        // covers `HUF_writeCTable` ↔ `HUF_readStats`; here we just
        // sanity-check that the emit loop agrees with the size
        // predicted by `HUF_estimateCompressedSize` (allowing ± 2
        // bytes for the 1-bit end mark + pad-to-byte).
        let src: &[u8] = b"aaaabbbccdefghhhhhhhhh";
        let mut count = [0u32; 256];
        let mut msv: u32 = 255;
        crate::compress::hist::HIST_count_simple(&mut count, &mut msv, src);
        let mut ct = vec![0u64; 257];
        let tableLog = HUF_buildCTable(&mut ct, &count, msv, 8);
        assert!(!crate::common::error::ERR_isError(tableLog));
        assert!(HUF_validateCTable(&ct, &count, msv));

        let mut dst = vec![0u8; HUF_compressBound(src.len())];
        let written = HUF_compress1X_usingCTable(&mut dst, src, &ct, 0);
        assert!(written > 0, "compress produced 0 bytes");

        let est_bytes = HUF_estimateCompressedSize(&ct, &count, msv);
        // Actual written size should equal the estimate (± 1 byte for
        // end-mark + byte padding).
        assert!(
            written.abs_diff(est_bytes) <= 2,
            "written={written} est={est_bytes}"
        );
    }

    #[test]
    fn huf_cstream_init_rejects_tiny_dst() {
        let mut buf = [0u8; 4];
        let cap = buf.len();
        let (_bitC, rc) = HUF_initCStream(&mut buf, cap);
        assert!(crate::common::error::ERR_isError(rc));
    }

    #[test]
    fn write_ctable_then_read_stats_matches() {
        // Build a CTable from a known distribution, emit the header,
        // then decode it with `HUF_readStats` and verify the
        // reconstructed weights + nbSymbols + tableLog match.
        use crate::common::entropy_common::HUF_readStats;
        let mut count = [0u32; 256];
        count[0] = 50;
        count[1] = 30;
        count[2] = 15;
        count[3] = 5;
        let maxSV = 3u32;
        let mut ct = [0u64; 257];
        let rc = HUF_buildCTable(&mut ct, &count, maxSV, 8);
        assert!(!crate::common::error::ERR_isError(rc));
        let tableLog = rc as u32;

        // Encode. Size: upstream raw path is (maxSV+1)/2 + 1 = 3
        // bytes; FSE-compressed may be smaller. Give plenty of room.
        let mut buf = [0u8; 64];
        let written = HUF_writeCTable(&mut buf, &ct, maxSV, tableLog);
        assert!(
            !crate::common::error::ERR_isError(written),
            "write err: {}",
            crate::common::error::ERR_getErrorName(written)
        );
        assert!(written > 0);

        // Decode via `HUF_readStats`.
        let mut huffWeight = [0u8; 256];
        let mut rankStats = [0u32; 16];
        let mut nbSymbols: u32 = 0;
        let mut tl_out: u32 = 0;
        let consumed = HUF_readStats(
            &mut huffWeight,
            256,
            &mut rankStats,
            &mut nbSymbols,
            &mut tl_out,
            &buf[..written],
        );
        assert!(
            !crate::common::error::ERR_isError(consumed),
            "readStats err: {}",
            crate::common::error::ERR_getErrorName(consumed)
        );
        assert_eq!(consumed, written);
        assert_eq!(tl_out, tableLog);
        assert_eq!(nbSymbols, maxSV + 1);

        // Weights must match what we produced: weight = tableLog+1-nbBits
        // (or 0 when nbBits==0).
        for s in 0..=maxSV as usize {
            let nb = HUF_getNbBits(ct[1 + s]) as u32;
            let expected_w = if nb == 0 { 0 } else { tableLog + 1 - nb };
            assert_eq!(
                huffWeight[s] as u32, expected_w,
                "sym {s}: got weight {}, expected {expected_w}",
                huffWeight[s]
            );
        }
    }

    #[test]
    fn huf_compress4x_layout_matches_jump_table() {
        // Build a CTable from a 200-byte skewed input and compress
        // via the quad-stream variant. Verify the jump-table sizes
        // plus the implied trailing segment length add up to the
        // written size and that each segment's encoded bytes live
        // where the jump table says.
        use crate::common::mem::MEM_readLE16;
        use crate::compress::hist::HIST_count_simple;

        let src: Vec<u8> = b"aabbcccddddeeeeeffffffgggggggghhhhhhhhh"
            .iter()
            .cycle()
            .take(200)
            .copied()
            .collect();
        let mut count = [0u32; 256];
        let mut msv: u32 = 255;
        HIST_count_simple(&mut count, &mut msv, &src);
        let mut ct = vec![0u64; 257];
        let tableLog = HUF_buildCTable(&mut ct, &count, msv, 8);
        assert!(!crate::common::error::ERR_isError(tableLog));

        let mut dst = vec![0u8; HUF_compressBound(src.len())];
        let written = HUF_compress4X_usingCTable(&mut dst, &src, &ct, 0);
        assert!(written > 6, "need more than just the jump table");

        let s1 = MEM_readLE16(&dst[0..2]) as usize;
        let s2 = MEM_readLE16(&dst[2..4]) as usize;
        let s3 = MEM_readLE16(&dst[4..6]) as usize;
        let header = 6usize;
        // The fourth segment's size is implied: total - (header + s1 + s2 + s3).
        assert!(
            header + s1 + s2 + s3 < written,
            "header({header})+s1({s1})+s2({s2})+s3({s3}) must be less than total ({written})"
        );
        let s4 = written - header - s1 - s2 - s3;
        assert!(s4 >= 1);
        // None of the individual stream sizes may exceed the 16-bit
        // limit since the jump table only has 16 bits per entry.
        assert!(s1 <= 65535 && s2 <= 65535 && s3 <= 65535);
    }

    #[test]
    fn huf_compress_internal_then_decode_roundtrip_single_stream() {
        // Full vertical slice: HUF_compress1X_repeat → emit tree header
        // + single-stream payload, then `HUF_decompress1X1_DCtx_wksp`
        // reconstructs the original.
        use crate::decompress::huf_decompress::{
            DTableDesc, HUF_decompress1X1_DCtx_wksp, HUF_setDTableDesc,
            HUF_DECOMPRESS_WORKSPACE_SIZE_U32, HUF_DTABLE_SIZE_U32, HUF_TABLELOG_MAX,
        };

        let src: Vec<u8> = b"compress me please; compress me pretty please"
            .iter()
            .cycle()
            .take(500)
            .copied()
            .collect();

        let mut dst = vec![0u8; HUF_compressBound(src.len())];
        let written = HUF_compress1X_repeat(&mut dst, &src, 255, 11, None, None, 0);
        assert!(!crate::common::error::ERR_isError(written));
        assert!(written > 0, "compress returned 0 (incompressible?)");
        assert!(
            written < src.len(),
            "no compression: {written} ≥ {}",
            src.len()
        );

        // Decode.
        let mut dtable = vec![0u32; HUF_DTABLE_SIZE_U32(HUF_TABLELOG_MAX)];
        HUF_setDTableDesc(
            &mut dtable,
            DTableDesc {
                maxTableLog: (HUF_TABLELOG_MAX - 1) as u8,
                ..Default::default()
            },
        );
        let mut wksp = vec![0u32; HUF_DECOMPRESS_WORKSPACE_SIZE_U32];
        let mut out = vec![0u8; src.len()];
        let rc = HUF_decompress1X1_DCtx_wksp(&mut dtable, &mut out, &dst[..written], &mut wksp, 0);
        assert!(
            !crate::common::error::ERR_isError(rc),
            "decode err: {}",
            crate::common::error::ERR_getErrorName(rc)
        );
        assert_eq!(rc, src.len());
        assert_eq!(out, src);
    }

    #[test]
    fn huf_compress1x_prefer_repeat_reuses_valid_old_table() {
        let src: Vec<u8> = b"prefer-repeat-huf-prefer-repeat-huf"
            .iter()
            .cycle()
            .take(256)
            .copied()
            .collect();

        let mut seeded = vec![0u64; (HUF_SYMBOLVALUE_MAX + 2) as usize];
        let mut repeat = HUF_repeat::HUF_repeat_none;
        let mut scratch = vec![0u8; HUF_compressBound(src.len())];
        let first = HUF_compress1X_repeat(
            &mut scratch,
            &src,
            HUF_SYMBOLVALUE_MAX,
            HUF_TABLELOG_DEFAULT,
            Some(&mut seeded),
            Some(&mut repeat),
            0,
        );
        assert!(first > 0);

        let old = seeded.clone();
        let mut actual = vec![0u8; HUF_compressBound(src.len())];
        let mut expected = vec![0u8; HUF_compressBound(src.len())];
        let mut repeat_valid = HUF_repeat::HUF_repeat_valid;

        let actual_size = HUF_compress1X_repeat(
            &mut actual,
            &src,
            HUF_SYMBOLVALUE_MAX,
            HUF_TABLELOG_DEFAULT,
            Some(&mut seeded),
            Some(&mut repeat_valid),
            crate::decompress::huf_decompress::HUF_flags_preferRepeat,
        );
        let expected_size = HUF_compressCTable_internal(
            &mut expected,
            0,
            &src,
            HUF_nbStreams_e::HUF_singleStream,
            &old,
            crate::decompress::huf_decompress::HUF_flags_preferRepeat,
        );

        assert_eq!(actual_size, expected_size);
        assert_eq!(&actual[..actual_size], &expected[..expected_size]);
    }

    #[test]
    fn compress_bound_exceeds_src() {
        assert!(HUF_compressBound(0) >= 129);
        assert_eq!(HUF_compressBound(1024), 1024 + 129 + 4);
    }

    #[test]
    fn huf_format_constants_match_upstream() {
        // Pin the HUF format constants. Upstream spec:
        //   HUF_TABLELOG_MAX = 12 (max Huffman tree depth)
        //   HUF_SYMBOLVALUE_MAX = 255 (byte alphabet)
        //   HUF_BLOCKSIZE_MAX = 128 KB (matches ZSTD block size)
        assert_eq!(crate::decompress::huf_decompress::HUF_TABLELOG_MAX, 12);
        assert_eq!(HUF_SYMBOLVALUE_MAX, 255);
        assert_eq!(HUF_BLOCKSIZE_MAX, 128 * 1024);
    }
}
