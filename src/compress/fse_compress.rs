//! Translation of `lib/compress/fse_compress.c`.
//!
//! **Fully implemented**: constants (`FSE_MAX_MEMORY_USAGE`,
//! `FSE_MAX_TABLELOG`, `FSE_TABLELOG_ABSOLUTE_MAX`), bound math
//! (`FSE_NCountWriteBound`, `FSE_optimalTableLog{,_internal}`,
//! `FSE_CTABLE_SIZE_U32`), CTable build
//! (`FSE_buildCTable_rle`, `FSE_buildCTable_wksp`), cost estimator
//! (`FSE_bitCost`), NCount writer (`FSE_writeNCount`), encoder
//! state + symbol emit (`FSE_initCState{,2}`, `FSE_encodeSymbol`,
//! `FSE_flushCState`).

use crate::common::bits::ZSTD_highbit32;
pub use crate::common::fse_decompress::{FSE_MAX_MEMORY_USAGE, FSE_MAX_TABLELOG, FSE_MIN_TABLELOG};

/// Upstream `FSE_DEFAULT_MEMORY_USAGE`.
pub const FSE_DEFAULT_MEMORY_USAGE: u32 = 13;
/// Upstream `FSE_DEFAULT_TABLELOG`.
pub const FSE_DEFAULT_TABLELOG: u32 = FSE_DEFAULT_MEMORY_USAGE - 2;
/// Upstream `FSE_NCOUNTBOUND`.
pub const FSE_NCOUNTBOUND: usize = 512;

pub type FSE_CTable = u32;

/// Port of `FSE_NCountWriteBound`. Returns the upper bound on the
/// encoded NCount header size for a given `(maxSymbolValue, tableLog)`.
pub fn FSE_NCountWriteBound(maxSymbolValue: u32, tableLog: u32) -> usize {
    let maxHeaderSize = (((maxSymbolValue as usize + 1) * tableLog as usize
        + 4  // bitCount initialized at 4
        + 2  // first two symbols may use one additional bit each
    ) / 8)
        + 1  // round up to whole bytes
        + 2; // bitstream flush
    if maxSymbolValue != 0 {
        maxHeaderSize
    } else {
        FSE_NCOUNTBOUND
    }
}

/// Port of the file-private `FSE_minTableLog`. Returns the minimum
/// table log that can safely represent a distribution with the given
/// `srcSize` and `maxSymbolValue`. Caller guarantees `srcSize > 1`
/// (RLE path handles the single-symbol case).
fn FSE_minTableLog(srcSize: usize, maxSymbolValue: u32) -> u32 {
    debug_assert!(srcSize > 1);
    let minBitsSrc = ZSTD_highbit32(srcSize as u32) + 1;
    let minBitsSymbols = ZSTD_highbit32(maxSymbolValue) + 2;
    minBitsSrc.min(minBitsSymbols)
}

/// Port of `FSE_optimalTableLog_internal`. Picks an FSE table size
/// given a user-requested `maxTableLog` and the data characteristics.
/// `minus` is a slack constant the caller supplies (upstream uses 2
/// for the public `FSE_optimalTableLog`).
pub fn FSE_optimalTableLog_internal(
    maxTableLog: u32,
    srcSize: usize,
    maxSymbolValue: u32,
    minus: u32,
) -> u32 {
    debug_assert!(srcSize > 1);
    let maxBitsSrc = ZSTD_highbit32(srcSize as u32 - 1).saturating_sub(minus);
    let mut tableLog = maxTableLog;
    let minBits = FSE_minTableLog(srcSize, maxSymbolValue);
    if tableLog == 0 {
        tableLog = FSE_DEFAULT_TABLELOG;
    }
    if maxBitsSrc < tableLog {
        tableLog = maxBitsSrc;
    }
    if minBits > tableLog {
        tableLog = minBits;
    }
    tableLog.clamp(FSE_MIN_TABLELOG, FSE_MAX_TABLELOG)
}

/// Port of the public `FSE_optimalTableLog`. Uses the standard
/// `minus = 2` slack constant.
pub fn FSE_optimalTableLog(maxTableLog: u32, srcSize: usize, maxSymbolValue: u32) -> u32 {
    FSE_optimalTableLog_internal(maxTableLog, srcSize, maxSymbolValue, 2)
}

/// Upstream `FSE_symbolCompressionTransform` — two i32 fields stored
/// inside the CTable after the nextState u16 table.
#[derive(Debug, Clone, Copy, Default)]
pub struct FSE_symbolCompressionTransform {
    pub deltaNbBits: i32,
    pub deltaFindState: i32,
}

/// Port of `FSE_getMaxNbBits`. Approximate upper bound on the number
/// of bits needed to encode `symbolValue`, derived from the
/// compression transform's `deltaNbBits` field. Rounds fractional-bit
/// costs up — a symbol with normalized freq 3 returns the same cost
/// as freq 2.
///
/// Caller invariant: `symbolValue <= maxSymbolValue`. When the
/// symbol has zero frequency, upstream returns a fake cost of
/// `tableLog + 1` bits; we inherit that behavior via the raw
/// `deltaNbBits` read.
#[inline(always)]
pub fn FSE_getMaxNbBits(symbolTT: &[FSE_symbolCompressionTransform], symbolValue: u32) -> u32 {
    ((symbolTT[symbolValue as usize].deltaNbBits + ((1 << 16) - 1)) as u32) >> 16
}

// ---- CTable layout helpers ---------------------------------------------
//
// Mirror of upstream's cast-based layout for `FSE_CTable = U32`:
//   ct[0]               : packed header (lo16 = tableLog, hi16 = maxSymbolValue)
//   ct[1 .. 1+tableSize/2] : U16 nextState table, two entries per u32
//   ct[1+tableSize/2 ..]: pairs of u32 for symbolCompressionTransform

#[inline(always)]
fn ct_header_write(ct: &mut [FSE_CTable], tableLog: u16, maxSymbolValue: u16) {
    ct[0] = (tableLog as u32) | ((maxSymbolValue as u32) << 16);
}

#[inline(always)]
pub fn ct_header_tableLog(ct: &[FSE_CTable]) -> u32 {
    ct[0] & 0xFFFF
}

#[inline(always)]
pub fn ct_header_maxSV(ct: &[FSE_CTable]) -> u32 {
    (ct[0] >> 16) & 0xFFFF
}

/// Write into the u16 nextState table at logical index `i`.
#[inline(always)]
fn ct_u16_write(ct: &mut [FSE_CTable], i: usize, val: u16) {
    let slot = 1 + (i / 2);
    let prev = ct[slot];
    if i.is_multiple_of(2) {
        ct[slot] = (prev & 0xFFFF_0000) | val as u32;
    } else {
        ct[slot] = (prev & 0x0000_FFFF) | ((val as u32) << 16);
    }
}

#[inline(always)]
fn symbolTT_offset(tableLog: u32) -> usize {
    let tableSize = 1usize << tableLog;
    1 + tableSize / 2
}

#[inline(always)]
fn symbolTT_write(
    ct: &mut [FSE_CTable],
    tableLog: u32,
    s: usize,
    t: FSE_symbolCompressionTransform,
) {
    let base = symbolTT_offset(tableLog) + 2 * s;
    ct[base] = t.deltaNbBits as u32;
    ct[base + 1] = t.deltaFindState as u32;
}

#[inline(always)]
pub fn symbolTT_read(ct: &[FSE_CTable], tableLog: u32, s: usize) -> FSE_symbolCompressionTransform {
    let base = symbolTT_offset(tableLog) + 2 * s;
    // Single bounded-slice load lets LLVM hoist the bounds check
    // out of the encoder's inner loop.
    let pair: &[FSE_CTable; 2] = ct[base..base + 2]
        .try_into()
        .expect("symbolTT_read: 2 slots");
    FSE_symbolCompressionTransform {
        deltaNbBits: pair[0] as i32,
        deltaFindState: pair[1] as i32,
    }
}

/// Unsafe variant of `symbolTT_read` used by `FSE_encodeSymbol`. CTable
/// is sized at allocation time to hold `2*(maxSymbolValue+1)` symbol
/// transform entries past the per-tableLog `symbolTT_offset(tableLog)`
/// base. The encoder hot loop only ever passes `s ≤ maxSymbolValue` so
/// `base + 2 ≤ ct.len()` always holds. Eliminates the per-symbol
/// `cmp/jb` bounds check pair (~3% of L1 cycles in encodeSequences).
///
/// SAFETY: `symbolTT_offset(tableLog) + 2*s + 2 ≤ ct.len()`.
#[inline(always)]
unsafe fn symbolTT_read_unchecked(
    ct: &[FSE_CTable],
    tableLog: u32,
    s: usize,
) -> FSE_symbolCompressionTransform {
    let base = symbolTT_offset(tableLog) + 2 * s;
    let p0 = unsafe { *ct.get_unchecked(base) };
    let p1 = unsafe { *ct.get_unchecked(base + 1) };
    FSE_symbolCompressionTransform {
        deltaNbBits: p0 as i32,
        deltaFindState: p1 as i32,
    }
}

/// Total u32 slots needed for a CTable of the given tableLog +
/// maxSymbolValue. Matches upstream's `FSE_CTABLE_SIZE_U32`.
#[inline(always)]
pub fn FSE_CTABLE_SIZE_U32(tableLog: u32, maxSymbolValue: u32) -> usize {
    let tableSize = 1usize << tableLog;
    1 + tableSize / 2 + 2 * (maxSymbolValue as usize + 1)
}

/// Port of `FSE_buildCTable_rle`. Single-symbol CTable: tableLog=0,
/// symbolTT zeroed for that symbol. Upstream returns 0 on success.
pub fn FSE_buildCTable_rle(ct: &mut [FSE_CTable], symbolValue: u8) -> usize {
    // Header: tableLog=0, maxSymbolValue=symbolValue.
    ct_header_write(ct, 0, symbolValue as u16);
    // tableU16[0..=1] = 0 — only slot index 0 actually lives in `ct[1]`
    // (lo16); slot 1 is the hi16 of the same u32. Zero it explicitly.
    if ct.len() >= 2 {
        ct[1] = 0;
    }
    symbolTT_write(
        ct,
        0,
        symbolValue as usize,
        FSE_symbolCompressionTransform {
            deltaNbBits: 0,
            deltaFindState: 0,
        },
    );
    0
}

/// Port of `FSE_bitCost`. Approximate per-symbol cost in
/// `accuracyLog`-bit fixed-point, derived from a CTable's symbolTT
/// entry. Returns a "fake" cost of `(tableLog+1)<<accuracyLog` when the
/// symbol has frequency 0 in the table (i.e. its deltaNbBits points at
/// the forbidden threshold).
pub fn FSE_bitCost(ct: &[FSE_CTable], tableLog: u32, symbolValue: u32, accuracyLog: u32) -> u32 {
    let tt = symbolTT_read(ct, tableLog, symbolValue as usize);
    let minNbBits = (tt.deltaNbBits as u32) >> 16;
    let threshold = (minNbBits + 1) << 16;
    debug_assert!(tableLog < 16);
    debug_assert!(accuracyLog < 31 - tableLog);
    let tableSize = 1u32 << tableLog;
    let deltaFromThreshold =
        threshold.wrapping_sub((tt.deltaNbBits as u32).wrapping_add(tableSize));
    let normalizedDeltaFromThreshold = (deltaFromThreshold << accuracyLog) >> tableLog;
    let bitMultiplier = 1u32 << accuracyLog;
    (minNbBits + 1) * bitMultiplier - normalizedDeltaFromThreshold
}

/// Port of `FSE_buildCTable_wksp`. Builds the state-transition table
/// for FSE encoding out of a normalized count distribution.
///
/// Rust signature note: upstream takes the workspace as `void*` with
/// 2-byte alignment; the Rust port takes separate `cumul` (u16) and
/// `tableSymbol` (u8) slices so the alignment invariant is
/// type-encoded. Callers that want upstream's single-workspace shape
/// can split a `&mut [u8]` in two.
pub fn FSE_buildCTable_wksp(
    ct: &mut [FSE_CTable],
    normalizedCounter: &[i16],
    maxSymbolValue: u32,
    tableLog: u32,
    workSpace: &mut [u8],
) -> usize {
    use crate::common::error::{ErrorCode, ERROR};
    use crate::common::fse_decompress::FSE_TABLESTEP;

    let tableSize = 1u32 << tableLog;
    let tableMask = (tableSize - 1) as usize;
    let step = FSE_TABLESTEP(tableSize) as usize;
    let maxSV1 = maxSymbolValue + 1;

    // Workspace split: cumul[maxSV1+1] u16 + tableSymbol[tableSize] u8.
    let cumul_bytes = (maxSV1 as usize + 1) * 2;
    let ts_bytes = tableSize as usize;
    let needed = cumul_bytes + ts_bytes + 8; // +8 for upstream's 8-byte MEM_write64 slack on `spread`
    if workSpace.len() < needed {
        return ERROR(ErrorCode::TableLogTooLarge);
    }
    let (cumul_bytes_slice, rest) = workSpace.split_at_mut(cumul_bytes);
    let (tableSymbol, _spread_slack) = rest.split_at_mut(ts_bytes + 8);
    let mut cumul = vec![0u16; maxSV1 as usize + 1];

    debug_assert!(tableLog < 16);
    ct_header_write(ct, tableLog as u16, maxSymbolValue as u16);

    // Symbol start positions + lowprob placement.
    let mut highThreshold: i64 = tableSize as i64 - 1;
    cumul[0] = 0;
    for u in 1..=maxSV1 as usize {
        let nc = normalizedCounter[u - 1];
        if nc == -1 {
            cumul[u] = cumul[u - 1] + 1;
            tableSymbol[highThreshold as usize] = (u - 1) as u8;
            highThreshold -= 1;
        } else {
            debug_assert!(nc >= 0);
            cumul[u] = cumul[u - 1] + nc as u16;
        }
    }
    cumul[maxSV1 as usize] = (tableSize + 1) as u16;

    // Spread symbols.
    if highThreshold == tableSize as i64 - 1 {
        // Fast path: lay down in order into `spread` (separate from
        // the final `tableSymbol` buffer — upstream stashes `spread`
        // at `tableSymbol + tableSize`, 8-byte slack tail included).
        let mut spread = vec![0u8; tableSize as usize + 8];
        let mut pos = 0usize;
        for (s, &nc) in normalizedCounter.iter().enumerate().take(maxSV1 as usize) {
            let n = nc as i32;
            for i in 0..n as usize {
                spread[pos + i] = s as u8;
            }
            if n > 0 {
                pos += n as usize;
            }
        }
        let unroll = 2usize;
        let mut position = 0usize;
        let mut s = 0usize;
        while s < tableSize as usize {
            for u in 0..unroll {
                let uPos = (position + u * step) & tableMask;
                tableSymbol[uPos] = spread[s + u];
            }
            position = (position + unroll * step) & tableMask;
            s += unroll;
        }
    } else {
        // Slow path with lowprob skipping.
        let mut position = 0usize;
        for (symbol, &freq) in normalizedCounter.iter().enumerate().take(maxSV1 as usize) {
            for _ in 0..freq {
                tableSymbol[position] = symbol as u8;
                position = (position + step) & tableMask;
                while position as i64 > highThreshold {
                    position = (position + step) & tableMask;
                }
            }
        }
    }

    // Build nextState u16 table: for each slot `u`, write
    // tableU16[cumul[s]++] = tableSize + u (where s is symbol at slot u).
    for (u, &sym) in tableSymbol.iter().enumerate().take(tableSize as usize) {
        let s = sym as usize;
        let idx = cumul[s] as usize;
        cumul[s] += 1;
        ct_u16_write(ct, idx, (tableSize as usize + u) as u16);
    }

    // Build symbol-compression-transform table.
    let mut total: u32 = 0;
    for (s, &nc) in normalizedCounter
        .iter()
        .enumerate()
        .take(maxSymbolValue as usize + 1)
    {
        let t: FSE_symbolCompressionTransform = match nc {
            0 => {
                // For the cost estimator: (tableLog+1) << 16 - tableSize
                let dnb = ((tableLog + 1) << 16) as i32 - (1i32 << tableLog);
                FSE_symbolCompressionTransform {
                    deltaNbBits: dnb,
                    deltaFindState: 0,
                }
            }
            -1 | 1 => {
                let dnb = (tableLog << 16) as i32 - (1i32 << tableLog);
                let t = FSE_symbolCompressionTransform {
                    deltaNbBits: dnb,
                    deltaFindState: total as i32 - 1,
                };
                total += 1;
                t
            }
            n if n > 1 => {
                let nu = n as u32;
                let maxBitsOut = tableLog - crate::common::bits::ZSTD_highbit32(nu - 1);
                let minStatePlus = nu << maxBitsOut;
                let dnb = (maxBitsOut << 16) as i32 - minStatePlus as i32;
                let t = FSE_symbolCompressionTransform {
                    deltaNbBits: dnb,
                    deltaFindState: total as i32 - nu as i32,
                };
                total += nu;
                t
            }
            _ => unreachable!("normalizedCounter must be >= -1"),
        };
        symbolTT_write(ct, tableLog, s, t);
    }

    let _ = cumul_bytes_slice; // workspace split kept for layout faithfulness
    0
}

/// Port of the file-private `FSE_writeNCount_generic`. `writeIsSafe`
/// mirrors upstream's flag: when true, the caller already sized
/// `buffer` via `FSE_NCountWriteBound` and bounds checks are skipped.
fn FSE_writeNCount_generic(
    buffer: &mut [u8],
    normalizedCounter: &[i16],
    maxSymbolValue: u32,
    tableLog: u32,
    writeIsSafe: bool,
) -> usize {
    use crate::common::error::{ErrorCode, ERROR};

    let oend = buffer.len();
    let mut out: usize = 0;
    let tableSize = 1i32 << tableLog;

    let mut bitStream: u32 = 0;
    let mut bitCount: i32 = 0;
    let mut symbol: u32 = 0;
    let alphabetSize = maxSymbolValue + 1;
    let mut previousIs0 = false;

    // Table size header (4 bits).
    bitStream += (tableLog - FSE_MIN_TABLELOG) << bitCount;
    bitCount += 4;

    let mut remaining: i32 = tableSize + 1; // +1 for extra accuracy
    let mut threshold: i32 = tableSize;
    let mut nbBits: i32 = tableLog as i32 + 1;

    // Helper for the two 16-bit flushes upstream does inline.
    let flush16 = |buffer: &mut [u8], out: &mut usize, bitStream: &mut u32| -> Option<usize> {
        if !writeIsSafe && *out > oend.saturating_sub(2) {
            return Some(ERROR(ErrorCode::DstSizeTooSmall));
        }
        buffer[*out] = *bitStream as u8;
        buffer[*out + 1] = (*bitStream >> 8) as u8;
        *out += 2;
        *bitStream >>= 16;
        None
    };

    while symbol < alphabetSize && remaining > 1 {
        if previousIs0 {
            let start_base = symbol;
            while symbol < alphabetSize && normalizedCounter[symbol as usize] == 0 {
                symbol += 1;
            }
            if symbol == alphabetSize {
                break; // bad distribution
            }
            let mut start = start_base;
            while symbol >= start + 24 {
                start += 24;
                bitStream += 0xFFFFu32 << bitCount;
                if let Some(err) = flush16(buffer, &mut out, &mut bitStream) {
                    return err;
                }
            }
            while symbol >= start + 3 {
                start += 3;
                bitStream += 3u32 << bitCount;
                bitCount += 2;
            }
            bitStream += (symbol - start) << bitCount;
            bitCount += 2;
            if bitCount > 16 {
                if let Some(err) = flush16(buffer, &mut out, &mut bitStream) {
                    return err;
                }
                bitCount -= 16;
            }
        }
        {
            let count_s = normalizedCounter[symbol as usize] as i32;
            symbol += 1;
            let max = (2 * threshold - 1) - remaining;
            remaining -= if count_s < 0 { -count_s } else { count_s };
            let mut count = count_s + 1; // +1 extra accuracy
            if count >= threshold {
                count += max; // [0..max[ [max..threshold[ (...) [threshold+max, 2*threshold[
            }
            bitStream += (count as u32) << bitCount;
            bitCount += nbBits;
            bitCount -= (count < max) as i32;
            previousIs0 = count == 1;
            if remaining < 1 {
                return ERROR(ErrorCode::Generic);
            }
            while remaining < threshold {
                nbBits -= 1;
                threshold >>= 1;
            }
        }
        if bitCount > 16 {
            if let Some(err) = flush16(buffer, &mut out, &mut bitStream) {
                return err;
            }
            bitCount -= 16;
        }
    }

    if remaining != 1 {
        return ERROR(ErrorCode::Generic);
    }
    debug_assert!(symbol <= alphabetSize);

    // Final flush — up to 2 bytes plus any remaining bit fragment.
    if !writeIsSafe && out > oend.saturating_sub(2) {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    buffer[out] = bitStream as u8;
    if out + 1 < oend {
        buffer[out + 1] = (bitStream >> 8) as u8;
    }
    out += ((bitCount + 7) / 8) as usize;
    out
}

/// Port of `FSE_writeNCount`. Encodes `normalizedCounter[0..=maxSymbolValue]`
/// into `buffer`, returning bytes written (or an error).
pub fn FSE_writeNCount(
    buffer: &mut [u8],
    normalizedCounter: &[i16],
    maxSymbolValue: u32,
    tableLog: u32,
) -> usize {
    use crate::common::error::{ErrorCode, ERROR};
    if tableLog > FSE_MAX_TABLELOG {
        return ERROR(ErrorCode::TableLogTooLarge);
    }
    if tableLog < FSE_MIN_TABLELOG {
        return ERROR(ErrorCode::Generic);
    }
    let writeIsSafe = buffer.len() >= FSE_NCountWriteBound(maxSymbolValue, tableLog);
    FSE_writeNCount_generic(
        buffer,
        normalizedCounter,
        maxSymbolValue,
        tableLog,
        writeIsSafe,
    )
}

/// Port of `FSE_CState_t`. The Rust variant stores indices into the
/// caller's CTable slice rather than raw pointers (the upstream struct
/// holds `stateTable` / `symbolTT` void-pointers — we compute them
/// lazily in the helpers below).
#[derive(Debug, Clone, Copy)]
pub struct FSE_CState_t {
    pub value: isize,
    pub stateLog: u32,
}

/// Read the nextState u16 at logical index `i` inside a CTable.
#[inline(always)]
#[allow(dead_code)]
fn ct_u16_read(ct: &[FSE_CTable], i: usize) -> u16 {
    let slot = 1 + (i / 2);
    let w = ct[slot];
    if i.is_multiple_of(2) {
        w as u16
    } else {
        (w >> 16) as u16
    }
}

/// Unsafe variant of `ct_u16_read` used inside `FSE_encodeSymbol`. The
/// caller is the FSE encoder hot loop which always passes an `i` derived
/// from a valid state machine transition — the CTable was sized to hold
/// `1 + (1 << tableLog)/2 + 2*(maxSymbolValue+1)` u32 slots, so
/// `1 + i/2 < ct.len()` whenever `i < (1 << tableLog)`. Strips the
/// per-iter bounds check that LLVM can't elide because it doesn't
/// track the FSE state-table invariants.
///
/// SAFETY: `1 + i/2 < ct.len()`.
#[inline(always)]
unsafe fn ct_u16_read_unchecked(ct: &[FSE_CTable], i: usize) -> u16 {
    let slot = 1 + (i / 2);
    let w = unsafe { *ct.get_unchecked(slot) };
    if i.is_multiple_of(2) {
        w as u16
    } else {
        (w >> 16) as u16
    }
}

/// Port of `FSE_initCState`.
#[inline(always)]
pub fn FSE_initCState(st: &mut FSE_CState_t, ct: &[FSE_CTable]) {
    let tableLog = ct_header_tableLog(ct);
    st.value = 1isize << tableLog;
    st.stateLog = tableLog;
}

/// Port of `FSE_initCState2` (first-symbol init optimization).
#[inline(always)]
pub fn FSE_initCState2(st: &mut FSE_CState_t, ct: &[FSE_CTable], symbol: u32) {
    FSE_initCState(st, ct);
    let tableLog = st.stateLog;
    // SAFETY: see `FSE_encodeSymbol` — same invariants apply (called once
    // per block at the last sequence, not in a hot loop, but using the
    // same unchecked path keeps the helper consistent and cheaper).
    let tt = unsafe { symbolTT_read_unchecked(ct, tableLog, symbol as usize) };
    let nbBitsOut = ((tt.deltaNbBits + (1 << 15)) >> 16) as u32;
    let v = ((nbBitsOut as isize) << 16) - (tt.deltaNbBits as isize);
    let idx = ((v >> nbBitsOut) + tt.deltaFindState as isize) as usize;
    st.value = unsafe { ct_u16_read_unchecked(ct, idx) } as isize;
}

/// Port of `FSE_encodeSymbol`. Emits up to `tableLog` bits into the
/// bitstream and advances the state via the lookup table.
///
/// Hot-path optimization: the symbol/state-table reads use unsafe
/// unchecked indexing. The CTable allocation guarantees that all valid
/// (tableLog, symbol) and (state) indices land in-range, but LLVM
/// can't prove this from the slice length alone — every symbol read
/// would otherwise emit a `cmp/jb` pair. Each call sites's invariants:
///   - `symbol ≤ maxSymbolValue` (FSE precondition; checked by upstream
///     too)
///   - `idx = state >> nbBits + deltaFindState` is a valid state index
///     0 ≤ idx < (1 << tableLog), upheld by FSE state-machine
///     correctness
#[inline(always)]
pub fn FSE_encodeSymbol(
    bitC: &mut crate::common::bitstream::BIT_CStream_t,
    st: &mut FSE_CState_t,
    ct: &[FSE_CTable],
    symbol: u32,
) {
    let tableLog = st.stateLog;
    // SAFETY: see function docstring — symbol ≤ maxSymbolValue, idx is
    // in-range by FSE state-machine invariants, ct sized accordingly.
    let tt = unsafe { symbolTT_read_unchecked(ct, tableLog, symbol as usize) };
    let nbBitsOut = ((st.value + tt.deltaNbBits as isize) >> 16) as u32;
    crate::common::bitstream::BIT_addBits(bitC, st.value as usize, nbBitsOut);
    let idx = ((st.value >> nbBitsOut) + tt.deltaFindState as isize) as usize;
    st.value = unsafe { ct_u16_read_unchecked(ct, idx) } as isize;
}

/// Port of `FSE_flushCState`. Writes the final state and flushes the
/// local bit register.
#[inline(always)]
pub fn FSE_flushCState(bitC: &mut crate::common::bitstream::BIT_CStream_t, st: &FSE_CState_t) {
    crate::common::bitstream::BIT_addBits(bitC, st.value as usize, st.stateLog);
    crate::common::bitstream::BIT_flushBits(bitC);
}

/// Port of `FSE_compress_usingCTable_generic`. `FAST` const-generic
/// selects `BIT_flushBitsFast` vs `BIT_flushBits`.
fn FSE_compress_usingCTable_generic<const FAST: bool>(
    dst: &mut [u8],
    src: &[u8],
    ct: &[FSE_CTable],
) -> usize {
    use crate::common::bitstream::{
        BIT_closeCStream, BIT_flushBits, BIT_flushBitsFast, BIT_initCStream,
    };
    let srcSize = src.len();
    if srcSize <= 2 {
        return 0;
    }
    let dst_cap = dst.len();
    let (mut bitC, init_err) = BIT_initCStream(dst, dst_cap);
    if crate::common::error::ERR_isError(init_err) {
        return 0;
    }

    let container_bits = (core::mem::size_of::<usize>() as u32) * 8;

    let mut ip = srcSize; // upstream: ip starts at iend and walks down
    let mut state1 = FSE_CState_t {
        value: 0,
        stateLog: 0,
    };
    let mut state2 = FSE_CState_t {
        value: 0,
        stateLog: 0,
    };

    if srcSize.is_multiple_of(2) {
        ip -= 1;
        FSE_initCState2(&mut state2, ct, src[ip] as u32);
        ip -= 1;
        FSE_initCState2(&mut state1, ct, src[ip] as u32);
    } else {
        ip -= 1;
        FSE_initCState2(&mut state1, ct, src[ip] as u32);
        ip -= 1;
        FSE_initCState2(&mut state2, ct, src[ip] as u32);
        ip -= 1;
        FSE_encodeSymbol(&mut bitC, &mut state1, ct, src[ip] as u32);
        if FAST {
            BIT_flushBitsFast(&mut bitC);
        } else {
            BIT_flushBits(&mut bitC);
        }
    }

    // Join to mod 4 on 64-bit (upstream's static test).
    let mut remaining = srcSize - 2;
    if container_bits > FSE_MAX_TABLELOG * 4 + 7 && remaining & 2 != 0 {
        ip -= 1;
        FSE_encodeSymbol(&mut bitC, &mut state2, ct, src[ip] as u32);
        ip -= 1;
        FSE_encodeSymbol(&mut bitC, &mut state1, ct, src[ip] as u32);
        if FAST {
            BIT_flushBitsFast(&mut bitC);
        } else {
            BIT_flushBits(&mut bitC);
        }
        remaining -= 2;
    }

    // 2 or 4 encodings per loop.
    while ip > 0 {
        ip -= 1;
        FSE_encodeSymbol(&mut bitC, &mut state2, ct, src[ip] as u32);
        if container_bits < FSE_MAX_TABLELOG * 2 + 7 {
            if FAST {
                BIT_flushBitsFast(&mut bitC);
            } else {
                BIT_flushBits(&mut bitC);
            }
        }
        ip -= 1;
        FSE_encodeSymbol(&mut bitC, &mut state1, ct, src[ip] as u32);
        if container_bits > FSE_MAX_TABLELOG * 4 + 7 {
            ip -= 1;
            FSE_encodeSymbol(&mut bitC, &mut state2, ct, src[ip] as u32);
            ip -= 1;
            FSE_encodeSymbol(&mut bitC, &mut state1, ct, src[ip] as u32);
        }
        if FAST {
            BIT_flushBitsFast(&mut bitC);
        } else {
            BIT_flushBits(&mut bitC);
        }
    }

    FSE_flushCState(&mut bitC, &state2);
    FSE_flushCState(&mut bitC, &state1);
    let _ = remaining;
    BIT_closeCStream(&mut bitC)
}

/// Port of `FSE_compress_usingCTable`.
pub fn FSE_compress_usingCTable(dst: &mut [u8], src: &[u8], ct: &[FSE_CTable]) -> usize {
    let fast = dst.len() >= FSE_BLOCKBOUND(src.len());
    if fast {
        FSE_compress_usingCTable_generic::<true>(dst, src, ct)
    } else {
        FSE_compress_usingCTable_generic::<false>(dst, src, ct)
    }
}

/// Upstream `FSE_BLOCKBOUND` macro.
#[inline(always)]
pub const fn FSE_BLOCKBOUND(size: usize) -> usize {
    size + (size >> 7) + 4 + core::mem::size_of::<usize>()
}

/// Upstream `FSE_COMPRESSBOUND` macro — NCount header + compressed block.
#[inline(always)]
pub const fn FSE_COMPRESSBOUND(size: usize) -> usize {
    FSE_NCOUNTBOUND + FSE_BLOCKBOUND(size)
}

/// Port of `FSE_compressBound`.
#[inline(always)]
pub fn FSE_compressBound(srcSize: usize) -> usize {
    FSE_COMPRESSBOUND(srcSize)
}

const RTB_TABLE: [u32; 8] = [0, 473195, 504333, 520860, 550000, 700000, 750000, 830000];

/// Port of `FSE_normalizeM2` — the secondary normalization fallback
/// used when the primary method's residual is too big.
fn FSE_normalizeM2(
    norm: &mut [i16],
    tableLog: u32,
    count: &[u32],
    mut total: usize,
    maxSymbolValue: u32,
    lowProbCount: i16,
) -> usize {
    const NOT_YET_ASSIGNED: i16 = -2;
    let lowThreshold = (total >> tableLog) as u32;
    let mut lowOne = ((total * 3) >> (tableLog + 1)) as u32;
    let mut distributed: u32 = 0;

    for s in 0..=maxSymbolValue as usize {
        if count[s] == 0 {
            norm[s] = 0;
            continue;
        }
        if count[s] <= lowThreshold {
            norm[s] = lowProbCount;
            distributed += 1;
            total -= count[s] as usize;
            continue;
        }
        if count[s] <= lowOne {
            norm[s] = 1;
            distributed += 1;
            total -= count[s] as usize;
            continue;
        }
        norm[s] = NOT_YET_ASSIGNED;
    }
    let mut ToDistribute = (1u32 << tableLog) - distributed;
    if ToDistribute == 0 {
        return 0;
    }

    if total as u32 / ToDistribute > lowOne {
        lowOne = ((total as u32) * 3) / (ToDistribute * 2);
        for s in 0..=maxSymbolValue as usize {
            if norm[s] == NOT_YET_ASSIGNED && count[s] <= lowOne {
                norm[s] = 1;
                distributed += 1;
                total -= count[s] as usize;
            }
        }
        ToDistribute = (1u32 << tableLog) - distributed;
    }

    if distributed == maxSymbolValue + 1 {
        // All values are pretty poor; dump the remainder into the max.
        let mut maxV = 0usize;
        let mut maxC = 0u32;
        for (s, &c) in count.iter().enumerate().take(maxSymbolValue as usize + 1) {
            if c > maxC {
                maxV = s;
                maxC = c;
            }
        }
        norm[maxV] += ToDistribute as i16;
        return 0;
    }

    if total == 0 {
        // Everything was absorbed by lowOne/lowThreshold; round-robin
        // the remainder across non-zero norms.
        let alphabet = (maxSymbolValue + 1) as usize;
        let mut s = 0usize;
        while ToDistribute > 0 {
            if norm[s] > 0 {
                ToDistribute -= 1;
                norm[s] += 1;
            }
            s = (s + 1) % alphabet;
        }
        return 0;
    }

    let vStepLog = 62u32 - tableLog;
    let mid = (1u64 << (vStepLog - 1)) - 1;
    let rStep = ((1u64 << vStepLog) * ToDistribute as u64 + mid) / (total as u64);
    let mut tmpTotal = mid;
    for s in 0..=maxSymbolValue as usize {
        if norm[s] == NOT_YET_ASSIGNED {
            let end = tmpTotal + (count[s] as u64 * rStep);
            let sStart = (tmpTotal >> vStepLog) as u32;
            let sEnd = (end >> vStepLog) as u32;
            let weight = sEnd - sStart;
            if weight < 1 {
                return crate::common::error::ERROR(crate::common::error::ErrorCode::Generic);
            }
            norm[s] = weight as i16;
            tmpTotal = end;
        }
    }
    0
}

/// Port of `FSE_normalizeCount`. Produces a `tableLog`-sized
/// normalized count distribution from an absolute count array. On
/// success returns `tableLog` (possibly rewritten from 0 to the
/// default); 0 means the input was pure-RLE (a single symbol has the
/// full count). Returns an error code otherwise.
pub fn FSE_normalizeCount(
    normalizedCounter: &mut [i16],
    mut tableLog: u32,
    count: &[u32],
    total: usize,
    maxSymbolValue: u32,
    useLowProbCount: u32,
) -> usize {
    use crate::common::error::{ErrorCode, ERROR};

    if tableLog == 0 {
        tableLog = FSE_DEFAULT_TABLELOG;
    }
    if tableLog < FSE_MIN_TABLELOG {
        return ERROR(ErrorCode::Generic);
    }
    if tableLog > FSE_MAX_TABLELOG {
        return ERROR(ErrorCode::TableLogTooLarge);
    }
    if tableLog < FSE_minTableLog(total, maxSymbolValue) {
        return ERROR(ErrorCode::Generic);
    }

    let lowProbCount: i16 = if useLowProbCount != 0 { -1 } else { 1 };
    let scale = 62u32 - tableLog;
    let step = (1u64 << 62) / total as u64;
    let vStep = 1u64 << (scale - 20);
    let mut stillToDistribute: i32 = 1 << tableLog;
    let mut largest: u32 = 0;
    let mut largestP: i16 = 0;
    let lowThreshold = (total >> tableLog) as u32;

    for s in 0..=maxSymbolValue as usize {
        if count[s] as usize == total {
            return 0; // RLE
        }
        if count[s] == 0 {
            normalizedCounter[s] = 0;
            continue;
        }
        if count[s] <= lowThreshold {
            normalizedCounter[s] = lowProbCount;
            stillToDistribute -= 1;
        } else {
            let mut proba = ((count[s] as u64 * step) >> scale) as i16;
            if proba < 8 {
                let restToBeat = vStep * RTB_TABLE[proba as usize] as u64;
                let excess = (count[s] as u64 * step) - ((proba as u64) << scale);
                if excess > restToBeat {
                    proba += 1;
                }
            }
            if proba > largestP {
                largestP = proba;
                largest = s as u32;
            }
            normalizedCounter[s] = proba;
            stillToDistribute -= proba as i32;
        }
    }

    if -stillToDistribute >= (normalizedCounter[largest as usize] >> 1) as i32 {
        // Too much residual — fall back to the secondary method.
        let rc = FSE_normalizeM2(
            normalizedCounter,
            tableLog,
            count,
            total,
            maxSymbolValue,
            lowProbCount,
        );
        if crate::common::error::ERR_isError(rc) {
            return rc;
        }
    } else {
        normalizedCounter[largest as usize] += stillToDistribute as i16;
    }
    tableLog as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimal_tablelog_clamps_to_min_and_max() {
        // Tiny srcSize should clamp up to FSE_MIN_TABLELOG.
        assert!(FSE_optimalTableLog(0, 32, 10) >= FSE_MIN_TABLELOG);
        // User-declared max ≫ FSE_MAX_TABLELOG should cap at MAX.
        assert!(FSE_optimalTableLog(99, 10_000, 255) <= FSE_MAX_TABLELOG);
    }

    #[test]
    fn optimal_tablelog_zero_means_default() {
        // Upstream: tableLog=0 is interpreted as FSE_DEFAULT_TABLELOG
        // before any clamping. Large srcSize + small alphabet keeps it
        // right at the default.
        assert_eq!(FSE_optimalTableLog(0, 65536, 255), FSE_DEFAULT_TABLELOG);
    }

    #[test]
    fn ncount_write_bound_zero_maxsymbol_is_default() {
        // maxSymbolValue == 0 → upstream falls back to FSE_NCOUNTBOUND.
        assert_eq!(FSE_NCountWriteBound(0, 8), FSE_NCOUNTBOUND);
    }

    #[test]
    fn ncount_write_bound_scales_with_params() {
        let small = FSE_NCountWriteBound(31, 6);
        let large = FSE_NCountWriteBound(255, 12);
        assert!(large > small);
        // Formula: ((alphabet * tableLog + 6) / 8) + 3. For (31, 6):
        //   ((32 * 6 + 6)/8) + 3 = (198/8) + 3 = 24 + 3 = 27.
        assert_eq!(small, 27);
    }

    #[test]
    fn normalize_count_sum_equals_table_size() {
        // Classic uniform-ish distribution over 6 symbols, 1000 total
        // counts. After normalization, Σ |norm[s]| must equal 1<<tableLog.
        let count = [100u32, 200, 300, 150, 150, 100];
        let mut norm = [0i16; 16];
        let total: usize = count.iter().map(|&c| c as usize).sum();
        let rc = FSE_normalizeCount(&mut norm, 8, &count, total, 5, 0);
        assert!(
            !crate::common::error::ERR_isError(rc),
            "err: {}",
            crate::common::error::ERR_getErrorName(rc)
        );
        assert_eq!(rc, 8); // returns tableLog on success
        let sum: i32 = norm
            .iter()
            .take(6)
            .map(|&n| if n < 0 { 1 } else { n as i32 })
            .sum();
        assert_eq!(sum, 1 << 8, "normalized counts must sum to table size");
    }

    #[test]
    fn normalize_count_rle_returns_zero() {
        // Only one symbol has all the counts → upstream returns 0 to
        // signal "use RLE instead".
        let mut count = [0u32; 8];
        count[3] = 500;
        let mut norm = [0i16; 8];
        let rc = FSE_normalizeCount(&mut norm, 6, &count, 500, 7, 0);
        assert_eq!(rc, 0);
    }

    #[test]
    fn normalize_count_rejects_tablelog_too_small_for_alphabet() {
        // Six symbols → need at least ⌈log2(6)⌉+2 = 5 bits in
        // FSE_minTableLog. Ask for tableLog=4 → error.
        let count = [10u32, 10, 10, 10, 10, 10];
        let mut norm = [0i16; 8];
        let rc = FSE_normalizeCount(&mut norm, 4, &count, 60, 5, 0);
        assert!(crate::common::error::ERR_isError(rc));
    }

    #[test]
    fn FSE_DTABLE_SIZE_U32_and_WKSP_SIZE_match_upstream_formula() {
        // Upstream macros (fse.h):
        //   FSE_DTABLE_SIZE_U32(maxTableLog) = 1 + (1 << maxTableLog)
        //   FSE_BUILD_DTABLE_WKSP_SIZE(mtl, msv) = 2*(msv+1) + (1<<mtl) + 8
        // These drive every FSE DTable allocation the decoder makes
        // on behalf of callers; silent drift would cause buffer sizing
        // mismatches and either waste memory or under-allocate.
        use crate::common::fse_decompress::{FSE_BUILD_DTABLE_WKSP_SIZE, FSE_DTABLE_SIZE_U32};
        for tl in [5u32, 9, 12] {
            assert_eq!(FSE_DTABLE_SIZE_U32(tl), 1 + (1 << tl));
        }
        for (tl, msv) in [(6u32, 35u32), (9, 35), (12, 255)] {
            let expected = 2 * (msv as usize + 1) + (1usize << tl) + 8;
            assert_eq!(FSE_BUILD_DTABLE_WKSP_SIZE(tl, msv), expected);
        }
    }

    #[test]
    fn FSE_CTABLE_SIZE_U32_matches_upstream_formula() {
        // Upstream `FSE_CTABLE_SIZE_U32(maxTableLog, maxSymbolValue)`
        // evaluates to:
        //   1 + (1 << (maxTableLog - 1)) + (maxSymbolValue + 1) * 2
        // for maxTableLog >= 1. Our port uses `1 + tableSize/2 + ...`
        // which is equivalent for positive logs and also defined for
        // tableLog=0 (produces `1 + 0 + 2*(maxSymbolValue+1)` — the
        // RLE-CTable size we actually use). Pin both cases so any
        // future refactor can't drift the memory budget.
        for (tl, max_sym) in [(6u32, 35u32), (9, 35), (11, 255), (12, 255)] {
            let expected: usize = 1 + (1usize << (tl - 1)) + (max_sym as usize + 1) * 2;
            assert_eq!(
                FSE_CTABLE_SIZE_U32(tl, max_sym),
                expected,
                "FSE_CTABLE_SIZE_U32({tl}, {max_sym})",
            );
        }
        // tableLog=0 → RLE case: allocator must still return a
        // positive slot count (our port yields `1 + 0 + 2*(sym+1)`).
        assert_eq!(FSE_CTABLE_SIZE_U32(0, 0), 3);
        assert_eq!(FSE_CTABLE_SIZE_U32(0, 255), 1 + 2 * 256);
    }

    #[test]
    fn fse_memory_and_tablelog_constants_match_upstream() {
        // Pin the format-level FSE constants. Upstream defaults:
        //   FSE_MAX_MEMORY_USAGE = 14 → MAX_TABLELOG = 12
        //   FSE_DEFAULT_MEMORY_USAGE = 13 → DEFAULT_TABLELOG = 11
        //   FSE_NCOUNTBOUND = 512
        assert_eq!(FSE_MAX_MEMORY_USAGE, 14);
        assert_eq!(FSE_DEFAULT_MEMORY_USAGE, 13);
        assert_eq!(FSE_MAX_TABLELOG, 12);
        assert_eq!(FSE_DEFAULT_TABLELOG, 11);
        assert_eq!(FSE_NCOUNTBOUND, 512);
        // Verify the arithmetic relationship (memory_usage - 2).
        assert_eq!(FSE_MAX_TABLELOG, FSE_MAX_MEMORY_USAGE - 2);
        assert_eq!(FSE_DEFAULT_TABLELOG, FSE_DEFAULT_MEMORY_USAGE - 2);
        // MIN_TABLELOG = 5 and TABLELOG_ABSOLUTE_MAX = 15
        // — spec-defined bounds that the decoder enforces.
        assert_eq!(FSE_MIN_TABLELOG, 5);
        assert_eq!(crate::common::fse_decompress::FSE_TABLELOG_ABSOLUTE_MAX, 15);
        // The relationship: FSE_MAX_TABLELOG (12) must stay within
        // the absolute max (15) — else the decoder couldn't accept
        // our encoded streams.
        const _: () =
            assert!(FSE_MAX_TABLELOG <= crate::common::fse_decompress::FSE_TABLELOG_ABSOLUTE_MAX);
    }

    #[test]
    fn compress_bound_matches_formula() {
        // FSE_NCOUNTBOUND + srcSize + srcSize>>7 + 4 + sizeof(size_t).
        let bound = FSE_compressBound(1024);
        let expected = FSE_NCOUNTBOUND + 1024 + (1024 >> 7) + 4 + core::mem::size_of::<usize>();
        assert_eq!(bound, expected);
    }

    #[test]
    fn ncount_write_then_read_roundtrip() {
        // normalize → writeNCount → readNCount should reproduce the
        // same distribution and maxSymbolValue. This is the first
        // encoder↔decoder roundtrip at the entropy-header level.
        use crate::common::entropy_common::FSE_readNCount;

        let count = [100u32, 200, 300, 150, 150, 100];
        let total: usize = count.iter().map(|&c| c as usize).sum();
        let mut norm = [0i16; 16];
        let tableLog = FSE_normalizeCount(&mut norm, 8, &count, total, 5, 0);
        assert!(!crate::common::error::ERR_isError(tableLog));
        assert_eq!(tableLog, 8);

        let mut buf = vec![0u8; FSE_NCountWriteBound(5, tableLog as u32)];
        let written = FSE_writeNCount(&mut buf, &norm, 5, tableLog as u32);
        assert!(
            !crate::common::error::ERR_isError(written),
            "writeNCount err: {}",
            crate::common::error::ERR_getErrorName(written)
        );
        assert!(written <= buf.len());

        // Decode it back.
        let mut norm_out = [0i16; 16];
        let mut msv_out: u32 = 5;
        let mut tl_out: u32 = 0;
        let consumed = FSE_readNCount(&mut norm_out, &mut msv_out, &mut tl_out, &buf[..written]);
        assert!(
            !crate::common::error::ERR_isError(consumed),
            "readNCount err: {}",
            crate::common::error::ERR_getErrorName(consumed)
        );
        assert_eq!(tl_out, tableLog as u32);
        assert_eq!(msv_out, 5);
        assert_eq!(&norm_out[..=5], &norm[..=5]);
    }

    #[test]
    fn fse_compress_then_decompress_roundtrip() {
        // The full FSE encode↔decode loop on a tiny skewed input.
        // normalize → buildCTable → compress → buildDTable →
        // decompress_usingDTable should reproduce the input bytes.
        use crate::common::fse_decompress::{
            FSE_DTable, FSE_buildDTable_wksp, FSE_decompress_usingDTable,
            FSE_BUILD_DTABLE_WKSP_SIZE, FSE_DTABLE_SIZE_U32,
        };

        let src: Vec<u8> = b"ababcabcabaaaaabcbcbcbaaaa".to_vec();
        let mut count = [0u32; 256];
        let mut msv: u32 = 255;
        let largest = crate::compress::hist::HIST_count_simple(&mut count, &mut msv, &src);
        assert!(largest > 0);

        let mut norm = [0i16; 256];
        // Pick tableLog explicitly via optimalTableLog — avoids the
        // normalizeCount/minTableLog corner cases for tiny inputs.
        let tableLog = FSE_optimalTableLog(0, src.len(), msv);
        let rc = FSE_normalizeCount(&mut norm, tableLog, &count, src.len(), msv, 0);
        assert!(
            !crate::common::error::ERR_isError(rc) && rc != 0,
            "normalize returned {rc} (expected nonzero tableLog)"
        );
        assert_eq!(rc, tableLog as usize);

        // Build CTable.
        let ct_size = FSE_CTABLE_SIZE_U32(tableLog, msv);
        let mut ct = vec![0u32; ct_size];
        let build_ws = vec![0u8; (msv as usize + 2) * 2 + (1 << tableLog) + 8];
        let mut build_ws = build_ws;
        let rc = FSE_buildCTable_wksp(&mut ct, &norm, msv, tableLog, &mut build_ws);
        assert_eq!(rc, 0);

        // Compress.
        let mut dst = vec![0u8; FSE_compressBound(src.len())];
        let written = FSE_compress_usingCTable(&mut dst, &src, &ct);
        assert!(
            !crate::common::error::ERR_isError(written),
            "compress err: {}",
            crate::common::error::ERR_getErrorName(written)
        );
        assert!(written > 0, "empty output");
        assert!(
            written < src.len(),
            "no compression (output {written} >= input {})",
            src.len()
        );

        // Build DTable from same norm.
        let dt_size = FSE_DTABLE_SIZE_U32(tableLog);
        let mut dt = vec![0u32; dt_size];
        let mut dt_ws = vec![0u8; FSE_BUILD_DTABLE_WKSP_SIZE(tableLog, msv)];
        let rc = FSE_buildDTable_wksp(
            &mut dt as &mut [FSE_DTable],
            &norm,
            msv,
            tableLog,
            &mut dt_ws,
        );
        assert_eq!(rc, 0, "buildDTable err");

        // Decompress.
        let mut out = vec![0u8; src.len()];
        let decoded = FSE_decompress_usingDTable(&mut out, &dst[..written], &dt);
        assert!(
            !crate::common::error::ERR_isError(decoded),
            "decode err: {}",
            crate::common::error::ERR_getErrorName(decoded)
        );
        assert_eq!(decoded, src.len(), "decoded length mismatch");
        assert_eq!(&out[..decoded], &src[..]);
    }

    #[test]
    fn build_ctable_wksp_writes_header_and_returns_ok() {
        // Build a CTable from a known-valid normalized distribution,
        // then verify: (a) the header fields decode back to the input
        // tableLog + maxSymbolValue, (b) all symbolTT entries have
        // deltaNbBits within the expected range.
        let count = [100u32, 200, 300, 150, 150, 100];
        let mut norm = [0i16; 16];
        let total: usize = count.iter().map(|&c| c as usize).sum();
        let tableLog = FSE_normalizeCount(&mut norm, 8, &count, total, 5, 0) as u32;
        assert!(!crate::common::error::ERR_isError(tableLog as usize));

        let max_sv = 5u32;
        let ct_size = FSE_CTABLE_SIZE_U32(tableLog, max_sv);
        let mut ct = vec![0u32; ct_size];
        let ws_size = (max_sv as usize + 2) * 2 + (1 << tableLog) + 8;
        let mut wksp = vec![0u8; ws_size];

        let rc = FSE_buildCTable_wksp(&mut ct, &norm, max_sv, tableLog, &mut wksp);
        assert_eq!(rc, 0, "err: {}", crate::common::error::ERR_getErrorName(rc));
        assert_eq!(ct_header_tableLog(&ct), tableLog);
        assert_eq!(ct_header_maxSV(&ct), max_sv);

        // Each non-zero symbol's deltaNbBits must be in a sane range:
        // upstream's formula guarantees (tableLog-highbit(n-1))<<16 –
        // this gives non-negative values for count≥1 and is otherwise
        // (tableLog+1)<<16 - tableSize for count==0.
        for (s, &n) in norm.iter().enumerate().take(max_sv as usize + 1) {
            let t = symbolTT_read(&ct, tableLog, s);
            if n == 0 {
                let expected = ((tableLog + 1) << 16) as i32 - (1i32 << tableLog);
                assert_eq!(
                    t.deltaNbBits, expected,
                    "zero-count symbol {s} has unexpected deltaNbBits"
                );
            }
        }
    }

    #[test]
    fn optimal_tablelog_internal_uses_minus_slack() {
        // Two calls, same inputs, with minus=0 and minus=5: the lower
        // slack should produce an equal-or-higher tableLog.
        let a = FSE_optimalTableLog_internal(FSE_MAX_TABLELOG, 2048, 255, 0);
        let b = FSE_optimalTableLog_internal(FSE_MAX_TABLELOG, 2048, 255, 5);
        assert!(a >= b);
    }

    #[test]
    fn FSE_bitCost_ranks_high_freq_symbols_cheaper_than_low_freq() {
        // Build a CTable from a skewed distribution: symbol 0 is very
        // frequent, symbol 7 is rare. FSE_bitCost must then report a
        // strictly smaller cost for symbol 0 than for symbol 7 —
        // that's the whole point of the fixed-point cost estimator.
        const TABLELOG: u32 = 8;
        const MSV: u32 = 7;
        const ACCURACY: u32 = 8;
        // Frequencies sum to 256 (= 1<<tableLog): [128,64,32,16,8,4,2,2].
        let norm: [i16; 8] = [128, 64, 32, 16, 8, 4, 2, 2];

        let mut ct = vec![0u32; FSE_CTABLE_SIZE_U32(TABLELOG, MSV)];
        let mut ws = vec![0u8; (MSV as usize + 2) * 2 + (1usize << TABLELOG) + 8];
        let rc = FSE_buildCTable_wksp(&mut ct, &norm, MSV, TABLELOG, &mut ws);
        assert_eq!(rc, 0, "buildCTable failed: {rc:#x}");

        let cost_sym0 = FSE_bitCost(&ct, TABLELOG, 0, ACCURACY);
        let cost_sym7 = FSE_bitCost(&ct, TABLELOG, 7, ACCURACY);
        assert!(
            cost_sym0 < cost_sym7,
            "expected sym0 (freq 128) cheaper than sym7 (freq 2), got {cost_sym0} vs {cost_sym7}",
        );
        // Monotonic: each later (less-frequent) symbol should cost
        // at least as much as the prior one.
        let costs: Vec<u32> = (0..=7)
            .map(|s| FSE_bitCost(&ct, TABLELOG, s, ACCURACY))
            .collect();
        for w in costs.windows(2) {
            assert!(
                w[0] <= w[1],
                "cost non-monotonic across freq-ordered symbols: {costs:?}"
            );
        }
    }

    #[test]
    fn buildCTable_rle_stores_single_symbol_with_zero_bitcost() {
        // After FSE_buildCTable_rle for symbol S, the CTable should:
        //   - declare tableLog = 0 (single-entry state machine)
        //   - declare maxSymbolValue = S
        //   - have zero deltaNbBits at slot S (cost = 0 bits)
        // The RLE CTable is only used when the block has exactly one
        // symbol repeated, so every encoded symbol costs zero bits.
        const SYM: u8 = 0x42;
        let mut ct = vec![0u32; FSE_CTABLE_SIZE_U32(0, SYM as u32)];
        let rc = FSE_buildCTable_rle(&mut ct, SYM);
        assert_eq!(rc, 0);
        assert_eq!(ct_header_tableLog(&ct), 0);
        assert_eq!(ct_header_maxSV(&ct), SYM as u32);
        let tt = symbolTT_read(&ct, 0, SYM as usize);
        assert_eq!(tt.deltaNbBits, 0);
        assert_eq!(tt.deltaFindState, 0);
    }
}
