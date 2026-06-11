//! Translation of `lib/common/fse_decompress.c`.
//!
//! Ports `FSE_buildDTable_wksp`, `FSE_decompress_usingDTable`,
//! `FSE_decompress_wksp{,_bmi2}`, and the inlined helpers
//! (`FSE_initDState`, `FSE_decodeSymbol{,Fast}`, `FSE_peekSymbol`,
//! `FSE_updateState`, `FSE_endOfDState`) from `fse.h`.
//!
//! Decoding-table layout matches upstream exactly: the caller-provided
//! `FSE_DTable` is a `U32` array; the first word holds the
//! `FSE_DTableHeader` (tableLog + fastMode), the following `1<<tableLog`
//! words each pack one `FSE_decode_t` (newState + symbol + nbBits).
//! The Rust port stores those entries with an identical bit layout
//! inside each `u32` so that code-complexity-comparator sees the same
//! two-step build → decode structure.

use crate::common::bits::ZSTD_highbit32;
use crate::common::bitstream::{
    BIT_DStream_overflow, BIT_DStream_t, BIT_DStream_unfinished, BIT_initDStream, BIT_readBits,
    BIT_readBitsFast, BIT_reloadDStream,
};
use crate::common::error::{ErrorCode, ERROR};

// Upstream defaults (matching lib/common/fse.h when built without
// overrides, which is how zstd itself is compiled).
pub const FSE_MAX_MEMORY_USAGE: u32 = 14;
pub const FSE_MAX_TABLELOG: u32 = FSE_MAX_MEMORY_USAGE - 2; // 12
pub const FSE_MAX_SYMBOL_VALUE: u32 = 255;
pub const FSE_MIN_TABLELOG: u32 = 5;
pub const FSE_TABLELOG_ABSOLUTE_MAX: u32 = 15;

/// Rust-only helper for C macros of the form `1 << n`. Valid zstd
/// callers keep `n` small; for invalid public inputs, saturating keeps
/// the subsequent upstream-style size check on the error path instead
/// of panicking on a Rust shift overflow.
#[inline(always)]
const fn fse_one_shift_saturating(n: u32) -> usize {
    if n as usize >= usize::BITS as usize {
        usize::MAX
    } else {
        1usize << n
    }
}

/// Port of `FSE_TABLESTEP`. Stride used to spread normalized symbols
/// across the decode table during table construction.
#[inline(always)]
pub const fn FSE_TABLESTEP(tableSize: u32) -> u32 {
    (tableSize >> 1) + (tableSize >> 3) + 3
}

/// Port of `FSE_DTABLE_SIZE_U32`. Number of `u32` entries in a DTable
/// of `maxTableLog` (1 header word + `1<<maxTableLog` decode entries).
#[inline(always)]
pub const fn FSE_DTABLE_SIZE_U32(maxTableLog: u32) -> usize {
    1usize.saturating_add(fse_one_shift_saturating(maxTableLog))
}

/// Port of `FSE_BUILD_DTABLE_WKSP_SIZE`. Scratch byte count required by
/// `FSE_buildDTable_wksp` for the given table parameters.
#[inline(always)]
pub const fn FSE_BUILD_DTABLE_WKSP_SIZE(maxTableLog: u32, maxSymbolValue: u32) -> usize {
    2usize
        .saturating_mul((maxSymbolValue as usize).saturating_add(1))
        .saturating_add(fse_one_shift_saturating(maxTableLog))
        .saturating_add(8)
}

/// Upstream `FSE_BUILD_DTABLE_WKSP_SIZE_U32`.
#[inline(always)]
const fn FSE_BUILD_DTABLE_WKSP_SIZE_U32(maxTableLog: u32, maxSymbolValue: u32) -> usize {
    FSE_BUILD_DTABLE_WKSP_SIZE(maxTableLog, maxSymbolValue).div_ceil(4)
}

/// Upstream `FSE_DECOMPRESS_WKSP_SIZE_U32`.
#[inline(always)]
const fn FSE_DECOMPRESS_WKSP_SIZE_U32(maxTableLog: u32, maxSymbolValue: u32) -> usize {
    FSE_DTABLE_SIZE_U32(maxTableLog)
        .saturating_add(1)
        .saturating_add(FSE_BUILD_DTABLE_WKSP_SIZE_U32(maxTableLog, maxSymbolValue))
        .saturating_add(((FSE_MAX_SYMBOL_VALUE + 1) as usize) / 2)
        .saturating_add(1)
}

/// Upstream `FSE_DECOMPRESS_WKSP_SIZE`.
#[inline(always)]
const fn FSE_DECOMPRESS_WKSP_SIZE(maxTableLog: u32, maxSymbolValue: u32) -> usize {
    FSE_DECOMPRESS_WKSP_SIZE_U32(maxTableLog, maxSymbolValue).saturating_mul(4)
}

pub type FSE_DTable = u32;

/// Rust-only helper: pack `(newState, symbol, nbBits)` into a single
/// `u32` DTable slot, matching upstream's `FSE_decode_t` bit layout.
#[inline(always)]
fn pack_decode(newState: u16, symbol: u8, nbBits: u8) -> u32 {
    (newState as u32) | ((symbol as u32) << 16) | ((nbBits as u32) << 24)
}

/// Rust-only helper: extract `newState` from a packed DTable slot.
#[inline(always)]
fn unpack_newState(slot: u32) -> u16 {
    slot as u16
}

/// Rust-only helper: extract the decoded symbol byte from a packed slot.
#[inline(always)]
fn unpack_symbol(slot: u32) -> u8 {
    (slot >> 16) as u8
}

/// Rust-only helper: extract the `nbBits` field from a packed slot.
#[inline(always)]
fn unpack_nbBits(slot: u32) -> u8 {
    (slot >> 24) as u8
}

/// Rust-only helper: pack `FSE_DTableHeader` into the DTable's first
/// `u32` — bits 0..16 = tableLog, bits 16..32 = fastMode.
#[inline(always)]
fn pack_header(tableLog: u16, fastMode: u16) -> u32 {
    (tableLog as u32) | ((fastMode as u32) << 16)
}

/// Rust-only helper: extract `tableLog` from the packed DTable header.
#[inline(always)]
fn header_tableLog(slot: u32) -> u32 {
    slot & 0xFFFF
}

/// Rust-only helper: extract `fastMode` from the packed DTable header.
#[inline(always)]
fn header_fastMode(slot: u32) -> u32 {
    (slot >> 16) & 0xFFFF
}

// ---- FSE_DState_t -----------------------------------------------------

/// Mirror of `FSE_DState_t`. The C version holds an untyped `const void*
/// table` pointer; in Rust we keep an offset into the caller's dtable
/// slice (the header sits at index 0, so `table_offset == 1`).
#[derive(Debug, Clone, Copy)]
pub struct FSE_DState_t {
    pub state: usize,
    pub table_offset: usize,
}

/// Port of `FSE_initDState`.
#[inline(always)]
pub fn FSE_initDState(dsp: &mut FSE_DState_t, bitD: &mut BIT_DStream_t, dt: &[FSE_DTable]) {
    let header = dt[0];
    let tableLog = header_tableLog(header);
    dsp.state = BIT_readBits(bitD, tableLog);
    BIT_reloadDStream(bitD);
    dsp.table_offset = 1;
}

/// Port of `FSE_peekSymbol`.
#[inline(always)]
pub fn FSE_peekSymbol(dsp: &FSE_DState_t, dt: &[FSE_DTable]) -> u8 {
    let slot = dt[dsp.table_offset + dsp.state];
    unpack_symbol(slot)
}

/// Port of `FSE_updateState`.
#[inline(always)]
pub fn FSE_updateState(dsp: &mut FSE_DState_t, bitD: &mut BIT_DStream_t, dt: &[FSE_DTable]) {
    let slot = dt[dsp.table_offset + dsp.state];
    let nbBits = unpack_nbBits(slot);
    let lowBits = BIT_readBits(bitD, nbBits as u32);
    dsp.state = unpack_newState(slot) as usize + lowBits;
}

/// Port of `FSE_decodeSymbol`.
#[inline(always)]
pub fn FSE_decodeSymbol(dsp: &mut FSE_DState_t, bitD: &mut BIT_DStream_t, dt: &[FSE_DTable]) -> u8 {
    let slot = dt[dsp.table_offset + dsp.state];
    let nbBits = unpack_nbBits(slot);
    let symbol = unpack_symbol(slot);
    let lowBits = BIT_readBits(bitD, nbBits as u32);
    dsp.state = unpack_newState(slot) as usize + lowBits;
    symbol
}

/// Port of `FSE_decodeSymbolFast`.
#[inline(always)]
pub fn FSE_decodeSymbolFast(
    dsp: &mut FSE_DState_t,
    bitD: &mut BIT_DStream_t,
    dt: &[FSE_DTable],
) -> u8 {
    let slot = dt[dsp.table_offset + dsp.state];
    let nbBits = unpack_nbBits(slot);
    let symbol = unpack_symbol(slot);
    let lowBits = BIT_readBitsFast(bitD, nbBits as u32);
    dsp.state = unpack_newState(slot) as usize + lowBits;
    symbol
}

/// Port of `FSE_endOfDState`.
#[inline(always)]
pub fn FSE_endOfDState(dsp: &FSE_DState_t) -> u32 {
    (dsp.state == 0) as u32
}

// ---- FSE_buildDTable_internal / _wksp ---------------------------------

/// Port of `FSE_buildDTable_internal`. Writes the header into `dt[0]`
/// and the decode entries into `dt[1..=1<<tableLog]`.
///
/// Upstream receives `workSpace` for `symbolNext` + `spread` scratch.
/// This Rust port preserves the same workspace-size contract, but uses
/// local typed arrays instead of reinterpreting an arbitrary `u8`
/// workspace as typed storage. That avoids alignment and aliasing
/// hazards without changing the required scratch-size checks.
pub fn FSE_buildDTable_internal(
    dt: &mut [FSE_DTable],
    normalizedCounter: &[i16],
    maxSymbolValue: u32,
    tableLog: u32,
    _workSpace: &mut [u8],
    wkspSize: usize,
) -> usize {
    if FSE_BUILD_DTABLE_WKSP_SIZE(tableLog, maxSymbolValue) > wkspSize {
        return ERROR(ErrorCode::MaxSymbolValueTooLarge);
    }
    if maxSymbolValue > FSE_MAX_SYMBOL_VALUE {
        return ERROR(ErrorCode::MaxSymbolValueTooLarge);
    }
    if tableLog > FSE_MAX_TABLELOG {
        return ERROR(ErrorCode::TableLogTooLarge);
    }
    if tableLog == 0 {
        return ERROR(ErrorCode::TableLogTooLarge);
    }

    let maxSV1 = maxSymbolValue + 1;
    let tableSize = 1u32 << tableLog;

    // Rust safety adaptation: upstream lays these arrays out inside
    // workSpace, while this port keeps typed scratch on the stack after
    // validating the caller supplied the upstream-required byte count.
    let mut symbolNext: [u32; (FSE_MAX_SYMBOL_VALUE + 1) as usize] =
        [0u32; (FSE_MAX_SYMBOL_VALUE + 1) as usize];
    let mut spread = [0u8; 1 << FSE_MAX_TABLELOG];

    let mut highThreshold: i64 = tableSize as i64 - 1;
    let mut fastMode: u16 = 1;
    let largeLimit: i16 = 1 << (tableLog - 1);

    // Init, lay down -1 symbols at the top of the decode table.
    for s in 0..maxSV1 {
        let nc = normalizedCounter[s as usize];
        if nc == -1 {
            // Upstream writes tableDecode[highThreshold--].symbol = s.
            let slot = pack_decode(0, s as u8, 0);
            dt[1 + highThreshold as usize] = slot;
            highThreshold -= 1;
            symbolNext[s as usize] = 1;
        } else {
            if nc >= largeLimit {
                fastMode = 0;
            }
            symbolNext[s as usize] = nc as u16 as u32;
        }
    }
    dt[0] = pack_header(tableLog as u16, fastMode);

    // Spread symbols.
    let tableMask = (tableSize - 1) as usize;
    let step = FSE_TABLESTEP(tableSize) as usize;

    if highThreshold == tableSize as i64 - 1 {
        // Fast path: lay down in order, then spread in two unrolled passes.
        let mut pos: usize = 0;
        for s in 0..maxSV1 {
            let n = normalizedCounter[s as usize] as i32;
            for i in 0..n as usize {
                spread[pos + i] = s as u8;
            }
            if n > 0 {
                pos += n as usize;
            }
        }
        debug_assert_eq!(pos, tableSize as usize);

        let mut position: usize = 0;
        let unroll = 2usize;
        debug_assert!((tableSize as usize).is_multiple_of(unroll));
        let mut s = 0;
        while s < tableSize as usize {
            for u in 0..unroll {
                let uPos = (position + (u * step)) & tableMask;
                let prev = dt[1 + uPos];
                // Only the symbol byte is set here; newState and nbBits
                // get filled in in the final pass below.
                dt[1 + uPos] = (prev & !0x00FF_0000) | ((spread[s + u] as u32) << 16);
            }
            position = (position + (unroll * step)) & tableMask;
            s += unroll;
        }
        debug_assert_eq!(position, 0);
    } else {
        // Slow path with lowprob region.
        let mut position: usize = 0;
        for s in 0..maxSV1 {
            let n = normalizedCounter[s as usize];
            for _ in 0..n {
                let prev = dt[1 + position];
                dt[1 + position] = (prev & !0x00FF_0000) | (s << 16);
                position = (position + step) & tableMask;
                while position as i64 > highThreshold {
                    position = (position + step) & tableMask;
                }
            }
        }
        if position != 0 {
            return ERROR(ErrorCode::Generic);
        }
    }

    // Fill nbBits and newState for every slot.
    for u in 0..tableSize as usize {
        let symbol = unpack_symbol(dt[1 + u]);
        let nextState = symbolNext[symbol as usize];
        symbolNext[symbol as usize] = nextState + 1;
        let nbBits = (tableLog - ZSTD_highbit32(nextState)) as u8;
        let newState = ((nextState << nbBits) - tableSize) as u16;
        dt[1 + u] = pack_decode(newState, symbol, nbBits);
    }

    0
}

/// Port of `FSE_buildDTable_wksp`. Thin wrapper that forwards to
/// `FSE_buildDTable_internal` using the provided workspace's length.
pub fn FSE_buildDTable_wksp(
    dt: &mut [FSE_DTable],
    normalizedCounter: &[i16],
    maxSymbolValue: u32,
    tableLog: u32,
    workSpace: &mut [u8],
) -> usize {
    let sz = workSpace.len();
    FSE_buildDTable_internal(
        dt,
        normalizedCounter,
        maxSymbolValue,
        tableLog,
        workSpace,
        sz,
    )
}

/// Port of legacy `FSE_buildDTable_raw`. Builds a trivial FSE DTable
/// where every symbol `s` in `0..=(1<<nbBits)-1` has a cell with
/// `nbBits` bits and `symbol = s`. Used by legacy decoders when the
/// FSE block header flags an uncompressed (raw) table. Modern zstd
/// never emits this table-type, but the helper stays for symmetry
/// with upstream's API surface.
pub fn FSE_buildDTable_raw(dt: &mut [FSE_DTable], nbBits: u32) -> usize {
    if nbBits < 1 {
        return ERROR(ErrorCode::Generic);
    }
    let tableSize = 1u32 << nbBits;
    if dt.len() < 1 + tableSize as usize {
        return ERROR(ErrorCode::TableLogTooLarge);
    }
    dt[0] = pack_header(nbBits as u16, 1);
    for s in 0..tableSize {
        dt[1 + s as usize] = pack_decode(0, s as u8, nbBits as u8);
    }
    0
}

/// Port of legacy `FSE_buildDTable_rle`. Single-symbol decoder: one
/// entry with `nbBits=0` and `newState=0`, always decoding to
/// `symbolValue`.
pub fn FSE_buildDTable_rle(dt: &mut [FSE_DTable], symbolValue: u8) -> usize {
    if dt.len() < 2 {
        return ERROR(ErrorCode::TableLogTooLarge);
    }
    dt[0] = pack_header(0, 0);
    dt[1] = pack_decode(0, symbolValue, 0);
    0
}

// ---- Decompression ----------------------------------------------------

/// Port of `FSE_decompress_usingDTable_generic`. Decodes the FSE
/// bitstream using two interleaved decoder states and the prebuilt
/// `dt`. The `FAST` const-generic selects `FSE_decodeSymbolFast` vs
/// `FSE_decodeSymbol` to match upstream's dual-template instantiation.
fn FSE_decompress_usingDTable_generic<const FAST: bool>(
    dst: &mut [u8],
    cSrc: &[u8],
    dt: &[FSE_DTable],
) -> usize {
    let maxDstSize = dst.len();
    let cSrcSize = cSrc.len();
    let mut op: usize = 0;
    let omax = maxDstSize;
    let olimit = omax.saturating_sub(3);

    let mut bitD = BIT_DStream_t::default();
    let rc = BIT_initDStream(&mut bitD, cSrc, cSrcSize);
    if crate::common::error::ERR_isError(rc) {
        return rc;
    }

    let mut s1 = FSE_DState_t {
        state: 0,
        table_offset: 0,
    };
    let mut s2 = FSE_DState_t {
        state: 0,
        table_offset: 0,
    };
    FSE_initDState(&mut s1, &mut bitD, dt);
    FSE_initDState(&mut s2, &mut bitD, dt);

    if BIT_reloadDStream(&mut bitD) == BIT_DStream_overflow {
        return ERROR(ErrorCode::CorruptionDetected);
    }

    #[inline(always)]
    fn get<const F: bool>(st: &mut FSE_DState_t, b: &mut BIT_DStream_t, dt: &[FSE_DTable]) -> u8 {
        if F {
            FSE_decodeSymbolFast(st, b, dt)
        } else {
            FSE_decodeSymbol(st, b, dt)
        }
    }

    // 4 symbols per loop.
    let container_bits = core::mem::size_of::<usize>() as u32 * 8;
    let long_reload_needed = FSE_MAX_TABLELOG * 4 + 7 > container_bits;
    let short_reload_needed = FSE_MAX_TABLELOG * 2 + 7 > container_bits;
    loop {
        let refill = BIT_reloadDStream(&mut bitD);
        if refill != BIT_DStream_unfinished || op >= olimit {
            break;
        }
        dst[op] = get::<FAST>(&mut s1, &mut bitD, dt);
        if short_reload_needed {
            BIT_reloadDStream(&mut bitD);
        }
        dst[op + 1] = get::<FAST>(&mut s2, &mut bitD, dt);
        if long_reload_needed && BIT_reloadDStream(&mut bitD) > BIT_DStream_unfinished {
            op += 2;
            // Need to exit the main loop and go to tail — mirror upstream's `break;`.
            // Tail loop handles termination.
            break;
        }
        dst[op + 2] = get::<FAST>(&mut s1, &mut bitD, dt);
        if short_reload_needed {
            BIT_reloadDStream(&mut bitD);
        }
        dst[op + 3] = get::<FAST>(&mut s2, &mut bitD, dt);
        op += 4;
    }

    #[inline(always)]
    fn tail_has_room_for_two(op: usize, omax: usize) -> bool {
        omax >= 2 && op <= omax - 2
    }

    // Tail.
    loop {
        if !tail_has_room_for_two(op, omax) {
            return ERROR(ErrorCode::DstSizeTooSmall);
        }
        dst[op] = get::<FAST>(&mut s1, &mut bitD, dt);
        op += 1;
        if BIT_reloadDStream(&mut bitD) == BIT_DStream_overflow {
            dst[op] = get::<FAST>(&mut s2, &mut bitD, dt);
            op += 1;
            break;
        }
        if !tail_has_room_for_two(op, omax) {
            return ERROR(ErrorCode::DstSizeTooSmall);
        }
        dst[op] = get::<FAST>(&mut s2, &mut bitD, dt);
        op += 1;
        if BIT_reloadDStream(&mut bitD) == BIT_DStream_overflow {
            dst[op] = get::<FAST>(&mut s1, &mut bitD, dt);
            op += 1;
            break;
        }
    }

    op
}

/// Port of `FSE_decompress_usingDTable`. Dispatches to the FAST or
/// non-FAST `FSE_decompress_usingDTable_generic` based on the
/// `fastMode` flag stored in the DTable header.
pub fn FSE_decompress_usingDTable(dst: &mut [u8], cSrc: &[u8], dt: &[FSE_DTable]) -> usize {
    let fastMode = header_fastMode(dt[0]);
    if fastMode != 0 {
        FSE_decompress_usingDTable_generic::<true>(dst, cSrc, dt)
    } else {
        FSE_decompress_usingDTable_generic::<false>(dst, cSrc, dt)
    }
}

// ---- wksp-based entry points ------------------------------------------

/// Port of `FSE_decompress_wksp_body`. Reads the NCount, builds the
/// DTable, then decodes. Layout of `workSpace` (bytes):
///   [0 .. sizeof(FSE_DecompressWksp))   : ncount[FSE_MAX_SYMBOL_VALUE+1]
///   [.. dtable]
///   [.. build scratch]
///
/// Rust safety adaptation: this port validates that `workSpace` is large
/// enough for the same upstream layout, but keeps `ncount` and `dtable`
/// as local typed arrays instead of borrowing typed views from an
/// arbitrary byte slice. The remaining workspace tail is still passed to
/// the DTable builder so upstream size checks are preserved.
pub fn FSE_decompress_wksp(
    dst: &mut [u8],
    cSrc: &[u8],
    maxLog: u32,
    workSpace: &mut [u8],
) -> usize {
    FSE_decompress_wksp_body_default(dst, cSrc, maxLog, workSpace)
}

/// Port of `FSE_decompress_wksp_body`. Reads the NCount header from
/// `cSrc`, builds an FSE DTable, then decodes the remaining bytes into
/// `dst`.
fn FSE_decompress_wksp_body(
    dst: &mut [u8],
    cSrc: &[u8],
    maxLog: u32,
    workSpace: &mut [u8],
    bmi2: i32,
) -> usize {
    // Upstream first reserves ncount[256] from workSpace. We keep the
    // typed array local and reserve the same byte count from the caller's
    // workspace before handing the remainder to the DTable build step.
    const N_COUNT_BYTES: usize = ((FSE_MAX_SYMBOL_VALUE + 1) as usize) * 2;
    let wksp_len = workSpace.len();
    if wksp_len < N_COUNT_BYTES {
        return ERROR(ErrorCode::Generic);
    }
    let (_ncount_wksp, dtable_and_build_wksp) = workSpace.split_at_mut(N_COUNT_BYTES);

    // Parse NCount.
    let mut ncount = [0i16; (FSE_MAX_SYMBOL_VALUE + 1) as usize];
    let mut maxSymbolValue: u32 = FSE_MAX_SYMBOL_VALUE;
    let mut tableLog: u32 = 0;
    let NCountLength = crate::common::entropy_common::FSE_readNCount_bmi2(
        &mut ncount,
        &mut maxSymbolValue,
        &mut tableLog,
        cSrc,
        bmi2,
    );
    if crate::common::error::ERR_isError(NCountLength) {
        return NCountLength;
    }
    if tableLog > maxLog {
        return ERROR(ErrorCode::TableLogTooLarge);
    }
    if tableLog > FSE_MAX_TABLELOG {
        return ERROR(ErrorCode::TableLogTooLarge);
    }
    if FSE_DECOMPRESS_WKSP_SIZE(tableLog, maxSymbolValue) > wksp_len {
        return ERROR(ErrorCode::TableLogTooLarge);
    }
    let cSrcSize = cSrc.len();
    debug_assert!(NCountLength <= cSrcSize);
    let ip_offset = NCountLength;
    let remaining = &cSrc[ip_offset..];

    // Build DTable. Upstream stores the DTable in workSpace immediately
    // after ncount. We use a local typed array to avoid alignment-sensitive
    // casts from `&mut [u8]`, and still carve the equivalent DTable bytes
    // before passing build scratch to the builder.
    let dt_sz = FSE_DTABLE_SIZE_U32(tableLog);
    let mut dtable = [0u32; FSE_DTABLE_SIZE_U32(FSE_MAX_TABLELOG)];
    let dtable = &mut dtable[..dt_sz];
    let build_wksp_sz = FSE_BUILD_DTABLE_WKSP_SIZE(tableLog, maxSymbolValue);
    let dtable_bytes = dt_sz * core::mem::size_of::<u32>();
    if dtable_and_build_wksp.len() < dtable_bytes + build_wksp_sz {
        return ERROR(ErrorCode::TableLogTooLarge);
    }
    let (_dtable_wksp, build_wksp) = dtable_and_build_wksp.split_at_mut(dtable_bytes);
    let rc = FSE_buildDTable_internal(
        dtable,
        &ncount,
        maxSymbolValue,
        tableLog,
        build_wksp,
        build_wksp_sz,
    );
    if crate::common::error::ERR_isError(rc) {
        return rc;
    }

    FSE_decompress_usingDTable(dst, remaining, dtable)
}

/// Port of `FSE_decompress_wksp_body_default`.
fn FSE_decompress_wksp_body_default(
    dst: &mut [u8],
    cSrc: &[u8],
    maxLog: u32,
    workSpace: &mut [u8],
) -> usize {
    FSE_decompress_wksp_body(dst, cSrc, maxLog, workSpace, 0)
}

/// Port of `FSE_decompress_wksp_bmi2`. Upstream conditionally dispatches
/// to a BMI2-attributed body when `DYNAMIC_BMI2` is enabled. This scalar
/// Rust port has no separate BMI2 codegen path, so the public wrapper
/// mirrors the default fallback.
pub fn FSE_decompress_wksp_bmi2(
    dst: &mut [u8],
    cSrc: &[u8],
    maxLog: u32,
    workSpace: &mut [u8],
    bmi2: i32,
) -> usize {
    FSE_decompress_wksp_body(dst, cSrc, maxLog, workSpace, bmi2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_dtable_rle_stores_symbol_and_zero_bits() {
        let mut dt = vec![0u32; 2];
        let rc = FSE_buildDTable_rle(&mut dt, 0x42);
        assert_eq!(rc, 0);
        assert_eq!(header_tableLog(dt[0]), 0);
        assert_eq!(header_fastMode(dt[0]), 0);
        assert_eq!(unpack_symbol(dt[1]), 0x42);
        assert_eq!(unpack_nbBits(dt[1]), 0);
        assert_eq!(unpack_newState(dt[1]), 0);
    }

    #[test]
    fn build_dtable_raw_populates_uniform_entries() {
        let nbBits = 3u32;
        let mut dt = vec![0u32; 1 + (1 << nbBits)];
        let rc = FSE_buildDTable_raw(&mut dt, nbBits);
        assert_eq!(rc, 0);
        assert_eq!(header_tableLog(dt[0]), nbBits);
        assert_eq!(header_fastMode(dt[0]), 1);
        for s in 0..(1 << nbBits) {
            assert_eq!(unpack_symbol(dt[1 + s as usize]), s as u8);
            assert_eq!(unpack_nbBits(dt[1 + s as usize]), nbBits as u8);
        }
    }

    #[test]
    fn build_dtable_raw_rejects_zero_bits() {
        let mut dt = vec![0u32; 2];
        let rc = FSE_buildDTable_raw(&mut dt, 0);
        assert!(crate::common::error::ERR_isError(rc));
    }

    #[test]
    fn build_dtable_rle_rejects_tiny_dt() {
        // RLE DTable requires at least 2 slots (header + 1 entry).
        let mut dt_empty: Vec<u32> = Vec::new();
        assert!(crate::common::error::ERR_isError(FSE_buildDTable_rle(
            &mut dt_empty,
            0
        )));
        let mut dt_one = vec![0u32; 1];
        assert!(crate::common::error::ERR_isError(FSE_buildDTable_rle(
            &mut dt_one,
            0
        )));
    }

    #[test]
    fn build_dtable_raw_rejects_too_small_dt() {
        // Raw DTable at nbBits=3 needs 1 + 2^3 = 9 slots. 8 is short.
        let mut dt_short = vec![0u32; 8];
        assert!(crate::common::error::ERR_isError(FSE_buildDTable_raw(
            &mut dt_short,
            3
        )));
    }

    #[test]
    fn build_dtable_workspace_size_u32_matches_upstream_boundaries() {
        assert_eq!(FSE_BUILD_DTABLE_WKSP_SIZE(0, 0), 11);
        assert_eq!(FSE_BUILD_DTABLE_WKSP_SIZE_U32(0, 0), 3);

        assert_eq!(FSE_BUILD_DTABLE_WKSP_SIZE(0, 1), 13);
        assert_eq!(FSE_BUILD_DTABLE_WKSP_SIZE_U32(0, 1), 4);

        assert_eq!(
            FSE_BUILD_DTABLE_WKSP_SIZE(FSE_MIN_TABLELOG, FSE_MAX_SYMBOL_VALUE),
            552
        );
        assert_eq!(
            FSE_BUILD_DTABLE_WKSP_SIZE_U32(FSE_MIN_TABLELOG, FSE_MAX_SYMBOL_VALUE),
            138
        );

        assert_eq!(
            FSE_BUILD_DTABLE_WKSP_SIZE(FSE_MAX_TABLELOG, FSE_MAX_SYMBOL_VALUE),
            4616
        );
        assert_eq!(
            FSE_BUILD_DTABLE_WKSP_SIZE_U32(FSE_MAX_TABLELOG, FSE_MAX_SYMBOL_VALUE),
            1154
        );

        assert_eq!(
            FSE_BUILD_DTABLE_WKSP_SIZE(FSE_TABLELOG_ABSOLUTE_MAX, FSE_MAX_SYMBOL_VALUE),
            33288
        );
        assert_eq!(
            FSE_BUILD_DTABLE_WKSP_SIZE_U32(FSE_TABLELOG_ABSOLUTE_MAX, FSE_MAX_SYMBOL_VALUE),
            8322
        );

        assert_eq!(
            FSE_BUILD_DTABLE_WKSP_SIZE(FSE_TABLELOG_ABSOLUTE_MAX, 0),
            32778
        );
        assert_eq!(
            FSE_BUILD_DTABLE_WKSP_SIZE_U32(FSE_TABLELOG_ABSOLUTE_MAX, 0),
            8195
        );
    }

    #[test]
    fn build_dtable_invalid_parameters_return_errors_without_overflow() {
        let mut dt = vec![0u32; FSE_DTABLE_SIZE_U32(FSE_MAX_TABLELOG)];
        let mut norm = [0i16; (FSE_MAX_SYMBOL_VALUE + 1) as usize];
        norm[0] = 1;
        let mut wksp =
            vec![0u8; FSE_BUILD_DTABLE_WKSP_SIZE(FSE_MIN_TABLELOG, FSE_MAX_SYMBOL_VALUE)];

        let rc = FSE_buildDTable_wksp(&mut dt, &norm, FSE_MAX_SYMBOL_VALUE, 0, &mut wksp);
        assert_eq!(
            crate::common::error::ERR_getErrorCode(rc),
            ErrorCode::TableLogTooLarge
        );

        let rc = FSE_buildDTable_wksp(
            &mut dt,
            &norm,
            FSE_MAX_SYMBOL_VALUE + 1,
            FSE_MIN_TABLELOG,
            &mut wksp,
        );
        assert_eq!(
            crate::common::error::ERR_getErrorCode(rc),
            ErrorCode::MaxSymbolValueTooLarge
        );

        let rc = FSE_buildDTable_wksp(&mut dt, &norm, FSE_MAX_SYMBOL_VALUE, usize::BITS, &mut []);
        assert_eq!(
            crate::common::error::ERR_getErrorCode(rc),
            ErrorCode::MaxSymbolValueTooLarge
        );
    }

    #[test]
    fn decompress_workspace_size_uses_build_dtable_u32_size() {
        assert_eq!(
            FSE_DECOMPRESS_WKSP_SIZE_U32(FSE_MIN_TABLELOG, FSE_MAX_SYMBOL_VALUE),
            301
        );
        assert_eq!(
            FSE_DECOMPRESS_WKSP_SIZE(FSE_MIN_TABLELOG, FSE_MAX_SYMBOL_VALUE),
            1204
        );

        assert_eq!(
            FSE_DECOMPRESS_WKSP_SIZE_U32(FSE_MAX_TABLELOG, FSE_MAX_SYMBOL_VALUE),
            5381
        );
        assert_eq!(
            FSE_DECOMPRESS_WKSP_SIZE(FSE_MAX_TABLELOG, FSE_MAX_SYMBOL_VALUE),
            21524
        );

        assert_eq!(
            FSE_DECOMPRESS_WKSP_SIZE_U32(FSE_TABLELOG_ABSOLUTE_MAX, FSE_MAX_SYMBOL_VALUE),
            41221
        );
        assert_eq!(
            FSE_DECOMPRESS_WKSP_SIZE(FSE_TABLELOG_ABSOLUTE_MAX, FSE_MAX_SYMBOL_VALUE),
            164884
        );
    }

    #[test]
    fn decompress_using_dtable_rejects_tiny_dst_without_underflow() {
        let mut dt = vec![0u32; 2];
        assert_eq!(FSE_buildDTable_rle(&mut dt, b'x'), 0);
        let src = [1u8];

        for cap in 0..2 {
            let mut dst = vec![0u8; cap];
            let rc = FSE_decompress_usingDTable(&mut dst, &src, &dt);
            assert_eq!(
                crate::common::error::ERR_getErrorCode(rc),
                ErrorCode::DstSizeTooSmall
            );
        }
    }

    #[test]
    fn decompress_wksp_rejects_workspace_after_ncount_but_before_heap_allocations() {
        use crate::compress::fse_compress::{
            FSE_NCountWriteBound, FSE_buildCTable_wksp, FSE_compressBound,
            FSE_compress_usingCTable, FSE_normalizeCount, FSE_optimalTableLog, FSE_writeNCount,
            FSE_BUILD_CTABLE_WORKSPACE_SIZE, FSE_CTABLE_SIZE_U32,
        };
        use crate::compress::hist::HIST_count_simple;

        let src: Vec<u8> = b"ababcabcabaaaaabcbcbcbaaaa".to_vec();
        let mut count = [0u32; 256];
        let mut maxSymbolValue = 255;
        assert!(HIST_count_simple(&mut count, &mut maxSymbolValue, &src) > 0);

        let mut norm = [0i16; 256];
        let tableLog = FSE_optimalTableLog(0, src.len(), maxSymbolValue);
        assert_eq!(
            FSE_normalizeCount(&mut norm, tableLog, &count, src.len(), maxSymbolValue, 0),
            tableLog as usize
        );

        let mut ncount = vec![0u8; FSE_NCountWriteBound(maxSymbolValue, tableLog)];
        let ncount_len = FSE_writeNCount(&mut ncount, &norm, maxSymbolValue, tableLog);
        assert!(!crate::common::error::ERR_isError(ncount_len));
        ncount.truncate(ncount_len);

        let mut ctable = vec![0u32; FSE_CTABLE_SIZE_U32(tableLog, maxSymbolValue)];
        let mut build_wksp = vec![0u8; FSE_BUILD_CTABLE_WORKSPACE_SIZE(maxSymbolValue, tableLog)];
        assert_eq!(
            FSE_buildCTable_wksp(
                &mut ctable,
                &norm,
                maxSymbolValue,
                tableLog,
                &mut build_wksp
            ),
            0
        );

        let mut payload = vec![0u8; FSE_compressBound(src.len())];
        let payload_len = FSE_compress_usingCTable(&mut payload, &src, &ctable);
        assert!(!crate::common::error::ERR_isError(payload_len));
        payload.truncate(payload_len);

        let mut stream = ncount;
        stream.extend_from_slice(&payload);

        let required = FSE_DECOMPRESS_WKSP_SIZE(tableLog, maxSymbolValue);
        assert!(required > 512);
        let mut workspace = vec![0u8; 512];
        let mut dst = vec![0u8; src.len()];

        let rc = FSE_decompress_wksp_bmi2(&mut dst, &stream, FSE_MAX_TABLELOG, &mut workspace, 0);
        assert_eq!(
            crate::common::error::ERR_getErrorCode(rc),
            ErrorCode::TableLogTooLarge
        );
    }
}
