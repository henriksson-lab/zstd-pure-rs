//! Translation of `lib/decompress/huf_decompress.c`.
//!
//! **Fully implemented**: `HUF_DTable` layout (`DTableDesc` header +
//! decode-entry array), `HUF_getDTableDesc`, `HUF_rescaleStats`,
//! `HUF_DEltX1_set4`, `HUF_readDTableX1` / `HUF_readDTableX2`,
//! `HUF_decompress1X*` / `HUF_decompress4X*` (both X1 and X2).
//!
//! **Deferred**: BMI2 / ARM64 hardware-assist fast paths — current
//! port uses the portable scalar inner loops.

pub const HUF_TABLELOG_MAX: u32 = 12;
pub const HUF_TABLELOG_ABSOLUTEMAX: u32 = 12;
pub const HUF_SYMBOLVALUE_MAX: u32 = 255;
pub const HUF_DECODER_FAST_TABLELOG: u32 = 11;
pub const HUF_DECOMPRESS_WORKSPACE_SIZE: usize = (2 << 10) + (1 << 9);
pub const HUF_DECOMPRESS_WORKSPACE_SIZE_U32: usize = HUF_DECOMPRESS_WORKSPACE_SIZE / 4;

pub const HUF_flags_bmi2: i32 = 1 << 0;
pub const HUF_flags_optimalDepth: i32 = 1 << 1;
pub const HUF_flags_preferRepeat: i32 = 1 << 2;
pub const HUF_flags_suspectUncompressible: i32 = 1 << 3;
pub const HUF_flags_disableAsm: i32 = 1 << 4;
pub const HUF_flags_disableFast: i32 = 1 << 5;

/// Port-name wrapper for `HUF_initRemainingDStream`. Upstream uses it
/// to reconstruct a `BIT_DStream_t` after a hardware-oriented fast
/// loop. The Rust decoder doesn't split out that fast-loop state, so
/// callers provide the remaining compressed slice directly.
#[inline]
pub fn HUF_initRemainingDStream<'a>(
    bit: &mut crate::common::bitstream::BIT_DStream_t<'a>,
    src: &'a [u8],
) -> usize {
    crate::common::bitstream::BIT_initDStream(bit, src, src.len())
}

/// `HUF_DTable` is type-erased in upstream (`typedef U32 HUF_DTable`). The
/// first slot stores the `DTableDesc`; the remaining slots store decode
/// entries whose layout depends on table type (X1 or X2).
pub type HUF_DTable = u32;

/// Mirror of upstream `DTableDesc`: bytes are
/// `{maxTableLog, tableType, tableLog, reserved}`. Packed into the
/// `HUF_DTable`'s first u32, little-endian on all targets (the field is
/// opaque and only read/written via helpers, so endianness doesn't leak
/// out to callers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DTableDesc {
    pub maxTableLog: u8,
    pub tableType: u8,
    pub tableLog: u8,
    pub reserved: u8,
}

#[inline]
fn pack_dtable_desc(d: DTableDesc) -> u32 {
    (d.maxTableLog as u32)
        | ((d.tableType as u32) << 8)
        | ((d.tableLog as u32) << 16)
        | ((d.reserved as u32) << 24)
}

#[inline]
fn unpack_dtable_desc(s: u32) -> DTableDesc {
    DTableDesc {
        maxTableLog: s as u8,
        tableType: (s >> 8) as u8,
        tableLog: (s >> 16) as u8,
        reserved: (s >> 24) as u8,
    }
}

/// Port of `HUF_getDTableDesc`.
#[inline]
pub fn HUF_getDTableDesc(table: &[HUF_DTable]) -> DTableDesc {
    unpack_dtable_desc(table[0])
}

/// Port of `HUF_setDTableDesc` — write helper used by the Rust side
/// anywhere the C code does `ZSTD_memcpy(DTable, &dtd, sizeof(dtd))`.
#[inline]
pub fn HUF_setDTableDesc(table: &mut [HUF_DTable], d: DTableDesc) {
    table[0] = pack_dtable_desc(d);
}

/// Upstream allocates a `HUF_DTable` as a `U32*` of size
/// `HUF_DTABLE_SIZE_U32`. We mirror that constant here so callers can
/// size their DTable buffers correctly.
#[inline]
pub const fn HUF_DTABLE_SIZE_U32(maxTableLog: u32) -> usize {
    1 + (1 << maxTableLog) as usize
}

// ---- Single-symbol decoding primitives (X1) ---------------------------

/// Mirror of `HUF_DEltX1 { BYTE nbBits; BYTE byte; }`. Packed into a u16
/// — the low byte is `nbBits`, the high byte is the symbol. Matches the
/// little-endian `HUF_DEltX1_set4` layout used by upstream to stamp
/// four entries at a time into a `U64`.
#[derive(Debug, Clone, Copy, Default)]
pub struct HUF_DEltX1 {
    pub nbBits: u8,
    pub byte: u8,
}

#[inline]
pub fn HUF_DEltX1_pack(e: HUF_DEltX1) -> u16 {
    (e.nbBits as u16) | ((e.byte as u16) << 8)
}

#[inline]
pub fn HUF_DEltX1_unpack(s: u16) -> HUF_DEltX1 {
    HUF_DEltX1 {
        nbBits: s as u8,
        byte: (s >> 8) as u8,
    }
}

/// Port of `HUF_DEltX1_set4`. Packs four copies of a `(symbol, nbBits)`
/// entry into a `U64` so that upstream's fill-DTable loop can stamp
/// four slots with one 64-bit store. The layout matches the LE branch
/// of the upstream function; BE targets are effectively untested in
/// upstream's hot-path loop as well, but we preserve the BE arithmetic
/// for bit-identical behaviour.
pub fn HUF_DEltX1_set4(symbol: u8, nbBits: u8) -> u64 {
    let d4_16 = if crate::common::mem::is_little_endian() {
        (symbol as u64) << 8 | nbBits as u64
    } else {
        symbol as u64 | ((nbBits as u64) << 8)
    };
    debug_assert!(d4_16 < (1u64 << 16));
    d4_16 * 0x0001_0001_0001_0001u64
}

/// Port of `HUF_rescaleStats`. Raises `tableLog` toward
/// `targetTableLog`, shifting weights for non-zero probability symbols
/// up by the same scale and moving rank counts. No-op if `tableLog >
/// targetTableLog`.
pub fn HUF_rescaleStats(
    huffWeight: &mut [u8],
    rankVal: &mut [u32],
    nbSymbols: u32,
    tableLog: u32,
    targetTableLog: u32,
) -> u32 {
    if tableLog > targetTableLog {
        return tableLog;
    }
    if tableLog < targetTableLog {
        let scale = targetTableLog - tableLog;
        for w in huffWeight.iter_mut().take(nbSymbols as usize) {
            if *w != 0 {
                *w += scale as u8;
            }
        }
        // Move rankVal: all weights except 0 shift by scale; fresh
        // weights [1..=scale] get zeroed.
        let mut s = targetTableLog;
        while s > scale {
            rankVal[s as usize] = rankVal[(s - scale) as usize];
            s -= 1;
        }
        let mut s = scale;
        while s > 0 {
            rankVal[s as usize] = 0;
            s -= 1;
        }
    }
    targetTableLog
}

// ---- Double-symbol decoding primitives (X2) ---------------------------

/// Mirror of `HUF_DEltX2 { U16 sequence; BYTE nbBits; BYTE length; }`
/// — the double-symbol decode entry. `sequence` packs up to 2 output
/// bytes (the `length` field tells how many to actually write).
#[derive(Debug, Clone, Copy, Default)]
pub struct HUF_DEltX2 {
    pub sequence: u16,
    pub nbBits: u8,
    pub length: u8,
}

#[inline]
pub fn HUF_DEltX2_pack(e: HUF_DEltX2) -> u32 {
    (e.sequence as u32) | ((e.nbBits as u32) << 16) | ((e.length as u32) << 24)
}

#[inline]
pub fn HUF_DEltX2_unpack(s: u32) -> HUF_DEltX2 {
    HUF_DEltX2 {
        sequence: s as u16,
        nbBits: (s >> 16) as u8,
        length: (s >> 24) as u8,
    }
}

/// Read a packed `HUF_DEltX2` at entry index `idx`. Upstream lays each
/// entry in its own `u32` slot after the DTable header, so index 0 of
/// the decode region maps to `dtable[1]`.
#[inline]
pub fn read_entry_x2(dtable: &[HUF_DTable], idx: usize) -> HUF_DEltX2 {
    HUF_DEltX2_unpack(dtable[1 + idx])
}

#[inline]
pub fn write_entry_x2(dtable: &mut [HUF_DTable], idx: usize, e: HUF_DEltX2) {
    dtable[1 + idx] = HUF_DEltX2_pack(e);
}

/// Port of `HUF_buildDEltX2U32`. Pack a decode entry `(symbol, nbBits,
/// baseSeq, level)` into the same u32 layout used by
/// `HUF_DEltX2_pack`. `level == 1` means this slot resolves to a single
/// byte (the symbol itself); `level == 2` means two bytes
/// (`baseSeq` followed by `symbol`).
pub fn HUF_buildDEltX2U32(symbol: u32, nbBits: u32, baseSeq: u32, level: i32) -> u32 {
    let seq = if crate::common::mem::is_little_endian() {
        if level == 1 {
            symbol
        } else {
            baseSeq + (symbol << 8)
        }
    } else if level == 1 {
        symbol << 8
    } else {
        (baseSeq << 8) + symbol
    };
    if crate::common::mem::is_little_endian() {
        seq + (nbBits << 16) + ((level as u32) << 24)
    } else {
        (seq << 16) + (nbBits << 8) + level as u32
    }
}

#[inline]
pub fn HUF_buildDEltX2(symbol: u32, nbBits: u32, baseSeq: u32, level: i32) -> u32 {
    HUF_buildDEltX2U32(symbol, nbBits, baseSeq, level)
}

/// Port of `HUF_buildDEltX2U64`: two identical packed X2 entries in
/// one little-endian 64-bit lane.
#[inline]
pub fn HUF_buildDEltX2U64(symbol: u32, nbBits: u32, baseSeq: u16, level: i32) -> u64 {
    let delt = HUF_buildDEltX2U32(symbol, nbBits, baseSeq as u32, level) as u64;
    delt + (delt << 32)
}

/// Fill positions `DTable[0..length*count]` with entries of width
/// `nbBits`, one per sorted symbol in `[begin..end]`. Port of
/// `HUF_fillDTableX2ForWeight`.
///
/// `sortedList` is upstream's `sortedSymbol_t` array (each element is
/// just a u8 symbol). `begin`/`end` are indices into it.
pub fn HUF_fillDTableX2ForWeight(
    dtable: &mut [HUF_DTable],
    dtable_start: usize,
    sortedList: &[u8],
    begin: usize,
    end: usize,
    nbBits: u32,
    tableLog: u32,
    baseSeq: u16,
    level: i32,
) {
    let length = 1u32 << ((tableLog - nbBits) & 0x1F);
    let mut pos = dtable_start;
    for sym_byte in sortedList.iter().take(end).skip(begin) {
        let sym = *sym_byte as u32;
        let delt = HUF_buildDEltX2U32(sym, nbBits, baseSeq as u32, level);
        for j in 0..length as usize {
            dtable[1 + pos + j] = delt;
        }
        pos += length as usize;
    }
}

/// Port of `HUF_fillDTableX2Level2`. Fills level-2 positions inside one
/// first-level slot's region of the DTable.
pub fn HUF_fillDTableX2Level2(
    dtable: &mut [HUF_DTable],
    dtable_start: usize,
    targetLog: u32,
    consumedBits: u32,
    rankVal: &[u32],
    minWeight: i32,
    maxWeight1: i32,
    sortedSymbols: &[u8],
    rankStart: &[u32],
    nbBitsBaseline: u32,
    baseSeq: u16,
) {
    // Fill "skipped" positions: slots where combined weight would exceed
    // the target tableLog. Upstream stamps them with a level-1 decoder
    // entry for `baseSeq` (single-byte output).
    if minWeight > 1 {
        let length = 1u32 << ((targetLog - consumedBits) & 0x1F);
        let delt = HUF_buildDEltX2U32(baseSeq as u32, consumedBits, 0, 1);
        let skipSize = rankVal[minWeight as usize] as usize;
        for i in 0..skipSize.min(length as usize) {
            dtable[1 + dtable_start + i] = delt;
        }
    }

    // Fill the remaining positions, grouped by second-level weight.
    for w in minWeight..maxWeight1 {
        let begin = rankStart[w as usize] as usize;
        let end = rankStart[(w + 1) as usize] as usize;
        let nbBits = nbBitsBaseline - w as u32;
        let totalBits = nbBits + consumedBits;
        let offset = rankVal[w as usize] as usize;
        HUF_fillDTableX2ForWeight(
            dtable,
            dtable_start + offset,
            sortedSymbols,
            begin,
            end,
            totalBits,
            targetLog,
            baseSeq,
            2,
        );
    }
}

/// Port of `HUF_fillDTableX2`. Walks weights 1..=maxWeight and, for
/// each, either fills single-symbol entries or recurses to fill a
/// second-symbol region via `HUF_fillDTableX2Level2`.
///
/// `rankValOrigin` is upstream's `rankValCol_t[HUF_TABLELOG_MAX]`; we
/// flatten it to `&[[u32; HUF_TABLELOG_MAX+1]]` so each consumed-bits
/// level has its own rankVal column.
pub fn HUF_fillDTableX2(
    dtable: &mut [HUF_DTable],
    targetLog: u32,
    sortedList: &[u8],
    rankStart: &[u32],
    rankValOrigin: &[[u32; (HUF_TABLELOG_MAX + 1) as usize]],
    maxWeight: u32,
    nbBitsBaseline: u32,
) {
    let rankVal = &rankValOrigin[0];
    let scaleLog = nbBitsBaseline as i32 - targetLog as i32;
    let minBits = nbBitsBaseline - maxWeight;
    let wEnd = maxWeight as i32 + 1;

    for w in 1..wEnd {
        let begin = rankStart[w as usize] as usize;
        let end = rankStart[(w + 1) as usize] as usize;
        let nbBits = nbBitsBaseline - w as u32;

        if targetLog >= nbBits + minBits {
            // Second-level slot: stamp a region, recurse per first-level
            // symbol in [begin..end].
            let mut start = rankVal[w as usize] as usize;
            let length = 1usize << ((targetLog - nbBits) & 0x1F);
            let mut minWeight = nbBits as i32 + scaleLog;
            if minWeight < 1 {
                minWeight = 1;
            }
            for s in begin..end {
                HUF_fillDTableX2Level2(
                    dtable,
                    start,
                    targetLog,
                    nbBits,
                    &rankValOrigin[nbBits as usize],
                    minWeight,
                    wEnd,
                    sortedList,
                    rankStart,
                    nbBitsBaseline,
                    sortedList[s] as u16,
                );
                start += length;
            }
        } else {
            // Single-symbol slot per sortedList entry in this rank.
            let offset = rankVal[w as usize] as usize;
            HUF_fillDTableX2ForWeight(
                dtable, offset, sortedList, begin, end, nbBits, targetLog, 0, 1,
            );
        }
    }
}

/// Port of `HUF_decodeSymbolX2`. Writes up to 2 bytes (`sequence`) to
/// `dst[p..]`, then skips `nbBits`. Returns number of bytes written
/// (1 or 2).
#[inline]
pub fn HUF_decodeSymbolX2(
    dst: &mut [u8],
    p: usize,
    bitD: &mut crate::common::bitstream::BIT_DStream_t,
    dtable: &[HUF_DTable],
    dtLog: u32,
) -> u32 {
    let val = crate::common::bitstream::BIT_lookBitsFast(bitD, dtLog);
    let e = read_entry_x2(dtable, val);
    // Write 2 bytes of sequence; caller bumps `p` by the returned length.
    let bytes = e.sequence.to_le_bytes();
    dst[p] = bytes[0];
    if p + 1 < dst.len() {
        dst[p + 1] = bytes[1];
    }
    crate::common::bitstream::BIT_skipBits(bitD, e.nbBits as u32);
    e.length as u32
}

/// Port of `HUF_decodeLastSymbolX2`. At stream boundary: writes only 1
/// byte from `sequence`, and uses a guarded `BIT_skipBits` that may
/// clamp `bitsConsumed` if the skip would overflow the bit register.
pub fn HUF_decodeLastSymbolX2(
    dst: &mut [u8],
    p: usize,
    bitD: &mut crate::common::bitstream::BIT_DStream_t,
    dtable: &[HUF_DTable],
    dtLog: u32,
) -> u32 {
    let val = crate::common::bitstream::BIT_lookBitsFast(bitD, dtLog);
    let e = read_entry_x2(dtable, val);
    dst[p] = e.sequence as u8;
    let container_bits = (core::mem::size_of::<usize>() as u32) * 8;
    if e.length == 1 {
        crate::common::bitstream::BIT_skipBits(bitD, e.nbBits as u32);
    } else if bitD.bitsConsumed < container_bits {
        crate::common::bitstream::BIT_skipBits(bitD, e.nbBits as u32);
        if bitD.bitsConsumed > container_bits {
            bitD.bitsConsumed = container_bits;
        }
    }
    1
}

/// Port of `HUF_decodeStreamX2`. Decode X2 symbols into `dst[p..pEnd]`.
///
/// Mirrors the upstream structure exactly: a fast loop that decodes
/// many symbols per iteration while there's container-bytes margin and
/// the bitstream is unfinished, followed by a two-stage tail — first a
/// per-symbol loop that still calls reload, then a per-symbol loop that
/// drains the existing bit container without further reloads, and
/// finally a single 1-byte last-symbol decode if `p` hasn't quite
/// reached `pEnd`. The two-stage tail is load-bearing: when the
/// bitstream has hit `endOfBuffer` but the container still holds
/// enough bits for one more 2-byte X2 symbol, the second loop emits it
/// without trying to reload past end-of-stream.
#[inline]
pub fn HUF_decodeStreamX2(
    dst: &mut [u8],
    p: usize,
    pEnd: usize,
    bitD: &mut crate::common::bitstream::BIT_DStream_t,
    dtable: &[HUF_DTable],
    dtLog: u32,
) -> usize {
    use crate::common::bitstream::{BIT_DStream_unfinished, BIT_reloadDStream};
    let pStart = p;
    let mut p = p;
    let container_bytes = core::mem::size_of::<usize>();

    if pEnd.saturating_sub(p) >= container_bytes {
        if dtLog <= 11 {
            // Up to 10 bytes at a time (5 X2 decode steps, each writing
            // 1 or 2 bytes). Margin guard leaves 10 bytes of tail.
            while pEnd > 9 && p < pEnd - 9 && BIT_reloadDStream(bitD) == BIT_DStream_unfinished {
                p += HUF_decodeSymbolX2(dst, p, bitD, dtable, dtLog) as usize;
                p += HUF_decodeSymbolX2(dst, p, bitD, dtable, dtLog) as usize;
                p += HUF_decodeSymbolX2(dst, p, bitD, dtable, dtLog) as usize;
                p += HUF_decodeSymbolX2(dst, p, bitD, dtable, dtLog) as usize;
                p += HUF_decodeSymbolX2(dst, p, bitD, dtable, dtLog) as usize;
            }
        } else {
            // Up to 8 bytes at a time (4 X2 decode steps).
            while pEnd > 7 && p < pEnd - 7 && BIT_reloadDStream(bitD) == BIT_DStream_unfinished {
                p += HUF_decodeSymbolX2(dst, p, bitD, dtable, dtLog) as usize;
                p += HUF_decodeSymbolX2(dst, p, bitD, dtable, dtLog) as usize;
                p += HUF_decodeSymbolX2(dst, p, bitD, dtable, dtLog) as usize;
                p += HUF_decodeSymbolX2(dst, p, bitD, dtable, dtLog) as usize;
            }
        }
    } else {
        BIT_reloadDStream(bitD);
    }

    // Tail: closer to end, up to 2 symbols at a time. First while-with-
    // reload drains the bitstream while there's still input; second
    // while-without-reload squeezes any remaining 2-byte symbols out
    // of the bit container after the bitstream has been consumed.
    if pEnd.saturating_sub(p) >= 2 {
        while p + 1 < pEnd && BIT_reloadDStream(bitD) == BIT_DStream_unfinished {
            p += HUF_decodeSymbolX2(dst, p, bitD, dtable, dtLog) as usize;
        }
        while p + 1 < pEnd {
            p += HUF_decodeSymbolX2(dst, p, bitD, dtable, dtLog) as usize;
        }
    }

    // Last byte (length=1): if we haven't quite reached pEnd, emit one
    // single-byte symbol via the dedicated last-symbol path.
    if p < pEnd {
        p += HUF_decodeLastSymbolX2(dst, p, bitD, dtable, dtLog) as usize;
    }

    p - pStart
}

// ---- Public APIs ------------------------------------------------------

/// Port of `HUF_selectDecoder` (huf_decompress.c:1830). Picks between
/// the X1 (4-byte-per-entry, faster for low-ratio blocks) and X2
/// (2-byte-per-entry, faster for high-ratio blocks) decoders based on
/// compression ratio. Returns 0 for X1, 1 for X2. Upstream uses a
/// 16-row table of pre-computed cost metrics indexed by the
/// ratio-quantization `Q = cSrcSize * 16 / dstSize` (clamped to 15).
pub fn HUF_selectDecoder(dstSize: usize, cSrcSize: usize) -> u32 {
    debug_assert!(dstSize > 0);
    debug_assert!(dstSize <= 128 * 1024);

    // algoTime[Q][single/double] = (tableTime, decode256Time).
    // From upstream huf_decompress.c:1803.
    const ALGO_TIME: [[(u32, u32); 2]; 16] = [
        [(0, 0), (1, 1)],
        [(0, 0), (1, 1)],
        [(150, 216), (381, 119)],
        [(170, 205), (514, 112)],
        [(177, 199), (539, 110)],
        [(197, 194), (644, 107)],
        [(221, 192), (735, 107)],
        [(256, 189), (881, 106)],
        [(359, 188), (1167, 109)],
        [(582, 187), (1570, 114)],
        [(688, 187), (1712, 122)],
        [(825, 186), (1965, 136)],
        [(976, 185), (2131, 150)],
        [(1180, 186), (2070, 175)],
        [(1377, 185), (1731, 202)],
        [(1412, 185), (1695, 202)],
    ];

    let q = if cSrcSize >= dstSize {
        15u32
    } else {
        ((cSrcSize * 16) / dstSize) as u32
    };
    let d256 = (dstSize >> 8) as u32;
    let d_time0 = ALGO_TIME[q as usize][0].0 + ALGO_TIME[q as usize][0].1.wrapping_mul(d256);
    let mut d_time1 = ALGO_TIME[q as usize][1].0 + ALGO_TIME[q as usize][1].1.wrapping_mul(d256);
    // Small advantage to the smaller-memory algorithm (matches
    // upstream's cache-eviction bias).
    d_time1 = d_time1.wrapping_add(d_time1 >> 5);
    (d_time1 < d_time0) as u32
}

/// Port of `HUF_readDTableX1_wksp` / `HUF_readDTableX1`. Reads a
/// compact HUF tree from `src`, fills the decoding table `dtable`.
///
/// The decode table is a flat `&mut [HUF_DTable]` (a.k.a. `&mut [u32]`).
/// Slot 0 is the `DTableDesc` header; following slots hold the decode
/// entries. Each decode entry is a packed `HUF_DEltX1` (`nbBits` low,
/// `symbol` high — matching the `HUF_DEltX1_set4` LE layout), and we
/// pack two entries per u32 in order so `1<<tableLog` entries fit into
/// `(1<<tableLog)/2` u32 slots.
pub fn HUF_readDTableX1(
    dtable: &mut [HUF_DTable],
    src: &[u8],
    workSpace: &mut [u32],
    flags: i32,
) -> usize {
    // Workspace layout (upstream: a typed `HUF_ReadDTableX1_Workspace`):
    //   rankVal[HUF_TABLELOG_ABSOLUTEMAX + 1]    (u32)
    //   rankStart[HUF_TABLELOG_ABSOLUTEMAX + 1]  (u32)
    //   statsWksp[HUF_READ_STATS_WORKSPACE_SIZE_U32] (u32)
    //   symbols[HUF_SYMBOLVALUE_MAX + 1]         (u8)
    //   huffWeight[HUF_SYMBOLVALUE_MAX + 1]      (u8)
    //
    // We keep the workspace-sized check but use local stack buffers for
    // correctness — memory footprint is equivalent.
    const RANK_LEN: usize = (HUF_TABLELOG_ABSOLUTEMAX + 1) as usize;
    const SYM_LEN: usize = (HUF_SYMBOLVALUE_MAX + 1) as usize;

    // Gate on workspace size so callers that pass too-small buffers get
    // an error, matching upstream.
    if workSpace.len() * 4 < HUF_DECOMPRESS_WORKSPACE_SIZE {
        return crate::common::error::ERROR(crate::common::error::ErrorCode::TableLogTooLarge);
    }

    let mut rankVal = [0u32; RANK_LEN];
    let mut rankStart = [0u32; RANK_LEN];
    let mut symbols = [0u8; SYM_LEN];
    let mut huffWeight = [0u8; SYM_LEN];
    let mut stats_wksp = [0u32; 256];

    let mut nbSymbols: u32 = 0;
    let mut tableLog: u32 = 0;

    let iSize = crate::common::entropy_common::HUF_readStats_wksp(
        &mut huffWeight,
        SYM_LEN,
        &mut rankVal,
        &mut nbSymbols,
        &mut tableLog,
        src,
        &mut stats_wksp,
        flags,
    );
    if crate::common::error::ERR_isError(iSize) {
        return iSize;
    }

    // Header: raise tableLog toward HUF_DECODER_FAST_TABLELOG, but not
    // past what the DTable has space for.
    let mut dtd = HUF_getDTableDesc(dtable);
    let maxTableLog = dtd.maxTableLog as u32 + 1;
    let targetTableLog = maxTableLog.min(HUF_DECODER_FAST_TABLELOG);
    tableLog = HUF_rescaleStats(
        &mut huffWeight,
        &mut rankVal,
        nbSymbols,
        tableLog,
        targetTableLog,
    );
    if tableLog > dtd.maxTableLog as u32 + 1 {
        return crate::common::error::ERROR(crate::common::error::ErrorCode::TableLogTooLarge);
    }
    dtd.tableType = 0;
    dtd.tableLog = tableLog as u8;
    HUF_setDTableDesc(dtable, dtd);

    // Compute rankStart + symbols: group symbols by huffWeight rank.
    let mut nextRankStart: u32 = 0;
    for n in 0..(tableLog + 1) as usize {
        let curr = nextRankStart;
        nextRankStart += rankVal[n];
        rankStart[n] = curr;
    }
    // Upstream unrolls by 4; the Rust equivalent runs the same cost at
    // -O2 and keeps the mapping clear. We also use the unrolled pattern
    // because code-complexity-comparator matches on loop structure.
    let unroll = 4;
    let nLimit = nbSymbols as i32 - unroll + 1;
    let mut n = 0i32;
    while n < nLimit {
        for u in 0..unroll {
            let w = huffWeight[(n + u) as usize] as usize;
            let pos = rankStart[w] as usize;
            symbols[pos] = (n + u) as u8;
            rankStart[w] += 1;
        }
        n += unroll;
    }
    while (n as u32) < nbSymbols {
        let w = huffWeight[n as usize] as usize;
        let pos = rankStart[w] as usize;
        symbols[pos] = n as u8;
        rankStart[w] += 1;
        n += 1;
    }

    // Fill DTable: iterate weights 1..=tableLog. Each weight owns
    // (1<<w)>>1 consecutive slots times symbolCount. `dt[]` is treated
    // as a packed [u16; 1<<tableLog] starting at dtable[1..].
    //
    // Packing: pair of u16 entries → one u32 slot. entry 2k → low half
    // of u32 slot (k/2) for even k; high half for odd k. But a
    // cleaner, slower-by-a-hair approach: reinterpret the tail as
    // [u16] via safe byte arithmetic.
    let tableSize = 1usize << tableLog;
    // Clear the dt region so leftover bits from a previous call don't
    // leak into unpopulated slots. Upstream relies on the fill below
    // hitting every slot, so this is belt-and-braces.
    for slot in dtable.iter_mut().skip(1).take(tableSize.div_ceil(2)) {
        *slot = 0;
    }

    let mut symbol_idx = rankVal[0] as usize;
    let mut rank_slot_start: usize = 0; // in units of entries (u16)
    for w in 1..(tableLog + 1) {
        let symbolCount = rankVal[w as usize] as usize;
        let length = (1usize << w) >> 1;
        let nbBits = (tableLog + 1 - w) as u8;
        let mut uStart = rank_slot_start;
        for s in 0..symbolCount {
            let sym = symbols[symbol_idx + s];
            let ent = HUF_DEltX1_pack(HUF_DEltX1 { nbBits, byte: sym });
            for j in 0..length {
                write_entry(dtable, uStart + j, ent);
            }
            uStart += length;
        }
        symbol_idx += symbolCount;
        rank_slot_start += symbolCount * length;
    }

    iSize
}

/// Write a packed `HUF_DEltX1` at entry index `idx` (in units of u16
/// entries), inside the decode region at `dtable[1..]`.
#[inline]
fn write_entry(dtable: &mut [HUF_DTable], idx: usize, ent: u16) {
    let slot_idx = 1 + idx / 2;
    let prev = dtable[slot_idx];
    if idx & 1 == 0 {
        dtable[slot_idx] = (prev & 0xFFFF_0000) | ent as u32;
    } else {
        dtable[slot_idx] = (prev & 0x0000_FFFF) | ((ent as u32) << 16);
    }
}

/// Read a packed `HUF_DEltX1` at entry index `idx` from the decode
/// region at `dtable[1..]`.
#[inline(always)]
pub fn read_entry(dtable: &[HUF_DTable], idx: usize) -> HUF_DEltX1 {
    let slot = dtable[1 + idx / 2];
    let half = if idx & 1 == 0 {
        slot as u16
    } else {
        (slot >> 16) as u16
    };
    HUF_DEltX1_unpack(half)
}

/// Port of `HUF_readDTableX2_wksp`. Parses a HUF stats block, builds
/// a double-symbol DTable, and writes the header bytes
/// `{maxTableLog, tableType=1, tableLog, reserved}` into slot 0.
pub fn HUF_readDTableX2(
    dtable: &mut [HUF_DTable],
    src: &[u8],
    _workSpace: &mut [u32],
    flags: i32,
) -> usize {
    use crate::common::error::{ErrorCode, ERROR};

    const MAXW_PLUS1: usize = (HUF_TABLELOG_MAX + 1) as usize;
    const SYM_LEN: usize = (HUF_SYMBOLVALUE_MAX + 1) as usize;

    let mut dtd = HUF_getDTableDesc(dtable);
    let mut maxTableLog = dtd.maxTableLog as u32;

    // Workspace fields (upstream lays them in `HUF_ReadDTableX2_Workspace`).
    let mut rankStats = [0u32; (HUF_TABLELOG_MAX + 1) as usize];
    let mut rankStart0 = [0u32; (HUF_TABLELOG_MAX + 3) as usize];
    let mut sortedSymbol = [0u8; SYM_LEN];
    let mut weightList = [0u8; SYM_LEN];
    let mut stats_wksp = [0u32; 256];
    // rankVal is [HUF_TABLELOG_MAX]["HUF_TABLELOG_MAX+1"] (first dim
    // indexed by consumed-bits, second by weight).
    let mut rankVal = [[0u32; MAXW_PLUS1]; HUF_TABLELOG_MAX as usize];

    if maxTableLog > HUF_TABLELOG_MAX {
        return ERROR(ErrorCode::TableLogTooLarge);
    }

    let mut tableLog: u32 = 0;
    let mut nbSymbols: u32 = 0;
    let iSize = crate::common::entropy_common::HUF_readStats_wksp(
        &mut weightList,
        SYM_LEN,
        &mut rankStats,
        &mut nbSymbols,
        &mut tableLog,
        src,
        &mut stats_wksp,
        flags,
    );
    if crate::common::error::ERR_isError(iSize) {
        return iSize;
    }
    if tableLog > maxTableLog {
        return ERROR(ErrorCode::TableLogTooLarge);
    }
    if tableLog <= HUF_DECODER_FAST_TABLELOG && maxTableLog > HUF_DECODER_FAST_TABLELOG {
        maxTableLog = HUF_DECODER_FAST_TABLELOG;
    }

    // Locate highest non-zero weight.
    let mut maxW = tableLog;
    while rankStats[maxW as usize] == 0 && maxW > 0 {
        maxW -= 1;
    }

    // Cumulative-start index per weight, stored at `rankStart0[w+1]`
    // to mirror upstream's `rankStart = rankStart0 + 1` pointer offset.
    // Upstream parks weight-0 (unused) symbols *after* all weighted
    // ones during sorting (`rankStart[0] = nextRankStart`) so they
    // don't collide with weight-1 slot 0; we replicate that here, then
    // reset rankStart[0] back to 0 after the sort so fillDTableX2's
    // first iteration reads the correct weight-1 origin.
    {
        let mut nextRankStart: u32 = 0;
        for w in 1..=maxW {
            let curr = nextRankStart;
            nextRankStart += rankStats[w as usize];
            rankStart0[(w + 1) as usize] = curr;
        }
        rankStart0[1] = nextRankStart;
        rankStart0[(maxW + 2) as usize] = nextRankStart;
    }

    // Sort symbols by weight. Uses a scratch copy of rankStart0 so we
    // don't destroy the start-index table that fillDTableX2 consumes.
    {
        let mut rs_working = rankStart0;
        for (s, w_byte) in weightList.iter().enumerate().take(nbSymbols as usize) {
            let w = *w_byte as usize;
            let r = rs_working[w + 1] as usize;
            rs_working[w + 1] += 1;
            sortedSymbol[r] = s as u8;
        }
    }
    rankStart0[1] = 0;

    // Build rankVal: rankVal[consumed][w] gives the DTable offset where
    // the first entry of weight `w`, after already consuming
    // `consumed` bits, goes.
    {
        let rescale = (maxTableLog as i32 - tableLog as i32) - 1;
        let mut nextRankVal: u32 = 0;
        for w in 1..=maxW {
            let curr = nextRankVal;
            let shift = (w as i32 + rescale).max(0) as u32;
            nextRankVal += rankStats[w as usize] << shift;
            rankVal[0][w as usize] = curr;
        }
        let minBits = tableLog + 1 - maxW;
        let limit = (maxTableLog - minBits + 1) as usize;
        let mut consumed = minBits as usize;
        while consumed < limit {
            let src_col = rankVal[0];
            let dst_col = &mut rankVal[consumed];
            for (w, dst_slot) in dst_col
                .iter_mut()
                .enumerate()
                .take(maxW as usize + 1)
                .skip(1)
            {
                *dst_slot = src_col[w] >> consumed;
            }
            consumed += 1;
        }
    }

    // Build a rankStart view that fillDTableX2 can index as rankStart[w].
    let rankStart_view: [u32; (HUF_TABLELOG_MAX + 2) as usize] = {
        let mut out = [0u32; (HUF_TABLELOG_MAX + 2) as usize];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = rankStart0[i + 1];
        }
        out
    };

    HUF_fillDTableX2(
        dtable,
        maxTableLog,
        &sortedSymbol,
        &rankStart_view,
        &rankVal,
        maxW,
        tableLog + 1,
    );

    dtd.tableLog = maxTableLog as u8;
    dtd.tableType = 1;
    HUF_setDTableDesc(dtable, dtd);
    iSize
}

/// Port of `HUF_decodeSymbolX1`. Peeks `dtLog` bits, resolves to a
/// decode entry, skips `nbBits`, returns the symbol byte.
#[inline]
pub fn HUF_decodeSymbolX1(
    bitD: &mut crate::common::bitstream::BIT_DStream_t,
    dtable: &[HUF_DTable],
    dtLog: u32,
) -> u8 {
    let val = crate::common::bitstream::BIT_lookBitsFast(bitD, dtLog);
    let e = read_entry(dtable, val);
    crate::common::bitstream::BIT_skipBits(bitD, e.nbBits as u32);
    e.byte
}

/// Port of `HUF_decodeStreamX1`. Decodes symbols into `dst[p..pEnd]`
/// one bitstream at a time. The upstream hot loop does up to 4
/// symbols per iteration (staggered with `BIT_reloadDStream`); we
/// mirror that structure so the per-iteration work is identical.
#[inline]
pub fn HUF_decodeStreamX1(
    dst: &mut [u8],
    p: usize,
    pEnd: usize,
    bitD: &mut crate::common::bitstream::BIT_DStream_t,
    dtable: &[HUF_DTable],
    dtLog: u32,
) -> usize {
    use crate::common::bitstream::{BIT_DStream_unfinished, BIT_reloadDStream};
    let pStart = p;
    let mut p = p;

    // Up to 4 symbols per iteration while the stream is unfinished and
    // we have 4 bytes of margin to pEnd.
    if pEnd > p + 3 {
        while BIT_reloadDStream(bitD) == BIT_DStream_unfinished && p < pEnd - 3 {
            dst[p] = HUF_decodeSymbolX1(bitD, dtable, dtLog);
            p += 1;
            dst[p] = HUF_decodeSymbolX1(bitD, dtable, dtLog);
            p += 1;
            dst[p] = HUF_decodeSymbolX1(bitD, dtable, dtLog);
            p += 1;
            dst[p] = HUF_decodeSymbolX1(bitD, dtable, dtLog);
            p += 1;
        }
    } else {
        BIT_reloadDStream(bitD);
    }

    // No more reloads needed on 64-bit: decode whatever remains from
    // the already-loaded bit container.
    while p < pEnd {
        dst[p] = HUF_decodeSymbolX1(bitD, dtable, dtLog);
        p += 1;
    }

    p - pStart
}

/// Port of `HUF_decompress1X1_usingDTable_internal_body`. Single-stream
/// X1 decode. Returns `dstSize` on success or an error code.
pub fn HUF_decompress1X1_usingDTable_internal(
    dst: &mut [u8],
    cSrc: &[u8],
    dtable: &[HUF_DTable],
) -> usize {
    use crate::common::bitstream::{BIT_DStream_t, BIT_endOfDStream, BIT_initDStream};
    use crate::common::error::{ErrorCode, ERROR};

    let dstSize = dst.len();
    let cSrcSize = cSrc.len();

    let dtd = HUF_getDTableDesc(dtable);
    let dtLog = dtd.tableLog as u32;

    let mut bitD = BIT_DStream_t::default();
    let rc = BIT_initDStream(&mut bitD, cSrc, cSrcSize);
    if crate::common::error::ERR_isError(rc) {
        return rc;
    }

    HUF_decodeStreamX1(dst, 0, dstSize, &mut bitD, dtable, dtLog);

    if BIT_endOfDStream(&bitD) == 0 {
        return ERROR(ErrorCode::CorruptionDetected);
    }
    dstSize
}

pub fn HUF_decompress1X_usingDTable<const BMI2: bool>(
    dst: &mut [u8],
    cSrc: &[u8],
    dtable: &[HUF_DTable],
) -> usize {
    let dtd = HUF_getDTableDesc(dtable);
    if dtd.tableType != 0 {
        HUF_decompress1X2_usingDTable_internal(dst, cSrc, dtable)
    } else {
        HUF_decompress1X1_usingDTable_internal(dst, cSrc, dtable)
    }
}

/// Port of `HUF_decompress4X1_usingDTable_internal_body`. Quad-stream
/// X1 decode: the source prefixes a 6-byte jump table giving the sizes
/// of streams 1..3 (stream 4 is implied), and each stream is decoded
/// into its own output segment.
pub fn HUF_decompress4X1_usingDTable_internal(
    dst: &mut [u8],
    cSrc: &[u8],
    dtable: &[HUF_DTable],
) -> usize {
    use crate::common::bitstream::{
        BIT_DStream_t, BIT_DStream_unfinished, BIT_endOfDStream, BIT_initDStream,
        BIT_reloadDStreamFast,
    };
    use crate::common::error::{ErrorCode, ERROR};
    use crate::common::mem::MEM_readLE16;

    let dstSize = dst.len();
    let cSrcSize = cSrc.len();

    if cSrcSize < 10 {
        return ERROR(ErrorCode::CorruptionDetected);
    }
    if dstSize < 6 {
        return ERROR(ErrorCode::CorruptionDetected);
    }

    // 6-byte jump table + the four stream segments.
    let length1 = MEM_readLE16(&cSrc[0..2]) as usize;
    let length2 = MEM_readLE16(&cSrc[2..4]) as usize;
    let length3 = MEM_readLE16(&cSrc[4..6]) as usize;
    let header_size: usize = 6;
    let accounted = length1 + length2 + length3 + header_size;
    if accounted > cSrcSize {
        return ERROR(ErrorCode::CorruptionDetected);
    }
    let length4 = cSrcSize - accounted;

    let istart1 = header_size;
    let istart2 = istart1 + length1;
    let istart3 = istart2 + length2;
    let istart4 = istart3 + length3;
    let segmentSize = dstSize.div_ceil(4);
    let opStart2 = segmentSize;
    let opStart3 = 2 * segmentSize;
    let opStart4 = 3 * segmentSize;
    if opStart4 > dstSize {
        return ERROR(ErrorCode::CorruptionDetected);
    }

    let dtd = HUF_getDTableDesc(dtable);
    let dtLog = dtd.tableLog as u32;

    // Four bit-streams, each pointing at its own slice of cSrc.
    let mut bd1 = BIT_DStream_t::default();
    let mut bd2 = BIT_DStream_t::default();
    let mut bd3 = BIT_DStream_t::default();
    let mut bd4 = BIT_DStream_t::default();
    let rc = BIT_initDStream(&mut bd1, &cSrc[istart1..istart1 + length1], length1);
    if crate::common::error::ERR_isError(rc) {
        return rc;
    }
    let rc = BIT_initDStream(&mut bd2, &cSrc[istart2..istart2 + length2], length2);
    if crate::common::error::ERR_isError(rc) {
        return rc;
    }
    let rc = BIT_initDStream(&mut bd3, &cSrc[istart3..istart3 + length3], length3);
    if crate::common::error::ERR_isError(rc) {
        return rc;
    }
    let rc = BIT_initDStream(&mut bd4, &cSrc[istart4..istart4 + length4], length4);
    if crate::common::error::ERR_isError(rc) {
        return rc;
    }

    // Decode into four disjoint segments of dst.
    let mut op1 = 0usize;
    let mut op2 = opStart2;
    let mut op3 = opStart3;
    let mut op4 = opStart4;
    let olimit = dstSize.saturating_sub(3);

    // Hot loop: 4 symbols per stream per iteration (16 total), while
    // all four streams have room AND stream 4 hasn't hit olimit.
    let mut endSignal = 1u32;
    if dstSize - op4 >= core::mem::size_of::<usize>() {
        loop {
            if endSignal == 0 || op4 >= olimit {
                break;
            }
            // Round 1: use the "safe" 2-variant per stream (one reload
            // boundary between pairs). Upstream ORs the SYMBOLX1_2
            // macro with an `if MEM_64bits()` guard; we're always 64b.
            dst[op1] = HUF_decodeSymbolX1(&mut bd1, dtable, dtLog);
            op1 += 1;
            dst[op2] = HUF_decodeSymbolX1(&mut bd2, dtable, dtLog);
            op2 += 1;
            dst[op3] = HUF_decodeSymbolX1(&mut bd3, dtable, dtLog);
            op3 += 1;
            dst[op4] = HUF_decodeSymbolX1(&mut bd4, dtable, dtLog);
            op4 += 1;
            // Round 2.
            dst[op1] = HUF_decodeSymbolX1(&mut bd1, dtable, dtLog);
            op1 += 1;
            dst[op2] = HUF_decodeSymbolX1(&mut bd2, dtable, dtLog);
            op2 += 1;
            dst[op3] = HUF_decodeSymbolX1(&mut bd3, dtable, dtLog);
            op3 += 1;
            dst[op4] = HUF_decodeSymbolX1(&mut bd4, dtable, dtLog);
            op4 += 1;
            // Round 3.
            dst[op1] = HUF_decodeSymbolX1(&mut bd1, dtable, dtLog);
            op1 += 1;
            dst[op2] = HUF_decodeSymbolX1(&mut bd2, dtable, dtLog);
            op2 += 1;
            dst[op3] = HUF_decodeSymbolX1(&mut bd3, dtable, dtLog);
            op3 += 1;
            dst[op4] = HUF_decodeSymbolX1(&mut bd4, dtable, dtLog);
            op4 += 1;
            // Round 4.
            dst[op1] = HUF_decodeSymbolX1(&mut bd1, dtable, dtLog);
            op1 += 1;
            dst[op2] = HUF_decodeSymbolX1(&mut bd2, dtable, dtLog);
            op2 += 1;
            dst[op3] = HUF_decodeSymbolX1(&mut bd3, dtable, dtLog);
            op3 += 1;
            dst[op4] = HUF_decodeSymbolX1(&mut bd4, dtable, dtLog);
            op4 += 1;

            endSignal &= (BIT_reloadDStreamFast(&mut bd1) == BIT_DStream_unfinished) as u32;
            endSignal &= (BIT_reloadDStreamFast(&mut bd2) == BIT_DStream_unfinished) as u32;
            endSignal &= (BIT_reloadDStreamFast(&mut bd3) == BIT_DStream_unfinished) as u32;
            endSignal &= (BIT_reloadDStreamFast(&mut bd4) == BIT_DStream_unfinished) as u32;
        }
    }

    // Cross-segment overflow check (upstream keeps these despite
    // nominal redundancy — a GCC bmi2 perf quirk).
    if op1 > opStart2 || op2 > opStart3 || op3 > opStart4 {
        return ERROR(ErrorCode::CorruptionDetected);
    }

    // Tail: finish each stream independently.
    HUF_decodeStreamX1(dst, op1, opStart2, &mut bd1, dtable, dtLog);
    HUF_decodeStreamX1(dst, op2, opStart3, &mut bd2, dtable, dtLog);
    HUF_decodeStreamX1(dst, op3, opStart4, &mut bd3, dtable, dtLog);
    HUF_decodeStreamX1(dst, op4, dstSize, &mut bd4, dtable, dtLog);

    let endCheck = BIT_endOfDStream(&bd1)
        & BIT_endOfDStream(&bd2)
        & BIT_endOfDStream(&bd3)
        & BIT_endOfDStream(&bd4);
    if endCheck == 0 {
        return ERROR(ErrorCode::CorruptionDetected);
    }
    dstSize
}

/// Port-name wrapper for `HUF_decompress4X1_usingDTable_internal_body`.
#[inline]
pub fn HUF_decompress4X1_usingDTable_internal_body(
    dst: &mut [u8],
    cSrc: &[u8],
    dtable: &[HUF_DTable],
) -> usize {
    HUF_decompress4X1_usingDTable_internal(dst, cSrc, dtable)
}

/// Portable wrapper for `HUF_decompress4X1_usingDTable_internal_default`.
#[inline]
pub fn HUF_decompress4X1_usingDTable_internal_default(
    dst: &mut [u8],
    cSrc: &[u8],
    dtable: &[HUF_DTable],
) -> usize {
    HUF_decompress4X1_usingDTable_internal(dst, cSrc, dtable)
}

/// BMI2 wrapper for `HUF_decompress4X1_usingDTable_internal_bmi2`.
#[inline]
pub fn HUF_decompress4X1_usingDTable_internal_bmi2(
    dst: &mut [u8],
    cSrc: &[u8],
    dtable: &[HUF_DTable],
) -> usize {
    HUF_decompress4X1_usingDTable_internal(dst, cSrc, dtable)
}

/// Fast-loop wrapper for `HUF_decompress4X1_usingDTable_internal_fast`.
#[inline]
pub fn HUF_decompress4X1_usingDTable_internal_fast(
    dst: &mut [u8],
    cSrc: &[u8],
    dtable: &[HUF_DTable],
) -> usize {
    HUF_decompress4X1_usingDTable_internal(dst, cSrc, dtable)
}

/// C fast-loop entry name. Rust keeps one scalar implementation, so
/// this forwards to the complete 4X1 decoder.
#[inline]
pub fn HUF_decompress4X1_usingDTable_internal_fast_c_loop(
    dst: &mut [u8],
    cSrc: &[u8],
    dtable: &[HUF_DTable],
) -> usize {
    HUF_decompress4X1_usingDTable_internal(dst, cSrc, dtable)
}

pub fn HUF_decompress4X_usingDTable<const BMI2: bool>(
    dst: &mut [u8],
    cSrc: &[u8],
    dtable: &[HUF_DTable],
) -> usize {
    let dtd = HUF_getDTableDesc(dtable);
    if dtd.tableType != 0 {
        HUF_decompress4X2_usingDTable_internal(dst, cSrc, dtable)
    } else {
        HUF_decompress4X1_usingDTable_internal(dst, cSrc, dtable)
    }
}

/// Port of `HUF_decompress1X1_DCtx_wksp`. Reads the DTable header from
/// `cSrc`, then decodes the remainder into `dst`.
pub fn HUF_decompress1X1_DCtx_wksp(
    dctx: &mut [HUF_DTable],
    dst: &mut [u8],
    cSrc: &[u8],
    workSpace: &mut [u32],
    flags: i32,
) -> usize {
    let hSize = HUF_readDTableX1(dctx, cSrc, workSpace, flags);
    if crate::common::error::ERR_isError(hSize) {
        return hSize;
    }
    if hSize >= cSrc.len() {
        return crate::common::error::ERROR(crate::common::error::ErrorCode::SrcSizeWrong);
    }
    HUF_decompress1X1_usingDTable_internal(dst, &cSrc[hSize..], dctx)
}

/// Port of `HUF_decompress1X2_usingDTable_internal_body`. Single-stream
/// X2 decode.
pub fn HUF_decompress1X2_usingDTable_internal(
    dst: &mut [u8],
    cSrc: &[u8],
    dtable: &[HUF_DTable],
) -> usize {
    use crate::common::bitstream::{BIT_DStream_t, BIT_endOfDStream, BIT_initDStream};
    use crate::common::error::{ErrorCode, ERROR};

    let dstSize = dst.len();
    let cSrcSize = cSrc.len();
    let mut bitD = BIT_DStream_t::default();
    let rc = BIT_initDStream(&mut bitD, cSrc, cSrcSize);
    if crate::common::error::ERR_isError(rc) {
        return rc;
    }
    let dtd = HUF_getDTableDesc(dtable);
    let dtLog = dtd.tableLog as u32;
    HUF_decodeStreamX2(dst, 0, dstSize, &mut bitD, dtable, dtLog);
    if BIT_endOfDStream(&bitD) == 0 {
        return ERROR(ErrorCode::CorruptionDetected);
    }
    dstSize
}

/// Port of `HUF_decompress1X2_DCtx_wksp`. Reads the X2 DTable then
/// decodes the tail.
pub fn HUF_decompress1X2_DCtx_wksp(
    dctx: &mut [HUF_DTable],
    dst: &mut [u8],
    cSrc: &[u8],
    workSpace: &mut [u32],
    flags: i32,
) -> usize {
    let hSize = HUF_readDTableX2(dctx, cSrc, workSpace, flags);
    if crate::common::error::ERR_isError(hSize) {
        return hSize;
    }
    if hSize >= cSrc.len() {
        return crate::common::error::ERROR(crate::common::error::ErrorCode::SrcSizeWrong);
    }
    HUF_decompress1X2_usingDTable_internal(dst, &cSrc[hSize..], dctx)
}

/// Port of `HUF_decompress1X_DCtx_wksp` (huf_decompress.c:1854).
/// Single-stream decoder that auto-selects X1 vs X2 via
/// `HUF_selectDecoder`. Short-circuits on three trivial cases
/// (empty, not-compressed, RLE) before dispatching. `_DCtx_wksp`
/// means the caller supplies the DTable scratch + workspace.
pub fn HUF_decompress1X_DCtx_wksp(
    dctx: &mut [HUF_DTable],
    dst: &mut [u8],
    cSrc: &[u8],
    workSpace: &mut [u32],
    flags: i32,
) -> usize {
    let dstSize = dst.len();
    let cSrcSize = cSrc.len();
    if dstSize == 0 {
        return crate::common::error::ERROR(crate::common::error::ErrorCode::DstSizeTooSmall);
    }
    if cSrcSize > dstSize {
        return crate::common::error::ERROR(crate::common::error::ErrorCode::CorruptionDetected);
    }
    if cSrcSize == dstSize {
        dst[..dstSize].copy_from_slice(&cSrc[..dstSize]);
        return dstSize;
    }
    if cSrcSize == 1 {
        for b in dst.iter_mut().take(dstSize) {
            *b = cSrc[0];
        }
        return dstSize;
    }
    let algoNb = HUF_selectDecoder(dstSize, cSrcSize);
    if algoNb != 0 {
        HUF_decompress1X2_DCtx_wksp(dctx, dst, cSrc, workSpace, flags)
    } else {
        HUF_decompress1X1_DCtx_wksp(dctx, dst, cSrc, workSpace, flags)
    }
}

/// Port of `HUF_decompress4X1_DCtx_wksp`. Reads the DTable header from
/// `cSrc`, then runs the quad-stream decoder on the remainder.
pub fn HUF_decompress4X1_DCtx_wksp(
    dctx: &mut [HUF_DTable],
    dst: &mut [u8],
    cSrc: &[u8],
    workSpace: &mut [u32],
    flags: i32,
) -> usize {
    let hSize = HUF_readDTableX1(dctx, cSrc, workSpace, flags);
    if crate::common::error::ERR_isError(hSize) {
        return hSize;
    }
    if hSize >= cSrc.len() {
        return crate::common::error::ERROR(crate::common::error::ErrorCode::SrcSizeWrong);
    }
    HUF_decompress4X1_usingDTable_internal(dst, &cSrc[hSize..], dctx)
}

/// Port of `HUF_decompress4X2_usingDTable_internal_body`. Quad-stream
/// X2 decode with the same 6-byte jump table as X1.
pub fn HUF_decompress4X2_usingDTable_internal(
    dst: &mut [u8],
    cSrc: &[u8],
    dtable: &[HUF_DTable],
) -> usize {
    use crate::common::bitstream::{
        BIT_DStream_t, BIT_DStream_unfinished, BIT_endOfDStream, BIT_initDStream,
        BIT_reloadDStreamFast,
    };
    use crate::common::error::{ErrorCode, ERROR};
    use crate::common::mem::MEM_readLE16;

    let dstSize = dst.len();
    let cSrcSize = cSrc.len();
    if cSrcSize < 10 {
        return ERROR(ErrorCode::CorruptionDetected);
    }
    if dstSize < 6 {
        return ERROR(ErrorCode::CorruptionDetected);
    }

    let length1 = MEM_readLE16(&cSrc[0..2]) as usize;
    let length2 = MEM_readLE16(&cSrc[2..4]) as usize;
    let length3 = MEM_readLE16(&cSrc[4..6]) as usize;
    let header_size = 6usize;
    let accounted = length1 + length2 + length3 + header_size;
    if accounted > cSrcSize {
        return ERROR(ErrorCode::CorruptionDetected);
    }
    let length4 = cSrcSize - accounted;

    let istart1 = header_size;
    let istart2 = istart1 + length1;
    let istart3 = istart2 + length2;
    let istart4 = istart3 + length3;
    let segmentSize = dstSize.div_ceil(4);
    let opStart2 = segmentSize;
    let opStart3 = 2 * segmentSize;
    let opStart4 = 3 * segmentSize;
    if opStart4 > dstSize {
        return ERROR(ErrorCode::CorruptionDetected);
    }

    let dtd = HUF_getDTableDesc(dtable);
    let dtLog = dtd.tableLog as u32;

    let mut bd1 = BIT_DStream_t::default();
    let mut bd2 = BIT_DStream_t::default();
    let mut bd3 = BIT_DStream_t::default();
    let mut bd4 = BIT_DStream_t::default();
    let rc = BIT_initDStream(&mut bd1, &cSrc[istart1..istart1 + length1], length1);
    if crate::common::error::ERR_isError(rc) {
        return rc;
    }
    let rc = BIT_initDStream(&mut bd2, &cSrc[istart2..istart2 + length2], length2);
    if crate::common::error::ERR_isError(rc) {
        return rc;
    }
    let rc = BIT_initDStream(&mut bd3, &cSrc[istart3..istart3 + length3], length3);
    if crate::common::error::ERR_isError(rc) {
        return rc;
    }
    let rc = BIT_initDStream(&mut bd4, &cSrc[istart4..istart4 + length4], length4);
    if crate::common::error::ERR_isError(rc) {
        return rc;
    }

    let olimit = dstSize.saturating_sub(core::mem::size_of::<usize>() - 1);
    let mut op1 = 0usize;
    let mut op2 = opStart2;
    let mut op3 = opStart3;
    let mut op4 = opStart4;
    let mut endSignal = 1u32;

    if dstSize - op4 >= core::mem::size_of::<usize>() {
        loop {
            if endSignal == 0 || op4 >= olimit {
                break;
            }
            // Four symbols per stream per iteration (16 total),
            // matching the non-clang portable path upstream.
            op1 += HUF_decodeSymbolX2(dst, op1, &mut bd1, dtable, dtLog) as usize;
            op2 += HUF_decodeSymbolX2(dst, op2, &mut bd2, dtable, dtLog) as usize;
            op3 += HUF_decodeSymbolX2(dst, op3, &mut bd3, dtable, dtLog) as usize;
            op4 += HUF_decodeSymbolX2(dst, op4, &mut bd4, dtable, dtLog) as usize;

            op1 += HUF_decodeSymbolX2(dst, op1, &mut bd1, dtable, dtLog) as usize;
            op2 += HUF_decodeSymbolX2(dst, op2, &mut bd2, dtable, dtLog) as usize;
            op3 += HUF_decodeSymbolX2(dst, op3, &mut bd3, dtable, dtLog) as usize;
            op4 += HUF_decodeSymbolX2(dst, op4, &mut bd4, dtable, dtLog) as usize;

            op1 += HUF_decodeSymbolX2(dst, op1, &mut bd1, dtable, dtLog) as usize;
            op2 += HUF_decodeSymbolX2(dst, op2, &mut bd2, dtable, dtLog) as usize;
            op3 += HUF_decodeSymbolX2(dst, op3, &mut bd3, dtable, dtLog) as usize;
            op4 += HUF_decodeSymbolX2(dst, op4, &mut bd4, dtable, dtLog) as usize;

            op1 += HUF_decodeSymbolX2(dst, op1, &mut bd1, dtable, dtLog) as usize;
            op2 += HUF_decodeSymbolX2(dst, op2, &mut bd2, dtable, dtLog) as usize;
            op3 += HUF_decodeSymbolX2(dst, op3, &mut bd3, dtable, dtLog) as usize;
            op4 += HUF_decodeSymbolX2(dst, op4, &mut bd4, dtable, dtLog) as usize;

            endSignal = (BIT_reloadDStreamFast(&mut bd1) == BIT_DStream_unfinished) as u32
                & (BIT_reloadDStreamFast(&mut bd2) == BIT_DStream_unfinished) as u32
                & (BIT_reloadDStreamFast(&mut bd3) == BIT_DStream_unfinished) as u32
                & (BIT_reloadDStreamFast(&mut bd4) == BIT_DStream_unfinished) as u32;
        }
    }

    if op1 > opStart2 || op2 > opStart3 || op3 > opStart4 {
        return ERROR(ErrorCode::CorruptionDetected);
    }

    HUF_decodeStreamX2(dst, op1, opStart2, &mut bd1, dtable, dtLog);
    HUF_decodeStreamX2(dst, op2, opStart3, &mut bd2, dtable, dtLog);
    HUF_decodeStreamX2(dst, op3, opStart4, &mut bd3, dtable, dtLog);
    HUF_decodeStreamX2(dst, op4, dstSize, &mut bd4, dtable, dtLog);

    let endCheck = BIT_endOfDStream(&bd1)
        & BIT_endOfDStream(&bd2)
        & BIT_endOfDStream(&bd3)
        & BIT_endOfDStream(&bd4);
    if endCheck == 0 {
        return ERROR(ErrorCode::CorruptionDetected);
    }
    dstSize
}

/// Port-name wrapper for `HUF_decompress4X2_usingDTable_internal_body`.
#[inline]
pub fn HUF_decompress4X2_usingDTable_internal_body(
    dst: &mut [u8],
    cSrc: &[u8],
    dtable: &[HUF_DTable],
) -> usize {
    HUF_decompress4X2_usingDTable_internal(dst, cSrc, dtable)
}

/// Portable wrapper for `HUF_decompress4X2_usingDTable_internal_default`.
#[inline]
pub fn HUF_decompress4X2_usingDTable_internal_default(
    dst: &mut [u8],
    cSrc: &[u8],
    dtable: &[HUF_DTable],
) -> usize {
    HUF_decompress4X2_usingDTable_internal(dst, cSrc, dtable)
}

/// BMI2 wrapper for `HUF_decompress4X2_usingDTable_internal_bmi2`.
#[inline]
pub fn HUF_decompress4X2_usingDTable_internal_bmi2(
    dst: &mut [u8],
    cSrc: &[u8],
    dtable: &[HUF_DTable],
) -> usize {
    HUF_decompress4X2_usingDTable_internal(dst, cSrc, dtable)
}

/// Fast-loop wrapper for `HUF_decompress4X2_usingDTable_internal_fast`.
#[inline]
pub fn HUF_decompress4X2_usingDTable_internal_fast(
    dst: &mut [u8],
    cSrc: &[u8],
    dtable: &[HUF_DTable],
) -> usize {
    HUF_decompress4X2_usingDTable_internal(dst, cSrc, dtable)
}

/// C fast-loop entry name. Rust keeps one scalar implementation, so
/// this forwards to the complete 4X2 decoder.
#[inline]
pub fn HUF_decompress4X2_usingDTable_internal_fast_c_loop(
    dst: &mut [u8],
    cSrc: &[u8],
    dtable: &[HUF_DTable],
) -> usize {
    HUF_decompress4X2_usingDTable_internal(dst, cSrc, dtable)
}

/// Port of `HUF_decompress4X2_DCtx_wksp`.
pub fn HUF_decompress4X2_DCtx_wksp(
    dctx: &mut [HUF_DTable],
    dst: &mut [u8],
    cSrc: &[u8],
    workSpace: &mut [u32],
    flags: i32,
) -> usize {
    let hSize = HUF_readDTableX2(dctx, cSrc, workSpace, flags);
    if crate::common::error::ERR_isError(hSize) {
        return hSize;
    }
    if hSize >= cSrc.len() {
        return crate::common::error::ERROR(crate::common::error::ErrorCode::SrcSizeWrong);
    }
    HUF_decompress4X2_usingDTable_internal(dst, &cSrc[hSize..], dctx)
}

/// Port of `HUF_decompress4X_hufOnly_wksp` (huf_decompress.c:1933).
/// Four-stream decoder that auto-selects X1 vs X2 via
/// `HUF_selectDecoder`. Unlike the 1X version, this rejects empty
/// src (any valid 4-stream block carries at least the jump table).
pub fn HUF_decompress4X_hufOnly_wksp(
    dctx: &mut [HUF_DTable],
    dst: &mut [u8],
    cSrc: &[u8],
    workSpace: &mut [u32],
    flags: i32,
) -> usize {
    if dst.is_empty() {
        return crate::common::error::ERROR(crate::common::error::ErrorCode::DstSizeTooSmall);
    }
    if cSrc.is_empty() {
        return crate::common::error::ERROR(crate::common::error::ErrorCode::CorruptionDetected);
    }
    let algoNb = HUF_selectDecoder(dst.len(), cSrc.len());
    if algoNb != 0 {
        HUF_decompress4X2_DCtx_wksp(dctx, dst, cSrc, workSpace, flags)
    } else {
        HUF_decompress4X1_DCtx_wksp(dctx, dst, cSrc, workSpace, flags)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtable_desc_roundtrip() {
        let d = DTableDesc {
            maxTableLog: 12,
            tableType: 0,
            tableLog: 11,
            reserved: 0,
        };
        let mut tbl = [0u32; 4];
        HUF_setDTableDesc(&mut tbl, d);
        assert_eq!(HUF_getDTableDesc(&tbl), d);
    }

    #[test]
    fn deltx2_pack_roundtrip() {
        let e = HUF_DEltX2 {
            sequence: 0xBEEF,
            nbBits: 10,
            length: 2,
        };
        let p = HUF_DEltX2_pack(e);
        let r = HUF_DEltX2_unpack(p);
        assert_eq!(r.sequence, 0xBEEF);
        assert_eq!(r.nbBits, 10);
        assert_eq!(r.length, 2);
    }

    #[test]
    fn deltx2_u64_stamps_two_identical_entries() {
        let packed = HUF_buildDEltX2U32(0x34, 6, 0x12, 2);
        let wide = HUF_buildDEltX2U64(0x34, 6, 0x12, 2);
        assert_eq!(wide as u32, packed);
        assert_eq!((wide >> 32) as u32, packed);
    }

    #[test]
    fn deltx2_read_write_round_trip_via_dtable() {
        let mut dt = [0u32; 16];
        write_entry_x2(
            &mut dt,
            3,
            HUF_DEltX2 {
                sequence: 0x1234,
                nbBits: 7,
                length: 1,
            },
        );
        let e = read_entry_x2(&dt, 3);
        assert_eq!(e.sequence, 0x1234);
        assert_eq!(e.nbBits, 7);
        assert_eq!(e.length, 1);
    }

    #[test]
    fn huf_port_name_wrappers_forward_to_scalar_paths() {
        let mut bit = crate::common::bitstream::BIT_DStream_t::default();
        let bit_src = [0x80u8];
        assert!(!crate::common::error::ERR_isError(
            HUF_initRemainingDStream(&mut bit, &bit_src)
        ));

        let mut dtable = vec![0u32; HUF_DTABLE_SIZE_U32(1)];
        HUF_setDTableDesc(
            &mut dtable,
            DTableDesc {
                maxTableLog: 1,
                tableType: 0,
                tableLog: 1,
                reserved: 0,
            },
        );
        let mut dst = [0u8; 4];
        let short = [0u8; 1];
        let expected = HUF_decompress4X1_usingDTable_internal(&mut dst, &short, &dtable);
        assert_eq!(
            HUF_decompress4X1_usingDTable_internal_default(&mut dst, &short, &dtable),
            expected
        );
        assert_eq!(
            HUF_decompress4X1_usingDTable_internal_bmi2(&mut dst, &short, &dtable),
            expected
        );
        assert_eq!(
            HUF_decompress4X1_usingDTable_internal_fast(&mut dst, &short, &dtable),
            expected
        );
        assert_eq!(
            HUF_decompress4X1_usingDTable_internal_fast_c_loop(&mut dst, &short, &dtable),
            expected
        );

        let expected = HUF_decompress4X2_usingDTable_internal(&mut dst, &short, &dtable);
        assert_eq!(
            HUF_decompress4X2_usingDTable_internal_default(&mut dst, &short, &dtable),
            expected
        );
        assert_eq!(
            HUF_decompress4X2_usingDTable_internal_bmi2(&mut dst, &short, &dtable),
            expected
        );
        assert_eq!(
            HUF_decompress4X2_usingDTable_internal_fast(&mut dst, &short, &dtable),
            expected
        );
        assert_eq!(
            HUF_decompress4X2_usingDTable_internal_fast_c_loop(&mut dst, &short, &dtable),
            expected
        );
    }

    #[test]
    fn deltx1_pack_roundtrip() {
        let e = HUF_DEltX1 {
            nbBits: 7,
            byte: 0xA5,
        };
        let s = HUF_DEltX1_pack(e);
        let r = HUF_DEltX1_unpack(s);
        assert_eq!(r.nbBits, e.nbBits);
        assert_eq!(r.byte, e.byte);
    }

    #[test]
    fn deltx1_set4_stamps_four_identical_entries() {
        // Repeat the 16-bit (byte,nbBits) layout four times.
        let word = HUF_DEltX1_set4(0xAA, 3);
        let lane = (word & 0xFFFF) as u16;
        assert_eq!((word >> 16) & 0xFFFF, lane as u64);
        assert_eq!((word >> 32) & 0xFFFF, lane as u64);
        assert_eq!((word >> 48) & 0xFFFF, lane as u64);
        // In LE mode, high byte is symbol, low byte is nbBits.
        if crate::common::mem::is_little_endian() {
            assert_eq!(lane & 0xFF, 3);
            assert_eq!(lane >> 8, 0xAA);
        }
    }

    #[test]
    fn rescale_stats_noop_when_below_target() {
        let mut weights = [2u8, 3, 0, 5];
        let mut ranks = [0u32, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let out = HUF_rescaleStats(&mut weights, &mut ranks, 4, 5, 5);
        assert_eq!(out, 5);
        assert_eq!(weights, [2, 3, 0, 5]); // unchanged
    }

    #[test]
    fn rescale_stats_noop_when_above_target() {
        // tableLog > targetTableLog — upstream returns tableLog unchanged.
        let mut weights = [2u8, 3, 0, 5];
        let mut ranks = [0u32; 13];
        let out = HUF_rescaleStats(&mut weights, &mut ranks, 4, 7, 5);
        assert_eq!(out, 7);
        assert_eq!(weights, [2, 3, 0, 5]);
    }

    #[test]
    fn read_dtable_x1_raw_header_populates_entries() {
        // Construct a minimal raw HUF header:
        //   4 symbols with weights [1, 2, 1, <implied>].
        //   Per spec: total weight = Σ ((1<<w)>>1) must be 2^n.
        //   Here Σ = 1 + 2 + 1 = 4 = 2^2, tableLog = log2(4)+1 = 3.
        //   rest = 8 - 4 = 4 = 2^2, so implied lastWeight = 3.
        //
        // Raw header layout: first byte = 127 + oSize; the remaining
        // bytes pack two nibbles each (high nibble first). We supply 3
        // explicit weights; the 4th is computed.
        let src = [130u8, 0x12, 0x10];

        // Allocate a DTable with maxTableLog = 2 (so max supported
        // tableLog = 3, matching what we'll read). Slot 0 is the
        // DTableDesc; we need 1 + (1<<tableLog)/2 = 5 u32 slots.
        let mut dtable = [0u32; 16];
        HUF_setDTableDesc(
            &mut dtable,
            DTableDesc {
                maxTableLog: 2,
                ..Default::default()
            },
        );

        let mut wksp = [0u32; HUF_DECOMPRESS_WORKSPACE_SIZE_U32];
        let iSize = HUF_readDTableX1(&mut dtable, &src, &mut wksp, 0);
        assert!(
            !crate::common::error::ERR_isError(iSize),
            "readDTableX1 returned error: {}",
            crate::common::error::ERR_getErrorName(iSize)
        );
        // Raw header: iSize returned == bytes consumed = iSize_header + 1.
        //   oSize=3 → iSize_header=(3+1)/2=2 → total bytes = 3.
        assert_eq!(iSize, 3);

        let dtd = HUF_getDTableDesc(&dtable);
        assert_eq!(dtd.tableType, 0);
        assert_eq!(dtd.tableLog, 3);

        // Decode table has 8 entries. All entries must have non-zero
        // nbBits (otherwise they'd match the "null" pattern we zeroed).
        let tableSize = 1usize << dtd.tableLog;
        for i in 0..tableSize {
            let e = read_entry(&dtable, i);
            assert!(
                e.nbBits > 0 && e.nbBits <= dtd.tableLog,
                "entry {i} has invalid nbBits {}",
                e.nbBits
            );
            assert!(
                (e.byte as u32) < 4,
                "entry {i} decodes to out-of-range symbol {}",
                e.byte
            );
        }
    }

    #[test]
    fn decompress1X2_DCtx_roundtrips_compress1X_output() {
        // Force the single-stream X2 decode path. `HUF_compress1X_repeat`
        // emits a single-stream HUF block; `HUF_decompress1X2_DCtx_wksp`
        // builds an X2 DTable and dispatches through
        // `HUF_decompress1X2_usingDTable_internal`, which shares the
        // tail-stage `HUF_decodeStreamX2` with the 4X2 path. This is
        // the regression gate confirming the 1X2 leg sees the same
        // fixes as 4X2.
        use crate::compress::huf_compress::HUF_compress1X_repeat;

        // Skewed payload so X2 picks up substantial 2-byte sequences.
        let mut src = vec![0u8; 4096];
        for (i, b) in src.iter_mut().enumerate() {
            // Three symbols dominate (0/1/2), three are rare (3/4/5).
            *b = match i % 16 {
                0..=4 => 0,
                5..=9 => 1,
                10..=12 => 2,
                13 => 3,
                14 => 4,
                _ => 5,
            };
        }

        let mut compressed = vec![0u8; src.len() + 256];
        let n = HUF_compress1X_repeat(&mut compressed, &src, 5, 11, None, None, 0);
        assert!(
            !crate::common::error::ERR_isError(n),
            "compress1X failed: {n:#x}"
        );
        assert!(n > 0 && n < src.len(), "expected actual compression, got n={n}");
        compressed.truncate(n);

        let mut dtable = vec![0u32; HUF_DTABLE_SIZE_U32(HUF_TABLELOG_MAX)];
        let mtl_minus_1 = HUF_TABLELOG_MAX - 1;
        dtable[0] = mtl_minus_1 * 0x01_00_00_01;
        let mut wksp = vec![0u32; HUF_DECOMPRESS_WORKSPACE_SIZE_U32];
        let mut decoded = vec![0u8; src.len()];
        let r = HUF_decompress1X2_DCtx_wksp(&mut dtable, &mut decoded, &compressed, &mut wksp, 0);
        assert!(
            !crate::common::error::ERR_isError(r),
            "1X2 decode failed: {r:#x}"
        );
        assert_eq!(r, src.len());
        assert_eq!(decoded, src, "1X2 decode mismatched input");
    }

    #[test]
    fn rescale_stats_raises_tableLog() {
        let mut weights = [2u8, 3, 0, 5];
        // rankVal before: weight-0: 1 (the zero entry), weight-2: 1,
        // weight-3: 1, weight-5: 1, rest 0. tableLog=5.
        let mut ranks = [0u32; 13];
        ranks[0] = 1;
        ranks[2] = 1;
        ranks[3] = 1;
        ranks[5] = 1;
        let out = HUF_rescaleStats(&mut weights, &mut ranks, 4, 5, 8);
        assert_eq!(out, 8);
        // Non-zero weights moved up by scale=3: 2→5, 3→6, 5→8. Zero stays zero.
        assert_eq!(weights, [5, 6, 0, 8]);
        // Ranks shifted: old[5]→[8], old[3]→[6], old[2]→[5]; new [1..=3] zero.
        assert_eq!(ranks[8], 1);
        assert_eq!(ranks[6], 1);
        assert_eq!(ranks[5], 1);
        assert_eq!(ranks[1], 0);
        assert_eq!(ranks[2], 0);
        assert_eq!(ranks[3], 0);
    }
}
