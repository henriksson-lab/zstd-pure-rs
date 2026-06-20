//! Public type definitions from `lib/zdict.h` (the `ZDICT_*` parameter
//! structs). The `zdict.c` functions are translated separately; this
//! module currently holds only the shared types that `cover.rs` /
//! `fastcover.rs` depend on.

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]

pub const ZDICT_DICTSIZE_MIN: usize = 256;
pub const ZDICT_CONTENTSIZE_MIN: usize = 128;

/// Port of `ZDICT_params_t`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ZDICT_params_t {
    /// optimize for a specific zstd compression level; 0 means default
    pub compressionLevel: i32,
    /// Write log to stderr; 0 = none (default) … 4 = debug
    pub notificationLevel: u32,
    /// force dictID value; 0 means auto mode (32-bits random value)
    pub dictID: u32,
}

/// Port of `ZDICT_cover_params_t`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZDICT_cover_params_t {
    pub k: u32,
    pub d: u32,
    pub steps: u32,
    pub nbThreads: u32,
    pub splitPoint: f64,
    pub shrinkDict: u32,
    pub shrinkDictMaxRegression: u32,
    pub zParams: ZDICT_params_t,
}

/// Port of `ZDICT_legacy_params_t`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZDICT_legacy_params_t {
    pub selectivityLevel: u32,
    pub zParams: ZDICT_params_t,
}

/// Port of `ZDICT_fastCover_params_t`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZDICT_fastCover_params_t {
    pub k: u32,
    pub d: u32,
    pub f: u32,
    pub steps: u32,
    pub nbThreads: u32,
    pub splitPoint: f64,
    pub accel: u32,
    pub shrinkDict: u32,
    pub shrinkDictMaxRegression: u32,
    pub zParams: ZDICT_params_t,
}

/* ======================================================================== */
/* Entropy analysis + dictionary finalization (zdict.c)                      */
/* ======================================================================== */

use crate::common::bits::ZSTD_highbit32;
use crate::common::error::{ERR_isError, ErrorCode, ERROR};
use crate::common::mem::MEM_writeLE32;
use crate::common::xxhash::XXH64;
use crate::common::zstd_internal::repStartValue;
use crate::compress::fse_compress::{FSE_normalizeCount, FSE_writeNCount};
use crate::compress::huf_compress::{
    HUF_CElt, HUF_buildCTable_wksp, HUF_writeCTable_wksp, HUF_CTABLE_WORKSPACE_SIZE_U32,
    HUF_WRITE_CTABLE_WORKSPACE_SIZE,
};
use crate::compress::zstd_compress::{
    ZSTD_CCtx, ZSTD_CDict, ZSTD_compressBegin_usingCDict_deprecated, ZSTD_compressBlock_deprecated,
    ZSTD_createCCtx, ZSTD_createCDict_advanced, ZSTD_getParams, ZSTD_parameters, ZSTD_seqToCodes,
};
use crate::decompress::zstd_ddict::{ZSTD_dictContentType_e, ZSTD_dictLoadMethod_e};
use crate::decompress::zstd_decompress::ZSTD_MAGIC_DICTIONARY;
use crate::decompress::zstd_decompress_block::{
    LLFSELog, MLFSELog, MaxLL, MaxML, OffFSELog, ZSTD_BLOCKSIZE_MAX, ZSTD_REP_NUM,
};

const ZSTD_CLEVEL_DEFAULT: i32 = 3;
const MAXREPOFFSET: usize = 1024;
const OFFCODE_MAX: u32 = 30; /* only applicable to first block */

/// Port of `ZDICT_fillNoise`.
fn ZDICT_fillNoise(buffer: &mut [u8], length: usize) {
    let prime1: u32 = 2654435761;
    let prime2: u32 = 2246822519;
    let mut acc: u32 = prime1;
    for p in 0..length {
        acc = acc.wrapping_mul(prime2);
        buffer[p] = (acc >> 21) as u8;
    }
}

/// Port of `EStats_ress_t`.
struct EStats_ress_t {
    dict: Box<ZSTD_CDict>,
    zc: Box<ZSTD_CCtx>,
    workPlace: Vec<u8>, // ZSTD_BLOCKSIZE_MAX
}

/// Port of `ZDICT_countEStats`.
#[allow(clippy::too_many_arguments)]
fn ZDICT_countEStats(
    esr: &mut EStats_ress_t,
    params: &ZSTD_parameters,
    countLit: &mut [u32],
    offsetcodeCount: &mut [u32],
    matchlengthCount: &mut [u32],
    litlengthCount: &mut [u32],
    repOffsets: &mut [u32],
    src: &[u8],
) {
    let blockSizeMax = core::cmp::min(ZSTD_BLOCKSIZE_MAX, 1usize << params.cParams.windowLog);
    let mut srcSize = src.len();
    if srcSize > blockSizeMax {
        srcSize = blockSizeMax; /* protection vs large samples */
    }
    let src = &src[..srcSize];
    {
        let errorCode = ZSTD_compressBegin_usingCDict_deprecated(&mut esr.zc, &esr.dict);
        if ERR_isError(errorCode) {
            return;
        }
    }
    let cSize = ZSTD_compressBlock_deprecated(&mut esr.zc, &mut esr.workPlace, src);
    if ERR_isError(cSize) {
        return;
    }

    if cSize != 0 {
        /* if == 0; block is not compressible */
        let seqStorePtr = match esr.zc.seqStore.as_mut() {
            Some(s) => s,
            None => return,
        };

        /* literals stats */
        for &b in &seqStorePtr.literals {
            countLit[b as usize] += 1;
        }

        /* seqStats */
        let nbSeq = seqStorePtr.sequences.len();
        ZSTD_seqToCodes(seqStorePtr);

        for u in 0..nbSeq {
            offsetcodeCount[seqStorePtr.ofCode[u] as usize] += 1;
        }
        for u in 0..nbSeq {
            matchlengthCount[seqStorePtr.mlCode[u] as usize] += 1;
        }
        for u in 0..nbSeq {
            litlengthCount[seqStorePtr.llCode[u] as usize] += 1;
        }

        if nbSeq >= 2 {
            /* rep offsets */
            let mut offset1 = seqStorePtr.sequences[0]
                .offBase
                .wrapping_sub(ZSTD_REP_NUM as u32);
            let mut offset2 = seqStorePtr.sequences[1]
                .offBase
                .wrapping_sub(ZSTD_REP_NUM as u32);
            if offset1 as usize >= MAXREPOFFSET {
                offset1 = 0;
            }
            if offset2 as usize >= MAXREPOFFSET {
                offset2 = 0;
            }
            repOffsets[offset1 as usize] += 3;
            repOffsets[offset2 as usize] += 1;
        }
    }
}

/// Port of `ZDICT_totalSampleSize`.
fn ZDICT_totalSampleSize(fileSizes: &[usize], nbFiles: u32) -> usize {
    let mut total = 0usize;
    for u in 0..nbFiles as usize {
        total += fileSizes[u];
    }
    total
}

/// Port of `offsetCount_t`.
#[derive(Clone, Copy, Default)]
struct offsetCount_t {
    offset: u32,
    count: u32,
}

/// Port of `ZDICT_insertSortCount`. `table` has `ZSTD_REP_NUM + 1` entries.
fn ZDICT_insertSortCount(table: &mut [offsetCount_t], val: u32, count: u32) {
    table[ZSTD_REP_NUM].offset = val;
    table[ZSTD_REP_NUM].count = count;
    let mut u = ZSTD_REP_NUM;
    while u > 0 {
        if table[u - 1].count >= table[u].count {
            break;
        }
        table.swap(u - 1, u);
        u -= 1;
    }
}

/// Port of `ZDICT_flatLit`.
fn ZDICT_flatLit(countLit: &mut [u32]) {
    for u in 1..256 {
        countLit[u] = 2;
    }
    countLit[0] = 4;
    countLit[253] = 1;
    countLit[254] = 1;
}

/// Port of `ZDICT_analyzeEntropy`. Builds the entropy-table header into
/// `dstBuffer`; returns its size or an error.
#[allow(clippy::too_many_arguments)]
fn ZDICT_analyzeEntropy(
    dstBuffer: &mut [u8],
    mut compressionLevel: i32,
    srcBuffer: &[u8],
    fileSizes: &[usize],
    nbFiles: u32,
    dictBuffer: &[u8],
    dictBufferSize: usize,
) -> usize {
    let mut countLit = [0u32; 256];
    let mut hufTable = vec![0 as HUF_CElt; 257];
    let mut offcodeCount = [0u32; OFFCODE_MAX as usize + 1];
    let mut offcodeNCount = [0i16; OFFCODE_MAX as usize + 1];
    let offcodeMax = ZSTD_highbit32((dictBufferSize + 128 * 1024) as u32);
    let mut matchLengthCount = [0u32; MaxML as usize + 1];
    let mut matchLengthNCount = [0i16; MaxML as usize + 1];
    let mut litLengthCount = [0u32; MaxLL as usize + 1];
    let mut litLengthNCount = [0i16; MaxLL as usize + 1];
    let mut repOffset = [0u32; MAXREPOFFSET];
    let mut bestRepOffset = [offsetCount_t::default(); ZSTD_REP_NUM + 1];
    let mut huffLog: u32 = 11;
    let mut Offlog: u32 = OffFSELog;
    let mut mlLog: u32 = MLFSELog;
    let mut llLog: u32 = LLFSELog;
    let mut total: u32;
    let mut pos: usize = 0;
    let mut eSize: usize = 0;
    let totalSrcSize = ZDICT_totalSampleSize(fileSizes, nbFiles);
    let averageSampleSize = totalSrcSize / (nbFiles as usize + (nbFiles == 0) as usize);
    let mut wksp = vec![0u32; HUF_CTABLE_WORKSPACE_SIZE_U32];

    /* init */
    if offcodeMax > OFFCODE_MAX {
        return ERROR(ErrorCode::DictionaryCreationFailed); /* too large dictionary */
    }
    for u in 0..256 {
        countLit[u] = 1; /* any character must be described */
    }
    for u in 0..=offcodeMax as usize {
        offcodeCount[u] = 1;
    }
    for u in 0..=MaxML as usize {
        matchLengthCount[u] = 1;
    }
    for u in 0..=MaxLL as usize {
        litLengthCount[u] = 1;
    }
    repOffset[1] = 1;
    repOffset[4] = 1;
    repOffset[8] = 1;
    if compressionLevel == 0 {
        compressionLevel = ZSTD_CLEVEL_DEFAULT;
    }
    let params = ZSTD_getParams(compressionLevel, averageSampleSize as u64, dictBufferSize);

    let dict = ZSTD_createCDict_advanced(
        dictBuffer,
        ZSTD_dictLoadMethod_e::ZSTD_dlm_byRef,
        ZSTD_dictContentType_e::ZSTD_dct_rawContent,
        params.cParams,
    );
    let zc = ZSTD_createCCtx();
    let mut esr = match (dict, zc) {
        (Some(d), Some(z)) => EStats_ress_t {
            dict: d,
            zc: z,
            workPlace: vec![0u8; ZSTD_BLOCKSIZE_MAX],
        },
        _ => return ERROR(ErrorCode::MemoryAllocation),
    };

    /* collect stats on all samples */
    let mut sample_pos: usize = 0;
    for u in 0..nbFiles as usize {
        ZDICT_countEStats(
            &mut esr,
            &params,
            &mut countLit,
            &mut offcodeCount,
            &mut matchLengthCount,
            &mut litLengthCount,
            &mut repOffset,
            &srcBuffer[sample_pos..sample_pos + fileSizes[u]],
        );
        sample_pos += fileSizes[u];
    }

    /* analyze, build stats, starting with literals */
    {
        let mut maxNbBits = HUF_buildCTable_wksp(&mut hufTable, &countLit, 255, huffLog, &mut wksp);
        if ERR_isError(maxNbBits) {
            return maxNbBits;
        }
        if maxNbBits == 8 {
            /* not compressible : will fail on HUF_writeCTable() */
            ZDICT_flatLit(&mut countLit);
            maxNbBits = HUF_buildCTable_wksp(&mut hufTable, &countLit, 255, huffLog, &mut wksp);
            debug_assert_eq!(maxNbBits, 9);
        }
        huffLog = maxNbBits as u32;
    }

    /* looking for most common first offsets */
    {
        let mut offset = 1u32;
        while (offset as usize) < MAXREPOFFSET {
            ZDICT_insertSortCount(&mut bestRepOffset, offset, repOffset[offset as usize]);
            offset += 1;
        }
    }

    total = 0;
    for u in 0..=offcodeMax as usize {
        total += offcodeCount[u];
    }
    let errorCode = FSE_normalizeCount(
        &mut offcodeNCount,
        Offlog,
        &offcodeCount,
        total as usize,
        offcodeMax,
        1,
    );
    if ERR_isError(errorCode) {
        return errorCode;
    }
    Offlog = errorCode as u32;

    total = 0;
    for u in 0..=MaxML as usize {
        total += matchLengthCount[u];
    }
    let errorCode = FSE_normalizeCount(
        &mut matchLengthNCount,
        mlLog,
        &matchLengthCount,
        total as usize,
        MaxML,
        1,
    );
    if ERR_isError(errorCode) {
        return errorCode;
    }
    mlLog = errorCode as u32;

    total = 0;
    for u in 0..=MaxLL as usize {
        total += litLengthCount[u];
    }
    let errorCode = FSE_normalizeCount(
        &mut litLengthNCount,
        llLog,
        &litLengthCount,
        total as usize,
        MaxLL,
        1,
    );
    if ERR_isError(errorCode) {
        return errorCode;
    }
    llLog = errorCode as u32;

    /* write result to buffer */
    {
        let mut writeWksp = vec![0u8; HUF_WRITE_CTABLE_WORKSPACE_SIZE + 64];
        let hhSize = HUF_writeCTable_wksp(
            &mut dstBuffer[pos..],
            &hufTable,
            255,
            huffLog,
            &mut writeWksp,
        );
        if ERR_isError(hhSize) {
            return hhSize;
        }
        pos += hhSize;
        eSize += hhSize;
    }
    {
        let ohSize = FSE_writeNCount(&mut dstBuffer[pos..], &offcodeNCount, OFFCODE_MAX, Offlog);
        if ERR_isError(ohSize) {
            return ohSize;
        }
        pos += ohSize;
        eSize += ohSize;
    }
    {
        let mhSize = FSE_writeNCount(&mut dstBuffer[pos..], &matchLengthNCount, MaxML, mlLog);
        if ERR_isError(mhSize) {
            return mhSize;
        }
        pos += mhSize;
        eSize += mhSize;
    }
    {
        let lhSize = FSE_writeNCount(&mut dstBuffer[pos..], &litLengthNCount, MaxLL, llLog);
        if ERR_isError(lhSize) {
            return lhSize;
        }
        pos += lhSize;
        eSize += lhSize;
    }

    if dstBuffer.len() - pos < 12 {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    /* at this stage, we don't use the result of "most common first offset" */
    let _ = &bestRepOffset;
    MEM_writeLE32(&mut dstBuffer[pos..], repStartValue[0]);
    MEM_writeLE32(&mut dstBuffer[pos + 4..], repStartValue[1]);
    MEM_writeLE32(&mut dstBuffer[pos + 8..], repStartValue[2]);
    eSize += 12;

    eSize
}

/// Port of `ZDICT_isError`.
pub fn ZDICT_isError(errorCode: usize) -> u32 {
    ERR_isError(errorCode) as u32
}

/// Port of `ZDICT_getErrorName`.
pub fn ZDICT_getErrorName(errorCode: usize) -> &'static str {
    crate::common::error::ERR_getErrorName(errorCode)
}

/// Port of `ZDICT_getDictID`.
pub fn ZDICT_getDictID(dictBuffer: &[u8]) -> u32 {
    if dictBuffer.len() < 8 {
        return 0;
    }
    if crate::common::mem::MEM_readLE32(dictBuffer) != ZSTD_MAGIC_DICTIONARY {
        return 0;
    }
    crate::common::mem::MEM_readLE32(&dictBuffer[4..])
}

/// Port of `ZDICT_getDictHeaderSize`. Returns the size of the dictionary
/// header (magic + dictID + entropy tables), or an error code.
pub fn ZDICT_getDictHeaderSize(dictBuffer: &[u8]) -> usize {
    use crate::common::mem::MEM_readLE32;
    use crate::compress::zstd_compress::{
        ZSTD_entropyCTables_t, ZSTD_loadCEntropy, ZSTD_reset_compressedBlockState,
    };

    if dictBuffer.len() <= 8 || MEM_readLE32(dictBuffer) != ZSTD_MAGIC_DICTIONARY {
        return ERROR(ErrorCode::DictionaryCorrupted);
    }
    // C uses a single ZSTD_compressedBlockState_t {rep, entropy}; the Rust
    // crate splits it (and the HUF workspace is internal to ZSTD_loadCEntropy).
    let mut rep = [0u32; 3];
    let mut entropy = ZSTD_entropyCTables_t::default();
    ZSTD_reset_compressedBlockState(&mut rep, &mut entropy);
    ZSTD_loadCEntropy(&mut entropy, &mut rep, dictBuffer)
}

/* ------------------------------------------------------------------------ */
/* Legacy dictionary trainer (zdict.c)                                       */
/* ------------------------------------------------------------------------ */

const LLIMIT: usize = 64; /* heuristic */
const MINMATCHLENGTH: usize = 7; /* heuristic */
const MINRATIO: u32 = 4;
const NOISELENGTH: usize = 32;
const DICTLISTSIZE_DEFAULT: u32 = 10000;
pub const ZDICT_MAX_SAMPLES_SIZE: usize = 2000usize << 20;
pub const ZDICT_MIN_SAMPLES_SIZE: usize = ZDICT_CONTENTSIZE_MIN * MINRATIO as usize;

/// Port of `ZDICT_count`. Counts common bytes between `buffer[pIn..]` and
/// `buffer[pMatch..]`. Presumes a noisy guard band after the buffer end.
fn ZDICT_count(buffer: &[u8], mut pIn: usize, mut pMatch: usize) -> usize {
    use crate::common::bits::ZSTD_NbCommonBytes;
    let pStart = pIn;
    let base = buffer.as_ptr();
    loop {
        // SAFETY: matches upstream `ZDICT_count`, which performs unchecked
        // word-at-a-time `MEM_readST` loads. The buffer ends with a
        // `NOISELENGTH`(=32)-byte noisy guard band appended by
        // `ZDICT_trainFromBuffer_legacy` (> `size_of::<usize>()`). `pIn` and
        // `pMatch` begin at distinct in-data positions, so the two suffixes
        // differ: the compare hits a non-zero `diff` within the guard band
        // before either offset's 8-byte load can reach the allocation end.
        // `ZSTD_NbCommonBytes` then bounds the final advance to the differing
        // word. Hence every `read_unaligned` stays within `buffer`. The
        // native-endian load mirrors `MEM_readST` (`from_ne_bytes`).
        let diff = unsafe {
            core::ptr::read_unaligned(base.add(pMatch) as *const usize)
                ^ core::ptr::read_unaligned(base.add(pIn) as *const usize)
        };
        if diff == 0 {
            pIn += core::mem::size_of::<usize>();
            pMatch += core::mem::size_of::<usize>();
            continue;
        }
        pIn += ZSTD_NbCommonBytes(diff) as usize;
        return pIn - pStart;
    }
}

/// Port of `dictItem`.
#[derive(Clone, Copy, Default)]
struct dictItem {
    pos: u32,
    length: u32,
    savings: u32,
}

/// Port of `ZDICT_initDictItem`.
fn ZDICT_initDictItem(d: &mut dictItem) {
    d.pos = 1;
    d.length = 0;
    d.savings = u32::MAX;
}

/// Port of `isIncluded`. (`buffer` end is a noisy guard band.)
fn isIncluded(buffer: &[u8], in_off: usize, into_off: usize, length: usize) -> i32 {
    let mut u = 0;
    while u < length {
        if buffer[in_off + u] != buffer[into_off + u] {
            break;
        }
        u += 1;
    }
    (u == length) as i32
}

/// Port of `ZDICT_tryMerge`. Checks if `elt` can be merged into the table;
/// returns the destination id, or 0 if not merged. `table[0].pos` is the
/// element count.
fn ZDICT_tryMerge(
    table: &mut [dictItem],
    mut elt: dictItem,
    eltNbToSkip: u32,
    buffer: &[u8],
) -> u32 {
    use crate::common::mem::MEM_read64;
    let tableSize = table[0].pos;
    let eltEnd = elt.pos + elt.length;

    /* tail overlap */
    let mut u = 1u32;
    while u < tableSize {
        if u == eltNbToSkip {
            u += 1;
            continue;
        }
        if (table[u as usize].pos > elt.pos) && (table[u as usize].pos <= eltEnd) {
            /* overlap, existing > new : append */
            let addedLength = table[u as usize].pos - elt.pos;
            table[u as usize].length += addedLength;
            table[u as usize].pos = elt.pos;
            table[u as usize].savings += elt.savings * addedLength / elt.length; /* rough */
            table[u as usize].savings += elt.length / 8; /* bonus */
            elt = table[u as usize];
            /* sort : improve rank */
            while (u > 1) && (table[(u - 1) as usize].savings < elt.savings) {
                table[u as usize] = table[(u - 1) as usize];
                u -= 1;
            }
            table[u as usize] = elt;
            return u;
        }
        u += 1;
    }

    /* front overlap */
    let mut u = 1u32;
    while u < tableSize {
        if u == eltNbToSkip {
            u += 1;
            continue;
        }
        if (table[u as usize].pos + table[u as usize].length >= elt.pos)
            && (table[u as usize].pos < elt.pos)
        {
            /* overlap, existing < new : append */
            let addedLength =
                eltEnd as i32 - (table[u as usize].pos + table[u as usize].length) as i32;
            table[u as usize].savings += elt.length / 8; /* bonus */
            if addedLength > 0 {
                table[u as usize].length += addedLength as u32;
                table[u as usize].savings += elt.savings * addedLength as u32 / elt.length;
            }
            elt = table[u as usize];
            while (u > 1) && (table[(u - 1) as usize].savings < elt.savings) {
                table[u as usize] = table[(u - 1) as usize];
                u -= 1;
            }
            table[u as usize] = elt;
            return u;
        }

        if MEM_read64(&buffer[table[u as usize].pos as usize..])
            == MEM_read64(&buffer[(elt.pos + 1) as usize..])
        {
            if isIncluded(
                buffer,
                table[u as usize].pos as usize,
                (elt.pos + 1) as usize,
                table[u as usize].length as usize,
            ) != 0
            {
                let addedLength = core::cmp::max(elt.length - table[u as usize].length, 1);
                table[u as usize].pos = elt.pos;
                table[u as usize].savings += elt.savings * addedLength / elt.length;
                table[u as usize].length = core::cmp::min(elt.length, table[u as usize].length + 1);
                return u;
            }
        }
        u += 1;
    }

    0
}

/// Port of `ZDICT_removeDictItem`. `table[0].pos` stores the element count.
fn ZDICT_removeDictItem(table: &mut [dictItem], id: u32) {
    let max = table[0].pos;
    if id == 0 {
        return; /* protection */
    }
    let mut u = id;
    while u < max - 1 {
        table[u as usize] = table[(u + 1) as usize];
        u += 1;
    }
    table[0].pos -= 1;
}

/// Port of `ZDICT_insertDictItem`.
fn ZDICT_insertDictItem(table: &mut [dictItem], maxSize: u32, elt: dictItem, buffer: &[u8]) {
    /* merge if possible */
    let mut mergeId = ZDICT_tryMerge(table, elt, 0, buffer);
    if mergeId != 0 {
        let mut newMerge = 1u32;
        while newMerge != 0 {
            newMerge = ZDICT_tryMerge(table, table[mergeId as usize], mergeId, buffer);
            if newMerge != 0 {
                ZDICT_removeDictItem(table, mergeId);
            }
            mergeId = newMerge;
        }
        return;
    }

    /* insert */
    {
        let mut nextElt = table[0].pos;
        if nextElt >= maxSize {
            nextElt = maxSize - 1;
        }
        let mut current = nextElt - 1;
        while table[current as usize].savings < elt.savings {
            table[(current + 1) as usize] = table[current as usize];
            current = current.wrapping_sub(1);
        }
        table[(current + 1) as usize] = elt;
        table[0].pos = nextElt + 1;
    }
}

/// Port of `ZDICT_dictSize`.
fn ZDICT_dictSize(dictList: &[dictItem]) -> u32 {
    let mut dictSize = 0u32;
    for u in 1..dictList[0].pos as usize {
        dictSize += dictList[u].length;
    }
    dictSize
}

/// Port of `ZDICT_analyzePos`.
///
/// `suffix0` is the full sentinel-prefixed array: the C `suffix` pointer is
/// `suffix0 + 1`, so the C `suffix[X]` is `suffix0[X+1]` here, and the C
/// `*(suffix+start-1)` (== `*((suffix+start)-1)`) is `suffix0[start]`.
fn ZDICT_analyzePos(
    doneMarks: &mut [u8],
    suffix0: &[i32],
    mut start: u32,
    buffer: &[u8],
    minRatio: u32,
) -> dictItem {
    use crate::common::mem::MEM_read16;
    // suffix[i] (C) == suffix0[i+1]
    let sfx = |i: u32| -> usize { suffix0[(i + 1) as usize] as usize };

    let mut lengthList = [0u32; LLIMIT];
    let mut cumulLength = [0u32; LLIMIT];
    let mut savings = [0u32; LLIMIT];
    let b = buffer;
    let mut maxLength: usize = LLIMIT;
    let mut pos: usize = sfx(start);
    let mut end = start;
    let mut solution = dictItem::default();

    /* init */
    doneMarks[pos] = 1;

    /* trivial repetition cases */
    if (MEM_read16(&b[pos..]) == MEM_read16(&b[pos + 2..]))
        || (MEM_read16(&b[pos + 1..]) == MEM_read16(&b[pos + 3..]))
        || (MEM_read16(&b[pos + 2..]) == MEM_read16(&b[pos + 4..]))
    {
        /* skip and mark segment */
        let pattern16 = MEM_read16(&b[pos + 4..]);
        let mut patternEnd = 6usize;
        while MEM_read16(&b[pos + patternEnd..]) == pattern16 {
            patternEnd += 2;
        }
        if b[pos + patternEnd] == b[pos + patternEnd - 1] {
            patternEnd += 1;
        }
        for u in 1..patternEnd {
            doneMarks[pos + u] = 1;
        }
        return solution;
    }

    /* look forward */
    {
        let mut length;
        loop {
            end += 1;
            length = ZDICT_count(b, pos, sfx(end));
            if length < MINMATCHLENGTH {
                break;
            }
        }
    }

    /* look backward */
    {
        let mut length;
        loop {
            length = ZDICT_count(b, pos, suffix0[start as usize] as usize);
            if length >= MINMATCHLENGTH {
                start -= 1;
            }
            if length < MINMATCHLENGTH {
                break;
            }
        }
    }

    /* exit if not found a minimum nb of repetitions */
    if end - start < minRatio {
        for idx in start..end {
            doneMarks[sfx(idx)] = 1;
        }
        return solution;
    }

    {
        let mut refinedStart = start;
        let mut refinedEnd = end;

        let mut mml = MINMATCHLENGTH as u32;
        loop {
            let mut currentChar: u8 = 0;
            let mut currentCount: u32 = 0;
            let mut currentID = refinedStart;
            let mut selectedCount: u32 = 0;
            let mut selectedID = currentID;
            for id in refinedStart..refinedEnd {
                if b[sfx(id) + mml as usize] != currentChar {
                    if currentCount > selectedCount {
                        selectedCount = currentCount;
                        selectedID = currentID;
                    }
                    currentID = id;
                    currentChar = b[sfx(id) + mml as usize];
                    currentCount = 0;
                }
                currentCount += 1;
            }
            if currentCount > selectedCount {
                /* for last */
                selectedCount = currentCount;
                selectedID = currentID;
            }

            if selectedCount < minRatio {
                break;
            }
            refinedStart = selectedID;
            refinedEnd = refinedStart + selectedCount;
            mml += 1;
        }

        /* evaluate gain based on new dict */
        start = refinedStart;
        pos = sfx(refinedStart);
        end = start;
        for l in lengthList.iter_mut() {
            *l = 0;
        }

        /* look forward */
        {
            let mut length;
            loop {
                end += 1;
                length = ZDICT_count(b, pos, sfx(end));
                if length >= LLIMIT {
                    length = LLIMIT - 1;
                }
                lengthList[length] += 1;
                if length < MINMATCHLENGTH {
                    break;
                }
            }
        }

        /* look backward */
        {
            let mut length = MINMATCHLENGTH;
            while (length >= MINMATCHLENGTH) && (start > 0) {
                length = ZDICT_count(b, pos, suffix0[start as usize] as usize);
                if length >= LLIMIT {
                    length = LLIMIT - 1;
                }
                lengthList[length] += 1;
                if length >= MINMATCHLENGTH {
                    start -= 1;
                }
            }
        }

        /* largest useful length */
        for c in cumulLength.iter_mut() {
            *c = 0;
        }
        cumulLength[maxLength - 1] = lengthList[maxLength - 1];
        for i in (0..=(maxLength - 2)).rev() {
            cumulLength[i] = cumulLength[i + 1] + lengthList[i];
        }

        {
            let mut u = LLIMIT - 1;
            while u >= MINMATCHLENGTH {
                if cumulLength[u] >= minRatio {
                    break;
                }
                u -= 1;
            }
            maxLength = u;
        }

        /* reduce maxLength in case of final into repetitive data */
        {
            let mut l = maxLength as u32;
            let c = b[pos + maxLength - 1];
            while b[pos + l as usize - 2] == c {
                l -= 1;
            }
            maxLength = l as usize;
        }
        if maxLength < MINMATCHLENGTH {
            return solution; /* skip : no long-enough solution */
        }

        /* calculate savings */
        savings[5] = 0;
        for u in MINMATCHLENGTH..=maxLength {
            savings[u] = savings[u - 1] + (lengthList[u] * (u as u32 - 3));
        }

        solution.pos = pos as u32;
        solution.length = maxLength as u32;
        solution.savings = savings[maxLength];

        /* mark positions done */
        {
            for id in start..end {
                let testedPos = sfx(id) as u32;
                let length: u32 = if testedPos == pos as u32 {
                    solution.length
                } else {
                    let mut length = ZDICT_count(b, pos, testedPos as usize) as u32;
                    if length > solution.length {
                        length = solution.length;
                    }
                    length
                };
                let pEnd = testedPos + length;
                for p in testedPos..pEnd {
                    doneMarks[p as usize] = 1;
                }
            }
        }
    }

    solution
}

/// Port of `ZDICT_trainBuffer_legacy`. Sorts the buffer with `divsufsort`,
/// then finds repeated patterns via `analyzePos`. `buffer` must end with a
/// noisy guard band. Returns 0 on success or an error code.
fn ZDICT_trainBuffer_legacy(
    dictList: &mut [dictItem],
    dictListSize: u32,
    buffer: &[u8],
    mut bufferSize: usize,
    fileSizes: &[usize],
    mut nbFiles: u32,
    mut minRatio: u32,
) -> usize {
    use crate::dict_builder::divsufsort::divsufsort;

    // suffix0 has bufferSize+2 entries; suffix = suffix0[1..].
    let mut suffix0 = vec![0i32; bufferSize + 2];
    let mut reverseSuffix = vec![0u32; bufferSize];
    let mut doneMarks = vec![0u8; bufferSize + 16]; /* +16 for overflow security */

    if minRatio < MINRATIO {
        minRatio = MINRATIO;
    }

    /* limit sample set size (divsufsort limitation) */
    while bufferSize > ZDICT_MAX_SAMPLES_SIZE {
        nbFiles -= 1;
        bufferSize -= fileSizes[nbFiles as usize];
    }

    /* sort */
    {
        let divResult = divsufsort(
            &buffer[..bufferSize],
            &mut suffix0[1..1 + bufferSize],
            bufferSize as i32,
        );
        if divResult != 0 {
            return ERROR(ErrorCode::Generic);
        }
    }
    suffix0[1 + bufferSize] = bufferSize as i32; /* leads into noise (suffix[bufferSize]) */
    suffix0[0] = bufferSize as i32; /* leads into noise (suffix[-1]) */

    /* build reverse suffix sort */
    for pos in 0..bufferSize {
        reverseSuffix[suffix0[1 + pos] as usize] = pos as u32;
    }
    /* (filePos border tracking is unused at this stage; omitted) */

    /* find patterns */
    {
        let mut cursor: usize = 0;
        while cursor < bufferSize {
            if doneMarks[cursor] != 0 {
                cursor += 1;
                continue;
            }
            let solution = ZDICT_analyzePos(
                &mut doneMarks,
                &suffix0,
                reverseSuffix[cursor],
                buffer,
                minRatio,
            );
            if solution.length == 0 {
                cursor += 1;
                continue;
            }
            ZDICT_insertDictItem(dictList, dictListSize, solution, buffer);
            cursor += solution.length as usize;
        }
    }

    0
}

const g_selectivity_default: u32 = 9;

/// Port of `ZDICT_trainFromBuffer_unsafe_legacy`. `samplesBuffer` must be
/// followed by a noisy guard band.
fn ZDICT_trainFromBuffer_unsafe_legacy(
    dictBuffer: &mut [u8],
    maxDictSize: usize,
    samplesBuffer: &[u8],
    samplesSizes: &[usize],
    nbSamples: u32,
    params: ZDICT_legacy_params_t,
) -> usize {
    let dictListSize = core::cmp::max(
        core::cmp::max(DICTLISTSIZE_DEFAULT, nbSamples),
        (maxDictSize / 16) as u32,
    );
    let mut dictList = vec![dictItem::default(); dictListSize as usize];
    let selectivity = if params.selectivityLevel == 0 {
        g_selectivity_default
    } else {
        params.selectivityLevel
    };
    let minRep = if selectivity > 30 {
        MINRATIO
    } else {
        nbSamples >> selectivity
    };
    let targetDictSize = maxDictSize;
    let samplesBuffSize = ZDICT_totalSampleSize(samplesSizes, nbSamples);
    let dictSize;

    /* checks */
    if maxDictSize < ZDICT_DICTSIZE_MIN {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    if samplesBuffSize < ZDICT_MIN_SAMPLES_SIZE {
        return ERROR(ErrorCode::DictionaryCreationFailed);
    }

    /* init */
    ZDICT_initDictItem(&mut dictList[0]);

    /* build dictionary */
    ZDICT_trainBuffer_legacy(
        &mut dictList,
        dictListSize,
        samplesBuffer,
        samplesBuffSize,
        samplesSizes,
        nbSamples,
        minRep,
    );

    /* (display of best matches omitted — diagnostics only) */

    /* create dictionary */
    {
        let mut dictContentSize = ZDICT_dictSize(&dictList);
        if (dictContentSize as usize) < ZDICT_CONTENTSIZE_MIN {
            return ERROR(ErrorCode::DictionaryCreationFailed); /* too small */
        }
        let _ = targetDictSize; /* used only by the omitted warnings */

        /* limit dictionary size */
        {
            let max = dictList[0].pos;
            let mut currentSize: u32 = 0;
            let mut n = 1u32;
            while n < max {
                currentSize += dictList[n as usize].length;
                if currentSize as usize > targetDictSize {
                    currentSize -= dictList[n as usize].length;
                    break;
                }
                n += 1;
            }
            dictList[0].pos = n;
            dictContentSize = currentSize;
        }

        /* build dict content (filled from the back of dictBuffer) */
        {
            let mut ptr = maxDictSize; // BYTE* ptr = dictBuffer + maxDictSize
            for u in 1..dictList[0].pos as usize {
                let l = dictList[u].length as usize;
                ptr -= l;
                if ptr > maxDictSize {
                    return ERROR(ErrorCode::Generic); /* should not happen */
                }
                let src = dictList[u].pos as usize;
                dictBuffer[ptr..ptr + l].copy_from_slice(&samplesBuffer[src..src + l]);
            }
        }

        dictSize = ZDICT_addEntropyTablesFromBuffer_advanced(
            dictBuffer,
            dictContentSize as usize,
            maxDictSize,
            samplesBuffer,
            samplesSizes,
            nbSamples,
            params.zParams,
        );
    }

    dictSize
}

/// Port of `ZDICT_trainFromBuffer_legacy`. Duplicates the samples buffer and
/// appends a noisy guard band before training.
pub fn ZDICT_trainFromBuffer_legacy(
    dictBuffer: &mut [u8],
    dictBufferCapacity: usize,
    samplesBuffer: &[u8],
    samplesSizes: &[usize],
    nbSamples: u32,
    params: ZDICT_legacy_params_t,
) -> usize {
    let sBuffSize = ZDICT_totalSampleSize(samplesSizes, nbSamples);
    if sBuffSize < ZDICT_MIN_SAMPLES_SIZE {
        return 0; /* not enough content => no dictionary */
    }
    let mut newBuff = vec![0u8; sBuffSize + NOISELENGTH];
    newBuff[..sBuffSize].copy_from_slice(&samplesBuffer[..sBuffSize]);
    ZDICT_fillNoise(&mut newBuff[sBuffSize..], NOISELENGTH); /* guard band */
    ZDICT_trainFromBuffer_unsafe_legacy(
        dictBuffer,
        dictBufferCapacity,
        &newBuff,
        samplesSizes,
        nbSamples,
        params,
    )
}

/// Port of `ZDICT_addEntropyTablesFromBuffer`.
pub fn ZDICT_addEntropyTablesFromBuffer(
    dictBuffer: &mut [u8],
    dictContentSize: usize,
    dictBufferCapacity: usize,
    samplesBuffer: &[u8],
    samplesSizes: &[usize],
    nbSamples: u32,
) -> usize {
    let params = ZDICT_params_t::default();
    ZDICT_addEntropyTablesFromBuffer_advanced(
        dictBuffer,
        dictContentSize,
        dictBufferCapacity,
        samplesBuffer,
        samplesSizes,
        nbSamples,
        params,
    )
}

/// Port of `ZDICT_trainFromBuffer`. The public default trainer: optimizes
/// fastCover with `d=8, steps=4, level=3`.
pub fn ZDICT_trainFromBuffer(
    dictBuffer: &mut [u8],
    dictBufferCapacity: usize,
    samplesBuffer: &[u8],
    samplesSizes: &[usize],
    nbSamples: u32,
) -> usize {
    let mut params = ZDICT_fastCover_params_t::default();
    params.d = 8;
    params.steps = 4;
    /* Default to level 3 */
    params.zParams.compressionLevel = 3;
    crate::dict_builder::fastcover::fastcover_c::ZDICT_optimizeTrainFromBuffer_fastCover(
        dictBuffer,
        dictBufferCapacity,
        samplesBuffer,
        samplesSizes,
        nbSamples,
        &mut params,
    )
}

/// Port of `ZDICT_maxRep`.
fn ZDICT_maxRep(reps: &[u32; ZSTD_REP_NUM]) -> u32 {
    let mut maxRep = reps[0];
    for r in 1..ZSTD_REP_NUM {
        maxRep = core::cmp::max(maxRep, reps[r]);
    }
    maxRep
}

/// Port of `ZDICT_finalizeDictionary`. Wraps `customDictContent` with a
/// dictionary header (magic + dictID), entropy tables, and padding,
/// writing the final dictionary into `dictBuffer`. Returns the dict size.
#[allow(clippy::too_many_arguments)]
pub fn ZDICT_finalizeDictionary(
    dictBuffer: &mut [u8],
    dictBufferCapacity: usize,
    customDictContent: &[u8],
    mut dictContentSize: usize,
    samplesBuffer: &[u8],
    samplesSizes: &[usize],
    nbSamples: u32,
    params: ZDICT_params_t,
) -> usize {
    const HBUFFSIZE: usize = 256;
    let mut header = [0u8; HBUFFSIZE];
    let compressionLevel = if params.compressionLevel == 0 {
        ZSTD_CLEVEL_DEFAULT
    } else {
        params.compressionLevel
    };
    /* The final dictionary content must be at least as large as the largest repcode. */
    let minContentSize = ZDICT_maxRep(&repStartValue) as usize;
    let paddingSize: usize;

    /* check conditions */
    if dictBufferCapacity < dictContentSize {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    if dictBufferCapacity < ZDICT_DICTSIZE_MIN {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }

    /* dictionary header */
    MEM_writeLE32(&mut header[..], ZSTD_MAGIC_DICTIONARY);
    {
        let randomID = XXH64(&customDictContent[..dictContentSize], 0);
        let compliantID = (randomID % ((1u64 << 31) - 32768)) as u32 + 32768;
        let dictID = if params.dictID != 0 {
            params.dictID
        } else {
            compliantID
        };
        MEM_writeLE32(&mut header[4..], dictID);
    }
    let mut hSize: usize = 8;

    /* entropy tables */
    {
        let eSize = ZDICT_analyzeEntropy(
            &mut header[hSize..],
            compressionLevel,
            samplesBuffer,
            samplesSizes,
            nbSamples,
            customDictContent,
            dictContentSize,
        );
        if ERR_isError(eSize) {
            return eSize;
        }
        hSize += eSize;
    }

    /* Shrink the content size if it doesn't fit in the buffer. */
    if hSize + dictContentSize > dictBufferCapacity {
        dictContentSize = dictBufferCapacity - hSize;
    }

    /* Pad the dictionary content with zeros if it is too small. */
    if dictContentSize < minContentSize {
        if hSize + minContentSize > dictBufferCapacity {
            return ERROR(ErrorCode::DstSizeTooSmall);
        }
        paddingSize = minContentSize - dictContentSize;
    } else {
        paddingSize = 0;
    }

    {
        let dictSize = hSize + paddingSize + dictContentSize;

        /* header | padding | content. Content copied first because
         * `customDictContent` and `dictBuffer` may overlap (use a temp). */
        let content_dst = hSize + paddingSize;
        let content: Vec<u8> = customDictContent[..dictContentSize].to_vec();
        dictBuffer[content_dst..content_dst + dictContentSize].copy_from_slice(&content);
        dictBuffer[..hSize].copy_from_slice(&header[..hSize]);
        for b in &mut dictBuffer[hSize..hSize + paddingSize] {
            *b = 0;
        }
        dictSize
    }
}

/// Port of `ZDICT_addEntropyTablesFromBuffer_advanced`.
fn ZDICT_addEntropyTablesFromBuffer_advanced(
    dictBuffer: &mut [u8],
    dictContentSize: usize,
    dictBufferCapacity: usize,
    samplesBuffer: &[u8],
    samplesSizes: &[usize],
    nbSamples: u32,
    params: ZDICT_params_t,
) -> usize {
    let compressionLevel = if params.compressionLevel == 0 {
        ZSTD_CLEVEL_DEFAULT
    } else {
        params.compressionLevel
    };
    let mut hSize: usize = 8;

    /* calculate entropy tables (content currently sits at the buffer end) */
    {
        let content: Vec<u8> =
            dictBuffer[dictBufferCapacity - dictContentSize..dictBufferCapacity].to_vec();
        let eSize = ZDICT_analyzeEntropy(
            &mut dictBuffer[hSize..dictBufferCapacity],
            compressionLevel,
            samplesBuffer,
            samplesSizes,
            nbSamples,
            &content,
            dictContentSize,
        );
        if ERR_isError(eSize) {
            return eSize;
        }
        hSize += eSize;
    }

    /* add dictionary header (after entropy tables) */
    MEM_writeLE32(&mut dictBuffer[..], ZSTD_MAGIC_DICTIONARY);
    {
        let randomID = XXH64(
            &dictBuffer[dictBufferCapacity - dictContentSize..dictBufferCapacity],
            0,
        );
        let compliantID = (randomID % ((1u64 << 31) - 32768)) as u32 + 32768;
        let dictID = if params.dictID != 0 {
            params.dictID
        } else {
            compliantID
        };
        MEM_writeLE32(&mut dictBuffer[4..], dictID);
    }

    if hSize + dictContentSize < dictBufferCapacity {
        dictBuffer.copy_within(
            dictBufferCapacity - dictContentSize..dictBufferCapacity,
            hSize,
        );
    }
    core::cmp::min(dictBufferCapacity, hSize + dictContentSize)
}
