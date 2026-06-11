//! `lib/common/` translation: shared primitives (bits, mem, errors, entropy).

pub mod bits;
pub mod bitstream;
pub mod debug;
pub mod entropy_common;
pub mod error;
pub mod fse_decompress;
pub mod mem;
#[cfg(feature = "std")]
pub mod pool;
#[cfg(feature = "std")]
pub mod threading;
pub mod xxhash;
pub mod zstd_common;
pub mod zstd_internal;

/// Upstream `lib/common/fse.h` namespace.
///
/// The translated implementation is split across `entropy_common`,
/// `fse_decompress`, and `compress::fse_compress`. This module mirrors
/// the header-level API surface without changing the implementation
/// module names.
pub mod fse {
    pub const FSE_VERSION_MAJOR: u32 = 0;
    pub const FSE_VERSION_MINOR: u32 = 9;
    pub const FSE_VERSION_RELEASE: u32 = 0;
    pub const FSE_VERSION_STRING: &str = "0.9.0";

    pub use crate::common::entropy_common::{
        FSE_getErrorName, FSE_isError, FSE_readNCount, FSE_readNCount_bmi2, FSE_versionNumber,
        FSE_BUILD_DTABLE_WKSP_SIZE, FSE_BUILD_DTABLE_WKSP_SIZE_U32, FSE_DECOMPRESS_WKSP_SIZE,
        FSE_DECOMPRESS_WKSP_SIZE_U32, FSE_DTABLE_SIZE, FSE_DTABLE_SIZE_U32, FSE_VERSION_NUMBER,
    };
    pub use crate::common::fse_decompress::{
        FSE_DState_t, FSE_DTable, FSE_buildDTable_internal, FSE_buildDTable_raw,
        FSE_buildDTable_rle, FSE_buildDTable_wksp, FSE_decodeSymbol, FSE_decodeSymbolFast,
        FSE_decompress_usingDTable, FSE_decompress_wksp, FSE_decompress_wksp_bmi2, FSE_endOfDState,
        FSE_initDState, FSE_peekSymbol, FSE_updateState, FSE_MAX_MEMORY_USAGE,
        FSE_MAX_SYMBOL_VALUE, FSE_MAX_TABLELOG, FSE_MIN_TABLELOG, FSE_TABLELOG_ABSOLUTE_MAX,
        FSE_TABLESTEP,
    };
    pub use crate::compress::fse_compress::{
        ct_header_maxSV, ct_header_tableLog, symbolTT_read, FSE_CState_t, FSE_CTable,
        FSE_NCountWriteBound, FSE_bitCost, FSE_buildCTable_rle, FSE_buildCTable_wksp,
        FSE_compressBound, FSE_compress_usingCTable, FSE_encodeSymbol, FSE_flushCState,
        FSE_getMaxNbBits, FSE_initCState, FSE_initCState2, FSE_normalizeCount, FSE_optimalTableLog,
        FSE_optimalTableLog_internal, FSE_symbolCompressionTransform, FSE_writeNCount,
        FSE_BLOCKBOUND, FSE_BUILD_CTABLE_WORKSPACE_SIZE, FSE_BUILD_CTABLE_WORKSPACE_SIZE_U32,
        FSE_COMPRESSBOUND, FSE_CTABLE_SIZE_U32, FSE_DEFAULT_MEMORY_USAGE, FSE_DEFAULT_TABLELOG,
        FSE_NCOUNTBOUND,
    };
    pub use crate::compress::zstd_compress_sequences::FSE_repeat;

    #[inline]
    pub fn FSE_CTABLE_SIZE(tableLog: u32, maxSymbolValue: u32) -> usize {
        FSE_CTABLE_SIZE_U32(tableLog, maxSymbolValue) * core::mem::size_of::<FSE_CTable>()
    }

    /// Upstream `FSE_buildCTable()` convenience entry point.
    ///
    /// The implementation mirrors the public non-`_wksp` API by using
    /// internally managed scratch space sized for the maximum supported
    /// `(maxSymbolValue, tableLog)` pair, then delegating to the translated
    /// workspace builder.
    pub fn FSE_buildCTable(
        ct: &mut [FSE_CTable],
        normalizedCounter: &[i16],
        maxSymbolValue: u32,
        tableLog: u32,
    ) -> usize {
        let mut wksp =
            [0u8; FSE_BUILD_CTABLE_WORKSPACE_SIZE(FSE_MAX_SYMBOL_VALUE, FSE_MAX_TABLELOG)];
        FSE_buildCTable_wksp(ct, normalizedCounter, maxSymbolValue, tableLog, &mut wksp)
    }
}

/// Upstream `lib/common/huf.h` namespace.
///
/// HUF support is translated into common entropy parsing plus separate
/// compression/decompression implementation files. This module exposes
/// that split implementation under the upstream common header name.
pub mod huf {
    pub const HUF_OPTIMAL_DEPTH_THRESHOLD: u32 =
        crate::compress::zstd_compress_sequences::ZSTD_btultra;

    pub use crate::common::entropy_common::{
        HUF_getErrorName, HUF_isError, HUF_readStats, HUF_readStats_wksp,
        HUF_READ_STATS_WORKSPACE_SIZE, HUF_READ_STATS_WORKSPACE_SIZE_U32,
    };
    pub use crate::compress::huf_compress::{
        nodeElt, rankPos, showCTableBits, showHNodeBits, showHNodeSymbols, showU32, HUF_CElt,
        HUF_CStream_t, HUF_CTableHeader, HUF_addBits, HUF_alignUpWorkspace, HUF_buildCTable,
        HUF_buildCTableFromTree, HUF_buildCTable_wksp, HUF_buildTree, HUF_cardinality,
        HUF_closeCStream, HUF_compress1X_repeat, HUF_compress1X_usingCTable,
        HUF_compress1X_usingCTable_body_loop, HUF_compress1X_usingCTable_body_loop_specialized,
        HUF_compress1X_usingCTable_internal, HUF_compress1X_usingCTable_internal_bmi2,
        HUF_compress1X_usingCTable_internal_body, HUF_compress1X_usingCTable_internal_body_loop,
        HUF_compress1X_usingCTable_internal_default, HUF_compress4X_repeat,
        HUF_compress4X_usingCTable, HUF_compress4X_usingCTable_internal, HUF_compressBound,
        HUF_compressCTable_internal, HUF_compressWeights, HUF_compress_internal, HUF_encodeSymbol,
        HUF_estimateCompressedSize, HUF_flushBits, HUF_getIndex, HUF_getNbBits, HUF_getNbBitsFast,
        HUF_getNbBitsFromCTable, HUF_getValue, HUF_getValueFast, HUF_initCStream, HUF_isSorted,
        HUF_mergeIndex1, HUF_minTableLog, HUF_nbStreams_e, HUF_optimalTableLog_internal,
        HUF_readCTable, HUF_readCTableHeader, HUF_setMaxHeight, HUF_setNbBits, HUF_setValue,
        HUF_sort, HUF_swapNodes, HUF_tightCompressBound, HUF_writeCTable, HUF_writeCTableHeader,
        HUF_writeCTable_wksp, HUF_zeroIndex1, HUF_BITS_IN_CONTAINER, HUF_BLOCKBOUND,
        HUF_BLOCKSIZE_MAX, HUF_COMPRESSBOUND, HUF_CTABLEBOUND, HUF_CTABLE_HEADER_UNUSED_SIZE,
        HUF_CTABLE_SIZE, HUF_CTABLE_SIZE_ST, HUF_CTABLE_WORKSPACE_SIZE,
        HUF_CTABLE_WORKSPACE_SIZE_U32, HUF_DTABLE_SIZE, HUF_TABLELOG_DEFAULT,
        HUF_WORKSPACE_MAX_ALIGNMENT, HUF_WORKSPACE_SIZE, HUF_WORKSPACE_SIZE_U64,
        HUF_WRITE_CTABLE_WORKSPACE_SIZE, MAX_FSE_TABLELOG_FOR_HUFF_HEADER,
        RANK_POSITION_DISTINCT_COUNT_CUTOFF, RANK_POSITION_LOG_BUCKETS_BEGIN,
        RANK_POSITION_MAX_COUNT_LOG, RANK_POSITION_TABLE_SIZE, STARTNODE,
    };
    pub use crate::compress::zstd_compress_literals::HUF_repeat;
    pub use crate::decompress::huf_decompress::{
        read_entry, read_entry_x2, write_entry_x2, DTableDesc, HUF_DEltX1, HUF_DEltX1_pack,
        HUF_DEltX1_set4, HUF_DEltX1_unpack, HUF_DEltX2, HUF_DEltX2_pack, HUF_DEltX2_unpack,
        HUF_DTable, HUF_buildDEltX2, HUF_buildDEltX2U32, HUF_buildDEltX2U64,
        HUF_decodeLastSymbolX2, HUF_decodeStreamX1, HUF_decodeStreamX2, HUF_decodeSymbolX1,
        HUF_decodeSymbolX2, HUF_decompress1X1_DCtx_wksp, HUF_decompress1X1_usingDTable_internal,
        HUF_decompress1X1_usingDTable_internal_bmi2, HUF_decompress1X1_usingDTable_internal_body,
        HUF_decompress1X2_DCtx_wksp, HUF_decompress1X2_usingDTable_internal,
        HUF_decompress1X2_usingDTable_internal_bmi2, HUF_decompress1X2_usingDTable_internal_body,
        HUF_decompress1X_DCtx_wksp, HUF_decompress4X1_DCtx_wksp,
        HUF_decompress4X1_usingDTable_internal, HUF_decompress4X1_usingDTable_internal_bmi2,
        HUF_decompress4X1_usingDTable_internal_body,
        HUF_decompress4X1_usingDTable_internal_default,
        HUF_decompress4X1_usingDTable_internal_fast,
        HUF_decompress4X1_usingDTable_internal_fast_c_loop, HUF_decompress4X2_DCtx_wksp,
        HUF_decompress4X2_usingDTable_internal, HUF_decompress4X2_usingDTable_internal_bmi2,
        HUF_decompress4X2_usingDTable_internal_body,
        HUF_decompress4X2_usingDTable_internal_default,
        HUF_decompress4X2_usingDTable_internal_fast,
        HUF_decompress4X2_usingDTable_internal_fast_c_loop, HUF_decompress4X_hufOnly_wksp,
        HUF_fillDTableX2, HUF_fillDTableX2ForWeight, HUF_fillDTableX2Level2, HUF_flags_bmi2,
        HUF_flags_disableAsm, HUF_flags_disableFast, HUF_flags_optimalDepth,
        HUF_flags_preferRepeat, HUF_flags_suspectUncompressible, HUF_getDTableDesc,
        HUF_initRemainingDStream, HUF_readDTableX1, HUF_readDTableX1_wksp, HUF_readDTableX2,
        HUF_readDTableX2_wksp, HUF_rescaleStats, HUF_selectDecoder, HUF_setDTableDesc,
        HUF_DECODER_FAST_TABLELOG, HUF_DECOMPRESS_WORKSPACE_SIZE,
        HUF_DECOMPRESS_WORKSPACE_SIZE_U32, HUF_DTABLE_SIZE_U32, HUF_SYMBOLVALUE_MAX,
        HUF_TABLELOG_ABSOLUTEMAX, HUF_TABLELOG_MAX,
    };

    /// Upstream `HUF_decompress1X_usingDTable()` runtime-flags entry point.
    #[inline]
    pub fn HUF_decompress1X_usingDTable(
        dst: &mut [u8],
        cSrc: &[u8],
        dtable: &[HUF_DTable],
        flags: i32,
    ) -> usize {
        if (flags & HUF_flags_bmi2) != 0 {
            crate::decompress::huf_decompress::HUF_decompress1X_usingDTable::<true>(
                dst, cSrc, dtable,
            )
        } else {
            crate::decompress::huf_decompress::HUF_decompress1X_usingDTable::<false>(
                dst, cSrc, dtable,
            )
        }
    }

    /// Upstream `HUF_decompress4X_usingDTable()` runtime-flags entry point.
    #[inline]
    pub fn HUF_decompress4X_usingDTable(
        dst: &mut [u8],
        cSrc: &[u8],
        dtable: &[HUF_DTable],
        flags: i32,
    ) -> usize {
        if (flags & HUF_flags_bmi2) != 0 {
            crate::decompress::huf_decompress::HUF_decompress4X_usingDTable::<true>(
                dst, cSrc, dtable,
            )
        } else {
            crate::decompress::huf_decompress::HUF_decompress4X_usingDTable::<false>(
                dst, cSrc, dtable,
            )
        }
    }

    /// Upstream `HUF_optimalTableLog()` full flags/workspace entry point.
    ///
    /// The translated implementation keeps the cheap three-argument helper in
    /// `compress::huf_compress`; this wrapper exposes the common-header shape
    /// and validates the caller-provided workspace size before delegating to the
    /// flag-aware implementation.
    #[inline]
    pub fn HUF_optimalTableLog(
        maxTableLog: u32,
        srcSize: usize,
        maxSymbolValue: u32,
        workSpace: &mut [u8],
        table: &mut [HUF_CElt],
        count: &[u32],
        flags: i32,
    ) -> u32 {
        debug_assert!(srcSize > 1);
        debug_assert!(workSpace.len() >= HUF_CTABLE_WORKSPACE_SIZE);
        crate::compress::huf_compress::HUF_optimalTableLog_internal(
            maxTableLog,
            srcSize,
            maxSymbolValue,
            table,
            count,
            flags,
        )
    }

    /// Upstream `HUF_validateCTable()` returns an `int` boolean.
    #[inline]
    pub fn HUF_validateCTable(ctable: &[HUF_CElt], count: &[u32], maxSymbolValue: u32) -> i32 {
        crate::compress::huf_compress::HUF_validateCTable(ctable, count, maxSymbolValue) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::{fse, huf};

    #[test]
    fn fse_header_namespace_spans_common_and_compress_translation() {
        assert_eq!(fse::FSE_versionNumber(), fse::FSE_VERSION_NUMBER);
        assert_eq!(fse::FSE_VERSION_STRING, "0.9.0");
        assert_eq!(fse::FSE_DTABLE_SIZE_U32(6), 65);
        assert_eq!(
            fse::FSE_CTABLE_SIZE(6, 255),
            fse::FSE_CTABLE_SIZE_U32(6, 255) * 4
        );
        assert_eq!(fse::FSE_repeat::FSE_repeat_valid as u32, 2);
        assert!(fse::FSE_compressBound(64) >= 64);

        let norm = [-1, 1, 1, 1, 1, 1, 1, 1];
        let mut ct = vec![0; fse::FSE_CTABLE_SIZE_U32(3, 7)];
        assert_eq!(fse::FSE_buildCTable(&mut ct, &norm, 7, 3), 0);
    }

    #[test]
    fn huf_header_namespace_spans_common_compress_and_decompress_translation() {
        assert_eq!(huf::HUF_TABLELOG_MAX, 12);
        assert_eq!(huf::HUF_DTABLE_SIZE_U32(6), 65);
        assert_eq!(huf::HUF_OPTIMAL_DEPTH_THRESHOLD, 8);
        assert_eq!(huf::HUF_repeat::HUF_repeat_valid as u32, 2);
        assert!(huf::HUF_compressBound(64) >= 64);

        let _: fn(&mut [u8], usize, &mut [u32], &mut u32, &mut u32, &[u8]) -> usize =
            huf::HUF_readStats;
        let _: fn(&mut [u8], &[u8], &[huf::HUF_DTable], i32) -> usize =
            huf::HUF_decompress1X_usingDTable;
        let _: fn(&mut [u8], &[u8], &[huf::HUF_DTable], i32) -> usize =
            huf::HUF_decompress4X_usingDTable;
        let _: fn(u32, usize, u32, &mut [u8], &mut [huf::HUF_CElt], &[u32], i32) -> u32 =
            huf::HUF_optimalTableLog;
        let _: fn(&[huf::HUF_CElt], &[u32], u32) -> i32 = huf::HUF_validateCTable;
    }
}
