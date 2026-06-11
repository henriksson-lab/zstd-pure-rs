//! `lib/decompress/` translation.

pub mod huf_decompress;
pub mod zstd_ddict;
pub mod zstd_decompress;
pub mod zstd_decompress_block;

pub use zstd_ddict::{
    ZSTD_DDict, ZSTD_DDict_dictContent, ZSTD_DDict_dictSize, ZSTD_copyDDictParameters,
    ZSTD_createDDict, ZSTD_createDDict_advanced, ZSTD_createDDict_byReference,
    ZSTD_dictContentType_e, ZSTD_dictLoadMethod_e, ZSTD_estimateDDictSize, ZSTD_freeDDict,
    ZSTD_getDictID_fromDDict, ZSTD_getDictID_fromDict, ZSTD_sizeof_DDict,
};
pub use zstd_decompress::{
    ZSTD_DCtx, ZSTD_DCtx_getParameter, ZSTD_DCtx_loadDictionary, ZSTD_DCtx_loadDictionary_advanced,
    ZSTD_DCtx_loadDictionary_byReference, ZSTD_DCtx_refDDict, ZSTD_DCtx_refPrefix,
    ZSTD_DCtx_refPrefix_advanced, ZSTD_DCtx_reset, ZSTD_DCtx_setFormat, ZSTD_DCtx_setMaxWindowSize,
    ZSTD_DCtx_setParameter, ZSTD_DResetDirective, ZSTD_DStream, ZSTD_DStreamInSize,
    ZSTD_DStreamOutSize, ZSTD_FrameHeader, ZSTD_FrameType_e, ZSTD_copyDCtx, ZSTD_createDCtx,
    ZSTD_createDCtx_advanced, ZSTD_createDStream, ZSTD_createDStream_advanced,
    ZSTD_dParam_getBounds, ZSTD_dParam_withinBounds, ZSTD_dParameter, ZSTD_decodingBufferSize_min,
    ZSTD_decompress, ZSTD_decompressBegin, ZSTD_decompressBegin_usingDDict,
    ZSTD_decompressBegin_usingDict, ZSTD_decompressBlock, ZSTD_decompressBlock_deprecated,
    ZSTD_decompressBound, ZSTD_decompressContinue, ZSTD_decompressDCtx, ZSTD_decompressStream,
    ZSTD_decompressStream_simpleArgs, ZSTD_decompress_usingDDict, ZSTD_decompress_usingDict,
    ZSTD_decompressionMargin, ZSTD_dictUses_e, ZSTD_estimateDCtxSize, ZSTD_estimateDStreamSize,
    ZSTD_estimateDStreamSize_fromFrame, ZSTD_findDecompressedSize, ZSTD_findFrameCompressedSize,
    ZSTD_findFrameCompressedSize_advanced, ZSTD_findFrameSizeInfo, ZSTD_format_e,
    ZSTD_frameHeaderSize, ZSTD_frameSizeInfo, ZSTD_freeDCtx, ZSTD_freeDStream,
    ZSTD_getDecompressedSize, ZSTD_getDictID_fromFrame, ZSTD_getFrameContentSize,
    ZSTD_getFrameHeader, ZSTD_getFrameHeader_advanced, ZSTD_initDStream,
    ZSTD_initDStream_usingDDict, ZSTD_initDStream_usingDict, ZSTD_initStaticDCtx,
    ZSTD_initStaticDDict, ZSTD_initStaticDStream, ZSTD_insertBlock, ZSTD_isFrame,
    ZSTD_isSkippableFrame, ZSTD_nextInputType, ZSTD_nextInputType_e, ZSTD_nextSrcSizeToDecompress,
    ZSTD_readSkippableFrame, ZSTD_resetDStream, ZSTD_sizeof_DCtx, ZSTD_sizeof_DStream,
    ZSTD_CONTENTSIZE_ERROR, ZSTD_CONTENTSIZE_UNKNOWN, ZSTD_DECOMPRESSION_MARGIN,
    ZSTD_FRAMEHEADERSIZE_MIN, ZSTD_FRAMEHEADERSIZE_PREFIX, ZSTD_FRAMEIDSIZE, ZSTD_MAGICNUMBER,
    ZSTD_MAGIC_DICTIONARY, ZSTD_MAGIC_SKIPPABLE_MASK, ZSTD_MAGIC_SKIPPABLE_START,
    ZSTD_SKIPPABLEHEADERSIZE, ZSTD_WINDOWLOG_ABSOLUTEMIN, ZSTD_WINDOWLOG_LIMIT_DEFAULT,
    ZSTD_WINDOWLOG_MAX_32, ZSTD_WINDOWLOG_MAX_64,
};
pub use zstd_decompress_block::{ZSTD_BLOCKHEADERSIZE, ZSTD_BLOCKSIZELOG_MAX, ZSTD_BLOCKSIZE_MAX};

#[cfg(test)]
mod tests {
    use super::{
        ZSTD_DCtx, ZSTD_DDict, ZSTD_createDDict, ZSTD_decompress, ZSTD_freeDDict,
        ZSTD_getFrameContentSize, ZSTD_isFrame, ZSTD_BLOCKHEADERSIZE,
    };

    #[test]
    fn decompress_module_reexports_public_api_surface() {
        let _dctx = ZSTD_DCtx::new();
        let ddict: Option<Box<ZSTD_DDict>> = ZSTD_createDDict(b"dict");
        assert_eq!(ZSTD_freeDDict(ddict), 0);
        assert_eq!(ZSTD_BLOCKHEADERSIZE, 3);

        let frame = include_bytes!("../../tests/fixtures/empty_l1.zst");
        assert_eq!(ZSTD_isFrame(frame), 1);
        assert_eq!(ZSTD_getFrameContentSize(frame), 0);
        assert_eq!(ZSTD_decompress(&mut [], frame), 0);
    }
}
