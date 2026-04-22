//! Translation of `lib/compress/zstd_compress_superblock.c`.
//!
//! Superblock compression splits a single block into multiple small
//! sub-blocks with independent entropy tables to better adapt to
//! local statistics. Used when `cctx.appliedParams.targetCBlockSize`
//! is set. Not yet ported — current v0.1 always emits single blocks
//! per `ZSTD_BLOCKSIZE_MAX` stride.

use crate::compress::seq_store::{MINMATCH, SeqDef, SeqStore_t, ZSTD_getSequenceLength};
use crate::compress::zstd_compress::{
    ZSTD_CCtx_params, ZSTD_entropyCTablesMetadata_t, ZSTD_entropyCTables_t,
    ZSTD_estimateBlockSize_symbolType, ZSTD_fseCTablesMetadata_t,
    ZSTD_hufCTablesMetadata_t, ZSTD_hufCTables_t,
};
use crate::decompress::zstd_decompress_block::SymbolEncodingType_e;

const BYTESCALE: usize = 256;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EstimatedBlockSize {
    pub estLitSize: usize,
    pub estBlockSize: usize,
}

/// Port of `ZSTD_compressSubBlock_literal`.
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_compressSubBlock_literal(
    hufTable: &[crate::compress::huf_compress::HUF_CElt],
    hufMetadata: &ZSTD_hufCTablesMetadata_t,
    literals: &[u8],
    dst: &mut [u8],
    bmi2: i32,
    writeEntropy: bool,
    entropyWritten: &mut i32,
) -> usize {
    use crate::common::mem::{MEM_writeLE24, MEM_writeLE32};
    use crate::compress::huf_compress::{HUF_compress1X_usingCTable, HUF_compress4X_usingCTable};
    use crate::compress::zstd_compress_literals::{
        ZSTD_compressRleLiteralsBlock, ZSTD_noCompressLiterals,
    };

    let litSize = literals.len();
    let header = if writeEntropy { 200 } else { 0 };
    let lhSize = 3 + (litSize >= (1024usize.saturating_sub(header))) as usize
        + (litSize >= (16 * 1024usize).saturating_sub(header)) as usize;
    let singleStream = lhSize == 3;
    let hType = if writeEntropy {
        hufMetadata.hType
    } else {
        SymbolEncodingType_e::set_repeat
    };
    let mut op = lhSize;
    let mut cLitSize = 0usize;

    *entropyWritten = 0;
    if litSize == 0 || hufMetadata.hType == SymbolEncodingType_e::set_basic {
        return ZSTD_noCompressLiterals(dst, literals);
    } else if hufMetadata.hType == SymbolEncodingType_e::set_rle {
        return ZSTD_compressRleLiteralsBlock(dst, literals);
    }

    if writeEntropy && hufMetadata.hType == SymbolEncodingType_e::set_compressed {
        dst[op..op + hufMetadata.hufDesSize].copy_from_slice(&hufMetadata.hufDesBuffer[..hufMetadata.hufDesSize]);
        op += hufMetadata.hufDesSize;
        cLitSize += hufMetadata.hufDesSize;
    }

    let cSize = if singleStream {
        HUF_compress1X_usingCTable(&mut dst[op..], literals, hufTable, if bmi2 != 0 { 1 } else { 0 })
    } else {
        HUF_compress4X_usingCTable(&mut dst[op..], literals, hufTable, if bmi2 != 0 { 1 } else { 0 })
    };
    if cSize == 0 || crate::common::error::ERR_isError(cSize) {
        return 0;
    }
    op += cSize;
    cLitSize += cSize;

    if !writeEntropy && cLitSize >= litSize {
        return ZSTD_noCompressLiterals(dst, literals);
    }
    if lhSize < (3 + (cLitSize >= 1024) as usize + (cLitSize >= 16 * 1024) as usize) {
        return ZSTD_noCompressLiterals(dst, literals);
    }

    match lhSize {
        3 => {
            let lhc = hType as u32 + ((singleStream as u32 ^ 1) << 2) + ((litSize as u32) << 4)
                + ((cLitSize as u32) << 14);
            MEM_writeLE24(&mut dst[..3], lhc);
        }
        4 => {
            let lhc = hType as u32 + (2 << 2) + ((litSize as u32) << 4) + ((cLitSize as u32) << 18);
            MEM_writeLE32(&mut dst[..4], lhc);
        }
        5 => {
            let lhc = hType as u32 + (3 << 2) + ((litSize as u32) << 4) + ((cLitSize as u32) << 22);
            MEM_writeLE32(&mut dst[..4], lhc);
            dst[4] = (cLitSize >> 10) as u8;
        }
        _ => unreachable!(),
    }
    *entropyWritten = 1;
    op
}

/// Port of `ZSTD_compressSubBlock_sequences`.
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_compressSubBlock_sequences(
    fseTables: &crate::compress::zstd_compress::ZSTD_fseCTables_t,
    fseMetadata: &ZSTD_fseCTablesMetadata_t,
    sequences: &[SeqDef],
    nbSeq: usize,
    llCode: &[u8],
    mlCode: &[u8],
    ofCode: &[u8],
    cctxParams: &ZSTD_CCtx_params,
    dst: &mut [u8],
    bmi2: i32,
    writeEntropy: bool,
    entropyWritten: &mut i32,
) -> usize {
    use crate::common::error::{ErrorCode, ERROR};
    use crate::common::mem::{MEM_writeLE16};
    use crate::compress::zstd_compress_sequences::ZSTD_encodeSequences;
    use crate::decompress::zstd_decompress_block::LONGNBSEQ;

    let longOffsets = (cctxParams.cParams.windowLog > 25) as i32;
    let mut op = 0usize;

    *entropyWritten = 0;
    if dst.len() < 4 {
        return ERROR(ErrorCode::DstSizeTooSmall);
    }
    if nbSeq < 128 {
        dst[op] = nbSeq as u8;
        op += 1;
    } else if nbSeq < LONGNBSEQ as usize {
        dst[op] = ((nbSeq >> 8) + 0x80) as u8;
        dst[op + 1] = nbSeq as u8;
        op += 2;
    } else {
        dst[op] = 0xFF;
        MEM_writeLE16(&mut dst[op + 1..], (nbSeq - LONGNBSEQ as usize) as u16);
        op += 3;
    }
    if nbSeq == 0 {
        return op;
    }

    let seqHead = op;
    op += 1;
    if writeEntropy {
        dst[seqHead] = ((fseMetadata.llType as u8) << 6)
            | ((fseMetadata.ofType as u8) << 4)
            | ((fseMetadata.mlType as u8) << 2);
        dst[op..op + fseMetadata.fseTablesSize]
            .copy_from_slice(&fseMetadata.fseTablesBuffer[..fseMetadata.fseTablesSize]);
        op += fseMetadata.fseTablesSize;
    } else {
        let repeat = SymbolEncodingType_e::set_repeat as u8;
        dst[seqHead] = (repeat << 6) | (repeat << 4) | (repeat << 2);
    }

    let bitstreamSize = ZSTD_encodeSequences(
        &mut dst[op..],
        &fseTables.matchlengthCTable,
        mlCode,
        &fseTables.offcodeCTable,
        ofCode,
        &fseTables.litlengthCTable,
        llCode,
        sequences,
        nbSeq,
        longOffsets,
        bmi2,
    );
    if crate::common::error::ERR_isError(bitstreamSize) {
        return bitstreamSize;
    }
    op += bitstreamSize;

    if writeEntropy && fseMetadata.lastCountSize != 0 && fseMetadata.lastCountSize + bitstreamSize < 4 {
        return 0;
    }
    if op - seqHead < 4 {
        return 0;
    }

    *entropyWritten = 1;
    op
}

/// Port of `ZSTD_compressSubBlock`.
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_compressSubBlock(
    entropy: &ZSTD_entropyCTables_t,
    entropyMetadata: &ZSTD_entropyCTablesMetadata_t,
    sequences: &[SeqDef],
    nbSeq: usize,
    literals: &[u8],
    llCode: &[u8],
    mlCode: &[u8],
    ofCode: &[u8],
    cctxParams: &ZSTD_CCtx_params,
    dst: &mut [u8],
    bmi2: i32,
    writeLitEntropy: bool,
    writeSeqEntropy: bool,
    litEntropyWritten: &mut i32,
    seqEntropyWritten: &mut i32,
    lastBlock: u32,
) -> usize {
    use crate::common::mem::MEM_writeLE24;
    use crate::decompress::zstd_decompress_block::{ZSTD_blockHeaderSize, blockType_e};

    let mut op = ZSTD_blockHeaderSize;
    let cLitSize = ZSTD_compressSubBlock_literal(
        &entropy.huf.CTable,
        &entropyMetadata.hufMetadata,
        literals,
        &mut dst[op..],
        bmi2,
        writeLitEntropy,
        litEntropyWritten,
    );
    if crate::common::error::ERR_isError(cLitSize) || cLitSize == 0 {
        return cLitSize;
    }
    op += cLitSize;

    let cSeqSize = ZSTD_compressSubBlock_sequences(
        &entropy.fse,
        &entropyMetadata.fseMetadata,
        sequences,
        nbSeq,
        llCode,
        mlCode,
        ofCode,
        cctxParams,
        &mut dst[op..],
        bmi2,
        writeSeqEntropy,
        seqEntropyWritten,
    );
    if crate::common::error::ERR_isError(cSeqSize) || cSeqSize == 0 {
        return cSeqSize;
    }
    op += cSeqSize;

    let cSize = op - ZSTD_blockHeaderSize;
    let cBlockHeader24 = lastBlock + ((blockType_e::bt_compressed as u32) << 1) + ((cSize as u32) << 3);
    MEM_writeLE24(&mut dst[..3], cBlockHeader24);
    op
}

/// Port of `ZSTD_seqDecompressedSize`.
pub fn ZSTD_seqDecompressedSize(
    seqStore: &SeqStore_t,
    sequences: &[SeqDef],
    nbSeqs: usize,
    litSize: usize,
    lastSubBlock: i32,
) -> usize {
    let mut matchLengthSum = 0usize;
    let mut litLengthSum = 0usize;
    let base = seqStore.sequences.as_ptr();
    let start = sequences.as_ptr();
    let start_idx = unsafe { start.offset_from(base) };
    debug_assert!(start_idx >= 0);
    let start_idx = start_idx as usize;
    debug_assert!(start_idx + nbSeqs <= seqStore.sequences.len());

    for n in 0..nbSeqs {
        let seqLen = ZSTD_getSequenceLength(seqStore, start_idx + n);
        litLengthSum += seqLen.litLength as usize;
        matchLengthSum += seqLen.matchLength as usize;
    }
    if lastSubBlock == 0 {
        debug_assert_eq!(litLengthSum, litSize);
    } else {
        debug_assert!(litLengthSum <= litSize);
    }
    matchLengthSum + litSize
}

/// Port of `ZSTD_compressSubBlock_multi`, adapted to the Rust port's
/// flattened block-state layout (`prevEntropy`/`nextEntropy` +
/// `prev_rep`/`next_rep`).
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_compressSubBlock_multi(
    seqStorePtr: &SeqStore_t,
    prevEntropy: &ZSTD_entropyCTables_t,
    nextEntropy: &mut ZSTD_entropyCTables_t,
    prev_rep: &[u32; 3],
    next_rep: &mut [u32; 3],
    entropyMetadata: &ZSTD_entropyCTablesMetadata_t,
    cctxParams: &ZSTD_CCtx_params,
    dst: &mut [u8],
    src: &[u8],
    bmi2: i32,
    lastBlock: u32,
    workspace: &mut [u32],
) -> usize {
    use crate::compress::seq_store::ZSTD_updateRep;
    use crate::compress::zstd_compress::ZSTD_noCompressBlock;

    let nbSeqs = seqStorePtr.sequences.len();
    let nbLiterals = seqStorePtr.literals.len();
    let mut sp = 0usize;
    let mut lp = 0usize;
    let mut ip = 0usize;
    let mut op = 0usize;
    let mut ll_code_ptr = 0usize;
    let mut ml_code_ptr = 0usize;
    let mut of_code_ptr = 0usize;
    let minTarget = crate::compress::zstd_compress::ZSTD_TARGETCBLOCKSIZE_MIN as usize;
    let targetCBlockSize = minTarget.max(cctxParams.targetCBlockSize);
    let mut writeLitEntropy = (entropyMetadata.hufMetadata.hType == SymbolEncodingType_e::set_compressed) as i32;
    let mut writeSeqEntropy = 1i32;

    if nbSeqs > 0 {
        let ebs = ZSTD_estimateSubBlockSize(
            &seqStorePtr.literals[lp..],
            &seqStorePtr.ofCode[of_code_ptr..],
            &seqStorePtr.llCode[ll_code_ptr..],
            &seqStorePtr.mlCode[ml_code_ptr..],
            nbSeqs,
            nextEntropy,
            entropyMetadata,
            workspace,
            writeLitEntropy != 0,
            writeSeqEntropy != 0,
        );
        let avgLitCost = if nbLiterals != 0 {
            (ebs.estLitSize * BYTESCALE) / nbLiterals
        } else {
            BYTESCALE
        };
        let avgSeqCost = ((ebs.estBlockSize - ebs.estLitSize) * BYTESCALE) / nbSeqs;
        let nbSubBlocks = ((ebs.estBlockSize + (targetCBlockSize / 2)) / targetCBlockSize).max(1);
        let avgBlockBudget = (ebs.estBlockSize * BYTESCALE) / nbSubBlocks;
        let mut blockBudgetSupp = 0usize;

        if ebs.estBlockSize > src.len() {
            return 0;
        }

        for n in 0..(nbSubBlocks.saturating_sub(1)) {
            let seqCount = sizeBlockSequences(
                &seqStorePtr.sequences[sp..],
                seqStorePtr.sequences.len() - sp,
                avgBlockBudget + blockBudgetSupp,
                avgLitCost,
                avgSeqCost,
                (n == 0) as i32,
            );
            if sp + seqCount == seqStorePtr.sequences.len() {
                break;
            }
            if seqCount == 0 {
                break;
            }

            let mut litEntropyWritten = 0;
            let mut seqEntropyWritten = 0;
            let litSize = countLiterals(seqStorePtr, &seqStorePtr.sequences[sp..], seqCount);
            let decompressedSize = ZSTD_seqDecompressedSize(
                seqStorePtr,
                &seqStorePtr.sequences[sp..],
                seqCount,
                litSize,
                0,
            );
            let cSize = ZSTD_compressSubBlock(
                nextEntropy,
                entropyMetadata,
                &seqStorePtr.sequences[sp..sp + seqCount],
                seqCount,
                &seqStorePtr.literals[lp..lp + litSize],
                &seqStorePtr.llCode[ll_code_ptr..ll_code_ptr + seqCount],
                &seqStorePtr.mlCode[ml_code_ptr..ml_code_ptr + seqCount],
                &seqStorePtr.ofCode[of_code_ptr..of_code_ptr + seqCount],
                cctxParams,
                &mut dst[op..],
                bmi2,
                writeLitEntropy != 0,
                writeSeqEntropy != 0,
                &mut litEntropyWritten,
                &mut seqEntropyWritten,
                0,
            );
            if crate::common::error::ERR_isError(cSize) {
                return cSize;
            }

            if cSize > 0 && cSize < decompressedSize {
                ip += decompressedSize;
                lp += litSize;
                op += cSize;
                ll_code_ptr += seqCount;
                ml_code_ptr += seqCount;
                of_code_ptr += seqCount;
                if litEntropyWritten != 0 {
                    writeLitEntropy = 0;
                }
                if seqEntropyWritten != 0 {
                    writeSeqEntropy = 0;
                }
                sp += seqCount;
                blockBudgetSupp = 0;
            }
        }
    }

    let mut litEntropyWritten = 0;
    let mut seqEntropyWritten = 0;
    let litSize = seqStorePtr.literals.len() - lp;
    let seqCount = seqStorePtr.sequences.len() - sp;
    let decompressedSize = ZSTD_seqDecompressedSize(
        seqStorePtr,
        &seqStorePtr.sequences[sp..],
        seqCount,
        litSize,
        1,
    );
    let cSize = ZSTD_compressSubBlock(
        nextEntropy,
        entropyMetadata,
        &seqStorePtr.sequences[sp..],
        seqCount,
        &seqStorePtr.literals[lp..],
        &seqStorePtr.llCode[ll_code_ptr..],
        &seqStorePtr.mlCode[ml_code_ptr..],
        &seqStorePtr.ofCode[of_code_ptr..],
        cctxParams,
        &mut dst[op..],
        bmi2,
        writeLitEntropy != 0,
        writeSeqEntropy != 0,
        &mut litEntropyWritten,
        &mut seqEntropyWritten,
        lastBlock,
    );
    if crate::common::error::ERR_isError(cSize) {
        return cSize;
    }
    if cSize > 0 && cSize < decompressedSize {
        ip += decompressedSize;
        op += cSize;
        if litEntropyWritten != 0 {
            writeLitEntropy = 0;
        }
        if seqEntropyWritten != 0 {
            writeSeqEntropy = 0;
        }
        sp += seqCount;
    }

    if writeLitEntropy != 0 {
        nextEntropy.huf = prevEntropy.huf.clone();
    }
    if writeSeqEntropy != 0 && ZSTD_needSequenceEntropyTables(&entropyMetadata.fseMetadata) != 0 {
        return 0;
    }

    if ip < src.len() {
        let cSize = ZSTD_noCompressBlock(&mut dst[op..], &src[ip..], lastBlock);
        if crate::common::error::ERR_isError(cSize) {
            return cSize;
        }
        op += cSize;
        if sp < seqStorePtr.sequences.len() {
            let mut rep = *prev_rep;
            for seq_idx in 0..sp {
                let ll0 = (ZSTD_getSequenceLength(seqStorePtr, seq_idx).litLength == 0) as u32;
                ZSTD_updateRep(&mut rep, seqStorePtr.sequences[seq_idx].offBase, ll0);
            }
            *next_rep = rep;
        }
    }

    op
}

/// Port of `ZSTD_estimateSubBlockSize_literal`.
pub fn ZSTD_estimateSubBlockSize_literal(
    literals: &[u8],
    huf: &ZSTD_hufCTables_t,
    hufMetadata: &ZSTD_hufCTablesMetadata_t,
    workspace: &mut [u32],
    writeEntropy: bool,
) -> usize {
    use crate::compress::hist::HIST_count_wksp;
    use crate::compress::huf_compress::{HUF_SYMBOLVALUE_MAX, HUF_estimateCompressedSize};

    let litSize = literals.len();
    let literalSectionHeaderSize = 3usize;

    match hufMetadata.hType {
        SymbolEncodingType_e::set_basic => litSize,
        SymbolEncodingType_e::set_rle => 1,
        SymbolEncodingType_e::set_compressed | SymbolEncodingType_e::set_repeat => {
            let mut maxSymbolValue = HUF_SYMBOLVALUE_MAX;
            let mut count = vec![0u32; maxSymbolValue as usize + 1];
            let largest = HIST_count_wksp(&mut count, &mut maxSymbolValue, literals, workspace);
            if crate::common::error::ERR_isError(largest) {
                return litSize;
            }
            let mut cLitSizeEstimate =
                HUF_estimateCompressedSize(&huf.CTable, &count, maxSymbolValue);
            if writeEntropy {
                cLitSizeEstimate += hufMetadata.hufDesSize;
            }
            cLitSizeEstimate + literalSectionHeaderSize
        }
    }
}

/// Port of `ZSTD_estimateSubBlockSize_sequences`.
pub fn ZSTD_estimateSubBlockSize_sequences(
    ofCodeTable: &[u8],
    llCodeTable: &[u8],
    mlCodeTable: &[u8],
    nbSeq: usize,
    fseTables: &crate::compress::zstd_compress::ZSTD_fseCTables_t,
    fseMetadata: &ZSTD_fseCTablesMetadata_t,
    workspace: &mut [u32],
    writeEntropy: bool,
) -> usize {
    use crate::decompress::zstd_decompress_block::{
        DefaultMaxOff, LL_bits, LL_defaultNorm, LL_defaultNormLog, MaxLL, MaxML, MaxOff,
        ML_bits, ML_defaultNorm, ML_defaultNormLog, OF_defaultNorm, OF_defaultNormLog,
    };

    const SEQUENCES_SECTION_HEADER_SIZE: usize = 3;
    if nbSeq == 0 {
        return SEQUENCES_SECTION_HEADER_SIZE;
    }

    let mut cSeqSizeEstimate = 0usize;
    cSeqSizeEstimate += ZSTD_estimateBlockSize_symbolType(
        fseMetadata.ofType,
        ofCodeTable,
        MaxOff,
        &fseTables.offcodeCTable,
        None,
        &OF_defaultNorm,
        OF_defaultNormLog,
        workspace,
    );
    cSeqSizeEstimate += ZSTD_estimateBlockSize_symbolType(
        fseMetadata.llType,
        llCodeTable,
        MaxLL,
        &fseTables.litlengthCTable,
        Some(&LL_bits),
        &LL_defaultNorm,
        LL_defaultNormLog,
        workspace,
    );
    cSeqSizeEstimate += ZSTD_estimateBlockSize_symbolType(
        fseMetadata.mlType,
        mlCodeTable,
        MaxML,
        &fseTables.matchlengthCTable,
        Some(&ML_bits),
        &ML_defaultNorm,
        ML_defaultNormLog,
        workspace,
    );
    let _ = DefaultMaxOff;
    if writeEntropy {
        cSeqSizeEstimate += fseMetadata.fseTablesSize;
    }
    cSeqSizeEstimate + SEQUENCES_SECTION_HEADER_SIZE
}

/// Port of `ZSTD_estimateSubBlockSize`.
pub fn ZSTD_estimateSubBlockSize(
    literals: &[u8],
    ofCodeTable: &[u8],
    llCodeTable: &[u8],
    mlCodeTable: &[u8],
    nbSeq: usize,
    entropy: &ZSTD_entropyCTables_t,
    entropyMetadata: &ZSTD_entropyCTablesMetadata_t,
    workspace: &mut [u32],
    writeLitEntropy: bool,
    writeSeqEntropy: bool,
) -> EstimatedBlockSize {
    let estLitSize = ZSTD_estimateSubBlockSize_literal(
        literals,
        &entropy.huf,
        &entropyMetadata.hufMetadata,
        workspace,
        writeLitEntropy,
    );
    let mut estBlockSize = ZSTD_estimateSubBlockSize_sequences(
        ofCodeTable,
        llCodeTable,
        mlCodeTable,
        nbSeq,
        &entropy.fse,
        &entropyMetadata.fseMetadata,
        workspace,
        writeSeqEntropy,
    );
    estBlockSize += estLitSize + crate::decompress::zstd_decompress_block::ZSTD_blockHeaderSize;
    EstimatedBlockSize { estLitSize, estBlockSize }
}

/// Port of `ZSTD_needSequenceEntropyTables`.
#[inline]
pub fn ZSTD_needSequenceEntropyTables(fseMetadata: &ZSTD_fseCTablesMetadata_t) -> i32 {
    if matches!(
        fseMetadata.llType,
        SymbolEncodingType_e::set_compressed | SymbolEncodingType_e::set_rle
    ) {
        return 1;
    }
    if matches!(
        fseMetadata.mlType,
        SymbolEncodingType_e::set_compressed | SymbolEncodingType_e::set_rle
    ) {
        return 1;
    }
    if matches!(
        fseMetadata.ofType,
        SymbolEncodingType_e::set_compressed | SymbolEncodingType_e::set_rle
    ) {
        return 1;
    }
    0
}

/// Port of `countLiterals`.
pub fn countLiterals(seqStore: &SeqStore_t, sp: &[SeqDef], seqCount: usize) -> usize {
    let mut total = 0usize;
    debug_assert!(seqCount <= sp.len());
    let base = seqStore.sequences.as_ptr();
    let start = sp.as_ptr();
    let start_idx = unsafe { start.offset_from(base) };
    debug_assert!(start_idx >= 0);
    let start_idx = start_idx as usize;
    debug_assert!(start_idx + seqCount <= seqStore.sequences.len());
    for n in 0..seqCount {
        total += ZSTD_getSequenceLength(seqStore, start_idx + n).litLength as usize;
    }
    total
}

/// Port of `sizeBlockSequences`.
pub fn sizeBlockSequences(
    sp: &[SeqDef],
    nbSeqs: usize,
    targetBudget: usize,
    avgLitCost: usize,
    avgSeqCost: usize,
    firstSubBlock: i32,
) -> usize {
    let mut budget = 0usize;
    debug_assert!(firstSubBlock == 0 || firstSubBlock == 1);
    debug_assert!(nbSeqs > 0);
    debug_assert!(nbSeqs <= sp.len());

    let headerSize = firstSubBlock as usize * 120 * BYTESCALE;
    budget += headerSize;

    budget += sp[0].litLength as usize * avgLitCost + avgSeqCost;
    if budget > targetBudget {
        return 1;
    }
    let mut inSize = sp[0].litLength as usize + sp[0].mlBase as usize + MINMATCH as usize;

    let mut n = 1usize;
    while n < nbSeqs {
        let currentCost = sp[n].litLength as usize * avgLitCost + avgSeqCost;
        budget += currentCost;
        inSize += sp[n].litLength as usize + sp[n].mlBase as usize + MINMATCH as usize;
        if budget > targetBudget && budget < inSize * BYTESCALE {
            break;
        }
        n += 1;
    }

    n
}

pub fn ZSTD_compressSuperBlock(
    zc: &mut crate::compress::zstd_compress::ZSTD_CCtx,
    dst: &mut [u8],
    src: &[u8],
    lastBlock: u32,
) -> usize {
    use crate::decompress::zstd_decompress_block::MaxSeq;

    if zc.seqStore.is_none() {
        return 0;
    }

    let mut entropyMetadata = ZSTD_entropyCTablesMetadata_t::default();
    let mut workspace_u32 =
        vec![0u32; crate::compress::hist::HIST_WKSP_SIZE_U32.max(MaxSeq as usize + 1)];
    let mut entropyWorkspace = vec![0u8; 4096];

    let rc = crate::compress::zstd_compress::ZSTD_buildBlockEntropyStats(
        zc.seqStore.as_mut().unwrap(),
        &zc.prevEntropy,
        &mut zc.nextEntropy,
        &zc.appliedParams,
        &mut entropyMetadata,
        &mut workspace_u32,
        &mut entropyWorkspace,
    );
    if crate::common::error::ERR_isError(rc) {
        return rc;
    }

    let seqStore = zc.seqStore.as_ref().unwrap();
    ZSTD_compressSubBlock_multi(
        seqStore,
        &zc.prevEntropy,
        &mut zc.nextEntropy,
        &zc.prev_rep,
        &mut zc.next_rep,
        &entropyMetadata,
        &zc.appliedParams,
        dst,
        src,
        0,
        lastBlock,
        &mut workspace_u32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::seq_store::{SeqStore_t, ZSTD_longLengthType_e};
    use crate::compress::zstd_compress::{
        ZSTD_CCtx_init_compressStream2, ZSTD_EndDirective, ZSTD_createCCtx,
        ZSTD_estimateBlockSize_literal, ZSTD_estimateBlockSize_sequences,
    };

    #[test]
    fn compressSuperBlock_tiny_input_falls_back_to_raw_block() {
        let mut cctx = ZSTD_createCCtx().unwrap();
        let mut dst = [0u8; 64];
        let src = b"superblock-test";
        cctx.requestedParams.compressionLevel = crate::compress::zstd_compress::ZSTD_CLEVEL_DEFAULT;
        assert_eq!(
            ZSTD_CCtx_init_compressStream2(&mut cctx, ZSTD_EndDirective::ZSTD_e_end, src.len()),
            0
        );
        let bss = crate::compress::zstd_compress::ZSTD_buildSeqStore(&mut cctx, src);
        assert!(!crate::common::error::ERR_isError(bss));
        let rc = ZSTD_compressSuperBlock(&mut cctx, &mut dst, src, 1);
        let expected =
            crate::compress::zstd_compress::ZSTD_noCompressBlock(&mut [0u8; 64], src, 1);
        assert_eq!(rc, expected);
    }

    #[test]
    fn need_sequence_entropy_tables_tracks_rle_and_compressed_modes() {
        let mut meta = ZSTD_fseCTablesMetadata_t::default();
        assert_eq!(ZSTD_needSequenceEntropyTables(&meta), 0);

        meta.llType = SymbolEncodingType_e::set_rle;
        assert_eq!(ZSTD_needSequenceEntropyTables(&meta), 1);

        meta.llType = SymbolEncodingType_e::set_basic;
        meta.ofType = SymbolEncodingType_e::set_compressed;
        assert_eq!(ZSTD_needSequenceEntropyTables(&meta), 1);
    }

    #[test]
    fn count_literals_uses_decoded_sequence_lengths() {
        let seq_store = SeqStore_t {
            sequences: vec![
                SeqDef { offBase: 4, litLength: 2, mlBase: 3 },
                SeqDef { offBase: 5, litLength: 9, mlBase: 4 },
                SeqDef { offBase: 6, litLength: 1, mlBase: 5 },
            ],
            literals: Vec::new(),
            llCode: Vec::new(),
            mlCode: Vec::new(),
            ofCode: Vec::new(),
            maxNbSeq: 8,
            maxNbLit: 0,
            longLengthType: ZSTD_longLengthType_e::ZSTD_llt_literalLength,
            longLengthPos: 1,
        };

        assert_eq!(
            countLiterals(&seq_store, &seq_store.sequences, 3),
            2 + (9 + 0x10000) + 1
        );
    }

    #[test]
    fn size_block_sequences_stops_when_budget_crosses_compressible_limit() {
        let seqs = vec![
            SeqDef { offBase: 4, litLength: 10, mlBase: 7 },
            SeqDef { offBase: 5, litLength: 30, mlBase: 9 },
            SeqDef { offBase: 6, litLength: 60, mlBase: 11 },
        ];

        let got = sizeBlockSequences(&seqs, seqs.len(), 1000, 10, 50, 0);
        assert_eq!(got, 2);
    }

    #[test]
    fn size_block_sequences_first_subblock_header_can_force_minimum_one_seq() {
        let seqs = vec![
            SeqDef { offBase: 4, litLength: 1, mlBase: 1 },
            SeqDef { offBase: 5, litLength: 1, mlBase: 1 },
        ];

        let got = sizeBlockSequences(&seqs, seqs.len(), 100, 1, 1, 1);
        assert_eq!(got, 1);
    }

    #[test]
    fn estimate_subblock_literal_uses_fixed_three_byte_header() {
        let literals = vec![b'a'; 300];
        let huf = ZSTD_hufCTables_t::default();
        let mut huf_meta = ZSTD_hufCTablesMetadata_t::default();
        huf_meta.hType = SymbolEncodingType_e::set_basic;
        let mut wksp = vec![0u32; crate::compress::hist::HIST_WKSP_SIZE_U32];

        let sub = ZSTD_estimateSubBlockSize_literal(&literals, &huf, &huf_meta, &mut wksp, false);
        let block = ZSTD_estimateBlockSize_literal(&literals, &huf, &huf_meta, &mut wksp, false);

        assert_eq!(sub, literals.len());
        assert_eq!(block, literals.len());
    }

    #[test]
    fn estimate_subblock_sequences_zero_seq_is_three_byte_header() {
        let fse = crate::compress::zstd_compress::ZSTD_fseCTables_t::default();
        let meta = ZSTD_fseCTablesMetadata_t::default();
        let mut wksp = vec![0u32; crate::compress::hist::HIST_WKSP_SIZE_U32];

        assert_eq!(
            ZSTD_estimateSubBlockSize_sequences(&[], &[], &[], 0, &fse, &meta, &mut wksp, false),
            3
        );
        assert_eq!(
            ZSTD_estimateBlockSize_sequences(&[], &[], &[], 0, &fse, &meta, &mut wksp, false),
            2
        );
    }

    #[test]
    fn estimate_subblock_size_adds_subsections_and_block_header() {
        let entropy = ZSTD_entropyCTables_t::default();
        let meta = ZSTD_entropyCTablesMetadata_t::default();
        let mut wksp = vec![0u32; crate::compress::hist::HIST_WKSP_SIZE_U32];
        let literals = b"abcde";

        let est = ZSTD_estimateSubBlockSize(
            literals,
            &[],
            &[],
            &[],
            0,
            &entropy,
            &meta,
            &mut wksp,
            false,
            false,
        );

        assert_eq!(est.estLitSize, literals.len());
        assert_eq!(
            est.estBlockSize,
            literals.len() + 3 + crate::decompress::zstd_decompress_block::ZSTD_blockHeaderSize
        );
    }

    #[test]
    fn compress_subblock_literal_raw_matches_no_compress_path() {
        let huf = ZSTD_hufCTables_t::default();
        let meta = ZSTD_hufCTablesMetadata_t::default();
        let literals = b"hello";
        let mut got = [0u8; 32];
        let mut expected = [0u8; 32];
        let mut entropy_written = -1;

        let n = ZSTD_compressSubBlock_literal(
            &huf.CTable,
            &meta,
            literals,
            &mut got,
            0,
            false,
            &mut entropy_written,
        );
        let m = crate::compress::zstd_compress_literals::ZSTD_noCompressLiterals(&mut expected, literals);

        assert_eq!(entropy_written, 0);
        assert_eq!(n, m);
        assert_eq!(&got[..n], &expected[..m]);
    }

    #[test]
    fn compress_subblock_sequences_zero_seq_writes_count_only() {
        let fse = crate::compress::zstd_compress::ZSTD_fseCTables_t::default();
        let meta = ZSTD_fseCTablesMetadata_t::default();
        let params = ZSTD_CCtx_params::default();
        let mut dst = [0u8; 16];
        let mut entropy_written = -1;

        let n = ZSTD_compressSubBlock_sequences(
            &fse,
            &meta,
            &[],
            0,
            &[],
            &[],
            &[],
            &params,
            &mut dst,
            0,
            false,
            &mut entropy_written,
        );

        assert_eq!(n, 1);
        assert_eq!(dst[0], 0);
        assert_eq!(entropy_written, 0);
    }

    #[test]
    fn compress_subblock_with_raw_literals_and_no_sequences_builds_compressed_block_header() {
        let entropy = ZSTD_entropyCTables_t::default();
        let meta = ZSTD_entropyCTablesMetadata_t::default();
        let params = ZSTD_CCtx_params::default();
        let mut dst = [0u8; 64];
        let mut lit_entropy_written = -1;
        let mut seq_entropy_written = -1;

        let n = ZSTD_compressSubBlock(
            &entropy,
            &meta,
            &[],
            0,
            b"abc",
            &[],
            &[],
            &[],
            &params,
            &mut dst,
            0,
            false,
            false,
            &mut lit_entropy_written,
            &mut seq_entropy_written,
            1,
        );

        assert_eq!(n, 3 + 4 + 1);
        assert_eq!(lit_entropy_written, 0);
        assert_eq!(seq_entropy_written, 0);
        assert_eq!(dst[0] & 1, 1);
    }

    #[test]
    fn seq_decompressed_size_adds_match_lengths_and_passed_literal_budget() {
        let seq_store = SeqStore_t {
            sequences: vec![
                SeqDef { offBase: 4, litLength: 2, mlBase: 3 },
                SeqDef { offBase: 5, litLength: 4, mlBase: 7 },
            ],
            literals: Vec::new(),
            llCode: Vec::new(),
            mlCode: Vec::new(),
            ofCode: Vec::new(),
            maxNbSeq: 8,
            maxNbLit: 0,
            longLengthType: ZSTD_longLengthType_e::ZSTD_llt_none,
            longLengthPos: 0,
        };

        let got = ZSTD_seqDecompressedSize(&seq_store, &seq_store.sequences, 2, 6, 0);
        assert_eq!(got, 6 + (3 + 3) + (7 + 3));
    }

    #[test]
    fn compress_subblock_multi_no_sequences_falls_back_to_single_raw_block() {
        let seq_store = SeqStore_t {
            sequences: Vec::new(),
            literals: b"hello superblock".to_vec(),
            llCode: Vec::new(),
            mlCode: Vec::new(),
            ofCode: Vec::new(),
            maxNbSeq: 0,
            maxNbLit: 0,
            longLengthType: ZSTD_longLengthType_e::ZSTD_llt_none,
            longLengthPos: 0,
        };
        let prev_entropy = ZSTD_entropyCTables_t::default();
        let mut next_entropy = ZSTD_entropyCTables_t::default();
        let meta = ZSTD_entropyCTablesMetadata_t::default();
        let params = ZSTD_CCtx_params::default();
        let mut dst = [0u8; 128];
        let src = b"hello superblock";
        let mut wksp = vec![0u32; crate::compress::hist::HIST_WKSP_SIZE_U32];
        let prev_rep = [1, 4, 8];
        let mut next_rep = [1, 4, 8];

        let got = ZSTD_compressSubBlock_multi(
            &seq_store,
            &prev_entropy,
            &mut next_entropy,
            &prev_rep,
            &mut next_rep,
            &meta,
            &params,
            &mut dst,
            src,
            0,
            1,
            &mut wksp,
        );
        let expected = crate::compress::zstd_compress::ZSTD_noCompressBlock(&mut [0u8; 128], src, 1);

        assert_eq!(got, expected);
        assert_eq!(next_rep, prev_rep);
    }

    #[test]
    fn compress2_roundtrips_when_target_block_size_enabled() {
        use crate::decompress::zstd_decompress::ZSTD_decompress;

        let src = b"aaaaabbbbbcccccdddddeeeeeaaaaabbbbbcccccdddddeeeee";
        let mut cctx = ZSTD_createCCtx().unwrap();
        cctx.requestedParams.compressionLevel = 5;
        cctx.requestedParams.targetCBlockSize = 32;
        let mut compressed =
            vec![0u8; crate::compress::zstd_compress::ZSTD_compressBound(src.len())];
        let csize = crate::compress::zstd_compress::ZSTD_compress2(&mut cctx, &mut compressed, src);
        assert!(csize > 0);

        let mut roundtrip = vec![0u8; src.len()];
        let dsize = ZSTD_decompress(&mut roundtrip, &compressed[..csize]);
        assert_eq!(dsize, src.len());
        assert_eq!(roundtrip, src);
    }
}
