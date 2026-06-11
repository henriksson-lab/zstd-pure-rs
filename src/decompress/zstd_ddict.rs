//! Translation of `lib/decompress/zstd_ddict.c`. Digested dictionary
//! for decompression.
//!
//! Scope note: raw-content dictionaries (`ZSTD_dct_rawContent`) and
//! zstd-format dictionaries prefixed with `ZSTD_MAGIC_DICTIONARY` are
//! accepted. The Rust port keeps the original dictionary bytes in the
//! DDict and reparses entropy tables when attaching it to a DCtx.

use crate::common::error::{ErrorCode, ERROR};

/// Mirror of upstream `ZSTD_dictLoadMethod_e`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZSTD_dictLoadMethod_e {
    ZSTD_dlm_byCopy = 0,
    ZSTD_dlm_byRef = 1,
}

/// Mirror of upstream `ZSTD_dictContentType_e`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZSTD_dictContentType_e {
    #[default]
    ZSTD_dct_auto = 0,
    ZSTD_dct_rawContent = 1,
    ZSTD_dct_fullDict = 2,
}

/// Digested dictionary. By-copy dictionaries own their content in
/// `dictBuffer`; by-reference dictionaries mirror upstream and keep a
/// raw pointer to caller-owned bytes.
pub struct ZSTD_DDict {
    dictBuffer: Vec<u8>,
    entropyBuffer: Vec<u8>,
    dictContent: *const u8,
    dictSize: usize,
    rawContent: *const u8,
    rawSize: usize,
    dictID: u32,
    entropyPresent: u32,
}

impl ZSTD_DDict {
    pub(crate) fn empty() -> Self {
        Self {
            dictBuffer: Vec::new(),
            entropyBuffer: Vec::new(),
            dictContent: core::ptr::null(),
            dictSize: 0,
            rawContent: core::ptr::null(),
            rawSize: 0,
            dictID: 0,
            entropyPresent: 0,
        }
    }

    pub(crate) fn borrowed(dict: &[u8]) -> Self {
        let ptr = if dict.is_empty() {
            core::ptr::null()
        } else {
            dict.as_ptr()
        };
        Self {
            dictBuffer: Vec::new(),
            entropyBuffer: Vec::new(),
            dictContent: ptr,
            dictSize: dict.len(),
            rawContent: ptr,
            rawSize: dict.len(),
            dictID: 0,
            entropyPresent: 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn raw_with_dict_id(dictID: u32, content: &[u8]) -> Self {
        let mut ddict = Self {
            dictBuffer: content.to_vec(),
            entropyBuffer: Vec::new(),
            dictContent: core::ptr::null(),
            dictSize: content.len(),
            rawContent: core::ptr::null(),
            rawSize: content.len(),
            dictID,
            entropyPresent: 0,
        };
        ddict.dictContent = if ddict.dictBuffer.is_empty() {
            core::ptr::null()
        } else {
            ddict.dictBuffer.as_ptr()
        };
        ddict.rawContent = ddict.dictContent;
        ddict
    }
}

/// Port of `ZSTD_DDict_dictContent`.
#[inline]
pub fn ZSTD_DDict_dictContent(ddict: &ZSTD_DDict) -> &[u8] {
    if ddict.dictSize == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(ddict.dictContent, ddict.dictSize) }
    }
}

#[inline]
pub(crate) fn ZSTD_DDict_rawContent(ddict: &ZSTD_DDict) -> &[u8] {
    if ddict.rawSize == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(ddict.rawContent, ddict.rawSize) }
    }
}

/// Port of `ZSTD_DDict_dictSize`.
#[inline]
pub fn ZSTD_DDict_dictSize(ddict: &ZSTD_DDict) -> usize {
    ddict.dictSize
}

/// Port of `ZSTD_getDictID_fromDDict`.
#[inline]
pub fn ZSTD_getDictID_fromDDict(ddict: &ZSTD_DDict) -> u32 {
    ddict.dictID
}

#[inline]
pub(crate) fn ZSTD_DDict_entropyPresent(ddict: &ZSTD_DDict) -> u32 {
    ddict.entropyPresent
}

#[inline]
pub(crate) fn ZSTD_DDict_originalContent(ddict: &ZSTD_DDict) -> &[u8] {
    if ddict.entropyBuffer.is_empty() {
        ZSTD_DDict_dictContent(ddict)
    } else {
        &ddict.entropyBuffer
    }
}

/// Port of `ZSTD_getDictID_fromDict`. Reads the 4-byte dictID from a
/// raw dictionary buffer immediately after the magic bytes. Returns
/// 0 when the dict is shorter than 8 bytes or doesn't start with the
/// `ZSTD_MAGIC_DICTIONARY` sentinel — upstream callers treat that as
/// "raw-content dictionary".
pub fn ZSTD_getDictID_fromDict(dict: &[u8]) -> u32 {
    use crate::common::mem::MEM_readLE32;
    use crate::decompress::zstd_decompress::{ZSTD_FRAMEIDSIZE, ZSTD_MAGIC_DICTIONARY};
    if dict.len() < 8 {
        return 0;
    }
    if MEM_readLE32(&dict[..4]) != ZSTD_MAGIC_DICTIONARY {
        return 0;
    }
    MEM_readLE32(&dict[ZSTD_FRAMEIDSIZE..ZSTD_FRAMEIDSIZE + 4])
}

/// Port of `ZSTD_estimateDDictSize`. Upper bound on the DDict
/// footprint for a given dict size and load method. `ZSTD_dlm_byRef`
/// saves the dict-copy bytes since the DDict would only reference
/// the caller's buffer — v0.1 always copies, but the estimate
/// honors the choice.
pub fn ZSTD_estimateDDictSize(dictSize: usize, dictLoadMethod: ZSTD_dictLoadMethod_e) -> usize {
    let copied = if dictLoadMethod == ZSTD_dictLoadMethod_e::ZSTD_dlm_byRef {
        0
    } else {
        dictSize
    };
    core::mem::size_of::<ZSTD_DDict>() + copied
}

/// Port of `ZSTD_sizeof_DDict`.
pub fn ZSTD_sizeof_DDict(ddict: Option<&ZSTD_DDict>) -> usize {
    match ddict {
        None => 0,
        Some(d) => {
            core::mem::size_of::<ZSTD_DDict>()
                + if d.dictBuffer.is_empty() {
                    0
                } else {
                    d.dictSize
                }
                + d.entropyBuffer.len()
        }
    }
}

/// Port of `ZSTD_loadEntropy_intoDDict`.
/// Returns 0 on success or an error code. Magic-tagged zstd-format
/// dictionaries are parsed with the shared decoder entropy loader.
pub(crate) fn ZSTD_loadEntropy_intoDDict(
    ddict: &mut ZSTD_DDict,
    dictContentType: ZSTD_dictContentType_e,
) -> usize {
    use crate::common::error::ERR_isError;
    use crate::common::mem::MEM_readLE32;
    use crate::decompress::zstd_decompress::{
        ZSTD_loadDEntropy, ZSTD_FRAMEIDSIZE, ZSTD_MAGIC_DICTIONARY,
    };
    use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

    ddict.dictID = 0;
    ddict.entropyPresent = 0;
    ddict.entropyBuffer.clear();
    ddict.rawContent = ddict.dictContent;
    ddict.rawSize = ddict.dictSize;
    if dictContentType == ZSTD_dictContentType_e::ZSTD_dct_rawContent {
        return 0;
    }

    if ddict.dictSize < 8 {
        if dictContentType == ZSTD_dictContentType_e::ZSTD_dct_fullDict {
            return ERROR(ErrorCode::DictionaryCorrupted);
        }
        return 0; // pure content mode
    }
    let dictID = {
        let dict = ZSTD_DDict_dictContent(ddict);
        let magic = MEM_readLE32(&dict[..4]);
        if magic != ZSTD_MAGIC_DICTIONARY {
            if dictContentType == ZSTD_dictContentType_e::ZSTD_dct_fullDict {
                return ERROR(ErrorCode::DictionaryCorrupted);
            }
            return 0;
        }

        MEM_readLE32(&dict[ZSTD_FRAMEIDSIZE..ZSTD_FRAMEIDSIZE + 4])
    };
    ddict.dictID = dictID;
    let entropy_size = {
        let dict = ZSTD_DDict_dictContent(ddict);
        let mut tmp_dctx = ZSTD_DCtx::new();
        let mut rep = [0u32; 3];
        ZSTD_loadDEntropy(&mut tmp_dctx, &mut rep, dict)
    };
    if ERR_isError(entropy_size) {
        ddict.entropyPresent = 0;
        return ERROR(ErrorCode::DictionaryCorrupted);
    }

    ddict.entropyPresent = 1;
    if ddict.dictBuffer.is_empty() {
        ddict.entropyBuffer = ZSTD_DDict_dictContent(ddict).to_vec();
    }
    ddict.rawContent = unsafe { ddict.dictContent.add(entropy_size) };
    ddict.rawSize = ddict.dictSize - entropy_size;
    0
}

/// Port of `ZSTD_initDDict_internal`. Initializes an already-allocated
/// DDict from caller bytes.
pub(crate) fn ZSTD_initDDict_internal(
    ddict: &mut ZSTD_DDict,
    dict: &[u8],
    dictLoadMethod: ZSTD_dictLoadMethod_e,
    dictContentType: ZSTD_dictContentType_e,
) -> usize {
    if dictLoadMethod == ZSTD_dictLoadMethod_e::ZSTD_dlm_byRef || dict.is_empty() {
        ddict.dictBuffer = Vec::new();
        ddict.dictContent = if dict.is_empty() {
            core::ptr::null()
        } else {
            dict.as_ptr()
        };
    } else {
        ddict.dictBuffer = dict.to_vec();
        ddict.dictContent = ddict.dictBuffer.as_ptr();
    };
    ddict.entropyBuffer = Vec::new();
    ddict.dictSize = dict.len();
    ddict.rawContent = if dict.is_empty() {
        core::ptr::null()
    } else {
        ddict.dictContent
    };
    ddict.rawSize = dict.len();
    ddict.dictID = 0;
    ddict.entropyPresent = 0;
    ZSTD_loadEntropy_intoDDict(ddict, dictContentType)
}

/// Port of `ZSTD_createDDict_advanced`.
pub fn ZSTD_createDDict_advanced(
    dict: &[u8],
    dictLoadMethod: ZSTD_dictLoadMethod_e,
    dictContentType: ZSTD_dictContentType_e,
) -> Option<Box<ZSTD_DDict>> {
    let mut ddict = Box::new(ZSTD_DDict::empty());
    let rc = ZSTD_initDDict_internal(&mut ddict, dict, dictLoadMethod, dictContentType);
    if crate::common::error::ERR_isError(rc) {
        return None;
    }
    Some(ddict)
}

/// Port of `ZSTD_createDDict`. Copies dict content.
pub fn ZSTD_createDDict(dictBuffer: &[u8]) -> Option<Box<ZSTD_DDict>> {
    ZSTD_createDDict_advanced(
        dictBuffer,
        ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy,
        ZSTD_dictContentType_e::ZSTD_dct_auto,
    )
}

/// Port of `ZSTD_createDDict_byReference`.
///
/// Mirrors the C API lifetime contract: `dictBuffer` must outlive the
/// returned DDict.
pub fn ZSTD_createDDict_byReference(dictBuffer: &[u8]) -> Option<Box<ZSTD_DDict>> {
    ZSTD_createDDict_advanced(
        dictBuffer,
        ZSTD_dictLoadMethod_e::ZSTD_dlm_byRef,
        ZSTD_dictContentType_e::ZSTD_dct_auto,
    )
}

/// Port of `ZSTD_freeDDict`. Rust's `Box::drop` is the freer; we
/// preserve the `size_t` return of the upstream signature (0 on success).
pub fn ZSTD_freeDDict(_ddict: Option<Box<ZSTD_DDict>>) -> usize {
    0
}

/// Port of `ZSTD_copyDDictParameters` (zstd_ddict.c:58). Seeds a DCtx
/// with a DDict's parameters. Upstream sets the prefix/virtualStart/
/// dictEnd/previousDstEnd pointers that it uses to resolve back-refs
/// into the dict; v0.1's decoder works by materializing the dict into
/// `stream_dict`, so we copy the same exposed DDict content there and
/// reparse the DDict's original bytes when serialized entropy is present.
pub fn ZSTD_copyDDictParameters(
    dctx: &mut crate::decompress::zstd_decompress_block::ZSTD_DCtx,
    ddict: &ZSTD_DDict,
) {
    use crate::common::error::ERR_isError;
    use crate::decompress::zstd_decompress::ZSTD_loadDEntropy;

    dctx.stream_dict = ZSTD_DDict_dictContent(ddict).to_vec();
    let dict_start = ddict.dictContent as usize;
    let dict_end = dict_start.wrapping_add(ddict.dictSize);
    dctx.prefixStart = Some(dict_start);
    dctx.virtualStart = Some(dict_start);
    dctx.dictEnd = Some(dict_end);
    dctx.previousDstEnd = Some(dict_end);
    dctx.dictID = ddict.dictID;
    if ddict.entropyPresent != 0 {
        let mut rep = [0u32; 3];
        let rc = ZSTD_loadDEntropy(dctx, &mut rep, ZSTD_DDict_originalContent(ddict));
        if !ERR_isError(rc) {
            dctx.ddict_rep = rep;
        }
        dctx.litEntropy = 1;
        dctx.fseEntropy = 1;
    } else {
        dctx.litEntropy = 0;
        dctx.fseEntropy = 0;
    }
}

#[cfg(test)]
mod byref_tests {
    use super::*;

    #[test]
    fn copyDDictParameters_propagates_dictID_and_fseEntropy() {
        // Parity with upstream `ZSTD_copyDDictParameters`
        // (zstd_ddict.c:58): copies `dictID` + sets both
        // `litEntropy` AND `fseEntropy` from `ddict.entropyPresent`.
        // Previously our port only wrote `stream_dict` + `litEntropy`.
        use crate::decompress::zstd_decompress_block::ZSTD_DCtx;
        let mut ddict = ZSTD_DDict::raw_with_dict_id(0xDEAD_BEEF, b"copy-params-dict");
        ddict.entropyPresent = 1;
        let mut dctx = ZSTD_DCtx::new();
        ZSTD_copyDDictParameters(&mut dctx, &ddict);
        assert_eq!(dctx.dictID, 0xDEAD_BEEF);
        assert_eq!(dctx.litEntropy, 1);
        assert_eq!(dctx.fseEntropy, 1);

        // When entropy isn't pre-digested, both flags clear.
        ddict.entropyPresent = 0;
        ddict.dictID = 0;
        let mut dctx = ZSTD_DCtx::new();
        ZSTD_copyDDictParameters(&mut dctx, &ddict);
        assert_eq!(dctx.dictID, 0);
        assert_eq!(dctx.litEntropy, 0);
        assert_eq!(dctx.fseEntropy, 0);
    }

    #[test]
    fn createDDict_byReference_matches_regular_creator_content() {
        let dict = b"byref-equivalence-test-dict ".repeat(4);
        let by_copy = ZSTD_createDDict(&dict).expect("by-copy");
        let by_ref = ZSTD_createDDict_byReference(&dict).expect("by-ref");
        assert_eq!(
            ZSTD_DDict_dictContent(&by_copy),
            ZSTD_DDict_dictContent(&by_ref)
        );
        assert_eq!(ZSTD_DDict_dictSize(&by_copy), ZSTD_DDict_dictSize(&by_ref));
    }

    #[test]
    fn createDDict_byReference_references_caller_bytes() {
        let mut dict = b"byref-copy-contract-test-dict ".repeat(4);
        let ddict = ZSTD_createDDict_byReference(&dict).expect("by-ref");

        dict.fill(b'X');

        assert_eq!(ZSTD_DDict_dictContent(&ddict), dict.as_slice());
    }

    #[test]
    fn freeDDict_accepts_none_and_some_without_panic() {
        // Free-None no-op, free-Some drops the Box — both return 0.
        assert_eq!(ZSTD_freeDDict(None), 0);
        let ddict = ZSTD_createDDict(&[1u8, 2, 3]).unwrap();
        assert_eq!(ZSTD_freeDDict(Some(ddict)), 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_minimal_zstd_dict(dictID: u32, content: &[u8]) -> Vec<u8> {
        use crate::common::mem::MEM_writeLE32;
        use crate::compress::fse_compress::FSE_writeNCount;
        use crate::compress::huf_compress::{
            HUF_buildCTable_wksp, HUF_readCTableHeader, HUF_writeCTable,
            HUF_CTABLE_WORKSPACE_SIZE_U32,
        };
        use crate::decompress::zstd_decompress::ZSTD_MAGIC_DICTIONARY;
        use crate::decompress::zstd_decompress_block::{
            LL_defaultNorm, LL_defaultNormLog, ML_defaultNorm, ML_defaultNormLog, MaxLL, MaxML,
            MaxOff, OF_defaultNorm, OF_defaultNormLog,
        };

        let mut out = Vec::new();
        let mut word = [0u8; 4];
        MEM_writeLE32(&mut word, ZSTD_MAGIC_DICTIONARY);
        out.extend_from_slice(&word);
        MEM_writeLE32(&mut word, dictID);
        out.extend_from_slice(&word);

        let mut count = [0u32; 256];
        for &b in content {
            count[b as usize] += 1;
        }
        for c in count.iter_mut().take(16) {
            if *c == 0 {
                *c = 1;
            }
        }
        let maxSymbolValue = count
            .iter()
            .enumerate()
            .rposition(|(_, &c)| c > 0)
            .unwrap_or(0) as u32;
        let totalCount = count.iter().sum::<u32>() as usize;
        let tableLog =
            crate::compress::huf_compress::HUF_optimalTableLog(11, totalCount, maxSymbolValue);
        let mut ct = vec![crate::compress::huf_compress::HUF_CElt::default(); 257];
        let mut wksp = vec![0u32; HUF_CTABLE_WORKSPACE_SIZE_U32];
        let rc = HUF_buildCTable_wksp(&mut ct, &count, maxSymbolValue, tableLog, &mut wksp);
        assert!(!crate::common::error::ERR_isError(rc));
        let tableLog = HUF_readCTableHeader(&ct).tableLog as u32;
        let mut huf_hdr = vec![0u8; 512];
        let written = HUF_writeCTable(&mut huf_hdr, &ct, maxSymbolValue, tableLog);
        assert!(!crate::common::error::ERR_isError(written));
        out.extend_from_slice(&huf_hdr[..written]);

        let mut fse_buf = vec![0u8; 256];
        let written = FSE_writeNCount(&mut fse_buf, &OF_defaultNorm, MaxOff, OF_defaultNormLog);
        assert!(!crate::common::error::ERR_isError(written));
        out.extend_from_slice(&fse_buf[..written]);
        let written = FSE_writeNCount(&mut fse_buf, &ML_defaultNorm, MaxML, ML_defaultNormLog);
        assert!(!crate::common::error::ERR_isError(written));
        out.extend_from_slice(&fse_buf[..written]);
        let written = FSE_writeNCount(&mut fse_buf, &LL_defaultNorm, MaxLL, LL_defaultNormLog);
        assert!(!crate::common::error::ERR_isError(written));
        out.extend_from_slice(&fse_buf[..written]);

        let safe = (content.len() as u32).clamp(1, 8);
        for rep in [safe, safe.saturating_sub(1).max(1), 1u32] {
            MEM_writeLE32(&mut word, rep);
            out.extend_from_slice(&word);
        }
        out.extend_from_slice(content);
        out
    }

    #[test]
    fn dict_method_and_contentType_discriminants_match_upstream() {
        // `ZSTD_dictLoadMethod_e` and `ZSTD_dictContentType_e` are
        // both part of the public advanced-API surface (consumed by
        // ZSTD_createCDict_advanced, ZSTD_CCtx_loadDictionary_advanced,
        // etc.). Upstream pins them at 0/1 and 0/1/2 respectively.
        // Accidental reordering would silently mis-route dict handling.
        assert_eq!(ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy as u32, 0);
        assert_eq!(ZSTD_dictLoadMethod_e::ZSTD_dlm_byRef as u32, 1);
        assert_eq!(ZSTD_dictContentType_e::ZSTD_dct_auto as u32, 0);
        assert_eq!(ZSTD_dictContentType_e::ZSTD_dct_rawContent as u32, 1);
        assert_eq!(ZSTD_dictContentType_e::ZSTD_dct_fullDict as u32, 2);
    }

    #[test]
    fn estimateDDictSize_byCopy_vs_byRef() {
        let by_copy = ZSTD_estimateDDictSize(1024, ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy);
        let by_ref = ZSTD_estimateDDictSize(1024, ZSTD_dictLoadMethod_e::ZSTD_dlm_byRef);
        assert_eq!(by_copy - by_ref, 1024);
        // Both include the struct overhead.
        assert_eq!(by_ref, core::mem::size_of::<ZSTD_DDict>());
    }

    #[test]
    fn getDictID_fromDict_returns_zero_for_raw_dict() {
        assert_eq!(ZSTD_getDictID_fromDict(b"no-magic-here"), 0);
        assert_eq!(ZSTD_getDictID_fromDict(&[1u8, 2, 3]), 0);
    }

    #[test]
    fn getDictID_fromDict_reads_id_after_magic() {
        use crate::common::mem::MEM_writeLE32;
        use crate::decompress::zstd_decompress::ZSTD_MAGIC_DICTIONARY;
        let mut buf = vec![0u8; 12];
        MEM_writeLE32(&mut buf[..4], ZSTD_MAGIC_DICTIONARY);
        MEM_writeLE32(&mut buf[4..8], 0xDEADBEEF);
        assert_eq!(ZSTD_getDictID_fromDict(&buf), 0xDEADBEEF);
    }

    #[test]
    fn getDictID_fromDDict_returns_preloaded_dict_id() {
        // Upstream `ZSTD_getDictID_fromDDict()` returns `ddict->dictID`.
        let ddict = ZSTD_DDict::raw_with_dict_id(0xDEAD_BEEF, b"raw-content");
        assert_eq!(ZSTD_getDictID_fromDDict(&ddict), 0xDEAD_BEEF);
    }

    #[test]
    fn create_raw_dict_exposes_content() {
        let dict_bytes = b"arbitrary-dict-content-that-isnt-a-real-dict";
        let ddict = ZSTD_createDDict(dict_bytes).expect("raw dict");
        assert_eq!(ZSTD_DDict_dictSize(&ddict), dict_bytes.len());
        assert_eq!(ZSTD_DDict_dictContent(&ddict), &dict_bytes[..]);
        // No magic prefix → dictID stays 0, entropy absent.
        assert_eq!(ZSTD_getDictID_fromDDict(&ddict), 0);
        assert_eq!(ddict.entropyPresent, 0);
    }

    #[test]
    fn initDDict_internal_reinitializes_existing_ddict() {
        let first = b"first-ddict-content";
        let second = b"second-ddict-content";
        let mut ddict = ZSTD_DDict::empty();

        assert_eq!(
            ZSTD_initDDict_internal(
                &mut ddict,
                first,
                ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy,
                ZSTD_dictContentType_e::ZSTD_dct_auto,
            ),
            0
        );
        assert_eq!(ZSTD_DDict_dictContent(&ddict), first);

        assert_eq!(
            ZSTD_initDDict_internal(
                &mut ddict,
                second,
                ZSTD_dictLoadMethod_e::ZSTD_dlm_byRef,
                ZSTD_dictContentType_e::ZSTD_dct_rawContent,
            ),
            0
        );
        assert_eq!(ZSTD_DDict_dictContent(&ddict), second);
        assert_eq!(ZSTD_DDict_dictSize(&ddict), second.len());
        assert_eq!(
            ZSTD_sizeof_DDict(Some(&ddict)),
            core::mem::size_of::<ZSTD_DDict>()
        );
    }

    #[test]
    fn create_too_short_dict_in_raw_mode_is_accepted() {
        // In `auto` mode a <8 byte buffer is accepted as pure content.
        let ddict = ZSTD_createDDict(&[1u8, 2, 3]).expect("short dict ok in auto");
        assert_eq!(ZSTD_DDict_dictSize(&ddict), 3);
    }

    #[test]
    fn create_full_dict_mode_rejects_short_input() {
        let rc = ZSTD_createDDict_advanced(
            &[1u8, 2, 3],
            ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy,
            ZSTD_dictContentType_e::ZSTD_dct_fullDict,
        );
        assert!(rc.is_none());
    }

    #[test]
    fn create_magic_prefix_dict_rejects_malformed_entropy_in_auto() {
        use crate::common::mem::MEM_writeLE32;
        use crate::decompress::zstd_decompress::{ZSTD_FRAMEIDSIZE, ZSTD_MAGIC_DICTIONARY};

        let mut dict = vec![0u8; 16];
        MEM_writeLE32(&mut dict[..4], ZSTD_MAGIC_DICTIONARY);
        MEM_writeLE32(
            &mut dict[ZSTD_FRAMEIDSIZE..ZSTD_FRAMEIDSIZE + 4],
            0xCAFEBABE,
        );
        assert!(ZSTD_createDDict(&dict).is_none());
    }

    #[test]
    fn loadEntropy_keeps_dictID_when_entropy_parse_fails() {
        use crate::common::mem::MEM_writeLE32;
        use crate::decompress::zstd_decompress::{ZSTD_FRAMEIDSIZE, ZSTD_MAGIC_DICTIONARY};

        let mut dict = vec![0u8; 16];
        MEM_writeLE32(&mut dict[..4], ZSTD_MAGIC_DICTIONARY);
        MEM_writeLE32(
            &mut dict[ZSTD_FRAMEIDSIZE..ZSTD_FRAMEIDSIZE + 4],
            0xCAFE_BABE,
        );

        let mut ddict = ZSTD_DDict::empty();
        let rc = ZSTD_initDDict_internal(
            &mut ddict,
            &dict,
            ZSTD_dictLoadMethod_e::ZSTD_dlm_byRef,
            ZSTD_dictContentType_e::ZSTD_dct_auto,
        );

        assert!(crate::common::error::ERR_isError(rc));
        assert_eq!(ZSTD_getDictID_fromDDict(&ddict), 0xCAFE_BABE);
        assert_eq!(ddict.entropyPresent, 0);
    }

    #[test]
    fn create_full_dict_mode_loads_entropy_and_exposes_raw_content() {
        use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

        let content = b"ddict-full-mode-content-for-entropy-loading ".repeat(3);
        let dict = build_minimal_zstd_dict(0xA5A5_1234, &content);
        let ddict = ZSTD_createDDict_advanced(
            &dict,
            ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy,
            ZSTD_dictContentType_e::ZSTD_dct_fullDict,
        )
        .expect("full dict");

        assert_eq!(ZSTD_getDictID_fromDDict(&ddict), 0xA5A5_1234);
        assert_eq!(ddict.entropyPresent, 1);
        assert_eq!(ZSTD_DDict_dictContent(&ddict), dict.as_slice());
        assert_eq!(ZSTD_DDict_dictSize(&ddict), dict.len());
        assert_eq!(ZSTD_DDict_originalContent(&ddict), dict.as_slice());
        assert_eq!(ZSTD_DDict_rawContent(&ddict), content.as_slice());

        let mut dctx = ZSTD_DCtx::new();
        ZSTD_copyDDictParameters(&mut dctx, &ddict);
        let dict_start = ZSTD_DDict_dictContent(&ddict).as_ptr() as usize;
        let dict_end = dict_start + ZSTD_DDict_dictSize(&ddict);
        assert_eq!(dctx.prefixStart, Some(dict_start));
        assert_eq!(dctx.virtualStart, Some(dict_start));
        assert_eq!(dctx.dictEnd, Some(dict_end));
        assert_eq!(dctx.previousDstEnd, Some(dict_end));
        assert_eq!(dctx.dictID, 0xA5A5_1234);
        assert_eq!(dctx.stream_dict, dict);
        assert_eq!(dctx.litEntropy, 1);
        assert_eq!(dctx.fseEntropy, 1);
        assert_eq!(dctx.ddict_rep, [8, 7, 1]);
    }

    #[test]
    fn by_reference_full_dict_keeps_preloaded_entropy_after_caller_mutates_header() {
        use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

        let content = b"ddict-byref-stable-entropy-content ".repeat(3);
        let mut dict = build_minimal_zstd_dict(0xCAFE_1234, &content);
        let ddict = ZSTD_createDDict_advanced(
            &dict,
            ZSTD_dictLoadMethod_e::ZSTD_dlm_byRef,
            ZSTD_dictContentType_e::ZSTD_dct_fullDict,
        )
        .expect("full by-ref dict");
        let entropy_size = ZSTD_DDict_rawContent(&ddict).as_ptr() as usize
            - ZSTD_DDict_dictContent(&ddict).as_ptr() as usize;
        assert!(entropy_size > 8);

        dict[8] ^= 0xFF;

        let mut dctx = ZSTD_DCtx::new();
        ZSTD_copyDDictParameters(&mut dctx, &ddict);
        assert_eq!(dctx.dictID, 0xCAFE_1234);
        assert_eq!(dctx.litEntropy, 1);
        assert_eq!(dctx.fseEntropy, 1);
        assert_eq!(dctx.ddict_rep, [8, 7, 1]);
        assert_eq!(dctx.stream_dict, dict);
    }

    #[test]
    fn sizeof_ddict_accounts_for_buffer() {
        let ddict = ZSTD_createDDict(b"abcdefghij").expect("ok");
        let sz = ZSTD_sizeof_DDict(Some(&ddict));
        assert!(sz >= core::mem::size_of::<ZSTD_DDict>() + 10);
        let by_ref = ZSTD_createDDict_byReference(b"abcdefghij").expect("ok");
        assert_eq!(
            ZSTD_sizeof_DDict(Some(&by_ref)),
            core::mem::size_of::<ZSTD_DDict>()
        );
        assert_eq!(ZSTD_sizeof_DDict(None), 0);
    }
}
