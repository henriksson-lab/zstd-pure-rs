//! Translation of `lib/decompress/zstd_ddict.c`. Digested dictionary
//! for decompression.
//!
//! Scope note: raw-content dictionaries (`ZSTD_dct_rawContent`) are
//! fully supported. Pre-digested HUF+FSE tables from dicts prefixed
//! with `ZSTD_MAGIC_DICTIONARY` need the `ZSTD_entropyDTables_t`
//! wire-up plus ext-dict sequence execution, both still out of scope
//! for v0.1. Magic-tagged dicts in `ZSTD_dct_fullDict` mode currently
//! return `DictionaryCreationFailed` to flag the gap.

#![allow(unused_variables)]

use crate::common::error::{ErrorCode, ERROR};

/// Mirror of upstream `ZSTD_dictLoadMethod_e`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZSTD_dictLoadMethod_e {
    ZSTD_dlm_byCopy,
    ZSTD_dlm_byRef,
}

/// Mirror of upstream `ZSTD_dictContentType_e`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZSTD_dictContentType_e {
    #[default]
    ZSTD_dct_auto,
    ZSTD_dct_rawContent,
    ZSTD_dct_fullDict,
}

/// Digested dictionary. Rust-side port owns its content via a `Vec<u8>`
/// in `dictBuffer`; the `byReference` variant still stores the borrow
/// via lifetime gymnastics in a future revision — for now, all dicts
/// are copied (`byCopy` semantics).
pub struct ZSTD_DDict {
    pub dictBuffer: Vec<u8>,
    pub dictSize: usize,
    pub dictID: u32,
    pub entropyPresent: u32,
}

/// Port of `ZSTD_DDict_dictContent`.
#[inline]
pub fn ZSTD_DDict_dictContent(ddict: &ZSTD_DDict) -> &[u8] {
    &ddict.dictBuffer[..ddict.dictSize]
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
pub fn ZSTD_estimateDDictSize(
    dictSize: usize,
    dictLoadMethod: ZSTD_dictLoadMethod_e,
) -> usize {
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
        Some(d) => core::mem::size_of::<ZSTD_DDict>() + d.dictBuffer.capacity(),
    }
}

/// Port of `ZSTD_loadEntropy_intoDDict` — raw-content branch only.
/// Returns 0 on success or an error code. For magic-tagged dicts with
/// `ZSTD_dct_fullDict`, we currently return
/// `DictionaryCreationFailed` to flag the unimplemented path.
fn ZSTD_loadEntropy_intoDDict(
    ddict: &mut ZSTD_DDict,
    dictContentType: ZSTD_dictContentType_e,
) -> usize {
    use crate::common::mem::MEM_readLE32;
    use crate::decompress::zstd_decompress::{ZSTD_FRAMEIDSIZE, ZSTD_MAGIC_DICTIONARY};

    ddict.dictID = 0;
    ddict.entropyPresent = 0;
    if dictContentType == ZSTD_dictContentType_e::ZSTD_dct_rawContent {
        return 0;
    }

    if ddict.dictSize < 8 {
        if dictContentType == ZSTD_dictContentType_e::ZSTD_dct_fullDict {
            return ERROR(ErrorCode::DictionaryCorrupted);
        }
        return 0; // pure content mode
    }
    let magic = MEM_readLE32(&ddict.dictBuffer[..4]);
    if magic != ZSTD_MAGIC_DICTIONARY {
        if dictContentType == ZSTD_dictContentType_e::ZSTD_dct_fullDict {
            return ERROR(ErrorCode::DictionaryCorrupted);
        }
        return 0;
    }

    // Magic-tagged: load dictID, but defer entropy-table parsing to a
    // future tick. Upstream would build HUF + FSE DTables here.
    ddict.dictID = MEM_readLE32(&ddict.dictBuffer[ZSTD_FRAMEIDSIZE..ZSTD_FRAMEIDSIZE + 4]);

    if dictContentType == ZSTD_dictContentType_e::ZSTD_dct_fullDict {
        // Pre-digested HUF/FSE not yet wired. Report a precise error
        // so callers know this path is a stub.
        return ERROR(ErrorCode::DictionaryCreationFailed);
    }
    // For `ZSTD_dct_auto`, fall back to raw-content semantics: dictID
    // is recognized but entropy tables aren't pre-built.
    ddict.entropyPresent = 0;
    0
}

/// Port of `ZSTD_createDDict_advanced`. Always copies the dict bytes
/// for now.
pub fn ZSTD_createDDict_advanced(
    dict: &[u8],
    _dictLoadMethod: ZSTD_dictLoadMethod_e,
    dictContentType: ZSTD_dictContentType_e,
) -> Option<Box<ZSTD_DDict>> {
    let mut ddict = Box::new(ZSTD_DDict {
        dictBuffer: dict.to_vec(),
        dictSize: dict.len(),
        dictID: 0,
        entropyPresent: 0,
    });
    let rc = ZSTD_loadEntropy_intoDDict(&mut ddict, dictContentType);
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

/// Port of `ZSTD_createDDict_byReference`. In the Rust port we still
/// take a copy (the lifetime plumbing for a true by-reference variant
/// lands with streaming dict support).
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

#[cfg(test)]
mod byref_tests {
    use super::*;

    #[test]
    fn createDDict_byReference_matches_regular_creator_content() {
        // byReference is currently implemented as a by-copy load in
        // the Rust port; the exposed content must nonetheless equal
        // the plain `createDDict` output for the same buffer.
        let dict = b"byref-equivalence-test-dict ".repeat(4);
        let by_copy = ZSTD_createDDict(&dict).expect("by-copy");
        let by_ref = ZSTD_createDDict_byReference(&dict).expect("by-ref");
        assert_eq!(ZSTD_DDict_dictContent(&by_copy), ZSTD_DDict_dictContent(&by_ref));
        assert_eq!(ZSTD_DDict_dictSize(&by_copy), ZSTD_DDict_dictSize(&by_ref));
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
    fn create_magic_prefix_dict_reads_dict_id_in_auto() {
        use crate::common::mem::MEM_writeLE32;
        use crate::decompress::zstd_decompress::{ZSTD_FRAMEIDSIZE, ZSTD_MAGIC_DICTIONARY};

        let mut dict = vec![0u8; 16];
        MEM_writeLE32(&mut dict[..4], ZSTD_MAGIC_DICTIONARY);
        MEM_writeLE32(&mut dict[ZSTD_FRAMEIDSIZE..ZSTD_FRAMEIDSIZE + 4], 0xCAFEBABE);
        // In auto mode, full-entropy loading is not yet wired so
        // `ZSTD_dct_auto` falls back to raw while still recognizing
        // the dictID.
        let ddict = ZSTD_createDDict(&dict).expect("auto-mode magic dict");
        assert_eq!(ZSTD_getDictID_fromDDict(&ddict), 0xCAFEBABE);
    }

    #[test]
    fn sizeof_ddict_accounts_for_buffer() {
        let ddict = ZSTD_createDDict(b"abcdefghij").expect("ok");
        let sz = ZSTD_sizeof_DDict(Some(&ddict));
        assert!(sz >= core::mem::size_of::<ZSTD_DDict>() + 10);
        assert_eq!(ZSTD_sizeof_DDict(None), 0);
    }
}
