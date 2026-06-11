//! Translation of `lib/common/error_private.{h,c}` and the public enum
//! from `lib/zstd_errors.h`.
//!
//! Upstream uses `size_t` return values, with errors encoded as
//! `(size_t)-ZSTD_error_foo` (very large numbers). We preserve that
//! calling convention in the Rust port because it is exposed through
//! the public C-style API (`ZSTD_isError`, `ZSTD_getErrorName`), and
//! keeping it bit-identical means FFI consumers of the original C API
//! could swap implementations without any source change.

/// ZSTD error code. Upstream uses a C enum, which can carry unnamed
/// integer values after casts. `Unknown` preserves those raw values
/// without creating invalid Rust enum discriminants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    NoError,
    Generic,
    PrefixUnknown,
    VersionUnsupported,
    FrameParameterUnsupported,
    FrameParameterWindowTooLarge,
    CorruptionDetected,
    ChecksumWrong,
    LiteralsHeaderWrong,
    DictionaryCorrupted,
    DictionaryWrong,
    DictionaryCreationFailed,
    ParameterUnsupported,
    ParameterCombinationUnsupported,
    ParameterOutOfBound,
    TableLogTooLarge,
    MaxSymbolValueTooLarge,
    MaxSymbolValueTooSmall,
    CannotProduceUncompressedBlock,
    StabilityConditionNotRespected,
    StageWrong,
    InitMissing,
    MemoryAllocation,
    WorkSpaceTooSmall,
    DstSizeTooSmall,
    SrcSizeWrong,
    DstBufferNull,
    NoForwardProgressDestFull,
    NoForwardProgressInputEmpty,
    FrameIndexTooLarge,
    SeekableIO,
    DstBufferWrong,
    SrcBufferWrong,
    SequenceProducerFailed,
    ExternalSequencesInvalid,
    MaxCode,
    Unknown(i32),
}

impl ErrorCode {
    /// Preserve the raw numeric value, matching a C enum cast.
    pub fn from_raw(code: i32) -> Self {
        match code {
            0 => Self::NoError,
            1 => Self::Generic,
            10 => Self::PrefixUnknown,
            12 => Self::VersionUnsupported,
            14 => Self::FrameParameterUnsupported,
            16 => Self::FrameParameterWindowTooLarge,
            20 => Self::CorruptionDetected,
            22 => Self::ChecksumWrong,
            24 => Self::LiteralsHeaderWrong,
            30 => Self::DictionaryCorrupted,
            32 => Self::DictionaryWrong,
            34 => Self::DictionaryCreationFailed,
            40 => Self::ParameterUnsupported,
            41 => Self::ParameterCombinationUnsupported,
            42 => Self::ParameterOutOfBound,
            44 => Self::TableLogTooLarge,
            46 => Self::MaxSymbolValueTooLarge,
            48 => Self::MaxSymbolValueTooSmall,
            49 => Self::CannotProduceUncompressedBlock,
            50 => Self::StabilityConditionNotRespected,
            60 => Self::StageWrong,
            62 => Self::InitMissing,
            64 => Self::MemoryAllocation,
            66 => Self::WorkSpaceTooSmall,
            70 => Self::DstSizeTooSmall,
            72 => Self::SrcSizeWrong,
            74 => Self::DstBufferNull,
            80 => Self::NoForwardProgressDestFull,
            82 => Self::NoForwardProgressInputEmpty,
            100 => Self::FrameIndexTooLarge,
            102 => Self::SeekableIO,
            104 => Self::DstBufferWrong,
            105 => Self::SrcBufferWrong,
            106 => Self::SequenceProducerFailed,
            107 => Self::ExternalSequencesInvalid,
            120 => Self::MaxCode,
            raw => Self::Unknown(raw),
        }
    }

    pub fn as_i32(self) -> i32 {
        match self {
            Self::NoError => 0,
            Self::Generic => 1,
            Self::PrefixUnknown => 10,
            Self::VersionUnsupported => 12,
            Self::FrameParameterUnsupported => 14,
            Self::FrameParameterWindowTooLarge => 16,
            Self::CorruptionDetected => 20,
            Self::ChecksumWrong => 22,
            Self::LiteralsHeaderWrong => 24,
            Self::DictionaryCorrupted => 30,
            Self::DictionaryWrong => 32,
            Self::DictionaryCreationFailed => 34,
            Self::ParameterUnsupported => 40,
            Self::ParameterCombinationUnsupported => 41,
            Self::ParameterOutOfBound => 42,
            Self::TableLogTooLarge => 44,
            Self::MaxSymbolValueTooLarge => 46,
            Self::MaxSymbolValueTooSmall => 48,
            Self::CannotProduceUncompressedBlock => 49,
            Self::StabilityConditionNotRespected => 50,
            Self::StageWrong => 60,
            Self::InitMissing => 62,
            Self::MemoryAllocation => 64,
            Self::WorkSpaceTooSmall => 66,
            Self::DstSizeTooSmall => 70,
            Self::SrcSizeWrong => 72,
            Self::DstBufferNull => 74,
            Self::NoForwardProgressDestFull => 80,
            Self::NoForwardProgressInputEmpty => 82,
            Self::FrameIndexTooLarge => 100,
            Self::SeekableIO => 102,
            Self::DstBufferWrong => 104,
            Self::SrcBufferWrong => 105,
            Self::SequenceProducerFailed => 106,
            Self::ExternalSequencesInvalid => 107,
            Self::MaxCode => 120,
            Self::Unknown(raw) => raw,
        }
    }
}

/// Rich error type for Rust-idiomatic APIs in this crate. The C-compatible
/// layer still uses `size_t`-encoded codes via `ERROR()` / `ERR_isError()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZstdError(pub ErrorCode);

impl core::fmt::Display for ZstdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(ERR_getErrorString(self.0))
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ZstdError {}

/// Translate an `ErrorCode` into upstream's `(size_t)-code` encoding.
///
/// Counterpart to the `ZSTD_ERROR(name)` macro expansion.
#[inline]
pub fn ERROR(code: ErrorCode) -> usize {
    (0usize).wrapping_sub(code.as_i32() as usize)
}

/// Mirror of upstream `ERR_isError(size_t code)`: any numeric return value
/// greater than `ERROR(maxCode)` is interpreted as an error.
#[inline]
pub fn ERR_isError(code: usize) -> bool {
    code > ERROR(ErrorCode::MaxCode)
}

/// Mirror of upstream `ERR_getErrorCode(size_t code)`: decode a `(size_t)-err`
/// return value into its `ErrorCode`. Non-error inputs decode to `NoError`.
#[inline]
pub fn ERR_getErrorCode(code: usize) -> ErrorCode {
    if !ERR_isError(code) {
        return ErrorCode::NoError;
    }
    // -code in 32-bit signed arithmetic.
    let raw = 0i32.wrapping_sub(code as i32);
    ErrorCode::from_raw(raw)
}

/// Mirror of upstream `ERR_getErrorName(size_t code)`: stringify a
/// `(size_t)-err` code.
#[inline]
pub fn ERR_getErrorName(code: usize) -> &'static str {
    ERR_getErrorString(ERR_getErrorCode(code))
}

/// Mirror of upstream `ERR_getErrorString(ERR_enum)`: a stable human-readable
/// label per error code. Strings are byte-for-byte the same as upstream's
/// messages to preserve compatibility with tooling that scrapes them.
pub fn ERR_getErrorString(code: ErrorCode) -> &'static str {
    match code {
        ErrorCode::NoError => "No error detected",
        ErrorCode::Generic => "Error (generic)",
        ErrorCode::PrefixUnknown => "Unknown frame descriptor",
        ErrorCode::VersionUnsupported => "Version not supported",
        ErrorCode::FrameParameterUnsupported => "Unsupported frame parameter",
        ErrorCode::FrameParameterWindowTooLarge => "Frame requires too much memory for decoding",
        ErrorCode::CorruptionDetected => "Data corruption detected",
        ErrorCode::ChecksumWrong => "Restored data doesn't match checksum",
        ErrorCode::LiteralsHeaderWrong => {
            "Header of Literals' block doesn't respect format specification"
        }
        ErrorCode::ParameterUnsupported => "Unsupported parameter",
        ErrorCode::ParameterCombinationUnsupported => "Unsupported combination of parameters",
        ErrorCode::ParameterOutOfBound => "Parameter is out of bound",
        ErrorCode::InitMissing => "Context should be init first",
        ErrorCode::MemoryAllocation => "Allocation error : not enough memory",
        ErrorCode::WorkSpaceTooSmall => "workSpace buffer is not large enough",
        ErrorCode::StageWrong => "Operation not authorized at current processing stage",
        ErrorCode::TableLogTooLarge => "tableLog requires too much memory : unsupported",
        ErrorCode::MaxSymbolValueTooLarge => "Unsupported max Symbol Value : too large",
        ErrorCode::MaxSymbolValueTooSmall => "Specified maxSymbolValue is too small",
        ErrorCode::CannotProduceUncompressedBlock => {
            "This mode cannot generate an uncompressed block"
        }
        ErrorCode::StabilityConditionNotRespected => {
            "pledged buffer stability condition is not respected"
        }
        ErrorCode::DictionaryCorrupted => "Dictionary is corrupted",
        ErrorCode::DictionaryWrong => "Dictionary mismatch",
        ErrorCode::DictionaryCreationFailed => "Cannot create Dictionary from provided samples",
        ErrorCode::DstSizeTooSmall => "Destination buffer is too small",
        ErrorCode::SrcSizeWrong => "Src size is incorrect",
        ErrorCode::DstBufferNull => "Operation on NULL destination buffer",
        ErrorCode::NoForwardProgressDestFull => {
            "Operation made no progress over multiple calls, due to output buffer being full"
        }
        ErrorCode::NoForwardProgressInputEmpty => {
            "Operation made no progress over multiple calls, due to input being empty"
        }
        ErrorCode::FrameIndexTooLarge => "Frame index is too large",
        ErrorCode::SeekableIO => "An I/O error occurred when reading/seeking",
        ErrorCode::DstBufferWrong => "Destination buffer is wrong",
        ErrorCode::SrcBufferWrong => "Source buffer is wrong",
        ErrorCode::SequenceProducerFailed => {
            "Block-level external sequence producer returned an error code"
        }
        ErrorCode::ExternalSequencesInvalid => "External sequences are not valid",
        _ => "Unspecified error code",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UPSTREAM_ERRORS: &[(ErrorCode, i32, &str)] = &[
        (ErrorCode::NoError, 0, "No error detected"),
        (ErrorCode::Generic, 1, "Error (generic)"),
        (ErrorCode::PrefixUnknown, 10, "Unknown frame descriptor"),
        (ErrorCode::VersionUnsupported, 12, "Version not supported"),
        (
            ErrorCode::FrameParameterUnsupported,
            14,
            "Unsupported frame parameter",
        ),
        (
            ErrorCode::FrameParameterWindowTooLarge,
            16,
            "Frame requires too much memory for decoding",
        ),
        (
            ErrorCode::CorruptionDetected,
            20,
            "Data corruption detected",
        ),
        (
            ErrorCode::ChecksumWrong,
            22,
            "Restored data doesn't match checksum",
        ),
        (
            ErrorCode::LiteralsHeaderWrong,
            24,
            "Header of Literals' block doesn't respect format specification",
        ),
        (
            ErrorCode::DictionaryCorrupted,
            30,
            "Dictionary is corrupted",
        ),
        (ErrorCode::DictionaryWrong, 32, "Dictionary mismatch"),
        (
            ErrorCode::DictionaryCreationFailed,
            34,
            "Cannot create Dictionary from provided samples",
        ),
        (ErrorCode::ParameterUnsupported, 40, "Unsupported parameter"),
        (
            ErrorCode::ParameterCombinationUnsupported,
            41,
            "Unsupported combination of parameters",
        ),
        (
            ErrorCode::ParameterOutOfBound,
            42,
            "Parameter is out of bound",
        ),
        (
            ErrorCode::TableLogTooLarge,
            44,
            "tableLog requires too much memory : unsupported",
        ),
        (
            ErrorCode::MaxSymbolValueTooLarge,
            46,
            "Unsupported max Symbol Value : too large",
        ),
        (
            ErrorCode::MaxSymbolValueTooSmall,
            48,
            "Specified maxSymbolValue is too small",
        ),
        (
            ErrorCode::CannotProduceUncompressedBlock,
            49,
            "This mode cannot generate an uncompressed block",
        ),
        (
            ErrorCode::StabilityConditionNotRespected,
            50,
            "pledged buffer stability condition is not respected",
        ),
        (
            ErrorCode::StageWrong,
            60,
            "Operation not authorized at current processing stage",
        ),
        (ErrorCode::InitMissing, 62, "Context should be init first"),
        (
            ErrorCode::MemoryAllocation,
            64,
            "Allocation error : not enough memory",
        ),
        (
            ErrorCode::WorkSpaceTooSmall,
            66,
            "workSpace buffer is not large enough",
        ),
        (
            ErrorCode::DstSizeTooSmall,
            70,
            "Destination buffer is too small",
        ),
        (ErrorCode::SrcSizeWrong, 72, "Src size is incorrect"),
        (
            ErrorCode::DstBufferNull,
            74,
            "Operation on NULL destination buffer",
        ),
        (
            ErrorCode::NoForwardProgressDestFull,
            80,
            "Operation made no progress over multiple calls, due to output buffer being full",
        ),
        (
            ErrorCode::NoForwardProgressInputEmpty,
            82,
            "Operation made no progress over multiple calls, due to input being empty",
        ),
        (
            ErrorCode::FrameIndexTooLarge,
            100,
            "Frame index is too large",
        ),
        (
            ErrorCode::SeekableIO,
            102,
            "An I/O error occurred when reading/seeking",
        ),
        (
            ErrorCode::DstBufferWrong,
            104,
            "Destination buffer is wrong",
        ),
        (ErrorCode::SrcBufferWrong, 105, "Source buffer is wrong"),
        (
            ErrorCode::SequenceProducerFailed,
            106,
            "Block-level external sequence producer returned an error code",
        ),
        (
            ErrorCode::ExternalSequencesInvalid,
            107,
            "External sequences are not valid",
        ),
        (ErrorCode::MaxCode, 120, "Unspecified error code"),
    ];

    #[test]
    fn error_encoding_roundtrip() {
        let code = ErrorCode::CorruptionDetected;
        let enc = ERROR(code);
        assert!(ERR_isError(enc));
        assert_eq!(ERR_getErrorCode(enc), code);
    }

    #[test]
    fn zero_and_small_values_are_not_errors() {
        assert!(!ERR_isError(0));
        assert!(!ERR_isError(1));
        assert!(!ERR_isError(1 << 20));
    }

    #[test]
    fn boundary_maxcode_is_not_error() {
        // `ERR_isError` uses strict > comparison against ERROR(maxCode).
        assert!(!ERR_isError(ERROR(ErrorCode::MaxCode)));
        // Anything "more negative" (a smaller enum value) yields a larger
        // size_t and IS an error.
        assert!(ERR_isError(ERROR(ErrorCode::CorruptionDetected)));
    }

    #[test]
    fn error_strings_match_upstream_literals() {
        for &(code, _raw, name) in UPSTREAM_ERRORS {
            assert_eq!(ERR_getErrorString(code), name, "{code:?}");
        }
    }

    #[test]
    fn error_name_via_size_t() {
        let enc = ERROR(ErrorCode::ChecksumWrong);
        assert_eq!(
            ERR_getErrorName(enc),
            "Restored data doesn't match checksum"
        );
    }

    #[test]
    fn from_raw_preserves_extreme_and_negative_inputs() {
        // C enum casts preserve the numeric value even when it isn't a
        // named enumerator.
        assert_eq!(ErrorCode::from_raw(i32::MIN).as_i32(), i32::MIN);
        assert_eq!(ErrorCode::from_raw(-1).as_i32(), -1);
        assert_eq!(ErrorCode::from_raw(-42).as_i32(), -42);
        assert_eq!(ErrorCode::from_raw(i32::MAX).as_i32(), i32::MAX);
        assert_eq!(ErrorCode::from_raw(999_999).as_i32(), 999_999);
        // Known-valid code still round-trips.
        assert_eq!(ErrorCode::from_raw(22), ErrorCode::ChecksumWrong);
    }

    #[test]
    fn getErrorName_on_non_error_returns_no_error_string() {
        // Contract: calling `ERR_getErrorName` on a code that isn't
        // actually an error (i.e. a small successful size like 100)
        // must return the NoError string, not garbage. This protects
        // the common pattern where a caller logs the error name
        // unconditionally after a call.
        assert_eq!(ERR_getErrorName(0), "No error detected");
        assert_eq!(ERR_getErrorName(100), "No error detected");
        assert_eq!(ERR_getErrorName(1_000_000), "No error detected");
        // Symmetrically for ERR_getErrorCode.
        assert_eq!(ERR_getErrorCode(0), ErrorCode::NoError);
        assert_eq!(ERR_getErrorCode(1_000_000), ErrorCode::NoError);
    }

    #[test]
    fn from_raw_roundtrips_every_named_variant() {
        // Regression gate: if someone adds a new ErrorCode variant
        // but forgets to wire it up in `from_raw`, this catches it.
        for &(code, raw, _name) in UPSTREAM_ERRORS {
            assert_eq!(code.as_i32(), raw, "raw value drifted for {code:?}");
            assert_eq!(
                ErrorCode::from_raw(raw),
                code,
                "from_raw({raw}) did not round-trip for {code:?}"
            );
        }
        // Unknown numeric codes preserve their raw C enum value.
        assert_eq!(ErrorCode::from_raw(99_999).as_i32(), 99_999);
    }

    #[test]
    fn unknown_encoded_error_name_matches_upstream_default() {
        let encoded_unknown_error = (0usize).wrapping_sub(2);
        assert!(ERR_isError(encoded_unknown_error));
        assert_eq!(ERR_getErrorCode(encoded_unknown_error).as_i32(), 2);
        assert_eq!(
            ERR_getErrorString(ErrorCode::Unknown(2)),
            "Unspecified error code"
        );
        assert_eq!(
            ERR_getErrorName(encoded_unknown_error),
            "Unspecified error code"
        );
    }

    #[test]
    fn size_t_error_threshold_matches_upstream() {
        assert!(ERR_isError((0usize).wrapping_sub(1)));
        assert!(ERR_isError((0usize).wrapping_sub(119)));
        assert!(!ERR_isError(ERROR(ErrorCode::MaxCode)));
        assert!(!ERR_isError((0usize).wrapping_sub(121)));
    }
}
