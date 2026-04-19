//! Translation of `lib/common/error_private.{h,c}` and the public enum
//! from `lib/zstd_errors.h`.
//!
//! Upstream uses `size_t` return values, with errors encoded as
//! `(size_t)-ZSTD_error_foo` (very large numbers). We preserve that
//! calling convention in the Rust port because it is exposed through
//! the public C-style API (`ZSTD_isError`, `ZSTD_getErrorName`), and
//! keeping it bit-identical means FFI consumers of the original C API
//! could swap implementations without any source change.

/// Sum of all ZSTD error codes. Values are fixed by upstream and must
/// not drift; they are serialized indirectly through
/// `(size_t)-code` return values from the C-compatible API.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    NoError = 0,
    Generic = 1,
    PrefixUnknown = 10,
    VersionUnsupported = 12,
    FrameParameterUnsupported = 14,
    FrameParameterWindowTooLarge = 16,
    CorruptionDetected = 20,
    ChecksumWrong = 22,
    LiteralsHeaderWrong = 24,
    DictionaryCorrupted = 30,
    DictionaryWrong = 32,
    DictionaryCreationFailed = 34,
    ParameterUnsupported = 40,
    ParameterCombinationUnsupported = 41,
    ParameterOutOfBound = 42,
    TableLogTooLarge = 44,
    MaxSymbolValueTooLarge = 46,
    MaxSymbolValueTooSmall = 48,
    CannotProduceUncompressedBlock = 49,
    StabilityConditionNotRespected = 50,
    StageWrong = 60,
    InitMissing = 62,
    MemoryAllocation = 64,
    WorkSpaceTooSmall = 66,
    DstSizeTooSmall = 70,
    SrcSizeWrong = 72,
    DstBufferNull = 74,
    NoForwardProgressDestFull = 80,
    NoForwardProgressInputEmpty = 82,
    FrameIndexTooLarge = 100,
    SeekableIO = 102,
    DstBufferWrong = 104,
    SrcBufferWrong = 105,
    SequenceProducerFailed = 106,
    ExternalSequencesInvalid = 107,
    MaxCode = 120,
}

impl ErrorCode {
    /// Reverse lookup: map a raw numeric code back to the enum. Unknown
    /// codes are clamped to `Generic`, matching upstream's tolerance.
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
            _ => Self::Generic,
        }
    }

    pub fn as_i32(self) -> i32 {
        self as i32
    }
}

/// Rich error type for Rust-idiomatic APIs in this crate. The C-compatible
/// layer still uses `size_t`-encoded codes via `to_code()` / `ERR_isError()`.
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
    (0usize).wrapping_sub(code as i32 as usize)
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
    use ErrorCode::*;
    match code {
        NoError => "No error detected",
        Generic => "Error (generic)",
        PrefixUnknown => "Unknown frame descriptor",
        VersionUnsupported => "Version not supported",
        FrameParameterUnsupported => "Unsupported frame parameter",
        FrameParameterWindowTooLarge => "Frame requires too much memory for decoding",
        CorruptionDetected => "Data corruption detected",
        ChecksumWrong => "Restored data doesn't match checksum",
        LiteralsHeaderWrong => "Header of Literals' block doesn't respect format specification",
        ParameterUnsupported => "Unsupported parameter",
        ParameterCombinationUnsupported => "Unsupported combination of parameters",
        ParameterOutOfBound => "Parameter is out of bound",
        InitMissing => "Context should be init first",
        MemoryAllocation => "Allocation error : not enough memory",
        WorkSpaceTooSmall => "workSpace buffer is not large enough",
        StageWrong => "Operation not authorized at current processing stage",
        TableLogTooLarge => "tableLog requires too much memory : unsupported",
        MaxSymbolValueTooLarge => "Unsupported max Symbol Value : too large",
        MaxSymbolValueTooSmall => "Specified maxSymbolValue is too small",
        CannotProduceUncompressedBlock => "This mode cannot generate an uncompressed block",
        StabilityConditionNotRespected => "pledged buffer stability condition is not respected",
        DictionaryCorrupted => "Dictionary is corrupted",
        DictionaryWrong => "Dictionary mismatch",
        DictionaryCreationFailed => "Cannot create Dictionary from provided samples",
        DstSizeTooSmall => "Destination buffer is too small",
        SrcSizeWrong => "Src size is incorrect",
        DstBufferNull => "Operation on NULL destination buffer",
        NoForwardProgressDestFull => {
            "Operation made no progress over multiple calls, due to output buffer being full"
        }
        NoForwardProgressInputEmpty => {
            "Operation made no progress over multiple calls, due to input being empty"
        }
        FrameIndexTooLarge => "Frame index is too large",
        SeekableIO => "An I/O error occurred when reading/seeking",
        DstBufferWrong => "Destination buffer is wrong",
        SrcBufferWrong => "Source buffer is wrong",
        SequenceProducerFailed => "Block-level external sequence producer returned an error code",
        ExternalSequencesInvalid => "External sequences are not valid",
        MaxCode => "Unspecified error code",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(ERR_getErrorString(ErrorCode::NoError), "No error detected");
        assert_eq!(
            ERR_getErrorString(ErrorCode::CorruptionDetected),
            "Data corruption detected"
        );
        assert_eq!(
            ERR_getErrorString(ErrorCode::ChecksumWrong),
            "Restored data doesn't match checksum"
        );
    }

    #[test]
    fn error_name_via_size_t() {
        let enc = ERROR(ErrorCode::ChecksumWrong);
        assert_eq!(ERR_getErrorName(enc), "Restored data doesn't match checksum");
    }

    #[test]
    fn from_raw_clamps_extreme_and_negative_inputs_to_Generic() {
        // Callers that build an `ErrorCode` from an arbitrary i32
        // (e.g. FFI return values from legacy APIs) must get a safe
        // `Generic` fallback for values outside the enum, including
        // i32::MIN / i32::MAX / negative / oversized.
        assert_eq!(ErrorCode::from_raw(i32::MIN), ErrorCode::Generic);
        assert_eq!(ErrorCode::from_raw(-1), ErrorCode::Generic);
        assert_eq!(ErrorCode::from_raw(-42), ErrorCode::Generic);
        assert_eq!(ErrorCode::from_raw(i32::MAX), ErrorCode::Generic);
        assert_eq!(ErrorCode::from_raw(999_999), ErrorCode::Generic);
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
        for code in [
            ErrorCode::NoError,
            ErrorCode::Generic,
            ErrorCode::PrefixUnknown,
            ErrorCode::VersionUnsupported,
            ErrorCode::FrameParameterUnsupported,
            ErrorCode::FrameParameterWindowTooLarge,
            ErrorCode::CorruptionDetected,
            ErrorCode::ChecksumWrong,
            ErrorCode::LiteralsHeaderWrong,
            ErrorCode::DictionaryCorrupted,
            ErrorCode::DictionaryWrong,
            ErrorCode::DictionaryCreationFailed,
            ErrorCode::ParameterUnsupported,
            ErrorCode::ParameterCombinationUnsupported,
            ErrorCode::ParameterOutOfBound,
            ErrorCode::TableLogTooLarge,
            ErrorCode::MaxSymbolValueTooLarge,
            ErrorCode::MaxSymbolValueTooSmall,
            ErrorCode::CannotProduceUncompressedBlock,
            ErrorCode::StabilityConditionNotRespected,
            ErrorCode::StageWrong,
            ErrorCode::InitMissing,
            ErrorCode::MemoryAllocation,
            ErrorCode::WorkSpaceTooSmall,
            ErrorCode::DstSizeTooSmall,
            ErrorCode::SrcSizeWrong,
            ErrorCode::DstBufferNull,
            ErrorCode::NoForwardProgressDestFull,
            ErrorCode::NoForwardProgressInputEmpty,
            ErrorCode::FrameIndexTooLarge,
            ErrorCode::SeekableIO,
            ErrorCode::DstBufferWrong,
            ErrorCode::SrcBufferWrong,
            ErrorCode::SequenceProducerFailed,
            ErrorCode::ExternalSequencesInvalid,
            ErrorCode::MaxCode,
        ] {
            let raw = code as i32;
            assert_eq!(
                ErrorCode::from_raw(raw), code,
                "from_raw({raw}) did not round-trip for {code:?}"
            );
        }
        // Unknown numeric codes clamp to Generic.
        assert_eq!(ErrorCode::from_raw(99_999), ErrorCode::Generic);
    }
}
