//! Translation of `lib/common/zstd_common.c`.
//!
//! These are the public, C-visible ABI accessors. Upstream defines
//! `ZSTD_VERSION_MAJOR/MINOR/RELEASE` in `lib/zstd.h`; we mirror those
//! values exactly so `ZSTD_versionNumber()` returns the same integer and
//! `ZSTD_versionString()` the same characters.

use crate::common::error::{ErrorCode, ERR_getErrorCode, ERR_getErrorName, ERR_isError};

pub const ZSTD_VERSION_MAJOR: u32 = 1;
pub const ZSTD_VERSION_MINOR: u32 = 6;
pub const ZSTD_VERSION_RELEASE: u32 = 0;
pub const ZSTD_VERSION_NUMBER: u32 =
    ZSTD_VERSION_MAJOR * 100 * 100 + ZSTD_VERSION_MINOR * 100 + ZSTD_VERSION_RELEASE;
pub const ZSTD_VERSION_STRING: &str = "1.6.0";

#[inline]
pub fn ZSTD_versionNumber() -> u32 {
    ZSTD_VERSION_NUMBER
}

#[inline]
pub fn ZSTD_versionString() -> &'static str {
    ZSTD_VERSION_STRING
}

#[inline]
pub fn ZSTD_isError(code: usize) -> bool {
    ERR_isError(code)
}

#[inline]
pub fn ZSTD_getErrorName(code: usize) -> &'static str {
    ERR_getErrorName(code)
}

#[inline]
pub fn ZSTD_getErrorCode(code: usize) -> ErrorCode {
    ERR_getErrorCode(code)
}

#[inline]
pub fn ZSTD_getErrorString(code: ErrorCode) -> &'static str {
    crate::common::error::ERR_getErrorString(code)
}

/// Upstream only returns `1` when the build was configured with
/// `ZSTD_IS_DETERMINISTIC_BUILD`. We default to `0` here and flip it in
/// a future config toggle if needed.
#[inline]
pub const fn ZSTD_isDeterministicBuild() -> i32 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_number_matches_string() {
        assert_eq!(ZSTD_versionNumber(), 1_06_00);
        assert_eq!(ZSTD_versionString(), "1.6.0");
        // Also pin the individual major/minor/release triplet so
        // a bump (e.g. to 1.6.1) updates all four constants together.
        assert_eq!(ZSTD_VERSION_MAJOR, 1);
        assert_eq!(ZSTD_VERSION_MINOR, 6);
        assert_eq!(ZSTD_VERSION_RELEASE, 0);
        // And verify the formula: MAJOR*10000 + MINOR*100 + RELEASE.
        assert_eq!(
            ZSTD_VERSION_NUMBER,
            ZSTD_VERSION_MAJOR * 10_000 + ZSTD_VERSION_MINOR * 100 + ZSTD_VERSION_RELEASE
        );
    }

    #[test]
    fn errorName_differs_across_common_codes() {
        use crate::common::error::ERROR;
        // Each major ErrorCode variant should map to a distinct,
        // non-empty human-readable string.
        let codes = [
            ErrorCode::DstSizeTooSmall,
            ErrorCode::CorruptionDetected,
            ErrorCode::SrcSizeWrong,
            ErrorCode::ParameterOutOfBound,
            ErrorCode::ParameterUnsupported,
        ];
        let mut seen = std::collections::HashSet::new();
        for c in codes {
            let name = ZSTD_getErrorName(ERROR(c));
            assert!(!name.is_empty());
            assert!(seen.insert(name), "duplicate error name: {name}");
        }
    }

    #[test]
    fn errorName_roundtrip_for_common_codes() {
        use crate::common::error::ERROR;
        let e = ERROR(ErrorCode::DstSizeTooSmall);
        assert!(ZSTD_isError(e));
        let name = ZSTD_getErrorName(e);
        assert!(name.contains("Destination") || name.contains("dst") || name.contains("small"));
    }

    #[test]
    fn errorCode_roundtrip() {
        use crate::common::error::ERROR;
        let e = ERROR(ErrorCode::ParameterOutOfBound);
        assert_eq!(ZSTD_getErrorCode(e), ErrorCode::ParameterOutOfBound);
        // And ZSTD_getErrorString routes through the ErrorCode.
        let s1 = ZSTD_getErrorString(ErrorCode::ParameterOutOfBound);
        let s2 = ZSTD_getErrorName(e);
        assert_eq!(s1, s2);
    }

    #[test]
    fn isError_distinguishes_code_from_size() {
        // Small successful sizes must NOT be treated as errors.
        assert!(!ZSTD_isError(0));
        assert!(!ZSTD_isError(100));
        assert!(!ZSTD_isError(1_000_000));
    }
}
