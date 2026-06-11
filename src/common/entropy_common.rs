//! Translation of `lib/common/entropy_common.c`.
//!
//! Covers:
//! - `FSE_readNCount` (parses an FSE normalized-count header)
//! - `HUF_readStats` / `HUF_readStats_wksp` — parses a HUF tree
//!   descriptor; the FSE-compressed branch routes through
//!   `fse_decompress::FSE_decompress_wksp_bmi2`, the raw branch
//!   (`iSize >= 128`) handles short headers in-module.
//! - thin version/error accessors

use crate::common::bits::{ZSTD_countTrailingZeros32, ZSTD_highbit32};
use crate::common::error::{ErrorCode, ERROR};
use crate::common::mem::MEM_readLE32;

pub const FSE_VERSION_NUMBER: u32 = 900;

/// Re-exports — canonical definitions live in `fse_decompress`.
pub use crate::common::fse_decompress::{
    FSE_MAX_SYMBOL_VALUE, FSE_MIN_TABLELOG, FSE_TABLELOG_ABSOLUTE_MAX,
};

/// Upstream `FSE_DTABLE_SIZE_U32(maxTableLog)` (`fse.h:241`). Number of
/// u32 entries a DTable of `maxTableLog` occupies (including the
/// 1-entry header).
#[inline]
pub const fn FSE_DTABLE_SIZE_U32(maxTableLog: u32) -> usize {
    1 + (1usize << maxTableLog)
}

/// Upstream `FSE_DTABLE_SIZE` — DTable size in bytes.
#[inline]
pub const fn FSE_DTABLE_SIZE(maxTableLog: u32) -> usize {
    FSE_DTABLE_SIZE_U32(maxTableLog) * 4
}

/// Upstream `FSE_BUILD_DTABLE_WKSP_SIZE` (`fse.h:267`). Scratch
/// required by `FSE_buildDTable_wksp`. Upstream formula:
/// `sizeof(short) * (maxSymbolValue + 1) + (1ULL << maxTableLog) + 8`.
#[inline]
pub const fn FSE_BUILD_DTABLE_WKSP_SIZE(maxTableLog: u32, maxSymbolValue: u32) -> usize {
    2 * (maxSymbolValue as usize + 1) + (1usize << maxTableLog) + 8
}

/// Upstream `FSE_BUILD_DTABLE_WKSP_SIZE_U32`. Same in u32 counts.
#[inline]
pub const fn FSE_BUILD_DTABLE_WKSP_SIZE_U32(maxTableLog: u32, maxSymbolValue: u32) -> usize {
    FSE_BUILD_DTABLE_WKSP_SIZE(maxTableLog, maxSymbolValue).div_ceil(4)
}

/// Upstream `FSE_DECOMPRESS_WKSP_SIZE_U32` (`fse.h:272`). Total
/// workspace needed by `FSE_decompress_wksp` — DTable + build
/// scratch + symbol-bitmap + slack.
#[inline]
pub const fn FSE_DECOMPRESS_WKSP_SIZE_U32(maxTableLog: u32, maxSymbolValue: u32) -> usize {
    FSE_DTABLE_SIZE_U32(maxTableLog)
        + 1
        + FSE_BUILD_DTABLE_WKSP_SIZE_U32(maxTableLog, maxSymbolValue)
        + ((FSE_MAX_SYMBOL_VALUE + 1) as usize) / 2
        + 1
}

/// Upstream `FSE_DECOMPRESS_WKSP_SIZE` — byte count of the above.
#[inline]
pub const fn FSE_DECOMPRESS_WKSP_SIZE(maxTableLog: u32, maxSymbolValue: u32) -> usize {
    FSE_DECOMPRESS_WKSP_SIZE_U32(maxTableLog, maxSymbolValue) * 4
}

// Re-exports — canonical definitions live in huf_decompress.
pub use crate::decompress::huf_decompress::{
    HUF_flags_bmi2, HUF_TABLELOG_ABSOLUTEMAX, HUF_TABLELOG_MAX,
};

/// Upstream `HUF_READ_STATS_WORKSPACE_SIZE_U32` (`huf.h:181`). Scratch
/// u32 count required by `HUF_readStats_wksp`. Upstream folds through
/// `FSE_DECOMPRESS_WKSP_SIZE_U32(6, HUF_TABLELOG_MAX-1)`.
pub const HUF_READ_STATS_WORKSPACE_SIZE_U32: usize =
    FSE_DECOMPRESS_WKSP_SIZE_U32(6, HUF_TABLELOG_MAX - 1);

/// Upstream `HUF_READ_STATS_WORKSPACE_SIZE` — byte count.
pub const HUF_READ_STATS_WORKSPACE_SIZE: usize = HUF_READ_STATS_WORKSPACE_SIZE_U32 * 4;

/// Port of `FSE_versionNumber`. Returns the FSE library version constant.
pub fn FSE_versionNumber() -> u32 {
    FSE_VERSION_NUMBER
}

/// Port of `FSE_isError`. Forwards to the common `ERR_isError` test.
pub fn FSE_isError(code: usize) -> u32 {
    crate::common::error::ERR_isError(code) as u32
}

/// Port of `FSE_getErrorName`. Forwards to `ERR_getErrorName`.
pub fn FSE_getErrorName(code: usize) -> &'static str {
    crate::common::error::ERR_getErrorName(code)
}

/// Port of `HUF_isError`. Forwards to the common `ERR_isError` test.
pub fn HUF_isError(code: usize) -> u32 {
    crate::common::error::ERR_isError(code) as u32
}

/// Port of `HUF_getErrorName`. Forwards to `ERR_getErrorName`.
pub fn HUF_getErrorName(code: usize) -> &'static str {
    crate::common::error::ERR_getErrorName(code)
}

/// Port of `FSE_readNCount_body` (and thus `FSE_readNCount` /
/// `FSE_readNCount_bmi2`). Decodes an FSE normalized-count header
/// starting at `src`, writing the counts into `normalizedCounter`,
/// updating `*maxSVPtr` and `*tableLogPtr`. Returns bytes consumed, or
/// an error code testable with `ERR_isError`.
pub fn FSE_readNCount(
    normalizedCounter: &mut [i16],
    maxSVPtr: &mut u32,
    tableLogPtr: &mut u32,
    src: &[u8],
) -> usize {
    FSE_readNCount_body(normalizedCounter, maxSVPtr, tableLogPtr, src, true)
}

/// Rust-only helper: `FSE_readNCount` with `clear_counts=false` so the
/// caller can reuse a pre-zeroed counter buffer across calls.
pub(crate) fn FSE_readNCount_no_clear(
    normalizedCounter: &mut [i16],
    maxSVPtr: &mut u32,
    tableLogPtr: &mut u32,
    src: &[u8],
) -> usize {
    FSE_readNCount_body(normalizedCounter, maxSVPtr, tableLogPtr, src, false)
}

/// Port of `FSE_readNCount_body`. Shared implementation behind
/// `FSE_readNCount` and `FSE_readNCount_no_clear`; `clear_counts`
/// toggles the initial zero-fill of `normalizedCounter`.
fn FSE_readNCount_body(
    normalizedCounter: &mut [i16],
    maxSVPtr: &mut u32,
    tableLogPtr: &mut u32,
    src: &[u8],
    clear_counts: bool,
) -> usize {
    let hbSize = src.len();

    // Upstream: if hbSize < 8, pad into an 8-byte zero-filled buffer and
    // recurse. We do the same — the returned byte-count is bounded by
    // the real input size and the validator catches over-read.
    if hbSize < 8 {
        let mut buffer = [0u8; 8];
        buffer[..hbSize].copy_from_slice(src);
        let countSize = FSE_readNCount_body(
            normalizedCounter,
            maxSVPtr,
            tableLogPtr,
            &buffer,
            clear_counts,
        );
        if crate::common::error::ERR_isError(countSize) {
            return countSize;
        }
        if countSize > hbSize {
            return ERROR(ErrorCode::CorruptionDetected);
        }
        return countSize;
    }

    let iend = hbSize;
    let mut ip: usize = 0;
    let mut charnum: u32 = 0;
    let maxSV1 = (*maxSVPtr).wrapping_add(1);
    let mut previous0 = false;

    if normalizedCounter.len() < maxSV1 as usize {
        return ERROR(ErrorCode::MaxSymbolValueTooSmall);
    }

    // Zero out the counter; symbols not present stay at 0.
    if clear_counts {
        normalizedCounter[..maxSV1 as usize].fill(0);
    }

    let mut bitStream: u32 = MEM_readLE32(&src[ip..]);
    let mut nbBits = ((bitStream & 0xF) + FSE_MIN_TABLELOG) as i32;
    if nbBits > FSE_TABLELOG_ABSOLUTE_MAX as i32 {
        return ERROR(ErrorCode::TableLogTooLarge);
    }
    bitStream >>= 4;
    let mut bitCount: i32 = 4;
    *tableLogPtr = nbBits as u32;
    let mut remaining: i32 = (1 << nbBits) + 1;
    let mut threshold: i32 = 1 << nbBits;
    nbBits += 1;

    loop {
        if previous0 {
            // Count repeats: each time the 2-bit code is 0b11, another
            // triple-zero repeat is encoded. Setting the high bit
            // avoids UB on ~0u32.
            let mut repeats = (ZSTD_countTrailingZeros32(!bitStream | 0x8000_0000) >> 1) as i32;
            while repeats >= 12 {
                charnum += 3 * 12;
                if ip <= iend - 7 {
                    ip += 3;
                } else {
                    bitCount -= 8 * (iend as i32 - 7 - ip as i32);
                    bitCount &= 31;
                    ip = iend - 4;
                }
                bitStream = MEM_readLE32(&src[ip..]) >> bitCount;
                repeats = (ZSTD_countTrailingZeros32(!bitStream | 0x8000_0000) >> 1) as i32;
            }
            charnum += 3 * repeats as u32;
            bitStream >>= 2 * repeats;
            bitCount += 2 * repeats;

            // Add the final repeat which is NOT 0b11.
            debug_assert!(bitStream & 3 < 3);
            charnum += bitStream & 3;
            bitCount += 2;

            if charnum >= maxSV1 {
                break;
            }

            if ip <= iend - 7 || (ip + (bitCount >> 3) as usize <= iend - 4) {
                ip += (bitCount >> 3) as usize;
                bitCount &= 7;
            } else {
                bitCount -= 8 * (iend as i32 - 4 - ip as i32);
                bitCount &= 31;
                ip = iend - 4;
            }
            bitStream = MEM_readLE32(&src[ip..]) >> bitCount;
        }

        let max = (2 * threshold - 1) - remaining;
        let mut count;
        if (bitStream & (threshold as u32 - 1)) < max as u32 {
            count = (bitStream & (threshold as u32 - 1)) as i32;
            bitCount += nbBits - 1;
        } else {
            count = (bitStream & (2 * threshold as u32 - 1)) as i32;
            if count >= threshold {
                count -= max;
            }
            bitCount += nbBits;
        }

        count -= 1; // extra accuracy
        if count >= 0 {
            remaining -= count;
        } else {
            debug_assert_eq!(count, -1);
            remaining += count;
        }
        if charnum as usize >= normalizedCounter.len() {
            return ERROR(ErrorCode::MaxSymbolValueTooSmall);
        }
        normalizedCounter[charnum as usize] = count as i16;
        charnum += 1;
        previous0 = count == 0;

        debug_assert!(threshold > 1);
        if remaining < threshold {
            if remaining <= 1 {
                break;
            }
            nbBits = ZSTD_highbit32(remaining as u32) as i32 + 1;
            threshold = 1 << (nbBits - 1);
        }
        if charnum >= maxSV1 {
            break;
        }

        if ip <= iend - 7 || (ip + (bitCount >> 3) as usize <= iend - 4) {
            ip += (bitCount >> 3) as usize;
            bitCount &= 7;
        } else {
            bitCount -= 8 * (iend as i32 - 4 - ip as i32);
            bitCount &= 31;
            ip = iend - 4;
        }
        bitStream = MEM_readLE32(&src[ip..]) >> bitCount;
    }

    if remaining != 1 {
        return ERROR(ErrorCode::CorruptionDetected);
    }
    if charnum > maxSV1 {
        return ERROR(ErrorCode::MaxSymbolValueTooSmall);
    }
    if bitCount > 32 {
        return ERROR(ErrorCode::CorruptionDetected);
    }
    *maxSVPtr = charnum.wrapping_sub(1);
    ip += ((bitCount + 7) >> 3) as usize;
    ip
}

/// Port of `FSE_readNCount_bmi2`. Our scalar path does not benefit from
/// BMI2 compiler hints at this stage; we keep the same signature for
/// code-complexity-comparator parity.
pub fn FSE_readNCount_bmi2(
    normalizedCounter: &mut [i16],
    maxSVPtr: &mut u32,
    tableLogPtr: &mut u32,
    src: &[u8],
    _bmi2: i32,
) -> usize {
    FSE_readNCount(normalizedCounter, maxSVPtr, tableLogPtr, src)
}

/// Port of `HUF_readStats`. Uses an in-function workspace sized for
/// `HUF_TABLELOG_MAX`.
pub fn HUF_readStats(
    huffWeight: &mut [u8],
    hwSize: usize,
    rankStats: &mut [u32],
    nbSymbolsPtr: &mut u32,
    tableLogPtr: &mut u32,
    src: &[u8],
) -> usize {
    let mut wksp = [0u32; HUF_READ_STATS_WORKSPACE_SIZE_U32];
    HUF_readStats_wksp(
        huffWeight,
        hwSize,
        rankStats,
        nbSymbolsPtr,
        tableLogPtr,
        src,
        &mut wksp,
        0,
    )
}

/// Port of `HUF_readStats_body` (and its BMI2 wrapper). Handles both
/// the raw layout (`iSize >= 128`, weights packed as half-bytes) and
/// the FSE-compressed layout (routed through
/// `fse_decompress::FSE_decompress_wksp_bmi2`). Returns the total
/// header size consumed.
pub fn HUF_readStats_wksp(
    huffWeight: &mut [u8],
    hwSize: usize,
    rankStats: &mut [u32],
    nbSymbolsPtr: &mut u32,
    tableLogPtr: &mut u32,
    src: &[u8],
    workSpace: &mut [u32],
    flags: i32,
) -> usize {
    if src.is_empty() {
        return ERROR(ErrorCode::SrcSizeWrong);
    }
    let mut iSize = src[0] as usize;
    let oSize;
    let mut read_from_fse = 0usize;

    if iSize >= 128 {
        // Raw (non-FSE-compressed) layout.
        oSize = iSize - 127;
        iSize = oSize.div_ceil(2);
        if iSize + 1 > src.len() {
            return ERROR(ErrorCode::SrcSizeWrong);
        }
        if oSize >= hwSize {
            return ERROR(ErrorCode::CorruptionDetected);
        }
        if huffWeight.len() < hwSize {
            return ERROR(ErrorCode::CorruptionDetected);
        }
        let payload = &src[1..];
        let mut n = 0;
        while n < oSize {
            huffWeight[n] = payload[n / 2] >> 4;
            huffWeight[n + 1] = payload[n / 2] & 0x0F;
            n += 2;
        }
    } else {
        // FSE-compressed header. Requires `FSE_decompress_wksp_bmi2`.
        if iSize + 1 > src.len() {
            return ERROR(ErrorCode::SrcSizeWrong);
        }
        if hwSize == 0 || huffWeight.len() < hwSize {
            return ERROR(ErrorCode::CorruptionDetected);
        }
        let bmi2 = (flags & HUF_flags_bmi2) != 0;
        oSize = crate::common::fse_decompress::FSE_decompress_wksp_bmi2(
            &mut huffWeight[..hwSize - 1],
            &src[1..1 + iSize],
            6,
            // workspace conversion: re-use the u32 slice as bytes
            bytemuck_u32_as_u8(workSpace),
            bmi2 as i32,
        );
        if crate::common::error::ERR_isError(oSize) {
            return oSize;
        }
        read_from_fse = oSize;
    }

    // Reset rank stats.
    if rankStats.len() < HUF_TABLELOG_MAX as usize + 1 {
        return ERROR(ErrorCode::CorruptionDetected);
    }
    for r in rankStats.iter_mut().take(HUF_TABLELOG_MAX as usize + 1) {
        *r = 0;
    }
    let _ = read_from_fse; // consumed below via oSize

    let mut weightTotal: u32 = 0;
    for n in 0..oSize {
        if huffWeight[n] as u32 > HUF_TABLELOG_MAX {
            return ERROR(ErrorCode::CorruptionDetected);
        }
        rankStats[huffWeight[n] as usize] += 1;
        weightTotal += (1u32 << huffWeight[n]) >> 1;
    }
    if weightTotal == 0 {
        return ERROR(ErrorCode::CorruptionDetected);
    }

    // Derive tableLog and last symbol weight (implied).
    let tableLog = ZSTD_highbit32(weightTotal) + 1;
    if tableLog > HUF_TABLELOG_MAX {
        return ERROR(ErrorCode::CorruptionDetected);
    }
    *tableLogPtr = tableLog;

    let total = 1u32 << tableLog;
    let rest = total - weightTotal;
    let verif = 1u32 << ZSTD_highbit32(rest);
    let lastWeight = ZSTD_highbit32(rest) + 1;
    if verif != rest {
        return ERROR(ErrorCode::CorruptionDetected);
    }
    huffWeight[oSize] = lastWeight as u8;
    rankStats[lastWeight as usize] += 1;

    // At least 2 even-count rank-1 weights required.
    if rankStats[1] < 2 || (rankStats[1] & 1) != 0 {
        return ERROR(ErrorCode::CorruptionDetected);
    }

    *nbSymbolsPtr = (oSize + 1) as u32;
    iSize + 1
}

/// Rust-only helper: view a `&mut [u32]` as `&mut [u8]` for passing to
/// byte-oriented FSE workspaces. Endianness is irrelevant because the
/// workspace is opaque scratch.
fn bytemuck_u32_as_u8(words: &mut [u32]) -> &mut [u8] {
    let len = words.len() * 4;
    // SAFETY: u32 has stricter alignment than u8; contiguous storage.
    unsafe { core::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, len) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_forwarders_mirror_zstd_errors() {
        let err = ERROR(ErrorCode::CorruptionDetected);
        assert_eq!(FSE_isError(err), 1);
        assert_eq!(HUF_isError(err), 1);
        assert_eq!(FSE_isError(0), 0);
        assert_eq!(HUF_isError(0), 0);
        assert!(FSE_getErrorName(err).contains("corruption"));
        assert!(HUF_getErrorName(err).contains("corruption"));
    }

    #[test]
    fn fse_version_matches_upstream() {
        assert_eq!(FSE_versionNumber(), 900);
    }

    #[test]
    fn fse_workspace_sizes_match_upstream_macros() {
        assert_eq!(FSE_DTABLE_SIZE_U32(6), 65);
        assert_eq!(FSE_DTABLE_SIZE(6), 260);

        assert_eq!(FSE_BUILD_DTABLE_WKSP_SIZE(0, 0), 11);
        assert_eq!(FSE_BUILD_DTABLE_WKSP_SIZE_U32(0, 0), 3);
        assert_eq!(FSE_BUILD_DTABLE_WKSP_SIZE(6, HUF_TABLELOG_MAX - 1), 96);
        assert_eq!(FSE_BUILD_DTABLE_WKSP_SIZE_U32(6, HUF_TABLELOG_MAX - 1), 24);
        assert_eq!(
            FSE_BUILD_DTABLE_WKSP_SIZE(FSE_TABLELOG_ABSOLUTE_MAX, FSE_MAX_SYMBOL_VALUE),
            33_288
        );
        assert_eq!(
            FSE_BUILD_DTABLE_WKSP_SIZE_U32(FSE_TABLELOG_ABSOLUTE_MAX, FSE_MAX_SYMBOL_VALUE),
            8_322
        );

        assert_eq!(HUF_READ_STATS_WORKSPACE_SIZE_U32, 219);
        assert_eq!(HUF_READ_STATS_WORKSPACE_SIZE, 876);
    }

    #[test]
    fn fse_readncount_rejects_tablelog_too_large() {
        // Construct a header whose first 4 bits (tableLog - MIN) produce
        // a resulting tableLog > FSE_TABLELOG_ABSOLUTE_MAX (15).
        // Trick: low nibble = 0xB → tableLog = 0xB + 5 = 16 > 15.
        let src = [0x0B, 0, 0, 0, 0, 0, 0, 0];
        let mut counts = [0i16; 256];
        let mut maxSV: u32 = 255;
        let mut tl: u32 = 0;
        let rc = FSE_readNCount(&mut counts, &mut maxSV, &mut tl, &src);
        assert!(crate::common::error::ERR_isError(rc));
        assert_eq!(
            crate::common::error::ERR_getErrorCode(rc),
            ErrorCode::TableLogTooLarge
        );
    }

    #[test]
    fn fse_readncount_rejects_counter_buffer_too_small() {
        let src = [0x00, 0, 0, 0, 0, 0, 0, 0];
        let mut counts = [0i16; 1];
        let mut maxSV: u32 = 255;
        let mut tl: u32 = 0;
        let rc = FSE_readNCount(&mut counts, &mut maxSV, &mut tl, &src);
        assert!(crate::common::error::ERR_isError(rc));
        assert_eq!(
            crate::common::error::ERR_getErrorCode(rc),
            ErrorCode::MaxSymbolValueTooSmall
        );
    }

    #[test]
    fn fse_readncount_no_clear_still_rejects_counter_buffer_too_small() {
        let src = [0x00, 0, 0, 0, 0, 0, 0, 0];
        let mut counts = [0i16; 1];
        let mut maxSV: u32 = 255;
        let mut tl: u32 = 0;
        let rc = FSE_readNCount_no_clear(&mut counts, &mut maxSV, &mut tl, &src);
        assert!(crate::common::error::ERR_isError(rc));
        assert_eq!(
            crate::common::error::ERR_getErrorCode(rc),
            ErrorCode::MaxSymbolValueTooSmall
        );
    }

    #[test]
    fn fse_readncount_short_header_recursion_preserves_clear_counts() {
        let src = [0x0B];
        let mut counts = [7i16; 4];
        let mut maxSV: u32 = 3;
        let mut tl: u32 = 0;
        let rc = FSE_readNCount_no_clear(&mut counts, &mut maxSV, &mut tl, &src);
        assert!(crate::common::error::ERR_isError(rc));
        assert_eq!(counts, [7i16; 4]);

        let mut counts = [7i16; 4];
        let mut maxSV: u32 = 3;
        let mut tl: u32 = 0;
        let rc = FSE_readNCount(&mut counts, &mut maxSV, &mut tl, &src);
        assert!(crate::common::error::ERR_isError(rc));
        assert_eq!(counts, [0i16; 4]);
    }

    #[test]
    fn huf_readstats_rejects_empty_input() {
        let mut hw = [0u8; 256];
        let mut rs = [0u32; 16];
        let mut ns: u32 = 0;
        let mut tl: u32 = 0;
        let mut wksp = [0u32; 64];
        let rc = HUF_readStats_wksp(&mut hw, 256, &mut rs, &mut ns, &mut tl, &[], &mut wksp, 0);
        assert!(crate::common::error::ERR_isError(rc));
    }

    #[test]
    fn huf_readstats_rejects_truncated_raw_header() {
        // First byte ≥ 128 selects the raw branch; claim oSize=10 → iSize=6,
        // need srcSize ≥ 7. Provide only 2 bytes.
        let src = [128 + 10, 0u8];
        let mut hw = [0u8; 256];
        let mut rs = [0u32; 16];
        let mut ns: u32 = 0;
        let mut tl: u32 = 0;
        let mut wksp = [0u32; 64];
        let rc = HUF_readStats_wksp(&mut hw, 256, &mut rs, &mut ns, &mut tl, &src, &mut wksp, 0);
        assert!(crate::common::error::ERR_isError(rc));
    }

    #[test]
    fn huf_readstats_rejects_invalid_output_capacity() {
        let src = [128 + 2, 0x11, 0x11];
        let mut hw = [0u8; 1];
        let mut rs = [0u32; 16];
        let mut ns: u32 = 0;
        let mut tl: u32 = 0;
        let mut wksp = [0u32; HUF_READ_STATS_WORKSPACE_SIZE_U32];
        let rc = HUF_readStats_wksp(&mut hw, 0, &mut rs, &mut ns, &mut tl, &src, &mut wksp, 0);
        assert!(crate::common::error::ERR_isError(rc));
        assert_eq!(
            crate::common::error::ERR_getErrorCode(rc),
            ErrorCode::CorruptionDetected
        );
    }
}
