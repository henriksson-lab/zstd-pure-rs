//! Translation of `lib/common/entropy_common.c`.
//!
//! Covers:
//! - `FSE_readNCount` (parses an FSE normalized-count header)
//! - `HUF_readStats` / `HUF_readStats_wksp` (parses a HUF tree descriptor)
//!   — depends on `FSE_decompress_wksp_bmi2`, so the FSE-compressed
//!   branch panics until `fse_decompress.rs` is implemented. The raw
//!   branch (`iSize >= 128`) works today.
//! - thin version/error accessors

#![allow(unused_variables)]

use crate::common::bits::{ZSTD_countTrailingZeros32, ZSTD_highbit32};
use crate::common::error::{ErrorCode, ERROR};
use crate::common::mem::MEM_readLE32;

pub const FSE_VERSION_NUMBER: u32 = 5;
pub const FSE_MIN_TABLELOG: u32 = 5;
pub const FSE_TABLELOG_ABSOLUTE_MAX: u32 = 15;

pub const HUF_TABLELOG_MAX: u32 = 12;
pub const HUF_TABLELOG_ABSOLUTE_MAX: u32 = 12;
pub const HUF_flags_bmi2: i32 = 1 << 0;

pub fn FSE_versionNumber() -> u32 {
    FSE_VERSION_NUMBER
}

pub fn FSE_isError(code: usize) -> u32 {
    crate::common::error::ERR_isError(code) as u32
}

pub fn FSE_getErrorName(code: usize) -> &'static str {
    crate::common::error::ERR_getErrorName(code)
}

pub fn HUF_isError(code: usize) -> u32 {
    crate::common::error::ERR_isError(code) as u32
}

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
    let hbSize = src.len();

    // Upstream: if hbSize < 8, pad into an 8-byte zero-filled buffer and
    // recurse. We do the same — the returned byte-count is bounded by
    // the real input size and the validator catches over-read.
    if hbSize < 8 {
        let mut buffer = [0u8; 8];
        buffer[..hbSize].copy_from_slice(src);
        let countSize = FSE_readNCount(normalizedCounter, maxSVPtr, tableLogPtr, &buffer);
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
    let maxSV1 = *maxSVPtr + 1;
    let mut previous0 = false;

    // Zero out the counter; symbols not present stay at 0.
    for c in normalizedCounter.iter_mut().take(maxSV1 as usize) {
        *c = 0;
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
    *maxSVPtr = charnum - 1;
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
    let mut wksp = [0u32; 256]; // upper-bound; real size FSE_DECOMPRESS_WKSP_SIZE_U32
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

/// Port of `HUF_readStats_body` (and its BMI2 wrapper). The
/// FSE-compressed branch reaches into `fse_decompress`, which is not
/// yet implemented — that branch panics with `yet to be translated`
/// until it lands.
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
        let payload = &src[1..];
        let mut n = 0;
        while n < oSize {
            huffWeight[n] = payload[n / 2] >> 4;
            if n + 1 < oSize {
                huffWeight[n + 1] = payload[n / 2] & 0x0F;
            } else {
                // Upstream writes both in a pair but the upper bound is
                // oSize, so the last iteration of the +=2 loop happens
                // only when n+1 < oSize; when oSize is odd, writing
                // huffWeight[n+1] would step past the valid range.
                // We mirror the upstream loop by writing only when
                // n+1<oSize.
            }
            n += 2;
        }
    } else {
        // FSE-compressed header. Requires `FSE_decompress_wksp_bmi2`.
        if iSize + 1 > src.len() {
            return ERROR(ErrorCode::SrcSizeWrong);
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

/// Helper: view a `&mut [u32]` as `&mut [u8]` for passing to
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
    fn fse_version_is_five() {
        assert_eq!(FSE_versionNumber(), 5);
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
}
