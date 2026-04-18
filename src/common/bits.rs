//! Translation of `lib/common/bits.h`.
//!
//! `ZSTD_countTrailingZeros*` / `ZSTD_countLeadingZeros*` map directly to
//! Rust's `u32::trailing_zeros` / `leading_zeros` intrinsics. Upstream
//! keeps De-Bruijn fallbacks for the pre-`__builtin_ctz` era; we rely on
//! LLVM to lower to `tzcnt`/`lzcnt` where the target supports them.

use crate::common::mem::{is_little_endian, MEM_64bits};

#[inline]
pub fn ZSTD_countTrailingZeros32(val: u32) -> u32 {
    debug_assert!(val != 0);
    val.trailing_zeros()
}

#[inline]
pub fn ZSTD_countLeadingZeros32(val: u32) -> u32 {
    debug_assert!(val != 0);
    val.leading_zeros()
}

#[inline]
pub fn ZSTD_countTrailingZeros64(val: u64) -> u32 {
    debug_assert!(val != 0);
    val.trailing_zeros()
}

#[inline]
pub fn ZSTD_countLeadingZeros64(val: u64) -> u32 {
    debug_assert!(val != 0);
    val.leading_zeros()
}

/// `ZSTD_NbCommonBytes(size_t val)` — returns the number of matched-prefix
/// bytes hidden inside `val` given the CPU's byte order.
#[inline]
pub fn ZSTD_NbCommonBytes(val: usize) -> u32 {
    if is_little_endian() {
        if MEM_64bits() != 0 {
            ZSTD_countTrailingZeros64(val as u64) >> 3
        } else {
            ZSTD_countTrailingZeros32(val as u32) >> 3
        }
    } else if MEM_64bits() != 0 {
        ZSTD_countLeadingZeros64(val as u64) >> 3
    } else {
        ZSTD_countLeadingZeros32(val as u32) >> 3
    }
}

#[inline]
pub fn ZSTD_highbit32(val: u32) -> u32 {
    debug_assert!(val != 0);
    31 - ZSTD_countLeadingZeros32(val)
}

/// Alias for the FSE/HUF code that calls this directly.
#[inline]
pub fn BIT_highbit32(val: u32) -> u32 {
    ZSTD_highbit32(val)
}

#[inline]
pub const fn ZSTD_rotateRight_U64(value: u64, count: u32) -> u64 {
    debug_assert!(count < 64);
    value.rotate_right(count)
}

#[inline]
pub const fn ZSTD_rotateRight_U32(value: u32, count: u32) -> u32 {
    debug_assert!(count < 32);
    value.rotate_right(count)
}

#[inline]
pub const fn ZSTD_rotateRight_U16(value: u16, count: u32) -> u16 {
    debug_assert!(count < 16);
    value.rotate_right(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctz32_basic() {
        assert_eq!(ZSTD_countTrailingZeros32(1), 0);
        assert_eq!(ZSTD_countTrailingZeros32(2), 1);
        assert_eq!(ZSTD_countTrailingZeros32(0x8000_0000), 31);
    }

    #[test]
    fn clz32_basic() {
        assert_eq!(ZSTD_countLeadingZeros32(1), 31);
        assert_eq!(ZSTD_countLeadingZeros32(0x8000_0000), 0);
    }

    #[test]
    fn highbit32() {
        assert_eq!(ZSTD_highbit32(1), 0);
        assert_eq!(ZSTD_highbit32(2), 1);
        assert_eq!(ZSTD_highbit32(255), 7);
        assert_eq!(ZSTD_highbit32(0x8000_0000), 31);
    }

    #[test]
    fn common_bytes_le_64() {
        // On LE 64-bit: the number of trailing zero BYTES is (tzcnt >> 3).
        if cfg!(target_endian = "little") && cfg!(target_pointer_width = "64") {
            assert_eq!(ZSTD_NbCommonBytes(0x01), 0);
            assert_eq!(ZSTD_NbCommonBytes(0x0100), 1);
            assert_eq!(ZSTD_NbCommonBytes(0x0100_0000_0000_0000), 7);
        }
    }

    #[test]
    fn rotate_right_u32() {
        assert_eq!(ZSTD_rotateRight_U32(0x0000_00ff, 8), 0xff00_0000);
        assert_eq!(ZSTD_rotateRight_U32(0x1234_5678, 0), 0x1234_5678);
    }

    #[test]
    fn ctz_clz_64_basic() {
        // Mirrors the 32-bit tests for the u64 intrinsics.
        assert_eq!(ZSTD_countTrailingZeros64(1), 0);
        assert_eq!(ZSTD_countTrailingZeros64(1 << 63), 63);
        assert_eq!(ZSTD_countLeadingZeros64(1), 63);
        assert_eq!(ZSTD_countLeadingZeros64(1 << 63), 0);
    }

    #[test]
    fn rotate_right_u16_and_u64() {
        assert_eq!(ZSTD_rotateRight_U16(0x00FF, 8), 0xFF00);
        assert_eq!(ZSTD_rotateRight_U16(0xABCD, 0), 0xABCD);
        assert_eq!(
            ZSTD_rotateRight_U64(0x0000_0000_0000_00FF, 8),
            0xFF00_0000_0000_0000,
        );
        assert_eq!(
            ZSTD_rotateRight_U64(0x1234_5678_9ABC_DEF0, 0),
            0x1234_5678_9ABC_DEF0,
        );
    }
}
