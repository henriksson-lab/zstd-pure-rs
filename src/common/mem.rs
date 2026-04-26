//! Translation of `lib/common/mem.h`.
//!
//! Upstream provides safe integer load/store helpers that respect native
//! endianness and explicit LE/BE variants. On all platforms zstd targets
//! in practice, plain unaligned memcpy is the reference implementation —
//! `core::ptr::read_unaligned` and `write_unaligned` in Rust generate the
//! same code.

#[inline(always)]
pub const fn is_little_endian() -> bool {
    cfg!(target_endian = "little")
}

/// Upstream `MEM_isLittleEndian`.
#[inline(always)]
pub const fn MEM_isLittleEndian() -> u32 {
    if is_little_endian() {
        1
    } else {
        0
    }
}

/// `MEM_64bits()` — constant `1` when `size_t` is 64-bit wide.
#[inline(always)]
pub const fn MEM_64bits() -> u32 {
    (core::mem::size_of::<usize>() == 8) as u32
}

/// `MEM_32bits()` — constant `1` when `size_t` is 32-bit wide.
#[inline(always)]
pub const fn MEM_32bits() -> u32 {
    (core::mem::size_of::<usize>() == 4) as u32
}

#[inline(always)]
pub fn MEM_read16(ptr: &[u8]) -> u16 {
    debug_assert!(ptr.len() >= 2);
    unsafe { (ptr.as_ptr() as *const u16).read_unaligned() }
}

#[inline(always)]
pub fn MEM_read24(ptr: &[u8]) -> u32 {
    // Upstream returns a U32 containing three bytes in natural endianness.
    MEM_read16(ptr) as u32 | ((ptr[2] as u32) << 16)
}

#[inline(always)]
pub fn MEM_read32(ptr: &[u8]) -> u32 {
    debug_assert!(ptr.len() >= 4);
    unsafe { (ptr.as_ptr() as *const u32).read_unaligned() }
}

#[inline(always)]
pub fn MEM_read64(ptr: &[u8]) -> u64 {
    debug_assert!(ptr.len() >= 8);
    unsafe { (ptr.as_ptr() as *const u64).read_unaligned() }
}

/// `size_t` load (`MEM_readST`): 4 or 8 bytes depending on target.
#[inline(always)]
pub fn MEM_readST(ptr: &[u8]) -> usize {
    if core::mem::size_of::<usize>() == 8 {
        MEM_read64(ptr) as usize
    } else {
        MEM_read32(ptr) as usize
    }
}

#[inline(always)]
pub fn MEM_write16(dst: &mut [u8], value: u16) {
    let b = value.to_ne_bytes();
    dst[..2].copy_from_slice(&b);
}

#[inline(always)]
pub fn MEM_write32(dst: &mut [u8], value: u32) {
    let b = value.to_ne_bytes();
    dst[..4].copy_from_slice(&b);
}

#[inline(always)]
pub fn MEM_write64(dst: &mut [u8], value: u64) {
    let b = value.to_ne_bytes();
    dst[..8].copy_from_slice(&b);
}

// ---- Little-endian ------------------------------------------------------

#[inline(always)]
pub fn MEM_readLE16(ptr: &[u8]) -> u16 {
    debug_assert!(ptr.len() >= 2);
    u16::from_le(unsafe { (ptr.as_ptr() as *const u16).read_unaligned() })
}

#[inline(always)]
pub fn MEM_readLE24(ptr: &[u8]) -> u32 {
    MEM_readLE16(ptr) as u32 | ((ptr[2] as u32) << 16)
}

#[inline(always)]
pub fn MEM_readLE32(ptr: &[u8]) -> u32 {
    debug_assert!(ptr.len() >= 4);
    u32::from_le(unsafe { (ptr.as_ptr() as *const u32).read_unaligned() })
}

#[inline(always)]
pub fn MEM_readLE64(ptr: &[u8]) -> u64 {
    debug_assert!(ptr.len() >= 8);
    u64::from_le(unsafe { (ptr.as_ptr() as *const u64).read_unaligned() })
}

#[inline(always)]
pub fn MEM_readLEST(ptr: &[u8]) -> usize {
    if core::mem::size_of::<usize>() == 8 {
        MEM_readLE64(ptr) as usize
    } else {
        MEM_readLE32(ptr) as usize
    }
}

#[inline(always)]
pub fn MEM_writeLE16(dst: &mut [u8], value: u16) {
    dst[..2].copy_from_slice(&value.to_le_bytes());
}

#[inline(always)]
pub fn MEM_writeLE24(dst: &mut [u8], value: u32) {
    MEM_writeLE16(dst, value as u16);
    dst[2] = (value >> 16) as u8;
}

#[inline(always)]
pub fn MEM_writeLE32(dst: &mut [u8], value: u32) {
    dst[..4].copy_from_slice(&value.to_le_bytes());
}

#[inline(always)]
pub fn MEM_writeLE64(dst: &mut [u8], value: u64) {
    dst[..8].copy_from_slice(&value.to_le_bytes());
}

#[inline(always)]
pub fn MEM_writeLEST(dst: &mut [u8], value: usize) {
    if core::mem::size_of::<usize>() == 8 {
        MEM_writeLE64(dst, value as u64);
    } else {
        MEM_writeLE32(dst, value as u32);
    }
}

// ---- Big-endian ---------------------------------------------------------

#[inline(always)]
pub fn MEM_readBE32(ptr: &[u8]) -> u32 {
    u32::from_be_bytes([ptr[0], ptr[1], ptr[2], ptr[3]])
}

#[inline(always)]
pub fn MEM_readBE64(ptr: &[u8]) -> u64 {
    u64::from_be_bytes([
        ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7],
    ])
}

#[inline(always)]
pub fn MEM_readBEST(ptr: &[u8]) -> usize {
    if core::mem::size_of::<usize>() == 8 {
        MEM_readBE64(ptr) as usize
    } else {
        MEM_readBE32(ptr) as usize
    }
}

#[inline(always)]
pub fn MEM_writeBE32(dst: &mut [u8], value: u32) {
    dst[..4].copy_from_slice(&value.to_be_bytes());
}

#[inline(always)]
pub fn MEM_writeBE64(dst: &mut [u8], value: u64) {
    dst[..8].copy_from_slice(&value.to_be_bytes());
}

#[inline(always)]
pub fn MEM_writeBEST(dst: &mut [u8], value: usize) {
    if core::mem::size_of::<usize>() == 8 {
        MEM_writeBE64(dst, value as u64);
    } else {
        MEM_writeBE32(dst, value as u32);
    }
}

// ---- swap helpers -------------------------------------------------------

#[inline(always)]
pub const fn MEM_swap32(x: u32) -> u32 {
    x.swap_bytes()
}

#[inline(always)]
pub const fn MEM_swap64(x: u64) -> u64 {
    x.swap_bytes()
}

#[inline(always)]
pub const fn MEM_swapST(x: usize) -> usize {
    x.swap_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn le_roundtrip_16() {
        let mut buf = [0u8; 2];
        MEM_writeLE16(&mut buf, 0x1234);
        assert_eq!(buf, [0x34, 0x12]);
        assert_eq!(MEM_readLE16(&buf), 0x1234);
    }

    #[test]
    fn MEM_read24_reads_native_endian_3_bytes() {
        // MEM_read24 is upstream's natural-endian 3-byte read. On LE
        // hosts (where our port runs) it should match MEM_readLE24.
        // Pins the semantic so a future refactor can't silently swap
        // in a BE composition.
        let bytes = [0xAB, 0xCD, 0xEF];
        // On LE, both readers produce the same value.
        assert_eq!(MEM_read24(&bytes), MEM_readLE24(&bytes));
        // Concrete value: 0xEFCDAB on LE.
        if is_little_endian() {
            assert_eq!(MEM_read24(&bytes), 0xEFCDAB);
        }
    }

    #[test]
    fn le_roundtrip_24() {
        let mut buf = [0u8; 3];
        MEM_writeLE24(&mut buf, 0x123456);
        assert_eq!(buf, [0x56, 0x34, 0x12]);
        assert_eq!(MEM_readLE24(&buf), 0x123456);
    }

    #[test]
    fn le_roundtrip_32() {
        let mut buf = [0u8; 4];
        MEM_writeLE32(&mut buf, 0xdeadbeef);
        assert_eq!(buf, [0xef, 0xbe, 0xad, 0xde]);
        assert_eq!(MEM_readLE32(&buf), 0xdeadbeef);
    }

    #[test]
    fn le_roundtrip_64() {
        let mut buf = [0u8; 8];
        MEM_writeLE64(&mut buf, 0x0123456789abcdef);
        assert_eq!(buf, [0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01]);
        assert_eq!(MEM_readLE64(&buf), 0x0123456789abcdef);
    }

    #[test]
    fn be_roundtrip_32() {
        let mut buf = [0u8; 4];
        MEM_writeBE32(&mut buf, 0xdeadbeef);
        assert_eq!(buf, [0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(MEM_readBE32(&buf), 0xdeadbeef);
    }

    #[test]
    fn is_le_is_const() {
        // Always true on x86/x86_64/aarch64. Guarded by cfg so big-endian
        // toolchains flip it.
        let b = MEM_isLittleEndian();
        assert!(b == 0 || b == 1);
    }

    #[test]
    fn best_roundtrip_matches_size_of_usize() {
        // `MEM_readBEST` / `MEM_writeBEST` dispatch on usize width,
        // parallel to the `*LEST` variants. On 64-bit targets they
        // route to BE64; on 32-bit to BE32. Verify write→read
        // roundtrip on the host's native width.
        let value: usize = 0x1234_5678;
        let mut buf = vec![0u8; 8];
        MEM_writeBEST(&mut buf, value);
        assert_eq!(MEM_readBEST(&buf), value);

        // First `size_of::<usize>()` bytes match writing via the
        // explicit BE helper.
        let mut explicit = vec![0u8; 8];
        if core::mem::size_of::<usize>() == 8 {
            MEM_writeBE64(&mut explicit, value as u64);
        } else {
            MEM_writeBE32(&mut explicit[..4], value as u32);
        }
        let w = core::mem::size_of::<usize>();
        assert_eq!(&buf[..w], &explicit[..w]);
    }

    #[test]
    fn be_roundtrip_64() {
        // BE64 writes MSB first, which is the byte order used for
        // xxhash secret mixing in some paths.
        let mut buf = [0u8; 8];
        MEM_writeBE64(&mut buf, 0x0123_4567_89ab_cdef);
        assert_eq!(buf, [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef]);
        assert_eq!(MEM_readBE64(&buf), 0x0123_4567_89ab_cdef);
    }

    #[test]
    fn st_roundtrips_match_size_of_usize() {
        // MEM_readST / MEM_readLEST dispatch on usize width. On our
        // only supported 64-bit target they should read 8 bytes; on
        // a 32-bit host they'd read 4. Assert the dispatch matches
        // the native size so a future 32-bit build catches regressions.
        let size = core::mem::size_of::<usize>();
        let mut buf = vec![0u8; 8];
        // Platform-native.
        let value_st: usize = 0xDEAD_BEEF;
        if size == 8 {
            MEM_writeLE64(&mut buf, value_st as u64);
        } else {
            MEM_writeLE32(&mut buf[..4], value_st as u32);
        }
        assert_eq!(MEM_readLEST(&buf), value_st);
        // Native endianness is LE on supported targets, so readST
        // should agree with readLEST.
        if MEM_isLittleEndian() == 1 {
            assert_eq!(MEM_readST(&buf), value_st);
        }

        // writeLEST round-trips through readLEST.
        let mut buf2 = vec![0u8; 8];
        MEM_writeLEST(&mut buf2, value_st);
        assert_eq!(MEM_readLEST(&buf2), value_st);
        // And the first `size` bytes match what we wrote via the
        // explicit-width helper.
        assert_eq!(&buf2[..size], &buf[..size]);
    }
}
