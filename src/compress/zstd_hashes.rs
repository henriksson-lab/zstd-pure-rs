//! Port of the hash functions from `lib/compress/zstd_compress_internal.h`.
//!
//! These are tiny multiplicative hashes used by every matcher
//! (`zstd_fast`, `zstd_double_fast`, `zstd_lazy`, `zstd_opt`,
//! `zstd_ldm`). They all follow the pattern
//!   `(value * prime) >> (N - hBits)`
//! with a size-specific prime so that keys of different widths spread
//! differently across the output bit range. `_PtrS` variants xor in a
//! salt before the right-shift — used by the long-distance matcher.
//!
//! Upstream keeps all of these static-inline in `zstd_compress_internal.h`
//! so callers see a single translation unit; we break them out into a
//! dedicated Rust module because Rust has no analogous inlining
//! requirement.

use crate::common::bits::ZSTD_NbCommonBytes;
use crate::common::mem::{MEM_64bits, MEM_readLE32, MEM_readLE64};

// Primes from upstream — must not drift.
pub const PRIME_3BYTES: u32 = 506_832_829;
pub const PRIME_4BYTES: u32 = 2_654_435_761;
pub const PRIME_5BYTES: u64 = 889_523_592_379;
pub const PRIME_6BYTES: u64 = 227_718_039_650_203;
pub const PRIME_7BYTES: u64 = 58_295_818_150_454_627;
pub const PRIME_8BYTES: u64 = 0xCF1B_BCDC_B7A5_6463;

/// Upstream `ZSTD_ROLL_HASH_CHAR_OFFSET` (zstd_compress_internal.h:979).
/// Byte bias added before multiplication so that leading zero bytes
/// still propagate into the rolling hash.
pub const ZSTD_ROLL_HASH_CHAR_OFFSET: u64 = 10;

/// Port of `ZSTD_ipow` (zstd_compress_internal.h:968). Fast u64
/// exponentiation by squaring.
#[inline]
pub fn ZSTD_ipow(mut base: u64, mut exponent: u64) -> u64 {
    let mut power: u64 = 1;
    while exponent != 0 {
        if exponent & 1 != 0 {
            power = power.wrapping_mul(base);
        }
        exponent >>= 1;
        base = base.wrapping_mul(base);
    }
    power
}

/// Port of `ZSTD_rollingHash_append` (zstd_compress_internal.h:984).
/// Multiplies `hash` by `prime8bytes` for each byte, adding the byte
/// (with `ZSTD_ROLL_HASH_CHAR_OFFSET` bias).
#[inline]
pub fn ZSTD_rollingHash_append(mut hash: u64, buf: &[u8]) -> u64 {
    for &b in buf {
        hash = hash.wrapping_mul(PRIME_8BYTES);
        hash = hash.wrapping_add(b as u64 + ZSTD_ROLL_HASH_CHAR_OFFSET);
    }
    hash
}

/// Port of `ZSTD_rollingHash_compute` (zstd_compress_internal.h:998).
/// Rolling hash over the whole buffer, seeded at 0.
#[inline]
pub fn ZSTD_rollingHash_compute(buf: &[u8]) -> u64 {
    ZSTD_rollingHash_append(0, buf)
}

/// Port of `ZSTD_rollingHash_primePower` (zstd_compress_internal.h:1007).
/// Pre-computes the multiplicative prime power for a window of `length`
/// bytes. Pass the result to `ZSTD_rollingHash_rotate`.
#[inline]
pub fn ZSTD_rollingHash_primePower(length: u32) -> u64 {
    debug_assert!(length > 0);
    ZSTD_ipow(PRIME_8BYTES, length.wrapping_sub(1) as u64)
}

/// Port of `ZSTD_rollingHash_rotate` (zstd_compress_internal.h:1015).
/// Slide the window by one byte: remove `toRemove` from the front,
/// add `toAdd` at the back.
#[inline]
pub fn ZSTD_rollingHash_rotate(hash: u64, toRemove: u8, toAdd: u8, primePower: u64) -> u64 {
    let removed = (toRemove as u64 + ZSTD_ROLL_HASH_CHAR_OFFSET).wrapping_mul(primePower);
    let after_remove = hash.wrapping_sub(removed);
    let after_mult = after_remove.wrapping_mul(PRIME_8BYTES);
    after_mult.wrapping_add(toAdd as u64 + ZSTD_ROLL_HASH_CHAR_OFFSET)
}

/// Port of `ZSTD_hash3`. 3-byte multiplicative hash; left-shifts `u`
/// by 8 so the low 24 bits act as the 3-byte key, then mixes with
/// `PRIME_3BYTES`, salts with `s`, and right-shifts to `h` bits.
#[inline]
pub fn ZSTD_hash3(u: u32, h: u32, s: u32) -> u32 {
    debug_assert!(h <= 32);
    (((u << 8).wrapping_mul(PRIME_3BYTES)) ^ s) >> (32 - h)
}

/// Port of `ZSTD_hash3Ptr`. Reads 3 bytes from `ptr` (LE) and hashes
/// them with `ZSTD_hash3`, salt = 0. Used only by `zstd_opt`.
#[inline]
pub fn ZSTD_hash3Ptr(ptr: &[u8], h: u32) -> usize {
    ZSTD_hash3(MEM_readLE32(ptr), h, 0) as usize
}

/// Port of `ZSTD_hash3PtrS`. Salted variant of `ZSTD_hash3Ptr`.
#[inline]
pub fn ZSTD_hash3PtrS(ptr: &[u8], h: u32, s: u32) -> usize {
    ZSTD_hash3(MEM_readLE32(ptr), h, s) as usize
}

/// Port of `ZSTD_hash4`. 4-byte multiplicative hash by `PRIME_4BYTES`
/// with salt `s`, reduced to `h` bits.
#[inline]
pub fn ZSTD_hash4(u: u32, h: u32, s: u32) -> u32 {
    debug_assert!(h <= 32);
    (u.wrapping_mul(PRIME_4BYTES) ^ s) >> (32 - h)
}

/// Port of `ZSTD_hash4Ptr`. Reads 4 bytes from `ptr` (LE) and hashes
/// with `ZSTD_hash4`, salt = 0.
#[inline]
pub fn ZSTD_hash4Ptr(ptr: &[u8], h: u32) -> usize {
    ZSTD_hash4(MEM_readLE32(ptr), h, 0) as usize
}

/// Port of `ZSTD_hash4PtrS`. Salted variant of `ZSTD_hash4Ptr`.
#[inline]
pub fn ZSTD_hash4PtrS(ptr: &[u8], h: u32, s: u32) -> usize {
    ZSTD_hash4(MEM_readLE32(ptr), h, s) as usize
}

/// Port of `ZSTD_hash5`. 5-byte multiplicative hash; left-shifts `u`
/// by 24 to anchor the 5-byte key in the high 40 bits, then mixes
/// with `PRIME_5BYTES`, salts with `s`, reduces to `h` bits.
#[inline]
pub fn ZSTD_hash5(u: u64, h: u32, s: u64) -> usize {
    debug_assert!(h <= 64);
    ((((u << 24).wrapping_mul(PRIME_5BYTES)) ^ s) >> (64 - h)) as usize
}

/// Port of `ZSTD_hash5Ptr`. Reads 8 bytes (LE), hashes with `ZSTD_hash5`,
/// salt = 0; only the low 5 of the 8 bytes matter after the shift.
#[inline]
pub fn ZSTD_hash5Ptr(p: &[u8], h: u32) -> usize {
    ZSTD_hash5(MEM_readLE64(p), h, 0)
}

/// Port of `ZSTD_hash5PtrS`. Salted variant of `ZSTD_hash5Ptr`.
#[inline]
pub fn ZSTD_hash5PtrS(p: &[u8], h: u32, s: u64) -> usize {
    ZSTD_hash5(MEM_readLE64(p), h, s)
}

/// Port of `ZSTD_hash6`. 6-byte multiplicative hash, analogous to
/// `ZSTD_hash5` with a 16-bit pre-shift and `PRIME_6BYTES`.
#[inline]
pub fn ZSTD_hash6(u: u64, h: u32, s: u64) -> usize {
    debug_assert!(h <= 64);
    ((((u << 16).wrapping_mul(PRIME_6BYTES)) ^ s) >> (64 - h)) as usize
}

/// Port of `ZSTD_hash6Ptr`. Reads 8 bytes (LE), hashes with `ZSTD_hash6`,
/// salt = 0.
#[inline]
pub fn ZSTD_hash6Ptr(p: &[u8], h: u32) -> usize {
    ZSTD_hash6(MEM_readLE64(p), h, 0)
}

/// Port of `ZSTD_hash6PtrS`. Salted variant of `ZSTD_hash6Ptr`.
#[inline]
pub fn ZSTD_hash6PtrS(p: &[u8], h: u32, s: u64) -> usize {
    ZSTD_hash6(MEM_readLE64(p), h, s)
}

/// Port of `ZSTD_hash7`. 7-byte multiplicative hash, analogous to
/// `ZSTD_hash5` with an 8-bit pre-shift and `PRIME_7BYTES`.
#[inline]
pub fn ZSTD_hash7(u: u64, h: u32, s: u64) -> usize {
    debug_assert!(h <= 64);
    ((((u << 8).wrapping_mul(PRIME_7BYTES)) ^ s) >> (64 - h)) as usize
}

/// Port of `ZSTD_hash7Ptr`. Reads 8 bytes (LE), hashes with `ZSTD_hash7`,
/// salt = 0.
#[inline]
pub fn ZSTD_hash7Ptr(p: &[u8], h: u32) -> usize {
    ZSTD_hash7(MEM_readLE64(p), h, 0)
}

/// Port of `ZSTD_hash7PtrS`. Salted variant of `ZSTD_hash7Ptr`.
#[inline]
pub fn ZSTD_hash7PtrS(p: &[u8], h: u32, s: u64) -> usize {
    ZSTD_hash7(MEM_readLE64(p), h, s)
}

/// Port of `ZSTD_hash8`. Full 64-bit multiplicative hash by
/// `PRIME_8BYTES` with salt `s`, reduced to `h` bits.
#[inline]
pub fn ZSTD_hash8(u: u64, h: u32, s: u64) -> usize {
    debug_assert!(h <= 64);
    ((u.wrapping_mul(PRIME_8BYTES) ^ s) >> (64 - h)) as usize
}

/// Port of `ZSTD_hash8Ptr`. Reads 8 bytes (LE), hashes with `ZSTD_hash8`,
/// salt = 0. Used by the double-fast long table.
#[inline]
pub fn ZSTD_hash8Ptr(p: &[u8], h: u32) -> usize {
    ZSTD_hash8(MEM_readLE64(p), h, 0)
}

/// Port of `ZSTD_hash8PtrS`. Salted variant of `ZSTD_hash8Ptr`.
#[inline]
pub fn ZSTD_hash8PtrS(p: &[u8], h: u32, s: u64) -> usize {
    ZSTD_hash8(MEM_readLE64(p), h, s)
}

/// Port of `ZSTD_hashPtr`. Dispatch to the MLS-specific hash.
#[inline(always)]
pub fn ZSTD_hashPtr(p: &[u8], hBits: u32, mls: u32) -> usize {
    debug_assert!(hBits <= 32);
    match mls {
        5 => ZSTD_hash5Ptr(p, hBits),
        6 => ZSTD_hash6Ptr(p, hBits),
        7 => ZSTD_hash7Ptr(p, hBits),
        8 => ZSTD_hash8Ptr(p, hBits),
        _ => ZSTD_hash4Ptr(p, hBits), // 4 is the default
    }
}

/// Port of `ZSTD_hashPtrSalted`.
#[inline]
pub fn ZSTD_hashPtrSalted(p: &[u8], hBits: u32, mls: u32, hashSalt: u64) -> usize {
    debug_assert!(hBits <= 32);
    match mls {
        5 => ZSTD_hash5PtrS(p, hBits, hashSalt),
        6 => ZSTD_hash6PtrS(p, hBits, hashSalt),
        7 => ZSTD_hash7PtrS(p, hBits, hashSalt),
        8 => ZSTD_hash8PtrS(p, hBits, hashSalt),
        _ => ZSTD_hash4PtrS(p, hBits, hashSalt as u32),
    }
}

// ---- Common-prefix match length --------------------------------------

/// Port of `ZSTD_count_2segments`. Match-length counter that can
/// cross from an ext-dict segment into the current prefix. When the
/// forward count reaches the end of the dict segment (`mEnd`), it
/// resumes counting from `iStart` in the prefix.
///
/// Rust signature note: upstream takes five raw `BYTE*` pointers,
/// assuming `ip`/`iEnd`/`iStart` live in one buffer and
/// `match`/`mEnd` in another. The Rust port takes two separate
/// slices (`input_buf` for the current input, `dict_buf` for the
/// ext-dict) plus byte offsets into each.
pub fn ZSTD_count_2segments(
    input_buf: &[u8],
    ip_pos: usize,
    iend_pos: usize,
    istart_pos: usize,
    dict_buf: &[u8],
    match_pos: usize,
    mend_pos: usize,
) -> usize {
    // vEnd is min(mEnd, iEnd) in the SAME (input) coordinate system,
    // so we compute it as "how much of the input we can compare before
    // the dict segment runs out".
    let dict_remaining = mend_pos - match_pos;
    let vend_pos = (ip_pos + dict_remaining).min(iend_pos);

    let mut matchLength = 0usize;
    while ip_pos + matchLength < vend_pos
        && match_pos + matchLength < mend_pos
        && input_buf[ip_pos + matchLength] == dict_buf[match_pos + matchLength]
    {
        matchLength += 1;
    }

    if match_pos + matchLength != mend_pos {
        return matchLength;
    }

    matchLength + ZSTD_count(input_buf, ip_pos + matchLength, istart_pos, iend_pos)
}

/// Port of `ZSTD_count`. Returns the number of leading bytes that
/// match between `buf[in_pos..]` and `buf[match_pos..]` up to
/// `buf[in_limit]`. Upstream's pointer-based loop becomes index-based
/// here; semantics are identical: read size_t words, XOR, count
/// common bytes on first mismatch.
///
/// Rust signature note: upstream takes three raw `BYTE*` pointers
/// into the same buffer. We take one `&[u8]` + three byte indices.
///
/// Marked `#[inline(always)]` to match upstream's `MEM_STATIC FORCE_INLINE_TEMPLATE`
/// — this is the per-match inner kernel of every strategy's matcher.
/// Profile showed `#[inline]` alone wasn't enough: ZSTD_count's call
/// boundary (push/pop/ret) consumed ~40% of its self-time at L1
/// because LLVM was reluctant to fully inline the moderate-sized
/// body. Forcing inline lets the matchers pay only for the actual
/// 8-byte-stride compare loop.
#[inline(always)]
pub fn ZSTD_count(buf: &[u8], in_pos: usize, match_pos: usize, in_limit: usize) -> usize {
    let word = core::mem::size_of::<usize>();
    let start = in_pos;
    let mut pIn = in_pos;
    let mut pMatch = match_pos;
    debug_assert!(in_limit >= word - 1);
    let inLoopLimit = in_limit - (word - 1);
    let base = buf.as_ptr();

    if pIn < inLoopLimit {
        debug_assert!(pMatch + word <= buf.len());
        debug_assert!(pIn + word <= buf.len());
        let diff = unsafe {
            (base.add(pMatch) as *const usize).read_unaligned()
                ^ (base.add(pIn) as *const usize).read_unaligned()
        };
        if diff != 0 {
            return ZSTD_NbCommonBytes(diff) as usize;
        }
        pIn += word;
        pMatch += word;
        while pIn < inLoopLimit {
            debug_assert!(pMatch + word <= buf.len());
            debug_assert!(pIn + word <= buf.len());
            let diff = unsafe {
                (base.add(pMatch) as *const usize).read_unaligned()
                    ^ (base.add(pIn) as *const usize).read_unaligned()
            };
            if diff == 0 {
                pIn += word;
                pMatch += word;
                continue;
            }
            pIn += ZSTD_NbCommonBytes(diff) as usize;
            return pIn - start;
        }
    }
    unsafe {
        if MEM_64bits() != 0
            && pIn + 3 < in_limit
            && (base.add(pMatch) as *const u32).read_unaligned()
                == (base.add(pIn) as *const u32).read_unaligned()
        {
            pIn += 4;
            pMatch += 4;
        }
        if pIn + 1 < in_limit
            && (base.add(pMatch) as *const u16).read_unaligned()
                == (base.add(pIn) as *const u16).read_unaligned()
        {
            pIn += 2;
            pMatch += 2;
        }
        if pIn < in_limit && *base.add(pMatch) == *base.add(pIn) {
            pIn += 1;
        }
    }
    pIn - start
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rollingHash_rotate_equals_recompute_on_slide() {
        // Core invariant of the rolling hash: sliding the window by
        // one byte via rotate() produces the same hash as recomputing
        // from scratch on the new slice. Pin against a simple 4-byte
        // window rotating through a 10-byte buffer.
        let buf = b"rolling123";
        let window: u32 = 4;
        let primePower = ZSTD_rollingHash_primePower(window);
        let mut hash = ZSTD_rollingHash_compute(&buf[..window as usize]);
        for i in 0..(buf.len() - window as usize) {
            hash = ZSTD_rollingHash_rotate(
                hash,
                buf[i],                   // toRemove = leading byte
                buf[i + window as usize], // toAdd = new trailing byte
                primePower,
            );
            let fresh = ZSTD_rollingHash_compute(&buf[i + 1..i + 1 + window as usize]);
            assert_eq!(hash, fresh, "slide {i}");
        }
    }

    #[test]
    fn count_2segments_stops_before_dict_end_on_mismatch() {
        // If the first segment mismatches before `mEnd`, upstream returns
        // that count and must not continue from `iStart`.
        let input = b"HELLOworld";
        let dict = b"HELxO";
        let n = ZSTD_count_2segments(input, 0, input.len(), 0, dict, 0, dict.len());
        assert_eq!(n, 3);
    }

    #[test]
    fn count_2segments_wraps_into_prefix_exactly() {
        // C convention:
        //   first compare `ip` against the ext-dict match until `mEnd`;
        //   if that reaches `mEnd`, continue with `ip + matchLength`
        //   against `iStart` in the current prefix.
        //
        // input prefix at iStart=0: "def..."
        // current ip=3:              "abcdefQ"
        // dict match:                "abc"
        // Result: 3 bytes from dict + 3 bytes from prefix.
        let input = b"defabcdefQ";
        let dict = b"abc";
        let n = ZSTD_count_2segments(input, 3, input.len(), 0, dict, 0, dict.len());
        assert_eq!(n, 6);
    }

    #[test]
    fn count_2segments_treats_mend_as_hard_dict_boundary() {
        // Upstream bounds the first leg with vEnd = min(ip + (mEnd - match), iEnd).
        // Extra bytes present after mEnd in the dict buffer must not be consumed;
        // after reaching mEnd, comparison resumes from iStart in the prefix.
        let input = b"defabcdefQ";
        let dict = b"abcXXX";
        let n = ZSTD_count_2segments(input, 3, input.len(), 0, dict, 0, 3);
        assert_eq!(n, 6);
    }

    #[test]
    fn hashptr_bit_range_respected() {
        // Hash output must fit within `hBits` bits regardless of input.
        let buf = [0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04];
        for mls in [4u32, 5, 6, 7, 8] {
            for hBits in [6u32, 12, 20, 32] {
                let h = ZSTD_hashPtr(&buf, hBits, mls);
                assert!(
                    h < (1usize << hBits.min(62)),
                    "mls={mls} hBits={hBits} h={h:#x}"
                );
            }
        }
    }

    #[test]
    fn hash4_anchored_to_upstream_formula() {
        // (u * prime) >> (32 - hBits). For u=1 and hBits=8:
        //   (1 * 2654435761) >> 24 = 158 (0x9E).
        assert_eq!(ZSTD_hash4(1, 8, 0), 158);
        assert_eq!(ZSTD_hash4(0, 32, 0), 0);
    }

    #[test]
    fn hash_family_matches_upstream_formula_vectors() {
        let buf = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let u32v = MEM_readLE32(&buf);
        let u64v = MEM_readLE64(&buf);

        assert_eq!(ZSTD_hash3(u32v, 16, 0), 48161);
        assert_eq!(ZSTD_hash4(u32v, 16, 0), 15933);
        assert_eq!(ZSTD_hash5(u64v, 16, 0), 22295);
        assert_eq!(ZSTD_hash6(u64v, 16, 0), 4145);
        assert_eq!(ZSTD_hash7(u64v, 16, 0), 951);
        assert_eq!(ZSTD_hash8(u64v, 16, 0), 64852);

        assert_eq!(ZSTD_hash4(u32v, 16, 0xDEAD_BEEF), 57488);
        assert_eq!(ZSTD_hash8(u64v, 16, 0xDEAD_BEEF_CAFE_BABE), 9209);
    }

    #[test]
    fn hashPtrSalted_dispatches_to_matching_saltedPtr_helper() {
        // Symmetric with the un-salted dispatch test. Note the mls=4
        // path takes `s: u32` (downcast from the u64 hashSalt).
        let buf = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let hBits = 12u32;
        let salt = 0xCAFE_BABE_DEAD_BEEFu64;
        assert_eq!(
            ZSTD_hashPtrSalted(&buf, hBits, 4, salt),
            ZSTD_hash4PtrS(&buf, hBits, salt as u32)
        );
        assert_eq!(
            ZSTD_hashPtrSalted(&buf, hBits, 5, salt),
            ZSTD_hash5PtrS(&buf, hBits, salt)
        );
        assert_eq!(
            ZSTD_hashPtrSalted(&buf, hBits, 6, salt),
            ZSTD_hash6PtrS(&buf, hBits, salt)
        );
        assert_eq!(
            ZSTD_hashPtrSalted(&buf, hBits, 7, salt),
            ZSTD_hash7PtrS(&buf, hBits, salt)
        );
        assert_eq!(
            ZSTD_hashPtrSalted(&buf, hBits, 8, salt),
            ZSTD_hash8PtrS(&buf, hBits, salt)
        );
    }

    #[test]
    fn hashPtr_dispatches_to_matching_mls_specific_helper() {
        // Verify `ZSTD_hashPtr(_, h, mls)` calls `hash<MLS>Ptr(_, h)`
        // for mls ∈ {5..=8} and falls back to hash4Ptr for mls=4
        // (or any unrecognized value). If dispatch ever drifts,
        // one strategy will silently load the wrong hash for MLS ≥ 5.
        let buf = [0x01u8, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let hBits = 16u32;
        assert_eq!(ZSTD_hashPtr(&buf, hBits, 4), ZSTD_hash4Ptr(&buf, hBits));
        assert_eq!(ZSTD_hashPtr(&buf, hBits, 5), ZSTD_hash5Ptr(&buf, hBits));
        assert_eq!(ZSTD_hashPtr(&buf, hBits, 6), ZSTD_hash6Ptr(&buf, hBits));
        assert_eq!(ZSTD_hashPtr(&buf, hBits, 7), ZSTD_hash7Ptr(&buf, hBits));
        assert_eq!(ZSTD_hashPtr(&buf, hBits, 8), ZSTD_hash8Ptr(&buf, hBits));
        // Unrecognized MLS falls through to 4.
        assert_eq!(ZSTD_hashPtr(&buf, hBits, 3), ZSTD_hash4Ptr(&buf, hBits));
        assert_eq!(ZSTD_hashPtr(&buf, hBits, 99), ZSTD_hash4Ptr(&buf, hBits));
    }

    #[test]
    fn hash_family_3_to_8_stay_within_hbits_range() {
        // For every MLS ∈ {3..=8}, `hash<MLS>` must output a value
        // strictly less than `1 << hBits`. Covers the full
        // ZSTD_hash3..ZSTD_hash8 family in one sweep — they're all
        // used by different strategies and each has its own prime.
        let buf = [0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89];
        let bits: u32 = 12;
        let max = 1usize << bits;
        assert!((ZSTD_hash3(MEM_readLE32(&buf), bits, 0) as usize) < max);
        assert!((ZSTD_hash4(MEM_readLE32(&buf), bits, 0) as usize) < max);
        assert!(ZSTD_hash5(MEM_readLE64(&buf), bits, 0) < max);
        assert!(ZSTD_hash6(MEM_readLE64(&buf), bits, 0) < max);
        assert!(ZSTD_hash7(MEM_readLE64(&buf), bits, 0) < max);
        assert!(ZSTD_hash8(MEM_readLE64(&buf), bits, 0) < max);

        // Two distinct inputs must produce distinct hashes (at the
        // bit-width tested they essentially always do — if not, the
        // hash constant is badly chosen).
        let buf2 = [0x00u8; 8];
        assert_ne!(ZSTD_hash4Ptr(&buf, bits), ZSTD_hash4Ptr(&buf2, bits));
        assert_ne!(ZSTD_hash8Ptr(&buf, bits), ZSTD_hash8Ptr(&buf2, bits));
    }

    #[test]
    fn hashptr_different_salts_diverge() {
        // Salt is XORed just before `>> (64 - hBits)`. For small hBits
        // the salt's low 32 bits get shifted out, so use a salt with
        // bits set near the top of the u64.
        let buf = [1u8; 8];
        let a = ZSTD_hashPtrSalted(&buf, 16, 5, 0);
        let b = ZSTD_hashPtrSalted(&buf, 16, 5, 0xDEAD_BEEF_0000_0000);
        assert_ne!(a, b);
    }

    #[test]
    fn count_identical_region() {
        // Seed buf with period-8 pattern "abcdefgh..." so offset-8
        // back-references find a full-buffer-length match.
        // ZSTD_count's matcher walks pIn forward up to `in_limit` and
        // pMatch at the same stride (pMatch < pIn so it stays valid).
        let mut buf = vec![0u8; 64];
        for (i, b) in buf.iter_mut().enumerate() {
            *b = b'a' + (i % 8) as u8;
        }
        // pIn=8, pMatch=0, in_limit=buf.len() → entire tail matches.
        let n = ZSTD_count(&buf, 8, 0, buf.len());
        assert_eq!(n, buf.len() - 8);
    }

    #[test]
    fn count_first_mismatch() {
        // Buf position 8 has 'a','a','a','b', position 0 has 'a','a','a','c'.
        let mut buf = vec![b'a'; 32];
        buf[3] = b'c';
        buf[11] = b'b';
        let n = ZSTD_count(&buf, 8, 0, buf.len());
        // Match length = 3 (first 3 bytes agree, fourth doesn't).
        assert_eq!(n, 3);
    }

    #[test]
    fn count_at_in_limit_stops() {
        let buf = [b'X'; 16];
        // pIn=4, pMatch=0, in_limit=10 → 6 matching bytes at most.
        let n = ZSTD_count(&buf, 4, 0, 10);
        assert_eq!(n, 6);
    }

    #[test]
    fn hash_different_mls_differ() {
        let buf = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let h4 = ZSTD_hashPtr(&buf, 16, 4);
        let h8 = ZSTD_hashPtr(&buf, 16, 8);
        // They should usually differ; primes are chosen to spread well.
        assert_ne!(h4, h8);
    }
}
