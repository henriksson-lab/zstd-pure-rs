//! Selected translations from `lib/common/zstd_internal.h` — the
//! shared helpers used by both compress and decompress paths.

use crate::common::mem::{MEM_read32, MEM_read64, MEM_write32, MEM_write64};

pub const WILDCOPY_OVERLENGTH: usize = 32;
pub const WILDCOPY_VECLEN: usize = 16;
pub const ZSTD_WORKSPACETOOLARGE_FACTOR: usize = 3;
pub const ZSTD_WORKSPACETOOLARGE_MAXDURATION: usize = 128;

/// Upstream `repStartValue[ZSTD_REP_NUM]`. The default repcode
/// history that every compressed block inherits before its first
/// sequence — `{1, 4, 8}` matches the three slots of upstream's
/// `ZSTD_compressedBlockState_t.rep[]`.
pub const repStartValue: [u32; 3] = [1, 4, 8];

/// Port of `ZSTD_invalidateRepCodes` body. Upstream resets all three
/// slots to zero so the next block can't reuse any repcode — useful
/// when the caller has committed a discontinuity upstream cannot see.
///
/// Rust-port note: upstream's signature is
/// `ZSTD_invalidateRepCodes(ZSTD_CCtx*)`; we take the `rep` slice
/// directly so the helper stays reusable from call sites that have
/// a `[u32; 3]` but no CCtx (e.g. tests, `ZSTD_compressFrame_fast`
/// standalone paths). The CCtx-receiving caller side invalidates
/// both `prev_rep` and `next_rep` explicitly via two calls.
#[inline]
pub fn ZSTD_invalidateRepCodes(rep: &mut [u32; 3]) {
    for r in rep.iter_mut() {
        *r = 0;
    }
}

/// Port of the rep-only part of `ZSTD_reset_compressedBlockState` —
/// initialize the rep history to `repStartValue`. The entropy-table
/// repeat-mode resets live with their owning types and are applied
/// by the caller (`ZSTD_entropyCTables_t::default()`).
#[inline]
pub fn ZSTD_reset_compressedBlockState_rep(rep: &mut [u32; 3]) {
    *rep = repStartValue;
}

/// Mirror of `ZSTD_overlap_e`. Upstream also declares (but doesn't
/// define) `ZSTD_overlap_dst_before_src`, so we preserve only the two
/// legitimate variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZSTD_overlap_e {
    ZSTD_no_overlap,
    ZSTD_overlap_src_before_dst,
}

/// Port of `ZSTD_selectAddr`. Upstream uses x86-specific inline asm
/// to encourage branchless codegen; semantically it is just a
/// conditional address select.
#[inline]
pub fn ZSTD_selectAddr<'a, T>(index: u32, lowLimit: u32, candidate: &'a T, backup: &'a T) -> &'a T {
    if index >= lowLimit {
        candidate
    } else {
        backup
    }
}

/// Port of `ZSTD_safecopyLiterals` (`zstd_compress_internal.h:700`).
/// Copies bytes from `src[ip..iend]` to `dst[op..]`, using a fast
/// non-overlapping bulk copy up to `ilimit_w` and a safe byte-by-byte
/// tail past it. Only called when the sequence extends past
/// `ilimit_w`, so the tail is the hot path.
///
/// Rust-port note: upstream takes raw pointers; our port takes
/// `(dst, op, src, ip, iend, ilimit_w)` as indices. Caller must
/// guarantee `iend > ilimit_w`.
pub fn ZSTD_safecopyLiterals(
    dst: &mut [u8],
    op: usize,
    src: &[u8],
    ip: usize,
    iend: usize,
    ilimit_w: usize,
) {
    debug_assert!(iend > ilimit_w);
    let mut op_cur = op;
    let mut ip_cur = ip;
    if ip_cur <= ilimit_w {
        let n = ilimit_w - ip_cur;
        dst[op_cur..op_cur + n].copy_from_slice(&src[ip_cur..ip_cur + n]);
        op_cur += n;
        ip_cur = ilimit_w;
    }
    while ip_cur < iend {
        dst[op_cur] = src[ip_cur];
        op_cur += 1;
        ip_cur += 1;
    }
}

/// Port of `ZSTD_copy8` — 8-byte memcpy via one 64-bit load/store.
#[inline]
pub fn ZSTD_copy8(dst: &mut [u8], src: &[u8]) {
    let v = MEM_read64(src);
    MEM_write64(dst, v);
}

/// Port of `ZSTD_copy16` — 16-byte memcpy as two 64-bit load/store pairs.
#[inline]
pub fn ZSTD_copy16(dst: &mut [u8], src: &[u8]) {
    let a = MEM_read64(&src[..8]);
    let b = MEM_read64(&src[8..16]);
    MEM_write64(&mut dst[..8], a);
    MEM_write64(&mut dst[8..16], b);
}

/// Port of `ZSTD_wildcopy`. Copies `length` bytes from `buf[src_ip..]`
/// to `buf[dst_op..]`, potentially reading up to `WILDCOPY_OVERLENGTH`
/// bytes past the logical end of the source region. Handles
/// short-offset self-overlap when `ovtype == ZSTD_overlap_src_before_dst`.
///
/// Rust signature note: upstream takes raw `(dst_ptr, src_ptr, length)`;
/// Rust forbids overlapping `&mut`/`&` into one buffer, so the Rust port
/// takes a single `&mut [u8]` plus two indices, mirroring the
/// `ZSTD_overlapCopy8` signature.
pub fn ZSTD_wildcopy(
    buf: &mut [u8],
    dst_op: usize,
    src_ip: usize,
    length: usize,
    ovtype: ZSTD_overlap_e,
) {
    let diff = dst_op as isize - src_ip as isize;
    let mut op = dst_op;
    let mut ip = src_ip;
    let oend = op + length;

    if ovtype == ZSTD_overlap_e::ZSTD_overlap_src_before_dst && diff < WILDCOPY_VECLEN as isize {
        // Short-offset self-overlap: advance by 8 bytes per step,
        // respecting the overlap invariant (`op - ip >= 8` is what
        // the caller of `ZSTD_wildcopy` promises in this branch).
        while op < oend {
            let v = MEM_read64(&buf[ip..]);
            MEM_write64(&mut buf[op..], v);
            op += 8;
            ip += 8;
        }
    } else {
        debug_assert!(diff >= WILDCOPY_VECLEN as isize || diff <= -(WILDCOPY_VECLEN as isize));
        // Upstream biases the first copy to 16 bytes then runs two
        // 16-byte copies per loop iteration. The wrapping `usize`
        // arithmetic on `ip`/`op` is safe because all indices stay
        // within `buf` by the caller's overlength contract.
        wild_copy16(buf, op, ip);
        if 16 >= length {
            return;
        }
        op += 16;
        ip += 16;
        while op < oend {
            wild_copy16(buf, op, ip);
            op += 16;
            ip += 16;
            wild_copy16(buf, op, ip);
            op += 16;
            ip += 16;
        }
    }
}

/// Helper for `ZSTD_wildcopy`. Takes a single buffer + indices so the
/// single-borrow model works even when dst and src regions are close.
fn wild_copy16(buf: &mut [u8], op: usize, ip: usize) {
    // Read first so we can write without aliasing concerns.
    let a = MEM_read64(&buf[ip..ip + 8]);
    let b = MEM_read64(&buf[ip + 8..ip + 16]);
    MEM_write64(&mut buf[op..op + 8], a);
    MEM_write64(&mut buf[op + 8..op + 16], b);
}

/// Port of `ZSTD_limitCopy`. Copies `min(dstCapacity, srcSize)` bytes
/// and returns the count.
pub fn ZSTD_limitCopy(dst: &mut [u8], src: &[u8]) -> usize {
    let n = dst.len().min(src.len());
    if n > 0 {
        dst[..n].copy_from_slice(&src[..n]);
    }
    n
}

/// Port of `ZSTD_cpuSupportsBmi2` (zstd_internal.h:320). Upstream
/// probes CPUID for BMI1 + BMI2 support. The Rust port uses runtime
/// detection for the same BMI2-gated HUF / sequence encoding paths.
#[inline]
pub fn ZSTD_cpuSupportsBmi2() -> i32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("bmi1") && std::is_x86_feature_detected!("bmi2") {
            1
        } else {
            0
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        0
    }
}

/// Port of `ZSTD_copy4`. Kept here so compress + decompress can share.
#[inline]
pub fn ZSTD_copy4(dst: &mut [u8], src: &[u8]) {
    let v = MEM_read32(src);
    MEM_write32(dst, v);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ZSTD_overlap_e_discriminants_match_upstream() {
        // Upstream: `typedef enum { ZSTD_no_overlap,
        // ZSTD_overlap_src_before_dst } ZSTD_overlap_e;` — default
        // sequential 0/1. `ZSTD_wildcopy` dispatches on this; flipping
        // the two values would silently swap which overlap mode is
        // "safe no-overlap" vs "upstream src before dst".
        assert_eq!(ZSTD_overlap_e::ZSTD_no_overlap as u32, 0);
        assert_eq!(ZSTD_overlap_e::ZSTD_overlap_src_before_dst as u32, 1);
    }

    #[test]
    fn wildcopy_handles_length_exactly_16() {
        // Exactly VECLEN: exercises the early-return path after the
        // first 16-byte copy.
        let mut buf = [0u8; 64];
        for (i, b) in buf.iter_mut().enumerate().take(16) {
            *b = (i + 1) as u8;
        }
        ZSTD_wildcopy(&mut buf, 32, 0, 16, ZSTD_overlap_e::ZSTD_no_overlap);
        for i in 0..16 {
            assert_eq!(buf[32 + i], (i + 1) as u8);
        }
    }

    #[test]
    fn copy4_is_exactly_4_bytes() {
        let src = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x01, 0x02];
        let mut dst = [0u8; 8];
        ZSTD_copy4(&mut dst, &src);
        assert_eq!(&dst[..4], &[0xAA, 0xBB, 0xCC, 0xDD]);
        assert_eq!(&dst[4..], &[0u8, 0, 0, 0]); // bytes 4..8 untouched
    }

    #[test]
    fn selectAddr_returns_candidate_only_when_index_in_range() {
        let candidate = 11u32;
        let backup = 22u32;
        assert_eq!(*ZSTD_selectAddr(9, 9, &candidate, &backup), 11);
        assert_eq!(*ZSTD_selectAddr(8, 9, &candidate, &backup), 22);
    }

    #[test]
    fn repStartValue_matches_upstream() {
        assert_eq!(repStartValue, [1u32, 4, 8]);
    }

    #[test]
    fn invalidateRepCodes_zeros_all_slots() {
        let mut r = [5u32, 9, 17];
        ZSTD_invalidateRepCodes(&mut r);
        assert_eq!(r, [0u32; 3]);
    }

    #[test]
    fn reset_compressedBlockState_rep_matches_default() {
        let mut r = [0u32, 0, 0];
        ZSTD_reset_compressedBlockState_rep(&mut r);
        assert_eq!(r, repStartValue);
    }

    #[test]
    fn wildcopy_no_overlap_copies_exactly() {
        let mut buf = [0u8; 64];
        // src region [0..16] = 0..15; dst region [32..48] must match after copy.
        for (i, slot) in buf.iter_mut().enumerate().take(16) {
            *slot = i as u8;
        }
        ZSTD_wildcopy(&mut buf, 32, 0, 16, ZSTD_overlap_e::ZSTD_no_overlap);
        for i in 0..16 {
            assert_eq!(buf[32 + i], i as u8);
        }
    }

    #[test]
    fn wildcopy_no_overlap_longer_than_16() {
        let mut buf = [0u8; 128];
        for (i, slot) in buf.iter_mut().enumerate().take(48) {
            *slot = i as u8;
        }
        ZSTD_wildcopy(&mut buf, 64, 0, 48, ZSTD_overlap_e::ZSTD_no_overlap);
        for i in 0..48 {
            assert_eq!(buf[64 + i], i as u8, "byte {i} mismatch");
        }
    }

    #[test]
    fn wildcopy_src_before_dst_small_offset_spreads() {
        // Short-offset repeat: seed buf[0..8] with 'A'..'H'; invoke
        // wildcopy with op=8, ip=0, length=24, src_before_dst. After
        // copy we expect buf[0..32] = ABCDEFGH repeated 4×.
        let mut buf = [0u8; 64];
        for (i, slot) in buf.iter_mut().enumerate().take(8) {
            *slot = b'A' + i as u8;
        }
        ZSTD_wildcopy(
            &mut buf,
            8,
            0,
            24,
            ZSTD_overlap_e::ZSTD_overlap_src_before_dst,
        );
        #[allow(clippy::needless_range_loop)]
        for i in 0..32 {
            assert_eq!(buf[i], b'A' + (i % 8) as u8, "pos {i}");
        }
    }

    #[test]
    fn copy4_copy8_copy16_sanity() {
        let src = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut dst = [0u8; 16];
        ZSTD_copy4(&mut dst, &src);
        assert_eq!(&dst[..4], &src[..4]);
        ZSTD_copy8(&mut dst, &src);
        assert_eq!(&dst[..8], &src[..8]);
        ZSTD_copy16(&mut dst, &src);
        assert_eq!(&dst[..16], &src[..16]);
    }

    #[test]
    fn limit_copy_caps_at_min() {
        let src = [1u8, 2, 3, 4, 5];
        let mut dst = [0u8; 8];
        let n = ZSTD_limitCopy(&mut dst, &src);
        assert_eq!(n, 5);
        assert_eq!(&dst[..5], &src);

        let mut dst2 = [0u8; 3];
        let n = ZSTD_limitCopy(&mut dst2, &src);
        assert_eq!(n, 3);
        assert_eq!(&dst2, &[1, 2, 3]);
    }
}
