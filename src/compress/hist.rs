//! Translation of `lib/compress/hist.c` — byte-histogram helpers used
//! by the entropy stage. No SIMD / SVE2 variant is ported; upstream's
//! portable "parallel stripe" loop is the reference implementation.

use crate::common::error::{ErrorCode, ERROR};
use crate::common::mem::MEM_read32;

/// Upstream threshold below which we use the simple single-counter
/// loop; above it, the 4-way parallel counters amortize cache-miss
/// cost per byte. Default path (no ARM SVE2).
pub const HIST_FAST_THRESHOLD: usize = 1500;

pub const HIST_WKSP_SIZE_U32: usize = 1024;
pub const HIST_WKSP_SIZE: usize = HIST_WKSP_SIZE_U32 * 4; // sizeof(unsigned)

/// Mirror of upstream's internal `HIST_checkInput_e`. In `trustInput`
/// the caller guarantees every byte is ≤ `*maxSymbolValuePtr`; in
/// `checkMaxSymbolValue` we enforce it and return an error if
/// violated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HIST_checkInput_e {
    TrustInput,
    CheckMaxSymbolValue,
}

/// Port of `HIST_isError`.
#[inline]
pub fn HIST_isError(code: usize) -> u32 {
    crate::common::error::ERR_isError(code) as u32
}

/// Port of `HIST_add`. Adds 1 to `count[b]` for every byte `b` in `src`.
pub fn HIST_add(count: &mut [u32], src: &[u8]) {
    for &b in src {
        count[b as usize] += 1;
    }
}

/// Port of `HIST_count_simple`. Returns the largest single-symbol
/// count. Sets `*maxSymbolValuePtr` to the highest non-zero symbol.
pub fn HIST_count_simple(count: &mut [u32], maxSymbolValuePtr: &mut u32, src: &[u8]) -> u32 {
    let maxSymbolValue = *maxSymbolValuePtr as usize;
    // Zero the prefix.
    for c in count.iter_mut().take(maxSymbolValue + 1) {
        *c = 0;
    }
    if src.is_empty() {
        *maxSymbolValuePtr = 0;
        return 0;
    }
    for &b in src {
        debug_assert!(b as u32 <= maxSymbolValue as u32);
        count[b as usize] += 1;
    }
    // Shrink maxSymbolValue to last non-zero slot.
    let mut msv = maxSymbolValue;
    while count[msv] == 0 && msv > 0 {
        msv -= 1;
    }
    *maxSymbolValuePtr = msv as u32;
    let mut largest = 0u32;
    for &c in count.iter().take(msv + 1) {
        if c > largest {
            largest = c;
        }
    }
    largest
}

/// Port of `HIST_count_parallel_wksp`. Four counter arrays (one per
/// byte-lane in a u32 stripe) are updated in parallel to amortize the
/// store-queue; then accumulated into `count`.
fn HIST_count_parallel_wksp(
    count: &mut [u32],
    maxSymbolValuePtr: &mut u32,
    src: &[u8],
    check: HIST_checkInput_e,
    workSpace: &mut [u32],
) -> usize {
    debug_assert!(*maxSymbolValuePtr <= 255);
    if src.is_empty() {
        for c in count.iter_mut().take(*maxSymbolValuePtr as usize + 1) {
            *c = 0;
        }
        *maxSymbolValuePtr = 0;
        return 0;
    }

    // The upstream workspace is partitioned into four 256-word counter
    // arrays. We slice likewise.
    for w in workSpace.iter_mut().take(4 * 256) {
        *w = 0;
    }
    let (c1, rest) = workSpace.split_at_mut(256);
    let (c2, rest) = rest.split_at_mut(256);
    let (c3, rest) = rest.split_at_mut(256);
    let (c4, _) = rest.split_at_mut(256);

    let iend = src.len();
    if iend < 4 {
        // Can't even prime the pipeline. Fall back to the scalar tail.
        for &b in src {
            c1[b as usize] += 1;
        }
    } else {
        // Pipeline the first u32 like upstream's `cached` variable.
        let mut ip = 4usize;
        let mut cached = MEM_read32(&src[0..4]);
        // Process in stripes of 16 bytes (4 × u32 loads).
        while ip + 15 < iend {
            let c = cached;
            cached = MEM_read32(&src[ip..ip + 4]);
            ip += 4;
            c1[(c & 0xFF) as usize] += 1;
            c2[((c >> 8) & 0xFF) as usize] += 1;
            c3[((c >> 16) & 0xFF) as usize] += 1;
            c4[(c >> 24) as usize] += 1;
            let c = cached;
            cached = MEM_read32(&src[ip..ip + 4]);
            ip += 4;
            c1[(c & 0xFF) as usize] += 1;
            c2[((c >> 8) & 0xFF) as usize] += 1;
            c3[((c >> 16) & 0xFF) as usize] += 1;
            c4[(c >> 24) as usize] += 1;
            let c = cached;
            cached = MEM_read32(&src[ip..ip + 4]);
            ip += 4;
            c1[(c & 0xFF) as usize] += 1;
            c2[((c >> 8) & 0xFF) as usize] += 1;
            c3[((c >> 16) & 0xFF) as usize] += 1;
            c4[(c >> 24) as usize] += 1;
            let c = cached;
            cached = MEM_read32(&src[ip..ip + 4]);
            ip += 4;
            c1[(c & 0xFF) as usize] += 1;
            c2[((c >> 8) & 0xFF) as usize] += 1;
            c3[((c >> 16) & 0xFF) as usize] += 1;
            c4[(c >> 24) as usize] += 1;
        }
        // Roll back the "cached" prefetch: its 4 bytes weren't yet
        // counted.
        ip -= 4;
        // Tail: remaining bytes counted scalar.
        while ip < iend {
            c1[src[ip] as usize] += 1;
            ip += 1;
        }
    }

    // Sum the four lanes into c1.
    let mut max: u32 = 0;
    for s in 0..256 {
        c1[s] += c2[s] + c3[s] + c4[s];
        if c1[s] > max {
            max = c1[s];
        }
    }

    let mut maxSymbolValue = 255usize;
    while c1[maxSymbolValue] == 0 && maxSymbolValue > 0 {
        maxSymbolValue -= 1;
    }
    if check == HIST_checkInput_e::CheckMaxSymbolValue
        && maxSymbolValue > *maxSymbolValuePtr as usize
    {
        return ERROR(ErrorCode::MaxSymbolValueTooSmall);
    }
    *maxSymbolValuePtr = maxSymbolValue as u32;
    // Copy lane-1 counts into caller's `count` array (memmove because
    // the slices may alias in upstream; in Rust they can't here).
    count[..=maxSymbolValue].copy_from_slice(&c1[..=maxSymbolValue]);
    max as usize
}

/// Port of `HIST_countFast_wksp`. Falls back to the simple loop below
/// the heuristic threshold; dispatches to the parallel version above.
pub fn HIST_countFast_wksp(
    count: &mut [u32],
    maxSymbolValuePtr: &mut u32,
    src: &[u8],
    workSpace: &mut [u32],
) -> usize {
    if src.len() < HIST_FAST_THRESHOLD {
        return HIST_count_simple(count, maxSymbolValuePtr, src) as usize;
    }
    if workSpace.len() < HIST_WKSP_SIZE_U32 {
        return ERROR(ErrorCode::WorkSpaceTooSmall);
    }
    HIST_count_parallel_wksp(
        count,
        maxSymbolValuePtr,
        src,
        HIST_checkInput_e::TrustInput,
        workSpace,
    )
}

/// Port of `HIST_count_wksp`. Same as `HIST_countFast_wksp` but checks
/// that every symbol fits within the caller-declared `*maxSymbolValuePtr`
/// — used when the caller doesn't control the alphabet.
pub fn HIST_count_wksp(
    count: &mut [u32],
    maxSymbolValuePtr: &mut u32,
    src: &[u8],
    workSpace: &mut [u32],
) -> usize {
    if workSpace.len() < HIST_WKSP_SIZE_U32 {
        return ERROR(ErrorCode::WorkSpaceTooSmall);
    }
    if *maxSymbolValuePtr < 255 {
        return HIST_count_parallel_wksp(
            count,
            maxSymbolValuePtr,
            src,
            HIST_checkInput_e::CheckMaxSymbolValue,
            workSpace,
        );
    }
    *maxSymbolValuePtr = 255;
    HIST_countFast_wksp(count, maxSymbolValuePtr, src, workSpace)
}

/// Port of `HIST_countFast` — the stack-allocated-workspace
/// convenience wrapper.
pub fn HIST_countFast(count: &mut [u32], maxSymbolValuePtr: &mut u32, src: &[u8]) -> usize {
    let mut tmp = [0u32; HIST_WKSP_SIZE_U32];
    HIST_countFast_wksp(count, maxSymbolValuePtr, src, &mut tmp)
}

/// Port of `HIST_count` — the stack-allocated-workspace convenience
/// wrapper.
pub fn HIST_count(count: &mut [u32], maxSymbolValuePtr: &mut u32, src: &[u8]) -> usize {
    let mut tmp = [0u32; HIST_WKSP_SIZE_U32];
    HIST_count_wksp(count, maxSymbolValuePtr, src, &mut tmp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_empty_input_zeros_max() {
        let mut count = [0u32; 256];
        let mut msv: u32 = 255;
        let largest = HIST_count_simple(&mut count, &mut msv, &[]);
        assert_eq!(largest, 0);
        assert_eq!(msv, 0);
    }

    #[test]
    fn simple_single_symbol() {
        let mut count = [0u32; 256];
        let mut msv: u32 = 255;
        let largest = HIST_count_simple(&mut count, &mut msv, &[b'a'; 10]);
        assert_eq!(largest, 10);
        assert_eq!(msv, b'a' as u32);
        assert_eq!(count[b'a' as usize], 10);
    }

    #[test]
    fn simple_mixed() {
        let mut count = [0u32; 256];
        let mut msv: u32 = 255;
        let src = b"the quick brown fox";
        let largest = HIST_count_simple(&mut count, &mut msv, src);
        // "the quick brown fox": most-common char is 'o' (2), 'e'(1), 'r'(1)...
        // Actually count: t=1, h=1, e=1, ' '=3, q=1, u=1, i=1, c=1, k=1, b=1, r=1, o=2, w=1, n=1, f=1, x=1
        // Largest = 3 (space).
        assert_eq!(largest, 3);
        assert_eq!(count[b' ' as usize], 3);
        assert_eq!(count[b'o' as usize], 2);
    }

    #[test]
    fn add_accumulates() {
        let mut count = [0u32; 256];
        HIST_add(&mut count, b"aaa");
        HIST_add(&mut count, b"ab");
        assert_eq!(count[b'a' as usize], 4);
        assert_eq!(count[b'b' as usize], 1);
    }

    #[test]
    fn count_fast_matches_simple_on_small_input() {
        // Below the fast threshold, HIST_countFast falls through to
        // HIST_count_simple, so the two paths must produce identical
        // results.
        let src = b"Hello, world! Hello again.";
        let mut c1 = [0u32; 256];
        let mut c2 = [0u32; 256];
        let mut m1: u32 = 255;
        let mut m2: u32 = 255;
        let l1 = HIST_count_simple(&mut c1, &mut m1, src);
        let l2 = HIST_countFast(&mut c2, &mut m2, src) as u32;
        assert_eq!(l1, l2);
        assert_eq!(m1, m2);
        assert_eq!(&c1[..=m1 as usize], &c2[..=m2 as usize]);
    }

    #[test]
    fn count_fast_matches_simple_on_large_input() {
        // Above threshold: the parallel path kicks in. Both must still
        // agree.
        let mut src: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        src.extend_from_slice(b"extra tail");
        let mut c1 = [0u32; 256];
        let mut c2 = [0u32; 256];
        let mut m1: u32 = 255;
        let mut m2: u32 = 255;
        let l1 = HIST_count_simple(&mut c1, &mut m1, &src);
        let l2 = HIST_countFast(&mut c2, &mut m2, &src) as u32;
        assert_eq!(l1, l2);
        assert_eq!(m1, m2);
        assert_eq!(c1, c2);
    }

    #[test]
    fn count_checks_max_symbol_value() {
        // HIST_count with maxSymbol < 255 must reject inputs containing
        // bytes above that cap.
        let src = [0u8, 1, 2, 200];
        let mut count = [0u32; 256];
        let mut msv: u32 = 10;
        let rc = HIST_count(&mut count, &mut msv, &src);
        assert!(crate::common::error::ERR_isError(rc));
    }

    #[test]
    fn count_wksp_rejects_too_small_workspace() {
        // HIST_count_wksp needs at least HIST_WKSP_SIZE_U32 u32 slots;
        // pass a smaller slice and expect WorkSpaceTooSmall.
        let src = b"hello";
        let mut count = [0u32; 256];
        let mut msv: u32 = 255;
        let mut tiny_ws = [0u32; 8]; // way less than HIST_WKSP_SIZE_U32
        let rc = HIST_count_wksp(&mut count, &mut msv, src, &mut tiny_ws);
        use crate::common::error::{ERR_getErrorCode, ErrorCode};
        assert!(crate::common::error::ERR_isError(rc));
        assert_eq!(ERR_getErrorCode(rc), ErrorCode::WorkSpaceTooSmall);
    }

    #[test]
    fn hist_isError_matches_underlying_err_check() {
        // HIST_isError is a thin cast over ERR_isError; confirm it
        // returns 1 for error codes and 0 for successful sizes.
        use crate::common::error::{ERROR, ErrorCode};
        let err = ERROR(ErrorCode::Generic);
        assert_eq!(HIST_isError(err), 1);
        assert_eq!(HIST_isError(0), 0);
        assert_eq!(HIST_isError(1024), 0);
    }
}
