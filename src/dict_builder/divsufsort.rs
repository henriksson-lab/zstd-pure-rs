//! Translation of `lib/dictBuilder/divsufsort.c` (libdivsufsort-lite,
//! Yuta Mori, BSD/MIT). Constructs the suffix array of a byte string.
//!
//! The C library is written entirely in terms of `int *` pointer
//! arithmetic over a single output array `SA` (plus the input `T` and the
//! bucket arrays). This port mirrors that 1:1 with raw pointers, so the
//! intricate pointer comparisons / differences / in-place rewrites
//! translate verbatim; the `unsafe` is contained behind the safe
//! [`divsufsort`] entry point.
//!
//! Compiled for the only configuration zstd uses: `ALPHABET_SIZE == 256`
//! and `SS_BLOCKSIZE == 1024`. The C preprocessor branches for other
//! configurations are not reproduced.

#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(clippy::many_single_char_names)]
// TODO(divsufsort): remove once the full file is translated and wired up.
#![allow(dead_code)]

// ---- Constants (ALPHABET_SIZE == 256, SS_BLOCKSIZE == 1024) ----
const ALPHABET_SIZE: i32 = 256;
const BUCKET_A_SIZE: usize = ALPHABET_SIZE as usize;
const BUCKET_B_SIZE: usize = (ALPHABET_SIZE * ALPHABET_SIZE) as usize;
const SS_INSERTIONSORT_THRESHOLD: i32 = 8;
const SS_BLOCKSIZE: i32 = 1024;
const SS_MISORT_STACKSIZE: usize = 16;
const SS_SMERGE_STACKSIZE: usize = 32;
const TR_INSERTIONSORT_THRESHOLD: i32 = 8;
const TR_STACKSIZE: usize = 64;

// `BUCKET_A(c0)` -> bucket_A[c0]
// `BUCKET_B(c0, c1)` -> bucket_B[(c1 << 8) | c0]
// `BUCKET_BSTAR(c0, c1)` -> bucket_B[(c0 << 8) | c1]
#[inline(always)]
fn bucket_b_idx(c0: i32, c1: i32) -> usize {
    (((c1) << 8) | (c0)) as usize
}
#[inline(always)]
fn bucket_bstar_idx(c0: i32, c1: i32) -> usize {
    (((c0) << 8) | (c1)) as usize
}

#[rustfmt::skip]
static lg_table: [i32; 256] = [
 -1,0,1,1,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
  6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
  6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
];

#[rustfmt::skip]
static sqq_table: [i32; 256] = [
  0,  16,  22,  27,  32,  35,  39,  42,  45,  48,  50,  53,  55,  57,  59,  61,
 64,  65,  67,  69,  71,  73,  75,  76,  78,  80,  81,  83,  84,  86,  87,  89,
 90,  91,  93,  94,  96,  97,  98,  99, 101, 102, 103, 104, 106, 107, 108, 109,
110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
128, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
143, 144, 144, 145, 146, 147, 148, 149, 150, 150, 151, 152, 153, 154, 155, 155,
156, 157, 158, 159, 160, 160, 161, 162, 163, 163, 164, 165, 166, 167, 167, 168,
169, 170, 170, 171, 172, 173, 173, 174, 175, 176, 176, 177, 178, 178, 179, 180,
181, 181, 182, 183, 183, 184, 185, 185, 186, 187, 187, 188, 189, 189, 190, 191,
192, 192, 193, 193, 194, 195, 195, 196, 197, 197, 198, 199, 199, 200, 201, 201,
202, 203, 203, 204, 204, 205, 206, 206, 207, 208, 208, 209, 209, 210, 211, 211,
212, 212, 213, 214, 214, 215, 215, 216, 217, 217, 218, 218, 219, 219, 220, 221,
221, 222, 222, 223, 224, 224, 225, 225, 226, 226, 227, 227, 228, 229, 229, 230,
230, 231, 231, 232, 232, 233, 234, 234, 235, 235, 236, 236, 237, 237, 238, 238,
239, 240, 240, 241, 241, 242, 242, 243, 243, 244, 244, 245, 245, 246, 246, 247,
247, 248, 248, 249, 249, 250, 250, 251, 251, 252, 252, 253, 253, 254, 254, 255,
];

/// Port of `ss_ilg` (SS_BLOCKSIZE == 1024 branch: `>= 256`).
#[inline(always)]
fn ss_ilg(n: i32) -> i32 {
    if (n & 0xff00) != 0 {
        8 + lg_table[((n >> 8) & 0xff) as usize]
    } else {
        lg_table[((n >> 0) & 0xff) as usize]
    }
}

/// Port of `ss_isqrt`.
fn ss_isqrt(x: i32) -> i32 {
    if x >= (SS_BLOCKSIZE * SS_BLOCKSIZE) {
        return SS_BLOCKSIZE;
    }
    let e: i32 = if ((x as u32) & 0xffff0000) != 0 {
        if ((x as u32) & 0xff000000) != 0 {
            24 + lg_table[((x >> 24) & 0xff) as usize]
        } else {
            16 + lg_table[((x >> 16) & 0xff) as usize]
        }
    } else if (x & 0x0000ff00) != 0 {
        8 + lg_table[((x >> 8) & 0xff) as usize]
    } else {
        lg_table[((x >> 0) & 0xff) as usize]
    };

    let mut y: i32;
    if e >= 16 {
        y = sqq_table[(x >> ((e - 6) - (e & 1))) as usize] << ((e >> 1) - 7);
        if e >= 24 {
            y = (y + 1 + x / y) >> 1;
        }
        y = (y + 1 + x / y) >> 1;
    } else if e >= 8 {
        y = (sqq_table[(x >> ((e - 6) - (e & 1))) as usize] >> (7 - (e >> 1))) + 1;
    } else {
        return sqq_table[x as usize] >> 4;
    }

    if x < (y * y) {
        y - 1
    } else {
        y
    }
}

/* ------------------------------------------------------------------------ */

/// Port of `ss_compare`. Compares two suffixes.
#[inline(always)]
unsafe fn ss_compare(T: *const u8, p1: *const i32, p2: *const i32, depth: i32) -> i32 {
    let mut U1 = T.offset((depth + *p1) as isize);
    let mut U2 = T.offset((depth + *p2) as isize);
    let U1n = T.offset((*p1.offset(1) + 2) as isize);
    let U2n = T.offset((*p2.offset(1) + 2) as isize);

    while (U1 < U1n) && (U2 < U2n) && (*U1 == *U2) {
        U1 = U1.offset(1);
        U2 = U2.offset(1);
    }

    if U1 < U1n {
        if U2 < U2n {
            *U1 as i32 - *U2 as i32
        } else {
            1
        }
    } else if U2 < U2n {
        -1
    } else {
        0
    }
}

/* ------------------------------------------------------------------------ */

/// Port of `ss_insertionsort`. Insertion sort for small size groups.
unsafe fn ss_insertionsort(T: *const u8, PA: *const i32, first: *mut i32, last: *mut i32, depth: i32) {
    let mut i: *mut i32;
    let mut j: *mut i32;
    let mut t: i32;
    let mut r: i32;

    i = last.offset(-2);
    while first <= i {
        t = *i;
        j = i.offset(1);
        loop {
            r = ss_compare(T, PA.offset(t as isize), PA.offset(*j as isize), depth);
            if r <= 0 {
                break;
            }
            loop {
                *j.offset(-1) = *j;
                j = j.offset(1);
                if !((j < last) && (*j < 0)) {
                    break;
                }
            }
            if last <= j {
                break;
            }
        }
        if r == 0 {
            *j = !*j;
        }
        *j.offset(-1) = t;
        i = i.offset(-1);
    }
}

/* ------------------------------------------------------------------------ */

/// Port of `ss_fixdown`.
#[inline(always)]
unsafe fn ss_fixdown(Td: *const u8, PA: *const i32, SA: *mut i32, mut i: i32, size: i32) {
    let mut j: i32;
    let mut k: i32;
    let v: i32;
    let mut c: i32;
    let mut d: i32;

    v = *SA.offset(i as isize);
    c = *Td.offset(*PA.offset(v as isize) as isize) as i32;
    loop {
        j = 2 * i + 1;
        if !(j < size) {
            break;
        }
        k = j;
        j += 1;
        d = *Td.offset(*PA.offset(*SA.offset(k as isize) as isize) as isize) as i32;
        let e = *Td.offset(*PA.offset(*SA.offset(j as isize) as isize) as isize) as i32;
        if d < e {
            k = j;
            d = e;
        }
        if d <= c {
            break;
        }
        *SA.offset(i as isize) = *SA.offset(k as isize);
        i = k;
    }
    *SA.offset(i as isize) = v;
}

/// Port of `ss_heapsort`. Simple top-down heapsort.
unsafe fn ss_heapsort(Td: *const u8, PA: *const i32, SA: *mut i32, size: i32) {
    let mut i: i32;
    let mut m: i32;
    let mut t: i32;

    m = size;
    if (size % 2) == 0 {
        m -= 1;
        if (*Td.offset(*PA.offset(*SA.offset((m / 2) as isize) as isize) as isize) as i32)
            < (*Td.offset(*PA.offset(*SA.offset(m as isize) as isize) as isize) as i32)
        {
            t = *SA.offset(m as isize);
            *SA.offset(m as isize) = *SA.offset((m / 2) as isize);
            *SA.offset((m / 2) as isize) = t;
        }
    }

    i = m / 2 - 1;
    while 0 <= i {
        ss_fixdown(Td, PA, SA, i, m);
        i -= 1;
    }
    if (size % 2) == 0 {
        t = *SA.offset(0);
        *SA.offset(0) = *SA.offset(m as isize);
        *SA.offset(m as isize) = t;
        ss_fixdown(Td, PA, SA, 0, m);
    }
    i = m - 1;
    while 0 < i {
        t = *SA.offset(0);
        *SA.offset(0) = *SA.offset(i as isize);
        ss_fixdown(Td, PA, SA, 0, i);
        *SA.offset(i as isize) = t;
        i -= 1;
    }
}

/* ------------------------------------------------------------------------ */

/// Port of `ss_median3`. Returns the median of three elements.
#[inline(always)]
unsafe fn ss_median3(
    Td: *const u8,
    PA: *const i32,
    mut v1: *mut i32,
    mut v2: *mut i32,
    v3: *mut i32,
) -> *mut i32 {
    if (*Td.offset(*PA.offset(*v1 as isize) as isize)) > (*Td.offset(*PA.offset(*v2 as isize) as isize)) {
        core::mem::swap(&mut v1, &mut v2);
    }
    if (*Td.offset(*PA.offset(*v2 as isize) as isize)) > (*Td.offset(*PA.offset(*v3 as isize) as isize)) {
        if (*Td.offset(*PA.offset(*v1 as isize) as isize)) > (*Td.offset(*PA.offset(*v3 as isize) as isize)) {
            return v1;
        } else {
            return v3;
        }
    }
    v2
}

/// Port of `ss_median5`. Returns the median of five elements.
#[inline(always)]
unsafe fn ss_median5(
    Td: *const u8,
    PA: *const i32,
    mut v1: *mut i32,
    mut v2: *mut i32,
    mut v3: *mut i32,
    mut v4: *mut i32,
    mut v5: *mut i32,
) -> *mut i32 {
    if (*Td.offset(*PA.offset(*v2 as isize) as isize)) > (*Td.offset(*PA.offset(*v3 as isize) as isize)) {
        core::mem::swap(&mut v2, &mut v3);
    }
    if (*Td.offset(*PA.offset(*v4 as isize) as isize)) > (*Td.offset(*PA.offset(*v5 as isize) as isize)) {
        core::mem::swap(&mut v4, &mut v5);
    }
    if (*Td.offset(*PA.offset(*v2 as isize) as isize)) > (*Td.offset(*PA.offset(*v4 as isize) as isize)) {
        core::mem::swap(&mut v2, &mut v4);
        core::mem::swap(&mut v3, &mut v5);
    }
    if (*Td.offset(*PA.offset(*v1 as isize) as isize)) > (*Td.offset(*PA.offset(*v3 as isize) as isize)) {
        core::mem::swap(&mut v1, &mut v3);
    }
    if (*Td.offset(*PA.offset(*v1 as isize) as isize)) > (*Td.offset(*PA.offset(*v4 as isize) as isize)) {
        core::mem::swap(&mut v1, &mut v4);
        core::mem::swap(&mut v3, &mut v5);
    }
    if (*Td.offset(*PA.offset(*v3 as isize) as isize)) > (*Td.offset(*PA.offset(*v4 as isize) as isize)) {
        return v4;
    }
    v3
}

/// Port of `ss_pivot`. Returns the pivot element.
#[inline(always)]
unsafe fn ss_pivot(Td: *const u8, PA: *const i32, mut first: *mut i32, mut last: *mut i32) -> *mut i32 {
    let mut middle: *mut i32;
    let mut t: i32;

    t = last.offset_from(first) as i32;
    middle = first.offset((t / 2) as isize);

    if t <= 512 {
        if t <= 32 {
            return ss_median3(Td, PA, first, middle, last.offset(-1));
        } else {
            t >>= 2;
            return ss_median5(
                Td,
                PA,
                first,
                first.offset(t as isize),
                middle,
                last.offset(-1 - t as isize),
                last.offset(-1),
            );
        }
    }
    t >>= 3;
    first = ss_median3(Td, PA, first, first.offset(t as isize), first.offset((t << 1) as isize));
    middle = ss_median3(Td, PA, middle.offset(-(t as isize)), middle, middle.offset(t as isize));
    last = ss_median3(
        Td,
        PA,
        last.offset(-1 - (t << 1) as isize),
        last.offset(-1 - t as isize),
        last.offset(-1),
    );
    ss_median3(Td, PA, first, middle, last)
}

/* ------------------------------------------------------------------------ */

/// Port of `ss_partition`. Binary partition for substrings.
#[inline(always)]
unsafe fn ss_partition(PA: *const i32, first: *mut i32, last: *mut i32, depth: i32) -> *mut i32 {
    let mut a: *mut i32;
    let mut b: *mut i32;
    let mut t: i32;
    a = first.offset(-1);
    b = last;
    loop {
        loop {
            a = a.offset(1);
            if !((a < b) && ((*PA.offset(*a as isize) + depth) >= (*PA.offset((*a + 1) as isize) + 1))) {
                break;
            }
            *a = !*a;
        }
        loop {
            b = b.offset(-1);
            if !((a < b) && ((*PA.offset(*b as isize) + depth) < (*PA.offset((*b + 1) as isize) + 1))) {
                break;
            }
        }
        if b <= a {
            break;
        }
        t = !*b;
        *b = *a;
        *a = t;
    }
    if first < a {
        *first = !*first;
    }
    a
}

/// Port of `ss_mintrosort`. Multikey introsort for medium size groups.
unsafe fn ss_mintrosort(
    T: *const u8,
    PA: *const i32,
    mut first: *mut i32,
    mut last: *mut i32,
    mut depth: i32,
) {
    #[derive(Clone, Copy)]
    struct StackEntry {
        a: *mut i32,
        b: *mut i32,
        c: i32,
        d: i32,
    }
    let mut stack = [StackEntry {
        a: core::ptr::null_mut(),
        b: core::ptr::null_mut(),
        c: 0,
        d: 0,
    }; SS_MISORT_STACKSIZE];
    let mut ssize: usize = 0;

    let mut a: *mut i32;
    let mut b: *mut i32;
    let mut c: *mut i32;
    let mut d: *mut i32;
    let mut e: *mut i32;
    let mut f: *mut i32;
    let mut s: i32;
    let mut t: i32;
    let mut limit: i32;
    let mut v: i32;
    let mut x: i32 = 0;

    // local Td reused inside the loop (declared here to mirror C scoping)
    macro_rules! stack_push {
        ($a:expr, $b:expr, $c:expr, $d:expr) => {{
            stack[ssize] = StackEntry {
                a: $a,
                b: $b,
                c: $c,
                d: $d,
            };
            ssize += 1;
        }};
    }
    macro_rules! stack_pop {
        ($a:ident, $b:ident, $c:ident, $d:ident) => {{
            if ssize == 0 {
                return;
            }
            ssize -= 1;
            $a = stack[ssize].a;
            $b = stack[ssize].b;
            $c = stack[ssize].c;
            $d = stack[ssize].d;
        }};
    }

    limit = ss_ilg(last.offset_from(first) as i32);
    loop {
        if (last.offset_from(first) as i32) <= SS_INSERTIONSORT_THRESHOLD {
            if 1 < last.offset_from(first) as i32 {
                ss_insertionsort(T, PA, first, last, depth);
            }
            stack_pop!(first, last, depth, limit);
            continue;
        }

        let Td = T.offset(depth as isize);
        let prev_limit = limit;
        limit -= 1;
        if prev_limit == 0 {
            ss_heapsort(Td, PA, first, last.offset_from(first) as i32);
        }
        if limit < 0 {
            a = first.offset(1);
            v = *Td.offset(*PA.offset(*first as isize) as isize) as i32;
            while a < last {
                x = *Td.offset(*PA.offset(*a as isize) as isize) as i32;
                if x != v {
                    if 1 < a.offset_from(first) as i32 {
                        break;
                    }
                    v = x;
                    first = a;
                }
                a = a.offset(1);
            }
            if (*Td.offset((*PA.offset(*first as isize) - 1) as isize) as i32) < v {
                first = ss_partition(PA, first, a, depth);
            }
            if (a.offset_from(first) as i32) <= (last.offset_from(a) as i32) {
                if 1 < a.offset_from(first) as i32 {
                    stack_push!(a, last, depth, -1);
                    last = a;
                    depth += 1;
                    limit = ss_ilg(a.offset_from(first) as i32);
                } else {
                    first = a;
                    limit = -1;
                }
            } else if 1 < last.offset_from(a) as i32 {
                stack_push!(first, a, depth + 1, ss_ilg(a.offset_from(first) as i32));
                first = a;
                limit = -1;
            } else {
                last = a;
                depth += 1;
                limit = ss_ilg(a.offset_from(first) as i32);
            }
            continue;
        }

        /* choose pivot */
        a = ss_pivot(Td, PA, first, last);
        v = *Td.offset(*PA.offset(*a as isize) as isize) as i32;
        core::ptr::swap(first, a);

        /* partition */
        b = first;
        loop {
            b = b.offset(1);
            if !(b < last) {
                break;
            }
            x = *Td.offset(*PA.offset(*b as isize) as isize) as i32;
            if x != v {
                break;
            }
        }
        a = b;
        if (a < last) && (x < v) {
            loop {
                b = b.offset(1);
                if !(b < last) {
                    break;
                }
                x = *Td.offset(*PA.offset(*b as isize) as isize) as i32;
                if x > v {
                    break;
                }
                if x == v {
                    core::ptr::swap(b, a);
                    a = a.offset(1);
                }
            }
        }
        c = last;
        loop {
            c = c.offset(-1);
            if !(b < c) {
                break;
            }
            x = *Td.offset(*PA.offset(*c as isize) as isize) as i32;
            if x != v {
                break;
            }
        }
        d = c;
        if (b < d) && (x > v) {
            loop {
                c = c.offset(-1);
                if !(b < c) {
                    break;
                }
                x = *Td.offset(*PA.offset(*c as isize) as isize) as i32;
                if x < v {
                    break;
                }
                if x == v {
                    core::ptr::swap(c, d);
                    d = d.offset(-1);
                }
            }
        }
        while b < c {
            core::ptr::swap(b, c);
            loop {
                b = b.offset(1);
                if !(b < c) {
                    break;
                }
                x = *Td.offset(*PA.offset(*b as isize) as isize) as i32;
                if x > v {
                    break;
                }
                if x == v {
                    core::ptr::swap(b, a);
                    a = a.offset(1);
                }
            }
            loop {
                c = c.offset(-1);
                if !(b < c) {
                    break;
                }
                x = *Td.offset(*PA.offset(*c as isize) as isize) as i32;
                if x < v {
                    break;
                }
                if x == v {
                    core::ptr::swap(c, d);
                    d = d.offset(-1);
                }
            }
        }

        if a <= d {
            c = b.offset(-1);

            s = a.offset_from(first) as i32;
            t = b.offset_from(a) as i32;
            if s > t {
                s = t;
            }
            e = first;
            f = b.offset(-(s as isize));
            while 0 < s {
                core::ptr::swap(e, f);
                s -= 1;
                e = e.offset(1);
                f = f.offset(1);
            }
            s = d.offset_from(c) as i32;
            t = last.offset_from(d).wrapping_sub(1) as i32;
            if s > t {
                s = t;
            }
            e = b;
            f = last.offset(-(s as isize));
            while 0 < s {
                core::ptr::swap(e, f);
                s -= 1;
                e = e.offset(1);
                f = f.offset(1);
            }

            a = first.offset(b.offset_from(a));
            c = last.offset(-(d.offset_from(c)));
            b = if v <= (*Td.offset((*PA.offset(*a as isize) - 1) as isize) as i32) {
                a
            } else {
                ss_partition(PA, a, c, depth)
            };

            if (a.offset_from(first) as i32) <= (last.offset_from(c) as i32) {
                if (last.offset_from(c) as i32) <= (c.offset_from(b) as i32) {
                    stack_push!(b, c, depth + 1, ss_ilg(c.offset_from(b) as i32));
                    stack_push!(c, last, depth, limit);
                    last = a;
                } else if (a.offset_from(first) as i32) <= (c.offset_from(b) as i32) {
                    stack_push!(c, last, depth, limit);
                    stack_push!(b, c, depth + 1, ss_ilg(c.offset_from(b) as i32));
                    last = a;
                } else {
                    stack_push!(c, last, depth, limit);
                    stack_push!(first, a, depth, limit);
                    first = b;
                    last = c;
                    depth += 1;
                    limit = ss_ilg(c.offset_from(b) as i32);
                }
            } else if (a.offset_from(first) as i32) <= (c.offset_from(b) as i32) {
                stack_push!(b, c, depth + 1, ss_ilg(c.offset_from(b) as i32));
                stack_push!(first, a, depth, limit);
                first = c;
            } else if (last.offset_from(c) as i32) <= (c.offset_from(b) as i32) {
                stack_push!(first, a, depth, limit);
                stack_push!(b, c, depth + 1, ss_ilg(c.offset_from(b) as i32));
                first = c;
            } else {
                stack_push!(first, a, depth, limit);
                stack_push!(c, last, depth, limit);
                first = b;
                last = c;
                depth += 1;
                limit = ss_ilg(c.offset_from(b) as i32);
            }
        } else {
            limit += 1;
            if (*Td.offset((*PA.offset(*first as isize) - 1) as isize) as i32) < v {
                first = ss_partition(PA, first, last, depth);
                limit = ss_ilg(last.offset_from(first) as i32);
            }
            depth += 1;
        }
    }
}

/* ------------------------------------------------------------------------ */

/// Port of `ss_blockswap`.
#[inline(always)]
unsafe fn ss_blockswap(mut a: *mut i32, mut b: *mut i32, mut n: i32) {
    let mut t: i32;
    while 0 < n {
        t = *a;
        *a = *b;
        *b = t;
        n -= 1;
        a = a.offset(1);
        b = b.offset(1);
    }
}

/// Port of `ss_rotate`.
#[inline(always)]
unsafe fn ss_rotate(mut first: *mut i32, middle: *mut i32, mut last: *mut i32) {
    let mut a: *mut i32;
    let mut b: *mut i32;
    let mut t: i32;
    let mut l: i32;
    let mut r: i32;
    l = middle.offset_from(first) as i32;
    r = last.offset_from(middle) as i32;
    while (0 < l) && (0 < r) {
        if l == r {
            ss_blockswap(first, middle, l);
            break;
        }
        if l < r {
            a = last.offset(-1);
            b = middle.offset(-1);
            t = *a;
            loop {
                // *a-- = *b, *b-- = *a;
                *a = *b;
                a = a.offset(-1);
                *b = *a;
                b = b.offset(-1);
                if b < first {
                    *a = t;
                    last = a;
                    r -= l + 1;
                    if r <= l {
                        break;
                    }
                    a = a.offset(-1);
                    b = middle.offset(-1);
                    t = *a;
                }
            }
        } else {
            a = first;
            b = middle;
            t = *a;
            loop {
                // *a++ = *b, *b++ = *a;
                *a = *b;
                a = a.offset(1);
                *b = *a;
                b = b.offset(1);
                if last <= b {
                    *a = t;
                    first = a.offset(1);
                    l -= r + 1;
                    if l <= r {
                        break;
                    }
                    a = a.offset(1);
                    b = middle;
                    t = *a;
                }
            }
        }
    }
}

/* ------------------------------------------------------------------------ */

/// Port of `ss_inplacemerge`.
unsafe fn ss_inplacemerge(
    T: *const u8,
    PA: *const i32,
    mut first: *mut i32,
    mut middle: *mut i32,
    mut last: *mut i32,
    depth: i32,
) {
    let mut p: *const i32;
    let mut a: *mut i32;
    let mut b: *mut i32;
    let mut len: i32;
    let mut half: i32;
    let mut q: i32;
    let mut r: i32;
    let mut x: i32;

    loop {
        if *last.offset(-1) < 0 {
            x = 1;
            p = PA.offset((!*last.offset(-1)) as isize);
        } else {
            x = 0;
            p = PA.offset((*last.offset(-1)) as isize);
        }
        a = first;
        len = middle.offset_from(first) as i32;
        half = len >> 1;
        r = -1;
        while 0 < len {
            b = a.offset(half as isize);
            q = ss_compare(
                T,
                PA.offset(if 0 <= *b { *b } else { !*b } as isize),
                p,
                depth,
            );
            if q < 0 {
                a = b.offset(1);
                half -= (len & 1) ^ 1;
            } else {
                r = q;
            }
            len = half;
            half >>= 1;
        }
        if a < middle {
            if r == 0 {
                *a = !*a;
            }
            ss_rotate(a, middle, last);
            last = last.offset(-(middle.offset_from(a)));
            middle = a;
            if first == middle {
                break;
            }
        }
        last = last.offset(-1);
        if x != 0 {
            loop {
                last = last.offset(-1);
                if !(*last < 0) {
                    break;
                }
            }
        }
        if middle == last {
            break;
        }
    }
}

/* ------------------------------------------------------------------------ */

/// Port of `ss_mergeforward`. Merge-forward with internal buffer.
unsafe fn ss_mergeforward(
    T: *const u8,
    PA: *const i32,
    first: *mut i32,
    middle: *mut i32,
    last: *mut i32,
    buf: *mut i32,
    depth: i32,
) {
    let mut a: *mut i32;
    let mut b: *mut i32;
    let mut c: *mut i32;
    let bufend: *mut i32;
    let t: i32;
    let mut r: i32;

    bufend = buf.offset(middle.offset_from(first)).offset(-1);
    ss_blockswap(buf, first, middle.offset_from(first) as i32);

    a = first;
    t = *a;
    b = buf;
    c = middle;
    loop {
        r = ss_compare(T, PA.offset(*b as isize), PA.offset(*c as isize), depth);
        if r < 0 {
            loop {
                *a = *b;
                a = a.offset(1);
                if bufend <= b {
                    *bufend = t;
                    return;
                }
                *b = *a;
                b = b.offset(1);
                if !(*b < 0) {
                    break;
                }
            }
        } else if r > 0 {
            loop {
                *a = *c;
                a = a.offset(1);
                *c = *a;
                c = c.offset(1);
                if last <= c {
                    while b < bufend {
                        *a = *b;
                        a = a.offset(1);
                        *b = *a;
                        b = b.offset(1);
                    }
                    *a = *b;
                    *b = t;
                    return;
                }
                if !(*c < 0) {
                    break;
                }
            }
        } else {
            *c = !*c;
            loop {
                *a = *b;
                a = a.offset(1);
                if bufend <= b {
                    *bufend = t;
                    return;
                }
                *b = *a;
                b = b.offset(1);
                if !(*b < 0) {
                    break;
                }
            }
            loop {
                *a = *c;
                a = a.offset(1);
                *c = *a;
                c = c.offset(1);
                if last <= c {
                    while b < bufend {
                        *a = *b;
                        a = a.offset(1);
                        *b = *a;
                        b = b.offset(1);
                    }
                    *a = *b;
                    *b = t;
                    return;
                }
                if !(*c < 0) {
                    break;
                }
            }
        }
    }
}

/// Port of `ss_mergebackward`. Merge-backward with internal buffer.
unsafe fn ss_mergebackward(
    T: *const u8,
    PA: *const i32,
    first: *mut i32,
    middle: *mut i32,
    last: *mut i32,
    buf: *mut i32,
    depth: i32,
) {
    let mut p1: *const i32;
    let mut p2: *const i32;
    let mut a: *mut i32;
    let mut b: *mut i32;
    let mut c: *mut i32;
    let bufend: *mut i32;
    let t: i32;
    let mut r: i32;
    let mut x: i32;

    bufend = buf.offset(last.offset_from(middle)).offset(-1);
    ss_blockswap(buf, middle, last.offset_from(middle) as i32);

    x = 0;
    if *bufend < 0 {
        p1 = PA.offset((!*bufend) as isize);
        x |= 1;
    } else {
        p1 = PA.offset((*bufend) as isize);
    }
    if *middle.offset(-1) < 0 {
        p2 = PA.offset((!*middle.offset(-1)) as isize);
        x |= 2;
    } else {
        p2 = PA.offset((*middle.offset(-1)) as isize);
    }
    a = last.offset(-1);
    t = *a;
    b = bufend;
    c = middle.offset(-1);
    loop {
        r = ss_compare(T, p1, p2, depth);
        if 0 < r {
            if (x & 1) != 0 {
                loop {
                    *a = *b;
                    a = a.offset(-1);
                    *b = *a;
                    b = b.offset(-1);
                    if !(*b < 0) {
                        break;
                    }
                }
                x ^= 1;
            }
            *a = *b;
            a = a.offset(-1);
            if b <= buf {
                *buf = t;
                break;
            }
            *b = *a;
            b = b.offset(-1);
            if *b < 0 {
                p1 = PA.offset((!*b) as isize);
                x |= 1;
            } else {
                p1 = PA.offset((*b) as isize);
            }
        } else if r < 0 {
            if (x & 2) != 0 {
                loop {
                    *a = *c;
                    a = a.offset(-1);
                    *c = *a;
                    c = c.offset(-1);
                    if !(*c < 0) {
                        break;
                    }
                }
                x ^= 2;
            }
            *a = *c;
            a = a.offset(-1);
            *c = *a;
            c = c.offset(-1);
            if c < first {
                while buf < b {
                    *a = *b;
                    a = a.offset(-1);
                    *b = *a;
                    b = b.offset(-1);
                }
                *a = *b;
                *b = t;
                break;
            }
            if *c < 0 {
                p2 = PA.offset((!*c) as isize);
                x |= 2;
            } else {
                p2 = PA.offset((*c) as isize);
            }
        } else {
            if (x & 1) != 0 {
                loop {
                    *a = *b;
                    a = a.offset(-1);
                    *b = *a;
                    b = b.offset(-1);
                    if !(*b < 0) {
                        break;
                    }
                }
                x ^= 1;
            }
            *a = !*b;
            a = a.offset(-1);
            if b <= buf {
                *buf = t;
                break;
            }
            *b = *a;
            b = b.offset(-1);
            if (x & 2) != 0 {
                loop {
                    *a = *c;
                    a = a.offset(-1);
                    *c = *a;
                    c = c.offset(-1);
                    if !(*c < 0) {
                        break;
                    }
                }
                x ^= 2;
            }
            *a = *c;
            a = a.offset(-1);
            *c = *a;
            c = c.offset(-1);
            if c < first {
                while buf < b {
                    *a = *b;
                    a = a.offset(-1);
                    *b = *a;
                    b = b.offset(-1);
                }
                *a = *b;
                *b = t;
                break;
            }
            if *b < 0 {
                p1 = PA.offset((!*b) as isize);
                x |= 1;
            } else {
                p1 = PA.offset((*b) as isize);
            }
            if *c < 0 {
                p2 = PA.offset((!*c) as isize);
                x |= 2;
            } else {
                p2 = PA.offset((*c) as isize);
            }
        }
    }
}

#[inline(always)]
fn getidx(a: i32) -> i32 {
    if 0 <= a {
        a
    } else {
        !a
    }
}

/// Port of `ss_swapmerge`. D&C based merge.
unsafe fn ss_swapmerge(
    T: *const u8,
    PA: *const i32,
    mut first: *mut i32,
    mut middle: *mut i32,
    mut last: *mut i32,
    buf: *mut i32,
    bufsize: i32,
    depth: i32,
) {
    #[derive(Clone, Copy)]
    struct StackEntry {
        a: *mut i32,
        b: *mut i32,
        c: *mut i32,
        d: i32,
    }
    let mut stack = [StackEntry {
        a: core::ptr::null_mut(),
        b: core::ptr::null_mut(),
        c: core::ptr::null_mut(),
        d: 0,
    }; SS_SMERGE_STACKSIZE];
    let mut ssize: usize = 0;

    let mut l: *mut i32;
    let mut r: *mut i32;
    let mut lm: *mut i32;
    let mut rm: *mut i32;
    let mut m: i32;
    let mut len: i32;
    let mut half: i32;
    let mut check: i32;
    let mut next: i32;

    // MERGE_CHECK(a, b, c)
    macro_rules! merge_check {
        ($a:expr, $b:expr, $c:expr) => {{
            let aa: *mut i32 = $a;
            let bb: *mut i32 = $b;
            let cc: i32 = $c;
            if ((cc & 1) != 0)
                || (((cc & 2) != 0)
                    && (ss_compare(
                        T,
                        PA.offset(getidx(*aa.offset(-1)) as isize),
                        PA.offset(*aa as isize),
                        depth,
                    ) == 0))
            {
                *aa = !*aa;
            }
            if ((cc & 4) != 0)
                && (ss_compare(
                    T,
                    PA.offset(getidx(*bb.offset(-1)) as isize),
                    PA.offset(*bb as isize),
                    depth,
                ) == 0)
            {
                *bb = !*bb;
            }
        }};
    }
    macro_rules! stack_push {
        ($a:expr, $b:expr, $c:expr, $d:expr) => {{
            stack[ssize] = StackEntry {
                a: $a,
                b: $b,
                c: $c,
                d: $d,
            };
            ssize += 1;
        }};
    }
    macro_rules! stack_pop {
        ($a:ident, $b:ident, $c:ident, $d:ident) => {{
            if ssize == 0 {
                return;
            }
            ssize -= 1;
            $a = stack[ssize].a;
            $b = stack[ssize].b;
            $c = stack[ssize].c;
            $d = stack[ssize].d;
        }};
    }

    check = 0;
    loop {
        if (last.offset_from(middle) as i32) <= bufsize {
            if (first < middle) && (middle < last) {
                ss_mergebackward(T, PA, first, middle, last, buf, depth);
            }
            merge_check!(first, last, check);
            stack_pop!(first, middle, last, check);
            continue;
        }

        if (middle.offset_from(first) as i32) <= bufsize {
            if first < middle {
                ss_mergeforward(T, PA, first, middle, last, buf, depth);
            }
            merge_check!(first, last, check);
            stack_pop!(first, middle, last, check);
            continue;
        }

        m = 0;
        len = core::cmp::min(
            middle.offset_from(first) as i32,
            last.offset_from(middle) as i32,
        );
        half = len >> 1;
        while 0 < len {
            if ss_compare(
                T,
                PA.offset(getidx(*middle.offset((m + half) as isize)) as isize),
                PA.offset(getidx(*middle.offset((-m - half - 1) as isize)) as isize),
                depth,
            ) < 0
            {
                m += half + 1;
                half -= (len & 1) ^ 1;
            }
            len = half;
            half >>= 1;
        }

        if 0 < m {
            lm = middle.offset(-(m as isize));
            rm = middle.offset(m as isize);
            ss_blockswap(lm, middle, m);
            l = middle;
            r = middle;
            next = 0;
            if rm < last {
                if *rm < 0 {
                    *rm = !*rm;
                    if first < lm {
                        loop {
                            l = l.offset(-1);
                            if !(*l < 0) {
                                break;
                            }
                        }
                        next |= 4;
                    }
                    next |= 1;
                } else if first < lm {
                    while *r < 0 {
                        r = r.offset(1);
                    }
                    next |= 2;
                }
            }

            if (l.offset_from(first) as i32) <= (last.offset_from(r) as i32) {
                stack_push!(r, rm, last, (next & 3) | (check & 4));
                middle = lm;
                last = l;
                check = (check & 3) | (next & 4);
            } else {
                if ((next & 2) != 0) && (r == middle) {
                    next ^= 6;
                }
                stack_push!(first, lm, l, (check & 3) | (next & 4));
                first = r;
                middle = rm;
                check = (next & 3) | (check & 4);
            }
        } else {
            if ss_compare(
                T,
                PA.offset(getidx(*middle.offset(-1)) as isize),
                PA.offset(*middle as isize),
                depth,
            ) == 0
            {
                *middle = !*middle;
            }
            merge_check!(first, last, check);
            stack_pop!(first, middle, last, check);
        }
    }
}

/* ------------------------------------------------------------------------ */

/// Port of `sssort`. Substring sort.
#[allow(clippy::too_many_arguments)]
unsafe fn sssort(
    T: *const u8,
    PA: *const i32,
    mut first: *mut i32,
    last: *mut i32,
    mut buf: *mut i32,
    mut bufsize: i32,
    depth: i32,
    n: i32,
    lastsuffix: i32,
) {
    let mut a: *mut i32;
    let mut b: *mut i32;
    let middle: *mut i32;
    let mut curbuf: *mut i32;
    let mut j: i32;
    let mut k: i32;
    let mut curbufsize: i32;
    let mut limit: i32;
    let mut i: i32;

    if lastsuffix != 0 {
        first = first.offset(1);
    }

    if (bufsize < SS_BLOCKSIZE)
        && (bufsize < (last.offset_from(first) as i32))
        && {
            limit = ss_isqrt(last.offset_from(first) as i32);
            bufsize < limit
        }
    {
        if SS_BLOCKSIZE < limit {
            limit = SS_BLOCKSIZE;
        }
        middle = last.offset(-(limit as isize));
        buf = middle;
        bufsize = limit;
    } else {
        middle = last;
        limit = 0;
    }
    a = first;
    i = 0;
    while SS_BLOCKSIZE < (middle.offset_from(a) as i32) {
        ss_mintrosort(T, PA, a, a.offset(SS_BLOCKSIZE as isize), depth);
        curbufsize = last.offset_from(a.offset(SS_BLOCKSIZE as isize)) as i32;
        curbuf = a.offset(SS_BLOCKSIZE as isize);
        if curbufsize <= bufsize {
            curbufsize = bufsize;
            curbuf = buf;
        }
        b = a;
        k = SS_BLOCKSIZE;
        j = i;
        while (j & 1) != 0 {
            ss_swapmerge(
                T,
                PA,
                b.offset(-(k as isize)),
                b,
                b.offset(k as isize),
                curbuf,
                curbufsize,
                depth,
            );
            b = b.offset(-(k as isize));
            k <<= 1;
            j >>= 1;
        }
        a = a.offset(SS_BLOCKSIZE as isize);
        i += 1;
    }
    ss_mintrosort(T, PA, a, middle, depth);
    k = SS_BLOCKSIZE;
    while i != 0 {
        if (i & 1) != 0 {
            ss_swapmerge(T, PA, a.offset(-(k as isize)), a, middle, buf, bufsize, depth);
            a = a.offset(-(k as isize));
        }
        k <<= 1;
        i >>= 1;
    }
    if limit != 0 {
        ss_mintrosort(T, PA, middle, last, depth);
        ss_inplacemerge(T, PA, first, middle, last, depth);
    }

    if lastsuffix != 0 {
        /* Insert last type B* suffix. */
        let PAi: [i32; 2] = [*PA.offset(*first.offset(-1) as isize), n - 2];
        a = first;
        i = *first.offset(-1);
        while (a < last) && ((*a < 0) || (0 < ss_compare(T, PAi.as_ptr(), PA.offset(*a as isize), depth))) {
            *a.offset(-1) = *a;
            a = a.offset(1);
        }
        *a.offset(-1) = i;
    }
}

/* ======================================================================== */
/* Suffix-array sort (tr_*)                                                  */
/* ======================================================================== */

/// Port of `tr_ilg`.
#[inline(always)]
fn tr_ilg(n: i32) -> i32 {
    if (n & 0xffff0000u32 as i32) != 0 {
        if (n & 0xff000000u32 as i32) != 0 {
            24 + lg_table[((n >> 24) & 0xff) as usize]
        } else {
            16 + lg_table[((n >> 16) & 0xff) as usize]
        }
    } else if (n & 0x0000ff00) != 0 {
        8 + lg_table[((n >> 8) & 0xff) as usize]
    } else {
        lg_table[((n >> 0) & 0xff) as usize]
    }
}

/// Port of `tr_insertionsort`. Simple insertionsort for small size groups.
unsafe fn tr_insertionsort(ISAd: *const i32, first: *mut i32, last: *mut i32) {
    let mut a: *mut i32;
    let mut b: *mut i32;
    let mut t: i32;
    let mut r: i32;

    a = first.offset(1);
    while a < last {
        t = *a;
        b = a.offset(-1);
        loop {
            r = *ISAd.offset(t as isize) - *ISAd.offset(*b as isize);
            if !(0 > r) {
                break;
            }
            loop {
                *b.offset(1) = *b;
                b = b.offset(-1);
                if !((first <= b) && (*b < 0)) {
                    break;
                }
            }
            if b < first {
                break;
            }
        }
        if r == 0 {
            *b = !*b;
        }
        *b.offset(1) = t;
        a = a.offset(1);
    }
}

/* ------------------------------------------------------------------------ */

/// Port of `tr_fixdown`.
#[inline(always)]
unsafe fn tr_fixdown(ISAd: *const i32, SA: *mut i32, mut i: i32, size: i32) {
    let mut j: i32;
    let mut k: i32;
    let v: i32;
    let mut c: i32;
    let mut d: i32;

    v = *SA.offset(i as isize);
    c = *ISAd.offset(v as isize);
    loop {
        j = 2 * i + 1;
        if !(j < size) {
            break;
        }
        k = j;
        j += 1;
        d = *ISAd.offset(*SA.offset(k as isize) as isize);
        let e = *ISAd.offset(*SA.offset(j as isize) as isize);
        if d < e {
            k = j;
            d = e;
        }
        if d <= c {
            break;
        }
        *SA.offset(i as isize) = *SA.offset(k as isize);
        i = k;
    }
    *SA.offset(i as isize) = v;
}

/// Port of `tr_heapsort`. Simple top-down heapsort.
unsafe fn tr_heapsort(ISAd: *const i32, SA: *mut i32, size: i32) {
    let mut i: i32;
    let mut m: i32;
    let mut t: i32;

    m = size;
    if (size % 2) == 0 {
        m -= 1;
        if (*ISAd.offset(*SA.offset((m / 2) as isize) as isize))
            < (*ISAd.offset(*SA.offset(m as isize) as isize))
        {
            t = *SA.offset(m as isize);
            *SA.offset(m as isize) = *SA.offset((m / 2) as isize);
            *SA.offset((m / 2) as isize) = t;
        }
    }

    i = m / 2 - 1;
    while 0 <= i {
        tr_fixdown(ISAd, SA, i, m);
        i -= 1;
    }
    if (size % 2) == 0 {
        t = *SA.offset(0);
        *SA.offset(0) = *SA.offset(m as isize);
        *SA.offset(m as isize) = t;
        tr_fixdown(ISAd, SA, 0, m);
    }
    i = m - 1;
    while 0 < i {
        t = *SA.offset(0);
        *SA.offset(0) = *SA.offset(i as isize);
        tr_fixdown(ISAd, SA, 0, i);
        *SA.offset(i as isize) = t;
        i -= 1;
    }
}

/* ------------------------------------------------------------------------ */

/// Port of `tr_median3`.
#[inline(always)]
unsafe fn tr_median3(ISAd: *const i32, mut v1: *mut i32, mut v2: *mut i32, v3: *mut i32) -> *mut i32 {
    if *ISAd.offset(*v1 as isize) > *ISAd.offset(*v2 as isize) {
        core::mem::swap(&mut v1, &mut v2);
    }
    if *ISAd.offset(*v2 as isize) > *ISAd.offset(*v3 as isize) {
        if *ISAd.offset(*v1 as isize) > *ISAd.offset(*v3 as isize) {
            return v1;
        } else {
            return v3;
        }
    }
    v2
}

/// Port of `tr_median5`.
#[inline(always)]
unsafe fn tr_median5(
    ISAd: *const i32,
    mut v1: *mut i32,
    mut v2: *mut i32,
    mut v3: *mut i32,
    mut v4: *mut i32,
    mut v5: *mut i32,
) -> *mut i32 {
    if *ISAd.offset(*v2 as isize) > *ISAd.offset(*v3 as isize) {
        core::mem::swap(&mut v2, &mut v3);
    }
    if *ISAd.offset(*v4 as isize) > *ISAd.offset(*v5 as isize) {
        core::mem::swap(&mut v4, &mut v5);
    }
    if *ISAd.offset(*v2 as isize) > *ISAd.offset(*v4 as isize) {
        core::mem::swap(&mut v2, &mut v4);
        core::mem::swap(&mut v3, &mut v5);
    }
    if *ISAd.offset(*v1 as isize) > *ISAd.offset(*v3 as isize) {
        core::mem::swap(&mut v1, &mut v3);
    }
    if *ISAd.offset(*v1 as isize) > *ISAd.offset(*v4 as isize) {
        core::mem::swap(&mut v1, &mut v4);
        core::mem::swap(&mut v3, &mut v5);
    }
    if *ISAd.offset(*v3 as isize) > *ISAd.offset(*v4 as isize) {
        return v4;
    }
    v3
}

/// Port of `tr_pivot`.
#[inline(always)]
unsafe fn tr_pivot(ISAd: *const i32, mut first: *mut i32, mut last: *mut i32) -> *mut i32 {
    let mut middle: *mut i32;
    let mut t: i32;

    t = last.offset_from(first) as i32;
    middle = first.offset((t / 2) as isize);

    if t <= 512 {
        if t <= 32 {
            return tr_median3(ISAd, first, middle, last.offset(-1));
        } else {
            t >>= 2;
            return tr_median5(
                ISAd,
                first,
                first.offset(t as isize),
                middle,
                last.offset(-1 - t as isize),
                last.offset(-1),
            );
        }
    }
    t >>= 3;
    first = tr_median3(ISAd, first, first.offset(t as isize), first.offset((t << 1) as isize));
    middle = tr_median3(ISAd, middle.offset(-(t as isize)), middle, middle.offset(t as isize));
    last = tr_median3(
        ISAd,
        last.offset(-1 - (t << 1) as isize),
        last.offset(-1 - t as isize),
        last.offset(-1),
    );
    tr_median3(ISAd, first, middle, last)
}

/* ------------------------------------------------------------------------ */

/// Port of `trbudget_t`.
struct TrBudget {
    chance: i32,
    remain: i32,
    incval: i32,
    count: i32,
}

/// Port of `trbudget_init`.
#[inline(always)]
fn trbudget_init(budget: &mut TrBudget, chance: i32, incval: i32) {
    budget.chance = chance;
    budget.incval = incval;
    budget.remain = incval;
}

/// Port of `trbudget_check`.
#[inline(always)]
fn trbudget_check(budget: &mut TrBudget, size: i32) -> i32 {
    if size <= budget.remain {
        budget.remain -= size;
        return 1;
    }
    if budget.chance == 0 {
        budget.count += size;
        return 0;
    }
    budget.remain += budget.incval - size;
    budget.chance -= 1;
    1
}

/* ------------------------------------------------------------------------ */

/// Port of `tr_partition`. Returns `(*pa, *pb)`.
#[inline(always)]
unsafe fn tr_partition(
    ISAd: *const i32,
    mut first: *mut i32,
    middle: *mut i32,
    mut last: *mut i32,
    v: i32,
) -> (*mut i32, *mut i32) {
    let mut a: *mut i32;
    let mut b: *mut i32;
    let mut c: *mut i32;
    let mut d: *mut i32;
    let mut e: *mut i32;
    let mut f: *mut i32;
    let mut t: i32;
    let mut s: i32;
    let mut x: i32 = 0;

    b = middle.offset(-1);
    loop {
        b = b.offset(1);
        if !(b < last) {
            break;
        }
        x = *ISAd.offset(*b as isize);
        if x != v {
            break;
        }
    }
    a = b;
    if (a < last) && (x < v) {
        loop {
            b = b.offset(1);
            if !(b < last) {
                break;
            }
            x = *ISAd.offset(*b as isize);
            if x > v {
                break;
            }
            if x == v {
                core::ptr::swap(b, a);
                a = a.offset(1);
            }
        }
    }
    c = last;
    loop {
        c = c.offset(-1);
        if !(b < c) {
            break;
        }
        x = *ISAd.offset(*c as isize);
        if x != v {
            break;
        }
    }
    d = c;
    if (b < d) && (x > v) {
        loop {
            c = c.offset(-1);
            if !(b < c) {
                break;
            }
            x = *ISAd.offset(*c as isize);
            if x < v {
                break;
            }
            if x == v {
                core::ptr::swap(c, d);
                d = d.offset(-1);
            }
        }
    }
    while b < c {
        core::ptr::swap(b, c);
        loop {
            b = b.offset(1);
            if !(b < c) {
                break;
            }
            x = *ISAd.offset(*b as isize);
            if x > v {
                break;
            }
            if x == v {
                core::ptr::swap(b, a);
                a = a.offset(1);
            }
        }
        loop {
            c = c.offset(-1);
            if !(b < c) {
                break;
            }
            x = *ISAd.offset(*c as isize);
            if x < v {
                break;
            }
            if x == v {
                core::ptr::swap(c, d);
                d = d.offset(-1);
            }
        }
    }

    if a <= d {
        c = b.offset(-1);
        s = a.offset_from(first) as i32;
        t = b.offset_from(a) as i32;
        if s > t {
            s = t;
        }
        e = first;
        f = b.offset(-(s as isize));
        while 0 < s {
            core::ptr::swap(e, f);
            s -= 1;
            e = e.offset(1);
            f = f.offset(1);
        }
        s = d.offset_from(c) as i32;
        t = (last.offset_from(d) - 1) as i32;
        if s > t {
            s = t;
        }
        e = b;
        f = last.offset(-(s as isize));
        while 0 < s {
            core::ptr::swap(e, f);
            s -= 1;
            e = e.offset(1);
            f = f.offset(1);
        }
        first = first.offset(b.offset_from(a));
        last = last.offset(-(d.offset_from(c)));
    }
    (first, last)
}

/// Port of `tr_copy`.
unsafe fn tr_copy(
    ISA: *mut i32,
    SA: *const i32,
    first: *mut i32,
    a: *mut i32,
    b: *mut i32,
    last: *mut i32,
    depth: i32,
) {
    let mut c: *mut i32;
    let mut d: *mut i32;
    let mut e: *mut i32;
    let mut s: i32;
    let v: i32;

    v = (b.offset_from(SA) - 1) as i32;
    c = first;
    d = a.offset(-1);
    while c <= d {
        s = *c - depth;
        if (0 <= s) && (*ISA.offset(s as isize) == v) {
            d = d.offset(1);
            *d = s;
            *ISA.offset(s as isize) = d.offset_from(SA) as i32;
        }
        c = c.offset(1);
    }
    c = last.offset(-1);
    e = d.offset(1);
    d = b;
    while e < d {
        s = *c - depth;
        if (0 <= s) && (*ISA.offset(s as isize) == v) {
            d = d.offset(-1);
            *d = s;
            *ISA.offset(s as isize) = d.offset_from(SA) as i32;
        }
        c = c.offset(-1);
    }
}

/// Port of `tr_partialcopy`.
unsafe fn tr_partialcopy(
    ISA: *mut i32,
    SA: *const i32,
    first: *mut i32,
    a: *mut i32,
    b: *mut i32,
    last: *mut i32,
    depth: i32,
) {
    let mut c: *mut i32;
    let mut d: *mut i32;
    let mut e: *mut i32;
    let mut s: i32;
    let v: i32;
    let mut rank: i32;
    let mut lastrank: i32;
    let mut newrank: i32 = -1;

    v = (b.offset_from(SA) - 1) as i32;
    lastrank = -1;
    c = first;
    d = a.offset(-1);
    while c <= d {
        s = *c - depth;
        if (0 <= s) && (*ISA.offset(s as isize) == v) {
            d = d.offset(1);
            *d = s;
            rank = *ISA.offset((s + depth) as isize);
            if lastrank != rank {
                lastrank = rank;
                newrank = d.offset_from(SA) as i32;
            }
            *ISA.offset(s as isize) = newrank;
        }
        c = c.offset(1);
    }

    lastrank = -1;
    e = d;
    while first <= e {
        rank = *ISA.offset(*e as isize);
        if lastrank != rank {
            lastrank = rank;
            newrank = e.offset_from(SA) as i32;
        }
        if newrank != rank {
            *ISA.offset(*e as isize) = newrank;
        }
        e = e.offset(-1);
    }

    lastrank = -1;
    c = last.offset(-1);
    e = d.offset(1);
    d = b;
    while e < d {
        s = *c - depth;
        if (0 <= s) && (*ISA.offset(s as isize) == v) {
            d = d.offset(-1);
            *d = s;
            rank = *ISA.offset((s + depth) as isize);
            if lastrank != rank {
                lastrank = rank;
                newrank = d.offset_from(SA) as i32;
            }
            *ISA.offset(s as isize) = newrank;
        }
        c = c.offset(-1);
    }
}

/// Port of `tr_introsort`.
unsafe fn tr_introsort(
    ISA: *mut i32,
    mut ISAd: *const i32,
    SA: *mut i32,
    mut first: *mut i32,
    mut last: *mut i32,
    budget: &mut TrBudget,
) {
    #[derive(Clone, Copy)]
    struct StackEntry {
        a: *const i32,
        b: *mut i32,
        c: *mut i32,
        d: i32,
        e: i32,
    }
    let mut stack = [StackEntry {
        a: core::ptr::null(),
        b: core::ptr::null_mut(),
        c: core::ptr::null_mut(),
        d: 0,
        e: 0,
    }; TR_STACKSIZE];
    let mut ssize: usize = 0;

    let mut a: *mut i32;
    let mut b: *mut i32;
    let mut c: *mut i32;
    let mut v: i32;
    let mut x: i32 = 0;
    let incr: i32 = ISAd.offset_from(ISA as *const i32) as i32;
    let mut limit: i32;
    let mut next: i32;
    let mut trlink: i32 = -1;

    macro_rules! stack_push5 {
        ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr) => {{
            stack[ssize] = StackEntry {
                a: $a,
                b: $b,
                c: $c,
                d: $d,
                e: $e,
            };
            ssize += 1;
        }};
    }
    macro_rules! stack_pop5 {
        ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident) => {{
            if ssize == 0 {
                return;
            }
            ssize -= 1;
            $a = stack[ssize].a;
            $b = stack[ssize].b;
            $c = stack[ssize].c;
            $d = stack[ssize].d;
            $e = stack[ssize].e;
        }};
    }

    limit = tr_ilg(last.offset_from(first) as i32);
    loop {
        if limit < 0 {
            if limit == -1 {
                /* tandem repeat partition */
                let (na, nb) = tr_partition(
                    ISAd.offset(-(incr as isize)),
                    first,
                    first,
                    last,
                    (last.offset_from(SA) - 1) as i32,
                );
                a = na;
                b = nb;

                /* update ranks */
                if a < last {
                    c = first;
                    v = (a.offset_from(SA) - 1) as i32;
                    while c < a {
                        *ISA.offset(*c as isize) = v;
                        c = c.offset(1);
                    }
                }
                if b < last {
                    c = a;
                    v = (b.offset_from(SA) - 1) as i32;
                    while c < b {
                        *ISA.offset(*c as isize) = v;
                        c = c.offset(1);
                    }
                }

                /* push */
                if 1 < b.offset_from(a) as i32 {
                    stack_push5!(core::ptr::null(), a, b, 0, 0);
                    stack_push5!(ISAd.offset(-(incr as isize)), first, last, -2, trlink);
                    trlink = ssize as i32 - 2;
                }
                if (a.offset_from(first) as i32) <= (last.offset_from(b) as i32) {
                    if 1 < a.offset_from(first) as i32 {
                        stack_push5!(ISAd, b, last, tr_ilg(last.offset_from(b) as i32), trlink);
                        last = a;
                        limit = tr_ilg(a.offset_from(first) as i32);
                    } else if 1 < last.offset_from(b) as i32 {
                        first = b;
                        limit = tr_ilg(last.offset_from(b) as i32);
                    } else {
                        stack_pop5!(ISAd, first, last, limit, trlink);
                    }
                } else if 1 < last.offset_from(b) as i32 {
                    stack_push5!(ISAd, first, a, tr_ilg(a.offset_from(first) as i32), trlink);
                    first = b;
                    limit = tr_ilg(last.offset_from(b) as i32);
                } else if 1 < a.offset_from(first) as i32 {
                    last = a;
                    limit = tr_ilg(a.offset_from(first) as i32);
                } else {
                    stack_pop5!(ISAd, first, last, limit, trlink);
                }
            } else if limit == -2 {
                /* tandem repeat copy */
                ssize -= 1;
                a = stack[ssize].b;
                b = stack[ssize].c;
                if stack[ssize].d == 0 {
                    tr_copy(ISA, SA, first, a, b, last, ISAd.offset_from(ISA as *const i32) as i32);
                } else {
                    if 0 <= trlink {
                        stack[trlink as usize].d = -1;
                    }
                    tr_partialcopy(
                        ISA,
                        SA,
                        first,
                        a,
                        b,
                        last,
                        ISAd.offset_from(ISA as *const i32) as i32,
                    );
                }
                stack_pop5!(ISAd, first, last, limit, trlink);
            } else {
                /* sorted partition */
                if 0 <= *first {
                    a = first;
                    loop {
                        *ISA.offset(*a as isize) = a.offset_from(SA) as i32;
                        a = a.offset(1);
                        if !((a < last) && (0 <= *a)) {
                            break;
                        }
                    }
                    first = a;
                }
                if first < last {
                    a = first;
                    loop {
                        *a = !*a;
                        a = a.offset(1);
                        if !(*a < 0) {
                            break;
                        }
                    }
                    next = if *ISA.offset(*a as isize) != *ISAd.offset(*a as isize) {
                        tr_ilg((a.offset_from(first) + 1) as i32)
                    } else {
                        -1
                    };
                    a = a.offset(1);
                    if a < last {
                        b = first;
                        v = (a.offset_from(SA) - 1) as i32;
                        while b < a {
                            *ISA.offset(*b as isize) = v;
                            b = b.offset(1);
                        }
                    }

                    /* push */
                    if trbudget_check(budget, a.offset_from(first) as i32) != 0 {
                        if (a.offset_from(first) as i32) <= (last.offset_from(a) as i32) {
                            stack_push5!(ISAd, a, last, -3, trlink);
                            ISAd = ISAd.offset(incr as isize);
                            last = a;
                            limit = next;
                        } else if 1 < last.offset_from(a) as i32 {
                            stack_push5!(ISAd.offset(incr as isize), first, a, next, trlink);
                            first = a;
                            limit = -3;
                        } else {
                            ISAd = ISAd.offset(incr as isize);
                            last = a;
                            limit = next;
                        }
                    } else {
                        if 0 <= trlink {
                            stack[trlink as usize].d = -1;
                        }
                        if 1 < last.offset_from(a) as i32 {
                            first = a;
                            limit = -3;
                        } else {
                            stack_pop5!(ISAd, first, last, limit, trlink);
                        }
                    }
                } else {
                    stack_pop5!(ISAd, first, last, limit, trlink);
                }
            }
            continue;
        }

        if (last.offset_from(first) as i32) <= TR_INSERTIONSORT_THRESHOLD {
            tr_insertionsort(ISAd, first, last);
            limit = -3;
            continue;
        }

        let prev_limit = limit;
        limit -= 1;
        if prev_limit == 0 {
            tr_heapsort(ISAd, first, last.offset_from(first) as i32);
            a = last.offset(-1);
            while first < a {
                x = *ISAd.offset(*a as isize);
                b = a.offset(-1);
                while (first <= b) && (*ISAd.offset(*b as isize) == x) {
                    *b = !*b;
                    b = b.offset(-1);
                }
                a = b;
            }
            limit = -3;
            continue;
        }

        /* choose pivot */
        a = tr_pivot(ISAd, first, last);
        core::ptr::swap(first, a);
        v = *ISAd.offset(*first as isize);

        /* partition */
        let (na, nb) = tr_partition(ISAd, first, first.offset(1), last, v);
        a = na;
        b = nb;
        if (last.offset_from(first) as i32) != (b.offset_from(a) as i32) {
            next = if *ISA.offset(*a as isize) != v {
                tr_ilg(b.offset_from(a) as i32)
            } else {
                -1
            };

            /* update ranks */
            c = first;
            v = (a.offset_from(SA) - 1) as i32;
            while c < a {
                *ISA.offset(*c as isize) = v;
                c = c.offset(1);
            }
            if b < last {
                c = a;
                v = (b.offset_from(SA) - 1) as i32;
                while c < b {
                    *ISA.offset(*c as isize) = v;
                    c = c.offset(1);
                }
            }

            /* push */
            if (1 < b.offset_from(a) as i32) && (trbudget_check(budget, b.offset_from(a) as i32) != 0) {
                if (a.offset_from(first) as i32) <= (last.offset_from(b) as i32) {
                    if (last.offset_from(b) as i32) <= (b.offset_from(a) as i32) {
                        if 1 < a.offset_from(first) as i32 {
                            stack_push5!(ISAd.offset(incr as isize), a, b, next, trlink);
                            stack_push5!(ISAd, b, last, limit, trlink);
                            last = a;
                        } else if 1 < last.offset_from(b) as i32 {
                            stack_push5!(ISAd.offset(incr as isize), a, b, next, trlink);
                            first = b;
                        } else {
                            ISAd = ISAd.offset(incr as isize);
                            first = a;
                            last = b;
                            limit = next;
                        }
                    } else if (a.offset_from(first) as i32) <= (b.offset_from(a) as i32) {
                        if 1 < a.offset_from(first) as i32 {
                            stack_push5!(ISAd, b, last, limit, trlink);
                            stack_push5!(ISAd.offset(incr as isize), a, b, next, trlink);
                            last = a;
                        } else {
                            stack_push5!(ISAd, b, last, limit, trlink);
                            ISAd = ISAd.offset(incr as isize);
                            first = a;
                            last = b;
                            limit = next;
                        }
                    } else {
                        stack_push5!(ISAd, b, last, limit, trlink);
                        stack_push5!(ISAd, first, a, limit, trlink);
                        ISAd = ISAd.offset(incr as isize);
                        first = a;
                        last = b;
                        limit = next;
                    }
                } else if (a.offset_from(first) as i32) <= (b.offset_from(a) as i32) {
                    if 1 < last.offset_from(b) as i32 {
                        stack_push5!(ISAd.offset(incr as isize), a, b, next, trlink);
                        stack_push5!(ISAd, first, a, limit, trlink);
                        first = b;
                    } else if 1 < a.offset_from(first) as i32 {
                        stack_push5!(ISAd.offset(incr as isize), a, b, next, trlink);
                        last = a;
                    } else {
                        ISAd = ISAd.offset(incr as isize);
                        first = a;
                        last = b;
                        limit = next;
                    }
                } else if (last.offset_from(b) as i32) <= (b.offset_from(a) as i32) {
                    if 1 < last.offset_from(b) as i32 {
                        stack_push5!(ISAd, first, a, limit, trlink);
                        stack_push5!(ISAd.offset(incr as isize), a, b, next, trlink);
                        first = b;
                    } else {
                        stack_push5!(ISAd, first, a, limit, trlink);
                        ISAd = ISAd.offset(incr as isize);
                        first = a;
                        last = b;
                        limit = next;
                    }
                } else {
                    stack_push5!(ISAd, first, a, limit, trlink);
                    stack_push5!(ISAd, b, last, limit, trlink);
                    ISAd = ISAd.offset(incr as isize);
                    first = a;
                    last = b;
                    limit = next;
                }
            } else {
                if (1 < b.offset_from(a) as i32) && (0 <= trlink) {
                    stack[trlink as usize].d = -1;
                }
                if (a.offset_from(first) as i32) <= (last.offset_from(b) as i32) {
                    if 1 < a.offset_from(first) as i32 {
                        stack_push5!(ISAd, b, last, limit, trlink);
                        last = a;
                    } else if 1 < last.offset_from(b) as i32 {
                        first = b;
                    } else {
                        stack_pop5!(ISAd, first, last, limit, trlink);
                    }
                } else if 1 < last.offset_from(b) as i32 {
                    stack_push5!(ISAd, first, a, limit, trlink);
                    first = b;
                } else if 1 < a.offset_from(first) as i32 {
                    last = a;
                } else {
                    stack_pop5!(ISAd, first, last, limit, trlink);
                }
            }
        } else if trbudget_check(budget, last.offset_from(first) as i32) != 0 {
            limit = tr_ilg(last.offset_from(first) as i32);
            ISAd = ISAd.offset(incr as isize);
        } else {
            if 0 <= trlink {
                stack[trlink as usize].d = -1;
            }
            stack_pop5!(ISAd, first, last, limit, trlink);
        }
    }
}

/* ------------------------------------------------------------------------ */

/// Port of `trsort`. Tandem repeat sort.
unsafe fn trsort(ISA: *mut i32, SA: *mut i32, n: i32, depth: i32) {
    let mut ISAd: *mut i32;
    let mut first: *mut i32;
    let mut last: *mut i32;
    let mut budget = TrBudget {
        chance: 0,
        remain: 0,
        incval: 0,
        count: 0,
    };
    let mut t: i32;
    let mut skip: i32;
    let mut unsorted: i32;

    trbudget_init(&mut budget, tr_ilg(n) * 2 / 3, n);
    ISAd = ISA.offset(depth as isize);
    while -n < *SA {
        first = SA;
        skip = 0;
        unsorted = 0;
        loop {
            t = *first;
            if t < 0 {
                first = first.offset(-(t as isize));
                skip += t;
            } else {
                if skip != 0 {
                    *first.offset(skip as isize) = skip;
                    skip = 0;
                }
                last = SA.offset((*ISA.offset(t as isize) + 1) as isize);
                if 1 < last.offset_from(first) as i32 {
                    budget.count = 0;
                    tr_introsort(ISA, ISAd as *const i32, SA, first, last, &mut budget);
                    if budget.count != 0 {
                        unsorted += budget.count;
                    } else {
                        skip = first.offset_from(last) as i32;
                    }
                } else if (last.offset_from(first) as i32) == 1 {
                    skip = -1;
                }
                first = last;
            }
            if !(first < SA.offset(n as isize)) {
                break;
            }
        }
        if skip != 0 {
            *first.offset(skip as isize) = skip;
        }
        if unsorted == 0 {
            break;
        }
        // ISAd += ISAd - ISA
        ISAd = ISAd.offset(ISAd.offset_from(ISA as *const i32));
    }
}

/* ======================================================================== */
/* Construction                                                              */
/* ======================================================================== */

/// Port of `sort_typeBstar`. Sorts suffixes of type B*; returns `m`.
/// (`openMP` is dropped — `LIBBSC_OPENMP` is not defined in zstd.)
unsafe fn sort_typeBstar(
    T: *const u8,
    SA: *mut i32,
    bucket_A: *mut i32,
    bucket_B: *mut i32,
    n: i32,
) -> i32 {
    let PAb: *mut i32;
    let ISAb: *mut i32;
    let buf: *mut i32;
    let mut i: i32;
    let mut j: i32;
    let mut k: i32;
    let mut t: i32;
    let mut m: i32;
    let bufsize: i32;
    let mut c0: i32;
    let mut c1: i32;

    // BUCKET_A(c0)        -> *bucket_A.offset(c0)
    // BUCKET_B(c0, c1)    -> *bucket_B.offset((c1 << 8) | c0)
    // BUCKET_BSTAR(c0,c1) -> *bucket_B.offset((c0 << 8) | c1)
    macro_rules! BUCKET_A {
        ($c0:expr) => {
            *bucket_A.offset(($c0) as isize)
        };
    }
    macro_rules! BUCKET_B {
        ($c0:expr, $c1:expr) => {
            *bucket_B.offset(bucket_b_idx($c0, $c1) as isize)
        };
    }
    macro_rules! BUCKET_BSTAR {
        ($c0:expr, $c1:expr) => {
            *bucket_B.offset(bucket_bstar_idx($c0, $c1) as isize)
        };
    }

    /* Initialize bucket arrays. */
    i = 0;
    while i < BUCKET_A_SIZE as i32 {
        *bucket_A.offset(i as isize) = 0;
        i += 1;
    }
    i = 0;
    while i < BUCKET_B_SIZE as i32 {
        *bucket_B.offset(i as isize) = 0;
        i += 1;
    }

    /* Count the number of occurrences of the first one or two characters of
       each type A, B and B* suffix. */
    i = n - 1;
    m = n;
    c0 = *T.offset((n - 1) as isize) as i32;
    while 0 <= i {
        /* type A suffix. */
        loop {
            c1 = c0;
            BUCKET_A!(c1) += 1;
            i -= 1;
            if !((0 <= i) && {
                c0 = *T.offset(i as isize) as i32;
                c0 >= c1
            }) {
                break;
            }
        }
        if 0 <= i {
            /* type B* suffix. */
            BUCKET_BSTAR!(c0, c1) += 1;
            m -= 1;
            *SA.offset(m as isize) = i;
            /* type B suffix. */
            i -= 1;
            c1 = c0;
            while (0 <= i) && {
                c0 = *T.offset(i as isize) as i32;
                c0 <= c1
            } {
                BUCKET_B!(c0, c1) += 1;
                i -= 1;
                c1 = c0;
            }
        }
    }
    m = n - m;

    /* Calculate the index of start/end point of each bucket. */
    c0 = 0;
    i = 0;
    j = 0;
    while c0 < ALPHABET_SIZE {
        t = i + BUCKET_A!(c0);
        BUCKET_A!(c0) = i + j; /* start point */
        i = t + BUCKET_B!(c0, c0);
        c1 = c0 + 1;
        while c1 < ALPHABET_SIZE {
            j += BUCKET_BSTAR!(c0, c1);
            BUCKET_BSTAR!(c0, c1) = j; /* end point */
            i += BUCKET_B!(c0, c1);
            c1 += 1;
        }
        c0 += 1;
    }

    if 0 < m {
        /* Sort the type B* suffixes by their first two characters. */
        PAb = SA.offset((n - m) as isize);
        ISAb = SA.offset(m as isize);
        i = m - 2;
        while 0 <= i {
            t = *PAb.offset(i as isize);
            c0 = *T.offset(t as isize) as i32;
            c1 = *T.offset((t + 1) as isize) as i32;
            BUCKET_BSTAR!(c0, c1) -= 1;
            *SA.offset(BUCKET_BSTAR!(c0, c1) as isize) = i;
            i -= 1;
        }
        t = *PAb.offset((m - 1) as isize);
        c0 = *T.offset(t as isize) as i32;
        c1 = *T.offset((t + 1) as isize) as i32;
        BUCKET_BSTAR!(c0, c1) -= 1;
        *SA.offset(BUCKET_BSTAR!(c0, c1) as isize) = m - 1;

        /* Sort the type B* substrings using sssort. */
        buf = SA.offset(m as isize);
        bufsize = n - (2 * m);
        c0 = ALPHABET_SIZE - 2;
        j = m;
        while 0 < j {
            c1 = ALPHABET_SIZE - 1;
            while c0 < c1 {
                i = BUCKET_BSTAR!(c0, c1);
                if 1 < (j - i) {
                    sssort(
                        T,
                        PAb,
                        SA.offset(i as isize),
                        SA.offset(j as isize),
                        buf,
                        bufsize,
                        2,
                        n,
                        (*SA.offset(i as isize) == (m - 1)) as i32,
                    );
                }
                j = i;
                c1 -= 1;
            }
            c0 -= 1;
        }

        /* Compute ranks of type B* substrings. */
        i = m - 1;
        while 0 <= i {
            if 0 <= *SA.offset(i as isize) {
                j = i;
                loop {
                    *ISAb.offset(*SA.offset(i as isize) as isize) = i;
                    i -= 1;
                    if !((0 <= i) && (0 <= *SA.offset(i as isize))) {
                        break;
                    }
                }
                *SA.offset((i + 1) as isize) = i - j;
                if i <= 0 {
                    break;
                }
            }
            j = i;
            loop {
                *SA.offset(i as isize) = !*SA.offset(i as isize);
                *ISAb.offset(*SA.offset(i as isize) as isize) = j;
                i -= 1;
                if !(*SA.offset(i as isize) < 0) {
                    break;
                }
            }
            *ISAb.offset(*SA.offset(i as isize) as isize) = j;
        }

        /* Construct the inverse suffix array of type B* suffixes using trsort. */
        trsort(ISAb, SA, m, 1);

        /* Set the sorted order of type B* suffixes. */
        i = n - 1;
        j = m;
        c0 = *T.offset((n - 1) as isize) as i32;
        while 0 <= i {
            i -= 1;
            c1 = c0;
            while (0 <= i) && {
                c0 = *T.offset(i as isize) as i32;
                c0 >= c1
            } {
                i -= 1;
                c1 = c0;
            }
            if 0 <= i {
                t = i;
                i -= 1;
                c1 = c0;
                while (0 <= i) && {
                    c0 = *T.offset(i as isize) as i32;
                    c0 <= c1
                } {
                    i -= 1;
                    c1 = c0;
                }
                j -= 1;
                *SA.offset(*ISAb.offset(j as isize) as isize) =
                    if (t == 0) || (1 < (t - i)) { t } else { !t };
            }
        }

        /* Calculate the index of start/end point of each bucket. */
        BUCKET_B!(ALPHABET_SIZE - 1, ALPHABET_SIZE - 1) = n; /* end point */
        c0 = ALPHABET_SIZE - 2;
        k = m - 1;
        while 0 <= c0 {
            i = BUCKET_A!(c0 + 1) - 1;
            c1 = ALPHABET_SIZE - 1;
            while c0 < c1 {
                t = i - BUCKET_B!(c0, c1);
                BUCKET_B!(c0, c1) = i; /* end point */

                /* Move all type B* suffixes to the correct position. */
                i = t;
                j = BUCKET_BSTAR!(c0, c1);
                while j <= k {
                    *SA.offset(i as isize) = *SA.offset(k as isize);
                    i -= 1;
                    k -= 1;
                }
                c1 -= 1;
            }
            BUCKET_BSTAR!(c0, c0 + 1) = i - BUCKET_B!(c0, c0) + 1; /* start point */
            BUCKET_B!(c0, c0) = i; /* end point */
            c0 -= 1;
        }
    }

    m
}

/// Port of `construct_SA`. Constructs the suffix array using the sorted
/// order of type B* suffixes.
unsafe fn construct_SA(T: *const u8, SA: *mut i32, bucket_A: *mut i32, bucket_B: *mut i32, n: i32, m: i32) {
    let mut i: *mut i32;
    let mut j: *mut i32;
    let mut k: *mut i32;
    let mut s: i32;
    let mut c0: i32;
    let mut c1: i32;
    let mut c2: i32;

    macro_rules! BUCKET_A {
        ($c0:expr) => {
            *bucket_A.offset(($c0) as isize)
        };
    }
    macro_rules! BUCKET_B {
        ($c0:expr, $c1:expr) => {
            *bucket_B.offset(bucket_b_idx($c0, $c1) as isize)
        };
    }
    macro_rules! BUCKET_BSTAR {
        ($c0:expr, $c1:expr) => {
            *bucket_B.offset(bucket_bstar_idx($c0, $c1) as isize)
        };
    }

    if 0 < m {
        /* Construct the sorted order of type B suffixes. */
        c1 = ALPHABET_SIZE - 2;
        while 0 <= c1 {
            /* Scan the suffix array from right to left. */
            i = SA.offset(BUCKET_BSTAR!(c1, c1 + 1) as isize);
            j = SA.offset((BUCKET_A!(c1 + 1) - 1) as isize);
            k = core::ptr::null_mut();
            c2 = -1;
            while i <= j {
                s = *j;
                if 0 < s {
                    *j = !s;
                    s -= 1;
                    c0 = *T.offset(s as isize) as i32;
                    if (0 < s) && ((*T.offset((s - 1) as isize) as i32) > c0) {
                        s = !s;
                    }
                    if c0 != c2 {
                        if 0 <= c2 {
                            BUCKET_B!(c2, c1) = k.offset_from(SA) as i32;
                        }
                        c2 = c0;
                        k = SA.offset(BUCKET_B!(c2, c1) as isize);
                    }
                    *k = s;
                    k = k.offset(-1);
                } else {
                    *j = !s;
                }
                j = j.offset(-1);
            }
            c1 -= 1;
        }
    }

    /* Construct the suffix array by using the sorted order of type B suffixes. */
    c2 = *T.offset((n - 1) as isize) as i32;
    k = SA.offset(BUCKET_A!(c2) as isize);
    *k = if (*T.offset((n - 2) as isize) as i32) < c2 {
        !(n - 1)
    } else {
        n - 1
    };
    k = k.offset(1);
    /* Scan the suffix array from left to right. */
    i = SA;
    j = SA.offset(n as isize);
    while i < j {
        s = *i;
        if 0 < s {
            // c0 = T[--s];
            s -= 1;
            c0 = *T.offset(s as isize) as i32;
            if (s == 0) || ((*T.offset((s - 1) as isize) as i32) < c0) {
                s = !s;
            }
            if c0 != c2 {
                BUCKET_A!(c2) = k.offset_from(SA) as i32;
                c2 = c0;
                k = SA.offset(BUCKET_A!(c2) as isize);
            }
            *k = s;
            k = k.offset(1);
        } else {
            *i = !s;
        }
        i = i.offset(1);
    }
}

/// Port of `construct_BWT`. Builds the BWT directly; returns the primary index.
unsafe fn construct_BWT(T: *const u8, SA: *mut i32, bucket_A: *mut i32, bucket_B: *mut i32, n: i32, m: i32) -> i32 {
    let mut i: *mut i32;
    let mut j: *mut i32;
    let mut k: *mut i32;
    let mut orig: *mut i32;
    let mut s: i32;
    let mut c0: i32;
    let mut c1: i32;
    let mut c2: i32;

    macro_rules! BUCKET_A {
        ($c0:expr) => {
            *bucket_A.offset(($c0) as isize)
        };
    }
    macro_rules! BUCKET_B {
        ($c0:expr, $c1:expr) => {
            *bucket_B.offset(bucket_b_idx($c0, $c1) as isize)
        };
    }
    macro_rules! BUCKET_BSTAR {
        ($c0:expr, $c1:expr) => {
            *bucket_B.offset(bucket_bstar_idx($c0, $c1) as isize)
        };
    }

    if 0 < m {
        c1 = ALPHABET_SIZE - 2;
        while 0 <= c1 {
            i = SA.offset(BUCKET_BSTAR!(c1, c1 + 1) as isize);
            j = SA.offset((BUCKET_A!(c1 + 1) - 1) as isize);
            k = core::ptr::null_mut();
            c2 = -1;
            while i <= j {
                s = *j;
                if 0 < s {
                    s -= 1;
                    c0 = *T.offset(s as isize) as i32;
                    *j = !c0;
                    if (0 < s) && ((*T.offset((s - 1) as isize) as i32) > c0) {
                        s = !s;
                    }
                    if c0 != c2 {
                        if 0 <= c2 {
                            BUCKET_B!(c2, c1) = k.offset_from(SA) as i32;
                        }
                        c2 = c0;
                        k = SA.offset(BUCKET_B!(c2, c1) as isize);
                    }
                    *k = s;
                    k = k.offset(-1);
                } else if s != 0 {
                    *j = !s;
                }
                j = j.offset(-1);
            }
            c1 -= 1;
        }
    }

    c2 = *T.offset((n - 1) as isize) as i32;
    k = SA.offset(BUCKET_A!(c2) as isize);
    *k = if (*T.offset((n - 2) as isize) as i32) < c2 {
        !(*T.offset((n - 2) as isize) as i32)
    } else {
        n - 1
    };
    k = k.offset(1);
    i = SA;
    j = SA.offset(n as isize);
    orig = SA;
    while i < j {
        s = *i;
        if 0 < s {
            s -= 1;
            c0 = *T.offset(s as isize) as i32;
            *i = c0;
            if (0 < s) && ((*T.offset((s - 1) as isize) as i32) < c0) {
                s = !(*T.offset((s - 1) as isize) as i32);
            }
            if c0 != c2 {
                BUCKET_A!(c2) = k.offset_from(SA) as i32;
                c2 = c0;
                k = SA.offset(BUCKET_A!(c2) as isize);
            }
            *k = s;
            k = k.offset(1);
        } else if s != 0 {
            *i = !s;
        } else {
            orig = i;
        }
        i = i.offset(1);
    }

    orig.offset_from(SA) as i32
}

/// Port of `construct_BWT_indexes`.
#[allow(clippy::too_many_arguments)]
unsafe fn construct_BWT_indexes(
    T: *const u8,
    SA: *mut i32,
    bucket_A: *mut i32,
    bucket_B: *mut i32,
    n: i32,
    m: i32,
    num_indexes: *mut u8,
    indexes: *mut i32,
) -> i32 {
    let mut i: *mut i32;
    let mut j: *mut i32;
    let mut k: *mut i32;
    let mut orig: *mut i32;
    let mut s: i32;
    let mut c0: i32;
    let mut c1: i32;
    let mut c2: i32;

    macro_rules! BUCKET_A {
        ($c0:expr) => {
            *bucket_A.offset(($c0) as isize)
        };
    }
    macro_rules! BUCKET_B {
        ($c0:expr, $c1:expr) => {
            *bucket_B.offset(bucket_b_idx($c0, $c1) as isize)
        };
    }
    macro_rules! BUCKET_BSTAR {
        ($c0:expr, $c1:expr) => {
            *bucket_B.offset(bucket_bstar_idx($c0, $c1) as isize)
        };
    }

    let mut mod_: i32 = n / 8;
    {
        mod_ |= mod_ >> 1;
        mod_ |= mod_ >> 2;
        mod_ |= mod_ >> 4;
        mod_ |= mod_ >> 8;
        mod_ |= mod_ >> 16;
        mod_ >>= 1;
        *num_indexes = ((n - 1) / (mod_ + 1)) as u8;
    }

    if 0 < m {
        c1 = ALPHABET_SIZE - 2;
        while 0 <= c1 {
            i = SA.offset(BUCKET_BSTAR!(c1, c1 + 1) as isize);
            j = SA.offset((BUCKET_A!(c1 + 1) - 1) as isize);
            k = core::ptr::null_mut();
            c2 = -1;
            while i <= j {
                s = *j;
                if 0 < s {
                    if (s & mod_) == 0 {
                        *indexes.offset((s / (mod_ + 1) - 1) as isize) = j.offset_from(SA) as i32;
                    }
                    s -= 1;
                    c0 = *T.offset(s as isize) as i32;
                    *j = !c0;
                    if (0 < s) && ((*T.offset((s - 1) as isize) as i32) > c0) {
                        s = !s;
                    }
                    if c0 != c2 {
                        if 0 <= c2 {
                            BUCKET_B!(c2, c1) = k.offset_from(SA) as i32;
                        }
                        c2 = c0;
                        k = SA.offset(BUCKET_B!(c2, c1) as isize);
                    }
                    *k = s;
                    k = k.offset(-1);
                } else if s != 0 {
                    *j = !s;
                }
                j = j.offset(-1);
            }
            c1 -= 1;
        }
    }

    c2 = *T.offset((n - 1) as isize) as i32;
    k = SA.offset(BUCKET_A!(c2) as isize);
    if (*T.offset((n - 2) as isize) as i32) < c2 {
        if ((n - 1) & mod_) == 0 {
            *indexes.offset(((n - 1) / (mod_ + 1) - 1) as isize) = k.offset_from(SA) as i32;
        }
        *k = !(*T.offset((n - 2) as isize) as i32);
        k = k.offset(1);
    } else {
        *k = n - 1;
        k = k.offset(1);
    }

    i = SA;
    j = SA.offset(n as isize);
    orig = SA;
    while i < j {
        s = *i;
        if 0 < s {
            if (s & mod_) == 0 {
                *indexes.offset((s / (mod_ + 1) - 1) as isize) = i.offset_from(SA) as i32;
            }
            s -= 1;
            c0 = *T.offset(s as isize) as i32;
            *i = c0;
            if c0 != c2 {
                BUCKET_A!(c2) = k.offset_from(SA) as i32;
                c2 = c0;
                k = SA.offset(BUCKET_A!(c2) as isize);
            }
            if (0 < s) && ((*T.offset((s - 1) as isize) as i32) < c0) {
                if (s & mod_) == 0 {
                    *indexes.offset((s / (mod_ + 1) - 1) as isize) = k.offset_from(SA) as i32;
                }
                *k = !(*T.offset((s - 1) as isize) as i32);
                k = k.offset(1);
            } else {
                *k = s;
                k = k.offset(1);
            }
        } else if s != 0 {
            *i = !s;
        } else {
            orig = i;
        }
        i = i.offset(1);
    }

    orig.offset_from(SA) as i32
}

/* ------------------------------------------------------------------------ */

/// Port of `divsufsort`. Constructs the suffix array of `T[0..n]` into
/// `SA[0..n]`. Returns 0 on success, -1 on bad arguments. (`openMP` is
/// dropped — not built in zstd.)
pub fn divsufsort(T: &[u8], SA: &mut [i32], n: i32) -> i32 {
    /* Check arguments. */
    if n < 0 {
        return -1;
    } else if n == 0 {
        return 0;
    } else if n == 1 {
        SA[0] = 0;
        return 0;
    } else if n == 2 {
        let m = (T[0] < T[1]) as usize;
        SA[m ^ 1] = 0;
        SA[m] = 1;
        return 0;
    }

    let mut bucket_A = vec![0i32; BUCKET_A_SIZE];
    let mut bucket_B = vec![0i32; BUCKET_B_SIZE];

    // SAFETY: T and SA have length >= n (caller contract); bucket arrays are
    // sized to the C constants. All internal pointer arithmetic stays within
    // these allocations, mirroring the C library.
    unsafe {
        let t = T.as_ptr();
        let sa = SA.as_mut_ptr();
        let ba = bucket_A.as_mut_ptr();
        let bb = bucket_B.as_mut_ptr();
        let m = sort_typeBstar(t, sa, ba, bb, n);
        construct_SA(t, sa, ba, bb, n, m);
    }
    0
}

/// Port of `divbwt`. Burrows-Wheeler transform. Raw-pointer signature
/// mirrors the C (NULL-able `A`/`num_indexes`/`indexes`). Returns the
/// primary index, or -1/-2 on error.
///
/// SAFETY: pointers must be valid for `n` elements (`A` for `n+1`), or null
/// where the C accepts null.
#[allow(clippy::too_many_arguments)]
pub unsafe fn divbwt(
    T: *const u8,
    U: *mut u8,
    A: *mut i32,
    n: i32,
    num_indexes: *mut u8,
    indexes: *mut i32,
) -> i32 {
    let mut pidx: i32;
    let mut i: i32;

    /* Check arguments. */
    if T.is_null() || U.is_null() || (n < 0) {
        return -1;
    } else if n <= 1 {
        if n == 1 {
            *U = *T;
        }
        return n;
    }

    // B = A, or allocate n+1 ints if A is null.
    let mut owned_B: Vec<i32>;
    let B: *mut i32 = if A.is_null() {
        owned_B = vec![0i32; (n + 1) as usize];
        owned_B.as_mut_ptr()
    } else {
        owned_B = Vec::new();
        A
    };
    let mut bucket_A = vec![0i32; BUCKET_A_SIZE];
    let mut bucket_B = vec![0i32; BUCKET_B_SIZE];

    /* Burrows-Wheeler Transform. */
    {
        let ba = bucket_A.as_mut_ptr();
        let bb = bucket_B.as_mut_ptr();
        let m = sort_typeBstar(T, B, ba, bb, n);

        if num_indexes.is_null() || indexes.is_null() {
            pidx = construct_BWT(T, B, ba, bb, n, m);
        } else {
            pidx = construct_BWT_indexes(T, B, ba, bb, n, m, num_indexes, indexes);
        }

        /* Copy to output string. */
        *U = *T.offset((n - 1) as isize);
        i = 0;
        while i < pidx {
            *U.offset((i + 1) as isize) = *B.offset(i as isize) as u8;
            i += 1;
        }
        i += 1;
        while i < n {
            *U.offset(i as isize) = *B.offset(i as isize) as u8;
            i += 1;
        }
        pidx += 1;
    }

    let _ = &mut owned_B; // dropped here (frees B when A was null)
    pidx
}
