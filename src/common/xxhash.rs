//! Translation of `lib/common/xxhash.{h,c}` — the subset of XXH64 that
//! zstd uses for frame checksums.
//!
//! The upstream header is ~7k lines because it also ships XXH32, XXH128
//! and SIMD paths. zstd only calls `XXH64` (oneshot) and the stateful
//! `XXH64_{reset,update,digest}` APIs on the frame-checksum path, so we
//! keep the translation tight to that subset. Reference vectors are
//! verified against the canonical XXH64 algorithm.

const PRIME64_1: u64 = 0x9E3779B185EBCA87;
const PRIME64_2: u64 = 0xC2B2AE3D27D4EB4F;
const PRIME64_3: u64 = 0x165667B19E3779F9;
const PRIME64_4: u64 = 0x85EBCA77C2B2AE63;
const PRIME64_5: u64 = 0x27D4EB2F165667C5;

#[inline]
fn round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(PRIME64_2))
        .rotate_left(31)
        .wrapping_mul(PRIME64_1)
}

#[inline]
fn merge_round(acc: u64, val: u64) -> u64 {
    let val = round(0, val);
    (acc ^ val).wrapping_mul(PRIME64_1).wrapping_add(PRIME64_4)
}

#[inline]
fn avalanche(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(PRIME64_2);
    h ^= h >> 29;
    h = h.wrapping_mul(PRIME64_3);
    h ^= h >> 32;
    h
}

#[inline]
fn finalize(mut h: u64, mut bytes: &[u8]) -> u64 {
    while bytes.len() >= 8 {
        let k1 = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        h ^= round(0, k1);
        h = h.rotate_left(27).wrapping_mul(PRIME64_1).wrapping_add(PRIME64_4);
        bytes = &bytes[8..];
    }
    if bytes.len() >= 4 {
        let k1 = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as u64;
        h ^= k1.wrapping_mul(PRIME64_1);
        h = h.rotate_left(23).wrapping_mul(PRIME64_2).wrapping_add(PRIME64_3);
        bytes = &bytes[4..];
    }
    for &b in bytes {
        h ^= (b as u64).wrapping_mul(PRIME64_5);
        h = h.rotate_left(11).wrapping_mul(PRIME64_1);
    }
    avalanche(h)
}

/// Oneshot hash (`XXH64(const void*, size_t, XXH64_hash_t)`).
pub fn XXH64(input: &[u8], seed: u64) -> u64 {
    let len = input.len() as u64;
    if input.len() >= 32 {
        let mut v1 = seed.wrapping_add(PRIME64_1).wrapping_add(PRIME64_2);
        let mut v2 = seed.wrapping_add(PRIME64_2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(PRIME64_1);

        let mut p = 0usize;
        let limit = input.len() - 32;
        while p <= limit {
            v1 = round(v1, u64::from_le_bytes(input[p..p + 8].try_into().unwrap()));
            v2 = round(v2, u64::from_le_bytes(input[p + 8..p + 16].try_into().unwrap()));
            v3 = round(v3, u64::from_le_bytes(input[p + 16..p + 24].try_into().unwrap()));
            v4 = round(v4, u64::from_le_bytes(input[p + 24..p + 32].try_into().unwrap()));
            p += 32;
        }

        let mut h = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));
        h = merge_round(h, v1);
        h = merge_round(h, v2);
        h = merge_round(h, v3);
        h = merge_round(h, v4);

        let tail = &input[p..];
        finalize(h.wrapping_add(len), tail)
    } else {
        finalize(seed.wrapping_add(PRIME64_5).wrapping_add(len), input)
    }
}

// ---- Stateful API -------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct XXH64_state_t {
    pub total_len: u64,
    pub v1: u64,
    pub v2: u64,
    pub v3: u64,
    pub v4: u64,
    pub mem64: [u8; 32],
    pub memsize: u32,
}

/// `XXH_OK` from upstream.
pub const XXH_OK: u32 = 0;

pub fn XXH64_reset(state: &mut XXH64_state_t, seed: u64) -> u32 {
    *state = XXH64_state_t {
        total_len: 0,
        v1: seed.wrapping_add(PRIME64_1).wrapping_add(PRIME64_2),
        v2: seed.wrapping_add(PRIME64_2),
        v3: seed,
        v4: seed.wrapping_sub(PRIME64_1),
        mem64: [0; 32],
        memsize: 0,
    };
    XXH_OK
}

pub fn XXH64_update(state: &mut XXH64_state_t, mut input: &[u8]) -> u32 {
    state.total_len = state.total_len.wrapping_add(input.len() as u64);

    if state.memsize as usize + input.len() < 32 {
        let off = state.memsize as usize;
        state.mem64[off..off + input.len()].copy_from_slice(input);
        state.memsize += input.len() as u32;
        return XXH_OK;
    }

    if state.memsize != 0 {
        let off = state.memsize as usize;
        let need = 32 - off;
        state.mem64[off..off + need].copy_from_slice(&input[..need]);
        state.v1 = round(state.v1, u64::from_le_bytes(state.mem64[0..8].try_into().unwrap()));
        state.v2 = round(state.v2, u64::from_le_bytes(state.mem64[8..16].try_into().unwrap()));
        state.v3 = round(state.v3, u64::from_le_bytes(state.mem64[16..24].try_into().unwrap()));
        state.v4 = round(state.v4, u64::from_le_bytes(state.mem64[24..32].try_into().unwrap()));
        input = &input[need..];
        state.memsize = 0;
    }

    while input.len() >= 32 {
        state.v1 = round(state.v1, u64::from_le_bytes(input[0..8].try_into().unwrap()));
        state.v2 = round(state.v2, u64::from_le_bytes(input[8..16].try_into().unwrap()));
        state.v3 = round(state.v3, u64::from_le_bytes(input[16..24].try_into().unwrap()));
        state.v4 = round(state.v4, u64::from_le_bytes(input[24..32].try_into().unwrap()));
        input = &input[32..];
    }

    if !input.is_empty() {
        state.mem64[..input.len()].copy_from_slice(input);
        state.memsize = input.len() as u32;
    }

    XXH_OK
}

pub fn XXH64_digest(state: &XXH64_state_t) -> u64 {
    let mut h = if state.total_len >= 32 {
        let mut h = state
            .v1
            .rotate_left(1)
            .wrapping_add(state.v2.rotate_left(7))
            .wrapping_add(state.v3.rotate_left(12))
            .wrapping_add(state.v4.rotate_left(18));
        h = merge_round(h, state.v1);
        h = merge_round(h, state.v2);
        h = merge_round(h, state.v3);
        h = merge_round(h, state.v4);
        h
    } else {
        state.v3.wrapping_add(PRIME64_5)
    };
    h = h.wrapping_add(state.total_len);
    finalize(h, &state.mem64[..state.memsize as usize])
}

#[cfg(test)]
mod tests {
    use super::*;

    // Canonical XXH64 reference vectors for the empty input across seeds.
    // Source: https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhsum.1
    // and the original reference implementation; cross-checked against
    // `openssl dgst`-style tooling.
    #[test]
    fn xxh64_empty_seed0() {
        assert_eq!(XXH64(&[], 0), 0xEF46DB3751D8E999);
    }

    #[test]
    fn xxh64_single_byte() {
        // `echo -n 'a' | xxh64sum` → 0xd24ec4f1a98c6e5b
        assert_eq!(XXH64(b"a", 0), 0xD24EC4F1A98C6E5B);
    }

    #[test]
    fn xxh64_nobody_inspects_short_input() {
        // Known vector: "Nobody inspects the spammish repetition" with seed 0.
        // This is from xxHash's canonical test set.
        let input = b"Nobody inspects the spammish repetition";
        assert_eq!(XXH64(input, 0), 0xFBCEA83C8A378BF1);
    }

    #[test]
    fn xxh64_with_seed() {
        // Canonical "abc" vector at seed 0.
        assert_eq!(XXH64(b"abc", 0), 0x44BC2CF5AD770999);
        // Different seeds must produce different hashes on the same input.
        let h0 = XXH64(b"abc", 0);
        let h1 = XXH64(b"abc", 1);
        let h_bad = XXH64(b"abc", 0xcafebabe);
        assert_ne!(h0, h1);
        assert_ne!(h0, h_bad);
        assert_ne!(h1, h_bad);
    }

    #[test]
    fn xxh64_large_input_roundtrips_state() {
        // Feed the same input through oneshot vs. stateful streaming in
        // odd-sized chunks, and verify digests agree across all >32-byte
        // paths.
        let data: Vec<u8> = (0..=255u8).cycle().take(1024).collect();
        let oneshot = XXH64(&data, 0xdeadbeef);

        let mut st = XXH64_state_t::default();
        XXH64_reset(&mut st, 0xdeadbeef);
        for chunk in data.chunks(7) {
            XXH64_update(&mut st, chunk);
        }
        assert_eq!(XXH64_digest(&st), oneshot);
    }

    #[test]
    fn xxh64_incremental_matches_oneshot_small() {
        let data = b"The quick brown fox jumps over the lazy dog";
        let oneshot = XXH64(data, 42);

        let mut st = XXH64_state_t::default();
        XXH64_reset(&mut st, 42);
        XXH64_update(&mut st, &data[..10]);
        XXH64_update(&mut st, &data[10..]);
        assert_eq!(XXH64_digest(&st), oneshot);
    }

    #[test]
    fn xxh64_single_byte_with_seed() {
        // Regression guard: the <32-byte path goes through avalanche.
        let h = XXH64(b"x", 0);
        let mut st = XXH64_state_t::default();
        XXH64_reset(&mut st, 0);
        XXH64_update(&mut st, b"x");
        assert_eq!(XXH64_digest(&st), h);
    }

    #[test]
    fn xxh64_reset_clobbers_prior_state() {
        // Feed data through one state, then reset with a different
        // seed and feed different data. The resulting digest must
        // match a fresh state + same-seed + same-data — no stale
        // bytes should leak through the reset.
        let mut st = XXH64_state_t::default();
        XXH64_reset(&mut st, 0);
        XXH64_update(&mut st, b"discard-this-prior-input-that-should-not-affect-next-digest");

        XXH64_reset(&mut st, 42);
        XXH64_update(&mut st, b"fresh");
        let after_reset = XXH64_digest(&st);

        let oneshot = XXH64(b"fresh", 42);
        assert_eq!(after_reset, oneshot);
    }
}
