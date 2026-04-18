//! Port of `SeqStore_t` and the storeSeq helpers from
//! `lib/compress/zstd_compress_internal.h`. This is the buffer the
//! match finder writes into — one `SeqDef` per found match, with the
//! literals packed contiguously in a side array.

#![allow(non_snake_case)]

pub use crate::decompress::zstd_decompress_block::ZSTD_REP_NUM;

/// Upstream MINMATCH.
pub const MINMATCH: u32 = 3;

/// Port of `ZSTD_SequenceLength`. Fully-decoded lit/match lengths
/// returned by `ZSTD_getSequenceLength` after the long-length bit is
/// resolved and `MINMATCH` is added back to matchLength.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ZSTD_SequenceLength {
    pub litLength: u32,
    pub matchLength: u32,
}

/// Port of `SeqDef`. Compact encoded sequence entry:
///   - `offBase`: if > `ZSTD_REP_NUM`, a full offset shifted
///     up by `ZSTD_REP_NUM`. Otherwise a 1/2/3 repcode id.
///   - `litLength`: literal byte count (u16; if it would overflow,
///     the parent `SeqStore_t` sets `longLengthType` + pos).
///   - `mlBase`: matchLength minus `MINMATCH` (u16; same overflow rule).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SeqDef {
    pub offBase: u32,
    pub litLength: u16,
    pub mlBase: u16,
}

/// Port of `ZSTD_longLengthType_e`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZSTD_longLengthType_e {
    #[default]
    ZSTD_llt_none,
    ZSTD_llt_literalLength,
    ZSTD_llt_matchLength,
}

/// Port of `SeqStore_t`. Upstream uses owned raw pointers into
/// caller-provided buffers — we keep the buffers inline (owned
/// `Vec`s) plus current-end indices that behave like the upstream
/// `sequences` / `lit` cursors.
#[derive(Debug)]
pub struct SeqStore_t {
    pub sequences: Vec<SeqDef>,
    pub literals: Vec<u8>,
    pub llCode: Vec<u8>,
    pub mlCode: Vec<u8>,
    pub ofCode: Vec<u8>,
    pub maxNbSeq: usize,
    pub maxNbLit: usize,
    pub longLengthType: ZSTD_longLengthType_e,
    pub longLengthPos: u32,
}

impl SeqStore_t {
    /// Create an empty store with upstream's typical capacities
    /// (maxNbSeq = `ZSTD_BLOCKSIZE_MAX / 3`, maxNbLit = `128 KB`).
    pub fn with_capacity(maxNbSeq: usize, maxNbLit: usize) -> Self {
        Self {
            sequences: Vec::with_capacity(maxNbSeq),
            literals: Vec::with_capacity(maxNbLit),
            llCode: Vec::with_capacity(maxNbSeq),
            mlCode: Vec::with_capacity(maxNbSeq),
            ofCode: Vec::with_capacity(maxNbSeq),
            maxNbSeq,
            maxNbLit,
            longLengthType: ZSTD_longLengthType_e::ZSTD_llt_none,
            longLengthPos: 0,
        }
    }

    pub fn reset(&mut self) {
        self.sequences.clear();
        self.literals.clear();
        self.llCode.clear();
        self.mlCode.clear();
        self.ofCode.clear();
        self.longLengthType = ZSTD_longLengthType_e::ZSTD_llt_none;
        self.longLengthPos = 0;
    }
}

/// Port of `ZSTD_storeLastLiterals`. Appends a trailing-literals run
/// to the seq store — used at the end of a block when the matcher
/// stops before the block boundary.
#[inline]
pub fn ZSTD_storeLastLiterals(ss: &mut SeqStore_t, anchor: &[u8]) {
    ss.literals.extend_from_slice(anchor);
}

/// Port of `ZSTD_resetSeqStore`. Clears the sequence + literals
/// cursors so the next block can reuse the allocation.
#[inline]
pub fn ZSTD_resetSeqStore(ss: &mut SeqStore_t) {
    ss.sequences.clear();
    ss.literals.clear();
    ss.longLengthType = ZSTD_longLengthType_e::ZSTD_llt_none;
}

/// Port of `ZSTD_getSequenceLength`. Decodes the long-length bit and
/// adds `MINMATCH` back to matchLength, returning both lengths.
///
/// `seqIdx` is the 0-based index of the sequence within
/// `seqStore.sequences` — upstream passes a `SeqDef*`, but since Rust
/// doesn't allow &SeqStore + &SeqDef simultaneously, we pass the
/// index and look up both pieces via the store.
pub fn ZSTD_getSequenceLength(seqStore: &SeqStore_t, seqIdx: usize) -> ZSTD_SequenceLength {
    let seq = seqStore.sequences[seqIdx];
    let mut seqLen = ZSTD_SequenceLength {
        litLength: seq.litLength as u32,
        matchLength: seq.mlBase as u32 + MINMATCH,
    };
    if seqStore.longLengthPos == seqIdx as u32 {
        match seqStore.longLengthType {
            ZSTD_longLengthType_e::ZSTD_llt_literalLength => seqLen.litLength += 0x10000,
            ZSTD_longLengthType_e::ZSTD_llt_matchLength => seqLen.matchLength += 0x10000,
            ZSTD_longLengthType_e::ZSTD_llt_none => {}
        }
    }
    seqLen
}

// ---- offBase helpers (match upstream's macros) ---------------------------

#[inline]
pub fn REPCODE_TO_OFFBASE(r: u32) -> u32 {
    debug_assert!(r >= 1 && r <= ZSTD_REP_NUM as u32);
    r
}

#[inline]
pub fn OFFSET_TO_OFFBASE(offset: u32) -> u32 {
    debug_assert!(offset > 0);
    offset + ZSTD_REP_NUM as u32
}

#[inline]
pub fn OFFBASE_IS_OFFSET(o: u32) -> bool {
    o > ZSTD_REP_NUM as u32
}

#[inline]
pub fn OFFBASE_IS_REPCODE(o: u32) -> bool {
    o <= ZSTD_REP_NUM as u32
}

#[inline]
pub fn OFFBASE_TO_OFFSET(o: u32) -> u32 {
    debug_assert!(OFFBASE_IS_OFFSET(o));
    o - ZSTD_REP_NUM as u32
}

#[inline]
pub fn OFFBASE_TO_REPCODE(o: u32) -> u32 {
    debug_assert!(OFFBASE_IS_REPCODE(o));
    o
}

/// Port of `ZSTD_storeSeqOnly`. Appends a `SeqDef` to the store
/// without copying literal bytes. Long litLength / mlBase are
/// flagged by setting `longLengthType` + `longLengthPos`.
pub fn ZSTD_storeSeqOnly(
    seqStore: &mut SeqStore_t,
    litLength: usize,
    offBase: u32,
    matchLength: usize,
) {
    debug_assert!(seqStore.sequences.len() < seqStore.maxNbSeq);
    debug_assert!(matchLength >= MINMATCH as usize);

    // litLength may exceed U16; flag as long.
    if litLength > 0xFFFF {
        debug_assert_eq!(seqStore.longLengthType, ZSTD_longLengthType_e::ZSTD_llt_none);
        seqStore.longLengthType = ZSTD_longLengthType_e::ZSTD_llt_literalLength;
        seqStore.longLengthPos = seqStore.sequences.len() as u32;
    }
    let mlBase = matchLength - MINMATCH as usize;
    if mlBase > 0xFFFF {
        debug_assert_eq!(seqStore.longLengthType, ZSTD_longLengthType_e::ZSTD_llt_none);
        seqStore.longLengthType = ZSTD_longLengthType_e::ZSTD_llt_matchLength;
        seqStore.longLengthPos = seqStore.sequences.len() as u32;
    }
    seqStore.sequences.push(SeqDef {
        offBase,
        litLength: litLength as u16,
        mlBase: mlBase as u16,
    });
}

/// Port of `ZSTD_storeSeq`. Writes a full sequence — appends
/// `litLength` bytes from `literals` into the store's literal
/// buffer, then calls `ZSTD_storeSeqOnly`.
///
/// Rust signature note: upstream takes raw `BYTE*` into the source
/// buffer plus a `litLimit` guard. The Rust port takes `literals:
/// &[u8]` whose length is the upper bound — upstream's wildcopy
/// over-read optimization isn't portable without more ceremony, so
/// we `copy_from_slice` (one memcpy) and accept the small perf loss.
pub fn ZSTD_storeSeq(
    seqStore: &mut SeqStore_t,
    litLength: usize,
    literals: &[u8],
    offBase: u32,
    matchLength: usize,
) {
    debug_assert!(litLength <= literals.len());
    debug_assert!(seqStore.literals.len() + litLength <= seqStore.maxNbLit);
    seqStore.literals.extend_from_slice(&literals[..litLength]);
    ZSTD_storeSeqOnly(seqStore, litLength, offBase, matchLength);
}

/// Port of `Repcodes_t`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Repcodes_t {
    pub rep: [u32; ZSTD_REP_NUM],
}

/// Port of `ZSTD_updateRep`. Updates `rep\[\]` in place following
/// upstream's sumtype:
///   - `OFFBASE_IS_OFFSET`: shift rep\[1\]→\[2\], \[0\]→\[1\], store
///     new offset at \[0\].
///   - Otherwise (repcode 1..=3): adjust by `ll0` (literal length == 0 signal)
///     and promote the selected old repcode.
pub fn ZSTD_updateRep(rep: &mut [u32; ZSTD_REP_NUM], offBase: u32, ll0: u32) {
    if OFFBASE_IS_OFFSET(offBase) {
        rep[2] = rep[1];
        rep[1] = rep[0];
        rep[0] = OFFBASE_TO_OFFSET(offBase);
    } else {
        let repCode = OFFBASE_TO_REPCODE(offBase) + ll0 - 1;
        if repCode > 0 {
            let currentOffset = if repCode == ZSTD_REP_NUM as u32 {
                rep[0] - 1
            } else {
                rep[repCode as usize]
            };
            if repCode >= 2 {
                rep[2] = rep[1];
            }
            rep[1] = rep[0];
            rep[0] = currentOffset;
        }
    }
}

/// Port of `ZSTD_newRep`. Non-destructive variant of
/// `ZSTD_updateRep`.
pub fn ZSTD_newRep(rep: &[u32; ZSTD_REP_NUM], offBase: u32, ll0: u32) -> Repcodes_t {
    let mut newReps = Repcodes_t { rep: *rep };
    ZSTD_updateRep(&mut newReps.rep, offBase, ll0);
    newReps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storeLastLiterals_appends() {
        let mut ss = SeqStore_t::with_capacity(16, 128);
        ss.literals.extend_from_slice(b"head");
        ZSTD_storeLastLiterals(&mut ss, b"tail");
        assert_eq!(ss.literals, b"headtail");
    }

    #[test]
    fn resetSeqStore_clears_cursors() {
        let mut ss = SeqStore_t::with_capacity(16, 128);
        ss.literals.extend_from_slice(b"junk");
        ZSTD_storeSeqOnly(&mut ss, 5, OFFSET_TO_OFFBASE(17), 4);
        ss.longLengthType = ZSTD_longLengthType_e::ZSTD_llt_matchLength;
        ZSTD_resetSeqStore(&mut ss);
        assert!(ss.sequences.is_empty());
        assert!(ss.literals.is_empty());
        assert_eq!(ss.longLengthType, ZSTD_longLengthType_e::ZSTD_llt_none);
    }

    #[test]
    fn getSequenceLength_adds_minmatch_back() {
        let mut ss = SeqStore_t::with_capacity(8, 128);
        ZSTD_storeSeqOnly(&mut ss, 5, OFFSET_TO_OFFBASE(17), 10);
        let got = ZSTD_getSequenceLength(&ss, 0);
        assert_eq!(got.litLength, 5);
        // mlBase = 10 - MINMATCH = 7; decoded = 7 + MINMATCH = 10.
        assert_eq!(got.matchLength, 10);
    }

    #[test]
    fn getSequenceLength_resolves_long_litLength() {
        let mut ss = SeqStore_t::with_capacity(8, 128);
        ZSTD_storeSeqOnly(&mut ss, 5, OFFSET_TO_OFFBASE(17), 8);
        ss.longLengthType = ZSTD_longLengthType_e::ZSTD_llt_literalLength;
        ss.longLengthPos = 0;
        let got = ZSTD_getSequenceLength(&ss, 0);
        // 5 + 0x10000 = 65541.
        assert_eq!(got.litLength, 65541);
        assert_eq!(got.matchLength, 8);
    }

    #[test]
    fn getSequenceLength_resolves_long_matchLength() {
        let mut ss = SeqStore_t::with_capacity(8, 128);
        ZSTD_storeSeqOnly(&mut ss, 4, OFFSET_TO_OFFBASE(17), 10);
        ss.longLengthType = ZSTD_longLengthType_e::ZSTD_llt_matchLength;
        ss.longLengthPos = 0;
        let got = ZSTD_getSequenceLength(&ss, 0);
        assert_eq!(got.litLength, 4);
        // decoded = 10 + 0x10000 = 65546.
        assert_eq!(got.matchLength, 10 + 0x10000);
    }

    #[test]
    fn getSequenceLength_long_pos_only_applies_to_target_seq() {
        let mut ss = SeqStore_t::with_capacity(8, 128);
        ZSTD_storeSeqOnly(&mut ss, 3, OFFSET_TO_OFFBASE(17), 8);
        ZSTD_storeSeqOnly(&mut ss, 2, OFFSET_TO_OFFBASE(17), 10);
        ss.longLengthType = ZSTD_longLengthType_e::ZSTD_llt_literalLength;
        ss.longLengthPos = 1; // second seq
        let s0 = ZSTD_getSequenceLength(&ss, 0);
        let s1 = ZSTD_getSequenceLength(&ss, 1);
        assert_eq!(s0.litLength, 3);           // untouched
        assert_eq!(s1.litLength, 2 + 0x10000); // long
    }

    #[test]
    fn offbase_sumtype_roundtrips() {
        // Full offset: encode → is_offset → decode.
        let ob = OFFSET_TO_OFFBASE(42);
        assert!(OFFBASE_IS_OFFSET(ob));
        assert_eq!(OFFBASE_TO_OFFSET(ob), 42);

        // Repcode: 1, 2, 3 don't cross the ZSTD_REP_NUM threshold.
        for r in 1..=3u32 {
            let ob = REPCODE_TO_OFFBASE(r);
            assert!(OFFBASE_IS_REPCODE(ob));
            assert_eq!(OFFBASE_TO_REPCODE(ob), r);
        }
    }

    #[test]
    fn store_seq_only_appends() {
        let mut ss = SeqStore_t::with_capacity(16, 1024);
        ZSTD_storeSeqOnly(&mut ss, 5, OFFSET_TO_OFFBASE(17), 4);
        assert_eq!(ss.sequences.len(), 1);
        let s = ss.sequences[0];
        assert_eq!(s.litLength, 5);
        assert!(OFFBASE_IS_OFFSET(s.offBase));
        assert_eq!(OFFBASE_TO_OFFSET(s.offBase), 17);
        assert_eq!(s.mlBase, 4 - MINMATCH as u16);
    }

    #[test]
    fn store_seq_appends_literals_and_bumps_sequences() {
        let mut ss = SeqStore_t::with_capacity(16, 1024);
        let src = b"hello matchdata";
        ZSTD_storeSeq(&mut ss, 5, src, REPCODE_TO_OFFBASE(1), 3);
        assert_eq!(ss.literals, b"hello");
        assert_eq!(ss.sequences.len(), 1);
        assert_eq!(ss.sequences[0].litLength, 5);
        // MINMATCH=3 → mlBase = 0.
        assert_eq!(ss.sequences[0].mlBase, 0);
    }

    #[test]
    fn store_seq_long_litlength_sets_flag() {
        let mut ss = SeqStore_t::with_capacity(16, 200_000);
        let lit = vec![b'x'; 0x10001];
        ZSTD_storeSeq(&mut ss, 0x10001, &lit, OFFSET_TO_OFFBASE(100), 5);
        assert_eq!(ss.longLengthType, ZSTD_longLengthType_e::ZSTD_llt_literalLength);
        assert_eq!(ss.longLengthPos, 0);
    }

    #[test]
    fn update_rep_full_offset_shifts_history() {
        let mut rep = [1u32, 4, 8];
        ZSTD_updateRep(&mut rep, OFFSET_TO_OFFBASE(42), 0);
        assert_eq!(rep, [42, 1, 4]);
    }

    #[test]
    fn update_rep_repcode_1_noop() {
        // Repcode 1 with ll0=0 → repCode = 1 + 0 - 1 = 0 → no change.
        let rep_before = [5u32, 10, 20];
        let mut rep = rep_before;
        ZSTD_updateRep(&mut rep, REPCODE_TO_OFFBASE(1), 0);
        assert_eq!(rep, rep_before);
    }

    #[test]
    fn update_rep_repcode_2_promotes() {
        // Repcode 2 with ll0=0 → repCode = 2 + 0 - 1 = 1 → pick rep[1],
        // shift rep[0]→rep[1].
        let mut rep = [5u32, 10, 20];
        ZSTD_updateRep(&mut rep, REPCODE_TO_OFFBASE(2), 0);
        assert_eq!(rep, [10, 5, 20]);
    }
}
