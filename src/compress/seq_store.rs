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
#[derive(Debug, Clone)]
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

/// Port of `ZSTD_deriveSeqStoreChunk` (`zstd_compress.c:4025`). Builds
/// a new `SeqStore_t` that represents the sequence subrange
/// `[startIdx, endIdx)` of `originalSeqStore`, used by the block
/// splitter to cost-estimate potential splits.
///
/// Upstream reuses the allocation via pointer arithmetic into
/// `sequencesStart`/`litStart`/`llCode`/`mlCode`/`ofCode`. Our port
/// clones the relevant subranges into fresh `Vec`s — correct but
/// allocates; the block splitter isn't yet wired into our compressor
/// so the allocation cost is moot for now.
///
/// `longLengthPos` is rebased or cleared if it falls outside the
/// chunk range, mirroring upstream exactly.
pub fn ZSTD_deriveSeqStoreChunk(
    originalSeqStore: &SeqStore_t,
    startIdx: usize,
    endIdx: usize,
) -> SeqStore_t {
    debug_assert!(endIdx <= originalSeqStore.sequences.len());
    debug_assert!(startIdx <= endIdx);

    let seqs_chunk: Vec<SeqDef> = originalSeqStore.sequences[startIdx..endIdx].to_vec();

    // Sum literals bytes before startIdx (consumed by prior chunks)
    // and within [startIdx, endIdx) (kept for this chunk).
    let prior_lits: usize = originalSeqStore.sequences[..startIdx]
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let mut n = s.litLength as usize;
            if i as u32 == originalSeqStore.longLengthPos
                && originalSeqStore.longLengthType == ZSTD_longLengthType_e::ZSTD_llt_literalLength
            {
                n += 0x10000;
            }
            n
        })
        .sum();

    let chunk_lits: usize = seqs_chunk
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let mut n = s.litLength as usize;
            let orig_idx = startIdx + i;
            if orig_idx as u32 == originalSeqStore.longLengthPos
                && originalSeqStore.longLengthType == ZSTD_longLengthType_e::ZSTD_llt_literalLength
            {
                n += 0x10000;
            }
            n
        })
        .sum();

    let is_last_chunk = endIdx == originalSeqStore.sequences.len();
    let lit_take = if is_last_chunk {
        // Take everything from prior_lits through end (includes trailing literals).
        originalSeqStore.literals.len().saturating_sub(prior_lits)
    } else {
        chunk_lits
    };
    let literals_chunk: Vec<u8> =
        originalSeqStore.literals[prior_lits..prior_lits + lit_take].to_vec();

    // Carry per-sequence code-tables if they've been materialized.
    let ll_chunk = if originalSeqStore.llCode.len() >= endIdx {
        originalSeqStore.llCode[startIdx..endIdx].to_vec()
    } else {
        Vec::new()
    };
    let ml_chunk = if originalSeqStore.mlCode.len() >= endIdx {
        originalSeqStore.mlCode[startIdx..endIdx].to_vec()
    } else {
        Vec::new()
    };
    let of_chunk = if originalSeqStore.ofCode.len() >= endIdx {
        originalSeqStore.ofCode[startIdx..endIdx].to_vec()
    } else {
        Vec::new()
    };

    let (longLengthType, longLengthPos) = match originalSeqStore.longLengthType {
        ZSTD_longLengthType_e::ZSTD_llt_none => (ZSTD_longLengthType_e::ZSTD_llt_none, 0),
        _ => {
            let p = originalSeqStore.longLengthPos as usize;
            if p < startIdx || p >= endIdx {
                (ZSTD_longLengthType_e::ZSTD_llt_none, 0)
            } else {
                (originalSeqStore.longLengthType, (p - startIdx) as u32)
            }
        }
    };

    SeqStore_t {
        sequences: seqs_chunk,
        literals: literals_chunk,
        llCode: ll_chunk,
        mlCode: ml_chunk,
        ofCode: of_chunk,
        maxNbSeq: originalSeqStore.maxNbSeq,
        maxNbLit: originalSeqStore.maxNbLit,
        longLengthType,
        longLengthPos,
    }
}

/// Port of `ZSTD_countSeqStoreLiteralsBytes` (`zstd_compress.c:3993`).
/// Sum of all sequences' literal-length fields, plus 0x10000 when the
/// long-length flag marks a literal-length overflow.
pub fn ZSTD_countSeqStoreLiteralsBytes(seqStore: &SeqStore_t) -> usize {
    let mut literalsBytes: usize = 0;
    for (i, seq) in seqStore.sequences.iter().enumerate() {
        literalsBytes += seq.litLength as usize;
        if i as u32 == seqStore.longLengthPos
            && seqStore.longLengthType == ZSTD_longLengthType_e::ZSTD_llt_literalLength
        {
            literalsBytes += 0x10000;
        }
    }
    literalsBytes
}

/// Port of `ZSTD_countSeqStoreMatchBytes` (`zstd_compress.c:4008`). Sum
/// of decoded match lengths (`mlBase + MINMATCH`) across the seqStore,
/// plus 0x10000 when the long-length flag marks a match-length overflow.
pub fn ZSTD_countSeqStoreMatchBytes(seqStore: &SeqStore_t) -> usize {
    let mut matchBytes: usize = 0;
    for (i, seq) in seqStore.sequences.iter().enumerate() {
        matchBytes += seq.mlBase as usize + MINMATCH as usize;
        if i as u32 == seqStore.longLengthPos
            && seqStore.longLengthType == ZSTD_longLengthType_e::ZSTD_llt_matchLength
        {
            matchBytes += 0x10000;
        }
    }
    matchBytes
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
        debug_assert_eq!(
            seqStore.longLengthType,
            ZSTD_longLengthType_e::ZSTD_llt_none
        );
        seqStore.longLengthType = ZSTD_longLengthType_e::ZSTD_llt_literalLength;
        seqStore.longLengthPos = seqStore.sequences.len() as u32;
    }
    let mlBase = matchLength - MINMATCH as usize;
    if mlBase > 0xFFFF {
        debug_assert_eq!(
            seqStore.longLengthType,
            ZSTD_longLengthType_e::ZSTD_llt_none
        );
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

/// Port of `ZSTD_resolveRepcodeToRawOffset` (`zstd_compress.c:4061`).
/// Given a repcode-flavored offBase, literal-length-zero flag, and the
/// current `rep[]`, returns the raw offset bytes the repcode resolves
/// to. The `(repCode == ZSTD_REP_NUM) → rep[0] - 1` corner case is
/// preserved for symmetry with `ZSTD_updateRep`; the caller must handle
/// the edge case where `rep[0] == 1` produces a zero offset (invalid,
/// discarded downstream by `ZSTD_seqStore_resolveOffCodes`).
pub fn ZSTD_resolveRepcodeToRawOffset(rep: &[u32; ZSTD_REP_NUM], offBase: u32, ll0: u32) -> u32 {
    debug_assert!(OFFBASE_IS_REPCODE(offBase));
    let adjustedRepCode = OFFBASE_TO_REPCODE(offBase) - 1 + ll0;
    if adjustedRepCode == ZSTD_REP_NUM as u32 {
        debug_assert!(ll0 != 0);
        rep[0].wrapping_sub(1)
    } else {
        rep[adjustedRepCode as usize]
    }
}

/// Port of `ZSTD_validateSeqStore` (`zstd_compress.c:3261`). Debug-
/// only assertion that every sequence's decoded matchLength respects
/// the minimum dictated by `cParams.minMatch` (3 when minMatch=3, else
/// 4). In release builds this is a no-op; in debug builds it panics
/// on violation.
pub fn ZSTD_validateSeqStore(seqStore: &SeqStore_t, minMatchParam: u32) {
    #[cfg(debug_assertions)]
    {
        let matchLenLowerBound: u32 = if minMatchParam == 3 { 3 } else { 4 };
        for idx in 0..seqStore.sequences.len() {
            let seqLen = ZSTD_getSequenceLength(seqStore, idx);
            debug_assert!(seqLen.matchLength >= matchLenLowerBound);
        }
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = (seqStore, minMatchParam);
    }
}

/// Port of `ZSTD_seqStore_resolveOffCodes` (`zstd_compress.c:4093`).
/// Walks `seqStore.sequences` in order, maintaining parallel
/// decompression-side (`dRepcodes`) and compression-side (`cRepcodes`)
/// repcode histories. Whenever a repcode-flavored `offBase` would
/// resolve to different raw offsets on the two sides — which can
/// happen after an RLE/raw-block emission perturbs the implicit
/// decoder history — the sequence's `offBase` is rewritten to
/// explicitly carry the compression-side raw offset via
/// `OFFSET_TO_OFFBASE`, so the decoder's rep history matches.
///
/// Post-condition: both rep histories are advanced — `dRepcodes` via
/// the (possibly rewritten) `seq.offBase` (the value the decoder will
/// see), `cRepcodes` via the original `offBase` (the compressor's
/// actual choice).
pub fn ZSTD_seqStore_resolveOffCodes(
    dRepcodes: &mut Repcodes_t,
    cRepcodes: &mut Repcodes_t,
    seqStore: &mut SeqStore_t,
    nbSeq: u32,
) {
    let longLitLenIdx = if seqStore.longLengthType == ZSTD_longLengthType_e::ZSTD_llt_literalLength
    {
        seqStore.longLengthPos
    } else {
        nbSeq
    };
    for idx in 0..(nbSeq as usize) {
        let seq = &mut seqStore.sequences[idx];
        let ll0 = (seq.litLength == 0 && idx as u32 != longLitLenIdx) as u32;
        let offBase = seq.offBase;
        debug_assert!(offBase > 0);
        if OFFBASE_IS_REPCODE(offBase) {
            let dRaw = ZSTD_resolveRepcodeToRawOffset(&dRepcodes.rep, offBase, ll0);
            let cRaw = ZSTD_resolveRepcodeToRawOffset(&cRepcodes.rep, offBase, ll0);
            if dRaw != cRaw {
                seq.offBase = OFFSET_TO_OFFBASE(cRaw);
            }
        }
        let new_offBase = seq.offBase;
        ZSTD_updateRep(&mut dRepcodes.rep, new_offBase, ll0);
        ZSTD_updateRep(&mut cRepcodes.rep, offBase, ll0);
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
    fn ZSTD_longLengthType_e_discriminants_match_upstream() {
        // Upstream: `typedef enum { ZSTD_llt_none, ZSTD_llt_literalLength,
        // ZSTD_llt_matchLength } ZSTD_longLengthType_e;` — default
        // sequential 0/1/2. `ZSTD_seqToCodes` stamps a single sequence's
        // long-length flag into this u32 field; drift would mis-route
        // the "which length exceeds the FSE code max" discriminator.
        assert_eq!(ZSTD_longLengthType_e::ZSTD_llt_none as u32, 0);
        assert_eq!(ZSTD_longLengthType_e::ZSTD_llt_literalLength as u32, 1);
        assert_eq!(ZSTD_longLengthType_e::ZSTD_llt_matchLength as u32, 2);
    }

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
        assert_eq!(s0.litLength, 3); // untouched
        assert_eq!(s1.litLength, 2 + 0x10000); // long
    }

    #[test]
    fn deriveSeqStoreChunk_excludes_long_length_at_end_boundary() {
        let mut ss = SeqStore_t::with_capacity(8, 128);
        ZSTD_storeSeqOnly(&mut ss, 3, OFFSET_TO_OFFBASE(10), 7);
        ZSTD_storeSeqOnly(&mut ss, 4, OFFSET_TO_OFFBASE(11), 8);
        ZSTD_storeSeqOnly(&mut ss, 5, OFFSET_TO_OFFBASE(12), 9);
        ss.literals.extend_from_slice(&[b'a'; 12]);
        ss.longLengthType = ZSTD_longLengthType_e::ZSTD_llt_matchLength;
        ss.longLengthPos = 2;

        let chunk = ZSTD_deriveSeqStoreChunk(&ss, 0, 2);
        assert_eq!(chunk.sequences.len(), 2);
        assert_eq!(chunk.longLengthType, ZSTD_longLengthType_e::ZSTD_llt_none);
        assert_eq!(chunk.longLengthPos, 0);
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
        assert_eq!(
            ss.longLengthType,
            ZSTD_longLengthType_e::ZSTD_llt_literalLength
        );
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

    #[test]
    fn update_rep_uses_rep0_minus_one_when_repCode_equals_ZSTD_REP_NUM() {
        // Special case: repcode 3 with ll0=1 → repCode = 3+1-1 = 3 (=
        // ZSTD_REP_NUM). Upstream uses `rep[0] - 1` as the new offset
        // in this branch (the "virtual 4th repcode" slot). Pin this
        // corner so a future refactor can't swap it with rep[2].
        let mut rep = [5u32, 10, 20];
        ZSTD_updateRep(&mut rep, REPCODE_TO_OFFBASE(3), 1);
        assert_eq!(rep, [4, 5, 10]); // new rep[0] = 5-1 = 4
    }

    #[test]
    fn newRep_is_non_destructive_to_input() {
        // `ZSTD_newRep` is the non-destructive sibling — returns a
        // fresh Repcodes_t without mutating the caller's array.
        let rep = [5u32, 10, 20];
        let result = ZSTD_newRep(&rep, OFFSET_TO_OFFBASE(42), 0);
        assert_eq!(rep, [5, 10, 20], "input must not be mutated");
        assert_eq!(result.rep, [42, 5, 10]);
    }
}
