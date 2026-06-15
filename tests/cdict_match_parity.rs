//! Regression test for a CDict dictionary-indexing bug.
//!
//! `ZSTD_initCDict_internal`'s rawContent path used to fill only the short
//! hash table (`ZSTD_fillHashTable`), which is correct for `ZSTD_fast` but
//! leaves the long-hash table (dfast) and chain/tree (lazy/btopt) empty. As a
//! result `ZSTD_compress_usingCDict` found far fewer dictionary matches than
//! the `ZSTD_CCtx_loadDictionary` path for every strategy above `fast` — the
//! same content compressed ~44% larger. This pins CDict compression to the
//! loadDictionary path so the two stay equivalent.
use zstd_pure_rs::prelude::*;

fn corpus() -> (Vec<u8>, Vec<Vec<u8>>) {
    // A dictionary of recurring phrases, and samples built from the same
    // phrases so dictionary matching has plenty to find.
    let phrases: [&[u8]; 4] = [
        b"the quick brown fox jumps over the lazy dog. ",
        b"pack my box with five dozen liquor jugs. ",
        b"how vexingly quick daft zebras jump! ",
        b"sphinx of black quartz, judge my vow. ",
    ];
    let mut dict = Vec::new();
    for i in 0..400u32 {
        dict.extend_from_slice(phrases[(i % 4) as usize]);
    }
    let mut samples = Vec::new();
    for i in 0..40u32 {
        let mut s = Vec::new();
        for j in 0..6u32 {
            s.extend_from_slice(phrases[((i + j) % 4) as usize]);
        }
        samples.push(s);
    }
    (dict, samples)
}

fn total_usingdict(content: &[u8], level: i32, samples: &[Vec<u8>]) -> usize {
    let mut tot = 0;
    for s in samples {
        let mut cctx = ZSTD_createCCtx().unwrap();
        let mut out = vec![0u8; ZSTD_compressBound(s.len())];
        // raw-content dict loaded into the CCtx (the path that always worked)
        let cs = ZSTD_compress_usingDict(&mut cctx, &mut out, s, content, level);
        assert!(!ERR_isError(cs));
        tot += cs;
    }
    tot
}

fn total_usingcdict(content: &[u8], level: i32, samples: &[Vec<u8>]) -> usize {
    let cp = ZSTD_getCParams(level, 0, content.len());
    let cd = ZSTD_createCDict_advanced(
        content,
        ZSTD_dictLoadMethod_e::ZSTD_dlm_byRef,
        ZSTD_dictContentType_e::ZSTD_dct_rawContent,
        cp,
    )
    .unwrap();
    let mut cctx = ZSTD_createCCtx().unwrap();
    let mut tot = 0;
    for s in samples {
        let mut out = vec![0u8; ZSTD_compressBound(s.len())];
        let cs = ZSTD_compress_usingCDict(&mut cctx, &mut out, s, &cd);
        assert!(!ERR_isError(cs));
        tot += cs;
    }
    tot
}

#[test]
fn cdict_matches_loaddict_across_strategies() {
    let (dict, samples) = corpus();
    // Levels spanning fast(1), dfast(3), lazy/btopt(higher).
    for level in [1, 3, 6, 12, 19] {
        let cp = ZSTD_getCParams(level, 0, dict.len());
        let a = total_usingdict(&dict, level, &samples);
        let b = total_usingcdict(&dict, level, &samples);
        // The CDict path must not be materially worse than loadDictionary.
        // Pre-fix it was ~44% larger for every strategy above `fast`.
        assert!(
            b <= a + a / 20 + 16,
            "level {level} (strat {:?}): usingCDict={b} vs usingDict={a} — CDict under-indexed?",
            cp.strategy
        );
    }
}
