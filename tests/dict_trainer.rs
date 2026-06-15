//! End-to-end tests for the translated ZDICT dictionary trainers.
//! Train a dictionary, validate its header, and round-trip a sample
//! through it (compress with dict -> decompress with dict).

use zstd_pure_rs::prelude::*;

/// Build a corpus of many small, repetitive-but-varied samples so the
/// trainers have recurring substrings / dmers to latch onto.
fn make_corpus() -> (Vec<u8>, Vec<usize>) {
    let mut samples = Vec::new();
    let mut sizes = Vec::new();
    let phrases: [&[u8]; 6] = [
        b"the quick brown fox jumps over the lazy dog. ",
        b"pack my box with five dozen liquor jugs. ",
        b"how vexingly quick daft zebras jump! ",
        b"the five boxing wizards jump quickly. ",
        b"sphinx of black quartz, judge my vow. ",
        b"jackdaws love my big sphinx of quartz. ",
    ];
    for i in 0..200u32 {
        let mut s = Vec::new();
        for j in 0..8u32 {
            s.extend_from_slice(phrases[((i + j) % 6) as usize]);
            s.extend_from_slice(&(i.wrapping_mul(2654435761).wrapping_add(j)).to_le_bytes());
        }
        sizes.push(s.len());
        samples.extend_from_slice(&s);
    }
    (samples, sizes)
}

/// Validate dict header and round-trip one sample through it.
fn check_dict(dict: &[u8], sample: &[u8]) {
    assert!(dict.len() >= 8, "dict too small: {}", dict.len());
    assert_eq!(
        &dict[..4],
        &ZSTD_MAGIC_DICTIONARY.to_le_bytes(),
        "dictionary must start with the zstd dictionary magic"
    );
    let dictID = ZDICT_getDictID(dict);
    assert_ne!(dictID, 0, "trained dict must have a non-zero dictID");

    // Round-trip: compress the sample with the dict, decompress with it.
    let mut cctx = ZSTD_createCCtx().expect("cctx");
    let mut cdst = vec![0u8; ZSTD_compressBound(sample.len())];
    let csize = ZSTD_compress_usingDict(&mut cctx, &mut cdst, sample, dict, 3);
    assert!(!ERR_isError(csize), "compress_usingDict failed: {csize:#x}");

    let mut dctx = ZSTD_createDCtx();
    let mut out = vec![0u8; sample.len()];
    let dsize = ZSTD_decompress_usingDict(&mut dctx, &mut out, &cdst[..csize], dict);
    assert!(!ERR_isError(dsize), "decompress_usingDict failed: {dsize:#x}");
    assert_eq!(&out[..dsize], sample, "dict round-trip mismatch");
}

#[test]
fn zdict_trainFromBuffer_default_trains_usable_dict() {
    let (samples, sizes) = make_corpus();
    let mut dict = vec![0u8; 16 * 1024];
    let cap = dict.len();
    let n = ZDICT_trainFromBuffer(&mut dict, cap, &samples, &sizes, sizes.len() as u32);
    assert!(!ERR_isError(n), "ZDICT_trainFromBuffer failed: {n:#x}");
    dict.truncate(n);
    check_dict(&dict, &samples[..sizes[0]]);
}

#[test]
fn zdict_trainFromBuffer_fastCover_explicit() {
    let (samples, sizes) = make_corpus();
    let mut dict = vec![0u8; 16 * 1024];
    let cap = dict.len();
    let mut params = ZDICT_fastCover_params_t::default();
    params.d = 8;
    params.k = 200;
    let n = ZDICT_trainFromBuffer_fastCover(
        &mut dict, cap,
        &samples,
        &sizes,
        sizes.len() as u32,
        params,
    );
    assert!(!ERR_isError(n), "fastCover train failed: {n:#x}");
    dict.truncate(n);
    check_dict(&dict, &samples[..sizes[0]]);
}

#[test]
fn zdict_trainFromBuffer_cover_explicit() {
    let (samples, sizes) = make_corpus();
    let mut dict = vec![0u8; 16 * 1024];
    let cap = dict.len();
    let mut params = ZDICT_cover_params_t::default();
    params.d = 8;
    params.k = 200;
    let n = ZDICT_trainFromBuffer_cover(
        &mut dict, cap,
        &samples,
        &sizes,
        sizes.len() as u32,
        params,
    );
    assert!(!ERR_isError(n), "cover train failed: {n:#x}");
    dict.truncate(n);
    check_dict(&dict, &samples[..sizes[0]]);
}

#[test]
fn zdict_trainFromBuffer_legacy_explicit() {
    let (samples, sizes) = make_corpus();
    let mut dict = vec![0u8; 16 * 1024];
    let cap = dict.len();
    let params = ZDICT_legacy_params_t::default();
    let n = ZDICT_trainFromBuffer_legacy(
        &mut dict, cap,
        &samples,
        &sizes,
        sizes.len() as u32,
        params,
    );
    assert!(!ERR_isError(n), "legacy train failed: {n:#x}");
    dict.truncate(n);
    check_dict(&dict, &samples[..sizes[0]]);
}

#[test]
fn dump_trained_dict_for_cross_check() {
    // Writes a trained dict + a sample so an external test can validate
    // upstream-format compatibility. Gated on an env var so it's a no-op
    // in normal runs.
    if std::env::var("ZSTD_PURE_RS_DUMP_DICT").is_err() {
        return;
    }
    let (samples, sizes) = make_corpus();
    let mut dict = vec![0u8; 16 * 1024];
    let cap = dict.len();
    let n = ZDICT_trainFromBuffer(&mut dict, cap, &samples, &sizes, sizes.len() as u32);
    assert!(!ERR_isError(n));
    dict.truncate(n);
    std::fs::write(".tmpccc/trained.dict", &dict).unwrap();
    std::fs::write(".tmpccc/sample.bin", &samples[..sizes[0] * 4]).unwrap();
}
