//! Phase 3 real-data harness: decompress canonical `.zst` fixtures
//! produced by the upstream `zstd` CLI and verify byte-exact parity
//! with their original inputs.
//!
//! Any divergence here is a decoder bug — the test file names encode
//! the upstream command used to make them.

use std::fs;
use std::path::PathBuf;

use zstd_pure_rs::common::error::{ERR_getErrorName, ERR_isError};
use zstd_pure_rs::decompress::zstd_decompress::{
    ZSTD_decompress, ZSTD_findDecompressedSize, ZSTD_CONTENTSIZE_ERROR, ZSTD_CONTENTSIZE_UNKNOWN,
};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn read_fixture(name: &str) -> Vec<u8> {
    fs::read(fixtures_dir().join(name)).unwrap_or_else(|e| {
        panic!(
            "missing fixture {name}: {e}; regenerate with scripts in tests/fixtures/"
        )
    })
}

#[test]
fn decompress_upstream_lorem50_level1() {
    // First 50 bytes of lorem. Upstream picks a raw block for such a
    // small input → exercises the frame-loop w/ windowed (non
    // single-segment) FHD.
    let compressed = read_fixture("lorem50_l1.zst");
    let expected = read_fixture("lorem50.txt");
    let declared = ZSTD_findDecompressedSize(&compressed);
    // FCS not declared for non-single-segment with fcsID=0.
    let dst_cap = if declared == ZSTD_CONTENTSIZE_UNKNOWN {
        compressed.len() * 32
    } else {
        declared as usize
    };
    let mut dst = vec![0u8; dst_cap.max(64)];
    let out = ZSTD_decompress(&mut dst, &compressed);
    assert!(
        !ERR_isError(out),
        "decoder returned error: {}",
        ERR_getErrorName(out)
    );
    assert_eq!(out, expected.len());
    assert_eq!(&dst[..out], &expected[..]);
}

#[test]
fn decompress_upstream_rep100_level1() {
    // 100 × 'a'. Generated via:
    //   printf 'aaaa...' | zstd --no-check -1
    // Exercises RLE-ish compression via back-references.
    let compressed = read_fixture("rep100_l1.zst");
    let expected = read_fixture("rep100.txt");
    let declared = ZSTD_findDecompressedSize(&compressed);
    assert_eq!(declared, expected.len() as u64);
    let mut dst = vec![0u8; declared as usize];
    let out = ZSTD_decompress(&mut dst, &compressed);
    assert!(
        !ERR_isError(out),
        "decoder returned error: {}",
        ERR_getErrorName(out)
    );
    assert_eq!(out, expected.len());
    assert_eq!(&dst[..out], &expected[..]);
}

/// Targeted test: pull the HUF-compressed literals section out of the
/// failing `lorem80_l1.zst` fixture and hand it directly to
/// `HUF_decompress1X1_DCtx_wksp`. Localizes whether the bug is in the
/// HUF layer or in the zstd glue above it.
#[test]
fn huf_literals_blob_from_lorem80_single_stream() {
    use zstd_pure_rs::decompress::huf_decompress::{
        HUF_decompress1X1_DCtx_wksp, HUF_DECOMPRESS_WORKSPACE_SIZE_U32, HUF_DTABLE_SIZE_U32,
        HUF_TABLELOG_MAX,
    };

    let zst = read_fixture("lorem80_l1.zst");
    // Frame header 6 bytes, then block header 3 bytes, then literals
    // header 3 bytes (lhlCode=0 path). HUF blob runs from offset 12
    // for 63 bytes. litSize = 73 → output buffer sized accordingly.
    let huf_start = 12;
    let huf_size = 63;
    let exp_lit = 73;
    let blob = &zst[huf_start..huf_start + huf_size];

    let mut dtable = vec![0u32; HUF_DTABLE_SIZE_U32(HUF_TABLELOG_MAX)];
    // DTableDesc header's maxTableLog must be set for readDTableX1 to
    // accept the tableLog returned by the upstream encoder.
    use zstd_pure_rs::decompress::huf_decompress::{DTableDesc, HUF_setDTableDesc};
    HUF_setDTableDesc(
        &mut dtable,
        DTableDesc {
            maxTableLog: (HUF_TABLELOG_MAX - 1) as u8,
            ..Default::default()
        },
    );
    let mut wksp = vec![0u32; HUF_DECOMPRESS_WORKSPACE_SIZE_U32];
    let mut out = vec![0u8; exp_lit];
    let rc = HUF_decompress1X1_DCtx_wksp(&mut dtable, &mut out, blob, &mut wksp, 0);
    assert!(
        !ERR_isError(rc),
        "HUF_decompress1X1_DCtx_wksp failed: {}",
        ERR_getErrorName(rc)
    );
    assert_eq!(rc, exp_lit, "decoded literal count");
}

/// Targeted test: run `ZSTD_decodeLiteralsBlock` on the full
/// 72-byte block of `lorem80_l1.zst` and verify it consumes 66 bytes
/// (3-byte literals header + 63-byte HUF blob) and yields 73 literals.
#[test]
fn decode_literals_block_lorem80() {
    use zstd_pure_rs::decompress::zstd_decompress_block::{
        streaming_operation, ZSTD_buildDefaultSeqTables, ZSTD_DCtx, ZSTD_decodeLiteralsBlock,
    };

    let zst = read_fixture("lorem80_l1.zst");
    // Block starts at offset 9 (after 6-byte frame header + 3-byte
    // block header). Block is 72 bytes.
    let block = &zst[9..9 + 72];

    let mut dctx = ZSTD_DCtx::new();
    ZSTD_buildDefaultSeqTables(&mut dctx);
    let mut dst = vec![0u8; 256];
    let consumed = ZSTD_decodeLiteralsBlock(
        &mut dctx,
        block,
        &mut dst,
        streaming_operation::not_streaming,
    );
    assert!(
        !ERR_isError(consumed),
        "decodeLiteralsBlock error: {}",
        ERR_getErrorName(consumed)
    );
    assert_eq!(consumed, 66, "should consume 3 header + 63 HUF bytes");
    assert_eq!(dctx.litSize, 73);
}

#[test]
fn decompress_upstream_lorem_level1_with_checksum() {
    // Same input as lorem_l1 but encoded with `--check`, which
    // appends a 4-byte XXH64 checksum trailer and sets the
    // `checksumFlag` in the frame header. Exercises the XXH64
    // block-update loop plus the trailer compare.
    let compressed = read_fixture("lorem_l1_checksum.zst");
    let expected = read_fixture("lorem.txt");
    let mut dst = vec![0u8; expected.len()];
    let out = ZSTD_decompress(&mut dst, &compressed);
    assert!(
        !ERR_isError(out),
        "decoder returned error: {}",
        ERR_getErrorName(out)
    );
    assert_eq!(out, expected.len());
    assert_eq!(&dst[..out], &expected[..]);
}

#[test]
fn decompress_upstream_lorem_level19() {
    // Level 19 uses much higher-compression settings (btultra strategy,
    // larger windowLog, potentially HUF-X2 double-symbol literals,
    // longer FSE tableLogs). The decoder's scalar path should still
    // handle it.
    let compressed = read_fixture("lorem_l19.zst");
    let expected = read_fixture("lorem.txt");
    let mut dst = vec![0u8; expected.len()];
    let out = ZSTD_decompress(&mut dst, &compressed);
    assert!(
        !ERR_isError(out),
        "decoder returned error: {}",
        ERR_getErrorName(out)
    );
    assert_eq!(out, expected.len());
    assert_eq!(&dst[..out], &expected[..]);
}

#[test]
fn decompress_zstd_header_level3() {
    // 182 KB of real C header text (upstream's `lib/zstd.h`) →
    // 47 KB compressed at level 3. Many blocks, diverse content.
    let compressed = read_fixture("zstd_h_l3.zst");
    let expected = read_fixture("zstd_h.txt");
    let mut dst = vec![0u8; expected.len()];
    let out = ZSTD_decompress(&mut dst, &compressed);
    assert!(!ERR_isError(out), "err: {}", ERR_getErrorName(out));
    assert_eq!(out, expected.len());
    // Don't dump ~182KB on mismatch; find first divergence instead.
    if dst[..out] != expected[..] {
        let first = dst
            .iter()
            .zip(expected.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!(
            "byte mismatch at index {first}: decoded=0x{:02x} expected=0x{:02x}",
            dst[first], expected[first]
        );
    }
}

#[test]
fn decompress_zstd_header_level9_with_checksum() {
    // Same input, level 9, with checksum. Covers high-compression
    // decoder path on a realistic payload.
    let compressed = read_fixture("zstd_h_l9_checksum.zst");
    let expected = read_fixture("zstd_h.txt");
    let mut dst = vec![0u8; expected.len()];
    let out = ZSTD_decompress(&mut dst, &compressed);
    assert!(!ERR_isError(out), "err: {}", ERR_getErrorName(out));
    assert_eq!(out, expected.len());
    if dst[..out] != expected[..] {
        let first = dst
            .iter()
            .zip(expected.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!(
            "byte mismatch at index {first}: decoded=0x{:02x} expected=0x{:02x}",
            dst[first], expected[first]
        );
    }
}

#[test]
fn decompress_binary_level3() {
    // Pseudo-random binary with occasional repeats. Level-3 strategy
    // ≈ lazy2 with medium search effort. Stresses the decoder on
    // high-entropy literals + realistic back-references.
    let compressed = read_fixture("binary_l3.zst");
    let expected = read_fixture("binary.bin");
    let mut dst = vec![0u8; expected.len()];
    let out = ZSTD_decompress(&mut dst, &compressed);
    assert!(!ERR_isError(out), "err: {}", ERR_getErrorName(out));
    assert_eq!(out, expected.len());
    assert_eq!(&dst[..out], &expected[..]);
}

#[test]
fn decompress_binary_level12() {
    // Same input, level 12 — btopt strategy on binary data. Exercises
    // higher-tableLog FSE + possibly HUF-X2 literals.
    let compressed = read_fixture("binary_l12.zst");
    let expected = read_fixture("binary.bin");
    let mut dst = vec![0u8; expected.len()];
    let out = ZSTD_decompress(&mut dst, &compressed);
    assert!(!ERR_isError(out), "err: {}", ERR_getErrorName(out));
    assert_eq!(out, expected.len());
    assert_eq!(&dst[..out], &expected[..]);
}

#[test]
fn decompress_empty_input_level1() {
    // A zero-byte payload compressed to a valid `.zst` frame.
    // Exercises the "last block, zero content" edge case.
    let compressed = read_fixture("empty_l1.zst");
    let declared = ZSTD_findDecompressedSize(&compressed);
    assert_eq!(declared, 0);
    let mut dst = vec![0u8; 16];
    let out = ZSTD_decompress(&mut dst, &compressed);
    assert!(!ERR_isError(out), "err: {}", ERR_getErrorName(out));
    assert_eq!(out, 0);
}

#[test]
fn decompress_upstream_lorem_level1() {
    // Generated via:
    //   python lorem-generator > tests/fixtures/lorem.txt
    //   zstd --no-check -1 -o tests/fixtures/lorem_l1.zst tests/fixtures/lorem.txt
    // 10106 → 2755 bytes. Exercises HUF literals + full FSE
    // sequence-decode path + back-reference matches of varying offsets.
    let compressed = read_fixture("lorem_l1.zst");
    let expected = read_fixture("lorem.txt");

    let declared = ZSTD_findDecompressedSize(&compressed);
    assert!(declared != ZSTD_CONTENTSIZE_ERROR);
    // Larger inputs may be multi-block; FCS should still be present.
    assert_eq!(declared, expected.len() as u64);

    let mut dst = vec![0u8; declared as usize];
    let out = ZSTD_decompress(&mut dst, &compressed);
    assert!(
        !ERR_isError(out),
        "decoder returned error: {}",
        ERR_getErrorName(out)
    );
    assert_eq!(out, expected.len(), "decoded length mismatch");
    if dst[..out] != expected[..] {
        // Find first divergence to localize the bug.
        let first = dst
            .iter()
            .zip(expected.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!(
            "byte mismatch at index {first}: decoded=0x{:02x} expected=0x{:02x}",
            dst[first], expected[first]
        );
    }
}

#[test]
fn decompress_upstream_hello_level1() {
    // Generated via:
    //   echo 'HELLO HELLO HELLO HELLO HELLO' | zstd --no-check -1 \
    //     -o tests/fixtures/hello_l1.zst
    let compressed = read_fixture("hello_l1.zst");
    let expected: &[u8] = b"HELLO HELLO HELLO HELLO HELLO\n";

    let declared = ZSTD_findDecompressedSize(&compressed);
    assert!(declared != ZSTD_CONTENTSIZE_ERROR, "bad frame");
    // The upstream fixture carries a declared FCS (singleSegment + fcsID=0).
    assert_ne!(
        declared, ZSTD_CONTENTSIZE_UNKNOWN,
        "expected declared FCS in fixture"
    );
    assert_eq!(declared, expected.len() as u64);

    let mut dst = vec![0u8; declared as usize];
    let out = ZSTD_decompress(&mut dst, &compressed);
    assert!(
        !ERR_isError(out),
        "decoder returned error: {}",
        ERR_getErrorName(out)
    );
    assert_eq!(out, expected.len());
    assert_eq!(&dst[..out], expected);
}
