//! Expanded real-data integration tests over committed fixtures and
//! vendored upstream golden inputs.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use zstd_pure_rs::common::error::{ERR_getErrorName, ERR_isError};
use zstd_pure_rs::compress::zstd_compress::{
    ZSTD_CCtx, ZSTD_CCtx_setParameter, ZSTD_CStreamOutSize, ZSTD_cParameter, ZSTD_compress,
    ZSTD_compress2, ZSTD_compressBound, ZSTD_compressStream, ZSTD_endStream, ZSTD_initCStream,
    ZSTD_writeSkippableFrame,
};
use zstd_pure_rs::decompress::zstd_decompress::{
    ZSTD_decompress, ZSTD_decompressStream, ZSTD_decompress_usingDict, ZSTD_initDStream,
};
use zstd_pure_rs::decompress::zstd_decompress_block::{ZSTD_DCtx, ZSTD_buildDefaultSeqTables};

fn upstream_zstd() -> Option<PathBuf> {
    let out = Command::new("which").arg("zstd").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let path = String::from_utf8(out.stdout).ok()?;
    let trimmed = path.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(PathBuf::from(trimmed))
    }
}

fn corpus_cases() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("zstd_h", include_bytes!("fixtures/zstd_h.txt").to_vec()),
        ("lorem", include_bytes!("fixtures/lorem.txt").to_vec()),
        ("binary", include_bytes!("fixtures/binary.bin").to_vec()),
        ("fastq", include_bytes!("fixtures/small.fastq").repeat(32)),
    ]
}

fn compress_one_shot(src: &[u8], level: i32) -> Vec<u8> {
    let mut dst = vec![0u8; ZSTD_compressBound(src.len()).max(64)];
    let n = ZSTD_compress(&mut dst, src, level);
    assert!(
        !ERR_isError(n),
        "compress level {level} failed: {}",
        ERR_getErrorName(n)
    );
    dst.truncate(n);
    dst
}

fn compress_with_checksum(src: &[u8], level: i32) -> Vec<u8> {
    let mut cctx = ZSTD_CCtx::default();
    assert_eq!(
        ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_compressionLevel, level),
        0
    );
    assert_eq!(
        ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_checksumFlag, 1),
        0
    );
    let mut dst = vec![0u8; ZSTD_compressBound(src.len()).max(64) + 4];
    let n = ZSTD_compress2(&mut cctx, &mut dst, src);
    assert!(
        !ERR_isError(n),
        "checksum compress level {level} failed: {}",
        ERR_getErrorName(n)
    );
    dst.truncate(n);
    dst
}

fn decompress_one_shot(frame: &[u8], expected_len: usize) -> Vec<u8> {
    let mut dst = vec![0u8; expected_len.max(1)];
    let n = ZSTD_decompress(&mut dst, frame);
    assert!(
        !ERR_isError(n),
        "decompress failed: {}",
        ERR_getErrorName(n)
    );
    dst.truncate(n);
    dst
}

fn upstream_decode(bin: &Path, frame: &[u8]) -> Vec<u8> {
    let mut child = Command::new(bin)
        .args(["-d", "-c", "-q", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn upstream zstd");
    let mut stdin = child.stdin.take().expect("upstream stdin");
    let frame = frame.to_vec();
    let writer = std::thread::spawn(move || {
        stdin.write_all(&frame).expect("write upstream stdin");
    });
    let out = child.wait_with_output().expect("wait upstream zstd");
    writer.join().expect("join upstream writer");
    assert!(
        out.status.success(),
        "upstream rejected frame: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

fn upstream_compress(bin: &Path, payload: &[u8], level: i32) -> Vec<u8> {
    let mut child = Command::new(bin)
        .args(["-c", "-q", &format!("-{level}"), "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn upstream zstd compressor");
    let mut stdin = child.stdin.take().expect("upstream stdin");
    let payload = payload.to_vec();
    let writer = std::thread::spawn(move || {
        stdin.write_all(&payload).expect("write upstream stdin");
    });
    let out = child.wait_with_output().expect("wait upstream zstd");
    writer.join().expect("join upstream writer");
    assert!(
        out.status.success(),
        "upstream compression failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

fn streaming_compress(src: &[u8], level: i32, chunk_size: usize) -> Vec<u8> {
    let mut cctx = ZSTD_CCtx::default();
    let init = ZSTD_initCStream(&mut cctx, level);
    assert!(
        !ERR_isError(init),
        "init stream failed: {}",
        ERR_getErrorName(init)
    );

    let mut dst = vec![0u8; ZSTD_compressBound(src.len()).max(ZSTD_CStreamOutSize()) + 1024];
    let mut dst_pos = 0usize;
    for chunk in src.chunks(chunk_size) {
        let mut src_pos = 0usize;
        let rc = ZSTD_compressStream(&mut cctx, &mut dst, &mut dst_pos, chunk, &mut src_pos);
        assert!(
            !ERR_isError(rc),
            "compressStream failed: {}",
            ERR_getErrorName(rc)
        );
        assert_eq!(src_pos, chunk.len(), "compressStream did not consume chunk");
    }
    loop {
        let rc = ZSTD_endStream(&mut cctx, &mut dst, &mut dst_pos);
        assert!(
            !ERR_isError(rc),
            "endStream failed: {}",
            ERR_getErrorName(rc)
        );
        if rc == 0 {
            break;
        }
    }
    dst.truncate(dst_pos);
    dst
}

fn streaming_decompress(frame: &[u8], expected_len: usize, chunk_size: usize) -> Vec<u8> {
    let mut dctx = ZSTD_DCtx::new();
    ZSTD_buildDefaultSeqTables(&mut dctx);
    let init = ZSTD_initDStream(&mut dctx);
    assert!(
        !ERR_isError(init),
        "init dstream failed: {}",
        ERR_getErrorName(init)
    );
    let mut out = vec![0u8; expected_len.max(1)];
    let mut out_pos = 0usize;
    let mut cursor = 0usize;
    while cursor < frame.len() {
        let end = (cursor + chunk_size).min(frame.len());
        let mut in_pos = 0usize;
        let rc = ZSTD_decompressStream(
            &mut dctx,
            &mut out,
            &mut out_pos,
            &frame[cursor..end],
            &mut in_pos,
        );
        assert!(
            !ERR_isError(rc),
            "decompressStream failed: {}",
            ERR_getErrorName(rc)
        );
        assert_eq!(
            in_pos,
            end - cursor,
            "decompressStream did not consume chunk"
        );
        cursor = end;
    }
    loop {
        let mut in_pos = 0usize;
        let rc = ZSTD_decompressStream(&mut dctx, &mut out, &mut out_pos, &[], &mut in_pos);
        assert!(
            !ERR_isError(rc),
            "decompressStream drain failed: {}",
            ERR_getErrorName(rc)
        );
        if rc == 0 {
            break;
        }
    }
    out.truncate(out_pos);
    out
}

fn skippable(payload: &[u8], variant: u32) -> Vec<u8> {
    let mut buf = vec![0u8; payload.len() + 8];
    let n = ZSTD_writeSkippableFrame(&mut buf, payload, variant);
    assert!(
        !ERR_isError(n),
        "write skippable failed: {}",
        ERR_getErrorName(n)
    );
    buf.truncate(n);
    buf
}

#[test]
fn real_corpus_compression_roundtrips_and_cross_decodes() {
    let upstream = upstream_zstd();
    let check_ratio = std::env::var_os("ZSTD_PURE_RS_RATIO_CHECK").is_some();
    let mut upstream_frames = Vec::new();
    let mut upstream_expected = Vec::new();
    for (name, payload) in corpus_cases() {
        for level in [1, 3, 5, 10, 19, 22] {
            let compressed = compress_one_shot(&payload, level);
            assert_eq!(
                decompress_one_shot(&compressed, payload.len()),
                payload,
                "{name} level {level} rust roundtrip mismatch"
            );
            if let Some(bin) = upstream.as_deref() {
                upstream_frames.extend_from_slice(&compressed);
                upstream_expected.extend_from_slice(&payload);

                if check_ratio && name == "zstd_h" {
                    let reference = upstream_compress(bin, &payload, level);
                    let slack = compressed.len() as f64 / reference.len().max(1) as f64;
                    assert!(
                        slack <= 3.0,
                        "{name} level {level} compressed size regressed: ours={} upstream={} slack={slack:.2}x",
                        compressed.len(),
                        reference.len()
                    );
                }
            }
            assert!(
                compressed.len() <= ZSTD_compressBound(payload.len()),
                "{name} level {level} exceeded compressBound"
            );
        }
    }
    if let Some(bin) = upstream.as_deref() {
        assert_eq!(
            upstream_decode(bin, &upstream_frames),
            upstream_expected,
            "batched upstream cross-decode mismatch"
        );
    }
}

#[test]
fn committed_fastq_fixture_exercises_one_shot_and_streaming_paths() {
    let payload = include_bytes!("fixtures/small.fastq").repeat(48);
    let compressed = compress_one_shot(&payload, 3);
    assert_eq!(decompress_one_shot(&compressed, payload.len()), payload);
    assert_eq!(
        streaming_decompress(&compressed, payload.len(), 17),
        payload
    );
}

#[test]
fn dictionary_roundtrips_cover_http_fastq_and_header_payloads() {
    let cases: Vec<(&str, Vec<u8>, Vec<u8>)> = vec![
        (
            "http",
            include_bytes!("../zstd/tests/golden-compression/http").to_vec(),
            include_bytes!("../zstd/tests/golden-compression/http").repeat(4),
        ),
        (
            "fastq",
            include_bytes!("fixtures/small.fastq")[..512].to_vec(),
            include_bytes!("fixtures/small.fastq").repeat(16),
        ),
        (
            "header",
            include_bytes!("fixtures/zstd_h.txt")[..4096].to_vec(),
            include_bytes!("fixtures/zstd_h.txt")[2048..24576].to_vec(),
        ),
    ];

    for (name, dict, payload) in cases {
        let mut cctx = ZSTD_CCtx::default();
        let mut compressed = vec![0u8; ZSTD_compressBound(payload.len()).max(64)];
        let n = zstd_pure_rs::compress::zstd_compress::ZSTD_compress_usingDict(
            &mut cctx,
            &mut compressed,
            &payload,
            &dict,
            3,
        );
        assert!(
            !ERR_isError(n),
            "{name} dict compression failed: {}",
            ERR_getErrorName(n)
        );
        compressed.truncate(n);

        let mut dctx = ZSTD_DCtx::new();
        let mut decoded = vec![0u8; payload.len()];
        let d = ZSTD_decompress_usingDict(&mut dctx, &mut decoded, &compressed, &dict);
        assert!(
            !ERR_isError(d),
            "{name} dict decompression failed: {}",
            ERR_getErrorName(d)
        );
        decoded.truncate(d);
        assert_eq!(decoded, payload, "{name} dict roundtrip mismatch");
    }
}

#[test]
fn real_data_streaming_roundtrips_over_awkward_chunk_sizes() {
    let payloads = [
        ("zstd_h", include_bytes!("fixtures/zstd_h.txt").to_vec()),
        ("binary", include_bytes!("fixtures/binary.bin").repeat(20)),
    ];
    for (name, payload) in payloads {
        for chunk in [1usize, 3, 7, 64, 4096, 131071, 131072, 131073] {
            let compressed = streaming_compress(&payload, 3, chunk);
            let decoded = streaming_decompress(&compressed, payload.len(), chunk);
            assert_eq!(decoded, payload, "{name} chunk {chunk} streaming mismatch");
        }
    }
}

#[test]
fn concatenated_and_skippable_real_frames_decode_in_order() {
    let a = include_bytes!("fixtures/lorem.txt").to_vec();
    let b = include_bytes!("fixtures/small.fastq").repeat(8);
    let c = include_bytes!("fixtures/binary.bin").to_vec();

    let frame_a = compress_one_shot(&a, 1);
    let frame_b = compress_with_checksum(&b, 3);
    let frame_c = compress_one_shot(&c, 10);

    let mut stream = Vec::new();
    stream.extend_from_slice(&skippable(b"leading metadata", 0));
    stream.extend_from_slice(&frame_a);
    stream.extend_from_slice(&skippable(b"between lorem and fastq", 7));
    stream.extend_from_slice(&frame_b);
    stream.extend_from_slice(&frame_c);
    stream.extend_from_slice(&skippable(b"trailing metadata", 15));

    let mut expected = Vec::new();
    expected.extend_from_slice(&a);
    expected.extend_from_slice(&b);
    expected.extend_from_slice(&c);

    assert_eq!(decompress_one_shot(&stream, expected.len()), expected);
    assert_eq!(streaming_decompress(&stream, expected.len(), 113), expected);
}

#[test]
fn vendored_upstream_golden_compression_inputs_emit_decodable_frames() {
    let upstream = upstream_zstd();
    for input in [
        "zstd/tests/golden-compression/http",
        "zstd/tests/golden-compression/huffman-compressed-larger",
        "zstd/tests/golden-compression/large-literal-and-match-lengths",
        "zstd/tests/golden-compression/PR-3517-block-splitter-corruption-test",
    ] {
        let payload = std::fs::read(input).unwrap_or_else(|e| panic!("read {input}: {e}"));
        for level in [1, 3, 9, 19] {
            let compressed = compress_one_shot(&payload, level);
            assert_eq!(
                decompress_one_shot(&compressed, payload.len()),
                payload,
                "{input} level {level} rust roundtrip mismatch"
            );
            if let Some(bin) = upstream.as_deref() {
                assert_eq!(
                    upstream_decode(bin, &compressed),
                    payload,
                    "{input} level {level} upstream cross-decode mismatch"
                );
            }
        }
    }
}
