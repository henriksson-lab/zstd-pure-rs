//! Optional parity tests over larger real-world corpora. The harness
//! is env-driven so it can run locally without bundling huge datasets.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use zstd_pure_rs::common::error::{ERR_getErrorName, ERR_isError};
use zstd_pure_rs::compress::zstd_compress::{ZSTD_compress, ZSTD_compressBound};
use zstd_pure_rs::decompress::zstd_decompress::ZSTD_decompress;

fn upstream_zstd() -> Option<PathBuf> {
    let out = Command::new("which").arg("zstd").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let path = String::from_utf8(out.stdout).ok()?;
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(PathBuf::from(trimmed))
}

fn candidate_corpora() -> Vec<(&'static str, PathBuf)> {
    let mut out = Vec::new();
    let env_one = std::env::var_os("ZSTD_PURE_RS_SILESIA")
        .map(PathBuf::from)
        .filter(|p| p.exists());
    let env_two = std::env::var_os("ZSTD_PURE_RS_ENWIK8")
        .map(PathBuf::from)
        .filter(|p| p.exists());
    let env_three = std::env::var_os("ZSTD_PURE_RS_FASTQ")
        .map(PathBuf::from)
        .filter(|p| p.exists());
    if let Some(path) = env_one {
        out.push(("silesia", path));
    }
    if let Some(path) = env_two {
        out.push(("enwik8", path));
    }
    if let Some(path) = env_three {
        out.push(("fastq", path));
    }
    for (label, raw) in [
        (
            "fastq",
            "/husky/henriksson/for_claude/skesa/external/err486835_mgenitalium/subset_1.fastq",
        ),
        (
            "fastq",
            "/husky/henriksson/for_claude/skesa/external/err486835_mgenitalium/subset_2.fastq",
        ),
        ("fastq", "/husky/henriksson/for_claude/skesa/mm2rs-rnafull-focus.fq"),
    ] {
        let path = PathBuf::from(raw);
        if path.exists() && !out.iter().any(|(_, existing)| existing == &path) {
            out.push((label, path));
        }
    }
    out
}

fn compress_with_rust(src: &[u8], level: i32) -> Vec<u8> {
    let mut dst = vec![0u8; ZSTD_compressBound(src.len()).max(64)];
    let out = ZSTD_compress(&mut dst, src, level);
    assert!(
        !ERR_isError(out),
        "compress failed at level {level}: {}",
        ERR_getErrorName(out)
    );
    dst.truncate(out);
    dst
}

fn decompress_with_rust(src: &[u8], expected_len: usize) -> Vec<u8> {
    let mut dst = vec![0u8; expected_len.max(1)];
    let out = ZSTD_decompress(&mut dst, src);
    assert!(
        !ERR_isError(out),
        "decompress failed: {}",
        ERR_getErrorName(out)
    );
    dst.truncate(out);
    dst
}

fn decompress_with_upstream(bin: &Path, frame: &[u8]) -> Vec<u8> {
    let mut child = Command::new(bin)
        .args(["-d", "-c", "-q", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn upstream zstd");
    // Stream stdin from a dedicated thread so a large compressed
    // frame doesn't deadlock against a full stdout pipe. Without
    // this, write_all() blocks on a full stdin pipe while the child
    // simultaneously blocks writing decompressed output to stdout —
    // since wait_with_output() (which drains stdout) hasn't been
    // called yet. Surfaces on >64 KB inputs.
    let mut stdin = child.stdin.take().expect("zstd stdin");
    let frame_owned = frame.to_vec();
    let writer = std::thread::spawn(move || {
        stdin.write_all(&frame_owned).expect("write_all to zstd stdin");
        drop(stdin);
    });
    let out = child.wait_with_output().expect("wait upstream");
    writer.join().expect("zstd stdin writer thread");
    assert!(
        out.status.success(),
        "upstream rejected our frame: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

#[test]
#[ignore = "requires optional large corpus files"]
fn large_corpus_roundtrip_and_upstream_cross_parity() {
    let corpora = candidate_corpora();
    if corpora.is_empty() {
        eprintln!(
            "no large corpora configured; set ZSTD_PURE_RS_SILESIA / ZSTD_PURE_RS_ENWIK8 / \
             ZSTD_PURE_RS_FASTQ or place FASTQ files under /husky/henriksson/for_claude/skesa"
        );
        return;
    }
    let upstream = upstream_zstd();
    for (label, path) in corpora {
        let payload = fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        assert!(
            !payload.is_empty(),
            "{} corpus {} was empty",
            label,
            path.display()
        );
        for &level in &[1, 3, 10, 19] {
            let compressed = compress_with_rust(&payload, level);
            let roundtrip = decompress_with_rust(&compressed, payload.len());
            assert_eq!(
                roundtrip,
                payload,
                "[{label} {} level {level}] rust roundtrip mismatch",
                path.display()
            );
            if let Some(bin) = upstream.as_deref() {
                let upstream_decoded = decompress_with_upstream(bin, &compressed);
                assert_eq!(
                    upstream_decoded,
                    payload,
                    "[{label} {} level {level}] upstream cross-decode mismatch",
                    path.display()
                );
            }
        }
    }
}
