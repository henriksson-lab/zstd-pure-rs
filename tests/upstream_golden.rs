//! Port of a small, durable subset of upstream zstd golden
//! decompression checks. These fixtures are copied into
//! `tests/fixtures/upstream-zstd`, so the tests stay self-contained.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use zstd_pure_rs::common::error::{ERR_getErrorName, ERR_isError};
use zstd_pure_rs::compress::zstd_compress::{ZSTD_compress, ZSTD_compressBound};
use zstd_pure_rs::decompress::zstd_decompress::{
    ZSTD_decompress, ZSTD_findDecompressedSize, ZSTD_getFrameContentSize, ZSTD_CONTENTSIZE_ERROR,
    ZSTD_CONTENTSIZE_UNKNOWN,
};

const BLOCK_128K_DECODED_SIZE: usize = 131_068;
const RLE_FIRST_BLOCK_DECODED_SIZE: usize = 1_048_576;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn upstream_golden_dir(kind: &str) -> PathBuf {
    repo_root().join("tests/fixtures/upstream-zstd").join(kind)
}

fn fixture_paths(kind: &str) -> Vec<PathBuf> {
    let mut paths = fs::read_dir(upstream_golden_dir(kind))
        .unwrap_or_else(|e| panic!("{kind} dir: {e}"))
        .map(|entry| entry.expect("dirent").path())
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

fn fixture_name(path: &Path) -> &str {
    path.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_else(|| panic!("{} has a non-UTF-8 file name", path.display()))
}

fn expected_golden_output(path: &Path) -> Vec<u8> {
    match fixture_name(path) {
        "block-128k.zst" => vec![0; BLOCK_128K_DECODED_SIZE],
        "empty-block.zst" => Vec::new(),
        "rle-first-block.zst" => vec![0; RLE_FIRST_BLOCK_DECODED_SIZE],
        "zeroSeq_2B.zst" => b"Hello World!\n".to_vec(),
        name => panic!("unexpected golden decompression fixture: {name}"),
    }
}

fn expected_golden_content_size(path: &Path) -> u64 {
    match fixture_name(path) {
        "block-128k.zst" | "empty-block.zst" | "zeroSeq_2B.zst" => ZSTD_CONTENTSIZE_UNKNOWN,
        "rle-first-block.zst" => RLE_FIRST_BLOCK_DECODED_SIZE as u64,
        name => panic!("unexpected golden decompression fixture: {name}"),
    }
}

fn expected_golden_error_content_size(path: &Path) -> u64 {
    match fixture_name(path) {
        "off0.bin.zst" | "truncated_huff_state.zst" | "zeroSeq_extraneous.zst" => {
            ZSTD_CONTENTSIZE_UNKNOWN
        }
        name => panic!("unexpected golden decompression error fixture: {name}"),
    }
}

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

fn decode_with_upstream(path: &Path) -> Option<Vec<u8>> {
    let upstream = upstream_zstd()?;
    let out = Command::new(upstream)
        .args(["-d", "-c", "-q"])
        .arg(path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(out.stdout)
}

fn upstream_rejects(path: &Path) -> Option<bool> {
    let upstream = upstream_zstd()?;
    let out = Command::new(upstream)
        .args(["-d", "-c", "-q"])
        .arg(path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .ok()?;
    Some(!out.success())
}

fn decode_frame_with_upstream(frame: &[u8]) -> Option<Vec<u8>> {
    let upstream = upstream_zstd()?;
    let mut child = Command::new(upstream)
        .args(["-d", "-c", "-q"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    child.stdin.take()?.write_all(frame).ok()?;
    let out = child.wait_with_output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(out.stdout)
}

fn compress_with_rust(src: &[u8]) -> Result<Vec<u8>, String> {
    let mut dst = vec![0u8; ZSTD_compressBound(src.len()).max(64)];
    let out = ZSTD_compress(&mut dst, src, 1);
    if ERR_isError(out) {
        return Err(ERR_getErrorName(out).to_string());
    }
    dst.truncate(out);
    Ok(dst)
}

fn decode_with_rust(src: &[u8]) -> Result<Vec<u8>, String> {
    let declared = ZSTD_findDecompressedSize(src);
    if declared == ZSTD_CONTENTSIZE_ERROR {
        return Err(format!(
            "bad frame size metadata: {}",
            ERR_getErrorName(declared as usize)
        ));
    }
    let cap = if declared == ZSTD_CONTENTSIZE_UNKNOWN {
        src.len().saturating_mul(64).max(1)
    } else {
        declared as usize
    };
    let mut dst = vec![0u8; cap];
    let out = ZSTD_decompress(&mut dst, src);
    if ERR_isError(out) {
        return Err(ERR_getErrorName(out).to_string());
    }
    dst.truncate(out);
    Ok(dst)
}

#[test]
fn upstream_golden_compression_inputs_roundtrip_and_cross_decode() {
    for path in fixture_paths("golden-compression") {
        let src = fs::read(&path).expect("fixture bytes");
        let compressed = std::panic::catch_unwind(|| compress_with_rust(&src))
            .unwrap_or_else(|_| panic!("{} triggered a compressor panic", path.display()))
            .unwrap_or_else(|e| panic!("{} should compress successfully: {e}", path.display()));

        let decoded = decode_with_rust(&compressed).unwrap_or_else(|e| {
            panic!(
                "{} rust-compressed frame should decode: {e}",
                path.display()
            )
        });
        assert_eq!(
            decoded,
            src,
            "{} rust-compressed roundtrip diverged from source bytes",
            path.display()
        );
        assert_eq!(
            ZSTD_getFrameContentSize(&compressed),
            src.len() as u64,
            "{} rust-compressed frame should declare the source size",
            path.display()
        );
        assert_eq!(
            ZSTD_findDecompressedSize(&compressed),
            src.len() as u64,
            "{} rust-compressed frame size discovery diverged from source size",
            path.display()
        );

        if let Some(upstream_decoded) = decode_frame_with_upstream(&compressed) {
            assert_eq!(
                upstream_decoded,
                src,
                "{} rust-compressed frame diverged from upstream zstd decode",
                path.display()
            );
        }
    }
}

#[test]
fn upstream_golden_decompression_frames_decode_successfully() {
    for path in fixture_paths("golden-decompression") {
        let src = fs::read(&path).expect("fixture bytes");
        let expected_content_size = expected_golden_content_size(&path);
        assert_eq!(
            ZSTD_getFrameContentSize(&src),
            expected_content_size,
            "{} frame content size metadata diverged from upstream golden expectation",
            path.display()
        );
        assert_eq!(
            ZSTD_findDecompressedSize(&src),
            expected_content_size,
            "{} decompressed size discovery diverged from upstream golden expectation",
            path.display()
        );

        let decoded = std::panic::catch_unwind(|| decode_with_rust(&src))
            .unwrap_or_else(|_| panic!("{} triggered a decoder panic", path.display()))
            .unwrap_or_else(|e| panic!("{} should decode successfully: {e}", path.display()));
        if let Some(expected) = decode_with_upstream(&path) {
            assert_eq!(
                decoded,
                expected,
                "{} decoded bytes diverged from upstream zstd",
                path.display()
            );
        }
        assert_eq!(
            decoded,
            expected_golden_output(&path),
            "{} decoded bytes diverged from vendored golden expectation",
            path.display()
        );
    }
}

#[test]
fn upstream_golden_error_frames_are_rejected() {
    for path in fixture_paths("golden-decompression-errors") {
        let src = fs::read(&path).expect("fixture bytes");
        if let Some(rejected) = upstream_rejects(&path) {
            assert!(
                rejected,
                "{} should be rejected by upstream zstd",
                path.display()
            );
        }
        let expected_content_size = expected_golden_error_content_size(&path);
        assert_eq!(
            ZSTD_getFrameContentSize(&src),
            expected_content_size,
            "{} frame content size metadata diverged from upstream golden error expectation",
            path.display()
        );
        assert_eq!(
            ZSTD_findDecompressedSize(&src),
            expected_content_size,
            "{} decompressed size discovery diverged from upstream golden error expectation",
            path.display()
        );
        let mut dst = vec![0u8; src.len().saturating_mul(64).max(64)];
        let rc = ZSTD_decompress(&mut dst, &src);
        assert!(
            ERR_isError(rc),
            "{} should have been rejected, but decode returned {rc}",
            path.display()
        );
    }
}
