//! Port of a small, durable subset of upstream `zstd/tests/` golden
//! decompression checks. These fixtures live in the vendored upstream
//! tree, so the tests stay self-contained.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use zstd_pure_rs::common::error::{ERR_getErrorName, ERR_isError};
use zstd_pure_rs::decompress::zstd_decompress::{
    ZSTD_decompress, ZSTD_findDecompressedSize, ZSTD_CONTENTSIZE_ERROR, ZSTD_CONTENTSIZE_UNKNOWN,
};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn upstream_golden_dir(kind: &str) -> PathBuf {
    repo_root().join("zstd/tests").join(kind)
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
fn upstream_golden_decompression_frames_decode_successfully() {
    for entry in fs::read_dir(upstream_golden_dir("golden-decompression")).expect("golden dir") {
        let path = entry.expect("dirent").path();
        let src = fs::read(&path).expect("fixture bytes");
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
        } else {
            assert!(
                !decoded.is_empty()
                    || path.file_name().and_then(|s| s.to_str()) == Some("empty-block.zst"),
                "{} decoded to an unexpected empty output",
                path.display()
            );
        }
    }
}

#[test]
fn upstream_golden_error_frames_are_rejected() {
    for entry in
        fs::read_dir(upstream_golden_dir("golden-decompression-errors")).expect("error dir")
    {
        let path = entry.expect("dirent").path();
        let src = fs::read(&path).expect("fixture bytes");
        let mut dst = vec![0u8; src.len().saturating_mul(64).max(64)];
        let rc = ZSTD_decompress(&mut dst, &src);
        assert!(
            ERR_isError(rc),
            "{} should have been rejected, but decode returned {rc}",
            path.display()
        );
    }
}
