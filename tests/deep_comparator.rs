//! Ignored integration test that wires the code-complexity-comparator
//! into the repo as an executable parity gate.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn comparator_bin() -> PathBuf {
    std::env::var_os("ZSTD_PURE_RS_CCC_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("/data/henriksson/github/claude/code-complexity-comparator/target/release/ccc-rs")
        })
}

fn workdir() -> PathBuf {
    std::env::var_os("ZSTD_PURE_RS_CCC_WORKDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("zstd-pure-rs-deep-comparator"))
}

#[test]
#[ignore = "slow parity harness over translated decompression modules"]
fn ccc_missing_name_report_is_zero_for_decompression_tree() {
    let ccc = comparator_bin();
    if !ccc.exists() {
        eprintln!("ccc-rs binary not found at {}", ccc.display());
        return;
    }

    let workdir = workdir();
    fs::create_dir_all(&workdir).expect("create comparator workdir");
    let rust_json = workdir.join("rust.json");
    let c_json = workdir.join("c.json");
    let mapping = repo_root().join("ccc_mapping.toml");

    let rust_status = Command::new(&ccc)
        .current_dir(repo_root())
        .args([
            "analyze",
            "src/decompress",
            "-l",
            "rust",
            "--recurse",
            "-o",
            rust_json.to_str().unwrap(),
        ])
        .status()
        .expect("run rust analyze");
    assert!(rust_status.success(), "rust analyze failed");

    let c_status = Command::new(&ccc)
        .current_dir(repo_root())
        .args([
            "analyze",
            "zstd/lib/decompress",
            "-l",
            "c",
            "--recurse",
            "-o",
            c_json.to_str().unwrap(),
        ])
        .status()
        .expect("run c analyze");
    assert!(c_status.success(), "c analyze failed");

    let out = Command::new(&ccc)
        .current_dir(repo_root())
        .args([
            "missing",
            rust_json.to_str().unwrap(),
            c_json.to_str().unwrap(),
            "--mapping",
            mapping.to_str().unwrap(),
        ])
        .output()
        .expect("run ccc missing");
    assert!(
        out.status.success(),
        "ccc missing failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("utf8 ccc output");
    assert!(
        stdout.contains("Missing in Rust (0):"),
        "expected zero missing C functions, got:\n{stdout}"
    );
}
