use std::env;
use std::path::PathBuf;
use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    let path = env::args().nth(1).expect("path");
    let level: i32 = env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let seconds: u32 = env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let zstd = upstream_zstd();

    let status = match Command::new(&zstd)
        .args(bench_args(level, seconds))
        .arg("--")
        .arg(&path)
        .status()
    {
        Ok(status) => status,
        Err(err) => {
            eprintln!("failed to run upstream zstd {:?}: {err}", zstd);
            return ExitCode::FAILURE;
        }
    };
    if !status.success() {
        eprintln!("upstream zstd benchmark failed with status {status}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn upstream_zstd() -> PathBuf {
    if let Ok(path) = env::var("ZSTD_BIN") {
        return PathBuf::from(path);
    }
    for path in ["zstd/programs/zstd", "zstd/zstd"] {
        let path = PathBuf::from(path);
        if path.is_file() {
            return path;
        }
    }
    PathBuf::from("zstd")
}

fn bench_args(level: i32, seconds: u32) -> Vec<String> {
    let mut args = Vec::new();
    if level < 0 {
        args.push(format!("--fast={}", level.unsigned_abs()));
        args.push("-b".to_string());
    } else {
        if level > 19 {
            args.push("--ultra".to_string());
        }
        args.push(format!("-b{level}"));
    }
    args.push(format!("-i{seconds}"));
    args
}
