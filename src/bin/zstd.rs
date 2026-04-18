//! `zstd` CLI binary — pure-Rust port of a subset of upstream's
//! `zstd/programs/zstdcli.c`. Supports compression + decompression
//! of `.zst` files end-to-end, raw-content dictionaries via `-D`,
//! XXH64 checksum trailers via `--check`, verbose ratio reporting
//! via `-v`, and stdin/stdout streaming via `-`.

use clap::Parser;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use zstd_pure_rs::common::error::{ERR_getErrorName, ERR_isError};
use zstd_pure_rs::compress::match_state::ZSTD_compressionParameters;
use zstd_pure_rs::compress::zstd_compress::{
    ZSTD_compressBound, ZSTD_compressFrame_fast, ZSTD_compress_usingDict, ZSTD_createCCtx,
    ZSTD_getCParams, ZSTD_FrameParameters,
};
use zstd_pure_rs::compress::zstd_compress_sequences::{ZSTD_btlazy2, ZSTD_fast};
use zstd_pure_rs::decompress::zstd_decompress::{
    ZSTD_decompress, ZSTD_decompress_usingDict, ZSTD_findDecompressedSize,
    ZSTD_findFrameSizeInfo, ZSTD_format_e, ZSTD_CONTENTSIZE_ERROR, ZSTD_CONTENTSIZE_UNKNOWN,
};
use zstd_pure_rs::decompress::zstd_decompress_block::ZSTD_DCtx;

/// Mirror of the most-used `zstd` flags. The subset is kept small on
/// purpose — we add flags as the matching features land.
#[derive(Parser, Debug)]
#[command(
    name = "zstd",
    about = "Pure-Rust port of the zstd CLI — compression, decompression, dict support, and streaming",
    version,
    disable_help_subcommand = true
)]
struct Cli {
    /// Decompress the input (default is compress).
    #[arg(short = 'd', long = "decompress")]
    decompress: bool,

    /// Write output to stdout (even when a file is given).
    #[arg(short = 'c', long = "stdout")]
    stdout: bool,

    /// Force overwrite of existing files.
    #[arg(short = 'f', long = "force")]
    force: bool,

    /// Quiet mode.
    #[arg(short = 'q', long = "quiet")]
    quiet: bool,

    /// Verbose mode — prints compression ratio on each file.
    #[arg(short = 'v', long = "verbose")]
    verbose: bool,

    /// Read from stdin explicitly (mirrors upstream's `-`).
    #[arg(short = 'o', long = "output-file")]
    output_file: Option<PathBuf>,

    /// Compression level (1..=22). Default is 3.
    #[arg(long = "level", short = 'L', default_value_t = 3)]
    level: i32,

    /// Path to a raw-content dictionary. Must be the same on the
    /// compressor and decompressor.
    #[arg(short = 'D', long = "dict")]
    dict: Option<PathBuf>,

    /// Add / require an XXH64 content checksum trailer on compression.
    /// Decompression validates when present regardless of this flag.
    #[arg(long = "check")]
    check: bool,

    /// Explicit opposite of `--check` (present for upstream CLI
    /// compatibility; currently the default).
    #[arg(long = "no-check", conflicts_with = "check")]
    no_check: bool,

    /// Input files. Use `-` for stdin.
    files: Vec<PathBuf>,
}

fn read_input(path: &Path) -> io::Result<Vec<u8>> {
    if path == Path::new("-") {
        let mut buf = Vec::new();
        io::stdin().read_to_end(&mut buf)?;
        Ok(buf)
    } else {
        fs::read(path)
    }
}

fn infer_output_path(input: &Path, decompress: bool) -> Option<PathBuf> {
    if decompress {
        // Strip `.zst` / `.zstd` extension for decompression.
        let s = input.to_string_lossy();
        for ext in [".zst", ".zstd"] {
            if let Some(stripped) = s.strip_suffix(ext) {
                return Some(PathBuf::from(stripped));
            }
        }
        None
    } else {
        Some(PathBuf::from(format!("{}.zst", input.display())))
    }
}

fn write_output(path: Option<&Path>, force: bool, data: &[u8]) -> io::Result<()> {
    match path {
        None => io::stdout().write_all(data),
        Some(p) => {
            if p.exists() && !force {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!("{} already exists; use -f to overwrite", p.display()),
                ));
            }
            fs::write(p, data)
        }
    }
}

fn decompress_bytes(src: &[u8], dict: Option<&[u8]>) -> Result<Vec<u8>, String> {
    // Size the output buffer from frame metadata:
    //   1. `ZSTD_findDecompressedSize` sums declared FCS across frames
    //      when present — exact.
    //   2. Otherwise, walk frames via `ZSTD_findFrameSizeInfo` to get
    //      `decompressedBound = nbBlocks * blockSizeMax` — a tight
    //      upper bound per frame.
    //   3. Fall back to a 32× src estimate only if no bound is
    //      available (shouldn't happen for well-formed zstd streams).
    let declared = ZSTD_findDecompressedSize(src);
    if declared == ZSTD_CONTENTSIZE_ERROR {
        return Err(format!(
            "invalid input: {}",
            ERR_getErrorName(declared as usize)
        ));
    }

    let dst_size = if declared != ZSTD_CONTENTSIZE_UNKNOWN {
        declared as usize
    } else {
        // Walk frames and sum decompressedBound.
        let mut bound: u64 = 0;
        let mut cursor = src;
        while !cursor.is_empty() {
            let info = ZSTD_findFrameSizeInfo(cursor, ZSTD_format_e::ZSTD_f_zstd1);
            if ERR_isError(info.compressedSize) {
                return Err(format!(
                    "frame walk failed: {}",
                    ERR_getErrorName(info.compressedSize)
                ));
            }
            bound = bound.saturating_add(info.decompressedBound);
            cursor = &cursor[info.compressedSize..];
        }
        bound as usize
    };

    let mut dst = vec![0u8; dst_size.max(1)];
    let out = if let Some(d) = dict {
        let mut dctx = ZSTD_DCtx::new();
        ZSTD_decompress_usingDict(&mut dctx, &mut dst, src, d)
    } else {
        ZSTD_decompress(&mut dst, src)
    };
    if ERR_isError(out) {
        return Err(ERR_getErrorName(out).to_string());
    }
    dst.truncate(out);
    Ok(dst)
}

fn compress_bytes(
    src: &[u8],
    level: i32,
    dict: Option<&[u8]>,
    checksum: bool,
) -> Result<Vec<u8>, String> {
    let bound = ZSTD_compressBound(src.len());
    if ERR_isError(bound) {
        return Err(format!("compressBound error: {}", ERR_getErrorName(bound)));
    }
    let mut dst = vec![0u8; bound.max(32)];
    let n = if let Some(d) = dict {
        // Dict path: checksum not yet routed through usingDict.
        let mut cctx = ZSTD_createCCtx().ok_or("cctx alloc failed")?;
        ZSTD_compress_usingDict(&mut cctx, &mut dst, src, d, level)
    } else {
        // Use compressFrame_fast directly so we can set the checksum
        // flag. Equivalent to ZSTD_compress otherwise.
        let mut cp: ZSTD_compressionParameters = ZSTD_getCParams(level, src.len() as u64, 0);
        cp.strategy = cp.strategy.clamp(ZSTD_fast, ZSTD_btlazy2);
        let fp = ZSTD_FrameParameters {
            contentSizeFlag: 1,
            checksumFlag: if checksum { 1 } else { 0 },
            noDictIDFlag: 1,
        };
        ZSTD_compressFrame_fast(&mut dst, src, cp, fp)
    };
    if ERR_isError(n) {
        return Err(ERR_getErrorName(n).to_string());
    }
    dst.truncate(n);
    Ok(dst)
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();

    let inputs = if cli.files.is_empty() {
        vec![PathBuf::from("-")]
    } else {
        cli.files.clone()
    };

    // Load the dict once (if provided) — used by every file.
    let dict_bytes: Option<Vec<u8>> = cli
        .dict
        .as_ref()
        .map(|p| fs::read(p).map_err(|e| format!("dict {}: {e}", p.display())))
        .transpose()?;
    let dict_ref = dict_bytes.as_deref();

    for input in &inputs {
        let src = read_input(input).map_err(|e| format!("{}: {e}", input.display()))?;
        let dst = if cli.decompress {
            decompress_bytes(&src, dict_ref)?
        } else {
            compress_bytes(&src, cli.level, dict_ref, cli.check)?
        };
        let out_path: Option<PathBuf> = if cli.stdout || input == Path::new("-") {
            cli.output_file.clone()
        } else {
            cli.output_file
                .clone()
                .or_else(|| infer_output_path(input, cli.decompress))
        };
        let out_label = out_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<stdout>".into());
        write_output(out_path.as_deref(), cli.force, &dst)
            .map_err(|e| format!("{out_label}: {e}"))?;
        if !cli.quiet && (out_path.is_some() || cli.verbose) {
            let verb = if cli.decompress { "decompressed" } else { "compressed" };
            let ratio = if !cli.decompress && !src.is_empty() {
                format!(" ({:.2}%)", (dst.len() as f64 * 100.0) / (src.len() as f64))
            } else {
                String::new()
            };
            eprintln!(
                "{}: {verb} {} -> {} bytes{ratio}",
                input.display(),
                src.len(),
                dst.len()
            );
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("zstd: {e}");
            ExitCode::from(1)
        }
    }
}
