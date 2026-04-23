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
use zstd_pure_rs::compress::zstd_compress::{
    ZSTD_CCtx_setFormat, ZSTD_FrameParameters, ZSTD_compress_advanced, ZSTD_compressBound,
    ZSTD_createCCtx, ZSTD_getCParams, ZSTD_parameters,
};
use zstd_pure_rs::compress::zstd_compress_sequences::{ZSTD_btlazy2, ZSTD_fast};
use zstd_pure_rs::decompress::zstd_decompress::{
    ZSTD_DCtx_setFormat, ZSTD_decompress, ZSTD_decompressStream, ZSTD_decompress_usingDict,
    ZSTD_findDecompressedSize, ZSTD_findFrameSizeInfo, ZSTD_format_e, ZSTD_CONTENTSIZE_ERROR,
    ZSTD_CONTENTSIZE_UNKNOWN,
};
use zstd_pure_rs::decompress::zstd_decompress_block::ZSTD_DCtx;

/// Mirror of the most-used `zstd` flags. The subset is kept small on
/// purpose — we add flags as the matching features land.
#[derive(Parser, Debug)]
#[command(
    name = "zstd",
    about = "Pure-Rust port of the zstd CLI — compression, decompression, dict support, and streaming",
    version,
    long_version = concat!(
        env!("CARGO_PKG_VERSION"),
        " (pure-rust port of libzstd ",
        "1.6.0",  // matches ZSTD_VERSION_STRING; gated by `cli_long_version_matches_library_ZSTD_VERSION_STRING` test
        ")",
    ),
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

    /// Write output to an explicit file path (mirrors upstream's `-o`).
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

    /// Emit / expect magicless-format frames (`ZSTD_f_zstd1_magicless`):
    /// skip the 4-byte magic prefix on compression, and require the
    /// same format on decompression. Saves 4 bytes/frame but breaks
    /// interoperability with standard zstd tools.
    #[arg(long = "magicless")]
    magicless: bool,

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

fn decompress_bytes(src: &[u8], dict: Option<&[u8]>, magicless: bool) -> Result<Vec<u8>, String> {
    // Size the output buffer from frame metadata:
    //   1. `ZSTD_findDecompressedSize` sums declared FCS across frames
    //      when present — exact.
    //   2. Otherwise, walk frames via `ZSTD_findFrameSizeInfo` to get
    //      `decompressedBound = nbBlocks * blockSizeMax` — a tight
    //      upper bound per frame.
    //   3. Fall back to a 32× src estimate only if no bound is
    //      available (shouldn't happen for well-formed zstd streams).
    let format = if magicless {
        ZSTD_format_e::ZSTD_f_zstd1_magicless
    } else {
        ZSTD_format_e::ZSTD_f_zstd1
    };
    // Magicless frames don't have a content-size probe-friendly path
    // via the plain `ZSTD_findDecompressedSize` helper (it hardcodes
    // zstd1 format). Fall back to a size-hint estimate in that case.
    let dst_size = if magicless {
        src.len().saturating_mul(32)
    } else {
        let declared = ZSTD_findDecompressedSize(src);
        if declared == ZSTD_CONTENTSIZE_ERROR {
            return Err(format!(
                "invalid input: {}",
                ERR_getErrorName(declared as usize)
            ));
        }
        if declared != ZSTD_CONTENTSIZE_UNKNOWN {
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
        }
    };

    let mut dst = vec![0u8; dst_size.max(1)];
    let out = if magicless {
        // Magicless decode must route through a format-aware dctx.
        let mut dctx = ZSTD_DCtx::new();
        let _ = ZSTD_DCtx_setFormat(&mut dctx, format);
        if let Some(d) = dict {
            ZSTD_decompress_usingDict(&mut dctx, &mut dst, src, d)
        } else {
            // Single-frame streaming decode threads dctx.format.
            let mut in_pos = 0usize;
            let mut out_pos = 0usize;
            let mut hint =
                ZSTD_decompressStream(&mut dctx, &mut dst, &mut out_pos, src, &mut in_pos);
            while hint != 0 && !ERR_isError(hint) {
                hint = ZSTD_decompressStream(&mut dctx, &mut dst, &mut out_pos, &[], &mut 0usize);
            }
            if ERR_isError(hint) {
                hint
            } else {
                out_pos
            }
        }
    } else if let Some(d) = dict {
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
    magicless: bool,
) -> Result<Vec<u8>, String> {
    let bound = ZSTD_compressBound(src.len());
    if ERR_isError(bound) {
        return Err(format!("compressBound error: {}", ERR_getErrorName(bound)));
    }
    let mut dst = vec![0u8; bound.max(32)];
    let format = if magicless {
        ZSTD_format_e::ZSTD_f_zstd1_magicless
    } else {
        ZSTD_format_e::ZSTD_f_zstd1
    };
    let n = if let Some(d) = dict {
        // Dict path: route through `ZSTD_compress_advanced` so that
        // `--check` threads through as `fParams.checksumFlag` and the
        // final frame carries the XXH64 trailer alongside the dict-
        // back-refs. The cctx.format slot picks up magicless.
        let mut cctx = ZSTD_createCCtx().ok_or("cctx alloc failed")?;
        if magicless {
            let _ = ZSTD_CCtx_setFormat(&mut cctx, ZSTD_format_e::ZSTD_f_zstd1_magicless);
        }
        let mut cp = ZSTD_getCParams(level, src.len() as u64, d.len());
        cp.strategy = cp.strategy.clamp(ZSTD_fast, ZSTD_btlazy2);
        let params = ZSTD_parameters {
            cParams: cp,
            fParams: ZSTD_FrameParameters {
                contentSizeFlag: 1,
                checksumFlag: if checksum { 1 } else { 0 },
                noDictIDFlag: 0,
            },
        };
        ZSTD_compress_advanced(&mut cctx, &mut dst, src, d, params)
    } else {
        // Route through the same CCtx-based public API path as the
        // dict case so CLI behavior stays aligned with the translated
        // `ZSTD_compress_advanced()` orchestration.
        let mut cctx = ZSTD_createCCtx().ok_or("cctx alloc failed")?;
        if magicless {
            let _ = ZSTD_CCtx_setFormat(&mut cctx, format);
        }
        let mut cp = ZSTD_getCParams(level, src.len() as u64, 0);
        cp.strategy = cp.strategy.clamp(ZSTD_fast, ZSTD_btlazy2);
        let params = ZSTD_parameters {
            cParams: cp,
            fParams: ZSTD_FrameParameters {
                contentSizeFlag: 1,
                checksumFlag: if checksum { 1 } else { 0 },
                noDictIDFlag: 1,
            },
        };
        ZSTD_compress_advanced(&mut cctx, &mut dst, src, &[], params)
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
            decompress_bytes(&src, dict_ref, cli.magicless)?
        } else {
            compress_bytes(&src, cli.level, dict_ref, cli.check, cli.magicless)?
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
            let verb = if cli.decompress {
                "decompressed"
            } else {
                "compressed"
            };
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
