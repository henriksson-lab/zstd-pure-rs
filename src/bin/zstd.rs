//! `zstd` CLI binary — pure-Rust port of a subset of upstream's
//! `zstd/programs/zstdcli.c`. Supports compression + decompression
//! of `.zst` files end-to-end, raw-content dictionaries via `-D`,
//! XXH64 checksum trailers via `--check` / `--no-check`, verbose
//! ratio reporting via `-v`, and buffered stdin/stdout via `-`.

use clap::Parser;
use std::ffi::OsString;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use zstd_pure_rs::common::error::{ERR_getErrorName, ERR_isError};
use zstd_pure_rs::common::xxhash::XXH64_state_t;
use zstd_pure_rs::compress::zstd_compress::{
    ZSTD_CCtx_setFormat, ZSTD_FrameParameters, ZSTD_compressBound, ZSTD_compress_advanced,
    ZSTD_createCCtx, ZSTD_getCParams, ZSTD_parameters,
};
use zstd_pure_rs::decompress::zstd_decompress::{
    ZSTD_DCtx_setFormat, ZSTD_decompress, ZSTD_decompressDCtx, ZSTD_decompress_usingDict,
    ZSTD_findFrameSizeInfo, ZSTD_format_e, ZSTD_CONTENTSIZE_ERROR, ZSTD_MAGICNUMBER,
    ZSTD_MAGIC_SKIPPABLE_MASK, ZSTD_MAGIC_SKIPPABLE_START,
};
use zstd_pure_rs::decompress::zstd_decompress_block::{
    ZSTD_DCtx, ZSTD_buildDefaultSeqTables, ZSTD_decoder_entropy_rep,
};

/// Mirror of the most-used `zstd` flags. The subset is kept small on
/// purpose — we add flags as the matching features land.
#[derive(Parser, Debug)]
#[command(
    name = "zstd",
    about = "Pure-Rust port of the zstd CLI — compression, decompression, dict support, and buffered stdin/stdout",
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

    /// Compression level; -L is a local alias, while upstream-style -N forms are also accepted. Out-of-range values are clamped. Default is 3.
    #[arg(long = "level", short = 'L', default_value_t = 3)]
    level: i32,

    /// Path to a raw-content dictionary. Must be the same on the
    /// compressor and decompressor.
    #[arg(short = 'D', long = "dict")]
    dict: Option<PathBuf>,

    /// Add an XXH64 content checksum trailer on compression, and validate it on decompression when present.
    /// This is the upstream CLI default.
    #[arg(long = "check")]
    check: bool,

    /// Disable the default XXH64 content checksum trailer.
    #[arg(long = "no-check")]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputDirective {
    Stdout,
    File,
}

fn normalize_level_short_args<I>(args: I) -> Vec<OsString>
where
    I: IntoIterator<Item = OsString>,
{
    let mut options_done = false;
    let mut normalized = Vec::new();
    for arg in args {
        let mut emit_original = true;
        if options_done {
            normalized.push(arg);
            continue;
        }
        if let Some(s) = arg.to_str() {
            if s == "--" {
                options_done = true;
            } else if s.len() > 1 && s.starts_with('-') && !s.starts_with("--") {
                let shorts = &s[1..];
                if !matches!(shorts.chars().next(), Some('D' | 'L' | 'o')) {
                    let mut replacements = Vec::new();
                    let mut chunk = String::new();
                    let mut digits = String::new();
                    let mut saw_digits = false;
                    for c in shorts.chars() {
                        if c.is_ascii_digit() {
                            if !chunk.is_empty() {
                                replacements.push(OsString::from(format!("-{chunk}")));
                                chunk.clear();
                            }
                            digits.push(c);
                            saw_digits = true;
                        } else {
                            if !digits.is_empty() {
                                replacements.push(OsString::from(format!("--level={digits}")));
                                digits.clear();
                            }
                            chunk.push(c);
                        }
                    }
                    if !digits.is_empty() {
                        replacements.push(OsString::from(format!("--level={digits}")));
                    }
                    if !chunk.is_empty() {
                        replacements.push(OsString::from(format!("-{chunk}")));
                    }
                    if saw_digits {
                        normalized.extend(replacements);
                        emit_original = false;
                    }
                }
            }
        }
        if emit_original {
            normalized.push(arg);
        }
    }
    normalized
}

fn last_checksum_directive(args: &[OsString]) -> Option<bool> {
    let mut last = None;
    for arg in args.iter().skip(1) {
        let Some(arg) = arg.to_str() else {
            continue;
        };
        if arg == "--" {
            break;
        }
        if arg == "--check" {
            last = Some(true);
        } else if arg == "--no-check" {
            last = Some(false);
        }
    }
    last
}

fn reject_attached_dict_arg(args: &[OsString]) -> Result<(), String> {
    for arg in args.iter().skip(1) {
        let Some(arg) = arg.to_str() else {
            continue;
        };
        if arg == "--" {
            break;
        }
        if arg.starts_with("-D") && arg != "-D" {
            return Err(format!(
                "{arg}: dictionary path must be passed as a separate argument after -D"
            ));
        }
    }
    Ok(())
}

fn last_output_directive(args: &[OsString]) -> Option<OutputDirective> {
    let mut last = None;
    let mut i = 1usize;
    while i < args.len() {
        let Some(arg) = args[i].to_str() else {
            i += 1;
            continue;
        };
        if arg == "--" {
            break;
        }
        if arg == "--stdout" || arg == "-c" {
            last = Some(OutputDirective::Stdout);
        } else if arg == "--output-file" || arg == "-o" {
            last = Some(OutputDirective::File);
            i += 1;
        } else if arg.starts_with("--output-file=") {
            last = Some(OutputDirective::File);
        } else if let Some(shorts) = arg.strip_prefix('-') {
            if !shorts.starts_with('-') && !shorts.chars().all(|c| c.is_ascii_digit()) {
                for c in shorts.chars() {
                    match c {
                        'c' => last = Some(OutputDirective::Stdout),
                        'o' => {
                            last = Some(OutputDirective::File);
                            break;
                        }
                        'D' | 'L' => break,
                        _ => {}
                    }
                }
            }
        }
        i += 1;
    }
    last
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

fn output_path_for_input(
    input: &Path,
    cli: &Cli,
    effective_output_file: Option<&PathBuf>,
) -> Result<Option<PathBuf>, String> {
    if let Some(output_file) = effective_output_file {
        return Ok(Some(output_file.clone()));
    }
    if cli.stdout || input == Path::new("-") {
        return Ok(None);
    }
    match infer_output_path(input, cli.decompress) {
        Some(path) => Ok(Some(path)),
        None => Err(format!(
            "{}: unknown suffix -- expected .zst or .zstd; use -c or -o to select output",
            input.display()
        )),
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

fn decompressed_bound_for_format(src: &[u8], format: ZSTD_format_e) -> Result<usize, String> {
    let mut bound: u64 = 0;
    let mut cursor = src;
    while !cursor.is_empty() {
        let info = ZSTD_findFrameSizeInfo(cursor, format);
        if ERR_isError(info.compressedSize) || info.decompressedBound == ZSTD_CONTENTSIZE_ERROR {
            return Err(format!(
                "frame walk failed: {}",
                ERR_getErrorName(info.compressedSize)
            ));
        }
        bound = bound.saturating_add(info.decompressedBound);
        cursor = &cursor[info.compressedSize..];
    }
    usize::try_from(bound).map_err(|_| "decompressed bound exceeds addressable memory".to_string())
}

fn strip_decode_checksums(src: &[u8], format: ZSTD_format_e) -> Result<Vec<u8>, String> {
    let mut stripped = Vec::with_capacity(src.len());
    let mut cursor = src;
    while !cursor.is_empty() {
        let info = ZSTD_findFrameSizeInfo(cursor, format);
        if ERR_isError(info.compressedSize) || info.decompressedBound == ZSTD_CONTENTSIZE_ERROR {
            return Err(format!(
                "frame walk failed: {}",
                ERR_getErrorName(info.compressedSize)
            ));
        }
        let frame = &cursor[..info.compressedSize];
        let is_skippable = format == ZSTD_format_e::ZSTD_f_zstd1
            && frame.len() >= 4
            && (u32::from_le_bytes([frame[0], frame[1], frame[2], frame[3]])
                & ZSTD_MAGIC_SKIPPABLE_MASK)
                == ZSTD_MAGIC_SKIPPABLE_START;
        if is_skippable {
            stripped.extend_from_slice(frame);
        } else {
            let fhd_index = if format == ZSTD_format_e::ZSTD_f_zstd1 {
                4
            } else {
                0
            };
            if frame.len() <= fhd_index {
                return Err("frame walk failed: Src size is incorrect".to_string());
            }
            let has_checksum = frame[fhd_index] & 0b0000_0100 != 0;
            if has_checksum {
                if frame.len() < 4 {
                    return Err("frame walk failed: Src size is incorrect".to_string());
                }
                let frame_start = stripped.len();
                stripped.extend_from_slice(&frame[..frame.len() - 4]);
                stripped[frame_start + fhd_index] &= !0b0000_0100;
            } else {
                stripped.extend_from_slice(frame);
            }
        }
        cursor = &cursor[info.compressedSize..];
    }
    Ok(stripped)
}

fn decompress_bytes(
    src: &[u8],
    dict: Option<&[u8]>,
    magicless: bool,
    ignore_checksum: bool,
) -> Result<Vec<u8>, String> {
    // Size the output buffer from format-aware frame metadata. When a
    // frame declares FCS the walker returns the exact size; otherwise
    // it returns nbBlocks * blockSizeMax for that frame. This matters
    // for magicless frames because the default `ZSTD_findDecompressedSize`
    // path assumes a 4-byte zstd1 magic prefix.
    let format = if magicless {
        ZSTD_format_e::ZSTD_f_zstd1_magicless
    } else {
        ZSTD_format_e::ZSTD_f_zstd1
    };
    let decoded_src;
    let src = if ignore_checksum {
        decoded_src = strip_decode_checksums(src, format)?;
        decoded_src.as_slice()
    } else {
        src
    };
    let dst_size = decompressed_bound_for_format(src, format)?;

    let mut dst = vec![0u8; dst_size.max(1)];
    let out = if magicless {
        // Magicless decode must route through a format-aware dctx.
        let mut dctx = ZSTD_DCtx::new();
        let _ = ZSTD_DCtx_setFormat(&mut dctx, format);
        if let Some(d) = dict {
            ZSTD_decompress_usingDict(&mut dctx, &mut dst, src, d)
        } else {
            ZSTD_buildDefaultSeqTables(&mut dctx);
            let mut rep = ZSTD_decoder_entropy_rep::default();
            let mut xxh = XXH64_state_t::default();
            ZSTD_decompressDCtx(&mut dctx, &mut rep, &mut xxh, &mut dst, src)
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

fn looks_like_zstd_input(src: &[u8], magicless: bool) -> bool {
    if magicless {
        return true;
    }
    if src.len() < 4 {
        return false;
    }
    let magic = u32::from_le_bytes([src[0], src[1], src[2], src[3]]);
    magic == ZSTD_MAGICNUMBER || (magic & ZSTD_MAGIC_SKIPPABLE_MASK) == ZSTD_MAGIC_SKIPPABLE_START
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
        let cp = ZSTD_getCParams(level, src.len() as u64, d.len());
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
        let cp = ZSTD_getCParams(level, src.len() as u64, 0);
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
    let raw_args: Vec<OsString> = std::env::args_os().collect();
    reject_attached_dict_arg(&raw_args)?;
    let output_directive = last_output_directive(&raw_args);
    let checksum_directive = last_checksum_directive(&raw_args);
    let cli = Cli::parse_from(normalize_level_short_args(raw_args));

    let inputs = if cli.files.is_empty() {
        vec![PathBuf::from("-")]
    } else {
        cli.files.clone()
    };

    let effective_output_file = match output_directive {
        Some(OutputDirective::Stdout) => None,
        Some(OutputDirective::File) | None => cli.output_file.clone(),
    };

    if effective_output_file.is_some() && inputs.len() > 1 {
        return Err("--output-file/-o can only be used with a single input".to_string());
    }

    // Load the dict once (if provided) — used by every file.
    let dict_bytes: Option<Vec<u8>> = cli
        .dict
        .as_ref()
        .map(|p| fs::read(p).map_err(|e| format!("dict {}: {e}", p.display())))
        .transpose()?;
    let dict_ref = dict_bytes.as_deref();

    for input in &inputs {
        let out_path = output_path_for_input(input, &cli, effective_output_file.as_ref())?;
        let src = read_input(input).map_err(|e| format!("{}: {e}", input.display()))?;
        let output_is_stdout = effective_output_file.is_none()
            && (matches!(output_directive, Some(OutputDirective::Stdout))
                || input == Path::new("-"));
        let dst = if cli.decompress {
            if cli.force && output_is_stdout && !looks_like_zstd_input(&src, cli.magicless) {
                src.clone()
            } else {
                decompress_bytes(
                    &src,
                    dict_ref,
                    cli.magicless,
                    checksum_directive == Some(false),
                )?
            }
        } else {
            let checksum = checksum_directive.unwrap_or(true);
            compress_bytes(&src, cli.level, dict_ref, checksum, cli.magicless)?
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
