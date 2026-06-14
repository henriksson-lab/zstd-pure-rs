//! `zstd` CLI binary — pure-Rust port of a subset of upstream's
//! `zstd/programs/zstdcli.c`. Supports compression + decompression
//! of `.zst` files end-to-end, raw-content dictionaries via `-D`,
//! XXH64 checksum trailers via `--check` / `--no-check`, verbose
//! ratio reporting via `-v`, and buffered stdin/stdout via `-`.

use clap::{error::ErrorKind, ArgAction, CommandFactory, Parser};
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, ErrorKind as IoErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, ExitCode, Stdio};

use zstd_pure_rs::common::error::{ERR_getErrorName, ERR_isError};
use zstd_pure_rs::common::xxhash::XXH64_state_t;
use zstd_pure_rs::compress::zstd_compress::{
    ZSTD_CCtx_loadDictionary, ZSTD_CCtx_setParameter, ZSTD_CCtx_setPledgedSrcSize,
    ZSTD_CStreamInSize, ZSTD_CStreamOutSize, ZSTD_cParameter, ZSTD_compress2, ZSTD_compressBound,
    ZSTD_compressStream, ZSTD_createCCtx, ZSTD_endStream, ZSTD_forceIgnoreChecksum_e,
    ZSTD_maxCLevel, ZSTD_minCLevel,
};
use zstd_pure_rs::decompress::zstd_decompress::{
    ZSTD_DCtx_setFormat, ZSTD_FrameHeader, ZSTD_decompress, ZSTD_decompressContinue_into_history,
    ZSTD_decompressDCtx, ZSTD_decompress_usingDict, ZSTD_findFrameSizeInfo, ZSTD_format_e,
    ZSTD_getFrameHeader, ZSTD_initDStream, ZSTD_nextSrcSizeToDecompress, ZSTD_resetDStream,
    ZSTD_CONTENTSIZE_ERROR, ZSTD_MAGICNUMBER, ZSTD_MAGIC_SKIPPABLE_MASK,
    ZSTD_MAGIC_SKIPPABLE_START,
};
#[cfg(test)]
use zstd_pure_rs::decompress::zstd_decompress::{
    ZSTD_getFrameContentSize, ZSTD_CONTENTSIZE_UNKNOWN,
};
use zstd_pure_rs::decompress::zstd_decompress_block::{
    ZSTD_DCtx, ZSTD_buildDefaultSeqTables, ZSTD_decoder_entropy_rep,
};

const UPSTREAM_ZSTD_VERSION: &str = "1.6.0";
const UPSTREAM_ZSTD_VERSION_BANNER: &str = "*** Zstandard CLI (64-bit) v1.6.0, by Yann Collet ***";
const FILE_OUTPUT_BUFFER_SIZE: usize = 1 << 20;
const STREAM_DECODE_WINDOW_LIMIT: u64 = 4 << 20;

/// Mirror of the most-used `zstd` flags. The subset is kept small on
/// purpose — we add flags as the matching features land.
#[derive(Parser, Debug)]
#[command(
    name = "zstd",
    about = "Pure-Rust port of the zstd CLI — compression, decompression, dict support, and buffered stdin/stdout",
    version = concat!(
        env!("CARGO_PKG_VERSION"),
        " (pure-rust port of libzstd ",
        "1.6.0",
        ")",
    ),
    long_version = concat!(
        env!("CARGO_PKG_VERSION"),
        " (pure-rust port of libzstd ",
        "1.6.0",  // matches ZSTD_VERSION_STRING; gated by `cli_long_version_matches_library_ZSTD_VERSION_STRING` test
        ")",
    ),
    disable_help_subcommand = true,
    args_override_self = true
)]
struct Cli {
    /// Decompress the input (default is compress).
    #[arg(short = 'd', long = "decompress", alias = "uncompress")]
    decompress: bool,

    /// Display full help and exit.
    #[arg(short = 'H', action = ArgAction::SetTrue, hide = true)]
    help_advanced: bool,

    /// Compress the input.
    #[arg(short = 'z', long = "compress", hide = true)]
    compress: bool,

    /// Test compressed input integrity without writing decoded output.
    #[arg(short = 't', long = "test", hide = true)]
    test: bool,

    /// Write output to stdout (even when a file is given).
    #[arg(short = 'c', long = "stdout")]
    stdout: bool,

    /// Force overwrite of existing files.
    #[arg(short = 'f', long = "force")]
    force: bool,

    /// Keep source files after processing. This port never removes inputs by default.
    #[arg(short = 'k', long = "keep", hide = true)]
    keep: bool,

    /// Remove source files after successful file output.
    #[arg(long = "rm", hide = true)]
    rm: bool,

    /// Gzip compatibility: do not store original filename.
    #[arg(short = 'n', hide = true)]
    no_name: bool,

    /// Copy unrecognized input through unchanged when decompressing.
    #[arg(long = "pass-through")]
    pass_through: bool,

    /// Disable pass-through of unrecognized input when decompressing.
    #[arg(long = "no-pass-through")]
    no_pass_through: bool,

    /// Quiet mode.
    #[arg(short = 'q', long = "quiet", action = ArgAction::Count)]
    quiet: u8,

    /// Verbose mode — prints compression ratio on each file.
    #[arg(short = 'v', long = "verbose", action = ArgAction::Count)]
    verbose: u8,

    /// Write output to an explicit file path (mirrors upstream's `-o`).
    #[arg(short = 'o', value_name = "OUTPUT")]
    output_file: Option<PathBuf>,

    #[arg(long = "__zstd_pure_internal_output_file", hide = true)]
    internal_output_file: Option<PathBuf>,

    /// Internal normalized compression level. Users select it with upstream-style -N forms.
    #[arg(long = "level", default_value_t = 3, hide = true)]
    level: i32,

    /// Internal normalized `--fast[=#]` level.
    #[arg(long = "__zstd_pure_internal_fast", hide = true)]
    fast_level: Option<u32>,

    /// Enable levels beyond 19, up to 22; requires more memory.
    #[arg(long = "ultra")]
    ultra: bool,

    /// Enable long distance matching. The pure-Rust subset treats it as an ultra-level unlock.
    #[arg(long = "long", num_args = 0..=1, require_equals = true, default_missing_value = "27", hide = true)]
    long: Option<String>,

    /// Maximum compression shorthand. The pure-Rust subset maps it to max level.
    #[arg(long = "max", hide = true)]
    max: bool,

    /// Path to a raw-content dictionary. Must be the same on the
    /// compressor and decompressor.
    #[arg(short = 'D', value_name = "DICT")]
    dict: Option<PathBuf>,

    #[arg(long = "__zstd_pure_internal_dict", hide = true)]
    internal_dict: Option<PathBuf>,

    /// Add an XXH64 content checksum trailer on compression, and validate it on decompression when present.
    /// This is the upstream CLI default.
    #[arg(short = 'C', long = "check")]
    check: bool,

    /// Disable the default XXH64 content checksum trailer.
    #[arg(long = "no-check")]
    no_check: bool,

    /// Emit the frame content-size field when known.
    #[arg(long = "content-size", hide = true)]
    content_size: bool,

    /// Do not emit the frame content-size field.
    #[arg(long = "no-content-size", hide = true)]
    no_content_size: bool,

    /// Do not emit a dictionary ID in the frame header.
    #[arg(long = "no-dictID", hide = true)]
    no_dict_id: bool,

    /// Accepted upstream knobs that are no-ops in this single-threaded subset.
    #[arg(long = "sparse", hide = true)]
    sparse: bool,

    #[arg(long = "no-sparse", hide = true)]
    no_sparse: bool,

    #[arg(long = "mmap-dict", hide = true)]
    mmap_dict: bool,

    #[arg(long = "no-mmap-dict", hide = true)]
    no_mmap_dict: bool,

    #[arg(long = "progress", hide = true)]
    progress: bool,

    #[arg(long = "no-progress", hide = true)]
    no_progress: bool,

    #[arg(long = "rsyncable", hide = true)]
    rsyncable: bool,

    #[arg(long = "adapt", num_args = 0..=1, require_equals = true, default_missing_value = "", hide = true)]
    adapt: Option<String>,

    #[arg(long = "format", require_equals = true, hide = true)]
    format: Option<String>,

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OperationDirective {
    Compress,
    Decompress,
    Test,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TerminalDirective {
    Version,
    ShortHelp,
    LongHelp,
}

struct SinkWriter;

impl Write for SinkWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn normalize_level_short_args<I>(args: I) -> Result<Vec<OsString>, String>
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
                    let mut chars = shorts.chars().peekable();
                    while let Some(c) = chars.next() {
                        if c.is_ascii_digit() {
                            if !chunk.is_empty() {
                                replacements.push(OsString::from(format!("-{chunk}")));
                                chunk.clear();
                            }
                            digits.push(c);
                            saw_digits = true;
                        } else {
                            if !digits.is_empty() {
                                let mut level = digits.parse::<u32>().map_err(|_| {
                                    "error: numeric value overflows 32-bit unsigned int".to_string()
                                })?;
                                digits.clear();
                                if matches!(c, 'K' | 'M' | 'G') {
                                    let shift = match c {
                                        'K' => 10,
                                        'M' => 20,
                                        'G' => 30,
                                        _ => unreachable!(),
                                    };
                                    level = level.checked_mul(1u32 << shift).ok_or_else(|| {
                                        "error: numeric value overflows 32-bit unsigned int"
                                            .to_string()
                                    })?;
                                    if chars.peek() == Some(&'i') {
                                        chars.next();
                                    }
                                    if chars.peek() == Some(&'B') {
                                        chars.next();
                                    }
                                    replacements
                                        .push(OsString::from(format!("--level={}", level as i32)));
                                    continue;
                                }
                                replacements
                                    .push(OsString::from(format!("--level={}", level as i32)));
                                digits.clear();
                            }
                            chunk.push(c);
                        }
                    }
                    if !digits.is_empty() {
                        let level = digits.parse::<u32>().map_err(|_| {
                            "error: numeric value overflows 32-bit unsigned int".to_string()
                        })?;
                        replacements.push(OsString::from(format!("--level={}", level as i32)));
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
    Ok(normalized)
}

fn normalize_fast_args<I>(args: I) -> Result<Vec<OsString>, String>
where
    I: IntoIterator<Item = OsString>,
{
    let mut options_done = false;
    let mut normalized = Vec::new();
    for arg in args {
        if options_done {
            normalized.push(arg);
            continue;
        }
        if let Some(s) = arg.to_str() {
            if s == "--" {
                options_done = true;
            } else if s == "--fast" {
                normalized.push(OsString::from("--__zstd_pure_internal_fast=1"));
                continue;
            } else if let Some(value) = s.strip_prefix("--fast=") {
                let (level, _) = read_unsigned_prefix_checked(value)?;
                if level == 0 {
                    return Err(format!("Incorrect parameter: {s}"));
                }
                let max_fast = u32::try_from(-ZSTD_minCLevel()).unwrap_or(0);
                normalized.push(OsString::from(format!(
                    "--__zstd_pure_internal_fast={}",
                    level.min(max_fast)
                )));
                continue;
            }
        }
        normalized.push(arg);
    }
    Ok(normalized)
}

fn read_unsigned_prefix_checked(input: &str) -> Result<(u32, &str), String> {
    let digits_len = input.bytes().take_while(|b| b.is_ascii_digit()).count();
    let mut value = if digits_len == 0 {
        0
    } else {
        input[..digits_len]
            .parse::<u32>()
            .map_err(|_| "error: numeric value overflows 32-bit unsigned int".to_string())?
    };
    let mut rest = &input[digits_len..];
    let shift = match rest.as_bytes().first().copied() {
        Some(b'K') => Some(10),
        Some(b'M') => Some(20),
        Some(b'G') => Some(30),
        _ => None,
    };
    if let Some(shift) = shift {
        value = value
            .checked_mul(1u32 << shift)
            .ok_or_else(|| "error: numeric value overflows 32-bit unsigned int".to_string())?;
        rest = &rest[1..];
        if let Some(after) = rest.strip_prefix('i') {
            rest = after;
        }
        if let Some(after) = rest.strip_prefix('B') {
            rest = after;
        }
    }
    Ok((value, rest))
}

fn normalize_attached_field_args<I>(args: I) -> Result<Vec<OsString>, String>
where
    I: IntoIterator<Item = OsString>,
{
    let mut options_done = false;
    let mut iter = args.into_iter();
    let mut normalized = Vec::new();
    while let Some(arg) = iter.next() {
        if options_done {
            normalized.push(arg);
            continue;
        }
        if let Some(s) = arg.to_str() {
            if s == "--" {
                options_done = true;
            } else if s.len() > 2
                && s.starts_with('-')
                && !s.starts_with("--")
                && s[1..].chars().any(|c| matches!(c, 'D' | 'o'))
            {
                let mut chars = s[1..].chars().peekable();
                while let Some(c) = chars.next() {
                    match c {
                        'D' | 'o' => {
                            if chars.peek() == Some(&'=') {
                                chars.next();
                                let value = chars.collect::<String>();
                                let long = if c == 'D' {
                                    "--__zstd_pure_internal_dict"
                                } else {
                                    "--__zstd_pure_internal_output_file"
                                };
                                normalized.push(OsString::from(format!("{long}={value}")));
                                break;
                            }
                            normalized.push(OsString::from(format!("-{c}")));
                            let value = iter
                                .next()
                                .ok_or_else(|| "error: missing command argument ".to_string())?;
                            if value.to_str().is_some_and(|value| value.starts_with('-')) {
                                return Err(
                                    "error: command cannot be separated from its argument by another command ".into(),
                                );
                            }
                            normalized.push(value);
                        }
                        '-' => return Err("Incorrect parameter: --".into()),
                        _ => normalized.push(OsString::from(format!("-{c}"))),
                    }
                }
                continue;
            }
        }
        normalized.push(arg);
    }
    Ok(normalized)
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
        } else if let Some(shorts) = arg.strip_prefix('-') {
            if !shorts.starts_with('-') && !shorts.chars().all(|c| c.is_ascii_digit()) {
                for c in shorts.chars() {
                    match c {
                        'C' => last = Some(true),
                        'D' | 'L' | 'o' => break,
                        _ => {}
                    }
                }
            }
        }
    }
    last
}

fn raw_display_level(args: &[OsString]) -> i32 {
    let mut display_level = 2i32;
    let mut i = 1usize;
    while i < args.len() {
        let Some(arg) = args[i].to_str() else {
            i += 1;
            continue;
        };
        if arg == "--" {
            break;
        }
        if arg == "--quiet" {
            display_level -= 1;
        } else if arg == "--verbose" {
            display_level += 1;
        } else if matches!(
            arg,
            "-D" | "-L"
                | "-o"
                | "--__zstd_pure_internal_dict"
                | "--level"
                | "--__zstd_pure_internal_output_file"
        ) {
            i += 1;
        } else if arg.starts_with("--__zstd_pure_internal_dict=")
            || arg.starts_with("--level=")
            || arg.starts_with("--__zstd_pure_internal_output_file=")
        {
        } else if let Some(shorts) = arg.strip_prefix('-') {
            if !shorts.starts_with('-') && !shorts.chars().all(|c| c.is_ascii_digit()) {
                for c in shorts.chars() {
                    match c {
                        'q' => display_level -= 1,
                        'v' => display_level += 1,
                        'D' | 'L' | 'o' => break,
                        _ => {}
                    }
                }
            }
        }
        i += 1;
    }
    display_level
}

#[cfg(test)]
fn raw_quiet_level(args: &[OsString]) -> u8 {
    2i32.saturating_sub(raw_display_level(args)) as u8
}

#[cfg(test)]
fn upstream_version_directive(args: &[OsString]) -> Option<(i32, usize)> {
    upstream_terminal_directive(args).and_then(|(directive, display_level, index)| {
        (directive == TerminalDirective::Version).then_some((display_level, index))
    })
}

fn upstream_terminal_directive(args: &[OsString]) -> Option<(TerminalDirective, i32, usize)> {
    let mut display_level = 2i32;
    for (index, arg) in args.iter().enumerate().skip(1) {
        let Some(arg) = arg.to_str() else {
            continue;
        };
        if arg == "--" {
            break;
        }
        match arg {
            "--version" => return Some((TerminalDirective::Version, display_level, index)),
            "--help" => return Some((TerminalDirective::LongHelp, display_level, index)),
            "--quiet" => {
                display_level -= 1;
                continue;
            }
            "--verbose" => {
                display_level += 1;
                continue;
            }
            _ => {}
        }
        if let Some(shorts) = arg.strip_prefix('-') {
            if !shorts.starts_with('-') && !shorts.chars().all(|c| c.is_ascii_digit()) {
                for c in shorts.chars() {
                    match c {
                        'V' => return Some((TerminalDirective::Version, display_level, index)),
                        'h' => return Some((TerminalDirective::ShortHelp, display_level, index)),
                        'H' => return Some((TerminalDirective::LongHelp, display_level, index)),
                        'q' => display_level -= 1,
                        'v' => display_level += 1,
                        'D' | 'L' | 'o' => break,
                        _ => {}
                    }
                }
            }
        }
    }
    None
}

fn print_upstream_help(directive: TerminalDirective) -> Result<(), String> {
    match directive {
        TerminalDirective::ShortHelp => Cli::command().print_help().map_err(|e| e.to_string())?,
        TerminalDirective::LongHelp => Cli::command()
            .print_long_help()
            .map_err(|e| e.to_string())?,
        TerminalDirective::Version => unreachable!("version has a separate printer"),
    }
    println!();
    Ok(())
}

fn print_upstream_version(display_level: i32) {
    if display_level < 2 {
        println!("{UPSTREAM_ZSTD_VERSION}");
    } else {
        println!("{UPSTREAM_ZSTD_VERSION_BANNER}");
        if display_level >= 3 {
            println!("*** supports: zstd, gzip, lz4, lzma, xz ");
        }
    }
}

fn validate_args_before_version_exit(
    args: &[OsString],
    version_index: usize,
) -> Result<(), String> {
    let prefix = args[..=version_index].to_vec();
    reject_level_alias_args(&prefix)?;
    reject_long_equals_field_args(&prefix)?;
    reject_long_field_alias_args(&prefix)?;
    reject_format_field_args(&prefix)?;
    reject_empty_adapt_equals_args(&prefix)?;
    let normalized = normalize_attached_field_args(prefix)?;
    reject_attached_separate_field_args(&normalized)
}

fn last_pass_through_directive(args: &[OsString]) -> Option<bool> {
    let mut last = None;
    for arg in args.iter().skip(1) {
        let Some(arg) = arg.to_str() else {
            continue;
        };
        if arg == "--" {
            break;
        }
        if arg == "--pass-through" {
            last = Some(true);
        } else if arg == "--no-pass-through" {
            last = Some(false);
        }
    }
    last
}

fn last_operation_directive(args: &[OsString]) -> Option<OperationDirective> {
    let mut last = None;
    for arg in args.iter().skip(1) {
        let Some(arg) = arg.to_str() else {
            continue;
        };
        if arg == "--" {
            break;
        }
        match arg {
            "--compress" => last = Some(OperationDirective::Compress),
            "--decompress" | "--uncompress" => last = Some(OperationDirective::Decompress),
            "--test" => last = Some(OperationDirective::Test),
            _ => {
                if let Some(shorts) = arg.strip_prefix('-') {
                    if !shorts.starts_with('-') && !shorts.chars().all(|c| c.is_ascii_digit()) {
                        for c in shorts.chars() {
                            match c {
                                'z' => last = Some(OperationDirective::Compress),
                                'd' => last = Some(OperationDirective::Decompress),
                                't' => last = Some(OperationDirective::Test),
                                'D' | 'L' | 'o' => break,
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
    }
    last
}

fn last_content_size_directive(args: &[OsString]) -> Option<bool> {
    let mut last = None;
    for arg in args.iter().skip(1) {
        let Some(arg) = arg.to_str() else {
            continue;
        };
        if arg == "--" {
            break;
        }
        if arg == "--content-size" {
            last = Some(true);
        } else if arg == "--no-content-size" {
            last = Some(false);
        }
    }
    last
}

fn last_dict_id_directive(args: &[OsString]) -> Option<bool> {
    let mut last = None;
    for arg in args.iter().skip(1) {
        let Some(arg) = arg.to_str() else {
            continue;
        };
        if arg == "--" {
            break;
        }
        if arg == "--no-dictID" {
            last = Some(false);
        }
    }
    last
}

fn last_remove_source_directive(args: &[OsString]) -> Option<bool> {
    let mut last = None;
    for arg in args.iter().skip(1) {
        let Some(arg) = arg.to_str() else {
            continue;
        };
        if arg == "--" {
            break;
        }
        match arg {
            "--rm" => last = Some(true),
            "--keep" => last = Some(false),
            _ => {
                if let Some(shorts) = arg.strip_prefix('-') {
                    if !shorts.starts_with('-') && !shorts.chars().all(|c| c.is_ascii_digit()) {
                        for c in shorts.chars() {
                            match c {
                                'k' => last = Some(false),
                                'D' | 'L' | 'o' => break,
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
    }
    last
}

fn read_adapt_int_field(input: &str) -> Result<(i32, &str), String> {
    let (negative, input) = if let Some(rest) = input.strip_prefix('-') {
        (true, rest)
    } else {
        (false, input)
    };
    let (parsed, rest) = read_unsigned_prefix_checked(input)
        .map_err(|_| "error: numeric value overflows 32-bit int".to_string())?;
    let parsed = parsed as i32;
    if negative {
        Ok((parsed.wrapping_neg(), rest))
    } else {
        Ok((parsed, rest))
    }
}

fn parse_adapt_parameters(input: &str) -> Result<(i32, i32), String> {
    let mut min = ZSTD_minCLevel();
    let mut max = ZSTD_maxCLevel();
    let mut rest = input;
    loop {
        if let Some(value) = rest.strip_prefix("min=") {
            let parsed;
            (parsed, rest) = read_adapt_int_field(value)?;
            min = parsed;
        } else if let Some(value) = rest.strip_prefix("max=") {
            let parsed;
            (parsed, rest) = read_adapt_int_field(value)?;
            max = parsed;
        } else {
            return Err(format!("Incorrect parameter: --adapt={input}"));
        }
        if let Some(after_comma) = rest.strip_prefix(',') {
            rest = after_comma;
            continue;
        }
        break;
    }
    if !rest.is_empty() || min > max {
        return Err(format!("Incorrect parameter: --adapt={input}"));
    }
    Ok((min, max))
}

fn reject_attached_separate_field_args(args: &[OsString]) -> Result<(), String> {
    let mut i = 1usize;
    while i < args.len() {
        let arg_os = &args[i];
        let Some(arg) = arg_os.to_str() else {
            i += 1;
            continue;
        };
        if arg == "--" {
            break;
        }
        if matches!(
            arg,
            "-D" | "-o" | "--__zstd_pure_internal_dict" | "--__zstd_pure_internal_output_file"
        ) {
            if args
                .get(i + 1)
                .and_then(|value| value.to_str())
                .is_some_and(|value| value.starts_with('-'))
            {
                return Err(
                    "command cannot be separated from its argument by another command".into(),
                );
            }
            i += 1;
        }
        i += 1;
    }
    Ok(())
}

fn reject_long_equals_field_args(args: &[OsString]) -> Result<(), String> {
    for arg in args.iter().skip(1) {
        let Some(arg) = arg.to_str() else {
            continue;
        };
        if arg == "--" {
            break;
        }
        if arg.starts_with("--dict=") {
            return Err(format!("Incorrect parameter: {arg}"));
        }
        if arg.starts_with("--output-file=") {
            return Err(format!("Incorrect parameter: {arg}"));
        }
    }
    Ok(())
}

fn reject_long_field_alias_args(args: &[OsString]) -> Result<(), String> {
    for arg in args.iter().skip(1) {
        let Some(arg) = arg.to_str() else {
            continue;
        };
        if arg == "--" {
            break;
        }
        if arg == "--dict" {
            return Err("Incorrect parameter: --dict".into());
        }
        if arg == "--output-file" {
            return Err("Incorrect parameter: --output-file".into());
        }
    }
    Ok(())
}

fn reject_level_alias_args(args: &[OsString]) -> Result<(), String> {
    for arg in args.iter().skip(1) {
        let Some(arg) = arg.to_str() else {
            continue;
        };
        if arg == "--" {
            break;
        }
        if arg == "--level" || arg.starts_with("--level=") {
            return Err(format!("Incorrect parameter: {arg}"));
        }
        if let Some(shorts) = arg.strip_prefix('-') {
            if !shorts.starts_with('-') {
                for c in shorts.chars() {
                    match c {
                        'L' => return Err("Incorrect parameter: -L".into()),
                        'D' | 'o' => break,
                        _ => {}
                    }
                }
            }
        }
    }
    Ok(())
}

fn reject_format_field_args(args: &[OsString]) -> Result<(), String> {
    for arg in args.iter().skip(1) {
        let Some(arg) = arg.to_str() else {
            continue;
        };
        if arg == "--" {
            break;
        }
        if arg == "--format" {
            return Err("Incorrect parameter: --format".into());
        }
    }
    Ok(())
}

fn reject_empty_adapt_equals_args(args: &[OsString]) -> Result<(), String> {
    for arg in args.iter().skip(1) {
        let Some(arg) = arg.to_str() else {
            continue;
        };
        if arg == "--" {
            break;
        }
        if arg == "--adapt=" {
            return Err("Incorrect parameter: --adapt=".into());
        }
    }
    Ok(())
}

fn long_window_log(long: Option<&str>) -> Result<Option<i32>, String> {
    long.map(|value| read_unsigned_prefix_checked(value).map(|(value, _)| value as i32))
        .transpose()
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
        } else if arg == "--__zstd_pure_internal_output_file" || arg == "-o" {
            last = Some(OutputDirective::File);
            i += 1;
        } else if arg.starts_with("--__zstd_pure_internal_output_file=") {
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

fn last_path_field(args: &[OsString], short: &str, internal_long: &str) -> Option<PathBuf> {
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
        if arg == short || arg == internal_long {
            if let Some(value) = args.get(i + 1) {
                last = Some(PathBuf::from(value));
            }
            i += 1;
        } else if let Some(value) = arg.strip_prefix(internal_long) {
            if let Some(value) = value.strip_prefix('=') {
                last = Some(PathBuf::from(value));
            }
        }
        i += 1;
    }
    last
}

fn output_file_stdout_mark_is_rejected(args: &[OsString]) -> bool {
    let mut i = 1usize;
    while i < args.len() {
        let Some(arg) = args[i].to_str() else {
            i += 1;
            continue;
        };
        if arg == "--" {
            break;
        }
        if arg == "-o" {
            if args
                .get(i + 1)
                .and_then(|value| value.to_str())
                .is_some_and(|value| value == "-")
            {
                return true;
            }
            i += 1;
        } else if arg == "--output-file=-" {
            return true;
        }
        i += 1;
    }
    false
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

fn compression_extension(format: &str) -> &'static str {
    match format {
        "gzip" => ".gz",
        "xz" => ".xz",
        "lzma" => ".lzma",
        "lz4" => ".lz4",
        _ => ".zst",
    }
}

fn infer_output_path(input: &Path, decompress: bool, format: &str) -> Option<PathBuf> {
    if decompress {
        // Strip recognized decompression suffixes, including tar shorthands.
        let s = input.to_string_lossy();
        for ext in [".tzst", ".tgz", ".txz", ".tlz4"] {
            if let Some(stripped) = s.strip_suffix(ext) {
                if stripped.is_empty() {
                    return None;
                }
                return Some(PathBuf::from(format!("{stripped}.tar")));
            }
        }
        for ext in [".zst", ".zstd", ".gz", ".xz", ".lzma", ".lz4"] {
            if let Some(stripped) = s.strip_suffix(ext) {
                if stripped.is_empty() {
                    return None;
                }
                return Some(PathBuf::from(stripped));
            }
        }
        None
    } else {
        Some(PathBuf::from(format!(
            "{}{}",
            input.display(),
            compression_extension(format)
        )))
    }
}

fn output_path_for_input(
    input: &Path,
    decompress: bool,
    explicit_stdout: bool,
    effective_output_file: Option<&PathBuf>,
    format: &str,
) -> Result<Option<PathBuf>, String> {
    if let Some(output_file) = effective_output_file {
        return Ok(Some(output_file.clone()));
    }
    if explicit_stdout || input == Path::new("-") {
        return Ok(None);
    }
    match infer_output_path(input, decompress, format) {
        Some(path) => Ok(Some(path)),
        None => Err(format!(
            "{}: unknown suffix (.zst/.tzst/.gz/.tgz/.xz/.txz/.lzma/.lz4/.tlz4 expected). Can't derive the output file name. Specify it with -o dstFileName. Ignoring.",
            input.display()
        )),
    }
}

fn require_user_confirmation(
    prompt: &str,
    abort_msg: &str,
    acceptable_letters: &[u8],
    has_stdin_input: bool,
) -> Result<(), String> {
    if has_stdin_input {
        eprintln!("stdin is an input - not proceeding.");
        return Err(String::new());
    }

    eprint!("{prompt}");
    io::stderr().flush().map_err(|e| e.to_string())?;
    let stdin = io::stdin();
    let mut stdin = stdin.lock();
    let mut answer = [0u8; 1];
    let accepted = stdin.read(&mut answer).map_err(|e| e.to_string())? == 1
        && acceptable_letters.contains(&answer[0]);
    let mut flush = Vec::new();
    stdin
        .read_until(b'\n', &mut flush)
        .map_err(|e| e.to_string())?;
    if !accepted {
        eprintln!("{abort_msg} ");
        return Err(String::new());
    }
    Ok(())
}

fn prepare_output_path(
    path: Option<&Path>,
    force: bool,
    display_level: i32,
    has_stdin_input: bool,
) -> Result<(), String> {
    let Some(path) = path else {
        return Ok(());
    };
    if force || !path.exists() {
        return Ok(());
    }
    if display_level <= 1 {
        if display_level >= 1 {
            eprintln!("zstd: {} already exists; not overwritten  ", path.display());
        }
        return Err(String::new());
    }
    eprint!("zstd: {} already exists; ", path.display());
    require_user_confirmation(
        "overwrite (y/n) ? ",
        "Not overwritten  ",
        b"yY",
        has_stdin_input,
    )
}

fn write_output(path: Option<&Path>, data: &[u8]) -> io::Result<()> {
    match path {
        None => io::stdout().write_all(data),
        Some(p) => fs::write(p, data),
    }
}

fn file_starts_like_zstd(input: &Path) -> io::Result<bool> {
    let mut file = File::open(input)?;
    let mut prefix = [0u8; 4];
    let mut filled = 0usize;
    while filled < prefix.len() {
        match file.read(&mut prefix[filled..])? {
            0 => return Ok(false),
            n => filled += n,
        }
    }
    let magic = u32::from_le_bytes(prefix);
    Ok(magic == ZSTD_MAGICNUMBER
        || (magic & ZSTD_MAGIC_SKIPPABLE_MASK) == ZSTD_MAGIC_SKIPPABLE_START)
}

fn read_exact_or_eof<R: Read>(reader: &mut R, buf: &mut [u8]) -> io::Result<bool> {
    let mut filled = 0usize;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..])? {
            0 if filled == 0 => return Ok(false),
            0 => {
                return Err(io::Error::new(
                    IoErrorKind::UnexpectedEof,
                    "unexpected end of file",
                ))
            }
            n => filled += n,
        }
    }
    Ok(true)
}

fn configure_zstd_compressor(
    cctx: &mut zstd_pure_rs::compress::zstd_compress::ZSTD_CCtx,
    level: i32,
    checksum: bool,
    content_size: bool,
    pledged_size: u64,
) -> Result<(), String> {
    for (param, value) in [
        (ZSTD_cParameter::ZSTD_c_compressionLevel, level),
        (
            ZSTD_cParameter::ZSTD_c_contentSizeFlag,
            if content_size { 1 } else { 0 },
        ),
        (
            ZSTD_cParameter::ZSTD_c_checksumFlag,
            if checksum { 1 } else { 0 },
        ),
        (ZSTD_cParameter::ZSTD_c_dictIDFlag, 0),
        (
            ZSTD_cParameter::ZSTD_c_format,
            ZSTD_format_e::ZSTD_f_zstd1 as i32,
        ),
    ] {
        let rc = ZSTD_CCtx_setParameter(cctx, param, value);
        if ERR_isError(rc) {
            return Err(ERR_getErrorName(rc).to_string());
        }
    }
    let rc = ZSTD_CCtx_setPledgedSrcSize(cctx, pledged_size);
    if ERR_isError(rc) {
        return Err(ERR_getErrorName(rc).to_string());
    }
    Ok(())
}

fn read_exact_vec<R: Read>(reader: &mut R, len: usize) -> io::Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(len);
    let spare = buf.spare_capacity_mut();
    let dst = unsafe { std::slice::from_raw_parts_mut(spare.as_mut_ptr() as *mut u8, len) };
    reader.read_exact(dst)?;
    unsafe {
        buf.set_len(len);
    }
    Ok(buf)
}

fn stream_buffered_compress_zstd_file_to_writer<W: Write>(
    input: &Path,
    writer: &mut W,
    level: i32,
    checksum: bool,
    content_size: bool,
) -> Result<(usize, usize), String> {
    let file = File::open(input).map_err(|e| format!("{}: {e}", input.display()))?;
    let src_size = file
        .metadata()
        .map_err(|e| format!("{}: {e}", input.display()))?
        .len();
    let mut reader = BufReader::new(file);
    if level >= 3 && src_size <= 256 << 20 {
        let src = read_exact_vec(&mut reader, src_size as usize)
            .map_err(|e| format!("{}: {e}", input.display()))?;
        let compressed = compress_bytes(
            &src,
            level,
            None,
            checksum,
            content_size,
            false,
            false,
            None,
            true,
        )?;
        writer.write_all(&compressed).map_err(|e| e.to_string())?;
        writer.flush().map_err(|e| e.to_string())?;
        return Ok((src.len(), compressed.len()));
    }
    let mut cctx = ZSTD_createCCtx().ok_or("cctx alloc failed")?;
    configure_zstd_compressor(&mut cctx, level, checksum, content_size, src_size)?;

    let file_chunk_size = ZSTD_CStreamInSize().max(1);
    let mut input_buf = vec![0u8; file_chunk_size];
    let output_bound = ZSTD_compressBound(file_chunk_size).saturating_add(64);
    let mut output_buf = vec![0u8; ZSTD_CStreamOutSize().max(output_bound).max(32)];
    let mut total_in = 0usize;
    let mut total_out = 0usize;

    loop {
        let read = reader
            .read(&mut input_buf)
            .map_err(|e| format!("{}: {e}", input.display()))?;
        if read == 0 {
            break;
        }
        total_in = total_in.saturating_add(read);
        let mut src_pos = 0usize;
        while src_pos < read {
            let mut dst_pos = 0usize;
            let rc = ZSTD_compressStream(
                &mut cctx,
                &mut output_buf,
                &mut dst_pos,
                &input_buf[..read],
                &mut src_pos,
            );
            if ERR_isError(rc) {
                return Err(ERR_getErrorName(rc).to_string());
            }
            if dst_pos > 0 {
                writer
                    .write_all(&output_buf[..dst_pos])
                    .map_err(|e| e.to_string())?;
                total_out = total_out.saturating_add(dst_pos);
            }
        }
    }

    loop {
        let mut dst_pos = 0usize;
        let rc = ZSTD_endStream(&mut cctx, &mut output_buf, &mut dst_pos);
        if ERR_isError(rc) {
            return Err(ERR_getErrorName(rc).to_string());
        }
        if dst_pos > 0 {
            writer
                .write_all(&output_buf[..dst_pos])
                .map_err(|e| e.to_string())?;
            total_out = total_out.saturating_add(dst_pos);
        }
        if rc == 0 {
            break;
        }
        if dst_pos == 0 {
            output_buf.resize(output_buf.len().saturating_add(rc.max(32)), 0);
        }
    }

    writer.flush().map_err(|e| e.to_string())?;
    Ok((total_in, total_out))
}

fn stream_buffered_compress_zstd_file(
    input: &Path,
    out_path: Option<&Path>,
    level: i32,
    checksum: bool,
    content_size: bool,
) -> Result<(usize, usize), String> {
    match out_path {
        Some(path) => {
            let tmp_path = path.with_extension(format!(
                "{}.tmp.{}",
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("out"),
                std::process::id()
            ));
            let file =
                File::create(&tmp_path).map_err(|e| format!("{}: {e}", tmp_path.display()))?;
            let mut writer = BufWriter::with_capacity(FILE_OUTPUT_BUFFER_SIZE, file);
            match stream_buffered_compress_zstd_file_to_writer(
                input,
                &mut writer,
                level,
                checksum,
                content_size,
            ) {
                Ok(sizes) => {
                    drop(writer);
                    fs::rename(&tmp_path, path).map_err(|e| format!("{}: {e}", path.display()))?;
                    Ok(sizes)
                }
                Err(err) => {
                    drop(writer);
                    let _ = fs::remove_file(&tmp_path);
                    Err(err)
                }
            }
        }
        None => {
            let stdout = io::stdout();
            let mut writer = stdout.lock();
            stream_buffered_compress_zstd_file_to_writer(
                input,
                &mut writer,
                level,
                checksum,
                content_size,
            )
        }
    }
}

fn stream_decompress_zstd_file_to_writer<W: Write>(
    input: &Path,
    writer: &mut W,
    ignore_checksum: bool,
) -> Result<(usize, usize), String> {
    {
        let mut header_prefix = [0u8; 18];
        let mut probe = File::open(input).map_err(|e| format!("{}: {e}", input.display()))?;
        let got = probe
            .read(&mut header_prefix)
            .map_err(|e| format!("{}: {e}", input.display()))?;
        let mut frame_header = ZSTD_FrameHeader::default();
        let header_rc = ZSTD_getFrameHeader(&mut frame_header, &header_prefix[..got]);
        if header_rc == 0 && frame_header.windowSize > STREAM_DECODE_WINDOW_LIMIT {
            let compressed = fs::read(input).map_err(|e| format!("{}: {e}", input.display()))?;
            let decoded = decompress_bytes(&compressed, None, false, ignore_checksum)?;
            writer.write_all(&decoded).map_err(|e| e.to_string())?;
            writer.flush().map_err(|e| e.to_string())?;
            return Ok((compressed.len(), decoded.len()));
        }
    }

    let file = File::open(input).map_err(|e| format!("{}: {e}", input.display()))?;
    let mut reader = BufReader::new(file);
    let mut dctx = ZSTD_DCtx::new();
    ZSTD_initDStream(&mut dctx);
    if ignore_checksum {
        dctx.forceIgnoreChecksum = ZSTD_forceIgnoreChecksum_e::ZSTD_d_ignoreChecksum;
    }
    let mut total_in = 0usize;
    let mut total_out = 0usize;
    let mut chunk = Vec::new();

    loop {
        let at_frame_boundary = total_in > 0 && ZSTD_nextSrcSizeToDecompress(&dctx) == 0;
        let expected = if at_frame_boundary {
            ZSTD_resetDStream(&mut dctx)
        } else {
            ZSTD_nextSrcSizeToDecompress(&dctx)
        };
        if expected == 0 {
            break;
        }

        chunk.resize(expected, 0);
        let got_chunk = read_exact_or_eof(&mut reader, &mut chunk)
            .map_err(|e| format!("{}: {e}", input.display()))?;
        if !got_chunk {
            if at_frame_boundary {
                break;
            }
            return Err(format!("{}: unexpected end of file ", input.display()));
        }
        total_in = total_in.saturating_add(chunk.len());

        let produced = ZSTD_decompressContinue_into_history(&mut dctx, &chunk)
            .map_err(|e| ERR_getErrorName(e).to_string())?;
        if !produced.is_empty() {
            writer.write_all(produced).map_err(|e| e.to_string())?;
            total_out = total_out.saturating_add(produced.len());
        }
    }

    writer.flush().map_err(|e| e.to_string())?;
    Ok((total_in, total_out))
}

fn stream_decompress_zstd_file(
    input: &Path,
    out_path: Option<&Path>,
    ignore_checksum: bool,
) -> Result<(usize, usize), String> {
    match out_path {
        Some(path) => {
            let tmp_path = path.with_extension(format!(
                "{}.tmp.{}",
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("out"),
                std::process::id()
            ));
            let file =
                File::create(&tmp_path).map_err(|e| format!("{}: {e}", tmp_path.display()))?;
            let mut writer = BufWriter::with_capacity(FILE_OUTPUT_BUFFER_SIZE, file);
            match stream_decompress_zstd_file_to_writer(input, &mut writer, ignore_checksum) {
                Ok(sizes) => {
                    drop(writer);
                    fs::rename(&tmp_path, path).map_err(|e| format!("{}: {e}", path.display()))?;
                    Ok(sizes)
                }
                Err(err) => {
                    drop(writer);
                    let _ = fs::remove_file(&tmp_path);
                    Err(err)
                }
            }
        }
        None => {
            let stdout = io::stdout();
            let mut writer = stdout.lock();
            stream_decompress_zstd_file_to_writer(input, &mut writer, ignore_checksum)
        }
    }
}

fn remove_source_file(input: &Path) -> Result<(), String> {
    if input == Path::new("-") {
        return Ok(());
    }
    fs::remove_file(input).map_err(|e| format!("{}: {e}", input.display()))
}

fn should_remove_source(remove_src_file: bool, out_path: Option<&Path>) -> bool {
    remove_src_file && out_path.is_some()
}

fn effective_remove_source(remove_src_file: bool, explicit_stdout: bool, test_mode: bool) -> bool {
    remove_src_file && !explicit_stdout && !test_mode
}

fn effective_decompress_pass_through(
    explicit_pass_through: bool,
    force: bool,
    pass_through_directive: Option<bool>,
    output_is_stdout: bool,
    test_mode: bool,
) -> bool {
    explicit_pass_through
        || (!test_mode && force && pass_through_directive != Some(false) && output_is_stdout)
}

fn path_is_stdout_mark(path: &Path) -> bool {
    path == Path::new("-")
}

#[cfg(unix)]
fn metadata_is_same_file(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;

    left.dev() == right.dev() && left.ino() == right.ino()
}

#[cfg(not(unix))]
fn metadata_is_same_file(_left: &fs::Metadata, _right: &fs::Metadata) -> bool {
    false
}

fn existing_files_are_same(left: &Path, right: &Path) -> io::Result<bool> {
    if path_is_stdout_mark(left) || path_is_stdout_mark(right) {
        return Ok(false);
    }
    let right_meta = match fs::metadata(right) {
        Ok(meta) => meta,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(e),
    };
    let left_meta = fs::metadata(left)?;
    if metadata_is_same_file(&left_meta, &right_meta) {
        return Ok(true);
    }
    let left_canon = fs::canonicalize(left)?;
    let right_canon = fs::canonicalize(right)?;
    Ok(left_canon == right_canon)
}

fn reject_same_input_output(input: &Path, output: Option<&Path>) -> Result<(), String> {
    let Some(output) = output else {
        return Ok(());
    };
    if existing_files_are_same(input, output).map_err(|e| format!("{}: {e}", output.display()))? {
        return Err("Refusing to open an output file which will overwrite the input file".into());
    }
    Ok(())
}

fn confirm_multi_input_output(
    out_path: &Path,
    inputs: &[PathBuf],
    force: bool,
    display_level: i32,
) -> Result<(), String> {
    if display_level >= 2 {
        eprintln!(
            "zstd: WARNING: all input files will be processed and concatenated into a single output file: {} ",
            out_path.display()
        );
        eprintln!(
            "The concatenated output CANNOT regenerate original file names nor directory structure. "
        );
    }
    if force {
        return Ok(());
    }
    if display_level <= 1 {
        if display_level >= 1 {
            eprintln!("Concatenating multiple processed inputs into a single output loses file metadata. ");
            eprintln!("Aborting. ");
        }
        return Err(String::new());
    }
    require_user_confirmation(
        "Proceed? (y/n): ",
        "Aborting...",
        b"yY",
        inputs.iter().any(|input| input == Path::new("-")),
    )
}

fn resolve_compression_level(level: i32, ultra: bool, display_level: i32) -> i32 {
    let max_level = if ultra { ZSTD_maxCLevel() } else { 19 };
    if level > max_level {
        if display_level >= 2 {
            eprintln!(
                "Warning : compression level higher than max, reduced to {max_level}. Specify --ultra to raise the limit to 22 and use --long=31 for maximum compression. Note that this requires high amounts of memory, and the resulting data might be rejected by third-party decoders and is therefore only recommended for archival purposes. "
            );
        }
        max_level
    } else {
        level
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

fn is_gzip_input(src: &[u8]) -> bool {
    src.len() >= 2 && src[0] == 0x1f && src[1] == 0x8b
}

fn is_xz_input(src: &[u8]) -> bool {
    src.len() >= 2 && src[0] == 0xfd && src[1] == 0x37
}

fn is_lzma_input(src: &[u8]) -> bool {
    src.len() >= 2 && src[0] == 0x5d && src[1] == 0x00
}

fn is_lz4_input(src: &[u8]) -> bool {
    src.len() >= 4 && src[..4] == [0x04, 0x22, 0x4d, 0x18]
}

fn run_filter(command: &str, args: &[&str], src: &[u8], format: &str) -> Result<Vec<u8>, String> {
    let mut child = ProcessCommand::new(command)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("{format} support is unavailable: {e}"))?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| format!("{format} support is unavailable: stdin pipe not opened"))?;
    let src = src.to_vec();
    let format_for_writer = format.to_string();
    let writer = std::thread::spawn(move || {
        stdin
            .write_all(&src)
            .map_err(|e| format!("{format_for_writer} encoder/decoder input failed: {e}"))
    });
    let out = child
        .wait_with_output()
        .map_err(|e| format!("{format} encoder/decoder failed: {e}"))?;
    let write_result = writer
        .join()
        .map_err(|_| format!("{format} encoder/decoder input thread panicked"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let detail = stderr.trim();
        if detail.is_empty() {
            return Err(format!("{format} encoder/decoder failed"));
        }
        return Err(format!("{format} encoder/decoder failed: {detail}"));
    }
    write_result?;
    Ok(out.stdout)
}

fn decompress_compat_format(src: &[u8], format: &str) -> Result<Vec<u8>, String> {
    match format {
        "gzip" => run_filter("gzip", &["-dc"], src, "gzip"),
        "xz" => run_filter("xz", &["-dc", "--format=xz"], src, "xz"),
        "lzma" => run_filter("xz", &["-dc", "--format=lzma"], src, "lzma"),
        "lz4" => run_filter("lz4", &["-dc"], src, "lz4"),
        _ => unreachable!(),
    }
}

fn read_gzip_cstring(src: &[u8], pos: &mut usize) -> Result<(), String> {
    while *pos < src.len() {
        let b = src[*pos];
        *pos += 1;
        if b == 0 {
            return Ok(());
        }
    }
    Err("gzip header is truncated".into())
}

fn gzip_payload_bounds(src: &[u8]) -> Result<(usize, usize), String> {
    if src.len() < 18 || !is_gzip_input(src) {
        return Err("unsupported format ".into());
    }
    if src[2] != 8 {
        return Err("gzip stream uses unsupported compression method".into());
    }
    let flags = src[3];
    if flags & 0b1110_0000 != 0 {
        return Err("gzip stream has reserved header flags set".into());
    }
    let mut pos = 10usize;
    if flags & 0x04 != 0 {
        if pos + 2 > src.len() {
            return Err("gzip header is truncated".into());
        }
        let xlen = u16::from_le_bytes([src[pos], src[pos + 1]]) as usize;
        pos += 2;
        if pos + xlen > src.len() {
            return Err("gzip header is truncated".into());
        }
        pos += xlen;
    }
    if flags & 0x08 != 0 {
        read_gzip_cstring(src, &mut pos)?;
    }
    if flags & 0x10 != 0 {
        read_gzip_cstring(src, &mut pos)?;
    }
    if flags & 0x02 != 0 {
        if pos + 2 > src.len() {
            return Err("gzip header is truncated".into());
        }
        pos += 2;
    }
    if pos + 8 > src.len() {
        return Err("gzip stream is truncated".into());
    }
    Ok((pos, src.len() - 8))
}

fn read_deflate_bits(src: &[u8], bit_pos: &mut usize, bits: usize) -> Result<u32, String> {
    let mut value = 0u32;
    for bit in 0..bits {
        let byte_pos = *bit_pos / 8;
        if byte_pos >= src.len() {
            return Err("gzip deflate stream is truncated".into());
        }
        let bit_in_byte = *bit_pos % 8;
        value |= u32::from((src[byte_pos] >> bit_in_byte) & 1) << bit;
        *bit_pos += 1;
    }
    Ok(value)
}

fn gzip_decompress_stored_blocks(src: &[u8]) -> Result<Vec<u8>, String> {
    let (payload_start, trailer_start) = gzip_payload_bounds(src)?;
    let payload = &src[payload_start..trailer_start];
    let mut bit_pos = 0usize;
    let mut dst = Vec::new();

    loop {
        let final_block = read_deflate_bits(payload, &mut bit_pos, 1)? != 0;
        let block_type = read_deflate_bits(payload, &mut bit_pos, 2)?;
        if block_type != 0 {
            return Err("gzip deflate stream uses unsupported compressed blocks".into());
        }
        bit_pos = (bit_pos + 7) & !7;
        let byte_pos = bit_pos / 8;
        if byte_pos + 4 > payload.len() {
            return Err("gzip stored block header is truncated".into());
        }
        let len = u16::from_le_bytes([payload[byte_pos], payload[byte_pos + 1]]);
        let nlen = u16::from_le_bytes([payload[byte_pos + 2], payload[byte_pos + 3]]);
        if len != !nlen {
            return Err("gzip stored block length check failed".into());
        }
        let data_start = byte_pos + 4;
        let data_end = data_start + usize::from(len);
        if data_end > payload.len() {
            return Err("gzip stored block is truncated".into());
        }
        dst.extend_from_slice(&payload[data_start..data_end]);
        bit_pos = data_end * 8;
        if final_block {
            break;
        }
    }

    let expected_crc = u32::from_le_bytes([
        src[trailer_start],
        src[trailer_start + 1],
        src[trailer_start + 2],
        src[trailer_start + 3],
    ]);
    let expected_size = u32::from_le_bytes([
        src[trailer_start + 4],
        src[trailer_start + 5],
        src[trailer_start + 6],
        src[trailer_start + 7],
    ]);
    if crc32_ieee(&dst) != expected_crc {
        return Err("gzip checksum mismatch".into());
    }
    if dst.len() as u32 != expected_size {
        return Err("gzip content size mismatch".into());
    }
    Ok(dst)
}

fn decompress_or_passthrough_bytes(
    input: &Path,
    src: &[u8],
    dict: Option<&[u8]>,
    magicless: bool,
    ignore_checksum: bool,
    pass_through: bool,
) -> Result<Vec<u8>, String> {
    if !magicless {
        if src.is_empty() {
            return Err(format!("{}: unexpected end of file ", input.display()));
        }
        if src.len() < 4 {
            if pass_through {
                return Ok(src.to_vec());
            }
            return Err(format!("{}: unknown header ", input.display()));
        }
    }
    if !magicless && is_gzip_input(src) {
        return gzip_decompress_stored_blocks(src)
            .or_else(|_| decompress_compat_format(src, "gzip"));
    }
    if !magicless && is_xz_input(src) {
        return decompress_compat_format(src, "xz");
    }
    if !magicless && is_lzma_input(src) {
        return decompress_compat_format(src, "lzma");
    }
    if !magicless && is_lz4_input(src) {
        return decompress_compat_format(src, "lz4");
    }
    if !looks_like_zstd_input(src, magicless) {
        if pass_through {
            return Ok(src.to_vec());
        }
        return Err(format!("{}: unsupported format ", input.display()));
    }
    decompress_bytes(src, dict, magicless, ignore_checksum)
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
    content_size: bool,
    dict_id: bool,
    magicless: bool,
    window_log: Option<i32>,
    source_size_known: bool,
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
    let mut cctx = ZSTD_createCCtx().ok_or("cctx alloc failed")?;
    for (param, value) in [
        (ZSTD_cParameter::ZSTD_c_compressionLevel, level),
        (
            ZSTD_cParameter::ZSTD_c_contentSizeFlag,
            if content_size { 1 } else { 0 },
        ),
        (
            ZSTD_cParameter::ZSTD_c_checksumFlag,
            if checksum { 1 } else { 0 },
        ),
        (
            ZSTD_cParameter::ZSTD_c_dictIDFlag,
            if dict.is_some() && dict_id { 1 } else { 0 },
        ),
        (ZSTD_cParameter::ZSTD_c_format, format as i32),
    ] {
        let rc = ZSTD_CCtx_setParameter(&mut cctx, param, value);
        if ERR_isError(rc) {
            return Err(ERR_getErrorName(rc).to_string());
        }
    }
    if let Some(window_log) = window_log {
        let rc = ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_windowLog, window_log);
        if ERR_isError(rc) {
            return Err(ERR_getErrorName(rc).to_string());
        }
    }
    if let Some(d) = dict {
        let rc = ZSTD_CCtx_loadDictionary(&mut cctx, d);
        if ERR_isError(rc) {
            return Err(ERR_getErrorName(rc).to_string());
        }
    }
    if source_size_known {
        let n = ZSTD_compress2(&mut cctx, &mut dst, src);
        if ERR_isError(n) {
            return Err(ERR_getErrorName(n).to_string());
        }
        dst.truncate(n);
    } else {
        let mut src_pos = 0usize;
        let mut dst_pos = 0usize;
        loop {
            let rc = ZSTD_compressStream(&mut cctx, &mut dst, &mut dst_pos, src, &mut src_pos);
            if ERR_isError(rc) {
                return Err(ERR_getErrorName(rc).to_string());
            }
            if src_pos >= src.len() {
                break;
            }
            if src_pos < src.len() && dst_pos == dst.len() {
                dst.resize(dst.len() + ZSTD_CStreamOutSize().max(32), 0);
            }
        }
        loop {
            let rc = ZSTD_endStream(&mut cctx, &mut dst, &mut dst_pos);
            if ERR_isError(rc) {
                return Err(ERR_getErrorName(rc).to_string());
            }
            if rc == 0 {
                break;
            }
            if dst_pos == dst.len() {
                dst.resize(dst.len() + rc.max(ZSTD_CStreamOutSize()).max(32), 0);
            }
        }
        dst.truncate(dst_pos);
    }
    Ok(dst)
}

fn crc32_ieee(src: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &byte in src {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

fn gzip_stored_blocks(src: &[u8]) -> Vec<u8> {
    let mut dst = Vec::with_capacity(src.len().saturating_add((src.len() / 65_535 + 1) * 5 + 18));
    dst.extend_from_slice(&[
        0x1f, 0x8b, // gzip magic
        0x08, // deflate
        0x00, // flags
        0x00, 0x00, 0x00, 0x00, // mtime
        0x00, // extra flags
        0x03, // Unix
    ]);

    if src.is_empty() {
        dst.extend_from_slice(&[0x01, 0x00, 0x00, 0xff, 0xff]);
    } else {
        let mut remaining = src;
        while !remaining.is_empty() {
            let take = remaining.len().min(65_535);
            let final_block = take == remaining.len();
            dst.push(if final_block { 0x01 } else { 0x00 });
            let len = take as u16;
            dst.extend_from_slice(&len.to_le_bytes());
            dst.extend_from_slice(&(!len).to_le_bytes());
            dst.extend_from_slice(&remaining[..take]);
            remaining = &remaining[take..];
        }
    }

    dst.extend_from_slice(&crc32_ieee(src).to_le_bytes());
    dst.extend_from_slice(&(src.len() as u32).to_le_bytes());
    dst
}

fn compress_bytes_for_format(
    src: &[u8],
    level: i32,
    dict: Option<&[u8]>,
    checksum: bool,
    content_size: bool,
    dict_id: bool,
    magicless: bool,
    window_log: Option<i32>,
    format: &str,
    source_size_known: bool,
) -> Result<Vec<u8>, String> {
    match format {
        "zstd" => compress_bytes(
            src,
            level,
            dict,
            checksum,
            content_size,
            dict_id,
            magicless,
            window_log,
            source_size_known,
        ),
        "gzip" => {
            if dict.is_some() {
                return Err("--format=gzip does not support zstd dictionaries".into());
            }
            if magicless {
                return Err("--format=gzip does not support --magicless".into());
            }
            Ok(gzip_stored_blocks(src))
        }
        "xz" | "lzma" | "lz4" => {
            if dict.is_some() {
                return Err(format!(
                    "--format={format} does not support zstd dictionaries"
                ));
            }
            if magicless {
                return Err(format!("--format={format} does not support --magicless"));
            }
            match format {
                "xz" => run_filter(
                    "xz",
                    &["-zc", "--format=xz", &format!("-{}", level.clamp(0, 9))],
                    src,
                    "xz",
                ),
                "lzma" => run_filter(
                    "xz",
                    &["-zc", "--format=lzma", &format!("-{}", level.clamp(0, 9))],
                    src,
                    "lzma",
                ),
                "lz4" => run_filter("lz4", &["-zc", "-q"], src, "lz4"),
                _ => unreachable!(),
            }
        }
        _ => Err(format!("Unsupported format: {format}")),
    }
}

fn run() -> Result<(), String> {
    let original_args: Vec<OsString> = std::env::args_os().collect();
    if let Some((directive, display_level, directive_index)) =
        upstream_terminal_directive(&original_args)
    {
        validate_args_before_version_exit(&original_args, directive_index)?;
        if directive == TerminalDirective::Version {
            print_upstream_version(display_level);
        } else {
            print_upstream_help(directive)?;
        }
        return Ok(());
    }
    let reject_stdout_output_file = output_file_stdout_mark_is_rejected(&original_args);
    reject_level_alias_args(&original_args)?;
    reject_long_equals_field_args(&original_args)?;
    reject_long_field_alias_args(&original_args)?;
    reject_format_field_args(&original_args)?;
    reject_empty_adapt_equals_args(&original_args)?;
    let raw_args = normalize_attached_field_args(original_args)?;
    reject_attached_separate_field_args(&raw_args)?;
    let output_directive = last_output_directive(&raw_args);
    let checksum_directive = last_checksum_directive(&raw_args);
    let content_size_directive = last_content_size_directive(&raw_args);
    let dict_id_directive = last_dict_id_directive(&raw_args);
    let remove_source_directive = last_remove_source_directive(&raw_args);
    let pass_through_directive = last_pass_through_directive(&raw_args);
    let operation_directive = last_operation_directive(&raw_args);
    let normalized_args = normalize_fast_args(normalize_level_short_args(raw_args.clone())?)?;
    let cli = match Cli::try_parse_from(normalized_args) {
        Ok(cli) => cli,
        Err(e) if matches!(e.kind(), ErrorKind::DisplayHelp | ErrorKind::DisplayVersion) => {
            e.print().map_err(|e| e.to_string())?;
            return Ok(());
        }
        Err(e) => return Err(e.to_string()),
    };
    if cli.help_advanced {
        Cli::command()
            .print_long_help()
            .map_err(|e| e.to_string())?;
        println!();
        return Ok(());
    }
    let output_format = cli.format.as_deref().unwrap_or("zstd");
    if let Some(format) = cli.format.as_deref() {
        if !matches!(format, "zstd" | "gzip" | "xz" | "lzma" | "lz4") {
            return Err(format!("Incorrect parameter: --format={format}"));
        }
    }

    let inputs = if cli.files.is_empty() {
        vec![PathBuf::from("-")]
    } else {
        cli.files.clone()
    };
    let operation = operation_directive.unwrap_or_else(|| {
        if cli.test {
            OperationDirective::Test
        } else if cli.decompress && !cli.compress {
            OperationDirective::Decompress
        } else {
            OperationDirective::Compress
        }
    });
    let decompress = matches!(
        operation,
        OperationDirective::Decompress | OperationDirective::Test
    );
    let test_mode = operation == OperationDirective::Test;

    let explicit_stdout = match output_directive {
        Some(OutputDirective::Stdout) => true,
        Some(OutputDirective::File) => false,
        None => cli.stdout,
    };
    let parsed_output_file = last_path_field(&raw_args, "-o", "--__zstd_pure_internal_output_file")
        .or_else(|| {
            cli.output_file
                .clone()
                .or_else(|| cli.internal_output_file.clone())
        });
    let effective_output_file = if explicit_stdout {
        None
    } else {
        parsed_output_file.clone()
    };
    if reject_stdout_output_file
        && parsed_output_file
            .as_deref()
            .is_some_and(path_is_stdout_mark)
    {
        return Err(
            "--output-file/-o does not accept '-' as an output file; use -c/--stdout".into(),
        );
    }
    let pass_through = pass_through_directive.unwrap_or(cli.pass_through && !cli.no_pass_through);
    let display_level = 2 + i32::from(cli.verbose) - i32::from(cli.quiet);
    let has_stdin_input = inputs.iter().any(|input| input == Path::new("-"));
    let requested_remove_src_file = remove_source_directive.unwrap_or(cli.rm && !cli.keep);
    if requested_remove_src_file && explicit_stdout && display_level >= 3 {
        eprintln!("Note: src files are not removed when output is stdout ");
    }
    let remove_src_file =
        effective_remove_source(requested_remove_src_file, explicit_stdout, test_mode);
    let mut requested_level = if cli.max {
        ZSTD_maxCLevel()
    } else if let Some(fast) = cli.fast_level {
        -(fast as i32)
    } else {
        cli.level
    };
    if let Some(adapt) = cli.adapt.as_deref().filter(|adapt| !adapt.is_empty()) {
        let (adapt_min, adapt_max) = parse_adapt_parameters(adapt)?;
        if requested_level < adapt_min {
            requested_level = adapt_min;
        }
        if requested_level > adapt_max {
            requested_level = adapt_max;
        }
    }
    let effective_level = resolve_compression_level(
        requested_level,
        cli.ultra || cli.long.is_some() || cli.max,
        display_level,
    );
    let window_log = long_window_log(cli.long.as_deref())?;
    let content_size = content_size_directive.unwrap_or(!cli.no_content_size);
    let dict_id = dict_id_directive.unwrap_or(!cli.no_dict_id);

    // Load the dict once (if provided) — used by every file.
    let parsed_dict = last_path_field(&raw_args, "-D", "--__zstd_pure_internal_dict")
        .or_else(|| cli.dict.clone().or_else(|| cli.internal_dict.clone()));
    let dict_bytes: Option<Vec<u8>> = parsed_dict
        .as_ref()
        .map(|p| fs::read(p).map_err(|e| format!("dict {}: {e}", p.display())))
        .transpose()?;
    let dict_ref = dict_bytes.as_deref();

    if let Some(out_path) = effective_output_file.as_ref().filter(|_| inputs.len() > 1) {
        if test_mode {
            return Err("test mode does not support concatenated output".into());
        }
        confirm_multi_input_output(out_path, &inputs, cli.force, display_level)?;
        let remove_concatenated_sources = false;
        if remove_src_file && display_level >= 2 {
            eprintln!("Since it's a destructive operation, input files will not be removed. ");
        }
        prepare_output_path(Some(out_path), cli.force, display_level, has_stdin_input)?;
        let file = File::create(out_path).map_err(|e| format!("{}: {e}", out_path.display()))?;
        let mut writer = BufWriter::with_capacity(FILE_OUTPUT_BUFFER_SIZE, file);
        let mut total_src_size = 0usize;
        let mut total_dst_size = 0usize;
        for input in &inputs {
            let checksum = checksum_directive.unwrap_or(true);
            if decompress
                && input != Path::new("-")
                && output_format == "zstd"
                && dict_ref.is_none()
                && !cli.magicless
                && file_starts_like_zstd(input).map_err(|e| format!("{}: {e}", input.display()))?
            {
                let (src_len, dst_len) = stream_decompress_zstd_file_to_writer(
                    input,
                    &mut writer,
                    checksum_directive == Some(false),
                )?;
                total_src_size = total_src_size.saturating_add(src_len);
                total_dst_size = total_dst_size.saturating_add(dst_len);
                continue;
            }
            if !decompress
                && input != Path::new("-")
                && output_format == "zstd"
                && dict_ref.is_none()
                && !cli.magicless
                && window_log.is_none()
            {
                let (src_len, dst_len) = stream_buffered_compress_zstd_file_to_writer(
                    input,
                    &mut writer,
                    effective_level,
                    checksum,
                    content_size,
                )?;
                total_src_size = total_src_size.saturating_add(src_len);
                total_dst_size = total_dst_size.saturating_add(dst_len);
                continue;
            }
            let src = read_input(input).map_err(|e| format!("{}: {e}", input.display()))?;
            total_src_size = total_src_size.saturating_add(src.len());
            let dst = if decompress {
                decompress_or_passthrough_bytes(
                    input,
                    &src,
                    dict_ref,
                    cli.magicless,
                    checksum_directive == Some(false),
                    pass_through,
                )?
            } else {
                compress_bytes_for_format(
                    &src,
                    effective_level,
                    dict_ref,
                    checksum,
                    content_size,
                    dict_id,
                    cli.magicless,
                    window_log,
                    output_format,
                    input != Path::new("-"),
                )?
            };
            writer
                .write_all(&dst)
                .map_err(|e| format!("{}: {e}", out_path.display()))?;
            total_dst_size = total_dst_size.saturating_add(dst.len());
        }
        writer
            .flush()
            .map_err(|e| format!("{}: {e}", out_path.display()))?;
        if should_remove_source(remove_concatenated_sources, Some(out_path)) {
            for input in &inputs {
                remove_source_file(input)?;
            }
        }
        if display_level >= 2 {
            if decompress {
                eprintln!(
                    "{} files decompressed : {} bytes total",
                    inputs.len(),
                    total_dst_size
                );
            } else {
                let ratio = if total_src_size == 0 {
                    0.0
                } else {
                    (total_dst_size as f64 * 100.0) / (total_src_size as f64)
                };
                eprintln!(
                    "  {} files compressed : {:.2}% ({} B => {} B)",
                    inputs.len(),
                    ratio,
                    total_src_size,
                    total_dst_size
                );
            }
        }
        return Ok(());
    }

    for input in &inputs {
        let out_path = if test_mode {
            None
        } else {
            output_path_for_input(
                input,
                decompress,
                explicit_stdout,
                effective_output_file.as_ref(),
                output_format,
            )?
        };
        reject_same_input_output(input, out_path.as_deref())?;
        if !test_mode {
            prepare_output_path(
                out_path.as_deref(),
                cli.force,
                display_level,
                has_stdin_input,
            )?;
        }
        let output_is_stdout = out_path.is_none();
        let effective_pass_through = effective_decompress_pass_through(
            pass_through,
            cli.force,
            pass_through_directive,
            output_is_stdout,
            test_mode,
        );
        if decompress
            && input != Path::new("-")
            && output_format == "zstd"
            && dict_ref.is_none()
            && !cli.magicless
            && file_starts_like_zstd(input).map_err(|e| format!("{}: {e}", input.display()))?
        {
            let (src_len, dst_len) = if test_mode {
                let mut sink = SinkWriter;
                stream_decompress_zstd_file_to_writer(
                    input,
                    &mut sink,
                    checksum_directive == Some(false),
                )?
            } else {
                stream_decompress_zstd_file(
                    input,
                    out_path.as_deref(),
                    checksum_directive == Some(false),
                )?
            };
            if should_remove_source(remove_src_file, out_path.as_deref()) {
                remove_source_file(input)?;
            }
            if test_mode {
                if display_level >= 2 {
                    eprintln!("{}: OK", input.display());
                }
                continue;
            }
            let effective_display_level = if output_is_stdout && display_level == 2 {
                1
            } else {
                display_level
            };
            if effective_display_level >= 2 {
                eprintln!(
                    "{}: decompressed {} -> {} bytes",
                    input.display(),
                    src_len,
                    dst_len
                );
            }
            continue;
        }
        let checksum = checksum_directive.unwrap_or(true);
        if !decompress
            && !test_mode
            && input != Path::new("-")
            && output_format == "zstd"
            && dict_ref.is_none()
            && !cli.magicless
            && window_log.is_none()
        {
            let (src_len, dst_len) = stream_buffered_compress_zstd_file(
                input,
                out_path.as_deref(),
                effective_level,
                checksum,
                content_size,
            )?;
            if should_remove_source(remove_src_file, out_path.as_deref()) {
                remove_source_file(input)?;
            }
            let effective_display_level = if output_is_stdout && display_level == 2 {
                1
            } else {
                display_level
            };
            if effective_display_level >= 2 {
                eprintln!(
                    "{}: compressed {} -> {} bytes",
                    input.display(),
                    src_len,
                    dst_len
                );
            }
            continue;
        }
        let src = read_input(input).map_err(|e| format!("{}: {e}", input.display()))?;
        let dst = if decompress {
            decompress_or_passthrough_bytes(
                input,
                &src,
                dict_ref,
                cli.magicless,
                checksum_directive == Some(false),
                effective_pass_through,
            )?
        } else {
            compress_bytes_for_format(
                &src,
                effective_level,
                dict_ref,
                checksum,
                content_size,
                dict_id,
                cli.magicless,
                window_log,
                output_format,
                input != Path::new("-"),
            )?
        };
        if test_mode {
            if display_level >= 2 {
                eprintln!("{}: OK", input.display());
            }
            continue;
        }
        let out_label = out_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<stdout>".into());
        write_output(out_path.as_deref(), &dst).map_err(|e| format!("{out_label}: {e}"))?;
        if should_remove_source(remove_src_file, out_path.as_deref()) {
            remove_source_file(input)?;
        }
        let effective_display_level = if output_is_stdout && display_level == 2 {
            1
        } else {
            display_level
        };
        if effective_display_level >= 2 {
            let verb = if decompress {
                "decompressed"
            } else {
                "compressed"
            };
            let ratio = if !decompress && !src.is_empty() {
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
    let args = std::env::args_os().collect::<Vec<_>>();
    let display_args = normalize_attached_field_args(args.clone()).unwrap_or(args);
    let display_level = raw_display_level(&display_args);
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            if display_level >= 1 && !e.is_empty() {
                eprintln!("zstd: {e}");
            }
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }

    #[test]
    fn repeated_quiet_and_verbose_flags_match_upstream_counts() {
        let cli = Cli::try_parse_from(
            normalize_level_short_args(args(&["zstd", "-qq", "-vv", "-c", "-"])).unwrap(),
        )
        .expect("parse repeated quiet/verbose");

        assert_eq!(cli.quiet, 2);
        assert_eq!(cli.verbose, 2);
        assert_eq!(raw_quiet_level(&args(&["zstd", "-qq", "-D", "-q", "-"])), 2);
        assert_eq!(raw_quiet_level(&args(&["zstd", "--quiet", "--", "-qq"])), 1);
        assert_eq!(raw_display_level(&args(&["zstd", "-qv", "file"])), 2);
        assert_eq!(raw_display_level(&args(&["zstd", "-qqv", "file"])), 1);
    }

    #[test]
    fn version_directive_exits_at_first_upstream_version_flag() {
        assert_eq!(
            upstream_version_directive(&args(&["zstd", "-V"])),
            Some((2, 1))
        );
        assert_eq!(
            upstream_version_directive(&args(&["zstd", "--version", "--level", "9"])),
            Some((2, 1))
        );
        assert_eq!(
            upstream_version_directive(&args(&["zstd", "-qV", "--format", "bad"])),
            Some((1, 1))
        );
        assert_eq!(
            upstream_version_directive(&args(&["zstd", "-vV"])),
            Some((3, 1))
        );
        assert_eq!(
            upstream_version_directive(&args(&["zstd", "--", "-V"])),
            None
        );
        let argv = args(&["zstd", "-D", "-V"]);
        let (_, version_index) = upstream_version_directive(&argv).unwrap();
        assert!(validate_args_before_version_exit(&argv, version_index).is_err());
    }

    #[test]
    fn terminal_directive_uses_first_upstream_exit_flag() {
        assert_eq!(
            upstream_terminal_directive(&args(&["zstd", "--help", "--version"])),
            Some((TerminalDirective::LongHelp, 2, 1))
        );
        assert_eq!(
            upstream_terminal_directive(&args(&["zstd", "--version", "--help"])),
            Some((TerminalDirective::Version, 2, 1))
        );
        assert_eq!(
            upstream_terminal_directive(&args(&["zstd", "-hV"])),
            Some((TerminalDirective::ShortHelp, 2, 1))
        );
        assert_eq!(
            upstream_terminal_directive(&args(&["zstd", "-qH", "--format"])),
            Some((TerminalDirective::LongHelp, 1, 1))
        );
    }

    #[test]
    fn pass_through_directive_uses_last_upstream_flag() {
        assert_eq!(
            last_pass_through_directive(&args(&["zstd", "--pass-through"])),
            Some(true)
        );
        assert_eq!(
            last_pass_through_directive(&args(&["zstd", "--pass-through", "--no-pass-through"])),
            Some(false)
        );
        assert_eq!(
            last_pass_through_directive(&args(&["zstd", "--no-pass-through", "--pass-through"])),
            Some(true)
        );
    }

    #[test]
    fn checksum_directive_accepts_upstream_short_c_and_uses_last_flag() {
        assert_eq!(last_checksum_directive(&args(&["zstd", "-C"])), Some(true));
        assert_eq!(
            last_checksum_directive(&args(&["zstd", "--no-check", "-C"])),
            Some(true)
        );
        assert_eq!(
            last_checksum_directive(&args(&["zstd", "-C", "--no-check"])),
            Some(false)
        );
        assert_eq!(
            last_checksum_directive(&args(&["zstd", "-qC", "input"])),
            Some(true)
        );
        assert_eq!(
            last_checksum_directive(&args(&["zstd", "-D=pathC", "-"])),
            None
        );
    }

    #[test]
    fn attached_separate_field_args_follow_upstream_guard() {
        assert!(reject_attached_separate_field_args(&args(&["zstd", "-D", "dict"])).is_ok());
        assert_eq!(
            normalize_attached_field_args(args(&["zstd", "-D=-q"])).unwrap(),
            args(&["zstd", "--__zstd_pure_internal_dict=-q"])
        );
        assert_eq!(
            reject_attached_separate_field_args(
                &normalize_attached_field_args(args(&["zstd", "-D", "-q"])).unwrap()
            )
            .unwrap_err(),
            "command cannot be separated from its argument by another command"
        );

        assert!(reject_attached_separate_field_args(&args(&["zstd", "-o", "out"])).is_ok());
        assert_eq!(
            normalize_attached_field_args(args(&["zstd", "-o=-q"])).unwrap(),
            args(&["zstd", "--__zstd_pure_internal_output_file=-q"])
        );
        assert_eq!(
            reject_attached_separate_field_args(
                &normalize_attached_field_args(args(&["zstd", "-o", "-q"])).unwrap()
            )
            .unwrap_err(),
            "command cannot be separated from its argument by another command"
        );
        assert_eq!(
            normalize_attached_field_args(args(&["zstd", "-co", "out", "in"])).unwrap(),
            args(&["zstd", "-c", "-o", "out", "in"])
        );
        assert_eq!(
            normalize_attached_field_args(args(&["zstd", "-oc", "out", "in"])).unwrap(),
            args(&["zstd", "-o", "out", "-c", "in"])
        );
        assert_eq!(
            normalize_attached_field_args(args(&["zstd", "-oout", "in"])).unwrap_err(),
            "error: missing command argument "
        );
        assert_eq!(
            normalize_attached_field_args(args(&["zstd", "-o-", "in"])).unwrap_err(),
            "Incorrect parameter: --"
        );

        assert_eq!(
            reject_long_equals_field_args(&args(&["zstd", "--dict=dict"])).unwrap_err(),
            "Incorrect parameter: --dict=dict"
        );
        assert_eq!(
            reject_long_equals_field_args(&args(&["zstd", "--output-file=out"])).unwrap_err(),
            "Incorrect parameter: --output-file=out"
        );
        assert_eq!(
            reject_long_field_alias_args(&args(&["zstd", "--dict", "dict"])).unwrap_err(),
            "Incorrect parameter: --dict"
        );
        assert_eq!(
            reject_long_field_alias_args(&args(&["zstd", "--output-file", "out"])).unwrap_err(),
            "Incorrect parameter: --output-file"
        );
    }

    #[test]
    fn level_aliases_are_rejected_like_upstream() {
        assert_eq!(
            reject_level_alias_args(&args(&["zstd", "--level", "9"])).unwrap_err(),
            "Incorrect parameter: --level"
        );
        assert_eq!(
            reject_level_alias_args(&args(&["zstd", "--level=9"])).unwrap_err(),
            "Incorrect parameter: --level=9"
        );
        assert_eq!(
            reject_level_alias_args(&args(&["zstd", "-L9"])).unwrap_err(),
            "Incorrect parameter: -L"
        );
        assert_eq!(
            reject_level_alias_args(&args(&["zstd", "-qL9"])).unwrap_err(),
            "Incorrect parameter: -L"
        );
        assert!(reject_level_alias_args(&args(&["zstd", "--", "--level=9"])).is_ok());
    }

    #[test]
    fn format_option_requires_upstream_equals_spelling() {
        let cli = Cli::try_parse_from(args(&["zstd", "--format=zstd", "-c", "-"]))
            .expect("parse upstream --format=zstd spelling");
        assert_eq!(cli.format.as_deref(), Some("zstd"));

        assert_eq!(
            reject_format_field_args(&args(&["zstd", "--format", "zstd", "-c", "-"])).unwrap_err(),
            "Incorrect parameter: --format"
        );
    }

    #[test]
    fn empty_adapt_equals_is_rejected_but_bare_adapt_is_allowed() {
        assert_eq!(
            reject_empty_adapt_equals_args(&args(&["zstd", "--adapt=", "-c", "-"])).unwrap_err(),
            "Incorrect parameter: --adapt="
        );
        assert!(reject_empty_adapt_equals_args(&args(&["zstd", "--adapt", "-c", "-"])).is_ok());
    }

    #[test]
    fn fast_and_long_options_use_upstream_numeric_prefix_parsing() {
        let fast = normalize_fast_args(args(&["zstd", "--fast=1", "-c", "-"]))
            .expect("numeric fast level is accepted");
        assert_eq!(
            fast,
            args(&["zstd", "--__zstd_pure_internal_fast=1", "-c", "-"])
        );
        assert_eq!(
            normalize_fast_args(args(&["zstd", "--fast=0abc", "-c", "-"])).unwrap_err(),
            "Incorrect parameter: --fast=0abc"
        );
        assert_eq!(
            normalize_fast_args(args(&["zstd", "--fast=1K", "-c", "-"])).unwrap(),
            args(&["zstd", "--__zstd_pure_internal_fast=1024", "-c", "-"])
        );
        assert_eq!(
            normalize_fast_args(args(&["zstd", "--fast=1M", "-c", "-"])).unwrap(),
            args(&["zstd", "--__zstd_pure_internal_fast=131071", "-c", "-"])
        );
        assert_eq!(
            normalize_fast_args(args(&["zstd", "--fast=1G", "-c", "-"])).unwrap(),
            args(&["zstd", "--__zstd_pure_internal_fast=131071", "-c", "-"])
        );
        assert_eq!(
            normalize_fast_args(args(&["zstd", "--fast=1abc", "-c", "-"])).unwrap(),
            args(&["zstd", "--__zstd_pure_internal_fast=1", "-c", "-"])
        );
        assert_eq!(
            normalize_fast_args(args(&["zstd", "--fast=1Kabc", "-c", "-"])).unwrap(),
            args(&["zstd", "--__zstd_pure_internal_fast=1024", "-c", "-"])
        );
        assert_eq!(
            normalize_fast_args(args(&["zstd", "--fast=4294967296", "-c", "-"])).unwrap_err(),
            "error: numeric value overflows 32-bit unsigned int"
        );
        assert_eq!(long_window_log(Some("abc")).unwrap(), Some(0));
        assert_eq!(long_window_log(Some("")).unwrap(), Some(0));
        assert_eq!(long_window_log(Some("1abc")).unwrap(), Some(1));
        assert_eq!(long_window_log(Some("1Kabc")).unwrap(), Some(1024));
        assert_eq!(long_window_log(Some("1Gabc")).unwrap(), Some(1073741824));
        assert_eq!(long_window_log(Some("27")).unwrap(), Some(27));
        assert_eq!(
            long_window_log(Some("4294967296")).unwrap_err(),
            "error: numeric value overflows 32-bit unsigned int"
        );
    }

    #[test]
    fn short_level_digits_accept_upstream_size_suffixes_and_overflow_errors() {
        assert_eq!(
            normalize_level_short_args(args(&["zstd", "-1Kq", "-c", "-"])).unwrap(),
            args(&["zstd", "--level=1024", "-q", "-c", "-"])
        );
        assert_eq!(
            normalize_level_short_args(args(&["zstd", "-1KiBq", "-c", "-"])).unwrap(),
            args(&["zstd", "--level=1024", "-q", "-c", "-"])
        );
        assert_eq!(
            normalize_level_short_args(args(&["zstd", "-1Gq", "-c", "-"])).unwrap(),
            args(&["zstd", "--level=1073741824", "-q", "-c", "-"])
        );
        assert_eq!(
            normalize_level_short_args(args(&["zstd", "-3Gq", "-c", "-"])).unwrap(),
            args(&["zstd", "--level=-1073741824", "-q", "-c", "-"])
        );
        assert_eq!(
            normalize_level_short_args(args(&["zstd", "-4294967296", "-c", "-"])).unwrap_err(),
            "error: numeric value overflows 32-bit unsigned int"
        );
    }

    #[test]
    fn adapt_parameters_validate_and_bound_levels_like_upstream() {
        assert_eq!(
            parse_adapt_parameters("min=5").unwrap(),
            (5, ZSTD_maxCLevel())
        );
        assert_eq!(
            parse_adapt_parameters("max=1").unwrap(),
            (ZSTD_minCLevel(), 1)
        );
        assert_eq!(parse_adapt_parameters("min=1,max=5").unwrap(), (1, 5));
        assert_eq!(
            parse_adapt_parameters("min=").unwrap(),
            (0, ZSTD_maxCLevel())
        );
        assert_eq!(
            parse_adapt_parameters("min=4294967296").unwrap_err(),
            "error: numeric value overflows 32-bit int"
        );
        assert_eq!(
            parse_adapt_parameters("max=-1").unwrap(),
            (ZSTD_minCLevel(), -1)
        );
        assert_eq!(parse_adapt_parameters("min=-5,max=-1").unwrap(), (-5, -1));
        assert_eq!(
            parse_adapt_parameters("min=3G").unwrap(),
            (-1073741824, ZSTD_maxCLevel())
        );
        assert!(parse_adapt_parameters("max=-3M").is_err());
        assert!(parse_adapt_parameters("bad").is_err());
        assert!(parse_adapt_parameters("min=6,max=2").is_err());

        let (adapt_min, adapt_max) = parse_adapt_parameters("min=5,max=9").unwrap();
        let mut requested_level = 3;
        if requested_level < adapt_min {
            requested_level = adapt_min;
        }
        if requested_level > adapt_max {
            requested_level = adapt_max;
        }
        assert_eq!(requested_level, 5);
    }

    #[test]
    fn ultra_unlocks_upstream_high_compression_levels() {
        let cli = Cli::try_parse_from(
            normalize_level_short_args(args(&["zstd", "--ultra", "-22", "-c", "input"])).unwrap(),
        )
        .expect("parse --ultra with level 22");

        assert!(cli.ultra);
        assert_eq!(cli.level, 22);
        assert_eq!(resolve_compression_level(22, true, 0), ZSTD_maxCLevel());
        assert_eq!(resolve_compression_level(22, false, 0), 19);
    }

    #[test]
    fn output_file_dash_rejection_tracks_source_spelling() {
        assert!(output_file_stdout_mark_is_rejected(&args(&[
            "zstd", "-o", "-", "-"
        ])));
        assert!(output_file_stdout_mark_is_rejected(&args(&[
            "zstd",
            "--output-file=-",
            "-"
        ])));
        assert!(!output_file_stdout_mark_is_rejected(&args(&[
            "zstd", "-o=-", "in.txt"
        ])));
        let normalized = normalize_attached_field_args(args(&["zstd", "-o=-", "in.txt"])).unwrap();
        assert_eq!(
            last_output_directive(&normalized),
            Some(OutputDirective::File)
        );
        let normalized =
            normalize_attached_field_args(args(&["zstd", "-oc", "out", "in.txt"])).unwrap();
        assert_eq!(
            last_output_directive(&normalized),
            Some(OutputDirective::Stdout)
        );
    }

    #[test]
    fn path_fields_use_last_upstream_value_across_equal_and_separate_forms() {
        let normalized =
            normalize_attached_field_args(args(&["zstd", "-o", "one", "-o=two", "in.txt"]))
                .unwrap();
        assert_eq!(
            last_path_field(&normalized, "-o", "--__zstd_pure_internal_output_file"),
            Some(PathBuf::from("two"))
        );

        let normalized =
            normalize_attached_field_args(args(&["zstd", "-o=one", "-o", "two", "in.txt"]))
                .unwrap();
        assert_eq!(
            last_path_field(&normalized, "-o", "--__zstd_pure_internal_output_file"),
            Some(PathBuf::from("two"))
        );

        let normalized =
            normalize_attached_field_args(args(&["zstd", "-D", "dict1", "-D=dict2", "in.txt"]))
                .unwrap();
        assert_eq!(
            last_path_field(&normalized, "-D", "--__zstd_pure_internal_dict"),
            Some(PathBuf::from("dict2"))
        );

        let normalized =
            normalize_attached_field_args(args(&["zstd", "-D=dict1", "-D", "dict2", "in.txt"]))
                .unwrap();
        assert_eq!(
            last_path_field(&normalized, "-D", "--__zstd_pure_internal_dict"),
            Some(PathBuf::from("dict2"))
        );
    }

    #[test]
    fn decompress_output_name_restores_tar_suffix_for_tzst() {
        assert_eq!(
            infer_output_path(Path::new("archive.tzst"), true, "zstd"),
            Some(PathBuf::from("archive.tar"))
        );
        assert_eq!(
            infer_output_path(Path::new("archive.tar.zst"), true, "zstd"),
            Some(PathBuf::from("archive.tar"))
        );
        assert_eq!(infer_output_path(Path::new(".zst"), true, "zstd"), None);
        assert_eq!(infer_output_path(Path::new(".tzst"), true, "zstd"), None);
    }

    #[test]
    fn compress_output_name_uses_selected_format_suffix() {
        assert_eq!(
            infer_output_path(Path::new("payload"), false, "zstd"),
            Some(PathBuf::from("payload.zst"))
        );
        assert_eq!(
            infer_output_path(Path::new("payload"), false, "gzip"),
            Some(PathBuf::from("payload.gz"))
        );
    }

    #[test]
    fn stdin_compression_does_not_emit_content_size_without_known_source_size() {
        let payload = b"stdin content-size parity payload".repeat(4);
        let from_file = compress_bytes(&payload, 3, None, false, true, true, false, None, true)
            .expect("known-size compression");
        let from_stdin = compress_bytes(&payload, 3, None, false, true, true, false, None, false)
            .expect("unknown-size compression");

        assert_eq!(ZSTD_getFrameContentSize(&from_file), payload.len() as u64);
        assert_eq!(
            ZSTD_getFrameContentSize(&from_stdin),
            ZSTD_CONTENTSIZE_UNKNOWN
        );
        assert_eq!(
            decompress_bytes(&from_stdin, None, false, false).unwrap(),
            payload
        );

        let empty_stdin = compress_bytes(&[], 3, None, false, true, true, false, None, false)
            .expect("empty unknown-size compression");
        assert_eq!(
            ZSTD_getFrameContentSize(&empty_stdin),
            ZSTD_CONTENTSIZE_UNKNOWN
        );
        assert!(decompress_bytes(&empty_stdin, None, false, false)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn output_dash_follows_resolved_output_directive() {
        assert_eq!(
            output_path_for_input(Path::new("payload"), false, true, None, "zstd").unwrap(),
            None
        );
        assert_eq!(
            output_path_for_input(
                Path::new("payload"),
                false,
                false,
                Some(&PathBuf::from("-")),
                "zstd",
            )
            .unwrap(),
            Some(PathBuf::from("-"))
        );
        assert_eq!(
            output_path_for_input(
                Path::new("payload"),
                false,
                true,
                Some(&PathBuf::from("-")),
                "zstd",
            )
            .unwrap(),
            Some(PathBuf::from("-"))
        );
    }

    #[test]
    fn gzip_stored_blocks_roundtrip_and_validate_trailer() {
        let payload = b"stored gzip payload ".repeat(5000);
        let gz = gzip_stored_blocks(&payload);
        assert_eq!(gzip_decompress_stored_blocks(&gz).unwrap(), payload);

        let mut corrupted = gz.clone();
        let trailer_pos = corrupted.len() - 8;
        corrupted[trailer_pos] ^= 0x01;
        assert_eq!(
            gzip_decompress_stored_blocks(&corrupted).unwrap_err(),
            "gzip checksum mismatch"
        );
    }

    #[test]
    fn short_decompression_inputs_match_upstream_header_errors() {
        let input = Path::new("short.zst");
        assert_eq!(
            decompress_or_passthrough_bytes(input, b"", None, false, false, false).unwrap_err(),
            "short.zst: unexpected end of file "
        );
        assert_eq!(
            decompress_or_passthrough_bytes(input, b"abc", None, false, false, false).unwrap_err(),
            "short.zst: unknown header "
        );
        assert_eq!(
            decompress_or_passthrough_bytes(input, b"", None, false, false, true).unwrap_err(),
            "short.zst: unexpected end of file "
        );
        assert_eq!(
            decompress_or_passthrough_bytes(input, b"abc", None, false, false, true).unwrap(),
            b"abc"
        );
    }

    #[test]
    fn compat_filter_drains_stdout_while_feeding_stdin() {
        let payload = vec![b'x'; 2 * 1024 * 1024];
        let out = run_filter("cat", &[], &payload, "cat").unwrap();
        assert_eq!(out, payload);
    }

    #[test]
    fn same_existing_input_and_output_is_rejected_even_with_force() {
        let path = std::env::temp_dir().join(format!(
            "zstd_pure_same_input_output_{}",
            std::process::id()
        ));
        fs::write(&path, b"same file guard").unwrap();

        let result = reject_same_input_output(&path, Some(&path));
        let _ = fs::remove_file(&path);

        assert_eq!(
            result.unwrap_err(),
            "Refusing to open an output file which will overwrite the input file"
        );
    }

    #[test]
    fn rm_source_lifecycle_skips_stdin_and_stdout() {
        assert!(!effective_remove_source(true, true, false));
        assert!(!effective_remove_source(true, false, true));
        assert!(effective_remove_source(true, false, false));
        assert!(!effective_remove_source(false, false, false));
        assert!(!should_remove_source(true, None));
        assert!(should_remove_source(true, Some(Path::new("out.zst"))));
        assert!(!should_remove_source(false, Some(Path::new("out.zst"))));
        assert!(remove_source_file(Path::new("-")).is_ok());
    }

    #[test]
    fn clap_parse_errors_return_through_quiet_gate() {
        let err = Cli::try_parse_from(
            normalize_level_short_args(args(&["zstd", "-qq", "--not-a-real-option"])).unwrap(),
        )
        .unwrap_err();

        assert!(!matches!(
            err.kind(),
            ErrorKind::DisplayHelp | ErrorKind::DisplayVersion
        ));
        assert_eq!(
            raw_quiet_level(&args(&["zstd", "-qq", "--not-a-real-option"])),
            2
        );
    }
}
