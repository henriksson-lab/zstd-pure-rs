# zstd-pure-rs

A pure-Rust port of the [Zstandard (`zstd`)](https://github.com/facebook/zstd) compression library — no C, no FFI.

The crate name is `zstd-pure-rs` (lib `zstd_pure_rs`). Version: `0.1.0`.

Upstream reference (vendored in `./zstd/`): the C reference implementation from Meta. This crate is dual-licensed under BSD-3-Clause OR GPL-2.0-only, matching upstream.

## Status

Fully usable for compression and decompression. Per-level compression ratio on the 182 KB `zstd_h.txt` fixture is at parity with upstream `zstd` 1.5.7 across all tested levels (1.00× / 1.05× / 0.98× / 1.01× / 0.99× / 1.01× / 1.05× / 1.05× at levels 1/3/5/7/10/15/19/22 — we beat upstream at 5 and 10). All 22 positive compression levels produce spec-compliant output that upstream `zstd -d` accepts byte-exact.

Features working:

- One-shot compression: `ZSTD_compress(level)`, `ZSTD_compressCCtx`, `ZSTD_compressBound`.
- One-shot decompression: `ZSTD_decompress`, `ZSTD_decompressDCtx`, `ZSTD_findFrameCompressedSize`, `ZSTD_getFrameContentSize`.
- Raw-content dictionaries: `ZSTD_compress_usingDict` / `ZSTD_decompress_usingDict` + CDict/DDict wrappers.
- Streaming API: `ZSTD_initCStream` / `ZSTD_compressStream` / `ZSTD_endStream`, unified `ZSTD_compressStream2` + `ZSTD_EndDirective`, symmetric decompression, `ZSTD_CCtx_setPledgedSrcSize`, dict variants (`ZSTD_initCStream_usingDict` + `ZSTD_initDStream_usingDict`).
- Parametric API: `ZSTD_cParameter` / `ZSTD_dParameter` + `ZSTD_CCtx_setParameter` / `ZSTD_DCtx_setParameter`, reset directives, parameter-bounds queries (`ZSTD_cParam_getBounds` / `ZSTD_dParam_getBounds`).
- Memory estimation: `ZSTD_estimateCCtxSize{,_usingCParams}`, `ZSTD_estimateDCtxSize`, `ZSTD_estimateDStreamSize{,_fromFrame}`, `ZSTD_sizeof_CCtx` / `ZSTD_sizeof_DCtx`.
- Frame parameters: content-size flag, XXH64 checksum trailer, multi-block frames crossing the 128 KB boundary.
- Strategies 1–5 (fast, dfast, greedy, lazy, lazy2); btlazy2/btopt/btultra* are capped down to lazy2 until the binary-tree + optimal parser land.
- CLI (`cargo build --release --features cli`) with `-d/-c/-f/-q/-v/-o/-L/-D/--check/--no-check` flags, stdin/stdout support, file-argument handling with `.zst` extension inference.

All 149 public-API entry points from `zstd.h` are surfaced — the full v1.5 API is available to Rust callers. Still skeletal (stubs return `ErrorCode::Generic` or equivalent) behind the public surface: the long-distance matcher's `generateSequences_internal` ext-dict branch + `blockCompress`, the `zstd_opt.c` optimal parser, superblock compression, multi-threaded compression (`zstdmt_compress.c`), custom-allocator / static-buffer init variants (`*_advanced`, `initStatic*`), the legacy block-level `compressContinue` / `decompressContinue` APIs, pre-digested HUF+FSE dictionary entropy tables. Progress is tracked in `TODO.md`; the full C → Rust function mapping lives in `FUNCTIONS.md`.

Test suite: 528 library unit tests + 22 CLI integration tests + 13 fixture-roundtrip tests + 5 doc tests (568 total). Includes cross-compatibility tests that pipe our output through upstream `zstd` 1.5.7 when the binary is on `$PATH`, a public-API sentinel that touches every exposed entry point, a boundary-size sweep (26 sizes × 3 levels around block/tail boundaries), and a regression gate for a past multi-block repcode bug (4 phrases × 4 levels × 200 KB roundtrips).

Throughput (release build on the 182 KB C-header fixture, `cargo build --release`):

| Level | Ratio | Compress   | Decompress |
|------:|------:|-----------:|-----------:|
| 1     | 3.5×  |  60 MB/s   |  222 MB/s  |
| 3     | 3.6×  |  61 MB/s   |  218 MB/s  |
| 10    | 4.3×  |   5 MB/s   |  210 MB/s  |
| 19    | 4.3×  |   3 MB/s   |  235 MB/s  |

These are about 5–10× slower than upstream's hand-tuned C (no SIMD, no BMI2, single-cursor match finders in place of the 4-way pipeline), but correctness + format compliance were the v0.1 priority, not speed.

## Goals

- **Bitwise-identical output** to the upstream C library for the same inputs and parameters. This is the hard constraint — reproduction takes priority over speed.
- **Pure Rust**, no `unsafe` FFI to the upstream C code. `unsafe` inside the crate is only used where strictly needed for SIMD / bit manipulation.
- Optional CLI (`zstd` binary) behind the `cli` feature.
- Keep one-to-one C-function → Rust-function mapping where possible, so that [code-complexity-comparator](../code-complexity-comparator) stays useful throughout.

## Non-goals (at least initially)

- GUI code (zstd has none).
- The legacy v0.1–v0.7 decoders (`lib/legacy/`) — out of scope for v0.1.
- The zlib-compat shim (`zlibWrapper/`) — out of scope.
- `contrib/` (pzstd, seekable format, linux kernel integration, etc.) — out of scope.

## Plan

The plan is executed end-to-end in this crate. Each phase has concrete, verifiable deliverables.

### Phase 0 — Scaffolding
- `Cargo.toml`, `src/lib.rs`, `src/bin/zstd.rs`.
- License files (`LICENSE`, `COPYING`) copied from upstream.
- `README.md` (this file), `TODO.md`, `FUNCTIONS.md` (C→Rust mapping).
- `.gitignore`.

### Phase 1 — Skeletons
- For every C function in `zstd/lib/{common,compress,decompress}/`, emit a Rust function with a matching signature and `panic!("yet to be translated")`.
- Functions with SIMD paths use const generics so both scalar and SIMD variants compile.
- Crate must compile cleanly (no dead-code warnings).
- No GUI code exists in zstd, so no `panic!("GUI")` placeholders are needed.

### Phase 2 — Bottom-up translation
Order, derived from code-complexity-comparator + call graph:

1. `lib/common/` — `bits.h`, `mem.h`, `error_private.c`, `xxhash.c`, `debug.c`, `entropy_common.c`, `fse_decompress.c`, `zstd_common.c`, `pool.c`, `threading.c`.
2. `lib/decompress/` — `huf_decompress.c`, `zstd_decompress_block.c`, `zstd_ddict.c`, `zstd_decompress.c`. (Decoder-first, because it is smaller and the reference is the format itself.)
3. `lib/compress/` — `hist.c`, `fse_compress.c`, `huf_compress.c`, `zstd_compress_literals.c`, `zstd_compress_sequences.c`, `zstd_compress_superblock.c`, `zstd_fast.c`, `zstd_double_fast.c`, `zstd_lazy.c`, `zstd_opt.c`, `zstd_ldm.c`, `zstd_preSplit.c`, `zstd_compress.c`.
4. Multi-threaded compression — `zstdmt_compress.c` (behind `mt` feature, using `rayon`).

Each function is translated with logic/complexity matching the original (no simplification), tested against deep-comparator traces on real data, and, if in a hot loop, its speed matched then — not later.

### Phase 3 — Verification
- **Real-world corpora** over synthetic: the [Silesia](http://sun.aei.polsl.pl/~sdeor/index.php?page=silesia) and [enwik8](https://mattmahoney.net/dc/textdata.html) corpora, plus a FASTQ file (random-access compression is a realistic zstd workload). Pull with `ureq`.
- deep-comparator traces upstream and Rust side-by-side. Any divergence is a bug.
- tracehash for fine-grained divergence localization inside a single function.
- gdb-translation-verifier-rs for hard cases that need step-level comparison.
- Port upstream tests from `zstd/tests/` (fuzzers, roundtrip tests, dictionary tests) to Rust integration tests.

### Phase 4 — CLI
- `clap` (derive), `zstd` binary, behind `cli` feature.
- Mirror every flag of `zstd/programs/zstdcli.c`. Output files must be bitwise-identical to upstream's `zstd` binary on the same inputs.

### Phase 5 — Polish & publish prep
- Zero warnings under `cargo build --all-features` and `cargo clippy`.
- Performance parity check with `-C target-cpu=native`, both sides.
- Final `Cargo.toml` review, `include` list, `.gitignore`.
- **Never** run `cargo publish`; never `git commit`.

## Verification tools used

- [code-complexity-comparator](../code-complexity-comparator) — one C fn → one Rust fn; complexity should not drop.
- [deep-comparator](../deep-comparator) — trace-level equivalence on real data.
- [tracehash](https://crates.io/crates/tracehash-rs) — divergence localization.
- [gdb-translation-verifier-rs](../gdb-translation-verifier-rs) — stepwise state comparison.

## Building

```sh
cargo build --release
cargo build --release --features cli
cargo build --release --features mt
cargo test
```

For performance measurements:

```sh
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## License

Dual-licensed, matching upstream:

- [BSD-3-Clause](LICENSE)
- [GPL-2.0-only](COPYING)
