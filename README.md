# zstd-pure-rs

A pure-Rust port of the [Zstandard (`zstd`)](https://github.com/facebook/zstd) compression library

**Don't trust anything below**

**This code is incomplete; translation ongoing**

## This is an LLM-mediated faithful (hopefully) translation, not the original code! 

Most users should probably first see if the existing original code works for them, unless they have reason otherwise. The original source
may have newer features and it has had more love in terms of fixing bugs. In fact, we aim to replicate bugs if they are present, for the
sake of reproducibility! (but then we might have added a few more in the process)

There are however cases when you might prefer this Rust version. We generally agree with [this manifesto](https://rewrites.bio/) but more specifically:
* We have had many issues with ensuring that our software works using existing containers (Docker, PodMan, Singularity). One size does not fit all and it eats our resources trying to keep up with every way of delivering software
* Common package managers do not work well. It was great when we had a few Linux distributions with stable procedures, but now there are just too many ecosystems (Homebrew, Conda). Conda has an NP-complete resolver which does not scale. Homebrew is only so-stable. And our dependencies in Python still break. These can no longer be considered professional serious options. Meanwhile, Cargo enables multiple versions of packages to be available, even within the same program(!)
* The future is the web. We deploy software in the web browser, and until now that has meant Javascript. This is a language where even the == operator is broken. Typescript is one step up, but a game changer is the ability to compile Rust code into webassembly, enabling performance and sharing of code with the backend. Translating code to Rust enables new ways of deployment and running code in the browser has especial benefits for science - researchers do not have deep pockets to run servers, so pushing compute to the user enables deployment that otherwise would be impossible
* Old CLI-based utilities are bad for the environment(!). A large amount of compute resources are spent creating and communicating via small files, which we can bypass by using code as libraries. Even better, we can avoid frequent reloading of databases by hoisting this stage, with up to 100x speedups in some cases. Less compute means faster compute and less electricity wasted
* LLM-mediated translations may actually be safer to use than the original code. This article shows that [running the same code on different operating systems can give somewhat different answers](https://doi.org/10.1038/nbt.3820). This is a gap that Rust+Cargo can reduce. Typesafe interfaces also reduce coding mistakes and error handling, as opposed to typical command-line scripting

But:

* **This approach should still be considered experimental**. The LLM technology is immature and has sharp corners. But there are opportunities to reap, and the genie is not going back into the bottle. This translation is as much aimed to learn how to improve the technology and get feedback on the results.
* Translations are not endorsed by the original authors unless otherwise noted. **Do not send bug reports to the original developers**. Use our Github issues page instead.
* **Do not trust the benchmarks on this page**. They are used to help evaluate the translation. If you want improved performance, you generally have to use this code as a library, and use the additional tricks it offers. We generally accept performance losses in order to reduce our dependency issues
* **Check the original Github pages for information about the package**. This README is kept sparse on purpose. It is not meant to be the primary source of information
* **If you are the author of the original code and wish to move to Rust, you can obtain ownership of this repository and crate**. Until then, our commitment is to offer an as-faithful-as-possible translation of a snapshot of your code. If we find serious bugs, we will report them to you. Otherwise we will just replicate them, to ensure comparability across studies that claim to use package XYZ v.666. Think of this like a fancy Ubuntu .deb-package of your software - that is how we treat it

This blurb might be out of date. Go to [this page](https://github.com/henriksson-lab/rustification) for the latest information and further information about how we approach translation


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
- CLI (`cargo build --release --features cli`) with `-d/-c/-f/-q/-v/-o/-L/-D/--check/--no-check/--magicless` flags, stdin/stdout support, file-argument handling with `.zst` extension inference.

The full v1.6 public API from `zstd.h` (stable + experimental) is surfaced — every upstream entry point has a Rust counterpart reachable through `zstd_pure_rs::prelude::*`. Still skeletal (stubs return `ErrorCode::Generic` or equivalent) behind the public surface: the long-distance matcher's `generateSequences_internal` ext-dict branch + `blockCompress`, the `zstd_opt.c` optimal parser, superblock compression, multi-threaded compression (`zstdmt_compress.c`), custom-allocator / static-buffer init variants (`*_advanced`, `initStatic*`), the legacy block-level `compressContinue` / `decompressContinue` APIs, and the compressor-side magic-prefix dict entropy seeding (`ZSTD_compress_usingDict` takes the raw-content path). Magic-prefix dict entropy **decode** is live (via `ZSTD_DCtx_loadDictionary` / `ZSTD_decompress_insertDictionary` / `ZSTD_loadDEntropy`). Progress is tracked in `TODO.md`; the full C → Rust function mapping lives in `FUNCTIONS.md`.

Test suite: 724 library unit tests + 33 CLI integration tests + 13 fixture-roundtrip tests + 5 doc tests (775 total). Includes cross-compatibility tests that pipe our output through upstream `zstd` 1.5.7 when the binary is on `$PATH`, a public-API sentinel that touches every exposed entry point, a boundary-size sweep (28 sizes × 3 levels around block/tail boundaries), a regression gate for a past multi-block repcode bug (4 phrases × 4 levels × 200 KB roundtrips), end-to-end `--magicless` CLI roundtrips (bare, with `-D` dict, with `--check` checksum, triple-flag composition, and across a 6-level strategy sweep), and `refPrefix` one-shot auto-clear gates proving single-use dict semantics match upstream for both the compressor (`endStream` / `compress2`) and decompressor (`decompressDCtx` / `decompressStream`) sides.

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
- **Pure Rust**, no `unsafe` FFI to the upstream C code. Only a single `unsafe` block remains inside the crate (a `u32`→`u8` slice reinterpret helper in `common/entropy_common.rs` for passing an FSE workspace by raw bytes).
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
- Zero warnings under `cargo build --all-features` and `cargo clippy --all-targets --features cli -- -D warnings`.
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

Same as the original code, [BSD-3-Clause](LICENSE)
