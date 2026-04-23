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
- Strategies 1–9 (fast, dfast, greedy, lazy, lazy2, btlazy2, btopt, btultra, btultra2), including no-dict, ext-dict, dict-match-state, row-hash, and LDM-assisted optimal-parser paths.
- CLI (`cargo build --release --features cli`) with `-d/-c/-f/-q/-v/-o/-L/-D/--check/--no-check/--magicless` flags, stdin/stdout support, file-argument handling with `.zst` extension inference.

The full v1.6 public API from `zstd.h` (stable + experimental) is surfaced — every upstream entry point has a Rust counterpart reachable through `zstd_pure_rs::prelude::*`. The current C→Rust function-name coverage backlog is closed for both compression and decompression under `code-complexity-comparator`; remaining gaps are mostly verification breadth, performance/shape differences from safe scalar factoring, and CLI flag completeness rather than missing library entry points. Magic-prefix dictionary entropy **decode** is live via `ZSTD_DCtx_loadDictionary` / `ZSTD_decompress_insertDictionary` / `ZSTD_loadDEntropy`, and DDict full-dictionary entropy is copied into DCtx state when attached. Progress is tracked in `TODO.md`; the full C → Rust function mapping lives in `FUNCTIONS.md`.

Test suite: 816 library unit tests without MT and 834 with `--features mt` as of the latest local run, plus CLI/integration/fixture/doc tests in the full suite. Coverage includes cross-compatibility tests that pipe our output through upstream `zstd` when the binary is on `$PATH`, a public-API sentinel that touches every exposed entry point, a boundary-size sweep around block/tail boundaries, a regression gate for a past multi-block repcode bug, end-to-end `--magicless` CLI roundtrips, and `refPrefix` one-shot auto-clear gates proving single-use dict semantics match upstream for both compressor and decompressor sides.

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
- Keep one-to-one C-function → Rust-function mapping where possible, so that code-complexity-comparator stays useful throughout.

## Non-goals (at least initially)

- The zlib-compat shim (`zlibWrapper/`) — out of scope.
- `contrib/` (pzstd, seekable format, linux kernel integration, etc.) — out of scope.

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
