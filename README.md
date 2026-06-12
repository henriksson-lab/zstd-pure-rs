# zstd-pure-rs

A pure-Rust port of the [Zstandard (`zstd`)](https://github.com/facebook/zstd) compression library

* 2026-06-02: Big audit
* 2026-04-27: Tested locally for core compression/decompression behavior. Treat performance and parity notes as status snapshots, not guarantees.
* Some features out of scope. Contact if you need them

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
* **Do not treat README status notes as performance guarantees**. Local benchmarks are used to help evaluate the translation, but reproducibility and dependency reduction take priority over speed claims here
* **Check the original Github pages for information about the package**. This README is kept sparse on purpose. It is not meant to be the primary source of information
* **If you are the author of the original code and wish to move to Rust, you can obtain ownership of this repository and crate**. Until then, our commitment is to offer an as-faithful-as-possible translation of a snapshot of your code. If we find serious bugs, we will report them to you. Otherwise we will just replicate them, to ensure comparability across studies that claim to use package XYZ v.666. Think of this like a fancy Ubuntu .deb-package of your software - that is how we treat it

This blurb might be out of date. Go to [this page](https://github.com/henriksson-lab/rustification) for the latest information and further information about how we approach translation


## Status

Usable for core compression and decompression, with ongoing CLI/API parity work. Focused local tests exercise all positive compression levels, and the CLI integration suite checks representative upstream `zstd -d` compatibility when `zstd` is available on `PATH`; this is compatibility evidence, not a performance guarantee.

Features working:

- One-shot compression: `ZSTD_compress(level)`, `ZSTD_compressCCtx`, `ZSTD_compressBound`.
- One-shot decompression: `ZSTD_decompress`, `ZSTD_decompressDCtx`, `ZSTD_findFrameCompressedSize`, `ZSTD_getFrameContentSize`.
- Raw-content dictionaries: `ZSTD_compress_usingDict` / `ZSTD_decompress_usingDict` + CDict/DDict wrappers.
- Buffered streaming compatibility wrappers: `ZSTD_initCStream` / `ZSTD_compressStream` / `ZSTD_endStream`, unified `ZSTD_compressStream2` + `ZSTD_EndDirective`, symmetric decompression, `ZSTD_CCtx_setPledgedSrcSize`, dict variants (`ZSTD_initCStream_usingDict` + `ZSTD_initDStream_usingDict`). Compression buffers input until `endStream`; block-by-block flush semantics are not yet equivalent to upstream.
- Parametric API: `ZSTD_cParameter` / `ZSTD_dParameter` + `ZSTD_CCtx_setParameter` / `ZSTD_DCtx_setParameter`, reset directives, parameter-bounds queries (`ZSTD_cParam_getBounds` / `ZSTD_dParam_getBounds`).
- Memory estimation: `ZSTD_estimateCCtxSize{,_usingCParams}`, `ZSTD_estimateDCtxSize`, `ZSTD_estimateDStreamSize{,_fromFrame}`, `ZSTD_sizeof_CCtx` / `ZSTD_sizeof_DCtx`.
- Frame parameters: content-size flag, XXH64 checksum trailer, multi-block frames crossing the 128 KB boundary.
- Strategies 1–9 (fast, dfast, greedy, lazy, lazy2, btlazy2, btopt, btultra, btultra2), including no-dict, ext-dict, dict-match-state, row-hash, and LDM-assisted optimal-parser paths.
- CLI (`cargo build --release --features cli`) with `-d/-c/-f/-q/-v/-o/-D/--check/--no-check/--magicless` flags, upstream-style level flags such as `-1` and `-19`, local `-L/--level` level selection, buffered stdin/stdout support, file-argument handling with `.zst`/`.zstd` extension inference and unknown-suffix rejection unless `-c`/`-o` is explicit, last-wins `-c`/`-o` and `--check`/`--no-check` handling, decode-side checksum validation when present, decode-side `--no-check`, `-d -c -f` stdout pass-through for unrecognized input, and multi-input `-o` rejection.

The main v1.6 `zstd.h` `ZSTD_` entry points and many experimental helpers, including translated helpers such as `ZSTD_compressSequencesAndLiterals`, are surfaced through `zstd_pure_rs::prelude::*`, but lower-level `HUF_` and `FSE_` helpers remain in their module namespaces unless explicitly re-exported there. Some parameter IDs and edge APIs remain intentionally unsupported and return the matching error codes. The current C→Rust function-name coverage backlog is closed for both compression and decompression under `code-complexity-comparator`; remaining gaps are mostly verification breadth, performance/shape differences from safe scalar factoring, unsupported parameter variants, and CLI flag completeness. Magic-prefix dictionary entropy **decode** is live via `ZSTD_DCtx_loadDictionary` / `ZSTD_decompress_insertDictionary` / `ZSTD_loadDEntropy`, and DDict full-dictionary entropy is copied into DCtx state when attached. Focused follow-up notes live in `TODO.md`; the full C → Rust function mapping lives in `FUNCTIONS.md`.

Test suite status as of the latest local audit run: `cargo test --features cli` passes. That includes the focused CLI integration suite, regular library/API unit tests, doc tests, and integration tests. The CLI suite covers file/stdin/stdout behavior, cross-compatibility cases that run only when upstream `zstd` is on `PATH`, boundary-size and multi-block regressions, end-to-end `--magicless` roundtrips, last-wins `-c`/`-o` and `--check`/`--no-check`, `-N` and clustered level flags, decode-side `--no-check`, no-suffix output rejection before decode, attached `-Ddict` rejection, `-d -c -f` stdout pass-through for unrecognized input, and multi-input `-o` rejection.

## Local benchmark snapshot

Measured 2026-06-12 on Linux 6.8 x86_64, Intel Xeon Gold 6138, `rustc 1.92.0`, release build from `cargo build --release --features cli`. The original comparator is the vendored upstream `zstd/programs/zstd` reporting `v1.6.0`. Input was the deterministic 311,951,360-byte `.tmp/bench/realistic_5x.tar` corpus, built from the public Silesia corpus plus enwik8 Wikipedia text as distinct files in a tar archive. Throughput uses decimal MB/s and median elapsed time across three repeated runs. RSS is the median GNU `/usr/bin/time` maximum resident set size across the same runs. Levels 20-22 were run with `--ultra` for both binaries. This is a local status snapshot, not a guarantee.

| Level | Rust speed | Original speed | Rust / original | Rust RSS | Original RSS | Rust size | Original size | Cross-decode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | 160.8 MB/s | 779.9 MB/s | 0.21x | 5.3 MiB | 25.3 MiB | 113,894,156 | 113,945,914 | pass |
| 2 | 135.0 MB/s | 678.2 MB/s | 0.20x | 5.9 MiB | 46.6 MiB | 106,672,917 | 106,796,772 | pass |
| 3 | 97.5 MB/s | 421.6 MB/s | 0.23x | 7.5 MiB | 81.2 MiB | 101,719,022 | 101,676,160 | pass |
| 4 | 73.1 MB/s | 371.4 MB/s | 0.20x | 8.4 MiB | 85.6 MiB | 99,809,816 | 99,857,954 | pass |
| 5 | 48.5 MB/s | 234.5 MB/s | 0.21x | 8.8 MiB | 86.2 MiB | 96,611,606 | 96,617,715 | pass |
| 6 | 39.8 MB/s | 183.5 MB/s | 0.22x | 9.1 MiB | 86.6 MiB | 94,019,853 | 94,043,466 | pass |
| 7 | 35.2 MB/s | 160.0 MB/s | 0.22x | 11.2 MiB | 95.6 MiB | 92,398,290 | 92,516,005 | pass |
| 8 | 26.9 MB/s | 124.8 MB/s | 0.22x | 71.6 MiB | 96.2 MiB | 91,470,937 | 91,491,427 | pass |
| 9 | 24.0 MB/s | 108.3 MB/s | 0.22x | 74.4 MiB | 187.8 MiB | 90,325,984 | 90,301,620 | pass |
| 10 | 18.5 MB/s | 86.9 MB/s | 0.21x | 84.4 MiB | 227.8 MiB | 89,181,962 | 89,194,111 | pass |
| 11 | 12.7 MB/s | 55.7 MB/s | 0.23x | 84.4 MiB | 227.2 MiB | 88,556,940 | 88,580,173 | pass |
| 12 | 10.1 MB/s | 48.7 MB/s | 0.21x | 104.4 MiB | 306.2 MiB | 88,422,454 | 88,456,447 | pass |
| 13 | 5.7 MB/s | 22.3 MB/s | 0.26x | 96.2 MiB | 274.4 MiB | 87,825,668 | 87,827,852 | pass |
| 14 | 5.7 MB/s | 19.0 MB/s | 0.30x | 112.2 MiB | 338.1 MiB | 87,239,542 | 87,254,148 | pass |
| 15 | 4.5 MB/s | 15.5 MB/s | 0.29x | 128.4 MiB | 402.8 MiB | 86,470,305 | 86,569,224 | pass |
| 16 | 2.9 MB/s | 12.0 MB/s | 0.24x | 97.8 MiB | 273.1 MiB | 83,721,679 | 83,765,173 | pass |
| 17 | 2.1 MB/s | 7.4 MB/s | 0.28x | 109.7 MiB | 464.4 MiB | 81,911,034 | 81,931,719 | pass |
| 18 | 1.7 MB/s | 4.3 MB/s | 0.40x | 110.6 MiB | 464.1 MiB | 80,620,022 | 80,635,069 | pass |
| 19 | 1.6 MB/s | 5.3 MB/s | 0.30x | 142.5 MiB | 593.1 MiB | 79,789,653 | 79,773,363 | pass |
| 20 | 1.3 MB/s | 3.3 MB/s | 0.39x | 198.8 MiB | 856.2 MiB | 78,360,063 | 78,354,300 | pass |
| 21 | 1.2 MB/s | 1.7 MB/s | 0.71x | 390.0 MiB | 1014.4 MiB | 77,770,193 | 77,754,755 | pass |
| 22 | 5.5 MB/s | 1.3 MB/s | 4.23x | 1158.1 MiB | 1087.8 MiB | 159,765,052 | 77,506,752 | pass |

Rust-compressed and original-compressed frames are generally **not byte-identical**. On this larger mixed corpus, upstream decoded each Rust frame back to the original input at every level 1-22. The level-22 Rust row is a post-fix single validation run using the conservative buffered streaming fallback for optimal-parser strategies; it replaces an earlier invalid windowed-stream result that both decoders rejected with `Data corruption detected`. A larger 466,432,000-byte repeat corpus gives a less noisy decompression comparison: Rust file-output median 1665.8 MB/s / 5.0 MiB RSS versus original median 1504.6 MB/s / 4.4 MiB. In test mode (`-t`, no output), Rust now streams at 6663.3 MB/s / 5.0 MiB RSS versus original 5830.4 MB/s / 4.4 MiB RSS; before the streaming test-mode fix, Rust `-t` staged the whole 466 MB output and reached about 458 MiB RSS. The CLI decompression path now decodes into an upstream-style output ring for frames up to a 4 MiB window and uses the whole-buffer decoder above that until the ring path is audited for high-window history.

## Goals

- **Bitwise-identical output** to the upstream C library for the same inputs and parameters. This is the hard constraint — reproduction takes priority over speed.
- **Pure Rust**, no `unsafe` FFI to the upstream C code. The crate still contains a small amount of in-tree `unsafe` (~50 occurrences across 9 files: pointer arithmetic in `compress/zstd_cwksp.rs`, allocator-Box plumbing in `compress/zstd_compress.rs`, raw-pointer slice reinterprets in `decompress/zstd_ddict.rs` and `common/entropy_common.rs`, `Box::from_raw`/`Arc::from_raw` round-trips in `common/pool.rs` + `common/threading.rs`, and a few `offset_from`/`add` calls in `compress/zstd_compress_superblock.rs` + `decompress/zstd_decompress.rs`). Driving this number toward zero is a goal but not a hard requirement.
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

## Library Use

Add the crate and import the prelude:

```rust
use zstd_pure_rs::prelude::*;

let src = b"data to compress".to_vec();

let mut compressed = vec![0u8; ZSTD_compressBound(src.len())];
let c_size = ZSTD_compress(&mut compressed, &src, 3);
assert!(!ERR_isError(c_size), "compress failed: {}", ERR_getErrorName(c_size));
compressed.truncate(c_size);

let mut decoded = vec![0u8; src.len()];
let d_size = ZSTD_decompress(&mut decoded, &compressed);
assert!(!ERR_isError(d_size), "decompress failed: {}", ERR_getErrorName(d_size));
decoded.truncate(d_size);

assert_eq!(decoded, src);
```

Raw-content dictionaries use explicit contexts:

```rust
use zstd_pure_rs::prelude::*;

let dict = b"common words and prefixes ".to_vec();
let src = b"common words and prefixes plus message payload".to_vec();

let mut cctx = ZSTD_createCCtx().expect("compression context");
let mut compressed = vec![0u8; ZSTD_compressBound(src.len())];
let c_size = ZSTD_compress_usingDict(&mut cctx, &mut compressed, &src, &dict, 3);
assert!(!ERR_isError(c_size));
compressed.truncate(c_size);

let mut dctx = ZSTD_createDCtx();
let mut decoded = vec![0u8; src.len() + 64];
let d_size = ZSTD_decompress_usingDict(&mut dctx, &mut decoded, &compressed, &dict);
assert!(!ERR_isError(d_size));
assert_eq!(&decoded[..d_size], &src[..]);
```

For performance measurements:

```sh
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## License

Same as the original code, [BSD-3-Clause](LICENSE)
