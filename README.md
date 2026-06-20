# zstd-pure-rs

A pure-Rust port of the [Zstandard (`zstd`)](https://github.com/facebook/zstd) compression library

**Beware that translation is immature technology. Check that this crate works on your data to avoid data loss**

* 2026-06-20: reached single thread parity again
* 2026-06-19: Renewed attempt at getting speed up to original code. bugs created and fixed in the process
* 2026-06-15: Getting closer to being a trustworthy but more testing needed
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
- Buffered streaming compatibility wrappers: `ZSTD_initCStream` / `ZSTD_compressStream` / `ZSTD_endStream`, unified `ZSTD_compressStream2` + `ZSTD_EndDirective`, symmetric decompression, `ZSTD_CCtx_setPledgedSrcSize`, dict variants (`ZSTD_initCStream_usingDict` + `ZSTD_initDStream_usingDict`). No-dictionary multithreaded compression feeds the translated zstdmt scheduler incrementally when `ZSTD_c_nbWorkers > 0`; dictionary/prefix MT streaming still falls back to the existing buffered-compatible paths until those boundaries are audited.
- Parametric API: `ZSTD_cParameter` / `ZSTD_dParameter` + `ZSTD_CCtx_setParameter` / `ZSTD_DCtx_setParameter`, reset directives, parameter-bounds queries (`ZSTD_cParam_getBounds` / `ZSTD_dParam_getBounds`).
- Memory estimation: `ZSTD_estimateCCtxSize{,_usingCParams}`, `ZSTD_estimateDCtxSize`, `ZSTD_estimateDStreamSize{,_fromFrame}`, `ZSTD_sizeof_CCtx` / `ZSTD_sizeof_DCtx`.
- Frame parameters: content-size flag, XXH64 checksum trailer, multi-block frames crossing the 128 KB boundary.
- Strategies 1–9 (fast, dfast, greedy, lazy, lazy2, btlazy2, btopt, btultra, btultra2), including no-dict, ext-dict, dict-match-state, row-hash, and LDM-assisted optimal-parser paths.
- CLI (`cargo build --release --features cli`) with `-d/-c/-f/-q/-v/-o/-D/-T/--threads/--single-thread/--jobsize/-B/--zstd=overlapLog=/--zstd=ovlog=/--check/--no-check/--magicless` flags, upstream-style level flags such as `-1` and `-19`, local `-L/--level` level selection, buffered stdin/stdout support, file-argument handling with `.zst`/`.zstd` extension inference and unknown-suffix rejection unless `-c`/`-o` is explicit, last-wins `-c`/`-o` and `--check`/`--no-check` handling, decode-side checksum validation when present, decode-side `--no-check`, `-d -c -f` stdout pass-through for unrecognized input, and multi-input `-o` rejection.

The main v1.6 `zstd.h` `ZSTD_` entry points and many experimental helpers, including translated helpers such as `ZSTD_compressSequencesAndLiterals`, are surfaced through `zstd_pure_rs::prelude::*`, but lower-level `HUF_` and `FSE_` helpers remain in their module namespaces unless explicitly re-exported there. Some parameter IDs and edge APIs remain intentionally unsupported and return the matching error codes. The current C→Rust function-name coverage backlog is closed for both compression and decompression under `code-complexity-comparator`; remaining gaps are mostly verification breadth, performance/shape differences from safe scalar factoring, unsupported parameter variants, and CLI flag completeness. Magic-prefix dictionary entropy **decode** is live via `ZSTD_DCtx_loadDictionary` / `ZSTD_decompress_insertDictionary` / `ZSTD_loadDEntropy`, and DDict full-dictionary entropy is copied into DCtx state when attached. The full C → Rust function mapping lives in `FUNCTIONS.md`.

Test suite status as of the latest local audit run: `cargo test --features cli` passes. That includes 1117 library tests, the 26-test zstd binary suite, 100 CLI roundtrip tests, integration fixtures, real-data and upstream-golden suites, and doctests. The CLI suite covers file/stdin/stdout behavior, cross-compatibility cases that run only when upstream `zstd` is on `PATH`, boundary-size and multi-block regressions, end-to-end `--magicless` roundtrips, last-wins `-c`/`-o` and `--check`/`--no-check`, `-N` and clustered level flags, decode-side `--no-check`, no-suffix output rejection before decode, attached `-Ddict` rejection, `-d -c -f` stdout pass-through for unrecognized input, multithreaded `-T1`/`-T2` file roundtrips, hidden MT `--jobsize`/`-B` and `--zstd=overlapLog`/`--zstd=ovlog` parsing, and multi-input `-o` rejection.

## Local benchmark snapshot

Measured 2026-06-20 on Linux 6.8 x86_64, Intel Xeon Gold 6138, `rustc 1.92.0`, generic release build from `cargo build --release --features cli` (no `-C target-cpu=native`; the upstream comparator is a generic vendored `zstd/programs/zstd` build reporting `v1.6.0`). The all-level table uses the deterministic 67,108,864-byte `.tmp/bench/realistic_64m.tar` fixture, built from the public Silesia corpus plus enwik8 Wikipedia text as distinct files in a tar archive. Commands were `--single-thread -LEVEL --no-check -f -q`, with `--ultra` added for levels 20-22. The table reports GNU `/usr/bin/time` user CPU seconds and max RSS; wall-clock timings on this host were noisy. `Rust/orig CPU` is original user time divided by Rust user time, so values below `1.00x` mean Rust used more CPU. `Byte-identical` compares the compressed frames against the vendored upstream CLI. This is a local status snapshot, not a guarantee.

Levels 1-22 currently use the file streaming path and are byte-identical on this fixture. The all-level table is single-threaded for both implementations.

| Level | Rust user | Original user | Rust/orig CPU | Rust RSS | Original RSS | Rust size | Original size | Byte-identical |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | 0.40 s | 0.34 s | 0.85x | 4.7 MiB | 3.1 MiB | 27304326 | 27304326 | ok |
| 2 | 0.55 s | 0.54 s | 0.98x | 5.9 MiB | 4.1 MiB | 25066283 | 25066283 | ok |
| 3 | 0.75 s | 0.57 s | 0.76x | 8.4 MiB | 5.3 MiB | 23889438 | 23889438 | ok |
| 4 | 0.90 s | 0.70 s | 0.78x | 9.7 MiB | 6.6 MiB | 23426701 | 23426701 | ok |
| 5 | 1.42 s | 0.99 s | 0.70x | 10.0 MiB | 6.9 MiB | 22633952 | 22633952 | ok |
| 6 | 1.86 s | 1.47 s | 0.79x | 10.0 MiB | 7.2 MiB | 21906687 | 21906687 | ok |
| 7 | 2.32 s | 1.89 s | 0.81x | 12.5 MiB | 9.7 MiB | 21482070 | 21482070 | ok |
| 8 | 3.07 s | 2.43 s | 0.79x | 12.5 MiB | 9.7 MiB | 21217035 | 21217035 | ok |
| 9 | 3.43 s | 2.48 s | 0.72x | 21.6 MiB | 16.6 MiB | 20876814 | 20876814 | ok |
| 10 | 5.39 s | 3.94 s | 0.73x | 31.6 MiB | 26.6 MiB | 20578309 | 20578309 | ok |
| 11 | 8.44 s | 5.86 s | 0.69x | 31.9 MiB | 26.6 MiB | 20437398 | 20437398 | ok |
| 12 | 9.04 s | 5.71 s | 0.63x | 51.6 MiB | 46.6 MiB | 20385376 | 20385376 | ok |
| 13 | 14.01 s | 12.97 s | 0.93x | 43.4 MiB | 38.8 MiB | 20135315 | 20135315 | ok |
| 14 | 20.28 s | 16.47 s | 0.81x | 59.4 MiB | 54.4 MiB | 20016886 | 20016886 | ok |
| 15 | 23.36 s | 21.63 s | 0.93x | 75.6 MiB | 70.6 MiB | 19779986 | 19779986 | ok |
| 16 | 30.08 s | 24.39 s | 0.81x | 44.4 MiB | 38.8 MiB | 19108591 | 19108591 | ok |
| 17 | 37.82 s | 32.19 s | 0.85x | 68.1 MiB | 58.8 MiB | 18617535 | 18617535 | ok |
| 18 | 44.35 s | 38.79 s | 0.87x | 68.8 MiB | 59.4 MiB | 18369495 | 18369495 | ok |
| 19 | 51.17 s | 45.27 s | 0.88x | 100.6 MiB | 90.9 MiB | 18123577 | 18123577 | ok |
| 20 | 61.53 s | 56.72 s | 0.92x | 227.8 MiB | 194.7 MiB | 17516254 | 17516254 | ok |
| 21 | 64.38 s | 54.41 s | 0.85x | 387.8 MiB | 386.6 MiB | 17295012 | 17295012 | ok |
| 22 | 64.43 s | 58.79 s | 0.91x | 708.4 MiB | 706.6 MiB | 17246022 | 17246022 | ok |

Rust-compressed and original-compressed frames are byte-identical for levels 1-22 on the single-threaded all-level fixture above.

The next table repeats the same fixture with 5 worker threads for both implementations: `-T5 -LEVEL --no-check -f -q`, with `--ultra` added for levels 20-22. Multi-threaded output is valid and cross-decodable but not byte-identical to upstream on this fixture, because the frame/job layout differs.

| Level | Rust user | Original user | Rust/orig CPU | Rust RSS | Original RSS | Rust size | Original size | Byte-identical |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | 0.54 s | 0.40 s | 0.74x | 39.9 MiB | 23.8 MiB | 27310395 | 27311558 | diff |
| 2 | 0.60 s | 0.48 s | 0.80x | 71.3 MiB | 46.6 MiB | 25091812 | 25093181 | diff |
| 3 | 0.88 s | 0.69 s | 0.78x | 127.8 MiB | 87.8 MiB | 23801763 | 23802052 | diff |
| 4 | 0.95 s | 0.77 s | 0.81x | 131.6 MiB | 90.9 MiB | 23342360 | 23342699 | diff |
| 5 | 1.42 s | 1.25 s | 0.88x | 138.1 MiB | 92.8 MiB | 22702112 | 22658348 | diff |
| 6 | 1.96 s | 1.60 s | 0.82x | 139.7 MiB | 92.8 MiB | 21989678 | 21938239 | diff |
| 7 | 2.33 s | 1.91 s | 0.82x | 159.4 MiB | 106.9 MiB | 21597903 | 21537422 | diff |
| 8 | 3.16 s | 2.43 s | 0.77x | 160.3 MiB | 103.8 MiB | 21341458 | 21256467 | diff |
| 9 | 3.51 s | 2.62 s | 0.75x | 201.2 MiB | 126.2 MiB | 20965632 | 20905318 | diff |
| 10 | 5.36 s | 3.77 s | 0.70x | 252.3 MiB | 166.6 MiB | 20684932 | 20615549 | diff |
| 11 | 7.36 s | 5.92 s | 0.80x | 252.9 MiB | 166.2 MiB | 20552059 | 20478281 | diff |
| 12 | 8.76 s | 6.22 s | 0.71x | 356.5 MiB | 245.3 MiB | 20505617 | 20429864 | diff |
| 13 | 18.35 s | 15.70 s | 0.86x | 274.7 MiB | 213.1 MiB | 20169077 | 20171692 | diff |
| 14 | 23.28 s | 19.87 s | 0.85x | 339.8 MiB | 276.9 MiB | 20054600 | 20057234 | diff |
| 15 | 28.37 s | 25.20 s | 0.89x | 404.2 MiB | 340.6 MiB | 19842064 | 19844780 | diff |
| 16 | 34.61 s | 29.08 s | 0.84x | 287.3 MiB | 213.1 MiB | 19122099 | 19124004 | diff |
| 17 | 44.03 s | 39.12 s | 0.89x | 227.4 MiB | 179.4 MiB | 18623807 | 18624691 | diff |
| 18 | 58.02 s | 45.08 s | 0.78x | 227.8 MiB | 181.2 MiB | 18377295 | 18378439 | diff |
| 19 | 65.13 s | 55.64 s | 0.85x | 308.8 MiB | 243.4 MiB | 18124549 | 18126952 | diff |
| 20 | 63.61 s | 53.88 s | 0.85x | 244.4 MiB | 243.4 MiB | 17515104 | 17516923 | diff |
| 21 | 68.93 s | 61.79 s | 0.90x | 403.4 MiB | 401.9 MiB | 17293770 | 17294481 | diff |
| 22 | 72.86 s | 65.74 s | 0.90x | 723.1 MiB | 722.8 MiB | 17245213 | 17245829 | diff |

The current level-1 no-check fast path is byte-identical on the larger real-data fixtures below. Timings are single sequential warm-cache runs using `--single-thread -1 --no-check`; RSS is from `/usr/bin/time`.

| Dataset | Input bytes | Rust wall/user/sys | Original wall/user/sys | Rust/orig CPU | Rust RSS | Original RSS | Rust bytes | Original bytes | Byte-identical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| realistic5x | 311951360 | 2.75/1.66/0.22 s | 1.62/1.44/0.17 s | 0.87x | 5.3 MiB | 3.4 MiB | 113894044 | 113894044 | ok |
| text466m | 466432000 | 0.20/0.11/0.07 s | 0.17/0.09/0.07 s | 0.82x | 5.3 MiB | 3.1 MiB | 97428 | 97428 | ok |
| repo_mix12x | 748083200 | 5.84/1.43/1.00 s | 1.72/1.16/0.56 s | 0.81x | 5.3 MiB | 3.1 MiB | 613071966 | 613071966 | ok |
| random2g | 2147483648 | 2.50/0.90/1.57 s | 2.37/0.73/1.63 s | 0.81x | 5.3 MiB | 3.1 MiB | 2147532810 | 2147532810 | ok |
| micro2700k | 2764801024 | 2.24/1.48/0.56 s | 1.91/1.42/0.49 s | 0.96x | 5.0 MiB | 3.1 MiB | 38037397 | 38037397 | ok |

A larger 466,432,000-byte repeat corpus gives a less noisy decompression comparison: Rust file-output median 1665.8 MB/s / 5.0 MiB RSS versus original median 1504.6 MB/s / 4.4 MiB. In test mode (`-t`, no output), Rust now streams at 6663.3 MB/s / 5.0 MiB RSS versus original 5830.4 MB/s / 4.4 MiB RSS; before the streaming test-mode fix, Rust `-t` staged the whole 466 MB output and reached about 458 MiB RSS. The CLI decompression path now uses the decoder's history-backed streaming path for frames up to a 4 MiB window and uses the whole-buffer decoder above that until high-window streaming history is audited.

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
