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

Test suite status as of the latest local CLI audit run: `cargo test --features cli --test cli_roundtrip` passes the focused CLI integration suite. That CLI suite covers file/stdin/stdout behavior, cross-compatibility cases that run only when upstream `zstd` is on `PATH`, boundary-size and multi-block regressions, end-to-end `--magicless` roundtrips, last-wins `-c`/`-o` and `--check`/`--no-check`, `-N` and clustered level flags, decode-side `--no-check`, no-suffix output rejection before decode, attached `-Ddict` rejection, `-d -c -f` stdout pass-through for unrecognized input, and multi-input `-o` rejection. Broader library/API coverage lives in the regular unit and integration tests, not only this CLI audit suite.

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
