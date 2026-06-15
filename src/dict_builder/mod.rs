//! Dictionary trainer — mirror of `zstd/lib/dictBuilder/`.
//!
//! One Rust file per upstream C source. Currently only the fastcover
//! trainer (`fastcover.c` / the `ZDICT_trainFromBuffer` entry point in
//! `zdict.c`) is ported, in [`fastcover`]; `cover.c`, `divsufsort.c`,
//! and the remaining `zdict.c` surface are not yet translated.

pub mod fastcover;

pub use fastcover::train_from_buffer;
