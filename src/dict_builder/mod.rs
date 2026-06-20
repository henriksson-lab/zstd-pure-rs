//! Dictionary trainer — mirror of `zstd/lib/dictBuilder/`.
//!
//! One Rust file per upstream C source: `divsufsort.rs` (suffix sorter),
//! `cover.rs` (COVER), `fastcover.rs` (fastCover), `zdict.rs` (the `ZDICT_*`
//! API, entropy analysis, finalization, and the legacy trainer).

mod cover;
mod divsufsort;
pub mod fastcover;
pub mod zdict;

pub use cover::{ZDICT_optimizeTrainFromBuffer_cover, ZDICT_trainFromBuffer_cover};
pub use fastcover::{ZDICT_optimizeTrainFromBuffer_fastCover, ZDICT_trainFromBuffer_fastCover};
pub use zdict::{
    ZDICT_addEntropyTablesFromBuffer, ZDICT_cover_params_t, ZDICT_fastCover_params_t,
    ZDICT_finalizeDictionary, ZDICT_getDictHeaderSize, ZDICT_getDictID, ZDICT_getErrorName,
    ZDICT_isError, ZDICT_legacy_params_t, ZDICT_params_t, ZDICT_trainFromBuffer,
    ZDICT_trainFromBuffer_legacy,
};
