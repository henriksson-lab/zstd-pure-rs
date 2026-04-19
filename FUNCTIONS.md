# C → Rust function mapping

One C function ↔ one Rust function, so code-complexity-comparator stays useful. Table is seeded per module and grown as translation proceeds.

**Machine-readable mirror: [`ccc_mapping.toml`](ccc_mapping.toml).** Every pair in this file is also pinned there for `ccc-rs compare --mapping ccc_mapping.toml`, which prevents fingerprint fallback from introducing spurious matches.

Legend: `skel` = skeleton (returns clean error / no-op — no panic), `impl` = implemented, `test` = tested against deep-comparator.

## Module summary (as of current revision)

| Module | Status |
|---|---|
| `common/{bits, mem, error, zstd_common, xxhash, debug, bitstream, entropy_common, fse_decompress, zstd_internal}` | **Impl** (incl. `repStartValue`, `ZSTD_invalidateRepCodes`) |
| `common/{pool, threading}` | API stubs (MT feature future) |
| `decompress/{huf_decompress, zstd_decompress_block, zstd_ddict, zstd_decompress}` | **Impl**. Magic-prefix dict entropy tables load via `ZSTD_DCtx_loadDictionary` / `ZSTD_decompress_insertDictionary` / `ZSTD_loadDEntropy`; the one-shot `ZSTD_decompress_usingDict` stays raw-content only. |
| `compress/{hist, fse_compress, huf_compress, zstd_compress_literals, zstd_compress_sequences, zstd_fast, zstd_double_fast, zstd_lazy, zstd_preSplit, zstd_compress}` | **Impl** (plus btlazy2 DUBT queue insertion — `updateDUBT` / `insertDUBT1`) |
| `compress/zstd_ldm` | Types + gear-hash rolling + bucket hash table + backward-match helpers + sequence-generator (prefix-only) + outer driver + skip helpers **impl**; ext-dict branch of generateSequences + blockCompress skeletal |
| `compress/zstd_opt` | Price helpers + pricing + stats-update + rescaleFreqs (no-dict) + hash3 + binary-tree insert + match-gatherer (prefix-only) + optLdm wiring **impl**; `compressBlock_bt{opt,ultra,ultra2}` skeletal (forward-DP parser not yet wired) |
| `compress/match_state` | Struct + window helpers **impl**; overflow correction + resolvers ported |
| `compress/seq_store` | Struct + offbase helpers + `getSequenceLength` **impl** |
| `compress/zstd_compress_superblock` | Skeletal (`ZSTD_compressSuperBlock` returns `Generic`; sub-block entropy re-emission not yet ported) |
| `compress/zstdmt_compress` | API-stub (MT feature future) |

## lib/common/

### error_private.c / error_private.h → common::error

| C symbol | Rust symbol | Status |
| --- | --- | --- |
| `ERR_getErrorString` | `common::error::ERR_getErrorString` | impl |
| `ERR_getErrorName`  | `common::error::ERR_getErrorName` | impl |
| `ERR_isError`       | `common::error::ERR_isError` | impl |
| `ERR_getErrorCode`  | `common::error::ERR_getErrorCode` | impl |

### zstd_common.c → common::zstd_common

| C symbol | Rust symbol | Status |
| --- | --- | --- |
| `ZSTD_versionNumber` | `common::zstd_common::ZSTD_versionNumber` | impl |
| `ZSTD_versionString` | `common::zstd_common::ZSTD_versionString` | impl |
| `ZSTD_isError`       | `common::zstd_common::ZSTD_isError` | impl |
| `ZSTD_getErrorName`  | `common::zstd_common::ZSTD_getErrorName` | impl |
| `ZSTD_getErrorCode`  | `common::zstd_common::ZSTD_getErrorCode` | impl |

### bits.h → common::bits

| C symbol | Rust symbol | Status |
| --- | --- | --- |
| `ZSTD_countTrailingZeros32` | `common::bits::ZSTD_countTrailingZeros32` | impl |
| `ZSTD_countLeadingZeros32`  | `common::bits::ZSTD_countLeadingZeros32` | impl |
| `ZSTD_countTrailingZeros64` | `common::bits::ZSTD_countTrailingZeros64` | impl |
| `ZSTD_countLeadingZeros64`  | `common::bits::ZSTD_countLeadingZeros64` | impl |
| `ZSTD_highbit32`            | `common::bits::ZSTD_highbit32` | impl |
| `ZSTD_NbCommonBytes`        | `common::bits::ZSTD_NbCommonBytes` | impl |
| `BIT_highbit32`             | `common::bits::BIT_highbit32` | impl |

### bitstream.h → common::bitstream

| C symbol | Rust symbol | Status |
| --- | --- | --- |
| `BIT_initCStream`    | `common::bitstream::BIT_initCStream` | impl |
| `BIT_addBits`        | `common::bitstream::BIT_addBits` | impl |
| `BIT_addBitsFast`    | `common::bitstream::BIT_addBitsFast` | impl |
| `BIT_flushBits`      | `common::bitstream::BIT_flushBits` | impl |
| `BIT_flushBitsFast`  | `common::bitstream::BIT_flushBitsFast` | impl |
| `BIT_closeCStream`   | `common::bitstream::BIT_closeCStream` | impl |
| `BIT_initDStream`    | `common::bitstream::BIT_initDStream` | impl |
| `BIT_lookBits`       | `common::bitstream::BIT_lookBits` | impl |
| `BIT_lookBitsFast`   | `common::bitstream::BIT_lookBitsFast` | impl |
| `BIT_readBits`       | `common::bitstream::BIT_readBits` | impl |
| `BIT_readBitsFast`   | `common::bitstream::BIT_readBitsFast` | impl |
| `BIT_reloadDStream`  | `common::bitstream::BIT_reloadDStream` | impl |
| `BIT_endOfDStream`   | `common::bitstream::BIT_endOfDStream` | impl |

### mem.h → common::mem

| C symbol | Rust symbol | Status |
| --- | --- | --- |
| `MEM_isLittleEndian`   | `common::mem::is_little_endian` | impl |
| `MEM_read16`           | `common::mem::MEM_read16` | impl |
| `MEM_read24`           | `common::mem::MEM_read24` | impl |
| `MEM_read32`           | `common::mem::MEM_read32` | impl |
| `MEM_read64`           | `common::mem::MEM_read64` | impl |
| `MEM_readST`           | `common::mem::MEM_readST` | impl |
| `MEM_write16`          | `common::mem::MEM_write16` | impl |
| `MEM_write32`          | `common::mem::MEM_write32` | impl |
| `MEM_write64`          | `common::mem::MEM_write64` | impl |
| `MEM_readLE16/24/32/64`| `common::mem::MEM_readLE*` | impl |
| `MEM_writeLE16/24/32/64` | `common::mem::MEM_writeLE*` | impl |
| `MEM_readBE32/64` + `MEM_writeBE32/64` | `common::mem::MEM_{read,write}BE*` | impl |

### xxhash.h/.c → common::xxhash
Complete function list deferred; skeleton module only.

### entropy_common.c → common::entropy_common
| C symbol | Rust symbol | Status |
| --- | --- | --- |
| `FSE_versionNumber`        | `common::entropy_common::FSE_versionNumber` | impl |
| `FSE_isError`              | `common::entropy_common::FSE_isError` | impl |
| `FSE_getErrorName`         | `common::entropy_common::FSE_getErrorName` | impl |
| `HUF_isError`              | `common::entropy_common::HUF_isError` | impl |
| `HUF_getErrorName`         | `common::entropy_common::HUF_getErrorName` | impl |
| `FSE_readNCount`           | `common::entropy_common::FSE_readNCount` | impl |
| `FSE_readNCount_bmi2`      | `common::entropy_common::FSE_readNCount_bmi2` | impl |
| `HUF_readStats`            | `common::entropy_common::HUF_readStats` | impl |
| `HUF_readStats_wksp`       | `common::entropy_common::HUF_readStats_wksp` | impl |

### fse_decompress.c → common::fse_decompress
| C symbol | Rust symbol | Status |
| --- | --- | --- |
| `FSE_buildDTable_internal` | `common::fse_decompress::FSE_buildDTable_internal` | impl |
| `FSE_buildDTable_wksp`     | `common::fse_decompress::FSE_buildDTable_wksp` | impl |
| `FSE_buildDTable_raw`      | `common::fse_decompress::FSE_buildDTable_raw` | impl |
| `FSE_buildDTable_rle`      | `common::fse_decompress::FSE_buildDTable_rle` | impl |
| `FSE_decompress_usingDTable` | `common::fse_decompress::FSE_decompress_usingDTable` | impl |
| `FSE_decompress_wksp`      | `common::fse_decompress::FSE_decompress_wksp` | impl |
| `FSE_decompress_wksp_bmi2` | `common::fse_decompress::FSE_decompress_wksp_bmi2` | impl |

### pool.c → common::pool
Skeleton only; thread pool — implementation will use `std::thread` / `rayon`.

### threading.c → common::threading
Skeleton only; threading primitives.

### debug.c → common::debug
Debug/trace stubs (no-op in release).

## lib/decompress/

### huf_decompress.c → decompress::huf_decompress
**Impl.** X1 + X2 table paths complete; `HUF_readDTableX{1,2}`, `HUF_decompress{1X,4X}_usingDTable`, `HUF_decompress{1X1,4X1,1X2,4X2}_DCtx_wksp`. BMI2/ARM64/asm variants deliberately deferred — scalar path is ground truth.

### zstd_decompress_block.c → decompress::zstd_decompress_block
**Impl.** `ZSTD_decodeLiteralsBlock`, `ZSTD_decodeSeqHeaders{,_probe}`, `ZSTD_buildFSETable` + `ZSTD_buildSeqTable{,_rle}`, `ZSTD_decompressSequences_body`, `ZSTD_execSequence`, `ZSTD_decompressBlock_internal` (scalar body path). Upstream-only prefetch / split-lit-buffer / long-offsets-32 variants out of scope.

### zstd_ddict.c → decompress::zstd_ddict
**Impl (raw).** `ZSTD_DDict` + `ZSTD_createDDict{,_advanced,_byReference}`, `ZSTD_freeDDict`, `ZSTD_DDict_dictContent`, `ZSTD_DDict_dictSize`, `ZSTD_getDictID_fromDDict`, `ZSTD_sizeof_DDict`, `ZSTD_loadEntropy_intoDDict` (raw branch). Pre-digested HUF+FSE entropy (`ZSTD_dct_fullDict`) still TODO.

### zstd_decompress.c → decompress::zstd_decompress
**Impl.** `ZSTD_getFrameHeader`, `ZSTD_decompressFrame{,_withOpStart}`, `ZSTD_decompressDCtx`, `ZSTD_decompress`, `ZSTD_decompress_using{Dict,DDict}`, `ZSTD_getFrameContentSize`, `ZSTD_findFrame{Compressed,Decompressed}Size`, `ZSTD_findFrameSizeInfo`, `readSkippableFrameSize`, `ZSTD_is{Frame,SkippableFrame}`, `ZSTD_getDictID_fromFrame`, `ZSTD_decompressBlock`, `ZSTD_getBlockSize`. Streaming: `ZSTD_initDStream{,_usingDict}`, `ZSTD_decompressStream`, `ZSTD_resetDStream`. Parametric: `ZSTD_dParameter`, `ZSTD_DCtx_setParameter`/`getParameter`/`reset` + `ZSTD_DResetDirective`. Memory estimation: `ZSTD_estimateDCtxSize`, `ZSTD_estimateDStreamSize{,_fromFrame}`.

## lib/compress/

### hist.c → compress::hist
**Impl.** `HIST_count_simple`, `HIST_count_parallel_wksp`, `HIST_count{,Fast}{,_wksp}`, `HIST_add`, `HIST_isError`.

### fse_compress.c → compress::fse_compress
**Impl.** `FSE_normalizeCount{,M2}`, `FSE_writeNCount{,_generic}`, `FSE_buildCTable_wksp`, `FSE_buildCTable_rle`, `FSE_optimalTableLog{,_internal}`, `FSE_compressBound`, `FSE_compress_usingCTable{,_generic}`, `FSE_CState_t` + `FSE_initCState{,2}` / `FSE_encodeSymbol` / `FSE_flushCState`, `FSE_bitCost`.

### huf_compress.c → compress::huf_compress
**Impl.** `HUF_buildCTable{,_wksp}` (sort → tree → setMaxHeight → build), `HUF_writeCTable`, `HUF_compress1X/4X_usingCTable`, `HUF_compress{1X,4X}_repeat`, `HUF_compress_internal`. BMI2/ASM variants intentionally skipped.

### zstd_compress_literals.c → compress::zstd_compress_literals
**Impl.** `ZSTD_noCompressLiterals`, `ZSTD_compressRleLiteralsBlock`, `ZSTD_minGain`, `ZSTD_minLiteralsToCompress`, `ZSTD_compressLiterals` (full HUF path + fallbacks).

### zstd_compress_sequences.c → compress::zstd_compress_sequences
**Impl.** `kInverseProbabilityLog256` table, `ZSTD_useLowProbCount`, `ZSTD_entropyCost`, `ZSTD_crossEntropyCost`, `ZSTD_NCountCost`, `ZSTD_fseBitCost`, `ZSTD_selectEncodingType`, `ZSTD_buildCTable`, `ZSTD_encodeSequences{,_body}`. Includes `FSE_repeat` / `ZSTD_DefaultPolicy_e` / strategy constants.

### zstd_compress_superblock.c → compress::zstd_compress_superblock
Skeleton (stub returns `ErrorCode::Generic`); current port emits standard single-block frames.

### zstd_fast.c → compress::zstd_fast
**Impl.** `ZSTD_fillHashTable{,ForCCtx,ForCDict}`, `ZSTD_getLowestPrefixIndex`, `ZSTD_match4Found_branch`, `ZSTD_compressBlock_fast{,_noDict_generic,_with_history}`. Single-cursor simplification of upstream's 4-way pipeline with complementary-insert + immediate-rep-drain + step reset.

### zstd_double_fast.c → compress::zstd_double_fast
**Impl.** `ZSTD_fillDoubleHashTable{,ForCCtx}`, `ZSTD_compressBlock_doubleFast{,_noDict_generic,_with_history}`. Long + short table with refinement + post-match inserts.

### zstd_lazy.c → compress::zstd_lazy
**Impl.** `ZSTD_insertAndFindFirstIndex{,_internal}`, `ZSTD_HcFindBestMatch_noDict`, unified `ZSTD_compressBlock_lazy_noDict_generic` with depth=0/1/2 for greedy/lazy/lazy2; public wrappers `ZSTD_compressBlock_{greedy,lazy,lazy2,btlazy2}` (btlazy2 falls through to lazy2). Cross-block `_with_history` variant.

### zstd_opt.c → compress::zstd_opt
**Partial.** Price helpers implemented: constants (`ZSTD_LITFREQ_ADD`, `ZSTD_MAX_PRICE`, `ZSTD_PREDEF_THRESHOLD`, `BITCOST_ACCURACY`, `BITCOST_MULTIPLIER`, `ZSTD_OPT_NUM`, `ZSTD_OPT_SIZE`, `MaxLit`), `ZSTD_bitWeight`, `ZSTD_fracWeight`, `WEIGHT`, `sum_u32`, `ZSTD_downscaleStats` + `base_directive_e`, `ZSTD_scaleStats`, `ZSTD_readMINMATCH`. Types: `ZSTD_match_t`, `ZSTD_optimal_t`, `ZSTD_OptPrice_e`, `optState_t`. Setup: `ZSTD_compressedLiterals`, `ZSTD_setBasePrices`. Pricing: `ZSTD_rawLiteralsCost`, `ZSTD_litLengthPrice`, `ZSTD_getMatchPrice`. Stats: `ZSTD_updateStats`. 3-byte hash: `ZSTD_insertAndFindFirstIndexHash3`. optLdm integration: `ZSTD_optLdm_t`, `ZSTD_optLdm_skipRawSeqStoreBytes`, `ZSTD_optLdm_maybeAddMatch`, `ZSTD_opt_getNextMatchAndUpdateSeqStore`, `ZSTD_optLdm_processMatchCandidate`. `compressBlock_bt{opt,ultra,ultra2}` still return `ErrorCode::Generic`; btopt/btultra currently clamped down to btlazy2 in `ZSTD_compress`.

### zstd_ldm.c → compress::zstd_ldm
**Partial.** Types + parameter helpers + **rolling gear-hash core + bucket hash table + backward-match helpers + reduceTable/limitTableUpdate** implemented. Types: `ZSTD_ParamSwitch_e`, `ldmEntry_t`, `ldmMatchCandidate_t`, `ldmParams_t`, `ldmRollingHashState_t`, `ldmState_t`, `rawSeq`, `RawSeqStore_t`. Params: `ZSTD_ldm_adjustParameters`, `ZSTD_ldm_getTableSize`, `ZSTD_ldm_getMaxNbSeq`. Gear hash: full 256-entry `ZSTD_ldm_gearTab`, `ZSTD_ldm_gear_init`, `ZSTD_ldm_gear_reset`, `ZSTD_ldm_gear_feed`. Bucket hash table: `ZSTD_ldm_getBucket`, `ZSTD_ldm_insertEntry`, `ZSTD_ldm_fillHashTable`. Match utilities: `ZSTD_ldm_countBackwardsMatch`, `ZSTD_ldm_countBackwardsMatch_2segments`, `ZSTD_ldm_reduceTable`, `ZSTD_ldm_limitTableUpdate`, `ZSTD_ldm_fillFastTables`. Sequence generator (prefix-only): `ZSTD_ldm_generateSequences_internal_prefixOnly`. Outer driver: `ZSTD_ldm_generateSequences` (chunked, no overflow correction yet). Skip helpers: `ZSTD_ldm_skipSequences`, `ZSTD_ldm_skipRawSeqStoreBytes`, `maybeSplitSequence`. `generateSequences{,_internal}` / `blockCompress` / `skipSequences` still skeletal (stubs return `ErrorCode::Generic` or no-op).

### zstd_preSplit.c → compress::zstd_presplit
**Impl.** `ZSTD_splitBlock` (both `fromBorders` and `byChunks` paths), `hash2`, `Fingerprint` / `FPStats`, `fpDistance`, `compareFingerprints`, `mergeEvents`.

### zstd_compress.c → compress::zstd_compress
**Impl.** All public entry points: `ZSTD_compressBound`, `ZSTD_compress{,CCtx,_usingDict,_usingCDict}`, `ZSTD_create{CCtx,CDict{,_byReference}}` + `ZSTD_free*`, `ZSTD_sizeof_CDict`, `ZSTD_LLcode` / `ZSTD_MLcode` / `ZSTD_seqToCodes`, `ZSTD_fseCTables_t` / `ZSTD_hufCTables_t` / `ZSTD_entropyCTables_t`, `ZSTD_buildSequencesStatistics`, `ZSTD_entropyCompressSeqStore{,_internal}`, `ZSTD_isRLE`, `ZSTD_noCompressBlock` / `ZSTD_rleCompressBlock`, `ZSTD_writeFrameHeader` / `ZSTD_writeLastEmptyBlock`, `ZSTD_compressFrame_fast{,_with_prefix}`, `ZSTD_getCParams`, the full upstream `ZSTD_DEFAULT_CPARAMS[4][23]` table. Streaming: `ZSTD_initCStream{,_srcSize,_usingDict}`, `ZSTD_compressStream`, `ZSTD_flushStream`, `ZSTD_endStream`, `ZSTD_resetCStream`, `ZSTD_CCtx_setPledgedSrcSize`. Parametric: `ZSTD_cParameter`, `ZSTD_CCtx_setParameter`/`getParameter`/`reset` + `ZSTD_ResetDirective`. Memory estimation: `ZSTD_estimateCCtxSize{,_usingCParams}` / `ZSTD_estimateCStreamSize{,_usingCParams}`.

### zstdmt_compress.c → compress::zstdmt_compress (feature `mt`)
Skeleton — API surface returns `None` / `ErrorCode::Generic` stubs so `mt` builds cleanly without spawning threads. A rayon-based implementation is a future effort; the single-threaded path already provides full correctness.
