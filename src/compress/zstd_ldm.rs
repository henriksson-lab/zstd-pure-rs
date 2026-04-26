//! Translation of `lib/compress/zstd_ldm.c`. Long-distance matcher.
//!
//! **Implemented**: static constants, parameter types, parameter
//! adjustment (`ZSTD_ldm_adjustParameters`), sizing
//! (`ZSTD_ldm_getTableSize`, `ZSTD_ldm_getMaxNbSeq`), rolling gear
//! hash (`ZSTD_ldm_gear_init/reset/feed`), bucket hash table
//! (`ZSTD_ldm_getBucket`, `ZSTD_ldm_insertEntry`,
//! `ZSTD_ldm_fillHashTable`), match utilities
//! (`countBackwardsMatch`, `countBackwardsMatch_2segments`,
//! `reduceTable`, `limitTableUpdate`, `fillFastTables`),
//! sequence generator
//! (`ZSTD_ldm_generateSequences_internal`) + chunked
//! outer driver (`ZSTD_ldm_generateSequences`), skip / split
//! helpers (`skipSequences`, `skipRawSeqStoreBytes`,
//! `maybeSplitSequence`).
//!
//! `ZSTD_ldm_blockCompress` is wired into both parser families:
//! fast/lazy strategies materialize LDM sequences directly, while
//! btopt/btultra receive them through the match-state `ldmSeqStore`
//! bridge as upstream candidate matches.

#![allow(non_snake_case)]

use crate::common::xxhash::XXH64;
use crate::compress::match_state::{
    ZSTD_MatchState_t, ZSTD_compressionParameters, ZSTD_window_correctOverflow,
    ZSTD_window_enforceMaxDist, ZSTD_window_hasExtDict, ZSTD_window_needOverflowCorrection,
    ZSTD_window_t,
};
use crate::compress::zstd_compress_sequences::{
    ZSTD_btlazy2, ZSTD_btopt, ZSTD_btultra, ZSTD_btultra2, ZSTD_dfast, ZSTD_fast, ZSTD_greedy,
    ZSTD_lazy, ZSTD_lazy2,
};
use crate::compress::zstd_double_fast::ZSTD_fillDoubleHashTable;
use crate::compress::zstd_fast::{
    ZSTD_dictTableLoadMethod_e, ZSTD_fillHashTable, ZSTD_tableFillPurpose_e, HASH_READ_SIZE,
};
use crate::compress::zstd_hashes::{ZSTD_count, ZSTD_count_2segments};

/// Upstream `LDM_BUCKET_SIZE_LOG`.
pub const LDM_BUCKET_SIZE_LOG: u32 = 4;
/// Upstream `LDM_MIN_MATCH_LENGTH`.
pub const LDM_MIN_MATCH_LENGTH: u32 = 64;
/// Upstream `ZSTD_LDM_BUCKETSIZELOG_MAX`.
pub const ZSTD_LDM_BUCKETSIZELOG_MAX: u32 = 8;
/// Upstream `ZSTD_HASHLOG_MIN`.
pub const ZSTD_HASHLOG_MIN: u32 = 6;
/// Upstream `ZSTD_HASHLOG_MAX` — capped at 30 for sanity.
pub const ZSTD_HASHLOG_MAX: u32 = 30;

/// Upstream `ZSTD_LDM_DEFAULT_WINDOW_LOG` (`zstd_ldm.h:21`) — aliased
/// onto `ZSTD_WINDOWLOG_LIMIT_DEFAULT` (27). Applied when LDM is
/// enabled and the caller hasn't set an explicit windowLog — a wider
/// window makes long-distance matches worthwhile.
pub const ZSTD_LDM_DEFAULT_WINDOW_LOG: u32 = 27;

/// Port of `ZSTD_ParamSwitch_e`. Used by LDM / block splitter to
/// express "auto / force-on / force-off".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZSTD_ParamSwitch_e {
    #[default]
    ZSTD_ps_auto = 0,
    ZSTD_ps_enable = 1,
    ZSTD_ps_disable = 2,
}

/// Port of `ldmEntry_t`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ldmEntry_t {
    pub offset: u32,
    pub checksum: u32,
}

/// Port of `ldmMatchCandidate_t`. Upstream stores `ldmEntry_t* bucket`;
/// the Rust port carries the bucket's start index into
/// `ldmState.hashTable` instead of a raw pointer.
#[derive(Debug, Clone, Copy, Default)]
pub struct ldmMatchCandidate_t {
    /// Offset of the split point, measured from the start of the
    /// input slice the candidate was discovered in.
    pub split: usize,
    pub hash: u32,
    pub checksum: u32,
    /// Index (into `ldmState.hashTable`) of the first entry in this
    /// candidate's bucket.
    pub bucket_idx: u32,
}

/// Port of `ldmState_t`. Upstream packs a `ZSTD_window_t`, an
/// `ldmEntry_t*` flat hash table, a per-bucket cursor array, a
/// split-index scratch buffer, and a candidate scratch buffer.
#[derive(Debug, Clone)]
pub struct ldmState_t {
    pub window: ZSTD_window_t,
    pub hashTable: Vec<ldmEntry_t>,
    pub loadedDictEnd: u32,
    pub bucketOffsets: Vec<u8>,
    pub splitIndices: [usize; LDM_BATCH_SIZE],
    pub matchCandidates: [ldmMatchCandidate_t; LDM_BATCH_SIZE],
}

impl ldmState_t {
    /// Allocate the LDM state tables according to `params`. Caller must
    /// have already resolved `params` through `ZSTD_ldm_adjustParameters`.
    pub fn new(params: &ldmParams_t) -> Self {
        let hSize = 1usize << params.hashLog;
        let bucketSize = 1usize << (params.hashLog - params.bucketSizeLog);
        Self {
            window: ZSTD_window_t::default(),
            hashTable: vec![ldmEntry_t::default(); hSize],
            loadedDictEnd: 0,
            bucketOffsets: vec![0u8; bucketSize],
            splitIndices: [0usize; LDM_BATCH_SIZE],
            matchCandidates: [ldmMatchCandidate_t::default(); LDM_BATCH_SIZE],
        }
    }
}

/// Port of `ZSTD_ldm_getBucket`. Returns the starting index in
/// `ldmState.hashTable` of the bucket associated with `hash`.
#[inline]
pub fn ZSTD_ldm_getBucket(_ldmState: &ldmState_t, hash: u32, bucketSizeLog: u32) -> usize {
    (hash as usize) << bucketSizeLog
}

/// Port of `ZSTD_ldm_insertEntry`. Writes `entry` into the bucket at
/// `hash` using the rolling per-bucket cursor in
/// `ldmState.bucketOffsets`. Advances the cursor modulo
/// `1 << bucketSizeLog`.
pub fn ZSTD_ldm_insertEntry(
    ldmState: &mut ldmState_t,
    hash: u32,
    entry: ldmEntry_t,
    bucketSizeLog: u32,
) {
    let bucket_base = ZSTD_ldm_getBucket(ldmState, hash, bucketSizeLog);
    let offset = ldmState.bucketOffsets[hash as usize] as usize;
    ldmState.hashTable[bucket_base + offset] = entry;
    let mask = (1u32 << bucketSizeLog) - 1;
    ldmState.bucketOffsets[hash as usize] = ((offset as u32).wrapping_add(1) & mask) as u8;
}

/// Port of `ZSTD_ldm_fillHashTable`. Walks `data` with the gear-hash,
/// and for each split position at least `minMatchLength` bytes into the
/// range, records an `ldmEntry_t` in the bucket hash table.
///
/// `abs_start` is the absolute base-relative index that `data[0]` maps
/// to — i.e. `entry.offset = abs_start + (split position in data)`. In
/// upstream this is computed from pointer subtraction against
/// `ldmState.window.base`; the Rust port passes it explicitly so LDM
/// stays decoupled from the rest of the window-management machinery.
pub fn ZSTD_ldm_fillHashTable(
    ldmState: &mut ldmState_t,
    data: &[u8],
    abs_start: u32,
    params: &ldmParams_t,
) {
    let minMatchLength = params.minMatchLength;
    let bucketSizeLog = params.bucketSizeLog;
    let hBits = params.hashLog - bucketSizeLog;
    let mask: u32 = if hBits >= 32 {
        u32::MAX
    } else {
        (1u32 << hBits) - 1
    };

    let mut hashState = ldmRollingHashState_t {
        rolling: 0,
        stopMask: 0,
    };
    ZSTD_ldm_gear_init(&mut hashState, params);

    let mut ip: usize = 0;
    let iend = data.len();
    let mut splits = [0usize; LDM_BATCH_SIZE];

    while ip < iend {
        let mut numSplits: u32 = 0;
        let hashed =
            ZSTD_ldm_gear_feed(&mut hashState, &data[ip..iend], &mut splits, &mut numSplits);

        for &split in splits.iter().take(numSplits as usize) {
            let split_abs_in_data = ip + split;
            if split_abs_in_data >= minMatchLength as usize {
                let split_pos = split_abs_in_data - minMatchLength as usize;
                let xxhash = XXH64(&data[split_pos..split_pos + minMatchLength as usize], 0);
                let hash = (xxhash as u32) & mask;
                let entry = ldmEntry_t {
                    offset: abs_start.wrapping_add(split_pos as u32),
                    checksum: (xxhash >> 32) as u32,
                };
                ZSTD_ldm_insertEntry(ldmState, hash, entry, bucketSizeLog);
            }
        }
        ip += hashed;
    }
}

/// Port of `ldmParams_t`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ldmParams_t {
    pub enableLdm: ZSTD_ParamSwitch_e,
    pub hashLog: u32,
    pub bucketSizeLog: u32,
    pub minMatchLength: u32,
    pub hashRateLog: u32,
    pub windowLog: u32,
}

/// Port of `ZSTD_ldm_adjustParameters`. Resolves "auto" defaults on
/// an `ldmParams_t` against the active cParams:
///   - `hashRateLog`: derived from either `hashLog` or the strategy
///     (mapping strategy 1..=9 → rate 7..=4 via `7 - strategy/3`).
///   - `hashLog`: clamped to `[ZSTD_HASHLOG_MIN, ZSTD_HASHLOG_MAX]`,
///     derived from `windowLog − hashRateLog` when `windowLog` is big
///     enough; otherwise pinned at the minimum.
///   - `minMatchLength`: defaults to `LDM_MIN_MATCH_LENGTH` (halved
///     for btultra+ strategies).
///   - `bucketSizeLog`: capped at `hashLog` and clamped to
///     `[LDM_BUCKET_SIZE_LOG, ZSTD_LDM_BUCKETSIZELOG_MAX]`.
pub fn ZSTD_ldm_adjustParameters(params: &mut ldmParams_t, cParams: &ZSTD_compressionParameters) {
    params.windowLog = cParams.windowLog;
    if params.hashRateLog == 0 {
        if params.hashLog > 0 {
            debug_assert!(params.hashLog <= ZSTD_HASHLOG_MAX);
            if params.windowLog > params.hashLog {
                params.hashRateLog = params.windowLog - params.hashLog;
            }
        } else {
            debug_assert!((1..=9).contains(&cParams.strategy));
            params.hashRateLog = 7 - (cParams.strategy / 3);
        }
    }
    if params.hashLog == 0 {
        if params.windowLog <= params.hashRateLog {
            params.hashLog = ZSTD_HASHLOG_MIN;
        } else {
            params.hashLog =
                (params.windowLog - params.hashRateLog).clamp(ZSTD_HASHLOG_MIN, ZSTD_HASHLOG_MAX);
        }
    }
    if params.minMatchLength == 0 {
        params.minMatchLength = LDM_MIN_MATCH_LENGTH;
        // ZSTD_btultra = 8, ZSTD_btultra2 = 9.
        if cParams.strategy >= 8 {
            params.minMatchLength /= 2;
        }
    }
    if params.bucketSizeLog == 0 {
        debug_assert!((1..=9).contains(&cParams.strategy));
        params.bucketSizeLog = cParams
            .strategy
            .clamp(LDM_BUCKET_SIZE_LOG, ZSTD_LDM_BUCKETSIZELOG_MAX);
    }
    params.bucketSizeLog = params.bucketSizeLog.min(params.hashLog);
}

/// Port of `ZSTD_ldm_getTableSize`. Returns the total number of bytes
/// required for the LDM state's hash + bucket tables (0 when LDM is
/// disabled).
///
/// Rust signature note: upstream uses `ZSTD_cwksp_alloc_size` for
/// cache-line rounding. The Rust port drops that alignment fudge — it
/// returns the raw byte count; callers that need alignment pad their
/// own allocation.
pub fn ZSTD_ldm_getTableSize(params: ldmParams_t) -> usize {
    if params.enableLdm != ZSTD_ParamSwitch_e::ZSTD_ps_enable {
        return 0;
    }
    let ldmHSize = 1usize << params.hashLog;
    let ldmBucketSizeLog = params.bucketSizeLog.min(params.hashLog);
    let ldmBucketSize = 1usize << (params.hashLog - ldmBucketSizeLog);
    ldmBucketSize + ldmHSize * core::mem::size_of::<ldmEntry_t>()
}

/// Port of `ZSTD_ldm_getMaxNbSeq`. Returns the upper bound on the
/// number of LDM sequences a chunk of `maxChunkSize` bytes can emit.
pub fn ZSTD_ldm_getMaxNbSeq(params: ldmParams_t, maxChunkSize: usize) -> usize {
    if params.enableLdm != ZSTD_ParamSwitch_e::ZSTD_ps_enable {
        return 0;
    }
    maxChunkSize / params.minMatchLength as usize
}

/// Port of the 256-entry gear hash table from
/// `zstd_ldm_geartab.h`. Identical values — do not modify.
#[rustfmt::skip]
pub const ZSTD_ldm_gearTab: [u64; 256] = [
    0xf5b8f72c5f77775c, 0x84935f266b7ac412, 0xb647ada9ca730ccc,
    0xb065bb4b114fb1de, 0x34584e7e8c3a9fd0, 0x4e97e17c6ae26b05,
    0x3a03d743bc99a604, 0xcecd042422c4044f, 0x76de76c58524259e,
    0x9c8528f65badeaca, 0x86563706e2097529, 0x2902475fa375d889,
    0xafb32a9739a5ebe6, 0xce2714da3883e639, 0x021eaf821722e69e,
    0x037b628620b628,    0x049a8d455d88caf5, 0x8556d711e6958140,
    0x04f7ae74fc605c1f,  0x829f0c3468bd3a20, 0x4ffdc885c625179e,
    0x8473de048a3daf1b, 0x51008822b05646b2, 0x69d75d12b2d1cc5f,
    0x8c9d4a19159154bc, 0xc3cc10f4abbd4003, 0xd06ddc1cecb97391,
    0xbe48e6e7ed80302e, 0x3481db31cee03547, 0xacc3f67cdaa1d210,
    0x65cb771d8c7f96cc, 0x8eb27177055723dd, 0xc789950d44cd94be,
    0x934feadc3700b12b, 0x5e485f11edbdf182, 0x1e2e2a46fd64767a,
    0x2969ca71d82efa7c, 0x9d46e9935ebbba2e, 0xe056b67e05e6822b,
    0x94d73f55739d03a0, 0xcd7010bdb69b5a03, 0x455ef9fcd79b82f4,
    0x869cb54a8749c161, 0x38d1a4fa6185d225, 0xb475166f94bbe9bb,
    0xa4143548720959f1, 0x7aed4780ba6b26ba, 0xd0ce264439e02312,
    0x84366d746078d508, 0xa8ce973c72ed17be, 0x21c323a29a430b01,
    0x9962d617e3af80ee, 0xab0ce91d9c8cf75b, 0x530e8ee6d19a4dbc,
    0x2ef68c0cf53f5d72, 0xc03a681640a85506, 0x496e4e9f9c310967,
    0x78580472b59b14a0, 0x273824c23b388577, 0x66bf923ad45cb553,
    0x47ae1a5a2492ba86, 0x35e304569e229659, 0x4765182a46870b6f,
    0x6cbab625e9099412, 0xddac9a2e598522c1, 0x7172086e666624f2,
    0xdf5003ca503b7837, 0x88c0c1db78563d09, 0x58d51865acfc289d,
    0x177671aec65224f1, 0xfb79d8a241e967d7, 0x2be1e101cad9a49a,
    0x6625682f6e29186b, 0x399553457ac06e50, 0x035dffb4c23abb74,
    0x429db2591f54aade, 0xc52802a8037d1009, 0x6acb27381f0b25f3,
    0xf45e2551ee4f823b, 0x8b0ea2d99580c2f7, 0x3bed519cbcb4e1e1,
    0x0ff452823dbb010a,  0x9d42ed614f3dd267, 0x5b9313c06257c57b,
    0xa114b8008b5e1442, 0xc1fe311c11c13d4b, 0x66e8763ea34c5568,
    0x8b982af1c262f05d, 0xee8876faaa75fbb7, 0x8a62a4d0d172bb2a,
    0xc13d94a3b7449a97, 0x6dbbba9dc15d037c, 0xc786101f1d92e0f1,
    0xd78681a907a0b79b, 0xf61aaf2962c9abb9, 0x2cfd16fcd3cb7ad9,
    0x868c5b6744624d21, 0x25e650899c74ddd7, 0xba042af4a7c37463,
    0x4eb1a539465a3eca, 0xbe09dbf03b05d5ca, 0x774e5a362b5472ba,
    0x47a1221229d183cd, 0x504b0ca18ef5a2df, 0xdffbdfbde2456eb9,
    0x46cd2b2fbee34634, 0xf2aef8fe819d98c3, 0x357f5276d4599d61,
    0x24a5483879c453e3, 0x088026889192b4b9,  0x28da96671782dbec,
    0x4ef37c40588e9aaa, 0x8837b90651bc9fb3, 0xc164f741d3f0e5d6,
    0xbc135a0a704b70ba, 0x069cd868f7622ada,  0xbc37ba89e0b9c0ab,
    0x47c14a01323552f6, 0x4f00794bacee98bb, 0x7107de7d637a69d5,
    0x88af793bb6f2255e, 0xf3c6466b8799b598, 0xc288c616aa7f3b59,
    0x81ca63cf42fca3fd, 0x88d85ace36a2674b, 0x0d056bd3792389e7,
    0xe55c396c4e9dd32d, 0xbefb504571e6c0a6, 0x96ab32115e91e8cc,
    0xbf8acb18de8f38d1, 0x66dae58801672606, 0x833b6017872317fb,
    0xb87c16f2d1c92864, 0xdb766a74e58b669c, 0x89659f85c61417be,
    0xc8daad856011ea0c, 0x76a4b565b6fe7eae, 0xa469d085f6237312,
    0xaaf0365683a3e96c, 0x4dbb746f8424f7b8, 0x0638755af4e4acc1,
    0x3d7807f5bde64486, 0x17be6d8f5bbb7639, 0x0903f0cd44dc35dc,
    0x67b672eafdf1196c, 0xa676ff93ed4c82f1, 0x521d1004c5053d9d,
    0x37ba9ad09ccc9202, 0x84e54d297aacfb51, 0x0a0b4b776a143445,
    0x0820d471e20b348e,  0x1874383cb83d46dc, 0x97edeec7a1efe11c,
    0xb330e50b1bdc42aa, 0x1dd91955ce70e032, 0xa514cdb88f2939d5,
    0x2791233fd90db9d3, 0x7b670a4cc50f7a9b, 0x77c07d2a05c6dfa5,
    0xe3778b6646d0a6fa, 0xb39c8eda47b56749, 0x933ed448addbef28,
    0xaf846af6ab7d0bf4, 0x0e5af208eb666e49,  0x5e6622f73534cd6a,
    0x297daeca42ef5b6e, 0x862daef3d35539a6, 0xe68722498f8e1ea9,
    0x981c53093dc0d572, 0xfa09b0bfbf86fbf5, 0x30b1e96166219f15,
    0x70e7d466bdc4fb83, 0x5a66736e35f2a8e9, 0xcddb59d2b7c1baef,
    0xd6c7d247d26d8996, 0xea4e39eac8de1ba3, 0x539c8bb19fa3aff2,
    0x09f90e4c5fd508d8,  0xa34e5956fbaf3385, 0x2e2f8e151d3ef375,
    0x173691e9b83faec1, 0xb85a8d56bf016379, 0x8382381267408ae3,
    0xb90f901bbdc0096d, 0x7c6ad32933bcec65, 0x76bb5e2f2c8ad595,
    0x390f851a6cf46d28, 0xc3e6064da1c2da72, 0xc52a0c101cfa5389,
    0xd78eaf84a3fbc530, 0x3781b9e2288b997e, 0x73c2f6dea83d05c4,
    0x04228e364c5b5ed7,  0x9d7a3edf0da43911, 0x8edcfeda24686756,
    0x5e7667a7b7a9b3a1, 0x4c4f389fa143791d, 0xb08bc1023da7cddc,
    0x7ab4be3ae529b1cc, 0x754e6132dbe74ff9, 0x71635442a839df45,
    0x2f6fb1643fbe52de, 0x961e0a42cf7a8177, 0xf3b45d83d89ef2ea,
    0xee3de4cf4a6e3e9b, 0xcd6848542c3295e7, 0xe4cee1664c78662f,
    0x9947548b474c68c4, 0x25d73777a5ed8b0b, 0x00c915b1d636b7fc,
    0x21c2ba75d9b0d2da, 0x5f6b5dcf608a64a1, 0xdcf333255ff9570c,
    0x633b922418ced4ee, 0xc136dde0b004b34a, 0x58cc83b05d4b2f5a,
    0x5eb424dda28e42d2, 0x62df47369739cd98, 0xb4e0b42485e4ce17,
    0x16e1f0c1f9a8d1e7, 0x8ec3916707560ebf, 0x62ba6e2df2cc9db3,
    0xcbf9f4ff77d83a16, 0x78d9d7d07d2bbcc4, 0xef554ce1e02c41f4,
    0x8d7581127eccf94d, 0xa9b53336cb3c8a05, 0x38c42c0bf45c4f91,
    0x640893cdf4488863, 0x80ec34bc575ea568, 0x39f324f5b48eaa40,
    0xe9d9ed1f8eff527f, 0x9224fc058cc5a214, 0xbaba00b04cfe7741,
    0x309a9f120fcf52af, 0xa558f3ec65626212, 0x424bec8b7adabe2f,
    0x41622513a6aea433, 0xb88da2d5324ca798, 0xd287733b245528a4,
    0x9a44697e6d68aec3, 0x7b1093be2f49bb28, 0x50bbec632e3d8aad,
    0x6cd90723e1ea8283, 0x897b9e7431b02bf3, 0x219efdcb338a7047,
    0x3b0311f0a27c0656, 0xdb17bf91c0db96e7, 0x8cd4fd6b4e85a5b2,
    0xfab071054ba6409d, 0x40d6fe831fa9dfd9, 0xaf358debad7d791e,
    0xeb8d0e25a65e3e58, 0xbbcbd3df14e08580, 0x0cf751f27ecdab2b,
    0x2b4da14f2613d8f4,
];

/// Upstream `LDM_BATCH_SIZE` — max splits returned in a single
/// `gear_feed` call.
pub const LDM_BATCH_SIZE: usize = 64;

/// Port of `ldmRollingHashState_t`.
#[derive(Debug, Clone, Copy)]
pub struct ldmRollingHashState_t {
    pub rolling: u64,
    pub stopMask: u64,
}

/// Port of `ZSTD_ldm_gear_init`. Initializes rolling-hash state from
/// an `ldmParams_t`.
pub fn ZSTD_ldm_gear_init(state: &mut ldmRollingHashState_t, params: &ldmParams_t) {
    let maxBitsInMask = params.minMatchLength.min(64);
    let hashRateLog = params.hashRateLog;
    state.rolling = u32::MAX as u64;
    if hashRateLog > 0 && hashRateLog <= maxBitsInMask {
        state.stopMask = (((1u64) << hashRateLog) - 1) << (maxBitsInMask - hashRateLog);
    } else {
        state.stopMask = (1u64 << hashRateLog) - 1;
    }
}

/// Port of `ZSTD_ldm_gear_reset`. Feeds `minMatchLength` bytes into
/// the rolling state WITHOUT recording any splits — used to "warm up"
/// the hash window before entering the main feed loop.
pub fn ZSTD_ldm_gear_reset(state: &mut ldmRollingHashState_t, data: &[u8], minMatchLength: usize) {
    let mut hash = state.rolling;
    let mut n = 0usize;
    while n + 3 < minMatchLength {
        hash = (hash << 1).wrapping_add(ZSTD_ldm_gearTab[data[n] as usize]);
        n += 1;
        hash = (hash << 1).wrapping_add(ZSTD_ldm_gearTab[data[n] as usize]);
        n += 1;
        hash = (hash << 1).wrapping_add(ZSTD_ldm_gearTab[data[n] as usize]);
        n += 1;
        hash = (hash << 1).wrapping_add(ZSTD_ldm_gearTab[data[n] as usize]);
        n += 1;
    }
    while n < minMatchLength {
        hash = (hash << 1).wrapping_add(ZSTD_ldm_gearTab[data[n] as usize]);
        n += 1;
    }
    state.rolling = hash;
}

/// Port of `ZSTD_ldm_gear_feed`. Rolls through `data[..size]`
/// recording split positions (where `hash & stopMask == 0`) into
/// `splits`. Returns the number of input bytes processed; stops early
/// when `splits` fills to `LDM_BATCH_SIZE`.
pub fn ZSTD_ldm_gear_feed(
    state: &mut ldmRollingHashState_t,
    data: &[u8],
    splits: &mut [usize; LDM_BATCH_SIZE],
    numSplits: &mut u32,
) -> usize {
    let mut hash = state.rolling;
    let mask = state.stopMask;
    let size = data.len();
    let mut n = 0usize;

    macro_rules! iter_once {
        () => {{
            hash = (hash << 1).wrapping_add(ZSTD_ldm_gearTab[data[n] as usize]);
            n += 1;
            if (hash & mask) == 0 {
                splits[*numSplits as usize] = n;
                *numSplits += 1;
                if *numSplits as usize == LDM_BATCH_SIZE {
                    state.rolling = hash;
                    return n;
                }
            }
        }};
    }

    while n + 3 < size {
        iter_once!();
        iter_once!();
        iter_once!();
        iter_once!();
    }
    while n < size {
        iter_once!();
    }
    state.rolling = hash;
    n
}

/// Port of `rawSeq`. A raw (litLength, matchLength, offset) triple as
/// emitted by LDM / external sequence producers.
#[derive(Debug, Clone, Copy, Default)]
pub struct rawSeq {
    pub offset: u32,
    pub litLength: u32,
    pub matchLength: u32,
}

/// Port of `RawSeqStore_t`. Upstream carries a raw `rawSeq*` + sizes;
/// the Rust port owns a `Vec<rawSeq>` and tracks the read cursor.
#[derive(Debug, Clone, Default)]
pub struct RawSeqStore_t {
    pub seq: Vec<rawSeq>,
    pub pos: usize,
    pub posInSequence: usize,
    pub size: usize,
    pub capacity: usize,
}

impl RawSeqStore_t {
    /// Allocate a fresh store with room for `capacity` sequences. All
    /// slots start zeroed; `size` / `pos` start at 0.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            seq: vec![rawSeq::default(); capacity],
            pos: 0,
            posInSequence: 0,
            size: 0,
            capacity,
        }
    }
}

/// Port of `ZSTD_ldm_countBackwardsMatch`. Counts bytes that match
/// walking backwards from `pIn_pos` (in `pIn`) and `pMatch_pos` (in
/// `pMatch`). Stops at either `pAnchor_pos` or `pMatchBase_pos`, or on
/// a mismatch.
pub fn ZSTD_ldm_countBackwardsMatch(
    pIn: &[u8],
    pIn_pos: usize,
    pAnchor_pos: usize,
    pMatch: &[u8],
    pMatch_pos: usize,
    pMatchBase_pos: usize,
) -> usize {
    let mut ip = pIn_pos;
    let mut mp = pMatch_pos;
    let mut matchLength = 0usize;
    while ip > pAnchor_pos && mp > pMatchBase_pos && pIn[ip - 1] == pMatch[mp - 1] {
        ip -= 1;
        mp -= 1;
        matchLength += 1;
    }
    matchLength
}

/// Port of `ZSTD_ldm_countBackwardsMatch_2segments`. Extends the plain
/// backward-match with a fall-through into an ext-dict segment when
/// the current segment's lower bound is reached and the current
/// segment is not itself the ext-dict.
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_ldm_countBackwardsMatch_2segments(
    pIn: &[u8],
    pIn_pos: usize,
    pAnchor_pos: usize,
    pMatch: &[u8],
    pMatch_pos: usize,
    pMatchBase_pos: usize,
    pExtDict: &[u8],
    pExtDictStart_pos: usize,
    pExtDictEnd_pos: usize,
) -> usize {
    let matchLength = ZSTD_ldm_countBackwardsMatch(
        pIn,
        pIn_pos,
        pAnchor_pos,
        pMatch,
        pMatch_pos,
        pMatchBase_pos,
    );
    // Match either entirely inside segment, or segment *is* ext-dict.
    if pMatch_pos.saturating_sub(matchLength) != pMatchBase_pos
        || pMatchBase_pos == pExtDictStart_pos
    {
        return matchLength;
    }
    matchLength
        + ZSTD_ldm_countBackwardsMatch(
            pIn,
            pIn_pos - matchLength,
            pAnchor_pos,
            pExtDict,
            pExtDictEnd_pos,
            pExtDictStart_pos,
        )
}

/// Port of `ZSTD_ldm_reduceTable`. Subtracts `reducerValue` from every
/// entry's `offset`, clamping at zero. Called when the window's base
/// index overflows.
pub fn ZSTD_ldm_reduceTable(table: &mut [ldmEntry_t], reducerValue: u32) {
    for e in table.iter_mut() {
        if e.offset < reducerValue {
            e.offset = 0;
        } else {
            e.offset -= reducerValue;
        }
    }
}

/// Port of `ZSTD_ldm_fillFastTables`. When LDM produces a long match,
/// the fast / dfast matchfinders still need their internal hash tables
/// primed over the matched region so follow-on blocks see history.
/// For greedy / lazy / lazy2 / bt* strategies the block compressors
/// seed their own tables, so this is a no-op.
pub fn ZSTD_ldm_fillFastTables(ms: &mut ZSTD_MatchState_t, data_end: &[u8]) -> usize {
    match ms.cParams.strategy {
        s if s == ZSTD_fast => {
            ZSTD_fillHashTable(
                ms,
                data_end,
                ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_fast,
                ZSTD_tableFillPurpose_e::ZSTD_tfp_forCCtx,
            );
        }
        s if s == ZSTD_dfast => {
            ZSTD_fillDoubleHashTable(
                ms,
                data_end,
                ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_fast,
                ZSTD_tableFillPurpose_e::ZSTD_tfp_forCCtx,
            );
        }
        s if s == ZSTD_greedy
            || s == ZSTD_lazy
            || s == ZSTD_lazy2
            || s == ZSTD_btlazy2
            || s == ZSTD_btopt
            || s == ZSTD_btultra
            || s == ZSTD_btultra2 =>
        {
            // Block compressors seed their own tables.
        }
        _ => debug_assert!(false, "invalid strategy id"),
    }
    0
}

/// Port of `ZSTD_ldm_limitTableUpdate`. Rewinds `ms.nextToUpdate` if
/// it has fallen more than 1024 positions behind `curr`. Prevents
/// re-filling an excessive amount of hash-table state after a long
/// LDM match.
pub fn ZSTD_ldm_limitTableUpdate(ms: &mut ZSTD_MatchState_t, curr: u32) {
    if curr > ms.nextToUpdate.wrapping_add(1024) {
        ms.nextToUpdate =
            curr.wrapping_sub(512u32.min(curr.wrapping_sub(ms.nextToUpdate).wrapping_sub(1024)));
    }
}

/// Port of `ZSTD_ldm_generateSequences_internal`.
///
/// Upstream takes `void const* src + size_t srcSize` and a window-base
/// pointer; the port here takes a `window_buf` view of the whole
/// window plus `ip_pos` / `iend_pos` offsets into it. `entry.offset`
/// fields are buffer-relative indexes into `window_buf`, matching
/// upstream's `split - window.base` semantics.
///
/// `lowestIndex` is the minimum valid buffer offset in the active
/// search domain. When an ext-dict is attached, this is
/// `ldmState.window.lowLimit`; otherwise it is `dictLimit`.
///
/// Returns the number of unconsumed literal bytes between the last
/// emitted sequence's anchor and `iend_pos` (upstream's `iend - anchor`).
///
pub fn ZSTD_ldm_generateSequences_internal(
    ldmState: &mut ldmState_t,
    rawSeqStore: &mut RawSeqStore_t,
    params: &ldmParams_t,
    window_buf: &[u8],
    ip_pos: usize,
    iend_pos: usize,
    lowestIndex: u32,
) -> usize {
    let minMatchLength = params.minMatchLength as usize;
    let entsPerBucket = 1u32 << params.bucketSizeLog;
    let hBits = params.hashLog - params.bucketSizeLog;
    let hashMask: u32 = if hBits >= 32 {
        u32::MAX
    } else {
        (1u32 << hBits) - 1
    };
    let extDict = ZSTD_window_hasExtDict(&ldmState.window);
    let dictLimit = ldmState.window.dictLimit as usize;
    let lowestIndex = lowestIndex as usize;
    let lowPrefixPtr = dictLimit;
    let dictStart = if extDict {
        ldmState
            .window
            .lowLimit
            .saturating_sub(ldmState.window.dictBase_offset) as usize
    } else {
        0
    };
    let dictEnd = if extDict {
        dictLimit.saturating_sub(ldmState.window.dictBase_offset as usize)
    } else {
        0
    };

    let istart = ip_pos;
    if iend_pos - istart < minMatchLength {
        return iend_pos - istart;
    }
    let ilimit = iend_pos - HASH_READ_SIZE;

    let mut anchor = istart;
    let mut ip = istart;

    let mut hashState = ldmRollingHashState_t {
        rolling: 0,
        stopMask: 0,
    };
    ZSTD_ldm_gear_init(&mut hashState, params);
    ZSTD_ldm_gear_reset(
        &mut hashState,
        &window_buf[ip..ip + minMatchLength],
        minMatchLength,
    );
    ip += minMatchLength;

    let mut splits = [0usize; LDM_BATCH_SIZE];

    while ip < ilimit {
        let mut numSplits: u32 = 0;
        let hashed = ZSTD_ldm_gear_feed(
            &mut hashState,
            &window_buf[ip..ilimit],
            &mut splits,
            &mut numSplits,
        );

        // Upstream caches (split, hash, checksum, bucket_idx) per
        // candidate; we recompute cheaply inside the main loop since
        // Rust's borrow rules make a simultaneous &mut hashTable +
        // &mut bucketOffsets + &candidates_scratch awkward.
        for &split in splits.iter().take(numSplits as usize) {
            let split_pos = ip + split - minMatchLength;
            let xxhash = XXH64(&window_buf[split_pos..split_pos + minMatchLength], 0);
            let hash = (xxhash as u32) & hashMask;
            let checksum = (xxhash >> 32) as u32;

            let newEntry = ldmEntry_t {
                offset: split_pos as u32,
                checksum,
            };

            // Overlap with a previously emitted sequence — table
            // insert only, no match.
            if split_pos < anchor {
                ZSTD_ldm_insertEntry(ldmState, hash, newEntry, params.bucketSizeLog);
                continue;
            }

            // Scan the bucket for the longest qualifying match. We
            // borrow the bucket *copy* so we can mutate the hash table
            // through insertEntry at the end without aliasing issues.
            let bucket_base = ZSTD_ldm_getBucket(ldmState, hash, params.bucketSizeLog);
            let mut forward_len: usize = 0;
            let mut backward_len: usize = 0;
            let mut best_total: usize = 0;
            let mut best_offset_index: Option<u32> = None;

            for k in 0..entsPerBucket as usize {
                let cur = ldmState.hashTable[bucket_base + k];
                if cur.checksum != checksum || cur.offset as usize <= lowestIndex {
                    continue;
                }
                let (cur_fwd, cur_bwd) = if extDict {
                    let curMatchBase = if cur.offset < ldmState.window.dictLimit {
                        0usize
                    } else {
                        1usize
                    };
                    let pMatch = if curMatchBase == 0 {
                        cur.offset.saturating_sub(ldmState.window.dictBase_offset) as usize
                    } else {
                        cur.offset.saturating_sub(ldmState.window.base_offset) as usize
                    };
                    let matchEnd = if curMatchBase == 0 { dictEnd } else { iend_pos };
                    let lowMatchPtr = if curMatchBase == 0 {
                        dictStart
                    } else {
                        lowPrefixPtr
                    };
                    let cur_fwd = ZSTD_count_2segments(
                        window_buf,
                        split_pos,
                        iend_pos,
                        lowPrefixPtr,
                        window_buf,
                        pMatch,
                        matchEnd,
                    );
                    if cur_fwd < minMatchLength {
                        continue;
                    }
                    let cur_bwd = ZSTD_ldm_countBackwardsMatch_2segments(
                        window_buf,
                        split_pos,
                        anchor,
                        window_buf,
                        pMatch,
                        lowMatchPtr,
                        window_buf,
                        dictStart,
                        dictEnd,
                    );
                    (cur_fwd, cur_bwd)
                } else {
                    let pMatch = cur.offset as usize;
                    let cur_fwd = ZSTD_count(window_buf, split_pos, pMatch, iend_pos);
                    if cur_fwd < minMatchLength {
                        continue;
                    }
                    let cur_bwd = ZSTD_ldm_countBackwardsMatch(
                        window_buf,
                        split_pos,
                        anchor,
                        window_buf,
                        pMatch,
                        lowestIndex,
                    );
                    (cur_fwd, cur_bwd)
                };
                let cur_total = cur_fwd + cur_bwd;
                if cur_total > best_total {
                    best_total = cur_total;
                    forward_len = cur_fwd;
                    backward_len = cur_bwd;
                    best_offset_index = Some(cur.offset);
                }
            }

            // No qualifying match — record position, advance.
            let Some(matched_offset) = best_offset_index else {
                ZSTD_ldm_insertEntry(ldmState, hash, newEntry, params.bucketSizeLog);
                continue;
            };

            // Emit sequence. Out of storage → error.
            if rawSeqStore.size == rawSeqStore.capacity {
                // Signal by leaving unconsumed input; upstream returns
                // dstSize_tooSmall, but we don't have a dst pointer —
                // surface via caller detecting anchor < iend_pos and
                // rawSeqStore.size == capacity.
                return iend_pos - anchor;
            }
            let seq = &mut rawSeqStore.seq[rawSeqStore.size];
            seq.litLength = (split_pos - backward_len - anchor) as u32;
            seq.matchLength = best_total as u32;
            seq.offset = (split_pos as u32).wrapping_sub(matched_offset);
            rawSeqStore.size += 1;

            // Insert only after reading the bucket — safe to do so
            // because we're done with `bestEntry`.
            ZSTD_ldm_insertEntry(ldmState, hash, newEntry, params.bucketSizeLog);

            anchor = split_pos + forward_len;

            // Jump ahead if match extended past the gear-feed boundary.
            if anchor > ip + hashed {
                let reset_start = anchor - minMatchLength;
                ZSTD_ldm_gear_reset(
                    &mut hashState,
                    &window_buf[reset_start..reset_start + minMatchLength],
                    minMatchLength,
                );
                ip = anchor - hashed;
                break;
            }
        }

        ip += hashed;
    }

    iend_pos - anchor
}

/// Port of `maybeSplitSequence`. Returns the current rawSeq, possibly
/// truncated so it fits within `remaining` bytes of input. If the
/// remaining window is ≤ the sequence's literal run, the match is
/// dropped (`offset = 0` signals "rest is literals"). If the
/// truncated match falls below `minMatch`, also drops it.
///
/// Advances `rawSeqStore` past `remaining` bytes on exit via
/// `ZSTD_ldm_skipSequences`, so the caller can call again without
/// double-consuming.
pub fn maybeSplitSequence(
    rawSeqStore: &mut RawSeqStore_t,
    remaining: u32,
    minMatch: u32,
) -> rawSeq {
    let mut sequence = rawSeqStore.seq[rawSeqStore.pos];
    debug_assert!(sequence.offset > 0);

    if remaining >= sequence.litLength.wrapping_add(sequence.matchLength) {
        rawSeqStore.pos += 1;
        return sequence;
    }
    // Sequence extends past `remaining` — clip it.
    if remaining <= sequence.litLength {
        sequence.offset = 0;
    } else if remaining < sequence.litLength.wrapping_add(sequence.matchLength) {
        sequence.matchLength = remaining.wrapping_sub(sequence.litLength);
        if sequence.matchLength < minMatch {
            sequence.offset = 0;
        }
    }
    ZSTD_ldm_skipSequences(rawSeqStore, remaining as usize, minMatch);
    sequence
}

/// Port of `ZSTD_ldm_skipRawSeqStoreBytes`. Advances both `pos` and
/// `posInSequence` forward by `nbBytes`, popping full sequences as
/// they're consumed. Mirror of `ZSTD_optLdm_skipRawSeqStoreBytes` in
/// `zstd_opt.rs` — same logic, separate symbol for 1:1 parity with
/// upstream.
pub fn ZSTD_ldm_skipRawSeqStoreBytes(rawSeqStore: &mut RawSeqStore_t, nbBytes: usize) {
    let mut currPos = (rawSeqStore.posInSequence as u32).wrapping_add(nbBytes as u32);
    while currPos > 0 && rawSeqStore.pos < rawSeqStore.size {
        let currSeq = rawSeqStore.seq[rawSeqStore.pos];
        let fullLen = currSeq.litLength.wrapping_add(currSeq.matchLength);
        if currPos >= fullLen {
            currPos = currPos.wrapping_sub(fullLen);
            rawSeqStore.pos += 1;
        } else {
            rawSeqStore.posInSequence = currPos as usize;
            return;
        }
    }
    if currPos == 0 || rawSeqStore.pos == rawSeqStore.size {
        rawSeqStore.posInSequence = 0;
    }
}

/// Port of `ZSTD_ldm_skipSequences`. Advances the read cursor through
/// `rawSeqStore` by `srcSize` bytes without emitting matches. Used
/// when a block compressor wants to skip past a region that LDM has
/// already annotated (e.g. when the outer frame chooses to emit raw
/// literals there).
///
/// When a partial match is consumed and the residual is shorter than
/// `minMatch`, it's folded into the next sequence's literals and the
/// current sequence is dropped.
pub fn ZSTD_ldm_skipSequences(rawSeqStore: &mut RawSeqStore_t, mut srcSize: usize, minMatch: u32) {
    while srcSize > 0 && rawSeqStore.pos < rawSeqStore.size {
        let pos = rawSeqStore.pos;
        let seq_litLength = rawSeqStore.seq[pos].litLength as usize;
        if srcSize <= seq_litLength {
            rawSeqStore.seq[pos].litLength =
                rawSeqStore.seq[pos].litLength.wrapping_sub(srcSize as u32);
            return;
        }
        srcSize -= seq_litLength;
        rawSeqStore.seq[pos].litLength = 0;
        let seq_matchLength = rawSeqStore.seq[pos].matchLength as usize;
        if srcSize < seq_matchLength {
            rawSeqStore.seq[pos].matchLength = rawSeqStore.seq[pos]
                .matchLength
                .wrapping_sub(srcSize as u32);
            if rawSeqStore.seq[pos].matchLength < minMatch {
                // Match too short — roll residual into next seq's
                // literals and drop this one.
                if rawSeqStore.pos + 1 < rawSeqStore.size {
                    let leftover = rawSeqStore.seq[pos].matchLength;
                    rawSeqStore.seq[pos + 1].litLength =
                        rawSeqStore.seq[pos + 1].litLength.wrapping_add(leftover);
                }
                rawSeqStore.pos += 1;
            }
            return;
        }
        srcSize -= seq_matchLength;
        rawSeqStore.seq[pos].matchLength = 0;
        rawSeqStore.pos += 1;
    }
}

/// Port of the outer `ZSTD_ldm_generateSequences`. Splits `src` into
/// 1 MiB chunks and runs `generateSequences_internal` on each,
/// stitching leftover literals from one chunk onto the first sequence
/// emitted by the next.
///
/// `window_buf` covers the full compressor window; `src_pos` /
/// `src_end` are offsets into it. `lowestIndex` is the minimum valid
/// buffer offset (upstream's `ldmState->window.dictLimit`).
#[allow(clippy::too_many_arguments)]
pub fn ZSTD_ldm_generateSequences(
    ldmState: &mut ldmState_t,
    sequences: &mut RawSeqStore_t,
    params: &ldmParams_t,
    window_buf: &[u8],
    src_pos: usize,
    src_end: usize,
    lowestIndex: u32,
) -> usize {
    let srcSize = src_end - src_pos;
    let maxDist = 1u32 << params.windowLog;
    const K_MAX_CHUNK_SIZE: usize = 1 << 20;
    let nbChunks = srcSize.div_ceil(K_MAX_CHUNK_SIZE).max(1);
    let mut leftoverSize: usize = 0;

    for chunk in 0..nbChunks {
        if sequences.size >= sequences.capacity {
            break;
        }
        let chunkStart = src_pos + chunk * K_MAX_CHUNK_SIZE;
        if chunkStart >= src_end {
            break;
        }
        let remaining = src_end - chunkStart;
        let chunkEnd = if remaining < K_MAX_CHUNK_SIZE {
            src_end
        } else {
            chunkStart + K_MAX_CHUNK_SIZE
        };
        let chunkSize = chunkEnd - chunkStart;
        let prevSize = sequences.size;
        let chunkStartAbs = chunkStart as u32;
        let chunkEndAbs = chunkEnd as u32;

        if ZSTD_window_needOverflowCorrection(
            &ldmState.window,
            0,
            maxDist,
            ldmState.loadedDictEnd,
            chunkStartAbs,
            chunkEndAbs,
        ) {
            let correction =
                ZSTD_window_correctOverflow(&mut ldmState.window, 0, maxDist, chunkStartAbs);
            ZSTD_ldm_reduceTable(&mut ldmState.hashTable, correction);
            ldmState.loadedDictEnd = 0;
        }

        ZSTD_window_enforceMaxDist(
            &mut ldmState.window,
            chunkEndAbs,
            maxDist,
            &mut ldmState.loadedDictEnd,
        );

        let chunkLowest = if ZSTD_window_hasExtDict(&ldmState.window) {
            ldmState.window.lowLimit
        } else {
            lowestIndex
        };
        let newLeftoverSize = ZSTD_ldm_generateSequences_internal(
            ldmState,
            sequences,
            params,
            window_buf,
            chunkStart,
            chunkEnd,
            chunkLowest,
        );

        if prevSize < sequences.size {
            sequences.seq[prevSize].litLength = sequences.seq[prevSize]
                .litLength
                .wrapping_add(leftoverSize as u32);
            leftoverSize = newLeftoverSize;
        } else {
            debug_assert_eq!(newLeftoverSize, chunkSize);
            leftoverSize += chunkSize;
        }
    }
    0
}

pub fn ZSTD_ldm_blockCompress(
    rawSeqStore: &mut RawSeqStore_t,
    ms: &mut ZSTD_MatchState_t,
    seqStore: &mut crate::compress::seq_store::SeqStore_t,
    rep: &mut [u32; 3],
    src: &[u8],
) -> usize {
    use crate::compress::match_state::ZSTD_matchState_dictMode;
    use crate::compress::seq_store::{ZSTD_storeSeq, ZSTD_updateRep, OFFSET_TO_OFFBASE};
    use crate::compress::zstd_compress::ZSTD_selectBlockCompressor;

    let cParams = ms.cParams;
    let minMatch = cParams.minMatch;
    let blockCompressor = ZSTD_selectBlockCompressor(
        cParams.strategy,
        ZSTD_ParamSwitch_e::ZSTD_ps_disable,
        ZSTD_matchState_dictMode(ms),
    );
    let istart = 0usize;
    let iend = src.len();
    let mut ip = istart;

    if cParams.strategy >= ZSTD_btopt {
        ms.ldmSeqStore = Some(rawSeqStore.clone());
        let lastLLSize = blockCompressor(ms, seqStore, rep, src);
        ms.ldmSeqStore = None;
        ZSTD_ldm_skipRawSeqStoreBytes(rawSeqStore, src.len());
        return lastLLSize;
    }

    debug_assert!(rawSeqStore.pos <= rawSeqStore.size);
    debug_assert!(rawSeqStore.size <= rawSeqStore.capacity);

    while rawSeqStore.pos < rawSeqStore.size && ip < iend {
        let sequence = maybeSplitSequence(rawSeqStore, (iend - ip) as u32, minMatch);
        if sequence.offset == 0 {
            break;
        }
        if ip + sequence.litLength as usize + sequence.matchLength as usize > iend {
            return crate::common::error::ERROR(crate::common::error::ErrorCode::Generic);
        }

        ZSTD_ldm_limitTableUpdate(ms, ip as u32);
        ZSTD_ldm_fillFastTables(ms, &src[..ip + sequence.litLength as usize]);

        let newLitLength = blockCompressor(
            ms,
            seqStore,
            rep,
            &src[ip..ip + sequence.litLength as usize],
        );
        if crate::common::error::ERR_isError(newLitLength) {
            return newLitLength;
        }
        ip += sequence.litLength as usize;
        ZSTD_updateRep(rep, OFFSET_TO_OFFBASE(sequence.offset), 0);
        ZSTD_storeSeq(
            seqStore,
            newLitLength,
            &src[ip - newLitLength..],
            OFFSET_TO_OFFBASE(sequence.offset),
            sequence.matchLength as usize,
        );
        ip += sequence.matchLength as usize;
    }

    ZSTD_ldm_limitTableUpdate(ms, ip as u32);
    ZSTD_ldm_fillFastTables(ms, &src[..ip]);
    blockCompressor(ms, seqStore, rep, &src[ip..])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_cparams(strategy: u32) -> ZSTD_compressionParameters {
        ZSTD_compressionParameters {
            windowLog: 23,
            chainLog: 22,
            hashLog: 21,
            searchLog: 5,
            minMatch: 4,
            targetLength: 32,
            strategy,
        }
    }

    #[test]
    fn adjust_parameters_defaults_hashRateLog_from_strategy() {
        // strategy=3 (greedy) → 7 - 3/3 = 6
        let mut lp = ldmParams_t::default();
        ZSTD_ldm_adjustParameters(&mut lp, &base_cparams(3));
        assert_eq!(lp.hashRateLog, 6);
        assert_eq!(lp.minMatchLength, LDM_MIN_MATCH_LENGTH);
    }

    #[test]
    fn adjust_parameters_btultra_halves_minmatch() {
        let mut lp = ldmParams_t::default();
        // strategy=8 (btultra) → minMatch = 64/2 = 32
        ZSTD_ldm_adjustParameters(&mut lp, &base_cparams(8));
        assert_eq!(lp.minMatchLength, LDM_MIN_MATCH_LENGTH / 2);
    }

    #[test]
    fn adjust_parameters_clamps_hashLog_to_range() {
        let mut lp = ldmParams_t::default();
        let mut cp = base_cparams(9);
        cp.windowLog = 8; // smaller than hashRateLog
        ZSTD_ldm_adjustParameters(&mut lp, &cp);
        assert_eq!(lp.hashLog, ZSTD_HASHLOG_MIN);
    }

    #[test]
    fn get_table_size_zero_when_ldm_disabled() {
        let lp = ldmParams_t {
            enableLdm: ZSTD_ParamSwitch_e::ZSTD_ps_auto,
            hashLog: 20,
            bucketSizeLog: 4,
            ..Default::default()
        };
        assert_eq!(ZSTD_ldm_getTableSize(lp), 0);
    }

    #[test]
    fn get_table_size_nonzero_when_enabled() {
        let lp = ldmParams_t {
            enableLdm: ZSTD_ParamSwitch_e::ZSTD_ps_enable,
            hashLog: 20,
            bucketSizeLog: 4,
            ..Default::default()
        };
        let sz = ZSTD_ldm_getTableSize(lp);
        assert!(sz > 0);
    }

    #[test]
    fn gear_hash_init_nondefault_stopmask() {
        let mut s = ldmRollingHashState_t {
            rolling: 0,
            stopMask: 0,
        };
        let params = ldmParams_t {
            enableLdm: ZSTD_ParamSwitch_e::ZSTD_ps_enable,
            minMatchLength: 64,
            hashRateLog: 7,
            ..Default::default()
        };
        ZSTD_ldm_gear_init(&mut s, &params);
        // minMatch=64, rate=7 → stopMask = ((1<<7)-1) << (64-7) = 0x7F << 57.
        assert_eq!(s.stopMask, 0x7Fu64 << 57);
        assert_eq!(s.rolling, u32::MAX as u64);
    }

    #[test]
    fn gear_hash_feed_produces_splits_on_random_data() {
        // A gear hash with hashRateLog=4 → stopMask hits ~1/16 of
        // iterations. On 10 KB of varied bytes, expect many splits.
        let params = ldmParams_t {
            enableLdm: ZSTD_ParamSwitch_e::ZSTD_ps_enable,
            minMatchLength: 64,
            hashRateLog: 4,
            ..Default::default()
        };
        let mut s = ldmRollingHashState_t {
            rolling: 0,
            stopMask: 0,
        };
        ZSTD_ldm_gear_init(&mut s, &params);
        let data: Vec<u8> = (0..10_000u32)
            .map(|i| ((i.wrapping_mul(31).wrapping_add(7)) & 0xFF) as u8)
            .collect();
        let mut splits = [0usize; LDM_BATCH_SIZE];
        let mut numSplits = 0u32;
        let consumed = ZSTD_ldm_gear_feed(&mut s, &data, &mut splits, &mut numSplits);
        assert!(consumed > 0);
        // 1/16 density over 10KB → roughly 500-700 splits, capped at
        // LDM_BATCH_SIZE (64).
        assert!(numSplits > 0);
        assert!(numSplits <= LDM_BATCH_SIZE as u32);
    }

    #[test]
    fn gear_hash_reset_seeds_state_without_recording_splits() {
        let params = ldmParams_t {
            enableLdm: ZSTD_ParamSwitch_e::ZSTD_ps_enable,
            minMatchLength: 64,
            hashRateLog: 4,
            ..Default::default()
        };
        let mut s = ldmRollingHashState_t {
            rolling: 0,
            stopMask: 0,
        };
        ZSTD_ldm_gear_init(&mut s, &params);
        let initial = s.rolling;
        let warmup = [0x42u8; 64];
        ZSTD_ldm_gear_reset(&mut s, &warmup, 64);
        assert_ne!(
            s.rolling, initial,
            "rolling should have mutated during reset"
        );
    }

    fn ldm_params_for(strategy: u32) -> ldmParams_t {
        let mut lp = ldmParams_t {
            enableLdm: ZSTD_ParamSwitch_e::ZSTD_ps_enable,
            ..Default::default()
        };
        ZSTD_ldm_adjustParameters(&mut lp, &base_cparams(strategy));
        lp
    }

    #[test]
    fn ldm_state_sizes_tables_from_params() {
        let lp = ldm_params_for(3);
        let st = ldmState_t::new(&lp);
        assert_eq!(st.hashTable.len(), 1usize << lp.hashLog);
        assert_eq!(
            st.bucketOffsets.len(),
            1usize << (lp.hashLog - lp.bucketSizeLog),
        );
    }

    #[test]
    fn insert_entry_rolls_per_bucket_cursor() {
        let lp = ldm_params_for(3);
        let mut st = ldmState_t::new(&lp);
        let hash = 42u32;
        // Insert 1 << bucketSizeLog entries; bucket cursor should
        // return to zero.
        let capacity = 1u32 << lp.bucketSizeLog;
        for i in 0..capacity {
            ZSTD_ldm_insertEntry(
                &mut st,
                hash,
                ldmEntry_t {
                    offset: 1000 + i,
                    checksum: i,
                },
                lp.bucketSizeLog,
            );
        }
        assert_eq!(st.bucketOffsets[hash as usize], 0);
        // The last inserted entry lives at bucket_base + capacity - 1.
        let base = ZSTD_ldm_getBucket(&st, hash, lp.bucketSizeLog);
        assert_eq!(
            st.hashTable[base + capacity as usize - 1].offset,
            1000 + capacity - 1
        );
    }

    #[test]
    fn fill_hash_table_populates_some_entries() {
        // Fill a reasonable-sized slab with varied bytes; expect LDM
        // to record at least a few entries.
        let lp = ldm_params_for(3);
        let mut st = ldmState_t::new(&lp);
        let data: Vec<u8> = (0..200_000u32)
            .map(|i| ((i.wrapping_mul(2654435761u32)) >> 24) as u8)
            .collect();
        ZSTD_ldm_fillHashTable(&mut st, &data, 0, &lp);
        let any_nonzero = st
            .hashTable
            .iter()
            .any(|e| e.offset != 0 || e.checksum != 0);
        assert!(
            any_nonzero,
            "expected fillHashTable to record at least one entry"
        );
    }

    #[test]
    fn fill_hash_table_entry_offsets_in_range() {
        // Every recorded entry's offset should be in
        // [abs_start, abs_start + data.len() - minMatchLength].
        let lp = ldm_params_for(3);
        let mut st = ldmState_t::new(&lp);
        let data: Vec<u8> = (0..150_000u32)
            .map(|i| ((i.wrapping_mul(2654435761u32)) >> 24) as u8)
            .collect();
        let abs_start = 1_000_000u32;
        ZSTD_ldm_fillHashTable(&mut st, &data, abs_start, &lp);
        let max_offset = abs_start
            .wrapping_add(data.len() as u32)
            .wrapping_sub(lp.minMatchLength);
        for e in st.hashTable.iter() {
            if e.offset != 0 || e.checksum != 0 {
                assert!(e.offset >= abs_start);
                assert!(e.offset <= max_offset);
            }
        }
    }

    #[test]
    fn countBackwardsMatch_stops_at_anchor() {
        let data = b"abcdefghij";
        // pIn_pos = 10, pAnchor_pos = 5, match against same buffer at
        // pMatch_pos = 10 — every byte trivially matches so we stop at
        // the anchor after 5 bytes.
        let n = ZSTD_ldm_countBackwardsMatch(data, 10, 5, data, 10, 0);
        assert_eq!(n, 5);
    }

    #[test]
    fn countBackwardsMatch_stops_at_mismatch() {
        let a = b"xxxyyzz";
        let b = b"yyzzz";
        // pIn_pos = 7 (end of a), pAnchor_pos = 0
        // pMatch_pos = 5 (end of b), pMatchBase_pos = 0
        // walk back: zz match, yy match (4 bytes), then 'y' vs 'y' match,
        // 'x' vs ?... b only has 5 bytes so pMatch hits base after 5.
        let n = ZSTD_ldm_countBackwardsMatch(a, 7, 0, b, 5, 0);
        // a[6]='z' b[4]='z' ✓, a[5]='z' b[3]='z' ✓, a[4]='y' b[2]='z' ✗
        assert_eq!(n, 2);
    }

    #[test]
    fn countBackwardsMatch_2segments_crosses_into_extdict() {
        // Split setup: "main" has "aabbccdd" at positions 0..8,
        // "ext" has "ccbbaa" at positions 0..6.
        // pIn walking back from position 8 in main with input "xxxaabbccdd"
        // — we want the match to span the segment boundary.
        let pIn = b"xxxaabbccdd";
        let pMatch = b"xxxaabbccdd"; // segment A, base at 3
        let pExt = b"ccbbaa"; // segment B

        // Walk back from pIn_pos=11, anchor=3, pMatch_pos=11, pMatchBase=5
        // pIn[10..]='d' vs pMatch[10..]='d' ✓
        // pIn[9..]='d' vs pMatch[9..]='d' ✓
        // pIn[8..]='c' vs pMatch[8..]='c' ✓
        // pIn[7..]='c' vs pMatch[7..]='c' ✓
        // pIn[6..]='b' vs pMatch[6..]='b' ✓
        // pMatch_pos=6, pMatchBase=5: still > base, continue
        // pIn[5..]='b' vs pMatch[5..]='b' ✓
        // pMatch_pos=5 == pMatchBase=5 — plain backward stops, 2seg kicks in.
        // Continue in ext at pExtDictEnd_pos=6:
        //   pIn[4..]='a' vs pExt[5..]='a' ✓
        //   pIn[3..]='a' vs pExt[4..]='a' ✓
        //   pIn[2..]='x' vs pExt[3..]='b' ✗ → stop.
        let n = ZSTD_ldm_countBackwardsMatch_2segments(pIn, 11, 3, pMatch, 11, 5, pExt, 0, 6);
        assert_eq!(n, 8);
    }

    #[test]
    fn reduce_table_shifts_offsets() {
        let mut t = vec![
            ldmEntry_t {
                offset: 100,
                checksum: 1,
            },
            ldmEntry_t {
                offset: 50,
                checksum: 2,
            },
            ldmEntry_t {
                offset: 10,
                checksum: 3,
            },
        ];
        ZSTD_ldm_reduceTable(&mut t, 60);
        assert_eq!(t[0].offset, 40);
        assert_eq!(t[1].offset, 0); // underflow clamped
        assert_eq!(t[2].offset, 0); // underflow clamped
        assert_eq!(t[0].checksum, 1); // checksums untouched
    }

    #[test]
    fn limit_table_update_rewinds_far_behind_cursor() {
        let mut ms = ZSTD_MatchState_t::new(base_cparams(3));
        ms.nextToUpdate = 100;
        // curr = 100 + 1024 + 800 = ahead by 1824
        ZSTD_ldm_limitTableUpdate(&mut ms, 1924);
        // curr - MIN(512, curr-nextToUpdate-1024) = 1924 - 512 = 1412
        assert_eq!(ms.nextToUpdate, 1412);
    }

    #[test]
    fn limit_table_update_no_op_when_close() {
        let mut ms = ZSTD_MatchState_t::new(base_cparams(3));
        ms.nextToUpdate = 100;
        // curr only 500 ahead — under the 1024 threshold.
        ZSTD_ldm_limitTableUpdate(&mut ms, 600);
        assert_eq!(ms.nextToUpdate, 100);
    }

    #[test]
    fn fill_fast_tables_no_op_for_lazy_strategies() {
        for strategy in [ZSTD_greedy, ZSTD_lazy, ZSTD_lazy2, ZSTD_btlazy2] {
            let mut ms = ZSTD_MatchState_t::new(base_cparams(strategy));
            let before: Vec<u32> = ms.hashTable.clone();
            let data = vec![7u8; 1024];
            ZSTD_ldm_fillFastTables(&mut ms, &data);
            assert_eq!(
                ms.hashTable, before,
                "lazy/bt strategies should not touch hashTable"
            );
        }
    }

    #[test]
    fn fill_fast_tables_seeds_for_ZSTD_fast() {
        let mut ms = ZSTD_MatchState_t::new(base_cparams(ZSTD_fast));
        let data = vec![7u8; 1024];
        let before_nonzero = ms.hashTable.iter().filter(|&&x| x != 0).count();
        ZSTD_ldm_fillFastTables(&mut ms, &data);
        let after_nonzero = ms.hashTable.iter().filter(|&&x| x != 0).count();
        // Constant-byte run fills exactly one slot (all hash to same
        // index). That's still >= before.
        assert!(after_nonzero >= before_nonzero);
    }

    #[test]
    fn generateSequences_prefixOnly_finds_long_repeat() {
        // Build a buffer that repeats a 200-byte block after a big
        // gap. Minimum LDM match is 64 bytes, so the repeat should be
        // discovered and emitted as a single sequence.
        let block: Vec<u8> = (0..200u32)
            .map(|i| ((i.wrapping_mul(2654435761)) >> 24) as u8)
            .collect();
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&block);
        // filler must NOT match the block — use a different PRNG seed
        let filler: Vec<u8> = (0..70_000u32)
            .map(|i| ((i.wrapping_mul(0xDEADBEEFu32).wrapping_add(1)) >> 16) as u8)
            .collect();
        buf.extend_from_slice(&filler);
        let repeat_start = buf.len();
        buf.extend_from_slice(&block);

        let lp = ldm_params_for(3);
        let mut st = ldmState_t::new(&lp);
        let mut store = RawSeqStore_t::with_capacity(1024);

        let leftover =
            ZSTD_ldm_generateSequences_internal(&mut st, &mut store, &lp, &buf, 0, buf.len(), 0);
        assert!(leftover <= buf.len());
        assert!(store.size > 0, "expected at least one LDM sequence");

        // Total consumed = sum(litLength + matchLength) for emitted
        // sequences. The last un-emitted range is `leftover` bytes.
        let consumed: u64 = store.seq[..store.size]
            .iter()
            .map(|s| s.litLength as u64 + s.matchLength as u64)
            .sum();
        assert_eq!(consumed + leftover as u64, buf.len() as u64);

        // At least one emitted sequence should correspond to the
        // repeated block: its offset should be ~repeat_start distance,
        // and matchLength ≥ minMatch.
        let hit = store.seq[..store.size].iter().any(|s| {
            s.matchLength >= lp.minMatchLength
                && (s.offset as usize) > repeat_start - block.len() - 100
                && (s.offset as usize) < repeat_start + 100
        });
        assert!(hit, "no sequence matched the long-distance repeat");
    }

    #[test]
    fn generateSequences_prefixOnly_no_match_on_random() {
        // Random buffer — LDM shouldn't produce sequences with very
        // long match lengths (anything over minMatch is a spurious
        // collision, which is astronomically unlikely on 32-bit XXH64).
        let buf: Vec<u8> = (0..20_000u32)
            .map(|i| ((i.wrapping_mul(0xDEADBEEFu32)) >> 16) as u8)
            .collect();
        let lp = ldm_params_for(3);
        let mut st = ldmState_t::new(&lp);
        let mut store = RawSeqStore_t::with_capacity(1024);

        let leftover =
            ZSTD_ldm_generateSequences_internal(&mut st, &mut store, &lp, &buf, 0, buf.len(), 0);
        // Allowed to be zero sequences; most of the buffer stays
        // as leftover literals.
        assert_eq!(
            store.seq[..store.size]
                .iter()
                .map(|s| s.litLength as u64 + s.matchLength as u64)
                .sum::<u64>()
                + leftover as u64,
            buf.len() as u64,
        );
    }

    #[test]
    fn generateSequences_extdict_finds_dictionary_match() {
        let dict = b"TTGACCATGGTACCTA".repeat(16);
        let src = [b"noise-".as_slice(), &dict[..96], b"-tail".as_slice()].concat();
        let buf = [dict.clone(), src.clone()].concat();

        let lp = ldm_params_for(3);
        let mut st = ldmState_t::new(&lp);
        st.window.dictBase_offset = 0;
        st.window.lowLimit = 0;
        st.window.dictLimit = dict.len() as u32;
        st.window.base_offset = dict.len() as u32;
        st.window.nextSrc = buf.len() as u32;

        ZSTD_ldm_fillHashTable(&mut st, &buf[..dict.len()], 0, &lp);

        let mut store = RawSeqStore_t::with_capacity(1024);
        let lowest = st.window.lowLimit;
        let leftover = ZSTD_ldm_generateSequences_internal(
            &mut st,
            &mut store,
            &lp,
            &buf,
            dict.len(),
            buf.len(),
            lowest,
        );

        assert!(
            leftover < src.len(),
            "expected ext-dict path to consume source bytes"
        );
        assert!(
            store.size > 0,
            "expected at least one ext-dict-backed LDM sequence"
        );
        assert!(
            store.seq[..store.size]
                .iter()
                .any(|s| s.matchLength >= lp.minMatchLength && s.offset > 0),
            "expected a real ext-dict match candidate"
        );
    }

    #[test]
    fn maybeSplitSequence_returns_whole_seq_when_fits() {
        let mut s = RawSeqStore_t::with_capacity(2);
        s.seq[0] = rawSeq {
            offset: 100,
            litLength: 5,
            matchLength: 10,
        };
        s.size = 1;
        let out = maybeSplitSequence(&mut s, 20, 4);
        assert_eq!(out.offset, 100);
        assert_eq!(out.litLength, 5);
        assert_eq!(out.matchLength, 10);
        assert_eq!(s.pos, 1);
    }

    #[test]
    fn maybeSplitSequence_truncates_match_when_partial() {
        let mut s = RawSeqStore_t::with_capacity(2);
        s.seq[0] = rawSeq {
            offset: 100,
            litLength: 5,
            matchLength: 20,
        };
        s.size = 1;
        // remaining = 10 → lit=5, match clipped to 5 (still ≥ minMatch).
        let out = maybeSplitSequence(&mut s, 10, 4);
        assert_eq!(out.offset, 100);
        assert_eq!(out.matchLength, 5);
    }

    #[test]
    fn maybeSplitSequence_drops_short_match() {
        let mut s = RawSeqStore_t::with_capacity(2);
        s.seq[0] = rawSeq {
            offset: 100,
            litLength: 5,
            matchLength: 20,
        };
        s.size = 1;
        // remaining = 8 → lit=5, match clipped to 3 — below minMatch=4.
        let out = maybeSplitSequence(&mut s, 8, 4);
        assert_eq!(out.offset, 0);
    }

    #[test]
    fn maybeSplitSequence_rest_is_literals_when_lit_dominates() {
        let mut s = RawSeqStore_t::with_capacity(2);
        s.seq[0] = rawSeq {
            offset: 100,
            litLength: 50,
            matchLength: 20,
        };
        s.size = 1;
        // remaining = 40 ≤ litLength=50 → offset cleared.
        let out = maybeSplitSequence(&mut s, 40, 4);
        assert_eq!(out.offset, 0);
    }

    #[test]
    fn ldm_skipRawSeqStoreBytes_mirrors_optLdm_variant() {
        // Same input, two different cursor impls — results must match.
        let seqs = [
            rawSeq {
                offset: 1,
                litLength: 3,
                matchLength: 7,
            },
            rawSeq {
                offset: 2,
                litLength: 5,
                matchLength: 8,
            },
        ];
        let mut a = RawSeqStore_t::with_capacity(2);
        a.seq[0] = seqs[0];
        a.seq[1] = seqs[1];
        a.size = 2;
        let mut b = RawSeqStore_t::with_capacity(2);
        b.seq[0] = seqs[0];
        b.seq[1] = seqs[1];
        b.size = 2;

        ZSTD_ldm_skipRawSeqStoreBytes(&mut a, 12);
        crate::compress::zstd_opt::ZSTD_optLdm_skipRawSeqStoreBytes(&mut b, 12);
        assert_eq!(a.pos, b.pos);
        assert_eq!(a.posInSequence, b.posInSequence);
    }

    #[test]
    fn skip_sequences_consumes_literals_partially() {
        let mut store = RawSeqStore_t::with_capacity(4);
        store.seq[0] = rawSeq {
            offset: 100,
            litLength: 50,
            matchLength: 30,
        };
        store.size = 1;
        ZSTD_ldm_skipSequences(&mut store, 20, 4);
        // 20 ≤ 50 → litLength shrinks, no advance.
        assert_eq!(store.seq[0].litLength, 30);
        assert_eq!(store.seq[0].matchLength, 30);
        assert_eq!(store.pos, 0);
    }

    #[test]
    fn skip_sequences_drops_short_match_residual() {
        let mut store = RawSeqStore_t::with_capacity(4);
        store.seq[0] = rawSeq {
            offset: 100,
            litLength: 5,
            matchLength: 10,
        };
        store.seq[1] = rawSeq {
            offset: 200,
            litLength: 3,
            matchLength: 20,
        };
        store.size = 2;
        // Skip 5 (full lit) + 7 (part of match) = 12 bytes. Residual
        // matchLength = 3, below minMatch=4, so it's dropped and the
        // 3 bytes move to seq[1].litLength.
        ZSTD_ldm_skipSequences(&mut store, 12, 4);
        assert_eq!(store.pos, 1);
        assert_eq!(store.seq[1].litLength, 6); // 3 + 3
    }

    #[test]
    fn skip_sequences_advances_past_full_sequences() {
        let mut store = RawSeqStore_t::with_capacity(4);
        store.seq[0] = rawSeq {
            offset: 100,
            litLength: 4,
            matchLength: 8,
        };
        store.seq[1] = rawSeq {
            offset: 200,
            litLength: 2,
            matchLength: 6,
        };
        store.size = 2;
        ZSTD_ldm_skipSequences(&mut store, 12, 4); // 4+8 = full seq[0]
        assert_eq!(store.pos, 1);
    }

    #[test]
    fn generateSequences_outer_stitches_leftover_literals() {
        // Build a buffer larger than one K_MAX_CHUNK_SIZE to exercise
        // the chunk-stitch path. Two 700 KB halves so chunk boundary
        // lands around 1 MB mark.
        let mut buf = Vec::with_capacity(1_400_000);
        let block: Vec<u8> = (0..200u32)
            .map(|i| ((i.wrapping_mul(2654435761)) >> 24) as u8)
            .collect();
        buf.extend_from_slice(&block);
        let filler: Vec<u8> = (0..1_399_600u32)
            .map(|i| ((i.wrapping_mul(0xDEADBEEFu32).wrapping_add(1)) >> 16) as u8)
            .collect();
        buf.extend_from_slice(&filler);
        let repeat_pos = buf.len();
        buf.extend_from_slice(&block);

        let lp = ldm_params_for(3);
        let mut st = ldmState_t::new(&lp);
        let mut store = RawSeqStore_t::with_capacity(4096);

        ZSTD_ldm_generateSequences(&mut st, &mut store, &lp, &buf, 0, buf.len(), 0);
        assert!(store.size > 0);
        // A repeat that starts at >1 MB distance should be caught.
        let hit = store.seq[..store.size].iter().any(|s| {
            s.matchLength >= lp.minMatchLength
                && (s.offset as usize) > repeat_pos - block.len() - 200
                && (s.offset as usize) < repeat_pos + 200
        });
        assert!(
            hit,
            "outer driver missed long-distance repeat past chunk boundary"
        );
    }

    #[test]
    fn raw_seq_store_with_capacity_is_empty() {
        let s = RawSeqStore_t::with_capacity(128);
        assert_eq!(s.capacity, 128);
        assert_eq!(s.size, 0);
        assert_eq!(s.pos, 0);
        assert_eq!(s.seq.len(), 128);
    }

    #[test]
    fn get_max_nb_seq_scales_with_chunk() {
        let lp = ldmParams_t {
            enableLdm: ZSTD_ParamSwitch_e::ZSTD_ps_enable,
            minMatchLength: 64,
            ..Default::default()
        };
        assert_eq!(ZSTD_ldm_getMaxNbSeq(lp, 65536), 1024);
        assert_eq!(ZSTD_ldm_getMaxNbSeq(lp, 0), 0);
    }

    #[test]
    fn generateSequences_enforces_max_distance_on_window_state() {
        let lp = ldmParams_t {
            enableLdm: ZSTD_ParamSwitch_e::ZSTD_ps_enable,
            hashLog: 8,
            bucketSizeLog: 4,
            minMatchLength: 8,
            hashRateLog: 4,
            windowLog: 10,
        };
        let mut st = ldmState_t::new(&lp);
        st.window.lowLimit = 5;
        st.window.dictLimit = 10;
        st.loadedDictEnd = 50;

        let data = vec![0u8; 2048];
        let rc = ZSTD_ldm_generateSequences(
            &mut st,
            &mut RawSeqStore_t::with_capacity(8),
            &lp,
            &data,
            0,
            data.len(),
            0,
        );

        assert_eq!(rc, 0);
        assert_eq!(st.window.lowLimit, 2048 - (1u32 << lp.windowLog));
        assert_eq!(st.window.dictLimit, st.window.lowLimit);
        assert_eq!(st.loadedDictEnd, 0);
    }

    #[test]
    fn blockCompress_accepts_empty_ldm_store_and_literals_only_input() {
        use crate::compress::match_state::{ZSTD_MatchState_t, ZSTD_compressionParameters};
        use crate::compress::seq_store::SeqStore_t;

        let mut store = RawSeqStore_t::with_capacity(4);
        let mut ms = ZSTD_MatchState_t::new(ZSTD_compressionParameters {
            windowLog: 20,
            hashLog: 10,
            chainLog: 10,
            minMatch: 4,
            strategy: ZSTD_fast,
            ..Default::default()
        });
        let mut seq = SeqStore_t::with_capacity(16, 256);
        let mut rep = [1u32, 4, 8];
        let rc = ZSTD_ldm_blockCompress(&mut store, &mut ms, &mut seq, &mut rep, b"x");
        assert_eq!(rc, 1);
    }

    #[test]
    fn blockCompress_materializes_queued_ldm_sequence_into_seqstore() {
        use crate::compress::match_state::{ZSTD_MatchState_t, ZSTD_compressionParameters};
        use crate::compress::seq_store::{SeqStore_t, ZSTD_getSequenceLength, OFFSET_TO_OFFBASE};

        let mut store = RawSeqStore_t::with_capacity(4);
        store.seq[0] = rawSeq {
            offset: 9,
            litLength: 4,
            matchLength: 8,
        };
        store.size = 1;

        let mut ms = ZSTD_MatchState_t::new(ZSTD_compressionParameters {
            windowLog: 20,
            hashLog: 10,
            chainLog: 10,
            minMatch: 4,
            strategy: ZSTD_fast,
            ..Default::default()
        });
        let mut seq = SeqStore_t::with_capacity(16, 256);
        let mut rep = [1u32, 4, 8];
        let src = b"abcdEFGHIJKLMNOPQRST";

        let trailing = ZSTD_ldm_blockCompress(&mut store, &mut ms, &mut seq, &mut rep, src);

        assert!(
            store.pos >= store.size,
            "queued raw sequence should be consumed"
        );
        let ldm_idx = seq
            .sequences
            .iter()
            .position(|s| s.offBase == OFFSET_TO_OFFBASE(9))
            .expect("queued LDM offset should be emitted");
        let seq_len = ZSTD_getSequenceLength(&seq, ldm_idx);
        assert_eq!(seq_len.matchLength, 8);
        assert!(trailing <= src.len());
    }

    #[test]
    fn blockCompress_btopt_exposes_ldm_sequences_to_opt_parser_and_clears_bridge() {
        use crate::compress::match_state::{ZSTD_MatchState_t, ZSTD_compressionParameters};
        use crate::compress::seq_store::{SeqStore_t, OFFSET_TO_OFFBASE};

        let mut store = RawSeqStore_t::with_capacity(2);
        store.seq[0] = rawSeq {
            offset: 4,
            litLength: 4,
            matchLength: 24,
        };
        store.seq[1] = rawSeq {
            offset: 4,
            litLength: 128,
            matchLength: 24,
        };
        store.size = 2;

        let cp = ZSTD_compressionParameters {
            windowLog: 18,
            chainLog: 15,
            hashLog: 14,
            searchLog: 4,
            minMatch: 4,
            targetLength: 32,
            strategy: ZSTD_btopt,
        };
        let mut ms = ZSTD_MatchState_t::new(cp);
        ms.chainTable = vec![0u32; 1 << cp.chainLog];
        ms.hashLog3 = 12;
        ms.hashTable3 = vec![0u32; 1 << ms.hashLog3];
        let mut seq = SeqStore_t::with_capacity(4096, 131072);
        let mut rep = [1u32, 4, 8];
        let src = b"abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

        let trailing = ZSTD_ldm_blockCompress(&mut store, &mut ms, &mut seq, &mut rep, src);

        assert!(trailing < src.len(), "btopt should use the LDM candidate");
        assert!(
            seq.sequences
                .iter()
                .any(|s| s.offBase == OFFSET_TO_OFFBASE(4) && s.mlBase as usize + 4 >= 24),
            "LDM candidate should be visible to the opt parser"
        );
        assert!(
            ms.ldmSeqStore.is_none(),
            "temporary opt-parser LDM bridge must not leak across blocks"
        );
        assert!(
            store.pos > 0 || store.posInSequence > 0,
            "raw sequence cursor should advance after the btopt block"
        );
    }
}
