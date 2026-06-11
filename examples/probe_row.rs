// Probe greedy row-hash and dump output for row-active input sizes.
// Argv: [n=32768] [wlog=17]
use zstd_pure_rs::common::error::ERR_isError;
use zstd_pure_rs::common::xxhash::XXH64;
use zstd_pure_rs::compress::match_state::ZSTD_rowMatchFinderUsed;
use zstd_pure_rs::compress::zstd_compress::{
    ZSTD_CCtx, ZSTD_CCtx_setParameter, ZSTD_cParameter, ZSTD_compress2, ZSTD_compressBound,
    ZSTD_createCCtx,
};

const SENTINEL_N: usize = 32768;
const SENTINEL_WLOG: i32 = 17;
const SENTINEL_SIZE: usize = 9925;
const SENTINEL_XXH64: u64 = 0xf434_870c_c257_55a8;

fn set(cctx: &mut ZSTD_CCtx, param: ZSTD_cParameter, value: i32) {
    let rc = ZSTD_CCtx_setParameter(cctx, param, value);
    assert!(
        !ERR_isError(rc),
        "setParameter({param:?}, {value}) failed: 0x{rc:x}"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(32768);
    let wlog: i32 = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(17);
    let input = std::fs::read("tests/fixtures/zstd_h.txt").unwrap();
    let src = &input[..n.min(input.len())];
    let mut cctx = ZSTD_createCCtx().unwrap();
    set(
        &mut cctx,
        ZSTD_cParameter::ZSTD_c_strategy,
        zstd_pure_rs::compress::zstd_compress_sequences::ZSTD_greedy as i32,
    );
    set(&mut cctx, ZSTD_cParameter::ZSTD_c_windowLog, wlog);
    set(&mut cctx, ZSTD_cParameter::ZSTD_c_chainLog, 16);
    set(&mut cctx, ZSTD_cParameter::ZSTD_c_hashLog, 17);
    set(&mut cctx, ZSTD_cParameter::ZSTD_c_searchLog, 3);
    set(&mut cctx, ZSTD_cParameter::ZSTD_c_minMatch, 4);
    set(&mut cctx, ZSTD_cParameter::ZSTD_c_targetLength, 2);
    set(&mut cctx, ZSTD_cParameter::ZSTD_c_useRowMatchFinder, 1);
    set(&mut cctx, ZSTD_cParameter::ZSTD_c_contentSizeFlag, 1);
    set(&mut cctx, ZSTD_cParameter::ZSTD_c_checksumFlag, 0);
    let mut dst = vec![0u8; ZSTD_compressBound(src.len()).max(64)];
    let n2 = ZSTD_compress2(&mut cctx, &mut dst, src);
    assert!(!ERR_isError(n2), "rust err 0x{:x}", n2);
    assert!(
        ZSTD_rowMatchFinderUsed(
            cctx.appliedParams.cParams.strategy,
            cctx.appliedParams.useRowMatchFinder,
        ),
        "row match finder not active: strategy={} windowLog={} useRowMatchFinder={:?}",
        cctx.appliedParams.cParams.strategy,
        cctx.appliedParams.cParams.windowLog,
        cctx.appliedParams.useRowMatchFinder
    );
    dst.truncate(n2);
    if src.len() == SENTINEL_N && wlog == SENTINEL_WLOG {
        let hash = XXH64(&dst, 0);
        assert_eq!(
            n2, SENTINEL_SIZE,
            "row-match upstream sentinel size mismatch"
        );
        assert_eq!(
            hash, SENTINEL_XXH64,
            "row-match upstream sentinel hash mismatch"
        );
        println!("row parity sentinel: size={n2} xxh64={hash:016x}");
    }
    std::fs::write("/tmp/rust_out.zst", &dst).unwrap();
    println!("rust output size: {}", n2);
}
