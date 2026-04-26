// Force greedy + row-hash and dump output for various input sizes.
// Argv: [n=8192] [wlog=17]
use zstd_pure_rs::common::error::ERR_isError;
use zstd_pure_rs::compress::zstd_compress::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(8192);
    let wlog: i32 = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(17);
    let input =
        std::fs::read("/data/henriksson/github/claude/zstd-pure-rs/tests/fixtures/zstd_h.txt")
            .unwrap();
    let src = &input[..n.min(input.len())];
    let mut cctx = ZSTD_createCCtx().unwrap();
    let _ = ZSTD_CCtx_setParameter(
        &mut cctx,
        ZSTD_cParameter::ZSTD_c_strategy,
        zstd_pure_rs::compress::zstd_compress_sequences::ZSTD_greedy as i32,
    );
    let _ = ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_windowLog, wlog);
    let _ = ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_chainLog, 16);
    let _ = ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_hashLog, 17);
    let _ = ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_searchLog, 3);
    let _ = ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_minMatch, 4);
    let _ = ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_targetLength, 2);
    let _ = ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_contentSizeFlag, 1);
    let _ = ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_checksumFlag, 0);
    let mut dst = vec![0u8; src.len() + 64];
    let n2 = ZSTD_compress2(&mut cctx, &mut dst, src);
    assert!(!ERR_isError(n2), "rust err 0x{:x}", n2);
    dst.truncate(n2);
    std::fs::write("/tmp/rust_out.zst", &dst).unwrap();
    println!("rust output size: {}", n2);
}
