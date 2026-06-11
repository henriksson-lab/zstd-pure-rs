use std::env;
use std::fs;
use std::time::Instant;
use zstd_pure_rs::common::error::{ERR_getErrorName, ERR_isError};
use zstd_pure_rs::common::xxhash::{XXH64_state_t, XXH64};
use zstd_pure_rs::compress::zstd_compress::{ZSTD_compress, ZSTD_compressBound};
use zstd_pure_rs::decompress::zstd_decompress::ZSTD_decompressDCtx;
use zstd_pure_rs::decompress::zstd_decompress_block::{
    ZSTD_DCtx, ZSTD_buildDefaultSeqTables, ZSTD_decoder_entropy_rep,
};

fn main() {
    let path = env::args().nth(1).expect("path");
    let level: i32 = env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let iterations: usize = env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    if iterations == 0 {
        eprintln!("iterations must be greater than 0");
        return;
    }

    let payload = fs::read(&path).unwrap();
    let mut compressed = vec![0u8; ZSTD_compressBound(payload.len()).max(64)];
    let mut decoded = vec![0u8; payload.len()];
    let mut dctx = ZSTD_DCtx::new();
    ZSTD_buildDefaultSeqTables(&mut dctx);
    let mut entropy_rep = ZSTD_decoder_entropy_rep::default();
    let mut xxh = XXH64_state_t::default();
    let n = ZSTD_compress(&mut compressed, &payload, level);
    if ERR_isError(n) {
        eprintln!("err: {}", ERR_getErrorName(n));
        return;
    }

    // Warm up
    let d = ZSTD_decompressDCtx(
        &mut dctx,
        &mut entropy_rep,
        &mut xxh,
        &mut decoded,
        &compressed[..n],
    );
    if ERR_isError(d) {
        eprintln!("decompress err: {}", ERR_getErrorName(d));
        return;
    }
    if d != payload.len() || decoded[..d] != payload {
        eprintln!(
            "decompress mismatch: decoded {} bytes, expected {}",
            d,
            payload.len()
        );
        return;
    }

    let start = Instant::now();
    let mut total_decoded = 0usize;
    let mut last_decoded = d;
    for _ in 0..iterations {
        let d = ZSTD_decompressDCtx(
            &mut dctx,
            &mut entropy_rep,
            &mut xxh,
            &mut decoded,
            &compressed[..n],
        );
        if ERR_isError(d) {
            eprintln!("decompress err: {}", ERR_getErrorName(d));
            return;
        }
        total_decoded += d;
        last_decoded = d;
    }
    let elapsed = start.elapsed();

    if last_decoded != payload.len() {
        eprintln!(
            "decompress size mismatch: decoded {} bytes, expected {}",
            last_decoded,
            payload.len()
        );
        return;
    }
    let decoded = &decoded[..last_decoded];
    let expected_checksum = XXH64(&payload, 0);
    let decoded_checksum = XXH64(decoded, 0);
    if decoded_checksum != expected_checksum {
        let first = decoded.iter().zip(payload.iter()).position(|(a, b)| a != b);
        eprintln!(
            "decompress checksum mismatch: decoded {decoded_checksum:016x}, expected {expected_checksum:016x}, first diff at {first:?}"
        );
        return;
    }
    let speed = total_decoded as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    eprintln!("decompress L{}: {:.1} MB/s", level, speed);
}
