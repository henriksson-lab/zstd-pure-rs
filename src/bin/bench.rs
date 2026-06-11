use std::env;
use std::fs;
use std::time::Instant;
use zstd_pure_rs::common::error::{ERR_getErrorName, ERR_isError};
use zstd_pure_rs::compress::zstd_compress::{ZSTD_compress, ZSTD_compressBound};
use zstd_pure_rs::decompress::zstd_decompress::ZSTD_decompress;

fn main() {
    let path = env::args().nth(1).expect("path");
    let level: i32 = env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let iterations: usize = env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    if iterations == 0 {
        eprintln!("iterations must be greater than 0");
        return;
    }

    let payload = fs::read(&path).unwrap();
    eprintln!(
        "file={} size={} level={} iters={}",
        path,
        payload.len(),
        level,
        iterations
    );

    let mut compressed = vec![0u8; ZSTD_compressBound(payload.len()).max(64)];
    let mut decoded = vec![0u8; payload.len()];

    // Warm up + correctness check
    let n = ZSTD_compress(&mut compressed, &payload, level);
    if ERR_isError(n) {
        eprintln!("compress err: {}", ERR_getErrorName(n));
        return;
    }
    if let Ok(p) = env::var("WRITE_OUR_FRAME") {
        fs::write(&p, &compressed[..n]).unwrap();
        eprintln!("wrote our frame to {}", p);
    }
    let d = ZSTD_decompress(&mut decoded, &compressed[..n]);
    if ERR_isError(d) {
        eprintln!("decompress err: {}", ERR_getErrorName(d));
        return;
    }
    decoded.truncate(d);
    if decoded != payload {
        eprintln!("ROUNDTRIP MISMATCH");
        let first = decoded.iter().zip(payload.iter()).position(|(a, b)| a != b);
        eprintln!("first diff at: {:?}", first);
        return;
    }
    eprintln!("roundtrip OK");
    decoded.resize(payload.len(), 0);

    // Compress benchmark
    let start = Instant::now();
    let mut total_compressed = 0usize;
    for _ in 0..iterations {
        let n = ZSTD_compress(&mut compressed, &payload, level);
        if ERR_isError(n) {
            eprintln!("compress err: {}", ERR_getErrorName(n));
            return;
        }
        total_compressed = n;
    }
    if total_compressed == 0 {
        eprintln!("compress err: produced empty frame");
        return;
    }
    let elapsed = start.elapsed();
    let compress_speed = (payload.len() * iterations) as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    let ratio = payload.len() as f64 / total_compressed as f64;
    eprintln!(
        "compress L{}: {} -> {} ({:.2}x) | {:.1} MB/s",
        level,
        payload.len(),
        total_compressed,
        ratio,
        compress_speed
    );

    let d = ZSTD_decompress(&mut decoded, &compressed[..total_compressed]);
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

    // Decompress benchmark
    let start = Instant::now();
    let mut total_decoded = 0usize;
    for _ in 0..iterations {
        let d = ZSTD_decompress(&mut decoded, &compressed[..total_compressed]);
        if ERR_isError(d) {
            eprintln!("decompress err: {}", ERR_getErrorName(d));
            return;
        }
        if d != payload.len() {
            eprintln!(
                "decompress size mismatch: decoded {} bytes, expected {}",
                d,
                payload.len()
            );
            return;
        }
        total_decoded += d;
    }
    let elapsed = start.elapsed();
    if decoded != payload {
        let first = decoded.iter().zip(payload.iter()).position(|(a, b)| a != b);
        eprintln!("decompress mismatch after benchmark at: {:?}", first);
        return;
    }
    let decompress_speed = total_decoded as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    eprintln!("decompress: {:.1} MB/s", decompress_speed);
}
