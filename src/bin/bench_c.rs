use std::env;
use std::fs;
use std::time::Instant;
use zstd_pure_rs::common::error::{ERR_getErrorName, ERR_isError};
use zstd_pure_rs::compress::zstd_compress::{ZSTD_compress, ZSTD_compressBound};

fn main() {
    let path = env::args().nth(1).expect("path");
    let level: i32 = env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let iterations: usize = env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(3);

    let payload = fs::read(&path).unwrap();
    let mut compressed = vec![0u8; ZSTD_compressBound(payload.len()).max(64)];
    let n = ZSTD_compress(&mut compressed, &payload, level);
    if ERR_isError(n) {
        eprintln!("err: {}", ERR_getErrorName(n));
        return;
    }

    let start = Instant::now();
    for _ in 0..iterations {
        ZSTD_compress(&mut compressed, &payload, level);
    }
    let elapsed = start.elapsed();
    let speed = (payload.len() * iterations) as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    eprintln!("compress L{}: {:.1} MB/s", level, speed);
}
