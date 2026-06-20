//! Dictionary-trainer benchmark harness (behind the `cli` feature).
//! usage: train <default|legacy|cover|fastcover> <maxDict> <out.dict> <file...>
use std::time::Instant;
use zstd_pure_rs::prelude::*;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        eprintln!("usage: train <mode> <maxDict> <out> <file...>");
        std::process::exit(2);
    }
    let mode = a[1].clone();
    let maxdict: usize = a[2].parse().unwrap();
    let out = a[3].clone();
    let mut samples = Vec::new();
    let mut sizes = Vec::new();
    for f in &a[4..] {
        let d = std::fs::read(f).unwrap();
        sizes.push(d.len());
        samples.extend_from_slice(&d);
    }
    let nb = sizes.len() as u32;
    let mut dict = vec![0u8; maxdict];
    let t = Instant::now();
    let n = match mode.as_str() {
        "default" => ZDICT_trainFromBuffer(&mut dict, maxdict, &samples, &sizes, nb),
        "legacy" => ZDICT_trainFromBuffer_legacy(
            &mut dict,
            maxdict,
            &samples,
            &sizes,
            nb,
            ZDICT_legacy_params_t::default(),
        ),
        "cover" => {
            let mut p = ZDICT_cover_params_t::default();
            p.k = 512;
            p.d = 8;
            p.steps = 4;
            ZDICT_optimizeTrainFromBuffer_cover(&mut dict, maxdict, &samples, &sizes, nb, &mut p)
        }
        "fastcover" => {
            let mut p = ZDICT_fastCover_params_t::default();
            p.k = 512;
            p.d = 8;
            p.steps = 4;
            ZDICT_optimizeTrainFromBuffer_fastCover(
                &mut dict, maxdict, &samples, &sizes, nb, &mut p,
            )
        }
        _ => {
            eprintln!("bad mode {mode}");
            std::process::exit(2);
        }
    };
    eprintln!("train_elapsed_ms={}", t.elapsed().as_millis());
    if ERR_isError(n) {
        eprintln!("train error {n:#x}");
        std::process::exit(1);
    }
    dict.truncate(n);
    std::fs::write(&out, &dict).unwrap();
}
