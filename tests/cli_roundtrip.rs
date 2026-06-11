//! Integration test: shell out to our own `zstd` binary and verify it
//! decompresses a hand-crafted raw-block `.zst` frame back to the
//! original payload.
//!
//! Only runs when the `cli` feature is enabled (the binary is guarded
//! by `required-features = ["cli"]` in Cargo.toml).

#![cfg(feature = "cli")]

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

const ZSTD_MAGIC: u32 = 0xFD2FB528;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn make_raw_frame(payload: &[u8]) -> Vec<u8> {
    let mut src = Vec::new();
    src.extend_from_slice(&ZSTD_MAGIC.to_le_bytes());
    src.push(0x20); // FHD: single-segment, fcsID=0 → 1-byte FCS
    src.push(payload.len() as u8);
    // Block header: lastBlock=1, bt_raw=0, cSize=payload.len().
    let bh = ((payload.len() as u32) << 3) | 1;
    src.push((bh & 0xFF) as u8);
    src.push(((bh >> 8) & 0xFF) as u8);
    src.push(((bh >> 16) & 0xFF) as u8);
    src.extend_from_slice(payload);
    src
}

fn bin_path() -> PathBuf {
    // Cargo sets CARGO_BIN_EXE_<name> for integration tests to point
    // at the compiled binary.
    PathBuf::from(env!("CARGO_BIN_EXE_zstd"))
}

fn temp_path(label: &str, suffix: &str) -> PathBuf {
    let n = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "zstd_pure_{label}_{}_{}_{suffix}",
        std::process::id(),
        n
    ))
}

fn decode_with_cli(args: &[&str], compressed: &[u8], expected: &[u8], label: &str) {
    let mut child = Command::new(bin_path())
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    {
        let mut stdin = child.stdin.take().unwrap();
        stdin.write_all(compressed).unwrap();
    }
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "{label}: our CLI decode failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.stdout, expected, "{label}: our CLI decoded mismatch");
}

#[test]
fn cli_decompresses_raw_block_frame_via_stdin() {
    let payload = b"HELLO FROM CLI TEST";
    let frame = make_raw_frame(payload);

    let mut child = Command::new(bin_path())
        .args(["-d", "-c", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn zstd");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(&frame)
        .expect("write frame");
    let out = child.wait_with_output().expect("wait");
    assert!(
        out.status.success(),
        "exit={:?} stderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.stdout, payload);
}

#[test]
fn cli_decompresses_frame_with_absent_fcs() {
    // Craft a non-single-segment frame: FHD=0x00 (fcsID=0,
    // singleSegment=0) → the FCS field is absent. The CLI's
    // decompress path falls back to `ZSTD_findFrameSizeInfo`
    // nbBlocks×blockSizeMax bound rather than the declared FCS.
    //
    // Build: magic + FHD(0x00) + wlByte(0x40 → windowLog=18) +
    //        block header: cSize=5, bt_raw=0, lastBlock=1 → (5<<3)|1.
    //        + 5 raw bytes.
    let mut frame = Vec::new();
    frame.extend_from_slice(&ZSTD_MAGIC.to_le_bytes());
    frame.push(0x00);
    frame.push(0x40);
    let bh = (5u32 << 3) | 1;
    frame.push((bh & 0xFF) as u8);
    frame.push(((bh >> 8) & 0xFF) as u8);
    frame.push(((bh >> 16) & 0xFF) as u8);
    frame.extend_from_slice(b"HELLO");

    let mut child = Command::new(bin_path())
        .args(["-d", "-c", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn zstd");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(&frame)
        .expect("write frame");
    let out = child.wait_with_output().expect("wait");
    assert!(
        out.status.success(),
        "exit={:?} stderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.stdout, b"HELLO");
}

/// Returns Some(path) if the vendored upstream `zstd` CLI has been built.
fn upstream_zstd() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("zstd/programs/zstd");
    p.is_file().then_some(p)
}

fn run_with_stdin(bin: &std::path::Path, args: &[&str], input: &[u8]) -> std::process::Output {
    let mut child = Command::new(bin)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.as_mut().unwrap().write_all(input).unwrap();
    child.wait_with_output().unwrap()
}

fn external_tool_available(command: &str) -> bool {
    Command::new(command)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

#[test]
fn cli_output_decompressed_by_upstream_zstd() {
    // Produce a frame with our compressor, then pipe it into the
    // vendored upstream zstd binary (if available) and assert the round
    // trip. This is the strongest spec-compliance check we can run
    // without a full reference implementation in-tree.
    let Some(upstream) = upstream_zstd() else {
        eprintln!("vendored upstream zstd/programs/zstd not built; skipping");
        return;
    };

    let payload: Vec<u8> = b"lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        .iter()
        .cycle()
        .take(3000)
        .copied()
        .collect();

    // Compress with our binary.
    let mut comp = Command::new(bin_path())
        .args(["-c", "-q", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn our compressor");
    comp.stdin.as_mut().unwrap().write_all(&payload).unwrap();
    let comp_out = comp.wait_with_output().expect("wait compressor");
    assert!(
        comp_out.status.success(),
        "our compress stderr: {}",
        String::from_utf8_lossy(&comp_out.stderr)
    );
    let compressed = comp_out.stdout;

    // Decompress via upstream zstd.
    let mut dec = Command::new(&upstream)
        .args(["-d", "-c", "-q", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn upstream zstd");
    dec.stdin.as_mut().unwrap().write_all(&compressed).unwrap();
    let dec_out = dec.wait_with_output().expect("wait upstream");
    assert!(
        dec_out.status.success(),
        "upstream zstd rejected our output: stderr={}",
        String::from_utf8_lossy(&dec_out.stderr)
    );
    assert_eq!(dec_out.stdout, payload, "upstream decoded != original");
    decode_with_cli(
        &["-d", "-c", "-q", "-"],
        &compressed,
        &payload,
        "our compressed frame",
    );
}

#[test]
fn cli_decompresses_upstream_output() {
    // Complementary direction: upstream compresses, we decompress.
    let Some(upstream) = upstream_zstd() else {
        eprintln!("vendored upstream zstd/programs/zstd not built; skipping");
        return;
    };

    let payload: Vec<u8> = b"Mary had a little lamb, its fleece was white as snow. "
        .iter()
        .cycle()
        .take(300_000)
        .copied()
        .collect();

    // Compress via upstream zstd at -1.
    let mut comp = Command::new(&upstream)
        .args(["-c", "-q", "-1", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn upstream compressor");
    comp.stdin.as_mut().unwrap().write_all(&payload).unwrap();
    let comp_out = comp.wait_with_output().expect("wait upstream compressor");
    assert!(comp_out.status.success());
    let compressed = comp_out.stdout;

    // Decompress via our binary.
    let mut dec = Command::new(bin_path())
        .args(["-d", "-c", "-q", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn our decompressor");
    dec.stdin.as_mut().unwrap().write_all(&compressed).unwrap();
    let dec_out = dec.wait_with_output().expect("wait our dec");
    assert!(
        dec_out.status.success(),
        "our decode of upstream output stderr={}",
        String::from_utf8_lossy(&dec_out.stderr)
    );
    assert_eq!(dec_out.stdout, payload);

    let compressed_path = temp_path("upstream_file_input", "zst");
    fs::write(&compressed_path, &compressed).expect("write upstream frame");
    let file_dec = Command::new(bin_path())
        .args(["-d", "-c", "-q"])
        .arg(&compressed_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn our file decompressor");
    let _ = fs::remove_file(&compressed_path);
    assert!(
        file_dec.status.success(),
        "our file decode of upstream output stderr={}",
        String::from_utf8_lossy(&file_dec.stderr)
    );
    assert_eq!(file_dec.stdout, payload);
}

#[test]
fn cli_compression_ratio_within_reasonable_factor_of_upstream() {
    // Sanity check: our level-1 ratio shouldn't be hugely worse than
    // upstream's level-1 on the same input. 2.0x slack absorbs the
    // missing optimizations (true lazy lookahead tuning, rowHash,
    // btopt) while catching gross regressions.
    let Some(upstream) = upstream_zstd() else {
        eprintln!("vendored upstream zstd/programs/zstd not built; skipping");
        return;
    };
    let payload = std::fs::read("tests/fixtures/zstd_h.txt").expect("read fixture");

    fn compress_via(bin: &std::path::Path, payload: &[u8], level: i32) -> Vec<u8> {
        let level_flag = format!("-{level}");
        let args = ["-c", "-q", &level_flag, "-"];
        let mut comp = Command::new(bin)
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        // Write stdin from a background thread so the parent can
        // drain stdout concurrently — avoids the classic
        // stdin-full/stdout-full mutual-deadlock when payload + output
        // both exceed the pipe buffer (Linux ~64 KB).
        let mut stdin = comp.stdin.take().unwrap();
        let payload_owned = payload.to_vec();
        let writer = std::thread::spawn(move || {
            let _ = stdin.write_all(&payload_owned);
            // Explicit drop closes stdin → signals EOF to the child.
            drop(stdin);
        });
        let out = comp.wait_with_output().unwrap();
        writer.join().unwrap();
        assert!(
            out.status.success(),
            "compress failed: stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
        out.stdout
    }

    for level in [1, 3, 5, 7, 10, 15, 19, 22] {
        let ours = compress_via(&bin_path(), &payload, level);
        let theirs = compress_via(&upstream, &payload, level);
        let slack = (ours.len() as f64) / (theirs.len() as f64);
        eprintln!(
            "level {level:>2}: ours={:>6} theirs={:>6} ratio={:.2}x",
            ours.len(),
            theirs.len(),
            slack
        );
        // After the step-reset fix we're within ~1.2x of upstream on
        // all levels we have real strategies for. Leave headroom at
        // 1.5x to absorb the remaining gap at btopt+ (which we
        // currently cap down to lazy2).
        assert!(
            slack < 1.5,
            "level {level} compression {}x worse than upstream ({} vs {})",
            slack,
            ours.len(),
            theirs.len()
        );
    }
}

#[test]
fn cli_multiblock_frame_upstream_cross_compat() {
    // 182 KB real-world C header file — forces a multi-block frame
    // (>128 KB block boundary) and exercises the entropy-state
    // carry-over path.
    let Some(upstream) = upstream_zstd() else {
        eprintln!("vendored upstream zstd/programs/zstd not built; skipping");
        return;
    };
    let payload = std::fs::read("tests/fixtures/zstd_h.txt").expect("read fixture");
    assert!(payload.len() > 128 * 1024);

    let mut comp = Command::new(bin_path())
        .args(["-c", "-q", "-3", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    comp.stdin.as_mut().unwrap().write_all(&payload).unwrap();
    let comp_out = comp.wait_with_output().unwrap();
    assert!(
        comp_out.status.success(),
        "our compress stderr: {}",
        String::from_utf8_lossy(&comp_out.stderr)
    );
    decode_with_cli(
        &["-d", "-c", "-q", "-"],
        &comp_out.stdout,
        &payload,
        "our multi-block frame",
    );

    let mut dec = Command::new(&upstream)
        .args(["-d", "-c", "-q", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    dec.stdin
        .as_mut()
        .unwrap()
        .write_all(&comp_out.stdout)
        .unwrap();
    let dec_out = dec.wait_with_output().unwrap();
    assert!(
        dec_out.status.success(),
        "upstream rejected multi-block output: stderr={}",
        String::from_utf8_lossy(&dec_out.stderr)
    );
    assert_eq!(dec_out.stdout, payload, "multi-block roundtrip mismatch");
}

#[test]
fn cli_output_decompressed_by_upstream_zstd_all_levels() {
    // Sweep every positive level 1..=22 on a single varied payload.
    // Gates every strategy clamp + getCParams row → upstream spec
    // compliance across the entire public level range.
    let Some(upstream) = upstream_zstd() else {
        eprintln!("vendored upstream zstd/programs/zstd not built; skipping");
        return;
    };
    let payload: Vec<u8> = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        .iter()
        .cycle()
        .take(4000)
        .copied()
        .collect();
    for level in 1i32..=22 {
        let mut comp = Command::new(bin_path())
            .args(["-c", "-q", &format!("-{level}"), "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let _ = comp.stdin.as_mut().unwrap().write_all(&payload);
        let comp_out = comp.wait_with_output().unwrap();
        assert!(
            comp_out.status.success(),
            "[level {level}] our compress stderr: {}",
            String::from_utf8_lossy(&comp_out.stderr)
        );

        let mut dec = Command::new(&upstream)
            .args(["-d", "-c", "-q", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        dec.stdin
            .as_mut()
            .unwrap()
            .write_all(&comp_out.stdout)
            .unwrap();
        let dec_out = dec.wait_with_output().unwrap();
        assert!(
            dec_out.status.success(),
            "[level {level}] upstream rejected our output: stderr={}",
            String::from_utf8_lossy(&dec_out.stderr)
        );
        assert_eq!(&dec_out.stdout, &payload, "[level {level}] mismatch");
        decode_with_cli(
            &["-d", "-c", "-q", "-"],
            &comp_out.stdout,
            &payload,
            &format!("our all-level frame level {level}"),
        );
    }
}

#[test]
fn cli_output_decompressed_by_upstream_zstd_across_levels() {
    let Some(upstream) = upstream_zstd() else {
        eprintln!("vendored upstream zstd/programs/zstd not built; skipping");
        return;
    };
    // Mix of patterns to stress different paths.
    let inputs: Vec<Vec<u8>> = vec![
        // Highly repetitive — exercises repcode.
        b"abc".iter().cycle().take(1024).copied().collect(),
        // Natural text — exercises varied sequences.
        b"The quick brown fox jumps over the lazy dog. "
            .iter()
            .cycle()
            .take(5000)
            .copied()
            .collect(),
        // All zeros → late-RLE downgrade path.
        vec![0u8; 2048],
        // Sparse with rare symbol bursts.
        {
            let mut v = vec![b'A'; 2000];
            for i in (0..v.len()).step_by(37) {
                v[i] = b'Z';
            }
            v
        },
    ];
    for (i, payload) in inputs.iter().enumerate() {
        for &level in &[1i32, 3, 10, 19] {
            let mut comp = Command::new(bin_path())
                .args(["-c", "-q", &format!("-{level}"), "-"])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            comp.stdin.as_mut().unwrap().write_all(payload).unwrap();
            let comp_out = comp.wait_with_output().unwrap();
            assert!(
                comp_out.status.success(),
                "[case {i}, level {level}] our compress stderr: {}",
                String::from_utf8_lossy(&comp_out.stderr)
            );

            let mut dec = Command::new(&upstream)
                .args(["-d", "-c", "-q", "-"])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            dec.stdin
                .as_mut()
                .unwrap()
                .write_all(&comp_out.stdout)
                .unwrap();
            let dec_out = dec.wait_with_output().unwrap();
            assert!(
                dec_out.status.success(),
                "[case {i}, level {level}] upstream rejected our output: stderr={}",
                String::from_utf8_lossy(&dec_out.stderr)
            );
            assert_eq!(
                &dec_out.stdout, payload,
                "[case {i}, level {level}] roundtrip mismatch"
            );
            decode_with_cli(
                &["-d", "-c", "-q", "-"],
                &comp_out.stdout,
                payload,
                &format!("our varied frame case {i} level {level}"),
            );
        }
    }
}

#[test]
fn cli_dict_output_decompressed_by_upstream() {
    // Our dict-compressed frame should decode correctly through
    // upstream `zstd -d -D dict` when they share the same dict.
    use std::fs;
    let Some(upstream) = upstream_zstd() else {
        eprintln!("vendored upstream zstd/programs/zstd not built; skipping");
        return;
    };

    let dict_path = temp_path("xdict", "bin");
    let dict = b"the quick brown fox jumps over the lazy dog near a river. ".repeat(40);
    fs::write(&dict_path, &dict).expect("write dict");

    let payload: Vec<u8> = b"the fox jumps near the river. the lazy dog. "
        .iter()
        .cycle()
        .take(600)
        .copied()
        .collect();

    let mut comp = Command::new(bin_path())
        .args(["-c", "-q", "-D"])
        .arg(&dict_path)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    comp.stdin.as_mut().unwrap().write_all(&payload).unwrap();
    let comp_out = comp.wait_with_output().unwrap();
    assert!(
        comp_out.status.success(),
        "our dict compress stderr: {}",
        String::from_utf8_lossy(&comp_out.stderr)
    );

    let mut dec = Command::new(&upstream)
        .args(["-d", "-c", "-q", "-D"])
        .arg(&dict_path)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    dec.stdin
        .as_mut()
        .unwrap()
        .write_all(&comp_out.stdout)
        .unwrap();
    let dec_out = dec.wait_with_output().unwrap();
    assert!(
        dec_out.status.success(),
        "upstream rejected our dict-compressed output: stderr={}",
        String::from_utf8_lossy(&dec_out.stderr)
    );
    assert_eq!(dec_out.stdout, payload);
    decode_with_cli(
        &["-d", "-c", "-q", "-D", dict_path.to_str().unwrap(), "-"],
        &comp_out.stdout,
        &payload,
        "our dict-compressed frame",
    );

    let _ = fs::remove_file(&dict_path);
}

#[test]
fn cli_fuzz_random_payloads_all_decode_via_upstream() {
    // Deterministic PRNG → random payloads of varying sizes; each
    // compressed via our CLI, then decompressed via upstream zstd,
    // with byte-exact verification.
    let Some(upstream) = upstream_zstd() else {
        eprintln!("vendored upstream zstd/programs/zstd not built; skipping");
        return;
    };
    fn xorshift(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }
    let mut state: u64 = 0x0123456789abcdef;
    for seed_iter in 0..8 {
        let size = (xorshift(&mut state) as usize) % 10_000 + 1;
        let mut payload = vec![0u8; size];
        for b in &mut payload {
            *b = (xorshift(&mut state) & 0xFF) as u8;
        }
        let level = (xorshift(&mut state) % 7 + 1) as i32;

        let mut comp = Command::new(bin_path())
            .args(["-c", "-q", &format!("-{level}"), "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let _ = comp.stdin.as_mut().unwrap().write_all(&payload);
        let comp_out = comp.wait_with_output().unwrap();
        assert!(
            comp_out.status.success(),
            "[iter {seed_iter} size {size} level {level}] our compress stderr: {}",
            String::from_utf8_lossy(&comp_out.stderr)
        );

        let mut dec = Command::new(&upstream)
            .args(["-d", "-c", "-q", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        dec.stdin
            .as_mut()
            .unwrap()
            .write_all(&comp_out.stdout)
            .unwrap();
        let dec_out = dec.wait_with_output().unwrap();
        assert!(
            dec_out.status.success(),
            "[iter {seed_iter} size {size} level {level}] upstream rejected: stderr={}",
            String::from_utf8_lossy(&dec_out.stderr)
        );
        assert_eq!(
            &dec_out.stdout, &payload,
            "[iter {seed_iter} size {size} level {level}] mismatch"
        );
        decode_with_cli(
            &["-d", "-c", "-q", "-"],
            &comp_out.stdout,
            &payload,
            &format!("our fuzz frame iter {seed_iter} size {size} level {level}"),
        );
    }
}

#[test]
fn cli_file_to_file_compress_then_decompress_roundtrip() {
    use std::fs;
    let input_path = temp_path("f2f_in", "txt");
    let compressed_path = temp_path("f2f", "zst");
    let decompressed_path = temp_path("f2f_out", "txt");

    let payload: Vec<u8> = b"file-to-file testing. "
        .iter()
        .cycle()
        .take(1500)
        .copied()
        .collect();
    fs::write(&input_path, &payload).expect("write input");

    // Compress via `zstd -q -o compressed_path input_path`.
    let comp = Command::new(bin_path())
        .args(["-q", "-f", "-o"])
        .arg(&compressed_path)
        .arg(&input_path)
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        comp.status.success(),
        "compress stderr: {}",
        String::from_utf8_lossy(&comp.stderr)
    );
    assert!(compressed_path.exists());

    // Decompress via `zstd -d -q -o decompressed_path compressed_path`.
    let dec = Command::new(bin_path())
        .args(["-d", "-q", "-f", "-o"])
        .arg(&decompressed_path)
        .arg(&compressed_path)
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        dec.status.success(),
        "decompress stderr: {}",
        String::from_utf8_lossy(&dec.stderr)
    );
    let recovered = fs::read(&decompressed_path).expect("read decompressed");
    assert_eq!(recovered, payload);

    let _ = fs::remove_file(&input_path);
    let _ = fs::remove_file(&compressed_path);
    let _ = fs::remove_file(&decompressed_path);
}

#[test]
fn cli_decompress_file_with_unknown_suffix_requires_explicit_output() {
    let tmp = std::env::temp_dir().join(format!("zstd_pure_unknown_suffix_{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let suffixed = tmp.join("payload.zst");
    let no_suffix = tmp.join("payload-copy");
    fs::write(&suffixed, make_raw_frame(b"unknown suffix payload")).unwrap();
    fs::copy(&suffixed, &no_suffix).unwrap();

    let out = Command::new(bin_path())
        .args(["-d", "-q"])
        .arg(&no_suffix)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();

    assert!(
        !out.status.success(),
        "unknown-suffix decompression should fail"
    );
    assert!(
        out.stdout.is_empty(),
        "unknown-suffix decompression must not write stdout"
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("unknown suffix"),
        "unknown-suffix decompression reported the wrong error: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let entry_count = fs::read_dir(&tmp).unwrap().count();
    assert_eq!(
        entry_count, 2,
        "unknown-suffix decompression created an unexpected output file"
    );

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cli_decompress_malformed_file_with_unknown_suffix_rejects_before_decode() {
    let tmp = std::env::temp_dir().join(format!(
        "zstd_pure_unknown_suffix_malformed_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let no_suffix = tmp.join("payload-copy");
    fs::write(&no_suffix, b"not a zstd frame").unwrap();

    let out = Command::new(bin_path())
        .args(["-d", "-q"])
        .arg(&no_suffix)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "malformed no-suffix input should fail"
    );
    assert!(
        stderr.contains("unknown suffix"),
        "expected suffix diagnostic, got: {stderr}"
    );
    assert!(
        !stderr.contains("frame walk failed") && !stderr.contains("Src size"),
        "no-suffix rejection should happen before decode, got: {stderr}"
    );
    assert!(out.stdout.is_empty(), "rejection must not write stdout");

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cli_help_advertises_compression_support() {
    let out = Command::new(bin_path()).arg("--help").output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("compression") || stdout.contains("compress"),
        "--help output didn't mention compression: {stdout}"
    );
}

#[test]
fn cli_help_mentions_every_supported_flag() {
    // --help must surface every documented flag so users can
    // discover the API without reading source. Regression gate
    // against an accidental clap attribute removal.
    let out = Command::new(bin_path()).arg("--help").output().unwrap();
    assert!(out.status.success());
    let help = String::from_utf8_lossy(&out.stdout);
    for needle in [
        "--decompress",
        "--stdout",
        "--force",
        "--pass-through",
        "--no-pass-through",
        "--quiet",
        "--verbose",
        "--check",
        "--no-check",
        "--magicless",
        "--ultra",
    ] {
        assert!(help.contains(needle), "--help missed {needle}: {help}");
    }
    // Short flags too.
    for needle in ["-d", "-c", "-f", "-q", "-v", "-o", "-D", "-C"] {
        assert!(
            help.contains(needle),
            "--help missed short flag {needle}: {help}"
        );
    }
    assert!(
        !help.contains("--level")
            && !help.contains("-L")
            && !help.contains("--output-file")
            && !help.contains("--dict"),
        "upstream-incompatible aliases leaked into help: {help}"
    );
}

#[test]
fn cli_version_flag_prints_version() {
    let out = Command::new(bin_path()).arg("--version").output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(!stdout.is_empty());
}

#[test]
fn cli_refuses_to_overwrite_existing_output_without_force() {
    // Safety gate: the CLI must refuse to overwrite an existing file
    // when `-f/--force` is NOT passed. Silent clobbering would be a
    // data-loss footgun.
    use std::fs;
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let input_path = tmp.join(format!("zstd_pure_overwrite_in_{pid}.txt"));
    let existing_out = tmp.join(format!("zstd_pure_overwrite_out_{pid}.zst"));

    let payload = b"overwrite-guard payload".repeat(30);
    fs::write(&input_path, &payload).unwrap();
    // Pre-create the output path with sentinel content so we can
    // verify it wasn't touched.
    let sentinel = b"DO NOT OVERWRITE THIS FILE";
    fs::write(&existing_out, sentinel).unwrap();

    let out = Command::new(bin_path())
        .args(["-q", "-o"])
        .arg(&existing_out)
        .arg(&input_path)
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "compress should have failed due to pre-existing output, stderr: {}",
        String::from_utf8_lossy(&out.stderr),
    );
    // Sentinel content must be preserved.
    assert_eq!(fs::read(&existing_out).unwrap(), sentinel);

    // With `-f`, the same invocation should succeed and overwrite.
    let out2 = Command::new(bin_path())
        .args(["-q", "-f", "-o"])
        .arg(&existing_out)
        .arg(&input_path)
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(out2.status.success(), "with -f should succeed");
    assert_ne!(fs::read(&existing_out).unwrap(), sentinel);

    let _ = fs::remove_file(&input_path);
    let _ = fs::remove_file(&existing_out);
}

#[test]
fn cli_refuses_same_input_and_output_even_with_force() {
    // Upstream treats an output path that is the same file as the
    // input as a hard error, distinct from ordinary overwrite
    // prompting or -f clobbering.
    let tmp = temp_path("same_input_output", "dir");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let input_path = tmp.join("payload.txt");
    let payload = b"same input output payload".repeat(8);
    fs::write(&input_path, &payload).unwrap();

    let comp = Command::new(bin_path())
        .args(["-q", "-f", "-o"])
        .arg(&input_path)
        .arg(&input_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        !comp.status.success(),
        "same input/output compression should fail even with -f",
    );
    assert!(
        comp.stdout.is_empty(),
        "same input/output compression must not write stdout",
    );
    assert_eq!(
        fs::read(&input_path).unwrap(),
        payload,
        "same input/output compression modified the input",
    );
    assert!(
        String::from_utf8_lossy(&comp.stderr).contains("overwrite the input file"),
        "same input/output compression emitted unexpected stderr: {}",
        String::from_utf8_lossy(&comp.stderr),
    );

    let compressed_path = tmp.join("payload.zst");
    let setup = Command::new(bin_path())
        .args(["-q", "-f", "-o"])
        .arg(&compressed_path)
        .arg(&input_path)
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        setup.status.success(),
        "setup compression failed: {}",
        String::from_utf8_lossy(&setup.stderr)
    );
    let compressed = fs::read(&compressed_path).unwrap();

    let dec = Command::new(bin_path())
        .args(["-d", "-q", "-f", "-o"])
        .arg(&compressed_path)
        .arg(&compressed_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        !dec.status.success(),
        "same input/output decompression should fail even with -f",
    );
    assert!(
        dec.stdout.is_empty(),
        "same input/output decompression must not write stdout",
    );
    assert_eq!(
        fs::read(&compressed_path).unwrap(),
        compressed,
        "same input/output decompression modified the input",
    );
    assert!(
        String::from_utf8_lossy(&dec.stderr).contains("overwrite the input file"),
        "same input/output decompression emitted unexpected stderr: {}",
        String::from_utf8_lossy(&dec.stderr),
    );

    if let Some(upstream) = upstream_zstd() {
        let upstream_input = tmp.join("upstream.txt");
        fs::write(&upstream_input, &payload).unwrap();
        let upstream_out = Command::new(upstream)
            .args(["-q", "-f", "-o"])
            .arg(&upstream_input)
            .arg(&upstream_input)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(
            !upstream_out.status.success(),
            "vendored upstream unexpectedly allowed same input/output",
        );
        assert_eq!(
            upstream_out.stdout, comp.stdout,
            "vendored upstream stdout differs for same input/output rejection",
        );
        assert!(
            String::from_utf8_lossy(&upstream_out.stderr).contains("overwrite the input file"),
            "vendored upstream same input/output diagnostic changed: {}",
            String::from_utf8_lossy(&upstream_out.stderr),
        );
        assert_eq!(
            fs::read(&upstream_input).unwrap(),
            payload,
            "vendored upstream modified same input/output source",
        );
    }

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cli_existing_output_prompt_accepts_and_overwrites() {
    // Upstream prompts before clobbering an existing output path in
    // normal display mode, and accepts confirmation when the first
    // response byte is y/Y.
    use std::fs;
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let payload = b"overwrite prompt accepted payload".repeat(12);
    let sentinel = b"existing content before prompt";

    for (label, answer) in [
        ("lowercase", b"y\n" as &[u8]),
        ("uppercase", b"Y\n"),
        ("first-byte", b"yn\n"),
    ] {
        let input_path = tmp.join(format!(
            "zstd_pure_overwrite_prompt_yes_{label}_in_{pid}.txt"
        ));
        let existing_out = tmp.join(format!(
            "zstd_pure_overwrite_prompt_yes_{label}_out_{pid}.zst"
        ));

        fs::write(&input_path, &payload).unwrap();
        fs::write(&existing_out, sentinel).unwrap();

        let mut child = Command::new(bin_path())
            .args(["-o"])
            .arg(&existing_out)
            .arg(&input_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.as_mut().unwrap().write_all(answer).unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "{label} confirmed overwrite should succeed, stderr: {}",
            String::from_utf8_lossy(&out.stderr),
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("already exists;") && stderr.contains("overwrite (y/n) ?"),
            "{label} confirmed overwrite did not prompt like upstream: {stderr}",
        );
        assert_ne!(
            fs::read(&existing_out).unwrap(),
            sentinel,
            "{label} confirmed overwrite left sentinel content unchanged",
        );
        assert!(
            out.stdout.is_empty(),
            "{label} confirmed overwrite should write the explicit file, not stdout",
        );

        let decoded = Command::new(bin_path())
            .args(["-d", "-q", "-c"])
            .arg(&existing_out)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(
            decoded.status.success(),
            "{label} decode confirmed overwrite output failed: {}",
            String::from_utf8_lossy(&decoded.stderr),
        );
        assert_eq!(decoded.stdout, payload);

        let _ = fs::remove_file(&input_path);
        let _ = fs::remove_file(&existing_out);
    }
}

#[test]
fn cli_existing_output_prompt_decline_preserves_file() {
    // A non-y/Y answer follows upstream's "Not overwritten" path and
    // must leave the existing file byte-for-byte intact.
    use std::fs;
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let input_path = tmp.join(format!("zstd_pure_overwrite_prompt_no_in_{pid}.txt"));
    let existing_out = tmp.join(format!("zstd_pure_overwrite_prompt_no_out_{pid}.zst"));

    let payload = b"overwrite prompt declined payload".repeat(12);
    let sentinel = b"preserve this existing file";
    fs::write(&input_path, &payload).unwrap();
    fs::write(&existing_out, sentinel).unwrap();

    let mut child = Command::new(bin_path())
        .args(["-o"])
        .arg(&existing_out)
        .arg(&input_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.as_mut().unwrap().write_all(b"n\n").unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        !out.status.success(),
        "declined overwrite should fail without clobbering",
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("already exists;")
            && stderr.contains("overwrite (y/n) ?")
            && stderr.contains("Not overwritten"),
        "declined overwrite emitted unexpected stderr: {stderr}",
    );
    assert_eq!(
        fs::read(&existing_out).unwrap(),
        sentinel,
        "declined overwrite modified existing output",
    );
    assert!(
        out.stdout.is_empty(),
        "declined overwrite refusal must not write stdout",
    );

    let _ = fs::remove_file(&input_path);
    let _ = fs::remove_file(&existing_out);
}

#[test]
fn cli_existing_output_prompt_refuses_when_input_is_stdin() {
    // Upstream never asks for confirmation when stdin is also an
    // input; UTIL_requireUserConfirmation short-circuits with this
    // diagnostic and preserves the destination.
    use std::fs;
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let existing_out = tmp.join(format!("zstd_pure_overwrite_prompt_stdin_out_{pid}.zst"));
    let sentinel = b"existing stdin prompt output";
    fs::write(&existing_out, sentinel).unwrap();

    let mut child = Command::new(bin_path())
        .args(["-o"])
        .arg(&existing_out)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"y\npayload")
        .unwrap();
    let out = child.wait_with_output().unwrap();

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "stdin input overwrite prompt should refuse without reading confirmation",
    );
    assert!(
        stderr.contains("already exists;")
            && stderr.contains("stdin is an input - not proceeding."),
        "stdin input overwrite emitted unexpected stderr: {stderr}",
    );
    assert!(
        !stderr.contains("overwrite (y/n) ?"),
        "stdin input overwrite should not prompt interactively: {stderr}",
    );
    assert_eq!(
        fs::read(&existing_out).unwrap(),
        sentinel,
        "stdin input overwrite modified existing output",
    );
    assert!(
        out.stdout.is_empty(),
        "stdin input overwrite refusal must not emit compressed data",
    );

    let _ = fs::remove_file(&existing_out);
}

#[test]
fn cli_stdout_then_output_file_uses_last_output_directive() {
    use std::fs;
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let output_path = tmp.join(format!("zstd_pure_stdout_plus_o_{pid}.zst"));
    let _ = fs::remove_file(&output_path);
    let payload = b"single input -c -o writes to file, not stdout".repeat(8);

    let mut child = Command::new(bin_path())
        .args(["-c", "-q", "-o"])
        .arg(&output_path)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.as_mut().unwrap().write_all(&payload).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "single-input -c -o should succeed, stderr: {}",
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(
        out.stdout.is_empty(),
        "-c -o should write to file and leave stdout empty",
    );
    assert!(
        output_path.exists(),
        "-c -o should create {}",
        output_path.display(),
    );

    let dec = Command::new(bin_path())
        .args(["-d", "-c", "-q"])
        .arg(&output_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let dec_out = dec.wait_with_output().unwrap();
    assert!(
        dec_out.status.success(),
        "decompress stderr: {}",
        String::from_utf8_lossy(&dec_out.stderr),
    );
    assert_eq!(dec_out.stdout, payload);

    let _ = fs::remove_file(&output_path);
}

#[test]
fn cli_stdout_output_order_is_last_wins() {
    use std::fs;
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let output_path = tmp.join(format!("zstd_pure_o_before_c_{pid}.zst"));
    let _ = fs::remove_file(&output_path);
    let payload = b"single input -o file -c writes to stdout".repeat(8);

    let mut child = Command::new(bin_path())
        .args(["-q", "-o"])
        .arg(&output_path)
        .args(["-c", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.as_mut().unwrap().write_all(&payload).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "-o ... -c should succeed, stderr: {}",
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(
        !out.stdout.is_empty(),
        "-o ... -c should write compressed bytes to stdout",
    );
    assert!(
        !output_path.exists(),
        "-o ... -c should not create {}",
        output_path.display(),
    );

    let mut dec = Command::new(bin_path())
        .args(["-d", "-c", "-q", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    dec.stdin.as_mut().unwrap().write_all(&out.stdout).unwrap();
    let dec_out = dec.wait_with_output().unwrap();
    assert!(
        dec_out.status.success(),
        "decompress stderr: {}",
        String::from_utf8_lossy(&dec_out.stderr),
    );
    assert_eq!(dec_out.stdout, payload);
}

#[test]
fn cli_long_stdout_output_order_is_last_wins() {
    // Long --stdout is an alias for -c in upstream's output-routing
    // state machine, including order-sensitive interaction with -o.
    let tmp = temp_path("long_stdout_order", "dir");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let payload = b"long stdout output order payload".repeat(8);
    let output_after_stdout = tmp.join("stdout-then-output.zst");
    let mut file_child = Command::new(bin_path())
        .args(["--stdout", "-q", "-o"])
        .arg(&output_after_stdout)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    file_child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(&payload)
        .unwrap();
    let file_out = file_child.wait_with_output().unwrap();
    assert!(
        file_out.status.success(),
        "--stdout before -o should write the later output file: {}",
        String::from_utf8_lossy(&file_out.stderr),
    );
    assert!(
        file_out.stdout.is_empty(),
        "--stdout before -o should leave stdout empty"
    );
    assert!(
        output_after_stdout.exists(),
        "--stdout before -o did not create the output file"
    );

    let decoded = Command::new(bin_path())
        .args(["-d", "-q", "--stdout"])
        .arg(&output_after_stdout)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        decoded.status.success(),
        "decode of --stdout before -o output failed: {}",
        String::from_utf8_lossy(&decoded.stderr)
    );
    assert_eq!(decoded.stdout, payload);

    let output_before_stdout = tmp.join("output-then-stdout.zst");
    let mut stdout_child = Command::new(bin_path())
        .args(["-q", "-o"])
        .arg(&output_before_stdout)
        .args(["--stdout", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    stdout_child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(&payload)
        .unwrap();
    let stdout_out = stdout_child.wait_with_output().unwrap();
    assert!(
        stdout_out.status.success(),
        "-o before --stdout should write stdout: {}",
        String::from_utf8_lossy(&stdout_out.stderr),
    );
    assert!(
        !stdout_out.stdout.is_empty(),
        "-o before --stdout did not emit compressed stdout"
    );
    assert!(
        !output_before_stdout.exists(),
        "-o before --stdout unexpectedly created the output file"
    );

    decode_with_cli(
        &["-d", "-q", "--stdout", "-"],
        &stdout_out.stdout,
        &payload,
        "-o before --stdout frame",
    );

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cli_accepts_equals_short_output_file_value() {
    // Upstream accepts `-o=FILE` as a field-bearing short option.
    // This is distinct from the rejected `-oFILE` spelling below.
    use std::fs;
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let output_path = tmp.join(format!("zstd_pure_equals_o_{pid}.zst"));
    let _ = fs::remove_file(&output_path);
    let output_arg = format!("-o={}", output_path.display());
    let payload = b"equals output option payload".repeat(8);

    let mut child = Command::new(bin_path())
        .args(["-q"])
        .arg(&output_arg)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.as_mut().unwrap().write_all(&payload).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "-o=FILE should succeed, stderr: {}",
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(
        out.stdout.is_empty(),
        "-o=FILE should write to the output file, not stdout",
    );
    assert!(output_path.exists(), "-o=FILE did not create output");

    let dec_out = Command::new(bin_path())
        .args(["-d", "-c", "-q"])
        .arg(&output_path)
        .output()
        .unwrap();
    assert!(
        dec_out.status.success(),
        "decompress stderr: {}",
        String::from_utf8_lossy(&dec_out.stderr),
    );
    assert_eq!(dec_out.stdout, payload);

    let _ = fs::remove_file(&output_path);
}

#[test]
fn cli_repeated_mixed_output_file_forms_use_last_value() {
    // Upstream mutates the output path as it parses each -o field.
    // The separated and -o= forms must share one order-sensitive
    // destination slot, with the last field value winning.
    let tmp = temp_path("mixed_output_fields", "dir");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let payload = b"mixed output field payload ".repeat(12);
    let first_path = tmp.join("first.zst");
    let second_path = tmp.join("second.zst");
    let third_path = tmp.join("third.zst");

    let second_arg = format!("-o={}", second_path.display());
    let mut child = Command::new(bin_path())
        .args(["-q", "-o"])
        .arg(&first_path)
        .arg(&second_arg)
        .arg("-o")
        .arg(&third_path)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.as_mut().unwrap().write_all(&payload).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "mixed -o forms should succeed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.stdout.is_empty(),
        "mixed -o forms should write the selected file, not stdout"
    );
    assert!(
        !first_path.exists() && !second_path.exists(),
        "stale -o values were used instead of the last value"
    );
    assert!(third_path.exists(), "last -o value did not receive output");

    let dec = Command::new(bin_path())
        .args(["-d", "-c", "-q"])
        .arg(&third_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        dec.status.success(),
        "mixed -o output did not decode: {}",
        String::from_utf8_lossy(&dec.stderr)
    );
    assert_eq!(dec.stdout, payload);

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cli_rejects_attached_short_output_file_value() {
    // Upstream rejects `-oFILE`: field-bearing commands may only use
    // a separate value or the `-o=FILE` form.
    let tmp = std::env::temp_dir().join(format!("zstd_pure_attached_o_{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let output_path = tmp.join("out.zst");
    let attached_output = format!("-o{}", output_path.display());

    let mut child = Command::new(bin_path())
        .args(["-q"])
        .arg(attached_output)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let _ = child.stdin.as_mut().unwrap().write_all(b"payload");
    let out = child.wait_with_output().unwrap();

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "attached -o value should be rejected"
    );
    assert!(
        stderr.contains("command cannot be separated from its argument by another command"),
        "unexpected attached -o diagnostic: {stderr}"
    );
    assert!(
        out.stdout.is_empty(),
        "rejected attached -o value must not emit compressed data"
    );
    assert!(
        !output_path.exists(),
        "rejected attached -o value created an output file"
    );

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cli_rejects_field_options_separated_from_values_by_another_option() {
    // Upstream's NEXT_FIELD rejects `-D -q` and `-o -q`: field
    // options may consume a separate argv value, but not another
    // command. Keep this distinct from valid `-D=-q` / `-o=-q`.
    let tmp = std::env::temp_dir().join(format!("zstd_pure_field_arg_{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    for args in [&["-c", "-D", "-q", "-"][..], &["-c", "-o", "-q", "-"][..]] {
        let mut child = Command::new(bin_path())
            .current_dir(&tmp)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let _ = child.stdin.as_mut().unwrap().write_all(b"payload");
        let out = child.wait_with_output().unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);

        assert!(
            !out.status.success(),
            "{args:?} should reject an option where a value is required",
        );
        assert!(
            stderr.contains("command cannot be separated"),
            "{args:?} emitted unexpected diagnostic: {stderr}",
        );
        assert!(
            out.stdout.is_empty(),
            "{args:?} rejection must not write stdout",
        );
    }
    assert!(
        !tmp.join("-q").exists(),
        "rejected -o -q invocation created an output file"
    );

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cli_output_file_dash_spellings_match_upstream() {
    // Upstream does not treat separated `-o -` as stdout: NEXT_FIELD
    // rejects a value that starts with `-`. Use `-c/--stdout` for
    // stdout output instead. The short equals form `-o=-` is
    // different: upstream accepts it as a literal file named `-`,
    // matching the generic `-o=FILE` spelling.
    let tmp = temp_path("dash_output", "dir");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let mut child = Command::new(bin_path())
        .current_dir(&tmp)
        .args(["-q", "-o", "-", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let _ = child.stdin.as_mut().unwrap().write_all(b"payload");
    let out = child.wait_with_output().unwrap();
    assert!(
        !out.status.success(),
        "separated -o - should reject output file '-' like upstream"
    );
    assert!(
        out.stdout.is_empty(),
        "separated -o - rejection must not emit compressed data"
    );
    assert!(
        !tmp.join("-").exists(),
        "separated -o - rejection created a literal '-' output file"
    );

    fs::write(tmp.join("in.txt"), b"dash output payload").unwrap();

    let out = Command::new(bin_path())
        .current_dir(&tmp)
        .args(["-q", "-o=-", "in.txt"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "-o=- should create a literal '-' output file, stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.stdout.is_empty(),
        "-o=- should write to a file, not stdout"
    );
    assert!(tmp.join("-").exists(), "-o=- did not create literal '-'");

    let dec = Command::new(bin_path())
        .current_dir(&tmp)
        .args(["-d", "-c", "-q", "./-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        dec.status.success(),
        "literal '-' output did not decode: {}",
        String::from_utf8_lossy(&dec.stderr)
    );
    assert_eq!(dec.stdout, b"dash output payload");

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cli_multi_input_output_file_requires_force() {
    use std::fs;
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let input_a = tmp.join(format!("zstd_pure_multi_o_a_{pid}.txt"));
    let input_b = tmp.join(format!("zstd_pure_multi_o_b_{pid}.txt"));
    let output_path = tmp.join(format!("zstd_pure_multi_o_{pid}.zst"));
    let _ = fs::remove_file(&output_path);

    fs::write(&input_a, b"first input").unwrap();
    fs::write(&input_b, b"second input").unwrap();

    let out = Command::new(bin_path())
        .args(["-q", "-o"])
        .arg(&output_path)
        .arg(&input_a)
        .arg(&input_b)
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "one -o for multiple inputs should fail, stderr: {}",
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("Concatenating multiple processed inputs"),
        "unexpected stderr: {}",
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(
        !output_path.exists(),
        "multi-input -o rejection should not create {}",
        output_path.display(),
    );

    let _ = fs::remove_file(&input_a);
    let _ = fs::remove_file(&input_b);
}

#[test]
fn cli_multi_input_output_file_accepts_prompt_confirmation() {
    use std::fs;
    let tmp = std::env::temp_dir();
    let pid = std::process::id();

    for (label, answer) in [
        ("lowercase", b"y\n" as &[u8]),
        ("uppercase", b"Y\n"),
        ("first-byte", b"yn\n"),
    ] {
        let input_a = tmp.join(format!("zstd_pure_multi_o_prompt_{label}_a_{pid}.txt"));
        let input_b = tmp.join(format!("zstd_pure_multi_o_prompt_{label}_b_{pid}.txt"));
        let output_path = tmp.join(format!("zstd_pure_multi_o_prompt_{label}_{pid}.zst"));
        let _ = fs::remove_file(&output_path);

        fs::write(&input_a, b"first prompt input").unwrap();
        fs::write(&input_b, b"second prompt input").unwrap();

        let mut child = Command::new(bin_path())
            .args(["-o"])
            .arg(&output_path)
            .arg(&input_a)
            .arg(&input_b)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.as_mut().unwrap().write_all(answer).unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "{label} confirmed multi-input -o should succeed, stderr: {}",
            String::from_utf8_lossy(&out.stderr),
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("Proceed? (y/n):"),
            "{label} confirmed multi-input -o did not prompt like upstream: {stderr}",
        );
        assert!(
            output_path.exists(),
            "{label} confirmed multi-input -o should create {}",
            output_path.display(),
        );
        assert!(
            out.stdout.is_empty(),
            "{label} confirmed multi-input -o should write the explicit file, not stdout",
        );

        let decoded = Command::new(bin_path())
            .args(["-d", "-q", "-c"])
            .arg(&output_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(
            decoded.status.success(),
            "{label} decode confirmed multi-input -o output failed: {}",
            String::from_utf8_lossy(&decoded.stderr),
        );
        assert_eq!(decoded.stdout, b"first prompt inputsecond prompt input");

        let _ = fs::remove_file(&input_a);
        let _ = fs::remove_file(&input_b);
        let _ = fs::remove_file(&output_path);
    }
}

#[test]
fn cli_multi_input_output_file_decline_aborts_without_output() {
    use std::fs;
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let input_a = tmp.join(format!("zstd_pure_multi_o_decline_a_{pid}.txt"));
    let input_b = tmp.join(format!("zstd_pure_multi_o_decline_b_{pid}.txt"));
    let output_path = tmp.join(format!("zstd_pure_multi_o_decline_{pid}.zst"));
    let _ = fs::remove_file(&output_path);

    fs::write(&input_a, b"first declined prompt input").unwrap();
    fs::write(&input_b, b"second declined prompt input").unwrap();

    let mut child = Command::new(bin_path())
        .args(["-o"])
        .arg(&output_path)
        .arg(&input_a)
        .arg(&input_b)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.as_mut().unwrap().write_all(b"n\n").unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        !out.status.success(),
        "declined multi-input -o prompt should fail",
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Proceed? (y/n):") && stderr.contains("Aborting"),
        "declined multi-input -o emitted unexpected stderr: {stderr}",
    );
    assert!(
        !output_path.exists(),
        "declined multi-input -o should not create {}",
        output_path.display(),
    );
    assert!(
        out.stdout.is_empty(),
        "declined multi-input -o refusal must not write stdout",
    );

    let _ = fs::remove_file(&input_a);
    let _ = fs::remove_file(&input_b);
}

#[test]
fn cli_multi_input_output_file_refuses_prompt_when_any_input_is_stdin() {
    // Upstream's multi-input concatenation confirmation also uses
    // UTIL_requireUserConfirmation, so any stdin input disables the
    // interactive prompt and aborts before producing the destination.
    use std::fs;
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let input_a = tmp.join(format!("zstd_pure_multi_o_stdin_a_{pid}.txt"));
    let output_path = tmp.join(format!("zstd_pure_multi_o_stdin_{pid}.zst"));
    let _ = fs::remove_file(&output_path);

    fs::write(&input_a, b"first stdin-prompt input").unwrap();

    let mut child = Command::new(bin_path())
        .args(["-o"])
        .arg(&output_path)
        .arg(&input_a)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"y\nsecond")
        .unwrap();
    let out = child.wait_with_output().unwrap();

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "multi-input -o with stdin should refuse without reading confirmation",
    );
    assert!(
        stderr.contains(
            "all input files will be processed and concatenated into a single output file"
        ) && stderr.contains("stdin is an input - not proceeding."),
        "multi-input -o with stdin emitted unexpected stderr: {stderr}",
    );
    assert!(
        !stderr.contains("Proceed? (y/n):"),
        "multi-input -o with stdin should not prompt interactively: {stderr}",
    );
    assert!(
        !output_path.exists(),
        "multi-input -o with stdin should not create {}",
        output_path.display(),
    );
    assert!(
        out.stdout.is_empty(),
        "multi-input -o with stdin refusal must not write stdout",
    );

    let _ = fs::remove_file(&input_a);
}

#[test]
fn cli_force_multi_input_output_file_concatenates_frames() {
    use std::fs;
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let input_a = tmp.join(format!("zstd_pure_multi_o_force_a_{pid}.txt"));
    let input_b = tmp.join(format!("zstd_pure_multi_o_force_b_{pid}.txt"));
    let output_path = tmp.join(format!("zstd_pure_multi_o_force_{pid}.zst"));
    let _ = fs::remove_file(&output_path);

    fs::write(&input_a, b"first input").unwrap();
    fs::write(&input_b, b"second input").unwrap();

    let out = Command::new(bin_path())
        .args(["-q", "-f", "-o"])
        .arg(&output_path)
        .arg(&input_a)
        .arg(&input_b)
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "forced multi-input -o should succeed, stderr: {}",
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(
        output_path.exists(),
        "forced multi-input -o should create {}",
        output_path.display(),
    );
    assert!(
        out.stdout.is_empty(),
        "forced multi-input -o should write the explicit file, not stdout",
    );

    let decoded = Command::new(bin_path())
        .args(["-d", "-q", "-c"])
        .arg(&output_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        decoded.status.success(),
        "decode forced multi-input -o output failed: {}",
        String::from_utf8_lossy(&decoded.stderr),
    );
    assert_eq!(decoded.stdout, b"first inputsecond input");

    let _ = fs::remove_file(&input_a);
    let _ = fs::remove_file(&input_b);
    let _ = fs::remove_file(&output_path);
}

#[test]
fn cli_rm_multi_input_output_file_preserves_sources() {
    // Upstream disables --rm when multiple inputs are concatenated
    // into one explicit -o destination, because removing the original
    // files would lose recoverable file boundaries and metadata.
    use std::fs;
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let input_a = tmp.join(format!("zstd_pure_multi_o_rm_a_{pid}.txt"));
    let input_b = tmp.join(format!("zstd_pure_multi_o_rm_b_{pid}.txt"));
    let output_path = tmp.join(format!("zstd_pure_multi_o_rm_{pid}.zst"));
    let _ = fs::remove_file(&output_path);

    fs::write(&input_a, b"first rm-preserved input").unwrap();
    fs::write(&input_b, b"second rm-preserved input").unwrap();

    let mut child = Command::new(bin_path())
        .args(["--rm", "-o"])
        .arg(&output_path)
        .arg(&input_a)
        .arg(&input_b)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.as_mut().unwrap().write_all(b"y\n").unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "--rm multi-input -o should succeed after confirmation: {}",
        String::from_utf8_lossy(&out.stderr),
    );

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Since it's a destructive operation, input files will not be removed."),
        "--rm multi-input -o did not warn that source removal was disabled: {stderr}",
    );
    assert!(
        input_a.exists() && input_b.exists(),
        "--rm multi-input -o removed one of the source files",
    );
    assert!(
        output_path.exists(),
        "--rm multi-input -o did not create {}",
        output_path.display(),
    );
    assert!(
        out.stdout.is_empty(),
        "--rm multi-input -o should write the explicit file, not stdout",
    );

    let decoded = Command::new(bin_path())
        .args(["-d", "-q", "-c"])
        .arg(&output_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        decoded.status.success(),
        "decode --rm multi-input -o output failed: {}",
        String::from_utf8_lossy(&decoded.stderr),
    );
    assert_eq!(
        decoded.stdout,
        b"first rm-preserved inputsecond rm-preserved input"
    );

    let _ = fs::remove_file(&input_a);
    let _ = fs::remove_file(&input_b);
    let _ = fs::remove_file(&output_path);
}

#[test]
fn cli_rm_removes_source_after_successful_file_output() {
    // Upstream `--rm` removes the source only after a successful
    // non-stdout file output. Cover both compression and decompression
    // lifecycle paths through the public CLI.
    let tmp = temp_path("rm_success", "dir");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let input_path = tmp.join("payload.txt");
    let compressed_path = tmp.join("payload.txt.zst");
    let payload = b"rm successful file output payload ".repeat(32);
    fs::write(&input_path, &payload).unwrap();

    let comp = Command::new(bin_path())
        .args(["-q", "--rm"])
        .arg(&input_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        comp.status.success(),
        "--rm compression failed: {}",
        String::from_utf8_lossy(&comp.stderr)
    );
    assert!(
        comp.stdout.is_empty(),
        "--rm file compression should write inferred file, not stdout"
    );
    assert!(
        !input_path.exists(),
        "--rm compression did not remove the source file after success"
    );
    assert!(
        compressed_path.exists(),
        "--rm compression did not create inferred output"
    );

    let dec = Command::new(bin_path())
        .args(["-d", "-q", "--rm"])
        .arg(&compressed_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        dec.status.success(),
        "--rm decompression failed: {}",
        String::from_utf8_lossy(&dec.stderr)
    );
    assert!(
        dec.stdout.is_empty(),
        "--rm file decompression should write inferred file, not stdout"
    );
    assert!(
        !compressed_path.exists(),
        "--rm decompression did not remove the compressed source after success"
    );
    assert_eq!(
        fs::read(&input_path).unwrap(),
        payload,
        "--rm decompression did not restore the original bytes"
    );

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cli_rm_preserves_source_for_stdout_and_failed_output() {
    // Upstream does not remove sources when the selected output is
    // stdout, and a failed decode must never remove the original file.
    let tmp = temp_path("rm_preserve", "dir");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let input_path = tmp.join("stdout-source.txt");
    let payload = b"rm stdout preservation payload ".repeat(16);
    fs::write(&input_path, &payload).unwrap();

    let comp = Command::new(bin_path())
        .args(["-q", "--rm", "-c"])
        .arg(&input_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        comp.status.success(),
        "--rm -c compression failed: {}",
        String::from_utf8_lossy(&comp.stderr)
    );
    assert!(
        !comp.stdout.is_empty(),
        "--rm -c should emit compressed data to stdout"
    );
    assert!(
        input_path.exists(),
        "--rm -c removed the source even though output was stdout"
    );
    assert_eq!(
        fs::read(&input_path).unwrap(),
        payload,
        "--rm -c modified the source file"
    );

    let malformed_path = tmp.join("malformed.zst");
    let inferred_path = tmp.join("malformed");
    fs::write(&malformed_path, b"not a valid zstd frame").unwrap();

    let dec = Command::new(bin_path())
        .args(["-d", "-q", "--rm"])
        .arg(&malformed_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(!dec.status.success(), "--rm malformed decode should fail");
    assert!(
        malformed_path.exists(),
        "--rm malformed decode removed the source after failure"
    );
    assert!(
        !inferred_path.exists(),
        "--rm malformed decode created an output file despite failure"
    );
    assert!(
        dec.stdout.is_empty(),
        "--rm malformed decode should not write stdout"
    );

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cli_keep_and_rm_are_last_wins_like_upstream() {
    // Upstream treats -k/--keep and --rm as order-sensitive source
    // removal directives. --rm after keep removes the source after a
    // successful file output; keep after --rm preserves it.
    let tmp = temp_path("rm_keep_order", "dir");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    for (label, first, second, expect_source_exists) in [
        ("short_keep_then_rm", "-k", "--rm", false),
        ("rm_then_short_keep", "--rm", "-k", true),
        ("long_keep_then_rm", "--keep", "--rm", false),
        ("rm_then_long_keep", "--rm", "--keep", true),
    ] {
        let input_path = tmp.join(format!("{label}.txt"));
        let output_path = tmp.join(format!("{label}.txt.zst"));
        fs::write(&input_path, b"keep rm option order payload").unwrap();

        let out = Command::new(bin_path())
            .args(["-q", first, second])
            .arg(&input_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{first} {second} compression failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            output_path.exists(),
            "{first} {second} did not create inferred output"
        );
        assert_eq!(
            input_path.exists(),
            expect_source_exists,
            "{first} {second} did not follow upstream source-removal order",
        );

        let _ = fs::remove_file(&input_path);
        let _ = fs::remove_file(&output_path);
    }

    for (label, first, second, expect_source_exists) in [
        ("decompress_short_keep_then_rm", "-k", "--rm", false),
        ("decompress_rm_then_short_keep", "--rm", "-k", true),
        ("decompress_long_keep_then_rm", "--keep", "--rm", false),
        ("decompress_rm_then_long_keep", "--rm", "--keep", true),
    ] {
        let input_path = tmp.join(format!("{label}.txt"));
        let compressed_path = tmp.join(format!("{label}.txt.zst"));
        fs::write(&input_path, b"decompress keep rm option order payload").unwrap();

        let comp = Command::new(bin_path())
            .args(["-q", "-f", "-o"])
            .arg(&compressed_path)
            .arg(&input_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(
            comp.status.success(),
            "setup compression failed: {}",
            String::from_utf8_lossy(&comp.stderr)
        );
        fs::remove_file(&input_path).unwrap();

        let out = Command::new(bin_path())
            .args(["-d", "-q", "-f", first, second])
            .arg(&compressed_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{first} {second} decompression failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            input_path.exists(),
            "{first} {second} did not create inferred decompressed output"
        );
        assert_eq!(
            compressed_path.exists(),
            expect_source_exists,
            "{first} {second} did not follow upstream decompression source-removal order",
        );

        let _ = fs::remove_file(&input_path);
        let _ = fs::remove_file(&compressed_path);
    }

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cli_multi_input_decompress_mixed_recognized_formats_to_one_output() {
    // Upstream accepts recognized zstd/gzip containers in the same
    // multi-input decompression command and concatenates decoded
    // payloads into the single explicit output.
    let tmp = temp_path("multi_format_decode", "dir");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let zstd_src = b"decoded from zstd member ".repeat(8);
    let gzip_src = b"decoded from gzip member ".repeat(8);
    let zstd_path = tmp.join("first.zst");
    let gzip_path = tmp.join("second.gz");
    let output_path = tmp.join("combined.out");

    let zstd_frame = run_with_stdin(&bin_path(), &["-q", "-c", "-"], &zstd_src);
    assert!(
        zstd_frame.status.success(),
        "zstd setup compression failed: {}",
        String::from_utf8_lossy(&zstd_frame.stderr)
    );
    let gzip_frame = run_with_stdin(&bin_path(), &["-q", "--format=gzip", "-c", "-"], &gzip_src);
    assert!(
        gzip_frame.status.success(),
        "gzip setup compression failed: {}",
        String::from_utf8_lossy(&gzip_frame.stderr)
    );
    fs::write(&zstd_path, zstd_frame.stdout).unwrap();
    fs::write(&gzip_path, gzip_frame.stdout).unwrap();

    let out = Command::new(bin_path())
        .args(["-d", "-q", "-f", "-o"])
        .arg(&output_path)
        .arg(&zstd_path)
        .arg(&gzip_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "mixed-format multi-input decode failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.stdout.is_empty(),
        "mixed-format multi-input -o should write the explicit file, not stdout"
    );

    let mut expected = zstd_src;
    expected.extend_from_slice(&gzip_src);
    assert_eq!(
        fs::read(&output_path).unwrap(),
        expected,
        "mixed recognized formats did not decode in input order",
    );

    if let Some(upstream) = upstream_zstd() {
        let upstream_output = tmp.join("upstream-combined.out");
        let upstream_out = Command::new(upstream)
            .args(["-d", "-q", "-f", "-o"])
            .arg(&upstream_output)
            .arg(&zstd_path)
            .arg(&gzip_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(
            upstream_out.status.success(),
            "vendored upstream rejected the same mixed-format invocation: {}",
            String::from_utf8_lossy(&upstream_out.stderr)
        );
        assert_eq!(
            fs::read(upstream_output).unwrap(),
            expected,
            "vendored upstream decoded mixed formats differently",
        );
    }

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cli_rejects_non_upstream_level_aliases() {
    // Upstream accepts numeric short forms like `-19`, but rejects
    // `-L` / `--level`. Keep the Rust CLI surface aligned.
    let payload: Vec<u8> = b"clamped-level test "
        .iter()
        .cycle()
        .take(200)
        .copied()
        .collect();
    for args in [
        vec!["-c", "-q", "-L", "99", "-"],
        vec!["-c", "-q", "-L99", "-"],
        vec!["-c", "-q", "--level=9", "-"],
        vec!["-c", "-q", "--level", "9", "-"],
    ] {
        let mut comp = Command::new(bin_path())
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        if let Err(err) = comp.stdin.as_mut().unwrap().write_all(&payload) {
            assert_eq!(
                err.kind(),
                std::io::ErrorKind::BrokenPipe,
                "{args:?} failed to write stdin with unexpected error: {err}",
            );
        }
        let out = comp.wait_with_output().unwrap();
        assert!(
            !out.status.success(),
            "{args:?} should have been rejected; stderr: {}",
            String::from_utf8_lossy(&out.stderr),
        );
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("Incorrect parameter"),
            "{args:?} emitted unexpected stderr: {}",
            String::from_utf8_lossy(&out.stderr),
        );
        assert!(
            out.stdout.is_empty(),
            "{args:?} rejection must not emit compressed data",
        );
    }
}

#[test]
fn cli_rejects_non_upstream_output_and_dict_long_aliases() {
    // Upstream exposes `-o` and `-D`, but not project-local long
    // aliases. Reject both separated and equals forms so hidden clap
    // convenience spellings cannot drift from the vendored CLI.
    let tmp = std::env::temp_dir().join(format!("zstd_pure_long_alias_{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let dict_path = tmp.join("dict.bin");
    let output_path = tmp.join("out.zst");
    fs::write(&dict_path, b"dictionary bytes").unwrap();

    for args in [
        vec![
            "-q".to_string(),
            "--output-file".to_string(),
            output_path.display().to_string(),
            "-".to_string(),
        ],
        vec![
            "-q".to_string(),
            format!("--output-file={}", output_path.display()),
            "-".to_string(),
        ],
        vec![
            "-c".to_string(),
            "-q".to_string(),
            "--dict".to_string(),
            dict_path.display().to_string(),
            "-".to_string(),
        ],
        vec![
            "-c".to_string(),
            "-q".to_string(),
            format!("--dict={}", dict_path.display()),
            "-".to_string(),
        ],
    ] {
        let mut child = Command::new(bin_path())
            .current_dir(&tmp)
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let _ = child.stdin.as_mut().unwrap().write_all(b"payload");
        let out = child.wait_with_output().unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);

        assert!(
            !out.status.success(),
            "{args:?} should reject non-upstream long alias",
        );
        assert!(
            stderr.contains("Incorrect parameter")
                || stderr.contains("unexpected argument")
                || stderr.contains("unrecognized"),
            "{args:?} emitted unexpected diagnostic: {stderr}",
        );
        assert!(
            out.stdout.is_empty(),
            "{args:?} rejection must not emit compressed data",
        );
    }
    assert!(
        !output_path.exists(),
        "rejected --output-file alias created an output file"
    );

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cli_accepts_upstream_short_level_syntax() {
    let payload: Vec<u8> = b"short level syntax payload "
        .iter()
        .cycle()
        .take(600)
        .copied()
        .collect();

    fn compress(args: &[&str], payload: &[u8]) -> Vec<u8> {
        let mut child = Command::new(bin_path())
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.as_mut().unwrap().write_all(payload).unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "compress {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr),
        );
        out.stdout
    }

    let short = compress(&["-c", "-q", "-1", "-"], &payload);
    let canonical = compress(&["-c", "-q", "-1", "-"], &payload);
    assert_eq!(short, canonical, "-1 should be the canonical level-1 form");

    for args in [
        ["-q", "-1c", "-"],
        ["-q", "-c1", "-"],
        ["-q", "-19c", "-"],
        ["-q", "-c19", "-"],
        ["-c", "-1q", "-"],
        ["-c", "-19q", "-"],
        ["-c", "-q1", "-"],
    ] {
        let clustered = compress(&args, &payload);
        let level = if args[1].contains("19") { "19" } else { "1" };
        let canonical_level = format!("-{level}");
        let canonical = compress(&["-c", "-q", &canonical_level, "-"], &payload);
        assert_eq!(
            clustered, canonical,
            "{:?} should behave like -c -{level}",
            args,
        );
    }
}

#[test]
fn cli_ultra_level_22_is_accepted_and_cross_decodes_upstream() {
    // Upstream requires `--ultra` to intentionally select the public
    // high-compression level range beyond 19. The translated CLI must
    // accept the same spelling and emit a standard frame.
    let Some(upstream) = upstream_zstd() else {
        eprintln!("vendored upstream zstd/programs/zstd not built; skipping");
        return;
    };

    let payload: Vec<u8> = b"ultra level payload with enough repeated content "
        .iter()
        .cycle()
        .take(4096)
        .copied()
        .collect();

    let mut comp = Command::new(bin_path())
        .args(["-c", "-q", "--ultra", "-22", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    comp.stdin.as_mut().unwrap().write_all(&payload).unwrap();
    let comp_out = comp.wait_with_output().unwrap();
    assert!(
        comp_out.status.success(),
        "--ultra -22 compression failed: {}",
        String::from_utf8_lossy(&comp_out.stderr)
    );

    let mut upstream_dec = Command::new(&upstream)
        .args(["-d", "-c", "-q", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    upstream_dec
        .stdin
        .as_mut()
        .unwrap()
        .write_all(&comp_out.stdout)
        .unwrap();
    let upstream_out = upstream_dec.wait_with_output().unwrap();
    assert!(
        upstream_out.status.success(),
        "upstream rejected --ultra -22 output: {}",
        String::from_utf8_lossy(&upstream_out.stderr)
    );
    assert_eq!(upstream_out.stdout, payload);
}

#[test]
fn cli_fast_long_and_max_shorthands_emit_decodable_frames() {
    let Some(upstream) = upstream_zstd() else {
        eprintln!("vendored upstream zstd/programs/zstd not built; skipping");
        return;
    };
    let payload: Vec<u8> = b"fast long max parser payload "
        .iter()
        .cycle()
        .take(4096)
        .copied()
        .collect();

    for args in [
        &["-c", "-q", "--fast", "-"][..],
        &["-c", "-q", "--fast=3", "-"],
        &["-c", "-q", "--fast=1K", "-"],
        &["-c", "-q", "--fast=1M", "-"],
        &["-c", "-q", "--long", "-"],
        &["-c", "-q", "--long=27", "-"],
        &["-c", "-q", "--max", "-"],
    ] {
        let mut comp = Command::new(bin_path())
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        comp.stdin.as_mut().unwrap().write_all(&payload).unwrap();
        let comp_out = comp.wait_with_output().unwrap();
        assert!(
            comp_out.status.success(),
            "{args:?} compression failed: {}",
            String::from_utf8_lossy(&comp_out.stderr)
        );

        let mut dec = Command::new(&upstream)
            .args(["-d", "-c", "-q", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        dec.stdin
            .as_mut()
            .unwrap()
            .write_all(&comp_out.stdout)
            .unwrap();
        let dec_out = dec.wait_with_output().unwrap();
        assert!(
            dec_out.status.success(),
            "upstream rejected {args:?} output: {}",
            String::from_utf8_lossy(&dec_out.stderr)
        );
        assert_eq!(dec_out.stdout, payload, "{args:?} decoded mismatch");
    }
}

#[test]
fn cli_g_suffix_numeric_options_match_upstream() {
    // Upstream 1.6.0's numeric helper accepts G/GB/GiB suffixes for
    // size-like numeric fields. For --fast and short numeric levels
    // these spellings should still emit decodable frames; --long=1G
    // parses the suffix too, but is rejected later as an out-of-bound
    // window log rather than as a malformed numeric argument.
    let payload = b"numeric suffix payload".repeat(8);

    for args in [
        &["-c", "-q", "--fast=1G", "-"][..],
        &["-c", "-q", "--fast=1GB", "-"],
        &["-c", "-q", "--fast=1GiB", "-"],
        &["-c", "-q", "-1G", "-"],
        &["-c", "-q", "-1GB", "-"],
        &["-c", "-q", "-1GiB", "-"],
    ] {
        let mut child = Command::new(bin_path())
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.as_mut().unwrap().write_all(&payload).unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "{args:?} should accept G suffix like upstream: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            !out.stdout.is_empty(),
            "{args:?} should emit compressed data"
        );
        decode_with_cli(
            &["-d", "-c", "-q", "-"],
            &out.stdout,
            &payload,
            &format!("{args:?} G-suffix frame"),
        );
    }

    for args in [
        &["-c", "-q", "--fast=G", "-"][..],
        &["-c", "-q", "-1Gx", "-"],
        &["-c", "-q", "-1Z", "-"],
        &["-c", "-q", "--long=1G", "-"],
        &["-c", "-q", "--long=1GB", "-"],
        &["-c", "-q", "--long=1GiB", "-"],
        &["-c", "-q", "--long=27G", "-"],
    ] {
        let mut child = Command::new(bin_path())
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let _ = child.stdin.as_mut().unwrap().write_all(&payload);
        let out = child.wait_with_output().unwrap();
        assert!(
            !out.status.success(),
            "{args:?} should reject malformed or out-of-bound numeric value"
        );
        assert!(
            out.stdout.is_empty(),
            "{args:?} rejection must not emit compressed data"
        );
    }
}

#[test]
fn cli_fast_numeric_parser_accepts_trailing_junk_after_nonzero_prefix() {
    // Upstream's --fast parser consumes the longest numeric prefix,
    // including K/M/G multipliers, and ignores trailing nonnumeric
    // bytes. This leniency only applies after a nonzero parsed value:
    // level 0 remains invalid, and overflowing prefixes still fail as
    // overflow rather than wrapping or falling back to a smaller value.
    let payload = b"fast numeric trailing junk payload ".repeat(32);

    fn compress(args: &[&str], payload: &[u8]) -> Vec<u8> {
        let mut child = Command::new(bin_path())
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.as_mut().unwrap().write_all(payload).unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "{args:?} should be accepted like upstream: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            !out.stdout.is_empty(),
            "{args:?} should emit compressed data"
        );
        out.stdout
    }

    for (trailing, canonical) in [
        (
            &["-c", "-q", "--fast=1abc", "-"][..],
            &["-c", "-q", "--fast=1", "-"][..],
        ),
        (
            &["-c", "-q", "--fast=1Kabc", "-"][..],
            &["-c", "-q", "--fast=1K", "-"][..],
        ),
        (
            &["-c", "-q", "--fast=3GBjunk", "-"][..],
            &["-c", "-q", "--fast=3GB", "-"][..],
        ),
    ] {
        let trailing_frame = compress(trailing, &payload);
        let canonical_frame = compress(canonical, &payload);
        assert_eq!(
            trailing_frame, canonical_frame,
            "{trailing:?} should parse like {canonical:?}"
        );
        decode_with_cli(
            &["-d", "-c", "-q", "-"],
            &trailing_frame,
            &payload,
            &format!("{trailing:?} trailing-junk --fast frame"),
        );

        if let Some(upstream) = upstream_zstd() {
            let upstream_trailing = run_with_stdin(&upstream, trailing, &payload);
            let upstream_canonical = run_with_stdin(&upstream, canonical, &payload);
            assert!(
                upstream_trailing.status.success(),
                "vendored upstream rejected {trailing:?}: {}",
                String::from_utf8_lossy(&upstream_trailing.stderr)
            );
            assert!(
                upstream_canonical.status.success(),
                "vendored upstream rejected {canonical:?}: {}",
                String::from_utf8_lossy(&upstream_canonical.stderr)
            );
            assert_eq!(
                upstream_trailing.stdout, upstream_canonical.stdout,
                "vendored upstream no longer parses {trailing:?} like {canonical:?}"
            );
        }
    }

    for args in [
        &["-c", "-q", "--fast=0abc", "-"][..],
        &["-c", "-q", "--fast=0", "-"],
    ] {
        let mut child = Command::new(bin_path())
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let _ = child.stdin.as_mut().unwrap().write_all(&payload);
        let out = child.wait_with_output().unwrap();
        assert!(
            !out.status.success(),
            "{args:?} should reject parsed fast level 0"
        );
        assert!(
            out.stdout.is_empty(),
            "{args:?} rejection must not emit compressed data"
        );
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("Incorrect parameter"),
            "{args:?} emitted unexpected diagnostic: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    for args in [
        &["-c", "-q", "--fast=184467440737095516160abc", "-"][..],
        &[
            "-c",
            "-q",
            "--fast=999999999999999999999999999999999999999",
            "-",
        ],
    ] {
        let mut child = Command::new(bin_path())
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let _ = child.stdin.as_mut().unwrap().write_all(&payload);
        let out = child.wait_with_output().unwrap();
        assert!(
            !out.status.success(),
            "{args:?} should preserve upstream overflow rejection"
        );
        assert!(
            out.stdout.is_empty(),
            "{args:?} overflow rejection must not emit compressed data"
        );
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("overflows 32-bit unsigned int"),
            "{args:?} emitted unexpected overflow diagnostic: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn cli_level_above_19_requires_ultra_to_change_level() {
    // Upstream clamps `-22` down to level 19 unless `--ultra` is
    // present. Keep the coverage honest by checking both paths:
    // `-22` alone is a warning-compatible spelling of `-19`, while
    // `--ultra -22` selects the higher level.
    let payload: Vec<u8> = b"ultra clamp coverage payload "
        .iter()
        .cycle()
        .take(4096)
        .copied()
        .collect();

    fn compress(args: &[&str], payload: &[u8]) -> Vec<u8> {
        let mut child = Command::new(bin_path())
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.as_mut().unwrap().write_all(payload).unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "compress {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr),
        );
        out.stdout
    }

    let level_19 = compress(&["-c", "-q", "-19", "-"], &payload);
    let clamped_22 = compress(&["-c", "-q", "-22", "-"], &payload);
    let ultra_22 = compress(&["-c", "-q", "--ultra", "-22", "-"], &payload);

    assert_eq!(
        clamped_22, level_19,
        "-22 without --ultra should mirror upstream's level-19 clamp",
    );
    assert!(
        !ultra_22.is_empty(),
        "--ultra -22 should be accepted and emit a frame",
    );
}

#[test]
fn cli_accepts_upstream_parser_aliases_and_noop_flags() {
    // These switches are accepted by upstream zstd even though this
    // small translated CLI either treats them as aliases or no-ops.
    // The test locks down parser compatibility without requiring the
    // flags to appear in normal help.
    let tmp = std::env::temp_dir().join(format!("zstd_pure_noop_flags_{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let input = tmp.join("input.txt");
    let output = tmp.join("output.zst");
    fs::write(&input, b"accepted upstream no-op flags ".repeat(16)).unwrap();

    for flag in [
        "--compress",
        "-z",
        "-k",
        "--keep",
        "-n",
        "--format=zstd",
        "--sparse",
        "--no-sparse",
        "--mmap-dict",
        "--no-mmap-dict",
        "--progress",
        "--no-progress",
        "--rsyncable",
        "--adapt",
    ] {
        let out = Command::new(bin_path())
            .args(["-q", flag, "-c"])
            .arg(&input)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{flag} should be accepted like upstream: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            !out.stdout.is_empty(),
            "{flag} should still produce compressed stdout",
        );
    }

    for flags in [&["--compress", "-z"][..], &["-k", "--keep"][..]] {
        let out = Command::new(bin_path())
            .arg("-q")
            .args(flags)
            .arg("-c")
            .arg(&input)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{flags:?} should be accepted as redundant upstream aliases: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            !out.stdout.is_empty(),
            "{flags:?} should still produce compressed stdout",
        );
    }

    let out = Command::new(bin_path())
        .args([
            "-q",
            "--keep",
            "-n",
            "--format=zstd",
            "--sparse",
            "--no-sparse",
            "--mmap-dict",
            "--no-mmap-dict",
            "--progress",
            "--no-progress",
            "--rsyncable",
            "--adapt",
            "-o",
        ])
        .arg(&output)
        .arg(&input)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "accepted upstream flags should not reject: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        output.exists(),
        "accepted no-op flags did not create output"
    );
    assert!(
        input.exists(),
        "-k/--keep compatibility path should preserve the input",
    );

    let dec = Command::new(bin_path())
        .args(["-d", "-q", "-c"])
        .arg(&output)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        dec.status.success(),
        "decode accepted-flag output failed: {}",
        String::from_utf8_lossy(&dec.stderr)
    );
    assert_eq!(dec.stdout, b"accepted upstream no-op flags ".repeat(16));

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cli_rejects_separated_format_and_malformed_adapt_values() {
    // Upstream accepts only the equals form for --format, and accepts
    // --adapt either bare or with well-formed min=/max= bounds. Keep
    // the translated parser strict in the same places.
    let tmp = std::env::temp_dir().join(format!(
        "zstd_pure_format_adapt_reject_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let input = tmp.join("input.txt");
    fs::write(&input, b"format adapt parser payload ".repeat(12)).unwrap();

    for args in [
        vec!["-q", "--format", "zstd", "-c"],
        vec!["-q", "--adapt=min=10,max=1", "-c"],
        vec!["-q", "--adapt=foo=1", "-c"],
        vec!["-q", "--adapt=foo=-1", "-c"],
        vec!["-q", "--adapt=min=1,garbage", "-c"],
        vec!["-q", "--adapt=max=-3Gx", "-c"],
        vec!["-q", "--adapt=", "-c"],
    ] {
        let out = Command::new(bin_path())
            .args(&args)
            .arg(&input)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(
            !out.status.success(),
            "{args:?} should be rejected like upstream",
        );
        assert!(
            out.stdout.is_empty(),
            "{args:?} rejection must not emit compressed data",
        );
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("Incorrect parameter")
                || String::from_utf8_lossy(&out.stderr).contains("unexpected argument")
                || String::from_utf8_lossy(&out.stderr).contains("a value is required"),
            "{args:?} emitted unexpected diagnostic: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    for args in [
        vec!["-q", "--adapt=min=1,max=3", "--format=zstd", "-c"],
        vec!["-q", "--adapt=max=4,min=1", "-c"],
        vec!["-q", "--adapt=max=-1", "-c"],
        vec!["-q", "--adapt=min=-1", "-c"],
        vec!["-q", "--adapt=min=-2,max=3", "-c"],
        vec!["-q", "--adapt=min=-2,max=-1", "-c"],
        vec!["-q", "--adapt=min=-1,max=1", "-c"],
        vec!["-q", "--adapt=max=-3G", "-c"],
        vec!["-q", "--adapt=min=", "-c"],
        vec!["-q", "--adapt=max=", "-c"],
    ] {
        let out = Command::new(bin_path())
            .args(&args)
            .arg(&input)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{args:?} should be accepted like upstream: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            !out.stdout.is_empty(),
            "{args:?} should produce compressed stdout",
        );
    }

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cli_format_gzip_emits_gzip_stream_like_upstream() {
    // Vendored upstream accepts --format=gzip and switches the output
    // container, not merely the parser surface. Assert the observable
    // gzip magic so this test catches a zstd-frame no-op translation.
    let Some(upstream) = upstream_zstd() else {
        eprintln!("vendored upstream zstd/programs/zstd not built; skipping");
        return;
    };

    let payload = b"format gzip accepted by upstream ".repeat(8);

    fn compress_with(bin: &std::path::Path, payload: &[u8]) -> Vec<u8> {
        let mut child = Command::new(bin)
            .args(["-q", "--format=gzip", "-c", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.as_mut().unwrap().write_all(payload).unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "--format=gzip should be accepted like upstream: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        out.stdout
    }

    let theirs = compress_with(&upstream, &payload);
    assert!(
        theirs.starts_with(&[0x1f, 0x8b, 0x08]),
        "vendored upstream --format=gzip did not emit a gzip stream"
    );

    let ours = compress_with(&bin_path(), &payload);
    assert!(
        ours.starts_with(&[0x1f, 0x8b, 0x08]),
        "--format=gzip emitted a non-gzip stream; first bytes: {:02x?}",
        &ours[..ours.len().min(4)]
    );

    let mut dec = Command::new(&upstream)
        .args(["-q", "-d", "--format=gzip", "-c", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    dec.stdin.as_mut().unwrap().write_all(&ours).unwrap();
    let dec_out = dec.wait_with_output().unwrap();
    assert!(
        dec_out.status.success(),
        "vendored upstream could not decode our --format=gzip stream: {}",
        String::from_utf8_lossy(&dec_out.stderr)
    );
    assert_eq!(dec_out.stdout, payload);
}

#[test]
fn cli_decompresses_upstream_gzip_stream_like_upstream() {
    let Some(upstream) = upstream_zstd() else {
        eprintln!("vendored upstream zstd/programs/zstd not built; skipping");
        return;
    };
    if !external_tool_available("gzip") {
        eprintln!("gzip filter not available; skipping upstream gzip decode parity");
        return;
    }

    let payload = b"gzip decompression parity payload ".repeat(64);
    let upstream_gzip = run_with_stdin(&upstream, &["-q", "--format=gzip", "-c", "-"], &payload);
    assert!(
        upstream_gzip.status.success(),
        "vendored upstream gzip compression failed: {}",
        String::from_utf8_lossy(&upstream_gzip.stderr)
    );
    assert!(
        upstream_gzip.stdout.starts_with(&[0x1f, 0x8b, 0x08]),
        "vendored upstream did not emit gzip bytes"
    );

    let upstream_decoded =
        run_with_stdin(&upstream, &["-q", "-d", "-c", "-"], &upstream_gzip.stdout);
    assert!(
        upstream_decoded.status.success(),
        "vendored upstream could not decode its gzip output: {}",
        String::from_utf8_lossy(&upstream_decoded.stderr)
    );
    assert_eq!(upstream_decoded.stdout, payload);

    let ours_decoded = run_with_stdin(&bin_path(), &["-q", "-d", "-c", "-"], &upstream_gzip.stdout);
    assert!(
        ours_decoded.status.success(),
        "our CLI rejected upstream gzip input: {}",
        String::from_utf8_lossy(&ours_decoded.stderr)
    );
    assert_eq!(ours_decoded.stdout, payload);
}

#[test]
fn cli_format_xz_lzma_lz4_cross_decode_like_upstream() {
    let Some(upstream) = upstream_zstd() else {
        eprintln!("vendored upstream zstd/programs/zstd not built; skipping");
        return;
    };

    let payload = b"alternate format cross-decode payload ".repeat(96);
    for format in ["xz", "lzma", "lz4"] {
        let required_tool = if format == "lz4" { "lz4" } else { "xz" };
        if !external_tool_available(required_tool) {
            eprintln!("{required_tool} filter not available; skipping --format={format}");
            continue;
        }

        let format_arg = format!("--format={format}");
        let args = ["-q", format_arg.as_str(), "-c", "-"];

        let theirs = run_with_stdin(&upstream, &args, &payload);
        if !theirs.status.success() {
            eprintln!(
                "vendored upstream {format} support unavailable; skipping: {}",
                String::from_utf8_lossy(&theirs.stderr)
            );
            continue;
        }
        let upstream_roundtrip =
            run_with_stdin(&upstream, &["-q", "-d", "-c", "-"], &theirs.stdout);
        assert!(
            upstream_roundtrip.status.success(),
            "vendored upstream could not decode its {format} output: {}",
            String::from_utf8_lossy(&upstream_roundtrip.stderr)
        );
        assert_eq!(upstream_roundtrip.stdout, payload);

        let ours_from_theirs =
            run_with_stdin(&bin_path(), &["-q", "-d", "-c", "-"], &theirs.stdout);
        assert!(
            ours_from_theirs.status.success(),
            "our CLI rejected upstream {format} input: {}",
            String::from_utf8_lossy(&ours_from_theirs.stderr)
        );
        assert_eq!(ours_from_theirs.stdout, payload);

        let ours = run_with_stdin(&bin_path(), &args, &payload);
        assert!(
            ours.status.success(),
            "our CLI rejected --format={format}: {}",
            String::from_utf8_lossy(&ours.stderr)
        );
        let upstream_from_ours = run_with_stdin(&upstream, &["-q", "-d", "-c", "-"], &ours.stdout);
        assert!(
            upstream_from_ours.status.success(),
            "vendored upstream rejected our {format} output: {}",
            String::from_utf8_lossy(&upstream_from_ours.stderr)
        );
        assert_eq!(upstream_from_ours.stdout, payload);

        let ours_roundtrip = run_with_stdin(&bin_path(), &["-q", "-d", "-c", "-"], &ours.stdout);
        assert!(
            ours_roundtrip.status.success(),
            "our CLI could not decode its own {format} output: {}",
            String::from_utf8_lossy(&ours_roundtrip.stderr)
        );
        assert_eq!(ours_roundtrip.stdout, payload);
    }
}

#[test]
fn cli_format_file_suffix_inference_matches_upstream() {
    let tmp = temp_path("format_suffix", "dir");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let payload = b"format suffix inference payload".repeat(20);
    for (format, ext) in [
        ("gzip", ".gz"),
        ("xz", ".xz"),
        ("lzma", ".lzma"),
        ("lz4", ".lz4"),
    ] {
        let required_tool = match format {
            "xz" | "lzma" => Some("xz"),
            "lz4" => Some("lz4"),
            _ => None,
        };
        if required_tool.is_some_and(|tool| !external_tool_available(tool)) {
            eprintln!("external filter not available; skipping --format={format} suffix inference");
            continue;
        }

        let input = tmp.join(format!("{format}.txt"));
        let compressed = PathBuf::from(format!("{}{}", input.display(), ext));
        fs::write(&input, &payload).unwrap();

        let out = Command::new(bin_path())
            .args(["-q", "-f", &format!("--format={format}")])
            .arg(&input)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "--format={format} file compression failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            out.stdout.is_empty(),
            "--format={format} file compression should write inferred output, not stdout"
        );
        assert!(
            compressed.exists(),
            "--format={format} did not create inferred {} output",
            compressed.display()
        );

        fs::remove_file(&input).unwrap();
        let dec = Command::new(bin_path())
            .args(["-q", "-d", "-f"])
            .arg(&compressed)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(
            dec.status.success(),
            "{format} decompression with inferred suffix failed: {}",
            String::from_utf8_lossy(&dec.stderr)
        );
        assert_eq!(
            fs::read(&input).unwrap(),
            payload,
            "{format} decompression did not recreate the stripped suffix path"
        );
    }

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cli_decompress_tar_shorthand_suffixes_restore_tar_path() {
    let tmp = temp_path("tar_suffix", "dir");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let payload = b"tar shorthand suffix payload".repeat(32);
    for (format, shorthand_ext, required_tool) in [
        ("zstd", ".tzst", None),
        ("gzip", ".tgz", None),
        ("xz", ".txz", Some("xz")),
        ("lz4", ".tlz4", Some("lz4")),
    ] {
        if required_tool.is_some_and(|tool| !external_tool_available(tool)) {
            eprintln!("external filter not available; skipping {shorthand_ext} suffix inference");
            continue;
        }

        let tar_path = tmp.join(format!("{format}.tar"));
        let compressed_path = tmp.join(format!("{format}{shorthand_ext}"));
        let _ = fs::remove_file(&tar_path);
        let _ = fs::remove_file(&compressed_path);
        fs::write(&tar_path, &payload).unwrap();

        let mut args = vec!["-q", "-f"];
        let format_arg;
        if format != "zstd" {
            format_arg = format!("--format={format}");
            args.push(format_arg.as_str());
        }
        args.push("-o");
        let comp = Command::new(bin_path())
            .args(&args)
            .arg(&compressed_path)
            .arg(&tar_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(
            comp.status.success(),
            "{shorthand_ext} setup compression failed: {}",
            String::from_utf8_lossy(&comp.stderr)
        );
        assert!(compressed_path.exists());
        fs::remove_file(&tar_path).unwrap();

        let dec = Command::new(bin_path())
            .args(["-q", "-d", "-f"])
            .arg(&compressed_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(
            dec.status.success(),
            "{shorthand_ext} decompression failed: {}",
            String::from_utf8_lossy(&dec.stderr)
        );
        assert!(
            dec.stdout.is_empty(),
            "{shorthand_ext} decompression should write inferred file, not stdout"
        );
        assert_eq!(
            fs::read(&tar_path).unwrap(),
            payload,
            "{shorthand_ext} did not restore the .tar output path"
        );
    }

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cli_rejects_unknown_format_before_reading_or_writing() {
    let tmp = temp_path("bad_format", "dir");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let input = tmp.join("input.txt");
    let output = tmp.join("output.bz2");
    fs::write(&input, b"unknown format should not be read into output").unwrap();

    let out = Command::new(bin_path())
        .args(["-q", "--format=bzip2", "-o"])
        .arg(&output)
        .arg(&input)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(!out.status.success(), "unknown --format should fail");
    assert!(
        out.stdout.is_empty(),
        "unknown --format rejection must not emit stdout"
    );
    assert!(
        !output.exists(),
        "unknown --format rejection created an output file"
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("Incorrect parameter: --format=bzip2"),
        "unexpected unknown --format diagnostic: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cli_default_stdin_paths_match_upstream_stdout_behavior() {
    let payload = b"default stdin stdout behavior ".repeat(64);

    let compressed = run_with_stdin(&bin_path(), &["-q"], &payload);
    assert!(
        compressed.status.success(),
        "default stdin compression failed: {}",
        String::from_utf8_lossy(&compressed.stderr)
    );
    assert!(
        !compressed.stdout.is_empty() && compressed.stdout != payload,
        "default stdin compression must emit compressed bytes to stdout"
    );

    if let Some(upstream) = upstream_zstd() {
        let upstream_decoded =
            run_with_stdin(&upstream, &["-q", "-d", "-c", "-"], &compressed.stdout);
        assert!(
            upstream_decoded.status.success(),
            "upstream rejected default-stdin compressed output: {}",
            String::from_utf8_lossy(&upstream_decoded.stderr)
        );
        assert_eq!(upstream_decoded.stdout, payload);
    }

    let decoded = run_with_stdin(&bin_path(), &["-q", "-d"], &compressed.stdout);
    assert!(
        decoded.status.success(),
        "default stdin decompression failed: {}",
        String::from_utf8_lossy(&decoded.stderr)
    );
    assert_eq!(
        decoded.stdout, payload,
        "stdin decompression without -c should still write stdout"
    );
}

#[test]
fn cli_operation_aliases_are_last_wins_like_upstream() {
    let payload = b"operation alias payload ".repeat(16);
    let mut comp = Command::new(bin_path())
        .args(["-q", "-c", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    comp.stdin.as_mut().unwrap().write_all(&payload).unwrap();
    let comp_out = comp.wait_with_output().unwrap();
    assert!(comp_out.status.success());

    let mut dec = Command::new(bin_path())
        .args(["-z", "-d", "-c", "-q", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    dec.stdin
        .as_mut()
        .unwrap()
        .write_all(&comp_out.stdout)
        .unwrap();
    let dec_out = dec.wait_with_output().unwrap();
    assert!(
        dec_out.status.success(),
        "last operation directive -d should decompress: {}",
        String::from_utf8_lossy(&dec_out.stderr)
    );
    assert_eq!(dec_out.stdout, payload);

    let mut recompress = Command::new(bin_path())
        .args(["--uncompress", "--compress", "-c", "-q", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    recompress
        .stdin
        .as_mut()
        .unwrap()
        .write_all(&payload)
        .unwrap();
    let recompress_out = recompress.wait_with_output().unwrap();
    assert!(
        recompress_out.status.success(),
        "last operation directive -z should compress: {}",
        String::from_utf8_lossy(&recompress_out.stderr)
    );
    assert_ne!(recompress_out.stdout, payload);
}

#[test]
fn cli_test_mode_aliases_validate_without_writing_output() {
    let tmp = std::env::temp_dir().join(format!("zstd_pure_test_mode_{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let input = tmp.join("payload.txt");
    let compressed = tmp.join("payload.txt.zst");
    let explicit_output = tmp.join("must-not-be-created");
    fs::write(&input, b"test mode payload ".repeat(16)).unwrap();

    let comp = Command::new(bin_path())
        .args(["-q", "-o"])
        .arg(&compressed)
        .arg(&input)
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        comp.status.success(),
        "setup compression failed: {}",
        String::from_utf8_lossy(&comp.stderr)
    );

    for args in [
        vec![
            "-t".to_string(),
            "-q".to_string(),
            compressed.display().to_string(),
        ],
        vec![
            "--test".to_string(),
            "-q".to_string(),
            "-o".to_string(),
            explicit_output.display().to_string(),
            compressed.display().to_string(),
        ],
    ] {
        let out = Command::new(bin_path())
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{args:?} should validate compressed input: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            out.stdout.is_empty(),
            "{args:?} test mode must not write decoded bytes to stdout",
        );
    }
    assert!(
        !explicit_output.exists(),
        "--test -o should validate only and not create an output file",
    );

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cli_test_mode_force_rejects_non_zstd_garbage() {
    // Upstream's legacy `-f` pass-through applies to decompression to
    // stdout, not to test/null-output mode. `zstd -t -f` must validate
    // and reject unrecognized input instead of treating it as copied.
    let tmp = temp_path("test_force_garbage", "dir");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let input = tmp.join("garbage.zst");
    let inferred_output = tmp.join("garbage");
    fs::write(&input, b"plain bytes are not a zstd frame").unwrap();

    let out = Command::new(bin_path())
        .args(["-t", "-f", "-q"])
        .arg(&input)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "-t -f should reject non-zstd input, stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.stdout.is_empty(),
        "-t -f rejection must not copy garbage to stdout"
    );
    assert!(
        !inferred_output.exists(),
        "-t -f must not create an inferred output file in test mode"
    );
    assert!(
        input.exists(),
        "-t -f rejection should preserve the input file"
    );

    for args in [
        &["-t", "--pass-through", "-q"][..],
        &["-t", "-f", "--pass-through", "-q"],
        &["-t", "--pass-through", "-f", "-q"],
    ] {
        let pass_through = Command::new(bin_path())
            .args(args)
            .arg(&input)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(
            pass_through.status.success(),
            "{args:?} should mirror upstream explicit pass-through in test mode: {}",
            String::from_utf8_lossy(&pass_through.stderr)
        );
        assert!(
            pass_through.stdout.is_empty(),
            "{args:?} test-mode pass-through must validate without copying to stdout"
        );
        assert!(
            !inferred_output.exists(),
            "{args:?} must not create an inferred output file in test mode"
        );
        assert!(input.exists(), "{args:?} should preserve the input file");
    }

    for (args, should_succeed) in [
        (
            &["-t", "--pass-through", "--no-pass-through", "-q"][..],
            false,
        ),
        (
            &["-t", "--no-pass-through", "--pass-through", "-q"][..],
            true,
        ),
        (
            &["-t", "-f", "--pass-through", "--no-pass-through", "-q"][..],
            false,
        ),
        (
            &["-t", "-f", "--no-pass-through", "--pass-through", "-q"][..],
            true,
        ),
    ] {
        let out = Command::new(bin_path())
            .args(args)
            .arg(&input)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert_eq!(
            out.status.success(),
            should_succeed,
            "{args:?} test-mode pass-through order diverged: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            out.stdout.is_empty(),
            "{args:?} test mode must never copy garbage to stdout"
        );
        assert!(
            !inferred_output.exists(),
            "{args:?} must not create an inferred output file in test mode"
        );

        if let Some(upstream) = upstream_zstd() {
            let upstream_out = Command::new(upstream)
                .args(args)
                .arg(&input)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .unwrap();
            assert_eq!(
                out.status.success(),
                upstream_out.status.success(),
                "{args:?} test-mode pass-through order changed upstream: ours stderr={} upstream stderr={}",
                String::from_utf8_lossy(&out.stderr),
                String::from_utf8_lossy(&upstream_out.stderr)
            );
            assert_eq!(
                out.stdout.is_empty(),
                upstream_out.stdout.is_empty(),
                "{args:?} test-mode stdout presence diverged from upstream"
            );
        }
    }

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cli_advanced_help_flag_prints_help_and_exits() {
    let out = Command::new(bin_path())
        .arg("-H")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "-H should print advanced help successfully: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let help = String::from_utf8_lossy(&out.stdout);
    assert!(
        help.contains("--ultra") && help.contains("-D") && help.contains("--magicless"),
        "-H help missed expected CLI flags: {help}",
    );
}

#[test]
fn cli_rejects_nonexistent_dict_path_with_nonzero_exit() {
    // Safety gate: `-D` pointing at a missing file must produce a
    // clean error message + non-zero exit, not a panic.
    let missing_dict = std::env::temp_dir().join(format!(
        "zstd_pure_missing_dict_{}.dict",
        std::process::id()
    ));
    // Ensure the file truly doesn't exist.
    let _ = std::fs::remove_file(&missing_dict);

    let mut child = Command::new(bin_path())
        .args(["-c", "-q"])
        .arg("-D")
        .arg(&missing_dict)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let _ = child.stdin.as_mut().unwrap().write_all(b"payload");
    let out = child.wait_with_output().unwrap();
    assert!(
        !out.status.success(),
        "should fail when -D points at missing file, stderr: {}",
        String::from_utf8_lossy(&out.stderr),
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains(&missing_dict.display().to_string()) || stderr.contains("dict"),
        "error message didn't reference the missing dict: {stderr}",
    );
}

#[test]
fn cli_rejects_attached_short_dict_value() {
    let tmp = std::env::temp_dir().join(format!("zstd_pure_attached_dict_{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let dict_path = tmp.join("dict-with-c-byte.bin");
    fs::write(&dict_path, b"dictionary content with c in the path").unwrap();
    let attached_dict = format!("-D{}", dict_path.display());

    let mut child = Command::new(bin_path())
        .args(["-c", "-q"])
        .arg(attached_dict)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let _ = child.stdin.as_mut().unwrap().write_all(b"payload");
    let out = child.wait_with_output().unwrap();

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "attached -D value should be rejected"
    );
    assert!(
        stderr.contains("command cannot be separated from its argument by another command"),
        "unexpected attached -D diagnostic: {stderr}"
    );
    assert!(
        out.stdout.is_empty(),
        "rejected attached -D value must not emit compressed data"
    );

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cli_accepts_equals_short_dict_value() {
    // Upstream zstdcli.c accepts `-D=DICT`: after case 'D' advances
    // past the flag byte, NEXT_FIELD consumes an attached value only
    // when it starts with '='. Keep this distinct from the rejected
    // `-Ddict` form above.
    let tmp = std::env::temp_dir().join(format!("zstd_pure_equals_dict_{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let dict_path = tmp.join("dict.bin");
    fs::write(&dict_path, b"equals dict prefix content ".repeat(4)).unwrap();
    let dict_arg = format!("-D={}", dict_path.display());
    let payload = b"equals dict prefix content payload ".repeat(16);

    let mut comp = Command::new(bin_path())
        .args(["-c", "-q"])
        .arg(&dict_arg)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    comp.stdin.as_mut().unwrap().write_all(&payload).unwrap();
    let comp_out = comp.wait_with_output().unwrap();
    assert!(
        comp_out.status.success(),
        "-D= dict compress failed: {}",
        String::from_utf8_lossy(&comp_out.stderr)
    );

    let mut without_dict = Command::new(bin_path())
        .args(["-d", "-c", "-q", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    without_dict
        .stdin
        .as_mut()
        .unwrap()
        .write_all(&comp_out.stdout)
        .unwrap();
    let without_dict_out = without_dict.wait_with_output().unwrap();
    assert!(
        !without_dict_out.status.success(),
        "-D= compression should actually depend on the dictionary"
    );
    assert!(
        without_dict_out.stdout.is_empty(),
        "failed no-dict decode must not emit recovered payload"
    );

    let mut dec = Command::new(bin_path())
        .args(["-d", "-c", "-q"])
        .arg(&dict_arg)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    dec.stdin
        .as_mut()
        .unwrap()
        .write_all(&comp_out.stdout)
        .unwrap();
    let dec_out = dec.wait_with_output().unwrap();
    assert!(
        dec_out.status.success(),
        "-D= dict decompress failed: {}",
        String::from_utf8_lossy(&dec_out.stderr)
    );
    assert_eq!(dec_out.stdout, payload);

    if let Some(upstream) = upstream_zstd() {
        let mut upstream_dec = Command::new(upstream)
            .args(["-d", "-c", "-q"])
            .arg(&dict_arg)
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        upstream_dec
            .stdin
            .as_mut()
            .unwrap()
            .write_all(&comp_out.stdout)
            .unwrap();
        let upstream_out = upstream_dec.wait_with_output().unwrap();
        assert!(
            upstream_out.status.success(),
            "upstream -D= dict decompress failed: {}",
            String::from_utf8_lossy(&upstream_out.stderr)
        );
        assert_eq!(upstream_out.stdout, payload);
    }

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cli_repeated_mixed_dict_forms_use_last_value() {
    // Repeated -D fields are order-sensitive in upstream. The
    // separated and -D= spellings must feed the same last-wins
    // dictionary slot for both compression and decompression.
    let tmp = temp_path("mixed_dict_fields", "dir");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let stale_dict = tmp.join("stale.dict");
    let middle_dict = tmp.join("middle.dict");
    let final_dict = tmp.join("final.dict");
    fs::write(&stale_dict, b"stale dictionary bytes ".repeat(4)).unwrap();
    fs::write(&middle_dict, b"middle dictionary bytes ".repeat(4)).unwrap();
    fs::write(&final_dict, b"final dictionary prefix bytes ".repeat(4)).unwrap();
    let middle_arg = format!("-D={}", middle_dict.display());
    let payload = b"final dictionary prefix bytes payload body ".repeat(20);

    let mut comp = Command::new(bin_path())
        .args(["-c", "-q", "-D"])
        .arg(&stale_dict)
        .arg(&middle_arg)
        .arg("-D")
        .arg(&final_dict)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    comp.stdin.as_mut().unwrap().write_all(&payload).unwrap();
    let comp_out = comp.wait_with_output().unwrap();
    assert!(
        comp_out.status.success(),
        "mixed -D forms should compress with the final dict: {}",
        String::from_utf8_lossy(&comp_out.stderr)
    );

    let mut stale_dec = Command::new(bin_path())
        .args(["-d", "-c", "-q", "-D"])
        .arg(&stale_dict)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    stale_dec
        .stdin
        .as_mut()
        .unwrap()
        .write_all(&comp_out.stdout)
        .unwrap();
    let stale_out = stale_dec.wait_with_output().unwrap();
    assert!(
        !stale_out.status.success(),
        "compressed stream unexpectedly decoded with stale first dict"
    );
    assert!(
        stale_out.stdout.is_empty(),
        "failed stale-dict decode must not emit payload"
    );

    let final_arg = format!("-D={}", final_dict.display());
    let mut final_dec = Command::new(bin_path())
        .args(["-d", "-c", "-q", "-D"])
        .arg(&stale_dict)
        .arg(&middle_arg)
        .arg(&final_arg)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    final_dec
        .stdin
        .as_mut()
        .unwrap()
        .write_all(&comp_out.stdout)
        .unwrap();
    let final_out = final_dec.wait_with_output().unwrap();
    assert!(
        final_out.status.success(),
        "mixed -D forms should decode with final dict: {}",
        String::from_utf8_lossy(&final_out.stderr)
    );
    assert_eq!(final_out.stdout, payload);

    if let Some(upstream) = upstream_zstd() {
        let mut upstream_dec = Command::new(upstream)
            .args(["-d", "-c", "-q", "-D"])
            .arg(&stale_dict)
            .arg(&middle_arg)
            .arg(&final_arg)
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        upstream_dec
            .stdin
            .as_mut()
            .unwrap()
            .write_all(&comp_out.stdout)
            .unwrap();
        let upstream_out = upstream_dec.wait_with_output().unwrap();
        assert!(
            upstream_out.status.success(),
            "upstream mixed -D decode rejected final dict stream: {}",
            String::from_utf8_lossy(&upstream_out.stderr)
        );
        assert_eq!(upstream_out.stdout, payload);
    }

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cli_rejects_decompressing_non_zstd_garbage_with_nonzero_exit() {
    // Safety gate: piping arbitrary non-zstd bytes to `-d -c -` must
    // exit non-zero and emit a diagnostic, never succeed silently
    // or panic.
    let garbage = b"this is not a zstd frame at all";
    let mut child = Command::new(bin_path())
        .args(["-d", "-c", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.as_mut().unwrap().write_all(garbage).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        !out.status.success(),
        "decompressing garbage should have failed, exit={:?} stderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr),
    );
    // Should emit some diagnostic to stderr (not silent).
    assert!(
        !out.stderr.is_empty(),
        "expected an error message on stderr"
    );
    assert!(
        out.stdout.is_empty(),
        "failed garbage decode must not emit stdout"
    );
}

#[test]
fn cli_decompress_empty_and_short_inputs_match_upstream() {
    // Upstream distinguishes EOF before any header byte from a short
    // non-zstd header. Explicit pass-through follows the same split:
    // empty input is still an EOF error, while 1..=3 unrecognized
    // bytes are copied when --pass-through is selected.
    let cases: &[(&str, &[&str], &[u8], bool, &[u8], &str)] = &[
        (
            "empty standard decode",
            &["-d", "-c", "-q", "-"],
            b"",
            false,
            b"",
            "unexpected end of file",
        ),
        (
            "one-byte standard decode",
            &["-d", "-c", "-q", "-"],
            b"a",
            false,
            b"",
            "unknown header",
        ),
        (
            "two-byte standard decode",
            &["-d", "-c", "-q", "-"],
            b"ab",
            false,
            b"",
            "unknown header",
        ),
        (
            "three-byte standard decode",
            &["-d", "-c", "-q", "-"],
            b"abc",
            false,
            b"",
            "unknown header",
        ),
        (
            "empty explicit pass-through",
            &["-d", "-c", "-q", "--pass-through", "-"],
            b"",
            false,
            b"",
            "unexpected end of file",
        ),
        (
            "one-byte explicit pass-through",
            &["-d", "-c", "-q", "--pass-through", "-"],
            b"a",
            true,
            b"a",
            "",
        ),
        (
            "two-byte explicit pass-through",
            &["-d", "-c", "-q", "--pass-through", "-"],
            b"ab",
            true,
            b"ab",
            "",
        ),
        (
            "three-byte explicit pass-through",
            &["-d", "-c", "-q", "--pass-through", "-"],
            b"abc",
            true,
            b"abc",
            "",
        ),
    ];

    for &(label, args, input, should_succeed, expected_stdout, stderr_needle) in cases {
        let ours = run_with_stdin(&bin_path(), args, input);
        assert_eq!(
            ours.status.success(),
            should_succeed,
            "{label}: status mismatch, stderr={}",
            String::from_utf8_lossy(&ours.stderr)
        );
        assert_eq!(
            ours.stdout, expected_stdout,
            "{label}: stdout did not match expected CLI behavior"
        );
        if should_succeed {
            assert!(
                ours.stderr.is_empty(),
                "{label}: successful quiet pass-through wrote stderr: {}",
                String::from_utf8_lossy(&ours.stderr)
            );
        } else {
            assert!(
                String::from_utf8_lossy(&ours.stderr).contains(stderr_needle),
                "{label}: expected `{stderr_needle}` diagnostic, got: {}",
                String::from_utf8_lossy(&ours.stderr)
            );
        }

        if let Some(upstream) = upstream_zstd() {
            let theirs = run_with_stdin(&upstream, args, input);
            assert_eq!(
                ours.status.success(),
                theirs.status.success(),
                "{label}: status diverged from upstream, ours stderr={} upstream stderr={}",
                String::from_utf8_lossy(&ours.stderr),
                String::from_utf8_lossy(&theirs.stderr)
            );
            assert_eq!(
                ours.stdout, theirs.stdout,
                "{label}: stdout diverged from upstream"
            );
            if should_succeed {
                assert!(
                    theirs.stderr.is_empty(),
                    "{label}: upstream quiet pass-through wrote stderr: {}",
                    String::from_utf8_lossy(&theirs.stderr)
                );
            } else {
                assert!(
                    String::from_utf8_lossy(&theirs.stderr).contains(stderr_needle),
                    "{label}: upstream diagnostic no longer contains `{stderr_needle}`: {}",
                    String::from_utf8_lossy(&theirs.stderr)
                );
            }
        }
    }
}

#[test]
fn cli_test_empty_and_short_inputs_match_upstream() {
    // Test mode validates to a null output. It shares the same
    // empty-vs-short split as decompression, but successful
    // pass-through validation must not copy the input to stdout.
    let cases: &[(&str, &[&str], &[u8], bool, &str)] = &[
        (
            "empty standard test",
            &["-t", "-q", "-"],
            b"",
            false,
            "unexpected end of file",
        ),
        (
            "one-byte standard test",
            &["-t", "-q", "-"],
            b"a",
            false,
            "unknown header",
        ),
        (
            "two-byte standard test",
            &["-t", "-q", "-"],
            b"ab",
            false,
            "unknown header",
        ),
        (
            "three-byte standard test",
            &["-t", "-q", "-"],
            b"abc",
            false,
            "unknown header",
        ),
        (
            "empty explicit pass-through test",
            &["-t", "-q", "--pass-through", "-"],
            b"",
            false,
            "unexpected end of file",
        ),
        (
            "one-byte explicit pass-through test",
            &["-t", "-q", "--pass-through", "-"],
            b"a",
            true,
            "",
        ),
        (
            "two-byte explicit pass-through test",
            &["-t", "-q", "--pass-through", "-"],
            b"ab",
            true,
            "",
        ),
        (
            "three-byte explicit pass-through test",
            &["-t", "-q", "--pass-through", "-"],
            b"abc",
            true,
            "",
        ),
    ];

    for &(label, args, input, should_succeed, stderr_needle) in cases {
        let ours = run_with_stdin(&bin_path(), args, input);
        assert_eq!(
            ours.status.success(),
            should_succeed,
            "{label}: status mismatch, stderr={}",
            String::from_utf8_lossy(&ours.stderr)
        );
        assert!(
            ours.stdout.is_empty(),
            "{label}: test mode must not write stdout"
        );
        if should_succeed {
            assert!(
                ours.stderr.is_empty(),
                "{label}: successful quiet test wrote stderr: {}",
                String::from_utf8_lossy(&ours.stderr)
            );
        } else {
            assert!(
                String::from_utf8_lossy(&ours.stderr).contains(stderr_needle),
                "{label}: expected `{stderr_needle}` diagnostic, got: {}",
                String::from_utf8_lossy(&ours.stderr)
            );
        }

        if let Some(upstream) = upstream_zstd() {
            let theirs = run_with_stdin(&upstream, args, input);
            assert_eq!(
                ours.status.success(),
                theirs.status.success(),
                "{label}: status diverged from upstream, ours stderr={} upstream stderr={}",
                String::from_utf8_lossy(&ours.stderr),
                String::from_utf8_lossy(&theirs.stderr)
            );
            assert_eq!(
                ours.stdout, theirs.stdout,
                "{label}: stdout diverged from upstream"
            );
            if should_succeed {
                assert!(
                    theirs.stderr.is_empty(),
                    "{label}: upstream quiet test wrote stderr: {}",
                    String::from_utf8_lossy(&theirs.stderr)
                );
            } else {
                assert!(
                    String::from_utf8_lossy(&theirs.stderr).contains(stderr_needle),
                    "{label}: upstream diagnostic no longer contains `{stderr_needle}`: {}",
                    String::from_utf8_lossy(&theirs.stderr)
                );
            }
        }
    }
}

#[test]
fn cli_force_decompress_stdout_passes_through_unrecognized_input() {
    let garbage = b"this is not compressed but should pass through with -d -c -f";
    let mut child = Command::new(bin_path())
        .args(["-d", "-c", "-f", "-q", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.as_mut().unwrap().write_all(garbage).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "pass-through should succeed, stderr: {}",
        String::from_utf8_lossy(&out.stderr),
    );
    assert_eq!(out.stdout, garbage);
}

#[test]
fn cli_no_pass_through_overrides_legacy_force_stdout_passthrough() {
    // Upstream's decompression/pass-through.sh rejects
    // `zstd --no-pass-through -dcf` on unrecognized input. The
    // explicit directive must override the legacy `-f -c` passthrough.
    let garbage = b"this is not compressed and must not pass through";
    let mut child = Command::new(bin_path())
        .args(["--no-pass-through", "-d", "-c", "-f", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.as_mut().unwrap().write_all(garbage).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        !out.status.success(),
        "--no-pass-through -dcf should reject unrecognized input"
    );
    assert!(
        out.stdout.is_empty(),
        "--no-pass-through -dcf must not copy unrecognized input"
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("unsupported")
            || String::from_utf8_lossy(&out.stderr).contains("frame"),
        "unexpected --no-pass-through diagnostic: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn cli_pass_through_directives_are_last_wins_like_upstream() {
    // --pass-through and --no-pass-through are order-sensitive
    // directives in upstream. This matters independently from legacy
    // `-f -c` pass-through: the last explicit directive controls
    // whether unrecognized input is copied or rejected.
    let garbage = b"unrecognized bytes for pass-through order";

    for (args, should_pass) in [
        (
            &["-d", "-c", "-q", "--pass-through", "--no-pass-through", "-"][..],
            false,
        ),
        (
            &["-d", "-c", "-q", "--no-pass-through", "--pass-through", "-"][..],
            true,
        ),
        (
            &[
                "-d",
                "-c",
                "-q",
                "-f",
                "--pass-through",
                "--no-pass-through",
                "-",
            ][..],
            false,
        ),
        (
            &[
                "-d",
                "-c",
                "-q",
                "-f",
                "--no-pass-through",
                "--pass-through",
                "-",
            ][..],
            true,
        ),
    ] {
        let out = run_with_stdin(&bin_path(), args, garbage);
        assert_eq!(
            out.status.success(),
            should_pass,
            "{args:?} pass-through status mismatch: stderr={}",
            String::from_utf8_lossy(&out.stderr),
        );
        if should_pass {
            assert_eq!(out.stdout, garbage, "{args:?} should copy input exactly");
        } else {
            assert!(
                out.stdout.is_empty(),
                "{args:?} rejection must not copy unrecognized input"
            );
        }

        if let Some(upstream) = upstream_zstd() {
            let upstream_out = run_with_stdin(&upstream, args, garbage);
            assert_eq!(
                out.status.success(),
                upstream_out.status.success(),
                "{args:?} status diverged from upstream: ours stderr={} upstream stderr={}",
                String::from_utf8_lossy(&out.stderr),
                String::from_utf8_lossy(&upstream_out.stderr),
            );
            assert_eq!(
                out.stdout, upstream_out.stdout,
                "{args:?} stdout diverged from upstream"
            );
        }
    }
}

#[test]
fn cli_pass_through_does_not_copy_recognized_but_malformed_compat_formats() {
    // --pass-through is for unrecognized input. A stream that has a
    // recognized non-zstd container signature must be decoded or
    // rejected; copying it would hide format-specific corruption.
    for (format, malformed) in [
        ("gzip", b"\x1f\x8b\x08\x00truncated gzip stream" as &[u8]),
        ("xz", b"\xfd\x37truncated xz stream"),
        ("lzma", b"\x5d\x00truncated lzma stream"),
        ("lz4", b"\x04\x22\x4d\x18truncated lz4 stream"),
    ] {
        let out = run_with_stdin(
            &bin_path(),
            &["-d", "-c", "--pass-through", "-q", "-"],
            malformed,
        );
        assert!(
            !out.status.success(),
            "malformed {format} must not pass through successfully"
        );
        assert!(
            out.stdout.is_empty(),
            "malformed recognized {format} was copied despite --pass-through"
        );

        if let Some(upstream) = upstream_zstd() {
            let upstream_out = run_with_stdin(
                &upstream,
                &["-d", "-c", "--pass-through", "-q", "-"],
                malformed,
            );
            if upstream_out.status.success() {
                eprintln!(
                    "vendored upstream accepted malformed {format}; skipping upstream rejection comparison"
                );
                continue;
            }
            assert!(
                upstream_out.stdout.is_empty(),
                "vendored upstream copied malformed {format} despite --pass-through"
            );
        }
    }
}

#[test]
fn cli_force_decompress_file_output_rejects_unrecognized_input() {
    use std::fs;
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let input_path = tmp.join(format!("zstd_pure_passthrough_bad_{pid}.zst"));
    let inferred_path = tmp.join(format!("zstd_pure_passthrough_bad_{pid}"));
    let explicit_path = tmp.join(format!("zstd_pure_passthrough_bad_{pid}.out"));
    let garbage = b"malformed file input must not be copied to output";

    let _ = fs::remove_file(&input_path);
    let _ = fs::remove_file(&inferred_path);
    let _ = fs::remove_file(&explicit_path);
    fs::write(&input_path, garbage).unwrap();

    let inferred = Command::new(bin_path())
        .args(["-d", "-f", "-q"])
        .arg(&input_path)
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        !inferred.status.success(),
        "inferred file output should reject malformed input",
    );
    assert!(
        !inferred_path.exists(),
        "malformed input was copied to inferred output"
    );

    let explicit = Command::new(bin_path())
        .args(["-d", "-f", "-q", "-o"])
        .arg(&explicit_path)
        .arg(&input_path)
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        !explicit.status.success(),
        "explicit file output should reject malformed input",
    );
    assert!(
        !explicit_path.exists(),
        "malformed input was copied to explicit output"
    );

    let _ = fs::remove_file(&input_path);
}

#[test]
fn cli_explicit_pass_through_decompress_file_output_copies_unrecognized_input() {
    use std::fs;
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let input_path = tmp.join(format!("zstd_pure_passthrough_src_{pid}.zst"));
    let output_path = tmp.join(format!("zstd_pure_passthrough_dst_{pid}.out"));
    let garbage = b"uncompressed bytes copied by explicit pass-through";

    let _ = fs::remove_file(&input_path);
    let _ = fs::remove_file(&output_path);
    fs::write(&input_path, garbage).unwrap();

    let out = Command::new(bin_path())
        .args(["-d", "--pass-through", "-q", "-o"])
        .arg(&output_path)
        .arg(&input_path)
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "explicit file output pass-through should succeed: {}",
        String::from_utf8_lossy(&out.stderr),
    );
    assert_eq!(
        fs::read(&output_path).unwrap(),
        garbage,
        "pass-through did not copy malformed input byte-exactly",
    );

    let _ = fs::remove_file(&input_path);
    let _ = fs::remove_file(&output_path);
}

#[test]
fn cli_rm_with_explicit_pass_through_follows_output_destination() {
    // Upstream treats explicit pass-through as a successful
    // decompression path: --rm removes the source after a file-output
    // copy, but still preserves it when the selected output is stdout.
    let tmp = temp_path("rm_passthrough", "dir");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let file_input = tmp.join("file-output.zst");
    let file_output = tmp.join("file-output.out");
    let stdout_input = tmp.join("stdout-output.zst");
    let garbage = b"explicit pass-through rm payload";
    fs::write(&file_input, garbage).unwrap();
    fs::write(&stdout_input, garbage).unwrap();

    let file_out = Command::new(bin_path())
        .args(["-d", "--pass-through", "--rm", "-q", "-o"])
        .arg(&file_output)
        .arg(&file_input)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        file_out.status.success(),
        "--rm pass-through file output failed: {}",
        String::from_utf8_lossy(&file_out.stderr)
    );
    assert!(
        file_out.stdout.is_empty(),
        "--rm pass-through file output must not write stdout"
    );
    assert_eq!(
        fs::read(&file_output).unwrap(),
        garbage,
        "--rm pass-through file output did not copy bytes"
    );
    assert!(
        !file_input.exists(),
        "--rm pass-through file output did not remove the source after success"
    );

    let stdout_out = Command::new(bin_path())
        .args(["-d", "--pass-through", "--rm", "-c", "-q"])
        .arg(&stdout_input)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        stdout_out.status.success(),
        "--rm pass-through stdout output failed: {}",
        String::from_utf8_lossy(&stdout_out.stderr)
    );
    assert_eq!(
        stdout_out.stdout, garbage,
        "--rm pass-through stdout output did not copy bytes"
    );
    assert!(
        stdout_input.exists(),
        "--rm pass-through stdout output removed the source"
    );

    if let Some(upstream) = upstream_zstd() {
        let upstream_input = tmp.join("upstream-file-output.zst");
        let upstream_output = tmp.join("upstream-file-output.out");
        fs::write(&upstream_input, garbage).unwrap();

        let upstream_out = Command::new(upstream)
            .args(["-d", "--pass-through", "--rm", "-q", "-o"])
            .arg(&upstream_output)
            .arg(&upstream_input)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(
            upstream_out.status.success(),
            "vendored upstream rejected --rm pass-through file output: {}",
            String::from_utf8_lossy(&upstream_out.stderr)
        );
        assert!(
            upstream_out.stdout.is_empty(),
            "vendored upstream wrote stdout for --rm pass-through file output"
        );
        assert_eq!(
            fs::read(&upstream_output).unwrap(),
            garbage,
            "vendored upstream copied different pass-through bytes"
        );
        assert!(
            !upstream_input.exists(),
            "vendored upstream did not remove source for --rm pass-through file output"
        );
    }

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn cli_quiet_flag_suppresses_stderr_on_file_output() {
    // `-q` must suppress the ratio / byte-count line even when a
    // file output path is given (which would otherwise trigger the
    // diagnostic). Important for pipeline contexts that read stderr.
    use std::fs;
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let input_path = tmp.join(format!("zstd_pure_q_in_{pid}.txt"));
    let output_path = tmp.join(format!("zstd_pure_q_out_{pid}.zst"));
    let payload = b"quiet flag test ".repeat(32);
    fs::write(&input_path, &payload).unwrap();
    let out = Command::new(bin_path())
        .args(["-q", "-f", "-o"])
        .arg(&output_path)
        .arg(&input_path)
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.is_empty(),
        "-q should suppress stderr but got: {stderr:?}"
    );
    let _ = fs::remove_file(&input_path);
    let _ = fs::remove_file(&output_path);
}

#[test]
fn cli_quiet_and_verbose_are_cumulative_for_display_level() {
    // Upstream increments/decrements display level for each -v/-q
    // occurrence. A -q followed by -v returns to the default level,
    // while -qqv remains quiet enough to suppress the summary line.
    use std::fs;
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let input_path = tmp.join(format!("zstd_pure_display_in_{pid}.txt"));
    let output_path = tmp.join(format!("zstd_pure_display_out_{pid}.zst"));
    let quiet_output_path = tmp.join(format!("zstd_pure_display_quiet_out_{pid}.zst"));
    let payload = b"display-level counter test ".repeat(32);
    fs::write(&input_path, &payload).unwrap();

    let default_level = Command::new(bin_path())
        .args(["-q", "-v", "-f", "-o"])
        .arg(&output_path)
        .arg(&input_path)
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        default_level.status.success(),
        "-q -v compression failed: {}",
        String::from_utf8_lossy(&default_level.stderr)
    );
    assert!(
        String::from_utf8_lossy(&default_level.stderr).contains("compressed"),
        "-q -v should restore the default summary diagnostic, got: {}",
        String::from_utf8_lossy(&default_level.stderr)
    );

    let quiet_level = Command::new(bin_path())
        .args(["-q", "-q", "-v", "-f", "-o"])
        .arg(&quiet_output_path)
        .arg(&input_path)
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        quiet_level.status.success(),
        "-qqv compression failed: {}",
        String::from_utf8_lossy(&quiet_level.stderr)
    );
    assert!(
        quiet_level.stderr.is_empty(),
        "-qqv should keep the display level below summary output, got: {}",
        String::from_utf8_lossy(&quiet_level.stderr)
    );

    let _ = fs::remove_file(&input_path);
    let _ = fs::remove_file(&output_path);
    let _ = fs::remove_file(&quiet_output_path);
}

#[test]
fn cli_decompress_verbose_prints_decompressed_diagnostic() {
    // `-d -v` over stdin should print a "decompressed N -> M" line
    // to stderr. Symmetric with the existing compress-side
    // `cli_verbose_flag_prints_ratio_to_stderr` check.
    let payload: Vec<u8> = b"the quick brown fox "
        .iter()
        .cycle()
        .take(400)
        .copied()
        .collect();
    // First compress via the CLI to produce a valid frame.
    let mut comp = Command::new(bin_path())
        .args(["-c", "-q", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    comp.stdin.as_mut().unwrap().write_all(&payload).unwrap();
    let comp_out = comp.wait_with_output().unwrap();
    assert!(comp_out.status.success());
    let compressed = comp_out.stdout;

    // Now decompress with -v; stderr must mention "decompressed"
    // and the payload.len() byte count.
    let mut dec = Command::new(bin_path())
        .args(["-d", "-c", "-v", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    dec.stdin.as_mut().unwrap().write_all(&compressed).unwrap();
    let dec_out = dec.wait_with_output().unwrap();
    assert!(dec_out.status.success());
    let stderr = String::from_utf8_lossy(&dec_out.stderr);
    assert!(
        stderr.contains("decompressed"),
        "stderr didn't include 'decompressed': {stderr}"
    );
    assert!(
        stderr.contains(&payload.len().to_string()),
        "stderr didn't mention decompressed byte count {}: {stderr}",
        payload.len()
    );
}

#[test]
fn cli_verbose_flag_prints_ratio_to_stderr() {
    let payload: Vec<u8> = b"hello verbose world. "
        .iter()
        .cycle()
        .take(400)
        .copied()
        .collect();
    let mut comp = Command::new(bin_path())
        .args(["-c", "-v", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    comp.stdin.as_mut().unwrap().write_all(&payload).unwrap();
    let out = comp.wait_with_output().unwrap();
    assert!(out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("compressed") && stderr.contains('%'),
        "stderr didn't include ratio: {stderr}"
    );
}

#[test]
fn cli_checksum_compress_and_upstream_validates() {
    // Produce a frame with `--check` and verify upstream `zstd` with
    // strict checksum validation accepts the output.
    let Some(upstream) = upstream_zstd() else {
        eprintln!("vendored upstream zstd/programs/zstd not built; skipping");
        return;
    };
    let payload: Vec<u8> = b"the brown fox jumps. "
        .iter()
        .cycle()
        .take(1200)
        .copied()
        .collect();

    let mut comp = Command::new(bin_path())
        .args(["-c", "-q", "--check", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    comp.stdin.as_mut().unwrap().write_all(&payload).unwrap();
    let comp_out = comp.wait_with_output().unwrap();
    assert!(comp_out.status.success());

    // --no-check forbids silent skip; upstream still validates the
    // checksum by default. Use --check to require presence.
    let mut dec = Command::new(&upstream)
        .args(["-d", "-c", "-q", "--check", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    dec.stdin
        .as_mut()
        .unwrap()
        .write_all(&comp_out.stdout)
        .unwrap();
    let dec_out = dec.wait_with_output().unwrap();
    assert!(
        dec_out.status.success(),
        "upstream rejected our checksum frame: stderr={}",
        String::from_utf8_lossy(&dec_out.stderr)
    );
    assert_eq!(dec_out.stdout, payload);
}

#[test]
fn cli_no_check_suppresses_checksum_flag_and_trailer() {
    // The CLI mirrors upstream: checksum is on by default, and
    // explicit `--no-check` clears the checksumFlag bit and trailer.
    let payload: Vec<u8> = b"explicit-no-check payload "
        .iter()
        .cycle()
        .take(900)
        .copied()
        .collect();

    fn compress(args: &[&str], payload: &[u8]) -> Vec<u8> {
        let mut child = Command::new(bin_path())
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.as_mut().unwrap().write_all(payload).unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "compress {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr)
        );
        out.stdout
    }

    let plain = compress(&["-c", "-q", "-"], &payload);
    let no_check = compress(&["-c", "-q", "--no-check", "-"], &payload);
    let explicit_check = compress(&["-c", "-q", "--check", "-"], &payload);
    let short_check = compress(&["-c", "-q", "-C", "-"], &payload);
    let check_then_no_check = compress(&["-c", "-q", "--check", "--no-check", "-"], &payload);
    let no_check_then_check = compress(&["-c", "-q", "--no-check", "--check", "-"], &payload);
    let no_check_then_short_check = compress(&["-c", "-q", "--no-check", "-C", "-"], &payload);
    let short_check_then_no_check = compress(&["-c", "-q", "-C", "--no-check", "-"], &payload);
    assert_eq!(
        plain, explicit_check,
        "default output should match explicit --check"
    );
    assert_eq!(
        plain, short_check,
        "default output should match explicit -C"
    );
    assert_eq!(
        plain[4] & 0b0000_0100,
        0b0000_0100,
        "default path did not set checksumFlag"
    );
    assert_eq!(
        no_check[4] & 0b0000_0100,
        0,
        "--no-check unexpectedly set checksumFlag"
    );
    assert_eq!(
        check_then_no_check[4] & 0b0000_0100,
        0,
        "--check --no-check should use the last checksum directive"
    );
    assert_eq!(
        no_check_then_check[4] & 0b0000_0100,
        0b0000_0100,
        "--no-check --check should use the last checksum directive"
    );
    assert_eq!(
        no_check_then_short_check[4] & 0b0000_0100,
        0b0000_0100,
        "--no-check -C should use the last checksum directive"
    );
    assert_eq!(
        short_check_then_no_check[4] & 0b0000_0100,
        0,
        "-C --no-check should use the last checksum directive"
    );
}

#[test]
fn cli_content_size_directives_are_last_wins() {
    // Upstream defaults to writing the content-size field only when
    // the source size is known from a file. For stdin, `--content-size`
    // has no size to record and behaves like `--no-content-size`.
    // The translated CLI should mirror both cases and use the last
    // directive when size is known.
    use std::fs;
    let payload: Vec<u8> = b"content size directive payload "
        .iter()
        .cycle()
        .take(900)
        .copied()
        .collect();

    fn compress_stdin(args: &[&str], payload: &[u8]) -> Vec<u8> {
        let mut child = Command::new(bin_path())
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.as_mut().unwrap().write_all(payload).unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "compress {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        out.stdout
    }

    fn compress_file(args: &[&str], input_path: &std::path::Path) -> Vec<u8> {
        let out = Command::new(bin_path())
            .args(args)
            .arg(input_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "compress {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        out.stdout
    }

    let input_path = temp_path("content_size", "src");
    fs::write(&input_path, &payload).unwrap();

    let default = compress_file(&["-c", "-q"], &input_path);
    let explicit = compress_file(&["-c", "-q", "--content-size"], &input_path);
    let suppressed = compress_file(&["-c", "-q", "--no-content-size"], &input_path);
    let suppressed_then_on = compress_file(
        &["-c", "-q", "--no-content-size", "--content-size"],
        &input_path,
    );
    let on_then_suppressed = compress_file(
        &["-c", "-q", "--content-size", "--no-content-size"],
        &input_path,
    );

    assert_eq!(default, explicit, "default should match --content-size");
    assert_eq!(
        suppressed_then_on, default,
        "--no-content-size --content-size should use the last directive",
    );
    assert_eq!(
        on_then_suppressed, suppressed,
        "--content-size --no-content-size should use the last directive",
    );
    assert_ne!(
        suppressed, default,
        "--no-content-size should change the frame header when size is known",
    );

    let stdin_default = compress_stdin(&["-c", "-q", "-"], &payload);
    let stdin_explicit = compress_stdin(&["-c", "-q", "--content-size", "-"], &payload);
    let stdin_suppressed = compress_stdin(&["-c", "-q", "--no-content-size", "-"], &payload);
    assert_eq!(
        stdin_default, stdin_explicit,
        "stdin default should match --content-size when no stream size is known",
    );
    assert_eq!(
        stdin_default, stdin_suppressed,
        "stdin default should omit content size like --no-content-size",
    );
    assert_eq!(
        stdin_default[4] & 0b1100_0000,
        0,
        "stdin compression must not encode a frame content size without a known source size",
    );

    for (label, frame) in [("default", &default), ("suppressed", &suppressed)] {
        let mut dec = Command::new(bin_path())
            .args(["-d", "-c", "-q", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        dec.stdin.as_mut().unwrap().write_all(frame).unwrap();
        let dec_out = dec.wait_with_output().unwrap();
        assert!(
            dec_out.status.success(),
            "{label} frame failed to decode: {}",
            String::from_utf8_lossy(&dec_out.stderr)
        );
        assert_eq!(dec_out.stdout, payload, "{label} frame decoded mismatch");
    }

    let _ = fs::remove_file(&input_path);
}

#[test]
fn cli_check_decodes_frame_without_checksum_trailer() {
    // `--check` validates a checksum when the frame carries one; it
    // must not reject valid frames that were encoded without a trailer.
    let payload: Vec<u8> = b"check accepts no-check frame "
        .iter()
        .cycle()
        .take(900)
        .copied()
        .collect();

    let mut comp = Command::new(bin_path())
        .args(["-c", "-q", "--no-check", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    comp.stdin.as_mut().unwrap().write_all(&payload).unwrap();
    let comp_out = comp.wait_with_output().unwrap();
    assert!(
        comp_out.status.success(),
        "no-check compression failed: {}",
        String::from_utf8_lossy(&comp_out.stderr)
    );

    let mut dec = Command::new(bin_path())
        .args(["-d", "-c", "-q", "--check", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    dec.stdin
        .as_mut()
        .unwrap()
        .write_all(&comp_out.stdout)
        .unwrap();
    let dec_out = dec.wait_with_output().unwrap();
    assert!(
        dec_out.status.success(),
        "--check should decode a no-check frame: {}",
        String::from_utf8_lossy(&dec_out.stderr)
    );
    assert_eq!(dec_out.stdout, payload);
}

#[test]
fn cli_no_check_skips_checksum_validation_on_decode() {
    let payload: Vec<u8> = b"decode no-check payload "
        .iter()
        .cycle()
        .take(900)
        .copied()
        .collect();

    let mut comp = Command::new(bin_path())
        .args(["-c", "-q", "--check", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    comp.stdin.as_mut().unwrap().write_all(&payload).unwrap();
    let comp_out = comp.wait_with_output().unwrap();
    assert!(
        comp_out.status.success(),
        "compress stderr: {}",
        String::from_utf8_lossy(&comp_out.stderr),
    );

    let mut corrupted = comp_out.stdout;
    let last = corrupted.last_mut().expect("checksum trailer byte");
    *last ^= 0x5a;

    let mut strict = Command::new(bin_path())
        .args(["-d", "-c", "-q", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    strict
        .stdin
        .as_mut()
        .unwrap()
        .write_all(&corrupted)
        .unwrap();
    let strict_out = strict.wait_with_output().unwrap();
    assert!(
        !strict_out.status.success(),
        "strict decode should reject checksum corruption",
    );

    let mut relaxed = Command::new(bin_path())
        .args(["-d", "-c", "-q", "--no-check", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    relaxed
        .stdin
        .as_mut()
        .unwrap()
        .write_all(&corrupted)
        .unwrap();
    let relaxed_out = relaxed.wait_with_output().unwrap();
    assert!(
        relaxed_out.status.success(),
        "--no-check decode should skip checksum validation, stderr: {}",
        String::from_utf8_lossy(&relaxed_out.stderr),
    );
    assert_eq!(relaxed_out.stdout, payload);

    let mut last_check = Command::new(bin_path())
        .args(["-d", "-c", "-q", "--no-check", "--check", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    last_check
        .stdin
        .as_mut()
        .unwrap()
        .write_all(&corrupted)
        .unwrap();
    let last_check_out = last_check.wait_with_output().unwrap();
    assert!(
        !last_check_out.status.success(),
        "--no-check --check should validate checksum corruption",
    );

    let mut last_no_check = Command::new(bin_path())
        .args(["-d", "-c", "-q", "--check", "--no-check", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    last_no_check
        .stdin
        .as_mut()
        .unwrap()
        .write_all(&corrupted)
        .unwrap();
    let last_no_check_out = last_no_check.wait_with_output().unwrap();
    assert!(
        last_no_check_out.status.success(),
        "--check --no-check should skip checksum validation, stderr: {}",
        String::from_utf8_lossy(&last_no_check_out.stderr),
    );
    assert_eq!(last_no_check_out.stdout, payload);
}

#[test]
fn cli_level1_no_check_matches_upstream_bitwise_on_representative_fixtures() {
    // Exact compressed-byte parity gate for a representative level-1
    // no-check matrix. These fixtures cover:
    // - tiny text (`lorem50.txt`)
    // - medium natural text (`lorem.txt`)
    // - pure RLE candidate (`rep100.txt`)
    // - medium binary-ish payload (`binary.bin`)
    let Some(upstream) = upstream_zstd() else {
        eprintln!("vendored upstream zstd/programs/zstd not built; skipping");
        return;
    };

    for fixture in [
        "tests/fixtures/lorem50.txt",
        "tests/fixtures/lorem.txt",
        "tests/fixtures/rep100.txt",
        "tests/fixtures/binary.bin",
    ] {
        let ours = Command::new(bin_path())
            .args(["-q", "--no-check", "-c", "-1", fixture])
            .output()
            .unwrap();
        assert!(
            ours.status.success(),
            "[{fixture}] our cli stderr: {}",
            String::from_utf8_lossy(&ours.stderr)
        );

        let theirs = Command::new(&upstream)
            .args(["-q", "--no-check", "-c", "-1", fixture])
            .output()
            .unwrap();
        assert!(
            theirs.status.success(),
            "[{fixture}] upstream stderr: {}",
            String::from_utf8_lossy(&theirs.stderr)
        );
        assert_eq!(
            ours.stdout, theirs.stdout,
            "[{fixture}] compressed bytes diverged from upstream"
        );

        let ours_decoded_by_upstream = {
            let mut dec = Command::new(&upstream)
                .args(["-d", "-q", "-c", "-"])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            dec.stdin.as_mut().unwrap().write_all(&ours.stdout).unwrap();
            dec.wait_with_output().unwrap()
        };
        assert!(
            ours_decoded_by_upstream.status.success(),
            "[{fixture}] upstream rejected our level-1 output: {}",
            String::from_utf8_lossy(&ours_decoded_by_upstream.stderr)
        );

        let theirs_decoded_by_us = {
            let mut dec = Command::new(bin_path())
                .args(["-d", "-q", "-c", "-"])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            dec.stdin
                .as_mut()
                .unwrap()
                .write_all(&theirs.stdout)
                .unwrap();
            dec.wait_with_output().unwrap()
        };
        assert!(
            theirs_decoded_by_us.status.success(),
            "[{fixture}] our CLI rejected upstream level-1 output: {}",
            String::from_utf8_lossy(&theirs_decoded_by_us.stderr)
        );

        let expected = fs::read(fixture).unwrap();
        assert_eq!(
            ours_decoded_by_upstream.stdout, expected,
            "[{fixture}] upstream decoded our output to different bytes"
        );
        assert_eq!(
            theirs_decoded_by_us.stdout, expected,
            "[{fixture}] our CLI decoded upstream output to different bytes"
        );
    }
}

#[test]
fn cli_dict_compress_and_decompress_roundtrip() {
    // Write a dict and a payload to tmp files, compress with -D, then
    // decompress with -D and verify byte-exact recovery.
    use std::fs;

    let tmp = std::env::temp_dir();
    let dict_path = tmp.join(format!("zstd_pure_dict_{}.bin", std::process::id()));
    let payload_path = tmp.join(format!("zstd_pure_payload_{}.txt", std::process::id()));

    let dict = b"the quick brown fox jumps over the lazy dog near a river. ".repeat(40);
    let payload: Vec<u8> = b"the fox jumps over the lazy dog near the river. "
        .iter()
        .cycle()
        .take(500)
        .copied()
        .collect();

    fs::write(&dict_path, &dict).expect("write dict");
    fs::write(&payload_path, &payload).expect("write payload");

    // Compress via `zstd -c -q -D dict - < payload`.
    let mut comp = Command::new(bin_path())
        .args(["-c", "-q", "-D"])
        .arg(&dict_path)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn compressor");
    comp.stdin.as_mut().unwrap().write_all(&payload).unwrap();
    let comp_out = comp.wait_with_output().expect("wait compressor");
    assert!(
        comp_out.status.success(),
        "dict compress stderr: {}",
        String::from_utf8_lossy(&comp_out.stderr)
    );
    let compressed = comp_out.stdout;

    // Decompress via `zstd -d -c -q -D dict - < compressed`.
    let mut dec = Command::new(bin_path())
        .args(["-d", "-c", "-q", "-D"])
        .arg(&dict_path)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn decompressor");
    dec.stdin.as_mut().unwrap().write_all(&compressed).unwrap();
    let dec_out = dec.wait_with_output().expect("wait decompressor");
    assert!(
        dec_out.status.success(),
        "dict decompress stderr: {}",
        String::from_utf8_lossy(&dec_out.stderr)
    );
    assert_eq!(dec_out.stdout, payload, "dict CLI roundtrip mismatch");

    let _ = fs::remove_file(&dict_path);
    let _ = fs::remove_file(&payload_path);
}

#[test]
fn cli_compress_then_decompress_roundtrips_via_stdin() {
    // Generate a medium-size compressible payload.
    let payload: Vec<u8> = b"The quick brown fox jumps over the lazy dog. "
        .iter()
        .cycle()
        .take(1500)
        .copied()
        .collect();

    // Compress via `zstd -c -`.
    let mut comp = Command::new(bin_path())
        .args(["-c", "-q", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn zstd compressor");
    comp.stdin
        .as_mut()
        .unwrap()
        .write_all(&payload)
        .expect("write payload");
    let comp_out = comp.wait_with_output().expect("wait compressor");
    assert!(
        comp_out.status.success(),
        "compress exit={:?} stderr={}",
        comp_out.status,
        String::from_utf8_lossy(&comp_out.stderr)
    );
    let compressed = comp_out.stdout;
    assert!(
        compressed.len() < payload.len(),
        "expected compressed ({}) < payload ({})",
        compressed.len(),
        payload.len()
    );

    // Decompress via `zstd -d -c -`.
    let mut dec = Command::new(bin_path())
        .args(["-d", "-c", "-q", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn zstd decompressor");
    dec.stdin
        .as_mut()
        .unwrap()
        .write_all(&compressed)
        .expect("write compressed");
    let dec_out = dec.wait_with_output().expect("wait decompressor");
    assert!(
        dec_out.status.success(),
        "decompress exit={:?} stderr={}",
        dec_out.status,
        String::from_utf8_lossy(&dec_out.stderr)
    );
    assert_eq!(dec_out.stdout, payload, "CLI roundtrip mismatch");
}

#[test]
#[allow(non_snake_case)]
fn cli_long_version_matches_library_ZSTD_VERSION_STRING() {
    // The CLI's `--version` / `-V` long form advertises the upstream
    // libzstd version the port mirrors. Pin that it matches the
    // library-side constant so a future version bump can't drift
    // the two out of sync, and keep the upstream short spelling as
    // an exact alias.
    let expected = zstd_pure_rs::common::zstd_common::ZSTD_VERSION_STRING;
    let mut canonical = None;
    for flag in ["--version", "-V"] {
        let out = Command::new(bin_path())
            .arg(flag)
            .output()
            .unwrap_or_else(|e| panic!("{flag} invocation failed: {e}"));
        assert!(
            out.status.success(),
            "{flag} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            out.stderr.is_empty(),
            "{flag} should print version to stdout only: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8(out.stdout).expect("utf-8 version");
        assert!(
            stdout.contains(expected),
            "{flag} output `{stdout}` does not contain `{expected}`",
        );
        if let Some(canonical) = &canonical {
            assert_eq!(
                &stdout, canonical,
                "{flag} output should match --version exactly",
            );
        } else {
            canonical = Some(stdout);
        }
    }
}

#[test]
#[allow(non_snake_case)]
fn cli_version_exits_left_to_right_like_upstream() {
    // Upstream exits immediately when it reaches -V/--version, so
    // later malformed arguments are ignored. Earlier field-option
    // errors still fire before -V can be processed.
    let expected = zstd_pure_rs::common::zstd_common::ZSTD_VERSION_STRING;

    let cases: &[(&[&str], bool, &str)] = &[
        (&["-V", "--level", "9"], true, "normal"),
        (&["-qV", "--format", "bad"], true, "quiet"),
        (&["-vV"], true, "verbose"),
        (&["-D", "-V"], false, "field_error"),
    ];

    for &(args, should_succeed, shape) in cases {
        let ours = Command::new(bin_path())
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert_eq!(
            ours.status.success(),
            should_succeed,
            "{args:?} exit mismatch: stdout={} stderr={}",
            String::from_utf8_lossy(&ours.stdout),
            String::from_utf8_lossy(&ours.stderr),
        );

        match shape {
            "normal" => {
                assert!(
                    ours.stderr.is_empty(),
                    "{args:?} version should not write stderr: {}",
                    String::from_utf8_lossy(&ours.stderr)
                );
                let stdout = String::from_utf8_lossy(&ours.stdout);
                assert!(
                    stdout.contains("Zstandard CLI") && stdout.contains(expected),
                    "{args:?} normal version output had wrong shape: {stdout}"
                );
                assert!(
                    !stdout.contains("supports:"),
                    "{args:?} normal version output should not include verbose support list: {stdout}"
                );
            }
            "quiet" => {
                assert!(
                    ours.stderr.is_empty(),
                    "{args:?} quiet version should not write stderr: {}",
                    String::from_utf8_lossy(&ours.stderr)
                );
                assert_eq!(
                    String::from_utf8_lossy(&ours.stdout),
                    format!("{expected}\n"),
                    "{args:?} quiet version output should be the bare version"
                );
            }
            "verbose" => {
                assert!(
                    ours.stderr.is_empty(),
                    "{args:?} verbose version should not write stderr: {}",
                    String::from_utf8_lossy(&ours.stderr)
                );
                let stdout = String::from_utf8_lossy(&ours.stdout);
                assert!(
                    stdout.contains("Zstandard CLI")
                        && stdout.contains(expected)
                        && stdout.contains("supports:"),
                    "{args:?} verbose version output had wrong shape: {stdout}"
                );
            }
            "field_error" => {
                assert!(
                    ours.stdout.is_empty(),
                    "{args:?} pre-version field error should not print version stdout"
                );
                assert!(
                    String::from_utf8_lossy(&ours.stderr).contains(
                        "command cannot be separated from its argument by another command"
                    ),
                    "{args:?} emitted unexpected pre-version error: {}",
                    String::from_utf8_lossy(&ours.stderr)
                );
            }
            _ => unreachable!(),
        }

        if let Some(upstream) = upstream_zstd() {
            let theirs = Command::new(upstream)
                .args(args)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .unwrap();
            assert_eq!(
                ours.status.success(),
                theirs.status.success(),
                "{args:?} exit status diverged from upstream: ours stdout={} stderr={} upstream stdout={} stderr={}",
                String::from_utf8_lossy(&ours.stdout),
                String::from_utf8_lossy(&ours.stderr),
                String::from_utf8_lossy(&theirs.stdout),
                String::from_utf8_lossy(&theirs.stderr),
            );
            assert_eq!(
                ours.stdout.is_empty(),
                theirs.stdout.is_empty(),
                "{args:?} stdout presence diverged from upstream",
            );
            assert_eq!(
                ours.stderr.is_empty(),
                theirs.stderr.is_empty(),
                "{args:?} stderr presence diverged from upstream",
            );
            if shape != "field_error" {
                assert_eq!(
                    ours.stdout, theirs.stdout,
                    "{args:?} version stdout diverged from upstream"
                );
                assert_eq!(
                    ours.stderr, theirs.stderr,
                    "{args:?} version stderr diverged from upstream"
                );
            } else {
                assert!(
                    String::from_utf8_lossy(&theirs.stderr).contains(
                        "command cannot be separated from its argument by another command"
                    ),
                    "{args:?} upstream emitted unexpected pre-version error: {}",
                    String::from_utf8_lossy(&theirs.stderr)
                );
            }
        }
    }
}

#[test]
#[allow(non_snake_case)]
fn cli_help_and_version_terminal_directives_are_first_wins_like_upstream() {
    // Upstream stops at the first terminal directive among help and
    // version flags. Later terminal flags or malformed field options
    // are ignored once help/version has already been selected.
    let expected_version = zstd_pure_rs::common::zstd_common::ZSTD_VERSION_STRING;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum TerminalOutput {
        Help,
        Version,
    }

    fn classify_stdout(stdout: &[u8], expected_version: &str) -> TerminalOutput {
        let stdout = String::from_utf8_lossy(stdout);
        if stdout.contains("Usage:") {
            TerminalOutput::Help
        } else if stdout.contains(expected_version) {
            TerminalOutput::Version
        } else {
            panic!("terminal output was neither help nor version: {stdout}");
        }
    }

    let cases: &[(&[&str], TerminalOutput)] = &[
        (&["--help", "--version"], TerminalOutput::Help),
        (&["--help", "--format"], TerminalOutput::Help),
        (&["-hV"], TerminalOutput::Help),
        (&["-H", "--version"], TerminalOutput::Help),
        (&["--version", "--help"], TerminalOutput::Version),
        (&["-V", "-H"], TerminalOutput::Version),
    ];

    for &(args, expected_terminal) in cases {
        let ours = Command::new(bin_path())
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(
            ours.status.success(),
            "{args:?} terminal directive failed: {}",
            String::from_utf8_lossy(&ours.stderr)
        );
        assert!(
            ours.stderr.is_empty(),
            "{args:?} terminal directive wrote stderr: {}",
            String::from_utf8_lossy(&ours.stderr)
        );
        assert_eq!(
            classify_stdout(&ours.stdout, expected_version),
            expected_terminal,
            "{args:?} did not honor first terminal directive",
        );

        if let Some(upstream) = upstream_zstd() {
            let theirs = Command::new(upstream)
                .args(args)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .unwrap();
            assert_eq!(
                ours.status.success(),
                theirs.status.success(),
                "{args:?} terminal status diverged from upstream: ours stderr={} upstream stderr={}",
                String::from_utf8_lossy(&ours.stderr),
                String::from_utf8_lossy(&theirs.stderr),
            );
            assert_eq!(
                ours.stderr.is_empty(),
                theirs.stderr.is_empty(),
                "{args:?} terminal stderr presence diverged from upstream",
            );
            assert_eq!(
                classify_stdout(&ours.stdout, expected_version),
                classify_stdout(&theirs.stdout, expected_version),
                "{args:?} terminal output class diverged from upstream",
            );
        }
    }
}

#[test]
fn cli_help_advertises_magicless_flag() {
    // Discoverability: the `--magicless` flag must show up in `--help`
    // so CLI users know the option exists. Cheap smoke test guarding
    // against accidentally removing it from the clap definition.
    let out = Command::new(bin_path())
        .arg("--help")
        .output()
        .expect("help invocation");
    assert!(out.status.success());
    let help = String::from_utf8(out.stdout).expect("utf-8 help");
    assert!(
        help.contains("--magicless"),
        "--help output did not mention --magicless: {help}",
    );
}

#[test]
fn cli_magicless_works_across_compression_levels() {
    // Magicless format must work across the full level range our
    // port supports. A level-specific bug (e.g. strategy gate that
    // only checks format on the fast path) would show up here as a
    // failing roundtrip at some level.
    for level in [1, 3, 5, 9, 15, 22] {
        let payload = format!("magicless-level-{level}-sweep payload ")
            .repeat(5)
            .into_bytes();

        let mut comp = Command::new(bin_path())
            .args(["-c", "-q", "--magicless", &format!("-{level}"), "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn compressor");
        comp.stdin.as_mut().unwrap().write_all(&payload).unwrap();
        let comp_out = comp.wait_with_output().expect("wait");
        assert!(
            comp_out.status.success(),
            "[level {level}] compress stderr: {}",
            String::from_utf8_lossy(&comp_out.stderr),
        );
        let compressed = comp_out.stdout;
        let leading =
            u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]);
        assert_ne!(
            leading, ZSTD_MAGIC,
            "[level {level}] magicless leaked magic",
        );

        let mut dec = Command::new(bin_path())
            .args(["-d", "-c", "-q", "--magicless", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn decompressor");
        dec.stdin.as_mut().unwrap().write_all(&compressed).unwrap();
        let dec_out = dec.wait_with_output().expect("wait");
        assert!(
            dec_out.status.success(),
            "[level {level}] decompress stderr: {}",
            String::from_utf8_lossy(&dec_out.stderr),
        );
        assert_eq!(
            dec_out.stdout, payload,
            "[level {level}] roundtrip mismatch",
        );
    }
}

#[test]
fn cli_magicless_plus_dict_plus_check_roundtrip_through_self() {
    // Triple-flag integration: `--magicless` + `-D` + `--check`.
    // Proves the CLI composes all three at once — no magic prefix,
    // dict-bearing back-ref history, XXH64 trailer validated on
    // decode. The interesting wire-order guarantees are:
    //   - no 4-byte magic prefix (frame starts with FHD)
    //   - dictID propagated into the frame header (when applicable)
    //   - XXH64 trailer appended after the block stream
    // Any drift in either format threading or flag handling breaks
    // this gate.
    use std::env;
    use std::fs;
    let tmp = env::temp_dir();
    let pid = std::process::id();
    let dict_path = tmp.join(format!("cli-magicless-triple-{pid}.dict"));
    let payload_path = tmp.join(format!("cli-magicless-triple-{pid}.src"));
    let compressed_path = tmp.join(format!("cli-magicless-triple-{pid}.zst"));
    let _ = fs::remove_file(&compressed_path);

    let dict = b"triple-flag-dict-prefix-bytes ".repeat(2);
    let payload = b"triple-flag-dict-prefix-bytes payload body ".repeat(4);
    fs::write(&dict_path, &dict).unwrap();
    fs::write(&payload_path, &payload).unwrap();

    let comp = Command::new(bin_path())
        .args([
            "-q",
            "--magicless",
            "--check",
            "-D",
            dict_path.to_str().unwrap(),
            "-o",
            compressed_path.to_str().unwrap(),
            payload_path.to_str().unwrap(),
        ])
        .output()
        .expect("spawn compressor");
    assert!(
        comp.status.success(),
        "compress stderr: {}",
        String::from_utf8_lossy(&comp.stderr),
    );
    let compressed = fs::read(&compressed_path).expect("read compressed");
    let leading = u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]);
    assert_ne!(leading, ZSTD_MAGIC, "triple-flag leaked magic");

    let mut dec = Command::new(bin_path())
        .args([
            "-d",
            "-c",
            "-q",
            "--magicless",
            "-D",
            dict_path.to_str().unwrap(),
            "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn decompressor");
    dec.stdin.as_mut().unwrap().write_all(&compressed).unwrap();
    let dec_out = dec.wait_with_output().expect("wait");
    assert!(
        dec_out.status.success(),
        "decompress stderr: {}",
        String::from_utf8_lossy(&dec_out.stderr),
    );
    assert_eq!(dec_out.stdout, payload, "triple-flag roundtrip mismatch");

    let _ = fs::remove_file(&dict_path);
    let _ = fs::remove_file(&payload_path);
    let _ = fs::remove_file(&compressed_path);
}

#[test]
fn cli_dict_plus_check_emits_xxh64_trailer() {
    // Regression gate for a bug where `-D <dict>` + `--check`
    // silently dropped the checksum flag. The current CLI routes
    // dictionary compression through a `CCtx`, `ZSTD_CCtx_loadDictionary`,
    // and `ZSTD_compress2`, so `ZSTD_c_checksumFlag` is honored. Proof:
    // output with
    // `-D + --check` should be exactly 4 bytes longer than
    // `-D + --no-check` (the XXH64 trailer), and contentSize bits of
    // the FHD byte must encode the checksumFlag.
    use std::fs;
    let dict_path = temp_path("dict_check", "dict");
    let payload_path = temp_path("dict_check", "src");
    let plain_out = temp_path("dict_check", "plain.zst");
    let check_out = temp_path("dict_check", "check.zst");
    let _ = fs::remove_file(&plain_out);
    let _ = fs::remove_file(&check_out);

    let dict = b"dict-check-prefix-bytes ".repeat(2);
    let payload = b"dict-check-prefix-bytes payload body ".repeat(4);
    fs::write(&dict_path, &dict).unwrap();
    fs::write(&payload_path, &payload).unwrap();

    for (check_flag, out_path) in [("--no-check", &plain_out), ("--check", &check_out)] {
        let comp = Command::new(bin_path())
            .args(["-q", check_flag, "-D"])
            .arg(&dict_path)
            .arg("-o")
            .arg(out_path)
            .arg(&payload_path)
            .output()
            .expect("spawn compressor");
        assert!(
            comp.status.success(),
            "compress {check_flag} stderr: {}",
            String::from_utf8_lossy(&comp.stderr),
        );
    }

    let plain = fs::read(&plain_out).expect("read plain");
    let with_check = fs::read(&check_out).expect("read check");

    assert_eq!(
        with_check.len(),
        plain.len() + 4,
        "checksum-flag should add exactly 4 bytes (XXH64 trailer) — \
         plain {} bytes, check {} bytes",
        plain.len(),
        with_check.len(),
    );
    // FHD byte (index 4, just after the 4-byte magic) encodes
    // `checksumFlag` at bit 2. With `--check` it must be set; without
    // it, clear. Other FHD bits (content-size flag, single-segment,
    // dictID flags) are unchanged.
    assert_eq!(
        with_check[4] & 0b0000_0100,
        0b0000_0100,
        "FHD byte checksumFlag bit not set with --check",
    );
    assert_eq!(
        plain[4] & 0b0000_0100,
        0,
        "FHD byte checksumFlag bit leaked into plain output",
    );

    let _ = fs::remove_file(&dict_path);
    let _ = fs::remove_file(&payload_path);
    let _ = fs::remove_file(&plain_out);
    let _ = fs::remove_file(&check_out);
}

#[test]
fn cli_magicless_plus_check_roundtrip_through_self() {
    // Combined-flags integration: `--magicless` + `--check` should
    // compose. The frame has no 4-byte magic AND a trailing XXH64
    // checksum. Proves the header-write path honors both (magicless
    // skips magic, `contentSize` flag reflects checksum request) AND
    // the streaming decoder validates the XXH64 over magicless bytes
    // without false-positive mismatch.
    let payload = b"magicless-plus-check-payload ".repeat(8);

    let mut comp = Command::new(bin_path())
        .args(["-c", "-q", "--magicless", "--check", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn compressor");
    comp.stdin.as_mut().unwrap().write_all(&payload).unwrap();
    let comp_out = comp.wait_with_output().expect("wait");
    assert!(
        comp_out.status.success(),
        "compress stderr: {}",
        String::from_utf8_lossy(&comp_out.stderr),
    );
    let compressed = comp_out.stdout;
    assert!(compressed.len() >= 4);
    let leading = u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]);
    assert_ne!(leading, ZSTD_MAGIC, "magicless+check leaked magic");

    let mut dec = Command::new(bin_path())
        .args(["-d", "-c", "-q", "--magicless", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn decompressor");
    dec.stdin.as_mut().unwrap().write_all(&compressed).unwrap();
    let dec_out = dec.wait_with_output().expect("wait");
    assert!(
        dec_out.status.success(),
        "decompress stderr: {}",
        String::from_utf8_lossy(&dec_out.stderr),
    );
    assert_eq!(dec_out.stdout, payload, "roundtrip mismatch");
}

#[test]
fn cli_magicless_decompresses_frame_with_absent_fcs() {
    // Magicless frame with no frame content size:
    // FHD=0x00 (fcsID=0, singleSegment=0), wlByte=0x40, then an RLE
    // last block. The decoded size is intentionally larger than
    // compressed_len * 32, so the CLI must use the format-aware frame
    // walker bound rather than the old magicless heuristic.
    let mut frame = Vec::new();
    frame.push(0x00);
    frame.push(0x40);
    let decoded_len = 1024u32;
    let bh = (decoded_len << 3) | (1 << 1) | 1;
    frame.push((bh & 0xFF) as u8);
    frame.push(((bh >> 8) & 0xFF) as u8);
    frame.push(((bh >> 16) & 0xFF) as u8);
    frame.push(b'Z');

    let mut dec = Command::new(bin_path())
        .args(["-d", "-c", "-q", "--magicless", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn decompressor");
    dec.stdin.as_mut().unwrap().write_all(&frame).unwrap();
    let dec_out = dec.wait_with_output().expect("wait");
    assert!(
        dec_out.status.success(),
        "decompress stderr: {}",
        String::from_utf8_lossy(&dec_out.stderr),
    );
    assert_eq!(dec_out.stdout, vec![b'Z'; decoded_len as usize]);
}

#[test]
fn cli_magicless_plus_dict_roundtrip_through_self() {
    // Combined-flags integration: `--magicless` + `-D` should compose
    // cleanly. Proves the CLI threads the format through the dict
    // path (via `ZSTD_c_format`, `ZSTD_CCtx_loadDictionary`, and
    // `ZSTD_compress2`) and the symmetric decode (magicless dctx +
    // dict decode).
    use std::fs;
    let dict_path = temp_path("magicless_dict_roundtrip", "dict");
    let payload_path = temp_path("magicless_dict_roundtrip", "src");
    let compressed_path = temp_path("magicless_dict_roundtrip", "zst");
    let _ = fs::remove_file(&compressed_path);

    let dict = b"dictprefix-bytes-matching-payload-some-content".repeat(2);
    let payload = b"dictprefix-bytes-matching-payload body content ".repeat(4);
    fs::write(&dict_path, &dict).unwrap();
    fs::write(&payload_path, &payload).unwrap();

    // Compress with --magicless + -D.
    let comp = Command::new(bin_path())
        .args([
            "-q",
            "--magicless",
            "-D",
            dict_path.to_str().unwrap(),
            "-o",
            compressed_path.to_str().unwrap(),
            payload_path.to_str().unwrap(),
        ])
        .output()
        .expect("spawn compressor");
    assert!(
        comp.status.success(),
        "compress stderr: {}",
        String::from_utf8_lossy(&comp.stderr),
    );
    let compressed = fs::read(&compressed_path).expect("read compressed");
    // No zstd1 magic prefix.
    assert!(compressed.len() >= 4);
    let leading = u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]);
    assert_ne!(leading, ZSTD_MAGIC, "magicless+dict leaked magic");

    // Decompress with --magicless + -D via stdin/stdout.
    let mut dec = Command::new(bin_path())
        .args([
            "-d",
            "-c",
            "-q",
            "--magicless",
            "-D",
            dict_path.to_str().unwrap(),
            "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn decompressor");
    dec.stdin.as_mut().unwrap().write_all(&compressed).unwrap();
    let dec_out = dec.wait_with_output().expect("wait");
    assert!(
        dec_out.status.success(),
        "decompress stderr: {}",
        String::from_utf8_lossy(&dec_out.stderr),
    );
    assert_eq!(dec_out.stdout, payload, "roundtrip mismatch");

    // Cleanup.
    let _ = fs::remove_file(&dict_path);
    let _ = fs::remove_file(&payload_path);
    let _ = fs::remove_file(&compressed_path);
}

#[test]
fn cli_magicless_output_rejected_by_standard_decompress() {
    // Complement to the roundtrip test: a magicless frame must fail
    // cleanly when decompressed without `--magicless`. Otherwise the
    // flag would be asymmetric (standard decoder silently accepting
    // magicless bytes as zstd1 would indicate a header-parsing bug).
    let payload = b"reject-magicless-without-flag ".repeat(4);

    let mut comp = Command::new(bin_path())
        .args(["-c", "-q", "--magicless", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn compressor");
    comp.stdin.as_mut().unwrap().write_all(&payload).unwrap();
    let comp_out = comp.wait_with_output().expect("wait compressor");
    assert!(comp_out.status.success());
    let compressed = comp_out.stdout;

    // Try decompressing WITHOUT --magicless — should fail.
    let mut dec = Command::new(bin_path())
        .args(["-d", "-c", "-q", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn decompressor");
    dec.stdin.as_mut().unwrap().write_all(&compressed).unwrap();
    let dec_out = dec.wait_with_output().expect("wait decompressor");
    assert!(
        !dec_out.status.success(),
        "standard decompress unexpectedly accepted magicless frame",
    );
}

#[test]
fn cli_magicless_roundtrip_through_self() {
    // End-to-end gate for the `--magicless` flag. Compress with our
    // CLI in magicless mode, assert the output lacks the 4-byte zstd1
    // magic prefix, then pipe it back through our CLI with the same
    // flag and verify the original payload is recovered.
    let payload = b"magicless-cli-roundtrip payload content repeated ".repeat(10);

    // Compress with --magicless.
    let mut comp = Command::new(bin_path())
        .args(["-c", "-q", "--magicless", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn magicless compressor");
    comp.stdin.as_mut().unwrap().write_all(&payload).unwrap();
    let comp_out = comp.wait_with_output().expect("wait compressor");
    assert!(
        comp_out.status.success(),
        "compress stderr: {}",
        String::from_utf8_lossy(&comp_out.stderr),
    );
    let compressed = comp_out.stdout;
    // First 4 bytes must NOT be the zstd1 magic (0xFD2FB528 LE).
    assert!(compressed.len() >= 4, "compressed output too short");
    let leading = u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]);
    assert_ne!(
        leading, ZSTD_MAGIC,
        "--magicless output leaked a zstd1 magic prefix",
    );

    // Decompress with --magicless.
    let mut dec = Command::new(bin_path())
        .args(["-d", "-c", "-q", "--magicless", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn magicless decompressor");
    dec.stdin
        .as_mut()
        .unwrap()
        .write_all(&compressed)
        .expect("write compressed");
    let dec_out = dec.wait_with_output().expect("wait decompressor");
    assert!(
        dec_out.status.success(),
        "decompress exit={:?} stderr={}",
        dec_out.status,
        String::from_utf8_lossy(&dec_out.stderr),
    );
    assert_eq!(dec_out.stdout, payload, "magicless CLI roundtrip mismatch",);
}

#[test]
fn cli_fastq_fixture_dict_roundtrip_and_upstream_cross_decode() {
    use std::fs;

    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let dict_path = tmp.join(format!("zstd-pure-fastq-dict-{pid}.dict"));
    let payload_path = tmp.join(format!("zstd-pure-fastq-payload-{pid}.fq"));

    let fastq = include_bytes!("fixtures/small.fastq");
    let dict = &fastq[..512];
    let payload = fastq.repeat(24);
    fs::write(&dict_path, dict).expect("write FASTQ dict");
    fs::write(&payload_path, &payload).expect("write FASTQ payload");

    let compressed = Command::new(bin_path())
        .args(["-c", "-q", "-D"])
        .arg(&dict_path)
        .arg(&payload_path)
        .output()
        .expect("spawn FASTQ dict compressor");
    assert!(
        compressed.status.success(),
        "FASTQ dict compress stderr: {}",
        String::from_utf8_lossy(&compressed.stderr)
    );

    let mut dec = Command::new(bin_path())
        .args(["-d", "-c", "-q", "-D"])
        .arg(&dict_path)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn FASTQ dict decompressor");
    dec.stdin
        .as_mut()
        .unwrap()
        .write_all(&compressed.stdout)
        .unwrap();
    let dec_out = dec.wait_with_output().expect("wait FASTQ dict decompress");
    assert!(
        dec_out.status.success(),
        "FASTQ dict decompress stderr: {}",
        String::from_utf8_lossy(&dec_out.stderr)
    );
    assert_eq!(dec_out.stdout, payload);

    if let Some(upstream) = upstream_zstd() {
        let mut upstream_dec = Command::new(upstream)
            .args(["-d", "-c", "-q", "-D"])
            .arg(&dict_path)
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn upstream FASTQ dict decompressor");
        upstream_dec
            .stdin
            .as_mut()
            .unwrap()
            .write_all(&compressed.stdout)
            .unwrap();
        let upstream_out = upstream_dec.wait_with_output().expect("wait upstream");
        assert!(
            upstream_out.status.success(),
            "upstream rejected FASTQ dict frame: {}",
            String::from_utf8_lossy(&upstream_out.stderr)
        );
        assert_eq!(upstream_out.stdout, payload);
    }

    let _ = fs::remove_file(&dict_path);
    let _ = fs::remove_file(&payload_path);
}
