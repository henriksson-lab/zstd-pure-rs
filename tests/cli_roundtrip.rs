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

const ZSTD_MAGIC: u32 = 0xFD2FB528;

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

/// Returns Some(path) if upstream `zstd` is available on $PATH.
fn upstream_zstd() -> Option<PathBuf> {
    let out = Command::new("which").arg("zstd").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let p = PathBuf::from(s.trim());
    if p.as_os_str().is_empty() {
        return None;
    }
    Some(p)
}

#[test]
fn cli_output_decompressed_by_upstream_zstd() {
    // Produce a frame with our compressor, then pipe it into the
    // real upstream zstd binary (if available) and assert the round
    // trip. This is the strongest spec-compliance check we can run
    // without a full reference implementation in-tree.
    let Some(upstream) = upstream_zstd() else {
        eprintln!("upstream zstd not on $PATH; skipping");
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
}

#[test]
fn cli_decompresses_upstream_output() {
    // Complementary direction: upstream compresses, we decompress.
    let Some(upstream) = upstream_zstd() else {
        eprintln!("upstream zstd not on $PATH; skipping");
        return;
    };

    let payload: Vec<u8> = b"Mary had a little lamb, its fleece was white as snow. "
        .iter()
        .cycle()
        .take(3000)
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
}

#[test]
fn cli_compression_ratio_within_reasonable_factor_of_upstream() {
    // Sanity check: our level-1 ratio shouldn't be hugely worse than
    // upstream's level-1 on the same input. 2.0x slack absorbs the
    // missing optimizations (true lazy lookahead tuning, rowHash,
    // btopt) while catching gross regressions.
    let Some(upstream) = upstream_zstd() else {
        eprintln!("upstream zstd not on $PATH; skipping");
        return;
    };
    let payload = std::fs::read("tests/fixtures/zstd_h.txt").expect("read fixture");

    fn compress_via(bin: &std::path::Path, payload: &[u8], level: i32) -> Vec<u8> {
        // Use the local long level form and upstream's `-N` short form.
        let is_ours = bin == bin_path();
        let lvl_str = level.to_string();
        let neg_flag = format!("-{level}");
        let args: Vec<&str> = if is_ours {
            vec!["-c", "-q", "--level", &lvl_str, "-"]
        } else {
            vec!["-c", "-q", &neg_flag, "-"]
        };
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
        eprintln!("upstream zstd not on $PATH; skipping");
        return;
    };
    let payload = std::fs::read("tests/fixtures/zstd_h.txt").expect("read fixture");
    assert!(payload.len() > 128 * 1024);

    let mut comp = Command::new(bin_path())
        .args(["-c", "-q", "-L", "3", "-"])
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
        eprintln!("upstream zstd not on $PATH; skipping");
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
            .args(["-c", "-q", "-L", &level.to_string(), "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        comp.stdin.as_mut().unwrap().write_all(&payload).unwrap();
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
    }
}

#[test]
fn cli_output_decompressed_by_upstream_zstd_across_levels() {
    let Some(upstream) = upstream_zstd() else {
        eprintln!("upstream zstd not on $PATH; skipping");
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
                .args(["-c", "-q", "-L", &level.to_string(), "-"])
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
        }
    }
}

#[test]
fn cli_dict_output_decompressed_by_upstream() {
    // Our dict-compressed frame should decode correctly through
    // upstream `zstd -d -D dict` when they share the same dict.
    use std::fs;
    let Some(upstream) = upstream_zstd() else {
        eprintln!("upstream zstd not on $PATH; skipping");
        return;
    };

    let tmp = std::env::temp_dir();
    let dict_path = tmp.join(format!("zstd_pure_xdict_{}.bin", std::process::id()));
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

    let _ = fs::remove_file(&dict_path);
}

#[test]
fn cli_fuzz_random_payloads_all_decode_via_upstream() {
    // Deterministic PRNG → random payloads of varying sizes; each
    // compressed via our CLI, then decompressed via upstream zstd,
    // with byte-exact verification.
    let Some(upstream) = upstream_zstd() else {
        eprintln!("upstream zstd not on $PATH; skipping");
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
            .args(["-c", "-q", "--level", &level.to_string(), "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        comp.stdin.as_mut().unwrap().write_all(&payload).unwrap();
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
    }
}

#[test]
fn cli_file_to_file_compress_then_decompress_roundtrip() {
    use std::fs;
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let input_path = tmp.join(format!("zstd_pure_f2f_in_{pid}.txt"));
    let compressed_path = tmp.join(format!("zstd_pure_f2f_{pid}.zst"));
    let decompressed_path = tmp.join(format!("zstd_pure_f2f_out_{pid}.txt"));

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
        "--quiet",
        "--verbose",
        "--output-file",
        "--level",
        "--dict",
        "--check",
        "--no-check",
        "--magicless",
    ] {
        assert!(help.contains(needle), "--help missed {needle}: {help}");
    }
    // Short flags too.
    for needle in ["-d", "-c", "-f", "-q", "-v", "-o", "-L", "-D"] {
        assert!(
            help.contains(needle),
            "--help missed short flag {needle}: {help}"
        );
    }
    assert!(
        help.contains("Out-of-range values are clamped"),
        "--help did not describe level clamping: {help}"
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
fn cli_accepts_stdout_and_output_file_for_single_input() {
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
fn cli_rejects_single_output_file_for_multiple_inputs() {
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
        String::from_utf8_lossy(&out.stderr).contains("single input"),
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
fn cli_accepts_out_of_range_level_and_clamps_silently() {
    // `-L` is a project-local alias for `--level`; out-of-range
    // values are clamped to zstd's supported bounds. Compression must
    // succeed and the output must roundtrip.
    let payload: Vec<u8> = b"clamped-level test "
        .iter()
        .cycle()
        .take(200)
        .copied()
        .collect();
    // Negative levels are passed as `--level=-N` to avoid clap tokenizing
    // `-5` as a short flag.
    for level_arg in ["-L", "--level=-5"].iter() {
        let (arg1, arg2): (&str, Option<&str>) = if *level_arg == "-L" {
            ("-L", Some("99"))
        } else {
            (level_arg, None)
        };
        let mut args = vec!["-c", "-q", arg1];
        if let Some(v) = arg2 {
            args.push(v);
        }
        args.push("-");
        let mut comp = Command::new(bin_path())
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        comp.stdin.as_mut().unwrap().write_all(&payload).unwrap();
        let out = comp.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "[{arg1}] compression should have succeeded; stderr: {}",
            String::from_utf8_lossy(&out.stderr),
        );
        let compressed = out.stdout;
        // Decompress and confirm roundtrip.
        let mut dec = Command::new(bin_path())
            .args(["-d", "-c", "-q", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        dec.stdin.as_mut().unwrap().write_all(&compressed).unwrap();
        let dec_out = dec.wait_with_output().unwrap();
        assert!(dec_out.status.success());
        assert_eq!(dec_out.stdout, payload, "[{arg1}] roundtrip mismatch");
    }
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
    let long = compress(&["-c", "-q", "-L", "1", "-"], &payload);
    assert_eq!(short, long, "-1 should behave like -L 1");

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
        let canonical = compress(&["-c", "-q", "-L", level, "-"], &payload);
        assert_eq!(
            clustered, canonical,
            "{:?} should behave like -c -L {level}",
            args,
        );
    }
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
        stderr.contains("separate argument"),
        "unexpected attached -D diagnostic: {stderr}"
    );
    assert!(
        out.stdout.is_empty(),
        "rejected attached -D value must not emit compressed data"
    );

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
        eprintln!("upstream zstd not on $PATH; skipping");
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
    let check_then_no_check = compress(&["-c", "-q", "--check", "--no-check", "-"], &payload);
    let no_check_then_check = compress(&["-c", "-q", "--no-check", "--check", "-"], &payload);
    assert_eq!(
        plain, explicit_check,
        "default output should match explicit --check"
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
        eprintln!("upstream zstd not on $PATH; skipping");
        return;
    };

    for fixture in [
        "tests/fixtures/lorem50.txt",
        "tests/fixtures/lorem.txt",
        "tests/fixtures/rep100.txt",
        "tests/fixtures/binary.bin",
    ] {
        let ours = Command::new(bin_path())
            .args(["-q", "--no-check", "-c", "-L", "1", fixture])
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
    // The CLI's `--version` (long form) advertises the upstream
    // libzstd version the port mirrors. Pin that it matches the
    // library-side constant so a future version bump can't drift
    // the two out of sync.
    let out = Command::new(bin_path())
        .arg("--version")
        .output()
        .expect("version invocation");
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).expect("utf-8 version");
    let expected = zstd_pure_rs::common::zstd_common::ZSTD_VERSION_STRING;
    assert!(
        stdout.contains(expected),
        "long-version output `{stdout}` does not contain `{expected}`",
    );
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
            .args(["-c", "-q", "--magicless", "-L", &level.to_string(), "-"])
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
    // silently dropped the checksum flag (the dict path used
    // `ZSTD_compress_usingDict` which ignores `fParams`). The fix
    // routes the dict path through `ZSTD_compress_advanced` so
    // `fParams.checksumFlag` is honored. Proof: output with
    // `-D + --check` should be exactly 4 bytes longer than
    // `-D + --no-check` (the XXH64 trailer), and contentSize bits of
    // the FHD byte must encode the checksumFlag.
    use std::env;
    use std::fs;
    let tmp = env::temp_dir();
    let pid = std::process::id();
    let dict_path = tmp.join(format!("cli-dict-check-{pid}.dict"));
    let payload_path = tmp.join(format!("cli-dict-check-{pid}.src"));
    let plain_out = tmp.join(format!("cli-dict-check-{pid}-plain.zst"));
    let check_out = tmp.join(format!("cli-dict-check-{pid}-check.zst"));
    let _ = fs::remove_file(&plain_out);
    let _ = fs::remove_file(&check_out);

    let dict = b"dict-check-prefix-bytes ".repeat(2);
    let payload = b"dict-check-prefix-bytes payload body ".repeat(4);
    fs::write(&dict_path, &dict).unwrap();
    fs::write(&payload_path, &payload).unwrap();

    for (args, out_path) in [
        (vec!["-q", "--no-check", "-D"], &plain_out),
        (vec!["-q", "--check", "-D"], &check_out),
    ] {
        let mut cmd_args = args;
        cmd_args.extend([
            dict_path.to_str().unwrap(),
            "-o",
            out_path.to_str().unwrap(),
            payload_path.to_str().unwrap(),
        ]);
        let comp = Command::new(bin_path())
            .args(&cmd_args)
            .output()
            .expect("spawn compressor");
        assert!(
            comp.status.success(),
            "compress args={cmd_args:?} stderr: {}",
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
    // path (via `ZSTD_CCtx_setFormat` + `ZSTD_compress_usingDict`)
    // and the symmetric decode (magicless dctx + dict decode).
    use std::env;
    use std::fs;
    let tmp = env::temp_dir();
    let dict_path = tmp.join("cli-magicless-dict-roundtrip.dict");
    let payload_path = tmp.join("cli-magicless-dict-roundtrip.src");
    let compressed_path = tmp.join("cli-magicless-dict-roundtrip.zst");
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
