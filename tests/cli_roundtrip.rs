//! Integration test: shell out to our own `zstd` binary and verify it
//! decompresses a hand-crafted raw-block `.zst` frame back to the
//! original payload.
//!
//! Only runs when the `cli` feature is enabled (the binary is guarded
//! by `required-features = ["cli"]` in Cargo.toml).

#![cfg(feature = "cli")]

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
    assert!(comp_out.status.success(), "our compress stderr: {}",
        String::from_utf8_lossy(&comp_out.stderr));
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
        // Our CLI takes `--level N`; upstream takes the `-N` short form.
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
    assert!(comp_out.status.success(),
        "our compress stderr: {}", String::from_utf8_lossy(&comp_out.stderr));

    let mut dec = Command::new(&upstream)
        .args(["-d", "-c", "-q", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    dec.stdin.as_mut().unwrap().write_all(&comp_out.stdout).unwrap();
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
        dec.stdin.as_mut().unwrap().write_all(&comp_out.stdout).unwrap();
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
            for i in (0..v.len()).step_by(37) { v[i] = b'Z'; }
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
            assert!(comp_out.status.success(),
                "[case {i}, level {level}] our compress stderr: {}",
                String::from_utf8_lossy(&comp_out.stderr));

            let mut dec = Command::new(&upstream)
                .args(["-d", "-c", "-q", "-"])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            dec.stdin.as_mut().unwrap().write_all(&comp_out.stdout).unwrap();
            let dec_out = dec.wait_with_output().unwrap();
            assert!(dec_out.status.success(),
                "[case {i}, level {level}] upstream rejected our output: stderr={}",
                String::from_utf8_lossy(&dec_out.stderr));
            assert_eq!(&dec_out.stdout, payload,
                "[case {i}, level {level}] roundtrip mismatch");
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
    assert!(comp_out.status.success(),
        "our dict compress stderr: {}", String::from_utf8_lossy(&comp_out.stderr));

    let mut dec = Command::new(&upstream)
        .args(["-d", "-c", "-q", "-D"])
        .arg(&dict_path)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    dec.stdin.as_mut().unwrap().write_all(&comp_out.stdout).unwrap();
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
        assert!(comp_out.status.success(),
            "[iter {seed_iter} size {size} level {level}] our compress stderr: {}",
            String::from_utf8_lossy(&comp_out.stderr));

        let mut dec = Command::new(&upstream)
            .args(["-d", "-c", "-q", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        dec.stdin.as_mut().unwrap().write_all(&comp_out.stdout).unwrap();
        let dec_out = dec.wait_with_output().unwrap();
        assert!(
            dec_out.status.success(),
            "[iter {seed_iter} size {size} level {level}] upstream rejected: stderr={}",
            String::from_utf8_lossy(&dec_out.stderr)
        );
        assert_eq!(&dec_out.stdout, &payload,
            "[iter {seed_iter} size {size} level {level}] mismatch");
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
        .iter().cycle().take(1500).copied().collect();
    fs::write(&input_path, &payload).expect("write input");

    // Compress via `zstd -q -o compressed_path input_path`.
    let comp = Command::new(bin_path())
        .args(["-q", "-f", "-o"])
        .arg(&compressed_path)
        .arg(&input_path)
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(comp.status.success(), "compress stderr: {}", String::from_utf8_lossy(&comp.stderr));
    assert!(compressed_path.exists());

    // Decompress via `zstd -d -q -o decompressed_path compressed_path`.
    let dec = Command::new(bin_path())
        .args(["-d", "-q", "-f", "-o"])
        .arg(&decompressed_path)
        .arg(&compressed_path)
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(dec.status.success(), "decompress stderr: {}", String::from_utf8_lossy(&dec.stderr));
    let recovered = fs::read(&decompressed_path).expect("read decompressed");
    assert_eq!(recovered, payload);

    let _ = fs::remove_file(&input_path);
    let _ = fs::remove_file(&compressed_path);
    let _ = fs::remove_file(&decompressed_path);
}

#[test]
fn cli_help_advertises_compression_support() {
    let out = Command::new(bin_path())
        .arg("--help")
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("compression") || stdout.contains("compress"),
        "--help output didn't mention compression: {stdout}"
    );
}

#[test]
fn cli_version_flag_prints_version() {
    let out = Command::new(bin_path())
        .arg("--version")
        .output()
        .unwrap();
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
fn cli_accepts_out_of_range_level_and_clamps_silently() {
    // Upstream silently clamps `-L 99` to MAX (22) and `-L -999999`
    // to MIN. Our CLI inherits this — compression must succeed and
    // the output must roundtrip. Prevents a regression that makes
    // a typo / user-error crash the CLI instead of DWIM'ing.
    let payload: Vec<u8> = b"clamped-level test ".iter().cycle().take(200).copied().collect();
    // Negative levels are passed as `--level=-N` to avoid clap tokenizing
    // `-5` as a short flag.
    for level_arg in ["-L", "--level=-5"].iter() {
        let (arg1, arg2): (&str, Option<&str>) = if *level_arg == "-L" {
            ("-L", Some("99"))
        } else {
            (level_arg, None)
        };
        let mut args = vec!["-c", "-q", arg1];
        if let Some(v) = arg2 { args.push(v); }
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
            "[-L {level}] compression should have succeeded; stderr: {}",
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
        assert_eq!(dec_out.stdout, payload, "[-L {level}] roundtrip mismatch");
    }
}

#[test]
fn cli_rejects_nonexistent_dict_path_with_nonzero_exit() {
    // Safety gate: `-D` pointing at a missing file must produce a
    // clean error message + non-zero exit, not a panic.
    let missing_dict = std::env::temp_dir()
        .join(format!("zstd_pure_missing_dict_{}.dict", std::process::id()));
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
    child.stdin.as_mut().unwrap().write_all(b"payload").unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        !out.status.success(),
        "should fail when -D points at missing file, stderr: {}",
        String::from_utf8_lossy(&out.stderr),
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains(&missing_dict.display().to_string())
            || stderr.contains("dict"),
        "error message didn't reference the missing dict: {stderr}",
    );
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
    let payload: Vec<u8> = b"the quick brown fox ".iter().cycle().take(400).copied().collect();
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
    let payload: Vec<u8> = b"hello verbose world. ".iter().cycle().take(400).copied().collect();
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
    dec.stdin.as_mut().unwrap().write_all(&comp_out.stdout).unwrap();
    let dec_out = dec.wait_with_output().unwrap();
    assert!(
        dec_out.status.success(),
        "upstream rejected our checksum frame: stderr={}",
        String::from_utf8_lossy(&dec_out.stderr)
    );
    assert_eq!(dec_out.stdout, payload);
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
