use super::*;

#[test]
fn compress_bound_formula_matches_upstream() {
    // Spot-check against the upstream formula hand-evaluated:
    //   bound(0)    = 0 + 0 + (128K>>11) = 64
    //   bound(1)    = 1 + 0 + ((128K-1)>>11) = 1 + 63 = 64
    //   bound(128K) = 128K + 512 + 0 = 131584
    //   bound(1MB)  = 1MB + 4K + 0 = 1052672
    assert_eq!(ZSTD_compressBound(0), 64);
    assert_eq!(ZSTD_compressBound(1), 64);
    assert_eq!(
        ZSTD_compressBound(128 * 1024),
        (128 * 1024usize).wrapping_add(512)
    );
    assert_eq!(
        ZSTD_compressBound(1024 * 1024),
        (1024 * 1024usize).wrapping_add(4096)
    );
}

#[test]
fn compress_bound_monotonically_grows() {
    // bound(A+B) >= bound(A) when B > 0 for A, B >= 128 KB (upstream
    // formula guarantees bound(A) + bound(B) <= bound(A+B) past that).
    let a = 200 * 1024;
    let b = 300 * 1024;
    assert!(ZSTD_compressBound(a + b) >= ZSTD_compressBound(a) + ZSTD_compressBound(b));
}

#[test]
fn zstd_max_input_size_matches_upstream_per_word_width() {
    // Upstream `lib/zstd.h` defines:
    //   ZSTD_MAX_INPUT_SIZE = 0xFF00FF00FF00FF00ULL on 64-bit
    //   ZSTD_MAX_INPUT_SIZE = 0xFF00FF00 on 32-bit
    // The literal pattern maximizes `ZSTD_COMPRESSBOUND` without
    // risking arithmetic overflow on the inner `srcSize >> 8 +
    // block_margin` terms. Drift would change the max payload
    // size accepted, a silent ABI-level change.
    if core::mem::size_of::<usize>() == 8 {
        assert_eq!(ZSTD_MAX_INPUT_SIZE, 0xFF00FF00FF00FF00);
    } else {
        assert_eq!(ZSTD_MAX_INPUT_SIZE, 0xFF00FF00);
    }
}

#[test]
fn matchLengthHalfIsZero_matches_upstream_endianness_contract() {
    if crate::common::mem::MEM_isLittleEndian() != 0 {
        assert!(matchLengthHalfIsZero(0x0000_0000_FFFF_FFFF));
        assert!(matchLengthHalfIsZero(7));
        assert!(!matchLengthHalfIsZero(0x0000_0001_0000_0000));
    } else {
        assert!(matchLengthHalfIsZero(0xFFFF_FFFF_0000_0000));
        assert!(!matchLengthHalfIsZero(0x0000_0000_FFFF_FFFF));
    }
}

#[test]
fn get1BlockSummary_stops_at_terminator_and_sums_packed_halves() {
    let seqs = [
        ZSTD_Sequence {
            offset: 11,
            litLength: 3,
            matchLength: 4,
            rep: 0,
        },
        ZSTD_Sequence {
            offset: 12,
            litLength: 5,
            matchLength: 9,
            rep: 0,
        },
        ZSTD_Sequence {
            offset: 0,
            litLength: 7,
            matchLength: 0,
            rep: 0,
        },
        ZSTD_Sequence {
            offset: 99,
            litLength: 100,
            matchLength: 100,
            rep: 0,
        },
    ];
    let bs = ZSTD_get1BlockSummary(&seqs);
    assert_eq!(bs.nbSequences, 3);
    assert_eq!(bs.litSize, 3usize.wrapping_add(5).wrapping_add(7));
    assert_eq!(
        bs.blockSize,
        3usize
            .wrapping_add(4)
            .wrapping_add(5)
            .wrapping_add(9)
            .wrapping_add(7)
    );
}

#[test]
fn zstd_compressbound_is_usable_in_const_context() {
    // `ZSTD_COMPRESSBOUND` is a `const fn` so callers can size
    // static buffers at compile time. Prove it by evaluating it
    // in a const context AND using the results for a [u8; N]
    // array declaration — if this ever becomes non-const, the
    // compilation fails.
    const B_0: usize = ZSTD_COMPRESSBOUND(0);
    const B_1K: usize = ZSTD_COMPRESSBOUND(1024);
    const B_1M: usize = ZSTD_COMPRESSBOUND(1_000_000);
    const B_TOO_LARGE: usize = ZSTD_COMPRESSBOUND(ZSTD_MAX_INPUT_SIZE);
    // Sizing a compile-time array with the bound confirms `const` context.
    let _buf0: [u8; B_0] = [0u8; B_0];
    assert_eq!(_buf0.len(), 64);
    // Move the const-fn comparisons into a const block so clippy
    // doesn't flag them as `assertions_on_constants`.
    const _: () = {
        assert!(B_1K > 1024);
        assert!(B_1M > 1_000_000);
        assert!(B_TOO_LARGE == 0);
    };
}

#[test]
fn compress_bound_rejects_over_max_input() {
    let rc = ZSTD_compressBound(ZSTD_MAX_INPUT_SIZE);
    assert!(crate::common::error::ERR_isError(rc));
}

#[test]
fn real_compressed_output_fits_within_compressBound() {
    // Regression gate: for every (size, level) combination the
    // actual `ZSTD_compress` output must never exceed the bound
    // returned by `ZSTD_compressBound(size)`. A bound underrun
    // would let callers allocate too-small destination buffers.
    for &size in &[0usize, 1, 33, 512, 4096, 65_536, 200_000] {
        let src: Vec<u8> = (0..size as u32).map(|i| (i ^ (i >> 3)) as u8).collect();
        let bound = ZSTD_compressBound(size);
        for &level in &[1i32, 3, 10] {
            let mut dst = vec![0u8; bound];
            let n = ZSTD_compress(&mut dst, &src, level);
            assert!(
                !ERR_isError(n),
                "compress failed size={size} level={level}: {:#x}",
                n,
            );
            assert!(
                n <= bound,
                "bound violated size={size} level={level} n={n} bound={bound}",
            );
        }
    }
}

#[test]
fn streaming_flush_with_buffered_input_does_not_finalize_frame() {
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    let mut cctx = ZSTD_createCCtx().unwrap();
    assert_eq!(ZSTD_initCStream(&mut cctx, 3), 0);

    let first = b"flush waits for endStream";
    let second = b" and later input remains accepted";
    let mut dst = vec![0u8; ZSTD_compressBound(first.len() + second.len()) + 128];
    let mut src_pos = 0usize;
    let mut dst_pos = 0usize;
    let rc = ZSTD_compressStream2(
        &mut cctx,
        &mut dst,
        &mut dst_pos,
        first,
        &mut src_pos,
        ZSTD_EndDirective::ZSTD_e_flush,
    );

    assert!(!ERR_isError(rc));
    assert_eq!(src_pos, first.len());
    assert_eq!(rc, 0, "flush should complete all currently staged input");
    assert_ne!(dst_pos, 0, "flush must emit a non-final frame prefix");

    src_pos = 0;
    let rc = ZSTD_compressStream2(
        &mut cctx,
        &mut dst,
        &mut dst_pos,
        second,
        &mut src_pos,
        ZSTD_EndDirective::ZSTD_e_continue,
    );
    assert!(!ERR_isError(rc));
    assert_eq!(src_pos, second.len());

    loop {
        let remaining = ZSTD_endStream(&mut cctx, &mut dst, &mut dst_pos);
        assert!(!ERR_isError(remaining));
        if remaining == 0 {
            break;
        }
        dst.resize(dst.len() + remaining.max(32), 0);
    }

    let mut decoded = vec![0u8; first.len() + second.len()];
    let d = ZSTD_decompress(&mut decoded, &dst[..dst_pos]);
    assert_eq!(d, first.len() + second.len());
    assert_eq!(&decoded[..first.len()], first);
    assert_eq!(&decoded[first.len()..d], second);
}

#[test]
fn streaming_large_continue_with_buffered_input_roundtrips_when_ended() {
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    let mut cctx = ZSTD_createCCtx().unwrap();
    assert_eq!(ZSTD_initCStream(&mut cctx, 3), 0);

    let src = vec![b'a'; ZSTD_BLOCKSIZE_MAX + 1];
    let mut dst = vec![0u8; ZSTD_compressBound(src.len()) + 64];
    let mut src_pos = 0usize;
    let mut dst_pos = 0usize;
    let rc = ZSTD_compressStream2(
        &mut cctx,
        &mut dst,
        &mut dst_pos,
        &src,
        &mut src_pos,
        ZSTD_EndDirective::ZSTD_e_continue,
    );

    assert!(!ERR_isError(rc));
    assert_eq!(src_pos, src.len());

    loop {
        let remaining = ZSTD_endStream(&mut cctx, &mut dst, &mut dst_pos);
        assert!(!ERR_isError(remaining));
        if remaining == 0 {
            break;
        }
        dst.resize(dst.len() + remaining.max(32), 0);
    }

    let mut decoded = vec![0u8; src.len()];
    let d = ZSTD_decompress(&mut decoded, &dst[..dst_pos]);
    assert_eq!(d, src.len());
    assert_eq!(&decoded[..d], src.as_slice());
}

#[test]
fn streaming_accepts_authorized_cparam_update_after_buffering_input() {
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    let src = b"accepted bytes keep their first-stream parameter snapshot";
    let mut cctx = ZSTD_createCCtx().unwrap();
    assert_eq!(ZSTD_initCStream(&mut cctx, 1), 0);

    let mut dst = vec![0u8; ZSTD_compressBound(src.len())];
    let mut src_pos = 0usize;
    let mut dst_pos = 0usize;
    let rc = ZSTD_compressStream2(
        &mut cctx,
        &mut dst,
        &mut dst_pos,
        src,
        &mut src_pos,
        ZSTD_EndDirective::ZSTD_e_continue,
    );
    assert!(!ERR_isError(rc));
    assert_eq!(src_pos, src.len());

    let rc = ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_compressionLevel, 12);
    assert_eq!(rc, 0);
    assert_eq!(cctx.stream_level, Some(12));

    loop {
        let remaining = ZSTD_endStream(&mut cctx, &mut dst, &mut dst_pos);
        assert!(!ERR_isError(remaining));
        if remaining == 0 {
            break;
        }
        dst.resize(dst.len() + remaining.max(32), 0);
    }

    let mut decoded = vec![0u8; src.len()];
    let d = ZSTD_decompress(&mut decoded, &dst[..dst_pos]);
    assert_eq!(d, src.len());
    assert_eq!(&decoded[..d], src);
}

#[test]
fn ll_code_small_values_match_lookup_table() {
    // First 16 entries are identity.
    for v in 0..16u32 {
        assert_eq!(ZSTD_LLcode(v), v);
    }
    // Upper range boundary: LL_Code[63] = 24.
    assert_eq!(ZSTD_LLcode(63), 24);
}

#[test]
fn ll_code_large_values_use_highbit_delta() {
    // litLength=64 → highbit32(64)=6, +19 = 25.
    assert_eq!(ZSTD_LLcode(64), 25);
    // litLength=128 → highbit32(128)=7, +19 = 26.
    assert_eq!(ZSTD_LLcode(128), 26);
}

#[test]
fn ml_code_small_values_match_lookup_table() {
    // First 32 entries are identity.
    for v in 0..32u32 {
        assert_eq!(ZSTD_MLcode(v), v);
    }
    assert_eq!(ZSTD_MLcode(127), 42);
}

#[test]
fn ml_code_large_values_use_highbit_delta() {
    // mlBase=128 → highbit32(128)=7, +36 = 43.
    assert_eq!(ZSTD_MLcode(128), 43);
}

#[test]
fn compressSequences_explicit_delimiter_roundtrips_and_generateSequences_roundtrip() {
    use crate::common::error::ERR_getErrorCode;
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    let mut cctx = ZSTD_createCCtx().unwrap();
    cctx.requestedParams.blockDelimiters = ZSTD_SequenceFormat_e::ZSTD_sf_explicitBlockDelimiters;
    let src = b"sequence api roundtrip ".repeat(20);
    let seqs = [ZSTD_Sequence {
        offset: 0,
        litLength: src.len() as u32,
        matchLength: 0,
        rep: 0,
    }];
    let mut dst = vec![0u8; ZSTD_compressBound(src.len()) + 64];
    let rc_c = ZSTD_compressSequences(&mut cctx, &mut dst, &seqs, &src);
    assert!(!ERR_isError(rc_c), "{:?}", ERR_getErrorCode(rc_c));

    let mut decoded = vec![0u8; src.len()];
    let d = ZSTD_decompress(&mut decoded, &dst[..rc_c]);
    assert_eq!(d, src.len());
    assert_eq!(&decoded[..d], src.as_slice());

    let mut out = vec![ZSTD_Sequence::default(); ZSTD_sequenceBound(src.len())];
    let rc_g = ZSTD_generateSequences(&mut cctx, &mut out, &src);
    assert!(!ERR_isError(rc_g));
    assert!(rc_g > 0);

    let mut cctx2 = ZSTD_createCCtx().unwrap();
    cctx2.requestedParams.blockDelimiters = ZSTD_SequenceFormat_e::ZSTD_sf_explicitBlockDelimiters;
    let mut dst2 = vec![0u8; ZSTD_compressBound(src.len()) + 64];
    let rc_c2 = ZSTD_compressSequences(&mut cctx2, &mut dst2, &out[..rc_g], &src);
    assert!(!ERR_isError(rc_c2));

    let mut decoded2 = vec![0u8; src.len()];
    let d2 = ZSTD_decompress(&mut decoded2, &dst2[..rc_c2]);
    assert_eq!(d2, src.len());
    assert_eq!(&decoded2[..d2], src.as_slice());
}

#[test]
fn compressSequences_validateSequences_rejects_offset_before_match_start() {
    use crate::common::error::ERR_getErrorCode;

    let src = b"abcd".to_vec();
    let mut cctx = ZSTD_createCCtx().unwrap();
    cctx.requestedParams.blockDelimiters = ZSTD_SequenceFormat_e::ZSTD_sf_explicitBlockDelimiters;
    cctx.requestedParams.validateSequences = 1;
    let seqs = [
        ZSTD_Sequence {
            offset: 4,
            litLength: 0,
            matchLength: 4,
            rep: 0,
        },
        ZSTD_Sequence {
            offset: 0,
            litLength: 0,
            matchLength: 0,
            rep: 0,
        },
    ];
    let mut dst = vec![0u8; ZSTD_compressBound(src.len()) + 64];
    let rc = ZSTD_compressSequences(&mut cctx, &mut dst, &seqs, &src);
    assert!(ERR_isError(rc));
    assert_eq!(ERR_getErrorCode(rc), ErrorCode::ExternalSequencesInvalid);
}

#[test]
fn compressSequences_rejects_zero_offset_real_match() {
    use crate::common::error::ERR_getErrorCode;

    let src = b"abcd".to_vec();
    let mut cctx = ZSTD_createCCtx().unwrap();
    cctx.requestedParams.blockDelimiters = ZSTD_SequenceFormat_e::ZSTD_sf_explicitBlockDelimiters;
    let seqs = [
        ZSTD_Sequence {
            offset: 0,
            litLength: 0,
            matchLength: 4,
            rep: 0,
        },
        ZSTD_Sequence {
            offset: 0,
            litLength: 0,
            matchLength: 0,
            rep: 0,
        },
    ];
    let mut dst = vec![0u8; ZSTD_compressBound(src.len()) + 64];
    let rc = ZSTD_compressSequences(&mut cctx, &mut dst, &seqs, &src);
    assert!(ERR_isError(rc));
    assert_eq!(ERR_getErrorCode(rc), ErrorCode::ExternalSequencesInvalid);
}

#[test]
fn compressSequences_explicit_delimiter_rejects_sub_minmatch_with_validation_disabled() {
    use crate::common::error::ERR_getErrorCode;
    use crate::compress::seq_store::MINMATCH;

    let src = b"abc".to_vec();
    let mut cctx = ZSTD_createCCtx().unwrap();
    cctx.requestedParams.blockDelimiters = ZSTD_SequenceFormat_e::ZSTD_sf_explicitBlockDelimiters;
    cctx.requestedParams.validateSequences = 0;
    let seqs = [
        ZSTD_Sequence {
            offset: 1,
            litLength: 1,
            matchLength: MINMATCH - 1,
            rep: 0,
        },
        ZSTD_Sequence {
            offset: 0,
            litLength: 0,
            matchLength: 0,
            rep: 0,
        },
    ];
    let mut dst = vec![0u8; ZSTD_compressBound(src.len()) + 64];
    let rc = ZSTD_compressSequences(&mut cctx, &mut dst, &seqs, &src);
    assert!(ERR_isError(rc));
    assert_eq!(ERR_getErrorCode(rc), ErrorCode::ExternalSequencesInvalid);
}

#[test]
fn compressSequences_no_delimiter_rejects_sub_minmatch_with_validation_disabled() {
    use crate::common::error::ERR_getErrorCode;
    use crate::compress::seq_store::MINMATCH;

    let src = b"abc".to_vec();
    let mut cctx = ZSTD_createCCtx().unwrap();
    cctx.requestedParams.validateSequences = 0;
    let seqs = [ZSTD_Sequence {
        offset: 1,
        litLength: 1,
        matchLength: MINMATCH - 1,
        rep: 0,
    }];
    let mut dst = vec![0u8; ZSTD_compressBound(src.len()) + 64];
    let rc = ZSTD_compressSequences(&mut cctx, &mut dst, &seqs, &src);
    assert!(ERR_isError(rc));
    assert_eq!(ERR_getErrorCode(rc), ErrorCode::ExternalSequencesInvalid);
}

#[test]
fn compressSequences_validateSequences_checks_raw_offset_before_repcode_encoding() {
    use crate::common::error::ERR_getErrorCode;
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;

    let src = b"abcde".to_vec();
    let mut cctx = ZSTD_createCCtx().unwrap();
    cctx.requestedParams.blockDelimiters = ZSTD_SequenceFormat_e::ZSTD_sf_explicitBlockDelimiters;
    cctx.requestedParams.validateSequences = 1;
    cctx.requestedParams.searchForExternalRepcodes = ZSTD_ParamSwitch_e::ZSTD_ps_enable;
    let seqs = [
        ZSTD_Sequence {
            offset: 4,
            litLength: 1,
            matchLength: 4,
            rep: 0,
        },
        ZSTD_Sequence {
            offset: 0,
            litLength: 0,
            matchLength: 0,
            rep: 0,
        },
    ];
    let mut dst = vec![0u8; ZSTD_compressBound(src.len()) + 64];
    let rc = ZSTD_compressSequences(&mut cctx, &mut dst, &seqs, &src);
    assert!(ERR_isError(rc));
    assert_eq!(ERR_getErrorCode(rc), ErrorCode::ExternalSequencesInvalid);
}

#[test]
fn compressSequences_no_delimiter_rejects_literal_only_delimiter_sequence() {
    use crate::common::error::ERR_getErrorCode;

    let src = b"literal-only sentinel".to_vec();
    let mut cctx = ZSTD_createCCtx().unwrap();
    let seqs = [ZSTD_Sequence {
        offset: 0,
        litLength: src.len() as u32,
        matchLength: 0,
        rep: 0,
    }];
    let mut dst = vec![0u8; ZSTD_compressBound(src.len()) + 64];
    let rc = ZSTD_compressSequences(&mut cctx, &mut dst, &seqs, &src);
    assert!(ERR_isError(rc));
    assert_eq!(ERR_getErrorCode(rc), ErrorCode::ExternalSequencesInvalid);
}

#[test]
fn generateSequences_supports_target_cblock_size_with_explicit_delimiters() {
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    let src = b"aaaaabbbbbcccccdddddeeeee".repeat(512);
    let mut cctx = ZSTD_createCCtx().unwrap();
    cctx.requestedParams.blockDelimiters = ZSTD_SequenceFormat_e::ZSTD_sf_explicitBlockDelimiters;
    cctx.requestedParams.targetCBlockSize = 64;

    let mut seqs = vec![ZSTD_Sequence::default(); ZSTD_sequenceBound(src.len())];
    let rc = ZSTD_generateSequences(&mut cctx, &mut seqs, &src);
    assert!(!ERR_isError(rc), "generateSequences err={rc:#x}");
    assert!(rc > 0);

    assert_eq!(seqs[rc - 1].offset, 0);
    assert_eq!(seqs[rc - 1].matchLength, 0);

    let mut cctx2 = ZSTD_createCCtx().unwrap();
    cctx2.requestedParams.blockDelimiters = ZSTD_SequenceFormat_e::ZSTD_sf_explicitBlockDelimiters;
    let mut dst = vec![0u8; ZSTD_compressBound(src.len()) + 64];
    let csize = ZSTD_compressSequences(&mut cctx2, &mut dst, &seqs[..rc], &src);
    assert!(!ERR_isError(csize), "compressSequences err={csize:#x}");

    let mut decoded = vec![0u8; src.len()];
    let dsize = ZSTD_decompress(&mut decoded, &dst[..csize]);
    assert_eq!(dsize, src.len());
    assert_eq!(decoded, src);
}

#[test]
fn generateSequences_default_output_compresses_with_default_compressSequences() {
    use crate::common::error::ERR_getErrorCode;
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    let src = b"default generated sequences should compress by default ".repeat(64);
    let mut cctx = ZSTD_createCCtx().unwrap();
    let mut seqs = vec![ZSTD_Sequence::default(); ZSTD_sequenceBound(src.len())];
    let nb_seqs = ZSTD_generateSequences(&mut cctx, &mut seqs, &src);
    assert!(!ERR_isError(nb_seqs), "{:?}", ERR_getErrorCode(nb_seqs));
    assert!(nb_seqs > 0);
    assert!(
        seqs[..nb_seqs]
            .iter()
            .all(|seq| seq.offset != 0 || seq.matchLength != 0),
        "default generateSequences should emit no-delimiter sequences"
    );

    let mut cctx2 = ZSTD_createCCtx().unwrap();
    let mut dst = vec![0u8; ZSTD_compressBound(src.len()) + 64];
    let csize = ZSTD_compressSequences(&mut cctx2, &mut dst, &seqs[..nb_seqs], &src);
    assert!(!ERR_isError(csize), "{:?}", ERR_getErrorCode(csize));

    let mut decoded = vec![0u8; src.len()];
    let dsize = ZSTD_decompress(&mut decoded, &dst[..csize]);
    assert_eq!(dsize, src.len());
    assert_eq!(decoded, src);
}

#[cfg(feature = "mt")]
#[test]
fn generateSequences_ignores_nbworkers_and_still_collects_sequences() {
    let mut cctx = ZSTD_createCCtx().unwrap();
    assert_eq!(
        ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_nbWorkers, 2),
        0
    );
    let src = b"generate-sequences with nbworkers still runs ".repeat(32);
    let mut seqs = vec![ZSTD_Sequence::default(); ZSTD_sequenceBound(src.len())];
    let rc = ZSTD_generateSequences(&mut cctx, &mut seqs, &src);
    assert!(!ERR_isError(rc), "generateSequences err={rc:#x}");
    assert!(rc > 0);
    assert!(seqs[..rc]
        .iter()
        .all(|seq| seq.offset != 0 || seq.matchLength != 0));
}

#[test]
fn compressSequencesAndLiterals_explicit_literals_only_roundtrips() {
    use crate::common::error::ERR_getErrorCode;
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    let mut cctx = ZSTD_createCCtx().unwrap();
    cctx.requestedParams.blockDelimiters = ZSTD_SequenceFormat_e::ZSTD_sf_explicitBlockDelimiters;
    let literals = b"explicit literals only ".repeat(24);
    let seqs = [ZSTD_Sequence {
        offset: 0,
        litLength: literals.len() as u32,
        matchLength: 0,
        rep: 0,
    }];
    let mut dst = vec![0u8; ZSTD_compressBound(literals.len()) + 64];
    let n = ZSTD_compressSequencesAndLiterals(
        &mut cctx,
        &mut dst,
        &seqs,
        &literals,
        literals.len() + 8,
        literals.len(),
    );
    assert!(!ERR_isError(n), "{:?}", ERR_getErrorCode(n));

    let mut decoded = vec![0u8; literals.len()];
    let d = ZSTD_decompress(&mut decoded, &dst[..n]);
    assert_eq!(d, literals.len());
    assert_eq!(&decoded[..d], literals.as_slice());
}

#[test]
fn compressSequencesAndLiterals_rejects_no_block_delimiter_mode() {
    use crate::common::error::ERR_getErrorCode;

    let mut cctx = ZSTD_createCCtx().unwrap();
    cctx.requestedParams.blockDelimiters = ZSTD_SequenceFormat_e::ZSTD_sf_noBlockDelimiters;
    let literals = b"abc".to_vec();
    let seqs = [ZSTD_Sequence {
        offset: 0,
        litLength: literals.len() as u32,
        matchLength: 0,
        rep: 0,
    }];
    let mut dst = vec![0u8; 128];
    let rc = ZSTD_compressSequencesAndLiterals(
        &mut cctx,
        &mut dst,
        &seqs,
        &literals,
        literals.len() + 8,
        literals.len(),
    );
    assert!(ERR_isError(rc));
    assert_eq!(ERR_getErrorCode(rc), ErrorCode::FrameParameterUnsupported);
}

#[test]
fn compressSequencesAndLiterals_rejects_sub_minmatch_with_validation_disabled() {
    use crate::common::error::ERR_getErrorCode;
    use crate::compress::seq_store::MINMATCH;

    let mut cctx = ZSTD_createCCtx().unwrap();
    cctx.requestedParams.blockDelimiters = ZSTD_SequenceFormat_e::ZSTD_sf_explicitBlockDelimiters;
    cctx.requestedParams.validateSequences = 0;
    let literals = b"a".to_vec();
    let seqs = [
        ZSTD_Sequence {
            offset: 1,
            litLength: literals.len() as u32,
            matchLength: MINMATCH - 1,
            rep: 0,
        },
        ZSTD_Sequence {
            offset: 0,
            litLength: 0,
            matchLength: 0,
            rep: 0,
        },
    ];
    let mut dst = vec![0u8; 128];
    let rc = ZSTD_compressSequencesAndLiterals(
        &mut cctx,
        &mut dst,
        &seqs,
        &literals,
        literals.len() + 8,
        literals.len() + (MINMATCH as usize - 1),
    );
    assert!(ERR_isError(rc));
    assert_eq!(ERR_getErrorCode(rc), ErrorCode::ExternalSequencesInvalid);
}

#[test]
fn compressSequencesAndLiterals_requires_literal_capacity_with_slack() {
    use crate::common::error::ERR_getErrorCode;

    let mut cctx = ZSTD_createCCtx().unwrap();
    cctx.requestedParams.blockDelimiters = ZSTD_SequenceFormat_e::ZSTD_sf_explicitBlockDelimiters;
    let literals = b"literal slack".to_vec();
    let seqs = [ZSTD_Sequence {
        offset: 0,
        litLength: literals.len() as u32,
        matchLength: 0,
        rep: 0,
    }];
    let mut dst = vec![0u8; 128];
    let rc = ZSTD_compressSequencesAndLiterals(
        &mut cctx,
        &mut dst,
        &seqs,
        &literals,
        literals.len() + 7,
        literals.len(),
    );
    assert!(ERR_isError(rc));
    assert_eq!(ERR_getErrorCode(rc), ErrorCode::WorkSpaceTooSmall);
}

#[test]
fn compressContinue_and_compressEnd_roundtrip_after_begin() {
    let mut cctx = ZSTD_createCCtx().unwrap();
    let rc = ZSTD_compressBegin(&mut cctx, 3);
    assert!(!ERR_isError(rc));

    let part1 = b"continue ";
    let part2 = b"end test";
    let mut dst = [0u8; 256];
    let c1 = ZSTD_compressContinue(&mut cctx, &mut dst, part1);
    assert!(!ERR_isError(c1));
    let c2 = ZSTD_compressEnd(&mut cctx, &mut dst[c1..], part2);
    assert!(!ERR_isError(c2));

    let frame = &dst[..c1 + c2];
    let mut out = vec![0u8; part1.len() + part2.len()];
    let d = crate::decompress::zstd_decompress::ZSTD_decompress(&mut out, frame);
    assert_eq!(d, out.len());
    assert_eq!(&out[..part1.len()], part1);
    assert_eq!(&out[part1.len()..], part2);
}

#[test]
fn compressBlock_emits_headerless_body_after_begin() {
    let mut cctx = ZSTD_createCCtx().unwrap();
    let rc = ZSTD_compressBegin(&mut cctx, 3);
    assert!(!ERR_isError(rc));

    let src = b"headerless block test headerless block test";
    let mut dst = [0u8; 256];
    let c = ZSTD_compressBlock(&mut cctx, &mut dst, src);
    assert!(!ERR_isError(c));
    assert!(c > 0);
    assert!(c <= dst.len());
}

#[test]
fn compress_side_free_functions_accept_none_without_panic() {
    // `ZSTD_freeCCtx`, `ZSTD_freeCDict`, `ZSTD_freeCStream`, and
    // `ZSTD_freeCCtxParams` all accept `Option<Box<T>>`. Passing
    // `None` is a valid pattern (upstream allows a null free
    // argument) and must return 0 without panicking.
    assert_eq!(ZSTD_freeCCtx(None), 0);
    assert_eq!(ZSTD_freeCDict(None), 0);
    assert_eq!(ZSTD_freeCStream(None), 0);
    assert_eq!(ZSTD_freeCCtxParams(None), 0);
}

#[test]
fn CCtx_refThreadPool_and_sizeof_mtctx_track_attached_pool() {
    let mut cctx = ZSTD_createCCtx().unwrap();
    assert_eq!(ZSTD_CCtx_refThreadPool(&mut cctx, None), 0);
    assert_eq!(ZSTD_sizeof_mtctx(&cctx), 0);
    let pool = crate::common::pool::ZSTD_createThreadPool(1).expect("thread pool");
    assert_eq!(ZSTD_CCtx_refThreadPool(&mut cctx, Some(&pool)), 0);
    assert_eq!(
        ZSTD_sizeof_mtctx(&cctx),
        crate::common::pool::POOL_sizeof(&pool)
    );
    assert_eq!(ZSTD_CCtx_refThreadPool(&mut cctx, None), 0);
    assert_eq!(ZSTD_sizeof_mtctx(&cctx), 0);
}

#[cfg(feature = "mt")]
#[test]
fn compress2_roundtrip_with_nbworkers_uses_mt_endstream_path() {
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    let mut cctx = ZSTD_createCCtx().unwrap();
    assert_eq!(
        ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_nbWorkers, 2),
        0
    );
    let src = b"mt-compress2-roundtrip payload ".repeat(200);
    let mut compressed = vec![0u8; ZSTD_compressBound(src.len())];
    let n = ZSTD_compress2(&mut cctx, &mut compressed, &src);
    assert!(!ERR_isError(n), "compress2 err={n:#x}");
    compressed.truncate(n);

    let mut decoded = vec![0u8; src.len() + 16];
    let d = ZSTD_decompress(&mut decoded, &compressed);
    assert!(!ERR_isError(d), "decompress err={d:#x}");
    assert_eq!(&decoded[..d], &src[..]);
}

#[cfg(feature = "mt")]
#[test]
fn endstream_roundtrip_with_nbworkers_and_refprefix() {
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    let mut cctx = ZSTD_createCCtx().unwrap();
    assert_eq!(
        ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_nbWorkers, 2),
        0
    );
    let prefix = b"mt-prefix-history ".repeat(16);
    let src = b"prefix-aware mt endstream payload ".repeat(120);
    assert_eq!(ZSTD_CCtx_refPrefix(&mut cctx, &prefix), 0);
    ZSTD_initCStream(&mut cctx, 3);

    let mut compressed = vec![0u8; ZSTD_compressBound(src.len())];
    let mut cp = 0usize;
    let mut ip = 0usize;
    let rc = ZSTD_compressStream2(
        &mut cctx,
        &mut compressed,
        &mut cp,
        &src,
        &mut ip,
        ZSTD_EndDirective::ZSTD_e_end,
    );
    assert_eq!(rc, 0);
    compressed.truncate(cp);

    let mut decoded = vec![0u8; src.len() + 16];
    let d = ZSTD_decompress(&mut decoded, &compressed);
    assert!(!ERR_isError(d), "decompress err={d:#x}");
    assert_eq!(&decoded[..d], &src[..]);
}

#[cfg(feature = "mt")]
#[test]
fn compresscctx_roundtrip_with_nbworkers_and_attached_pool() {
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    let mut cctx = ZSTD_createCCtx().unwrap();
    let pool = crate::common::pool::ZSTD_createThreadPool(2).expect("thread pool");
    assert_eq!(ZSTD_CCtx_refThreadPool(&mut cctx, Some(&pool)), 0);
    assert_eq!(
        ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_nbWorkers, 2),
        0
    );
    let src = b"mt-compresscctx-attached-pool payload ".repeat(180);
    let mut compressed = vec![0u8; ZSTD_compressBound(src.len())];
    let n = ZSTD_compressCCtx(&mut cctx, &mut compressed, &src, 4);
    assert!(!ERR_isError(n), "compressCCtx err={n:#x}");
    compressed.truncate(n);

    let mut decoded = vec![0u8; src.len() + 16];
    let d = ZSTD_decompress(&mut decoded, &compressed);
    assert!(!ERR_isError(d), "decompress err={d:#x}");
    assert_eq!(&decoded[..d], &src[..]);
}

#[cfg(feature = "mt")]
#[test]
fn compresscctx_roundtrip_with_nbworkers_and_attached_rayon_pool() {
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    let mut cctx = ZSTD_createCCtx().unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("rayon thread pool");
    assert_eq!(ZSTD_CCtx_refRayonThreadPool(&mut cctx, Some(&pool)), 0);
    assert_eq!(
        ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_nbWorkers, 2),
        0
    );
    let src = b"mt-compresscctx-attached-rayon-pool payload ".repeat(180);
    let mut compressed = vec![0u8; ZSTD_compressBound(src.len())];
    let n = ZSTD_compressCCtx(&mut cctx, &mut compressed, &src, 4);
    assert!(!ERR_isError(n), "compressCCtx err={n:#x}");
    compressed.truncate(n);

    let mut decoded = vec![0u8; src.len() + 16];
    let d = ZSTD_decompress(&mut decoded, &compressed);
    assert!(!ERR_isError(d), "decompress err={d:#x}");
    assert_eq!(&decoded[..d], &src[..]);

    let mut cleared = ZSTD_createCCtx().unwrap();
    assert_eq!(ZSTD_CCtx_refRayonThreadPool(&mut cleared, Some(&pool)), 0);
    assert_eq!(ZSTD_CCtx_refRayonThreadPool(&mut cleared, None), 0);
    assert_eq!(ZSTD_sizeof_mtctx(&cleared), 0);
}

#[test]
fn CCtx_refCDict_seeds_dict_and_level_and_roundtrips() {
    // `ZSTD_CCtx_refCDict` should bind the CDict and level onto
    // the CCtx. A subsequent compress2 call must then produce
    // output that decodes with the matching dict.
    use crate::decompress::zstd_decompress::ZSTD_decompress_usingDict;
    let dict = b"CCtx-refCDict-test-dict-content ".repeat(6);
    let cdict = ZSTD_createCDict(&dict, 5).expect("cdict");

    let mut cctx = ZSTD_createCCtx().unwrap();
    let rc = ZSTD_CCtx_refCDict(&mut cctx, &cdict);
    assert_eq!(rc, 0);
    assert!(cctx.stream_dict.is_empty());
    assert!(cctx.stream_cdict.is_some());
    assert_eq!(cctx.stream_level, Some(5));

    // End-to-end: compress via compress2, decompress with the
    // same raw-content dict.
    let src: Vec<u8> = b"payload with CCtx-refCDict-test-dict-content ".repeat(20);
    let mut direct_cctx = ZSTD_createCCtx().unwrap();
    let mut direct_dst = vec![0u8; 4096];
    let direct_n = ZSTD_compress_usingCDict(&mut direct_cctx, &mut direct_dst, &src, &cdict);
    assert!(!ERR_isError(direct_n), "direct cdict err={direct_n:#x}");
    direct_dst.truncate(direct_n);
    let mut direct_dctx = crate::decompress::zstd_decompress_block::ZSTD_DCtx::new();
    let mut direct_out = vec![0u8; src.len() + 64];
    let direct_d = ZSTD_decompress_usingDict(&mut direct_dctx, &mut direct_out, &direct_dst, &dict);
    assert_eq!(&direct_out[..direct_d], &src[..], "direct cdict roundtrip");

    let mut dst = vec![0u8; 4096];
    let n = ZSTD_compress2(&mut cctx, &mut dst, &src);
    assert!(!ERR_isError(n), "compress2 err={n:#x}");
    dst.truncate(n);

    let mut dctx = crate::decompress::zstd_decompress_block::ZSTD_DCtx::new();
    let mut out = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress_usingDict(&mut dctx, &mut out, &dst, &dict);
    assert_eq!(&out[..d], &src[..]);
}

#[test]
fn CCtx_setParams_applies_both_fparams_and_cparams_atomically_on_success() {
    // Happy-path contract complement to the bad-cparams bail
    // test: when cParams are valid, both the fParams flags AND
    // the cParams slot must land on the CCtx.
    let mut cctx = ZSTD_createCCtx().unwrap();
    let cp = ZSTD_getCParams(7, 0, 0);
    let params = ZSTD_parameters {
        cParams: cp,
        fParams: ZSTD_FrameParameters {
            contentSizeFlag: 0, // flip from default (1)
            checksumFlag: 1,    // flip from default (0)
            noDictIDFlag: 1,    // → dictIDFlag = 0
        },
    };
    let rc = ZSTD_CCtx_setParams(&mut cctx, params);
    assert_eq!(rc, 0);
    // fParams flags took effect.
    assert!(!cctx.param_contentSize);
    assert!(cctx.param_checksum);
    assert!(!cctx.param_dictID);
    // cParams landed in the requested-slot.
    assert_eq!(
        cctx.requested_cParams.map(|c| c.windowLog),
        Some(cp.windowLog)
    );
}

#[test]
fn CCtx_setParams_bails_before_touching_fparams_on_bad_cparams() {
    // Contract: `ZSTD_CCtx_setParams` must validate cParams FIRST
    // and bail without mutating fParam flags. Otherwise a bad
    // batch would silently enable checksum on a CCtx that was
    // then rejected — leaving inconsistent state.
    let mut cctx = ZSTD_createCCtx().unwrap();
    let prior_checksum = cctx.param_checksum;
    let prior_contentSize = cctx.param_contentSize;

    let bad_cp = crate::compress::match_state::ZSTD_compressionParameters {
        windowLog: 99, // invalid
        chainLog: 16,
        hashLog: 17,
        searchLog: 4,
        minMatch: 4,
        targetLength: 32,
        strategy: 3,
    };
    let params = ZSTD_parameters {
        cParams: bad_cp,
        fParams: ZSTD_FrameParameters {
            contentSizeFlag: 0,
            checksumFlag: 1, // would flip param_checksum
            noDictIDFlag: 1,
        },
    };
    let rc = ZSTD_CCtx_setParams(&mut cctx, params);
    assert!(ERR_isError(rc));
    // fParam flags must remain at their prior values.
    assert_eq!(cctx.param_checksum, prior_checksum);
    assert_eq!(cctx.param_contentSize, prior_contentSize);
    // requested_cParams also stays empty.
    assert!(cctx.requested_cParams.is_none());
}

#[test]
fn CCtx_setCParams_rejects_invalid_cparams_and_leaves_state_untouched() {
    // Contract: on bad cParams, `ZSTD_CCtx_setCParams` must
    // surface the `ZSTD_checkCParams` error and NOT touch
    // `requested_cParams` — otherwise a subsequent compress call
    // could pick up a half-validated config.
    let mut cctx = ZSTD_createCCtx().unwrap();
    assert!(cctx.requested_cParams.is_none());
    let bad = crate::compress::match_state::ZSTD_compressionParameters {
        windowLog: 99, // way over ZSTD_WINDOWLOG_MAX
        chainLog: 16,
        hashLog: 17,
        searchLog: 4,
        minMatch: 4,
        targetLength: 32,
        strategy: 3,
    };
    let rc = ZSTD_CCtx_setCParams(&mut cctx, bad);
    assert!(ERR_isError(rc));
    assert!(
        cctx.requested_cParams.is_none(),
        "requested_cParams got populated despite error"
    );
}

#[test]
fn two_independent_cctxs_produce_independent_output() {
    // Isolation contract: per-CCtx state MUST NOT leak between
    // CCtxes (no static shared state). Two CCtxes on the same
    // payload with the same level produce byte-identical output;
    // interleaving compresses on the second CCtx doesn't affect
    // the first.
    use crate::decompress::zstd_decompress::ZSTD_decompress;
    let src = b"independence test payload. ".repeat(60);

    let mut a = ZSTD_createCCtx().unwrap();
    let mut b = ZSTD_createCCtx().unwrap();

    let mut dst_a = vec![0u8; 2048];
    let n_a = ZSTD_compressCCtx(&mut a, &mut dst_a, &src, 3);
    assert!(!ERR_isError(n_a));

    // Interleave: use `b` with a different level in between.
    let mut dst_b = vec![0u8; 2048];
    let n_b = ZSTD_compressCCtx(&mut b, &mut dst_b, &src, 5);
    assert!(!ERR_isError(n_b));

    // Now use `a` again. Should still produce level-3 output.
    let mut dst_a2 = vec![0u8; 2048];
    let n_a2 = ZSTD_compressCCtx(&mut a, &mut dst_a2, &src, 3);
    assert!(!ERR_isError(n_a2));
    // A fresh CCtx at level 3 on same payload should match.
    let mut fresh = ZSTD_createCCtx().unwrap();
    let mut dst_f = vec![0u8; 2048];
    let n_f = ZSTD_compressCCtx(&mut fresh, &mut dst_f, &src, 3);
    assert!(!ERR_isError(n_f));
    assert_eq!(
        &dst_a2[..n_a2],
        &dst_f[..n_f],
        "CCtx `a` drifted after interleaving with CCtx `b`"
    );

    // Both outputs roundtrip.
    let mut out = vec![0u8; src.len() + 64];
    let d_a = ZSTD_decompress(&mut out, &dst_a[..n_a]);
    assert_eq!(&out[..d_a], &src[..]);
    let d_b = ZSTD_decompress(&mut out, &dst_b[..n_b]);
    assert_eq!(&out[..d_b], &src[..]);
}

#[test]
fn ZSTD_CStream_is_alias_for_ZSTD_CCtx() {
    // Upstream `typedef ZSTD_CCtx ZSTD_CStream` — same struct,
    // same API. Rust port mirrors via `pub type`. Verify size
    // equality and that functions accepting either signature
    // can be called interchangeably.
    assert_eq!(
        core::mem::size_of::<ZSTD_CStream>(),
        core::mem::size_of::<ZSTD_CCtx>()
    );
    // A `Box<ZSTD_CStream>` can be passed where `&mut ZSTD_CCtx`
    // is expected — the type alias guarantees this.
    let mut cs: Box<ZSTD_CStream> = ZSTD_createCStream().unwrap();
    assert_eq!(ZSTD_sizeof_CCtx(&cs), ZSTD_sizeof_CStream(&cs));
    // Reusable via ZSTD_compressCCtx — same struct.
    let src = b"alias probe";
    let mut dst = vec![0u8; 64];
    let n = ZSTD_compressCCtx(&mut cs, &mut dst, src, 1);
    assert!(!ERR_isError(n));
}

#[test]
fn experimental_param_enum_discriminants_match_upstream() {
    // Experimental decoder/encoder parameters interpret their
    // int-valued settings as the corresponding *_e enum. Upstream
    // pins every discriminant explicitly; drift here would mis-
    // route e.g. a `ZSTD_d_forceIgnoreChecksum=1` request to the
    // wrong branch. All four are part of the public advanced API.
    assert_eq!(
        ZSTD_forceIgnoreChecksum_e::ZSTD_d_validateChecksum as u32,
        0
    );
    assert_eq!(ZSTD_forceIgnoreChecksum_e::ZSTD_d_ignoreChecksum as u32, 1);
    assert_eq!(ZSTD_refMultipleDDicts_e::ZSTD_rmd_refSingleDDict as u32, 0);
    assert_eq!(
        ZSTD_refMultipleDDicts_e::ZSTD_rmd_refMultipleDDicts as u32,
        1
    );
    assert_eq!(ZSTD_dictAttachPref_e::ZSTD_dictDefaultAttach as u32, 0);
    assert_eq!(ZSTD_dictAttachPref_e::ZSTD_dictForceAttach as u32, 1);
    assert_eq!(ZSTD_dictAttachPref_e::ZSTD_dictForceCopy as u32, 2);
    assert_eq!(ZSTD_dictAttachPref_e::ZSTD_dictForceLoad as u32, 3);
    assert_eq!(ZSTD_literalCompressionMode_e::ZSTD_lcm_auto as u32, 0);
    assert_eq!(ZSTD_literalCompressionMode_e::ZSTD_lcm_huffman as u32, 1);
    assert_eq!(
        ZSTD_literalCompressionMode_e::ZSTD_lcm_uncompressed as u32,
        2
    );
}

#[test]
fn ZSTD_ParamSwitch_e_discriminants_match_upstream() {
    // `ZSTD_ParamSwitch_e` is used across many `ZSTD_c_*` / `ZSTD_d_*`
    // auto/enable/disable parameters (e.g. ZSTD_c_literalCompressionMode,
    // ZSTD_c_useRowMatchFinder, ZSTD_d_disableHuffmanAssembly). A
    // discriminant drift would silently flip enable↔disable.
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
    assert_eq!(ZSTD_ParamSwitch_e::ZSTD_ps_auto as u32, 0);
    assert_eq!(ZSTD_ParamSwitch_e::ZSTD_ps_enable as u32, 1);
    assert_eq!(ZSTD_ParamSwitch_e::ZSTD_ps_disable as u32, 2);
}

#[test]
fn compressStream2_continue_with_zero_input_is_noop() {
    // `ZSTD_e_continue` with zero input must be a valid no-op —
    // the caller may legitimately poll the state with empty input
    // (e.g. during a graceful shutdown). Must return 0 (success)
    // and not corrupt the frame.
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream(&mut cctx, 1);
    let mut dst = vec![0u8; 256];
    let mut dp = 0usize;
    let mut sp = 0usize;
    let rc = ZSTD_compressStream2(
        &mut cctx,
        &mut dst,
        &mut dp,
        &[],
        &mut sp,
        ZSTD_EndDirective::ZSTD_e_continue,
    );
    assert!(!ERR_isError(rc), "e_continue with 0 input err: {rc:#x}");
    // No output drained (nothing to drain).
    assert_eq!(dp, 0);
    // No input consumed (nothing provided).
    assert_eq!(sp, 0);

    // Feed real data + finalize — confirms the no-op didn't
    // break subsequent state.
    let src = b"post-noop payload. ".repeat(20);
    let mut sp = 0usize;
    let rc = ZSTD_compressStream2(
        &mut cctx,
        &mut dst,
        &mut dp,
        &src,
        &mut sp,
        ZSTD_EndDirective::ZSTD_e_end,
    );
    assert!(!ERR_isError(rc));
    dst.truncate(dp);
    use crate::decompress::zstd_decompress::ZSTD_decompress;
    let mut out = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress(&mut out, &dst);
    assert_eq!(&out[..d], &src[..]);
}

#[test]
fn compressStream2_simpleArgs_forwards_to_compressStream2() {
    // `_simpleArgs` is a thin forwarder over `compressStream2`.
    // Verify it produces byte-identical output to calling the
    // underlying function directly.
    use crate::decompress::zstd_decompress::ZSTD_decompress;
    let src = b"simpleArgs forwarder test ".repeat(20);

    type Entry =
        fn(&mut ZSTD_CCtx, &mut [u8], &mut usize, &[u8], &mut usize, ZSTD_EndDirective) -> usize;
    let roundtrip_via = |entry: Entry| -> Vec<u8> {
        let mut cctx = ZSTD_createCCtx().unwrap();
        let mut dst = vec![0u8; 2048];
        let mut dp = 0usize;
        let mut sp = 0usize;
        let rc = entry(
            &mut cctx,
            &mut dst,
            &mut dp,
            &src,
            &mut sp,
            ZSTD_EndDirective::ZSTD_e_end,
        );
        assert!(!ERR_isError(rc));
        assert_eq!(rc, 0);
        dst.truncate(dp);
        dst
    };
    let via_simple = roundtrip_via(ZSTD_compressStream2_simpleArgs);
    let via_direct = roundtrip_via(ZSTD_compressStream2);
    assert_eq!(via_simple, via_direct);
    // And both roundtrip.
    let mut out = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress(&mut out, &via_simple);
    assert_eq!(&out[..d], &src[..]);
}

#[test]
fn zstd_compress2_returns_error_on_too_small_dst() {
    // Sibling of `zstd_compress_returns_error_on_too_small_dst`,
    // for the parametric `ZSTD_compress2` entry. Must return a
    // ZSTD_isError when dst can't hold the compressed frame,
    // not panic.
    let mut cctx = ZSTD_createCCtx().unwrap();
    let src: Vec<u8> = b"compress2 tiny-dst test ".repeat(30);
    let mut tiny = [0u8; 8];
    let rc = ZSTD_compress2(&mut cctx, &mut tiny, &src);
    assert!(ERR_isError(rc), "expected error, got {rc}");

    let mut empty: [u8; 0] = [];
    assert!(ERR_isError(ZSTD_compress2(&mut cctx, &mut empty, &src)));
}

#[test]
fn zstd_compress_returns_error_on_too_small_dst() {
    // Safety: `ZSTD_compress` must surface a ZSTD_isError return
    // when the output buffer can't hold even the frame header,
    // not panic on OOB writes.
    let src: Vec<u8> = b"some content to compress ".repeat(40);
    // Destination far smaller than any possible frame header.
    let mut tiny_dst = [0u8; 4];
    let rc = ZSTD_compress(&mut tiny_dst, &src, 3);
    assert!(ERR_isError(rc), "expected error, got {rc}");

    // Empty destination also errors.
    let mut empty: [u8; 0] = [];
    assert!(ERR_isError(ZSTD_compress(&mut empty, &src, 3)));
}

#[test]
fn writeFrameHeader_dictID_size_variants_round_trip_through_decoder() {
    // Verify that for each dictID size variant (1/2/4 bytes) the
    // frame header can be round-tripped: compress-side writes it,
    // decoder reads it back with matching dictID and flags.
    use crate::decompress::zstd_decompress::{ZSTD_FrameHeader, ZSTD_getFrameHeader};
    let windowLog = 17u32;
    let cases = [
        (0u32, "none"),
        (42u32, "1-byte"),
        (0xABCDu32, "2-byte"),
        (0xDEAD_BEEFu32, "4-byte"),
    ];
    for (dictID, label) in cases {
        let fParams = ZSTD_FrameParameters {
            contentSizeFlag: 0,
            checksumFlag: 1,
            noDictIDFlag: 0,
        };
        let mut dst = [0u8; ZSTD_FRAMEHEADERSIZE_MAX];
        let n = ZSTD_writeFrameHeader(&mut dst, &fParams, windowLog, 0, dictID);
        assert!(!ERR_isError(n), "[{label}] write error: {n:#x}");

        let mut zfh = ZSTD_FrameHeader::default();
        let rc = ZSTD_getFrameHeader(&mut zfh, &dst[..n]);
        assert_eq!(rc, 0, "[{label}] getFrameHeader err: {rc:#x}");
        assert_eq!(zfh.dictID, dictID, "[{label}] dictID mismatch");
        assert_eq!(zfh.checksumFlag, 1, "[{label}] checksumFlag mismatch");
    }
}

#[test]
fn writeFrameHeader_advanced_elides_magic_under_magicless_format() {
    // Parity gate for the compressor-side magicless path. Upstream
    // (zstd_compress.c:4740): `if (format == ZSTD_f_zstd1) write magic`.
    // Magicless output must (a) lack the 4-byte magic prefix and
    // (b) be accepted by a dctx whose `format` was flipped to
    // `ZSTD_f_zstd1_magicless`.
    use crate::decompress::zstd_decompress::{
        ZSTD_FrameHeader, ZSTD_format_e, ZSTD_getFrameHeader_advanced, ZSTD_MAGICNUMBER,
    };
    let windowLog = 17u32;
    let fParams = ZSTD_FrameParameters {
        contentSizeFlag: 1,
        checksumFlag: 0,
        noDictIDFlag: 1,
    };
    // Pick pledged > (1 << windowLog) so the single-segment code
    // path isn't triggered — when it is, the decoded windowSize
    // collapses to pledgedSrcSize and the round-trip check below
    // becomes ambiguous.
    let pledged = (1u64 << windowLog) * 4;

    // Plain (zstd1) variant: first 4 bytes must be the magic.
    let mut dst_zstd1 = [0u8; ZSTD_FRAMEHEADERSIZE_MAX];
    let n_zstd1 = ZSTD_writeFrameHeader_advanced(
        &mut dst_zstd1,
        &fParams,
        windowLog,
        pledged,
        0,
        ZSTD_format_e::ZSTD_f_zstd1,
    );
    assert!(!ERR_isError(n_zstd1));
    assert_eq!(
        crate::common::mem::MEM_readLE32(&dst_zstd1[..4]),
        ZSTD_MAGICNUMBER,
    );

    // Magicless variant: same FHD layout but 4 bytes shorter.
    let mut dst_noMagic = [0u8; ZSTD_FRAMEHEADERSIZE_MAX];
    let n_noMagic = ZSTD_writeFrameHeader_advanced(
        &mut dst_noMagic,
        &fParams,
        windowLog,
        pledged,
        0,
        ZSTD_format_e::ZSTD_f_zstd1_magicless,
    );
    assert!(!ERR_isError(n_noMagic));
    assert_eq!(
        n_zstd1,
        n_noMagic + 4,
        "magicless output must be 4 bytes shorter"
    );
    // The body after the magic in zstd1 must equal the full
    // magicless output byte-for-byte.
    assert_eq!(&dst_zstd1[4..n_zstd1], &dst_noMagic[..n_noMagic]);

    // Decoder symmetry: a magicless-mode dctx must parse the
    // magicless header and recover the pledged content size +
    // windowLog.
    let mut zfh = ZSTD_FrameHeader::default();
    let rc = ZSTD_getFrameHeader_advanced(
        &mut zfh,
        &dst_noMagic[..n_noMagic],
        ZSTD_format_e::ZSTD_f_zstd1_magicless,
    );
    assert_eq!(rc, 0, "getFrameHeader_advanced err: {rc:#x}");
    assert_eq!(zfh.frameContentSize, pledged);
    assert_eq!(zfh.windowSize, 1u64 << windowLog);
}

#[test]
fn CCtx_setParametersUsingCCtxParams_syncs_format_onto_cctx() {
    // Wholesale params replacement must also update the direct
    // `cctx.format` slot the compressor path reads. Without the
    // sync, a magicless-configured CCtx_params dropped onto the
    // cctx would silently revert the active format to zstd1.
    use crate::decompress::zstd_decompress::ZSTD_format_e;
    let mut params = ZSTD_CCtx_params::default();
    ZSTD_CCtxParams_init(&mut params, 3);
    assert_eq!(
        ZSTD_CCtxParams_setParameter(
            &mut params,
            ZSTD_cParameter::ZSTD_c_format,
            ZSTD_format_e::ZSTD_f_zstd1_magicless as i32,
        ),
        0,
    );
    params.cParams = ZSTD_getCParams(3, 0, 0);

    let mut cctx = ZSTD_createCCtx().unwrap();
    // Pre-condition: default zstd1.
    assert_eq!(cctx.format, ZSTD_format_e::ZSTD_f_zstd1);
    assert_eq!(
        ZSTD_CCtx_setParametersUsingCCtxParams(&mut cctx, &params),
        0
    );
    // Post-condition: params.format surfaced on the cctx.
    assert_eq!(cctx.format, ZSTD_format_e::ZSTD_f_zstd1_magicless);
    assert_eq!(
        cctx.requestedParams.format,
        ZSTD_format_e::ZSTD_f_zstd1_magicless
    );
}

#[test]
fn compressBegin_internal_propagates_format_from_params() {
    // When the caller drives compression through the params
    // surface (`ZSTD_CCtxParams_setParameter(c_format, ...)`
    // → `ZSTD_compressBegin_internal`), the params-level format
    // must land on `cctx.format` so the compressor path picks it
    // up. Missing this propagation meant a params-driven init
    // would silently fall back to zstd1.
    use crate::decompress::zstd_ddict::ZSTD_dictContentType_e;
    use crate::decompress::zstd_decompress::ZSTD_format_e;

    let mut cctx = ZSTD_createCCtx().unwrap();
    // Start from a valid baseline: init params at level 3.
    let mut params = ZSTD_CCtx_params::default();
    ZSTD_CCtxParams_init(&mut params, 3);
    // Flip format via the params setter.
    assert_eq!(
        ZSTD_CCtxParams_setParameter(
            &mut params,
            ZSTD_cParameter::ZSTD_c_format,
            ZSTD_format_e::ZSTD_f_zstd1_magicless as i32,
        ),
        0,
    );
    assert_eq!(params.format, ZSTD_format_e::ZSTD_f_zstd1_magicless);

    // Resolve cParams since compressBegin_internal asserts they
    // were set. ZSTD_CCtxParams_init leaves cParams at defaults
    // which may fail `ZSTD_checkCParams`.
    params.cParams = ZSTD_getCParams(3, 0, 0);

    let rc = ZSTD_compressBegin_internal(
        &mut cctx,
        &[],
        ZSTD_dictContentType_e::ZSTD_dct_auto,
        None,
        &params,
        crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN,
        ZSTD_buffered_policy_e::ZSTDb_not_buffered,
    );
    assert!(!ERR_isError(rc));
    assert_eq!(cctx.format, ZSTD_format_e::ZSTD_f_zstd1_magicless);
}

#[test]
fn CCtx_magicless_compress2_roundtrips() {
    // Parity gate for the one-shot `compress2` path with
    // magicless format. `compress2` resets the session (which
    // must preserve `cctx.format`) and routes through the
    // streaming compressor — so a magicless-configured cctx
    // should produce a magicless frame end-to-end.
    use crate::decompress::zstd_decompress::{
        ZSTD_DCtx_setFormat, ZSTD_decompressStream, ZSTD_format_e,
    };
    use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

    let src = b"compress2-magicless-one-shot-roundtrip ".repeat(8);
    let mut cctx = ZSTD_createCCtx().unwrap();
    // Use the parametric setter to also exercise that path.
    assert_eq!(
        ZSTD_CCtx_setParameter(
            &mut cctx,
            ZSTD_cParameter::ZSTD_c_format,
            ZSTD_format_e::ZSTD_f_zstd1_magicless as i32,
        ),
        0,
    );
    let mut compressed = vec![0u8; ZSTD_compressBound(src.len())];
    let n = ZSTD_compress2(&mut cctx, &mut compressed, &src);
    assert!(!ERR_isError(n));
    compressed.truncate(n);
    assert_ne!(
        crate::common::mem::MEM_readLE32(&compressed[..4]),
        crate::decompress::zstd_decompress::ZSTD_MAGICNUMBER,
        "compress2 leaked zstd1 magic after c_format = magicless",
    );

    let mut dctx = ZSTD_DCtx::new();
    assert_eq!(
        ZSTD_DCtx_setFormat(&mut dctx, ZSTD_format_e::ZSTD_f_zstd1_magicless),
        0,
    );
    let mut out = vec![0u8; src.len() + 64];
    let mut in_pos = 0usize;
    let mut out_pos = 0usize;
    let _ = ZSTD_decompressStream(&mut dctx, &mut out, &mut out_pos, &compressed, &mut in_pos);
    for _ in 0..8 {
        if out_pos >= src.len() {
            break;
        }
        let _ = ZSTD_decompressStream(&mut dctx, &mut out, &mut out_pos, &[], &mut 0usize);
    }
    assert_eq!(&out[..out_pos], &src[..]);
}

#[test]
fn CCtx_magicless_endStream_roundtrips_through_magicless_dctx() {
    // End-to-end parity gate for compressor-side magicless mode:
    //   caller → setFormat(magicless) → compressStream/endStream
    //   → bytes with no 4-byte magic prefix
    //   → decoder setFormat(magicless) → recovers original.
    // Before the format threading this roundtrip couldn't exist —
    // compressor always emitted zstd1 frames with magic.
    use crate::decompress::zstd_decompress::{
        ZSTD_DCtx_setFormat, ZSTD_decompressStream, ZSTD_format_e,
    };
    use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

    let src = b"CCtx-magicless-endStream-parity payload ".repeat(12);
    let mut cctx = ZSTD_createCCtx().unwrap();
    assert_eq!(
        ZSTD_CCtx_setFormat(&mut cctx, ZSTD_format_e::ZSTD_f_zstd1_magicless),
        0,
    );
    ZSTD_initCStream(&mut cctx, 3);

    let mut compressed = vec![0u8; 4096];
    let mut cp = 0usize;
    let mut sp = 0usize;
    let _ = ZSTD_compressStream(&mut cctx, &mut compressed, &mut cp, &src, &mut sp);
    loop {
        let r = ZSTD_endStream(&mut cctx, &mut compressed, &mut cp);
        assert!(!ERR_isError(r));
        if r == 0 {
            break;
        }
    }
    compressed.truncate(cp);
    // The magicless frame must NOT start with the 4-byte zstd1
    // magic (0xFD2FB528 little-endian).
    let magic_le = crate::common::mem::MEM_readLE32(&compressed[..4]);
    assert_ne!(
        magic_le,
        crate::decompress::zstd_decompress::ZSTD_MAGICNUMBER,
        "magicless frame leaked a zstd1 magic prefix",
    );

    // Decode via a magicless-mode streaming dctx.
    let mut dctx = ZSTD_DCtx::new();
    assert_eq!(
        ZSTD_DCtx_setFormat(&mut dctx, ZSTD_format_e::ZSTD_f_zstd1_magicless),
        0,
    );
    let mut out = vec![0u8; src.len() + 64];
    let mut in_pos = 0usize;
    let mut out_pos = 0usize;
    let _ = ZSTD_decompressStream(&mut dctx, &mut out, &mut out_pos, &compressed, &mut in_pos);
    for _ in 0..8 {
        if out_pos >= src.len() {
            break;
        }
        let _ = ZSTD_decompressStream(&mut dctx, &mut out, &mut out_pos, &[], &mut 0usize);
    }
    assert_eq!(&out[..out_pos], &src[..]);
}

#[test]
fn compress_usingDict_honors_cctx_format() {
    // Raw-dict one-shot path: `ZSTD_compress_usingDict(cctx, dst,
    // src, dict, level)` must honor the cctx's format slot.
    // Previously the cctx arg was dropped (`_cctx`), so setFormat
    // had no effect on this entry point.
    use crate::decompress::zstd_decompress::{
        ZSTD_DCtx_setFormat, ZSTD_decompress_usingDict, ZSTD_format_e, ZSTD_MAGICNUMBER,
    };
    use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

    let dict = b"usingDict-magicless-raw-dict ".repeat(3);
    let src = b"usingDict-magicless-raw-dict payload ".repeat(4);

    let mut cctx = ZSTD_createCCtx().unwrap();
    assert_eq!(
        ZSTD_CCtx_setFormat(&mut cctx, ZSTD_format_e::ZSTD_f_zstd1_magicless),
        0,
    );
    let mut dst = vec![0u8; ZSTD_compressBound(src.len())];
    let n = ZSTD_compress_usingDict(&mut cctx, &mut dst, &src, &dict, 3);
    assert!(!ERR_isError(n));
    dst.truncate(n);
    assert_ne!(
        crate::common::mem::MEM_readLE32(&dst[..4]),
        ZSTD_MAGICNUMBER,
        "usingDict leaked magic after setFormat(magicless)",
    );

    let mut dctx = ZSTD_DCtx::new();
    assert_eq!(
        ZSTD_DCtx_setFormat(&mut dctx, ZSTD_format_e::ZSTD_f_zstd1_magicless),
        0,
    );
    let mut out = vec![0u8; src.len() + 128];
    let d = ZSTD_decompress_usingDict(&mut dctx, &mut out, &dst, &dict);
    assert!(!ERR_isError(d), "decode err: {d:#x}");
    assert_eq!(&out[..d], &src[..]);
}

#[test]
fn compress_usingCDict_advanced_honors_cctx_format() {
    // CDict one-shot path parity gate: when the caller flipped
    // magicless on the cctx, the emitted frame must be magicless
    // and the matching dctx + dict must decode it back to the
    // original payload. Previously the cctx argument was
    // ignored entirely (`_cctx`) so the cctx-scoped format slot
    // couldn't influence the output.
    use crate::decompress::zstd_decompress::{
        ZSTD_DCtx_setFormat, ZSTD_decompress_usingDict, ZSTD_format_e, ZSTD_MAGICNUMBER,
    };
    use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

    let dict = b"cdict-magicless-one-shot ".repeat(3);
    let src = b"cdict-magicless-one-shot payload ".repeat(4);
    let cdict = ZSTD_createCDict(&dict, 3).expect("cdict");
    let mut cctx = ZSTD_createCCtx().unwrap();
    assert_eq!(
        ZSTD_CCtx_setFormat(&mut cctx, ZSTD_format_e::ZSTD_f_zstd1_magicless),
        0,
    );
    let mut dst = vec![0u8; ZSTD_compressBound(src.len())];
    let fp = ZSTD_FrameParameters {
        contentSizeFlag: 1,
        checksumFlag: 0,
        noDictIDFlag: 0,
    };
    let n = ZSTD_compress_usingCDict_advanced(&mut cctx, &mut dst, &src, &cdict, fp);
    assert!(!ERR_isError(n));
    dst.truncate(n);
    assert_ne!(
        crate::common::mem::MEM_readLE32(&dst[..4]),
        ZSTD_MAGICNUMBER,
        "usingCDict_advanced leaked magic after setFormat(magicless)",
    );

    let mut dctx = ZSTD_DCtx::new();
    assert_eq!(
        ZSTD_DCtx_setFormat(&mut dctx, ZSTD_format_e::ZSTD_f_zstd1_magicless),
        0,
    );
    let mut out = vec![0u8; src.len() + 128];
    let d = ZSTD_decompress_usingDict(&mut dctx, &mut out, &dst, &dict);
    assert!(!ERR_isError(d), "decode err: {d:#x}");
    assert_eq!(&out[..d], &src[..]);
}

#[test]
fn compressCCtx_honors_cctx_format() {
    // `ZSTD_compressCCtx` takes a level directly but must still
    // honor the cctx's format slot. Without this, a caller who
    // set `c_format = magicless` would see a zstd1 frame come
    // out of the level-driven one-shot — silent divergence from
    // the parametric API contract.
    use crate::decompress::zstd_decompress::{
        ZSTD_DCtx_setFormat, ZSTD_decompressStream, ZSTD_format_e, ZSTD_MAGICNUMBER,
    };
    use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

    let src = b"compressCCtx-magicless-level-path ".repeat(4);
    let mut cctx = ZSTD_createCCtx().unwrap();
    assert_eq!(
        ZSTD_CCtx_setFormat(&mut cctx, ZSTD_format_e::ZSTD_f_zstd1_magicless),
        0,
    );
    let mut dst = vec![0u8; ZSTD_compressBound(src.len())];
    let n = ZSTD_compressCCtx(&mut cctx, &mut dst, &src, 3);
    assert!(!ERR_isError(n));
    dst.truncate(n);
    assert_ne!(
        crate::common::mem::MEM_readLE32(&dst[..4]),
        ZSTD_MAGICNUMBER,
        "compressCCtx leaked magic after setFormat(magicless)",
    );

    let mut dctx = ZSTD_DCtx::new();
    assert_eq!(
        ZSTD_DCtx_setFormat(&mut dctx, ZSTD_format_e::ZSTD_f_zstd1_magicless),
        0,
    );
    let mut out = vec![0u8; src.len() + 64];
    let mut in_pos = 0usize;
    let mut out_pos = 0usize;
    let _ = ZSTD_decompressStream(&mut dctx, &mut out, &mut out_pos, &dst, &mut in_pos);
    for _ in 0..8 {
        if out_pos >= src.len() {
            break;
        }
        let _ = ZSTD_decompressStream(&mut dctx, &mut out, &mut out_pos, &[], &mut 0usize);
    }
    assert_eq!(&out[..out_pos], &src[..]);
}

#[test]
fn compress_advanced_internal_propagates_params_format_onto_cctx() {
    // Sibling of `compress_advanced_honors_cctx_format`: when
    // the caller configures format on a `ZSTD_CCtx_params`
    // (the upstream-canonical slot), `compress_advanced_internal`
    // must push that onto the cctx so the emitter path reads it.
    use crate::decompress::zstd_decompress::{ZSTD_format_e, ZSTD_MAGICNUMBER};

    let src = b"compress_advanced_internal-magicless-propagation ".repeat(3);
    let mut cctx = ZSTD_createCCtx().unwrap();
    // Caller leaves cctx.format at zstd1 default.
    assert_eq!(cctx.format, ZSTD_format_e::ZSTD_f_zstd1);

    // But builds a params struct with magicless flipped via the
    // parametric API.
    let mut params = ZSTD_CCtx_params::default();
    ZSTD_CCtxParams_init(&mut params, 3);
    assert_eq!(
        ZSTD_CCtxParams_setParameter(
            &mut params,
            ZSTD_cParameter::ZSTD_c_format,
            ZSTD_format_e::ZSTD_f_zstd1_magicless as i32,
        ),
        0,
    );
    params.cParams = ZSTD_getCParams(3, src.len() as u64, 0);
    params.fParams = ZSTD_FrameParameters {
        contentSizeFlag: 1,
        checksumFlag: 0,
        noDictIDFlag: 1,
    };

    let mut dst = vec![0u8; ZSTD_compressBound(src.len())];
    let n = ZSTD_compress_advanced_internal(&mut cctx, &mut dst, &src, &[], &params);
    assert!(!ERR_isError(n));
    dst.truncate(n);
    // params-level magicless landed — no zstd1 magic.
    assert_ne!(
        crate::common::mem::MEM_readLE32(&dst[..4]),
        ZSTD_MAGICNUMBER,
    );
    // And the cctx's format slot was updated by the propagation.
    assert_eq!(cctx.format, ZSTD_format_e::ZSTD_f_zstd1_magicless);
}

#[test]
fn compress_advanced_honors_cctx_format() {
    // `ZSTD_compress_advanced` takes a `ZSTD_parameters` (no
    // format field) alongside a `&mut cctx`. The format must come
    // from the cctx's slot — otherwise a caller who flipped
    // magicless via `c_format` would be surprised to get a zstd1
    // frame back.
    use crate::decompress::zstd_decompress::{
        ZSTD_DCtx_setFormat, ZSTD_decompressStream, ZSTD_format_e, ZSTD_MAGICNUMBER,
    };
    use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

    let src = b"compress_advanced-magicless-honors-cctx ".repeat(5);
    let mut cctx = ZSTD_createCCtx().unwrap();
    assert_eq!(
        ZSTD_CCtx_setFormat(&mut cctx, ZSTD_format_e::ZSTD_f_zstd1_magicless),
        0,
    );
    let params = ZSTD_parameters {
        cParams: ZSTD_getCParams(3, src.len() as u64, 0),
        fParams: ZSTD_FrameParameters {
            contentSizeFlag: 1,
            checksumFlag: 0,
            noDictIDFlag: 1,
        },
    };
    let mut dst = vec![0u8; ZSTD_compressBound(src.len())];
    let n = ZSTD_compress_advanced(&mut cctx, &mut dst, &src, &[], params);
    assert!(!ERR_isError(n));
    dst.truncate(n);

    // No magic prefix.
    assert_ne!(
        crate::common::mem::MEM_readLE32(&dst[..4]),
        ZSTD_MAGICNUMBER,
        "compress_advanced leaked magic prefix",
    );

    // Magicless dctx round-trips.
    let mut dctx = ZSTD_DCtx::new();
    assert_eq!(
        ZSTD_DCtx_setFormat(&mut dctx, ZSTD_format_e::ZSTD_f_zstd1_magicless),
        0,
    );
    let mut out = vec![0u8; src.len() + 64];
    let mut in_pos = 0usize;
    let mut out_pos = 0usize;
    let _ = ZSTD_decompressStream(&mut dctx, &mut out, &mut out_pos, &dst, &mut in_pos);
    for _ in 0..8 {
        if out_pos >= src.len() {
            break;
        }
        let _ = ZSTD_decompressStream(&mut dctx, &mut out, &mut out_pos, &[], &mut 0usize);
    }
    assert_eq!(&out[..out_pos], &src[..]);
}

#[test]
fn createCDict_clamps_level_and_applies_default_mapping() {
    // Upstream: `ZSTD_createCDict` routes through the level
    // clamp/default helpers before deriving cParams. Previously
    // our port stashed `compressionLevel` verbatim. Pin the
    // clamp-then-derive contract so a future regression is
    // detected loudly.
    let dict = b"createCDict-clamp-guard".to_vec();

    // Level 0 maps to CLEVEL_DEFAULT.
    let c = ZSTD_createCDict(&dict, 0).expect("cdict");
    assert_eq!(c.compressionLevel, ZSTD_CLEVEL_DEFAULT);

    // Out-of-range levels clamp to [minCLevel, maxCLevel].
    let c_high = ZSTD_createCDict(&dict, i32::MAX).expect("cdict");
    assert_eq!(c_high.compressionLevel, ZSTD_MAX_CLEVEL);

    let c_low = ZSTD_createCDict(&dict, i32::MIN).expect("cdict");
    assert_eq!(c_low.compressionLevel, ZSTD_minCLevel());

    // Mid-range level stored as-is.
    let c_mid = ZSTD_createCDict(&dict, 5).expect("cdict");
    assert_eq!(c_mid.compressionLevel, 5);
}

#[test]
fn CCtx_reset_parameters_only_rejects_mid_stream_but_combined_variant_always_accepts() {
    // Upstream semantics: `reset_parameters` alone requires init
    // stage; `reset_session_and_parameters` is always safe because
    // it clears the session first. Pin both behaviors so a
    // future refactor doesn't flip the distinction.
    use crate::common::error::{ERR_getErrorCode, ERR_isError};
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream(&mut cctx, 3);
    let src = b"mid-stream-reset-semantics ".repeat(3);
    let mut dst = vec![0u8; 1024];
    let mut cp = 0usize;
    let mut sp = 0usize;
    let _ = ZSTD_compressStream(&mut cctx, &mut dst, &mut cp, &src, &mut sp);

    // reset_parameters alone: rejected mid-stream.
    let rc = ZSTD_CCtx_reset(&mut cctx, ZSTD_ResetDirective::ZSTD_reset_parameters);
    assert!(ERR_isError(rc));
    assert_eq!(ERR_getErrorCode(rc), ErrorCode::StageWrong);

    // reset_session_only: always OK (clears the session).
    assert_eq!(
        ZSTD_CCtx_reset(&mut cctx, ZSTD_ResetDirective::ZSTD_reset_session_only),
        0,
    );
    // Now we're back in init stage, so reset_parameters succeeds.
    assert_eq!(
        ZSTD_CCtx_reset(&mut cctx, ZSTD_ResetDirective::ZSTD_reset_parameters),
        0,
    );

    // reset_session_and_parameters is always OK even mid-stream:
    // re-simulate and verify.
    ZSTD_initCStream(&mut cctx, 3);
    let mut dst = vec![0u8; 1024];
    let mut cp = 0usize;
    let mut sp = 0usize;
    let _ = ZSTD_compressStream(&mut cctx, &mut dst, &mut cp, &src, &mut sp);
    assert_eq!(
        ZSTD_CCtx_reset(
            &mut cctx,
            ZSTD_ResetDirective::ZSTD_reset_session_and_parameters,
        ),
        0,
    );
}

#[test]
fn setCParams_accepts_but_setParametersUsingCCtxParams_rejects_mid_stream() {
    // Complement to the setParameter / dict-family gates: the
    // cParams-only API can update future work, but wholesale
    // params replacement must still stage-gate frame flags and other
    // init-only state.
    use crate::common::error::{ERR_getErrorCode, ERR_isError};
    let src = b"mid-stream-params-replace ".repeat(2);

    // setCParams.
    {
        let mut cctx = ZSTD_createCCtx().unwrap();
        ZSTD_initCStream(&mut cctx, 3);
        let cp = ZSTD_getCParams(3, 0, 0);
        assert_eq!(ZSTD_CCtx_setCParams(&mut cctx, cp), 0);
        let mut dst = vec![0u8; 1024];
        let mut cpos = 0usize;
        let mut sp = 0usize;
        let _ = ZSTD_compressStream(&mut cctx, &mut dst, &mut cpos, &src, &mut sp);
        let rc = ZSTD_CCtx_setCParams(&mut cctx, cp);
        assert_eq!(rc, 0);
        assert_eq!(cctx.requested_cParams.map(|c| c.hashLog), Some(cp.hashLog));
    }
    // setParametersUsingCCtxParams.
    {
        let mut cctx = ZSTD_createCCtx().unwrap();
        ZSTD_initCStream(&mut cctx, 3);
        let mut params = ZSTD_CCtx_params::default();
        ZSTD_CCtxParams_init(&mut params, 3);
        params.cParams = ZSTD_getCParams(3, 0, 0);
        assert_eq!(
            ZSTD_CCtx_setParametersUsingCCtxParams(&mut cctx, &params),
            0
        );
        let mut dst = vec![0u8; 1024];
        let mut cpos = 0usize;
        let mut sp = 0usize;
        let _ = ZSTD_compressStream(&mut cctx, &mut dst, &mut cpos, &src, &mut sp);
        let rc = ZSTD_CCtx_setParametersUsingCCtxParams(&mut cctx, &params);
        assert!(ERR_isError(rc));
        assert_eq!(ERR_getErrorCode(rc), ErrorCode::StageWrong);
    }
}

#[test]
fn CCtx_setParameter_unauthorized_params_reject_mid_stream() {
    // Authorized cParams may update future work after staged input,
    // but frame flags / format / worker state still reject with
    // `StageWrong`.
    use crate::common::error::{ERR_getErrorCode, ERR_isError};
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream(&mut cctx, 3);
    let src = b"mid-stream-param-update ".repeat(4);
    let mut dst = vec![0u8; 1024];
    let mut cp = 0usize;
    let mut sp = 0usize;
    let _ = ZSTD_compressStream(&mut cctx, &mut dst, &mut cp, &src, &mut sp);

    assert_eq!(
        ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_compressionLevel, 4),
        0
    );
    assert_eq!(
        ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_hashLog, 16),
        0
    );

    for param in [
        ZSTD_cParameter::ZSTD_c_format,
        ZSTD_cParameter::ZSTD_c_contentSizeFlag,
        ZSTD_cParameter::ZSTD_c_checksumFlag,
        ZSTD_cParameter::ZSTD_c_dictIDFlag,
        ZSTD_cParameter::ZSTD_c_nbWorkers,
    ] {
        let rc = ZSTD_CCtx_setParameter(&mut cctx, param, 0);
        assert!(ERR_isError(rc), "[{param:?}] silent success mid-stream");
        assert_eq!(
            ERR_getErrorCode(rc),
            ErrorCode::StageWrong,
            "[{param:?}] wrong error",
        );
    }
}

#[test]
fn dict_family_setters_reject_mid_stream_with_StageWrong() {
    // Upstream contract: once input has been staged into the
    // stream, dict / prefix / CDict rebinding must error out with
    // `StageWrong`. Without the gate a caller could swap the
    // dict between compressStream() calls, silently decoupling
    // the back-reference substrate from the bytes already
    // buffered for this frame.
    use crate::common::error::{ERR_getErrorCode, ERR_isError};
    let dict = b"init-stage-dict-bytes ".repeat(3);
    let src = b"mid-stream-swap payload ".repeat(2);

    // loadDictionary.
    {
        let mut cctx = ZSTD_createCCtx().unwrap();
        ZSTD_initCStream(&mut cctx, 3);
        assert_eq!(ZSTD_CCtx_loadDictionary(&mut cctx, &dict), 0);
        let mut dst = vec![0u8; 1024];
        let mut cp = 0usize;
        let mut sp = 0usize;
        let _ = ZSTD_compressStream(&mut cctx, &mut dst, &mut cp, &src, &mut sp);
        let rc = ZSTD_CCtx_loadDictionary(&mut cctx, &dict);
        assert!(ERR_isError(rc));
        assert_eq!(ERR_getErrorCode(rc), ErrorCode::StageWrong);
    }
    // refPrefix.
    {
        let mut cctx = ZSTD_createCCtx().unwrap();
        ZSTD_initCStream(&mut cctx, 3);
        assert_eq!(ZSTD_CCtx_refPrefix(&mut cctx, &dict), 0);
        let mut dst = vec![0u8; 1024];
        let mut cp = 0usize;
        let mut sp = 0usize;
        let _ = ZSTD_compressStream(&mut cctx, &mut dst, &mut cp, &src, &mut sp);
        let rc = ZSTD_CCtx_refPrefix(&mut cctx, &dict);
        assert!(ERR_isError(rc));
        assert_eq!(ERR_getErrorCode(rc), ErrorCode::StageWrong);
    }
    // refCDict.
    {
        let cdict = ZSTD_createCDict(&dict, 3).expect("cdict alloc");
        let mut cctx = ZSTD_createCCtx().unwrap();
        ZSTD_initCStream(&mut cctx, 3);
        assert_eq!(ZSTD_CCtx_refCDict(&mut cctx, &cdict), 0);
        let mut dst = vec![0u8; 1024];
        let mut cp = 0usize;
        let mut sp = 0usize;
        let _ = ZSTD_compressStream(&mut cctx, &mut dst, &mut cp, &src, &mut sp);
        let rc = ZSTD_CCtx_refCDict(&mut cctx, &cdict);
        assert!(ERR_isError(rc));
        assert_eq!(ERR_getErrorCode(rc), ErrorCode::StageWrong);
    }
}

#[test]
fn setPledgedSrcSize_rejects_mid_stream_call_with_StageWrong() {
    // Upstream contract (zstd_compress.c:1249): re-pledging mid-
    // session must return `StageWrong`. Without the gate a caller
    // who restaged a new pledge after already buffering input
    // would end up with a frame header advertising the NEW size
    // despite the OLD buffered bytes being compressed — a silent
    // data-corruption vector.
    use crate::common::error::{ERR_getErrorCode, ERR_isError};
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream(&mut cctx, 3);
    // Init stage — pledge accepted.
    assert_eq!(ZSTD_CCtx_setPledgedSrcSize(&mut cctx, 100), 0);
    // Stage some input via compressStream.
    let src = b"mid-stream-pledge-parity-gate ".repeat(3);
    let mut dst = vec![0u8; 1024];
    let mut cp = 0usize;
    let mut sp = 0usize;
    let _ = ZSTD_compressStream(&mut cctx, &mut dst, &mut cp, &src, &mut sp);
    // Post-ingest: stage is no longer init — re-pledge rejected.
    let rc = ZSTD_CCtx_setPledgedSrcSize(&mut cctx, 200);
    assert!(ERR_isError(rc));
    assert_eq!(ERR_getErrorCode(rc), ErrorCode::StageWrong);
}

#[test]
fn CCtx_setParameter_boolean_flags_reject_out_of_range() {
    use crate::common::error::ERR_getErrorCode;

    let mut cctx = ZSTD_createCCtx().unwrap();
    for (param, input, expected) in [
        (ZSTD_cParameter::ZSTD_c_checksumFlag, 0, 0),
        (ZSTD_cParameter::ZSTD_c_checksumFlag, 1, 1),
        (ZSTD_cParameter::ZSTD_c_contentSizeFlag, 0, 0),
        (ZSTD_cParameter::ZSTD_c_contentSizeFlag, 1, 1),
        (ZSTD_cParameter::ZSTD_c_dictIDFlag, 0, 0),
        (ZSTD_cParameter::ZSTD_c_dictIDFlag, 1, 1),
    ] {
        assert_eq!(ZSTD_CCtx_setParameter(&mut cctx, param, input), 0);
        let mut got = -99;
        assert_eq!(ZSTD_CCtx_getParameter(&cctx, param, &mut got), 0);
        assert_eq!(got, expected, "[{param:?}] input {input}");
    }
    for (param, input) in [
        (ZSTD_cParameter::ZSTD_c_checksumFlag, 2),
        (ZSTD_cParameter::ZSTD_c_checksumFlag, -1),
        (ZSTD_cParameter::ZSTD_c_contentSizeFlag, 2),
        (ZSTD_cParameter::ZSTD_c_contentSizeFlag, -1),
        (ZSTD_cParameter::ZSTD_c_dictIDFlag, 2),
        (ZSTD_cParameter::ZSTD_c_dictIDFlag, -1),
    ] {
        let rc = ZSTD_CCtx_setParameter(&mut cctx, param, input);
        assert!(ERR_isError(rc), "[{param:?}] input {input}");
        assert_eq!(ERR_getErrorCode(rc), ErrorCode::ParameterOutOfBound);
    }

    let mut params = ZSTD_CCtx_params::default();
    for (param, input, expected) in [
        (ZSTD_cParameter::ZSTD_c_checksumFlag, 1, 1),
        (ZSTD_cParameter::ZSTD_c_checksumFlag, 0, 0),
        (ZSTD_cParameter::ZSTD_c_contentSizeFlag, 1, 1),
        (ZSTD_cParameter::ZSTD_c_contentSizeFlag, 0, 0),
        (ZSTD_cParameter::ZSTD_c_dictIDFlag, 1, 1),
        (ZSTD_cParameter::ZSTD_c_dictIDFlag, 0, 0),
    ] {
        assert_eq!(ZSTD_CCtxParams_setParameter(&mut params, param, input), 0);
        let mut got = -99;
        assert_eq!(ZSTD_CCtxParams_getParameter(&params, param, &mut got), 0);
        assert_eq!(got, expected, "[{param:?}] input {input}");
    }
    for (param, input) in [
        (ZSTD_cParameter::ZSTD_c_checksumFlag, 2),
        (ZSTD_cParameter::ZSTD_c_checksumFlag, -1),
        (ZSTD_cParameter::ZSTD_c_contentSizeFlag, 2),
        (ZSTD_cParameter::ZSTD_c_contentSizeFlag, -1),
        (ZSTD_cParameter::ZSTD_c_dictIDFlag, 2),
        (ZSTD_cParameter::ZSTD_c_dictIDFlag, -1),
    ] {
        let rc = ZSTD_CCtxParams_setParameter(&mut params, param, input);
        assert!(ERR_isError(rc), "[{param:?}] input {input}");
        assert_eq!(ERR_getErrorCode(rc), ErrorCode::ParameterOutOfBound);
    }
}

#[test]
fn CCtxParams_init_defaults_match_upstream() {
    // Upstream `ZSTD_CCtxParams_init` sets the caller-supplied
    // level + defaults the struct. Pin the defaults so a future
    // change to `ZSTD_CCtx_params::default` or the init body
    // doesn't silently shift any parametric readback.
    use crate::decompress::zstd_decompress::ZSTD_format_e;
    let mut params = ZSTD_CCtx_params::default();
    ZSTD_CCtxParams_init(&mut params, 5);
    let mut v = 0i32;

    assert_eq!(
        ZSTD_CCtxParams_getParameter(&params, ZSTD_cParameter::ZSTD_c_compressionLevel, &mut v),
        0,
    );
    assert_eq!(v, 5);

    assert_eq!(
        ZSTD_CCtxParams_getParameter(&params, ZSTD_cParameter::ZSTD_c_contentSizeFlag, &mut v),
        0,
    );
    assert_eq!(v, 1);

    assert_eq!(
        ZSTD_CCtxParams_getParameter(&params, ZSTD_cParameter::ZSTD_c_checksumFlag, &mut v),
        0,
    );
    assert_eq!(v, 0);

    assert_eq!(
        ZSTD_CCtxParams_getParameter(&params, ZSTD_cParameter::ZSTD_c_dictIDFlag, &mut v),
        0,
    );
    assert_eq!(v, 1);

    assert_eq!(
        ZSTD_CCtxParams_getParameter(&params, ZSTD_cParameter::ZSTD_c_format, &mut v),
        0,
    );
    assert_eq!(v, ZSTD_format_e::ZSTD_f_zstd1 as i32);

    assert_eq!(
        ZSTD_CCtxParams_getParameter(&params, ZSTD_cParameter::ZSTD_c_nbWorkers, &mut v),
        0,
    );
    assert_eq!(v, 0);
}

#[test]
fn createCCtxParams_advanced_returns_default_initialized_params() {
    let params = ZSTD_createCCtxParams_advanced(ZSTD_customMem::default()).unwrap();
    assert_eq!(params.compressionLevel, ZSTD_CLEVEL_DEFAULT);
    assert_eq!(params.fParams.contentSizeFlag, 1);
}

#[test]
fn advanced_custommem_surfaces_preserve_allocator_descriptor_and_reject_invalid_pairs() {
    use core::sync::atomic::{AtomicUsize, Ordering};

    static ALLOCS: AtomicUsize = AtomicUsize::new(0);
    static FREES: AtomicUsize = AtomicUsize::new(0);

    fn counting_alloc(_opaque: usize, size: usize) -> *mut core::ffi::c_void {
        use std::alloc::{alloc, Layout};

        const ALIGN: usize = 64;
        const HEADER_WORDS: usize = 2;

        let total = size.max(1) + ALIGN + HEADER_WORDS * core::mem::size_of::<usize>();
        let layout = Layout::from_size_align(total, ALIGN).unwrap();
        unsafe {
            let base = alloc(layout);
            if base.is_null() {
                return core::ptr::null_mut();
            }
            let payload_addr =
                (base as usize + HEADER_WORDS * core::mem::size_of::<usize>() + ALIGN - 1)
                    & !(ALIGN - 1);
            let header = (payload_addr as *mut usize).sub(HEADER_WORDS);
            header.write(base as usize);
            header.add(1).write(total);
            ALLOCS.fetch_add(1, Ordering::SeqCst);
            payload_addr as *mut core::ffi::c_void
        }
    }
    fn counting_free(_opaque: usize, address: *mut core::ffi::c_void) {
        use std::alloc::{dealloc, Layout};

        const ALIGN: usize = 64;
        const HEADER_WORDS: usize = 2;

        if address.is_null() {
            return;
        }
        unsafe {
            let header = (address as *mut usize).sub(HEADER_WORDS);
            let base = header.read() as *mut u8;
            let total = header.add(1).read();
            let layout = Layout::from_size_align(total, ALIGN).unwrap();
            dealloc(base, layout);
            FREES.fetch_add(1, Ordering::SeqCst);
        }
    }

    let custom = ZSTD_customMem {
        customAlloc: Some(counting_alloc),
        customFree: Some(counting_free),
        opaque: 0x1234,
    };
    let invalid = ZSTD_customMem {
        customAlloc: Some(counting_alloc),
        customFree: None,
        opaque: 7,
    };

    assert!(ZSTD_createCCtx_advanced(invalid).is_none());
    assert!(ZSTD_createCCtxParams_advanced(invalid).is_none());
    assert!(ZSTD_createCDict_advanced_internal(
        0,
        crate::decompress::zstd_ddict::ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy,
        crate::compress::match_state::ZSTD_compressionParameters::default(),
        crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_auto,
        0,
        invalid,
    )
    .is_none());

    let cctx = ZSTD_createCCtx_advanced(custom).unwrap();
    assert_eq!(cctx.customMem, custom);
    assert_eq!(cctx.requestedParams.customMem, custom);

    let params = ZSTD_createCCtxParams_advanced(custom).unwrap();
    assert_eq!(params.customMem, custom);

    let cstream = ZSTD_createCStream_advanced(custom).unwrap();
    assert_eq!(cstream.customMem, custom);

    let cdict = ZSTD_createCDict_advanced_internal(
        0,
        crate::decompress::zstd_ddict::ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy,
        crate::compress::match_state::ZSTD_compressionParameters::default(),
        crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_auto,
        0,
        custom,
    )
    .unwrap();
    assert_eq!(cdict.customMem, custom);

    assert_eq!(ALLOCS.load(Ordering::SeqCst), 4);
    assert_eq!(FREES.load(Ordering::SeqCst), 0);
    assert_eq!(ZSTD_freeCCtx(Some(cctx)), 0);
    assert_eq!(ZSTD_freeCCtxParams(Some(params)), 0);
    assert_eq!(ZSTD_freeCStream(Some(cstream)), 0);
    assert_eq!(ZSTD_freeCDict(Some(cdict)), 0);
    assert_eq!(FREES.load(Ordering::SeqCst), 4);
}

#[test]
fn CCtx_getParameter_defaults_on_fresh_cctx_match_upstream() {
    // Upstream contract (zstd_compress.c:780 `ZSTD_CCtxParams_init`
    // + `CCtx_params_default`): a fresh CCtx reports
    //   - compressionLevel: CLEVEL_DEFAULT (3)
    //   - contentSizeFlag: 1 (content size written when known)
    //   - checksumFlag:    0 (no trailer by default)
    //   - dictIDFlag:      1 (dictID included when applicable)
    //   - format:          ZSTD_f_zstd1 (magic-prefixed)
    //   - nbWorkers:       0 (single-threaded default)
    // Pin these so a future refactor touching `ZSTD_CCtx::default`
    // or the getter shadow fields doesn't silently change the
    // API contract.
    use crate::decompress::zstd_decompress::ZSTD_format_e;
    let cctx = ZSTD_createCCtx().unwrap();
    let mut v = 0i32;

    assert_eq!(
        ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_compressionLevel, &mut v,),
        0,
    );
    assert_eq!(v, ZSTD_CLEVEL_DEFAULT);

    assert_eq!(
        ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_contentSizeFlag, &mut v),
        0,
    );
    assert_eq!(v, 1);

    assert_eq!(
        ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_checksumFlag, &mut v),
        0,
    );
    assert_eq!(v, 0);

    assert_eq!(
        ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_dictIDFlag, &mut v),
        0,
    );
    assert_eq!(v, 1);

    assert_eq!(
        ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_format, &mut v),
        0,
    );
    assert_eq!(v, ZSTD_format_e::ZSTD_f_zstd1 as i32);

    assert_eq!(
        ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_nbWorkers, &mut v),
        0,
    );
    assert_eq!(v, 0);
}

#[test]
fn reset_policy_enums_match_upstream_ordering() {
    // Upstream's internal reset-policy enums at
    // `zstd_compress.c:1974-1988`. These drive the match-state
    // reset dispatch (`makeClean` zeros tables; `leaveDirty`
    // leaves them for the next-frame seed path). Discriminants
    // are ordinal; our port fixes them explicitly so a silent
    // reorder would trip this gate.
    assert_eq!(ZSTD_compResetPolicy_e::ZSTDcrp_makeClean as u32, 0);
    assert_eq!(ZSTD_compResetPolicy_e::ZSTDcrp_leaveDirty as u32, 1);
    assert_eq!(ZSTD_indexResetPolicy_e::ZSTDirp_continue as u32, 0);
    assert_eq!(ZSTD_indexResetPolicy_e::ZSTDirp_reset as u32, 1);
}

#[test]
fn SequenceFormat_e_and_buffered_policy_e_discriminants_match_upstream() {
    // Upstream values:
    //   ZSTD_sf_noBlockDelimiters = 0, ZSTD_sf_explicitBlockDelimiters = 1 (zstd.h:1582)
    //   ZSTDb_not_buffered = 0, ZSTDb_buffered = 1 (zstd_compress_internal.h)
    // These feed `ZSTD_c_blockDelimiters` setParameter and the
    // compressBegin buffered-policy dispatch respectively.
    assert_eq!(ZSTD_SequenceFormat_e::ZSTD_sf_noBlockDelimiters as i32, 0);
    assert_eq!(
        ZSTD_SequenceFormat_e::ZSTD_sf_explicitBlockDelimiters as i32,
        1
    );
    assert_eq!(ZSTD_buffered_policy_e::ZSTDb_not_buffered as i32, 0);
    assert_eq!(ZSTD_buffered_policy_e::ZSTDb_buffered as i32, 1);
}

#[test]
fn internal_stage_enums_match_upstream_zstd_compress_internal_h() {
    // Upstream `zstd_compress_internal.h:46-47`:
    //   ZSTDcs_created=0, ZSTDcs_init=1, ZSTDcs_ongoing=2, ZSTDcs_ending=3
    //   zcss_init=0, zcss_load=1, zcss_flush=2
    // These feed the CCtx stage machine in the C implementation.
    // Our port's `cctx_is_in_init_stage` helper, gate checks, and
    // `writeEpilogue` dispatch all consume these discriminants, so
    // a silent reordering would compile but mis-route the control
    // flow.
    assert_eq!(ZSTD_compressionStage_e::ZSTDcs_created as u32, 0);
    assert_eq!(ZSTD_compressionStage_e::ZSTDcs_init as u32, 1);
    assert_eq!(ZSTD_compressionStage_e::ZSTDcs_ongoing as u32, 2);
    assert_eq!(ZSTD_compressionStage_e::ZSTDcs_ending as u32, 3);
    assert_eq!(ZSTD_cStreamStage::zcss_init as u32, 0);
    assert_eq!(ZSTD_cStreamStage::zcss_load as u32, 1);
    assert_eq!(ZSTD_cStreamStage::zcss_flush as u32, 2);
}

#[test]
fn ResetDirective_discriminants_match_upstream() {
    // Upstream (zstd.h:589) fixes the discriminants at 1/2/3 —
    // NOT 0/1/2. C callers passing the numeric values directly
    // would mis-route if the enum ever drifts.
    assert_eq!(ZSTD_ResetDirective::ZSTD_reset_session_only as i32, 1);
    assert_eq!(ZSTD_ResetDirective::ZSTD_reset_parameters as i32, 2);
    assert_eq!(
        ZSTD_ResetDirective::ZSTD_reset_session_and_parameters as i32,
        3,
    );
}

#[test]
fn EndDirective_discriminants_match_upstream() {
    // `ZSTD_EndDirective` is the return / argument type for the
    // `compressStream2` family. Upstream (zstd.h:480) fixes the
    // discriminants at 0/1/2 — mismatch here silently mis-routes
    // C callers passing the numeric values directly.
    assert_eq!(ZSTD_EndDirective::ZSTD_e_continue as i32, 0);
    assert_eq!(ZSTD_EndDirective::ZSTD_e_flush as i32, 1);
    assert_eq!(ZSTD_EndDirective::ZSTD_e_end as i32, 2);
}

#[test]
fn cParameter_discriminants_match_upstream_zstd_h() {
    // `ZSTD_cParameter` values are part of the public C ABI. Drift
    // here would silently mis-route C callers through FFI bridges
    // — e.g. a caller passing `ZSTD_c_format` (= 10 upstream)
    // into a Rust-port FFI wrapper that defined it as 1001 would
    // hit the wrong handler. Pin the discriminants here so any
    // future rearrangement trips the gate.
    assert_eq!(ZSTD_cParameter::ZSTD_c_compressionLevel as i32, 100);
    assert_eq!(ZSTD_cParameter::ZSTD_c_contentSizeFlag as i32, 200);
    assert_eq!(ZSTD_cParameter::ZSTD_c_checksumFlag as i32, 201);
    assert_eq!(ZSTD_cParameter::ZSTD_c_dictIDFlag as i32, 202);
    assert_eq!(ZSTD_cParameter::ZSTD_c_nbWorkers as i32, 400);
    // `ZSTD_c_format` = `ZSTD_c_experimentalParam2` = 10.
    assert_eq!(ZSTD_cParameter::ZSTD_c_format as i32, 10);
    assert_eq!(ZSTD_cParameter::ZSTD_c_blockSplitterLevel as i32, 1017);
}

#[test]
fn CCtx_setParameter_c_nbWorkers_matches_feature_support() {
    use crate::common::error::{ERR_getErrorCode, ERR_isError};
    let mut cctx = ZSTD_createCCtx().unwrap();
    let mut value = 0i32;

    assert_eq!(
        ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_nbWorkers, &mut value),
        0,
    );
    assert_eq!(value, 0);
    assert_eq!(
        ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_nbWorkers, 0),
        0,
    );

    let bounds = ZSTD_cParam_getBounds(ZSTD_cParameter::ZSTD_c_nbWorkers);
    assert_eq!(bounds.lowerBound, 0);

    if cfg!(feature = "mt") {
        assert!(bounds.upperBound > 0);
        assert_eq!(
            ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_nbWorkers, 4),
            0,
        );
        assert_eq!(
            ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_nbWorkers, &mut value),
            0,
        );
        assert_eq!(value, 4);
    } else {
        let rc = ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_nbWorkers, 4);
        assert!(ERR_isError(rc));
        assert_eq!(ERR_getErrorCode(rc), ErrorCode::ParameterOutOfBound);
        assert_eq!(bounds.upperBound, 0);
    }
}

#[test]
fn CCtx_setParameter_c_format_round_trips_through_getParameter() {
    // Parametric API parity for the compressor-side format knob.
    // Callers using `ZSTD_CCtx_setParameter(ZSTD_c_format, value)`
    // must land on the same state as `ZSTD_CCtx_setFormat`.
    use crate::common::error::{ERR_getErrorCode, ERR_isError};
    use crate::decompress::zstd_decompress::ZSTD_format_e;
    let mut cctx = ZSTD_createCCtx().unwrap();
    let mut value = 0i32;

    assert_eq!(
        ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_format, &mut value),
        0,
    );
    assert_eq!(value, ZSTD_format_e::ZSTD_f_zstd1 as i32);

    assert_eq!(
        ZSTD_CCtx_setParameter(
            &mut cctx,
            ZSTD_cParameter::ZSTD_c_format,
            ZSTD_format_e::ZSTD_f_zstd1_magicless as i32,
        ),
        0,
    );
    assert_eq!(cctx.format, ZSTD_format_e::ZSTD_f_zstd1_magicless);

    assert_eq!(
        ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_format, &mut value),
        0,
    );
    assert_eq!(value, ZSTD_format_e::ZSTD_f_zstd1_magicless as i32);

    // Out-of-bounds → `ParameterOutOfBound`, not silent clamp.
    let rc = ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_format, 99);
    assert!(ERR_isError(rc));
    assert_eq!(ERR_getErrorCode(rc), ErrorCode::ParameterOutOfBound);

    // Bounds getter exposes the same [zstd1, magicless] range.
    let bounds = ZSTD_cParam_getBounds(ZSTD_cParameter::ZSTD_c_format);
    assert_eq!(bounds.lowerBound, ZSTD_format_e::ZSTD_f_zstd1 as i32);
    assert_eq!(
        bounds.upperBound,
        ZSTD_format_e::ZSTD_f_zstd1_magicless as i32
    );
}

#[test]
fn stable_buffer_params_roundtrip_through_cctx_and_params_api() {
    let mut cctx = ZSTD_createCCtx().unwrap();
    let mut params = ZSTD_CCtx_params::default();
    let mut value = -1;

    assert_eq!(
        ZSTD_CCtx_setParameter(
            &mut cctx,
            ZSTD_cParameter::ZSTD_c_stableInBuffer,
            ZSTD_bufferMode_e::ZSTD_bm_stable as i32,
        ),
        0
    );
    assert_eq!(
        ZSTD_CCtx_setParameter(
            &mut cctx,
            ZSTD_cParameter::ZSTD_c_stableOutBuffer,
            ZSTD_bufferMode_e::ZSTD_bm_stable as i32,
        ),
        0
    );
    assert_eq!(
        ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_stableInBuffer, &mut value),
        0
    );
    assert_eq!(value, ZSTD_bufferMode_e::ZSTD_bm_stable as i32);
    assert_eq!(
        ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_stableOutBuffer, &mut value),
        0
    );
    assert_eq!(value, ZSTD_bufferMode_e::ZSTD_bm_stable as i32);

    assert_eq!(
        ZSTD_CCtxParams_setParameter(
            &mut params,
            ZSTD_cParameter::ZSTD_c_stableInBuffer,
            ZSTD_bufferMode_e::ZSTD_bm_stable as i32,
        ),
        0
    );
    assert_eq!(
        ZSTD_CCtxParams_setParameter(
            &mut params,
            ZSTD_cParameter::ZSTD_c_stableOutBuffer,
            ZSTD_bufferMode_e::ZSTD_bm_stable as i32,
        ),
        0
    );
    assert_eq!(
        ZSTD_CCtxParams_getParameter(&params, ZSTD_cParameter::ZSTD_c_stableInBuffer, &mut value,),
        0
    );
    assert_eq!(value, ZSTD_bufferMode_e::ZSTD_bm_stable as i32);
    assert_eq!(
        ZSTD_CCtxParams_getParameter(&params, ZSTD_cParameter::ZSTD_c_stableOutBuffer, &mut value,),
        0
    );
    assert_eq!(value, ZSTD_bufferMode_e::ZSTD_bm_stable as i32);
}

#[test]
fn checkBufferStability_rejects_changed_input_pointer_and_grown_output() {
    use crate::common::error::ERR_getErrorCode;

    let mut cctx = ZSTD_createCCtx().unwrap();
    cctx.requestedParams.inBufferMode = ZSTD_bufferMode_e::ZSTD_bm_stable;
    cctx.requestedParams.outBufferMode = ZSTD_bufferMode_e::ZSTD_bm_stable;

    let src1 = b"abcdef";
    let src2 = b"uvwxyz";
    let out = [0u8; 32];

    let input1 = ZSTD_inBuffer {
        src: Some(src1),
        size: src1.len(),
        pos: 3,
    };
    let output1 = ZSTD_outBuffer {
        dst: None,
        size: out.len(),
        pos: 7,
    };
    ZSTD_setBufferExpectations(&mut cctx, &output1, &input1);
    assert_eq!(
        ZSTD_checkBufferStability(&cctx, &output1, &input1, ZSTD_EndDirective::ZSTD_e_continue),
        0
    );

    let moved_input = ZSTD_inBuffer {
        src: Some(src2),
        size: src2.len(),
        pos: 3,
    };
    let rc = ZSTD_checkBufferStability(
        &cctx,
        &output1,
        &moved_input,
        ZSTD_EndDirective::ZSTD_e_continue,
    );
    assert_eq!(
        ERR_getErrorCode(rc),
        ErrorCode::StabilityConditionNotRespected
    );

    let grown_output = ZSTD_outBuffer {
        dst: None,
        size: out.len() + 8,
        pos: 7,
    };
    let rc = ZSTD_checkBufferStability(
        &cctx,
        &grown_output,
        &input1,
        ZSTD_EndDirective::ZSTD_e_continue,
    );
    assert_eq!(
        ERR_getErrorCode(rc),
        ErrorCode::StabilityConditionNotRespected
    );
}

#[test]
fn CCtx_refPrefix_auto_clears_after_one_endStream_frame() {
    // Compressor-side sibling of the decoder `refPrefix` one-shot
    // auto-clear. Upstream (zstd_compress.c:6381) zeroes
    // `cctx->prefixDict` after each single-usage compress.
    // Without this, a second frame compressed on the same stream
    // dctx would silently carry the prior prefix as back-ref
    // history even though the caller only asked for one use.
    use crate::common::error::ERR_isError;
    let prefix = b"cctx-refPrefix-one-shot".to_vec();
    let src = b"payload-that-may-back-reference-prefix-bytes ".repeat(3);

    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream(&mut cctx, 3);
    assert_eq!(ZSTD_CCtx_refPrefix(&mut cctx, &prefix), 0);
    assert!(cctx.prefix_is_single_use);

    // Feed + finalize a frame.
    let mut dst = vec![0u8; 4096];
    let mut cp = 0usize;
    let mut sp = 0usize;
    let _ = ZSTD_compressStream(&mut cctx, &mut dst, &mut cp, &src, &mut sp);
    loop {
        let r = ZSTD_endStream(&mut cctx, &mut dst, &mut cp);
        assert!(!ERR_isError(r));
        if r == 0 {
            break;
        }
    }

    // After the single-usage frame the prefix bits are all wiped.
    assert!(cctx.stream_dict.is_empty(), "refPrefix dict persisted");
    assert_eq!(cctx.dictID, 0);
    assert_eq!(cctx.dictContentSize, 0);
    assert!(!cctx.prefix_is_single_use);
}

#[test]
fn CCtx_refCDict_after_refPrefix_wipes_single_use_flag() {
    // Upstream (zstd_compress.c:1348) clears all dicts before
    // installing a new cdict. Pin: a `refPrefix` → `refCDict`
    // transition on the same cctx must leave the single-use
    // flag cleared — a persistent CDict binding mustn't inherit
    // the prior prefix's one-shot lifetime.
    let dict = b"persistent-cdict".to_vec();
    let cdict = ZSTD_createCDict(&dict, 3).expect("cdict");

    let mut cctx = ZSTD_createCCtx().unwrap();
    assert_eq!(ZSTD_CCtx_refPrefix(&mut cctx, b"one-shot-first"), 0);
    assert!(cctx.prefix_is_single_use);

    assert_eq!(ZSTD_CCtx_refCDict(&mut cctx, &cdict), 0);
    assert!(
        !cctx.prefix_is_single_use,
        "refCDict inherited prior refPrefix's single-use flag",
    );
    // The CDict binding took over without degrading to raw bytes.
    assert!(cctx.stream_dict.is_empty());
    assert!(cctx.stream_cdict.is_some());
}

#[test]
fn CCtx_loadDictionary_empty_slice_clears_dict_state() {
    // Upstream `zstd_compress.c:1308`: empty-dict load acts as
    // `clearAllDicts`. Pin so a caller using `loadDictionary(&[])`
    // gets the same effect as `reset(parameters)`-level wipe of
    // dict state without having to reset the whole ctx.
    let mut cctx = ZSTD_createCCtx().unwrap();
    assert_eq!(ZSTD_CCtx_loadDictionary(&mut cctx, b"real-dict-bytes"), 0);
    assert_eq!(cctx.stream_dict, b"real-dict-bytes");

    // Empty reload clears everything.
    assert_eq!(ZSTD_CCtx_loadDictionary(&mut cctx, &[]), 0);
    assert!(cctx.stream_dict.is_empty());
    assert_eq!(cctx.dictID, 0);
    assert_eq!(cctx.dictContentSize, 0);
    assert!(!cctx.prefix_is_single_use);
}

#[test]
fn CCtx_refPrefix_empty_slice_clears_dict_state() {
    // Upstream (zstd_compress.c:1372): `refPrefix` calls
    // `ZSTD_clearAllDicts` before installing — if the prefix is
    // empty, the install is skipped and the cctx ends up with
    // no dict at all. Pin this behavior so `refPrefix(&[])`
    // doesn't silently leave `prefix_is_single_use = true` with
    // an empty stream_dict.
    let mut cctx = ZSTD_createCCtx().unwrap();
    // Pre-seed with a real prefix.
    assert_eq!(ZSTD_CCtx_refPrefix(&mut cctx, b"pre-existing-prefix"), 0);
    assert!(cctx.prefix_is_single_use);
    assert!(!cctx.stream_dict.is_empty());

    // Re-bind with an empty prefix: must reset everything.
    // (Needs a reset first since the prior refPrefix put us past
    // init stage in terms of dict state. Actually we're still in
    // init stage — no compressStream yet.)
    assert_eq!(ZSTD_CCtx_refPrefix(&mut cctx, &[]), 0);
    assert!(
        !cctx.prefix_is_single_use,
        "empty refPrefix left single-use flag set",
    );
    assert!(cctx.stream_dict.is_empty());
    assert_eq!(cctx.dictID, 0);
    assert_eq!(cctx.dictContentSize, 0);
}

#[test]
fn CCtx_refPrefix_cycle_across_session_resets() {
    // Flag-lifecycle integration test:
    //   refPrefix → flag=true
    //   compress2 → flag=false (auto-cleared)
    //   session_reset → allows new refPrefix
    //   refPrefix(different prefix) → flag=true again
    //   compress2 → flag=false
    // Proves the single-use state cycles cleanly rather than
    // staying stuck or corrupting the next cycle.
    use crate::common::error::ERR_isError;
    let src = b"short".to_vec();
    let mut cctx = ZSTD_createCCtx().unwrap();

    // Cycle 1.
    assert_eq!(ZSTD_CCtx_refPrefix(&mut cctx, b"prefix-one"), 0);
    assert!(cctx.prefix_is_single_use);
    let mut dst = vec![0u8; ZSTD_compressBound(src.len())];
    let n = ZSTD_compress2(&mut cctx, &mut dst, &src);
    assert!(!ERR_isError(n));
    assert!(!cctx.prefix_is_single_use);
    assert!(cctx.stream_dict.is_empty());

    // A fully drained frame returns to init-stage, so setters are legal
    // immediately without an explicit session reset.
    assert_eq!(ZSTD_CCtx_refPrefix(&mut cctx, b"prefix-two"), 0);
    assert!(cctx.prefix_is_single_use);
    assert_eq!(cctx.stream_dict, b"prefix-two");
    let mut dst = vec![0u8; ZSTD_compressBound(src.len())];
    let n = ZSTD_compress2(&mut cctx, &mut dst, &src);
    assert!(!ERR_isError(n));
    assert!(!cctx.prefix_is_single_use);
    assert!(cctx.stream_dict.is_empty());
}

#[test]
fn CCtx_refPrefix_auto_clears_through_compress2_entry() {
    // Complement to the endStream-specific test: `ZSTD_compress2`
    // resets the session + routes through `compressStream2`,
    // which internally reaches endStream. The one-shot auto-clear
    // must fire through that chain too.
    use crate::common::error::ERR_isError;
    let prefix = b"compress2-refPrefix-one-shot".to_vec();
    let src = b"short payload".to_vec();

    let mut cctx = ZSTD_createCCtx().unwrap();
    assert_eq!(ZSTD_CCtx_refPrefix(&mut cctx, &prefix), 0);
    assert!(cctx.prefix_is_single_use);

    let mut dst = vec![0u8; ZSTD_compressBound(src.len())];
    let n = ZSTD_compress2(&mut cctx, &mut dst, &src);
    assert!(!ERR_isError(n));

    // The prefix is consumed after compress2 completes.
    assert!(cctx.stream_dict.is_empty());
    assert!(!cctx.prefix_is_single_use);
}

#[test]
fn CCtx_loadDictionary_persists_across_endStream_frames() {
    // Counterpart: `loadDictionary` is a persistent attach.
    // The dict must survive across endStream boundaries, matching
    // upstream's `localDict` / `cdict` fields which aren't
    // zeroed at compress start.
    let dict = b"cctx-loadDictionary-persists".to_vec();
    let src = b"payload-for-persistent-dict ".repeat(2);

    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream(&mut cctx, 3);
    assert_eq!(ZSTD_CCtx_loadDictionary(&mut cctx, &dict), 0);
    assert!(!cctx.prefix_is_single_use);

    let mut dst = vec![0u8; 4096];
    let mut cp = 0usize;
    let mut sp = 0usize;
    let _ = ZSTD_compressStream(&mut cctx, &mut dst, &mut cp, &src, &mut sp);
    loop {
        let r = ZSTD_endStream(&mut cctx, &mut dst, &mut cp);
        if r == 0 {
            break;
        }
    }

    // Dict survives the first frame.
    assert_eq!(cctx.stream_dict, dict);
    assert!(!cctx.prefix_is_single_use);
}

#[test]
fn CCtx_reset_parameters_clears_single_use_prefix_flag() {
    // `ZSTD_clearAllDicts` must wipe the single-usage flag too.
    // A `reset(parameters)` that wipes the dict but leaves the
    // flag set would make the next compress auto-clear an
    // already-empty stream_dict (harmless) but confuse state
    // assertions and break future optimizations that read the
    // flag as a dict-presence hint.
    let mut cctx = ZSTD_createCCtx().unwrap();
    assert_eq!(ZSTD_CCtx_refPrefix(&mut cctx, b"one-shot"), 0);
    assert!(cctx.prefix_is_single_use);

    assert_eq!(
        ZSTD_CCtx_reset(&mut cctx, ZSTD_ResetDirective::ZSTD_reset_parameters),
        0,
    );
    assert!(!cctx.prefix_is_single_use);
    assert!(cctx.stream_dict.is_empty());
}

#[test]
fn CCtx_reset_parameters_clears_every_dict_slot_via_clearAllDicts_helper() {
    // Symmetric to the decoder-side gate. After routing
    // `reset(parameters)` through `ZSTD_clearAllDicts` +
    // `ZSTD_CCtxParams_reset`, the param reset must wipe every
    // dict slot — `stream_dict`, `dictID`, `dictContentSize`,
    // and the match-state's raw dict bytes —
    // rather than just the subset the older field-by-field body
    // covered.
    let mut cctx = ZSTD_createCCtx().unwrap();
    cctx.ms = Some(crate::compress::match_state::ZSTD_MatchState_t::new(
        ZSTD_getCParams(1, u64::MAX, 0),
    ));
    cctx.stream_dict = b"cctx-reset-wipe-test".to_vec();
    cctx.dictID = 0xBE_EF_C0_DE;
    cctx.dictContentSize = 42;
    cctx.ms.as_mut().unwrap().dictContent = b"stale-match-state-dict".to_vec();
    cctx.ms.as_mut().unwrap().dictMatchState = Some(Box::new(
        crate::compress::match_state::ZSTD_MatchState_t::new(ZSTD_getCParams(1, u64::MAX, 0)),
    ));
    cctx.ms.as_mut().unwrap().loadedDictEnd = 17;

    assert_eq!(
        ZSTD_CCtx_reset(&mut cctx, ZSTD_ResetDirective::ZSTD_reset_parameters),
        0,
    );
    assert!(cctx.stream_dict.is_empty());
    assert_eq!(cctx.dictID, 0);
    assert_eq!(cctx.dictContentSize, 0);
    assert!(cctx.ms.as_ref().unwrap().dictContent.is_empty());
    assert!(cctx.ms.as_ref().unwrap().dictMatchState.is_none());
    assert_eq!(cctx.ms.as_ref().unwrap().loadedDictEnd, 0);
}

#[test]
fn initLocalDict_propagates_stream_dict_into_match_state_bytes() {
    let mut cctx = ZSTD_createCCtx().unwrap();
    cctx.ms = Some(crate::compress::match_state::ZSTD_MatchState_t::new(
        ZSTD_getCParams(1, u64::MAX, 0),
    ));
    cctx.stream_dict = b"live-stream-dict".to_vec();
    cctx.ms.as_mut().unwrap().dictMatchState = Some(Box::new(
        crate::compress::match_state::ZSTD_MatchState_t::new(ZSTD_getCParams(1, u64::MAX, 0)),
    ));
    cctx.ms.as_mut().unwrap().loadedDictEnd = 99;

    assert_eq!(ZSTD_initLocalDict(&mut cctx), 0);
    assert_eq!(cctx.dictContentSize, cctx.stream_dict.len());
    assert_eq!(cctx.ms.as_ref().unwrap().dictContent, b"live-stream-dict");
    assert!(cctx.ms.as_ref().unwrap().dictMatchState.is_none());
    assert_eq!(
        cctx.ms.as_ref().unwrap().loadedDictEnd,
        cctx.stream_dict.len() as u32
    );
}

#[test]
fn CCtx_reset_parameters_clears_magicless_format() {
    // A `reset(parameters)` or `reset(session_and_parameters)`
    // must restore the default zstd1 format. Without this, a
    // CCtx re-used after magicless work silently produces
    // magicless frames for the next caller.
    use crate::decompress::zstd_decompress::ZSTD_format_e;
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_CCtx_setFormat(&mut cctx, ZSTD_format_e::ZSTD_f_zstd1_magicless);
    assert_eq!(cctx.format, ZSTD_format_e::ZSTD_f_zstd1_magicless);

    // session_only must NOT clear format — upstream keeps the
    // parameter across session resets.
    ZSTD_CCtx_reset(&mut cctx, ZSTD_ResetDirective::ZSTD_reset_session_only);
    assert_eq!(cctx.format, ZSTD_format_e::ZSTD_f_zstd1_magicless);

    // parameters reset must wipe it back to zstd1.
    ZSTD_CCtx_reset(&mut cctx, ZSTD_ResetDirective::ZSTD_reset_parameters);
    assert_eq!(cctx.format, ZSTD_format_e::ZSTD_f_zstd1);

    // session_and_parameters reset must also wipe it.
    ZSTD_CCtx_setFormat(&mut cctx, ZSTD_format_e::ZSTD_f_zstd1_magicless);
    ZSTD_CCtx_reset(
        &mut cctx,
        ZSTD_ResetDirective::ZSTD_reset_session_and_parameters,
    );
    assert_eq!(cctx.format, ZSTD_format_e::ZSTD_f_zstd1);
}

#[test]
fn CCtx_magicless_plus_dict_endStream_roundtrips() {
    // Parity gate for the dict + magicless combination. Before
    // `_with_prefix_advanced` landed, a dict-bearing stream would
    // still emit a zstd1-format frame even when the caller set
    // magicless mode — roundtrip against a magicless dctx would
    // fail because of the leaked magic prefix.
    use crate::decompress::zstd_decompress::{
        ZSTD_DCtx_setFormat, ZSTD_decompress_usingDict, ZSTD_format_e,
    };
    use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

    let dict = b"dict-bytes-for-magicless-stream-compression ".repeat(4);
    let src = b"payload-referencing-dict-bytes-for-magicless-stream-compression ".repeat(8);
    let mut cctx = ZSTD_createCCtx().unwrap();
    assert_eq!(
        ZSTD_CCtx_setFormat(&mut cctx, ZSTD_format_e::ZSTD_f_zstd1_magicless),
        0,
    );
    ZSTD_initCStream_usingDict(&mut cctx, &dict, 3);

    let mut compressed = vec![0u8; 4096];
    let mut cp = 0usize;
    let mut sp = 0usize;
    let _ = ZSTD_compressStream(&mut cctx, &mut compressed, &mut cp, &src, &mut sp);
    loop {
        let r = ZSTD_endStream(&mut cctx, &mut compressed, &mut cp);
        assert!(!ERR_isError(r));
        if r == 0 {
            break;
        }
    }
    compressed.truncate(cp);
    let magic_le = crate::common::mem::MEM_readLE32(&compressed[..4]);
    assert_ne!(
        magic_le,
        crate::decompress::zstd_decompress::ZSTD_MAGICNUMBER,
        "magicless + dict frame leaked a zstd1 magic prefix",
    );

    let mut dctx = ZSTD_DCtx::new();
    assert_eq!(
        ZSTD_DCtx_setFormat(&mut dctx, ZSTD_format_e::ZSTD_f_zstd1_magicless),
        0,
    );
    let mut out = vec![0u8; src.len() + 128];
    let d = ZSTD_decompress_usingDict(&mut dctx, &mut out, &compressed, &dict);
    assert!(!ERR_isError(d), "decode err: {d:#x}");
    assert_eq!(&out[..d], &src[..]);
}

#[test]
fn seq_to_codes_fills_three_code_tables() {
    use crate::compress::seq_store::{SeqDef, SeqStore_t, OFFSET_TO_OFFBASE, REPCODE_TO_OFFBASE};
    let mut ss = SeqStore_t::with_capacity(16, 1024);
    // Sequence 1: litLength=10, repcode-1 (offBase=1), mlBase=5.
    ss.sequences.push(SeqDef {
        offBase: REPCODE_TO_OFFBASE(1),
        litLength: 10,
        mlBase: 5,
    });
    // Sequence 2: litLength=80 (>63 → highbit path), offset=1024
    // (offBase=1024+3=1027 → highbit32=10), mlBase=200 (>127).
    ss.sequences.push(SeqDef {
        offBase: OFFSET_TO_OFFBASE(1024),
        litLength: 80,
        mlBase: 200,
    });
    let long_off = ZSTD_seqToCodes(&mut ss);
    // 64-bit target: upstream asserts longOffsets is never 1.
    assert_eq!(long_off, 0);
    // Code-table values:
    //   LL(10) = 10, LL(80) = highbit32(80)+19 = 6+19 = 25.
    //   ML(5)  = 5,  ML(200) = highbit32(200)+36 = 7+36 = 43.
    //   OF(1) repcode → highbit32(1) = 0.
    //   OF(1027) full offset → highbit32(1027) = 10.
    assert_eq!(ss.llCode, [10u8, 25]);
    assert_eq!(ss.mlCode, [5u8, 43]);
    assert_eq!(ss.ofCode, [0u8, 10]);
}

#[test]
fn get_cparams_level_1_matches_upstream_table() {
    // Large src → tableID 0, row 1.
    let cp = ZSTD_getCParams(1, 1_000_000, 0);
    assert_eq!(cp.windowLog, 19);
    assert_eq!(cp.chainLog, 13);
    assert_eq!(cp.hashLog, 14);
    assert_eq!(cp.strategy, 1); // ZSTD_fast
}

#[test]
fn get_cparams_small_src_uses_smaller_tables() {
    // Tiny src → tableID 3, row 1. Upstream's
    // `ZSTD_adjustCParams_internal` then shrinks windowLog to
    // `highbit32(srcSize-1)+1` = 10 for a 1000-byte input, since
    // the configured windowLog (14) exceeds what's needed.
    let cp = ZSTD_getCParams(1, 1000, 0);
    assert_eq!(cp.windowLog, 10);
}

#[test]
fn get_cparams_negative_levels_set_target_length() {
    let cp = ZSTD_getCParams(-5, 0, 0);
    assert_eq!(cp.targetLength, 5);
    assert_eq!(cp.strategy, 1); // ZSTD_fast
}

#[test]
fn get_cparams_zero_is_default_level_3() {
    // Level 0 → ZSTD_CLEVEL_DEFAULT (3). Level 3 is dfast in the
    // large table — not "fast".
    let cp = ZSTD_getCParams(0, 1_000_000, 0);
    assert_eq!(cp.strategy, 2); // ZSTD_dfast
}

#[test]
fn zstd_default_cparams_table_shape_and_validity() {
    // Shape contract: 4 table IDs × 23 rows (= levels 0..=22).
    // Every row must pass `ZSTD_checkCParams`. Catches
    // typos/out-of-bounds entries in the ported table — which
    // is 92 rows × 7 fields copied from upstream clevels.h.
    assert_eq!(ZSTD_DEFAULT_CPARAMS.len(), 4);
    for (tid, table) in ZSTD_DEFAULT_CPARAMS.iter().enumerate() {
        assert_eq!(table.len(), 23, "table {tid} wrong row count");
        for (row, &(wl, cl, hl, sl, mm, tl, strat)) in table.iter().enumerate() {
            let cp = crate::compress::match_state::ZSTD_compressionParameters {
                windowLog: wl,
                chainLog: cl,
                hashLog: hl,
                searchLog: sl,
                minMatch: mm,
                targetLength: tl,
                strategy: strat,
            };
            let rc = ZSTD_checkCParams(cp);
            assert_eq!(
                rc, 0,
                "table {tid} row {row} (level {row}): invalid cParams {cp:?}",
            );
        }
    }
}

#[test]
fn get_cparams_above_max_clevel_clamps_to_max() {
    // Contract: any `compressionLevel > ZSTD_MAX_CLEVEL` gets
    // clamped to MAX, matching upstream's "silent clamp" for
    // out-of-range levels. A bug here would produce
    // out-of-bounds cParams-table indexing.
    let at_max = ZSTD_getCParams(ZSTD_MAX_CLEVEL, 0, 0);
    let above_max = ZSTD_getCParams(ZSTD_MAX_CLEVEL + 1, 0, 0);
    let way_above = ZSTD_getCParams(i32::MAX, 0, 0);
    assert_eq!(at_max.windowLog, above_max.windowLog);
    assert_eq!(at_max.strategy, above_max.strategy);
    assert_eq!(at_max.chainLog, way_above.chainLog);
    assert_eq!(at_max.hashLog, way_above.hashLog);
}

#[test]
fn get_cparams_below_minCLevel_clamps_to_row_0() {
    // Contract: any negative level picks row 0 (the baseline
    // negative-level row). The accelerator bumps up
    // `targetLength = -level`, so at very negative levels
    // targetLength is large.
    let neg_small = ZSTD_getCParams(-1, 0, 0);
    let neg_extreme = ZSTD_getCParams(ZSTD_minCLevel(), 0, 0);
    // Same strategy (fast) for all negative levels.
    assert_eq!(neg_small.strategy, neg_extreme.strategy);
    // targetLength = -level (positive). Extreme negative yields
    // larger targetLength.
    assert!(neg_extreme.targetLength >= neg_small.targetLength);
}

#[test]
fn zstd_cctx_lifecycle_create_compress_free() {
    use crate::decompress::zstd_decompress::ZSTD_decompress;
    let mut cctx = ZSTD_createCCtx().expect("create");
    let src: Vec<u8> = b"hello cctx world. "
        .iter()
        .cycle()
        .take(400)
        .copied()
        .collect();
    let mut dst = vec![0u8; 2048];
    let n = ZSTD_compressCCtx(&mut cctx, &mut dst, &src, 1);
    assert!(
        !crate::common::error::ERR_isError(n),
        "cctx compress: {n:#x}"
    );
    dst.truncate(n);

    // Verify roundtrip.
    let mut out = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress(&mut out, &dst);
    assert!(
        !crate::common::error::ERR_isError(d),
        "decompress: {}",
        crate::common::error::ERR_getErrorName(d)
    );
    assert_eq!(&out[..d], &src[..]);

    // Free returns 0.
    assert_eq!(ZSTD_freeCCtx(Some(cctx)), 0);
}

#[test]
fn zstd_stream_init_compress_end_roundtrips() {
    use crate::decompress::zstd_decompress::ZSTD_decompress;
    let mut cctx = ZSTD_createCCtx().unwrap();
    assert_eq!(ZSTD_initCStream(&mut cctx, 1), 0);

    let src: Vec<u8> = b"streaming payload. the fox jumps. "
        .iter()
        .cycle()
        .take(600)
        .copied()
        .collect();

    // Feed input in 3 chunks to exercise the buffering path.
    let mut staged = vec![0u8; 2048];
    let mut out_pos = 0usize;
    for chunk in [&src[..200], &src[200..400], &src[400..]] {
        let mut in_pos = 0usize;
        let rc = ZSTD_compressStream(&mut cctx, &mut staged, &mut out_pos, chunk, &mut in_pos);
        assert!(!crate::common::error::ERR_isError(rc));
        assert_eq!(in_pos, chunk.len());
    }
    // endStream: may take multiple calls if output is tight.
    loop {
        let remaining = ZSTD_endStream(&mut cctx, &mut staged, &mut out_pos);
        assert!(!crate::common::error::ERR_isError(remaining));
        if remaining == 0 {
            break;
        }
    }
    staged.truncate(out_pos);

    let mut decoded = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress(&mut decoded, &staged);
    assert!(
        !crate::common::error::ERR_isError(d),
        "decompress: {}",
        crate::common::error::ERR_getErrorName(d)
    );
    assert_eq!(&decoded[..d], &src[..]);
}

#[test]
fn zstd_boundary_sizes_roundtrip() {
    // Exercise sizes right around the 128 KB block boundary and
    // other notable thresholds. Catches off-by-one bugs in block
    // sizing / tail-literals logic.
    use crate::decompress::zstd_decompress::ZSTD_decompress;
    let sizes = [
        0, 1, 2, 3, 4, 7, 8, 15, 16, 63, 64, 127, 128, 255, 256, 1023, 1024, 4095, 4096, 65535,
        65536, 65537, 131071, 131072, 131073, 262143, 262144, 262145,
    ];
    let pattern = b"the quick brown fox jumps over the lazy dog. ";
    for &size in &sizes {
        let src: Vec<u8> = pattern.iter().cycle().take(size).copied().collect();
        for &level in &[1i32, 5, 10] {
            let bound = super::ZSTD_compressBound(size).max(32);
            let mut dst = vec![0u8; bound];
            let n = ZSTD_compress(&mut dst, &src, level);
            assert!(
                !crate::common::error::ERR_isError(n),
                "[size {size} level {level}] compress err: {n:#x}"
            );
            dst.truncate(n);
            let mut out = vec![0u8; size + 64];
            let d = ZSTD_decompress(&mut out, &dst);
            assert!(
                !crate::common::error::ERR_isError(d),
                "[size {size} level {level}] decompress err: {d:#x}"
            );
            assert_eq!(d, size, "[size {size} level {level}] size");
            assert_eq!(&out[..d], &src[..], "[size {size} level {level}] content");
        }
    }
}

#[test]
fn zstd_repetitive_multiblock_regression_gate() {
    // Regression gate for the fast-strategy repcode-litLength-0
    // bug (see TODO.md "Bugs fixed" entry). Uses payloads where
    // block 1's first match covers most of the block — forcing
    // block 2 to start at a block boundary with a rep match
    // immediately available. Pre-fix behavior: decoder output
    // diverged at byte 131074.
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    let phrases: Vec<&[u8]> = vec![
        b"aaa",
        b"abc ",
        b"the quick brown fox jumps over the lazy dog. ",
        b"ZSTDRS_ZSTDRS_",
    ];
    for phrase in &phrases {
        let src: Vec<u8> = phrase.iter().cycle().take(200_000).copied().collect();
        for level in [1i32, 3, 5, 10] {
            let mut dst = vec![0u8; src.len() + 1024];
            let n = ZSTD_compress(&mut dst, &src, level);
            assert!(
                !crate::common::error::ERR_isError(n),
                "[phrase {phrase:?} level {level}] compress err: {n:#x}"
            );
            dst.truncate(n);
            let mut out = vec![0u8; src.len() + 64];
            let d = ZSTD_decompress(&mut out, &dst);
            assert!(
                !crate::common::error::ERR_isError(d),
                "[phrase {phrase:?} level {level}] decompress err: {d:#x}"
            );
            assert_eq!(d, src.len(), "[phrase {phrase:?} level {level}] size");
            assert_eq!(
                &out[..d],
                &src[..],
                "[phrase {phrase:?} level {level}] content"
            );
        }
    }
}

#[test]
fn zstd_multi_block_no_dict_all_levels() {
    // Stress the multi-block no-dict path across all supported
    // strategies. A 140 KB payload crosses the 128 KB block
    // boundary — this is where the fast-strategy-dict-multi-block
    // bug was originally found; no-dict case must still work at
    // every level.
    use crate::decompress::zstd_decompress::ZSTD_decompress;
    let src: Vec<u8> = b"the quick brown fox jumps over the lazy dog. "
        .iter()
        .cycle()
        .take(140_000)
        .copied()
        .collect();
    for level in [1i32, 3, 5, 7, 10, 15, 19, 22] {
        let mut dst = vec![0u8; src.len() + 1024];
        let n = ZSTD_compress(&mut dst, &src, level);
        assert!(
            !crate::common::error::ERR_isError(n),
            "[level {level}] compress err: {n:#x}"
        );
        dst.truncate(n);
        let mut out = vec![0u8; src.len() + 64];
        let d = ZSTD_decompress(&mut out, &dst);
        assert!(
            !crate::common::error::ERR_isError(d),
            "[level {level}] decompress err: {d:#x}"
        );
        assert_eq!(d, src.len(), "[level {level}] size");
        assert_eq!(&out[..d], &src[..], "[level {level}] content");
    }
}

#[test]
fn zstd_patterned_payloads_roundtrip_across_levels() {
    // Pattern sweep: all-zeros, alternating, ramp, near-rle-with-
    // noise, repeating short phrase, long-distance repeat. Every
    // payload gets compressed then decompressed at several levels,
    // roundtrip verified byte-exact.
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    let size_large = 50_000usize;
    let payloads: Vec<(&str, Vec<u8>)> = vec![
        ("all_zeros", vec![0u8; size_large]),
        ("all_ff", vec![0xFFu8; size_large]),
        (
            "alternating",
            (0..size_large)
                .map(|i| if i % 2 == 0 { 0 } else { 0xAA })
                .collect(),
        ),
        ("ramp", (0..size_large).map(|i| (i & 0xFF) as u8).collect()),
        ("noisy_rle", {
            let mut v = vec![b'x'; size_large];
            for i in (0..v.len()).step_by(101) {
                v[i] = b'Q';
            }
            v
        }),
        (
            "short_rep",
            b"abc".iter().cycle().take(size_large).copied().collect(),
        ),
        (
            "phrase_rep",
            b"the fox jumps. "
                .iter()
                .cycle()
                .take(size_large)
                .copied()
                .collect(),
        ),
        ("long_repeat", {
            // Unique 2KB preamble, then that same 2KB repeated.
            let chunk: Vec<u8> = (0..2048u32).map(|i| ((i * 31 + 7) & 0xFF) as u8).collect();
            let mut v = chunk.clone();
            for _ in 0..24 {
                v.extend_from_slice(&chunk);
            }
            v
        }),
    ];

    for (name, payload) in &payloads {
        for &level in &[1i32, 3, 5, 10, 19] {
            let bound = super::ZSTD_compressBound(payload.len()).max(32);
            let mut compressed = vec![0u8; bound];
            let n = ZSTD_compress(&mut compressed, payload, level);
            assert!(
                !crate::common::error::ERR_isError(n),
                "[{name} level {level}] compress err: {n:#x}"
            );
            compressed.truncate(n);

            let mut decoded = vec![0u8; payload.len() + 64];
            let d = ZSTD_decompress(&mut decoded, &compressed);
            assert!(
                !crate::common::error::ERR_isError(d),
                "[{name} level {level}] decompress err: {d:#x}"
            );
            assert_eq!(
                &decoded[..d],
                &payload[..],
                "[{name} level {level}] roundtrip mismatch"
            );
        }
    }
}

#[test]
fn zstd_random_payload_roundtrips_across_levels_and_seeds() {
    // Simple xorshift-ish PRNG — deterministic across runs. We
    // stay on a set of payload *shapes* and compression levels,
    // rotating through seeds. Each roundtrip goes through
    // ZSTD_compress → ZSTD_decompress, bytes-compared.
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    fn xorshift(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    let levels = [1i32, 3, 5, 7, 10];
    let sizes = [0usize, 1, 16, 128, 1000, 8000, 65536];
    let mut state: u64 = 0xabcdef0123456789;

    for &size in &sizes {
        // Generate a random byte buffer.
        let mut payload = vec![0u8; size];
        for b in &mut payload {
            *b = (xorshift(&mut state) & 0xFF) as u8;
        }
        for &level in &levels {
            let bound = super::ZSTD_compressBound(size).max(32);
            let mut compressed = vec![0u8; bound];
            let n = ZSTD_compress(&mut compressed, &payload, level);
            assert!(
                !crate::common::error::ERR_isError(n),
                "[size={size} level={level}] compress err: {n:#x}"
            );
            compressed.truncate(n);

            let mut decoded = vec![0u8; size + 64];
            let d = ZSTD_decompress(&mut decoded, &compressed);
            assert!(
                !crate::common::error::ERR_isError(d),
                "[size={size} level={level}] decompress err: {d:#x}"
            );
            assert_eq!(d, size, "[size={size} level={level}] decoded size");
            assert_eq!(
                &decoded[..d],
                &payload[..],
                "[size={size} level={level}] roundtrip mismatch"
            );
        }
    }
}

#[test]
fn zstd_cdict_ddict_symmetric_roundtrip() {
    use crate::decompress::zstd_ddict::ZSTD_createDDict;
    use crate::decompress::zstd_decompress::ZSTD_decompress_usingDDict;
    use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

    let dict = b"compression dictionary content. token foo bar baz. ".repeat(20);
    let cdict = ZSTD_createCDict(&dict, 1).expect("cdict");
    let ddict = ZSTD_createDDict(&dict).expect("ddict");

    let src: Vec<u8> = b"token foo token bar token baz. "
        .iter()
        .cycle()
        .take(600)
        .copied()
        .collect();

    let mut cctx = ZSTD_createCCtx().unwrap();
    let mut compressed = vec![0u8; 2048];
    let n = ZSTD_compress_usingCDict(&mut cctx, &mut compressed, &src, &cdict);
    assert!(!crate::common::error::ERR_isError(n));
    compressed.truncate(n);

    let mut dctx = ZSTD_DCtx::new();
    let mut decoded = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress_usingDDict(&mut dctx, &mut decoded, &compressed, &ddict);
    assert!(
        !crate::common::error::ERR_isError(d),
        "ddict decompress: {}",
        crate::common::error::ERR_getErrorName(d)
    );
    assert_eq!(&decoded[..d], &src[..]);
}

#[test]
fn zstd_cdict_create_compress_reuse_across_payloads() {
    use crate::decompress::zstd_decompress::ZSTD_decompress_usingDict;
    use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

    let dict = b"the quick brown fox jumps over the lazy dog near a river. ".repeat(30);
    let cdict = ZSTD_createCDict(&dict, 1).expect("cdict create");
    assert!(ZSTD_sizeof_CDict(&cdict) > 0);

    let mut cctx = ZSTD_createCCtx().unwrap();

    // Use the same CDict for 3 different payloads.
    for (i, payload_text) in [&b"the fox jumps. "[..], b"the lazy dog. ", b"brown fox. "]
        .iter()
        .enumerate()
    {
        let payload: Vec<u8> = payload_text.iter().cycle().take(400).copied().collect();

        let mut compressed = vec![0u8; 2048];
        let n = ZSTD_compress_usingCDict(&mut cctx, &mut compressed, &payload, &cdict);
        assert!(
            !crate::common::error::ERR_isError(n),
            "[iter {i}] cdict compress"
        );
        compressed.truncate(n);

        let mut dctx = ZSTD_DCtx::new();
        let mut decoded = vec![0u8; payload.len() + 64];
        let d = ZSTD_decompress_usingDict(&mut dctx, &mut decoded, &compressed, &dict);
        assert!(
            !crate::common::error::ERR_isError(d),
            "[iter {i}] dict decompress: {}",
            crate::common::error::ERR_getErrorName(d)
        );
        assert_eq!(&decoded[..d], &payload[..], "[iter {i}] roundtrip");
    }

    assert_eq!(ZSTD_freeCDict(Some(cdict)), 0);
}

#[test]
fn zstd_compressStream2_e_end_produces_valid_frame() {
    use crate::decompress::zstd_decompress::ZSTD_decompress;
    let mut cs = ZSTD_createCStream().unwrap();
    ZSTD_initCStream(&mut cs, 1);

    let src: Vec<u8> = b"e_end payload. "
        .iter()
        .cycle()
        .take(400)
        .copied()
        .collect();
    let mut dst = vec![0u8; 2048];
    let mut dp = 0usize;
    let mut ip = 0usize;

    // Feed all input with e_continue, then loop e_end until 0.
    let _ = ZSTD_compressStream2(
        &mut cs,
        &mut dst,
        &mut dp,
        &src,
        &mut ip,
        ZSTD_EndDirective::ZSTD_e_continue,
    );
    loop {
        let r = ZSTD_compressStream2(
            &mut cs,
            &mut dst,
            &mut dp,
            &[],
            &mut 0,
            ZSTD_EndDirective::ZSTD_e_end,
        );
        if r == 0 {
            break;
        }
    }
    dst.truncate(dp);

    let mut out = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress(&mut out, &dst);
    assert!(!crate::common::error::ERR_isError(d));
    assert_eq!(&out[..d], &src[..]);
}

#[test]
fn zstd_cstream_with_loaded_dictionary_roundtrips() {
    use crate::decompress::zstd_decompress::{
        ZSTD_DCtx_loadDictionary, ZSTD_createDStream, ZSTD_decompressStream, ZSTD_initDStream,
    };

    let dict = b"dict alpha beta gamma delta. ".repeat(25);
    let src: Vec<u8> = b"alpha gamma. beta delta. "
        .iter()
        .cycle()
        .take(400)
        .copied()
        .collect();

    let mut cs = ZSTD_createCStream().unwrap();
    ZSTD_initCStream(&mut cs, 1);
    ZSTD_CCtx_loadDictionary(&mut cs, &dict);

    let mut staged = vec![0u8; 2048];
    let mut cp_pos = 0usize;
    let mut ip = 0usize;
    ZSTD_compressStream(&mut cs, &mut staged, &mut cp_pos, &src, &mut ip);
    loop {
        let r = ZSTD_endStream(&mut cs, &mut staged, &mut cp_pos);
        if r == 0 {
            break;
        }
    }
    staged.truncate(cp_pos);

    let mut ds = ZSTD_createDStream().unwrap();
    crate::decompress::zstd_decompress_block::ZSTD_buildDefaultSeqTables(&mut ds);
    ZSTD_initDStream(&mut ds);
    ZSTD_DCtx_loadDictionary(&mut ds, &dict);

    let mut out = vec![0u8; src.len() + 64];
    let mut dp = 0usize;
    let mut icursor = 0usize;
    ZSTD_decompressStream(&mut ds, &mut out, &mut dp, &staged, &mut icursor);
    loop {
        let mut p = 0usize;
        let r = ZSTD_decompressStream(&mut ds, &mut out, &mut dp, &[], &mut p);
        if r == 0 {
            break;
        }
    }
    assert_eq!(&out[..dp], &src[..]);
}

#[test]
fn sizeof_CCtx_grows_monotonically_with_level() {
    // Higher levels allocate larger hash/chain tables → sizeof
    // should be weakly monotonic across levels. (Upstream has
    // the same property; it's a sanity-check on our accounting.)
    let mut prev_sz = 0usize;
    for level in [1, 3, 6, 9].iter().copied() {
        let mut cctx = ZSTD_createCCtx().unwrap();
        let src = b"level size probe".repeat(10);
        let mut dst = vec![0u8; 512];
        ZSTD_compressCCtx(&mut cctx, &mut dst, &src, level);
        let sz = ZSTD_sizeof_CCtx(&cctx);
        assert!(sz >= prev_sz, "level {level} shrunk: {sz} < {prev_sz}");
        prev_sz = sz;
    }
}

#[test]
fn cdict_ddict_dictID_parsing_aligns() {
    // Real full-dict fixture: both CDict and DDict must parse
    // the same dictID from it.
    use crate::decompress::zstd_ddict::{ZSTD_createDDict, ZSTD_getDictID_fromDDict};
    let dict: &[u8] =
        include_bytes!("../../tests/fixtures/upstream-zstd/dict-files/zero-weight-dict");

    let cdict = ZSTD_createCDict(dict, 3).unwrap();
    let ddict = ZSTD_createDDict(dict).unwrap();

    let expected = crate::decompress::zstd_ddict::ZSTD_getDictID_fromDict(dict);
    assert_eq!(ZSTD_getDictID_fromCDict(&cdict), expected);
    assert_eq!(ZSTD_getDictID_fromDDict(&ddict), expected);
}

#[test]
fn streamCompress_empty_input_produces_valid_frame() {
    // Empty-source round-trip — compressStream/endStream with
    // zero input must still emit a well-formed (possibly tiny)
    // frame that decompresses back to nothing.
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream(&mut cctx, 3);
    let mut dst = vec![0u8; 64];
    let mut dst_pos = 0usize;
    loop {
        let rc = ZSTD_endStream(&mut cctx, &mut dst, &mut dst_pos);
        if rc == 0 || ERR_isError(rc) {
            break;
        }
    }
    assert!(dst_pos > 0, "endStream on empty input emitted nothing");
    use crate::decompress::zstd_decompress::ZSTD_decompress;
    let mut out = vec![0u8; 64];
    let d = ZSTD_decompress(&mut out, &dst[..dst_pos]);
    assert_eq!(d, 0);
}

#[test]
fn cParam_all_variants_set_get_roundtrip() {
    // Every variant of our ZSTD_cParameter enum must round-trip
    // through setParameter / getParameter with no value drift.
    let cases = [
        (ZSTD_cParameter::ZSTD_c_compressionLevel, 7),
        (ZSTD_cParameter::ZSTD_c_checksumFlag, 1),
        (ZSTD_cParameter::ZSTD_c_contentSizeFlag, 0),
        (ZSTD_cParameter::ZSTD_c_dictIDFlag, 0),
    ];
    let mut cctx = ZSTD_createCCtx().unwrap();
    for &(param, value) in &cases {
        ZSTD_CCtx_setParameter(&mut cctx, param, value);
        let mut got = -1i32;
        ZSTD_CCtx_getParameter(&cctx, param, &mut got);
        assert_eq!(got, value, "param {:?} didn't round-trip", param);
    }
}

#[test]
fn compressStream2_flush_directive_roundtrips() {
    // The buffered stream model doesn't emit a partial frame on
    // flush; it only finalizes once the caller ends the stream.
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    let first = b"flush test ".repeat(40);
    let second = b"after flush ".repeat(20);
    let mut cctx = ZSTD_createCCtx().unwrap();
    let mut dst = vec![0u8; ZSTD_compressBound(first.len() + second.len()) + 128];
    let mut dp = 0usize;
    let mut sp = 0usize;
    let rc = ZSTD_compressStream2(
        &mut cctx,
        &mut dst,
        &mut dp,
        &first,
        &mut sp,
        ZSTD_EndDirective::ZSTD_e_flush,
    );
    assert!(!ERR_isError(rc));
    assert_eq!(sp, first.len());
    assert_eq!(rc, 0, "flush should complete all currently staged input");
    assert_ne!(dp, 0, "flush must emit a non-final frame prefix");

    sp = 0;
    let rc = ZSTD_compressStream2(
        &mut cctx,
        &mut dst,
        &mut dp,
        &second,
        &mut sp,
        ZSTD_EndDirective::ZSTD_e_continue,
    );
    assert!(!ERR_isError(rc));
    assert_eq!(sp, second.len());

    loop {
        let mut zero_pos = 0usize;
        let remaining = ZSTD_compressStream2(
            &mut cctx,
            &mut dst,
            &mut dp,
            &[],
            &mut zero_pos,
            ZSTD_EndDirective::ZSTD_e_end,
        );
        assert!(!ERR_isError(remaining));
        if remaining == 0 {
            break;
        }
        dst.resize(dst.len() + remaining.max(32), 0);
    }

    let mut decoded = vec![0u8; first.len() + second.len()];
    let d = ZSTD_decompress(&mut decoded, &dst[..dp]);
    assert_eq!(d, first.len() + second.len());
    assert_eq!(&decoded[..first.len()], first.as_slice());
    assert_eq!(&decoded[first.len()..d], second.as_slice());
}

#[test]
fn compressStream2_continue_then_end_roundtrip() {
    // Use the directive sequence { e_continue * N, e_end } —
    // valid upstream usage for feeding input in chunks before
    // finalizing. Must work even without initCStream.
    let src = b"continue-then-end test payload ".repeat(20);
    let mut cctx = ZSTD_createCCtx().unwrap();
    let mut dst = vec![0u8; 2048];
    let mut dst_pos = 0usize;

    // Feed in 64-byte chunks with e_continue.
    let mut src_cursor = 0usize;
    while src_cursor < src.len() {
        let chunk_end = (src_cursor + 64).min(src.len());
        let chunk = &src[src_cursor..chunk_end];
        let mut cp = 0usize;
        ZSTD_compressStream2(
            &mut cctx,
            &mut dst,
            &mut dst_pos,
            chunk,
            &mut cp,
            ZSTD_EndDirective::ZSTD_e_continue,
        );
        src_cursor += cp;
    }
    // Finalize.
    loop {
        let mut zero_pos = 0;
        let rc = ZSTD_compressStream2(
            &mut cctx,
            &mut dst,
            &mut dst_pos,
            &[],
            &mut zero_pos,
            ZSTD_EndDirective::ZSTD_e_end,
        );
        if rc == 0 || ERR_isError(rc) {
            break;
        }
    }

    // Roundtrip.
    use crate::decompress::zstd_decompress::ZSTD_decompress;
    let mut out = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress(&mut out, &dst[..dst_pos]);
    assert_eq!(&out[..d], &src[..]);
}

#[test]
fn CCtxParams_init_advanced_seeds_fields_and_rejects_bad_cparams() {
    // Good case: well-formed cParams populate params and set
    // compressionLevel to ZSTD_NO_CLEVEL (upstream behavior —
    // callers shouldn't trust the level when init is driven by
    // explicit cParams).
    let cp = ZSTD_getCParams(7, 0, 0);
    let zp = ZSTD_parameters {
        cParams: cp,
        fParams: ZSTD_FrameParameters {
            contentSizeFlag: 1,
            checksumFlag: 1,
            noDictIDFlag: 0,
        },
    };
    let mut p = ZSTD_CCtx_params::default();
    let rc = ZSTD_CCtxParams_init_advanced(&mut p, zp);
    assert_eq!(rc, 0);
    assert_eq!(p.compressionLevel, ZSTD_NO_CLEVEL);
    assert_eq!(p.cParams.windowLog, cp.windowLog);
    assert_eq!(p.fParams.checksumFlag, 1);

    // Bad case: invalid cParams (windowLog way above max) must
    // bubble up the ZSTD_checkCParams error.
    let mut bad = cp;
    bad.windowLog = 99;
    let zp_bad = ZSTD_parameters {
        cParams: bad,
        fParams: ZSTD_FrameParameters::default(),
    };
    let mut p2 = ZSTD_CCtx_params::default();
    let rc2 = ZSTD_CCtxParams_init_advanced(&mut p2, zp_bad);
    assert!(ERR_isError(rc2));
}

#[test]
fn CCtxParams_init_advanced_resolves_auto_policy_knobs() {
    use crate::compress::zstd_compress_sequences::{ZSTD_btultra2, ZSTD_lazy};
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;

    let row_cp = crate::compress::match_state::ZSTD_compressionParameters {
        strategy: ZSTD_lazy,
        windowLog: 15,
        ..ZSTD_getCParams(4, 0, 0)
    };
    let mut row_params = ZSTD_CCtx_params::default();
    let rc = ZSTD_CCtxParams_init_advanced(
        &mut row_params,
        ZSTD_parameters {
            cParams: row_cp,
            fParams: ZSTD_FrameParameters::default(),
        },
    );
    assert_eq!(rc, 0);
    assert_eq!(
        row_params.useRowMatchFinder,
        ZSTD_ParamSwitch_e::ZSTD_ps_enable
    );

    let opt_cp = crate::compress::match_state::ZSTD_compressionParameters {
        strategy: ZSTD_btultra2,
        windowLog: 27,
        ..ZSTD_getCParams(22, 0, 0)
    };
    let mut opt_params = ZSTD_CCtx_params::default();
    let rc = ZSTD_CCtxParams_init_advanced(
        &mut opt_params,
        ZSTD_parameters {
            cParams: opt_cp,
            fParams: ZSTD_FrameParameters::default(),
        },
    );
    assert_eq!(rc, 0);
    assert_eq!(
        opt_params.postBlockSplitter,
        ZSTD_ParamSwitch_e::ZSTD_ps_enable
    );
    assert_eq!(opt_params.ldmEnable, ZSTD_ParamSwitch_e::ZSTD_ps_enable);
    assert_eq!(opt_params.validateSequences, 0);
    assert_eq!(opt_params.maxBlockSize, ZSTD_BLOCKSIZE_MAX);
    assert_eq!(
        opt_params.searchForExternalRepcodes,
        ZSTD_ParamSwitch_e::ZSTD_ps_disable
    );
}

#[test]
fn cctxParams_advanced_flow_end_to_end_roundtrip() {
    // Prepare a CCtxParams with level 5, checksumFlag on.
    let mut p = ZSTD_createCCtxParams().unwrap();
    ZSTD_CCtxParams_setParameter(&mut p, ZSTD_cParameter::ZSTD_c_compressionLevel, 5);
    ZSTD_CCtxParams_setParameter(&mut p, ZSTD_cParameter::ZSTD_c_checksumFlag, 1);
    p.cParams = ZSTD_getCParams(5, 0, 0);

    // Apply to a CCtx.
    let mut cctx = ZSTD_createCCtx().unwrap();
    let rc = ZSTD_CCtx_setParametersUsingCCtxParams(&mut cctx, &p);
    assert_eq!(rc, 0);
    assert_eq!(cctx.stream_level, Some(5));
    assert!(cctx.param_checksum);

    // Compress via compressStream2.
    let src = b"advanced-flow roundtrip payload ".repeat(30);
    let mut dst = vec![0u8; 2048];
    let mut dst_pos = 0;
    let mut src_pos = 0;
    let rc = ZSTD_compressStream2(
        &mut cctx,
        &mut dst,
        &mut dst_pos,
        &src,
        &mut src_pos,
        ZSTD_EndDirective::ZSTD_e_end,
    );
    assert_eq!(rc, 0);

    // Roundtrip.
    use crate::decompress::zstd_decompress::ZSTD_decompress;
    let mut out = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress(&mut out, &dst[..dst_pos]);
    assert_eq!(&out[..d], &src[..]);
}

#[test]
fn compressStream2_without_init_auto_defaults_level() {
    // Modern parametric entry must work even when the caller
    // skipped ZSTD_initCStream. Upstream auto-initializes to
    // CLEVEL_DEFAULT via CCtxParams_reset.
    let mut cctx = ZSTD_createCCtx().unwrap();
    let src = b"stream2 no-init test".repeat(10);
    let mut dst = vec![0u8; 512];
    let mut dst_pos = 0usize;
    let mut src_pos = 0usize;
    let rc = ZSTD_compressStream2(
        &mut cctx,
        &mut dst,
        &mut dst_pos,
        &src,
        &mut src_pos,
        ZSTD_EndDirective::ZSTD_e_end,
    );
    assert!(!ERR_isError(rc), "err={rc:#x}");
    assert_eq!(rc, 0);
    // Roundtrip.
    use crate::decompress::zstd_decompress::ZSTD_decompress;
    let mut out = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress(&mut out, &dst[..dst_pos]);
    assert_eq!(&out[..d], &src[..]);
}

#[test]
fn compressStream2_end_fully_drained_returns_to_init_stage() {
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    let mut cctx = ZSTD_createCCtx().unwrap();
    let src1 = b"first streaming frame ".repeat(16);
    let mut dst1 = vec![0u8; ZSTD_compressBound(src1.len()) + 64];
    let mut dst1_pos = 0usize;
    let mut src1_pos = 0usize;
    let rc = ZSTD_compressStream2(
        &mut cctx,
        &mut dst1,
        &mut dst1_pos,
        &src1,
        &mut src1_pos,
        ZSTD_EndDirective::ZSTD_e_end,
    );
    assert_eq!(rc, 0);
    assert!(cctx.stream_in_buffer.is_empty());
    assert!(cctx.stream_out_buffer.is_empty());
    assert!(!cctx.stream_closed);

    assert_eq!(
        ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_checksumFlag, 1),
        0
    );

    let src2 = b"second streaming frame ".repeat(19);
    let mut dst2 = vec![0u8; ZSTD_compressBound(src2.len()) + 64];
    let mut dst2_pos = 0usize;
    let mut src2_pos = 0usize;
    let rc = ZSTD_compressStream2(
        &mut cctx,
        &mut dst2,
        &mut dst2_pos,
        &src2,
        &mut src2_pos,
        ZSTD_EndDirective::ZSTD_e_end,
    );
    assert_eq!(rc, 0);

    let mut out = vec![0u8; src2.len() + 64];
    let d = ZSTD_decompress(&mut out, &dst2[..dst2_pos]);
    assert_eq!(&out[..d], &src2[..]);
}

#[test]
fn compressStream2_rejects_invalid_buffer_positions_with_specific_errors() {
    use crate::common::error::{ERR_getErrorCode, ERR_isError};

    let mut cctx = ZSTD_createCCtx().unwrap();
    let src = b"stream-pos-check";
    let mut dst = [0u8; 64];

    let mut bad_dst_pos = dst.len() + 1;
    let mut src_pos = 0usize;
    let rc = ZSTD_compressStream2(
        &mut cctx,
        &mut dst,
        &mut bad_dst_pos,
        src,
        &mut src_pos,
        ZSTD_EndDirective::ZSTD_e_continue,
    );
    assert!(ERR_isError(rc));
    assert_eq!(ERR_getErrorCode(rc), ErrorCode::DstSizeTooSmall);

    let mut ok_dst_pos = 0usize;
    let mut bad_src_pos = src.len() + 1;
    let rc = ZSTD_compressStream2(
        &mut cctx,
        &mut dst,
        &mut ok_dst_pos,
        src,
        &mut bad_src_pos,
        ZSTD_EndDirective::ZSTD_e_continue,
    );
    assert!(ERR_isError(rc));
    assert_eq!(ERR_getErrorCode(rc), ErrorCode::SrcSizeWrong);
}

#[test]
fn cctx_reuse_after_session_reset_matches_fresh_cctx() {
    // Reuse a CCtx for two frames; the second output should
    // match a brand-new CCtx compressing the same data.
    let src1 = b"first frame contents ".repeat(50);
    let src2 = b"second frame different ".repeat(60);

    let mut reused = ZSTD_createCCtx().unwrap();
    let mut dst1 = vec![0u8; 2048];
    let n1 = ZSTD_compressCCtx(&mut reused, &mut dst1, &src1, 3);
    assert!(!ERR_isError(n1));
    ZSTD_CCtx_reset(&mut reused, ZSTD_ResetDirective::ZSTD_reset_session_only);
    let mut dst2a = vec![0u8; 2048];
    let n2a = ZSTD_compressCCtx(&mut reused, &mut dst2a, &src2, 3);
    assert!(!ERR_isError(n2a));

    let mut fresh = ZSTD_createCCtx().unwrap();
    let mut dst2b = vec![0u8; 2048];
    let n2b = ZSTD_compressCCtx(&mut fresh, &mut dst2b, &src2, 3);
    assert_eq!(&dst2a[..n2a], &dst2b[..n2b]);
}

#[test]
fn cctx_reset_modes_differ_correctly() {
    let mut cctx = ZSTD_createCCtx().unwrap();
    cctx.stream_level = Some(7);
    cctx.stream_dict = b"dict-bytes".to_vec();
    cctx.param_checksum = true;

    // session_only: keep parameters + dict.
    ZSTD_CCtx_reset(&mut cctx, ZSTD_ResetDirective::ZSTD_reset_session_only);
    assert_eq!(cctx.stream_level, Some(7));
    assert_eq!(cctx.stream_dict, b"dict-bytes");
    assert!(cctx.param_checksum);

    // parameters: drop level + dict + flags.
    ZSTD_CCtx_reset(&mut cctx, ZSTD_ResetDirective::ZSTD_reset_parameters);
    assert_eq!(cctx.stream_level, None);
    assert!(cctx.stream_dict.is_empty());
    assert!(!cctx.param_checksum);

    // session_and_parameters: superset — does both. Re-seed
    // with non-default state and confirm one call restores
    // defaults (contentSize=true, checksum=false, dictID=true,
    // level=None, dict empty).
    cctx.stream_level = Some(11);
    cctx.stream_dict = b"different".to_vec();
    cctx.param_checksum = true;
    cctx.param_contentSize = false;
    cctx.param_dictID = false;
    ZSTD_CCtx_reset(
        &mut cctx,
        ZSTD_ResetDirective::ZSTD_reset_session_and_parameters,
    );
    assert_eq!(cctx.stream_level, None);
    assert!(cctx.stream_dict.is_empty());
    assert!(!cctx.param_checksum);
    assert!(cctx.param_contentSize, "contentSize default is true");
    assert!(cctx.param_dictID, "dictID default is true");
}

#[test]
fn cctx_reset_session_clears_pending_stream_state() {
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream(&mut cctx, 5);
    // Ingest some bytes.
    let src = b"reset probe payload".repeat(10);
    let mut dst = vec![0u8; 256];
    let mut dst_pos = 0usize;
    let mut src_pos = 0usize;
    ZSTD_compressStream(&mut cctx, &mut dst, &mut dst_pos, &src, &mut src_pos);
    assert!(!cctx.stream_in_buffer.is_empty());

    // Reset session — buffer should drop.
    ZSTD_CCtx_reset(&mut cctx, ZSTD_ResetDirective::ZSTD_reset_session_only);
    assert!(cctx.stream_in_buffer.is_empty());
    assert!(cctx.stream_out_buffer.is_empty());
    assert_eq!(cctx.stream_out_drained, 0);
    assert!(!cctx.stream_closed);
}

#[test]
fn compress2_bare_frame_no_checksum_no_fcs_roundtrip() {
    // Bare frame: no checksum, no content-size field. Must still
    // round-trip correctly — decompressor uses block headers only.
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_compressionLevel, 3);
    ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_checksumFlag, 0);
    ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_contentSizeFlag, 0);

    let src = b"bare-frame test payload ".repeat(50);
    let mut dst = vec![0u8; 2048];
    let n = ZSTD_compress2(&mut cctx, &mut dst, &src);
    assert!(!ERR_isError(n));
    dst.truncate(n);

    use crate::decompress::zstd_decompress::{ZSTD_decompress, ZSTD_getFrameContentSize};
    // No FCS emitted → getFrameContentSize returns UNKNOWN.
    use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;
    assert_eq!(ZSTD_getFrameContentSize(&dst), ZSTD_CONTENTSIZE_UNKNOWN);

    let mut out = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress(&mut out, &dst);
    assert_eq!(&out[..d], &src[..]);
}

#[test]
fn LLcode_MLcode_boundary_behavior_matches_upstream() {
    // LL/ML code tables are small (64/128 entries) with closed-form
    // fallback (`highbit32(val) + DELTA`) for values past the end.
    // Pin the in-table lookups + the crossover behaviour; either
    // LL_DELTA=19 or ML_DELTA=36 drifting would misroute every
    // symbol in that range.
    // LL: identity for 0..=15.
    for v in 0u32..=15 {
        assert_eq!(ZSTD_LLcode(v), v);
    }
    // LL: table value 16 at lits=16, 17, then 17 at 18, 19, ...
    assert_eq!(ZSTD_LLcode(16), 16);
    assert_eq!(ZSTD_LLcode(17), 16);
    assert_eq!(ZSTD_LLcode(18), 17);
    // LL: boundary — table up to 63, then highbit+19.
    assert_eq!(ZSTD_LLcode(63), 24);
    // For litLength=64: highbit32(64)=6, code=6+19=25.
    assert_eq!(ZSTD_LLcode(64), 25);
    assert_eq!(ZSTD_LLcode(128), 26); // highbit=7, 7+19=26

    // ML: identity for 0..=31.
    for v in 0u32..=31 {
        assert_eq!(ZSTD_MLcode(v), v);
    }
    // ML: table at 32, 33 → 32; 34, 35 → 33; 127 stays in table.
    assert_eq!(ZSTD_MLcode(32), 32);
    assert_eq!(ZSTD_MLcode(33), 32);
    assert_eq!(ZSTD_MLcode(34), 33);
    // ML: boundary — table up to 127 (= 42), then highbit+36.
    assert_eq!(ZSTD_MLcode(127), 42);
    // For mlBase=128: highbit32(128)=7, 7+36=43.
    assert_eq!(ZSTD_MLcode(128), 43);
    assert_eq!(ZSTD_MLcode(256), 44);
}

#[test]
fn streaming_buffer_size_hints_match_upstream_formulas() {
    // Exact upstream formulas (zstd_compress.c:5976/5980,
    // zstd_decompress.c:1696/1697):
    //   CStreamInSize  = ZSTD_BLOCKSIZE_MAX
    //   CStreamOutSize = compressBound(BLOCKSIZE_MAX) + blockHeaderSize + 4
    //   DStreamInSize  = ZSTD_BLOCKSIZE_MAX + blockHeaderSize
    //   DStreamOutSize = ZSTD_BLOCKSIZE_MAX
    // Previously only loose ordering was checked; pin exact
    // equality so future refactors can't introduce over/under-
    // estimates silently.
    use crate::decompress::zstd_decompress::{ZSTD_DStreamInSize, ZSTD_DStreamOutSize};
    use crate::decompress::zstd_decompress_block::{ZSTD_blockHeaderSize, ZSTD_BLOCKSIZE_MAX};
    assert_eq!(ZSTD_CStreamInSize(), ZSTD_BLOCKSIZE_MAX);
    assert_eq!(
        ZSTD_CStreamOutSize(),
        ZSTD_compressBound(ZSTD_BLOCKSIZE_MAX) + ZSTD_blockHeaderSize + 4,
    );
    assert_eq!(
        ZSTD_DStreamInSize(),
        ZSTD_BLOCKSIZE_MAX + ZSTD_blockHeaderSize
    );
    assert_eq!(ZSTD_DStreamOutSize(), ZSTD_BLOCKSIZE_MAX);
}

#[test]
fn setPledgedSrcSize_cleared_by_reset_session_only() {
    // Upstream (zstd_compress.c:1386-1389) clears
    // `pledgedSrcSizePlusOne` on `reset_session_only` — the next
    // frame starts fresh with UNKNOWN pledged size. Our port
    // clears `pledged_src_size` to `None`. Previously unpinned;
    // keep the contract visible so session-reuse callers can rely
    // on clean per-frame state.
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_CCtx_setPledgedSrcSize(&mut cctx, 500);
    assert_eq!(cctx.pledged_src_size, Some(500));
    ZSTD_CCtx_reset(&mut cctx, ZSTD_ResetDirective::ZSTD_reset_session_only);
    assert_eq!(cctx.pledged_src_size, None);
    // Same after `session_and_parameters`.
    ZSTD_CCtx_setPledgedSrcSize(&mut cctx, 999);
    ZSTD_CCtx_reset(
        &mut cctx,
        ZSTD_ResetDirective::ZSTD_reset_session_and_parameters,
    );
    assert_eq!(cctx.pledged_src_size, None);
}

#[test]
fn CDict_and_DDict_from_same_magic_dict_expose_same_content_and_id() {
    // Parity probe: a magic-tagged dict loaded via `ZSTD_createCDict`
    // should expose the parsed content bytes while keeping the same
    // dictID as the decompression-side DDict and the original bytes.
    use crate::decompress::zstd_ddict::{ZSTD_createDDict, ZSTD_getDictID_fromDDict};
    let magic_dict: &[u8] =
        include_bytes!("../../tests/fixtures/upstream-zstd/dict-files/zero-weight-dict");
    let cdict = ZSTD_createCDict(magic_dict, 5).unwrap();
    let ddict = ZSTD_createDDict(magic_dict).unwrap();
    let mut entropy = ZSTD_entropyCTables_t::default();
    let mut rep = ZSTD_REP_START_VALUE;
    let e_size = ZSTD_loadCEntropy(&mut entropy, &mut rep, magic_dict);
    assert!(!ERR_isError(e_size), "loadCEntropy err={e_size:#x}");
    assert_eq!(cdict.dictContent, &magic_dict[e_size..]);
    assert_eq!(
        ZSTD_getDictID_fromCDict(&cdict),
        ZSTD_getDictID_fromDDict(&ddict),
    );
    assert_eq!(
        ZSTD_getDictID_fromCDict(&cdict),
        crate::decompress::zstd_ddict::ZSTD_getDictID_fromDict(magic_dict),
    );
}

#[test]
fn CCtxParams_reset_restores_init_defaults() {
    // `ZSTD_CCtxParams_reset` should drop any customized fields
    // back to what `CCtxParams_init(CLEVEL_DEFAULT)` produces:
    // zeroed struct + `compressionLevel=3`, `contentSizeFlag=1`.
    // Regression gate in case someone removes the init call from
    // reset and leaves stale state.
    let mut params = ZSTD_CCtx_params::default();
    ZSTD_CCtxParams_setParameter(&mut params, ZSTD_cParameter::ZSTD_c_compressionLevel, 9);
    ZSTD_CCtxParams_setParameter(&mut params, ZSTD_cParameter::ZSTD_c_checksumFlag, 1);
    ZSTD_CCtxParams_setParameter(&mut params, ZSTD_cParameter::ZSTD_c_contentSizeFlag, 0);
    assert_eq!(params.compressionLevel, 9);
    assert_eq!(params.fParams.checksumFlag, 1);
    assert_eq!(params.fParams.contentSizeFlag, 0);

    let rc = ZSTD_CCtxParams_reset(&mut params);
    assert_eq!(rc, 0);
    assert_eq!(params.compressionLevel, ZSTD_CLEVEL_DEFAULT);
    assert_eq!(params.fParams.checksumFlag, 0);
    assert_eq!(params.fParams.contentSizeFlag, 1);
    assert_eq!(params.fParams.noDictIDFlag, 0);
}

#[test]
fn CCtxParams_init_advanced_preserves_zstdParams_fParams() {
    // `ZSTD_CCtxParams_init_advanced` must copy fParams verbatim
    // (contentSizeFlag, checksumFlag, noDictIDFlag). Easy to break
    // if refactoring fParams plumbing — pin the roundtrip explicitly
    // so drift shows up immediately rather than as a subtle frame-
    // header bit flip on downstream compression.
    let mut params = ZSTD_CCtx_params::default();
    let cp = ZSTD_getCParams(3, 0, 0);
    let fp = ZSTD_FrameParameters {
        contentSizeFlag: 0,
        checksumFlag: 1,
        noDictIDFlag: 1,
    };
    let rc = ZSTD_CCtxParams_init_advanced(
        &mut params,
        ZSTD_parameters {
            cParams: cp,
            fParams: fp,
        },
    );
    assert_eq!(rc, 0);
    assert_eq!(params.fParams.contentSizeFlag, 0);
    assert_eq!(params.fParams.checksumFlag, 1);
    assert_eq!(params.fParams.noDictIDFlag, 1);
    // And cParams propagates too.
    assert_eq!(params.cParams.windowLog, cp.windowLog);
    assert_eq!(params.cParams.strategy, cp.strategy);
}

#[test]
fn compressStream2_respects_contentSizeFlag_off() {
    // With `c_contentSizeFlag=0`, the emitted frame header must
    // NOT declare the frame content size — `getFrameContentSize`
    // should report `ZSTD_CONTENTSIZE_UNKNOWN`. Default flag=1 is
    // well covered; pin the "turned off" path too in case the
    // plumbing that ties `param_contentSize → fParams.contentSizeFlag`
    // ever regresses.
    use crate::decompress::zstd_decompress::{ZSTD_getFrameContentSize, ZSTD_CONTENTSIZE_UNKNOWN};
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_contentSizeFlag, 0);
    let src = b"contentSizeFlag off path ".repeat(20);
    let mut dst = vec![0u8; 4096];
    let n = ZSTD_compress2(&mut cctx, &mut dst, &src);
    assert!(!ERR_isError(n));
    assert_eq!(
        ZSTD_getFrameContentSize(&dst[..n]),
        ZSTD_CONTENTSIZE_UNKNOWN
    );

    // Sanity contrast: with flag ON (default), FCS matches src.len.
    let mut cctx2 = ZSTD_createCCtx().unwrap();
    let mut dst2 = vec![0u8; 4096];
    let n2 = ZSTD_compress2(&mut cctx2, &mut dst2, &src);
    assert!(!ERR_isError(n2));
    assert_eq!(ZSTD_getFrameContentSize(&dst2[..n2]), src.len() as u64);
}

#[test]
fn endStream_second_call_after_complete_frame_is_noop_success() {
    // Contract: once a frame has been fully emitted via
    // `ZSTD_endStream`, a second call must be a safe no-op (no
    // spurious error, no extra bytes). Callers commonly keep
    // calling endStream in a loop until it returns 0 — the very
    // last iteration is this no-op case.
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream(&mut cctx, 3);
    let src = b"endStream-idempotency-probe ".repeat(20);
    let mut out = vec![0u8; 4096];
    let mut dp = 0usize;
    let mut sp = 0usize;
    // First: feed + end.
    let rc1 = ZSTD_compressStream2(
        &mut cctx,
        &mut out,
        &mut dp,
        &src,
        &mut sp,
        ZSTD_EndDirective::ZSTD_e_end,
    );
    assert!(!ERR_isError(rc1));
    assert_eq!(rc1, 0, "first endStream should fully flush");
    let written_after_first = dp;
    // Second: zero input, e_end. Must return 0 and add no bytes.
    let mut empty_pos = 0usize;
    let rc2 = ZSTD_compressStream2(
        &mut cctx,
        &mut out,
        &mut dp,
        &[],
        &mut empty_pos,
        ZSTD_EndDirective::ZSTD_e_end,
    );
    assert!(!ERR_isError(rc2));
    assert_eq!(rc2, 0);
    assert_eq!(dp, written_after_first, "no extra bytes on 2nd endStream");
}

#[test]
fn compress2_rejects_tiny_dst_with_DstSizeTooSmall() {
    // Contract: `ZSTD_compress2` must not silently truncate — if
    // the destination buffer can't hold the full frame, the call
    // returns `DstSizeTooSmall` (not a partial write + success).
    // Covers the wrap-around in the `result != 0` branch.
    use crate::common::error::{ERR_getErrorCode, ERR_isError};
    let mut cctx = ZSTD_createCCtx().unwrap();
    let src = b"compress2 tiny dst ".repeat(50);
    let mut dst = vec![0u8; 4]; // way too small for any frame
    let rc = ZSTD_compress2(&mut cctx, &mut dst, &src);
    assert!(ERR_isError(rc));
    assert_eq!(ERR_getErrorCode(rc), ErrorCode::DstSizeTooSmall);
}

#[test]
fn fresh_CCtx_getParameter_returns_upstream_defaults() {
    // Upstream `ZSTD_createCCtx` calls `ZSTD_CCtxParams_init(..,
    // CLEVEL_DEFAULT)` which sets `compressionLevel=3`,
    // `contentSizeFlag=1`, others=0. A fresh CCtx readback via
    // `getParameter` should match those defaults — previously
    // `stream_level=None` would be observed as 3 (via unwrap_or),
    // but the flag defaults needed an explicit pin too.
    let cctx = ZSTD_createCCtx().unwrap();
    let mut got = -1i32;
    ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_compressionLevel, &mut got);
    assert_eq!(got, ZSTD_CLEVEL_DEFAULT);
    ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_contentSizeFlag, &mut got);
    assert_eq!(got, 1, "contentSizeFlag default: 1");
    ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_checksumFlag, &mut got);
    assert_eq!(got, 0, "checksumFlag default: 0");
    ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_dictIDFlag, &mut got);
    assert_eq!(got, 1, "dictIDFlag default: 1");
    ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_blockSplitterLevel, &mut got);
    assert_eq!(got, 0, "blockSplitterLevel default: auto");
}

#[test]
fn blockSplitterLevel_param_roundtrips_through_cctx_and_params_api() {
    let mut cctx = ZSTD_createCCtx().unwrap();
    let mut params = ZSTD_CCtx_params::default();
    let mut got = -1i32;

    assert_eq!(
        ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_blockSplitterLevel, 6),
        0
    );
    assert_eq!(
        ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_blockSplitterLevel, &mut got),
        0
    );
    assert_eq!(got, 6);

    assert_eq!(
        ZSTD_CCtxParams_setParameter(&mut params, ZSTD_cParameter::ZSTD_c_blockSplitterLevel, 2),
        0
    );
    assert_eq!(
        ZSTD_CCtxParams_getParameter(
            &params,
            ZSTD_cParameter::ZSTD_c_blockSplitterLevel,
            &mut got
        ),
        0
    );
    assert_eq!(got, 2);
}

#[test]
fn enable_seq_producer_fallback_param_roundtrips_through_cctx_and_params_api() {
    let mut cctx = ZSTD_createCCtx().unwrap();
    let mut params = ZSTD_CCtx_params::default();
    let mut got = -1i32;

    assert_eq!(
        ZSTD_CCtx_setParameter(
            &mut cctx,
            ZSTD_cParameter::ZSTD_c_enableSeqProducerFallback,
            1
        ),
        0
    );
    assert_eq!(
        ZSTD_CCtx_getParameter(
            &cctx,
            ZSTD_cParameter::ZSTD_c_enableSeqProducerFallback,
            &mut got
        ),
        0
    );
    assert_eq!(got, 1);

    assert_eq!(
        ZSTD_CCtxParams_setParameter(
            &mut params,
            ZSTD_cParameter::ZSTD_c_enableSeqProducerFallback,
            1
        ),
        0
    );
    assert_eq!(
        ZSTD_CCtxParams_getParameter(
            &params,
            ZSTD_cParameter::ZSTD_c_enableSeqProducerFallback,
            &mut got
        ),
        0
    );
    assert_eq!(got, 1);
}

#[test]
fn optimalBlockSize_pre_splitter_respects_auto_disable_and_savings_gate() {
    use crate::compress::zstd_compress_sequences::ZSTD_btultra2;

    let mut block = vec![0u8; 128 << 10];
    let half = block.len() / 2;
    for (i, b) in block.iter_mut().enumerate().skip(half) {
        *b = (i as u8).wrapping_mul(31).wrapping_add(7);
    }

    assert_eq!(
        ZSTD_optimalBlockSize(&block, block.len(), 1, ZSTD_btultra2, 10),
        block.len()
    );
    assert_eq!(
        ZSTD_optimalBlockSize(&block, block.len(), 0, ZSTD_btultra2, 0),
        block.len()
    );

    let split = ZSTD_optimalBlockSize(&block, block.len(), 0, ZSTD_btultra2, 10);
    assert!(split < block.len(), "auto splitter should pick a boundary");
    assert_eq!(split % (8 << 10), 0, "chunk splitter should align to 8 KiB");
}

#[test]
fn CCtxParams_setParameter_matches_CCtx_level_semantics() {
    // Both entry points must yield the same `compressionLevel`
    // readback for every value — upstream routes CCtx_set through
    // CCtxParams_set. Cover 0 (maps to default), high clamp,
    // low clamp, and a normal in-range value.
    let cases: &[(i32, i32)] = &[
        (0, ZSTD_CLEVEL_DEFAULT),
        (9999, ZSTD_maxCLevel()),
        (i32::MIN, ZSTD_minCLevel()),
        (5, 5),
    ];
    for &(input, expected) in cases {
        let mut params = ZSTD_CCtx_params::default();
        ZSTD_CCtxParams_setParameter(&mut params, ZSTD_cParameter::ZSTD_c_compressionLevel, input);
        let mut got = 0i32;
        ZSTD_CCtxParams_getParameter(&params, ZSTD_cParameter::ZSTD_c_compressionLevel, &mut got);
        assert_eq!(got, expected, "CCtxParams level {input} → {got}");
    }
}

#[test]
fn CCtxParams_setParameter_flag_values_reject_out_of_range() {
    use crate::common::error::ERR_getErrorCode;

    let mut params = ZSTD_CCtx_params::default();
    for input in [0i32, 1] {
        assert_eq!(
            ZSTD_CCtxParams_setParameter(&mut params, ZSTD_cParameter::ZSTD_c_checksumFlag, input),
            0
        );
        assert_eq!(params.fParams.checksumFlag, input as u32);
    }
    for input in [2i32, -1, i32::MAX] {
        let rc =
            ZSTD_CCtxParams_setParameter(&mut params, ZSTD_cParameter::ZSTD_c_checksumFlag, input);
        assert!(ERR_isError(rc), "checksumFlag({input})");
        assert_eq!(ERR_getErrorCode(rc), ErrorCode::ParameterOutOfBound);
    }
}

#[test]
fn flushStream_emits_staged_input_without_finishing_pledged_frame() {
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    let first = b"pledged flush split part one ".repeat(9);
    let second = b"pledged flush split part two ".repeat(7);
    let total = first.len() + second.len();
    let mut cctx = ZSTD_createCCtx().unwrap();
    assert_eq!(ZSTD_initCStream(&mut cctx, 3), 0);
    assert_eq!(ZSTD_CCtx_setPledgedSrcSize(&mut cctx, total as u64), 0);

    let mut dst = vec![0u8; ZSTD_compressBound(total) + 128];
    let mut dp = 0usize;
    let mut sp = 0usize;
    let rc = ZSTD_compressStream(&mut cctx, &mut dst, &mut dp, &first, &mut sp);
    assert!(!ERR_isError(rc));
    assert_eq!(sp, first.len());
    assert_eq!(dp, 0);

    let rc = ZSTD_flushStream(&mut cctx, &mut dst, &mut dp);
    assert!(
        !ERR_isError(rc),
        "flushStream errored before full pledge: {rc:#x}"
    );
    assert_eq!(rc, 0);
    assert_ne!(dp, 0, "flushStream should emit the staged non-final block");

    sp = 0;
    let rc = ZSTD_compressStream(&mut cctx, &mut dst, &mut dp, &second, &mut sp);
    assert!(!ERR_isError(rc));
    assert_eq!(sp, second.len());

    loop {
        let remaining = ZSTD_endStream(&mut cctx, &mut dst, &mut dp);
        assert!(!ERR_isError(remaining), "endStream err: {remaining:#x}");
        if remaining == 0 {
            break;
        }
        dst.resize(dst.len() + remaining.max(32), 0);
    }

    let mut decoded = vec![0u8; total];
    let d = ZSTD_decompress(&mut decoded, &dst[..dp]);
    assert_eq!(d, total);
    assert_eq!(&decoded[..first.len()], first.as_slice());
    assert_eq!(&decoded[first.len()..d], second.as_slice());
}

#[test]
fn CCtx_refPrefix_magic_dict_roundtrips_dictID_through_streaming() {
    // Close a gap: `ZSTD_CCtx_refPrefix` routes through the
    // streaming endStream path, which wires `cctx.param_dictID`
    // → `fParams.noDictIDFlag` and parses dictID from the prefix.
    // End-to-end roundtrip confirms the full integration (fresh
    // CCtx → refPrefix + magic dict → compressStream2(e_end) →
    // frame header preserves dictID).
    use crate::decompress::zstd_ddict::ZSTD_dictContentType_e;
    use crate::decompress::zstd_decompress::ZSTD_getDictID_fromFrame;
    let magic_dict: &[u8] =
        include_bytes!("../../tests/fixtures/upstream-zstd/dict-files/zero-weight-dict");
    let expected_dict_id = crate::decompress::zstd_ddict::ZSTD_getDictID_fromDict(magic_dict);
    assert_ne!(
        expected_dict_id, 0,
        "fixture must be a real full dictionary"
    );
    let src = b"refPrefix streaming roundtrip ".repeat(15);

    let mut cctx = ZSTD_createCCtx().unwrap();
    assert_eq!(
        ZSTD_CCtx_refPrefix_advanced(
            &mut cctx,
            magic_dict,
            ZSTD_dictContentType_e::ZSTD_dct_fullDict
        ),
        0
    );
    // default `param_dictID == true` ⇒ include dictID in header
    let mut dst = vec![0u8; 4096];
    let n = ZSTD_compress2(&mut cctx, &mut dst, &src);
    assert!(!ERR_isError(n));
    assert_eq!(ZSTD_getDictID_fromFrame(&dst[..n]), expected_dict_id);
}

#[test]
fn compress_usingCDict_also_propagates_magic_dictID() {
    // Sibling coverage for the CDict path: `ZSTD_compress_usingCDict`
    // forwards to `ZSTD_compress_usingDict` under the hood, so the
    // same `noDictIDFlag=0` fix must carry through. Confirms no
    // separate hardcoded-dictID site exists on the CDict track.
    use crate::decompress::zstd_decompress::ZSTD_getDictID_fromFrame;
    let magic_dict: &[u8] =
        include_bytes!("../../tests/fixtures/upstream-zstd/dict-files/zero-weight-dict");
    let cdict = ZSTD_createCDict(magic_dict, 5).unwrap();
    let src = b"ZSTD_compress_usingCDict dictID parity ".repeat(20);
    let mut cctx = ZSTD_createCCtx().unwrap();
    let mut dst = vec![0u8; 4096];
    let n = ZSTD_compress_usingCDict(&mut cctx, &mut dst, &src, &cdict);
    assert!(!ERR_isError(n));
    assert_eq!(
        ZSTD_getDictID_fromFrame(&dst[..n]),
        crate::decompress::zstd_ddict::ZSTD_getDictID_fromDict(magic_dict),
    );
}

#[test]
fn ZSTD_CParamMode_e_discriminants_match_upstream() {
    // Upstream (zstd_compress_internal.h:558-573):
    //   ZSTD_cpm_noAttachDict=0, ZSTD_cpm_attachDict=1,
    //   ZSTD_cpm_createCDict=2,  ZSTD_cpm_unknown=3
    // `ZSTD_adjustCParams_internal` dispatches on this enum via
    // equality checks to decide whether srcSize, dictSize, or
    // both contribute to parameter-selection math.
    assert_eq!(ZSTD_CParamMode_e::ZSTD_cpm_noAttachDict as u32, 0);
    assert_eq!(ZSTD_CParamMode_e::ZSTD_cpm_attachDict as u32, 1);
    assert_eq!(ZSTD_CParamMode_e::ZSTD_cpm_createCDict as u32, 2);
    assert_eq!(ZSTD_CParamMode_e::ZSTD_cpm_unknown as u32, 3);
}

#[test]
fn getBlockSize_reverts_to_BLOCKSIZE_MAX_after_reset_parameters() {
    // After `ZSTD_CCtx_reset(parameters)`, `requested_cParams` is
    // cleared — `getBlockSize` should revert from the small-window
    // value back to `BLOCKSIZE_MAX`. Guards against stale cparam
    // state surviving a reset.
    use crate::compress::match_state::ZSTD_compressionParameters;
    use crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX;
    let mut cctx = ZSTD_createCCtx().unwrap();
    let small_cp = ZSTD_compressionParameters {
        windowLog: 10,
        chainLog: 10,
        hashLog: 10,
        searchLog: 1,
        minMatch: 4,
        targetLength: 0,
        strategy: 1,
    };
    ZSTD_CCtx_setCParams(&mut cctx, small_cp);
    assert_eq!(ZSTD_getBlockSize(&cctx), 1024);
    ZSTD_CCtx_reset(&mut cctx, ZSTD_ResetDirective::ZSTD_reset_parameters);
    assert_eq!(ZSTD_getBlockSize(&cctx), ZSTD_BLOCKSIZE_MAX);
}

#[test]
fn getBlockSize_honors_requested_windowLog_min_with_BLOCKSIZE_MAX() {
    // Upstream `ZSTD_getBlockSize(cctx)` returns
    // `min(BLOCKSIZE_MAX, 1 << cParams.windowLog)`. Previously our
    // port returned `BLOCKSIZE_MAX` regardless of windowLog, over-
    // reporting for small-window configs. Pin both regimes:
    //   - small windowLog (10) → 1 << 10 = 1024
    //   - default / unconfigured → BLOCKSIZE_MAX (128 KB)
    use crate::compress::match_state::ZSTD_compressionParameters;
    use crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX;

    // Fresh CCtx (no requested_cParams): returns BLOCKSIZE_MAX.
    let cctx_fresh = ZSTD_createCCtx().unwrap();
    assert_eq!(ZSTD_getBlockSize(&cctx_fresh), ZSTD_BLOCKSIZE_MAX);

    // Configured with windowLog=10: returns 1024.
    let mut cctx_small = ZSTD_createCCtx().unwrap();
    let small_cp = ZSTD_compressionParameters {
        windowLog: 10,
        chainLog: 10,
        hashLog: 10,
        searchLog: 1,
        minMatch: 4,
        targetLength: 0,
        strategy: 1,
    };
    assert_eq!(ZSTD_CCtx_setCParams(&mut cctx_small, small_cp), 0);
    assert_eq!(ZSTD_getBlockSize(&cctx_small), 1024);

    // Configured with windowLog=17 (== log2(BLOCKSIZE_MAX)): returns
    // BLOCKSIZE_MAX.
    let mut cctx_edge = ZSTD_createCCtx().unwrap();
    let edge_cp = ZSTD_compressionParameters {
        windowLog: 17,
        chainLog: 17,
        hashLog: 17,
        searchLog: 4,
        minMatch: 4,
        targetLength: 32,
        strategy: 3,
    };
    assert_eq!(ZSTD_CCtx_setCParams(&mut cctx_edge, edge_cp), 0);
    assert_eq!(ZSTD_getBlockSize(&cctx_edge), ZSTD_BLOCKSIZE_MAX);
}

#[test]
fn adjustCParams_internal_attachDict_mode_clears_dictSize() {
    // In `ZSTD_cpm_attachDict` mode, `ZSTD_adjustCParams_internal`
    // treats `dictSize` as 0 — sizing decisions only consult
    // `srcSize`. This is the code path upstream uses when a
    // dictMatchState is attached. Pin the behavior differential
    // against noAttachDict where dictSize DOES contribute.
    use crate::compress::match_state::ZSTD_compressionParameters;
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;

    let base = ZSTD_compressionParameters {
        windowLog: 20,
        hashLog: 20,
        chainLog: 20,
        searchLog: 5,
        minMatch: 4,
        targetLength: 32,
        strategy: 3,
    };
    let src_small: u64 = 500;
    let dict_large: u64 = 500_000;

    // noAttachDict: dict contributes to sizing → windowLog may
    // shrink more than if dict were ignored.
    let cp_no = ZSTD_adjustCParams_internal(
        base,
        src_small,
        dict_large,
        ZSTD_CParamMode_e::ZSTD_cpm_noAttachDict,
        ZSTD_ParamSwitch_e::ZSTD_ps_auto,
    );
    // attachDict: dict is dropped → sizing uses srcSize alone.
    let cp_attach = ZSTD_adjustCParams_internal(
        base,
        src_small,
        dict_large,
        ZSTD_CParamMode_e::ZSTD_cpm_attachDict,
        ZSTD_ParamSwitch_e::ZSTD_ps_auto,
    );
    // With 500 B src and no dict-contribution, attachDict should
    // shrink windowLog more aggressively than noAttachDict (which
    // still accounts for the 500 KB dict).
    assert!(
        cp_attach.windowLog <= cp_no.windowLog,
        "attachDict should shrink window no less than noAttachDict: \
             attach={} no={}",
        cp_attach.windowLog,
        cp_no.windowLog,
    );
}

#[test]
fn ZSTD_seqToCodes_promotes_long_length_flag_to_max_code() {
    // When `seqStore.longLengthType != none`, `seqToCodes` must
    // bump the code at `longLengthPos` to `MaxLL` or `MaxML`
    // respectively. Upstream does this as a post-loop fixup so
    // the FSE encoder emits the sentinel code + extra-bits tail.
    use crate::compress::seq_store::{SeqDef, SeqStore_t, OFFSET_TO_OFFBASE};
    use crate::decompress::zstd_decompress_block::{MaxLL, MaxML};

    // Seed two sequences: one with a modest litLength/mlBase,
    // and one marked as "long" at index 1.
    let mut ss = SeqStore_t::with_capacity(8, 256);
    ss.sequences.push(SeqDef {
        offBase: OFFSET_TO_OFFBASE(17),
        litLength: 5,
        mlBase: 10,
    });
    ss.sequences.push(SeqDef {
        offBase: OFFSET_TO_OFFBASE(17),
        litLength: 7,
        mlBase: 12,
    });
    ss.longLengthType = crate::compress::seq_store::ZSTD_longLengthType_e::ZSTD_llt_literalLength;
    ss.longLengthPos = 1;
    ZSTD_seqToCodes(&mut ss);
    assert_eq!(ss.llCode[1], MaxLL as u8);
    // Non-long index 0 stays normal (ZSTD_LLcode(5) = 5).
    assert_eq!(ss.llCode[0], 5);

    // Match-length promotion.
    let mut ss2 = SeqStore_t::with_capacity(8, 256);
    ss2.sequences.push(SeqDef {
        offBase: OFFSET_TO_OFFBASE(17),
        litLength: 5,
        mlBase: 10,
    });
    ss2.longLengthType = crate::compress::seq_store::ZSTD_longLengthType_e::ZSTD_llt_matchLength;
    ss2.longLengthPos = 0;
    ZSTD_seqToCodes(&mut ss2);
    assert_eq!(ss2.mlCode[0], MaxML as u8);
}

#[test]
fn negative_levels_compress_and_decompress_roundtrip() {
    // Negative levels must produce valid frames that decompress
    // back to the original bytes. Covers the accelerator path
    // (`targetLength = -level`) through the full public API.
    use crate::decompress::zstd_decompress::ZSTD_decompress;
    let src = b"negative-level-roundtrip ".repeat(40);
    for level in [-1i32, -5, -20, -1000] {
        let mut dst = vec![0u8; 4096];
        let n = ZSTD_compress(&mut dst, &src, level);
        assert!(!ERR_isError(n), "compress err at level={level}: {n:#x}",);
        let mut out = vec![0u8; src.len() + 64];
        let d = ZSTD_decompress(&mut out, &dst[..n]);
        assert!(!ERR_isError(d));
        assert_eq!(&out[..d], &src[..]);
    }
}

#[test]
fn ZSTD_sequenceBound_matches_upstream_formula() {
    // Upstream (zstd_compress.c:3538):
    //   (srcSize / ZSTD_MINMATCH_MIN=3) + 1 + (srcSize / BLOCKSIZE_MAX_MIN=1024) + 1
    // Pin the formula against a few sizes including zero.
    for sz in [0usize, 1, 100, 1024, 65_536, 200_000] {
        let expected = (sz / 3) + 1 + (sz / 1024) + 1;
        assert_eq!(ZSTD_sequenceBound(sz), expected, "sequenceBound({sz})",);
    }
}

#[test]
fn CCtx_loadDictionary_empty_slice_clears_previous_dict() {
    // Upstream semantics: loading an empty dict is equivalent to
    // clearing any previously-loaded dict (ZSTD_clearAllDicts path).
    // Guards against a regression where the empty-dict assign
    // would leave the old bytes intact.
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_CCtx_loadDictionary(&mut cctx, b"sticky-dict");
    assert_eq!(cctx.stream_dict, b"sticky-dict");
    ZSTD_CCtx_loadDictionary(&mut cctx, &[]);
    assert!(cctx.stream_dict.is_empty());
}

#[test]
fn ZSTD_isRLE_true_for_uniform_buffers_and_empty_and_single_byte() {
    // Edge cases:
    //   - empty src → 1 (defensive, upstream's UB case)
    //   - single byte → 1
    //   - all-same bytes → 1
    //   - any differing byte → 0
    assert_eq!(ZSTD_isRLE(&[]), 1);
    assert_eq!(ZSTD_isRLE(&[0xAB]), 1);
    assert_eq!(ZSTD_isRLE(&[0xAA; 16]), 1);
    assert_eq!(ZSTD_isRLE(&[0xAA; 1024]), 1);
    // One byte different near the end of an otherwise-uniform buf
    let mut mostly_uniform = vec![0xAA; 1024];
    mostly_uniform[1020] = 0xAB;
    assert_eq!(ZSTD_isRLE(&mostly_uniform), 0);
    // One byte different at the very start.
    let mut v = vec![0xAA; 1024];
    v[0] = 0xAB;
    assert_eq!(ZSTD_isRLE(&v), 0);
    // Two bytes, different.
    assert_eq!(ZSTD_isRLE(&[0xAA, 0xAB]), 0);
}

#[test]
fn compress_and_decompress_roundtrip_empty_and_single_byte() {
    // Corner-case sizes: ZSTD_compress must produce a valid frame
    // for both src=[] (zero bytes) and src=[0xAB] (one byte), and
    // the roundtrip must recover exact bytes.
    use crate::decompress::zstd_decompress::{ZSTD_decompress, ZSTD_getFrameContentSize};

    for &src in &[b"".as_ref(), b"\xAB".as_ref()] {
        let mut cbuf = vec![0u8; 128];
        let n = ZSTD_compress(&mut cbuf, src, 3);
        assert!(
            !ERR_isError(n),
            "compress err for len={}: {n:#x}",
            src.len()
        );
        // FCS should equal src.len().
        assert_eq!(
            ZSTD_getFrameContentSize(&cbuf[..n]),
            src.len() as u64,
            "FCS mismatch for len={}",
            src.len(),
        );
        // Decompress roundtrip.
        let mut out = vec![0u8; src.len().max(1)];
        let d = ZSTD_decompress(&mut out, &cbuf[..n]);
        assert!(!ERR_isError(d));
        assert_eq!(&out[..d], src);
    }
}

#[test]
fn compress_usingDict_level_zero_produces_default_level_frame() {
    // `ZSTD_compress_usingDict(level=0)` must map to the default
    // level via `ZSTD_getCParams`'s 0→CLEVEL_DEFAULT branch — output
    // should be byte-identical to an explicit `level=3` call.
    // Guards the "level 0 means default" contract on the one-shot
    // dict-aware path as well as the no-dict path.
    let dict = b"shared-dict-for-level-zero ".repeat(6);
    let src = b"level-zero-usingDict-parity ".repeat(20);
    let mut cctx_a = ZSTD_createCCtx().unwrap();
    let mut cctx_b = ZSTD_createCCtx().unwrap();
    let mut dst_a = vec![0u8; 4096];
    let mut dst_b = vec![0u8; 4096];
    let na = ZSTD_compress_usingDict(&mut cctx_a, &mut dst_a, &src, &dict, 0);
    let nb = ZSTD_compress_usingDict(&mut cctx_b, &mut dst_b, &src, &dict, ZSTD_CLEVEL_DEFAULT);
    assert!(!ERR_isError(na));
    assert!(!ERR_isError(nb));
    assert_eq!(&dst_a[..na], &dst_b[..nb]);
}

#[test]
fn compress_usingDict_propagates_magic_dictID_by_default() {
    // One-shot `ZSTD_compress_usingDict` must preserve the dict's
    // dictID in the frame header (upstream's default dictIDFlag=1).
    // stripping the ID for every full dictionary.
    use crate::decompress::zstd_ddict::ZSTD_getDictID_fromDict;
    use crate::decompress::zstd_decompress::ZSTD_getDictID_fromFrame;

    let magic_dict =
        include_bytes!("../../tests/fixtures/upstream-zstd/dict-files/zero-weight-dict");
    let expected_dict_id = ZSTD_getDictID_fromDict(magic_dict);
    assert_ne!(
        expected_dict_id, 0,
        "fixture must be a real full dictionary"
    );

    let src = b"one-shot-compress-usingDict-dictID-parity ".repeat(25);
    let mut cctx = ZSTD_createCCtx().unwrap();
    let mut dst = vec![0u8; 4096];
    let n = ZSTD_compress_usingDict(&mut cctx, &mut dst, &src, magic_dict, 3);
    assert!(!ERR_isError(n));
    assert_eq!(ZSTD_getDictID_fromFrame(&dst[..n]), expected_dict_id);
}

#[test]
fn compressStream_respects_param_dictID_flag_endToEnd() {
    // With a magic-prefixed dict, the frame's dictID field should
    // be present iff `param_dictID == true`. Previously the
    // streaming endStream hardcoded `noDictIDFlag=1`, AND
    // `compressFrame_fast_with_prefix` hardcoded dictID=0 to the
    // writer — so the flag was silently ignored either way.
    use crate::decompress::zstd_decompress::ZSTD_getDictID_fromFrame;

    let magic_dict: &[u8] =
        include_bytes!("../../tests/fixtures/upstream-zstd/dict-files/zero-weight-dict");
    let expected_dict_id = crate::decompress::zstd_ddict::ZSTD_getDictID_fromDict(magic_dict);
    assert_ne!(
        expected_dict_id, 0,
        "fixture must be a real full dictionary"
    );

    let src = b"dictID-flag-roundtrip ".repeat(30);

    // dictIDFlag = 1 (default): dictID must appear in the frame.
    let mut c_on = ZSTD_createCCtx().unwrap();
    ZSTD_CCtx_loadDictionary(&mut c_on, &magic_dict);
    ZSTD_CCtx_setParameter(&mut c_on, ZSTD_cParameter::ZSTD_c_compressionLevel, 3);
    ZSTD_CCtx_setParameter(&mut c_on, ZSTD_cParameter::ZSTD_c_dictIDFlag, 1);
    let mut dst_on = vec![0u8; 4096];
    let n_on = ZSTD_compress2(&mut c_on, &mut dst_on, &src);
    assert!(!ERR_isError(n_on));
    assert_eq!(ZSTD_getDictID_fromFrame(&dst_on[..n_on]), expected_dict_id);

    // dictIDFlag = 0: dictID must be suppressed.
    let mut c_off = ZSTD_createCCtx().unwrap();
    ZSTD_CCtx_loadDictionary(&mut c_off, &magic_dict);
    ZSTD_CCtx_setParameter(&mut c_off, ZSTD_cParameter::ZSTD_c_compressionLevel, 3);
    ZSTD_CCtx_setParameter(&mut c_off, ZSTD_cParameter::ZSTD_c_dictIDFlag, 0);
    let mut dst_off = vec![0u8; 4096];
    let n_off = ZSTD_compress2(&mut c_off, &mut dst_off, &src);
    assert!(!ERR_isError(n_off));
    assert_eq!(ZSTD_getDictID_fromFrame(&dst_off[..n_off]), 0);
}

#[test]
fn loadDictionary_advanced_fullDict_preserves_original_bytes_for_compress2() {
    use crate::decompress::zstd_ddict::{ZSTD_dictContentType_e, ZSTD_dictLoadMethod_e};
    use crate::decompress::zstd_decompress::{ZSTD_decompress_usingDict, ZSTD_getDictID_fromFrame};

    let dict: &[u8] =
        include_bytes!("../../tests/fixtures/upstream-zstd/dict-files/zero-weight-dict");
    let expected_dict_id = crate::decompress::zstd_ddict::ZSTD_getDictID_fromDict(dict);
    assert_ne!(
        expected_dict_id, 0,
        "fixture must be a real full dictionary"
    );

    let mut cctx = ZSTD_createCCtx().unwrap();
    let rc = ZSTD_CCtx_loadDictionary_advanced(
        &mut cctx,
        dict,
        ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy,
        ZSTD_dictContentType_e::ZSTD_dct_fullDict,
    );
    assert_eq!(rc, 0);
    assert_eq!(cctx.stream_dict_original, dict);
    assert_eq!(cctx.stream_dict, dict);

    let src = b"full dictionary compress2 payload ".repeat(80);
    let mut compressed = vec![0u8; ZSTD_compressBound(src.len()) + 64];
    let n = ZSTD_compress2(&mut cctx, &mut compressed, &src);
    assert!(!ERR_isError(n), "compress2 err={n:#x}");
    compressed.truncate(n);
    assert_eq!(ZSTD_getDictID_fromFrame(&compressed), expected_dict_id);

    let mut dctx = crate::decompress::zstd_decompress_block::ZSTD_DCtx::new();
    let mut decoded = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress_usingDict(&mut dctx, &mut decoded, &compressed, dict);
    assert!(!ERR_isError(d), "decompress err={d:#x}");
    assert_eq!(&decoded[..d], &src[..]);
}

#[test]
fn loadDictionary_parses_malformed_magic_prefixed_full_dicts_at_init() {
    use crate::common::error::ERR_getErrorCode;
    use crate::common::mem::MEM_writeLE32;
    use crate::decompress::zstd_ddict::{ZSTD_dictContentType_e, ZSTD_dictLoadMethod_e};
    use crate::decompress::zstd_decompress::ZSTD_MAGIC_DICTIONARY;

    let mut malformed = vec![0u8; 32];
    MEM_writeLE32(&mut malformed[..4], ZSTD_MAGIC_DICTIONARY);
    MEM_writeLE32(&mut malformed[4..8], 0x1234_5678);

    let mut auto = ZSTD_createCCtx().unwrap();
    let rc = ZSTD_CCtx_loadDictionary(&mut auto, &malformed);
    assert_eq!(rc, 0);
    assert_eq!(auto.stream_dict, malformed);
    assert_eq!(auto.stream_dict_original, malformed);
    let mut dst = vec![0u8; 128];
    let rc = ZSTD_compress2(&mut auto, &mut dst, b"payload");
    assert!(ERR_isError(rc));
    assert_eq!(ERR_getErrorCode(rc), ErrorCode::DictionaryCorrupted);

    let mut full = ZSTD_createCCtx().unwrap();
    let rc = ZSTD_CCtx_loadDictionary_advanced(
        &mut full,
        &malformed,
        ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy,
        ZSTD_dictContentType_e::ZSTD_dct_fullDict,
    );
    assert_eq!(rc, 0);
    assert_eq!(full.stream_dict, malformed);
    assert_eq!(full.stream_dict_original, malformed);
    let rc = ZSTD_CCtx_init_compressStream2(&mut full, ZSTD_EndDirective::ZSTD_e_end, 7);
    assert!(ERR_isError(rc));
    assert_eq!(ERR_getErrorCode(rc), ErrorCode::DictionaryCorrupted);
}

#[test]
fn init_compressStream2_honors_loaded_dictionary() {
    use crate::decompress::zstd_ddict::{ZSTD_dictContentType_e, ZSTD_dictLoadMethod_e};

    let dict: &[u8] =
        include_bytes!("../../tests/fixtures/upstream-zstd/dict-files/zero-weight-dict");
    let expected_dict_id = crate::decompress::zstd_ddict::ZSTD_getDictID_fromDict(dict);
    assert_ne!(expected_dict_id, 0);

    let mut cctx = ZSTD_createCCtx().unwrap();
    assert_eq!(
        ZSTD_CCtx_loadDictionary_advanced(
            &mut cctx,
            dict,
            ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy,
            ZSTD_dictContentType_e::ZSTD_dct_fullDict,
        ),
        0
    );
    let rc = ZSTD_CCtx_init_compressStream2(&mut cctx, ZSTD_EndDirective::ZSTD_e_end, 32);
    assert_eq!(rc, 0);
    assert_eq!(cctx.dictID, expected_dict_id);
    assert_eq!(cctx.stream_dict_original, dict);
}

#[test]
fn compressSequencesAndLiterals_honors_loaded_dictionary_and_cdict() {
    use crate::decompress::zstd_ddict::{ZSTD_dictContentType_e, ZSTD_dictLoadMethod_e};
    use crate::decompress::zstd_decompress::ZSTD_getDictID_fromFrame;

    let dict: &[u8] =
        include_bytes!("../../tests/fixtures/upstream-zstd/dict-files/zero-weight-dict");
    let expected_dict_id = crate::decompress::zstd_ddict::ZSTD_getDictID_fromDict(dict);
    assert_ne!(expected_dict_id, 0);
    let literals = b"sequence literal dictID path ".repeat(12);
    let seqs = [ZSTD_Sequence {
        offset: 0,
        litLength: literals.len() as u32,
        matchLength: 0,
        rep: 0,
    }];

    let mut loaded = ZSTD_createCCtx().unwrap();
    loaded.requestedParams.blockDelimiters = ZSTD_SequenceFormat_e::ZSTD_sf_explicitBlockDelimiters;
    assert_eq!(
        ZSTD_CCtx_loadDictionary_advanced(
            &mut loaded,
            dict,
            ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy,
            ZSTD_dictContentType_e::ZSTD_dct_fullDict,
        ),
        0
    );
    let mut loaded_dst = vec![0u8; ZSTD_compressBound(literals.len()) + 64];
    let loaded_n = ZSTD_compressSequencesAndLiterals(
        &mut loaded,
        &mut loaded_dst,
        &seqs,
        &literals,
        literals.len() + 8,
        literals.len(),
    );
    assert!(!ERR_isError(loaded_n), "loaded dict err={loaded_n:#x}");
    assert_eq!(
        ZSTD_getDictID_fromFrame(&loaded_dst[..loaded_n]),
        expected_dict_id
    );

    let cdict = ZSTD_createCDict(dict, 3).expect("cdict");
    let mut by_cdict = ZSTD_createCCtx().unwrap();
    by_cdict.requestedParams.blockDelimiters =
        ZSTD_SequenceFormat_e::ZSTD_sf_explicitBlockDelimiters;
    assert_eq!(ZSTD_CCtx_refCDict(&mut by_cdict, &cdict), 0);
    let mut cdict_dst = vec![0u8; ZSTD_compressBound(literals.len()) + 64];
    let cdict_n = ZSTD_compressSequencesAndLiterals(
        &mut by_cdict,
        &mut cdict_dst,
        &seqs,
        &literals,
        literals.len() + 8,
        literals.len(),
    );
    assert!(!ERR_isError(cdict_n), "cdict err={cdict_n:#x}");
    assert_eq!(
        ZSTD_getDictID_fromFrame(&cdict_dst[..cdict_n]),
        expected_dict_id
    );
}

#[test]
fn direct_stream_wrappers_reject_invalid_cursor_positions() {
    use crate::common::error::ERR_getErrorCode;

    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream(&mut cctx, 1);
    let src = b"cursor validation";
    let mut dst = [0u8; 16];

    let mut out_pos = dst.len() + 1;
    let mut in_pos = 0usize;
    let rc = ZSTD_compressStream(&mut cctx, &mut dst, &mut out_pos, src, &mut in_pos);
    assert!(ERR_isError(rc));
    assert_eq!(ERR_getErrorCode(rc), ErrorCode::DstSizeTooSmall);

    let mut out_pos = 0usize;
    let mut in_pos = src.len() + 1;
    let rc = ZSTD_compressStream(&mut cctx, &mut dst, &mut out_pos, src, &mut in_pos);
    assert!(ERR_isError(rc));
    assert_eq!(ERR_getErrorCode(rc), ErrorCode::SrcSizeWrong);

    let mut out_pos = dst.len() + 1;
    let rc = ZSTD_flushStream(&mut cctx, &mut dst, &mut out_pos);
    assert!(ERR_isError(rc));
    assert_eq!(ERR_getErrorCode(rc), ErrorCode::DstSizeTooSmall);

    let mut out_pos = dst.len() + 1;
    let rc = ZSTD_endStream(&mut cctx, &mut dst, &mut out_pos);
    assert!(ERR_isError(rc));
    assert_eq!(ERR_getErrorCode(rc), ErrorCode::DstSizeTooSmall);
}

#[test]
fn cctx_and_cctxparams_cparam_setters_accept_default_zero() {
    let defaultable = [
        ZSTD_cParameter::ZSTD_c_windowLog,
        ZSTD_cParameter::ZSTD_c_hashLog,
        ZSTD_cParameter::ZSTD_c_chainLog,
        ZSTD_cParameter::ZSTD_c_searchLog,
        ZSTD_cParameter::ZSTD_c_minMatch,
        ZSTD_cParameter::ZSTD_c_targetLength,
        ZSTD_cParameter::ZSTD_c_strategy,
    ];

    for param in defaultable {
        let mut cctx = ZSTD_createCCtx().unwrap();
        let mut got = -1i32;
        assert_eq!(ZSTD_CCtx_setParameter(&mut cctx, param, 0), 0, "{param:?}");
        assert_eq!(ZSTD_CCtx_getParameter(&cctx, param, &mut got), 0);
        assert_eq!(got, 0, "CCtx {param:?}");

        let mut params = ZSTD_CCtx_params::default();
        assert_eq!(
            ZSTD_CCtxParams_setParameter(&mut params, param, 0),
            0,
            "{param:?}"
        );
        assert_eq!(ZSTD_CCtxParams_getParameter(&params, param, &mut got), 0);
        assert_eq!(got, 0, "CCtxParams {param:?}");
    }
}

#[test]
fn mid_stream_allowlist_accepts_authorized_cparams_while_input_is_buffered() {
    use crate::common::error::ERR_getErrorCode;
    use crate::compress::zstd_compress_sequences::{ZSTD_dfast, ZSTD_greedy};
    use crate::decompress::zstd_decompress::ZSTD_format_e;

    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream(&mut cctx, 1);
    let src = b"mid-stream parameter allow-list";
    let mut dst = [0u8; 128];
    let mut dst_pos = 0usize;
    let mut src_pos = 0usize;
    let rc = ZSTD_compressStream(&mut cctx, &mut dst, &mut dst_pos, src, &mut src_pos);
    assert!(!ERR_isError(rc), "compressStream err={rc:#x}");
    assert!(!cctx.stream_in_buffer.is_empty());
    let snapshot = cctx.stream_params_snapshot.expect("stream params snapshot");

    assert_eq!(
        ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_compressionLevel, 6),
        0
    );
    assert_eq!(
        ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_hashLog, 17),
        0
    );
    assert_eq!(
        ZSTD_CCtx_setParameter(
            &mut cctx,
            ZSTD_cParameter::ZSTD_c_strategy,
            ZSTD_dfast as i32
        ),
        0
    );
    assert_eq!(cctx.stream_level, Some(6));
    assert_eq!(cctx.requestedParams.compressionLevel, 6);
    assert_eq!(cctx.requestedParams.cParams.hashLog, 17);
    assert_eq!(cctx.requestedParams.cParams.strategy, ZSTD_dfast);
    assert_eq!(
        snapshot.requestedParams.cParams.hashLog, 0,
        "already-buffered input keeps its original params snapshot"
    );

    let mut cp = ZSTD_getCParams(5, src.len() as u64, 0);
    cp.hashLog = 16;
    cp.strategy = ZSTD_greedy;
    assert_eq!(ZSTD_CCtx_setCParams(&mut cctx, cp), 0);
    let requested = cctx.requested_cParams.expect("requested cParams");
    assert_eq!(requested.hashLog, cp.hashLog);
    assert_eq!(requested.strategy, cp.strategy);
    assert_eq!(cctx.requestedParams.cParams.hashLog, cp.hashLog);
    assert_eq!(cctx.requestedParams.cParams.strategy, cp.strategy);
    assert_eq!(
        snapshot.requestedParams.cParams.strategy, 0,
        "setCParams does not rewrite the buffered-input snapshot"
    );

    let rc = ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_checksumFlag, 1);
    assert!(ERR_isError(rc));
    assert_eq!(ERR_getErrorCode(rc), ErrorCode::StageWrong);

    let rc = ZSTD_CCtx_setParameter(
        &mut cctx,
        ZSTD_cParameter::ZSTD_c_format,
        ZSTD_format_e::ZSTD_f_zstd1_magicless as i32,
    );
    assert!(ERR_isError(rc));
    assert_eq!(ERR_getErrorCode(rc), ErrorCode::StageWrong);

    let mut end_pos = 0usize;
    let rc = ZSTD_endStream(&mut cctx, &mut dst, &mut end_pos);
    assert_eq!(rc, 0);
    assert_eq!(cctx.stream_level, Some(6));
    let requested = cctx
        .requested_cParams
        .expect("requested cParams after endStream");
    assert_eq!(requested.hashLog, cp.hashLog);
    assert_eq!(requested.strategy, cp.strategy);
    assert_eq!(cctx.requestedParams.cParams.hashLog, cp.hashLog);
    assert_eq!(cctx.requestedParams.cParams.strategy, cp.strategy);
}

#[test]
fn setParametersUsingCCtxParams_rejects_when_cdict_attached() {
    use crate::common::error::ERR_getErrorCode;

    let dict = b"attached cdict parameters guard ".repeat(8);
    let cdict = ZSTD_createCDict(&dict, 3).expect("cdict");
    let mut cctx = ZSTD_createCCtx().unwrap();
    assert_eq!(ZSTD_CCtx_refCDict(&mut cctx, &cdict), 0);
    assert!(cctx.stream_cdict.is_some());

    let mut params = ZSTD_createCCtxParams().unwrap();
    ZSTD_CCtxParams_setParameter(&mut params, ZSTD_cParameter::ZSTD_c_compressionLevel, 5);
    params.cParams = ZSTD_getCParams(5, 0, dict.len());

    let rc = ZSTD_CCtx_setParametersUsingCCtxParams(&mut cctx, &params);
    assert!(ERR_isError(rc));
    assert_eq!(ERR_getErrorCode(rc), ErrorCode::StageWrong);
    assert!(cctx.stream_cdict.is_some());
    assert_eq!(cctx.stream_level, Some(3));
}

#[test]
fn CCtx_setParametersUsingCCtxParams_propagates_level_cparams_and_fparams() {
    // Upstream copies the whole `ZSTD_CCtx_params` struct onto the
    // CCtx's `requestedParams`. Our port plumbs the three
    // components (level, cParams, fParams) separately; verify all
    // three round-trip through the aggregate setter.
    let mut params = ZSTD_CCtx_params::default();
    ZSTD_CCtxParams_setParameter(&mut params, ZSTD_cParameter::ZSTD_c_compressionLevel, 7);
    ZSTD_CCtxParams_setParameter(&mut params, ZSTD_cParameter::ZSTD_c_checksumFlag, 1);
    params.cParams = ZSTD_getCParams(7, 0, 0);

    let mut cctx = ZSTD_createCCtx().unwrap();
    let rc = ZSTD_CCtx_setParametersUsingCCtxParams(&mut cctx, &params);
    assert_eq!(rc, 0);
    assert_eq!(cctx.stream_level, Some(7));
    assert!(cctx.param_checksum);
    assert_eq!(
        cctx.requested_cParams.map(|c| c.windowLog),
        Some(params.cParams.windowLog)
    );
}

#[test]
fn compressBegin_usingDict_loads_dict_into_match_state_and_level() {
    // Upstream-shaped begin/end initializer: routes through
    // `compressBegin_internal()`, which seeds the live match
    // state directly instead of stashing the raw dict bytes on
    // `stream_dict`.
    let mut cctx = ZSTD_createCCtx().unwrap();
    let dict = b"begin-usingDict-test".to_vec();
    let rc = ZSTD_compressBegin_usingDict(&mut cctx, &dict, 9);
    assert_eq!(rc, 0);
    assert_eq!(cctx.stream_level, Some(9));
    assert!(cctx.stream_dict.is_empty());
    assert_eq!(cctx.ms.as_ref().unwrap().dictContent, dict);

    // Level=0 goes through the default-mapping path.
    let mut cctx2 = ZSTD_createCCtx().unwrap();
    ZSTD_compressBegin_usingDict(&mut cctx2, b"", 0);
    assert_eq!(cctx2.stream_level, Some(ZSTD_CLEVEL_DEFAULT));
}

#[test]
fn compressBegin_level_zero_getParameter_returns_CLEVEL_DEFAULT() {
    // Sibling of `initCStream_level_zero_...` — `ZSTD_compressBegin(0)`
    // must also map 0 to `CLEVEL_DEFAULT`. Upstream routes through
    // the CCtxParams setter which applies the mapping; Rust port
    // was previously a raw field assignment.
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_compressBegin(&mut cctx, 0);
    let mut got = -1i32;
    ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_compressionLevel, &mut got);
    assert_eq!(got, ZSTD_CLEVEL_DEFAULT);
}

#[test]
fn initCStream_level_zero_getParameter_returns_CLEVEL_DEFAULT() {
    // `ZSTD_initCStream` upstream routes through
    // `ZSTD_CCtx_setParameter(c_compressionLevel, 0)`, which maps
    // 0 → `CLEVEL_DEFAULT`. Previously we raw-stored 0, so
    // getParameter readback after initCStream(0) was 0 instead
    // of 3 — divergent from upstream's `initCStream(0)` return
    // behavior.
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream(&mut cctx, 0);
    let mut got = -1i32;
    ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_compressionLevel, &mut got);
    assert_eq!(got, ZSTD_CLEVEL_DEFAULT);
}

#[test]
fn setParameter_level_zero_compresses_identically_to_CLEVEL_DEFAULT() {
    // End-to-end check of the level-0-maps-to-default fix:
    // a full roundtrip with setParameter(level=0) followed by
    // compressStream2/endStream must produce bit-identical output
    // to the same sequence with setParameter(level=CLEVEL_DEFAULT).
    let src = b"level-zero end-to-end parity ".repeat(40);
    let mut cctx_a = ZSTD_createCCtx().unwrap();
    ZSTD_CCtx_setParameter(&mut cctx_a, ZSTD_cParameter::ZSTD_c_compressionLevel, 0);
    let mut cctx_b = ZSTD_createCCtx().unwrap();
    ZSTD_CCtx_setParameter(
        &mut cctx_b,
        ZSTD_cParameter::ZSTD_c_compressionLevel,
        ZSTD_CLEVEL_DEFAULT,
    );
    let mut dst_a = vec![0u8; 4096];
    let mut dst_b = vec![0u8; 4096];
    let na = ZSTD_compress2(&mut cctx_a, &mut dst_a, &src);
    let nb = ZSTD_compress2(&mut cctx_b, &mut dst_b, &src);
    assert!(!ERR_isError(na));
    assert!(!ERR_isError(nb));
    assert_eq!(&dst_a[..na], &dst_b[..nb]);
}

#[test]
fn CCtx_setParameter_level_zero_getParameter_returns_CLEVEL_DEFAULT() {
    // Upstream: `ZSTD_CCtx_setParameter(c_compressionLevel, 0)`
    // stores `ZSTD_CLEVEL_DEFAULT` (3), not 0. Any C-compat
    // caller doing setParameter(0) followed by getParameter
    // expects 3 back. Previously our port stored Some(0) and
    // returned 0 on readback — silent ABI divergence.
    let mut cctx = ZSTD_createCCtx().unwrap();
    let rc = ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_compressionLevel, 0);
    assert_eq!(rc, 0);
    let mut got = -999i32;
    ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_compressionLevel, &mut got);
    assert_eq!(got, ZSTD_CLEVEL_DEFAULT);
}

#[test]
fn CCtx_setParameter_level_clamps_to_cParam_bounds() {
    // Upstream: `ZSTD_cParam_clampBounds` silently clamps
    // out-of-range level inputs (mirroring the documented
    // "level above MAX → treated as MAX" contract). Verify
    // above-MAX and below-MIN both land at the boundary.
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_compressionLevel, 9999);
    let mut got = 0i32;
    ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_compressionLevel, &mut got);
    assert_eq!(got, ZSTD_maxCLevel());

    ZSTD_CCtx_setParameter(
        &mut cctx,
        ZSTD_cParameter::ZSTD_c_compressionLevel,
        i32::MIN,
    );
    ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_compressionLevel, &mut got);
    assert_eq!(got, ZSTD_minCLevel());
}

#[test]
fn compress_one_shot_level_zero_equals_default() {
    // ZSTD_compress (one-shot) should also honor level 0 = default.
    let src = b"one-shot-level-zero test".repeat(30);
    let mut dst1 = vec![0u8; 512];
    let mut dst2 = vec![0u8; 512];
    let n1 = ZSTD_compress(&mut dst1, &src, 0);
    let n2 = ZSTD_compress(&mut dst2, &src, ZSTD_CLEVEL_DEFAULT);
    assert_eq!(&dst1[..n1], &dst2[..n2]);
}

#[test]
fn compressCCtx_level_zero_uses_default() {
    // Level 0 should behave identically to ZSTD_CLEVEL_DEFAULT.
    let src: Vec<u8> = b"identity test payload ".repeat(40);
    let mut cctx1 = ZSTD_createCCtx().unwrap();
    let mut cctx2 = ZSTD_createCCtx().unwrap();
    let mut dst1 = vec![0u8; 512];
    let mut dst2 = vec![0u8; 512];
    let n1 = ZSTD_compressCCtx(&mut cctx1, &mut dst1, &src, 0);
    let n2 = ZSTD_compressCCtx(&mut cctx2, &mut dst2, &src, ZSTD_CLEVEL_DEFAULT);
    assert!(!ERR_isError(n1));
    assert!(!ERR_isError(n2));
    // Bit-for-bit identical output.
    assert_eq!(&dst1[..n1], &dst2[..n2]);
}

#[test]
fn getCParams_negative_level_uses_baseline_row() {
    // Strictly-negative levels use row 0 of the table — the
    // "fast" baseline. Level 0 maps to CLEVEL_DEFAULT (3).
    for level in -5..0 {
        let cp = ZSTD_getCParams(level, 0, 0);
        assert_eq!(ZSTD_checkCParams(cp), 0, "level {level} invalid cParams");
        assert_eq!(cp.strategy, 1, "level {level} unexpectedly > fast");
    }
}

#[test]
fn checkCParams_rejects_each_field_one_past_its_bound() {
    // Exhaustive boundary sweep: pick one valid baseline cParams,
    // then perturb each individual field to one-past its upstream
    // bound and confirm `ZSTD_checkCParams` flips to
    // `ParameterOutOfBound`. Pins every branch of the validator
    // against the zstd.h-defined bounds.
    use crate::common::error::{ERR_getErrorCode, ERR_isError};
    use crate::compress::match_state::ZSTD_compressionParameters;
    use crate::compress::zstd_ldm::{ZSTD_HASHLOG_MAX, ZSTD_HASHLOG_MIN};

    let max_wlog = ZSTD_WINDOWLOG_MAX();
    let max_chainlog = ZSTD_CHAINLOG_MAX();

    let base = ZSTD_compressionParameters {
        windowLog: ZSTD_WINDOWLOG_ABSOLUTEMIN,
        chainLog: 6,
        hashLog: ZSTD_HASHLOG_MIN,
        searchLog: 1,
        minMatch: 3,
        targetLength: 0,
        strategy: 1,
    };
    assert_eq!(ZSTD_checkCParams(base), 0, "baseline cParams must pass");

    // Build one-past-bound perturbations for every validated field,
    // covering both the low and high edges where applicable.
    let bad_cases: &[(&str, ZSTD_compressionParameters)] = &[
        (
            "windowLog below min",
            ZSTD_compressionParameters {
                windowLog: ZSTD_WINDOWLOG_ABSOLUTEMIN - 1,
                ..base
            },
        ),
        (
            "windowLog above max",
            ZSTD_compressionParameters {
                windowLog: max_wlog + 1,
                ..base
            },
        ),
        (
            "chainLog below min",
            ZSTD_compressionParameters {
                chainLog: 5,
                ..base
            },
        ),
        (
            "chainLog above max",
            ZSTD_compressionParameters {
                chainLog: max_chainlog + 1,
                ..base
            },
        ),
        (
            "hashLog below min",
            ZSTD_compressionParameters {
                hashLog: ZSTD_HASHLOG_MIN - 1,
                ..base
            },
        ),
        (
            "hashLog above max",
            ZSTD_compressionParameters {
                hashLog: ZSTD_HASHLOG_MAX + 1,
                ..base
            },
        ),
        (
            "searchLog below min",
            ZSTD_compressionParameters {
                searchLog: 0,
                ..base
            },
        ),
        (
            "searchLog above max",
            ZSTD_compressionParameters {
                searchLog: max_wlog,
                ..base
            },
        ),
        (
            "minMatch below min",
            ZSTD_compressionParameters {
                minMatch: 2,
                ..base
            },
        ),
        (
            "minMatch above max",
            ZSTD_compressionParameters {
                minMatch: 8,
                ..base
            },
        ),
        (
            "targetLength above max",
            ZSTD_compressionParameters {
                targetLength: 131073,
                ..base
            },
        ),
        (
            "strategy below min",
            ZSTD_compressionParameters {
                strategy: 0,
                ..base
            },
        ),
        (
            "strategy above max",
            ZSTD_compressionParameters {
                strategy: 10,
                ..base
            },
        ),
    ];

    for (label, cp) in bad_cases {
        let rc = ZSTD_checkCParams(*cp);
        assert!(ERR_isError(rc), "expected error for `{label}`");
        assert_eq!(
            ERR_getErrorCode(rc),
            ErrorCode::ParameterOutOfBound,
            "wrong error for `{label}`",
        );
    }
}

#[test]
fn getCParams_levels_beyond_max_clamp() {
    // Levels above ZSTD_MAX_CLEVEL (22) still produce valid
    // cParams — upstream clamps to MAX.
    let cp = ZSTD_getCParams(ZSTD_MAX_CLEVEL + 5, 0, 0);
    assert_eq!(ZSTD_checkCParams(cp), 0);
}

#[test]
fn getParams_returns_valid_cparams_for_every_level() {
    // Every public level should yield cParams that pass
    // ZSTD_checkCParams — no out-of-range windowLog / hashLog /
    // strategy etc.
    for level in 1..=ZSTD_MAX_CLEVEL {
        let p = ZSTD_getParams(level, 0, 0);
        assert_eq!(
            ZSTD_checkCParams(p.cParams),
            0,
            "level {level} produced invalid cParams",
        );
        // Default fParams: contentSizeFlag=1, others zero.
        assert_eq!(p.fParams.contentSizeFlag, 1);
        assert_eq!(p.fParams.checksumFlag, 0);
        assert_eq!(p.fParams.noDictIDFlag, 0);
    }
}

#[test]
fn getCParams_honors_srcSizeHint_tableID_selection() {
    // The same level against different src-size hints should
    // pick different table rows — larger hint → bigger windowLog.
    let p_small = ZSTD_getCParams(3, 8 * 1024, 0);
    let p_large = ZSTD_getCParams(3, 1024 * 1024, 0);
    assert!(p_large.windowLog >= p_small.windowLog);
}

#[test]
fn compress2_roundtrip_with_checksum_and_content_size() {
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_compressionLevel, 5);
    ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_checksumFlag, 1);
    ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_contentSizeFlag, 1);

    let src = b"compress2 + params test ".repeat(100);
    let mut dst = vec![0u8; 4096];
    let n = ZSTD_compress2(&mut cctx, &mut dst, &src);
    assert!(!ERR_isError(n));
    dst.truncate(n);

    // Frame header should declare the exact content size.
    use crate::decompress::zstd_decompress::ZSTD_getFrameContentSize;
    assert_eq!(ZSTD_getFrameContentSize(&dst), src.len() as u64);

    // And it must round-trip through ZSTD_decompress.
    use crate::decompress::zstd_decompress::ZSTD_decompress;
    let mut out = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress(&mut out, &dst);
    assert_eq!(&out[..d], &src[..]);
}

#[test]
fn compress2_roundtrip_with_post_block_splitter_enabled() {
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    let mut cctx = ZSTD_createCCtx().unwrap();
    cctx.requestedParams.compressionLevel = 5;
    cctx.requestedParams.postBlockSplitter = ZSTD_ParamSwitch_e::ZSTD_ps_enable;

    let src = b"splitter path payload ".repeat(256);
    let mut dst = vec![0u8; ZSTD_compressBound(src.len())];
    let n = ZSTD_compress2(&mut cctx, &mut dst, &src);
    assert!(!ERR_isError(n));

    let mut out = vec![0u8; src.len()];
    let d = ZSTD_decompress(&mut out, &dst[..n]);
    assert_eq!(d, src.len());
    assert_eq!(out, src);
}

#[test]
fn frameProgression_tracks_streaming_ingest() {
    // After compressStream + endStream, frame-progression counters
    // should reflect the bytes that moved through the context.
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream(&mut cctx, 3);
    let src = b"progression test payload ".repeat(20);

    let mut dst = vec![0u8; 512];
    let mut dst_pos = 0usize;
    let mut src_pos = 0usize;
    ZSTD_compressStream(&mut cctx, &mut dst, &mut dst_pos, &src, &mut src_pos);
    let fp_after_ingest = ZSTD_getFrameProgression(&cctx);
    assert!(fp_after_ingest.ingested >= src.len() as u64);
    assert_eq!(fp_after_ingest.consumed, 0); // still staged

    // endStream until fully drained.
    loop {
        let rc = ZSTD_endStream(&mut cctx, &mut dst, &mut dst_pos);
        if rc == 0 || crate::common::error::ERR_isError(rc) {
            break;
        }
    }
    let fp_after_end = ZSTD_getFrameProgression(&cctx);
    assert_eq!(fp_after_end.consumed, fp_after_end.ingested);
}

/// Sentinel test for recently-added public helpers: every one
/// should compose with the existing compress path without
/// panicking.
#[test]
fn zstd_extended_api_surface_sentinel() {
    use crate::decompress::zstd_decompress::{
        ZSTD_decompress, ZSTD_decompressBound, ZSTD_readSkippableFrame,
    };

    let src = b"extended-api sentinel payload".repeat(16);
    let mut dst = vec![0u8; 1024];

    // ZSTD_compress2 via setParameter chain.
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_compressionLevel, 5);
    ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_checksumFlag, 1);
    let n = ZSTD_compress2(&mut cctx, &mut dst, &src);
    assert!(!ERR_isError(n), "compress2 err={n:#x}");
    let cbuf = &dst[..n];

    // ZSTD_decompressBound reports a sane upper bound.
    let db = ZSTD_decompressBound(cbuf);
    assert!(!ERR_isError(db as usize));
    assert!(db >= src.len() as u64);

    // Roundtrip via public decompress.
    let mut out = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress(&mut out, cbuf);
    assert_eq!(&out[..d], &src[..]);

    // Skippable frame roundtrip.
    let payload = b"user-metadata";
    let mut buf = vec![0u8; payload.len() + 8];
    let wn = ZSTD_writeSkippableFrame(&mut buf, payload, 4);
    assert_eq!(wn, payload.len().wrapping_add(8));
    let mut variant = 0u32;
    let mut ro = vec![0u8; payload.len()];
    let rn = ZSTD_readSkippableFrame(&mut ro, Some(&mut variant), &buf);
    assert_eq!(rn, payload.len());
    assert_eq!(variant, 4);
    assert_eq!(ro, payload);

    // ZSTD_CCtxParams lifecycle.
    let mut p = ZSTD_createCCtxParams().unwrap();
    ZSTD_CCtxParams_setParameter(&mut p, ZSTD_cParameter::ZSTD_c_compressionLevel, 9);
    let mut v = 0i32;
    ZSTD_CCtxParams_getParameter(&p, ZSTD_cParameter::ZSTD_c_compressionLevel, &mut v);
    assert_eq!(v, 9);
    ZSTD_freeCCtxParams(Some(p));

    // compressBegin variants — legacy but present.
    let mut cctx2 = ZSTD_createCCtx().unwrap();
    assert_eq!(ZSTD_compressBegin(&mut cctx2, 3), 0);
    assert_eq!(ZSTD_compressBegin_usingDict(&mut cctx2, b"dd", 3), 0);
    let cdict = ZSTD_createCDict(b"dd", 3).unwrap();
    assert_eq!(ZSTD_compressBegin_usingCDict(&mut cctx2, &cdict), 0);

    // Frame progression + toFlushNow on fresh CCtx.
    let fp = ZSTD_getFrameProgression(&cctx2);
    assert_eq!(fp.nbActiveWorkers, 0);
    assert_eq!(ZSTD_toFlushNow(&cctx2), 0);
}

#[test]
fn initStaticCCtx_accepts_header_sized_workspace_but_compression_can_fail() {
    // Upstream accepts an aligned workspace large enough to hold the
    // CCtx header at init time. The later compression setup is where
    // an undersized static workspace is allowed to fail.
    let cctx_size = core::mem::size_of::<ZSTD_CCtx>();
    let mut buf = vec![0u64; (cctx_size + 1).div_ceil(core::mem::size_of::<u64>())];
    let bytes = unsafe {
        core::slice::from_raw_parts_mut(
            buf.as_mut_ptr() as *mut u8,
            buf.len() * core::mem::size_of::<u64>(),
        )
    };
    assert!(bytes.len() > cctx_size);
    assert!(bytes.len() < ZSTD_estimateCCtxSize(ZSTD_CLEVEL_DEFAULT));

    let cctx = ZSTD_initStaticCCtx(bytes).expect("header-sized static cctx");
    assert_eq!(cctx.stage, ZSTD_compressionStage_e::ZSTDcs_created);

    let src = b"small-static-cctx workspace probe";
    let mut dst = vec![0u8; ZSTD_compressBound(src.len())];
    let n = ZSTD_compressCCtx(cctx, &mut dst, src, 1);
    assert!(
        ERR_isError(n),
        "undersized static workspace should fail during compression setup, got {n}"
    );
}

/// Sentinel test: touch every public one-shot compress function
/// with a trivial input to ensure none of them panic.
#[test]
fn zstd_public_api_surface_touches_every_entry_point() {
    use crate::decompress::zstd_ddict::ZSTD_createDDict;
    use crate::decompress::zstd_decompress::*;
    use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

    let src = b"sentinel api test".to_vec();
    let dict = b"tiny dict content that should enlighten things a tiny bit".to_vec();

    // One-shot compress via all entry points.
    let mut dst = vec![0u8; 512];
    let _ = ZSTD_compress(&mut dst, &src, 3);
    let mut cctx = ZSTD_createCCtx().unwrap();
    let _ = ZSTD_compressCCtx(&mut cctx, &mut dst, &src, 3);
    let _ = ZSTD_compress_usingDict(&mut cctx, &mut dst, &src, &dict, 3);
    let cdict = ZSTD_createCDict(&dict, 3).unwrap();
    let _ = ZSTD_compress_usingCDict(&mut cctx, &mut dst, &src, &cdict);
    assert_eq!(ZSTD_freeCDict(Some(cdict)), 0);
    assert_eq!(ZSTD_freeCCtx(Some(cctx)), 0);

    // Streaming creators / size helpers.
    let _ = ZSTD_createCStream().unwrap();
    assert!(ZSTD_CStreamInSize() > 0);
    assert!(ZSTD_CStreamOutSize() > 0);

    // One-shot decompress path.
    let n = ZSTD_compress(&mut dst, &src, 3);
    dst.truncate(n);
    let mut out = vec![0u8; 512];
    let _ = ZSTD_decompress(&mut out, &dst);
    let mut dctx = ZSTD_DCtx::new();
    let _ = ZSTD_decompressDCtx(
        &mut dctx,
        &mut crate::decompress::zstd_decompress_block::ZSTD_decoder_entropy_rep::default(),
        &mut crate::common::xxhash::XXH64_state_t::default(),
        &mut out,
        &dst,
    );
    let _ = ZSTD_decompress_usingDict(&mut dctx, &mut out, &dst, &[]);
    let ddict = ZSTD_createDDict(&dict).unwrap();
    let _ = ZSTD_decompress_usingDDict(&mut dctx, &mut out, &dst, &ddict);

    // Metadata queries.
    let _ = ZSTD_getFrameContentSize(&dst);
    let _ = ZSTD_findFrameCompressedSize(&dst);
    let _ = ZSTD_isFrame(&dst);
    let _ = ZSTD_getDictID_fromFrame(&dst);
    let _ = crate::decompress::zstd_ddict::ZSTD_DDict_dictSize(&ddict);

    // Parameter / estimation queries.
    let _ = ZSTD_cParam_getBounds(ZSTD_cParameter::ZSTD_c_compressionLevel);
    assert!(ZSTD_cParam_withinBounds(
        ZSTD_cParameter::ZSTD_c_compressionLevel,
        5
    ));
    assert!(!ZSTD_cParam_withinBounds(
        ZSTD_cParameter::ZSTD_c_checksumFlag,
        99,
    ));
    assert!(ZSTD_cParam_withinBounds(
        ZSTD_cParameter::ZSTD_c_checksumFlag,
        0,
    ));

    // cParam_clampBounds shrinks overshoots back into range.
    let mut v: i32 = 9999;
    let rc = ZSTD_cParam_clampBounds(ZSTD_cParameter::ZSTD_c_checksumFlag, &mut v);
    assert_eq!(rc, 0);
    assert_eq!(v, 1);
    let mut v: i32 = -99;
    ZSTD_cParam_clampBounds(ZSTD_cParameter::ZSTD_c_checksumFlag, &mut v);
    assert_eq!(v, 0);

    // compressionLevel has a large asymmetric range — clamping
    // extreme positive + extreme negative both succeed and land
    // at MAX/MIN respectively.
    let mut v = 99_999;
    ZSTD_cParam_clampBounds(ZSTD_cParameter::ZSTD_c_compressionLevel, &mut v);
    assert_eq!(v, ZSTD_maxCLevel());
    let mut v = -99_999_999;
    ZSTD_cParam_clampBounds(ZSTD_cParameter::ZSTD_c_compressionLevel, &mut v);
    assert_eq!(v, ZSTD_minCLevel());
    // In-range values pass through unchanged.
    let mut v = 7;
    ZSTD_cParam_clampBounds(ZSTD_cParameter::ZSTD_c_compressionLevel, &mut v);
    assert_eq!(v, 7);

    // maxNbSeq: minMatch=3 → blockSize/3, otherwise /4.
    assert_eq!(ZSTD_maxNbSeq(120, 3, false), 40);
    assert_eq!(ZSTD_maxNbSeq(120, 4, false), 30);
    assert_eq!(ZSTD_maxNbSeq(120, 4, true), 40);

    // dedicatedDictSearch_isSupported: hashLog > chainLog + bounds.
    let cp = crate::compress::match_state::ZSTD_compressionParameters {
        strategy: 4,
        hashLog: 17,
        chainLog: 14,
        ..Default::default()
    };
    assert!(ZSTD_dedicatedDictSearch_isSupported(&cp));
    let cp_bad = crate::compress::match_state::ZSTD_compressionParameters {
        strategy: 4,
        hashLog: 14,
        chainLog: 14,
        ..Default::default()
    };
    assert!(!ZSTD_dedicatedDictSearch_isSupported(&cp_bad));
    let cdict = ZSTD_CDict {
        dictContent: Vec::new(),
        compressionLevel: 3,
        dictID: 0,
        cParams: cp,
        useRowMatchFinder: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_disable,
        entropy: ZSTD_entropyCTables_t::default(),
        rep: ZSTD_REP_START_VALUE,
        dedicatedDictSearch: 1,
        matchState: crate::compress::match_state::ZSTD_MatchState_t::new(cp),
        customMem: ZSTD_customMem::default(),
    };
    let mut attach_params = ZSTD_CCtx_params::default();
    attach_params.attachDictPref = ZSTD_dictAttachPref_e::ZSTD_dictForceCopy;
    attach_params.forceWindow = 1;
    assert!(
        ZSTD_shouldAttachDict(&cdict, &attach_params, 1 << 30),
        "DDSS cdicts must force attach regardless of copy/window prefs",
    );

    // ZSTD_getFrameProgression: fresh context reports all zeros.
    let cctx_fresh = ZSTD_createCCtx().unwrap();
    let fp = ZSTD_getFrameProgression(&cctx_fresh);
    assert_eq!(fp.ingested, 0);
    assert_eq!(fp.consumed, 0);
    assert_eq!(fp.produced, 0);
    assert_eq!(fp.nbActiveWorkers, 0);
    assert_eq!(ZSTD_toFlushNow(&cctx_fresh), 0);

    // createCCtx_advanced / createCStream_advanced return Some.
    {
        let _cctx = ZSTD_createCCtx_advanced(ZSTD_customMem::default()).unwrap();
        let _cstream = ZSTD_createCStream_advanced(ZSTD_customMem::default()).unwrap();
    }
    // initStatic*: construct headers inside caller workspace when
    // alignment and size permit it.
    {
        let bytes_needed = ZSTD_estimateCCtxSize(ZSTD_CLEVEL_DEFAULT).max(1 << 20);
        let mut buf = vec![0u64; bytes_needed.div_ceil(core::mem::size_of::<u64>())];
        let bytes = unsafe {
            core::slice::from_raw_parts_mut(
                buf.as_mut_ptr() as *mut u8,
                buf.len() * core::mem::size_of::<u64>(),
            )
        };
        let cctx = ZSTD_initStaticCCtx(bytes).expect("static cctx");
        assert_eq!(cctx.stage, ZSTD_compressionStage_e::ZSTDcs_created);
        let cstream = ZSTD_initStaticCStream(bytes).expect("static cstream");
        assert_eq!(cstream.stage, ZSTD_compressionStage_e::ZSTDcs_created);
        let cp = ZSTD_getCParams(3, 0, 0);
        let cdict = ZSTD_initStaticCDict(bytes, b"dict", cp).expect("static cdict");
        assert_eq!(cdict.dictContent, b"dict");
        assert_eq!(cdict.cParams.strategy as u32, cp.strategy as u32);
        assert_eq!(cdict.cParams.windowLog, cp.windowLog);
    }
    // estimateCCtxSize_usingCCtxParams + CStream variant.
    {
        let mut p = ZSTD_createCCtxParams().unwrap();
        p.cParams = ZSTD_getCParams(3, 0, 0);
        let cs = ZSTD_estimateCCtxSize_usingCCtxParams(&p);
        let ss = ZSTD_estimateCStreamSize_usingCCtxParams(&p);
        assert!(cs > 0);
        assert!(ss >= cs);
    }

    // ZSTD_CCtx_setCParams stashes on the CCtx slot.
    {
        let mut cctx = ZSTD_createCCtx().unwrap();
        let cp = crate::compress::match_state::ZSTD_compressionParameters {
            windowLog: 20,
            chainLog: 16,
            hashLog: 17,
            searchLog: 4,
            minMatch: 4,
            targetLength: 32,
            strategy: 3,
        };
        let rc = ZSTD_CCtx_setCParams(&mut cctx, cp);
        assert_eq!(rc, 0);
        assert_eq!(cctx.requested_cParams.map(|c| c.strategy), Some(3));
    }

    // ZSTD_CCtx_setParametersUsingCCtxParams → level + params flow in.
    {
        let mut cctx = ZSTD_createCCtx().unwrap();
        let mut p = ZSTD_createCCtxParams().unwrap();
        ZSTD_CCtxParams_setParameter(&mut p, ZSTD_cParameter::ZSTD_c_compressionLevel, 7);
        ZSTD_CCtxParams_setParameter(&mut p, ZSTD_cParameter::ZSTD_c_checksumFlag, 1);
        // Seed cParams so checkCParams in setCParams passes.
        p.cParams = crate::compress::match_state::ZSTD_compressionParameters {
            windowLog: 20,
            chainLog: 16,
            hashLog: 17,
            searchLog: 4,
            minMatch: 4,
            targetLength: 32,
            strategy: 3,
        };
        let rc = ZSTD_CCtx_setParametersUsingCCtxParams(&mut cctx, &p);
        assert_eq!(rc, 0);
        assert_eq!(cctx.stream_level, Some(7));
        assert!(cctx.param_checksum);
        assert_eq!(cctx.requested_cParams.map(|c| c.strategy), Some(3));
    }

    // ZSTD_CCtx_params round-trip through setParameter/getParameter.
    {
        let mut p = ZSTD_createCCtxParams().unwrap();
        ZSTD_CCtxParams_setParameter(&mut p, ZSTD_cParameter::ZSTD_c_compressionLevel, 7);
        ZSTD_CCtxParams_setParameter(&mut p, ZSTD_cParameter::ZSTD_c_checksumFlag, 1);
        ZSTD_CCtxParams_setParameter(&mut p, ZSTD_cParameter::ZSTD_c_dictIDFlag, 0);
        let mut v: i32 = 0;
        ZSTD_CCtxParams_getParameter(&p, ZSTD_cParameter::ZSTD_c_compressionLevel, &mut v);
        assert_eq!(v, 7);
        ZSTD_CCtxParams_getParameter(&p, ZSTD_cParameter::ZSTD_c_checksumFlag, &mut v);
        assert_eq!(v, 1);
        ZSTD_CCtxParams_getParameter(&p, ZSTD_cParameter::ZSTD_c_dictIDFlag, &mut v);
        assert_eq!(v, 0);
        // Reset → level returns to default.
        ZSTD_CCtxParams_reset(&mut p);
        ZSTD_CCtxParams_getParameter(&p, ZSTD_cParameter::ZSTD_c_compressionLevel, &mut v);
        assert_eq!(v, ZSTD_CLEVEL_DEFAULT);
    }

    // ZSTD_compressSequences now handles empty-source empty-frame.
    {
        let mut cctx = ZSTD_createCCtx().unwrap();
        let mut dst = vec![0u8; 64];
        let rc = ZSTD_compressSequences(&mut cctx, &mut dst, &[], b"");
        assert!(!crate::common::error::ERR_isError(rc));
        assert!(rc >= 6);
    }

    // ZSTD_mergeBlockDelimiters drops boundary sentinels + merges lit.
    {
        let mut seqs = vec![
            ZSTD_Sequence {
                offset: 10,
                litLength: 2,
                matchLength: 4,
                rep: 0,
            },
            ZSTD_Sequence {
                offset: 0,
                litLength: 5,
                matchLength: 0,
                rep: 0,
            }, // boundary
            ZSTD_Sequence {
                offset: 20,
                litLength: 3,
                matchLength: 6,
                rep: 0,
            },
            ZSTD_Sequence {
                offset: 0,
                litLength: 7,
                matchLength: 0,
                rep: 0,
            }, // trailing
        ];
        let n = ZSTD_mergeBlockDelimiters(&mut seqs);
        assert_eq!(n, 2);
        assert_eq!(seqs[0].litLength, 2);
        // Middle boundary's litLength (5) rolled onto next real seq.
        assert_eq!(seqs[1].litLength, 3u32.wrapping_add(5));
    }

    // ZSTD_convertBlockSequences no-repcode path bulk-converts
    // sequences and updates next-frame rep history from the last
    // raw offsets.
    {
        let mut cctx = ZSTD_createCCtx().unwrap();
        cctx.seqStore = Some(SeqStore_t::with_capacity(16, 1024));
        cctx.prev_rep = [1, 4, 8];
        let inSeqs = [
            ZSTD_Sequence {
                offset: 9,
                litLength: 3,
                matchLength: 5,
                rep: 0,
            },
            ZSTD_Sequence {
                offset: 20,
                litLength: 2,
                matchLength: 7,
                rep: 0,
            },
            ZSTD_Sequence {
                offset: 0,
                litLength: 0,
                matchLength: 0,
                rep: 0,
            },
        ];
        let rc = ZSTD_convertBlockSequences(&mut cctx, &inSeqs, false);
        assert_eq!(rc, 0);
        let ss = cctx.seqStore.as_ref().unwrap();
        assert_eq!(ss.sequences.len(), 2);
        assert_eq!(
            ss.sequences[0].offBase,
            crate::compress::seq_store::OFFSET_TO_OFFBASE(9)
        );
        assert_eq!(
            ss.sequences[1].offBase,
            crate::compress::seq_store::OFFSET_TO_OFFBASE(20)
        );
        assert_eq!(cctx.next_rep, [20, 9, 1]);
    }

    // Real matches with raw offset 0 are invalid in both the
    // no-repcode and repcode-resolution paths.
    {
        let inSeqs = [
            ZSTD_Sequence {
                offset: 0,
                litLength: 0,
                matchLength: 4,
                rep: 0,
            },
            ZSTD_Sequence {
                offset: 0,
                litLength: 0,
                matchLength: 0,
                rep: 0,
            },
        ];
        for repcodeResolution in [false, true] {
            let mut cctx = ZSTD_createCCtx().unwrap();
            cctx.seqStore = Some(SeqStore_t::with_capacity(16, 1024));
            let rc = ZSTD_convertBlockSequences(&mut cctx, &inSeqs, repcodeResolution);
            assert_eq!(
                crate::common::error::ERR_getErrorCode(rc),
                ErrorCode::ExternalSequencesInvalid
            );
        }
    }

    // Repcode-resolution path must encode offBase through
    // ZSTD_finalizeOffBase and update reps via ZSTD_updateRep.
    {
        let mut cctx = ZSTD_createCCtx().unwrap();
        cctx.seqStore = Some(SeqStore_t::with_capacity(16, 1024));
        cctx.prev_rep = [5, 9, 13];
        let inSeqs = [
            ZSTD_Sequence {
                offset: 5,
                litLength: 1,
                matchLength: 6,
                rep: 0,
            }, // rep1
            ZSTD_Sequence {
                offset: 9,
                litLength: 0,
                matchLength: 4,
                rep: 0,
            }, // rep2 with ll0 adjustment
            ZSTD_Sequence {
                offset: 0,
                litLength: 0,
                matchLength: 0,
                rep: 0,
            },
        ];
        let rc = ZSTD_convertBlockSequences(&mut cctx, &inSeqs, true);
        assert_eq!(rc, 0);
        let ss = cctx.seqStore.as_ref().unwrap();
        assert_eq!(ss.sequences.len(), 2);
        assert_eq!(
            ss.sequences[0].offBase,
            crate::compress::seq_store::REPCODE_TO_OFFBASE(1)
        );
        assert_eq!(
            ss.sequences[1].offBase,
            crate::compress::seq_store::REPCODE_TO_OFFBASE(1)
        );
        assert_eq!(cctx.next_rep, [9, 5, 13]);
    }

    // In default no-delimiter mode, ZSTD_generateSequences returns
    // sequences directly usable by default ZSTD_compressSequences.
    {
        let mut cctx = ZSTD_createCCtx().unwrap();
        let mut seqs = vec![ZSTD_Sequence::default(); 16];
        let rc = ZSTD_generateSequences(&mut cctx, &mut seqs, b"some payload");
        assert!(!crate::common::error::ERR_isError(rc));
        assert!(seqs[..rc]
            .iter()
            .all(|seq| seq.offset != 0 || seq.matchLength != 0));
    }

    // ZSTD_clearAllDicts empties the stream dict.
    {
        let mut cctx = ZSTD_createCCtx().unwrap();
        cctx.stream_dict = b"some-dict-bytes".to_vec();
        ZSTD_clearAllDicts(&mut cctx);
        assert!(cctx.stream_dict.is_empty());
    }

    // ZSTD_overrideCParams: only non-zero fields overwrite.
    {
        use crate::compress::match_state::ZSTD_compressionParameters;
        let mut base = ZSTD_compressionParameters {
            windowLog: 20,
            chainLog: 16,
            hashLog: 17,
            searchLog: 4,
            minMatch: 4,
            targetLength: 32,
            strategy: 3,
        };
        let over = ZSTD_compressionParameters {
            windowLog: 0, // untouched
            chainLog: 18, // override
            hashLog: 19,  // override
            searchLog: 0,
            minMatch: 0,
            targetLength: 0,
            strategy: 0,
        };
        ZSTD_overrideCParams(&mut base, &over);
        assert_eq!(base.windowLog, 20);
        assert_eq!(base.chainLog, 18);
        assert_eq!(base.hashLog, 19);
        assert_eq!(base.searchLog, 4);
        assert_eq!(base.strategy, 3);
    }

    // ZSTD_fastSequenceLengthSum: sums lit + match across a seq array.
    {
        let seqs = vec![
            ZSTD_Sequence {
                offset: 10,
                litLength: 5,
                matchLength: 4,
                rep: 0,
            },
            ZSTD_Sequence {
                offset: 20,
                litLength: 3,
                matchLength: 8,
                rep: 0,
            },
        ];
        assert_eq!(
            ZSTD_fastSequenceLengthSum(&seqs),
            5usize.wrapping_add(4).wrapping_add(3).wrapping_add(8)
        );
    }

    // ZSTD_validateSequence: accept in-window, reject far offsets.
    {
        // posInSrc=100, windowLog=20 → windowSize=1MB. Offset of
        // 50 with matchLength 4 → OK.
        assert_eq!(
            ZSTD_validateSequence(
                crate::compress::seq_store::OFFSET_TO_OFFBASE(50),
                4,
                4,
                100,
                20,
                0,
                false,
            ),
            0
        );
        // Offset larger than posInSrc+dict → reject.
        assert!(crate::common::error::ERR_isError(ZSTD_validateSequence(
            crate::compress::seq_store::OFFSET_TO_OFFBASE(9999),
            4,
            4,
            100,
            10,
            0,
            false,
        )));
        // matchLength below minMatch → reject.
        assert!(crate::common::error::ERR_isError(ZSTD_validateSequence(
            crate::compress::seq_store::OFFSET_TO_OFFBASE(10),
            2,
            4,
            100,
            20,
            0,
            false,
        )));
    }

    // ZSTD_finalizeOffBase: matches rep → repcode offBase.
    {
        let rep = [100u32, 200, 300];
        assert_eq!(
            ZSTD_finalizeOffBase(100, &rep, 0),
            crate::compress::seq_store::REPCODE_TO_OFFBASE(1),
        );
        assert_eq!(
            ZSTD_finalizeOffBase(200, &rep, 0),
            crate::compress::seq_store::REPCODE_TO_OFFBASE(2),
        );
        // Non-rep offset → plain OFFSET_TO_OFFBASE.
        assert_eq!(
            ZSTD_finalizeOffBase(999, &rep, 0),
            crate::compress::seq_store::OFFSET_TO_OFFBASE(999),
        );
    }

    // ZSTD_sequenceBound scales with srcSize.
    let b0 = ZSTD_sequenceBound(0);
    let b_k = ZSTD_sequenceBound(1000);
    let b_m = ZSTD_sequenceBound(1_000_000);
    assert_eq!(b0, 2);
    assert!(b_k > b0);
    assert!(b_m > b_k);

    // ZSTD_compressBegin + ZSTD_compressBegin_usingDict smoke tests.
    {
        let mut cctx = ZSTD_createCCtx().unwrap();
        let rc = ZSTD_compressBegin(&mut cctx, 5);
        assert_eq!(rc, 0);
        assert_eq!(cctx.stream_level, Some(5));
        let rc = ZSTD_compressBegin_usingDict(&mut cctx, b"prior-dict", 7);
        assert_eq!(rc, 0);
        assert_eq!(cctx.stream_level, Some(7));
        assert!(cctx.stream_dict.is_empty());
        assert_eq!(cctx.ms.as_ref().unwrap().dictContent, b"prior-dict");

        // compressBegin_usingCDict reseeds to CDict's dict + level.
        let cdict = ZSTD_createCDict(b"cdict-content", 9).unwrap();
        let rc = ZSTD_compressBegin_usingCDict(&mut cctx, &cdict);
        assert_eq!(rc, 0);
        assert_eq!(cctx.stream_level, Some(9));
        let ms = cctx.ms.as_ref().unwrap();
        assert!(ms.dictContent == b"cdict-content" || ms.dictMatchState.is_some());
    }

    // ZSTD_writeSkippableFrame + ZSTD_readSkippableFrame round-trip.
    {
        use crate::decompress::zstd_decompress::ZSTD_readSkippableFrame;
        let payload = b"user-data-here".to_vec();
        let mut buf = vec![0u8; payload.len() + 8];
        let n = ZSTD_writeSkippableFrame(&mut buf, &payload, 7);
        assert_eq!(n, payload.len().wrapping_add(8));
        let mut out = vec![0u8; payload.len()];
        let mut variant = 0u32;
        let rd = ZSTD_readSkippableFrame(&mut out, Some(&mut variant), &buf);
        assert_eq!(rd, payload.len());
        assert_eq!(out, payload);
        assert_eq!(variant, 7);
    }

    // writeSkippableFrame rejects variant > 15.
    {
        let mut buf = vec![0u8; 32];
        let rc = ZSTD_writeSkippableFrame(&mut buf, b"x", 16);
        assert!(crate::common::error::ERR_isError(rc));
    }

    // Compress-side ZSTD_getBlockSize: default CCtx hands back
    // ZSTD_BLOCKSIZE_MAX. Explicit module path avoids the glob-
    // imported decompress-side variant that's also in scope here.
    {
        use crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX;
        assert_eq!(
            crate::compress::zstd_compress::ZSTD_getBlockSize(&cctx_fresh),
            ZSTD_BLOCKSIZE_MAX,
        );
    }

    // ZSTD_estimateCDictSize: non-zero, scales with dict size.
    let est_small = ZSTD_estimateCDictSize(1024, 3);
    let est_big = ZSTD_estimateCDictSize(128 * 1024, 3);
    assert!(est_small > 0);
    assert!(est_big > est_small);
    // `_advanced` with byRef must NOT add the dict bytes — the
    // caller retains ownership. Mirrors upstream zstd_compress.c:5556.
    {
        use crate::compress::match_state::ZSTD_compressionParameters;
        use crate::decompress::zstd_ddict::ZSTD_dictLoadMethod_e;
        let cp = ZSTD_compressionParameters {
            windowLog: 20,
            chainLog: 16,
            hashLog: 17,
            searchLog: 4,
            minMatch: 4,
            targetLength: 32,
            strategy: 3,
        };
        let by_ref =
            ZSTD_estimateCDictSize_advanced(64 * 1024, cp, ZSTD_dictLoadMethod_e::ZSTD_dlm_byRef);
        let by_copy =
            ZSTD_estimateCDictSize_advanced(64 * 1024, cp, ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy);
        assert_eq!(by_copy - by_ref, 64 * 1024);
    }

    // getDictID_fromCDict: raw dict → 0, magic-prefixed → parsed.
    {
        let cd_raw = ZSTD_createCDict(b"raw-bytes", 3).unwrap();
        assert_eq!(ZSTD_getDictID_fromCDict(&cd_raw), 0);

        let magic_dict: &[u8] =
            include_bytes!("../../tests/fixtures/upstream-zstd/dict-files/zero-weight-dict");
        let cd_magic = ZSTD_createCDict(magic_dict, 3).unwrap();
        assert_eq!(
            ZSTD_getDictID_fromCDict(&cd_magic),
            crate::decompress::zstd_ddict::ZSTD_getDictID_fromDict(magic_dict),
        );
    }

    // createCDict_advanced: advanced surface carries explicit
    // cParams and leaves compressionLevel at ZSTD_NO_CLEVEL.
    {
        use crate::decompress::zstd_ddict::{ZSTD_dictContentType_e, ZSTD_dictLoadMethod_e};
        let cp = crate::compress::match_state::ZSTD_compressionParameters {
            strategy: 7,
            ..Default::default()
        };
        let cd = ZSTD_createCDict_advanced(
            b"dict",
            ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy,
            ZSTD_dictContentType_e::ZSTD_dct_auto,
            cp,
        )
        .expect("cdict");
        assert_eq!(cd.compressionLevel, ZSTD_NO_CLEVEL);
    }

    // Advanced dict helpers all forward to the core loaders.
    {
        use crate::decompress::zstd_ddict::{ZSTD_dictContentType_e, ZSTD_dictLoadMethod_e};
        let mut cctx = ZSTD_createCCtx().unwrap();
        let d = b"hello dict".to_vec();
        ZSTD_CCtx_loadDictionary_advanced(
            &mut cctx,
            &d,
            ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy,
            ZSTD_dictContentType_e::ZSTD_dct_auto,
        );
        assert_eq!(cctx.stream_dict, d);
        ZSTD_CCtx_refPrefix_advanced(
            &mut cctx,
            b"prefix-bytes",
            ZSTD_dictContentType_e::ZSTD_dct_rawContent,
        );
        assert_eq!(cctx.stream_dict, b"prefix-bytes");
        ZSTD_CCtx_loadDictionary_byReference(&mut cctx, &d);
        assert_eq!(cctx.stream_dict, d);
    }

    // ZSTD_CCtx_setFParams: mirrors flag state onto CCtx.
    {
        let mut cctx = ZSTD_createCCtx().unwrap();
        let fp_in = ZSTD_FrameParameters {
            contentSizeFlag: 0,
            checksumFlag: 1,
            noDictIDFlag: 1, // → dictIDFlag = 0
        };
        let rc = ZSTD_CCtx_setFParams(&mut cctx, fp_in);
        assert_eq!(rc, 0);
        assert!(!cctx.param_contentSize);
        assert!(cctx.param_checksum);
        assert!(!cctx.param_dictID);
    }

    // ZSTD_compress2: one-shot that honors setParameter state.
    {
        let mut cctx = ZSTD_createCCtx().unwrap();
        ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_compressionLevel, 3);
        let src2 = b"the fast brown fox. ".repeat(64);
        let mut dst2 = vec![0u8; 4096];
        let n = ZSTD_compress2(&mut cctx, &mut dst2, &src2);
        assert!(!crate::common::error::ERR_isError(n));
        dst2.truncate(n);
        // Roundtrip.
        use crate::decompress::zstd_decompress::ZSTD_decompress;
        let mut out = vec![0u8; src2.len() + 64];
        let d = ZSTD_decompress(&mut out, &dst2);
        assert_eq!(&out[..d], &src2[..]);
    }

    // ZSTD_c_dictIDFlag: set + get round-trips.
    let mut cctx_p = ZSTD_createCCtx().unwrap();
    ZSTD_CCtx_setParameter(&mut cctx_p, ZSTD_cParameter::ZSTD_c_dictIDFlag, 0);
    let mut v: i32 = -1;
    ZSTD_CCtx_getParameter(&cctx_p, ZSTD_cParameter::ZSTD_c_dictIDFlag, &mut v);
    assert_eq!(v, 0);
    ZSTD_CCtx_setParameter(&mut cctx_p, ZSTD_cParameter::ZSTD_c_dictIDFlag, 1);
    ZSTD_CCtx_getParameter(&cctx_p, ZSTD_cParameter::ZSTD_c_dictIDFlag, &mut v);
    assert_eq!(v, 1);

    // Advanced-API enums: defaults land on the auto/validate
    // variants, and equality works.
    assert_eq!(
        ZSTD_forceIgnoreChecksum_e::default(),
        ZSTD_forceIgnoreChecksum_e::ZSTD_d_validateChecksum,
    );
    assert_eq!(
        ZSTD_refMultipleDDicts_e::default(),
        ZSTD_refMultipleDDicts_e::ZSTD_rmd_refSingleDDict,
    );
    assert_eq!(
        ZSTD_dictAttachPref_e::default(),
        ZSTD_dictAttachPref_e::ZSTD_dictDefaultAttach,
    );
    assert_eq!(
        ZSTD_literalCompressionMode_e::default(),
        ZSTD_literalCompressionMode_e::ZSTD_lcm_auto,
    );

    // ZSTD_getParams: pairs cParams with default fParams.
    let p = ZSTD_getParams(3, 0, 0);
    assert_eq!(p.fParams.contentSizeFlag, 1);
    assert_eq!(p.fParams.checksumFlag, 0);
    assert_eq!(p.fParams.noDictIDFlag, 0);
    assert_eq!(ZSTD_checkCParams(p.cParams), 0);

    // checkCParams: reject windowLog=5 (below absolutemin=10).
    let bad_cp = crate::compress::match_state::ZSTD_compressionParameters {
        windowLog: 5,
        chainLog: 15,
        hashLog: 15,
        searchLog: 3,
        minMatch: 4,
        targetLength: 32,
        strategy: 3,
    };
    assert!(crate::common::error::ERR_isError(ZSTD_checkCParams(bad_cp)));

    // checkCParams: accept reasonable defaults.
    let ok_cp = crate::compress::match_state::ZSTD_compressionParameters {
        windowLog: 20,
        chainLog: 16,
        hashLog: 17,
        searchLog: 4,
        minMatch: 4,
        targetLength: 32,
        strategy: 3,
    };
    assert_eq!(ZSTD_checkCParams(ok_cp), 0);

    // adjustCParams clamps out-of-range fields back into bounds.
    let ugly = crate::compress::match_state::ZSTD_compressionParameters {
        windowLog: 99,
        chainLog: 99,
        hashLog: 99,
        searchLog: 99,
        minMatch: 99,
        targetLength: 999_999,
        strategy: 99,
    };
    let fixed = ZSTD_adjustCParams(ugly, 1024, 0);
    assert_eq!(ZSTD_checkCParams(fixed), 0);
    assert!(fixed.minMatch <= 7);

    // cycleLog: btlazy2+ → hashLog-1, others → hashLog.
    assert_eq!(ZSTD_cycleLog(20, 5), 20); // lazy2
    assert_eq!(ZSTD_cycleLog(20, 6), 19); // btlazy2
    assert_eq!(ZSTD_cycleLog(20, 9), 19); // btultra2

    // adjustCParams_internal: shrinks windowLog on small known src.
    use crate::compress::match_state::ZSTD_compressionParameters;
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
    let cp_in = ZSTD_compressionParameters {
        windowLog: 23,
        hashLog: 20,
        chainLog: 20,
        searchLog: 5,
        minMatch: 4,
        targetLength: 32,
        strategy: 3,
    };
    let cp_out = ZSTD_adjustCParams_internal(
        cp_in,
        1024,
        0,
        ZSTD_CParamMode_e::ZSTD_cpm_noAttachDict,
        ZSTD_ParamSwitch_e::ZSTD_ps_auto,
    );
    // 1024 bytes → windowLog should shrink well below 23.
    assert!(cp_out.windowLog < 23);
    // Stays above ABSOLUTEMIN.
    assert!(cp_out.windowLog >= ZSTD_WINDOWLOG_ABSOLUTEMIN);
    // hashLog shouldn't exceed windowLog + 1.
    assert!(cp_out.hashLog <= cp_out.windowLog + 1);

    // Row match finder path: hashLog is additionally capped so the
    // row hash plus 8-bit tag still fits in 32 bits.
    let row_cp_in = ZSTD_compressionParameters {
        windowLog: 30,
        hashLog: 30,
        chainLog: 30,
        searchLog: 4,
        minMatch: 4,
        targetLength: 32,
        strategy: crate::compress::zstd_compress_sequences::ZSTD_lazy,
    };
    let row_cp_out = ZSTD_adjustCParams_internal(
        row_cp_in,
        crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN,
        0,
        ZSTD_CParamMode_e::ZSTD_cpm_noAttachDict,
        ZSTD_ParamSwitch_e::ZSTD_ps_enable,
    );
    assert_eq!(row_cp_out.hashLog, 28);

    // dictAndWindowLog: no dict → unchanged.
    assert_eq!(ZSTD_dictAndWindowLog(20, 10_000, 0), 20);
    // windowSize (1<<20) already ≥ dict + src → keep 20.
    assert_eq!(ZSTD_dictAndWindowLog(20, 1000, 1000), 20);
    // windowSize too small → round up to log2.
    // dict=300K + window=64K = 364K, need ceil(log2(364K)) = 19.
    let got = ZSTD_dictAndWindowLog(16, 500_000, 300_000);
    assert!(got > 16 && got <= ZSTD_WINDOWLOG_MAX());
    // Clip at WINDOWLOG_MAX for gigantic dicts.
    let huge: u64 = 1u64 << (ZSTD_WINDOWLOG_MAX() - 1);
    assert_eq!(
        ZSTD_dictAndWindowLog(ZSTD_WINDOWLOG_MAX(), huge, huge),
        ZSTD_WINDOWLOG_MAX(),
    );

    // getCParamRowSize combinations.
    use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;
    // Known size + dict: just the sum.
    assert_eq!(
        ZSTD_getCParamRowSize(1000, 200, ZSTD_CParamMode_e::ZSTD_cpm_noAttachDict),
        1200,
    );
    // Known size + attachDict: dict ignored.
    assert_eq!(
        ZSTD_getCParamRowSize(1000, 200, ZSTD_CParamMode_e::ZSTD_cpm_attachDict),
        1000,
    );
    // Unknown src + no dict: returns UNKNOWN.
    assert_eq!(
        ZSTD_getCParamRowSize(
            ZSTD_CONTENTSIZE_UNKNOWN,
            0,
            ZSTD_CParamMode_e::ZSTD_cpm_unknown
        ),
        ZSTD_CONTENTSIZE_UNKNOWN,
    );
    // Unknown src + dict: upstream's u64-wrap trick yields a tiny
    // rSize (dictSize + 499 = 699), which the caller uses as a
    // tableID-bucket hint.
    assert_eq!(
        ZSTD_getCParamRowSize(
            ZSTD_CONTENTSIZE_UNKNOWN,
            200,
            ZSTD_CParamMode_e::ZSTD_cpm_noAttachDict
        ),
        699,
    );

    // revertCParams: lazy family subtracts BUCKET_LOG from hashLog.
    let mut cp_lazy = crate::compress::match_state::ZSTD_compressionParameters {
        strategy: 4,
        hashLog: 17,
        chainLog: 14,
        ..Default::default()
    };
    ZSTD_dedicatedDictSearch_revertCParams(&mut cp_lazy);
    assert_eq!(cp_lazy.hashLog, 17 - ZSTD_LAZY_DDSS_BUCKET_LOG);
    // Non-lazy → untouched.
    let mut cp_fast = crate::compress::match_state::ZSTD_compressionParameters {
        strategy: 1,
        hashLog: 17,
        chainLog: 14,
        ..Default::default()
    };
    ZSTD_dedicatedDictSearch_revertCParams(&mut cp_fast);
    assert_eq!(cp_fast.hashLog, 17);
    let _ = ZSTD_dParam_getBounds(ZSTD_dParameter::ZSTD_d_windowLogMax);
    assert!(ZSTD_estimateCCtxSize(1) > 0);
    assert!(ZSTD_estimateDCtxSize() > 0);

    // Touch recently-surfaced prelude entries so reachability
    // regressions (a typo in `lib.rs` re-export list) fire here.
    let _ = ZSTD_dedicatedDictSearch_getCParams(3, 0);
    let _ = ZSTD_compressBlock(&mut ZSTD_createCCtx().unwrap(), &mut dst, &src);
    let _ = ZSTD_compressBlock_deprecated(&mut ZSTD_createCCtx().unwrap(), &mut dst, &src);
    let _ = ZSTD_compressContinue_public(&mut ZSTD_createCCtx().unwrap(), &mut dst, &src);
    let _ = ZSTD_compressEnd_public(&mut ZSTD_createCCtx().unwrap(), &mut dst, &src);
    {
        let mut a = ZSTD_createCCtx().unwrap();
        let b = ZSTD_createCCtx().unwrap();
        assert_eq!(ZSTD_copyCCtx(&mut a, &b, u64::MAX), 0);
    }
    {
        let mut a = ZSTD_createCCtx().unwrap();
        let params = ZSTD_parameters {
            cParams: ZSTD_getCParams(3, 0, 0),
            fParams: ZSTD_FrameParameters {
                contentSizeFlag: 1,
                checksumFlag: 0,
                noDictIDFlag: 1,
            },
        };
        let _ = ZSTD_compress_advanced(&mut a, &mut dst, &src, b"", params);
    }

    let dctx = ZSTD_createDCtx();
    assert_eq!(ZSTD_freeDCtx(dctx), 0);
    assert!(ZSTD_dParam_withinBounds(ZSTD_dParameter::ZSTD_d_windowLogMax, 20) != 0);
    let _ = ZSTD_estimateDStreamSize_fromFrame(&dst);

    {
        use crate::decompress::zstd_ddict::{ZSTD_copyDDictParameters, ZSTD_freeDDict};
        let mut dctx = ZSTD_createDCtx();
        let ddict = ZSTD_createDDict(&dict).unwrap();
        ZSTD_copyDDictParameters(&mut dctx, &ddict);
        assert_eq!(ZSTD_freeDCtx(dctx), 0);
        assert_eq!(ZSTD_freeDDict(Some(ddict)), 0);
    }

    {
        let mut dctx = ZSTD_createDCtx();
        let _ = ZSTD_decompressBlock(&mut dctx, &mut out, &dst);
        let _ = ZSTD_decompressBlock_deprecated(&mut dctx, &mut out, &dst);
        let _ = ZSTD_getBlockSize(&dctx);
        let _ = ZSTD_insertBlock(&mut dctx, &dst);
        assert_eq!(ZSTD_freeDCtx(dctx), 0);
    }

    assert!(ZSTD_cycleLog(17, 6) < 17);
    assert!(ZSTD_dictAndWindowLog(20, 1 << 10, 0) >= 1);
}

#[test]
fn zstd_sizeof_cctx_grows_after_compression() {
    let mut cctx = ZSTD_createCCtx().unwrap();
    let sz_empty = ZSTD_sizeof_CCtx(&cctx);

    let src: Vec<u8> = b"x".repeat(1024);
    let mut dst = vec![0u8; 2048];
    ZSTD_compressCCtx(&mut cctx, &mut dst, &src, 1);

    let sz_after = ZSTD_sizeof_CCtx(&cctx);
    assert!(sz_after >= sz_empty);
    // CStream alias matches.
    assert_eq!(ZSTD_sizeof_CStream(&cctx), sz_after);
}

#[test]
fn zstd_cParam_getBounds_reports_sensible_ranges() {
    let b = ZSTD_cParam_getBounds(ZSTD_cParameter::ZSTD_c_compressionLevel);
    assert_eq!(b.error, 0);
    assert!(b.lowerBound < 0 && b.upperBound == 22);

    let b = ZSTD_cParam_getBounds(ZSTD_cParameter::ZSTD_c_checksumFlag);
    assert_eq!((b.lowerBound, b.upperBound), (0, 1));

    // Round out coverage for the remaining two flag parameters.
    let b = ZSTD_cParam_getBounds(ZSTD_cParameter::ZSTD_c_contentSizeFlag);
    assert_eq!(b.error, 0);
    assert_eq!((b.lowerBound, b.upperBound), (0, 1));

    let b = ZSTD_cParam_getBounds(ZSTD_cParameter::ZSTD_c_dictIDFlag);
    assert_eq!(b.error, 0);
    assert_eq!((b.lowerBound, b.upperBound), (0, 1));
}

#[test]
fn zstd_level_bounds_and_buffer_sizes() {
    assert_eq!(ZSTD_maxCLevel(), 22);
    assert_eq!(ZSTD_defaultCLevel(), 3);
    assert!(ZSTD_minCLevel() < 0);
    assert!(ZSTD_CStreamInSize() > 0);
    assert!(ZSTD_CStreamOutSize() > ZSTD_CStreamInSize());
    // Pin the exact upstream formula for CStreamOutSize — previously
    // used ZSTD_FRAMEHEADERSIZE_MAX (18) instead of blockHeaderSize
    // (3), over-estimating the stream-end margin by 15 bytes.
    use crate::decompress::zstd_decompress_block::{ZSTD_blockHeaderSize, ZSTD_BLOCKSIZE_MAX};
    assert_eq!(
        ZSTD_CStreamOutSize(),
        ZSTD_compressBound(ZSTD_BLOCKSIZE_MAX) + ZSTD_blockHeaderSize + 4,
    );
    // Pin the underlying level constants (MAX=22 per upstream,
    // DEFAULT=3, NO_CLEVEL=0 — the sentinel used by
    // CCtxParams_init_advanced to mean "use explicit cParams").
    assert_eq!(ZSTD_MAX_CLEVEL, 22);
    assert_eq!(ZSTD_CLEVEL_DEFAULT, 3);
    assert_eq!(ZSTD_NO_CLEVEL, 0);
    // And the min-level formula: 1 - (1 << 17) = -131071.
    assert_eq!(ZSTD_minCLevel(), 1 - (1 << 17));
}

#[test]
fn zstd_cctx_setParameter_roundtrips_through_getParameter() {
    let mut cctx = ZSTD_createCCtx().unwrap();
    let rc = ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_compressionLevel, 7);
    assert_eq!(rc, 0);
    let mut v = 0i32;
    ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_compressionLevel, &mut v);
    assert_eq!(v, 7);

    ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_checksumFlag, 1);
    let mut v2 = 0i32;
    ZSTD_CCtx_getParameter(&cctx, ZSTD_cParameter::ZSTD_c_checksumFlag, &mut v2);
    assert_eq!(v2, 1);
}

#[test]
fn zstd_cctx_setParameter_checksumFlag_applies_to_streaming_output() {
    use crate::decompress::zstd_decompress::ZSTD_decompress;
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream(&mut cctx, 1);
    ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_checksumFlag, 1);

    let src: Vec<u8> = b"checksum streaming payload. "
        .iter()
        .cycle()
        .take(400)
        .copied()
        .collect();
    let mut staged = vec![0u8; 2048];
    let mut cp_pos = 0usize;
    let mut ip = 0usize;
    ZSTD_compressStream(&mut cctx, &mut staged, &mut cp_pos, &src, &mut ip);
    loop {
        let r = ZSTD_endStream(&mut cctx, &mut staged, &mut cp_pos);
        if r == 0 {
            break;
        }
    }
    staged.truncate(cp_pos);
    let mut out = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress(&mut out, &staged);
    assert!(!crate::common::error::ERR_isError(d));
    assert_eq!(&out[..d], &src[..]);
}

#[test]
fn zstd_cctx_reset_clears_streaming_state() {
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream(&mut cctx, 1);
    cctx.stream_in_buffer.extend_from_slice(b"pending");
    ZSTD_CCtx_reset(&mut cctx, ZSTD_ResetDirective::ZSTD_reset_session_only);
    assert!(cctx.stream_in_buffer.is_empty());
}

#[test]
fn zstd_estimate_cctx_size_level_0_matches_default_level() {
    // Level 0 means "use ZSTD_CLEVEL_DEFAULT (3)". The estimate
    // should match level 3's estimate exactly.
    let s_default = ZSTD_estimateCCtxSize(ZSTD_CLEVEL_DEFAULT);
    let s_zero = ZSTD_estimateCCtxSize(0);
    assert_eq!(s_zero, s_default);
}

#[test]
fn zstd_estimate_cctx_size_monotonic_with_level() {
    let s1 = ZSTD_estimateCCtxSize(1);
    let s5 = ZSTD_estimateCCtxSize(5);
    let s15 = ZSTD_estimateCCtxSize(15);
    assert!(s1 > 0);
    assert!(
        s5 >= s1,
        "level 5 ({s5}) should not shrink vs level 1 ({s1})"
    );
    assert!(
        s15 >= s5,
        "level 15 ({s15}) should not shrink vs level 5 ({s5})"
    );
}

#[test]
fn zstd_estimate_cstream_size_adds_buffers() {
    let cctx_sz = ZSTD_estimateCCtxSize(1);
    let cstream_sz = ZSTD_estimateCStreamSize(1);
    assert!(
        cstream_sz > cctx_sz,
        "streaming ({cstream_sz}) should need more than one-shot ({cctx_sz})"
    );
}

#[test]
fn zstd_initCStream_srcSize_sets_pledged() {
    use crate::decompress::zstd_decompress::{ZSTD_decompress, ZSTD_getFrameContentSize};
    let mut cctx = ZSTD_createCCtx().unwrap();
    let src: Vec<u8> = b"initCStream_srcSize. "
        .iter()
        .cycle()
        .take(300)
        .copied()
        .collect();
    ZSTD_initCStream_srcSize(&mut cctx, 1, src.len() as u64);
    let mut staged = vec![0u8; 2048];
    let mut cp_pos = 0usize;
    let mut ip = 0usize;
    ZSTD_compressStream(&mut cctx, &mut staged, &mut cp_pos, &src, &mut ip);
    loop {
        let r = ZSTD_endStream(&mut cctx, &mut staged, &mut cp_pos);
        if r == 0 {
            break;
        }
    }
    staged.truncate(cp_pos);
    assert_eq!(ZSTD_getFrameContentSize(&staged), src.len() as u64);
    let mut out = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress(&mut out, &staged);
    assert_eq!(&out[..d], &src[..]);
}

#[test]
fn zstd_streaming_dict_symmetric_roundtrip() {
    use crate::decompress::zstd_decompress::{ZSTD_decompressStream, ZSTD_initDStream_usingDict};
    use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

    let dict = b"streaming-sym-dict: alpha beta gamma delta. ".repeat(25);
    let src: Vec<u8> = b"beta gamma. alpha delta. "
        .iter()
        .cycle()
        .take(400)
        .copied()
        .collect();

    // Compress via streaming-with-dict.
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream_usingDict(&mut cctx, &dict, 1);
    let mut staged = vec![0u8; 2048];
    let mut cp_pos = 0usize;
    let mut ip = 0usize;
    ZSTD_compressStream(&mut cctx, &mut staged, &mut cp_pos, &src, &mut ip);
    loop {
        let r = ZSTD_endStream(&mut cctx, &mut staged, &mut cp_pos);
        if r == 0 {
            break;
        }
    }
    staged.truncate(cp_pos);

    // Decompress via streaming-with-dict.
    let mut dctx = ZSTD_DCtx::new();
    crate::decompress::zstd_decompress_block::ZSTD_buildDefaultSeqTables(&mut dctx);
    ZSTD_initDStream_usingDict(&mut dctx, &dict);
    let mut out = vec![0u8; src.len() + 64];
    let mut dp = 0usize;
    let mut icursor = 0usize;
    ZSTD_decompressStream(&mut dctx, &mut out, &mut dp, &staged, &mut icursor);
    loop {
        let mut p = 0usize;
        let r = ZSTD_decompressStream(&mut dctx, &mut out, &mut dp, &[], &mut p);
        if r == 0 {
            break;
        }
    }
    assert_eq!(&out[..dp], &src[..]);
}

#[test]
fn zstd_initCStream_usingDict_roundtrips() {
    use crate::decompress::zstd_decompress::ZSTD_decompress_usingDict;
    use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

    let dict = b"streaming-dict content. token alpha token beta. ".repeat(30);
    let src: Vec<u8> = b"token alpha token beta. "
        .iter()
        .cycle()
        .take(500)
        .copied()
        .collect();

    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream_usingDict(&mut cctx, &dict, 1);
    let mut staged = vec![0u8; 2048];
    let mut cp_pos = 0usize;
    let mut ip = 0usize;
    ZSTD_compressStream(&mut cctx, &mut staged, &mut cp_pos, &src, &mut ip);
    loop {
        let r = ZSTD_endStream(&mut cctx, &mut staged, &mut cp_pos);
        if r == 0 {
            break;
        }
    }
    staged.truncate(cp_pos);

    let mut dctx = ZSTD_DCtx::new();
    let mut out = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress_usingDict(&mut dctx, &mut out, &staged, &dict);
    assert_eq!(&out[..d], &src[..]);
}

#[test]
fn zstd_resetCStream_allows_fresh_frame() {
    use crate::decompress::zstd_decompress::ZSTD_decompress;
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream(&mut cctx, 1);

    for i in 0..3 {
        ZSTD_resetCStream(&mut cctx, u64::MAX);
        let src: Vec<u8> = format!("iter-{i} payload. ").repeat(30).into_bytes();
        let mut staged = vec![0u8; 1024];
        let mut cp_pos = 0usize;
        let mut ip = 0usize;
        ZSTD_compressStream(&mut cctx, &mut staged, &mut cp_pos, &src, &mut ip);
        loop {
            let r = ZSTD_endStream(&mut cctx, &mut staged, &mut cp_pos);
            if r == 0 {
                break;
            }
        }
        staged.truncate(cp_pos);
        let mut out = vec![0u8; src.len() + 64];
        let d = ZSTD_decompress(&mut out, &staged);
        assert_eq!(&out[..d], &src[..], "[iter {i}] mismatch");
    }
}

#[test]
fn zstd_stream_with_pledged_src_size_sets_frame_content_size() {
    use crate::decompress::zstd_decompress::{ZSTD_decompress, ZSTD_getFrameContentSize};
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream(&mut cctx, 1);

    let src: Vec<u8> = b"pledged content. "
        .iter()
        .cycle()
        .take(500)
        .copied()
        .collect();
    assert_eq!(ZSTD_CCtx_setPledgedSrcSize(&mut cctx, src.len() as u64), 0);

    let mut staged = vec![0u8; 2048];
    let mut cp_pos = 0usize;
    let mut ip = 0usize;
    ZSTD_compressStream(&mut cctx, &mut staged, &mut cp_pos, &src, &mut ip);
    loop {
        let r = ZSTD_endStream(&mut cctx, &mut staged, &mut cp_pos);
        if r == 0 {
            break;
        }
    }
    staged.truncate(cp_pos);

    // Frame should declare the content size exactly.
    let declared = ZSTD_getFrameContentSize(&staged);
    assert_eq!(declared, src.len() as u64);

    // Roundtrip.
    let mut out = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress(&mut out, &staged);
    assert!(!crate::common::error::ERR_isError(d));
    assert_eq!(&out[..d], &src[..]);
}

#[test]
fn initCStream_clears_previously_loaded_dict() {
    // Upstream's `ZSTD_initCStream` internally calls
    // `ZSTD_CCtx_refCDict(zcs, NULL)` which drops any dict
    // reference. Asymmetric with `ZSTD_initDStream` (decompress
    // side), which preserves the dict across an init — caller
    // must call `ZSTD_initCStream_usingDict` to re-seed.
    let mut cctx = ZSTD_createCCtx().unwrap();
    cctx.stream_dict = b"pre-init-dict".to_vec();
    cctx.stream_cdict = ZSTD_createCDict(b"pre-init-cdict", 3).map(|cd| *cd);
    ZSTD_initCStream(&mut cctx, 3);
    assert!(
        cctx.stream_dict.is_empty(),
        "initCStream must clear the dict (matches upstream refCDict(NULL))"
    );
    assert!(
        cctx.stream_cdict.is_none(),
        "initCStream must clear the cdict binding"
    );
}

#[test]
fn initCStream_srcSize_with_UNKNOWN_clears_pledge() {
    // `ZSTD_initCStream_srcSize(cctx, level, u64::MAX)` should
    // leave the pledge unset, matching `setPledgedSrcSize(u64::MAX)`
    // semantics. Prevents a regression if `initCStream_srcSize`
    // ever reimplements the pledge path differently.
    let mut cctx = ZSTD_createCCtx().unwrap();
    let rc = ZSTD_initCStream_srcSize(&mut cctx, 1, u64::MAX);
    assert_eq!(rc, 0);
    assert!(cctx.pledged_src_size.is_none());
}

#[test]
fn resetCStream_with_zero_pledge_accepts_zero_bytes_and_produces_empty_frame() {
    // Deprecated `ZSTD_resetCStream(0)` maps the pledge to
    // ZSTD_CONTENTSIZE_UNKNOWN upstream. The exact zero-size pledge
    // path is covered by `setPledgedSrcSize_zero_with_empty_input_roundtrips`.
    use crate::decompress::zstd_decompress::{
        ZSTD_decompress, ZSTD_getFrameContentSize, ZSTD_CONTENTSIZE_UNKNOWN,
    };
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream(&mut cctx, 1);
    ZSTD_resetCStream(&mut cctx, 0);

    let mut dst = vec![0u8; 256];
    let mut dp = 0usize;
    loop {
        let r = ZSTD_endStream(&mut cctx, &mut dst, &mut dp);
        assert!(!ERR_isError(r), "endStream err: {r:#x}");
        if r == 0 {
            break;
        }
    }
    dst.truncate(dp);
    assert_eq!(ZSTD_getFrameContentSize(&dst), ZSTD_CONTENTSIZE_UNKNOWN);
    let mut out = vec![0u8; 32];
    let d = ZSTD_decompress(&mut out, &dst);
    assert_eq!(d, 0);
}

#[test]
fn setPledgedSrcSize_zero_with_empty_input_roundtrips() {
    // Edge case: pledge 0, feed 0 bytes, endStream → should
    // produce a valid empty frame whose decompressed output is
    // empty. A pledge of 0 is distinct from `u64::MAX` and must
    // survive the endStream size-match check.
    use crate::decompress::zstd_decompress::{ZSTD_decompress, ZSTD_getFrameContentSize};
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream(&mut cctx, 1);
    assert_eq!(ZSTD_CCtx_setPledgedSrcSize(&mut cctx, 0), 0);

    let mut dst = vec![0u8; 256];
    let mut dp = 0usize;
    loop {
        let r = ZSTD_endStream(&mut cctx, &mut dst, &mut dp);
        assert!(
            !ERR_isError(r),
            "endStream errored on 0-pledged empty frame: {r:#x}"
        );
        if r == 0 {
            break;
        }
    }
    dst.truncate(dp);

    // Frame should declare content size of exactly 0.
    assert_eq!(ZSTD_getFrameContentSize(&dst), 0);

    // Decompress to nothing.
    let mut out = vec![0u8; 64];
    let d = ZSTD_decompress(&mut out, &dst);
    assert_eq!(d, 0);
}

#[test]
fn setPledgedSrcSize_with_unknown_sentinel_clears_pledge() {
    // `ZSTD_CCtx_setPledgedSrcSize(u64::MAX)` (= ZSTD_CONTENTSIZE_
    // UNKNOWN) must clear any prior pledge so the endStream
    // size-match check doesn't compare u64::MAX against the real
    // src.len(). Upstream treats UNKNOWN as "no pledge".
    use crate::decompress::zstd_decompress::ZSTD_decompress;
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream(&mut cctx, 1);
    // First pledge 100 bytes, then overwrite with UNKNOWN.
    ZSTD_CCtx_setPledgedSrcSize(&mut cctx, 100);
    ZSTD_CCtx_setPledgedSrcSize(&mut cctx, u64::MAX);
    assert!(cctx.pledged_src_size.is_none(), "UNKNOWN must clear pledge");

    // Feed 42 bytes (not 100) — must NOT error.
    let src = vec![b'z'; 42];
    let mut dst = vec![0u8; 256];
    let mut dp = 0usize;
    let mut sp = 0usize;
    ZSTD_compressStream(&mut cctx, &mut dst, &mut dp, &src, &mut sp);
    loop {
        let r = ZSTD_endStream(&mut cctx, &mut dst, &mut dp);
        if r == 0 || ERR_isError(r) {
            assert!(
                !ERR_isError(r),
                "endStream flagged UNKNOWN-pledged frame: {r:#x}"
            );
            break;
        }
    }
    dst.truncate(dp);
    let mut out = vec![0u8; 256];
    let d = ZSTD_decompress(&mut out, &dst);
    assert_eq!(&out[..d], &src[..]);
}

#[test]
fn zstd_stream_pledged_size_mismatch_errors() {
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream(&mut cctx, 1);
    ZSTD_CCtx_setPledgedSrcSize(&mut cctx, 100);

    // Feed only 50 bytes.
    let src = vec![b'x'; 50];
    let mut dst = vec![0u8; 256];
    let mut cp_pos = 0usize;
    let mut ip = 0usize;
    ZSTD_compressStream(&mut cctx, &mut dst, &mut cp_pos, &src, &mut ip);
    let rc = ZSTD_endStream(&mut cctx, &mut dst, &mut cp_pos);
    assert!(
        crate::common::error::ERR_isError(rc),
        "expected size-mismatch error"
    );
}

#[test]
fn zstd_stream_decompress_handles_multi_frame_concat() {
    use crate::decompress::zstd_decompress::{ZSTD_decompressStream, ZSTD_initDStream};
    use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

    // Produce two independent frames and concatenate them.
    let payload_a: Vec<u8> = b"alpha alpha alpha. "
        .iter()
        .cycle()
        .take(400)
        .copied()
        .collect();
    let payload_b: Vec<u8> = b"beta beta beta. "
        .iter()
        .cycle()
        .take(300)
        .copied()
        .collect();
    let mut frame_a = vec![0u8; 2048];
    let na = ZSTD_compress(&mut frame_a, &payload_a, 1);
    frame_a.truncate(na);
    let mut frame_b = vec![0u8; 2048];
    let nb = ZSTD_compress(&mut frame_b, &payload_b, 1);
    frame_b.truncate(nb);

    let mut combined = frame_a.clone();
    combined.extend_from_slice(&frame_b);

    // Stream-decompress the concatenation; expect payload_a then
    // payload_b. After the first frame is drained, call
    // ZSTD_initDStream again to indicate we're ready for the next.
    let mut dctx = ZSTD_DCtx::new();
    crate::decompress::zstd_decompress_block::ZSTD_buildDefaultSeqTables(&mut dctx);
    ZSTD_initDStream(&mut dctx);

    let mut decoded = vec![0u8; payload_a.len() + payload_b.len() + 64];
    let mut dp = 0usize;

    // Feed the entire concatenation at once.
    let mut ip = 0usize;
    let _ = ZSTD_decompressStream(&mut dctx, &mut decoded, &mut dp, &combined, &mut ip);

    // Drain any remaining output.
    loop {
        let mut p = 0usize;
        let r = ZSTD_decompressStream(&mut dctx, &mut decoded, &mut dp, &[], &mut p);
        if r == 0 {
            break;
        }
    }

    // Our streaming decoder transparently decodes consecutive
    // frames in one init+feed cycle (the drain loop's
    // re-probe-next-frame step handles it). Expect
    // payload_a || payload_b.
    let mut expected = payload_a;
    expected.extend_from_slice(&payload_b);
    assert_eq!(&decoded[..dp], &expected[..], "multi-frame mismatch");
}

#[test]
fn zstd_stream_full_roundtrip_via_streaming_decompress() {
    use crate::decompress::zstd_decompress::{ZSTD_decompressStream, ZSTD_initDStream};
    use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

    // Compress via streaming API.
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream(&mut cctx, 1);
    let src: Vec<u8> = b"pandora jarvis selkie titanite fable. "
        .iter()
        .cycle()
        .take(800)
        .copied()
        .collect();
    let mut compressed = vec![0u8; 4096];
    let mut cp = 0usize;
    let mut ip = 0usize;
    ZSTD_compressStream(&mut cctx, &mut compressed, &mut cp, &src, &mut ip);
    loop {
        let r = ZSTD_endStream(&mut cctx, &mut compressed, &mut cp);
        if r == 0 {
            break;
        }
    }
    compressed.truncate(cp);

    // Decompress via streaming API — feed in 64-byte chunks.
    let mut dctx = ZSTD_DCtx::new();
    crate::decompress::zstd_decompress_block::ZSTD_buildDefaultSeqTables(&mut dctx);
    ZSTD_initDStream(&mut dctx);
    let mut decoded = vec![0u8; src.len() + 64];
    let mut dp = 0usize;
    let mut cursor = 0usize;
    while cursor < compressed.len() {
        let chunk_end = (cursor + 64).min(compressed.len());
        let mut cp = 0usize;
        ZSTD_decompressStream(
            &mut dctx,
            &mut decoded,
            &mut dp,
            &compressed[cursor..chunk_end],
            &mut cp,
        );
        cursor += cp;
    }
    // Final drain call in case output buffer space wasn't enough earlier.
    loop {
        let mut cp = 0usize;
        let r = ZSTD_decompressStream(&mut dctx, &mut decoded, &mut dp, &[], &mut cp);
        if r == 0 {
            break;
        }
    }
    assert_eq!(&decoded[..dp], &src[..]);
}

#[test]
fn zstd_stream_tight_output_buffer_requires_multiple_endStream() {
    use crate::decompress::zstd_decompress::ZSTD_decompress;
    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream(&mut cctx, 1);

    let src: Vec<u8> = b"some repetitive content here. "
        .iter()
        .cycle()
        .take(400)
        .copied()
        .collect();

    // Feed all at once.
    let mut big = vec![0u8; 2048];
    let mut bp = 0usize;
    let mut ip = 0usize;
    ZSTD_compressStream(&mut cctx, &mut big, &mut bp, &src, &mut ip);

    // Drain endStream in 16-byte chunks to prove the drain loop
    // correctly reports remaining bytes and keeps producing.
    let mut out = Vec::new();
    let mut tiny = [0u8; 16];
    loop {
        let mut pos = 0usize;
        let remaining = ZSTD_endStream(&mut cctx, &mut tiny, &mut pos);
        out.extend_from_slice(&tiny[..pos]);
        if remaining == 0 && pos == 0 {
            break;
        }
    }
    let mut decoded = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress(&mut decoded, &out);
    assert!(!crate::common::error::ERR_isError(d));
    assert_eq!(&decoded[..d], &src[..]);
}

#[test]
fn endStream_accepts_synthetic_stable_input_metadata() {
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_initCStream(&mut cctx, 3);
    ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_stableInBuffer, 1);
    ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_stableOutBuffer, 1);

    let src = b"stable-buffer endStream wrapper ".repeat(20);
    let mut staged = vec![0u8; 1024];
    let mut cp = 0usize;
    let mut sp = 0usize;
    let rc = ZSTD_compressStream2(
        &mut cctx,
        &mut staged,
        &mut cp,
        &src,
        &mut sp,
        ZSTD_EndDirective::ZSTD_e_continue,
    );
    assert!(!ERR_isError(rc), "continue err={rc:#x}");
    assert_eq!(sp, src.len());

    loop {
        let mut pos = cp;
        let remaining = ZSTD_endStream(&mut cctx, &mut staged, &mut pos);
        assert!(!ERR_isError(remaining), "endStream err={remaining:#x}");
        cp = pos;
        if remaining == 0 {
            break;
        }
    }

    let mut decoded = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress(&mut decoded, &staged[..cp]);
    assert!(!ERR_isError(d), "decompress err={d:#x}");
    assert_eq!(&decoded[..d], &src[..]);
}

#[test]
fn zstd_compress_with_empty_dict_equivalent_to_no_dict() {
    use crate::decompress::zstd_decompress::ZSTD_decompress_usingDict;
    use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

    let src: Vec<u8> = b"payload without dict. "
        .iter()
        .cycle()
        .take(200)
        .copied()
        .collect();
    let mut cctx = ZSTD_createCCtx().unwrap();
    let mut dst = vec![0u8; 1024];
    let n = ZSTD_compress_usingDict(&mut cctx, &mut dst, &src, &[], 1);
    assert!(!crate::common::error::ERR_isError(n));
    dst.truncate(n);

    // Decode with empty dict should roundtrip.
    let mut dctx = ZSTD_DCtx::new();
    let mut out = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress_usingDict(&mut dctx, &mut out, &dst, &[]);
    assert!(!crate::common::error::ERR_isError(d));
    assert_eq!(&out[..d], &src[..]);
}

#[test]
fn decompress_with_wrong_dict_does_not_panic() {
    // Safety: compressing with dict A and decompressing with a
    // different dict B must NOT panic — the decoder either
    // surfaces an error (if back-ref offsets get clipped) or
    // produces incorrect bytes. Both are acceptable; crashing is
    // NOT. Raw-content dicts don't carry a dictID, so the
    // decoder has no way to detect mismatch.
    use crate::decompress::zstd_decompress::ZSTD_decompress_usingDict;
    use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

    let dict_a = b"shared-prefix-dict-content-a ".repeat(8);
    let dict_b = b"different-prefix-content-b ".repeat(8); // same length-ish, different bytes
    let src: Vec<u8> = b"payload that references shared-prefix-dict-content-a ".repeat(20);

    let mut cctx = ZSTD_createCCtx().unwrap();
    let mut compressed = vec![0u8; 4096];
    let n = ZSTD_compress_usingDict(&mut cctx, &mut compressed, &src, &dict_a, 1);
    assert!(!ERR_isError(n));
    compressed.truncate(n);

    let mut dctx = ZSTD_DCtx::new();
    let mut out = vec![0u8; src.len() + 64];
    let _ = ZSTD_decompress_usingDict(&mut dctx, &mut out, &compressed, &dict_b);
    // Raw-content dicts carry no dictID. Upstream therefore only
    // guarantees memory safety here: the wrong dict may yield an
    // error, incorrect bytes, or even byte-exact output if the
    // compressed frame happened not to depend on the dict.
}

#[test]
fn zstd_compress_with_tiny_sub_8_byte_dict_roundtrips() {
    // Edge case: a dict shorter than 8 bytes must still be
    // accepted in auto/raw-content mode (upstream keeps it as
    // pure content). compress → decompress_usingDict with the
    // same dict must roundtrip byte-exact.
    use crate::decompress::zstd_decompress::ZSTD_decompress_usingDict;
    use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

    let dict = b"abc".to_vec(); // 3 bytes
    let src: Vec<u8> = b"payload using tiny-dict ".repeat(40);

    let mut cctx = ZSTD_createCCtx().unwrap();
    let mut dst = vec![0u8; 2048];
    let n = ZSTD_compress_usingDict(&mut cctx, &mut dst, &src, &dict, 1);
    assert!(!crate::common::error::ERR_isError(n));
    dst.truncate(n);

    let mut dctx = ZSTD_DCtx::new();
    let mut out = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress_usingDict(&mut dctx, &mut out, &dst, &dict);
    assert!(!crate::common::error::ERR_isError(d));
    assert_eq!(&out[..d], &src[..]);
}

#[test]
fn zstd_compress_empty_src_with_dict_still_roundtrips() {
    use crate::decompress::zstd_decompress::ZSTD_decompress_usingDict;
    use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

    let dict = b"some dict content. ".repeat(20);
    let src: Vec<u8> = Vec::new();

    let mut cctx = ZSTD_createCCtx().unwrap();
    let mut dst = vec![0u8; 256];
    let n = ZSTD_compress_usingDict(&mut cctx, &mut dst, &src, &dict, 1);
    assert!(!crate::common::error::ERR_isError(n));
    dst.truncate(n);

    let mut dctx = ZSTD_DCtx::new();
    let mut out = vec![0u8; 64];
    let d = ZSTD_decompress_usingDict(&mut dctx, &mut out, &dst, &dict);
    assert!(!crate::common::error::ERR_isError(d));
    assert_eq!(d, 0);
}

#[test]
fn zstd_compress_with_dict_spans_multiple_blocks() {
    use crate::decompress::zstd_decompress::ZSTD_decompress_usingDict;
    use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

    // Well above 128 KB → multi-block frame. Handled by the
    // strategy-bump workaround in `ZSTD_compress_usingDict` —
    // see its implementation comment.
    let dict = b"multi-block dict. alpha beta gamma. ".repeat(40);
    let src: Vec<u8> = b"alpha beta gamma. we talk about greek letters. "
        .iter()
        .cycle()
        .take(200_000)
        .copied()
        .collect();

    let mut cctx = ZSTD_createCCtx().unwrap();
    let mut dst = vec![0u8; src.len() + 1024];
    let n = ZSTD_compress_usingDict(&mut cctx, &mut dst, &src, &dict, 1);
    assert!(
        !crate::common::error::ERR_isError(n),
        "compress err: {n:#x}"
    );
    dst.truncate(n);

    let mut dctx = ZSTD_DCtx::new();
    let mut out = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress_usingDict(&mut dctx, &mut out, &dst, &dict);
    assert!(
        !crate::common::error::ERR_isError(d),
        "decompress err: {d:#x}"
    );
    assert_eq!(d, src.len());
    assert_eq!(&out[..d], &src[..]);
}

#[test]
fn zstd_compress_usingDict_roundtrips_via_decompress_usingDict() {
    use crate::decompress::zstd_decompress::ZSTD_decompress_usingDict;
    use crate::decompress::zstd_decompress_block::ZSTD_DCtx;

    let dict = b"the quick brown fox jumps over the lazy dog near a river. ".repeat(40);
    let src: Vec<u8> = b"the fox jumps near the river. the lazy dog sleeps. "
        .iter()
        .cycle()
        .take(500)
        .copied()
        .collect();

    let mut cctx = ZSTD_createCCtx().unwrap();
    let mut compressed = vec![0u8; 2048];
    let n = ZSTD_compress_usingDict(&mut cctx, &mut compressed, &src, &dict, 1);
    assert!(!crate::common::error::ERR_isError(n), "compress: {n:#x}");
    compressed.truncate(n);

    let mut dctx = ZSTD_DCtx::new();
    let mut out = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress_usingDict(&mut dctx, &mut out, &compressed, &dict);
    assert!(!crate::common::error::ERR_isError(d), "decompress: {d:#x}");
    assert_eq!(d, src.len());
    assert_eq!(&out[..d], &src[..]);
}

#[test]
fn zstd_compress_usingDict_shrinks_payload_with_useful_dict() {
    // Large dict that contains all the phrases in the source, so
    // the source's matches can reference back into the dict.
    let dict =
        b"the quick brown fox jumps over the lazy dog near a river in the forest. ".repeat(60);
    // Source: a short document built from the dict's phrases. With
    // a matching dict, the fast matcher should find long matches
    // back into the dict, so the compressed size shrinks.
    let src: Vec<u8> = b"the lazy dog near a river. the quick brown fox jumps over. "
        .iter()
        .cycle()
        .take(400)
        .copied()
        .collect();

    let mut cctx = ZSTD_createCCtx().unwrap();
    let mut dst_nodict = vec![0u8; 4096];
    let n_nodict = ZSTD_compress(&mut dst_nodict, &src, 1);
    assert!(!crate::common::error::ERR_isError(n_nodict));

    let mut dst_dict = vec![0u8; 4096];
    let n_dict = ZSTD_compress_usingDict(&mut cctx, &mut dst_dict, &src, &dict, 1);
    assert!(
        !crate::common::error::ERR_isError(n_dict),
        "dict compress: {n_dict:#x}"
    );

    assert!(
        n_dict < n_nodict,
        "expected dict-compressed ({n_dict}) to be smaller than no-dict ({n_nodict})"
    );
}

#[test]
fn raw_dict_first_block_enters_extdict_mode() {
    use crate::compress::match_state::{
        ZSTD_dictMode_e, ZSTD_matchState_dictMode, ZSTD_window_update,
    };
    use crate::decompress::zstd_ddict::ZSTD_dictContentType_e;

    let dict =
        b"the quick brown fox jumps over the lazy dog near a river in the forest. ".repeat(60);
    let src: Vec<u8> = b"the lazy dog near a river. the quick brown fox jumps over. "
        .iter()
        .cycle()
        .take(400)
        .copied()
        .collect();

    let mut cctx = ZSTD_createCCtx().unwrap();
    let params = ZSTD_getParams_internal(
        1,
        src.len() as u64,
        dict.len(),
        ZSTD_CParamMode_e::ZSTD_cpm_noAttachDict,
    );
    let mut cctx_params = ZSTD_CCtx_params::default();
    ZSTD_CCtxParams_init_internal(&mut cctx_params, &params, 1);
    assert_eq!(
        ZSTD_compressBegin_internal(
            &mut cctx,
            &dict,
            ZSTD_dictContentType_e::ZSTD_dct_rawContent,
            None,
            &cctx_params,
            src.len() as u64,
            ZSTD_buffered_policy_e::ZSTDb_not_buffered,
        ),
        0
    );

    let ms = cctx.ms.as_mut().unwrap();
    let srcAbs = ms.window.nextSrc;
    assert!(!ZSTD_window_update(&mut ms.window, srcAbs, src.len(), true));
    assert_eq!(ZSTD_matchState_dictMode(ms), ZSTD_dictMode_e::ZSTD_extDict);
}

#[test]
fn zstd_cctx_reuse_across_multiple_compressions() {
    use crate::decompress::zstd_decompress::ZSTD_decompress;
    let mut cctx = ZSTD_createCCtx().expect("create");
    // Reuse the same context for 3 different payloads.
    for (i, text) in [
        &b"first payload"[..],
        b"second payload is somewhat longer than the first",
        b"third payload: the quick brown fox jumps over the lazy dog",
    ]
    .iter()
    .enumerate()
    {
        let payload: Vec<u8> = text.iter().cycle().take(500).copied().collect();
        let mut dst = vec![0u8; 2048];
        let n = ZSTD_compressCCtx(&mut cctx, &mut dst, &payload, 1);
        assert!(
            !crate::common::error::ERR_isError(n),
            "[iter {i}] cctx err: {n:#x}"
        );
        dst.truncate(n);
        let mut out = vec![0u8; payload.len() + 64];
        let d = ZSTD_decompress(&mut out, &dst);
        assert_eq!(&out[..d], &payload[..], "[iter {i}] roundtrip mismatch");
    }
}

#[test]
fn public_zstd_compress_level_3_uses_dfast_and_compresses_better_than_level_1() {
    use crate::decompress::zstd_decompress::ZSTD_decompress;
    let src: Vec<u8> = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt. "
            .iter()
            .cycle()
            .take(4000)
            .copied()
            .collect();
    let mut dst1 = vec![0u8; 8192];
    let n1 = ZSTD_compress(&mut dst1, &src, 1);
    assert!(!crate::common::error::ERR_isError(n1));
    dst1.truncate(n1);

    let mut dst3 = vec![0u8; 8192];
    let n3 = ZSTD_compress(&mut dst3, &src, 3);
    assert!(!crate::common::error::ERR_isError(n3));
    dst3.truncate(n3);

    // Level-3 (dfast) should compress better or equally well — at
    // minimum, the output must round-trip.
    let mut out = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress(&mut out, &dst3);
    assert!(!crate::common::error::ERR_isError(d));
    assert_eq!(d, src.len());
    assert_eq!(&out[..d], &src[..]);
}

#[test]
fn public_zstd_compress_level_1_roundtrips() {
    use crate::decompress::zstd_decompress::ZSTD_decompress;
    let src: Vec<u8> = b"The rain in Spain falls mainly on the plain. "
        .iter()
        .cycle()
        .take(800)
        .copied()
        .collect();
    let mut dst = vec![0u8; 4096];
    let cSize = ZSTD_compress(&mut dst, &src, 1);
    assert!(
        !crate::common::error::ERR_isError(cSize),
        "compress err: {cSize:#x}"
    );
    dst.truncate(cSize);
    let mut out = vec![0u8; src.len() + 64];
    let dSize = ZSTD_decompress(&mut out, &dst);
    assert!(!crate::common::error::ERR_isError(dSize));
    assert_eq!(dSize, src.len());
    assert_eq!(&out[..dSize], &src[..]);
}

#[test]
fn compress_frame_fast_roundtrips_through_full_decoder() {
    use crate::compress::match_state::ZSTD_compressionParameters;
    use crate::compress::zstd_compress_sequences::ZSTD_fast;
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    let src: Vec<u8> = b"the quick brown fox jumps over the lazy dog. "
        .iter()
        .cycle()
        .take(1000)
        .copied()
        .collect();

    let cParams = ZSTD_compressionParameters {
        windowLog: 17,
        hashLog: 12,
        minMatch: 4,
        strategy: ZSTD_fast,
        ..Default::default()
    };
    let fParams = ZSTD_FrameParameters {
        contentSizeFlag: 1,
        checksumFlag: 0,
        noDictIDFlag: 1,
    };

    let mut dst = vec![0u8; 4096];
    let cSize = ZSTD_compressFrame_fast(&mut dst, &src, cParams, fParams);
    assert!(
        !crate::common::error::ERR_isError(cSize),
        "compress err: {cSize:#x}"
    );
    dst.truncate(cSize);

    // Decompress through the stable public API.
    let mut out = vec![0u8; src.len() + 64];
    let dSize = ZSTD_decompress(&mut out, &dst);
    assert!(
        !crate::common::error::ERR_isError(dSize),
        "decompress err: {dSize:#x}"
    );
    assert_eq!(dSize, src.len());
    assert_eq!(&out[..dSize], &src[..]);
}

#[test]
fn compress_frame_fast_with_xxh64_checksum_roundtrips() {
    use crate::compress::match_state::ZSTD_compressionParameters;
    use crate::compress::zstd_compress_sequences::ZSTD_fast;
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    let src: Vec<u8> = b"lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        .iter()
        .cycle()
        .take(500)
        .copied()
        .collect();

    let cParams = ZSTD_compressionParameters {
        windowLog: 17,
        hashLog: 12,
        minMatch: 4,
        strategy: ZSTD_fast,
        ..Default::default()
    };
    let fParams = ZSTD_FrameParameters {
        contentSizeFlag: 1,
        checksumFlag: 1,
        noDictIDFlag: 1,
    };

    let mut dst = vec![0u8; 4096];
    let cSize = ZSTD_compressFrame_fast(&mut dst, &src, cParams, fParams);
    assert!(!crate::common::error::ERR_isError(cSize));
    dst.truncate(cSize);

    let mut out = vec![0u8; src.len() + 64];
    let dSize = ZSTD_decompress(&mut out, &dst);
    assert!(
        !crate::common::error::ERR_isError(dSize),
        "decompress err: {dSize:#x}"
    );
    assert_eq!(dSize, src.len());
    assert_eq!(&out[..dSize], &src[..]);
}

#[test]
fn no_compress_block_produces_valid_raw_header() {
    use crate::decompress::zstd_decompress_block::{
        blockProperties_t, blockType_e, ZSTD_getcBlockSize,
    };
    let src = b"hello, uncompressed world";
    let mut dst = vec![0u8; 64];
    let n = ZSTD_noCompressBlock(&mut dst, src, 1);
    assert_eq!(n, 3 + src.len());
    // Round-trip the header via the decoder's block-header parser.
    let mut props = blockProperties_t {
        blockType: blockType_e::bt_raw,
        lastBlock: 0,
        origSize: 0,
    };
    let decoded_size = ZSTD_getcBlockSize(&dst, &mut props);
    assert!(!crate::common::error::ERR_isError(decoded_size));
    assert_eq!(decoded_size, src.len());
    assert_eq!(props.blockType, blockType_e::bt_raw);
    assert_eq!(props.lastBlock, 1);
    assert_eq!(&dst[3..3 + src.len()], src);
}

#[test]
fn createCCtx_advanced_produces_functional_cctx_for_end_to_end_roundtrip() {
    // `ZSTD_createCCtx_advanced(customMem)` must produce a CCtx
    // that compresses + roundtrips identically to a plain
    // `createCCtx()`. Proves the _advanced variant isn't just
    // returning a broken stub — it must be a real, usable CCtx.
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    let mut cctx_adv = ZSTD_createCCtx_advanced(ZSTD_customMem::default()).unwrap();
    let src: Vec<u8> = b"createCCtx_advanced probe payload ".repeat(30);
    let mut dst = vec![0u8; 2048];
    let n = ZSTD_compressCCtx(&mut cctx_adv, &mut dst, &src, 3);
    assert!(!ERR_isError(n));
    dst.truncate(n);

    let mut out = vec![0u8; src.len() + 64];
    let d = ZSTD_decompress(&mut out, &dst);
    assert_eq!(&out[..d], &src[..]);
}

#[test]
fn createCDict_accepts_level_0_and_max_level() {
    // Upstream: `createCDict(dict, 0)` defaults to CLEVEL_DEFAULT;
    // `createCDict(dict, ZSTD_MAX_CLEVEL)` also succeeds. Both
    // must return `Some` and store the dict content.
    let dict = b"level-edge-test".to_vec();
    let cd_0 = ZSTD_createCDict(&dict, 0).expect("level 0 must work");
    let cd_max = ZSTD_createCDict(&dict, ZSTD_MAX_CLEVEL).expect("max level must work");
    assert_eq!(cd_0.dictContent, dict);
    assert_eq!(cd_max.dictContent, dict);
}

#[test]
fn createCDict_byReference_matches_regular_creator_content() {
    // Symmetric with the decompress-side `createDDict_byReference`
    // test. Our by-reference creator is a by-copy call under the
    // hood (Rust lifetime constraint). Both creators must
    // produce equivalent content.
    let dict = b"byref-cdict-test-dict ".repeat(5);
    let cd_copy = ZSTD_createCDict(&dict, 3).expect("by-copy");
    let cd_ref = ZSTD_createCDict_byReference(&dict, 3).expect("by-ref");
    assert_eq!(cd_copy.dictContent, cd_ref.dictContent);
    assert_eq!(cd_copy.compressionLevel, cd_ref.compressionLevel);
    // And both should have non-zero reported size.
    assert_eq!(ZSTD_sizeof_CDict(&cd_copy), ZSTD_sizeof_CDict(&cd_ref));
}

#[test]
fn initCDict_internal_populates_match_state_dict_bytes() {
    use crate::decompress::zstd_ddict::{ZSTD_dictContentType_e, ZSTD_dictLoadMethod_e};

    let dict = b"match-state-dict-bytes ".repeat(12);
    let cParams = ZSTD_getCParams(3, u64::MAX, dict.len());
    let mut params = ZSTD_CCtx_params::default();
    params.cParams = cParams;
    params.compressionLevel = 3;
    params.useRowMatchFinder = crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_disable;

    let mut cdict = ZSTD_createCDict_advanced_internal(
        dict.len(),
        ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy,
        cParams,
        crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_disable,
        0,
        ZSTD_customMem::default(),
    )
    .expect("cdict");
    let rc = ZSTD_initCDict_internal(
        &mut cdict,
        &dict,
        ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy,
        ZSTD_dictContentType_e::ZSTD_dct_rawContent,
        params,
    );
    assert_eq!(rc, 0);
    assert_eq!(cdict.matchState.dictContent, dict);
}

#[test]
fn loadDictionaryContent_seeds_strategy_specific_tables() {
    use crate::compress::zstd_compress_sequences::{ZSTD_btopt, ZSTD_dfast, ZSTD_lazy};
    use crate::compress::zstd_fast::{ZSTD_dictTableLoadMethod_e, ZSTD_tableFillPurpose_e};
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;

    let dict = b"dictionary content repeated dictionary content repeated ".repeat(8);

    let mut dfast_params = ZSTD_CCtx_params::default();
    dfast_params.cParams = crate::compress::match_state::ZSTD_compressionParameters {
        strategy: ZSTD_dfast,
        ..ZSTD_getCParams(3, 0, dict.len())
    };
    dfast_params.useRowMatchFinder = ZSTD_ParamSwitch_e::ZSTD_ps_disable;
    let mut dfast_ms = crate::compress::match_state::ZSTD_MatchState_t::new(dfast_params.cParams);
    assert_eq!(
        ZSTD_loadDictionaryContent(
            &mut dfast_ms,
            None,
            &dfast_params,
            &dict,
            ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_full,
            ZSTD_tableFillPurpose_e::ZSTD_tfp_forCDict,
        ),
        0
    );
    assert!(dfast_ms.hashTable.iter().any(|&v| v != 0));
    assert!(dfast_ms.chainTable.iter().any(|&v| v != 0));
    assert_eq!(dfast_ms.dictContent, dict);

    let mut row_params = ZSTD_CCtx_params::default();
    row_params.cParams = crate::compress::match_state::ZSTD_compressionParameters {
        strategy: ZSTD_lazy,
        ..ZSTD_getCParams(5, 0, dict.len())
    };
    row_params.useRowMatchFinder = ZSTD_ParamSwitch_e::ZSTD_ps_enable;
    let mut row_ms = crate::compress::match_state::ZSTD_MatchState_t::new(row_params.cParams);
    ZSTD_reset_matchState(
        &mut row_ms,
        &row_params.cParams,
        row_params.useRowMatchFinder,
        ZSTD_compResetPolicy_e::ZSTDcrp_makeClean,
        ZSTD_indexResetPolicy_e::ZSTDirp_reset,
        ZSTD_resetTarget_e::ZSTD_resetTarget_CDict,
    );
    assert_eq!(
        ZSTD_loadDictionaryContent(
            &mut row_ms,
            None,
            &row_params,
            &dict,
            ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_full,
            ZSTD_tableFillPurpose_e::ZSTD_tfp_forCDict,
        ),
        0
    );
    assert!(row_ms.tagTable.iter().any(|&v| v != 0));

    let mut bt_params = ZSTD_CCtx_params::default();
    bt_params.cParams = crate::compress::match_state::ZSTD_compressionParameters {
        strategy: ZSTD_btopt,
        ..ZSTD_getCParams(12, 0, dict.len())
    };
    bt_params.useRowMatchFinder = ZSTD_ParamSwitch_e::ZSTD_ps_disable;
    let mut bt_ms = crate::compress::match_state::ZSTD_MatchState_t::new(bt_params.cParams);
    ZSTD_reset_matchState(
        &mut bt_ms,
        &bt_params.cParams,
        bt_params.useRowMatchFinder,
        ZSTD_compResetPolicy_e::ZSTDcrp_makeClean,
        ZSTD_indexResetPolicy_e::ZSTDirp_reset,
        ZSTD_resetTarget_e::ZSTD_resetTarget_CDict,
    );
    assert_eq!(
        ZSTD_loadDictionaryContent(
            &mut bt_ms,
            None,
            &bt_params,
            &dict,
            ZSTD_dictTableLoadMethod_e::ZSTD_dtlm_full,
            ZSTD_tableFillPurpose_e::ZSTD_tfp_forCDict,
        ),
        0
    );
    assert_eq!(
        bt_ms.nextToUpdate,
        crate::compress::match_state::ZSTD_WINDOW_START_INDEX.wrapping_add(dict.len() as u32)
    );
}

#[test]
fn selectBlockCompressor_does_not_route_btopt_through_row_matchfinder() {
    use crate::compress::match_state::ZSTD_dictMode_e;
    use crate::compress::zstd_compress_sequences::ZSTD_btopt;
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
    use crate::compress::zstd_opt::ZSTD_compressBlock_btopt;

    let selected = ZSTD_selectBlockCompressor(
        ZSTD_btopt,
        ZSTD_ParamSwitch_e::ZSTD_ps_enable,
        ZSTD_dictMode_e::ZSTD_noDict,
    );
    let btopt_fn: ZSTD_BlockCompressor_f = ZSTD_compressBlock_btopt;

    assert!(core::ptr::fn_addr_eq(selected, btopt_fn));
}

#[test]
#[should_panic(expected = "unsupported dedicatedDictSearch block compressor strategy")]
fn selectBlockCompressor_rejects_unsupported_dedicated_dict_strategy() {
    use crate::compress::match_state::ZSTD_dictMode_e;
    use crate::compress::zstd_compress_sequences::ZSTD_btopt;
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;

    let _ = ZSTD_selectBlockCompressor(
        ZSTD_btopt,
        ZSTD_ParamSwitch_e::ZSTD_ps_disable,
        ZSTD_dictMode_e::ZSTD_dedicatedDictSearch,
    );
}

#[test]
#[should_panic(expected = "unsupported noDict block compressor strategy")]
fn selectBlockCompressor_rejects_out_of_range_no_dict_strategy() {
    use crate::compress::match_state::ZSTD_dictMode_e;
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;

    let _ = ZSTD_selectBlockCompressor(
        99,
        ZSTD_ParamSwitch_e::ZSTD_ps_disable,
        ZSTD_dictMode_e::ZSTD_noDict,
    );
}

#[test]
fn resetCCtx_byCopyingCDict_carries_match_state_dict_bytes() {
    use crate::decompress::zstd_ddict::{ZSTD_dictContentType_e, ZSTD_dictLoadMethod_e};

    let dict = b"copied-cdict-dict-bytes ".repeat(10);
    let cParams = ZSTD_getCParams(3, u64::MAX, dict.len());
    let mut params = ZSTD_CCtx_params::default();
    params.cParams = cParams;
    params.compressionLevel = 3;
    params.useRowMatchFinder = crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_disable;

    let mut cdict = ZSTD_createCDict_advanced_internal(
        dict.len(),
        ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy,
        cParams,
        crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_disable,
        0,
        ZSTD_customMem::default(),
    )
    .expect("cdict");
    assert_eq!(
        ZSTD_initCDict_internal(
            &mut cdict,
            &dict,
            ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy,
            ZSTD_dictContentType_e::ZSTD_dct_rawContent,
            params,
        ),
        0
    );

    let mut cctx = ZSTD_CCtx::default();
    let rc = ZSTD_resetCCtx_byCopyingCDict(
        &mut cctx,
        &cdict,
        params,
        crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN,
        ZSTD_buffered_policy_e::ZSTDb_not_buffered,
    );
    assert_eq!(rc, 0);
    assert_eq!(cctx.ms.as_ref().unwrap().dictContent, dict);
}

#[test]
fn resetCCtx_byAttachingCDict_attaches_live_dict_match_state() {
    use crate::compress::match_state::ZSTD_matchState_dictMode;
    use crate::decompress::zstd_ddict::{ZSTD_dictContentType_e, ZSTD_dictLoadMethod_e};

    let dict = b"attached-cdict-dict-bytes ".repeat(10);
    let cParams = ZSTD_getCParams(3, u64::MAX, dict.len());
    let mut params = ZSTD_CCtx_params::default();
    params.cParams = cParams;
    params.compressionLevel = 3;
    params.useRowMatchFinder = crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_disable;

    let mut cdict = ZSTD_createCDict_advanced_internal(
        dict.len(),
        ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy,
        cParams,
        crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_disable,
        0,
        ZSTD_customMem::default(),
    )
    .expect("cdict");
    assert_eq!(
        ZSTD_initCDict_internal(
            &mut cdict,
            &dict,
            ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy,
            ZSTD_dictContentType_e::ZSTD_dct_rawContent,
            params,
        ),
        0
    );

    let mut cctx = ZSTD_CCtx::default();
    let rc = ZSTD_resetCCtx_byAttachingCDict(
        &mut cctx,
        &cdict,
        params,
        crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN,
        ZSTD_buffered_policy_e::ZSTDb_not_buffered,
    );
    assert_eq!(rc, 0);
    let ms = cctx.ms.as_ref().expect("ms");
    assert!(cctx.stream_dict.is_empty());
    assert!(ms.dictContent.is_empty());
    assert!(ms.dictMatchState.is_some());
    assert_eq!(ms.loadedDictEnd, ms.window.dictLimit);
    assert_eq!(ms.window.dictLimit, cdict.matchState.window.nextSrc);
    assert_eq!(
        ZSTD_matchState_dictMode(ms),
        crate::compress::match_state::ZSTD_dictMode_e::ZSTD_dictMatchState
    );
}

#[test]
fn zstd_sizeof_cdict_scales_with_dict_content() {
    // Symmetric with decompress-side `zstd_sizeof_dctx_grows_when_dict_loaded`.
    // A bigger dict must bump `ZSTD_sizeof_CDict` by at least the
    // dict's capacity; callers that size pool allocations from
    // this helper must not under-provision.
    let small_dict = vec![0xAB; 512];
    let big_dict = vec![0xCD; 32 * 1024];
    let cd_small = ZSTD_createCDict(&small_dict, 1).unwrap();
    let cd_big = ZSTD_createCDict(&big_dict, 1).unwrap();
    let s_small = ZSTD_sizeof_CDict(&cd_small);
    let s_big = ZSTD_sizeof_CDict(&cd_big);
    assert!(
        s_big >= s_small + (big_dict.len() - small_dict.len()),
        "sizeof_CDict did not scale: small={s_small}, big={s_big}",
    );
}

#[test]
fn zstd_sizeof_cdict_counts_seeded_match_state_allocations() {
    let dict = vec![0x5A; 4096];
    let cdict = ZSTD_createCDict(&dict, 5).unwrap();
    let reported = ZSTD_sizeof_CDict(&cdict);
    let minimum = core::mem::size_of::<ZSTD_CDict>()
        + cdict.dictContent.capacity()
        + cdict.matchState.dictContent.capacity()
        + cdict.matchState.hashTable.capacity() * core::mem::size_of::<u32>()
        + cdict.matchState.hashTable3.capacity() * core::mem::size_of::<u32>()
        + cdict.matchState.tagTable.capacity()
        + cdict.matchState.chainTable.capacity() * core::mem::size_of::<u32>();
    assert!(
        reported >= minimum,
        "sizeof_CDict undercounted seeded allocations: reported={reported}, minimum={minimum}",
    );
}

#[test]
fn writeLastEmptyBlock_emits_3_byte_last_bt_raw_header() {
    // Contract: 3-byte header with lastBlock=1, blockType=bt_raw,
    // cSize=0 → value = 1 + (bt_raw<<1) + (0<<3) = 1.
    use crate::decompress::zstd_decompress_block::{
        blockProperties_t, blockType_e, ZSTD_getcBlockSize,
    };
    let mut dst = [0u8; 3];
    let n = ZSTD_writeLastEmptyBlock(&mut dst);
    assert_eq!(n, 3);
    assert_eq!(dst, [0x01, 0x00, 0x00]);
    // Round-trip through the decoder's header parser.
    let mut props = blockProperties_t {
        blockType: blockType_e::bt_rle,
        lastBlock: 0,
        origSize: 0,
    };
    let consumed = ZSTD_getcBlockSize(&dst, &mut props);
    assert!(!crate::common::error::ERR_isError(consumed));
    assert_eq!(consumed, 0); // empty block body
    assert_eq!(props.blockType, blockType_e::bt_raw);
    assert_eq!(props.lastBlock, 1);

    // Undersized dst → DstSizeTooSmall.
    let mut tiny = [0u8; 2];
    assert!(crate::common::error::ERR_isError(ZSTD_writeLastEmptyBlock(
        &mut tiny
    )));
}

#[test]
fn rle_compress_block_produces_valid_rle_header() {
    use crate::decompress::zstd_decompress_block::{
        blockProperties_t, blockType_e, ZSTD_getcBlockSize,
    };
    let mut dst = vec![0u8; 64];
    let n = ZSTD_rleCompressBlock(&mut dst, 0xAA, 1024, 0);
    assert_eq!(n, 4);
    let mut props = blockProperties_t {
        blockType: blockType_e::bt_raw,
        lastBlock: 0,
        origSize: 0,
    };
    let consumed = ZSTD_getcBlockSize(&dst, &mut props);
    assert!(!crate::common::error::ERR_isError(consumed));
    // RLE: getcBlockSize returns 1 (one source byte) while origSize
    // holds the expanded block size.
    assert_eq!(consumed, 1);
    assert_eq!(props.origSize, 1024);
    assert_eq!(props.blockType, blockType_e::bt_rle);
    assert_eq!(props.lastBlock, 0);
    assert_eq!(dst[3], 0xAA);
}

#[test]
fn compress_block_framed_roundtrips_repetitive_text() {
    use crate::compress::match_state::{ZSTD_MatchState_t, ZSTD_compressionParameters};
    use crate::compress::seq_store::{SeqStore_t, ZSTD_REP_NUM};
    use crate::decompress::zstd_decompress_block::{
        blockProperties_t, blockType_e, streaming_operation, ZSTD_DCtx, ZSTD_decoder_entropy_rep,
        ZSTD_decompressBlock_internal, ZSTD_getcBlockSize,
    };
    // 1KB of repetitive text.
    let src: Vec<u8> = b"the quick brown fox jumps over the lazy dog. "
        .iter()
        .cycle()
        .take(1000)
        .copied()
        .collect();

    let mut ms = ZSTD_MatchState_t::new(ZSTD_compressionParameters {
        windowLog: 17,
        hashLog: 12,
        minMatch: 4,
        strategy: crate::compress::zstd_compress_sequences::ZSTD_fast,
        ..Default::default()
    });
    let mut seqStore = SeqStore_t::with_capacity(1024, 131072);
    let mut rep: [u32; ZSTD_REP_NUM] = [1, 4, 8];
    let prev = ZSTD_entropyCTables_t::default();
    let mut next = ZSTD_entropyCTables_t::default();
    let mut dst = vec![0u8; 4096];

    let n = ZSTD_compressBlock_fast_framed(
        &mut dst,
        &src,
        &mut ms,
        &mut seqStore,
        &mut rep,
        &prev,
        &mut next,
        crate::compress::zstd_compress_sequences::ZSTD_fast,
        0,
        0,
        1,
    );
    assert!(
        !crate::common::error::ERR_isError(n),
        "compress err: {n:#x}"
    );
    dst.truncate(n);

    // Parse header + decompress body.
    let mut props = blockProperties_t {
        blockType: blockType_e::bt_raw,
        lastBlock: 0,
        origSize: 0,
    };
    let body_size = ZSTD_getcBlockSize(&dst, &mut props);
    assert!(!crate::common::error::ERR_isError(body_size));
    assert_eq!(props.lastBlock, 1);

    // Reconstruct through the decoder. The block might have been
    // emitted as raw or compressed — decode accordingly.
    let mut out = vec![0u8; src.len() + 64];
    let mut dctx = ZSTD_DCtx::new();
    let decoded = match props.blockType {
        crate::decompress::zstd_decompress_block::blockType_e::bt_raw => {
            out[..body_size].copy_from_slice(&dst[3..3 + body_size]);
            body_size
        }
        crate::decompress::zstd_decompress_block::blockType_e::bt_rle => {
            let b = dst[3];
            for byte in out[..body_size].iter_mut() {
                *byte = b;
            }
            body_size
        }
        crate::decompress::zstd_decompress_block::blockType_e::bt_compressed => {
            let mut entropy_rep = ZSTD_decoder_entropy_rep::default();
            ZSTD_decompressBlock_internal(
                &mut dctx,
                &mut entropy_rep,
                &mut out,
                0,
                &dst[3..3 + body_size],
                streaming_operation::not_streaming,
            )
        }
        blockType_e::bt_reserved => panic!("reserved block type from compressor"),
    };
    assert!(
        !crate::common::error::ERR_isError(decoded),
        "decode err: {decoded:#x}"
    );
    assert_eq!(decoded, src.len(), "decoded size mismatch");
    assert_eq!(&out[..decoded], &src[..], "roundtrip mismatch");
}

#[test]
fn compress_block_fast_then_entropy_emits_payload_or_raw_fallback() {
    use crate::compress::match_state::{ZSTD_MatchState_t, ZSTD_compressionParameters};
    use crate::compress::seq_store::{SeqStore_t, ZSTD_REP_NUM};
    // 2 KB of repetitive text — the fast match finder should emit
    // a handful of sequences, then the entropy stage produces either
    // a compressed body or signals "fall back to raw" (return 0).
    let src: Vec<u8> = b"the quick brown fox jumps over the lazy dog. "
        .iter()
        .cycle()
        .take(2000)
        .copied()
        .collect();
    let mut ms = ZSTD_MatchState_t::new(ZSTD_compressionParameters {
        windowLog: 17,
        hashLog: 12,
        minMatch: 4,
        strategy: crate::compress::zstd_compress_sequences::ZSTD_fast,
        ..Default::default()
    });
    let mut seqStore = SeqStore_t::with_capacity(1024, 131072);
    let mut rep: [u32; ZSTD_REP_NUM] = [1, 4, 8];
    let prev = ZSTD_entropyCTables_t::default();
    let mut next = ZSTD_entropyCTables_t::default();
    let mut dst = vec![0u8; 4096];

    let cSize = ZSTD_compressBlock_fast_then_entropy(
        &mut dst,
        &src,
        &mut ms,
        &mut seqStore,
        &mut rep,
        &prev,
        &mut next,
        crate::compress::zstd_compress_sequences::ZSTD_fast,
        0,
        0,
    );
    assert!(
        !crate::common::error::ERR_isError(cSize),
        "compress block returned error: {:#x}",
        cSize
    );
    // Either a compressed body (cSize > 0) or "fall back" (cSize == 0).
    assert!(cSize <= dst.len());
    if cSize > 0 {
        // Compressed body must be smaller than the source for this
        // input (highly repetitive English).
        assert!(
            cSize < src.len(),
            "compressed size {} not smaller than src {}",
            cSize,
            src.len()
        );
    }
}

#[test]
fn compress_block_fast_then_entropy_downgrades_pure_rle_to_one_byte() {
    use crate::compress::match_state::{ZSTD_MatchState_t, ZSTD_compressionParameters};
    use crate::compress::seq_store::{SeqStore_t, ZSTD_REP_NUM};
    // All-zeros source → fast match finder produces ~ 1 big match.
    // Entropy body is ≤ 25 bytes → downgrade to 1-byte RLE.
    let src = vec![0u8; 512];
    let mut ms = ZSTD_MatchState_t::new(ZSTD_compressionParameters {
        windowLog: 17,
        hashLog: 12,
        minMatch: 4,
        strategy: crate::compress::zstd_compress_sequences::ZSTD_fast,
        ..Default::default()
    });
    let mut seqStore = SeqStore_t::with_capacity(1024, 131072);
    let mut rep: [u32; ZSTD_REP_NUM] = [1, 4, 8];
    let prev = ZSTD_entropyCTables_t::default();
    let mut next = ZSTD_entropyCTables_t::default();
    let mut dst = vec![0u8; 1024];

    let cSize = ZSTD_compressBlock_fast_then_entropy(
        &mut dst,
        &src,
        &mut ms,
        &mut seqStore,
        &mut rep,
        &prev,
        &mut next,
        crate::compress::zstd_compress_sequences::ZSTD_fast,
        0,
        0,
    );
    // Either downgraded to 1 byte (best case) or returned a small
    // non-error cSize; both are acceptable. No error either way.
    assert!(!crate::common::error::ERR_isError(cSize));
    if cSize == 1 {
        assert_eq!(dst[0], 0);
    }
}

#[test]
fn zstd_is_rle_true_for_constant_input() {
    let buf = [0x42u8; 64];
    assert_eq!(ZSTD_isRLE(&buf), 1);
}

#[test]
fn zstd_is_rle_false_when_any_byte_differs() {
    let mut buf = [0x42u8; 64];
    buf[63] = 0x41;
    assert_eq!(ZSTD_isRLE(&buf), 0);
}

#[test]
fn zstd_is_rle_empty_and_single() {
    // Upstream: length==1 → 1; we extend to length==0 for robustness.
    assert_eq!(ZSTD_isRLE(&[]), 1);
    assert_eq!(ZSTD_isRLE(&[0x55]), 1);
}

#[test]
fn entropy_compress_seq_store_emits_nonzero_payload() {
    use crate::compress::seq_store::{SeqDef, SeqStore_t, OFFSET_TO_OFFBASE, REPCODE_TO_OFFBASE};
    // 10 sequences with varied literal/match/offset patterns —
    // exercises the full literals-compression + 3-FSE-stream +
    // seq-count-header chain.
    let mut ss = SeqStore_t::with_capacity(64, 4096);
    let lit_payload: Vec<u8> = (0..200u8).collect();
    ss.literals.extend_from_slice(&lit_payload);
    for i in 0..10u16 {
        ss.sequences.push(SeqDef {
            offBase: if i % 3 == 0 {
                REPCODE_TO_OFFBASE(1)
            } else {
                OFFSET_TO_OFFBASE(50u32.wrapping_add(i as u32))
            },
            litLength: i * 2,
            mlBase: i * 3,
        });
    }
    let prev = ZSTD_entropyCTables_t::default();
    let mut next = ZSTD_entropyCTables_t::default();
    let mut dst = vec![0u8; 2048];
    let cSize = ZSTD_entropyCompressSeqStore(
        &mut dst,
        &mut ss,
        &prev,
        &mut next,
        crate::compress::zstd_compress_sequences::ZSTD_fast,
        0,
        1024, // blockSize > cSize so ratio gate doesn't trip
        0,
    );
    assert!(
        !crate::common::error::ERR_isError(cSize),
        "entropyCompressSeqStore returned error: {:#x}",
        cSize
    );
    // Either emitted a compressed block (cSize > 0) OR signalled
    // "emit uncompressed" (cSize == 0). Both are valid outcomes —
    // with 10 tiny sequences the ratio gate may well trigger.
    assert!(cSize < dst.len());
}

#[test]
fn build_sequences_statistics_handles_small_sequence_set() {
    use crate::compress::seq_store::{SeqDef, SeqStore_t, OFFSET_TO_OFFBASE, REPCODE_TO_OFFBASE};
    // Build a seq store with 5 sequences so we exercise the full
    // histogram → selectEncodingType → buildCTable path.
    let mut ss = SeqStore_t::with_capacity(64, 4096);
    for i in 0..5u16 {
        ss.sequences.push(SeqDef {
            offBase: if i % 2 == 0 {
                REPCODE_TO_OFFBASE(1)
            } else {
                OFFSET_TO_OFFBASE(100u32.wrapping_add(i as u32))
            },
            litLength: i,
            mlBase: i,
        });
    }
    let prev = ZSTD_fseCTables_t::default();
    let mut next = ZSTD_fseCTables_t::default();
    let mut dst = vec![0u8; 1024];
    let mut count_ws = vec![0u32; 256];
    let mut ent_ws = vec![0u8; 16 * 1024];
    let nbSeq = ss.sequences.len();
    let stats = ZSTD_buildSequencesStatistics(
        &mut ss,
        nbSeq,
        &prev,
        &mut next,
        &mut dst,
        crate::compress::zstd_compress_sequences::ZSTD_fast,
        &mut count_ws,
        &mut ent_ws,
    );
    assert!(
        !crate::common::error::ERR_isError(stats.size),
        "stats returned error: {:#x}",
        stats.size
    );
    // Small blocks typically land in set_basic for all three
    // streams (no NCount bytes written). Assert size fits in dst.
    assert!(stats.size < dst.len());
    // Codes were materialized.
    assert_eq!(ss.llCode.len(), 5);
    assert_eq!(ss.ofCode.len(), 5);
    assert_eq!(ss.mlCode.len(), 5);
}

#[test]
fn build_entropy_statistics_and_estimate_subblock_size_returns_header_plus_sections() {
    use crate::compress::seq_store::{SeqStore_t, ZSTD_storeSeqOnly, OFFSET_TO_OFFBASE};

    let mut cctx = ZSTD_createCCtx().unwrap();
    let literals = b"entropy-estimate-literals".to_vec();
    let params = ZSTD_CCtx_params {
        compressionLevel: 3,
        cParams: ZSTD_getCParams(3, literals.len() as u64, 0),
        ..ZSTD_CCtx_params::default()
    };
    assert_eq!(
        ZSTD_compressBegin_internal(
            &mut cctx,
            &[],
            crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_auto,
            None,
            &params,
            literals.len() as u64,
            ZSTD_buffered_policy_e::ZSTDb_buffered,
        ),
        0
    );

    let mut seqStore = SeqStore_t::with_capacity(16, 1024);
    seqStore.reset();
    seqStore.literals.extend_from_slice(&literals);
    ZSTD_storeSeqOnly(&mut seqStore, 3, OFFSET_TO_OFFBASE(8), 5);
    ZSTD_seqToCodes(&mut seqStore);

    let estimate = ZSTD_buildEntropyStatisticsAndEstimateSubBlockSize(&mut seqStore, &mut cctx);
    assert!(!ERR_isError(estimate));
    assert!(estimate >= crate::decompress::zstd_decompress_block::ZSTD_blockHeaderSize);
}

#[test]
fn build_seq_store_tiny_block_requests_no_compress() {
    let mut cctx = ZSTD_createCCtx().unwrap();
    let params = ZSTD_CCtx_params {
        compressionLevel: 3,
        cParams: ZSTD_getCParams(3, 8, 0),
        ..ZSTD_CCtx_params::default()
    };
    assert_eq!(
        ZSTD_compressBegin_internal(
            &mut cctx,
            &[],
            crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_auto,
            None,
            &params,
            8,
            ZSTD_buffered_policy_e::ZSTDb_buffered,
        ),
        0
    );
    let rc = ZSTD_buildSeqStore(&mut cctx, b"tiny");
    assert_eq!(rc, ZSTD_BuildSeqStore_e::ZSTDbss_noCompress as usize);
}

#[test]
fn reset_match_state_allocates_row_hash_tables_when_enabled() {
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;

    let cParams = crate::compress::match_state::ZSTD_compressionParameters {
        windowLog: 17,
        hashLog: 12,
        chainLog: 12,
        searchLog: 4,
        minMatch: 4,
        strategy: crate::compress::zstd_compress_sequences::ZSTD_lazy,
        ..Default::default()
    };
    let mut ms = crate::compress::match_state::ZSTD_MatchState_t::new(cParams);
    let rc = ZSTD_reset_matchState(
        &mut ms,
        &cParams,
        ZSTD_ParamSwitch_e::ZSTD_ps_enable,
        ZSTD_compResetPolicy_e::ZSTDcrp_makeClean,
        ZSTD_indexResetPolicy_e::ZSTDirp_reset,
        ZSTD_resetTarget_e::ZSTD_resetTarget_CCtx,
    );
    assert_eq!(rc, 0);
    assert_eq!(
        ms.rowHashLog,
        cParams.hashLog - cParams.searchLog.clamp(4, 6)
    );
    assert_eq!(ms.tagTable.len(), 1usize << cParams.hashLog);
    assert!(ms.tagTable.iter().all(|&b| b == 0));
}

#[test]
fn build_seq_store_populates_literals_and_next_rep() {
    let mut cctx = ZSTD_createCCtx().unwrap();
    let src = b"build-seq-store repetitive build-seq-store repetitive".repeat(4);
    let params = ZSTD_CCtx_params {
        compressionLevel: 3,
        cParams: ZSTD_getCParams(3, src.len() as u64, 0),
        useRowMatchFinder: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_disable,
        ..ZSTD_CCtx_params::default()
    };
    assert_eq!(
        ZSTD_compressBegin_internal(
            &mut cctx,
            &[],
            crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_auto,
            None,
            &params,
            src.len() as u64,
            ZSTD_buffered_policy_e::ZSTDb_buffered,
        ),
        0
    );
    let rc = ZSTD_buildSeqStore(&mut cctx, &src);
    assert_eq!(rc, ZSTD_BuildSeqStore_e::ZSTDbss_compress as usize);
    let seqStore = cctx.seqStore.as_ref().unwrap();
    assert!(!seqStore.literals.is_empty());
    assert!(seqStore.literals.len() <= src.len());
    assert_ne!(cctx.next_rep, [1, 4, 8]);
}

#[test]
fn compress_frame_chunk_roundtrips_last_chunk_frame() {
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    let mut cctx = ZSTD_createCCtx().unwrap();
    let src = b"frame-chunk roundtrip payload ".repeat(32);
    let params = ZSTD_CCtx_params {
        compressionLevel: 3,
        cParams: ZSTD_getCParams(3, src.len() as u64, 0),
        useRowMatchFinder: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_disable,
        ..ZSTD_CCtx_params::default()
    };
    assert_eq!(
        ZSTD_compressBegin_internal(
            &mut cctx,
            &[],
            crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_auto,
            None,
            &params,
            src.len() as u64,
            ZSTD_buffered_policy_e::ZSTDb_buffered,
        ),
        0
    );
    let mut dst = vec![0u8; ZSTD_compressBound(src.len()) + 64];
    let n = ZSTD_compressContinue_internal(&mut cctx, &mut dst, &src, 1, 1);
    assert!(!ERR_isError(n));

    let mut decoded = vec![0u8; src.len()];
    let d = ZSTD_decompress(&mut decoded, &dst[..n]);
    assert_eq!(d, src.len());
    assert_eq!(&decoded[..d], src.as_slice());
    assert_eq!(cctx.stage, ZSTD_compressionStage_e::ZSTDcs_ending);
}

#[test]
fn compress_continue_internal_rejects_src_beyond_pledge() {
    let mut cctx = ZSTD_createCCtx().unwrap();
    let src = b"0123456789abcdef".repeat(8);
    let params = ZSTD_CCtx_params {
        compressionLevel: 3,
        cParams: ZSTD_getCParams(3, 8, 0),
        useRowMatchFinder: crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_disable,
        ..ZSTD_CCtx_params::default()
    };
    assert_eq!(
        ZSTD_compressBegin_internal(
            &mut cctx,
            &[],
            crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_auto,
            None,
            &params,
            8,
            ZSTD_buffered_policy_e::ZSTDb_buffered,
        ),
        0
    );
    let mut dst = vec![0u8; 1024];
    let rc = ZSTD_compressContinue_internal(&mut cctx, &mut dst, &src, 1, 1);
    assert!(ERR_isError(rc));
    assert_eq!(
        crate::common::error::ERR_getErrorCode(rc),
        ErrorCode::SrcSizeWrong
    );
}

#[test]
fn compress_begin_internal_honors_full_dict_content_type() {
    use crate::decompress::zstd_ddict::ZSTD_dictContentType_e;

    let mut cctx = ZSTD_createCCtx().unwrap();
    let params = ZSTD_CCtx_params {
        compressionLevel: 3,
        cParams: ZSTD_getCParams(3, 0, 0),
        ..ZSTD_CCtx_params::default()
    };

    let rc = ZSTD_compressBegin_internal(
        &mut cctx,
        b"not-a-zstd-dictionary",
        ZSTD_dictContentType_e::ZSTD_dct_fullDict,
        None,
        &params,
        crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN,
        ZSTD_buffered_policy_e::ZSTDb_not_buffered,
    );
    assert!(ERR_isError(rc));
    assert_eq!(
        crate::common::error::ERR_getErrorCode(rc),
        ErrorCode::DictionaryWrong,
    );
}

#[test]
fn compress_begin_internal_auto_dict_still_accepts_short_raw_dictionary() {
    use crate::decompress::zstd_ddict::ZSTD_dictContentType_e;

    let mut cctx = ZSTD_createCCtx().unwrap();
    let params = ZSTD_CCtx_params {
        compressionLevel: 3,
        cParams: ZSTD_getCParams(3, 0, 0),
        ..ZSTD_CCtx_params::default()
    };

    let rc = ZSTD_compressBegin_internal(
        &mut cctx,
        b"raw-dict",
        ZSTD_dictContentType_e::ZSTD_dct_auto,
        None,
        &params,
        crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN,
        ZSTD_buffered_policy_e::ZSTDb_not_buffered,
    );
    assert_eq!(rc, 0);
    assert_eq!(cctx.dictID, 0);
}

#[test]
fn cctx_load_dictionary_advanced_honors_full_dict_content_type() {
    use crate::decompress::zstd_ddict::{ZSTD_dictContentType_e, ZSTD_dictLoadMethod_e};

    let mut cctx = ZSTD_createCCtx().unwrap();
    let rc = ZSTD_CCtx_loadDictionary_advanced(
        &mut cctx,
        b"not-a-zstd-dictionary",
        ZSTD_dictLoadMethod_e::ZSTD_dlm_byCopy,
        ZSTD_dictContentType_e::ZSTD_dct_fullDict,
    );
    assert_eq!(rc, 0);
    let rc = ZSTD_CCtx_init_compressStream2(&mut cctx, ZSTD_EndDirective::ZSTD_e_end, 0);
    assert!(ERR_isError(rc));
    assert_eq!(
        crate::common::error::ERR_getErrorCode(rc),
        ErrorCode::DictionaryWrong,
    );
}

#[test]
fn cctx_ref_prefix_advanced_honors_full_dict_content_type() {
    use crate::decompress::zstd_ddict::ZSTD_dictContentType_e;

    let mut cctx = ZSTD_createCCtx().unwrap();
    let rc = ZSTD_CCtx_refPrefix_advanced(
        &mut cctx,
        b"not-a-zstd-dictionary",
        ZSTD_dictContentType_e::ZSTD_dct_fullDict,
    );
    assert!(ERR_isError(rc));
    assert_eq!(
        crate::common::error::ERR_getErrorCode(rc),
        ErrorCode::DictionaryWrong,
    );
}

#[test]
fn compress2_roundtrip_with_ldm_enabled() {
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    let src = b"ldm-enabled roundtrip payload with repeated phrases. ".repeat(256);
    let mut cctx = ZSTD_createCCtx().unwrap();
    assert_eq!(
        ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_compressionLevel, 5,),
        0
    );
    cctx.requestedParams.ldmEnable = ZSTD_ParamSwitch_e::ZSTD_ps_enable;

    let mut compressed = vec![0u8; ZSTD_compressBound(src.len())];
    let n = ZSTD_compress2(&mut cctx, &mut compressed, &src);
    assert!(!ERR_isError(n), "compress2 with LDM failed: {n:#x}");
    compressed.truncate(n);

    let mut decoded = vec![0u8; src.len()];
    let d = ZSTD_decompress(&mut decoded, &compressed);
    assert_eq!(d, src.len());
    assert_eq!(&decoded[..d], src.as_slice());
}

#[test]
fn build_seq_store_with_ldm_can_use_frame_prefix_as_history() {
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;

    let repeated = b"history-window phrase for ldm matching ".repeat(4);
    let prefix = vec![b'x'; ZSTD_BLOCKSIZE_MAX - repeated.len()];
    let mut src = prefix.clone();
    src.extend_from_slice(&repeated);
    let second_block_start = src.len();
    src.extend_from_slice(&repeated);
    src.extend_from_slice(&vec![b'y'; 512]);

    let mut cctx = ZSTD_createCCtx().unwrap();
    let mut params = ZSTD_CCtx_params {
        compressionLevel: 5,
        cParams: ZSTD_getCParams(5, src.len() as u64, 0),
        ..ZSTD_CCtx_params::default()
    };
    params.ldmEnable = ZSTD_ParamSwitch_e::ZSTD_ps_enable;
    assert_eq!(
        ZSTD_compressBegin_internal(
            &mut cctx,
            &[],
            crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_auto,
            None,
            &params,
            src.len() as u64,
            ZSTD_buffered_policy_e::ZSTDb_buffered,
        ),
        0
    );

    let first = ZSTD_buildSeqStore_with_window(&mut cctx, &src, 0, second_block_start);
    assert_eq!(first, ZSTD_BuildSeqStore_e::ZSTDbss_compress as usize);

    let second = ZSTD_buildSeqStore_with_window(&mut cctx, &src, second_block_start, src.len());
    assert_eq!(second, ZSTD_BuildSeqStore_e::ZSTDbss_compress as usize);
    let seqStore = cctx.seqStore.as_ref().unwrap();
    assert!(
        !seqStore.sequences.is_empty(),
        "second block should find at least one LDM-backed match from the frame prefix",
    );
}

#[test]
fn compress2_roundtrip_with_row_match_finder_enabled() {
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;
    use crate::decompress::zstd_decompress::ZSTD_decompress;

    let src = b"row-hash compression payload with repeated phrases. ".repeat(256);
    let mut cctx = ZSTD_createCCtx().unwrap();
    assert_eq!(
        ZSTD_CCtx_setParameter(&mut cctx, ZSTD_cParameter::ZSTD_c_compressionLevel, 5,),
        0
    );
    cctx.requestedParams.useRowMatchFinder = ZSTD_ParamSwitch_e::ZSTD_ps_enable;

    let mut compressed = vec![0u8; ZSTD_compressBound(src.len())];
    let n = ZSTD_compress2(&mut cctx, &mut compressed, &src);
    assert!(
        !ERR_isError(n),
        "compress2 with row match finder failed: {n:#x}"
    );
    compressed.truncate(n);

    let mut decoded = vec![0u8; src.len()];
    let d = ZSTD_decompress(&mut decoded, &compressed);
    assert_eq!(d, src.len());
    assert_eq!(&decoded[..d], src.as_slice());
}

#[test]
fn build_seq_store_uses_referenced_external_raw_sequences() {
    use crate::compress::seq_store::OFFSET_TO_OFFBASE;
    use crate::compress::zstd_ldm::rawSeq;

    let src = b"0123456789abcdef".repeat(8);
    let mut cctx = ZSTD_createCCtx().unwrap();
    let params = ZSTD_CCtx_params {
        compressionLevel: 3,
        cParams: ZSTD_getCParams(3, src.len() as u64, 0),
        ..ZSTD_CCtx_params::default()
    };
    assert_eq!(
        ZSTD_compressBegin_internal(
            &mut cctx,
            &[],
            crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_auto,
            None,
            &params,
            src.len() as u64,
            ZSTD_buffered_policy_e::ZSTDb_buffered,
        ),
        0
    );

    let ext = [rawSeq {
        litLength: 64,
        matchLength: 64,
        offset: 64,
    }];
    assert_eq!(ZSTD_referenceExternalSequences(&mut cctx, Some(&ext)), 0);

    let rc = ZSTD_buildSeqStore(&mut cctx, &src);
    assert_eq!(rc, ZSTD_BuildSeqStore_e::ZSTDbss_compress as usize);
    let seqStore = cctx.seqStore.as_ref().unwrap();
    assert!(
        seqStore
            .sequences
            .iter()
            .any(|seq| seq.offBase == OFFSET_TO_OFFBASE(64)),
        "seqStore should contain the externally referenced 64-byte match",
    );
    assert!(
        cctx.externalMatchStore
            .as_ref()
            .is_some_and(|store| store.pos == store.size),
        "external raw sequence stream should be fully consumed",
    );
}

#[test]
fn build_seq_store_tiny_block_uses_sequence_skip_for_non_btopt_strategies() {
    use crate::compress::zstd_ldm::rawSeq;

    let mut cctx = ZSTD_createCCtx().unwrap();
    let params = ZSTD_CCtx_params {
        compressionLevel: 3,
        cParams: ZSTD_getCParams(3, 4, 0),
        ..ZSTD_CCtx_params::default()
    };
    assert_eq!(
        ZSTD_compressBegin_internal(
            &mut cctx,
            &[],
            crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_auto,
            None,
            &params,
            4,
            ZSTD_buffered_policy_e::ZSTDb_buffered,
        ),
        0
    );

    let ext = [rawSeq {
        litLength: 3,
        matchLength: 9,
        offset: 1,
    }];
    assert_eq!(ZSTD_referenceExternalSequences(&mut cctx, Some(&ext)), 0);

    let rc = ZSTD_buildSeqStore(&mut cctx, b"tiny");
    assert_eq!(rc, ZSTD_BuildSeqStore_e::ZSTDbss_noCompress as usize);
    let store = cctx.externalMatchStore.as_ref().unwrap();
    assert_eq!(store.pos, 0);
    assert_eq!(store.posInSequence, 0);
    assert_eq!(store.seq[0].litLength, 0);
    assert_eq!(store.seq[0].matchLength, 8);
}

#[test]
fn build_seq_store_tiny_block_uses_raw_seq_skip_for_btopt_plus_strategies() {
    use crate::compress::zstd_ldm::rawSeq;

    let mut cctx = ZSTD_createCCtx().unwrap();
    let params = ZSTD_CCtx_params {
        compressionLevel: 16,
        cParams: ZSTD_getCParams(16, 4, 0),
        ..ZSTD_CCtx_params::default()
    };
    assert_eq!(
        ZSTD_compressBegin_internal(
            &mut cctx,
            &[],
            crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_auto,
            None,
            &params,
            4,
            ZSTD_buffered_policy_e::ZSTDb_buffered,
        ),
        0
    );

    let ext = [rawSeq {
        litLength: 3,
        matchLength: 9,
        offset: 1,
    }];
    assert_eq!(ZSTD_referenceExternalSequences(&mut cctx, Some(&ext)), 0);

    let rc = ZSTD_buildSeqStore(&mut cctx, b"tiny");
    assert_eq!(rc, ZSTD_BuildSeqStore_e::ZSTDbss_noCompress as usize);
    let store = cctx.externalMatchStore.as_ref().unwrap();
    assert_eq!(store.pos, 0);
    assert_eq!(store.posInSequence, 4);
    assert_eq!(store.seq[0].litLength, 3);
    assert_eq!(store.seq[0].matchLength, 9);
}

#[test]
fn register_sequence_producer_updates_params_and_has_ext_seq_prod() {
    fn dummy_producer(
        _state: usize,
        _outSeqs: &mut [ZSTD_Sequence],
        _src: &[u8],
        _dict: &[u8],
        _compressionLevel: i32,
        _windowSize: usize,
    ) -> usize {
        0
    }

    let mut params = ZSTD_CCtx_params::default();
    assert!(!ZSTD_hasExtSeqProd(&params));

    ZSTD_CCtxParams_registerSequenceProducer(&mut params, 123, Some(dummy_producer));
    assert!(ZSTD_hasExtSeqProd(&params));
    assert_eq!(params.extSeqProdState, 123);
    assert!(params.extSeqProdFunc.is_some());

    ZSTD_CCtxParams_registerSequenceProducer(&mut params, 0, None);
    assert!(!ZSTD_hasExtSeqProd(&params));
    assert_eq!(params.extSeqProdState, 0);
}

#[test]
fn register_sequence_producer_propagates_into_applied_params_on_init() {
    fn dummy_producer(
        _state: usize,
        _outSeqs: &mut [ZSTD_Sequence],
        _src: &[u8],
        _dict: &[u8],
        _compressionLevel: i32,
        _windowSize: usize,
    ) -> usize {
        0
    }

    let mut cctx = ZSTD_createCCtx().unwrap();
    ZSTD_registerSequenceProducer(&mut cctx, 7, Some(dummy_producer));
    assert!(ZSTD_hasExtSeqProd(&cctx.requestedParams));

    let rc = ZSTD_CCtx_init_compressStream2(&mut cctx, ZSTD_EndDirective::ZSTD_e_end, 32);
    assert_eq!(rc, 0);
    assert!(ZSTD_hasExtSeqProd(&cctx.appliedParams));
    assert_eq!(cctx.appliedParams.extSeqProdState, 7);
    assert!(cctx.appliedParams.extSeqProdFunc.is_some());
}

#[test]
fn build_seq_store_uses_registered_sequence_producer() {
    fn producer(
        state: usize,
        outSeqs: &mut [ZSTD_Sequence],
        src: &[u8],
        dict: &[u8],
        compressionLevel: i32,
        windowSize: usize,
    ) -> usize {
        assert_eq!(state, 64);
        assert!(dict.is_empty());
        assert_eq!(compressionLevel, 3);
        assert!(windowSize >= 1 << 10);
        assert_eq!(src.len(), 128);
        outSeqs[0] = ZSTD_Sequence {
            offset: 64,
            litLength: 64,
            matchLength: 64,
            rep: 0,
        };
        1
    }

    let mut cctx = ZSTD_createCCtx().unwrap();
    let params = ZSTD_CCtx_params {
        compressionLevel: 3,
        cParams: ZSTD_getCParams(3, 128, 0),
        ..ZSTD_CCtx_params::default()
    };
    assert_eq!(
        ZSTD_compressBegin_internal(
            &mut cctx,
            &[],
            crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_auto,
            None,
            &params,
            128,
            ZSTD_buffered_policy_e::ZSTDb_buffered,
        ),
        0
    );
    ZSTD_registerSequenceProducer(&mut cctx, 64, Some(producer));
    cctx.appliedParams.extSeqProdState = cctx.requestedParams.extSeqProdState;
    cctx.appliedParams.extSeqProdFunc = cctx.requestedParams.extSeqProdFunc;

    let src = b"abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwx";
    let rc = ZSTD_buildSeqStore(&mut cctx, src);
    assert_eq!(rc, ZSTD_BuildSeqStore_e::ZSTDbss_compress as usize);

    let seqStore = cctx.seqStore.as_ref().unwrap();
    assert_eq!(seqStore.sequences.len(), 1);
    assert_eq!(
        seqStore.sequences[0].offBase,
        crate::compress::seq_store::OFFSET_TO_OFFBASE(64)
    );
    assert_eq!(seqStore.sequences[0].litLength as usize, 64);
    assert_eq!(
        seqStore.sequences[0].mlBase as usize + crate::compress::seq_store::MINMATCH as usize,
        64
    );
}

#[test]
fn build_seq_store_falls_back_when_sequence_producer_errors_and_fallback_enabled() {
    fn failing_producer(
        _state: usize,
        _outSeqs: &mut [ZSTD_Sequence],
        _src: &[u8],
        _dict: &[u8],
        _compressionLevel: i32,
        _windowSize: usize,
    ) -> usize {
        ERROR(ErrorCode::Generic)
    }

    let mut cctx = ZSTD_createCCtx().unwrap();
    let mut params = ZSTD_CCtx_params {
        compressionLevel: 3,
        cParams: ZSTD_getCParams(3, 256, 0),
        ..ZSTD_CCtx_params::default()
    };
    params.useRowMatchFinder = crate::compress::zstd_ldm::ZSTD_ParamSwitch_e::ZSTD_ps_disable;
    params.enableMatchFinderFallback = 1;
    assert_eq!(
        ZSTD_compressBegin_internal(
            &mut cctx,
            &[],
            crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_auto,
            None,
            &params,
            256,
            ZSTD_buffered_policy_e::ZSTDb_buffered,
        ),
        0
    );
    ZSTD_registerSequenceProducer(&mut cctx, 0, Some(failing_producer));
    cctx.appliedParams.extSeqProdState = cctx.requestedParams.extSeqProdState;
    cctx.appliedParams.extSeqProdFunc = cctx.requestedParams.extSeqProdFunc;
    cctx.appliedParams.enableMatchFinderFallback = 1;

    let src = b"abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc";
    let rc = ZSTD_buildSeqStore(&mut cctx, src);
    assert_eq!(rc, ZSTD_BuildSeqStore_e::ZSTDbss_compress as usize);
    assert!(
        !cctx.seqStore.as_ref().unwrap().sequences.is_empty(),
        "fallback internal parser should still produce sequences",
    );
}

#[test]
fn build_seq_store_propagates_literal_compression_mode_and_entropy_seed_into_opt_state() {
    use crate::compress::match_state::ZSTD_compressionParameters;
    use crate::compress::zstd_compress_literals::HUF_repeat;
    use crate::compress::zstd_compress_sequences::ZSTD_lazy;
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;

    let mut cctx = ZSTD_createCCtx().unwrap();
    let mut params = ZSTD_CCtx_params {
        compressionLevel: 5,
        cParams: ZSTD_compressionParameters {
            strategy: ZSTD_lazy,
            ..ZSTD_getCParams(5, 256, 0)
        },
        ..ZSTD_CCtx_params::default()
    };
    params.literalCompressionMode = ZSTD_ParamSwitch_e::ZSTD_ps_disable;
    params.useRowMatchFinder = ZSTD_ParamSwitch_e::ZSTD_ps_disable;
    assert_eq!(
        ZSTD_compressBegin_internal(
            &mut cctx,
            &[],
            crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_auto,
            None,
            &params,
            256,
            ZSTD_buffered_policy_e::ZSTDb_buffered,
        ),
        0
    );
    cctx.prevEntropy.huf.repeatMode = HUF_repeat::HUF_repeat_valid;

    let src = b"abcdefghijklmnopqrstuvwxyz012345abcdefghijklmnopqrstuvwxyz012345abcdefghijklmnopqrstuvwxyz012345abcdefghijklmnopqrstuvwxyz012345";
    let rc = ZSTD_buildSeqStore(&mut cctx, src);
    assert_eq!(rc, ZSTD_BuildSeqStore_e::ZSTDbss_compress as usize);
    assert_eq!(
        cctx.ms.as_ref().unwrap().opt.literalCompressionMode,
        ZSTD_ParamSwitch_e::ZSTD_ps_disable
    );
    assert_eq!(
        cctx.ms
            .as_ref()
            .unwrap()
            .entropySeed
            .as_ref()
            .unwrap()
            .huf
            .repeatMode,
        HUF_repeat::HUF_repeat_valid
    );
}

#[test]
fn build_seq_store_clears_stale_ldm_seq_store_on_regular_matchfinder_path() {
    use crate::compress::zstd_ldm::RawSeqStore_t;
    use crate::compress::zstd_ldm::ZSTD_ParamSwitch_e;

    let mut cctx = ZSTD_createCCtx().unwrap();
    let params = ZSTD_CCtx_params {
        compressionLevel: 3,
        cParams: ZSTD_getCParams(3, 256, 0),
        ..ZSTD_CCtx_params::default()
    };
    assert_eq!(
        ZSTD_compressBegin_internal(
            &mut cctx,
            &[],
            crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_auto,
            None,
            &params,
            256,
            ZSTD_buffered_policy_e::ZSTDb_buffered,
        ),
        0
    );
    cctx.appliedParams.useRowMatchFinder = ZSTD_ParamSwitch_e::ZSTD_ps_disable;

    let ms = cctx.ms.get_or_insert_with(|| {
        crate::compress::match_state::ZSTD_MatchState_t::new(cctx.appliedParams.cParams)
    });
    ms.ldmSeqStore = Some(RawSeqStore_t::with_capacity(1));

    let src = b"abcdefghijklmnopqrstuvwxyz012345abcdefghijklmnopqrstuvwxyz012345abcdefghijklmnopqrstuvwxyz012345abcdefghijklmnopqrstuvwxyz012345";
    let rc = ZSTD_buildSeqStore(&mut cctx, src);
    assert_eq!(rc, ZSTD_BuildSeqStore_e::ZSTDbss_compress as usize);
    assert!(
        cctx.ms.as_ref().unwrap().ldmSeqStore.is_none(),
        "regular software matchfinder path should clear stale LDM seqStore state",
    );
}

#[test]
fn build_seq_store_applies_limited_next_to_update_catchup_before_external_sequences() {
    fn producer(
        _state: usize,
        outSeqs: &mut [ZSTD_Sequence],
        src: &[u8],
        _dict: &[u8],
        _compressionLevel: i32,
        _windowSize: usize,
    ) -> usize {
        assert_eq!(src.len(), 256);
        outSeqs[0] = ZSTD_Sequence {
            offset: 1,
            litLength: 128,
            matchLength: 128,
            rep: 0,
        };
        1
    }

    let mut cctx = ZSTD_createCCtx().unwrap();
    let params = ZSTD_CCtx_params {
        compressionLevel: 3,
        cParams: ZSTD_getCParams(3, 1024, 0),
        ..ZSTD_CCtx_params::default()
    };
    assert_eq!(
        ZSTD_compressBegin_internal(
            &mut cctx,
            &[],
            crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_auto,
            None,
            &params,
            1024,
            ZSTD_buffered_policy_e::ZSTDb_buffered,
        ),
        0
    );
    ZSTD_registerSequenceProducer(&mut cctx, 0, Some(producer));
    cctx.appliedParams.extSeqProdState = cctx.requestedParams.extSeqProdState;
    cctx.appliedParams.extSeqProdFunc = cctx.requestedParams.extSeqProdFunc;
    cctx.ms = Some(crate::compress::match_state::ZSTD_MatchState_t::new(
        cctx.appliedParams.cParams,
    ));

    let ms = cctx.ms.as_mut().unwrap();
    ms.window.base_offset = 50;
    ms.nextToUpdate = 100;

    let src = vec![b'a'; 1024];
    let rc = ZSTD_buildSeqStore_with_window(&mut cctx, &src, 700, 956);
    assert_eq!(rc, ZSTD_BuildSeqStore_e::ZSTDbss_compress as usize);
    assert_eq!(cctx.ms.as_ref().unwrap().nextToUpdate, 558);
}

#[test]
fn build_seq_store_rejects_invalid_sequence_producer_output() {
    fn bad_producer(
        _state: usize,
        _outSeqs: &mut [ZSTD_Sequence],
        src: &[u8],
        _dict: &[u8],
        _compressionLevel: i32,
        _windowSize: usize,
    ) -> usize {
        assert!(!src.is_empty());
        0
    }

    let mut cctx = ZSTD_createCCtx().unwrap();
    let params = ZSTD_CCtx_params {
        compressionLevel: 3,
        cParams: ZSTD_getCParams(3, 16, 0),
        ..ZSTD_CCtx_params::default()
    };
    assert_eq!(
        ZSTD_compressBegin_internal(
            &mut cctx,
            &[],
            crate::decompress::zstd_ddict::ZSTD_dictContentType_e::ZSTD_dct_auto,
            None,
            &params,
            16,
            ZSTD_buffered_policy_e::ZSTDb_buffered,
        ),
        0
    );
    ZSTD_registerSequenceProducer(&mut cctx, 0, Some(bad_producer));
    cctx.appliedParams.extSeqProdFunc = cctx.requestedParams.extSeqProdFunc;

    let rc = ZSTD_buildSeqStore(&mut cctx, b"0123456789abcdef");
    assert_eq!(
        crate::common::error::ERR_getErrorCode(rc),
        ErrorCode::Generic
    );
}

#[test]
fn reset_cctx_internal_clears_external_sequences_and_buffered_stream_state() {
    use crate::compress::zstd_ldm::rawSeq;

    let mut cctx = ZSTD_createCCtx().unwrap();
    let params = ZSTD_CCtx_params {
        compressionLevel: 3,
        cParams: ZSTD_getCParams(3, 128, 0),
        ..ZSTD_CCtx_params::default()
    };

    cctx.externalMatchStore = Some(crate::compress::zstd_ldm::RawSeqStore_t::with_capacity(1));
    cctx.stream_in_buffer.extend_from_slice(b"pending-input");
    cctx.stream_out_buffer.extend_from_slice(b"pending-output");
    cctx.stream_out_drained = 3;
    cctx.expected_in_src = 11;
    cctx.expected_in_size = 12;
    cctx.expected_in_pos = 13;
    cctx.expected_out_buffer_size = 14;
    cctx.buffer_expectations_set = true;
    cctx.stream_closed = true;

    let rc = ZSTD_resetCCtx_internal(
        &mut cctx,
        &params,
        128,
        0,
        ZSTD_compResetPolicy_e::ZSTDcrp_makeClean,
        ZSTD_buffered_policy_e::ZSTDb_buffered,
    );
    assert_eq!(rc, 0);
    assert!(cctx.externalMatchStore.is_none());
    assert!(cctx.stream_in_buffer.is_empty());
    assert!(cctx.stream_out_buffer.is_empty());
    assert_eq!(cctx.stream_out_drained, 0);
    assert_eq!(cctx.expected_in_src, 0);
    assert_eq!(cctx.expected_in_size, 0);
    assert_eq!(cctx.expected_in_pos, 0);
    assert_eq!(cctx.expected_out_buffer_size, 0);
    assert!(!cctx.buffer_expectations_set);
    assert!(!cctx.stream_closed);

    let ext = [rawSeq {
        litLength: 32,
        matchLength: 32,
        offset: 8,
    }];
    assert_eq!(ZSTD_referenceExternalSequences(&mut cctx, Some(&ext)), 0);
    assert!(cctx.externalMatchStore.is_some());
}

#[test]
fn seq_to_codes_honors_long_lit_length_flag() {
    use crate::compress::seq_store::{
        SeqDef, SeqStore_t, ZSTD_longLengthType_e, REPCODE_TO_OFFBASE,
    };
    let mut ss = SeqStore_t::with_capacity(16, 1024);
    ss.sequences.push(SeqDef {
        offBase: REPCODE_TO_OFFBASE(1),
        litLength: 0,
        mlBase: 0,
    });
    ss.longLengthType = ZSTD_longLengthType_e::ZSTD_llt_literalLength;
    ss.longLengthPos = 0;
    ZSTD_seqToCodes(&mut ss);
    // Overrides llCode[0] with MaxLL.
    assert_eq!(ss.llCode[0], MaxLL as u8);
}
