//! Pure-Rust zstd dictionary trainer (fastcover variant).
//!
//! Ported from the blosc2-pure-rs project, which previously carried this code
//! while depending on this crate's primitives. It builds a zstd dictionary
//! (magic header + entropy tables + content) from a set of training samples,
//! matching the layout produced by upstream `ZDICT_trainFromBuffer`.

use crate::common::error::ERR_isError;
use crate::common::xxhash::XXH64;
use crate::compress::fse_compress::{FSE_normalizeCount, FSE_writeNCount};
use crate::compress::huf_compress::{
    HUF_buildCTable_wksp, HUF_writeCTable, HUF_CElt, HUF_CTABLE_WORKSPACE_SIZE_U32,
};
use crate::compress::zstd_compress::{
    ZSTD_compressBegin_usingCDict_deprecated, ZSTD_compressBlock_deprecated, ZSTD_compressBound,
    ZSTD_compress_usingCDict, ZSTD_createCCtx, ZSTD_createCDict, ZSTD_createCDict_advanced,
    ZSTD_getParams, ZSTD_seqToCodes,
};
use crate::decompress::zstd_ddict::{ZSTD_dictContentType_e, ZSTD_dictLoadMethod_e};
use crate::compress::zstd_hashes::ZSTD_hashPtr;
use crate::common::bits::ZSTD_highbit32;
use crate::decompress::zstd_decompress::ZSTD_MAGIC_DICTIONARY;
use crate::decompress::zstd_decompress_block::{
    LLFSELog, MLFSELog, MaxLL, MaxML, MaxOff, OffFSELog,
};

/// Train a zstd dictionary from `samples_buffer` (the concatenated samples,
/// each `sample_sizes[i]` bytes). `dict_capacity` is the maximum dictionary
/// size to produce; `min_useful_dict` is the smallest dictionary considered
/// worthwhile. Returns the serialized dictionary (starting with the zstd
/// dictionary magic) or `None` if no useful dictionary could be built.
pub fn train_from_buffer(
    samples_buffer: &[u8],
    sample_sizes: &[usize],
    dict_capacity: usize,
    min_useful_dict: usize,
) -> Option<Vec<u8>> {
    if dict_capacity < min_useful_dict || samples_buffer.is_empty() || sample_sizes.is_empty() {
        return None;
    }

    let samples_len = sample_sizes.iter().try_fold(0usize, |acc, &size| {
        if size == 0 {
            None
        } else {
            acc.checked_add(size)
        }
    })?;
    let target = dict_capacity.min(samples_len);
    if target < min_useful_dict || samples_buffer.len() < samples_len {
        return None;
    }

    let content = zstd_fastcover_content(samples_buffer, sample_sizes, target, min_useful_dict)?;

    let training_sample_count =
        zstd_fastcover_training_count(sample_sizes).unwrap_or(sample_sizes.len());
    let entropy_len = sample_sizes[..training_sample_count]
        .iter()
        .try_fold(0usize, |acc, &size| acc.checked_add(size))?;
    let entropy_samples = &samples_buffer[..entropy_len];
    let entropy_sample_sizes = &sample_sizes[..training_sample_count];
    finalize_zstd_fallback_dict(
        &content,
        entropy_samples,
        entropy_sample_sizes,
        dict_capacity,
        min_useful_dict,
    )
}

#[derive(Clone, Copy)]
struct FastCoverSegment {
    begin: usize,
    end: usize,
    score: u32,
}

fn zstd_fastcover_content(
    samples_buffer: &[u8],
    sample_sizes: &[usize],
    dict_capacity: usize,
    min_useful_dict: usize,
) -> Option<Vec<u8>> {
    const D: usize = 8;
    const DEFAULT_K_CANDIDATES: [usize; 5] = [50, 537, 1024, 1511, 1998];

    if dict_capacity < min_useful_dict || sample_sizes.len() < 5 {
        return None;
    }
    let training_sample_count = sample_sizes.len() * 3 / 4;
    if training_sample_count < 5 || training_sample_count >= sample_sizes.len() {
        return None;
    }
    let mut offsets = Vec::with_capacity(sample_sizes.len() + 1);
    offsets.push(0usize);
    for &size in sample_sizes {
        offsets.push(offsets.last()?.checked_add(size)?);
    }
    let total_size = offsets[training_sample_count];
    if *offsets.last()? > samples_buffer.len() || total_size < D {
        return None;
    }

    let nb_dmers = total_size.checked_sub(D)?.checked_add(1)?;
    let mut best = Vec::new();
    let mut best_score = usize::MAX;
    let training_offsets = &offsets[..=training_sample_count];
    let entropy_samples = &samples_buffer[..total_size];
    for k in DEFAULT_K_CANDIDATES
        .into_iter()
        .filter(|&k| k >= D && k <= dict_capacity)
    {
        let Some(candidate) = fastcover_build_dictionary(
            samples_buffer,
            training_offsets,
            nb_dmers,
            k,
            dict_capacity,
        ) else {
            continue;
        };
        let Some(score) = fastcover_candidate_score(
            samples_buffer,
            sample_sizes,
            &offsets,
            training_sample_count,
            &candidate,
            entropy_samples,
            &sample_sizes[..training_sample_count],
            dict_capacity,
            min_useful_dict,
        ) else {
            continue;
        };
        if score < best_score {
            best_score = score;
            best = candidate;
        }
    }
    (!best.is_empty()).then_some(best)
}

#[allow(clippy::too_many_arguments)]
fn fastcover_candidate_score(
    samples_buffer: &[u8],
    sample_sizes: &[usize],
    offsets: &[usize],
    test_sample_start: usize,
    content: &[u8],
    entropy_samples: &[u8],
    entropy_sample_sizes: &[usize],
    dict_capacity: usize,
    min_useful_dict: usize,
) -> Option<usize> {
    let dict = finalize_zstd_fallback_dict(
        content,
        entropy_samples,
        entropy_sample_sizes,
        dict_capacity,
        min_useful_dict,
    )?;
    let cdict = ZSTD_createCDict(&dict, 3)?;
    let max_sample_size = sample_sizes[test_sample_start..].iter().copied().max()?;
    let mut dst = vec![0u8; ZSTD_compressBound(max_sample_size)];
    let mut cctx = ZSTD_createCCtx()?;
    let mut score = dict.len();
    for idx in test_sample_start..sample_sizes.len() {
        let sample = &samples_buffer[offsets[idx]..offsets[idx + 1]];
        let written = ZSTD_compress_usingCDict(&mut cctx, &mut dst, sample, &cdict);
        if ERR_isError(written) {
            return None;
        }
        score = score.checked_add(written)?;
    }
    Some(score)
}

fn zstd_fastcover_training_count(sample_sizes: &[usize]) -> Option<usize> {
    let training_sample_count = sample_sizes.len() * 3 / 4;
    if training_sample_count < 5 || training_sample_count >= sample_sizes.len() {
        return None;
    }
    Some(training_sample_count)
}

fn fastcover_build_dictionary(
    samples_buffer: &[u8],
    offsets: &[usize],
    nb_dmers: usize,
    k: usize,
    dict_capacity: usize,
) -> Option<Vec<u8>> {
    const D: usize = 8;
    const F: u32 = 20;
    const MAX_ZERO_SCORE_RUN: usize = 10;

    let mut freqs = fastcover_compute_frequencies(samples_buffer, offsets)?;
    let (num_epochs, epoch_size) = fastcover_compute_epochs(dict_capacity, nb_dmers, k);
    if num_epochs == 0 || epoch_size == 0 {
        return None;
    }

    let mut segment_freqs = vec![0u16; 1usize << F];
    let mut dict = vec![0u8; dict_capacity];
    let mut tail = dict_capacity;
    let mut zero_score_run = 0usize;
    let mut epoch = 0usize;

    while tail > 0 {
        let epoch_begin = epoch.checked_mul(epoch_size)?;
        let segment = fastcover_select_segment(
            samples_buffer,
            &mut freqs,
            epoch_begin,
            epoch_begin + epoch_size,
            k,
            &mut segment_freqs,
        )?;
        if segment.score == 0 {
            zero_score_run += 1;
            if zero_score_run >= MAX_ZERO_SCORE_RUN {
                break;
            }
            epoch = (epoch + 1) % num_epochs;
            continue;
        }
        zero_score_run = 0;

        let segment_size = (segment.end - segment.begin + D - 1).min(tail);
        if segment_size < D {
            break;
        }
        tail -= segment_size;
        dict[tail..tail + segment_size]
            .copy_from_slice(&samples_buffer[segment.begin..segment.begin + segment_size]);
        epoch = (epoch + 1) % num_epochs;
    }

    (tail < dict_capacity).then(|| dict[tail..].to_vec())
}

fn fastcover_compute_frequencies(samples_buffer: &[u8], offsets: &[usize]) -> Option<Vec<u32>> {
    const D: usize = 8;
    const F: u32 = 20;

    let mut freqs = vec![0u32; 1usize << F];
    for window in offsets.windows(2) {
        let mut pos = window[0];
        let end = window[1];
        while pos + D <= end {
            let idx = ZSTD_hashPtr(&samples_buffer[pos..], F, D as u32);
            freqs[idx] = freqs[idx].wrapping_add(1);
            pos += 1;
        }
    }
    Some(freqs)
}

fn fastcover_compute_epochs(max_dict_size: usize, nb_dmers: usize, k: usize) -> (usize, usize) {
    let min_epoch_size = k * 10;
    let mut num = (max_dict_size / k).max(1);
    let mut size = nb_dmers / num;
    if size >= min_epoch_size {
        return (num, size);
    }
    size = min_epoch_size.min(nb_dmers);
    num = nb_dmers / size;
    (num.max(1), size)
}

fn fastcover_select_segment(
    samples_buffer: &[u8],
    freqs: &mut [u32],
    begin: usize,
    end: usize,
    k: usize,
    segment_freqs: &mut [u16],
) -> Option<FastCoverSegment> {
    const D: usize = 8;
    const F: u32 = 20;

    let dmers_in_k = k - D + 1;
    let mut best = FastCoverSegment {
        begin: 0,
        end: 0,
        score: 0,
    };
    let mut active = FastCoverSegment {
        begin,
        end: begin,
        score: 0,
    };

    while active.end < end {
        let idx = ZSTD_hashPtr(&samples_buffer[active.end..], F, D as u32);
        if segment_freqs[idx] == 0 {
            active.score = active.score.wrapping_add(freqs[idx]);
        }
        active.end += 1;
        segment_freqs[idx] = segment_freqs[idx].wrapping_add(1);

        if active.end - active.begin == dmers_in_k + 1 {
            let del_idx = ZSTD_hashPtr(&samples_buffer[active.begin..], F, D as u32);
            segment_freqs[del_idx] = segment_freqs[del_idx].wrapping_sub(1);
            if segment_freqs[del_idx] == 0 {
                active.score = active.score.wrapping_sub(freqs[del_idx]);
            }
            active.begin += 1;
        }

        if active.score > best.score {
            best = active;
        }
    }

    while active.begin < end {
        let del_idx = ZSTD_hashPtr(&samples_buffer[active.begin..], F, D as u32);
        segment_freqs[del_idx] = segment_freqs[del_idx].wrapping_sub(1);
        active.begin += 1;
    }

    for pos in best.begin..best.end {
        let idx = ZSTD_hashPtr(&samples_buffer[pos..], F, D as u32);
        freqs[idx] = 0;
    }

    Some(best)
}

fn finalize_zstd_fallback_dict(
    content: &[u8],
    entropy_samples: &[u8],
    entropy_sample_sizes: &[usize],
    dict_maxsize: usize,
    min_useful_dict: usize,
) -> Option<Vec<u8>> {
    if content.is_empty() || dict_maxsize < min_useful_dict {
        return None;
    }

    build_minimal_zstd_dict(
        content,
        entropy_samples,
        entropy_sample_sizes,
        dict_maxsize,
        min_useful_dict,
    )
}

fn build_minimal_zstd_dict(
    content: &[u8],
    entropy_samples: &[u8],
    entropy_sample_sizes: &[usize],
    dict_capacity: usize,
    min_useful_dict: usize,
) -> Option<Vec<u8>> {
    const MIN_CONTENT_SIZE: usize = 8;

    if content.is_empty() || dict_capacity < min_useful_dict || dict_capacity < content.len() {
        return None;
    }

    let mut out = Vec::with_capacity(content.len() + 256);
    out.extend_from_slice(&ZSTD_MAGIC_DICTIONARY.to_le_bytes());
    let random_id = XXH64(content, 0);
    let compliant_id = (random_id % ((1u64 << 31) - 32768)) + 32768;
    out.extend_from_slice(&(compliant_id as u32).to_le_bytes());

    let entropy_start = out.len();
    if append_zstd_entropy_tables_from_block_samples(
        &mut out,
        content,
        entropy_samples,
        entropy_sample_sizes,
    )
    .is_none()
    {
        out.truncate(entropy_start);
        let sample_len = entropy_samples.len().min(content.len());
        let huf_source = if sample_len == 0 {
            content
        } else {
            &entropy_samples[..sample_len]
        };
        if append_minimal_zstd_huf_table(&mut out, huf_source).is_none() {
            out.truncate(entropy_start);
            append_minimal_zstd_huf_table(&mut out, content)?;
        }
        append_zstd_fallback_sequence_tables(&mut out, content.len())?;
    }

    let header_len = out.len() + 12;
    if header_len > dict_capacity {
        return None;
    }
    let content_len = content.len().min(dict_capacity - header_len);
    if content_len < MIN_CONTENT_SIZE && header_len + MIN_CONTENT_SIZE > dict_capacity {
        return None;
    }
    let padding_len = MIN_CONTENT_SIZE.saturating_sub(content_len);
    for rep in [1u32, 4, 8] {
        out.extend_from_slice(&rep.to_le_bytes());
    }
    out.resize(out.len() + padding_len, 0);
    out.extend_from_slice(&content[..content_len]);
    if out.len() > dict_capacity {
        return None;
    }
    Some(out)
}

fn append_zstd_entropy_tables_from_block_samples(
    out: &mut Vec<u8>,
    content: &[u8],
    entropy_samples: &[u8],
    entropy_sample_sizes: &[usize],
) -> Option<()> {
    const ZDICT_OFFCODE_MAX: u32 = 30;
    const ZSTD_BLOCKSIZE_MAX: usize = 128 * 1024;

    let samples_len = entropy_sample_sizes
        .iter()
        .try_fold(0usize, |acc, &size| acc.checked_add(size))?;
    if samples_len == 0 || samples_len > entropy_samples.len() {
        return None;
    }

    let offcode_max = ZSTD_highbit32((content.len() + ZSTD_BLOCKSIZE_MAX) as u32);
    if offcode_max > ZDICT_OFFCODE_MAX || offcode_max > MaxOff {
        return None;
    }

    // C `ZDICT_analyzeEntropy`: counts are seeded with 1 for every symbol in range.
    let mut literal_count = [1u32; 256];
    let mut offcode_count = zstd_seeded_offcode_counts(offcode_max);
    let mut ml_count = vec![1u32; (MaxML + 1) as usize];
    let mut ll_count = vec![1u32; (MaxLL + 1) as usize];

    // C builds the entropy CDict with cParams derived from the average sample
    // size and the dict (content) size: `ZSTD_getParams(level, avgSampleSize,
    // dictSize)` then `ZSTD_createCDict_advanced(content, byRef, rawContent, ...)`.
    let avg_sample_size = samples_len / entropy_sample_sizes.len().max(1);
    let params = ZSTD_getParams(3, avg_sample_size as u64, content.len());
    let block_size_max = ZSTD_BLOCKSIZE_MAX.min(1usize << params.cParams.windowLog);
    let cdict = ZSTD_createCDict_advanced(
        content,
        ZSTD_dictLoadMethod_e::ZSTD_dlm_byRef,
        ZSTD_dictContentType_e::ZSTD_dct_rawContent,
        params.cParams,
    )?;
    let mut cctx = ZSTD_createCCtx()?;
    let mut work_place = vec![0u8; ZSTD_BLOCKSIZE_MAX];
    let mut sample_offset = 0usize;
    for &sample_size in entropy_sample_sizes {
        let sample_end = sample_offset.checked_add(sample_size)?;
        let sample = &entropy_samples[sample_offset..sample_end];
        let sample = &sample[..sample.len().min(block_size_max)];
        sample_offset = sample_end;
        if sample.is_empty() {
            continue;
        }

        if ERR_isError(ZSTD_compressBegin_usingCDict_deprecated(&mut cctx, &cdict)) {
            return None;
        }
        let csize = ZSTD_compressBlock_deprecated(&mut cctx, &mut work_place, sample);
        if ERR_isError(csize) {
            return None;
        }
        if csize == 0 {
            continue;
        }

        let seq_store = cctx.seqStore.as_mut()?;
        for &literal in &seq_store.literals {
            literal_count[literal as usize] = literal_count[literal as usize].saturating_add(1);
        }
        ZSTD_seqToCodes(seq_store);
        for idx in 0..seq_store.sequences.len() {
            let offcode = *seq_store.ofCode.get(idx)? as u32;
            if offcode > offcode_max {
                return None;
            }
            offcode_count[offcode as usize] = offcode_count[offcode as usize].saturating_add(1);

            let ml_code = *seq_store.mlCode.get(idx)? as u32;
            let ll_code = *seq_store.llCode.get(idx)? as u32;
            if ml_code > MaxML || ll_code > MaxLL {
                return None;
            }
            ml_count[ml_code as usize] = ml_count[ml_code as usize].saturating_add(1);
            ll_count[ll_code as usize] = ll_count[ll_code as usize].saturating_add(1);
        }
    }

    append_zstd_huf_table_from_counts(out, &literal_count)?;
    append_normalized_zstd_count(
        out,
        &offcode_count,
        offcode_max,
        ZDICT_OFFCODE_MAX,
        OffFSELog,
    )?;
    append_normalized_zstd_count(out, &ml_count, MaxML, MaxML, MLFSELog)?;
    append_normalized_zstd_count(out, &ll_count, MaxLL, MaxLL, LLFSELog)?;
    Some(())
}

fn append_zstd_fallback_sequence_tables(out: &mut Vec<u8>, dict_content_len: usize) -> Option<()> {
    const ZDICT_OFFCODE_MAX: u32 = 30;

    let offcode_max = ZSTD_highbit32((dict_content_len + (128 * 1024)) as u32);
    if offcode_max > ZDICT_OFFCODE_MAX || offcode_max > MaxOff {
        return None;
    }
    let offcode_count = zstd_seeded_offcode_counts(offcode_max);
    let ml_count = vec![1u32; (MaxML + 1) as usize];
    let ll_count = vec![1u32; (MaxLL + 1) as usize];
    append_normalized_zstd_count(
        out,
        &offcode_count,
        offcode_max,
        ZDICT_OFFCODE_MAX,
        OffFSELog,
    )?;
    append_normalized_zstd_count(out, &ml_count, MaxML, MaxML, MLFSELog)?;
    append_normalized_zstd_count(out, &ll_count, MaxLL, MaxLL, LLFSELog)?;
    Some(())
}

fn zstd_seeded_offcode_counts(offcode_max: u32) -> Vec<u32> {
    let mut offcode_count = vec![0u32; (MaxOff + 1) as usize];
    for count in offcode_count.iter_mut().take(offcode_max as usize + 1) {
        *count = 1;
    }
    offcode_count
}

fn append_normalized_zstd_count(
    out: &mut Vec<u8>,
    count: &[u32],
    normalize_max_symbol: u32,
    write_max_symbol: u32,
    table_log: u32,
) -> Option<()> {
    let total = count
        .iter()
        .take(normalize_max_symbol as usize + 1)
        .map(|&count| count as usize)
        .sum::<usize>();
    let mut normalized = vec![0i16; count.len()];
    let normalized_log = FSE_normalizeCount(
        &mut normalized,
        table_log,
        count,
        total,
        normalize_max_symbol,
        1,
    );
    if ERR_isError(normalized_log) || normalized_log == 0 {
        return None;
    }
    let mut fse_header = vec![0u8; 256];
    let written = FSE_writeNCount(
        &mut fse_header,
        &normalized,
        write_max_symbol,
        normalized_log as u32,
    );
    if ERR_isError(written) {
        return None;
    }
    out.extend_from_slice(&fse_header[..written]);
    Some(())
}

fn append_minimal_zstd_huf_table(out: &mut Vec<u8>, huf_source: &[u8]) -> Option<()> {
    let mut count = [1u32; 256];
    for &byte in huf_source {
        count[byte as usize] += 1;
    }
    append_zstd_huf_table_from_counts(out, &count)
}

fn append_zstd_huf_table_from_counts(out: &mut Vec<u8>, count: &[u32; 256]) -> Option<()> {
    // C `ZDICT_analyzeEntropy`: `huffLog = 11` is passed as the *cap* to
    // `HUF_buildCTable_wksp` (not an "optimal" log), and the resulting
    // `maxNbBits` is then used directly for `HUF_writeCTable`.
    const HUFFLOG: u32 = 11;
    let max_symbol_value = 255;
    let mut ctable = vec![0 as HUF_CElt; 257];
    let mut workspace = vec![0u32; HUF_CTABLE_WORKSPACE_SIZE_U32];
    let mut max_nb_bits =
        HUF_buildCTable_wksp(&mut ctable, count, max_symbol_value, HUFFLOG, &mut workspace);
    if ERR_isError(max_nb_bits) {
        return None;
    }
    if max_nb_bits == 8 {
        // Pathological (incompressible) literals: C replaces the distribution
        // with `ZDICT_flatLit` (a mostly-flat but encodable distribution) and
        // rebuilds — yielding maxNbBits == 9.
        let mut flat_count = [2u32; 256];
        flat_count[0] = 4;
        flat_count[253] = 1;
        flat_count[254] = 1;
        max_nb_bits =
            HUF_buildCTable_wksp(&mut ctable, &flat_count, max_symbol_value, HUFFLOG, &mut workspace);
        if ERR_isError(max_nb_bits) {
            return None;
        }
    }
    let huff_log = max_nb_bits as u32;
    let mut huf_header = vec![0u8; 512];
    let written = HUF_writeCTable(&mut huf_header, &ctable, max_symbol_value, huff_log);
    if ERR_isError(written) {
        return None;
    }
    out.extend_from_slice(&huf_header[..written]);
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_from_buffer_produces_magic_dictionary() {
        // Build many small, varied-but-repetitive samples so fastcover has
        // recurring d-mers to latch onto.
        let mut samples = Vec::new();
        let mut sizes = Vec::new();
        for i in 0..64u32 {
            let mut s = Vec::new();
            for j in 0..40u32 {
                s.extend_from_slice(&((i % 7) * 1000 + (j % 11)).to_le_bytes());
            }
            sizes.push(s.len());
            samples.extend_from_slice(&s);
        }

        let dict = train_from_buffer(&samples, &sizes, 4096, 8)
            .expect("dictionary training should succeed for repetitive samples");
        assert!(dict.len() >= 8);
        assert_eq!(
            &dict[..4],
            &ZSTD_MAGIC_DICTIONARY.to_le_bytes(),
            "dictionary must start with the zstd dictionary magic"
        );
    }
}


