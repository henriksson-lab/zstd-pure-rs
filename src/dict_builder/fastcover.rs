//! Translation of `lib/dictBuilder/fastcover.c` (the fastCover dictionary
//! trainer). The verbatim translation lives in [`fastcover_c`]; its public
//! entry points are re-exported here.

pub use fastcover_c::{
    ZDICT_optimizeTrainFromBuffer_fastCover, ZDICT_trainFromBuffer_fastCover,
};

#[allow(dead_code)]
#[allow(non_upper_case_globals)]
pub mod fastcover_c {
    use crate::common::error::{ErrorCode, ERROR};
    use crate::compress::zstd_hashes::{ZSTD_hash6Ptr, ZSTD_hash8Ptr};
    use crate::dict_builder::cover::{COVER_computeEpochs, COVER_segment_t, COVER_sum};
    use crate::dict_builder::zdict::ZDICT_cover_params_t;

    pub const FASTCOVER_MAX_SAMPLES_SIZE: usize = u32::MAX as usize; // 64-bit
    pub const FASTCOVER_MAX_F: u32 = 31;
    pub const FASTCOVER_MAX_ACCEL: usize = 10;
    pub const FASTCOVER_DEFAULT_SPLITPOINT: f64 = 0.75;
    pub const DEFAULT_F: u32 = 20;
    pub const DEFAULT_ACCEL: u32 = 1;

    /// Port of `FASTCOVER_hashPtrToIndex`. `p` must have >= 8 readable bytes.
    #[inline]
    unsafe fn FASTCOVER_hashPtrToIndex(p: *const u8, f: u32, d: u32) -> usize {
        let slice = core::slice::from_raw_parts(p, 8);
        if d == 6 {
            ZSTD_hash6Ptr(slice, f)
        } else {
            ZSTD_hash8Ptr(slice, f)
        }
    }

    /// Port of `FASTCOVER_accel_t`.
    #[derive(Clone, Copy, Default)]
    pub struct FASTCOVER_accel_t {
        pub finalize: u32,
        pub skip: u32,
    }

    /// Port of `FASTCOVER_defaultAccelParameters`.
    pub static FASTCOVER_defaultAccelParameters: [FASTCOVER_accel_t; FASTCOVER_MAX_ACCEL + 1] = [
        FASTCOVER_accel_t { finalize: 100, skip: 0 }, // accel = 0 (defaults to 1)
        FASTCOVER_accel_t { finalize: 100, skip: 0 }, // accel = 1
        FASTCOVER_accel_t { finalize: 50, skip: 1 },
        FASTCOVER_accel_t { finalize: 34, skip: 2 },
        FASTCOVER_accel_t { finalize: 25, skip: 3 },
        FASTCOVER_accel_t { finalize: 20, skip: 4 },
        FASTCOVER_accel_t { finalize: 17, skip: 5 },
        FASTCOVER_accel_t { finalize: 14, skip: 6 },
        FASTCOVER_accel_t { finalize: 13, skip: 7 },
        FASTCOVER_accel_t { finalize: 11, skip: 8 },
        FASTCOVER_accel_t { finalize: 10, skip: 9 },
    ];

    /// Port of `FASTCOVER_ctx_t`.
    pub struct FASTCOVER_ctx_t {
        pub samples: *const u8,
        pub offsets: Vec<usize>,
        pub samplesSizes: *const usize,
        pub nbSamples: usize,
        pub nbTrainSamples: usize,
        pub nbTestSamples: usize,
        pub nbDmers: usize,
        pub freqs: Vec<u32>,
        pub d: u32,
        pub f: u32,
        pub accelParams: FASTCOVER_accel_t,
        pub displayLevel: i32,
    }

    impl Default for FASTCOVER_ctx_t {
        fn default() -> Self {
            FASTCOVER_ctx_t {
                samples: core::ptr::null(),
                offsets: Vec::new(),
                samplesSizes: core::ptr::null(),
                nbSamples: 0,
                nbTrainSamples: 0,
                nbTestSamples: 0,
                nbDmers: 0,
                freqs: Vec::new(),
                d: 0,
                f: 0,
                accelParams: FASTCOVER_accel_t::default(),
                displayLevel: 0,
            }
        }
    }

    /// Port of `FASTCOVER_selectSegment`.
    pub fn FASTCOVER_selectSegment(
        ctx: &FASTCOVER_ctx_t,
        freqs: &mut [u32],
        begin: u32,
        end: u32,
        parameters: ZDICT_cover_params_t,
        segmentFreqs: &mut [u16],
    ) -> COVER_segment_t {
        let k = parameters.k;
        let d = parameters.d;
        let f = ctx.f;
        let dmersInK = k - d + 1;

        let mut bestSegment = COVER_segment_t {
            begin: 0,
            end: 0,
            score: 0,
        };
        let mut activeSegment = COVER_segment_t {
            begin,
            end: begin,
            score: 0,
        };

        while activeSegment.end < end {
            let idx = unsafe { FASTCOVER_hashPtrToIndex(ctx.samples.add(activeSegment.end as usize), f, d) };
            if segmentFreqs[idx] == 0 {
                activeSegment.score += freqs[idx];
            }
            activeSegment.end += 1;
            segmentFreqs[idx] += 1;
            if activeSegment.end - activeSegment.begin == dmersInK + 1 {
                let delIndex =
                    unsafe { FASTCOVER_hashPtrToIndex(ctx.samples.add(activeSegment.begin as usize), f, d) };
                segmentFreqs[delIndex] -= 1;
                if segmentFreqs[delIndex] == 0 {
                    activeSegment.score -= freqs[delIndex];
                }
                activeSegment.begin += 1;
            }
            if activeSegment.score > bestSegment.score {
                bestSegment = activeSegment;
            }
        }

        /* Zero out rest of segmentFreqs array */
        while activeSegment.begin < end {
            let delIndex =
                unsafe { FASTCOVER_hashPtrToIndex(ctx.samples.add(activeSegment.begin as usize), f, d) };
            segmentFreqs[delIndex] -= 1;
            activeSegment.begin += 1;
        }

        {
            let mut pos = bestSegment.begin;
            while pos != bestSegment.end {
                let i = unsafe { FASTCOVER_hashPtrToIndex(ctx.samples.add(pos as usize), f, d) };
                freqs[i] = 0;
                pos += 1;
            }
        }

        bestSegment
    }

    /// Port of `FASTCOVER_checkParameters`.
    pub fn FASTCOVER_checkParameters(
        parameters: ZDICT_cover_params_t,
        maxDictSize: usize,
        f: u32,
        accel: u32,
    ) -> i32 {
        if parameters.d == 0 || parameters.k == 0 {
            return 0;
        }
        if parameters.d != 6 && parameters.d != 8 {
            return 0;
        }
        if parameters.k as usize > maxDictSize {
            return 0;
        }
        if parameters.d > parameters.k {
            return 0;
        }
        if f > FASTCOVER_MAX_F || f == 0 {
            return 0;
        }
        if parameters.splitPoint <= 0.0 || parameters.splitPoint > 1.0 {
            return 0;
        }
        if accel > 10 || accel == 0 {
            return 0;
        }
        1
    }

    /// Port of `FASTCOVER_ctx_destroy`.
    pub fn FASTCOVER_ctx_destroy(ctx: &mut FASTCOVER_ctx_t) {
        ctx.freqs = Vec::new();
        ctx.offsets = Vec::new();
    }

    /// Port of `FASTCOVER_computeFrequency`.
    pub fn FASTCOVER_computeFrequency(freqs: &mut [u32], ctx: &FASTCOVER_ctx_t) {
        let f = ctx.f;
        let d = ctx.d;
        let skip = ctx.accelParams.skip;
        let readLength = core::cmp::max(d as usize, 8);
        for i in 0..ctx.nbTrainSamples {
            let mut start = ctx.offsets[i]; /* start of current dmer */
            let currSampleEnd = ctx.offsets[i + 1];
            while start + readLength <= currSampleEnd {
                let dmerIndex = unsafe { FASTCOVER_hashPtrToIndex(ctx.samples.add(start), f, d) };
                freqs[dmerIndex] += 1;
                start = start + skip as usize + 1;
            }
        }
    }

    /// Port of `FASTCOVER_ctx_init`. Returns 0 on success or an error code.
    #[allow(clippy::too_many_arguments)]
    pub fn FASTCOVER_ctx_init(
        ctx: &mut FASTCOVER_ctx_t,
        samples: &[u8],
        samplesSizes: &[usize],
        nbSamples: u32,
        d: u32,
        splitPoint: f64,
        f: u32,
        accelParams: FASTCOVER_accel_t,
        displayLevel: i32,
    ) -> usize {
        let totalSamplesSize = COVER_sum(samplesSizes, nbSamples);
        let nbTrainSamples = if splitPoint < 1.0 {
            (nbSamples as f64 * splitPoint) as u32
        } else {
            nbSamples
        };
        let nbTestSamples = if splitPoint < 1.0 {
            nbSamples - nbTrainSamples
        } else {
            nbSamples
        };
        let trainingSamplesSize = if splitPoint < 1.0 {
            COVER_sum(samplesSizes, nbTrainSamples)
        } else {
            totalSamplesSize
        };
        let _testSamplesSize = if splitPoint < 1.0 {
            COVER_sum(&samplesSizes[nbTrainSamples as usize..], nbTestSamples)
        } else {
            totalSamplesSize
        };
        ctx.displayLevel = displayLevel;

        /* Checks */
        if totalSamplesSize < core::cmp::max(d as usize, core::mem::size_of::<u64>())
            || totalSamplesSize >= FASTCOVER_MAX_SAMPLES_SIZE
        {
            return ERROR(ErrorCode::SrcSizeWrong);
        }
        if nbTrainSamples < 5 {
            return ERROR(ErrorCode::SrcSizeWrong);
        }
        if nbTestSamples < 1 {
            return ERROR(ErrorCode::SrcSizeWrong);
        }

        /* Zero the context (fresh) */
        *ctx = FASTCOVER_ctx_t::default();
        ctx.displayLevel = displayLevel;
        ctx.samples = samples.as_ptr();
        ctx.samplesSizes = samplesSizes.as_ptr();
        ctx.nbSamples = nbSamples as usize;
        ctx.nbTrainSamples = nbTrainSamples as usize;
        ctx.nbTestSamples = nbTestSamples as usize;
        ctx.nbDmers = trainingSamplesSize - core::cmp::max(d as usize, core::mem::size_of::<u64>()) + 1;
        ctx.d = d;
        ctx.f = f;
        ctx.accelParams = accelParams;

        /* The offsets of each file */
        ctx.offsets = vec![0usize; nbSamples as usize + 1];

        /* Fill offsets from the samplesSizes */
        ctx.offsets[0] = 0;
        for i in 1..=nbSamples as usize {
            ctx.offsets[i] = ctx.offsets[i - 1] + samplesSizes[i - 1];
        }

        /* Initialize frequency array of size 2^f */
        ctx.freqs = vec![0u32; 1usize << f];

        /* Compute frequencies. `freqs` aliases `ctx.freqs` in C; move it out so
         * `&ctx` + `&mut freqs` don't conflict, then put it back. */
        let mut freqs_local = core::mem::take(&mut ctx.freqs);
        FASTCOVER_computeFrequency(&mut freqs_local, ctx);
        ctx.freqs = freqs_local;
        0
    }

    /// Port of `FASTCOVER_buildDictionary`.
    #[allow(clippy::too_many_arguments)]
    pub fn FASTCOVER_buildDictionary(
        ctx: &FASTCOVER_ctx_t,
        freqs: &mut [u32],
        dict: &mut [u8],
        dictBufferCapacity: usize,
        parameters: ZDICT_cover_params_t,
        segmentFreqs: &mut [u16],
    ) -> usize {
        let mut tail = dictBufferCapacity;
        let epochs = COVER_computeEpochs(
            dictBufferCapacity as u32,
            ctx.nbDmers as u32,
            parameters.k,
            1,
        );
        let maxZeroScoreRun: usize = 10;
        let mut zeroScoreRun: usize = 0;
        let mut epoch: usize = 0;
        while tail > 0 {
            let epochBegin = (epoch as u32).wrapping_mul(epochs.size);
            let epochEnd = epochBegin + epochs.size;
            let segment =
                FASTCOVER_selectSegment(ctx, freqs, epochBegin, epochEnd, parameters, segmentFreqs);
            if segment.score == 0 {
                zeroScoreRun += 1;
                if zeroScoreRun >= maxZeroScoreRun {
                    break;
                }
                epoch = (epoch + 1) % epochs.num as usize;
                continue;
            }
            zeroScoreRun = 0;
            let segmentSize = core::cmp::min(
                (segment.end - segment.begin + parameters.d - 1) as usize,
                tail,
            );
            if segmentSize < parameters.d as usize {
                break;
            }
            tail -= segmentSize;
            unsafe {
                let src = core::slice::from_raw_parts(ctx.samples.add(segment.begin as usize), segmentSize);
                dict[tail..tail + segmentSize].copy_from_slice(src);
            }
            epoch = (epoch + 1) % epochs.num as usize;
        }
        tail
    }

    /// Port of `FASTCOVER_convertToCoverParams`.
    pub fn FASTCOVER_convertToCoverParams(
        fastCoverParams: crate::dict_builder::zdict::ZDICT_fastCover_params_t,
        coverParams: &mut ZDICT_cover_params_t,
    ) {
        coverParams.k = fastCoverParams.k;
        coverParams.d = fastCoverParams.d;
        coverParams.steps = fastCoverParams.steps;
        coverParams.nbThreads = fastCoverParams.nbThreads;
        coverParams.splitPoint = fastCoverParams.splitPoint;
        coverParams.zParams = fastCoverParams.zParams;
        coverParams.shrinkDict = fastCoverParams.shrinkDict;
    }

    /// Port of `FASTCOVER_convertToFastCoverParams`.
    pub fn FASTCOVER_convertToFastCoverParams(
        coverParams: ZDICT_cover_params_t,
        fastCoverParams: &mut crate::dict_builder::zdict::ZDICT_fastCover_params_t,
        f: u32,
        accel: u32,
    ) {
        fastCoverParams.k = coverParams.k;
        fastCoverParams.d = coverParams.d;
        fastCoverParams.steps = coverParams.steps;
        fastCoverParams.nbThreads = coverParams.nbThreads;
        fastCoverParams.splitPoint = coverParams.splitPoint;
        fastCoverParams.f = f;
        fastCoverParams.accel = accel;
        fastCoverParams.zParams = coverParams.zParams;
        fastCoverParams.shrinkDict = coverParams.shrinkDict;
    }

    /// Port of `ZDICT_trainFromBuffer_fastCover`. Single-threaded fastCover
    /// training entry point. Returns the dictionary size or an error code.
    pub fn ZDICT_trainFromBuffer_fastCover(
        dictBuffer: &mut [u8],
        dictBufferCapacity: usize,
        samplesBuffer: &[u8],
        samplesSizes: &[usize],
        nbSamples: u32,
        mut parameters: crate::dict_builder::zdict::ZDICT_fastCover_params_t,
    ) -> usize {
        use crate::common::error::ERR_isError;
        use crate::dict_builder::cover::COVER_warnOnSmallCorpus;
        use crate::dict_builder::zdict::{ZDICT_finalizeDictionary, ZDICT_DICTSIZE_MIN};

        let displayLevel = parameters.zParams.notificationLevel as i32;
        /* Assign splitPoint and f if not provided */
        parameters.splitPoint = 1.0;
        parameters.f = if parameters.f == 0 { DEFAULT_F } else { parameters.f };
        parameters.accel = if parameters.accel == 0 {
            DEFAULT_ACCEL
        } else {
            parameters.accel
        };
        /* Convert to cover parameter */
        let mut coverParams = ZDICT_cover_params_t::default();
        FASTCOVER_convertToCoverParams(parameters, &mut coverParams);
        /* Checks */
        if FASTCOVER_checkParameters(coverParams, dictBufferCapacity, parameters.f, parameters.accel) == 0 {
            return ERROR(ErrorCode::ParameterOutOfBound);
        }
        if nbSamples == 0 {
            return ERROR(ErrorCode::SrcSizeWrong);
        }
        if dictBufferCapacity < ZDICT_DICTSIZE_MIN {
            return ERROR(ErrorCode::DstSizeTooSmall);
        }
        /* Assign corresponding FASTCOVER_accel_t */
        let accelParams = FASTCOVER_defaultAccelParameters[parameters.accel as usize];
        /* Initialize context */
        let mut ctx = FASTCOVER_ctx_t::default();
        {
            let initVal = FASTCOVER_ctx_init(
                &mut ctx,
                samplesBuffer,
                samplesSizes,
                nbSamples,
                coverParams.d,
                parameters.splitPoint,
                parameters.f,
                accelParams,
                displayLevel,
            );
            if ERR_isError(initVal) {
                return initVal;
            }
        }
        COVER_warnOnSmallCorpus(dictBufferCapacity, ctx.nbDmers, displayLevel);
        /* Build the dictionary */
        let mut segmentFreqs = vec![0u16; 1usize << parameters.f];
        let dictionarySize;
        {
            // freqs aliases ctx.freqs; move out / restore.
            let mut freqs = core::mem::take(&mut ctx.freqs);
            let tail = FASTCOVER_buildDictionary(
                &ctx,
                &mut freqs,
                dictBuffer,
                dictBufferCapacity,
                coverParams,
                &mut segmentFreqs,
            );
            ctx.freqs = freqs;
            let nbFinalizeSamples =
                (ctx.nbTrainSamples * ctx.accelParams.finalize as usize / 100) as u32;
            // customDictContent == dict + tail overlaps dictBuffer; copy out.
            let content: Vec<u8> = dictBuffer[tail..dictBufferCapacity].to_vec();
            dictionarySize = ZDICT_finalizeDictionary(
                dictBuffer,
                dictBufferCapacity,
                &content,
                dictBufferCapacity - tail,
                samplesBuffer,
                samplesSizes,
                nbFinalizeSamples,
                coverParams.zParams,
            );
        }
        FASTCOVER_ctx_destroy(&mut ctx);
        dictionarySize
    }

    /// Port of `FASTCOVER_tryParameters`. Tries one parameter set and updates
    /// `best`. (C passes an owning opaque pointer for threading; here it is a
    /// plain sequential call.)
    fn FASTCOVER_tryParameters(
        ctx: &FASTCOVER_ctx_t,
        best: &mut crate::dict_builder::cover::COVER_best_t,
        dictBufferCapacity: usize,
        parameters: ZDICT_cover_params_t,
    ) {
        use crate::dict_builder::cover::{
            COVER_best_finish, COVER_dictSelectionError, COVER_dictSelectionIsError, COVER_selectDict,
        };

        let totalCompressedSize = ERROR(ErrorCode::Generic);
        let mut segmentFreqs = vec![0u16; 1usize << ctx.f];
        let mut dict = vec![0u8; dictBufferCapacity];
        let mut selection = COVER_dictSelectionError(ERROR(ErrorCode::Generic));
        /* Copy the frequencies because we need to modify them */
        let mut freqs = ctx.freqs.clone();
        {
            let tail = FASTCOVER_buildDictionary(
                ctx,
                &mut freqs,
                &mut dict,
                dictBufferCapacity,
                parameters,
                &mut segmentFreqs,
            );
            let nbFinalizeSamples =
                (ctx.nbTrainSamples * ctx.accelParams.finalize as usize / 100) as u32;
            let content = dict[tail..dictBufferCapacity].to_vec();
            let total_samples_size = ctx.offsets[ctx.nbSamples];
            let samples = unsafe { core::slice::from_raw_parts(ctx.samples, total_samples_size) };
            let samplesSizes = unsafe { core::slice::from_raw_parts(ctx.samplesSizes, ctx.nbSamples) };
            selection = COVER_selectDict(
                &content,
                dictBufferCapacity,
                dictBufferCapacity - tail,
                samples,
                samplesSizes,
                nbFinalizeSamples,
                ctx.nbTrainSamples,
                ctx.nbSamples,
                parameters,
                &ctx.offsets,
                totalCompressedSize,
            );
            let _ = COVER_dictSelectionIsError(&selection); /* C logs/cleanups; best_finish still called */
        }
        COVER_best_finish(best, parameters, &selection);
        // `selection` (and its Vec) drops here == COVER_dictSelectionFree.
    }

    /// Port of `ZDICT_optimizeTrainFromBuffer_fastCover`. Tries a grid of
    /// (d, k) parameters and returns the best dictionary. The POOL parallelism
    /// is collapsed to a sequential loop (deterministic result).
    pub fn ZDICT_optimizeTrainFromBuffer_fastCover(
        dictBuffer: &mut [u8],
        dictBufferCapacity: usize,
        samplesBuffer: &[u8],
        samplesSizes: &[usize],
        nbSamples: u32,
        parameters: &mut crate::dict_builder::zdict::ZDICT_fastCover_params_t,
    ) -> usize {
        use crate::common::error::ERR_isError;
        use crate::dict_builder::cover::{
            COVER_best_destroy, COVER_best_init, COVER_best_start, COVER_best_t, COVER_best_wait,
            COVER_warnOnSmallCorpus,
        };
        use crate::dict_builder::zdict::ZDICT_DICTSIZE_MIN;

        let _nbThreads = parameters.nbThreads; // POOL collapsed to sequential
        let splitPoint = if parameters.splitPoint <= 0.0 {
            FASTCOVER_DEFAULT_SPLITPOINT
        } else {
            parameters.splitPoint
        };
        let kMinD = if parameters.d == 0 { 6 } else { parameters.d };
        let kMaxD = if parameters.d == 0 { 8 } else { parameters.d };
        let kMinK = if parameters.k == 0 { 50 } else { parameters.k };
        let kMaxK = if parameters.k == 0 { 2000 } else { parameters.k };
        let kSteps = if parameters.steps == 0 { 40 } else { parameters.steps };
        let kStepSize = core::cmp::max((kMaxK - kMinK) / kSteps, 1);
        let f = if parameters.f == 0 { DEFAULT_F } else { parameters.f };
        let accel = if parameters.accel == 0 { DEFAULT_ACCEL } else { parameters.accel };
        let shrinkDict: u32 = 0;
        let displayLevel = parameters.zParams.notificationLevel as i32;
        let _ = displayLevel;

        /* Checks */
        if splitPoint <= 0.0 || splitPoint > 1.0 {
            return ERROR(ErrorCode::ParameterOutOfBound);
        }
        if accel == 0 || accel as usize > FASTCOVER_MAX_ACCEL {
            return ERROR(ErrorCode::ParameterOutOfBound);
        }
        if kMinK < kMaxD || kMaxK < kMinK {
            return ERROR(ErrorCode::ParameterOutOfBound);
        }
        if nbSamples == 0 {
            return ERROR(ErrorCode::SrcSizeWrong);
        }
        if dictBufferCapacity < ZDICT_DICTSIZE_MIN {
            return ERROR(ErrorCode::DstSizeTooSmall);
        }

        /* Initialization */
        let mut best = COVER_best_t::default();
        COVER_best_init(&mut best);
        let mut coverParams = ZDICT_cover_params_t::default();
        FASTCOVER_convertToCoverParams(*parameters, &mut coverParams);
        let accelParams = FASTCOVER_defaultAccelParameters[accel as usize];
        let mut warned = false;

        /* Loop through d first because each new value needs a new context */
        let mut d = kMinD;
        while d <= kMaxD {
            let mut ctx = FASTCOVER_ctx_t::default();
            {
                let childDisplayLevel = if displayLevel == 0 { 0 } else { displayLevel - 1 };
                let initVal = FASTCOVER_ctx_init(
                    &mut ctx,
                    samplesBuffer,
                    samplesSizes,
                    nbSamples,
                    d,
                    splitPoint,
                    f,
                    accelParams,
                    childDisplayLevel,
                );
                if ERR_isError(initVal) {
                    COVER_best_destroy(&mut best);
                    return initVal;
                }
            }
            if !warned {
                COVER_warnOnSmallCorpus(dictBufferCapacity, ctx.nbDmers, displayLevel);
                warned = true;
            }
            /* Loop through k reusing the same context */
            let mut k = kMinK;
            while k <= kMaxK {
                let mut p = coverParams;
                p.k = k;
                p.d = d;
                p.splitPoint = splitPoint;
                p.steps = kSteps;
                p.shrinkDict = shrinkDict;
                p.zParams.notificationLevel = ctx.displayLevel as u32;
                if FASTCOVER_checkParameters(p, dictBufferCapacity, ctx.f, accel) == 0 {
                    k += kStepSize;
                    continue;
                }
                COVER_best_start(&mut best);
                FASTCOVER_tryParameters(&ctx, &mut best, dictBufferCapacity, p);
                k += kStepSize;
            }
            COVER_best_wait(&best);
            FASTCOVER_ctx_destroy(&mut ctx);
            d += 2;
        }

        /* Fill the output buffer and parameters with the best */
        let dictSize = best.dictSize;
        if ERR_isError(best.compressedSize) {
            let compressedSize = best.compressedSize;
            COVER_best_destroy(&mut best);
            return compressedSize;
        }
        FASTCOVER_convertToFastCoverParams(best.parameters, parameters, f, accel);
        dictBuffer[..dictSize].copy_from_slice(&best.dict[..dictSize]);
        COVER_best_destroy(&mut best);
        dictSize
    }
}
