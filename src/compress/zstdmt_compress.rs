//! Translation of `lib/compress/zstdmt_compress.c`. Multi-threaded
//! compression. Behind the `mt` feature.
//!
//! Current v0.1 provides the API surface as returnable errors /
//! no-ops so the `mt` feature can compile without triggering panics.
//! A real port would use `rayon` for parallel block compression, but
//! the sequential path already handles all correctness requirements —
//! MT is purely a speed optimization.

use core::marker::PhantomData;
use crate::compress::zstd_compress::{
    ZSTD_CCtx, ZSTD_CCtxParams_setParameter, ZSTD_CCtx_params, ZSTD_EndDirective,
    ZSTD_CCtx_trace, ZSTD_compressBegin_advanced_internal, ZSTD_compressBound,
    ZSTD_compressContinue_public, ZSTD_compressEnd_public, ZSTD_createCCtx_advanced,
    ZSTD_customMem, ZSTD_cycleLog, ZSTD_frameProgression, ZSTD_invalidateRepCodes,
    ZSTD_referenceExternalSequences, ZSTD_sizeof_CCtx, ZSTD_writeLastEmptyBlock,
    ZSTD_CCtxParams_init, ZSTD_CLEVEL_DEFAULT, ZSTD_CParamMode_e,
    ZSTD_getCParamsFromCCtxParams,
};
use crate::compress::zstd_hashes::{
    ZSTD_rollingHash_append, ZSTD_rollingHash_compute, ZSTD_rollingHash_primePower,
    ZSTD_rollingHash_rotate,
};
use crate::compress::zstd_compress::{
    ZSTD_cParameter, ZSTD_WINDOWLOG_MAX,
};
use crate::compress::zstd_ldm::{ZSTD_ParamSwitch_e, ZSTD_ldm_getMaxNbSeq};
use crate::compress::zstd_ldm::{RawSeqStore_t, rawSeq};
use crate::decompress::zstd_ddict::ZSTD_dictContentType_e;
use crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX;
use std::sync::OnceLock;
use std::time::Instant;

pub const ZSTDMT_JOBLOG_MAX: u32 = 30;
pub const ZSTDMT_JOBSIZE_MIN: usize = 1 << 20;
pub const ZSTDMT_JOBSIZE_MAX: usize = 1 << ZSTDMT_JOBLOG_MAX;
pub const RSYNC_LENGTH: usize = 32;
pub const RSYNC_MIN_BLOCK_LOG: usize = 17;
pub const RSYNC_MIN_BLOCK_SIZE: usize = 1 << RSYNC_MIN_BLOCK_LOG;

#[derive(Debug, Clone, Copy, Default)]
pub struct Range {
    pub start: usize,
    pub size: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SyncPoint {
    pub toLoad: usize,
    pub flush: i32,
}

#[derive(Debug, Clone, Default)]
pub struct InBuff_t {
    pub prefix: Range,
    pub buffer: Buffer,
    pub filled: usize,
}

#[derive(Debug, Clone, Default)]
pub struct RoundBuff_t {
    pub buffer: Buffer,
    pub capacity: usize,
    pub pos: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RSyncState_t {
    pub hash: u64,
    pub hitMask: u64,
    pub primePower: u64,
}

#[derive(Debug, Clone, Default)]
pub struct SerialState {
    pub nextJobID: u32,
    pub params: ZSTD_CCtx_params,
}

#[derive(Debug, Clone, Default)]
pub struct ZSTDMT_jobDescription {
    pub consumed: usize,
    pub cSize: usize,
    pub dstBuff: Buffer,
    pub srcBuff: Buffer,
    pub prefix: Range,
    pub src: Range,
    pub jobID: u32,
    pub firstJob: u32,
    pub lastJob: u32,
    pub dstFlushed: usize,
    pub frameChecksumNeeded: u32,
    pub params: ZSTD_CCtx_params,
    pub fullFrameSize: u64,
}

impl Buffer {
    #[inline]
    pub fn start(&self) -> usize {
        self.data.as_ptr() as usize
    }
}

impl Range {
    #[inline]
    pub fn end(&self) -> usize {
        self.start + self.size
    }
}

#[inline]
fn range_as_slice(range: Range) -> &'static [u8] {
    if range.size == 0 {
        &[]
    } else {
        // `Range` always points into buffers owned by the MT context.
        unsafe { core::slice::from_raw_parts(range.start as *const u8, range.size) }
    }
}

pub struct ZSTDMT_CCtx {
    _priv: PhantomData<()>,
    pub jobs: Vec<ZSTDMT_jobDescription>,
    pub bufPool: Option<Box<ZSTDMT_bufferPool>>,
    pub cctxPool: Option<Box<ZSTDMT_CCtxPool>>,
    pub seqPool: Option<Box<ZSTDMT_seqPool>>,
    pub params: ZSTD_CCtx_params,
    pub targetSectionSize: usize,
    pub targetPrefixSize: usize,
    pub jobReady: u32,
    pub inBuff: InBuff_t,
    pub roundBuff: RoundBuff_t,
    pub serial: SerialState,
    pub rsync: RSyncState_t,
    pub jobIDMask: u32,
    pub doneJobID: u32,
    pub nextJobID: u32,
    pub frameEnded: u32,
    pub allJobsCompleted: u32,
    pub frameContentSize: u64,
    pub consumed: u64,
    pub produced: u64,
    pub cMem: ZSTD_customMem,
}

impl Default for ZSTDMT_CCtx {
    fn default() -> Self {
        Self {
            _priv: PhantomData,
            jobs: Vec::new(),
            bufPool: None,
            cctxPool: None,
            seqPool: None,
            params: ZSTD_CCtx_params::default(),
            targetSectionSize: ZSTD_BLOCKSIZE_MAX,
            targetPrefixSize: 0,
            jobReady: 0,
            inBuff: InBuff_t::default(),
            roundBuff: RoundBuff_t::default(),
            serial: SerialState::default(),
            rsync: RSyncState_t::default(),
            jobIDMask: 0,
            doneJobID: 0,
            nextJobID: 0,
            frameEnded: 0,
            allJobsCompleted: 1,
            frameContentSize: 0,
            consumed: 0,
            produced: 0,
            cMem: ZSTD_customMem,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Buffer {
    pub data: Vec<u8>,
}

impl Buffer {
    #[inline]
    pub fn capacity(&self) -> usize {
        self.data.len()
    }
}

#[derive(Debug, Clone)]
pub struct ZSTDMT_bufferPool {
    pub bufferSize: usize,
    pub totalBuffers: usize,
    pub buffers: Vec<Buffer>,
    pub _cMem: ZSTD_customMem,
}

pub type ZSTDMT_seqPool = ZSTDMT_bufferPool;

#[derive(Debug)]
pub struct ZSTDMT_CCtxPool {
    pub totalCCtx: usize,
    pub cctxs: Vec<Box<ZSTD_CCtx>>,
    pub _cMem: ZSTD_customMem,
}

/// Port of `GetCurrentClockTimeMicroseconds`.
///
/// Upstream uses `times(2)` / `_SC_CLK_TCK` and returns a monotonic-ish
/// process clock in microseconds for mutex-wait diagnostics. Rust's
/// `Instant` gives us the same practical property for the debug-only
/// timing callsites that will eventually use this helper.
pub fn GetCurrentClockTimeMicroseconds() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_micros() as u64
}

/// Port of `ZSTDMT_freeBufferPool`.
pub fn ZSTDMT_freeBufferPool(bufPool: Option<Box<ZSTDMT_bufferPool>>) {
    if let Some(mut bufPool) = bufPool {
        bufPool.buffers.clear();
        bufPool.buffers.shrink_to(0);
        bufPool.bufferSize = 0;
        bufPool.totalBuffers = 0;
    }
}

/// Port of `ZSTDMT_createBufferPool`.
pub fn ZSTDMT_createBufferPool(
    maxNbBuffers: u32,
    cMem: ZSTD_customMem,
) -> Option<Box<ZSTDMT_bufferPool>> {
    Some(Box::new(ZSTDMT_bufferPool {
        bufferSize: 64 * 1024,
        totalBuffers: maxNbBuffers as usize,
        buffers: Vec::new(),
        _cMem: cMem,
    }))
}

/// Port of `ZSTDMT_sizeof_bufferPool`.
pub fn ZSTDMT_sizeof_bufferPool(bufPool: &ZSTDMT_bufferPool) -> usize {
    core::mem::size_of::<ZSTDMT_bufferPool>()
        + bufPool.buffers.capacity() * core::mem::size_of::<Buffer>()
        + bufPool.buffers.iter().map(Buffer::capacity).sum::<usize>()
}

/// Port of `ZSTDMT_setBufferSize`.
pub fn ZSTDMT_setBufferSize(bufPool: &mut ZSTDMT_bufferPool, bSize: usize) {
    bufPool.bufferSize = bSize;
}

/// Port of `ZSTDMT_expandBufferPool`.
pub fn ZSTDMT_expandBufferPool(
    mut srcBufPool: Box<ZSTDMT_bufferPool>,
    maxNbBuffers: u32,
) -> Option<Box<ZSTDMT_bufferPool>> {
    if srcBufPool.totalBuffers < maxNbBuffers as usize {
        srcBufPool.totalBuffers = maxNbBuffers as usize;
    }
    Some(srcBufPool)
}

/// Port of `ZSTDMT_getBuffer`.
pub fn ZSTDMT_getBuffer(bufPool: &mut ZSTDMT_bufferPool) -> Buffer {
    let bSize = bufPool.bufferSize;
    if let Some(pos) = bufPool
        .buffers
        .iter()
        .position(|buf| buf.capacity() >= bSize && (buf.capacity() >> 3) <= bSize)
    {
        return bufPool.buffers.swap_remove(pos);
    }
    Buffer { data: vec![0; bSize] }
}

/// Port of `ZSTDMT_resizeBuffer`.
pub fn ZSTDMT_resizeBuffer(bufPool: &ZSTDMT_bufferPool, mut buffer: Buffer) -> Buffer {
    if buffer.capacity() < bufPool.bufferSize {
        buffer.data.resize(bufPool.bufferSize, 0);
    }
    buffer
}

/// Port of `ZSTDMT_releaseBuffer`.
pub fn ZSTDMT_releaseBuffer(bufPool: &mut ZSTDMT_bufferPool, buf: Buffer) {
    if buf.capacity() == 0 {
        return;
    }
    if bufPool.buffers.len() < bufPool.totalBuffers {
        bufPool.buffers.push(buf);
    }
}

/// Port of `ZSTDMT_sizeof_seqPool`.
pub fn ZSTDMT_sizeof_seqPool(seqPool: &ZSTDMT_seqPool) -> usize {
    ZSTDMT_sizeof_bufferPool(seqPool)
}

/// Port of `bufferToSeq`.
pub fn bufferToSeq(buffer: Buffer) -> RawSeqStore_t {
    let capacity = buffer.capacity() / core::mem::size_of::<rawSeq>();
    RawSeqStore_t::with_capacity(capacity)
}

/// Port of `seqToBuffer`.
pub fn seqToBuffer(seq: RawSeqStore_t) -> Buffer {
    Buffer {
        data: vec![0; seq.capacity * core::mem::size_of::<rawSeq>()],
    }
}

/// Port of `ZSTDMT_getSeq`.
pub fn ZSTDMT_getSeq(seqPool: &mut ZSTDMT_seqPool) -> RawSeqStore_t {
    if seqPool.bufferSize == 0 {
        RawSeqStore_t::default()
    } else {
        bufferToSeq(ZSTDMT_getBuffer(seqPool))
    }
}

/// Port of `ZSTDMT_resizeSeq`.
pub fn ZSTDMT_resizeSeq(seqPool: &ZSTDMT_seqPool, seq: RawSeqStore_t) -> RawSeqStore_t {
    bufferToSeq(ZSTDMT_resizeBuffer(seqPool, seqToBuffer(seq)))
}

/// Port of `ZSTDMT_releaseSeq`.
pub fn ZSTDMT_releaseSeq(seqPool: &mut ZSTDMT_seqPool, seq: RawSeqStore_t) {
    ZSTDMT_releaseBuffer(seqPool, seqToBuffer(seq));
}

/// Port of `ZSTDMT_setNbSeq`.
pub fn ZSTDMT_setNbSeq(seqPool: &mut ZSTDMT_seqPool, nbSeq: usize) {
    ZSTDMT_setBufferSize(seqPool, nbSeq * core::mem::size_of::<rawSeq>());
}

/// Port of `ZSTDMT_createSeqPool`.
pub fn ZSTDMT_createSeqPool(
    nbWorkers: u32,
    cMem: ZSTD_customMem,
) -> Option<Box<ZSTDMT_seqPool>> {
    let mut seqPool = ZSTDMT_createBufferPool(nbWorkers, cMem)?;
    ZSTDMT_setNbSeq(&mut seqPool, 0);
    Some(seqPool)
}

/// Port of `ZSTDMT_freeSeqPool`.
pub fn ZSTDMT_freeSeqPool(_seqPool: Option<Box<ZSTDMT_seqPool>>) {}

/// Port of `ZSTDMT_expandSeqPool`.
pub fn ZSTDMT_expandSeqPool(
    pool: Box<ZSTDMT_seqPool>,
    nbWorkers: u32,
) -> Option<Box<ZSTDMT_seqPool>> {
    ZSTDMT_expandBufferPool(pool, nbWorkers)
}

/// Port of `ZSTDMT_freeCCtxPool`.
pub fn ZSTDMT_freeCCtxPool(pool: Option<Box<ZSTDMT_CCtxPool>>) {
    if let Some(mut pool) = pool {
        pool.cctxs.clear();
        pool.cctxs.shrink_to(0);
        pool.totalCCtx = 0;
    }
}

/// Port of `ZSTDMT_createCCtxPool`.
pub fn ZSTDMT_createCCtxPool(
    nbWorkers: i32,
    cMem: ZSTD_customMem,
) -> Option<Box<ZSTDMT_CCtxPool>> {
    if nbWorkers <= 0 {
        return None;
    }
    let mut cctxs = Vec::new();
    if let Some(cctx) = ZSTD_createCCtx_advanced(cMem) {
        cctxs.push(cctx);
    } else {
        return None;
    }
    Some(Box::new(ZSTDMT_CCtxPool {
        totalCCtx: nbWorkers as usize,
        cctxs,
        _cMem: cMem,
    }))
}

/// Port of `ZSTDMT_expandCCtxPool`.
pub fn ZSTDMT_expandCCtxPool(
    mut srcPool: Box<ZSTDMT_CCtxPool>,
    nbWorkers: i32,
) -> Option<Box<ZSTDMT_CCtxPool>> {
    if nbWorkers > srcPool.totalCCtx as i32 {
        srcPool.totalCCtx = nbWorkers as usize;
    }
    Some(srcPool)
}

/// Port of `ZSTDMT_sizeof_CCtxPool`.
pub fn ZSTDMT_sizeof_CCtxPool(cctxPool: &ZSTDMT_CCtxPool) -> usize {
    core::mem::size_of::<ZSTDMT_CCtxPool>()
        + cctxPool.cctxs.capacity() * core::mem::size_of::<Box<ZSTD_CCtx>>()
        + cctxPool.cctxs.iter().map(|c| ZSTD_sizeof_CCtx(c)).sum::<usize>()
}

/// Port of `ZSTDMT_getCCtx`.
pub fn ZSTDMT_getCCtx(cctxPool: &mut ZSTDMT_CCtxPool) -> Option<Box<ZSTD_CCtx>> {
    if let Some(cctx) = cctxPool.cctxs.pop() {
        Some(cctx)
    } else {
        ZSTD_createCCtx_advanced(ZSTD_customMem)
    }
}

/// Port of `ZSTDMT_releaseCCtx`.
pub fn ZSTDMT_releaseCCtx(pool: &mut ZSTDMT_CCtxPool, cctx: Option<Box<ZSTD_CCtx>>) {
    if let Some(cctx) = cctx {
        if pool.cctxs.len() < pool.totalCCtx {
            pool.cctxs.push(cctx);
        }
    }
}

/// Port of `ZSTDMT_serialState_reset`.
pub fn ZSTDMT_serialState_reset(
    serialState: &mut SerialState,
    seqPool: &mut ZSTDMT_seqPool,
    params: ZSTD_CCtx_params,
    jobSize: usize,
    _dict: Option<&[u8]>,
) -> i32 {
    if params.ldmParams.enableLdm == ZSTD_ParamSwitch_e::ZSTD_ps_enable {
        ZSTDMT_setNbSeq(seqPool, ZSTD_ldm_getMaxNbSeq(params.ldmParams, jobSize));
    }
    serialState.nextJobID = 0;
    serialState.params = params;
    0
}

/// Port of `ZSTDMT_serialState_init`.
pub fn ZSTDMT_serialState_init(serialState: &mut SerialState) -> i32 {
    *serialState = SerialState::default();
    0
}

/// Port of `ZSTDMT_serialState_free`.
pub fn ZSTDMT_serialState_free(serialState: &mut SerialState) {
    serialState.nextJobID = 0;
    serialState.params = ZSTD_CCtx_params::default();
}

/// Port of `ZSTDMT_serialState_genSequences`.
pub fn ZSTDMT_serialState_genSequences(
    serialState: &mut SerialState,
    _seqStore: &mut RawSeqStore_t,
    _src: Range,
    jobID: u32,
) {
    if serialState.nextJobID == jobID {
        serialState.nextJobID = serialState.nextJobID.wrapping_add(1);
    } else if serialState.nextJobID < jobID {
        serialState.nextJobID = jobID.wrapping_add(1);
    }
}

/// Port of `ZSTDMT_serialState_applySequences`.
pub fn ZSTDMT_serialState_applySequences(
    serialState: &SerialState,
    jobCCtx: &mut ZSTD_CCtx,
    seqStore: &RawSeqStore_t,
) {
    if seqStore.size > 0 {
        debug_assert!(serialState.params.ldmParams.enableLdm == ZSTD_ParamSwitch_e::ZSTD_ps_enable);
        let _ = ZSTD_referenceExternalSequences(jobCCtx, Some(&seqStore.seq[..seqStore.size]));
    }
}

/// Port of `ZSTDMT_serialState_ensureFinished`.
pub fn ZSTDMT_serialState_ensureFinished(
    serialState: &mut SerialState,
    jobID: u32,
    cSize: usize,
) {
    if crate::common::error::ERR_isError(cSize) && serialState.nextJobID <= jobID {
        serialState.nextJobID = jobID + 1;
    }
}

/// Port of `ZSTDMT_freeJobsTable`.
pub fn ZSTDMT_freeJobsTable(jobTable: &mut Vec<ZSTDMT_jobDescription>, _nbJobs: u32, _cMem: ZSTD_customMem) {
    jobTable.clear();
}

/// Port of `ZSTDMT_createJobsTable`.
pub fn ZSTDMT_createJobsTable(nbJobsPtr: &mut u32, _cMem: ZSTD_customMem) -> Option<Vec<ZSTDMT_jobDescription>> {
    let nbJobsLog2 = 32 - nbJobsPtr.saturating_sub(1).leading_zeros();
    let nbJobs = 1u32 << nbJobsLog2;
    *nbJobsPtr = nbJobs.max(1);
    Some(vec![ZSTDMT_jobDescription::default(); *nbJobsPtr as usize])
}

/// Port of `ZSTDMT_expandJobsTable`.
pub fn ZSTDMT_expandJobsTable(mtctx: &mut ZSTDMT_CCtx, nbWorkers: u32) -> usize {
    let mut nbJobs = nbWorkers + 2;
    if nbJobs > mtctx.jobIDMask + 1 {
        mtctx.jobs = ZSTDMT_createJobsTable(&mut nbJobs, mtctx.cMem).unwrap_or_default();
        mtctx.jobIDMask = nbJobs.saturating_sub(1);
    }
    0
}

/// Port of `ZSTDMT_CCtxParam_setNbWorkers`.
pub fn ZSTDMT_CCtxParam_setNbWorkers(params: &mut ZSTD_CCtx_params, nbWorkers: u32) -> usize {
    ZSTD_CCtxParams_setParameter(params, ZSTD_cParameter::ZSTD_c_nbWorkers, nbWorkers as i32)
}

/// Port of `ZSTDMT_getFrameProgression`.
pub fn ZSTDMT_getFrameProgression(mtctx: &ZSTDMT_CCtx) -> ZSTD_frameProgression {
    use crate::common::error::ERR_isError;

    let mut fps = ZSTD_frameProgression {
        ingested: mtctx.consumed + mtctx.inBuff.filled as u64,
        consumed: mtctx.consumed,
        produced: mtctx.produced,
        flushed: mtctx.produced,
        currentJobID: mtctx.nextJobID,
        nbActiveWorkers: 0,
    };
    let lastJobNb = mtctx.nextJobID + mtctx.jobReady;
    for jobNb in mtctx.doneJobID..lastJobNb {
        let wJobID = (jobNb & mtctx.jobIDMask) as usize;
        if let Some(jobPtr) = mtctx.jobs.get(wJobID) {
            let produced = if ERR_isError(jobPtr.cSize) { 0 } else { jobPtr.cSize };
            let flushed = if ERR_isError(jobPtr.cSize) {
                0
            } else {
                debug_assert!(jobPtr.dstFlushed <= produced);
                jobPtr.dstFlushed
            };
            fps.ingested += jobPtr.src.size as u64;
            fps.consumed += jobPtr.consumed as u64;
            fps.produced += produced as u64;
            fps.flushed += flushed as u64;
            fps.nbActiveWorkers += (jobPtr.consumed < jobPtr.src.size) as u32;
        }
    }
    fps
}

/// Port of `ZSTDMT_computeTargetJobLog`.
pub fn ZSTDMT_computeTargetJobLog(params: &ZSTD_CCtx_params) -> u32 {
    let jobLog = if params.ldmParams.enableLdm == ZSTD_ParamSwitch_e::ZSTD_ps_enable {
        core::cmp::max(21, ZSTD_cycleLog(params.cParams.chainLog, params.cParams.strategy) + 3)
    } else {
        core::cmp::max(20, params.cParams.windowLog + 2)
    };
    core::cmp::min(jobLog, ZSTDMT_JOBLOG_MAX)
}

/// Port of `ZSTDMT_overlapLog_default`.
pub fn ZSTDMT_overlapLog_default(strat: u32) -> i32 {
    match strat {
        9 => 9,
        7 | 8 => 8,
        5 | 6 => 7,
        _ => 6,
    }
}

/// Port of `ZSTDMT_overlapLog`.
pub fn ZSTDMT_overlapLog(ovlog: i32, strat: u32) -> i32 {
    debug_assert!((0..=9).contains(&ovlog));
    if ovlog == 0 { ZSTDMT_overlapLog_default(strat) } else { ovlog }
}

/// Port of `ZSTDMT_computeOverlapSize`.
pub fn ZSTDMT_computeOverlapSize(params: &ZSTD_CCtx_params) -> usize {
    let overlapRLog = 9 - ZSTDMT_overlapLog(params.overlapLog, params.cParams.strategy);
    let mut ovLog = if overlapRLog >= 8 { 0 } else { params.cParams.windowLog as i32 - overlapRLog };
    if params.ldmParams.enableLdm == ZSTD_ParamSwitch_e::ZSTD_ps_enable {
        ovLog = core::cmp::min(params.cParams.windowLog, ZSTDMT_computeTargetJobLog(params) - 2) as i32 - overlapRLog;
    }
    if ovLog <= 0 { 0 } else { 1usize << ovLog.min(ZSTD_WINDOWLOG_MAX() as i32) }
}

/// Port of `ZSTDMT_releaseAllJobResources`.
pub fn ZSTDMT_releaseAllJobResources(mtctx: &mut ZSTDMT_CCtx) {
    for job in &mut mtctx.jobs {
        if let Some(bufPool) = mtctx.bufPool.as_mut() {
            ZSTDMT_releaseBuffer(bufPool, core::mem::take(&mut job.dstBuff));
        }
        *job = ZSTDMT_jobDescription::default();
    }
    mtctx.inBuff.buffer = Buffer::default();
    mtctx.inBuff.filled = 0;
    mtctx.allJobsCompleted = 1;
}

/// Port of `ZSTDMT_waitForAllJobsCompleted`.
pub fn ZSTDMT_waitForAllJobsCompleted(mtctx: &mut ZSTDMT_CCtx) {
    mtctx.doneJobID = mtctx.nextJobID;
}

/// Port of `ZSTDMT_resize`.
pub fn ZSTDMT_resize(mtctx: &mut ZSTDMT_CCtx, nbWorkers: u32) -> usize {
    ZSTDMT_expandJobsTable(mtctx, nbWorkers);
    if let Some(pool) = mtctx.bufPool.take() {
        mtctx.bufPool = ZSTDMT_expandBufferPool(pool, 2 * nbWorkers + 3);
    }
    if let Some(pool) = mtctx.cctxPool.take() {
        mtctx.cctxPool = ZSTDMT_expandCCtxPool(pool, nbWorkers as i32);
    }
    if let Some(pool) = mtctx.seqPool.take() {
        mtctx.seqPool = ZSTDMT_expandSeqPool(pool, nbWorkers);
    }
    ZSTDMT_CCtxParam_setNbWorkers(&mut mtctx.params, nbWorkers)
}

/// Port of `ZSTDMT_updateCParams_whileCompressing`.
pub fn ZSTDMT_updateCParams_whileCompressing(mtctx: &mut ZSTDMT_CCtx, cctxParams: &ZSTD_CCtx_params) {
    use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;

    let saved_wlog = mtctx.params.cParams.windowLog;
    let mut cParams = ZSTD_getCParamsFromCCtxParams(
        cctxParams,
        ZSTD_CONTENTSIZE_UNKNOWN,
        0,
        ZSTD_CParamMode_e::ZSTD_cpm_noAttachDict,
    );
    mtctx.params.compressionLevel = cctxParams.compressionLevel;
    cParams.windowLog = saved_wlog;
    mtctx.params.cParams = cParams;
}

/// Port of `ZSTDMT_isOverlapped`.
pub fn ZSTDMT_isOverlapped(buffer: &Buffer, range: Range) -> i32 {
    let bufferStart = buffer.start();
    let bufferEnd = bufferStart + buffer.capacity();
    if range.start == 0 || bufferStart == 0 || range.size == 0 || buffer.capacity() == 0 {
        return 0;
    }
    ((bufferStart < range.end()) && (range.start < bufferEnd)) as i32
}

/// Port of `ZSTDMT_getInputDataInUse`.
pub fn ZSTDMT_getInputDataInUse(mtctx: &ZSTDMT_CCtx) -> Range {
    let roundBuffCapacity = mtctx.roundBuff.capacity;
    if mtctx.targetSectionSize == 0 {
        return Range::default();
    }
    let nbJobs1stRoundMin = roundBuffCapacity / mtctx.targetSectionSize;
    if mtctx.nextJobID < nbJobs1stRoundMin as u32 {
        return Range::default();
    }
    for jobID in mtctx.doneJobID..mtctx.nextJobID {
        let wJobID = (jobID & mtctx.jobIDMask) as usize;
        if let Some(job) = mtctx.jobs.get(wJobID) {
            if job.consumed < job.src.size {
                return if job.prefix.size == 0 { job.src } else { job.prefix };
            }
        }
    }
    Range::default()
}

/// Port of `ZSTDMT_doesOverlapWindow`.
pub fn ZSTDMT_doesOverlapWindow(buffer: &Buffer, window: Range) -> i32 {
    let extDict = Range {
        start: window.start,
        size: 0,
    };
    let prefix = Range {
        start: window.start,
        size: window.size,
    };
    if ZSTDMT_isOverlapped(buffer, extDict) != 0 {
        return 1;
    }
    ZSTDMT_isOverlapped(buffer, prefix)
}

/// Port of `ZSTDMT_waitForLdmComplete`.
pub fn ZSTDMT_waitForLdmComplete(mtctx: &mut ZSTDMT_CCtx, buffer: &Buffer) {
    if mtctx.params.ldmParams.enableLdm != ZSTD_ParamSwitch_e::ZSTD_ps_enable {
        return;
    }

    let mut attempts = 0u32;
    while attempts < 2 {
        let inUse = ZSTDMT_getInputDataInUse(mtctx);
        if ZSTDMT_doesOverlapWindow(buffer, inUse) == 0 {
            break;
        }
        attempts += 1;
    }
}

/// Port of `ZSTDMT_tryGetInputRange`.
pub fn ZSTDMT_tryGetInputRange(mtctx: &mut ZSTDMT_CCtx) -> i32 {
    let inUse = ZSTDMT_getInputDataInUse(mtctx);
    let spaceLeft = mtctx.roundBuff.capacity.saturating_sub(mtctx.roundBuff.pos);
    let spaceNeeded = mtctx.targetSectionSize;
    if spaceNeeded == 0 || mtctx.roundBuff.buffer.capacity() < spaceNeeded {
        return 0;
    }
    if spaceLeft < spaceNeeded {
        mtctx.roundBuff.pos = mtctx.inBuff.prefix.size;
    }
    let start = mtctx.roundBuff.pos;
    let end = (start + spaceNeeded).min(mtctx.roundBuff.buffer.capacity());
    let buffer = Buffer { data: mtctx.roundBuff.buffer.data[start..end].to_vec() };
    if ZSTDMT_isOverlapped(&buffer, inUse) != 0 {
        return 0;
    }
    mtctx.inBuff.buffer = buffer;
    mtctx.inBuff.filled = 0;
    1
}

/// Port of `findSynchronizationPoint`.
pub fn findSynchronizationPoint(mtctx: &ZSTDMT_CCtx, input: &[u8], input_pos: usize) -> SyncPoint {
    let istart = &input[input_pos..];
    let primePower = mtctx.rsync.primePower;
    let hitMask = mtctx.rsync.hitMask;
    let mut syncPoint = SyncPoint {
        toLoad: core::cmp::min(
            input.len().saturating_sub(input_pos),
            mtctx.targetSectionSize.saturating_sub(mtctx.inBuff.filled),
        ),
        flush: 0,
    };
    if mtctx.params.rsyncable == 0 {
        return syncPoint;
    }
    if mtctx.inBuff.filled + input.len().saturating_sub(input_pos) < RSYNC_MIN_BLOCK_SIZE {
        return syncPoint;
    }
    if mtctx.inBuff.filled + syncPoint.toLoad < RSYNC_LENGTH {
        return syncPoint;
    }

    let buffered = &mtctx.inBuff.buffer.data;
    let (mut pos, mut hash, prev): (usize, u64, &[u8]) = if mtctx.inBuff.filled < RSYNC_MIN_BLOCK_SIZE {
        let pos = RSYNC_MIN_BLOCK_SIZE - mtctx.inBuff.filled;
        if pos >= RSYNC_LENGTH {
            let prev = &istart[pos - RSYNC_LENGTH..pos];
            (pos, ZSTD_rollingHash_compute(prev), prev)
        } else {
            debug_assert!(mtctx.inBuff.filled >= RSYNC_LENGTH);
            let prev = &buffered[mtctx.inBuff.filled - RSYNC_LENGTH..mtctx.inBuff.filled];
            let mut hash = ZSTD_rollingHash_compute(&prev[pos..]);
            hash = ZSTD_rollingHash_append(hash, &istart[..pos]);
            (pos, hash, prev)
        }
    } else {
        debug_assert!(mtctx.inBuff.filled >= RSYNC_LENGTH);
        let prev = &buffered[mtctx.inBuff.filled - RSYNC_LENGTH..mtctx.inBuff.filled];
        let hash = ZSTD_rollingHash_compute(prev);
        if (hash & hitMask) == hitMask {
            syncPoint.toLoad = 0;
            syncPoint.flush = 1;
            return syncPoint;
        }
        (0, hash, prev)
    };

    while pos < syncPoint.toLoad {
        let toRemove = if pos < RSYNC_LENGTH {
            prev[pos]
        } else {
            istart[pos - RSYNC_LENGTH]
        };
        hash = ZSTD_rollingHash_rotate(hash, toRemove, istart[pos], primePower);
        if (hash & hitMask) == hitMask {
            syncPoint.toLoad = pos + 1;
            syncPoint.flush = 1;
            break;
        }
        pos += 1;
    }
    syncPoint
}

/// Port of `ZSTDMT_writeLastEmptyBlock`.
pub fn ZSTDMT_writeLastEmptyBlock(job: &mut ZSTDMT_jobDescription) {
    job.dstBuff = Buffer { data: vec![0; 3] };
    job.cSize = ZSTD_writeLastEmptyBlock(&mut job.dstBuff.data);
}

/// Port of `ZSTDMT_createCompressionJob`.
pub fn ZSTDMT_createCompressionJob(mtctx: &mut ZSTDMT_CCtx, srcSize: usize, endOp: ZSTD_EndDirective) -> usize {
    if mtctx.nextJobID > mtctx.doneJobID + mtctx.jobIDMask {
        return 0;
    }
    let endFrame = endOp == ZSTD_EndDirective::ZSTD_e_end;
    let jobID = (mtctx.nextJobID & mtctx.jobIDMask) as usize;
    if mtctx.jobs.len() <= jobID {
        mtctx.jobs.resize(jobID + 1, ZSTDMT_jobDescription::default());
    }
    let job = &mut mtctx.jobs[jobID];
    let srcBuff = core::mem::take(&mut mtctx.inBuff.buffer);
    let srcStart = srcBuff.start();
    job.src = Range { start: mtctx.inBuff.buffer.start(), size: srcSize };
    job.src = Range { start: srcStart, size: srcSize };
    job.srcBuff = srcBuff;
    job.prefix = mtctx.inBuff.prefix;
    job.consumed = 0;
    job.cSize = 0;
    job.jobID = mtctx.nextJobID;
    job.firstJob = (mtctx.nextJobID == 0) as u32;
    job.lastJob = endFrame as u32;
    job.dstFlushed = 0;
    job.params = mtctx.params;
    job.fullFrameSize = mtctx.frameContentSize;
    job.frameChecksumNeeded =
        (mtctx.params.fParams.checksumFlag != 0 && endFrame && mtctx.nextJobID > 0) as u32;
    job.dstBuff = Buffer::default();
    if srcSize == 0 && mtctx.nextJobID > 0 && endFrame {
        ZSTDMT_writeLastEmptyBlock(job);
        }
    mtctx.roundBuff.pos += srcSize;
    mtctx.inBuff.filled = 0;
    if !endFrame {
        let newPrefixSize = srcSize.min(mtctx.targetPrefixSize);
        let start = job.src.start + srcSize.saturating_sub(newPrefixSize);
        mtctx.inBuff.prefix = Range {
            start,
            size: newPrefixSize,
        };
    } else {
        mtctx.inBuff.prefix = Range::default();
        mtctx.frameEnded = 1;
        if mtctx.nextJobID == 0 {
            mtctx.params.fParams.checksumFlag = 0;
        }
    }
    mtctx.nextJobID += 1;
    0
}

/// Port of `ZSTDMT_flushProduced`.
pub fn ZSTDMT_flushProduced(
    mtctx: &mut ZSTDMT_CCtx,
    output: &mut [u8],
    output_pos: &mut usize,
    _blockToFlush: u32,
    end: ZSTD_EndDirective,
) -> usize {
    use crate::common::error::ERR_isError;

    if mtctx.doneJobID < mtctx.nextJobID {
        let wJobID = (mtctx.doneJobID & mtctx.jobIDMask) as usize;
        let mut job_completed = false;
        let (src_size, c_size);

        if let Some(job) = mtctx.jobs.get_mut(wJobID) {
            let c_result = job.cSize;
            if ERR_isError(c_result) {
                ZSTDMT_waitForAllJobsCompleted(mtctx);
                ZSTDMT_releaseAllJobResources(mtctx);
                return c_result;
            }

            c_size = c_result;
            src_size = job.src.size;
            let toFlush = core::cmp::min(
                c_size.saturating_sub(job.dstFlushed),
                output.len().saturating_sub(*output_pos),
            );
            if toFlush > 0 && !job.dstBuff.data.is_empty() {
                output[*output_pos..*output_pos + toFlush]
                    .copy_from_slice(&job.dstBuff.data[job.dstFlushed..job.dstFlushed + toFlush]);
                *output_pos += toFlush;
                job.dstFlushed += toFlush;
            }
            if job.consumed == src_size && job.dstFlushed == c_size {
                job_completed = true;
            }
            if c_size > job.dstFlushed {
                return c_size - job.dstFlushed;
            }
            if src_size > job.consumed {
                return 1;
            }
        } else {
            return 0;
        }

        if job_completed {
            if let Some(job) = mtctx.jobs.get_mut(wJobID) {
                let dst = core::mem::take(&mut job.dstBuff);
                if let Some(bufPool) = mtctx.bufPool.as_mut() {
                    ZSTDMT_releaseBuffer(bufPool, dst);
                }
                job.cSize = 0;
            }
            mtctx.consumed += src_size as u64;
            mtctx.produced += c_size as u64;
            mtctx.doneJobID += 1;
        }
    }
    if mtctx.doneJobID < mtctx.nextJobID {
        return 1;
    }
    if mtctx.jobReady != 0 {
        return 1;
    }
    if mtctx.inBuff.filled > 0 {
        return 1;
    }
    mtctx.allJobsCompleted = mtctx.frameEnded;
    if end == ZSTD_EndDirective::ZSTD_e_end {
        (mtctx.frameEnded == 0) as usize
    } else {
        0
    }
}

/// Port of `ZSTDMT_createCCtx_advanced_internal`.
pub fn ZSTDMT_createCCtx_advanced_internal(nbWorkers: u32, cMem: ZSTD_customMem) -> Option<Box<ZSTDMT_CCtx>> {
    if nbWorkers < 1 {
        return None;
    }
    let mut nbJobs = nbWorkers + 2;
    let jobs = ZSTDMT_createJobsTable(&mut nbJobs, cMem)?;
    let mut mtctx = Box::new(ZSTDMT_CCtx::default());
    mtctx.params.nbWorkers = nbWorkers as i32;
    mtctx.cMem = cMem;
    mtctx.jobs = jobs;
    mtctx.jobIDMask = nbJobs - 1;
    mtctx.bufPool = ZSTDMT_createBufferPool(2 * nbWorkers + 3, cMem);
    mtctx.cctxPool = ZSTDMT_createCCtxPool(nbWorkers as i32, cMem);
    mtctx.seqPool = ZSTDMT_createSeqPool(nbWorkers, cMem);
    Some(mtctx)
}

/// Port of `ZSTDMT_initCStream_internal`.
pub fn ZSTDMT_initCStream_internal(
    mtctx: &mut ZSTDMT_CCtx,
    mut params: ZSTD_CCtx_params,
    pledgedSrcSize: u64,
) -> usize {
    use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;

    params.cParams = ZSTD_getCParamsFromCCtxParams(
        &params,
        pledgedSrcSize,
        0,
        ZSTD_CParamMode_e::ZSTD_cpm_noAttachDict,
    );
    if pledgedSrcSize == ZSTD_CONTENTSIZE_UNKNOWN {
        params.cParams.windowLog = params.cParams.windowLog.max(20);
    }
    if params.nbWorkers != mtctx.params.nbWorkers {
        ZSTDMT_resize(mtctx, params.nbWorkers as u32);
    }
    mtctx.params = params;
    mtctx.frameContentSize = pledgedSrcSize;
    mtctx.targetPrefixSize = ZSTDMT_computeOverlapSize(&mtctx.params);
    mtctx.targetSectionSize = if mtctx.params.jobSize == 0 {
        1usize << ZSTDMT_computeTargetJobLog(&mtctx.params)
    } else {
        mtctx.params.jobSize
    };
    if mtctx.targetSectionSize < mtctx.targetPrefixSize {
        mtctx.targetSectionSize = mtctx.targetPrefixSize;
    }
    if let Some(bufPool) = mtctx.bufPool.as_mut() {
        ZSTDMT_setBufferSize(bufPool, crate::compress::zstd_compress::ZSTD_compressBound(mtctx.targetSectionSize));
    }
    0
}

/// Port of `ZSTDMT_compressionJob`.
pub fn ZSTDMT_compressionJob(job: &mut ZSTDMT_jobDescription) {
    let mut jobParams = job.params;
    let mut rawSeqStore = if jobParams.ldmParams.enableLdm == ZSTD_ParamSwitch_e::ZSTD_ps_enable {
        RawSeqStore_t::with_capacity(ZSTD_ldm_getMaxNbSeq(jobParams.ldmParams, job.src.size))
    } else {
        RawSeqStore_t::default()
    };
    let mut lastCBlockSize = 0usize;

    if job.jobID != 0 {
        jobParams.fParams.checksumFlag = 0;
    }
    jobParams.ldmEnable = ZSTD_ParamSwitch_e::ZSTD_ps_disable;
    jobParams.ldmParams.enableLdm = ZSTD_ParamSwitch_e::ZSTD_ps_disable;
    jobParams.nbWorkers = 0;
    jobParams.extSeqProdState = 0;
    jobParams.extSeqProdFunc = None;

    if job.lastJob != 0 && job.src.size == 0 && job.jobID > 0 {
        if job.dstBuff.capacity() < 3 {
            job.dstBuff = Buffer { data: vec![0; 3] };
        }
        job.cSize = ZSTD_writeLastEmptyBlock(&mut job.dstBuff.data);
        job.consumed = job.src.size;
        return;
    }

    if job.dstBuff.capacity() == 0 {
        let bound = ZSTD_compressBound(job.src.size.max(1));
        job.dstBuff = Buffer { data: vec![0; bound] };
    }

    let mut cctx = match ZSTD_createCCtx_advanced(jobParams.customMem) {
        Some(cctx) => cctx,
        None => {
            job.cSize = crate::common::error::ERROR(crate::common::error::ErrorCode::MemoryAllocation);
            return;
        }
    };

    let src = range_as_slice(job.src);
    let prefix = range_as_slice(job.prefix);
    ZSTDMT_serialState_genSequences(&mut SerialState { nextJobID: job.jobID, params: job.params }, &mut rawSeqStore, job.src, job.jobID);

    let pledgedSrcSize = if job.firstJob != 0 {
        job.fullFrameSize
    } else {
        job.src.size as u64
    };
    let initError = ZSTD_compressBegin_advanced_internal(
        &mut cctx,
        prefix,
        ZSTD_dictContentType_e::ZSTD_dct_rawContent,
        None,
        &jobParams,
        pledgedSrcSize,
    );
    if crate::common::error::ERR_isError(initError) {
        job.cSize = initError;
        return;
    }

    // The current MT shim handles pre-generated raw sequences and
    // disables in-worker LDM itself. Keep the worker CCtx aligned
    // with that model even if begin/reset helpers preserved caller-
    // level advanced knobs.
    cctx.requestedParams.ldmEnable = ZSTD_ParamSwitch_e::ZSTD_ps_disable;
    cctx.appliedParams.ldmEnable = ZSTD_ParamSwitch_e::ZSTD_ps_disable;
    cctx.requestedParams.ldmParams.enableLdm = ZSTD_ParamSwitch_e::ZSTD_ps_disable;
    cctx.appliedParams.ldmParams.enableLdm = ZSTD_ParamSwitch_e::ZSTD_ps_disable;
    cctx.requestedParams.extSeqProdState = 0;
    cctx.appliedParams.extSeqProdState = 0;
    cctx.requestedParams.extSeqProdFunc = None;
    cctx.appliedParams.extSeqProdFunc = None;

    ZSTDMT_serialState_applySequences(
        &SerialState { nextJobID: job.jobID + 1, params: job.params },
        &mut cctx,
        &rawSeqStore,
    );

    let mut op = 0usize;
    if job.firstJob == 0 {
        let hSize = ZSTD_compressContinue_public(&mut cctx, &mut job.dstBuff.data[op..], &[]);
        if crate::common::error::ERR_isError(hSize) {
            job.cSize = hSize;
            return;
        }
        op += hSize;
        ZSTD_invalidateRepCodes(&mut cctx);
    }

    let chunkSize = 4 * ZSTD_BLOCKSIZE_MAX;
    let nbChunks = src.len().div_ceil(chunkSize);
    for chunkNb in 1..nbChunks {
        let start = (chunkNb - 1) * chunkSize;
        let end = start + chunkSize;
        let cSize = ZSTD_compressContinue_public(&mut cctx, &mut job.dstBuff.data[op..], &src[start..end]);
        if crate::common::error::ERR_isError(cSize) {
            job.cSize = cSize;
            return;
        }
        op += cSize;
        job.cSize += cSize;
        job.consumed = end;
    }

    if nbChunks > 0 || job.lastJob != 0 {
        let start = chunkSize.saturating_mul(nbChunks.saturating_sub(1));
        let lastSrc = &src[start..];
        lastCBlockSize = if job.lastJob != 0 {
            ZSTD_compressEnd_public(&mut cctx, &mut job.dstBuff.data[op..], lastSrc)
        } else {
            ZSTD_compressContinue_public(&mut cctx, &mut job.dstBuff.data[op..], lastSrc)
        };
        if crate::common::error::ERR_isError(lastCBlockSize) {
            job.cSize = lastCBlockSize;
            return;
        }
    }

    ZSTD_CCtx_trace(&mut cctx, 0);
    job.cSize += lastCBlockSize;
    job.consumed = job.src.size;
}

/// Port of `ZSTDMT_createCCtx`. Allocates the MT context header and
/// its owned job/pool scaffolding; the actual parallel compression
/// pipeline remains stubbed separately.
pub fn ZSTDMT_createCCtx(nbWorkers: u32) -> Option<Box<ZSTDMT_CCtx>> {
    ZSTDMT_createCCtx_advanced_internal(nbWorkers, ZSTD_customMem)
}

/// Port of `ZSTDMT_createCCtx_advanced`. The upstream internal helper
/// also accepts an optional external thread-pool pointer; the current
/// Rust MT shim owns its synchronous execution model and therefore
/// only exposes the public `(nbWorkers, customMem)` surface.
pub fn ZSTDMT_createCCtx_advanced(
    nbWorkers: u32,
    cMem: ZSTD_customMem,
) -> Option<Box<ZSTDMT_CCtx>> {
    ZSTDMT_createCCtx_advanced_internal(nbWorkers, cMem)
}

/// Port of `ZSTDMT_freeCCtx`. Drops the Box; returns 0.
pub fn ZSTDMT_freeCCtx(mtctx: Option<Box<ZSTDMT_CCtx>>) -> usize {
    let Some(mut mtctx) = mtctx else {
        return 0;
    };
    ZSTDMT_releaseAllJobResources(&mut mtctx);
    let nbJobs = mtctx.jobIDMask + 1;
    ZSTDMT_freeJobsTable(&mut mtctx.jobs, nbJobs, mtctx.cMem);
    ZSTDMT_freeBufferPool(mtctx.bufPool.take());
    ZSTDMT_freeCCtxPool(mtctx.cctxPool.take());
    ZSTDMT_freeSeqPool(mtctx.seqPool.take());
    ZSTDMT_serialState_free(&mut mtctx.serial);
    mtctx.roundBuff.buffer = Buffer::default();
    mtctx.roundBuff.capacity = 0;
    mtctx.roundBuff.pos = 0;
    0
}

/// Port of `ZSTDMT_compressStream_generic`. The Rust port currently
/// executes queued jobs immediately on the caller thread while still
/// using the MT context/job buffering state machine.
pub fn ZSTDMT_compressStream_generic(
    mtctx: &mut ZSTDMT_CCtx,
    output: &mut [u8],
    output_pos: &mut usize,
    input: &[u8],
    input_pos: &mut usize,
    end_op: crate::compress::zstd_compress::ZSTD_EndDirective,
) -> usize {
    use crate::common::error::{ErrorCode, ERR_isError, ERROR};
    use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;

    if *output_pos > output.len() || *input_pos > input.len() {
        return ERROR(ErrorCode::Generic);
    }
    if mtctx.frameEnded != 0 && end_op == ZSTD_EndDirective::ZSTD_e_continue {
        return ERROR(ErrorCode::StageWrong);
    }

    if mtctx.params.cParams.windowLog == 0 {
        let mut params = mtctx.params;
        ZSTD_CCtxParams_init(&mut params, ZSTD_CLEVEL_DEFAULT);
        params.nbWorkers = params.nbWorkers.max(1);
        let init = ZSTDMT_initCStream_internal(mtctx, params, ZSTD_CONTENTSIZE_UNKNOWN);
        if ERR_isError(init) {
            return init;
        }
    }

    let src = &input[*input_pos..];
    let mut effective_end = end_op;

    if !src.is_empty() {
        mtctx.inBuff.buffer = Buffer { data: src.to_vec() };
        mtctx.inBuff.filled = src.len();
        *input_pos = input.len();
    }

    if *input_pos < input.len() && effective_end == ZSTD_EndDirective::ZSTD_e_end {
        effective_end = ZSTD_EndDirective::ZSTD_e_flush;
    }

    let should_create_job = mtctx.jobReady != 0
        || mtctx.inBuff.filled >= mtctx.targetSectionSize
        || (effective_end != ZSTD_EndDirective::ZSTD_e_continue && mtctx.inBuff.filled > 0)
        || (effective_end == ZSTD_EndDirective::ZSTD_e_end && mtctx.frameEnded == 0);

    let queuedJobID = mtctx.nextJobID;
    if should_create_job {
        let create = ZSTDMT_createCompressionJob(mtctx, mtctx.inBuff.filled, effective_end);
        if ERR_isError(create) {
            return create;
        }
        if queuedJobID < mtctx.nextJobID {
            let wJobID = (queuedJobID & mtctx.jobIDMask) as usize;
            if let Some(job) = mtctx.jobs.get_mut(wJobID) {
                ZSTDMT_compressionJob(job);
                if ERR_isError(job.cSize) {
                    return job.cSize;
                }
            } else {
                return ERROR(ErrorCode::Generic);
            }
        }
    }

    let remaining = ZSTDMT_flushProduced(mtctx, output, output_pos, 0, effective_end);
    if *input_pos < input.len() {
        remaining.max(1)
    } else {
        remaining
    }
}

/// Port of `ZSTDMT_sizeof_CCtx`. Walks the MT context's owned
/// allocations in the current Rust port: the job table, owned pools,
/// and round-buffer reservation. Fields not yet modeled here
/// upstream, such as `factory` / `cdictLocal`, are naturally omitted.
#[inline]
pub fn ZSTDMT_sizeof_CCtx(mtctx: &ZSTDMT_CCtx) -> usize {
    core::mem::size_of::<ZSTDMT_CCtx>()
        + mtctx.jobs.capacity() * core::mem::size_of::<ZSTDMT_jobDescription>()
        + mtctx
            .bufPool
            .as_ref()
            .map(|p| ZSTDMT_sizeof_bufferPool(p))
            .unwrap_or(0)
        + mtctx
            .cctxPool
            .as_ref()
            .map(|p| ZSTDMT_sizeof_CCtxPool(p))
            .unwrap_or(0)
        + mtctx
            .seqPool
            .as_ref()
            .map(|p| ZSTDMT_sizeof_seqPool(p))
            .unwrap_or(0)
        + mtctx.roundBuff.capacity
}

/// Port of `ZSTDMT_toFlushNow`.
#[inline]
pub fn ZSTDMT_toFlushNow(mtctx: &ZSTDMT_CCtx) -> usize {
    let jobID = mtctx.doneJobID;
    if jobID == mtctx.nextJobID {
        return 0;
    }

    let wJobID = (jobID & mtctx.jobIDMask) as usize;
    let Some(jobPtr) = mtctx.jobs.get(wJobID) else {
        return 0;
    };
    let cResult = jobPtr.cSize;
    let produced = if crate::common::error::ERR_isError(cResult) {
        0
    } else {
        cResult
    };
    let flushed = if crate::common::error::ERR_isError(cResult) {
        0
    } else {
        jobPtr.dstFlushed
    };
    produced.saturating_sub(flushed)
}

/// Port of `ZSTDMT_nextInputSizeHint`. Suggests the remaining space in
/// the current target section; when the section is exactly full,
/// upstream wraps back to a full-section hint.
#[inline]
pub fn ZSTDMT_nextInputSizeHint(mtctx: &ZSTDMT_CCtx) -> usize {
    let hint = mtctx.targetSectionSize.saturating_sub(mtctx.inBuff.filled);
    if hint == 0 {
        mtctx.targetSectionSize
    } else {
        hint
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zstdmt_api_surface_stubs_behave_safely() {
        // The MT context still executes jobs synchronously, but the
        // API surface should be live and safe to call.
        use crate::compress::zstd_compress::ZSTD_EndDirective;

        // Creator now constructs the owned MT context scaffolding for
        // positive worker counts.
        let ctx = ZSTDMT_createCCtx(1).expect("mt ctx");
        assert_eq!(ctx.params.nbWorkers, 1);
        assert_eq!(ctx.jobIDMask + 1, 4);
        assert!(ctx.bufPool.is_some());
        assert!(ctx.cctxPool.is_some());
        assert!(ctx.seqPool.is_some());
        let advanced = ZSTDMT_createCCtx_advanced(1, ZSTD_customMem).expect("advanced mt ctx");
        assert_eq!(advanced.params.nbWorkers, 1);
        assert!(ZSTDMT_createCCtx(0).is_none());
        assert!(ZSTDMT_createCCtx_advanced(0, ZSTD_customMem).is_none());

        // Free accepts None without panicking.
        assert_eq!(ZSTDMT_freeCCtx(None), 0);

        // Size / flush-now queries work on a default stub ctx.
        let stub = ZSTDMT_CCtx::default();
        assert_eq!(ZSTDMT_sizeof_CCtx(&stub), core::mem::size_of::<ZSTDMT_CCtx>());
        assert_eq!(ZSTDMT_toFlushNow(&stub), 0);
        assert_eq!(ZSTDMT_nextInputSizeHint(&stub), ZSTD_BLOCKSIZE_MAX);

        let live_size = ZSTDMT_sizeof_CCtx(&ctx);
        assert!(live_size > core::mem::size_of::<ZSTDMT_CCtx>());

        let mut params = ZSTD_CCtx_params::default();
        ZSTD_CCtxParams_init(&mut params, 3);
        params.nbWorkers = 1;
        let mut mt = ZSTDMT_createCCtx(1).expect("mt");
        assert_eq!(ZSTDMT_initCStream_internal(&mut mt, params, 1), 0);
        let mut dst = [0u8; 128];
        let mut dp = 0usize;
        let mut sp = 0usize;
        let rc = ZSTDMT_compressStream_generic(
            &mut mt, &mut dst, &mut dp, b"x", &mut sp,
            ZSTD_EndDirective::ZSTD_e_end,
        );
        assert_eq!(rc, 0);
        assert_eq!(sp, 1);
        assert!(dp > 0);
    }

    #[test]
    fn zstdmt_next_input_size_hint_tracks_remaining_section_space() {
        let mut mt = ZSTDMT_CCtx::default();
        mt.targetSectionSize = 1024;
        mt.inBuff.filled = 100;
        assert_eq!(ZSTDMT_nextInputSizeHint(&mt), 924);

        mt.inBuff.filled = 1024;
        assert_eq!(ZSTDMT_nextInputSizeHint(&mt), 1024);
    }

    #[test]
    fn zstdmt_update_cparams_while_compressing_rederives_but_preserves_windowlog() {
        use crate::decompress::zstd_decompress::ZSTD_CONTENTSIZE_UNKNOWN;

        let mut mt = ZSTDMT_CCtx::default();
        mt.params.cParams.windowLog = 23;

        let mut params = ZSTD_CCtx_params::default();
        ZSTD_CCtxParams_init(&mut params, 9);
        params.cParams.windowLog = 18;
        params.cParams.hashLog = 9;
        params.cParams.chainLog = 9;

        ZSTDMT_updateCParams_whileCompressing(&mut mt, &params);

        let mut expected = ZSTD_getCParamsFromCCtxParams(
            &params,
            ZSTD_CONTENTSIZE_UNKNOWN,
            0,
            ZSTD_CParamMode_e::ZSTD_cpm_noAttachDict,
        );
        expected.windowLog = 23;

        assert_eq!(mt.params.compressionLevel, 9);
        assert_eq!(mt.params.cParams.windowLog, 23);
        assert_eq!(mt.params.cParams.hashLog, expected.hashLog);
        assert_eq!(mt.params.cParams.chainLog, expected.chainLog);
        assert_eq!(mt.params.cParams.strategy, expected.strategy);
    }

    #[test]
    fn zstdmt_compress_stream_generic_roundtrips_single_job_frame() {
        use crate::decompress::zstd_decompress::ZSTD_decompress;

        let src = b"the quick brown fox jumps over the lazy dog";
        let mut params = ZSTD_CCtx_params::default();
        ZSTD_CCtxParams_init(&mut params, 3);
        params.nbWorkers = 1;
        let mut mt = ZSTDMT_createCCtx(1).expect("mt");
        assert_eq!(ZSTDMT_initCStream_internal(&mut mt, params, src.len() as u64), 0);

        let mut compressed = vec![0u8; ZSTD_compressBound(src.len())];
        let mut cpos = 0usize;
        let mut spos = 0usize;
        let rem = ZSTDMT_compressStream_generic(
            &mut mt,
            &mut compressed,
            &mut cpos,
            src,
            &mut spos,
            ZSTD_EndDirective::ZSTD_e_end,
        );
        assert_eq!(rem, 0);
        assert_eq!(spos, src.len());

        let mut roundtrip = vec![0u8; src.len()];
        let dsize = ZSTD_decompress(&mut roundtrip, &compressed[..cpos]);
        assert_eq!(dsize, src.len());
        assert_eq!(roundtrip, src);
    }

    #[test]
    fn zstdmt_compress_stream_generic_supports_partial_output_flush() {
        use crate::decompress::zstd_decompress::ZSTD_decompress;

        let src = b"abcdefabcdefabcdefabcdefabcdefabcdef";
        let mut mt = ZSTDMT_createCCtx(1).expect("mt");
        let mut first = [0u8; 8];
        let mut first_pos = 0usize;
        let mut input_pos = 0usize;
        let mut remaining = ZSTDMT_compressStream_generic(
            &mut mt,
            &mut first,
            &mut first_pos,
            src,
            &mut input_pos,
            ZSTD_EndDirective::ZSTD_e_end,
        );
        assert_eq!(input_pos, src.len());
        assert!(remaining > 0);
        assert!(ZSTDMT_toFlushNow(&mt) > 0);

        let mut tail = vec![0u8; remaining];
        let mut tail_pos = 0usize;
        let empty = [];
        let mut empty_pos = 0usize;
        remaining = ZSTDMT_compressStream_generic(
            &mut mt,
            &mut tail,
            &mut tail_pos,
            &empty,
            &mut empty_pos,
            ZSTD_EndDirective::ZSTD_e_end,
        );
        assert_eq!(remaining, 0);

        let mut frame = first[..first_pos].to_vec();
        frame.extend_from_slice(&tail[..tail_pos]);
        let mut roundtrip = vec![0u8; src.len()];
        let dsize = ZSTD_decompress(&mut roundtrip, &frame);
        assert_eq!(dsize, src.len());
        assert_eq!(roundtrip, src);
    }

    #[test]
    fn zstdmt_get_frame_progression_ignores_errored_job_output() {
        use crate::common::error::{ERROR, ErrorCode};

        let mut mt = ZSTDMT_CCtx::default();
        mt.jobIDMask = 0;
        mt.nextJobID = 1;
        mt.jobReady = 0;
        mt.consumed = 11;
        mt.produced = 7;
        mt.inBuff.filled = 5;
        mt.jobs = vec![ZSTDMT_jobDescription {
            src: Range { start: 0, size: 13 },
            consumed: 3,
            cSize: ERROR(ErrorCode::Generic),
            dstFlushed: 9,
            ..ZSTDMT_jobDescription::default()
        }];

        let fps = ZSTDMT_getFrameProgression(&mt);
        assert_eq!(fps.ingested, 29);
        assert_eq!(fps.consumed, 14);
        assert_eq!(fps.produced, 7);
        assert_eq!(fps.flushed, 7);
        assert_eq!(fps.nbActiveWorkers, 1);
    }

    #[test]
    fn zstdmt_flush_produced_releases_completed_job_and_marks_all_done() {
        let mut mt = ZSTDMT_createCCtx(1).expect("mt");
        mt.frameEnded = 1;
        mt.nextJobID = 1;
        mt.doneJobID = 0;
        mt.jobIDMask = 3;
        mt.jobs[0] = ZSTDMT_jobDescription {
            src: Range { start: 0, size: 4 },
            consumed: 4,
            cSize: 3,
            dstFlushed: 0,
            dstBuff: Buffer { data: vec![1, 2, 3] },
            ..ZSTDMT_jobDescription::default()
        };

        let mut out = [0u8; 8];
        let mut out_pos = 0usize;
        let remaining = ZSTDMT_flushProduced(
            &mut mt,
            &mut out,
            &mut out_pos,
            0,
            ZSTD_EndDirective::ZSTD_e_end,
        );

        assert_eq!(remaining, 0);
        assert_eq!(out_pos, 3);
        assert_eq!(&out[..3], &[1, 2, 3]);
        assert_eq!(mt.doneJobID, 1);
        assert_eq!(mt.consumed, 4);
        assert_eq!(mt.produced, 3);
        assert_eq!(mt.allJobsCompleted, 1);
        assert_eq!(mt.jobs[0].cSize, 0);
        assert!(mt.jobs[0].dstBuff.data.is_empty());
    }

    #[test]
    fn zstdmt_flush_produced_returns_error_and_clears_jobs() {
        use crate::common::error::{ERROR, ERR_isError, ErrorCode};

        let mut mt = ZSTDMT_createCCtx(1).expect("mt");
        mt.nextJobID = 1;
        mt.doneJobID = 0;
        mt.jobIDMask = 3;
        mt.inBuff.buffer = Buffer { data: vec![9, 8, 7] };
        mt.inBuff.filled = 3;
        mt.allJobsCompleted = 0;
        mt.jobs[0] = ZSTDMT_jobDescription {
            src: Range { start: 0, size: 4 },
            consumed: 1,
            cSize: ERROR(ErrorCode::MemoryAllocation),
            dstBuff: Buffer { data: vec![1, 2, 3] },
            ..ZSTDMT_jobDescription::default()
        };

        let mut out = [0u8; 8];
        let mut out_pos = 0usize;
        let code = ZSTDMT_flushProduced(
            &mut mt,
            &mut out,
            &mut out_pos,
            0,
            ZSTD_EndDirective::ZSTD_e_continue,
        );

        assert!(ERR_isError(code));
        assert_eq!(mt.doneJobID, mt.nextJobID);
        assert_eq!(mt.inBuff.filled, 0);
        assert!(mt.inBuff.buffer.data.is_empty());
        assert_eq!(mt.allJobsCompleted, 1);
        assert!(mt.jobs.iter().all(|job| job.cSize == 0 && job.dstBuff.data.is_empty()));
    }

    #[test]
    fn find_synchronization_point_flushes_immediately_when_buffer_tail_already_hits() {
        let mut mt = ZSTDMT_CCtx::default();
        mt.params.rsyncable = 1;
        mt.targetSectionSize = RSYNC_MIN_BLOCK_SIZE;
        mt.inBuff.buffer = Buffer {
            data: vec![b'a'; RSYNC_MIN_BLOCK_SIZE],
        };
        mt.inBuff.filled = RSYNC_MIN_BLOCK_SIZE;
        let tail_hash = ZSTD_rollingHash_compute(
            &mt.inBuff.buffer.data[mt.inBuff.filled - RSYNC_LENGTH..mt.inBuff.filled],
        );
        mt.rsync.primePower = ZSTD_rollingHash_primePower(RSYNC_LENGTH as u32);
        mt.rsync.hitMask = tail_hash;

        let sync = findSynchronizationPoint(&mt, b"trailing input", 0);
        assert_eq!(sync.toLoad, 0);
        assert_eq!(sync.flush, 1);
    }

    #[test]
    fn find_synchronization_point_uses_buffered_tail_when_scanning_new_input() {
        let mut mt = ZSTDMT_CCtx::default();
        mt.params.rsyncable = 1;
        mt.targetSectionSize = RSYNC_MIN_BLOCK_SIZE + 3;
        mt.inBuff.buffer = Buffer {
            data: vec![b'b'; RSYNC_MIN_BLOCK_SIZE],
        };
        mt.inBuff.filled = RSYNC_MIN_BLOCK_SIZE;
        mt.rsync.primePower = ZSTD_rollingHash_primePower(RSYNC_LENGTH as u32);

        let prev = &mt.inBuff.buffer.data[mt.inBuff.filled - RSYNC_LENGTH..mt.inBuff.filled];
        let prev_hash = ZSTD_rollingHash_compute(prev);
        let next_byte = b'c';
        let target_hash =
            ZSTD_rollingHash_rotate(prev_hash, prev[0], next_byte, mt.rsync.primePower);
        mt.rsync.hitMask = target_hash;

        let sync = findSynchronizationPoint(&mt, &[next_byte, b'd', b'e'], 0);
        assert_eq!(sync.toLoad, 1);
        assert_eq!(sync.flush, 1);
    }
}
