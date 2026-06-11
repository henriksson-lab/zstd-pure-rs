//! Translation of `lib/compress/zstdmt_compress.c`. Multi-threaded
//! compression. Behind the `mt` feature.
//!
//! The current port executes jobs on a reusable worker pool and keeps
//! the serial orchestration state that upstream uses to prepare work
//! for those workers. Job creation, buffer-pool handoff, worker CCtx
//! reuse, round-buffer sizing, and completion cleanup follow the
//! upstream lifecycle shape.

#![allow(clippy::field_reassign_with_default)]

use crate::common::bits::ZSTD_highbit32;
use crate::common::pool::{
    POOL_add, POOL_ctx, POOL_free, POOL_resize, POOL_sizeof, ZSTD_createThreadPool,
};
use crate::common::xxhash::XXH64_state_t;
use crate::compress::match_state::ZSTD_WINDOW_START_INDEX;
use crate::compress::zstd_compress::ZSTD_customMem_validate;
use crate::compress::zstd_compress::ZSTD_WINDOWLOG_MAX;
use crate::compress::zstd_compress::{
    ZSTD_CCtx, ZSTD_CCtxParams_init, ZSTD_CCtx_params, ZSTD_CCtx_trace, ZSTD_CParamMode_e,
    ZSTD_EndDirective, ZSTD_compressBegin_advanced_internal, ZSTD_compressBound,
    ZSTD_compressContinue_public, ZSTD_compressEnd_public, ZSTD_createCCtx_advanced,
    ZSTD_customMem, ZSTD_cycleLog, ZSTD_frameProgression, ZSTD_getCParamsFromCCtxParams,
    ZSTD_invalidateRepCodes, ZSTD_referenceExternalSequences, ZSTD_sizeof_CCtx,
    ZSTD_writeLastEmptyBlock, ZSTD_CLEVEL_DEFAULT,
};
use crate::compress::zstd_hashes::{
    ZSTD_rollingHash_append, ZSTD_rollingHash_compute, ZSTD_rollingHash_primePower,
    ZSTD_rollingHash_rotate,
};
use crate::compress::zstd_ldm::{
    ldmParams_t, ldmState_t, rawSeq, RawSeqStore_t, ZSTD_ParamSwitch_e, ZSTD_ldm_adjustParameters,
    ZSTD_ldm_fillHashTable, ZSTD_ldm_generateSequences, ZSTD_ldm_getMaxNbSeq,
};
use crate::decompress::zstd_ddict::ZSTD_dictContentType_e;
use crate::decompress::zstd_decompress_block::ZSTD_BLOCKSIZE_MAX;
use core::marker::PhantomData;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::OnceLock;
use std::time::Instant;

#[cfg(target_pointer_width = "32")]
pub const ZSTDMT_JOBLOG_MAX: u32 = 29;
#[cfg(not(target_pointer_width = "32"))]
pub const ZSTDMT_JOBLOG_MAX: u32 = 30;
#[cfg(target_pointer_width = "32")]
pub const ZSTDMT_NBWORKERS_MAX: u32 = 64;
#[cfg(not(target_pointer_width = "32"))]
pub const ZSTDMT_NBWORKERS_MAX: u32 = 256;
pub const ZSTDMT_JOBSIZE_MIN: usize = 512 * 1024;
pub const ZSTDMT_JOBSIZE_MAX: usize = 1 << ZSTDMT_JOBLOG_MAX;
pub const RSYNC_LENGTH: usize = 32;
pub const RSYNC_MIN_BLOCK_SIZE: usize = ZSTD_BLOCKSIZE_MAX;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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
    pub ldmState: Option<ldmState_t>,
    pub ldmNextSrc: u32,
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
    pub rawSeqStore: RawSeqStore_t,
}

impl Buffer {
    #[inline]
    fn from_vec(data: Vec<u8>) -> Self {
        Self {
            data,
            view: Range::default(),
        }
    }

    #[inline]
    fn borrowed(start: usize, size: usize) -> Self {
        Self {
            data: Vec::new(),
            view: Range { start, size },
        }
    }

    #[inline]
    fn is_borrowed(&self) -> bool {
        self.view.start != 0 || self.view.size != 0
    }

    #[inline]
    pub fn start(&self) -> usize {
        if self.is_borrowed() {
            self.view.start
        } else {
            self.data.as_ptr() as usize
        }
    }
}

impl Range {
    #[inline]
    pub fn end(&self) -> usize {
        self.start + self.size
    }
}

/// Rust-only helper: reinterpret a `Range` as a borrowed byte slice
/// pointing at MT-context-owned memory. Returns an empty slice when
/// `range.size == 0`. Caller must guarantee lifetime.
#[inline]
fn range_as_slice(range: Range) -> &'static [u8] {
    if range.size == 0 {
        &[]
    } else {
        // `Range` always points into buffers owned by the MT context.
        unsafe { core::slice::from_raw_parts(range.start as *const u8, range.size) }
    }
}

#[inline]
fn ZSTDMT_inBuffFilledSlice(mtctx: &ZSTDMT_CCtx) -> &[u8] {
    let filled = mtctx.inBuff.filled.min(mtctx.inBuff.buffer.capacity());
    if filled == 0 {
        &[]
    } else if mtctx.inBuff.buffer.is_borrowed() {
        range_as_slice(Range {
            start: mtctx.inBuff.buffer.start(),
            size: filled,
        })
    } else {
        &mtctx.inBuff.buffer.data[..filled]
    }
}

#[inline]
fn ZSTDMT_roundBufferOffset(roundBuff: &RoundBuff_t, start: usize, size: usize) -> Option<usize> {
    let base = roundBuff.buffer.start();
    let end = start.checked_add(size)?;
    let rel = start.checked_sub(base)?;
    if end <= base.checked_add(roundBuff.buffer.data.len())? {
        Some(rel)
    } else {
        None
    }
}

fn ZSTDMT_copyToInputBuffer(mtctx: &mut ZSTDMT_CCtx, src: &[u8]) -> bool {
    let filled = mtctx.inBuff.filled;
    if mtctx.inBuff.buffer.is_borrowed() {
        let Some(offset) = ZSTDMT_roundBufferOffset(
            &mtctx.roundBuff,
            mtctx.inBuff.buffer.start(),
            filled + src.len(),
        ) else {
            return false;
        };
        let dst_start = offset + filled;
        let dst_end = dst_start + src.len();
        mtctx.roundBuff.buffer.data[dst_start..dst_end].copy_from_slice(src);
    } else {
        if mtctx.inBuff.buffer.data.len() < filled + src.len() {
            mtctx.inBuff.buffer.data.resize(filled + src.len(), 0);
        }
        mtctx.inBuff.buffer.data[filled..filled + src.len()].copy_from_slice(src);
    }
    true
}

pub struct ZSTDMT_CCtx {
    _priv: PhantomData<()>,
    pub jobs: Vec<ZSTDMT_jobDescription>,
    pub threadPool: Option<Box<POOL_ctx>>,
    pub threadPoolRef: usize,
    pub rayonThreadPoolRef: usize,
    jobReceivers: Vec<Option<Receiver<ZSTDMT_completedJob>>>,
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
    pub xxhState: XXH64_state_t,
    pub cMem: ZSTD_customMem,
}

impl Default for ZSTDMT_CCtx {
    fn default() -> Self {
        Self {
            _priv: PhantomData,
            jobs: Vec::new(),
            threadPool: None,
            threadPoolRef: 0,
            rayonThreadPoolRef: 0,
            jobReceivers: Vec::new(),
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
            xxhState: XXH64_state_t::default(),
            cMem: ZSTD_customMem::default(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Buffer {
    pub data: Vec<u8>,
    view: Range,
}

impl Buffer {
    #[inline]
    pub fn capacity(&self) -> usize {
        if self.is_borrowed() {
            self.view.size
        } else {
            self.data.len()
        }
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

#[inline]
fn ZSTDMT_bufPoolMaxNbBuffers(nbWorkers: u32) -> u32 {
    2 * nbWorkers + 3
}

#[inline]
fn ZSTDMT_seqPoolMaxNbBuffers(nbWorkers: u32) -> u32 {
    nbWorkers
}

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
        + bufPool.totalBuffers * core::mem::size_of::<Buffer>()
        + bufPool.buffers.iter().map(Buffer::capacity).sum::<usize>()
}

/// Port of `ZSTDMT_setBufferSize`.
pub fn ZSTDMT_setBufferSize(bufPool: &mut ZSTDMT_bufferPool, bSize: usize) {
    bufPool.bufferSize = bSize;
}

/// Port of `ZSTDMT_expandBufferPool`.
pub fn ZSTDMT_expandBufferPool(
    srcBufPool: Box<ZSTDMT_bufferPool>,
    maxNbBuffers: u32,
) -> Option<Box<ZSTDMT_bufferPool>> {
    if srcBufPool.totalBuffers >= maxNbBuffers as usize {
        return Some(srcBufPool);
    }
    let cMem = srcBufPool._cMem;
    let bSize = srcBufPool.bufferSize;
    drop(srcBufPool);
    let mut newBufPool = ZSTDMT_createBufferPool(maxNbBuffers, cMem)?;
    ZSTDMT_setBufferSize(&mut newBufPool, bSize);
    Some(newBufPool)
}

/// Port of `ZSTDMT_getBuffer`.
pub fn ZSTDMT_getBuffer(bufPool: &mut ZSTDMT_bufferPool) -> Buffer {
    let bSize = bufPool.bufferSize;
    if let Some(buf) = bufPool.buffers.pop() {
        let availBufferSize = buf.capacity();
        if availBufferSize >= bSize && (availBufferSize >> 3) <= bSize {
            return buf;
        }
    }
    Buffer::from_vec(vec![0; bSize])
}

/// Port of `ZSTDMT_resizeBuffer`.
pub fn ZSTDMT_resizeBuffer(bufPool: &ZSTDMT_bufferPool, mut buffer: Buffer) -> Buffer {
    if buffer.is_borrowed() {
        let mut resized = vec![0; bufPool.bufferSize];
        let to_copy = buffer.capacity().min(resized.len());
        if to_copy != 0 {
            resized[..to_copy].copy_from_slice(&range_as_slice(buffer.view)[..to_copy]);
        }
        return Buffer::from_vec(resized);
    }
    if buffer.capacity() < bufPool.bufferSize {
        buffer.data.resize(bufPool.bufferSize, 0);
    }
    buffer
}

/// Port of `ZSTDMT_releaseBuffer`.
pub fn ZSTDMT_releaseBuffer(bufPool: &mut ZSTDMT_bufferPool, buf: Buffer) {
    if buf.is_borrowed() || buf.capacity() == 0 {
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
    Buffer::from_vec(vec![0; seq.capacity * core::mem::size_of::<rawSeq>()])
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
pub fn ZSTDMT_createSeqPool(nbWorkers: u32, cMem: ZSTD_customMem) -> Option<Box<ZSTDMT_seqPool>> {
    let mut seqPool = ZSTDMT_createBufferPool(ZSTDMT_seqPoolMaxNbBuffers(nbWorkers), cMem)?;
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
    ZSTDMT_expandBufferPool(pool, ZSTDMT_seqPoolMaxNbBuffers(nbWorkers))
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
pub fn ZSTDMT_createCCtxPool(nbWorkers: i32, cMem: ZSTD_customMem) -> Option<Box<ZSTDMT_CCtxPool>> {
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
    srcPool: Box<ZSTDMT_CCtxPool>,
    nbWorkers: i32,
) -> Option<Box<ZSTDMT_CCtxPool>> {
    if nbWorkers <= srcPool.totalCCtx as i32 {
        return Some(srcPool);
    }
    let cMem = srcPool._cMem;
    drop(srcPool);
    ZSTDMT_createCCtxPool(nbWorkers, cMem)
}

/// Port of `ZSTDMT_sizeof_CCtxPool`.
pub fn ZSTDMT_sizeof_CCtxPool(cctxPool: &ZSTDMT_CCtxPool) -> usize {
    core::mem::size_of::<ZSTDMT_CCtxPool>()
        + cctxPool.totalCCtx * core::mem::size_of::<Box<ZSTD_CCtx>>()
        + cctxPool
            .cctxs
            .iter()
            .map(|c| ZSTD_sizeof_CCtx(c))
            .sum::<usize>()
}

/// Port of `ZSTDMT_getCCtx`.
pub fn ZSTDMT_getCCtx(cctxPool: &mut ZSTDMT_CCtxPool) -> Option<Box<ZSTD_CCtx>> {
    if let Some(cctx) = cctxPool.cctxs.pop() {
        Some(cctx)
    } else {
        ZSTD_createCCtx_advanced(cctxPool._cMem)
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
    mut params: ZSTD_CCtx_params,
    jobSize: usize,
    dict: Option<&[u8]>,
) -> i32 {
    let mut ldmState = None;
    if params.ldmParams.enableLdm == ZSTD_ParamSwitch_e::ZSTD_ps_enable {
        ZSTD_ldm_adjustParameters(&mut params.ldmParams, &params.cParams);
        ZSTDMT_setNbSeq(seqPool, ZSTD_ldm_getMaxNbSeq(params.ldmParams, jobSize));
        let mut state = ldmState_t::new(&params.ldmParams);
        if let Some(dict) = dict.filter(|dict| !dict.is_empty()) {
            ZSTD_ldm_fillHashTable(&mut state, dict, ZSTD_WINDOW_START_INDEX, &params.ldmParams);
            state.loadedDictEnd = if params.forceWindow != 0 {
                0
            } else {
                dict.len().min(u32::MAX as usize) as u32
            };
        }
        ldmState = Some(state);
    } else {
        params.ldmParams = ldmParams_t::default();
    }
    serialState.nextJobID = 0;
    params.jobSize = jobSize;
    serialState.params = params;
    serialState.ldmState = ldmState;
    serialState.ldmNextSrc = ZSTD_WINDOW_START_INDEX;
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
    serialState.ldmState = None;
    serialState.ldmNextSrc = ZSTD_WINDOW_START_INDEX;
}

/// Rust-only helper: advances the serial job-order gate after this
/// file has prepared (or intentionally skipped) external sequences.
fn ZSTDMT_serialState_noteSequencesFinished(
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
pub fn ZSTDMT_serialState_ensureFinished(serialState: &mut SerialState, jobID: u32, cSize: usize) {
    if crate::common::error::ERR_isError(cSize) && serialState.nextJobID <= jobID {
        serialState.nextJobID = jobID + 1;
    }
}

/// Port of `ZSTDMT_freeJobsTable`.
pub fn ZSTDMT_freeJobsTable(
    jobTable: &mut Vec<ZSTDMT_jobDescription>,
    _nbJobs: u32,
    _cMem: ZSTD_customMem,
) {
    jobTable.clear();
}

/// Port of `ZSTDMT_createJobsTable`.
pub fn ZSTDMT_createJobsTable(
    nbJobsPtr: &mut u32,
    _cMem: ZSTD_customMem,
) -> Option<Vec<ZSTDMT_jobDescription>> {
    let nbJobsLog2 = ZSTD_highbit32(*nbJobsPtr) + 1;
    let nbJobs = 1u32 << nbJobsLog2;
    *nbJobsPtr = nbJobs;
    Some(vec![ZSTDMT_jobDescription::default(); *nbJobsPtr as usize])
}

/// Port of `ZSTDMT_expandJobsTable`.
pub fn ZSTDMT_expandJobsTable(mtctx: &mut ZSTDMT_CCtx, nbWorkers: u32) -> usize {
    let mut nbJobs = nbWorkers + 2;
    if nbJobs > mtctx.jobIDMask + 1 {
        mtctx.jobs = ZSTDMT_createJobsTable(&mut nbJobs, mtctx.cMem).unwrap_or_default();
        mtctx.jobReceivers.clear();
        mtctx.jobReceivers.resize_with(mtctx.jobs.len(), || None);
        mtctx.jobIDMask = nbJobs.saturating_sub(1);
    }
    0
}

/// Port of `ZSTDMT_CCtxParam_setNbWorkers`.
pub fn ZSTDMT_CCtxParam_setNbWorkers(params: &mut ZSTD_CCtx_params, nbWorkers: u32) -> usize {
    let nbWorkers = nbWorkers.min(ZSTDMT_NBWORKERS_MAX);
    params.nbWorkers = nbWorkers as i32;
    params.overlapLog = 6;
    params.jobSize = 0;
    nbWorkers as usize
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
    debug_assert!(mtctx.jobReady <= 1);
    let lastJobNb = mtctx.nextJobID + mtctx.jobReady;
    for jobNb in mtctx.doneJobID..lastJobNb {
        let wJobID = (jobNb & mtctx.jobIDMask) as usize;
        if let Some(jobPtr) = mtctx.jobs.get(wJobID) {
            let produced = if ERR_isError(jobPtr.cSize) {
                0
            } else {
                jobPtr.cSize
            };
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
            let worker_pending = mtctx
                .jobReceivers
                .get(wJobID)
                .and_then(|slot| slot.as_ref())
                .is_some();
            fps.nbActiveWorkers += (worker_pending || jobPtr.consumed < jobPtr.src.size) as u32;
        }
    }
    fps
}

/// Port of `ZSTDMT_computeTargetJobLog`.
pub fn ZSTDMT_computeTargetJobLog(params: &ZSTD_CCtx_params) -> u32 {
    let jobLog = if params.ldmParams.enableLdm == ZSTD_ParamSwitch_e::ZSTD_ps_enable {
        core::cmp::max(
            21,
            ZSTD_cycleLog(params.cParams.chainLog, params.cParams.strategy) + 3,
        )
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
    if ovlog == 0 {
        ZSTDMT_overlapLog_default(strat)
    } else {
        ovlog
    }
}

/// Port of `ZSTDMT_computeOverlapSize`.
pub fn ZSTDMT_computeOverlapSize(params: &ZSTD_CCtx_params) -> usize {
    let overlapRLog = 9 - ZSTDMT_overlapLog(params.overlapLog, params.cParams.strategy);
    let mut ovLog = if overlapRLog >= 8 {
        0
    } else {
        params.cParams.windowLog as i32 - overlapRLog
    };
    if params.ldmParams.enableLdm == ZSTD_ParamSwitch_e::ZSTD_ps_enable {
        ovLog = core::cmp::min(
            params.cParams.windowLog,
            ZSTDMT_computeTargetJobLog(params) - 2,
        ) as i32
            - overlapRLog;
    }
    if ovLog <= 0 {
        0
    } else {
        1usize << ovLog.min(ZSTD_WINDOWLOG_MAX() as i32)
    }
}

/// Port of `ZSTDMT_releaseAllJobResources`.
pub fn ZSTDMT_releaseAllJobResources(mtctx: &mut ZSTDMT_CCtx) {
    ZSTDMT_waitForAllJobsCompleted(mtctx);
    for job in &mut mtctx.jobs {
        if let Some(bufPool) = mtctx.bufPool.as_mut() {
            ZSTDMT_releaseBuffer(bufPool, core::mem::take(&mut job.dstBuff));
        }
        if let Some(seqPool) = mtctx.seqPool.as_mut() {
            ZSTDMT_releaseSeq(seqPool, core::mem::take(&mut job.rawSeqStore));
        }
        *job = ZSTDMT_jobDescription::default();
    }
    mtctx.jobReceivers.clear();
    mtctx.inBuff.buffer = Buffer::default();
    mtctx.inBuff.filled = 0;
    mtctx.allJobsCompleted = 1;
}

/// Port of `ZSTDMT_waitForAllJobsCompleted`.
pub fn ZSTDMT_waitForAllJobsCompleted(mtctx: &mut ZSTDMT_CCtx) {
    for jobID in mtctx.doneJobID..mtctx.nextJobID {
        let wJobID = (jobID & mtctx.jobIDMask) as usize;
        let _ = ZSTDMT_joinPendingJob(mtctx, wJobID);
    }
    mtctx.doneJobID = mtctx.nextJobID;
}

/// Rust-only helper: returns the active `POOL_ctx`, preferring the
/// owned `threadPool` over any externally-borrowed `threadPoolRef`.
fn ZSTDMT_getThreadPool(mtctx: &ZSTDMT_CCtx) -> Option<&POOL_ctx> {
    if let Some(pool) = mtctx.threadPool.as_ref() {
        Some(pool)
    } else if mtctx.threadPoolRef != 0 {
        Some(unsafe { &*(mtctx.threadPoolRef as *const POOL_ctx) })
    } else {
        None
    }
}

/// Rust-only helper: binds an externally-owned `POOL_ctx` to the MT
/// context, freeing any previously-owned pool. Passing `None` falls
/// back to lazily creating an internal pool sized to `params.nbWorkers`.
pub fn ZSTDMT_setThreadPool(mtctx: &mut ZSTDMT_CCtx, pool: Option<&POOL_ctx>) {
    if let Some(pool) = pool {
        POOL_free(mtctx.threadPool.take());
        mtctx.threadPoolRef = pool as *const POOL_ctx as usize;
        mtctx.rayonThreadPoolRef = 0;
    } else {
        mtctx.threadPoolRef = 0;
        if mtctx.threadPool.is_none() && mtctx.params.nbWorkers > 0 {
            mtctx.threadPool = ZSTD_createThreadPool(mtctx.params.nbWorkers as usize);
        }
    }
}

/// Rust-only helper: routes MT job dispatch through a borrowed Rayon
/// thread pool instead of the in-house `POOL_ctx`. Mutually exclusive
/// with `ZSTDMT_setThreadPool`.
pub fn ZSTDMT_setRayonThreadPool(mtctx: &mut ZSTDMT_CCtx, pool: Option<&rayon::ThreadPool>) {
    if let Some(pool) = pool {
        POOL_free(mtctx.threadPool.take());
        mtctx.threadPoolRef = 0;
        mtctx.rayonThreadPoolRef = pool as *const rayon::ThreadPool as usize;
    } else {
        mtctx.rayonThreadPoolRef = 0;
        if mtctx.threadPool.is_none() && mtctx.threadPoolRef == 0 && mtctx.params.nbWorkers > 0 {
            mtctx.threadPool = ZSTD_createThreadPool(mtctx.params.nbWorkers as usize);
        }
    }
}

struct ZSTDMT_queuedJob {
    job: ZSTDMT_jobDescription,
    cctx: Box<ZSTD_CCtx>,
    tx: mpsc::Sender<ZSTDMT_completedJob>,
}

struct ZSTDMT_completedJob {
    job: ZSTDMT_jobDescription,
    cctx: Box<ZSTD_CCtx>,
}

/// Rust-only helper: thunk used as `POOL_function` for a queued job.
/// Reconstructs the boxed `ZSTDMT_queuedJob`, runs the upstream
/// `ZSTDMT_compressionJob` body, then ships the result back through
/// the channel.
fn ZSTDMT_runQueuedJob(opaque: *mut core::ffi::c_void) {
    let queued = unsafe { Box::from_raw(opaque as *mut ZSTDMT_queuedJob) };
    let mut job = queued.job;
    let mut cctx = queued.cctx;
    ZSTDMT_compressionJob(&mut job, &mut cctx);
    let _ = queued.tx.send(ZSTDMT_completedJob { job, cctx });
}

/// Rust-only helper: drives LDM sequence generation for job `wJobID`,
/// substituting for the upstream worker-side LDM loop. When LDM is
/// disabled, only advances `serial.nextJobID`; otherwise builds the
/// `rawSeqStore` over `prefix ++ src` and parks it on the job slot.
fn ZSTDMT_prepareJobSequences(mtctx: &mut ZSTDMT_CCtx, wJobID: usize) -> usize {
    let Some(job) = mtctx.jobs.get(wJobID) else {
        return crate::common::error::ERROR(crate::common::error::ErrorCode::Generic);
    };
    let prefix_range = job.prefix;
    let src_range = job.src;
    let job_id = job.jobID;

    let serial_params = mtctx.serial.params;
    if serial_params.ldmParams.enableLdm != ZSTD_ParamSwitch_e::ZSTD_ps_enable {
        let mut seq_store = RawSeqStore_t::default();
        ZSTDMT_serialState_noteSequencesFinished(
            &mut mtctx.serial,
            &mut seq_store,
            src_range,
            job_id,
        );
        if let Some(job) = mtctx.jobs.get_mut(wJobID) {
            job.rawSeqStore = seq_store;
        }
        return 0;
    }

    let prefix = range_as_slice(prefix_range);
    let src = range_as_slice(src_range);
    let ldmParams: ldmParams_t = serial_params.ldmParams;
    let required_capacity = ZSTD_ldm_getMaxNbSeq(ldmParams, src.len()).max(1);

    let mut rawSeqStore = if let Some(seqPool) = mtctx.seqPool.as_mut() {
        let mut seq = ZSTDMT_getSeq(seqPool);
        if seq.capacity < required_capacity {
            seq = RawSeqStore_t::with_capacity(required_capacity);
        } else {
            if seq.seq.len() < required_capacity {
                seq.seq.resize(required_capacity, rawSeq::default());
            }
            seq.capacity = required_capacity;
        }
        seq
    } else {
        RawSeqStore_t::with_capacity(required_capacity)
    };
    rawSeqStore.size = 0;
    rawSeqStore.pos = 0;
    rawSeqStore.posInSequence = 0;

    if mtctx.serial.ldmState.is_none() {
        mtctx.serial.ldmState = Some(ldmState_t::new(&ldmParams));
        mtctx.serial.ldmNextSrc = ZSTD_WINDOW_START_INDEX;
    }
    let ldmState = mtctx
        .serial
        .ldmState
        .as_mut()
        .expect("LDM state initialized");
    // The serial LDM tables persist across jobs, but this Rust MT shim
    // only materializes `prefix ++ src` for matching. Older table
    // entries remain available when they point into that prefix; entries
    // older than the carried prefix are rejected by the low-limit below.
    let src_abs_start = mtctx.serial.ldmNextSrc;
    let window_abs_start = src_abs_start.wrapping_sub(prefix.len() as u32);
    ldmState.window.base_offset = window_abs_start;
    ldmState.window.nextSrc = window_abs_start.wrapping_add((prefix.len() + src.len()) as u32);
    ldmState.window.dictBase_offset = window_abs_start;
    ldmState.window.dictLimit = window_abs_start;
    ldmState.window.lowLimit = window_abs_start;
    if !prefix.is_empty() {
        ZSTD_ldm_fillHashTable(ldmState, prefix, window_abs_start, &ldmParams);
    }
    let mut window_buf = Vec::with_capacity(prefix.len() + src.len());
    window_buf.extend_from_slice(prefix);
    window_buf.extend_from_slice(src);
    let rc = ZSTD_ldm_generateSequences(
        ldmState,
        &mut rawSeqStore,
        &ldmParams,
        &window_buf,
        prefix.len(),
        window_buf.len(),
        0,
    );
    if crate::common::error::ERR_isError(rc) {
        return rc;
    }
    mtctx.serial.ldmNextSrc = src_abs_start.wrapping_add(src.len() as u32);
    ZSTDMT_serialState_noteSequencesFinished(
        &mut mtctx.serial,
        &mut rawSeqStore,
        src_range,
        job_id,
    );
    if let Some(job) = mtctx.jobs.get_mut(wJobID) {
        job.rawSeqStore = rawSeqStore;
    }
    0
}

/// Rust-only helper: queues job `wJobID` onto either the in-house
/// `POOL_ctx` or the borrowed Rayon pool. Pulls a worker CCtx from the
/// pool, wires a oneshot `mpsc` channel for completion, and stores the
/// receiver in `jobReceivers[wJobID]`.
fn ZSTDMT_spawnCompressionJob(mtctx: &mut ZSTDMT_CCtx, wJobID: usize) -> usize {
    let Some(job) = mtctx.jobs.get(wJobID).cloned() else {
        return crate::common::error::ERROR(crate::common::error::ErrorCode::Generic);
    };
    if mtctx.jobReceivers.len() <= wJobID {
        mtctx.jobReceivers.resize_with(wJobID + 1, || None);
    }
    let pool_ptr = mtctx
        .threadPool
        .as_ref()
        .map(|p| p.as_ref() as *const POOL_ctx as usize)
        .unwrap_or(mtctx.threadPoolRef);
    if pool_ptr == 0 && mtctx.rayonThreadPoolRef == 0 {
        return crate::common::error::ERROR(crate::common::error::ErrorCode::Generic);
    }
    let cctx = if let Some(pool) = mtctx.cctxPool.as_mut() {
        ZSTDMT_getCCtx(pool)
    } else {
        ZSTD_createCCtx_advanced(mtctx.cMem)
    };
    let Some(cctx) = cctx else {
        return crate::common::error::ERROR(crate::common::error::ErrorCode::MemoryAllocation);
    };
    let (tx, rx) = mpsc::channel();
    let queued = Box::new(ZSTDMT_queuedJob { job, cctx, tx });
    mtctx.jobReceivers[wJobID] = Some(rx);
    let queued_ptr = Box::into_raw(queued) as *mut core::ffi::c_void;
    if mtctx.rayonThreadPoolRef != 0 {
        let pool = unsafe { &*(mtctx.rayonThreadPoolRef as *const rayon::ThreadPool) };
        let queued_addr = queued_ptr as usize;
        pool.spawn(move || ZSTDMT_runQueuedJob(queued_addr as *mut core::ffi::c_void));
    } else {
        let pool = unsafe { &*(pool_ptr as *const POOL_ctx) };
        POOL_add(pool, ZSTDMT_runQueuedJob, queued_ptr);
    }
    0
}

/// Rust-only helper: blocks on the completion channel for job
/// `wJobID`, copies the completed `ZSTDMT_jobDescription` back into
/// the job table, and returns the worker CCtx to the CCtx pool.
fn ZSTDMT_joinPendingJob(mtctx: &mut ZSTDMT_CCtx, wJobID: usize) -> usize {
    let Some(slot) = mtctx.jobReceivers.get_mut(wJobID) else {
        return 0;
    };
    let Some(rx) = slot.take() else {
        return 0;
    };
    match rx.recv() {
        Ok(completed) => {
            if mtctx.jobs.len() <= wJobID {
                mtctx
                    .jobs
                    .resize(wJobID + 1, ZSTDMT_jobDescription::default());
            }
            mtctx.jobs[wJobID] = completed.job;
            if let Some(pool) = mtctx.cctxPool.as_mut() {
                ZSTDMT_releaseCCtx(pool, Some(completed.cctx));
            }
            0
        }
        Err(_) => crate::common::error::ERROR(crate::common::error::ErrorCode::Generic),
    }
}

/// Rust-only helper: non-blocking variant of `ZSTDMT_joinPendingJob`.
/// Returns 0 if the worker is still running, the completed job state
/// if ready, or an error on channel disconnect.
fn ZSTDMT_tryJoinFinishedJob(mtctx: &mut ZSTDMT_CCtx, wJobID: usize) -> usize {
    let Some(slot) = mtctx.jobReceivers.get_mut(wJobID) else {
        return 0;
    };
    let Some(rx) = slot.as_ref() else {
        return 0;
    };
    match rx.try_recv() {
        Ok(completed) => {
            slot.take();
            if mtctx.jobs.len() <= wJobID {
                mtctx
                    .jobs
                    .resize(wJobID + 1, ZSTDMT_jobDescription::default());
            }
            mtctx.jobs[wJobID] = completed.job;
            if let Some(pool) = mtctx.cctxPool.as_mut() {
                ZSTDMT_releaseCCtx(pool, Some(completed.cctx));
            }
            0
        }
        Err(TryRecvError::Empty) => 0,
        Err(TryRecvError::Disconnected) => {
            slot.take();
            crate::common::error::ERROR(crate::common::error::ErrorCode::Generic)
        }
    }
}

/// Port of `ZSTDMT_resize`.
pub fn ZSTDMT_resize(mtctx: &mut ZSTDMT_CCtx, nbWorkers: u32) -> usize {
    use crate::common::error::{ErrorCode, ERROR};

    if let Some(pool) = mtctx.threadPool.as_mut() {
        if POOL_resize(pool, nbWorkers as usize) != 0 {
            return ERROR(ErrorCode::MemoryAllocation);
        }
    }
    let jobs_error = ZSTDMT_expandJobsTable(mtctx, nbWorkers);
    if crate::common::error::ERR_isError(jobs_error) {
        return jobs_error;
    }
    if let Some(pool) = mtctx.bufPool.take() {
        mtctx.bufPool = ZSTDMT_expandBufferPool(pool, ZSTDMT_bufPoolMaxNbBuffers(nbWorkers));
    }
    if mtctx.bufPool.is_none() {
        return ERROR(ErrorCode::MemoryAllocation);
    }
    if let Some(pool) = mtctx.cctxPool.take() {
        mtctx.cctxPool = ZSTDMT_expandCCtxPool(pool, nbWorkers as i32);
    }
    if mtctx.cctxPool.is_none() {
        return ERROR(ErrorCode::MemoryAllocation);
    }
    if let Some(pool) = mtctx.seqPool.take() {
        mtctx.seqPool = ZSTDMT_expandSeqPool(pool, nbWorkers);
    }
    if mtctx.seqPool.is_none() {
        return ERROR(ErrorCode::MemoryAllocation);
    }
    if mtctx.threadPool.is_none() && mtctx.threadPoolRef == 0 && mtctx.rayonThreadPoolRef == 0 {
        mtctx.threadPool = ZSTD_createThreadPool(nbWorkers as usize);
        if mtctx.threadPool.is_none() {
            return ERROR(ErrorCode::MemoryAllocation);
        }
    }
    ZSTDMT_CCtxParam_setNbWorkers(&mut mtctx.params, nbWorkers)
}

/// Port of `ZSTDMT_updateCParams_whileCompressing`.
pub fn ZSTDMT_updateCParams_whileCompressing(
    mtctx: &mut ZSTDMT_CCtx,
    cctxParams: &ZSTD_CCtx_params,
) {
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
    let nbJobs1stRoundMin = if mtctx.targetSectionSize == 0 {
        0
    } else {
        roundBuffCapacity / mtctx.targetSectionSize
    };
    if mtctx.nextJobID < nbJobs1stRoundMin as u32 {
        return Range::default();
    }

    for jobID in mtctx.doneJobID..mtctx.nextJobID {
        let wJobID = (jobID & mtctx.jobIDMask) as usize;
        if let Some(job) = mtctx.jobs.get(wJobID) {
            if job.consumed < job.src.size {
                return if job.prefix.size == 0 {
                    job.src
                } else {
                    job.prefix
                };
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
    let spaceNeeded = mtctx.targetSectionSize;
    if spaceNeeded == 0 || mtctx.roundBuff.buffer.capacity() < spaceNeeded {
        return 0;
    }
    let roundStart = mtctx.roundBuff.buffer.start();
    let spaceLeft = mtctx.roundBuff.capacity.saturating_sub(mtctx.roundBuff.pos);

    if spaceLeft < spaceNeeded {
        let prefix = mtctx.inBuff.prefix;
        if prefix.size > mtctx.roundBuff.capacity.saturating_sub(spaceNeeded) {
            return 0;
        }

        let prefixDst = Buffer::borrowed(roundStart, prefix.size);
        if ZSTDMT_isOverlapped(&prefixDst, ZSTDMT_getInputDataInUse(mtctx)) != 0 {
            return 0;
        }
        ZSTDMT_waitForLdmComplete(mtctx, &prefixDst);

        if prefix.size != 0 {
            if let Some(srcOffset) =
                ZSTDMT_roundBufferOffset(&mtctx.roundBuff, prefix.start, prefix.size)
            {
                mtctx
                    .roundBuff
                    .buffer
                    .data
                    .copy_within(srcOffset..srcOffset + prefix.size, 0);
            } else {
                let prefixCopy = range_as_slice(prefix).to_vec();
                mtctx.roundBuff.buffer.data[..prefix.size].copy_from_slice(&prefixCopy);
            }
        }
        mtctx.inBuff.prefix = Range {
            start: roundStart,
            size: prefix.size,
        };
        mtctx.roundBuff.pos = prefix.size;
    }

    let start = mtctx.roundBuff.pos;
    if mtctx.roundBuff.capacity.saturating_sub(start) < spaceNeeded {
        return 0;
    }
    let buffer = Buffer::borrowed(roundStart + start, spaceNeeded);
    if ZSTDMT_isOverlapped(&buffer, ZSTDMT_getInputDataInUse(mtctx)) != 0 {
        return 0;
    }
    ZSTDMT_waitForLdmComplete(mtctx, &buffer);
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

    let buffered = ZSTDMT_inBuffFilledSlice(mtctx);
    let (mut pos, mut hash, prev): (usize, u64, &[u8]) =
        if mtctx.inBuff.filled < RSYNC_MIN_BLOCK_SIZE {
            let pos = RSYNC_MIN_BLOCK_SIZE - mtctx.inBuff.filled;
            if pos >= syncPoint.toLoad {
                return syncPoint;
            }
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
            debug_assert!(mtctx.inBuff.filled >= RSYNC_MIN_BLOCK_SIZE);
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
    if job.dstBuff.capacity() < crate::decompress::zstd_decompress_block::ZSTD_blockHeaderSize {
        job.dstBuff = Buffer::from_vec(vec![0; 3]);
    }
    job.cSize = ZSTD_writeLastEmptyBlock(&mut job.dstBuff.data);
}

/// Port of `ZSTDMT_createCompressionJob`.
pub fn ZSTDMT_createCompressionJob(
    mtctx: &mut ZSTDMT_CCtx,
    srcSize: usize,
    endOp: ZSTD_EndDirective,
) -> usize {
    if mtctx.nextJobID > mtctx.doneJobID + mtctx.jobIDMask {
        return 0;
    }
    let endFrame = endOp == ZSTD_EndDirective::ZSTD_e_end;
    let jobID = (mtctx.nextJobID & mtctx.jobIDMask) as usize;
    if mtctx.jobs.len() <= jobID {
        mtctx
            .jobs
            .resize(jobID + 1, ZSTDMT_jobDescription::default());
    }
    {
        let job = &mut mtctx.jobs[jobID];
        let srcBuff = core::mem::take(&mut mtctx.inBuff.buffer);
        let srcStart = srcBuff.start();
        job.src = Range {
            start: srcStart,
            size: srcSize,
        };
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
        if mtctx.params.fParams.checksumFlag != 0 && srcSize != 0 {
            crate::common::xxhash::XXH64_update(&mut mtctx.xxhState, range_as_slice(job.src));
        }
        let neededDstSize = ZSTD_compressBound(srcSize.max(1));
        job.dstBuff = if let Some(bufPool) = mtctx.bufPool.as_mut() {
            let mut dst = ZSTDMT_getBuffer(bufPool);
            if dst.capacity() < neededDstSize {
                dst.data.resize(neededDstSize, 0);
            }
            dst
        } else {
            Buffer::from_vec(vec![0; neededDstSize])
        };
        if srcSize == 0 && mtctx.nextJobID > 0 && endFrame {
            ZSTDMT_writeLastEmptyBlock(job);
            job.consumed = job.src.size;
        }
    }
    mtctx.roundBuff.pos += srcSize;
    mtctx.inBuff.filled = 0;
    if !endFrame {
        let newPrefixSize = srcSize.min(mtctx.targetPrefixSize);
        let start = mtctx.jobs[jobID].src.start + srcSize.saturating_sub(newPrefixSize);
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
    ZSTDMT_prepareJobSequences(mtctx, jobID)
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
        let join = if _blockToFlush != 0 {
            ZSTDMT_joinPendingJob(mtctx, wJobID)
        } else {
            ZSTDMT_tryJoinFinishedJob(mtctx, wJobID)
        };
        if ERR_isError(join) {
            ZSTDMT_waitForAllJobsCompleted(mtctx);
            ZSTDMT_releaseAllJobResources(mtctx);
            return join;
        }
        let mut job_completed = false;
        let (src_size, mut c_size);

        if let Some(job) = mtctx.jobs.get_mut(wJobID) {
            if mtctx
                .jobReceivers
                .get(wJobID)
                .and_then(|slot| slot.as_ref())
                .is_some()
            {
                return 1;
            }
            let c_result = job.cSize;
            if ERR_isError(c_result) {
                ZSTDMT_waitForAllJobsCompleted(mtctx);
                ZSTDMT_releaseAllJobResources(mtctx);
                return c_result;
            }

            c_size = c_result;
            if job.frameChecksumNeeded != 0 {
                use crate::common::mem::MEM_writeLE32;
                use crate::common::xxhash::XXH64_digest;

                if job.dstBuff.data.len() < c_size + 4 {
                    job.dstBuff.data.resize(c_size + 4, 0);
                }
                MEM_writeLE32(
                    &mut job.dstBuff.data[c_size..],
                    XXH64_digest(&mtctx.xxhState) as u32,
                );
                job.cSize += 4;
                job.frameChecksumNeeded = 0;
                c_size = job.cSize;
            }
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
                if let Some(seqPool) = mtctx.seqPool.as_mut() {
                    let rawSeqStore = core::mem::take(&mut job.rawSeqStore);
                    ZSTDMT_releaseSeq(seqPool, rawSeqStore);
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
pub fn ZSTDMT_createCCtx_advanced_internal(
    mut nbWorkers: u32,
    cMem: ZSTD_customMem,
) -> Option<Box<ZSTDMT_CCtx>> {
    if nbWorkers < 1 {
        return None;
    }
    nbWorkers = nbWorkers.min(ZSTDMT_NBWORKERS_MAX);
    if !ZSTD_customMem_validate(cMem) {
        return None;
    }
    let mut nbJobs = nbWorkers + 2;
    let jobs = ZSTDMT_createJobsTable(&mut nbJobs, cMem)?;
    let mut mtctx = Box::new(ZSTDMT_CCtx::default());
    mtctx.params.nbWorkers = nbWorkers as i32;
    mtctx.cMem = cMem;
    mtctx.threadPool = ZSTD_createThreadPool(nbWorkers as usize);
    mtctx.jobReceivers.resize_with(jobs.len(), || None);
    mtctx.jobs = jobs;
    mtctx.jobIDMask = nbJobs - 1;
    mtctx.bufPool = ZSTDMT_createBufferPool(ZSTDMT_bufPoolMaxNbBuffers(nbWorkers), cMem);
    mtctx.cctxPool = ZSTDMT_createCCtxPool(nbWorkers as i32, cMem);
    mtctx.seqPool = ZSTDMT_createSeqPool(nbWorkers, cMem);
    if mtctx.threadPool.is_none()
        || mtctx.bufPool.is_none()
        || mtctx.cctxPool.is_none()
        || mtctx.seqPool.is_none()
    {
        return None;
    }
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
        let resize_error = ZSTDMT_resize(mtctx, params.nbWorkers as u32);
        if crate::common::error::ERR_isError(resize_error) {
            return resize_error;
        }
    }
    if params.jobSize != 0 {
        params.jobSize = params.jobSize.clamp(ZSTDMT_JOBSIZE_MIN, ZSTDMT_JOBSIZE_MAX);
    }
    ZSTDMT_releaseAllJobResources(mtctx);
    mtctx.params = params;
    mtctx.frameContentSize = pledgedSrcSize;
    mtctx.targetPrefixSize = ZSTDMT_computeOverlapSize(&mtctx.params);
    mtctx.targetSectionSize = if mtctx.params.jobSize == 0 {
        1usize << ZSTDMT_computeTargetJobLog(&mtctx.params)
    } else {
        mtctx.params.jobSize
    };
    if mtctx.params.rsyncable != 0 {
        let jobSizeKB = (mtctx.targetSectionSize >> 10)
            .max(1)
            .min(u32::MAX as usize) as u32;
        let rsyncBits = ZSTD_highbit32(jobSizeKB) + 10;
        mtctx.rsync.hash = 0;
        mtctx.rsync.hitMask = if rsyncBits >= 64 {
            u64::MAX
        } else {
            (1u64 << rsyncBits) - 1
        };
        mtctx.rsync.primePower = ZSTD_rollingHash_primePower(RSYNC_LENGTH as u32);
    } else {
        mtctx.rsync = RSyncState_t::default();
    }
    if mtctx.targetSectionSize < mtctx.targetPrefixSize {
        mtctx.targetSectionSize = mtctx.targetPrefixSize;
    }
    if let Some(bufPool) = mtctx.bufPool.as_mut() {
        ZSTDMT_setBufferSize(
            bufPool,
            crate::compress::zstd_compress::ZSTD_compressBound(mtctx.targetSectionSize),
        );
    }
    let windowSize = if mtctx.params.ldmParams.enableLdm == ZSTD_ParamSwitch_e::ZSTD_ps_enable {
        1usize << mtctx.params.cParams.windowLog
    } else {
        0
    };
    let nbSlackBuffers = 2 + usize::from(mtctx.targetPrefixSize > 0);
    let slackSize = mtctx.targetSectionSize.saturating_mul(nbSlackBuffers);
    let nbWorkers = mtctx.params.nbWorkers.max(1) as usize;
    let sectionsSize = mtctx.targetSectionSize.saturating_mul(nbWorkers);
    let roundBuffCapacity = windowSize.max(sectionsSize).saturating_add(slackSize);
    if mtctx.roundBuff.capacity < roundBuffCapacity {
        mtctx.roundBuff.buffer = Buffer::from_vec(vec![0; roundBuffCapacity]);
        mtctx.roundBuff.capacity = roundBuffCapacity;
    }
    mtctx.jobReady = 0;
    mtctx.doneJobID = 0;
    mtctx.nextJobID = 0;
    mtctx.frameEnded = 0;
    mtctx.allJobsCompleted = 0;
    mtctx.consumed = 0;
    mtctx.produced = 0;
    crate::common::xxhash::XXH64_reset(&mut mtctx.xxhState, 0);
    mtctx.inBuff.prefix = Range::default();
    mtctx.roundBuff.pos = 0;
    if let Some(seqPool) = mtctx.seqPool.as_mut() {
        let _ = ZSTDMT_serialState_reset(
            &mut mtctx.serial,
            seqPool,
            mtctx.params,
            mtctx.targetSectionSize,
            None,
        );
    } else {
        mtctx.serial = SerialState {
            nextJobID: 0,
            params: mtctx.params,
            ldmState: None,
            ldmNextSrc: ZSTD_WINDOW_START_INDEX,
        };
    }
    0
}

/// Port of `ZSTDMT_compressionJob`.
pub fn ZSTDMT_compressionJob(job: &mut ZSTDMT_jobDescription, cctx: &mut Box<ZSTD_CCtx>) {
    let mut jobParams = job.params;
    let mut lastCBlockSize = 0usize;

    if job.jobID != 0 {
        jobParams.fParams.checksumFlag = 0;
    }
    jobParams.ldmEnable = ZSTD_ParamSwitch_e::ZSTD_ps_disable;
    jobParams.ldmParams.enableLdm = ZSTD_ParamSwitch_e::ZSTD_ps_disable;
    jobParams.nbWorkers = 0;
    jobParams.extSeqProdState = 0;
    jobParams.extSeqProdFunc = None;
    jobParams.forceWindow = (job.firstJob == 0) as i32;
    if job.firstJob == 0 {
        jobParams.deterministicRefPrefix = 0;
    }

    if job.lastJob != 0 && job.src.size == 0 && job.jobID > 0 {
        if job.dstBuff.capacity() < 3 {
            job.dstBuff = Buffer::from_vec(vec![0; 3]);
        }
        job.cSize = ZSTD_writeLastEmptyBlock(&mut job.dstBuff.data);
        job.consumed = job.src.size;
        return;
    }

    let bound = ZSTD_compressBound(job.src.size.max(1));
    if job.dstBuff.capacity() < bound {
        job.dstBuff.data.resize(bound, 0);
    }

    let src = range_as_slice(job.src);
    let prefix = range_as_slice(job.prefix);

    let pledgedSrcSize = if job.firstJob != 0 {
        job.fullFrameSize
    } else {
        job.src.size as u64
    };
    let initError = ZSTD_compressBegin_advanced_internal(
        cctx,
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
        &SerialState {
            nextJobID: job.jobID + 1,
            params: job.params,
            ldmState: None,
            ldmNextSrc: ZSTD_WINDOW_START_INDEX,
        },
        cctx,
        &job.rawSeqStore,
    );

    let mut op = 0usize;
    if job.firstJob == 0 {
        let hSize = ZSTD_compressContinue_public(cctx, &mut job.dstBuff.data[op..], &[]);
        if crate::common::error::ERR_isError(hSize) {
            job.cSize = hSize;
            return;
        }
        ZSTD_invalidateRepCodes(cctx);
    }

    let chunkSize = 4 * ZSTD_BLOCKSIZE_MAX;
    let nbChunks = src.len().div_ceil(chunkSize);
    for chunkNb in 1..nbChunks {
        let start = (chunkNb - 1) * chunkSize;
        let end = start + chunkSize;
        let cSize =
            ZSTD_compressContinue_public(cctx, &mut job.dstBuff.data[op..], &src[start..end]);
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
            ZSTD_compressEnd_public(cctx, &mut job.dstBuff.data[op..], lastSrc)
        } else {
            ZSTD_compressContinue_public(cctx, &mut job.dstBuff.data[op..], lastSrc)
        };
        if crate::common::error::ERR_isError(lastCBlockSize) {
            job.cSize = lastCBlockSize;
            return;
        }
    }

    ZSTD_CCtx_trace(cctx, 0);
    job.cSize += lastCBlockSize;
    job.consumed = job.src.size;
}

/// Port of `ZSTDMT_createCCtx`. Allocates the MT context header and
/// its owned job/pool scaffolding.
pub fn ZSTDMT_createCCtx(nbWorkers: u32) -> Option<Box<ZSTDMT_CCtx>> {
    ZSTDMT_createCCtx_advanced_internal(nbWorkers, ZSTD_customMem::default())
}

/// Port of `ZSTDMT_createCCtx_advanced`. The upstream internal helper
/// also accepts an optional external thread-pool pointer; the current
/// Rust port owns its worker threads internally and therefore only
/// exposes the public `(nbWorkers, customMem)` surface.
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
    POOL_free(mtctx.threadPool.take());
    mtctx.threadPoolRef = 0;
    mtctx.rayonThreadPoolRef = 0;
    ZSTDMT_serialState_free(&mut mtctx.serial);
    mtctx.roundBuff.buffer = Buffer::default();
    mtctx.roundBuff.capacity = 0;
    mtctx.roundBuff.pos = 0;
    0
}

/// Port of `ZSTDMT_compressStream_generic`. Jobs are dispatched onto
/// background worker threads and flushed in-order through the
/// existing MT context/job buffering state machine.
pub fn ZSTDMT_compressStream_generic(
    mtctx: &mut ZSTDMT_CCtx,
    output: &mut [u8],
    output_pos: &mut usize,
    input: &[u8],
    input_pos: &mut usize,
    end_op: crate::compress::zstd_compress::ZSTD_EndDirective,
) -> usize {
    use crate::common::error::{ERR_isError, ErrorCode, ERROR};
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

    let mut forwardInputProgress = false;
    let mut effective_end = end_op;

    if mtctx.jobReady == 0 && *input_pos < input.len() {
        if mtctx.inBuff.buffer.capacity() == 0 {
            debug_assert_eq!(mtctx.inBuff.filled, 0);
            let _ = ZSTDMT_tryGetInputRange(mtctx);
        }
        if mtctx.inBuff.buffer.capacity() != 0 {
            let syncPoint = findSynchronizationPoint(mtctx, input, *input_pos);
            if syncPoint.flush != 0 && effective_end == ZSTD_EndDirective::ZSTD_e_continue {
                effective_end = ZSTD_EndDirective::ZSTD_e_flush;
            }
            let to_load = syncPoint
                .toLoad
                .min(input.len().saturating_sub(*input_pos))
                .min(mtctx.targetSectionSize.saturating_sub(mtctx.inBuff.filled));
            if !ZSTDMT_copyToInputBuffer(mtctx, &input[*input_pos..*input_pos + to_load]) {
                return ERROR(ErrorCode::Generic);
            }
            *input_pos += to_load;
            mtctx.inBuff.filled += to_load;
            forwardInputProgress = to_load > 0;
        }
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
            let Some(job) = mtctx.jobs.get(wJobID) else {
                return ERROR(ErrorCode::Generic);
            };
            let already_complete = job.lastJob != 0
                && job.src.size == 0
                && job.jobID > 0
                && job.consumed == job.src.size
                && job.cSize > 0;
            if !already_complete {
                let spawn = ZSTDMT_spawnCompressionJob(mtctx, wJobID);
                if ERR_isError(spawn) {
                    return spawn;
                }
            }
        }
    }

    let remaining = ZSTDMT_flushProduced(
        mtctx,
        output,
        output_pos,
        (!forwardInputProgress) as u32,
        effective_end,
    );
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
        + mtctx.jobReceivers.capacity()
            * core::mem::size_of::<Option<Receiver<ZSTDMT_completedJob>>>()
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
        + ZSTDMT_getThreadPool(mtctx).map(POOL_sizeof).unwrap_or(0)
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
        // The MT context now owns live worker-pool scaffolding; this
        // test keeps the surface-level API contracts pinned.
        use crate::compress::zstd_compress::ZSTD_EndDirective;

        // Creator now constructs the owned MT context scaffolding for
        // positive worker counts.
        let ctx = ZSTDMT_createCCtx(1).expect("mt ctx");
        assert_eq!(ctx.params.nbWorkers, 1);
        assert_eq!(ctx.jobIDMask + 1, 4);
        assert!(ctx.bufPool.is_some());
        assert!(ctx.cctxPool.is_some());
        assert!(ctx.seqPool.is_some());
        assert_eq!(
            ctx.bufPool.as_ref().unwrap().totalBuffers,
            ZSTDMT_bufPoolMaxNbBuffers(1) as usize
        );
        assert_eq!(
            ctx.seqPool.as_ref().unwrap().totalBuffers,
            ZSTDMT_seqPoolMaxNbBuffers(1) as usize
        );
        let advanced =
            ZSTDMT_createCCtx_advanced(1, ZSTD_customMem::default()).expect("advanced mt ctx");
        assert_eq!(advanced.params.nbWorkers, 1);
        assert!(ZSTDMT_createCCtx(0).is_none());
        assert!(ZSTDMT_createCCtx_advanced(0, ZSTD_customMem::default()).is_none());

        // Free accepts None without panicking.
        assert_eq!(ZSTDMT_freeCCtx(None), 0);

        // Size / flush-now queries work on a default empty ctx.
        let empty = ZSTDMT_CCtx::default();
        assert_eq!(
            ZSTDMT_sizeof_CCtx(&empty),
            core::mem::size_of::<ZSTDMT_CCtx>()
        );
        assert_eq!(ZSTDMT_toFlushNow(&empty), 0);
        assert_eq!(ZSTDMT_nextInputSizeHint(&empty), ZSTD_BLOCKSIZE_MAX);

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
        let mut rc = ZSTDMT_compressStream_generic(
            &mut mt,
            &mut dst,
            &mut dp,
            b"x",
            &mut sp,
            ZSTD_EndDirective::ZSTD_e_end,
        );
        let empty = [];
        while rc != 0 {
            let mut empty_pos = 0usize;
            rc = ZSTDMT_compressStream_generic(
                &mut mt,
                &mut dst,
                &mut dp,
                &empty,
                &mut empty_pos,
                ZSTD_EndDirective::ZSTD_e_end,
            );
        }
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
    fn zstdmt_header_constants_match_upstream_defaults() {
        #[cfg(target_pointer_width = "32")]
        assert_eq!(ZSTDMT_NBWORKERS_MAX, 64);
        #[cfg(not(target_pointer_width = "32"))]
        assert_eq!(ZSTDMT_NBWORKERS_MAX, 256);
        assert_eq!(ZSTDMT_JOBSIZE_MIN, 512 * 1024);
        assert_eq!(RSYNC_MIN_BLOCK_SIZE, ZSTD_BLOCKSIZE_MAX);
    }

    #[test]
    fn zstdmt_ldm_target_job_log_uses_strategy_cycle_log() {
        let mut params = ZSTD_CCtx_params::default();
        params.ldmParams.enableLdm = ZSTD_ParamSwitch_e::ZSTD_ps_enable;
        params.cParams.chainLog = 20;
        params.cParams.strategy = 6;

        assert_eq!(ZSTDMT_computeTargetJobLog(&params), 22);

        params.cParams.strategy = 5;
        assert_eq!(ZSTDMT_computeTargetJobLog(&params), 23);
    }

    #[test]
    fn zstdmt_resize_borrowed_buffer_preserves_existing_bytes() {
        let backing = [1u8, 2, 3, 4];
        let pool = ZSTDMT_bufferPool {
            bufferSize: 8,
            totalBuffers: 1,
            buffers: Vec::new(),
            _cMem: ZSTD_customMem::default(),
        };
        let borrowed = Buffer::borrowed(backing.as_ptr() as usize, backing.len());

        let resized = ZSTDMT_resizeBuffer(&pool, borrowed);

        assert_eq!(resized.capacity(), 8);
        assert_eq!(&resized.data[..backing.len()], &backing);
        assert_eq!(&resized.data[backing.len()..], &[0, 0, 0, 0]);
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
    fn zstdmt_create_job_prepares_ldm_sequences_serially() {
        let mut params = ZSTD_CCtx_params::default();
        ZSTD_CCtxParams_init(&mut params, 3);
        params.nbWorkers = 1;
        params.ldmEnable = ZSTD_ParamSwitch_e::ZSTD_ps_enable;
        params.ldmParams.enableLdm = ZSTD_ParamSwitch_e::ZSTD_ps_enable;
        params.ldmParams.minMatchLength = 64;
        params.ldmParams.hashLog = 8;
        params.ldmParams.bucketSizeLog = 4;
        params.ldmParams.hashRateLog = 4;

        let mut mt = ZSTDMT_createCCtx(1).expect("mt");
        assert_eq!(ZSTDMT_initCStream_internal(&mut mt, params, 0), 0);

        let src = vec![b'a'; 4096];
        mt.inBuff.buffer = Buffer::from_vec(src.clone());
        mt.inBuff.filled = src.len();

        assert_eq!(
            ZSTDMT_createCompressionJob(&mut mt, src.len(), ZSTD_EndDirective::ZSTD_e_end),
            0
        );
        let job = &mt.jobs[0];
        assert!(job.rawSeqStore.capacity > 0);
        assert!(mt
            .serial
            .ldmState
            .as_ref()
            .is_some_and(|state| !state.hashTable.is_empty()));
    }

    #[test]
    fn zstdmt_create_job_takes_dst_buffer_from_pool() {
        let mut params = ZSTD_CCtx_params::default();
        ZSTD_CCtxParams_init(&mut params, 3);
        params.nbWorkers = 1;

        let mut mt = ZSTDMT_createCCtx(1).expect("mt");
        assert_eq!(ZSTDMT_initCStream_internal(&mut mt, params, 0), 0);
        {
            let pool = mt.bufPool.as_mut().expect("buf pool");
            ZSTDMT_setBufferSize(pool, 1024);
            ZSTDMT_releaseBuffer(pool, Buffer::from_vec(vec![0xA5; 1024]));
            assert_eq!(pool.buffers.len(), 1);
        }

        let src = b"pooled destination buffer job payload".to_vec();
        mt.inBuff.buffer = Buffer::from_vec(src.clone());
        mt.inBuff.filled = src.len();

        assert_eq!(
            ZSTDMT_createCompressionJob(&mut mt, src.len(), ZSTD_EndDirective::ZSTD_e_end),
            0
        );
        assert_eq!(mt.bufPool.as_ref().unwrap().buffers.len(), 0);
        assert_eq!(mt.jobs[0].dstBuff.capacity(), 1024);

        ZSTDMT_releaseAllJobResources(&mut mt);
    }

    #[test]
    fn zstdmt_init_sizes_round_buffer_like_upstream_slack_window() {
        let mut params = ZSTD_CCtx_params::default();
        ZSTD_CCtxParams_init(&mut params, 3);
        params.nbWorkers = 2;
        params.jobSize = ZSTD_BLOCKSIZE_MAX;
        params.overlapLog = 1;

        let mut mt = ZSTDMT_createCCtx(2).expect("mt");
        assert_eq!(ZSTDMT_initCStream_internal(&mut mt, params, 0), 0);

        let nb_slack_buffers = 2 + usize::from(mt.targetPrefixSize > 0);
        let expected = mt
            .targetSectionSize
            .saturating_mul(mt.params.nbWorkers as usize)
            .saturating_add(mt.targetSectionSize.saturating_mul(nb_slack_buffers));
        assert_eq!(mt.roundBuff.capacity, expected);
        assert_eq!(mt.roundBuff.pos, 0);
    }

    #[test]
    fn zstdmt_try_get_input_range_borrows_round_buffer_memory() {
        let mut params = ZSTD_CCtx_params::default();
        ZSTD_CCtxParams_init(&mut params, 3);
        params.nbWorkers = 1;
        params.jobSize = ZSTDMT_JOBSIZE_MIN;
        params.overlapLog = 1;

        let mut mt = ZSTDMT_createCCtx(1).expect("mt");
        assert_eq!(ZSTDMT_initCStream_internal(&mut mt, params, 0), 0);

        let round_start = mt.roundBuff.buffer.start();
        assert_eq!(ZSTDMT_tryGetInputRange(&mut mt), 1);
        assert!(mt.inBuff.buffer.is_borrowed());
        assert_eq!(mt.inBuff.buffer.start(), round_start);
        assert_eq!(mt.inBuff.buffer.capacity(), mt.targetSectionSize);
        assert!(mt.inBuff.buffer.data.is_empty());

        let src = b"range lives in round buffer";
        assert!(ZSTDMT_copyToInputBuffer(&mut mt, src));
        mt.inBuff.filled = src.len();
        assert_eq!(&mt.roundBuff.buffer.data[..src.len()], src);

        assert_eq!(
            ZSTDMT_createCompressionJob(&mut mt, src.len(), ZSTD_EndDirective::ZSTD_e_flush),
            0
        );
        let job = &mt.jobs[0];
        assert!(job.srcBuff.is_borrowed());
        assert_eq!(job.src.start, round_start);
        assert_eq!(range_as_slice(job.src), src);
    }

    #[test]
    fn zstdmt_try_get_input_range_wraps_prefix_into_round_buffer_start() {
        let mut mt = ZSTDMT_CCtx {
            targetSectionSize: 10,
            targetPrefixSize: 4,
            roundBuff: RoundBuff_t {
                buffer: Buffer::from_vec(vec![0; 32]),
                capacity: 32,
                pos: 28,
            },
            ..ZSTDMT_CCtx::default()
        };
        let round_start = mt.roundBuff.buffer.start();
        mt.roundBuff.buffer.data[24..28].copy_from_slice(b"pref");
        mt.inBuff.prefix = Range {
            start: round_start + 24,
            size: 4,
        };

        assert_eq!(ZSTDMT_tryGetInputRange(&mut mt), 1);

        assert_eq!(&mt.roundBuff.buffer.data[..4], b"pref");
        assert_eq!(
            mt.inBuff.prefix,
            Range {
                start: round_start,
                size: 4
            }
        );
        assert!(mt.inBuff.buffer.is_borrowed());
        assert_eq!(mt.inBuff.buffer.start(), round_start + 4);
        assert_eq!(mt.inBuff.buffer.capacity(), mt.targetSectionSize);
        assert_eq!(mt.roundBuff.pos, 4);

        assert!(ZSTDMT_copyToInputBuffer(&mut mt, b"0123456789"));
        mt.inBuff.filled = 10;
        assert_eq!(&mt.roundBuff.buffer.data[4..14], b"0123456789");
    }

    #[test]
    fn zstdmt_worker_cctx_returns_to_pool_after_join() {
        let mut params = ZSTD_CCtx_params::default();
        ZSTD_CCtxParams_init(&mut params, 3);
        params.nbWorkers = 1;

        let mut mt = ZSTDMT_createCCtx(1).expect("mt");
        assert_eq!(ZSTDMT_initCStream_internal(&mut mt, params, 0), 0);
        assert_eq!(mt.cctxPool.as_ref().unwrap().cctxs.len(), 1);

        mt.inBuff.buffer = Buffer::from_vec(b"x".to_vec());
        mt.inBuff.filled = mt.inBuff.buffer.data.len();
        let job_size = mt.inBuff.filled;

        assert_eq!(
            ZSTDMT_createCompressionJob(&mut mt, job_size, ZSTD_EndDirective::ZSTD_e_end),
            0
        );
        let wJobID = (mt.doneJobID & mt.jobIDMask) as usize;
        assert_eq!(ZSTDMT_spawnCompressionJob(&mut mt, wJobID), 0);
        assert_eq!(mt.cctxPool.as_ref().unwrap().cctxs.len(), 0);
        assert_eq!(ZSTDMT_joinPendingJob(&mut mt, wJobID), 0);
        assert_eq!(mt.cctxPool.as_ref().unwrap().cctxs.len(), 1);
    }

    #[test]
    fn zstdmt_empty_final_job_after_prior_job_is_not_posted_to_worker() {
        use crate::common::error::ERR_isError;

        let mut params = ZSTD_CCtx_params::default();
        ZSTD_CCtxParams_init(&mut params, 3);
        params.nbWorkers = 1;

        let mut mt = ZSTDMT_createCCtx(1).expect("mt");
        assert_eq!(ZSTDMT_initCStream_internal(&mut mt, params, 0), 0);

        let mut out = vec![0u8; ZSTD_compressBound(16) + 16];
        let mut out_pos = 0usize;
        let mut input_pos = 0usize;
        let mut remaining = ZSTDMT_compressStream_generic(
            &mut mt,
            &mut out,
            &mut out_pos,
            b"first job",
            &mut input_pos,
            ZSTD_EndDirective::ZSTD_e_flush,
        );
        let empty = [];
        while remaining != 0 {
            let mut empty_pos = 0usize;
            remaining = ZSTDMT_compressStream_generic(
                &mut mt,
                &mut out,
                &mut out_pos,
                &empty,
                &mut empty_pos,
                ZSTD_EndDirective::ZSTD_e_flush,
            );
        }
        assert_eq!(mt.nextJobID, 1);
        assert_eq!(mt.doneJobID, 1);
        assert_eq!(mt.frameEnded, 0);

        POOL_free(mt.threadPool.take());
        mt.threadPoolRef = 0;
        mt.rayonThreadPoolRef = 0;

        let before_final = mt.nextJobID;
        let mut empty_pos = 0usize;
        let rc = ZSTDMT_compressStream_generic(
            &mut mt,
            &mut out,
            &mut out_pos,
            &empty,
            &mut empty_pos,
            ZSTD_EndDirective::ZSTD_e_end,
        );

        assert!(
            !ERR_isError(rc),
            "empty final job must not require a worker"
        );
        assert_eq!(rc, 0);
        assert_eq!(mt.nextJobID, before_final + 1);
        assert_eq!(mt.doneJobID, mt.nextJobID);
        assert_eq!(mt.frameEnded, 1);
        assert!(mt.jobReceivers.iter().all(Option::is_none));
    }

    #[test]
    fn zstdmt_compress_stream_generic_roundtrips_single_job_frame() {
        use crate::decompress::zstd_decompress::ZSTD_decompress;

        let src = b"the quick brown fox jumps over the lazy dog";
        let mut params = ZSTD_CCtx_params::default();
        ZSTD_CCtxParams_init(&mut params, 3);
        params.nbWorkers = 1;
        let mut mt = ZSTDMT_createCCtx(1).expect("mt");
        assert_eq!(
            ZSTDMT_initCStream_internal(&mut mt, params, src.len() as u64),
            0
        );

        let mut compressed = vec![0u8; ZSTD_compressBound(src.len())];
        let mut cpos = 0usize;
        let mut spos = 0usize;
        let mut rem = ZSTDMT_compressStream_generic(
            &mut mt,
            &mut compressed,
            &mut cpos,
            src,
            &mut spos,
            ZSTD_EndDirective::ZSTD_e_end,
        );
        let empty = [];
        while rem != 0 {
            let mut empty_pos = 0usize;
            rem = ZSTDMT_compressStream_generic(
                &mut mt,
                &mut compressed,
                &mut cpos,
                &empty,
                &mut empty_pos,
                ZSTD_EndDirective::ZSTD_e_end,
            );
        }
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
        let empty = [];
        let mut empty_pos = 0usize;
        while remaining != 0 && first_pos == 0 {
            empty_pos = 0;
            remaining = ZSTDMT_compressStream_generic(
                &mut mt,
                &mut first,
                &mut first_pos,
                &empty,
                &mut empty_pos,
                ZSTD_EndDirective::ZSTD_e_end,
            );
        }
        assert_eq!(input_pos, src.len());
        assert!(remaining > 0);
        assert!(ZSTDMT_toFlushNow(&mt) > 0);

        let mut tail = vec![0u8; remaining];
        let mut tail_pos = 0usize;
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
    fn zstdmt_multijob_checksum_trailer_is_emitted_and_truncation_fails() {
        use crate::common::error::ERR_isError;
        use crate::decompress::zstd_decompress::{ZSTD_decompress, ZSTD_CONTENTSIZE_UNKNOWN};

        let src = vec![b'M'; ZSTDMT_JOBSIZE_MIN * 2 + 4096];
        let mut params = ZSTD_CCtx_params::default();
        ZSTD_CCtxParams_init(&mut params, 3);
        params.nbWorkers = 2;
        params.jobSize = ZSTDMT_JOBSIZE_MIN;
        params.overlapLog = 1;
        params.fParams.checksumFlag = 1;

        let mut mt = ZSTDMT_createCCtx(2).expect("mt");
        assert_eq!(
            ZSTDMT_initCStream_internal(&mut mt, params, ZSTD_CONTENTSIZE_UNKNOWN),
            0
        );

        let mut compressed = vec![0u8; ZSTD_compressBound(src.len()) + 64 * 1024];
        let mut cpos = 0usize;
        let mut spos = 0usize;
        let mut remaining = ZSTDMT_compressStream_generic(
            &mut mt,
            &mut compressed,
            &mut cpos,
            &src,
            &mut spos,
            ZSTD_EndDirective::ZSTD_e_end,
        );
        while remaining != 0 {
            if spos < src.len() {
                remaining = ZSTDMT_compressStream_generic(
                    &mut mt,
                    &mut compressed,
                    &mut cpos,
                    &src,
                    &mut spos,
                    ZSTD_EndDirective::ZSTD_e_end,
                );
            } else {
                let empty = [];
                let mut empty_pos = 0usize;
                remaining = ZSTDMT_compressStream_generic(
                    &mut mt,
                    &mut compressed,
                    &mut cpos,
                    &empty,
                    &mut empty_pos,
                    ZSTD_EndDirective::ZSTD_e_end,
                );
            }
        }

        assert_eq!(spos, src.len());
        assert!(mt.nextJobID > 1, "test must exercise multiple MT jobs");
        assert_ne!(
            compressed[4] & 0x04,
            0,
            "frame header must request checksum"
        );

        let expected = crate::common::xxhash::XXH64(&src, 0) as u32;
        let trailer = crate::common::mem::MEM_readLE32(&compressed[cpos - 4..cpos]);
        assert_eq!(trailer, expected, "MT frame must end with XXH64 trailer");

        let mut decoded = vec![0u8; src.len()];
        let dsize = ZSTD_decompress(&mut decoded, &compressed[..cpos]);
        assert_eq!(dsize, src.len());
        assert_eq!(decoded, src);

        let truncated = &compressed[..cpos - 4];
        let mut rejected = vec![0u8; src.len() + 4096];
        let rc = ZSTD_decompress(&mut rejected, truncated);
        assert!(
            ERR_isError(rc),
            "checksum frame without trailer must be rejected (rc={rc})"
        );
    }

    #[test]
    fn zstdmt_compress_stream_generic_consumes_by_target_section() {
        let src = vec![b'a'; ZSTDMT_JOBSIZE_MIN + 4096];
        let mut params = ZSTD_CCtx_params::default();
        ZSTD_CCtxParams_init(&mut params, 3);
        params.nbWorkers = 1;
        params.jobSize = ZSTDMT_JOBSIZE_MIN;
        params.overlapLog = 1;

        let mut mt = ZSTDMT_createCCtx(1).expect("mt");
        assert_eq!(
            ZSTDMT_initCStream_internal(&mut mt, params, src.len() as u64),
            0
        );
        assert_eq!(mt.targetSectionSize, ZSTDMT_JOBSIZE_MIN);

        let mut out = vec![0u8; ZSTD_compressBound(src.len())];
        let mut out_pos = 0usize;
        let mut input_pos = 0usize;
        let remaining = ZSTDMT_compressStream_generic(
            &mut mt,
            &mut out,
            &mut out_pos,
            &src,
            &mut input_pos,
            ZSTD_EndDirective::ZSTD_e_end,
        );

        assert_eq!(input_pos, mt.targetSectionSize);
        assert!(remaining > 0);
        assert_eq!(mt.nextJobID, 1);
        assert_ne!(mt.frameEnded, 1);
    }

    #[test]
    fn zstdmt_get_frame_progression_ignores_errored_job_output() {
        use crate::common::error::{ErrorCode, ERROR};

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
            dstBuff: Buffer::from_vec(vec![1, 2, 3]),
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
        use crate::common::error::{ERR_isError, ErrorCode, ERROR};

        let mut mt = ZSTDMT_createCCtx(1).expect("mt");
        mt.nextJobID = 1;
        mt.doneJobID = 0;
        mt.jobIDMask = 3;
        mt.inBuff.buffer = Buffer::from_vec(vec![9, 8, 7]);
        mt.inBuff.filled = 3;
        mt.allJobsCompleted = 0;
        mt.jobs[0] = ZSTDMT_jobDescription {
            src: Range { start: 0, size: 4 },
            consumed: 1,
            cSize: ERROR(ErrorCode::MemoryAllocation),
            dstBuff: Buffer::from_vec(vec![1, 2, 3]),
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
        assert!(mt
            .jobs
            .iter()
            .all(|job| job.cSize == 0 && job.dstBuff.data.is_empty()));
    }

    #[test]
    fn zstdmt_get_input_data_in_use_skips_first_round_buffer_pass() {
        let mut mt = ZSTDMT_CCtx {
            targetSectionSize: 10,
            roundBuff: RoundBuff_t {
                capacity: 40,
                ..RoundBuff_t::default()
            },
            doneJobID: 0,
            nextJobID: 3,
            jobIDMask: 3,
            jobs: vec![ZSTDMT_jobDescription::default(); 4],
            ..ZSTDMT_CCtx::default()
        };
        mt.jobs[0] = ZSTDMT_jobDescription {
            src: Range {
                start: 1000,
                size: 10,
            },
            consumed: 0,
            ..ZSTDMT_jobDescription::default()
        };

        assert_eq!(ZSTDMT_getInputDataInUse(&mt), Range::default());

        mt.nextJobID = 4;
        assert_eq!(
            ZSTDMT_getInputDataInUse(&mt),
            Range {
                start: 1000,
                size: 10
            }
        );
    }

    #[test]
    fn find_synchronization_point_flushes_immediately_when_buffer_tail_already_hits() {
        let mut mt = ZSTDMT_CCtx::default();
        mt.params.rsyncable = 1;
        mt.targetSectionSize = ZSTDMT_JOBSIZE_MIN;
        mt.inBuff.buffer = Buffer::from_vec(vec![b'a'; ZSTDMT_JOBSIZE_MIN]);
        mt.inBuff.filled = ZSTDMT_JOBSIZE_MIN;
        let tail_hash = ZSTD_rollingHash_compute(
            &mt.inBuff.buffer.data[mt.inBuff.filled - RSYNC_LENGTH..mt.inBuff.filled],
        );
        mt.rsync.primePower =
            crate::compress::zstd_hashes::ZSTD_rollingHash_primePower(RSYNC_LENGTH as u32);
        mt.rsync.hitMask = tail_hash;

        let sync = findSynchronizationPoint(&mt, b"trailing input", 0);
        assert_eq!(sync.toLoad, 0);
        assert_eq!(sync.flush, 1);
    }

    #[test]
    fn zstdmt_init_rsync_mask_uses_target_section_kilobytes() {
        let mut params = ZSTD_CCtx_params::default();
        ZSTD_CCtxParams_init(&mut params, 3);
        params.nbWorkers = 1;
        params.jobSize = ZSTDMT_JOBSIZE_MIN;
        params.rsyncable = 1;

        let mut mt = ZSTDMT_createCCtx(1).expect("mt");
        assert_eq!(ZSTDMT_initCStream_internal(&mut mt, params, 0), 0);

        let rsync_bits = ZSTD_highbit32((ZSTDMT_JOBSIZE_MIN >> 10) as u32) + 10;
        assert_eq!(rsync_bits, 19);
        assert_eq!(mt.rsync.hitMask, (1u64 << rsync_bits) - 1);
    }

    #[test]
    fn find_synchronization_point_waits_for_minimum_rsync_block_size() {
        let mut mt = ZSTDMT_CCtx::default();
        mt.params.rsyncable = 1;
        mt.targetSectionSize = ZSTDMT_JOBSIZE_MIN;
        mt.inBuff.buffer = Buffer::from_vec(vec![b'a'; RSYNC_LENGTH]);
        mt.inBuff.filled = RSYNC_LENGTH;
        mt.rsync.primePower =
            crate::compress::zstd_hashes::ZSTD_rollingHash_primePower(RSYNC_LENGTH as u32);
        mt.rsync.hitMask = 0;

        let sync = findSynchronizationPoint(&mt, b"short rsync input", 0);
        assert_eq!(sync.toLoad, b"short rsync input".len());
        assert_eq!(sync.flush, 0);
    }

    #[test]
    fn find_synchronization_point_uses_buffered_tail_when_scanning_new_input() {
        let mut mt = ZSTDMT_CCtx::default();
        mt.params.rsyncable = 1;
        mt.targetSectionSize = ZSTDMT_JOBSIZE_MIN + 3;
        mt.inBuff.buffer = Buffer::from_vec(vec![b'b'; ZSTDMT_JOBSIZE_MIN]);
        mt.inBuff.filled = ZSTDMT_JOBSIZE_MIN;
        mt.rsync.primePower =
            crate::compress::zstd_hashes::ZSTD_rollingHash_primePower(RSYNC_LENGTH as u32);

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
