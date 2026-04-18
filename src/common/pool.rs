//! Translation of `lib/common/pool.c` — a tiny thread pool used by
//! multi-threaded compression. v0.1 provides clean API stubs that
//! satisfy callers without spawning threads; a real port will sit on
//! `std::thread` or `rayon` and land with `zstdmt_compress.c`.

#![allow(unused_variables)]

use core::marker::PhantomData;

pub struct POOL_ctx {
    _priv: PhantomData<()>,
}

/// Port of `POOL_create`. Returns `None` — the MT path is not yet
/// active; callers should fall back to single-threaded execution.
pub fn POOL_create(_numThreads: usize, _queueSize: usize) -> Option<Box<POOL_ctx>> {
    None
}

/// Port of `POOL_free`. Drops the Box.
pub fn POOL_free(_ctx: Option<Box<POOL_ctx>>) {}

/// Port of `POOL_sizeof`.
pub fn POOL_sizeof(_ctx: &POOL_ctx) -> usize {
    core::mem::size_of::<POOL_ctx>()
}

/// Port of `POOL_add`. No-op in the stub (no worker to schedule).
pub fn POOL_add(_ctx: &mut POOL_ctx, _job: fn(*mut core::ffi::c_void), _opaque: *mut core::ffi::c_void) {}

/// Port of `POOL_tryAdd`. Returns 0 (not enqueued) in the stub.
pub fn POOL_tryAdd(
    _ctx: &mut POOL_ctx,
    _job: fn(*mut core::ffi::c_void),
    _opaque: *mut core::ffi::c_void,
) -> i32 {
    0
}

/// Port of `POOL_resize`. Returns 0 on success; resizing is a no-op.
pub fn POOL_resize(_ctx: &mut POOL_ctx, _numThreads: usize) -> usize {
    0
}

/// Port of `POOL_joinJobs`. No-op — no jobs are ever queued.
pub fn POOL_joinJobs(_ctx: &mut POOL_ctx) {}

/// Port of `ZSTD_createThreadPool`. Public wrapper over `POOL_create`.
/// MT compression is stubbed in v0.1, so this always returns `None` —
/// callers fall back to single-threaded.
pub fn ZSTD_createThreadPool(_numThreads: usize) -> Option<Box<POOL_ctx>> {
    None
}

/// Port of `ZSTD_freeThreadPool`. Drops a pool allocated via
/// `ZSTD_createThreadPool`. Alias for `POOL_free`.
pub fn ZSTD_freeThreadPool(pool: Option<Box<POOL_ctx>>) {
    POOL_free(pool)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_and_thread_pool_creators_return_none_in_v0_1() {
        // Contract: v0.1 has no MT backend. Both creators must
        // return None regardless of thread count / queue size so
        // callers fall back to single-threaded execution.
        assert!(POOL_create(4, 8).is_none());
        assert!(POOL_create(0, 0).is_none());
        assert!(ZSTD_createThreadPool(4).is_none());
        assert!(ZSTD_createThreadPool(0).is_none());
        // free variants accept None without panicking.
        POOL_free(None);
        ZSTD_freeThreadPool(None);
    }
}
