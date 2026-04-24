//! Translation of `lib/common/pool.c` — a tiny thread pool used by
//! multi-threaded compression.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

#[derive(Clone, Copy)]
struct Task {
    job: fn(*mut core::ffi::c_void),
    opaque: usize,
}

#[derive(Default)]
struct PoolState {
    queue: VecDeque<Task>,
    queue_limit: usize,
    active_jobs: usize,
    shutdown: bool,
    retiring_workers: usize,
    live_workers: usize,
}

struct Shared {
    state: Mutex<PoolState>,
    work_ready: Condvar,
    queue_space: Condvar,
    idle: Condvar,
}

pub struct POOL_ctx {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl Drop for POOL_ctx {
    fn drop(&mut self) {
        {
            let mut state = self.shared.state.lock().expect("pool mutex");
            state.shutdown = true;
            self.shared.work_ready.notify_all();
            self.shared.queue_space.notify_all();
            self.shared.idle.notify_all();
        }
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

fn pool_thread(shared: Arc<Shared>) {
    loop {
        let task = {
            let mut state = shared.state.lock().expect("pool mutex");
            loop {
                if state.shutdown {
                    state.live_workers = state.live_workers.saturating_sub(1);
                    shared.idle.notify_all();
                    return;
                }
                if let Some(task) = state.queue.pop_front() {
                    state.active_jobs += 1;
                    shared.queue_space.notify_one();
                    break task;
                }
                if state.retiring_workers > 0 {
                    state.retiring_workers -= 1;
                    state.live_workers = state.live_workers.saturating_sub(1);
                    shared.idle.notify_all();
                    return;
                }
                state = shared.work_ready.wait(state).expect("pool condvar");
            }
        };

        (task.job)(task.opaque as *mut core::ffi::c_void);

        let mut state = shared.state.lock().expect("pool mutex");
        state.active_jobs = state.active_jobs.saturating_sub(1);
        if state.active_jobs == 0 && state.queue.is_empty() {
            shared.idle.notify_all();
        }
    }
}

fn spawn_workers(ctx: &mut POOL_ctx, count: usize) {
    for _ in 0..count {
        let shared = Arc::clone(&ctx.shared);
        ctx.workers.push(thread::spawn(move || pool_thread(shared)));
    }
    let mut state = ctx.shared.state.lock().expect("pool mutex");
    state.live_workers += count;
}

fn join_finished_workers(ctx: &mut POOL_ctx) {
    let mut idx = 0usize;
    while idx < ctx.workers.len() {
        if ctx.workers[idx].is_finished() {
            let handle = ctx.workers.swap_remove(idx);
            let _ = handle.join();
        } else {
            idx += 1;
        }
    }
}

/// Port of `POOL_create`.
pub fn POOL_create(numThreads: usize, queueSize: usize) -> Option<Box<POOL_ctx>> {
    POOL_create_advanced(
        numThreads,
        queueSize,
        crate::compress::zstd_compress::ZSTD_customMem::default(),
    )
}

/// Port of `POOL_create_advanced` (pool.c:115).
pub fn POOL_create_advanced(
    numThreads: usize,
    queueSize: usize,
    _customMem: crate::compress::zstd_compress::ZSTD_customMem,
) -> Option<Box<POOL_ctx>> {
    if numThreads == 0 {
        return None;
    }
    let queue_limit = queueSize.max(1);
    let shared = Arc::new(Shared {
        state: Mutex::new(PoolState {
            queue_limit,
            ..PoolState::default()
        }),
        work_ready: Condvar::new(),
        queue_space: Condvar::new(),
        idle: Condvar::new(),
    });
    let mut ctx = Box::new(POOL_ctx {
        shared,
        workers: Vec::with_capacity(numThreads),
    });
    spawn_workers(&mut ctx, numThreads);
    Some(ctx)
}

/// Port of `POOL_free`.
pub fn POOL_free(ctx: Option<Box<POOL_ctx>>) {
    drop(ctx);
}

/// Port of `POOL_sizeof`.
pub fn POOL_sizeof(ctx: &POOL_ctx) -> usize {
    let state = ctx.shared.state.lock().expect("pool mutex");
    core::mem::size_of::<POOL_ctx>()
        + core::mem::size_of::<Shared>()
        + core::mem::size_of::<PoolState>()
        + ctx.workers.capacity() * core::mem::size_of::<JoinHandle<()>>()
        + state.queue.capacity() * core::mem::size_of::<Task>()
}

/// Port of `POOL_add`.
pub fn POOL_add(ctx: &POOL_ctx, job: fn(*mut core::ffi::c_void), opaque: *mut core::ffi::c_void) {
    let mut state = ctx.shared.state.lock().expect("pool mutex");
    while !state.shutdown && state.queue.len() >= state.queue_limit {
        state = ctx.shared.queue_space.wait(state).expect("pool condvar");
    }
    if state.shutdown {
        return;
    }
    state.queue.push_back(Task {
        job,
        opaque: opaque as usize,
    });
    ctx.shared.work_ready.notify_one();
}

/// Port of `POOL_tryAdd`.
pub fn POOL_tryAdd(
    ctx: &POOL_ctx,
    job: fn(*mut core::ffi::c_void),
    opaque: *mut core::ffi::c_void,
) -> i32 {
    let mut state = ctx.shared.state.lock().expect("pool mutex");
    if state.shutdown || state.queue.len() >= state.queue_limit {
        return 0;
    }
    state.queue.push_back(Task {
        job,
        opaque: opaque as usize,
    });
    ctx.shared.work_ready.notify_one();
    1
}

/// Port of `POOL_resize`.
pub fn POOL_resize(ctx: &mut POOL_ctx, numThreads: usize) -> usize {
    if numThreads == 0 {
        return 0;
    }
    {
        let mut state = ctx.shared.state.lock().expect("pool mutex");
        let current = state.live_workers;
        if numThreads > current {
            drop(state);
            spawn_workers(ctx, numThreads - current);
        } else if numThreads < current {
            state.retiring_workers += current - numThreads;
            ctx.shared.work_ready.notify_all();
        }
    }
    join_finished_workers(ctx);
    0
}

/// Port of `POOL_joinJobs`.
pub fn POOL_joinJobs(ctx: &POOL_ctx) {
    let mut state = ctx.shared.state.lock().expect("pool mutex");
    while !state.shutdown && (!state.queue.is_empty() || state.active_jobs != 0) {
        state = ctx.shared.idle.wait(state).expect("pool condvar");
    }
}

/// Port of `ZSTD_createThreadPool`.
pub fn ZSTD_createThreadPool(numThreads: usize) -> Option<Box<POOL_ctx>> {
    POOL_create(numThreads, 1)
}

/// Port of `ZSTD_freeThreadPool`.
pub fn ZSTD_freeThreadPool(pool: Option<Box<POOL_ctx>>) {
    POOL_free(pool)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn pool_add_tryadd_resize_and_joinjobs_execute_work() {
        let mut pool = POOL_create(2, 4).expect("pool");
        let seen = Arc::new(AtomicUsize::new(0));

        fn bump(ptr: *mut core::ffi::c_void) {
            let counter = unsafe { Arc::<AtomicUsize>::from_raw(ptr as *const AtomicUsize) };
            counter.fetch_add(1, Ordering::SeqCst);
            let _ = Arc::into_raw(counter);
        }

        for _ in 0..2 {
            let raw = Arc::into_raw(Arc::clone(&seen)) as *mut core::ffi::c_void;
            POOL_add(&mut pool, bump, raw);
        }
        let raw = Arc::into_raw(Arc::clone(&seen)) as *mut core::ffi::c_void;
        assert_eq!(POOL_tryAdd(&mut pool, bump, raw), 1);
        POOL_joinJobs(&mut pool);
        assert_eq!(seen.load(Ordering::SeqCst), 3);

        assert_eq!(POOL_resize(&mut pool, 4), 0);
        assert!(POOL_sizeof(&pool) > 0);
    }

    #[test]
    fn pool_tryadd_reports_full_queue() {
        let mut pool = POOL_create(1, 1).expect("pool");

        fn wait_once(ptr: *mut core::ffi::c_void) {
            let pair = unsafe {
                Arc::<(AtomicUsize, AtomicUsize)>::from_raw(
                    ptr as *const (AtomicUsize, AtomicUsize),
                )
            };
            pair.1.store(1, Ordering::SeqCst);
            while pair.0.load(Ordering::SeqCst) == 0 {
                thread::sleep(Duration::from_millis(1));
            }
            let _ = Arc::into_raw(pair);
        }

        let pair = Arc::new((AtomicUsize::new(0), AtomicUsize::new(0)));
        let raw1 = Arc::into_raw(Arc::clone(&pair)) as *mut core::ffi::c_void;
        POOL_add(&mut pool, wait_once, raw1);
        while pair.1.load(Ordering::SeqCst) == 0 {
            thread::yield_now();
        }
        let raw2 = Arc::into_raw(Arc::clone(&pair)) as *mut core::ffi::c_void;
        assert_eq!(POOL_tryAdd(&mut pool, wait_once, raw2), 1);
        let raw3 = Arc::into_raw(Arc::clone(&pair)) as *mut core::ffi::c_void;
        assert_eq!(POOL_tryAdd(&mut pool, wait_once, raw3), 0);
        let _ = unsafe {
            Arc::<(AtomicUsize, AtomicUsize)>::from_raw(raw3 as *const (AtomicUsize, AtomicUsize))
        };
        pair.0.store(1, Ordering::SeqCst);
        POOL_joinJobs(&mut pool);
    }

    #[test]
    fn pool_and_thread_pool_creators_construct_headers_for_positive_counts() {
        assert!(POOL_create(4, 8).is_some());
        assert!(POOL_create(1, 0).is_some());
        assert!(POOL_create(0, 0).is_none());
        assert!(ZSTD_createThreadPool(4).is_some());
        assert!(ZSTD_createThreadPool(0).is_none());
        POOL_free(None);
        ZSTD_freeThreadPool(None);
    }
}
