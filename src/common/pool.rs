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
    live_workers: usize,
    thread_limit: usize,
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

/// Port of `POOL_thread` (`pool.c:67`). Worker loop: waits on the
/// condvar for a job, executes it, and signals idle when the queue
/// drains. Exits when shutdown is flagged and no more work should run.
fn pool_thread(shared: Arc<Shared>) {
    loop {
        let task = {
            let mut state = shared.state.lock().expect("pool mutex");
            loop {
                while state.queue.is_empty() || state.active_jobs >= state.thread_limit {
                    if state.shutdown {
                        state.live_workers = state.live_workers.saturating_sub(1);
                        shared.idle.notify_all();
                        return;
                    }
                    state = shared.work_ready.wait(state).expect("pool condvar");
                }
                let task = state.queue.pop_front().expect("non-empty pool queue");
                state.active_jobs += 1;
                shared.queue_space.notify_one();
                break task;
            }
        };

        (task.job)(task.opaque as *mut core::ffi::c_void);

        let mut state = shared.state.lock().expect("pool mutex");
        state.active_jobs = state.active_jobs.saturating_sub(1);
        shared.queue_space.notify_one();
        if state.active_jobs == 0 && state.queue.is_empty() {
            shared.idle.notify_all();
        }
    }
}

/// Rust-only helper: spawn `count` worker threads onto the pool and
/// return how many thread handles were successfully installed.
fn spawn_workers(ctx: &mut POOL_ctx, count: usize) -> Result<usize, ()> {
    if ctx.workers.try_reserve_exact(count).is_err() {
        return Err(());
    }
    let mut spawned = 0usize;
    for _ in 0..count {
        let shared = Arc::clone(&ctx.shared);
        match thread::Builder::new().spawn(move || pool_thread(shared)) {
            Ok(handle) => {
                ctx.workers.push(handle);
                spawned += 1;
            }
            Err(_) => {
                return Err(());
            }
        }
    }
    Ok(spawned)
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
    let queue_size = queueSize.checked_add(1)?;
    let mut queue = VecDeque::new();
    if queue.try_reserve_exact(queue_size).is_err() {
        return None;
    }
    let mut workers = Vec::new();
    if workers.try_reserve_exact(numThreads).is_err() {
        return None;
    }
    let shared = Arc::new(Shared {
        state: Mutex::new(PoolState {
            queue,
            queue_limit: queueSize,
            thread_limit: numThreads,
            ..PoolState::default()
        }),
        work_ready: Condvar::new(),
        queue_space: Condvar::new(),
        idle: Condvar::new(),
    });
    let mut ctx = Box::new(POOL_ctx { shared, workers });
    match spawn_workers(&mut ctx, numThreads) {
        Ok(spawned) if spawned == numThreads => {
            let mut state = ctx.shared.state.lock().expect("pool mutex");
            state.live_workers = spawned;
        }
        Ok(spawned) => {
            let mut state = ctx.shared.state.lock().expect("pool mutex");
            state.live_workers = spawned;
            drop(state);
            drop(ctx);
            return None;
        }
        Err(_) => {
            let mut state = ctx.shared.state.lock().expect("pool mutex");
            state.live_workers = ctx.workers.len();
            drop(state);
            drop(ctx);
            return None;
        }
    }
    Some(ctx)
}

/// Port of `POOL_free`.
pub fn POOL_free(ctx: Option<Box<POOL_ctx>>) {
    drop(ctx);
}

fn pool_sizeof_ctx(ctx: &POOL_ctx) -> usize {
    let state = ctx.shared.state.lock().expect("pool mutex");
    core::mem::size_of::<POOL_ctx>()
        + core::mem::size_of::<Shared>()
        + ctx.workers.len() * core::mem::size_of::<JoinHandle<()>>()
        + (state.queue_limit + 1) * core::mem::size_of::<Task>()
}

pub trait PoolSizeofArg {
    fn pool_sizeof(self) -> usize;
}

impl PoolSizeofArg for &POOL_ctx {
    fn pool_sizeof(self) -> usize {
        pool_sizeof_ctx(self)
    }
}

impl PoolSizeofArg for &Box<POOL_ctx> {
    fn pool_sizeof(self) -> usize {
        pool_sizeof_ctx(self)
    }
}

impl PoolSizeofArg for Option<&POOL_ctx> {
    fn pool_sizeof(self) -> usize {
        self.map_or(0, pool_sizeof_ctx)
    }
}

impl PoolSizeofArg for Option<&Box<POOL_ctx>> {
    fn pool_sizeof(self) -> usize {
        self.map_or(0, |ctx| pool_sizeof_ctx(ctx))
    }
}

/// Port of `POOL_sizeof`.
pub fn POOL_sizeof<C: PoolSizeofArg>(ctx: C) -> usize {
    ctx.pool_sizeof()
}

fn is_queue_full(state: &PoolState) -> bool {
    if state.queue_limit > 0 {
        state.queue.len() >= state.queue_limit
    } else {
        state.active_jobs == state.thread_limit || !state.queue.is_empty()
    }
}

fn pool_add_internal(
    state: &mut PoolState,
    shared: &Shared,
    job: fn(*mut core::ffi::c_void),
    opaque: *mut core::ffi::c_void,
) {
    if state.shutdown {
        return;
    }
    state.queue.push_back(Task {
        job,
        opaque: opaque as usize,
    });
    shared.work_ready.notify_one();
}

/// Port of `POOL_add`.
pub fn POOL_add(ctx: &POOL_ctx, job: fn(*mut core::ffi::c_void), opaque: *mut core::ffi::c_void) {
    let mut state = ctx.shared.state.lock().expect("pool mutex");
    while is_queue_full(&state) && !state.shutdown {
        state = ctx.shared.queue_space.wait(state).expect("pool condvar");
    }
    pool_add_internal(&mut state, &ctx.shared, job, opaque);
}

/// Port of `POOL_tryAdd`.
pub fn POOL_tryAdd(
    ctx: &POOL_ctx,
    job: fn(*mut core::ffi::c_void),
    opaque: *mut core::ffi::c_void,
) -> i32 {
    let mut state = ctx.shared.state.lock().expect("pool mutex");
    if is_queue_full(&state) {
        return 0;
    }
    pool_add_internal(&mut state, &ctx.shared, job, opaque);
    1
}

/// Port of `POOL_resize`.
pub fn POOL_resize(ctx: &mut POOL_ctx, numThreads: usize) -> usize {
    let current = {
        let mut state = ctx.shared.state.lock().expect("pool mutex");
        if numThreads <= state.live_workers {
            if numThreads == 0 {
                return 1;
            }
            state.thread_limit = numThreads;
            ctx.shared.work_ready.notify_all();
            return 0;
        }

        state.thread_limit = numThreads;
        state.live_workers
    };

    let result = match spawn_workers(ctx, numThreads - current) {
        Ok(spawned) => {
            let mut state = ctx.shared.state.lock().expect("pool mutex");
            state.live_workers += spawned;
            if spawned == numThreads - current {
                0
            } else {
                1
            }
        }
        Err(_) => {
            let mut state = ctx.shared.state.lock().expect("pool mutex");
            state.live_workers += ctx.workers.len().saturating_sub(current);
            1
        }
    };
    ctx.shared.work_ready.notify_all();
    result
}

/// Port of `POOL_joinJobs`.
pub fn POOL_joinJobs(ctx: &POOL_ctx) {
    let mut state = ctx.shared.state.lock().expect("pool mutex");
    while !state.queue.is_empty() || state.active_jobs != 0 {
        state = ctx.shared.idle.wait(state).expect("pool condvar");
    }
}

/// Port of `ZSTD_createThreadPool`.
pub fn ZSTD_createThreadPool(numThreads: usize) -> Option<Box<POOL_ctx>> {
    POOL_create(numThreads, 0)
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
        assert_eq!(POOL_resize(&mut pool, 0), 1);
        assert!(POOL_sizeof(&pool) > 0);
        assert_eq!(POOL_sizeof(None::<&POOL_ctx>), 0);
    }

    #[test]
    fn pool_sizeof_accounts_for_preallocated_queue_slots() {
        let small = POOL_create(1, 0).expect("pool");
        let larger = POOL_create(1, 8).expect("pool");

        assert!(POOL_sizeof(&larger) > POOL_sizeof(&small));
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
    fn pool_zero_queue_size_reports_full_when_all_threads_busy() {
        let mut pool = POOL_create(1, 0).expect("pool");

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
        assert_eq!(POOL_tryAdd(&mut pool, wait_once, raw2), 0);
        let _ = unsafe {
            Arc::<(AtomicUsize, AtomicUsize)>::from_raw(raw2 as *const (AtomicUsize, AtomicUsize))
        };
        pair.0.store(1, Ordering::SeqCst);
        POOL_joinJobs(&mut pool);
    }

    #[test]
    fn pool_free_drains_queued_jobs_after_downsize() {
        let mut pool = POOL_create(3, 16).expect("pool");
        let seen = Arc::new(AtomicUsize::new(0));

        fn delayed_bump(ptr: *mut core::ffi::c_void) {
            thread::sleep(Duration::from_millis(2));
            let counter = unsafe { Arc::<AtomicUsize>::from_raw(ptr as *const AtomicUsize) };
            counter.fetch_add(1, Ordering::SeqCst);
            let _ = Arc::into_raw(counter);
        }

        for _ in 0..16 {
            let raw = Arc::into_raw(Arc::clone(&seen)) as *mut core::ffi::c_void;
            POOL_add(&mut pool, delayed_bump, raw);
        }
        assert_eq!(POOL_resize(&mut pool, 1), 0);
        POOL_free(Some(pool));
        assert_eq!(seen.load(Ordering::SeqCst), 16);
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
