//! Translation of `lib/common/threading.c` / `threading.h`.
//!
//! The upstream layer is mostly a portability shim around pthread /
//! Win32 primitives. In Rust we expose the same conceptual surface on
//! top of `std::sync` and `std::thread`.

use std::ffi::c_void;
use std::sync::{Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;

pub struct ZSTD_pthread_mutex_t {
    inner: Mutex<()>,
}

impl Default for ZSTD_pthread_mutex_t {
    fn default() -> Self {
        Self {
            inner: Mutex::new(()),
        }
    }
}

pub type ZSTD_pthread_mutex_guard<'a> = MutexGuard<'a, ()>;

pub struct ZSTD_pthread_cond_t {
    inner: Condvar,
}

impl Default for ZSTD_pthread_cond_t {
    fn default() -> Self {
        Self {
            inner: Condvar::new(),
        }
    }
}

#[derive(Default)]
pub struct ZSTD_pthread_t {
    inner: Option<JoinHandle<usize>>,
}

pub type ZSTD_thread_routine = fn(*mut c_void) -> *mut c_void;

/// Port of `ZSTD_pthread_mutex_init`.
pub fn ZSTD_pthread_mutex_init(mutex: &mut Option<ZSTD_pthread_mutex_t>, _attr: Option<()>) -> i32 {
    *mutex = Some(ZSTD_pthread_mutex_t::default());
    0
}

/// Port of `ZSTD_pthread_mutex_destroy`.
pub fn ZSTD_pthread_mutex_destroy(mutex: &mut Option<ZSTD_pthread_mutex_t>) -> i32 {
    *mutex = None;
    0
}

/// Port of `ZSTD_pthread_mutex_lock`.
pub fn ZSTD_pthread_mutex_lock<'a>(
    mutex: &'a ZSTD_pthread_mutex_t,
) -> ZSTD_pthread_mutex_guard<'a> {
    mutex.inner.lock().expect("threading mutex")
}

/// Port of `ZSTD_pthread_mutex_unlock`.
pub fn ZSTD_pthread_mutex_unlock(guard: ZSTD_pthread_mutex_guard<'_>) -> i32 {
    drop(guard);
    0
}

/// Port of `ZSTD_pthread_cond_init`.
pub fn ZSTD_pthread_cond_init(cond: &mut Option<ZSTD_pthread_cond_t>, _attr: Option<()>) -> i32 {
    *cond = Some(ZSTD_pthread_cond_t::default());
    0
}

/// Port of `ZSTD_pthread_cond_destroy`.
pub fn ZSTD_pthread_cond_destroy(cond: &mut Option<ZSTD_pthread_cond_t>) -> i32 {
    *cond = None;
    0
}

/// Port of `ZSTD_pthread_cond_wait`.
pub fn ZSTD_pthread_cond_wait<'a>(
    cond: &ZSTD_pthread_cond_t,
    guard: ZSTD_pthread_mutex_guard<'a>,
) -> ZSTD_pthread_mutex_guard<'a> {
    cond.inner.wait(guard).expect("threading condvar")
}

/// Port of `ZSTD_pthread_cond_signal`.
pub fn ZSTD_pthread_cond_signal(cond: &ZSTD_pthread_cond_t) {
    cond.inner.notify_one();
}

/// Port of `ZSTD_pthread_cond_broadcast`.
pub fn ZSTD_pthread_cond_broadcast(cond: &ZSTD_pthread_cond_t) {
    cond.inner.notify_all();
}

/// Port of `ZSTD_pthread_create`.
pub fn ZSTD_pthread_create(
    thread: &mut ZSTD_pthread_t,
    _unused: Option<()>,
    start_routine: ZSTD_thread_routine,
    arg: *mut c_void,
) -> i32 {
    let arg_bits = arg as usize;
    thread.inner = Some(std::thread::spawn(move || {
        start_routine(arg_bits as *mut c_void) as usize
    }));
    0
}

/// Port of `ZSTD_pthread_join`.
pub fn ZSTD_pthread_join(thread: &mut ZSTD_pthread_t) -> i32 {
    match thread.inner.take() {
        Some(handle) => match handle.join() {
            Ok(_) => 0,
            Err(_) => 1,
        },
        None => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn mutex_and_condvar_wait_signal_roundtrip() {
        let mut mutex = None;
        let mut cond = None;
        assert_eq!(ZSTD_pthread_mutex_init(&mut mutex, None), 0);
        assert_eq!(ZSTD_pthread_cond_init(&mut cond, None), 0);
        let mutex = mutex.expect("mutex");
        let cond = cond.expect("cond");

        struct WaitSignalArgs {
            cond: Arc<ZSTD_pthread_cond_t>,
            ready: Arc<AtomicBool>,
        }

        let cond = Arc::new(cond);
        let ready = Arc::new(AtomicBool::new(false));
        let args = Box::new(WaitSignalArgs {
            cond: Arc::clone(&cond),
            ready: Arc::clone(&ready),
        });

        let mut thread = ZSTD_pthread_t::default();
        fn signal_ready(arg: *mut c_void) -> *mut c_void {
            let args = unsafe { Box::from_raw(arg as *mut WaitSignalArgs) };
            std::thread::sleep(Duration::from_millis(10));
            args.ready.store(true, Ordering::SeqCst);
            ZSTD_pthread_cond_signal(&args.cond);
            core::ptr::null_mut()
        }
        assert_eq!(
            ZSTD_pthread_create(
                &mut thread,
                None,
                signal_ready,
                Box::into_raw(args) as *mut c_void,
            ),
            0
        );

        let mut guard = ZSTD_pthread_mutex_lock(&mutex);
        while !ready.load(Ordering::SeqCst) {
            guard = ZSTD_pthread_cond_wait(&cond, guard);
        }
        assert_eq!(ZSTD_pthread_mutex_unlock(guard), 0);
        assert_eq!(ZSTD_pthread_join(&mut thread), 0);
    }

    #[test]
    fn thread_create_and_join_executes_start_routine() {
        let seen = Arc::new(AtomicUsize::new(0));
        let raw = Arc::into_raw(Arc::clone(&seen)) as *mut c_void;

        fn bump(arg: *mut c_void) -> *mut c_void {
            let seen = unsafe { Arc::<AtomicUsize>::from_raw(arg as *const AtomicUsize) };
            seen.fetch_add(1, Ordering::SeqCst);
            let _ = Arc::into_raw(seen);
            core::ptr::null_mut()
        }

        let mut thread = ZSTD_pthread_t::default();
        assert_eq!(ZSTD_pthread_create(&mut thread, None, bump, raw), 0);
        assert_eq!(ZSTD_pthread_join(&mut thread), 0);
        assert_eq!(seen.load(Ordering::SeqCst), 1);
        assert_eq!(ZSTD_pthread_join(&mut thread), 0);
    }

    #[test]
    fn condvar_broadcast_wakes_multiple_waiters() {
        let cond = Arc::new(ZSTD_pthread_cond_t::default());
        let started = Arc::new(AtomicUsize::new(0));
        let awakened = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(AtomicBool::new(false));

        let mut handles = Vec::new();
        for _ in 0..2 {
            let cond = Arc::clone(&cond);
            let started = Arc::clone(&started);
            let awakened = Arc::clone(&awakened);
            let gate = Arc::clone(&gate);
            handles.push(std::thread::spawn(move || {
                let mutex = ZSTD_pthread_mutex_t::default();
                let mut guard = ZSTD_pthread_mutex_lock(&mutex);
                started.fetch_add(1, Ordering::SeqCst);
                while !gate.load(Ordering::SeqCst) {
                    guard = ZSTD_pthread_cond_wait(&cond, guard);
                }
                awakened.fetch_add(1, Ordering::SeqCst);
                let _ = ZSTD_pthread_mutex_unlock(guard);
            }));
        }

        while started.load(Ordering::SeqCst) != 2 {
            std::thread::yield_now();
        }
        gate.store(true, Ordering::SeqCst);
        ZSTD_pthread_cond_broadcast(&cond);

        for handle in handles {
            handle.join().expect("waiter");
        }
        assert_eq!(awakened.load(Ordering::SeqCst), 2);
    }
}
