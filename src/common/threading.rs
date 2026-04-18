//! Translation of `lib/common/threading.c` — thin wrappers around
//! pthread/Win32 primitives. In pure Rust we'll map these onto
//! `std::sync::Mutex` / `Condvar` / `std::thread` during Phase 2.

#![allow(unused_variables)]
