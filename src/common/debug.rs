//! Translation of `lib/common/debug.{c,h}`.
//!
//! Upstream exposes a `g_debuglevel` global initialized from the
//! `DEBUGLEVEL` macro and a family of `DEBUGLOG` macros. This port keeps
//! the global verbosity state; Rust call sites use native assertions/logging
//! instead of C preprocessor macros.

use core::sync::atomic::{AtomicI32, Ordering};

/// Default upstream debug level when `DEBUGLEVEL` is not supplied by the
/// compiler command line.
pub const DEBUGLEVEL: i32 = 0;

static G_DEBUGLEVEL: AtomicI32 = AtomicI32::new(DEBUGLEVEL);

/// Rust-only setter for the global `g_debuglevel`. Upstream exposes the
/// global directly; we wrap it in an atomic to keep it safely mutable.
#[inline]
pub fn set_debug_level(lvl: i32) {
    G_DEBUGLEVEL.store(lvl, Ordering::Relaxed);
}

/// Rust-only getter for the global `g_debuglevel`.
#[inline]
pub fn debug_level() -> i32 {
    G_DEBUGLEVEL.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_debug_level_roundtrips_through_debug_level() {
        // Contract: set_debug_level is a one-shot write; debug_level
        // reads the current value. Test the full set→get roundtrip
        // and restore the default so other tests aren't affected.
        let prior = debug_level();
        set_debug_level(3);
        assert_eq!(debug_level(), 3);
        set_debug_level(DEBUGLEVEL);
        assert_eq!(debug_level(), DEBUGLEVEL);
        // Restore.
        set_debug_level(prior);
    }
}
