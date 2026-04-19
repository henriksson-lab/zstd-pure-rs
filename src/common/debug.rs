//! Translation of `lib/common/debug.{c,h}`.
//!
//! Upstream exposes a `g_debuglevel` global and a family of `DEBUGLOG`
//! macros. In release builds these compile to nothing; here we mirror
//! that by gating on `debug_assertions`.

use core::sync::atomic::{AtomicI32, Ordering};

static G_DEBUGLEVEL: AtomicI32 = AtomicI32::new(0);

#[inline]
pub fn set_debug_level(lvl: i32) {
    G_DEBUGLEVEL.store(lvl, Ordering::Relaxed);
}

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
        set_debug_level(0);
        assert_eq!(debug_level(), 0);
        // Restore.
        set_debug_level(prior);
    }
}
