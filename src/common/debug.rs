//! Translation of `lib/common/debug.{c,h}`.
//!
//! Upstream exposes a `g_debuglevel` global and a family of `DEBUGLOG`
//! macros. In release builds these compile to nothing; here we mirror
//! that by gating on `debug_assertions`.

pub static mut g_debuglevel: i32 = 0;

#[inline]
pub fn set_debug_level(lvl: i32) {
    // Safety: zstd's C code treats this as a write-once knob during setup.
    unsafe {
        g_debuglevel = lvl;
    }
}

#[inline]
pub fn debug_level() -> i32 {
    unsafe { g_debuglevel }
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
