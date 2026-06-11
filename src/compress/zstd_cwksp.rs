//! Translation of `lib/compress/zstd_cwksp.h`.

#![allow(non_snake_case)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use core::ptr::null_mut;

use crate::common::error::{ERR_isError, ErrorCode, ERROR};

pub const ZSTD_CWKSP_ALIGNMENT_BYTES: usize = 64;
pub const ZSTD_CWKSP_ASAN_REDZONE_SIZE: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ZSTD_cwksp_alloc_phase_e {
    ZSTD_cwksp_alloc_objects = 0,
    ZSTD_cwksp_alloc_aligned_init_once = 1,
    ZSTD_cwksp_alloc_aligned = 2,
    ZSTD_cwksp_alloc_buffers = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZSTD_cwksp_static_alloc_e {
    ZSTD_cwksp_dynamic_alloc = 0,
    ZSTD_cwksp_static_alloc = 1,
}

#[derive(Debug, Default)]
pub struct ZSTD_cwksp {
    pub owned: Option<Box<[usize]>>,
    pub workspace: *mut u8,
    pub workspaceEnd: *mut u8,
    pub objectEnd: *mut u8,
    pub tableEnd: *mut u8,
    pub tableValidEnd: *mut u8,
    pub allocStart: *mut u8,
    pub initOnceStart: *mut u8,
    pub allocFailed: u8,
    pub workspaceOversizedDuration: i32,
    pub phase: ZSTD_cwksp_alloc_phase_e,
    pub isStatic: ZSTD_cwksp_static_alloc_e,
}

impl Default for ZSTD_cwksp_alloc_phase_e {
    fn default() -> Self {
        Self::ZSTD_cwksp_alloc_objects
    }
}

impl Default for ZSTD_cwksp_static_alloc_e {
    fn default() -> Self {
        Self::ZSTD_cwksp_dynamic_alloc
    }
}

/// Rust-only internal helper: unsigned `<=` compare of two raw pointers
/// via their addresses, mirroring C's pointer ordering used throughout
/// `zstd_cwksp.h`'s consistency asserts.
#[inline]
fn ptr_le(a: *mut u8, b: *mut u8) -> bool {
    (a as usize) <= (b as usize)
}

/// Rust-only internal helper: unsigned `<` compare of two raw pointers
/// via their addresses, used by the reserve / overflow checks.
#[inline]
fn ptr_lt(a: *mut u8, b: *mut u8) -> bool {
    (a as usize) < (b as usize)
}

#[inline]
fn ptr_add(ptr: *mut u8, bytes: usize) -> Option<*mut u8> {
    (ptr as usize)
        .checked_add(bytes)
        .map(|addr| addr as *mut u8)
}

#[inline]
fn ptr_sub(ptr: *mut u8, bytes: usize) -> Option<*mut u8> {
    (ptr as usize)
        .checked_sub(bytes)
        .map(|addr| addr as *mut u8)
}

/// Port of `ZSTD_cwksp_assert_internal_consistency`.
pub fn ZSTD_cwksp_assert_internal_consistency(ws: &ZSTD_cwksp) {
    debug_assert!(ptr_le(ws.workspace, ws.objectEnd));
    debug_assert!(ptr_le(ws.objectEnd, ws.tableEnd));
    debug_assert!(ptr_le(ws.objectEnd, ws.tableValidEnd));
    debug_assert!(ptr_le(ws.tableEnd, ws.allocStart));
    debug_assert!(ptr_le(ws.tableValidEnd, ws.allocStart));
    debug_assert!(ptr_le(ws.allocStart, ws.workspaceEnd));
    debug_assert!(ptr_le(ws.initOnceStart, ZSTD_cwksp_initialAllocStart(ws)));
    debug_assert!(ptr_le(ws.workspace, ws.initOnceStart));
}

/// Port of `ZSTD_cwksp_align`.
pub fn ZSTD_cwksp_align(size: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    let mask = align - 1;
    size.wrapping_add(mask) & !mask
}

/// Port of `ZSTD_cwksp_alloc_size`.
pub fn ZSTD_cwksp_alloc_size(size: usize) -> usize {
    size
}

/// Port of `ZSTD_cwksp_aligned_alloc_size`.
pub fn ZSTD_cwksp_aligned_alloc_size(size: usize, alignment: usize) -> usize {
    ZSTD_cwksp_alloc_size(ZSTD_cwksp_align(size, alignment))
}

/// Port of `ZSTD_cwksp_aligned64_alloc_size`.
pub fn ZSTD_cwksp_aligned64_alloc_size(size: usize) -> usize {
    ZSTD_cwksp_aligned_alloc_size(size, ZSTD_CWKSP_ALIGNMENT_BYTES)
}

/// Port of `ZSTD_cwksp_slack_space_required`.
pub fn ZSTD_cwksp_slack_space_required() -> usize {
    ZSTD_CWKSP_ALIGNMENT_BYTES * 2
}

/// Port of `ZSTD_cwksp_bytes_to_align_ptr`.
pub fn ZSTD_cwksp_bytes_to_align_ptr(ptr: *mut u8, alignBytes: usize) -> usize {
    debug_assert!(alignBytes.is_power_of_two());
    let mask = alignBytes - 1;
    let bytes = (alignBytes - ((ptr as usize) & mask)) & mask;
    debug_assert!(bytes < alignBytes);
    bytes
}

/// Port of `ZSTD_cwksp_initialAllocStart`.
pub fn ZSTD_cwksp_initialAllocStart(ws: &ZSTD_cwksp) -> *mut u8 {
    debug_assert!(ZSTD_CWKSP_ALIGNMENT_BYTES.is_power_of_two());
    let end = ws.workspaceEnd as usize;
    (end - (end % ZSTD_CWKSP_ALIGNMENT_BYTES)) as *mut u8
}

/// Port of `ZSTD_cwksp_reserve_internal_buffer_space`.
pub fn ZSTD_cwksp_reserve_internal_buffer_space(ws: &mut ZSTD_cwksp, bytes: usize) -> *mut u8 {
    let Some(alloc) = ptr_sub(ws.allocStart, bytes) else {
        ws.allocFailed = 1;
        return null_mut();
    };
    let bottom = ws.tableEnd;
    ZSTD_cwksp_assert_internal_consistency(ws);
    debug_assert!(ptr_le(bottom, alloc));
    if ptr_lt(alloc, bottom) {
        ws.allocFailed = 1;
        return null_mut();
    }
    if ptr_lt(alloc, ws.tableValidEnd) {
        ws.tableValidEnd = alloc;
    }
    ws.allocStart = alloc;
    alloc
}

/// Port of `ZSTD_cwksp_internal_advance_phase`.
pub fn ZSTD_cwksp_internal_advance_phase(
    ws: &mut ZSTD_cwksp,
    phase: ZSTD_cwksp_alloc_phase_e,
) -> usize {
    debug_assert!(phase >= ws.phase);
    if phase > ws.phase {
        if ws.phase < ZSTD_cwksp_alloc_phase_e::ZSTD_cwksp_alloc_aligned_init_once
            && phase >= ZSTD_cwksp_alloc_phase_e::ZSTD_cwksp_alloc_aligned_init_once
        {
            ws.tableValidEnd = ws.objectEnd;
            ws.initOnceStart = ZSTD_cwksp_initialAllocStart(ws);

            let alloc = ws.objectEnd;
            let bytesToAlign = ZSTD_cwksp_bytes_to_align_ptr(alloc, ZSTD_CWKSP_ALIGNMENT_BYTES);
            let Some(objectEnd) = ptr_add(alloc, bytesToAlign) else {
                return ERROR(ErrorCode::MemoryAllocation);
            };
            if ptr_lt(ws.workspaceEnd, objectEnd) {
                return ERROR(ErrorCode::MemoryAllocation);
            }
            ws.objectEnd = objectEnd;
            ws.tableEnd = objectEnd;
            if ptr_lt(ws.tableValidEnd, ws.tableEnd) {
                ws.tableValidEnd = ws.tableEnd;
            }
        }
        ws.phase = phase;
        ZSTD_cwksp_assert_internal_consistency(ws);
    }
    0
}

/// Port of `ZSTD_cwksp_owns_buffer`.
pub fn ZSTD_cwksp_owns_buffer(ws: &ZSTD_cwksp, ptr: *const u8) -> bool {
    !ptr.is_null()
        && (ws.workspace as usize) <= (ptr as usize)
        && (ptr as usize) < (ws.workspaceEnd as usize)
}

/// Port of `ZSTD_cwksp_reserve_internal`.
pub fn ZSTD_cwksp_reserve_internal(
    ws: &mut ZSTD_cwksp,
    bytes: usize,
    phase: ZSTD_cwksp_alloc_phase_e,
) -> *mut u8 {
    let rc = ZSTD_cwksp_internal_advance_phase(ws, phase);
    if ERR_isError(rc) || bytes == 0 {
        return null_mut();
    }
    ZSTD_cwksp_reserve_internal_buffer_space(ws, bytes)
}

/// Port of `ZSTD_cwksp_reserve_buffer`.
pub fn ZSTD_cwksp_reserve_buffer(ws: &mut ZSTD_cwksp, bytes: usize) -> *mut u8 {
    ZSTD_cwksp_reserve_internal(
        ws,
        bytes,
        ZSTD_cwksp_alloc_phase_e::ZSTD_cwksp_alloc_buffers,
    )
}

/// Port of `ZSTD_cwksp_reserve_aligned_init_once`.
pub fn ZSTD_cwksp_reserve_aligned_init_once(ws: &mut ZSTD_cwksp, bytes: usize) -> *mut u8 {
    let alignedBytes = ZSTD_cwksp_align(bytes, ZSTD_CWKSP_ALIGNMENT_BYTES);
    let ptr = ZSTD_cwksp_reserve_internal(
        ws,
        alignedBytes,
        ZSTD_cwksp_alloc_phase_e::ZSTD_cwksp_alloc_aligned_init_once,
    );
    debug_assert_eq!((ptr as usize) & (ZSTD_CWKSP_ALIGNMENT_BYTES - 1), 0);
    if !ptr.is_null() && ptr_lt(ptr, ws.initOnceStart) {
        let zero_len = ((ws.initOnceStart as usize) - (ptr as usize)).min(alignedBytes);
        unsafe { core::ptr::write_bytes(ptr, 0, zero_len) };
        ws.initOnceStart = ptr;
    }
    ptr
}

/// Port of `ZSTD_cwksp_reserve_aligned64`.
pub fn ZSTD_cwksp_reserve_aligned64(ws: &mut ZSTD_cwksp, bytes: usize) -> *mut u8 {
    let alignedBytes = ZSTD_cwksp_align(bytes, ZSTD_CWKSP_ALIGNMENT_BYTES);
    let ptr = ZSTD_cwksp_reserve_internal(
        ws,
        alignedBytes,
        ZSTD_cwksp_alloc_phase_e::ZSTD_cwksp_alloc_aligned,
    );
    debug_assert_eq!((ptr as usize) & (ZSTD_CWKSP_ALIGNMENT_BYTES - 1), 0);
    ptr
}

/// Port of `ZSTD_cwksp_reserve_table`.
pub fn ZSTD_cwksp_reserve_table(ws: &mut ZSTD_cwksp, bytes: usize) -> *mut u8 {
    let phase = ZSTD_cwksp_alloc_phase_e::ZSTD_cwksp_alloc_aligned_init_once;
    if ws.phase < phase {
        let rc = ZSTD_cwksp_internal_advance_phase(ws, phase);
        if ERR_isError(rc) {
            return null_mut();
        }
    }
    let alloc = ws.tableEnd;
    let Some(end) = ptr_add(alloc, bytes) else {
        ws.allocFailed = 1;
        return null_mut();
    };
    let top = ws.allocStart;
    debug_assert_eq!(bytes & (core::mem::size_of::<u32>() - 1), 0);
    ZSTD_cwksp_assert_internal_consistency(ws);
    debug_assert!(ptr_le(end, top));
    if ptr_lt(top, end) {
        ws.allocFailed = 1;
        return null_mut();
    }
    ws.tableEnd = end;
    debug_assert_eq!(bytes & (ZSTD_CWKSP_ALIGNMENT_BYTES - 1), 0);
    debug_assert_eq!((alloc as usize) & (ZSTD_CWKSP_ALIGNMENT_BYTES - 1), 0);
    alloc
}

/// Port of `ZSTD_cwksp_reserve_object`.
pub fn ZSTD_cwksp_reserve_object(ws: &mut ZSTD_cwksp, bytes: usize) -> *mut u8 {
    let roundedBytes = ZSTD_cwksp_align(bytes, core::mem::size_of::<usize>());
    let alloc = ws.objectEnd;
    let end = ptr_add(alloc, roundedBytes);
    debug_assert_eq!((alloc as usize) % core::mem::align_of::<usize>(), 0);
    debug_assert_eq!(bytes % core::mem::align_of::<usize>(), 0);
    ZSTD_cwksp_assert_internal_consistency(ws);
    if ws.phase != ZSTD_cwksp_alloc_phase_e::ZSTD_cwksp_alloc_objects
        || end.is_none()
        || ptr_lt(ws.workspaceEnd, end.unwrap())
    {
        ws.allocFailed = 1;
        return null_mut();
    }
    let end = end.unwrap();
    ws.objectEnd = end;
    ws.tableEnd = end;
    ws.tableValidEnd = end;
    alloc
}

/// Port of `ZSTD_cwksp_reserve_object_aligned`.
pub fn ZSTD_cwksp_reserve_object_aligned(
    ws: &mut ZSTD_cwksp,
    byteSize: usize,
    alignment: usize,
) -> *mut u8 {
    let surplus = if alignment > core::mem::size_of::<usize>() {
        alignment - core::mem::size_of::<usize>()
    } else {
        0
    };
    let reservationSize = byteSize.wrapping_add(surplus);
    let start = ZSTD_cwksp_reserve_object(ws, reservationSize);
    if start.is_null() || surplus == 0 {
        return start;
    }
    debug_assert!(alignment.is_power_of_two());
    let mask = alignment - 1;
    ((start as usize).wrapping_add(surplus) & !mask) as *mut u8
}

/// Port of `ZSTD_cwksp_mark_tables_dirty`.
pub fn ZSTD_cwksp_mark_tables_dirty(ws: &mut ZSTD_cwksp) {
    debug_assert!(ptr_le(ws.objectEnd, ws.tableValidEnd));
    debug_assert!(ptr_le(ws.tableValidEnd, ws.allocStart));
    ws.tableValidEnd = ws.objectEnd;
    ZSTD_cwksp_assert_internal_consistency(ws);
}

/// Port of `ZSTD_cwksp_mark_tables_clean`.
pub fn ZSTD_cwksp_mark_tables_clean(ws: &mut ZSTD_cwksp) {
    debug_assert!(ptr_le(ws.objectEnd, ws.tableValidEnd));
    debug_assert!(ptr_le(ws.tableValidEnd, ws.allocStart));
    if ptr_lt(ws.tableValidEnd, ws.tableEnd) {
        ws.tableValidEnd = ws.tableEnd;
    }
    ZSTD_cwksp_assert_internal_consistency(ws);
}

/// Port of `ZSTD_cwksp_clean_tables`.
pub fn ZSTD_cwksp_clean_tables(ws: &mut ZSTD_cwksp) {
    debug_assert!(ptr_le(ws.objectEnd, ws.tableValidEnd));
    debug_assert!(ptr_le(ws.tableValidEnd, ws.allocStart));
    if ptr_lt(ws.tableValidEnd, ws.tableEnd) {
        let len = (ws.tableEnd as usize) - (ws.tableValidEnd as usize);
        unsafe { core::ptr::write_bytes(ws.tableValidEnd, 0, len) };
    }
    ZSTD_cwksp_mark_tables_clean(ws);
}

/// Port of `ZSTD_cwksp_clear_tables`.
pub fn ZSTD_cwksp_clear_tables(ws: &mut ZSTD_cwksp) {
    ws.tableEnd = ws.objectEnd;
    ZSTD_cwksp_assert_internal_consistency(ws);
}

/// Port of `ZSTD_cwksp_clear`.
pub fn ZSTD_cwksp_clear(ws: &mut ZSTD_cwksp) {
    ws.tableEnd = ws.objectEnd;
    ws.allocStart = ZSTD_cwksp_initialAllocStart(ws);
    ws.allocFailed = 0;
    if ws.phase > ZSTD_cwksp_alloc_phase_e::ZSTD_cwksp_alloc_aligned_init_once {
        ws.phase = ZSTD_cwksp_alloc_phase_e::ZSTD_cwksp_alloc_aligned_init_once;
    }
    ZSTD_cwksp_assert_internal_consistency(ws);
}

/// Port of `ZSTD_cwksp_sizeof`.
pub fn ZSTD_cwksp_sizeof(ws: &ZSTD_cwksp) -> usize {
    (ws.workspaceEnd as usize).wrapping_sub(ws.workspace as usize)
}

/// Port of `ZSTD_cwksp_used`.
pub fn ZSTD_cwksp_used(ws: &ZSTD_cwksp) -> usize {
    (ws.tableEnd as usize)
        .wrapping_sub(ws.workspace as usize)
        .wrapping_add((ws.workspaceEnd as usize).wrapping_sub(ws.allocStart as usize))
}

/// Port of `ZSTD_cwksp_init`.
pub fn ZSTD_cwksp_init(
    ws: &mut ZSTD_cwksp,
    start: *mut u8,
    size: usize,
    isStatic: ZSTD_cwksp_static_alloc_e,
) {
    debug_assert_eq!((start as usize) & (core::mem::size_of::<usize>() - 1), 0);
    ws.owned = None;
    ws.workspace = start;
    ws.workspaceEnd = start.wrapping_add(size);
    ws.objectEnd = ws.workspace;
    ws.tableValidEnd = ws.objectEnd;
    ws.initOnceStart = ZSTD_cwksp_initialAllocStart(ws);
    ws.phase = ZSTD_cwksp_alloc_phase_e::ZSTD_cwksp_alloc_objects;
    ws.isStatic = isStatic;
    ZSTD_cwksp_clear(ws);
    ws.workspaceOversizedDuration = 0;
    ZSTD_cwksp_assert_internal_consistency(ws);
}

/// Port of `ZSTD_cwksp_create`.
pub fn ZSTD_cwksp_create(ws: &mut ZSTD_cwksp, size: usize) -> usize {
    let word_size = core::mem::size_of::<usize>();
    let words = size / word_size + usize::from(size % word_size != 0);
    let mut workspace = Vec::<usize>::new();
    if workspace.try_reserve_exact(words).is_err() {
        return ERROR(ErrorCode::MemoryAllocation);
    }
    workspace.resize(words, 0);
    let mut workspace = workspace.into_boxed_slice();
    let ptr = workspace.as_mut_ptr().cast::<u8>();
    ZSTD_cwksp_init(
        ws,
        ptr,
        size,
        ZSTD_cwksp_static_alloc_e::ZSTD_cwksp_dynamic_alloc,
    );
    ws.owned = Some(workspace);
    0
}

/// Port of `ZSTD_cwksp_free`.
pub fn ZSTD_cwksp_free(ws: &mut ZSTD_cwksp) {
    *ws = ZSTD_cwksp::default();
}

/// Port of `ZSTD_cwksp_move`.
pub fn ZSTD_cwksp_move(dst: &mut ZSTD_cwksp, src: &mut ZSTD_cwksp) {
    *dst = core::mem::take(src);
}

/// Port of `ZSTD_cwksp_reserve_failed`.
pub fn ZSTD_cwksp_reserve_failed(ws: &ZSTD_cwksp) -> bool {
    ws.allocFailed != 0
}

/// Port of `ZSTD_cwksp_estimated_space_within_bounds`.
pub fn ZSTD_cwksp_estimated_space_within_bounds(ws: &ZSTD_cwksp, estimatedSpace: usize) -> bool {
    estimatedSpace.wrapping_sub(ZSTD_cwksp_slack_space_required()) <= ZSTD_cwksp_used(ws)
        && ZSTD_cwksp_used(ws) <= estimatedSpace
}

/// Port of `ZSTD_cwksp_available_space`.
pub fn ZSTD_cwksp_available_space(ws: &ZSTD_cwksp) -> usize {
    (ws.allocStart as usize).wrapping_sub(ws.tableEnd as usize)
}

/// Port of `ZSTD_cwksp_check_available`.
pub fn ZSTD_cwksp_check_available(ws: &ZSTD_cwksp, additionalNeededSpace: usize) -> bool {
    ZSTD_cwksp_available_space(ws) >= additionalNeededSpace
}

/// Port of `ZSTD_cwksp_check_too_large`.
pub fn ZSTD_cwksp_check_too_large(ws: &ZSTD_cwksp, additionalNeededSpace: usize) -> bool {
    ZSTD_cwksp_check_available(
        ws,
        additionalNeededSpace
            .wrapping_mul(crate::common::zstd_internal::ZSTD_WORKSPACETOOLARGE_FACTOR),
    )
}

/// Port of `ZSTD_cwksp_check_wasteful`.
pub fn ZSTD_cwksp_check_wasteful(ws: &ZSTD_cwksp, additionalNeededSpace: usize) -> bool {
    ZSTD_cwksp_check_too_large(ws, additionalNeededSpace)
        && ws.workspaceOversizedDuration
            > crate::common::zstd_internal::ZSTD_WORKSPACETOOLARGE_MAXDURATION as i32
}

/// Port of `ZSTD_cwksp_bump_oversized_duration`.
pub fn ZSTD_cwksp_bump_oversized_duration(ws: &mut ZSTD_cwksp, additionalNeededSpace: usize) {
    if ZSTD_cwksp_check_too_large(ws, additionalNeededSpace) {
        ws.workspaceOversizedDuration += 1;
    } else {
        ws.workspaceOversizedDuration = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_object_table_and_buffer_maintain_workspace_order() {
        let mut ws = ZSTD_cwksp::default();
        assert_eq!(ZSTD_cwksp_create(&mut ws, 1024), 0);

        let obj = ZSTD_cwksp_reserve_object(&mut ws, 16);
        assert!(!obj.is_null());
        let table = ZSTD_cwksp_reserve_table(&mut ws, 64);
        assert!(!table.is_null());
        let buf = ZSTD_cwksp_reserve_buffer(&mut ws, 32);
        assert!(!buf.is_null());
        assert!(ZSTD_cwksp_owns_buffer(&ws, obj));
        assert!(ZSTD_cwksp_owns_buffer(&ws, table));
        assert!(ZSTD_cwksp_owns_buffer(&ws, buf));
    }

    #[test]
    fn mark_and_clean_tables_updates_valid_range() {
        let mut ws = ZSTD_cwksp::default();
        assert_eq!(ZSTD_cwksp_create(&mut ws, 1024), 0);
        let _ = ZSTD_cwksp_reserve_object(&mut ws, 16);
        let _ = ZSTD_cwksp_reserve_table(&mut ws, 64);
        ZSTD_cwksp_mark_tables_dirty(&mut ws);
        assert_eq!(ws.tableValidEnd, ws.objectEnd);
        ZSTD_cwksp_clean_tables(&mut ws);
        assert_eq!(ws.tableValidEnd, ws.tableEnd);
    }

    #[test]
    fn aligned_alloc_size_helpers_match_upstream_rounding() {
        assert_eq!(ZSTD_cwksp_aligned_alloc_size(0, 8), 0);
        assert_eq!(ZSTD_cwksp_aligned_alloc_size(1, 8), 8);
        assert_eq!(ZSTD_cwksp_aligned64_alloc_size(65), 128);
    }

    #[test]
    fn create_reports_allocation_failure() {
        let mut ws = ZSTD_cwksp::default();
        let rc = ZSTD_cwksp_create(&mut ws, usize::MAX);
        assert!(ERR_isError(rc));
        assert!(ws.owned.is_none());
    }

    #[test]
    fn create_workspace_has_malloc_like_pointer_alignment() {
        let mut ws = ZSTD_cwksp::default();
        assert_eq!(ZSTD_cwksp_create(&mut ws, 128), 0);
        assert_eq!(
            (ws.workspace as usize) & (core::mem::align_of::<usize>() - 1),
            0
        );
        assert_eq!(ZSTD_cwksp_sizeof(&ws), 128);
    }

    #[test]
    fn init_replaces_previous_owned_workspace_with_external_buffer() {
        let mut ws = ZSTD_cwksp::default();
        assert_eq!(ZSTD_cwksp_create(&mut ws, 128), 0);
        assert!(ws.owned.is_some());

        let mut external = [0usize; 32];
        let ptr = external.as_mut_ptr().cast::<u8>();
        ZSTD_cwksp_init(
            &mut ws,
            ptr,
            core::mem::size_of_val(&external),
            ZSTD_cwksp_static_alloc_e::ZSTD_cwksp_static_alloc,
        );

        assert!(ws.owned.is_none());
        assert_eq!(ws.workspace, ptr);
        assert_eq!(ZSTD_cwksp_sizeof(&ws), core::mem::size_of_val(&external));
        assert_eq!(
            ws.isStatic,
            ZSTD_cwksp_static_alloc_e::ZSTD_cwksp_static_alloc
        );
    }

    #[test]
    fn workspace_accounting_uses_size_t_wrapping() {
        let ws = ZSTD_cwksp {
            workspace: 0x1000usize as *mut u8,
            workspaceEnd: 0x0ff0usize as *mut u8,
            tableEnd: 0x0ff8usize as *mut u8,
            allocStart: 0x1008usize as *mut u8,
            ..ZSTD_cwksp::default()
        };

        assert_eq!(ZSTD_cwksp_sizeof(&ws), 0x0ff0usize.wrapping_sub(0x1000));
        assert_eq!(
            ZSTD_cwksp_used(&ws),
            0x0ff8usize
                .wrapping_sub(0x1000)
                .wrapping_add(0x0ff0usize.wrapping_sub(0x1008))
        );
        assert_eq!(
            ZSTD_cwksp_available_space(&ws),
            0x1008usize.wrapping_sub(0x0ff8)
        );
    }

    #[test]
    fn oversized_object_reservation_fails_without_advancing_bounds() {
        let mut ws = ZSTD_cwksp::default();
        assert_eq!(ZSTD_cwksp_create(&mut ws, 128), 0);

        let object_end = ws.objectEnd;
        let huge_aligned_object = usize::MAX & !(core::mem::align_of::<usize>() - 1);
        assert!(ZSTD_cwksp_reserve_object(&mut ws, huge_aligned_object).is_null());
        assert_eq!(ws.objectEnd, object_end);
        assert!(ZSTD_cwksp_reserve_failed(&ws));
    }

    #[test]
    fn overflowing_aligned_reservations_follow_zero_sized_path() {
        let mut ws = ZSTD_cwksp::default();
        assert_eq!(ZSTD_cwksp_create(&mut ws, 128), 0);
        let alloc_start = ws.allocStart;

        assert!(ZSTD_cwksp_reserve_aligned64(&mut ws, usize::MAX).is_null());
        assert_eq!(ws.allocStart, alloc_start);
        assert_eq!(ws.phase, ZSTD_cwksp_alloc_phase_e::ZSTD_cwksp_alloc_aligned);
        assert!(!ZSTD_cwksp_reserve_failed(&ws));
    }

    #[test]
    fn overflowing_object_aligned_size_wraps_before_reserving() {
        let mut ws = ZSTD_cwksp::default();
        assert_eq!(ZSTD_cwksp_create(&mut ws, 128), 0);
        let object_end = ws.objectEnd;

        let ptr = ZSTD_cwksp_reserve_object_aligned(&mut ws, usize::MAX - 47, 64);
        assert!(!ptr.is_null());
        assert_eq!((ptr as usize) & (ZSTD_CWKSP_ALIGNMENT_BYTES - 1), 0);
        assert_eq!(ws.objectEnd as usize, object_end as usize + 8);
        assert!(!ZSTD_cwksp_reserve_failed(&ws));
    }
}
