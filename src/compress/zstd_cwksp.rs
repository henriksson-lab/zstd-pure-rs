//! Translation of `lib/compress/zstd_cwksp.h`.

#![allow(non_snake_case)]

use core::ptr::null_mut;

use crate::common::error::{ErrorCode, ERR_isError, ERROR};

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
    pub owned: Option<Box<[u8]>>,
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

#[inline]
fn ptr_le(a: *mut u8, b: *mut u8) -> bool {
    (a as usize) <= (b as usize)
}

#[inline]
fn ptr_lt(a: *mut u8, b: *mut u8) -> bool {
    (a as usize) < (b as usize)
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
    (size + mask) & !mask
}

/// Port of `ZSTD_cwksp_alloc_size`.
pub fn ZSTD_cwksp_alloc_size(size: usize) -> usize {
    size
}

/// Port of `ZSTD_cwksp_slack_space_required`.
pub fn ZSTD_cwksp_slack_space_required() -> usize {
    ZSTD_CWKSP_ALIGNMENT_BYTES * 2
}

/// Port of `ZSTD_cwksp_bytes_to_align_ptr`.
pub fn ZSTD_cwksp_bytes_to_align_ptr(ptr: *mut u8, alignBytes: usize) -> usize {
    debug_assert!(alignBytes.is_power_of_two());
    let mask = alignBytes - 1;
    (alignBytes - ((ptr as usize) & mask)) & mask
}

/// Port of `ZSTD_cwksp_initialAllocStart`.
pub fn ZSTD_cwksp_initialAllocStart(ws: &ZSTD_cwksp) -> *mut u8 {
    let end = ws.workspaceEnd as usize;
    (end - (end % ZSTD_CWKSP_ALIGNMENT_BYTES)) as *mut u8
}

/// Port of `ZSTD_cwksp_reserve_internal_buffer_space`.
pub fn ZSTD_cwksp_reserve_internal_buffer_space(
    ws: &mut ZSTD_cwksp,
    bytes: usize,
) -> *mut u8 {
    ZSTD_cwksp_assert_internal_consistency(ws);
    if bytes > ZSTD_cwksp_available_space(ws) {
        ws.allocFailed = 1;
        return null_mut();
    }
    let alloc = unsafe { ws.allocStart.sub(bytes) };
    if ptr_lt(alloc, ws.tableEnd) {
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
            let bytesToAlign =
                ZSTD_cwksp_bytes_to_align_ptr(alloc, ZSTD_CWKSP_ALIGNMENT_BYTES);
            let objectEnd = unsafe { alloc.add(bytesToAlign) };
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
    if !ptr.is_null() && ptr_lt(ptr, ws.initOnceStart) {
        let zero_len = ((ws.initOnceStart as usize) - (ptr as usize)).min(alignedBytes);
        unsafe { core::ptr::write_bytes(ptr, 0, zero_len) };
        ws.initOnceStart = ptr;
    }
    ptr
}

/// Port of `ZSTD_cwksp_reserve_aligned64`.
pub fn ZSTD_cwksp_reserve_aligned64(ws: &mut ZSTD_cwksp, bytes: usize) -> *mut u8 {
    ZSTD_cwksp_reserve_internal(
        ws,
        ZSTD_cwksp_align(bytes, ZSTD_CWKSP_ALIGNMENT_BYTES),
        ZSTD_cwksp_alloc_phase_e::ZSTD_cwksp_alloc_aligned,
    )
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
    let end = unsafe { alloc.add(bytes) };
    if ptr_lt(ws.allocStart, end) {
        ws.allocFailed = 1;
        return null_mut();
    }
    ws.tableEnd = end;
    alloc
}

/// Port of `ZSTD_cwksp_reserve_object`.
pub fn ZSTD_cwksp_reserve_object(ws: &mut ZSTD_cwksp, bytes: usize) -> *mut u8 {
    let roundedBytes = ZSTD_cwksp_align(bytes, core::mem::size_of::<usize>());
    let alloc = ws.objectEnd;
    let end = unsafe { alloc.add(roundedBytes) };
    if ws.phase != ZSTD_cwksp_alloc_phase_e::ZSTD_cwksp_alloc_objects
        || ptr_lt(ws.workspaceEnd, end)
    {
        ws.allocFailed = 1;
        return null_mut();
    }
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
    let start = ZSTD_cwksp_reserve_object(ws, byteSize + surplus);
    if start.is_null() || surplus == 0 {
        return start;
    }
    debug_assert!(alignment.is_power_of_two());
    let mask = alignment - 1;
    (((start as usize) + surplus) & !mask) as *mut u8
}

/// Port of `ZSTD_cwksp_mark_tables_dirty`.
pub fn ZSTD_cwksp_mark_tables_dirty(ws: &mut ZSTD_cwksp) {
    ws.tableValidEnd = ws.objectEnd;
    ZSTD_cwksp_assert_internal_consistency(ws);
}

/// Port of `ZSTD_cwksp_mark_tables_clean`.
pub fn ZSTD_cwksp_mark_tables_clean(ws: &mut ZSTD_cwksp) {
    if ptr_lt(ws.tableValidEnd, ws.tableEnd) {
        ws.tableValidEnd = ws.tableEnd;
    }
    ZSTD_cwksp_assert_internal_consistency(ws);
}

/// Port of `ZSTD_cwksp_clean_tables`.
pub fn ZSTD_cwksp_clean_tables(ws: &mut ZSTD_cwksp) {
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
    (ws.workspaceEnd as usize).saturating_sub(ws.workspace as usize)
}

/// Port of `ZSTD_cwksp_used`.
pub fn ZSTD_cwksp_used(ws: &ZSTD_cwksp) -> usize {
    (ws.tableEnd as usize).saturating_sub(ws.workspace as usize)
        + (ws.workspaceEnd as usize).saturating_sub(ws.allocStart as usize)
}

/// Port of `ZSTD_cwksp_init`.
pub fn ZSTD_cwksp_init(
    ws: &mut ZSTD_cwksp,
    start: *mut u8,
    size: usize,
    isStatic: ZSTD_cwksp_static_alloc_e,
) {
    ws.workspace = start;
    ws.workspaceEnd = unsafe { start.add(size) };
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
    let mut workspace = vec![0u8; size].into_boxed_slice();
    let ptr = workspace.as_mut_ptr();
    ws.owned = Some(workspace);
    ZSTD_cwksp_init(ws, ptr, size, ZSTD_cwksp_static_alloc_e::ZSTD_cwksp_dynamic_alloc);
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
pub fn ZSTD_cwksp_estimated_space_within_bounds(
    ws: &ZSTD_cwksp,
    estimatedSpace: usize,
) -> bool {
    estimatedSpace.saturating_sub(ZSTD_cwksp_slack_space_required()) <= ZSTD_cwksp_used(ws)
        && ZSTD_cwksp_used(ws) <= estimatedSpace
}

/// Port of `ZSTD_cwksp_available_space`.
pub fn ZSTD_cwksp_available_space(ws: &ZSTD_cwksp) -> usize {
    (ws.allocStart as usize).saturating_sub(ws.tableEnd as usize)
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
            * crate::common::zstd_internal::ZSTD_WORKSPACETOOLARGE_FACTOR,
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
}
