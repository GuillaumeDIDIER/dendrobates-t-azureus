use alloc::alloc::{GlobalAlloc, Layout};
use core::ptr::null_mut;
use polling_serial::serial_println;
use x86_64::instructions::bochs_breakpoint;
use x86_64::structures::paging::mapper::MapToError::PageAlreadyMapped;
use x86_64::{
    structures::paging::{
        mapper::MapToError, FrameAllocator, Mapper, Page, PageTableFlags, Size4KiB,
    },
    VirtAddr,
};

pub const HEAP_START: usize = 0x4444_4444_0000;
pub const HEAP_SIZE: usize = 0x1 << 20; // 1MB Heap

pub struct Dummy;

unsafe impl GlobalAlloc for Dummy {
    unsafe fn alloc(&self, _layout: Layout) -> *mut u8 {
        null_mut()
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        panic!("dealloc should never be called");
    }
}

pub fn init_heap(
    mapper: &mut impl Mapper<Size4KiB>,
    frame_allocator: &mut impl FrameAllocator<Size4KiB>,
) -> Result<(), MapToError> {
    let page_range = {
        let heap_start = VirtAddr::new(HEAP_START as u64);
        let heap_end = heap_start + HEAP_SIZE as u64 - 1u64;
        let heap_start_page = Page::containing_address(heap_start);
        let heap_end_page = Page::containing_address(heap_end);
        Page::range_inclusive(heap_start_page, heap_end_page)
    };

    let mut i = 0;
    for page in page_range {
        let frame = frame_allocator
            .allocate_frame()
            .ok_or(MapToError::FrameAllocationFailed)?;
        let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE;
        unsafe { mapper.map_to(page, frame, flags, frame_allocator)?.flush() };
        if i == 0 {
            serial_println!("Mapped {:?} at {:?}", page, frame);
        }
        i = i + 1;
    }

    unsafe {
        super::ALLOCATOR.lock().init(HEAP_START, HEAP_SIZE);
    }

    Ok(())
}
