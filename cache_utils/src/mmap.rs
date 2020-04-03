#![cfg(feature = "std")]

use core::ptr::null_mut;
use core::slice::{from_raw_parts, from_raw_parts_mut};
use nix::sys::mman;

/* from linux kernel headers.
#define HUGETLB_FLAG_ENCODE_SHIFT       26
#define HUGETLB_FLAG_ENCODE_MASK        0x3f

#define HUGETLB_FLAG_ENCODE_64KB        (16 << HUGETLB_FLAG_ENCODE_SHIFT)
#define HUGETLB_FLAG_ENCODE_512KB       (19 << HUGETLB_FLAG_ENCODE_SHIFT)
#define HUGETLB_FLAG_ENCODE_1MB         (20 << HUGETLB_FLAG_ENCODE_SHIFT)
#define HUGETLB_FLAG_ENCODE_2MB         (21 << HUGETLB_FLAG_ENCODE_SHIFT)
*/

pub struct MMappedMemory {
    pointer: *mut u8,
    size: usize,
}

impl MMappedMemory {
    pub unsafe fn new(size: usize) -> MMappedMemory {
        let p: *mut u8 = mman::mmap(
            null_mut(),
            size,
            mman::ProtFlags::PROT_READ | mman::ProtFlags::PROT_WRITE,
            mman::MapFlags::MAP_PRIVATE
                | mman::MapFlags::MAP_ANONYMOUS
                | mman::MapFlags::MAP_HUGETLB,
            -1,
            0,
        )
        .unwrap() as *mut u8;
        MMappedMemory { pointer: p, size }
    }

    pub fn slice(&self) -> &[u8] {
        unsafe { from_raw_parts(self.pointer, self.size) }
    }

    pub fn slice_mut(&self) -> &mut [u8] {
        unsafe { from_raw_parts_mut(self.pointer, self.size) }
    }
}
