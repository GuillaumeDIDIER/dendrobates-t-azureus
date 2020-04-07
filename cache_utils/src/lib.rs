#![cfg_attr(feature = "no_std", no_std)]
#![feature(ptr_internals)]

use static_assertions::assert_cfg;

assert_cfg!(
    all(
        not(all(feature = "std", feature = "no_std")),
        any(feature = "std", feature = "no_std")
    ),
    "Choose std or no-std but not both"
);

pub mod cache_info;
pub mod calibration;
pub mod complex_addressing;
#[cfg(feature = "std")]
pub mod mmap;
pub mod prefetcher;

use core::arch::x86_64 as arch_x86;
use core::ptr;

// rdtsc no fence
pub unsafe fn rdtsc_nofence() -> u64 {
    arch_x86::_rdtsc()
}
// rdtsc (has mfence before and after)
pub unsafe fn rdtsc_fence() -> u64 {
    arch_x86::_mm_mfence();
    let tsc: u64 = arch_x86::_rdtsc();
    arch_x86::_mm_mfence();
    tsc
}

pub unsafe fn maccess<T>(p: *const T) {
    ptr::read_volatile(p);
}

// flush (cflush)
pub unsafe fn flush(p: *const u8) {
    arch_x86::_mm_clflush(p);
}

// future enhancements
// prefetch
// long nop (64 nops)
