#![cfg_attr(feature = "no_std", no_std)]
#![feature(linked_list_cursors)]
#![allow(clippy::missing_safety_doc)]
#![deny(unsafe_op_in_unsafe_fn)]
extern crate alloc;

use core::arch::x86_64 as arch_x86;
use core::ptr;

use cpuid::CPUVendor;
use static_assertions::assert_cfg;

assert_cfg!(
    all(
        not(all(feature = "use_std", feature = "no_std")),
        any(feature = "use_std", feature = "no_std")
    ),
    "Choose std or no-std but not both"
);

pub mod cache_info;
pub mod calibration;
#[cfg(feature = "use_std")]
pub mod mmap;
pub mod prefetcher;

pub mod frequency;

#[cfg(feature = "use_std")]
pub mod calibrate_2t;

#[cfg(feature = "use_std")]
pub mod ip_tool;
pub mod numa_analysis;

// rdtsc no fence
pub unsafe fn rdtsc_nofence() -> u64 {
    unsafe { arch_x86::_rdtsc() }
}
// rdtsc (has mfence before and after)
pub unsafe fn rdtsc_fence() -> u64 {
    unsafe { arch_x86::_mm_mfence() };
    let tsc: u64 = unsafe { arch_x86::_rdtsc() };
    unsafe { arch_x86::_mm_mfence() };
    tsc
}

pub unsafe fn rdpru_fenced() -> u64 {
    let [hi, lo]: [u32; 2];
    unsafe { arch_x86::_mm_mfence() };
    unsafe {
        core::arch::asm!(
        "rdpru",
        out("edx") hi,
        out("eax") lo,
        in("ecx") 1u32,
        options(nostack, nomem, preserves_flags),
        )
    };
    let ret = (u64::from(hi) << 32) | u64::from(lo);
    unsafe { arch_x86::_mm_mfence() };
    ret
}

pub fn has_rdpru() -> bool {
    if CPUVendor::get_cpu_vendor() == CPUVendor::AMD {
        // The RDPRU instruction is supported if the feature flag CPUID Fn8000_0008 EBX[4]=1. The 16-bit
        // field in CPUID Fn8000_0008-EDX[31:16] returns the largest ECX value that returns a valid register.
        let cpuid = unsafe { arch_x86::__cpuid(0x8000_0008) };
        cpuid.ebx & (0x1 << 4) != 0
    } else {
        false
    }
}

pub unsafe fn maccess<T>(p: *const T) {
    unsafe { ptr::read_volatile(p) };
}

pub unsafe fn maccess_fenced<T>(p: *const T) {
    unsafe { arch_x86::_mm_mfence() };
    unsafe { ptr::read_volatile(p) };
    unsafe { arch_x86::_mm_mfence() };
}

// flush (cflush)
pub unsafe fn flush(p: *const u8) {
    unsafe { arch_x86::_mm_clflush(p) };
}

pub unsafe fn flush_fenced(p: *const u8) {
    unsafe { arch_x86::_mm_mfence() };
    unsafe { arch_x86::_mm_clflush(p) };
    unsafe { arch_x86::_mm_mfence() };
}

pub fn noop<T>(_: *const T) {}

#[cfg(feature = "use_std")]
pub fn find_core_per_socket() -> u8 {
    // FIXME error handling
    use std::process::Command;
    use std::str::from_utf8;

    let core_per_socket_out = Command::new("sh")
        .arg("-c")
        .arg("lscpu | grep socket | cut -b 22-")
        .output()
        .expect("Failed to detect cpu count");
    //println!("{:#?}", core_per_socket_str);

    let core_per_socket_str = from_utf8(&core_per_socket_out.stdout).unwrap();

    //println!("Number of cores per socket: {}", cps_str);

    let core_per_socket: u8 = core_per_socket_str[0..(core_per_socket_str.len() - 1)]
        .trim()
        .parse()
        .unwrap_or(0);
    core_per_socket
}

// future enhancements
// prefetch
// long nop (64 nops)
