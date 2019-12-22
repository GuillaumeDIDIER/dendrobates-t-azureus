#![no_std]

/// Stuff to do in here :
/// This module is meant to compute and return info about the caching structure
/// Should include if needed the work for reverse engineering L3 complex addressing
/// May also have a module to deal with prefetchers
extern crate alloc;

use alloc::boxed::Box;
use core::arch::x86_64 as arch_x86;
use polling_serial::serial_println;

pub fn test() {
    let x = Box::new(41);
    let cr = unsafe { arch_x86::__cpuid_count(0x04, 0) };
    serial_println!("{:?}", cr);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheType {
    Null = 0,
    Data = 1,
    Instruction = 2,
    Unified = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheInfo {
    cache_type: CacheType,
    level: u8,
    self_init: bool,
    fully_assoc: bool,
    core_for_cache: u16,
    core_in_package: u16,
    cache_line_size: u16,
    physical_line_partition: u16,
    associativity: u16,
    sets: u32,
    wbinvd_no_guarantee: bool,
    inclusive: bool,
    complex_cache_indexing: bool,
}
