#![no_std]

/// Stuff to do in here :
/// This module is meant to compute and return info about the caching structure
/// Should include if needed the work for reverse engineering L3 complex addressing
/// May also have a module to deal with prefetchers
extern crate alloc;

use alloc::vec::Vec;
use core::arch::x86_64 as arch_x86;
use polling_serial::serial_println;
use vga_buffer::println;

pub fn test() {
    let cr = unsafe { arch_x86::__cpuid_count(0x04, 0) };
    serial_println!(
        "EAX {:x}, EBX {:x}, ECX {:x}, EDX {:x}",
        cr.eax,
        cr.ebx,
        cr.ecx,
        cr.edx
    );
    println!(
        "EAX {:x}, EBX {:x}, ECX {:x}, EDX {:x}",
        cr.eax, cr.ebx, cr.ecx, cr.edx
    );
    let cache_type = cr.eax & 0x1f;
    let cache_level = cr.eax >> 5 & 0x7;
    println!("type {}, level {}", cache_type, cache_level);
}

pub fn get_cache_info() -> Vec<CacheInfo> {
    let mut ret = Vec::new();
    let mut i = 0;

    while let Some(cache_info) =
        CacheInfo::fromCpuidResult(&unsafe { arch_x86::__cpuid_count(0x04, i) })
    {
        ret.push(cache_info);
        i += 1;
    }
    ret
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
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
    //self_init: bool,
    //fully_assoc: bool,
    //core_for_cache: u16,
    //core_in_package: u16,
    //cache_line_size: u16,
    //physical_line_partition: u16,
    //associativity: u16,
    //sets: u32,
    //wbinvd_no_guarantee: bool,
    //inclusive: bool,
    //complex_cache_indexing: bool,
}

impl CacheInfo {
    pub fn fromCpuidResult(cr: &arch_x86::CpuidResult) -> Option<CacheInfo> {
        let ctype = cr.eax & 0x1f;
        let cache_type = match ctype {
            0 => {
                return None;
            }
            1 => CacheType::Data,
            2 => CacheType::Instruction,
            3 => CacheType::Unified,
            _ => {
                return None;
            }
        };
        let level: u8 = (cr.eax >> 5 & 0x7) as u8;
        Some(CacheInfo { cache_type, level })
    }
}
