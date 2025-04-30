#![deny(unsafe_op_in_unsafe_fn)]

use cache_utils::calibration::{
    calibrate_fixed_freq_2_thread_numa, only_reload, CalibrateOperation2T, CalibrationOptions,
    Verbosity, CLFLUSH_BUCKET_NUMBER, CLFLUSH_BUCKET_SIZE, CLFLUSH_NUM_ITER,
};
use cache_utils::mmap::MMappedMemory;
use cache_utils::{flush, maccess, noop};
use nix::sched::{sched_getaffinity, CpuSet};
use nix::unistd::Pid;

use core::arch::x86_64 as arch_x86;

use cache_utils::ip_tool::Function;
use calibration_results::calibration_2t::CalibrateResult2TNuma;
use core::cmp::min;
use numa_utils::{available_nodes, NumaNode};
use std::process::Command;
use std::str::from_utf8;
/*
   We need to look at
   - clflush, measure reload (RAM)
   - clflush followed by prefetch L3, measure reload (pL3)
   - clflush followed by prefetch L2, measure reload (pL2)
   - clflush followed by prefetch L2, measure reload (pL1)
   - Load L1, measure reload (L1)
   - Load L1, evict from L1, measure reload (eL2)
   - Load L1, evict L1 + L2, measure reload? (eL3)
   - measure nop (nop)

   Important things to look at : detailed histograms of the diagonals
   Medians for all the core combinations. (Overlapped + separate)

   Checks that can be done :
   - Validate p vs e method to get hit from a specific cache level ?
   - Identify the timing range for the various level of the cache

   Plot system design : core & slice identification must be manual
   Generate detailed graphs with the Virtual Slices, and cores, in SCV format
   Use a python script to perform the slice and core permutations.

   [ ] Refactor IP_Tool from prefetcher reverse into cache util, generate the requisite templates
   [ ] Generate to various preparation functions.
   [ ] Add the required CalibrateOperation2T
   [ ] Result exploitation ?
   - [ ] Determine the CSV format suitable for the plots for histograms
   - [ ] Determine the CSV format suitable for the median plots
   - [ ] Output the histogram CSV
   - [ ] Output the Median CSV
   - [ ] Make the plots
*/

unsafe fn function_call(f: &Function, addr: *const u8) -> u64 {
    unsafe { (f.fun)(addr) }
}

unsafe fn prepare_RAM(p: *const u8) {
    unsafe { flush(p) };
}

unsafe fn prepare_pL3(p: *const u8) {
    unsafe { maccess(p) };
    unsafe { arch_x86::_mm_mfence() };
    unsafe { flush(p) };
    unsafe { arch_x86::_mm_mfence() };
    unsafe { arch_x86::_mm_prefetch::<{ arch_x86::_MM_HINT_T2 }>(p as *const i8) };
    unsafe { arch_x86::__cpuid_count(0, 0) };
}

unsafe fn prepare_pL2(p: *const u8) {
    unsafe { maccess(p) };
    unsafe { arch_x86::_mm_mfence() };
    unsafe { flush(p) };
    unsafe { arch_x86::_mm_mfence() };
    unsafe { arch_x86::_mm_prefetch::<{ arch_x86::_MM_HINT_T1 }>(p as *const i8) };
    unsafe { arch_x86::__cpuid_count(0, 0) };
}

unsafe fn prepare_pL1(p: *const u8) {
    unsafe { maccess(p) };
    unsafe { arch_x86::_mm_mfence() };
    unsafe { flush(p) };
    unsafe { arch_x86::_mm_mfence() };
    unsafe { arch_x86::_mm_prefetch::<{ arch_x86::_MM_HINT_T0 }>(p as *const i8) };
    unsafe { arch_x86::__cpuid_count(0, 0) };
}

unsafe fn prepare_L1(p: *const u8) {
    unsafe { only_reload(p) };
}

unsafe fn prepare_eL2(p: *const u8) {
    unimplemented!()
}

unsafe fn prepare_eL3(p: *const u8) {
    unimplemented!()
}

unsafe fn multiple_access(p: *const u8) {
    unsafe {
        maccess::<u8>(p);
        maccess::<u8>(p);
        arch_x86::_mm_mfence();
        maccess::<u8>(p);
        arch_x86::_mm_mfence();
        maccess::<u8>(p);
        arch_x86::_mm_mfence();
        maccess::<u8>(p);
        maccess::<u8>(p);
    }
}

const SIZE: usize = 2 << 20;
const MAX_SEQUENCE: usize = 2048 * 64;

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
struct ASV {
    pub attacker: u8,
    pub slice: u8,
    pub victim: u8,
}

struct ResultAnalysis {
    // indexed by bucket size
    pub miss: Vec<u32>,
    pub miss_cum_sum: Vec<u32>,
    pub miss_total: u32,
    pub hit: Vec<u32>,
    pub hit_cum_sum: Vec<u32>,
    pub hit_total: u32,
    pub error_miss_less_than_hit: Vec<u32>,
    pub error_hit_less_than_miss: Vec<u32>,
    pub min_error_hlm: u32,
    pub min_error_mlh: u32,
}

// Split the threshold and error in two separate structs ?

#[derive(Debug, Clone, Copy)]
struct Threshold {
    pub error_rate: f32,
    pub threshold: usize,
    // extend with other possible algorithm ?
    pub is_hlm: bool,
    pub num_true_hit: u32,
    pub num_false_hit: u32,
    pub num_true_miss: u32,
    pub num_false_miss: u32,
}

fn main() {
    let measure_reload =
        cache_utils::ip_tool::Function::try_new(1, 0, cache_utils::ip_tool::TIMED_MACCESS).unwrap();
    let measure_nop =
        cache_utils::ip_tool::Function::try_new(1, 0, cache_utils::ip_tool::TIMED_NOP).unwrap();
    // Grab a slice of memory

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

    println!("Number of cores per socket: {}", core_per_socket);

    let m = MMappedMemory::new(SIZE, true, false, |i: usize| i as u8);
    let array = m.slice();

    let cache_line_size = 64;

    let node = available_nodes().unwrap().into_iter().next().unwrap();

    // Generate core iterator
    let mut core_pairs: Vec<(NumaNode, usize, usize)> = Vec::new();
    let old = sched_getaffinity(Pid::from_raw(0)).unwrap();

    for i in 0..CpuSet::count() {
        for j in 0..CpuSet::count() {
            if old.is_set(i).unwrap() && old.is_set(j).unwrap() {
                core_pairs.push((node, i, j));
                println!("{},{}", i, j);
            }
        }
    }

    // operations
    // Call calibrate 2T \o/

    let verbose_level = Verbosity::RawResult;

    let pointer = (&array[0]) as *const u8;
    if pointer as usize & (cache_line_size - 1) != 0 {
        panic!("not aligned nicely");
    }

    let operations = [
        CalibrateOperation2T {
            prepare: prepare_RAM,
            op: function_call,
            name: "RAM_load",
            display_name: "Load from RAM",
            t: &measure_reload,
        },
        CalibrateOperation2T {
            prepare: prepare_pL3,
            op: function_call,
            name: "pL3_load",
            display_name: "Load from L3 (prefetch)",
            t: &measure_reload,
        },
        CalibrateOperation2T {
            prepare: prepare_pL2,
            op: function_call,
            name: "pL2_load",
            display_name: "Load from L2 (prefetch)",
            t: &measure_reload,
        },
        CalibrateOperation2T {
            prepare: prepare_pL1,
            op: function_call,
            name: "pL1_load",
            display_name: "Load from L1 (prefetch)",
            t: &measure_reload,
        },
        CalibrateOperation2T {
            prepare: prepare_L1,
            op: function_call,
            name: "L1_load",
            display_name: "Load from L1 (Reload)",
            t: &measure_reload,
        },
        CalibrateOperation2T {
            prepare: noop::<u8>,
            op: function_call,
            name: "pL3_load",
            display_name: "Load from L3 (prefetch)",
            t: &measure_nop,
        },
    ];

    let r: Result<
        Vec<CalibrateResult2TNuma<CLFLUSH_BUCKET_SIZE, CLFLUSH_BUCKET_NUMBER>>,
        nix::Error,
    > = unsafe {
        calibrate_fixed_freq_2_thread_numa(
            pointer,
            64,                                      // FIXME : MAGIC
            min(array.len(), MAX_SEQUENCE) as isize, // MAGIC
            &mut core_pairs.into_iter(),
            &operations,
            CalibrationOptions {
                iterations: CLFLUSH_NUM_ITER,
                verbosity: verbose_level,
                optimised_addresses: true,
                measure_hash: false,
            },
            core_per_socket,
        )
    };

    unimplemented!();
}
