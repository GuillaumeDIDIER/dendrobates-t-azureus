#![allow(clippy::missing_safety_doc)]

use crate::{flush, maccess, rdtsc_fence};

use core::arch::x86_64 as arch_x86;
#[cfg(feature = "no_std")]
use polling_serial::{serial_print as print, serial_println as println};

#[derive(Ord, PartialOrd, Eq, PartialEq)]
pub enum Verbosity {
    NoOutput,
    Thresholds,
    RawResult,
    Debug,
}

extern crate alloc;
use crate::calibration::Verbosity::*;
use crate::complex_addressing::AddressHasher;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::min;
use itertools::Itertools;

pub unsafe fn only_reload(p: *const u8) -> u64 {
    let t = rdtsc_fence();
    maccess(p);
    rdtsc_fence() - t
}

pub unsafe fn flush_and_reload(p: *const u8) -> u64 {
    flush(p);
    let t = rdtsc_fence();
    maccess(p);
    rdtsc_fence() - t
}

pub unsafe fn load_and_flush(p: *const u8) -> u64 {
    maccess(p);
    let t = rdtsc_fence();
    flush(p);
    rdtsc_fence() - t
}

pub unsafe fn flush_and_flush(p: *const u8) -> u64 {
    flush(p);
    let t = rdtsc_fence();
    flush(p);
    rdtsc_fence() - t
}
pub unsafe fn only_flush(p: *const u8) -> u64 {
    let t = rdtsc_fence();
    flush(p);
    rdtsc_fence() - t
}

pub unsafe fn l3_and_reload(p: *const u8) -> u64 {
    flush(p);
    arch_x86::_mm_mfence();
    arch_x86::_mm_prefetch(p as *const i8, arch_x86::_MM_HINT_T2);
    arch_x86::__cpuid_count(0, 0);
    let t = rdtsc_fence();
    maccess(p);
    rdtsc_fence() - t
}

const BUCKET_SIZE: usize = 5;
const BUCKET_NUMBER: usize = 250;

// TODO same as below, also add the whole page calibration

pub fn calibrate_access(array: &[u8; 4096]) -> u64 {
    println!("Calibrating...");

    // Allocate a target array
    // TBD why size, why the position in the array, why the type (usize)
    //    let mut array = Vec::<usize>::with_capacity(5 << 10);
    //    array.resize(5 << 10, 1);

    //    let array = array.into_boxed_slice();

    // Histograms bucket of 5 and max at 400 cycles
    // Magic numbers to be justified
    // 80 is a size of screen
    let mut hit_histogram = vec![0; BUCKET_NUMBER]; //Vec::<u32>::with_capacity(BUCKET_NUMBER);
                                                    //hit_histogram.resize(BUCKET_NUMBER, 0);

    let mut miss_histogram = hit_histogram.clone();

    // the address in memory we are going to target
    let pointer = &array[0] as *const u8;

    println!("buffer start {:p}", pointer);

    if pointer as usize & 0x3f != 0 {
        panic!("not aligned nicely");
    }

    // do a large sample of accesses to a cached line
    unsafe { maccess(pointer) };
    for i in 0..(4 << 10) {
        for _ in 0..(1 << 10) {
            let d = unsafe { only_reload(pointer.offset(i & (!0x3f))) } as usize;
            hit_histogram[min(BUCKET_NUMBER - 1, d / BUCKET_SIZE) as usize] += 1;
        }
    }

    // do a large numer of accesses to uncached line
    unsafe { flush(pointer) };
    for i in 0..(4 << 10) {
        for _ in 0..(1 << 10) {
            let d = unsafe { flush_and_reload(pointer.offset(i & (!0x3f))) } as usize;
            miss_histogram[min(BUCKET_NUMBER - 1, d / BUCKET_SIZE) as usize] += 1;
        }
    }

    let mut hit_max = 0;
    let mut hit_max_i = 0;
    let mut miss_min_i = 0;
    for i in 0..hit_histogram.len() {
        println!(
            "{:3}: {:10} {:10}",
            i * BUCKET_SIZE,
            hit_histogram[i],
            miss_histogram[i]
        );
        if hit_max < hit_histogram[i] {
            hit_max = hit_histogram[i];
            hit_max_i = i;
        }
        if miss_histogram[i] > 3 /* Magic */ && miss_min_i == 0 {
            miss_min_i = i
        }
    }
    println!("Miss min {}", miss_min_i * BUCKET_SIZE);
    println!("Max hit {}", hit_max_i * BUCKET_SIZE);

    let mut min = u32::max_value();
    let mut min_i = 0;
    for i in hit_max_i..miss_min_i {
        if min > hit_histogram[i] + miss_histogram[i] {
            min = hit_histogram[i] + miss_histogram[i];
            min_i = i;
        }
    }

    println!("Threshold {}", min_i * BUCKET_SIZE);
    println!("Calibration done.");
    (min_i * BUCKET_SIZE) as u64
}

const CFLUSH_BUCKET_SIZE: usize = 1;
const CFLUSH_BUCKET_NUMBER: usize = 500;

const CFLUSH_NUM_ITER: u32 = 1 << 11;

pub fn calibrate_flush(
    array: &[u8],
    cache_line_size: usize,
    verbose_level: Verbosity,
) -> Vec<CalibrateResult> {
    let pointer = (&array[0]) as *const u8;

    if pointer as usize & (cache_line_size - 1) != 0 {
        panic!("not aligned nicely");
    }

    calibrate_impl(
        pointer,
        cache_line_size,
        array.len() as isize,
        &[
            (load_and_flush, "clflush hit"),
            (flush_and_flush, "clflush miss"),
        ],
        CFLUSH_BUCKET_NUMBER,
        CFLUSH_BUCKET_SIZE,
        CFLUSH_NUM_ITER,
        verbose_level,
    )
}

#[derive(Debug)]
pub struct CalibrateResult {
    offset: isize,
    histogram: Vec<Vec<u32>>,
    median: Vec<u64>,
    min: Vec<u64>,
    max: Vec<u64>,
}

pub unsafe fn calibrate(
    p: *const u8,
    increment: usize,
    len: isize,
    operations: &[(unsafe fn(*const u8) -> u64, &str)],
    buckets_num: usize,
    bucket_size: usize,
    num_iterations: u32,
    verbosity_level: Verbosity,
) -> Vec<CalibrateResult> {
    calibrate_impl(
        p,
        increment,
        len,
        operations,
        buckets_num,
        bucket_size,
        num_iterations,
        verbosity_level,
    )
}

const SPURIOUS_THRESHOLD: u32 = 1;
fn calibrate_impl(
    p: *const u8,
    increment: usize,
    len: isize,
    operations: &[(unsafe fn(*const u8) -> u64, &str)],
    buckets_num: usize,
    bucket_size: usize,
    num_iterations: u32,
    verbosity_level: Verbosity,
) -> Vec<CalibrateResult> {
    // TODO : adapt this to detect CPU generation and grab the correct masks.
    // These are the skylake masks.
    let masks: [usize; 3] = [
        0b1111_0011_0011_0011_0010_0100_1100_0100_000000,
        0b1011_1010_1101_0111_1110_1010_1010_0010_000000,
        0b0110_1101_0111_1101_0101_1101_0101_0001_000000,
    ];

    let hasher = AddressHasher::new(&masks);

    if verbosity_level >= Thresholds {
        println!(
            "Calibrating {}...",
            operations.iter().map(|(_, name)| { name }).format(", ")
        );
    }

    let to_bucket = |time: u64| -> usize { time as usize / bucket_size };
    let from_bucket = |bucket: usize| -> u64 { (bucket * bucket_size) as u64 };
    let mut ret = Vec::new();
    if verbosity_level >= Thresholds {
        println!(
            "CSV: address, hash, {} min, {} median, {} max",
            operations.iter().map(|(_, name)| name).format(" min, "),
            operations.iter().map(|(_, name)| name).format(" median, "),
            operations.iter().map(|(_, name)| name).format(" max, ")
        );
    }
    for i in (0..len).step_by(increment) {
        let pointer = unsafe { p.offset(i) };
        let hash = hasher.hash(pointer as usize);

        if verbosity_level >= Thresholds {
            println!("Calibration for {:p} (hash: {:x})", pointer, hash);
        }

        // TODO add some useful impl to CalibrateResults
        let mut calibrate_result = CalibrateResult {
            offset: i,
            histogram: Vec::new(),
            median: vec![0; operations.len()],
            min: vec![0; operations.len()],
            max: vec![0; operations.len()],
        };
        calibrate_result.histogram.reserve(operations.len());

        for op in operations {
            let mut hist = vec![0; buckets_num];
            for _ in 0..num_iterations {
                let time = unsafe { op.0(pointer) };
                let bucket = min(buckets_num - 1, to_bucket(time));
                hist[bucket] += 1;
            }
            calibrate_result.histogram.push(hist);
        }

        let mut sums = vec![0; operations.len()];

        let median_thresholds: Vec<u32> = calibrate_result
            .histogram
            .iter()
            .map(|h| (num_iterations - h[buckets_num - 1]) / 2)
            .collect();

        if verbosity_level >= RawResult {
            println!(
                "time {}",
                operations.iter().map(|(_, name)| name).format(" ")
            );
        }

        for j in 0..buckets_num - 1 {
            if verbosity_level >= RawResult {
                print!("{:3}:", from_bucket(j));
            }
            // ignore the last bucket : spurious context switches etc.
            for op in 0..operations.len() {
                let hist = &calibrate_result.histogram[op][j];
                let min = &mut calibrate_result.min[op];
                let max = &mut calibrate_result.max[op];
                let med = &mut calibrate_result.median[op];
                let sum = &mut sums[op];
                if verbosity_level >= RawResult {
                    print!("{:10}", hist);
                }

                if *min == 0 {
                    // looking for min
                    if *hist > SPURIOUS_THRESHOLD {
                        *min = from_bucket(j);
                    }
                } else if *hist > SPURIOUS_THRESHOLD {
                    *max = from_bucket(j);
                }

                if *med == 0 {
                    *sum += *hist;
                    if *sum >= median_thresholds[op] {
                        *med = from_bucket(j);
                    }
                }
            }
            if verbosity_level >= RawResult {
                println!();
            }
        }
        if verbosity_level >= Thresholds {
            for (j, (_, op)) in operations.iter().enumerate() {
                println!(
                    "{}: min {}, median {}, max {}",
                    op,
                    calibrate_result.min[j],
                    calibrate_result.median[j],
                    calibrate_result.max[j]
                );
            }
            println!(
                "CSV: {:p}, {:x}, {}, {}, {}",
                pointer,
                hash,
                calibrate_result.min.iter().format(", "),
                calibrate_result.median.iter().format(", "),
                calibrate_result.max.iter().format(", ")
            );
        }
        ret.push(calibrate_result);
    }
    ret
}

#[allow(non_snake_case)]
pub fn calibrate_L3_miss_hit(
    array: &[u8],
    cache_line_size: usize,
    verbose_level: Verbosity,
) -> CalibrateResult {
    if verbose_level > NoOutput {
        println!("Calibrating L3 access...");
    }
    let pointer = (&array[0]) as *const u8;

    let r = calibrate_impl(
        pointer,
        cache_line_size,
        array.len() as isize,
        &[(l3_and_reload, "L3 hit")],
        512,
        2,
        1 << 11,
        verbose_level,
    );

    r.into_iter().next().unwrap()
}
