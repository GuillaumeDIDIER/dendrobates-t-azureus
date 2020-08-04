#![allow(clippy::missing_safety_doc)]

use crate::complex_addressing::{cache_slicing, CacheSlicing};
use crate::{flush, maccess, rdtsc_fence};

use cpuid::{CPUVendor, MicroArchitecture};

use core::arch::x86_64 as arch_x86;
#[cfg(feature = "no_std")]
use polling_serial::{serial_print as print, serial_println as println};

//#[cfg(feature = "use_std")]
//use nix::errno::Errno;
#[cfg(feature = "use_std")]
use nix::sched::{sched_getaffinity, sched_setaffinity, CpuSet};
#[cfg(feature = "use_std")]
use nix::unistd::Pid;
//#[cfg(feature = "use_std")]
//use nix::Error::Sys;
#[cfg(feature = "use_std")]
use nix::Error;
#[cfg(feature = "use_std")]
use std::sync::Arc;
#[cfg(feature = "use_std")]
use std::thread;

extern crate alloc;
use crate::calibration::Verbosity::*;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::min;
use core::ptr::null_mut;
use core::sync::atomic::{spin_loop_hint, AtomicBool, AtomicPtr, Ordering};
use itertools::Itertools;

use atomic::Atomic;

#[derive(Ord, PartialOrd, Eq, PartialEq)]
pub enum Verbosity {
    NoOutput,
    Thresholds,
    RawResult,
    Debug,
}

pub struct HistParams {
    pub iterations: u32,
    pub bucket_size: usize,
    pub bucket_number: usize,
}

pub struct CalibrationOptions {
    pub hist_params: HistParams,
    pub verbosity: Verbosity,
    pub optimised_addresses: bool,
}

impl CalibrationOptions {
    pub fn new(hist_params: HistParams, verbosity: Verbosity) -> CalibrationOptions {
        CalibrationOptions {
            hist_params,
            verbosity,
            optimised_addresses: false,
        }
    }
}

pub unsafe fn only_reload(p: *const u8) -> u64 {
    let t = rdtsc_fence();
    maccess(p);
    rdtsc_fence() - t
}

pub unsafe fn flush_and_reload(p: *const u8) -> u64 {
    flush(p);
    only_reload(p)
}

pub unsafe fn reload_and_flush(p: *const u8) -> u64 {
    let r = only_reload(p);
    flush(p);
    r
}

pub unsafe fn only_flush(p: *const u8) -> u64 {
    let t = rdtsc_fence();
    flush(p);
    rdtsc_fence() - t
}

pub unsafe fn load_and_flush(p: *const u8) -> u64 {
    maccess(p);
    only_flush(p)
}

pub unsafe fn flush_and_flush(p: *const u8) -> u64 {
    flush(p);
    only_flush(p)
}

pub unsafe fn l3_and_reload(p: *const u8) -> u64 {
    flush(p);
    arch_x86::_mm_mfence();
    arch_x86::_mm_prefetch(p as *const i8, arch_x86::_MM_HINT_T2);
    arch_x86::__cpuid_count(0, 0);
    only_reload(p)
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

pub const CFLUSH_BUCKET_SIZE: usize = 1;
pub const CFLUSH_BUCKET_NUMBER: usize = 500;

pub const CFLUSH_NUM_ITER: u32 = 1 << 11;

pub fn calibrate_flush(
    array: &[u8],
    cache_line_size: usize,
    verbose_level: Verbosity,
) -> Vec<CalibrateResult> {
    let pointer = (&array[0]) as *const u8;

    if pointer as usize & (cache_line_size - 1) != 0 {
        panic!("not aligned nicely");
    }

    calibrate_impl_fixed_freq(
        pointer,
        cache_line_size,
        array.len() as isize,
        &[
            CalibrateOperation {
                op: load_and_flush,
                name: "clflush_hit",
                display_name: "clflush hit",
            },
            CalibrateOperation {
                op: flush_and_flush,
                name: "clflush_miss",
                display_name: "clflush miss",
            },
        ],
        HistParams {
            bucket_number: CFLUSH_BUCKET_NUMBER,
            bucket_size: CFLUSH_BUCKET_SIZE,
            iterations: CFLUSH_NUM_ITER,
        },
        verbose_level,
    )
}

#[derive(Debug)]
pub struct CalibrateResult {
    pub offset: isize,
    pub histogram: Vec<Vec<u32>>,
    pub median: Vec<u64>,
    pub min: Vec<u64>,
    pub max: Vec<u64>,
}

pub struct CalibrateOperation<'a> {
    pub op: unsafe fn(*const u8) -> u64,
    pub name: &'a str,
    pub display_name: &'a str,
}

pub unsafe fn calibrate(
    p: *const u8,
    increment: usize,
    len: isize,
    operations: &[CalibrateOperation],
    buckets_num: usize,
    bucket_size: usize,
    num_iterations: u32,
    verbosity_level: Verbosity,
) -> Vec<CalibrateResult> {
    calibrate_impl_fixed_freq(
        p,
        increment,
        len,
        operations,
        HistParams {
            bucket_number: buckets_num,
            bucket_size,
            iterations: num_iterations,
        },
        verbosity_level,
    )
}

const SPURIOUS_THRESHOLD: u32 = 1;
fn calibrate_impl_fixed_freq(
    p: *const u8,
    increment: usize,
    len: isize,
    operations: &[CalibrateOperation],
    hist_params: HistParams,
    verbosity_level: Verbosity,
) -> Vec<CalibrateResult> {
    if verbosity_level >= Thresholds {
        println!(
            "Calibrating {}...",
            operations
                .iter()
                .map(|operation| { operation.display_name })
                .format(", ")
        );
    }

    let to_bucket = |time: u64| -> usize { time as usize / hist_params.bucket_size };
    let from_bucket = |bucket: usize| -> u64 { (bucket * hist_params.bucket_size) as u64 };

    // FIXME : Core per socket
    let slicing = if let Some(uarch) = MicroArchitecture::get_micro_architecture() {
        if let Some(vendor_family_model_stepping) = MicroArchitecture::get_family_model_stepping() {
            Some(cache_slicing(
                uarch,
                8,
                vendor_family_model_stepping.0,
                vendor_family_model_stepping.1,
                vendor_family_model_stepping.2,
            ))
        } else {
            None
        }
    } else {
        None
    };

    let h = if let Some(s) = slicing {
        if s.can_hash() {
            Some(|addr: usize| -> u8 { slicing.unwrap().hash(addr).unwrap() })
        } else {
            None
        }
    } else {
        None
    };
    // TODO fix the GROSS hack of using max cpu core supported

    let mut ret = Vec::new();
    if verbosity_level >= Thresholds {
        print!("CSV: address, ");
        if h.is_some() {
            print!("hash, ");
        }
        println!(
            "{} min, {} median, {} max",
            operations
                .iter()
                .map(|operation| operation.name)
                .format(" min, "),
            operations
                .iter()
                .map(|operation| operation.name)
                .format(" median, "),
            operations
                .iter()
                .map(|operation| operation.name)
                .format(" max, ")
        );
    }
    if verbosity_level >= RawResult {
        print!("RESULT:address,");
        if h.is_some() {
            print!("hash,");
        }
        println!(
            "time,{}",
            operations
                .iter()
                .map(|operation| operation.name)
                .format(",")
        );
    }

    for i in (0..len).step_by(increment) {
        let pointer = unsafe { p.offset(i) };
        let hash = h.map(|h| h(pointer as usize));

        if verbosity_level >= Thresholds {
            print!("Calibration for {:p}", pointer);
            if let Some(h) = hash {
                print!(" (hash: {:x})", h)
            }
            println!();
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
            let mut hist = vec![0; hist_params.bucket_number];
            for _ in 0..hist_params.iterations {
                let time = unsafe { (op.op)(pointer) };
                let bucket = min(hist_params.bucket_number - 1, to_bucket(time));
                hist[bucket] += 1;
            }
            calibrate_result.histogram.push(hist);
        }

        let mut sums = vec![0; operations.len()];

        let median_thresholds: Vec<u32> = calibrate_result
            .histogram
            .iter()
            .map(|h| (hist_params.iterations - h[hist_params.bucket_number - 1]) / 2)
            .collect();

        for j in 0..hist_params.bucket_number - 1 {
            if verbosity_level >= RawResult {
                print!("RESULT:{:p},", pointer);
                if let Some(h) = hash {
                    print!("{:x},", h);
                }
                print!("{}", from_bucket(j));
            }
            // ignore the last bucket : spurious context switches etc.
            for op in 0..operations.len() {
                let hist = &calibrate_result.histogram[op][j];
                let min = &mut calibrate_result.min[op];
                let max = &mut calibrate_result.max[op];
                let med = &mut calibrate_result.median[op];
                let sum = &mut sums[op];
                if verbosity_level >= RawResult {
                    print!(",{}", hist);
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
            for (j, op) in operations.iter().enumerate() {
                println!(
                    "{}: min {}, median {}, max {}",
                    op.display_name,
                    calibrate_result.min[j],
                    calibrate_result.median[j],
                    calibrate_result.max[j]
                );
            }
            print!("CSV: {:p}, ", pointer);
            if let Some(h) = hash {
                print!("{:x}, ", h)
            }
            println!(
                "{}, {}, {}",
                calibrate_result.min.iter().format(", "),
                calibrate_result.median.iter().format(", "),
                calibrate_result.max.iter().format(", ")
            );
        }
        ret.push(calibrate_result);
    }
    ret
}

#[cfg(feature = "use_std")]
pub struct CalibrateOperation2T<'a> {
    pub prepare: unsafe fn(*const u8) -> (),
    pub op: unsafe fn(*const u8) -> u64,
    pub name: &'a str,
    pub display_name: &'a str,
}

#[cfg(feature = "use_std")]
pub struct CalibrateResult2T {
    pub main_core: usize,
    pub helper_core: usize,
    pub res: Result<Vec<CalibrateResult>, nix::Error>, // TODO

                                                       // TODO
}

fn wait(turn_lock: &AtomicBool, turn: bool) {
    while turn_lock.load(Ordering::Acquire) != turn {
        spin_loop_hint();
    }
    assert_eq!(turn_lock.load(Ordering::Relaxed), turn);
}

fn next(turn_lock: &AtomicBool) {
    turn_lock.fetch_xor(true, Ordering::Release);
}

#[cfg(feature = "use_std")]
pub unsafe fn calibrate_fixed_freq_2_thread<I: Iterator<Item = (usize, usize)>>(
    p: *const u8,
    increment: usize,
    len: isize,
    cores: &mut I,
    operations: &[CalibrateOperation2T],
    options: CalibrationOptions,
    core_per_socket: u8,
) -> Vec<CalibrateResult2T> {
    calibrate_fixed_freq_2_thread_impl(
        p,
        increment,
        len,
        cores,
        operations,
        options,
        core_per_socket,
    )
}

pub fn get_cache_slicing(core_per_socket: u8) -> Option<CacheSlicing> {
    if let Some(uarch) = MicroArchitecture::get_micro_architecture() {
        if let Some(vendor_family_model_stepping) = MicroArchitecture::get_family_model_stepping() {
            Some(cache_slicing(
                uarch,
                core_per_socket,
                vendor_family_model_stepping.0,
                vendor_family_model_stepping.1,
                vendor_family_model_stepping.2,
            ))
        } else {
            None
        }
    } else {
        None
    }
}

const OPTIMISED_ADDR_ITER_FACTOR: u32 = 16;

// TODO : Add the optimised address support
// TODO : Modularisation / factorisation of some of the common code with the single threaded no_std version ?

#[cfg(feature = "use_std")]
fn calibrate_fixed_freq_2_thread_impl<I: Iterator<Item = (usize, usize)>>(
    p: *const u8,
    increment: usize,
    len: isize,
    cores: &mut I,
    operations: &[CalibrateOperation2T],
    mut options: CalibrationOptions,
    core_per_socket: u8,
) -> Vec<CalibrateResult2T> {
    if options.verbosity >= Thresholds {
        println!(
            "Calibrating {}...",
            operations
                .iter()
                .map(|operation| { operation.display_name })
                .format(", ")
        );
    }

    let bucket_size = options.hist_params.bucket_size;

    let to_bucket = |time: u64| -> usize { time as usize / bucket_size };
    let from_bucket = |bucket: usize| -> u64 { (bucket * bucket_size) as u64 };

    let slicing = get_cache_slicing(core_per_socket);

    let h = if let Some(s) = slicing {
        if s.can_hash() {
            Some(|addr: usize| -> u8 { slicing.unwrap().hash(addr).unwrap() })
        } else {
            None
        }
    } else {
        None
    };

    let mut ret = Vec::new();

    let helper_thread_params = Arc::new(HelperThreadParams {
        turn: AtomicBool::new(false),
        stop: AtomicBool::new(true),
        op: Atomic::new(operations[0].prepare),
        address: AtomicPtr::new(null_mut()),
    });

    if options.verbosity >= Thresholds {
        print!("CSV: main_core, helper_core, address, ");
        if h.is_some() {
            print!("hash, ");
        }
        println!(
            "{} min, {} median, {} max",
            operations
                .iter()
                .map(|operation| operation.name)
                .format(" min, "),
            operations
                .iter()
                .map(|operation| operation.name)
                .format(" median, "),
            operations
                .iter()
                .map(|operation| operation.name)
                .format(" max, ")
        );
    }

    if options.verbosity >= RawResult {
        print!("RESULT:main_core,helper_core,address,");
        if h.is_some() {
            print!("hash,");
        }
        println!(
            "time,{}",
            operations
                .iter()
                .map(|operation| operation.name)
                .format(",")
        );
    }

    let image_antecedent = match slicing {
        Some(s) => s.image_antecedent(len as usize - 1),
        None => None,
    };
    if image_antecedent.is_some() {
        options.hist_params.iterations *= OPTIMISED_ADDR_ITER_FACTOR;
    }

    let old = sched_getaffinity(Pid::from_raw(0)).unwrap();

    for (main_core, helper_core) in cores {
        // set main thread affinity

        if options.verbosity >= Thresholds {
            println!(
                "Calibration for main_core {}, helper {}.",
                main_core, helper_core
            );
        }

        eprintln!(
            "Calibration for main_core {}, helper {}.",
            main_core, helper_core
        );

        let mut core = CpuSet::new();
        match core.set(main_core) {
            Ok(_) => {}
            Err(e) => {
                ret.push(CalibrateResult2T {
                    main_core,
                    helper_core,
                    res: Err(e),
                });
                continue;
            }
        }

        match sched_setaffinity(Pid::from_raw(0), &core) {
            Ok(_) => {}
            Err(e) => {
                ret.push(CalibrateResult2T {
                    main_core,
                    helper_core,
                    res: Err(e),
                });
                continue;
            }
        }

        let helper_thread = if helper_core != main_core {
            helper_thread_params.stop.store(false, Ordering::Relaxed);
            // set up the helper thread

            let htp = helper_thread_params.clone();
            let hc = helper_core;
            Some(thread::spawn(move || {
                calibrate_fixed_freq_2_thread_helper(htp, hc)
            }))
        } else {
            None
        };
        // do the calibration
        let mut calibrate_result_vec = Vec::new();

        let offsets: Box<dyn Iterator<Item = isize>> = match image_antecedent {
            Some(ref ima) => Box::new(ima.values().copied()),
            None => Box::new((0..len as isize).step_by(increment)),
        };

        for i in offsets {
            let pointer = unsafe { p.offset(i) };
            helper_thread_params
                .address
                .store(pointer as *mut u8, Ordering::Relaxed);

            let hash = h.map(|h| h(pointer as usize));

            if options.verbosity >= Thresholds {
                print!("Calibration for {:p}", pointer);
                if let Some(h) = hash {
                    print!(" (hash: {:x})", h)
                }
                println!();
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

            if helper_core != main_core {
                for op in operations {
                    helper_thread_params.op.store(op.prepare, Ordering::Relaxed);
                    let mut hist = vec![0; options.hist_params.bucket_number];
                    for _ in 0..options.hist_params.iterations {
                        next(&helper_thread_params.turn);
                        wait(&helper_thread_params.turn, false);
                        let _time = unsafe { (op.op)(pointer) };
                    }
                    for _ in 0..options.hist_params.iterations {
                        next(&helper_thread_params.turn);
                        wait(&helper_thread_params.turn, false);
                        let time = unsafe { (op.op)(pointer) };
                        let bucket = min(options.hist_params.bucket_number - 1, to_bucket(time));
                        hist[bucket] += 1;
                    }
                    calibrate_result.histogram.push(hist);
                }
            } else {
                for op in operations {
                    let mut hist = vec![0; options.hist_params.bucket_number];
                    for _ in 0..options.hist_params.iterations {
                        unsafe { (op.prepare)(pointer) };
                        unsafe { arch_x86::_mm_mfence() }; // Test with this ?
                        let _time = unsafe { (op.op)(pointer) };
                    }
                    for _ in 0..options.hist_params.iterations {
                        unsafe { (op.prepare)(pointer) };
                        unsafe { arch_x86::_mm_mfence() }; // Test with this ?
                        let time = unsafe { (op.op)(pointer) };
                        let bucket = min(options.hist_params.bucket_number - 1, to_bucket(time));
                        hist[bucket] += 1;
                    }
                    calibrate_result.histogram.push(hist);
                }
            }
            let mut sums = vec![0; operations.len()];

            let median_thresholds: Vec<u32> = calibrate_result
                .histogram
                .iter()
                .map(|h| {
                    (options.hist_params.iterations - h[options.hist_params.bucket_number - 1]) / 2
                })
                .collect();

            for j in 0..options.hist_params.bucket_number - 1 {
                if options.verbosity >= RawResult {
                    print!("RESULT:{},{},{:p},", main_core, helper_core, pointer);
                    if let Some(h) = hash {
                        print!("{:x},", h);
                    }
                    print!("{}", from_bucket(j));
                }
                // ignore the last bucket : spurious context switches etc.
                for op in 0..operations.len() {
                    let hist = &calibrate_result.histogram[op][j];
                    let min = &mut calibrate_result.min[op];
                    let max = &mut calibrate_result.max[op];
                    let med = &mut calibrate_result.median[op];
                    let sum = &mut sums[op];
                    if options.verbosity >= RawResult {
                        print!(",{}", hist);
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
                if options.verbosity >= RawResult {
                    println!();
                }
            }
            if options.verbosity >= Thresholds {
                for (j, op) in operations.iter().enumerate() {
                    println!(
                        "{}: min {}, median {}, max {}",
                        op.display_name,
                        calibrate_result.min[j],
                        calibrate_result.median[j],
                        calibrate_result.max[j]
                    );
                }
                print!("CSV: {},{},{:p}, ", main_core, helper_core, pointer);
                if let Some(h) = hash {
                    print!("{:x}, ", h)
                }
                println!(
                    "{}, {}, {}",
                    calibrate_result.min.iter().format(", "),
                    calibrate_result.median.iter().format(", "),
                    calibrate_result.max.iter().format(", ")
                );
            }
            calibrate_result_vec.push(calibrate_result);
        }

        ret.push(CalibrateResult2T {
            main_core,
            helper_core,
            res: Ok(calibrate_result_vec),
        });

        if helper_core != main_core {
            // terminate the thread
            helper_thread_params.stop.store(true, Ordering::Relaxed);
            next(&helper_thread_params.turn);
            wait(&helper_thread_params.turn, false);
            // join thread.
            helper_thread.unwrap().join();
        }
    }

    sched_setaffinity(Pid::from_raw(0), &old).unwrap();

    ret
    // return the result
    // TODO
}
#[cfg(feature = "use_std")]
struct HelperThreadParams {
    turn: AtomicBool,
    stop: AtomicBool,
    op: Atomic<unsafe fn(*const u8)>,
    address: AtomicPtr<u8>,
}

#[cfg(feature = "use_std")]
fn calibrate_fixed_freq_2_thread_helper(
    params: Arc<HelperThreadParams>,
    helper_core: usize,
) -> Result<(), Error> {
    // set thread affinity
    let mut core = CpuSet::new();
    match core.set(helper_core) {
        Ok(_) => {}
        Err(_e) => {
            unimplemented!();
        }
    }

    match sched_setaffinity(Pid::from_raw(0), &core) {
        Ok(_) => {}
        Err(_e) => {
            unimplemented!();
        }
    }

    loop {
        // grab lock
        wait(&params.turn, true);
        if params.stop.load(Ordering::Relaxed) {
            next(&params.turn);
            return Ok(());
        }
        // get the relevant parameters
        let addr: *const u8 = params.address.load(Ordering::Relaxed);
        let op = params.op.load(Ordering::Relaxed);
        unsafe { op(addr) };
        // release lock
        next(&params.turn);
    }
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

    let r = calibrate_impl_fixed_freq(
        pointer,
        cache_line_size,
        array.len() as isize,
        &[CalibrateOperation {
            op: l3_and_reload,
            name: "l3_hit",
            display_name: "L3 hit",
        }],
        HistParams {
            bucket_number: 512,
            bucket_size: 2,
            iterations: 1 << 11,
        },
        verbose_level,
    );

    r.into_iter().next().unwrap()
}
