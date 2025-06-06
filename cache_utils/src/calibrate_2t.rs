extern crate std;

use crate::calibration::Verbosity::{RawResult, Thresholds};
use crate::calibration::{
    get_cache_attack_slicing, get_vpn, CalibrateResult, CalibrationOptions, HashMap, ASVP,
    SPURIOUS_THRESHOLD,
};
use alloc::vec;
use cache_slice::determine_slice;
use cache_slice::utils::core_per_package;
use calibration_results::calibration::StaticHistCalibrateResult;
use calibration_results::histograms::{SimpleBucketU64, StaticHistogram};
use core::arch::x86_64 as arch_x86;
use cpuid::complex_addressing::CacheAttackSlicing;
use itertools::Itertools;
use nix::sched::{sched_getaffinity, sched_setaffinity, CpuSet};
use nix::unistd::Pid;
use nix::Error;
use numa_utils::NumaNode;

use calibration_results::calibration_2t::CalibrateResult2TNuma;
use calibration_results::numa_results::NumaCalibrationResult;
use rmp_serde::{Deserializer, Serializer};
use serde::{Deserialize, Serialize};
use std::ptr::null_mut;
use std::sync::{Arc, Mutex};
use std::thread;
use std::vec::Vec;
use std::{eprintln, print, println};
use turn_lock::TurnHandle;
use zstd;

pub struct CalibrateOperation2T<'a, T> {
    pub prepare: unsafe fn(*const u8) -> (),
    pub op: unsafe fn(&T, *const u8) -> u64,
    pub name: &'a str,
    pub display_name: &'a str,
    pub t: &'a T,
}

pub struct CalibrateResult2T {
    pub main_core: usize,
    pub helper_core: usize,
    pub res: Result<Vec<CalibrateResult>, nix::Error>, // TODO

                                                       // TODO
}

pub unsafe fn calibrate_fixed_freq_2_thread_numa<
    I: Iterator<Item = (NumaNode, usize, usize)>,
    T,
    const WIDTH: u64,
    const N: usize,
>(
    p: *const u8,
    increment: usize,
    len: isize,
    locations: &mut I,
    operations: &[CalibrateOperation2T<T>],
    options: CalibrationOptions,
    core_per_socket: u8,
) -> Result<Vec<CalibrateResult2TNuma<WIDTH, N>>, nix::Error> {
    calibrate_fixed_freq_2_thread_numa_impl(
        p,
        increment,
        len,
        locations,
        operations,
        options,
        core_per_socket,
    )
}

fn calibrate_fixed_freq_2_thread_numa_impl<
    I: Iterator<Item = (NumaNode, usize, usize)>,
    T,
    const WIDTH: u64,
    const N: usize,
>(
    p: *const u8,
    increment: usize,
    len: isize,
    locations: &mut I,
    operations: &[CalibrateOperation2T<T>],
    mut options: CalibrationOptions,
    core_per_socket: u8,
) -> Result<Vec<CalibrateResult2TNuma<WIDTH, N>>, nix::Error> {
    if options.verbosity >= Thresholds {
        println!(
            "Calibrating {}...",
            operations
                .iter()
                .map(|operation| { operation.display_name })
                .format(", ")
        );
    }

    let cache_line_length = 64; // FIXME MAGIC

    let slicing = match get_cache_attack_slicing(core_per_socket, cache_line_length) {
        Some(v) => v,
        None => panic!("Unable to determine cache slicing !"),
    };

    let store = std::fs::File::create("./tmp.msgpack.zst").unwrap();
    let mut encoder = zstd::Encoder::new(store, 0).unwrap().auto_finish();
    let mut serializer = Serializer::new(&mut encoder);
    let mut count = 0;
    //let mut ret = Vec::new();

    let mut turn_handles = TurnHandle::new(
        2,
        HelperThreadParams {
            stop: true,
            op: operations[0].prepare,
            address: null_mut(),
        },
    );

    let helper_turn_handle = Arc::new(Mutex::new(turn_handles.pop().unwrap()));
    let mut main_turn_handle = turn_handles.pop().unwrap();

    let mut params = main_turn_handle.wait();

    if options.verbosity >= Thresholds {
        println!(
            "CSV: numa_node, main_core, helper_core, address, hash, {} min, {} median, {} max",
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
        println!(
            "RESULT: numa_node, main_core, helper_core, address, hash, bucket, time_min, time_max, {}",
            operations
                .iter()
                .map(|operation| operation.name)
                .format(",")
        );
    }

    let image_antecedent = slicing.image_antecedent(len as usize - 1);
    //assert!(image_antecedent.contains_key(&0));
    //let mut sorted = image_antecedent.keys().copied().collect::<Vec<usize>>();
    //sorted.sort();
    //println!("Image antecedent: {:?}", sorted);
    match slicing {
        CacheAttackSlicing::ComplexAddressing(_) | CacheAttackSlicing::SimpleAddressing(_) => {
            options.iterations *= OPTIMISED_ADDR_ITER_FACTOR;
        }
        _ => {}
    }

    let mut current_numa_node = None;
    let old = sched_getaffinity(Pid::from_raw(0)).unwrap();

    let mut total_rejected: u64 = 0;
    let mut total_tries: u64 = 0;

    for (numa_node, main_core, helper_core) in locations {
        // set main thread affinity

        if options.verbosity >= Thresholds {
            println!(
                "Calibration for numa_node {}, main_core {}, helper {}.",
                numa_node, main_core, helper_core
            );

            eprintln!(
                "Calibration for numa_node {}, main_core {}, helper {}.",
                numa_node, main_core, helper_core
            );
        }

        if current_numa_node != Some(numa_node) {
            match numa_utils::set_memory_node(numa_node) {
                Ok(_) => {}
                Err(e) => {
                    panic!(
                        "Numa failed, setting node to {}, error handling unimplemented: {:?}",
                        numa_node, e
                    );
                }
            }
            current_numa_node = Some(numa_node)
        }

        let mut core = CpuSet::new();
        match core.set(main_core) {
            Ok(_) => {}
            Err(e) => {
                return Err(e);
            }
        }

        match sched_setaffinity(Pid::from_raw(0), &core) {
            Ok(_) => {}
            Err(e) => {
                return Err(e);
            }
        }

        let helper_thread = if helper_core != main_core {
            params.stop = false;
            // set up the helper thread

            let hc = helper_core;
            let th = helper_turn_handle.clone();
            Some(thread::spawn(move || {
                calibrate_fixed_freq_2_thread_helper(th, hc)
            }))
        } else {
            None
        };
        // do the calibration
        let mut calibrate_result_vec = Vec::new();

        //let offsets = (0..len as isize).step_by(increment);
        let offsets = image_antecedent.values().copied();

        /*
        let offsets: Box<dyn Iterator<Item = isize>> = match image_antecedent {
            Some(ref ima) => Box::new(ima.values().copied()),
            None => Box::new((0..len as isize).step_by(cache_line_length)),
        };*/
        let nb_cores = core_per_package();

        for i in offsets {
            let pointer = unsafe { p.offset(i) };
            params.address = pointer;

            let mut hash = slicing.hash(pointer as usize);
            if options.measure_hash {
                hash = determine_slice(pointer, main_core as u8, nb_cores).unwrap();
            }

            if options.verbosity >= Thresholds {
                print!("Calibration for {:p}", pointer);
                print!(" (hash: {:x})", hash);
                println!();
            }

            // TODO add some useful impl to CalibrateResults
            let mut calibrate_result = StaticHistCalibrateResult {
                page: get_vpn(pointer),
                offset: i,
                hash,
                histogram: Vec::new(),
                median: vec![0; operations.len()],
                min: vec![0; operations.len()],
                max: vec![0; operations.len()],
                count: vec![0; operations.len()],
            };
            calibrate_result.histogram.reserve(operations.len());

            if helper_core != main_core {
                for op in operations {
                    params = main_turn_handle.wait();
                    params.op = op.prepare;
                    let mut rejected: u32 = 0;
                    let mut histogram = StaticHistogram::<WIDTH, N>::empty();
                    for _ in 0..options.warmup_iterations {
                        main_turn_handle.next();
                        params = main_turn_handle.wait();
                        let _time = unsafe { (op.op)(op.t, pointer) };
                    }
                    for _ in 0..options.iterations {
                        main_turn_handle.next();
                        params = main_turn_handle.wait();
                        let time = unsafe { (op.op)(op.t, pointer) };
                        match histogram.get_mut(time) {
                            Some(b) => {
                                *b += 1;
                            }
                            None => {
                                rejected += 1;
                            }
                        }
                    }
                    histogram[&SimpleBucketU64::<WIDTH, N>::MAX] += rejected; // This should probably be handled better.
                    calibrate_result.histogram.push(histogram);
                    total_rejected += rejected as u64;
                    total_tries += options.iterations as u64;
                }
            } else {
                for op in operations {
                    let mut rejected: u32 = 0;
                    let mut histogram = StaticHistogram::<WIDTH, N>::empty();
                    for _ in 0..options.warmup_iterations {
                        unsafe { (op.prepare)(pointer) };
                        unsafe { arch_x86::_mm_mfence() }; // Test with this ?
                        let _time = unsafe { (op.op)(op.t, pointer) };
                    }
                    for _ in 0..options.iterations {
                        unsafe { (op.prepare)(pointer) };
                        unsafe { arch_x86::_mm_mfence() }; // Test with this ?
                        let time = unsafe { (op.op)(op.t, pointer) };
                        match histogram.get_mut(time) {
                            Some(b) => {
                                *b += 1;
                            }
                            None => {
                                rejected += 1;
                            }
                        }
                    }
                    histogram[&SimpleBucketU64::<WIDTH, N>::MAX] += rejected; // This should probably be handled better.
                    calibrate_result.histogram.push(histogram);
                    total_rejected += rejected as u64;
                    total_tries += options.iterations as u64;
                }
            }
            let mut sums = vec![0; operations.len()];

            let median_thresholds: Vec<u32> = calibrate_result
                .histogram
                .iter()
                .map(|h| (options.iterations - h[&SimpleBucketU64::<WIDTH, N>::MAX]) / 2)
                .collect();

            for j in SimpleBucketU64::<WIDTH, N>::MIN..=SimpleBucketU64::<WIDTH, N>::MAX {
                if options.verbosity >= RawResult {
                    print!(
                        "RESULT: {}, {}, {}, {:p}, {:x}",
                        numa_node, main_core, helper_core, pointer, hash
                    );
                    //print!("{:x},", hash);
                    let bucket_index: usize = j.into();
                    let time_min: u64 = j.into();
                    let time_max = time_min + WIDTH - 1;
                    print!("{}, {}, {}", bucket_index, time_min, time_max);
                }
                // ignore the last bucket : spurious context switches etc.
                for op in 0..operations.len() {
                    let hist = &calibrate_result.histogram[op][&j];
                    let min = &mut calibrate_result.min[op];
                    let max = &mut calibrate_result.max[op];
                    let med = &mut calibrate_result.median[op];
                    let count = &mut calibrate_result.count[op];
                    let sum = &mut sums[op];
                    if options.verbosity >= RawResult {
                        print!(", {}", hist);
                    }

                    *count += *hist as u64;
                    let time: u64 = j.into();
                    if *min == 0 {
                        // looking for min
                        if *hist > SPURIOUS_THRESHOLD {
                            *min = time;
                        }
                    } else if *hist > SPURIOUS_THRESHOLD {
                        let time: u64 = j.into();
                        *max = time + WIDTH - 1;
                    }

                    if *med == 0 {
                        *sum += *hist;
                        if *sum >= median_thresholds[op] {
                            *med = j.into();
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
                        "{}: min {}, median {}, max {}, count {}",
                        op.display_name,
                        calibrate_result.min[j],
                        calibrate_result.median[j],
                        calibrate_result.max[j],
                        calibrate_result.count[j],
                    );
                }
                print!(
                    "CSV: {}, {}, {}, {:p}, {:x}",
                    numa_node, main_core, helper_core, pointer, hash
                );
                //print!("{:x}, ", hash);
                println!(
                    "{}, {}, {}",
                    calibrate_result.min.iter().format(", "),
                    calibrate_result.median.iter().format(", "),
                    calibrate_result.max.iter().format(", ")
                );
            }
            calibrate_result_vec.push(calibrate_result);
        }

        let data = CalibrateResult2TNuma {
            numa_node,
            main_core,
            helper_core,
            res: calibrate_result_vec,
        };
        data.serialize(&mut serializer)
            .expect("Failed to serialize");
        count += 1;
        //ret.push(data);

        if helper_core != main_core {
            // terminate the thread
            params.stop = true;
            main_turn_handle.next();
            params = main_turn_handle.wait();
            // join thread.
            helper_thread
                .unwrap()
                .join()
                .expect("Failed to join thread");
            // FIXME error handling
        }
    }

    if let Err(e) = numa_utils::reset_memory_node() {
        eprintln!("Error reseting numa node: {:?}", e)
    }

    println!(
        "Total rejected {} out of {} tries",
        total_rejected, total_tries
    );

    sched_setaffinity(Pid::from_raw(0), &old).unwrap();

    drop(encoder);
    let store = std::fs::File::open("./tmp.msgpack.zst").unwrap();
    let decoder = zstd::Decoder::new(store).unwrap();
    let mut ret = Vec::new();
    let mut deserializer = Deserializer::new(decoder);
    for _i in 0..count {
        let data = CalibrateResult2TNuma::<WIDTH, N>::deserialize(&mut deserializer).unwrap();
        ret.push(data);
    }

    Ok(ret)
}
/*
pub unsafe fn calibrate_fixed_freq_2_thread<
    const WIDTH: u64,
    const N: usize,
    I: Iterator<Item = (usize, usize)>,
    T,
>(
    p: *const u8,
    increment: usize,
    len: isize,
    cores: &mut I,
    operations: &[CalibrateOperation2T<T>],
    options: CalibrationOptions,
    core_per_socket: u8,
) -> Vec<CalibrateResult2T> {
    calibrate_fixed_freq_2_thread_impl::<WIDTH, N, I, T>(
        p,
        increment,
        len,
        cores,
        operations,
        options,
        core_per_socket,
    )
}*/

const OPTIMISED_ADDR_ITER_FACTOR: u32 = 16;

struct HelperThreadParams {
    stop: bool,
    op: unsafe fn(*const u8),
    address: *const u8,
}

// TODO : Add the optimised address support
// TODO : Modularisation / factorisation of some of the common code with the single threaded no_std version ?
/*
//#[cfg(feature = "use_std")]
fn calibrate_fixed_freq_2_thread_impl<
    const WIDTH: u64,
    const N: usize,
    I: Iterator<Item = (usize, usize)>,
    T,
>(
    p: *const u8,
    cache_line_length: usize,
    len: isize,
    cores: &mut I,
    operations: &[CalibrateOperation2T<T>],
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

    let bucket_size = WIDTH;

    let to_bucket = |time: u64| -> usize { (time / bucket_size) as usize };
    let from_bucket = |bucket: usize| -> u64 { (bucket as u64) * bucket_size };

    let slicing = match get_cache_attack_slicing(core_per_socket, cache_line_length) {
        Some(v) => v,
        None => panic!("Unable to determine cache slicing !"),
    };

    let mut ret = Vec::new();

    let mut turn_handles = TurnHandle::new(
        2,
        HelperThreadParams {
            stop: true,
            op: operations[0].prepare,
            address: null_mut(),
        },
    );

    let helper_turn_handle = Arc::new(Mutex::new(turn_handles.pop().unwrap()));
    let mut main_turn_handle = turn_handles.pop().unwrap();

    let mut params = main_turn_handle.wait();

    if options.verbosity >= Thresholds {
        println!(
            "CSV: main_core, helper_core, address, hash, {} min, {} median, {} max",
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
        println!(
            "RESULT:main_core,helper_core,address,hash,time,{}",
            operations
                .iter()
                .map(|operation| operation.name)
                .format(",")
        );
    }

    let image_antecedent = slicing.image_antecedent(len as usize - 1);

    match slicing {
        CacheAttackSlicing::ComplexAddressing(_) | CacheAttackSlicing::SimpleAddressing(_) => {
            options.iterations *= OPTIMISED_ADDR_ITER_FACTOR;
        }
        _ => {}
    }

    let old = sched_getaffinity(Pid::from_raw(0)).unwrap();

    for (main_core, helper_core) in cores {
        // set main thread affinity

        if options.verbosity >= Thresholds {
            println!(
                "Calibration for main_core {}, helper {}.",
                main_core, helper_core
            );

            eprintln!(
                "Calibration for main_core {}, helper {}.",
                main_core, helper_core
            );
        }

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
            params.stop = false;
            // set up the helper thread

            let hc = helper_core;
            let th = helper_turn_handle.clone();
            Some(thread::spawn(move || {
                calibrate_fixed_freq_2_thread_helper(th, hc)
            }))
        } else {
            None
        };
        // do the calibration
        let mut calibrate_result_vec = Vec::new();

        let offsets = image_antecedent.values().copied();

        /*
        let offsets: Box<dyn Iterator<Item = isize>> = match image_antecedent {
            Some(ref ima) => Box::new(ima.values().copied()),
            None => Box::new((0..len as isize).step_by(cache_line_length)),
        };*/
        let nb_cores = core_per_package();

        for i in offsets {
            let pointer = unsafe { p.offset(i) };
            params.address = pointer;

            //let hash = slicing.hash(pointer as usize);
            //let hash = determine_slice(pointer, main_core as u8, nb_cores).unwrap();

            if options.verbosity >= Thresholds {
                print!("Calibration for {:p}", pointer);
                //print!(" (hash: {:x})", hash);
                println!();
            }

            // TODO add some useful impl to CalibrateResults
            let mut calibrate_result = CalibrateResult {
                page: get_vpn(pointer),
                offset: i,
                histogram: Vec::new(),
                median: vec![0; operations.len()],
                min: vec![0; operations.len()],
                max: vec![0; operations.len()],
                count: vec![0; operations.len()],
            };
            calibrate_result.histogram.reserve(operations.len());

            if helper_core != main_core {
                for op in operations {
                    params = main_turn_handle.wait();
                    params.op = op.prepare;
                    let mut hist = vec![0; N];
                    for _ in 0..options.iterations {
                        main_turn_handle.next();
                        params = main_turn_handle.wait();
                        let _time = unsafe { (op.op)(op.t, pointer) };
                    }
                    for _ in 0..options.iterations {
                        main_turn_handle.next();
                        params = main_turn_handle.wait();
                        let time = unsafe { (op.op)(op.t, pointer) };
                        let bucket = min(N - 1, to_bucket(time));
                        hist[bucket] += 1;
                    }
                    calibrate_result.histogram.push(hist);
                }
            } else {
                for op in operations {
                    let mut hist = vec![0; N];
                    for _ in 0..options.iterations {
                        unsafe { (op.prepare)(pointer) };
                        unsafe { arch_x86::_mm_mfence() }; // Test with this ?
                        let _time = unsafe { (op.op)(op.t, pointer) };
                    }
                    for _ in 0..options.iterations {
                        unsafe { (op.prepare)(pointer) };
                        unsafe { arch_x86::_mm_mfence() }; // Test with this ?
                        let time = unsafe { (op.op)(op.t, pointer) };
                        let bucket = min(N - 1, to_bucket(time));
                        hist[bucket] += 1;
                    }
                    calibrate_result.histogram.push(hist);
                }
            }
            let mut sums = vec![0; operations.len()];

            let median_thresholds: Vec<u32> = calibrate_result
                .histogram
                .iter()
                .map(|h| (options.iterations - h[N - 1]) / 2)
                .collect();

            for j in 0..N - 1 {
                if options.verbosity >= RawResult {
                    print!("RESULT:{},{},{:p},", main_core, helper_core, pointer);
                    print!("{:x},", hash);
                    print!("{}", from_bucket(j));
                }
                // ignore the last bucket : spurious context switches etc.
                for op in 0..operations.len() {
                    let hist = &calibrate_result.histogram[op][j];
                    let min = &mut calibrate_result.min[op];
                    let max = &mut calibrate_result.max[op];
                    let med = &mut calibrate_result.median[op];
                    let count = &mut calibrate_result.count[op];
                    let sum = &mut sums[op];
                    if options.verbosity >= RawResult {
                        print!(",{}", hist);
                    }

                    *count += *hist;
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
                        "{}: min {}, median {}, max {}, count {}",
                        op.display_name,
                        calibrate_result.min[j],
                        calibrate_result.median[j],
                        calibrate_result.max[j],
                        calibrate_result.count[j],
                    );
                }
                print!("CSV: {},{},{:p}, ", main_core, helper_core, pointer);
                print!("{:x}, ", hash);
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
            params.stop = true;
            main_turn_handle.next();
            params = main_turn_handle.wait();
            // join thread.
            helper_thread
                .unwrap()
                .join()
                .expect("Failed to join thread");
            // FIXME error handling
        }
    }

    sched_setaffinity(Pid::from_raw(0), &old).unwrap();

    ret
    // return the result
    // TODO
}
*/
fn calibrate_fixed_freq_2_thread_helper(
    turn_handle: Arc<Mutex<TurnHandle<HelperThreadParams>>>,
    helper_core: usize,
) -> Result<(), Error> {
    let mut turn_handle = turn_handle.lock().unwrap();
    // set thread affinity
    let mut core = CpuSet::new();
    match core.set(helper_core) {
        Ok(_) => {}
        Err(e) => {
            let mut params = turn_handle.wait();
            params.stop = true;
            turn_handle.next();
            return Err(e);
        }
    }

    match sched_setaffinity(Pid::from_raw(0), &core) {
        Ok(_) => {}
        Err(e) => {
            let mut params = turn_handle.wait();
            params.stop = true;
            turn_handle.next();
            return Err(e);
        }
    }

    loop {
        // grab lock
        let params = turn_handle.wait();
        if params.stop {
            turn_handle.next();
            return Ok(());
        }
        // get the relevant parameters
        let addr: *const u8 = params.address;
        let op = params.op;
        unsafe { op(addr) };
        // release lock
        turn_handle.next();
    }
}

// ------------------- Analysis ------------------

pub fn calibration_result_to_ASVP<T, Analysis: Fn(CalibrateResult) -> T>(
    results: Vec<CalibrateResult2T>,
    analysis: Analysis,
    slicing: &impl Fn(usize) -> usize,
) -> Result<HashMap<ASVP, T>, nix::Error> {
    let mut analysis_result: HashMap<ASVP, T> = HashMap::new();
    for calibrate_2t_result in results {
        let attacker = calibrate_2t_result.main_core;
        let victim = calibrate_2t_result.helper_core;
        match calibrate_2t_result.res {
            Err(e) => return Err(e),
            Ok(calibrate_1t_results) => {
                for result_1t in calibrate_1t_results {
                    let offset = result_1t.offset;
                    let page = result_1t.page;
                    let addr = page + offset as usize;
                    let slice = slicing(addr as usize);
                    let analysed = analysis(result_1t);
                    let asvp = ASVP {
                        attacker,
                        slice,
                        victim,
                        page,
                    };
                    analysis_result.insert(asvp, analysed);
                }
            }
        }
    }
    Ok(analysis_result)
}
