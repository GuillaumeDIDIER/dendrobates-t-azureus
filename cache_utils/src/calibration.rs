#![allow(clippy::missing_safety_doc)]
#![cfg(target_arch = "x86_64")]

extern crate alloc; //#[cfg(feature = "serde_support")]
                    //use serde::{Deserialize, Serialize};
#[cfg(feature = "use_std")]
extern crate std;
use crate::{flush, maccess, rdtsc_fence};
use cpuid::complex_addressing::{cache_slicing, CacheAttackSlicing, CacheSlicing};

use core::cmp::min;
use cpuid::MicroArchitecture;

use core::arch::x86_64 as arch_x86;
#[cfg(feature = "no_std")]
use polling_serial::{serial_print as print, serial_println as println};

#[cfg(feature = "use_std")]
pub use crate::calibrate_2t::*;

use crate::calibration::Verbosity::*;
use alloc::vec;
use alloc::vec::Vec;
use calibration_results::calibration::{Slice, VPN};
use calibration_results::classifiers::ErrorPredictor;
use core::hash::{Hash, Hasher};
use core::ops::{Add, AddAssign};
#[cfg(all(feature = "no_std", not(feature = "use_std")))]
pub use hashbrown::HashMap;
use itertools::Itertools;
#[cfg(feature = "use_std")]
pub use std::collections::HashMap;

#[derive(Ord, PartialOrd, Eq, PartialEq)]
pub enum Verbosity {
    NoOutput,
    Thresholds,
    RawResult,
    Debug,
}

pub struct CalibrationOptions {
    pub iterations: u32,
    pub verbosity: Verbosity,
    pub optimised_addresses: bool,
    pub measure_hash: bool,
    pub warmup_iterations: u32,
}

pub const CALIBRATION_WARMUP_ITER: u32 = 128;

impl CalibrationOptions {
    pub fn new(iterations: u32, verbosity: Verbosity) -> CalibrationOptions {
        CalibrationOptions {
            iterations,
            verbosity,
            optimised_addresses: false,
            measure_hash: false,
            warmup_iterations: CALIBRATION_WARMUP_ITER,
        }
    }
}

pub unsafe fn only_reload(p: *const u8) -> u64 {
    let t = unsafe { rdtsc_fence() };
    unsafe { maccess(p) };
    let r = unsafe { rdtsc_fence() } - t;
    r
}

pub unsafe fn flush_and_reload(p: *const u8) -> u64 {
    unsafe {
        flush(p);
        only_reload(p)
    }
}

pub unsafe fn reload_and_flush(p: *const u8) -> u64 {
    let r = unsafe { only_reload(p) };
    unsafe { flush(p) };
    r
}

pub unsafe fn only_flush(p: *const u8) -> u64 {
    let t = unsafe { rdtsc_fence() };
    unsafe { flush(p) };
    let r = unsafe { rdtsc_fence() } - t;
    r
}

pub unsafe fn load_and_flush(p: *const u8) -> u64 {
    unsafe {
        maccess(p);
        only_flush(p)
    }
}

pub unsafe fn flush_and_flush(p: *const u8) -> u64 {
    unsafe {
        flush(p);
        only_flush(p)
    }
}

pub unsafe fn l3_and_reload(p: *const u8) -> u64 {
    unsafe {
        flush(p);
        arch_x86::_mm_mfence();
        arch_x86::_mm_prefetch::<{ arch_x86::_MM_HINT_T2 }>(p as *const i8);
        arch_x86::__cpuid_count(0, 0);
        only_reload(p)
    }
}

pub const PAGE_SHIFT: usize = 12;
pub const PAGE_LEN: usize = 1 << PAGE_SHIFT;

pub fn get_vpn<T>(p: *const T) -> usize {
    (p as usize) & (!(PAGE_LEN - 1)) // FIXME
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
            hit_histogram[min(BUCKET_NUMBER - 1, d / BUCKET_SIZE)] += 1;
        }
    }

    // do a large numer of accesses to uncached line
    unsafe { flush(pointer) };
    for i in 0..(4 << 10) {
        for _ in 0..(1 << 10) {
            let d = unsafe { flush_and_reload(pointer.offset(i & (!0x3f))) } as usize;
            miss_histogram[min(BUCKET_NUMBER - 1, d / BUCKET_SIZE)] += 1;
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

    let mut min = u32::MAX;
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

pub const CLFLUSH_BUCKET_SIZE: u64 = 1;
pub const CLFLUSH_BUCKET_NUMBER: usize = 1500;

pub const CLFLUSH_NUM_ITER: u32 = 1 << 10;
pub const CLFLUSH_NUM_ITERATION_AV: u32 = 1 << 8;

/*<const WIDTH: u64, const N: usize>*/
pub fn calibrate_flush(
    array: &[u8],
    cache_line_size: usize,
    verbose_level: Verbosity,
) -> Vec<CalibrateResult> {
    let pointer = (&array[0]) as *const u8;

    if pointer as usize & (cache_line_size - 1) != 0 {
        panic!("not aligned nicely");
    }

    calibrate_impl_fixed_freq::<CLFLUSH_BUCKET_SIZE, CLFLUSH_BUCKET_NUMBER>(
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
        CLFLUSH_NUM_ITER,
        verbose_level,
    )
}

#[derive(Debug)]
pub struct CalibrateResult {
    pub page: VPN,
    pub offset: isize,
    pub histogram: Vec<Vec<u32>>,
    pub median: Vec<u64>,
    pub min: Vec<u64>,
    pub max: Vec<u64>,
    pub count: Vec<u32>,
}

pub struct CalibrateOperation<'a> {
    pub op: unsafe fn(*const u8) -> u64,
    pub name: &'a str,
    pub display_name: &'a str,
}

pub unsafe fn calibrate<const WIDTH: u64, const N: usize>(
    p: *const u8,
    increment: usize,
    len: isize,
    operations: &[CalibrateOperation],
    num_iterations: u32,
    verbosity_level: Verbosity,
) -> Vec<CalibrateResult> {
    calibrate_impl_fixed_freq::<WIDTH, N>(
        p,
        increment,
        len,
        operations,
        num_iterations,
        verbosity_level,
    )
}

pub const SPURIOUS_THRESHOLD: u32 = 1;
fn calibrate_impl_fixed_freq<const WIDTH: u64, const N: usize>(
    p: *const u8,
    increment: usize,
    len: isize,
    operations: &[CalibrateOperation],
    iterations: u32,
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

    let to_bucket = |time: u64| -> usize { (time / WIDTH) as usize };
    let from_bucket = |bucket: usize| -> u64 { (bucket as u64) * WIDTH };

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

    let h = if let Some(s) = slicing.as_ref() {
        if s.can_hash() {
            Some(|addr: usize| -> u8 { slicing.as_ref().unwrap().hash(addr).unwrap() })
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
            page: get_vpn(pointer),
            offset: i,
            histogram: Vec::new(),
            median: vec![0; operations.len()],
            min: vec![0; operations.len()],
            max: vec![0; operations.len()],
            count: vec![0; operations.len()],
        };
        calibrate_result.histogram.reserve(operations.len());

        for op in operations {
            let mut hist = vec![0; N];
            for _ in 0..iterations {
                let time = unsafe { (op.op)(pointer) };
                let bucket = min(N - 1, to_bucket(time));
                hist[bucket] += 1;
            }
            calibrate_result.histogram.push(hist);
        }

        let mut sums = vec![0; operations.len()];

        let median_thresholds: Vec<u32> = calibrate_result
            .histogram
            .iter()
            .map(|h| (iterations - h[N - 1]) / 2)
            .collect();

        for j in 0..N - 1 {
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
                let count = &mut calibrate_result.count[op];
                let sum = &mut sums[op];
                if verbosity_level >= RawResult {
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

fn get_cache_slicing(core_per_socket: u8) -> Option<CacheSlicing> {
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

pub fn get_cache_attack_slicing(
    core_per_socket: u8,
    cache_line_length: usize,
) -> Option<CacheAttackSlicing> {
    if let Some(uarch) = MicroArchitecture::get_micro_architecture() {
        if let Some(vendor_family_model_stepping) = MicroArchitecture::get_family_model_stepping() {
            Some(CacheAttackSlicing::from(
                cache_slicing(
                    uarch,
                    core_per_socket,
                    vendor_family_model_stepping.0,
                    vendor_family_model_stepping.1,
                    vendor_family_model_stepping.2,
                ),
                cache_line_length,
            )) // FIXME Cache length magic number
        } else {
            Some(CacheAttackSlicing::Unsupported(!(cache_line_length - 1))) //,None
        }
    } else {
        Some(CacheAttackSlicing::Unsupported(!(cache_line_length - 1))) //None
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

    let r = calibrate_impl_fixed_freq::<2, 512>(
        pointer,
        cache_line_size,
        array.len() as isize,
        &[CalibrateOperation {
            op: l3_and_reload,
            name: "l3_hit",
            display_name: "L3 hit",
        }],
        1 << 11,
        verbose_level,
    );

    r.into_iter().next().unwrap()
}

/*
   ASVP trait ?
   Easily put any combination, use None to signal Any possible value, Some to signal fixed value.
*/

/*
// Broken due to lifetime issues
trait PartialLocationProjection: Sized {
    fn try_project<'s, 'a>(&'s self, params: &'a LocationParameters) -> Option<Self<'a>>;
    fn project<'s, 'a>(&'s self, params: &'a LocationParameters) -> Self<'a> {
        self.try_project(params).expect("Impossible projection")
    }
}*/

/* #[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Default)]
pub struct MemoryLocation {
    pub slice: u8,
    pub numa_node: u8,
    pub vpn: usize,
} */

#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Default)]
pub struct ASVP {
    pub attacker: usize,
    pub slice: Slice,
    pub victim: usize,
    pub page: VPN,
}

#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Default)]
pub struct CSP {
    pub core: usize,
    pub slice: Slice,
    pub page: VPN,
}

#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Default)]
pub struct ASP {
    pub attacker: usize,
    pub slice: Slice,
    pub page: VPN,
}

#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Default)]
pub struct SVP {
    pub slice: Slice,
    pub victim: usize,
    pub page: VPN,
}

#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Default)]
pub struct SP {
    pub slice: Slice,
    pub page: VPN,
}

#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Default)]
pub struct AV {
    pub attacker: usize,
    pub victim: usize,
}
/*

#[derive(Debug, Clone)]
pub struct HitMissRawHistogram {
    pub hit: Vec<u32>,
    pub miss: Vec<u32>,
}

// ALL Histogram deal in buckets : FIXME we should clearly distinguish bucket vs time.
// Thresholds are less than equal.
impl HitMissRawHistogram {
    // This might be inefficient.
    pub fn from(
        mut calibrate_result: CalibrateResult,
        hit_index: usize,
        miss_index: usize,
    ) -> Self {
        calibrate_result.histogram.push(Vec::default());
        let hit = calibrate_result.histogram.swap_remove(hit_index);
        calibrate_result.histogram.push(Vec::default());
        let miss = calibrate_result.histogram.swap_remove(miss_index);
        HitMissRawHistogram { hit, miss }
    }

    pub fn empty(len: usize) -> Self {
        Self {
            hit: vec![0; len],
            miss: vec![0; len],
        }
    }
}

// Addition logic

// Tough case, both references.

impl Add for &HitMissRawHistogram {
    type Output = HitMissRawHistogram;

    fn add(self, rhs: &HitMissRawHistogram) -> Self::Output {
        assert_eq!(self.hit.len(), rhs.hit.len());
        assert_eq!(self.miss.len(), rhs.miss.len());
        assert_eq!(self.hit.len(), self.miss.len());
        let len = self.hit.len();
        let mut r = HitMissRawHistogram {
            hit: vec![0; len],
            miss: vec![0; len],
        };
        for i in 0..len {
            r.hit[i] = self.hit[i] + rhs.hit[i];
            r.miss[i] = self.miss[i] + rhs.miss[i];
        }
        r
    }
}

// most common case re-use of self is possible. (Or a reduction to such a case)

impl AddAssign<&HitMissRawHistogram> for HitMissRawHistogram {
    //type Rhs = &RawHistogram;
    fn add_assign(&mut self, rhs: &Self) {
        assert_eq!(self.hit.len(), rhs.hit.len());
        assert_eq!(self.miss.len(), rhs.miss.len());
        assert_eq!(self.hit.len(), self.miss.len());

        for i in 0..self.hit.len() {
            self.hit[i] += rhs.hit[i];
            self.miss[i] += rhs.miss[i];
        }
    }
}

// Fallback to most common case

impl Add for HitMissRawHistogram {
    type Output = HitMissRawHistogram;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<&Self> for HitMissRawHistogram {
    type Output = HitMissRawHistogram;

    fn add(mut self, rhs: &Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<HitMissRawHistogram> for &HitMissRawHistogram {
    type Output = HitMissRawHistogram;

    fn add(self, mut rhs: HitMissRawHistogram) -> Self::Output {
        rhs += self;
        rhs
    }
}

impl AddAssign<Self> for HitMissRawHistogram {
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}
*/

/*
// TODO Refactor, this histogram should not know about how classifier work, and classifier should be the one exploiting it.
// FIXME Histograms are the one responsible for time <-> bucket mapping decisions too.
impl HistogramCumSum {
    pub fn from(raw_histogram: HitMissRawHistogram) -> Self {
        let len = raw_histogram.miss.len();

        assert_eq!(raw_histogram.hit.len(), len);

        // Cum Sums
        let miss_cum_sum = cum_sum(&raw_histogram.miss);
        let hit_cum_sum = cum_sum(&raw_histogram.hit);
        let miss_total = miss_cum_sum[len - 1];
        let hit_total = hit_cum_sum[len - 1];
        Self {
            num_hit: hit_total,
            num_miss: miss_total,
            hit: raw_histogram.hit,
            miss: raw_histogram.miss,
            hit_cum_sum,
            miss_cum_sum,
        }
    }

    pub fn from_calibrate(
        calibrate_result: CalibrateResult,
        hit_index: usize,
        miss_index: usize,
    ) -> Self {
        Self::from(HitMissRawHistogram::from(calibrate_result, hit_index, miss_index))
    }

    pub fn error_for_threshold(&self, threshold: Threshold) -> ErrorPrediction {
        if threshold.miss_faster_than_hit {
            ErrorPrediction {
                true_hit: self.num_hit - self.hit_cum_sum[threshold.bucket_index],
                true_miss: self.miss_cum_sum[threshold.bucket_index],
                false_hit: self.num_miss - self.miss_cum_sum[threshold.bucket_index],
                false_miss: self.hit_cum_sum[threshold.bucket_index],
            }
        } else {
            ErrorPrediction {
                true_hit: self.hit_cum_sum[threshold.bucket_index],
                true_miss: self.num_miss - self.miss_cum_sum[threshold.bucket_index],
                false_hit: self.miss_cum_sum[threshold.bucket_index],
                false_miss: self.num_hit - self.hit_cum_sum[threshold.bucket_index],
            }
        }
    }

    pub fn len(&self) -> usize {
        self.hit.len()
    }

    pub fn empty(len: usize) -> Self {
        Self {
            num_hit: 0,
            num_miss: 0,
            hit: vec![0; len],
            miss: vec![0; len],
            hit_cum_sum: vec![0; len],
            miss_cum_sum: vec![0; len],
        }
    }
}

// Addition logic

// Tough case, both references.

impl Add for &HistogramCumSum {
    type Output = HistogramCumSum;

    fn add(self, rhs: &HistogramCumSum) -> Self::Output {
        assert_eq!(self.hit.len(), self.miss.len());
        assert_eq!(self.hit.len(), self.hit_cum_sum.len());
        assert_eq!(self.hit.len(), self.miss_cum_sum.len());
        assert_eq!(self.hit.len(), rhs.hit.len());
        assert_eq!(self.hit.len(), rhs.miss.len());
        assert_eq!(self.hit.len(), rhs.hit_cum_sum.len());
        assert_eq!(self.hit.len(), rhs.miss_cum_sum.len());
        let len = self.len();
        let mut r = HistogramCumSum {
            num_hit: self.num_hit + rhs.num_hit,
            num_miss: self.num_miss + rhs.num_miss,
            hit: vec![0; len],
            miss: vec![0; len],
            hit_cum_sum: vec![0; len],
            miss_cum_sum: vec![0; len],
        };
        for i in 0..len {
            r.hit[i] = self.hit[i] + rhs.hit[i];
            r.miss[i] = self.miss[i] + rhs.miss[i];
            r.hit_cum_sum[i] = self.hit_cum_sum[i] + rhs.hit_cum_sum[i];
            r.miss_cum_sum[i] = self.miss_cum_sum[i] + rhs.miss_cum_sum[i];
        }
        r
    }
}

// most common case re-use of self is possible. (Or a reduction to such a case)

impl AddAssign<&Self> for HistogramCumSum {
    fn add_assign(&mut self, rhs: &Self) {
        assert_eq!(self.hit.len(), self.miss.len());
        assert_eq!(self.hit.len(), self.hit_cum_sum.len());
        assert_eq!(self.hit.len(), self.miss_cum_sum.len());
        assert_eq!(self.hit.len(), rhs.hit.len());
        assert_eq!(self.hit.len(), rhs.miss.len());
        assert_eq!(self.hit.len(), rhs.hit_cum_sum.len());
        assert_eq!(self.hit.len(), rhs.miss_cum_sum.len());
        self.num_hit += rhs.num_hit;
        self.num_miss += rhs.num_miss;
        let len = self.len();
        for i in 0..len {
            self.hit[i] += rhs.hit[i];
            self.miss[i] += rhs.miss[i];
            self.hit_cum_sum[i] += rhs.hit_cum_sum[i];
            self.miss_cum_sum[i] += rhs.miss_cum_sum[i];
        }
    }
}

// Fallback to most common case

impl Add for HistogramCumSum {
    type Output = HistogramCumSum;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<&Self> for HistogramCumSum {
    type Output = HistogramCumSum;

    fn add(mut self, rhs: &Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<HistogramCumSum> for &HistogramCumSum {
    type Output = HistogramCumSum;

    fn add(self, mut rhs: HistogramCumSum) -> Self::Output {
        rhs += self;
        rhs
    }
}

impl AddAssign<Self> for HistogramCumSum {
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}
*/

// most common case re-use of self is possible. (Or a reduction to such a case)

// Fallback to most common case

/*
#[derive(Debug, Clone)]
pub struct ErrorPredictions {
    pub histogram: HistogramCumSum,
    pub error_miss_less_than_hit: Vec<u32>,
    pub error_hit_less_than_miss: Vec<u32>,
}

impl ErrorPredictions {
    // BUGGY TODO
    pub fn predict_errors(hist: HistogramCumSum) -> Self {
        let mut error_miss_less_than_hit = vec![0; hist.len() - 1];
        let mut error_hit_less_than_miss = vec![0; hist.len() - 1];
        for threshold_bucket_index in 0..(hist.len() - 1) {
            error_miss_less_than_hit[threshold_bucket_index] = hist
                .error_for_threshold(Threshold {
                    bucket_index: threshold_bucket_index,
                    miss_faster_than_hit: true,
                })
                .total_error();

            error_hit_less_than_miss[threshold_bucket_index] = hist
                .error_for_threshold(Threshold {
                    bucket_index: threshold_bucket_index,
                    miss_faster_than_hit: false,
                })
                .total_error();
        }
        Self {
            histogram: hist,
            error_miss_less_than_hit,
            error_hit_less_than_miss,
        }
    }

    pub fn empty(len: usize) -> Self {
        Self::predict_errors(HistogramCumSum::empty(len))
    }

    pub fn debug(&self) {
        println!("Debug:HEADER TBD");
        for i in 0..(self.histogram.len() - 1) {
            println!(
                "Debug:{:5},{:5},{:6},{:6},{:6}, {:6}",
                self.histogram.hit[i],
                self.histogram.miss[i],
                self.histogram.hit_cum_sum[i],
                self.histogram.miss_cum_sum[i],
                self.error_miss_less_than_hit[i],
                self.error_hit_less_than_miss[i]
            );
        }
        let i = self.histogram.len() - 1;
        println!(
            "Debug:{:5},{:5},{:6},{:6}",
            self.histogram.hit[i],
            self.histogram.miss[i],
            self.histogram.hit_cum_sum[i],
            self.histogram.miss_cum_sum[i]
        );
    }
}

// Addition logic

// Tough case, both references.

impl Add for &ErrorPredictions {
    type Output = ErrorPredictions;

    fn add(self, rhs: &ErrorPredictions) -> Self::Output {
        assert_eq!(
            self.error_hit_less_than_miss.len(),
            rhs.error_hit_less_than_miss.len()
        );
        assert_eq!(
            self.error_hit_less_than_miss.len(),
            self.error_miss_less_than_hit.len()
        );
        assert_eq!(
            self.error_miss_less_than_hit.len(),
            rhs.error_miss_less_than_hit.len()
        );
        let len = self.error_miss_less_than_hit.len();
        let mut r = ErrorPredictions {
            histogram: &self.histogram + &rhs.histogram,
            error_miss_less_than_hit: vec![0; len],
            error_hit_less_than_miss: vec![0; len],
        };
        for i in 0..len {
            r.error_miss_less_than_hit[i] =
                self.error_miss_less_than_hit[i] + rhs.error_miss_less_than_hit[i];
            r.error_hit_less_than_miss[i] =
                self.error_hit_less_than_miss[i] + rhs.error_hit_less_than_miss[i];
        }
        r
    }
}

// most common case re-use of self is possible. (Or a reduction to such a case)

impl AddAssign<&Self> for ErrorPredictions {
    fn add_assign(&mut self, rhs: &Self) {
        assert_eq!(
            self.error_hit_less_than_miss.len(),
            rhs.error_hit_less_than_miss.len()
        );
        assert_eq!(
            self.error_hit_less_than_miss.len(),
            self.error_miss_less_than_hit.len()
        );
        assert_eq!(
            self.error_miss_less_than_hit.len(),
            rhs.error_miss_less_than_hit.len()
        );
        self.histogram += &rhs.histogram;
        for i in 0..self.error_hit_less_than_miss.len() {
            self.error_hit_less_than_miss[i] += rhs.error_hit_less_than_miss[i];
            self.error_miss_less_than_hit[i] += rhs.error_miss_less_than_hit[i];
        }
    }
}

// Fallback to most common case

impl Add for ErrorPredictions {
    type Output = ErrorPredictions;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<&Self> for ErrorPredictions {
    type Output = ErrorPredictions;

    fn add(mut self, rhs: &Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<ErrorPredictions> for &ErrorPredictions {
    type Output = ErrorPredictions;

    fn add(self, mut rhs: ErrorPredictions) -> Self::Output {
        rhs += self;
        rhs
    }
}

impl AddAssign<Self> for ErrorPredictions {
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}
*/

/*
#[derive(Debug, Clone)]
pub struct PotentialClassifiers<const WIDTH: u64, const N: usize> (Vec<(Box<dyn ErrorPredictor<WIDTH, N>>, ErrorPrediction)>);


impl<const WIDTH: u64, const N: usize> PotentialClassifiers<WIDTH, N> {
    pub fn median(mut self) -> Option<(Box<dyn ErrorPredictor<WIDTH, N>>, ErrorPrediction)> {
        if self.threshold_errors.len() > 0 {
            let index = (self.threshold_errors.len() - 1) / 2;
            self.threshold_errors.push(Default::default());
            Some(self.threshold_errors.swap_remove(index))
        } else {
            None
        }
    }

    pub fn minimizing_total_error(error_pred: ErrorPredictions) -> Self {
        let mut min_error = u32::max_value();
        let mut threshold_errors = Vec::new();
        for i in 0..error_pred.error_miss_less_than_hit.len() {
            if error_pred.error_miss_less_than_hit[i] < min_error {
                min_error = error_pred.error_miss_less_than_hit[i];
                threshold_errors = Vec::new();
            }
            if error_pred.error_hit_less_than_miss[i] < min_error {
                min_error = error_pred.error_hit_less_than_miss[i];
                threshold_errors = Vec::new();
            }
            if error_pred.error_miss_less_than_hit[i] == min_error {
                let threshold = Threshold {
                    bucket_index: i,
                    miss_faster_than_hit: true,
                };
                let error = error_pred.histogram.error_for_threshold(threshold);
                threshold_errors.push(ThresholdError { threshold, error })
            }
            if error_pred.error_hit_less_than_miss[i] == min_error {
                let threshold = Threshold {
                    bucket_index: i,
                    miss_faster_than_hit: false,
                };
                let error = error_pred.histogram.error_for_threshold(threshold);
                threshold_errors.push(ThresholdError { threshold, error })
            }
        }
        Self { threshold_errors }
    }
}
*/

/*
pub fn compute_threshold_error() -> (Threshold, ()) {
    unimplemented!();
} // TODO
*/

#[cfg(test)]
mod tests {
    use calibration_results::map_values;
    #[cfg(all(feature = "no_std", not(feature = "use_std")))]
    use hashbrown::HashMap;
    #[cfg(feature = "use_std")]
    extern crate std;
    #[cfg(feature = "use_std")]
    use std::collections::HashMap;

    #[test]
    fn test_map_values() {
        let mut input = HashMap::new();
        input.insert(0, "a");
        input.insert(1, "b");
        let output = map_values(input, |c| c.to_uppercase());
        assert_eq!(output[&0], "A");
        assert_eq!(output[&1], "B");
    }
}
