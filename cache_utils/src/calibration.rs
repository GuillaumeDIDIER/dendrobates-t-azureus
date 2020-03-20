use crate::{flush, maccess, rdtsc_fence};

#[cfg(feature = "no_std")]
use polling_serial::serial_println as println;

#[derive(Ord, PartialOrd, Eq, PartialEq)]
pub enum Verbosity {
    NoOutput,
    Thresholds,
    RawResult,
    Debug,
}

extern crate alloc;
use crate::calibration::Verbosity::{Debug, NoOutput, RawResult, Thresholds};
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::min;

// calibration, todo
// this will require getting a nice page to do some amusing stuff on it.
// it will have to return some results later.

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

const CFLUSH_NUM_ITER: usize = 1 << 11;
const CFLUSH_SPURIOUS_THRESHOLD: usize = 1;

/* TODO Code cleanup :
  - change type back to a slice OK
  - change return type to return thresholds per cache line ?
  - change iteration to be per cache line OK
  - take the cache line size as a parameter OK
  - parametrize 4k vs 2M ? Or just use the slice length ? OK
*/

pub fn calibrate_flush(
    array: &[u8],
    cache_line_size: usize,
    verbose_level: Verbosity,
) -> Vec<(usize, Vec<(usize, usize)>, usize)> {
    if verbose_level > NoOutput {
        println!("Calibrating cflush...");
    }
    let mut ret = Vec::new();
    // Allocate a target array
    // TBD why size, why the position in the array, why the type (usize)
    //let mut array = Vec::<usize>::with_capacity(5 << 10);
    //array.resize(5 << 10, 1);

    //let array = array.into_boxed_slice();

    // Histograms bucket of 5 and max at 400 cycles
    // Magic numbers to be justified
    // 80 is a size of screen

    // the address in memory we are going to target
    let pointer = (&array[0]) as *const u8;

    if pointer as usize & (cache_line_size - 1) != 0 {
        panic!("not aligned nicely");
    }
    // do a large sample of accesses to a cached line
    for i in (0..(array.len() as isize)).step_by(cache_line_size) {
        let mut hit_histogram = vec![0; CFLUSH_BUCKET_NUMBER];

        let mut miss_histogram = hit_histogram.clone();
        if verbose_level >= Thresholds {
            println!("Calibration for {:p}", unsafe { pointer.offset(i) });
        }
        unsafe { load_and_flush(pointer.offset(i)) }; // align down on 64 bytes
        for _ in 1..CFLUSH_NUM_ITER {
            let d = unsafe { load_and_flush(pointer.offset(i)) } as usize;
            hit_histogram[min(CFLUSH_BUCKET_NUMBER - 1, d / CFLUSH_BUCKET_SIZE) as usize] += 1;
        }

        // do a large numer of accesses to uncached line
        unsafe { flush(pointer.offset(i)) };

        unsafe { load_and_flush(pointer.offset(i)) };
        for _ in 0..CFLUSH_NUM_ITER {
            let d = unsafe { flush_and_flush(pointer.offset(i)) } as usize;
            miss_histogram[min(CFLUSH_BUCKET_NUMBER - 1, d / CFLUSH_BUCKET_SIZE) as usize] += 1;
        }

        // extract min, max, & median of the distribution.
        // set the threshold to mid point between miss max & hit min.

        // determine :
        // Hit min, max, median
        // Miss min, miss max, median
        // If there is no overlap the threshold is trivial
        // If there is Grab the point where the ratio is balanced

        let mut hit_min = 0;
        let mut hit_max = 0;
        let mut miss_min = 0;
        let mut miss_max = 0;
        let mut miss_med = 0;
        let mut hit_med = 0;
        let mut hit_sum = 0;
        let mut miss_sum = 0;

        //let mut hit_max: (usize, u32) = (0, 0);
        //let mut miss_max: (usize, u32) = (0, 0);

        for i in 0..(hit_histogram.len() - 1) {
            // ignore the last bucket, spurious context switches
            if verbose_level >= RawResult {
                println!(
                    "{:3}: {:10} {:10}",
                    i * CFLUSH_BUCKET_SIZE,
                    hit_histogram[i],
                    miss_histogram[i]
                );
            }

            for (min, max, med, sum, hist) in &mut [
                (
                    &mut hit_min,
                    &mut hit_max,
                    &mut hit_med,
                    &mut hit_sum,
                    &hit_histogram,
                ),
                (
                    &mut miss_min,
                    &mut miss_max,
                    &mut miss_med,
                    &mut miss_sum,
                    &miss_histogram,
                ),
            ] {
                if **min == 0 {
                    // looking for min
                    if hist[i] > CFLUSH_SPURIOUS_THRESHOLD {
                        **min = i;
                    }
                } else {
                    // min found, looking for max
                    if hist[i] > CFLUSH_SPURIOUS_THRESHOLD {
                        **max = i;
                    }
                }

                if **med == 0 {
                    **sum += hist[i];
                    if **sum >= (CFLUSH_NUM_ITER - hist[hist.len() - 1]) / 2 {
                        **med = i;
                    }
                }
            }
            if verbose_level >= Debug {
                println!("sum hit {} miss {}", hit_sum, miss_sum);
            }
        }

        if verbose_level >= Thresholds {
            println!("Hits: min {} max {} med {}", hit_min, hit_max, hit_med);
            println!("Miss: min {} max {} med {}", miss_min, miss_max, miss_med);
        }
        //println!("Miss max {}", miss_max.0 * CFLUSH_BUCKET_SIZE);
        //println!("Max hit {}", hit_max.0 * CFLUSH_BUCKET_SIZE);
        let mut threshold: (usize, u32) = (0, u32::max_value());
        /*for i in miss_max.0..hit_max.0 {
            if hit_histogram[i] + miss_histogram[i] < threshold.1 {
                threshold = (i, hit_histogram[i] + miss_histogram[i]);
            }
        }*/
        if verbose_level > NoOutput {
            println!("Threshold {}", threshold.0 * CFLUSH_BUCKET_SIZE);
            println!("Calibration done.");
        }

        ret.push((
            i as usize,
            hit_histogram
                .iter()
                .zip(&miss_histogram)
                .map(|(&x, &y)| (x, y))
                .collect(),
            threshold.0,
        ));
    }
    ret
}
