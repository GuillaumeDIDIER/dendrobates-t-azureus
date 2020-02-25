use crate::{flush, maccess, rdtsc_fence};
use polling_serial::serial_println;

extern crate alloc;
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

pub fn calibrate_access() -> u64 {
    serial_println!("Calibrating...");

    // Allocate a target array
    // TBD why size, why the position in the array, why the type (usize)
    let mut array = Vec::<usize>::with_capacity(5 << 10);
    array.resize(5 << 10, 1);

    let array = array.into_boxed_slice();

    // Histograms bucket of 5 and max at 400 cycles
    // Magic numbers to be justified
    // 80 is a size of screen
    let mut hit_histogram = vec![0; BUCKET_NUMBER]; //Vec::<u32>::with_capacity(BUCKET_NUMBER);
                                                    //hit_histogram.resize(BUCKET_NUMBER, 0);

    let mut miss_histogram = hit_histogram.clone();

    // the address in memory we are going to target
    let pointer = (&array[2048] as *const usize) as *const u8;

    // do a large sample of accesses to a cached line
    unsafe { maccess(pointer) };
    for _ in 0..(4 << 20) {
        let d = unsafe { only_reload(pointer) } as usize;
        hit_histogram[min(BUCKET_NUMBER - 1, d / BUCKET_SIZE) as usize] += 1;
    }

    // do a large numer of accesses to uncached line
    unsafe { flush(pointer) };
    for _ in 0..(4 << 20) {
        let d = unsafe { flush_and_reload(pointer) } as usize;
        miss_histogram[min(BUCKET_NUMBER - 1, d / BUCKET_SIZE) as usize] += 1;
    }

    let mut hit_max = 0;
    let mut hit_max_i = 0;
    let mut miss_min_i = 0;
    for i in 0..hit_histogram.len() {
        serial_println!(
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
    serial_println!("Miss min {}", miss_min_i * BUCKET_SIZE);
    serial_println!("Max hit {}", hit_max_i * BUCKET_SIZE);

    let mut min = u32::max_value();
    let mut min_i = 0;
    for i in hit_max_i..miss_min_i {
        if min > hit_histogram[i] + miss_histogram[i] {
            min = hit_histogram[i] + miss_histogram[i];
            min_i = i;
        }
    }

    serial_println!("Threshold {}", min_i * BUCKET_SIZE);
    serial_println!("Calibration done.");
    (min_i * BUCKET_SIZE) as u64
}

const CFLUSH_BUCKET_SIZE: usize = 1;
const CFLUSH_BUCKET_NUMBER: usize = 250;

pub fn calibrate_flush() -> u64 {
    serial_println!("Calibrating cflush...");

    // Allocate a target array
    // TBD why size, why the position in the array, why the type (usize)
    let mut array = Vec::<usize>::with_capacity(5 << 10);
    array.resize(5 << 10, 1);

    let array = array.into_boxed_slice();

    // Histograms bucket of 5 and max at 400 cycles
    // Magic numbers to be justified
    // 80 is a size of screen
    let mut hit_histogram = vec![0; CFLUSH_BUCKET_NUMBER];

    let mut miss_histogram = hit_histogram.clone();

    // the address in memory we are going to target
    let pointer = (&array[2048] as *const usize) as *const u8;

    // do a large sample of accesses to a cached line
    for _ in 0..(4 << 20) {
        let d = unsafe { load_and_flush(pointer) } as usize;
        hit_histogram[min(CFLUSH_BUCKET_NUMBER - 1, d / CFLUSH_BUCKET_SIZE) as usize] += 1;
    }

    // do a large numer of accesses to uncached line
    unsafe { flush(pointer) };
    for _ in 0..(4 << 20) {
        let d = unsafe { flush_and_flush(pointer) } as usize;
        miss_histogram[min(CFLUSH_BUCKET_NUMBER - 1, d / CFLUSH_BUCKET_SIZE) as usize] += 1;
    }

    let mut hit_max: (usize, u32) = (0, 0);
    let mut miss_max: (usize, u32) = (0, 0);

    for i in 0..hit_histogram.len() {
        serial_println!(
            "{:3}: {:10} {:10}",
            i * CFLUSH_BUCKET_SIZE,
            hit_histogram[i],
            miss_histogram[i]
        );
        if hit_max.1 < hit_histogram[i] {
            hit_max = (i, hit_histogram[i]);
        }
        if miss_max.1 < miss_histogram[i] {
            miss_max = (i, miss_histogram[i]);
        }
    }
    serial_println!("Miss max {}", miss_max.0 * CFLUSH_BUCKET_SIZE);
    serial_println!("Max hit {}", hit_max.0 * CFLUSH_BUCKET_SIZE);
    let mut threshold: (usize, u32) = (0, u32::max_value());
    for i in miss_max.0..hit_max.0 {
        if hit_histogram[i] + miss_histogram[i] < threshold.1 {
            threshold = (i, hit_histogram[i] + miss_histogram[i]);
        }
    }

    serial_println!("Threshold {}", threshold.0 * CFLUSH_BUCKET_SIZE);
    serial_println!("Calibration done.");
    (threshold.0 * CFLUSH_BUCKET_SIZE) as u64
}
