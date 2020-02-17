use crate::{flush, maccess, rdtsc_fence, rdtsc_nofence};
use polling_serial::{serial_print, serial_println};
use vga_buffer::println;

extern crate alloc;
use alloc::vec::Vec;
use core::cmp::min;

// calibration, todo
// this will require getting a nice page to do some amusing stuff on it.
// it will have to return some results later.

unsafe fn only_reload(p: *const u8) -> u64 {
    let t = rdtsc_fence();
    maccess(p);
    let d = rdtsc_fence() - t;
    d
}

unsafe fn flush_and_reload(p: *const u8) -> u64 {
    flush(p);
    let t = rdtsc_fence();
    maccess(p);
    let d = rdtsc_fence() - t;
    d
}

pub fn calibrate_access() -> u64 {
    serial_println!("Calibrating...");

    let mut array = Vec::<usize>::with_capacity(5 << 10);
    array.resize(5 << 10, 1);

    let array = array.into_boxed_slice();

    let mut hit_histogram = Vec::<u32>::with_capacity(80);
    hit_histogram.resize(80, 0);

    let mut miss_histogram = hit_histogram.clone();

    let pointer = (&array[2048] as *const usize) as *const u8;
    // sanity check
    println!(
        "&array[0]: {:p}, array[2048]{:p}",
        (&array[0] as *const usize) as *const u8,
        (&array[2048] as *const usize) as *const u8
    );

    unsafe { maccess(pointer) };
    for _ in 0..(4 << 20) {
        let d = unsafe { only_reload(pointer) };
        hit_histogram[min(79, d / 5) as usize] += 1;
    }

    unsafe { flush(pointer) };
    for _ in 0..(4 << 20) {
        let d = unsafe { flush_and_reload(pointer) };
        miss_histogram[min(79, d / 5) as usize] += 1;
    }

    // Todo plot and analyze histogram

    serial_println!("Threshold {}", -1);
    serial_println!("Calibration done.");
    (-1_i64) as u64
}

pub fn calibrate_flush() -> u64 {
    (-1_i64) as u64
}
