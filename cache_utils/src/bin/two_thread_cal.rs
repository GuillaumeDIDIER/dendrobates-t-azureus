use cache_utils::calibration::{
    calibrate_fixed_freq_2_thread, flush_and_reload, load_and_flush, only_flush, only_reload,
    reload_and_flush, CalibrateOperation2T, CalibrationOptions, HistParams, Verbosity,
    CFLUSH_BUCKET_NUMBER, CFLUSH_BUCKET_SIZE, CFLUSH_NUM_ITER,
};
use cache_utils::mmap::MMappedMemory;
use cache_utils::{flush, maccess, noop};
use nix::sched::{sched_getaffinity, CpuSet};
use nix::unistd::Pid;

use core::arch::x86_64 as arch_x86;
use std::process::Command;
use std::str::from_utf8;

unsafe fn multiple_access(p: *const u8) {
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

const SIZE: usize = 2 << 20;

fn main() {
    // Grab a slice of memory

    let core_per_socket_out = Command::new("sh")
        .arg("-c")
        .arg("lscpu | grep socket | cut -b 22-")
        .output()
        .expect("Failed to detect cpu count");
    //println!("{:#?}", core_per_socket_str);

    let core_per_socket_str = from_utf8(&core_per_socket_out.stdout).unwrap();

    //println!("Number of cores per socket: {}", cps_str);

    let core_per_socket: i32 = core_per_socket_str[0..(core_per_socket_str.len() - 1)]
        .parse()
        .unwrap_or(0);

    println!("Number of cores per socket: {}", core_per_socket);

    let m = MMappedMemory::new(SIZE);
    let array = m.slice();

    let cache_line_size = 64;

    // Generate core iterator
    let mut core_pairs: Vec<(usize, usize)> = Vec::new();
    let old = sched_getaffinity(Pid::from_raw(0)).unwrap();

    for i in 0..CpuSet::count() {
        for j in 0..CpuSet::count() {
            if old.is_set(i).unwrap() && old.is_set(j).unwrap() {
                core_pairs.push((i, j));
                println!("{},{}", i, j);
            }
        }
    }

    // operations
    // Call calibrate 2T \o/

    let verbose_level = Verbosity::RawResult;

    unsafe {
        let pointer = (&array[0]) as *const u8;

        if pointer as usize & (cache_line_size - 1) != 0 {
            panic!("not aligned nicely");
        }
        calibrate_fixed_freq_2_thread(
            pointer,
            64,
            array.len() as isize >> 3,
            &mut core_pairs.into_iter(),
            &[
                CalibrateOperation2T {
                    prepare: multiple_access,
                    op: only_flush,
                    name: "clflush_remote_hit",
                    display_name: "clflush remote hit",
                },
                CalibrateOperation2T {
                    prepare: multiple_access,
                    op: load_and_flush,
                    name: "clflush_shared_hit",
                    display_name: "clflush shared hit",
                },
                CalibrateOperation2T {
                    prepare: flush,
                    op: only_flush,
                    name: "clflush_miss_f",
                    display_name: "clflush miss - f",
                },
                CalibrateOperation2T {
                    prepare: flush,
                    op: load_and_flush,
                    name: "clflush_local_hit_f",
                    display_name: "clflush local hit - f",
                },
                CalibrateOperation2T {
                    prepare: noop::<u8>,
                    op: only_flush,
                    name: "clflush_miss_n",
                    display_name: "clflush miss - n",
                },
                CalibrateOperation2T {
                    prepare: noop::<u8>,
                    op: load_and_flush,
                    name: "clflush_local_hit_n",
                    display_name: "clflush local hit - n",
                },
                CalibrateOperation2T {
                    prepare: noop::<u8>,
                    op: flush_and_reload,
                    name: "reload_miss",
                    display_name: "reload miss",
                },
                CalibrateOperation2T {
                    prepare: multiple_access,
                    op: reload_and_flush,
                    name: "reload_remote_hit",
                    display_name: "reload remote hit",
                },
                CalibrateOperation2T {
                    prepare: multiple_access,
                    op: only_reload,
                    name: "reload_shared_hit",
                    display_name: "reload shared hit",
                },
                CalibrateOperation2T {
                    prepare: noop::<u8>,
                    op: only_reload,
                    name: "reload_local_hit",
                    display_name: "reload local hit",
                },
            ],
            CalibrationOptions {
                hist_params: HistParams {
                    bucket_number: CFLUSH_BUCKET_NUMBER,
                    bucket_size: CFLUSH_BUCKET_SIZE,
                    iterations: CFLUSH_NUM_ITER,
                },
                verbosity: verbose_level,
                optimised_addresses: true,
            },
        );
    }
}
