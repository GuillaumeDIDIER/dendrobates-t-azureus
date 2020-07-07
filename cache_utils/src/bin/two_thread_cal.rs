use cache_utils::calibration::{
    calibrate_fixed_freq_2_thread, flush_and_reload, load_and_flush, only_flush, only_reload,
    reload_and_flush, CalibrateOperation2T, HistParams, Verbosity, CFLUSH_BUCKET_NUMBER,
    CFLUSH_BUCKET_SIZE, CFLUSH_NUM_ITER,
};
use cache_utils::mmap::MMappedMemory;
use cache_utils::{flush, maccess, noop};
use core::sync::atomic::spin_loop_hint;
use core::sync::atomic::{AtomicBool, Ordering};
use nix::sched::{sched_getaffinity, CpuSet};
use nix::unistd::Pid;
use std::sync::Arc;
use std::thread;

/*
fn wait(turn_lock: &AtomicBool, turn: bool) {
    while turn_lock.load(Ordering::Acquire) != turn {
        spin_loop_hint();
    }
    assert_eq!(turn_lock.load(Ordering::Relaxed), turn);
}

fn next(turn_lock: &AtomicBool) {
    turn_lock.fetch_xor(true, Ordering::Release);
}

fn ping(turn_lock: &AtomicBool) {
    wait(turn_lock, false);
    println!("ping");
    next(turn_lock);
}

fn pong_thread(turn_lock: Arc<AtomicBool>, stop: Arc<AtomicBool>) {
    while pong(&turn_lock, &stop) {

    }
}

fn pong(turn_lock: &AtomicBool, stop: &AtomicBool) -> bool {
    wait(turn_lock, true);
    if stop.load(Ordering::Relaxed) {
        return false;
    }
    println!("pong");
    next(turn_lock);
    true
}



fn joke() {
    let turn_counter = Arc::new(AtomicBool::new(false));
    let stop = Arc::new(AtomicBool::new(false));
    let tcc = turn_counter.clone();
    let sc = stop.clone();

    let thread = thread::spawn(|| {
        pong_thread(tcc, sc)
    });

    for _ in 0..10 {
        ping(&turn_counter);
    }
    wait(&turn_counter, false);
    stop.store(true, Ordering::Relaxed);
    next(&turn_counter);
    thread.join().unwrap();
    println!("Okay");
}
*/

use core::arch::x86_64 as arch_x86;

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
    let m = MMappedMemory::new(SIZE);
    let array = m.slice();

    let cache_line_size = 64;

    // Generate core iterator
    let mut core_pairs: Vec<(usize, usize)> = Vec::new();
    let mut i = 1;
    let old = sched_getaffinity(Pid::from_raw(0)).unwrap();
    while i < CpuSet::count() {
        if old.is_set(i).unwrap() {
            core_pairs.push((0, i));
            println!("{},{}", 0, i);
        }
        i = i << 1;
    }
    for i in 1..CpuSet::count() {
        if old.is_set(i).unwrap() {
            core_pairs.push((i, 0));
            println!("{},{}", i, 0);
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
            array.len() as isize,
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
            HistParams {
                bucket_number: CFLUSH_BUCKET_NUMBER,
                bucket_size: CFLUSH_BUCKET_SIZE,
                iterations: CFLUSH_NUM_ITER,
            },
            verbose_level,
        );
    }
}
