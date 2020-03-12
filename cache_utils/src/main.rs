// TODO create a nice program that can run on a system and will do the calibration.
// Calibration has to be sequential
// Will pin on each core one after the other

//fn execute_on_core(FnOnce)

#![feature(vec_resize_default)]

use cache_utils::calibration::calibrate_flush;
use cache_utils::calibration::Verbosity;

use nix::errno::Errno;
use nix::sched::{sched_getaffinity, sched_setaffinity, CpuSet};
use nix::unistd::Pid;
use nix::Error::Sys;

#[repr(align(4096))]
struct Page {
    pub mem: [u8; 4096],
}

pub fn main() {
    println!("Hello World!");

    let p = Box::new(Page { mem: [0; 4096] });

    let m: &[u8] = &p.mem;

    eprintln!("Count: {}", CpuSet::count());

    let old = sched_getaffinity(Pid::from_raw(0)).unwrap();

    eprintln!("old: {:?}", old);

    for i in 0..(CpuSet::count() - 1) {
        if old.is_set(i).unwrap() {
            println!("Iteration {}...", i);
            let mut core = CpuSet::new();
            core.set(i).unwrap();

            match sched_setaffinity(Pid::from_raw(0), &core) {
                Ok(()) => {
                    calibrate_flush(m, 64, Verbosity::Thresholds);
                    sched_setaffinity(Pid::from_raw(0), &old).unwrap();
                    println!("Iteration {}...ok ", i);
                }
                Err(Sys(Errno::EINVAL)) => {
                    println!("skipping");
                    continue;
                }
                Err(_) => {
                    panic!("Unexpected error while setting affinity");
                }
            }
        }
    }

    // Let's grab all the list of CPUS
    // Then iterate the calibration on each CPU core.
}
