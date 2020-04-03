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

use cache_utils::mmap::MMappedMemory;

/* from linux kernel headers.
#define HUGETLB_FLAG_ENCODE_SHIFT       26
#define HUGETLB_FLAG_ENCODE_MASK        0x3f

#define HUGETLB_FLAG_ENCODE_64KB        (16 << HUGETLB_FLAG_ENCODE_SHIFT)
#define HUGETLB_FLAG_ENCODE_512KB       (19 << HUGETLB_FLAG_ENCODE_SHIFT)
#define HUGETLB_FLAG_ENCODE_1MB         (20 << HUGETLB_FLAG_ENCODE_SHIFT)
#define HUGETLB_FLAG_ENCODE_2MB         (21 << HUGETLB_FLAG_ENCODE_SHIFT)
*/

const SIZE: usize = 2 << 20;
/*
#[repr(align(4096))]
struct Page {
    pub mem: [u8; 4096],
}
*/
pub fn main() {
    /*let array: &[u8] = unsafe {
        let p: *mut u8 = mman::mmap(
            null_mut(),
            SIZE,
            mman::ProtFlags::PROT_READ | mman::ProtFlags::PROT_WRITE,
            mman::MapFlags::MAP_PRIVATE
                | mman::MapFlags::MAP_ANONYMOUS
                | mman::MapFlags::MAP_HUGETLB,
            -1,
            0,
        )
        .unwrap() as *mut u8;
        /*addr: *mut c_void,
        length: size_t,
        prot: ProtFlags,
        flags: MapFlags,
        fd: RawFd,
        offset: off_t*/

        &*slice_from_raw_parts(p, SIZE)
    };*/
    let m = unsafe { MMappedMemory::new(SIZE) };
    let array = m.slice();
    /*
        let p = Box::new(Page { mem: [0; 4096] });

        let m: &[u8] = &p.mem;
    */
    let old = sched_getaffinity(Pid::from_raw(0)).unwrap();

    for i in 0..(CpuSet::count() - 1) {
        if old.is_set(i).unwrap() {
            println!("Iteration {}...", i);
            let mut core = CpuSet::new();
            core.set(i).unwrap();

            match sched_setaffinity(Pid::from_raw(0), &core) {
                Ok(()) => {
                    calibrate_flush(array, 64, Verbosity::NoOutput);
                    calibrate_flush(array, 64, Verbosity::Thresholds);
                    sched_setaffinity(Pid::from_raw(0), &old).unwrap();
                    println!("Iteration {}...ok ", i);
                    eprintln!("Iteration {}...ok ", i);
                }
                Err(Sys(Errno::EINVAL)) => {
                    println!("skipping");
                    continue;
                }
                Err(e) => {
                    panic!("Unexpected error while setting affinity: {}", e);
                }
            }
        }
    }

    // Let's grab all the list of CPUS
    // Then iterate the calibration on each CPU core.
}
