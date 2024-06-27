use cache_slice::determine_slice;
use cache_slice::utils::core_per_package;
use nix::sched::{sched_getaffinity, sched_setaffinity, CpuSet};
use nix::unistd::Pid;


pub fn main() {
    let nb_cores = core_per_package();
    println!("Found {} cores", nb_cores);

    let target = vec![0x0123456789abcdefu64; 64];

    let old = sched_getaffinity(Pid::from_raw(0)).unwrap();
    let mut core_set = Vec::new();
    for i in 0..CpuSet::count() {
        if old.is_set(i).unwrap() {
            core_set.push(i);
        }
    }


    for core in core_set {
        let mut cpu_set = CpuSet::new();
        cpu_set.set(core).unwrap();
        sched_setaffinity(Pid::this(), &cpu_set).unwrap();
        for addr in target.iter() {
            let slice = determine_slice(addr as *const u64 as *const u8, core as u8, nb_cores);
            match slice {
                Some(slice) => {
                    println!("({:2}) Slice for addr {:x}: {}", core, addr as *const u64 as usize, slice)
                }
                None => {
                    eprintln!("({:2}) Failed to find slice for addr {:x}", core, addr as *const u64 as usize)
                }
            }
        }
        for addr in target.iter() {
            let slice = determine_slice(addr as *const u64 as *const u8, 0, nb_cores);
            match slice {
                Some(slice) => {
                    println!("({:2}) Slice for addr {:x}: {}", core, addr as *const u64 as usize, slice)
                }
                None => {
                    eprintln!("({:2}) Failed to find slice for addr {:x}", core, addr as *const u64 as usize)
                }
            }
        }
        sched_setaffinity(Pid::this(), &old).unwrap();
    }
}
