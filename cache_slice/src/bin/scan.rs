use cache_slice::monitor_address;
use cache_slice::utils::core_per_package;
use nix::sched::{sched_getaffinity, CpuSet};


pub fn main() {
    let nb_cores = core_per_package();
    println!("Found {} cores", nb_cores);

    let target = vec![0x0123456789abcdefu64, 64];
    for core in 0..CpuSet::count() {
        for addr in target.iter() {
            let res = unsafe { monitor_address(addr as *const u64 as *const u8, core as u8, nb_cores) };
            let slice = res.iter().enumerate().max_by_key(|(i, val)| { val });
            match slice {
                Some((slice, _)) => {
                    println!("({:2}) Slice for addr {:x}: {}", core, addr as *const u64 as usize, slice)
                }
                None => {
                    eprintln!("({:2}) Failed to find slice for addr {:x}", core, addr as *const u64 as usize)
                }
            }
        }
        for addr in target.iter() {
            let res = unsafe { monitor_address(addr as *const u64 as *const u8, 0, nb_cores) };
            let slice = res.iter().enumerate().max_by_key(|(i, val)| { val });
            match slice {
                Some((slice, _)) => {
                    println!("({:2}) Slice for addr {:x}: {}", 0, addr as *const u64 as usize, slice)
                }
                None => {
                    eprintln!("({:2}) Failed to find slice for addr {:x}", 0, addr as *const u64 as usize)
                }
            }
        }
    }
}
