use cache_slice::monitor_address;
use cache_slice::utils::core_per_package;
use nix::sched::{sched_getaffinity, sched_setaffinity, CpuSet};
use nix::unistd::Pid;


pub fn main() {
    let nb_cores = core_per_package();
    println!("Found {} cores", nb_cores);

    let target = vec![0x0123456789abcdefu64; 1024];

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
        for addr in target.iter().step_by(8) {
            let address = addr as *const u64 as *const u8;
            let res = unsafe { monitor_address(address, core as u8, nb_cores) }.unwrap();
            print!("({:2}) {:x}:", core, address as usize);
            for slice in res {
                print!(" {:6}", slice)
            }
            println!();
            /*let slice = res.iter().enumerate().max_by_key(|(_i, val)| { **val });
            match slice {
                Some((slice, _)) => {
                    println!("({:2}) Slice for addr {:x}: {}", core, addr as *const u64 as usize, slice)
                }
                None => {
                    eprintln!("({:2}) Failed to find slice for addr {:x}", core, addr as *const u64 as usize)
                }
            }*/
        }
        /*for addr in target.iter() {
            let res = unsafe { monitor_address(addr as *const u64 as *const u8, 0, nb_cores) }.unwrap();
            let slice = res.iter().enumerate().max_by_key(|(_i, val)| { **val });
            match slice {
                Some((slice, _)) => {
                    println!("({:2}) Slice for addr {:x}: {}", 0, addr as *const u64 as usize, slice)
                }
                None => {
                    eprintln!("({:2}) Failed to find slice for addr {:x}", 0, addr as *const u64 as usize)
                }
            }
        }*/
        sched_setaffinity(Pid::this(), &old).unwrap();
    }
}
