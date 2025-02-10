#![deny(unsafe_op_in_unsafe_fn)]

use cache_utils::cache_info::get_cache_info;
use cpuid::complex_addressing::{cache_slicing, CacheAttackSlicing};
use cpuid::MicroArchitecture;

use cache_utils::find_core_per_socket;
use std::process::Command;
use std::str::from_utf8;

pub fn main() {
    println!("{:#?}", get_cache_info());

    let core_per_socket_out = Command::new("sh")
        .arg("-c")
        .arg("lscpu | grep socket | cut -b 22-")
        .output()
        .expect("Failed to detect cpu count");
    //println!("{:#?}", core_per_socket_str);

    let core_per_socket_str = from_utf8(&core_per_socket_out.stdout).unwrap();

    //println!("Number of cores per socket: {}", cps_str);

    let core_per_socket: u8 = core_per_socket_str[0..(core_per_socket_str.len() - 1)]
        .trim()
        .parse()
        .unwrap_or(0);

    let core_per_socket_2 = find_core_per_socket();
    assert_eq!(core_per_socket, core_per_socket_2);

    println!("Number of cores per socket: {}", core_per_socket);

    if let Some(uarch) = MicroArchitecture::get_micro_architecture() {
        if let Some(vendor_family_model_stepping) = MicroArchitecture::get_family_model_stepping() {
            println!("{:?}", uarch);
            let slicing = cache_slicing(
                uarch,
                core_per_socket,
                vendor_family_model_stepping.0,
                vendor_family_model_stepping.1,
                vendor_family_model_stepping.2,
            );
            println!("{:?}", slicing);
            let attack_slicing = CacheAttackSlicing::from(slicing, 64);
            println!("{:?}", attack_slicing);
            println!("{:?}", attack_slicing.image((1 << 12) - 1));
            println!("{:?}", attack_slicing.kernel_compl_basis((1 << 12) - 1));
            println!("{:?}", attack_slicing.image_antecedent((1 << 12) - 1));
        } else {
            println!("No vendor family stepping");
        }
    } else {
        println!("Unknown uarch");
    }

    #[cfg(feature = "numa")]
    {
        use numactl_sys::{
            bitmask, numa_all_nodes_ptr, numa_available, numa_bitmask_isbitset, numa_bitmask_weight,
        };

        let numa_available = unsafe { numa_available() };
        if numa_available < 0 {
            eprintln!("Numa is unavailable");
            return;
        }
        println!("Numa is available");
        let max_node = unsafe { numactl_sys::numa_max_node() }.try_into();
        let max_node = match max_node {
            Ok(max_node) => {
                println!("Max node available: {}", max_node);
                max_node
            }
            Err(_e) => {
                eprintln!("Could not determine max node");
                return;
            }
        };

        let mask = unsafe { numa_all_nodes_ptr as *const bitmask };
        let num_nodes = unsafe { numa_bitmask_weight(mask) };
        println!("Number of available nodes: {}", num_nodes);

        for i in 0..=max_node {
            if unsafe { numa_bitmask_isbitset(mask, i) != 0 } {
                println!("Found node {}", i);
            }
        }
    }
}
