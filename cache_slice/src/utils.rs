#![cfg(target_arch = "x86_64")]
use raw_cpuid::{CpuId, CpuIdReaderNative, ExtendedTopologyIter, TopologyType};

fn get_topology_iterator() -> Option<ExtendedTopologyIter<CpuIdReaderNative>> {
    let cpuid = CpuId::new();
    let topology_iter = if let Some(t) = cpuid.get_extended_topology_info_v2() {
        Some(t)
    } else if let Some(t) = cpuid.get_extended_topology_info() {
        Some(t)
    } else {
        //panic!("Unsupported CPU");
        None
    };
    topology_iter
}

pub fn threads_per_package() -> Option<u16> {
    if let Some(topology_iter) = get_topology_iterator() {
        let mut t_per_package = None;
        for level in topology_iter {
            if let Some(t_per_package) = t_per_package {
                assert!(t_per_package <= level.processors())
            }
            t_per_package = Some(level.processors())
        }
        t_per_package
    } else {
        None
    }
}

pub fn core_per_package() -> u16 {
    if let Some(topology_iter) = get_topology_iterator() {
        let mut t_per_core = None;
        let mut t_per_package = None;
        for level in topology_iter {
            //println!("{:?}", level);
            match level.level_type() {
                TopologyType::SMT => {
                    assert_eq!(t_per_core, None);
                    t_per_core = Some(level.processors());
                }
                _ => {
                    // TODO identify the right level ?
                    if let Some(t_per_package) = t_per_package {
                        assert!(t_per_package <= level.processors())
                    }
                    // Or change the API to enable the user to specify the topology level to use according to the CPU micro-arch.
                    t_per_package = Some(level.processors())
                }
            }
        }
        if let Some(t_per_core) = t_per_core {
            if let Some(t_per_package) = t_per_package {
                if t_per_package % t_per_core == 0 {
                    return t_per_package / t_per_core;
                }
            }
        }
        0
    } else {
        1
    }
}
