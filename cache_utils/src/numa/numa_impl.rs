#![cfg(all(feature = "numa", feature = "use_std"))]
extern crate std;

use crate::numa::NumaError;
pub use calibration_results::numa::numa_impl::*;
use core::cmp::PartialEq;
use core::ffi::c_uint;
use core::fmt::{Display, Formatter};
use numactl_sys;
use numactl_sys::{
    bitmask, numa_all_nodes_ptr, numa_available, numa_bitmask_alloc, numa_bitmask_clearall,
    numa_bitmask_free, numa_bitmask_isbitset, numa_bitmask_setbit, numa_get_membind,
    numa_migrate_pages, numa_num_possible_nodes, numa_set_membind,
};
#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Mutex;

#[derive(Debug, PartialEq)]
struct Numa {
    available_nodes: HashSet<NumaNode>,
    max_node: c_uint,
}

impl Numa {
    fn initialize() -> Option<Numa> {
        let numa_available = unsafe { numa_available() };
        if numa_available < 0 {
            return None;
        }
        let max_node = unsafe { numactl_sys::numa_max_node() }.try_into();
        let max_node = match max_node {
            Ok(max_node) => max_node,
            Err(_e) => {
                return None;
            }
        };

        let mask = unsafe { numa_all_nodes_ptr as *const bitmask };
        //let num_nodes = unsafe { numa_bitmask_weight(mask) };
        let mut available_nodes = HashSet::new();
        for i in 0..=max_node {
            if unsafe { numa_bitmask_isbitset(mask, i) != 0 } {
                available_nodes.insert(NumaNode { index: i });
            }
        }
        // TODO numa_set_strict

        Some(Self {
            available_nodes,
            max_node,
        })
    }
}

#[derive(Debug, PartialEq)]
enum NumaState {
    Uninitialized,
    Failed,
    Initialized(Numa),
}

static NUMA: Mutex<NumaState> = Mutex::new(NumaState::Uninitialized);

pub fn available_nodes() -> Result<HashSet<NumaNode>, NumaError> {
    let mut option = NUMA.lock().unwrap();
    if *option == NumaState::Failed {
        return Err(NumaError::FailedInit);
    }
    if *option == NumaState::Uninitialized {
        match Numa::initialize() {
            Some(numa) => {
                *option = NumaState::Initialized(numa);
            }
            None => {
                *option = NumaState::Failed;
                return Err(NumaError::FailedInit);
            }
        }
    }
    if let NumaState::Initialized(numa) = &*option {
        Ok(numa.available_nodes.clone())
    } else {
        unreachable!()
    }
}

pub fn set_memory_node(node: NumaNode) -> Result<(), NumaError> {
    let mut numa = NUMA.lock().unwrap();
    let numa = match &mut *numa {
        NumaState::Uninitialized => {
            return Err(NumaError::Uninitialized);
        }
        NumaState::Failed => {
            return Err(NumaError::FailedInit);
        }
        NumaState::Initialized(numa) => numa,
    };

    if !numa.available_nodes.contains(&node) {
        return Err(NumaError::IllegalNode);
    }

    let n = unsafe { numa_num_possible_nodes() };
    let n = if n >= 0 {
        n as u32
    } else {
        return Err(NumaError::FailedInit);
    };

    let old_bitmask = unsafe { numa_get_membind() };

    let bitmask = unsafe { numa_bitmask_alloc(n) };
    unsafe { numa_bitmask_clearall(bitmask) };
    unsafe { numa_bitmask_setbit(bitmask, node.index) };

    let ret = unsafe { numa_migrate_pages(0, old_bitmask, bitmask) };
    if ret < 0 {
        eprintln!("Error: {:?}", nix::errno::Errno::last());
    }
    unsafe { numa_set_membind(bitmask) };

    unsafe { numa_bitmask_free(bitmask) };
    unsafe { numa_bitmask_free(old_bitmask) };

    if ret != 0 {
        Err(NumaError::FailedMigration)
    } else {
        Ok(())
    }
}
/*
numa_migrate_pages() simply uses the migrate_pages system call to cause the pages of the calling task,
or a specified task, to be migated from one set of nodes to another. See migrate_pages(2).
The bit masks representing the nodes should be allocated with numa_allocate_nodemask(),
or with numa_bitmask_alloc() using an n value returned from numa_num_possible_nodes().
A task's current node set can be gotten by calling numa_get_membind().
Bits in the tonodes mask can be set by calls to numa_bitmask_setbit().
*/
pub fn reset_memory_node() -> Result<(), NumaError> {
    unsafe { numa_set_membind(numa_all_nodes_ptr) };
    Ok(())
}
