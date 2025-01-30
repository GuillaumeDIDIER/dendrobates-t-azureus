#![cfg(not(feature = "numa"))]

/* Code */

//#![cfg(all(feature = "use_std", not(feature = "numa")))]
extern crate std;
use crate::NumaError;
pub use numa_types::numa_noop::*;
use std::collections::HashSet;
pub fn available_nodes() -> Result<HashSet<NumaNode>, NumaError> {
    let mut r = HashSet::<NumaNode>::new();
    r.insert(NumaNode {});
    Ok(r)
}

pub fn set_memory_node(node: NumaNode) -> Result<(), NumaError> {
    Ok(())
}

pub fn set_memory_nodes(node: HashSet<NumaNode>) -> Result<(), NumaError> {
    Ok(())
}

pub fn reset_memory_node() -> Result<(), NumaError> {
    Ok(())
}

pub fn numa_node_of_cpu(cpu: usize) -> Result<NumaNode, NumaError> {
    Ok(NumaNode {})
}
