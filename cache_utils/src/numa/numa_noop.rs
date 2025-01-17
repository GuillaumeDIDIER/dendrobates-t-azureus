#![cfg(all(feature = "use_std", not(feature = "numa")))]
extern crate std;
use crate::numa::NumaError;
pub use calibration_results::numa::numa_none::*;
use core::cmp::PartialEq;
use core::fmt::{Display, Formatter};
use std::collections::HashSet;

pub fn available_nodes() -> Result<HashSet<NumaNode>, NumaError> {
    let mut r = HashSet::<NumaNode>::new();
    r.insert(NumaNode {});
    Ok(r)
}

pub fn set_memory_node(node: crate::numa::NumaNode) -> Result<(), NumaError> {
    Ok(())
}

pub fn reset_memory_node() -> Result<(), NumaError> {
    Ok(())
}
