#![cfg(all(feature = "use_std", not(feature = "numa")))]
extern crate std;
use crate::numa::NumaError;
use core::cmp::PartialEq;
use core::fmt::{Display, Formatter};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]

struct NumaNode {}
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

impl Display for crate::numa::NumaNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "<NumaNone>")
    }
}
