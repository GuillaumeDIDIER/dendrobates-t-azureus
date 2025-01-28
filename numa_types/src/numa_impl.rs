#![cfg(feature = "numa")]

use core::cmp::PartialEq;
use core::ffi::c_uint;
use core::fmt::{Display, Formatter};

#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct NumaNode {
    pub index: c_uint,
}

impl Display for NumaNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.index)
    }
}

impl Into<u8> for NumaNode {
    fn into(self) -> u8 {
        if self.index > (u8::MAX as c_uint) {
            panic!("Too many numa Nodes")
        }
        self.index as u8
    }
}
