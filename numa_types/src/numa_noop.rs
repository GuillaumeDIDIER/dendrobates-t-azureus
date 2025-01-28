#![cfg(not(feature = "numa"))]

use core::cmp::PartialEq;
use core::fmt::{Display, Formatter};
#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct NumaNode {}

impl Display for NumaNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "<NumaNone>")
    }
}
impl Into<u8> for NumaNode {
    fn into(self) -> u8 {
        0
    }
}
