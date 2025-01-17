#![cfg(not(feature = "numa"))]

use core::cmp::PartialEq;
use core::fmt::{Display, Formatter};
#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct NumaNode {}

impl Display for crate::numa::NumaNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "<NumaNone>")
    }
}
