use crate::calibration::StaticHistCalibrateResult;
use crate::numa::NumaNode;
#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct CalibrateResult2TNuma<const WIDTH: u64, const N: usize> {
    pub numa_node: NumaNode,
    pub main_core: usize,
    pub helper_core: usize,
    pub res: Vec<StaticHistCalibrateResult<WIDTH, N>>,
}
