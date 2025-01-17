use crate::histograms::StaticHistogram;
#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

pub type VPN = usize;

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct StaticHistCalibrateResult<const WIDTH: u64, const N: usize> {
    pub page: VPN,
    pub offset: isize,
    pub hash: usize,
    pub histogram: Vec<StaticHistogram<WIDTH, N>>,
    pub median: Vec<u64>,
    pub min: Vec<u64>,
    pub max: Vec<u64>,
    pub count: Vec<u64>,
}
