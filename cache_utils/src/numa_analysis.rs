use crate::calibrate_2t::CalibrateResult2TNuma;
use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

/** This module is used to factorize the analysis code, to enable off-line analysis
*/

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct OperationNames {
    pub name: String,
    pub display_name: String,
}

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct NumaCalibrationResult<const WIDTH: u64, const N: usize> {
    pub operations: Vec<OperationNames>,
    pub results: Vec<CalibrateResult2TNuma<WIDTH, N>>,
}

pub fn run_numa_analysis<const WIDTH: u64, const N: usize>(
    data: NumaCalibrationResult<WIDTH, N>,
) -> () /*TODO*/ {
    unimplemented!()
}

pub const BUCKET_NUMBER: usize = 1024;
pub const BUCKET_SIZE: u64 = 1;
