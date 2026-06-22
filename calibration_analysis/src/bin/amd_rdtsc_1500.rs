#![feature(generic_const_exprs)]
#![deny(unsafe_op_in_unsafe_fn)]

use calibration_analysis::{CacheOps, make_projection, run_tsc_from_file};
use calibration_results::calibration::{
    CoreLocParameters, CoreLocation, LocationParameters, PartialLocationOwned,
};
use calibration_results::calibration_2t::calibration_result_to_location_map_parallel;
use calibration_results::histograms::{SimpleBucketU64, StaticHistogram};
use calibration_results::numa_results::BUCKET_SIZE;
use calibration_results::numa_results::NumaCalibrationResult;
use calibration_results::reduce;
use num::integer::gcd;
use rayon::prelude::*;
use std::collections::HashSet;
use std::env::args;
use std::path::Path;
const BUCKET_NUMBER: usize = 1500;
fn main() {
    let mut args = args();
    args.next();
    for argument in args {
        let r = run_tsc_from_file::<BUCKET_SIZE, BUCKET_NUMBER>(&argument);
        println!("{argument}: {:?}", r);
    }
}
