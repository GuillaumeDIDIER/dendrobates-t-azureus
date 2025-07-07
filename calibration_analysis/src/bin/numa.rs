#![feature(generic_const_exprs)]
#![deny(unsafe_op_in_unsafe_fn)]

use calibration_results::numa_results::{BUCKET_NUMBER, BUCKET_SIZE};
use rayon::prelude::*;
use std::env::args;
use std::fmt::Display;
/*
Design to do, we need to extract, for both FR and FF the raw calibration results (HashMap<AVMLoc, Histograms>)
-> From there we can compute the consolidations for all possible models.

-> Separately, we want to make some histograms, and also try to figure out a way to compare stuffs ?

 */

/***********************
 * PGFPlots Histograms *
 ***********************/

/*
    println!(
    "[{}] Reload Single Threshold: {}, Error Prediction: {}",
    base_name.as_ref(),
    reload_single_threshold.0,
    reload_single_threshold.1
);*/

/* TODO : Evaluate if ndarray would be better than our current hashmaps*/

fn main() {
    if let Ok(num_threads) = std::thread::available_parallelism() {
        let rayon_thread = (num_threads.get() * 15) >> 4;
        rayon::ThreadPoolBuilder::new()
            .num_threads(rayon_thread)
            .build_global()
            .unwrap();
    }
    let mut args = args();
    args.next();
    for argument in args {
        let r = calibration_analysis::run_analysis_from_file::<BUCKET_SIZE, { BUCKET_NUMBER / 2 }>(
            &argument,
        );
        println!("{argument}: {:?}", r);
    }
}
