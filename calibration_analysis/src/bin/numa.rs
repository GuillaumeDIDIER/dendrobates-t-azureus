use calibration_results::calibration_2t::{
    calibration_result_to_location_map, calibration_result_to_location_map_parallel,
};
use calibration_results::numa_results::{BUCKET_NUMBER, BUCKET_SIZE, NumaCalibrationResult};
//use lzma_rs::xz_decompress;
use rmp_serde::Deserializer;
use serde::Deserialize;
use std::env::args;
use std::fs;
use std::io::Cursor;
use calibration_results::calibration::{CoreLocation, StaticHistCalibrateResult};
/*
Design to do, we need to extract, for both FR and FF the raw calibration results (HashMap<AVMLoc, Histograms>)
-> From there we can compute the consolidations for all possible models.

-> Separately, we want to make some histograms, and also try to figure out a way to compare stuffs ?

 */

struct CacheOps<T> {
    flush_hit: T,
    flush_miss: T,
    reload_hit: T,
    reload_miss: T,
}

pub fn run_numa_analysis<const WIDTH: u64, const N: usize>(
    data: NumaCalibrationResult<WIDTH, N>,
) -> () /*TODO*/ {
    /* We need dual and single threshold analysis */
    let results = data.results;
    let topology_info = data.topology_info;
    let slicings = data.slicing;
    let operations = data.operations;
    let uarch = data.micro_architecture;

    let core_location = |core: usize| unsafe {
        // Eventually we need to integrate https://docs.rs/raw-cpuid/latest/raw_cpuid/struct.ExtendedTopologyIter.html
        let node = topology_info[&core].into();
        CoreLocation {
            socket: node,
            core: core as u16,
        }
    };

    let location_map = calibration_result_to_location_map_parallel(results, &|static_hist_result| { () }, &|addr| {slicings.1.hash(addr).try_into().expect("Slice index doesn't fit u8")}, &core_location);
    /*
    let calibration_analysis = calibration_result_to_location_map(
        calibrate_results2t_vec,
        &|calibration_results_1run: StaticHistCalibrateResult<WIDTH, N>| {
            let mut hits = None;
            let mut miss = None;
            for (i, hist) in calibration_results_1run.histogram.into_iter().enumerate() {
                if i == HIT_INDEX {
                    hits = Some(hist);
                } else if i == MISS_INDEX {
                    miss = Some(hist);
                }
            }
            (hits.unwrap(), miss.unwrap())
        },
        &h,
        &core_location,
    );
    */

    // 1. For each location, extract, FLUSH_HIT / FLUSH_MISS - RELOAD_HIT / RELOAD_MISS
    // 2. From there, compute the various reductions from thresholding, without losing the initial data.
    //    (This will require careful use of references, and probably warrants some sort of helper, given we have 34 different configs)
    // 3. Use the reductions to determine thresholds.
    // 4. Compute the expected errors, with average, min, max and stddev.
    println!("Number of entries: {}", location_map.iter().count());
    /* TODO Outstanding statistic question on how to validate if both types of cache cleanup (nope vs explicit flush) give the same distributions
    */


    //unimplemented!()
}

fn run_analysis_from_file(name: &str) -> Result<(), ()> {
    let path = if name.ends_with(&NumaCalibrationResult::<BUCKET_SIZE, BUCKET_NUMBER>::EXTENSION) {
        name.to_owned()
    } else {
        format!(
            "{}.{}",
            name,
            NumaCalibrationResult::<BUCKET_SIZE, BUCKET_NUMBER>::EXTENSION
        )
    };

    let new_result = NumaCalibrationResult::<BUCKET_SIZE, BUCKET_NUMBER>::read_msgpack(&path);
    let new_result = match new_result {
        Ok(r) => {r}
        Err(e) => {eprintln!("{:?}", e); panic!();}
    };

    eprintln!("Read and deserialized {}.msgpack.xz", name);
    println!("Operations");
    for op in &new_result.operations {
        println!("{}: {}", op.name, op.display_name);
    }
    println!(
        "Number of Calibration Results: {}",
        new_result.results.len()
    );
    println!("Micro-architecture: {:?}", new_result.micro_architecture);
    run_numa_analysis(new_result);
    Ok(())
}

fn main() {
    let mut args = args();
    args.next();
    for argument in args {
        let r = run_analysis_from_file(&argument);
        println!("{argument}: {:?}", r);
    }
}
