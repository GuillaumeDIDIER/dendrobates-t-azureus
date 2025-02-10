use crate::calibration::{AVMLocation, CoreLocation, StaticHistCalibrateResult};
use alloc::vec::Vec;
use numa_types::NumaNode;
#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};
#[cfg(any(feature = "use_std", not(feature = "no_std")))]
extern crate std;
#[cfg(all(feature = "no_std", not(feature = "use_std")))]
use hashbrown::HashMap;
#[cfg(any(feature = "use_std", not(feature = "no_std")))]
use rayon::prelude::*;
#[cfg(any(feature = "use_std", not(feature = "no_std")))]
use std::collections::HashMap;

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct CalibrateResult2TNuma<const WIDTH: u64, const N: usize> {
    pub numa_node: NumaNode,
    pub main_core: usize,
    pub helper_core: usize,
    pub res: Vec<StaticHistCalibrateResult<WIDTH, N>>,
}

// TODO: Do a reduction on conflicting calibration_granularity entries
pub fn calibration_result_to_location_map<
    const WIDTH: u64,
    const N: usize,
    T,
    Analysis: Fn(StaticHistCalibrateResult<WIDTH, N>) -> T,
>(
    results: Vec<CalibrateResult2TNuma<WIDTH, N>>,
    analysis: &Analysis, /*Todo slicing*/
    slice_mapping: &impl Fn(usize) -> u8,
    core_location: &impl Fn(usize) -> CoreLocation, // This is the caller's job,
                                                    // he can use numa_node_of_cpu as an approximation, or use CPUID.
                                                    // NB, this aso means we need to dump that info from the machines, for the analysis.
) -> HashMap<AVMLocation, T> {
    let mut analysis_result = HashMap::new();
    for calibrate_2t_result in results {
        let node = calibrate_2t_result.numa_node;
        let attacker = calibrate_2t_result.main_core;
        let victim = calibrate_2t_result.helper_core;
        let attacker_location = core_location(attacker);
        let victim_location = core_location(victim);
        for r in calibrate_2t_result.res {
            let offset = r.offset;
            let vpn = r.page;
            let slice = slice_mapping(r.hash);
            let analysed = analysis(r);
            let location = AVMLocation {
                attacker: attacker_location,
                victim: victim_location,
                memory_numa_node: node,
                memory_slice: slice,
                memory_vpn: vpn,
                memory_offset: offset,
            };
            if analysis_result.contains_key(&location) {
                panic!("Duplicate Location");
            } else {
                analysis_result.insert(location, analysed);
            }
        }
    }
    analysis_result
}
#[cfg(feature = "use_std")]
// TODO: Do a reduction on conflicting calibration_granularity entries
pub fn calibration_result_to_location_map_parallel<
    const WIDTH: u64,
    const N: usize,
    T: Send,
    Analysis: Send + Sync + Fn(StaticHistCalibrateResult<WIDTH, N>) -> T,
>(
    results: Vec<CalibrateResult2TNuma<WIDTH, N>>,
    analysis: &Analysis, /*Todo slicing*/
    slice_mapping: &(impl Send + Sync + Fn(usize) -> u8),
    core_location: &(impl Send + Sync + Fn(usize) -> CoreLocation), // This is the caller's job,
                                                                    // he can use numa_node_of_cpu as an approximation, or use CPUID.
                                                                    // NB, this aso means we need to dump that info from the machines, for the analysis.
) -> HashMap<AVMLocation, T> {
    let result = results
        .into_par_iter()
        .flat_map(|calibrate_2t_result| {
            let node = calibrate_2t_result.numa_node;
            let attacker = calibrate_2t_result.main_core;
            let victim = calibrate_2t_result.helper_core;
            let attacker_location = core_location(attacker);
            let victim_location = core_location(victim);
            calibrate_2t_result.res.into_par_iter().map(move |r| {
                let offset = r.offset;
                let vpn = r.page;
                let slice = slice_mapping(r.hash);
                let analysed = analysis(r);
                let location = AVMLocation {
                    attacker: attacker_location,
                    victim: victim_location,
                    memory_numa_node: node,
                    memory_slice: slice,
                    memory_vpn: vpn,
                    memory_offset: offset,
                };
                (location, analysed)
            })
        })
        .collect();
    result
}
