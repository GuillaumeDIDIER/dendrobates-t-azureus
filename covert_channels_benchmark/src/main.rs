#![deny(unsafe_op_in_unsafe_fn)]

use cache_utils::calibration::CLFLUSH_NUM_ITER;
use calibration_results::calibration::{CoreLocParameters, ErrorPrediction, LocationParameters};
use covert_channels_evaluation::{benchmark_channel, BenchmarkResults, BenchmarkStats, CovertChannel, CovertChannelBenchmarkResult};
use flush_flush::FlushAndFlush;
use flush_reload::FlushAndReload;
use nix::sched::{sched_getaffinity, CpuSet};
use nix::unistd::Pid;
use num_rational::Rational64;
use numa_utils::NumaNode;
use std::io::{stdout, Write};

//const NUM_BYTES: usize = 1 << 12;
const NUM_BYTES: usize = 1 << 10;

const NUM_PAGES: usize = 1;

const NUM_PAGES_2: usize = 4;

//const NUM_PAGE_MAX: usize = 32;
const NUM_PAGE_MAX: usize = 16;

const NUM_ITER: usize = 2;

fn run_benchmark<T: CovertChannel + 'static + Clone>(
    name: &str,
    constructor: impl Fn(NumaNode, usize, usize) -> T,
    num_iter: usize,
    num_pages: &Vec<usize>,
    old: CpuSet,
    iterate_locations: bool,
) -> BenchmarkStats {
    let mut results = Vec::new();
    let num_entries = num_pages.len();

    let mut count = vec![0; num_entries];
    if iterate_locations {
        for i in numa_utils::available_nodes().unwrap() {
            for j in 0..CpuSet::count() {
                for k in 0..CpuSet::count() {
                    if old.is_set(j).unwrap() && old.is_set(k).unwrap() && j != k {
                        let mut channel = constructor(i, j, k);
                        for (l, num_page) in num_pages.iter().enumerate() {
                            eprintln!(
                                "Benchmarking {} location({},{},{})  with {} pages",
                                name, i, j, k, num_page
                            );

                            for _ in 0..num_iter {
                                count[l] += 1;
                                //eprint!(".");
                                stdout().flush().expect("Failed to flush");

                                let rc = benchmark_channel(channel, *num_page, NUM_BYTES);
                                let r = rc.0;
                                channel = rc.1;
                                results.push((r, *num_page, i, j, k, l));
                            }
                            //eprintln!()
                        }
                    }
                }
            }
        }
    } else {
        let mut channel = constructor(Default::default(), 0, 0);
        for (i, num_page) in num_pages.iter().enumerate() {
            eprintln!("Benchmarking {} with {} pages", name, num_page);
            for _ in 0..num_iter {
                count[i] += 1;
                //eprint!(".");
                stdout().flush().expect("Failed to flush");

                let rc = benchmark_channel(channel.clone(), *num_page, NUM_BYTES);
                let r = rc.0;
                channel = rc.1;
                results.push((r, *num_page, Default::default(), 0, 0, i));
            }
            //eprintln!();
        }
    }

    let mut average_p = vec![0.0; num_entries];
    let mut average_C = vec![0.0; num_entries];
    let mut average_T = vec![0.0; num_entries];
    for result in results.iter() {
        println!(
            "num_page: {}, node: {}, main: {}, helper: {}, result: {:?}",
            result.1, result.2, result.3, result.4, result.0
        );
        println!(
            "C: {}, T: {}",
            result.0.capacity(),
            result.0.true_capacity()
        );
        println!(
            "Detailed:\"{}\",{},{},{},{},{},{},{}",
            name,
            result.1,
            result.2,
            result.3,
            result.4,
            result.0.csv(),
            result.0.capacity(),
            result.0.true_capacity()
        );
        average_p[result.5] += result.0.error.error_rate();
        average_C[result.5] += result.0.capacity();
        average_T[result.5] += result.0.true_capacity()
    }
    for i in 0..num_entries {
        average_p[i] /= count[i] as f64;
        average_C[i] /= count[i] as f64;
        average_T[i] /= count[i] as f64;
        println!(
            "{} - {} Average p: {} C: {}, T: {}",
            name, i, average_p[i], average_C[i], average_T[i]
        );
    }
    let mut var_p = vec![0.0; num_entries];
    let mut var_C = vec![0.0; num_entries];
    let mut var_T = vec![0.0; num_entries];
    for result in results.iter() {
        let p = result.0.error.error_rate() - average_p[result.5];
        var_p[result.5] += p * p;
        let C = result.0.capacity() - average_C[result.5];
        var_C[result.5] += C * C;
        let T = result.0.true_capacity() - average_T[result.5];
        var_T[result.5] += T * T;
    }
    for i in 0..num_entries {
        var_p[i] /= count[i] as f64;
        var_C[i] /= count[i] as f64;
        var_T[i] /= count[i] as f64;
        println!(
            "{} - {} Variance of p: {}, C: {}, T:{}",
            name, i, var_p[i], var_C[i], var_T[i]
        );
        println!(
            "CSV:\"{}\",{},{},{},{},{},{},{}",
            name, i, average_p[i], average_C[i], average_T[i], var_p[i], var_C[i], var_T[i]
        );
    }
    BenchmarkStats {
        raw_res: results,
        average_p,
        var_p,
        average_C,
        var_C,
        average_T,
        var_T,
    }
}

fn norm_threshold(v: &Vec<ErrorPrediction>) -> Rational64 {
    let mut result = Rational64::new(0, 1);
    let mut count = 0;
    for epred in v {
        result += epred.error_ratio();
        count += 1;
    }
    result / count
}

fn norm_location(v: &Vec<(Rational64, Vec<ErrorPrediction>)>) -> Rational64 {
    let mut result = Rational64::new(0, 1);
    let mut count = 0;
    for entry in v {
        result += entry.0;
        count += 1;
    }
    result / count
}

fn main() {
    let old = sched_getaffinity(Pid::from_raw(0)).unwrap();
    println!(
        "Detailed:Benchmark,Pages,numa_node,main_core,helper_core,{},C,T",
        CovertChannelBenchmarkResult::csv_header()
    );
    println!("CSV:Benchmark,Pages,p,C,T,var_p,var_C,var_T");

    // Refactor re-design.
    // We need to list benchmarks that need to be run
    // We need to determine which of these require core configuration.
    // We should serialize the raw data set, with
    // AVMLocation, Error Measurement (True positive, true negative, false positive, false negative), and the execution times.
    // covert_channel_evaluation should probably be made to include the processing of the raw results.

    let num_pages = (1..=NUM_PAGE_MAX).collect();
    //let num_pages = (1..=3).collect();

    let mut results = BenchmarkResults::default();

    // What we could do
    // Topology un-aware ? {ST, DT} x {FF, FR}
    // Numa-MAV, ST x {FF, FR}
    // Numa-MAV-Addr, ST x {FF, FR}
    // Numa-M-Core-AV-Addr, ST x {FF, FR}
    // Numa-M-Core-AV-Addr, ST x {FF, FR}, Best-Numa-M-Core-AV-Addr

    /*    let (topology_unaware_ff_channel, old_mask, _node, _atttack, _victim) =
            FlushAndFlush::new_any_location(
                false,
                LocationParameters {
                    attacker: CoreLocParameters {
                        socket: true,
                        core: true,
                    },
                    victim: CoreLocParameters {
                        socket: true,
                        core: true,
                    },
                    memory_numa_node: true,
                    memory_slice: true,
                    memory_vpn: true,
                    memory_offset: true,
                },
                LocationParameters {
                    attacker: CoreLocParameters {
                        socket: false,
                        core: false,
                    },
                    victim: CoreLocParameters {
                        socket: false,
                        core: false,
                    },
                    memory_numa_node: false,
                    memory_slice: false,
                    memory_vpn: false,
                    memory_offset: false,
                },
                norm_threshold,
                norm_location,
                cache_utils::classifiers::SimpleThresholdBuilder {},
                CLFLUSH_NUM_ITER,
            )
            .unwrap();

        set_affinity(&old_mask).unwrap();

        let tolopology_unware_ff = run_benchmark(
            "Topology Unaware F+F",
            |i, j, k| {
                let mut r = topology_unaware_ff_channel.clone();
                r.set_location(Some(i), Some(j), Some(k)).unwrap();
                r
            },
            NUM_ITER,
            &num_pages,
            old,
            true,
        );

        let (topology_unaware_fr_channel, old_mask, _node, _atttack, _victim) =
            FlushAndReload::new_any_location(
                false,
                LocationParameters {
                    attacker: CoreLocParameters {
                        socket: true,
                        core: true,
                    },
                    victim: CoreLocParameters {
                        socket: true,
                        core: true,
                    },
                    memory_numa_node: true,
                    memory_slice: true,
                    memory_vpn: true,
                    memory_offset: true,
                },
                LocationParameters {
                    attacker: CoreLocParameters {
                        socket: false,
                        core: false,
                    },
                    victim: CoreLocParameters {
                        socket: false,
                        core: false,
                    },
                    memory_numa_node: false,
                    memory_slice: false,
                    memory_vpn: false,
                    memory_offset: false,
                },
                norm_threshold,
                norm_location,
                cache_utils::classifiers::SimpleThresholdBuilder {},
                CLFLUSH_NUM_ITER,
            )
            .unwrap();
        set_affinity(&old_mask).unwrap();

        let tolopolgy_unware_fr = run_benchmark(
            "Topology Unaware F+R",
            |i, j, k| {
                let mut r = topology_unaware_fr_channel.clone();
                r.set_location(Some(i), Some(j), Some(k)).unwrap();
                r
            },
            NUM_ITER,
            &num_pages,
            old,
            true,
        );
    */
    let singlethreshold_ff = run_benchmark(
        "Topology Aware F+F",
        |i, j, k| {
            let (mut r, (node, attacker, victim)) = FlushAndFlush::new_with_locations(
                vec![(i, j, k)].into_iter(),
                LocationParameters {
                    attacker: CoreLocParameters {
                        socket: true,
                        core: true,
                    },
                    victim: CoreLocParameters {
                        socket: true,
                        core: true,
                    },
                    memory_numa_node: true,
                    memory_slice: true,
                    memory_vpn: true,
                    memory_offset: true,
                },
                LocationParameters {
                    attacker: CoreLocParameters {
                        socket: true,
                        core: true,
                    },
                    victim: CoreLocParameters {
                        socket: true,
                        core: true,
                    },
                    memory_numa_node: true,
                    memory_slice: true,
                    memory_vpn: true,
                    memory_offset: false,
                },
                norm_threshold,
                norm_location,
                calibration_results::classifiers::SimpleThresholdBuilder {},
                CLFLUSH_NUM_ITER,
            )
            .unwrap();
            r.set_location(Some(i), Some(j), Some(k)).unwrap();
            r
        },
        NUM_ITER,
        &num_pages,
        old,
        true,
    );

    let fr = run_benchmark(
        "Topology Aware F+R",
        |i, j, k| {
            let (mut r, (node, attacker, victim)) = FlushAndReload::new_with_locations(
                vec![(i, j, k)].into_iter(),
                LocationParameters {
                    attacker: CoreLocParameters {
                        socket: true,
                        core: true,
                    },
                    victim: CoreLocParameters {
                        socket: true,
                        core: true,
                    },
                    memory_numa_node: true,
                    memory_slice: true,
                    memory_vpn: true,
                    memory_offset: true,
                },
                LocationParameters {
                    attacker: CoreLocParameters {
                        socket: true,
                        core: true,
                    },
                    victim: CoreLocParameters {
                        socket: true,
                        core: true,
                    },
                    memory_numa_node: true,
                    memory_slice: true,
                    memory_vpn: true,
                    memory_offset: false,
                },
                norm_threshold,
                norm_location,
                calibration_results::classifiers::SimpleThresholdBuilder {},
                CLFLUSH_NUM_ITER,
            )
            .unwrap();
            r.set_location(Some(i), Some(j), Some(k)).unwrap();
            r
        },
        NUM_ITER,
        &num_pages,
        old,
        true,
    );
}
