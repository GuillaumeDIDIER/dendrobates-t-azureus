use covert_channels_evaluation::{BenchmarkResults, CovertChannelBenchmarkResult};
use numa_utils::NumaNode;
use rayon::prelude::*;
use std::cmp::max;
use std::collections::HashMap;
use std::env::args;
use std::fs::File;
use std::io::Write;

fn analyze_result(name: String, r: BenchmarkResults) {
    eprintln!("Processing {}...", name);
    let mut max_node = None;
    let mut max_core = None;
    for (benchmark, results) in r.results {
        if max_node.is_none() {
            let mut local_max_node = NumaNode::default();
            let mut local_max_core = 0;
            for raw_res in results.raw_res.iter() {
                let k = (raw_res.2, raw_res.3, raw_res.4);
                if max_node.is_none() {
                    local_max_node = max(local_max_node, raw_res.2);
                    local_max_core = max(local_max_core, max(raw_res.3, raw_res.4));
                }
            }
            max_node = Some(local_max_node);
            max_core = Some(local_max_core);
        }
        let num_logical_core = max_core.unwrap() + 1;
        let num_phy_core = num_logical_core;

        let mut heatmap = HashMap::<(NumaNode, usize, usize), _>::new();
        for raw_res in results.raw_res {
            let k = (
                raw_res.2,
                raw_res.3 % num_phy_core,
                raw_res.4 % num_phy_core,
            );
            let entry = heatmap.entry(k).or_insert_with(|| Vec::new());
            entry.push(raw_res.0);
        }

        let mut heatmap: Vec<((NumaNode, usize, usize), _)> = heatmap
            .into_par_iter()
            .map(|(k, v)| {
                let reduced = v.into_par_iter().reduce(
                    || CovertChannelBenchmarkResult {
                        num_bytes_transmitted: 0,
                        error: Default::default(),
                        time_rdtsc: 0,
                        time_seconds: Default::default(),
                    },
                    |a, b| CovertChannelBenchmarkResult {
                        num_bytes_transmitted: a.num_bytes_transmitted + b.num_bytes_transmitted,
                        error: a.error + b.error,
                        time_rdtsc: a.time_rdtsc + b.time_rdtsc,
                        time_seconds: a.time_seconds + b.time_seconds,
                    },
                );
                (k, reduced)
            })
            .collect();
        heatmap.par_sort_by_key(|(k, _)| {
            (k.0.index as usize * num_phy_core + k.1) * num_phy_core + k.2
        });

        let error_map = heatmap
            .par_iter()
            .map(|(k, v)| {
                format!(
                    "{}, {}, {}\n",
                    (k.0.index as usize * num_phy_core + k.1),
                    k.2,
                    v.error.error_rate()
                )
            })
            .collect::<Vec<_>>()
            .join("");
        let true_capacity_map = heatmap
            .par_iter()
            .map(|(k, v)| {
                format!(
                    "{}, {}, {}\n",
                    (k.0.index as usize * num_phy_core + k.1),
                    k.2,
                    v.true_capacity()
                )
            })
            .collect::<Vec<_>>()
            .join("");
        println!("{}", benchmark);
        let error_map_name = format!("{}-{}-{}.csv", name, benchmark, "ErrorMap");
        // println!("{}", error_map);
        {
            let mut file = File::create(error_map_name).unwrap();
            writeln!(&mut file, "{}", error_map);
        }

        let true_capacity_map_name = format!("{}-{}-{}.csv", name, benchmark, "TrueCap");
        // println!("{}", true_capacity_map);
        {
            let mut file = File::create(true_capacity_map_name).unwrap();
            writeln!(&mut file, "{}", true_capacity_map);
        }
    }
}

fn main() {
    let mut args = args();
    args.next();
    for argument in args {
        let suffix = ".CBR.msgpack.zst";
        let basename = argument.strip_suffix(suffix).unwrap().to_owned();
        let result = BenchmarkResults::read_msgpack(&argument);
        match result {
            Ok(results) => {
                analyze_result(basename, results);
            }
            Err(e) => {
                eprintln!("Error: {}, {}", argument, e);
            }
        }
    }
}
