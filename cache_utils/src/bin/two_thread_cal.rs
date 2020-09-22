use cache_utils::calibration::{
    calibrate_fixed_freq_2_thread, flush_and_reload, get_cache_slicing, load_and_flush, only_flush,
    only_reload, reload_and_flush, CalibrateOperation2T, CalibrateResult2T, CalibrationOptions,
    HistParams, Verbosity, CFLUSH_BUCKET_NUMBER, CFLUSH_BUCKET_SIZE, CFLUSH_NUM_ITER,
};
use cache_utils::mmap::MMappedMemory;
use cache_utils::{flush, maccess, noop};
use nix::sched::{sched_getaffinity, CpuSet};
use nix::unistd::Pid;

use core::arch::x86_64 as arch_x86;

use std::collections::HashMap;
use std::process::Command;
use std::str::from_utf8;

unsafe fn multiple_access(p: *const u8) {
    maccess::<u8>(p);
    maccess::<u8>(p);
    arch_x86::_mm_mfence();
    maccess::<u8>(p);
    arch_x86::_mm_mfence();
    maccess::<u8>(p);
    arch_x86::_mm_mfence();
    maccess::<u8>(p);
    maccess::<u8>(p);
}

const SIZE: usize = 2 << 20;

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
struct ASV {
    pub attacker: u8,
    pub slice: u8,
    pub victim: u8,
}

struct ResultAnalysis {
    // indexed by bucket size
    pub miss: Vec<u32>,
    pub miss_cum_sum: Vec<u32>,
    pub miss_total: u32,
    pub hit: Vec<u32>,
    pub hit_cum_sum: Vec<u32>,
    pub hit_total: u32,
    pub error_miss_less_than_hit: Vec<u32>,
    pub error_hit_less_than_miss: Vec<u32>,
    pub min_error_hlm: u32,
    pub min_error_mlh: u32,
}

#[derive(Debug, Clone, Copy)]
struct Threshold {
    pub error_rate: f32,
    pub threshold: usize,
    // extend with other possible algorithm ?
    pub is_hlm: bool,
    pub num_true_hit: u32,
    pub num_false_hit: u32,
    pub num_true_miss: u32,
    pub num_false_miss: u32,
}

fn main() {
    // Grab a slice of memory

    let core_per_socket_out = Command::new("sh")
        .arg("-c")
        .arg("lscpu | grep socket | cut -b 22-")
        .output()
        .expect("Failed to detect cpu count");
    //println!("{:#?}", core_per_socket_str);

    let core_per_socket_str = from_utf8(&core_per_socket_out.stdout).unwrap();

    //println!("Number of cores per socket: {}", cps_str);

    let core_per_socket: u8 = core_per_socket_str[0..(core_per_socket_str.len() - 1)]
        .parse()
        .unwrap_or(0);

    println!("Number of cores per socket: {}", core_per_socket);

    let m = MMappedMemory::new(SIZE);
    let array = m.slice();

    let cache_line_size = 64;

    // Generate core iterator
    let mut core_pairs: Vec<(usize, usize)> = Vec::new();
    let old = sched_getaffinity(Pid::from_raw(0)).unwrap();

    for i in 0..CpuSet::count() {
        for j in 0..CpuSet::count() {
            if old.is_set(i).unwrap() && old.is_set(j).unwrap() {
                core_pairs.push((i, j));
                println!("{},{}", i, j);
            }
        }
    }

    // operations
    // Call calibrate 2T \o/

    let verbose_level = Verbosity::RawResult;

    unsafe {
        let pointer = (&array[0]) as *const u8;

        if pointer as usize & (cache_line_size - 1) != 0 {
            panic!("not aligned nicely");
        }

        let operations = [
            CalibrateOperation2T {
                prepare: maccess::<u8>,
                op: only_flush,
                name: "clflush_remote_hit",
                display_name: "clflush remote hit",
            },
            CalibrateOperation2T {
                prepare: maccess::<u8>,
                op: load_and_flush,
                name: "clflush_shared_hit",
                display_name: "clflush shared hit",
            },
            CalibrateOperation2T {
                prepare: flush,
                op: only_flush,
                name: "clflush_miss_f",
                display_name: "clflush miss - f",
            },
            CalibrateOperation2T {
                prepare: flush,
                op: load_and_flush,
                name: "clflush_local_hit_f",
                display_name: "clflush local hit - f",
            },
            CalibrateOperation2T {
                prepare: noop::<u8>,
                op: only_flush,
                name: "clflush_miss_n",
                display_name: "clflush miss - n",
            },
            CalibrateOperation2T {
                prepare: noop::<u8>,
                op: load_and_flush,
                name: "clflush_local_hit_n",
                display_name: "clflush local hit - n",
            },
            CalibrateOperation2T {
                prepare: noop::<u8>,
                op: flush_and_reload,
                name: "reload_miss",
                display_name: "reload miss",
            },
            CalibrateOperation2T {
                prepare: maccess::<u8>,
                op: reload_and_flush,
                name: "reload_remote_hit",
                display_name: "reload remote hit",
            },
            CalibrateOperation2T {
                prepare: maccess::<u8>,
                op: only_reload,
                name: "reload_shared_hit",
                display_name: "reload shared hit",
            },
            CalibrateOperation2T {
                prepare: noop::<u8>,
                op: only_reload,
                name: "reload_local_hit",
                display_name: "reload local hit",
            },
        ];

        let r = calibrate_fixed_freq_2_thread(
            pointer,
            64,                        // FIXME : MAGIC
            array.len() as isize >> 3, // MAGIC
            &mut core_pairs.into_iter(),
            &operations,
            CalibrationOptions {
                hist_params: HistParams {
                    bucket_number: CFLUSH_BUCKET_NUMBER,
                    bucket_size: CFLUSH_BUCKET_SIZE,
                    iterations: CFLUSH_NUM_ITER,
                },
                verbosity: verbose_level,
                optimised_addresses: true,
            },
            core_per_socket,
        );

        let mut analysis = HashMap::<ASV, ResultAnalysis>::new();

        let miss_name = "clflush_miss_n";
        let hit_name = "clflush_remote_hit";

        let miss_index = operations
            .iter()
            .position(|op| op.name == miss_name)
            .unwrap();
        let hit_index = operations
            .iter()
            .position(|op| op.name == hit_name)
            .unwrap();

        let slicing = get_cache_slicing(core_per_socket);

        let h = if let Some(s) = slicing {
            if s.can_hash() {
                |addr: usize| -> u8 { slicing.unwrap().hash(addr).unwrap() }
            } else {
                panic!("No slicing function known");
            }
        } else {
            panic!("No slicing function known");
        };

        for result in r {
            match result.res {
                Err(e) => {
                    eprintln!("Ooops : {:#?}", e);
                    panic!()
                }
                Ok(results) => {
                    for r in results {
                        let offset = r.offset;
                        let miss_hist = &r.histogram[miss_index];
                        let hit_hist = &r.histogram[hit_index];

                        if miss_hist.len() != hit_hist.len() {
                            panic!("Maformed results");
                        }
                        let len = miss_hist.len();
                        let mut miss_cum_sum = vec![0; len];
                        let mut hit_cum_sum = vec![0; len];
                        miss_cum_sum[0] = miss_hist[0];
                        hit_cum_sum[0] = hit_hist[0];
                        for i in 1..len {
                            miss_cum_sum[i] = miss_hist[i] + miss_cum_sum[i - 1];
                            hit_cum_sum[i] = hit_hist[i] + hit_cum_sum[i - 1];
                        }
                        let miss_total = miss_cum_sum[len - 1];
                        let hit_total = hit_cum_sum[len - 1];

                        let mut error_miss_less_than_hit = vec![0; len - 1];
                        let mut error_hit_less_than_miss = vec![0; len - 1];

                        let mut min_error_hlm = u32::max_value();
                        let mut min_error_mlh = u32::max_value();

                        for i in 0..(len - 1) {
                            error_hit_less_than_miss[i] =
                                miss_cum_sum[i] + (hit_total - hit_cum_sum[i]);
                            error_miss_less_than_hit[i] =
                                hit_cum_sum[i] + (miss_total - miss_cum_sum[i]);

                            if error_hit_less_than_miss[i] < min_error_hlm {
                                min_error_hlm = error_hit_less_than_miss[i];
                            }
                            if error_miss_less_than_hit[i] < min_error_mlh {
                                min_error_mlh = error_miss_less_than_hit[i];
                            }
                        }

                        analysis.insert(
                            ASV {
                                attacker: result.main_core as u8,
                                slice: h(offset as usize),
                                victim: result.helper_core as u8,
                            },
                            ResultAnalysis {
                                miss: miss_hist.clone(),
                                miss_cum_sum,
                                miss_total,
                                hit: hit_hist.clone(),
                                hit_cum_sum,
                                hit_total,
                                error_miss_less_than_hit,
                                error_hit_less_than_miss,
                                min_error_hlm,
                                min_error_mlh,
                            },
                        );
                    }
                }
            }
        }
        let mut thresholds = HashMap::new();
        for (asv, results) in analysis {
            let hlm = results.min_error_hlm < results.min_error_mlh;
            let (errors, min_error) = if hlm {
                (&results.error_hit_less_than_miss, results.min_error_hlm)
            } else {
                (&results.error_miss_less_than_hit, results.min_error_mlh)
            };

            let mut threshold_vec = Vec::new();

            // refactor some of this logic into methods of analysis ?

            for i in 0..errors.len() {
                if errors[i] == min_error {
                    let num_true_hit;
                    let num_false_hit;
                    let num_true_miss;
                    let num_false_miss;
                    if hlm {
                        num_true_hit = results.hit_cum_sum[i];
                        num_false_hit = results.miss_cum_sum[i];
                        num_true_miss = results.miss_total - num_false_hit;
                        num_false_miss = results.hit_total - num_true_hit;
                    } else {
                        num_true_miss = results.miss_cum_sum[i];
                        num_false_miss = results.hit_cum_sum[i];
                        num_true_hit = results.hit_total - num_false_miss;
                        num_false_hit = results.miss_total - num_true_miss;
                    }
                    threshold_vec.push(Threshold {
                        threshold: i,
                        is_hlm: hlm,
                        num_true_hit,
                        num_false_hit,
                        num_true_miss,
                        num_false_miss,
                        error_rate: min_error as f32
                            / (results.hit_total + results.miss_total) as f32,
                    })
                }
                /*

                */
            }
            thresholds.insert(asv, threshold_vec);
        }
        eprintln!("Thresholds :\n{:#?}", thresholds);
        println!("Thresholds :\n{:#?}", thresholds);
    }
}
