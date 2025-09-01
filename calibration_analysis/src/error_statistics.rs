use crate::{CacheOps, JsonOutput, Output, QuadErrors, QuadThresholdErrors};
use calibration_results::calibration::{AVMLocation, ErrorPrediction, LocationParameters, PartialLocation, PartialLocationOwned};
use calibration_results::classifiers::{ErrorPredictor, compute_theoretical_optimum_error};
use calibration_results::histograms::{SimpleBucketU64, StaticHistogram, StaticHistogramCumSum};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::sync::{Mutex, RwLock};
use json::{object, JsonValue};
use json::object::Object;

#[derive(Debug, Clone, Copy)]
pub struct StatisticsResults {
    pub average: ErrorPrediction,
    pub min: (ErrorPrediction, AVMLocation),
    pub q1: (ErrorPrediction, AVMLocation),
    pub med: (ErrorPrediction, AVMLocation),
    pub q3: (ErrorPrediction, AVMLocation),
    pub max: (ErrorPrediction, AVMLocation),
}

fn error_prediction_json(e: ErrorPrediction) -> JsonValue {
    object! {
        "true_hit" => e.true_hit,
        "true_miss" => e.true_miss,
        "false_hit" => e.false_hit,
        "false_miss" => e.false_miss,
        "error_rate" => e.error_rate(),
    }
}

pub fn avm_location_json(avm_loc: AVMLocation) -> JsonValue {
    object! {
        "memory_node" => avm_loc.memory_numa_node.index,
        "attacker_socket" => avm_loc.attacker.socket,
        "attacker_core" => avm_loc.attacker.core,
        "victim_socket" => avm_loc.victim.socket,
        "victim_core" => avm_loc.victim.core,
        "memory_page" => avm_loc.memory_vpn,
        "memory_offset" => avm_loc.memory_offset,
        // Do not include slice, we don't have it properly.
    }
}

pub fn location_parameters_json(avm_loc: LocationParameters) -> JsonValue {
    object! {
        "memory_node" => avm_loc.memory_numa_node,
        "attacker_socket" => avm_loc.attacker.socket,
        "attacker_core" => avm_loc.attacker.core,
        "victim_socket" => avm_loc.victim.socket,
        "victim_core" => avm_loc.victim.core,
        "memory_page" => avm_loc.memory_vpn,
        "memory_offset" => avm_loc.memory_offset,
        // Do not include slice, we don't have it properly.
    }
}

fn partial_location_json(partial_location: &PartialLocationOwned) -> JsonValue {
    let mut result = Object::new();
    let params = partial_location.get_params();
    let loc = partial_location.get_location();
    if params.memory_numa_node {
        result.insert("memory_node", loc.memory_numa_node.index.into());
    }
    if params.attacker.socket {
        result.insert("attacker_socket", loc.attacker.socket.into());
    }
    if params.attacker.core {
        result.insert("attacker_core", loc.attacker.core.into());
    }
    if params.victim.socket {
        result.insert("victim_socket", loc.victim.socket.into());
    }
    if params.victim.core {
        result.insert("victim_core", loc.victim.core.into());
    }
    if params.memory_vpn {
        result.insert("memory_page", loc.memory_vpn.into());
    }
    if params.memory_offset {
        result.insert("memory_offset", loc.memory_offset.into());
    }
    JsonValue::Object(result)
}
impl JsonOutput for StatisticsResults {
    fn to_json(&self, base: &Object) -> Vec<Object> {
        let mut result = base.clone();
        result.insert("avg_err", error_prediction_json(self.average));
        result.insert("min_err", error_prediction_json(self.min.0));
        result.insert("min_err_loc", avm_location_json(self.min.1));
        result.insert("q1_err", error_prediction_json(self.q1.0));
        result.insert("q1_err_loc", avm_location_json(self.q1.1));
        result.insert("med_err", error_prediction_json(self.med.0));
        result.insert("med_err_loc", avm_location_json(self.med.1));
        result.insert("q3_err", error_prediction_json(self.q3.0));
        result.insert("q3_err_loc", avm_location_json(self.q3.1));
        result.insert("max_err", error_prediction_json(self.max.0));
        result.insert("max_err_loc", avm_location_json(self.max.1));
        vec![result]
    }
}

#[derive(Debug, Clone)]
pub struct ErrorStatisticsResults {
    pub average: StatisticsResults,
    pub choice: Vec<(String, PartialLocationOwned, StatisticsResults)>,
}

impl JsonOutput for ErrorStatisticsResults {
    fn to_json(&self, base: &Object) -> Vec<Object> {
        let mut key = base.clone();
        key.insert("selection", JsonValue::Null);
        let mut result = self.average.to_json(&key);
        for choice in &self.choice {
            let mut key = base.clone();
            key.insert("selection", JsonValue::String(choice.0.clone()));
            key.insert("selected_location", partial_location_json(&choice.1));
            result.extend(choice.2.to_json(&key));
        }
        result
    }
}

impl Output for ErrorStatisticsResults {
    fn write(&self, output_file: &mut File, name: impl AsRef<str>) {
        writeln!(
            output_file,
            "{}-Average: {}",
            name.as_ref(),
            self.average.average
        )
        .expect("Failed to output");
        writeln!(output_file, "{}-Min: {}", name.as_ref(), self.average.min.0)
            .expect("Failed to output");
        writeln!(output_file, "{}-Q1: {}", name.as_ref(), self.average.q1.0)
            .expect("Failed to output");
        writeln!(output_file, "{}-Med: {}", name.as_ref(), self.average.med.0)
            .expect("Failed to output");
        writeln!(output_file, "{}-Q3: {}", name.as_ref(), self.average.q3.0)
            .expect("Failed to output");
        writeln!(output_file, "{}-Max: {}", name.as_ref(), self.average.max.0)
            .expect("Failed to output");
        for choice in &self.choice {
            writeln!(
                output_file,
                "{}-{}-Location: {}",
                name.as_ref(),
                choice.0,
                choice.1
            )
            .expect("Failed to output");
            writeln!(
                output_file,
                "{}-{}-Average: {}",
                name.as_ref(),
                choice.0,
                choice.2.average
            )
            .expect("Failed to output");
            writeln!(
                output_file,
                "{}-{}-Min: {}",
                name.as_ref(),
                choice.0,
                choice.2.min.0
            )
            .expect("Failed to output");
            writeln!(
                output_file,
                "{}-{}-Q1: {}",
                name.as_ref(),
                choice.0,
                choice.2.q1.0
            )
            .expect("Failed to output");
            writeln!(
                output_file,
                "{}-{}-Med: {}",
                name.as_ref(),
                choice.0,
                choice.2.med.0
            )
            .expect("Failed to output");
            writeln!(
                output_file,
                "{}-{}-Q3: {}",
                name.as_ref(),
                choice.0,
                choice.2.q3.0
            )
            .expect("Failed to output");
            writeln!(
                output_file,
                "{}-{}-Max: {}",
                name.as_ref(),
                choice.0,
                choice.2.max.0
            )
            .expect("Failed to output");
        }
    }
}

pub fn compute_statistics_no_projection<const WIDTH: u64, const N: usize>(
    full_location_map: &HashMap<AVMLocation, CacheOps<StaticHistogram<{ WIDTH }, { N + N }>>>,
    choices: Vec<(String, LocationParameters)>,
    thresholds_map: &HashMap<AVMLocation, QuadThresholdErrors<SimpleBucketU64<WIDTH, { N + N }>>>,
) -> QuadErrors<ErrorStatisticsResults> {
    assert_ne!(full_location_map.par_iter().count(), 1);
    println!("Computing statistics...");

    let all_error_predictions: Vec<(AVMLocation, QuadErrors<ErrorPrediction>)> = full_location_map
        .par_iter()
        .map(|(k, hists)| {
            let thresholds = &thresholds_map[k];
            let hist_cum_sums = CacheOps {
                flush_hit: StaticHistogramCumSum::from(&hists.flush_hit),
                flush_miss: StaticHistogramCumSum::from(&hists.flush_miss),
                reload_hit: StaticHistogramCumSum::from(&hists.reload_hit),
                reload_miss: StaticHistogramCumSum::from(&hists.reload_miss),
            };
            let flush_single_error = thresholds
                .flush_single_threshold
                .0
                .error_prediction(&hist_cum_sums.flush_hit, &hist_cum_sums.flush_miss);
            let flush_dual_error = thresholds
                .flush_dual_threshold
                .0
                .error_prediction(&hist_cum_sums.flush_hit, &hist_cum_sums.flush_miss);
            let reload_single_error = thresholds
                .reload_single_threshold
                .0
                .error_prediction(&hist_cum_sums.reload_hit, &hist_cum_sums.reload_miss);
            let reload_dual_error = thresholds
                .reload_dual_threshold
                .0
                .error_prediction(&hist_cum_sums.reload_hit, &hist_cum_sums.reload_miss);
            let flush_th_error = compute_theoretical_optimum_error(
                &hist_cum_sums.flush_hit,
                &hist_cum_sums.flush_miss,
            );
            let reload_th_error = compute_theoretical_optimum_error(
                &hist_cum_sums.reload_hit,
                &hist_cum_sums.reload_miss,
            );

            (
                *k,
                QuadErrors {
                    flush_single_error,
                    flush_dual_error,
                    flush_th_error,
                    reload_single_error,
                    reload_dual_error,
                    reload_th_error,
                },
            )
        })
        .collect();

    println!("Computed all error predictions");

    let mut quad_all_error_pred: QuadErrors<Vec<(AVMLocation, ErrorPrediction)>> = QuadErrors {
        flush_single_error: all_error_predictions
            .par_iter()
            .map(|(l, q)| (*l, q.flush_single_error))
            .collect(),
        flush_dual_error: all_error_predictions
            .par_iter()
            .map(|(l, q)| (*l, q.flush_dual_error))
            .collect(),
        flush_th_error: all_error_predictions
            .par_iter()
            .map(|(l, q)| (*l, q.flush_th_error))
            .collect(),
        reload_single_error: all_error_predictions
            .par_iter()
            .map(|(l, q)| (*l, q.reload_single_error))
            .collect(),
        reload_dual_error: all_error_predictions
            .par_iter()
            .map(|(l, q)| (*l, q.reload_dual_error))
            .collect(),
        reload_th_error: all_error_predictions
            .par_iter()
            .map(|(l, q)| (*l, q.reload_th_error))
            .collect(),
    };

    println!("Remapped");

    /*for (choice_name, choice_projection) in */
    let results = choices
        .into_iter()
        /*.into_par_iter()*/
        .map(|(choice_name, choice_projection)| {
            // we probably want to turn this into an accumulate / reduce parallel version.
            let mapped = RwLock::new(HashMap::<
                PartialLocationOwned,
                Mutex<Vec<(AVMLocation, QuadErrors<ErrorPrediction>)>>,
            >::new());

            all_error_predictions.par_iter().for_each(|(k, q)| {
                let errors = q;
                let choice_partial_location = PartialLocationOwned::new(choice_projection, *k);
                let guard = mapped.read().unwrap();
                if let Some(entry) = guard.get(&choice_partial_location) {
                    let mut inner_vec = entry.lock().unwrap();
                    inner_vec.push((*k, *errors));
                } else {
                    drop(guard);
                    let mut write_guard = mapped.write().unwrap();
                    write_guard
                        .entry(choice_partial_location)
                        .or_insert_with(|| Mutex::new(Vec::new()));
                    drop(write_guard);
                    mapped
                        .read()
                        .unwrap()
                        .get(&choice_partial_location)
                        .unwrap()
                        .lock()
                        .unwrap()
                        .push((*k, *errors));
                }
            });

            let two_level_error_predictions: Vec<(
                PartialLocationOwned,
                Vec<(AVMLocation, QuadErrors<ErrorPrediction>)>,
            )> = mapped
                .into_inner()
                .unwrap()
                .into_par_iter()
                .map(|(k, v)| {
                    let map = v.into_inner().unwrap().into_par_iter().collect();

                    (k, map)
                })
                .collect();

            let quad_two_level_error_predictions: QuadErrors<
                Vec<(
                    PartialLocationOwned,
                    ErrorPrediction,
                    Vec<(AVMLocation, ErrorPrediction)>,
                )>,
            > = QuadErrors {
                flush_single_error: two_level_error_predictions
                    .par_iter()
                    .map(|(p, qv)| {
                        let v: Vec<(AVMLocation, ErrorPrediction)> = qv
                            .par_iter()
                            .map(|(l, q)| (*l, q.flush_single_error))
                            .collect();
                        //v.par_sort_by_key(|(_l, e)| e.error_ratio()); // TODO, remove this.
                        let e_avg = v.par_iter().map(|(_l, e)| e).sum();
                        (*p, e_avg, v)
                    })
                    .collect(),
                flush_dual_error: two_level_error_predictions
                    .par_iter()
                    .map(|(p, qv)| {
                        let v: Vec<(AVMLocation, ErrorPrediction)> = qv
                            .par_iter()
                            .map(|(l, q)| (*l, q.flush_dual_error))
                            .collect();
                        //v.par_sort_by_key(|(_l, e)| e.error_ratio()); // TODO, remove this.
                        let e_avg = v.par_iter().map(|(_l, e)| e).sum();
                        (*p, e_avg, v)
                    })
                    .collect(),
                flush_th_error: two_level_error_predictions
                    .par_iter()
                    .map(|(p, qv)| {
                        let v: Vec<(AVMLocation, ErrorPrediction)> =
                            qv.par_iter().map(|(l, q)| (*l, q.flush_th_error)).collect();
                        //v.par_sort_by_key(|(_l, e)| e.error_ratio()); // TODO, remove this.
                        let e_avg = v.par_iter().map(|(_l, e)| e).sum();
                        (*p, e_avg, v)
                    })
                    .collect(),
                reload_single_error: two_level_error_predictions
                    .par_iter()
                    .map(|(p, qv)| {
                        let v: Vec<(AVMLocation, ErrorPrediction)> = qv
                            .par_iter()
                            .map(|(l, q)| (*l, q.reload_single_error))
                            .collect();
                        //v.par_sort_by_key(|(_l, e)| e.error_ratio()); // TODO, remove this.
                        let e_avg = v.par_iter().map(|(_l, e)| e).sum();
                        (*p, e_avg, v)
                    })
                    .collect(),
                reload_dual_error: two_level_error_predictions
                    .par_iter()
                    .map(|(p, qv)| {
                        let v: Vec<(AVMLocation, ErrorPrediction)> = qv
                            .par_iter()
                            .map(|(l, q)| (*l, q.reload_dual_error))
                            .collect();
                        //v.par_sort_by_key(|(_l, e)| e.error_ratio()); // TODO, remove this.
                        let e_avg = v.par_iter().map(|(_l, e)| e).sum();
                        (*p, e_avg, v)
                    })
                    .collect(),
                reload_th_error: two_level_error_predictions
                    .par_iter()
                    .map(|(p, qv)| {
                        let v: Vec<(AVMLocation, ErrorPrediction)> = qv
                            .par_iter()
                            .map(|(l, q)| (*l, q.reload_th_error))
                            .collect();
                        //v.par_sort_by_key(|(_l, e)| e.error_ratio()); // TODO, remove this.
                        let e_avg = v.par_iter().map(|(_l, e)| e).sum();
                        (*p, e_avg, v)
                    })
                    .collect(),
            };

            // TODO replace this with quickselect by  key

            //quad_two_level_error_predictions
            //    .apply_mut(|v| v.par_sort_by_key(|(_p, e, _v)| e.error_ratio()));

            let mut quad_best_choice = quad_two_level_error_predictions.map(|v| {
                v.into_par_iter()
                    .min_by_key(|(_p, e, _v)| e.error_ratio())
                    .unwrap()
            });

            quad_best_choice.apply_mut(|v| v.2.par_sort_by_key(|(_l, e)| e.error_ratio()));

            let best_min = quad_best_choice.apply(|v| {
                v.2[0]
                /* *v.2.par_iter()
                .min_by_key(|(_l, e)| e.error_ratio())
                .unwrap()*/
            });
            let best_max = quad_best_choice.apply(|v| {
                let len = v.2.len();
                v.2[len - 1]
                /* *v.2.par_iter()
                .max_by_key(|(_l, e)| e.error_ratio())
                .unwrap()*/
            });

            let best_med = quad_best_choice.apply(|v| {
                /*let inner = &mut v.2;
                let len = inner.len();
                quickselect_by_key(inner, (len - 1) >> 1, |(_l, e)| e.error_ratio())
                    .unwrap()
                    .1*/
                let len = v.2.len();
                v.2[(len - 1) >> 1]
            });

            let best_q1 = quad_best_choice.apply(|v| {
                /*let inner = &mut v.2;
                let len = inner.len();
                quickselect_by_key(inner, (len - 1) >> 2, |(_l, e)| e.error_ratio())
                    .unwrap()
                    .1*/
                let len = v.2.len();
                v.2[(len - 1) >> 2] // Use the definition, first such that at least 25% are <=
            });
            let best_q3 = quad_best_choice.apply(|v| {
                /*let inner = &mut v.2;
                let len = inner.len();
                quickselect_by_key(inner, (3 * len - 1) >> 2, |(_l, e)| e.error_ratio())
                    .unwrap()
                    .1*/
                let len = v.2.len();
                v.2[(3 * len - 1) >> 2] // Use the definition, first such that at least 75% are <=
            });

            // This one is easy, it's already computed.
            let best_avg = quad_best_choice.apply(|v| v.1);

            let best_location = quad_best_choice.apply(|v| v.0);
            let flush_single_error = (
                choice_name.clone(),
                best_location.flush_single_error,
                StatisticsResults {
                    average: best_avg.flush_single_error,
                    min: (best_min.flush_single_error.1, best_min.flush_single_error.0),
                    q1: (best_q1.flush_single_error.1, best_q1.flush_single_error.0),
                    med: (best_med.flush_single_error.1, best_med.flush_single_error.0),
                    q3: (best_q3.flush_single_error.1, best_q3.flush_single_error.0),
                    max: (best_max.flush_single_error.1, best_max.flush_single_error.0),
                },
            );

            let flush_dual_error = (
                choice_name.clone(),
                best_location.flush_dual_error,
                StatisticsResults {
                    average: best_avg.flush_dual_error,
                    min: (best_min.flush_dual_error.1, best_min.flush_dual_error.0),
                    q1: (best_q1.flush_dual_error.1, best_q1.flush_dual_error.0),
                    med: (best_med.flush_dual_error.1, best_med.flush_dual_error.0),
                    q3: (best_q3.flush_dual_error.1, best_q3.flush_dual_error.0),
                    max: (best_max.flush_dual_error.1, best_max.flush_dual_error.0),
                },
            );

            let flush_th_error = (
                choice_name.clone(),
                best_location.flush_th_error,
                StatisticsResults {
                    average: best_avg.flush_th_error,
                    min: (best_min.flush_th_error.1, best_min.flush_th_error.0),
                    q1: (best_q1.flush_th_error.1, best_q1.flush_th_error.0),
                    med: (best_med.flush_th_error.1, best_med.flush_th_error.0),
                    q3: (best_q3.flush_th_error.1, best_q3.flush_th_error.0),
                    max: (best_max.flush_th_error.1, best_max.flush_th_error.0),
                },
            );

            let reload_single_error = (
                choice_name.clone(),
                best_location.reload_single_error,
                StatisticsResults {
                    average: best_avg.reload_single_error,
                    min: (
                        best_min.reload_single_error.1,
                        best_min.reload_single_error.0,
                    ),
                    q1: (best_q1.reload_single_error.1, best_q1.reload_single_error.0),
                    med: (
                        best_med.reload_single_error.1,
                        best_med.reload_single_error.0,
                    ),
                    q3: (best_q3.reload_single_error.1, best_q3.reload_single_error.0),
                    max: (
                        best_max.reload_single_error.1,
                        best_max.reload_single_error.0,
                    ),
                },
            );

            let reload_dual_error = (
                choice_name.clone(),
                best_location.reload_dual_error,
                StatisticsResults {
                    average: best_avg.reload_dual_error,
                    min: (best_min.reload_dual_error.1, best_min.reload_dual_error.0),
                    q1: (best_q1.reload_dual_error.1, best_q1.reload_dual_error.0),
                    med: (best_med.reload_dual_error.1, best_med.reload_dual_error.0),
                    q3: (best_q3.reload_dual_error.1, best_q3.reload_dual_error.0),
                    max: (best_max.reload_dual_error.1, best_max.reload_dual_error.0),
                },
            );
            let reload_th_error = (
                choice_name,
                best_location.reload_th_error,
                StatisticsResults {
                    average: best_avg.reload_th_error,
                    min: (best_min.reload_th_error.1, best_min.reload_th_error.0),
                    q1: (best_q1.reload_th_error.1, best_q1.reload_th_error.0),
                    med: (best_med.reload_th_error.1, best_med.reload_th_error.0),
                    q3: (best_q3.reload_th_error.1, best_q3.reload_th_error.0),
                    max: (best_max.reload_th_error.1, best_max.reload_th_error.0),
                },
            );
            QuadErrors {
                flush_single_error,
                flush_dual_error,
                flush_th_error,
                reload_single_error,
                reload_dual_error,
                reload_th_error,
            }
        })
        .collect::<Vec<_>>();
    println!("Computed choices");

    let mut choice = QuadErrors {
        flush_single_error: Vec::new(),
        flush_dual_error: Vec::new(),
        flush_th_error: Vec::new(),
        reload_single_error: Vec::new(),
        reload_dual_error: Vec::new(),
        reload_th_error: Vec::new(),
    };

    for q in results {
        choice.flush_single_error.push(q.flush_single_error);
        choice.flush_dual_error.push(q.flush_dual_error);
        choice.flush_th_error.push(q.flush_th_error);
        choice.reload_single_error.push(q.reload_single_error);
        choice.reload_dual_error.push(q.reload_dual_error);
        choice.reload_th_error.push(q.reload_th_error);
    }

    // Rewrite this in parallel.

    // We want to produce a QuadErrors<Vec<String, Location, StatisticsResults>>, but we don't start from a quad errors :/
    // The final swap should be reasonably easy though.

    // This is faster than quickselect, apparently
    quad_all_error_pred.apply_mut(|all_e_p| all_e_p.par_sort_by_key(|(_l, e)| e.error_ratio()));

    //theoretical_all_error_pred.apply_mut(|all_e_p| all_e_p.par_sort_by_key(|(_l, _p, e)| e.error_ratio()));

    let all_min = quad_all_error_pred.apply(|all_e_p| {
        /* *all_e_p
        .par_iter()
        .min_by_key(|(_l, _p, e)| e.error_ratio())
        .unwrap()*/
        all_e_p[0]
    });
    let all_max = quad_all_error_pred.apply(|all_e_p| {
        /* *all_e_p
        .par_iter()
        .max_by_key(|(_l, _p, e)| e.error_ratio())
        .unwrap()*/
        let len = all_e_p.len();
        all_e_p[len - 1]
    }); // TODO, extend rayon to have min_max_by_key ?

    let all_med = quad_all_error_pred.apply(|all_e_p| {
        let len = all_e_p.len();
        /*quickselect_by_key(&mut all_e_p, (len - 1) >> 1, |(_l, _p, e)| e.error_ratio())
        .unwrap()
        .1*/
        all_e_p[(len - 1) >> 1]
    });
    let all_q1 = quad_all_error_pred.apply(|all_e_p| {
        let len = all_e_p.len();
        /*quickselect_by_key(&mut all_e_p, (len - 1) >> 2, |(_l, _p, e)| e.error_ratio())
        .unwrap()
        .1*/
        all_e_p[(len - 1) >> 2]
    });
    let all_q3 = quad_all_error_pred.apply(|all_e_p| {
        let len = all_e_p.len();
        /*quickselect_by_key(&mut all_e_p, (3*len - 1) >> 2, |(_l, _p, e)| e.error_ratio())
        .unwrap()
        .1*/
        all_e_p[(3 * len - 1) >> 2]
    });

    let all_avg: QuadErrors<ErrorPrediction> =
        quad_all_error_pred.apply(|all_e_p| all_e_p.par_iter().map(|(_l, e)| e).sum());

    println!("Extracted statistics");

    let flush_single_error = ErrorStatisticsResults {
        average: StatisticsResults {
            average: all_avg.flush_single_error,
            min: (all_min.flush_single_error.1, all_min.flush_single_error.0),
            q1: (all_q1.flush_single_error.1, all_q1.flush_single_error.0),
            med: (all_med.flush_single_error.1, all_med.flush_single_error.0),
            q3: (all_q3.flush_single_error.1, all_q3.flush_single_error.0),
            max: (all_max.flush_single_error.1, all_max.flush_single_error.0),
        },
        choice: choice.flush_single_error,
    };
    let flush_dual_error = ErrorStatisticsResults {
        average: StatisticsResults {
            average: all_avg.flush_dual_error,
            min: (all_min.flush_dual_error.1, all_min.flush_dual_error.0),
            q1: (all_q1.flush_dual_error.1, all_q1.flush_dual_error.0),
            med: (all_med.flush_dual_error.1, all_med.flush_dual_error.0),
            q3: (all_q3.flush_dual_error.1, all_q3.flush_dual_error.0),
            max: (all_max.flush_dual_error.1, all_max.flush_dual_error.0),
        },
        choice: choice.flush_dual_error,
    };
    let flush_th_error = ErrorStatisticsResults {
        average: StatisticsResults {
            average: all_avg.flush_th_error,
            min: (all_min.flush_th_error.1, all_min.flush_th_error.0),
            q1: (all_q1.flush_th_error.1, all_q1.flush_th_error.0),
            med: (all_med.flush_th_error.1, all_med.flush_th_error.0),
            q3: (all_q3.flush_th_error.1, all_q3.flush_th_error.0),
            max: (all_max.flush_th_error.1, all_max.flush_th_error.0),
        },
        choice: choice.flush_th_error,
    };

    let reload_single_error = ErrorStatisticsResults {
        average: StatisticsResults {
            average: all_avg.reload_single_error,
            min: (all_min.reload_single_error.1, all_min.reload_single_error.0),
            q1: (all_q1.reload_single_error.1, all_q1.reload_single_error.0),
            med: (all_med.reload_single_error.1, all_med.reload_single_error.0),
            q3: (all_q3.reload_single_error.1, all_q3.reload_single_error.0),
            max: (all_max.reload_single_error.1, all_max.reload_single_error.0),
        },
        choice: choice.reload_single_error,
    };
    let reload_dual_error = ErrorStatisticsResults {
        average: StatisticsResults {
            average: all_avg.reload_dual_error,
            min: (all_min.reload_dual_error.1, all_min.reload_dual_error.0),
            q1: (all_q1.reload_dual_error.1, all_q1.reload_dual_error.0),
            med: (all_med.reload_dual_error.1, all_med.reload_dual_error.0),
            q3: (all_q3.reload_dual_error.1, all_q3.reload_dual_error.0),
            max: (all_max.reload_dual_error.1, all_max.reload_dual_error.0),
        },
        choice: choice.reload_dual_error,
    };

    let reload_th_error = ErrorStatisticsResults {
        average: StatisticsResults {
            average: all_avg.reload_th_error,
            min: (all_min.reload_th_error.1, all_min.reload_th_error.0),
            q1: (all_q1.reload_th_error.1, all_q1.reload_th_error.0),
            med: (all_med.reload_th_error.1, all_med.reload_th_error.0),
            q3: (all_q3.reload_th_error.1, all_q3.reload_th_error.0),
            max: (all_max.reload_th_error.1, all_max.reload_th_error.0),
        },
        choice: choice.reload_th_error,
    };

    println!("Done");

    QuadErrors {
        flush_single_error,
        flush_dual_error,
        flush_th_error,
        reload_single_error,
        reload_dual_error,
        reload_th_error,
    }
}

pub fn compute_statistics<const WIDTH: u64, const N: usize>(
    full_location_map: &HashMap<AVMLocation, CacheOps<StaticHistogram<{ WIDTH }, { N + N }>>>,
    known_projection: LocationParameters,
    choices: Vec<(String, LocationParameters)>,
    thresholds_map: &HashMap<
        PartialLocationOwned,
        QuadThresholdErrors<SimpleBucketU64<WIDTH, { N + N }>>,
    >,
) -> QuadErrors<ErrorStatisticsResults> {
    assert_ne!(full_location_map.par_iter().count(), 1);

    println!("Computing statistics...");

    let all_error_predictions: Vec<(
        AVMLocation,
        PartialLocationOwned,
        QuadErrors<ErrorPrediction>,
    )> = full_location_map
        .par_iter()
        .map(|(k, hists)| {
            let key = PartialLocationOwned::new(known_projection, *k);
            let thresholds = &thresholds_map[&key];
            let hist_cum_sums = CacheOps {
                flush_hit: StaticHistogramCumSum::from(&hists.flush_hit),
                flush_miss: StaticHistogramCumSum::from(&hists.flush_miss),
                reload_hit: StaticHistogramCumSum::from(&hists.reload_hit),
                reload_miss: StaticHistogramCumSum::from(&hists.reload_miss),
            };
            let flush_single_error = thresholds
                .flush_single_threshold
                .0
                .error_prediction(&hist_cum_sums.flush_hit, &hist_cum_sums.flush_miss);
            let flush_dual_error = thresholds
                .flush_dual_threshold
                .0
                .error_prediction(&hist_cum_sums.flush_hit, &hist_cum_sums.flush_miss);
            let reload_single_error = thresholds
                .reload_single_threshold
                .0
                .error_prediction(&hist_cum_sums.reload_hit, &hist_cum_sums.reload_miss);
            let reload_dual_error = thresholds
                .reload_dual_threshold
                .0
                .error_prediction(&hist_cum_sums.reload_hit, &hist_cum_sums.reload_miss);
            let flush_th_error = compute_theoretical_optimum_error(
                &hist_cum_sums.flush_hit,
                &hist_cum_sums.flush_miss,
            );
            let reload_th_error = compute_theoretical_optimum_error(
                &hist_cum_sums.reload_hit,
                &hist_cum_sums.reload_miss,
            );

            (
                *k,
                key,
                QuadErrors {
                    flush_single_error,
                    flush_dual_error,
                    flush_th_error,
                    reload_single_error,
                    reload_dual_error,
                    reload_th_error,
                },
            )
        })
        .collect();

    println!("Computed all error predictions");

    let mut quad_all_error_pred: QuadErrors<
        Vec<(AVMLocation, PartialLocationOwned, ErrorPrediction)>,
    > = QuadErrors {
        flush_single_error: all_error_predictions
            .par_iter()
            .map(|(l, p, q)| (*l, *p, q.flush_single_error))
            .collect(),
        flush_dual_error: all_error_predictions
            .par_iter()
            .map(|(l, p, q)| (*l, *p, q.flush_dual_error))
            .collect(),
        flush_th_error: all_error_predictions
            .par_iter()
            .map(|(l, p, q)| (*l, *p, q.flush_th_error))
            .collect(),
        reload_single_error: all_error_predictions
            .par_iter()
            .map(|(l, p, q)| (*l, *p, q.reload_single_error))
            .collect(),
        reload_dual_error: all_error_predictions
            .par_iter()
            .map(|(l, p, q)| (*l, *p, q.reload_dual_error))
            .collect(),
        reload_th_error: all_error_predictions
            .par_iter()
            .map(|(l, p, q)| (*l, *p, q.reload_th_error))
            .collect(),
    };

    println!("Remapped");

    /*for (choice_name, choice_projection) in */
    let results = choices
        /*.into_iter()*/
        .into_par_iter()
        .map(|(choice_name, choice_projection)| {
            // we probably want to turn this into an accumulate / reduce parallel version.
            let mapped = RwLock::new(HashMap::<
                PartialLocationOwned,
                Mutex<Vec<(AVMLocation, QuadErrors<ErrorPrediction>)>>,
            >::new());

            all_error_predictions.par_iter().for_each(|(k, _p, q)| {
                let errors = q;
                let choice_partial_location = PartialLocationOwned::new(choice_projection, *k);
                let guard = mapped.read().unwrap();
                if let Some(entry) = guard.get(&choice_partial_location) {
                    let mut inner_vec = entry.lock().unwrap();
                    inner_vec.push((*k, *errors));
                } else {
                    drop(guard);
                    let mut write_guard = mapped.write().unwrap();
                    write_guard
                        .entry(choice_partial_location)
                        .or_insert_with(|| Mutex::new(Vec::new()));
                    drop(write_guard);
                    mapped
                        .read()
                        .unwrap()
                        .get(&choice_partial_location)
                        .unwrap()
                        .lock()
                        .unwrap()
                        .push((*k, *errors));
                }
            });

            let two_level_error_predictions: Vec<(
                PartialLocationOwned,
                Vec<(AVMLocation, QuadErrors<ErrorPrediction>)>,
            )> = mapped
                .into_inner()
                .unwrap()
                .into_par_iter()
                .map(|(k, v)| {
                    let map = v.into_inner().unwrap().into_par_iter().collect();

                    (k, map)
                })
                .collect();

            let quad_two_level_error_predictions: QuadErrors<
                Vec<(
                    PartialLocationOwned,
                    ErrorPrediction,
                    Vec<(AVMLocation, ErrorPrediction)>,
                )>,
            > = QuadErrors {
                flush_single_error: two_level_error_predictions
                    .par_iter()
                    .map(|(p, qv)| {
                        let v: Vec<(AVMLocation, ErrorPrediction)> = qv
                            .par_iter()
                            .map(|(l, q)| (*l, q.flush_single_error))
                            .collect();
                        //v.par_sort_by_key(|(_l, e)| e.error_ratio()); // TODO, remove this.
                        let e_avg = v.par_iter().map(|(_l, e)| e).sum();
                        (*p, e_avg, v)
                    })
                    .collect(),
                flush_dual_error: two_level_error_predictions
                    .par_iter()
                    .map(|(p, qv)| {
                        let v: Vec<(AVMLocation, ErrorPrediction)> = qv
                            .par_iter()
                            .map(|(l, q)| (*l, q.flush_dual_error))
                            .collect();
                        //v.par_sort_by_key(|(_l, e)| e.error_ratio()); // TODO, remove this.
                        let e_avg = v.par_iter().map(|(_l, e)| e).sum();
                        (*p, e_avg, v)
                    })
                    .collect(),
                flush_th_error: two_level_error_predictions
                    .par_iter()
                    .map(|(p, qv)| {
                        let v: Vec<(AVMLocation, ErrorPrediction)> =
                            qv.par_iter().map(|(l, q)| (*l, q.flush_th_error)).collect();
                        //v.par_sort_by_key(|(_l, e)| e.error_ratio()); // TODO, remove this.
                        let e_avg = v.par_iter().map(|(_l, e)| e).sum();
                        (*p, e_avg, v)
                    })
                    .collect(),
                reload_single_error: two_level_error_predictions
                    .par_iter()
                    .map(|(p, qv)| {
                        let v: Vec<(AVMLocation, ErrorPrediction)> = qv
                            .par_iter()
                            .map(|(l, q)| (*l, q.reload_single_error))
                            .collect();
                        //v.par_sort_by_key(|(_l, e)| e.error_ratio()); // TODO, remove this.
                        let e_avg = v.par_iter().map(|(_l, e)| e).sum();
                        (*p, e_avg, v)
                    })
                    .collect(),
                reload_dual_error: two_level_error_predictions
                    .par_iter()
                    .map(|(p, qv)| {
                        let v: Vec<(AVMLocation, ErrorPrediction)> = qv
                            .par_iter()
                            .map(|(l, q)| (*l, q.reload_dual_error))
                            .collect();
                        //v.par_sort_by_key(|(_l, e)| e.error_ratio()); // TODO, remove this.
                        let e_avg = v.par_iter().map(|(_l, e)| e).sum();
                        (*p, e_avg, v)
                    })
                    .collect(),
                reload_th_error: two_level_error_predictions
                    .par_iter()
                    .map(|(p, qv)| {
                        let v: Vec<(AVMLocation, ErrorPrediction)> = qv
                            .par_iter()
                            .map(|(l, q)| (*l, q.reload_th_error))
                            .collect();
                        //v.par_sort_by_key(|(_l, e)| e.error_ratio()); // TODO, remove this.
                        let e_avg = v.par_iter().map(|(_l, e)| e).sum();
                        (*p, e_avg, v)
                    })
                    .collect(),
            };

            // TODO replace this with quickselect by  key

            //quad_two_level_error_predictions
            //    .apply_mut(|v| v.par_sort_by_key(|(_p, e, _v)| e.error_ratio()));

            let mut quad_best_choice = quad_two_level_error_predictions.map(|v| {
                v.into_par_iter()
                    .min_by_key(|(_p, e, _v)| e.error_ratio())
                    .unwrap()
            });

            quad_best_choice.apply_mut(|v| v.2.par_sort_by_key(|(_l, e)| e.error_ratio()));

            let best_min = quad_best_choice.apply(|v| {
                v.2[0]
                /* *v.2.par_iter()
                .min_by_key(|(_l, e)| e.error_ratio())
                .unwrap()*/
            });
            let best_max = quad_best_choice.apply(|v| {
                let len = v.2.len();
                v.2[len - 1]
                /* *v.2.par_iter()
                .max_by_key(|(_l, e)| e.error_ratio())
                .unwrap()*/
            });

            let best_med = quad_best_choice.apply(|v| {
                /*let inner = &mut v.2;
                let len = inner.len();
                quickselect_by_key(inner, (len - 1) >> 1, |(_l, e)| e.error_ratio())
                    .unwrap()
                    .1*/
                let len = v.2.len();
                v.2[(len - 1) >> 1]
            });

            let best_q1 = quad_best_choice.apply(|v| {
                /*let inner = &mut v.2;
                let len = inner.len();
                quickselect_by_key(inner, (len - 1) >> 2, |(_l, e)| e.error_ratio())
                    .unwrap()
                    .1*/
                let len = v.2.len();
                v.2[(len - 1) >> 2] // Use the definition, first such that at least 25% are <=
            });
            let best_q3 = quad_best_choice.apply(|v| {
                /*let inner = &mut v.2;
                let len = inner.len();
                quickselect_by_key(inner, (3 * len - 1) >> 2, |(_l, e)| e.error_ratio())
                    .unwrap()
                    .1*/
                let len = v.2.len();
                v.2[(3 * len - 1) >> 2] // Use the definition, first such that at least 75% are <=
            });

            // This one is easy, it's already computed.
            let best_avg = quad_best_choice.apply(|v| v.1);

            let best_location = quad_best_choice.apply(|v| v.0);
            let flush_single_error = (
                choice_name.clone(),
                best_location.flush_single_error,
                StatisticsResults {
                    average: best_avg.flush_single_error,
                    min: (best_min.flush_single_error.1, best_min.flush_single_error.0),
                    q1: (best_q1.flush_single_error.1, best_q1.flush_single_error.0),
                    med: (best_med.flush_single_error.1, best_med.flush_single_error.0),
                    q3: (best_q3.flush_single_error.1, best_q3.flush_single_error.0),
                    max: (best_max.flush_single_error.1, best_max.flush_single_error.0),
                },
            );

            let flush_dual_error = (
                choice_name.clone(),
                best_location.flush_dual_error,
                StatisticsResults {
                    average: best_avg.flush_dual_error,
                    min: (best_min.flush_dual_error.1, best_min.flush_dual_error.0),
                    q1: (best_q1.flush_dual_error.1, best_q1.flush_dual_error.0),
                    med: (best_med.flush_dual_error.1, best_med.flush_dual_error.0),
                    q3: (best_q3.flush_dual_error.1, best_q3.flush_dual_error.0),
                    max: (best_max.flush_dual_error.1, best_max.flush_dual_error.0),
                },
            );

            let flush_th_error = (
                choice_name.clone(),
                best_location.flush_th_error,
                StatisticsResults {
                    average: best_avg.flush_th_error,
                    min: (best_min.flush_th_error.1, best_min.flush_th_error.0),
                    q1: (best_q1.flush_th_error.1, best_q1.flush_th_error.0),
                    med: (best_med.flush_th_error.1, best_med.flush_th_error.0),
                    q3: (best_q3.flush_th_error.1, best_q3.flush_th_error.0),
                    max: (best_max.flush_th_error.1, best_max.flush_th_error.0),
                },
            );

            let reload_single_error = (
                choice_name.clone(),
                best_location.reload_single_error,
                StatisticsResults {
                    average: best_avg.reload_single_error,
                    min: (
                        best_min.reload_single_error.1,
                        best_min.reload_single_error.0,
                    ),
                    q1: (best_q1.reload_single_error.1, best_q1.reload_single_error.0),
                    med: (
                        best_med.reload_single_error.1,
                        best_med.reload_single_error.0,
                    ),
                    q3: (best_q3.reload_single_error.1, best_q3.reload_single_error.0),
                    max: (
                        best_max.reload_single_error.1,
                        best_max.reload_single_error.0,
                    ),
                },
            );

            let reload_dual_error = (
                choice_name.clone(),
                best_location.reload_dual_error,
                StatisticsResults {
                    average: best_avg.reload_dual_error,
                    min: (best_min.reload_dual_error.1, best_min.reload_dual_error.0),
                    q1: (best_q1.reload_dual_error.1, best_q1.reload_dual_error.0),
                    med: (best_med.reload_dual_error.1, best_med.reload_dual_error.0),
                    q3: (best_q3.reload_dual_error.1, best_q3.reload_dual_error.0),
                    max: (best_max.reload_dual_error.1, best_max.reload_dual_error.0),
                },
            );
            let reload_th_error = (
                choice_name,
                best_location.reload_th_error,
                StatisticsResults {
                    average: best_avg.reload_th_error,
                    min: (best_min.reload_th_error.1, best_min.reload_th_error.0),
                    q1: (best_q1.reload_th_error.1, best_q1.reload_th_error.0),
                    med: (best_med.reload_th_error.1, best_med.reload_th_error.0),
                    q3: (best_q3.reload_th_error.1, best_q3.reload_th_error.0),
                    max: (best_max.reload_th_error.1, best_max.reload_th_error.0),
                },
            );
            QuadErrors {
                flush_single_error,
                flush_dual_error,
                flush_th_error,
                reload_single_error,
                reload_dual_error,
                reload_th_error,
            }
        })
        .collect::<Vec<_>>();
    println!("Computed choices");

    let mut choice = QuadErrors {
        flush_single_error: Vec::new(),
        flush_dual_error: Vec::new(),
        flush_th_error: Vec::new(),
        reload_single_error: Vec::new(),
        reload_dual_error: Vec::new(),
        reload_th_error: Vec::new(),
    };

    for q in results {
        choice.flush_single_error.push(q.flush_single_error);
        choice.flush_dual_error.push(q.flush_dual_error);
        choice.flush_th_error.push(q.flush_th_error);
        choice.reload_single_error.push(q.reload_single_error);
        choice.reload_dual_error.push(q.reload_dual_error);
        choice.reload_th_error.push(q.reload_th_error);
    }

    // Rewrite this in parallel.

    // We want to produce a QuadErrors<Vec<String, Location, StatisticsResults>>, but we don't start from a quad errors :/
    // The final swap should be reasonably easy though.

    // TODO replace this with a quickselect by key.
    quad_all_error_pred.apply_mut(|all_e_p| all_e_p.par_sort_by_key(|(_l, _p, e)| e.error_ratio()));

    //theoretical_all_error_pred.apply_mut(|all_e_p| all_e_p.par_sort_by_key(|(_l, _p, e)| e.error_ratio()));

    let all_min = quad_all_error_pred.apply(|all_e_p| {
        /* *all_e_p
        .par_iter()
        .min_by_key(|(_l, _p, e)| e.error_ratio())
        .unwrap()*/
        all_e_p[0]
    });
    let all_max = quad_all_error_pred.apply(|all_e_p| {
        /* *all_e_p
        .par_iter()
        .max_by_key(|(_l, _p, e)| e.error_ratio())
        .unwrap()*/
        let len = all_e_p.len();
        all_e_p[len - 1]
    }); // TODO, extend rayon to have min_max_by_key ?

    let all_med = quad_all_error_pred.apply(|all_e_p| {
        let len = all_e_p.len();
        /*quickselect_by_key(&mut all_e_p, (len - 1) >> 1, |(_l, _p, e)| e.error_ratio())
        .unwrap()
        .1*/
        all_e_p[(len - 1) >> 1]
    });
    let all_q1 = quad_all_error_pred.apply(|all_e_p| {
        let len = all_e_p.len();
        /*quickselect_by_key(&mut all_e_p, (len - 1) >> 2, |(_l, _p, e)| e.error_ratio())
        .unwrap()
        .1*/
        all_e_p[(len - 1) >> 2]
    });
    let all_q3 = quad_all_error_pred.apply(|all_e_p| {
        let len = all_e_p.len();
        /*quickselect_by_key(&mut all_e_p, (3*len - 1) >> 2, |(_l, _p, e)| e.error_ratio())
        .unwrap()
        .1*/
        all_e_p[(3 * len - 1) >> 2]
    });

    let all_avg: QuadErrors<ErrorPrediction> =
        quad_all_error_pred.apply(|all_e_p| all_e_p.par_iter().map(|(_l, _p, e)| e).sum());

    println!("Extracted statistics");

    let flush_single_error = ErrorStatisticsResults {
        average: StatisticsResults {
            average: all_avg.flush_single_error,
            min: (all_min.flush_single_error.2, all_min.flush_single_error.0),
            q1: (all_q1.flush_single_error.2, all_q1.flush_single_error.0),
            med: (all_med.flush_single_error.2, all_med.flush_single_error.0),
            q3: (all_q3.flush_single_error.2, all_q3.flush_single_error.0),
            max: (all_max.flush_single_error.2, all_max.flush_single_error.0),
        },
        choice: choice.flush_single_error,
    };
    let flush_dual_error = ErrorStatisticsResults {
        average: StatisticsResults {
            average: all_avg.flush_dual_error,
            min: (all_min.flush_dual_error.2, all_min.flush_dual_error.0),
            q1: (all_q1.flush_dual_error.2, all_q1.flush_dual_error.0),
            med: (all_med.flush_dual_error.2, all_med.flush_dual_error.0),
            q3: (all_q3.flush_dual_error.2, all_q3.flush_dual_error.0),
            max: (all_max.flush_dual_error.2, all_max.flush_dual_error.0),
        },
        choice: choice.flush_dual_error,
    };
    let flush_th_error = ErrorStatisticsResults {
        average: StatisticsResults {
            average: all_avg.flush_th_error,
            min: (all_min.flush_th_error.2, all_min.flush_th_error.0),
            q1: (all_q1.flush_th_error.2, all_q1.flush_th_error.0),
            med: (all_med.flush_th_error.2, all_med.flush_th_error.0),
            q3: (all_q3.flush_th_error.2, all_q3.flush_th_error.0),
            max: (all_max.flush_th_error.2, all_max.flush_th_error.0),
        },
        choice: choice.flush_th_error,
    };

    let reload_single_error = ErrorStatisticsResults {
        average: StatisticsResults {
            average: all_avg.reload_single_error,
            min: (all_min.reload_single_error.2, all_min.reload_single_error.0),
            q1: (all_q1.reload_single_error.2, all_q1.reload_single_error.0),
            med: (all_med.reload_single_error.2, all_med.reload_single_error.0),
            q3: (all_q3.reload_single_error.2, all_q3.reload_single_error.0),
            max: (all_max.reload_single_error.2, all_max.reload_single_error.0),
        },
        choice: choice.reload_single_error,
    };
    let reload_dual_error = ErrorStatisticsResults {
        average: StatisticsResults {
            average: all_avg.reload_dual_error,
            min: (all_min.reload_dual_error.2, all_min.reload_dual_error.0),
            q1: (all_q1.reload_dual_error.2, all_q1.reload_dual_error.0),
            med: (all_med.reload_dual_error.2, all_med.reload_dual_error.0),
            q3: (all_q3.reload_dual_error.2, all_q3.reload_dual_error.0),
            max: (all_max.reload_dual_error.2, all_max.reload_dual_error.0),
        },
        choice: choice.reload_dual_error,
    };

    let reload_th_error = ErrorStatisticsResults {
        average: StatisticsResults {
            average: all_avg.reload_th_error,
            min: (all_min.reload_th_error.2, all_min.reload_th_error.0),
            q1: (all_q1.reload_th_error.2, all_q1.reload_th_error.0),
            med: (all_med.reload_th_error.2, all_med.reload_th_error.0),
            q3: (all_q3.reload_th_error.2, all_q3.reload_th_error.0),
            max: (all_max.reload_th_error.2, all_max.reload_th_error.0),
        },
        choice: choice.reload_th_error,
    };

    println!("Done");

    QuadErrors {
        flush_single_error,
        flush_dual_error,
        flush_th_error,
        reload_single_error,
        reload_dual_error,
        reload_th_error,
    }
}
