use crate::{CacheOps, Output, QuadErrors, QuadThresholdErrors};
use calibration_results::calibration::{
    AVMLocation, ErrorPrediction, LocationParameters, PartialLocation, PartialLocationOwned,
};
use calibration_results::classifiers::ErrorPredictor;
use calibration_results::histograms::{SimpleBucketU64, StaticHistogram, StaticHistogramCumSum};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::sync::{Mutex, RwLock};

#[derive(Debug, Clone, Copy)]
pub struct StatisticsResults {
    pub average: ErrorPrediction,
    pub min: (ErrorPrediction, AVMLocation),
    pub q1: (ErrorPrediction, AVMLocation),
    pub med: (ErrorPrediction, AVMLocation),
    pub q3: (ErrorPrediction, AVMLocation),
    pub max: (ErrorPrediction, AVMLocation),
}

#[derive(Debug, Clone, Copy)]
pub struct ErrorStatisticsResults {
    pub average: StatisticsResults,
    pub best: (PartialLocationOwned, StatisticsResults),
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
        writeln!(
            output_file,
            "{}-Best-Location: {}",
            name.as_ref(),
            self.best.0
        )
        .expect("Failed to output");
        writeln!(
            output_file,
            "{}-Best-Average: {}",
            name.as_ref(),
            self.best.1.average
        )
        .expect("Failed to output");
        writeln!(
            output_file,
            "{}-Best-Min: {}",
            name.as_ref(),
            self.best.1.min.0
        )
        .expect("Failed to output");
        writeln!(
            output_file,
            "{}-Best-Q1: {}",
            name.as_ref(),
            self.best.1.q1.0
        )
        .expect("Failed to output");
        writeln!(
            output_file,
            "{}-Best-Med: {}",
            name.as_ref(),
            self.best.1.med.0
        )
        .expect("Failed to output");
        writeln!(
            output_file,
            "{}-Best-Q3: {}",
            name.as_ref(),
            self.best.1.q3.0
        )
        .expect("Failed to output");
        writeln!(
            output_file,
            "{}-Best-Max: {}",
            name.as_ref(),
            self.best.1.max.0
        )
        .expect("Failed to output");
    }
}

pub fn compute_statistics<const WIDTH: u64, const N: usize>(
    full_location_map: &HashMap<AVMLocation, CacheOps<StaticHistogram<{ WIDTH }, { N + N }>>>,
    projection: LocationParameters,
    thresholds_map: &HashMap<
        PartialLocationOwned,
        QuadThresholdErrors<SimpleBucketU64<WIDTH, { N + N }>>,
    >,
) -> QuadErrors<ErrorStatisticsResults> {
    assert_ne!(full_location_map.par_iter().count(), 1);

    let mut all_error_predictions: Vec<(
        AVMLocation,
        PartialLocationOwned,
        QuadErrors<ErrorPrediction>,
    )> = full_location_map
        .par_iter()
        .map(|(k, hists)| {
            let key = PartialLocationOwned::new(projection, *k);
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

            (
                *k,
                key,
                QuadErrors {
                    flush_single_error,
                    flush_dual_error,
                    reload_single_error,
                    reload_dual_error,
                },
            )
        })
        .collect();

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
        reload_single_error: all_error_predictions
            .par_iter()
            .map(|(l, p, q)| (*l, *p, q.reload_single_error))
            .collect(),
        reload_dual_error: all_error_predictions
            .par_iter()
            .map(|(l, p, q)| (*l, *p, q.reload_dual_error))
            .collect(),
    };

    // we probably want to turn this into an accumulate / reduce parallel version.
    let mut mapped = RwLock::new(HashMap::<
        PartialLocationOwned,
        Mutex<Vec<(AVMLocation, QuadErrors<ErrorPrediction>)>>,
    >::new());

    all_error_predictions.into_par_iter().for_each(|(k, p, q)| {
        let errors = q;
        let guard = mapped.read().unwrap();
        if let Some(entry) = guard.get(&p) {
            let mut inner_vec = entry.lock().unwrap();
            inner_vec.push((k, errors));
        } else {
            drop(guard);
            let mut write_guard = mapped.write().unwrap();
            write_guard
                .entry(p)
                .or_insert_with(|| Mutex::new(Vec::new()));
            drop(write_guard);
            mapped
                .read()
                .unwrap()
                .get(&p)
                .unwrap()
                .lock()
                .unwrap()
                .push((k, errors));
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

    let mut quad_two_level_error_predictions: QuadErrors<
        Vec<(
            PartialLocationOwned,
            ErrorPrediction,
            Vec<(AVMLocation, ErrorPrediction)>,
        )>,
    > = QuadErrors {
        flush_single_error: two_level_error_predictions
            .par_iter()
            .map(|(p, qv)| {
                let mut v: Vec<(AVMLocation, ErrorPrediction)> = qv
                    .par_iter()
                    .map(|(l, q)| (*l, q.flush_single_error))
                    .collect();
                v.par_sort_by_key(|(_l, e)| e.error_ratio());
                let e_avg = v.par_iter().map(|(_l, e)| e).sum();
                (*p, e_avg, v)
            })
            .collect(),
        flush_dual_error: two_level_error_predictions
            .par_iter()
            .map(|(p, qv)| {
                let mut v: Vec<(AVMLocation, ErrorPrediction)> = qv
                    .par_iter()
                    .map(|(l, q)| (*l, q.flush_dual_error))
                    .collect();
                v.par_sort_by_key(|(_l, e)| e.error_ratio());
                let e_avg = v.par_iter().map(|(_l, e)| e).sum();
                (*p, e_avg, v)
            })
            .collect(),
        reload_single_error: two_level_error_predictions
            .par_iter()
            .map(|(p, qv)| {
                let mut v: Vec<(AVMLocation, ErrorPrediction)> = qv
                    .par_iter()
                    .map(|(l, q)| (*l, q.reload_single_error))
                    .collect();
                v.par_sort_by_key(|(_l, e)| e.error_ratio());
                let e_avg = v.par_iter().map(|(_l, e)| e).sum();
                (*p, e_avg, v)
            })
            .collect(),
        reload_dual_error: two_level_error_predictions
            .par_iter()
            .map(|(p, qv)| {
                let mut v: Vec<(AVMLocation, ErrorPrediction)> = qv
                    .par_iter()
                    .map(|(l, q)| (*l, q.reload_dual_error))
                    .collect();
                v.par_sort_by_key(|(_l, e)| e.error_ratio());
                let e_avg = v.par_iter().map(|(_l, e)| e).sum();
                (*p, e_avg, v)
            })
            .collect(),
    };

    quad_two_level_error_predictions
        .apply_mut(|v| v.par_sort_by_key(|(_p, e, _v)| e.error_ratio()));

    let best_min = quad_two_level_error_predictions.apply(|v| v[0].2[0]);
    let best_max = quad_two_level_error_predictions.apply(|v| {
        let inner = &v[0].2;
        let len = inner.len();
        inner[len - 1]
    });

    let best_med = quad_two_level_error_predictions.apply(|v| {
        let inner = &v[0].2;
        let len = inner.len();
        inner[(len + 1) >> 1]
    });

    let best_q1 = quad_two_level_error_predictions.apply(|v| {
        let inner = &v[0].2;
        let len = inner.len();
        inner[(len + 1) >> 2]
    });
    let best_q3 = quad_two_level_error_predictions.apply(|v| {
        let inner = &v[0].2;
        let len = inner.len();
        inner[3 * (len + 1) >> 2]
    });

    // This one is easy, it's already computed.
    let best_avg = quad_two_level_error_predictions.apply(|v| v[0].1);

    let best_location = quad_two_level_error_predictions.apply(|v| v[0].0);

    quad_all_error_pred.apply_mut(|all_e_p| all_e_p.par_sort_by_key(|(_l, _p, e)| e.error_ratio()));

    let all_min = quad_all_error_pred.apply(|all_e_p| all_e_p[0]);
    let all_max = quad_all_error_pred.apply(|all_e_p| all_e_p[all_e_p.len() - 1]);

    let all_med = quad_all_error_pred.apply(|all_e_p| all_e_p[(all_e_p.len() + 1) >> 1]);
    let all_q1 = quad_all_error_pred.apply(|all_e_p| all_e_p[(all_e_p.len() + 1) >> 2]);
    let all_q3 = quad_all_error_pred.apply(|all_e_p| all_e_p[3 * (all_e_p.len() + 1) >> 2]);

    let all_avg: QuadErrors<ErrorPrediction> =
        quad_all_error_pred.apply(|all_e_p| all_e_p.par_iter().map(|(_l, _p, e)| e).sum());

    let flush_single_error = ErrorStatisticsResults {
        average: StatisticsResults {
            average: all_avg.flush_single_error,
            min: (all_min.flush_single_error.2, all_min.flush_single_error.0),
            q1: (all_q1.flush_single_error.2, all_q1.flush_single_error.0),
            med: (all_med.flush_single_error.2, all_med.flush_single_error.0),
            q3: (all_q3.flush_single_error.2, all_q3.flush_single_error.0),
            max: (all_max.flush_single_error.2, all_max.flush_single_error.0),
        },
        best: (
            best_location.flush_single_error,
            StatisticsResults {
                average: best_avg.flush_single_error,
                min: (best_min.flush_single_error.1, best_min.flush_single_error.0),
                q1: (best_q1.flush_single_error.1, best_q1.flush_single_error.0),
                med: (best_med.flush_single_error.1, best_med.flush_single_error.0),
                q3: (best_q3.flush_single_error.1, best_q3.flush_single_error.0),
                max: (best_max.flush_single_error.1, best_max.flush_single_error.0),
            },
        ),
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
        best: (
            best_location.flush_dual_error,
            StatisticsResults {
                average: best_avg.flush_dual_error,
                min: (best_min.flush_dual_error.1, best_min.flush_dual_error.0),
                q1: (best_q1.flush_dual_error.1, best_q1.flush_dual_error.0),
                med: (best_med.flush_dual_error.1, best_med.flush_dual_error.0),
                q3: (best_q3.flush_dual_error.1, best_q3.flush_dual_error.0),
                max: (best_max.flush_dual_error.1, best_max.flush_dual_error.0),
            },
        ),
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
        best: (
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
        ),
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
        best: (
            best_location.reload_dual_error,
            StatisticsResults {
                average: best_avg.reload_dual_error,
                min: (best_min.reload_dual_error.1, best_min.reload_dual_error.0),
                q1: (best_q1.reload_dual_error.1, best_q1.reload_dual_error.0),
                med: (best_med.reload_dual_error.1, best_med.reload_dual_error.0),
                q3: (best_q3.reload_dual_error.1, best_q3.reload_dual_error.0),
                max: (best_max.reload_dual_error.1, best_max.reload_dual_error.0),
            },
        ),
    };

    QuadErrors {
        flush_single_error,
        flush_dual_error,
        reload_single_error,
        reload_dual_error,
    }
}
