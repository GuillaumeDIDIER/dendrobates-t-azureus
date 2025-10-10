#![feature(generic_const_exprs)]
#![deny(unsafe_op_in_unsafe_fn)]

mod error_statistics;
//mod utils;

use calibration_results::calibration::{
    AVMLocation, CoreLocParameters, CoreLocation, ErrorPrediction, LocationParameters,
    PartialLocation, PartialLocationOwned,
};
use calibration_results::calibration_2t::calibration_result_to_location_map_parallel;
use calibration_results::histograms::group2_histogram_cum_sum;
use calibration_results::histograms::{
    Bucket, SimpleBucketU64, StaticHistogram, StaticHistogramCumSum,
};
use calibration_results::numa_results::NumaCalibrationResult;
use calibration_results::reduce;
use pgfplots::axis::plot::Type2D::ConstLeft;
use pgfplots::axis::plot::coordinate::Coordinate2D;
use pgfplots::axis::plot::{Plot2D, PlotKey};
use pgfplots::axis::{Axis, AxisKey};
use pgfplots::groupplot::{GroupDimension, GroupPlot};
use pgfplots::{Engine, Picture};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
//use rmp_serde::Deserializer;
//use serde::Deserialize;
use crate::error_statistics::location_parameters_json;
use calibration_results::classifiers::{
    DualThreshold, DualThresholdBuilder, ErrorPredictionsBuilder, ErrorPredictor,
    SimpleThresholdBuilder, Threshold, compute_theoretical_optimum_error,
};
use json::JsonValue;
use json::object::Object;
use num::integer::gcd;
use std::fmt::Display;
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[derive(Default)]
pub struct CacheOps<T> {
    pub flush_hit: T,
    pub flush_miss: T,
    pub reload_opt_hit: T,
    pub reload_opt_miss: T, // same as flush_miss
    pub reload_hit: T,
    pub reload_miss: T,
}

pub trait Histogram {
    fn histogram_points(&self) -> (Vec<Coordinate2D>, f64);
}

impl<const WIDTH: u64, const N: usize> Histogram for StaticHistogram<WIDTH, N> {
    fn histogram_points(&self) -> (Vec<Coordinate2D>, f64) {
        let mut ymax = 0.0;
        let mut vec: Vec<Coordinate2D> = self
            .iter()
            .map(|(bucket, value)| {
                let time: u64 = bucket.into();
                if ymax < value as f64 {
                    ymax = value as f64
                }
                Coordinate2D::from((time as f64, value as f64))
            })
            .collect();
        //let end = vec[vec.len() - 1];
        //vec.push(Coordinate2D::from((end.x, 0f64)));
        //vec.push(Coordinate2D::from((0f64, 0f64)));
        (vec, ymax)
    }
}

impl<const WIDTH: u64, const N: usize> Histogram for StaticHistogramCumSum<WIDTH, N> {
    fn histogram_points(&self) -> (Vec<Coordinate2D>, f64) {
        let mut ymax = 0.0;
        let mut vec: Vec<Coordinate2D> = self
            .iter()
            .map(|(bucket, value)| {
                let time: u64 = bucket.into();
                if ymax < value.count as f64 {
                    ymax = value.count as f64
                }
                Coordinate2D::from((time as f64, value.count as f64))
            })
            .collect();
        //let end = vec[vec.len() - 1];
        //vec.push(Coordinate2D::from((end.x, 0f64)));
        //vec.push(Coordinate2D::from((0f64, 0f64)));
        (vec, ymax)
    }
}

fn make_plot<const WIDTH: u64, const N: usize>(
    hit: StaticHistogramCumSum<WIDTH, N>,
    miss: StaticHistogramCumSum<WIDTH, N>,
) -> (Vec<Plot2D>, f64) {
    let mut hit_hist = Plot2D::new();
    let mut ymax = 0.0;
    hit_hist.add_key(PlotKey::Type2D(ConstLeft));
    {
        let (mut coordinates, ymax_local) = hit.histogram_points();
        if ymax < ymax_local {
            ymax = ymax_local
        }
        let end = coordinates[coordinates.len() - 1];
        coordinates.push(Coordinate2D::from((end.x, 0f64)));
        coordinates.push(Coordinate2D::from((0f64, 0f64)));
        hit_hist.coordinates = coordinates;
    }
    hit_hist.add_key(PlotKey::Custom(String::from(
        "draw=HistBlue, fill=HistBlueLight, very thin",
    )));

    let mut miss_hist = Plot2D::new();
    miss_hist.add_key(PlotKey::Type2D(ConstLeft));
    {
        let (coordinates, ymax_local) = miss.histogram_points();
        if ymax < ymax_local {
            ymax = ymax_local
        }
        miss_hist.coordinates = coordinates;
    }
    miss_hist.add_key(PlotKey::Custom(String::from("draw=HistRed, thin")));
    (vec![hit_hist, miss_hist], ymax)
}

pub fn make_plot_by2<const WIDTH: u64, const N: usize>(
    hit: StaticHistogramCumSum<WIDTH, { N + N }>,
    miss: StaticHistogramCumSum<WIDTH, { N + N }>,
) -> (Vec<Plot2D>, f64)
where
    [(); { WIDTH + WIDTH } as usize]:,
{
    let hit_by2: StaticHistogramCumSum<{ WIDTH + WIDTH }, N> = group2_histogram_cum_sum(hit);
    let miss_by2: StaticHistogramCumSum<{ WIDTH + WIDTH }, N> = group2_histogram_cum_sum(miss);
    make_plot::<{ WIDTH + WIDTH }, N>(hit_by2, miss_by2)
}

pub fn make_projection<const WIDTH: u64, const N: usize>(
    location_map: &HashMap<AVMLocation, CacheOps<StaticHistogram<{ WIDTH }, { N + N }>>>, // FIXME
    projection: LocationParameters,
) -> HashMap<PartialLocationOwned, CacheOps<StaticHistogramCumSum<{ WIDTH }, { N + N }>>>
//where
//    [(); { WIDTH + WIDTH } as usize]:,
{
    let shallow_copy = location_map.par_iter().collect();
    let mapped = reduce(
        // TODO, this should run in parallel if possible.
        shallow_copy,
        |avm_location| PartialLocationOwned::new(projection, *avm_location),
        || CacheOps::<StaticHistogram<WIDTH, { N + N }>>::default(),
        |acc, v, _k, _rk| {
            acc.flush_hit += &v.flush_hit;
            acc.flush_miss += &v.flush_miss;
            acc.reload_hit += &v.reload_hit;
            acc.reload_miss += &v.reload_miss;
            acc.reload_opt_hit += &v.reload_opt_hit;
            acc.reload_opt_miss += &v.reload_opt_miss;
        },
        |acc, rk| CacheOps {
            flush_hit: StaticHistogramCumSum::from(acc.flush_hit),
            flush_miss: StaticHistogramCumSum::from(acc.flush_miss),
            reload_hit: StaticHistogramCumSum::from(acc.reload_hit),
            reload_miss: StaticHistogramCumSum::from(acc.reload_miss),
            reload_opt_hit: StaticHistogramCumSum::from(acc.reload_opt_hit),
            reload_opt_miss: StaticHistogramCumSum::from(acc.reload_opt_miss),
        }, /* Figure out if we should build the HistCumSums here*/
    );

    //let count = mapped.iter().count();
    //println!(
    //    "[{}] Number of entries: {}",
    //    base_name.as_ref(),
    //    mapped.iter().count()
    //);
    mapped
}

struct MultiThresholdErrors<T: Bucket> {
    pub flush_single_threshold: (Threshold<T>, ErrorPrediction),
    pub flush_dual_threshold: (DualThreshold<T>, ErrorPrediction),
    pub reload_single_threshold: (Threshold<T>, ErrorPrediction),
    pub reload_dual_threshold: (DualThreshold<T>, ErrorPrediction),
    pub reload_opt_single_threshold: (Threshold<T>, ErrorPrediction),
    pub reload_opt_dual_threshold: (DualThreshold<T>, ErrorPrediction),
}

#[derive(Clone, Copy, Debug)]
struct MultiErrors<T> {
    pub flush_single_error: T,
    pub flush_dual_error: T,
    pub flush_th_error: T,
    pub reload_single_error: T,
    pub reload_dual_error: T,
    pub reload_th_error: T,
    pub reload_opt_single_error: T,
    pub reload_opt_dual_error: T,
    pub reload_opt_th_error: T,
}

pub trait Output {
    fn write(&self, output_file: &mut impl Write, name: impl AsRef<str>);
    fn latex_table(&self, output: &mut String, name: impl AsRef<str>);
    fn boxplot(&self, output: &mut String, name: impl AsRef<str>);
}

pub trait JsonOutput {
    fn to_json(&self, base: &Object) -> Vec<Object>;
}

impl<T: Output> Output for MultiErrors<T> {
    fn write(&self, output_file: &mut impl Write, name: impl AsRef<str>) {
        self.flush_single_error
            .write(output_file, format!("{}-FF-S", name.as_ref()));
        self.flush_dual_error
            .write(output_file, format!("{}-FF-D", name.as_ref()));
        self.flush_th_error
            .write(output_file, format!("{}-FF-Th", name.as_ref()));
        self.reload_single_error
            .write(output_file, format!("{}-FR-S", name.as_ref()));
        self.reload_dual_error
            .write(output_file, format!("{}-FR-D", name.as_ref()));
        self.reload_th_error
            .write(output_file, format!("{}-FR-Th", name.as_ref()));
        self.reload_opt_single_error
            .write(output_file, format!("{}-FRO-S", name.as_ref()));
        self.reload_opt_dual_error
            .write(output_file, format!("{}-FRO-D", name.as_ref()));
        self.reload_opt_th_error
            .write(output_file, format!("{}-FRO-Th", name.as_ref()));
    }

    fn latex_table(&self, output_file: &mut String, name: impl AsRef<str>) {
        self.flush_single_error
            .latex_table(output_file, format!("{}-FF-S", name.as_ref()));
        self.flush_dual_error
            .latex_table(output_file, format!("{}-FF-D", name.as_ref()));
        //self.flush_th_error
        //    .latex_table(output_file, format!("{}-FF-Th", name.as_ref()));
        self.reload_single_error
            .latex_table(output_file, format!("{}-FR-S", name.as_ref()));
        self.reload_dual_error
            .latex_table(output_file, format!("{}-FR-D", name.as_ref()));
        //self.reload_th_error
        //    .latex_table(output_file, format!("{}-FR-Th", name.as_ref()));
        self.reload_opt_single_error
            .latex_table(output_file, format!("{}-FRO-S", name.as_ref()));
        self.reload_opt_dual_error
            .latex_table(output_file, format!("{}-FRO-D", name.as_ref()));
        //self.reload_opt_th_error
        //    .latex_table(output_file, format!("{}-FRO-Th", name.as_ref()));
    }

    fn boxplot(&self, output: &mut String, name: impl AsRef<str>) {
        self.flush_single_error
            .boxplot(output, format!("{}-FF-S", name.as_ref()));
        self.flush_dual_error
            .boxplot(output, format!("{}-FF-D", name.as_ref()));
        //self.flush_th_error
        //    .boxplot(output, format!("{}-FF-Th", name.as_ref()));
        self.reload_single_error
            .boxplot(output, format!("{}-FR-S", name.as_ref()));
        self.reload_dual_error
            .boxplot(output, format!("{}-FR-D", name.as_ref()));
        //self.reload_th_error
        //    .boxplot(output, format!("{}-FR-Th", name.as_ref()));
        self.reload_opt_single_error
            .boxplot(output, format!("{}-FRO-S", name.as_ref()));
        self.reload_opt_dual_error
            .boxplot(output, format!("{}-FRO-D", name.as_ref()));
        //self.reload_opt_th_error
        //    .boxplot(output, format!("{}-FRO-Th", name.as_ref()));
    }
}

impl<T: Sized + Send + Sync> MultiErrors<T> {
    pub fn apply<U: Send>(&self, f: impl Sync + Fn(&T) -> U) -> MultiErrors<U> {
        let mut r = vec![
            &self.flush_single_error,
            &self.flush_dual_error,
            &self.flush_th_error,
            &self.reload_single_error,
            &self.reload_dual_error,
            &self.reload_th_error,
            &self.reload_opt_single_error,
            &self.reload_opt_dual_error,
            &self.reload_opt_th_error,
        ]
        .into_par_iter()
        .map(|e| (f(e)))
        .collect::<Vec<U>>()
        .into_iter();
        MultiErrors {
            flush_single_error: r.next().unwrap(),
            flush_dual_error: r.next().unwrap(),
            flush_th_error: r.next().unwrap(),
            reload_single_error: r.next().unwrap(),
            reload_dual_error: r.next().unwrap(),
            reload_th_error: r.next().unwrap(),
            reload_opt_single_error: r.next().unwrap(),
            reload_opt_dual_error: r.next().unwrap(),
            reload_opt_th_error: r.next().unwrap(),
        }
    }

    pub fn apply_mut<U: Send>(&mut self, f: impl Sync + Fn(&mut T) -> U) -> MultiErrors<U> {
        let mut r = vec![
            &mut self.flush_single_error,
            &mut self.flush_dual_error,
            &mut self.flush_th_error,
            &mut self.reload_single_error,
            &mut self.reload_dual_error,
            &mut self.reload_th_error,
            &mut self.reload_opt_single_error,
            &mut self.reload_opt_dual_error,
            &mut self.reload_opt_th_error,
        ]
        .into_par_iter()
        .map(|e| (f(e)))
        .collect::<Vec<U>>()
        .into_iter();
        MultiErrors {
            flush_single_error: r.next().unwrap(),
            flush_dual_error: r.next().unwrap(),
            flush_th_error: r.next().unwrap(),
            reload_single_error: r.next().unwrap(),
            reload_dual_error: r.next().unwrap(),
            reload_th_error: r.next().unwrap(),
            reload_opt_single_error: r.next().unwrap(),
            reload_opt_dual_error: r.next().unwrap(),
            reload_opt_th_error: r.next().unwrap(),
        }
    }

    pub fn map<U: Send>(self, f: impl Sync + Fn(T) -> U) -> MultiErrors<U> {
        let mut r = vec![
            self.flush_single_error,
            self.flush_dual_error,
            self.flush_th_error,
            self.reload_single_error,
            self.reload_dual_error,
            self.reload_th_error,
            self.reload_opt_single_error,
            self.reload_opt_dual_error,
            self.reload_opt_th_error,
        ]
        .into_par_iter()
        .map(|e| (f(e)))
        .collect::<Vec<U>>()
        .into_iter();
        MultiErrors {
            flush_single_error: r.next().unwrap(),
            flush_dual_error: r.next().unwrap(),
            flush_th_error: r.next().unwrap(),
            reload_single_error: r.next().unwrap(),
            reload_dual_error: r.next().unwrap(),
            reload_th_error: r.next().unwrap(),
            reload_opt_single_error: r.next().unwrap(),
            reload_opt_dual_error: r.next().unwrap(),
            reload_opt_th_error: r.next().unwrap(),
        }
    }
}

impl<T: JsonOutput> JsonOutput for MultiErrors<T> {
    fn to_json(&self, base: &Object) -> Vec<Object> {
        let mut result = Vec::new();
        let mut ff_s_base = base.clone();
        ff_s_base.insert("method", JsonValue::String(String::from("FF")));
        ff_s_base.insert("classifier", JsonValue::String(String::from("S")));
        result.extend(self.flush_single_error.to_json(&ff_s_base));

        let mut ff_d_base = base.clone();
        ff_d_base.insert("method", JsonValue::String(String::from("FF")));
        ff_d_base.insert("classifier", JsonValue::String(String::from("D")));
        result.extend(self.flush_dual_error.to_json(&ff_d_base));

        let mut fr_s_base = base.clone();
        fr_s_base.insert("method", JsonValue::String(String::from("FR")));
        fr_s_base.insert("classifier", JsonValue::String(String::from("S")));
        result.extend(self.reload_single_error.to_json(&fr_s_base));

        let mut fr_d_base = base.clone();
        fr_d_base.insert("method", JsonValue::String(String::from("FR")));
        fr_d_base.insert("classifier", JsonValue::String(String::from("D")));
        result.extend(self.reload_dual_error.to_json(&fr_d_base));

        let mut fr_s_base = base.clone();
        fr_s_base.insert("method", JsonValue::String(String::from("FRO")));
        fr_s_base.insert("classifier", JsonValue::String(String::from("S")));
        result.extend(self.reload_opt_single_error.to_json(&fr_s_base));

        let mut fr_d_base = base.clone();
        fr_d_base.insert("method", JsonValue::String(String::from("FRO")));
        fr_d_base.insert("classifier", JsonValue::String(String::from("D")));
        result.extend(self.reload_opt_dual_error.to_json(&fr_d_base));

        let mut ff_t_base = base.clone();
        ff_t_base.insert("method", JsonValue::String(String::from("FF")));
        ff_t_base.insert("classifier", JsonValue::String(String::from("T")));
        result.extend(self.flush_th_error.to_json(&ff_t_base));

        let mut fr_t_base = base.clone();
        fr_t_base.insert("method", JsonValue::String(String::from("FR")));
        fr_t_base.insert("classifier", JsonValue::String(String::from("T")));
        result.extend(self.reload_th_error.to_json(&fr_t_base));

        let mut fr_t_base = base.clone();
        fr_t_base.insert("method", JsonValue::String(String::from("FRO")));
        fr_t_base.insert("classifier", JsonValue::String(String::from("T")));
        result.extend(self.reload_opt_th_error.to_json(&fr_t_base));
        result
    }
}

fn write_out_threshold_info<T: Bucket>(
    mut out: impl Write,
    basename: &str,
    thresholds: &MultiThresholdErrors<T>,
    // thresholds: &(QuadThresholdErrors<T>, TheoreticalError<ErrorPrediction>),
) -> std::io::Result<()>
where
    Threshold<T>: Display,
    DualThreshold<T>: Display,
{
    writeln!(
        out,
        "{}-FF-Single: {}, Error prediction: {}",
        basename, thresholds.flush_single_threshold.0, thresholds.flush_single_threshold.1
    )?;
    writeln!(
        out,
        "{}-FF-Dual: {}, Error prediction: {} ",
        basename, thresholds.flush_dual_threshold.0, thresholds.flush_dual_threshold.1
    )?;
    /*writeln!(
        out,
        "{}-FF-Th: Error prediction: {} ",
        basename, thresholds.1.flush_flush
    )?;*/
    writeln!(
        out,
        "{}-FR-Single: {}, Error prediction: {} ",
        basename, thresholds.reload_single_threshold.0, thresholds.reload_single_threshold.1
    )?;
    writeln!(
        out,
        "{}-FR-Dual: {}, Error prediction: {} ",
        basename, thresholds.reload_dual_threshold.0, thresholds.reload_dual_threshold.1
    )?; /*?;
    writeln!(
    out,
    "{}-FR-Th: Error prediction: {} ",
    basename, thresholds.1.flush_reload
    )*/
    writeln!(
        out,
        "{}-FRO-Single: {}, Error prediction: {} ",
        basename,
        thresholds.reload_opt_single_threshold.0,
        thresholds.reload_opt_single_threshold.1
    )?;
    writeln!(
        out,
        "{}-FRO-Dual: {}, Error prediction: {} ",
        basename, thresholds.reload_opt_dual_threshold.0, thresholds.reload_opt_dual_threshold.1
    ) /*?;
    writeln!(
    out,
    "{}-FRO-Th: Error prediction: {} ",
    basename, thresholds.1.flush_reload
    )*/
}

fn write_out_multi_errors<T>(
    mut out: impl Write,
    basename: &str,
    multi_errors: &MultiErrors<T>,
) -> std::io::Result<()>
where
    T: Display,
{
    writeln!(
        out,
        "{}-FF-Single-Error: {}",
        basename, multi_errors.flush_single_error
    )?;
    writeln!(
        out,
        "{}-FF-Dual-Error: {}",
        basename, multi_errors.flush_dual_error
    )?;
    writeln!(
        out,
        "{}-FR-Single-Error: {}",
        basename, multi_errors.reload_single_error
    )?;
    writeln!(
        out,
        "{}-FR-Dual-Error: {}",
        basename, multi_errors.reload_dual_error
    )?;
    writeln!(
        out,
        "{}-FRO-Single-Error: {}",
        basename, multi_errors.reload_opt_single_error
    )?;
    writeln!(
        out,
        "{}-FRO-Dual-Error: {}",
        basename, multi_errors.reload_opt_dual_error
    )
}

fn compute_errors<const WIDTH: u64, const N: usize>(
    projected: &HashMap<
        PartialLocationOwned,
        CacheOps<StaticHistogramCumSum<{ WIDTH }, { N + N }>>,
    >,
) -> HashMap<
    PartialLocationOwned,
    //(
    MultiThresholdErrors<SimpleBucketU64<WIDTH, { N + N }>>,
    //    TheoreticalError<ErrorPrediction>,
    //),
> {
    let simple_threshold_builder: SimpleThresholdBuilder<WIDTH, { N + N }> =
        SimpleThresholdBuilder();
    let dual_threshold_builder: DualThresholdBuilder<WIDTH, { N + N }> = DualThresholdBuilder();

    projected
        .par_iter()
        .map(|(k, histograms)| {
            let flush_single_threshold = simple_threshold_builder
                .find_best_classifier(&histograms.flush_hit, &histograms.flush_miss)
                .expect("Failed to compute single threshold");
            let flush_dual_threshold = dual_threshold_builder
                .find_best_classifier(&histograms.flush_hit, &histograms.flush_miss)
                .expect("Failed to compute dual threshold");

            let reload_single_threshold = simple_threshold_builder
                .find_best_classifier(&histograms.reload_hit, &histograms.reload_miss)
                .expect("Failed to compute single threshold");
            let reload_dual_threshold = dual_threshold_builder
                .find_best_classifier(&histograms.reload_hit, &histograms.reload_miss)
                .expect("Failed to compute dual threshold");
            let reload_opt_single_threshold = simple_threshold_builder
                .find_best_classifier(&histograms.reload_opt_hit, &histograms.reload_opt_miss)
                .expect("Failed to compute single threshold");
            let reload_opt_dual_threshold = dual_threshold_builder
                .find_best_classifier(&histograms.reload_opt_hit, &histograms.reload_opt_miss)
                .expect("Failed to compute dual threshold");
            /*
            let flush_flush_optimal =
                compute_theoretical_optimum_error(&histograms.flush_hit, &histograms.flush_miss);
            let flush_reload_optimal =
                compute_theoretical_optimum_error(&histograms.reload_hit, &histograms.reload_miss);
            */
            (
                *k,
                //(
                MultiThresholdErrors {
                    flush_single_threshold,
                    flush_dual_threshold,
                    reload_single_threshold,
                    reload_dual_threshold,
                    reload_opt_single_threshold,
                    reload_opt_dual_threshold,
                }, /*,
                       TheoreticalError {
                           flush_flush: flush_flush_optimal,
                           flush_reload: flush_reload_optimal,
                       },
                   ),*/
            )
        })
        .collect()
}

fn compute_errors_no_projection<const WIDTH: u64, const N: usize>(
    histogram_map: &HashMap<AVMLocation, CacheOps<StaticHistogram<{ WIDTH }, { N + N }>>>,
) -> HashMap<
    AVMLocation,
    //(
    MultiThresholdErrors<SimpleBucketU64<WIDTH, { N + N }>>,
    //    TheoreticalError<ErrorPrediction>,
    //),
> {
    let simple_threshold_builder: SimpleThresholdBuilder<WIDTH, { N + N }> =
        SimpleThresholdBuilder();
    let dual_threshold_builder: DualThresholdBuilder<WIDTH, { N + N }> = DualThresholdBuilder();

    histogram_map
        .par_iter()
        .map(|(k, histograms)| {
            let flush_hits = StaticHistogramCumSum::from(&histograms.flush_hit);
            let flush_miss = StaticHistogramCumSum::from(&histograms.flush_miss);

            let reload_hits = StaticHistogramCumSum::from(&histograms.reload_hit);
            let reload_miss = StaticHistogramCumSum::from(&histograms.reload_miss);
            let reload_opt_hits = StaticHistogramCumSum::from(&histograms.reload_opt_hit);
            let reload_opt_miss = StaticHistogramCumSum::from(&histograms.reload_opt_miss);
            let flush_single_threshold = simple_threshold_builder
                .find_best_classifier(&flush_hits, &flush_miss)
                .expect("Failed to compute single threshold");
            let flush_dual_threshold = dual_threshold_builder
                .find_best_classifier(&flush_hits, &flush_miss)
                .expect("Failed to compute dual threshold");

            let reload_single_threshold = simple_threshold_builder
                .find_best_classifier(&reload_hits, &reload_miss)
                .expect("Failed to compute single threshold");
            let reload_dual_threshold = dual_threshold_builder
                .find_best_classifier(&reload_hits, &reload_miss)
                .expect("Failed to compute dual threshold");
            let reload_opt_single_threshold = simple_threshold_builder
                .find_best_classifier(&reload_opt_hits, &reload_opt_miss)
                .expect("Failed to compute single threshold");
            let reload_opt_dual_threshold = dual_threshold_builder
                .find_best_classifier(&reload_opt_hits, &reload_opt_miss)
                .expect("Failed to compute dual threshold");
            /*
            let flush_flush_optimal =
                compute_theoretical_optimum_error(&histograms.flush_hit, &histograms.flush_miss);
            let flush_reload_optimal =
                compute_theoretical_optimum_error(&histograms.reload_hit, &histograms.reload_miss);
            */
            (
                *k,
                //(
                MultiThresholdErrors {
                    flush_single_threshold,
                    flush_dual_threshold,
                    reload_single_threshold,
                    reload_dual_threshold,
                    reload_opt_single_threshold,
                    reload_opt_dual_threshold,
                }, /*,
                       TheoreticalError {
                           flush_flush: flush_flush_optimal,
                           flush_reload: flush_reload_optimal,
                       },
                   ),*/
            )
        })
        .collect()
}

fn make_projection_plots<const WIDTH: u64, const N: usize>(
    projected: HashMap<PartialLocationOwned, CacheOps<StaticHistogramCumSum<{ WIDTH }, { N + N }>>>,
    folder: impl AsRef<Path>,
    base_name: impl AsRef<str>,
    flush_flush: bool,
    flush_reload: bool,
    flush_reload_opt: bool,
    group_dimension: GroupDimension,
    sort_criteria: &fn(PartialLocationOwned, PartialLocationOwned) -> Ordering,
    /* Thresholding Strategies */
    /* Filter */
) -> Result<(), ()>
where
    [(); { WIDTH + WIDTH } as usize]:,
{
    let mut picture_ff = Picture::new();
    let mut picture_fr = Picture::new();
    let mut picture_fro = Picture::new(); // TODO

    let count = projected.par_iter().count();
    println!("[{}] Number of entries: {}", base_name.as_ref(), count);

    if count == 0 {
        return Err(());
    }
    if count == 1 {
        match group_dimension {
            GroupDimension::Horizontal(1)
            | GroupDimension::Vertical(1)
            | GroupDimension::Rectangle(1, 1) => {}
            GroupDimension::Horizontal(_)
            | GroupDimension::Vertical(_)
            | GroupDimension::Rectangle(_, _) => {
                eprintln!(
                    "[{}] Invalid dimension: {:?}",
                    base_name.as_ref(),
                    group_dimension
                );
                return Err(());
            }
        }
        let histograms = projected.into_iter().next().unwrap().1;

        let (mut flush_plots, flush_ymax) =
            make_plot_by2(histograms.flush_hit, histograms.flush_miss);
        let mut flush_hist = Axis::new();
        flush_hist.plots.append(&mut flush_plots);

        flush_hist.set_title("Flush+Flush"); // FIXME, this should be customizable by the caller
        flush_hist.add_key(AxisKey::Custom(format!(
            "height=10cm, width=20cm, xmin=0,xmax={}, ymin=0, tick align=outside, tick pos=left, axis on top,", N+N//
        )));

        let mut flush_single_threshold_plot = Plot2D::new();

        /*
        let flush_single_threshold_x: u64 = flush_single_threshold.0.bucket_index.into();
        flush_single_threshold_plot.coordinates = vec![
            Coordinate2D::from((flush_single_threshold_x as f64, 0.)),
            Coordinate2D::from((flush_single_threshold_x as f64, flush_ymax)),
        ];*/

        /*
                let mut flush_dual_threshold_plot_1 = Plot2D::new();
                let flush_dual_threshold_x = flush_dual_threshold.0.get_thresholds();
                let flush_dual_threshold_x1: u64 = flush_dual_threshold_x.0.into();

                flush_dual_threshold_plot_1.coordinates = vec![
                    Coordinate2D::from((flush_dual_threshold_x1 as f64, 0.)),
                    Coordinate2D::from((flush_dual_threshold_x1 as f64, flush_ymax)),
                ];

                let mut flush_dual_threshold_plot_2 = Plot2D::new();
                let flush_dual_threshold_x2: u64 = flush_dual_threshold_x.1.into();

                flush_dual_threshold_plot_2.coordinates = vec![
                    Coordinate2D::from((flush_dual_threshold_x2 as f64, 0.)),
                    Coordinate2D::from((flush_dual_threshold_x2 as f64, flush_ymax)),
                ];

                flush_hist.plots.push(flush_single_threshold_plot);
                flush_hist.plots.push(flush_dual_threshold_plot_1);
                flush_hist.plots.push(flush_dual_threshold_plot_2);
        */
        picture_ff.axes.push(Box::new(flush_hist));
        picture_ff.add_to_preamble(vec![String::from(
            r#"\definecolor{HistRed}{HTML}{E41A1C}
\definecolor{HistRedLight}{HTML}{FAB4AE}
\definecolor{HistBlue}{HTML}{377EB8}
\definecolor{HistBlueLight}{HTML}{B3CDE3}
"#,
        )]);
        let picture_ff_jobname = format!("{}-FF", base_name.as_ref());
        let picture_ff_jobname_tex = format!("{}-FF.tex", base_name.as_ref());

        std::fs::write(
            folder.as_ref().join(picture_ff_jobname_tex),
            picture_ff.standalone_string(),
        )
        .expect("Failed to write plot");
        picture_ff
            .to_pdf(folder.as_ref(), &picture_ff_jobname, Engine::LuaLatex)
            .inspect_err(|e| {
                eprintln!("Failed to create {}: {}", picture_ff_jobname, e);
            });

        // ------ Reload -------------
        /*
        let reload_single_threshold = simple_threshold_builder
            .find_best_classifier(&histograms.reload_hit, &histograms.reload_miss)
            .expect("Failed to compute single threshold");
        let reload_dual_threshold = dual_threshold_builder
            .find_best_classifier(&histograms.reload_hit, &histograms.reload_miss)
            .expect("Failed to compute dual threshold");

        println!(
            "[{}] Reload Single Threshold: {}, Error Prediction: {}",
            base_name.as_ref(),
            reload_single_threshold.0,
            reload_single_threshold.1
        );
        println!(
            "[{}] Reload Dual Threshold: {}, Error Prediction: {}",
            base_name.as_ref(),
            reload_dual_threshold.0,
            reload_dual_threshold.1
        );
        */
        let (mut reload_plots, reload_ymax) =
            make_plot_by2(histograms.reload_hit, histograms.reload_miss);
        let mut reload_hist = Axis::new();
        reload_hist.plots.append(&mut reload_plots);

        reload_hist.set_title("Flush+Reload"); // FIXME, this should be customizable by the caller
        reload_hist.add_key(AxisKey::Custom(format!(
            "height=10cm, width=20cm, xmin=0,xmax={}, ymin=0, tick align=outside, tick pos=left, axis on top,", N+N//
        )));
        /*
        let mut reload_single_threshold_plot = Plot2D::new();

        let reload_single_threshold_x: u64 = reload_single_threshold.0.bucket_index.into();
        reload_single_threshold_plot.coordinates = vec![
            Coordinate2D::from((reload_single_threshold_x as f64, 0.)),
            Coordinate2D::from((reload_single_threshold_x as f64, reload_ymax)),
        ];

        let mut reload_dual_threshold_plot_1 = Plot2D::new();
        let reload_dual_threshold_x = reload_dual_threshold.0.get_thresholds();
        let reload_dual_threshold_x1: u64 = reload_dual_threshold_x.0.into();

        reload_dual_threshold_plot_1.coordinates = vec![
            Coordinate2D::from((reload_dual_threshold_x1 as f64, 0.)),
            Coordinate2D::from((reload_dual_threshold_x1 as f64, reload_ymax)),
        ];

        let mut reload_dual_threshold_plot_2 = Plot2D::new();
        let reload_dual_threshold_x2: u64 = reload_dual_threshold_x.1.into();

        reload_dual_threshold_plot_2.coordinates = vec![
            Coordinate2D::from((reload_dual_threshold_x2 as f64, 0.)),
            Coordinate2D::from((reload_dual_threshold_x2 as f64, reload_ymax)),
        ];

        reload_hist.plots.push(reload_single_threshold_plot);
        reload_hist.plots.push(reload_dual_threshold_plot_1);
        reload_hist.plots.push(reload_dual_threshold_plot_2);
        */
        picture_fr.axes.push(Box::new(reload_hist));
        picture_fr.add_to_preamble(vec![String::from(
            r#"\definecolor{HistRed}{HTML}{E41A1C}
\definecolor{HistRedLight}{HTML}{FAB4AE}
\definecolor{HistBlue}{HTML}{377EB8}
\definecolor{HistBlueLight}{HTML}{B3CDE3}
"#,
        )]);
        let picture_fr_jobname = format!("{}-FR", base_name.as_ref());
        let picture_fr_jobname_tex = format!("{}-FR.tex", base_name.as_ref());

        std::fs::write(
            folder.as_ref().join(picture_fr_jobname_tex),
            picture_fr.standalone_string(),
        )
        .expect("Failed to write plot");
        picture_fr
            .to_pdf(folder.as_ref(), picture_fr_jobname, Engine::LuaLatex)
            .inspect_err(|e| {
                eprintln!("Failed to create {}: {}", picture_ff_jobname, e);
            });

        // ------ Reload Opt -------------

        let (mut reload_opt_plots, reload_opt_ymax) =
            make_plot_by2(histograms.reload_opt_hit, histograms.reload_opt_miss);
        let mut reload_opt_hist = Axis::new();
        reload_opt_hist.plots.append(&mut reload_opt_plots);

        reload_opt_hist.set_title("Flush+Reload Opt"); // FIXME, this should be customizable by the caller
        reload_opt_hist.add_key(AxisKey::Custom(format!(
            "height=10cm, width=20cm, xmin=0,xmax={}, ymin=0, tick align=outside, tick pos=left, axis on top,", N+N//
        )));

        picture_fro.axes.push(Box::new(reload_opt_hist));
        picture_fro.add_to_preamble(vec![String::from(
            r#"\definecolor{HistRed}{HTML}{E41A1C}
\definecolor{HistRedLight}{HTML}{FAB4AE}
\definecolor{HistBlue}{HTML}{377EB8}
\definecolor{HistBlueLight}{HTML}{B3CDE3}
"#,
        )]);
        let picture_fro_jobname = format!("{}-FRO", base_name.as_ref());
        let picture_fro_jobname_tex = format!("{}-FRO.tex", base_name.as_ref());

        std::fs::write(
            folder.as_ref().join(picture_fro_jobname_tex),
            picture_fro.standalone_string(),
        )
        .expect("Failed to write plot");
        picture_fro
            .to_pdf(folder.as_ref(), picture_fro_jobname, Engine::LuaLatex)
            .inspect_err(|e| {
                eprintln!("Failed to create {}: {}", picture_ff_jobname, e);
            });

        // -----------------------------------------------------------------------------------------
    } else {
        // -----------------------------------------------------------------------------------------
        /* count > 1 */
        let mut sorted_histograms: Vec<(_, _)> = projected.into_par_iter().collect();
        sorted_histograms.sort_by(
            |arg0: &(
                PartialLocationOwned,
                CacheOps<StaticHistogramCumSum<WIDTH, { N + N }>>,
            ),
             arg1: &(
                PartialLocationOwned,
                CacheOps<StaticHistogramCumSum<WIDTH, { N + N }>>,
            )| sort_criteria(arg0.0, arg1.0),
        );
        // TODO figure out how to handle the multiple plots.

        let mut flush_group = GroupPlot::new();
        flush_group.dimension = group_dimension;
        let mut reload_group = GroupPlot::new();
        reload_group.dimension = group_dimension;
        let mut reload_opt_group = GroupPlot::new();
        reload_opt_group.dimension = group_dimension;

        let mut ymax_flush = 0.0;
        let mut ymax_reload = 0.0;
        let mut ymax_reload_opt = 0.0;

        for item in sorted_histograms.into_iter().enumerate() {
            let mut flush_axis = Axis::new();
            let mut reload_axis = Axis::new();
            let mut reload_opt_axis = Axis::new();

            let (mut plots_flush, ymax_local) =
                make_plot_by2(item.1.1.flush_hit, item.1.1.flush_miss);
            if ymax_flush < ymax_local {
                ymax_flush = ymax_local
            }
            let (mut plots_reload, ymax_local) =
                make_plot_by2(item.1.1.reload_hit, item.1.1.reload_miss);
            if ymax_reload < ymax_local {
                ymax_reload = ymax_local
            }
            let (mut plots_reload_opt, ymax_local) =
                make_plot_by2(item.1.1.reload_opt_hit, item.1.1.reload_opt_miss);
            if ymax_reload_opt < ymax_local {
                ymax_reload_opt = ymax_local
            }

            // TODO costumize the title :/
            // Probably through a closure mapping location -> name
            flush_axis.plots.append(&mut plots_flush);
            flush_axis.set_title(format!(
                "Location: A: {}, V: {}, N: {}",
                item.1.0.get_attacker_socket().unwrap(),
                item.1.0.get_victim_socket().unwrap(),
                item.1.0.get_numa_node().unwrap()
            ));
            reload_axis.plots.append(&mut plots_reload);
            reload_axis.set_title(format!(
                "Location: A: {}, V: {}, N: {}",
                item.1.0.get_attacker_socket().unwrap(),
                item.1.0.get_victim_socket().unwrap(),
                item.1.0.get_numa_node().unwrap()
            ));
            reload_opt_axis.plots.append(&mut plots_reload_opt);
            reload_opt_axis.set_title(format!(
                "Location: A: {}, V: {}, N: {}",
                item.1.0.get_attacker_socket().unwrap(),
                item.1.0.get_victim_socket().unwrap(),
                item.1.0.get_numa_node().unwrap()
            ));
            println!(
                "Location: A: {}, V: {}, N: {}",
                item.1.0.get_attacker_socket().unwrap(),
                item.1.0.get_victim_socket().unwrap(),
                item.1.0.get_numa_node().unwrap()
            );
            //let index_1 = item.0 / 4;
            //let index_2 = item.0 % 4;
            flush_group.groups.push(flush_axis);
            reload_group.groups.push(reload_axis);
            reload_opt_group.groups.push(reload_opt_axis);
        }
        flush_group.set_title("FF TBD");
        flush_group.add_key(AxisKey::Custom(format!("height=10cm, width=20cm, xmin=0,xmax={}, ymin=0, ymax={}, tick align=outside, tick pos=left, axis on top,", N+N, ymax_flush)));

        picture_ff.axes.push(Box::new(flush_group));
        picture_ff.add_to_preamble(vec![String::from(
            r#"\definecolor{HistRed}{HTML}{E41A1C}
\definecolor{HistRedLight}{HTML}{FAB4AE}
\definecolor{HistBlue}{HTML}{377EB8}
\definecolor{HistBlueLight}{HTML}{B3CDE3}
"#,
        )]);
        let picture_ff_jobname = format!("{}-FF", base_name.as_ref());
        let picture_ff_jobname_tex = format!("{}-FF.tex", base_name.as_ref());

        std::fs::write(
            folder.as_ref().join(picture_ff_jobname_tex),
            picture_ff.standalone_string(),
        )
        .expect("Failed to write plot");
        match picture_ff.to_pdf(folder.as_ref(), &picture_ff_jobname, Engine::LuaLatex) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Failed to create PDF: {}, {:?}", picture_ff_jobname, e)
            }
        }

        //---------

        reload_group.set_title("FR-TBD");
        reload_group.add_key(AxisKey::Custom(format!("height=10cm, width=20cm, xmin=0,xmax={}, ymin=0, ymax={}, tick align=outside, tick pos=left, axis on top,", N+N, ymax_reload)));

        picture_fr.axes.push(Box::new(reload_group));
        picture_fr.add_to_preamble(vec![String::from(
            r#"\definecolor{HistRed}{HTML}{E41A1C}
\definecolor{HistRedLight}{HTML}{FAB4AE}
\definecolor{HistBlue}{HTML}{377EB8}
\definecolor{HistBlueLight}{HTML}{B3CDE3}
"#,
        )]);
        let picture_fr_jobname = format!("{}-FR", base_name.as_ref());
        let picture_fr_jobname_tex = format!("{}-FR.tex", base_name.as_ref());

        std::fs::write(
            folder.as_ref().join(picture_fr_jobname_tex),
            picture_fr.standalone_string(),
        )
        .expect("Failed to write plot");
        match picture_fr.to_pdf(folder.as_ref(), &picture_fr_jobname, Engine::LuaLatex) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Failed to create PDF: {}, {:?}", picture_fr_jobname, e)
            }
        }

        //---------

        reload_opt_group.set_title("FR-TBD");
        reload_opt_group.add_key(AxisKey::Custom(format!("height=10cm, width=20cm, xmin=0,xmax={}, ymin=0, ymax={}, tick align=outside, tick pos=left, axis on top,", N+N, ymax_reload_opt)));

        picture_fro.axes.push(Box::new(reload_opt_group));
        picture_fro.add_to_preamble(vec![String::from(
            r#"\definecolor{HistRed}{HTML}{E41A1C}
\definecolor{HistRedLight}{HTML}{FAB4AE}
\definecolor{HistBlue}{HTML}{377EB8}
\definecolor{HistBlueLight}{HTML}{B3CDE3}
"#,
        )]);
        let picture_fro_jobname = format!("{}-FRO", base_name.as_ref());
        let picture_fro_jobname_tex = format!("{}-FRO.tex", base_name.as_ref());

        std::fs::write(
            folder.as_ref().join(picture_fro_jobname_tex),
            picture_fro.standalone_string(),
        )
        .expect("Failed to write plot");
        match picture_fro.to_pdf(folder.as_ref(), &picture_fro_jobname, Engine::LuaLatex) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Failed to create PDF: {}, {:?}", picture_fr_jobname, e)
            }
        }
    }

    Ok(())
}

fn ordering_all(a: PartialLocationOwned, b: PartialLocationOwned) -> std::cmp::Ordering {
    Ordering::Equal
}

fn ordering_numa(a: PartialLocationOwned, b: PartialLocationOwned) -> Ordering {
    match a.get_numa_node().unwrap().cmp(&b.get_numa_node().unwrap()) {
        Ordering::Less => Ordering::Less,
        Ordering::Greater => Ordering::Greater,
        Ordering::Equal => {
            match a
                .get_attacker_socket()
                .unwrap()
                .cmp(&b.get_attacker_socket().unwrap())
            {
                Ordering::Less => Ordering::Less,
                Ordering::Greater => Ordering::Greater,
                Ordering::Equal => a
                    .get_victim_socket()
                    .unwrap()
                    .cmp(&b.get_victim_socket().unwrap()),
            }
        }
    }
}

struct Statistics<T> {
    min: T,
    max: T,
    med: T,
    q1: T,
    q3: T,
    avg: T,
}

fn build_statistics(
    mut multi_series: MultiErrors<Vec<(ErrorPrediction, PartialLocationOwned)>>,
) -> Statistics<MultiErrors<ErrorPrediction>> {
    multi_series.apply_mut(|series| series.sort_by_key(|(e, _l)| e.error_ratio()));

    let count = multi_series.flush_single_error.len();

    let min = multi_series.apply(|series| series[0].0);
    let max = multi_series.apply(|series| {
        let len = series.len();
        series[len - 1].0
    });
    let med = multi_series.apply(|series| {
        let len = series.len();
        series[(len - 1) >> 1].0
    });
    let q1 = multi_series.apply(|series| {
        let len = series.len();
        series[(len - 1) >> 2].0
    });
    let q3 = multi_series.apply(|series| {
        let len = series.len();
        series[(3 * len - 1) >> 2].0
    });

    let avg = multi_series.map(|series| series.into_par_iter().map(|(e, _p)| e).sum());

    Statistics {
        min,
        max,
        med,
        q1,
        q3,
        avg,
    }
}

pub fn run_analysis<const WIDTH: u64, const N: usize>(
    data: NumaCalibrationResult<WIDTH, { N + N }>,
    folder: impl AsRef<Path>,
    basename: impl AsRef<str>,
) -> ()
where
    [(); { WIDTH + WIDTH } as usize]:,
{
    // First, let's extract the data we want.

    println!(
        "Running analysis with folder: {} and basename: {}",
        folder.as_ref().display(),
        basename.as_ref()
    );
    /* We need dual and single threshold analysis */
    let results = data.results;
    let topology_info = data.topology_info;
    let slicings = data.slicing;
    let operations = data.operations;

    let names = CacheOps {
        flush_hit: HashSet::from([String::from("clflush_remote_hit")]),
        flush_miss: HashSet::from([
            String::from("clflush_miss_f"),
            //String::from("clflush_miss_n"),
        ]),
        reload_hit: HashSet::from([String::from("reload_remote_hit")]),
        reload_miss: HashSet::from([String::from("reload_miss_n")]),
        reload_opt_hit: HashSet::from([String::from("shared_hit")]),
        reload_opt_miss: HashSet::from([String::from("reload_miss_f")]),
    };

    let mut indexes = CacheOps {
        flush_hit: HashSet::new(),
        flush_miss: HashSet::new(),
        reload_hit: HashSet::new(),
        reload_miss: HashSet::new(),
        reload_opt_hit: HashSet::new(),
        reload_opt_miss: HashSet::new(),
    };

    for (i, name) in operations.iter().enumerate() {
        if names.flush_hit.contains(&name.name) {
            indexes.flush_hit.insert(i);
        }
        if names.flush_miss.contains(&name.name) {
            indexes.flush_miss.insert(i);
        }
        if names.reload_hit.contains(&name.name) {
            indexes.reload_hit.insert(i);
        }
        if names.reload_miss.contains(&name.name) {
            indexes.reload_miss.insert(i);
        }
        if names.reload_opt_hit.contains(&name.name) {
            indexes.reload_opt_hit.insert(i);
        }
        if names.reload_opt_miss.contains(&name.name) {
            indexes.reload_opt_miss.insert(i);
        }
    }

    let uarch = data.micro_architecture;

    let core_location = |core: usize| {
        // Eventually we need to integrate https://docs.rs/raw-cpuid/latest/raw_cpuid/struct.ExtendedTopologyIter.html
        let node = topology_info[&core].into();
        CoreLocation {
            socket: node,
            core: core as u16,
        }
    };

    // Build the Location Map from the raw structures

    let location_map = calibration_result_to_location_map_parallel(
        results,
        &|static_hist_result| {
            let mut result = CacheOps::<StaticHistogram<WIDTH, { N + N }>>::default();
            for (i, hist) in static_hist_result.histogram.into_iter().enumerate() {
                if indexes.flush_hit.contains(&i) {
                    result.flush_hit += &hist;
                }
                if indexes.flush_miss.contains(&i) {
                    result.flush_miss += &hist;
                }
                if indexes.reload_hit.contains(&i) {
                    result.reload_hit += &hist;
                }
                if indexes.reload_miss.contains(&i) {
                    result.reload_miss += &hist;
                }
                if indexes.reload_opt_hit.contains(&i) {
                    result.reload_opt_hit += &hist;
                }
                if indexes.reload_opt_miss.contains(&i) {
                    result.reload_opt_miss += &hist;
                }
            }
            result
        },
        &|addr| {
            slicings
                .1
                .hash(addr)
                .try_into()
                .expect("Slice index doesn't fit u8")
        },
        &core_location,
    );

    // Print some metadata
    let num_entries = location_map.iter().count();
    println!("Number of entries: {}", num_entries);

    let numa_nodes_set: HashSet<_> = location_map.keys().map(|l| l.memory_numa_node).collect();

    let attacker_socket_set: HashSet<_> = location_map.keys().map(|l| l.attacker.socket).collect();
    let victim_socket_set: HashSet<_> = location_map.keys().map(|l| l.victim.socket).collect();

    let attacker_core_set: HashSet<_> = location_map.keys().map(|l| l.attacker.core).collect();
    let victim_core_set: HashSet<_> = location_map.keys().map(|l| l.victim.core).collect();

    let target_set: HashSet<_> = location_map
        .keys()
        .map(|l| (l.memory_numa_node, l.memory_offset))
        .collect();

    let numa_node_count = numa_nodes_set.iter().count();
    println!("Number of Numa Nodes: {}", numa_node_count);
    let attacker_socket_count = attacker_socket_set.iter().count();
    println!("Number of Attacker Socket: {}", attacker_socket_count);
    let victim_socket_count = victim_socket_set.iter().count();
    println!("Number of Victim Sockets: {}", victim_socket_count);

    let attacker_core_count = attacker_core_set.iter().count();
    println!("Number of Attacker Core: {}", attacker_core_count);
    let victim_core_count = victim_core_set.iter().count();
    println!("Number of Victim Core: {}", victim_core_count);

    let distinct_target_count = target_set.iter().count();
    println!("Number of Targets (Node x Addr): {}", distinct_target_count);

    // Output file

    let path = folder
        .as_ref()
        .join(format!("{}.Errors.txt", basename.as_ref()));
    let mut output_file = std::fs::File::create(path).unwrap();

    let mut latex_table = String::new();
    let mut boxplots = String::new();

    // Note, the micro-architecture P5 corresponds to unknown architectures. Eventually we need to migrate all the data files.
    writeln!(
        output_file,
        "MicroArchitecture: {:?} - {:?}",
        data.micro_architecture.0.0, data.micro_architecture.1
    )
    .unwrap_or_default();

    let projection_full = LocationParameters {
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
    };

    let projection_socket = LocationParameters {
        attacker: CoreLocParameters {
            socket: true,
            core: false,
        },
        victim: CoreLocParameters {
            socket: true,
            core: false,
        },
        memory_numa_node: true,
        memory_slice: false,
        memory_vpn: false,
        memory_offset: false,
    };

    let projection_numa_m_core_av = LocationParameters {
        attacker: CoreLocParameters {
            socket: true,
            core: true,
        },
        victim: CoreLocParameters {
            socket: true,
            core: true,
        },
        memory_numa_node: true,
        memory_slice: false,
        memory_vpn: false,
        memory_offset: false,
    };

    let projection_numa_m_core_av_addr = LocationParameters {
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
    };

    let projection_numa_avm_addr = LocationParameters {
        attacker: CoreLocParameters {
            socket: true,
            core: false,
        },
        victim: CoreLocParameters {
            socket: true,
            core: false,
        },
        memory_numa_node: true,
        memory_slice: true,
        memory_vpn: true,
        memory_offset: true,
    };

    let basename_all = format!("{}-all", basename.as_ref());

    let projected_full = make_projection(&location_map, projection_full);

    let errors = compute_errors(&projected_full);

    let count = errors.par_iter().count();
    assert_eq!(count, 1);
    let full_threshold_errors = errors.iter().next().unwrap().1;

    write_out_threshold_info(&mut output_file, "Full", full_threshold_errors)
        .map_err(|e| eprintln!("Failed to write to file"))
        .unwrap_or_default();

    make_projection_plots(
        projected_full,
        &folder,
        basename_all,
        true,
        true,
        true,
        GroupDimension::Horizontal(1),
        &(ordering_all as fn(PartialLocationOwned, PartialLocationOwned) -> std::cmp::Ordering),
    )
    .expect("Failed to make Full projection plot.");

    let mut json = Vec::new();

    if num_entries > 1 {
        let stat =
            error_statistics::compute_statistics(&location_map, projection_full, vec![], &errors);
        writeln!(output_file);
        stat.write(&mut output_file, "Full-AVM-Errors");
        stat.latex_table(&mut latex_table, "TU");
        stat.boxplot(&mut boxplots, "TU");
        let mut base = Object::new();
        base.insert("projection", location_parameters_json(projection_full));
        json.extend(stat.to_json(&base))
    }

    /*
    List of models:
    Numa(MAV), Numa(MA), Numa(AV), Numa(A) x (No Addr / Addr)

    (Numa(M)+Core(AV), Numa(MV)+Core(A), Numa(M) + Core(A)) x (No Addr / Addr)


     */

    if victim_socket_count * attacker_socket_count * numa_node_count > 1 {
        let basename_numa = format!("{}-node-aware", basename.as_ref());

        let projected_numa = make_projection(&location_map, projection_socket);
        let numa_threshold_errors = compute_errors(&projected_numa);
        if num_entries > victim_socket_count * attacker_socket_count * numa_node_count {
            let stat = error_statistics::compute_statistics(
                &location_map,
                projection_socket,
                vec![(String::from("Best"), projection_socket)],
                &numa_threshold_errors,
            );
            writeln!(output_file);
            stat.write(&mut output_file, "Numa-AVM-Errors");
            stat.latex_table(&mut latex_table, "Numa-AVM");
            stat.boxplot(&mut boxplots, "Numa-AVM");
        }

        // Compute the min, max, median and average error.

        /*
        writeln!(output_file, "% Numa-FF-Dual-Boxplot:  ").unwrap_or_default();
        writeln!(output_file,
                 "%\\addplot[boxplot prepared={{lower whisker={}, lower quartile={}, median={}, average={}, upper quartile={}, upper whisker={},}},] coordinates {{}};",
                 numa_stats.min.flush_dual_error.error_rate() * 1.,
                 numa_stats.q1.flush_dual_error.error_rate() * 1.,
                 numa_stats.med.flush_dual_error.error_rate() * 1.,
                 numa_stats.avg.flush_dual_error.error_rate() * 1.,
                 numa_stats.q3.flush_dual_error.error_rate() * 1.,
                 numa_stats.max.flush_dual_error.error_rate() * 1.).unwrap_or_default();
        writeln!(output_file, "% Numa-FF-Single-Boxplot:").unwrap_or_default();
        writeln!(output_file,
                 "\\addplot[boxplot prepared={{lower whisker={}, lower quartile={}, median={}, average={}, upper quartile={}, upper whisker={},}},] coordinates {{}};",
                 numa_stats.min.flush_single_error.error_rate() * 1.,
                 numa_stats.q1.flush_single_error.error_rate() * 1.,
                 numa_stats.med.flush_single_error.error_rate() * 1.,
                 numa_stats.avg.flush_single_error.error_rate() * 1.,
                 numa_stats.q3.flush_single_error.error_rate() * 1.,
                 numa_stats.max.flush_single_error.error_rate() * 1.).unwrap_or_default();
        writeln!(output_file, "% Numa-FR-Dual-Boxplot:  ").unwrap_or_default();
        writeln!(output_file,
                 "%\\addplot[boxplot prepared={{lower whisker={}, lower quartile={}, median={}, average={}, upper quartile={}, upper whisker={},}},] coordinates {{}};",
                 numa_stats.min.reload_dual_error.error_rate() * 1.,
                 numa_stats.q1.reload_dual_error.error_rate() * 1.,
                 numa_stats.med.reload_dual_error.error_rate() * 1.,
                 numa_stats.avg.reload_dual_error.error_rate() * 1.,
                 numa_stats.q3.reload_dual_error.error_rate() * 1.,
                 numa_stats.max.reload_dual_error.error_rate() * 1.).unwrap_or_default();
        writeln!(output_file, "% Numa-FR-Single-Boxplot:").unwrap_or_default();
        writeln!(output_file,
                 "\\addplot[boxplot prepared={{lower whisker={}, lower quartile={}, median={}, average={}, upper quartile={}, upper whisker={},}},] coordinates {{}};",
                 numa_stats.min.reload_single_error.error_rate() * 1.,
                 numa_stats.q1.reload_single_error.error_rate() * 1.,
                 numa_stats.med.reload_single_error.error_rate() * 1.,
                 numa_stats.avg.reload_single_error.error_rate() * 1.,
                 numa_stats.q3.reload_single_error.error_rate() * 1.,
                 numa_stats.max.reload_single_error.error_rate() * 1.).unwrap_or_default();
        writeln!(output_file, "% Full-FF-Dual-Boxplot:  ").unwrap_or_default();
        writeln!(output_file,
                 "\\addplot[boxplot prepared={{lower whisker={}, lower quartile={}, median={}, average={}, upper quartile={}, upper whisker={},}},] coordinates {{}};",
                 full_stats.min.flush_dual_error.error_rate() * 1.,
                 full_stats.q1.flush_dual_error.error_rate() * 1.,
                 full_stats.med.flush_dual_error.error_rate() * 1.,
                 full_stats.avg.flush_dual_error.error_rate() * 1.,
                 full_stats.q3.flush_dual_error.error_rate() * 1.,
                 full_stats.max.flush_dual_error.error_rate() * 1.).unwrap_or_default();
        writeln!(output_file, "% Full-FF-Single-Boxplot:").unwrap_or_default();
        writeln!(output_file,
                 "\\addplot[boxplot prepared={{lower whisker={}, lower quartile={}, median={}, average={}, upper quartile={}, upper whisker={},}},] coordinates {{}};",
                 full_stats.min.flush_single_error.error_rate() * 1.,
                 full_stats.q1.flush_single_error.error_rate() * 1.,
                 full_stats.med.flush_single_error.error_rate() * 1.,
                 full_stats.avg.flush_single_error.error_rate() * 1.,
                 full_stats.q3.flush_single_error.error_rate() * 1.,
                 full_stats.max.flush_single_error.error_rate() * 1.).unwrap_or_default();
        writeln!(output_file, "% Full-FR-Dual-Boxplot:  ").unwrap_or_default();
        writeln!(output_file,
                 "\\addplot[boxplot prepared={{lower whisker={}, lower quartile={}, median={}, average={}, upper quartile={}, upper whisker={},}},] coordinates {{}};",
                 full_stats.min.reload_dual_error.error_rate() * 1.,
                 full_stats.q1.reload_dual_error.error_rate() * 1.,
                 full_stats.med.reload_dual_error.error_rate() * 1.,
                 full_stats.avg.reload_dual_error.error_rate() * 1.,
                 full_stats.q3.reload_dual_error.error_rate() * 1.,
                 full_stats.max.reload_dual_error.error_rate() * 1.).unwrap_or_default();
        writeln!(output_file, "% Full-FR-Single-Boxplot:").unwrap_or_default();
        writeln!(output_file,
                 "\\addplot[boxplot prepared={{lower whisker={}, lower quartile={}, median={}, average={}, upper quartile={}, upper whisker={},}},] coordinates {{}};",
                 full_stats.min.reload_single_error.error_rate() * 1.,
                 full_stats.q1.reload_single_error.error_rate() * 1.,
                 full_stats.med.reload_single_error.error_rate() * 1.,
                 full_stats.avg.reload_single_error.error_rate() * 1.,
                 full_stats.q3.reload_single_error.error_rate() * 1.,
                 full_stats.max.reload_single_error.error_rate() * 1.).unwrap_or_default();
        */
        make_projection_plots(
            projected_numa,
            &folder,
            basename_numa,
            true,
            true,
            true,
            GroupDimension::Rectangle(victim_socket_count * attacker_socket_count, numa_node_count),
            &(ordering_numa
                as fn(PartialLocationOwned, PartialLocationOwned) -> std::cmp::Ordering),
        )
        .expect("Failed to make Full projection plot.");
        /*
        writeln!(output_file).unwrap_or_default();
        writeln!(output_file, "Flush-Reload-Data:").unwrap_or_default();
        writeln!(output_file, "\\stats{{{}}}{{{}}}{{{}}}{{{}}}{{{}}}{{{}}} & \\stats{{{}}}{{{}}}{{{}}}{{{}}}{{{}}}{{{}}} & \\stats{{{}}}{{{}}}{{{}}}{{{}}}{{{}}}{{{}}} \\\\",
            full_stats.avg.reload_single_error.error_rate() * 100., full_stats.min.reload_single_error.error_rate() * 100., full_stats.q1.reload_single_error.error_rate() * 100., full_stats.med.reload_single_error.error_rate() * 100., full_stats.q3.reload_single_error.error_rate() * 100., full_stats.max.reload_single_error.error_rate() * 100.,
            full_stats.avg.reload_dual_error.error_rate() * 100., full_stats.min.reload_dual_error.error_rate() * 100., full_stats.q1.reload_dual_error.error_rate() * 100., full_stats.med.reload_dual_error.error_rate() * 100., full_stats.q3.reload_dual_error.error_rate() * 100., full_stats.max.reload_dual_error.error_rate() * 100.,
            numa_stats.avg.reload_single_error.error_rate() * 100., numa_stats.min.reload_single_error.error_rate() * 100., numa_stats.q1.reload_single_error.error_rate() * 100., numa_stats.med.reload_single_error.error_rate() * 100., numa_stats.q3.reload_single_error.error_rate() * 100., numa_stats.max.reload_single_error.error_rate() * 100.,
        ).unwrap_or_default();
        writeln!(output_file, "Flush-Flush-Data:").unwrap_or_default();
        writeln!(output_file, "\\stats{{{}}}{{{}}}{{{}}}{{{}}}{{{}}}{{{}}} & \\stats{{{}}}{{{}}}{{{}}}{{{}}}{{{}}}{{{}}} & \\stats{{{}}}{{{}}}{{{}}}{{{}}}{{{}}}{{{}}} \\\\",
                 full_stats.avg.flush_single_error.error_rate() * 100., full_stats.min.flush_single_error.error_rate() * 100., full_stats.q1.flush_single_error.error_rate() * 100., full_stats.med.flush_single_error.error_rate() * 100., full_stats.q3.flush_single_error.error_rate() * 100., full_stats.max.flush_single_error.error_rate() * 100.,
                 full_stats.avg.flush_dual_error.error_rate() * 100., full_stats.min.flush_dual_error.error_rate() * 100., full_stats.q1.flush_dual_error.error_rate() * 100., full_stats.med.flush_dual_error.error_rate() * 100., full_stats.q3.flush_dual_error.error_rate() * 100., full_stats.max.flush_dual_error.error_rate() * 100.,
                 numa_stats.avg.flush_single_error.error_rate() * 100., numa_stats.min.flush_single_error.error_rate() * 100., numa_stats.q1.flush_single_error.error_rate() * 100., numa_stats.med.flush_single_error.error_rate() * 100., numa_stats.q3.flush_single_error.error_rate() * 100., numa_stats.max.flush_single_error.error_rate() * 100.,
        ).unwrap_or_default();
        */

        // ---------------------------------------------------------------------------------
    } else { // hence victim_socket_count * attacker_socket_count * numa_node_count <= 1
        // Here we have extra single socket analysis.

        // TODO Single Socket analysis
    }

    if numa_node_count * victim_core_count * attacker_core_count > 1 {
        {
            let projected_numa_avm_addr = make_projection(&location_map, projection_numa_avm_addr);
            let numa_avm_addr_threshold_errors = compute_errors(&projected_numa_avm_addr);

            let stat = error_statistics::compute_statistics(
                &location_map,
                projection_numa_avm_addr,
                vec![
                    (String::from("Best-AVM-Addr"), projection_numa_avm_addr),
                    (String::from("Best-AVM"), projection_socket),
                ],
                &numa_avm_addr_threshold_errors,
            );
            writeln!(output_file);
            stat.write(&mut output_file, "Numa-AVM-Addr-Errors");
            stat.latex_table(&mut latex_table, "Numa-AVM-Addr");
            stat.boxplot(&mut boxplots, "Numa-AVM-Addr");
        }
        {
            let projected_numa_m_core_av =
                make_projection(&location_map, projection_numa_m_core_av);
            let numa_m_core_av_threshold_errors = compute_errors(&projected_numa_m_core_av);

            let stat = error_statistics::compute_statistics(
                &location_map,
                projection_numa_m_core_av,
                vec![(String::from("Best"), projection_numa_m_core_av)],
                &numa_m_core_av_threshold_errors,
            );
            writeln!(output_file);
            stat.write(&mut output_file, "Numa-M-Core-AV-Errors");
            stat.latex_table(&mut latex_table, "Numa-M-Core-AV");
            stat.boxplot(&mut boxplots, "Numa-M-Core-AV");
        }
        // This is way too slow, and that treatment should be simpler, as we aren't doing any projection.
        {
            let numa_m_core_av_addr_threshold_errors = compute_errors_no_projection(&location_map);

            let stat = error_statistics::compute_statistics_no_projection(
                &location_map,
                vec![
                    (String::from("Best-Addr"), projection_numa_m_core_av_addr),
                    (String::from("Best-No-Addr"), projection_numa_m_core_av),
                ],
                &numa_m_core_av_addr_threshold_errors,
            );
            writeln!(output_file);
            stat.write(&mut output_file, "Numa-M-Core-AV-Addr-Errors");
            stat.latex_table(&mut latex_table, "Numa-M-Core-AV-Addr");
            stat.boxplot(&mut boxplots, "Numa-M-Core-AV-Addr");
        }
    }
    writeln!(output_file, "{}", latex_table);
    writeln!(output_file);
    writeln!(output_file, "{}", boxplots);
}

pub fn run_analysis_from_file<const WIDTH: u64, const N: usize>(name: &str) -> Result<(), ()>
where
    [(); { N + N }]:,
    [(); { WIDTH + WIDTH } as usize]:,
{
    let suffix = format!(".{}", NumaCalibrationResult::<WIDTH, { N + N }>::EXTENSION);
    let suffix_zstd = format!(
        ".{}",
        NumaCalibrationResult::<WIDTH, { N + N }>::EXTENSION_ZSTD
    );

    let folder = <str as AsRef<Path>>::as_ref(&name)
        .canonicalize()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    let mut basename = name.to_owned();

    if basename.ends_with(&suffix) {
        basename = basename.strip_suffix(&suffix).unwrap().to_owned();
    } else if basename.ends_with(&suffix_zstd) {
        basename = basename.strip_suffix(&suffix_zstd).unwrap().to_owned();
    }

    let name = <String as AsRef<Path>>::as_ref(&basename)
        .file_name()
        .unwrap()
        .to_os_string()
        .into_string()
        .unwrap();

    let candidate_zstd = format!(
        "{}.{}",
        basename,
        NumaCalibrationResult::<WIDTH, { N + N }>::EXTENSION_ZSTD
    );
    let candidate_raw = format!(
        "{}.{}",
        basename,
        NumaCalibrationResult::<WIDTH, { N + N }>::EXTENSION
    );

    let (results, format) = /*if std::fs::exists(&candidate_zstd).unwrap()
        && std::fs::exists(&candidate_raw).unwrap()
    {
        let r1 =
            NumaCalibrationResult::<BUCKET_SIZE, BUCKET_NUMBER>::read_msgpack_zstd(&candidate_zstd);
        let r2 = NumaCalibrationResult::<BUCKET_SIZE, BUCKET_NUMBER>::read_msgpack(&candidate_raw);
        if r1 != r2 {
            eprintln!("Data mismatch");
            return Err(());
        }
        println!("{} and {} match", candidate_zstd, candidate_raw);
        (
            r1,
            NumaCalibrationResult::<BUCKET_SIZE, BUCKET_NUMBER>::EXTENSION_ZSTD,
        )
    } else */ if std::fs::exists(&candidate_zstd).unwrap() {
        (
            NumaCalibrationResult::<WIDTH, {N+N}>::read_msgpack_zstd(&candidate_zstd),
            NumaCalibrationResult::<WIDTH, {N+N}>::EXTENSION_ZSTD,
        )
    } else if std::fs::exists(&candidate_raw).unwrap() {
        (
            NumaCalibrationResult::<WIDTH, {N+N}>::read_msgpack(&candidate_raw),
            NumaCalibrationResult::<WIDTH, {N+N}>::EXTENSION,
        )
    } else {
        return Err(());
    };

    let results = match results {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{:?}", e);
            panic!();
        }
    };

    eprintln!("Read and deserialized {}.{}", name, format);
    println!("Operations");
    for op in &results.operations {
        println!("{}: {}", op.name, op.display_name);
    }
    println!("Number of Calibration Results: {}", results.results.len());
    println!("Micro-architecture: {:?}", results.micro_architecture);
    run_analysis::<WIDTH, N>(results, folder, name);
    Ok(())
}

pub fn run_tsc_from_file<const WIDTH: u64, const N: usize>(name: &str) -> Result<(), ()> {
    let suffix = format!(".{}", NumaCalibrationResult::<WIDTH, N>::EXTENSION);
    let suffix_zstd = format!(".{}", NumaCalibrationResult::<WIDTH, N>::EXTENSION_ZSTD);

    let folder = <str as AsRef<Path>>::as_ref(&name)
        .canonicalize()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    let mut basename = name.to_owned();

    if basename.ends_with(&suffix) {
        basename = basename.strip_suffix(&suffix).unwrap().to_owned();
    } else if basename.ends_with(&suffix_zstd) {
        basename = basename.strip_suffix(&suffix_zstd).unwrap().to_owned();
    }

    let name = <String as AsRef<Path>>::as_ref(&basename)
        .file_name()
        .unwrap()
        .to_os_string()
        .into_string()
        .unwrap();

    let candidate_zstd = format!(
        "{}.{}",
        basename,
        NumaCalibrationResult::<WIDTH, N>::EXTENSION_ZSTD
    );
    let candidate_raw = format!(
        "{}.{}",
        basename,
        NumaCalibrationResult::<WIDTH, N>::EXTENSION
    );

    let (results, format) = if std::fs::exists(&candidate_zstd).unwrap() {
        (
            NumaCalibrationResult::<WIDTH, N>::read_msgpack_zstd(&candidate_zstd),
            NumaCalibrationResult::<WIDTH, N>::EXTENSION_ZSTD,
        )
    } else if std::fs::exists(&candidate_raw).unwrap() {
        (
            NumaCalibrationResult::<WIDTH, N>::read_msgpack(&candidate_raw),
            NumaCalibrationResult::<WIDTH, N>::EXTENSION,
        )
    } else {
        return Err(());
    };

    let numa_results = match results {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{:?}", e);
            panic!();
        }
    };

    eprintln!("Read and deserialized {}.{}", name, format);
    /*println!("Operations");
    for op in &numa_results.operations {
        println!("{}: {}", op.name, op.display_name);
    }
    println!(
        "Number of Calibration Results: {}",
        numa_results.results.len()
    );
    println!("Micro-architecture: {:?}", numa_results.micro_architecture);
    */
    // Compute histogram stacks and then try finding the GCD ?
    println!(
        "Running analysis with folder: {} and basename: {}",
        folder.display(),
        basename
    );
    /* We need dual and single threshold analysis */
    let results = numa_results.results;
    let topology_info = numa_results.topology_info;
    let slicings = numa_results.slicing;
    let operations = numa_results.operations;

    /*let names = CacheOps {
        flush_hit: HashSet::from([String::from("clflush_remote_hit")]),
        flush_miss: HashSet::from([
            String::from("clflush_miss_f"),
            //String::from("clflush_miss_n"),
        ]),
        reload_hit: HashSet::from([String::from("reload_remote_hit")]),
        reload_miss: HashSet::from([String::from("reload_miss")]),
    };

    let mut indexes = CacheOps {
        flush_hit: HashSet::new(),
        flush_miss: HashSet::new(),
        reload_hit: HashSet::new(),
        reload_miss: HashSet::new(),
    };

    for (i, name) in operations.iter().enumerate() {
        if names.flush_hit.contains(&name.name) {
            indexes.flush_hit.insert(i);
        }
        if names.flush_miss.contains(&name.name) {
            indexes.flush_miss.insert(i);
        }
        if names.reload_hit.contains(&name.name) {
            indexes.reload_hit.insert(i);
        }
        if names.reload_miss.contains(&name.name) {
            indexes.reload_miss.insert(i);
        }
    }*/

    let core_location = |core: usize| {
        // Eventually we need to integrate https://docs.rs/raw-cpuid/latest/raw_cpuid/struct.ExtendedTopologyIter.html
        let node = topology_info[&core].into();
        CoreLocation {
            socket: node,
            core: core as u16,
        }
    };

    let location_map = calibration_result_to_location_map_parallel(
        results,
        &|static_hist_result| {
            let mut result = StaticHistogram::<WIDTH, N>::default();
            for (i, hist) in static_hist_result.histogram.into_iter().enumerate() {
                result += &hist;
            }
            result
        },
        &|addr| {
            slicings
                .1
                .hash(addr)
                .try_into()
                .expect("Slice index doesn't fit u8")
        },
        &core_location,
    );

    // 1. For each location, extract, FLUSH_HIT / FLUSH_MISS - RELOAD_HIT / RELOAD_MISS
    // 2. From there, compute the various reductions from thresholding, without losing the initial data.
    //    (This will require careful use of references, and probably warrants some sort of helper, given we have 34 different configs)
    // 3. Use the reductions to determine thresholds.
    // 4. Compute the expected errors, with average, min, max and stddev.
    //println!("Number of entries: {}", location_map.iter().count());

    let projection_full = LocationParameters {
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
    };

    let basename_rdtsc = format!("{}-rdtsc", basename);

    //let shallow_copy = location_map.par_iter().collect();
    let projected_full = reduce(
        location_map,
        |avm_location| PartialLocationOwned::new(projection_full, avm_location),
        || StaticHistogram::<WIDTH, N>::default(),
        |acc, v, _k, _rk| {
            *acc += &v;
        },
        |acc, rk| acc, /* Figure out if we should build the HistCumSums here*/
    );

    let count = projected_full.par_iter().count();
    assert_eq!(count, 1);
    let histogram = projected_full.into_iter().next().unwrap().1;
    let mut indices: Vec<u64> = Vec::new();
    for (i, val) in histogram.iter() {
        if val > 0 && i != SimpleBucketU64::<WIDTH, N>::MAX {
            indices.push(i.into());
        }
    }
    println!("Indices found: {:?}", indices);
    let first = indices[0];
    /*for e in indices.iter_mut() {
        *e -= first;
    }*/
    let gcd = indices.into_iter().reduce(gcd).unwrap();
    println!("Found GCD: {}", gcd);
    Ok(())
}
