#![feature(generic_const_exprs)]
#![deny(unsafe_op_in_unsafe_fn)]

use calibration_results::histograms::{
    StaticHistogram, StaticHistogramCumSum, group2_histogram_cum_sum,
};
use pgfplots::axis::plot::Type2D::ConstLeft;
use pgfplots::axis::plot::coordinate::Coordinate2D;
use pgfplots::axis::plot::{Plot2D, PlotKey};

#[derive(Default)]
pub struct CacheOps<T> {
    pub flush_hit: T,
    pub flush_miss: T,
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
    let mut flush_hit_hist = Plot2D::new();
    let mut ymax = 0.0;
    flush_hit_hist.add_key(PlotKey::Type2D(ConstLeft));
    {
        let (mut coordinates, ymax_local) = hit.histogram_points();
        if ymax < ymax_local {
            ymax = ymax_local
        }
        let end = coordinates[coordinates.len() - 1];
        coordinates.push(Coordinate2D::from((end.x, 0f64)));
        coordinates.push(Coordinate2D::from((0f64, 0f64)));
        flush_hit_hist.coordinates = coordinates;
    }
    flush_hit_hist.add_key(PlotKey::Custom(String::from(
        "draw=HistBlue, fill=HistBlueLight, very thin",
    )));

    let mut flush_miss_hist = Plot2D::new();
    flush_miss_hist.add_key(PlotKey::Type2D(ConstLeft));
    {
        let (coordinates, ymax_local) = miss.histogram_points();
        if ymax < ymax_local {
            ymax = ymax_local
        }
        flush_miss_hist.coordinates = coordinates;
    }
    flush_miss_hist.add_key(PlotKey::Custom(String::from("draw=HistRed, thin")));
    (vec![flush_hit_hist, flush_miss_hist], ymax)
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
