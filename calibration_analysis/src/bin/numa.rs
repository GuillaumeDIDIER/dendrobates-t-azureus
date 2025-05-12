#![feature(generic_const_exprs)]

use calibration_results::calibration_2t::{
    calibration_result_to_location_map, calibration_result_to_location_map_parallel,
};
use calibration_results::numa_results::{BUCKET_NUMBER, BUCKET_SIZE, NumaCalibrationResult};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
//use lzma_rs::xz_decompress;
use calibration_results::calibration::{
    AVMLocation, CoreLocParameters, CoreLocation, ErrorPrediction, LocationParameters,
    PartialLocation, PartialLocationOwned, StaticHistCalibrateResult,
};
use calibration_results::classifiers::{
    DualThreshold, DualThresholdBuilder, ErrorPredictionsBuilder, SimpleThresholdBuilder, Threshold,
};
use calibration_results::histograms::{
    Bucket, SimpleBucketU64, StaticHistogram, StaticHistogramCumSum, group2_histogram_cum_sum,
};
use calibration_results::reduce;
use pgfplots::axis::plot::Type2D::ConstLeft;
use pgfplots::axis::plot::coordinate::Coordinate2D;
use pgfplots::axis::plot::{Plot2D, PlotKey};
use pgfplots::axis::{Axis, AxisKey};
use pgfplots::groupplot::{GroupDimension, GroupPlot};
use pgfplots::{Engine, Picture};
use rmp_serde::Deserializer;
use serde::Deserialize;
use std::env::args;
use std::path::Path;
/*
Design to do, we need to extract, for both FR and FF the raw calibration results (HashMap<AVMLoc, Histograms>)
-> From there we can compute the consolidations for all possible models.

-> Separately, we want to make some histograms, and also try to figure out a way to compare stuffs ?

 */

#[derive(Default)]
struct CacheOps<T> {
    flush_hit: T,
    flush_miss: T,
    reload_hit: T,
    reload_miss: T,
}

/***********************
 * PGFPlots Histograms *
 ***********************/

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

fn make_plot_by2<const WIDTH: u64, const N: usize>(
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

fn make_projection_plots<const WIDTH: u64, const N: usize>(
    location_map: &HashMap<AVMLocation, CacheOps<StaticHistogram<{ WIDTH }, { N + N }>>>,
    projection: LocationParameters,
    folder: impl AsRef<Path>,
    base_name: impl AsRef<str>,
    flush_flush: bool,
    flush_reload: bool,
    group_dimension: GroupDimension,
    sort_criteria: impl FnMut(
        &(
            PartialLocationOwned,
            CacheOps<StaticHistogramCumSum<{ WIDTH }, { N + N }>>,
        ),
        &(
            PartialLocationOwned,
            CacheOps<StaticHistogramCumSum<{ WIDTH }, { N + N }>>,
        ),
    ) -> Ordering,
    /* Thresholding Strategies */
    /* Filter */
) -> Result<(), ()>
where
    [(); { WIDTH + WIDTH } as usize]:,
{
    let shallow_copy = location_map.iter().collect();
    let mapped = reduce(
        shallow_copy,
        |avm_location| PartialLocationOwned::new(projection, *avm_location),
        || CacheOps::<StaticHistogram<WIDTH, { N + N }>>::default(),
        |acc, v, _k, _rk| {
            acc.flush_hit += &v.flush_hit;
            acc.flush_miss += &v.flush_miss;
            acc.reload_hit += &v.reload_hit;
            acc.reload_miss += &v.reload_miss;
        },
        |acc, rk| CacheOps {
            flush_hit: StaticHistogramCumSum::from(acc.flush_hit),
            flush_miss: StaticHistogramCumSum::from(acc.flush_miss),
            reload_hit: StaticHistogramCumSum::from(acc.reload_hit),
            reload_miss: StaticHistogramCumSum::from(acc.reload_miss),
        }, /* Figure out if we should build the HistCumSums here*/
    );

    let mut picture_ff = Picture::new();
    let mut picture_fr = Picture::new();

    let count = mapped.iter().count();
    println!(
        "[{}] Number of entries: {}",
        base_name.as_ref(),
        mapped.iter().count()
    );

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
        let histograms = mapped.into_iter().next().unwrap().1;
        /*
                let simple_threshold_builder: SimpleThresholdBuilder<WIDTH, { N + N }> =
                    SimpleThresholdBuilder();
                let dual_threshold_builder: DualThresholdBuilder<WIDTH, { N + N }> = DualThresholdBuilder();

                let flush_single_threshold = simple_threshold_builder
                    .find_best_classifier(&histograms.flush_hit, &histograms.flush_miss)
                    .expect("Failed to compute single threshold");
                let flush_dual_threshold = dual_threshold_builder
                    .find_best_classifier(&histograms.flush_hit, &histograms.flush_miss)
                    .expect("Failed to compute dual threshold");

                println!(
                    "[{}] Flush Single Threshold: {}, Error Prediction: {}",
                    base_name.as_ref(),
                    flush_single_threshold.0,
                    flush_single_threshold.1
                );
                println!(
                    "[{}] Flush Dual Threshold: {}, Error Prediction: {}",
                    base_name.as_ref(),
                    flush_dual_threshold.0,
                    flush_dual_threshold.1
                );
        */
        let (mut flush_plots, flush_ymax) =
            make_plot_by2(histograms.flush_hit, histograms.flush_miss);
        let mut flush_hist = Axis::new();
        flush_hist.plots.append(&mut flush_plots);

        flush_hist.set_title("Flush+Flush"); // FIXME, this should be customizable by the caller
        flush_hist.add_key(AxisKey::Custom(String::from(
            "height=10cm, width=20cm, xmin=0,xmax=1024, ymin=0, tick align=outside, tick pos=left, axis on top,", //
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
            .to_pdf(folder.as_ref(), picture_ff_jobname, Engine::LuaLatex)
            .expect("Failed to create PDF");

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
        reload_hist.add_key(AxisKey::Custom(String::from(
            "height=10cm, width=20cm, xmin=0,xmax=1024, ymin=0, tick align=outside, tick pos=left, axis on top,", //
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
            .expect("Failed to create PDF");
        // -----------------------------------------------------------------------------------------
    } else {
        // -----------------------------------------------------------------------------------------
        /* count > 1 */
        let mut sorted_histograms: Vec<(_, _)> = mapped.into_iter().collect();
        sorted_histograms.sort_by(sort_criteria);
        // TODO figure out how to handle the multiple plots.

        let mut flush_group = GroupPlot::new();
        flush_group.dimension = group_dimension;
        let mut reload_group = GroupPlot::new();
        reload_group.dimension = group_dimension;

        let mut ymax_flush = 0.0;
        let mut ymax_reload = 0.0;

        for item in sorted_histograms.into_iter().enumerate() {
            let mut flush_axis = Axis::new();
            let mut reload_axis = Axis::new();

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
        }
        flush_group.set_title("FF TBD");
        flush_group.add_key(AxisKey::Custom(format!("height=10cm, width=20cm, xmin=0,xmax=1024, ymin=0, ymax={}, tick align=outside, tick pos=left, axis on top,", ymax_flush)));

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
        picture_ff
            .to_pdf(folder.as_ref(), picture_ff_jobname, Engine::LuaLatex)
            .expect("Failed to create PDF");

        //---------

        reload_group.set_title("FR-TBD");
        reload_group.add_key(AxisKey::Custom(format!("height=10cm, width=20cm, xmin=0,xmax=1024, ymin=0, ymax={}, tick align=outside, tick pos=left, axis on top,", ymax_flush)));

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
        picture_fr
            .to_pdf(folder.as_ref(), picture_fr_jobname, Engine::LuaLatex)
            .expect("Failed to create PDF");
    }

    Ok(())
}

/* TODO : Evaluate if ndarray would be better thann our current hashmaps*/

pub fn run_numa_analysis<const WIDTH: u64, const N: usize>(
    data: NumaCalibrationResult<WIDTH, { N + N }>,
    folder: impl AsRef<Path>,
    basename: impl AsRef<str>,
) -> ()
where
    [(); { WIDTH + WIDTH } as usize]:,
{
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
    println!("Number of entries: {}", location_map.iter().count());

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

    //let shallow_copy = location_map.iter().collect();

    /*let tmp = shallow_copy.iter().find(|a, b| true);
    println!("{:?}", tmp);*/

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

    let basename_all = format!("{}-all", basename.as_ref());
    let ordering_all = |a: &(
        PartialLocationOwned,
        CacheOps<StaticHistogramCumSum<{ WIDTH }, { N + N }>>,
    ),
                        b: &(
        PartialLocationOwned,
        CacheOps<StaticHistogramCumSum<{ WIDTH }, { N + N }>>,
    )| { Ordering::Equal };
    make_projection_plots(
        &location_map,
        projection_full,
        &folder,
        basename_all,
        true,
        true,
        GroupDimension::Horizontal(1),
        &ordering_all,
    )
    .expect("Failed to make Full projection plot.");

    let basename_numa = format!("{}-node-aware", basename.as_ref());
    let ordering_numa = |a: &(
        PartialLocationOwned,
        CacheOps<StaticHistogramCumSum<{ WIDTH }, { N + N }>>,
    ),
                         b: &(
        PartialLocationOwned,
        CacheOps<StaticHistogramCumSum<{ WIDTH }, { N + N }>>,
    )| {
        match a
            .0
            .get_numa_node()
            .unwrap()
            .cmp(&b.0.get_numa_node().unwrap())
        {
            Ordering::Less => Ordering::Less,
            Ordering::Greater => Ordering::Greater,
            Ordering::Equal => {
                match a
                    .0
                    .get_attacker_socket()
                    .unwrap()
                    .cmp(&b.0.get_attacker_socket().unwrap())
                {
                    Ordering::Less => Ordering::Less,
                    Ordering::Greater => Ordering::Greater,
                    Ordering::Equal => {
                        a.0.get_victim_socket()
                            .unwrap()
                            .cmp(&b.0.get_victim_socket().unwrap())
                    }
                }
            }
        }
    };

    /*if victim_socket_count * attacker_socket_count * numa_node_count > 1 {
        make_projection_plots(
            &location_map,
            projection_socket,
            &folder,
            basename_numa,
            true,
            true,
            GroupDimension::Rectangle(victim_socket_count * attacker_socket_count, numa_node_count),
            &ordering_numa,
        )
        .expect("Failed to make Full projection plot.");
    }*/
    //----------

    /*
    Displaying a histogram :
    https://docs.rs/pgfplots/latest/pgfplots/axis/plot/enum.Type2D.html#variant.ConstLeft (TBC), but the bucket 0 correspond to values in [O;WIDTH[

    1 axis environment, push two plots (HIT and MISS)

    */

    /* TODO Outstanding statistic question on how to validate if both types of cache cleanup (nope vs explicit flush) give the same distributions
     */

    /*
    List of configurations :
    For Single Threshold & Dual Threshold

    Topology Unaware

    Socket Aware (Known AV, Known A, Chosen AV, Chosen A, Chosen A-Known V)
    Socket + Addr Aware (Known AV, Known A, Chosen AV, Chosen A, Chosen A-Known V) x (Chosen A | Known A)

    Core Aware

    Core + Addr Aware

    Addr Aware (Chosen A | Known A)

     */

    //unimplemented!()
}

fn run_analysis_from_file(name: &str) -> Result<(), ()> {
    let suffix = format!(
        ".{}",
        NumaCalibrationResult::<BUCKET_SIZE, BUCKET_NUMBER>::EXTENSION
    );
    let suffix_zstd = format!(
        ".{}",
        NumaCalibrationResult::<BUCKET_SIZE, BUCKET_NUMBER>::EXTENSION_ZSTD
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
        NumaCalibrationResult::<BUCKET_SIZE, BUCKET_NUMBER>::EXTENSION_ZSTD
    );
    let candidate_raw = format!(
        "{}.{}",
        basename,
        NumaCalibrationResult::<BUCKET_SIZE, BUCKET_NUMBER>::EXTENSION
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
            NumaCalibrationResult::<BUCKET_SIZE, BUCKET_NUMBER>::read_msgpack_zstd(&candidate_zstd),
            NumaCalibrationResult::<BUCKET_SIZE, BUCKET_NUMBER>::EXTENSION_ZSTD,
        )
    } else if std::fs::exists(&candidate_raw).unwrap() {
        (
            NumaCalibrationResult::<BUCKET_SIZE, BUCKET_NUMBER>::read_msgpack(&candidate_raw),
            NumaCalibrationResult::<BUCKET_SIZE, BUCKET_NUMBER>::EXTENSION,
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
    run_numa_analysis::<BUCKET_SIZE, 512>(results, folder, name);
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
