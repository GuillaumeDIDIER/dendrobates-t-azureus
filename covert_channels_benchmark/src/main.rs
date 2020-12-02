#![feature(unsafe_block_in_unsafe_fn)]
#![deny(unsafe_op_in_unsafe_fn)]

use std::io::{stdout, Write};

use covert_channels_evaluation::{benchmark_channel, CovertChannel, CovertChannelBenchmarkResult};
use flush_flush::naive::NaiveFlushAndFlush;
use flush_flush::{FlushAndFlush, SingleFlushAndFlush};
use flush_reload::naive::NaiveFlushAndReload;

const NUM_BYTES: usize = 1 << 14; //20

const NUM_PAGES: usize = 1;

const NUM_PAGES_2: usize = 4;

const NUM_PAGE_MAX: usize = 32;

const NUM_ITER: usize = 32;

struct BenchmarkStats {
    raw_res: Vec<CovertChannelBenchmarkResult>,
    average_p: f64,
    var_p: f64,
    average_C: f64,
    var_C: f64,
    average_T: f64,
    var_T: f64,
}

fn run_benchmark<T: CovertChannel + 'static>(
    name: &str,
    constructor: impl Fn() -> T,
    num_iter: usize,
    num_pages: usize,
) -> BenchmarkStats {
    let mut results = Vec::new();
    print!("Benchmarking {} with {} pages", name, num_pages);
    for _ in 0..num_iter {
        print!(".");
        stdout().flush().expect("Failed to flush");
        let channel = constructor();
        let r = benchmark_channel(channel, num_pages, NUM_BYTES);
        results.push(r);
    }
    println!();
    let mut average_p = 0.0;
    let mut average_C = 0.0;
    let mut average_T = 0.0;
    for result in results.iter() {
        println!("{:?}", result);
        println!("C: {}, T: {}", result.capacity(), result.true_capacity());
        println!("Detailed:\"{}\",{},{},{},{}", name, num_pages, result.csv(), result.capacity(), result.true_capacity());
        average_p += result.error_rate;
        average_C += result.capacity();
        average_T += result.true_capacity()
    }
    average_p /= num_iter as f64;
    average_C /= num_iter as f64;
    average_T /= num_iter as f64;
    println!(
        "{} - {} Average p: {} C: {}, T: {}",
        name, num_pages, average_p, average_C, average_T
    );
    let mut var_p = 0.0;
    let mut var_C = 0.0;
    let mut var_T = 0.0;
    for result in results.iter() {
        let p = result.error_rate - average_p;
        var_p += p * p;
        let C = result.capacity() - average_C;
        var_C += C * C;
        let T = result.true_capacity() - average_T;
        var_T += T * T;
    }
    var_p /= num_iter as f64;
    var_C /= num_iter as f64;
    var_T /= num_iter as f64;
    println!(
        "{} - {} Variance of p: {}, C: {}, T:{}",
        name, num_pages, var_p, var_C, var_T
    );
    println!("CSV:\"{}\",{},{},{},{},{},{},{}",name,num_pages,average_p, average_C, average_T, var_p, var_C, var_T);
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

fn main() {
    println!("Detailed:Benchmark,Pages,{},C,T",CovertChannelBenchmarkResult::csv_header());
    println!("CSV:Benchmark,Pages,p,C,T,var_p,var_C,var_T");
    for num_pages in 1..=32 {
        /*println!("Benchmarking F+F");
        for _ in 0..16 {
            // TODO Use the best possible ASV, not best possible AV
            let (channel, old, receiver, sender) = match SingleFlushAndFlush::new_any_two_core(true) {
                Err(e) => {
                    panic!("{:?}", e);
                }
                Ok(r) => r,
            };

            let r = benchmark_channel(channel, NUM_PAGES, NUM_BYTES);
            println!("{:?}", r);
                    println!("C: {}, T: {}", r.capacity(), r.true_capacity());

        }*/

        let naive_ff = run_benchmark(
            "Naive F+F",
            || NaiveFlushAndFlush::from_threshold(202),
            NUM_ITER << 4,
            num_pages,
        );

        let better_ff = run_benchmark(
            "Better F+F",
            || {
                match FlushAndFlush::new_any_two_core(true) {
                    Err(e) => {
                        panic!("{:?}", e);
                    }
                    Ok(r) => r,
                }
                .0
            },
            NUM_ITER,
            num_pages,
        );

        let fr = run_benchmark(
            "F+R",
            || NaiveFlushAndReload::from_threshold(230),
            NUM_ITER,
            num_pages,
        );
    }
}
