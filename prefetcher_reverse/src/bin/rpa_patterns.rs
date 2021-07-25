#![deny(unsafe_op_in_unsafe_fn)]
use basic_timing_cache_channel::TopologyAwareError;
use cache_side_channel::CacheStatus::Hit;
use cache_side_channel::{
    set_affinity, ChannelHandle, CoreSpec, MultipleAddrCacheSideChannel, SingleAddrCacheSideChannel,
};
use cache_utils::calibration::PAGE_LEN;
use cache_utils::maccess;
use cache_utils::mmap;
use cache_utils::mmap::MMappedMemory;
use flush_flush::{FFHandle, FFPrimitives, FlushAndFlush};
use nix::Error;
use prefetcher_reverse::{reference_patterns, Prober, CACHE_LINE_LEN, PAGE_CACHELINE_LEN};
use rand::seq::SliceRandom;
use std::iter::Cycle;
use std::ops::Range;

pub const NUM_ITERATION: usize = 1 << 10;
pub const NUM_PAGES: usize = 256;

fn exp(delay: u64) {
    for (name, pattern) in reference_patterns() {
        let mut prober = Prober::new(63).unwrap();
        prober.set_delay(delay);

        println!("{}", name);
        let result = prober.full_page_probe(pattern, NUM_ITERATION as u32, 100);
        println!("{}", result);
    }
}

fn main() {
    for delay in [0, 5, 10, 50] {
        println!("Delay after each access: {} us", delay);
        exp(delay);
    }
}
