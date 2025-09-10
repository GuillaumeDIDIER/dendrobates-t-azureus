#![deny(unsafe_op_in_unsafe_fn)]

pub mod naive;

use basic_timing_cache_channel::{
    SingleChannel, TimingChannelPrimitives, TopologyAwareTimingChannel,
};

use cache_side_channel::MultipleAddrCacheSideChannel;
use cache_utils::calibration::only_flush;
use num_rational::Rational64;

#[derive(Debug, Default, Clone)]
pub struct FFPrimitives {}

impl TimingChannelPrimitives for FFPrimitives {
    unsafe fn attack(&self, addr: *const u8) -> u64 {
        unsafe { only_flush(addr) }
    }
    unsafe fn reset(&self, addr: *const u8) {
        unsafe { only_flush(addr) };
    }
    unsafe fn attack_reset(&self, addr: *const u8) -> u64 {
        unsafe { only_flush(addr) }
    }
    //const NEED_RESET: bool = false;
}

pub type FlushAndFlush<const W: u64, const N: usize, E, NFThresh, NFLoc> =
    TopologyAwareTimingChannel<
        W, //CLFLUSH_BUCKET_SIZE,
        N, //CLFLUSH_BUCKET_NUMBER,
        FFPrimitives,
        E,
        Rational64,
        NFThresh,
        NFLoc,
    >;

pub type FFHandle<const W: u64, const N: usize, E, NFThresh, NFLoc> =
    <FlushAndFlush<W, N, E, NFThresh, NFLoc> as MultipleAddrCacheSideChannel>::Handle;

pub type SingleFlushAndFlush<const W: u64, const N: usize, E, NFThresh, NFLoc> =
    SingleChannel<FlushAndFlush<W, N, E, NFThresh, NFLoc>>;
