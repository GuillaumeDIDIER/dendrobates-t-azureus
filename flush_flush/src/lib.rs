#![deny(unsafe_op_in_unsafe_fn)]

pub mod naive;

use basic_timing_cache_channel::{
    SingleChannel, TimingChannelPrimitives, TopologyAwareTimingChannel,
};

use cache_side_channel::MultipleAddrCacheSideChannel;
use cache_utils::calibration::only_flush;
use cache_utils::calibration::CLFLUSH_BUCKET_NUMBER;
use cache_utils::calibration::CLFLUSH_BUCKET_SIZE;
use num_rational::Rational64;

#[derive(Debug, Default)]
pub struct FFPrimitives {}

impl TimingChannelPrimitives for FFPrimitives {
    unsafe fn attack(&self, addr: *const u8) -> u64 {
        unsafe { only_flush(addr) }
    }
    const NEED_RESET: bool = false;
}

pub type FlushAndFlush<E, NFThresh, NFLoc> = TopologyAwareTimingChannel<
    CLFLUSH_BUCKET_SIZE,
    CLFLUSH_BUCKET_NUMBER,
    FFPrimitives,
    E,
    Rational64,
    NFThresh,
    NFLoc,
>;

pub type FFHandle<E, NFThresh, NFLoc> =
    <FlushAndFlush<E, NFThresh, NFLoc> as MultipleAddrCacheSideChannel>::Handle;

pub type SingleFlushAndFlush<E, NFThresh, NFLoc> = SingleChannel<FlushAndFlush<E, NFThresh, NFLoc>>;
