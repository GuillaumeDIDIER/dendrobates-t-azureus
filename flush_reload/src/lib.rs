#![deny(unsafe_op_in_unsafe_fn)]

pub mod naive;

use basic_timing_cache_channel::{
    SingleChannel, TimingChannelPrimitives, TopologyAwareTimingChannel,
};

use cache_side_channel::MultipleAddrCacheSideChannel;
use cache_utils::calibration::only_reload;
use cache_utils::calibration::CLFLUSH_BUCKET_NUMBER;
use cache_utils::calibration::CLFLUSH_BUCKET_SIZE;
use num_rational::Rational64;

#[derive(Debug, Default)]
pub struct FRPrimitives {}

impl TimingChannelPrimitives for FRPrimitives {
    unsafe fn attack(&self, addr: *const u8) -> u64 {
        unsafe { only_reload(addr) }
    }
    const NEED_RESET: bool = true;
}

pub type FlushAndReload<E, NFThresh, NFLoc> = TopologyAwareTimingChannel<
    CLFLUSH_BUCKET_SIZE,
    CLFLUSH_BUCKET_NUMBER,
    FRPrimitives,
    E,
    Rational64,
    NFThresh,
    NFLoc,
>;

pub type FRHandle<E, NFThresh, NFLoc> =
    <FlushAndReload<E, NFThresh, NFLoc> as MultipleAddrCacheSideChannel>::Handle;

pub type SingleFlushAndReload<E, NFThresh, NFLoc> =
    SingleChannel<FlushAndReload<E, NFThresh, NFLoc>>;
