#![deny(unsafe_op_in_unsafe_fn)]

pub mod naive;

use basic_timing_cache_channel::{
    SingleChannel, TimingChannelPrimitives, TopologyAwareTimingChannel,
};

use cache_side_channel::MultipleAddrCacheSideChannel;
use cache_utils::calibration::{only_flush, only_flush_rdpru, only_reload, only_reload_rdpru};
use cache_utils::has_rdpru;
use num_rational::Rational64;

#[derive(Debug, Clone)]
pub struct FFPrimitives {
    flush: unsafe fn(*const u8) -> u64,
}

impl Default for FFPrimitives {
    fn default() -> Self {
        if has_rdpru() {
            Self {
                flush: only_flush_rdpru,
            }
        } else {
            Self { flush: only_flush }
        }
    }
}

impl TimingChannelPrimitives for FFPrimitives {
    unsafe fn attack(&self, addr: *const u8) -> u64 {
        unsafe { (self.flush)(addr) }
    }
    unsafe fn reset(&self, addr: *const u8) {
        unsafe { (self.flush)(addr) };
    }
    unsafe fn attack_reset(&self, addr: *const u8) -> u64 {
        unsafe { (self.flush)(addr) }
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
