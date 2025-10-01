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
pub struct FRPrimitives {
    reload: unsafe fn(*const u8) -> u64,
    flush: unsafe fn(*const u8) -> u64,
}

impl Default for FRPrimitives {
    fn default() -> Self {
        if has_rdpru() {
            Self {
                reload: only_reload_rdpru,
                flush: only_flush_rdpru,
            }
        } else {
            Self {
                reload: only_reload,
                flush: only_flush,
            }
        }
    }
}

impl TimingChannelPrimitives for FRPrimitives {
    unsafe fn attack(&self, addr: *const u8) -> u64 {
        unsafe { (self.reload)(addr) }
    }

    unsafe fn attack_reset(&self, addr: *const u8) -> u64 {
        let r = unsafe { (self.reload)(addr) };
        unsafe { (self.flush)(addr) };
        r
    }

    unsafe fn reset(&self, addr: *const u8) {
        unsafe { (self.flush)(addr) };
    }
    //const NEED_RESET: bool = true;
}

pub type FlushAndReload<const W: u64, const N: usize, E, NFThresh, NFLoc> =
    TopologyAwareTimingChannel<
        W, //CLFLUSH_BUCKET_SIZE,
        N, //CLFLUSH_BUCKET_NUMBER,
        FRPrimitives,
        E,
        Rational64,
        NFThresh,
        NFLoc,
    >;

pub type FRHandle<const W: u64, const N: usize, E, NFThresh, NFLoc> =
    <FlushAndReload<W, N, E, NFThresh, NFLoc> as MultipleAddrCacheSideChannel>::Handle;

pub type SingleFlushAndReload<const W: u64, const N: usize, E, NFThresh, NFLoc> =
    SingleChannel<FlushAndReload<W, N, E, NFThresh, NFLoc>>;

pub type FlushAndReloadCovertOpt<const W: u64, const N: usize, E, NFThresh, NFLoc> =
    TopologyAwareTimingChannel<
        W, //CLFLUSH_BUCKET_SIZE,
        N, //CLFLUSH_BUCKET_NUMBER,
        FRPrimitives,
        E,
        Rational64,
        NFThresh,
        NFLoc,
        false,
    >;

pub type FROHandle<const W: u64, const N: usize, E, NFThresh, NFLoc> =
    <FlushAndReloadCovertOpt<W, N, E, NFThresh, NFLoc> as MultipleAddrCacheSideChannel>::Handle;
