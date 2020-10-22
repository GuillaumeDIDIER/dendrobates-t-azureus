#![feature(unsafe_block_in_unsafe_fn)]
#![deny(unsafe_op_in_unsafe_fn)]

use cache_side_channel::{
    CacheStatus, ChannelFatalError, SideChannelError, SingleAddrCacheSideChannel,
};
use cache_utils::calibration::only_reload;
use cache_utils::flush;

pub mod naive;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
