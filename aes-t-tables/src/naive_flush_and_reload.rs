use crate::{CacheStatus, ChannelFatalError, SideChannelError, SingleAddrCacheSideChannel};
use cache_utils::calibration::only_reload;
use cache_utils::flush;

#[derive(Debug)]
pub struct NaiveFlushAndReload {
    pub threshold: u64,
    current: Option<*const u8>,
}

impl NaiveFlushAndReload {
    pub fn from_threshold(threshold: u64) -> Self {
        NaiveFlushAndReload {
            threshold,
            current: None,
        }
    }
}

impl SingleAddrCacheSideChannel for NaiveFlushAndReload {
    fn test_single(&mut self, addr: *const u8) -> Result<CacheStatus, SideChannelError> {
        if self.current != Some(addr) {
            panic!(); // FIXME
        }
        let t = unsafe { only_reload(addr) };
        if t > self.threshold {
            Ok(CacheStatus::Miss)
        } else {
            Ok(CacheStatus::Hit)
        }
    }

    fn prepare_single(&mut self, addr: *const u8) -> Result<(), SideChannelError> {
        unsafe { flush(addr) };
        self.current = Some(addr);
        Ok(())
    }

    fn victim_single(&mut self, operation: &dyn Fn()) {
        operation()
    }

    fn calibrate_single(
        &mut self,
        _addresses: impl IntoIterator<Item = *const u8>,
    ) -> Result<(), ChannelFatalError> {
        Ok(())
    }
}
