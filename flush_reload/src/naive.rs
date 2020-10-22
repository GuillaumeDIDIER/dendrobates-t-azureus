use cache_side_channel::{
    CacheStatus, ChannelFatalError, SideChannelError, SingleAddrCacheSideChannel,
};
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
    /// # Safety
    ///
    /// addr needs to be a valid pointer
    unsafe fn test_single(&mut self, addr: *const u8) -> Result<CacheStatus, SideChannelError> {
        if self.current != Some(addr) {
            return Err(SideChannelError::AddressNotReady(addr));
        }
        let t = unsafe { only_reload(addr) };
        if t > self.threshold {
            Ok(CacheStatus::Miss)
        } else {
            Ok(CacheStatus::Hit)
        }
    }

    /// # Safety:
    ///
    /// addr needs to be a valid pointer
    unsafe fn prepare_single(&mut self, addr: *const u8) -> Result<(), SideChannelError> {
        unsafe { flush(addr) };
        self.current = Some(addr);
        Ok(())
    }

    fn victim_single(&mut self, operation: &dyn Fn()) {
        operation()
    }

    /// # Safety
    ///
    /// addr needs to be a valid pointer
    unsafe fn calibrate_single(
        &mut self,
        _addresses: impl IntoIterator<Item = *const u8>,
    ) -> Result<(), ChannelFatalError> {
        Ok(())
    }
}
