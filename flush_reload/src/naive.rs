use cache_side_channel::{
    CacheStatus, ChannelFatalError, CoreSpec, SideChannelError, SingleAddrCacheSideChannel,
};
use cache_utils::calibration::only_reload;
use cache_utils::flush;
use covert_channels_evaluation::{BitIterator, CovertChannel};
use nix::sched::{sched_getaffinity, CpuSet};
use nix::unistd::Pid;

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

unsafe impl Send for NaiveFlushAndReload {}
unsafe impl Sync for NaiveFlushAndReload {}

impl CoreSpec for NaiveFlushAndReload {
    fn main_core(&self) -> CpuSet {
        sched_getaffinity(Pid::from_raw(0)).unwrap()
    }

    fn helper_core(&self) -> CpuSet {
        sched_getaffinity(Pid::from_raw(0)).unwrap()
    }
}

impl CovertChannel for NaiveFlushAndReload {
    const BIT_PER_PAGE: usize = 1;

    unsafe fn transmit<'a>(&self, page: *const u8, bits: &mut BitIterator<'a>) {
        unimplemented!()
    }

    unsafe fn receive(&self, page: *const u8) -> Vec<bool> {
        unimplemented!()
        /*
        let r = self.test_single(page);
        match r {
            Err(e) => unimplemented!(),
            Ok(status) => match status {
                CacheStatus::Hit => vec![true],
                CacheStatus::Miss => vec![false],
            },
        }
         */
    }

    unsafe fn ready_page(&mut self, page: *const u8) {
        unimplemented!()
    }
}
