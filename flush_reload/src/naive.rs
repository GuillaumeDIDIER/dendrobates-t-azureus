use cache_side_channel::{
    CacheStatus, ChannelFatalError, CoreSpec, SideChannelError, SingleAddrCacheSideChannel,
};
use cache_utils::calibration::{get_vpn, only_flush, only_reload, VPN};
use cache_utils::flush;
use covert_channels_evaluation::{BitIterator, CovertChannel};
use nix::sched::{sched_getaffinity, CpuSet};
use nix::unistd::Pid;
use std::collections::HashMap;
use std::thread::current;

#[derive(Debug)]
pub struct NaiveFlushAndReload {
    pub threshold: u64,
    current: HashMap<VPN, *const u8>,
    main_core: CpuSet,
    helper_core: CpuSet,
}

impl NaiveFlushAndReload {
    pub fn from_threshold(threshold: u64) -> Self {
        NaiveFlushAndReload {
            threshold,
            current: Default::default(),
            main_core: sched_getaffinity(Pid::from_raw(0)).unwrap(),
            helper_core: sched_getaffinity(Pid::from_raw(0)).unwrap(),
        }
    }
    unsafe fn test_impl(&self, addr: *const u8) -> Result<CacheStatus, SideChannelError> {
        let vpn = get_vpn(addr);
        if self.current.get(&vpn) != Some(&addr) {
            return Err(SideChannelError::AddressNotReady(addr));
        }
        let t = unsafe { only_reload(addr) };
        unsafe { flush(addr) };
        if t > self.threshold {
            Ok(CacheStatus::Miss)
        } else {
            Ok(CacheStatus::Hit)
        }
    }

    pub fn set_cores(&mut self, main_core: usize, helper_core: usize) {
        self.main_core = CpuSet::new();
        self.main_core.set(main_core).unwrap();

        self.helper_core = CpuSet::new();
        self.helper_core.set(helper_core).unwrap();
    }
}

impl SingleAddrCacheSideChannel for NaiveFlushAndReload {
    /// # Safety
    ///
    /// addr needs to be a valid pointer
    unsafe fn test_single(&mut self, addr: *const u8) -> Result<CacheStatus, SideChannelError> {
        unsafe { self.test_impl(addr) }
    }

    /// # Safety:
    ///
    /// addr needs to be a valid pointer
    unsafe fn prepare_single(&mut self, addr: *const u8) -> Result<(), SideChannelError> {
        unsafe { flush(addr) };
        let vpn = get_vpn(addr);
        self.current.insert(vpn, addr);
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
        self.main_core
    }

    fn helper_core(&self) -> CpuSet {
        self.helper_core
    }
}

impl CovertChannel for NaiveFlushAndReload {
    const BIT_PER_PAGE: usize = 1;

    unsafe fn transmit<'a>(&self, page: *const u8, bits: &mut BitIterator<'a>) {
        let vpn = get_vpn(page);
        let addr = self.current.get(&vpn).unwrap();
        if let Some(b) = bits.next() {
            if b {
                unsafe { only_reload(*addr) };
            } else {
                unsafe { only_flush(*addr) };
            }
        }
    }

    unsafe fn receive(&self, page: *const u8) -> Vec<bool> {
        let r = unsafe { self.test_impl(page) };
        match r {
            Err(e) => panic!(),
            Ok(status) => match status {
                CacheStatus::Hit => vec![true],
                CacheStatus::Miss => vec![false],
            },
        }
    }

    unsafe fn ready_page(&mut self, page: *const u8) {
        unsafe { self.prepare_single(page) };
    }
}
