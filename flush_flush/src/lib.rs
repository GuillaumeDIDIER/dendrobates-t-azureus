#![feature(unsafe_block_in_unsafe_fn)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod naive;

use cache_side_channel::SideChannelError::{AddressNotCalibrated, AddressNotReady};
use cache_side_channel::{
    CacheStatus, ChannelFatalError, CoreSpec, MultipleAddrCacheSideChannel, SideChannelError,
    SingleAddrCacheSideChannel,
};
use cache_utils::calibration::{
    accumulate, calibrate_fixed_freq_2_thread, calibration_result_to_ASVP, get_cache_slicing,
    get_vpn, only_flush, only_reload, CalibrateOperation2T, CalibrationOptions, ErrorPredictions,
    HistParams, HistogramCumSum, PotentialThresholds, Verbosity, ASVP, CFLUSH_BUCKET_NUMBER,
    CFLUSH_BUCKET_SIZE, CFLUSH_NUM_ITER, PAGE_LEN, PAGE_SHIFT,
};
use cache_utils::calibration::{ErrorPrediction, Slice, Threshold, ThresholdError, AV, SP, VPN};
use cache_utils::complex_addressing::CacheSlicing;
use cache_utils::{find_core_per_socket, flush, maccess, noop};
use nix::sched::{sched_getaffinity, sched_setaffinity, CpuSet};
use nix::unistd::Pid;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::{Debug, Formatter};

pub struct FlushAndFlush {
    thresholds: HashMap<SP, ThresholdError>,
    addresses_ready: HashSet<*const u8>,
    slicing: CacheSlicing,
    attacker_core: usize,
    victim_core: usize,
    preferred_address: HashMap<*const u8, *const u8>,
}

#[derive(Debug)]
pub enum FlushAndFlushError {
    NoSlicing,
    Nix(nix::Error),
}

#[derive(Debug)]
pub struct SingleFlushAndFlush(FlushAndFlush);

impl SingleFlushAndFlush {
    pub fn new(attacker_core: usize, victim_core: usize) -> Result<Self, FlushAndFlushError> {
        FlushAndFlush::new(attacker_core, victim_core).map(|ff| SingleFlushAndFlush(ff))
    }

    pub fn new_any_single_core() -> Result<(Self, CpuSet, usize), FlushAndFlushError> {
        FlushAndFlush::new_any_single_core()
            .map(|(ff, old, core)| (SingleFlushAndFlush(ff), old, core))
    }

    pub fn new_any_two_core(
        distinct: bool,
    ) -> Result<(Self, CpuSet, usize, usize), FlushAndFlushError> {
        FlushAndFlush::new_any_two_core(distinct)
            .map(|(ff, old, attacker, victim)| (SingleFlushAndFlush(ff), old, attacker, victim))
    }
}

impl CoreSpec for SingleFlushAndFlush {
    fn main_core(&self) -> CpuSet {
        self.0.main_core()
    }

    fn helper_core(&self) -> CpuSet {
        self.0.helper_core()
    }
}

impl SingleAddrCacheSideChannel for SingleFlushAndFlush {
    unsafe fn test_single(&mut self, addr: *const u8) -> Result<CacheStatus, SideChannelError> {
        unsafe { self.0.test_single(addr) }
    }

    unsafe fn prepare_single(&mut self, addr: *const u8) -> Result<(), SideChannelError> {
        unsafe { self.0.prepare_single(addr) }
    }

    fn victim_single(&mut self, operation: &dyn Fn()) {
        self.0.victim_single(operation)
    }

    unsafe fn calibrate_single(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<(), ChannelFatalError> {
        unsafe { self.0.calibrate_single(addresses) }
    }
}

impl FlushAndFlush {
    pub fn new(attacker_core: usize, victim_core: usize) -> Result<Self, FlushAndFlushError> {
        if let Some(slicing) = get_cache_slicing(find_core_per_socket()) {
            if !slicing.can_hash() {
                return Err(FlushAndFlushError::NoSlicing);
            }

            let ret = Self {
                thresholds: Default::default(),
                addresses_ready: Default::default(),
                slicing,
                attacker_core,
                victim_core,
                preferred_address: Default::default(),
            };
            Ok(ret)
        } else {
            Err(FlushAndFlushError::NoSlicing)
        }
    }

    // Takes a buffer / list of addresses or pages
    // Takes a list of core pairs
    // Run optimized calibration and processes results
    fn calibration_for_core_pairs<'a>(
        core_pairs: impl Iterator<Item = (usize, usize)> + Clone,
        pages: impl Iterator<Item = &'a [u8]>,
    ) -> Result<HashMap<AV, (ErrorPrediction, HashMap<SP, ThresholdError>)>, FlushAndFlushError>
    {
        let core_per_socket = find_core_per_socket();

        let operations = [
            CalibrateOperation2T {
                prepare: maccess::<u8>,
                op: only_flush,
                name: "clflush_remote_hit",
                display_name: "clflush remote hit",
            },
            CalibrateOperation2T {
                prepare: noop::<u8>,
                op: only_flush,
                name: "clflush_miss",
                display_name: "clflush miss",
            },
        ];
        const HIT_INDEX: usize = 0;
        const MISS_INDEX: usize = 1;

        let mut calibrate_results2t_vec = Vec::new();

        let slicing = match get_cache_slicing(core_per_socket) {
            Some(s) => s,
            None => {
                return Err(FlushAndFlushError::NoSlicing);
            }
        };
        let h = |addr: usize| slicing.hash(addr).unwrap();

        for page in pages {
            // FIXME Cache line size is magic
            let mut r = unsafe {
                calibrate_fixed_freq_2_thread(
                    &page[0] as *const u8,
                    64,
                    page.len() as isize,
                    &mut core_pairs.clone(),
                    &operations,
                    CalibrationOptions {
                        hist_params: HistParams {
                            bucket_number: CFLUSH_BUCKET_NUMBER,
                            bucket_size: CFLUSH_BUCKET_SIZE,
                            iterations: CFLUSH_NUM_ITER,
                        },
                        verbosity: Verbosity::NoOutput,
                        optimised_addresses: true,
                    },
                    core_per_socket,
                )
            };
            calibrate_results2t_vec.append(&mut r);
        }
        let analysis: HashMap<ASVP, ThresholdError> = calibration_result_to_ASVP(
            calibrate_results2t_vec,
            |cal_1t_res| {
                let e = ErrorPredictions::predict_errors(HistogramCumSum::from_calibrate(
                    cal_1t_res, HIT_INDEX, MISS_INDEX,
                ));
                PotentialThresholds::minimizing_total_error(e)
                    .median()
                    .unwrap()
            },
            &h,
        )
        .map_err(|e| FlushAndFlushError::Nix(e))?;

        let asvp_best_av_errors: HashMap<AV, (ErrorPrediction, HashMap<SP, ThresholdError>)> =
            accumulate(
                analysis,
                |asvp: ASVP| AV {
                    attacker: asvp.attacker,
                    victim: asvp.victim,
                },
                || (ErrorPrediction::default(), HashMap::new()),
                |acc: &mut (ErrorPrediction, HashMap<SP, ThresholdError>),
                 threshold_error,
                 asvp: ASVP,
                 av| {
                    assert_eq!(av.attacker, asvp.attacker);
                    assert_eq!(av.victim, asvp.victim);
                    let sp = SP {
                        slice: asvp.slice,
                        page: asvp.page,
                    };
                    acc.0 += threshold_error.error;
                    acc.1.insert(sp, threshold_error);
                },
            );
        Ok(asvp_best_av_errors)
    }

    fn new_with_core_pairs(
        core_pairs: impl Iterator<Item = (usize, usize)> + Clone,
    ) -> Result<(Self, usize, usize), FlushAndFlushError> {
        let m = MMappedMemory::new(PAGE_LEN, false);
        let array: &[u8] = m.slice();

        let mut res = Self::calibration_for_core_pairs(core_pairs, vec![array].into_iter())?;

        let mut best_error_rate = 1.0;
        let mut best_av = Default::default();

        // Select the proper core

        for (av, (global_error_pred, thresholds)) in res.iter() {
            if global_error_pred.error_rate() < best_error_rate {
                best_av = *av;
                best_error_rate = global_error_pred.error_rate();
            }
        }
        Self::new(best_av.attacker, best_av.victim)
            .map(|this| (this, best_av.attacker, best_av.victim))

        // Set no threshold as calibrated on local array that will get dropped.
    }

    pub fn new_any_single_core() -> Result<(Self, CpuSet, usize), FlushAndFlushError> {
        // Generate core iterator
        let mut core_pairs: Vec<(usize, usize)> = Vec::new();

        let old = sched_getaffinity(Pid::from_raw(0)).unwrap();

        for i in 0..CpuSet::count() {
            if old.is_set(i).unwrap() {
                core_pairs.push((i, i));
            }
        }

        // Generate all single core pairs

        // Call out to private constructor that takes a core pair list, determines best and makes the choice.
        // The private constructor will set the correct affinity for main (attacker thread)

        Self::new_with_core_pairs(core_pairs.into_iter()).map(|(channel, attacker, victim)| {
            assert_eq!(attacker, victim);
            (channel, old, attacker)
        })
    }

    pub fn new_any_two_core(
        distinct: bool,
    ) -> Result<(Self, CpuSet, usize, usize), FlushAndFlushError> {
        let old = sched_getaffinity(Pid::from_raw(0)).unwrap();

        let mut core_pairs: Vec<(usize, usize)> = Vec::new();

        for i in 0..CpuSet::count() {
            if old.is_set(i).unwrap() {
                for j in 0..CpuSet::count() {
                    if old.is_set(j).unwrap() {
                        if i != j || !distinct {
                            core_pairs.push((i, j));
                        }
                    }
                }
            }
        }

        Self::new_with_core_pairs(core_pairs.into_iter()).map(|(channel, attacker, victim)| {
            if distinct {
                assert_ne!(attacker, victim);
            }
            (channel, old, attacker, victim)
        })
    }

    fn get_slice(&self, addr: *const u8) -> Slice {
        self.slicing.hash(addr as usize).unwrap()
    }

    pub fn set_cores(&mut self, attacker: usize, victim: usize) -> Result<(), FlushAndFlushError> {
        let old_attacker = self.attacker_core;
        let old_victim = self.victim_core;

        self.attacker_core = attacker;
        self.victim_core = victim;

        let pages: Vec<VPN> = self
            .thresholds
            .keys()
            .map(|sp: &SP| sp.page)
            //.copied()
            .collect();
        match self.recalibrate(pages) {
            Ok(()) => Ok(()),
            Err(e) => {
                self.attacker_core = old_attacker;
                self.victim_core = old_victim;
                Err(e)
            }
        }
    }

    fn recalibrate(
        &mut self,
        pages: impl IntoIterator<Item = VPN>,
    ) -> Result<(), FlushAndFlushError> {
        // unset readiness status.
        // Call calibration with core pairs with a single core pair
        // Use results \o/ (or error out)

        self.addresses_ready.clear();

        // Fixme refactor in depth core pairs to make explicit main vs helper.
        let core_pairs = vec![(self.attacker_core, self.victim_core)];

        let pages: HashSet<&[u8]> = self
            .thresholds
            .keys()
            .map(|sp: &SP| unsafe { &*slice_from_raw_parts(sp.page as *const u8, PAGE_LEN) })
            .collect();

        let mut res = Self::calibration_for_core_pairs(core_pairs.into_iter(), pages.into_iter())?;
        assert_eq!(res.keys().count(), 1);
        self.thresholds = res
            .remove(&AV {
                attacker: self.attacker_core,
                victim: self.victim_core,
            })
            .unwrap()
            .1;
        Ok(())
    }

    unsafe fn test_impl<'a, 'b, 'c>(
        &'a self,
        addresses: &'b mut (impl Iterator<Item = &'c *const u8> + Clone),
        limit: u32,
    ) -> Result<Vec<(*const u8, CacheStatus)>, SideChannelError> {
        let mut result = Vec::new();
        let mut tmp = Vec::new();
        let mut i = 0;
        for addr in addresses {
            i += 1;
            let t = unsafe { only_flush(*addr) };
            tmp.push((addr, t));
            if i == limit {
                break;
            }
        }
        for (addr, time) in tmp {
            if !self.addresses_ready.contains(&addr) {
                return Err(AddressNotReady(*addr));
            }
            let vpn: VPN = (*addr as usize) & (!0xfff); // FIXME
            let slice = self.get_slice(*addr);
            let threshold_error = &self.thresholds[&SP { slice, page: vpn }];
            // refactor this into a struct threshold method ?
            if threshold_error.threshold.is_hit(time) {
                result.push((*addr, CacheStatus::Hit))
            } else {
                result.push((*addr, CacheStatus::Miss))
            }
        }
        Ok(result)
    }

    unsafe fn prepare_impl<'a, 'b, 'c>(
        &'a mut self,
        addresses: &'b mut (impl Iterator<Item = &'c *const u8> + Clone),
        limit: u32,
    ) -> Result<(), SideChannelError> {
        use core::arch::x86_64 as arch_x86;
        let mut i = 0;
        let addresses_cloned = addresses.clone();
        for addr in addresses_cloned {
            i += 1;
            let vpn: VPN = get_vpn(*addr);
            let slice = self.get_slice(*addr);
            if self.addresses_ready.contains(&addr) {
                continue;
            }
            if !self.thresholds.contains_key(&SP { slice, page: vpn }) {
                return Err(AddressNotCalibrated(*addr));
            }
            if i == limit {
                break;
            }
        }
        i = 0;
        for addr in addresses {
            i += 1;
            unsafe { flush(*addr) };
            //println!("{:p}", *addr);
            self.addresses_ready.insert(*addr);
            if i == limit {
                break;
            }
        }
        unsafe { arch_x86::_mm_mfence() };
        Ok(())
    }
}

impl Debug for FlushAndFlush {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("FlushAndFlush")
            .field("thresholds", &self.thresholds)
            .field("addresses_ready", &self.addresses_ready)
            .field("slicing", &self.slicing)
            .finish()
    }
}

impl CoreSpec for FlushAndFlush {
    fn main_core(&self) -> CpuSet {
        let mut main = CpuSet::new();
        main.set(self.attacker_core);
        main
    }

    fn helper_core(&self) -> CpuSet {
        let mut helper = CpuSet::new();
        helper.set(self.victim_core);
        helper
    }
}

use cache_side_channel::CacheStatus::Hit;
use cache_utils::calibration::cum_sum;
use cache_utils::mmap::MMappedMemory;
use covert_channels_evaluation::{BitIterator, CovertChannel};
use std::ptr::slice_from_raw_parts;

impl MultipleAddrCacheSideChannel for FlushAndFlush {
    const MAX_ADDR: u32 = 3;

    unsafe fn test<'a, 'b, 'c>(
        &'a mut self,
        addresses: &'b mut (impl Iterator<Item = &'c *const u8> + Clone),
    ) -> Result<Vec<(*const u8, CacheStatus)>, SideChannelError> {
        unsafe { self.test_impl(addresses, Self::MAX_ADDR) }
    }

    unsafe fn prepare<'a, 'b, 'c>(
        &'a mut self,
        addresses: &'b mut (impl Iterator<Item = &'c *const u8> + Clone),
    ) -> Result<(), SideChannelError> {
        unsafe { self.prepare_impl(addresses, Self::MAX_ADDR) }
    }

    fn victim(&mut self, operation: &dyn Fn()) {
        operation(); // TODO use a different helper core ?
    }

    // TODO
    // To split into several functions
    // Calibration
    // Make predictions out of results -> probably in cache_utils
    //  Compute Threshold & Error
    // Compute stats from (A,V,S,P) into (A,V), or other models -> in cache_utils
    // Use a generic function ? fn <T> reduce (HashMap<(A,S,V,P), Result>, Fn (A,S,V,P) -> T, a reduction method)

    // Determine best core (A,V) amongst options -> in here
    // Extract results out of calibration -> in self.calibrate

    unsafe fn calibrate(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<(), ChannelFatalError> {
        let core_pair = vec![(self.attacker_core, self.victim_core)];

        let pages = addresses
            .into_iter()
            .map(|addr: *const u8| unsafe {
                &*slice_from_raw_parts(get_vpn(addr) as *const u8, PAGE_LEN)
            })
            .collect::<HashSet<&[u8]>>();

        let mut res =
            match Self::calibration_for_core_pairs(core_pair.into_iter(), pages.into_iter()) {
                Err(e) => {
                    return Err(ChannelFatalError::Oops);
                }
                Ok(r) => r,
            };
        assert_eq!(res.keys().count(), 1);
        let t = res
            .remove(&AV {
                attacker: self.attacker_core,
                victim: self.victim_core,
            })
            .unwrap()
            .1;

        for (sp, threshold) in t {
            //println!("Inserting sp: {:?} => Threshold: {:?}", sp, threshold);
            self.thresholds.insert(sp, threshold);
        }

        Ok(())
    }
}

unsafe impl Send for FlushAndFlush {}
unsafe impl Sync for FlushAndFlush {}

impl CovertChannel for SingleFlushAndFlush {
    const BIT_PER_PAGE: usize = 1; //PAGE_SHIFT - 6; // FIXME MAGIC cache line size

    unsafe fn transmit<'a>(&self, page: *const u8, bits: &mut BitIterator<'a>) {
        let mut offset = 0;

        let page = self.0.preferred_address[&page];

        if let Some(b) = bits.next() {
            //println!("Transmitting {} on page {:p}", b, page);
            if b {
                unsafe { only_reload(page) };
            } else {
                unsafe { only_flush(page) };
            }
        }
    }

    unsafe fn receive(&self, page: *const u8) -> Vec<bool> {
        let addresses: Vec<*const u8> = vec![self.0.preferred_address[&page]];
        let r = unsafe { self.0.test_impl(&mut addresses.iter(), u32::max_value()) };
        match r {
            Err(e) => panic!("{:?}", e),
            Ok(status_vec) => {
                assert_eq!(status_vec.len(), 1);
                let received = status_vec[0].1 == Hit;
                return vec![received];
            }
        }
    }

    unsafe fn ready_page(&mut self, page: *const u8) {
        let r = unsafe { self.0.calibrate(vec![page].into_iter()) }.unwrap();
        let mut best_error_rate = 1.0;
        let mut best_slice = 0;
        for (sp, threshold_error) in self
            .0
            .thresholds
            .iter()
            .filter(|kv| kv.0.page == page as VPN)
        {
            if threshold_error.error.error_rate() < best_error_rate {
                best_error_rate = threshold_error.error.error_rate();
                best_slice = sp.slice;
            }
        }
        for i in 0..PAGE_LEN {
            let addr = unsafe { page.offset(i as isize) };
            if self.0.get_slice(addr) == best_slice {
                self.0.preferred_address.insert(page, addr);
                let r = unsafe {
                    self.0
                        .prepare_impl(&mut vec![addr].iter(), u32::max_value())
                }
                .unwrap();
                break;
            }
        }
    }
}

impl CovertChannel for FlushAndFlush {
    const BIT_PER_PAGE: usize = 1; //PAGE_SHIFT - 6; // FIXME MAGIC cache line size

    unsafe fn transmit<'a>(&self, page: *const u8, bits: &mut BitIterator<'a>) {
        let mut offset = 0;

        if Self::BIT_PER_PAGE == 1 {
            let page = self.preferred_address[&page];

            if let Some(b) = bits.next() {
                //println!("Transmitting {} on page {:p}", b, page);
                if b {
                    unsafe { only_reload(page) };
                } else {
                    unsafe { only_flush(page) };
                }
            }
        } else {
            for i in 0..Self::BIT_PER_PAGE {
                if let Some(b) = bits.next() {
                    if b {
                        offset += 1 << i + 6; // Magic FIXME cache line size
                    }
                }
            }
            unsafe { maccess(page.offset(offset as isize)) };
        }
    }

    unsafe fn receive(&self, page: *const u8) -> Vec<bool> {
        if Self::BIT_PER_PAGE == 1 {
            let addresses: Vec<*const u8> = vec![self.preferred_address[&page]];
            let r = unsafe { self.test_impl(&mut addresses.iter(), u32::max_value()) };
            match r {
                Err(e) => panic!("{:?}", e),
                Ok(status_vec) => {
                    assert_eq!(status_vec.len(), 1);
                    let received = status_vec[0].1 == Hit;
                    //println!("Received {} on page {:p}", received, page);
                    return vec![received];
                }
            }
        } else {
            let addresses = (0..PAGE_LEN)
                .step_by(64)
                .map(|o| unsafe { page.offset(o as isize) })
                .collect::<HashSet<*const u8>>();
            let r = unsafe { self.test_impl(&mut addresses.iter(), u32::max_value()) };
            match r {
                Err(e) => panic!("{:?}", e),
                Ok(status_vec) => {
                    for (addr, status) in status_vec {
                        if status == Hit {
                            let offset = unsafe { addr.offset_from(page) } >> 6; // Fixme cache line size magic
                            let mut res = Vec::new();
                            for i in 0..Self::BIT_PER_PAGE {
                                res.push((offset & (1 << i)) != 0);
                            }
                            return res;
                        }
                    }
                }
            }
            vec![false; Self::BIT_PER_PAGE]
        }
    }

    unsafe fn ready_page(&mut self, page: *const u8) {
        let r = unsafe { self.calibrate(vec![page].into_iter()) }.unwrap();
        if Self::BIT_PER_PAGE == 1 {
            let mut best_error_rate = 1.0;
            let mut best_slice = 0;
            for (sp, threshold_error) in
                self.thresholds.iter().filter(|kv| kv.0.page == page as VPN)
            {
                if threshold_error.error.error_rate() < best_error_rate {
                    best_error_rate = threshold_error.error.error_rate();
                    best_slice = sp.slice;
                }
            }
            for i in 0..PAGE_LEN {
                let addr = unsafe { page.offset(i as isize) };
                if self.get_slice(addr) == best_slice {
                    self.preferred_address.insert(page, addr);
                    let r = unsafe { self.prepare_impl(&mut vec![addr].iter(), u32::max_value()) }
                        .unwrap();

                    break;
                }
            }
        } else {
            let addresses = (0..PAGE_LEN)
                .step_by(64)
                .map(|o| unsafe { page.offset(o as isize) })
                .collect::<Vec<*const u8>>();
            //println!("{:#?}", addresses);
            let r = unsafe { self.prepare_impl(&mut addresses.iter(), u32::max_value()) }.unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
