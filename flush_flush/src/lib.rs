#![feature(unsafe_block_in_unsafe_fn)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod naive;

use cache_side_channel::SideChannelError::{AddressNotCalibrated, AddressNotReady};
use cache_side_channel::{
    CacheStatus, ChannelFatalError, MultipleAddrCacheSideChannel, SideChannelError,
    SingleAddrCacheSideChannel,
};
use cache_utils::calibration::{
    calibrate_fixed_freq_2_thread, get_cache_slicing, get_vpn, only_flush, CalibrateOperation2T,
    CalibrationOptions, HistParams, Verbosity, CFLUSH_BUCKET_NUMBER, CFLUSH_BUCKET_SIZE,
    CFLUSH_NUM_ITER, PAGE_LEN,
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
}

#[derive(Debug)]
pub enum FlushAndFlushError {
    NoSlicing,
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
                            iterations: CFLUSH_NUM_ITER << 1,
                        },
                        verbosity: Verbosity::NoOutput,
                        optimised_addresses: true,
                    },
                    core_per_socket,
                )
            };
            calibrate_results2t_vec.append(&mut r);
        }
        unimplemented!();
    }

    fn new_with_core_pairs(
        core_pairs: impl Iterator<Item = (usize, usize)> + Clone,
    ) -> Result<(Self, usize, usize), FlushAndFlushError> {
        let m = MMappedMemory::new(PAGE_LEN);
        let array: &[u8] = m.slice();

        let res = Self::calibration_for_core_pairs(core_pairs, vec![array].into_iter());

        // Call the calibration function on a local page sized buffer.

        // Classical analysis flow to generate all ASVP, Threshold, Error.

        // Reduction to determine average / max error for each core.

        // Select the proper core
        unimplemented!();
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

    pub fn set_cores(&mut self, attacker: usize, victim: usize) -> Result<(), nix::Error> {
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

    fn recalibrate(&mut self, pages: impl IntoIterator<Item = VPN>) -> Result<(), nix::Error> {
        // unset readiness status.
        // Call calibration with core pairs with a single core pair
        // Use results \o/ (or error out)

        unimplemented!();
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

use cache_utils::calibration::cum_sum;
use cache_utils::mmap::MMappedMemory;

impl MultipleAddrCacheSideChannel for FlushAndFlush {
    const MAX_ADDR: u32 = 3;

    unsafe fn test<'a, 'b, 'c>(
        &'a mut self,
        addresses: &'b mut (impl Iterator<Item = &'c *const u8> + Clone),
    ) -> Result<Vec<(*const u8, CacheStatus)>, SideChannelError> {
        let mut result = Vec::new();
        let mut tmp = Vec::new();
        let mut i = 0;
        for addr in addresses {
            i += 1;
            let t = unsafe { only_flush(*addr) };
            tmp.push((addr, t));
            if i == Self::MAX_ADDR {
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

    unsafe fn prepare<'a, 'b, 'c>(
        &'a mut self,
        addresses: &'b mut (impl Iterator<Item = &'c *const u8> + Clone),
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
            if i == Self::MAX_ADDR {
                break;
            }
        }
        i = 0;
        for addr in addresses {
            i += 1;
            unsafe { flush(*addr) };
            self.addresses_ready.insert(*addr);
            if i == Self::MAX_ADDR {
                break;
            }
        }
        unsafe { arch_x86::_mm_mfence() };
        Ok(())
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
        unimplemented!()
        /*
        let mut pages = HashMap::<VPN, HashSet<*const u8>>::new();
        for addr in addresses {
            let page = get_vpn(addr);
            pages.entry(page).or_insert_with(HashSet::new).insert(addr);
        }

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

        // Generate core iterator
        let mut core_pairs: Vec<(usize, usize)> = Vec::new();

        let old = sched_getaffinity(Pid::from_raw(0)).unwrap();

        for i in 0..CpuSet::count() {
            if old.is_set(i).unwrap() {
                core_pairs.push((i, i));
            }
        }

        // Probably needs more metadata
        let mut per_core: HashMap<usize, HashMap<VPN, HashMap<Slice, (Threshold, f32)>>> =
            HashMap::new();

        let mut core_averages: HashMap<usize, (f32, u32)> = HashMap::new();

        for (page, _) in pages {
            let p = page as *const u8;
            let r = unsafe {
                calibrate_fixed_freq_2_thread(
                    p,
                    64,                // FIXME : MAGIC
                    PAGE_LEN as isize, // MAGIC
                    &mut core_pairs.clone().into_iter(),
                    &operations,
                    CalibrationOptions {
                        hist_params: HistParams {
                            bucket_number: CFLUSH_BUCKET_NUMBER,
                            bucket_size: CFLUSH_BUCKET_SIZE,
                            iterations: CFLUSH_NUM_ITER << 1,
                        },
                        verbosity: Verbosity::NoOutput,
                        optimised_addresses: true,
                    },
                    core_per_socket,
                )
            };

            /* TODO refactor a good chunk of calibration result analysis to make thresholds in a separate function
            Generating Cumulative Sums and then using that to compute error count for each possible threshold is a recurring joke.
            It might be worth in a second time to refactor this to handle more generic strategies (such as double thresholds)
            What about handling non attributes values (time values that are not attributed as hit or miss)
            */

            /*

            Non Naive F+F flow
            Vec<CalibrationResult2T> -> ASVP,Thresholds,Error Does not care as much. Can probably re-use functions to build a single one.
            Add API to query predicted error rate, compare with covert channel result.
            */

            for result2t in r {
                if result2t.main_core != result2t.helper_core {
                    panic!("Unexpected core numbers");
                }
                let core = result2t.main_core;
                match result2t.res {
                    Err(e) => panic!("Oops: {:#?}", e),
                    Ok(results_1t) => {
                        for r1t in results_1t {
                            // This will be turned into map_values style functions + Calibration1T -> Reasonable Type

                            // Already handled
                            let offset = r1t.offset;
                            let addr = unsafe { p.offset(offset) };
                            let slice = self.get_slice(addr);

                            // To Raw histogram
                            let miss_hist = &r1t.histogram[MISS_INDEX];
                            let hit_hist = &r1t.histogram[HIT_INDEX];
                            if miss_hist.len() != hit_hist.len() {
                                panic!("Maformed results");
                            }
                            let len = miss_hist.len();

                            // Cum Sums
                            let miss_cum_sum = cum_sum(miss_hist);
                            let hit_cum_sum = cum_sum(hit_hist);
                            let miss_total = miss_cum_sum[len - 1];
                            let hit_total = hit_cum_sum[len - 1];

                            // Error rate per threshold computations

                            // Threshold is less than equal => miss, strictly greater than => hit
                            let mut error_miss_less_than_hit = vec![0; len - 1];
                            // Threshold is less than equal => hit, strictly greater than => miss
                            let mut error_hit_less_than_miss = vec![0; len - 1];

                            let mut min_error_hlm = u32::max_value();
                            let mut min_error_mlh = u32::max_value();

                            for i in 0..(len - 1) {
                                error_hit_less_than_miss[i] =
                                    miss_cum_sum[i] + (hit_total - hit_cum_sum[i]);
                                error_miss_less_than_hit[i] =
                                    hit_cum_sum[i] + (miss_total - miss_cum_sum[i]);

                                if error_hit_less_than_miss[i] < min_error_hlm {
                                    min_error_hlm = error_hit_less_than_miss[i];
                                }
                                if error_miss_less_than_hit[i] < min_error_mlh {
                                    min_error_mlh = error_miss_less_than_hit[i];
                                }
                            }

                            let hlm = min_error_hlm < min_error_mlh;

                            let (errors, min_error) = if hlm {
                                (&error_hit_less_than_miss, min_error_hlm)
                            } else {
                                (&error_miss_less_than_hit, min_error_mlh)
                            };

                            // Find the min -> gives potetial thresholds with info
                            let mut potential_thresholds = Vec::new();

                            for i in 0..errors.len() {
                                if errors[i] == min_error {
                                    let num_true_hit;
                                    let num_false_hit;
                                    let num_true_miss;
                                    let num_false_miss;
                                    if hlm {
                                        num_true_hit = hit_cum_sum[i];
                                        num_false_hit = miss_cum_sum[i];
                                        num_true_miss = miss_total - num_false_hit;
                                        num_false_miss = hit_total - num_true_hit;
                                    } else {
                                        num_true_miss = miss_cum_sum[i];
                                        num_false_miss = hit_cum_sum[i];
                                        num_true_hit = hit_total - num_false_miss;
                                        num_false_hit = miss_total - num_true_miss;
                                    }
                                    potential_thresholds.push((
                                        i,
                                        num_true_hit,
                                        num_false_hit,
                                        num_true_miss,
                                        num_false_miss,
                                        min_error as f32 / (hit_total + miss_total) as f32,
                                    ));
                                }
                            }

                            let index = (potential_thresholds.len() - 1) / 2;
                            let (threshold, _, _, _, _, error_rate) = potential_thresholds[index];
                            // insert in per_core
                            if per_core
                                .entry(core)
                                .or_insert_with(HashMap::new)
                                .entry(page)
                                .or_insert_with(HashMap::new)
                                .insert(
                                    slice,
                                    (
                                        Threshold {
                                            bucket_index: threshold, // FIXME the bucket to time conversion
                                            miss_faster_than_hit: !hlm,
                                        },
                                        error_rate,
                                    ),
                                )
                                .is_some()
                            {
                                panic!("Duplicate slice result");
                            }
                            let core_average = core_averages.get(&core).unwrap_or(&(0.0, 0));
                            let new_core_average =
                                (core_average.0 + error_rate, core_average.1 + 1);
                            core_averages.insert(core, new_core_average);
                        }
                    }
                }
            }
        }

        // We now get ASVP stuff with the correct core(in theory)

        // We now have a HashMap associating stuffs to cores, iterate on it and select the best.
        let mut best_core = 0;

        let mut best_error_rate = {
            let ca = core_averages[&0];
            ca.0 / ca.1 as f32
        };
        for (core, average) in core_averages {
            let error_rate = average.0 / average.1 as f32;
            if error_rate < best_error_rate {
                best_core = core;
                best_error_rate = error_rate;
            }
        }
        let mut thresholds = HashMap::new();
        println!("Best core: {}, rate: {}", best_core, best_error_rate);
        let tmp = per_core.remove(&best_core).unwrap();
        for (page, per_page) in tmp {
            let page_entry = thresholds.entry(page).or_insert_with(HashMap::new);
            for (slice, per_slice) in per_page {
                println!(
                    "page: {:x}, slice: {}, threshold: {:?}, error_rate: {}",
                    page, slice, per_slice.0, per_slice.1
                );
                page_entry.insert(slice, per_slice.0);
            }
        }
        self.thresholds = thresholds;
        println!("{:#?}", self.thresholds);

        // TODO handle error better for affinity setting and other issues.

        self.addresses_ready.clear();

        let mut cpuset = CpuSet::new();
        cpuset.set(best_core).unwrap();
        sched_setaffinity(Pid::from_raw(0), &cpuset).unwrap();
        Ok(())
        */
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
