#![deny(unsafe_op_in_unsafe_fn)]

// TODO

// Common logic for the ability to calibrate along slices
// Core issues should be orthogonal
// Extend to multithread ?

// Should be used by F+F and non Naive F+R

use std::cmp::Ordering;
//use crate::naive::NaiveTimingChannelHandle;
use cache_side_channel::table_side_channel::{
    MultipleTableCacheSideChannel, SingleTableCacheSideChannel, TableAttackResult,
    TableCacheSideChannel,
};
use cache_side_channel::{
    BitIterator, CacheStatus, ChannelFatalError, ChannelHandle, CovertChannel, LocationSpec,
    MultipleAddrCacheSideChannel, SideChannelError, SingleAddrCacheSideChannel,
};
use cache_utils::calibration::{
    CALIBRATION_WARMUP_ITER, CalibrateOperation2T, CalibrationOptions, HashMap, PAGE_LEN,
    PAGE_SHIFT, Verbosity, calibrate_fixed_freq_2_thread_numa, get_cache_attack_slicing, get_vpn,
    only_flush, only_reload,
};
use cache_utils::mmap::MMappedMemory;
use cache_utils::{find_core_per_socket, flush, maccess, noop};
use calibration_results::calibration::{
    AVMLocation, CoreLocation, ErrorPrediction, LocationParameters, PartialLocation,
    PartialLocationOwned, StaticHistCalibrateResult, VPN,
};
use calibration_results::calibration_2t::{
    CalibrateResult2TNuma, calibration_result_to_location_map,
};
use calibration_results::classifiers::{ErrorPredictionsBuilder, ErrorPredictor, HitClassifier};
use calibration_results::histograms::{SimpleBucketU64, StaticHistogram, StaticHistogramCumSum};
use calibration_results::{map_values, reduce};
use cpuid::complex_addressing::CacheAttackSlicing;
use nix::sched::CpuSet;
use nix::sched::sched_getaffinity;
use nix::unistd::Pid;
use numa_utils::NumaNode;
use numa_utils::numa_node_of_cpu;
use rand::seq::IndexedRandom;
use std::collections::HashSet;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::ptr::{null, slice_from_raw_parts};

pub mod naive;
pub mod topology_aware_generic_classifier;
pub mod topology_aware_single_threshold;

const CACHE_LINE_LENGTH: usize = 64; // FIXME MAGIC to be autodetected.

pub trait TimingChannelPrimitives: Debug + Send + Sync + Default {
    unsafe fn reset(&self, addr: *const u8);
    unsafe fn attack(&self, addr: *const u8) -> u64;
    unsafe fn attack_reset(&self, addr: *const u8) -> u64;
    //const NEED_RESET: bool;
}

#[derive(Debug)]
pub struct TopologyAwareTimingChannelHandle<
    const WIDTH: u64,
    const N: usize,
    T: HitClassifier<SimpleBucketU64<WIDTH, N>>,
> {
    threshold: T, // Todo, reduce copies ?
    addr: *const u8,
    ready: bool,
    calibration_epoch: usize,
}

pub struct CovertChannelHandle<T: MultipleAddrCacheSideChannel>(T::Handle);

impl<const WIDTH: u64, const N: usize, T: HitClassifier<SimpleBucketU64<WIDTH, N>>> ChannelHandle
    for TopologyAwareTimingChannelHandle<WIDTH, N, T>
{
    fn to_const_u8_pointer(&self) -> *const u8 {
        self.addr
    }
}

#[derive(Debug)]
pub enum TopologyAwareError {
    NoSlicing,
    Nix(nix::Error),
    NeedRecalibration,
    UnsupportedParameterValue,
    Oops,
}

#[derive(Clone)]
pub struct TopologyAwareTimingChannel<
    const WIDTH: u64,
    const N: usize,
    T: TimingChannelPrimitives,
    E: ErrorPredictionsBuilder<WIDTH, N>,
    Norm: Ord + Debug + Clone,
    NFThres: Fn(&Vec<ErrorPrediction>) -> Norm,
    NFLoc: Fn(&Vec<(Norm, Vec<ErrorPrediction>)>) -> Norm,
    const COVERT_CHANNEL_RESET: bool = true,
> {
    slicing: CacheAttackSlicing,
    t: T,
    fixed_location: (Option<NumaNode>, Option<usize>, Option<usize>),
    calibration_granularity: LocationParameters,
    threshold_granularity: LocationParameters,
    thresholds: HashMap<PartialLocationOwned, (E::E, Norm)>,
    norm_threshold: NFThres,
    norm_location: NFLoc,
    //addresses: HashSet<*const u8>,
    preferred_address: HashMap<VPN, *const u8>,
    calibration_epoch: usize,
    calibration_iterations: u32,
    // Note, this is very likely never needed, in fact.
    error_prediction_builder: E,
    // And error prediction builder, or set of error prediction builder is only needed for polymorphic versions.
}

unsafe impl<
    const WIDTH: u64,
    const N: usize,
    T: TimingChannelPrimitives + Send,
    E: ErrorPredictionsBuilder<WIDTH, N> + Send,
    Norm: Ord + Send + Debug + Clone,
    NFThres: Send + Fn(&Vec<ErrorPrediction>) -> Norm,
    NFLoc: Send + Fn(&Vec<(Norm, Vec<ErrorPrediction>)>) -> Norm,
    const COVERT_CHANNEL_RESET: bool,
> Send for TopologyAwareTimingChannel<WIDTH, N, T, E, Norm, NFThres, NFLoc, COVERT_CHANNEL_RESET>
{
}

unsafe impl<
    const WIDTH: u64,
    const N: usize,
    T: TimingChannelPrimitives + Sync,
    E: ErrorPredictionsBuilder<WIDTH, N> + Sync,
    Norm: Ord + Sync + Debug + Clone,
    NFThres: Sync + Fn(&Vec<ErrorPrediction>) -> Norm,
    NFLoc: Sync + Fn(&Vec<(Norm, Vec<ErrorPrediction>)>) -> Norm,
    const COVERT_CHANNEL_RESET: bool,
> Sync for TopologyAwareTimingChannel<WIDTH, N, T, E, Norm, NFThres, NFLoc, COVERT_CHANNEL_RESET>
{
}

impl<
    const WIDTH: u64,
    const N: usize,
    T: TimingChannelPrimitives,
    E: ErrorPredictionsBuilder<WIDTH, N>,
    Norm: Ord + Debug + Clone,
    NFThres: Fn(&Vec<ErrorPrediction>) -> Norm,
    NFLoc: Fn(&Vec<(Norm, Vec<ErrorPrediction>)>) -> Norm,
    const COVERT_CHANNEL_RESET: bool,
> TopologyAwareTimingChannel<WIDTH, N, T, E, Norm, NFThres, NFLoc, COVERT_CHANNEL_RESET>
{
    pub fn new(
        fixed_location: (Option<NumaNode>, Option<usize>, Option<usize>), // Those might need to change to a simpler types (core numbers, numa node ?)
        calibration_granularity: LocationParameters,
        threshold_granularity: LocationParameters,
        norm_threshold: NFThres,
        norm_location: NFLoc,
        e: E,
        calibration_iterations: u32,
    ) -> Result<Self, TopologyAwareError> {
        if !threshold_granularity.is_subset(&calibration_granularity) {
            //panic!("2");
            return Err(TopologyAwareError::UnsupportedParameterValue);
        }
        if let Some(slicing) = get_cache_attack_slicing(find_core_per_socket(), CACHE_LINE_LENGTH) {
            Ok(Self {
                thresholds: Default::default(),
                norm_threshold,
                norm_location,
                //addresses: Default::default(),
                slicing,
                preferred_address: Default::default(),
                t: Default::default(),
                fixed_location: (fixed_location.0, fixed_location.1, fixed_location.2),
                calibration_granularity,
                calibration_epoch: 0,
                threshold_granularity,
                calibration_iterations,
                error_prediction_builder: e,
            })
        } else {
            //panic!("3");
            Err(TopologyAwareError::NoSlicing)
        }
    }

    // Takes a buffer / list of addresses or pages
    // Takes a list of core pairs
    // Run optimized calibration and processes results
    fn calibration_for_locations<'a, 'b, 'c>(
        t: &T,
        locations: impl Iterator<Item = (NumaNode, usize, usize)> + Clone,
        pages: impl Iterator<Item = &'a [u8]>,
        calibration_granularity: &'b LocationParameters,
        threshold_granularity: &'c LocationParameters,
        norm_threshold: &NFThres,
        error_prediction_builder: &E,
        calibration_iterations: u32,
    ) -> Result<HashMap<PartialLocationOwned, (E::E, Norm, Vec<ErrorPrediction>)>, TopologyAwareError>
    {
        println!("Calibrating...");
        let core_per_socket = find_core_per_socket();

        let operations = [

            CalibrateOperation2T {
                prepare: maccess::<u8>,
                op: if COVERT_CHANNEL_RESET {T::attack_reset} else {T::attack},
                name: "hit",
                display_name: "hit",
                t: &t,
            },
            CalibrateOperation2T {
                prepare: if COVERT_CHANNEL_RESET {noop::<u8>} else {flush},
                op: if COVERT_CHANNEL_RESET {T::attack_reset} else {T::attack},
                name: "miss",
                display_name: "miss",
                t: &t,
            },
        ];
        const HIT_INDEX: usize = 0;
        const MISS_INDEX: usize = 1;

        let mut calibrate_results2t_vec = Vec::new();

        let slicing = match get_cache_attack_slicing(core_per_socket, CACHE_LINE_LENGTH) {
            Some(s) => s,
            None => {
                return Err(TopologyAwareError::NoSlicing);
            }
        };

        let h = match slicing {
            CacheAttackSlicing::Unsupported(_) => |slice: usize| {
                const PAGE_MASK: usize = (1 << PAGE_SHIFT) - 1;
                let r = (slice & PAGE_MASK) / CACHE_LINE_LENGTH;
                if r > (u8::MAX as usize) {
                    panic!("Unsupported, the size of slice in AVMLocation is too small");
                }
                r
            },
            CacheAttackSlicing::ComplexAddressing(_) => |slice: usize| slice,
            CacheAttackSlicing::SimpleAddressing(_) => |slice: usize| {
                if slice > (u8::MAX as usize) {
                    panic!("Unsupported, the size of slice in AVMLocation is too small");
                }
                slice
            },
            CacheAttackSlicing::NoSlice => |slice: usize| 0,
        };

        for page in pages {
            // FIXME Cache line size is magic
            let mut r: Result<Vec<CalibrateResult2TNuma<WIDTH, N>>, _> = unsafe {
                calibrate_fixed_freq_2_thread_numa(
                    &page[0] as *const u8,
                    CACHE_LINE_LENGTH,
                    page.len() as isize,
                    &mut locations.clone(),
                    &operations,
                    CalibrationOptions {
                        iterations: calibration_iterations,
                        verbosity: Verbosity::NoOutput,
                        optimised_addresses: true,
                        measure_hash: false,
                        warmup_iterations: CALIBRATION_WARMUP_ITER,
                    },
                    core_per_socket,
                )
            };
            if let Ok(mut r) = r {
                calibrate_results2t_vec.append(&mut r);
            } else {
                panic!("calibration failed");
            }
        }

        let core_location = |core: usize| unsafe {
            // Eventually we need to integrate https://docs.rs/raw-cpuid/latest/raw_cpuid/struct.ExtendedTopologyIter.html
            let node = numa_node_of_cpu(core).unwrap().into();
            CoreLocation {
                socket: node,
                core: core as u16,
            }
        }; // FIXME we need to fix the socket number

        let calibration_analysis = calibration_result_to_location_map(
            calibrate_results2t_vec,
            &|calibration_results_1run: StaticHistCalibrateResult<WIDTH, N>| {
                let mut hits = None;
                let mut miss = None;
                for (i, hist) in calibration_results_1run.histogram.into_iter().enumerate() {
                    if i == HIT_INDEX {
                        hits = Some(hist);
                    } else if i == MISS_INDEX {
                        miss = Some(hist);
                    }
                }
                (hits.unwrap(), miss.unwrap())
            },
            &h,
            &core_location,
        );

        // Now reduce according to calibration_granularity
        let consolidated_histograms = reduce(
            calibration_analysis,
            |location: AVMLocation| PartialLocationOwned::new(*calibration_granularity, location),
            || {
                (
                    StaticHistogram::<WIDTH, N>::empty(),
                    StaticHistogram::<WIDTH, N>::empty(),
                )
            },
            |acc, val, _, _| {
                acc.0 += &val.0;
                acc.1 += &val.1;
            },
            |acc, _| {
                (
                    StaticHistogramCumSum::from(acc.0),
                    StaticHistogramCumSum::from(acc.1),
                )
            },
        );

        // Then enumerate error predictors
        let classifiers = error_prediction_builder.enumerate_classifiers();
        let len = classifiers.len();
        // Then do the error predictions
        let errors = map_values(consolidated_histograms, |hists, _k| {
            let mut res = Vec::new();
            for classifier in classifiers.iter() {
                let r = classifier.error_prediction(&hists.0, &hists.1);
                res.push(r);
            }
            res
        });
        // Then pick the best according to Norm.
        // This is done with a rather complex call to reduce.
        // The accumulator is a vector of vectors of error predictions, whose first index is the classifier index.
        // The aggregation adds up the various elements of the vector mapping classifier to prediction errors inside those.
        // Then reduction can call the norm to measure the norm, and pick the best.
        let result = reduce(
            errors,
            |partial_location| {
                /* Need to implement projection here*/
                partial_location.project(threshold_granularity)
            },
            || vec![vec![]; len],
            |acc, v, _k, _rk| {
                for i in 0..len {
                    acc[i].push(v[i]);
                }
            },
            |accumulator, _rk| {
                let mut minimum = None;
                let mut antecedents = Vec::new();
                for i in 0..len {
                    let norm = norm_threshold(&accumulator[i]);
                    if let Some(ref min) = minimum {
                        match Ord::cmp(min, &norm) {
                            Ordering::Less => {}
                            Ordering::Equal => {
                                antecedents.push((&classifiers[i], i));
                            }
                            Ordering::Greater => {
                                antecedents.clear();
                                antecedents.push((&classifiers[i], i));
                                minimum = Some(norm);
                            }
                        }
                    } else {
                        minimum = Some(norm);
                        antecedents.push((&classifiers[i], i));
                    }
                }
                assert_ne!(minimum, None);
                let norm = minimum.unwrap();
                let best = error_prediction_builder
                    .select_best_classifier(antecedents)
                    .unwrap();
                ((*(best.0)).clone(), norm, accumulator[best.1].clone())
            },
        );
        if result.iter().count() == 1 {
            let threshold = result.iter().next().unwrap();
            eprintln!(
                "Single Threshold: {:?} ({:?})",
                threshold.1.0, threshold.1.1
            );
        }
        Ok(result)
    }

    pub fn new_with_locations(
        locations: impl Iterator<Item = (NumaNode, usize, usize)> + Clone,
        calibration_granularity: LocationParameters,
        threshold_granularity: LocationParameters,
        norm_threshold: NFThres,
        norm_location: NFLoc,
        e: E,
        calibration_iterations: u32,
    ) -> Result<(Self, (Option<NumaNode>, Option<usize>, Option<usize>)), TopologyAwareError> {
        if !threshold_granularity.is_subset(&calibration_granularity) {
            panic!("1");
            return Err(TopologyAwareError::UnsupportedParameterValue);
        }

        /*if !threshold_granularity.attacker.core
            || threshold_granularity.victim.core
            || !threshold_granularity.memory_numa_node
        {
            println!("FIXME Unsupported lower granularity");
            return Err(TopologyAwareError::UnsupportedParameterValue);
        }*/

        let m = MMappedMemory::new(PAGE_LEN, false, false, |i| i as u8);
        let array: &[u8] = m.slice();

        let t = Default::default();

        let res = Self::calibration_for_locations(
            &t,
            locations,
            vec![array].into_iter(),
            &calibration_granularity,
            &threshold_granularity,
            &norm_threshold,
            &e,
            calibration_iterations,
        )?;

        let per_location = reduce(
            res,
            |location| {
                let numa = location.get_numa_node();
                let attacker = location.get_attacker_core();
                let victim = location.get_victim_core();
                (numa, attacker, victim)
            },
            || HashMap::new(),
            |acc, v, k, _rk| {
                acc.insert(k, v);
            },
            |acc, _rk| {
                let error_preds_vec = acc
                    .values()
                    .map(|(_classifier, n, v)| (n.clone(), v.clone()))
                    .collect::<Vec<_>>();
                let thresholds = acc
                    .iter()
                    .map(|(k, (classifier, norm, _error_predictions))| {
                        (k.clone(), (classifier.clone(), norm.clone()))
                    })
                    .collect::<HashMap<_, _>>();
                (acc, error_preds_vec)
            },
        );

        let (chosen_location, (hashmap, _v)) = per_location
            .into_iter()
            .min_by_key(|(k, (_h, v))| norm_location(v))
            .unwrap();

        let location = (
            chosen_location.0,
            chosen_location.1.map(|a| a as usize),
            chosen_location.2.map(|a| a as usize),
        );

        // Set no threshold as calibrated on local array that will get dropped.
        Self::new(
            location,
            calibration_granularity,
            threshold_granularity,
            norm_threshold,
            norm_location,
            e,
            calibration_iterations,
        )
        .map(|mut this| {
            if !this.is_memory_target_sensitive() {
                let thresholds = map_values(hashmap, |(e, n, v), _k| (e, n));
                this.thresholds = thresholds;
            }
            (this, location)
        })
    }

    pub fn new_any_single_core(
        calibration_granularity: LocationParameters,
        threshold_granularity: LocationParameters,
        norm_threshold: NFThres,
        norm_location: NFLoc,
        e: E,
        calibration_iterations: u32,
    ) -> Result<(Self, CpuSet, Option<NumaNode>, Option<usize>), TopologyAwareError> {
        // Generate core iterator
        let mut locations: Vec<(NumaNode, usize, usize)> = Vec::new();

        let old = sched_getaffinity(Pid::from_raw(0)).unwrap();
        let available_numa_nodes = numa_utils::available_nodes().unwrap();
        for i in available_numa_nodes {
            for j in 0..CpuSet::count() {
                if old.is_set(j).unwrap() {
                    locations.push((i, j, j));
                }
            }
        }
        // Generate all single core pairs

        // Call out to private constructor that takes a core pair list, determines best and makes the choice.
        // The private constructor will set the correct affinity for main (attacker thread)

        Self::new_with_locations(
            locations.into_iter(),
            calibration_granularity,
            threshold_granularity,
            norm_threshold,
            norm_location,
            e,
            calibration_iterations,
        )
        .map(|(channel, (node, attacker, victim))| {
            assert_eq!(attacker, victim);
            (channel, old, node, attacker)
        })
    }

    pub fn new_any_location(
        distinct: bool,
        calibration_granularity: LocationParameters,
        threshold_granularity: LocationParameters,
        norm_threshold: NFThres,
        norm_location: NFLoc,
        e: E,
        calibration_iterations: u32,
    ) -> Result<(Self, CpuSet, Option<NumaNode>, Option<usize>, Option<usize>), TopologyAwareError>
    {
        let old = sched_getaffinity(Pid::from_raw(0)).unwrap();

        let mut locations: Vec<(NumaNode, usize, usize)> = Vec::new();
        let available_numa_nodes = numa_utils::available_nodes().unwrap();
        for i in available_numa_nodes {
            for j in 0..CpuSet::count() {
                if old.is_set(j).unwrap() {
                    for k in 0..CpuSet::count() {
                        if old.is_set(k).unwrap() {
                            if j != k || !distinct {
                                locations.push((i, j, k));
                            }
                        }
                    }
                }
            }
        }

        Self::new_with_locations(
            locations.into_iter(),
            calibration_granularity,
            threshold_granularity,
            norm_threshold,
            norm_location,
            e,
            calibration_iterations,
        )
        .map(|(channel, (node, attacker, victim))| {
            if distinct {
                assert_ne!(attacker, victim);
            }
            (channel, old, node, attacker, victim)
        })
    }

    fn get_slice(&self, addr: *const u8) -> usize {
        // This will not work well if slicing is not known FIXME
        let slice = self.slicing.hash(addr as usize);

        match self.slicing {
            CacheAttackSlicing::Unsupported(_) => {
                const PAGE_MASK: usize = (1 << PAGE_SHIFT) - 1;
                let r = (slice & PAGE_MASK) / CACHE_LINE_LENGTH;
                if r > (u8::MAX as usize) {
                    panic!("Unsupported, the size of slice in AVMLocation is too small");
                }
                r
            }
            CacheAttackSlicing::ComplexAddressing(_) => slice,
            CacheAttackSlicing::SimpleAddressing(_) => {
                if slice > (u8::MAX as usize) {
                    panic!("Unsupported, the size of slice in AVMLocation is too small");
                }
                slice
            }
            CacheAttackSlicing::NoSlice => 0,
        }
    }

    fn is_memory_target_sensitive(&self) -> bool {
        self.threshold_granularity.memory_slice
            || self.threshold_granularity.memory_vpn
            || self.threshold_granularity.memory_offset
    }

    fn build_partial_location(&self, addr: *const u8) -> PartialLocationOwned {
        let vpn = get_vpn(addr);
        let attacker = self.fixed_location.1.unwrap_or_default();
        let victim = self.fixed_location.2.unwrap_or_default();
        let attacker_socket = numa_node_of_cpu(attacker).unwrap().into();
        let victim_socket = numa_node_of_cpu(victim).unwrap().into();
        let location = AVMLocation {
            attacker: CoreLocation {
                socket: attacker_socket,
                core: attacker as u16,
            },
            victim: CoreLocation {
                socket: victim_socket,
                core: victim as u16,
            },
            memory_numa_node: self.fixed_location.0.unwrap_or_default(),
            memory_slice: self.get_slice(addr),
            memory_vpn: vpn,
            memory_offset: (addr as isize) - (vpn as isize),
        };
        PartialLocationOwned::new(self.threshold_granularity, location)
    }

    // Updating the thread sched_affinity is up to the caller !
    pub fn set_cores(&mut self, main: usize, helper: usize) -> Result<(), TopologyAwareError> {
        self.set_location(self.fixed_location.0, Some(main), Some(helper))
    }
    // Updating the thread sched_affinity is up to the caller !
    pub fn set_location(
        &mut self,
        node: Option<NumaNode>,
        main: Option<usize>,
        helper: Option<usize>,
    ) -> Result<(), TopologyAwareError> {
        let old_location = self.fixed_location;
        self.fixed_location = (node, main, helper);
        if (self.fixed_location.0 != old_location.0 && self.threshold_granularity.memory_numa_node)
            || (self.fixed_location.1 != old_location.1 && self.threshold_granularity.attacker.core)
            || (self.fixed_location.2 != old_location.2 && self.threshold_granularity.victim.core)
            || (self.fixed_location.1.map(numa_node_of_cpu) != old_location.1.map(numa_node_of_cpu)
                && self.threshold_granularity.attacker.socket)
            || (self.fixed_location.2.map(numa_node_of_cpu) != old_location.2.map(numa_node_of_cpu)
                && self.threshold_granularity.victim.socket)
        {
            match self.recalibrate() {
                Ok(()) => Ok(()),
                Err(e) => {
                    self.fixed_location = old_location;
                    Err(e)
                }
            }
        } else {
            Ok(())
        }
    }

    fn build_location_vector(&self) -> Vec<(NumaNode, usize, usize)> {
        let numa_nodes = if let Some(numa_node) = self.fixed_location.0 {
            vec![numa_node]
        } else {
            numa_utils::available_nodes().unwrap().into_iter().collect()
        };

        let attackers = if let Some(attacker) = self.fixed_location.1 {
            vec![attacker]
        } else {
            (0..CpuSet::count()).collect()
        };

        let victims = if let Some(victim) = self.fixed_location.2 {
            vec![victim]
        } else {
            (0..CpuSet::count()).collect()
        };

        let mut locations = vec![];
        for node in numa_nodes {
            for attacker in &attackers {
                for victim in &victims {
                    locations.push((node, *attacker, *victim));
                }
            }
        }
        locations
    }

    fn recalibrate(&mut self) -> Result<(), TopologyAwareError> {
        // unset readiness status.
        // Call calibration with core pairs with a single core pair
        // Use results \o/ (or error out)

        //self.addresses.clear();
        let locations = self.build_location_vector();

        let m;
        let pages = if self.is_memory_target_sensitive() {
            self.thresholds
                .iter()
                .map(|(k, _v)| unsafe {
                    let vpn = k.get_vpn().unwrap() as *const u8;
                    let ret = &*slice_from_raw_parts(vpn, PAGE_LEN);
                    ret
                })
                .collect()
        } else {
            m = MMappedMemory::new(PAGE_LEN, false, false, |i| i as u8);
            let array: &[u8] = m.slice();
            let mut hashset = HashSet::new();
            hashset.insert(array);
            hashset
        };

        let res = Self::calibration_for_locations(
            &self.t,
            locations.into_iter(),
            pages.into_iter(),
            &self.calibration_granularity,
            &self.threshold_granularity,
            &self.norm_threshold,
            &self.error_prediction_builder,
            self.calibration_iterations,
        );

        let hashmap = match res {
            Err(_e) => {
                return Err(TopologyAwareError::Oops);
            }
            Ok(r) => r,
        };
        let hashmap = map_values(hashmap, |(e, n, _v), _k| (e, n));
        self.thresholds.extend(hashmap);
        self.calibration_epoch += 1;
        Ok(())
    }

    unsafe fn test_one_impl(
        &self,
        handle: &mut TopologyAwareTimingChannelHandle<WIDTH, N, E::E>,
        reset: bool,
    ) -> Result<CacheStatus, SideChannelError> {
        if handle.calibration_epoch != self.calibration_epoch {
            return Err(SideChannelError::NeedRecalibration);
        }
        let time = unsafe { self.t.attack(handle.addr) };
        if reset {
            unsafe { self.t.reset(handle.addr) };
        }
        if let Ok(bucket) = SimpleBucketU64::try_from(time) {
            if handle.threshold.is_hit(bucket) {
                Ok(CacheStatus::Hit)
            } else {
                Ok(CacheStatus::Miss)
            }
        } else {
            Ok(CacheStatus::Miss) //Err(SideChannelError::Retry)
        }
    }

    unsafe fn test_impl(
        &self,
        addresses: &mut Vec<&mut TopologyAwareTimingChannelHandle<WIDTH, N, E::E>>,
        limit: u32,
        reset: bool,
    ) -> Result<Vec<(*const u8, CacheStatus)>, SideChannelError> {
        let mut result = Vec::new();
        let mut tmp = Vec::new();
        let mut i = 0;
        for addr in addresses {
            let r = unsafe { self.test_one_impl(addr, false) };
            tmp.push((addr.to_const_u8_pointer(), r));
            i += 1;
            if i == limit {
                break;
            }
        }
        for (addr, r) in tmp {
            match r {
                Ok(status) => {
                    result.push((addr, status));
                }
                Err(e) => {
                    return Err(e);
                }
            }
            if reset {
                unsafe { self.t.reset(addr) };
            }
        }
        Ok(result)
    }

    unsafe fn prepare_one_impl(
        &self,
        handle: &mut TopologyAwareTimingChannelHandle<WIDTH, N, E::E>,
    ) -> Result<(), SideChannelError> {
        if handle.calibration_epoch != self.calibration_epoch {
            return Err(SideChannelError::NeedRecalibration);
        }
        unsafe { flush(handle.addr) }; // FIXME if the set up is not a flush.
        handle.ready = true;
        Ok(())
    }

    unsafe fn prepare_impl(
        &mut self,
        addresses: &mut Vec<&mut TopologyAwareTimingChannelHandle<WIDTH, N, E::E>>,
        limit: u32,
    ) -> Result<(), SideChannelError> {
        // Iterate on addresse prparig them, error early exit
        let mut i = 0;
        for handle in addresses {
            match unsafe { self.prepare_one_impl(handle) } {
                Ok(_) => {}
                Err(e) => {
                    return Err(e);
                }
            }
            i += 1;
            if i == limit {
                break;
            }
        }
        Ok(())
    }
}

impl<
    const WIDTH: u64,
    const N: usize,
    T: TimingChannelPrimitives,
    E: ErrorPredictionsBuilder<WIDTH, N>,
    Norm: Ord + Debug + Clone,
    NFThres: Fn(&Vec<ErrorPrediction>) -> Norm,
    NFLoc: Fn(&Vec<(Norm, Vec<ErrorPrediction>)>) -> Norm,
    const COVERT_CHANNEL_RESET: bool,
> Debug for TopologyAwareTimingChannel<WIDTH, N, T, E, Norm, NFThres, NFLoc, COVERT_CHANNEL_RESET>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Topology Aware Channel")
            .field("thresholds", &self.thresholds)
            //.field("addresses", &self.addresses)
            .field("slicing", &self.slicing)
            .field("location", &self.fixed_location)
            .field("preferred_addresses", &self.preferred_address)
            .field("calibration_epoch", &self.calibration_epoch)
            .field("primitive", &self.t)
            .field("calibration_iterations", &self.calibration_iterations)
            .field("calibration_granularity", &self.calibration_granularity)
            .field("threshold_granularity", &self.threshold_granularity)
            .finish()
        // todo, check field list.
    }
}

impl<
    const WIDTH: u64,
    const N: usize,
    T: TimingChannelPrimitives,
    E: ErrorPredictionsBuilder<WIDTH, N>,
    Norm: Ord + Debug + Clone,
    NFThres: Fn(&Vec<ErrorPrediction>) -> Norm,
    NFLoc: Fn(&Vec<(Norm, Vec<ErrorPrediction>)>) -> Norm,
    const COVERT_CHANNEL_RESET: bool,
> LocationSpec
    for TopologyAwareTimingChannel<WIDTH, N, T, E, Norm, NFThres, NFLoc, COVERT_CHANNEL_RESET>
{
    fn main_core(&self) -> CpuSet {
        let mut main = CpuSet::new();
        main.set(self.fixed_location.1.unwrap()).unwrap();
        main
    }

    fn helper_core(&self) -> CpuSet {
        let mut helper = CpuSet::new();
        helper.set(self.fixed_location.2.unwrap()).unwrap();
        helper
    }

    fn numa_nodes(&self) -> HashSet<NumaNode> {
        let mut r = HashSet::new();
        r.insert(self.fixed_location.0.unwrap());
        r
    }
}

impl<
    const WIDTH: u64,
    const N: usize,
    T: TimingChannelPrimitives,
    E: ErrorPredictionsBuilder<WIDTH, N>,
    Norm: Ord + Debug + Clone,
    NFThres: Fn(&Vec<ErrorPrediction>) -> Norm,
    NFLoc: Fn(&Vec<(Norm, Vec<ErrorPrediction>)>) -> Norm,
    const COVERT_CHANNEL_RESET: bool,
> MultipleAddrCacheSideChannel
    for TopologyAwareTimingChannel<WIDTH, N, T, E, Norm, NFThres, NFLoc, COVERT_CHANNEL_RESET>
{
    type Handle = TopologyAwareTimingChannelHandle<WIDTH, N, E::E>;
    const MAX_ADDR: u32 = 0;

    unsafe fn test<'a>(
        &mut self,
        addresses: &mut Vec<&'a mut Self::Handle>,
        reset: bool,
    ) -> Result<Vec<(*const u8, CacheStatus)>, SideChannelError>
    where
        Self::Handle: 'a,
    {
        unsafe { self.test_impl(addresses, Self::MAX_ADDR, reset) }
    }

    unsafe fn prepare<'a>(
        &mut self,
        addresses: &mut Vec<&'a mut Self::Handle>,
    ) -> Result<(), SideChannelError>
    where
        Self::Handle: 'a,
    {
        unsafe { self.prepare_impl(addresses, Self::MAX_ADDR) }
    }

    fn victim(&mut self, operation: &dyn Fn()) {
        operation(); // TODO use a different helper core ?
    }

    // this function tolerates multiple handle on the same cache line
    // should the invariant be fixed to one handle per line & calibration epoch ?
    unsafe fn calibrate(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<Vec<Self::Handle>, ChannelFatalError> {
        let locations = self.build_location_vector();

        let m;
        let pages = if self.is_memory_target_sensitive() {
            addresses
                .clone()
                .into_iter()
                .filter(|addr| {
                    let partial_location = self.build_partial_location(*addr);
                    self.thresholds.get(&partial_location).is_none()
                })
                .map(|addr: *const u8| unsafe {
                    let p = get_vpn(addr) as *const u8;
                    let ret = &*slice_from_raw_parts(p, PAGE_LEN);
                    ret
                })
                .collect::<HashSet<&[u8]>>()
        } else {
            let partial_location = self.build_partial_location(null());
            if self.thresholds.get(&partial_location).is_none() {
                m = MMappedMemory::new(PAGE_LEN, false, false, |i| i as u8);
                let array: &[u8] = m.slice();
                let mut hashset = HashSet::new();
                hashset.insert(array);
                hashset
            } else {
                // We already have thresholds, we only need to build handles.
                HashSet::new()
            }
        };

        if !pages.is_empty() {
            let result = Self::calibration_for_locations(
                &self.t,
                locations.into_iter(),
                pages.into_iter(),
                &self.calibration_granularity,
                &self.threshold_granularity,
                &self.norm_threshold,
                &self.error_prediction_builder,
                self.calibration_iterations,
            );
            let hashmap = match result {
                Err(_e) => {
                    return Err(ChannelFatalError::Oops);
                }
                Ok(r) => r,
            };
            let hashmap = map_values(hashmap, |(e, n, _v), _k| (e, n));
            for (k, v) in hashmap {
                self.thresholds.insert(k, v);
            }
            if self.thresholds.iter().count() == 1 {
                eprintln!("Single Threshold: {:?}", self.thresholds);
            }
        }

        // extract all the thresholds.
        let mut result = vec![];
        for addr in addresses {
            //let vpn = get_vpn(addr);
            //let slice = self.slicing.hash(addr as usize);
            let partial_location = self.build_partial_location(addr);
            let classifier = self.thresholds[&partial_location].0.clone();
            let handle = TopologyAwareTimingChannelHandle {
                threshold: classifier,
                addr,
                ready: false,
                calibration_epoch: self.calibration_epoch,
            };
            result.push(handle);
        }

        Ok(result)
    }
}

impl<
    const WIDTH: u64,
    const N: usize,
    T: TimingChannelPrimitives,
    E: ErrorPredictionsBuilder<WIDTH, N>,
    Norm: Ord + Debug + Clone,
    NFThres: Fn(&Vec<ErrorPrediction>) -> Norm,
    NFLoc: Fn(&Vec<(Norm, Vec<ErrorPrediction>)>) -> Norm,
    const COVERT_CHANNEL_RESET: bool,
> SingleAddrCacheSideChannel for TopologyAwareTimingChannel<WIDTH, N, T, E, Norm, NFThres, NFLoc, COVERT_CHANNEL_RESET>
{
    type Handle = TopologyAwareTimingChannelHandle<WIDTH, N, E::E>;

    unsafe fn test_single(
        &mut self,
        handle: &mut Self::Handle,
        reset: bool,
    ) -> Result<CacheStatus, SideChannelError> {
        unsafe { self.test_one_impl(handle, reset) }
    }

    unsafe fn prepare_single(&mut self, handle: &mut Self::Handle) -> Result<(), SideChannelError> {
        unsafe { self.prepare_one_impl(handle) }
    }

    fn victim_single(&mut self, operation: &dyn Fn()) {
        self.victim(operation)
    }

    unsafe fn calibrate_single(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<Vec<Self::Handle>, ChannelFatalError> {
        unsafe { self.calibrate(addresses) }
    }
}

impl<
    'a,
    const WIDTH: u64,
    const N: usize,
    T: TimingChannelPrimitives,
    E: ErrorPredictionsBuilder<WIDTH, N>,
    Norm: Ord + Send + Sync + Debug + Clone,
    NFThres: Send + Sync + Fn(&Vec<ErrorPrediction>) -> Norm,
    NFLoc: Send + Sync + Fn(&Vec<(Norm, Vec<ErrorPrediction>)>) -> Norm,
    const COVERT_CHANNEL_RESET: bool,
> CovertChannel for TopologyAwareTimingChannel<WIDTH, N, T, E, Norm, NFThres, NFLoc, COVERT_CHANNEL_RESET>
{
    type CovertChannelHandle =
        CovertChannelHandle<TopologyAwareTimingChannel<WIDTH, N, T, E, Norm, NFThres, NFLoc, COVERT_CHANNEL_RESET>>;
    const BIT_PER_PAGE: usize = 1;

    unsafe fn transmit<'b>(
        &self,
        handle: &mut Self::CovertChannelHandle,
        bits: &mut BitIterator<'b>,
    ) {
        let page = handle.0.addr;

        if let Some(b) = bits.next() {
            if b {
                unsafe { only_reload(page) };
            } else if !COVERT_CHANNEL_RESET {
                unsafe { only_flush(page) };
            }
        }
    }

    unsafe fn receive(&self, handle: &mut Self::CovertChannelHandle) -> Vec<bool> {
        let r = unsafe { self.test_one_impl(&mut handle.0, COVERT_CHANNEL_RESET) }; // transmit does the reload / flush as needed.
        match r {
            Err(e) => panic!("{:?}", e),
            Ok(status) => {
                let received = status == CacheStatus::Hit;
                return vec![received];
            }
        }
    }

    unsafe fn ready_page(&mut self, page: *const u8) -> Result<Self::CovertChannelHandle, ()> {
        let vpn: VPN = get_vpn(page);
        // Check if the page has already been readied. If so should error out ?
        if self.preferred_address.get(&vpn).is_some() {
            return Err(());
        }
        let mut location_params = self.threshold_granularity;
        location_params.memory_offset = false;
        location_params.memory_slice = false;

        let vpn = get_vpn(page);
        let attacker = self.fixed_location.1.unwrap_or_default();
        let victim = self.fixed_location.2.unwrap_or_default();
        let attacker_socket = numa_node_of_cpu(attacker).unwrap().into();
        let victim_socket = numa_node_of_cpu(victim).unwrap().into();
        let location = AVMLocation {
            attacker: CoreLocation {
                socket: attacker_socket,
                core: attacker as u16,
            },
            victim: CoreLocation {
                socket: victim_socket,
                core: victim as u16,
            },
            memory_numa_node: self.fixed_location.0.unwrap_or_default(),
            memory_slice: Default::default(),
            memory_vpn: vpn,
            memory_offset: Default::default(),
        };

        let partial_location = PartialLocationOwned::new(location_params, location);

        if self
            .thresholds
            .iter()
            .find(|(k, v)| k.project(&location_params) == partial_location)
            .is_none()
        {
            let pages = vec![page];
            match unsafe { self.calibrate(pages.into_iter()) } {
                Ok(r) => {}
                Err(e) => {
                    return Err(());
                }
            }
        }

        let threshold = self
            .thresholds
            .iter()
            .filter(|(k, _v)| k.project(&location_params) == partial_location)
            .min_by_key(|(k, v)| v.1.clone()); // TODO This is where a change is needed to avoid picking the best address.

        if threshold.is_none() {
            return Err(());
        }
        let (location, (classifier, _norm)) = threshold.unwrap();

        let addr = if let Some(offset) = location.get_offset() {
            unsafe { page.offset(offset) }
        } else {
            let mut candidates = vec![];
            if let Some(slice) = location.get_slice() {
                for i in 0..PAGE_LEN {
                    let addr = unsafe { page.offset(i as isize) };
                    if self.get_slice(addr) == slice {
                        candidates.push(addr);
                    }
                }
            } else {
                for i in 0..PAGE_LEN {
                    let addr = unsafe { page.offset(i as isize) };
                    candidates.push(addr);
                }
            }
            let addr = candidates.choose(&mut rand::rng()).unwrap();
            *addr
        };
        self.preferred_address.insert(vpn, addr);

        let mut handle = Self::CovertChannelHandle {
            0: TopologyAwareTimingChannelHandle {
                threshold: classifier.clone(),
                addr,
                ready: false,
                calibration_epoch: self.calibration_epoch,
            },
        };
        unsafe { self.prepare_one_impl(&mut handle.0) }.unwrap();

        Ok(handle)
    }

    unsafe fn unready_page(&mut self, handle: Self::CovertChannelHandle) -> Result<(), ()> {
        let vpn = get_vpn(handle.0.addr);
        if let Some(addr) = self.preferred_address.get(&vpn)
            && *addr == handle.0.addr
        {
            self.preferred_address.remove(&vpn);
            Ok(())
        } else {
            Err(())
        }
    }
}

impl<
    const WIDTH: u64,
    const N: usize,
    T: TimingChannelPrimitives,
    E: ErrorPredictionsBuilder<WIDTH, N>,
    Norm: Ord + Debug + Clone,
    NFThres: Fn(&Vec<ErrorPrediction>) -> Norm,
    NFLoc: Fn(&Vec<(Norm, Vec<ErrorPrediction>)>) -> Norm,
    const COVERT_CHANNEL_RESET: bool,
> TableCacheSideChannel<TopologyAwareTimingChannelHandle<WIDTH, N, E::E>>
    for TopologyAwareTimingChannel<WIDTH, N, T, E, Norm, NFThres, NFLoc, COVERT_CHANNEL_RESET>
{
    unsafe fn tcalibrate(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<Vec<TopologyAwareTimingChannelHandle<WIDTH, N, E::E>>, ChannelFatalError> {
        unsafe { self.tcalibrate_multi(addresses) }
    }

    unsafe fn attack<'a, 'b, 'c, 'd>(
        &'a mut self,
        addresses: &'b mut Vec<&'c mut TopologyAwareTimingChannelHandle<WIDTH, N, E::E>>,
        victim: &'d dyn Fn(),
        num_iteration: u32,
    ) -> Result<Vec<TableAttackResult>, ChannelFatalError>
    where
        TopologyAwareTimingChannelHandle<WIDTH, N, E::E>: 'c,
    {
        unsafe { self.attack_multi(addresses, victim, num_iteration) }
    }
}

// Extra helper for single address per page variants.
#[derive(Debug)]
pub struct SingleChannel<T: SingleAddrCacheSideChannel> {
    inner: T,
}

impl<T: SingleAddrCacheSideChannel> SingleChannel<T> {
    pub fn new(inner: T) -> Self {
        Self { inner }
    }
}

impl<T: SingleAddrCacheSideChannel> LocationSpec for SingleChannel<T> {
    fn main_core(&self) -> CpuSet {
        self.inner.main_core()
    }

    fn helper_core(&self) -> CpuSet {
        self.inner.helper_core()
    }

    fn numa_nodes(&self) -> HashSet<NumaNode> {
        self.inner.numa_nodes()
    }
}

impl<T: SingleAddrCacheSideChannel> SingleAddrCacheSideChannel for SingleChannel<T> {
    type Handle = T::Handle;

    unsafe fn test_single(
        &mut self,
        handle: &mut Self::Handle,
        reset: bool,
    ) -> Result<CacheStatus, SideChannelError> {
        unsafe { self.inner.test_single(handle, reset) }
    }

    unsafe fn prepare_single(&mut self, handle: &mut Self::Handle) -> Result<(), SideChannelError> {
        unsafe { self.inner.prepare_single(handle) }
    }

    fn victim_single(&mut self, operation: &dyn Fn()) {
        self.inner.victim_single(operation)
    }

    unsafe fn calibrate_single(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<Vec<Self::Handle>, ChannelFatalError> {
        unsafe { self.inner.calibrate_single(addresses) }
    }
}

impl<T: SingleAddrCacheSideChannel>
    TableCacheSideChannel<<SingleChannel<T> as SingleAddrCacheSideChannel>::Handle>
    for SingleChannel<T>
{
    unsafe fn tcalibrate(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<Vec<<SingleChannel<T> as SingleAddrCacheSideChannel>::Handle>, ChannelFatalError>
    {
        unsafe { self.inner.tcalibrate_single(addresses) }
    }

    unsafe fn attack<'a, 'b, 'c, 'd>(
        &'a mut self,
        addresses: &'b mut Vec<&'c mut <SingleChannel<T> as SingleAddrCacheSideChannel>::Handle>,
        victim: &'d dyn Fn(),
        num_iteration: u32,
    ) -> Result<Vec<TableAttackResult>, ChannelFatalError>
    where
        <SingleChannel<T> as SingleAddrCacheSideChannel>::Handle: 'c,
    {
        unsafe { self.inner.attack_single(addresses, victim, num_iteration) }
    }
}

/*
impl<T: MultipleAddrCacheSideChannel + Sync + Send> CovertChannel for SingleChannel<T> {
    type Handle = CovertChannelHandle<T>;
    const BIT_PER_PAGE: usize = 1;

    unsafe fn transmit<'a>(&self, handle: &mut Self::Handle, bits: &mut BitIterator<'a>) {
        unimplemented!()
    }

    unsafe fn receive(&self, handle: &mut Self::Handle) -> Vec<bool> {
        let r = unsafe { self.test_single(handle) };
        match r {
            Err(e) => panic!("{:?}", e),
            Ok(status_vec) => {
                assert_eq!(status_vec.len(), 1);
                let received = status_vec[0].1 == Hit;
                //println!("Received {} on page {:p}", received, page);
                return vec![received];
            }
        }
    }

    unsafe fn ready_page(&mut self, page: *const u8) -> Self::Handle {
        unimplemented!()
    }
}
*/

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
