#![deny(unsafe_op_in_unsafe_fn)]
use crate::Probe::{Flush, FullFlush, Load};
use basic_timing_cache_channel::{TopologyAwareError, TopologyAwareTimingChannel};
use cache_side_channel::CacheStatus::Hit;
use cache_side_channel::{
    set_affinity, CacheStatus, CoreSpec, MultipleAddrCacheSideChannel, SingleAddrCacheSideChannel,
};
use cache_utils::calibration::{Threshold, PAGE_LEN};
use cache_utils::mmap::MMappedMemory;
use flush_flush::{FFHandle, FFPrimitives, FlushAndFlush};
use flush_reload::naive::{NFRHandle, NaiveFlushAndReload};
use flush_reload::{FRHandle, FRPrimitives, FlushAndReload};
use nix::sys::stat::stat;
use rand::seq::SliceRandom;
use std::fmt::{Display, Error, Formatter};
use std::iter::{Cycle, Peekable};
use std::ops::Range;

// NB these may need to be changed / dynamically measured.
pub const CACHE_LINE_LEN: usize = 64;
pub const PAGE_CACHELINE_LEN: usize = PAGE_LEN / CACHE_LINE_LEN;

pub struct Prober {
    pages: Vec<MMappedMemory<u8>>,
    ff_handles: Vec<Vec<FFHandle>>,
    fr_handles: Vec<Vec<NFRHandle>>,
    page_indexes: Peekable<Cycle<Range<usize>>>,
    ff_channel: FlushAndFlush,
    fr_channel: NaiveFlushAndReload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Probe {
    Load(usize),
    Flush(usize),
    FullFlush,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeType {
    Load,
    Flush,
    FullFlush,
}

#[derive(Debug)]
pub struct ProbePattern {
    pub pattern: Vec<usize>,
    pub probe: Probe,
}

#[derive(Debug)]
pub enum ProberError {
    NoMem(nix::Error),
    TopologyError(TopologyAwareError),
    Nix(nix::Error),
}

/**
Result of running a probe pattern num_iteration times,
*/
pub type SinglePR = u32;
pub type FullPR = Vec<u32>;

#[derive(Debug)]
pub enum ProbeResult {
    Load(SinglePR),
    Flush(SinglePR),
    FullFlush(FullPR),
}

#[derive(Debug)]
pub struct ProbePatternResult {
    pub num_iteration: u32,
    pub pattern_result: Vec<u32>,
    pub probe_result: ProbeResult,
}

#[derive(Debug)]
pub struct DPRItem<PR> {
    pub pattern_result: Vec<u32>,
    pub probe_result: PR,
}

#[derive(Debug)]
pub struct DualProbeResult {
    pub probe_offset: usize,
    pub load: DPRItem<SinglePR>,
    pub flush: DPRItem<SinglePR>,
}

#[derive(Debug)]
pub struct FullPageDualProbeResults {
    pub pattern: Vec<usize>,
    pub num_iteration: u32,
    pub single_probe_results: Vec<DualProbeResult>,
    pub full_flush_results: DPRItem<FullPR>,
}

#[derive(Debug)]
pub struct SingleProbeResult {
    pub probe_offset: usize,
    pub pattern_result: Vec<u32>,
    pub probe_result: u32,
}

#[derive(Debug)]
pub struct FullPageSingleProbeResult {
    pub pattern: Vec<usize>,
    pub probe_type: ProbeType,
    pub num_iteration: u32,
    pub results: Vec<SingleProbeResult>,
}

// Helper function
/**
This function is a helper that determine what is the maximum stride for a pattern of len accesses
starting at a given offset, both forward and backward.

Special case for length 0.
 */
fn max_stride(offset: usize, len: usize) -> (isize, isize) {
    if len == 0 {
        (0, 0)
    } else {
        let min = -((offset / (len * CACHE_LINE_LEN)) as isize);
        let max = ((PAGE_LEN - offset) / (len * CACHE_LINE_LEN)) as isize;
        (min, max)
    }
}

impl Prober {
    pub fn new(num_pages: usize) -> Result<Prober, ProberError> {
        let mut vec = Vec::new();
        let mut handles = Vec::new();
        let (mut ff_channel, cpuset, core) = match FlushAndFlush::new_any_single_core() {
            Ok(res) => res,
            Err(err) => {
                return Err(ProberError::TopologyError(err));
            }
        };
        let old_affinity = match set_affinity(&ff_channel.main_core()) {
            Ok(old) => old,
            Err(nixerr) => return Err(ProberError::Nix(nixerr)),
        };
        let mut fr_channel = NaiveFlushAndReload::new(Threshold {
            bucket_index: 250,
            miss_faster_than_hit: false,
        });
        /*
        let mut fr_channel = match FlushAndReload::new(core, core) {
            Ok(res) => res,
            Err(err) => {
                return Err(ProberError::TopologyError(err));
            }
        };*/

        for i in 0..num_pages {
            let mut p = match MMappedMemory::<u8>::try_new(PAGE_LEN, false) {
                Ok(p) => p,
                Err(e) => {
                    return Err(ProberError::NoMem(e));
                }
            };
            for j in 0..PAGE_LEN {
                p[j] = (i * PAGE_CACHELINE_LEN + j) as u8;
            }
            let page_addresses =
                ((0..PAGE_LEN).step_by(CACHE_LINE_LEN)).map(|offset| &p[offset] as *const u8);
            let ff_page_handles = unsafe { ff_channel.calibrate(page_addresses.clone()) }.unwrap();
            let fr_page_handles = unsafe { fr_channel.calibrate_single(page_addresses) }.unwrap();

            vec.push(p);
            handles.push((ff_page_handles, fr_page_handles));
        }

        let mut page_indexes = (0..(handles.len())).cycle().peekable();

        handles.shuffle(&mut rand::thread_rng());

        let mut ff_handles = Vec::new();
        let mut fr_handles = Vec::new();

        for (ff_handle, fr_handle) in handles {
            ff_handles.push(ff_handle);
            fr_handles.push(fr_handle);
        }

        Ok(Prober {
            pages: vec,
            ff_handles,
            fr_handles,
            page_indexes,
            ff_channel,
            fr_channel,
        })
    }

    /*
        fn probe(&mut self, probe_type: Probe, offset: usize) -> CacheStatus {
            let page_index = self.page_indexes.peek().unwrap();
            match probe_type {
                Probe::Load => {
                    let h = &mut self.handles[*page_index][offset].fr;
                    unsafe { self.fr_channel.test_single(h, false) }.unwrap()
                }
                Probe::Flush => {
                    let h = &mut self.handles[*page_index][offset].ff;
                    unsafe { self.ff_channel.test_single(h, false) }.unwrap()
                }
            }
        }
    */

    fn probe_pattern_once(
        &mut self,
        pattern: &ProbePattern,
        result: Option<&mut ProbePatternResult>,
    ) {
        enum ProbeOutput {
            Single(CacheStatus),
            Full(Vec<(*const u8, CacheStatus)>),
        }

        self.page_indexes.next();
        let page_index = *self.page_indexes.peek().unwrap();

        let mut ff_handles = self.ff_handles[page_index].iter_mut().collect();

        unsafe { self.ff_channel.prepare(&mut ff_handles) };

        let mut pattern_res = vec![CacheStatus::Miss; pattern.pattern.len()];
        for (i, offset) in pattern.pattern.iter().enumerate() {
            let h = &mut self.fr_handles[page_index][*offset];
            pattern_res[i] = unsafe { self.fr_channel.test_single(h, false) }.unwrap()
        }

        let mut probe_out = match pattern.probe {
            Load(offset) => {
                let h = &mut self.fr_handles[page_index][offset];
                ProbeOutput::Single(unsafe { self.fr_channel.test_single(h, false) }.unwrap())
            }
            Flush(offset) => {
                let h = &mut self.ff_handles[page_index][offset];
                ProbeOutput::Single(unsafe { self.ff_channel.test_single(h, false) }.unwrap())
            }
            Probe::FullFlush => {
                ProbeOutput::Full(unsafe { self.ff_channel.test(&mut ff_handles, true).unwrap() })
            }
        };

        if let Some(result_ref) = result {
            result_ref.num_iteration += 1;

            match result_ref.probe_result {
                ProbeResult::Load(ref mut r) | ProbeResult::Flush(ref mut r) => {
                    if let ProbeOutput::Single(status) = probe_out {
                        if status == Hit {
                            *r += 1;
                        }
                    } else {
                        panic!()
                    }
                }
                ProbeResult::FullFlush(ref mut v) => {
                    if let ProbeOutput::Full(vstatus) = probe_out {
                        for (i, status) in vstatus.iter().enumerate() {
                            if status.1 == Hit {
                                v[i] += 1;
                            }
                        }
                    } else {
                        panic!()
                    }
                }
            }

            for (i, res) in pattern_res.into_iter().enumerate() {
                if res == Hit {
                    result_ref.pattern_result[i] += 1
                }
            }
        }
    }

    pub fn probe_pattern(
        &mut self,
        pattern: &ProbePattern,
        num_iteration: u32,
        warmup: u32,
    ) -> ProbePatternResult {
        let mut result = ProbePatternResult {
            num_iteration: 0,
            pattern_result: vec![0; pattern.pattern.len()],
            probe_result: match pattern.probe {
                Load(_) => ProbeResult::Load(0),
                Flush(_) => ProbeResult::Flush(0),
                Probe::FullFlush => ProbeResult::FullFlush(vec![0; PAGE_CACHELINE_LEN]),
            },
        };
        for _ in 0..warmup {
            self.probe_pattern_once(pattern, None);
        }

        for _ in 0..num_iteration {
            self.probe_pattern_once(pattern, Some(&mut result));
        }

        result
    }

    fn full_page_probe_helper(
        &mut self,
        pattern: &mut ProbePattern,
        probe_type: ProbeType,
        num_iteration: u32,
        warmup: u32,
    ) -> FullPageSingleProbeResult {
        let mut result = FullPageSingleProbeResult {
            pattern: pattern.pattern.clone(),
            probe_type,
            num_iteration,
            results: vec![],
        };
        for offset in 0..PAGE_CACHELINE_LEN {
            pattern.probe = match probe_type {
                ProbeType::Load => Probe::Load(offset),
                ProbeType::Flush => Probe::Flush(offset),
                ProbeType::FullFlush => FullFlush,
            };
            let r = self.probe_pattern(pattern, num_iteration, warmup);
            result.results.push(SingleProbeResult {
                probe_offset: offset,
                pattern_result: r.pattern_result,
                probe_result: match r.probe_result {
                    ProbeResult::Load(r) => r,
                    ProbeResult::Flush(r) => r,
                    ProbeResult::FullFlush(r) => r[offset],
                },
            });
        }
        result
    }

    pub fn full_page_probe(
        &mut self,
        pattern: Vec<usize>,
        num_iteration: u32,
        warmup: u32,
    ) -> FullPageDualProbeResults {
        let mut probe_pattern = ProbePattern {
            pattern: pattern,
            probe: Probe::FullFlush,
        };
        let res_flush = self.full_page_probe_helper(
            &mut probe_pattern,
            ProbeType::Flush,
            num_iteration,
            warmup,
        );
        let res_load =
            self.full_page_probe_helper(&mut probe_pattern, ProbeType::Load, num_iteration, warmup);
        probe_pattern.probe = FullFlush;
        let res_full_flush = self.probe_pattern(&probe_pattern, num_iteration, warmup);
        // TODO results

        FullPageDualProbeResults {
            pattern: probe_pattern.pattern,
            num_iteration,
            single_probe_results: res_flush
                .results
                .into_iter()
                .enumerate()
                .zip(res_load.results.into_iter())
                .map(|((offset, flush), load)| DualProbeResult {
                    probe_offset: offset,
                    load: DPRItem {
                        pattern_result: load.pattern_result,
                        probe_result: load.probe_result,
                    },
                    flush: DPRItem {
                        pattern_result: flush.pattern_result,
                        probe_result: flush.probe_result,
                    },
                })
                .collect(),
            full_flush_results: DPRItem {
                pattern_result: res_full_flush.pattern_result,
                probe_result: match res_full_flush.probe_result {
                    ProbeResult::FullFlush(r) => r,
                    _ => {
                        unreachable!()
                    }
                },
            },
        }
    }
}

impl Display for FullPageDualProbeResults {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut indices = vec![None; PAGE_CACHELINE_LEN];
        let pat_len = self.pattern.len();
        let divider = (PAGE_CACHELINE_LEN * self.num_iteration as usize) as f32;
        for (i, &offset) in self.pattern.iter().enumerate() {
            indices[offset] = Some(i);
        }
        // Display header
        let mut r = writeln!(f, "{:^3} {:^7} | {:^8} {:^8} {:^8} {:^8} | {:^8} {:^8} {:^8} {:^8} | {:^8} {:^8} {:^8} {:^8}",
                       "pat", "offset",
               "SF Ac H", "SF Ac HR", "SF Pr H", "SF Pr HR",
               "SR Ac H", "SR Ac HR", "SR Pr H", "SR Pr HR",
               "FF Ac H", "FF Ac HR", "FF Pr H", "FF Pr HR");
        match r {
            Ok(_) => {}
            Err(e) => {
                return Err(e);
            }
        }

        for i in 0..PAGE_CACHELINE_LEN {
            let index = indices[i];

            let (pat, sf_ac_h, sf_ac_hr, sr_ac_h, sr_ac_hr, ff_ac_h, ff_ac_hr) = match index {
                None => (
                    String::from(""),
                    String::from(""),
                    String::from(""),
                    String::from(""),
                    String::from(""),
                    String::from(""),
                    String::from(""),
                ),
                Some(index) => {
                    let pat = format!("{:3}", index);
                    let sf_ac: u32 = self
                        .single_probe_results
                        .iter()
                        .map(|d| d.flush.pattern_result[index])
                        .sum();
                    let sf_ac_h = format!("{:8}", sf_ac);
                    let sf_ac_hr = format!("{:8}", sf_ac as f32 / divider);

                    let sr_ac: u32 = self
                        .single_probe_results
                        .iter()
                        .map(|d| d.load.pattern_result[index])
                        .sum();
                    let sr_ac_h = format!("{:8}", sr_ac);
                    let sr_ac_hr = format!("{:8}", sr_ac as f32 / divider);

                    let ff_ac = self.full_flush_results.pattern_result[index];
                    let ff_ac_h = format!("{:8}", ff_ac);
                    let ff_ac_hr = format!("{:8}", ff_ac as f32 / self.num_iteration as f32);
                    (pat, sf_ac_h, sf_ac_hr, sr_ac_h, sr_ac_hr, ff_ac_h, ff_ac_hr)
                }
            };

            let sf_pr = self.single_probe_results[i].flush.probe_result;
            let sf_pr_h = format!("{:8}", sf_pr);
            let sf_pr_hr = format!("{:8}", sf_pr as f32 / self.num_iteration as f32);

            let sr_pr = self.single_probe_results[i].load.probe_result;
            let sr_pr_h = format!("{:8}", sr_pr);
            let sr_pr_hr = format!("{:8}", sr_pr as f32 / self.num_iteration as f32);

            let ff_pr = self.full_flush_results.probe_result[i];
            let ff_pr_h = format!("{:8}", ff_pr);
            let ff_pr_hr = format!("{:8}", ff_pr as f32 / self.num_iteration as f32);

            r = writeln!(f, "{:>3} {:>7} | {:>8} {:^8} {:>8} {:^8} | {:>8} {:^8} {:>8} {:^8} | {:>8} {:^8} {:>8} {:^8}",
                             pat, i,
                             sf_ac_h, sf_ac_hr, sf_pr_h, sf_pr_hr,
                             sr_ac_h, sr_ac_hr, sr_pr_h, sr_pr_hr,
                             ff_ac_h, ff_ac_hr, ff_pr_h, ff_pr_hr);
            match r {
                Ok(_) => {}
                Err(e) => {
                    return Err(e);
                }
            };
            // display lines
        }
        write!(f, "Num_iteration: {}", self.num_iteration)
    }
}
