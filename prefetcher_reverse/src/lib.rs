use crate::Probe::{Flush, Load};
use basic_timing_cache_channel::{TopologyAwareError, TopologyAwareTimingChannel};
use cache_side_channel::CacheStatus::Hit;
use cache_side_channel::{
    set_affinity, CacheStatus, CoreSpec, MultipleAddrCacheSideChannel, SingleAddrCacheSideChannel,
};
use cache_utils::calibration::PAGE_LEN;
use cache_utils::mmap::MMappedMemory;
use flush_flush::{FFHandle, FFPrimitives, FlushAndFlush};
use flush_reload::{FRHandle, FRPrimitives, FlushAndReload};
use nix::sys::stat::stat;
use rand::seq::SliceRandom;
use std::iter::{Cycle, Peekable};
use std::ops::Range;

// NB these may need to be changed / dynamically measured.
pub const CACHE_LINE_LEN: usize = 64;
pub const PAGE_CACHELINE_LEN: usize = PAGE_LEN / CACHE_LINE_LEN;

pub struct Prober {
    pages: Vec<MMappedMemory<u8>>,
    ff_handles: Vec<Vec<FFHandle>>,
    fr_handles: Vec<Vec<FRHandle>>,
    page_indexes: Peekable<Cycle<Range<usize>>>,
    ff_channel: FlushAndFlush,
    fr_channel: FlushAndReload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Probe {
    Load(usize),
    Flush(usize),
    FullFlush,
}

pub struct ProbePattern {
    pub pattern: Vec<usize>,
    pub probe: Probe,
}

enum ProberError {
    NoMem(nix::Error),
    TopologyError(TopologyAwareError),
    Nix(nix::Error),
}

/**
Result of running a probe pattern num_iteration times,
*/

pub enum ProbeResult {
    Load(u32),
    Flush(u32),
    FullFlush(Vec<u32>),
}

pub struct ProbePatternResult {
    pub num_iteration: u32,
    pub pattern_result: Vec<u32>,
    pub probe_result: ProbeResult,
}

struct DPRItem {
    pattern_result: Vec<u32>,
    probe_result: u32,
}

struct DualProbeResult {
    probe_offset: usize,
    load: DPRItem,
    flush: DPRItem,
}

pub struct FullPageDualProbeResults {
    num_iteration: u32,
    results: Vec<DualProbeResult>,
}

struct SingleProbeResult {
    probe_offset: usize,
    pattern_result: Vec<u32>,
    probe_result: u32,
}

pub struct FullPageSingleProbeResult {
    probe_type: Probe,
    num_iteration: u32,
    results: Vec<SingleProbeResult>,
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
    fn new(num_pages: usize) -> Result<Prober, ProberError> {
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
        let mut fr_channel = match FlushAndReload::new(core, core) {
            Ok(res) => res,
            Err(err) => {
                return Err(ProberError::TopologyError(err));
            }
        };

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
            let fr_page_handles = unsafe { fr_channel.calibrate(page_addresses) }.unwrap();

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
}
