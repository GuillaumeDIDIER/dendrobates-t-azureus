#![feature(unsafe_block_in_unsafe_fn)]
#![deny(unsafe_op_in_unsafe_fn)]
use basic_timing_cache_channel::TopologyAwareError;
use cache_side_channel::CacheStatus::Hit;
use cache_side_channel::{
    set_affinity, ChannelHandle, CoreSpec, MultipleAddrCacheSideChannel, SingleAddrCacheSideChannel,
};
use cache_utils::calibration::PAGE_LEN;
use cache_utils::maccess;
use cache_utils::mmap;
use cache_utils::mmap::MMappedMemory;
use flush_flush::{FFHandle, FFPrimitives, FlushAndFlush};
use nix::Error;
use rand::seq::SliceRandom;
use std::iter::Cycle;
use std::ops::Range;

pub const CACHE_LINE_LEN: usize = 64;

pub const PAGE_CACHELINE_LEN: usize = PAGE_LEN / CACHE_LINE_LEN;

pub const NUM_ITERATION: usize = 1 << 10;
pub const NUM_PAGES: usize = 256;

fn max_stride(offset: usize, len: usize) -> (isize, isize) {
    if len == 0 {
        (1, 1)
    } else {
        let min = -((offset / (len * CACHE_LINE_LEN)) as isize);
        let max = ((PAGE_LEN - offset) / (len * CACHE_LINE_LEN)) as isize;
        (min, max)
    }
}

// TODO negative stride
fn generate_pattern(offset: usize, len: usize, stride: isize) -> Option<Vec<usize>> {
    let end = (offset as isize + stride * len as isize) * CACHE_LINE_LEN as isize;
    if end < 0 || end > PAGE_LEN as isize {
        return None;
    }
    let mut res = Vec::with_capacity(len);
    let mut addr = offset as isize;
    for _ in 0..len {
        res.push(addr as usize);
        addr += stride;
    }
    Some(res)
}

fn execute_pattern(
    channel: &mut FlushAndFlush,
    page_handles: &mut Vec<&mut <FlushAndFlush as MultipleAddrCacheSideChannel>::Handle>,
    pattern: &Vec<usize>,
) -> Vec<bool> {
    for offset in pattern {
        let pointer = page_handles[*offset].to_const_u8_pointer();
        unsafe { maccess(pointer) };
    }

    let mut measures = unsafe { channel.test(page_handles) };

    let mut res = vec![false; PAGE_CACHELINE_LEN];

    for (i, status) in measures.unwrap().into_iter().enumerate() {
        res[i] = (status.1 == Hit)
    }
    res
}

fn execute_pattern_probe1(
    channel: &mut FlushAndFlush,
    page_handles: &mut Vec<&mut <FlushAndFlush as MultipleAddrCacheSideChannel>::Handle>,
    pattern: &Vec<usize>,
    probe_offset: usize,
) -> bool {
    for offset in pattern {
        let pointer = page_handles[*offset].to_const_u8_pointer();
        unsafe { maccess(pointer) };
    }

    let mut measure = unsafe { channel.test_single(&mut page_handles[probe_offset]) };

    measure.unwrap() == Hit
}

enum ProberError {
    NoMem(Error),
    TopologyError(TopologyAwareError),
    Nix(nix::Error),
}

struct Prober {
    pages: Vec<MMappedMemory<u8>>,
    handles: Vec<Vec<FFHandle>>,
    page_indexes: Cycle<Range<usize>>,
    channel: FlushAndFlush,
}

struct ProbeResult {
    probe_all_initial: [u32; PAGE_CACHELINE_LEN],
    probe_1: [u32; PAGE_CACHELINE_LEN],
    probe_all_final: [u32; PAGE_CACHELINE_LEN],
}

impl Prober {
    fn new(num_pages: usize) -> Result<Prober, ProberError> {
        let mut vec = Vec::new();
        let mut handles = Vec::new();
        let (mut channel, cpuset, core) = match FlushAndFlush::new_any_single_core(FFPrimitives {})
        {
            Ok(res) => res,
            Err(err) => {
                return Err(ProberError::TopologyError(err));
            }
        };
        let old_affinity = match set_affinity(&channel.main_core()) {
            Ok(old) => old,
            Err(nixerr) => return Err(ProberError::Nix(nixerr)),
        }; // FIXME error handling
        for i in 0..NUM_PAGES {
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
            let page_handles = unsafe { channel.calibrate(page_addresses) }.unwrap();
            vec.push(p);
            handles.push(page_handles);
        }

        let mut page_indexes = (0..(handles.len())).cycle();

        handles.shuffle(&mut rand::thread_rng());
        let mut handles_mutref = Vec::new();
        for page in handles.iter_mut() {
            handles_mutref.push(
                page.iter_mut()
                    .collect::<Vec<&mut <FlushAndFlush as MultipleAddrCacheSideChannel>::Handle>>(),
            );
        }

        Ok(Prober {
            pages: vec,
            handles,
            page_indexes,
            channel,
        })
    }

    fn probe_pattern(&mut self, pattern: Vec<usize>) -> ProbeResult {
        let mut handles_mutref = Vec::new();
        for page in self.handles.iter_mut() {
            handles_mutref.push(
                page.iter_mut()
                    .collect::<Vec<&mut <FlushAndFlush as MultipleAddrCacheSideChannel>::Handle>>(),
            );
        }
        let mut probe_all_result_first = [0; PAGE_CACHELINE_LEN];
        for _ in 0..NUM_ITERATION {
            let page_index = self.page_indexes.next().unwrap();
            unsafe { self.channel.prepare(&mut handles_mutref[page_index]) };
            let res = execute_pattern(&mut self.channel, &mut handles_mutref[page_index], &pattern);
            for j in 0..PAGE_CACHELINE_LEN {
                if res[j] {
                    probe_all_result_first[j] += 1;
                }
            }
        }
        let mut probe1_result = [0; PAGE_CACHELINE_LEN];
        for i in 0..PAGE_CACHELINE_LEN {
            for _ in 0..NUM_ITERATION {
                let page_index = self.page_indexes.next().unwrap();
                unsafe { self.channel.prepare(&mut handles_mutref[page_index]) };
                let res = execute_pattern_probe1(
                    &mut self.channel,
                    &mut handles_mutref[page_index],
                    &pattern,
                    i,
                );
                if res {
                    probe1_result[i] += 1;
                }
            }
        }
        let mut probe_all_result = [0; PAGE_CACHELINE_LEN];
        for _ in 0..NUM_ITERATION {
            let page_index = self.page_indexes.next().unwrap();
            unsafe { self.channel.prepare(&mut handles_mutref[page_index]) };
            let res = execute_pattern(&mut self.channel, &mut handles_mutref[page_index], &pattern);
            for j in 0..PAGE_CACHELINE_LEN {
                if res[j] {
                    probe_all_result[j] += 1;
                }
            }
        }

        ProbeResult {
            probe_all_initial: probe_all_result_first,
            probe_1: probe1_result,
            probe_all_final: probe_all_result,
        }
    }
}

fn main() {
    let mut vec = Vec::new();
    let mut handles = Vec::new();
    let (mut channel, cpuset, core) = FlushAndFlush::new_any_single_core(FFPrimitives {}).unwrap();
    let old_affinity = set_affinity(&channel.main_core());
    for i in 0..NUM_PAGES {
        let mut p = MMappedMemory::<u8>::new(PAGE_LEN, false);
        for j in 0..PAGE_LEN {
            p[j] = (i * PAGE_CACHELINE_LEN + j) as u8;
        }
        let page_addresses =
            ((0..PAGE_LEN).step_by(CACHE_LINE_LEN)).map(|offset| &p[offset] as *const u8);
        let page_handles = unsafe { channel.calibrate(page_addresses) }.unwrap();
        println!("{:p}", page_handles[0].to_const_u8_pointer());
        vec.push(p);
        handles.push(page_handles);
    }
    println!();

    let mut page_indexes = (0..(handles.len())).cycle();

    handles.shuffle(&mut rand::thread_rng());
    let mut handles_mutref = Vec::new();
    for page in handles.iter_mut() {
        handles_mutref.push(
            page.iter_mut()
                .collect::<Vec<&mut <FlushAndFlush as MultipleAddrCacheSideChannel>::Handle>>(),
        );
    }

    // Use an std::iter::Cycle iterator for pages.

    /*
    TODO List :
    Calibration & core selection (select one or two cores with optimal error)
    Then allocate a bunch of pages, and do accesses on each of them.

    (Let's start with stride patterns: for len in 0..16, and then for stride in 1..maxs_stride(len),
    generate a vec of addresses and get the victim to execute, then dump all the page)

    Sanity check on one pattern : do full dump, vs do dump per address.

    Both can be done using the FlushFlush channel

     */

    let pattern = generate_pattern(1, 4, 4).unwrap();
    println!("{:?}", pattern);
    let mut probe_all_result_first = [0; PAGE_CACHELINE_LEN];
    for _ in 0..NUM_ITERATION {
        let page_index = page_indexes.next().unwrap();
        unsafe { channel.prepare(&mut handles_mutref[page_index]) };
        let res = execute_pattern(&mut channel, &mut handles_mutref[page_index], &pattern);
        for j in 0..PAGE_CACHELINE_LEN {
            if res[j] {
                probe_all_result_first[j] += 1;
            }
        }
    }
    let mut probe1_result = [0; PAGE_CACHELINE_LEN];
    for i in 0..PAGE_CACHELINE_LEN {
        for _ in 0..NUM_ITERATION {
            let page_index = page_indexes.next().unwrap();
            unsafe { channel.prepare(&mut handles_mutref[page_index]) };
            let res =
                execute_pattern_probe1(&mut channel, &mut handles_mutref[page_index], &pattern, i);
            if res {
                probe1_result[i] += 1;
            }
        }
    }
    let mut probe_all_result = [0; PAGE_CACHELINE_LEN];
    for _ in 0..NUM_ITERATION {
        let page_index = page_indexes.next().unwrap();
        unsafe { channel.prepare(&mut handles_mutref[page_index]) };
        let res = execute_pattern(&mut channel, &mut handles_mutref[page_index], &pattern);
        for j in 0..PAGE_CACHELINE_LEN {
            if res[j] {
                probe_all_result[j] += 1;
            }
        }
    }

    for i in 0..PAGE_CACHELINE_LEN {
        println!(
            "{:2} {:4} {:4} {:4}",
            i, probe_all_result_first[i], probe1_result[i], probe_all_result[i]
        );
    }

    let pattern = generate_pattern(0, 3, 12).unwrap();
    println!("{:?}", pattern);
    let mut probe_all_result_first = [0; PAGE_CACHELINE_LEN];
    for _ in 0..NUM_ITERATION {
        let page_index = page_indexes.next().unwrap();
        unsafe { channel.prepare(&mut handles_mutref[page_index]) };
        let res = execute_pattern(&mut channel, &mut handles_mutref[page_index], &pattern);
        for j in 0..PAGE_CACHELINE_LEN {
            if res[j] {
                probe_all_result_first[j] += 1;
            }
        }
    }
    let mut probe1_result = [0; PAGE_CACHELINE_LEN];
    for i in 0..PAGE_CACHELINE_LEN {
        for _ in 0..NUM_ITERATION {
            let page_index = page_indexes.next().unwrap();
            unsafe { channel.prepare(&mut handles_mutref[page_index]) };
            let res =
                execute_pattern_probe1(&mut channel, &mut handles_mutref[page_index], &pattern, i);
            if res {
                probe1_result[i] += 1;
            }
        }
    }
    let mut probe_all_result = [0; PAGE_CACHELINE_LEN];
    for _ in 0..NUM_ITERATION {
        let page_index = page_indexes.next().unwrap();
        unsafe { channel.prepare(&mut handles_mutref[page_index]) };
        let res = execute_pattern(&mut channel, &mut handles_mutref[page_index], &pattern);
        for j in 0..PAGE_CACHELINE_LEN {
            if res[j] {
                probe_all_result[j] += 1;
            }
        }
    }

    for i in 0..PAGE_CACHELINE_LEN {
        println!(
            "{:2} {:4} {:4} {:4}",
            i, probe_all_result_first[i], probe1_result[i], probe_all_result[i]
        );
    }

    println!("Hello, world!");
    println!("{:?}", generate_pattern(0, 5, 1));
    println!("{:?}", generate_pattern(5, 0, 1));
    println!("{:?}", generate_pattern(1, 5, 5));
    println!("{:?}", generate_pattern(0, 16, 16));
}
