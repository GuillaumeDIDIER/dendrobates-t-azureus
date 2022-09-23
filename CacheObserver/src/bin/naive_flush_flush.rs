#![deny(unsafe_op_in_unsafe_fn)]

use basic_timing_cache_channel::naive::NaiveTimingChannelHandle;
use cache_side_channel::{
    CacheStatus, ChannelHandle, MultipleAddrCacheSideChannel, SingleAddrCacheSideChannel,
};
use cache_utils::calibration::{Threshold, PAGE_LEN};
use cache_utils::maccess;
use cache_utils::mmap::MMappedMemory;
use flush_flush::naive::NaiveFlushAndFlush;
use CacheObserver::CACHE_LINE_LEN;

const ITERATIONS: i32 = 128;
const THRESHOLD: usize = 175; // For Cyber Cobaye

fn dump_range(
    channel: &mut NaiveFlushAndFlush,
    handles: &mut Vec<&mut NaiveTimingChannelHandle>,
) -> Vec<CacheStatus> {
    let mut res = Vec::new();
    for handle in handles {
        let result = unsafe { channel.test_single(handle, true) }.unwrap();
        res.push(result)
    }
    res
}

fn run_experiment(
    channel: &mut NaiveFlushAndFlush,
    handles: &mut Vec<&mut NaiveTimingChannelHandle>,
    offsets: &Vec<usize>,
) -> Vec<i32> {
    let mut res = vec![0; handles.len()];
    let max_offset = handles.len();
    for offset in offsets {
        if *offset >= max_offset {
            panic!(
                "Illegal offset in experiment: {} (legal offsets < {})",
                offset, max_offset
            );
        }
    }

    for i in 0..ITERATIONS {
        for offset in offsets {
            let handle: &mut NaiveTimingChannelHandle = handles[*offset];
            let ptr = handle.to_const_u8_pointer();
            unsafe { maccess(ptr) };
            unsafe { core::arch::x86_64::_mm_mfence() };
        }
        let result = dump_range(channel, handles);
        assert_eq!(res.len(), result.len());
        for (i, status) in result.into_iter().enumerate() {
            if status == CacheStatus::Hit {
                res[i] += 1;
            }
        }
    }
    res
}

fn main() {
    let mut channel = NaiveFlushAndFlush::new(Threshold {
        bucket_index: THRESHOLD,
        miss_faster_than_hit: true,
    }); // Fixme grab threshold from known configs.
    let range = MMappedMemory::<u8>::new(PAGE_LEN, false, false, |i: usize| i as u8);

    let slice = range.slice();
    let mut addresses = Vec::new();
    for i in (0..(range.len())).step_by(CACHE_LINE_LEN) {
        addresses.push(&range[i] as *const u8);
    }

    let mut handles = unsafe { channel.calibrate(addresses) }.unwrap();
    let mut handle_refs = handles.iter_mut().collect();

    let empty_vec = vec![];
    let fetch_0_9 = (0..10).collect();

    let empty_pattern_result = run_experiment(&mut channel, &mut handle_refs, &empty_vec);

    let fetch_0_9_result = run_experiment(&mut channel, &mut handle_refs, &fetch_0_9);

    assert_eq!(handles.len(), empty_pattern_result.len());
    assert_eq!(handles.len(), fetch_0_9_result.len());

    println!("Offset,Addr,HR_NoAccess,HR_Access_0_9");
    for i in 0..handles.len() {
        println!(
            "{},{:p},{},{}",
            i,
            handles[i].to_const_u8_pointer(),
            empty_pattern_result[i],
            fetch_0_9_result[i]
        );
    }
}
