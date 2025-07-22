#![deny(unsafe_op_in_unsafe_fn)]
use turn_lock::TurnHandle;

const PAGE_SIZE: usize = 1 << 12; // FIXME Magic

// design docs

// Build a channel using x pages + one synchronisation primitive.

// F+R only use one line per page
// F+F should use several line per page
// Each page has 1<<12 bytes / 1<<6 bytes per line, hence 64 lines (or 6 bits of info).

// General structure : two threads, a transmitter and a reciever. Transmitter generates bytes, Reciever reads bytes, then on join compare results for accuracy.
// Also time in order to determine duration, in rdtsc and seconds.

use bit_field::BitField;
pub use cache_side_channel::CovertChannel;
use cache_side_channel::{restore_affinity, set_affinity, BitIterator};
use cache_utils::mmap::MMappedMemory;
use cache_utils::rdtsc_fence;
use num_rational::Rational64;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, AddAssign};
use std::sync::Arc;
use std::thread;
/*  TODO : replace page with a handle type,
    require exclusive handle access,
    Handle protected by the turn lock
*/
/**
 * Safety considerations : Not ensure thread safety, need proper locking as needed.
 */

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct ChannelError {
    pub true_zero: usize,
    pub true_one: usize,
    pub false_one: usize,
    pub false_zero: usize,
}

impl ChannelError {
    pub fn bit_transmitted(&self) -> usize {
        self.true_one + self.false_one + self.true_zero + self.false_zero
    }

    pub fn one_received(&self) -> usize {
        self.true_one + self.false_one
    }

    pub fn zero_received(&self) -> usize {
        self.true_zero + self.false_zero
    }

    pub fn one_transmitted(&self) -> usize {
        self.true_one + self.false_zero
    }

    pub fn zero_transmitted(&self) -> usize {
        self.true_zero + self.false_one
    }

    pub fn bit_error(&self) -> usize {
        self.false_zero + self.false_one
    }

    pub fn error_rate(&self) -> f64 {
        self.bit_error() as f64 / self.bit_transmitted() as f64
    }

    pub fn error_ratio(&self) -> Rational64 {
        Rational64::new(self.bit_error() as i64, self.bit_transmitted() as i64)
    }
}

impl Add for &ChannelError {
    type Output = ChannelError;

    fn add(self, rhs: Self) -> Self::Output {
        ChannelError {
            true_zero: self.true_zero + rhs.true_zero,
            true_one: self.true_one + rhs.true_one,
            false_one: self.false_one + rhs.false_one,
            false_zero: self.false_zero + rhs.false_zero,
        }
    }
}

impl AddAssign<&Self> for ChannelError {
    fn add_assign(&mut self, rhs: &Self) {
        self.true_zero += rhs.true_zero;
        self.true_one += rhs.true_one;
        self.false_zero += rhs.false_zero;
        self.false_one += rhs.false_one;
    }
}

impl Add for ChannelError {
    type Output = ChannelError;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<&Self> for ChannelError {
    type Output = ChannelError;

    fn add(mut self, rhs: &Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<ChannelError> for &ChannelError {
    type Output = ChannelError;

    fn add(self, mut rhs: ChannelError) -> Self::Output {
        rhs += self;
        rhs
    }
}

impl AddAssign<Self> for ChannelError {
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

impl Sum<Self> for ChannelError {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(ChannelError::default(), |a, b| a + b)
    }
}

impl<'a> Sum<&'a Self> for ChannelError {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(ChannelError::default(), |a, b| a + b)
    }
}

#[derive(Debug)]
pub struct CovertChannelBenchmarkResult {
    pub num_bytes_transmitted: usize,
    pub error: ChannelError,
    pub time_rdtsc: u64,
    pub time_seconds: std::time::Duration,
}

impl CovertChannelBenchmarkResult {
    pub fn capacity(&self) -> f64 {
        (self.num_bytes_transmitted * 8) as f64 / self.time_seconds.as_secs_f64()
    }

    pub fn true_capacity(&self) -> f64 {
        let p = self.error.error_rate();
        if p == 0.0 || p == 0.0 {
            self.capacity()
        } else {
            self.capacity() * (1.0 + ((1.0 - p) * f64::log2(1.0 - p) + p * f64::log2(p)))
        }
    }

    pub fn csv(&self) -> String {
        format!(
            "{},{},{},{},{}",
            self.num_bytes_transmitted,
            self.error.bit_error(),
            self.error.error_rate(),
            self.time_rdtsc,
            self.time_seconds.as_nanos()
        )
    }

    pub fn csv_header() -> String {
        format!("bytes_transmitted,bits_error,error_rate,time_rdtsc,time_nanosec")
    }
}

struct CovertChannelParams<T: CovertChannel + Send> {
    handles: Vec<TurnHandle<T::CovertChannelHandle>>,
    covert_channel: Arc<T>,
}

unsafe impl<T: 'static + CovertChannel + Send> Send for CovertChannelParams<T> {}

fn transmit_thread<T: CovertChannel>(
    num_bytes: usize,
    mut params: CovertChannelParams<T>,
) -> (u64, std::time::Instant, Vec<u8>) {
    let _old_affinity = set_affinity(&(*params.covert_channel).helper_core());

    let mut result = Vec::new();
    result.reserve(num_bytes);
    for _ in 0..num_bytes {
        let byte = rand::random();
        result.push(byte);
    }

    let mut _bit_sent = 0;
    let mut bit_iter = BitIterator::new(&result);
    let start_time = std::time::Instant::now();
    let start = unsafe { rdtsc_fence() };
    while !bit_iter.at_end() {
        for page in params.handles.iter_mut() {
            let mut handle = page.wait();
            unsafe { params.covert_channel.transmit(&mut *handle, &mut bit_iter) };
            _bit_sent += T::BIT_PER_PAGE;
            page.next();
            if bit_iter.at_end() {
                break;
            }
        }
    }
    (start, start_time, result)
}

pub fn benchmark_channel<T: 'static + Send + CovertChannel>(
    mut channel: T,
    num_pages: usize,
    num_bytes: usize,
) -> CovertChannelBenchmarkResult {
    // Allocate pages

    let old_affinity = set_affinity(&channel.main_core()).unwrap();
    let nodes = channel.numa_nodes();
    numa_utils::set_memory_nodes(nodes).unwrap();

    // TODO, restore old numa nodes afterwards.

    let size = num_pages * PAGE_SIZE;
    let m = MMappedMemory::new(size, false, false, |i| (i / PAGE_SIZE) as u8);
    let mut receiver_turn_handles = Vec::new();
    let mut transmit_turn_handles = Vec::new();

    let array: &[u8] = m.slice();
    for i in 0..num_pages {
        let addr = &array[i * PAGE_SIZE] as *const u8;
        let handle = unsafe { channel.ready_page(addr) }.unwrap();
        let mut turns = TurnHandle::new(2, handle);
        let mut t_iter = turns.drain(0..);
        let transmit_lock = t_iter.next().unwrap();
        let receive_lock = t_iter.next().unwrap();

        assert!(t_iter.next().is_none());

        transmit_turn_handles.push(transmit_lock);
        receiver_turn_handles.push(receive_lock);
    }

    let covert_channel_arc = Arc::new(channel);
    let params = CovertChannelParams {
        handles: transmit_turn_handles,
        covert_channel: covert_channel_arc.clone(),
    };

    let helper = thread::spawn(move || transmit_thread(num_bytes, params));
    // Create the thread parameters
    let mut received_bytes: Vec<u8> = Vec::new();
    let mut received_bits = VecDeque::<bool>::new();
    while received_bytes.len() < num_bytes {
        for handle in receiver_turn_handles.iter_mut() {
            let mut page = handle.wait();
            let bits = unsafe { covert_channel_arc.receive(&mut *page) };
            handle.next();
            received_bits.extend(&mut bits.iter());
            while received_bits.len() >= u8::BIT_LENGTH {
                let mut byte = 0;
                for _i in 0..u8::BIT_LENGTH {
                    byte <<= 1;
                    let bit = received_bits.pop_front().unwrap();
                    byte |= bit as u8;
                }
                received_bytes.push(byte);
            }
            if received_bytes.len() >= num_bytes {
                break;
            }
        }
        // TODO
        // receiver thread
    }

    let stop = unsafe { rdtsc_fence() };
    let stop_time = std::time::Instant::now();
    let r = helper.join();
    let (start, start_time, sent_bytes) = match r {
        Ok(r) => r,
        Err(e) => panic!("Join Error: {:#?}", e),
    };
    assert_eq!(sent_bytes.len(), received_bytes.len());
    assert_eq!(num_bytes, received_bytes.len());

    restore_affinity(&old_affinity);

    let mut num_bit_error = 0;
    for i in 0..num_bytes {
        num_bit_error += (sent_bytes[i] ^ received_bytes[i]).count_ones() as usize;
    }

    let error_rate = (num_bit_error as f64) / ((num_bytes * u8::BIT_LENGTH) as f64);

    CovertChannelBenchmarkResult {
        num_bytes_transmitted: num_bytes,
        num_bit_errors: num_bit_error,
        error_rate,
        time_rdtsc: stop - start,
        time_seconds: stop_time - start_time,
    }
}

#[cfg(test)]
mod tests {
    use crate::BitIterator;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn test_bit_vec() {
        let bits = vec![0x55, 0xf];
        let bit_iter = BitIterator::new(&bits);
        let results = vec![
            false, true, false, true, false, true, false, true, false, false, false, false, true,
            true, true, true,
        ];
        for (i, bit) in bit_iter.enumerate() {
            assert_eq!(results[i], bit);
        }
    }
}
