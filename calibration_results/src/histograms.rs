extern crate alloc;

use core::iter::Step;
use core::ops::{Index, IndexMut};
#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "serde_support")]
use serde_big_array::BigArray;
use std::fmt::Debug;
use std::iter::zip;
use std::ops::AddAssign;
/**********
 * Traits *
 **********/

pub trait Bucket: Into<u64> + TryFrom<u64> + Ord + Eq + Copy + Send + Sync + Debug {}

/***********
 * Structs *
 ***********/
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub struct NaiveBucketU64(u64);

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub struct SimpleBucketU64<const WIDTH: u64, const N: usize>(usize);

// Dynamic Bucket ? (Where the size is known only at runtime ?)
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct StaticHistogram<const WIDTH: u64, const N: usize> {
    #[cfg_attr(feature = "serde_support", serde(with = "BigArray"))]
    data: [u32; N],
}

#[derive(Debug, Clone, Copy)]
pub struct HistogramCumSumItem {
    pub count: u32,
    pub cumulative_count: u64,
}
#[derive(Debug, Clone)]
pub struct StaticHistogramCumSum<const WIDTH: u64, const N: usize> {
    data: [HistogramCumSumItem; N],
}

/***********
 *  Impls  *
 ***********/

/* Naive Bucket */
impl From<u64> for NaiveBucketU64 {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl Into<u64> for NaiveBucketU64 {
    fn into(self) -> u64 {
        self.0
    }
}

impl Into<usize> for NaiveBucketU64 {
    fn into(self) -> usize {
        self.0 as usize
    }
}

impl Bucket for NaiveBucketU64 {}

/* Simple Bucket*/

impl<const WIDTH: u64, const N: usize> SimpleBucketU64<WIDTH, N> {
    pub const MAX: SimpleBucketU64<WIDTH, N> = Self(N - 1);
    pub const MIN: SimpleBucketU64<WIDTH, N> = Self(0);

    pub const fn abs_diff(self, other: Self) -> usize {
        self.0.abs_diff(other.0)
    }
}

impl<const WIDTH: u64, const N: usize> TryFrom<u64> for SimpleBucketU64<WIDTH, N> {
    type Error = ();

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        let r = (value / WIDTH) as usize;
        if r >= N {
            Err(())
        } else {
            Ok(SimpleBucketU64::<WIDTH, N>(r))
        }
    }
}

impl<const WIDTH: u64, const N: usize> Into<u64> for SimpleBucketU64<WIDTH, N> {
    fn into(self) -> u64 {
        self.0 as u64 * WIDTH
    }
}

impl<const WIDTH: u64, const N: usize> Into<usize> for SimpleBucketU64<WIDTH, N> {
    fn into(self) -> usize {
        self.0
    }
}
impl<const WIDTH: u64, const N: usize> Bucket for SimpleBucketU64<WIDTH, N> {}

impl<const WIDTH: u64, const N: usize> Step for SimpleBucketU64<WIDTH, N> {
    fn steps_between(start: &Self, end: &Self) -> (usize, Option<usize>) {
        Step::steps_between(&start.0, &end.0)
    }

    fn forward_checked(start: Self, count: usize) -> Option<Self> {
        if let Some(n) = Step::forward_checked(start.0, count) {
            if n < N { Some(Self(n)) } else { None }
        } else {
            None
        }
    }

    fn backward_checked(start: Self, count: usize) -> Option<Self> {
        if let Some(n) = Step::backward_checked(start.0, count) {
            Some(Self(n))
        } else {
            None
        }
    }
}

/* Static Histogram */
impl<const WIDTH: u64, const N: usize> StaticHistogram<WIDTH, N> {
    pub fn empty() -> Self {
        Self { data: [0; N] }
    }

    pub fn get(&self, time: u64) -> Option<&u32> {
        let bucket = SimpleBucketU64::<WIDTH, N>::try_from(time);
        match bucket {
            Ok(index) => Some(&self.data[index.0]),
            Err(_) => None,
        }
    }

    pub fn get_mut(&mut self, time: u64) -> Option<&mut u32> {
        let bucket = SimpleBucketU64::<WIDTH, N>::try_from(time);
        match bucket {
            Ok(index) => Some(&mut self.data[index.0]),
            Err(_) => None,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (SimpleBucketU64<WIDTH, N>, u32)> {
        let buckets = SimpleBucketU64::<WIDTH, N>::MIN..=SimpleBucketU64::<WIDTH, N>::MAX;
        let values = self.data.iter().copied();
        zip(buckets, values)
    }
}

impl<const WIDTH: u64, const N: usize> Default for StaticHistogram<WIDTH, N> {
    fn default() -> Self {
        StaticHistogram::empty()
    }
}
impl<const WIDTH: u64, const N: usize> Index<&SimpleBucketU64<WIDTH, N>>
    for StaticHistogram<WIDTH, N>
{
    type Output = u32;

    fn index(&self, index: &SimpleBucketU64<WIDTH, N>) -> &Self::Output {
        &self.data[index.0]
    }
}

impl<const WIDTH: u64, const N: usize> IndexMut<&SimpleBucketU64<WIDTH, N>>
    for StaticHistogram<WIDTH, N>
{
    fn index_mut(&mut self, index: &SimpleBucketU64<WIDTH, N>) -> &mut Self::Output {
        &mut self.data[index.0]
    }
}

impl<const WIDTH: u64, const N: usize> Index<u64> for StaticHistogram<WIDTH, N> {
    type Output = u32;

    fn index(&self, index: u64) -> &Self::Output {
        let bucket = SimpleBucketU64::try_from(index).expect("Invalid time");
        &self[&bucket]
    }
}

impl<const WIDTH: u64, const N: usize> IndexMut<u64> for StaticHistogram<WIDTH, N> {
    fn index_mut(&mut self, index: u64) -> &mut Self::Output {
        let bucket = SimpleBucketU64::try_from(index).expect("Invalid time");
        &mut self[&bucket]
    }
}

impl<const WIDTH: u64, const N: usize> AddAssign<&Self> for StaticHistogram<WIDTH, N> {
    fn add_assign(&mut self, rhs: &Self) {
        for i in 0..N {
            self.data[i] += rhs.data[i];
        }
    }
}

/* Static Histogram Cum Sum*/
impl<const WIDTH: u64, const N: usize> StaticHistogramCumSum<WIDTH, N> {
    pub fn get(&self, time: u64) -> Option<&HistogramCumSumItem> {
        let bucket = SimpleBucketU64::<WIDTH, N>::try_from(time);
        match bucket {
            Ok(index) => Some(&self.data[index.0]),
            Err(_) => None,
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = (SimpleBucketU64<WIDTH, N>, HistogramCumSumItem)> {
        let buckets = SimpleBucketU64::<WIDTH, N>::MIN..=SimpleBucketU64::<WIDTH, N>::MAX;
        let values = self.data.iter().copied();
        zip(buckets, values)
    }
}

impl<const WIDTH: u64, const N: usize> From<StaticHistogram<WIDTH, N>>
    for StaticHistogramCumSum<WIDTH, N>
{
    fn from(value: StaticHistogram<WIDTH, N>) -> Self {
        let mut cumul = 0u64;
        let mut r = Self {
            data: [HistogramCumSumItem {
                count: 0,
                cumulative_count: 0,
            }; N],
        };
        for (i, &count) in value.data.iter().enumerate() {
            cumul += count as u64;
            r.data[i] = HistogramCumSumItem {
                count,
                cumulative_count: cumul,
            };
        }
        r
    }
}

impl<const WIDTH: u64, const N: usize> Index<&SimpleBucketU64<WIDTH, N>>
    for StaticHistogramCumSum<WIDTH, N>
{
    type Output = HistogramCumSumItem;

    fn index(&self, index: &SimpleBucketU64<WIDTH, N>) -> &Self::Output {
        &self.data[index.0]
    }
}

impl<const WIDTH: u64, const N: usize> Index<u64> for StaticHistogramCumSum<WIDTH, N> {
    type Output = HistogramCumSumItem;

    fn index(&self, index: u64) -> &Self::Output {
        let bucket = SimpleBucketU64::try_from(index).expect("Invalid time");
        &self[&bucket]
    }
}

pub fn group2_histogram<const WIDTH: u64, const N: usize>(
    input: StaticHistogram<WIDTH, { N + N }>,
) -> StaticHistogram<{ WIDTH + WIDTH }, N> {
    let mut res: StaticHistogram<{ WIDTH + WIDTH }, N> = StaticHistogram::empty();
    for i in 0..N {
        let value = input.data[2 * i] + input.data[2 * i + 1];
        res.data[i] = value;
    }
    res
}

pub fn group2_histogram_cum_sum<const WIDTH: u64, const N: usize>(
    input: StaticHistogramCumSum<WIDTH, { N + N }>,
) -> StaticHistogramCumSum<{ WIDTH + WIDTH }, N> {
    let mut res: StaticHistogram<{ WIDTH + WIDTH }, N> = StaticHistogram::empty();
    for i in 0..N {
        let count = input.data[2 * i].count + input.data[2 * i + 1].count;
        res.data[i] = count;
    }
    StaticHistogramCumSum::from(res)
}
