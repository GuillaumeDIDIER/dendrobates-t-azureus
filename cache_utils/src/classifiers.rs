/**
 * This module regroups various methods to classify hits and misses
 *
 * It relies on histograms and cumulative sums, with the following convention
 * a histogram is a Vector or array of u32 (counts), associated with a bucket (SimpleBucketU64 for now)
 * (Measurements which are u64 must be converted into buckets first)
 *
 * Histogram class $i$ contains value in range `[WIDTH * i, WIDTH * (i + 1) -1]`
 * CumSum omit the implicit (0,0) point, and so `CumSum[0] == Hist[0]`, they also have N classes (thus including the total sum)
 *
 * Threshold only have N-1 possible meaningful values, thus Threshold i, means classes 0 to i (inclusive) are below, and class i+1 - N are above.
 */

use crate::histograms::{Bucket, SimpleBucketU64, StaticHistogramCumSum};
use crate::calibration::{ErrorPrediction};
use alloc::vec::Vec;
use alloc::boxed::Box;
use alloc::vec;
use itertools::Itertools;

// Problem : Histogram need to be refactored to use buckets too.
pub trait HitClassifier<T: Bucket> {
    fn is_hit(&self, bucket: T) -> bool;
    fn is_time_hit(&self, time: u64) -> Result<bool, T::Error>;
    fn is_miss(&self, bucket: T) -> bool;
    fn is_time_miss(&self, time: u64) -> Result<bool, T::Error>;

    //fn error_prediction(&self, hits: HistogramCumSum, miss: HistogramCumSum) -> ErrorPrediction;
}


pub trait ErrorPredictor<const WIDTH: u64, const N: usize>: HitClassifier<SimpleBucketU64<WIDTH, N>> {
    fn error_prediction(&self, hits: &StaticHistogramCumSum<WIDTH, N>, miss: &StaticHistogramCumSum<WIDTH, N>) -> ErrorPrediction;
}

pub trait ErrorPredictionsBuilder<const WIDTH: u64, const N: usize> {
    type E: ErrorPredictor<WIDTH, N> + 'static;
    fn enumerate_error_predictions(&self, hits: &StaticHistogramCumSum<WIDTH, N>, miss: &StaticHistogramCumSum<WIDTH, N>) -> Vec<(Self::E, ErrorPrediction)>
    where
        Self: Sized,
    {
        self.enumerate_classifiers().into_iter().map(|e| {
            let pred = e.error_prediction(hits, miss);
            (e, pred)
        }).collect()
    }


    fn enumerate_classifiers(&self) -> Vec<Self::E>
    where
        Self: Sized;

    fn find_best_classifier(&self, hits: &StaticHistogramCumSum<WIDTH, N>, miss: &StaticHistogramCumSum<WIDTH, N>) -> Option<(Self::E, ErrorPrediction)>;
}

pub trait DynErrorPredictionsBuilder<const WIDTH: u64, const N: usize> {
    fn enumerate_error_predictions(&self, hits: &StaticHistogramCumSum<WIDTH, N>, miss: &StaticHistogramCumSum<WIDTH, N>) -> Vec<(Box<dyn ErrorPredictor<WIDTH, N>>, ErrorPrediction)>;
    fn enumerate_classifiers(&self) -> Vec<Box<dyn ErrorPredictor<WIDTH, N>>>;

    fn find_best_classifier(&self, hits: &StaticHistogramCumSum<WIDTH, N>, miss: &StaticHistogramCumSum<WIDTH, N>) -> Option<(Box<dyn ErrorPredictor<WIDTH, N>>, ErrorPrediction)>;
}

// Thresholds are less than equal.
// TODO check if there is a mismatch between the ErrorPrediction above and the decision below.
// usize for bucket, u64 for time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Threshold<T: Bucket> {
    pub bucket_index: T,
    pub miss_faster_than_hit: bool,
}

struct SimpleThresholdBuilder<const WIDTH: u64, const N: usize> ();

impl<T: Bucket> HitClassifier<T> for Threshold<T> {
    fn is_hit(&self, bucket: T) -> bool {
        if bucket <= self.bucket_index {
            !self.miss_faster_than_hit
        } else {
            self.miss_faster_than_hit
        }
    }

    fn is_time_hit(&self, time: u64) -> Result<bool, T::Error> {
        let bucket = T::try_from(time)?;
        Ok(self.is_hit(bucket))
    }

    fn is_miss(&self, bucket: T) -> bool {
        if bucket <= self.bucket_index {
            self.miss_faster_than_hit
        } else {
            !self.miss_faster_than_hit
        }
    }

    fn is_time_miss(&self, time: u64) -> Result<bool, T::Error> {
        let bucket = T::try_from(time)?;
        Ok(self.is_miss(bucket))
    }
}

impl<const WIDTH: u64, const N: usize> ErrorPredictor<WIDTH, N> for Threshold<SimpleBucketU64<WIDTH, N>> {
    fn error_prediction(&self, hits: &StaticHistogramCumSum<WIDTH, N>, miss: &StaticHistogramCumSum<WIDTH, N>) -> ErrorPrediction {
        if self.miss_faster_than_hit {
            ErrorPrediction {
                true_hit: hits[&SimpleBucketU64::<WIDTH, N>::MAX].cumulative_count - hits[&self.bucket_index].cumulative_count,
                true_miss: miss[&self.bucket_index].cumulative_count,
                false_hit: hits[&self.bucket_index].cumulative_count,
                false_miss: miss[&SimpleBucketU64::<WIDTH, N>::MAX].cumulative_count - miss[&self.bucket_index].cumulative_count,
            }
        } else {
            ErrorPrediction {
                true_hit: hits[&self.bucket_index].cumulative_count,
                true_miss: miss[&SimpleBucketU64::<WIDTH, N>::MAX].cumulative_count - miss[&self.bucket_index].cumulative_count,
                false_hit: hits[&SimpleBucketU64::<WIDTH, N>::MAX].cumulative_count - hits[&self.bucket_index].cumulative_count,
                false_miss: miss[&self.bucket_index].cumulative_count,
            }
        }
    }
}

impl<const WIDTH: u64, const N: usize> ErrorPredictionsBuilder<WIDTH, N> for SimpleThresholdBuilder<WIDTH, N> {
    type E = Threshold<SimpleBucketU64<WIDTH, N>>;

    /*fn enumerate_error_predictions(&self, hits: &StaticHistogramCumSum<WIDTH, N>, miss: &StaticHistogramCumSum<WIDTH, N>) -> Vec<(Self::E, ErrorPrediction)>
    where
        Self: Sized,
    {
        self.enumerate_classifiers().into_iter().map(|e| {
            let pred = e.error_prediction(hits, miss);
            (e, pred)
        }).collect()
    }*/

    fn enumerate_classifiers(&self) -> Vec<Self::E> {
        let mut res = Vec::with_capacity(2 * N);
        for i in SimpleBucketU64::<WIDTH, N>::MIN..=SimpleBucketU64::<WIDTH, N>::MAX {
            res.push(Threshold::<SimpleBucketU64<WIDTH, N>> { bucket_index: i, miss_faster_than_hit: true });
            res.push(Threshold::<SimpleBucketU64<WIDTH, N>> { bucket_index: i, miss_faster_than_hit: false });
        }
        res
    }

    fn find_best_classifier(&self, hits: &StaticHistogramCumSum<WIDTH, N>, miss: &StaticHistogramCumSum<WIDTH, N>) -> Option<(Self::E, ErrorPrediction)> {
        let all_classifiers = ErrorPredictionsBuilder::enumerate_error_predictions(self, hits, miss);
        let min_classifiers = all_classifiers.into_iter().min_set_by_key(|(_e, pred)| { pred.total_error() });
        let n = min_classifiers.len();
        if n == 0 {
            None
        } else {
            Some(min_classifiers[n - 1 / 2])
        }
    }
}

/*impl<const WIDTH: u64, const N: usize> DynErrorPredictionsBuilder<WIDTH, N> for SimpleThresholdBuilder<WIDTH, N> {
    fn enumerate_error_predictions(&self, hits: &StaticHistogramCumSum<WIDTH, N>, miss: &StaticHistogramCumSum<WIDTH, N>) -> Vec<(Box<dyn ErrorPredictor<WIDTH, N>>, ErrorPrediction)>
    {
        ErrorPredictionsBuilder::enumerate_error_predictions(&self, hits, miss).into_iter().map(|e, pred| { (Box::<dyn ErrorPredictor<WIDTH, N>>::new(e), pred) }).collect()
    }

    fn enumerate_classifiers(&self) -> Vec<Box<dyn ErrorPredictor<WIDTH, N>>> {
        ErrorPredictionsBuilder::enumerate_classifiers(&self).into_iter().map(|c| { Box::<dyn ErrorPredictor<WIDTH, N>>::new(c) }).collect()
    }
}*/

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DualThreshold<T: Bucket> {
    miss_to_hit_index: T,
    hit_to_miss_index: T,
    center_hit: bool,
}

pub struct DualThresholdBuilder<const WIDTH: u64, const N: usize> ();

impl<T: Bucket> DualThreshold<T> {
    pub fn new(miss_to_hit_index: T, hit_to_miss_index: T) -> Self {
        Self {
            miss_to_hit_index,
            hit_to_miss_index,
            center_hit: miss_to_hit_index < hit_to_miss_index,
        }
    }
    pub fn set_miss_to_hit(&mut self, miss_to_hit_index: &T) {
        self.miss_to_hit_index = *miss_to_hit_index;
        self.center_hit = self.miss_to_hit_index < self.hit_to_miss_index;
    }

    pub fn set_hit_to_miss(&mut self, hit_to_miss_index: &T) {
        self.hit_to_miss_index = *hit_to_miss_index;
        self.center_hit = self.miss_to_hit_index < self.hit_to_miss_index;
    }
}
impl<T: Bucket> HitClassifier<T> for DualThreshold<T> {
    fn is_hit(&self, bucket: T) -> bool {
        if self.center_hit {
            bucket > self.miss_to_hit_index && bucket <= self.hit_to_miss_index
        } else {
            bucket <= self.hit_to_miss_index || bucket > self.miss_to_hit_index
        }
    }

    fn is_time_hit(&self, time: u64) -> Result<bool, T::Error> {
        let bucket = T::try_from(time)?;
        Ok(self.is_hit(bucket))
    }

    fn is_miss(&self, bucket: T) -> bool {
        if self.center_hit {
            bucket <= self.miss_to_hit_index || bucket > self.hit_to_miss_index
        } else {
            bucket > self.hit_to_miss_index && bucket <= self.miss_to_hit_index
        }
    }

    fn is_time_miss(&self, time: u64) -> Result<bool, T::Error> {
        let bucket = T::try_from(time)?;
        Ok(self.is_miss(bucket))
    }
}

impl<const WIDTH: u64, const N: usize> ErrorPredictor<WIDTH, N> for DualThreshold<SimpleBucketU64<WIDTH, N>> {
    fn error_prediction(&self, hits: &StaticHistogramCumSum<WIDTH, N>, miss: &StaticHistogramCumSum<WIDTH, N>) -> ErrorPrediction {
        if self.center_hit {
            let lower_true_miss = miss[&self.miss_to_hit_index].cumulative_count;
            let lower_false_hits = hits[&self.miss_to_hit_index].cumulative_count;
            let center_false_miss = miss[&self.hit_to_miss_index].cumulative_count - lower_true_miss;
            let center_true_hits = hits[&self.hit_to_miss_index].cumulative_count - lower_false_hits;
            let total_true_miss = miss[&SimpleBucketU64::<WIDTH, N>::MAX].cumulative_count - center_false_miss;
            let total_false_hits = hits[&SimpleBucketU64::<WIDTH, N>::MAX].cumulative_count - center_true_hits;
            ErrorPrediction {
                true_hit: center_true_hits,
                true_miss: total_true_miss,
                false_hit: total_false_hits,
                false_miss: center_false_miss,
            }
        } else {
            let lower_true_hits = hits[&self.hit_to_miss_index].cumulative_count;
            let lower_false_miss = miss[&self.hit_to_miss_index].cumulative_count;
            let center_true_miss = miss[&self.miss_to_hit_index].cumulative_count - lower_false_miss;
            let center_false_hits = hits[&self.miss_to_hit_index].cumulative_count - lower_true_hits;
            let total_false_miss = miss[&SimpleBucketU64::<WIDTH, N>::MAX].cumulative_count - center_true_miss;
            let total_true_hits = hits[&SimpleBucketU64::<WIDTH, N>::MAX].cumulative_count - center_false_hits;
            ErrorPrediction {
                true_hit: total_true_hits,
                true_miss: center_true_miss,
                false_hit: center_false_hits,
                false_miss: total_false_miss,
            }
        }
    }
}


impl<const WIDTH: u64, const N: usize> ErrorPredictionsBuilder<WIDTH, N> for DualThresholdBuilder<WIDTH, N> {
    type E = DualThreshold<SimpleBucketU64<WIDTH, N>>;

    fn enumerate_classifiers(&self) -> Vec<Self::E> {
        let mut res = Vec::with_capacity(N * (N - 1));
        for i in SimpleBucketU64::<WIDTH, N>::MIN..=SimpleBucketU64::<WIDTH, N>::MAX {
            for j in SimpleBucketU64::<WIDTH, N>::MIN..=SimpleBucketU64::<WIDTH, N>::MAX {
                if i != j {
                    res.push(DualThreshold::<SimpleBucketU64<WIDTH, N>>::new(i, j));
                }
            }
        }
        res
    }

    fn find_best_classifier(&self, hits: &StaticHistogramCumSum<WIDTH, N>, miss: &StaticHistogramCumSum<WIDTH, N>) -> Option<(Self::E, ErrorPrediction)> {
        // 1. Find classifier minimizing the error
        let all_classifiers = ErrorPredictionsBuilder::enumerate_error_predictions(self, hits, miss);
        let min_classifiers = all_classifiers.into_iter().min_set_by_key(|(_e, pred)| { pred.total_error() });
        let n = min_classifiers.len();
        if n == 0 {
            return None;
        }
        // 2. Find the classifier minimizing the distance to the others ?
        let mut total_distance = vec![0usize; n];
        for i in 0..n {
            for j in 0..n {
                total_distance[i] += distance(&min_classifiers[i].0, &min_classifiers[j].0)
            }
        }
        let indexes = total_distance.into_iter().enumerate().min_set_by_key(|(_, k)| *k);
        assert!(indexes.len() > 0);
        let index = (indexes.len() - 1) / 2;

        Some(min_classifiers[index])
    }
}

fn distance<const WIDTH: u64, const N: usize>(a: &DualThreshold<SimpleBucketU64<WIDTH, N>>, b: &DualThreshold<SimpleBucketU64<WIDTH, N>>) -> usize {
    a.miss_to_hit_index.abs_diff(b.miss_to_hit_index) + a.hit_to_miss_index.abs_diff(b.hit_to_miss_index) + if a.center_hit ^ b.center_hit { N } else { 0 }
}

impl<const WIDTH: u64, const N: usize, T> DynErrorPredictionsBuilder<WIDTH, N> for T
where
    T: Sized + ErrorPredictionsBuilder<WIDTH, N>,
{
    fn enumerate_error_predictions(&self, hits: &StaticHistogramCumSum<WIDTH, N>, miss: &StaticHistogramCumSum<WIDTH, N>) -> Vec<(Box<dyn ErrorPredictor<WIDTH, N>>, ErrorPrediction)>
    {
        ErrorPredictionsBuilder::enumerate_error_predictions(self, hits, miss).into_iter().map(|(e, pred)| { (Box::new(e) as Box<dyn ErrorPredictor<WIDTH, N>>, pred) }).collect()
    }

    fn enumerate_classifiers(&self) -> Vec<Box<dyn ErrorPredictor<WIDTH, N>>> {
        ErrorPredictionsBuilder::enumerate_classifiers(self).into_iter().map(|c| { Box::new(c) as Box<dyn ErrorPredictor<WIDTH, N>> }).collect()
    }

    fn find_best_classifier(&self, hits: &StaticHistogramCumSum<WIDTH, N>, miss: &StaticHistogramCumSum<WIDTH, N>) -> Option<(Box<dyn ErrorPredictor<WIDTH, N>>, ErrorPrediction)> {
        if let Some((e, pred)) = ErrorPredictionsBuilder::find_best_classifier(self, hits, miss) {
            Some((Box::new(e), pred))
        } else {
            None
        }
    }
}