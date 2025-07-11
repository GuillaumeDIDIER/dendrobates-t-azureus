use crate::calibration::ErrorPrediction;
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
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::borrow::Borrow;
use core::cmp::Ordering;
use core::fmt::{Debug, Display, Formatter};
use itertools::Itertools;

// Problem : Histogram need to be refactored to use buckets too.
pub trait HitClassifier<T: Bucket>: Debug + Send + Sync {
    fn is_hit(&self, bucket: T) -> bool;
    fn is_time_hit(&self, time: u64) -> Result<bool, T::Error>;
    fn is_miss(&self, bucket: T) -> bool;
    fn is_time_miss(&self, time: u64) -> Result<bool, T::Error>;

    //fn error_prediction(&self, hits: HistogramCumSum, miss: HistogramCumSum) -> ErrorPrediction;
}

pub trait ErrorPredictor<const WIDTH: u64, const N: usize>:
    HitClassifier<SimpleBucketU64<WIDTH, N>>
{
    fn error_prediction(
        &self,
        hits: &StaticHistogramCumSum<WIDTH, N>,
        miss: &StaticHistogramCumSum<WIDTH, N>,
    ) -> ErrorPrediction;
}

pub trait ErrorPredictionsBuilder<const WIDTH: u64, const N: usize>: Send + Sync {
    type E: ErrorPredictor<WIDTH, N> + 'static + Clone;
    fn enumerate_error_predictions(
        &self,
        hits: &StaticHistogramCumSum<WIDTH, N>,
        miss: &StaticHistogramCumSum<WIDTH, N>,
    ) -> Vec<(Self::E, ErrorPrediction)>
    where
        Self: Sized,
    {
        self.enumerate_classifiers()
            .into_iter()
            .map(|e| {
                let pred = e.error_prediction(hits, miss);
                (e, pred)
            })
            .collect()
    }

    fn enumerate_classifiers(&self) -> Vec<Self::E>
    where
        Self: Sized;

    fn find_best_classifier(
        &self,
        hits: &StaticHistogramCumSum<WIDTH, N>,
        miss: &StaticHistogramCumSum<WIDTH, N>,
    ) -> Option<(Self::E, ErrorPrediction)>;

    // Out of equivalent classifiers from an error rate, pick one.
    fn select_best_classifier<T: Borrow<Self::E> + Copy, U: Copy>(
        &self,
        classifiers: Vec<(T, U)>,
    ) -> Option<(T, U)>;
}

pub trait DynErrorPredictionsBuilder<const WIDTH: u64, const N: usize> {
    fn enumerate_error_predictions(
        &self,
        hits: &StaticHistogramCumSum<WIDTH, N>,
        miss: &StaticHistogramCumSum<WIDTH, N>,
    ) -> Vec<(Box<dyn ErrorPredictor<WIDTH, N>>, ErrorPrediction)>;
    fn enumerate_classifiers(&self) -> Vec<Box<dyn ErrorPredictor<WIDTH, N>>>;

    fn find_best_classifier(
        &self,
        hits: &StaticHistogramCumSum<WIDTH, N>,
        miss: &StaticHistogramCumSum<WIDTH, N>,
    ) -> Option<(Box<dyn ErrorPredictor<WIDTH, N>>, ErrorPrediction)>;
}

// Thresholds are less than equal.
// TODO check if there is a mismatch between the ErrorPrediction above and the decision below.
// usize for bucket, u64 for time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Threshold<T: Bucket> {
    pub bucket_index: T,
    pub miss_faster_than_hit: bool,
}

#[derive(Debug, Clone)]
pub struct SimpleThresholdBuilder<const WIDTH: u64, const N: usize>();

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

impl<const WIDTH: u64, const N: usize> ErrorPredictor<WIDTH, N>
    for Threshold<SimpleBucketU64<WIDTH, N>>
{
    fn error_prediction(
        &self,
        hits: &StaticHistogramCumSum<WIDTH, N>,
        miss: &StaticHistogramCumSum<WIDTH, N>,
    ) -> ErrorPrediction {
        if self.miss_faster_than_hit {
            ErrorPrediction {
                true_hit: hits[&SimpleBucketU64::<WIDTH, N>::MAX].cumulative_count
                    - hits[&self.bucket_index].cumulative_count,
                true_miss: miss[&self.bucket_index].cumulative_count,
                false_hit: hits[&self.bucket_index].cumulative_count,
                false_miss: miss[&SimpleBucketU64::<WIDTH, N>::MAX].cumulative_count
                    - miss[&self.bucket_index].cumulative_count,
            }
        } else {
            ErrorPrediction {
                true_hit: hits[&self.bucket_index].cumulative_count,
                true_miss: miss[&SimpleBucketU64::<WIDTH, N>::MAX].cumulative_count
                    - miss[&self.bucket_index].cumulative_count,
                false_hit: hits[&SimpleBucketU64::<WIDTH, N>::MAX].cumulative_count
                    - hits[&self.bucket_index].cumulative_count,
                false_miss: miss[&self.bucket_index].cumulative_count,
            }
        }
    }
}

impl<const WIDTH: u64, const N: usize> PartialOrd for Threshold<SimpleBucketU64<WIDTH, N>> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const WIDTH: u64, const N: usize> Ord for Threshold<SimpleBucketU64<WIDTH, N>> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.miss_faster_than_hit.cmp(&other.miss_faster_than_hit) {
            Ordering::Less => Ordering::Less,
            Ordering::Equal => self.bucket_index.cmp(&other.bucket_index),
            Ordering::Greater => Ordering::Greater,
        }
    }
}

impl<const WIDTH: u64, const N: usize> ErrorPredictionsBuilder<WIDTH, N>
    for SimpleThresholdBuilder<WIDTH, N>
{
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
            res.push(Threshold::<SimpleBucketU64<WIDTH, N>> {
                bucket_index: i,
                miss_faster_than_hit: true,
            });
            res.push(Threshold::<SimpleBucketU64<WIDTH, N>> {
                bucket_index: i,
                miss_faster_than_hit: false,
            });
        }
        res
    }

    fn find_best_classifier(
        &self,
        hits: &StaticHistogramCumSum<WIDTH, N>,
        miss: &StaticHistogramCumSum<WIDTH, N>,
    ) -> Option<(Self::E, ErrorPrediction)> {
        fn helper<const WIDTH: u64, const N: usize>(
            res: &mut Vec<(Threshold<SimpleBucketU64<WIDTH, N>>, ErrorPrediction)>,
            error: &mut u64,
            threshold: Threshold<SimpleBucketU64<WIDTH, N>>,
            hits: &StaticHistogramCumSum<WIDTH, N>,
            miss: &StaticHistogramCumSum<WIDTH, N>,
        ) {
            let error_pred = threshold.error_prediction(hits, miss);
            match error_pred.total_error().cmp(&error) {
                Ordering::Less => {
                    res.clear();
                    *error = error_pred.total_error();
                    res.push((threshold, error_pred));
                }
                Ordering::Equal => {
                    res.push((threshold, error_pred));
                }
                Ordering::Greater => { /* Nothing to do*/ }
            }
        }
        /*let all_classifiers =
            ErrorPredictionsBuilder::enumerate_error_predictions(self, hits, miss);
        let min_classifiers = all_classifiers
            .into_iter()
            .min_set_by_key(|(_e, pred)| pred.total_error());
        let n = min_classifiers.len();
        if n == 0 {
            None
        } else {
            Some(min_classifiers[(n - 1) / 2])
        }*/
        //let mut res = Vec::with_capacity(2 * N);
        let mut res = Vec::new();
        let mut error = u64::MAX;
        for i in SimpleBucketU64::<WIDTH, N>::MIN..=SimpleBucketU64::<WIDTH, N>::MAX {
            let t1 = Threshold::<SimpleBucketU64<WIDTH, N>> {
                bucket_index: i,
                miss_faster_than_hit: true,
            };
            helper(&mut res, &mut error, t1, hits, miss);
            /*let e1 = t1.error_prediction(hits, miss);

            match e1.total_error().cmp(&error) {
                Ordering::Less => {
                    res.clear();
                    error = e1.total_error();
                    res.push((t1, e1));
                }
                Ordering::Equal => {
                    res.push((t1, e1));
                }
                Ordering::Greater => { /* Nothing to do*/ }
            }*/
            let t2 = Threshold::<SimpleBucketU64<WIDTH, N>> {
                bucket_index: i,
                miss_faster_than_hit: false,
            };
            helper(&mut res, &mut error, t2, hits, miss);
            /*
            let e2 = t2.error_prediction(hits, miss);
            match e2.total_error().cmp(&error) {
                Ordering::Less => {
                    res.clear();
                    error = e2.total_error();
                    res.push((t2, e2));
                }
                Ordering::Equal => {
                    res.push((t2, e2));
                }
                Ordering::Greater => { /* Nothing to do*/ }
            }*/
        }
        let n = res.len();
        if n == 0 { None } else { Some(res[(n - 1) / 2]) }
    }

    fn select_best_classifier<T: Borrow<Self::E> + Copy, U: Copy>(
        &self,
        mut classifiers: Vec<(T, U)>,
    ) -> Option<(T, U)> {
        let n = classifiers.len();
        if n == 0 {
            None
        } else {
            classifiers.sort_by(|a, b| (*a).0.borrow().cmp(b.0.borrow()));
            Some(classifiers[(n - 1) / 2])
        }
    }
}

impl<const WIDTH: u64, const N: usize> Display for Threshold<SimpleBucketU64<WIDTH, N>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let mut time: u64 = self.bucket_index.into();
        time += WIDTH;
        if self.miss_faster_than_hit {
            write!(f, "miss < {}", time)
        } else {
            write!(f, "hit < {}", time)
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

#[derive(Debug, Clone)]
pub struct DualThresholdBuilder<const WIDTH: u64, const N: usize>();

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

    pub fn get_thresholds(&self) -> (T, T) {
        (self.miss_to_hit_index, self.hit_to_miss_index)
    }
}

// Important note the the x_to_y index means index i includes x. (It's an <= threshold).
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

impl<const WIDTH: u64, const N: usize> ErrorPredictor<WIDTH, N>
    for DualThreshold<SimpleBucketU64<WIDTH, N>>
{
    fn error_prediction(
        &self,
        hits: &StaticHistogramCumSum<WIDTH, N>,
        miss: &StaticHistogramCumSum<WIDTH, N>,
    ) -> ErrorPrediction {
        if self.center_hit {
            let lower_true_miss = miss[&self.miss_to_hit_index].cumulative_count;
            let lower_false_hits = hits[&self.miss_to_hit_index].cumulative_count;
            let center_false_miss =
                miss[&self.hit_to_miss_index].cumulative_count - lower_true_miss;
            let center_true_hits =
                hits[&self.hit_to_miss_index].cumulative_count - lower_false_hits;
            let total_true_miss =
                miss[&SimpleBucketU64::<WIDTH, N>::MAX].cumulative_count - center_false_miss;
            let total_false_hits =
                hits[&SimpleBucketU64::<WIDTH, N>::MAX].cumulative_count - center_true_hits;
            ErrorPrediction {
                true_hit: center_true_hits,
                true_miss: total_true_miss,
                false_hit: total_false_hits,
                false_miss: center_false_miss,
            }
        } else {
            let lower_true_hits = hits[&self.hit_to_miss_index].cumulative_count;
            let lower_false_miss = miss[&self.hit_to_miss_index].cumulative_count;
            let center_true_miss =
                miss[&self.miss_to_hit_index].cumulative_count - lower_false_miss;
            let center_false_hits =
                hits[&self.miss_to_hit_index].cumulative_count - lower_true_hits;
            let total_false_miss =
                miss[&SimpleBucketU64::<WIDTH, N>::MAX].cumulative_count - center_true_miss;
            let total_true_hits =
                hits[&SimpleBucketU64::<WIDTH, N>::MAX].cumulative_count - center_false_hits;
            ErrorPrediction {
                true_hit: total_true_hits,
                true_miss: center_true_miss,
                false_hit: center_false_hits,
                false_miss: total_false_miss,
            }
        }
    }
}

impl<const WIDTH: u64, const N: usize> DualThresholdBuilder<WIDTH, N> {
    pub fn find_best_classifier_slow(
        &self,
        hits: &StaticHistogramCumSum<WIDTH, N>,
        miss: &StaticHistogramCumSum<WIDTH, N>,
    ) -> Option<(DualThreshold<SimpleBucketU64<WIDTH, N>>, ErrorPrediction)> {
        let mut center_hits_t1_candidates = Vec::new();
        let mut center_miss_t1_candidates = Vec::new();
        let mut center_hits_t1_error = u64::MAX;
        let mut center_miss_t1_error = u64::MAX;
        let mut candidates = Vec::new();
        //let mut center_hits_candidates = Vec::new();
        //let mut center_miss_candidates = Vec::new();
        let mut error = u64::MAX;
        //let mut center_hits_error = u64::MAX;
        //let mut center_miss_error = u64::MAX;
        let mut range = SimpleBucketU64::<WIDTH, N>::MIN..=SimpleBucketU64::<WIDTH, N>::MAX;
        let first = range.next().unwrap();

        let total_miss = miss[&SimpleBucketU64::<WIDTH, N>::MAX].cumulative_count;
        let total_hits = hits[&SimpleBucketU64::<WIDTH, N>::MAX].cumulative_count;

        // For the first bucket, with everything as miss.
        /*let center_hit_t1_error_prediction = ErrorPrediction {
            true_hit: 0,
            true_miss: miss[&first].cumulative_count,
            false_hit: hits[&first].cumulative_count,
            false_miss: 0,
        };*/
        center_hits_t1_error = hits[&first].cumulative_count; //center_hit_t1_error_prediction.total_error();
        center_hits_t1_candidates.push((first/*, center_hit_t1_error_prediction*/));

        center_miss_t1_error = miss[&first].cumulative_count; //center_hit_t1_error_prediction.total_error();
        center_miss_t1_candidates.push((first/*, center_hit_t1_error_prediction*/));

        /*let center_hits_error_prediction = center_hit_t1_error_prediction + ErrorPrediction{
            true_hit: 0,
            true_miss: total_miss - miss[&first].cumulative_count,
            false_hit: total_hits - hits[&first].cumulative_count,
            false_miss: 0,
        };*/

        match total_hits.cmp(&total_miss) {
            Ordering::Less => {
                error = total_hits;
                candidates.push((first, first, true));
            }
            Ordering::Equal => {
                error = total_hits;
                candidates.push((first, first, true));
                candidates.push((first, first, false));
            }
            Ordering::Greater => {
                error = total_miss;
                candidates.push((first, first, false));
            }
        }

        for i in range {
            let extra_hits = hits[&i].count as u64;
            let extra_miss = miss[&i].count as u64;
            let center_hits_t1_unchanged_error = center_hits_t1_error + extra_miss;
            let center_hits_t1_new_error = hits[&i].cumulative_count;

            let center_miss_t1_unchanged_error = center_miss_t1_error + extra_hits;
            let center_miss_t1_new_error = miss[&i].cumulative_count;

            match center_hits_t1_new_error.cmp(&center_hits_t1_unchanged_error) {
                Ordering::Less => {
                    center_hits_t1_candidates.clear();
                    center_hits_t1_candidates.push(i);
                    center_hits_t1_error = center_hits_t1_new_error;
                }
                Ordering::Equal => {
                    center_hits_t1_candidates.push(i);
                    center_hits_t1_error = center_hits_t1_new_error;
                }
                Ordering::Greater => {
                    center_hits_t1_error = center_hits_t1_unchanged_error;
                }
            }

            match center_miss_t1_new_error.cmp(&center_miss_t1_unchanged_error) {
                Ordering::Less => {
                    center_miss_t1_candidates.clear();
                    center_miss_t1_candidates.push(i);
                    center_miss_t1_error = center_miss_t1_new_error;
                }
                Ordering::Equal => {
                    center_miss_t1_candidates.push(i);
                    center_miss_t1_error = center_miss_t1_new_error;
                }
                Ordering::Greater => {
                    center_miss_t1_error = center_miss_t1_unchanged_error;
                }
            }

            let center_hits_error_new =
                center_hits_t1_error + total_hits - hits[&i].cumulative_count;
            match center_hits_error_new.cmp(&error) {
                Ordering::Less => {
                    candidates.clear();
                    for candidate_t1 in center_hits_t1_candidates.iter() {
                        candidates.push((*candidate_t1, i, true));
                    }
                    error = center_hits_error_new;
                }
                Ordering::Equal => {
                    for candidate_t1 in center_hits_t1_candidates.iter() {
                        candidates.push((*candidate_t1, i, true));
                    }
                }
                Ordering::Greater => { /* Nothing to do, the set of best candidates unchanged */ }
            }

            let center_miss_error_new =
                center_miss_t1_error + total_miss - miss[&i].cumulative_count;
            match center_miss_error_new.cmp(&error) {
                Ordering::Less => {
                    candidates.clear();
                    for candidate_t1 in center_miss_t1_candidates.iter() {
                        candidates.push((*candidate_t1, i, false));
                    }
                    error = center_miss_error_new;
                }
                Ordering::Equal => {
                    for candidate_t1 in center_miss_t1_candidates.iter() {
                        candidates.push((*candidate_t1, i, false));
                    }
                }
                Ordering::Greater => { /* Nothing to do, the set of best candidates unchanged */ }
            }
        }

        // NOTE, this is rather expensive. Can we do better ?

        // 1. Find classifier minimizing the error
        /*let all_classifiers =
            ErrorPredictionsBuilder::enumerate_error_predictions(self, hits, miss);
        let min_classifiers = all_classifiers
            .into_iter()
            .min_set_by_key(|(_e, pred)| pred.total_error());

         */

        let mut min_classifiers: Vec<(
            DualThreshold<SimpleBucketU64<{ WIDTH }, { N }>>,
            ErrorPrediction,
        )> = Vec::with_capacity(candidates.len());
        for (t1, t2, center_hit) in candidates {
            assert!(t1 <= t2);
            let dual_threshold = if center_hit {
                DualThreshold::new(t1, t2)
            } else {
                if t1 == t2 {
                    panic!("Tried to use all miss as a dual threshold classifier, unsupported");
                }
                DualThreshold::new(t2, t1)
            };
            let error_pred = dual_threshold.error_prediction(hits, miss);
            assert_eq!(error_pred.total_error(), error);
            min_classifiers.push((dual_threshold, error_pred))
        }
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
        let indexes = total_distance
            .into_iter()
            .enumerate()
            .min_set_by_key(|(_, k)| *k);
        assert!(indexes.len() > 0);
        let index = (indexes.len() - 1) / 2;

        Some(min_classifiers[index])
    }
}

impl<const WIDTH: u64, const N: usize> ErrorPredictionsBuilder<WIDTH, N>
    for DualThresholdBuilder<WIDTH, N>
{
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

    fn find_best_classifier(
        &self,
        hits: &StaticHistogramCumSum<WIDTH, N>,
        miss: &StaticHistogramCumSum<WIDTH, N>,
    ) -> Option<(Self::E, ErrorPrediction)> {
        let mut center_hits_t1_error = u64::MAX;
        let mut center_miss_t1_error = u64::MAX;

        //let mut center_hits_candidates = Vec::new();
        //let mut center_miss_candidates = Vec::new();
        let mut error = u64::MAX;
        //let mut center_hits_error = u64::MAX;
        //let mut center_miss_error = u64::MAX;
        let mut range = SimpleBucketU64::<WIDTH, N>::MIN..=SimpleBucketU64::<WIDTH, N>::MAX;
        let first = range.next().unwrap();

        let total_miss = miss[&SimpleBucketU64::<WIDTH, N>::MAX].cumulative_count;
        let total_hits = hits[&SimpleBucketU64::<WIDTH, N>::MAX].cumulative_count;

        // For the first bucket, with everything as miss.
        /*let center_hit_t1_error_prediction = ErrorPrediction {
            true_hit: 0,
            true_miss: miss[&first].cumulative_count,
            false_hit: hits[&first].cumulative_count,
            false_miss: 0,
        };*/
        center_hits_t1_error = hits[&first].cumulative_count; //center_hit_t1_error_prediction.total_error();

        center_miss_t1_error = miss[&first].cumulative_count; //center_hit_t1_error_prediction.total_error();

        let mut center_hits_t1_candidate = first;
        let mut center_miss_t1_candidate = first;

        let mut candidate = (first, first, true);

        let mut center_hits_t1_flip = false;
        let mut center_miss_t1_flip = false;
        let mut candidate_flip = false;

        /*let center_hits_error_prediction = center_hit_t1_error_prediction + ErrorPrediction{
            true_hit: 0,
            true_miss: total_miss - miss[&first].cumulative_count,
            false_hit: total_hits - hits[&first].cumulative_count,
            false_miss: 0,
        };*/

        match total_hits.cmp(&total_miss) {
            Ordering::Less => {
                error = total_hits;
                candidate = (first, first, true);
            }
            Ordering::Equal => {
                error = total_hits;
                candidate = (first, first, true);
            }
            Ordering::Greater => {
                error = total_miss;
                candidate = (first, first, false);
            }
        }

        for i in range {
            let extra_hits = hits[&i].count as u64;
            let extra_miss = miss[&i].count as u64;
            let center_hits_t1_unchanged_error = center_hits_t1_error + extra_miss;
            let center_hits_t1_new_error = hits[&i].cumulative_count;

            let center_miss_t1_unchanged_error = center_miss_t1_error + extra_hits;
            let center_miss_t1_new_error = miss[&i].cumulative_count;

            match center_hits_t1_new_error.cmp(&center_hits_t1_unchanged_error) {
                Ordering::Less => {
                    center_hits_t1_candidate = i;
                    center_hits_t1_error = center_hits_t1_new_error;
                }
                Ordering::Equal => {
                    if center_hits_t1_flip {
                        center_hits_t1_candidate = i;
                    }
                    center_hits_t1_flip = !center_hits_t1_flip;
                    center_hits_t1_error = center_hits_t1_new_error;
                }
                Ordering::Greater => {
                    center_hits_t1_error = center_hits_t1_unchanged_error;
                }
            }

            match center_miss_t1_new_error.cmp(&center_miss_t1_unchanged_error) {
                Ordering::Less => {
                    center_miss_t1_candidate = i;
                    center_miss_t1_error = center_miss_t1_new_error;
                }
                Ordering::Equal => {
                    if center_miss_t1_flip {
                        center_miss_t1_candidate = i;
                    }
                    center_miss_t1_flip = !center_miss_t1_flip;
                    center_miss_t1_error = center_miss_t1_new_error;
                }
                Ordering::Greater => {
                    center_miss_t1_error = center_miss_t1_unchanged_error;
                }
            }

            let center_hits_error_new =
                center_hits_t1_error + total_hits - hits[&i].cumulative_count;
            match center_hits_error_new.cmp(&error) {
                Ordering::Less => {
                    candidate = (center_hits_t1_candidate, i, true);
                    error = center_hits_error_new;
                }
                Ordering::Equal => {
                    if candidate_flip {
                        candidate = (center_hits_t1_candidate, i, true);
                    }
                    candidate_flip = !candidate_flip;
                }
                Ordering::Greater => { /* Nothing to do, the set of best candidates unchanged */ }
            }

            let center_miss_error_new =
                center_miss_t1_error + total_miss - miss[&i].cumulative_count;
            match center_miss_error_new.cmp(&error) {
                Ordering::Less => {
                    candidate = (center_miss_t1_candidate, i, false);
                    error = center_miss_error_new;
                }
                Ordering::Equal => {
                    if candidate_flip {
                        candidate = (center_miss_t1_candidate, i, false);
                    }
                    candidate_flip = !candidate_flip;
                }
                Ordering::Greater => { /* Nothing to do, the set of best candidates unchanged */ }
            }
        }

        let (t1, t2, center_hit) = candidate;

        assert!(t1 <= t2);
        let dual_threshold = if center_hit {
            DualThreshold::new(t1, t2)
        } else {
            if t1 == t2 {
                panic!("Tried to use all miss as a dual threshold classifier, unsupported");
            }
            DualThreshold::new(t2, t1)
        };
        let error_pred = dual_threshold.error_prediction(hits, miss);
        assert_eq!(error_pred.total_error(), error);

        Some((dual_threshold, error_pred))
    }

    fn select_best_classifier<T: Borrow<Self::E> + Copy, U: Copy>(
        &self,
        classifiers: Vec<(T, U)>,
    ) -> Option<(T, U)> {
        // TODO This function seems wrong ?
        let n = classifiers.len();
        if n == 0 {
            return None;
        }
        // 2. Find the classifier minimizing the distance to the others ?
        let mut total_distance = vec![0usize; n];
        for i in 0..n {
            for j in 0..n {
                total_distance[i] += distance(classifiers[i].0.borrow(), classifiers[j].0.borrow())
            }
        }
        let indexes = total_distance
            .into_iter()
            .enumerate()
            .min_set_by_key(|(_, k)| *k);
        assert!(indexes.len() > 0);
        let index = (indexes.len() - 1) / 2;

        Some(classifiers[index])
    }
}

impl<const WIDTH: u64, const N: usize> Display for DualThreshold<SimpleBucketU64<WIDTH, N>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let mut m_t_h_time: u64 = self.miss_to_hit_index.into();
        m_t_h_time += WIDTH;
        let mut h_t_m_time: u64 = self.hit_to_miss_index.into();
        h_t_m_time += WIDTH;
        if self.center_hit {
            write!(f, "{} < hit < {}", m_t_h_time, h_t_m_time)
        } else {
            write!(f, "{} < miss < {}", h_t_m_time, m_t_h_time)
        }
    }
}

fn distance<const WIDTH: u64, const N: usize>(
    a: &DualThreshold<SimpleBucketU64<WIDTH, N>>,
    b: &DualThreshold<SimpleBucketU64<WIDTH, N>>,
) -> usize {
    a.miss_to_hit_index.abs_diff(b.miss_to_hit_index)
        + a.hit_to_miss_index.abs_diff(b.hit_to_miss_index)
        + if a.center_hit ^ b.center_hit { N } else { 0 }
}

impl<const WIDTH: u64, const N: usize, T> DynErrorPredictionsBuilder<WIDTH, N> for T
where
    T: Sized + ErrorPredictionsBuilder<WIDTH, N>,
{
    fn enumerate_error_predictions(
        &self,
        hits: &StaticHistogramCumSum<WIDTH, N>,
        miss: &StaticHistogramCumSum<WIDTH, N>,
    ) -> Vec<(Box<dyn ErrorPredictor<WIDTH, N>>, ErrorPrediction)> {
        ErrorPredictionsBuilder::enumerate_error_predictions(self, hits, miss)
            .into_iter()
            .map(|(e, pred)| (Box::new(e) as Box<dyn ErrorPredictor<WIDTH, N>>, pred))
            .collect()
    }

    fn enumerate_classifiers(&self) -> Vec<Box<dyn ErrorPredictor<WIDTH, N>>> {
        ErrorPredictionsBuilder::enumerate_classifiers(self)
            .into_iter()
            .map(|c| Box::new(c) as Box<dyn ErrorPredictor<WIDTH, N>>)
            .collect()
    }

    fn find_best_classifier(
        &self,
        hits: &StaticHistogramCumSum<WIDTH, N>,
        miss: &StaticHistogramCumSum<WIDTH, N>,
    ) -> Option<(Box<dyn ErrorPredictor<WIDTH, N>>, ErrorPrediction)> {
        if let Some((e, pred)) = ErrorPredictionsBuilder::find_best_classifier(self, hits, miss) {
            Some((Box::new(e), pred))
        } else {
            None
        }
    }
}

pub fn compute_theoretical_optimum_error<const WIDTH: u64, const N: usize, T>(
    hits: &StaticHistogramCumSum<WIDTH, N>,
    miss: &StaticHistogramCumSum<WIDTH, N>,
) -> ErrorPrediction {
    let mut res = ErrorPrediction {
        true_hit: 0,
        true_miss: 0,
        false_hit: 0,
        false_miss: 0,
    };
    for i in SimpleBucketU64::<WIDTH, N>::MIN..=SimpleBucketU64::<WIDTH, N>::MAX {
        if hits[&i].count >= miss[&i].count {
            res.true_hit += hits[&i].count as u64;
            res.false_hit += miss[&i].count as u64;
        } else {
            res.false_miss += hits[&i].count as u64;
            res.true_miss += miss[&i].count as u64;
        }
    }
    res
}
