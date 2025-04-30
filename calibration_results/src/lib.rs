#![feature(generic_const_exprs)]
#![feature(step_trait)]
#![deny(unsafe_op_in_unsafe_fn)]
#![cfg_attr(feature = "no_std", no_std)]
extern crate alloc;

use core::hash::Hash;
use static_assertions::assert_cfg;
#[cfg(any(feature = "use_std", not(feature = "no_std")))]
extern crate std;
#[cfg(all(feature = "no_std", not(feature = "use_std")))]
use hashbrown::HashMap;
#[cfg(any(feature = "use_std", not(feature = "no_std")))]
use std::collections::HashMap;

assert_cfg!(
    all(
        not(all(feature = "use_std", feature = "no_std")),
        any(feature = "use_std", feature = "no_std")
    ),
    "Choose std or no-std but not both"
);
pub mod calibration;
pub mod calibration_2t;
pub mod classifiers;
pub mod histograms;
pub mod numa_results;

pub fn map_values<K, U, V, F>(input: HashMap<K, U>, f: F) -> HashMap<K, V>
where
    K: Hash + Eq,
    F: Fn(U, &K) -> V,
{
    let mut results = HashMap::new();
    for (k, u) in input {
        let f_u = f(u, &k);
        results.insert(k, f_u);
    }
    results
}

pub fn accumulate<K, V, RK, Reduction, Accumulator, Accumulation, AccumulatorDefault>(
    input: HashMap<K, V>,
    reduction: Reduction,
    accumulator_default: AccumulatorDefault,
    aggregation: Accumulation,
) -> HashMap<RK, Accumulator>
where
    K: Hash + Eq + Copy,
    RK: Hash + Eq + Copy,
    Reduction: Fn(K) -> RK,
    Accumulation: Fn(&mut Accumulator, V, K, RK) -> (),
    AccumulatorDefault: Fn() -> Accumulator,
{
    let mut accumulators = HashMap::new();
    for (k, v) in input {
        let rk = reduction(k);
        aggregation(
            accumulators
                .entry(rk)
                .or_insert_with(|| accumulator_default()),
            v,
            k,
            rk,
        );
    }
    accumulators
}

pub fn reduce<K, V, RK, RV, Reduction, Accumulator, Accumulation, AccumulatorDefault, Extract>(
    input: HashMap<K, V>,
    reduction: Reduction,
    accumulator_default: AccumulatorDefault,
    aggregation: Accumulation,
    extraction: Extract,
) -> HashMap<RK, RV>
where
    K: Hash + Eq + Copy,
    RK: Hash + Eq + Copy,
    Reduction: Fn(K) -> RK,
    AccumulatorDefault: Fn() -> Accumulator,
    Accumulation: Fn(&mut Accumulator, V, K, RK) -> (),
    Extract: Fn(Accumulator, &RK) -> RV,
{
    let accumulators = accumulate(input, reduction, accumulator_default, aggregation);
    let result = map_values(accumulators, extraction);
    result
}
