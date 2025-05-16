use crate::histograms::StaticHistogram;
use alloc::vec::Vec;
use core::fmt::{Debug, Display, Formatter};
use numa_types::NumaNode;
#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};
//#[cfg(any(feature = "use_std", not(feature = "no_std")))]
//extern crate std;
//#[cfg(all(feature = "use_std", not(feature = "no_std")))]
//use std::hash::{Hash, Hasher};
//#[cfg(any(feature = "no_std", not(feature = "use_std")))]
use core::hash::{Hash, Hasher};
use num_rational::Rational64;
use std::ops::{Add, AddAssign};
use std::vec;

pub type VPN = usize;

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct StaticHistCalibrateResult<const WIDTH: u64, const N: usize> {
    pub page: VPN,
    pub offset: isize,
    pub hash: usize,
    pub histogram: Vec<StaticHistogram<WIDTH, N>>,
    pub median: Vec<u64>,
    pub min: Vec<u64>,
    pub max: Vec<u64>,
    pub count: Vec<u64>,
}

#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Default)]

pub struct CoreLocParameters {
    pub socket: bool,
    pub core: bool,
}

#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Default)]

pub struct LocationParameters {
    pub attacker: CoreLocParameters,
    pub victim: CoreLocParameters,
    pub memory_numa_node: bool,
    pub memory_slice: bool,
    pub memory_vpn: bool,
    pub memory_offset: bool,
}

/**
Simple topological location of a core

Note, this should eventually be improved to match the complete CPUID topology v2 leaf levels.
*/
#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Default)]
pub struct CoreLocation {
    pub socket: u8,
    pub core: u16,
}

#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Default)]
pub struct AVMLocation {
    pub attacker: CoreLocation,
    pub victim: CoreLocation,
    pub memory_numa_node: NumaNode,
    pub memory_slice: usize,
    pub memory_vpn: usize,
    pub memory_offset: isize,
}

impl LocationParameters {
    pub fn is_subset(&self, other: &Self) -> bool {
        (!self.attacker.socket || other.attacker.socket)
            && (!self.attacker.core || other.attacker.core)
            && (!self.victim.socket || other.victim.socket)
            && (!self.victim.core || other.victim.core)
            && (!self.memory_numa_node || other.memory_numa_node)
            && (!self.memory_slice || other.memory_slice)
            && (!self.memory_vpn || other.memory_vpn)
            && (!self.memory_offset || other.memory_offset)
    }
}

pub trait PartialLocation {
    fn get_params(&self) -> &LocationParameters;
    fn get_location(&self) -> &AVMLocation;

    fn get_attacker_socket(&self) -> Option<u8> {
        if self.get_params().attacker.socket {
            Some(self.get_location().attacker.socket)
        } else {
            None
        }
    }

    fn get_attacker_core(&self) -> Option<u16> {
        if self.get_params().attacker.core {
            Some(self.get_location().attacker.core)
        } else {
            None
        }
    }

    fn get_victim_socket(&self) -> Option<u8> {
        if self.get_params().attacker.socket {
            Some(self.get_location().victim.socket)
        } else {
            None
        }
    }

    fn get_victim_core(&self) -> Option<u16> {
        if self.get_params().victim.core {
            Some(self.get_location().victim.core)
        } else {
            None
        }
    }
    fn get_numa_node(&self) -> Option<NumaNode> {
        if self.get_params().memory_numa_node {
            Some(self.get_location().memory_numa_node)
        } else {
            None
        }
    }

    fn get_slice(&self) -> Option<usize> {
        if self.get_params().memory_slice {
            Some(self.get_location().memory_slice)
        } else {
            None
        }
    }

    fn get_vpn(&self) -> Option<usize> {
        if self.get_params().memory_vpn {
            Some(self.get_location().memory_vpn)
        } else {
            None
        }
    }
    fn get_offset(&self) -> Option<isize> {
        if self.get_params().memory_offset {
            Some(self.get_location().memory_offset)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Eq)]
pub struct PartialLocationOwned {
    params: LocationParameters,
    location: AVMLocation,
}

#[derive(Debug, Clone, Copy, Eq)]
pub struct PartialLocationRef<'a> {
    params: &'a LocationParameters,
    location: AVMLocation,
}

trait PartialEqImpl {
    fn eq_impl(&self, other: &Self) -> bool;
}

impl<T> PartialEqImpl for T
where
    T: PartialLocation,
{
    fn eq_impl(&self, other: &Self) -> bool {
        let self_params = self.get_params();
        let other_params = other.get_params();
        let self_location = self.get_location();
        let other_location = other.get_location();

        let res = (self_params == other_params)
            && ((!self_params.attacker.socket)
                || self_location.attacker.socket == other_location.attacker.socket)
            && ((!self_params.attacker.core)
                || self_location.attacker.core == other_location.attacker.core)
            && ((!self_params.victim.socket)
                || self_location.victim.socket == other_location.victim.socket)
            && ((!self_params.victim.core)
                || self_location.victim.core == other_location.victim.core)
            && ((!self_params.memory_numa_node)
                || self_location.memory_numa_node == other_location.memory_numa_node)
            && ((!self_params.memory_slice)
                || self_location.memory_slice == other_location.memory_slice)
            && ((!self_params.memory_vpn) || self_location.memory_vpn == other_location.memory_vpn)
            && ((!self_params.memory_offset)
                || self_location.memory_offset == other_location.memory_offset);
        res
    }
}

impl PartialEq for PartialLocationOwned {
    fn eq(&self, other: &Self) -> bool {
        self.eq_impl(other)
    }
}

impl<'a> PartialEq for PartialLocationRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.eq_impl(other)
    }
}

impl PartialLocation for PartialLocationOwned {
    fn get_params(&self) -> &LocationParameters {
        &self.params
    }

    fn get_location(&self) -> &AVMLocation {
        &self.location
    }
}

impl<'a> PartialLocation for PartialLocationRef<'a> {
    fn get_params(&self) -> &LocationParameters {
        self.params
    }

    fn get_location(&self) -> &AVMLocation {
        &self.location
    }
}

trait HashImpl {
    fn hash_impl<H: Hasher>(&self, state: &mut H);
}

impl<T> HashImpl for T
where
    T: PartialLocation,
{
    fn hash_impl<H: Hasher>(&self, state: &mut H) {
        let self_params = self.get_params();
        let self_location = self.get_location();
        self_params.hash(state);
        if self_params.attacker.socket {
            self_location.attacker.socket.hash(state);
        }
        if self_params.attacker.core {
            self_location.attacker.core.hash(state);
        }
        if self_params.victim.socket {
            self_location.victim.socket.hash(state);
        }
        if self_params.victim.core {
            self_location.victim.core.hash(state);
        }
        if self_params.memory_numa_node {
            self_location.memory_numa_node.hash(state);
        }
        if self_params.memory_slice {
            self_location.memory_slice.hash(state);
        }
        if self_params.memory_vpn {
            self_location.memory_vpn.hash(state);
        }
        if self_params.memory_offset {
            self_location.memory_offset.hash(state);
        }
    }
}

impl Hash for PartialLocationOwned {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash_impl(state)
    }
}

impl<'a> Hash for PartialLocationRef<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash_impl(state)
    }
}

impl PartialLocationOwned {
    pub fn new(params: LocationParameters, location: AVMLocation) -> Self {
        Self { params, location }
    }
    fn try_project(&self, params: &LocationParameters) -> Option<Self> {
        if !params.is_subset(&self.params) {
            None
        } else {
            let mut res = self.clone();
            res.params = params.clone();
            Some(res)
        }
    }
    pub fn project(&self, params: &LocationParameters) -> Self {
        self.try_project(params).expect("Impossible projection")
    }
}

impl<'a> PartialLocationRef<'a> {
    pub fn new(params: &'a LocationParameters, location: AVMLocation) -> Self {
        Self { params, location }
    }
    fn try_project<'s, 'b>(
        &'s self,
        params: &'b LocationParameters,
    ) -> Option<PartialLocationRef<'b>> {
        if !params.is_subset(&self.params) {
            None
        } else {
            let res = PartialLocationRef {
                params,
                location: self.location.clone(),
            };
            Some(res)
        }
    }
    pub fn project<'s, 'b>(&'s self, params: &'b LocationParameters) -> PartialLocationRef<'b> {
        self.try_project(params).expect("Impossible projection")
    }
}

pub type Slice = usize;

pub fn cum_sum(vector: &[u32]) -> Vec<u32> {
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

#[derive(Debug, Clone, Copy, Default)]
pub struct ErrorPrediction {
    pub true_hit: u64,
    pub true_miss: u64,
    pub false_hit: u64,
    pub false_miss: u64,
}

impl ErrorPrediction {
    pub fn total_error(&self) -> u64 {
        self.false_hit + self.false_miss
    }
    pub fn total(&self) -> u64 {
        self.false_hit + self.false_miss + self.true_hit + self.true_miss
    }
    pub fn error_rate(&self) -> f64 {
        (self.false_miss + self.false_hit) as f64 / (self.total() as f64)
    }
    pub fn error_ratio(&self) -> Rational64 {
        Rational64::new(
            (self.false_hit + self.false_miss) as i64,
            self.total() as i64,
        )
    }
}

impl Add for &ErrorPrediction {
    type Output = ErrorPrediction;

    fn add(self, rhs: &ErrorPrediction) -> Self::Output {
        ErrorPrediction {
            true_hit: self.true_hit + rhs.true_hit,
            true_miss: self.true_miss + rhs.true_miss,
            false_hit: self.false_hit + rhs.false_hit,
            false_miss: self.true_miss + rhs.false_miss,
        }
    }
}

impl AddAssign<&Self> for ErrorPrediction {
    fn add_assign(&mut self, rhs: &Self) {
        self.true_hit += rhs.true_hit;
        self.true_miss += rhs.true_miss;
        self.false_hit += rhs.false_hit;
        self.false_miss += rhs.false_miss;
    }
}

impl Add for ErrorPrediction {
    type Output = ErrorPrediction;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<&Self> for ErrorPrediction {
    type Output = ErrorPrediction;

    fn add(mut self, rhs: &Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<ErrorPrediction> for &ErrorPrediction {
    type Output = ErrorPrediction;

    fn add(self, mut rhs: ErrorPrediction) -> Self::Output {
        rhs += self;
        rhs
    }
}

impl AddAssign<Self> for ErrorPrediction {
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

impl Display for ErrorPrediction {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{}% ({} True Hits, {} True Miss, {} False Hit, {} False Miss)",
            self.error_rate() * 100.,
            self.true_hit,
            self.true_miss,
            self.false_hit,
            self.false_miss
        )
    }
}
