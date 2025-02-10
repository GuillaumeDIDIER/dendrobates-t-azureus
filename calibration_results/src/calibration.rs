use crate::histograms::StaticHistogram;
use alloc::vec::Vec;
use core::fmt::Debug;
use numa_types::NumaNode;
#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};
//#[cfg(any(feature = "use_std", not(feature = "no_std")))]
//extern crate std;
//#[cfg(all(feature = "use_std", not(feature = "no_std")))]
//use std::hash::{Hash, Hasher};
//#[cfg(any(feature = "no_std", not(feature = "use_std")))]
use core::hash::{Hash, Hasher};

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
    pub memory_slice: u8,
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

    fn get_attacker_socker(&self) -> Option<u8> {
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

    fn get_victim_socker(&self) -> Option<u8> {
        if self.get_params().attacker.socket {
            Some(self.get_location().attacker.socket)
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

    fn get_slice(&self) -> Option<u8> {
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
