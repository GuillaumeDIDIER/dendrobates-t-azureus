use crate::complex_addressing::CacheSlicing::{
    ComplexAddressing, NoSlice, SimpleAddressing, Unsupported,
};
use cpuid::{CPUVendor, MicroArchitecture};

#[cfg(feature = "no_std")]
use hashbrown::HashMap;
#[cfg(feature = "no_std")]
use hashbrown::HashSet;

#[cfg(feature = "use_std")]
use std::collections::HashMap;
#[cfg(feature = "use_std")]
use std::collections::HashSet;

#[derive(Debug, Copy, Clone)]
pub struct SimpleAddressingParams {
    pub shift: u8, // How many trailing zeros
    pub bits: u8,  // How many ones
}

#[derive(Debug, Copy, Clone)]
pub enum CacheSlicing {
    Unsupported,
    ComplexAddressing(&'static [usize]),
    SimpleAddressing(SimpleAddressingParams),
    NoSlice,
}
const SANDYBRIDGE_TO_SKYLAKE_FUNCTIONS: [usize; 4] = [
    0b0110_1101_0111_1101_0101_1101_0101_0001_000000,
    0b1011_1010_1101_0111_1110_1010_1010_0010_000000,
    0b1111_0011_0011_0011_0010_0100_1100_0100_000000,
    0b0, // TODO
];

const KABYLAKE_i9_FUNCTIONS: [usize; 4] = [
    0b0000_1111_1111_1101_0101_1101_0101_0001_000000,
    0b0000_0110_1111_1011_1010_1100_0100_1000_000000,
    0b0000_1111_1110_0001_1111_1100_1011_0000_000000,
    0b0, // TODO
];
// missing functions for more than 8 cores.

// FIXME : Need to account for Family Model (and potentially stepping)
// Amongst other thing Crystal well products have a different function. (0x6_46)
// Same thing for Kaby Lake with 8 cores apparently.

pub fn cache_slicing(
    uarch: MicroArchitecture,
    physical_cores: u8,
    vendor: CPUVendor,
    family_model_display: u32,
    stepping: u32,
) -> CacheSlicing {
    let trailing_zeros = physical_cores.trailing_zeros();
    if physical_cores != (1 << trailing_zeros) {
        return Unsupported;
    }

    match uarch {
        MicroArchitecture::Skylake | MicroArchitecture::CoffeeLake => {
            ComplexAddressing(&SANDYBRIDGE_TO_SKYLAKE_FUNCTIONS[0..((trailing_zeros + 1) as usize)])
        }
        MicroArchitecture::KabyLake => {
            ComplexAddressing(&KABYLAKE_i9_FUNCTIONS[0..((trailing_zeros + 1) as usize)])
        }
        MicroArchitecture::SandyBridge
        | MicroArchitecture::HaswellE
        | MicroArchitecture::Broadwell
        | MicroArchitecture::IvyBridge
        | MicroArchitecture::IvyBridgeE => {
            ComplexAddressing(&SANDYBRIDGE_TO_SKYLAKE_FUNCTIONS[0..((trailing_zeros) as usize)])
        }
        MicroArchitecture::Haswell => {
            if family_model_display == 0x06_46 {
                // Crystal Well
                Unsupported
            } else {
                ComplexAddressing(&SANDYBRIDGE_TO_SKYLAKE_FUNCTIONS[0..((trailing_zeros) as usize)])
            }
        }
        MicroArchitecture::Nehalem | MicroArchitecture::Westmere => {
            Unsupported //SimpleAddressing(((physical_cores - 1) as usize) << 6 + 8) // Hardcoded for 4 cores FIXME !!!
        }
        _ => Unsupported,
    }
}

fn hash(addr: usize, mask: usize) -> u8 {
    ((addr & mask).count_ones() & 1) as u8
}

impl CacheSlicing {
    pub fn can_hash(&self) -> bool {
        match self {
            Unsupported | NoSlice | SimpleAddressing(_) => false,
            ComplexAddressing(_) => true,
        }
    }
    pub fn hash(&self, addr: usize) -> Option<u8> {
        match self {
            SimpleAddressing(mask) => None, //Some(addr & *mask),
            ComplexAddressing(masks) => {
                let mut res = 0;
                for mask in *masks {
                    res <<= 1;
                    res |= hash(addr, *mask);
                }
                Some(res)
            }
            _ => None,
        }
    }

    pub fn image(&self, mask: usize) -> Option<HashSet<usize>> {
        None
    }

    pub fn kernel_compl_basis(&self, mask: usize) -> Option<HashMap<usize, usize>> {
        None
    }
}
