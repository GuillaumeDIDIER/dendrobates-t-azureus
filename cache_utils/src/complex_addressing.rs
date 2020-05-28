use crate::complex_addressing::CacheSlicing::{
    ComplexAddressing, NoSlice, SimpleAddressing, Unsupported,
};
use cpuid::MicroArchitecture;

#[derive(Debug, Copy, Clone)]
pub enum CacheSlicing {
    Unsupported,
    ComplexAddressing(&'static [usize]),
    SimpleAddressing(&'static usize),
    NoSlice,
}
const SANDYBRIDGE_TO_SKYLAKE_FUNCTIONS: [usize; 4] = [
    0b0110_1101_0111_1101_0101_1101_0101_0001_000000,
    0b1011_1010_1101_0111_1110_1010_1010_0010_000000,
    0b1111_0011_0011_0011_0010_0100_1100_0100_000000,
    0b0, // TODO
];
// missing functions for more than 8 cores.

pub fn cache_slicing(uarch: MicroArchitecture, physical_cores: u8) -> CacheSlicing {
    let trailing_zeros = physical_cores.trailing_zeros();
    if physical_cores != (1 << trailing_zeros) {
        return Unsupported;
    }

    match uarch {
        MicroArchitecture::Skylake
        | MicroArchitecture::KabyLake
        | MicroArchitecture::CoffeeLake => {
            ComplexAddressing(&SANDYBRIDGE_TO_SKYLAKE_FUNCTIONS[0..((trailing_zeros + 1) as usize)])
        }
        MicroArchitecture::SandyBridge => {
            ComplexAddressing(&SANDYBRIDGE_TO_SKYLAKE_FUNCTIONS[0..((trailing_zeros) as usize)])
        }
        _ => Unsupported,
    }
}

fn hash(addr: usize, mask: usize) -> usize {
    ((addr & mask).count_ones() & 1) as usize
}

impl CacheSlicing {
    pub fn can_hash(&self) -> bool {
        match self {
            Unsupported | NoSlice => false,
            ComplexAddressing(_) | SimpleAddressing(_) => true,
        }
    }
    pub fn hash(&self, addr: usize) -> Option<usize> {
        match self {
            SimpleAddressing(&mask) => Some((addr & mask)),
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
}
