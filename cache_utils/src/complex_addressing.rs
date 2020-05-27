use crate::complex_addressing::CacheSlicing::{ComplexAddressing, Unsupported};
use cpuid::MicroArchitecture;

pub enum CacheSlicing {
    Unsupported,
    ComplexAddressing(&'static [usize]),
    SimpleAddressing(&'static usize),
    NoSlice,
}
const SANDYBRIDGE_TO_SKYLAKE_FUNCTIONS: [usize; 3] = [
    0b0110_1101_0111_1101_0101_1101_0101_0001_000000,
    0b1011_1010_1101_0111_1110_1010_1010_0010_000000,
    0b1111_0011_0011_0011_0010_0100_1100_0100_000000,
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
        _ => Unsupported,
    }
}

pub struct AddressHasher<'a> {
    masks: &'a [usize],
}

fn hash(addr: usize, mask: usize) -> u32 {
    (addr & mask).count_ones() & 1
}

impl AddressHasher<'_> {
    pub fn new(masks: &[usize]) -> AddressHasher {
        AddressHasher { masks }
    }
    pub fn hash(&self, addr: usize) -> u32 {
        let mut res = 0;
        for mask in self.masks {
            res <<= 1;
            res |= hash(addr, *mask);
        }
        res
    }
}
