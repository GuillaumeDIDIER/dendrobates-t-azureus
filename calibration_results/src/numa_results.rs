extern crate alloc;

use crate::calibration_2t::CalibrateResult2TNuma;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use cpuid::complex_addressing::{CacheAttackSlicing, CacheSlicing};
use cpuid::{CPUVendor, MicroArchitecture};
use numa_types::NumaNode;
#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};
#[cfg(any(feature = "use_std", not(feature = "no_std")))]
extern crate std;
#[cfg(all(feature = "no_std", not(feature = "use_std")))]
use hashbrown::HashMap;
use lzma_rs::xz_compress;
#[cfg(any(feature = "use_std", not(feature = "no_std")))]
use lzma_rs::xz_decompress;
use rmp_serde::{Deserializer, Serializer};
#[cfg(any(feature = "use_std", not(feature = "no_std")))]
use std::collections::HashMap;

/** This module is used to factorize the analysis code, to enable off-line analysis
*/

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct OperationNames {
    pub name: String,
    pub display_name: String,
}

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct NumaCalibrationResult<const WIDTH: u64, const N: usize> {
    pub operations: Vec<OperationNames>,
    pub results: Vec<CalibrateResult2TNuma<WIDTH, N>>,
    pub topology_info: HashMap<usize, NumaNode>,
    pub micro_architecture: ((CPUVendor, u32, u32), MicroArchitecture),
    pub slicing: (CacheSlicing, CacheAttackSlicing),
}

pub const BUCKET_NUMBER: usize = 1024;
pub const BUCKET_SIZE: u64 = 1;

#[cfg(all(
    any(feature = "use_std", not(feature = "no_std")),
    feature = "serde_support"
))]
impl<const WIDTH: u64, const N: usize> NumaCalibrationResult<WIDTH, N> {
    pub const EXTENSION: &'static str = ".NumaResults.msgpack.xz";
    pub fn read_msgpack(path: impl AsRef<std::path::Path>) -> Result<Self, ()> {
        let buf = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                return Err(());
            }
        };
        let mut data = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut data);
        xz_decompress(&mut &buf[..], &mut cursor).unwrap();
        let mut deserializer = Deserializer::new(&data[..]);
        NumaCalibrationResult::<WIDTH, N>::deserialize(&mut deserializer).map_err(|_e| {})
    }

    pub fn write_msgpack(&self, path: impl AsRef<std::path::Path>) -> Result<(), ()> {
        let mut f1 = std::fs::File::create(path).map_err(|_e| {})?;
        let mut buf = Vec::new();
        let mut s = Serializer::new(&mut buf);
        self.serialize(&mut s).map_err(|_e| {})?;
        xz_compress(&mut &buf[..], &mut f1).map_err(|_e| {})
    }
}
