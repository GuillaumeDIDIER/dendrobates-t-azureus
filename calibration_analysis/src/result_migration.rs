use crate::result_migration::Format::{MsgPack, Zstd};
use calibration_results::calibration_2t::CalibrateResult2TNuma;
use calibration_results::numa_results::{
    BUCKET_NUMBER, BUCKET_SIZE, NumaCalibrationResultV2, NumaNode, OperationNames,
};
use cpuid::complex_addressing::{CacheAttackSlicing, CacheSlicing};
use cpuid::{CPUVendor, MicroArchitecture};
use rmp_serde::{Deserializer, Serializer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env::args;
use std::path::PathBuf;

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct NumaCalibrationResult<const WIDTH: u64, const N: usize> {
    pub operations: Vec<OperationNames>,
    pub results: Vec<CalibrateResult2TNuma<WIDTH, N>>,
    pub topology_info: HashMap<usize, NumaNode>,
    pub micro_architecture: ((CPUVendor, u32, u32), MicroArchitecture),
    pub slicing: (CacheSlicing, CacheAttackSlicing),
}

#[cfg(all(
    any(feature = "use_std", not(feature = "no_std")),
    feature = "serde_support"
))]
impl<const WIDTH: u64, const N: usize> NumaCalibrationResult<WIDTH, N> {
    pub const EXTENSION: &'static str = "NumaResults.msgpack";
    pub const EXTENSION_ZSTD: &'static str = "NumaResults.msgpack.zst";
    pub fn read_msgpack(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let buf = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                return Err(format!("Failed to open path: {}", e));
            }
        };
        let mut deserializer = Deserializer::new(&buf[..]);
        NumaCalibrationResult::<WIDTH, N>::deserialize(&mut deserializer)
            .map_err(|e| format!("{:?}", e))
    }

    pub fn write_msgpack(&self, path: impl AsRef<std::path::Path>) -> Result<(), ()> {
        let mut f1 = std::fs::File::create(path).map_err(|_e| {})?;
        let mut s = Serializer::new(&mut f1);
        self.serialize(&mut s).map_err(|_e| {})
    }

    pub fn read_msgpack_zstd(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let buf = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                return Err(format!("Failed to open path: {}", e));
            }
        };
        let mut decoder = zstd::Decoder::new(&buf[..]).map_err(|e| format!("{:?}", e))?;
        let mut deserializer = Deserializer::new(&mut decoder);
        NumaCalibrationResult::<WIDTH, N>::deserialize(&mut deserializer)
            .map_err(|e| format!("{:?}", e))
    }

    pub fn write_msgpack_zstd(&self, path: impl AsRef<std::path::Path>) -> Result<(), ()> {
        let f1 = std::fs::File::create(path).map_err(|_e| {})?;
        let mut encoder = zstd::Encoder::new(f1, 0).map_err(|_e| {})?.auto_finish();
        let mut s = Serializer::new(&mut encoder);
        self.serialize(&mut s).map_err(|_e| {})
    }
}

#[derive(Debug)]
enum Format {
    MsgPack,
    Zstd,
}
fn find_src_msgpack<const WIDTH: u64, const N: usize>(
    name: impl AsRef<str>,
) -> Result<(String, Format), String> {
    if name
        .as_ref()
        .ends_with(NumaCalibrationResult::<WIDTH, N>::EXTENSION_ZSTD)
    {
        if std::fs::exists(name.as_ref()).unwrap_or(false) {
            Ok((
                name.as_ref()
                    .strip_suffix(
                        &(String::from(".") + NumaCalibrationResult::<WIDTH, N>::EXTENSION_ZSTD),
                    )
                    .unwrap()
                    .to_owned(),
                Zstd,
            ))
        } else {
            Err(String::from("File not found"))
        }
    } else if name
        .as_ref()
        .ends_with(NumaCalibrationResult::<WIDTH, N>::EXTENSION)
    {
        if std::fs::exists(name.as_ref()).unwrap_or(false) {
            Ok((
                name.as_ref()
                    .strip_suffix(
                        &(String::from(".") + NumaCalibrationResult::<WIDTH, N>::EXTENSION),
                    )
                    .unwrap()
                    .to_owned(),
                MsgPack,
            ))
        } else {
            Err(String::from("File not found"))
        }
    } else {
        let zst_candidate =
            name.as_ref().to_owned() + "." + NumaCalibrationResult::<WIDTH, N>::EXTENSION_ZSTD;
        let msgpack_candidate =
            name.as_ref().to_owned() + "." + NumaCalibrationResult::<WIDTH, N>::EXTENSION;
        if std::fs::exists(&zst_candidate).unwrap_or(false) {
            Ok((name.as_ref().to_owned(), Zstd))
        } else if std::fs::exists(&msgpack_candidate).unwrap_or(false) {
            Ok((name.as_ref().to_owned(), MsgPack))
        } else if std::fs::exists(name.as_ref()).unwrap_or(false) {
            Err(String::from("Could not determine format"))
        } else {
            Err(String::from("File not found"))
        }
    }
}

fn migrate_result<const WIDTH: u64, const N: usize>(name: impl AsRef<str>) -> Result<(), String> {
    println!("Migrating {}...", name.as_ref());

    let (base_name, format) = find_src_msgpack::<WIDTH, N>(name)?;

    println!("Base name: {}, Format: {:?}", base_name, format);

    let old_data = match format {
        Zstd => NumaCalibrationResult::<WIDTH, N>::read_msgpack_zstd(
            PathBuf::from(&base_name)
                .with_added_extension(NumaCalibrationResult::<WIDTH, N>::EXTENSION_ZSTD),
        ),
        MsgPack => NumaCalibrationResult::<WIDTH, N>::read_msgpack(
            PathBuf::from(&base_name)
                .with_added_extension(NumaCalibrationResult::<WIDTH, N>::EXTENSION),
        ),
    }?;

    println!("Successfully deserialized");

    let new_data = NumaCalibrationResultV2 {
        operations: old_data.operations,
        results: old_data.results,
        topology_info: old_data.topology_info,
        vendor_family_model_stepping: old_data.micro_architecture.0,
        slicing: old_data.slicing,
    };

    println!("Successfully converted");

    let path = PathBuf::from(&base_name)
        .with_added_extension(NumaCalibrationResultV2::<WIDTH, N>::EXTENSION_ZSTD);
    println!("Destination Path: {}", path.display());

    new_data
        .write_msgpack_zstd(path)
        .map_err(|()| String::from("Failed to write result."))?;
    Ok(())
}

/**
This tool takes the old format and then reserializes it into the new format, dropping the micro_arch field.
*/
pub fn migrate_results<const WIDTH: u64, const N: usize>() -> Result<(), ()> {
    if let Ok(num_threads) = std::thread::available_parallelism() {
        let rayon_thread = (num_threads.get() * 15) >> 4;
        println!("Using {} cores", rayon_thread);
        rayon::ThreadPoolBuilder::new()
            .num_threads(rayon_thread)
            .build_global()
            .unwrap();
    }
    let mut args = args();
    args.next();
    for argument in args {
        let r = migrate_result::<BUCKET_SIZE, BUCKET_NUMBER>(&argument);
        println!("{argument}: {:?}", r);
    }
    Ok(())
}
