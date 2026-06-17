use crate::result_migration::Format::{MsgPack, Zstd};
use calibration_results::calibration_2t::CalibrateResult2TNuma;
use calibration_results::numa_results::{
    BUCKET_NUMBER, BUCKET_SIZE, NumaCalibrationResult, NumaCalibrationResultV2,
};
use rmp_serde::Deserializer;
use serde::Deserialize;
use std::env::args;
use std::path::PathBuf;

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
