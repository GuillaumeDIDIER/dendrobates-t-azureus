use calibration_results::calibration_2t::CalibrateResult2TNuma;
use calibration_results::numa_results::{BUCKET_NUMBER, BUCKET_SIZE, NumaCalibrationResult};
use rmp_serde::Deserializer;
use serde::Deserialize;
use std::env::args;

/**
This tool grab serialized temporary results in a tmp.msgpack.zst (argument 2), and update the NumaResult file (argument 1), with those results.
This is used when saving the fixed_freq experiment failed.
*/
fn main() -> Result<(), ()> {
    // Check arguments
    let mut args: Vec<String> = args().collect();
    if args.len() != 4 {
        eprintln!("Wrong number of arguments");
        return Err(());
    }
    assert_eq!(args.len(), 4);
    let dest = args.pop().unwrap();
    let src = args.pop().unwrap();
    let numa_src = args.pop().unwrap();
    println!(
        "Attempting to merge {} with {} into {}",
        numa_src, src, dest
    );

    if !std::fs::exists(&numa_src).unwrap() {
        eprintln!("Numa src doesn't exist");
        return Err(());
    }

    if !std::fs::exists(&src).unwrap() {
        eprintln!("Data Source doesn't exist");
        return Err(());
    }

    if std::fs::exists(&dest).unwrap() {
        eprintln!("Destination exists");
        return Err(());
    }

    // Open the template file
    let mut numa_results = if numa_src.ends_with(".zst") {
        NumaCalibrationResult::<BUCKET_SIZE, BUCKET_NUMBER>::read_msgpack_zstd(&numa_src)
    } else {
        NumaCalibrationResult::<BUCKET_SIZE, BUCKET_NUMBER>::read_msgpack(&numa_src)
    }
    .map_err(|e| eprintln!("Failed to deserialize: {:?}", e))?;

    // Save the number of entries expected
    let count = numa_results.results.len();

    // Drop the old data
    numa_results.results = Vec::new();

    // Open the serialized data

    if src.ends_with(".zst") {
        let store = std::fs::File::open(src).unwrap();
        let decoder = zstd::Decoder::new(store).unwrap();
        let mut deserializer = Deserializer::new(decoder);

        // Read the data and insert it in the Numa result
        for _i in 0..count {
            let data =
                CalibrateResult2TNuma::<BUCKET_SIZE, BUCKET_NUMBER>::deserialize(&mut deserializer)
                    .unwrap();
            numa_results.results.push(data);
        }
    } else {
        let store = std::fs::File::open(src).unwrap();
        let mut deserializer = Deserializer::new(store);

        // Read the data and insert it in the Numa result
        for _i in 0..count {
            let data =
                CalibrateResult2TNuma::<BUCKET_SIZE, BUCKET_NUMBER>::deserialize(&mut deserializer)
                    .unwrap();
            numa_results.results.push(data);
        }
    }

    // Write out the new result.
    if dest.ends_with(".zst") {
        numa_results.write_msgpack_zstd(dest).unwrap();
    } else {
        numa_results.write_msgpack(dest).unwrap();
    }

    Ok(())
}
