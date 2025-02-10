use calibration_results::calibration_2t::{
    calibration_result_to_location_map, calibration_result_to_location_map_parallel,
};
use calibration_results::numa_results::{BUCKET_NUMBER, BUCKET_SIZE, NumaCalibrationResult};
//use lzma_rs::xz_decompress;
use rmp_serde::Deserializer;
use serde::Deserialize;
use std::env::args;
use std::fs;
use std::io::Cursor;

/*
Design to do, we need to extract, for both FR and FF the raw calibration results (HashMap<AVMLoc, Histograms>)
-> From there we can compute the consolidations for all possible models.

-> Separately, we want to make some histograms, and also try to figure out a way to compare stuffs ?

 */

pub fn run_numa_analysis<const WIDTH: u64, const N: usize>(
    data: NumaCalibrationResult<WIDTH, N>,
) -> () /*TODO*/ {
    /* We need dual and single threshold analysis */
    let results = data.results;
    let topology_info = data.topology_info;
    let slicings = data.slicing;
    let operations = data.operations;
    let uarch = data.micro_architecture;
    //let location_map = calibration_result_to_location_map_parallel(results, &|static_hist_result| { () }, &(), &());
    unimplemented!()
}

fn run_analysis_from_file(name: &str) -> Result<(), ()> {
    /*eprintln!("Analysing file {}", name);
    let result = {
        let mut file = match fs::File::open(format!("{}.msgpack", name)) {
            Ok(d) => d,
            Err(e) => {
                return Err(());
            }
        };
        let mut deserializer = Deserializer::new(&mut file);
        NumaCalibrationResult::<BUCKET_SIZE, BUCKET_NUMBER>::deserialize(&mut deserializer).unwrap()
    };
    eprintln!("Read and deserialized {}.msgpack", name);
    /*{
        let mut buf = Vec::new();
        let mut s = Serializer::new(&mut buf);
        result.serialize(&mut s).unwrap();
        let mut f1 = std::fs::File::create(format!("{}.msgpack.xz", name)).unwrap();
        xz_compress(&mut &buf[..], &mut f1).expect("Failed to compress data to the output file")
        f1.flush().unwrap();
    }
    eprintln!("Serialized and wrote {}.msgpack.xz", name);*/*/
    let path = format!(
        "{}.{}",
        name,
        NumaCalibrationResult::<BUCKET_SIZE, BUCKET_NUMBER>::EXTENSION
    );
    let new_result = NumaCalibrationResult::<BUCKET_SIZE, BUCKET_NUMBER>::read_msgpack(&path)
        .expect(&format!("Failed to read msgpack file {}", &path));

    eprintln!("Read and deserialized {}.msgpack.xz", name);
    println!("Operations");
    for op in &new_result.operations {
        println!("{}: {}", op.name, op.display_name);
    }
    println!(
        "Number of Calibration Results: {}",
        new_result.results.len()
    );
    Ok(())
}

fn main() {
    let mut args = args();
    args.next();
    for argument in args {
        let r = run_analysis_from_file(&argument);
        println!("{argument}: {:?}", r);
    }
}
