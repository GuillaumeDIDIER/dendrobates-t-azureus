use cache_utils::numa_analysis::{NumaCalibrationResult, BUCKET_NUMBER, BUCKET_SIZE};
use lzma_rs::{xz_compress, xz_decompress};
use rmp_serde::{Deserializer, Serializer};
use serde::{Deserialize, Serialize};
use std::env::args;
use std::fs;
use std::io::{Cursor, Write};

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
    let new_result = {
        let buf = match fs::read(format!("{}.msgpack.xz", name)) {
            Ok(d) => d,
            Err(e) => {
                return Err(());
            }
        };
        let mut data = Vec::new();
        let mut cursor = Cursor::new(&mut data);
        xz_decompress(&mut &buf[..], &mut cursor).unwrap();
        let mut deserializer = Deserializer::new(&data[..]);
        NumaCalibrationResult::<BUCKET_SIZE, BUCKET_NUMBER>::deserialize(&mut deserializer).unwrap()
    };
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
