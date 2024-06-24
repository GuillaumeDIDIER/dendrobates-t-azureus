use cache_slice::msr;
use std::env;
fn main() {
    let mut args = env::args().into_iter();
    args.next();
    for arg in args {
        match arg.parse::<u32>() {
            Ok(msr) => {
                match msr::read_msr_on_cpu(msr, 0) {
                    Ok(result) => {
                        println!("MSR {}: {:x}", msr, result);
                    },
                    Err(e) => {
                        eprintln!("Error, failed to read MSR {}: {}", msr, e);
                    }
                }
            },
            Err(e) => {
                eprintln!("Error: {}", e);
                eprintln!("{} is not a valid MSR number", arg);
            }
        }
    }
}