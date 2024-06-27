use std::io::Result;
use cache_slice::msr;
use std::env;

const MSR_UNCORE_RATIO_LIMIT: u64 = 1568;

fn get_uncore_ratio() -> Result<(u8, u8)> {
    let result = msr::read_msr_on_cpu(MSR_UNCORE_RATIO_LIMIT, 0)?;
    Ok(((result >> 8) as u8, (result % (2 << 8)) as u8))
}

fn set_uncore_ratio(min: u8, max: u8) -> Result<()> {
    let value = (max as u64) + ((min as u64) << 8);

    msr::write_msr_on_cpu(
        MSR_UNCORE_RATIO_LIMIT,
        0,
        value
    )
}

fn main() {
    let args: Vec<_> = env::args().collect();

    let (current_min, current_max) = get_uncore_ratio().unwrap();

    println!("Current ratio: min: {}, max: {}", current_min, current_max);
    
    let ratio = if args.len() > 1 {
        args[1].parse::<u8>().unwrap()
    } else {
        (current_max + current_min)/2
    };

    println!("Setting ratio to {}", ratio);
    let _ = set_uncore_ratio(ratio, ratio).unwrap();
}