#![cfg(target_arch = "x86_64")]
use cpuid::MicroArchitecture;

fn main() {
    println!("{:?}", MicroArchitecture::get_micro_architecture());
}
