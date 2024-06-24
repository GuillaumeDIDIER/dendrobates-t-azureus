use core::mem::size_of;
use std::format;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::io::{Result, Error};

pub fn write_msr_on_cpu(msr: u64, cpu: u8, value: u64) -> Result<()> {
    let path = format!("/dev/cpu/{}/msr", cpu);
    let file: File = OpenOptions::new().write(true).open(path).expect("Failed to open MSR, are you running as root ?");
    match file.write_at(&value.to_ne_bytes(), msr) {
        Ok(size) => {
            if size == size_of::<u64>() {
                Ok(())
            } else {
                Err(Error::other("Failed to write complete value"))
            }
        }
        Err(e) => Err(e)
    }
}

pub fn read_msr_on_cpu(msr: u64, cpu: u8) -> Result<u64> {
    let path = format!("/dev/cpu/{}/msr", cpu);
    let file: File = OpenOptions::new().read(true).open(path).expect("Failed to open MSR, are you running as root ?");
    let mut read_data = [0u8; size_of::<u64>()];
    match file.read_at(&mut read_data, msr) {
        Ok(size) => {
            if size == size_of::<u64>() {
                Ok(u64::from_ne_bytes(read_data))
            } else {
                Err(Error::other("Failed to write complete value"))
            }
        }
        Err(e) => Err(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // TODO how can we test model specific register read / write ?
}
