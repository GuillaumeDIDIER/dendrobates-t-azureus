#![deny(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::_mm_clflush;
use crate::arch::CpuClass::{IntelCore, IntelXeon, IntelXeonSP};
use crate::arch::get_performance_counters_xeon;
use crate::Error::UnsupportedCPU;
use crate::msr::{read_msr_on_cpu, write_msr_on_cpu};

pub mod msr;
pub mod utils;
mod arch;

pub enum Error {
    UnsupportedCPU,
    InvalidParameter,
    IO(std::io::Error),
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Error::IO(value)
    }
}

const NUM_POKE: usize = 10000;

unsafe fn poke(addr: *const u8) {
    for _i in 0..NUM_POKE {
        unsafe { _mm_clflush(addr) };
    }
}

unsafe fn monitor_xeon(addr: *const u8, cpu: u8, max_cbox: usize) -> Result<Vec<u32>, Error> {
    let performance_counters = if let Some(p) = get_performance_counters_xeon() {
        p
    } else {
        return Err(UnsupportedCPU);
    };

    if (performance_counters.max_slice as usize) < max_cbox {
        return Err(Error::InvalidParameter);
    }

    // Freeze counters
    for i in 0..max_cbox {
        write_msr_on_cpu(performance_counters.msr_pmon_ctr0[i], cpu, performance_counters.val_box_freeze)?;
    }

    // Reset counters
    for i in 0..max_cbox {
        write_msr_on_cpu(performance_counters.msr_pmon_ctl0[i], cpu, performance_counters.val_box_reset)?;
    }

    // Enable counting
    for i in 0..max_cbox {
        write_msr_on_cpu(performance_counters.msr_pmon_ctl0[i], cpu, performance_counters.val_enable_counting)?;
    }

    // Select event
    for i in 0..max_cbox {
        write_msr_on_cpu(performance_counters.msr_pmon_ctl0[i], cpu, performance_counters.val_select_event)?;
        write_msr_on_cpu(performance_counters.msr_pmon_box_filter[i], cpu, performance_counters.val_filter)?;
    }

    // Unfreeze
    for i in 0..max_cbox {
        write_msr_on_cpu(performance_counters.msr_pmon_box_ctl[i], cpu, performance_counters.val_box_unfreeze)?;
    }

    unsafe { poke(addr) };

    // Freeze counters
    for i in 0..max_cbox {
        write_msr_on_cpu(performance_counters.msr_pmon_ctr0[i], cpu, performance_counters.val_box_freeze)?;
    }

    // Read counters
    let mut result = Vec::new();
    for i in 0..max_cbox {
        let result = read_msr_on_cpu(performance_counters.msr_pmon_ctr0[i], cpu)?;
        result.push(result)
    }

    Ok(result)
}

fn monitor_core(addr: *const u8, cpu: u8, max_core: u8) -> Result<Vec<u32>, Error> {
    // Note, we need to add the workaround for one missing perf counter here.
    unimplemented!()
}

pub unsafe fn monitor_address(addr: *const u8, cpu: u8, max_cbox: u16) -> Result<Vec<u32>, Error> {
    match arch::determine_cpu_class() {
        Some(IntelCore) => {
            unimplemented!()
        }
        Some(IntelXeon) => {
            unsafe { monitor_xeon(addr, cpu, max_cbox as usize) }
        }
        Some(IntelXeonSP) => { // TODO
            Err(UnsupportedCPU)
        }
        None => {
            Err(UnsupportedCPU)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = 2;
        assert_eq!(result, 2);
    }
}
