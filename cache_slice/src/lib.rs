#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
use crate::arch::CpuClass::{IntelCore, IntelXeon, IntelXeonSP};
#[cfg(target_arch = "x86_64")]
use crate::arch::{get_performance_counters_core, get_performance_counters_xeon};
#[cfg(target_arch = "x86_64")]
use crate::msr::{read_msr_on_cpu, write_msr_on_cpu};
use crate::Error::{InvalidParameter, UnsupportedCPU};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_mm_clflush;

mod arch;
pub mod msr;
pub mod utils;

#[derive(Debug)]
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

const NUM_POKE: usize = 100000;
#[cfg(target_arch = "x86_64")]
unsafe fn poke(addr: *const u8) {
    for _i in 0..NUM_POKE {
        unsafe { _mm_clflush(addr) };
    }
}
#[cfg(target_arch = "x86_64")]
unsafe fn monitor_xeon(addr: *const u8, cpu: u8, max_cbox: usize) -> Result<Vec<u64>, Error> {
    let performance_counters = if let Some(p) = get_performance_counters_xeon() {
        p
    } else {
        return Err(UnsupportedCPU);
    };

    if (performance_counters.max_slice as usize) < max_cbox {
        return Err(InvalidParameter);
    }

    // Freeze counters
    for i in 0..max_cbox {
        write_msr_on_cpu(
            performance_counters.msr_pmon_ctr0[i],
            cpu,
            performance_counters.val_box_freeze,
        )?;
    }

    // Reset counters
    for i in 0..max_cbox {
        write_msr_on_cpu(
            performance_counters.msr_pmon_box_ctl[i],
            cpu,
            performance_counters.val_box_reset,
        )?;
    }

    // Enable counting
    for i in 0..max_cbox {
        write_msr_on_cpu(
            performance_counters.msr_pmon_ctl0[i],
            cpu,
            performance_counters.val_enable_counting,
        )?;
    }

    // Select event
    for i in 0..max_cbox {
        write_msr_on_cpu(
            performance_counters.msr_pmon_ctl0[i],
            cpu,
            performance_counters.val_select_event,
        )?;
        write_msr_on_cpu(
            performance_counters.msr_pmon_box_filter[i],
            cpu,
            performance_counters.val_filter,
        )?;
    }

    // Unfreeze
    for i in 0..max_cbox {
        write_msr_on_cpu(
            performance_counters.msr_pmon_box_ctl[i],
            cpu,
            performance_counters.val_box_unfreeze,
        )?;
    }

    unsafe { poke(addr) };

    // Freeze counters
    for i in 0..max_cbox {
        write_msr_on_cpu(
            performance_counters.msr_pmon_box_ctl[i],
            cpu,
            performance_counters.val_box_freeze,
        )?;
    }

    // Read counters
    let mut results = Vec::new();
    for i in 0..max_cbox {
        let result = read_msr_on_cpu(performance_counters.msr_pmon_ctr0[i], cpu)?;
        if result < NUM_POKE as u64 {
            results.push(0);
        } else {
            results.push(result - NUM_POKE as u64);
        }
    }

    Ok(results)
}
#[cfg(target_arch = "x86_64")]
unsafe fn monitor_core(addr: *const u8, cpu: u8) -> Result<Vec<u64>, Error> {
    // Note, we need to add the workaround for one missing perf counter here.
    let performance_counters = if let Some(p) = get_performance_counters_core() {
        p
    } else {
        return Err(UnsupportedCPU);
    };
    #[cfg(debug_assertions)]
    eprint!("Finding the number of CBox available... ");
    let max_cbox = (read_msr_on_cpu(performance_counters.msr_unc_cbo_config, cpu)? & 0xF) as usize; // TODO magic number (mask for bit 3:0)
    #[cfg(debug_assertions)]
    eprintln!("{}", max_cbox);

    if max_cbox > performance_counters.max_slice as usize {
        return Err(InvalidParameter);
    }

    #[cfg(debug_assertions)]
    eprintln!("Disabling counters");
    write_msr_on_cpu(
        performance_counters.msr_unc_perf_global_ctr,
        cpu,
        performance_counters.val_disable_ctrs,
    )?;

    #[cfg(debug_assertions)]
    eprint!("Resetting counters...");
    for i in 0..max_cbox {
        #[cfg(debug_assertions)]
        eprint!(" {i}");
        write_msr_on_cpu(
            performance_counters.msr_unc_cbo_per_ctr0[i],
            cpu,
            performance_counters.val_reset_ctrs,
        )?;
    }
    #[cfg(debug_assertions)]
    eprintln!(" ok");

    #[cfg(debug_assertions)]
    eprintln!("Selecting events");
    for i in 0..max_cbox {
        write_msr_on_cpu(
            performance_counters.msr_unc_cbo_perfevtsel0[i],
            cpu,
            performance_counters.val_select_evt_core,
        )?;
    }

    #[cfg(debug_assertions)]
    eprintln!("enabling counters");
    write_msr_on_cpu(
        performance_counters.msr_unc_perf_global_ctr,
        cpu,
        performance_counters.val_enable_ctrs,
    )?;

    unsafe { poke(addr) };

    /*
    // Commented out in original code : TODO, check if this makes any difference ?
    write_msr_on_cpu(performance_counters.msr_unc_perf_global_ctr, cpu, performance_counters.val_disable_ctrs)?;

    */
    // Read counters
    #[cfg(debug_assertions)]
    eprintln!("Gathering results");
    let mut results = Vec::new();
    for i in 0..max_cbox {
        let result = read_msr_on_cpu(performance_counters.msr_unc_cbo_per_ctr0[i], cpu)?;
        if result < NUM_POKE as u64 {
            results.push(0);
        } else {
            results.push(result - NUM_POKE as u64);
        }
    }

    #[cfg(debug_assertions)]
    eprintln!("disabling counters again");
    write_msr_on_cpu(
        performance_counters.msr_unc_perf_global_ctr,
        cpu,
        performance_counters.val_disable_ctrs,
    )?;

    Ok(results)
}

// Note: max_cbox is not used on Intel Core.
#[cfg(target_arch = "x86_64")]
pub unsafe fn monitor_address(addr: *const u8, cpu: u8, max_cbox: u16) -> Result<Vec<u64>, Error> {
    match arch::determine_cpu_class() {
        Some(IntelCore) => unsafe { monitor_core(addr, cpu) },
        Some(IntelXeon) => unsafe { monitor_xeon(addr, cpu, max_cbox as usize) },
        Some(IntelXeonSP) => {
            // TODO
            Err(UnsupportedCPU)
        }
        None => Err(UnsupportedCPU),
    }
}
#[cfg(target_arch = "x86_64")]
pub fn determine_slice(addr: *const u8, core: u8, nb_cores: u16) -> Option<usize> {
    let res = unsafe { monitor_address(addr, core, nb_cores) }.unwrap();

    let slice = res.iter().enumerate().max_by_key(|(_i, val)| **val);

    let maxi = res.iter().max().unwrap();
    let slice = if *maxi == 0 { None } else { slice };

    match slice {
        Some((slice, _)) => Some(slice),
        None => None,
    }
}
