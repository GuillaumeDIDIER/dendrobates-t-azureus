use crate::frequency::Error::{Unimplemented, UnsupportedPlatform};

use crate::rdtsc_fence;
#[cfg(all(target_os = "linux", feature = "std"))]
use libc::sched_getcpu;
#[cfg(all(target_os = "linux", feature = "std"))]
use std::convert::TryInto;
#[cfg(all(target_os = "linux", feature = "std"))]
use std::os::raw::{c_uint, c_ulong};

pub enum Error {
    InsufficentPrivileges,
    UnsupportedPlatform,
    Unimplemented,
}

#[cfg(all(target_os = "linux", feature = "std"))]
#[link(name = "cpupower")]
extern "C" {
    //unsigned long cpufreq_get_freq_kernel(unsigned int cpu);
    fn cpufreq_get_freq_kernel(cpu: c_uint) -> c_ulong;
}

pub fn get_freq_cpufreq_kernel() -> Result<u64, Error> {
    // TODO Add memorization
    return match unsafe { sched_getcpu() }.try_into() {
        Ok(cpu) => Ok(unsafe { cpufreq_get_freq_kernel(cpu) }),
        Err(e) => Err(Unimplemented),
    };
}

pub fn get_frequency() -> Result<u64, Error> {
    if cfg!(target_os = "linux") && cfg!(feature = "std") {
        return get_freq_cpufreq_kernel();
    }

    if cfg!(target_os = "none") {
        // TODO check CPL
        // if sufficient privileges use rdmsr
        // Otherwise return insufficent privileges
        return Err(Unimplemented);
    }
    Err(UnsupportedPlatform)
}

pub fn get_frequency_change_period(period: u64) -> Result<u64, Error> {
    let mut t: u64 = 0;
    let mut freq: u64 = 0;
    let mut last_freq_change: u64 = 0;
    for _ in 0..period {
        let f = get_frequency();
        let time = unsafe { rdtsc_fence() };
        match f {
            Ok(f) => {
                if f != freq {
                    t += time - last_freq_change;
                    last_freq_change = time;
                    freq = f;
                }
            }
            Err(e) => {
                return Err(e);
            }
        }
    }
    return Ok(t / period);
}