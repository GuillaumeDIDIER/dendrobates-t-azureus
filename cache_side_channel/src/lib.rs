#![feature(specialization)]
#![feature(unsafe_block_in_unsafe_fn)]
#![deny(unsafe_op_in_unsafe_fn)]

use std::fmt::Debug;

pub mod table_side_channel;

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
    const MAX_ADDR: u32;
    /// # Safety
    ///
    /// addresses must contain only valid pointers to read.
    unsafe fn test<'a, 'b, 'c>(
        &'a mut self,
        addresses: &'b mut (impl Iterator<Item = &'c *const u8> + Clone),
    ) -> Result<Vec<(*const u8, CacheStatus)>, SideChannelError>;

    /// # Safety
    ///
    /// addresses must contain only valid pointers to read.
    unsafe fn prepare<'a, 'b, 'c>(
        &'a mut self,
        addresses: &'b mut (impl Iterator<Item = &'c *const u8> + Clone),
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

impl<T: MultipleAddrCacheSideChannel> SingleAddrCacheSideChannel for T {
    unsafe fn test_single(&mut self, addr: *const u8) -> Result<CacheStatus, SideChannelError> {
        let addresses = vec![addr];
        unsafe { self.test(&mut addresses.iter()) }.map(|v| v[0].1)
    }

    unsafe fn prepare_single(&mut self, addr: *const u8) -> Result<(), SideChannelError> {
        let addresses = vec![addr];
        unsafe { self.prepare(&mut addresses.iter()) }
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

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
