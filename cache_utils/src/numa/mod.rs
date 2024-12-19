mod numa_impl;
mod numa_noop;

#[cfg(feature = "numa")]
pub use numa_impl::*;

#[cfg(not(feature = "numa"))]
pub use numa_noop::*;

#[derive(Debug, PartialEq, Eq)]
pub enum NumaError {
    FailedInit,
    Uninitialized,
    IllegalNode,
    FailedMigration,
}
