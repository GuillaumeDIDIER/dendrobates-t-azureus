pub mod numa_impl;
pub mod numa_noop;

#[cfg(feature = "numa")]
pub use numa_impl::*;

#[cfg(not(feature = "numa"))]
pub use numa_noop::*;
