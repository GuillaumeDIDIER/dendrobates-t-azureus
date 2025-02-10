#![feature(step_trait)]
#![deny(unsafe_op_in_unsafe_fn)]
#![cfg_attr(feature = "no_std", no_std)]
extern crate alloc;

use static_assertions::assert_cfg;
assert_cfg!(
    all(
        not(all(feature = "use_std", feature = "no_std")),
        any(feature = "use_std", feature = "no_std")
    ),
    "Choose std or no-std but not both"
);
pub mod calibration;
pub mod calibration_2t;
pub mod histograms;
pub mod numa_results;
