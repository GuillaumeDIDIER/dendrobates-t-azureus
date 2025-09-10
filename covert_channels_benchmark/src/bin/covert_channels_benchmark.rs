#![deny(unsafe_op_in_unsafe_fn)]

use cache_utils::calibration::CLFLUSH_BUCKET_NUMBER;
use cache_utils::calibration::CLFLUSH_BUCKET_SIZE;

fn main() {
    covert_channels_benchmark::convert_channel_benchmark::<CLFLUSH_BUCKET_SIZE, CLFLUSH_BUCKET_NUMBER>(
    )
}
