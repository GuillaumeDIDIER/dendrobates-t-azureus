use calibration_analysis::result_migration::migrate_results;
use calibration_results::numa_results::{BUCKET_NUMBER, BUCKET_SIZE};

/** See result_migration.rs */
fn main() -> Result<(), ()> {
    migrate_results::<BUCKET_SIZE, BUCKET_NUMBER>()
}
