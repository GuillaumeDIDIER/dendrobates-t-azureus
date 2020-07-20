use cache_utils::cache_info::get_cache_info;
use cache_utils::complex_addressing::cache_slicing;
use cpuid::MicroArchitecture;

pub fn main() {
    println!("{:#?}", get_cache_info());

    if let Some(uarch) = MicroArchitecture::get_micro_architecture() {
        if let Some(vendor_family_model_stepping) = MicroArchitecture::get_family_model_stepping() {
            let slicing = cache_slicing(
                uarch,
                8,
                vendor_family_model_stepping.0,
                vendor_family_model_stepping.1,
                vendor_family_model_stepping.2,
            );
            println!("{:?}", slicing.image((1 << 12) - 1));
            println!("{:?}", slicing.kernel_compl_basis((1 << 12) - 1));
            println!("{:?}", slicing.image_antecedent((1 << 12) - 1));
        }
    }
}
