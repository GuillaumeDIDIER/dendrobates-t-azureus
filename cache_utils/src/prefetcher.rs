use x86_64::registers::model_specific::Msr;

const MSR_MISC_FEATURE8CONTROL: u32 = 0x1a4;

pub fn prefetcher_status() -> bool {
    let msr = Msr::new(MSR_MISC_FEATURE8CONTROL);
    let value = unsafe { msr.read() };

    value & 0xf != 0xf
}

pub fn enable_prefetchers(status: bool) {
    let mut msr = Msr::new(MSR_MISC_FEATURE8CONTROL);
    let mut value = unsafe { msr.read() } & !0xf;
    if !status {
        value |= 0xf;
    }
    unsafe { msr.write(value) };
}
