use crate::arch::CpuClass::{IntelCore, IntelXeon, IntelXeonSP};
use raw_cpuid::CpuId;

pub(crate) enum CpuClass {
    IntelCore,
    IntelXeon,
    IntelXeonSP,
    // Add further CPUs later on
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn determine_cpu_class() -> Option<CpuClass> {
    let cpuid = CpuId::new();
    let info = if let Some(info) = cpuid.get_feature_info() {
        info
    } else {
        return None;
    };

    // Todo, sift through the documentation to add support for more CPUs
    match (info.family_id(), info.model_id()) {
        (0x06, 0x2d) | (0x06, 0x3e) | (06, 0x3f) | (0x06, 0x56) | (0x06, 0x4f) => Some(IntelXeon),
        (0x06, 0x55) => Some(IntelXeonSP),
        // 42, 58, 60, 69, 70, 61, 71, 78, 94, 142, 158
        (0x06, 0x2a)
        | (0x06, 0x3a)
        | (0x06, 0x3c)
        | (0x06, 0x45)
        | (0x06, 0x46)
        | (0x06, 0x3d)
        | (0x06, 0x47)
        | (0x06, 0x4e)
        | (0x06, 0x5e)
        | (0x06, 0x8e)
        | (0x06, 0x9e)
        | (0x06, 0xa5)
        | (0x06, 0x66)
        | (0x06, 0x7e)
        | (0x06, 0x8c)
        | (0x06, 0x8d)
        | (0x06, 0xa7)
        | (0x06, 0x9a)
        | (0x06, 0x97)
        | (0x06, 0xba)
        | (0x06, 0xb7) => Some(IntelCore),
        _ => None,
    }
}
#[cfg(target_arch = "x86_64")]
pub(crate) fn get_performance_counters_xeon() -> Option<&'static XeonPerfCounters> {
    let cpuid = CpuId::new();
    let info = if let Some(info) = cpuid.get_feature_info() {
        info
    } else {
        return None;
    };
    if info.family_id() != 6 {
        return None;
    }
    match info.model_id() {
        0x2d /* 45 */ => Some(&SANDY_BRIDGE_XEON),
        0x3e /* 62 */ => Some(&IVY_BRIDGE_XEON),
        0x3f /* 63 */ => Some(&HASWELL_XEON),
        0x4f | 0x56 /* 86 */ => Some(&BROADWELL_XEON),
        _ => None,
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn get_performance_counters_core() -> Option<&'static CorePerfCounters> {
    let cpuid = CpuId::new();
    let info = if let Some(info) = cpuid.get_feature_info() {
        info
    } else {
        return None;
    };
    if info.family_id() != 6 {
        return None;
    }
    // TODO, review if the list can be extended to further CPUs
    // TODO, add post Cannon Lake stuff
    match info.model_id() {
        0x2a | 0x3a | 0x3c | 0x45 | 0x46 | 0x3d | 0x47 => Some(&SANDYBRIDGE_TO_BROADWELL_CORE),
        0x4e | 0x5e | 0x8e | 0x9e | 0xa5 => Some(&SKYLAKE_KABYLAKE_CORE),
        0x66 | 0x7e | 0x8c | 0x8d => Some(&CANNON_LAKE_TO_TIGER_LAKE_CORE),
        0xa7 => {
            eprintln!("Rocket Lake may be like Skylake or like Ice Lake.");
            eprintln!("You need to edit the code in arch.rs, and validate if the perf counters work like any of those two.");
            eprintln!("For now assuming Ice Lake by default");
            Some(&CANNON_LAKE_TO_TIGER_LAKE_CORE)
        }
        0x9a | 0x97 | 0xba | 0xb7 => Some(&ALDER_LAKE_TO_RAPTOR_LAKE_CORE),
        _ => None,
    }
}

pub struct XeonPerfCounters {
    pub max_slice: u16,
    pub msr_pmon_ctr0: &'static [u64],
    pub msr_pmon_box_filter: &'static [u64],
    pub msr_pmon_ctl0: &'static [u64],
    pub msr_pmon_box_ctl: &'static [u64],
    pub val_box_freeze: u64,
    pub val_box_reset: u64,
    pub val_enable_counting: u64,
    pub val_select_event: u64,
    pub val_filter: u64,
    pub val_box_unfreeze: u64,
}

pub struct CorePerfCounters {
    pub msr_unc_cbo_config: u64,
    pub max_slice: u16,
    pub msr_unc_perf_global_ctr: u64,
    pub val_enable_ctrs: u64,
    pub msr_unc_cbo_perfevtsel0: &'static [u64],
    pub msr_unc_cbo_per_ctr0: &'static [u64],
    pub val_disable_ctrs: u64,
    pub val_select_evt_core: u64,
    pub val_reset_ctrs: u64,
}

const SANDY_BRIDGE_XEON: XeonPerfCounters = XeonPerfCounters {
    max_slice: 8,
    msr_pmon_ctr0: &[0xd16, 0xd36, 0xd56, 0xd76, 0xd96, 0xdb6, 0xdd6, 0xdf6],
    msr_pmon_box_filter: &[0xd14, 0xd34, 0xd54, 0xd74, 0xd94, 0xdb4, 0xdd4, 0xdf4],
    msr_pmon_ctl0: &[0xd10, 0xd30, 0xd50, 0xd70, 0xd90, 0xdb0, 0xdd0, 0xdf0],
    msr_pmon_box_ctl: &[0xd04, 0xd24, 0xd44, 0xd64, 0xd84, 0xda4, 0xdc4, 0xde4],
    val_box_freeze: 0x10100,
    val_box_reset: 0x10103,
    val_enable_counting: 0x400000,
    val_select_event: 0x401134,
    val_filter: 0x7c0000,
    val_box_unfreeze: 0x10000,
};

const IVY_BRIDGE_XEON: XeonPerfCounters = XeonPerfCounters {
    max_slice: 15,
    msr_pmon_ctr0: &[
        0xd16, 0xd36, 0xd56, 0xd76, 0xd96, 0xdb6, 0xdd6, 0xdf6, 0xe16, 0xe36, 0xe56, 0xe76, 0xe96,
        0xeb6, 0xed6,
    ],
    msr_pmon_box_filter: &[
        0xd14, 0xd34, 0xd54, 0xd74, 0xd94, 0xdb4, 0xdd4, 0xdf4, 0xe14, 0xe34, 0xe54, 0xe74, 0xe94,
        0xeb4, 0xed4,
    ],
    msr_pmon_ctl0: &[
        0xd10, 0xd30, 0xd50, 0xd70, 0xd90, 0xdb0, 0xdd0, 0xdf0, 0xe10, 0xe30, 0xe50, 0xe70, 0xe90,
        0xeb0, 0xed0,
    ],
    msr_pmon_box_ctl: &[
        0xd04, 0xd24, 0xd44, 0xd64, 0xd84, 0xda4, 0xdc4, 0xde4, 0xe04, 0xe24, 0xe44, 0xe64, 0xe84,
        0xea4, 0xec4,
    ],
    val_box_freeze: 0x30100,
    val_box_reset: 0x30103,
    val_enable_counting: 0x400000,
    val_select_event: 0x401134,
    val_filter: 0x7e0010,
    val_box_unfreeze: 0x30000,
};

const HASWELL_XEON: XeonPerfCounters = XeonPerfCounters {
    max_slice: 18,
    msr_pmon_ctr0: &[
        0xe08, 0xe18, 0xe28, 0xe38, 0xe48, 0xe58, 0xe68, 0xe78, 0xe88, 0xe98, 0xea8, 0xeb8, 0xec8,
        0xed8, 0xee8, 0xef8, 0xf08, 0xf18,
    ],
    msr_pmon_box_filter: &[
        0xe05, 0xe15, 0xe25, 0xe35, 0xe45, 0xe55, 0xe65, 0xe75, 0xe85, 0xe95, 0xea5, 0xeb5, 0xec5,
        0xed5, 0xee5, 0xef5, 0xf05, 0xf15,
    ],
    msr_pmon_ctl0: &[
        0xe01, 0xe11, 0xe21, 0xe31, 0xe41, 0xe51, 0xe61, 0xe71, 0xe81, 0xe91, 0xea1, 0xeb1, 0xec1,
        0xed1, 0xee1, 0xef1, 0xf01, 0xf11,
    ],
    msr_pmon_box_ctl: &[
        0xe00, 0xe10, 0xe20, 0xe30, 0xe40, 0xe50, 0xe60, 0xe70, 0xe80, 0xe90, 0xea0, 0xeb0, 0xec0,
        0xed0, 0xee0, 0xef0, 0xf00, 0xf10,
    ],
    val_box_freeze: 0x30100,
    val_box_reset: 0x30103,
    val_enable_counting: 0x400000,
    val_select_event: 0x401134,
    val_filter: 0x7e0020,
    val_box_unfreeze: 0x30000,
};

const BROADWELL_XEON: XeonPerfCounters = XeonPerfCounters {
    max_slice: 24,
    msr_pmon_ctr0: &[
        0xe08, 0xe18, 0xe28, 0xe38, 0xe48, 0xe58, 0xe68, 0xe78, 0xe88, 0xe98, 0xea8, 0xeb8, 0xec8,
        0xed8, 0xee8, 0xef8, 0xf08, 0xf18, 0xf28, 0xf38, 0xf48, 0xf58, 0xf68, 0xf78,
    ],
    msr_pmon_box_filter: &[
        0xe05, 0xe15, 0xe25, 0xe35, 0xe45, 0xe55, 0xe65, 0xe75, 0xe85, 0xe95, 0xea5, 0xeb5, 0xec5,
        0xed5, 0xee5, 0xef5, 0xf05, 0xf15, 0xf25, 0xf35, 0xf45, 0xf55, 0xf65, 0xf75,
    ],
    msr_pmon_ctl0: &[
        0xe01, 0xe11, 0xe21, 0xe31, 0xe41, 0xe51, 0xe61, 0xe71, 0xe81, 0xe91, 0xea1, 0xeb1, 0xec1,
        0xed1, 0xee1, 0xef1, 0xf01, 0xf11, 0xf21, 0xf31, 0xf41, 0xf51, 0xf61, 0xf71,
    ],
    msr_pmon_box_ctl: &[
        0xe00, 0xe10, 0xe20, 0xe30, 0xe40, 0xe50, 0xe60, 0xe70, 0xe80, 0xe90, 0xea0, 0xeb0, 0xec0,
        0xed0, 0xee0, 0xef0, 0xf00, 0xf10, 0xf20, 0xf30, 0xf40, 0xf50, 0xf60, 0xf70,
    ],
    val_box_freeze: 0x30100,
    val_box_reset: 0x30103,
    val_enable_counting: 0x400000,
    val_select_event: 0x401134,
    val_filter: 0xfe0020,
    val_box_unfreeze: 0x30000,
};

// TODO find appropriate values
const ALDER_LAKE_TO_RAPTOR_LAKE_CORE: CorePerfCounters = CorePerfCounters {
    msr_unc_cbo_config: 0x396,
    max_slice: 10,
    msr_unc_perf_global_ctr: 0x2ff0,
    val_enable_ctrs: 0x20000000, // To validate
    msr_unc_cbo_perfevtsel0: &[
        0x2000, 0x2008, 0x2010, 0x2018, 0x2020, 0x2028, 0x2030, 0x2038, 0x2040, 0x2048,
    ],
    msr_unc_cbo_per_ctr0: &[
        0x2002, 0x200a, 0x2012, 0x201a, 0x2022, 0x202a, 0x2032, 0x203a, 0x2042, 0x204a,
    ],
    val_disable_ctrs: 0x0,         // To validate
    val_select_evt_core: 0x408f34, // To validate
    val_reset_ctrs: 0x0,           // To validate
};

// TODO verify his on ICELAKE, and appropriate values. Also deal with backport Cypress Cove ?
const CANNON_LAKE_TO_TIGER_LAKE_CORE: CorePerfCounters = CorePerfCounters {
    msr_unc_cbo_config: 0x396,
    max_slice: 8, // To validate
    msr_unc_perf_global_ctr: 0xe01,
    val_enable_ctrs: 0x20000000, // To validate
    msr_unc_cbo_perfevtsel0: &[0x700, 0x708, 0x710, 0x718, 0x720, 0x728, 0x730, 0x738],
    msr_unc_cbo_per_ctr0: &[0x702, 0x70a, 0x712, 0x71a, 0x722, 0x72a, 0x732, 0x73a],
    val_disable_ctrs: 0x0,         // To validate
    val_select_evt_core: 0x408f34, // To validate
    val_reset_ctrs: 0x0,           // To validate
};

const SKYLAKE_KABYLAKE_CORE: CorePerfCounters = CorePerfCounters {
    msr_unc_cbo_config: 0x396,
    max_slice: 7,
    msr_unc_perf_global_ctr: 0xe01,
    val_enable_ctrs: 0x20000000,
    msr_unc_cbo_perfevtsel0: &[0x700, 0x710, 0x720, 0x730, 0x740, 0x750, 0x760],
    msr_unc_cbo_per_ctr0: &[0x706, 0x716, 0x726, 0x736, 0x746, 0x756, 0x766],
    val_disable_ctrs: 0x0,
    val_select_evt_core: 0x408f34,
    val_reset_ctrs: 0x0,
};

// This is documented in Intel SDM, 20.3.4.6 (in March 2024 edition)

const SANDYBRIDGE_TO_BROADWELL_CORE: CorePerfCounters = CorePerfCounters {
    msr_unc_cbo_config: 0x396,
    max_slice: 4,
    msr_unc_perf_global_ctr: 0x391,
    // Go in MSR_UNC_PERF_GLOBAL_CTR EN (bit 29) set to one, and route PMI to core 1-4 upon overflow.
    val_enable_ctrs: 0x2000000f,
    msr_unc_cbo_perfevtsel0: &[0x700, 0x710, 0x720, 0x730],
    msr_unc_cbo_per_ctr0: &[0x706, 0x716, 0x726, 0x736],
    val_disable_ctrs: 0x0,
    // Counter Mask (bit 28-24) 0, Inv (23) 0, EN (22) 1, OVF (20) 0, E (18) 0,
    // Unit Mask (bit 15-8) 0x8f, Event Select (bit 7-0) 0x34
    // Event selection from https://perfmon-events.intel.com
    // UNC_CBO_CACHE_LOOKUP.ANY_MESI
    // L3 Lookup any request that access cache and found line in MESI-state. 	EventSel=34H UMask=8FH
    // Counter=0,1
    val_select_evt_core: 0x408f34,
    // TODO
    val_reset_ctrs: 0x0,
};
