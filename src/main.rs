// main.rs
// main file of the kernel

#![no_std] // This is a free standing program
#![no_main] // This has no crt0
#![feature(custom_test_frameworks)]
#![test_runner(dendrobates_tinctoreus_azureus::test_runner)]
#![reexport_test_harness_main = "test_main"]
extern crate alloc;

use bootloader::{entry_point, BootInfo};
use core::panic::PanicInfo;
use dendrobates_tinctoreus_azureus::allocator;
use polling_serial::serial_println;
use vga_buffer::println;

use core::cmp::Ord;
use core::ops::Sub;

#[cfg(not(test))]
use dendrobates_tinctoreus_azureus::hlt_loop;

use bootloader::bootinfo::MemoryRegionType::{InUse, Usable};
use bootloader::bootinfo::{FrameRange, MemoryMap, MemoryRegion};
use cache_utils::maccess;
use cache_utils::prefetcher::{enable_prefetchers, prefetcher_fun};
use dendrobates_tinctoreus_azureus::memory;
#[cfg(not(test))]
use vga_buffer::{set_colors, Color, ForegroundColor};
use x86_64::structures::paging::frame::PhysFrameRange;
use x86_64::structures::paging::{
    Mapper, MapperAllSizes, Page, PageSize, PageTableFlags, PhysFrame, Size4KiB,
};
use x86_64::PhysAddr;
use x86_64::VirtAddr;

use arrayref;
use cache_utils::calibration::Verbosity;

// Custom panic handler, required for freestanding program
#[cfg(not(test))]
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    serial_println!("{}", info);
    set_colors(ForegroundColor::LightRed, Color::Blue);
    println!("{}", info);
    x86_64::instructions::bochs_breakpoint();
    hlt_loop();
}

fn distance<T: Sub<Output = T> + Ord>(a: T, b: T) -> T {
    if a > b {
        a - b
    } else {
        b - a
    }
}

// 4k with metric suffix is not capitalized.
#[allow(non_upper_case_globals)]
const VICTIM_4k_START: u64 = 0x0ccc_0000_0000_u64;
#[allow(non_upper_case_globals)]
const VICTIM_4k_END: u64 = VICTIM_4k_START + (1 << 21);
entry_point!(kernel_main);

// Kernel entry point
fn kernel_main(boot_info: &'static BootInfo) -> ! {
    // TODO: Take care of cpuid stuff and set-up all floating point extensions
    // TODO: We may also need to enable debug registers ?

    println!("Hello Blue Frog");
    dendrobates_tinctoreus_azureus::init();
    #[cfg(test)]
    test_main();

    //serial_println!("Memory map: {:#?}", boot_info.memory_map);

    let phys_mem_offset = VirtAddr::new(boot_info.physical_memory_offset);

    // Let's reserve some memory for evil purposes

    // create our memoryMap carving out one 2MiB region for us
    let mut memory_map = MemoryMap::new();

    let mut victim = None;

    for region in boot_info.memory_map.iter() {
        /*if region.region_type == Usable {
            serial_println!("Usable Region: {:?}", region);
        }*/
        let new_region = {
            if victim.is_none()
                && region.region_type == Usable
                && region.range.end_addr() - region.range.start_addr() >= (1 << 21)
            {
                if region.range.start_addr() & ((1 << 21) - 1) == 0 {
                    victim = Some(MemoryRegion {
                        range: FrameRange::new(
                            region.range.start_addr(),
                            region.range.start_addr() + (1 << 21),
                        ),
                        region_type: InUse,
                    });
                    MemoryRegion {
                        range: FrameRange::new(
                            region.range.start_addr() + (1 << 21),
                            region.range.end_addr(),
                        ),
                        region_type: Usable,
                    }
                } else if region.range.end_addr() & ((1 << 21) - 1) == 0 {
                    victim = Some(MemoryRegion {
                        range: FrameRange::new(
                            region.range.end_addr() - (1 << 21),
                            region.range.end_addr(),
                        ),
                        region_type: InUse,
                    });
                    MemoryRegion {
                        range: FrameRange::new(
                            region.range.start_addr(),
                            region.range.end_addr() - (1 << 21),
                        ),
                        region_type: Usable,
                    }
                } else {
                    *region
                }
            } else {
                *region
            }
        };
        memory_map.add_region(new_region);
    }

    // Save the physical addresses, and map 4k pages at a well chosen virtual address range to it.
    // Also grab the virtual addresses for the offset mapping.
    let victim = if let Some(victim) = victim {
        memory_map.add_region(victim);
        victim
    } else {
        unimplemented!();
    };

    // Cast all this to proper references

    // new: initialize a mapper
    let mut frame_allocator = unsafe { memory::BootInfoFrameAllocator::init(memory_map) };

    let mut mapper = unsafe { memory::init(phys_mem_offset) };

    serial_println!("Physical memory offset: {:?}", phys_mem_offset);

    /*
    let addresses = [
        // the identity-mapped vga buffer page
        0xb8000,
        // some code page
        0x20_1008,
        // some stack page
        0x0100_0020_1a10,
        // virtual address mapped to physical address 0
        boot_info.physical_memory_offset,
    ];

    for &address in &addresses {
        let virt = VirtAddr::new(address);
        // new: use the `mapper.translate_addr` method
        let phys = mapper.translate_addr(virt);
        serial_println!("{:?} -> {:?}", virt, phys);
    }
    */
    for (page, frame) in (VICTIM_4k_START..VICTIM_4k_END)
        .step_by(Size4KiB::SIZE as usize)
        .zip(PhysFrameRange {
            start: PhysFrame::containing_address(PhysAddr::new(victim.range.start_addr())),
            end: PhysFrame::containing_address(PhysAddr::new(victim.range.end_addr())),
        })
    {
        //serial_println!("Mapping page {:x} on frame {:?}", page, frame);
        unsafe {
            mapper.map_to(
                Page::<Size4KiB>::containing_address(VirtAddr::new(page)),
                frame,
                PageTableFlags::PRESENT | PageTableFlags::WRITABLE,
                &mut frame_allocator,
            )
        }
        .expect("Failed to map the experiment buffer")
        .flush();
        let phys = mapper.translate_addr(VirtAddr::new(page));
        /*serial_println!(
            "Mapped page {:p}({:?}) on frame {:?}",
            page as *mut u8,
            VirtAddr::new(page),
            phys
        );*/

        unsafe { maccess(page as *mut u8) };
    }

    allocator::init_heap(&mut mapper, &mut frame_allocator).expect("heap initialization failed");

    let caches = cache_utils::cache_info::get_cache_info();
    serial_println!("Caches:");
    serial_println!("{:#?}", caches);

    println!("Caches: {:?}", caches);
    let mut cache_line_size: Option<u16> = None;
    for cache in caches {
        if let Some(cache_line_size) = cache_line_size {
            if cache_line_size != cache.cache_line_size {
                unimplemented!("Does not support multiple cache line for now");
            }
        } else {
            cache_line_size = Some(cache.cache_line_size)
        }
    }

    let cache_line_size = cache_line_size.unwrap_or(64) as usize;

    serial_println!("cache line size: {}", cache_line_size);

    println!(
        "prefetcher status: {}",
        cache_utils::prefetcher::prefetcher_status()
    );

    let threshold_access_p = cache_utils::calibration::calibrate_access(unsafe {
        arrayref::array_ref![
            core::slice::from_raw_parts(VICTIM_4k_START as *mut u8, 4096),
            0,
            4096
        ]
    });
    let flush_result_p = cache_utils::calibration::calibrate_flush(
        unsafe {
            arrayref::array_ref![
                core::slice::from_raw_parts(VICTIM_4k_START as *mut u8, 4096),
                0,
                4096
            ]
        },
        cache_line_size,
        Verbosity::RawResult,
    );
    cache_utils::prefetcher::enable_prefetchers(false);
    serial_println!("Prefetcher disabled");
    let threshold_access = cache_utils::calibration::calibrate_access(unsafe {
        arrayref::array_ref![
            core::slice::from_raw_parts(VICTIM_4k_START as *mut u8, 4096),
            0,
            4096
        ]
    });
    let flush_resut = cache_utils::calibration::calibrate_flush(
        unsafe {
            arrayref::array_ref![
                core::slice::from_raw_parts(VICTIM_4k_START as *mut u8, 4096),
                0,
                4096
            ]
        },
        cache_line_size,
        Verbosity::RawResult,
    );
    serial_println!("Please compare histograms for sanity");

    if distance(threshold_access_p, threshold_access) > 10 {
        panic!("Inconsistent thresholds");
    }

    let threshold_flush = 0; // FIXME

    serial_println!("0");
    let r_no_prefetch = unsafe {
        prefetcher_fun(
            VICTIM_4k_START as *mut u8,
            (victim.range.start_addr() as *mut u8).offset(phys_mem_offset.as_u64() as isize),
            threshold_flush,
        )
    };
    serial_println!("1");
    enable_prefetchers(true);
    let r_prefetch = unsafe {
        prefetcher_fun(
            VICTIM_4k_START as *mut u8,
            (victim.range.start_addr() as *mut u8).offset(phys_mem_offset.as_u64() as isize),
            threshold_flush,
        )
    };

    for (i, (&npf, pf)) in r_no_prefetch.iter().zip(r_prefetch).enumerate() {
        serial_println!("{} {} {}", i, npf, pf);
    }

    // Calibration
    // disable pretechers
    // Calibrate hit / miss rdtsc threshold
    // evaluate cflush hit / miss threshold ?
    // enable prefetcher
    // do the same

    // access the page
    // for i from 1 to 10
    // with prefetcher disabled and then enabled
    // repeat a few time
    // access i consectutive cache line with timing
    // average / plot the times

    // plot any difference

    // Calibration probably deserves some kind of helper function in cache_util
    // This may be tricky to do in a generic way without adding some fixed noise ?

    // Old stuff below

    /* serial_print!("Input a character: ");

    let c = { polling_serial::SERIAL1.lock().read() };

    serial_println!("\nYoutyped '{:x}'", c);


    serial_println!("Preparing nasty fault...");
    unsafe {
        *(0xdead_beef as *mut u64) = 42;
    }

    serial_println!("Survived ? oO");
    */

    // magic break ?
    // x86_64::instructions::bochs_breakpoint();
    panic!("Ooops Sorry");
}

#[cfg(test)]
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    use dendrobates_tinctoreus_azureus::test_panic_handler;
    test_panic_handler(info);
}

#[test_case]
fn float_test() {
    serial_println!("Testing float computations...");
    use volatile::Volatile;
    // Make a few floating points test;
    let vf: f32 = 84798.0;
    let vd: f64 = 0.828494623655914;

    let a: Volatile<f32> = Volatile::new(42.0);
    let b: Volatile<f32> = Volatile::new(2019.);
    let rf = a.read() * b.read();

    let c: Volatile<f64> = Volatile::new(15.410);
    let d: Volatile<f64> = Volatile::new(18.600);

    let rd = c.read() / d.read();

    serial_print!(
        "  {:?} * {:?} = {:?} expected {:?}...",
        a.read(),
        b.read(),
        rf,
        vf
    );
    if rf == vf {
        serial_println!("[ok]");
    } else {
        serial_println!("[fail]");
    }
    serial_print!(
        "  {:?} / {:?} = {:?} expected {:?}...",
        c.read(),
        d.read(),
        rd,
        vd
    );
    if rd == vd {
        serial_println!("[ok]");
    } else {
        serial_println!("[fail]");
    }
    assert_eq!(rf, vf);
    assert_eq!(rd, vd);
    serial_println!("Testing float computations... [ok]");
}
