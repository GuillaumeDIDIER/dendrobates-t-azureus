// main.rs
// main file of the kernel

#![no_std] // This is a free standing program
#![no_main] // This has no crt0
#![feature(custom_test_frameworks)]
#![test_runner(dendrobates_tinctoreus_azureus::test_runner)]
#![reexport_test_harness_main = "test_main"]
extern crate alloc;

use alloc::boxed::Box;
use bootloader::{entry_point, BootInfo};
use core::panic::PanicInfo;
use dendrobates_tinctoreus_azureus::allocator;
use polling_serial::serial_print;
use polling_serial::serial_println;
use vga_buffer; // required for custom panic handler
use vga_buffer::println;
use x86_64;

#[cfg(not(test))]
use dendrobates_tinctoreus_azureus::hlt_loop;

use dendrobates_tinctoreus_azureus::memory::create_example_mapping;
#[cfg(not(test))]
use vga_buffer::{set_colors, Color, ForegroundColor};

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

entry_point!(kernel_main);

// Kernel entry point
fn kernel_main(boot_info: &'static BootInfo) -> ! {
    // TODO: Take care of cpuid stuff and set-up all floating point exetnsions
    // TODO: We may also need to enable debug registers ?

    println!("Hello Blue Frog");
    dendrobates_tinctoreus_azureus::init();
    x86_64::instructions::interrupts::int3(); // new
    #[cfg(test)]
    test_main();

    x86_64::instructions::interrupts::int3();

    use dendrobates_tinctoreus_azureus::memory;
    use x86_64::structures::paging::{MapperAllSizes, PageTable};
    use x86_64::{structures::paging::Page, VirtAddr};

    let phys_mem_offset = VirtAddr::new(boot_info.physical_memory_offset);
    // new: initialize a mapper
    let mut frame_allocator =
        unsafe { memory::BootInfoFrameAllocator::init(&boot_info.memory_map) };

    let mut mapper = unsafe { memory::init(phys_mem_offset) };

    let addresses = [
        // the identity-mapped vga buffer page
        0xb8000,
        // some code page
        0x201008,
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

    allocator::init_heap(&mut mapper, &mut frame_allocator).expect("heap initialization failed");

    let x = Box::new(41);

    serial_print!("Input a character: ");

    let c = { polling_serial::SERIAL1.lock().read() };

    serial_println!("\nYoutyped '{:x}'", c);

    serial_println!("Preparing nasty fault...");
    unsafe {
        *(0xdead_beef as *mut u64) = 42;
    }

    serial_println!("Survived ? oO");

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
