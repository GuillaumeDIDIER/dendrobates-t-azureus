// main.rs
// main file of the kernel

#![no_std] // This is a free standing program
#![no_main] // This has no crt0
#![feature(custom_test_frameworks)]
#![test_runner(dendrobates_tinctoreus_azureus::test_runner)]
#![reexport_test_harness_main = "test_main"]

use polling_serial::{serial_print, serial_println};
use vga_buffer::{print, println, set_colors, Color, ForegroundColor};

use core::fmt::Write;
use core::panic::PanicInfo;
use vga_buffer; // required for custom panic handler

use dendrobates_tinctoreus_azureus::hlt_loop;
use x86_64;

use bootloader::BootInfo;

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
pub extern "C" fn kernal_main(boot_info: &'static BootInfo) -> ! {
    // TODO: Take care of cpuid stuff and set-up all floating point exetnsions
    // TODO: We may also need to enable debug registers ?

    println!("Hello Blue Frog");
    dendrobates_tinctoreus_azureus::init();
    x86_64::instructions::interrupts::int3(); // new
    #[cfg(test)]
    test_main();

    println!("Preparing nasty fault...");

    x86_64::instructions::interrupts::int3();

    use x86_64::registers::control::Cr3;

    let (level_4_page_table, flags) = Cr3::read();
    println!(
        "Level 4 page table at: {:?}, flags {:?}",
        level_4_page_table.start_address(),
        flags
    );

    unsafe {
        *(0xdeadbeef as *mut u64) = 42;
    }

    println!("Survived ? oO");

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
    if (rf == vf) {
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
    if (rd == vd) {
        serial_println!("[ok]");
    } else {
        serial_println!("[fail]");
    }
    assert_eq!(rf, vf);
    assert_eq!(rd, vd);
    serial_println!("Testing float computations... [ok]");
}

//#[test_case]
//fn failing_assertion() {
//    print!("trivial assertion... ");
//    serial_print!("trivial assertion... ");
//    assert_eq!(1, 1);
//    println!("[ok]");
//    serial_println!("[ok]");
//}
