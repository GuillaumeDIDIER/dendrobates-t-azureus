#![no_std]
#![no_main]
#![feature(custom_test_frameworks)]
#![test_runner(dendrobates_tinctoreus_azureus::test_runner)]
#![reexport_test_harness_main = "test_main"]

use core::panic::PanicInfo;
use polling_serial::{serial_print, serial_println};
use vga_buffer::{print, println};

#[no_mangle] // don't mangle the name of this function
pub extern "C" fn _start() -> ! {
    test_main();

    loop {}
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    dendrobates_tinctoreus_azureus::test_panic_handler(info)
}

#[test_case]
fn test_println() {
    serial_print!("test_println... ");
    println!("test_println output");
    serial_println!("[ok]");
}
