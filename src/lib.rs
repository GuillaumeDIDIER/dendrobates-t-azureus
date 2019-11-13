#![no_std]
#![cfg_attr(test, no_main)]
#![feature(custom_test_frameworks)]
#![test_runner(crate::test_runner)]
#![reexport_test_harness_main = "test_main"]
#![feature(abi_x86_interrupt)]
#![feature(asm)]

use core::panic::PanicInfo;



#[cfg(test)]
use vga_buffer::print;

#[cfg(test)]
use polling_serial::serial_print;

use polling_serial::serial_println;


use vga_buffer::println;
use x86_64::instructions::bochs_breakpoint;

pub mod gdt;
pub mod interrupts;
pub mod memory;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum QemuExitCode {
    Success = 0x10,
    Failed = 0x11,
}

// Custom panic handler, required for freestanding program

pub fn test_panic_handler(info: &PanicInfo) -> ! {
    serial_println!("[failed]\n");
    serial_println!("Error: {}\n", info);
    exit_qemu(QemuExitCode::Failed);
}

// Assumes isa-debug-device at 0xf4, of size 4
pub fn exit_qemu(exit_code: QemuExitCode) -> ! {
    use x86_64::instructions::port::Port;

    unsafe {
        let mut port = Port::new(0xf4);
        port.write(exit_code as u32);
    }
    loop {
        bochs_breakpoint();
    }
}

pub fn test_runner(tests: &[&dyn Fn()]) {
    println!("Running {} tests", tests.len());
    serial_println!("Running {} tests", tests.len());
    for test in tests {
        test();
    }

    exit_qemu(QemuExitCode::Success);
}

pub fn init() {
    gdt::init();
    interrupts::init_idt();
}

pub fn hlt_loop() -> ! {
    loop {
        bochs_breakpoint();
        x86_64::instructions::hlt();
    }
}


#[cfg(test)]
use bootloader::{entry_point, BootInfo};

#[cfg(test)]
entry_point!(test_kernel_main);

/// Entry point for `cargo xtest`
#[cfg(test)]
fn test_kernel_main(_boot_info: &'static BootInfo) -> ! {
    init();
    test_main();
    loop {}
}

#[cfg(test)]
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    test_panic_handler(info)
}

#[test_case]
fn trivial_assertion() {
    print!("trivial assertion... ");
    serial_print!("trivial assertion... ");
    assert_eq!(1, 1);
    println!("[ok]");
    serial_println!("[ok]");
}

#[test_case]
fn printf_test() {
    serial_print!("Testing VGA print/println... ");
    println!("Are frogs blue?");
    println!("Yes");
    serial_println!("[ok]");
}
