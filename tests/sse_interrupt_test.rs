/*
 *  This test is meant to check that floating point registers do not get clobbered by interrupts.
 *  Wires int3 to an interrupt handler that clobbers fp registers.
 *  And then execute a floating point code to detect if registers got clobbered.
 */

// TODO

#![no_std]
#![no_main]
#![feature(custom_test_frameworks)]
#![test_runner(dendrobates_tinctoreus_azureus::test_runner)]
#![reexport_test_harness_main = "test_main"]
#![feature(abi_x86_interrupt)]

use core::panic::PanicInfo;
use lazy_static::lazy_static;
use polling_serial::{serial_print, serial_println};
use vga_buffer::{print, println};
use volatile::Volatile;
use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame};

use bootloader::{entry_point, BootInfo};

entry_point!(test_kernel_main);

/// Entry point for `cargo xtest`

fn test_kernel_main(_boot_info: &'static BootInfo) -> ! {
    dendrobates_tinctoreus_azureus::init();
    test_main();

    loop {}
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    dendrobates_tinctoreus_azureus::test_panic_handler(info)
}

lazy_static! {
    static ref IDT: InterruptDescriptorTable = {
        let mut idt = InterruptDescriptorTable::new();
        idt.breakpoint.set_handler_fn(breakpoint_handler);
        idt
    };
}

// For now.
extern "x86-interrupt" fn breakpoint_handler(stack_frame: &mut InterruptStackFrame) {
    println!("EXCEPTION: BREAKPOINT\n{:#?}", stack_frame);
    let e: Volatile<f32> = Volatile::new(15.213);
    serial_println!("e: {:?}", e.read());
}

#[test_case]
fn test_intr() {
    serial_println!("Testing float computations with int3...");
    IDT.load();
    use volatile::Volatile;
    // Make a few floating points test;
    let vf: f32 = 84798.0;
    let vd: f64 = 0.828494623655914;

    let a: Volatile<f32> = Volatile::new(42.0);
    let b: Volatile<f32> = Volatile::new(2019.);

    let a1 = a.read();
    let b1 = b.read();
    x86_64::instructions::interrupts::int3(); // new
    let rf = a1 * b1;

    let c: Volatile<f64> = Volatile::new(15.410);
    let d: Volatile<f64> = Volatile::new(18.600);

    let c1 = c.read();
    let d1 = d.read();

    x86_64::instructions::interrupts::int3(); // new

    let rd = c1 / d1;

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
