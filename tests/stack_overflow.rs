// in tests/stack_overflow.rs

#![no_std]
#![no_main]
#![feature(abi_x86_interrupt)]
#![feature(asm)]

use core::panic::PanicInfo;
use dendrobates_tinctoreus_azureus::{exit_qemu, QemuExitCode};
use polling_serial::{serial_print, serial_println};

#[no_mangle]
pub extern "C" fn _start() -> ! {
    serial_print!("stack_overflow... ");

    dendrobates_tinctoreus_azureus::gdt::init();
    init_test_idt();

    // trigger a stack overflow
    stack_overflow(0);

    panic!("Execution continued after stack overflow");
}

#[allow(unconditional_recursion)]
fn stack_overflow(i: u64) -> u64 {
    let a = stack_overflow(i + 1); // for each recursion, the return address is pushed
    a + 1
}
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    dendrobates_tinctoreus_azureus::test_panic_handler(info)
}

use lazy_static::lazy_static;
use x86_64::structures::idt::InterruptDescriptorTable;

lazy_static! {
    static ref TEST_IDT: InterruptDescriptorTable = {
        let mut idt = InterruptDescriptorTable::new();
        unsafe {
            idt.double_fault
                .set_handler_fn(test_double_fault_handler)
                .set_stack_index(dendrobates_tinctoreus_azureus::gdt::DOUBLE_FAULT_IST_INDEX);
        }

        idt
    };
}

pub fn init_test_idt() {
    TEST_IDT.load();
}

use x86_64::structures::idt::InterruptStackFrame;

extern "x86-interrupt" fn test_double_fault_handler(sf: &mut InterruptStackFrame, e: u64) {
    // LLVM bug causing misaligned stacks when error codes are present.
    // This code realigns the stack and then grabs the correct values by doing some pointer arithmetic
    let _stack_frame: &mut InterruptStackFrame;
    let _error_code: u64;

    unsafe {
        asm!("push rax" :::: "intel");
        let s = sf as *mut InterruptStackFrame;
        _stack_frame = &mut *((s as *mut u64).offset(1) as *mut InterruptStackFrame);
        _error_code = *(&e as *const u64).offset(1);
    }
    // End Hack
    serial_println!("[ok]");
    exit_qemu(QemuExitCode::Success);
}
