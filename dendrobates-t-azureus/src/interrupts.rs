use lazy_static::lazy_static;
use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame};

use crate::gdt;
use crate::hlt_loop;
use polling_serial::serial_println;
use vga_buffer::println;

lazy_static! {
    static ref IDT: InterruptDescriptorTable = {
        let mut idt = InterruptDescriptorTable::new();
        idt.breakpoint.set_handler_fn(breakpoint_handler);
        idt.double_fault.set_handler_fn(double_fault_handler);
        unsafe {
            idt.double_fault
                .set_handler_fn(double_fault_handler)
                .set_stack_index(gdt::DOUBLE_FAULT_IST_INDEX);
        }
        idt.page_fault.set_handler_fn(page_fault_handler);
        idt
    };
}

pub fn init_idt() {
    IDT.load();
}

// For now.
extern "x86-interrupt" fn breakpoint_handler(stack_frame: InterruptStackFrame) {
    serial_println!("EXCEPTION: BREAKPOINT\n{:#?}", stack_frame);
    println!("EXCEPTION: BREAKPOINT\n{:#?}", stack_frame);
    x86_64::instructions::bochs_breakpoint();
}

extern "x86-interrupt" fn double_fault_handler(mut sf: InterruptStackFrame, e: u64) -> ! {
    // LLVM bug causing misaligned stacks when error codes are present.
    // This code realigns the stack and then grabs the correct values by doing some pointer arithmetic
    let stack_frame: &mut InterruptStackFrame;
    let error_code: u64;

    unsafe {
        llvm_asm!("push rax" :::: "intel");
        let s = (&mut sf) as *mut InterruptStackFrame;
        stack_frame = &mut *((s as *mut u64).offset(1) as *mut InterruptStackFrame);
        error_code = *(&e as *const u64).offset(1);
    }
    // End Hack

    println!(
        "=====\nUNRECOVERABLE EXCEPTION:\nDouble Fault\nError Code {:x?}\n{:#?}\n=====",
        error_code, stack_frame
    );
    serial_println!(
        "=====\nUNRECOVERABLE EXCEPTION:\nDouble Fault\nError Code {:x?}\n{:#?}\n=====",
        error_code,
        stack_frame
    );

    panic!("Unrecoverable exception");
}

use x86_64::instructions::bochs_breakpoint;
use x86_64::structures::idt::PageFaultErrorCode;

extern "x86-interrupt" fn page_fault_handler(mut sf: InterruptStackFrame, e: PageFaultErrorCode) {
    // LLVM bug causing misaligned stacks when error codes are present.
    // This code realigns the stack and then grabs the correct values by doing some pointer arithmetic
    let stack_frame: &mut InterruptStackFrame;
    let error_code: PageFaultErrorCode;

    use x86_64::registers::control::Cr2;

    unsafe {
        llvm_asm!("push rax" :::: "intel");
        let s = (&mut sf) as *mut InterruptStackFrame;
        stack_frame = &mut *((s as *mut u64).offset(1) as *mut InterruptStackFrame);
        error_code = *(&e as *const PageFaultErrorCode).offset(1) as PageFaultErrorCode;
    }

    serial_println!("EXCEPTION: PAGE FAULT");
    serial_println!("Accessed Address: {:?}", Cr2::read());
    serial_println!("Error Code: {:?}", error_code);
    serial_println!("{:#?}", stack_frame);

    serial_println!("Halting...");
    bochs_breakpoint();

    hlt_loop();
}
