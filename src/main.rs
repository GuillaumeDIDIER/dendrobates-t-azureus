// main.rs
// main file of the kernel

#![no_std] // This is a free standing program
#![no_main] // This has no crt0

use core::fmt::Write;
use core::panic::PanicInfo; // required for custom panic handler

use x86_64;

use vga_buffer;

// Custom panic handler, required for freestanding program
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

// static greeting string, for hello world kernel
static HELLO: &[u8] = b"Hello Blue Frog!";

static YES: &[u8] = b"yes";
static NO: &[u8] = b"no";

static a: f64 = 420.0;
static b: f64 = 42.0;

static d: f64 = 0.1;

// Kernel entry point
#[no_mangle]
pub extern "C" fn _start() -> ! {
    // TODO: Take care of cpuid stuff and set-up all floating point exetnsions
    // TODO: We may also need to enable debug registers ?

    let vga_buffer = 0xb8000 as *mut u8;

    for (i, &byte) in HELLO.iter().enumerate() {
        unsafe {
            *vga_buffer.offset(i as isize * 2) = byte;
            *vga_buffer.offset(i as isize * 2 + 1) = 0xb;
        }
    }
    // magic break ?
    x86_64::instructions::bochs_breakpoint();
    let c = a * d;
    x86_64::instructions::bochs_breakpoint();
    if b == c {
        for (i, &byte) in YES.iter().enumerate() {
            unsafe {
                *vga_buffer.offset(i as isize * 2) = byte;
                *vga_buffer.offset(i as isize * 2 + 1) = 0xb;
            }
        }
    } else {
        for (i, &byte) in NO.iter().enumerate() {
            unsafe {
                *vga_buffer.offset(i as isize * 2) = byte;
                *vga_buffer.offset(i as isize * 2 + 1) = 0xb;
            }
        }
    }

    x86_64::instructions::bochs_breakpoint();

    writeln!(
        vga_buffer::WRITER.lock(),
        "The numbers are {} and {}",
        42,
        1.0 / 3.0
    )
    .unwrap();

    writeln!(
        vga_buffer::WRITER.lock(),
        "a is {}, b is {}, c is {}, d is {}",
        a,
        b,
        c,
        d
    )
    .unwrap();

    loop {}
}
