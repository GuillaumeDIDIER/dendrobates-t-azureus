#![no_std]
#![no_main]

use core::panic::PanicInfo;
use dendrobates_tinctoreus_azureus::{exit_qemu, QemuExitCode};
use polling_serial::{serial_print, serial_println};

use bootloader::{entry_point, BootInfo};

entry_point!(test_kernel_main);

/// Entry point for `cargo xtest`

fn test_kernel_main(_boot_info: &'static BootInfo) -> ! {
    dendrobates_tinctoreus_azureus::init();
    should_fail();
    serial_println!("[test did not panic]");
    exit_qemu(QemuExitCode::Failed);
}

fn should_fail() {
    serial_print!("should_fail... ");
    assert_eq!(0, 1);
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    serial_println!("[ok]");
    exit_qemu(QemuExitCode::Success);
}
