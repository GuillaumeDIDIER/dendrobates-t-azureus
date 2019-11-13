// A quick and dirty crate for serial I/O inspired by rust-os-dev/uart and code in
// gamozolab/orange_slice
//
// Supports Write but also polling which may be useful
// Note : Swicthing to interrupt driven read could be interesting but should not be mandatory
#![no_std]

extern crate x86_64;

use core::fmt;
use lazy_static::lazy_static;
use spin::Mutex;
use x86_64::instructions::port::Port;

pub struct SerialPort {
    /// Data register.
    /// Reading this registers read from the Receive buffer.
    /// Writing to this register writes to the Transmit buffer.
    data: Port<u8>, // 0

    /// Interrupt Enable Register.
    int_en: Port<u8>, // 1

    /// Interrupt Identification and FIFO control registers
    fifo_ctrl: Port<u8>, // 2

    /// Line Control Register. The most significant bit of this register is the DLAB.
    line_ctrl: Port<u8>, // 3

    /// Modem Control Register.
    modem_ctrl: Port<u8>, // 4

    /// Line Status Register.
    line_status: Port<u8>, // 5

    /// Modem Status Register.
    //modem_status: Port<u8>, // 6

    /// Scratch Register.
    scratch: Port<u8>, // 7
}

const SCRATCH_VALUE: u8 = 0x42;

const DISBALE_INTERRUPTS: u8 = 0x00;

const DLAB: u8 = 0x80;

const DIVISOR: u16 = 0x03;

const PARITY_MODE: u8 = 0x03;

const FIFO14: u8 = 0xC7;

const RTS_BST: u8 = 0x0B;

const READY_TO_SEND: u8 = 0x20;

const READY_TO_READ: u8 = 0x01;

impl SerialPort {
    /// Tries to create and initialize a serial port on the given base port
    /// Will return None if the serial port doesn't work
    /// Unsafe as calling this on an arbitrary port that's not serial port has serious consequences
    pub unsafe fn init_new(base: u16) -> Option<SerialPort> {
        let mut p = SerialPort {
            data: Port::new(base),
            int_en: Port::new(base + 1),
            fifo_ctrl: Port::new(base + 2),
            line_ctrl: Port::new(base + 3),
            modem_ctrl: Port::new(base + 4),
            line_status: Port::new(base + 5),
            //modem_status: Port::new(base + 6),
            scratch: Port::new(base + 7),
        };

        // scratchpad test
        p.scratch.write(SCRATCH_VALUE);
        if p.scratch.read() != SCRATCH_VALUE {
            return None;
        }

        // disable all interrupts
        p.int_en.write(DISBALE_INTERRUPTS);

        // enable DLAB to set the divisor
        p.line_ctrl.write(DLAB);

        // set the divisor hi and lo
        p.data.write(DIVISOR as u8);
        p.int_en.write((DIVISOR >> 8) as u8);

        // clear DLAB
        // set mode to 8 bits No parity 1 stop bit (8-N-1)
        p.line_ctrl.write(PARITY_MODE);

        // Set-up FIFOs depth 14 just in case
        p.fifo_ctrl.write(FIFO14);

        // Set up RTS DSR
        p.modem_ctrl.write(RTS_BST);

        Some(p)
    }

    pub fn send(&mut self, byte: u8) {
        unsafe {
            while self.line_status.read() & READY_TO_SEND == 0 {}
            self.data.write(byte);
        }
    }

    pub fn try_read(&mut self) -> Option<u8> {
        unsafe {
            if self.line_status.read() & READY_TO_READ == 0 {
                None
            } else {
                Some(self.data.read())
            }
        }
    }

    pub fn read(&mut self) -> u8 {
        unsafe {
            while self.line_status.read() & READY_TO_READ == 0 {}
            self.data.read()
        }
    }
}

impl fmt::Write for SerialPort {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for byte in s.bytes() {
            self.send(byte);
        }
        Ok(())
    }
}

lazy_static! {
    pub static ref SERIAL1: Mutex<SerialPort> = {
        let serial_port = unsafe { SerialPort::init_new(0x3F8).unwrap() };
        Mutex::new(serial_port)
    };
}

#[doc(hidden)]
pub fn _print(args: ::core::fmt::Arguments) {
    use core::fmt::Write;
    SERIAL1
        .lock()
        .write_fmt(args)
        .expect("Printing to serial failed");
}

/// Prints to the host through the serial interface.
#[macro_export]
macro_rules! serial_print {
    ($($arg:tt)*) => {
        $crate::_print(format_args!($($arg)*));
    };
}

/// Prints to the host through the serial interface, appending a newline.
#[macro_export]
macro_rules! serial_println {
    () => ($crate::serial_print!("\n"));
    ($fmt:expr) => ($crate::serial_print!(concat!($fmt, "\n")));
    ($fmt:expr, $($arg:tt)*) => ($crate::serial_print!(
        concat!($fmt, "\n"), $($arg)*));
}
