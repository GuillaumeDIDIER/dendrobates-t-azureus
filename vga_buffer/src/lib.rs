#![no_std] // This is a free standing program

use core::fmt;
use core::intrinsics::transmute;
use lazy_static::lazy_static;
use spin::Mutex;
use volatile::Volatile;

pub const BUFFER_HEIGHT: usize = 25;
pub const BUFFER_WIDTH: usize = 80;

const BRIGHT_BIT: u8 = 0x4;
const EMPTY_CHAR: u8 = b' ';
const NEWLINE_CHAR: u8 = b'\n';
const BACKSPACE_CHAR: u8 = 0x8;
const UNPRINTABLE_CHAR: u8 = 0xfe;

const PRINTABLE_RANGE_START: u8 = 0x20;
const PRINTABLE_RANGE_STOP: u8 = 0x7f;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Color {
    Black = 0x0,
    Blue = 0x1,
    Green = 0x2,
    Cyan = 0x3,
    Red = 0x4,
    Magenta = 0x5,
    Brown = 0x6,
    LightGray = 0x7,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ForegroundColor {
    Black = 0x0,
    Blue = 0x1,
    Green = 0x2,
    Cyan = 0x3,
    Red = 0x4,
    Magenta = 0x5,
    Brown = 0x6,
    LightGray = 0x7,
    DarkGray = 0x8,
    LightBlue = 0x9,
    LightGreen = 0xa,
    LightCyan = 0xb,
    LightRed = 0xc,
    Pink = 0xd,
    Yellow = 0xe,
    White = 0xf,
}

impl ForegroundColor {
    pub fn new(color: Color, bright: bool) -> ForegroundColor {
        if bright {
            unsafe { transmute(BRIGHT_BIT | (color as u8)) }
        } else {
            unsafe { transmute(color) }
        }
    }
}

impl From<Color> for ForegroundColor {
    fn from(color: Color) -> Self {
        unsafe { transmute(color) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct ColorCode(u8);

impl ColorCode {
    fn new_with_blink(foreground: ForegroundColor, background: Color, blink: bool) -> ColorCode {
        ColorCode((foreground as u8) | (blink as u8) << 7 | (background as u8) << 4)
    }

    fn new(foreground: ForegroundColor, background: Color) -> ColorCode {
        ColorCode::new_with_blink(foreground, background, false)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct ScreenChar {
    ascii_char: u8,
    color_code: ColorCode,
}

#[repr(transparent)]
struct Buffer {
    chars: [[Volatile<ScreenChar>; BUFFER_WIDTH]; BUFFER_HEIGHT],
}

// The writer could eventually also show the cursor

pub struct Writer {
    column_position: usize,
    row_position: usize,
    color_code: ColorCode,
    buffer: &'static mut Buffer,
}

impl Writer {
    fn new(color_code: ColorCode, buffer: &'static mut Buffer) -> Writer {
        let mut w = Writer {
            column_position: 0,
            row_position: 0,
            color_code,
            buffer,
        };
        w.clear_screen();
        w
    }

    fn clear_screen(&mut self) {
        for i in 0..BUFFER_HEIGHT {
            for j in 0..BUFFER_WIDTH {
                self.draw(
                    i,
                    j,
                    ScreenChar {
                        ascii_char: EMPTY_CHAR,
                        color_code: self.color_code,
                    },
                );
            }
        }
    }

    fn getchar(&self, row: usize, col: usize) -> ScreenChar {
        self.buffer.chars[row][col].read()
    }

    fn draw(&mut self, row: usize, col: usize, sc: ScreenChar) {
        self.buffer.chars[row][col].write(sc);
    }

    fn scroll(&mut self) {
        for i in 0..BUFFER_HEIGHT - 1 {
            for j in 0..BUFFER_WIDTH {
                let sc = self.getchar(i + 1, j);
                self.draw(i, j, sc);
            }
        }
        for j in 0..BUFFER_WIDTH {
            self.draw(
                BUFFER_HEIGHT - 1,
                j,
                ScreenChar {
                    ascii_char: EMPTY_CHAR,
                    color_code: self.color_code,
                },
            );
        }
    }

    fn set_cursor_row(&mut self, row_position: usize) {
        assert!(row_position < BUFFER_HEIGHT);
        self.row_position = row_position;
    }

    fn set_cursor_col(&mut self, column_position: usize) {
        assert!(column_position < BUFFER_WIDTH);
        self.column_position = column_position;
    }

    fn cursor_offset(&mut self, offset: isize) {
        let mut line_pos: isize = offset + self.column_position as isize;
        let mut height_pos: isize = self.row_position as isize;
        while line_pos < 0 {
            height_pos -= 1;
            line_pos += BUFFER_WIDTH as isize;
        }

        while line_pos >= BUFFER_WIDTH as isize {
            height_pos += 1;
            line_pos -= BUFFER_WIDTH as isize;
        }

        while height_pos >= BUFFER_HEIGHT as isize {
            height_pos -= 1;
            self.scroll();
        }

        if height_pos < 0 {
            height_pos = 0;
            line_pos = 0;
        }
        assert!(height_pos >= 0);
        assert!(line_pos >= 0);
        self.set_cursor_col(line_pos as usize);
        self.set_cursor_row(height_pos as usize);
    }

    fn put_byte(&mut self, byte: u8) {
        match byte {
            NEWLINE_CHAR => {
                // Empty line
                for i in self.column_position..BUFFER_WIDTH {
                    self.draw(
                        self.row_position,
                        i,
                        ScreenChar {
                            ascii_char: EMPTY_CHAR,
                            color_code: self.color_code,
                        },
                    );
                }
                self.set_cursor_col(0);
                self.cursor_offset(BUFFER_WIDTH as isize)
            }
            BACKSPACE_CHAR => {
                self.draw(
                    self.row_position,
                    self.column_position,
                    ScreenChar {
                        ascii_char: EMPTY_CHAR,
                        color_code: self.color_code,
                    },
                );
                self.cursor_offset(-1);
            }
            byte => {
                self.draw(
                    self.row_position,
                    self.column_position,
                    ScreenChar {
                        ascii_char: byte,
                        color_code: self.color_code,
                    },
                );
                self.cursor_offset(1);
            }
        }
    }

    fn put_bytes(&mut self, s: &str) {
        for byte in s.bytes() {
            match byte {
                // printable ASCII byte or newline
                PRINTABLE_RANGE_START..=PRINTABLE_RANGE_STOP | NEWLINE_CHAR | BACKSPACE_CHAR => {
                    self.put_byte(byte)
                }
                // not part of printable ASCII range
                _ => self.put_byte(UNPRINTABLE_CHAR),
            }
        }
    }
}

impl fmt::Write for Writer {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.put_bytes(s);
        Ok(())
    }
}

#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => ($crate::_print(format_args!($($arg)*)));
}

#[macro_export]
macro_rules! println {
    () => ($crate::print!("\n"));
    ($($arg:tt)*) => ($crate::print!("{}\n", format_args!($($arg)*)));
}

#[doc(hidden)]
pub fn _print(args: fmt::Arguments) {
    use core::fmt::Write;
    WRITER.lock().write_fmt(args).unwrap();
}

pub fn set_colors(fg: ForegroundColor, bg: Color) {
    WRITER.lock().color_code = ColorCode::new(fg, bg);
}

lazy_static! {
    pub static ref WRITER: Mutex<Writer> = Mutex::new(Writer::new(
        ColorCode::new(ForegroundColor::Yellow, Color::Blue),
        unsafe { &mut *(0xb8000 as *mut Buffer) },
    ));
}
