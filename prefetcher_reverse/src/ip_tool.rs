use cache_utils::mmap::MMappedMemory;
use lazy_static::lazy_static;
use std::collections::LinkedList;
use std::sync::Mutex;

struct WXRange {
    bitmap: Vec<bool>, // fixme bit vector
    pages: Vec<MMappedMemory<u8>>,
}

struct WXAllocator {
    ranges: LinkedList<WXRange>,
    // Possible improvement : a dedicated data structure, with optimised lookup of which range
    // contains the right address, plus reasonably easy ability to merge nodes
}

impl WXAllocator {
    fn new() -> Self {
        WXAllocator {
            ranges: LinkedList::<WXRange>::new(),
        }
    }
}

pub struct FunctionTemplate {
    start: unsafe extern "C" fn(*const u8) -> u64,
    ip: *const u8,
    end: *const u8,
}
lazy_static! {
    static ref wx_allocator: Mutex<WXAllocator> = Mutex::new(WXAllocator::new());
}
const TIMED_MACCESS: FunctionTemplate = FunctionTemplate {
    start: timed_maccess_template,
    ip: timed_maccess_template_ip as *const u8,
    end: timed_maccess_template_end as *const u8,
};

const TIMED_CLFLUSH: FunctionTemplate = FunctionTemplate {
    start: timed_clflush_template,
    ip: timed_clflush_template_ip as *const u8,
    end: timed_clflush_template_end as *const u8,
};

global_asm!(
    ".global timed_maccess_template",
    "timed_maccess_template:",
    "mfence",
    "lfence",
    "rdtsc",
    "shl rdx, 32",
    "mov rsi, rdx",
    "add rsi, rax",
    "mfence",
    "lfence",
    ".global timed_maccess_template_ip",
    "timed_maccess_template_ip:",
    "mov rdi, [rdi]",
    "mfence",
    "lfence",
    "rdtsc",
    "shl rdx, 32",
    "add rax, rdx",
    "mfence",
    "lfence",
    "sub rax, rsi",
    "ret",
    ".global timed_maccess_template_end",
    "timed_maccess_template_end:",
    "nop",
    ".global timed_clflush_template",
    "timed_clflush_template:",
    "mfence",
    "lfence",
    "rdtsc",
    "shl rdx, 32",
    "mov rsi, rdx",
    "add rsi, rax",
    "mfence",
    "lfence",
    ".global timed_clflush_template_ip",
    "timed_clflush_template_ip:",
    "clflush [rdi]",
    "mfence",
    "lfence",
    "rdtsc",
    "shl rdx, 32",
    "add rax, rdx",
    "mfence",
    "lfence",
    "sub rax, rsi",
    "ret",
    ".global timed_clflush_template_end",
    "timed_clflush_template_end:",
    "nop",
);

extern "C" {
    fn timed_maccess_template(pointer: *const u8) -> u64;
    fn timed_maccess_template_ip();
    fn timed_maccess_template_end();
    fn timed_clflush_template(pointer: *const u8) -> u64;
    fn timed_clflush_template_ip();
    fn timed_clflush_template_end();
}

pub fn tmp_test() {
    let size = timed_maccess_template_end as *const u8 as usize
        - timed_maccess_template as *const u8 as usize;
    println!("maccess function size : {}", size);
    let size = timed_clflush_template_end as *const u8 as usize
        - timed_clflush_template as *const u8 as usize;
    println!("clflush function size : {}", size);
    let mem: u8 = 42;
    let p = &mem as *const u8;
    println!("maccess {:p} : {}", p, unsafe { timed_maccess_template(p) });
    println!("clflush {:p} : {}", p, unsafe { timed_clflush_template(p) });
}
