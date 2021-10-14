global_asm!(
    ".global timed_maccess_template",
    "timed_maccess_template:",
    "mfence",
    "rdtsc",
    "shl rdx, 32",
    "mov rsi, rdx",
    "add rsi, rax",
    "mfence",
    "mov rdi, [rdi]",
    "mfence",
    "rdtsc",
    "shl rdx, 32",
    "add rax, rdx",
    "mfence",
    "sub rax, rsi",
    "ret",
    ".global timed_maccess_template_end",
    "timed_maccess_template_end:",
    "nop",
    ".global timed_clflush_template",
    "timed_clflush_template:",
    "mfence",
    "rdtsc",
    "shl rdx, 32",
    "mov rsi, rdx",
    "add rsi, rax",
    "mfence",
    "clflush [rdi]",
    "mfence",
    "rdtsc",
    "shl rdx, 32",
    "add rax, rdx",
    "mfence",
    "sub rax, rsi",
    "ret",
    ".global timed_clflush_template_end",
    "timed_clflush_template_end:",
    "nop",
);

extern "C" {
    fn timed_maccess_template(pointer: *const u8) -> u64;
    fn timed_maccess_template_end();
    fn timed_clflush_template(pointer: *const u8) -> u64;
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
