global_asm!(
    ".global timed_maccess_template",
    "timed_maccess_template:",
    "nop",
    "ret",
    ".global timed_maccess_template_end",
    "timed_maccess_template_end:",
    "nop",
);

extern "C" {
    fn timed_maccess_template(pointer: *const u8) -> u64;
    fn timed_maccess_template_end();
}

pub fn tmp_test() {
    let size = timed_maccess_template_end as *const u8 as usize
        - timed_maccess_template as *const u8 as usize;
    println!("function size : {}", size);
}
