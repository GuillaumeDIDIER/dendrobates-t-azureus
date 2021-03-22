use cache_utils::calibration::PAGE_LEN;

pub const CACHE_LINE_LEN: usize = 64;

fn max_stride(len: usize) -> isize {
    if len == 0 {
        1
    } else {
        (PAGE_LEN / (len * CACHE_LINE_LEN)) as isize
    }
}

fn generate_pattern(len: usize, stride: isize) -> Vec<isize> {
    if (stride * len as isize * CACHE_LINE_LEN as isize).abs() as usize > PAGE_LEN {
        panic!("This is illegal");
    }
    let mut res = Vec::with_capacity(len);
    for i in 0..len {
        res.push(i as isize * stride * CACHE_LINE_LEN as isize);
    }
    res
}

fn main() {
    /*
    TODO List :
    Calibration & core selection (select one or two cores with optimal error)
    Then allocate a bunch of pages, and do accesses on each of them.

    (Let's start with stride patterns : for len in 0..16, and then for stride in 1..maxs_stride(len), generate a vac of addresses and get the victim to execute, then dump all the page)

     */

    println!("Hello, world!");
    println!("{:?}", generate_pattern(5, 2));
    println!("{:?}", generate_pattern(5, 1));
    println!("{:?}", generate_pattern(0, 1));
    println!("{:?}", generate_pattern(5, 5));
    println!("{:?}", generate_pattern(16, 16));
}
