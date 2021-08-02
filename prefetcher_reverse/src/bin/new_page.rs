use prefetcher_reverse::{Prober, PAGE_CACHELINE_LEN};

pub const NUM_ITERATION: usize = 1 << 10;

fn exp(delay: u64) {
    let mut prober = Prober::<2>::new(63).unwrap();
    prober.set_delay(delay);
    let pattern = (0usize..(PAGE_CACHELINE_LEN * 2usize)).collect::<Vec<usize>>();
    let result = prober.full_page_probe(pattern, NUM_ITERATION as u32, 100);
    println!("{}", result);
}

fn main() {
    for delay in [0, 5, 10, 50] {
        println!("Delay after each access: {} us", delay);
        exp(delay);
    }
}
