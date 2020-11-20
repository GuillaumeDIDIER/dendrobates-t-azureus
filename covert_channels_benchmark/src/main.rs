#![feature(unsafe_block_in_unsafe_fn)]
#![deny(unsafe_op_in_unsafe_fn)]

use covert_channels_evaluation::benchmark_channel;
use flush_flush::FlushAndFlush;

fn main() {
    for _ in 0..16 {
        //let sender = 0;
        //let receiver = 2;
        let (channel, old, receiver, sender) = match FlushAndFlush::new_any_two_core(true) {
            Err(e) => {
                panic!("{:?}", e);
            }
            Ok(r) => r,
        };

        let r = benchmark_channel(channel, 1, 1 << 15);
        println!("{:?}", r);
    }
}
