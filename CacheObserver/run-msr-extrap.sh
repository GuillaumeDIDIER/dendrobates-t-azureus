#!/bin/bash
PREFETCH_MSR=$1
sudo wrmsr -a 0x1a4 $PREFETCH_MSR
sudo echo wrmsr -a 0x1a4 $PREFETCH_MSR
sudo rdmsr -a 0x1a4
cargo run --release --bin extra_access_pattern > extrap-with-${PREFETCH_MSR}-prefetcher.log
sudo rdmsr -a 0x1a4

