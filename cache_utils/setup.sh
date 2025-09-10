#!/bin/bash

SUDO=sudo-g5k

$SUDO apt-get update
$SUDO apt-get install libnuma-dev curl git bzip2 gcc hwloc htop libclang-dev libcpufreq-dev -y
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain none -y
bash -c 'source "$HOME/.cargo/env" && rustup toolchain install nightly --profile complete --component cargo && rustup default nightly'
source $HOME/.cargo/env
cd ~/dendrobates-t-azureus/cache_utils/
cargo build --release --bin numa_calibration
cargo build --release --bin numa_calibration_1500
cargo build --release --bin numa_calibration_3000
cd ../covert_channels_benchmark
cargo build --release --bin covert_channels_benchmark
cargo build --release --bin covert_channels_benchmark_1500
cargo build --release --bin covert_channels_benchmark_3000
