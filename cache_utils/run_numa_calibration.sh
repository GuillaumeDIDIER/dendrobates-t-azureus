#!/bin/bash

echo "$0"
abs_self=`realpath "$0"`
echo $abs_self
cache_utils=`dirname "$abs_self"`
echo $cache_utils

#pushd $cache_utils
#cargo build --release --bin numa_calibration
#popd

mkdir -p /tmp/numa_cal
pushd /tmp/numa_cal

sudo-g5k cpupower frequency-set -g performance

sudo-g5k sh -c "echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo"

$cache_utils/../target/release/numa_calibration > log.txt 2> err.txt
xz *.txt

popd

cp *.xz ./
rm -Rf /tmp/numa_cal


