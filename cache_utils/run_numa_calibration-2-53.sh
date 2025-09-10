#!/bin/bash

SUDO=sudo

echo "$0"
abs_self=`realpath "$0"`
echo $abs_self
cache_utils=`dirname "$abs_self"`
echo $cache_utils

#pushd $cache_utils
#cargo build --release --bin numa_calibration
#popd

$SUDO apt install msr-tools
$SUDO modprobe msr

lstopo --of xml > topo.xml
lscpu > cpu.txt


mkdir -p /tmp/numa_cal_variable
pushd /tmp/numa_cal_variable

$SUDO sh -c "echo 0 > /proc/sys/kernel/numa_balancing"

$cache_utils/../target/release/numa_calibration > log.txt 2> err.txt

$SUDO sh -c "echo 1 > /proc/sys/kernel/numa_balancing"

xz *.txt

popd

mkdir -p ./variable_freq
cp /tmp/numa_cal_variable/*.xz ./variable_freq/
rm -Rf /tmp/numa_cal_variable

mkdir -p /tmp/numa_cal_fixed
pushd /tmp/numa_cal_fixed

$SUDO wrmsr -a 420 0xff

$SUDO cpupower frequency-set -g performance
$SUDO sh -c "echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo"
$SUDO sh -c "echo 0 > /proc/sys/kernel/numa_balancing"

$cache_utils/../target/release/numa_calibration > log.txt 2> err.txt

$SUDO sh -c "echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo"
$SUDO sh -c "echo 1 > /proc/sys/kernel/numa_balancing"
# restore the original configuration
$SUDO wrmsr -a 420 0x00

xz *.txt

popd


mkdir -p ./fixed_freq
cp /tmp/numa_cal_fixed/*.xz ./fixed_freq/
rm -Rf /tmp/numa_cal_fixed

