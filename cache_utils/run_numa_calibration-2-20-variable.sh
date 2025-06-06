#!/bin/bash

echo "$0"
abs_self=`realpath "$0"`
echo $abs_self
cache_utils=`dirname "$abs_self"`
echo $cache_utils

#pushd $cache_utils
#cargo build --release --bin numa_calibration
#popd

sudo-g5k apt install msr-tools
sudo-g5k modprobe msr

lstopo --of xml > topo.xml
lscpu > cpu.txt

mkdir -p /tmp/numa_cal_variable
pushd /tmp/numa_cal_variable

sudo-g5k sh -c "echo 0 > /proc/sys/kernel/numa_balancing"

$cache_utils/../target/release/numa_calibration > log.txt 2> err.txt

sudo-g5k sh -c "echo 1 > /proc/sys/kernel/numa_balancing"

xz *.txt


popd

mkdir ./variable_freq
cp /tmp/numa_cal_variable/*.xz /tmp/numa_cal_variable/*.zst ./variable_freq/

rm -Rf /tmp/numa_cal_variable

#mkdir -p /tmp/numa_cal_fixed
#pushd /tmp/numa_cal_fixed

#sudo-g5k wrmsr -a 420 0xf

#sudo-g5k cpupower frequency-set -g performance
#sudo-g5k sh -c "echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo"
#sudo-g5k sh -c "echo 0 > /proc/sys/kernel/numa_balancing"

#$cache_utils/../target/release/numa_calibration > log.txt 2> err.txt

#sudo-g5k sh -c "echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo"
#sudo-g5k sh -c "echo 1 > /proc/sys/kernel/numa_balancing"
#sudo-g5k wrmsr -a 420 0

#xz *.txt

#popd


#mkdir ./fixed_freq
#cp /tmp/numa_cal_fixed/*.xz /tmp/numa_cal_fixed/*.zst ./fixed_freq/
#rm -Rf /tmp/numa_cal_fixed


