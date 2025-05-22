!/bin/bash

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

$cache_utils/../target/release/numa_calibration_3000 > log.txt 2> err.txt

sudo-g5k sh -c "echo 1 > /proc/sys/kernel/numa_balancing"

xz *.txt


popd

mkdir -p ./variable_freq_3000
cp /tmp/numa_cal_variable/*.xz /tmp/numa_cal_variable/*.zst ./variable_freq_3000/

rm -Rf /tmp/numa_cal_variable

mkdir -p /tmp/numa_cal_fixed
pushd /tmp/numa_cal_fixed

# MSR TBD, according to the Architecture manual
# MSR C000_0108h, controls enabling / disabling hardware
# prefetchers. See the appropriate BIOS and
# Kernel Developer’s Guide or Processor
# Programming Reference Manual for details.

# For family 1A, 57238_C1_pub_1.pdf documents MSRC000_0108 [Prefetch Control]
# Bit 5, 3, 2, 1 and 0 should be set to 1 to disable prefetchers.
sudo-g5k rdmsr    0xC0000108
sudo-g5k wrmsr -a 0xC0000108 0x2f

sudo cpupower frequency-set -g performance
#sudo sh -c "echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo"
sudo-g5k sh -c "echo 0 > /sys/devices/system/cpu/cpufreq/boost"
sudo-g5k sh -c "echo 0 > /proc/sys/kernel/numa_balancing"

$cache_utils/../target/release/numa_calibration_3000 > log.txt 2> err.txt

#sudo sh -c "echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo"
sudo-g5k sh -c "echo 1 > /sys/devices/system/cpu/cpufreq/boost"
sudo-g5k sh -c "echo 1 > /proc/sys/kernel/numa_balancing"
sudo-g5k wrmsr -a 0xC0000108 0x0

sudo-g5k rdmsr 0xC0000108
#3c0


xz *.txt

popd


mkdir -p ./fixed_freq_3000
cp /tmp/numa_cal_fixed/*.xz /tmp/numa_cal_fixed/*.zst ./fixed_freq_3000/
rm -Rf /tmp/numa_cal_fixed
