>&2 echo "# Running the following commands with sudo to set-up"
>&2 echo 'sudo sh -c "echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo"'
>&2 echo sudo cpupower frequency-set -g performance

# performance cpu frequency governor
sudo cpupower frequency-set -g performance

# No Turbo Boost
sudo sh -c "echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo"

cargo run --release --bin two_thread_cal "$@"

>&2 echo "# Please run the following commands to restore configuration"
>&2 echo 'sudo sh -c "echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo"'
>&2 echo sudo cpupower frequency-set -g powersave
