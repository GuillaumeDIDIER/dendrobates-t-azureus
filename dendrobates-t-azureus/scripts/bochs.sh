#!/bin/sh
set -e
#bootimage build
dd if=/dev/zero of=target/x86_64-D.TinctoriusAzureus/debug/bootimage-kernel.iso count=1008 bs=512
dd if=$1 of=target/x86_64-D.TinctoriusAzureus/debug/bootimage-kernel.iso conv=notrunc
./scripts/syms.sh target/x86_64-D.TinctoriusAzureus/debug/dendrobates_tinctoreus_azureus
bochs -f scripts/bochsrc
