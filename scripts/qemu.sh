#!/bin/sh
qemu-system-x86_64 -drive format=raw,file="$1" -device isa-debug-exit,iobase=0xf4,iosize=0x04 -serial stdio -display none
