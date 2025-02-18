#!/bin/sh

echo "$0"
abs_self=`realpath "$0"`
echo $abs_self
cache_utils=`dirname "$abs_self"`
echo $cache_utils

mkdir -p /tmp/numa_cal
pushd /tmp/numa_cal

echo $cache_utils/../target/release/numa_calibration > log.txt 2> err.txt
xz *.txt

popd

cp *.xz ./
rm -Rf /tmp/numa_cal