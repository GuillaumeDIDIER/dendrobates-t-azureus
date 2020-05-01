#!/bin/sh
awk '/^Iteration [:digit:]*[.]*/ ' < log.txt > iterations.txt
rm results.csv
awk -f `dirname $0`/analyse_iterations.awk < iterations.txt # This uses system to split off awk scripts doing the analysis
grep -v -e "0,0$" results.csv > results_lite.csv
#paste -d"," *.csv > combined.csv
