#!/bin/sh
awk '/^Iteration [:digit:]*[.]*/ ' < log.txt > iterations.txt
awk -f `dirname $0`/analyse_iterations.awk < iterations.txt # This uses system to split off awk scripts doing the analysis
paste -d"," *.csv > combined.csv
