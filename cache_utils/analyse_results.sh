#!/bin/bash
# Assumes the input file has no extension and isn't bzip compressed
# Automatically runs the different analysis scripts

exit_echo () {
  echo $@;
  exit 1;
}

[[ $1 ]] || exit_echo "Usage: $0 <file>";
[[ -f $1.slices.csv ]] || exit_echo "$1.slices.csv not found"
[[ -f $1.cores.csv ]] || exit_echo "$1.cores.csv not found"

bash 2T/analyse.sh $1

. venv/bin/activate
python analyse_csv.py $1
python analyse_medians.py $1
