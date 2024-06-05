#!/bin/bash
# Automatically run the different analysis scripts

exit_echo () {
  echo $@;
  exit 1;
}

NAME="${1%%.*}"

[[ $1 ]] || exit_echo "Usage: $0 <file>";
[[ -f $NAME.slices.csv ]] || exit_echo "$NAME.slices.csv not found"
[[ -f $NAME.cores.csv ]] || exit_echo "$NAME.cores.csv not found"

bash 2T/analyse.sh $1

. venv/bin/activate
python analyse_csv.py $NAME
python analyse_medians.py $NAME
