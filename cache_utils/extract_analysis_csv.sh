#!/bin/sh
grep '^Analysis:' "$1.txt" | cut -b 10- > "$1.csv"
grep '^AVAnalysis:' "$1.txt" | cut -b 12- > "$1.AV.csv"
grep '^AttackerAnalysis:' "$1.txt" | cut -b 18- > "$1.A.csv"
