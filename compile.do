#!/usr/bin/bash

gcc -Wall -o precalculate_table source/precalculate_table.c source/elliptic_integral.c -lm
./precalculate_table

python3 setup.py build_ext --inplace
