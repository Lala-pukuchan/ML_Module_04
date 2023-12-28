#!/bin/bash

# make results directory
if [ ! -d "results" ]; then
    mkdir results
fi

# make results/ex directory
for i in {0..9}
do
    if [ ! -d "results/ex0$i" ]; then
        mkdir "results/ex0$i"
    fi
done

# get current directory
currentDir=$(pwd)

# Export the new path to PYTHONPATH
export PYTHONPATH="$currentDir:$PYTHONPATH"

# execute test.py
for i in {0..9}
do
    if [[ $i -ne 7 ]]; then
        python3 ex0$i/test.py > results/ex0$i/result.txt
    elif [[ $i -eq 7 ]]; then
        # python3 ex07/benchmark_train.py
        # python3 ex07/space_avocado.py
        continue
    fi
done
