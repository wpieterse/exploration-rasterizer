#!/bin/bash

clear

cd build

cmake --build .

sudo cpupower frequency-set --governor performance > /dev/null
taskset 0x1 perf record -F 1000 --call-graph dwarf ./output/exploration/rasterizer/main
sudo cpupower frequency-set --governor powersave > /dev/null

# sde64 -spr -- ./output/exploration/rasterizer/main

cd ..
