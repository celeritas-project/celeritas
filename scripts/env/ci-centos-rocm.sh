#!/bin/sh -e

export PATH=/opt/rocm/llvm/bin:${PATH}
export ASAN_OPTIONS=detect_leaks=0
