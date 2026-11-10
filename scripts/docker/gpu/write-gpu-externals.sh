#!/bin/sh -e
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Print a Spack packages config with the CUDA or ROCm toolkit installed in the
# base image as a non-buildable external, using its installed version.
#-----------------------------------------------------------------------------#

CUDA_ROOT=/usr/local/cuda
ROCM_ROOT=/opt/rocm

if [ -f "${CUDA_ROOT}/version.json" ]; then
  version=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["cuda"]["version"])' \
    "${CUDA_ROOT}/version.json")
  cat <<EOS
packages:
  cuda:
    buildable: false
    externals:
      - spec: cuda@${version}
        prefix: ${CUDA_ROOT}
EOS
elif [ -f "${ROCM_ROOT}/.info/version" ]; then
  version=$(cut -d- -f1 "${ROCM_ROOT}/.info/version")
  cat <<EOS
packages:
  hip:
    buildable: false
    externals:
      - spec: hip@${version} +rocm
        prefix: ${ROCM_ROOT}
  hsa-rocr-dev:
    buildable: false
    externals:
      - spec: hsa-rocr-dev@${version}
        prefix: ${ROCM_ROOT}
  llvm-amdgpu:
    buildable: false
    externals:
      - spec: llvm-amdgpu@${version}
        prefix: ${ROCM_ROOT}/llvm
        extra_attributes:
          compilers:
            c: ${ROCM_ROOT}/llvm/bin/clang
            cxx: ${ROCM_ROOT}/llvm/bin/clang++
EOS
else
  echo "error: no CUDA (${CUDA_ROOT}) or ROCm (${ROCM_ROOT}) installation found" >&2
  exit 1
fi
