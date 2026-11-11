#!/bin/sh -e
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Print a Spack packages config with the CUDA or ROCm toolkit and the GPU
# libraries used by Celeritas, installed in the base image, as non-buildable
# externals with their installed versions. ('spack external find' cannot
# detect the header-only libraries such as thrust, cub, hipcub, and rocprim.)
#-----------------------------------------------------------------------------#

CUDA_ROOT=/usr/local/cuda
ROCM_ROOT=/opt/rocm

# Print a non-buildable external: package name, spec, prefix
external() {
  cat <<EOS
  $1:
    buildable: false
    externals:
      - spec: $2
        prefix: $3
EOS
}

# Print the version of a CCCL library bundled with CUDA: header, macro prefix
cccl_version() {
  tmpdir=$(mktemp -d)
  cat > "${tmpdir}/version.cc" <<EOS
#include <cstdio>
#include <$1>
int main()
{
    std::printf("%d.%d.%d\n", $2_MAJOR_VERSION, $2_MINOR_VERSION,
                $2_SUBMINOR_VERSION);
}
EOS
  c++ -I"${CUDA_ROOT}/include/cccl" -I"${CUDA_ROOT}/include" \
    -o "${tmpdir}/version" "${tmpdir}/version.cc"
  "${tmpdir}/version"
  rm -rf "${tmpdir}"
}

if [ -x "${CUDA_ROOT}/bin/nvcc" ]; then
  # The nvidia/cuda images set CUDA_VERSION (with the patch version);
  # otherwise use the release reported by nvcc
  version=${CUDA_VERSION:-$("${CUDA_ROOT}/bin/nvcc" --version \
    | sed -n 's/.*release \([0-9.]*\),.*/\1/p')}
  thrust_version=$(cccl_version thrust/version.h THRUST)
  cub_version=$(cccl_version cub/version.cuh CUB)
  echo "packages:"
  external cuda "cuda@${version}" "${CUDA_ROOT}"
  external thrust "thrust@${thrust_version}" "${CUDA_ROOT}"
  external cub "cub@${cub_version}" "${CUDA_ROOT}"
elif [ -f "${ROCM_ROOT}/.info/version" ]; then
  # Spack versions the ROCm libraries with the ROCm release
  version=$(cut -d- -f1 "${ROCM_ROOT}/.info/version")
  echo "packages:"
  external hip "hip@${version} +rocm" "${ROCM_ROOT}"
  for pkg in hsa-rocr-dev hipcub hiprand rocprim rocrand rocthrust; do
    external "${pkg}" "${pkg}@${version}" "${ROCM_ROOT}"
  done
  external llvm-amdgpu "llvm-amdgpu@${version}" "${ROCM_ROOT}/llvm"
  cat <<EOS
        extra_attributes:
          compilers:
            c: ${ROCM_ROOT}/llvm/bin/clang
            cxx: ${ROCM_ROOT}/llvm/bin/clang++
EOS
else
  echo "error: no CUDA (${CUDA_ROOT}) or ROCm (${ROCM_ROOT}) installation found" >&2
  exit 1
fi
