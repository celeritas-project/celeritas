/*-----------------------------------*-C-*-----------------------------------
 * Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
 * See the top-level COPYRIGHT file for details.
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *---------------------------------------------------------------------------*/
/*!
 * \file base/device_runtime_api.h
 * Include CUDA or HIP runtime APIs for compiling with host/cc compiler.
 */
#ifndef CELERITAS_DEVICE_RUNTIME_API_H
#define CELERITAS_DEVICE_RUNTIME_API_H

#include "celeritas_config.h"

#if CELERITAS_USE_HIP && !defined(__HIPCC__)
/* Assume we're on an AMD system but not being invoked by the rocm compiler */
#    define __HIP_PLATFORM_AMD__ 1
#    define __HIP_PLATFORM_HCC__ 1
#endif

#if CELERITAS_USE_CUDA
#    include <cuda_runtime_api.h>
#elif CELERITAS_USE_HIP
#    include <hip/hip_runtime.h>
#endif

/*!
 * \def CELER_EU_PER_MP
 *
 * Execution units per multiprocessor.  AMD multiprocessors each have 4 SIMD
 * execution units per multiprocessor, but  there is no device attribute or
 * compiler definition that provides this information.
 * For CUDA, each streaming multiprocessor (MP) is a single "execution unit".
 */
#if CELERITAS_USE_CUDA
#    define CELER_EU_PER_MP 1
#elif CELERITAS_USE_HIP
#    if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#        define CELER_EU_PER_MP 4
#    elif defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_NVCC__)
#        define CELER_EU_PER_MP 1
#    else
#        warning "Unknown HIP device configuration"
#        define CELER_EU_PER_MP 0
#    endif
#else
/* HIP and CUDA are disabled */
#    define CELER_EU_PER_MP 0
#endif

#endif /* CELERITAS_DEVICE_RUNTIME_API_H */
