/*-----------------------------------*-C-*-----------------------------------
 * Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
 * See the top-level COPYRIGHT file for details.
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *---------------------------------------------------------------------------*/
/*!
 * \file corecel/device_runtime_api.h
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

#if CELERITAS_USE_CUDA || CELERITAS_USE_HIP
#    include <thrust/mr/memory_resource.h>
#endif

/*!
 * \def CELER_EU_PER_CU
 *
 * Execution units per compute unit.  AMD multiprocessors each have 4 SIMD
 * units per compute unit, but there is no device attribute or
 * compiler definition that provides this information.
 * For CUDA, each streaming multiprocessor (MP) is a single "execution unit"
 * and a "compute unit".
 */
#if CELERITAS_USE_CUDA
#    define CELER_EU_PER_CU 1
#elif CELERITAS_USE_HIP
#    if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#        define CELER_EU_PER_CU 4
#    elif defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_NVCC__)
#        define CELER_EU_PER_CU 1
#    else
#        warning "Unknown HIP device configuration"
#        define CELER_EU_PER_CU 0
#    endif
#else
/* HIP and CUDA are disabled */
#    define CELER_EU_PER_CU 0
#endif

/*!
 * Declare a dummy variable to be referenced in disabled \c CELER_BLAH calls.
 *
 * With this declaration, the build will fail if this include is missing.
 * (Unfortunately, since the use of this symbol is embedded in a macro, IWYU
 * won't include this file automatically.)
 */
extern int celeritas_device_runtime_api_h_;

#endif /* CELERITAS_DEVICE_RUNTIME_API_H */
