//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/DeviceRuntimeApi.hh
//! \brief Include CUDA or HIP runtime APIs for compiling with host/cc
//! compiler.
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#if CELERITAS_USE_CUDA
#    include <cuda_runtime_api.h>
#elif CELERITAS_USE_HIP
#    ifndef __HIPCC__
/* Assume we're on an AMD system but not being invoked by the rocm compiler */
#        define __HIP_PLATFORM_AMD__ 1
#        define __HIP_PLATFORM_HCC__ 1
#    endif
#    include <hip/hip_runtime.h>
#endif

/*!
 * \def CELER_DEVICE_PLATFORM
 *
 * API prefix token for the device offloading type.
 */
/*!
 * \def CELER_DEVICE_API_SYMBOL
 *
 * Add a prefix "hip" or "cuda" to a code token.
 */
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
#    define CELER_DEVICE_PLATFORM cuda
#    define CELER_DEVICE_PLATFORM_UPPER_STR "CUDA"
#    define CELER_DEVICE_API_SYMBOL(TOK) cuda##TOK
#    define CELER_EU_PER_CU 1
#elif CELERITAS_USE_HIP
#    define CELER_DEVICE_PLATFORM hip
#    define CELER_DEVICE_PLATFORM_UPPER_STR "HIP"
#    define CELER_DEVICE_API_SYMBOL(TOK) hip##TOK
#    if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#        define CELER_EU_PER_CU 4
#    elif defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_NVCC__)
#        define CELER_EU_PER_CU 1
#    else
#        define CELER_EU_PER_CU 0
#    endif
#else
#    define CELER_DEVICE_PLATFORM none
#    define CELER_DEVICE_PLATFORM_UPPER_STR ""
#    define CELER_DEVICE_API_SYMBOL(TOK) void
#    define CELER_EU_PER_CU 0
#endif

/*!
 * This macro informs downstream Celeritas code (namely, Stream) that it's safe
 * to use types from the device APIs.
 */
#define CELER_DEVICE_RUNTIME_INCLUDED

/*!
 * Declare a dummy variable for disabled \c CELER_DEVICE_API_CALL calls.
 *
 * With this declaration, the build will fail if this include is missing.
 * (Unfortunately, since the use of this symbol is embedded in a macro, IWYU
 * won't include this file automatically.)
 */
extern int const CorecelDeviceRuntimeApiHh;
