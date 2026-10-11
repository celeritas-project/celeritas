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

#include "corecel/Assert.hh"

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
 * This macro informs downstream Celeritas code (namely, Stream and
 * DeviceEvent) that it's safe to use types from the device APIs.
 */
#define CELER_DEVICE_RUNTIME_INCLUDED

/*!
 * \def CELER_DEVICE_API_CALL
 *
 * Safely and portably dispatch a CUDA/HIP API call.
 *
 * When CUDA or HIP support is enabled, execute the wrapped statement
 * prepend the argument with "cuda" or "hip" and throw a
 * RuntimeError if it fails. If no device platform is enabled, throw an
 * unconfigured assertion.
 *
 * \par Example:
 * \code
   CELER_DEVICE_API_CALL(Malloc(&ptr_gpu, 100 * sizeof(float)));
   CELER_DEVICE_API_CALL(DeviceSynchronize());
 * \endcode
 */
#if CELERITAS_USE_CUDA || CELERITAS_USE_HIP
#    define CELER_DEVICE_API_CALL(STMT) \
        do \
        { \
            using ErrT_ = CELER_DEVICE_API_SYMBOL(Error_t); \
            ErrT_ result_ = CELER_DEVICE_API_SYMBOL(STMT); \
            if (CELER_UNLIKELY(result_ != CELER_DEVICE_API_SYMBOL(Success))) \
            { \
                result_ = CELER_DEVICE_API_SYMBOL(GetLastError)(); \
                CELER_RUNTIME_FAIL_( \
                    CELER_DEVICE_PLATFORM_UPPER_STR, \
                    CELER_DEVICE_API_SYMBOL(GetErrorString)(result_), \
                    #STMT); \
            } \
        } while (0)
#else
#    define CELER_DEVICE_API_CALL(STMT) \
        do \
        { \
            CELER_NOT_CONFIGURED("CUDA or HIP"); \
        } while (0)
#endif

/*!
 * \def CELER_USE_THRUST
 * \def CELER_CUB_HAS_TRANSFORM
 * \def CELER_CUB_HAS_FLAGGEDIF
 * \def CELER_HIPCUB_HAS_TRANSFORM
 *
 * Determine if CUB or hipCUB is available, and if so, check the version.
 *
 * CUDA has included CUB since CUDA 11, but ROCm does not include hipCUB by
 * default, so test for the availability of hipCUB and use thrust instead if
 * it's unavailable.
 *
 * DeviceTransform::Transform is unavailable in earlier versions of CUB/hipCUB,
 * so the code uses thrust::transform in that case.
 *
 * DeviceSelect::FlaggedIf is unavailable in earlier versions of CUB and
 * doesn't work with hipCUB versions 3.4.0 through 4.1.0 when using celeritas
 * OpaqueId data types, so use a transform and DeviceSelect::Flagged instead.
 *
 * \note These are unneeded and undefined if this isn't a HIP or CUDA source
 * file. This check is necessary because the version header file needs to be
 * included.
 */
#if CELER_DEVICE_SOURCE
#    if CELERITAS_USE_CUDA
#        include <cub/version.cuh>
#    elif CELERITAS_USE_HIP && CELERITAS_HAVE_HIPCUB
#        include <hipcub/hipcub_version.hpp>
#    endif
#    if CELERITAS_USE_HIP && !CELERITAS_HAVE_HIPCUB
#        define CELER_USE_THRUST 1
#    endif
#    if CELERITAS_USE_CUDA && CUB_VERSION >= 200800
#        define CELER_CUB_HAS_TRANSFORM 1
#        define CELER_CUB_HAS_FLAGGEDIF 1
#    elif CELERITAS_USE_CUDA && CUB_VERSION >= 200500
#        define CELER_CUB_HAS_FLAGGEDIF 1
#    elif CELERITAS_USE_HIP && HIPCUB_VERSION >= 400100
#        define CELER_HIPCUB_HAS_TRANSFORM 1
#    endif
#else
#    define CELER_USE_THRUST 0
#    define CELER_CUB_HAS_TRANSFORM 0
#    define CELER_CUB_HAS_FLAGGEDIF 0
#    define CELER_HIPCUB_HAS_TRANSFORM 0
#endif
