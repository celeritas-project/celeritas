//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/*!
 * \file corecel/math/Atomics.hh
 * \brief Atomics for use in kernel code (CUDA/HIP/OpenMP).
 *
 * \note On CPU, these functions assume the atomic add is being done in
 * with \em track-level parallelism rather than \em event-level because these
 * utilities are meant for "kernel" code. Multiple independent events
 * must \em not use these functions to simultaneously modify shared data.
 *
 * ---------------------------------------------------------------------------*/
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"

#include "Algorithms.hh"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
#    error "Celeritas requires CUDA arch 6.0 (P100) or greater"
#endif

#if defined(_OPENMP) && CELERITAS_OPENMP == CELERITAS_OPENMP_TRACK
//! Capture the subsequent expression as an OpenMP atomic
#    define CELER_CAPTURE_IF_OPENMP_TRACK _Pragma("omp atomic capture")
#else
//! Simply scope the next block
#    define CELER_CAPTURE_IF_OPENMP_TRACK
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Add to a value, returning the original value.
 */
template<class T>
CELER_FORCEINLINE_FUNCTION T atomic_add(T* address, T value)
{
#if CELER_DEVICE_COMPILE
    return atomicAdd(address, value);
#else
    CELER_EXPECT(address);
    T initial;
    CELER_CAPTURE_IF_OPENMP_TRACK
    {
        initial = *address;
        *address += value;
    }
    return initial;
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Set the value to the minimum of the actual and given, returning old.
 */
template<class T>
CELER_FORCEINLINE_FUNCTION T atomic_min(T* address, T value)
{
#if CELER_DEVICE_COMPILE
    return atomicMin(address, value);
#else
    CELER_EXPECT(address);
    T initial;
    CELER_CAPTURE_IF_OPENMP_TRACK
    {
        initial = *address;
        *address = celeritas::min(initial, value);
    }
    return initial;
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Set the value to the maximum of the actual and given, returning old.
 */
template<class T>
CELER_FORCEINLINE_FUNCTION T atomic_max(T* address, T value)
{
#if CELER_DEVICE_COMPILE
    return atomicMax(address, value);
#else
    CELER_EXPECT(address);
    T initial;
    CELER_CAPTURE_IF_OPENMP_TRACK
    {
        initial = *address;
        *address = celeritas::max(initial, value);
    }
    return initial;
#endif
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
