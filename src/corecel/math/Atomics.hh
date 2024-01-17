//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/Atomics.hh
//! \brief Atomics for use in kernel code (CUDA/HIP/OpenMP).
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"

#include "Algorithms.hh"

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
#    ifdef _OPENMP
#        pragma omp atomic capture
#    endif
    {
        initial = *address;
        *address += value;
    }
    return initial;
#endif
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
//---------------------------------------------------------------------------//
/*!
 * Atomic addition specialization for double-precision on older platforms.
 *
 * From CUDA C Programming guide v10.1 p127
 */
inline __device__ double atomic_add(double* address, double val)
{
    CELER_EXPECT(address);
    ull_int* address_as_ull = reinterpret_cast<ull_int*>(address);
    ull_int old = *address_as_ull;
    ull_int assumed;
    do
    {
        assumed = old;
        old = atomicCAS(
            address_as_ull,
            assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
        // Note: uses integer comparison to avoid hang in case of NaN (since
        // NaN != NaN)
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

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
#    ifdef _OPENMP
#        pragma omp atomic capture
#    endif
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
#    ifdef _OPENMP
#        pragma omp atomic capture
#    endif
    {
        initial = *address;
        *address = celeritas::max(initial, value);
    }
    return initial;
#endif
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ <= 300)
//---------------------------------------------------------------------------//
/*!
 * Software emulation of atomic max for older systems.
 *
 * This is a modification of the "software double-precision add" algorithm.
 * TODO: combine this algorithm with the atomic_add and genericize on operation
 * if we ever need to implement the atomics for other types.
 */
inline __device__ ull_int atomic_max(ull_int* address, ull_int val)
{
    CELER_EXPECT(address);
    ull_int old = *address;
    ull_int assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address, assumed, celeritas::max(val, assumed));
    } while (assumed != old);
    return old;
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
