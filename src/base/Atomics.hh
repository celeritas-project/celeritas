//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Atomics.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Assert.hh"
#include "Macros.hh"
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
#ifdef __CUDA_ARCH__
    return atomicAdd(address, value);
#else
    REQUIRE(address);
    T initial = *address;
    *address += value;
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
#ifdef __CUDA_ARCH__
    return atomicMin(address, value);
#else
    REQUIRE(address);
    T initial = *address;
    *address  = celeritas::min(initial, value);
    return initial;
#endif
}

//---------------------------------------------------------------------------//
} // namespace celeritas

//---------------------------------------------------------------------------//
