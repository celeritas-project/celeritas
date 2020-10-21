//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngStatePointers.cuh
//---------------------------------------------------------------------------//
#pragma once

/*!
 * \def QUALIFIERS
 *
 * Define NVCC QUALIFIERS so as to make curand functions work on both host
 * and device. Note that QUALIFIERS defined in curand_kernel.h is only for
 * device functions and re-definition of the macro with the host extension
 * is a limited scope for the random module of celeritas.
 */
#define QUALIFIERS static __forceinline__ __host__ __device__

#include <curand_kernel.h>
#include "base/Span.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! State data for a CUDA RNG
using RngState = curandState_t;

//---------------------------------------------------------------------------//
//! Initializer for an RNG
struct RngSeed
{
    using value_type = unsigned long long;
    value_type seed;
};

//---------------------------------------------------------------------------//
/*!
 * Device pointers to a vector of CUDA random number generator states.
 */
struct RngStatePointers
{
    span<RngState> rng;

    //! Whether the interface is initialized
    explicit CELER_FUNCTION operator bool() const { return !rng.empty(); }

    //! State size
    CELER_FUNCTION size_type size() const { return rng.size(); }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
